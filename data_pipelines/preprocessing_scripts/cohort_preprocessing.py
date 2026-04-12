import logging
import pickle
import re
import pandas as pd

from utils.load_yaml_helper import get_artifacts_dir

logger = logging.getLogger(__name__)


def fix_label_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix cohort label mismatches caused by MIMIC data inconsistencies between
    edstays.disposition and the actual admissions/transfers records.

    Case 1: Labeled ED_DISCHARGE_* but has hadm_id + admittime → patient was actually admitted.
      Sub-case: has first_icu_intime → ED_DIRECT_ICU or ED_WARD_ICU (based on time delta).
      Sub-case: no ICU → ED_WARD_DISCHARGE.

    Case 2: Labeled ED_WARD_DISCHARGE but no hadm_id / admittime → patient was not admitted.
      → reclassify as ED_DISCHARGE_STABLE.
    """
    discharge_labels = ['ED_DISCHARGE_STABLE', 'ED_DISCHARGE_RETURN_ICU', 'ED_DISCHARGE_DIED_72H']

    # Case 1: discharge label but has an actual admission record
    admitted_mask = (
        df['cohort_label'].isin(discharge_labels) &
        df['hadm_id'].notna() &
        df['admittime'].notna()
    )
    logger.info(f"Case 1 (discharge label but admitted): {admitted_mask.sum()} rows")

    def reclassify_admitted(row):
        if pd.notna(row['first_icu_intime']):
            delta = (pd.to_datetime(row['first_icu_intime']) - pd.to_datetime(row['admittime'])).total_seconds()
            return 'ED_DIRECT_ICU' if delta <= 600 else 'ED_WARD_ICU'
        return 'ED_WARD_DISCHARGE'

    df.loc[admitted_mask, 'cohort_label'] = df[admitted_mask].apply(reclassify_admitted, axis=1)

    # Case 2: ward label but no admission record
    not_admitted_mask = (
        (df['cohort_label'] == 'ED_WARD_DISCHARGE') &
        df['hadm_id'].isna() &
        df['admittime'].isna()
    )
    logger.info(f"Case 2 (ward label but no admission): {not_admitted_mask.sum()} rows")
    df.loc[not_admitted_mask, 'cohort_label'] = 'ED_DISCHARGE_STABLE'

    logger.info(f"Updated cohort label distribution:\n{df['cohort_label'].value_counts()}")
    return df


def merge_duplicate_hadm_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive ED visits that share a hadm_id into a single row.
    Some patients have two back-to-back ED stays under one admission (e.g. brief
    discharge and immediate return). We keep the first stay's identifiers and
    merge the second stay's ed_outtime/disposition/cohort_label into it,
    storing the second ed_stay_id in ed_stay_id_2.
    """
    input_len = len(df)

    dup_counts = df['hadm_id'].value_counts()
    dup_hadm_ids = dup_counts[dup_counts == 2].index

    singles = df[~df['hadm_id'].isin(dup_hadm_ids)].copy()
    dups = df[df['hadm_id'].isin(dup_hadm_ids)].sort_values(['hadm_id', 'ed_intime'])

    first_rows = dups.groupby('hadm_id').nth(0).reset_index(drop=True)
    second_rows = dups.groupby('hadm_id').nth(1).reset_index(drop=True)

    merged_dups = first_rows.copy()
    merged_dups['ed_outtime'] = second_rows['ed_outtime'].values
    merged_dups['disposition'] = second_rows['disposition'].values
    merged_dups['cohort_label'] = second_rows['cohort_label'].values
    merged_dups['ed_stay_id_2'] = second_rows['ed_stay_id'].values
    merged_dups.drop(columns=['base_pathway'], inplace=True)

    singles['ed_stay_id_2'] = float('nan')
    singles.drop(columns=['base_pathway'], inplace=True)

    result = pd.concat([singles, merged_dups], ignore_index=True)

    logger.info(f"Input rows: {input_len:,}")
    logger.info(f"Output rows: {len(result):,}  (expected {input_len - len(dup_hadm_ids):,})")
    logger.info(f"Max hadm_id count: {result['hadm_id'].value_counts().max()}  (expected 1)")

    return result


def save_stay_id_remap(df: pd.DataFrame) -> None:
    """
    Build and save a stay_id_remap dict: {ed_stay_id_2 -> ed_stay_id}.
    Used by downstream pipelines (e.g. vitals) to remap vitals from a
    patient's second consecutive ED stay onto the merged first stay.
    Saved to artifacts/stay_id_remap.pkl.
    """
    stay_id_remap = (
        df[df['ed_stay_id_2'].notna()]
        .set_index('ed_stay_id_2')['ed_stay_id']
        .to_dict()
    )
    artifacts_dir = get_artifacts_dir()
    out_path = artifacts_dir / "stay_id_remap.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(stay_id_remap, f)
    logger.info(f"Saved stay_id_remap with {len(stay_id_remap):,} entries to {out_path}")


def simplify_race_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse MIMIC's free-text race values into 5 broad categories + Other.
    """
    _RACE_MAP = [
        ('White',           r'white|portuguese|brazilian'),
        ('Black',           r'black|african'),
        ('Hispanic',        r'hispanic|latino|south american'),
        ('Asian',           r'asian|pacific islander|hawaiian'),
        ('Native American', r'american indian|alaska native'),
    ]

    def _collapse(val):
        if pd.isna(val):
            return 'Other'
        v = str(val).lower()
        for label, pattern in _RACE_MAP:
            if re.search(pattern, v):
                return label
        return 'Other'

    df = df.copy()
    df['race'] = df['race'].apply(_collapse)
    logger.debug(f"Race distribution after simplification:\n{df['race'].value_counts()}")
    return df


def fix_mismatched_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix rows where stay_window_start > stay_window_end.
    Rows with no hadm_id (ED-only, no valid window anchor) are dropped.
    Remaining rows get stay_window_end reset to ed_outtime.
    """
    disjointed_rows = df[df['stay_window_start'] > df['stay_window_end']].index

    rows_to_drop = df.loc[disjointed_rows][df.loc[disjointed_rows, 'hadm_id'].isna()].index
    df = df.drop(index=rows_to_drop)
    logger.info(f"Dropped {len(rows_to_drop):,} rows with no hadm_id and disjointed window")

    disjointed_rows = df[df['stay_window_start'] > df['stay_window_end']].index
    df.loc[disjointed_rows, 'stay_window_end'] = df.loc[disjointed_rows, 'ed_outtime']
    logger.info(f"Reset stay_window_end to ed_outtime for {len(disjointed_rows):,} admitted rows")

    return df
