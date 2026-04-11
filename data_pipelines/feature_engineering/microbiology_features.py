import logging
import pandas as pd

from data_pipelines.preprocessing_scripts.microbiology_preprocessing import classify_culture_result

logger = logging.getLogger(__name__)

# Top 20 specimen types from the full cohort (by volume, after storetime filtering).
# Remaining spec_type_desc values → 'OTHER' to keep the action space tractable.
# Top 20 covers ~96.8% of records.
TOP_SPEC_TYPES = [
    'URINE', 'BLOOD CULTURE', 'STOOL', 'SWAB', 'TISSUE', 'SPUTUM',
    'ABSCESS', 'PERITONEAL FLUID', 'CSF;SPINAL FLUID', 'PLEURAL FLUID',
    'SEROLOGY/BLOOD', 'Rapid Respiratory Viral Screen & Culture',
    'JOINT FLUID', 'FLUID,OTHER', 'Blood (EBV)', 'BRONCHOALVEOLAR LAVAGE',
    'MRSA SCREEN', 'BILE', 'Blood (CMV AB)', 'IMMUNOLOGY',
]

# Worst-case result priority for collapsing multiple results per group.
# Lower index = higher priority (POSITIVE is most clinically conservative).
_RESULT_PRIORITY = {'POSITIVE': 0, 'NEGATIVE': 1, 'CANCELLED': 2, 'OTHER': 3}


def add_action_space(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map spec_type_desc to action_space: top 20 specimen types retained by name,
    all others collapsed to 'OTHER'. Top 20 covers ~96.8% of records.
    """
    df = df.copy()
    df['action_space'] = df['spec_type_desc'].where(
        df['spec_type_desc'].isin(TOP_SPEC_TYPES), other='OTHER'
    )
    coverage = (df['action_space'] != 'OTHER').mean()
    logger.info(f"action_space top-20 coverage: {coverage:.1%}")
    logger.debug(f"action_space value counts:\n{df['action_space'].value_counts()}")
    return df


def label_culture_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply classify_culture_result to each row, adding a culture_result column.
    Classification uses org_name (primary) then comments (fallback).
    Returns: POSITIVE, NEGATIVE, CANCELLED, or OTHER.
    """
    df = df.copy()
    df['culture_result'] = df.apply(
        lambda r: classify_culture_result(r['org_name'], r['comments']), axis=1
    )
    logger.info(f"culture_result distribution:\n{df['culture_result'].value_counts()}")
    return df


def collapse_micro_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per (ed_stay_id, charttime, action_space).

    Multiple rows can exist for the same culture order due to multiple organisms
    or antibiotic sensitivity test entries. Collapse rules:
      - culture_result: worst-case (POSITIVE > NEGATIVE > CANCELLED > OTHER)
      - storetime: latest (when all results in the batch are back)
      - subject_id, hadm_id: same within group, take first

    Output schema:
      subject_id | ed_stay_id | hadm_id | charttime | storetime | action_space | culture_result
    """
    def _worst_case(series):
        mapped = series.map(_RESULT_PRIORITY)
        return series.iloc[mapped.argmin()]

    before = len(df)
    collapsed = (
        df.groupby(['ed_stay_id', 'charttime', 'action_space'], observed=True)
        .agg(
            subject_id=('subject_id', 'first'),
            hadm_id=('hadm_id', 'first'),
            storetime=('storetime', 'max'),
            culture_result=('culture_result', _worst_case),
        )
        .reset_index()
        [['subject_id', 'ed_stay_id', 'hadm_id', 'charttime', 'storetime',
          'action_space', 'culture_result']]
    )
    logger.info(f"Collapsed {before:,} rows → {len(collapsed):,} "
                f"(one per ed_stay_id / charttime / action_space)")
    logger.debug(f"culture_result distribution after collapse:\n{collapsed['culture_result'].value_counts()}")
    return collapsed