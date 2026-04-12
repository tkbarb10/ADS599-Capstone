import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def remap_stay_ids(df: pd.DataFrame, stay_id_remap: dict) -> pd.DataFrame:
    """
    Remap ed_stay_id_2 values in vitals back to their canonical ed_stay_id.
    Patients with two consecutive ED stays under one hadm_id have their second
    stay's vitals merged onto the first stay via the remap dict.
    """
    df = df.copy()
    remapped = df['ed_stay_id'].map(stay_id_remap)
    n = remapped.notna().sum()
    df['ed_stay_id'] = remapped.fillna(df['ed_stay_id']).astype(df['ed_stay_id'].dtype)
    logger.info(f"Remapped {n:,} vitals rows to canonical ed_stay_id")
    return df


def filter_to_cohort(df: pd.DataFrame, stay_ids: list) -> pd.DataFrame:
    """Drop rows for ed_stay_ids not in the cohort."""
    before = len(df)
    df = df[df['ed_stay_id'].isin(stay_ids)].reset_index(drop=True)
    logger.info(f"Filtered to cohort: dropped {before - len(df):,} rows, remaining {len(df):,}")
    return df


def fill_triage_charttimes(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Triage rows have no charttime — fill with ed_intime from cohort.
    Merges ed_intime, fillna on charttime for triage rows, then drops the helper col.
    """
    df = df.merge(cohort[['ed_stay_id', 'ed_intime']], on='ed_stay_id', how='left')
    df['charttime'] = df['charttime'].fillna(df['ed_intime'])
    df = df.drop(columns=['ed_intime'])
    logger.info("Filled triage charttimes with ed_intime")
    return df


def drop_pre_admission_rows(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Drop vital rows with charttime < ed_intime (recorded before the ED visit started).

    Uses >= (not >) so that triage rows are kept — their charttime was filled with
    ed_intime by fill_triage_charttimes, so they sit exactly at the boundary.
    Vitals rows that also land at charttime == ed_intime are handled separately by
    drop_same_time_vitals (which filters on source == 'vitals').
    """
    df = df.merge(cohort[['ed_stay_id', 'ed_intime']], on='ed_stay_id', how='left')
    before = len(df)
    df = df[df['charttime'] >= df['ed_intime']].reset_index(drop=True)
    df = df.drop(columns=['ed_intime'])
    logger.info(f"Dropped {before - len(df):,} rows with charttime < ed_intime")
    return df


def drop_same_time_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop vitals rows where charttime == ed_intime (time_since_last_min == 0 and source == 'vitals').
    These are duplicate baseline readings already represented by the triage row.
    Expected ~73K rows dropped.
    """
    drop_idx = df[(df['time_since_last_min'] == 0) & (df['source'] == 'vitals')].index
    df = df.drop(index=drop_idx).reset_index(drop=True)
    logger.info(f"Dropped {len(drop_idx):,} same-time vitals rows")
    return df


def clean_vital_sign_outliers(df):
    d = df.copy()

    # ── Temperature ──────────────────────────────────────────────────────────

    # Values >900: assumed extra digit entered (e.g. 986 → 98.6)
    d['temperature'] = d['temperature'].apply(lambda x: x / 10 if pd.notna(x) and x > 900 else x)

    # Values 28–40: assumed recorded in Celsius, convert to Fahrenheit
    d['temperature'] = d['temperature'].apply(
        lambda x: round((x * 1.8) + 32, 1) if pd.notna(x) and 28 < x <= 40 else x
        )
    
    # Values 5–10: assumed missing leading digit (e.g. 9.8 → 98)
    d['temperature'] = d['temperature'].apply(lambda x: x * 10 if pd.notna(x) and 5 < x < 10 else x)

    # Assuming that temps outside of this range are mistakes in imputation or imputed as celsius so converting to null to drop
    d['temperature'] = d['temperature'].apply(lambda x: np.nan if pd.notna(x) and x > 115 or x < 70 else x)

    # ── Heart rate ───────────────────────────────────────────────────────────
    # Values >500: assumed extra digits entered (e.g. 800 → 80)
    d['heartrate'] = d['heartrate'].apply(lambda x: x / 10 if pd.notna(x) and x > 500 else x)

    # Values <20: no recoverable pattern → null for imputation
    d['heartrate'] = d['heartrate'].apply(lambda x: np.nan if pd.notna(x) and x < 20 else x)

    # ── Respiratory rate ─────────────────────────────────────────────────────
    # Values >1000: assumed two extra digits (e.g. 1800 → 18)
    d['resprate'] = d['resprate'].apply(lambda x: round(x / 100) if pd.notna(x) and x > 1000 else x)

    # Values over 100: assuming some mistake in imputation so converting to null
    d['resprate'] = d['resprate'].apply(lambda x: np.nan if pd.notna(x) and x > 100 else x)

    # Values <4: no recoverable pattern → null for imputation
    d['resprate'] = d['resprate'].apply(lambda x: np.nan if pd.notna(x) and x < 4 else x)

    # ── O2 saturation ────────────────────────────────────────────────────────

    # Values ==0, >100, or <40: not recoverable → null for removal
    d['o2sat'] = d['o2sat'].apply(lambda x: np.nan if pd.notna(x) and (x == 0 or x > 100 or x < 75) else x)

    # ── Systolic BP ──────────────────────────────────────────────────────────
    # Spot-checking: values ≤270 appear accurate; above that no clear correction pattern → null
    # Values <40: too low to be plausible → null
    d['sbp'] = d['sbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 270 or x < 40) else x)

    # ── Diastolic BP ─────────────────────────────────────────────────────────
    # Values >150: likely charting errors with no clear pattern → null
    # Values <20: too low to be plausible → null
    d['dbp'] = d['dbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 150 or x < 20) else x)

    return d


def clean_pain_column(df):
    d = df.copy()

    # Normalize to lowercase string, strip punctuation artifacts (quotes, >, -, +)
    d['pain'] = (
        d['pain'].astype('str').str.lower().str.strip()
        .str.strip('"').str.strip('\u201c').str.strip('\u201d')
        .str.strip('>').str.strip('-').str.strip('+')
    )

    # Range entries like "5-7" → take the lower (first) value
    d['pain'] = d['pain'].str.replace(r'(\d)-\d', r'\1', regex=True)

    # Coerce to numeric; anything non-numeric (text descriptions) → NaN
    d['pain'] = pd.to_numeric(d['pain'], errors='coerce')

    # Values >10: not a valid 0–10 pain scale entry → NaN
    # Could be mis-entries, carryover from another field, or a different scale (e.g. GCS)
    d['pain'] = d['pain'].apply(lambda x: np.nan if pd.notna(x) and x > 10 else x)

    return d