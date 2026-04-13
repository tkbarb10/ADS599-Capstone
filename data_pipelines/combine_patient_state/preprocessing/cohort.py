import pandas as pd
from datasets import load_dataset

ICU_LABELS = {'ED_DIRECT_ICU', 'ED_WARD_ICU', 'ED_DISCHARGE_RETURN_ICU'}

BASE_COLS = [
    'ed_stay_id', 'subject_id', 'hadm_id', 'ed_intime', 'ed_outtime',
    'stay_window_start', 'stay_window_end', 'admittime', 'dischtime',
    'arrival_transport', 'gender', 'anchor_age', 'admission_type', 'cohort_label',
]

DATETIME_COLS = [
    'stay_window_start', 'stay_window_end',
    'ed_intime', 'ed_outtime',
    'admittime', 'dischtime',
]


def load_cohort(src_repo: str, hf_cfg: dict) -> pd.DataFrame:
    """Load cohort from HuggingFace, keep BASE_COLS, parse datetimes, add terminal_event."""
    df = load_dataset(
        src_repo,
        name=hf_cfg['cohort']['config_name'],
        split=hf_cfg['cohort']['split_name'],
    ).to_pandas()

    cohort = df[BASE_COLS].copy()
    del df

    for col in DATETIME_COLS:
        cohort[col] = pd.to_datetime(cohort[col])

    # anchor_age may arrive as float64 from HF (null-promoted int).  Cast to int64
    # so Arrow maps it to a clean primitive type and the dataset card YAML stays valid.
    cohort['anchor_age'] = cohort['anchor_age'].fillna(0).astype('int64')

    cohort['terminal_event'] = cohort['cohort_label'].apply(
        lambda x: 'transfer_icu' if x in ICU_LABELS else 'discharge'
    )
    # stay_window_end is already truncated to first_icu_intime for ICU patients

    return cohort
