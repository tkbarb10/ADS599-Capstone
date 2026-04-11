import logging
import pandas as pd

logger = logging.getLogger(__name__)

_ADMIN_PATTERNS = [
    "BY SAME PHYSICIAN", "BY DIFFERENT PHYSICIAN",
    "DISTINCT PROCEDURAL SERVICE", "SEPARATE STRUCTURE",
    "MOD SEDATION", "FEE ADJUSTED", "___", "CAD ",
]


def filter_admin_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove administrative/billing entries from exam_name.
    These are billing modifiers attached to studies, not imaging exams themselves.
    exam_name is uppercased and stripped before matching.
    """
    df = df.copy()
    df['exam_name'] = df['exam_name'].str.upper().str.strip()
    mask = df['exam_name'].str.contains('|'.join(_ADMIN_PATTERNS), na=False)
    before = len(df)
    df = df[~mask].copy()
    logger.info(f"Removed {before - len(df):,} admin rows — remaining: {len(df):,}")
    return df


def resolve_duplicate_exams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one exam per ED stay: highest acuity wins; ties broken by earliest charttime.
    Must be called after assign_rad_acuity so exam_acuity is present.
    """
    before = len(df)
    df = (
        df.sort_values(['ed_stay_id', 'exam_acuity', 'charttime'],
                       ascending=[True, False, True])
        .drop_duplicates(subset='ed_stay_id', keep='first')
        .reset_index(drop=True)
    )
    logger.info(f"Resolved duplicates: {before:,} rows → {len(df):,} (one per ed_stay_id)")
    return df