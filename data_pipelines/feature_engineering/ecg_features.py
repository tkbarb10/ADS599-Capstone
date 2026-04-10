import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

_REPORT_COLS = [f"report_{i}" for i in range(18)]

_ABNORMAL = re.compile(r"""
    ACUTE\s+INFARCT|
    INFARCT(?:ION)?|
    ISCH[EAE]MIA|
    MYOCARDIAL\s+INJURY|
    STRONGLY\s+SUGGESTS|
    CONSIDER\s+ACUTE|
    POSSIBLY\s+ACUTE|
    PROBABLY\s+ACUTE|
    COMPLETE\s+AV\s+BLOCK|
    LEFT\s+BUNDLE\s+BRANCH\s+BLOCK|
    RIGHT\s+BUNDLE\s+BRANCH\s+BLOCK|
    VENTRICULAR\s+TACHYCARDIA|
    ATRIAL\s+FIBRILLATION|
    ATRIAL\s+FLUTTER|
    VENTRICULAR\s+HYPERTROPHY|
    PROLONGED\s+QT|
    LONG\s+QTc|
    PERICARDITIS|
    ST\s+ELEVATION.*INFARCT|
    WIDE[\s\-]QRS\s+TACHYCARDIA|
    COMPLETE\s+HEART\s+BLOCK|
    EXTREME\s+TACHYCARDIA|
    VENTRICULAR\s+PREEXCITATION|
    WPW\s+PATTERN
""", re.IGNORECASE | re.VERBOSE)

_NEUTRAL = re.compile(r"""
    BORDERLINE|
    POSSIBLE(?!\s+ACUTE)|
    PROBABLE(?!\s+ACUTE)|
    NONSPECIFIC|
    NON[\s\-]SPECIFIC|
    CANNOT\s+RULE\s+OUT|
    MAY\s+BE\s+DUE|
    EQUIVOCAL|
    ABNORMAL\s+FOR\s+AGE|
    CONSIDER(?!\s+ACUTE)|
    AGE\s+UNDETERMINED|
    AGE\s+INDETERMINATE|
    ST[\s\-]T\s+CHANGES|
    T\s+WAVE\s+CHANGES|
    REPOLARIZATION\s+ABNORMALITY|
    CONDUCTION\s+DEFECT|
    AXIS\s+DEVIATION|
    ATRIAL\s+ABNORMALITY|
    ATRIAL\s+ENLARGEMENT|
    LOW\s+QRS\s+VOLTAGE|
    POOR\s+R\s+WAVE\s+PROGRESSION|
    ALL\s+12\s+LEADS\s+ARE\s+MISSING|
    NOT\s+ENOUGH\s+LEADS|
    DATA\s+QUALITY|
    UNSUITABLE\s+FOR\s+ANALYSIS|
    RECORDING\s+UNSUITABLE|
    TECHNICAL\s+ERROR|
    ANALYSIS\s+ERROR|
    PEDIATRIC|
    MEASUREMENT\s+ERROR|
    NO\s+FURTHER\s+ANALYSIS
""", re.IGNORECASE | re.VERBOSE)

_NORMAL = re.compile(r"""
    ^\s*NORMAL\s+ECG\s*$|
    WITHIN\s+NORMAL\s+LIMITS|
    NORMAL\s+VARIANT|
    AVAILABLE\s+LEADS\s+NORMAL|
    NORMAL\s+ECG\s+BASED\s+ON|
    NORMAL\s+ECG\s+EXCEPT\s+FOR\s+RATE|
    ^SINUS\s+RHYTHM\s*\.?\s*$
""", re.IGNORECASE | re.VERBOSE)


def _classify_cell(value) -> int:
    """Score a single report cell. Returns 2=abnormal, 1=neutral, 0=normal, -1=empty."""
    if value is None or str(value).strip() in ('nan', '', 'None'):
        return -1
    v = str(value).strip()
    if _ABNORMAL.search(v):
        return 2
    if _NEUTRAL.search(v):
        return 1
    if _NORMAL.search(v):
        return 0
    return 1  # unrecognized → neutral, not normal


def score_ecg_acuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each of the 18 report columns and take the row-level max.
    Returns df with report columns dropped and ecg_acuity added:
      0 = normal, 1 = neutral/unknown, 2 = abnormal.
    Rows where all 18 columns are empty (no ECG data) are assigned 1.
    """
    df = df.copy()
    score_cols = []
    for col in _REPORT_COLS:
        s = f"{col}_score"
        df[s] = df[col].apply(_classify_cell)
        score_cols.append(s)

    df['ecg_acuity'] = df[score_cols].max(axis=1).replace(-1, 1).astype(int)
    df = df.drop(columns=_REPORT_COLS + score_cols)

    logger.info(f"ecg_acuity distribution:\n{df['ecg_acuity'].value_counts().sort_index()}")
    return df


def resolve_duplicate_ecgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one ECG per ED stay: highest acuity wins; ties broken by earliest ecg_time.
    """
    before = len(df)
    df = (
        df.sort_values(['ed_stay_id', 'ecg_acuity', 'ecg_time'],
                       ascending=[True, False, True])
        .drop_duplicates(subset='ed_stay_id', keep='first')
        .reset_index(drop=True)
    )
    logger.info(f"Resolved duplicates: {before:,} rows → {len(df):,} (one per ed_stay_id)")
    return df