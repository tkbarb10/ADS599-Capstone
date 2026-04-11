import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)


def filter_storetime(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Drop microbiology rows where storetime (result returned) is after the patient's
    stay window end. storetime can legitimately be null for pending cultures — those
    are retained.

    Stay window end priority: first_icu_intime > dischtime > ed_outtime.

    Why this matters:
      charttime (culture ordered) is already bounded by the SQL query window, but
      storetime (result available) can fall well after the patient left. Including
      post-stay results would give the model information that didn't exist during
      the stay. ~53% of raw rows are dropped by this filter.
    """
    timing = (
        cohort[['ed_stay_id', 'ed_outtime', 'dischtime', 'first_icu_intime']]
        .copy()
        .assign(
            ed_outtime=lambda x: pd.to_datetime(x['ed_outtime']),
            dischtime=lambda x: pd.to_datetime(x['dischtime']),
            first_icu_intime=lambda x: pd.to_datetime(x['first_icu_intime']),
        )
    )
    timing['stay_window_end'] = (
        timing['first_icu_intime']
        .fillna(timing['dischtime'])
        .fillna(timing['ed_outtime'])
    )

    df = df.copy()
    df['storetime'] = pd.to_datetime(df['storetime'])
    before = len(df)
    df = df.merge(timing[['ed_stay_id', 'stay_window_end']], on='ed_stay_id', how='left')
    df = df[df['storetime'].isna() | (df['storetime'] <= df['stay_window_end'])]
    df = df.drop(columns=['stay_window_end'])
    logger.info(f"Dropped {before - len(df):,} rows where storetime > stay_window_end "
                f"({(before - len(df)) / before:.1%}) - remaining: {len(df):,}")
    return df.reset_index(drop=True)


def classify_culture_result(org_name, comments) -> str:
    """
    Classify a single microbiology result as POSITIVE, NEGATIVE, CANCELLED, or OTHER.

    Priority: org_name checked first (most reliable signal), then comments text.
    A non-null org_name that isn't explicitly negative/cancelled is treated as POSITIVE
    — the presence of an organism name implies growth was observed.
    """
    # Step 1: check org_name
    if pd.notna(org_name):
        val = str(org_name).upper().strip()
        if re.search(r'\bNEGATIVE\b|\bNOT\b', val):
            return 'NEGATIVE'
        if re.search(r'\bPOSITIVE\b', val):
            return 'POSITIVE'
        if re.search(r'\bCANCELLED\b|\bCANCELED\b', val):
            return 'CANCELLED'
        return 'POSITIVE'

    # Step 2: check comments
    if pd.notna(comments):
        val = str(comments).lower().strip()

        if re.search(r'\bcancell?ed\b', val):
            return 'CANCELLED'
        if re.search(r'\btest not performed\b', val):
            return 'CANCELLED'
        if re.search(r'\bpatient credited\b', val):
            return 'CANCELLED'

        if re.search(r'\bindeterminate\b', val):
            return 'NEGATIVE'

        if val == 'no growth.':
            return 'NEGATIVE'
        if re.search(r'\bno\b.+\b(seen|found|isolated)\b', val):
            return 'NEGATIVE'
        if re.search(r'\bno\b.+growth', val):
            return 'NEGATIVE'
        if re.search(r'\bnot detected\b', val):
            return 'NEGATIVE'
        if re.search(r'\bnegative\b|\bnonreactive\b', val):
            return 'NEGATIVE'

        if re.search(r'^\s*[\d<>]', val):
            return 'POSITIVE'
        if re.search(r'<\s*\d[\d,]*\s*(cfu|organisms)', val):
            return 'POSITIVE'
        if re.search(r'growth', val):
            return 'POSITIVE'
        if re.search(r'\bconsistent with\b|\bcontamination\b|\bpositive\b', val):
            return 'POSITIVE'
        if re.search(r'(?<!\bno\s)(?<!\bnon)\breactive\b', val):
            return 'POSITIVE'

        # Placeholder sentinel — only underscores, dashes, and spaces
        if re.fullmatch(r'[\s_-]+', val):
            return 'OTHER'

        # Anything left with actual content is likely a positive clinical note
        return 'POSITIVE'

    return 'OTHER'