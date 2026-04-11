import logging
import pandas as pd

logger = logging.getLogger(__name__)

_NON_ADMIN = [
    'not given', 'refused', 'held', 'due', 'stopped', 'restarted', 'rate change'
]


def filter_non_admin_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop eMAR rows with non-administration event_txt values.
    Pyxis rows (in_er=True) have null event_txt and are retained.
    """
    mask = df['event_txt'].str.lower().str.contains('|'.join(_NON_ADMIN), na=False)
    before = len(df)
    df = df[~mask].copy()
    logger.info(f"Dropped {before - len(df):,} non-admin event rows - remaining: {len(df):,}")
    return df


def coalesce_charttime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip timezone from charttime column.
    SQL already outputs a single charttime column; this just normalizes tz.
    """
    df = df.copy()
    df['charttime'] = pd.to_datetime(df['charttime'])
    if df['charttime'].dt.tz is not None:
        df['charttime'] = df['charttime'].dt.tz_localize(None)
    return df
