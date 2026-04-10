import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)


def add_ed_boarding_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stay window columns and ED boarding time.

    stay_window_start: ed_intime
    stay_window_end:   dischtime (admitted) or ed_outtime (ED-only)
    ed_boarding_time_min: minutes in ED after admission decision (null for ED-only patients).
    """
    df = df.copy()
    df['stay_window_start'] = pd.to_datetime(df['ed_intime'])
    df['stay_window_end'] = pd.to_datetime(df['dischtime'].fillna(df['ed_outtime']))

    df['ed_boarding_time_min'] = (
        (pd.to_datetime(df['ed_outtime']) - pd.to_datetime(df['admittime']))
        .dt.total_seconds() / 60
    )

    logger.info(f"Patients with boarding time (admitted):   {df['ed_boarding_time_min'].notna().sum():,}")
    logger.info(f"Patients without boarding time (ED-only): {df['ed_boarding_time_min'].isna().sum():,}")
    return df


def create_time_step_col(df: pd.DataFrame, time_block: int) -> pd.DataFrame:
    """
    Add time_steps column: number of time blocks in each patient's stay window.

    Args:
        df:         cohort DataFrame with stay_window_start and stay_window_end columns
        time_block: size of each time block in minutes (from settings['time_block'])
    """
    df = df.copy()
    total = pd.to_datetime(df['stay_window_end']) - pd.to_datetime(df['stay_window_start'])
    df['time_steps'] = total.apply(
        lambda x: math.ceil(x.total_seconds() / 60 / time_block) + 1 if pd.notna(x) else None
    )

    df = df[df['time_steps'].notna() & (df['time_steps'] > 0)].copy()
    logger.info(f"Rows after dropping invalid stay windows: {len(df):,}")
    return df
