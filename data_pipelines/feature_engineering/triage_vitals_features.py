import logging
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_COLS = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
VITAL_COLS = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
FILL_FORWARD_COLS = ['pain', 'acuity', 'chiefcomplaint']


def add_time_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes since previous charttime within each stay. 0 for the first row."""
    df = df.copy()
    df['time_since_last_min'] = (
        df.groupby('ed_stay_id')['charttime']
        .diff()
        .dt.total_seconds() / 60
    ).fillna(0)
    return df


def forward_fill_and_flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Forward-fill pain/acuity/chiefcomplaint within stays (known at triage).
    2. Create {col}_missing indicator for all numeric + fill-forward cols (before ffill on vitals).
    3. Forward-fill numeric vitals within stays.
    4. Mean-impute any remaining NaN in numeric cols.

    Missing indicators are created after ffill on fill_forward but before ffill on numeric
    so that truly unobserved vital timesteps are flagged.
    """
    df = df.copy()

    # Step 1 — forward fill known-at-triage cols
    df[FILL_FORWARD_COLS] = df.groupby('ed_stay_id')[FILL_FORWARD_COLS].ffill()

    # Step 2 — missing indicators before vital ffill
    for col in set(FILL_FORWARD_COLS + NUMERIC_COLS):
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # Step 3 — forward fill numeric vitals
    df[NUMERIC_COLS] = df.groupby('ed_stay_id')[NUMERIC_COLS].ffill()

    # Step 4 — mean imputation for remainder
    col_means = df[NUMERIC_COLS].mean()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(col_means)
    logger.info(f"Post-imputation NaN counts:\n{df[NUMERIC_COLS].isna().sum()}")

    return df


def add_vital_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-timestep derived features:
      - {col}_delta: change from previous reading (groupby diff)
      - {col}_rate_per_min: delta / time_since_last_min
      - mean_arterial_pressure: dbp + (sbp - dbp) / 3
      - {col}_rolling1h: 1-hour rolling mean (time-indexed, per stay)

    Triage rows get 0 for all delta/rate columns (no prior reading).
    Vital cols renamed to current_{col} at the end.
    """
    df = df.copy()

    # Deltas for vitals + pain
    deltas = df.groupby('ed_stay_id')[VITAL_COLS + ['pain']].diff()
    deltas.columns = [f'{col}_delta' for col in VITAL_COLS + ['pain']]
    df = pd.concat([df.reset_index(drop=True), deltas.reset_index(drop=True)], axis=1)

    # Rate per minute
    for col in VITAL_COLS:
        df[f'{col}_rate_per_min'] = df[f'{col}_delta'] / df['time_since_last_min']

    # MAP
    df['mean_arterial_pressure'] = df['dbp'] + (df['sbp'] - df['dbp']) / 3

    # Rolling 1-hour averages (set charttime as index for time-based window)
    df = df.set_index('charttime')
    rolling = (
        df.groupby('ed_stay_id')[VITAL_COLS]
        .rolling('1h')
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[[f'{col}_rolling1h' for col in VITAL_COLS]] = rolling
    df = df.reset_index()

    # Fill triage delta/rate with 0 (these are genuinely 0, not missing)
    triage_idx = df[df['source'] == 'triage'].index
    fill_cols = [c for c in df.columns if c.endswith('_delta') or c.endswith('_rate_per_min')]
    df.loc[triage_idx, fill_cols] = df.loc[triage_idx, fill_cols].fillna(0)

    # Rename vitals to current_*
    rename_map = {col: f'current_{col}' for col in VITAL_COLS + ['pain', 'mean_arterial_pressure']}
    df = df.rename(columns=rename_map)

    logger.info(f"Feature matrix shape: {df.shape}")
    return df
