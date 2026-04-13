import gc
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def snap_vitals(df_patient: pd.DataFrame, src_repo: str, hf_cfg: dict) -> pd.DataFrame:
    """
    Load triage/vitals, snap each reading to the nearest preceding step (merge_asof backward),
    keep the last reading per (ed_stay_id, step_idx), merge onto df_patient, set vitals_checked,
    then forward-fill vital columns within each stay.

    source column is kept but NOT forward-filled (marks triage row = start of stay).
    acuity is merged but NOT included in vital_cols ffill list.
    """
    tv_df = load_dataset(src_repo, name=hf_cfg['triage_vitals']['config_name'],
                         split=hf_cfg['triage_vitals']['split_name']).to_pandas()
    tv_df['charttime'] = pd.to_datetime(tv_df['charttime'])
    logger.info(f'triage_vitals loaded -- {len(tv_df):,} rows, '
                f'{tv_df["ed_stay_id"].nunique():,} stays')

    vital_cols = [c for c in tv_df.columns
                  if c not in ('ed_stay_id', 'subject_id', 'charttime', 'source', 'acuity')]

    dp_sorted = df_patient[['ed_stay_id', 'step_idx', 'time']].sort_values('time')

    vital_snapped = pd.merge_asof(
        tv_df.sort_values('charttime'),
        dp_sorted,
        left_on='charttime', right_on='time',
        by='ed_stay_id', direction='backward',
    )
    vital_last = (
        vital_snapped
        .sort_values(['ed_stay_id', 'step_idx', 'charttime'])
        .groupby(['ed_stay_id', 'step_idx'])
        .last()
        .reset_index()
    )

    merge_cols = ['ed_stay_id', 'step_idx', 'source', 'acuity'] + vital_cols
    df_patient = df_patient.merge(vital_last[merge_cols], on=['ed_stay_id', 'step_idx'], how='left')

    # vitals_checked = 1 at steps with an actual measurement (before ffill)
    df_patient['vitals_checked'] = df_patient[vital_cols].notna().any(axis=1).astype(int)

    # Forward-fill vitals within each stay; source does NOT get ffilled
    df_patient[vital_cols] = df_patient.groupby('ed_stay_id')[vital_cols].ffill()

    del tv_df, vital_snapped, vital_last; gc.collect()
    logger.info(f'vitals_checked=1 rows: {df_patient["vitals_checked"].sum():,}')
    return df_patient


def compute_time_since_last_vitals(df_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Add time_since_last_min: minutes elapsed since the most recent vitals_checked=1 row
    within each stay. Resets to 0 at every vitals_checked row. Rows before the first
    vital check default to 0.
    """
    df_patient['_last_vital_time'] = df_patient['time'].where(df_patient['vitals_checked'] == 1)
    df_patient['_last_vital_time'] = df_patient.groupby('ed_stay_id')['_last_vital_time'].ffill()

    df_patient['time_since_last_min'] = (
        (df_patient['time'] - df_patient['_last_vital_time'])
        .dt.total_seconds()
        .div(60)
        .fillna(0)
        .round()
        .astype(int)
    )
    df_patient.drop(columns='_last_vital_time', inplace=True)

    logger.info(f'time_since_last_min -- max: {df_patient["time_since_last_min"].max():,}, '
                f'mean: {df_patient["time_since_last_min"].mean():.1f}')
    return df_patient
