import gc
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _snap_acuity(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    df_src: pd.DataFrame,
    time_col: str,
    acuity_col: str,
    out_prefix: str,
) -> pd.DataFrame:
    """
    Snap acuity readings to the nearest preceding step, forward-fill, then OHE into
    {out_prefix}_Normal, {out_prefix}_Moderate, {out_prefix}_Acute.

    Acuity values in df_src are expected to already be 1/2/3 (shifted before calling).
    Rows with no prior reading default to 0 (no imaging yet).
    """
    snapped = pd.merge_asof(
        df_src.sort_values(time_col),
        dp_sorted,
        left_on=time_col, right_on='time',
        by='ed_stay_id', direction='backward',
    )
    last = (
        snapped.dropna(subset=['step_idx'])
        .groupby(['ed_stay_id', 'step_idx'])[acuity_col]
        .max()
        .reset_index()
    )
    last['step_idx'] = last['step_idx'].astype(int)

    df_patient = df_patient.merge(last, on=['ed_stay_id', 'step_idx'], how='left')
    df_patient[acuity_col] = (
        df_patient.groupby('ed_stay_id')[acuity_col]
        .ffill()
        .fillna(0)
        .astype(int)
    )

    df_patient[f'{out_prefix}_Normal']   = (df_patient[acuity_col] == 1).astype(int)
    df_patient[f'{out_prefix}_Moderate'] = (df_patient[acuity_col] == 2).astype(int)
    df_patient[f'{out_prefix}_Acute']    = (df_patient[acuity_col] == 3).astype(int)
    df_patient.drop(columns=acuity_col, inplace=True)

    del snapped, last
    return df_patient


def snap_ecg_ohe(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    src_repo: str,
    hf_cfg: dict,
) -> pd.DataFrame:
    """
    Load ECG data, snap to step, forward-fill, OHE into ecg_status_{Normal,Moderate,Acute}.
    """
    df_ecg = load_dataset(src_repo, name=hf_cfg['ecg']['config_name'],
                          split=hf_cfg['ecg']['split_name']).to_pandas()
    df_ecg.rename(columns={'stay_id': 'ed_stay_id'}, inplace=True)
    df_ecg['ecg_time'] = pd.to_datetime(df_ecg['ecg_time']).dt.tz_localize(None)
    df_ecg['ecg_acuity'] = df_ecg['ecg_acuity'] + 1   # 0->1=Normal, 1->2=Moderate, 2->3=Acute
    logger.info(f'ECG loaded -- {len(df_ecg):,} rows')

    df_patient = _snap_acuity(df_patient, dp_sorted, df_ecg,
                              time_col='ecg_time', acuity_col='ecg_acuity',
                              out_prefix='ecg_status')
    del df_ecg; gc.collect()
    logger.info('ECG OHE complete.')
    return df_patient


def snap_rad_ohe(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    src_repo: str,
    hf_cfg: dict,
) -> pd.DataFrame:
    """
    Load radiology data, snap to step, forward-fill, OHE into rad_status_{Normal,Moderate,Acute}.
    """
    df_rad = load_dataset(src_repo, name=hf_cfg['radiology']['config_name'],
                          split=hf_cfg['radiology']['split_name']).to_pandas()
    df_rad.rename(columns={'stay_id': 'ed_stay_id'}, inplace=True)
    df_rad['exam_time'] = pd.to_datetime(df_rad['exam_time']).dt.tz_localize(None)
    df_rad['rad_acuity_level'] = df_rad['rad_acuity_level'] + 1   # 0->1, 1->2, 2->3
    logger.info(f'Radiology loaded -- {len(df_rad):,} rows')

    df_patient = _snap_acuity(df_patient, dp_sorted, df_rad,
                              time_col='exam_time', acuity_col='rad_acuity_level',
                              out_prefix='rad_status')
    del df_rad; gc.collect()
    logger.info('Radiology OHE complete.')
    return df_patient
