import gc
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset

from data_pipelines.combine_patient_state.feature_engineering.ohe_utils import merge_pivot_into_patient

logger = logging.getLogger(__name__)


def expand_med_flags(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    stay_last_idx: pd.DataFrame,
    src_repo: str,
    hf_cfg: dict,
) -> pd.DataFrame:
    """
    Load dispensed meds (Administered only), snap to step, keep first admin per
    (ed_stay_id, drug_class), expand flag from first admin step to last_idx.

    Column value = 1 from first administration onward; 0 before.
    """
    df_meds = load_dataset(src_repo, name=hf_cfg['dispensed_meds']['config_name'],
                           split=hf_cfg['dispensed_meds']['split_name']).to_pandas()
    df_meds = df_meds[df_meds['event_txt'] == 'Administered'].copy()
    df_meds['charttime'] = pd.to_datetime(df_meds['charttime'])
    logger.info(f'meds (Administered) loaded -- {len(df_meds):,} rows')

    med_classes = sorted(df_meds['drug_class'].unique())
    logger.info(f'drug classes: {len(med_classes)}')

    # Initialize only columns that don't already exist (guards against re-running on a
    # checkpoint that already contains med columns, which would create duplicates)
    new_cols = [m for m in med_classes if m not in df_patient.columns]
    if new_cols:
        df_patient.index = df_patient['step_idx']
        med_init = pd.DataFrame(0, index=df_patient.index, columns=new_cols)
        df_patient = pd.concat([df_patient, med_init], axis=1).copy()
        del med_init
    else:
        logger.info('Med columns already present -- skipping initialization')

    # Snap charttime -> step_idx
    med_snapped = pd.merge_asof(
        df_meds.sort_values('charttime'),
        dp_sorted,
        left_on='charttime', right_on='time',
        by='ed_stay_id', direction='backward',
    )
    med_snapped = med_snapped.dropna(subset=['step_idx'])
    med_snapped['step_idx'] = med_snapped['step_idx'].astype(int)

    # Keep first administration per (ed_stay_id, drug_class)
    first_admin = (
        med_snapped.sort_values('charttime')
        .groupby(['ed_stay_id', 'drug_class'], as_index=False)
        .first()
        .merge(stay_last_idx, on='ed_stay_id', how='left')
        .dropna(subset=['step_idx', 'last_idx'])
    )
    first_admin[['step_idx', 'last_idx']] = first_admin[['step_idx', 'last_idx']].astype(int)

    # Expand from first admin step to last_idx
    admin_len = (first_admin['last_idx'] - first_admin['step_idx'] + 1).clip(lower=0)
    admin_off = np.arange(admin_len.sum())
    admin_off -= np.repeat(admin_len.cumsum().shift(1, fill_value=0).values, admin_len.values)
    admin_exp = first_admin.loc[first_admin.index.repeat(admin_len)].copy()
    admin_exp['row_idx'] = (
        np.repeat(first_admin['step_idx'].values, admin_len.values) + admin_off
    )

    med_pivot = admin_exp.pivot_table(
        index='row_idx', columns='drug_class', values='last_idx',
        aggfunc='count', fill_value=0,
    ).clip(upper=1)

    # rename row_idx -> step_idx for merge_pivot_into_patient
    med_pivot.index.name = 'step_idx'
    df_patient = merge_pivot_into_patient(df_patient, med_pivot)

    del df_meds, med_snapped, first_admin, admin_exp, med_pivot
    gc.collect()
    logger.info('Medications expansion complete.')
    return df_patient
