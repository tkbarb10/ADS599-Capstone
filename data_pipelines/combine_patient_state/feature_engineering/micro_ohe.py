import gc
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset

from data_pipelines.combine_patient_state.feature_engineering.ohe_utils import merge_pivot_into_patient

logger = logging.getLogger(__name__)

RESULT_MAP = {'NEGATIVE': '_Negative', 'POSITIVE': '_Positive'}
PRIORITY   = {'_Positive': 2, '_Negative': 1, '_Other': 0}


def expand_micro_ohe(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    stay_last_idx: pd.DataFrame,
    src_repo: str,
    hf_cfg: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load microbiology, expand pending/result windows into OHE columns on df_patient.

    Columns added: {action}_Pending, {action}_Negative, {action}_Positive, {action}_Other.
    Also adds micro_ordered indicator.

    Returns (df_patient, micro_actions).
    """
    df_micro = load_dataset(src_repo, name=hf_cfg['microbiology']['config_name'],
                            split=hf_cfg['microbiology']['split_name']).to_pandas()
    df_micro['charttime'] = pd.to_datetime(df_micro['charttime'])
    df_micro['storetime'] = pd.to_datetime(df_micro['storetime'])
    logger.info(f'microbiology loaded -- {len(df_micro):,} rows')

    micro_actions = sorted(df_micro['action_space'].unique())
    logger.info(f'micro actions: {len(micro_actions)}')

    # Initialize all OHE columns at once
    df_patient.index = df_patient['step_idx']
    ohe_init = pd.DataFrame(
        0,
        index=df_patient.index,
        columns=[f'{a}_{s}' for a in micro_actions
                 for s in ('Pending', 'Negative', 'Positive', 'Other')],
    )
    df_patient = pd.concat([df_patient, ohe_init], axis=1).copy()
    del ohe_init

    df_micro_orders  = df_micro.dropna(subset=['charttime'])
    df_micro_results = df_micro.dropna(subset=['storetime'])

    # Add result_suffix only to results (not orders) to avoid column conflict on merge
    df_micro_results = df_micro_results.copy()
    df_micro_results['result_suffix'] = (
        df_micro_results['culture_result'].map(RESULT_MAP).fillna('_Other')
    )

    # Snap order time -> order_step
    micro_order_steps = pd.merge_asof(
        df_micro_orders.sort_values('charttime'),
        dp_sorted,
        left_on='charttime', right_on='time',
        by='ed_stay_id', direction='backward',
    ).rename(columns={'step_idx': 'order_step'})

    # Snap result time -> result_step
    micro_result_steps = pd.merge_asof(
        df_micro_results.sort_values('storetime'),
        dp_sorted,
        left_on='storetime', right_on='time',
        by='ed_stay_id', direction='backward',
    ).rename(columns={'step_idx': 'result_step'})

    # Join result_step back onto order_steps
    micro_order_steps = micro_order_steps.merge(
        micro_result_steps[['ed_stay_id', 'charttime', 'action_space', 'result_step', 'result_suffix']],
        on=['ed_stay_id', 'charttime', 'action_space'], how='left',
    ).merge(stay_last_idx, on='ed_stay_id', how='left')

    micro_order_steps = micro_order_steps.dropna(subset=['order_step', 'last_idx'])
    micro_order_steps[['order_step', 'last_idx']] = (
        micro_order_steps[['order_step', 'last_idx']].astype(int)
    )
    micro_order_steps['result_step'] = (
        micro_order_steps['result_step']
        .fillna(micro_order_steps['last_idx'] + 1)
        .astype(int)
    )
    micro_order_steps = micro_order_steps.sort_values('charttime').reset_index(drop=True)

    # -- Pending windows
    m_pending = micro_order_steps[
        micro_order_steps['result_step'] > micro_order_steps['order_step']
    ].copy()
    m_pend_len = m_pending['result_step'] - m_pending['order_step']
    m_pend_off = np.arange(m_pend_len.sum())
    m_pend_off -= np.repeat(
        m_pend_len.cumsum().shift(1, fill_value=0).values, m_pend_len.values
    )
    m_pend_exp = m_pending.loc[m_pending.index.repeat(m_pend_len)].copy()
    m_pend_exp['step_idx'] = (
        np.repeat(m_pending['order_step'].values, m_pend_len.values) + m_pend_off
    )
    m_pend_exp['ohe_col'] = m_pend_exp['action_space'] + '_Pending'

    # -- Result windows
    m_result = micro_order_steps[
        micro_order_steps['last_idx'] >= micro_order_steps['result_step']
    ].copy()
    m_result['ohe_col'] = m_result['action_space'] + m_result['result_suffix']
    m_res_len = m_result['last_idx'] - m_result['result_step'] + 1
    m_res_off = np.arange(m_res_len.sum())
    m_res_off -= np.repeat(
        m_res_len.cumsum().shift(1, fill_value=0).values, m_res_len.values
    )
    m_res_exp = m_result.loc[m_result.index.repeat(m_res_len)].copy()
    m_res_exp['step_idx'] = (
        np.repeat(m_result['result_step'].values, m_res_len.values) + m_res_off
    )

    # Deduplicate at (step_idx, action_space): Positive > Negative > Other
    m_res_exp['result_priority'] = m_res_exp['result_suffix'].map(PRIORITY)
    m_res_exp = (
        m_res_exp
        .sort_values('result_priority', ascending=False)
        .drop_duplicates(subset=['step_idx', 'action_space'], keep='first')
        .drop(columns='result_priority')
    )

    m_all = pd.concat([
        m_pend_exp[['step_idx', 'ohe_col']].assign(val=1),
        m_res_exp[['step_idx', 'ohe_col']].assign(val=1),
    ])
    m_pivot = m_all.pivot_table(
        index='step_idx', columns='ohe_col', values='val', aggfunc='max', fill_value=0,
    )

    df_patient = merge_pivot_into_patient(df_patient, m_pivot)

    # micro_ordered indicator
    micro_pending_cols = [f'{a}_Pending' for a in micro_actions]
    df_patient['micro_ordered'] = (
        df_patient.groupby('ed_stay_id')[micro_pending_cols]
        .diff().fillna(0).gt(0).any(axis=1).astype(int)
    )

    del df_micro, df_micro_orders, df_micro_results
    del micro_order_steps, micro_result_steps
    del m_pending, m_pend_exp, m_result, m_res_exp, m_all, m_pivot
    gc.collect()
    logger.info('Microbiology OHE expansion complete.')
    return df_patient, micro_actions
