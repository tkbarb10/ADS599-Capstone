import gc
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset

from data_pipelines.combine_patient_state.feature_engineering.ohe_utils import merge_pivot_into_patient

logger = logging.getLogger(__name__)


def expand_labs_ohe(
    df_patient: pd.DataFrame,
    dp_sorted: pd.DataFrame,
    stay_last_idx: pd.DataFrame,
    src_repo: str,
    hf_cfg: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load labs, expand pending/result windows into OHE columns on df_patient.

    Columns added: {action}_Pending, {action}_Normal, {action}_Abnormal for every unique action.
    Also adds labs_ordered indicator (1 when a new _Pending transitions 0->1).

    Returns (df_patient, lab_actions).
    """
    df_labs = load_dataset(src_repo, name=hf_cfg['labs']['config_name'],
                           split=hf_cfg['labs']['split_name']).to_pandas()
    df_labs['order_time']  = pd.to_datetime(df_labs['order_time'])
    df_labs['result_time'] = pd.to_datetime(df_labs['result_time'])
    logger.info(f'labs loaded -- {len(df_labs):,} rows')

    lab_actions = sorted(df_labs['action'].unique())
    logger.info(f'lab actions: {len(lab_actions)}')

    # Initialize all OHE columns at once to avoid DataFrame fragmentation
    df_patient.index = df_patient['step_idx']
    ohe_init = pd.DataFrame(
        0,
        index=df_patient.index,
        columns=[f'{a}_{s}' for a in lab_actions for s in ('Pending', 'Normal', 'Abnormal')],
    )
    df_patient = pd.concat([df_patient, ohe_init], axis=1).copy()
    del ohe_init

    df_labs_orders  = df_labs.dropna(subset=['order_time'])
    df_labs_results = df_labs.dropna(subset=['result_time'])

    # Snap order_time -> order_step
    order_steps = pd.merge_asof(
        df_labs_orders.sort_values('order_time'),
        dp_sorted,
        left_on='order_time', right_on='time',
        by='ed_stay_id', direction='backward',
    ).rename(columns={'step_idx': 'order_step'})

    # Snap result_time -> result_step
    result_steps = pd.merge_asof(
        df_labs_results.sort_values('result_time'),
        dp_sorted,
        left_on='result_time', right_on='time',
        by='ed_stay_id', direction='backward',
    ).rename(columns={'step_idx': 'result_step'})

    # Join result_step back onto order_steps
    order_steps = order_steps.merge(
        result_steps[['ed_stay_id', 'order_time', 'action', 'result_step']],
        on=['ed_stay_id', 'order_time', 'action'], how='left',
    ).merge(stay_last_idx, on='ed_stay_id', how='left')

    order_steps = order_steps.dropna(subset=['order_step', 'last_idx'])
    order_steps[['order_step', 'last_idx']] = order_steps[['order_step', 'last_idx']].astype(int)
    order_steps['result_step'] = (
        order_steps['result_step'].fillna(order_steps['last_idx'] + 1).astype(int)
    )
    order_steps = order_steps.sort_values('order_time').reset_index(drop=True)

    # Deduplicate to worst-case result per (ed_stay_id, action, result_step)
    # Abnormal (1) beats Normal (0)
    order_steps['result_priority'] = order_steps['abnormal'].astype(int)
    order_steps = (
        order_steps
        .sort_values('result_priority', ascending=False)
        .drop_duplicates(subset=['ed_stay_id', 'action', 'result_step'], keep='first')
        .sort_values('order_time')
        .reset_index(drop=True)
    )

    # -- Pending windows (order_step to result_step - 1)
    pending = order_steps[order_steps['result_step'] > order_steps['order_step']].copy()
    pend_len = pending['result_step'] - pending['order_step']
    pend_off = np.arange(pend_len.sum())
    pend_off -= np.repeat(pend_len.cumsum().shift(1, fill_value=0).to_numpy(), pend_len.to_numpy())
    pend_exp = pending.loc[pending.index.repeat(pend_len)].copy()
    pend_exp['step_idx'] = np.repeat(pending['order_step'].to_numpy(), pend_len.to_numpy()) + pend_off
    pend_exp['ohe_col'] = pend_exp['action'] + '_Pending'

    # -- Result windows (result_step to last_idx)
    result_valid = order_steps[order_steps['last_idx'] >= order_steps['result_step']].copy()
    result_valid['ohe_col'] = np.where(
        result_valid['abnormal'],
        result_valid['action'] + '_Abnormal',
        result_valid['action'] + '_Normal',
    )
    res_len = result_valid['last_idx'] - result_valid['result_step'] + 1
    res_off = np.arange(res_len.sum())
    res_off -= np.repeat(res_len.cumsum().shift(1, fill_value=0).to_numpy(), res_len.to_numpy())
    res_exp = result_valid.loc[result_valid.index.repeat(res_len)].copy()
    res_exp['step_idx'] = np.repeat(result_valid['result_step'].to_numpy(), res_len.to_numpy()) + res_off

    # Deduplicate result_exp at (step_idx, action): Abnormal beats Normal
    res_exp['result_priority'] = res_exp['ohe_col'].str.endswith('_Abnormal').astype(int)
    res_exp = (
        res_exp
        .sort_values('result_priority', ascending=False)
        .drop_duplicates(subset=['step_idx', 'action'], keep='first')
        .drop(columns='result_priority')
    )

    all_exp = pd.concat([
        pend_exp[['step_idx', 'ohe_col']].assign(val=1),
        res_exp[['step_idx', 'ohe_col']].assign(val=1),
    ])
    ohe_pivot = all_exp.pivot_table(
        index='step_idx', columns='ohe_col', values='val', aggfunc='max', fill_value=0,
    )

    df_patient = merge_pivot_into_patient(df_patient, ohe_pivot)

    # labs_ordered: 1 when a _Pending column transitions 0->1
    pending_cols = [f'{a}_Pending' for a in lab_actions]
    df_patient['labs_ordered'] = (
        df_patient.groupby('ed_stay_id')[pending_cols]
        .diff().fillna(0).gt(0).any(axis=1).astype(int)
    )

    del df_labs, df_labs_orders, df_labs_results, order_steps, result_steps
    del pending, pend_exp, result_valid, res_exp, all_exp, ohe_pivot
    gc.collect()
    logger.info('Labs OHE expansion complete.')
    return df_patient, lab_actions
