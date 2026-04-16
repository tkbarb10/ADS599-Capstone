"""
modeling/data_prep/rl.py

Loads sbs_predictions from HuggingFace and prepares (state, action, reward,
next_state, terminal) data for offline RL training.

Functions:
  load_and_prep_rl(hf_cfg)            -- full prep pipeline, returns (df, state_cols)
  create_action_col(df, cols, suffix) -- indicator column for when orders were placed
  add_delta_p_icu(df)                 -- within-stay change in LSTM confidence
  add_event_idx(df)                   -- within-stay event counter
  split_rl_data(df)                   -- stratified 80/20 split by subject_id
  scaling(train, test)                -- StandardScaler fit on train, transform both
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling.data_prep.columns import get_column_groups
from modeling.rl_agent.rl_functions import reward_function
from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)

settings = load_yaml("project_setup/settings.yaml")
rl_config = load_yaml("modeling/config/rl_agent.yaml")
rl_data = rl_config['data']

random_state = rl_config['random_state']
test_size = rl_data['test_size']
scaling_cols = rl_data['scaling_cols']


def create_action_col(df: pd.DataFrame, cols: List[str], suffix: str) -> pd.DataFrame:
    """Create an indicator column for when an order was placed.

    Detects the first 0->1 transition in any of `cols` within each stay,
    then marks the row *before* that transition as the order row.
    """
    result_onset = (
        df.groupby('ed_stay_id')[cols]
        .diff()
        .gt(0)
        .any(axis=1)
    )
    df[f'{suffix}_ordered'] = result_onset.shift(-1, fill_value=False).astype(int)
    return df


def add_delta_p_icu(df: pd.DataFrame) -> pd.DataFrame:
    """Within-stay change in LSTM ICU confidence (used in reward shaping)."""
    df['delta_p_icu'] = df.groupby('ed_stay_id')['p_icu'].diff().fillna(0)
    return df


def add_event_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Within-stay event counter. Data is event-driven, not time-based."""
    df['event_idx'] = df.groupby('ed_stay_id').cumcount()
    return df


def split_rl_data(
    df: pd.DataFrame,
    test_size: float = test_size,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split by subject_id."""
    subject_label = (
        df[['subject_id', 'label']]
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True)
    )

    train_subs, test_subs = train_test_split(
        subject_label['subject_id'],
        test_size=test_size,
        stratify=subject_label['label'],
        random_state=random_state,
    )

    df_train = df[df['subject_id'].isin(train_subs)].copy()
    df_test = df[df['subject_id'].isin(test_subs)].copy()

    logger.info(f'Train stays: {df_train["ed_stay_id"].nunique():,}')
    logger.info(df_train.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'Test stays: {df_test["ed_stay_id"].nunique():,}')
    logger.info(df_test.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'Overlap: {set(train_subs) & set(test_subs)}')

    return df_train, df_test


def scaling(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train rows only, transform both splits."""
    cols = [c for c in scaling_cols if c in train.columns]
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    logger.info(f'Scaled {len(cols)} columns')
    return train, test, scaler


def load_and_prep_rl(hf_cfg: dict) -> tuple[pd.DataFrame, List[str]]:
    """Full RL data prep pipeline.

    Loads sbs_predictions from HuggingFace, creates action indicator columns,
    derives delta_p_icu and event_idx, sets action_taken from the provider's
    true disposition label, and returns the prepared DataFrame with state_cols.

    Args:
        hf_cfg: hugging_face section of settings.yaml.

    Returns:
        (df, state_cols)
        df         -- prepared DataFrame with all RL columns
        state_cols -- ordered list of feature columns for the Q-network
    """
    logger.info('Loading sbs_predictions from HuggingFace...')
    ds = load_dataset(
        hf_cfg['step_by_step'],
        name=hf_cfg['sbs_data']['config_name'],
        split=hf_cfg['sbs_data']['split_name'],
        verification_mode='no_checks',
    )
    df = ds.to_pandas().copy()
    logger.info(f'Loaded {len(df):,} rows -- {df["ed_stay_id"].nunique():,} unique stays')

    # Create action indicator columns for ECG, rad, and meds (results only in raw data)
    groups_pre = get_column_groups(df)
    df = create_action_col(df, groups_pre.ecg_ohe, 'ecg')
    df = create_action_col(df, groups_pre.rad_ohe, 'rad')
    df = create_action_col(df, groups_pre.dispensed_meds, 'meds')

    # Re-detect column groups now that action cols exist
    groups = get_column_groups(df)
    state_cols = groups.state_cols

    df = add_delta_p_icu(df)
    df = add_event_idx(df)

    # Non-terminal rows: provider is implicitly waiting (action=2)
    # Terminal row: provider made the actual disposition call (0=discharge, 1=transfer_icu)
    df['action_taken'] = np.where(df['terminal_code'] == 1, df['label'], 2)

    # Compute per-row reward
    reward_cfg = rl_config['reward']
    df['reward'] = df.apply(
        lambda row: reward_function(
            action=row['action_taken'],
            terminal=[row['label'], row['terminal_code']],
            event_idx=row['event_idx'],
            p_icu=row['p_icu'],
            class_ratio=reward_cfg['class_ratio'],
            time_weight=reward_cfg['time_penalty'],
            base_reward=reward_cfg['base_reward'],
        ),
        axis=1,
    )

    logger.info(f'State cols: {len(state_cols)}')
    return df, state_cols
