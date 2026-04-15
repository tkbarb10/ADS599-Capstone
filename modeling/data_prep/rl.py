"""
modeling/data_prep/rl.py

Loads confidence_delta_data from HuggingFace and builds (state, action,
reward, next_state, terminal) tensors for offline RL training.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler

from utils.load_yaml_helper import load_yaml
logger = logging.getLogger(__name__)

settings = load_yaml("project_setup/settings.yaml")
rl_config = load_yaml("modeling/config/rl_agent.yaml")
rl_data = rl_config['data']

hf_cfg = settings['hugging_face']
data_path = hf_cfg['modeling_data_repo']
data_cfg = hf_cfg['full_patient_state']

random_state = rl_config['random_state']
test_size = rl_data['test_size']
batch_size = rl_data['batch_size']
scaling_cols = rl_data['scaling_cols']


action_cols = ['vitals_checked', 'labs_ordered', 'micro_ordered', 'ecg_ordered', 'rad_ordered', 'meds_ordered']
terminal_states = ['discharge', 'transfer_icu']

def split_rl_data(df: pd.DataFrame):
    stay_labels = (
        df.drop_duplicates('ed_stay_id')[['ed_stay_id', 'terminal_event']]
        .reset_index(drop=True)
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(split.split(stay_labels, stay_labels['terminal_event']))

    train_stays = set(stay_labels.iloc[train_idx]['ed_stay_id'])
    test_stays = set(stay_labels.iloc[test_idx]['ed_stay_id'])

    df_train = df[df['ed_stay_id'].isin(train_stays)].copy()
    df_test = df[df['ed_stay_id'].isin(test_stays)].copy()

    logger.info(f'Train stays: {df_train["ed_stay_id"].nunique():,}')
    logger.info(df_train.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'\nTest stays: {df_test["ed_stay_id"].nunique():,}')
    logger.info(df_test.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'\nOverlap: {train_stays.intersection(test_stays)}')

def scaling(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train rows only, transform splits."""
    cols = [c for c in scaling_cols if c in train.columns]
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    logger.info(f'Scaled {len(cols)} columns')
    return train, test, scaler

# Before this, rewards need to be computed
# actions taken need to be condensed
# time steps within groups
# terminal_event mapped to int

def create_replay_buffer(df_train: pd.DataFrame, df_test: pd.DataFrame, state_cols: list[str]):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    df_train.sort_values(['ed_stay_id', 'time_steps'], inplace=True)
    df_test.sort_values(['ed_stay_id', 'time_steps'], inplace=True)

    grouped_train = df_train.groupby('ed_stay_id', as_index=False)
    grouped_test = df_test.groupby('ed_stay_id', as_index=False)

    state_train = torch.tensor(df_train[state_cols].values.astype(np.float32))
    time_train = torch.tensor(df_train['time_steps'].values.astype(np.float32))
    action_train = torch.tensor(df_train['action_taken'].values.astype(np.int32))
    reward_train = torch.tensor(df_train['reward'].values.astype(np.float32))
    terminal_train = torch.tensor(df_train[['terminal_code', 'terminal_event']].values.astype(np.float32))
    next_state_train = torch.tensor(grouped_train[state_cols].shift(-1).fillna(0).values.astype(np.float32))

    logger.info(f'State dim:      {state_train.shape}')
    logger.info(f'Action dim:     {action_train.shape}')
    logger.info(f'Reward dim:     {reward_train.shape}')
    logger.info(f'Terminal dim:   {terminal_train.shape}')
    logger.info(f'Next state dim: {next_state_train.shape}')

    del df_train, grouped_train

    # Tuple order: (state, time, action, reward, terminal, next_state)
    final_train = TensorDataset(state_train, time_train, action_train, reward_train, terminal_train, next_state_train)
    del state_train, time_train, action_train, reward_train, terminal_train, next_state_train

    r_sampler = RandomSampler(final_train, replacement=False)
    batch_sampler = BatchSampler(r_sampler, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(dataset=final_train, shuffle=False, batch_sampler=batch_sampler)
    del final_train

    state_test = torch.tensor(df_test[state_cols].values.astype(np.float32))
    time_test = torch.tensor(df_test['time_steps'].values.astype(np.float32))
    action_test = torch.tensor(df_test['action_taken'].values.astype(np.int32))
    reward_test = torch.tensor(df_test['reward'].values.astype(np.float32))
    terminal_test = torch.tensor(df_test[['terminal_code', 'terminal_event']].values.astype(np.float32))
    next_state_test = torch.tensor(grouped_test[state_cols].shift(-1).fillna(0).values.astype(np.float32))

    test_loader = DataLoader(
        TensorDataset(state_test, time_test, action_test, reward_test, terminal_test, next_state_test),
        batch_size=batch_size, shuffle=False,
    )

    return train_loader, test_loader

