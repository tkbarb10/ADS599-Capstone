"""
modeling/data_prep/lstm.py

Loads full_patient_state from HuggingFace, scales features, and builds
padded sequences for LSTM training and inference.
"""

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedShuffleSplit

from modeling.data_prep.columns import get_column_groups
from utils.load_yaml_helper import load_yaml

config = load_yaml('modeling/config/lstm.yaml')
data_config = config['data']
target_col = data_config['target_col']
scaling_cols = data_config['scaling_cols']

pad_length = data_config['pad_length']
stay_length = data_config['stay_length']
test_size = data_config['test_size']
random_state = config['random_state']

logger = logging.getLogger(__name__)


TERMINAL_MAP = {'discharge': 0, 'transfer_icu': 1}

def load_and_prep_lstm(hf_cfg: dict) -> tuple[pd.DataFrame, List[str]]:
    logger.info('Loading full_patient_state from HuggingFace...')
    ds = load_dataset(
        hf_cfg['modeling_data_repo'],
        name=hf_cfg['full_patient_state']['config_name'],
        split=hf_cfg['full_patient_state']['split_name'],
        verification_mode='no_checks',
        )
    df = ds.to_pandas().copy()
    logger.info(f'Loaded {len(df):,} rows -- {df["ed_stay_id"].nunique():,} unique stays')

    df['gender'] = df['gender'].map({'F': 1, 'M': 0})
    df.drop(columns=['admission_type', 'recon_n_total_meds', 'recon_n_drug_classes'],
            inplace=True, errors='ignore')

    groups = get_column_groups(df)
    state_cols = groups.state_cols
    logger.info(f'State cols: {len(state_cols)}')

    df['label'] = df['terminal_event'].map(TERMINAL_MAP)

    return df, state_cols

def remove_outlier_stays(df: pd.DataFrame):
    valid_stays = df.groupby("ed_stay_id").size()
    valid_length = df[df['total_length'] < stay_length]['ed_stay_id'].values
    valid_actions = valid_stays[valid_stays < pad_length].index
    df_modeling = df[df["ed_stay_id"].isin(valid_actions)]
    df_modeling = df_modeling[df_modeling['ed_stay_id'].isin(valid_length)]
    logger.info(f"Stays after length filter: {df_modeling['ed_stay_id'].nunique():,}")
    return df_modeling

def split_data(df: pd.DataFrame, test_size: float=test_size):
    stay_labels = (
        df.drop_duplicates("ed_stay_id")[["ed_stay_id", "terminal_code"]]
        .reset_index(drop=True)
    )

    train_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(train_split.split(stay_labels, stay_labels["terminal_code"]))

    train_stays = set(stay_labels.iloc[train_idx]["ed_stay_id"])
    test_stays  = set(stay_labels.iloc[test_idx]["ed_stay_id"])

    df_train = df[df["ed_stay_id"].isin(train_stays)].copy()
    df_test  = df[df["ed_stay_id"].isin(test_stays)].copy()

    logger.info(f"Train stays: {df_train['ed_stay_id'].nunique():,}")
    logger.info(df_train.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f"\nTest stays:  {df_test['ed_stay_id'].nunique():,}")
    logger.info(df_test.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f"\nOverlap: {train_stays.intersection(test_stays)}")

    return df_train, df_test

def pad_stays(df, state_cols: List[str], max_len: int=pad_length):
    grouped = df.groupby("ed_stay_id")
    states, labels, lengths = [], [], []

    for stay_id, group in grouped:
        s = group[state_cols].values.astype(np.float32)
        pad_len = max_len - len(s)
        s = np.pad(s, ((0, pad_len), (0, 0)))  # pad rows, not features
        label = int(group["terminal_code"].iloc[0])  # single label per stay
        states.append(s)
        labels.append(label)
        lengths.append(len(group))

    return np.stack(states), np.array(labels, dtype=np.int64), np.array(lengths)
