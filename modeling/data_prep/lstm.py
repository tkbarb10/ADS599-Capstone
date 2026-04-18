"""
modeling/data_prep/lstm.py

Loads full_patient_state from HuggingFace, scales features, and builds
padded sequences for LSTM training and inference.

Functions:
  load_and_prep_lstm(hf_cfg) -- load + encode, return (df, state_cols)
  remove_outlier_stays(df) -- drop stays exceeding pad_length or stay_length
  split_data(df) -- stratified 80/10/10 train/test/val by subject_id
  scaling(train, test, val) -- StandardScaler fit on train, applied to all three
  pad_stays(df, state_cols) -- zero-pad sequences to pad_length, return arrays
  pad_data(train, test, val, ...) -- convert splits to DataLoaders
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling.data_prep.columns import get_column_groups, TERMINAL_MAP
from utils.load_yaml_helper import load_yaml

config = load_yaml('modeling/config/lstm.yaml')
data_config = config['data']
target_col = data_config['target_col']
pad_length = data_config['pad_length']
stay_length = data_config['stay_length']
random_state = config['random_state']
batch_size = data_config['batch_size']

logger = logging.getLogger(__name__)

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

    # Downcast to reduce memory: binary OHE cols to int8, continuous to float32.
    # pad_stays casts everything to float32 before building tensors anyway.
    binary_cols = [c for c in (
        groups.lab_ohe + groups.micro_ohe + groups.ecg_ohe + groups.rad_ohe
        + groups.dispensed_meds + groups.recon + groups.arrival + groups.missing
    ) if c in df.columns]
    df[binary_cols] = df[binary_cols].astype(np.int8)
    float_cols = [c for c in df.select_dtypes('float64').columns if c not in binary_cols]
    df[float_cols] = df[float_cols].astype(np.float32)
    df = df.copy()  # defragment after repeated column-level dtype changes
    logger.info(f'Downcast {len(binary_cols)} binary cols to int8, {len(float_cols)} float cols to float32')

    df['label'] = df['terminal_event'].map(TERMINAL_MAP)

    return df, state_cols


def remove_outlier_stays(df: pd.DataFrame) -> pd.DataFrame:
    valid_stays = df.groupby('ed_stay_id').size()
    valid_length = df[df['total_length'] < stay_length]['ed_stay_id'].values
    valid_actions = valid_stays[valid_stays < pad_length].index
    df_modeling = df[df['ed_stay_id'].isin(valid_actions)]
    df_modeling = df_modeling[df_modeling['ed_stay_id'].isin(valid_length)]
    logger.info(f'Stays after length filter: {df_modeling["ed_stay_id"].nunique():,}')
    return df_modeling


def split_data(
    df: pd.DataFrame,
    train_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/10/10 split by subject_id.

    Splitting by subject_id (not ed_stay_id) prevents data leakage when a
    patient has multiple ED visits. Because every row of a given stay shares
    the same subject_id, all time steps for each stay land in the same split --
    the groupby in pad_stays always sees complete sequences.
    """
    subject_label = (
        df[['subject_id', 'label']]
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True)
    )

    train_subs, holdout_subs = train_test_split(
        subject_label['subject_id'],
        train_size=train_size,
        stratify=subject_label['label'],
        random_state=random_state,
    )
    holdout_label = subject_label[subject_label['subject_id'].isin(holdout_subs)]
    test_subs, val_subs = train_test_split(
        holdout_label['subject_id'],
        test_size=0.5,
        stratify=holdout_label['label'],
        random_state=random_state,
    )

    df_train = df[df['subject_id'].isin(train_subs)].copy()
    df_test = df[df['subject_id'].isin(test_subs)].copy()
    df_val = df[df['subject_id'].isin(val_subs)].copy()

    logger.info(f'Train stays: {df_train["ed_stay_id"].nunique():,}')
    logger.info(df_train.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'Test stays:  {df_test["ed_stay_id"].nunique():,}')
    logger.info(df_test.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'Val stays:   {df_val["ed_stay_id"].nunique():,}')
    logger.info(df_val.drop_duplicates('ed_stay_id')['terminal_event'].value_counts())
    logger.info(f'Overlap check: {set(train_subs) & set(holdout_subs)}')

    return df_train, df_test, df_val


def scaling(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train rows only, transform all three splits."""
    groups = get_column_groups(train)
    cols = [c for c in groups.scaling_cols if c in train.columns]
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    val[cols] = scaler.transform(val[cols])
    logger.info(f'Scaled {len(cols)} columns')
    return train, test, val, scaler


def pad_stays(df: pd.DataFrame, state_cols: List[str], max_len: int = pad_length):
    grouped = df.groupby('ed_stay_id')
    states, labels, lengths = [], [], []

    for _, group in grouped:
        s = group[state_cols].values.astype(np.float32)
        pad_len = max_len - len(s)
        s = np.pad(s, ((0, pad_len), (0, 0)))  # pad rows, not features
        label = int(group['label'].iloc[0])
        states.append(s)
        labels.append(label)
        lengths.append(len(group))

    return np.stack(states), np.array(labels, dtype=np.int64), np.array(lengths)


def pad_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
    state_cols: List[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    import gc
    print("Padding train loader...")
    # -- Train ------------------------------------------------------------
    s_train, y_train, train_len = pad_stays(df=train, state_cols=state_cols)
    y_train_t = torch.tensor(y_train)
    state_dim = s_train.shape[2]
    n_train = s_train.shape[0]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(s_train), y_train_t, torch.tensor(train_len)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(random_state)
    )
    del s_train, y_train, train_len
    gc.collect()
    print("Train loader created and intermediate file deleted, processing test loader...")
    # -- Test -------------------------------------------------------------
    s_test, y_test, test_len = pad_stays(df=test, state_cols=state_cols)
    y_test_t = torch.tensor(y_test)
    n_test = s_test.shape[0]
    test_loader = DataLoader(
        TensorDataset(torch.tensor(s_test), y_test_t, torch.tensor(test_len)),
        batch_size=batch_size, shuffle=False,
    )
    del s_test, y_test, test_len
    gc.collect()

    # -- Val --------------------------------------------------------------
    s_val, y_val, val_len = pad_stays(df=val, state_cols=state_cols)
    y_val_t = torch.tensor(y_val)
    n_val = s_val.shape[0]
    val_loader = DataLoader(
        TensorDataset(torch.tensor(s_val), y_val_t, torch.tensor(val_len)),
        batch_size=batch_size, shuffle=False,
    )
    del s_val, y_val, val_len
    gc.collect()

    logger.info(f'State dim:   {state_dim}')
    logger.info(f'Train stays: {n_train:,}  discharge={(y_train_t==0).sum():,}  icu={(y_train_t==1).sum():,}')
    logger.info(f'Test stays:  {n_test:,}   discharge={(y_test_t==0).sum():,}  icu={(y_test_t==1).sum():,}')
    logger.info(f'Val stays:   {n_val:,}    discharge={(y_val_t==0).sum():,}  icu={(y_val_t==1).sum():,}')

    return train_loader, test_loader, val_loader
