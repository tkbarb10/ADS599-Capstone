"""
modeling/rl_agent/environment.py

Replay buffer for offline RL training on patient trajectory data.

Builds MDP tuples (state, event_idx, action, reward, terminal, next_state)
from train/test DataFrames and wraps them in DataLoaders.

MDP tuple column meaning:
  state      -- (B, state_dim) feature tensor
  event_idx  -- (B,) within-stay event counter
  action     -- (B,) int64 provider disposition (0=discharge, 1=transfer_icu)
  reward     -- (B,) pre-computed reward
  terminal   -- (B, 2) float: [label, terminal_code]
                  terminal[:, 0] = true label (0/1) -- used in reward
                  terminal[:, 1] = is_terminal flag -- used in Bellman masking
  next_state -- (B, state_dim) next row's features (zeros at terminal rows)
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        state_cols: List[str],
        batch_size: int,
        random_state: int,
    ):
        self.df_train = df_train
        self.df_test = df_test
        self.state_cols = state_cols
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_tensors(self, df: pd.DataFrame):
        df = df.sort_values(['ed_stay_id', 'event_idx'])

        state = torch.tensor(df[self.state_cols].values.astype(np.float32))
        event_idx = torch.tensor(df['event_idx'].values.astype(np.float32))
        action = torch.tensor(df['action_taken'].values.astype(np.int64))
        reward = torch.tensor(df['reward'].values.astype(np.float32))
        terminal = torch.tensor(
            df[['label', 'terminal_code']].values.astype(np.float32)
        )
        next_state = torch.tensor(
            df.groupby('ed_stay_id')[self.state_cols]
            .shift(-1)
            .fillna(0)
            .values.astype(np.float32)
        )

        logger.info(f'State dim:      {state.shape}')
        logger.info(f'Action dim:     {action.shape}')
        logger.info(f'Reward dim:     {reward.shape}')
        logger.info(f'Terminal dim:   {terminal.shape}')
        logger.info(f'Next state dim: {next_state.shape}')

        return TensorDataset(state, event_idx, action, reward, terminal, next_state)

    def build(self) -> tuple[DataLoader, DataLoader]:
        """Build and return (train_loader, test_loader)."""
        torch.manual_seed(self.random_state)

        train_dataset = self._build_tensors(self.df_train)
        r_sampler = RandomSampler(train_dataset, replacement=False)
        batch_sampler = BatchSampler(r_sampler, batch_size=self.batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

        test_dataset = self._build_tensors(self.df_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
