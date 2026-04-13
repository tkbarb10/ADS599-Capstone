"""
modeling/rl_agent/agent.py

DQN agent with Conservative Q-Learning (CQL) regularization.

  - ProviderNetwork  -- Q-network mapping state -> Q-values for 2 actions
  - DQNAgent         -- wraps target/online networks, Bellman update, CQL loss
"""

import logging

import torch
import torch.nn as nn

from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)
