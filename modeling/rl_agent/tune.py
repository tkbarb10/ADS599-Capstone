"""
modeling/rl_agent/tune.py

Hyperparameter tuning for the RL agent using Optuna.
Tunes network architecture, optimizer params (lr, weight_decay),
and RL-specific params (gamma, cql_weight, target_update_freq).
Reward shaping params (base_reward, class_ratio, time_penalty) are
tuned separately -- see reward_sweep.py.
Writes best params back to modeling/config/rl_agent.yaml.

Usage:
    python -m modeling.rl_agent.tune
"""

import argparse
import logging
from pathlib import Path

import optuna
import torch

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.rl import load_and_prep
from modeling.rl_agent.agent import DQNAgent
from modeling.rl_agent.environment import PatientEnvironment, ReplayBuffer

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/rl_agent.yaml')
logger = setup_logging(settings['logging']['tune_path'])
