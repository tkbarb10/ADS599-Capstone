"""
modeling/rl_agent/train.py

Offline DQN training loop using hyperparameters from modeling/config/rl_agent.yaml.
Saves trained agent weights to artifacts/.

Usage:
    python -m modeling.rl_agent.train
"""

import argparse
import logging
from pathlib import Path

import torch

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.rl import load_and_prep
from modeling.rl_agent.agent import DQNAgent
from modeling.rl_agent.environment import PatientEnvironment, ReplayBuffer

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/rl_agent.yaml')
logger = setup_logging(settings['logging']['train_path'])
