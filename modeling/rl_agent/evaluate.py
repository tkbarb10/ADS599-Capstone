"""
modeling/rl_agent/evaluate.py

Evaluates the trained RL agent at terminal timesteps only.
Reports F1, precision, recall (ICU as positive class) and decision
timing relative to actual provider decisions.

Usage:
    python -m modeling.rl_agent.evaluate
"""

import logging
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.rl import load_and_prep
from modeling.rl_agent.agent import DQNAgent

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/rl_agent.yaml')
logger = setup_logging(settings['logging']['train_path'])
