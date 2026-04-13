"""
modeling/lstm/tune.py

Hyperparameter tuning for the LSTM using Optuna.
Writes best params back to modeling/config/lstm.yaml.

Usage:
    python -m modeling.lstm.tune
"""

import argparse
import logging
from pathlib import Path

import optuna
import torch

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.lstm import load_and_prep
from modeling.lstm.model import SbsSequenceModeling

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/lstm.yaml')
logger = setup_logging(settings['logging']['tune_path'])
