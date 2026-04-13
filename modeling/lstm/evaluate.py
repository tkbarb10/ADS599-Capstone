"""
modeling/lstm/evaluate.py

Evaluates trained LSTM on held-out data.
Reports AUC, F1, precision, recall, and calibration metrics.

Usage:
    python -m modeling.lstm.evaluate
"""

import logging
from pathlib import Path

import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.lstm import load_and_prep
from modeling.lstm.model import StepwiseWrapper

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/lstm.yaml')
logger = setup_logging(settings['logging']['train_path'])
