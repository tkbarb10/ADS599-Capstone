"""
modeling/traditional_ml/evaluate.py

Evaluates trained traditional ML models.
Reports AUC, F1, precision, recall, and classification report.

Usage:
    python -m modeling.traditional_ml.evaluate
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.traditional_ml import load_and_prep

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['train_path'])
