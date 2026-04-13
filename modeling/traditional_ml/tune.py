"""
modeling/traditional_ml/tune.py

Hyperparameter tuning for logistic regression, random forest, and XGBoost
using Optuna. Writes best params back to modeling/config/traditional_ml.yaml.

Usage:
    python -m modeling.traditional_ml.tune
"""

import argparse
import logging
from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.traditional_ml import load_and_prep

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['tune_path'])
