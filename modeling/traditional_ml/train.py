"""
modeling/traditional_ml/train.py

Trains logistic regression, random forest, and XGBoost models using
hyperparameters from modeling/config/traditional_ml.yaml.
Saves fitted models and scaler to artifacts/.

Usage:
    python -m modeling.traditional_ml.train
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.traditional_ml import load_and_prep

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['train_path'])
