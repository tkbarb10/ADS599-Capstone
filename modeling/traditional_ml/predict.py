"""
modeling/traditional_ml/predict.py

Generates predictions from trained traditional ML models on held-out data.

Usage:
    python -m modeling.traditional_ml.predict
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.traditional_ml import load_and_prep

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['predict_path'])
