"""
modeling/data_prep/traditional_ml.py

Loads full_patient_state from HuggingFace, scales features, and returns
a flat (one-row-per-step) DataFrame ready for traditional ML training.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)
