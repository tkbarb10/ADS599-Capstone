"""
modeling/lstm/predict.py

Runs StepwiseWrapper inference on full_patient_state to generate per-timestep
p_icu probabilities, then pushes the result to HuggingFace as confidence_delta_data.

Usage:
    python -m modeling.lstm.predict
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.lstm import load_and_prep
from modeling.lstm.model import StepwiseWrapper

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/lstm.yaml')
logger = setup_logging(settings['logging']['predict_path'])
