"""
modeling/lstm/train.py

Trains LSTM models using hyperparameters from modeling/config/lstm.yaml.
Saves model weights and fitted scaler to artifacts/.

Usage:
    python -m modeling.lstm.train
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.lstm import load_and_prep
from modeling.lstm.model import SbsSequenceModeling

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/lstm.yaml')
logger = setup_logging(settings['logging']['train_path'])
