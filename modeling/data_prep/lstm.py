"""
modeling/data_prep/lstm.py

Loads full_patient_state from HuggingFace, scales features, and builds
padded sequences for LSTM training and inference.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)
