"""
modeling/lstm/model.py

LSTM model definitions:
  - SequenceModeling      -- final-state classifier, returns (B, 2) logits
  - SbsSequenceModeling   -- per-timestep classifier, returns (B, T, 2) logits
  - StepwiseWrapper       -- wraps SbsSequenceModeling for inference, exposes p_icu
"""

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)
