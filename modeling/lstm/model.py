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
from torch.nn.utils.rnn import pack_padded_sequence

from utils.load_yaml_helper import load_yaml

config = load_yaml("modeling/config/lstm.yaml")
model_config = config['model']

input_size = model_config['input_size']
hidden_size = model_config['hidden_size']
num_layers = model_config['num_layers']
batch_first = model_config['batch_first']
dropout = model_config['dropout']
num_classes = model_config['num_classes']

logger = logging.getLogger(__name__)


class LSTMSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=batch_first, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers, batch, hidden_size)
        # take the last layer's hidden state for classification
        out = self.fc(self.dropout(h_n[-1]))
        return out

