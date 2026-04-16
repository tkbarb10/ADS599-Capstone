"""
modeling/rl_agent/agent.py

Q-network for offline DQN training.

  ProviderNetwork -- maps patient state -> Q-values for 2 actions (discharge / transfer_icu)
                     Architecture is fully config-driven via hidden_sizes list.
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class ProviderNetwork(nn.Module):
    """Feedforward Q-network.

    Builds the layer stack dynamically from hidden_sizes so the architecture
    is controlled entirely by rl_agent.yaml.

    Args:
        input_size:   Number of state features (derived at load time from state_cols).
        hidden_sizes: List of hidden layer widths, e.g. [256, 256, 128].
        output_size:  Number of actions (2: discharge, transfer_icu).
        dropout:      Dropout rate applied after each hidden layer.
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, output_size) Q-values
