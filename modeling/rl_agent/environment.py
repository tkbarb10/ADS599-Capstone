"""
modeling/rl_agent/environment.py

Offline RL environment and replay buffer for patient trajectory data.

  - PatientEnvironment  -- wraps the confidence_delta_data DataFrame as a step/reset env
  - ReplayBuffer        -- fixed-capacity circular buffer storing (s, a, r, s', terminal) tuples
"""

import logging
from collections import deque

import numpy as np
import torch

from utils.load_yaml_helper import load_yaml

logger = logging.getLogger(__name__)
