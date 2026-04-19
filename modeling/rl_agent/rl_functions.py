"""
modeling/rl_agent/rl_functions.py

Reward function and Bellman equation for offline DQN training.
"""

import torch

from modeling.data_prep.columns import TERMINAL_MAP

ACTION_LABELS = {v: k for k, v in TERMINAL_MAP.items()} | {2: 'wait'}  # {0: 'discharge', 1: 'transfer_icu', 2: 'wait'}


def reward_function(
    action: int,
    terminal: list,
    event_idx: int,
    p_icu: float,
    class_ratio: float = 2.0,
    time_weight: float = -0.001,
    base_reward: float = 0.4,
) -> float:
    """Confidence-weighted correctness reward + event-based time penalty.

    At terminal rows: reward scaled by LSTM ICU confidence (p_icu).
      - Correct discharge: +base_reward * p_icu
      - Correct ICU transfer: +base_reward * class_ratio * p_icu  (minority bonus)
      - Wrong discharge (missed): -base_reward * class_ratio * p_icu  (heaviest penalty)
      - Wrong ICU transfer: -base_reward * p_icu

    All rows: event_idx * time_weight (encourages earlier decisions).

    Args:
        action: Provider's action (0=discharge, 1=transfer_icu, 2=wait).
        terminal: [label, terminal_code] -- label is true disposition (0/1),
                     terminal_code is is_terminal flag (0/1).
        event_idx: Within-stay event counter.
        p_icu: LSTM step-by-step ICU transfer probability.
        class_ratio: Multiplier applied to ICU-class rewards/penalties.
        time_weight: Per-event time penalty (negative).
        base_reward: Base reward magnitude.

    Returns:
        Scalar reward value.
    """
    label = int(terminal[0])
    is_terminal = int(terminal[1])
    action_label = ACTION_LABELS[int(action)]
    true_label = ACTION_LABELS[label]
    time_penalty = event_idx * time_weight
    lt_reward = 0.0

    if is_terminal:
        confidence = float(p_icu)
        if action_label == true_label:
            if action_label == 'transfer_icu':
                lt_reward += base_reward * class_ratio * confidence
            else:
                lt_reward += base_reward * confidence
        else:
            if action_label == 'discharge' and true_label == 'transfer_icu':
                lt_reward -= base_reward * class_ratio * confidence
            else:
                lt_reward -= base_reward * confidence

    return lt_reward + time_penalty


def bellman(
    model: torch.nn.Module,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    terminal: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Compute Q-learning target values.

    Q_target(s,a) = r + gamma * max_a' Q(s', a') * (1 - is_terminal)

    Args:
        model: Target network used for stable Q-value estimation.
        reward: (batch_size,) reward tensor.
        next_state: (batch_size, state_dim) next state tensor.
        terminal: (batch_size, num_actions) tensor -- terminal[:, 1] is the is_terminal flag.
        gamma: Discount factor.

    Returns:
        (batch_size,) Q-target tensor.
    """
    is_terminal = terminal[:, 1]
    model.eval()
    with torch.no_grad():
        max_q = model(next_state).max(dim=1).values
    return reward + gamma * max_q * (1 - is_terminal) # cancels out and just left with the reward if is a terminal row
