"""
modeling/rl_agent/train.py

Offline DQN training loop with CQL regularization and target network.

Pipeline:
  1. Load and prep sbs_predictions via load_and_prep_rl
  2. Split and scale
  3. Build replay buffer (train + test DataLoaders)
  4. Instantiate train_network and target_network (ProviderNetwork)
  5. Train with CQL loss; update target_network every target_update_freq steps
  6. Save learned policy weights to artifacts/rl_policy/

Usage:
    python -m modeling.rl_agent.train
"""

import copy
import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modeling.data_prep.rl import load_and_prep_rl, split_rl_data, scaling
from modeling.rl_agent.agent import ProviderNetwork
from modeling.rl_agent.environment import ReplayBuffer
from modeling.rl_agent.rl_functions import bellman
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/rl_agent.yaml')
logger = setup_logging(settings['logging']['rl_train_path'])

PROJECT_ROOT = Path(__file__).parents[2]

# -- Config -------------------------------------------------------------------
random_state = cfg['random_state']

network_cfg = cfg['network']
train_cfg = cfg['training']
artifact_cfg = cfg['artifacts']
hf_cfg = settings['hugging_face']

epochs = train_cfg['epochs']
learning_rate = train_cfg['learning_rate']
weight_decay = train_cfg['weight_decay']
gamma = train_cfg['gamma']
target_update_freq = train_cfg['target_update_freq']
cql_weight = train_cfg['cql_weight']
training_update = train_cfg['training_update']
batch_size = cfg['data']['batch_size']

# -- Reproducibility ----------------------------------------------------------
torch.manual_seed(random_state)
np.random.seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -- Device -------------------------------------------------------------------
device = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else 'cpu'
)

# =============================================================================
# Training loop
# =============================================================================

def training_loop(loader, train_net, target_net, optimizer, loss_fn, epoch):
    train_net.train()
    total_loss = 0.0
    total_q_loss = 0.0
    total_cql = 0.0
    num_batches = len(loader)

    for step, (s, _, action, reward, terminal, ns) in enumerate(loader):
        s = s.to(device)
        ns = ns.to(device)
        reward = reward.to(device)
        action = action.to(device)
        terminal = terminal.to(device)

        # Bellman targets from frozen target network
        q_targets = bellman(target_net, reward, ns, terminal, gamma)

        # Re-enable train mode after bellman sets eval
        train_net.train()

        q_all = train_net(s)                                              # (B, 3)
        q_taken = q_all.gather(1, action.unsqueeze(1)).squeeze(1)        # (B,)

        q_loss = loss_fn(q_taken, q_targets)
        cql_reg = (torch.logsumexp(q_all, dim=1) - q_taken).mean()
        loss = q_loss + cql_weight * cql_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_q_loss += q_loss.item()
        total_cql += cql_reg.item()

        if step % target_update_freq == 0:
            target_net.load_state_dict(train_net.state_dict())

        if step % training_update == 0:
            logger.info(
                f'Epoch {epoch} | Step {step:,}/{num_batches:,} | '
                f'Loss: {loss.item():.4f}  Q-loss: {q_loss.item():.4f}  '
                f'CQL: {cql_reg.item():.4f}  '
                f'Q-taken: {q_taken.mean().item():.4f}  Q-all: {q_all.mean().item():.4f}'
            )

    return total_loss / num_batches, total_q_loss / num_batches, total_cql / num_batches


def eval_loop(loader, train_net, target_net, loss_fn):
    train_net.eval()
    total_loss = 0.0

    with torch.no_grad():
        for s, _, action, reward, terminal, ns in loader:
            s = s.to(device)
            ns = ns.to(device)
            reward = reward.to(device)
            action = action.to(device)
            terminal = terminal.to(device)

            q_targets = bellman(target_net, reward, ns, terminal, gamma)
            q_all = train_net(s)
            q_taken = q_all.gather(1, action.unsqueeze(1)).squeeze(1)
            q_loss = loss_fn(q_taken, q_targets)
            cql_reg = (torch.logsumexp(q_all, dim=1) - q_taken).mean()
            total_loss += (q_loss + cql_weight * cql_reg).item()

    return total_loss / len(loader)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    logger.info(f'Using device: {device}')

    df, state_cols = load_and_prep_rl(hf_cfg)
    df_train, df_test = split_rl_data(df)
    df_train, df_test, scaler = scaling(df_train, df_test)

    buffer = ReplayBuffer(
        df_train=df_train,
        df_test=df_test,
        state_cols=state_cols,
        batch_size=batch_size,
        random_state=random_state,
    )
    train_loader, test_loader = buffer.build()

    input_size = len(state_cols)
    train_network = ProviderNetwork(
        input_size=input_size,
        hidden_sizes=network_cfg['hidden_sizes'],
        output_size=network_cfg['output_size'],
        dropout=network_cfg['dropout'],
    ).to(device)

    target_network = copy.deepcopy(train_network)
    target_network.load_state_dict(train_network.state_dict())

    optimizer = torch.optim.AdamW(
        train_network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    logger.info(f'Input size: {input_size}  |  Architecture: {network_cfg["hidden_sizes"]}')

    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        logger.info(f'Epoch {epoch} / {epochs} --------------------')
        avg_loss, avg_q_loss, avg_cql = training_loop(
            train_loader, train_network, target_network, optimizer, loss_fn, epoch
        )
        eval_loss = eval_loop(test_loader, train_network, target_network, loss_fn)
        logger.info(
            f'Epoch {epoch} summary | '
            f'Train loss: {avg_loss:.4f}  Q-loss: {avg_q_loss:.4f}  '
            f'CQL: {avg_cql:.4f}  |  Test loss: {eval_loss:.4f}'
        )
        epoch_metrics.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_q_loss': avg_q_loss,
            'train_cql': avg_cql,
            'eval_loss': eval_loss,
        })

    # -- Save policy and metrics ----------------------------------------------
    policy_dir = PROJECT_ROOT / artifact_cfg['policy_dir']
    policy_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    policy_path = policy_dir / f'rl_policy_{timestamp}.pt'
    torch.save(train_network.state_dict(), policy_path)
    logger.info(f'Saved policy -> {policy_path.name}')

    scaler_path = policy_dir / f'rl_scaler_{timestamp}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f'Saved scaler  -> {scaler_path.name}')

    metrics_path = policy_dir / f'rl_metrics_{timestamp}.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_metrics[0].keys())
        writer.writeheader()
        writer.writerows(epoch_metrics)
    logger.info(f'Saved metrics -> {metrics_path.name}')
