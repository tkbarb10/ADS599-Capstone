"""
modeling/rl_agent/evaluate.py

Evaluates the trained RL policy on the held-out test split.

Two perspectives:
  1. Terminal row analysis -- at the actual decision point, did the agent agree?
     Reports pct_wait and F1 on rows where the agent chose to act.
  2. First-commit analysis -- when did the agent first predict a terminal action,
     and was it accurate? How does that compare to when the provider decided?

Plots saved to artifacts/rl_agent_evaluation/.

Usage:
    python -m modeling.rl_agent.evaluate
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

from modeling.data_prep.columns import get_column_groups
from modeling.data_prep.rl import load_and_prep_rl, split_rl_data
from modeling.rl_agent.agent import ProviderNetwork
from modeling.rl_agent.rl_functions import ACTION_LABELS
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/rl_agent.yaml')
logger = setup_logging(settings['logging']['evaluate_path'])

PROJECT_ROOT = Path(__file__).parents[2]

network_cfg = cfg['network']
artifact_cfg = cfg['artifacts']
hf_cfg = settings['hugging_face']
batch_size = cfg['data']['batch_size']

policy_dir = PROJECT_ROOT / artifact_cfg['policy_dir']
eval_dir = PROJECT_ROOT / 'modeling/artifacts/rl_agent_evaluation'

device = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else 'cpu'
)

TARGET_NAMES = ['Discharge', 'ICU Transfer']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'No files matching {pattern!r} in {directory}')
    return matches[-1]


def load_policy(policy_path: Path, input_size: int) -> ProviderNetwork:
    model = ProviderNetwork(
        input_size=input_size,
        hidden_sizes=network_cfg['hidden_sizes'],
        output_size=network_cfg['output_size'],
        dropout=network_cfg['dropout'],
    ).to(device)
    model.load_state_dict(torch.load(policy_path, map_location=device))
    model.eval()
    logger.info(f'Loaded policy: {policy_path.name}')
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: ProviderNetwork, df: pd.DataFrame, state_cols: List[str]) -> pd.DataFrame:
    """Run policy over df rows, add pred_action and per-action Q-value columns."""
    states = torch.tensor(df[state_cols].values.astype(np.float32))
    all_q = []

    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = states[i:i + batch_size].to(device)
            all_q.append(model(batch).cpu())

    q_vals = torch.cat(all_q, dim=0).numpy()  # (N, 3)
    df = df.copy()
    df['pred_action'] = q_vals.argmax(axis=1)
    df['q_discharge'] = q_vals[:, 0]
    df['q_transfer'] = q_vals[:, 1]
    df['q_wait'] = q_vals[:, 2]
    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def first_commit_per_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stay first event where agent predicted a terminal action (0 or 1).

    Returns a DataFrame with one row per committed stay:
      ed_stay_id, commit_event_idx, commit_action, label, provider_event_idx
    """
    commits = (
        df[df['pred_action'].isin([0, 1])]
        .sort_values(['ed_stay_id', 'event_idx'])
        .drop_duplicates('ed_stay_id', keep='first')
        [['ed_stay_id', 'event_idx', 'pred_action', 'label']]
        .rename(columns={'event_idx': 'commit_event_idx', 'pred_action': 'commit_action'})
    )
    terminal_events = (
        df[df['terminal_code'] == 1][['ed_stay_id', 'event_idx']]
        .rename(columns={'event_idx': 'provider_event_idx'})
    )
    return commits.merge(terminal_events, on='ed_stay_id', how='left')


def terminal_row_metrics(df_test: pd.DataFrame) -> dict:
    """Metrics at the true terminal row (terminal_code == 1) for each stay.

    F1 is computed only on rows where the agent chose to act (pred_action in {0, 1}).
    Rows where the agent predicted wait are counted separately via pct_wait.
    """
    df_term = df_test[df_test['terminal_code'] == 1].copy()
    n_terminal = len(df_term)
    n_predict_wait = int((df_term['pred_action'] == 2).sum())
    n_predict_act = n_terminal - n_predict_wait

    acted = df_term[df_term['pred_action'].isin([0, 1])]
    if len(acted) == 0:
        f1_dis = f1_icu = f1_macro = 0.0
    else:
        f1_dis = float(f1_score(acted['label'], acted['pred_action'], pos_label=0, zero_division=0))
        f1_icu = float(f1_score(acted['label'], acted['pred_action'], pos_label=1, zero_division=0))
        f1_macro = float(f1_score(acted['label'], acted['pred_action'], average='macro', zero_division=0))

    return {
        'n_terminal': n_terminal,
        'n_predict_wait': n_predict_wait,
        'n_predict_act': n_predict_act,
        'pct_wait': round(n_predict_wait / n_terminal, 4) if n_terminal else 0.0,
        'f1_discharge': round(f1_dis, 4),
        'f1_icu': round(f1_icu, 4),
        'f1_macro': round(f1_macro, 4),
    }


def first_commit_metrics(df_commits: pd.DataFrame, n_total_stays: int) -> dict:
    """Metrics at the first event where the agent committed to a terminal action."""
    n_committed = len(df_commits)
    n_no_commit = n_total_stays - n_committed

    if n_committed == 0:
        return {
            'n_stays_committed': 0,
            'n_stays_no_commit': n_no_commit,
            'pct_no_commit': 1.0,
            'f1_discharge': 0.0,
            'f1_icu': 0.0,
            'f1_macro': 0.0,
            'median_commit_event': None,
            'median_provider_event': None,
            'pct_early': 0.0,
        }

    f1_dis = float(f1_score(df_commits['label'], df_commits['commit_action'], pos_label=0, zero_division=0))
    f1_icu = float(f1_score(df_commits['label'], df_commits['commit_action'], pos_label=1, zero_division=0))
    f1_macro = float(f1_score(df_commits['label'], df_commits['commit_action'], average='macro', zero_division=0))
    pct_early = float((df_commits['commit_event_idx'] < df_commits['provider_event_idx']).mean())

    return {
        'n_stays_committed': n_committed,
        'n_stays_no_commit': n_no_commit,
        'pct_no_commit': round(n_no_commit / n_total_stays, 4),
        'f1_discharge': round(f1_dis, 4),
        'f1_icu': round(f1_icu, 4),
        'f1_macro': round(f1_macro, 4),
        'median_commit_event': int(df_commits['commit_event_idx'].median()),
        'median_provider_event': int(df_commits['provider_event_idx'].median()),
        'pct_early': round(pct_early, 4),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_csv: Path, eval_dir: Path, timestamp: str) -> None:
    df = pd.read_csv(metrics_csv)
    best_epoch = int(df['eval_loss'].idxmin()) + 1

    # Two subplots with independent y-axes -- train loss varies much more than eval loss,
    # so a shared axis makes eval appear as a flat line.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(df['epoch'], df['train_loss'], lw=2, color='#0077BB', label='Train Loss (total)')
    ax1.plot(df['epoch'], df['train_q_loss'], lw=2, color='#009988', linestyle='--',
             label='Train Q-Loss (no CQL)')
    ax1.axvline(best_epoch, color='gray', linestyle=':', lw=1.5,
                label=f'Best Epoch ({best_epoch})')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2.plot(df['epoch'], df['eval_loss'], lw=2, color='#EE6677', label='Eval Loss')
    ax2.axvline(best_epoch, color='gray', linestyle=':', lw=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Evaluation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    fig.suptitle('RL Agent Training and Evaluation Loss', fontsize=13, fontweight='bold')
    fig.tight_layout()

    out = eval_dir / f'rl_loss_curves_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved loss curves -> {out.name}')


def plot_terminal_action_dist(df_test: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    """Grouped bar: for each true label, fraction of terminal rows the agent predicted each action."""
    df_term = df_test[df_test['terminal_code'] == 1].copy()
    action_names = {0: 'Discharge', 1: 'ICU Transfer', 2: 'Wait'}
    colors = {'Discharge': '#0077BB', 'ICU Transfer': '#CC3311', 'Wait': '#BBBBBB'}
    true_groups = [0, 1]
    group_labels = ['True: Discharge', 'True: ICU Transfer']

    fractions = {}
    for g in true_groups:
        sub = df_term[df_term['label'] == g]
        total = len(sub)
        for a in [0, 1, 2]:
            fractions[(g, a)] = int((sub['pred_action'] == a).sum()) / total if total else 0.0

    x = np.arange(len(true_groups))
    width = 0.25
    offsets = [-width, 0, width]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (a, a_name) in enumerate(action_names.items()):
        vals = [fractions[(g, a)] for g in true_groups]
        ax.bar(x + offsets[i], vals, width, label=a_name, color=colors[a_name], alpha=0.85)

    for gi, g in enumerate(true_groups):
        n = int((df_term['label'] == g).sum())
        ax.text(x[gi], 1.08, f'n={n:,}', ha='center', fontsize=9, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Fraction of Terminal Rows')
    ax.set_title('Agent Predicted Action at True Terminal Events')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    out = eval_dir / f'rl_terminal_action_dist_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved terminal action distribution -> {out.name}')


def plot_terminal_confusion(df_test: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    """2x2 confusion matrix on terminal rows where agent predicted a terminal action."""
    acted = df_test[(df_test['terminal_code'] == 1) & (df_test['pred_action'].isin([0, 1]))]
    if len(acted) == 0:
        logger.warning('No acted terminal rows -- skipping confusion matrix')
        return

    cm = confusion_matrix(acted['label'], acted['pred_action'])
    cm_norm = confusion_matrix(acted['label'], acted['pred_action'], normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-Normalized Proportion', fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(TARGET_NAMES)
    ax.set_yticklabels(TARGET_NAMES)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix\nRL Agent (Terminal Rows, Acted Only)')
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.text(0.99, -0.13, f'n = {len(acted):,}', transform=ax.transAxes,
            ha='right', fontsize=9, color='gray')

    out = eval_dir / f'rl_terminal_confusion_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved terminal confusion matrix -> {out.name}')


def plot_f1_comparison(term_metrics: dict, commit_metrics: dict,
                       eval_dir: Path, timestamp: str) -> None:
    """Grouped bar: F1(discharge) and F1(ICU) at terminal row vs first commit."""
    groups = ['At Terminal Row', 'At First Commit']
    f1_discharge = [term_metrics['f1_discharge'], commit_metrics['f1_discharge']]
    f1_icu = [term_metrics['f1_icu'], commit_metrics['f1_icu']]

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars_dis = ax.bar(x - width / 2, f1_discharge, width, color='#0077BB', alpha=0.85,
                      label='F1 (Discharge)')
    bars_icu = ax.bar(x + width / 2, f1_icu, width, color='#CC3311', alpha=0.85,
                      label='F1 (ICU Transfer)')

    for bar, val in zip(bars_dis, f1_discharge):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#0077BB')
    for bar, val in zip(bars_icu, f1_icu):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#CC3311')

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score: Terminal Row vs. First Commit')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.25)

    out = eval_dir / f'rl_f1_comparison_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved F1 comparison -> {out.name}')


def plot_commit_timing(df_commits: pd.DataFrame, df_test: pd.DataFrame,
                       eval_dir: Path, timestamp: str) -> None:
    """Overlaid histograms: agent first-commit event index vs provider decision event index."""
    provider_events = df_test[df_test['terminal_code'] == 1]['event_idx']
    max_event = int(max(provider_events.max(), df_commits['commit_event_idx'].max()))
    bins = np.arange(0, max_event + 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_commits['commit_event_idx'], bins=bins, alpha=0.6, color='#0077BB',
            label=f'Agent First Commit  (n={len(df_commits):,})', density=True)
    ax.hist(provider_events, bins=bins, alpha=0.6, color='#EE7733',
            label=f'Provider Decision  (n={len(provider_events):,})', density=True)
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Density')
    ax.set_title('Decision Timing: Agent First Commit vs. Provider Decision')
    ax.legend()
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'rl_commit_timing_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved commit timing -> {out.name}')


def _with_stay_buckets(df_commits: pd.DataFrame) -> pd.DataFrame:
    """Add stay_bucket and steps_earlier columns to df_commits.

    Four equal-frequency quartile bins computed from the actual provider_event_idx
    distribution. Labels include the event-count ranges so they are self-explanatory.
    """
    df = df_commits.copy()
    df['steps_earlier'] = (df['provider_event_idx'] - df['commit_event_idx']).clip(lower=0)

    lengths = df['provider_event_idx']
    q1 = int(lengths.quantile(0.25))
    q2 = int(lengths.quantile(0.50))
    q3 = int(lengths.quantile(0.75))

    labels = [
        f'Q1 (≤{q1} events)',
        f'Q2 ({q1+1}-{q2})',
        f'Q3 ({q2+1}-{q3})',
        f'Q4 ({q3+1}+)',
    ]
    bins = [-0.5, q1 + 0.5, q2 + 0.5, q3 + 0.5, lengths.max() + 0.5]
    df['stay_bucket'] = pd.cut(df['provider_event_idx'], bins=bins, labels=labels)
    return df


def plot_f1_by_stay_length(df_commits: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    """Bar chart: F1(ICU transfer) at first commit, segmented by total stay length."""
    df = _with_stay_buckets(df_commits)
    rows = []
    for bucket, grp in df.groupby('stay_bucket', observed=True):
        if len(grp) == 0:
            continue
        f1_icu = float(f1_score(grp['label'], grp['commit_action'], pos_label=1, zero_division=0))
        rows.append({'bucket': str(bucket), 'f1_icu': f1_icu, 'n': len(grp)})
    metrics = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    ax.bar(x, metrics['f1_icu'], color='#4C72B0', alpha=0.85)
    for xi, row in zip(x, metrics.itertuples()):
        if row.f1_icu > 0:
            ax.text(xi, row.f1_icu + 0.01, f'{row.f1_icu:.2f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['bucket'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 (transfer_icu)')
    ax.set_title('RL Agent F1 by Stay Length')
    ax.grid(axis='y', alpha=0.25)

    out = eval_dir / f'rl_f1_by_stay_length_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved F1 by stay length -> {out.name}')


def plot_early_detection_by_stay_length(df_commits: pd.DataFrame,
                                        eval_dir: Path, timestamp: str) -> None:
    """Bar chart: median steps earlier than provider, segmented by total stay length."""
    df = _with_stay_buckets(df_commits)
    rows = []
    for bucket, grp in df.groupby('stay_bucket', observed=True):
        if len(grp) == 0:
            continue
        median_steps = float(grp['steps_earlier'].median())
        rows.append({'bucket': str(bucket), 'median_steps_earlier': median_steps, 'n': len(grp)})
    metrics = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    ax.bar(x, metrics['median_steps_earlier'], color='#CC3311', alpha=0.85)
    for xi, row in zip(x, metrics.itertuples()):
        ax.text(xi, row.median_steps_earlier + 0.5, f'{row.median_steps_earlier:.0f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['bucket'])
    ax.set_ylabel('Median steps earlier than provider')
    ax.set_title('Early Detection by Stay Length')
    ax.grid(axis='y', alpha=0.25)

    out = eval_dir / f'rl_early_detection_by_stay_length_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved early detection by stay length -> {out.name}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logger.info(f'Using device: {device}')

    # 1. Load and prep data
    df, state_cols = load_and_prep_rl(hf_cfg)
    _, df_test = split_rl_data(df)

    # 2. Scale test features using the saved scaler (fit on train, not test)
    scaler = pickle.load(open(_latest(policy_dir, 'rl_scaler_*.pkl'), 'rb'))
    cols_to_scale = [c for c in get_column_groups(df_test).scaling_cols if c in df_test.columns]
    df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

    # 3. Load policy and run inference
    policy = load_policy(_latest(policy_dir, 'rl_policy_*.pt'), len(state_cols))
    df_test = run_inference(policy, df_test, state_cols)

    # 4. Metrics
    n_stays = df_test['ed_stay_id'].nunique()
    term_metrics = terminal_row_metrics(df_test)
    df_commits = first_commit_per_stay(df_test)
    commit_metrics = first_commit_metrics(df_commits, n_stays)

    logger.info(f'Terminal row metrics: {term_metrics}')
    logger.info(f'First commit metrics: {commit_metrics}')

    # 5. Plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(_latest(policy_dir, 'rl_metrics_*.csv'), eval_dir, timestamp)
    plot_terminal_action_dist(df_test, eval_dir, timestamp)
    plot_terminal_confusion(df_test, eval_dir, timestamp)
    plot_f1_comparison(term_metrics, commit_metrics, eval_dir, timestamp)
    plot_commit_timing(df_commits, df_test, eval_dir, timestamp)
    plot_f1_by_stay_length(df_commits, eval_dir, timestamp)
    plot_early_detection_by_stay_length(df_commits, eval_dir, timestamp)

    # 6. Save per-stay commit summary
    df_commits.to_csv(eval_dir / f'rl_commit_summary_{timestamp}.csv', index=False)
    logger.info(f'Saved commit summary -> rl_commit_summary_{timestamp}.csv')

    logger.info('RL evaluation complete.')
