"""
modeling/lstm/evaluate.py

Loads LSTM test predictions and training losses from artifacts/lstm/, computes
evaluation metrics, and saves plots to artifacts/evaluation/.  Optionally overlays
traditional ML models on the shared comparison plots.

Usage:
    python -m modeling.lstm.evaluate
    python -m modeling.lstm.evaluate --models lstm log_reg random_forest xgboost
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml('project_setup/settings.yaml')
config_trad = load_yaml('modeling/config/traditional_ml.yaml')
config_lstm = load_yaml('modeling/config/lstm.yaml')
logger = setup_logging(settings['logging']['evaluate_path'])

PROJECT_ROOT = Path(__file__).parents[2]

TARGET_NAMES = ['Discharge', 'ICU']

MODEL_COLORS = {
    'log_reg': '#0077BB',
    'random_forest': '#009988',
    'xgboost': '#EE7733',
    'lstm': '#CC3311',
}
MODEL_LABELS = {
    'log_reg': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lstm': 'LSTM',
}

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


def _prior_correct(p: np.ndarray, prior_test: float, prior_train: float = 0.5) -> np.ndarray:
    """Correct predicted probabilities from balanced training prior to true test prior.

    LSTM uses CrossEntropyLoss(weight=[1.0, n_dis/n_icu]), making the effective
    training prior 0.5. This Bayes odds correction maps probabilities back to the
    natural ICU prevalence space for calibration and distribution plots.
    """
    odds = p / np.clip(1.0 - p, 1e-10, None)
    correction = (prior_test / (1.0 - prior_test)) / (prior_train / (1.0 - prior_train))
    odds_corr = odds * correction
    return odds_corr / (1.0 + odds_corr)


def _load_lstm_predictions() -> pd.DataFrame:
    artifact_dir = PROJECT_ROOT / config_lstm['artifacts']['dir']
    path = _latest(artifact_dir, 'test_predictions_*.parquet')
    logger.info(f'[lstm] Loading predictions: {path.name}')
    df = pd.read_parquet(path)

    missing = {'n_actions', 'total_length'} - set(df.columns)
    if missing:
        logger.info(f'Columns {missing} not in predictions parquet -- deriving from HuggingFace...')
        from datasets import load_dataset
        hf_cfg = settings['hugging_face']
        raw = load_dataset(
            hf_cfg['modeling_data_repo'],
            name=hf_cfg['full_patient_state']['config_name'],
            split=hf_cfg['full_patient_state']['split_name'],
            verification_mode='no_checks',
            columns=['ed_stay_id', 'total_length'],
        ).to_pandas()

        if 'n_actions' in missing:
            n_actions = raw.groupby('ed_stay_id').size().rename('n_actions')
            df = df.join(n_actions, on='ed_stay_id')

        if 'total_length' in missing:
            los = (
                raw[['ed_stay_id', 'total_length']]
                .drop_duplicates('ed_stay_id')
                .set_index('ed_stay_id')['total_length']
            )
            df = df.join(los, on='ed_stay_id')

        del raw

    return df


def _load_trad_predictions(model_name: str) -> pd.DataFrame:
    artifact_dir = PROJECT_ROOT / config_trad['artifacts'][f'{model_name}_dir']
    path = _latest(artifact_dir, 'test_predictions_*.parquet')
    logger.info(f'[{model_name}] Loading predictions: {path.name}')
    return pd.read_parquet(path)


def _load_losses() -> pd.DataFrame:
    artifact_dir = PROJECT_ROOT / config_lstm['artifacts']['dir']
    path = _latest(artifact_dir, 'training_losses_*.parquet')
    logger.info(f'Loading losses: {path.name}')
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        'Accuracy': round(float((y_true == y_pred).mean()), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'F1': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_true, y_prob)), 4),
        'PR-AUC': round(float(average_precision_score(y_true, y_prob)), 4),
        'Brier Score': round(float(brier_score_loss(y_true, y_prob)), 4),
    }


# ---------------------------------------------------------------------------
# Shared plots (same interface as traditional_ml/evaluate.py)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name: str, eval_dir: Path, timestamp: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-Normalized Proportion', fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')

    ax.set_xticks(range(len(TARGET_NAMES)))
    ax.set_yticks(range(len(TARGET_NAMES)))
    ax.set_xticklabels(TARGET_NAMES)
    ax.set_yticklabels(TARGET_NAMES)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix\n{MODEL_LABELS[model_name]}')
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.text(0.99, -0.13, f'n = {len(y_true):,}', transform=ax.transAxes,
            ha='right', fontsize=9, color='gray')

    out = eval_dir / f'confusion_matrix_{model_name}_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved confusion matrix -> {out.name}')


def plot_roc_curve(model_results: dict, eval_dir: Path, timestamp: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, _, y_prob) in model_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[model_name],
                label=f'{MODEL_LABELS[model_name]}  (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier  (AUC = 0.5000)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'roc_curve_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved ROC curve -> {out.name}')


def plot_pr_curve(model_results: dict, eval_dir: Path, timestamp: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    baseline = 0.0
    for model_name, (y_true, _, y_prob) in model_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[model_name],
                label=f'{MODEL_LABELS[model_name]}  (AP = {ap:.4f})')
        baseline = float(y_true.mean())

    ax.axhline(baseline, color='k', linestyle='--', lw=1,
               label=f'No-Skill Baseline  ({baseline:.4f})')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'pr_curve_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved PR curve -> {out.name}')


def plot_calibration_curve(model_results: dict, eval_dir: Path, timestamp: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, _, y_prob) in model_results.items():
        prior_test = float(y_true.mean())
        p_corr = _prior_correct(y_prob.values if hasattr(y_prob, 'values') else y_prob, prior_test)

        # raw (dashed)
        frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred_raw, frac_pos_raw, '--', lw=1.5, alpha=0.45,
                color=MODEL_COLORS[model_name])

        # prior-corrected (solid)
        frac_pos, mean_pred = calibration_curve(y_true, p_corr, n_bins=10)
        ax.plot(mean_pred, frac_pos, 's-', lw=2, color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name])

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Mean Prior-Corrected Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Observed Rate)')
    ax.set_title('Calibration Curve\n(solid = prior-corrected, dashed = raw)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'calibration_curve_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved calibration curve -> {out.name}')


def plot_prob_distribution(model_results: dict, eval_dir: Path, timestamp: str) -> None:
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    class_colors = {0: '#4477AA', 1: '#EE6677'}
    for ax, (model_name, (y_true, _, y_prob)) in zip(axes, model_results.items()):
        prior_test = float(y_true.mean())
        p_arr = y_prob.values if hasattr(y_prob, 'values') else np.asarray(y_prob)
        p_corr = _prior_correct(p_arr, prior_test)

        for label_val, label_name in [(0, 'Discharge'), (1, 'ICU')]:
            mask = (y_true == label_val).values if hasattr(y_true, 'values') else y_true == label_val
            ax.hist(p_corr[mask], bins=30, alpha=0.6, color=class_colors[label_val],
                    label=label_name, density=True, edgecolor='white', linewidth=0.5)

        ax.axvline(prior_test, color='black', linestyle='--', lw=1.2,
                   label=f'ICU Prevalence ({prior_test:.1%})')
        ax.set_xlabel('Prior-Corrected Predicted Probability (ICU Transfer)')
        ax.set_ylabel('Density')
        ax.set_title(MODEL_LABELS[model_name])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Prior-Corrected Predicted Probability Distribution by True Class',
                 fontsize=13, fontweight='bold', y=1.02)
    out = eval_dir / f'prob_distribution_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved probability distribution -> {out.name}')


# ---------------------------------------------------------------------------
# LSTM-specific plots
# ---------------------------------------------------------------------------

def plot_loss_curves(losses_df: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    best_epoch = int(losses_df['eval_loss'].idxmin()) + 1  # 1-indexed

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses_df['epoch'], losses_df['train_loss'], lw=2, color='#0077BB', label='Train Loss')
    ax.plot(losses_df['epoch'], losses_df['eval_loss'], lw=2, color='#EE6677', label='Val Loss')
    ax.axvline(best_epoch, color='gray', linestyle='--', lw=1,
               label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('LSTM Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'lstm_loss_curves_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved loss curves -> {out.name}')


def _segmented_metrics(df: pd.DataFrame, bucket_col: str, bins, labels) -> pd.DataFrame:
    """Compute F1 for discharge and ICU classes for each bin of bucket_col."""
    df = df.copy()
    df['bucket'] = pd.cut(df[bucket_col], bins=bins, labels=labels, right=False)
    rows = []
    for label, group in df.groupby('bucket', observed=True):
        if len(group) == 0:
            continue
        f1_discharge = f1_score(group['label'], group['pred_label'], pos_label=0, zero_division=0)
        f1_icu = f1_score(group['label'], group['pred_label'], pos_label=1, zero_division=0)
        rows.append({'bucket': str(label), 'f1_discharge': f1_discharge, 'f1_icu': f1_icu, 'n': len(group)})
    return pd.DataFrame(rows)


def _plot_segmented(metrics_df: pd.DataFrame, x_label: str, title: str,
                    out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(metrics_df))
    width = 0.35

    bars_discharge = ax1.bar(x - width / 2, metrics_df['f1_discharge'], width,
                             color='#0077BB', alpha=0.8, label='F1 (Discharge)')
    bars_icu = ax1.bar(x + width / 2, metrics_df['f1_icu'], width,
                       color='#CC3311', alpha=0.8, label='F1 (ICU)')

    count_color = '#333333'
    ax2.plot(x, metrics_df['n'], 'o--', color=count_color, lw=1.5, markersize=5, label='Sample Count')

    for bar, val in zip(bars_discharge, metrics_df['f1_discharge']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#0077BB')
    for bar, val in zip(bars_icu, metrics_df['f1_icu']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#CC3311')

    label_offset = metrics_df['n'].max() * 0.03
    for xi, n in zip(x, metrics_df['n']):
        ax2.text(xi, n + label_offset, f'n={n:,}', ha='center', va='bottom',
                 fontsize=8, color=count_color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['bucket'])
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Metric Value')
    ax2.set_ylabel('Sample Count', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    ax1.grid(axis='y', alpha=0.25)

    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f'Saved segmented plot -> {out_path.name}')


def plot_by_num_actions(df_preds: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    bins = [1, 5, 10, 20, 50, float('inf')]
    labels = ['1-4', '5-9', '10-19', '20-49', '50+']
    metrics_df = _segmented_metrics(df_preds, 'n_actions', bins, labels)

    out = eval_dir / f'lstm_by_num_actions_{timestamp}.png'
    _plot_segmented(
        metrics_df,
        x_label='Number of Actions (Sequence Length)',
        title='LSTM Performance by Sequence Length',
        out_path=out,
    )


def plot_by_los(df_preds: pd.DataFrame, eval_dir: Path, timestamp: str) -> None:
    bins = [0, 1, 3, 7, 14, float('inf')]
    labels = ['<1 day', '1-2 days', '3-6 days', '7-13 days', '14+ days']
    metrics_df = _segmented_metrics(df_preds, 'total_length', bins, labels)

    out = eval_dir / f'lstm_by_los_{timestamp}.png'
    _plot_segmented(
        metrics_df,
        x_label='Length of Stay (Days)',
        title='LSTM Performance by Length of Stay',
        out_path=out,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LSTM model.')
    parser.add_argument(
        '--models', nargs='+',
        choices=['lstm', 'log_reg', 'random_forest', 'xgboost'],
        default=['lstm'],
        help='Models to include. lstm is always evaluated; others are overlaid on comparison plots.',
    )
    args = parser.parse_args()

    eval_dir = PROJECT_ROOT / config_trad['artifacts']['evaluation_dir']
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # -- Load predictions -----------------------------------------------------
    df_lstm = _load_lstm_predictions()
    model_results = {
        'lstm': (df_lstm['label'], df_lstm['pred_label'], df_lstm['pred_prob'])
    }

    for m in args.models:
        if m == 'lstm':
            continue
        df = _load_trad_predictions(m)
        model_results[m] = (df['label'], df['pred_label'], df['pred_prob'])

    # -- Metrics table --------------------------------------------------------
    rows = []
    for model_name, (y_true, y_pred, y_prob) in model_results.items():
        row = compute_metrics(y_true, y_pred, y_prob)
        row['Model'] = MODEL_LABELS[model_name]
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index('Model')
    logger.info(f'\n{metrics_df.to_string()}')

    metrics_path = eval_dir / f'metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_path)
    logger.info(f'Metrics saved -> {metrics_path.name}')

    # -- Confusion matrices ---------------------------------------------------
    for model_name, (y_true, y_pred, _) in model_results.items():
        plot_confusion_matrix(y_true, y_pred, model_name, eval_dir, timestamp)

    # -- Multi-model comparison plots -----------------------------------------
    plot_roc_curve(model_results, eval_dir, timestamp)
    plot_pr_curve(model_results, eval_dir, timestamp)
    plot_calibration_curve(model_results, eval_dir, timestamp)
    plot_prob_distribution(model_results, eval_dir, timestamp)

    # -- LSTM-specific plots --------------------------------------------------
    losses_df = _load_losses()
    plot_loss_curves(losses_df, eval_dir, timestamp)
    plot_by_num_actions(df_lstm, eval_dir, timestamp)
    plot_by_los(df_lstm, eval_dir, timestamp)

    logger.info('Evaluation complete.')
