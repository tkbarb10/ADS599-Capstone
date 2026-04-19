"""
modeling/traditional_ml/evaluate.py

Loads prediction parquets for one or more trained models, computes evaluation
metrics, and saves plots and a metrics summary to artifacts/evaluation/.

Usage:
    python -m modeling.traditional_ml.evaluate --models log_reg
    python -m modeling.traditional_ml.evaluate --models log_reg random_forest xgboost
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
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
config = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['evaluate_path'])

PROJECT_ROOT = Path(__file__).parents[2]

TARGET_NAMES = ['Discharge', 'ICU']
DROP_COLS = {'label', 'pred_label', 'pred_prob', 'ed_stay_id', 'subject_id', 'in_ed', 'in_ward'}

MODEL_COLORS = {
    'log_reg': '#0077BB',
    'random_forest': '#009988',
    'xgboost': '#EE7733',
}
MODEL_LABELS = {
    'log_reg': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
}
MODEL_PREFIX = {
    'log_reg': 'log_reg',
    'random_forest': 'random_forest',
    'xgboost': 'xgboost',
}

# Publication-quality rcParams applied globally for the session
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
# Helpers
# ---------------------------------------------------------------------------

def _latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'No files matching {pattern!r} in {directory}')
    return matches[-1]


def _prior_correct(p: np.ndarray, prior_test: float, prior_train: float = 0.5) -> np.ndarray:
    """Correct predicted probabilities from balanced training prior to true test prior.

    All models are trained with class_weight='balanced' or equivalent, making the
    effective training prior 0.5. This shifts predicted probabilities toward 0.5
    relative to the true ICU prevalence (~7.9%). The Bayes odds correction maps
    them back to the natural prevalence space for calibration and distribution plots.
    """
    odds = p / np.clip(1.0 - p, 1e-10, None)
    correction = (prior_test / (1.0 - prior_test)) / (prior_train / (1.0 - prior_train))
    odds_corr = odds * correction
    return odds_corr / (1.0 + odds_corr)


def _load_predictions(model_name: str, tuned: bool = False) -> pd.DataFrame:
    artifact_dir = PROJECT_ROOT / config['artifacts'][f'{model_name}_dir']
    pattern = f'best_tuned_test_predictions_*.parquet' if tuned else 'test_predictions_*.parquet'
    path = _latest(artifact_dir, pattern)
    logger.info(f'[{model_name}] Loading predictions: {path.name}')
    return pd.read_parquet(path)


def _load_model(model_name: str, tuned: bool = False) -> object:
    artifact_dir = PROJECT_ROOT / config['artifacts'][f'{model_name}_dir']
    pattern = f'best_tuned_{MODEL_PREFIX[model_name]}_*.pkl' if tuned else f'{MODEL_PREFIX[model_name]}_[0-9]*.pkl'
    path = _latest(artifact_dir, pattern)
    logger.info(f'[{model_name}] Loading model: {path.name}')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        'Accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'F1 (ICU)': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'F1 (Macro)': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_true, y_prob)), 4),
        'PR-AUC': round(float(average_precision_score(y_true, y_prob)), 4),
        'Brier Score': round(float(brier_score_loss(y_true, y_prob)), 4),
    }


# ---------------------------------------------------------------------------
# Plots
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
    # Restore all spines for the matrix box
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
    """Precision-Recall curve -- more informative than ROC for imbalanced classes."""
    fig, ax = plt.subplots(figsize=(7, 6))

    baseline = 0.0
    for model_name, (y_true, _, y_prob) in model_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[model_name],
                label=f'{MODEL_LABELS[model_name]}  (AP = {ap:.4f})')
        baseline = float(y_true.mean())  # same across models (shared test set)

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
        prior_test = float(np.asarray(y_true).mean())
        y_prob_arr = np.asarray(y_prob)

        frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob_arr, n_bins=10)
        ax.plot(mean_pred_raw, frac_pos_raw, '--', lw=1.5,
                color=MODEL_COLORS[model_name], alpha=0.45)

        y_prob_corr = _prior_correct(y_prob_arr, prior_test)
        frac_pos_corr, mean_pred_corr = calibration_curve(y_true, y_prob_corr, n_bins=10)
        ax.plot(mean_pred_corr, frac_pos_corr, 's-', lw=2,
                color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Mean Predicted Probability (Prior-Corrected)')
    ax.set_ylabel('Fraction of Positives (Observed Rate)')
    ax.set_title('Calibration Curve\n(solid = prior-corrected, dashed = raw)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.25)

    out = eval_dir / f'calibration_curve_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved calibration curve -> {out.name}')


def plot_prob_distribution(model_results: dict, eval_dir: Path, timestamp: str) -> None:
    """Predicted probability distributions by true class -- shows model discrimination.

    Probabilities are prior-corrected from the balanced training space (effective
    prior 0.5) back to the true test prevalence so distributions reflect the
    natural ICU rate rather than the resampled training distribution.
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    class_colors = {0: '#4477AA', 1: '#EE6677'}
    for ax, (model_name, (y_true, _, y_prob)) in zip(axes, model_results.items()):
        prior_test = float(np.asarray(y_true).mean())
        y_prob_corr = _prior_correct(np.asarray(y_prob), prior_test)

        for label_val, label_name in [(0, 'Discharge'), (1, 'ICU')]:
            mask = np.asarray(y_true) == label_val
            ax.hist(y_prob_corr[mask], bins=30, alpha=0.6, color=class_colors[label_val],
                    label=label_name, density=True, edgecolor='white', linewidth=0.5)

        ax.axvline(prior_test, color='gray', linestyle='--', lw=1.5,
                   label=f'ICU Prevalence ({prior_test:.1%})')
        ax.set_xlabel('Predicted Probability (ICU Transfer, Prior-Corrected)')
        ax.set_ylabel('Density')
        ax.set_title(MODEL_LABELS[model_name])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Predicted Probability Distribution by True Class\n(Prior-Corrected for Class Weighting)',
                 fontsize=13, fontweight='bold', y=1.02)
    out = eval_dir / f'prob_distribution_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved probability distribution -> {out.name}')


def plot_feature_importance(model, df_preds: pd.DataFrame, model_name: str,
                             eval_dir: Path, timestamp: str, top_n: int = 20) -> None:
    feature_cols = [c for c in df_preds.columns if c not in DROP_COLS]

    if model_name == 'log_reg':
        importances = np.abs(model.coef_[0])
        x_label = 'Absolute Coefficient Value (Standardized Features)'
    else:
        importances = model.feature_importances_
        x_label = 'Feature Importance (Mean Decrease in Impurity / Gain)'

    feat_series = pd.Series(importances, index=feature_cols).nlargest(top_n).iloc[::-1]
    feat_values = feat_series.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
    bars = ax.barh(feat_series.index, feat_values,
                   color=MODEL_COLORS[model_name], edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, feat_values):
        ax.text(val + feat_series.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8, color='#444444')

    ax.set_xlabel(x_label)
    ax.set_title(f'Top {top_n} Feature Importances\n{MODEL_LABELS[model_name]}')
    ax.set_xlim(right=feat_series.max() * 1.15)
    ax.grid(axis='x', alpha=0.25)

    out = eval_dir / f'feature_importance_{model_name}_{timestamp}.png'
    fig.savefig(out)
    plt.close(fig)
    logger.info(f'Saved feature importance -> {out.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate traditional ML models.')
    parser.add_argument('--models', nargs='+', required=True,
                        choices=list(MODEL_LABELS),
                        help='One or more models: log_reg | random_forest | xgboost')
    parser.add_argument('--tuned', action='store_true',
                        help='Load best_tuned_* artifacts instead of baseline trained models')
    args = parser.parse_args()

    eval_dir = PROJECT_ROOT / config['artifacts']['evaluation_dir']
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_prefix = 'tuned_model_comparison' if args.tuned else 'metrics'

    # -----------------------------------------------------------------------
    # Load predictions and models
    # -----------------------------------------------------------------------
    model_results = {}  # model_name -> (y_true, y_pred, y_prob)
    model_preds = {}
    models = {}

    for model_name in args.models:
        df = _load_predictions(model_name, tuned=args.tuned)
        model_results[model_name] = (df['label'], df['pred_label'], df['pred_prob'])
        model_preds[model_name] = df
        models[model_name] = _load_model(model_name, tuned=args.tuned)

    # -----------------------------------------------------------------------
    # Metrics table
    # -----------------------------------------------------------------------
    rows = []
    for model_name, (y_true, y_pred, y_prob) in model_results.items():
        row = compute_metrics(y_true, y_pred, y_prob)
        row['Model'] = MODEL_LABELS[model_name]
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index('Model')
    logger.info(f'\n{metrics_df.to_string()}')

    metrics_path = eval_dir / f'{file_prefix}_{timestamp}.csv'
    metrics_df.to_csv(metrics_path)
    logger.info(f'Metrics saved -> {metrics_path.name}')

    # -----------------------------------------------------------------------
    # Confusion matrices (one per model)
    # -----------------------------------------------------------------------
    plot_ts = f'{file_prefix}_{timestamp}'
    for model_name, (y_true, y_pred, _) in model_results.items():
        plot_confusion_matrix(y_true, y_pred, model_name, eval_dir, plot_ts)

    # -----------------------------------------------------------------------
    # Multi-model comparison plots
    # -----------------------------------------------------------------------
    plot_roc_curve(model_results, eval_dir, plot_ts)
    plot_pr_curve(model_results, eval_dir, plot_ts)
    plot_calibration_curve(model_results, eval_dir, plot_ts)
    plot_prob_distribution(model_results, eval_dir, plot_ts)

    # -----------------------------------------------------------------------
    # Feature importance (one per model)
    # -----------------------------------------------------------------------
    for model_name in args.models:
        plot_feature_importance(models[model_name], model_preds[model_name],
                                model_name, eval_dir, plot_ts)

    logger.info('Evaluation complete.')
