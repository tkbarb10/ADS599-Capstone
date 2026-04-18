"""
modeling/traditional_ml/tune.py

Hyperparameter tuning for logistic regression, random forest, and XGBoost
using HalvingGridSearchCV. Tunes on the saved validation set, re-evaluates
on the saved test set. Saves best model, params JSON, and prediction parquets.

Usage:
    python -m modeling.traditional_ml.tune --models log_reg
    python -m modeling.traditional_ml.tune --models log_reg random_forest xgboost
"""

import argparse
import json
import pickle
from datetime import datetime
from math import prod
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBClassifier

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml('project_setup/settings.yaml')
cfg = load_yaml('modeling/config/traditional_ml.yaml')
tune_cfg = load_yaml('modeling/config/tune.yaml')
logger = setup_logging(settings['logging']['tune_path'])

PROJECT_ROOT = Path(__file__).parents[2]
DROP_COLS = ['label', 'ed_stay_id', 'subject_id', 'in_ed', 'in_ward']

MODEL_PREFIX = {
    'log_reg': 'log_reg',
    'random_forest': 'random_forest',
    'xgboost': 'xgboost',
}


def _latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'No files matching {pattern!r} in {directory}')
    return matches[-1]


def _build_model(model_name: str, scale_pos_weight: float = 1.0):
    if model_name == 'log_reg':
        return LogisticRegression(random_state=10, class_weight='balanced')
    if model_name == 'random_forest':
        return RandomForestClassifier(random_state=10, n_jobs=-1, class_weight='balanced')
    if model_name == 'xgboost':
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=10,
            n_jobs=-1,
            scale_pos_weight=round(scale_pos_weight, 2),
            verbosity=0,
        )
    raise ValueError(f'Unknown model: {model_name}')


def tune_model(model_name: str, timestamp: str) -> None:
    artifact_dir = PROJECT_ROOT / cfg['artifacts'][f'{model_name}_dir']

    # Load pre-scaled splits saved during training
    train_path = _latest(artifact_dir, 'train_[0-9]*.parquet')
    val_path = _latest(artifact_dir, 'validation_[0-9]*.parquet')
    test_path = _latest(artifact_dir, 'test_[0-9]*.parquet')
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    logger.info(f'[{model_name}] train={len(train_df):,} from {train_path.name}')
    logger.info(f'[{model_name}] val={len(val_df):,} from {val_path.name}')
    logger.info(f'[{model_name}] test={len(test_df):,} from {test_path.name}')

    X_train = train_df.drop(columns=DROP_COLS, errors='ignore')
    y_train = train_df['label']
    X_val = val_df.drop(columns=DROP_COLS, errors='ignore')
    y_val = val_df['label']
    X_test = test_df.drop(columns=DROP_COLS, errors='ignore')
    y_test = test_df['label']

    param_grid = tune_cfg[model_name]
    grid_size = prod(len(v) for v in param_grid.values())
    logger.info(
        f'[{model_name}] {X_train.shape[1]} features, '
        f'{grid_size:,} param combos, '
        f'train class balance: {y_train.mean():.3%} positive'
    )

    # For XGBoost, compute scale_pos_weight from training set class ratio
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = _build_model(model_name, scale_pos_weight=spw)

    hgs_args = {k: v for k, v in tune_cfg['halving_grid_search'].items()}
    search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        **hgs_args,
        error_score=np.nan,
    )

    logger.info(f'[{model_name}] Starting HalvingGridSearchCV on train set...')
    t0 = time()
    search.fit(X_train, y_train)
    elapsed = time() - t0

    logger.info(
        f'[{model_name}] Done in {elapsed:.1f}s -- '
        f'best train CV macro F1: {search.best_score_:.4f}'
    )
    logger.info(f'[{model_name}] Best params: {search.best_params_}')

    # Sanity check on validation set
    val_pred_check = search.best_estimator_.predict(X_val)
    val_f1_check = f1_score(y_val, val_pred_check, average='macro')
    logger.info(f'[{model_name}] Val macro F1 (best estimator): {val_f1_check:.4f}')

    # Evaluate best estimator on test set
    best = search.best_estimator_
    test_pred = best.predict(X_test)
    test_prob = best.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, test_pred, average='macro')
    logger.info(f'[{model_name}] Test macro F1: {test_f1:.4f}')
    logger.info(
        f'[{model_name}] Classification report (test):\n'
        + classification_report(y_test, test_pred)
    )

    # Validation set predictions
    val_pred = best.predict(X_val)
    val_prob = best.predict_proba(X_val)[:, 1]

    # Save artifacts
    model_path = artifact_dir / f'best_tuned_{model_name}_{timestamp}.pkl'
    params_path = artifact_dir / f'best_tuned_{model_name}_params_{timestamp}.json'
    test_out = artifact_dir / f'best_tuned_test_predictions_{timestamp}.parquet'
    val_out = artifact_dir / f'best_tuned_validation_predictions_{timestamp}.parquet'

    with open(model_path, 'wb') as f:
        pickle.dump(best, f)

    params_doc = {
        'tuned_at': timestamp,
        'model': model_name,
        'meta': {
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'train_cv_f1_macro': round(search.best_score_, 6),
            'val_f1_macro': round(val_f1_check, 6),
            'test_f1_macro': round(test_f1, 6),
        },
        'best_params': search.best_params_,
    }
    with open(params_path, 'w') as f:
        json.dump(params_doc, f, indent=2)

    test_out_df = test_df.copy()
    test_out_df['pred_label'] = test_pred
    test_out_df['pred_prob'] = test_prob
    test_out_df.to_parquet(test_out, index=False)

    val_out_df = val_df.copy()
    val_out_df['pred_label'] = val_pred
    val_out_df['pred_prob'] = val_prob
    val_out_df.to_parquet(val_out, index=False)

    logger.info(f'[{model_name}] Model saved  -> {model_path.name}')
    logger.info(f'[{model_name}] Params saved -> {params_path.name}')
    logger.info(f'[{model_name}] Test preds   -> {test_out.name}')
    logger.info(f'[{model_name}] Val preds    -> {val_out.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune traditional ML models with HalvingGridSearchCV.')
    parser.add_argument(
        '--models', nargs='+', required=True,
        choices=list(MODEL_PREFIX),
        help='One or more models to tune: log_reg | random_forest | xgboost',
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model_name in args.models:
        logger.info(f'=== Tuning {model_name} ===')
        tune_model(model_name, timestamp)
        logger.info(f'=== {model_name} complete ===\n')