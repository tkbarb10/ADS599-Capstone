"""
modeling/traditional_ml/predict.py

Loads the latest trained model from artifacts, runs predictions on the
saved test and validation sets, and saves results as parquet files.

Usage:
    python -m modeling.traditional_ml.predict --model log_reg
    python -m modeling.traditional_ml.predict --model random_forest
    python -m modeling.traditional_ml.predict --model xgboost
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml('project_setup/settings.yaml')
config = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['predict_path'])

PROJECT_ROOT = Path(__file__).parents[2]

DROP_COLS = ['label', 'ed_stay_id', 'subject_id']

MODEL_PREFIX = {
    'log_reg': 'log_reg',
    'random_forest': 'random_forest',
    'xgboost': 'xgboost',
}


def _latest(directory: Path, pattern: str) -> Path:
    """Return the most recently modified file matching pattern in directory."""
    matches = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'No files matching {pattern!r} in {directory}')
    return matches[-1]


def _predict(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP_COLS, errors='ignore')
    out = df.copy()
    out['pred_label'] = model.predict(X)
    out['pred_prob'] = model.predict_proba(X)[:, 1]
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run predictions from a trained traditional ML model.')
    parser.add_argument('--model', required=True, choices=list(MODEL_PREFIX),
                        help='Model to use: log_reg | random_forest | xgboost')
    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / config['artifacts'][f'{args.model}_dir']

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    prefix = MODEL_PREFIX[args.model]
    model_path = _latest(artifact_dir, f'{prefix}_*.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f'Loaded model: {model_path}')

    # -----------------------------------------------------------------------
    # Load held-out sets (already scaled / TF-IDF encoded from training run)
    # -----------------------------------------------------------------------
    test_path = _latest(artifact_dir, 'test_[0-9]*.parquet')
    val_path = _latest(artifact_dir, 'validation_[0-9]*.parquet')
    test = pd.read_parquet(test_path)
    validation = pd.read_parquet(val_path)
    logger.info(f'Loaded test ({len(test):,}) from {test_path.name}')
    logger.info(f'Loaded validation ({len(validation):,}) from {val_path.name}')

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------
    test_preds = _predict(model, test)
    val_preds = _predict(model, validation)
    logger.info(
        f'Test -- predicted {test_preds["pred_label"].sum():,} positive '
        f'({test_preds["pred_label"].mean():.1%})'
    )
    logger.info(
        f'Validation -- predicted {val_preds["pred_label"].sum():,} positive '
        f'({val_preds["pred_label"].mean():.1%})'
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_out = artifact_dir / f'test_predictions_{timestamp}.parquet'
    val_out = artifact_dir / f'validation_predictions_{timestamp}.parquet'

    test_preds.to_parquet(test_out, index=False)
    val_preds.to_parquet(val_out, index=False)

    logger.info(f'Test predictions saved -> {test_out}')
    logger.info(f'Validation predictions saved -> {val_out}')