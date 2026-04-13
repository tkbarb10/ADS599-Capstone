"""
modeling/traditional_ml/random_forest.py

Trains a random forest classifier using hyperparameters from
modeling/config/traditional_ml.yaml. Saves the fitted model, scaler,
and held-out sets to artifacts/random_forest/.

Usage:
    python -m modeling.traditional_ml.random_forest
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from modeling.data_prep.traditional_ml import load_and_prep, training_split, scaling

PROJECT_ROOT = Path(__file__).parents[2]

settings = load_yaml('project_setup/settings.yaml')
config = load_yaml('modeling/config/traditional_ml.yaml')
logger = setup_logging(settings['logging']['train_path'])


if __name__ == '__main__':
    data_config = config['data']
    target_col = data_config['target_col']

    # -----------------------------------------------------------------------
    # Load and prep
    # -----------------------------------------------------------------------
    df = load_and_prep(hf_cfg=settings['hugging_face'])
    train, test, validation = training_split(df=df)
    train, test, validation, scaler = scaling(train=train, test=test, validation=validation)
    logger.info(f'train={len(train):,}  test={len(test):,}  validation={len(validation):,}')

    X_train = train.drop(columns=['label', 'ed_stay_id', 'subject_id'], errors='ignore')
    y_train = train['label']
    X_test = test.drop(columns=['label', 'ed_stay_id', 'subject_id'], errors='ignore')
    y_test = test['label']
    X_valid = validation.drop(columns=['label', 'ed_stay_id', 'subject_id'], errors='ignore')
    y_valid = validation['label']

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info('Training RandomForestClassifier...')
    model = RandomForestClassifier(**config['random_forest'])
    model.fit(X_train, y_train)
    logger.info('Training complete')

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifact_dir = PROJECT_ROOT / config['artifacts']['random_forest_dir']
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / f'random_forest_{timestamp}.pkl'
    scaler_path = artifact_dir / f'scaler_{timestamp}.pkl'
    test_path = artifact_dir / f'test_{timestamp}.parquet'
    validation_path = artifact_dir / f'validation_{timestamp}.parquet'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    test.to_parquet(test_path, index=False)
    validation.to_parquet(validation_path, index=False)

    logger.info(f'Model saved    -> {model_path}')
    logger.info(f'Scaler saved   -> {scaler_path}')
    logger.info(f'Test set saved -> {test_path}')
    logger.info(f'Val set saved  -> {validation_path}')