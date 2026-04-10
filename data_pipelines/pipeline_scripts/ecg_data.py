import argparse
from pathlib import Path

from datasets import Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from data_pipelines.feature_engineering.ecg_features import (
    score_ecg_acuity,
    resolve_duplicate_ecgs,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = Path(__file__).parent.parent.parent / "sql_scripts"

if __name__ == "__main__":
    hf = settings['hugging_face']
    ecg_hf = hf['ecg']

    parser = argparse.ArgumentParser(description="Build and push ECG acuity features to HuggingFace.")
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--config-name",  default=ecg_hf['config_name'])
    parser.add_argument("--split-name",   default=ecg_hf['split_name'])
    parser.add_argument("--data-dir",     default=ecg_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "ecg_data.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
    df = query_to_df(client, query)
    logger.info(f"Raw BQ shape: {df.shape}")
    logger.info(f"Unique ED stays: {df['ed_stay_id'].nunique():,}")

    # 2. Score ECG acuity across 18 report columns
    df = score_ecg_acuity(df)

    # 3. Resolve to one ECG per ED stay (highest acuity, earliest time on tie)
    df = resolve_duplicate_ecgs(df)

    logger.info(f"Final shape: {df.shape}")

    # 4. Push to HuggingFace
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")