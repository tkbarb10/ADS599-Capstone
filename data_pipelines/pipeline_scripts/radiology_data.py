import argparse

from datasets import Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml, get_sql_dir
from utils.logging_helper import setup_logging
from data_pipelines.preprocessing_scripts.radiology_preprocessing import (
    filter_admin_rows,
    resolve_duplicate_exams,
)
from data_pipelines.feature_engineering.radiology_features import assign_rad_acuity

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

if __name__ == "__main__":
    hf = settings['hugging_face']
    rad_hf = hf['radiology']

    parser = argparse.ArgumentParser(description="Build and push radiology acuity features to HuggingFace.")
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--config-name",  default=rad_hf['config_name'])
    parser.add_argument("--split-name",   default=rad_hf['split_name'])
    parser.add_argument("--data-dir",     default=rad_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "radiology_data.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
    df = query_to_df(client, query)
    logger.info(f"Raw BQ shape: {df.shape}")
    logger.info(f"Unique ED stays: {df['ed_stay_id'].nunique():,}")

    # 2. Remove administrative/billing entries
    df = filter_admin_rows(df)

    # 3. Assign acuity level per exam row (CPT lookup then string matching)
    df = assign_rad_acuity(df)

    # 4. Deduplicate to one row per ED stay (highest acuity, earliest time on tie)
    df = resolve_duplicate_exams(df)

    # Rename to final output schema
    df = df.rename(columns={'charttime': 'exam_time', 'exam_acuity': 'rad_acuity_level'})
    df = df[['ed_stay_id', 'subject_id', 'hadm_id', 'exam_time', 'rad_acuity_level']]

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"rad_acuity_level distribution:\n{df['rad_acuity_level'].value_counts().sort_index()}")

    # 5. Push to HuggingFace
    # Note: ED stays absent from this table (no imaging or all admin rows filtered)
    # receive rad_acuity_level=0 when merged onto the cohort.
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")