import argparse

from datasets import load_dataset, Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml, get_sql_dir
from utils.logging_helper import setup_logging
from data_pipelines.preprocessing_scripts.microbiology_preprocessing import filter_storetime
from data_pipelines.feature_engineering.microbiology_features import (
    add_action_space,
    label_culture_results,
    collapse_micro_rows,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

if __name__ == "__main__":
    hf = settings['hugging_face']
    micro_hf = hf['microbiology']
    cohort_hf = hf['cohort']

    parser = argparse.ArgumentParser(description="Build and push microbiology features to HuggingFace.")
    parser.add_argument("--src-repo-id",  default=hf['interim_data_repo'])
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--config-name",  default=micro_hf['config_name'])
    parser.add_argument("--split-name",   default=micro_hf['split_name'])
    parser.add_argument("--data-dir",     default=micro_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    # Microbiology culture events for cohort patients.
    # charttime (culture ordered) is bounded by the SQL stay window.
    # storetime (result returned) requires a separate filter — see step 3.
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "microbiology.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
    df = query_to_df(client, query)
    logger.info(f"Raw BQ shape: {df.shape}")
    logger.info(f"Unique ED stays: {df['ed_stay_id'].nunique():,}")

    # 2. Load cohort for stay window timing (needed for storetime filter)
    logger.info(f"Loading cohort from {args.src_repo_id} / {cohort_hf['config_name']}")
    cohort_ds = load_dataset(
        args.src_repo_id,
        name=cohort_hf['config_name'],
        split=cohort_hf['split_name'],
    )
    df_cohort = cohort_ds.to_pandas()
    logger.info(f"Cohort loaded: {len(df_cohort):,} rows")

    # 3. Filter rows where storetime exceeds stay window end
    df = filter_storetime(df, df_cohort)

    # 4. Map spec_type_desc → action_space (top 20 + OTHER)
    df = add_action_space(df)

    # 5. Classify culture result from org_name / comments
    df = label_culture_results(df)

    # 6. Collapse to one row per (ed_stay_id, charttime, action_space)
    df = collapse_micro_rows(df)

    logger.info(f"Final shape: {df.shape}")

    # 7. Push to HuggingFace
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")