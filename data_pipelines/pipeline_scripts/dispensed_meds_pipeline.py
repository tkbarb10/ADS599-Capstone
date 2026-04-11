import argparse
from pathlib import Path

from datasets import load_dataset, Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml, get_sql_dir
from utils.logging_helper import setup_logging
from data_pipelines.preprocessing_scripts.admitted_meds_preprocessing import (
    filter_non_admin_events,
    coalesce_charttime,
)
from data_pipelines.feature_engineering.meds_features import (
    map_drug_class,
    add_minutes_into_stay,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

if __name__ == "__main__":
    hf = settings['hugging_face']
    meds_hf = hf['dispensed_meds']
    cohort_hf = hf['cohort']
    time_block = settings['time_block']

    parser = argparse.ArgumentParser(description="Build and push dispensed meds to HuggingFace.")
    parser.add_argument("--src-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--config-name", default=meds_hf['config_name'])
    parser.add_argument("--split-name", default=meds_hf['split_name'])
    parser.add_argument("--data-dir", default=meds_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "admitted_meds.sql").read_text()
    df = query_to_df(client, query)
    logger.info(f"Raw BQ shape: {df.shape}")
    logger.debug(f"in_er value counts:\n{df['in_er'].value_counts()}")

    # 2. Preprocess
    df = filter_non_admin_events(df)
    df = coalesce_charttime(df)

    # 3. Map drug classes
    df = map_drug_class(df)

    # 4. Load cohort from HuggingFace for ed_intime join
    logger.info(f"\nLoading cohort from {args.src_repo_id} / {cohort_hf['config_name']}\n")
    cohort_ds = load_dataset(
        args.src_repo_id,
        name=cohort_hf['config_name'],
        split=cohort_hf['split_name'],
    )
    df_cohort = cohort_ds.to_pandas()
    logger.info(f"\nCohort loaded - {len(df_cohort):,} rows\n")

    # 5. Compute minutes_into_stay and time_step
    df = add_minutes_into_stay(df, df_cohort, time_block)

    logger.info(f"Final shape: {df.shape}")

    # 6. Push to HuggingFace
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"\nPushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")
