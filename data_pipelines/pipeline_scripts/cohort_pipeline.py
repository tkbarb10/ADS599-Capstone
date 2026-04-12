import argparse
import os
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

from utils.bq import get_client, query_to_df
from utils.logging_helper import setup_logging
from utils.load_yaml_helper import load_yaml
from data_pipelines.preprocessing_scripts.cohort_preprocessing import (
    fix_label_mismatches,
    merge_duplicate_hadm_id,
    save_stay_id_remap,
    simplify_race_column,
    fix_mismatched_times,
)
from data_pipelines.feature_engineering.cohort_df_features import (
    add_ed_boarding_time,
    create_time_step_col,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")

if __name__ == "__main__":
    hf_defaults = settings['hugging_face']
    cohort_hf = hf_defaults['cohort']
    parser = argparse.ArgumentParser(description="Build and push processed cohort to HuggingFace.")
    parser.add_argument("--repo-id", default=hf_defaults['interim_data_repo'])
    parser.add_argument("--config-name", default=cohort_hf['config_name'])
    parser.add_argument("--split-name", default=cohort_hf['split_name'])
    parser.add_argument("--data-dir", default=cohort_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    load_dotenv()
    DESTINATION = os.environ.get("COHORT_DESTINATION")
    client, PROJECT_NAME = get_client()
    df = query_to_df(client, f"SELECT * FROM `{DESTINATION}`")
    logger.info(f"Loaded from BQ - shape: {df.shape}")
    logger.debug(f"Cohort label distribution (raw):\n{df['cohort_label'].value_counts()}")

    # 2. Fix label mismatches
    df = fix_label_mismatches(df)

    # 3. Remove AGAINST ADVICE rows (already filtered in SQL but guard here)
    before = len(df)
    df = df[df['discharge_location'] != 'AGAINST ADVICE'].copy()
    logger.info(f"Dropped {before - len(df):,} AGAINST ADVICE rows - remaining: {len(df):,}")

    # 4. Collapse consecutive ED visits sharing a hadm_id
    df = merge_duplicate_hadm_id(df)

    # 4a. Save stay_id_remap for downstream pipelines (e.g. vitals)
    save_stay_id_remap(df)

    # 5. Simplify race column
    df = simplify_race_column(df)

    # 6. Add stay window + boarding time columns
    df = add_ed_boarding_time(df)

    # 7. Fix stay windows where start > end
    df = fix_mismatched_times(df)

    # 8. Add time_steps column
    time_block = settings['time_block']
    df = create_time_step_col(df, time_block)

    logger.info(f"Final cohort shape: {df.shape}")
    logger.debug(f"Cohort label distribution (final):\n{df['cohort_label'].value_counts()}")

    # 9. Push to HuggingFace
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.repo_id} / {args.config_name}")
