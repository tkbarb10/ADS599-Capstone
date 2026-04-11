import argparse
import pandas as pd
from datasets import Dataset
from utils.bq import get_client, query_to_df
from utils.logging_helper import setup_logging
from utils.load_yaml_helper import load_yaml, get_sql_dir
from data_pipelines.feature_engineering.lab_features import collapse_lab_rows, actions_column

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

client, PROJECT_NAME = get_client()

if __name__ == "__main__":
    hf_defaults = settings['hugging_face']
    lab_settings = hf_defaults['labs']

    parser = argparse.ArgumentParser(description="Build and push the labs dataset for this project to a Hugging Face data repo")
    parser.add_argument("--repo-id", default=hf_defaults['interim_data_repo'])
    parser.add_argument("--config-name", default=lab_settings['config_name'])
    parser.add_argument("--split-name", default=lab_settings['split_name'])
    parser.add_argument("--data-dir", default=lab_settings['data_dir'])
    args = parser.parse_args()

    logger.info("Running labs query (may take several minutes)...")
    query = (SQL_DIR / "labs.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
    df_labs = query_to_df(client, query)
    logger.info(f"Shape: {df_labs.shape}")
    logger.info(f"Unique ED visits with labs: {df_labs['ed_stay_id'].nunique():,}")
    logger.info(f"Ordered location breakdown:\n\n{df_labs['ordered_location'].value_counts()}")

    # Creates variable of how many lab events in the pulled dataframe returned results after patient was transferred to ICU
    results_after_transfer = df_labs['result_after_icu_transfer'].sum()
    logger.info(f"\nResult after ICU transfer: {results_after_transfer:,} rows")
    rows_remaining = df_labs.shape[0] - results_after_transfer

    # Remove items where lab results came back after the patient had been transferred
    mask = df_labs[df_labs['result_after_icu_transfer'] == False].index
    df_labs = df_labs.loc[mask, :]

    logger.info(f"After removal, remaining rows should equal {rows_remaining}.  Rows Remaining: {df_labs.shape[0]}")

    df_labs = collapse_lab_rows(df_labs)
    df_labs = actions_column(df_labs)
    logger.info(f"There should be 19 unique actions created now.  Actions: {df_labs['action'].nunique()}")

    # Change times to datetimes
    df_labs['order_time'] = pd.to_datetime(df_labs['order_time'])
    df_labs['result_time'] = pd.to_datetime(df_labs['result_time'])

    ds = Dataset.from_pandas(df_labs)
    ds.push_to_hub(repo_id=args.repo_id, config_name=args.config_name, split=args.split_name, data_dir=args.data_dir)
    logger.info(f"\nSuccessfully pushed the labs dataframe to Hugging Face: {args.repo_id}/{args.data_dir}")
