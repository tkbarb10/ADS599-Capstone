# ============================================================
# COHORT IDENTIFICATION: ED VISITS FOR RL PROJECT
# ADS599 Capstone - MIMIC-IV
# ============================================================
# This script builds the base patient cohort and saves it to BigQuery
# as in this format: project_id.dataset_id.table_id. All feature queries can
# reference it without repeating the full CTE chain.
#
# Run this script first before any other pipeline notebooks.
#
# Pathway Labels:
#   ED_DIRECT_ICU: ED → ICU (no intermediate ward)
#   ED_WARD_ICU: ED → Ward → ICU (delayed escalation)
#   ED_WARD_DISCHARGE: ED → Ward → Discharged (no ICU)
#   ED_DISCHARGE_STABLE: ED → Home, no ICU return or death within 72h
#   ED_DISCHARGE_RETURN_ICU: ED → Home → ED → ICU within 72h
#   ED_DISCHARGE_DIED_72H: ED → Home → Died within 72h
#
# Inclusion: disposition IN ('HOME', 'ADMITTED') only.
# Excluded: TRANSFER (no follow-up data), EXPIRED (died in ED, out of scope),
#           LEFT WITHOUT BEING SEEN / LEFT AGAINST MEDICAL ADVICE / ELOPED
#           (patient-driven departure, not a provider disposition decision).
# ============================================================

import argparse
import warnings
from pathlib import Path

from utils.logging_helper import setup_logging
from utils.load_yaml_helper import load_yaml
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from dotenv import load_dotenv
import os

warnings.filterwarnings(action='ignore', message="Your application has authenticated using end user")

load_dotenv()

PROJECT_NAME = os.environ.get('PROJECT_NAME')
if not PROJECT_NAME:
    raise ValueError("Need to set PROJECT_NAME env variable to initiate big query client")

client = bigquery.Client(project=PROJECT_NAME)

SQL_DIR = Path(__file__).parent.parent / "sql_scripts"
query = (SQL_DIR / "cohort_base.sql").read_text()

logger = setup_logging()

def run(dataset_name: str, table_name: str) -> None:
    """Build the base cohort and save it to BigQuery.

    Args:
        dataset_name: BQ dataset to write into (e.g. 'rl_project')
        table_name:   BQ table name (e.g. 'cohort_base')
    """
    destination = f"{PROJECT_NAME}.{dataset_name}.{table_name}"

    # Create dataset if needed
    dataset_ref = client.dataset(dataset_name, project=PROJECT_NAME)
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_name} already exists")
    except NotFound:
        from google.cloud.bigquery import Dataset
        ds = Dataset(dataset_ref)
        ds.location = "US"
        client.create_dataset(ds)
        logger.info(f"Created dataset: {PROJECT_NAME}.{dataset_name}")

    # Write COHORT_DESTINATION to .env only if not already set
    env_path = Path(__file__).parent.parent / ".env"
    key = "COHORT_DESTINATION"
    existing = env_path.read_text() if env_path.exists() else ""
    if f"{key}=" not in existing:
        with open(env_path, 'a') as f:
            f.write(f"\n{key}={destination}\n")
        logger.info(f"Written to .env: {key}={destination}")
    else:
        logger.info(f".env already contains {key}, skipping write")

    # Save cohort to BigQuery
    job_config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition="WRITE_TRUNCATE"
    )
    client.query(query, job_config=job_config).result()
    logger.info(f"Cohort saved to BigQuery: {destination}")

    # Validation
    df_cohort = client.query(f"SELECT * FROM `{destination}`").to_dataframe()
    unknown = (df_cohort['cohort_label'] == 'UNKNOWN').sum()
    logger.info(
        f"Shape: {df_cohort.shape}\n"
        f"Null race count: {df_cohort['race'].isna().sum()}  (should be 0)\n"
        f"Total ED visits: {len(df_cohort):,}\n"
        f"--- Cohort Label Distribution ---\n"
        f"{df_cohort['cohort_label'].value_counts().to_string()}\n"
        f"UNKNOWN labels: {unknown}  (should be 0)\n"
        f"Cohort base created and saved successfully."
    )


if __name__ == "__main__":
    settings = load_yaml("project_setup/settings.yaml")
    defaults = settings["cohort_base"]

    parser = argparse.ArgumentParser(description="Build the cohort_base table in BigQuery.")
    parser.add_argument("--dataset", default=defaults["dataset"],
                        help=f"BigQuery dataset name (default: {defaults['dataset']})")
    parser.add_argument("--table",   default=defaults["table"],
                        help=f"BigQuery table name (default: {defaults['table']})")
    args = parser.parse_args()

    run(dataset_name=args.dataset, table_name=args.table)
