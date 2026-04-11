import argparse

from datasets import Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml, get_sql_dir
from utils.logging_helper import setup_logging
from data_pipelines.feature_engineering.medrecon_features import (
    map_etcdesc_to_class,
    build_recon_pivot,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

if __name__ == "__main__":
    hf = settings['hugging_face']
    recon_hf = hf['medrecon']

    parser = argparse.ArgumentParser(description="Build and push medrecon features to HuggingFace.")
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    parser.add_argument("--config-name",  default=recon_hf['config_name'])
    parser.add_argument("--split-name",   default=recon_hf['split_name'])
    parser.add_argument("--data-dir",     default=recon_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    # Pre-arrival medication reconciliation from ed.medrecon.
    # Medications the patient was already taking before the ED visit.
    # Treated as static/baseline state — not time-varying actions.
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "medrecon.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
    df = query_to_df(client, query)
    logger.info(f"Shape before dedup: {df.shape}")
    logger.info(f"Unique ED visits: {df['ed_stay_id'].nunique():,}")

    # 2. Deduplicate
    # Same patient + stay + drug identity can appear multiple times
    # due to multiple GSN ontology group mappings. Collapse to one row per drug per visit.
    dupes = df.duplicated(subset=['subject_id', 'ed_stay_id', 'name', 'gsn', 'ndc', 'etccode']).sum()
    df = df.drop_duplicates(subset=['subject_id', 'ed_stay_id', 'name', 'gsn', 'ndc', 'etccode'],
                            ignore_index=True)
    logger.info(f"Dropped {dupes:,} duplicate rows — shape after dedup: {df.shape}")

    # 3. Map etcdescription → shared 22-class vocabulary
    df = map_etcdesc_to_class(df)

    # 4. Build binary feature pivot (one row per ed_stay_id)
    # etcdescription nulls (0.39%) fall through to 'Other' in map_etcdesc_to_class.
    # Visits with no medrecon record receive all-zero recon_* flags when merged
    # onto the cohort in the combined state pipeline.
    df_features = build_recon_pivot(df)

    logger.info(f"Final shape: {df_features.shape}")

    # 5. Push to HuggingFace
    ds = Dataset.from_pandas(df_features, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")
