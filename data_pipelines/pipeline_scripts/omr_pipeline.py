import argparse

from datasets import Dataset
from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import load_yaml, get_sql_dir
from utils.logging_helper import setup_logging

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

if __name__ == "__main__":
    hf = settings['hugging_face']
    height_hf = hf['omr_height']
    weight_hf = hf['omr_weight']

    parser = argparse.ArgumentParser(description="Build and push OMR height/weight to HuggingFace.")
    parser.add_argument("--dest-repo-id", default=hf['interim_data_repo'])
    args = parser.parse_args()

    # 1. Load from BigQuery
    # Outpatient measurement records from hosp.omr.
    # Blood pressure excluded (captured in triage and ed.vitalsign).
    # BMI excluded (derived from height and weight). eGFR excluded.
    client, PROJECT_NAME = get_client()
    query = (SQL_DIR / "omr.sql").read_text()
    df = query_to_df(client, query)
    logger.info(f"Raw BQ shape: {df.shape}")
    logger.info(f"Unique subjects: {df['subject_id'].nunique():,}")
    logger.debug(f"result_name value counts:\n{df['result_name'].value_counts()}")

    # 2. Preprocessing
    # Drop rows where result_value is '.' (unparseable placeholder)
    dot_rows = df[df['result_value'] == '.'].index
    df = df.drop(index=dot_rows)
    logger.info(f"Dropped {len(dot_rows):,} unparseable '.' rows")

    # Strip unit suffixes from result_value, then cast to float
    df['result_value'] = df['result_value'].str.strip('>').astype(float)

    # Strip unit labels from result_name so height/weight filter is consistent
    df['result_name'] = (df['result_name']
                         .str.replace(r'\s*\(Lbs\)', '', regex=True)
                         .str.replace(r'\s*\(Inches\)', '', regex=True)
                         .str.strip())

    # 3. Split into height and weight, drop duplicates
    omr_h = df[df['result_name'] == 'Height'].drop_duplicates().reset_index(drop=True)
    omr_w = df[df['result_name'] == 'Weight'].drop_duplicates().reset_index(drop=True)
    logger.info(f"Height rows: {len(omr_h):,} - unique subjects: {omr_h['subject_id'].nunique():,}")
    logger.info(f"Weight rows: {len(omr_w):,} - unique subjects: {omr_w['subject_id'].nunique():,}")

    # 4. Push height to HuggingFace
    ds_h = Dataset.from_pandas(omr_h, preserve_index=False)
    ds_h.push_to_hub(args.dest_repo_id, config_name=height_hf['config_name'],
                     split=height_hf['split_name'], data_dir=height_hf['data_dir'])
    logger.info(f"Pushed height to HuggingFace: {args.dest_repo_id} / {height_hf['config_name']}")

    # 5. Push weight to HuggingFace
    ds_w = Dataset.from_pandas(omr_w, preserve_index=False)
    ds_w.push_to_hub(args.dest_repo_id, config_name=weight_hf['config_name'],
                     split=weight_hf['split_name'], data_dir=weight_hf['data_dir'])
    logger.info(f"Pushed weight to HuggingFace: {args.dest_repo_id} / {weight_hf['config_name']}")