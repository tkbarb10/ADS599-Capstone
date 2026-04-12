import argparse
import pickle
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, load_dataset

from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import get_sql_dir, load_yaml, get_artifacts_dir
from utils.logging_helper import setup_logging
from data_pipelines.preprocessing_scripts.triage_vitals_preprocessing import (
    clean_vital_sign_outliers,
    clean_pain_column,
    remap_stay_ids,
    filter_to_cohort,
    fill_triage_charttimes,
    drop_pre_admission_rows,
    drop_same_time_vitals,
)
from data_pipelines.feature_engineering.triage_vitals_features import (
    add_time_since_last,
    forward_fill_and_flag_missing,
    add_vital_features,
)

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")
SQL_DIR = get_sql_dir()

with open(get_artifacts_dir() / "stay_id_remap.pkl", "rb") as f:
    stay_id_remap = pickle.load(f)

client, PROJECT_NAME = get_client()

triage_query = (SQL_DIR / "triage.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
vitals_query = (SQL_DIR / "vitals.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
df_triage = query_to_df(client, triage_query)
df_vitals = query_to_df(client, vitals_query)

logger.info(f"Triage shape: {df_triage.shape}  (should be ~399,573 rows)")
logger.info(f"Vitals shape: {df_vitals.shape}")
logger.info(f"Unique ED visits with vitals: {df_vitals['ed_stay_id'].nunique():,}")
logger.info(f"Avg vital readings per visit: {len(df_vitals) / df_vitals['ed_stay_id'].nunique():.1f}")
logger.info(f"Vitals rows to remap: {df_vitals['ed_stay_id'].isin(stay_id_remap).sum():,}")

df_triage['source'] = 'triage'
df_vitals['source'] = 'vitals'

combined_df = (
    pd.concat([df_triage, df_vitals], ignore_index=True)
    .sort_values(by=['ed_stay_id', 'source', 'charttime'], ascending=[True, True, True])
    .reset_index(drop=True)
)

if __name__ == "__main__":
    load_dotenv()
    hf = settings['hugging_face']
    tv_hf = hf['triage_vitals']

    parser = argparse.ArgumentParser(description="Build triage+vitals feature dataset and push to HuggingFace.")
    parser.add_argument("--repo-id",     default=hf['interim_data_repo'])
    parser.add_argument("--config-name", default=tv_hf['config_name'])
    parser.add_argument("--split-name",  default=tv_hf['split_name'])
    parser.add_argument("--data-dir",    default=tv_hf['data_dir'])
    args = parser.parse_args()

    # 1. Remap second consecutive stay vitals → canonical ed_stay_id
    df_vitals_remapped = remap_stay_ids(df_vitals, stay_id_remap)
    logger.info(f"Vitals after stay_id remap — unique stays: {df_vitals_remapped['ed_stay_id'].nunique():,}")

    # 2. Load cohort from HuggingFace
    df_cohort = load_dataset(hf['interim_data_repo'], name='cohort_full',
                             split='cohort_base').to_pandas()
    stay_ids = df_cohort['ed_stay_id'].tolist()
    logger.info(f"Cohort size: {len(stay_ids):,} unique ED stays")

    # 3. Rebuild combined with remapped vitals, drop rhythm, filter to cohort
    df_triage_src = df_triage.copy()
    combined = (
        pd.concat([df_triage_src, df_vitals_remapped], ignore_index=True)
        .sort_values(by=['ed_stay_id', 'source', 'charttime'], ascending=[True, True, True])
        .reset_index(drop=True)
        .drop(columns=['rhythm'])
    )
    combined = filter_to_cohort(combined, stay_ids)

    # 4. Fill triage charttimes from cohort ed_intime
    combined = fill_triage_charttimes(combined, df_cohort)

    # 5. Drop vitals recorded before ed_intime
    combined = drop_pre_admission_rows(combined, df_cohort)

    # 6. Compute time_since_last_min (needed before drop_same_time_vitals)
    combined = add_time_since_last(combined)

    # 7. Drop vitals rows at time 0 (duplicate of triage baseline)
    combined = drop_same_time_vitals(combined)

    # 8. Clean pain column, cast to numeric, null outliers
    combined = clean_pain_column(combined)
    numeric_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors='coerce')
    combined = clean_vital_sign_outliers(combined)
    logger.info(f"Post-cleaning NaN counts:\n{combined.isna().sum().sort_values(ascending=False).head(10)}")

    # 9. Forward fill, missing indicators, mean imputation
    combined = forward_fill_and_flag_missing(combined)

    # 10. Feature engineering: deltas, rates, rolling averages, MAP, rename
    combined = add_vital_features(combined)
    logger.info(f"Final shape: {combined.shape}")

    # 11. Push to HuggingFace
    ds = Dataset.from_pandas(combined, preserve_index=False)
    ds.push_to_hub(args.repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"Pushed to HuggingFace: {args.repo_id} / {args.config_name}")
