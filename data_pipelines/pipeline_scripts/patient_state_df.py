import argparse

from datasets import load_dataset, Dataset
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
import pandas as pd

logger = setup_logging()
settings = load_yaml("project_setup/settings.yaml")

def setup_empty_patient_state(df: pd.DataFrame) -> pd.DataFrame:
    # Sort before expanding so each stay's rows are contiguous after repeat
    new_df = (
        df[['ed_stay_id', 'subject_id', 'hadm_id', 'time_steps']]
        .sort_values(by=['subject_id', 'ed_stay_id', 'hadm_id'])
        .reset_index(drop=True)
    )

    new_df = new_df.loc[new_df.index.repeat(new_df['time_steps'])].reset_index(drop=True)

    row_counts = new_df.groupby('ed_stay_id').size().reset_index(name='row_count')
    row_counts = row_counts.merge(df[['ed_stay_id', 'time_steps']].drop_duplicates(), on='ed_stay_id')
    mismatched = row_counts[row_counts['row_count'] != row_counts['time_steps']]
    logger.info(f"\nMismatched stays: {len(mismatched)}\n")

    return new_df


if __name__ == "__main__":
    hf_defaults = settings['hugging_face']
    cohort_hf = hf_defaults['cohort']
    state_hf = hf_defaults['empty_patient_state']

    parser = argparse.ArgumentParser(description="Build and push empty patient state to HuggingFace.")
    parser.add_argument("--src-repo-id", default=hf_defaults['interim_data_repo'])
    parser.add_argument("--dest-repo-id", default=hf_defaults['interim_data_repo'])
    parser.add_argument("--config-name", default=state_hf['config_name'])
    parser.add_argument("--split-name", default=state_hf['split_name'])
    parser.add_argument("--data-dir", default=state_hf['data_dir'])
    args = parser.parse_args()

    # 1. Load cohort from HuggingFace
    logger.info(f"\nLoading cohort from {args.src_repo_id} / {cohort_hf['config_name']}\n")
    cohort_ds = load_dataset(
        args.src_repo_id,
        name=cohort_hf['config_name'],
        split=cohort_hf['split_name'],
    )
    df_cohort = cohort_ds.to_pandas()
    logger.info(f"\nCohort loaded - shape: {df_cohort.shape}\n")

    # 2. Build empty patient state
    df_state = setup_empty_patient_state(df_cohort)
    logger.info(f"\nEmpty patient state shape: {df_state.shape}\n")

    # 3. Push to HuggingFace
    ds = Dataset.from_pandas(df_state, preserve_index=False)
    ds.push_to_hub(args.dest_repo_id, config_name=args.config_name,
                   split=args.split_name, data_dir=args.data_dir)
    logger.info(f"\nPushed to HuggingFace: {args.dest_repo_id} / {args.config_name}")