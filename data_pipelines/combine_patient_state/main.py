"""
combine_patient_state/main.py

Builds the full event-driven patient state DataFrame and pushes it to HuggingFace.

Usage:
    python -m data_pipelines.combine_patient_state.main

Optional flags:
    --force-rebuild   Ignore any existing checkpoint and rebuild from scratch
    --src-repo-id     HuggingFace interim data repo  (default from settings.yaml)
    --dest-repo-id    HuggingFace modeling data repo (default from settings.yaml)
    --config-name     HF config name for output dataset
    --split-name      HF split name for output dataset
    --data-dir        HF data dir for output dataset
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
from datasets import Dataset

from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

from data_pipelines.combine_patient_state.preprocessing.cohort import load_cohort
from data_pipelines.combine_patient_state.preprocessing.step_index import (
    collect_event_times,
    build_step_index,
    drop_out_of_window,
    drop_single_step_stays,
)
from data_pipelines.combine_patient_state.feature_engineering.vitals import (
    snap_vitals,
    compute_time_since_last_vitals,
)
from data_pipelines.combine_patient_state.feature_engineering.labs_ohe import expand_labs_ohe
from data_pipelines.combine_patient_state.feature_engineering.micro_ohe import expand_micro_ohe
from data_pipelines.combine_patient_state.feature_engineering.meds_flags import expand_med_flags
from data_pipelines.combine_patient_state.feature_engineering.imaging import (
    snap_ecg_ohe,
    snap_rad_ohe,
)
from data_pipelines.combine_patient_state.feature_engineering.patient_features import (
    add_location_flags,
    add_height_weight,
    add_medrecon,
    add_derived_features,
)
from data_pipelines.combine_patient_state.validation_checks.checks import run_all_checks

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

settings = load_yaml("project_setup/settings.yaml")
logger   = setup_logging(settings['logging']['patient_combination_path'])

CHECKPOINT = Path(__file__).parent / '_checkpoint.parquet'


def _save_checkpoint(df: pd.DataFrame, step: str) -> None:
    df.to_parquet(CHECKPOINT, compression='snappy', index=False)
    logger.info(f'Checkpoint saved after: {step}')


def _load_checkpoint() -> pd.DataFrame:
    logger.info(f'Loading checkpoint from {CHECKPOINT}')
    return pd.read_parquet(CHECKPOINT)


def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    hf = settings['hugging_face']
    out_hf = hf['full_patient_state']

    parser = argparse.ArgumentParser(
        description='Build and push full patient state to HuggingFace.'
    )
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Ignore checkpoint and rebuild from scratch')
    parser.add_argument('--src-repo-id', default=hf['interim_data_repo'])
    parser.add_argument('--dest-repo-id', default=hf['modeling_data_repo'])
    parser.add_argument('--config-name', default=out_hf['config_name'])
    parser.add_argument('--split-name', default=out_hf['split_name'])
    parser.add_argument('--data-dir', default=out_hf['data_dir'])
    args = parser.parse_args()

    SRC = args.src_repo_id
    DEST = args.dest_repo_id

    # Determine resume point
    if CHECKPOINT.exists() and not args.force_rebuild:
        df_patient = _load_checkpoint()
        logger.info(f'Resuming -- loaded checkpoint shape: {df_patient.shape}')
    else:
        df_patient = None

    # -----------------------------------------------------------------------
    # Step 1: Load cohort (always required -- provides labels and stay window)
    # -----------------------------------------------------------------------
    logger.info('Step 1: Loading cohort...')
    cohort = load_cohort(SRC, hf)
    logger.info(f'Cohort loaded -- {len(cohort):,} rows, '
                f'terminal_event distribution:\n{cohort["terminal_event"].value_counts().to_string()}')

    # -----------------------------------------------------------------------
    # Step 2: Build step index
    # -----------------------------------------------------------------------
    if df_patient is None or not _has_col(df_patient, 'step_idx'):
        logger.info('Step 2: Collecting event timestamps and building step index...')
        frames = collect_event_times(SRC, hf)
        df_patient = build_step_index(frames, cohort)
        df_patient = drop_out_of_window(df_patient)
        df_patient = drop_single_step_stays(df_patient)
        _save_checkpoint(df_patient, 'step_index')
        gc.collect()
    else:
        logger.info('Step 2: Skipping (checkpoint contains step_idx)')

    # Pre-compute dp_sorted and stay_last_idx once -- reused by labs/micro/meds/imaging
    # Sorting just by time is accurate here.  pd.merge_asof in later steps requires the merge key to be monotonically increasing.  Merging by ed_stay_id ensures that the 
    # events get linked to the right stay
    dp_sorted = df_patient[['ed_stay_id', 'step_idx', 'time']].sort_values('time')
    stay_last_idx = (
        df_patient.groupby('ed_stay_id')['step_idx']
        .max()
        .reset_index(name='last_idx')
    )

    # -----------------------------------------------------------------------
    # Step 3: Triage / vitals
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'vitals_checked'):
        logger.info('Step 3: Snapping vitals...')
        df_patient = snap_vitals(df_patient, SRC, hf)
        df_patient = compute_time_since_last_vitals(df_patient)
        _save_checkpoint(df_patient, 'vitals')
        gc.collect()
    else:
        logger.info('Step 3: Skipping (vitals_checked already present)')

    # -----------------------------------------------------------------------
    # Step 4: Labs OHE
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'labs_ordered'):
        logger.info('Step 4: Expanding labs OHE...')
        df_patient, lab_actions = expand_labs_ohe(df_patient, dp_sorted, stay_last_idx, SRC, hf)
        _save_checkpoint(df_patient, 'labs_ohe')
        gc.collect()
    else:
        # Recover lab_actions from existing columns if resuming past labs step.
        # Lab action names always contain a hyphen (e.g. Chemistry-Blood).
        # ecg_status_Normal and rad_status_Normal are excluded by the hyphen check.
        lab_actions = sorted({
            c[:-len('_Normal')]
            for c in df_patient.columns if c.endswith('_Normal') and '-' in c
        })
        logger.info(f'Step 4: Skipping (labs_ordered present; {len(lab_actions)} actions recovered)')

    # -----------------------------------------------------------------------
    # Step 5: Micro OHE
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'micro_ordered'):
        logger.info('Step 5: Expanding microbiology OHE...')
        df_patient, micro_actions = expand_micro_ohe(df_patient, dp_sorted, stay_last_idx, SRC, hf)
        _save_checkpoint(df_patient, 'micro_ohe')
        gc.collect()
    else:
        micro_actions = sorted({
            c.rsplit('_', 1)[0]
            for c in df_patient.columns if c.endswith('_Negative')
        })
        logger.info(f'Step 5: Skipping (micro_ordered present; {len(micro_actions)} actions recovered)')

    # -----------------------------------------------------------------------
    # Step 6: Medications
    # -----------------------------------------------------------------------
    # meds_expanded is a sentinel column added after expand_med_flags() so the
    # resume check doesn't depend on any specific drug class name.
    if not _has_col(df_patient, 'meds_expanded'):
        logger.info('Step 6: Expanding medication flags...')
        df_patient = expand_med_flags(df_patient, dp_sorted, stay_last_idx, SRC, hf)
        df_patient['meds_expanded'] = 1
        _save_checkpoint(df_patient, 'meds')
        gc.collect()
    else:
        logger.info('Step 6: Skipping (meds_expanded sentinel present)')

    # -----------------------------------------------------------------------
    # Step 7: ECG + Radiology
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'ecg_status_Normal'):
        logger.info('Step 7: Snapping ECG and radiology OHE...')
        df_patient = snap_ecg_ohe(df_patient, dp_sorted, SRC, hf)
        df_patient = snap_rad_ohe(df_patient, dp_sorted, SRC, hf)
        _save_checkpoint(df_patient, 'imaging')
        gc.collect()
    else:
        logger.info('Step 7: Skipping (ecg_status_Normal already present)')

    # -----------------------------------------------------------------------
    # Step 8: Location flags, height/weight, medrecon
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'in_ed'):
        logger.info('Step 8: Adding location flags, height/weight, and medrecon...')
        df_patient = add_location_flags(df_patient)
        df_patient = add_height_weight(df_patient, SRC, hf)
        df_patient = add_medrecon(df_patient, SRC, hf)
        _save_checkpoint(df_patient, 'location_static')
        gc.collect()
    else:
        logger.info('Step 8: Skipping (in_ed already present)')

    # -----------------------------------------------------------------------
    # Step 9: Derived features (arrival OHE, acuity, total_length, drop cols)
    # -----------------------------------------------------------------------
    if not _has_col(df_patient, 'total_length'):
        logger.info('Step 9: Adding derived features...')
        df_patient = add_derived_features(df_patient)
        _save_checkpoint(df_patient, 'derived_features')
        gc.collect()
    else:
        logger.info('Step 9: Skipping (total_length already present)')

    # -----------------------------------------------------------------------
    # Step 10: Final validation
    # -----------------------------------------------------------------------
    logger.info('Step 10: Running final validation checks...')
    df_patient = run_all_checks(df_patient, cohort, lab_actions, micro_actions)

    # -----------------------------------------------------------------------
    # Step 11: Push to HuggingFace
    # -----------------------------------------------------------------------
    # Write to a temp parquet first, then load with Dataset.from_parquet.
    # Dataset.from_pandas fingerprints the entire table via dill serialization,
    # which blows RAM on a 7M row x 256 col DataFrame.  from_parquet memory-maps
    # the file instead, keeping peak memory low.  Annoying, but better than not being
    # able to save it
    logger.info(f'Step 11: Pushing to {DEST} / {args.config_name}...')
    _push_tmp = CHECKPOINT.parent / '_push_tmp.parquet'
    try:
        df_patient.to_parquet(_push_tmp, compression='snappy', index=False)
        del df_patient
        ds = Dataset.from_parquet(str(_push_tmp))
        ds.push_to_hub(
            DEST,
            config_name=args.config_name,
            split=args.split_name,
            data_dir=args.data_dir,
        )
        logger.info(f'Pushed to {DEST} / config={args.config_name} / split={args.split_name}  '
                    f'-- rows: {len(ds):,}  columns: {len(ds.column_names)}')
    finally:
        if _push_tmp.exists():
            _push_tmp.unlink()

    # Clean up checkpoint on success
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        logger.info('Checkpoint deleted after successful push.')
