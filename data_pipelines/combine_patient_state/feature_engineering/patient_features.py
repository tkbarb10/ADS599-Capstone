import gc
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Columns dropped before the final push (non-predictive identifiers / raw timestamps / sentinels)
DROP_COLS = [
    'ed_intime', 'ed_outtime', 'admittime', 'dischtime',
    'source', 'chiefcomplaint_missing',
    'meds_expanded',   # checkpoint sentinel added in main.py
]


def add_location_flags(df_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by (ed_stay_id, time), then add:
      in_ed        -- 1 while time <= ed_outtime
      in_ward      -- 1 from admittime onward (admitted patients only)
      terminal_code -- 1 on the last row of each stay
    """
    df_patient = df_patient.sort_values(['ed_stay_id', 'time']).reset_index(drop=True)

    df_patient['in_ed'] = (df_patient['time'] <= df_patient['ed_outtime']).astype(int)

    df_patient['in_ward'] = 0
    admitted = df_patient['admittime'].notna()
    df_patient.loc[admitted & (df_patient['time'] >= df_patient['admittime']), 'in_ward'] = 1

    last_mask = ~df_patient.duplicated(subset='ed_stay_id', keep='last')
    df_patient['terminal_code'] = 0
    df_patient.loc[last_mask, 'terminal_code'] = 1

    logger.info(f'in_ed=1: {df_patient["in_ed"].sum():,}  '
                f'in_ward=1: {df_patient["in_ward"].sum():,}  '
                f'terminal_code=1: {df_patient["terminal_code"].sum():,}')
    return df_patient


def add_height_weight(df_patient: pd.DataFrame, src_repo: str, hf_cfg: dict) -> pd.DataFrame:
    """Merge per-subject mean height and weight (static, subject-level averages)."""
    weight_df = load_dataset(src_repo, name=hf_cfg['omr_weight']['config_name'],
                             split=hf_cfg['omr_weight']['split_name']).to_pandas()
    height_df = load_dataset(src_repo, name=hf_cfg['omr_height']['config_name'],
                             split=hf_cfg['omr_height']['split_name']).to_pandas()

    avg_weight = weight_df.groupby('subject_id')['result_value'].mean().rename('weight').reset_index()
    avg_height = height_df.groupby('subject_id')['result_value'].mean().rename('height').reset_index()

    df_patient = df_patient.merge(avg_weight, on='subject_id', how='left')
    df_patient = df_patient.merge(avg_height, on='subject_id', how='left')

    del weight_df, height_df, avg_weight, avg_height; gc.collect()
    logger.info(f'Weight null rate: {df_patient["weight"].isna().mean():.1%}  '
                f'Height null rate: {df_patient["height"].isna().mean():.1%}')
    return df_patient


def add_medrecon(df_patient: pd.DataFrame, src_repo: str, hf_cfg: dict) -> pd.DataFrame:
    """
    Merge pre-arrival medrecon pivot (static, stay-level) onto df_patient.
    recon_* columns are broadcast to all rows of each stay.
    Missing stays receive all-zero flags.
    """
    df_recon = load_dataset(src_repo, name=hf_cfg['medrecon']['config_name'],
                            split=hf_cfg['medrecon']['split_name']).to_pandas()
    recon_cols = [c for c in df_recon.columns if c.startswith('recon_')]
    df_patient = df_patient.merge(
        df_recon[['ed_stay_id'] + recon_cols], on='ed_stay_id', how='left'
    )
    df_patient[recon_cols] = df_patient[recon_cols].fillna(0).astype(int)

    del df_recon; gc.collect()
    logger.info(f'Medrecon merged -- {len(recon_cols)} recon columns added.')
    return df_patient


def add_derived_features(df_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Final feature engineering pass:
      - height_missing / weight_missing flags + fill nulls with 0
      - OHE arrival_transport
      - Forward-fill acuity
      - total_length (days from stay_window_start to stay_window_end)
      - Drop non-predictive columns
    """
    # Height / weight missing indicators
    df_patient['height_missing'] = df_patient['height'].isna().astype(int)
    df_patient['weight_missing'] = df_patient['weight'].isna().astype(int)
    df_patient[['height', 'weight']] = df_patient[['height', 'weight']].fillna(0)

    # OHE arrival_transport
    arrival_dummies = pd.get_dummies(
        df_patient['arrival_transport'], prefix='arrival_transport', dtype=int
    )
    df_patient = pd.concat([df_patient, arrival_dummies], axis=1)
    df_patient.drop(columns='arrival_transport', inplace=True)

    # Forward-fill acuity (triage value carries forward)
    df_patient['acuity'] = df_patient['acuity'].ffill()

    # Total stay length in days
    df_patient['total_length'] = (
        (df_patient['stay_window_end'] - df_patient['stay_window_start']).dt.days
    )

    # Drop non-predictive raw columns
    cols_to_drop = [c for c in DROP_COLS if c in df_patient.columns]
    df_patient.drop(columns=cols_to_drop, inplace=True)

    # Sanitize float64 columns that are integer-valued (e.g. age, acuity, step counts
    # stored as float64 because of upstream NaN promotion).  Cast to int64 so Arrow
    # maps them to clean primitive types -- prevents PyYAML from serializing pandas
    # dtype objects into the HuggingFace dataset card with Python-specific YAML tags
    # (!!python/object:...) that HF's validator rejects.
    for col in df_patient.select_dtypes(include='float64').columns:
        non_null = df_patient[col].dropna()
        if len(non_null) > 0 and (non_null % 1 == 0).all():
            df_patient[col] = df_patient[col].fillna(0).astype('int64')

    logger.info(f'Derived features added -- final shape: {df_patient.shape}')
    return df_patient
