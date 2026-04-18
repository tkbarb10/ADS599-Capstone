"""
prep_data.py — Run once to generate parquet data files for the Patient Portal app.
Saves 3 parquets to streamlit/ root (parent of this script's directory).

    cd streamlit/app
    python prep_data.py
"""
import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent  # streamlit/ root

PATIENT_REPO = "ADS599-Capstone/interim_data"
DATA_REPO = "ADS599-Capstone/sbs_predictions"

print("Loading HuggingFace datasets — this may take a few minutes...")

patient_df = load_dataset(PATIENT_REPO, name='cohort_full', split='cohort_base').to_pandas()
weight_df = load_dataset(PATIENT_REPO, name='weight', split='weight_full').to_pandas()
height_df = load_dataset(PATIENT_REPO, name='height', split='height_full').to_pandas()
tv_df = load_dataset(PATIENT_REPO, name='triage_vitals', split='triage_vitals_full').to_pandas()
data_df = load_dataset(DATA_REPO, name='sbs_preds', split='sbs_preds_full').to_pandas()

print(f"  cohort: {len(patient_df):,} rows") # type: ignore
print(f"  data_df: {data_df.shape}") # type: ignore

# ── Sample 5k stays (1k ICU, 4k discharge) ───────────────────────────────
rng = np.random.default_rng(10)
stay_meta = data_df.drop_duplicates('ed_stay_id')[['ed_stay_id', 'terminal_event']] # type: ignore
icu_stays = stay_meta[stay_meta['terminal_event'] == 'transfer_icu']['ed_stay_id'].tolist()
dis_stays = stay_meta[stay_meta['terminal_event'] == 'discharge']['ed_stay_id'].tolist()
n_icu = min(1000, len(icu_stays))
n_dis = min(4000, len(dis_stays))
sampled_ids = set(
    rng.choice(icu_stays, n_icu, replace=False).tolist() +
    rng.choice(dis_stays, n_dis, replace=False).tolist()
)
data_df = data_df[data_df['ed_stay_id'].isin(sampled_ids)].copy() # type: ignore
print(f"  Sampled {len(sampled_ids):,} stays ({n_icu:,} ICU, {n_dis:,} discharge)")

with open(OUT_DIR / 'sampled_stay_ids.json', 'w') as f:
    json.dump(sorted(sampled_ids), f)

# ── Derive and save state_cols ────────────────────────────────────────────
dispensed_meds = list(data_df.loc[:, 'ACE Inhibitor':'Other'].columns) if 'ACE Inhibitor' in data_df.columns else [] # type: ignore
recon       = [c for c in data_df.columns if c.startswith('recon_')] # type: ignore
vitals      = [c for c in data_df.columns if c.startswith('current_')] # type: ignore
vital_change = [c for c in data_df.columns if c.endswith(('_rolling1h', '_delta', '_rate_per_min'))] # type: ignore
lab_ohe     = [c for c in data_df.columns if c.endswith(('_Normal', '_Pending', '_Abnormal')) and '-' in c] # type: ignore
micro_ohe   = [c for c in data_df.columns if c.endswith(('_Pending', '_Positive', '_Negative', '_Other')) and '-' not in c and not c.startswith(('ecg_status', 'rad_status'))] # type: ignore
ecg_ohe     = [c for c in data_df.columns if c.startswith('ecg_status')] # type: ignore
rad_ohe     = [c for c in data_df.columns if c.startswith('rad_status')] # type: ignore
arrival     = [c for c in data_df.columns if c.startswith('arrival_')] # type: ignore
missing     = [c for c in data_df.columns if c.endswith('_missing')] # type: ignore

state_cols = (
    ['gender', 'anchor_age', 'acuity', 'height', 'weight', 'time_since_last_min']
    + dispensed_meds + recon + vitals + vital_change
    + lab_ohe + micro_ohe + ecg_ohe + rad_ohe + arrival + missing
)
state_cols = [c for c in state_cols if c in data_df.columns] # type: ignore

with open(OUT_DIR / 'state_cols.json', 'w') as f:
    json.dump(state_cols, f)
print(f"  state_cols.json: {len(state_cols)} features")

eds = set(data_df['ed_stay_id'].unique()) # type: ignore

# ── Height / weight — single value per patient ────────────────────────────
avg_weight = weight_df.groupby('subject_id')['result_value'].mean().rename('weight').reset_index() # type: ignore
avg_height = height_df.groupby('subject_id')['result_value'].mean().rename('height').reset_index() # type: ignore

# ── patient_stats.parquet ─────────────────────────────────────────────────
# One row per ED stay (filtered to modeling data). Used by Tab 1.
cohort_cols = [
    'ed_stay_id', 'subject_id', 'ed_intime',
    'arrival_transport', 'gender', 'anchor_age', 'language',
]

triage_vital_cols = [
    c for c in tv_df.columns # type: ignore
    if c.startswith('current') and c != 'current_mean_arterial_pressure'
]
triage_cols = triage_vital_cols + ['acuity', 'chiefcomplaint', 'ed_stay_id']
triage_baseline = tv_df[tv_df['source'] == 'triage'][triage_cols] # type: ignore

patient_stats = (
    patient_df[cohort_cols] # type: ignore
    .merge(avg_weight, on='subject_id', how='left')
    .merge(avg_height, on='subject_id', how='left')
    .merge(triage_baseline, on='ed_stay_id', how='left')
)
patient_stats = patient_stats[patient_stats['ed_stay_id'].isin(eds)].copy()
patient_stats['ed_intime'] = pd.to_datetime(patient_stats['ed_intime'])

patient_stats.to_parquet(OUT_DIR / 'patient_stats.parquet', index=False)
print(f"  patient_stats.parquet: {len(patient_stats):,} rows, {len(patient_stats.columns)} cols")

# ── patient_probability.parquet ───────────────────────────────────────────
# One row per (ed_stay_id, step_idx). Used by Tab 2 trajectory chart.
prob_cols = [
    'ed_stay_id', 'in_ed', 'in_ward', 'terminal_event',
    'terminal_code', 'step_idx', 'time', 'p_icu',
]
patient_probability = data_df[prob_cols].copy() # type: ignore
patient_probability.to_parquet(OUT_DIR / 'patient_probability.parquet', index=False)
print(f"  patient_probability.parquet: {len(patient_probability):,} rows")

# ── step_features.parquet ─────────────────────────────────────────────────
# One row per (ed_stay_id, step_idx). Used by Tab 2 "what changed" and Tab 3 waterfall.
step_feature_cols = [
    'ed_stay_id', 'step_idx', 'time', 'in_ed', 'in_ward', 'terminal_code',
    # Vitals
    'current_temperature', 'current_heartrate', 'current_resprate',
    'current_o2sat', 'current_sbp', 'current_dbp', 'current_pain', 'current_map',
    # Labs — most clinically common
    'Chemistry-Blood_Pending',      'Chemistry-Blood_Normal',      'Chemistry-Blood_Abnormal',
    'Hematology-Blood_Pending',     'Hematology-Blood_Normal',     'Hematology-Blood_Abnormal',
    'Blood Gas-Blood_Pending',      'Blood Gas-Blood_Normal',      'Blood Gas-Blood_Abnormal',
    'BLOOD CULTURE_Pending',        'BLOOD CULTURE_Negative',      'BLOOD CULTURE_Positive',
    # ECG / Radiology
    'ecg_status_Normal', 'ecg_status_Moderate', 'ecg_status_Acute',
    'rad_status_Normal', 'rad_status_Moderate', 'rad_status_Acute',
    # Medications dispensed during stay
    'Antibiotic', 'IV Fluid', 'Analgesic - Opioid/NSAID', 'Analgesic - Acetaminophen',
    'Antiemetic', 'Anticoagulant', 'Corticosteroid',
    'Benzodiazepine - Sedative/Anxiolytic', 'Beta Blocker', 'Diuretic', 'Bronchodilator',
    # Model output
    'p_icu',
]
available_cols = [c for c in step_feature_cols if c in data_df.columns] # type: ignore
missing_cols   = [c for c in step_feature_cols if c not in data_df.columns] # type: ignore
if missing_cols:
    print(f"  WARNING: columns not found in data_df: {missing_cols}")

step_features = data_df[available_cols].copy() # type: ignore
step_features.to_parquet(OUT_DIR / 'step_features.parquet', index=False)
print(f"  step_features.parquet: {len(step_features):,} rows, {len(available_cols)} cols")

print("\nDone. Parquets saved to:", OUT_DIR)