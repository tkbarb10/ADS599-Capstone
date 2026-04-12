"""
prep_data.py — Run once to generate parquet data files for the Patient Portal app.
Saves 3 parquets to streamlit/ root (parent of this script's directory).

    cd streamlit/app
    python prep_data.py
"""
import pandas as pd
from datasets import load_dataset
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent  # streamlit/ root

PATIENT_REPO = "ADS599-Capstone/interim_data"
DATA_REPO    = "ADS599-Capstone/confidence_delta_data"

print("Loading HuggingFace datasets — this may take a few minutes...")

patient_df = load_dataset(PATIENT_REPO, name='cohort_full',      split='cohort_base').to_pandas()
weight_df  = load_dataset(PATIENT_REPO, name='weight',           split='weight_full').to_pandas()
height_df  = load_dataset(PATIENT_REPO, name='height',           split='height_full').to_pandas()
tv_df      = load_dataset(PATIENT_REPO, name='triage_vitals',    split='triage_vitals_full').to_pandas()
data_df    = load_dataset(DATA_REPO,    name='default',          split='rl_train').to_pandas()

print(f"  cohort: {len(patient_df):,} rows")
print(f"  data_df: {data_df.shape}")

eds = set(data_df['ed_stay_id'].unique())

# ── Height / weight — single value per patient ────────────────────────────
avg_weight = weight_df.groupby('subject_id')['result_value'].mean().rename('weight').reset_index()
avg_height = height_df.groupby('subject_id')['result_value'].mean().rename('height').reset_index()

# ── patient_stats.parquet ─────────────────────────────────────────────────
# One row per ED stay (filtered to modeling data). Used by Tab 1.
cohort_cols = [
    'ed_stay_id', 'subject_id', 'ed_intime',
    'arrival_transport', 'gender', 'anchor_age', 'language',
]

triage_vital_cols = [
    c for c in tv_df.columns
    if c.startswith('current') and c != 'current_mean_arterial_pressure'
]
triage_cols = triage_vital_cols + ['acuity', 'chiefcomplaint', 'ed_stay_id']
triage_baseline = tv_df[tv_df['source'] == 'triage'][triage_cols]

patient_stats = (
    patient_df[cohort_cols]
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
    'terminal_code', 'step_idx', 'p_icu',
]
patient_probability = data_df[prob_cols].copy()
patient_probability.to_parquet(OUT_DIR / 'patient_probability.parquet', index=False)
print(f"  patient_probability.parquet: {len(patient_probability):,} rows")

# ── step_features.parquet ─────────────────────────────────────────────────
# One row per (ed_stay_id, step_idx). Used by Tab 2 "what changed" and Tab 3 waterfall.
step_feature_cols = [
    'ed_stay_id', 'step_idx', 'in_ed', 'in_ward', 'terminal_code',
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
available_cols = [c for c in step_feature_cols if c in data_df.columns]
missing_cols   = [c for c in step_feature_cols if c not in data_df.columns]
if missing_cols:
    print(f"  WARNING: columns not found in data_df: {missing_cols}")

step_features = data_df[available_cols].copy()
step_features.to_parquet(OUT_DIR / 'step_features.parquet', index=False)
print(f"  step_features.parquet: {len(step_features):,} rows, {len(available_cols)} cols")

print("\nDone. Parquets saved to:", OUT_DIR)