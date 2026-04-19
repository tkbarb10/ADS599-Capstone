"""
modeling/data_prep/traditional_ml.py

Loads full_patient_state from HuggingFace, filters to the first 60 minutes
of each ED stay, aggregates to one row per stay, encodes chief complaint,
and returns a DataFrame ready for traditional ML training.

Functions:
  load_and_prep(hf_cfg, use_tfidf) -- full data prep pipeline, one row per stay
  tfidf_encode(train, test, val) -- TF-IDF encoding for chief complaint (XGBoost)
  training_split(df) -- stratified train / test / validation split by subject_id
  scaling(train, test, val) -- StandardScaler fit on train, applied to all
"""

import logging
import re

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling.data_prep.columns import get_column_groups, NON_TRAIN_COLS
from utils.load_yaml_helper import load_yaml

config = load_yaml('modeling/config/traditional_ml.yaml')
data_config = config['data']
target_col = data_config['target_col']

TERMINAL_MAP = {'discharge': 0, 'transfer_icu': 1}

CC_CATEGORIES = [
    ('chest pain|chest tightness|chest pressure',            'chest_pain'),
    ('shortness of breath|dyspnea|sob|difficulty breathing', 'dyspnea'),
    ('altered mental|confusion|altered loc|ams|unresponsive','altered_mental_status'),
    ('syncope|syncopal|fainting|passed out|loss of consciousness', 'syncope'),
    ('abdominal pain|abd pain|stomach pain|abdominal cramp',  'abdominal_pain'),
    ('nausea|vomiting|n/v|emesis',                            'nausea_vomiting'),
    ('fever|febrile|temperature',                             'fever'),
    ('headache|head pain|migraine',                           'headache'),
    ('back pain|lower back|lumbar',                           'back_pain'),
    ('fall|fell|mechanical fall',                             'fall'),
    ('trauma|injury|mvc|motor vehicle|accident',              'trauma'),
    ('seizure|convulsion',                                    'seizure'),
    ('stroke|cva|facial droop|weakness one side|slurred',     'stroke_like'),
    ('palpitation|heart racing|tachycardia',                  'palpitations'),
    ('weakness|fatigue|malaise',                              'weakness_fatigue'),
    ('urinary|dysuria|frequency|uti',                         'urinary'),
    ('leg pain|leg swelling|dvt|calf pain',                   'leg_complaint'),
    ('psychiatric|suicidal|overdose|ingestion',               'psychiatric_overdose'),
    ('bleeding|hemorrhage|blood',                             'bleeding'),
    ('pain',                                                  'pain_other'),
]

logger = logging.getLogger(__name__)


def _map_cc(text: str) -> str:
    text = str(text).lower().strip() if pd.notna(text) else ''
    for pattern, category in CC_CATEGORIES:
        if re.search(pattern, text):
            return category
    return 'other'


def load_and_prep(hf_cfg: dict, use_tfidf: bool = False) -> pd.DataFrame:
    """
    Loads full_patient_state, filters to the first 60 minutes per stay,
    aggregates to one row per stay, and encodes chief complaint.

    Args:
        hf_cfg: The hugging_face section of settings.yaml.
        use_tfidf: If True, keep chiefcomplaint raw for downstream tfidf_encode()
                    (XGBoost). If False, apply CC category OHE and drop the column.

    Returns:
        df_stay with a 'label' column (0=discharge, 1=transfer_icu).
    """
    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    logger.info('Loading full_patient_state from HuggingFace...')
    ds = load_dataset(
        hf_cfg['modeling_data_repo'],
        name=hf_cfg['full_patient_state']['config_name'],
        split=hf_cfg['full_patient_state']['split_name'],
        verification_mode='no_checks',
    )
    df = ds.to_pandas().copy()  # defragment -- HF arrow conversion produces a fragmented frame
    logger.info(f'Loaded {len(df):,} rows -- {df["ed_stay_id"].nunique():,} unique stays')

    # ------------------------------------------------------------------
    # Preprocessing
    # Gender encoding; everything else (pain, height/weight, acuity,
    # arrival_transport OHE) is already handled in full_patient_state.
    # ------------------------------------------------------------------
    df['gender'] = df['gender'].map({'F': 1, 'M': 0})
    df.drop(columns=['admission_type', 'recon_n_total_meds', 'recon_n_drug_classes'],
            inplace=True, errors='ignore')

    # ------------------------------------------------------------------
    # Detect column groups for aggregation
    # ------------------------------------------------------------------
    groups = get_column_groups(df)
    binary_max_cols = groups.binary_max_cols

    # Exclude temporal, action, location, and metadata columns from static_cols
    # so they are not accidentally aggregated with 'first'.
    known = (
        set(groups.vitals) | set(groups.vital_change) | set(binary_max_cols)
        | set(groups.action_flags) | set(groups.location_flags)
        | NON_TRAIN_COLS | {'time_since_last_min'}
    )
    static_cols = [c for c in df.columns if c not in known]

    logger.info(
        f'Columns -- lab:{len(groups.lab_ohe)} micro:{len(groups.micro_ohe)} '
        f'recon:{len(groups.recon)} disp_meds:{len(groups.dispensed_meds)} '
        f'vitals:{len(groups.vitals)} static:{len(static_cols)}'
    )

    # ------------------------------------------------------------------
    # Filter to first 60 minutes
    # ------------------------------------------------------------------
    df['time'] = pd.to_datetime(df['time'])
    df['stay_window_start'] = pd.to_datetime(df['stay_window_start'])
    df['minutes_into_stay'] = (df['time'] - df['stay_window_start']).dt.total_seconds() / 60

    df_1h = df[df['minutes_into_stay'] <= 60].copy()

    stays_missing = set(df['ed_stay_id'].unique()) - set(df_1h['ed_stay_id'].unique())
    if stays_missing:
        fallback = (
            df[df['ed_stay_id'].isin(stays_missing)]
            .sort_values('time')
            .groupby('ed_stay_id').first()
            .reset_index()
        )
        df_1h = pd.concat([df_1h, fallback], ignore_index=True)
        logger.info(f'Fallback to first row for {len(stays_missing):,} stays')

    logger.info(f'First-hour window: {len(df_1h):,} rows -- {df_1h["ed_stay_id"].nunique():,} stays')

    # ------------------------------------------------------------------
    # Aggregate to one row per stay
    # ------------------------------------------------------------------
    agg_dict = {}
    for c in groups.vitals:
        if c in df_1h.columns:
            agg_dict[c] = 'first'  # triage value
    for c in binary_max_cols:
        if c in df_1h.columns:
            agg_dict[c] = 'max'
    skip = {'ed_stay_id', 'subject_id', 'hadm_id', 'time', 'stay_window_start', 'minutes_into_stay'}
    for c in static_cols:
        if c in df_1h.columns and c not in skip:
            agg_dict[c] = 'first'

    # target_col is excluded from agg_dict via NON_TRAIN_COLS; carry it through explicitly
    if target_col in df_1h.columns:
        agg_dict[target_col] = 'first'

    df_1h = df_1h.copy()  # defragment before groupby
    df_stay = df_1h.groupby(['ed_stay_id', 'subject_id'], sort=False).agg(agg_dict)
    df_stay = df_stay.reset_index().copy()
    logger.info(f'Aggregated to {len(df_stay):,} stays -- {df_stay.shape[1]} columns')

    # ------------------------------------------------------------------
    # Chief complaint encoding
    # ------------------------------------------------------------------
    if not use_tfidf:
        df_stay['cc_category'] = df_stay['chiefcomplaint'].apply(_map_cc)
        cc_dummies = pd.get_dummies(df_stay['cc_category'], prefix='cc', dtype=int)
        df_stay = pd.concat([df_stay, cc_dummies], axis=1).drop(columns=['cc_category', 'chiefcomplaint'])
        logger.info(f'CC category columns added: {len(cc_dummies.columns)}')
    else:
        logger.info('Keeping raw chiefcomplaint for TF-IDF encoding after split')

    # ------------------------------------------------------------------
    # Drop non-feature columns (action flags + location flags)
    # These are excluded from agg_dict via `known`, but drop explicitly
    # as a safeguard in case of naming changes or data updates.
    # ------------------------------------------------------------------
    drop_non_features = [c for c in (groups.action_flags + groups.location_flags) if c in df_stay.columns]
    if drop_non_features:
        df_stay.drop(columns=drop_non_features, inplace=True)
        logger.info(f'Dropped non-feature columns: {drop_non_features}')

    # ------------------------------------------------------------------
    # Encode target
    # ------------------------------------------------------------------
    df_stay['label'] = df_stay[target_col].map(TERMINAL_MAP)
    df_stay.drop(columns=target_col, inplace=True)
    logger.info(f'Label distribution: {df_stay["label"].value_counts().to_dict()}')

    return df_stay


def tfidf_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    validation: pd.DataFrame,
    max_features: int = 50,
) -> tuple:
    """
    Fits TF-IDF on train chiefcomplaint only, then transforms test and validation.
    Drops the raw chiefcomplaint column from all three splits.
    Returns (train, test, validation, fitted_vectorizer).
    """
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))

    def _apply(df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        texts = df['chiefcomplaint'].fillna('unknown')
        matrix = tfidf.fit_transform(texts) if fit else tfidf.transform(texts)
        tfidf_df = pd.DataFrame(
            matrix.toarray(),  # type: ignore[union-attr]
            columns=[f'cc_tfidf_{t}' for t in tfidf.get_feature_names_out()],
            index=df.index,
        )
        return pd.concat([df.drop(columns='chiefcomplaint'), tfidf_df], axis=1)

    train = _apply(train, fit=True)
    test = _apply(test, fit=False)
    validation = _apply(validation, fit=False)

    logger.info(f'TF-IDF encoded {max_features} features from chiefcomplaint')
    return train, test, validation, tfidf


def training_split(df: pd.DataFrame, stratify_label: str = 'label', train_size: float = 0.8):
    if train_size >= 1:
        raise ValueError(f'Train size must be less than 1. Got {train_size}.')
    random_state = config['general'].get('random_state', 10)

    # Split by subject_id to prevent leakage across multiple visits
    subject_label = df[['subject_id', stratify_label]].drop_duplicates('subject_id', keep='first')
    train_subs, holdout_subs = train_test_split(
        subject_label['subject_id'],
        train_size=train_size,
        stratify=subject_label[stratify_label],
        random_state=random_state,
    )
    train_set = df[df['subject_id'].isin(train_subs)].copy()
    holdout = df[df['subject_id'].isin(holdout_subs)].copy()

    holdout_label = holdout[['subject_id', stratify_label]].drop_duplicates('subject_id', keep='first')
    test_subs, val_subs = train_test_split(
        holdout_label['subject_id'],
        test_size=0.5,
        stratify=holdout_label[stratify_label],
        random_state=random_state,
    )
    test_set = holdout[holdout['subject_id'].isin(test_subs)].copy()
    validation_set = holdout[holdout['subject_id'].isin(val_subs)].copy()

    return train_set, test_set, validation_set


def scaling(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
    groups = get_column_groups(train)
    cols = [c for c in groups.scaling_cols if c in train.columns]
    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    validation = validation.copy()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    validation[cols] = scaler.transform(validation[cols])
    return train, test, validation, scaler