import logging
import math
import re
import pandas as pd

logger = logging.getLogger(__name__)

NAME_TO_CLASS = {
    r'morphine|hydromorphone|dilaudid|fentanyl|oxycodone|tramadol|ketorolac|toradol': 'Analgesic - Opioid/NSAID',
    r'acetaminophen|tylenol': 'Analgesic - Acetaminophen',
    r'ibuprofen|naproxen': 'Analgesic - NSAID',
    r'ondansetron|zofran|promethazine|phenergan|metoclopramide|reglan|prochlorperazine': 'Antiemetic',
    r'vancomycin|ceftriaxone|cefazolin|piperacillin|zosyn|azithromycin|ciprofloxacin|metronidazole|flagyl|ampicillin|meropenem|levofloxacin': 'Antibiotic',
    r'lorazepam|ativan|diazepam|valium|midazolam|versed|alprazolam': 'Benzodiazepine - Sedative/Anxiolytic',
    r'heparin|enoxaparin|lovenox|warfarin|coumadin|apixaban|rivaroxaban': 'Anticoagulant',
    r'metoprolol|lopressor|labetalol|atenolol|carvedilol': 'Beta Blocker',
    r'lisinopril|enalapril|captopril|ramipril': 'ACE Inhibitor',
    r'amlodipine|diltiazem|verapamil|nifedipine': 'Calcium Channel Blocker',
    r'nitroglycerin|nitro': 'Nitrate',
    r'amiodarone|adenosine|digoxin': 'Antiarrhythmic',
    r'furosemide|lasix|bumetanide|torsemide|hydrochlorothiazide': 'Diuretic',
    r'methylprednisolone|prednisone|dexamethasone|hydrocortisone': 'Corticosteroid',
    r'pantoprazole|omeprazole|famotidine|pepcid|ranitidine': 'GI - Acid Suppression',
    r'albuterol|ipratropium|atrovent|levalbuterol': 'Bronchodilator',
    r'insulin|dextrose|glucagon': 'Insulin/Glucose',
    r'haloperidol|haldol|olanzapine|quetiapine|risperidone': 'Antipsychotic',
    r'levetiracetam|keppra|phenytoin|dilantin|valproate|depakote|lacosamide': 'Anticonvulsant',
    r'normal saline|sodium chloride 0\.9|lactated|ringer|d5w|dextrose 5': 'IV Fluid',
    r'aspirin|clopidogrel|plavix|ticagrelor': 'Antiplatelet',
}

# Stable alphabetically-sorted vocabulary so action_idx is consistent across runs
_CLASS_VOCAB = sorted(set(NAME_TO_CLASS.values()) | {'Other'})
CLASS_TO_IDX = {c: i for i, c in enumerate(_CLASS_VOCAB)}


def _classify(name: str) -> str:
    if pd.isna(name):
        return 'Other'
    name_lower = str(name).lower()
    for pattern, drug_class in NAME_TO_CLASS.items():
        if re.search(pattern, name_lower):
            return drug_class
    return 'Other'


def map_drug_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add drug_class and action_idx columns via regex on medication name."""
    df = df.copy()
    df['drug_class'] = df['medication'].apply(_classify)
    df['action_idx'] = df['drug_class'].map(CLASS_TO_IDX).astype(int)
    logger.info(f"Drug class distribution:\n{df['drug_class'].value_counts().head(10)}")
    return df


def add_minutes_into_stay(
    df: pd.DataFrame,
    cohort: pd.DataFrame,
    time_block: int,
) -> pd.DataFrame:
    """
    Join cohort ed_intime, compute minutes_into_stay and time_step bucket.
    Filters to [0, 1440] minute window. Drops rows with no cohort match.

    Args:
        df: meds DataFrame with ed_stay_id and charttime columns
        cohort: cohort DataFrame with ed_stay_id and ed_intime columns
        time_block: minutes per time step bucket (from settings['time_block'])
    """
    df = df.copy()

    ed_intime = cohort[['ed_stay_id', 'ed_intime']].drop_duplicates('ed_stay_id').copy()
    ed_intime['ed_intime'] = pd.to_datetime(ed_intime['ed_intime'])
    if ed_intime['ed_intime'].dt.tz is not None:
        ed_intime['ed_intime'] = ed_intime['ed_intime'].dt.tz_localize(None)

    before = len(df)
    df = df.merge(ed_intime, on='ed_stay_id', how='inner')
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} rows with no cohort match")

    df['minutes_into_stay'] = (
        (df['charttime'] - df['ed_intime']).dt.total_seconds() / 60
    )
    df = df[(df['minutes_into_stay'] >= 0) & (df['minutes_into_stay'] <= 1440)].copy()
    logger.info(f"Rows after [0, 1440] min window filter: {len(df):,}")

    df['time_step'] = (df['minutes_into_stay'] / time_block).apply(math.floor).astype(int)

    df = df.drop(columns=['ed_intime'])
    return df
