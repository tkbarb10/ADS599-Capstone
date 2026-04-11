import logging
import pandas as pd

from data_pipelines.feature_engineering.drug_classes import safe_col

logger = logging.getLogger(__name__)

# Ordered substring mapping from ETC description labels to the shared 22-class vocabulary.
# Priority: first match wins. Case-insensitive substring match on etcdescription value.
# Chronic maintenance classes (statins, thyroid, vitamins, etc.) explicitly mapped to Other
# — they provide limited discriminative signal for ED acuity prediction.
ETCDESC_TO_CLASS = [
    ('opioid',                          'Analgesic - Opioid/NSAID'),
    ('analgesic opioid',                'Analgesic - Opioid/NSAID'),
    ('narcotic',                        'Analgesic - Opioid/NSAID'),
    ('nsaid',                           'Analgesic - NSAID'),
    ('salicylate analgesic',            'Analgesic - NSAID'),
    ('non-opioid',                      'Analgesic - Acetaminophen'),
    ('acetaminophen',                   'Analgesic - Acetaminophen'),
    ('antiemetic',                      'Antiemetic'),
    ('antibiotic',                      'Antibiotic'),
    ('anti-infective',                  'Antibiotic'),
    ('antimicrobial',                   'Antibiotic'),
    ('benzodiazepine',                  'Benzodiazepine - Sedative/Anxiolytic'),
    ('sedative',                        'Benzodiazepine - Sedative/Anxiolytic'),
    ('anxiolytic',                      'Benzodiazepine - Sedative/Anxiolytic'),
    ('anticoagulant',                   'Anticoagulant'),
    ('thrombin inhibitor',              'Anticoagulant'),
    ('factor xa',                       'Anticoagulant'),
    ('beta blocker',                    'Beta Blocker'),
    ('beta-adrenergic blocking',        'Beta Blocker'),
    ('ace inhibitor',                   'ACE Inhibitor'),
    ('angiotensin converting enzyme',   'ACE Inhibitor'),
    ('calcium channel blocker',         'Calcium Channel Blocker'),
    ('dihydropyridine',                 'Calcium Channel Blocker'),
    ('nitrate',                         'Nitrate'),
    ('antiarrhythmic',                  'Antiarrhythmic'),
    ('cardiac glycoside',               'Antiarrhythmic'),
    ('diuretic',                        'Diuretic'),
    ('corticosteroid',                  'Corticosteroid'),
    ('glucocorticoid',                  'Corticosteroid'),
    ('steroid',                         'Corticosteroid'),
    ('proton pump inhibitor',           'GI - Acid Suppression'),
    ('h2-receptor',                     'GI - Acid Suppression'),
    ('acid secretion',                  'GI - Acid Suppression'),
    ('bronchodilator',                  'Bronchodilator'),
    ('beta 2-adrenergic',               'Bronchodilator'),
    ('anticholinergic bronch',          'Bronchodilator'),
    ('insulin',                         'Insulin/Glucose'),
    ('antidiabetic',                    'Insulin/Glucose'),
    ('sulfonylurea',                    'Insulin/Glucose'),
    ('antipsychotic',                   'Antipsychotic'),
    ('anticonvulsant',                  'Anticonvulsant'),
    ('gaba analog',                     'Anticonvulsant'),
    ('antiepileptic',                   'Anticonvulsant'),
    ('sodium chloride',                 'IV Fluid'),
    ('iv solution',                     'IV Fluid'),
    ('electrolyte replacement',         'IV Fluid'),
    ('antiplatelet',                    'Antiplatelet'),
    ('platelet aggregation inhibitor',  'Antiplatelet'),
    ('salicylate',                      'Antiplatelet'),
    # Chronic maintenance — limited discriminative signal for ED acuity
    ('statin',                          'Other'),
    ('hmg-coa reductase',               'Other'),
    ('thyroid',                         'Other'),
    ('vitamin',                         'Other'),
    ('supplement',                      'Other'),
    ('laxative',                        'Other'),
    ('antidepressant',                  'Other'),
]


def map_etcdesc_to_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map etcdescription to the shared 22-class vocabulary via substring match.
    Priority: first match in ETCDESC_TO_CLASS wins. Null/unmatched → 'Other'.

    Why 22 groups:
      Raw etcdescription has 1,199 unique labels — too many to encode as model
      features. Collapsed to match the dispensed_meds feature space so recon_*
      and meds_* features are directly comparable in the RL state vector.
    """
    def _map(val):
        if pd.isna(val):
            return 'Other'
        v = str(val).lower()
        for pattern, cls in ETCDESC_TO_CLASS:
            if pattern in v:
                return cls
        return 'Other'

    df = df.copy()
    df['shared_class'] = df['etcdescription'].apply(_map)
    logger.info(f"shared_class distribution:\n{df['shared_class'].value_counts()}")
    other_pct = (df['shared_class'] == 'Other').mean() * 100
    logger.info(f"Records mapped to 'Other': {(df['shared_class'] == 'Other').sum():,} ({other_pct:.1f}%)")
    return df


def build_recon_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary drug class feature matrix — one row per ed_stay_id.

    Output columns: subject_id, ed_stay_id, recon_<class> (uint8 flags),
    recon_n_total_meds, recon_n_drug_classes.

    Why binary not count:
      Presence/absence of a drug class is the clinically relevant signal.
      A patient on 3 beta blockers carries the same cardiac risk as one on 1.
      Binary flags are also memory-efficient (uint8 vs float64).

    Low-variance dropping:
      Classes present in fewer than 1% of visits are dropped — the model
      sees almost no positive examples to learn a signal from.
      recon_other is also dropped explicitly: ~48% prevalence but near-constant
      across patients, so it adds no discriminative signal.

    Note: visits with no medrecon record will receive all-zero recon_* flags
    when this table is merged onto the cohort in the combined state pipeline.
    """
    LOW_VARIANCE_THRESHOLD = 0.01

    pairs = (df[['ed_stay_id', 'shared_class']]
             .drop_duplicates()
             .assign(flag=1))
    pivot = (pairs
             .set_index(['ed_stay_id', 'shared_class'])['flag']
             .unstack(fill_value=0)
             .astype('uint8'))
    pivot.columns = [f'recon_{safe_col(c)}' for c in pivot.columns]
    pivot = pivot.reset_index()

    flag_cols = [c for c in pivot.columns if c.startswith('recon_')]
    prevalence = pivot[flag_cols].mean()
    drop_cols = prevalence[prevalence < LOW_VARIANCE_THRESHOLD].index.tolist()
    if 'recon_other' in pivot.columns:
        drop_cols.append('recon_other')
    pivot = pivot.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} low-variance/uninformative columns: {drop_cols}")

    counts = (df.groupby('ed_stay_id')
              .agg(recon_n_total_meds=('name', 'count'),
                   recon_n_drug_classes=('shared_class', 'nunique'))
              .reset_index())

    subject_map = df[['ed_stay_id', 'subject_id']].drop_duplicates('ed_stay_id')
    result = subject_map.merge(pivot, on='ed_stay_id', how='right')
    result = result.merge(counts, on='ed_stay_id', how='left')
    logger.info(f"Recon feature matrix shape: {result.shape}")
    return result
