# Shared 22-class drug vocabulary used across all medication feature modules.
# meds_features.py maps via regex on raw drug names.
# medrecon_features.py maps via substring match on ETC description strings.
# Both produce the same output classes so recon_* and meds_* features occupy
# the same feature space and are directly comparable in the RL state vector.

DRUG_CLASSES = [
    'ACE Inhibitor',
    'Analgesic - Acetaminophen',
    'Analgesic - NSAID',
    'Analgesic - Opioid/NSAID',
    'Antiarrhythmic',
    'Antibiotic',
    'Anticoagulant',
    'Anticonvulsant',
    'Antiemetic',
    'Antiplatelet',
    'Antipsychotic',
    'Benzodiazepine - Sedative/Anxiolytic',
    'Beta Blocker',
    'Bronchodilator',
    'Calcium Channel Blocker',
    'Corticosteroid',
    'Diuretic',
    'GI - Acid Suppression',
    'Insulin/Glucose',
    'IV Fluid',
    'Nitrate',
    'Other',
]

CLASS_TO_IDX = {c: i for i, c in enumerate(DRUG_CLASSES)}


def safe_col(s: str) -> str:
    """Normalize a drug class name to a valid snake_case column name."""
    return (str(s).lower()
            .replace(' ', '_').replace('-', '_').replace('/', '_')
            .replace('(', '').replace(')', '').replace(',', '')
            .replace('&', 'and')[:60])
