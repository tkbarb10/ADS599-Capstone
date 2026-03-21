import pandas as pd
import numpy as np
import re

def fix_triage_outliers(df):
    d = df.copy()

    # ── Temperature ──────────────────────────────────────────────────────────
    # Values >900: assumed extra digit entered (e.g. 986 → 98.6)
    d['temperature'] = d['temperature'].apply(lambda x: x / 10 if pd.notna(x) and x > 900 else x)

    # Values 28–40: assumed recorded in Celsius, convert to Fahrenheit
    d['temperature'] = d['temperature'].apply(
        lambda x: round((x * 1.8) + 32, 1) if pd.notna(x) and 28 < x <= 40 else x
    )

    # Values 5–10: assumed missing leading digit (e.g. 9.8 → 98)
    d['temperature'] = d['temperature'].apply(lambda x: x * 10 if pd.notna(x) and 5 < x < 10 else x)

    # Values >115: no recoverable pattern → null for imputation
    d['temperature'] = d['temperature'].apply(lambda x: np.nan if pd.notna(x) and x > 115 or x < 70 else x)

    # ── Heart rate ───────────────────────────────────────────────────────────
    # Values >500: assumed extra digits entered (e.g. 800 → 80)
    d['heartrate'] = d['heartrate'].apply(lambda x: x / 10 if pd.notna(x) and x > 500 else x)

    # Values <20: no recoverable pattern → null for imputation
    d['heartrate'] = d['heartrate'].apply(lambda x: np.nan if pd.notna(x) and x < 20 else x)

    # ── Respiratory rate ─────────────────────────────────────────────────────
    # Values >1000: assumed two extra digits (e.g. 1800 → 18)
    d['resprate'] = d['resprate'].apply(lambda x: round(x / 100) if pd.notna(x) and x > 1000 else x)

    # Values 100–1000: assumed one extra digit (e.g. 180 → 18)
    d['resprate'] = d['resprate'].apply(lambda x: round(x / 10) if pd.notna(x) and x > 100 else x)

    # Values <4: no recoverable pattern → null for imputation
    d['resprate'] = d['resprate'].apply(lambda x: np.nan if pd.notna(x) and x < 4 else x)

    # ── O2 saturation ────────────────────────────────────────────────────────
    # Values 900–1010: assumed extra digit (e.g. 980 → 98, 1000 → 100)
    d['o2sat'] = d['o2sat'].apply(lambda x: int(x / 10) if pd.notna(x) and 900 < x < 1010 else x)

    # Values 0–10: assumed missing leading 9 (e.g. 8 → 98)
    d['o2sat'] = d['o2sat'].apply(lambda x: x + 90 if pd.notna(x) and 0 < x <= 10 else x)

    # Values ==0, >100, or <40: not recoverable → null for imputation
    d['o2sat'] = d['o2sat'].apply(lambda x: np.nan if pd.notna(x) and (x == 0 or x > 100 or x < 40) else x)

    # ── Systolic BP ──────────────────────────────────────────────────────────
    # Spot-checking: values ≤270 appear accurate; above that no clear correction pattern → null
    # Values <40: too low to be plausible → null
    d['sbp'] = d['sbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 270 or x < 40) else x)

    # ── Diastolic BP ─────────────────────────────────────────────────────────
    # Values >150: likely charting errors with no clear pattern → null
    # Values <20: too low to be plausible → null
    d['dbp'] = d['dbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 150 or x < 20) else x)

    return d


def clean_pain_column(df):
    d = df.copy()

    # Normalize to lowercase string, strip punctuation artifacts (quotes, >, -, +)
    d['pain'] = (
        d['pain'].astype('str').str.lower().str.strip()
        .str.strip('"').str.strip('\u201c').str.strip('\u201d')
        .str.strip('>').str.strip('-').str.strip('+')
    )

    # Range entries like "5-7" → take the lower (first) value
    d['pain'] = d['pain'].str.replace(r'(\d)-\d', r'\1', regex=True)

    # Coerce to numeric; anything non-numeric (text descriptions) → NaN
    d['pain'] = pd.to_numeric(d['pain'], errors='coerce')

    # Values >10: not a valid 0–10 pain scale entry → NaN
    # Could be mis-entries, carryover from another field, or a different scale (e.g. GCS)
    d['pain'] = d['pain'].apply(lambda x: np.nan if pd.notna(x) and x > 10 else x)

    # Round floats to nearest integer (pain scale is whole numbers), fill NaN with 'Other'
    d['pain'] = d['pain'].apply(lambda x: round(x) if pd.notna(x) else x).fillna('Other')

    return d


_RACE_MAP = [
    ('White', r'white|portuguese|brazilian'),
    ('Black', r'black|african'),
    ('Hispanic', r'hispanic|latino|south american'),
    ('Asian', r'asian|pacific islander|hawaiian'),
    ('Native American', r'american indian|alaska native'),
]

def simplify_race_column(df):
    d = df.copy()

    def _collapse(val):
        if pd.isna(val):
            return 'Other'
        v = str(val).lower()
        for label, pattern in _RACE_MAP:
            if re.search(pattern, v):
                return label
        return 'Other'

    d['race'] = d['race'].apply(_collapse)
    return d