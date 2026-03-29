import pandas as pd
import numpy as np

def null_triage_outliers(df):
    d = df.copy()

    # ── Temperature ──────────────────────────────────────────────────────────

    # Assuming that temps outside of this range are mistakes in imputation or imputed as celsius so converting to null to drop
    d['temperature'] = d['temperature'].apply(lambda x: np.nan if pd.notna(x) and x > 115 or x < 70 else x)

    # ── Heart rate ───────────────────────────────────────────────────────────
    # Values >500: assumed extra digits entered (e.g. 800 → 80)
    d['heartrate'] = d['heartrate'].apply(lambda x: x / 10 if pd.notna(x) and x > 500 else x)

    # Values <20: no recoverable pattern → null for imputation
    d['heartrate'] = d['heartrate'].apply(lambda x: np.nan if pd.notna(x) and x < 20 else x)

    # ── Respiratory rate ─────────────────────────────────────────────────────
    # Values >1000: assumed two extra digits (e.g. 1800 → 18)
    d['resprate'] = d['resprate'].apply(lambda x: round(x / 100) if pd.notna(x) and x > 1000 else x)

    # Values over 100: assuming some mistake in imputation so converting to null
    d['resprate'] = d['resprate'].apply(lambda x: np.nan if pd.notna(x) and x > 100 else x)

    # Values <4: no recoverable pattern → null for imputation
    d['resprate'] = d['resprate'].apply(lambda x: np.nan if pd.notna(x) and x < 4 else x)

    # ── O2 saturation ────────────────────────────────────────────────────────

    # Values ==0, >100, or <40: not recoverable → null for removal
    d['o2sat'] = d['o2sat'].apply(lambda x: np.nan if pd.notna(x) and (x == 0 or x > 100 or x < 75) else x)

    # ── Systolic BP ──────────────────────────────────────────────────────────
    # Spot-checking: values ≤270 appear accurate; above that no clear correction pattern → null
    # Values <40: too low to be plausible → null
    d['sbp'] = d['sbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 270 or x < 40) else x)

    # ── Diastolic BP ─────────────────────────────────────────────────────────
    # Values >150: likely charting errors with no clear pattern → null
    # Values <20: too low to be plausible → null
    d['dbp'] = d['dbp'].apply(lambda x: np.nan if pd.notna(x) and (x > 150 or x < 20) else x)

    return d


def null_pain_column(df):
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

    return d