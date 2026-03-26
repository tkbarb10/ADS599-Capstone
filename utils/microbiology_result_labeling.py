import re
import pandas as pd

def classify_culture_result(org_name, comments):
    # --- Step 1: check org_name ---
    if pd.notna(org_name):
        val = str(org_name).upper().strip()
        if re.search(r'\bNEGATIVE\b|\bNOT\b', val):
            return 'NEGATIVE'
        if re.search(r'\bPOSITIVE\b', val):
            return 'POSITIVE'
        if re.search(r'\bCANCELLED\b|\bCANCELED\b', val):
            return 'CANCELLED'
        return 'POSITIVE'

    # --- Step 2: check comments (lowercase everything) ---
    if pd.notna(comments):
        val = str(comments).lower().strip()

        if re.search(r'\bcancell?ed\b', val):
            return 'CANCELLED'
        if re.search(r'\btest not performed\b', val):
            return 'CANCELLED'
        if re.search(r'\bpatient credited\b', val):
            return 'CANCELLED'

        if re.search(r'\bindeterminate\b', val):
            return 'NEGATIVE'

        # Negative patterns
        if val == 'no growth.':
            return 'NEGATIVE'
        if re.search(r'\bno\b.+\b(seen|found|isolated)\b', val):
            return 'NEGATIVE'
        if re.search(r'\bno\b.+growth', val):
            return 'NEGATIVE'
        if re.search(r'\bnot detected\b', val):
            return 'NEGATIVE'
        if re.search(r'\bnegative\b|\bnonreactive\b', val):
            return 'NEGATIVE'

        # Positive patterns
        if re.search(r'^\s*[\d<>]', val):
            return 'POSITIVE'
        if re.search(r'<\s*\d[\d,]*\s*(cfu|organisms)', val):
            return 'POSITIVE'
        if re.search(r'growth', val):
            return 'POSITIVE'
        if re.search(r'\bconsistent with\b|\bcontamination\b|\bpositive\b', val):
            return 'POSITIVE'
        if re.search(r'(?<!\bno\s)(?<!\bnon)\breactive\b', val):
            return 'POSITIVE'

        # Placeholder sentinel — only underscores, dashes, and spaces
        if re.fullmatch(r'[\s_-]+', val):
            return 'OTHER'

        # Anything left with actual content is likely a positive clinical note
        return 'POSITIVE'

    return 'OTHER'