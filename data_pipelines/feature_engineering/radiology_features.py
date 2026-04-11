import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

# CPT code → acuity level lookup.
# Level 2: high-acuity — CT of critical regions, CTA, MRI brain, portable/fluoroscopy.
# Level 1: moderate-acuity — standard CT, MRI spine, ultrasound, echocardiogram.
# Unknown CPT codes fall through to string matching → 0 if unrecognized.
_CPT_ACUITY = {
    "71250": 2, "71260": 2, "71270": 2,   # CT chest
    "71275": 2,                            # CTA chest
    "74177": 2, "74178": 2,               # CT abdomen/pelvis combined
    "70450": 2, "70460": 2, "70470": 2,   # CT head
    "70496": 2, "70498": 2,               # CTA head/neck
    "70551": 2, "70552": 2, "70553": 2,   # MRI brain
    "77001": 2, "77002": 2,               # fluoroscopy/line placement
    "71045": 1, "71046": 1,               # chest X-ray (1-2 views)
    "71047": 1, "71048": 1,               # chest X-ray (3-4 views)
    "74150": 1, "74160": 1, "74170": 1,   # CT abdomen only
    "74179": 1,                            # CT abdomen/pelvis w/o contrast
    "72192": 1, "72193": 1, "72194": 1,   # CT pelvis
    "72141": 1, "72146": 1, "72148": 1,   # MRI spine
    "76700": 1, "76705": 1,               # ultrasound abdomen
    "76856": 1, "76857": 1,               # ultrasound pelvis
    "93306": 1, "93307": 1, "93308": 1,   # echocardiogram
}

_HIGH_ACUITY_STRINGS = [
    "CTA ",
    "CT HEAD", "CT CHEST", "CT ABD", "CT PELVIS", "CT C-SPINE",
    "COMPUTED TOMOGRAPHY HEAD", "COMPUTED TOMOGRAPHY CHEST",
    "MR HEAD", "MRI BRAIN", "MAGNETIC RESONANCE HEAD",
    "CHEST (PORTABLE", "CHEST PORT", "PORTABLE CHEST",
    "FLUORO", "LINE PLACEMENT", "GUIDANCE",
]

_MODERATE_ACUITY_STRINGS = [
    "CT ", " CT",
    "MRI", "MR ",
    "ULTRASOUND", " US", "US.",
    "RENAL U", "PELVIS U",
    "ECHO",
    "PA & LAT", "AP & LAT",
    "X-RAY", "RADIOGRAPH",
]


def _is_cpt_code(value: str) -> bool:
    return bool(re.match(r"^\d{5}$", str(value).strip()))


def assign_rad_acuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign exam_acuity level per row from cpt_code (priority) or exam_name.

    cpt_code uses a direct lookup table; exam_name uses ordered string matching.
    Ordering matters for string matching — high-acuity patterns are checked first
    so "CT HEAD" doesn't fall into the generic "CT " moderate bucket.

    Level 2: high-acuity — CTA, CT of critical regions, MRI brain,
             portable chest (implies unstable patient), fluoroscopy/line placement.
    Level 1: moderate-acuity — standard CT, MRI spine, ultrasound, echo, chest X-ray.
    Level 0: routine/elective — mammography, extremity films, unrecognized exams.
    """
    def _score_row(row) -> int:
        cpt = row.get('cpt_code')
        if cpt and not pd.isna(cpt) and _is_cpt_code(str(cpt)):
            return _CPT_ACUITY.get(str(cpt).strip(), 0)
        exam = row.get('exam_name', '')
        if pd.isna(exam):
            return 0
        name = str(exam).upper().strip()
        if any(h in name for h in _HIGH_ACUITY_STRINGS):
            return 2
        if any(m in name for m in _MODERATE_ACUITY_STRINGS):
            return 1
        return 0

    df = df.copy()
    df['exam_acuity'] = df.apply(_score_row, axis=1)
    logger.info(f"exam_acuity distribution:\n{df['exam_acuity'].value_counts().sort_index()}")
    return df