"""
modeling/data_prep/columns.py

Reusable column group detection for full_patient_state.
Used by both traditional_ml.py and lstm.py so the naming-convention logic
is defined once and shared.

Usage:
    from modeling.data_prep.columns import get_column_groups, NON_TRAIN_COLS

    groups = get_column_groups(df)
    state_cols = groups.state_cols          # LSTM / RL state sequences
    binary_max = groups.binary_max_cols     # traditional ML aggregation
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

# Pure identifiers, timestamps, and target columns -- never training features.
# Note: time_since_last_min is NOT here; it is a real feature for LSTM and RL.
# traditional_ml.py excludes it separately via the aggregation skip set.
NON_TRAIN_COLS: frozenset = frozenset({
    'ed_stay_id', 'subject_id', 'hadm_id', 'time', 'step_idx',
    'stay_window_start', 'stay_window_end', 'cohort_label',
    'terminal_code', 'terminal_event', 'total_length',
})

# vitals_checked, labs_ordered, micro_ordered -- part of the RL action tuple,
# not state features. Available via groups.action_flags for the RL agent.
_ACTION_COLS = ('vitals_checked', 'labs_ordered', 'micro_ordered')

# in_ed, in_ward -- tracking columns used by the Streamlit app, not training features.
_LOCATION_COLS = ('in_ed', 'in_ward')


@dataclass
class ColumnGroups:
    vitals: List[str]           # current_* -- triage / most-recent vital readings
    vital_change: List[str]     # _rolling1h, _delta, _rate_per_min -- temporal derivatives
    lab_ohe: List[str]          # {category}-{fluid}_Normal/Pending/Abnormal
    micro_ohe: List[str]        # {spec_type}_Pending/Positive/Negative/Other
    status_ohe: List[str]       # ecg_status_* and rad_status_*
    dispensed_meds: List[str]   # ACE Inhibitor through Other (by drug class)
    recon: List[str]            # recon_* -- pre-arrival medication reconciliation flags
    arrival: List[str]          # arrival_* -- OHE arrival transport columns
    missing: List[str]          # *_missing -- missingness indicator columns
    action_flags: List[str]     # RL action tuple columns (not state features)
    location_flags: List[str]   # Streamlit tracking columns (not training features)

    @property
    def binary_max_cols(self) -> List[str]:
        """
        Clinical OHE feature columns aggregated with max() in the traditional ML
        one-row aggregation. action_flags and location_flags are excluded.
        """
        return (
            self.lab_ohe + self.micro_ohe + self.recon
            + self.dispensed_meds + self.status_ohe
        )

    @property
    def state_cols(self) -> List[str]:
        """
        Ordered feature column list for LSTM / RL sequences.
        Includes time_since_last_min and vital_change (trends).
        action_flags and location_flags are excluded -- action_flags are part
        of the RL action tuple; location_flags are for tracking only.
        """
        return (
            ['gender', 'anchor_age', 'acuity', 'height', 'weight', 'time_since_last_min']
            + self.dispensed_meds
            + self.recon
            + self.vitals
            + self.vital_change
            + self.lab_ohe
            + self.micro_ohe
            + self.status_ohe
            + self.arrival
            + self.missing
        )


def get_column_groups(df: pd.DataFrame) -> ColumnGroups:
    """
    Detects all feature column groups from full_patient_state using naming conventions.

    Args:
        df: full_patient_state DataFrame (post-load, before any aggregation).

    Returns:
        ColumnGroups with every feature group populated.
    """
    lab_ohe = [
        c for c in df.columns
        if c.endswith(('_Normal', '_Pending', '_Abnormal')) and '-' in c
    ]
    micro_ohe = [
        c for c in df.columns
        if c.endswith(('_Pending', '_Positive', '_Negative', '_Other'))
        and '-' not in c
        and not c.startswith('ecg_status')
        and not c.startswith('rad_status')
    ]
    status_ohe = [
        c for c in df.columns
        if c.startswith('ecg_status') or c.startswith('rad_status')
    ]
    vitals = [c for c in df.columns if c.startswith('current_')]
    vital_change = [
        c for c in df.columns
        if c.endswith(('_rolling1h', '_delta', '_rate_per_min'))
    ]
    recon = [c for c in df.columns if c.startswith('recon_')]
    arrival = [c for c in df.columns if c.startswith('arrival_')]
    missing = [c for c in df.columns if c.endswith('_missing')]
    action_flags = [c for c in _ACTION_COLS if c in df.columns]
    location_flags = [c for c in _LOCATION_COLS if c in df.columns]
    dispensed_meds = df.loc[:, 'ACE Inhibitor':'Other'].columns.to_list()

    return ColumnGroups(
        vitals=vitals,
        vital_change=vital_change,
        lab_ohe=lab_ohe,
        micro_ohe=micro_ohe,
        status_ohe=status_ohe,
        dispensed_meds=dispensed_meds,
        recon=recon,
        arrival=arrival,
        missing=missing,
        action_flags=action_flags,
        location_flags=location_flags,
    )