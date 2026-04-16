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

from dataclasses import dataclass, field
from typing import List, Literal
import pandas as pd

TERMINAL_MAP = {'discharge': 0, 'transfer_icu': 1}

# Pure identifiers, timestamps, and target columns -- never training features.
# traditional_ml.py excludes it separately via the aggregation skip set.
NON_TRAIN_COLS: frozenset = frozenset({
    'ed_stay_id', 'subject_id', 'hadm_id', 'time', 'step_idx',
    'stay_window_start', 'stay_window_end', 'cohort_label',
    'terminal_code', 'terminal_event', 'total_length', 'admission_type'
})

# Part of the RL action tuple, not state features. Available via groups.action_flags for the RL agent.
_ACTION_COLS = ('vitals_checked', 'labs_ordered', 'micro_ordered', 'ecg_ordered', 'rad_ordered', 'meds_ordered')

# These are the actions the RL agent can take to terminate the session
_TERMINAL_ACTIONS = ('discharge', 'transfer_icu')

# in_ed, in_ward -- tracking columns used by the Streamlit app, not training features.
_LOCATION_COLS = ('in_ed', 'in_ward')


@dataclass
class ColumnGroups:
    vitals: List[str]           # current_* -- triage / most-recent vital readings
    vital_change: List[str]     # _rolling1h, _delta, _rate_per_min -- temporal derivatives
    lab_ohe: List[str]          # {category}-{fluid}_Normal/Pending/Abnormal
    micro_ohe: List[str]        # {spec_type}_Pending/Positive/Negative/Other
    ecg_ohe: List[str]       # ecg_status_* and rad_status_*
    rad_ohe: List[str]
    dispensed_meds: List[str]   # ACE Inhibitor through Other (by drug class)
    recon: List[str]            # recon_* -- pre-arrival medication reconciliation flags
    arrival: List[str]          # arrival_* -- OHE arrival transport columns
    missing: List[str]          # *_missing -- missingness indicator columns
    action_flags: List[str]     # RL action tuple columns (not state features)
    location_flags: List[str]   # Streamlit tracking columns (not training features)
    terminal_actions: List[str] = field(default_factory=lambda: list(_TERMINAL_ACTIONS))

    @property
    def binary_max_cols(self) -> List[str]:
        """
        Clinical OHE feature columns aggregated with max() in the traditional ML
        one-row aggregation. action_flags and location_flags are excluded.
        """
        return (
            self.lab_ohe + self.micro_ohe + self.recon
            + self.dispensed_meds + self.ecg_ohe + self.rad_ohe
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
            + self.ecg_ohe
            + self.rad_ohe
            + self.arrival
            + self.missing
        )
    
    @property
    def action_cols(self) -> List[str]:
        return self.action_flags
    
    @property
    def terminal_row(self) -> str:
        return 'terminal_code'

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
    ecg_ohe = [
        c for c in df.columns if c.startswith('ecg_status')
        ]
    rad_ohe = [
        c for c in df.columns if c.startswith("rad_status")
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
        ecg_ohe=ecg_ohe,
        rad_ohe=rad_ohe,
        dispensed_meds=dispensed_meds,
        recon=recon,
        arrival=arrival,
        missing=missing,
        action_flags=action_flags,
        location_flags=location_flags,
    )

def create_action_cols(df: pd.DataFrame, cols: list[str], suffix: Literal['ecg', 'rad', 'meds']):
        """Columns in full patient state for meds, ecg and rad are results only.  This function creates indicator columns for when those orders were placed (the time step before)"""
        result_onset = (
            df.groupby('ed_stay_id')[cols]
            .diff()
            .gt(0)
            .any(axis=1)
        )
        new_column = f'{suffix}_ordered'
        df[new_column] = result_onset.shift(-1, fill_value=False).astype(int)
        return df