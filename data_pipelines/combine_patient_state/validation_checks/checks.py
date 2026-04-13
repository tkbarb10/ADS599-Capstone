import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _pass(msg: str) -> None:
    logger.info(f'[PASS] {msg}')


def _warn(msg: str) -> None:
    logger.warning(f'[WARN] {msg}')


def _fail(msg: str) -> None:
    logger.error(f'[FAIL] {msg}')
    raise AssertionError(msg)


def run_all_checks(
    df_patient: pd.DataFrame,
    cohort: pd.DataFrame,
    lab_actions: list[str],
    micro_actions: list[str],
) -> pd.DataFrame:
    """
    Consolidated validation pass. Each check runs exactly once.
    Fixes OHE dtype issues in-place rather than asserting on them.
    Raises AssertionError on any critical failure.

    Returns df_patient (may have step_idx reassigned if out-of-window rows were dropped).
    """
    logger.info('Running final validation checks...')

    # 1. No duplicate (ed_stay_id, step_idx)
    n_dups = df_patient.duplicated(subset=['ed_stay_id', 'step_idx']).sum()
    if n_dups > 0:
        _fail(f'{n_dups:,} duplicate (ed_stay_id, step_idx) rows found')
    else:
        _pass('No duplicate (ed_stay_id, step_idx)')

    # 2. All times within stay window (drop any remaining stragglers and reassign step_idx)
    before = (df_patient['time'] < df_patient['stay_window_start']).sum()
    after  = (df_patient['time'] > df_patient['stay_window_end']).sum()
    if before + after > 0:
        _warn(f'{before + after:,} out-of-window rows -- dropping and reassigning step_idx')
        df_patient = df_patient[
            (df_patient['time'] >= df_patient['stay_window_start']) &
            (df_patient['time'] <= df_patient['stay_window_end'])
        ].reset_index(drop=True)
        df_patient['step_idx'] = df_patient.index
    _pass('All times within stay window')

    # 3. Lab OHE mutual exclusivity: _Normal + _Abnormal <= 1
    lab_conflicts = 0
    for a in lab_actions:
        n = (df_patient[f'{a}_Normal'] + df_patient[f'{a}_Abnormal'] > 1).sum()
        if n > 0:
            lab_conflicts += n
            logger.warning(f'  Lab {a}: {n} rows with _Normal=1 and _Abnormal=1')
    if lab_conflicts > 0:
        _fail(f'Lab OHE exclusivity failed -- {lab_conflicts} conflicting rows')
    else:
        _pass('Lab OHE mutual exclusivity (_Normal + _Abnormal <= 1)')

    # 4. Micro OHE mutual exclusivity: _Negative + _Positive + _Other <= 1
    micro_conflicts = 0
    for a in micro_actions:
        n = (
            df_patient[f'{a}_Negative'] +
            df_patient[f'{a}_Positive'] +
            df_patient[f'{a}_Other'] > 1
        ).sum()
        if n > 0:
            micro_conflicts += n
            logger.warning(f'  Micro {a}: {n} rows with multiple result columns = 1')
    if micro_conflicts > 0:
        _fail(f'Micro OHE exclusivity failed -- {micro_conflicts} conflicting rows')
    else:
        _pass('Micro OHE mutual exclusivity (_Negative + _Positive + _Other <= 1)')

    # 5. Every stay ends with terminal_code in {0, 1}
    last_rows = df_patient[~df_patient.duplicated(subset='ed_stay_id', keep='last')]
    bad_terminal = (~last_rows['terminal_code'].isin([0, 1])).sum()
    if bad_terminal > 0:
        _fail(f'{bad_terminal} stays have terminal_code outside {{0, 1}}')
    else:
        _pass('All stays have terminal_code in {0, 1}')

    # 6. Cohort coverage
    patient_stays = set(df_patient['ed_stay_id'].unique())
    cohort_stays  = set(cohort['ed_stay_id'].unique())
    only_in_cohort  = cohort_stays - patient_stays
    only_in_patient = patient_stays - cohort_stays
    if only_in_patient:
        _fail(f'{len(only_in_patient)} stays in df_patient not in cohort')
    if only_in_cohort:
        # Single-step stays may have been dropped; log as warning not error
        _warn(f'{len(only_in_cohort)} cohort stays absent from df_patient '
              f'(likely dropped as single-step)')
    else:
        _pass('Cohort coverage matches')

    # 7. OHE dtype fix -- convert any float OHE columns to int
    ohe_cols = (
        [f'{a}_{s}' for a in lab_actions for s in ('Pending', 'Normal', 'Abnormal')] +
        [f'{a}_{s}' for a in micro_actions for s in ('Pending', 'Negative', 'Positive', 'Other')]
    )
    float_ohe = [c for c in ohe_cols if c in df_patient.columns
                 and df_patient[c].dtype == float]
    if float_ohe:
        df_patient[float_ohe] = df_patient[float_ohe].fillna(0).astype(int)
        _warn(f'Converted {len(float_ohe)} float OHE columns to int')
    else:
        _pass('All OHE columns are int dtype')

    # 8. Exactly 1 triage row per stay
    if 'source' in df_patient.columns:
        triage_counts = df_patient[df_patient['source'] == 'triage'].shape[0]
        n_stays = df_patient['ed_stay_id'].nunique()
        if triage_counts != n_stays:
            _fail(f'Expected {n_stays} triage rows, found {triage_counts}')
        else:
            _pass('Each stay has exactly 1 triage row')

    # 9. Vital column null rates (logged as info -- nulls are expected for rare vitals)
    vital_cols = [c for c in df_patient.columns if c.startswith('current_')]
    if vital_cols:
        null_rates = df_patient[vital_cols].isna().mean().sort_values(ascending=False)
        logger.info(f'Vital null rates:\n{null_rates.to_string()}')

    logger.info(f'Final df_patient shape: {df_patient.shape}  '
                f'unique stays: {df_patient["ed_stay_id"].nunique():,}')
    return df_patient
