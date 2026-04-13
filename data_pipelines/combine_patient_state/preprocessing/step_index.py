"""Purpose of this script is to set up a event driven time series for a patient stay from the time they enter the ED, to the time they enter the 
ICU or are discharged.  We load a feature dataset with timestamp events, add the timestamps to the cohort base, then remove the dataset from memory.  Once all timestamps have been added,
duplicates are removed, the dataframe is sorted by ed_stay_id then time, and a unique step_idx is added for each ed_stay grouping.  The observed results of the actions are added back in
downstream steps"""

import gc
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def collect_times(df: pd.DataFrame, stay_col: str, time_col: str) -> pd.DataFrame:
    """Extract (ed_stay_id, event_time) pairs from a dataset, dropping nulls."""
    return (
        df[[stay_col, time_col]]
        .rename(columns={stay_col: 'ed_stay_id', time_col: 'event_time'})
        .dropna(subset=['event_time'])
        .copy()
    )


def collect_event_times(src_repo: str, hf_cfg: dict) -> list[pd.DataFrame]:
    """
    Load each interim dataset, extract event timestamps, delete immediately.
    Returns a list of (ed_stay_id, event_time) DataFrames -- one per source/time column.
    """
    frames = []

    logger.info('Collecting lab timestamps...')
    _df = load_dataset(src_repo, name=hf_cfg['labs']['config_name'],
                       split=hf_cfg['labs']['split_name']).to_pandas()
    frames += [collect_times(_df, 'ed_stay_id', 'order_time'),
               collect_times(_df, 'ed_stay_id', 'result_time')]
    del _df; gc.collect()

    logger.info('Collecting microbiology timestamps...')
    _df = load_dataset(src_repo, name=hf_cfg['microbiology']['config_name'],
                       split=hf_cfg['microbiology']['split_name']).to_pandas()
    frames += [collect_times(_df, 'ed_stay_id', 'charttime'),
               collect_times(_df, 'ed_stay_id', 'storetime')]
    del _df; gc.collect()

    logger.info('Collecting medication timestamps (Administered only)...')
    _df = load_dataset(src_repo, name=hf_cfg['dispensed_meds']['config_name'],
                       split=hf_cfg['dispensed_meds']['split_name']).to_pandas()
    _df = _df[_df['event_txt'] == 'Administered']
    frames.append(collect_times(_df, 'ed_stay_id', 'charttime'))
    del _df; gc.collect()

    logger.info('Collecting ECG timestamps...')
    _df = load_dataset(src_repo, name=hf_cfg['ecg']['config_name'],
                       split=hf_cfg['ecg']['split_name']).to_pandas()
    _df = _df.rename(columns={'stay_id': 'ed_stay_id'})
    frames.append(collect_times(_df, 'ed_stay_id', 'ecg_time'))
    del _df; gc.collect()

    logger.info('Collecting radiology timestamps...')
    _df = load_dataset(src_repo, name=hf_cfg['radiology']['config_name'],
                       split=hf_cfg['radiology']['split_name']).to_pandas()
    _df = _df.rename(columns={'stay_id': 'ed_stay_id'})
    frames.append(collect_times(_df, 'ed_stay_id', 'exam_time'))
    del _df; gc.collect()

    logger.info('Collecting triage/vitals timestamps...')
    _df = load_dataset(src_repo, name=hf_cfg['triage_vitals']['config_name'],
                       split=hf_cfg['triage_vitals']['split_name']).to_pandas()
    frames.append(collect_times(_df, 'ed_stay_id', 'charttime'))
    del _df; gc.collect()

    logger.info(f'Timestamp collection complete -- {len(frames)} frame(s) collected.')
    return frames


def build_step_index(frames: list[pd.DataFrame], cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Union all event timestamps with stay window boundaries, deduplicate, sort,
    and assign a globally unique step_idx (= row position).
    Returns df_patient with cohort base columns merged in.
    """
    # Include stay window boundaries so every stay has at least 2 steps
    boundary_frames = [
        cohort[['ed_stay_id', 'stay_window_start']].rename(
            columns={'stay_window_start': 'event_time'}),
        cohort[['ed_stay_id', 'stay_window_end']].rename(
            columns={'stay_window_end': 'event_time'}),
    ]

    all_events = pd.concat(boundary_frames + frames)
    all_events['event_time'] = pd.to_datetime(all_events['event_time']).dt.tz_localize(None)
    del frames; gc.collect()

    valid_stays = set(cohort['ed_stay_id'])
    step_index = (
        all_events[all_events['ed_stay_id'].isin(valid_stays)]
        .drop_duplicates()
        .sort_values(['ed_stay_id', 'event_time'])
        .reset_index(drop=True)
        .rename(columns={'event_time': 'time'})
    )
    step_index['step_idx'] = step_index.index   # globally unique row position
    del all_events; gc.collect()

    df_patient = step_index.merge(cohort, on='ed_stay_id', how='left')
    del step_index; gc.collect()

    logger.info(f'Step index built -- shape: {df_patient.shape}, '
                f'unique stays: {df_patient["ed_stay_id"].nunique():,}, '
                f'avg steps/stay: {len(df_patient) / df_patient["ed_stay_id"].nunique():.1f}')
    return df_patient


def drop_out_of_window(df_patient: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where time falls outside [stay_window_start, stay_window_end]."""
    before_mask = df_patient['time'] < df_patient['stay_window_start']
    after_mask  = df_patient['time'] > df_patient['stay_window_end']

    n_before = before_mask.sum()
    n_after  = after_mask.sum()

    if n_before > 0:
        logger.warning(f'[DROP] {n_before:,} rows before stay_window_start')
    if n_after > 0:
        logger.warning(f'[DROP] {n_after:,} rows after stay_window_end')

    df_patient = df_patient[~before_mask & ~after_mask].reset_index(drop=True)
    df_patient['step_idx'] = df_patient.index   # reassign after drop

    assert (df_patient['time'] >= df_patient['stay_window_start']).all()
    assert (df_patient['time'] <= df_patient['stay_window_end']).all()
    logger.info(f'[PASS] All times within stay window -- shape: {df_patient.shape}')
    return df_patient


def drop_single_step_stays(df_patient: pd.DataFrame) -> pd.DataFrame:
    """Remove stays that have only 1 step (need >= 2 for sequence modeling)."""
    step_counts = df_patient.groupby('ed_stay_id').size()
    single_step = step_counts[step_counts < 2].index

    if len(single_step) > 0:
        logger.warning(f'[DROP] {len(single_step):,} stays with only 1 step')
        df_patient = (df_patient[~df_patient['ed_stay_id'].isin(single_step)]
                      .reset_index(drop=True))
        df_patient['step_idx'] = df_patient.index

    step_counts = df_patient.groupby('ed_stay_id').size()
    logger.info(f'Step counts -- min: {step_counts.min()}, '
                f'max: {step_counts.max()}, median: {step_counts.median():.0f}')
    return df_patient
