import logging
import pandas as pd

logger = logging.getLogger(__name__)

def collapse_lab_rows(df: pd.DataFrame):
    """
    Collapse to one row per (ed_stay_id, category, fluid, order_time)
    If any lab in the group was flagged abnormal, the group result is abnormal
    result_time: take the latest (when all results in the batch are back)
    subject_id, hadm_id, ordered_location: same within a group, take first
    """
    logger.info("The following step groups labs and collapses them to one row so this process might take a few min to process")

    labs = (
    df.groupby(['ed_stay_id', 'category', 'fluid', 'order_time'], observed=True)
    .agg(
        subject_id=('subject_id', 'first'),
        hadm_id=('hadm_id', 'first'),
        result_time=('result_time', 'max'),
        ordered_location=('ordered_location', 'first'),
        abnormal=('flag', lambda x: x.notna().any())
    )
    .reset_index()
    )
    return labs

def actions_column(df: pd.DataFrame):
    """
    Create the actions column by combining the lab category and fluid.  This will create 19 separate actions that will be used to create
    columns in the patient state before modeling
    """
    df['action'] = df['category'] + "-" + df['fluid']
    return df