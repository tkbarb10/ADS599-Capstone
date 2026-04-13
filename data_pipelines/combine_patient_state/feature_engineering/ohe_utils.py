import pandas as pd


def merge_pivot_into_patient(
    df_patient: pd.DataFrame,
    pivot: pd.DataFrame,
    step_col: str = 'step_idx',
) -> pd.DataFrame:
    """
    Merge a pivot table back into df_patient on step_col using a left join.

    Avoids .loc-based assignment, which raises ValueError on fragmented DataFrames.
    Unmatched rows (step_idx not in pivot) keep their existing values unchanged.
    Calls .copy() at the end to defragment the block layout.

    Parameters
    ----------
    df_patient : DataFrame with a step_col column
    pivot      : pivot_table output with step_col as the index
    step_col   : join key (default 'step_idx')
    """
    ohe_cols = [c for c in pivot.columns if c in df_patient.columns]
    if not ohe_cols:
        return df_patient.copy()

    upd = pivot[ohe_cols].reset_index()   # step_idx becomes a plain column
    upd = upd.rename(columns={c: c + '__upd' for c in ohe_cols})

    df_patient = df_patient.reset_index(drop=True).merge(upd, on=step_col, how='left')
    for col in ohe_cols:
        df_patient[col] = df_patient[col + '__upd'].fillna(df_patient[col]).astype(int)
        df_patient.drop(columns=col + '__upd', inplace=True)

    return df_patient.copy()
