import numpy as np
import pandas as pd


def bootstrap_trials(x_df, y_df, stratify_col, random_state=None):
    """
    Trial-level stratified bootstrap.
    Samples rows with replacement, preserving class proportions in stratify_col.
    """
    rng = np.random.default_rng(random_state)

    boot_idx = []
    for val in y_df[stratify_col].unique():
        idx = np.where(y_df[stratify_col].values == val)[0]
        sampled = rng.choice(idx, size=len(idx), replace=True)
        boot_idx.append(sampled)

    boot_idx = np.concatenate(boot_idx) if len(
        boot_idx) else np.array([], dtype=int)

    return (
        x_df.iloc[boot_idx].reset_index(drop=True),
        y_df.iloc[boot_idx].reset_index(drop=True),
    )


def get_condition_values(y_df: pd.DataFrame, condition_col: str):
    """
    Return ordered condition values if categorical, else sorted uniques.
    """
    col = y_df[condition_col]
    if hasattr(col, 'cat'):
        return list(col.cat.categories)
    return sorted(pd.unique(col))


