import copy
import pandas as pd
import numpy as np


# ==========================================================
# VARIABLE CATEGORIES (DECODING)
# ==========================================================

DEFAULT_DECODING_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w", "accel", "ang_accel"],
    "latent_vars": ["cur_ff_distance", "cur_ff_angle", "nxt_ff_distance", "nxt_ff_angle"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["stop"],
    "visibility_vars": ["cur_vis", "nxt_vis", "cur_in_memory", "nxt_in_memory"],
    "ff_count_vars": ["num_ff_visible", "num_ff_in_memory", "log1p_num_ff_visible", "log1p_num_ff_in_memory"],
}

# All non-spike variables (no spike history term on decoding side)
DEFAULT_DECODING_VAR_CATEGORIES["non_spike_vars"] = [
    var
    for category, vars_list in DEFAULT_DECODING_VAR_CATEGORIES.items()
    # keep everything; decoding design does not include 'spike_hist'
    for var in vars_list
]

STOP_DECODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_DECODING_VAR_CATEGORIES)
STOP_DECODING_VAR_CATEGORIES.update({
    "cluster_vars": [
        "event_is_first_in_cluster", "gap_since_prev", "gap_till_next",
        "cluster_duration", "cluster_progress", "bin_t_from_cluster",
        "log_n_events", "event_t_from_cluster",
    ],
})

VIS_DECODING_VAR_CATEGORIES = copy.deepcopy(STOP_DECODING_VAR_CATEGORIES)
VIS_DECODING_VAR_CATEGORIES.update({
    "visibility_vars": ["ff_on", "group_ff_on", "ff_off", "group_ff_off"],
})

PN_DECODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_DECODING_VAR_CATEGORIES)
PN_DECODING_VAR_CATEGORIES.update({
    "event_vars": ["cur_ff_on", "cur_ff_off"],
})


def clean_binary_and_drop_constant(df, tol=1e-8):
    df = df.copy()
    drop_cols = []

    for col in df.columns:
        s = df[col]
        vals = s.dropna().values

        # quantize to remove floating noise
        vals_q = np.round(vals / tol) * tol
        unique_vals = np.unique(vals_q)

        # constant column
        if unique_vals.size <= 1:
            drop_cols.append(col)
            continue

        # exactly two unique values → map to {0,1}
        if unique_vals.size == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}

            s_q = np.round(s / tol) * tol
            df[col] = s_q.map(mapping)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df

def truncate_columns_to_percentiles(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Truncate values in specified columns of a DataFrame to the 1st and 99th percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Column names to truncate.

    Returns
    -------
    pd.DataFrame
        DataFrame with truncated values.
    """
    df = df.copy()

    for col in columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def add_other_category_from_df(categories, design_df):
    all_vars = set(design_df.columns)

    assigned = set(
        v
        for cat, vars_list in categories.items()
        if cat != 'non_spike_vars'
        for v in vars_list
    )

    other_vars = sorted(all_vars - assigned)

    categories = categories.copy()
    categories['other_vars'] = other_vars

    return categories