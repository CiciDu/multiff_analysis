import copy
import pandas as pd
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import smooth_neural_data
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import detrend_neural_data
import numpy as np


# ==========================================================
# VARIABLE CATEGORIES (DECODING)
# ==========================================================

ONE_FF_STYLE_DECODING_COLS = [
    'v', 'w', 'd', 'phi',
    'r_targ', 'theta_targ',
    'eye_ver', 'eye_hor',
    # though they are not in one_ff_gam, we add them to the design
    'accel', 'ang_accel',
    # though the following are not in one_ff_gam, they might be useful for multiff
    'time', 
    'eye_world_speed',
    'stop', 'capture',
]


DEFAULT_DECODING_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w", "accel", "ang_accel"],
    "latent_vars": ["cur_ff_distance", "cur_ff_angle", "nxt_ff_distance", "nxt_ff_angle"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["stop", "capture"],
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

FS_DECODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_DECODING_VAR_CATEGORIES)
FS_DECODING_VAR_CATEGORIES['event_vars'].extend(['ff_on_in_bin', 'ff_off_in_bin', 'group_ff_on_in_bin', 'group_ff_off_in_bin',
                                                 'ff_vis_start', 'ff_vis_end', 'global_burst_start'])

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

    unassigned_vars = sorted(all_vars - assigned)

    categories = categories.copy()
    categories['unassigned_vars'] = unassigned_vars

    return categories


def get_processed_spike_rates(
    spikes_df,
    smooth_spikes=True,
    detrend_spikes=False,
    smoothing_width=10,
    drop_bad_neurons=True,
    fs_bin_width=None,
):
    """
    Build full-session binned rates, optionally drop unstable units, then detrend or smooth.

    Parameters
    ----------
    fs_bin_width : float or None
        Bin width (s) for ``make_full_session_binned_spikes``. If None, uses 0.05 s when
        detrending (historical default) and 0.005 s when smoothing.
    drop_bad_neurons : bool
        If True, runs ``drop_nonstationary_neurons`` on the full-session rate matrix
        before detrending or smoothing.
    """
    if smooth_spikes and detrend_spikes:
        raise ValueError("smooth_spikes and detrend_spikes cannot both be True")

    if not detrend_spikes and not smooth_spikes:
        return None

    if fs_bin_width is None:
        bin_w = 0.05 if detrend_spikes else 0.005
    else:
        bin_w = float(fs_bin_width)

    fs_counts, time_bins_df = smooth_neural_data.make_full_session_binned_spikes(
        spikes_df, bin_width=bin_w
    )
    if fs_counts.empty or fs_counts.shape[1] == 0:
        return None

    fs_rates_hz = fs_counts / bin_w
    if drop_bad_neurons:
        fs_rates_hz = detrend_neural_data.drop_nonstationary_neurons(fs_rates_hz)

    if detrend_spikes:
        detrended_df = detrend_neural_data.detrend_spikes_session_wide(
            fs_rates_hz,
            time_bins_df,
            bin_size=bin_w,
            center_method='subtract',
        )
        processed_spike_rates, _cluster_columns = detrend_neural_data.reshape_detrended_df_to_wide(
            detrended_df,
            value_col='detrended_rate_hz',
        )
    else:
        smoothed_df, time_bins_df = smooth_neural_data.smooth_contiguous_spike_rates(
            fs_rates_hz, time_bins_df, width=smoothing_width
        )
        time_cols = ['time_bin_start', 'time_bin_end', 'time_bin_center']
        processed_spike_rates = smoothed_df.copy()
        processed_spike_rates[time_cols] = time_bins_df[time_cols]

    return processed_spike_rates


def _build_clean_var_categories(valid_vars, var_categories):
    """
    Core helper: filter var_categories using a provided set of valid_vars.
    """

    valid_vars = set(valid_vars)  # ensure set

    clean_var_categories = {}

    for category, var_list in var_categories.items():

        # skip non_spike_vars (we rebuild it later)
        if category == 'non_spike_vars':
            continue

        # filter
        filtered = [v for v in var_list if v in valid_vars]

        if filtered:
            clean_var_categories[category] = filtered

    # --- rebuild non_spike_vars ---
    all_non_spike = []
    for category, var_list in clean_var_categories.items():
        if category != 'spike_hist_vars':
            all_non_spike.extend(var_list)

    # preserve order (no sorting)
    seen = set()
    clean_var_categories['non_spike_vars'] = [
        v for v in all_non_spike if not (v in seen or seen.add(v))
    ]

    return clean_var_categories


def build_clean_var_categories_from_feats(feats_to_decode, var_categories):
    """
    Use feats_to_decode columns as source of valid vars.
    """

    return _build_clean_var_categories(feats_to_decode.columns, var_categories)