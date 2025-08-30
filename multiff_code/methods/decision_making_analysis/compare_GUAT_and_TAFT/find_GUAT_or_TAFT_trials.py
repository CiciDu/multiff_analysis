from decision_making_analysis.GUAT import GUAT_utils

import os
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def give_up_after_trying_func(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, max_point_index=None, max_cluster_distance=50):
    """
    Find the trials where the monkey has stopped more than once to catch a firefly but failed to succeed, and the monkey gave up.
    """

    GUAT_trials_df = make_GUAT_trials_df(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)
    GUAT_indices_df, GUAT_point_indices_for_anim = _get_GUAT_or_TAFT_info(
        GUAT_trials_df, monkey_information, max_point_index=max_point_index)

    GUAT_w_ff_df, GUAT_expanded_trials_df = GUAT_utils.get_GUAT_w_ff_df(GUAT_indices_df,
                                                                        GUAT_trials_df,
                                                                        ff_dataframe,
                                                                        monkey_information,
                                                                        ff_real_position_sorted,
                                                                        )
    give_up_after_trying_trials = GUAT_expanded_trials_df['trial'].values

    # only keep the GUAT_indices_df in GUAT_w_ff_df
    GUAT_trials_df = GUAT_trials_df[GUAT_trials_df['cluster_index'].isin(
        GUAT_w_ff_df['cluster_index'].unique())]
    GUAT_indices_df = GUAT_indices_df[GUAT_indices_df['cluster_index'].isin(
        GUAT_w_ff_df['cluster_index'].unique())]
    GUAT_point_indices_for_anim = only_get_point_indices_for_anim(
        GUAT_trials_df, monkey_information, max_point_index=None)

    return give_up_after_trying_trials, GUAT_indices_df, GUAT_trials_df, GUAT_point_indices_for_anim, GUAT_w_ff_df


def try_a_few_times_func(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_point_index=None, max_cluster_distance=50):
    """
    Find the trials where the monkey has stopped more than one times to catch a target
    """

    TAFT_trials_df = make_TAFT_trials_df(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)
    try_a_few_times_trials = TAFT_trials_df['trial'].unique()
    TAFT_indices_df, try_a_few_times_indices_for_anim = _get_GUAT_or_TAFT_info(
        TAFT_trials_df, monkey_information, max_point_index=max_point_index)
    return try_a_few_times_trials, TAFT_indices_df, TAFT_trials_df, try_a_few_times_indices_for_anim


def _get_GUAT_or_TAFT_info(trials_df, monkey_information, max_point_index=None):

    # Initialize lists to store indices
    point_indices = []
    indices_corr_trials = []
    indices_corr_clusters = []
    point_indices_for_anim = []

    # Iterate over the rows of trials_df
    for _, row in trials_df.iterrows():
        first_stop_point_index = row['first_stop_point_index']
        last_stop_point_index = row['last_stop_point_index']
        indices_to_add = list(
            range(first_stop_point_index, last_stop_point_index))

        point_indices.extend(indices_to_add)
        indices_corr_trials.extend([row['trial']] * len(indices_to_add))
        indices_corr_clusters.extend(
            [row['cluster_index']] * len(indices_to_add))
        point_indices_for_anim.extend(
            range(first_stop_point_index - 20, last_stop_point_index + 21))

    if max_point_index is None:
        max_point_index = monkey_information['point_index'].max()

    # Convert lists to numpy arrays
    point_indices = np.array(point_indices)
    indices_corr_trials = np.array(indices_corr_trials)
    indices_corr_clusters = np.array(indices_corr_clusters)
    point_indices_for_anim = np.unique(np.array(point_indices_for_anim))

    # Filter indices based on max_point_index
    indices_to_keep = point_indices < max_point_index
    indices_df = pd.DataFrame({
        'point_index': point_indices[indices_to_keep],
        'trial': indices_corr_trials[indices_to_keep],
        'cluster_index': indices_corr_clusters[indices_to_keep]
    })

    point_indices_for_anim = point_indices_for_anim[point_indices_for_anim < max_point_index]
    return indices_df, point_indices_for_anim


def only_get_point_indices_for_anim(trials_df, monkey_information, max_point_index=None):
    point_indices_for_anim = []
    for _, row in trials_df.iterrows():
        first_stop_point_index = row['first_stop_point_index']
        last_stop_point_index = row['last_stop_point_index']
        point_indices_for_anim.extend(
            range(first_stop_point_index - 20, last_stop_point_index + 21))

    if max_point_index is None:
        max_point_index = monkey_information['point_index'].max()

    point_indices_for_anim = np.unique(np.array(point_indices_for_anim))
    point_indices_for_anim = point_indices_for_anim[point_indices_for_anim < max_point_index]
    return point_indices_for_anim


def make_GUAT_trials_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_GUAT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)

    GUAT_trials_df = _make_trials_df(monkey_sub)

    GUAT_trials_df.reset_index(drop=True, inplace=True)

    return GUAT_trials_df


def make_TAFT_trials_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)

    TAFT_trials_df = _make_trials_df(monkey_sub)

    TAFT_trials_df.reset_index(drop=True, inplace=True)

    return TAFT_trials_df


def _make_trials_df(monkey_sub):

    trials_df = monkey_sub[['stop_cluster_id', 'trial']
                           ].drop_duplicates().reset_index(drop=True)

    # Calculate the number of stops for each cluster
    trials_df['num_stops'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial']).size().values

    # only keep the trials with more than one stop
    trials_df = trials_df[trials_df['num_stops'] > 1].copy()
    monkey_sub = monkey_sub.merge(trials_df[['stop_cluster_id', 'trial']], on=[
                                  'stop_cluster_id', 'trial'], how='inner')

    # Get stop indices for each cluster
    trials_df['stop_indices'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])[
        'point_index'].apply(list).values

    # Assign cluster indices
    trials_df['cluster_index'] = np.arange(len(trials_df))

    # Get first, second, and last stop point indices for each cluster
    trials_df['first_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].first().values
    trials_df['second_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].nth(1).values
    trials_df['last_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].last().values

    # Get stop times for first, second, and last stops
    monkey_sub.set_index('point_index', inplace=True)
    trials_df['first_stop_time'] = monkey_sub.loc[trials_df['first_stop_point_index'], 'time'].values
    trials_df['second_stop_time'] = monkey_sub.loc[trials_df['second_stop_point_index'], 'time'].values
    trials_df['last_stop_time'] = monkey_sub.loc[trials_df['last_stop_point_index'], 'time'].values

    return trials_df


def _take_out_monkey_subset_for_GUAT(
    monkey_information: pd.DataFrame,
    ff_caught_T_new: np.ndarray,
    ff_real_position_sorted: np.ndarray,  # shape (n_trials, 2) → [[x, y], ...]
    max_cluster_distance: float = 50,
    far_distance_fill: float = 500.0,
) -> pd.DataFrame:
    """
    Extract a subset of monkey information for GUAT analysis.
    Assumes `trial` is 0-based.
    """
    # Preconditions
    if ff_real_position_sorted.ndim != 2 or ff_real_position_sorted.shape[1] != 2:
        raise ValueError(
            "ff_real_position_sorted must be (n_trials, 2) [x, y].")
    required_cols = {"trial", "monkey_x", "monkey_y"}
    missing = required_cols - set(monkey_information.columns)
    if missing:
        raise KeyError(
            f"monkey_information missing columns: {sorted(missing)}")

    # 1) Add stop cluster IDs
    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance)

    # 2) Base subset (copy to avoid chained assignment)
    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted
    ).copy()

    # 3) Trials (ensure ints, 0-based)
    trials = monkey_sub["trial"].to_numpy()
    if not np.issubdtype(trials.dtype, np.integer):
        if np.any(~np.isfinite(trials)):
            raise ValueError(
                "monkey_sub['trial'] contains NaN/inf; cannot index targets.")
        trials = trials.astype(int)

    n_trials = ff_real_position_sorted.shape[0]

    def safe_pick(idx: np.ndarray) -> np.ndarray:
        """Return [x,y] rows for valid indices; invalid rows become NaN."""
        out = np.full((idx.shape[0], 2), np.nan, dtype=float)
        valid = (idx >= 0) & (idx < n_trials)
        out[valid] = ff_real_position_sorted[idx[valid]]
        return out

    # --- 0-based trial indexing ---
    current_idx = trials
    last_idx = trials - 1
    last_last_idx = trials - 2
    next_idx = trials + 1
    next_next_idx = trials + 2

    last_xy = safe_pick(last_idx)
    last_last_xy = safe_pick(last_last_idx)
    current_xy = safe_pick(current_idx)
    next_xy = safe_pick(next_idx)
    next_next_xy = safe_pick(next_next_idx)

    monkey_sub[["last_target_x", "last_target_y"]] = last_xy
    monkey_sub[["last_last_target_x", "last_last_target_y"]] = last_last_xy
    monkey_sub[["current_target_x", "current_target_y"]] = current_xy
    monkey_sub[["next_target_x", "next_target_y"]] = next_xy
    monkey_sub[["next_next_target_x", "next_next_target_y"]] = next_next_xy

    # 5) Distances
    mx = monkey_sub["monkey_x"].to_numpy()
    my = monkey_sub["monkey_y"].to_numpy()

    monkey_sub["distance_to_last_target"] = np.hypot(
        mx - last_xy[:, 0],       my - last_xy[:, 1])
    monkey_sub["distance_to_last_last_target"] = np.hypot(
        mx - last_last_xy[:, 0],  my - last_last_xy[:, 1])
    monkey_sub["distance_to_target"] = np.hypot(
        mx - current_xy[:, 0],    my - current_xy[:, 1])
    monkey_sub["distance_to_next_target"] = np.hypot(
        mx - next_xy[:, 0],       my - next_xy[:, 1])
    monkey_sub["distance_to_next_next_target"] = np.hypot(
        mx - next_next_xy[:, 0],  my - next_next_xy[:, 1])

    # Fill NaNs from out-of-bounds picks
    for col in [
        "distance_to_last_target",
        "distance_to_last_last_target",
        "distance_to_target",
        "distance_to_next_target",
        "distance_to_next_next_target",
    ]:
        monkey_sub[col] = monkey_sub[col].fillna(far_distance_fill)

    # 7) Filter: clusters too close to any target
    dcols = [
        "distance_to_target",
        "distance_to_last_target",
        "distance_to_last_last_target",
        "distance_to_next_target",
        "distance_to_next_next_target",
    ]
    close_mask = (monkey_sub[dcols] < max_cluster_distance).any(axis=1)
    close_to_target_clusters = monkey_sub.loc[close_mask, "stop_cluster_id"].unique(
    )

    print(
        f"When taking out monkey subset for GUAT, "
        f"{len(close_to_target_clusters)} clusters out of "
        f"{monkey_sub['stop_cluster_id'].nunique()} are too close to a target "
        f"(threshold={max_cluster_distance}) and are filtered out."
    )

    monkey_sub = monkey_sub[~monkey_sub["stop_cluster_id"].isin(
        close_to_target_clusters)].copy()

    # 8) Filter out clusters that span multiple trials
    counts = (
        monkey_sub[["stop_cluster_id", "trial"]]
        .drop_duplicates()
        .groupby("stop_cluster_id")
        .size()
    )
    single_trial_clusters = counts[counts == 1].index
    monkey_sub = monkey_sub[monkey_sub["stop_cluster_id"].isin(
        single_trial_clusters)].copy()

    # 9) Sort and reset index
    if "point_index" in monkey_sub.columns:
        monkey_sub = monkey_sub.sort_values("point_index", kind="mergesort")
    monkey_sub = monkey_sub.reset_index(drop=True)

    return monkey_sub


def _take_out_monkey_subset_to_get_num_stops_near_target(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):
    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance, use_ff_caught_time_new_to_separate_clusters=True)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                                          min_stop_per_cluster=1)
    # Keep clusters that are close to the targets
    monkey_sub = _keep_clusters_close_to_target(
        monkey_sub, max_cluster_distance)

    # For each trial, keep the latest stop cluster
    monkey_sub = _keep_latest_cluster_for_each_trial(monkey_sub)

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)
    return monkey_sub


def _take_out_monkey_subset_for_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):

    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance, use_ff_caught_time_new_to_separate_clusters=True)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted)

    # Keep clusters that are close to the targets
    monkey_sub = _keep_clusters_close_to_target(
        monkey_sub, max_cluster_distance)

    # For each trial, keep the latest stop cluster
    monkey_sub = _keep_latest_cluster_for_each_trial(monkey_sub)

    # if two trials share the same stop cluster, then keep the trial with the smaller trial number
    monkey_sub.sort_values(by=['stop_cluster_id', 'trial'], inplace=True)
    unique_combo_to_keep = monkey_sub.groupby(
        'stop_cluster_id')['trial'].first().reset_index(drop=False)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'stop_cluster_id', 'trial'], how='inner')

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)

    return monkey_sub


def _keep_clusters_close_to_target(monkey_sub, max_cluster_distance=50):
    close_to_target_clusters = monkey_sub[(
        monkey_sub['distance_to_target'] < max_cluster_distance)]['stop_cluster_id'].unique()
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(
        close_to_target_clusters)].copy()
    return monkey_sub


def _keep_latest_cluster_for_each_trial(monkey_sub):
    monkey_sub.sort_values(by=['trial', 'stop_cluster_id'], inplace=True)
    unique_combo_to_keep = monkey_sub[[
        'trial', 'stop_cluster_id']].groupby('trial').tail(1)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'trial', 'stop_cluster_id'], how='inner')
    return monkey_sub


def _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                             min_stop_per_cluster=2):

    # Filter for new distinct stops within the time range
    monkey_sub = monkey_information[monkey_information['whether_new_distinct_stop'] == True].copy(
    )
    monkey_sub = monkey_sub[monkey_sub['time'].between(
        ff_caught_T_new[0], ff_caught_T_new[-1])]
    # Assign trial numbers and target positions
    monkey_sub[['target_x', 'target_y']
               ] = ff_real_position_sorted[monkey_sub['trial'].values]
    # Calculate distances to targets
    monkey_sub['distance_to_target'] = np.sqrt(
        (monkey_sub['monkey_x'] - monkey_sub['target_x'])**2 + (monkey_sub['monkey_y'] - monkey_sub['target_y'])**2)

    # Find clusters with more than one stop
    cluster_counts = monkey_sub['stop_cluster_id'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >=
                                    min_stop_per_cluster].index
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(valid_clusters)]

    return monkey_sub


import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def add_stop_cluster_id(
    monkey_information: pd.DataFrame,
    max_cluster_distance=50,   # cm
    use_ff_caught_time_new_to_separate_clusters: bool = False,
    ff_caught_times: np.ndarray | None = None,   # seconds, sorted
    capture_split_window_s: float = 0.3,         # seconds
    stop_id_col: str = "stop_id",
    stop_start_flag_col: str = "whether_new_distinct_stop",
    cumdist_col: str = "cum_distance",           # cm
    time_col: str = "time",
    point_index_col: str = "point_index",
    col_exists_ok: bool = True,
) -> pd.DataFrame:
    """
    Assign stop clusters based on cumulative distance between consecutive stops (in cm).
    Adds:
      - stop_cluster_id (Int64; <NA> on non-stop rows)
      - stop_cluster_start_point, stop_cluster_end_point
      - stop_cluster_size (Int64): number of stops in the cluster
    """
    df = monkey_information.copy()

    # If caller is OK with existing column, return early
    if 'stop_cluster_id' in df.columns and col_exists_ok:
        print("stop_cluster_id column already exists in the dataframe, skipping the addition of stop_cluster_id column")
        return df

    # Clean up prior cluster columns to avoid merge suffixes
    df = df.drop(columns=[
        "stop_cluster_id",
        "stop_cluster_start_point",
        "stop_cluster_end_point",
        "stop_cluster_size",
    ], errors="ignore")

    # Required columns present?
    needed = {stop_start_flag_col, stop_id_col, point_index_col, cumdist_col, time_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One row per stop onset
    stop_rows = df.loc[df[stop_start_flag_col]].copy()
    if stop_rows.empty:
        df["stop_cluster_id"] = pd.array([pd.NA]*len(df), dtype="Int64")
        df["stop_cluster_start_point"] = pd.NA
        df["stop_cluster_end_point"] = pd.NA
        df["stop_cluster_size"] = pd.array([pd.NA]*len(df), dtype="Int64")
        return df

    stop_table = (
        stop_rows.sort_values(stop_id_col)[[stop_id_col, point_index_col, cumdist_col, time_col]]
        .rename(columns={
            point_index_col: "stop_point_index",
            cumdist_col: "stop_cumdist_cm",
            time_col: "stop_time_s"
        })
        .reset_index(drop=True)
    )

    if stop_table[stop_id_col].duplicated().any():
        raise ValueError("Duplicate stop_id among stop starts; expected unique stop_ids.")

    # Distance-based split
    d_cum_cm = np.diff(stop_table["stop_cumdist_cm"].to_numpy())
    new_cluster = np.r_[True, d_cum_cm > float(max_cluster_distance)]

    # Optional capture-based split
    if use_ff_caught_time_new_to_separate_clusters and ff_caught_times is not None and len(ff_caught_times) > 0:
        caps = np.asarray(ff_caught_times, dtype=float)
        if not np.all(np.diff(caps) >= 0):
            caps = np.sort(caps)
        if len(stop_table) >= 2:
            t_prev = stop_table["stop_time_s"].to_numpy()[:-1]
            t_next = stop_table["stop_time_s"].to_numpy()[1:]
            idx = np.searchsorted(caps, t_prev, side="right")
            has_cap = np.zeros_like(t_prev, dtype=bool)
            valid = idx < caps.size
            has_cap[valid] = (caps[idx[valid]] >= (t_prev[valid] - capture_split_window_s)) & \
                             (caps[idx[valid]] <= (t_next[valid] + capture_split_window_s))
            new_cluster = np.r_[True, (d_cum_cm > float(max_cluster_distance)) | has_cap]

    # Assign cluster ids per stop
    stop_table["stop_cluster_id"] = np.cumsum(new_cluster.astype(np.int64)) - 1

    # Per-cluster stats (ensure key remains a column)
    bounds = (
        stop_table.groupby("stop_cluster_id", sort=True, as_index=False)
        .agg(
            stop_cluster_start_point=("stop_point_index", "min"),
            stop_cluster_end_point=("stop_point_index", "max"),
            stop_cluster_size=("stop_point_index", "count"),
        )
    )

    # Map cluster id to all samples via stop_id (left join keeps non-stop rows)
    df = df.merge(
        stop_table[[stop_id_col, "stop_cluster_id"]],
        on=stop_id_col,
        how="left"
    )

    # Attach bounds (per cluster), including size
    df = df.merge(bounds, on="stop_cluster_id", how="left")

    # Final dtypes
    df["stop_cluster_id"] = df["stop_cluster_id"].astype("Int64")
    if "stop_cluster_size" in df.columns:
        df["stop_cluster_size"] = df["stop_cluster_size"].astype("Int64")

    return df



def further_identify_cluster_start_and_end_based_on_ff_capture_time(stop_points_df):

    stop_points_df = stop_points_df.sort_values(by='point_index')
    # find the point index that has marked a new trial compared to previous point idnex
    stop_points_df['new_trial'] = stop_points_df['trial'].diff().fillna(1)

    # print the number of new trials
    print(
        f'The number of new trials that are used to separate stop clusters is {stop_points_df["new_trial"].sum().astype(int)}')

    # Mark those points as cluster_start, and the points after as cluster_end
    stop_points_df.reset_index(drop=True, inplace=True)
    index_to_mark_as_end = stop_points_df[stop_points_df['new_trial']
                                          == 1].index.values
    stop_points_df.loc[index_to_mark_as_end, 'cluster_end'] = True
    index_to_mark_as_start = index_to_mark_as_end + 1
    index_to_mark_as_start = index_to_mark_as_start[index_to_mark_as_start < len(
        stop_points_df)]
    stop_points_df.loc[index_to_mark_as_start, 'cluster_start'] = True

    # check correctness
    if (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() > 1) | \
            (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() < 0):
        raise ValueError(
            'The number of cluster start and end points are not the same')

    return stop_points_df
