from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_one_stop_w_ff_df(stop_category_df):
    one_stop_sub = stop_category_df[stop_category_df['attempt_type'] == 'miss'].reset_index(drop=True)
    one_stop_w_ff_df = one_stop_sub[['time', 'point_index', 'stop_id', 'stop_cluster_id', 'stop_cluster_size', 'trial', 'target_index']].copy()
    one_stop_w_ff_df['latest_visible_ff'] = one_stop_sub['associated_ff'].astype(int)
    one_stop_w_ff_df['ff_index'] = one_stop_sub['associated_ff']
    one_stop_w_ff_df['num_stops'] = 1
    one_stop_w_ff_df['stop_time'] = one_stop_sub['time']
    one_stop_w_ff_df['first_stop_time'] = one_stop_sub['time']
    one_stop_w_ff_df['first_stop_point_index'] = one_stop_sub['point_index']
    return one_stop_w_ff_df

def make_temp_one_stop_w_ff_df(one_stop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one_stop_w_ff_df from a long one_stop_df (rows = point_index × ff candidates).
    - nearby_alive_ff_indices: unique, stably ordered list per point_index
    - latest_visible_ff: deterministic tie-break on time_since_last_vis, then ff_distance, then ff_index
    """

    # ensure required columns exist
    required = {"point_index", "time", "ff_index", "time_since_last_vis",
                "stop_id",}
    missing = required - set(one_stop_df.columns)
    if missing:
        raise KeyError(f"one_stop_df missing columns: {missing}")

    # sort to define deterministic tie-breaks:
    # 1) smallest time_since_last_vis (most recently seen)
    # 2) nearest ff_distance
    # 3) smallest ff_index
    sort_cols = [c for c in ["point_index", "time_since_last_vis",
                             "ff_distance", "ff_index"] if c in one_stop_df.columns]
    asc = [True] * len(sort_cols)
    df = one_stop_df.sort_values(
        sort_cols, ascending=asc, kind="mergesort").copy()

    # unique list of FFs per stop (keep stable order given by the sort above)
    ff_lists = (
        df[["point_index", "ff_index"]]
        .drop_duplicates(["point_index", "ff_index"])
        .groupby("point_index", sort=False)["ff_index"]
        .agg(list)
        .reset_index()
        .rename(columns={"ff_index": "nearby_alive_ff_indices"})
    )

    # latest_visible_ff = first row per point_index after the deterministic sort
    latest_ff = (
        df.groupby("point_index", sort=False, as_index=False)
          .first()[["point_index", "ff_index"]]
          .rename(columns={"ff_index": "latest_visible_ff"})
    )

    # bring in stop-level metadata (dedup by point_index)
    stop_meta_cols = [
        "target_index", "time", "point_index", "ff_distance",
        "distance_to_next_ff_capture", "min_distance_from_adjacent_stops",
        "stop_id",
    ]
    stop_meta_cols = [c for c in stop_meta_cols if c in df.columns]
    stop_meta = df[stop_meta_cols].drop_duplicates("point_index")

    # assemble
    out = (
        ff_lists
        .merge(latest_ff, on="point_index", how="left")
        .merge(stop_meta, on="point_index", how="left")
    )

    # add convenience columns
    out["num_stops"] = 1
    out["whether_w_ff_near_stops"] = 1
    if "target_index" in out.columns:
        out["trial"] = out["target_index"]

    # normalize dtypes
    for col in ["point_index", "target_index", "latest_visible_ff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(
                "Int64").astype("int64", errors="ignore")

    # add column names
    out['stop_time'] = out['time']
    out['first_stop_time'] = out['time']
    out['first_stop_point_index'] = out['point_index']

    # per your previous pattern
    out["stop_indices"] = out["first_stop_point_index"].apply(lambda x: [
                                                              int(x)])

    return out


# def get_distinct_stops_df(monkey_information, min_distance_between_distinct_stops=15):
#     # we need to get distinct stop point_index because sometimes a few point index can indicate the same stop

#     stop_points_df = monkey_information[monkey_information['monkey_speeddummy'] == 0].copy()

#     # take out stops that are not too close to previous stop points
#     stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance'].diff()
#     stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance_from_last_stop_point'].fillna(100)

#     distinct_stops_df = stop_points_df[stop_points_df['cum_distance_from_last_stop_point'] > min_distance_between_distinct_stops]

#     return distinct_stops_df


def filter_stops_by_distance_to_adjacent_stops(distinct_stops_df, min_distance_from_adjacent_stops):
    # take out stops that are not within min_distance_from_adjacent_stops of any other stop
    delta_x_from_last_stop = distinct_stops_df['monkey_x'].diff().fillna(
        min_distance_from_adjacent_stops * 2)
    delta_y_from_last_stop = distinct_stops_df['monkey_y'].diff().fillna(
        min_distance_from_adjacent_stops * 2)
    delta_x_from_next_stop = - \
        distinct_stops_df['monkey_x'].diff(-1).fillna(
            min_distance_from_adjacent_stops * 2)
    delta_y_from_next_stop = - \
        distinct_stops_df['monkey_y'].diff(-1).fillna(
            min_distance_from_adjacent_stops * 2)
    distinct_stops_df['distance_from_last_stop'] = np.sqrt(
        delta_x_from_last_stop ** 2 + delta_y_from_last_stop ** 2)
    distinct_stops_df['distance_from_next_stop'] = np.sqrt(
        delta_x_from_next_stop ** 2 + delta_y_from_next_stop ** 2)
    distinct_stops_df['min_distance_from_adjacent_stops'] = distinct_stops_df[[
        'distance_from_last_stop', 'distance_from_next_stop']].min(axis=1)
    filtered_stops_df = distinct_stops_df[(
        distinct_stops_df['min_distance_from_adjacent_stops'] > min_distance_from_adjacent_stops)].copy()

    return filtered_stops_df


def filter_stops_based_on_distance_to_ff_capture(
    filtered_stops_df: pd.DataFrame,
    monkey_information: pd.DataFrame,
    ff_caught_T_new: np.ndarray,
    min_cum_distance_to_ff_capture: float,
) -> pd.DataFrame:
    """
    Keep only stops whose cumulative distance is at least
    `min_cum_distance_to_ff_capture` away from the nearest firefly capture
    (both left and right in cumulative distance).

    Two-sided rule: nearest capture (either <= or >= stop_cum) is considered.
    """

    # Preconditions: ensure monotonic inputs
    time = monkey_information["time"].to_numpy()
    if not np.all(np.diff(time) >= 0):
        raise ValueError(
            "monkey_information['time'] must be sorted ascending.")

    if not np.all(np.diff(ff_caught_T_new) >= 0):
        ff_caught_T_new = np.sort(ff_caught_T_new)

    cumdist = monkey_information["cum_distance"].to_numpy()

    # Map capture times -> cumulative distance via interpolation
    capture_cumdist = np.interp(ff_caught_T_new, time, cumdist)

    # Stops cumulative distance
    stop_cum = filtered_stops_df["cum_distance"].to_numpy()

    # --- Right-side (next capture) ---
    idx_right = np.searchsorted(capture_cumdist, stop_cum, side="left")
    right_dist = np.full_like(stop_cum, np.inf, dtype=float)
    valid_r = idx_right < len(capture_cumdist)
    right_dist[valid_r] = capture_cumdist[idx_right[valid_r]] - \
        stop_cum[valid_r]

    # --- Left-side (previous capture) ---
    idx_left = idx_right - 1
    left_dist = np.full_like(stop_cum, np.inf, dtype=float)
    valid_l = idx_left >= 0
    left_dist[valid_l] = stop_cum[valid_l] - capture_cumdist[idx_left[valid_l]]

    # --- Two-sided distance ---
    nearest_dist = np.minimum(left_dist, right_dist)

    out = filtered_stops_df.copy()
    out["distance_to_nearest_ff_capture"] = nearest_dist

    # Keep stops far enough from *any* capture
    out = out[nearest_dist > min_cum_distance_to_ff_capture].copy()
    return out


def make_temp_one_stop_df(
    filtered_stops_df: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    ff_real_position_sorted,
    min_distance_from_ff: float = 25,
    max_distance_to_ff: float = 50,
    max_allowed_time_since_last_vis=3,
    eliminate_stops_too_close_to_any_target=True,
) -> pd.DataFrame:
    """
    Build a stop×FF table where:
      1) Any stop whose nearest FF is closer than `min_distance_from_ff` is eliminated entirely (because a distance smaller than 25 to a ff signifies a capture, not a miss).
      2) Remaining rows are only those FF within `max_distance_to_ff` of the stop.

    Returns a long table with one row per (stop point_index × nearby FF).
    """

    ff_dataframe = ff_dataframe[ff_dataframe['time_since_last_vis']
                                <= max_allowed_time_since_last_vis].copy()

    # also make sure that the ff is not the current or the previous target
    ff_dataframe = ff_dataframe[ff_dataframe['ff_index']
                                != ff_dataframe['target_index']]
    ff_dataframe = ff_dataframe[ff_dataframe['ff_index']
                                != ff_dataframe['target_index'] - 1].copy()

    # --- Select only needed columns for a tight merge ---
    stop_cols = [
        "point_index", "target_index", "monkey_x", "monkey_y", "time",
        "min_distance_from_adjacent_stops",
        "distance_to_next_ff_capture",
        "stop_id", "stop_cluster_id", "stop_cluster_size",
    ]
    stop_cols = [c for c in stop_cols if c in filtered_stops_df.columns]
    ff_cols = ["point_index", "ff_index", "ff_distance", "time_since_last_vis"]
    ff_cols = [c for c in ff_cols if c in ff_dataframe.columns]

    # Inner-join: we only care about stops that have at least one FF row
    merged = filtered_stops_df[stop_cols].merge(
        ff_dataframe[ff_cols], on="point_index", how="inner"
    )

    # Ensure ff_distance is numeric and drop NaNs (rows without a valid distance)
    merged = merged[pd.to_numeric(
        merged["ff_distance"], errors="coerce").notna()].copy()

    # --- Eliminate stops that are "too close" to any FF (< min_distance_from_ff) ---
    min_ff_per_stop = merged.groupby("point_index")[
        "ff_distance"].transform("min")
    keep_stops = min_ff_per_stop >= min_distance_from_ff

    # --- Keep only FF within the allowed max distance ---
    within_max = merged["ff_distance"] <= max_distance_to_ff

    one_stop_df = merged[keep_stops & within_max].copy()

    # ✅ ensure ff_distance is numeric and safe to compare
    one_stop_df["time_since_last_vis"] = pd.to_numeric(
        one_stop_df["time_since_last_vis"], errors="coerce").fillna(np.inf)
    one_stop_df["ff_distance"] = pd.to_numeric(
        one_stop_df["ff_distance"], errors="coerce").fillna(np.inf)

    # Optional: sort for stability
    sort_cols = [c for c in ["point_index", "time_since_last_vis",
                             "ff_distance"] if c in one_stop_df.columns]
    if sort_cols:
        one_stop_df.sort_values(sort_cols, inplace=True)

    # Normalize dtypes where helpful
    for col in ("point_index", "target_index", "ff_index"):
        if col in one_stop_df.columns:
            one_stop_df[col] = one_stop_df[col].astype(
                "int64", errors="ignore")

    return one_stop_df


def get_ff_info_for_GUAT(GUAT_indices_df,
                         GUAT_trials_df,
                         ff_dataframe,
                         monkey_information,
                         ff_real_position_sorted,
                         max_distance_to_stop_for_GUAT_target=50,
                         max_allowed_time_since_last_vis=3):
    """
    Process GUAT trials to add firefly context and filter for trials with nearby fireflies.

    This function takes the base GUAT_trials_df and expands it with firefly proximity information,
    then filters to keep only trials where fireflies are actually near the stops.

    Input DataFrames:
    - GUAT_trials_df: Base trials with cluster information (trial, cluster_index, stop times, etc.)
    - GUAT_indices_df: Point-by-point indices for each cluster

    Output DataFrames:
    - GUAT_ff_info: Filtered subset of trials that have fireflies near stops
    - GUAT_expanded_trials_df: All trials with firefly context added (unfiltered)

    Key Differences:
    - GUAT_trials_df: Base trials without firefly context
    - GUAT_expanded_trials_df: Base trials + firefly proximity annotations
    - GUAT_ff_info: Only trials where whether_w_ff_near_stops == 1
    """

    # Step 1: Extract point indices and cluster assignments
    GUAT_df = GUAT_indices_df[['point_index', 'temp_stop_cluster_id']].copy()

    # Step 2: Merge with firefly dataframe to get firefly context for each stop point
    GUAT_ff_info = GUAT_df.merge(ff_dataframe, on='point_index', how='left')

    # Step 3: Apply temporal filtering - only keep fireflies visible recently
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['time_since_last_vis']
                                <= max_allowed_time_since_last_vis]

    # Step 4: Apply spatial filtering - only keep fireflies close to monkey position
    # (within max_distance_to_stop_for_GUAT_target to the center of the firefly)
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['ff_distance']
                                < max_distance_to_stop_for_GUAT_target].copy()

    # also make sure that the ff is not the current or the previous target
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['ff_index']
                                != GUAT_ff_info['target_index']]
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['ff_index']
                                != GUAT_ff_info['target_index'] - 1].copy()

    # Step 5: Aggregate firefly information by cluster
    # Group by cluster_index so that ff_index becomes a list of ff_indices for each cluster
    GUAT_ff_info2 = GUAT_ff_info[['temp_stop_cluster_id', 'ff_index']].drop_duplicates(
    ).groupby('temp_stop_cluster_id')['ff_index'].apply(list).reset_index(drop=False)
    GUAT_ff_info2.rename(
        columns={'ff_index': 'nearby_alive_ff_indices'}, inplace=True)

    # Step 6: Find the most recently visible firefly for each cluster
    GUAT_ff_info.sort_values(by=['temp_stop_cluster_id', 'time_since_last_vis'], ascending=[
                             True, True], inplace=True)
    GUAT_ff_info2['latest_visible_ff'] = GUAT_ff_info.groupby('temp_stop_cluster_id')[
        'ff_index'].first().values

    # Step 7: Create GUAT_expanded_trials_df - merge base trials with firefly context
    # This adds firefly proximity information to all trials (unfiltered)
    GUAT_expanded_trials_df = GUAT_trials_df.merge(
        GUAT_ff_info2, on='temp_stop_cluster_id', how='left')
    GUAT_expanded_trials_df.sort_values(by='temp_stop_cluster_id', inplace=True)

    # Step 8: Add flag indicating whether fireflies are near stops
    # Mark whether_w_ff_near_stops as 1 if nearby_alive_ff_indices is not NA
    GUAT_expanded_trials_df['whether_w_ff_near_stops'] = (
        ~GUAT_expanded_trials_df['nearby_alive_ff_indices'].isna()).values.astype(int)

    # Step 9: Create GUAT_ff_info - filter to keep only trials with nearby fireflies
    # This is the filtered subset where whether_w_ff_near_stops == 1
    GUAT_ff_info = GUAT_expanded_trials_df[GUAT_expanded_trials_df['whether_w_ff_near_stops'] == 1].reset_index(
        drop=True)
    # Step 10: Final processing of GUAT_ff_info
    GUAT_ff_info['target_index'] = GUAT_ff_info['trial']
    GUAT_ff_info['latest_visible_ff'] = GUAT_ff_info['latest_visible_ff'].astype(
        'int64')

    GUAT_ff_info.sort_values(by=['trial', 'first_stop_time'], inplace=True)
    GUAT_ff_info['ff_index'] = GUAT_ff_info['latest_visible_ff']

    # Add stop point indices and handle duplicates
    GUAT_vs_TAFT_utils.add_stop_point_index(
        GUAT_ff_info, monkey_information, ff_real_position_sorted)
    # print('before calling deal_with_duplicated_stop_point_index, len(GUAT_ff_info)', len(
    #     GUAT_ff_info))
    GUAT_ff_info = GUAT_vs_TAFT_utils.deal_with_duplicated_stop_point_index(
        GUAT_ff_info)

    # print('after calling deal_with_duplicated_stop_point_index, len(GUAT_ff_info)', len(
    #     GUAT_ff_info))
    return GUAT_ff_info


def set_time_of_eval(GUAT_w_ff_df, monkey_information, time_with_respect_to_first_stop=None, time_with_respect_to_second_stop=None, time_with_respect_to_last_stop=None):
    GUAT_w_ff_df = GUAT_w_ff_df.copy()

    # make sure that only one of the three time_with_respect_to_* is not None
    if (time_with_respect_to_first_stop is not None) & (time_with_respect_to_second_stop is not None):
        raise ValueError(
            'Only one of the three time_with_respect_to_* can be not None.')
    if (time_with_respect_to_first_stop is not None) & (time_with_respect_to_last_stop is not None):
        raise ValueError(
            'Only one of the three time_with_respect_to_* can be not None.')
    if (time_with_respect_to_second_stop is not None) & (time_with_respect_to_last_stop is not None):
        raise ValueError(
            'Only one of the three time_with_respect_to_* can be not None.')

    if time_with_respect_to_first_stop is not None:
        time_of_eval = GUAT_w_ff_df['first_stop_time'] + \
            time_with_respect_to_first_stop
    elif time_with_respect_to_second_stop is not None:
        time_of_eval = GUAT_w_ff_df['second_stop_time'] + \
            time_with_respect_to_second_stop
    else:
        time_of_eval = GUAT_w_ff_df['last_stop_time'] + \
            time_with_respect_to_last_stop
    GUAT_w_ff_df['time_of_eval'] = time_of_eval
    GUAT_w_ff_df['point_index_of_eval'] = monkey_information['point_index'].values[np.searchsorted(
        monkey_information['time'].values, time_of_eval, side='right')-1]
    return GUAT_w_ff_df
