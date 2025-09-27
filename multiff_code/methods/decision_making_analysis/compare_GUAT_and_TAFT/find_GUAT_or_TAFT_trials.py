from typing import Iterable, Optional, Tuple
from typing import Iterable, Tuple, Optional
from typing import Iterable, Tuple
from decision_making_analysis.GUAT import GUAT_utils
from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils

import os
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


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
            [row['temp_stop_cluster_id']] * len(indices_to_add))
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
        'temp_stop_cluster_id': indices_corr_clusters[indices_to_keep]
    })

    return indices_df


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


def make_TAFT_trials_df(stop_category_df):
    new_TAFT_df = stop_category_df.loc[stop_category_df['attempt_type'] == 'TAFT', [
        'stop_cluster_id', 'stop_cluster_size', 'trial', 'time', 'point_index', 'associated_ff']].copy()
    new_TAFT_df = new_TAFT_df.rename(columns={'associated_ff': 'ff_index'})

    TAFT_trials_df = _make_trials_df(new_TAFT_df)
    TAFT_trials_df.reset_index(drop=True, inplace=True)
    return TAFT_trials_df


def make_GUAT_trials_df(stop_category_df, ff_real_position_sorted, monkey_information):

    new_GUAT_df = stop_category_df.loc[stop_category_df['attempt_type'] == 'GUAT', [
        'stop_cluster_id', 'stop_cluster_size', 'trial', 'time', 'point_index', 'associated_ff']].copy()
    new_GUAT_df = new_GUAT_df.rename(columns={'associated_ff': 'ff_index'})

    GUAT_trials_df = _make_trials_df(new_GUAT_df)
    GUAT_trials_df.reset_index(drop=True, inplace=True)

    # also get GUAT_w_ff_df
    GUAT_w_ff_df = GUAT_trials_df.merge(new_GUAT_df[[
                                        'stop_cluster_id', 'ff_index']].drop_duplicates(), on='stop_cluster_id', how='left')
    GUAT_w_ff_df['ff_index'] = GUAT_w_ff_df['ff_index'].astype(int)

    GUAT_w_ff_df['latest_visible_ff'] = GUAT_w_ff_df['ff_index']
    GUAT_w_ff_df['cur_ff_index'] = GUAT_w_ff_df['ff_index']
    GUAT_w_ff_df[['cur_ff_x', 'cur_ff_y']
                 ] = ff_real_position_sorted[GUAT_w_ff_df['ff_index']]
    GUAT_w_ff_df['target_index'] = GUAT_trials_df['trial']

    GUAT_vs_TAFT_utils.add_stop_point_index(
        GUAT_w_ff_df, monkey_information, ff_real_position_sorted)

    # assert no duplicated ['cur_ff_index', 'stop_point_index']
    assert len(GUAT_w_ff_df[GUAT_w_ff_df.duplicated(
        subset=['cur_ff_index', 'stop_point_index'])]) == 0
    return GUAT_trials_df, GUAT_w_ff_df


def make_temp_TAFT_trials_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):
            
    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)

    TAFT_trials_df = _make_trials_df(monkey_sub, stop_cluster_id_col='temp_stop_cluster_id',
                                     stop_cluster_size_col='temp_stop_cluster_size')

    TAFT_trials_df.reset_index(drop=True, inplace=True)

    return TAFT_trials_df


def add_temp_stop_cluster_id(
    monkey_information: pd.DataFrame,
    ff_caught_T_new,
    max_cluster_distance=50,   # cm
    use_ff_caught_time_new_to_separate_clusters: bool = True,
    stop_id_col: str = "stop_id",
    stop_start_flag_col: str = "whether_new_distinct_stop",
    cumdist_col: str = "cum_distance",           # cm
    time_col: str = "time",
    point_index_col: str = "point_index",
    col_exists_ok: bool = False,
) -> pd.DataFrame:
    """
    Assign stop clusters based on cumulative distance between consecutive stops (in cm).
    Adds:
      - temp_stop_cluster_id (Int64; <NA> on non-stop rows)
      - temp_stop_cluster_start_point, temp_stop_cluster_end_point
      - temp_stop_cluster_size (Int64): number of stops in the cluster
    """
    df = monkey_information.copy()

    # If caller is OK with existing column, return early
    if 'temp_stop_cluster_id' in df.columns and col_exists_ok:
        print("temp_stop_cluster_id column already exists in the dataframe, skipping the addition of temp_stop_cluster_id column")
        return df

    # Clean up prior cluster columns to avoid merge suffixes
    df = df.drop(columns=[
        "temp_stop_cluster_id",
        "temp_stop_cluster_start_point",
        "temp_stop_cluster_end_point",
        "temp_stop_cluster_size",
    ], errors="ignore")

    # Required columns present?
    needed = {stop_start_flag_col, stop_id_col,
              point_index_col, cumdist_col, time_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One row per stop onset
    stop_rows = df.loc[df[stop_start_flag_col]].copy()
    if stop_rows.empty:
        df["temp_stop_cluster_id"] = pd.array([pd.NA]*len(df), dtype="Int64")
        df["temp_stop_cluster_start_point"] = pd.NA
        df["temp_stop_cluster_end_point"] = pd.NA
        df["temp_stop_cluster_size"] = pd.array([pd.NA]*len(df), dtype="Int64")
        return df

    stop_table = (
        stop_rows.sort_values(stop_id_col)[
            [stop_id_col, point_index_col, cumdist_col, time_col]]
        .rename(columns={
            point_index_col: "stop_point_index",
            cumdist_col: "stop_cumdist_cm",
            time_col: "stop_time_s"
        })
        .reset_index(drop=True)
    )

    if stop_table[stop_id_col].duplicated().any():
        raise ValueError(
            "Duplicate stop_id among stop starts; expected unique stop_ids.")

    # Distance-based split
    d_cum_cm = np.diff(stop_table["stop_cumdist_cm"].to_numpy())
    new_cluster = np.r_[True, d_cum_cm > float(max_cluster_distance)]

    # >>> OPTIONAL CAPTURE-BASED SPLIT (eps-shift assigns border to the NEXT gap) <<<
    if use_ff_caught_time_new_to_separate_clusters and ff_caught_T_new is not None and len(ff_caught_T_new) > 0:
        caps = np.asarray(ff_caught_T_new, dtype=float)
        if not np.all(np.diff(caps) >= 0):
            caps = np.sort(caps)

        if len(stop_table) >= 2:
            eps = 0.001  # seconds; tiny shift to resolve boundary cases deterministically

            # Consecutive stop times
            t_prev = stop_table['stop_time_s'].to_numpy()[:-1]
            t_next = stop_table['stop_time_s'].to_numpy()[1:]

            lo = np.searchsorted(caps, t_prev - eps, side='left')
            hi = np.searchsorted(caps, t_next - eps, side='left')
            has_cap_between = (hi > lo)

            # Start a new cluster at stop j if either distance jumps or a capture lies in its preceding gap
            new_cluster = np.r_[True, (d_cum_cm > float(
                max_cluster_distance)) | has_cap_between]

    # Assign cluster ids per stop
    stop_table["temp_stop_cluster_id"] = np.cumsum(new_cluster.astype(np.int64)) - 1

    # Per-cluster stats (ensure key remains a column)
    bounds = (
        stop_table.groupby("temp_stop_cluster_id", sort=True, as_index=False)
        .agg(
            temp_stop_cluster_start_point=("stop_point_index", "min"),
            temp_stop_cluster_end_point=("stop_point_index", "max"),
            temp_stop_cluster_size=("stop_point_index", "count"),
        )
    )

    # Map cluster id to all samples via stop_id (left join keeps non-stop rows)
    df = df.merge(
        stop_table[[stop_id_col, "temp_stop_cluster_id"]],
        on=stop_id_col,
        how="left"
    )

    # Attach bounds (per cluster), including size
    df = df.merge(bounds, on="temp_stop_cluster_id", how="left")

    # Final dtypes
    df['temp_stop_cluster_id'] = df["temp_stop_cluster_id"].astype("Int64")
    if "temp_stop_cluster_size" in df.columns:
        df["temp_stop_cluster_size"] = df["temp_stop_cluster_size"].astype("Int64")

    return df


def _make_trials_df(monkey_sub: pd.DataFrame, stop_cluster_id_col='stop_cluster_id',
                    stop_cluster_size_col='stop_cluster_size') -> pd.DataFrame:
    # Work on a sorted copy so first/second/last are well-defined
    ms = monkey_sub.sort_values(
        [stop_cluster_id_col, 'time'], kind='stable').copy()
    g = ms.groupby([stop_cluster_id_col, 'trial'], sort=False)

    _consistency = g[stop_cluster_size_col].nunique().rename(
        'nuniq').reset_index()
    if (_consistency['nuniq'] > 1).any():
        raise ValueError(
            f'{stop_cluster_size_col} varies within a cluster; cannot keep a single value.')

    agg_spec = {
        'num_stops': ('point_index', 'size'),
        'stop_indices': ('point_index', list),
        'first_stop_point_index': ('point_index', 'first'),
        'second_stop_point_index': ('point_index', lambda s: s.iloc[1] if len(s) > 1 else pd.NA),
        'last_stop_point_index': ('point_index', 'last'),
        'first_stop_time': ('time', 'first'),
        'second_stop_time': ('time', lambda s: s.iloc[1] if len(s) > 1 else pd.NA),
        'last_stop_time': ('time', 'last'),
        'stop_cluster_size': (stop_cluster_size_col, 'max'),  # keep it
    }

    trials_df = g.agg(**agg_spec).reset_index()
    trials_df = trials_df[trials_df['num_stops'] > 1].reset_index(drop=True)
    for col in ['first_stop_point_index', 'second_stop_point_index', 'last_stop_point_index']:
        trials_df[col] = trials_df[col].astype('Int64')

    return trials_df


def _take_out_monkey_subset_for_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=50):

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted)

    # Keep clusters that are close to the targets
    monkey_sub = _keep_clusters_close_to_target(
        monkey_sub, max_cluster_distance)

    # For each trial, keep the latest stop cluster if there are multiple stop clusters during the same trial close to the target; but this is unlikely
    monkey_sub = _keep_latest_cluster_for_each_trial(monkey_sub)

    # if two trials share the same stop cluster, then keep the trial with the smaller trial number
    # (Actually I don't know when this would happen. Need to verify later)
    monkey_sub.sort_values(by=['temp_stop_cluster_id', 'trial'], inplace=True)
    unique_combo_to_keep = monkey_sub.groupby(
        'temp_stop_cluster_id')['trial'].first().reset_index(drop=False)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'temp_stop_cluster_id', 'trial'], how='inner')

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)

    return monkey_sub


def _keep_clusters_close_to_target(monkey_sub, max_cluster_distance=50):
    close_to_target_stop_clusters = monkey_sub[(
        monkey_sub['distance_to_target'] < max_cluster_distance)]['temp_stop_cluster_id'].unique()
    monkey_sub = monkey_sub[monkey_sub['temp_stop_cluster_id'].isin(
        close_to_target_stop_clusters)].copy()
    return monkey_sub


def _keep_latest_cluster_for_each_trial(monkey_sub):
    monkey_sub.sort_values(by=['trial', 'temp_stop_cluster_id'], inplace=True)
    unique_combo_to_keep = monkey_sub[[
        'trial', 'temp_stop_cluster_id']].groupby('trial').tail(1)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'trial', 'temp_stop_cluster_id'], how='inner')
    return monkey_sub


def _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                             min_stop_per_cluster=2):
    """
    Extract a subset of monkey stop events for GUAT/TAFT analysis.

    This function filters the monkey's behavioral data to focus on valid stop 
    clusters within the time window of a firefly capture episode. It assigns 
    trial target positions, computes distance-to-target at each stop, and 
    removes clusters that do not meet a minimum number of stops.

    Steps performed:
    1. Keep only rows marked as new distinct stops within the capture timeframe.
    2. Attach target (firefly) positions based on trial indices.
    3. Compute Euclidean distance between monkey position and target position.
    4. Retain only stop clusters that have at least `min_stop_per_cluster` stops.

    """

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
    cluster_counts = monkey_sub['temp_stop_cluster_id'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >=
                                    min_stop_per_cluster].index
    monkey_sub = monkey_sub[monkey_sub['temp_stop_cluster_id'].isin(valid_clusters)]

    return monkey_sub



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


def _add_target_distances(
    monkey_sub: pd.DataFrame,
    ff_real_position_sorted: np.ndarray,  # shape: (n_trials, 2) -> [x, y]
    *,
    offsets: Iterable[int] = (-2, -1, 0, 1, 2),
    na_distance_fill: float = 500.0,
    trial_col: str = 'trial',
    pos_cols: Tuple[str, str] = ('monkey_x', 'monkey_y'),
) -> pd.DataFrame:
    '''
    For each row, compute distances to targets at trial offsets (e.g., {-2, -1, 0, +1, +2}).

    Adds only distance columns:
      distance_to_target_{+k}

    No XY columns are added.
    '''
    if ff_real_position_sorted.ndim != 2 or ff_real_position_sorted.shape[1] != 2:
        raise ValueError(
            'ff_real_position_sorted must be (n_trials, 2) [x, y].')

    trials = monkey_sub[trial_col].to_numpy()
    if not np.issubdtype(trials.dtype, np.integer):
        if np.any(~np.isfinite(trials)):
            raise ValueError(
                f"monkey_sub['{trial_col}'] contains NaN/inf; cannot index targets.")
        trials = trials.astype(int)

    n_trials = ff_real_position_sorted.shape[0]

    def _safe_pick(idx: np.ndarray) -> np.ndarray:
        out = np.full((idx.shape[0], 2), np.nan, dtype=float)
        valid = (idx >= 0) & (idx < n_trials)
        out[valid] = ff_real_position_sorted[idx[valid]]
        return out

    mx = monkey_sub[pos_cols[0]].to_numpy(dtype=float)
    my = monkey_sub[pos_cols[1]].to_numpy(dtype=float)

    for off in offsets:
        xy = _safe_pick(trials + off)
        dcol = f'distance_to_target_{off:+d}'
        d = np.hypot(mx - xy[:, 0], my - xy[:, 1]).astype(float)
        monkey_sub[dcol] = pd.Series(d).fillna(na_distance_fill).to_numpy()

    return monkey_sub
