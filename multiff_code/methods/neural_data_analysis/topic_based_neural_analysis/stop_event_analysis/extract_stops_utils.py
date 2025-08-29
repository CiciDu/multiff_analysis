

from typing import Tuple, Callable, Optional
import pandas as pd
import numpy as np


def filter_no_capture_stops_vectorized(no_capture_stops_df, ff_caught_T_new, capture_match_window):
    """
    Filter out stops that are too close in time to any capture.

    Parameters
    ----------
    no_capture_stops_df : pd.DataFrame
        DataFrame containing stops that are *not* directly associated with captures.
        Must include a 'time' column.
    ff_caught_T_new : array-like
        Sorted or unsorted array of capture times.
    capture_match_window : float
        Minimum allowed time (in seconds) between a stop and the nearest capture.

    Returns
    -------
    df_filtered : pd.DataFrame
        Subset of no_capture_stops_df with stops kept only if they are at least
        `capture_match_window` away from the nearest capture.
    """

    # Ensure capture times are sorted ascending
    cap = np.sort(np.asarray(ff_caught_T_new))

    # Convert stop times to numpy array for vectorized operations
    t = no_capture_stops_df["time"].to_numpy()

    # For each stop time t, find the insertion point among sorted capture times.
    # idx[i] gives the index where t[i] would be inserted to keep 'cap' sorted.
    idx = np.searchsorted(cap, t, side="left")  # shape (N_stops,)

    # Initialize arrays of distances to left/right nearest capture times
    left_dt = np.full(t.shape, np.inf, dtype=float)
    right_dt = np.full(t.shape, np.inf, dtype=float)

    # Fill distances to the capture immediately before each stop
    valid_left = idx > 0
    left_dt[valid_left] = np.abs(t[valid_left] - cap[idx[valid_left] - 1])

    # Fill distances to the capture immediately after each stop
    valid_right = idx < cap.size
    right_dt[valid_right] = np.abs(cap[idx[valid_right]] - t[valid_right])

    # Minimum distance to *any* capture (either left or right)
    min_dt = np.minimum(left_dt, right_dt)

    # Keep stops whose minimum distance is at least the threshold
    keep_mask = min_dt >= capture_match_window

    # Return filtered DataFrame (reset index for cleanliness)
    df_filtered = no_capture_stops_df[keep_mask].reset_index(drop=True)
    return df_filtered


def filter_stops_df_by_debounce(stops_df, stop_debounce) -> pd.DataFrame:
    # Debounce: merge stops closer than cfg.stop_debounce
    if len(stops_df) > 1 and stop_debounce > 0:
        merged = [stops_df.iloc[0].to_dict()]
        for _, row in stops_df.iloc[1:].iterrows():
            if row["stop_time"] - merged[-1]["stop_time"] < stop_debounce:
                # keep the first one; alternatively average them
                continue
            merged.append(row.to_dict())
        stops_df = pd.DataFrame(merged)

    return stops_df.reset_index(drop=True)


def compute_stop_stats(monkey_information: pd.DataFrame) -> pd.DataFrame:
    """
    From per-sample `monkey_information`, compute one row per stop_id with duration and basic fields.

    Requires columns: ['point_index', 'time', 'stop_id'].
    Returns a DataFrame with unique stop_ids and columns:
      ['stop_id', 'point_index', 'time', 'stop_id_start_time', 'stop_id_end_time', 'stop_id_duration', ...original first-row cols]

    This function creates stops_with_stats: a comprehensive table of all stops with their temporal statistics.
    Each row represents one unique stop event with calculated start time, end time, and duration.
    """
    required = {"point_index", "time", "stop_id"}
    missing = required - set(monkey_information.columns)
    if missing:
        raise KeyError(
            f"compute_stop_stats: missing columns {sorted(missing)}")

    # Consider only rows that belong to a stop
    stops_df = monkey_information.loc[monkey_information["stop_id"].notna()].copy(
    )

    # Aggregate per stop_id over time
    stop_stats = stops_df.groupby("stop_id", as_index=True)["time"].agg(
        stop_id_start_time="min",
        stop_id_end_time="max"
    )
    stop_stats["stop_id_duration"] = (
        stop_stats["stop_id_end_time"] - stop_stats["stop_id_start_time"]
    )

    # Merge back; keep stable order by point_index (ascending)
    stops_df = stops_df.merge(stop_stats, on="stop_id", how="left")
    stops_df = stops_df.sort_values("point_index", kind="stable")

    # Reduce to one representative row per stop_id (the first encountered in time)
    unique_stops_df = (
        stops_df.groupby("stop_id", as_index=False, sort=False)
        .first()
        .reset_index(drop=True)
    )
    return unique_stops_df


def add_stop_id_to_closest_stop_to_capture_df(
    closest_stop_to_capture_df: pd.DataFrame,
    monkey_information: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce captures_df with stop_id resolved via a MERGE (safer than .loc with row positions).

    Requires:
      closest_stop_to_capture_df: columns ['cur_ff_index','time','point_index','stop_time', distance_col]
      monkey_information: columns ['point_index','stop_id']

    Returns columns (default): ['cur_ff_index','stop_id','time','point_index','stop_time', distance_col]
    """
    req1 = {"cur_ff_index", "time", "point_index", "stop_time"}
    req2 = {"point_index", "stop_id"}
    miss1 = req1 - set(closest_stop_to_capture_df.columns)
    miss2 = req2 - set(monkey_information.columns)
    if miss1:
        raise KeyError(
            f"add_stop_id_to_closest_stop_to_capture_df: closest_stop_to_capture_df missing {sorted(miss1)}")
    if miss2:
        raise KeyError(
            f"add_stop_id_to_closest_stop_to_capture_df: monkey_information missing {sorted(miss2)}")

    # Map stop_id via merge on point_index (robust even if index is not positional)
    map_df = monkey_information[["point_index", "stop_id"]].copy()
    closest_stop_to_capture_df = closest_stop_to_capture_df.merge(
        map_df, on="point_index", how="left")

    return closest_stop_to_capture_df


def prepare_no_capture_and_captures(
    monkey_information: pd.DataFrame,
    closest_stop_to_capture_df: pd.DataFrame,
    ff_caught_T_new: np.ndarray | pd.Series,
    *,
    min_stop_duration: float = 0.02,
    max_stop_duration: float = 1.0,
    capture_match_window: float = 0.3,
    stop_debounce: float = 0.1,
    distance_thresh: float = 25.0,
    distance_col: str = "distance_from_ff_to_stop",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end:
      1) Build captures_df with stop_id
      2) Build per-stop table + durations
      3) Derive no-capture stops (apply duration filters)
      4) Keep only captures within `distance_thresh`
      5) Vectorized temporal filtering of no-capture stops via user-provided function

    Returns:
      valid_captures_df, filtered_no_capture_stops_df, stops_with_stats

      Where:
      - valid_captures_df: Captures that occurred within distance_thresh of stops
      - filtered_no_capture_stops_df: Stops that didn't result in captures (filtered by duration and temporal proximity)
      - stops_with_stats: Complete table of all stops with temporal statistics (start/end times, duration)

    """
    # 1) Captures with stop_id
    if 'stop_id' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df = add_stop_id_to_closest_stop_to_capture_df(
            closest_stop_to_capture_df,
            monkey_information,
        )

    captures_df = closest_stop_to_capture_df[["cur_ff_index", "stop_id", "time",
                                              "point_index", "stop_time", 'distance_from_ff_to_stop']].copy()

    # 2) Per-stop stats
    # stops_with_stats: Comprehensive table of all stops with temporal statistics (start time, end time, duration)
    # Each row represents one unique stop event from the monkey_information data
    stops_with_stats = compute_stop_stats(monkey_information)

    # 3) No-capture stops: exclude any stop_id present in captures
    no_capture_stops_df = stops_with_stats.loc[
        ~stops_with_stats["stop_id"].isin(captures_df["stop_id"])
    ].reset_index(drop=True)

    # Duration filters
    no_capture_stops_df = no_capture_stops_df.loc[
        no_capture_stops_df["stop_id_duration"] >= min_stop_duration
    ].reset_index(drop=True)

    no_capture_stops_df = no_capture_stops_df.loc[
        no_capture_stops_df["stop_id_duration"] <= max_stop_duration
    ].reset_index(drop=True)

    # 4) Keep only “good” captures within spatial threshold
    valid_captures_df = captures_df.loc[
        pd.to_numeric(captures_df[distance_col],
                      errors="coerce") <= distance_thresh
    ].copy()
    valid_captures_df['stop_point_index'] = valid_captures_df['point_index']
    valid_captures_df['stop_time'] = valid_captures_df['time']

    # 5) Temporal filtering against capture times (optional)
    filtered_no_capture_stops_df = filter_no_capture_stops_vectorized(
        no_capture_stops_df, ff_caught_T_new, capture_match_window
    )
    
    filtered_no_capture_stops_df['stop_point_index'] = filtered_no_capture_stops_df['point_index']
    filtered_no_capture_stops_df['stop_time'] = filtered_no_capture_stops_df['time']
    filtered_no_capture_stops_df = filter_stops_df_by_debounce(
        filtered_no_capture_stops_df, stop_debounce
    )

    return valid_captures_df.reset_index(drop=True), filtered_no_capture_stops_df.reset_index(drop=True), stops_with_stats.reset_index(drop=True)
