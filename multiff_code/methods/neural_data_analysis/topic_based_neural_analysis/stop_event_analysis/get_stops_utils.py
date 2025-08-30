

from typing import Tuple, Callable, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.ndimage import gaussian_filter1d


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
      valid_captures_df, filtered_no_capture_stops_df, unique_stops_df

      Where:
      - valid_captures_df: Captures that occurred within distance_thresh of stops
      - filtered_no_capture_stops_df: Stops that didn't result in captures (filtered by duration and temporal proximity)
      - unique_stops_df: Complete table of all stops with temporal statistics (start/end times, duration)

    """
    # 1) Captures with stop_id
    if 'stop_id' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df = add_stop_id_to_closest_stop_to_capture_df(
            closest_stop_to_capture_df,
            monkey_information,
        )

    captures_df = closest_stop_to_capture_df[["cur_ff_index", "stop_id", "time",
                                              "point_index", "stop_time", 'distance_from_ff_to_stop']].copy()
    
    captures_df = (
        captures_df
        .sort_values(by=['stop_time', 'distance_from_ff_to_stop'])
        .drop_duplicates(subset=['stop_id'])
    )

    # 2) Per-stop stats
    # unique_stops_df: Comprehensive table of all stops with temporal statistics (start time, end time, duration)
    # Each row represents one unique stop event from the monkey_information data
    unique_stops_df = extract_unique_stops(monkey_information)

    # 3) No-capture stops: exclude any stop_id present in captures
    no_capture_stops_df = unique_stops_df.loc[
        ~unique_stops_df["stop_id"].isin(captures_df["stop_id"])
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
    
    valid_captures_df = valid_captures_df.reset_index(drop=True)
    valid_captures_df[['stop_id_duration', 'stop_id_start_time', 'stop_id_end_time']] = monkey_information.loc[valid_captures_df['point_index'], ['stop_id_duration', 'stop_id_start_time', 'stop_id_end_time']].values

    # 5) Temporal filtering against capture times (optional)
    filtered_no_capture_stops_df = filter_no_capture_stops_vectorized(
        no_capture_stops_df, ff_caught_T_new, capture_match_window
    )
    
    filtered_no_capture_stops_df['stop_point_index'] = filtered_no_capture_stops_df['point_index']
    filtered_no_capture_stops_df['stop_time'] = filtered_no_capture_stops_df['time']
    filtered_no_capture_stops_df = filter_stops_df_by_debounce(
        filtered_no_capture_stops_df, stop_debounce
    )

    return valid_captures_df, filtered_no_capture_stops_df.reset_index(drop=True), unique_stops_df.reset_index(drop=True)


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


def extract_unique_stops(monkey_information: pd.DataFrame) -> pd.DataFrame:
    # """
    # From per-sample `monkey_information`, compute one row per stop_id with duration and basic fields.

    # Requires columns: ['point_index', 'time', 'stop_id'].
    # Returns a DataFrame with unique stop_ids and columns:
    #   ['stop_id', 'point_index', 'time', 'stop_id_start_time', 'stop_id_end_time', 'stop_id_duration', ...original first-row cols]

    # This function creates unique_stops_df: a comprehensive table of all stops with their temporal statistics.
    # Each row represents one unique stop event with calculated start time, end time, and duration.
    # """
    # required = {"point_index", "time", "stop_id"}
    # missing = required - set(monkey_information.columns)
    # if missing:
    #     raise KeyError(
    #         f"extract_unique_stops: missing columns {sorted(missing)}")

    # # Consider only rows that belong to a stop
    # stops_df = monkey_information.loc[monkey_information["stop_id"].notna()].copy(
    # )

    # # Aggregate per stop_id over time
    # stop_stats = stops_df.groupby("stop_id", as_index=True)["time"].agg(
    #     stop_id_start_time="min",
    #     stop_id_end_time="max"
    # )
    # stop_stats["stop_id_duration"] = (
    #     stop_stats["stop_id_end_time"] - stop_stats["stop_id_start_time"]
    # )

    # # Merge back; keep stable order by point_index (ascending)
    # stops_df = stops_df.merge(stop_stats, on="stop_id", how="left")
    # stops_df = stops_df.sort_values("point_index", kind="stable")

    # # Reduce to one representative row per stop_id (the first encountered in time)
    # unique_stops_df = (
    #     stops_df.groupby("stop_id", as_index=False, sort=False)
    #     .first()
    #     .reset_index(drop=True)
    # )
    
    unique_stops_df = monkey_information[['stop_id', 'point_index', 'time', 'stop_id_start_time', 'stop_id_end_time', 'stop_id_duration']].groupby(('stop_id')).first().reset_index(drop=False)
    

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


from scipy import stats 
from scipy.ndimage import gaussian_filter1d

def plot_inter_stop_intervals(onsets,
                              suggest_window: tuple = (0.1, 0.6),
                              linear_max: float = 2.0,
                              smooth_sigma_bins: int = 2) -> dict:
    """
    Build ISI histograms (linear + log) and auto-suggest a debounce by
    finding the histogram valley within `suggest_window` seconds.

    Returns dict with:
      - 'onsets': array of stop onset times
      - 'isi': array of inter-stop intervals (s)
      - 'suggested_debounce': float or None
      - 'fig': matplotlib Figure
    """

    isi = np.diff(onsets)
    isi = isi[np.isfinite(isi) & (isi > 0)]

    # ---- Linear histogram (focus on short intervals up to linear_max) ----
    lin_bins = np.linspace(0, linear_max, 81)  # 80 bins
    lin_hist, lin_edges = np.histogram(isi, bins=lin_bins)
    lin_centers = (lin_edges[:-1] + lin_edges[1:]) / 2

    # Smooth to find a stable valley
    if smooth_sigma_bins > 0:
        lin_hist_sm = gaussian_filter1d(lin_hist.astype(float), sigma=smooth_sigma_bins, mode="nearest")
    else:
        lin_hist_sm = lin_hist.astype(float)

    # Find valley (minimum) inside suggest_window
    w0, w1 = suggest_window
    mask = (lin_centers >= w0) & (lin_centers <= w1)
    suggested = None
    if np.any(mask):
        # Choose center with minimum smoothed count in the window
        idx = np.argmin(lin_hist_sm[mask])
        suggested = float(lin_centers[mask][idx])

    # ---- Log histogram (broader view) ----
    # Avoid zeros: clamp lower bound
    lo = max(1e-3, np.min(isi[isi > 0]) * 0.8)
    hi = max(0.5, np.max(isi))
    log_bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    # Linear
    ax1.bar(lin_centers, lin_hist, width=np.diff(lin_edges), align="center", alpha=0.6, edgecolor="none")
    ax1.plot(lin_centers, lin_hist_sm, linewidth=2)
    if suggested is not None:
        ax1.axvline(suggested, linestyle="--", color="k", alpha=0.8, label=f"Suggested debounce ≈ {suggested:.3f}s")
        ax1.legend()
    ax1.set_title("Inter-stop intervals (linear scale)")
    ax1.set_xlabel("ISI (s)")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, linear_max)
    ax1.grid(alpha=0.3)

    # Log
    ax2.hist(isi, bins=log_bins, alpha=0.7)
    if suggested is not None:
        ax2.axvline(suggested, linestyle="--", color="k", alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_title("Inter-stop intervals (log scale)")
    ax2.set_xlabel("ISI (s, log)")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()

    return {
        "onsets": onsets,
        "isi": isi,
        "suggested_debounce": suggested,
        "fig": fig,
        "ax1": ax1,
        "ax2": ax2,
    }



import pandas as pd
import numpy as np

def _expand_trials(trials_df: pd.DataFrame,
                   monkey_information: pd.DataFrame,
                   stop_indices_col: str = "stop_indices",
                   out_index_col: str = "stop_point_index") -> pd.DataFrame:
    """
    Explode `trials_df[stop_indices_col]` so each stop index is its own row,
    and add `stop_time` (and `stop_cluster_id` if not already present) by
    mapping from `monkey_information` via positional indexing.

    Assumes `stop_indices` are integer point indices into `monkey_information`.
    """
    df = (
        trials_df
        .explode(stop_indices_col, ignore_index=True)
        .rename(columns={stop_indices_col: out_index_col})
        .copy()
    )

    # Ensure integer positional indices
    idx = df[out_index_col].astype("int64").to_numpy()

    # Map stop_time from monkey_information (positional)
    mi_time = monkey_information["time"].to_numpy()
    df["stop_time"] = mi_time[idx]

    # If stop_cluster_id is not already present, map it too (if exists in MI)
    if "stop_cluster_id" not in df.columns and "stop_cluster_id" in monkey_information.columns:
        mi_cluster = monkey_information["stop_cluster_id"].to_numpy()
        df["stop_cluster_id"] = mi_cluster[idx]

    return df


def _add_cluster_ordering(df: pd.DataFrame,
                          cluster_col: str = "stop_cluster_id",
                          order_col: str = "stop_point_index") -> pd.DataFrame:
    """
    Sort within clusters and add:
      - cluster_size
      - order_in_cluster (0-based)
      - is_first / is_last / is_middle
    """
    out = df.sort_values([cluster_col, order_col], ascending=[True, True]).reset_index(drop=True)

    out["cluster_size"] = out.groupby(cluster_col)[order_col].transform("size")
    out["order_in_cluster"] = out.groupby(cluster_col).cumcount()

    out["is_first"] = out["order_in_cluster"].eq(0)
    out["is_last"]  = out["order_in_cluster"].eq(out["cluster_size"] - 1)
    out["is_middle"] = (~out["is_first"]) & (~out["is_last"])
    return out

