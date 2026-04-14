"""
Trajectory clustering by stable-phase commitment.

Pipeline:
- Trajectories are taken from behavior data (columns ``cur_ff_rel_x``, ``cur_ff_rel_y``,
  ``curv_of_traj``, ``bin_in_new_seg``, ``new_segment``) in target-centered coordinates.
- The *stable phase* is the earliest segment of time along a trajectory where local
  curvature is consistently low (rolling standard deviation below a threshold), interpreted
  as commitment to a movement policy.
- The *entry point* is the 2D position at the first index of that stable phase.
- Entry points are discretized into spatial bins; trajectories are grouped by bin and
  visualized (optionally K-means on entry XY is available as an alternative).
- Optionally, trajectories can be restricted to those whose stable-phase entry falls in
  *central* bins (median bin ± radius), dropping peripheral entries; an optional
  minimum count per bin removes sparse bins among the retained points.
- Use ``filter_stable_entries_inside_bounds`` (or ``require_inside_bounds`` on the
  central-bin filter) so entries outside the percentile binning rectangle are not
  labeled via clipping while still plotting outside the grid.
"""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans


# =========================
# 1. SEGMENT START STATES (optional helper)
# =========================
def get_segment_start_states(df):
    """First row per ``new_segment`` after sorting by ``bin_in_new_seg``."""
    return (
        df.sort_values("bin_in_new_seg")
        .groupby("new_segment")
        .first()
        .reset_index()
    )


# =========================
# 2. STATE BINNING BY SEGMENT START (optional; not stable-phase entry)
# =========================
def assign_state_bins(seg_df, bin_size=50, percentile=95):
    """
    Bin *segment start* positions in target-centered coordinates.

    For clustering by stable-phase *entry*, use ``assign_stable_bins`` with
    ``compute_bin_bounds_from_entries`` instead.
    """
    seg_df = seg_df.copy()

    p_low = (100 - percentile) / 2
    p_high = 100 - p_low

    x_min, x_max = np.percentile(seg_df["cur_ff_rel_x"], [p_low, p_high])
    y_min, y_max = np.percentile(seg_df["cur_ff_rel_y"], [p_low, p_high])

    seg_df["rel_x_clipped"] = seg_df["cur_ff_rel_x"].clip(x_min, x_max)
    seg_df["rel_y_clipped"] = seg_df["cur_ff_rel_y"].clip(y_min, y_max)

    seg_df["state_bin_x"] = np.floor(
        (seg_df["rel_x_clipped"] - x_min) / bin_size
    ).astype(int)
    seg_df["state_bin_y"] = np.floor(
        (seg_df["rel_y_clipped"] - y_min) / bin_size
    ).astype(int)

    seg_df["state_cluster"] = seg_df.groupby(["state_bin_x", "state_bin_y"]).ngroup()

    return seg_df, (x_min, x_max, y_min, y_max)


# =========================
# 3. MAP BACK
# =========================
def map_clusters_to_df(df, seg_df):
    return df.merge(
        seg_df[["new_segment", "state_cluster"]], on="new_segment", how="left"
    )


# =========================
# 4. CENTRAL FILTER
# =========================
def filter_central_clusters(df, seg_df, x_radius=1, y_radius=1):
    center_x = seg_df["state_bin_x"].median()
    center_y = seg_df["state_bin_y"].median()

    mask = (
        (seg_df["state_bin_x"] >= center_x - x_radius)
        & (seg_df["state_bin_x"] <= center_x + x_radius)
        & (seg_df["state_bin_y"] >= center_y - y_radius)
        & (seg_df["state_bin_y"] <= center_y + y_radius)
    )

    seg_df = seg_df[mask]
    df = df[df["new_segment"].isin(seg_df["new_segment"])]

    return df, seg_df


# =========================
# 5. STABLE PHASE: LOCAL CURVATURE CONSISTENCY
# =========================
def find_stable_phase_entry_index(
    curv,
    std_thresh=0.002,
    range_thresh=0.005,      # e.g. 0.01; None = disabled
    min_points_after_entry=5,
    n_persist=1,
):
    curv = np.asarray(curv, dtype=float)
    n = len(curv)
    if n < min_points_after_entry:
        return None

    # Evaluate each candidate start i using the full truncated suffix curv[i:].
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        w = curv[i:]
        if len(w) < min_points_after_entry or np.all(np.isnan(w)):
            continue
        # Require curvature sign consistency: nonzero finite values must be all
        # nonnegative or all nonpositive. Zeros are allowed with either sign.
        w_finite = w[np.isfinite(w)]
        w_nonzero = w_finite[w_finite != 0]
        sign_ok = (
            True
            if len(w_nonzero) == 0
            else (np.all(w_nonzero > 0) or np.all(w_nonzero < 0))
        )
        std_ok = np.nanstd(w) < std_thresh
        range_ok = (np.nanmax(w) - np.nanmin(w)) < range_thresh if range_thresh is not None else True
        if std_ok and range_ok and sign_ok:
            ok[i] = True

    last_start = n - min_points_after_entry
    for i in range(0, last_start + 1):
        if i + n_persist > len(ok):
            break
        if np.all(ok[i : i + n_persist]) and (n - i) >= min_points_after_entry:
            return i

    return None

# =========================
# 6. RESAMPLE
# =========================
def resample_traj(traj, n_points=20):
    if traj is None or len(traj) < 5:
        return None

    t_old = np.linspace(0, 1, len(traj))
    t_new = np.linspace(0, 1, n_points)

    x = np.interp(t_new, t_old, traj[:, 0])
    y = np.interp(t_new, t_old, traj[:, 1])

    return np.stack([x, y], axis=1)




def compute_bin_bounds_from_entries(X, percentile=95):
    """
    Robust outer bounds for binning entry points (same idea as ``assign_state_bins``).

    Returns (x_min, x_max, y_min, y_max) from symmetric percentiles of ``X``.
    """
    if X is None or len(X) == 0:
        raise ValueError("No entry points to define bounds.")
    p_low = (100 - percentile) / 2
    p_high = 100 - p_low
    x_min, x_max = np.percentile(X[:, 0], [p_low, p_high])
    y_min, y_max = np.percentile(X[:, 1], [p_low, p_high])
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    return (x_min, x_max, y_min, y_max)


# =========================
# 8. CLUSTER / BIN
# =========================
def cluster_trajectories(X, n_clusters=3):
    """Optional: group trajectories by K-means on entry XY (not the default grid)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    return labels, kmeans


def assign_stable_bins(X, bounds, bin_size=50):
    """
    Discretize entry positions into spatial bins; labels are deterministic bin ids.

    ``bounds`` is (x_min, x_max, y_min, y_max) in target-centered coordinates.
    Points outside are clipped to the bin at the edge (same as ``assign_state_bins``).
    """
    x_min, x_max, y_min, y_max = bounds

    x = np.clip(X[:, 0], x_min, x_max)
    y = np.clip(X[:, 1], y_min, y_max)

    bin_x = np.floor((x - x_min) / bin_size).astype(int)
    bin_y = np.floor((y - y_min) / bin_size).astype(int)

    bin_pairs = np.stack([bin_x, bin_y], axis=1)

    unique_bins = np.unique(bin_pairs, axis=0)
    bin_to_label = {tuple(b): i for i, b in enumerate(unique_bins)}

    labels = np.array([bin_to_label[tuple(b)] for b in bin_pairs])

    return labels, bin_pairs


def stable_entry_bin_indices(X, bounds, bin_size):
    """
    Spatial bin indices ``(bin_x, bin_y)`` for each stable entry, matching
    ``assign_stable_bins`` (coordinates are clipped to ``bounds`` before binning).
    """
    x_min, x_max, y_min, y_max = bounds
    x = np.clip(X[:, 0], x_min, x_max)
    y = np.clip(X[:, 1], y_min, y_max)
    bin_x = np.floor((x - x_min) / bin_size).astype(int)
    bin_y = np.floor((y - y_min) / bin_size).astype(int)
    return bin_x, bin_y


def filter_stable_entries_inside_bounds(X, seg_ids, entry_by_seg, bounds):
    """
    Keep only trajectories whose stable-phase entry lies inside the binning rectangle.

    ``assign_stable_bins`` / ``stable_entry_bin_indices`` *clip* out-of-range
    coordinates to the edges, so entries outside ``bounds`` can still receive a
    central bin label while their true (raw) position plots outside the grid.
    Apply this filter *before* central-bin logic and again after recomputing
    percentile ``bounds`` if you want markers and labels aligned with the grid.
    """
    if X is None or len(X) == 0:
        return np.empty((0, 2)), np.array([]), {}

    x_min, x_max, y_min, y_max = bounds
    inside = (
        (X[:, 0] >= x_min)
        & (X[:, 0] <= x_max)
        & (X[:, 1] >= y_min)
        & (X[:, 1] <= y_max)
    )

    if not np.any(inside):
        return np.empty((0, 2)), np.array([]), {}

    X_out = X[inside]
    seg_out = seg_ids[inside]
    kept = set(seg_out.tolist())
    entry_out = {k: entry_by_seg[k] for k in kept if k in entry_by_seg}
    return X_out, seg_out, entry_out


def filter_stable_entries_by_central_bins(
    X,
    seg_ids,
    entry_by_seg,
    bounds,
    bin_size=50,
    x_radius=1,
    y_radius=1,
    min_bin_count=None,
    require_inside_bounds=True,
):
    """
    Keep only trajectories whose stable-phase entry lies in *central* spatial bins.

    Central bins are defined relative to the median bin index along x and y (same
    construction as ``filter_central_clusters`` for segment-start bins): entries with
    bin indices within ``x_radius`` / ``y_radius`` of that median are kept; others
    are discarded as peripheral.

    If ``min_bin_count`` is set (> 0), among points that pass the central rule, drop
    entries whose ``(bin_x, bin_y)`` occurs fewer than ``min_bin_count`` times
    (sparse pockets), counting only over the centrally retained set.

    When ``require_inside_bounds`` is True (default), entries whose *raw* stable
    position lies outside ``bounds`` are dropped first so clipping cannot fake a
    central bin assignment.

    Parameters
    ----------
    bounds : tuple
        ``(x_min, x_max, y_min, y_max)`` used to define bin indices (typically from
        ``compute_bin_bounds_from_entries`` on the same ``X`` before filtering).
    require_inside_bounds : bool
        If True, drop raw entries outside ``bounds`` before applying the central-bin
        rule (recommended).

    Returns
    -------
    X_out, seg_ids_out, entry_by_seg_out
        Filtered arrays and dict; empty if nothing passes.
    """
    if X is None or len(X) == 0:
        return np.empty((0, 2)), np.array([]), {}

    if require_inside_bounds:
        X, seg_ids, entry_by_seg = filter_stable_entries_inside_bounds(
            X, seg_ids, entry_by_seg, bounds
        )
        if len(X) == 0:
            return np.empty((0, 2)), np.array([]), {}

    bin_x, bin_y = stable_entry_bin_indices(X, bounds, bin_size)
    cx = float(np.median(bin_x))
    cy = float(np.median(bin_y))

    central = (
        (bin_x >= cx - x_radius)
        & (bin_x <= cx + x_radius)
        & (bin_y >= cy - y_radius)
        & (bin_y <= cy + y_radius)
    )

    mask = central
    if min_bin_count is not None and int(min_bin_count) > 1 and np.any(mask):
        bx_m = bin_x[mask]
        by_m = bin_y[mask]
        c = Counter(zip(bx_m.tolist(), by_m.tolist()))
        heavy = np.zeros(len(X), dtype=bool)
        for i in np.where(mask)[0]:
            if c[(int(bin_x[i]), int(bin_y[i]))] >= int(min_bin_count):
                heavy[i] = True
        mask = heavy

    if not np.any(mask):
        return np.empty((0, 2)), np.array([]), {}

    X_out = X[mask]
    seg_out = seg_ids[mask]
    kept = set(seg_out.tolist())
    entry_out = {k: entry_by_seg[k] for k in kept if k in entry_by_seg}
    return X_out, seg_out, entry_out


def extract_stable_traj(seg_df, min_len=5, std_thresh=0.002, range_thresh=None):
    seg_df = seg_df.sort_values("bin_in_new_seg")
    curv = seg_df["curv_of_traj"].values
    xy = seg_df[["cur_ff_rel_x", "cur_ff_rel_y"]].values
    start = find_stable_phase_entry_index(
        curv, std_thresh=std_thresh,
        range_thresh=range_thresh, min_points_after_entry=min_len,
    )
    if start is None:
        return None
    return xy[start:]


def build_traj_dataset(df, min_len=5, std_thresh=0.002, range_thresh=None, n_persist=1):
    X, seg_ids, entry_by_seg = [], [], {}
    for seg_id, seg_df in df.groupby("new_segment"):
        seg_df = seg_df.sort_values("bin_in_new_seg")
        curv = seg_df["curv_of_traj"].values
        xy = seg_df[["cur_ff_rel_x", "cur_ff_rel_y"]].values
        start = find_stable_phase_entry_index(
            curv, std_thresh=std_thresh,
            range_thresh=range_thresh, min_points_after_entry=min_len,
            n_persist=n_persist,
        )
        if start is None:
            continue
        stable_entry = xy[start]
        X.append(stable_entry)
        seg_ids.append(seg_id)
        entry_by_seg[seg_id] = (float(stable_entry[0]), float(stable_entry[1]))
    if len(X) == 0:
        return np.empty((0, 2)), np.array([]), {}
    return np.array(X), np.array(seg_ids), entry_by_seg


def run_trajectory_binning_pipeline(
    df, bin_size=50, percentile=95, min_len=5,
    std_thresh=0.002, range_thresh=None, n_persist=1,
):
    X, seg_ids, entry_by_seg = build_traj_dataset(
        df, min_len=min_len, std_thresh=std_thresh,
        range_thresh=range_thresh, n_persist=n_persist,
    )
    if len(X) == 0:
        return seg_ids, np.array([]), None, np.empty((0, 2)), entry_by_seg
    bounds = compute_bin_bounds_from_entries(X, percentile=percentile)
    labels, bin_pairs = assign_stable_bins(X, bounds, bin_size=bin_size)
    return seg_ids, labels, bounds, bin_pairs, entry_by_seg


def preprocess_df(df, bin_size=50, percentile=95, x_radius=1, y_radius=1):
    """Steps 1-4: segment states, binning, map back, central filter."""
    seg_df = get_segment_start_states(df)
    seg_df, bounds = assign_state_bins(seg_df, bin_size=bin_size, percentile=percentile)
    df = map_clusters_to_df(df, seg_df)
    df, seg_df = filter_central_clusters(df, seg_df, x_radius=x_radius, y_radius=y_radius)
    return df, seg_df, bounds


def build_and_filter_stable_entries(
    df,
    std_thresh=0.003,
    range_thresh=0.008,
    min_len=5,
    n_persist=1,
    bin_size=50,
    percentile=95,
    x_radius=1,
    y_radius=1,
    min_bin_count=2,
):
    """Steps 5-6: build stable entries, central filter, inside-bounds filter, assign bins."""
    print(f"  Num trajectories before filtering: {len(df)}")
    
    X, seg_ids, entry_by_seg = build_traj_dataset(
        df, std_thresh=std_thresh, range_thresh=range_thresh,
        min_len=min_len, n_persist=n_persist,
    )
    print(f"  Num trajectories with stable phase: {len(X)}")

    if len(X) == 0:
        return X, seg_ids, entry_by_seg, df, None, None, np.empty((0, 2))

    bounds_for_entry = compute_bin_bounds_from_entries(X, percentile=percentile)
    X, seg_ids, entry_by_seg = filter_stable_entries_by_central_bins(
        X, seg_ids, entry_by_seg, bounds_for_entry,
        bin_size=bin_size, x_radius=x_radius, y_radius=y_radius,
        min_bin_count=min_bin_count,
    )
    df = df[df["new_segment"].isin(seg_ids)]
    print(f"  After central (+ sparse) filter: {len(X)}")
    if len(X) == 0:
        return X, seg_ids, entry_by_seg, df, None, None, np.empty((0, 2))

    stable_bounds = compute_bin_bounds_from_entries(X, percentile=percentile)
    X, seg_ids, entry_by_seg = filter_stable_entries_inside_bounds(
        X, seg_ids, entry_by_seg, stable_bounds
    )
    df = df[df["new_segment"].isin(seg_ids)]
    if len(X) == 0:
        return X, seg_ids, entry_by_seg, df, stable_bounds, None, np.empty((0, 2))

    labels, bin_pairs = assign_stable_bins(X, stable_bounds, bin_size=bin_size)

    return X, seg_ids, entry_by_seg, df, stable_bounds, labels, bin_pairs