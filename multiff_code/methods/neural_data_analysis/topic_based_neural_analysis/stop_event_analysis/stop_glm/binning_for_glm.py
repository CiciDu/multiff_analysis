import numpy as np
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, get_stops_utils, psth_postprocessing, psth_stats


def bin_timeseries_weighted(values, dt_array, bin_idx_array, how='mean'):
    """
    Sparse time-weighted aggregation into bins.
    Always returns only the bins that actually appear (sorted), plus their IDs.

    Parameters
    ----------
    values : (L,) or (L, K) float
        Value for each overlapped piece.
    dt_array : (L,) float
        Duration (seconds) of each piece.
    bin_idx_array : (L,) int
        Non-negative bin IDs for each piece (may be non-contiguous).
    how : {'mean','sum'}
        'sum'  → ∑(values * dt) per used bin
        'mean' → time-weighted mean = ∑(values * dt) / ∑dt per used bin

    Returns
    -------
    out : (M,) or (M, K) float
        Aggregated values per used bin (M = number of unique bin IDs).
    exposure : (M,) float
        Per-used-bin exposure seconds (∑dt).
    bin_ids : (M,) int
        Sorted unique bin IDs corresponding to rows in `out`/`exposure`.
    """
    V  = np.asarray(values, float)
    dt = np.asarray(dt_array, float)
    bi = np.asarray(bin_idx_array, int)

    if V.ndim == 1:
        V = V[:, None]

    if not (len(V) == len(dt) == len(bi)):
        raise ValueError('values, dt_array, and bin_idx_array must have the same length')
    if np.any(bi < 0):
        raise ValueError('bin_idx_array must be non-negative')

    # Drop invalid rows (NaNs) and clamp negative durations to zero
    valid = np.isfinite(dt) & np.all(np.isfinite(V), axis=1)
    if not np.all(valid):
        V, dt, bi = V[valid], dt[valid], bi[valid]
    if V.size == 0:
        # nothing to aggregate
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,), int)
    dt = np.maximum(dt, 0.0)

    # Map arbitrary bin IDs → compact positions via np.unique (fast, vectorized)
    # used: sorted unique bin IDs; pos: position of each piece in used (0..M-1)
    used_bins, pos = np.unique(bi, return_inverse=True)
    M, K = used_bins.size, V.shape[1]

    # Exposure per used bin: ∑dt
    exposure = np.bincount(pos, weights=dt, minlength=M).astype(float)

    # Weighted sums per used bin: ∑(v * dt) for each feature
    out_sum = np.zeros((M, K), float)
    for k in range(K):
        out_sum[:, k] = np.bincount(pos, weights=V[:, k] * dt, minlength=M)

    # Finalize
    if how == 'sum':
        out = out_sum
    elif how == 'mean':
        with np.errstate(invalid='ignore', divide='ignore'):
            out = out_sum / exposure[:, None]
        out[~np.isfinite(out)] = np.nan
    else:
        raise ValueError("how must be 'mean' or 'sum'")
    
    weighted_values = out.squeeze()

    return weighted_values, exposure, used_bins



def build_bin_assignments(time, bins, assume_sorted=True, check_nonoverlap=False):
    """
    O(n + m) interval–bin overlap for sorted, non-overlapping bins.

    Convention (left-hold):
      value[i] applies on [t[i], t[i+1]) for i = 0..n-2.

    Parameters
    ----------
    time : (n,) strictly increasing float array
    bins : (m, 2) float array of [left, right] per bin
           Must be sorted by left edge ascending. Non-overlapping recommended.
    assume_sorted : bool
        If False, we sort bins by left edge first.
    check_nonoverlap : bool
        If True, assert that bins do not overlap.

    Returns
    -------
    sample_idx : (L,) int    # interval index i contributing
    bin_idx_array    : (L,) int    # bin index j receiving contribution
    dt_array   : (L,) float  # overlap duration for that (i, j)
    m          : int         # number of bins
    """
    t = np.asarray(time, float)
    n = t.size
    assert n >= 2, 'need ≥2 time points'
    bins = np.asarray(bins, float)
    assert bins.ndim == 2 and bins.shape[1] == 2, 'bins must be (m,2)'

    # intervals (left-hold): [t[i], t[i+1]) for i=0..n-2
    seg_lo = t[:-1]
    seg_hi = t[1:]

    # sort bins if needed
    if not assume_sorted:
        order = np.argsort(bins[:, 0], kind='mergesort')
        bins = bins[order]

    if check_nonoverlap:
        if np.any(bins[1:, 0] < bins[:-1, 1]):
            raise ValueError('bins overlap; two-pointer assumes non-overlap')

    bin_lo = bins[:, 0]
    bin_hi = bins[:, 1]
    m = bins.shape[0]

    sample_idx = []
    bin_idx_array = []
    dt_array = []

    i = 0  # interval pointer (0..n-2)
    j = 0  # bin pointer (0..m-1)

    while i < n - 1 and j < m:
        lo = max(seg_lo[i], bin_lo[j])
        hi = min(seg_hi[i], bin_hi[j])

        if hi > lo:
            sample_idx.append(i)
            bin_idx_array.append(j)
            dt_array.append(hi - lo)

        # advance whichever segment ends first
        if seg_hi[i] <= bin_hi[j]:
            i += 1
        else:
            j += 1

        # skip bins entirely before next interval
        while j < m and bin_hi[j] <= seg_lo[i] if i < n - 1 else False:
            j += 1

        # skip intervals entirely before next bin
        while i < n - 1 and seg_hi[i] <= bin_lo[j] if j < m else False:
            i += 1

    if sample_idx:
        sample_idx = np.asarray(sample_idx, dtype=int)
        bin_idx_array    = np.asarray(bin_idx_array,    dtype=int)
        dt_array   = np.asarray(dt_array,   dtype=float)
    else:
        sample_idx = np.zeros(0, dtype=int)
        bin_idx_array    = np.zeros(0, dtype=int)
        dt_array   = np.zeros(0, dtype=float)

    return sample_idx, bin_idx_array, dt_array, m



def pick_stop_window(df, stop_time_col='stop_time',
                     next_event_col='next_time',
                     prev_event_col='prev_time',
                     pre_s=0.6, post_s=1.0, min_pre_bins=10, min_post_bins=20, bin_dt=0.04):
    out = df.copy()
    stop_t = out[stop_time_col].astype(float)

    # nominal window
    t0_nom = stop_t - float(pre_s)
    t1_nom = stop_t + float(post_s)
    t0 = t0_nom.copy()
    t1 = t1_nom.copy()

    # clip to midpoints with neighbors (only where defined)
    if prev_event_col in out.columns:
        prev_t = out[prev_event_col].astype(float)
        mask = prev_t.notna()
        t0[mask] = np.maximum(t0[mask], 0.5 * (prev_t[mask] + stop_t[mask]))
    if next_event_col in out.columns:
        next_t = out[next_event_col].astype(float)
        mask = next_t.notna()
        t1[mask] = np.minimum(t1[mask], 0.5 * (next_t[mask] + stop_t[mask]))

    out['new_seg_start_time'] = t0
    out['new_seg_end_time'] = t1

    # truncation flags
    eps = 1e-9
    out['is_truncated_pre'] = (out['new_seg_start_time'] > (t0_nom + eps))
    out['is_truncated_post'] = (out['new_seg_end_time'] < (t1_nom - eps))

    # bin counts
    dt = float(bin_dt)
    out['n_pre_bins'] = np.floor(
        (stop_t - out['new_seg_start_time']) / dt).astype(int)
    out['n_post_bins'] = np.floor(
        (out['new_seg_end_time'] - stop_t) / dt).astype(int)

    out['new_seg_start_time'] = stop_t - out['n_pre_bins'] * dt
    out['new_seg_end_time'] = stop_t + out['n_post_bins'] * dt
    out['new_seg_duration'] = out['new_seg_end_time'] - out['new_seg_start_time']

    # quality flag
    out['ok_window'] = (out['n_pre_bins'] >= int(min_pre_bins)) & (
        out['n_post_bins'] >= int(min_post_bins))
    
    new_seg_info = out
    return new_seg_info


def stops_windows_to_bins2d(picked_windows,
                            stop_id_col='stop_id',
                            stop_time_col='stop_time',
                            win_t0_col='new_seg_start_time',
                            win_t1_col='new_seg_end_time',
                            n_pre_col='n_pre_bins',
                            n_post_col='n_post_bins',
                            ok_col='ok_window',
                            only_ok=True,
                            bin_dt=None,
                            tol=1e-9):
    """
    Turn stop-centered windows into per-stop fixed-width bins.

    Produces:
      - bins_2d: (N_bins, 2) array of [left, right] for each bin across all stops
      - meta: tidy DataFrame with per-bin metadata (stop_id, indices, centers, etc.)

    Assumptions
    -----------
    - Each row in `stops_df` describes one stop with snapped edges (new_seg_start_time/new_seg_end_time)
      consistent with integer bin counts n_pre_bins/n_post_bins.
    - If `bin_dt` is None, we infer it from the first row with positive pre or post bins.

    Parameters
    ----------
    only_ok : bool
        If True and `ok_col` exists, keep only rows where ok_window is True.
    bin_dt : float or None
        Bin width. If None, infer from (stop - new_seg_start_time)/n_pre or (new_seg_end_time - stop)/n_post.

    Returns
    -------
    bins_2d : ndarray, shape (N, 2)
        All bins concatenated: [t_left, t_right].
    meta : DataFrame, shape (N, ?)
        Per-bin info: stop_id, k_within_stop, is_pre, t_left, t_right, t_center,
        rel_left, rel_right, rel_center, exposure_s (=bin_dt), stop_time.
    """
    df = picked_windows.copy()

    # Optional filter
    if only_ok and ok_col in df.columns:
        df = df[df[ok_col].astype(bool)].copy()

    required = [stop_id_col, stop_time_col,
                win_t0_col, win_t1_col, n_pre_col, n_post_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'missing required columns: {missing}')

    # Infer bin_dt if needed
    if bin_dt is None:
        # Gather candidates from any row with positive counts
        dts = []
        for _, r in df.iterrows():
            npre = int(r[n_pre_col])
            npost = int(r[n_post_col])
            if npre > 0:
                dts.append((float(r[stop_time_col]) -
                           float(r[win_t0_col])) / npre)
            if npost > 0:
                dts.append((float(r[win_t1_col]) -
                           float(r[stop_time_col])) / npost)
        if not dts:
            raise ValueError(
                'cannot infer bin_dt: no rows with positive pre/post bin counts')
        # Use median for robustness
        bin_dt = float(np.median(dts))

    bins_list = []
    meta_rows = []

    for _, r in df.iterrows():
        stop_id = r[stop_id_col]
        s = float(r[stop_time_col])
        npre = int(r[n_pre_col])
        npost = int(r[n_post_col])
        n_bins = npre + npost
        if n_bins <= 0:
            continue

        # Build bin edges centered around stop (pre first, then post)
        # Left edge of the first bin is s - npre*dt
        left0 = s - npre * bin_dt
        lefts = left0 + bin_dt * np.arange(n_bins)
        rights = lefts + bin_dt
        centers = 0.5 * (lefts + rights)

        # Sanity: edges should lie within [new_seg_start_time, new_seg_end_time] up to tol
        if win_t0_col in r and win_t1_col in r:
            t0 = float(r[win_t0_col])
            t1 = float(r[win_t1_col])
            if (lefts[0] < t0 - tol) or (rights[-1] > t1 + tol):
                # If upstream rounding created tiny drift, gently clip
                lefts[0] = max(lefts[0], t0)
                rights[-1] = min(rights[-1], t1)

        # Append bins and metadata
        bins_list.append(np.column_stack([lefts, rights]))

        # Per-bin flags: first npre bins are 'pre'
        is_pre = np.zeros(n_bins, dtype=bool)
        if npre > 0:
            is_pre[:npre] = True

        meta_rows.append(pd.DataFrame({
            'stop_id': stop_id,
            'k_within_stop': np.arange(n_bins, dtype=int),
            'is_pre': is_pre,
            't_left': lefts,
            't_right': rights,
            't_center': centers,
            'rel_left': lefts - s,
            'rel_right': rights - s,
            'rel_center': centers - s,
            'exposure_s': np.full(n_bins, bin_dt),
            'stop_time': np.full(n_bins, s),
        }))

    if not bins_list:
        # No bins created
        return np.zeros((0, 2), float), pd.DataFrame(columns=[
            'stop_id', 'k_within_stop', 'is_pre', 't_left', 't_right', 't_center',
            'rel_left', 'rel_right', 'rel_center', 'exposure_s', 'stop_time'
        ])

    bins_2d = np.vstack(bins_list)
    meta = pd.concat(meta_rows, ignore_index=True)

    # Global bin index in time order (stable sort)
    order = np.argsort(np.asarray(meta['t_left'], dtype=float))
    bins_2d = bins_2d[order]
    meta = meta.iloc[order].reset_index(drop=True)
    meta['bin'] = np.arange(len(meta), dtype=int)

    return bins_2d, meta



def bin_spikes_by_cluster(spikes_df,
                          bins_2d,
                          time_col='time',
                          cluster_col='cluster',
                          clusters=None,
                          assume_sorted_bins=True,
                          check_nonoverlap=False):
    """
    Bin point spikes into possibly disjoint, sorted bins.

    Bins use the half-open convention [left, right): left-inclusive, right-exclusive.
    Each spike increments exactly one bin if it falls inside; spikes in gaps are ignored.

    Parameters
    ----------
    spikes_df : DataFrame with columns [time_col, cluster_col]
    bins_2d   : (M, 2) ndarray of [left, right] per bin, sorted by left edge
    time_col  : str, spike time column name
    cluster_col : str, cluster/unit id column name (int or str ok)
    clusters  : optional sequence of cluster IDs to include (and order to use).
                If None, uses sorted unique IDs found in spikes_df after masking to bins.
    assume_sorted_bins : bool, if False, will sort bins by left edge
    check_nonoverlap   : bool, if True, raise if bins overlap

    Returns
    -------
    counts : (M, C) int ndarray
        Spike counts per bin (rows) and per cluster (columns).
    cluster_ids : (C,) ndarray
        Cluster IDs corresponding to columns of `counts`.
    """
    # Extract arrays
    t = np.asarray(spikes_df[time_col], float)
    cl = np.asarray(spikes_df[cluster_col])

    bins = np.asarray(bins_2d, float)
    assert bins.ndim == 2 and bins.shape[1] == 2, 'bins_2d must be shape (M, 2)'
    if not assume_sorted_bins:
        order = np.argsort(bins[:, 0], kind='mergesort')
        bins = bins[order]
    if check_nonoverlap and np.any(bins[1:, 0] < bins[:-1, 1]):
        raise ValueError('bins overlap; expected non-overlapping bins for single assignment')

    lefts  = bins[:, 0]
    rights = bins[:, 1]
    M = bins.shape[0]

    # Map each spike time -> candidate bin via left edges,
    # then keep only spikes that also satisfy t < rights[idx]
    idx = np.searchsorted(lefts, t, side='right') - 1
    valid = (idx >= 0) & (idx < M)
    valid &= t < rights[np.clip(idx, 0, M-1)]
    if not np.any(valid):
        # No spikes fall in any bin
        if clusters is None:
            return np.zeros((M, 0), dtype=int), np.array([], dtype=cl.dtype)
        else:
            return np.zeros((M, len(clusters)), dtype=int), np.asarray(clusters)

    idx = idx[valid]
    cl  = cl[valid]

    # Choose cluster columns
    if clusters is None:
        # use sorted unique cluster IDs present in the filtered spikes
        # (preserves numeric order; if you prefer first-seen order, use pd.unique)
        cluster_ids = np.unique(cl)
    else:
        cluster_ids = np.asarray(clusters)
    C = cluster_ids.size

    # Build mapping cluster_id -> column index
    # For speed, if cluster_ids are numeric and sorted, use searchsorted
    try:
        col = np.searchsorted(cluster_ids, cl)
        # Ensure cl values are indeed in cluster_ids; otherwise filter them out
        in_range = (col >= 0) & (col < C) & (cluster_ids[col] == cl)
        idx = idx[in_range]
        col = col[in_range]
    except Exception:
        # Fallback: general mapping (works for strings too)
        id2col = {cid: k for k, cid in enumerate(cluster_ids)}
        col = np.fromiter((id2col.get(x, -1) for x in cl), count=cl.size, dtype=int)
        keep = col >= 0
        idx = idx[keep]
        col = col[keep]

    # Accumulate counts
    counts = np.zeros((M, C), dtype=int)
    np.add.at(counts, (idx, col), 1)
    return counts, cluster_ids
