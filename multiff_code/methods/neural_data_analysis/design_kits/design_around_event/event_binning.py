import numpy as np
import pandas as pd
from pandas.api import types as pdt
import statsmodels.api as sm


from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, psth_postprocessing, psth_stats
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import get_stops_utils
import neural_data_analysis.design_kits.design_around_event.event_binning as event_binning

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
    V = np.asarray(values, float)
    dt = np.asarray(dt_array, float)
    bi = np.asarray(bin_idx_array, int)

    if V.ndim == 1:
        V = V[:, None]

    if not (len(V) == len(dt) == len(bi)):
        raise ValueError(
            'values, dt_array, and bin_idx_array must have the same length')
    if np.any(bi < 0):
        raise ValueError('bin_idx_array must be non-negative')

    # # # Drop invalid rows (NaNs) and clamp negative durations to zero
    # # valid = np.isfinite(dt) & np.all(np.isfinite(V), axis=1)
    # # if not np.all(valid):
    # #     V, dt, bi = V[valid], dt[valid], bi[valid]
    
    # # Drop rows only if dt is invalid
    # valid_dt = np.isfinite(dt)
    # if not np.all(valid_dt):
    #     V, dt, bi = V[valid_dt], dt[valid_dt], bi[valid_dt]

    # 1) Row is valid if dt is finite AND at least one value is finite
    valid = np.isfinite(dt) & np.any(np.isfinite(V), axis=1)

    if not np.all(valid):
        V, dt, bi = V[valid], dt[valid], bi[valid]

    # 2) Clamp negative durations
    dt = np.maximum(dt, 0.0)


    # Clamp negative durations
    dt = np.maximum(dt, 0.0)

    # Mask invalid values per feature (do NOT drop rows)
    V = np.where(np.isfinite(V), V, 0.0)


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
    t = np.asarray(time, float)
    n = t.size
    assert n >= 2, 'need ≥2 time points'

    bins = np.asarray(bins, float)
    assert bins.ndim == 2 and bins.shape[1] == 2

    m = bins.shape[0]
    orig_bin_idx = np.arange(m)

    if not assume_sorted:
        order = np.argsort(bins[:, 0], kind='mergesort')
        bins = bins[order]
        orig_bin_idx = orig_bin_idx[order]

    if check_nonoverlap:
        if np.any(bins[1:, 0] < bins[:-1, 1]):
            raise ValueError('bins overlap')

    seg_lo = t[:-1]
    seg_hi = t[1:]
    bin_lo = bins[:, 0]
    bin_hi = bins[:, 1]

    sample_idx = []
    bin_idx_array = []
    dt_array = []

    i = 0
    j = 0

    while i < n - 1 and j < m:
        lo = max(seg_lo[i], bin_lo[j])
        hi = min(seg_hi[i], bin_hi[j])

        if hi > lo:
            sample_idx.append(i)
            bin_idx_array.append(orig_bin_idx[j])  # 👈 ORIGINAL bin index
            dt_array.append(hi - lo)

        if seg_hi[i] <= bin_hi[j]:
            i += 1
        else:
            j += 1

        while j < m and i < n - 1 and bin_hi[j] <= seg_lo[i]:
            j += 1
        while i < n - 1 and j < m and seg_hi[i] <= bin_lo[j]:
            i += 1

    return (
        np.asarray(sample_idx, int),
        np.asarray(bin_idx_array, int),
        np.asarray(dt_array, float),
        m
    )


def pick_event_window(df, event_time_col='stop_time',
                      prev_event_col='prev_time',
                      next_event_col='next_time',
                      pre_s=0.6, post_s=1.0, min_pre_bins=10, min_post_bins=20, bin_dt=0.04):
    out = df.copy()
    event_t = out[event_time_col].astype(float)

    # nominal window
    t0_nom = event_t - float(pre_s)
    t1_nom = event_t + float(post_s)
    t0 = t0_nom.copy()
    t1 = t1_nom.copy()

    # clip to midpoints with neighbors (only where defined)
    if prev_event_col in out.columns:
        prev_t = out[prev_event_col].astype(float)
        mask = prev_t.notna()
        t0[mask] = np.maximum(t0[mask], 0.5 * (prev_t[mask] + event_t[mask]))
    if next_event_col in out.columns:
        next_t = out[next_event_col].astype(float)
        mask = next_t.notna()
        t1[mask] = np.minimum(t1[mask], 0.5 * (next_t[mask] + event_t[mask]))

    out['new_seg_start_time'] = t0
    out['new_seg_end_time'] = t1

    # truncation flags
    eps = 1e-9
    out['is_truncated_pre'] = (out['new_seg_start_time'] > (t0_nom + eps))
    out['is_truncated_post'] = (out['new_seg_end_time'] < (t1_nom - eps))

    # bin counts
    dt = float(bin_dt)
    out['n_pre_bins'] = np.floor(
        (event_t - out['new_seg_start_time']) / dt).astype(int)
    out['n_post_bins'] = np.floor(
        (out['new_seg_end_time'] - event_t) / dt).astype(int)

    out['new_seg_start_time'] = event_t - out['n_pre_bins'] * dt
    out['new_seg_end_time'] = event_t + out['n_post_bins'] * dt
    out['new_seg_duration'] = out['new_seg_end_time'] - out['new_seg_start_time']

    # quality flag
    out['ok_window'] = (out['n_pre_bins'] >= int(min_pre_bins)) & (
        out['n_post_bins'] >= int(min_post_bins))

    new_seg_info = out
    return new_seg_info

def bin_spikes_by_cluster(spikes_df,
                          bins_2d,
                          time_col='time',
                          cluster_col='cluster',
                          clusters=None,
                          assume_sorted_bins=True,
                          check_nonoverlap=False):
    """
    Bin point spikes into possibly disjoint bins.

    Bins use the half-open convention [left, right): left-inclusive, right-exclusive.
    Each spike increments exactly one bin if it falls inside; spikes in gaps are ignored.

    Returns counts in the SAME ORDER as the input bins_2d.
    """
    import numpy as np

    # Extract arrays
    t = np.asarray(spikes_df[time_col], float)
    cl = np.asarray(spikes_df[cluster_col])

    bins = np.asarray(bins_2d, float)
    assert bins.ndim == 2 and bins.shape[1] == 2, 'bins_2d must be shape (M, 2)'

    M = bins.shape[0]

    # Track original bin order
    orig_order = np.arange(M)

    # Optionally sort bins internally
    if not assume_sorted_bins:
        order = np.argsort(bins[:, 0], kind='mergesort')
        bins = bins[order]
        orig_order = orig_order[order]

    if check_nonoverlap and np.any(bins[1:, 0] < bins[:-1, 1]):
        raise ValueError(
            'bins overlap; expected non-overlapping bins for single assignment')

    lefts = bins[:, 0]
    rights = bins[:, 1]

    # Assign spikes to bins (in sorted-bin space)
    idx = np.searchsorted(lefts, t, side='right') - 1
    valid = (idx >= 0) & (idx < M)
    valid &= t < rights[np.clip(idx, 0, M - 1)]

    if not np.any(valid):
        if clusters is None:
            counts = np.zeros((M, 0), dtype=int)
            cluster_ids = np.array([], dtype=cl.dtype)
        else:
            counts = np.zeros((M, len(clusters)), dtype=int)
            cluster_ids = np.asarray(clusters)

        # Restore original bin order
        if not assume_sorted_bins:
            inv_order = np.argsort(orig_order)
            counts = counts[inv_order]

        return counts, cluster_ids

    idx = idx[valid]
    cl = cl[valid]

    # Choose cluster columns
    if clusters is None:
        cluster_ids = np.unique(cl)
    else:
        cluster_ids = np.asarray(clusters)
    C = cluster_ids.size

    # Map cluster IDs -> column indices
    try:
        col = np.searchsorted(cluster_ids, cl)
        in_range = (col >= 0) & (col < C) & (cluster_ids[col] == cl)
        idx = idx[in_range]
        col = col[in_range]
    except Exception:
        id2col = {cid: k for k, cid in enumerate(cluster_ids)}
        col = np.fromiter((id2col.get(x, -1) for x in cl),
                          count=cl.size, dtype=int)
        keep = col >= 0
        idx = idx[keep]
        col = col[keep]

    # Accumulate counts (still in sorted-bin order)
    counts = np.zeros((M, C), dtype=int)
    np.add.at(counts, (idx, col), 1)

    # Restore original bin order
    if not assume_sorted_bins:
        inv_order = np.argsort(orig_order)
        counts = counts[inv_order]

    return counts, cluster_ids


def _is_dummy_col(s: pd.Series, tol_decimals: int = 12) -> bool:
    """
    Return True if the column is:
      - boolean dtype, or
      - numeric with unique values subset of {0, 1} (allowing 0.0/1.0).
    """
    if pdt.is_bool_dtype(s):
        return True
    x = pd.to_numeric(s, errors='coerce').dropna().unique()
    if x.size == 0:
        return False
    x = np.round(x.astype(float), tol_decimals)
    return np.isin(x, [0.0, 1.0]).all()


def selective_zscore(
    df: pd.DataFrame,
    *,
    # treat centered & its square as “do not scale”
    centered_suffixes=('_c', '_c2'),
    zscored_suffixes=('_z', '_z2'),    # already standardized → skip
    mean_tol: float = 1e-8,              # if mean≈0 and std≈1, assume already z-scored
    std_tol: float = 1e-6,
    ddof: int = 0
):
    """
    Z-score only the appropriate continuous columns.

    We **skip** columns that are:
      • dummies (0/1) or boolean,
      • already centered or squared-centered (name ends with any of `centered_suffixes`),
      • already z-scored or squared-z (name ends with any of `zscored_suffixes`),
      • near-constant (std ~ 0),
      • the intercept column named 'const' (if present).

    Returns
    -------
    out : DataFrame
        A copy of df where selected columns are z-scored (NaNs preserved).
    scaled : list[str]
        Names of the columns that were actually scaled.
    """
    out = df.copy()
    scaled: list[str] = []

    # Work only on numeric dtypes; strings/objects are ignored automatically.
    for col in out.select_dtypes(include='number').columns:
        # 1) never touch an intercept if it's already present
        if col == 'const':
            continue

        # 2) respect naming conventions: *_c / *_c2 / *_z / *_z2 are left alone
        if col.endswith(centered_suffixes) or col.endswith(zscored_suffixes):
            continue

        s = out[col]

        # 3) skip dummies / boolean indicators
        if _is_dummy_col(s):
            continue

        # 4) compute stats on numeric view (NaNs preserved)
        x = pd.to_numeric(s, errors='coerce')
        m = x.mean()
        sd = x.std(ddof=ddof)

        # 5) skip near-constant (or non-finite std)
        if not np.isfinite(sd) or sd <= 1e-12:
            continue

        # 6) if it already looks z-scored (mean≈0, std≈1), skip
        if abs(m) < mean_tol and abs(sd - 1.0) < std_tol:
            continue

        # 7) z-score; this preserves NaNs exactly where they were
        out[col] = (x - m) / sd
        scaled.append(col)

    return out, scaled


def make_new_seg_info_for_stop_design(stops_with_stats, closest_stop_to_capture_df, monkey_information):

    new_seg_info = event_binning.pick_event_window(stops_with_stats,
                                                    pre_s=0.2, post_s=1.0, min_pre_bins=1, min_post_bins=20, bin_dt=0.04)

    if 'stop_id' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df = get_stops_utils.add_stop_id_to_closest_stop_to_capture_df(
            closest_stop_to_capture_df,
            monkey_information,
        )
        
    if 'captured' not in new_seg_info.columns:
        closest_stop_to_capture_df['captured'] = 1
        new_seg_info = new_seg_info.merge(closest_stop_to_capture_df[['stop_id', 'captured']].drop_duplicates(), on='stop_id', how='left')
        new_seg_info['captured'] = new_seg_info['captured'].fillna(0)
        
    new_seg_info['event_id'] = new_seg_info['stop_id']
    new_seg_info['event_time'] = new_seg_info['stop_time']

    return new_seg_info

def _event_windows_to_bins2d_local(
    picked_windows,
    *,
    event_id_col='event_id',
    event_time_col='event_time',
    win_t0_col='new_seg_start_time',
    win_t1_col='new_seg_end_time',
    n_pre_col='n_pre_bins',
    n_post_col='n_post_bins',
    ok_col='ok_window',
    only_ok=True,
    bin_dt=None,
    tol=1e-9,
):
    """
    Turn event-centered windows into per-event fixed-width bins (LOCAL mode).

    Produces:
      - bins_2d: (N_bins, 2) array of [left, right] for each bin across all events
      - meta: tidy DataFrame with per-bin metadata
    """

    df = picked_windows.copy()

    # Optional filter
    if only_ok and ok_col in df.columns:
        df = df[df[ok_col].astype(bool)].copy()

    required = [
        event_id_col, event_time_col,
        win_t0_col, win_t1_col,
        n_pre_col, n_post_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'missing required columns: {missing}')

    # --------------------------------------------------
    # Infer bin_dt if needed
    # --------------------------------------------------
    if bin_dt is None:
        dts = []
        for _, r in df.iterrows():
            npre = int(r[n_pre_col])
            npost = int(r[n_post_col])
            if npre > 0:
                dts.append(
                    (float(r[event_time_col]) - float(r[win_t0_col])) / npre
                )
            if npost > 0:
                dts.append(
                    (float(r[win_t1_col]) - float(r[event_time_col])) / npost
                )
        if not dts:
            raise ValueError(
                'cannot infer bin_dt: no rows with positive pre/post bin counts'
            )
        bin_dt = float(np.median(dts))

    bins_list = []
    meta_rows = []

    # --------------------------------------------------
    # Main loop over events
    # --------------------------------------------------
    for _, r in df.iterrows():
        event_id = r[event_id_col]
        s = float(r[event_time_col])
        npre = int(r[n_pre_col])
        npost = int(r[n_post_col])
        n_bins = npre + npost
        if n_bins <= 0:
            continue

        # Build local bins centered on event
        left0 = s - npre * bin_dt
        lefts = left0 + bin_dt * np.arange(n_bins)
        rights = lefts + bin_dt
        centers = 0.5 * (lefts + rights)

        # Clip tiny numerical drift to window
        t0 = float(r[win_t0_col])
        t1 = float(r[win_t1_col])
        if lefts[0] < t0 - tol:
            lefts[0] = t0
        if rights[-1] > t1 + tol:
            rights[-1] = t1

        bins_list.append(np.column_stack([lefts, rights]))

        # Pre / post flags
        is_pre = np.zeros(n_bins, dtype=bool)
        if npre > 0:
            is_pre[:npre] = True

        meta_rows.append(pd.DataFrame({
            'event_id': event_id,
            'k_within_seg': np.arange(n_bins, dtype=int),
            'is_pre': is_pre,
            't_left': lefts,
            't_right': rights,
            't_center': centers,
            'rel_left': lefts - s,
            'rel_right': rights - s,
            'rel_center': centers - s,
            'exposure_s': np.full(n_bins, bin_dt),
            'event_time': np.full(n_bins, s),
        }))

    # --------------------------------------------------
    # Empty case
    # --------------------------------------------------
    if not bins_list:
        empty_cols = [
            'event_id', 'k_within_seg', 'is_pre',
            't_left', 't_right', 't_center',
            'rel_left', 'rel_right', 'rel_center',
            'exposure_s', 'event_time', 'bin',
        ]
        return np.zeros((0, 2), float), pd.DataFrame(columns=empty_cols)

    # --------------------------------------------------
    # Concatenate + global ordering
    # --------------------------------------------------
    bins_2d = np.vstack(bins_list)
    meta = pd.concat(meta_rows, ignore_index=True)

    order = np.argsort(meta['t_left'].to_numpy(dtype=float))
    bins_2d = bins_2d[order]
    meta = meta.iloc[order].reset_index(drop=True)

    meta['bin'] = np.arange(len(meta), dtype=int)

    return bins_2d, meta


def _event_windows_to_bins2d_global(
    picked_windows,
    *,
    global_bins_2d,
    event_id_col='event_id',
    event_time_col='event_time',
    win_t0_col='new_seg_start_time',
    win_t1_col='new_seg_end_time',
    n_pre_col='n_pre_bins',
    n_post_col='n_post_bins',
    ok_col='ok_window',
    only_ok=True,
    enforce_exact_counts=False,
):
    """
    Global-bin mode:
    Select bins directly from global_bins_2d that fall within each event window.

    No local bins are constructed.
    bin_dt is ignored by design.
    """

    df = picked_windows.copy()

    if only_ok and ok_col in df.columns:
        df = df[df[ok_col].astype(bool)].copy()

    required = [
        event_id_col, event_time_col,
        win_t0_col, win_t1_col,
        n_pre_col, n_post_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'missing required columns: {missing}')

    global_bins_2d = np.asarray(global_bins_2d, dtype=float)
    if global_bins_2d.ndim != 2 or global_bins_2d.shape[1] != 2:
        raise ValueError('global_bins_2d must have shape (B, 2)')

    g_left = global_bins_2d[:, 0]
    g_right = global_bins_2d[:, 1]
    g_center = 0.5 * (g_left + g_right)

    bins_list = []
    meta_rows = []

    for _, r in df.iterrows():
        event_id = r[event_id_col]
        s = float(r[event_time_col])
        t0 = float(r[win_t0_col])
        t1 = float(r[win_t1_col])
        npre = int(r[n_pre_col])
        npost = int(r[n_post_col])

        # bins fully inside the window
        in_window = (g_left >= t0) & (g_right <= t1)
        if not np.any(in_window):
            continue

        # pre / post relative to event
        is_pre = g_right <= s
        is_post = g_left >= s

        pre_idx = np.where(in_window & is_pre)[0]
        post_idx = np.where(in_window & is_post)[0]

        # choose bins
        if enforce_exact_counts:
            if (npre > 0 and pre_idx.size < npre) or (npost > 0 and post_idx.size < npost):
                continue  # drop this event entirely

            pre_idx = pre_idx[-npre:] if npre > 0 else np.array([], dtype=int)
            post_idx = post_idx[:npost] if npost > 0 else np.array([], dtype=int)
        else:
            pre_idx = pre_idx[-npre:] if npre > 0 else np.array([], dtype=int)
            post_idx = post_idx[:npost] if npost > 0 else np.array([], dtype=int)

        sel_idx = np.concatenate([pre_idx, post_idx])
        if sel_idx.size == 0:
            continue

        bins_list.append(global_bins_2d[sel_idx])

        k_within = np.arange(sel_idx.size, dtype=int)
        is_pre_flag = np.zeros(sel_idx.size, dtype=bool)
        if pre_idx.size > 0:
            is_pre_flag[:pre_idx.size] = True

        meta_rows.append(pd.DataFrame({
            'event_id': event_id,
            'global_bin': sel_idx.astype(int),
            'k_within_seg': k_within,
            'is_pre': is_pre_flag,
            't_left': g_left[sel_idx],
            't_right': g_right[sel_idx],
            't_center': g_center[sel_idx],
            'rel_left': g_left[sel_idx] - s,
            'rel_right': g_right[sel_idx] - s,
            'rel_center': g_center[sel_idx] - s,
            'event_time': np.full(sel_idx.size, s),
        }))

    if not bins_list:
        empty_cols = [
            'event_id', 'global_bin', 'k_within_seg', 'is_pre',
            't_left', 't_right', 't_center',
            'rel_left', 'rel_right', 'rel_center',
            'event_time', 'bin',
        ]
        return np.zeros((0, 2)), pd.DataFrame(columns=empty_cols)

    bins_2d = np.vstack(bins_list)
    meta = pd.concat(meta_rows, ignore_index=True)

    # stable time ordering
    order = np.argsort(meta['t_left'].to_numpy())
    bins_2d = bins_2d[order]
    meta = meta.iloc[order].reset_index(drop=True)
    meta['bin'] = np.arange(len(meta), dtype=int)

    return bins_2d, meta


def event_windows_to_bins2d(
    picked_windows,
    *,
    global_bins_2d=None,
    bin_dt=None,
    enforce_exact_counts=False,
    **kwargs,
):
    """
    Turn event-centered windows into bins.

    Mode is inferred:
      - global mode if global_bins_2d is provided
      - local mode otherwise
    """

    if global_bins_2d is not None:
        if bin_dt is not None:
            print('bin_dt is ignored when global_bins_2d is provided')

        return _event_windows_to_bins2d_global(
            picked_windows,
            global_bins_2d=global_bins_2d,
            enforce_exact_counts=enforce_exact_counts,
            **kwargs,
        )

    # ---- local mode (original behavior) ----
    if bin_dt is None:
        raise ValueError('bin_dt is required when global_bins_2d is not provided')

    return _event_windows_to_bins2d_local(
        picked_windows,
        bin_dt=bin_dt,
        **kwargs,
    )
