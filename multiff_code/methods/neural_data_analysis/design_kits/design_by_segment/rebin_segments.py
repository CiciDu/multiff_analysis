# new script
# new script

from data_wrangling import specific_utils
from planning_analysis.plan_indicators import diff_in_curv_utils
import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_around_event.event_binning import (
    build_bin_assignments,
    bin_timeseries_weighted,
    bin_spikes_by_cluster,
    event_windows_to_bins2d,
)

import numpy as np
import pandas as pd



def _make_left_hold_signal_from_df(
    df,
    *,
    time_col,
    value_cols,
    extra_cols=None,
    t_end=None,
):
    """
    Build a left-hold (sample-and-hold) signal representation from a dataframe.

    Returns
    -------
    times : (T+1,) float
        Strictly increasing time knots defining intervals [times[i], times[i+1]).
    values : (T, D) float
        Values held over each interval.
    extras : dict[str, np.ndarray]
        Optional per-sample arrays aligned to values (length T), e.g. old segment ids.
    """
    if extra_cols is None:
        extra_cols = []

    cols = list(value_cols) + list(extra_cols)
    df0 = df.loc[:, cols].copy()

    # Sort + drop duplicate timestamps (keep first occurrence)
    df0 = df0.sort_values(time_col, kind='mergesort')
    df0 = df0.drop_duplicates(subset=[time_col], keep='first')

    times0 = df0[time_col].to_numpy(dtype=float)
    if times0.size == 0:
        return np.array([], dtype=float), np.empty((0, len(value_cols)), dtype=float), {}

    values = df0[value_cols].to_numpy(dtype=float)

    extras = {}
    for c in extra_cols:
        # keep raw dtype (often int)
        extras[c] = df0[c].to_numpy()

    # Close the last interval
    if t_end is None:
        # Ensure a positive last interval even if user didn't provide t_end
        times = np.r_[times0, times0[-1] + 0.01]
    else:
        t_end = float(t_end)
        if not np.isfinite(t_end) or t_end <= times0[-1]:
            times = np.r_[times0, times0[-1] + 0.01]
        else:
            times = np.r_[times0, t_end]

    return times, values, extras


def _assign_intervals_to_new_segments(
    times,
    new_seg_info,
    *,
    seg_id_col='new_segment',
    start_col='new_seg_start_time',
    end_col='new_seg_end_time',
):
    """
    Assign each left-hold interval [times[i], times[i+1]) to a new_segment id using midpoints.

    Returns
    -------
    sample_segment : (T,) int
        new_segment id for each interval, or -1 if none.
    seg_ids : (S,) int
        Segment ids in sorted order used for assignment.
    seg_starts : (S,) float
    seg_ends : (S,) float
    """
    if times.size < 2:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    seg_df = new_seg_info.loc[:, [seg_id_col, start_col, end_col]].copy()
    seg_df = seg_df.sort_values(start_col, kind='mergesort')

    seg_ids = seg_df[seg_id_col].to_numpy(dtype=int)
    seg_starts = seg_df[start_col].to_numpy(dtype=float)
    seg_ends = seg_df[end_col].to_numpy(dtype=float)

    # Midpoints of sample intervals
    mid = 0.5 * (times[:-1] + times[1:])

    # Find candidate segment index for each midpoint: last start <= mid
    idx = np.searchsorted(seg_starts, mid, side='right') - 1

    sample_segment = np.full(mid.shape, -1, dtype=int)
    valid = (idx >= 0) & (mid < seg_ends[idx])

    # Map to segment ids
    sample_segment[valid] = seg_ids[idx[valid]]

    return sample_segment, seg_ids, seg_starts, seg_ends


def rebin_all_segments_global_bins(
    df,
    new_seg_info,
    bins_2d=None,
    *,
    time_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    require_full_bin=False,
    add_support_duration=False,
):
    """
    Rebin across ALL segments using a predefined global bins_2d, vectorized over segments.

    Public interface intentionally matches rebin_all_segments_local_bins.

    Parameters
    ----------
    bins_2d : (B, 2) array-like
        Global bins [left, right) in absolute time. Required for global mode.
    bin_width : unused here (kept for signature symmetry)
    respect_old_segment : bool
        If True, only samples from the same old `segment_col` are used for each new segment,
        requiring new_seg_info to have a column named `segment_col`.
    require_full_bin : bool
        If True, only keep bins fully contained in the segment window.
    add_support_duration : bool
        If True, add bin_support_dt = total overlap duration accumulated into that (segment, new_bin).
    Returns
    -------
    out : pd.DataFrame
        Rebinned values per (`new_segment`, `new_bin`).
    bin_edges : np.ndarray, optional
        Only returned when add_bin_edges=True. Shape (len(out), 2) with [left, right).
    """

    if how not in ('mean', 'sum'):
        raise ValueError("how must be 'mean' or 'sum'")

    if bins_2d is None:
        raise ValueError('bins_2d is required for global-bins mode')

    bins_2d = np.asarray(bins_2d, dtype=float)
    if bins_2d.ndim != 2 or bins_2d.shape[1] != 2:
        raise ValueError('bins_2d must have shape (B, 2)')

    # Identify value columns
    exclude = {
        # time_col,
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Build signal
    extra_cols = [segment_col] if respect_old_segment else []
    # close to overall max segment end so last interval is meaningful w.r.t. segments
    t_end = float(np.nanmax(new_seg_info['new_seg_end_time'].to_numpy(dtype=float)))
    times, values, extras = _make_left_hold_signal_from_df(
        df,
        time_col=time_col,
        value_cols=value_cols,
        extra_cols=extra_cols,
        t_end=t_end,
    )
    if times.size < 2:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Assign each interval to a new_segment id
    sample_segment, _, _, _ = _assign_intervals_to_new_segments(times, new_seg_info)
    if sample_segment.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    valid_samples = sample_segment >= 0
    if not np.any(valid_samples):
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Global overlaps ONCE
    sample_idx, bin_idx, dt_arr, _ = build_bin_assignments(
        times,
        bins_2d,
        assume_sorted=True,
        check_nonoverlap=False,
    )
    if sample_idx.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Keep only overlaps whose sample interval belongs to some new segment
    keep = valid_samples[sample_idx]
    sample_idx = sample_idx[keep]
    bin_idx = bin_idx[keep]
    dt_arr = dt_arr[keep]

    new_seg_id = sample_segment[sample_idx].astype(int)

    # respect_old_segment mask (optional)
    if respect_old_segment:
        if segment_col not in new_seg_info.columns:
            raise ValueError(f"respect_old_segment=True requires new_seg_info to have column '{segment_col}'")

        old_seg_per_sample = extras[segment_col]
        required_old_seg = new_seg_info.set_index('new_segment')[segment_col]

        # required_old_seg indexed by new segment id
        req = required_old_seg.loc[new_seg_id].to_numpy()
        ok_old = old_seg_per_sample[sample_idx] == req

        sample_idx = sample_idx[ok_old]
        bin_idx = bin_idx[ok_old]
        dt_arr = dt_arr[ok_old]
        new_seg_id = new_seg_id[ok_old]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # require_full_bin mask (optional)
    if require_full_bin:
        seg_starts = new_seg_info.set_index('new_segment')['new_seg_start_time']
        seg_ends = new_seg_info.set_index('new_segment')['new_seg_end_time']

        seg_t0 = seg_starts.loc[new_seg_id].to_numpy(dtype=float)
        seg_t1 = seg_ends.loc[new_seg_id].to_numpy(dtype=float)

        b0 = bins_2d[bin_idx, 0]
        b1 = bins_2d[bin_idx, 1]
        full = (b0 >= seg_t0) & (b1 <= seg_t1)

        sample_idx = sample_idx[full]
        bin_idx = bin_idx[full]
        dt_arr = dt_arr[full]
        new_seg_id = new_seg_id[full]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Sparse accumulation into (new_segment, global_bin)
    n_bins = bins_2d.shape[0]
    seg_ids = new_seg_info['new_segment'].to_numpy(dtype=int)
    seg_max = int(np.max(seg_ids)) if seg_ids.size else -1
    n_segments = seg_max + 1

    n_dim = values.shape[1]
    acc = np.zeros((n_segments, n_bins, n_dim), dtype=float)
    time_acc = np.zeros((n_segments, n_bins), dtype=float)

    np.add.at(acc, (new_seg_id, bin_idx), values[sample_idx] * dt_arr[:, None])
    np.add.at(time_acc, (new_seg_id, bin_idx), dt_arr)

    valid = time_acc > 0
    if not np.any(valid):
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    if how == 'mean':
        acc[valid] /= time_acc[valid, None]

    out_seg, out_bin = np.nonzero(valid)

    out = pd.DataFrame(acc[out_seg, out_bin], columns=value_cols)
    out.insert(0, 'new_bin', out_bin.astype(int))
    out.insert(0, 'new_segment', out_seg.astype(int))

    if add_support_duration:
        out['bin_support_dt'] = time_acc[out_seg, out_bin]

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
            + ([segment_col] if segment_col in new_seg_info.columns else [])
        ],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    if add_bin_edges:
        bin_edges = bins_2d[out_bin, :]
        return out, bin_edges
    return out


def rebin_all_segments_local_bins(
    df,
    new_seg_info,
    bin_width=None,
    *,
    time_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    add_support_duration=False,
):
    """
    Provably identical refactor of the original rebin_segment_data
    (local / segment-defined bins).
    Returns
    -------
    out : pd.DataFrame
        Rebinned values per (`new_segment`, `new_bin`).
    bin_edges : np.ndarray, optional
        Only returned when add_bin_edges=True. Shape (len(out), 2) with [left, right).
    """

    if how not in ('mean', 'sum'):
        raise ValueError("how must be 'mean' or 'sum'")

    if bin_width is None:
        raise ValueError('bin_width is required for local-bins mode')

    dt = float(bin_width)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('bin_width must be positive')

    # --------------------------------------------------
    # Identify value columns (IDENTICAL logic)
    # --------------------------------------------------
    exclude = {
        # time_col,
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # --------------------------------------------------
    # Prepare data access (IDENTICAL logic)
    # --------------------------------------------------
    if respect_old_segment:
        df_by_segment = {
            seg: g.sort_values(time_col)
            for seg, g in df.groupby(segment_col)
        }
    else:
        df_sorted = df.sort_values(time_col)
        times_all = df_sorted[time_col].to_numpy()

    out_blocks = []
    edges_blocks = [] if add_bin_edges else None

    # --------------------------------------------------
    # Loop over new segments (IDENTICAL logic)
    # --------------------------------------------------
    for _, r in new_seg_info.sort_values('new_segment').iterrows():
        new_seg_id = int(r['new_segment'])
        t0 = float(r['new_seg_start_time'])
        t1 = float(r['new_seg_end_time'])

        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        # ---- select data (IDENTICAL) ----
        if respect_old_segment:
            old_seg_id = r[segment_col]
            seg_df = df_by_segment.get(old_seg_id)
            if seg_df is None:
                continue

            seg_df = seg_df[
                (seg_df[time_col] >= t0) &
                (seg_df[time_col] < t1)
            ]
        else:
            i0 = np.searchsorted(times_all, t0, side='left')
            i1 = np.searchsorted(times_all, t1, side='left')
            seg_df = df_sorted.iloc[i0:i1]

        if seg_df.empty:
            continue

        # --------------------------------------------------
        # Build left-hold signal (IDENTICAL)
        # --------------------------------------------------
        times = seg_df[time_col].to_numpy(dtype=float)

        times, uniq_idx = np.unique(times, return_index=True)
        values = seg_df.iloc[uniq_idx][value_cols].to_numpy(dtype=float)

        # close final interval at segment end
        if times[-1] < t1:
            times = np.r_[times, times[-1] + 0.01]

        if times.size < 2:
            continue

        # --------------------------------------------------
        # Build local bins (IDENTICAL)
        # --------------------------------------------------
        n_bins = int(np.floor((t1 - t0) / dt))
        if n_bins <= 0:
            continue

        lefts = t0 + dt * np.arange(n_bins)
        rights = lefts + dt
        seg_bins_2d = np.column_stack([lefts, rights])

        # --------------------------------------------------
        # Interval → bin overlap (IDENTICAL)
        # --------------------------------------------------
        sample_idx, bin_idx, dt_arr, _ = build_bin_assignments(
            times,
            seg_bins_2d,
            assume_sorted=True,
            check_nonoverlap=False,
        )
        if sample_idx.size == 0:
            continue

        # --------------------------------------------------
        # Time-weighted aggregation (IDENTICAL)
        # --------------------------------------------------
        weighted_vals, _, used_bins = bin_timeseries_weighted(
            values[sample_idx],
            dt_arr,
            bin_idx,
            how=how,
        )

        if weighted_vals.ndim == 1:
            weighted_vals = weighted_vals[None, :]

        block = pd.DataFrame(weighted_vals, columns=value_cols)
        block.insert(0, 'new_bin', used_bins.astype(int))
        block.insert(0, 'new_segment', new_seg_id)

        # --------------------------------------------------
        # Optional extras (safe additions)
        # --------------------------------------------------
        if add_support_duration:
            support = np.zeros(seg_bins_2d.shape[0], dtype=float)
            np.add.at(support, bin_idx, dt_arr)
            block['bin_support_dt'] = support[used_bins.astype(int)]

        if add_bin_edges:
            edges_blocks.append(seg_bins_2d[used_bins.astype(int), :])

        out_blocks.append(block)

    if not out_blocks:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    out = pd.concat(out_blocks, ignore_index=True)

    # --------------------------------------------------
    # Merge segment metadata (IDENTICAL)
    # --------------------------------------------------
    merge_cols = [
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    ]
    if segment_col in new_seg_info.columns:
        merge_cols.append(segment_col)

    out = out.merge(
        new_seg_info[merge_cols],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    if add_bin_edges:
        bin_edges = np.vstack(edges_blocks) if edges_blocks else np.empty((0, 2))
        return out, bin_edges
    return out



def rebin_all_segments_global_bins_pick_point(
    df,
    new_seg_info,
    bins_2d=None,
    *,
    time_col='time',
    segment_col='segment',
    respect_old_segment=True,
    require_full_bin=False,
    add_bin_edges=False,
):
    """
    Rebin across ALL segments using predefined global bins_2d,
    but select ONE representative point per (new_segment, bin)
    instead of a weighted sum / mean.

    Representative point = sample with MAX overlap duration (dt)
    within that (new_segment, global_bin).

    Parameters
    ----------
    bins_2d : (B, 2) array-like
        Global bins [left, right) in absolute time.
    bin_width : unused (kept for signature symmetry)
    respect_old_segment : bool
        If True, only samples from the same old `segment_col`
        are used for each new segment.
    require_full_bin : bool
        If True, only keep bins fully contained in the segment window.

    Returns
    -------
    out : pd.DataFrame
        Rebinned values per (`new_segment`, `new_bin`)
        using a single representative point.
    bin_edges : np.ndarray, optional
        Only returned when add_bin_edges=True.
    """

    if bins_2d is None:
        raise ValueError('bins_2d is required for global-bins mode')

    bins_2d = np.asarray(bins_2d, dtype=float)
    if bins_2d.ndim != 2 or bins_2d.shape[1] != 2:
        raise ValueError('bins_2d must have shape (B, 2)')

    # Identify value columns
    exclude = {
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Build left-hold signal
    extra_cols = [segment_col] if respect_old_segment else []
    t_end = float(np.nanmax(new_seg_info['new_seg_end_time'].to_numpy(dtype=float)))

    times, values, extras = _make_left_hold_signal_from_df(
        df,
        time_col=time_col,
        value_cols=value_cols,
        extra_cols=extra_cols,
        t_end=t_end,
    )
    if times.size < 2:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Assign each interval to a new segment
    sample_segment, _, _, _ = _assign_intervals_to_new_segments(times, new_seg_info)
    valid_samples = sample_segment >= 0
    if not np.any(valid_samples):
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Global bin overlaps
    sample_idx, bin_idx, dt_arr, _ = build_bin_assignments(
        times,
        bins_2d,
        assume_sorted=True,
        check_nonoverlap=False,
    )
    if sample_idx.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Keep only samples belonging to a new segment
    keep = valid_samples[sample_idx]
    sample_idx = sample_idx[keep]
    bin_idx = bin_idx[keep]
    dt_arr = dt_arr[keep]
    new_seg_id = sample_segment[sample_idx].astype(int)

    # respect_old_segment filter
    if respect_old_segment:
        if segment_col not in new_seg_info.columns:
            raise ValueError(
                f"respect_old_segment=True requires new_seg_info to have column '{segment_col}'"
            )

        old_seg_per_sample = extras[segment_col]
        required_old_seg = new_seg_info.set_index('new_segment')[segment_col]
        req = required_old_seg.loc[new_seg_id].to_numpy()

        ok = old_seg_per_sample[sample_idx] == req
        sample_idx = sample_idx[ok]
        bin_idx = bin_idx[ok]
        dt_arr = dt_arr[ok]
        new_seg_id = new_seg_id[ok]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # require_full_bin filter
    if require_full_bin:
        seg_starts = new_seg_info.set_index('new_segment')['new_seg_start_time']
        seg_ends = new_seg_info.set_index('new_segment')['new_seg_end_time']

        seg_t0 = seg_starts.loc[new_seg_id].to_numpy(dtype=float)
        seg_t1 = seg_ends.loc[new_seg_id].to_numpy(dtype=float)

        b0 = bins_2d[bin_idx, 0]
        b1 = bins_2d[bin_idx, 1]

        full = (b0 >= seg_t0) & (b1 <= seg_t1)

        sample_idx = sample_idx[full]
        bin_idx = bin_idx[full]
        dt_arr = dt_arr[full]
        new_seg_id = new_seg_id[full]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # ------------------------------------------------------------------
    # Pick ONE best sample per (new_segment, bin): max dt
    # ------------------------------------------------------------------
    key = np.stack([new_seg_id, bin_idx], axis=1)

    # stable sort: first by bin, then segment, then -dt
    order = np.lexsort((-dt_arr, bin_idx, new_seg_id))
    key_sorted = key[order]

    # keep first occurrence of each (segment, bin)
    _, first_idx = np.unique(key_sorted, axis=0, return_index=True)
    chosen = order[first_idx]

    out_seg = new_seg_id[chosen]
    out_bin = bin_idx[chosen]
    out_vals = values[sample_idx[chosen]]

    out = pd.DataFrame(out_vals, columns=value_cols)
    out.insert(0, 'new_bin', out_bin.astype(int))
    out.insert(0, 'new_segment', out_seg.astype(int))

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
            + ([segment_col] if segment_col in new_seg_info.columns else [])
        ],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    if add_bin_edges:
        return out, bins_2d[out_bin]

    return out
