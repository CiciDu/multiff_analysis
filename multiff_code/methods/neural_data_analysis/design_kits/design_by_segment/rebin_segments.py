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


def rebin_all_segments_global_bins_pick_point(
    df,
    new_seg_info,
    bins_2d=None,
    *,
    bin_left_col=None,
    bin_right_col=None,
    bin_center_col='time',
    segment_col='segment',
    respect_old_segment=True,
    require_full_bin=False,
    add_bin_edges=False,
):
    """
    Rebin across ALL segments using predefined global bins_2d,
    but select ONE representative old bin/sample per (new_segment, bin)
    instead of a weighted sum / mean.

    Representative point = old interval with MAX overlap duration (dt)
    within that (new_segment, global_bin).

    Old interval interpretation
    ---------------------------
    - If bin_left_col and bin_right_col are provided and present in df,
      use them directly.
    - Otherwise use bin_center_col and infer old edges by midpoint approximation.

    Parameters
    ----------
    bins_2d : (B, 2) array-like
        Global bins [left, right) in absolute time.
    respect_old_segment : bool
        If True, only rows from the same old `segment_col`
        are used for each new segment.
    require_full_bin : bool
        If True, only keep bins fully contained in the segment window.

    Returns
    -------
    out : pd.DataFrame
        Rebinned values per (`new_segment`, `new_bin`)
        using a single representative old interval.
    bin_edges : np.ndarray, optional
        Only returned when add_bin_edges=True.
    """
    if bins_2d is None:
        raise ValueError('bins_2d is required for global-bins mode')

    bins_2d = np.asarray(bins_2d, dtype=float)
    if bins_2d.ndim != 2 or bins_2d.shape[1] != 2:
        raise ValueError('bins_2d must have shape (B, 2)')

    exclude = {
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    edge_cols = {c for c in [bin_left_col, bin_right_col, bin_center_col] if c is not None}

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude and c not in edge_cols
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    extra_cols = [segment_col] if respect_old_segment else []
    old_left, old_right, values, extras = _prepare_old_intervals_from_df(
        df,
        value_cols=value_cols,
        extra_cols=extra_cols,
        bin_left_col=bin_left_col,
        bin_right_col=bin_right_col,
        bin_center_col=bin_center_col,
    )

    if old_left.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Assign each old interval to a new segment by overlap
    n_old = old_left.size
    sample_segment = np.full(n_old, -1, dtype=int)

    seg_rows = new_seg_info.sort_values('new_segment')
    seg_starts = seg_rows['new_seg_start_time'].to_numpy(dtype=float)
    seg_ends = seg_rows['new_seg_end_time'].to_numpy(dtype=float)
    seg_ids = seg_rows['new_segment'].to_numpy(dtype=int)

    for seg_id, t0, t1 in zip(seg_ids, seg_starts, seg_ends):
        overlap = (old_right > t0) & (old_left < t1)
        sample_segment[overlap] = seg_id

    valid_samples = sample_segment >= 0
    if not np.any(valid_samples):
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Global bin overlaps
    sample_idx, bin_idx, dt_arr = _build_bin_assignments_from_intervals(
        old_left,
        old_right,
        bins_2d,
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
        seg_starts_map = new_seg_info.set_index('new_segment')['new_seg_start_time']
        seg_ends_map = new_seg_info.set_index('new_segment')['new_seg_end_time']

        seg_t0 = seg_starts_map.loc[new_seg_id].to_numpy(dtype=float)
        seg_t1 = seg_ends_map.loc[new_seg_id].to_numpy(dtype=float)

        b0 = bins_2d[bin_idx, 0]
        b1 = bins_2d[bin_idx, 1]

        full = (b0 >= seg_t0) & (b1 <= seg_t1)

        sample_idx = sample_idx[full]
        bin_idx = bin_idx[full]
        dt_arr = dt_arr[full]
        new_seg_id = new_seg_id[full]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Pick ONE best sample per (new_segment, bin): max dt
    key = np.stack([new_seg_id, bin_idx], axis=1)

    # stable sort: first by new_segment, then by bin, then by -dt
    order = np.lexsort((-dt_arr, bin_idx, new_seg_id))
    key_sorted = key[order]

    # keep first occurrence of each (new_segment, bin)
    _, first_idx = np.unique(key_sorted, axis=0, return_index=True)
    chosen = order[first_idx]

    out_seg = new_seg_id[chosen]
    out_bin = bin_idx[chosen]
    out_vals = values[sample_idx[chosen]]

    out = pd.DataFrame(out_vals, columns=value_cols)
    out.insert(0, 'new_bin', out_bin.astype(int))
    out.insert(0, 'new_segment', out_seg.astype(int))

    # Preserve original NA behavior for label-like columns
    label_cols = get_integer_label_columns(df, value_cols)
    binary_label_cols = {c for c in label_cols if _is_binary_series(df[c])}
    for c in label_cols:
        if c in out.columns:
            out[c] = _make_nullable_int_label_series(
                out[c],
                binary=(c in binary_label_cols),
            )

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


def get_integer_label_columns(
    df,
    value_cols,
    include_substrings=('index', 'id'),
    exclude_substrings=(),
    atol=1e-8,
):
    """
    Return columns that look like integer label columns.

    Rules
    -----
    A column is selected if:
    1) its name contains one of include_substrings
    2) its name does not contain any exclude_substrings
    3) its non-NA numeric values are all approximately integers

    Additionally:
    - Print columns that are binary (only 0/1 values)
    """
    label_cols = []

    for c in value_cols:
        name = str(c).lower()

        # Must contain at least one include substring
        if not any(sub in name for sub in include_substrings):
            continue

        # Must not contain excluded substrings
        if any(sub in name for sub in exclude_substrings):
            continue

        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if s.empty:
            continue

        x = s.to_numpy(dtype=float)

        # Check integer-like
        if np.all(np.isclose(x, np.round(x), atol=atol, rtol=0)):
            label_cols.append(c)

            # --- NEW: check if binary ---
            unique_vals = np.unique(np.round(x))
            if set(unique_vals).issubset({0, 1}):
                print(f'Binary column detected: {c}')

    return label_cols

def _is_binary_series(s: pd.Series, tol_decimals: int = 12) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    x = pd.to_numeric(s, errors='coerce').dropna().to_numpy()
    if x.size == 0:
        return False
    x = np.round(x.astype(float), tol_decimals)
    return np.isin(x, [0.0, 1.0]).all()


def _prepare_old_intervals_from_df(
    df,
    *,
    value_cols,
    extra_cols=None,
    bin_left_col=None,
    bin_right_col=None,
    bin_center_col=None,
):
    """
    Build explicit old intervals [old_left, old_right) from df.

    Priority
    --------
    1) If both bin_left_col and bin_right_col are provided and present in df,
       use them directly.
    2) Otherwise, use bin_center_col and infer edges by midpoint approximation.

    Returns
    -------
    old_left : (N,) float
    old_right : (N,) float
    values : (N, D) float
    extras : dict[str, np.ndarray]
    """
    if extra_cols is None:
        extra_cols = []

    has_edges = (
        bin_left_col is not None
        and bin_right_col is not None
        and bin_left_col in df.columns
        and bin_right_col in df.columns
    )

    if has_edges:
        cols = [bin_left_col, bin_right_col] + list(value_cols) + list(extra_cols)
        df0 = df.loc[:, cols].copy()

        # Stable sort, then keep first exact interval occurrence
        df0 = df0.sort_values([bin_left_col, bin_right_col], kind='mergesort')
        df0 = df0.drop_duplicates(subset=[bin_left_col, bin_right_col], keep='first')

        old_left = pd.to_numeric(df0[bin_left_col], errors='coerce').to_numpy(dtype=float)
        old_right = pd.to_numeric(df0[bin_right_col], errors='coerce').to_numpy(dtype=float)

    else:
        if bin_center_col is None or bin_center_col not in df.columns:
            raise ValueError(
                'Need either bin_left_col/bin_right_col columns or a valid bin_center_col'
            )

        cols = [bin_center_col] + list(value_cols) + list(extra_cols)
        df0 = df.loc[:, cols].copy()

        # Stable sort, then keep first center occurrence
        df0 = df0.sort_values(bin_center_col, kind='mergesort')
        df0 = df0.drop_duplicates(subset=[bin_center_col], keep='first')

        centers = pd.to_numeric(df0[bin_center_col], errors='coerce').to_numpy(dtype=float)
        if centers.size == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.empty((0, len(value_cols)), dtype=float),
                {},
            )

        if centers.size == 1:
            raise ValueError(
                'Cannot infer bin edges from a single bin center. '
                'Provide bin_left_col/bin_right_col explicitly.'
            )

        # Midpoint approximation
        midpoints = 0.5 * (centers[:-1] + centers[1:])
        first_left = centers[0] - 0.5 * (centers[1] - centers[0])
        last_right = centers[-1] + 0.5 * (centers[-1] - centers[-2])

        edges = np.r_[first_left, midpoints, last_right]
        old_left = edges[:-1]
        old_right = edges[1:]

    values = df0[value_cols].to_numpy(dtype=float)

    extras = {}
    for c in extra_cols:
        extras[c] = df0[c].to_numpy()

    valid = (
        np.isfinite(old_left)
        & np.isfinite(old_right)
        & (old_right > old_left)
    )

    old_left = old_left[valid]
    old_right = old_right[valid]
    values = values[valid]

    for c in extra_cols:
        extras[c] = extras[c][valid]

    return old_left, old_right, values, extras


def _build_bin_assignments_from_intervals(
    old_left,
    old_right,
    bins_2d,
):
    """
    Compute overlap assignments between old intervals [old_left, old_right)
    and new bins [bin_left, bin_right).

    Returns
    -------
    sample_idx : (K,) int
    bin_idx : (K,) int
    dt_arr : (K,) float
    """
    old_left = np.asarray(old_left, dtype=float)
    old_right = np.asarray(old_right, dtype=float)
    bins_2d = np.asarray(bins_2d, dtype=float)

    if old_left.size == 0 or bins_2d.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    sample_idx_list = []
    bin_idx_list = []
    dt_list = []

    i = 0
    j = 0
    n_old = old_left.size
    n_bins = bins_2d.shape[0]

    while i < n_old and j < n_bins:
        s0 = old_left[i]
        s1 = old_right[i]
        b0 = bins_2d[j, 0]
        b1 = bins_2d[j, 1]

        if s1 <= b0:
            i += 1
            continue
        if b1 <= s0:
            j += 1
            continue

        dt = min(s1, b1) - max(s0, b0)
        if dt > 0:
            sample_idx_list.append(i)
            bin_idx_list.append(j)
            dt_list.append(dt)

        if s1 <= b1:
            i += 1
        else:
            j += 1

    return (
        np.asarray(sample_idx_list, dtype=int),
        np.asarray(bin_idx_list, dtype=int),
        np.asarray(dt_list, dtype=float),
    )


def _make_nullable_int_label_series(values_1d, *, binary=False):
    """
    Cast label values to nullable Int64 while preserving NA.
    For binary labels, coerce non-NA values to 0/1 only.
    """
    s = pd.to_numeric(pd.Series(values_1d), errors='coerce').astype('Int64')

    if binary:
        non_na = s.notna()
        s.loc[non_na] = (s.loc[non_na] > 0).astype(int)
        s = s.astype('Int64')

    return s


def rebin_all_segments_local_bins(
    df,
    new_seg_info,
    bin_width=None,
    *,
    bin_left_col=None,
    bin_right_col=None,
    bin_center_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    add_support_duration=False,
):
    """
    Local / segment-defined bins with:
      - explicit old intervals from bin_left/bin_right if available
      - otherwise midpoint approximation from bin_center_col
      - binary cols preserved as binary (ANY-1)
      - label cols ('index'/'id') set via max-overlap representative sample per bin
      - label cols preserve explicit NA as nullable Int64 (same behavior as original)
    """
    if how not in ('mean', 'sum'):
        raise ValueError("how must be 'mean' or 'sum'")

    if bin_width is None:
        raise ValueError('bin_width is required for local-bins mode')

    dt = float(bin_width)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('bin_width must be positive')

    exclude = {
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    edge_cols = {c for c in [bin_left_col, bin_right_col, bin_center_col] if c is not None}

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude and c not in edge_cols
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Label cols
    label_cols = get_integer_label_columns(df, value_cols)
    binary_label_cols = {c for c in label_cols if _is_binary_series(df[c])}

    # Prepare old intervals
    extra_cols = [segment_col] if respect_old_segment else []
    old_left, old_right, values, extras = _prepare_old_intervals_from_df(
        df,
        value_cols=value_cols,
        extra_cols=extra_cols,
        bin_left_col=bin_left_col,
        bin_right_col=bin_right_col,
        bin_center_col=bin_center_col,
    )

    if old_left.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    out_blocks = []
    edges_blocks = [] if add_bin_edges else None

    for _, r in new_seg_info.sort_values('new_segment').iterrows():
        new_seg_id = int(r['new_segment'])
        t0 = float(r['new_seg_start_time'])
        t1 = float(r['new_seg_end_time'])

        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        # Select old intervals that overlap this segment
        if respect_old_segment:
            old_seg_id = r[segment_col]
            keep = (
                (extras[segment_col] == old_seg_id)
                & (old_right > t0)
                & (old_left < t1)
            )
        else:
            keep = (old_right > t0) & (old_left < t1)

        if not np.any(keep):
            continue

        seg_old_left = old_left[keep]
        seg_old_right = old_right[keep]
        seg_values = values[keep]

        # Local bins
        n_bins = int(np.floor((t1 - t0) / dt))
        if n_bins <= 0:
            continue

        lefts = t0 + dt * np.arange(n_bins)
        rights = lefts + dt
        seg_bins_2d = np.column_stack([lefts, rights])

        # Overlaps
        sample_idx, bin_idx, dt_arr = _build_bin_assignments_from_intervals(
            seg_old_left,
            seg_old_right,
            seg_bins_2d,
        )
        if sample_idx.size == 0:
            continue

        # Aggregate using existing helper to preserve original binary behavior
        weighted_vals, _, used_bins = bin_timeseries_weighted(
            seg_values[sample_idx],
            dt_arr,
            bin_idx,
            how=how,
            preserve_binary=True,
        )
        if weighted_vals.ndim == 1:
            weighted_vals = weighted_vals[None, :]

        block = pd.DataFrame(weighted_vals, columns=value_cols)
        block.insert(0, 'bin_in_new_seg', used_bins.astype(int))
        block.insert(0, 'new_segment', new_seg_id)

        # Label cols via representative max-dt sample per bin
        if label_cols:
            label_values_raw = seg_values[:, [value_cols.index(c) for c in label_cols]]

            order = np.lexsort((-dt_arr, bin_idx))  # bin asc, dt desc
            bin_sorted = bin_idx[order]
            _, first_pos = np.unique(bin_sorted, return_index=True)
            chosen = order[first_pos]

            chosen_bins = bin_idx[chosen]
            chosen_sample = sample_idx[chosen]
            bin_to_rep_sample = {int(b): int(s) for b, s in zip(chosen_bins, chosen_sample)}

            rep_mat = np.zeros((used_bins.size, len(label_cols)), dtype=object)
            for i, b in enumerate(used_bins.astype(int)):
                sidx = bin_to_rep_sample.get(int(b), None)
                if sidx is None:
                    rep_mat[i, :] = np.nan
                else:
                    rep_mat[i, :] = label_values_raw[sidx, :]

            for j, c in enumerate(label_cols):
                block[c] = _make_nullable_int_label_series(
                    rep_mat[:, j],
                    binary=(c in binary_label_cols),
                )

        # Extras
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

    # Merge segment metadata
    merge_cols = ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
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
    
    
def rebin_all_segments_global_bins(
    df,
    new_seg_info,
    bins_2d=None,
    *,
    bin_left_col=None,
    bin_right_col=None,
    bin_center_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    require_full_bin=False,
    add_support_duration=False,
):
    if how not in ('mean', 'sum'):
        raise ValueError("how must be 'mean' or 'sum'")

    if bins_2d is None:
        raise ValueError('bins_2d is required for global-bins mode')

    bins_2d = np.asarray(bins_2d, dtype=float)
    if bins_2d.ndim != 2 or bins_2d.shape[1] != 2:
        raise ValueError('bins_2d must have shape (B, 2)')

    exclude = {
        'new_segment',
        'new_seg_start_time',
        'new_seg_end_time',
        'new_seg_duration',
    }
    if respect_old_segment:
        exclude.add(segment_col)

    edge_cols = {c for c in [bin_left_col, bin_right_col, bin_center_col] if c is not None}

    value_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude and c not in edge_cols
    ]
    if not value_cols:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    binary_cols = [c for c in value_cols if _is_binary_series(df[c])]
    binary_idx = (
        np.array([value_cols.index(c) for c in binary_cols], dtype=int)
        if binary_cols else np.array([], dtype=int)
    )

    label_cols = get_integer_label_columns(df, value_cols)
    label_idx = (
        np.array([value_cols.index(c) for c in label_cols], dtype=int)
        if label_cols else np.array([], dtype=int)
    )

    binary_label_cols = {c for c in label_cols if _is_binary_series(df[c])}

    extra_cols = [segment_col] if respect_old_segment else []
    old_left, old_right, values, extras = _prepare_old_intervals_from_df(
        df,
        value_cols=value_cols,
        extra_cols=extra_cols,
        bin_left_col=bin_left_col,
        bin_right_col=bin_right_col,
        bin_center_col=bin_center_col,
    )

    if old_left.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # Assign old intervals → new segments
    n_old = old_left.size
    sample_segment = np.full(n_old, -1, dtype=int)

    seg_rows = new_seg_info.sort_values('new_segment')
    seg_starts = seg_rows['new_seg_start_time'].to_numpy(dtype=float)
    seg_ends = seg_rows['new_seg_end_time'].to_numpy(dtype=float)
    seg_ids = seg_rows['new_segment'].to_numpy(dtype=int)

    for seg_id, t0, t1 in zip(seg_ids, seg_starts, seg_ends):
        overlap = (old_right > t0) & (old_left < t1)
        sample_segment[overlap] = seg_id

    valid_samples = sample_segment >= 0
    if not np.any(valid_samples):
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    sample_idx, bin_idx, dt_arr = _build_bin_assignments_from_intervals(
        old_left,
        old_right,
        bins_2d,
    )
    if sample_idx.size == 0:
        return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    keep = valid_samples[sample_idx]
    sample_idx = sample_idx[keep]
    bin_idx = bin_idx[keep]
    dt_arr = dt_arr[keep]
    new_seg_id = sample_segment[sample_idx].astype(int)

    if respect_old_segment:
        if segment_col not in new_seg_info.columns:
            raise ValueError(
                f"respect_old_segment=True requires new_seg_info to have column '{segment_col}'"
            )

        old_seg_per_sample = extras[segment_col]
        required_old_seg = new_seg_info.set_index('new_segment')[segment_col]

        req = required_old_seg.loc[new_seg_id].to_numpy()
        ok_old = old_seg_per_sample[sample_idx] == req

        sample_idx = sample_idx[ok_old]
        bin_idx = bin_idx[ok_old]
        dt_arr = dt_arr[ok_old]
        new_seg_id = new_seg_id[ok_old]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    if require_full_bin:
        seg_starts_map = new_seg_info.set_index('new_segment')['new_seg_start_time']
        seg_ends_map = new_seg_info.set_index('new_segment')['new_seg_end_time']

        seg_t0 = seg_starts_map.loc[new_seg_id].to_numpy(dtype=float)
        seg_t1 = seg_ends_map.loc[new_seg_id].to_numpy(dtype=float)

        b0 = bins_2d[bin_idx, 0]
        b1 = bins_2d[bin_idx, 1]
        full = (b0 >= seg_t0) & (b1 <= seg_t1)

        sample_idx = sample_idx[full]
        bin_idx = bin_idx[full]
        dt_arr = dt_arr[full]
        new_seg_id = new_seg_id[full]

        if sample_idx.size == 0:
            return (pd.DataFrame(), np.empty((0, 2))) if add_bin_edges else pd.DataFrame()

    # -------------------------------------------------
    # ✅ TRUE sparse accumulation (compressed pairs)
    # -------------------------------------------------
    n_dim = values.shape[1]

    pair_keys = np.stack([new_seg_id, bin_idx], axis=1)
    pair_unique, pair_ids = np.unique(pair_keys, axis=0, return_inverse=True)

    n_pairs = pair_unique.shape[0]

    # Time accumulation
    time_acc = np.bincount(pair_ids, weights=dt_arr, minlength=n_pairs)

    # Value accumulation
    acc = np.zeros((n_pairs, n_dim), dtype=float)
    for d in range(n_dim):
        acc[:, d] = np.bincount(
            pair_ids,
            weights=values[sample_idx, d] * dt_arr,
            minlength=n_pairs,
        )

    if how == 'mean':
        acc /= time_acc[:, None]

    # Binary handling
    if binary_idx.size > 0:
        v_sel = values[sample_idx][:, binary_idx]
        dt_on = (v_sel > 0.5).astype(float) * dt_arr[:, None]

        on_time = np.zeros((n_pairs, binary_idx.size), dtype=float)
        for i in range(binary_idx.size):
            on_time[:, i] = np.bincount(
                pair_ids,
                weights=dt_on[:, i],
                minlength=n_pairs,
            )

        acc[:, binary_idx] = (on_time > 0).astype(float)

    # Label handling (max dt representative)
    if label_idx.size > 0:
        order = np.lexsort((-dt_arr, bin_idx, new_seg_id))
        key_sorted = np.stack([new_seg_id[order], bin_idx[order]], axis=1)
        _, first = np.unique(key_sorted, axis=0, return_index=True)
        chosen = order[first]

        rep_keys = np.stack([new_seg_id[chosen], bin_idx[chosen]], axis=1)
        _, rep_idx = np.unique(rep_keys, axis=0, return_inverse=True)

        rep_vals = values[sample_idx[chosen]][:, label_idx]
        acc[rep_idx[:, None], label_idx] = rep_vals

    out_seg = pair_unique[:, 0]
    out_bin = pair_unique[:, 1]

    out = pd.DataFrame(acc, columns=value_cols)
    out.insert(0, 'new_bin', out_bin.astype(int))
    out.insert(0, 'new_segment', out_seg.astype(int))

    if add_support_duration:
        out['bin_support_dt'] = time_acc

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
            + ([segment_col] if segment_col in new_seg_info.columns else [])
        ],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    for c in label_cols:
        if c in out.columns:
            out[c] = _make_nullable_int_label_series(
                out[c],
                binary=(c in binary_label_cols),
            )

    if add_bin_edges:
        bin_edges = bins_2d[out_bin, :]
        return out, bin_edges

    return out   
    
