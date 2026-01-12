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


def _rebin_timeseries_core(
    *,
    times,
    values,
    bins_2d,
    how,
):
    """
    Core overlap->aggregation for a left-hold signal.
    Relies on external functions:
      - build_bin_assignments(times, bins_2d, ...)
      - bin_timeseries_weighted(values, dt_arr, bin_idx_arr, how=...)
    """
    if times.size < 2 or bins_2d.size == 0:
        return None, None, None

    sample_idx, bin_idx, dt_arr, _ = build_bin_assignments(
        times,
        bins_2d,
        assume_sorted=True,
        check_nonoverlap=False,
    )

    if sample_idx.size == 0:
        return None, None, None

    weighted_vals, _, used_bins = bin_timeseries_weighted(
        values[sample_idx],
        dt_arr,
        bin_idx,
        how=how,
    )

    if weighted_vals.ndim == 1:
        weighted_vals = weighted_vals[None, :]

    # used_bins are indices into bins_2d passed in
    return weighted_vals, used_bins.astype(int), dt_arr


def rebin_all_segments_global_bins(
    df,
    new_seg_info,
    bins_2d=None,
    bin_width=None,
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
        return pd.DataFrame()

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
        return pd.DataFrame()

    # Assign each interval to a new_segment id
    sample_segment, _, _, _ = _assign_intervals_to_new_segments(times, new_seg_info)
    if sample_segment.size == 0:
        return pd.DataFrame()

    valid_samples = sample_segment >= 0
    if not np.any(valid_samples):
        return pd.DataFrame()

    # Global overlaps ONCE
    sample_idx, bin_idx, dt_arr, _ = build_bin_assignments(
        times,
        bins_2d,
        assume_sorted=True,
        check_nonoverlap=False,
    )
    if sample_idx.size == 0:
        return pd.DataFrame()

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
            return pd.DataFrame()

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
            return pd.DataFrame()

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
        return pd.DataFrame()

    if how == 'mean':
        acc[valid] /= time_acc[valid, None]

    out_seg, out_bin = np.nonzero(valid)

    out = pd.DataFrame(acc[out_seg, out_bin], columns=value_cols)
    out.insert(0, 'new_bin', out_bin.astype(int))
    out.insert(0, 'new_segment', out_seg.astype(int))

    if add_support_duration:
        out['bin_support_dt'] = time_acc[out_seg, out_bin]

    if add_bin_edges:
        out['bin_left'] = bins_2d[out_bin, 0]
        out['bin_right'] = bins_2d[out_bin, 1]

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
            + ([segment_col] if segment_col in new_seg_info.columns else [])
        ],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    return out


def rebin_all_segments_local_bins(
    df,
    new_seg_info,
    bins_2d=None,          # unused, kept for interface symmetry
    bin_width=None,
    *,
    time_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    require_full_bin=False,      # no-op by design (kept for symmetry)
    add_support_duration=False,
):
    """
    Provably identical refactor of the original rebin_segment_data
    (local / segment-defined bins).
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
        return pd.DataFrame()

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
            block['bin_left'] = seg_bins_2d[used_bins.astype(int), 0]
            block['bin_right'] = seg_bins_2d[used_bins.astype(int), 1]

        out_blocks.append(block)

    if not out_blocks:
        return pd.DataFrame()

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

    return out


def rebin_all_segments(
    df,
    new_seg_info,
    *,
    mode,
    bins_2d=None,
    bin_width=None,
    time_col='time',
    segment_col='segment',
    how='mean',
    respect_old_segment=True,
    add_bin_edges=False,
    require_full_bin=False,
    add_support_duration=False,
):
    """
    Unified public wrapper.

    mode:
      - 'global': uses bins_2d (vectorized across segments)
      - 'local':  uses bin_width (segment-local bins)
    """
    if mode == 'global':
        return rebin_all_segments_global_bins(
            df,
            new_seg_info,
            bins_2d=bins_2d,
            bin_width=bin_width,
            time_col=time_col,
            segment_col=segment_col,
            how=how,
            respect_old_segment=respect_old_segment,
            add_bin_edges=add_bin_edges,
            require_full_bin=require_full_bin,
            add_support_duration=add_support_duration,
        )

    if mode == 'local':
        return rebin_all_segments_local_bins(
            df,
            new_seg_info,
            bins_2d=bins_2d,
            bin_width=bin_width,
            time_col=time_col,
            segment_col=segment_col,
            how=how,
            respect_old_segment=respect_old_segment,
            add_bin_edges=add_bin_edges,
            require_full_bin=require_full_bin,
            add_support_duration=add_support_duration,
        )

    raise ValueError("mode must be 'global' or 'local'")



def rebin_spike_data(
    spikes_df,
    new_seg_info,
    bin_width,
    *,
    time_col='time',
    cluster_col='cluster',
):
    """
    Segment-based PSTH-style rebinning for spikes.
    Dense output with zero-filled bins.
    Indexed by (new_segment, new_bin).
    """
    bins_2d, meta = segment_windows_to_bins2d(
        new_seg_info,
        bin_width=bin_width
    )

    if bins_2d.size == 0:
        return pd.DataFrame()

    counts, cluster_ids = bin_spikes_by_cluster(
        spikes_df[[time_col, cluster_col]],
        bins_2d,
        time_col=time_col,
        cluster_col=cluster_col,
        assume_sorted_bins=True,
        check_nonoverlap=False
    )

    out = meta[['new_segment', 'new_bin']].copy()
    for j, cid in enumerate(cluster_ids):
        out[f'cluster_{cid}'] = counts[:, j]

    out.set_index(['new_segment', 'new_bin'], inplace=True)
    out.sort_index(inplace=True)
    out.reset_index(inplace=True, drop=False)

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time',
                'new_seg_end_time', 'new_seg_duration']
        ],
        on='new_segment',
        how='left',
        validate='many_to_one'
    )

    return out


def add_curv_info(info_to_add, curv_df, which_ff_info):
    curv_df = curv_df.copy()
    columns_to_rename = {'ff_index': f'{which_ff_info}ff_index',
                         'cntr_arc_curv': f'{which_ff_info}cntr_arc_curv',
                         'opt_arc_curv': f'{which_ff_info}opt_arc_curv',
                         'opt_arc_d_heading': f'{which_ff_info}opt_arc_dheading', }

    curv_df.rename(columns=columns_to_rename, inplace=True)

    columns_added = list(columns_to_rename.values())
    # delete f'{which_ff_info}ff_index' from columns_added
    columns_added.remove(f'{which_ff_info}ff_index')

    curv_df_sub = curv_df[columns_added +
                          [f'{which_ff_info}ff_index', 'point_index']].drop_duplicates()

    info_to_add.drop(columns=columns_added, inplace=True, errors='ignore')
    info_to_add = info_to_add.merge(
        curv_df_sub, on=['point_index', f'{which_ff_info}ff_index'], how='left')

    return info_to_add, columns_added


def add_to_both_ff_when_seen_df(both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df):
    curv_df = curv_df.set_index('stop_point_index')
    both_ff_when_seen_df[f'{which_ff_info}ff_angle_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_angle']
    both_ff_when_seen_df[f'{which_ff_info}ff_distance_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_distance']
    # both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['cntr_arc_curv']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_curv']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_d_heading']
    both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[
        f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_of_traj']


def get_angle_from_cur_arc_end_to_nxt_ff(both_ff_df):
    both_ff_df['angle_opt_cur_end_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
        both_ff_df['nxt_ff_x'], both_ff_df['nxt_ff_y'], both_ff_df['cur_opt_arc_end_x'], both_ff_df['cur_opt_arc_end_y'], both_ff_df['cur_opt_arc_end_heading'])
    both_ff_df['angle_cntr_cur_end_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
        both_ff_df['nxt_ff_x'], both_ff_df['nxt_ff_y'], both_ff_df['cur_cntr_arc_end_x'], both_ff_df['cur_cntr_arc_end_y'], both_ff_df['cur_cntr_arc_end_heading'])

    return both_ff_df


def find_diff_in_curv_info(both_ff_df, point_indexes_before_stop, monkey_information, ff_real_position_sorted, ff_caught_T_new,
                           curv_traj_window_before_stop=[-25, 0], use_curv_to_ff_center=False, ff_radius_for_opt_arc=10):

    cur_end_to_next_ff_curv = compute_cur_end_to_next_ff_curv_for_pn(
        both_ff_df, use_curv_to_ff_center=use_curv_to_ff_center, ff_radius_for_opt_arc=ff_radius_for_opt_arc)
    prev_stop_to_next_ff_curv, _ = diff_in_curv_utils.compute_prev_stop_to_next_ff_curv(both_ff_df['nxt_ff_index'].values, point_indexes_before_stop,
                                                                                        monkey_information, ff_real_position_sorted, ff_caught_T_new,
                                                                                        curv_traj_window_before_stop=curv_traj_window_before_stop)
    prev_stop_to_next_ff_curv['ref_point_index'] = cur_end_to_next_ff_curv['point_index'].values

    diff_in_curv_df = diff_in_curv_utils.make_diff_in_curv_df(
        prev_stop_to_next_ff_curv, cur_end_to_next_ff_curv)
    return diff_in_curv_df


def compute_cur_end_to_next_ff_curv_for_pn(both_ff_df, use_curv_to_ff_center=False, ff_radius_for_opt_arc=10):
    mock_monkey_info = diff_in_curv_utils._build_mock_monkey_info(
        both_ff_df, use_curv_to_ff_center=use_curv_to_ff_center)
    null_arc_curv_df = diff_in_curv_utils._make_null_arc_curv_df(
        mock_monkey_info, ff_radius_for_opt_arc=ff_radius_for_opt_arc)
    cur_end_to_next_ff_curv = diff_in_curv_utils._compute_curv_from_cur_end(
        null_arc_curv_df, mock_monkey_info)
    cur_end_to_next_ff_curv['ref_point_index'] = cur_end_to_next_ff_curv['point_index']
    return cur_end_to_next_ff_curv


def _merge_both_ff_df(cur_curv_df, nxt_ff_info):
    # add 'cur_' to all columns in cur_curv_df except 'point_index'
    cur_curv_df = cur_curv_df.copy()
    nxt_ff_info = nxt_ff_info.copy()
    cur_curv_df.columns = ['cur_' + col if col !=
                           'point_index' else col for col in cur_curv_df.columns]
    # add 'nxt_' to all columns in nxt_curv_df except 'point_index'
    nxt_ff_info.columns = ['nxt_' + col if col !=
                           'point_index' else col for col in nxt_ff_info.columns]

    both_ff_df = cur_curv_df.merge(nxt_ff_info, on='point_index', how='left')

    both_ff_df['cur_opt_arc_end_heading'] = both_ff_df['cur_monkey_angle'] + \
        both_ff_df['cur_opt_arc_d_heading']
    both_ff_df['cur_cntr_arc_end_heading'] = both_ff_df['cur_monkey_angle'] + \
        both_ff_df['cur_cntr_arc_d_heading']
    return both_ff_df


def add_diff_in_curv_info(df, both_ff_df, monkey_information, ff_real_position_sorted, ff_caught_T_new):
    # get point_index_before_stop from heading_info_df
    # check for NA in point_index_before_stop
    if both_ff_df['point_index_before_stop'].isna().any():
        raise ValueError(
            'There are NA in point_index_before_stop in both_ff_df. Please check the heading_info_df.')

    diff_in_curv_info = find_diff_in_curv_info(
        both_ff_df, both_ff_df['point_index_before_stop'].values, monkey_information, ff_real_position_sorted, ff_caught_T_new)
    diff_in_curv_info.rename(
        columns={'ref_point_index': 'point_index'}, inplace=True)

    columns_to_merge = ['traj_curv_to_stop', 'curv_from_stop_to_nxt_ff',
                        'opt_curv_to_cur_ff', 'curv_from_cur_end_to_nxt_ff',
                        'd_curv_null_arc', 'd_curv_monkey',
                        'abs_d_curv_null_arc', 'abs_d_curv_monkey',
                        'diff_in_d_curv', 'diff_in_abs_d_curv']

    df.drop(columns=columns_to_merge, errors='ignore', inplace=True)
    df = df.merge(diff_in_curv_info[['point_index'] +
                  columns_to_merge], on='point_index', how='left')
    return df


def compute_overlap_and_drop(df1, col1, df2, col2):
    """
    Computes percentage of overlapped values in df1[col1] and df2[col2],
    prints the percentages, and returns new DataFrames with the overlapping
    rows dropped.

    Args:
        df1 (pd.DataFrame): First DataFrame
        col1 (str): Column name in df1 to check for overlap
        df2 (pd.DataFrame): Second DataFrame
        col2 (str): Column name in df2 to check for overlap

    Returns:
        df1_filtered (pd.DataFrame): df1 with overlapping rows dropped
        df2_filtered (pd.DataFrame): df2 with overlapping rows dropped
    """
    a = df1[col1].values
    b = df2[col2].values

    overlap = np.intersect1d(a, b)

    percentage_a = len(overlap) / len(a) * 100 if len(a) > 0 else 0
    percentage_b = len(overlap) / len(b) * 100 if len(b) > 0 else 0
    percentage_avg = len(overlap) / ((len(a) + len(b)) / 2) * \
        100 if (len(a) + len(b)) > 0 else 0

    if len(overlap) == 0:
        return df1, df2

    print(f"Overlap: {overlap}")
    print(f"Percentage overlap relative to df1: {percentage_a:.2f}%")
    print(f"Percentage overlap relative to df2: {percentage_b:.2f}%")
    print(f"Average percentage overlap: {percentage_avg:.2f}%")

    df1_filtered = df1[~df1[col1].isin(overlap)].copy()
    df2_filtered = df2[~df2[col2].isin(overlap)].copy()

    return df1_filtered, df2_filtered


def randomly_assign_random_dummy_based_on_targets(y_var):
    # randomly select 50% of all_targets to assign random_dummy to be true
    all_targets = y_var['target_index'].unique()
    half_targets = np.random.choice(
        all_targets, size=int(len(all_targets)*0.5), replace=False)
    y_var['random_dummy'] = 0
    y_var.loc[y_var['target_index'].isin(half_targets), 'random_dummy'] = 1
    return y_var


# ============================================================
# Helper: build bins once for all segments
# ============================================================

def _build_segment_bins(new_seg_info, bin_width):
    bins_list = []
    meta_rows = []

    for _, r in new_seg_info.sort_values('new_segment').iterrows():
        seg_id = int(r['new_segment'])
        t0 = float(r['new_seg_start_time'])
        t1 = float(r['new_seg_end_time'])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        dt = float(bin_width)
        n_bins = int(np.floor((t1 - t0) / dt))
        if n_bins <= 0:
            continue

        lefts = t0 + dt * np.arange(n_bins)
        rights = lefts + dt

        bins_list.append(np.column_stack([lefts, rights]))
        meta_rows.append(pd.DataFrame({
            'new_segment': seg_id,
            'new_bin': np.arange(n_bins, dtype=int),
        }))

    if not bins_list:
        return np.zeros((0, 2)), pd.DataFrame(columns=['new_segment', 'new_bin'])

    bins_2d = np.vstack(bins_list)
    meta = pd.concat(meta_rows, ignore_index=True)
    return bins_2d, meta


# ============================================================
# 1️⃣ concat_new_seg_info (kept for compatibility; minimal)
# ============================================================


def concat_new_seg_info(df, new_seg_info, bin_width=None):
    df = df.sort_values(by='time')
    new_seg_info = new_seg_info.sort_values(by='new_segment')
    concat_seg_data = []

    for _, row in new_seg_info.iterrows():
        if 'segment' in df.columns:
            mask = (df['segment'] == row['segment']) & (
                df['time'] >= row['new_seg_start_time']) & (df['time'] < row['new_seg_end_time'])
        else:
            mask = (df['time'] >= row['new_seg_start_time']) & (
                df['time'] < row['new_seg_end_time'])
        seg_df = df.loc[mask].copy()

        # Assign new bins relative to segment start
        if bin_width is not None:
            seg_df['new_bin'] = (
                (seg_df['time'] - row['new_seg_start_time']) // bin_width).astype(int)

        for col in ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']:
            seg_df[col] = row[col]

        concat_seg_data.append(seg_df)

    result = pd.concat(concat_seg_data, ignore_index=True)
    result.sort_values(by=['new_segment', 'time'], inplace=True)
    result['new_segment'] = result['new_segment'].astype(int)
    return result


def segment_windows_to_bins2d(
    new_seg_info,
    *,
    seg_id_col='new_segment',
    t0_col='new_seg_start_time',
    t1_col='new_seg_end_time',
    bin_width
):
    """
    Segment-based bin constructor (PSTH-style, but NOT event-based).

    Returns
    -------
    bins_2d : (N_bins, 2) array
        [t_left, t_right] for each bin across all segments
    meta : DataFrame
        Columns: new_segment, new_bin, t_left, t_right, bin
    """
    bins_list = []
    meta_rows = []

    for _, r in new_seg_info.sort_values(seg_id_col).iterrows():
        seg_id = int(r[seg_id_col])
        t0 = float(r[t0_col])
        t1 = float(r[t1_col])

        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        dt = float(bin_width)
        n_bins = int(np.floor((t1 - t0) / dt))
        if n_bins <= 0:
            continue

        lefts = t0 + dt * np.arange(n_bins)
        rights = lefts + dt

        bins_list.append(np.column_stack([lefts, rights]))

        meta_rows.append(pd.DataFrame({
            'new_segment': seg_id,
            'new_bin': np.arange(n_bins, dtype=int),
            't_left': lefts,
            't_right': rights,
        }))

    if not bins_list:
        return np.zeros((0, 2)), pd.DataFrame(
            columns=['new_segment', 'new_bin', 't_left', 't_right', 'bin']
        )

    bins_2d = np.vstack(bins_list)
    meta = pd.concat(meta_rows, ignore_index=True)

    # global bin index (PSTH-style)
    meta['bin'] = np.arange(len(meta), dtype=int)

    return bins_2d, meta


def _get_new_seg_info(planning_data):
    new_seg_info = planning_data[[
        'segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']].drop_duplicates()
    new_seg_info['new_segment'] = pd.factorize(
        new_seg_info['segment'])[0]
    return new_seg_info


# def select_segment_data_around_event_time(planning_data, start_t_rel_event=-0.25, end_t_rel_event=1.25):
#     planning_data['new_seg_start_time'] = planning_data['event_time'] + start_t_rel_event
#     planning_data['new_seg_end_time'] = planning_data['event_time'] + end_t_rel_event
#     planning_data['new_seg_duration'] = start_t_rel_event + end_t_rel_event
#     planning_data = planning_data[planning_data['time'].between(planning_data['new_seg_start_time'], planning_data['new_seg_end_time'])]
#     return planning_data

def calculate_angle_from_stop_to_nxt_ff(monkey_information, point_index_before_stop, nxt_ff_x, nxt_ff_y):
    mx_before_stop, my_before_stop, m_angle_before_stop = monkey_information.loc[point_index_before_stop, [
        'monkey_x', 'monkey_y', 'monkey_angle']].values.T
    angle_from_stop_to_nxt_ff = specific_utils.calculate_angles_to_ff_centers(
        nxt_ff_x, nxt_ff_y, mx_before_stop, my_before_stop, m_angle_before_stop)
    return m_angle_before_stop, angle_from_stop_to_nxt_ff


def add_ff_visible_or_in_memory_info_by_point(df, ff_dataframe, max_in_memory_time_since_seen=2):
    """
    For each point_index, add:
      - log1p_num_ff_visible (0/1),  num_ff_visible (uint8): unique visible FFs at that point
      - log1p_num_ff_in_memory (0/1), num_ff_in_memory (uint8): unique in-memory FFs at that point

    Expects in ff_dataframe: ['ff_index', 'point_index', 'visible', 'time_since_last_vis'].
    Merges onto df['point_index'].
    """

    required = {'ff_index', 'point_index', 'visible', 'time_since_last_vis'}
    missing = required - set(ff_dataframe.columns)
    if missing:
        raise KeyError(f"ff_dataframe missing columns: {sorted(missing)}")

    # Visible: filter by visible==True, then count unique ff_index per point_index
    visible_pairs = (
        ff_dataframe.loc[ff_dataframe['visible'].astype(
            bool), ['ff_index', 'point_index']]
        .drop_duplicates()
    )
    vis_counts = (
        visible_pairs.groupby('point_index')['ff_index']
        .nunique()
        .reset_index(name='num_ff_visible')
    )
    vis_counts['log1p_num_ff_visible'] = np.log1p(vis_counts['num_ff_visible'])

    # In-memory: time_since_last_vis < threshold, then count unique ff_index per point_index
    mem_pairs = (
        ff_dataframe.loc[ff_dataframe['time_since_last_vis'] < max_in_memory_time_since_seen,
                         ['ff_index', 'point_index']]
        .drop_duplicates()
    )
    mem_counts = (
        mem_pairs.groupby('point_index')['ff_index']
        .nunique()
        .reset_index(name='num_ff_in_memory')
    )
    mem_counts['log1p_num_ff_in_memory'] = np.log1p(mem_counts['num_ff_in_memory'])

    # Merge onto df
    out = (
        df.merge(vis_counts, on='point_index', how='left')
        .merge(mem_counts, on='point_index', how='left')
    )

    # Fill + compact dtypes
    for col in ['log1p_num_ff_visible', 'log1p_num_ff_in_memory', 'num_ff_visible', 'num_ff_in_memory']:
        if col not in out:
            out[col] = 0
        out[col] = out[col].fillna(0).astype('uint8')

    return out


def add_ff_visible_dummy(df, ff_index_col, ff_dataframe):
    # Keep only rows where the FF is visible

    right = (
        ff_dataframe.loc[ff_dataframe['visible'].astype(
            bool), ['ff_index', 'point_index']]
        .rename(columns={'ff_index': ff_index_col})   # align key name
        .drop_duplicates()                            # avoid merge blow-up
        .assign(whether_ff_visible_dummy=1)
    )

    out = df.merge(right, on=[ff_index_col, 'point_index'], how='left')
    out['whether_ff_visible_dummy'] = out['whether_ff_visible_dummy'].fillna(
        0).astype('uint8')
    return out


def add_ff_in_memory_dummy(df, ff_index_col, ff_dataframe, max_in_memory_time_since_seen=2):
    # Keep only rows where the FF is in memory
    ff_dataframe_in_memory = ff_dataframe[ff_dataframe['time_since_last_vis']
                                          < max_in_memory_time_since_seen].copy()

    ff_dataframe_in_memory = ff_dataframe_in_memory[['ff_index', 'point_index']].rename(
        # align key name
        columns={'ff_index': ff_index_col}).drop_duplicates()
    ff_dataframe_in_memory['whether_ff_in_memory_dummy'] = 1

    out = df.merge(ff_dataframe_in_memory, on=[
                   ff_index_col, 'point_index'], how='left')
    out['whether_ff_in_memory_dummy'] = out['whether_ff_in_memory_dummy'].fillna(
        0).astype('uint8')

    return out
