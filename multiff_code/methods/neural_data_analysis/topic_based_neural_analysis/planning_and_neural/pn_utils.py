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
    # first make sure that the intended columns are not already in df
    if 'num_ff_visible' in df.columns or 'num_ff_in_memory' in df.columns or 'log1p_num_ff_visible' in df.columns or 'log1p_num_ff_in_memory' in df.columns:
        print('num_ff_visible or num_ff_in_memory or log1p_num_ff_visible or log1p_num_ff_in_memory already in df. Dropping them and adding new ones.')
        df = df.drop(columns=['num_ff_visible', 'num_ff_in_memory', 'log1p_num_ff_visible', 'log1p_num_ff_in_memory'], errors='ignore')

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










