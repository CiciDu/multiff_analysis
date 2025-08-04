import sys
from data_wrangling import process_monkey_information, specific_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling
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


def add_angle_from_cur_arc_end_to_nxt_ff(df, cur_curv_df, nxt_curv_df):
    angle_df = get_angle_from_cur_arc_end_to_nxt_ff(
        cur_curv_df, nxt_curv_df)
    df = df.merge(angle_df[['point_index', 'cur_opt_arc_end_heading', 'cur_cntr_arc_end_heading', 'angle_opt_arc_from_cur_end_to_nxt',
                            'angle_cntr_arc_from_cur_end_to_nxt']], on='point_index', how='left')
    return df


def get_angle_from_cur_arc_end_to_nxt_ff(cur_curv_df, nxt_curv_df):
    # add 'cur_' to all columns in cur_curv_df except 'point_index'
    cur_curv_df.columns = ['cur_' + col if col !=
                           'point_index' else col for col in cur_curv_df.columns]
    # add 'nxt_' to all columns in nxt_curv_df except 'point_index'
    nxt_curv_df.columns = ['nxt_' + col if col !=
                           'point_index' else col for col in nxt_curv_df.columns]

    both_curv_df = cur_curv_df.merge(nxt_curv_df, on='point_index', how='left')

    both_curv_df['cur_opt_arc_end_heading'] = both_curv_df['cur_monkey_angle'] + \
        both_curv_df['cur_opt_arc_d_heading']
    both_curv_df['cur_cntr_arc_end_heading'] = both_curv_df['cur_monkey_angle'] + \
        both_curv_df['cur_cntr_arc_d_heading']

    both_curv_df['angle_opt_arc_from_cur_end_to_nxt'] = specific_utils.calculate_angles_to_ff_centers(
        both_curv_df['nxt_ff_x'], both_curv_df['nxt_ff_y'], both_curv_df['cur_opt_arc_end_x'], both_curv_df['cur_opt_arc_end_y'], both_curv_df['cur_opt_arc_end_heading'])
    both_curv_df['angle_cntr_arc_from_cur_end_to_nxt'] = specific_utils.calculate_angles_to_ff_centers(
        both_curv_df['nxt_ff_x'], both_curv_df['nxt_ff_y'], both_curv_df['cur_cntr_arc_end_x'], both_curv_df['cur_cntr_arc_end_y'], both_curv_df['cur_cntr_arc_end_heading'])

    return both_curv_df


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


def rebin_segment_data(df, new_seg_info, bin_width=0.2):
    # This function rebins the data by segment and time bin, and takes the median of the data within each bin
    # It makes sure that the bins are perfectly aligned within segments (whereas in the previous method, bins are continuously assigned to all time points)
    # df must contain columns: segment, seg_start_time, seg_end_time, time, bin,

    concat_seg_data = concat_new_seg_info(
        df, new_seg_info, bin_width=bin_width)

    # Take the median of the data within each bin
    rebinned_data = concat_seg_data.groupby(
        ['new_segment', 'new_bin']).median().reset_index(drop=False)

    return rebinned_data


def rebin_spike_data(spikes_df, new_seg_info, bin_width=0.2):
    """
    Re-bin spike data based on new segment definitions and specified bin width.

    Args:
        spikes_df (pd.DataFrame): Original spike data with 'time' and 'cluster' columns.
        new_seg_info (pd.DataFrame): DataFrame with new segment start/end times and metadata.
        bin_width (float): Width of each time bin (in seconds).

    Returns:
        pd.DataFrame: Wide-format binned spike matrix indexed by (new_segment, new_bin),
                      with spike counts per unit as columns.
    """
    # Assign each spike to a segment and compute new time bins
    concat_seg_data = concat_new_seg_info(
        spikes_df, new_seg_info, bin_width=bin_width)

    # Convert binned spike data into wide-format matrix
    rebinned_spike_data = _rebin_spike_data(concat_seg_data, new_segments=new_seg_info['new_segment'].unique())

    return rebinned_spike_data


def _rebin_spike_data(concat_seg_data, new_segments=None):
    """
    Create a wide-format binned spike matrix with shape [segments × bins, clusters].

    Returns:
        rebinned_spike_data: DataFrame indexed by (new_segment, new_bin), with spike counts per unit.
    """
    # Group and count spikes by segment, bin, and cluster
    rebinned_spike_data = (
        concat_seg_data
        .groupby(['new_segment', 'new_bin', 'cluster'])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all (segment, bin) combinations and all clusters are included
    if new_segments is None:
        new_segments = concat_seg_data['new_segment'].unique()
    bins = np.arange(concat_seg_data['new_bin'].max() + 1)
    clusters = np.sort(concat_seg_data['cluster'].unique())
    full_index = pd.MultiIndex.from_product(
        [new_segments, bins], names=['new_segment', 'new_bin'])

    rebinned_spike_data = rebinned_spike_data.reindex(
        index=full_index, columns=clusters, fill_value=0)

    # Clean column names
    rebinned_spike_data.columns.name = None
    rebinned_spike_data.columns = [
        f'cluster_{i}' for i in clusters]
    rebinned_spike_data.reset_index(drop=False, inplace=True)
    return rebinned_spike_data


def _get_new_seg_info(planning_data):
    new_seg_info = planning_data[[
        'segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']].drop_duplicates()
    new_seg_info['new_segment'] = pd.factorize(
        new_seg_info['segment'])[0]
    return new_seg_info


# def select_segment_data_around_event_time(planning_data, pre_event_window=0.25, post_event_window=0.75):
#     planning_data['new_seg_start_time'] = planning_data['event_time'] - pre_event_window
#     planning_data['new_seg_end_time'] = planning_data['event_time'] + post_event_window
#     planning_data['new_seg_duration'] = pre_event_window + post_event_window
#     planning_data = planning_data[planning_data['time'].between(planning_data['new_seg_start_time'], planning_data['new_seg_end_time'])]
#     return planning_data
