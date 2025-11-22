
from decision_making_analysis.ff_data_acquisition import ff_data_utils

from pattern_discovery import cluster_analysis
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import curvature_utils
from decision_making_analysis.data_enrichment import rsw_vs_rcap_utils
from decision_making_analysis.ff_data_acquisition import cluster_replacement_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import math


def find_current_and_alternative_ff_info(
    miss_events_df,
    ff_real_position_sorted,
    ff_life_sorted,
    ff_dataframe,
    monkey_information,
    columns_to_sort_alt_ff_by=('abs_curv_diff', 'time_since_last_vis'),
    max_cluster_distance=50,
    max_time_since_last_vis=3,
    max_distance_to_ref_point=400,
    add_ff_in_cluster=True,
):
    """
    Identify current and alternative fireflies and compute associated curvature features.

    Parameters
    ----------
    miss_events_df : pd.DataFrame
        DataFrame that contains the information of the missed events
    ff_real_position_sorted : pd.DataFrame
        Sorted firefly position data.
    ff_life_sorted : pd.DataFrame
        Sorted firefly lifespan information.
    ff_dataframe : pd.DataFrame
        Main firefly DataFrame containing event-level information.
    monkey_information : pd.DataFrame
        Per-trial or per-point metadata for the monkey (e.g., position, time).
    columns_to_sort_alt_ff_by : tuple of str, optional
        Columns used to sort alternative fireflies by priority.
    max_cluster_distance : float, optional
        Maximum spatial distance for clustering missed fireflies.
    max_time_since_last_vis : float, optional
        Maximum time threshold (in seconds) since the last firefly visibility.
    max_distance_to_ref_point : float, optional
        Maximum distance threshold (in cm) between the stop and firefly position.

    Returns
    -------
    tuple
        miss_event_cur_ff : pd.DataFrame
            Processed DataFrame for current fireflies (miss/retry candidates).
        miss_event_alt_ff : pd.DataFrame
            Processed DataFrame for alternative fireflies (switch candidates).
    """
    print(f'[INFO] max_cluster_distance = {max_cluster_distance} '
          f'(should match the one used to make miss_events_df)')

    # Identify current (missed) firefly candidates
    cur_ff_info = find_miss_event_cur_ff(
        miss_events_df,
        ff_real_position_sorted,
        ff_dataframe,
        monkey_information,
        ff_life_sorted,
        max_time_since_last_vis=max_time_since_last_vis,
        max_cluster_distance=max_cluster_distance,
        max_distance_to_ref_point=max_distance_to_ref_point,
        add_ff_in_cluster=add_ff_in_cluster,
    )

    # Identify alternative (switch) firefly candidates
    nxt_ff_info = find_miss_event_alt_ff(
        cur_ff_info,
        ff_dataframe,
        monkey_information,
        ff_real_position_sorted,
        max_time_since_last_vis=max_time_since_last_vis,
        max_distance_to_ref_point=max_distance_to_ref_point,
    )

    # Retain relevant info and polish both DataFrames
    cur_ff_info, nxt_ff_info = retain_useful_cur_and_nxt_info(
        cur_ff_info, nxt_ff_info
    )

    miss_event_cur_ff = polish_miss_event_cur_ff(cur_ff_info)
    miss_event_alt_ff = polish_miss_event_alt_ff(
        nxt_ff_info,
        cur_ff_info,
        ff_real_position_sorted,
        ff_life_sorted,
        ff_dataframe,
        monkey_information,
        max_cluster_distance=max_cluster_distance,
        columns_to_sort_alt_ff_by=columns_to_sort_alt_ff_by,
        max_time_since_last_vis=max_time_since_last_vis,
    )

    # Ensure both DataFrames share the same point indices
    miss_event_cur_ff, miss_event_alt_ff = (
        make_sure_miss_event_alt_ff_and_miss_event_cur_ff_have_the_same_point_indices(
            miss_event_cur_ff, miss_event_alt_ff
        )
    )

    return miss_event_cur_ff, miss_event_alt_ff


def add_features_to_miss_event_ff_info(miss_event_ff_info,
                                       ff_dataframe,
                                       monkey_information,
                                       ff_real_position_sorted,
                                       ff_caught_T_new,
                                       curv_of_traj_df,
                                       curvature_df,
                                       last_seen_and_next_seen_attributes_to_add=[
                                           'ff_distance', 'ff_angle', 'ff_angle_boundary'],
                                       ff_priority_criterion='ff_distance',
                                       ):

    miss_event_ff_info['time'] = monkey_information.loc[miss_event_ff_info['point_index'], 'time'].values

    miss_event_ff_info = cluster_replacement_utils.supply_info_of_ff_last_seen_and_next_seen_to_df(
        miss_event_ff_info,
        ff_dataframe,
        monkey_information,
        ff_real_position_sorted,
        ff_caught_T_new,
        attributes_to_add=last_seen_and_next_seen_attributes_to_add,
        curv_of_traj_df=curv_of_traj_df,
    )

    miss_event_ff_info = add_arc_info_to_ff_info(
        miss_event_ff_info, curvature_df, monkey_information, ff_caught_T_new, curv_of_traj_df)

    # Note: in order not to feed the input with additional data, we will let curv_of_traj_df = curv_of_traj_df
    miss_event_ff_info = ff_data_utils.add_curv_diff_to_df(
        miss_event_ff_info, monkey_information, curv_of_traj_df, ff_real_position_sorted=ff_real_position_sorted)

    miss_event_ff_info.sort_values(
        by=['point_index', ff_priority_criterion], inplace=True)

    return miss_event_ff_info


def retain_rows_in_df1_that_share_or_not_share_columns_with_df2(df1, df2, columns, whether_share=True):
    temp_df = df2[columns].copy()
    temp_df['_share'] = True
    temp_df.drop_duplicates(inplace=True)
    df1 = pd.merge(df1, temp_df, on=columns, how='left')
    df1['_share'] = df1['_share'] == True
    if whether_share:
        df1 = df1[df1['_share'] == True].drop(['_share'], axis=1)
    else:
        df1 = df1[df1['_share'] != True].drop(['_share'], axis=1)
    return df1


def take_optimal_row_per_group_based_on_columns(df, columns, groupby_column='point_index'):
    '''
    This function takes the optimal row per group based on the columns specified.
    '''
    df = df.sort_values(columns, ascending=True)
    df = df.groupby(groupby_column).first().reset_index(drop=False)
    return df


def supply_info_of_cluster_to_df(df, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=50):

    ff_time = monkey_information.loc[df['point_index'].values, 'time'].values
    ff_cluster = cluster_analysis.find_alive_ff_clusters(ff_real_position_sorted[df['ff_index'].values], ff_real_position_sorted, ff_time-10, ff_time+10,
                                                         ff_life_sorted, max_distance=max_cluster_distance)
    ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        ff_cluster, df['point_index'].values)
    # new_df = ff_data_utils.find_many_ff_info_anew(ff_cluster_df['ff_index'].values, ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    return ff_cluster_df


def find_miss_event_cur_ff(miss_events_df, ff_real_position_sorted, ff_dataframe, monkey_information, ff_life_sorted,
                           max_time_since_last_vis=3,
                           max_cluster_distance=50,
                           max_distance_to_ref_point=400,
                           add_ff_in_cluster=True,
                           ):
    """
    Build a per-(point_index, ff_index) table of current-firefly candidates for
    miss/abort events, filtered by recent visibility and distance

    This function expands each miss/abort evaluation point by its nearby alive
    fireflies, queries geometry/visibility features for those pairs, and returns
    one row per unique (point_index, ff_index) that passes the filters.
    -----
    - Duplicates by (point_index, ff_index) are removed, keeping the first occurrence.
    - Visibility filtering always applies 'time_since_last_vis' <= max_time_since_last_vis.
    - Distances are computed between firefly position ('ff_real_position_sorted[ff_index]')
      and monkey position at 'first_stop_point_index'.
    - This function does not modify inputs in-place.
    """

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == True].copy()

    temp_miss_events_df = (
        miss_events_df[['ff_index', 'point_index', 'target_index',
                        'time', 'num_stops', 'first_stop_point_index', 'total_stop_time',
                        'eventual_outcome', 'event_type']]
        .assign(ff_index=lambda df: df['ff_index'].astype(int),
                point_index=lambda df: df['point_index'].astype(int),
                target_index=lambda df: df['target_index'].astype(int))
    )

    if add_ff_in_cluster:
        temp_miss_event_cur_ff = supply_info_of_cluster_to_df(
            temp_miss_events_df, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=max_cluster_distance)
    else:
        temp_miss_event_cur_ff = temp_miss_events_df.copy()

    miss_event_cur_ff = ff_data_utils.find_many_ff_info_anew(
        temp_miss_event_cur_ff['ff_index'].values, temp_miss_event_cur_ff['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    miss_event_cur_ff = miss_event_cur_ff.drop_duplicates(
        subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)
    miss_event_cur_ff = miss_event_cur_ff[miss_event_cur_ff['time_since_last_vis']
                                          <= max_time_since_last_vis]

    # now we need to add back some columns
    columns_to_add_back = temp_miss_events_df[[
        'point_index', 'num_stops', 'total_stop_time', 'eventual_outcome', 'event_type']].drop_duplicates()
    miss_event_cur_ff = pd.merge(miss_event_cur_ff, columns_to_add_back, on=[
        'point_index'], how='left')

    # get distance_to_ref_point
    miss_event_cur_ff = miss_event_cur_ff.merge(
        temp_miss_events_df[['point_index', 'first_stop_point_index']], on='point_index', how='left')
    ff_x, ff_y = ff_real_position_sorted[miss_event_cur_ff['ff_index'].values].T
    monkey_x, monkey_y = monkey_information.loc[miss_event_cur_ff['first_stop_point_index'].values, [
        'monkey_x', 'monkey_y']].values.T
    miss_event_cur_ff['distance_to_ref_point'] = np.sqrt(
        (ff_x - monkey_x)**2 + (ff_y - monkey_y)**2)
    miss_event_cur_ff = miss_event_cur_ff[miss_event_cur_ff['distance_to_ref_point'] < max_distance_to_ref_point].copy(
    )

    miss_event_cur_ff = miss_event_cur_ff.drop_duplicates(
        subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)

    return miss_event_cur_ff


def find_miss_event_alt_ff(miss_event_cur_ff, ff_dataframe,
                           monkey_information, ff_real_position_sorted,
                           max_time_since_last_vis=3,
                           max_distance_to_ref_point=400,
                           ):
    """
    Construct a table of “next-firefly” (alternative target) candidates for each
    miss/abort evaluation point, filtered by recent visibility and distance

    The function starts from `miss_event_cur_ff` (current-firefly candidates per
    evaluation point), then:
      1) Gathers all fireflies available at the same `point_index` from `ff_dataframe`,
         filters by `time_since_last_vis <= max_time_since_last_vis`, and keeps a
         compact set of geometry/angle features.
      2) Enforces a maximum distance to the stop and removes duplicate (point_index, ff_index),
         keeping the closest instance.
    """

    miss_event_alt_ff = ff_dataframe[ff_dataframe['point_index'].isin(
        miss_event_cur_ff['point_index'].values)].copy()
    miss_event_alt_ff = miss_event_alt_ff[miss_event_alt_ff['time_since_last_vis']
                                          <= max_time_since_last_vis]
    miss_event_alt_ff = miss_event_alt_ff[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary',
                                           'abs_ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index', 'abs_curv_diff']]
    miss_event_alt_ff['distance_to_ref_point'] = miss_event_alt_ff['ff_distance']
    miss_event_alt_ff = miss_event_alt_ff[miss_event_alt_ff['distance_to_ref_point']
                                          < max_distance_to_ref_point].copy()

    # use merge to add 'total_stop_time' to miss_event_alt_ff
    miss_event_alt_ff = miss_event_alt_ff.merge(
        miss_event_cur_ff[['point_index', 'total_stop_time', 'eventual_outcome', 'event_type']], on='point_index', how='left')

    miss_event_alt_ff.sort_values(
        by=['point_index', 'ff_index', 'distance_to_ref_point'], inplace=True)
    miss_event_alt_ff = miss_event_alt_ff.drop_duplicates(
        subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)

    return miss_event_alt_ff


def add_distance_to_ref_point(df, monkey_information, ff_real_position_sorted):
    stop_monkey_x, stop_monkey_y = monkey_information.loc[df['point_index'].values, [
        'monkey_x', 'monkey_y']].values.T
    ff_x, ff_y = ff_real_position_sorted[df['ff_index'].values].T
    df['distance_to_ref_point'] = np.sqrt(
        (stop_monkey_x - ff_x)**2 + (stop_monkey_y - ff_y)**2)


def retain_useful_cur_and_nxt_info(miss_event_cur_ff, miss_event_alt_ff, eliminate_cases_with_close_nxt_ff=True,
                                   min_nxt_ff_distance_to_stop=0):
    # we need to eliminate the info of the ff in miss_event_alt_ff that's also in miss_event_cur_ff at the same point indices
    miss_event_alt_ff = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        miss_event_alt_ff, miss_event_cur_ff, columns=['point_index', 'ff_index'], whether_share=False)

    # then, we eliminate the cases where miss_event_alt_ff has at least one ff that's within min_nxt_ff_distance_to_stop to the current point, so that the separation between the current and alternative ff is not too small
    # Note: right now we set min_nxt_ff_distance_to_stop to 0, so that we don't eliminate any cases
    if eliminate_cases_with_close_nxt_ff:
        miss_event_alt_ff = miss_event_alt_ff[miss_event_alt_ff['ff_distance']
                                              > min_nxt_ff_distance_to_stop].copy()

    # also eliminate nxt_ff if it's at the back of the monkey
    miss_event_alt_ff = miss_event_alt_ff[miss_event_alt_ff['ff_angle_boundary'].between(
        -90*math.pi/180, 90*math.pi/180)].copy()

    # then, we eliminate the info in miss_event_cur_ff that does not have corresponding info in miss_event_alt_ff (with the same point_index)
    miss_event_cur_ff = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        miss_event_cur_ff, miss_event_alt_ff, columns=['point_index'], whether_share=True)
    return miss_event_cur_ff, miss_event_alt_ff


def make_sure_miss_event_alt_ff_and_miss_event_cur_ff_have_the_same_point_indices(miss_event_cur_ff, miss_event_alt_ff):
    miss_event_cur_ff = miss_event_cur_ff[miss_event_cur_ff['point_index'].isin(
        miss_event_alt_ff['point_index'].values)].copy()
    miss_event_alt_ff = miss_event_alt_ff[miss_event_alt_ff['point_index'].isin(
        miss_event_cur_ff['point_index'].values)].copy()
    return miss_event_cur_ff, miss_event_alt_ff


def polish_miss_event_cur_ff(miss_event_cur_ff):
    miss_event_cur_ff = miss_event_cur_ff.sort_values(
        by=['point_index']).reset_index(drop=True)
    miss_event_cur_ff = miss_event_cur_ff[['num_stops', 'ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary', 'time_since_last_vis',
                                           'duration_of_last_vis_period', 'point_index', 'ff_index', 'distance_to_ref_point', 'total_stop_time', 'eventual_outcome', 'event_type']].copy()
    miss_event_cur_ff.sort_values(by=['point_index'], inplace=True)

    miss_event_cur_ff = _clip_time_since_last_vis(
        miss_event_cur_ff)

    miss_event_cur_ff = _add_num_ff_in_cluster(
        miss_event_cur_ff)

    return miss_event_cur_ff


def polish_miss_event_alt_ff(miss_event_alt_ff, miss_event_cur_ff, ff_real_position_sorted, ff_life_sorted, ff_dataframe, monkey_information,
                             columns_to_sort_alt_ff_by=['abs_curv_diff', 'time_since_last_vis'], max_cluster_distance=50,
                             max_time_since_last_vis=3,
                             take_one_row_for_each_point_and_find_cluster=False):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == True].copy()
    if take_one_row_for_each_point_and_find_cluster:
        # take the optimal ff from miss_event_alt_ff based on columns_to_sort_alt_ff_by
        miss_event_alt_ff = take_optimal_row_per_group_based_on_columns(
            miss_event_alt_ff, columns_to_sort_alt_ff_by, groupby_column='point_index')

        # now, let's re-find miss_event_alt_ff by considering clusters
        miss_event_alt_ff_old = miss_event_alt_ff.copy()
        miss_event_alt_ff = supply_info_of_cluster_to_df(
            miss_event_alt_ff, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=max_cluster_distance)
    else:
        miss_event_alt_ff_old = miss_event_alt_ff.copy()

    # find the info of additional columnsfor miss_event_alt_ff
    miss_event_alt_ff = ff_data_utils.find_many_ff_info_anew(
        miss_event_alt_ff['ff_index'].values, miss_event_alt_ff['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    miss_event_alt_ff = miss_event_alt_ff[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary',
                                           'time_since_last_vis', 'duration_of_last_vis_period', 'point_index', 'ff_index']].copy()
    miss_event_alt_ff = miss_event_alt_ff.merge(miss_event_alt_ff_old[[
        'point_index', 'ff_index', 'distance_to_ref_point', 'total_stop_time', 'eventual_outcome', 'event_type']], on=['point_index', 'ff_index'], how='left')

    miss_event_alt_ff = miss_event_alt_ff[(
        miss_event_alt_ff['time_since_last_vis'] <= max_time_since_last_vis)]

    # # since we just added cluster ff, once again we need to eliminate the info of the ff in miss_event_alt_ff that's also in miss_event_cur_ff at the same point indices
    miss_event_alt_ff = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        miss_event_alt_ff, miss_event_cur_ff, columns=['point_index', 'ff_index'], whether_share=False)

    miss_event_alt_ff.sort_values(by=['point_index'], inplace=True)

    miss_event_alt_ff = _clip_time_since_last_vis(
        miss_event_alt_ff)

    miss_event_alt_ff = _add_num_ff_in_cluster(
        miss_event_alt_ff)

    return miss_event_alt_ff


def _clip_time_since_last_vis(df, max_time_since_last_vis=5):
    df.loc[df['time_since_last_vis'] > max_time_since_last_vis,
           'time_since_last_vis'] = max_time_since_last_vis
    return df


def _add_num_ff_in_cluster(df):
    num_ff_in_cluster = df.groupby(
        'point_index').size().reset_index(drop=False)
    num_ff_in_cluster.columns = ['point_index', 'num_ff_in_cluster']
    df = df.merge(num_ff_in_cluster, on='point_index', how='left')
    return df


def find_curv_diff_for_ff_info(ff_info, monkey_information, ff_real_position_sorted, curv_of_traj_df=None):
    if curv_of_traj_df is None:
        raise ValueError(
            'curv_of_traj_df is None. Please provide curv_of_traj_df.')
    ff_info_temp = ff_info.groupby(
        ['point_index', 'ff_index']).first().reset_index(drop=False).copy()
    ff_info_temp['monkey_x'] = monkey_information.loc[ff_info_temp['point_index'], 'monkey_x'].values
    ff_info_temp['monkey_y'] = monkey_information.loc[ff_info_temp['point_index'], 'monkey_y'].values
    ff_info_temp['monkey_angle'] = monkey_information.loc[ff_info_temp['point_index'],
                                                          'monkey_angle'].values
    ff_info_temp['ff_x'] = ff_real_position_sorted[ff_info_temp['ff_index'].values, 0]
    ff_info_temp['ff_y'] = ff_real_position_sorted[ff_info_temp['ff_index'].values, 1]
    temp_curvature_df = curvature_utils.make_curvature_df(
        ff_info_temp, curv_of_traj_df, ff_radius_for_opt_arc=10)
    temp_curvature_df.loc[:, 'curv_diff'] = temp_curvature_df['opt_arc_curv'].values - \
        temp_curvature_df['curv_of_traj'].values
    # temp_curvature_df.loc[:,'abs_curv_diff'] = np.abs(temp_curvature_df.loc[:,'curv_diff'])
    if 'curv_diff' in ff_info.columns:
        ff_info.drop(['curv_diff'], axis=1, inplace=True)
    ff_info = ff_info.merge(temp_curvature_df[['ff_index', 'point_index', 'curv_diff']].drop_duplicates(), on=[
                            'ff_index', 'point_index'], how='left')
    ff_info['abs_curv_diff'] = np.abs(ff_info['curv_diff'].values)
    return ff_info, temp_curvature_df


def assign_new_point_index_to_combine_across_sessions(
    important_info,
    new_point_index_start,
    new_point_index_col='new_point_index'
):
    assert 'miss_events_df' in important_info, 'miss_events_df must be in important_info.keys()'

    # reference df for mapping
    ref_df = important_info['miss_events_df']
    if ref_df['point_index'].isna().any():
        raise ValueError('miss_events_df has NaN in point_index')

    unique_point_index = ref_df['point_index'].unique().astype(int)
    point_index_map = pd.DataFrame({
        'point_index': unique_point_index,
        'new_number': range(new_point_index_start, new_point_index_start + len(unique_point_index))
    }).set_index('point_index')

    for df_name, df in important_info.items():
        if df['point_index'].isna().any():
            raise ValueError(f'{df_name} has NaN in point_index')
        mapped = df['point_index'].map(point_index_map['new_number'])
        if mapped.isna().any():
            missing = df.loc[mapped.isna(), 'point_index'].unique()
            raise KeyError(f'{df_name}: unmapped point_index values {missing}')
        important_info[df_name][new_point_index_col] = mapped.values

    return important_info, point_index_map



def find_possible_objects_of_pursuit(all_relevant_indices, ff_dataframe, max_distance_from_ref_point_to_missed_target=50,
                                     max_allowed_time_since_last_vis=3):
    # find corresponding info in ff_dataframe at time (in-memory ff and visible ff)
    ff_info = ff_dataframe.loc[ff_dataframe['point_index'].isin(
        all_relevant_indices)].copy()
    ff_info = ff_info[ff_info['time_since_last_vis']
                      <= max_allowed_time_since_last_vis]

    # among them, find ff close to monkey's position (within max_distance_from_ref_point_to_missed_target to the center of the ff), all of them can be possible targets
    ff_info = ff_info[ff_info['ff_distance'] <
                      max_distance_from_ref_point_to_missed_target].copy()

    return ff_info


def add_arc_info_to_ff_info(df, curvature_df, monkey_information, ff_caught_T_new, curv_of_traj_df):
    """Merge curvature (arc) information into current and next firefly DataFrames."""

    arc_cols = [
        'curv_of_traj', 'curvature_lower_bound', 'curvature_upper_bound',
        'opt_arc_curv', 'curv_diff', 'abs_curv_diff',
    ]
    sub_df = curvature_df[['ff_index', 'point_index'] + arc_cols].copy()

    for col in arc_cols:
        df.drop(
            columns=col, errors='ignore', inplace=True)

    df = pd.merge(
        df, sub_df, on=['ff_index', 'point_index'], how='left'
    )
    df = curvature_utils.fill_up_NAs_in_columns_related_to_curvature(
        df, monkey_information, ff_caught_T_new,
        curv_of_traj_df=curv_of_traj_df,
    )
    return df


def set_point_of_eval(miss_events_df, monkey_information, time_with_respect_to_first_stop=None, time_with_respect_to_second_stop=None, time_with_respect_to_last_stop=None):

    miss_events_df = miss_events_df.copy()

    # Exactly one of the three offsets must be provided
    options = [
        ('first_stop_time', time_with_respect_to_first_stop),
        ('second_stop_time', time_with_respect_to_second_stop),
        ('last_stop_time', time_with_respect_to_last_stop),
    ]
    provided = [(col, offset) for col, offset in options if offset is not None]
    if len(provided) != 1:
        raise ValueError(
            'Provide exactly one of time_with_respect_to_first_stop, time_with_respect_to_second_stop, or time_with_respect_to_last_stop.')

    base_col, offset = provided[0]

    # Compute time_of_eval; preserve NaNs if base times are missing
    base_times = miss_events_df[base_col].to_numpy(copy=False)
    time_of_eval = base_times + offset
    miss_events_df['time_of_eval'] = time_of_eval

    # Ensure monkey_information is sorted by time (without mutating the original)
    mi_time = monkey_information['time'].to_numpy(copy=False)
    mi_point_index = monkey_information['point_index'].to_numpy(copy=False)
    if mi_time.ndim != 1 or mi_point_index.ndim != 1:
        raise ValueError(
            'monkey_information columns "time" and "point_index" must be 1D.')

    # Sort if needed
    if not np.all(mi_time[1:] >= mi_time[:-1]):
        order = np.argsort(mi_time)
        mi_time = mi_time[order]
        mi_point_index = mi_point_index[order]

    # Vectorized search: for each eval time, find the most recent point <= time
    idx = np.searchsorted(mi_time, time_of_eval, side='right') - 1
    # Clamp to valid range; if all eval times are before the first sample, idx becomes -1 → clamp to 0
    if mi_time.size == 0:
        raise ValueError('monkey_information is empty.')
    idx = np.clip(idx, 0, mi_time.size - 1)

    # For rows where time_of_eval is NaN, set point_index_of_eval to NaN as well
    nan_mask = pd.isna(time_of_eval)
    point_index_of_eval = mi_point_index[idx].astype(
        float)  # cast to float to allow NaN
    point_index_of_eval[nan_mask] = np.nan

    miss_events_df['point_index_of_eval'] = point_index_of_eval
    if 'point_index' in miss_events_df.columns:
        print('Warning: "point_index" column already exists in miss_events_df, but it will be overwritten by "point_index_of_eval".')
    if 'time' in miss_events_df.columns:
        print('Warning: "time" column already exists in miss_events_df, but it will be overwritten by "time_of_eval".')

    miss_events_df['point_index'] = miss_events_df['point_index_of_eval']
    miss_events_df['time'] = miss_events_df['time_of_eval']
    return miss_events_df


def fill_in_eventual_outcome_and_event_type(df):
    for col in ['eventual_outcome', 'event_type']:
        # check that each point_index has at most one unique non-null value
        check = df.groupby('point_index')[col].nunique(dropna=True)
        if (check > 1).any():
            bad_idx = check[check > 1].index.tolist()
            raise ValueError(
                f'Column "{col}" has multiple non-null values for point_index: {bad_idx}'
            )

        # build mapping from point_index → non-null value
        mapping = df.groupby('point_index')[col].first()

        # fill missing values using that mapping
        df[col] = df[col].fillna(df['point_index'].map(mapping))

    return df
