from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils
from data_wrangling import specific_utils
from planning_analysis.plan_factors import build_factor_comp_utils
import numpy as np
import pandas as pd
import math


def get_eye_toward_ff_time_perc_df(data_of_segments, ff_to_include=['cur_ff', 'nxt_ff'],
                                   list_of_max_degrees=[5, 10]):
    # example output columns: left_eye_to_cur_ff_time_perc_5, left_eye_to_cur_ff_time_perc_10

    df = data_of_segments.copy()

    eye_columns = []
    eye_to_gaze = {'left_eye': 'gaze_mky_view_angle_l',
                   'right_eye': 'gaze_mky_view_angle_r'
                   }

    for eye in ['left_eye', 'right_eye']:
        for ff in ff_to_include:
            for max_degrees in list_of_max_degrees:
                new_column = f'{eye}_{ff}_time_perc_{max_degrees}'
                df[new_column] = df['dt']
                df.loc[(df[eye_to_gaze[eye]] - df[f'{ff}_angle']).abs(
                ) > max_degrees/180 * math.pi, new_column] = 0
                df.loc[df[eye_to_gaze[eye]].isnull(), new_column] = 0
                eye_columns.append(new_column)

    eye_toward_ff_time_perc_df = df[eye_columns + ['dt', 'stop_point_index']
                                    ].groupby('stop_point_index').sum()

    for column in eye_columns:
        eye_toward_ff_time_perc_df[column] = eye_toward_ff_time_perc_df[column]/(
            eye_toward_ff_time_perc_df['dt'].values)

    eye_toward_ff_time_perc_df.reset_index(drop=False, inplace=True)
    return eye_toward_ff_time_perc_df


def add_column_curv_of_traj_before_stop(df, curv_of_traj_df_w_one_sided_window):
    curv_of_traj_df_w_one_sided_window = curv_of_traj_df_w_one_sided_window[[
        'point_index', 'curv_of_traj']].copy()
    # change unit
    curv_of_traj_df_w_one_sided_window['curv_of_traj'] = curv_of_traj_df_w_one_sided_window['curv_of_traj'] * 180/math.pi * 100
    curv_of_traj_df_w_one_sided_window = curv_of_traj_df_w_one_sided_window.rename(columns={
        'point_index': 'point_index_before_stop',
        'curv_of_traj': 'curv_of_traj_before_stop'})
    df = df.merge(curv_of_traj_df_w_one_sided_window,
                  on='point_index_before_stop', how='left')
    return df


def process_heading_info_df(heading_info_df):
    heading_info_df = heading_info_df.copy()
    # add some columns
    if 'angle_from_cur_ff_landing_to_nxt_ff' in heading_info_df.columns:
        heading_info_df[['angle_from_m_before_stop_to_nxt_ff', 'angle_from_cur_ff_landing_to_nxt_ff']] = heading_info_df[[
            'angle_from_m_before_stop_to_nxt_ff', 'angle_from_cur_ff_landing_to_nxt_ff']] * (180/np.pi)
        _add_diff_in_d_heading_to_cur_ff(heading_info_df)
        _add_diff_in_d_heading_to_nxt_ff(heading_info_df)
        heading_info_df['ratio_of_angle_to_nxt_ff'] = heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'] / \
            heading_info_df['angle_from_m_before_stop_to_nxt_ff']
    return heading_info_df


def _add_diff_in_d_heading_to_nxt_ff(heading_info_df):
    """
    Calculate the difference in directional heading angles between null and actual movement directions.

    This function computes the difference between the angle from the current firefly null trajectorylanding point 
    to the next firefly and the angle from the monkey's position before stopping to the next firefly.
    This helps quantify how the monkey's planned direction compares to their actual heading direction.

    Parameters:
    -----------
    heading_info_df : pandas.DataFrame
        DataFrame containing heading information with columns:
        - 'angle_from_cur_ff_landing_to_nxt_ff': angle from current firefly landing to next firefly
        - 'angle_from_m_before_stop_to_nxt_ff': angle from monkey position before stop to next firefly

    Returns:
    --------
    None
        Modifies the input DataFrame in-place by adding two new columns:
        - 'diff_in_angle_to_nxt_ff': difference between the two angles
        - 'diff_in_abs_angle_to_nxt_ff': difference between absolute values of the two angles
    """

    heading_info_df['diff_in_angle_to_nxt_ff'] = heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'] - \
        heading_info_df['angle_from_m_before_stop_to_nxt_ff']
    heading_info_df['diff_in_abs_angle_to_nxt_ff'] = np.abs(
        heading_info_df['angle_from_cur_ff_landing_to_nxt_ff']) - np.abs(heading_info_df['angle_from_m_before_stop_to_nxt_ff'])


def _add_diff_in_d_heading_to_cur_ff(heading_info_df):
    heading_info_df['diff_in_d_heading_to_cur_ff'] = heading_info_df['d_heading_of_traj'] - \
        heading_info_df['cur_d_heading_of_arc']
    heading_info_df['diff_in_d_heading_to_cur_ff'] = heading_info_df['diff_in_d_heading_to_cur_ff'] * 180/math.pi % 360
    heading_info_df.loc[heading_info_df['diff_in_d_heading_to_cur_ff'] > 180,
                        'diff_in_d_heading_to_cur_ff'] = heading_info_df.loc[heading_info_df['diff_in_d_heading_to_cur_ff'] > 180,
                                                                             'diff_in_d_heading_to_cur_ff'] - 360


def find_ff_visible_info_in_a_period(list_of_ff_index, ff_dataframe_visible, start_point_index=None, end_point_index=None, start_time=None, end_time=None):
    if (start_point_index is not None) & (end_point_index is not None):
        ff_info = ff_dataframe_visible[ff_dataframe_visible['point_index'].between(
            start_point_index, end_point_index)].copy()
    elif (start_time is not None) & (end_time is not None):
        ff_info = ff_dataframe_visible[ff_dataframe_visible['time'].between(
            start_time, end_time)].copy()
    else:
        raise ValueError(
            'Please provide either start_point_index and end_point_index or start_time and end_time.')
    ff_info = ff_info[ff_info['ff_index'].isin(list_of_ff_index)].copy()
    ff_visible_info = ff_info.copy()
    return ff_visible_info


# def find_ff_visible_duration_in_a_period(list_of_ff_index, ff_dataframe_visible, start_point_index=None, end_point_index=None, start_time=None, end_time=None):
#     ff_visible_info = find_ff_visible_info_in_a_period(list_of_ff_index, ff_dataframe_visible, start_point_index=start_point_index, end_point_index=end_point_index,
#                                                        start_time=start_time, end_time=end_time)
#     ff_visible_duration = ff_visible_info[[
#         'ff_index', 'dt']].groupby('ff_index').sum()
#     ff_visible_duration = ff_visible_duration.rename(
#         columns={'dt': 'visible_duration'})
#     ff_visible_duration = ff_visible_duration.copy()
#     return ff_visible_duration


def get_info_between_two_stops(stops_near_ff_df):
    df = stops_near_ff_df[[
        'stop_point_index', 'd_from_cur_ff_to_nxt_ff']].copy()
    df['cum_distance_between_two_stops'] = stops_near_ff_df['cum_distance_between_two_stops']
    df['time_between_two_stops'] = stops_near_ff_df['next_stop_time'] - \
        stops_near_ff_df['stop_time']
    return df


def get_distance_between_stop_and_arena_edge(stops_near_ff_df):
    # Get the radius of stop points
    radius = np.linalg.norm(
        stops_near_ff_df[['stop_x', 'stop_y']].values, axis=1)
    distance_between_stop_and_arena_edge = 1000 - radius
    return distance_between_stop_and_arena_edge


def get_nxt_ff_last_seen_info_before_next_stop(nxt_ff_df2, ff_dataframe_visible, monkey_information, stops_near_ff_df, ff_real_position_sorted):
    last_seen_point_index = build_factor_comp_utils._get_point_index_of_nxt_ff_last_seen_before_next_stop(
        ff_dataframe_visible, stops_near_ff_df)
    nxt_ff_df2 = find_stops_near_ff_utils.find_ff_info(
        stops_near_ff_df.nxt_ff_index.values, last_seen_point_index, monkey_information, ff_real_position_sorted)
    nxt_ff_df2['stop_point_index'] = stops_near_ff_df['stop_point_index'].values
    nxt_ff_df2['time_last_seen'] = monkey_information.loc[nxt_ff_df2['point_index'].values, 'time'].values
    nxt_ff_df2 = nxt_ff_df2.merge(stops_near_ff_df[[
                                  'stop_point_index', 'next_stop_time']], on='stop_point_index', how='left')
    nxt_ff_df2['time_nxt_ff_last_seen_before_next_stop'] = nxt_ff_df2['next_stop_time'].values - \
        nxt_ff_df2['time_last_seen'].values

    nxt_ff_last_seen_info = nxt_ff_df2[[
        'ff_distance', 'ff_angle', 'time_nxt_ff_last_seen_before_next_stop']].copy()
    nxt_ff_last_seen_info.rename(columns={'ff_distance': 'nxt_ff_distance_when_nxt_ff_last_seen_before_next_stop',
                                          'ff_angle': 'nxt_ff_angle_when_nxt_ff_last_seen_before_next_stop'}, inplace=True)
    return nxt_ff_last_seen_info


def add_d_monkey_angle(plan_y_df, cur_ff_df2, stops_near_ff_df):
    plan_y_df = plan_y_df.merge(stops_near_ff_df[[
                                'stop_point_index', 'stop_monkey_angle', 'monkey_angle_before_stop']], how='left')
    cur_ff_df2['monkey_angle_when_cur_ff_first_seen'] = cur_ff_df2['monkey_angle'] * 180 / math.pi
    if 'monkey_angle_when_cur_ff_first_seen' not in plan_y_df.columns:
        plan_y_df = plan_y_df.merge(cur_ff_df2[[
                                    'ff_index', 'monkey_angle_when_cur_ff_first_seen']], left_on='cur_ff_index', right_on='ff_index', how='left')
    plan_y_df['stop_monkey_angle'] = plan_y_df['stop_monkey_angle'] * 180/math.pi
    plan_y_df['monkey_angle_before_stop'] = plan_y_df['monkey_angle_before_stop'] * 180/math.pi
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen'] = plan_y_df['stop_monkey_angle'] - \
        plan_y_df['monkey_angle_when_cur_ff_first_seen']
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen2'] = plan_y_df['monkey_angle_before_stop'] - \
        plan_y_df['monkey_angle_when_cur_ff_first_seen']
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen'] = find_stops_near_ff_utils.confine_angle_to_within_180(
        plan_y_df['d_monkey_angle_since_cur_ff_first_seen'].values)
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen2'] = find_stops_near_ff_utils.confine_angle_to_within_180(
        plan_y_df['d_monkey_angle_since_cur_ff_first_seen2'].values)
    return plan_y_df


def add_dir_from_cur_ff_same_side(plan_y_df):
    plan_y_df['dir_from_cur_ff_to_stop'] = np.sign(
        plan_y_df['angle_from_cur_ff_to_stop'])
    plan_y_df['dir_from_cur_ff_to_nxt_ff'] = np.sign(
        plan_y_df['angle_from_cur_ff_to_nxt_ff'])
    plan_y_df['dir_from_cur_ff_same_side'] = plan_y_df['dir_from_cur_ff_to_stop'] == plan_y_df['dir_from_cur_ff_to_nxt_ff']
    plan_y_df['dir_from_cur_ff_same_side'] = plan_y_df['dir_from_cur_ff_same_side'].astype(
        int)


def make_cluster_df_as_part_of_plan_factors(stops_near_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted,
                                            stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150, ff_radius=10,
                                            list_of_cur_ff_cluster_radius=[
                                                100, 200, 300],
                                            list_of_nxt_ff_cluster_radius=[
                                                100, 200, 300],
                                            guarantee_cur_ff_info_for_cluster=False,
                                            guarantee_nxt_ff_info_for_cluster=False,
                                            flash_or_vis='vis',
                                            columns_not_to_include=[]
                                            ):

    all_start_time = stops_near_ff_df['stop_time'].values - \
        stop_period_duration
    all_end_time = stops_near_ff_df['next_stop_time'].values
    all_segment_id = stops_near_ff_df['stop_point_index'].values

    monkey_info_in_all_stop_periods = only_cur_ff_utils.find_monkey_info_in_all_stop_periods(
        all_start_time, all_end_time, all_segment_id, monkey_information)
    monkey_info_to_add = [column for column in monkey_info_in_all_stop_periods.columns if (
        column not in ff_dataframe_visible.columns)]
    ff_info_in_all_stop_periods = ff_dataframe_visible.merge(
        monkey_info_in_all_stop_periods[monkey_info_to_add + ['point_index']], on=['point_index'], how='right')
    ff_info_in_all_stop_periods = ff_info_in_all_stop_periods[~ff_info_in_all_stop_periods['ff_index'].isnull(
    )].copy()

    # for each ff_index in each stop_period, we preserve only one row
    vis_time_info = ff_info_in_all_stop_periods.groupby(['ff_index', 'stop_point_index']).agg(earliest_vis_point_index=('point_index', 'min'),
                                                                                              latest_vis_point_index=(
                                                                                                  'point_index', 'max'),
                                                                                              earliest_vis_time=(
                                                                                                  'time', 'min'),
                                                                                              latest_vis_time=(
                                                                                                  'time', 'max'),
                                                                                              vis_duration=('dt', 'sum'))

    if guarantee_cur_ff_info_for_cluster:
        cur_ff_info = stops_near_ff_df[['stop_point_index', 'cur_ff_index']].rename(
            columns={'cur_ff_index': 'ff_index'}).copy()
        vis_time_info = vis_time_info.merge(
            cur_ff_info, on=['ff_index', 'stop_point_index'], how='outer')
    if guarantee_nxt_ff_info_for_cluster:
        nxt_ff_info = stops_near_ff_df[['stop_point_index', 'nxt_ff_index']].rename(
            columns={'nxt_ff_index': 'ff_index'}).copy()
        vis_time_info = vis_time_info.merge(
            nxt_ff_info, on=['ff_index', 'stop_point_index'], how='outer')

    vis_time_info.reset_index(drop=False, inplace=True)
    vis_time_info['ff_index'] = vis_time_info['ff_index'].astype(int)

    # add stops_near_ff_df info to ff_info_in_all_stop_periods, but also be careful to duplicated columns
    stops_near_ff_columns_to_add = [column for column in stops_near_ff_df.columns if (
        column not in vis_time_info.columns)]
    ff_info_in_all_stop_periods = vis_time_info.merge(
        stops_near_ff_df[stops_near_ff_columns_to_add + ['stop_point_index']], on='stop_point_index', how='left')
    ff_info_in_all_stop_periods.reset_index(drop=True, inplace=True)

    # add info at ref point
    _, _, cur_ff_df = nxt_ff_utils.get_nxt_ff_df_and_cur_ff_df(
        stops_near_ff_df)
    cur_ff_df2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(cur_ff_df, monkey_information, ff_real_position_sorted,
                                                                          ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)

    ref_info = cur_ff_df2[['stop_point_index', 'point_index', 'monkey_x', 'monkey_y',
                           'monkey_angle']].rename(columns={'point_index': 'ref_point_index'}).copy()
    ref_info['ref_time'] = monkey_information.loc[ref_info['ref_point_index'].values, 'time'].values
    ref_info['stop_time'] = monkey_information.loc[ref_info['stop_point_index'].values, 'time'].values
    ref_info['beginning_time'] = ref_info['stop_time'] - stop_period_duration
    ref_info_columns_to_add = [column for column in ref_info.columns if (
        column not in ff_info_in_all_stop_periods.columns)]
    ff_info_in_all_stop_periods = ff_info_in_all_stop_periods.merge(
        ref_info[ref_info_columns_to_add + ['stop_point_index']], on='stop_point_index', how='left')

    # add ff info
    ff_info_in_all_stop_periods['ff_x'], ff_info_in_all_stop_periods[
        'ff_y'] = ff_real_position_sorted[ff_info_in_all_stop_periods['ff_index'].values].T
    ff_info_in_all_stop_periods = only_cur_ff_utils._add_basic_ff_info_to_df_for_ff(
        ff_info_in_all_stop_periods, ff_radius=ff_radius)
    ff_info_in_all_stop_periods = furnish_ff_info_in_all_stop_periods(
        ff_info_in_all_stop_periods)

    # identify clusters based on various criteria
    ff_info_in_all_stop_periods, all_cluster_names = build_factor_comp_utils._find_clusters_in_ff_info_in_all_stop_periods(ff_info_in_all_stop_periods, list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius,
                                                                                                                           list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius)

    # get cluster info
    cluster_factors_df, cluster_agg_df = only_cur_ff_utils.get_cluster_and_agg_df(ff_info_in_all_stop_periods, all_cluster_names,
                                                                                  flash_or_vis=flash_or_vis, columns_not_to_include=columns_not_to_include)
    # combine cluster info
    cluster_df = cluster_factors_df.merge(
        cluster_agg_df, on='stop_point_index', how='outer').reset_index(drop=True)

    return cluster_df


def furnish_ff_info_in_all_stop_periods(df):

    df['earliest_vis_rel_time'] = df['earliest_vis_time'] - df['beginning_time']
    df['latest_vis_rel_time'] = df['latest_vis_time'] - df['beginning_time']

    df['cur_ff_distance'] = np.linalg.norm(
        [df['cur_ff_x'] - df['ff_x'], df['cur_ff_y'] - df['ff_y']], axis=0)
    df['ff_distance_to_cur_ff'] = np.linalg.norm(
        [df['cur_ff_x'] - df['ff_x'], df['cur_ff_y'] - df['ff_y']], axis=0)
    df['ff_distance_to_nxt_ff'] = np.linalg.norm(
        [df['nxt_ff_x'] - df['ff_x'], df['nxt_ff_y'] - df['ff_y']], axis=0)

    df['angle_diff_boundary'] = df['ff_angle'] - df['ff_angle_boundary']
    df['angle_diff_boundary'] = df['angle_diff_boundary'] % (2*math.pi)
    df.loc[df['angle_diff_boundary'] > math.pi, 'angle_diff_boundary'] = df.loc[df['angle_diff_boundary']
                                                                                > math.pi, 'angle_diff_boundary'] - 2*math.pi
    return df


def add_monkey_speed_stats_to_df(df, stops_near_ff_df, monkey_information):
    # get info between some time before stop (determined by beginning_time) and stop time
    monkey_speed_stat_df_bbas = build_factor_comp_utils._get_monkey_speed_stat_df_bbas(
        stops_near_ff_df, monkey_information)
    df = df.merge(
        monkey_speed_stat_df_bbas, on='stop_point_index', how='left')

    # get info between stop time and next stop time
    monkey_speed_stat_df_bsans = build_factor_comp_utils._get_monkey_speed_stat_df_bsans(
        stops_near_ff_df, monkey_information)
    df = df.merge(
        monkey_speed_stat_df_bsans, on='stop_point_index', how='left')
    return df


def add_monkey_eye_stats_to_df(df, stops_near_ff_df, monkey_information):
    # get info between some time before stop (determined by beginning_time) and stop time
    eye_stat_df_bbas, eye_toward_ff_time_perc_df_bbas = build_factor_comp_utils._get_eye_stats_bbas(
        stops_near_ff_df, monkey_information)
    df = df.merge(eye_stat_df_bbas, on='stop_point_index', how='left').merge(
        eye_toward_ff_time_perc_df_bbas, on='stop_point_index', how='left').reset_index(drop=True)

    # get info between stop time and next stop time
    eye_stat_df_bsans, eye_toward_ff_time_perc_df_bsans = build_factor_comp_utils._get_eye_stats_bsans(
        stops_near_ff_df, monkey_information)
    df = df.merge(eye_stat_df_bsans, on='stop_point_index', how='left').merge(
        eye_toward_ff_time_perc_df_bsans, on='stop_point_index', how='left').reset_index(drop=True)
    return df


def find_curv_of_traj_stat_df(df, curv_of_traj_df, start_time_column='stop_time',
                              end_time_column='next_stop_time', groupby_column='stop_point_index'
                              ):

    curv_of_traj_df = curv_of_traj_df.copy()
    curv_of_traj_df['curv_of_traj'] = curv_of_traj_df['curv_of_traj'] * \
        180/math.pi * 100

    data_of_segments = build_factor_comp_utils._take_out_info_of_all_segments(curv_of_traj_df,
                                                                              all_start_time=df[start_time_column].values,
                                                                              all_end_time=df[end_time_column].values,
                                                                              all_segment_id=df[groupby_column].values,
                                                                              group_id='stop_point_index')

    curv_of_traj_stat_df = build_factor_comp_utils._find_summary_stats_of_each_segment(data_of_segments,
                                                                                       stat_columns=[
                                                                                           'curv_of_traj'],
                                                                                       stat_column_prefixes=[
                                                                                           'curv'],
                                                                                       groupby_column='stop_point_index')

    if curv_of_traj_stat_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Warning: {curv_of_traj_stat_df.isnull().any(axis=1).sum()} rows have NaN values in curv_of_traj_stat_df.')

    return curv_of_traj_stat_df
