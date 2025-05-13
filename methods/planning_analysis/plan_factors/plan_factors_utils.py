import sys
from planning_analysis.show_planning import nxt_ff_utils, show_planning_class, show_planning_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_class, plot_stops_near_ff_utils, plot_monkey_heading_helper_class, stops_near_ff_based_on_ref_class
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils
from data_wrangling import specific_utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
import os
from math import pi


def drop_na_in_x_var(x_var_df, y_var_df):
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)

    if x_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in x_var_df: {x_var_df.isnull().any(axis=1).sum()} out of {x_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in x_var_df
        x_var_df = x_var_df.dropna()
        y_var_df = y_var_df.loc[x_var_df.index].copy()

    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    return x_var_df, y_var_df


def drop_na_in_x_and_y_var(x_var_df, y_var_df):
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)

    if x_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in x_var_df: {x_var_df.isnull().any(axis=1).sum()} out of {x_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in x_var_df
        x_var_df = x_var_df.dropna()
        y_var_df = y_var_df.loc[x_var_df.index].copy()
    if y_var_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in y_var_df (after cleaning x_var_df and the corresponding rows in y_var_df): {y_var_df.isnull().any(axis=1).sum()} out of {y_var_df.shape[0]} rows. The rows with NaN values will be dropped.')
        # drop rows with NA in y_var_df
        y_var_df = y_var_df.dropna()
        x_var_df = x_var_df.loc[y_var_df.index].copy()

    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)
    return x_var_df, y_var_df


def find_curv_of_traj_stat_df(df_to_iter, curv_of_traj_df, start_time_column='stop_time',
                              end_time_column='next_stop_time', add_to_df_to_iter=True,
                              ):

    groupby_column = 'stop_point_index'
    stat_prefix = ['curv']
    curv_of_traj_df = curv_of_traj_df.copy()
    curv_of_traj_df['curv_of_traj'] = curv_of_traj_df['curv_of_traj'] * \
        180/math.pi * 100

    df_for_stat_extended = extend_df_based_on_groups(curv_of_traj_df,
                                                     all_start_time=df_to_iter[start_time_column].values,
                                                     all_end_time=df_to_iter[end_time_column].values,
                                                     all_group_id=df_to_iter[groupby_column].values,
                                                     group_id='stop_point_index')

    curv_of_traj_stat_df = find_stat_of_columns_after_groupby(df_for_stat_extended,
                                                              groupby_column='stop_point_index',
                                                              stat_columns=[
                                                                  'curv_of_traj'],
                                                              stat_prefix=stat_prefix)

    if add_to_df_to_iter:
        df_to_iter = add_stat_columns_to_df_to_iter(
            curv_of_traj_stat_df, df_to_iter, stat_prefix, groupby_column)

    return curv_of_traj_stat_df, df_to_iter


def extend_df_based_on_groups(ori_df, all_start_time, all_end_time, all_group_id, group_id='stop_point_index'):
    # This function takes an original DataFrame (ori_df) and extends it by including all rows that fall within specified time intervals.
    # It also assigns a group ID to each of these rows based on the provided group IDs. Here's a step-by-step explanation of what the function does:

    # Take out all the time points between start time and stop time for all captured ff
    extended_cum_indices = []
    extended_group_id = []
    for i in range(len(all_start_time)):
        start_time = all_start_time[i]
        end_time = all_end_time[i]
        current_group_id = all_group_id[i]
        # Find the corresponding monkey information:
        cum_indices = ori_df.loc[ori_df['time'].between(
            start_time, end_time)].index.values
        extended_cum_indices.extend(cum_indices)
        extended_group_id.extend([current_group_id] * len(cum_indices))
    extended_cum_indices = np.array(extended_cum_indices).astype('int')
    extended_df = ori_df.loc[extended_cum_indices].copy()
    extended_df[group_id] = extended_group_id
    extended_df.reset_index(drop=True, inplace=True)

    return extended_df


def add_stat_columns_to_df_to_iter(stat_df, df_to_iter, stat_prefix, groupby_column):
    for prefix in stat_prefix:
        columns_to_add = [f'{prefix}_mean', f'{prefix}_std', f'{prefix}_min', f'{prefix}_Q1', f'{prefix}_median', f'{prefix}_Q3',
                          f'{prefix}_max', f'{prefix}_iqr', f'{prefix}_range']
        # drop the columns if already exist
        df_to_iter = df_to_iter.drop(columns=columns_to_add, errors='ignore')
        df_to_iter = df_to_iter.merge(
            stat_df[columns_to_add + [groupby_column]], on=groupby_column, how='left')
    return df_to_iter


def find_stat_of_columns_after_groupby(df_for_stat_extended,
                                       groupby_column='stop_point_index',
                                       stat_columns=['curv_of_traj'],
                                       stat_prefix=None):

    if stat_prefix is None:
        stat_prefix = stat_columns
    stat_prefix = [prefix + '_' for prefix in stat_prefix]

    # make a dict of column to prefix
    column_prefix_map = {column: prefix for column, prefix in zip(stat_columns, stat_prefix)}

    # Note: The describe() method after using groupby generates a df with multi-level column names,
    # where the first level is the original column name (e.g., 'curv_of_traj') and the second level is the statistical measure (e.g., 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max').
    
    # Group by the specified column and calculate descriptive statistics
    stat_df = df_for_stat_extended.groupby(groupby_column)[stat_columns].describe()

    # Rename the percentile columns for better readability
    stat_df.rename(columns={'25%': 'Q1', '50%': 'median', '75%': 'Q3'}, inplace=True)

    # Flatten the multi-level column names and add prefixes
    stat_df.columns = [f"{column_prefix_map[column[0]]}{column[1]}" for column in stat_df.columns]

    for prefix in stat_prefix:
        stat_df[prefix + 'range'] = stat_df[prefix + 'max'] - stat_df[prefix + 'min']
        stat_df[prefix + 'iqr'] = stat_df[prefix + 'Q3'] - stat_df[prefix + 'Q1']
        stat_df.drop(columns=prefix + 'count', inplace=True)
    stat_df.reset_index(drop=False, inplace=True)
    return stat_df


def get_eye_perc_df(df_for_stat_extended, list_of_max_degrees=[5, 10]):
    df = df_for_stat_extended.copy()

    eye_columns = []
    eye_to_gaze = {'left_eye': 'gaze_mky_view_angle_l',
                   'right_eye': 'gaze_mky_view_angle_r'
                   }

    for eye in ['left_eye', 'right_eye']:
        for cur_or_nxt in ['cur', 'nxt']:
            for max_degrees in list_of_max_degrees:
                new_column = f'{eye}_{cur_or_nxt}_ff_time_perc_{max_degrees}'
                df[new_column] = df['dt']
                df.loc[(df[eye_to_gaze[eye]] - df[f'{cur_or_nxt}_ff_angle']).abs(
                ) > max_degrees/180 * math.pi, new_column] = 0
                df.loc[df[eye_to_gaze[eye]].isnull(), new_column] = 0
                eye_columns.append(new_column)

    eye_perc_df = df[eye_columns + ['dt', 'stop_point_index']
                     ].groupby('stop_point_index').sum()

    for column in eye_columns:
        eye_perc_df[column] = eye_perc_df[column]/(eye_perc_df['dt'].values)

    eye_perc_df.reset_index(drop=False, inplace=True)
    return eye_perc_df


def get_df_for_stat_extended_for_eye_info(stops_near_ff_df, monkey_information):
    df_to_iter = stops_near_ff_df.copy()
    groupby_column = 'stop_point_index'
    start_time_column = 'beginning_time'
    end_time_column = 'stop_time'

    df_for_stat_extended = extend_df_based_on_groups(monkey_information,
                                                     all_start_time=df_to_iter[start_time_column].values,
                                                     all_end_time=df_to_iter[end_time_column].values,
                                                     all_group_id=df_to_iter[groupby_column].values,
                                                     group_id='stop_point_index')

    df_for_stat_extended = df_for_stat_extended.merge(stops_near_ff_df[['stop_point_index', 'cur_ff_x', 'cur_ff_y', 'nxt_ff_x', 'nxt_ff_y']],
                                                      on='stop_point_index', how='left')

    df_for_stat_extended['cur_ff_angle'] = specific_utils.calculate_angles_to_ff_centers(df_for_stat_extended['cur_ff_x'], df_for_stat_extended['cur_ff_y'], df_for_stat_extended['monkey_x'],
                                                                                         df_for_stat_extended['monkey_y'], df_for_stat_extended['monkey_angle'])

    df_for_stat_extended['nxt_ff_angle'] = specific_utils.calculate_angles_to_ff_centers(df_for_stat_extended['nxt_ff_x'], df_for_stat_extended['nxt_ff_y'], df_for_stat_extended['monkey_x'],
                                                                                         df_for_stat_extended['monkey_y'], df_for_stat_extended['monkey_angle'])
    return df_for_stat_extended


def make_plan_y_df(heading_info_df, curv_of_traj_df, curv_of_traj_df_w_one_sided_window):

    curv_of_traj_stat_df, _ = find_curv_of_traj_stat_df(
        heading_info_df, curv_of_traj_df, add_to_df_to_iter=False)
    curv_of_traj_stat_df = curv_of_traj_stat_df[['curv_mean', 'curv_std', 'curv_min', 'curv_Q1', 'curv_median', 'curv_Q3',
                                                'curv_max', 'curv_iqr', 'curv_range', 'stop_point_index']]
    # print the number of rows with NA and drop them
    if curv_of_traj_stat_df.isnull().any(axis=1).sum() > 0:
        print(
            f'Number of rows with NaN values in plan_y_df: {curv_of_traj_stat_df.isnull().any(axis=1).sum()} out of {curv_of_traj_stat_df.shape[0]} rows.')
        # drop rows with NA in plan_y_df
        # plan_y_df = plan_y_df.dropna()

    # merge plan_y_df with heading_info_df to get more variables
    # heading_info_df = heading_info_df[['stop_point_index', 'angle_from_m_before_stop_to_nxt_ff', 'angle_from_cur_ff_landing_to_nxt_ff',
    #                                    'd_heading_of_traj', 'cur_d_heading_of_arc',
    #                                    'stop_time', 'NXT_time_ff_last_seen_bbas',
    #                                     'nxt_ff_cluster_last_seen_time_bbas',
    #                                     'nxt_ff_last_flash_time_bbas',
    #                                     'nxt_ff_cluster_last_flash_time_bbas']].copy()
    heading_info_df = process_heading_info_df(heading_info_df)

    columns_to_add = ['curv_mean', 'curv_std', 'curv_min', 'curv_Q1', 'curv_median', 'curv_Q3',
                      'curv_max', 'curv_iqr', 'curv_range']
    # drop the columns if already exist
    heading_info_df = heading_info_df.drop(
        columns=columns_to_add, errors='ignore')
    plan_y_df = heading_info_df.merge(
        curv_of_traj_stat_df[columns_to_add + ['stop_point_index']], on='stop_point_index', how='left')

    if curv_of_traj_df_w_one_sided_window is not None:
        plan_y_df = _add_column_curv_of_traj_before_stop(
            plan_y_df, curv_of_traj_df_w_one_sided_window)

    add_dir_from_cur_ff_same_side(plan_y_df)

    plan_y_df = plan_y_df.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_y_df


def _add_column_curv_of_traj_before_stop(df, curv_of_traj_df_w_one_sided_window):
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
    if 'angle_from_cur_ff_landing_to_nxt_ff' in heading_info_df.columns:
        heading_info_df[['angle_from_m_before_stop_to_nxt_ff', 'angle_from_cur_ff_landing_to_nxt_ff']] = heading_info_df[[
            'angle_from_m_before_stop_to_nxt_ff', 'angle_from_cur_ff_landing_to_nxt_ff']] * (180/np.pi)
        heading_info_df['diff_in_d_heading_of_traj_from_null'] = heading_info_df['d_heading_of_traj'] - \
            heading_info_df['cur_d_heading_of_arc']
        heading_info_df['diff_in_d_heading_of_traj_from_null'] = heading_info_df['diff_in_d_heading_of_traj_from_null'] * 180/math.pi % 360
        heading_info_df.loc[heading_info_df['diff_in_d_heading_of_traj_from_null'] > 180,
                            'diff_in_d_heading_of_traj_from_null'] = heading_info_df.loc[heading_info_df['diff_in_d_heading_of_traj_from_null'] > 180, 'diff_in_d_heading_of_traj_from_null'] - 360
        heading_info_df['diff_in_d_heading'] = heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'] - \
            heading_info_df['angle_from_m_before_stop_to_nxt_ff']
        heading_info_df['diff_in_abs_d_heading'] = np.abs(
            heading_info_df['angle_from_cur_ff_landing_to_nxt_ff']) - np.abs(heading_info_df['angle_from_m_before_stop_to_nxt_ff'])
        heading_info_df['ratio'] = heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'] / \
            heading_info_df['angle_from_m_before_stop_to_nxt_ff']
    return heading_info_df


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


def find_ff_visible_duration_in_a_period(list_of_ff_index, ff_dataframe_visible, start_point_index=None, end_point_index=None, start_time=None, end_time=None):
    ff_visible_info = find_ff_visible_info_in_a_period(list_of_ff_index, ff_dataframe_visible, start_point_index=start_point_index, end_point_index=end_point_index,
                                                       start_time=start_time, end_time=end_time)
    ff_visible_duration = ff_visible_info[[
        'ff_index', 'dt']].groupby('ff_index').sum()
    ff_visible_duration = ff_visible_duration.rename(
        columns={'dt': 'visible_duration'})
    ff_visible_duration = ff_visible_duration.copy()
    return ff_visible_duration


def get_other_factors(stops_near_ff_df):
    other_factors = stops_near_ff_df[[
        'stop_point_index', 'd_from_cur_ff_to_nxt_ff']].copy()
    # Get the radius of stop points
    radius = np.linalg.norm(
        stops_near_ff_df[['stop_x', 'stop_y']].values, axis=1)
    other_factors['distance_between_stop_and_arena_edge'] = 1000 - radius

    other_factors['cum_distance_between_two_stops'] = stops_near_ff_df['cum_distance_between_two_stops']
    other_factors['time_between_two_stops'] = stops_near_ff_df['next_stop_time'] - \
        stops_near_ff_df['stop_time']
    return other_factors


def get_point_index_of_nxt_ff_last_seen_before_next_stop(ff_dataframe_visible, stops_near_ff_df):
    all_group_identifiers, all_stop_time, all_next_stop_time, all_ff_index = stops_near_ff_df['stop_point_index'].values, stops_near_ff_df[
        'stop_time'].values, stops_near_ff_df['next_stop_time'].values, stops_near_ff_df['nxt_ff_index'].values
    # time_between_stop_and_next_stop = stops_near_ff_df['next_stop_time'].values - stops_near_ff_df['stop_time'].values
    ff_last_seen_info = nxt_ff_utils.find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_group_identifiers, all_stop_time-2.5,
                                                                                                              all_next_stop_time, ff_dataframe_visible, first_or_last='last'
                                                                                                              )
    last_seen_point_index = ff_last_seen_info['point_index'].values
    # replace values on na in last_seen_point_index with 0
    last_seen_point_index[np.isnan(last_seen_point_index)] = 0

    return last_seen_point_index


def get_nxt_ff_last_seen_info_before_next_stop(nxt_ff_df2, ff_dataframe_visible, monkey_information, stops_near_ff_df, ff_real_position_sorted):
    last_seen_point_index = get_point_index_of_nxt_ff_last_seen_before_next_stop(
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
    plan_y_df['monkey_angle_when_cur_ff_first_seen'] = cur_ff_df2['monkey_angle'].values * 180 / math.pi
    plan_y_df['stop_monkey_angle'] = plan_y_df['stop_monkey_angle'] * 180/math.pi
    plan_y_df['monkey_angle_before_stop'] = plan_y_df['monkey_angle_before_stop'] * 180/math.pi
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen'] = plan_y_df['stop_monkey_angle'] - \
        plan_y_df['monkey_angle_when_cur_ff_first_seen']
    plan_y_df['d_monkey_angle2'] = plan_y_df['monkey_angle_before_stop'] - \
        plan_y_df['monkey_angle_when_cur_ff_first_seen']
    plan_y_df['d_monkey_angle_since_cur_ff_first_seen'] = find_stops_near_ff_utils.confine_angle_to_within_180(
        plan_y_df['d_monkey_angle_since_cur_ff_first_seen'].values)
    plan_y_df['d_monkey_angle2'] = find_stops_near_ff_utils.confine_angle_to_within_180(
        plan_y_df['d_monkey_angle2'].values)
    return plan_y_df


def add_dir_from_cur_ff_same_side(plan_y_df):
    plan_y_df['dir_from_cur_ff_to_stop'] = np.sign(
        plan_y_df['angle_from_cur_ff_to_stop'])
    plan_y_df['dir_from_cur_ff_to_nxt_ff'] = np.sign(
        plan_y_df['angle_from_cur_ff_to_nxt_ff'])
    plan_y_df['dir_from_cur_ff_same_side'] = plan_y_df['dir_from_cur_ff_to_stop'] == plan_y_df['dir_from_cur_ff_to_nxt_ff']


def get_eye_data_etc(stops_near_ff_df, monkey_information, ff_real_position_sorted, max_degrees=5,
                     column_suffix=''):
    left_eye_nxt_ff_time_perc = []  # meaning left_eye_within_n_deg_to_nxt_ff_time
    right_eye_nxt_ff_time_perc = []
    left_eye_cur_ff_time_perc = []
    right_eye_cur_ff_time_perc = []
    left_eye_nxt_ff_after_stop_time_perc = []
    right_eye_nxt_ff_after_stop_time_perc = []
    raw_eye_data_df = pd.DataFrame()
    monkey_speed_df = pd.DataFrame()
    for index, row in stops_near_ff_df.iterrows():
        duration = [row['stop_time'] - 2, row['stop_time']]
        duration_length = duration[1] - duration[0]
        ff_index = row['cur_ff_index']
        left_eye_time, right_eye_time, monkey_sub = _get_left_and_right_eye_time(
            ff_index, duration, monkey_information, ff_real_position_sorted, max_degrees)
        left_eye_cur_ff_time_perc.append(left_eye_time/duration_length)
        right_eye_cur_ff_time_perc.append(right_eye_time/duration_length)
        raw_eye_data_row = get_quartile_data_in_a_row(
            monkey_sub, ['LDy', 'LDz', 'RDy', 'RDz'], suffix='')
        monkey_speed_row = get_quartile_data_in_a_row(
            monkey_sub, ['monkey_speed', 'monkey_dw'], suffix='')

        ff_index = row['nxt_ff_index']
        left_eye_time, right_eye_time, monkey_sub = _get_left_and_right_eye_time(
            ff_index, duration, monkey_information, ff_real_position_sorted, max_degrees)
        left_eye_nxt_ff_time_perc.append(left_eye_time/duration_length)
        right_eye_nxt_ff_time_perc.append(right_eye_time/duration_length)

        duration = [row['stop_time'], row['next_stop_time']]
        duration_length = duration[1] - duration[0]
        left_eye_time, right_eye_time, monkey_sub = _get_left_and_right_eye_time(
            ff_index, duration, monkey_information, ff_real_position_sorted, max_degrees)
        left_eye_nxt_ff_after_stop_time_perc.append(
            left_eye_time/duration_length)
        right_eye_nxt_ff_after_stop_time_perc.append(
            right_eye_time/duration_length)

        raw_eye_data_row2 = get_quartile_data_in_a_row(
            monkey_sub, ['LDy', 'LDz', 'RDy', 'RDz'], suffix='_after_stop')
        raw_eye_data_row = pd.concat(
            [raw_eye_data_row, raw_eye_data_row2], axis=1)
        raw_eye_data_df = pd.concat(
            [raw_eye_data_df, raw_eye_data_row], axis=0)

        monkey_speed_row2 = get_quartile_data_in_a_row(
            monkey_sub, ['monkey_speed', 'monkey_dw'], suffix='_after_stop')
        monkey_speed_row = pd.concat(
            [monkey_speed_row, monkey_speed_row2], axis=1)
        monkey_speed_df = pd.concat(
            [monkey_speed_df, monkey_speed_row], axis=0)

    raw_eye_data_df = raw_eye_data_df.reset_index(drop=True)
    monkey_speed_df = monkey_speed_df.reset_index(drop=True)

    eye_factor_df = pd.DataFrame({'left_eye_nxt_ff_time_perc' + column_suffix: np.array(left_eye_nxt_ff_time_perc),
                                  'right_eye_nxt_ff_time_perc' + column_suffix: np.array(right_eye_nxt_ff_time_perc),
                                  'left_eye_cur_ff_time_perc' + column_suffix: np.array(left_eye_cur_ff_time_perc),
                                  'right_eye_cur_ff_time_perc' + column_suffix: np.array(right_eye_cur_ff_time_perc),
                                  'left_eye_nxt_ff_after_stop_time_perc' + column_suffix: np.array(left_eye_nxt_ff_after_stop_time_perc),
                                  'right_eye_nxt_ff_after_stop_time_perc' + column_suffix: np.array(right_eye_nxt_ff_after_stop_time_perc),
                                  })
    return eye_factor_df, raw_eye_data_df, monkey_speed_df


def get_quartile_data_in_a_row(monkey_sub, columns, suffix=''):
    quartile_data_dict = {}
    for column in columns:
        quartile_data_dict[column + '_Q1' +
                           suffix] = monkey_sub[column].quantile(0.25)
        quartile_data_dict[column + '_median' +
                           suffix] = monkey_sub[column].quantile(0.5)
        quartile_data_dict[column + '_Q3' +
                           suffix] = monkey_sub[column].quantile(0.75)
    quartile_data_row = pd.DataFrame(quartile_data_dict, index=[0])
    return quartile_data_row


def _get_left_and_right_eye_time(ff_index, duration, monkey_information, ff_real_position_sorted, max_degrees=5):

    monkey_sub = monkey_information[(monkey_information['time'] >= duration[0]) & (
        monkey_information['time'] <= duration[1])].copy()
    monkey_sub[['ff_x', 'ff_y']] = ff_real_position_sorted[ff_index]
    monkey_sub['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(monkey_sub['ff_x'], monkey_sub['ff_y'], monkey_sub['monkey_x'],
                                                                           monkey_sub['monkey_y'], monkey_sub['monkey_angle'])
    left_eye = monkey_sub[(monkey_sub['gaze_mky_view_angle_l'] -
                           monkey_sub['ff_angle']).abs() <= max_degrees/180 * math.pi].copy()
    left_eye_time = left_eye['dt'].sum()
    right_eye = monkey_sub[(monkey_sub['gaze_mky_view_angle_r'] -
                            monkey_sub['ff_angle']).abs() <= max_degrees/180 * math.pi].copy()
    right_eye_time = right_eye['dt'].sum()

    return left_eye_time, right_eye_time, monkey_sub


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
    all_group_id = stops_near_ff_df['stop_point_index'].values

    monkey_info_in_all_stop_periods = only_cur_ff_utils.find_monkey_info_in_all_stop_periods(
        all_start_time, all_end_time, all_group_id, monkey_information)
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
    ff_info_in_all_stop_periods, all_cluster_names = find_clusters_in_ff_info_in_all_stop_periods(ff_info_in_all_stop_periods, list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius,
                                                                                                  list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius)

    # get cluster info
    cluster_factors_df, cluster_agg_df = only_cur_ff_utils.get_cluster_and_agg_df(ff_info_in_all_stop_periods, all_cluster_names,
                                                                                  flash_or_vis=flash_or_vis, columns_not_to_include=columns_not_to_include)
    # combine cluster info
    cluster_df = cluster_factors_df.merge(
        cluster_agg_df, on='stop_point_index', how='outer').reset_index(drop=True)

    return cluster_df


def find_clusters_in_ff_info_in_all_stop_periods(ff_info_in_all_stop_periods,
                                                 list_of_cur_ff_cluster_radius=[
                                                     100, 200, 300],
                                                 list_of_nxt_ff_cluster_radius=[
                                                     100, 200, 300],
                                                 ):

    all_cluster_names = []
    for n_cm in list_of_cur_ff_cluster_radius:
        column = f'cur_ff_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_in_all_stop_periods[column] = False
        ff_info_in_all_stop_periods.loc[ff_info_in_all_stop_periods['ff_distance_to_cur_ff']
                                        <= n_cm, column] = True

    for n_cm in list_of_nxt_ff_cluster_radius:
        column = f'nxt_ff_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_in_all_stop_periods[column] = False
        ff_info_in_all_stop_periods.loc[ff_info_in_all_stop_periods['ff_distance_to_nxt_ff']
                                        <= n_cm, column] = True

    whether_each_cluster_has_enough = ff_info_in_all_stop_periods[all_cluster_names + [
        'stop_point_index']].groupby('stop_point_index').sum() == 0
    where_need_cur_ff = np.where(whether_each_cluster_has_enough)
    stop_periods = whether_each_cluster_has_enough.index.values[where_need_cur_ff[0]]
    groups = np.array(all_cluster_names)[where_need_cur_ff[1]]

    if len(where_need_cur_ff[0]) > 0:
        raise ValueError('Some clusters have 0 ff!')

    return ff_info_in_all_stop_periods, all_cluster_names


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


monkey_info_columns = ['left_eye_cur_ff_time_perc_5',
                       'left_eye_cur_ff_time_perc_10',
                       'left_eye_nxt_ff_time_perc_5',
                       'left_eye_nxt_ff_time_perc_10',
                       'right_eye_cur_ff_time_perc_5',
                       'right_eye_cur_ff_time_perc_10',
                       'right_eye_nxt_ff_time_perc_5',
                       'right_eye_nxt_ff_time_perc_10',
                       'LDy_std',
                       'LDz_std',
                       'RDy_std',
                       'RDz_std',
                       'LDy_range',
                       'LDy_iqr',
                       'LDz_range',
                       'LDz_iqr',
                       'RDy_range',
                       'RDy_iqr',
                       'RDz_range',
                       'RDz_iqr',
                       'monkey_speed_range',
                       'monkey_speed_iqr',
                       'monkey_dw_range',
                       'monkey_dw_iqr',
                       'monkey_speed_std',
                       'monkey_dw_std',
                       'curv_range',
                       'curv_iqr']

more_monkey_info_columns = ['d_heading_of_traj',
                            'ref_curv_of_traj',
                            'curv_mean',
                            'curv_std',
                            'curv_min',
                            'curv_Q1',
                            'curv_median',
                            'curv_Q3',
                            'curv_max']


ff_at_ref_columns = ['nxt_ff_distance_at_ref',
                     'nxt_ff_angle_at_ref',
                     'cur_ff_distance_at_ref',
                     'cur_ff_angle_at_ref',
                     'cur_ff_angle_boundary_at_ref',
                     'cur_ff_angle_diff_boundary_at_ref']


# def _process_plan_x_for_prediction(plan_x):
#     plan_x['curv_range'] = plan_x['curv_max'] - plan_x['curv_min']
#     plan_x['curv_iqr'] = plan_x['curv_Q3'] - plan_x['curv_Q1']

#     non_cluster_columns_to_save = ['distance_between_stop_and_arena_edge'] + ff_at_ref_columns + monkey_info_columns

#     cluster_columns_to_save = [col for col in plan_x.columns if (col in non_cluster_columns_to_save) | ('cluster' in col)]
#     plan_x = plan_x[cluster_columns_to_save].copy()
#     return plan_x


def process_plan_x_to_predict_monkey_info(plan_x, for_classification=False):
    plan_x['curv_range'] = plan_x['curv_max'] - plan_x['curv_min']
    plan_x['curv_iqr'] = plan_x['curv_Q3'] - plan_x['curv_Q1']

    non_cluster_columns_to_save = [
        'distance_between_stop_and_arena_edge'] + ff_at_ref_columns + monkey_info_columns

    cluster_columns_to_save = [col for col in plan_x.columns if (
        col in non_cluster_columns_to_save) | ('cluster' in col)]

    if for_classification:
        if 'dir_from_cur_ff_to_nxt_ff' in plan_x.columns:
            cluster_columns_to_save.append('dir_from_cur_ff_to_nxt_ff')

    plan_x = plan_x[cluster_columns_to_save].copy()
    return plan_x


def process_plan_x_to_predict_ff_info(plan_x, plan_y):

    non_cluster_columns_to_save = ['distance_between_stop_and_arena_edge'] + ff_at_ref_columns \
        + monkey_info_columns + more_monkey_info_columns

    # delete 'curv_range' and 'curv_iqr' from non_cluster_columns_to_save
    non_cluster_columns_to_save.remove('curv_range')
    non_cluster_columns_to_save.remove('curv_iqr')

    cluster_columns_to_save = [col for col in plan_x.columns if (
        col in non_cluster_columns_to_save) | ('cluster' in col)]
    # delete any column that contains 'nxt'
    cluster_columns_to_save = [
        col for col in cluster_columns_to_save if 'nxt' not in col]

    plan_x = plan_x[cluster_columns_to_save].copy()

    # add columns
    columns_from_plan_y_to_add = ['angle_from_cur_ff_to_stop',
                                  'diff_in_d_heading_of_traj_from_null',
                                  'curv_of_traj_before_stop',
                                  'dir_from_cur_ff_to_stop']

    plan_x[columns_from_plan_y_to_add] = plan_y[columns_from_plan_y_to_add].values.copy()

    plan_x = plan_x[cluster_columns_to_save].copy()

    return plan_x


def delete_monkey_info_in_plan_x(plan_x):
    columns_to_drop = monkey_info_columns + more_monkey_info_columns
    # delete 'curv_range'
    columns_to_drop.remove('curv_range')
    plan_x = plan_x.drop(columns=columns_to_drop, errors='ignore')

    # columns_to_preserve = [col for col in plan_x.columns if ('cluster' in col) | ('ref' in col)]
    # plan_x = plan_x[columns_to_preserve].copy()
    return plan_x


def make_plan_xy_test_and_plan_xy_ctrl(plan_x_tc, plan_y_tc):
    # concat plan_x_tc and plan_y_tc but drop duplicated columns
    plan_xy = pd.concat([plan_x_tc, plan_y_tc], axis=1)
    # drop duplicated columns
    plan_xy = plan_xy.loc[:, ~plan_xy.columns.duplicated()]

    plan_xy_test = plan_xy[plan_xy['whether_test']
                           == 1].reset_index(drop=True).copy()
    plan_xy_ctrl = plan_xy[plan_xy['whether_test']
                           == 0].reset_index(drop=True).copy()
    return plan_xy_test, plan_xy_ctrl


def quickly_process_plan_xy_test_and_ctrl(plan_xy_test, plan_xy_ctrl, column_for_split, whether_filter_info, finalized_params):
    test_and_ctrl_df = pd.concat([plan_xy_test, plan_xy_ctrl], axis=0)
    ctrl_df = test_and_ctrl_df[test_and_ctrl_df[column_for_split].isnull()].copy(
    )
    test_df = test_and_ctrl_df[~test_and_ctrl_df[column_for_split].isnull()].copy(
    )

    if whether_filter_info:
        test_df, ctrl_df = test_vs_control_utils.filter_both_df(
            test_df, ctrl_df, **finalized_params)

    return test_df, ctrl_df


def make_plan_x_df(stops_near_ff_df, heading_info_df, both_ff_at_ref_df, ff_dataframe, monkey_information, ff_real_position_sorted,
                   stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150, ff_radius=10,
                   list_of_cur_ff_cluster_radius=[100, 200, 300],
                   list_of_nxt_ff_cluster_radius=[100, 200, 300],
                   use_speed_data=False, use_eye_data=False,
                   guarantee_cur_ff_info_for_cluster=False,
                   guarantee_nxt_ff_info_for_cluster=False,
                   columns_not_to_include=[],
                   flash_or_vis='vis',
                   ):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()
    cluster_df = make_cluster_df_as_part_of_plan_factors(stops_near_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted,
                                                         stop_period_duration=stop_period_duration, ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, ff_radius=ff_radius,
                                                         list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                         columns_not_to_include=columns_not_to_include,
                                                         guarantee_cur_ff_info_for_cluster=guarantee_cur_ff_info_for_cluster,
                                                         guarantee_nxt_ff_info_for_cluster=guarantee_nxt_ff_info_for_cluster,
                                                         flash_or_vis=flash_or_vis,
                                                         )
    # use stop_period to get stop_point_index
    other_factors = get_other_factors(stops_near_ff_df)
    other_factors.reset_index(drop=True, inplace=True)

    plan_x_df = cluster_df.merge(
        other_factors, on='stop_point_index', how='left')
    plan_x_df = plan_x_df.merge(
        both_ff_at_ref_df, on='stop_point_index', how='left').reset_index(drop=True)

    # nxt_ff_last_seen_info = get_nxt_ff_last_seen_info_before_next_stop(nxt_ff_df2, ff_dataframe_visible, monkey_information,
    #                                                                                       stops_near_ff_df, ff_real_position_sorted)
    # plan_x_df = pd.concat([plan_x_df, nxt_ff_last_seen_info])

    if use_speed_data or use_eye_data:
        df_for_stat_extended = get_df_for_stat_extended_for_eye_info(
            stops_near_ff_df, monkey_information)

    if use_speed_data:
        monkey_speed_stat_df = find_stat_of_columns_after_groupby(df_for_stat_extended,
                                                                  groupby_column='stop_point_index',
                                                                  stat_columns=['monkey_speed', 'monkey_dw'])
        columns_to_preserve = [column for column in monkey_speed_stat_df if ('iqr' in column) | (
            'range' in column) | ('std' in column) | (column == 'stop_point_index')]
        monkey_speed_stat_df = monkey_speed_stat_df[columns_to_preserve].copy()

        plan_x_df = plan_x_df.merge(
            monkey_speed_stat_df, on='stop_point_index', how='left')

    if use_eye_data:
        eye_stat_df = find_stat_of_columns_after_groupby(df_for_stat_extended,
                                                         groupby_column='stop_point_index',
                                                         stat_columns=['LDy', 'LDz', 'RDy', 'RDz'])
        columns_to_preserve = [column for column in eye_stat_df if ('iqr' in column) | (
            'range' in column) | ('std' in column) | (column == 'stop_point_index')]
        eye_stat_df = eye_stat_df[columns_to_preserve].copy()

        eye_perc_df = get_eye_perc_df(df_for_stat_extended)

        plan_x_df = plan_x_df.merge(eye_stat_df, on='stop_point_index', how='left').merge(
            eye_perc_df, on='stop_point_index', how='left').reset_index(drop=True)

    # only keep the rows with stop_point_index that are in heading_info_df
    plan_x_df = plan_x_df[plan_x_df['stop_point_index'].isin(
        heading_info_df['stop_point_index'])].copy()
    plan_x_df = plan_x_df.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_x_df
