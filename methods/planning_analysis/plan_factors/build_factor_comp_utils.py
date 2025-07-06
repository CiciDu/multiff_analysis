from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils, build_factor_comp_utils, build_factor_comp
from data_wrangling import specific_utils

import numpy as np
import pandas as pd
import math

def _add_stat_columns_to_df(stat_df, df, stat_columns, groupby_column):
    '''
    Add statistical columns of a variable or multiple variables to a dataframe through merging.

    Args:
        stat_df (pd.DataFrame): DataFrame containing statistical information.
        df (pd.DataFrame): DataFrame to which statistical columns will be added.
        stat_columns (list): List of statistical columns to add.
    '''
    for prefix in stat_columns:
        columns_to_add = [f'{prefix}_mean', f'{prefix}_std', f'{prefix}_min', f'{prefix}_Q1', f'{prefix}_median', f'{prefix}_Q3',
                          f'{prefix}_max', f'{prefix}_iqr', f'{prefix}_range']
        # drop the columns if already exist
        df = df.drop(columns=columns_to_add, errors='ignore')
        df = df.merge(
            stat_df[columns_to_add + [groupby_column]], on=groupby_column, how='left')
    return df


def _take_out_info_of_all_segments(ori_df, all_start_time, all_end_time, all_segment_id, group_id='stop_point_index'):
    # This function iterates through all segments and take out all rows that fall within specified time intervals.
    # There can potentially be overlapped rows if more than one interval contains that time point.
    extended_cum_indices = []
    extended_group_id = []
    for i in range(len(all_start_time)):
        start_time = all_start_time[i]
        end_time = all_end_time[i]
        current_group_id = all_segment_id[i]
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


def _prepare_data_of_segments_based_on_stop_point_index(stops_near_ff_df, monkey_information, 
                                           groupby_column='stop_point_index',
                                           start_time_column='beginning_time',
                                           end_time_column='stop_time'):
    df = stops_near_ff_df.copy()
    

    data_of_segments = _take_out_info_of_all_segments(monkey_information,
                                                      all_start_time=df[start_time_column].values,
                                                      all_end_time=df[end_time_column].values,
                                                      all_segment_id=df[groupby_column].values,
                                                      group_id='stop_point_index')

    data_of_segments = data_of_segments.merge(stops_near_ff_df[['stop_point_index', 'cur_ff_x', 'cur_ff_y', 'nxt_ff_x', 'nxt_ff_y']],
                                              on='stop_point_index', how='left')

    data_of_segments['cur_ff_angle'] = specific_utils.calculate_angles_to_ff_centers(data_of_segments['cur_ff_x'], data_of_segments['cur_ff_y'], data_of_segments['monkey_x'],
                                                                                     data_of_segments['monkey_y'], data_of_segments['monkey_angle'])

    data_of_segments['nxt_ff_angle'] = specific_utils.calculate_angles_to_ff_centers(data_of_segments['nxt_ff_x'], data_of_segments['nxt_ff_y'], data_of_segments['monkey_x'],
                                                                                     data_of_segments['monkey_y'], data_of_segments['monkey_angle'])
    return data_of_segments


def _get_quartile_data_in_a_row(monkey_sub, columns, suffix=''):
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


def _get_point_index_of_nxt_ff_last_seen_before_next_stop(ff_dataframe_visible, stops_near_ff_df):
    all_segment_identifiers, all_stop_time, all_next_stop_time, all_ff_index = stops_near_ff_df['stop_point_index'].values, stops_near_ff_df[
        'stop_time'].values, stops_near_ff_df['next_stop_time'].values, stops_near_ff_df['nxt_ff_index'].values
    # time_between_stop_and_next_stop = stops_near_ff_df['next_stop_time'].values - stops_near_ff_df['stop_time'].values
    ff_last_seen_info = nxt_ff_utils.find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_segment_identifiers, all_stop_time-2.5,
                                                                                                              all_next_stop_time, ff_dataframe_visible, first_or_last='last'
                                                                                                              )
    last_seen_point_index = ff_last_seen_info['point_index'].values
    # replace values on na in last_seen_point_index with 0
    last_seen_point_index[np.isnan(last_seen_point_index)] = 0

    return last_seen_point_index


def _find_summary_stats_of_each_segment(data_of_segments,
                                        groupby_column='stop_point_index',
                                        stat_columns=['curv_of_traj'],
                                        stat_column_prefixes=None):

    if stat_column_prefixes is None:
        stat_column_prefixes = stat_columns
    stat_column_prefixes = [prefix + '_' for prefix in stat_column_prefixes]

    # make a dict of column to prefix
    column_prefix_map = {column: prefix for column,
                         prefix in zip(stat_columns, stat_column_prefixes)}

    # Note: The describe() method after using groupby generates a df with multi-level column names,
    # where the first level is the original column name (e.g., 'curv_of_traj') and the second level is the statistical measure (e.g., 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max').

    # Group by the specified column and calculate descriptive statistics
    stat_df = data_of_segments.groupby(groupby_column)[stat_columns].describe()

    # Rename the percentile columns for better readability
    stat_df.rename(
        columns={'25%': 'Q1', '50%': 'median', '75%': 'Q3'}, inplace=True)

    # Flatten the multi-level column names and add prefixes
    stat_df.columns = [
        f"{column_prefix_map[column[0]]}{column[1]}" for column in stat_df.columns]

    for prefix in stat_column_prefixes:
        stat_df[prefix + 'range'] = stat_df[prefix + 'max'] - \
            stat_df[prefix + 'min']
        stat_df[prefix + 'iqr'] = stat_df[prefix + 'Q3'] - \
            stat_df[prefix + 'Q1']
        stat_df.drop(columns=prefix + 'count', inplace=True)
    stat_df.reset_index(drop=False, inplace=True)
    return stat_df

def _get_monkey_speed_stat_df(data_of_segments):
    monkey_speed_stat_df = build_factor_comp_utils._find_summary_stats_of_each_segment(data_of_segments,
                                                                                       groupby_column='stop_point_index',
                                                                                       stat_columns=['monkey_speed', 'monkey_dw'])
    columns_to_preserve = [column for column in monkey_speed_stat_df if ('iqr' in column) | (
        'range' in column) | ('std' in column) | (column == 'stop_point_index')]
    monkey_speed_stat_df = monkey_speed_stat_df[columns_to_preserve].copy()
    return monkey_speed_stat_df


def _get_monkey_speed_stat_df_bbas(stops_near_ff_df, monkey_information):
    data_of_segments = build_factor_comp_utils._prepare_data_of_segments_based_on_stop_point_index(
        stops_near_ff_df, monkey_information)
    monkey_speed_stat_df_bbas = _get_monkey_speed_stat_df(data_of_segments)
    return monkey_speed_stat_df_bbas


def _get_monkey_speed_stat_df_bsans(stops_near_ff_df, monkey_information):
    data_of_segments = build_factor_comp_utils._prepare_data_of_segments_based_on_stop_point_index(
        stops_near_ff_df, monkey_information, start_time_column='stop_time', end_time_column='next_stop_time')
    monkey_speed_stat_df_bsans = _get_monkey_speed_stat_df(data_of_segments)
    return monkey_speed_stat_df_bsans


def _get_eye_stats(data_of_segments, ff_to_include=['cur_ff', 'nxt_ff']):
    eye_stat_df = build_factor_comp_utils._find_summary_stats_of_each_segment(data_of_segments,
                                                                              groupby_column='stop_point_index',
                                                                              stat_columns=['LDy', 'LDz', 'RDy', 'RDz'])
    columns_to_preserve = [column for column in eye_stat_df if ('iqr' in column) | (
        'range' in column) | ('std' in column) | (column == 'stop_point_index')]
    eye_stat_df = eye_stat_df[columns_to_preserve].copy()
    eye_toward_ff_time_perc_df = build_factor_comp.get_eye_toward_ff_time_perc_df(
        data_of_segments, ff_to_include=ff_to_include)
    return eye_stat_df, eye_toward_ff_time_perc_df


def _get_eye_stats_bbas(stops_near_ff_df, monkey_information):
    data_of_segments = build_factor_comp_utils._prepare_data_of_segments_based_on_stop_point_index(
        stops_near_ff_df, monkey_information)
    eye_stat_df_bbas, eye_toward_ff_time_perc_df_bbas = _get_eye_stats(
        data_of_segments)
    return eye_stat_df_bbas, eye_toward_ff_time_perc_df_bbas


def _get_eye_stats_bsans(stops_near_ff_df, monkey_information):
    data_of_segments = build_factor_comp_utils._prepare_data_of_segments_based_on_stop_point_index(
        stops_near_ff_df, monkey_information, start_time_column='stop_time', end_time_column='next_stop_time')
    eye_stat_df_bsans, eye_toward_ff_time_perc_df_bsans = _get_eye_stats(
        data_of_segments, ff_to_include=['nxt_ff'])

    eye_stat_df_bsans.columns = [
        col + '_bsans' if col != 'stop_point_index' else col for col in eye_stat_df_bsans.columns]
    eye_toward_ff_time_perc_df_bsans.columns = [
        col + '_bsans' if col != 'stop_point_index' else col for col in eye_toward_ff_time_perc_df_bsans.columns]

    return eye_stat_df_bsans, eye_toward_ff_time_perc_df_bsans  


def _find_clusters_in_ff_info_in_all_stop_periods(ff_info_in_all_stop_periods,
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
