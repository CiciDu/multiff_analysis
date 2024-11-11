from decision_making_analysis.decision_making import decision_making_utils
from data_wrangling import basic_func
from planning_analysis.show_planning import alt_ff_utils
from null_behaviors import show_null_trajectory
from pattern_discovery import cluster_analysis
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
from math import pi
import warnings


def extract_key_info_from_data_item_for_stops_near_ff_class(data_item):
    data_item_info = {'monkey_information': data_item.monkey_information,
                        'ff_dataframe': data_item.ff_dataframe,
                        'ff_caught_T_sorted': data_item.ff_caught_T_sorted,
                        'ff_real_position_sorted': data_item.ff_real_position_sorted,
                        'ff_life_sorted': data_item.ff_life_sorted,
                        'PlotTrials_args': data_item.PlotTrials_args,
                        'monkey_name': data_item.monkey_name,
                        'data_name': data_item.data_name}
    return data_item_info


def find_captured_ff_info_for_making_stops_near_ff_df(monkey_information, ff_dataframe_visible, ff_caught_T_sorted, ff_real_position_sorted, stop_period_duration=2, max_diff_between_caught_time_and_stop_time=0.2):
    
    all_closest_point_to_capture_df = alt_ff_utils.get_closest_stop_time_to_all_capture_time(ff_caught_T_sorted, monkey_information, stop_ff_index_array=np.arange(len(ff_caught_T_sorted)),
                                                                                             ff_real_position_sorted=ff_real_position_sorted, drop_rows_where_stop_is_not_inside_reward_boundary=True)
    print('finding captured_ff_info...')
    captured_ff_info = alt_ff_utils.get_all_captured_ff_first_seen_and_last_seen_info(all_closest_point_to_capture_df, stop_period_duration,
                                                                                            ff_dataframe_visible, monkey_information, drop_na=True)

    # drop rows in captured_ff_info where the stop_time and capture_time are more than n seconds apart.
    captured_ff_info = drop_rows_in_where_stop_time_and_capture_time_is_too_far_apart(captured_ff_info, all_closest_point_to_capture_df, max_diff_between_caught_time_and_stop_time=max_diff_between_caught_time_and_stop_time)

    # eliminate boundary cases
    selected_point_index = captured_ff_info.stop_point_index.values
    time_of_stops = monkey_information.loc[selected_point_index, 'monkey_t'].values
    crossing_boundary_time = monkey_information.loc[monkey_information['crossing_boundary']==1, 'monkey_t'].values
    CB_indices, non_CB_indices, left_input_time = decision_making_utils.find_time_points_that_are_within_n_seconds_after_crossing_boundary(time_of_stops, crossing_boundary_time, 
                                                                                                                                           n_seconds_before_crossing_boundary=0.2, 
                                                                                                                                           n_seconds_after_crossing_boundary=stop_period_duration + 0.2)
    selected_point_index = selected_point_index[non_CB_indices]
    print(f'{len(CB_indices)} rows out of {len(captured_ff_info)} rows in captured_ff_info were dropped because they are within n seconds before or after crossing boundary.')

    if 'stop_point_index' not in captured_ff_info.columns:
        captured_ff_info['stop_point_index'] = captured_ff_info['point_index']
    captured_ff_info = captured_ff_info[captured_ff_info['stop_point_index'].isin(selected_point_index)].copy()

    return captured_ff_info


def drop_rows_in_where_stop_time_and_capture_time_is_too_far_apart(captured_ff_info, all_closest_point_to_capture_df, max_diff_between_caught_time_and_stop_time=0.2):
    all_closest_point_to_capture_df = all_closest_point_to_capture_df.rename(columns={'stop_ff_index': 'ff_index'})
    captured_ff_info = captured_ff_info.merge(all_closest_point_to_capture_df[['ff_index', 'caught_time', 'diff_from_caught_time']], on='ff_index', how='left')
    # if there's any NA, raise error
    if len(captured_ff_info[captured_ff_info['diff_from_caught_time'].isna()]) > 0:
        raise ValueError('There are NA values in diff_from_caught_time. This should not happen.')
    if captured_ff_info['diff_from_caught_time'].abs().max() >= max_diff_between_caught_time_and_stop_time:
        max_diff = captured_ff_info['diff_from_caught_time'].abs().max()
        num_rows_exceeding = len(captured_ff_info[captured_ff_info['diff_from_caught_time'].abs() >= max_diff_between_caught_time_and_stop_time])
        # calculate the percentege of rows where all_closest_point_to_capture_df['diff_from_caught_time'] is positive
        percentage_of_rows_where_diff_is_positive = len(all_closest_point_to_capture_df[all_closest_point_to_capture_df['diff_from_caught_time'] > 0]) / len(all_closest_point_to_capture_df)
        warning_message = f'There is a problem with the closest point to capture time in that the difference between the time and the caught time is greater than {max_diff_between_caught_time_and_stop_time} seconds.' + \
                            f'The maximum difference is {max_diff} seconds. Additionally, there are {num_rows_exceeding} rows that exceed this limit. ' + \
                            f'Furthermore, the percentage of rows where the difference is positive is {percentage_of_rows_where_diff_is_positive * 100} %'
        print(warning_message)  
        print(f'{num_rows_exceeding} rows out of {len(captured_ff_info)} rows in captured_ff_info have been removed because the diff_from_caught_time is greater than {max_diff_between_caught_time_and_stop_time} seconds. The maximum diff_from_caught_time is {max_diff} seconds.')
        captured_ff_info = captured_ff_info[captured_ff_info['diff_from_caught_time'].abs() < max_diff_between_caught_time_and_stop_time].copy()
    return captured_ff_info


def make_shared_stops_near_ff_df(monkey_information, ff_dataframe_visible, ff_real_position_sorted, ff_caught_T_sorted,
                                 ff_flash_sorted, ff_life_sorted,
                                 remove_cases_where_monkey_too_close_to_edge=False, 
                                 stop_period_duration=2,
                                 min_time_between_stop_and_alt_ff_caught_time=0.1,
                                 min_distance_between_stop_and_alt_ff=25,
                                 max_distance_between_stop_and_alt_ff=500,
                                 min_time_between_stop_ff_first_seen_time_and_stop=0.2,
                                 ):
    print('Making shared_stops_near_ff_df...')

    shared_stops_near_ff_df = find_captured_ff_info_for_making_stops_near_ff_df(monkey_information, ff_dataframe_visible, ff_caught_T_sorted, ff_real_position_sorted, stop_period_duration=stop_period_duration)

    shared_stops_near_ff_df[['stop_x', 'stop_y', 'monkey_angle', 'stop_time', 'stop_cum_distance']] = monkey_information.loc[shared_stops_near_ff_df['stop_point_index'], ['monkey_x', 'monkey_y', 'monkey_angle', 'monkey_t', 'cum_distance']].values

    shared_stops_near_ff_df = alt_ff_utils.rename_first_and_last_seen_info_columns(shared_stops_near_ff_df, prefix='STOP_')

    shared_stops_near_ff_df.rename(columns={'ff_index': 'stop_ff_index',
                                            'monkey_angle': 'stop_monkey_angle',
                                            }, inplace=True)
    shared_stops_near_ff_df['stop_ff_capture_time'] = ff_caught_T_sorted[shared_stops_near_ff_df['stop_ff_index'].values]
    
    # add alt ff info
    all_alt_ff_df = alt_ff_utils.get_all_alt_ff_df_from_ff_dataframe(shared_stops_near_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, monkey_information,
                                                        min_time_between_stop_and_alt_ff_caught_time=min_time_between_stop_and_alt_ff_caught_time,
                                                        min_distance_between_stop_and_alt_ff=min_distance_between_stop_and_alt_ff,
                                                        max_distance_between_stop_and_alt_ff=max_distance_between_stop_and_alt_ff)

    original_length = len(shared_stops_near_ff_df)
    shared_stops_near_ff_df = shared_stops_near_ff_df.merge(all_alt_ff_df.drop(columns=['stop_point_index']), on='stop_ff_index', how='inner')
    print(f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} rows in shared_stops_near_ff_df have been removed because the alt_ff was not found from ff_dataframe.')
    shared_stops_near_ff_df['next_stop_cum_distance'] = monkey_information.loc[shared_stops_near_ff_df['next_stop_point_index'], 'cum_distance'].values
    shared_stops_near_ff_df['cum_distance_between_two_stops'] = shared_stops_near_ff_df['next_stop_cum_distance'] - shared_stops_near_ff_df['stop_cum_distance']
    if len(shared_stops_near_ff_df['stop_ff_index'].unique()) != len(shared_stops_near_ff_df):
        warnings.warn('There are duplicated stop_ff_index in shared_stops_near_ff_df. This should not happen.')


    shared_stops_near_ff_df, all_alt_ff_df = process_instances_where_monkey_is_too_close_to_edge(shared_stops_near_ff_df, all_alt_ff_df, monkey_information, remove_cases_where_monkey_too_close_to_edge=remove_cases_where_monkey_too_close_to_edge)
    shared_stops_near_ff_df = add_monkey_info_before_stop(monkey_information, shared_stops_near_ff_df)
    shared_stops_near_ff_df['stop_ff_x'], shared_stops_near_ff_df['stop_ff_y'] = ff_real_position_sorted[shared_stops_near_ff_df['stop_ff_index'].values].T
    shared_stops_near_ff_df['d_from_stop_ff_to_stop'] = np.linalg.norm(shared_stops_near_ff_df[['stop_x', 'stop_y']].values - shared_stops_near_ff_df[['stop_ff_x', 'stop_ff_y']].values, axis=1)

    shared_stops_near_ff_df = alt_ff_utils.add_if_alt_ff_and_alt_ff_cluster_flash_bbas(shared_stops_near_ff_df, ff_real_position_sorted, 
                                                                                    ff_flash_sorted, ff_life_sorted, stop_period_duration=stop_period_duration)

    shared_stops_near_ff_df = alt_ff_utils.add_if_alt_ff_and_alt_ff_cluster_flash_bsans(shared_stops_near_ff_df, ff_real_position_sorted, 
                                                                                    ff_flash_sorted, ff_life_sorted)
    
    shared_stops_near_ff_df = shared_stops_near_ff_df.sort_values(by='stop_point_index')

    shared_stops_near_ff_df = add_stop_ff_cluster_50_size(shared_stops_near_ff_df, ff_real_position_sorted, ff_life_sorted)

    len_before = len(shared_stops_near_ff_df)
    shared_stops_near_ff_df = shared_stops_near_ff_df[shared_stops_near_ff_df['stop_time'] - shared_stops_near_ff_df['STOP_time_ff_first_seen_bbas'] >= min_time_between_stop_ff_first_seen_time_and_stop].copy().reset_index(drop=True)
    print(f'{len_before - len(shared_stops_near_ff_df)} rows out of {len_before} rows in shared_stops_near_ff_df have been removed because the time between stop_time and STOP_time_ff_first_seen_bbas is less than {min_time_between_stop_ff_first_seen_time_and_stop} seconds.')

    #shared_stops_near_ff_df = check_for_different_ref_points_to_remove_rows_with_big_stop_or_alt_ff_angle_boundary(shared_stops_near_ff_df, monkey_information, ff_real_position_sorted)
    return shared_stops_near_ff_df, all_alt_ff_df


def process_instances_where_monkey_is_too_close_to_edge(shared_stops_near_ff_df, all_alt_ff_df, monkey_information, remove_cases_where_monkey_too_close_to_edge=False):
    if remove_cases_where_monkey_too_close_to_edge is True:
        original_length = len(shared_stops_near_ff_df)
        shared_stops_near_ff_df = remove_cases_where_monkey_too_close_to_edge_func(shared_stops_near_ff_df, monkey_information)
        print(f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} rows in shared_stops_near_ff_df have been removed because the monkey was too close to the edge.')
        all_alt_ff_df = all_alt_ff_df[all_alt_ff_df['stop_point_index'].isin(shared_stops_near_ff_df['stop_point_index'])].copy().sort_values(by='stop_point_index').reset_index(drop=True)
    else:
        # if there's crossing boundary between the stop and the next stop, we shall remove the row
        original_length = len(shared_stops_near_ff_df)
        crossing_boundary_points = monkey_information.loc[monkey_information['crossing_boundary']==1, 'point_index'].values
        shared_stops_near_ff_df['crossing_boundary'] = np.diff(np.searchsorted(crossing_boundary_points, shared_stops_near_ff_df[['stop_point_index', 'next_stop_point_index']].values), axis=1).flatten()
        shared_stops_near_ff_df = shared_stops_near_ff_df[shared_stops_near_ff_df['crossing_boundary']==0].copy().reset_index(drop=True)
        all_alt_ff_df = all_alt_ff_df[all_alt_ff_df['stop_point_index'].isin(shared_stops_near_ff_df['stop_point_index'])].copy().sort_values(by='stop_point_index').reset_index(drop=True)
        print(f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} rows in shared_stops_near_ff_df have been removed because the monkey has crossed boundary between two stops.')
    return shared_stops_near_ff_df, all_alt_ff_df


def add_stop_ff_cluster_50_size(shared_stops_near_ff_df, ff_real_position_sorted, ff_life_sorted, empty_cluster_ok=False):
    ff_positions = shared_stops_near_ff_df[['stop_ff_x', 'stop_ff_y']].values
    if 'stop_ff_capture_time' in shared_stops_near_ff_df.columns:
        array_of_end_time_of_evaluation = shared_stops_near_ff_df['stop_ff_capture_time'].values
    else:
        array_of_end_time_of_evaluation = shared_stops_near_ff_df['stop_time'].values
    ff_indices_of_each_cluster = cluster_analysis.find_ff_clusters(ff_positions, ff_real_position_sorted, shared_stops_near_ff_df['beginning_time'].values,
                                                                    array_of_end_time_of_evaluation, ff_life_sorted, max_distance=50, empty_cluster_ok=empty_cluster_ok)
    all_cluster_size = np.array([len(array) for array in ff_indices_of_each_cluster])
    shared_stops_near_ff_df['stop_ff_cluster_50_size'] = all_cluster_size   
    return shared_stops_near_ff_df


def check_for_different_ref_points_to_remove_rows_with_big_stop_or_alt_ff_angle_boundary(shared_stops_near_ff_df, monkey_information, ff_real_position_sorted):
    ref_point_params_based_on_mode = {'time after stop ff visible': [0, 0.1], 'distance': [-150, -100, -50]}
    variations_list = basic_func.init_variations_list_func(ref_point_params_based_on_mode)
    print('===================== Clearing shared_stops_near_ff_df based on ref_point_mode and ref_point_value =====================')
    for index, row in variations_list.iterrows():
        stops_near_ff_df, alt_ff_df, stop_ff_df = alt_ff_utils.get_alt_ff_df_and_stop_ff_df(shared_stops_near_ff_df)
        ref_point_mode = row['ref_point_mode']
        ref_point_value = row['ref_point_value']
        alt_ff_df2 = find_ff_info_based_on_ref_point(alt_ff_df, monkey_information, ff_real_position_sorted,
                                                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                    point_index_stop_ff_first_seen=stop_ff_df['point_index_ff_first_seen'].values)
        stop_ff_df2 = find_ff_info_based_on_ref_point(stop_ff_df, monkey_information, ff_real_position_sorted,
                                                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
        # find rows where ff_angle in alt_ff_df2 or stop_ff_df2 that have a bigger magnitude than 45 degrees
        rows_to_delete_from_alt_ff = alt_ff_df2[alt_ff_df2['ff_angle'].abs() > math.pi/4].index
        rows_to_delete_from_stop_ff = stop_ff_df2[stop_ff_df2['ff_angle'].abs() > math.pi/4].index
        rows_to_delete = np.concatenate([rows_to_delete_from_alt_ff, rows_to_delete_from_stop_ff])
        print(f'ref_point_mode: {ref_point_mode}, ref_point_value: {ref_point_value}')
        print(f'{len(rows_to_delete)} rows out of {len(shared_stops_near_ff_df)} have been removed because the ff_angle is greater than 45 degrees. ' +
              f'This is {round(len(rows_to_delete)/len(shared_stops_near_ff_df)*100, 2)} percent of row. Among them, {len(rows_to_delete_from_alt_ff)} rows are from alt_ff_df and {len(rows_to_delete_from_stop_ff)} rows are from stop_ff_df.')
        shared_stops_near_ff_df = shared_stops_near_ff_df[~shared_stops_near_ff_df.index.isin(rows_to_delete)].copy().reset_index(drop=True)
    print('===================== Finished shared_stops_near_ff_df based on ref_point_mode and ref_point_value =====================')
    return shared_stops_near_ff_df


def check_for_unique_stop_point_index_or_ff_index(shared_stops_near_ff_df):
    shared_stops_near_ff_df = shared_stops_near_ff_df.copy()
    shared_stops_near_ff_df.drop_duplicates(subset=[['stop_point_index', 'stop_ff_index', 'alt_ff_index']], inplace=True)
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['stop_point_index'].unique()):
        raise ValueError('There are duplicated stop_point_index in shared_stops_near_ff_df for the same stop_ff_index or alt_ff_index')
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['stop_ff_index'].unique()):
        raise ValueError('There are duplicated stop_ff_index in shared_stops_near_ff_df for the same stop_point_index or alt_ff_index')
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['alt_ff_index'].unique()):
        raise ValueError('There are duplicated stop_ff_index in shared_stops_near_ff_df for the same stop_point_index or stop_ff_index')
    

def remove_cases_where_monkey_too_close_to_edge_func(stops_near_ff_df, monkey_information, distance_to_area_edge=50):
    monkey_information['monkey_r'] = np.sqrt(monkey_information['monkey_x']**2 + monkey_information['monkey_y']**2)
    # eliminate cases where the monkey has been within 50cm of the area edge
    edge_points = monkey_information.loc[monkey_information['monkey_r'] > 1000 - distance_to_area_edge, 'point_index'].values
    stops_near_ff_df['edge'] = np.diff(np.searchsorted(edge_points, stops_near_ff_df[['stop_point_index', 'next_stop_point_index']].values), axis=1).flatten()
    stops_near_ff_df = stops_near_ff_df[stops_near_ff_df['edge']==0].copy().reset_index(drop=True)
    return stops_near_ff_df


def calculate_info_based_on_monkey_angles(stops_near_ff_df, monkey_angles):
    info_df = stops_near_ff_df[['stop_point_index', 'd_from_stop_ff_to_stop', 'd_from_stop_ff_to_alt_ff']].copy()
    info_df['monkey_angles'] = monkey_angles
    info_df['angle_from_stop_ff_to_stop'] = basic_func.calculate_angles_to_ff_centers(ff_x=stops_near_ff_df['stop_x'].values, ff_y=stops_near_ff_df['stop_y'], \
                                                                                                  mx=stops_near_ff_df['stop_ff_x'].values, my=stops_near_ff_df['stop_ff_y'], m_angle=monkey_angles)
    info_df['angle_from_stop_ff_to_alt_ff'] = basic_func.calculate_angles_to_ff_centers(ff_x=stops_near_ff_df['alt_ff_x'].values, ff_y=stops_near_ff_df['alt_ff_y'], \
                                                                                                              mx=stops_near_ff_df['stop_ff_x'].values, my=stops_near_ff_df['stop_ff_y'], m_angle=monkey_angles)
    info_df['dir_from_stop_ff_to_stop'] = np.sign(info_df['angle_from_stop_ff_to_stop'])
    info_df['dir_from_stop_ff_to_alt_ff'] = np.sign(info_df['angle_from_stop_ff_to_alt_ff'])
    return info_df


def add_monkey_info_before_stop(monkey_information, stops_near_ff_df):
    # add the info about the monkey before the stop; the time is the most recent time when the speed was greater than 20 cm/s
    stops_near_ff_df = stops_near_ff_df.copy()
    stops_near_ff_df['stop_counter'] = np.arange(len(stops_near_ff_df))

    # we take out all the potential information from monkey_information first
    monkey_info_sub = monkey_information[monkey_information['monkey_speed'] > 20].copy()
    monkey_info_sub['closest_future_stop'] = np.searchsorted(stops_near_ff_df['stop_point_index'].values, monkey_info_sub.index.values)
    monkey_info_sub.sort_values(by='monkey_t', inplace=True)
    # then, for each stop, we only keep the most recent monkey info before the stop
    monkey_info_sub = monkey_info_sub.groupby('closest_future_stop').tail(1).reset_index(drop=False)
    monkey_info_sub.rename(columns={'closest_future_stop': 'stop_counter',
                                    'monkey_angles': 'monkey_angle_before_stop',
                                    'point_index': 'point_index_before_stop'}, inplace=True)

    # then, we furnish stops_near_ff_df with the monkey info before the stop by merging
    stops_near_ff_df = pd.merge(stops_near_ff_df, monkey_info_sub[['stop_counter', 'point_index_before_stop', 'monkey_angle_before_stop']], on='stop_counter', how='left')
    # forward fill the nan values; this is necessary because sometimes we don't have any point with speed > 20 between two stops, in which case the two stops share the same monkey info before stop
    stops_near_ff_df['monkey_angle_before_stop'] = stops_near_ff_df['monkey_angle_before_stop'].ffill()
    stops_near_ff_df['point_index_before_stop'] = stops_near_ff_df['point_index_before_stop'].ffill()
    # drop rows with na in 'point_index_before_stop'
    stops_near_ff_df.dropna(subset=['point_index_before_stop'], inplace=True)
    stops_near_ff_df['point_index_before_stop'] = stops_near_ff_df['point_index_before_stop'].astype(int)
    stops_near_ff_df['distance_before_stop'] = monkey_information.loc[stops_near_ff_df['point_index_before_stop'].values, 'cum_distance'].values
    stops_near_ff_df['time_before_stop'] = monkey_information.loc[stops_near_ff_df['point_index_before_stop'].values, 'monkey_t'].values
    stops_near_ff_df.drop(columns=['stop_counter'], inplace=True)

    return stops_near_ff_df


def modify_position_of_ff_with_big_angle_for_finding_null_arc(ff_df, remove_i_o_modify_rows_with_big_ff_angles=True, verbose=True):
    ff_df = ff_df.copy()
    original_ff_df = ff_df.copy()

    if remove_i_o_modify_rows_with_big_ff_angles is False:
        print('Note that in the function modify_position_of_ff_with_big_angle_for_finding_null_arc, even if remove_i_o_modify_rows_with_big_ff_angles is False, there might still be rows eiminated. \
              The function is suggested not to be used unless for calculating curvature.')

    # remove rows where monkey is within the ff (ff_distance <= 25)
    monkey_not_within_ff = np.where(ff_df['ff_distance'].values > 25)[0]
    ff_df = ff_df.iloc[monkey_not_within_ff].copy()
    indices_of_kept_rows = monkey_not_within_ff

    # remove rows where ff_y_relative is negative
    ff_x_relative, ff_y_relative = show_null_trajectory.find_relative_xy_positions(ff_df.ff_x.values, ff_df.ff_y.values, ff_df.monkey_x.values, ff_df.monkey_y.values, ff_df.monkey_angle.values)
    if np.any(ff_y_relative < 0):
        percent_negative = np.sum(ff_y_relative < 0) / len(ff_y_relative)
        if verbose:
            print('Warning: {}% of ff_y_relative are negative. This should not happen. These rows will be eliminated'.format(round(percent_negative*100, 3)))
        indices_of_kept_rows = indices_of_kept_rows[np.where(ff_y_relative > 0)[0]]
        ff_df = original_ff_df.iloc[indices_of_kept_rows].copy()
        ff_x_relative, ff_y_relative = show_null_trajectory.find_relative_xy_positions(ff_df.ff_x.values, ff_df.ff_y.values, ff_df.monkey_x.values, ff_df.monkey_y.values, ff_df.monkey_angle.values)
        #raise ValueError('ff_y_relative should be positive. Maybe I should increase min_reaction_time.')

    # deal with rows where ff_x_relative is bigger than ff_y_relative
    points_with_big_angle = np.where(np.abs(ff_x_relative) > ff_y_relative)[0]
    if len(points_with_big_angle) > 0:
        percent_big_angle = len(points_with_big_angle) / len(ff_x_relative)
        if remove_i_o_modify_rows_with_big_ff_angles:
            if verbose:
                print('Warning: {}% of ff_x_relative are bigger than ff_y_relative. In these cases, the rows will be eliminated'.format(round(percent_big_angle*100, 3)))
            points_with_not_big_angles = np.where(np.abs(ff_x_relative) <= ff_y_relative)[0]
            indices_of_kept_rows = indices_of_kept_rows[points_with_not_big_angles]
            ff_df = original_ff_df.iloc[indices_of_kept_rows].copy()
        else:
            if verbose:
                print('Warning: {}% of ff_x_relative are bigger than ff_y_relative. In these cases, the absolute values of ff_x_relative are changed to be the same as ff_y_relative'.format(round(percent_big_angle*100, 3)))
            ff_x_relative[points_with_big_angle] = ff_y_relative[points_with_big_angle] * np.sign(points_with_big_angle)
            ff_angle = np.arctan2(ff_y_relative, ff_x_relative) - math.pi/2
            ff_distance = np.sqrt(ff_x_relative**2 + ff_y_relative**2)
            ff_df['ff_angle'] = ff_angle
            ff_df['ff_distance'] = ff_distance
            ff_df['ff_angle_boundary'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=ff_angle, distances_to_ff=ff_distance, ff_radius=10)

            ff_x, ff_y = show_null_trajectory.turn_relative_xy_positions_to_absolute_xy_positions(ff_x_relative, ff_y_relative, ff_df.monkey_x, ff_df.monkey_y, ff_df.monkey_angle)
            ff_df['ff_x'] = ff_x
            ff_df['ff_y'] = ff_y

            ## to only keep the big angle rows (for the purpose of plotting/debugging)
            # indices_of_kept_rows = indices_of_kept_rows[points_with_big_angle]
            # ff_df = ff_df.iloc[indices_of_kept_rows].copy()

    return ff_df, indices_of_kept_rows
    

def find_ff_info_n_seconds_ago(ff_df, monkey_information, ff_real_position_sorted, n_seconds=-1):
    #new_ff_df = ff_df[['ff_index']].copy()
    ff_df = ff_df.copy()
    ff_df['time'] = ff_df['stop_time'] + n_seconds
    ff_df['point_index'] = np.searchsorted(monkey_information['monkey_t'].values, ff_df['time'].values, side='right') - 1
    ff_df['point_index'] = np.clip(ff_df['point_index'], 0, len(monkey_information)-1)
    new_ff_df = find_ff_info(ff_df.ff_index.values, ff_df.point_index.values, monkey_information, ff_real_position_sorted)
    new_ff_df['stop_point_index'] = ff_df['stop_point_index'].values
    return new_ff_df


def find_ff_info_n_cm_ago(ff_df, monkey_information, ff_real_position_sorted, n_cm=-50):
    #new_ff_df = ff_df[['ff_index']].copy()
    ff_df = ff_df.copy()
    ff_df['cum_distance'] = ff_df['stop_cum_distance'] + n_cm
    ff_df['point_index'] = np.searchsorted(monkey_information['cum_distance'].values, ff_df['cum_distance'].values, side='right') - 1
    ff_df['point_index'] = np.clip(ff_df['point_index'], 0, len(monkey_information)-1)
    new_ff_df = find_ff_info(ff_df.ff_index.values, ff_df.point_index.values, monkey_information, ff_real_position_sorted)
    new_ff_df['stop_point_index'] = ff_df['stop_point_index'].values
    return new_ff_df

def find_ff_info(all_ff_index, all_point_index, monkey_information, ff_real_position_sorted):
    ff_df = pd.DataFrame({'ff_index': all_ff_index, 'point_index': all_point_index})
    ff_df[['ff_x', 'ff_y']] = ff_real_position_sorted[ff_df['ff_index'].values]
    ff_df[['monkey_x', 'monkey_y', 'monkey_angle']] = monkey_information.loc[ff_df['point_index'], ['monkey_x', 'monkey_y', 'monkey_angles']].values
    ff_df['ff_distance'] = np.linalg.norm(ff_df[['monkey_x', 'monkey_y']].values - ff_real_position_sorted[ff_df['ff_index'].values], axis=1)
    ff_df['ff_angle'] = basic_func.calculate_angles_to_ff_centers(ff_x=ff_df['ff_x'].values, ff_y=ff_df['ff_y'].values, \
                                                                            mx=ff_df['monkey_x'].values, my=ff_df['monkey_y'].values, m_angle=ff_df['monkey_angle'].values)
    ff_df['ff_angle_boundary'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=ff_df.ff_angle.values, distances_to_ff=ff_df.ff_distance.values, ff_radius=10)
    return ff_df


def normalize(array):
    array = (array - array.mean()) / array.std()
    return array


def plot_relationship(alt_curv_counted, traj_curv_counted, slope=None, show_plot=True, change_units_to_degrees_per_m=True):
    
    alt_curv_counted = alt_curv_counted.copy()
    traj_curv_counted = traj_curv_counted.copy()

    if change_units_to_degrees_per_m:
        alt_curv_counted = alt_curv_counted * (180/np.pi) * 100
        traj_curv_counted = traj_curv_counted * (180/np.pi) * 100

    slope, intercept, r_value, p_value, std_err = stats.linregress(alt_curv_counted, traj_curv_counted)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(alt_curv_counted, traj_curv_counted)
    # calculate and plot linear correlation
    x_min = min(alt_curv_counted)
    x_max = max(alt_curv_counted)
    ax.plot(np.array([x_min, x_max]), np.array([x_min, x_max])*slope+intercept, color='red')
    plt.ylabel('curv_of_traj - curv_to_stop_ff')
    plt.xlabel('curv_to_alt_ff - curv_to_stop_ff')
    plt.title('r_value = %f' % r_value + ', slope = %f' % slope)
    ax.grid()
    ax.axvline(x=0, color='black', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--')

    if show_plot:
        plt.show()
        
    return ax


def find_relative_curvature(alt_ff_counted_df, stop_ff_counted_df, curv_of_traj_counted, use_curvature_to_ff_center):
    if curv_of_traj_counted is None:
        raise ValueError('curv_of_traj_counted cannot be None')

    if use_curvature_to_ff_center:
        curv_var = 'curv_to_ff_center'
    else:
        curv_var = 'optimal_curvature'

    alt_ff_counted_df = alt_ff_counted_df.copy()
    stop_ff_counted_df = stop_ff_counted_df.copy()

    traj_curv_counted = curv_of_traj_counted - stop_ff_counted_df[curv_var]
    alt_curv_counted = alt_ff_counted_df[curv_var] - stop_ff_counted_df[curv_var]
    
    traj_curv_counted = traj_curv_counted.values
    alt_curv_counted = alt_curv_counted.values
    return traj_curv_counted, alt_curv_counted


def find_outliers_in_a_column(df, column, outlier_z_score_threshold=2):
    outlier_positions = basic_func.find_outlier_position_index(df[column].values, outlier_z_score_threshold=outlier_z_score_threshold)
    non_outlier_positions = np.setdiff1d(range(len(df)), outlier_positions)
    return outlier_positions, non_outlier_positions



def confine_angle_to_within_one_pie(angle_array):
    while np.any(angle_array > math.pi):
        angle_array[angle_array > math.pi] = angle_array[angle_array > math.pi] - 2*math.pi
    while np.any(angle_array < -math.pi):
        angle_array[angle_array < -math.pi] = angle_array[angle_array < -math.pi] + 2*math.pi
    return angle_array


def confine_angle_to_within_180(angle_array):
    while np.any(angle_array > 180):
        angle_array[angle_array > 180] = angle_array[angle_array > 180] - 2*180
    while np.any(angle_array < -180):
        angle_array[angle_array < -180] = angle_array[angle_array < -180] + 2*180
    return angle_array


def organize_snf_streamline_organizing_info_kwargs(ref_point_params, curv_of_traj_params, overall_params):
    snf_streamline_organizing_info_kwargs = {
                                'ref_point_mode': ref_point_params['ref_point_mode'],
                                'ref_point_value': ref_point_params['ref_point_value'],
                                'eliminate_outliers': overall_params['eliminate_outliers'],
                                'curv_of_traj_mode': curv_of_traj_params['curv_of_traj_mode'],
                                'window_for_curv_of_traj': curv_of_traj_params['window_for_curv_of_traj'],
                                'truncate_curv_of_traj_by_time_of_capture': curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'],
                                'remove_i_o_modify_rows_with_big_ff_angles': overall_params['remove_i_o_modify_rows_with_big_ff_angles'],
                                'use_curvature_to_ff_center': overall_params['use_curvature_to_ff_center']}
    return snf_streamline_organizing_info_kwargs


def add_instances_to_polar_plot(axes, stops_near_ff_df, alt_ff_df2, monkey_information, max_instances, color='green',
                                start='stop_point_index', end='next_stop_point_index'):    
    
    traj_df, stop_point_df, next_stop_point_df = _get_important_df_for_polar_plot(stops_near_ff_df, alt_ff_df2, monkey_information, max_instances, start, end)

    # Visualize ff_info
    axes.scatter(traj_df['monkey_angle_from_ref'].values, traj_df['monkey_distance_from_ref'].values, 
                 c=color, alpha=0.3, zorder=2, marker='o', s=1) # originally it was s=15
    if start == 'ref_point_index':
        axes.scatter(stop_point_df['monkey_angle_from_ref'].values, stop_point_df['monkey_distance_from_ref'].values,
                     c='red', alpha=0.5, zorder=3, marker='s', s=3)
    if end == 'next_stop_point_index':
        axes.scatter(next_stop_point_df['monkey_angle_from_ref'].values, next_stop_point_df['monkey_distance_from_ref'].values,
                    c='blue', alpha=0.5, zorder=3, marker='*', s=3)
    return axes


def add_instances_to_plotly_polar_plot(fig, stops_near_ff_df, alt_ff_df2, monkey_information, max_instances, color='green', point_color='blue',
                                start='stop_point_index', end='next_stop_point_index', legendgroup='Test data'):    
    
    traj_df, stop_point_df, next_stop_point_df = _get_important_df_for_polar_plot(stops_near_ff_df, alt_ff_df2, monkey_information, max_instances, start, end)

    # Main scatter plot for each subset
    fig.add_trace(go.Scatterpolar(
        r=traj_df['monkey_distance_from_ref'].values,
        theta=traj_df['monkey_angle_from_ref'].values * 180/pi,
        mode='markers',
        marker=dict(color=color, size=2, opacity=0.5),
        name=legendgroup,
        legendgroup=legendgroup,
    ))

    # Additional markers for start and end points
    if start == 'ref_point_index':
        fig.add_trace(go.Scatterpolar(
            r=stop_point_df['monkey_distance_from_ref'],
            theta=stop_point_df['monkey_angle_from_ref'] * 180/pi,
            mode='markers',
            marker=dict(color=point_color, size=4, opacity=0.7),
            name='stop Points',
            legendgroup=legendgroup,
        ))
    if end == 'next_stop_point_index':
        fig.add_trace(go.Scatterpolar(
            r=next_stop_point_df['monkey_distance_from_ref'],
            theta=next_stop_point_df['monkey_angle_from_ref'] * 180/pi,
            mode='markers',
            marker=dict(color=point_color, size=4, opacity=0.7),
            name='next Stop Points',
            legendgroup=legendgroup,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction="clockwise")
        ),
        title="Monkey Movement Polar Plot"
    )

    return fig

def _get_important_df_for_polar_plot(stops_near_ff_df, alt_ff_df2, monkey_information, max_instances, start, end):
    traj_df = pd.DataFrame()
    stop_point_df = pd.DataFrame()
    next_stop_point_df = pd.DataFrame()

    for index, row in stops_near_ff_df.iterrows():
        if index >= max_instances:
            break
        
        monkey_sub = _get_monkey_sub_for_polar_plot(monkey_information, row, alt_ff_df2, start, end)
        traj_df = pd.concat([traj_df, monkey_sub[['monkey_angle_from_ref', 'monkey_distance_from_ref']]], axis=0)
        if start == 'ref_point_index':
            stop_point_df = pd.concat([stop_point_df, monkey_sub.loc[[row['stop_point_index']]]], axis=0)
        if end == 'next_stop_point_index':
            next_stop_point_df = pd.concat([next_stop_point_df, monkey_sub.loc[[row['next_stop_point_index']]]], axis=0)
    return traj_df, stop_point_df, next_stop_point_df


def _get_monkey_sub_for_polar_plot(monkey_information, row, alt_ff_df2, start, end):
    point_index_dict = {'stop_point_index': row['stop_point_index'],
                        'next_stop_point_index': row['next_stop_point_index'],
                        'ref_point_index': alt_ff_df2.loc[alt_ff_df2['stop_point_index']==row['stop_point_index'], 'point_index'].item()
                        }
    
    monkey_sub = monkey_information.loc[point_index_dict[start]: point_index_dict[end] + 1].copy()
    
    # rotated monkey_x and monkey_y in reference to monkey angle at the reference point
    monkey_ref_xy = monkey_sub.loc[point_index_dict[start], ['monkey_x', 'monkey_y']].values
    monkey_ref_angle = monkey_sub.loc[point_index_dict[start], 'monkey_angle'].item()
    monkey_sub['monkey_distance_from_ref'] = np.linalg.norm(monkey_sub[['monkey_x', 'monkey_y']].values - monkey_ref_xy, axis=1)
    monkey_sub['monkey_angle_from_ref'] = np.arctan2(monkey_sub['monkey_y'] - monkey_ref_xy[1], monkey_sub['monkey_x'] - monkey_ref_xy[0]) - monkey_ref_angle
    return monkey_sub


def check_ff_vs_cluster(df, ff_column, cluster_column):
    len_subset = len(df[df[ff_column] < df[cluster_column]])
    print(f'There are {len_subset} rows where {ff_column} < {cluster_column} out of {len(df)} rows')
    len_subset = len(df[df[ff_column] > df[cluster_column]])
    print(f'There are {len_subset} rows where {ff_column} > {cluster_column} out of {len(df)} rows')
    len_subset = len(df[(df[ff_column].isnull()) & (~df[cluster_column].isnull())])
    print(f'There are {len_subset} rows where {ff_column} is null but {cluster_column} is not null out of {len(df)} rows')
    len_subset = len(df[(~df[ff_column].isnull()) & (df[cluster_column].isnull())])
    print(f'There are {len_subset} rows where {ff_column} is not null but {cluster_column} is null out of {len(df)} rows')

def find_df_name(monkey_name, ref_point_mode, ref_point_value):
    if ref_point_mode == 'time':
        ref_point_mode_name = 'time'
    elif ref_point_mode == 'distance':
        ref_point_mode_name = 'dist'
        ref_point_value = int(ref_point_value)
    elif ref_point_mode == 'time after stop ff visible':
        ref_point_mode_name = 'stop'
    else:
        ref_point_mode_name = 'special'

    if monkey_name is not None:
        if len(monkey_name.split('_')) > 1:
            df_name = monkey_name.split('_')[1] + '_' + ref_point_mode_name + '_' + str(abs(ref_point_value))
            df_name = df_name.replace('.', '_')
            return df_name
        
    # otherwise
    df_name = ref_point_mode_name + '_' + str(abs(ref_point_value))
    df_name = df_name.replace('.', '_')
    return df_name


def find_diff_in_curv_df_name(ref_point_mode=None, ref_point_value=None, curv_traj_window_before_stop=[0, 0]):
    if (ref_point_mode is not None) & (ref_point_value is not None):
        ref_df_name = find_df_name(None, ref_point_mode, ref_point_value)
        ref_df_name = ref_df_name + '_'
    else:
        ref_df_name = ''
    df_name = ref_df_name + f'window_{curv_traj_window_before_stop[0]}cm_{curv_traj_window_before_stop[1]}cm'
    return df_name


def find_ff_info_based_on_ref_point(ff_info, monkey_information, ff_real_position_sorted, ref_point_mode='distance', ref_point_value=-150,
                                    point_index_stop_ff_first_seen=None,
                                    # Note: ref_point_mode can be 'time', 'distance', or ‘time after stop ff visible’
                                    ): 
    if ref_point_mode == 'time':
        if ref_point_value >= 0:
            raise ValueError('ref_point_value must be negative for ref_point_mode = "time"')
        ff_info2 = find_ff_info_n_seconds_ago(ff_info, monkey_information, ff_real_position_sorted, n_seconds=ref_point_value)
    elif ref_point_mode == 'distance':
        if ref_point_value >= 0:
            raise ValueError('ref_point_value must be negative for ref_point_mode = "distance"')
        if 'stop_cum_distance' not in ff_info.columns:
            ff_info['stop_cum_distance'] = monkey_information.loc[ff_info['stop_point_index'].values, 'cum_distance'].values
        ff_info2 = find_ff_info_n_cm_ago(ff_info, monkey_information, ff_real_position_sorted, n_cm=ref_point_value)
    elif ref_point_mode == 'time after stop ff visible':
        if point_index_stop_ff_first_seen is None:
            point_index_stop_ff_first_seen = ff_info['point_index_ff_first_seen'].values
        all_time = monkey_information.loc[point_index_stop_ff_first_seen, 'time'].values + ref_point_value
        new_point_index = np.searchsorted(monkey_information['monkey_t'].values, all_time, side='right') - 1
        new_point_index = np.clip(new_point_index, 0, len(monkey_information)-1)
        ff_info2 = find_ff_info(ff_info.ff_index.values, new_point_index, monkey_information, ff_real_position_sorted)
        ff_info2['stop_point_index'] = ff_info['stop_point_index'].values
    else:
        raise ValueError('ref_point_mode not recognized')
    ff_info2 = ff_info2.sort_values(by='stop_point_index').reset_index(drop=True)
    return ff_info2

def process_shared_stops_near_ff_df(shared_stops_near_ff_df):
    shared_stops_near_ff_df['temp_id'] = np.arange(len(shared_stops_near_ff_df))
    original_len = len(shared_stops_near_ff_df)
    stop_periods_stop_ff_not_visible = shared_stops_near_ff_df[shared_stops_near_ff_df['STOP_point_index_ff_first_seen_bbas'].isnull()].temp_id.values
    stop_periods_alt_ff_not_visible = shared_stops_near_ff_df[shared_stops_near_ff_df[['ALT_time_ff_first_seen_bbas', 'ALT_time_ff_first_seen_bsans']].isnull().all(axis=1) == True].temp_id.values
    stop_periods_to_remove = np.concatenate([stop_periods_alt_ff_not_visible, stop_periods_stop_ff_not_visible])
    shared_stops_near_ff_df = shared_stops_near_ff_df[~shared_stops_near_ff_df['temp_id'].isin(stop_periods_to_remove)].copy()
    shared_stops_near_ff_df.drop(columns=['temp_id'], inplace=True)
    print(f'Removed {original_len - len(shared_stops_near_ff_df)} rows out of {original_len} rows where stop_ff was not visible bbas or alt_ff was not visible both bbas and bsans')
    print(f'shared_stops_near_ff_df has {len(shared_stops_near_ff_df)} rows')
    return shared_stops_near_ff_df
