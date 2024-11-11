from re import search
import sys
from decision_making_analysis import trajectory_info
from null_behaviors import find_best_arc, curvature_utils, curvature_utils
from visualization.plotly_tools import plotly_preparation
from visualization import monkey_heading_functions

import os
import warnings
import numpy as np
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import math
from scipy import stats


def initialize_curv_of_traj_df(all_point_index, monkey_information):
    curv_of_traj_df = pd.DataFrame({'point_index': all_point_index})
    monkey_information = monkey_information.copy()
    curv_of_traj_df = curv_of_traj_df.merge(monkey_information[['point_index', 'time', 'monkey_angle', 'cum_distance']], on='point_index', how='left')
    return curv_of_traj_df


def find_curv_of_traj_df_based_on_shifts(all_point_index, monkey_information, ff_caught_T_sorted, shifts, reindex=True, truncate_curv_of_traj_by_time_of_capture=False):
    if reindex:
        monkey_information = reindex_monkey_information_based_on_all_point_index_and_shifts(all_point_index, monkey_information, shifts)  
    
    monkey_information = monkey_information.loc[:max(all_point_index) + max(shifts) + 100].copy()
    curv_of_traj_df = find_curv_of_traj_df_based_on_point_index_window(all_point_index, shifts[0], shifts[-1], monkey_information, ff_caught_T_sorted, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)

    return curv_of_traj_df


def reindex_monkey_information_based_on_all_point_index_and_shifts(all_point_index, monkey_information, shifts):
    shifts_array = np.array(shifts)
    point_index_of_interest = []
    for point_index in all_point_index:
        point_index_of_interest.extend((point_index + shifts_array).tolist())
    point_index_of_interest = np.unique(np.array(point_index_of_interest))
    monkey_information = monkey_information.reindex(point_index_of_interest).copy()  
    monkey_information['point_index'] = monkey_information.index 
    return monkey_information 


def find_curv_of_traj_df_based_on_point_index_window(all_point_index, lower_end, upper_end, monkey_information, ff_caught_T_sorted, truncate_curv_of_traj_by_time_of_capture=False):   
    if lower_end > 0:
        warnings.warn('lower_end is greater than 0. This is not recommended.')
    
    curv_of_traj_df = initialize_curv_of_traj_df(all_point_index, monkey_information)
    curv_of_traj_df['point_index_lower_end'] = curv_of_traj_df['point_index'] + lower_end
    curv_of_traj_df['point_index_upper_end'] = curv_of_traj_df['point_index'] + upper_end

    if truncate_curv_of_traj_by_time_of_capture:
        curv_of_traj_df = truncate_curv_of_traj_by_time_of_capture_new_func(curv_of_traj_df, monkey_information, ff_caught_T_sorted)
    curv_of_traj_df = find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(curv_of_traj_df, monkey_information)

    return curv_of_traj_df



def find_curv_of_traj_df_based_on_time_window(all_point_index, lower_end, upper_end, monkey_information, ff_caught_T_sorted, truncate_curv_of_traj_by_time_of_capture=False): 
    if lower_end > 0:
        warnings.warn('lower_end is greater than 0. This is not recommended.')
      
    curv_of_traj_df = initialize_curv_of_traj_df(all_point_index, monkey_information)
    curv_of_traj_df['time_lower_end'] = curv_of_traj_df['time'] + lower_end
    curv_of_traj_df['time_upper_end'] = curv_of_traj_df['time'] + upper_end
    curv_of_traj_df['point_index_lower_end'] = np.searchsorted(monkey_information['time'].values, curv_of_traj_df['time_lower_end'].values, side='right') - 1
    curv_of_traj_df['point_index_upper_end'] = np.searchsorted(monkey_information['time'].values, curv_of_traj_df['time_upper_end'].values, side='right')
    curv_of_traj_df.drop(columns=['time_lower_end', 'time_upper_end'], inplace=True)
    if truncate_curv_of_traj_by_time_of_capture:
        curv_of_traj_df = truncate_curv_of_traj_by_time_of_capture_new_func(curv_of_traj_df, monkey_information, ff_caught_T_sorted)    
    curv_of_traj_df = find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(curv_of_traj_df, monkey_information)

    return curv_of_traj_df


def find_curv_of_traj_df_based_on_distance_window(all_point_index, lower_end, upper_end, monkey_information, ff_caught_T_sorted, truncate_curv_of_traj_by_time_of_capture=False):   
    if lower_end > 0:
        warnings.warn('lower_end is greater than 0. This is not recommended.')

    # see if point_index of monkey_information is in order
    if not np.array_equal(monkey_information['point_index'].values, np.arange(len(monkey_information))):
        raise ValueError('point_index of monkey_information is not in order. This is not allowed.')

    curv_of_traj_df = initialize_curv_of_traj_df(all_point_index, monkey_information)
    curv_of_traj_df['cum_distance_lower_end'] = curv_of_traj_df['cum_distance'] + lower_end
    curv_of_traj_df['cum_distance_upper_end'] = curv_of_traj_df['cum_distance'] + upper_end
    curv_of_traj_df['point_index_lower_end'] = np.searchsorted(monkey_information['cum_distance'].values, curv_of_traj_df['cum_distance_lower_end'].values, side='right') - 1
    curv_of_traj_df['point_index_upper_end'] = np.searchsorted(monkey_information['cum_distance'].values, curv_of_traj_df['cum_distance_upper_end'].values, side='right')
    curv_of_traj_df.drop(columns=['cum_distance_lower_end', 'cum_distance_upper_end'], inplace=True)
    if truncate_curv_of_traj_by_time_of_capture:
        curv_of_traj_df = truncate_curv_of_traj_by_time_of_capture_new_func(curv_of_traj_df, monkey_information, ff_caught_T_sorted)    
    curv_of_traj_df = find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(curv_of_traj_df, monkey_information)

    return curv_of_traj_df



def find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(curv_of_traj_df, monkey_information):
    # curv_of_traj_df has to have the following 2 columns at the minimal: point_index_lower_end, point_index_upper_end

    curv_of_traj_df.loc[curv_of_traj_df['point_index_lower_end'] <= 0, 'point_index_lower_end'] = 0
    curv_of_traj_df.loc[curv_of_traj_df['point_index_upper_end'] <= 0, 'point_index_upper_end'] = 1
    curv_of_traj_df.loc[curv_of_traj_df['point_index_lower_end'] >= monkey_information.point_index.max(), 'point_index_lower_end'] = monkey_information.point_index.max() - 1
    curv_of_traj_df.loc[curv_of_traj_df['point_index_upper_end'] >= monkey_information.point_index.max(), 'point_index_upper_end'] = monkey_information.point_index.max()

    monkey_info_for_lower_end = monkey_information.copy()
    monkey_info_for_lower_end.rename(columns={'point_index': 'point_index_lower_end',
                                              'monkey_angle': 'monkey_angle_lower_end',
                                              'cum_distance': 'cum_distance_lower_end',
                                              'monkey_t': 'time_lower_end'}, inplace=True)
    monkey_info_for_upper_end = monkey_information.copy()
    monkey_info_for_upper_end.rename(columns={'point_index': 'point_index_upper_end',
                                              'monkey_angle': 'monkey_angle_upper_end',
                                              'cum_distance': 'cum_distance_upper_end',
                                              'monkey_t': 'time_upper_end'}, inplace=True)
    curv_of_traj_df = curv_of_traj_df.merge(monkey_info_for_lower_end[['point_index_lower_end', 'monkey_angle_lower_end', 'cum_distance_lower_end', 'time_lower_end']], on='point_index_lower_end', how='left')
    curv_of_traj_df = curv_of_traj_df.merge(monkey_info_for_upper_end[['point_index_upper_end', 'monkey_angle_upper_end', 'cum_distance_upper_end', 'time_upper_end']], on='point_index_upper_end', how='left')

    curv_of_traj_df['delta_distance'] = curv_of_traj_df['cum_distance_upper_end'] - curv_of_traj_df['cum_distance_lower_end']
    curv_of_traj_df['delta_monkey_angle'] = curv_of_traj_df['monkey_angle_upper_end'] - curv_of_traj_df['monkey_angle_lower_end']
    for i in range(2):
        curv_of_traj_df.loc[curv_of_traj_df['delta_monkey_angle'] > math.pi, 'delta_monkey_angle'] = curv_of_traj_df.loc[curv_of_traj_df['delta_monkey_angle'] > math.pi, 'delta_monkey_angle'] - 2*math.pi
        curv_of_traj_df.loc[curv_of_traj_df['delta_monkey_angle'] < -math.pi, 'delta_monkey_angle'] = curv_of_traj_df.loc[curv_of_traj_df['delta_monkey_angle'] < -math.pi, 'delta_monkey_angle'] + 2*math.pi
        
    curv_of_traj_df['curvature_of_traj'] = curv_of_traj_df['delta_monkey_angle'] / curv_of_traj_df['delta_distance'] 
    curv_of_traj_df['curvature_of_traj_deg_over_cm'] = curv_of_traj_df['curvature_of_traj'] * 180/np.pi * 100 # so that the unit is degree/cm

    curv_of_traj_df['min_point_index'] = curv_of_traj_df['point_index_lower_end']
    curv_of_traj_df['max_point_index'] = curv_of_traj_df['point_index_upper_end']
    curv_of_traj_df['initial_monkey_angle'] = curv_of_traj_df['monkey_angle_lower_end']
    curv_of_traj_df['final_monkey_angle'] = curv_of_traj_df['monkey_angle_upper_end']

    return curv_of_traj_df





def truncate_curv_of_traj_by_time_of_capture_new_func(curv_of_traj_df, monkey_information, ff_caught_T_sorted):
    # curv_of_traj_df must contain point_index, point_index_lower_end, point_index_upper_end
    searchsorted_result = np.searchsorted(monkey_information['monkey_t'].values, ff_caught_T_sorted, side='right')
    searchsorted_result[searchsorted_result > len(monkey_information)-1] = len(monkey_information)-1
    ff_caught_point_index = monkey_information['point_index'].values[searchsorted_result]
    point_index_values = curv_of_traj_df[['point_index_lower_end', 'point_index', 'point_index_upper_end']].values
    # In each row, the change of index indicates that there's at least one a capture between the two points
    corresponding_ff_caught_point_index_iloc = np.searchsorted(ff_caught_point_index, point_index_values)  
    rows_involving_capture = np.where(np.diff(corresponding_ff_caught_point_index_iloc, axis=1) != 0)[0]

    corresponding_ff_caught_point_index_iloc[corresponding_ff_caught_point_index_iloc > len(ff_caught_point_index)-1] = len(ff_caught_point_index)-1

    # now, we shall truncate the rows that involve capture
    curv_of_traj_df = curv_of_traj_df.copy()
    biggest_ff_caught_point_index_before_point_index = ff_caught_point_index[corresponding_ff_caught_point_index_iloc[:, 1] - 1]
    
    
    smallest_ff_caught_point_index_after_point_index = ff_caught_point_index[corresponding_ff_caught_point_index_iloc[:, 1]]
    # make the smallest point index one less than the actual ff_caught_point_index
    smallest_ff_caught_point_index_after_point_index -= 1
    # but still make sure that the smallest point index is larger than the point_index
    smallest_ff_caught_point_index_after_point_index[smallest_ff_caught_point_index_after_point_index < curv_of_traj_df['point_index'].values] = curv_of_traj_df['point_index'].values[smallest_ff_caught_point_index_after_point_index < curv_of_traj_df['point_index'].values]
    
    # for point_index_lower_end, we keep the larger one between the original value and the corresponding_ff_caught_point_index that falls between point_index_lower_end and point_index
    curv_of_traj_df.loc[rows_involving_capture, 'point_index_lower_end'] = np.maximum(biggest_ff_caught_point_index_before_point_index[rows_involving_capture], curv_of_traj_df.loc[rows_involving_capture, 'point_index_lower_end'].values)
    # for point_index_upper_end, we keep the smaller one between the original value and the corresponding_ff_caught_point_index that falls between point_index and point_index_upper_end
    curv_of_traj_df.loc[rows_involving_capture, 'point_index_upper_end'] = np.minimum(smallest_ff_caught_point_index_after_point_index[rows_involving_capture], curv_of_traj_df.loc[rows_involving_capture, 'point_index_upper_end'].values)
    return curv_of_traj_df

    

def find_curv_of_traj_df_in_duration(curv_of_traj_df, duration_to_plot):
    curv_of_traj_df_in_duration = curv_of_traj_df[curv_of_traj_df['time'].between(duration_to_plot[0], duration_to_plot[1], inclusive='both')].copy()
    return curv_of_traj_df_in_duration


def find_all_curv_of_traj_df(monkey_information, ff_caught_T_sorted, 
                             all_time_windows=[x*2 for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]],
                             truncate_curv_of_traj_by_time_of_capture=False):
    all_point_index = monkey_information.index.values
    all_curv_of_traj_df = pd.DataFrame()
    for time_window in all_time_windows:
        half_time_window = time_window/2
        print('time_window:', time_window)
        temp_curv_of_traj_df = find_curv_of_traj_df_based_on_time_window(all_point_index, -half_time_window, half_time_window, monkey_information, ff_caught_T_sorted, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        temp_curv_of_traj_df['time_window'] = time_window
        all_curv_of_traj_df = pd.concat([all_curv_of_traj_df, temp_curv_of_traj_df], axis=0)
    all_curv_of_traj_df['time'] = monkey_information.loc[all_curv_of_traj_df.point_index.values, 'monkey_t']
    all_curv_of_traj_df['curvature_of_traj_deg_over_cm'] = all_curv_of_traj_df['curvature_of_traj'] * 180/math.pi * 100 # so that the unit is degree/cm
    return all_curv_of_traj_df




def calculate_difference_in_curvature_of_traj(new_curv_df):
    # calculate curvature_of_traj_diff and curvature_of_traj_diff_over_dt
    new_curv_df['curvature_of_traj_diff'] = new_curv_df['curvature_of_traj_deg_over_cm'].diff()
    new_curv_df['curvature_of_traj_diff'] = new_curv_df['curvature_of_traj_diff'].fillna(0)
    new_curv_df['curvature_of_traj_diff_over_dt'] = new_curv_df['curvature_of_traj_deg_over_cm'].diff() / new_curv_df['time'].diff()
    new_curv_df['curvature_of_traj_diff_over_dt'] = new_curv_df['curvature_of_traj_diff_over_dt'].fillna(0)
    new_curv_df['curvature_of_traj_diff_over_distance'] = new_curv_df['curvature_of_traj_deg_over_cm'].diff() / (new_curv_df['delta_distance']*100).diff()
    new_curv_df['curvature_of_traj_diff_over_distance'] = new_curv_df['curvature_of_traj_diff_over_distance'].fillna(0)
    return new_curv_df




def find_curv_of_traj_df_based_on_from_current_point_to_right_before_stop(stops_near_ff_df, monkey_information):
    monkey_information['monkey_angle'] = monkey_information['monkey_angles']
    new_df = stops_near_ff_df[['stop_point_index', 'point_index_before_stop', 'monkey_angle_before_stop']].copy()
    new_df.reset_index(inplace=True, drop=True)
    new_df['prev_point_index_before_stop'] = new_df['point_index_before_stop'].shift(1)
    new_df.loc[0, 'prev_point_index_before_stop'] = max(0, new_df.loc[0, 'stop_point_index']-500)
    new_df['prev_time_before_stop'] = monkey_information.loc[new_df['prev_point_index_before_stop'], 'monkey_t'].values

    point_index_lower_end = []
    point_index_upper_end = []

    for index, row in new_df.iterrows():
        prev_point_index_before_stop = int(row.prev_point_index_before_stop)
        point_index_before_stop = int(row.point_index_before_stop)
        point_index_lower_end.extend(range(prev_point_index_before_stop, point_index_before_stop))
        point_index_upper_end.extend([point_index_before_stop] * (point_index_before_stop - prev_point_index_before_stop))

    new_curv_of_traj_df = pd.DataFrame({'point_index': point_index_lower_end, 'point_index_lower_end': point_index_lower_end, 'point_index_upper_end': point_index_upper_end})
    new_curv_of_traj_df = add_basic_monkey_information(new_curv_of_traj_df, monkey_information)
    new_curv_of_traj_df = find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(new_curv_of_traj_df, monkey_information)
    # also add monkey direction info
    new_curv_of_traj_df = monkey_heading_functions.add_monkey_heading_info_to_curv_of_traj_df(new_curv_of_traj_df, monkey_information)
    new_curv_of_traj_df = calculate_difference_in_curvature_of_traj(new_curv_of_traj_df)

    return new_curv_of_traj_df




def find_curv_of_traj_df_based_on_from_current_point_to_right_before_stop_in_duration(row, duration, monkey_information):
    monkey_info_sub = monkey_information[monkey_information['monkey_t'].between(duration[0], duration[1], inclusive='both')].copy()
    new_curv_df = monkey_info_sub[['point_index', 'time', 'monkey_angle', 'monkey_x', 'monkey_y', 'delta_distance', 'cum_distance']].copy()
    new_curv_df = new_curv_df[new_curv_df['point_index'] < row.point_index_before_stop]

    # add the information from row to new_curv_df
    new_curv_df['point_index'] = new_curv_df['point_index'].astype(int)
    new_curv_df['stop_point_index'] = row.stop_point_index.astype(int)
    new_curv_df['time_before_stop'] = row.time_before_stop 
    
    # calculate curvature_of_traj
    new_curv_df['point_index_lower_end'] = new_curv_df['point_index'] 
    new_curv_df['point_index_upper_end'] = row.point_index_before_stop.astype(int)
    new_curv_df = find_curv_of_traj_df_based_on_lower_and_upper_ends_of_point_index(new_curv_df, monkey_information)

    # add some other variables
    new_curv_df = calculate_difference_in_curvature_of_traj(new_curv_df)
    new_curv_df['rel_time'] = np.round(new_curv_df['time_before_stop'] - new_curv_df['time'], 2)

    return new_curv_df



def add_basic_monkey_information(curv_of_traj_df, monkey_information):
    monkey_information = monkey_information.copy()
    curv_of_traj_df = curv_of_traj_df.merge(monkey_information[['point_index', 'time', 'monkey_angle', 'cum_distance']], on='point_index', how='left')
    return curv_of_traj_df




def find_traj_curv_descr(curv_of_traj_mode, lower_end, upper_end):
    traj_curv_descr='Traj Curv: ' + curv_of_traj_mode + ' ' + str(lower_end) + ' to ' + str(upper_end)
    if curv_of_traj_mode == 'time':
        traj_curv_descr = traj_curv_descr + ' sec'
    else:
        traj_curv_descr = traj_curv_descr + ' cm'
    return traj_curv_descr


def find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, monkey_information, ff_caught_T_sorted, stops_near_ff_df=None, curv_of_traj_mode='time', truncate_curv_of_traj_by_time_of_capture=False):
    if curv_of_traj_mode == 'now to stop':
        if stops_near_ff_df is None:
            raise ValueError('stops_near_ff_df must be specified if curv_of_traj_mode is from current point to right before stop')
        curv_of_traj_df = find_curv_of_traj_df_based_on_from_current_point_to_right_before_stop(stops_near_ff_df, monkey_information)
        traj_curv_descr = 'Traj Curv: From Current Point to Right Before Stop'
    else:
        if window_for_curv_of_traj is None:
            raise ValueError('window_for_curv_of_traj must be specified, in the format of [a, b]')
        lower_end = window_for_curv_of_traj[0]
        upper_end = window_for_curv_of_traj[1]
        if lower_end < upper_end:
            point_index_for_curv_of_traj_df = monkey_information.index.values
            if curv_of_traj_mode == 'time':
                curv_of_traj_df = find_curv_of_traj_df_based_on_time_window(point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_sorted, 
                                                                                                    truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
            elif curv_of_traj_mode == 'distance':
                curv_of_traj_df = find_curv_of_traj_df_based_on_distance_window(point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_sorted, 
                                                                                                        truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
            traj_curv_descr = find_traj_curv_descr(curv_of_traj_mode, lower_end, upper_end)
        else:
            raise ValueError('lower_end of the window_for_curv_of_traj_df cannot be smaller than the upper_end')
        
    curv_of_traj_df = monkey_heading_functions.add_monkey_heading_info_to_curv_of_traj_df(curv_of_traj_df, monkey_information)
    curv_of_traj_df = calculate_difference_in_curvature_of_traj(curv_of_traj_df)

    return curv_of_traj_df, traj_curv_descr


def get_curv_of_traj_trace_name(curv_of_traj_mode, window_for_curv_of_traj):
    if curv_of_traj_mode == 'now to stop':
        curv_of_traj_trace_name = 'Curv of Traj: now to stop'
    else:
        curv_of_traj_trace_name = 'Curv of Traj: ' + str(window_for_curv_of_traj[0]) + ' to ' + str(window_for_curv_of_traj[1])
        if curv_of_traj_mode == 'time':
            curv_of_traj_trace_name = curv_of_traj_trace_name + ' s'
        elif curv_of_traj_mode == 'distance':
            curv_of_traj_trace_name = curv_of_traj_trace_name + ' cm'
        else:
            raise ValueError('curv_of_traj_mode must be time window or distance window or now to stop')  
    return curv_of_traj_trace_name
                    
