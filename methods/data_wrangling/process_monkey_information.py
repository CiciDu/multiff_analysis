import sys
from data_wrangling import time_offset_utils, retrieve_raw_data, general_utils
from pattern_discovery import pattern_by_trials, pattern_by_trials

import os
import math
from math import pi
import re
import os.path
import neo
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from os.path import exists
from scipy.ndimage import gaussian_filter1d
from non_behavioral_analysis import eye_positions

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_or_retrieve_monkey_information(raw_data_folder_path, interocular_dist, min_distance_to_calculate_angle=5, speed_threshold_for_distinct_stop=1,
                                        exists_ok=True, save_data=True):
    processed_data_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'processed_data')
    monkey_information_path = os.path.join(processed_data_folder_path, 'monkey_information.csv')
    if exists(monkey_information_path) & exists_ok:
        print("Retrieved monkey_information")
        monkey_information = pd.read_csv(monkey_information_path).drop(["Unnamed: 0"], axis=1)
    else:
        smr_markers_start_time, smr_markers_end_time = time_offset_utils.find_smr_markers_start_and_end_time(raw_data_folder_path)
        monkey_information = pd.DataFrame({'time': np.arange(0, smr_markers_end_time, 0.01)})
        monkey_information['point_index'] = np.arange(len(monkey_information['time']))          
        monkey_information = _trim_monkey_information(monkey_information, smr_markers_start_time, smr_markers_end_time)
        monkey_information = _add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path = raw_data_folder_path,
                                                                    variables = ['LDy', 'LDz', 'RDy', 'RDz', 'MonkeyX', 'MonkeyY', 'LateralV', 'ForwardV', 'AngularV'])
        monkey_information.rename(columns={'MonkeyX': 'monkey_x', 'MonkeyY': 'monkey_y', 'AngularV': 'monkey_dw'}, inplace=True)
        monkey_information = _get_monkey_speed_and_dw(monkey_information)

        add_monkey_angle_column(monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)
        monkey_information.drop(columns=['LateralV', 'ForwardV'], inplace=True)

        # convert the eye position data
        monkey_information = eye_positions.convert_eye_positions_in_monkey_information(monkey_information, add_left_and_right_eyes_info=True, interocular_dist=interocular_dist)
        if save_data:
            monkey_information.to_csv(monkey_information_path)
            print("Saved monkey_information")

    monkey_information.index = monkey_information.point_index.values
    monkey_information = add_more_columns_to_monkey_information(monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop) 
    monkey_information = take_out_suspicious_information_from_monkey_information(monkey_information)
    return monkey_information


# def make_or_retrieve_monkey_information(raw_data_folder_path, interocular_dist, min_distance_to_calculate_angle=5, speed_threshold_for_distinct_stop=1,
#                                         exists_ok=True, save_data=True):
#     processed_data_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'processed_data')
#     monkey_information_path = os.path.join(processed_data_folder_path, 'monkey_information.csv')
#     if exists(monkey_information_path) & exists_ok:
#         print("Retrieved monkey_information")
#         monkey_information = pd.read_csv(monkey_information_path).drop(["Unnamed: 0"], axis=1)
#     else:
#         smr_markers_start_time, smr_markers_end_time = time_offset_utils.find_smr_markers_start_and_end_time(raw_data_folder_path)
#         monkey_information = pd.DataFrame({'time': np.arange(0, smr_markers_end_time, 0.01)})
#         monkey_information['point_index'] = np.arange(len(monkey_information['time']))          
#         monkey_information = _trim_monkey_information(monkey_information, smr_markers_start_time, smr_markers_end_time)
#         monkey_information = _add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path = raw_data_folder_path,
#                                                                     variables = ['LDy', 'LDz', 'RDy', 'RDz', 'MonkeyX', 'MonkeyY', 'LateralV', 'ForwardV', 'AngularV'])
#         monkey_information.rename(columns={'MonkeyX': 'monkey_x', 'MonkeyY': 'monkey_y', 'AngularV': 'monkey_dw'}, inplace=True)
#         monkey_information = _get_monkey_speed_and_dw(monkey_information)

#         add_monkey_angle_column(monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)
#         monkey_information.drop(columns=['LateralV', 'ForwardV'], inplace=True)

#         # convert the eye position data
#         monkey_information = eye_positions.convert_eye_positions_in_monkey_information(monkey_information, add_left_and_right_eyes_info=True, interocular_dist=interocular_dist)
#         if save_data:
#             monkey_information.to_csv(monkey_information_path)
#             print("Saved monkey_information")

#     monkey_information.index = monkey_information.point_index.values
#     monkey_information = add_more_columns_to_monkey_information(monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop) 
#     monkey_information = take_out_suspicious_information_from_monkey_information(monkey_information)
#     return monkey_information


def _get_monkey_speed_and_dw(monkey_information):
    monkey_information['monkey_speed'] = LA.norm(monkey_information[['LateralV', 'ForwardV']].values, axis=1)
    monkey_information['monkey_dw'] = monkey_information['monkey_dw'] * pi/180
    #monkey_information['monkey_speed'] = gaussian_filter1d(monkey_information['monkey_speed'], 1)
    #monkey_information['monkey_dw'] = gaussian_filter1d(monkey_information['monkey_dw'], 1)

    # check if any point has a speed that's too high. Print the number of such points (and proportion of them) as well as highest speed
    too_high_speed_points = monkey_information[monkey_information['monkey_speed'] > 200]
    if len(too_high_speed_points) > 0:
        print("There are", len(too_high_speed_points), "points with speed greater than 200 cm/s")
        print("The proportion of such points is", len(too_high_speed_points)/len(monkey_information))
        print("The highest speed is", np.max(monkey_information['monkey_speed']))
        monkey_information.loc[ monkey_information['monkey_speed'] > 200, 'monkey_speed'] = 200
    monkey_information['monkey_speeddummy'] = ((monkey_information['monkey_speed'] > 0.1) | \
                                                (np.abs(monkey_information['monkey_dw']) > 0.0035)).astype(int) 
    return monkey_information

def _get_derivative_of_a_column(monkey_information, column_name, derivative_name):
    dvar = np.diff(monkey_information[column_name])
    dvar1 = np.append(dvar[0], dvar)
    dvar2 = np.append(dvar, dvar[-1])
    avg_dvar = (dvar1 + dvar2)/2
    monkey_information[derivative_name] = avg_dvar
    return monkey_information


def add_delta_distance_and_cum_distance_to_monkey_information(monkey_information):
    monkey_x = monkey_information['monkey_x']
    monkey_y = monkey_information['monkey_y']

    monkey_information['delta_distance'] = np.sqrt((monkey_x.diff())**2 + (monkey_y.diff())**2)
    monkey_information['delta_distance'] = monkey_information['delta_distance'].fillna(0)

    monkey_information['cum_distance'] = np.cumsum(monkey_information['delta_distance'])


def take_out_suspicious_information_from_monkey_information(monkey_information):
    # find delta_position
    delta_time = np.diff(monkey_information['time'].values)
    ceiling_of_delta_position = max(20, np.max(delta_time)*200*3) # times 1.5 to make the criterion slightly looser
    monkey_information, abnormal_point_index = _drop_rows_where_delta_position_exceeds_a_ceiling(monkey_information, ceiling_of_delta_position)

    print('The number of points that were removed due to delta_position exceeding the ceiling is', len(abnormal_point_index))

    # Since sometimes the erroneous points can occur several in the row, we shall repeat the procedure, until the points all come back to normal.
    # However, if the process has been repeated for more than 5 times, then we'll raise a warning.
    procedure_counter = 1
    while len(abnormal_point_index) > 0:
        # repeat the procedure above
        monkey_information, abnormal_point_index = _drop_rows_where_delta_position_exceeds_a_ceiling(monkey_information, ceiling_of_delta_position)
        procedure_counter += 1
        if procedure_counter == 5:
            print("Warning: there are still erroneous points in the monkey information after 5 times of correction!")

    if procedure_counter > 1:
        print("The procedure to remove erroneous points was repeated", procedure_counter, "times")

    return monkey_information


def _drop_rows_where_delta_position_exceeds_a_ceiling(monkey_information, ceiling_of_delta_position):
    delta_x = np.diff(monkey_information['monkey_x'].values)
    delta_y = np.diff(monkey_information['monkey_y'].values)
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    # for the points where delta_position is greater than the ceiling
    above_ceiling_point_index = np.where(delta_position > ceiling_of_delta_position)[0]
    # inspect if points are away from the boundary (because if it's near the boundary, then we accept the big delta position)
    corr_monkey_info = monkey_information.iloc[above_ceiling_point_index+1].copy()
    corr_monkey_info['distances_from_center'] = np.linalg.norm(corr_monkey_info[['monkey_x', 'monkey_y']].values, axis=1)
    # if so, delete these points
    abnormal_point_index = corr_monkey_info[corr_monkey_info['distances_from_center'] < 1000 - ceiling_of_delta_position].index
    monkey_information = monkey_information.drop(abnormal_point_index)
    return monkey_information, abnormal_point_index


def _calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points):
    num_points_past = int(np.floor(num_points / 2))
    num_points_future = num_points - num_points_past

    # make sure that the two numbers don't go out of bound
    if i - num_points_past < 0:
        num_points_past = i
        num_points_future = num_points - num_points_past
        if i + num_points_future >= total_points:
            delta_x = 0
            delta_y = 0
            current_delta_position = 0
            return delta_x, delta_y, current_delta_position, num_points_past, num_points_future
    
    if i + num_points_future >= total_points:
        num_points_future = total_points - i - 1
        num_points_past = num_points - num_points_future
        if i - num_points_past < 0:
            delta_x = 0
            delta_y = 0
            current_delta_position = 0
            return delta_x, delta_y, current_delta_position, num_points_past, num_points_future

    delta_x = monkey_information['monkey_x'].values[i+num_points_future] - monkey_information['monkey_x'].values[i-num_points_past]
    delta_y = monkey_information['monkey_y'].values[i+num_points_future] - monkey_information['monkey_y'].values[i-num_points_past]
    current_delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    return delta_x, delta_y, current_delta_position, num_points_past, num_points_future


def add_more_columns_to_monkey_information(monkey_information, speed_threshold_for_distinct_stop=1):
    monkey_information = _get_derivative_of_a_column(monkey_information, column_name='monkey_dw', derivative_name='monkey_ddw')
    monkey_information = _get_derivative_of_a_column(monkey_information, column_name='monkey_speed', derivative_name='monkey_ddv')
    add_crossing_boundary_column(monkey_information)
    add_delta_distance_and_cum_distance_to_monkey_information(monkey_information)
    monkey_information = add_whether_new_distinct_stop_column(monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
    # assign "stop_id" to each stop, with each whether_new_distinct_stop==True marking a new stop id
    monkey_information['stop_id'] = monkey_information['whether_new_distinct_stop'].cumsum() - 1
    monkey_information.loc[monkey_information['monkey_speeddummy'] == 1, 'stop_id'] = np.nan        
    monkey_information['dt'] = (monkey_information['time'].shift(-1) - monkey_information['time']).ffill()
    return monkey_information


def _add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path, variables = ['LDy', 'LDz', 'RDy', 'RDz']):
    monkey_information = monkey_information.copy()
    signal_df = time_offset_utils.make_signal_df(raw_data_folder_path)
    time_bins = general_utils.find_time_bins_for_an_array(monkey_information['time'].values)

    # add time_box to monkey_information
    monkey_information.loc[:, 'time_box'] = np.arange(1, len(monkey_information)+1)
    # group signal_df.time based on intervals in monkey_information['time'], thus adding the column time_box to signal_df
    signal_df.loc[:, 'time_box'] = np.digitize(signal_df['Time'].values, time_bins)
    # use groupby and then find average for LDy, LDz, RDy, RDz
    variables.append('time_box')
    condensed_signal_df = signal_df[variables]
    condensed_signal_df = condensed_signal_df.groupby('time_box').median().reset_index(drop=False)

    # Put these info into monkey_information
    monkey_information = monkey_information.merge(condensed_signal_df, how='left', on='time_box')
    monkey_information.drop(columns=['time_box'], inplace=True)
    return monkey_information


def _trim_monkey_information(monkey_information, smr_markers_start_time, smr_markers_end_time):
    # Chop off the beginning part and the end part of monkey_information
    time = monkey_information['time'].values
    if monkey_information['time'][0] < smr_markers_start_time:
        valid_points = np.where((time >= smr_markers_start_time) & (time <= smr_markers_end_time))[0]
        monkey_information = monkey_information.iloc[valid_points]

    return monkey_information

def add_monkey_speed_column(monkey_information):
    delta_time = np.diff(monkey_information['time'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5)

    # If the monkey's delta_position at one point exceeds 50, we replace it with the previous speed.
    # (This can happen when the monkey reaches the boundary and comes out at another place)
    while np.where(delta_position >= ceiling_of_delta_position)[0].size > 0:
        above_ceiling_point_index = np.where(delta_position>=ceiling_of_delta_position)[0]
        # find the previous speed for all those points
        delta_position_prev = np.append(np.array([0]), delta_position)
        delta_position[above_ceiling_point_index] = delta_position_prev[above_ceiling_point_index]  

    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(monkey_speed[0], monkey_speed)
    monkey_information['monkey_speed'] = monkey_speed
    # and make sure that the monkey_speed does not exceed maximum speed
    monkey_information.loc[monkey_information['monkey_speed'] > 200, 'monkey_speed'] = 200


def add_monkey_dw_column(monkey_information):
    # positive dw means the monkey is turning counterclockwise
    delta_time = np.diff(monkey_information['time'])
    delta_angle = np.diff(monkey_information['monkey_angle'])
    delta_angle = np.remainder(delta_angle, 2*pi)
    delta_angle[delta_angle >= pi] = delta_angle[delta_angle >= pi]-2*pi
    monkey_dw = np.divide(delta_angle, delta_time)
    monkey_dw = np.append(monkey_dw[0], monkey_dw)
    monkey_information['monkey_dw'] = monkey_dw
    #monkey_information['monkey_dw'] = gaussian_filter1d(monkey_information['monkey_dw'], 1)



def add_crossing_boundary_column(monkey_information):
    delta_time = np.diff(monkey_information['time'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5)
    crossing_boundary = np.append(0, (delta_position > ceiling_of_delta_position).astype('int'))
    monkey_information['crossing_boundary'] = crossing_boundary


def add_whether_new_distinct_stop_column(monkey_information, speed_threshold_for_distinct_stop=1):
    # The standard for distinct stop is that every two stops should be separated by at least one point 
    # that has a speed greater than than speed_threshold_for_distinct_stop cm/s.
    monkey_df = monkey_information.copy()
    # mark whether the speed is over the threshold
    monkey_df['speed_over_threshold'] = monkey_df['monkey_speed'] > speed_threshold_for_distinct_stop
    # get the cumulative sum of the speed_over_threshold
    monkey_df['cum_speed_over_threshold'] = monkey_df['speed_over_threshold'].cumsum()
    # take out the stop points
    monkey_df = monkey_df[monkey_df['monkey_speeddummy'] == 0].copy()
    # get distinct stops
    monkey_df = monkey_df.groupby('cum_speed_over_threshold').first().reset_index(drop=False)
    # add back the info to monkey_information
    monkey_information['whether_new_distinct_stop'] = False
    monkey_information.loc[monkey_information['point_index'].isin(monkey_df['point_index']), 'whether_new_distinct_stop'] = True
    return monkey_information


def add_monkey_angle_column(monkey_information, min_distance_to_calculate_angle=5):
    # Add angle of the monkey
    monkey_angles = [pi/2]  # The monkey is at 90 degree angle at the beginning
    list_of_num_points = [0]
    list_of_num_points_past = [0]
    list_of_num_points_future = [0]
    list_of_delta_positions = [0]
    current_angle = pi/2 # This keeps track of the current angle during the iterations
    previous_angle = pi/2
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])

    # Find the time in the data that is closest (right before) the time where we wan to know the monkey's angular position.
    total_points = len(monkey_information['time'])
    num_points = 1

    for i in range(1, total_points):

        if num_points < 1:
            num_points = 1

        if num_points >= total_points:
            # use the below so that the algorithm will not simply get stuck after num_points exceeds the total number;
            # rather, we give a num_points a chance to come down a little and re-do the calculation again.
            num_points = num_points - min(total_points - 1, 5)

        delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        # first, let's make current_delta_position within min_distance_to_calculate_angle so that we can shed off excessive distance
        while (current_delta_position > min_distance_to_calculate_angle) and (num_points > 1):
            num_points -= 1 
            # now distribute the num_points to the past and future and calculate delta position 
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        # then, we make sure that current_delta_position is just above min_distance_to_calculate_angle
        while (current_delta_position <= min_distance_to_calculate_angle) and (num_points < total_points):
            num_points += 1 
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        if current_delta_position < 50:
            # calculate the angle defined by two points
            current_angle = math.atan2(delta_y, delta_x)
        else:
            # Otherwise, most likely the monkey has crossed the boundary and come out at another place; we shall keep the current angle, and not update it
            current_angle = previous_angle
        
        monkey_angles.append(current_angle)
        previous_angle = current_angle
        list_of_num_points.append(num_points)
        list_of_num_points_past.append(num_points_past)
        list_of_num_points_future.append(num_points_future)
        list_of_delta_positions.append(current_delta_position)
    monkey_information['monkey_angle'] = np.array(monkey_angles)
