import sys
import os
import numpy as np
import sys
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from contextlib import contextmanager
from os.path import exists
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def calculate_angles_to_ff_centers(ff_x, ff_y, mx, my, m_angle):
    """
    Calculate the angle to the center of a firefly or multiple fireflies from the monkey's or the agent's perspective
    Positive angle means to the left of the monkey, and negative angle means to the right of the monkey

    Parameters
    ----------
    ff_position: np.array
        containing the x-coordinates and the y-coordinates of all fireflies
    mx: np.array
        the x-coordinates of the monkey/agent
    my: np.array
        the y-coordinates of the monkey/agent
    m_angle: np.array
        the angles that the monkey/agent heads toward


    Returns
    -------
    angles_to_ff: np.array
        containing the angles of the centers of the fireflies to the monkey/agent

    """

    # find the angles of the given fireflies to the agent
    angles_to_ff = np.arctan2(ff_y-my, ff_x-mx)-m_angle
    # make sure that the angles are between (-pi, pi]
    angles_to_ff = np.remainder(angles_to_ff, 2*pi)

    # if distance to ff is very small, make the angle 0:
    try:
        distances_to_ff = LA.norm(np.stack([ff_x-mx, ff_y-my], axis=1), axis=1)
    except Exception as e:
        distances_to_ff = LA.norm(np.array([ff_x-mx, ff_y-my]), axis=0)
    if np.any(distances_to_ff < 0.001):
        print(
            f'Note: {np.sum(distances_to_ff < 0.001)} fireflies are too close to the monkey/agent. Their angles are set to 0.')
        angles_to_ff[distances_to_ff < 0.001] = 0

    try:
        angles_to_ff[angles_to_ff >
                     pi] = angles_to_ff[angles_to_ff > pi] - 2*pi
    except TypeError:
        # then angles_to_ff must be a scalar
        if angles_to_ff > pi:
            angles_to_ff = angles_to_ff - 2*pi

    try:
        angles_to_ff[angles_to_ff <= -
                     pi] = angles_to_ff[angles_to_ff <= -pi] + 2*pi
    except TypeError:
        # then angles_to_ff must be a scalar
        if angles_to_ff <= -pi:
            angles_to_ff = angles_to_ff + 2*pi

    return angles_to_ff


def calculate_angles_to_ff_boundaries(angles_to_ff, distances_to_ff, ff_radius=10):
    """
    Calculate the angle to the boundary of a firefly or multiple fireflies from the monkey's or the agent's perspective

    Parameters
    ----------
    angles_to_ff: np.array
        containing the angles of the centers of the fireflies to the monkey/agent
    distances_to_ff: np.array
        containing the distances of the fireflies to the agent
    ff_radius: num
        the radius of the visible area of each firefly  

    Returns
    -------
    angles_to_boundaries: np.array
        containing the smallest angles of the reward boundaries of the fireflies to the agent

    """

    # Adjust the angle based on reward boundary (i.e. find the smallest angle from the agent to the reward boundary)
    # using trignometry
    side_opposite = ff_radius
    # hypotenuse cannot be smaller than side_opposite
    hypotenuse = np.clip(distances_to_ff, a_min=side_opposite, a_max=2000)
    theta = np.arcsin(np.divide(side_opposite, hypotenuse))
    # we use absolute values of angles here so that the adjustment will only make the angles smaller
    angle_adjusted_abs = np.abs(angles_to_ff) - np.abs(theta)
    # thus we can find the smallest absolute angle to the firefly, which is the absolute angle to the boundary of the firefly
    angles_to_boundaries_abs = np.clip(angle_adjusted_abs, 0, pi)
    # restore the signs of the angles
    angles_to_boundaries = np.sign(angles_to_ff) * angles_to_boundaries_abs

    # for the points where the monkey/agent is within the ff_radius, the reward boundary will be 0
    if isinstance(angles_to_boundaries, float) is False:
        angles_to_boundaries[distances_to_ff <= ff_radius] = 0
    else:
        if distances_to_ff <= ff_radius:
            angles_to_boundaries = 0

    return angles_to_boundaries


def calculate_change_in_abs_ff_angle(current_ff_index, angles_to_ff, angles_to_boundaries, ff_real_position_sorted, monkey_x_array,
                                     monkey_y_array, monkey_angles_array, in_memory_indices):
    # To also calculate delta_angles_to_ff and delta_angles_to_boundaries:
    prev_monkey_xy_relevant = np.stack(
        [monkey_x_array[in_memory_indices-1], monkey_y_array[in_memory_indices-1]], axis=1)
    prev_ff_distance_relevant = LA.norm(
        prev_monkey_xy_relevant-ff_real_position_sorted[current_ff_index], axis=1)
    prev_monkey_angles_relevant = monkey_angles_array[in_memory_indices-1]
    prev_angles_to_ff = calculate_angles_to_ff_centers(ff_x=ff_real_position_sorted[current_ff_index, 0], ff_y=ff_real_position_sorted[
                                                       current_ff_index, 1], mx=prev_monkey_xy_relevant[:, 0], my=prev_monkey_xy_relevant[:, 1], m_angle=prev_monkey_angles_relevant)
    prev_angles_to_boundaries = calculate_angles_to_ff_boundaries(
        angles_to_ff=prev_angles_to_ff, distances_to_ff=prev_ff_distance_relevant)
    delta_abs_angles_to_ff = np.abs(angles_to_ff) - np.abs(prev_angles_to_ff)
    delta_abs_angles_to_boundary = np.abs(
        angles_to_boundaries) - np.abs(prev_angles_to_boundaries)
    return delta_abs_angles_to_ff, delta_abs_angles_to_boundary


def get_distance_between_two_points(currentTrial, ff_caught_T_new, monkey_information, ff_believed_position_sorted):
    """
    Find the absolute displacement between the target for the currentTrial and the target for currentTrial.
    Return 9999 if the monkey has hit the border at one point.

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 

    Returns
    -------
    displacement: numeric
        the distance between the starting and ending points of the monkey during a trial; 
        returns 9999 if the monkey has hit the border at any point during the trial

    """
    duration = [ff_caught_T_new[currentTrial-1], ff_caught_T_new[currentTrial]]
    cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & (
        monkey_information['time'] <= duration[1]))[0]
    displacement = 0
    if len(cum_pos_index) > 1:
        cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_pos_index]), np.array(
            monkey_information['monkey_y'].iloc[cum_pos_index])
        # If the monkey has hit the boundary
        if np.any(cum_mx[1:]-cum_mx[:-1] > 10) or np.any(cum_my[1:]-cum_my[:-1] > 10):
            displacement = 9999
        else:
            displacement = LA.norm(
                ff_believed_position_sorted[currentTrial]-ff_believed_position_sorted[currentTrial-1])
    return displacement


def get_cum_distance_traveled(currentTrial, ff_caught_T_new, monkey_information):
    """
    Find the length of the trajectory run by the monkey in the current trial

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time


    Returns
    -------
    distance: numeric
        the length of the trajectory run by the monkey in the current trial

    """
    duration = [ff_caught_T_new[currentTrial-1], ff_caught_T_new[currentTrial]]
    cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & (
        monkey_information['time'] <= duration[1]))[0]
    distance = 0
    if len(cum_pos_index) > 1:
        cum_t = np.array(monkey_information['time'].iloc[cum_pos_index])
        cum_speed = np.array(
            monkey_information['monkey_speed'].iloc[cum_pos_index])
        distance = np.sum((cum_t[1:] - cum_t[:-1])*cum_speed[1:])
    return distance


def find_currentTrial_or_num_trials_or_duration(ff_caught_T_new, currentTrial=None, num_trials=None, duration=None):
    # Among currentTrial, num_trials, duration, either currentTrial and num_trials must be specified, or duration must be specified
    if duration is None:
        duration = [ff_caught_T_new[currentTrial-num_trials],
                    ff_caught_T_new[currentTrial]]
    # elif duration[1] > ff_caught_T_new[-1]:
    #    raise ValueError("The second element of duration must be smaller than the last element of ff_caught_T_new")

    if currentTrial is None:
        try:
            if len(ff_caught_T_new) > 0:
                # Take the max of the results from two similar methods
                # Method 1:
                earlier_trials = np.where(ff_caught_T_new <= duration[1])[0]
                if len(earlier_trials) > 0:
                    currentTrial = earlier_trials[-1]
                else:
                    currentTrial = 1
                # Method 2:
                later_trials = np.where(ff_caught_T_new >= duration[0])[0]
                if len(later_trials) > 0:
                    currentTrial_2 = later_trials[0]
                else:
                    currentTrial_2 = 1
                currentTrial = max(currentTrial, currentTrial_2)
        except Exception as e:
            print('Finding currentTrial failed:',
                  e, 'currentTrial is set to None')
            currentTrial = None
    if num_trials is None:
        try:
            if len(ff_caught_T_new) > 0:
                trials_after_first_capture = np.where(
                    ff_caught_T_new <= duration[0])[0]
                if len(trials_after_first_capture) > 0:
                    num_trials = max(
                        1, currentTrial-trials_after_first_capture[-1])
                else:
                    num_trials = 1
        except Exception as e:
            num_trials = None

    return currentTrial, num_trials, duration


def initialize_monkey_sessions_df(raw_data_dir_name='all_monkey_data/raw_monkey_data'):
    list_of_monkey_name = []
    list_of_data_name = []
    for monkey_name in ['monkey_Bruno', 'monkey_Schro']:  # 'monkey_Quigley'
        monkey_path = os.path.join(raw_data_dir_name, monkey_name)
        for data_name in os.listdir(monkey_path):
            if data_name[0] == 'd':
                list_of_monkey_name.append(monkey_name)
                list_of_data_name.append(data_name)
    sessions_df = pd.DataFrame(
        {'monkey_name': list_of_monkey_name, 'data_name': list_of_data_name})
    sessions_df['finished'] = False
    return sessions_df


def check_whether_finished(sessions_df, monkey_name, data_name):
    current_session_info = ((sessions_df['monkey_name'] == monkey_name) & (
        sessions_df['data_name'] == data_name))
    whether_finished = sessions_df.loc[current_session_info, 'finished'].item()
    return whether_finished


def init_variations_list_func(ref_point_params_based_on_mode, folder_path=None, monkey_name=None):
    key_value_pairs = []
    for key, values in ref_point_params_based_on_mode.items():
        key_value_pairs.extend([[key, i] for i in values])
    variations_list = pd.DataFrame(key_value_pairs, columns=[
                                   'ref_point_mode', 'ref_point_value'])

    variations_list['monkey_name'] = monkey_name
    variations_list['stored'] = False
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        df_path = os.path.join(folder_path, 'variations_list.csv')
        variations_list.to_csv(df_path)
    return variations_list


def reorganize_data_into_chunks(monkey_information):
    prev_speed = 0
    chunk_counter = 0
    # Each time point has a number corresponding to the index of the chunk it belongs to
    chunk_numbers = []
    # w[i] shows the index of the time point at which the i-th chunk starts
    new_chunk_indices = [0]

    # for each time point
    for i in range(len(monkey_information['monkey_speed'])):
        speed = monkey_information['monkey_speed'].values[i]
        # if the speed is above half of the full speed (100 cm/s) and if the previous speed is below half of the full speed
        if (speed > 100) & (prev_speed <= 100):
            # start a new chunk
            chunk_counter += 1
            new_chunk_indices.append(i)
        chunk_numbers.append(chunk_counter)
        prev_speed = speed

    chunk_numbers = np.array(chunk_numbers)
    new_chunk_indices = np.array(new_chunk_indices)

    return chunk_numbers, new_chunk_indices


def take_out_valid_intervals_based_on_ff_caught_time(ff_caught_T_new,
                                                     gap_too_large_threshold=100,
                                                     min_combined_valid_interval_length=50):
    """
    Process firefly catch timestamps to identify valid time intervals of continuous activity.

    This function identifies periods of continuous firefly catching activity by:
    1. Finding gaps between catches that are too large (invalid intervals)
    2. Combining continuous valid intervals
    3. Removing intervals that are too short

    Args:
        ff_caught_T_new (array-like): Array of timestamps when fireflies were caught
        gap_too_large_threshold (int, optional): Maximum allowed gap between catches. Defaults to 100.
        min_combined_valid_interval_length (int, optional): Minimum length for a valid interval. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame containing start and end times of valid intervals
    """
    return _process_valid_intervals(ff_caught_T_new, gap_too_large_threshold, min_combined_valid_interval_length)


def _process_valid_intervals(timestamps, gap_threshold, min_length):
    """
    Helper function to process timestamps and identify valid intervals.

    Args:
        timestamps (array-like): Array of timestamps
        gap_threshold (int): Maximum allowed gap between timestamps
        min_length (int): Minimum length for a valid interval

    Returns:
        pd.DataFrame: DataFrame containing start and end times of valid intervals
    """
    # Create DataFrame with timestamps
    df = pd.DataFrame(timestamps, columns=['caught_t'])

    # Calculate previous timestamp by shifting values up, fill first value with 0
    df['prev_caught_t'] = df['caught_t'].shift(1).fillna(0)

    # Calculate time intervals between consecutive timestamps
    df['interval'] = df['caught_t'] - df['prev_caught_t']

    # Mark intervals as invalid if they're larger than the threshold
    df['invalid_interval'] = df['interval'] > gap_threshold

    # Group continuous valid intervals together
    df['combd_valid_interval_group'] = df['invalid_interval'].cumsum()

    # Remove all invalid intervals from consideration
    df = df[~df['invalid_interval']]

    # Calculate total length of each combined valid interval
    df['combd_valid_interval_length'] = df.groupby('combd_valid_interval_group')[
        'interval'].transform('sum')

    # Record start and end times of each valid interval
    df['combd_valid_interval_start'] = df.groupby('combd_valid_interval_group')[
        'caught_t'].transform('first')
    df['combd_valid_interval_end'] = df.groupby('combd_valid_interval_group')[
        'caught_t'].transform('last')

    # Remove intervals that are too short
    df = df[df['combd_valid_interval_length'] > min_length]

    # Create final DataFrame with just the start and end times of valid intervals
    valid_intervals_df = df[['combd_valid_interval_start',
                             'combd_valid_interval_end']].drop_duplicates()
    return valid_intervals_df


def take_out_valid_cluster_intervals(cluster_caught_T_new,
                                     gap_too_large_threshold=100,
                                     min_combined_valid_interval_length=50):
    """
    Process cluster catch timestamps to identify valid time intervals of continuous cluster activity.

    This function identifies periods of continuous cluster catching activity by:
    1. Finding gaps between cluster catches that are too large (invalid intervals)
    2. Combining continuous valid intervals
    3. Removing intervals that are too short

    Args:
        cluster_caught_T_new (array-like): Array of timestamps when clusters were caught
        gap_too_large_threshold (int, optional): Maximum allowed gap between cluster catches. Defaults to 100.
        min_combined_valid_interval_length (int, optional): Minimum length for a valid interval. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame containing start and end times of valid cluster intervals
    """
    return _process_valid_intervals(cluster_caught_T_new, gap_too_large_threshold, min_combined_valid_interval_length)

