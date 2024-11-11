import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import basic_func

import os
import numpy as np
import pandas as pd
from math import pi
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)





def make_empty_ff_dataframe():
   ff_dataframe = pd.DataFrame(columns = ['ff_index', 'point_index', 'time', 'target_index', 'ff_x', 'ff_y',
                                        'monkey_x', 'monkey_y', 'visible', 'memory', 'ff_distance', 'ff_angle',
                                        'ff_angle_boundary', 'left_right', 'abs_delta_ff_angle',
                                        'abs_delta_ff_angle_boundary', 'target_x', 'target_y',
                                        'ffdistance2target', 'abs_ffangle_decreasing',
                                        'abs_ffangle_boundary_decreasing', 'IS_TARGET', 'monkey_dw',
                                        'ff_index_string', 'dw_same_sign_as_ffangle',
                                        'dw_same_sign_as_ffangle_boundary'])
   return ff_dataframe




def find_valid_indices(monkey_information, ff_caught_T_sorted, player='monkey', num_missed_index=None):
    monkey_t_array0 = np.array(monkey_information['monkey_t'])
    if player == "monkey":
        if num_missed_index is None:
            num_missed_index = np.where(monkey_t_array0 > ff_caught_T_sorted[0])[0][0]
        
    else:
        upper_limit = len(monkey_t_array0)-1
        num_missed_index = 0
    upper_limit = np.where(monkey_t_array0 < ff_caught_T_sorted[-1])[0]
    if len(upper_limit) == 0:
        valid_index = np.array([])
    else:
        upper_limit = upper_limit[-1]
        if num_missed_index >= upper_limit +1:
            valid_index = np.array([])
        else:
            valid_index = np.arange(num_missed_index, upper_limit+1)
    
    return valid_index, num_missed_index





def find_visible_indices_AND_memory_array_AND_time_since_last_visible(current_ff_index, monkey_information, valid_index, obs_ff_indices_in_ff_dataframe, ff_flash_sorted, ff_real_position_sorted, ff_caught_T_sorted, max_distance, player='monkey', max_memory=100, truncate_info_beyond_capture=True): 
    monkey_t_array_all = monkey_information['monkey_t'][:valid_index[-1]+1].values
    monkey_angles_array_all = monkey_information['monkey_angles'][:valid_index[-1]+1].values
    monkey_x_array_all = monkey_information['monkey_x'][:valid_index[-1]+1].values
    monkey_y_array_all = monkey_information['monkey_y'][:valid_index[-1]+1].values
    # Here we use monkey_t_array_all instead of monkey_t_array because if the ff has been visible before the beginning of monkey_t_array, then there might be memory points at the beginning of monkey_t_array
    if player == "monkey":
        raw_visible_indices = find_visible_indices(current_ff_index, ff_flash_sorted, ff_caught_T_sorted, ff_real_position_sorted, monkey_t_array_all, monkey_x_array_all, monkey_y_array_all, monkey_angles_array_all, max_distance)

    else: # Otherwise, if the player is "agent"  
        raw_visible_indices = find_visible_indices_for_agent(current_ff_index, obs_ff_indices_in_ff_dataframe, valid_index=range(valid_index[-1]+1))
    visible_indices = raw_visible_indices[raw_visible_indices >= valid_index[0]]


    memory_array = np.array([])
    in_memory_indices = np.array([])
    time_since_last_visible = np.array([])
    if len(raw_visible_indices) > 0:
        memory_array = make_memory_array(raw_visible_indices, current_ff_index, ff_caught_T_sorted, monkey_t_array_all, max_memory=max_memory, truncate_info_beyond_capture=truncate_info_beyond_capture)
        if len(memory_array) > valid_index[0]:
            memory_valid_index = range(valid_index[0], min(valid_index[-1]+1, len(memory_array))) # this is necessary because memory_array has been truncated by last_alive_point
            memory_array = memory_array[memory_valid_index]

            # Find the point indices where the firefly is in memory or visible
            in_memory_indices = np.where(memory_array > 0)[0]
            # Find the corresponding memory for these points and only keep those, since we don't need information of the ff
            # when the ff is neither visible nor in memory
            memory_array = memory_array[in_memory_indices]
            time_since_last_visible = find_time_since_last_visible(raw_visible_indices, monkey_t_array_all)
            time_since_last_visible = time_since_last_visible[in_memory_indices]

    return visible_indices, memory_array, in_memory_indices, time_since_last_visible





def find_visible_indices(current_ff_index, ff_flash_sorted, ff_caught_T_sorted, ff_real_position_sorted, monkey_t_array, monkey_x_array, monkey_y_array, monkey_angles_array, max_distance):
    i = current_ff_index
    ff_flash = ff_flash_sorted[i]
    # visible_indices contains the indices of the points when the ff is visible (within a suitable distance & at the right angle)
    visible_indices = np.array([])
    all_cum_indices = []
    for j in range(len(ff_flash)):
        visible_duration = ff_flash[j]
        # Find the corresponding monkey information:
        cum_iloc_indices = np.where((monkey_t_array >= visible_duration[0]) & (monkey_t_array <= visible_duration[1]))[0].tolist()
        all_cum_indices.extend(cum_iloc_indices)
    all_cum_indices = np.array(all_cum_indices).astype('int')
    if len(all_cum_indices) > 0: 
        cum_mx, cum_my, cum_angle = monkey_x_array[all_cum_indices], monkey_y_array[all_cum_indices], monkey_angles_array[all_cum_indices]
        distances_to_ff = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis = 1)
        valid_distance_indices = np.where(distances_to_ff < max_distance)[0]
        if len(valid_distance_indices) > 0:
            angles_to_ff = basic_func.calculate_angles_to_ff_centers(ff_x=ff_real_position_sorted[i, 0], ff_y=ff_real_position_sorted[i, 1], mx=cum_mx[valid_distance_indices], my=cum_my[valid_distance_indices], m_angle=cum_angle[valid_distance_indices])
            angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angles_to_ff, distances_to_ff=distances_to_ff[valid_distance_indices])
            overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_to_boundaries) <= 2*pi/9)[0]]
            # Store these points from the current duration into visible_indices
            visible_indices = all_cum_indices[overall_valid_indices]
    
    # See if the current ff has been captured at any point
    # If it has been captured, then its index i should be smaller than the number of caught fireflies (i.e. the number of elements in ff_caught_T_sorted)
    if current_ff_index < len(ff_caught_T_sorted):
        # Find the index of the time at which the ff is captured
        last_alive_point = np.where(monkey_t_array <= ff_caught_T_sorted[current_ff_index])[0][-1]
        # Truncate visible_indices so that its last point does not exceed last_live_time
        visible_indices = visible_indices[visible_indices <= last_alive_point]
    
    return visible_indices



def find_visible_indices_for_agent(current_ff_index, obs_ff_indices_in_ff_dataframe, valid_index):
    # We'll only consider the points of time when the ff of interest was in obs space
    whether_in_obs = []
    # iterate through every point (step taken by the agent)
    for index in valid_index:
        # take out all the fireflies in the obs space
        obs_ff_indices = obs_ff_indices_in_ff_dataframe[index]
        # if the ff of interest was in the obs space
        if current_ff_index in obs_ff_indices:
            whether_in_obs.append(True) 
        else:
            whether_in_obs.append(False)
    # find the point indices where the ff of interest was in the obs space
    cum_iloc_indices = np.array(whether_in_obs).nonzero()[0]
    if len(cum_iloc_indices) == 0:
        # The ff of interest has never been in the obs space, so we move on to the next ff
        visible_indices = np.array([]) 
    else: 
        visible_indices = cum_iloc_indices
    return visible_indices




def make_memory_array(visible_indices, current_ff_index, ff_caught_T_sorted, monkey_t_array, max_memory=100, truncate_info_beyond_capture=True):
    # Make an array of points to denote memory, with 0 means being invisible, and 100 being fully visible. 
    # After a firefly turns from being visible to being invisible, memory will decrease by 1 for each additional step taken by the monkey/agent.    
    # We append max_memory elements at the end of initial_memory_array to aid iteration through this array later
    initial_memory_array = np.zeros(max(visible_indices[-1]+max_memory, len(monkey_t_array)), dtype=int)
    # Make sure that the points where the ff is fully visible has a memory of max_memory (100 by default)
    initial_memory_array[visible_indices] = max_memory
    

    # We preserve the first element of initial_memory_array and then iterate through initial_memory_array to make a new list to 
    # denote memory (replacing some 0s with other numbers based on time). 
    
  
    memory_array = [initial_memory_array[0]]
    for k in range(1, len(initial_memory_array)):
        # If the ff is currently invisible
        if initial_memory_array[k] == 0:
            # Then its memory is the memory from the previous point minus one
            memory_array.append(memory_array[k-1]-1)
        else: # Else, the firefly is visible
            memory_array.append(max_memory)
    memory_array = np.array(memory_array)
    # We need to make sure that the length of memory_array does not exceed the number of data points in monkey_t_array
    if len(memory_array) > len(monkey_t_array):
        # We also truncate memory_array so that its length does not surpass the length of monkey_t_array
        memory_array = memory_array[:len(monkey_t_array)]
    
    # Trim the points beyond last_alive_point
    if truncate_info_beyond_capture:
        if current_ff_index < len(ff_caught_T_sorted):
            # Find the index of the time at which the ff is captured
            last_alive_point = np.where(monkey_t_array <= ff_caught_T_sorted[current_ff_index])[0][-1]
            # Truncate visible_indices so that its last point does not exceed last_live_time
            memory_array = memory_array[:last_alive_point+1]

    return memory_array



def find_time_since_last_visible(visible_indices, monkey_t_array):
    invisible_indices = np.setdiff1d(np.arange(len(monkey_t_array)), visible_indices)
    monkey_t_array_copy = monkey_t_array.copy()
    monkey_t_array_copy[invisible_indices] = np.nan
    # use forward fill to fill in the nan values
    monkey_t_array_copy = pd.Series(monkey_t_array_copy).ffill().values
    # and fill the rest of NA to be the the first value of monkey_t_array
    monkey_t_array_copy = pd.Series(monkey_t_array_copy).fillna(monkey_t_array[0]).values

    time_since_last_visible = monkey_t_array - monkey_t_array_copy

    ''' Below is another method with similar speed
    monkey_t_array_diff = np.diff(monkey_t_array)
    monkey_t_array_diff = np.insert(monkey_t_array_diff, 0, 0)
    monkey_t_array_diff[visible_indices] = 0
    time_since_last_visible = []
    for i in monkey_t_array_diff:
        if i == 0:
            time_since_last_visible.append(0)
        else:
            time_since_last_visible.append(time_since_last_visible[-1]+i)
    time_since_last_visible = np.array(time_since_last_visible)
    '''
    return time_since_last_visible


def find_time_till_next_visible(visible_indices, monkey_t_array):
    invisible_indices = np.setdiff1d(np.arange(len(monkey_t_array)), visible_indices)
    monkey_t_array_copy = monkey_t_array.copy()
    monkey_t_array_copy[invisible_indices] = np.nan
    # use forward fill to fill in the nan values
    monkey_t_array_copy = pd.Series(monkey_t_array_copy).fillna(method='bfill').values
    # and fill the rest of NA to be the the first value of monkey_t_array
    monkey_t_array_copy = pd.Series(monkey_t_array_copy).fillna(monkey_t_array[-1]).values

    time_till_next_visible = monkey_t_array_copy - monkey_t_array

    return time_till_next_visible


def add_caught_time_and_whether_caught_to_ff_dataframe(ff_dataframe, ff_caught_T_sorted, ff_life_sorted, dt=0.016):
    env_end_time = ff_life_sorted[-1, -1] + 100
    num_ff_to_be_added = len(ff_life_sorted) - len(ff_caught_T_sorted)
    ff_caught_T_sorted_extended = np.append(ff_caught_T_sorted, np.repeat(env_end_time, num_ff_to_be_added))
    ff_dataframe['caught_time'] = ff_caught_T_sorted_extended[np.array(ff_dataframe['ff_index'])]
    ff_dataframe['whether_caught'] = (np.abs(ff_dataframe['caught_time'] - ff_dataframe['time']) < 0.016+0.01).astype('int')
    return ff_dataframe



def keep_only_ff_that_monkey_has_passed_by_closely(ff_dataframe, duration=None, max_distance_to_ff=100):
    if duration is not None:
        ff_dataframe = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]
    ff_indices_to_keep = ff_dataframe[ff_dataframe['ff_distance'] < max_distance_to_ff].ff_index.unique()
    ff_dataframe = ff_dataframe[ff_dataframe['ff_index'].isin(ff_indices_to_keep)]
    return ff_dataframe 

    