from multiff_analysis.functions.data_wrangling import basic_func
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





def make_ff_dataframe_func(monkey_information, ff_caught_T_sorted, ff_flash_sorted,  
                    ff_real_position_sorted, ff_life_sorted, player = "monkey", 
                    max_distance = 400, ff_radius = 10, max_memory = 100,
                    data_folder_name = None, num_missed_index = None, print_progress = True, 
                    obs_ff_indices_in_ff_dataframe = None,
                    ff_noisy_xy_in_obs = None):

    """
    Make a dataframe called ff_dataframe that contains various information about all visible or "in-memory" fireflies at each time point


    Parameters
    ----------
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_caught_T_sorted: np.array
        containing the time when each captured firefly gets captured
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    player: str
        "monkey" or "agent" 
    max_distance: num
        the distance beyond which the firefly cannot be considered visible
    ff_radius: num
        the radius of a firefly; the current setting of the game sets it to be 10
    max_memory: num
        the numeric value of the variable "memory" for a firefly when it's fully visible
    data_folder_name: str, default is None
        the place to store the output as a csv
    num_missed_index: num, default is None
        the number of invalid indices at the beginning of any array in monkey_information;
        if default is used, then it will be calculated as the number of indices till the capture of first the firefly
    print_progress: bool
        whether to print the progress of making ff_dataframe
    obs_ff_indices_in_ff_dataframe: list
        a variable to be passed if the player is "agent"; it contains the correct indices of fireflies 
    ff_noisy_xy_in_obs: pd.dataframe
        containing the noisy x, y coordinates of fireflies at different points in time, collected from env

    Returns
    -------
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
        

    """

    if len(ff_caught_T_sorted) < 1:
       return make_empty_ff_dataframe()

    # Let's use data from monkey_information. But we shall cut off portion that is before the time of capturing the first target and the after capturing the last target
    monkey_t_array0 = np.array(monkey_information['monkey_t'])
    if player == "monkey":
        if num_missed_index is None:
            num_missed_index = np.where(monkey_t_array0 > ff_caught_T_sorted[0])[0][0]
    else:
        upper_limit = len(monkey_t_array0)-1
        num_missed_index = 0
    upper_limit = np.where(monkey_t_array0 < ff_caught_T_sorted[-1])[0]
    if len(upper_limit) == 0:
        return make_empty_ff_dataframe()
    else:
        upper_limit = upper_limit[-1]
    valid_index = np.arange(num_missed_index, upper_limit+1)
    monkey_t_array = monkey_t_array0[valid_index]
    monkey_x_array = np.array(monkey_information['monkey_x'])[valid_index]
    monkey_y_array = np.array(monkey_information['monkey_y'])[valid_index]
    monkey_angles_array = np.array(monkey_information['monkey_angles'])[valid_index]



    ff_index = []
    point_index = []
    time = []
    target_index = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    visible = []
    memory = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []
    abs_delta_ff_angle = []
    abs_delta_ff_angle_boundary = []
    left_right = []
    total_ff_num = len(ff_life_sorted)

    # For each firefly in the data (except the one captured first) 

    starting_ff = {"monkey": 1, "agent": 0}
    for i in range(starting_ff[player], total_ff_num):
      current_ff_index = i
      if player == "monkey":
        # Go through every visible duration of the same ff
        ff_flash = ff_flash_sorted[i]
        # visible_indices contains the indices of the points when the ff is visible (within a suitable distance & at the right angle)
        visible_indices = np.array([])
        all_cum_indices = []
        for j in range(len(ff_flash)):
          visible_duration = ff_flash[j]
          # Find the corresponding monkey information:
          cum_indices = np.where((monkey_t_array >= visible_duration[0]) & (monkey_t_array <= visible_duration[1]))[0].tolist()
          all_cum_indices.extend(cum_indices)
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

        
      else: # Otherwise, if the player is "agent"  
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
        cum_indices = np.array(whether_in_obs).nonzero()[0]
        if len(cum_indices) == 0:
          # The ff of interest has never been in the obs space, so we move on to the next ff
          continue
        else: 
          visible_indices = cum_indices

      # The following part is once again shared by "monkey" and "agent"
      if len(visible_indices) > 0:
        # Make an array of points to denote memory, with 0 means being invisible, and 100 being fully visible. 
        # After a firefly turns from being visible to being invisible, memory will decrease by 1 for each additional step taken by the monkey/agent.    
        # We append max_memory elements at the end of initial_memory_array to aid iteration through this array later
        initial_memory_array = np.zeros(visible_indices[-1]+max_memory, dtype=int)
        # Make sure that the points where the ff is fully visible has a memory of max_memory (100 by default)
        initial_memory_array[visible_indices] = max_memory
        
        # See if the current ff has been captured at any point
        # If it has been captured, then its index i should be smaller than the number of caught fireflies (i.e. the number of elements in ff_caught_T_sorted)
        if i < len(ff_caught_T_sorted):
          # Find the index of the time at which the ff is captured
          last_live_time = np.where(monkey_t_array <= ff_caught_T_sorted[i])[0][-1]
          # Truncate initial_memory_array so that its last point does not exceed last_live_time
          initial_memory_array = initial_memory_array[:last_live_time+1]
        
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
        

        # Find the point indices where the firefly is in memory or visible
        in_memory_indices = np.where(memory_array > 0)[0]
        # Find the corresponding memory for these points and only keep those, since we don't need information of the ff
        # when the ff is neither visible nor in memory
        memory_array = memory_array[in_memory_indices]
        num_points_in_memory = len(memory_array)
        
        # Append the values for this ff; Using list operations is faster than np.append here
        ff_index = ff_index + [current_ff_index] * num_points_in_memory
        point_index = point_index + [point + num_missed_index for point in in_memory_indices.tolist()]
        relevant_time = monkey_t_array[in_memory_indices]
        time = time + relevant_time.tolist()
        target_index = target_index + np.digitize(relevant_time, ff_caught_T_sorted).tolist()
        ff_x = ff_x + [(ff_real_position_sorted[current_ff_index][0])]*num_points_in_memory
        ff_y = ff_y + [(ff_real_position_sorted[current_ff_index][1])]*num_points_in_memory
        monkey_x = monkey_x + monkey_x_array[in_memory_indices].tolist()
        monkey_y = monkey_y + monkey_y_array[in_memory_indices].tolist()
        visible = visible + [1 if point == 100 else 0 for point in memory_array.tolist()]
        memory = memory + memory_array.tolist()
        # In the following, "relevant" means in memory
        monkey_xy_relevant = np.stack([monkey_x_array[in_memory_indices], monkey_y_array[in_memory_indices]],axis=1)
        ff_distance_relevant = LA.norm(monkey_xy_relevant-ff_real_position_sorted[current_ff_index], axis=1)
        ff_distance = ff_distance + ff_distance_relevant.tolist()
        monkey_angles_relevant = monkey_angles_array[in_memory_indices]    
        angles_to_ff = basic_func.calculate_angles_to_ff_centers(ff_x=ff_real_position_sorted[current_ff_index, 0], ff_y=ff_real_position_sorted[current_ff_index, 1], mx=monkey_xy_relevant[:, 0], my=monkey_xy_relevant[:, 1], m_angle=monkey_angles_relevant)
        angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angles_to_ff, distances_to_ff=ff_distance_relevant)      
        ff_angle = ff_angle + angles_to_ff.tolist()
        ff_angle_boundary = ff_angle_boundary + angles_to_boundaries.tolist()
        left_right = left_right + (np.array(angles_to_ff) > 0).astype(int).tolist()
        
        ## To also calculate delta_angles_to_ff and delta_angles_to_boundaries:
        delta_abs_angles_to_ff, delta_abs_angles_to_boundary = basic_func.calculate_change_in_abs_ff_angles(current_ff_index, angles_to_ff, angles_to_boundaries, ff_real_position_sorted, monkey_x_array, monkey_y_array, monkey_angles_array, in_memory_indices)
        abs_delta_ff_angle = abs_delta_ff_angle + delta_abs_angles_to_ff.tolist()
        abs_delta_ff_angle_boundary = abs_delta_ff_angle_boundary + delta_abs_angles_to_boundary.tolist()
              

      if i % 100 == 0:
        if print_progress:
          print("Making ff_dataframe: ", i, " out of ", total_ff_num, " total number of fireflies ")


    # Now let's create a dictionary from the lists
    ff_dict = {'ff_index': ff_index, 'point_index': point_index, 'time': time, 'target_index': target_index,
                'ff_x': ff_x, 'ff_y': ff_y, 'monkey_x': monkey_x, 'monkey_y': monkey_y, 'visible': visible,
                'memory': memory, 'ff_distance': ff_distance, 'ff_angle': ff_angle, 'ff_angle_boundary': ff_angle_boundary, 
                'left_right': left_right, 'abs_delta_ff_angle': abs_delta_ff_angle, 'abs_delta_ff_angle_boundary': abs_delta_ff_angle_boundary}
    ff_dataframe = pd.DataFrame(ff_dict)


    if ff_noisy_xy_in_obs is not None:
      ff_dataframe = pd.merge(ff_dataframe, ff_noisy_xy_in_obs, how="left", on=["ff_index", "point_index"])
      ff_dataframe.loc[ff_dataframe['ff_x_noisy'].isnull(), 'ff_x_noisy'] = ff_dataframe.loc[ff_dataframe['ff_x_noisy'].isnull(), 'ff_x']
      ff_dataframe.loc[ff_dataframe['ff_y_noisy'].isnull(), 'ff_y_noisy'] = ff_dataframe.loc[ff_dataframe['ff_y_noisy'].isnull(), 'ff_y']

    # if a path is provided, then we will store the dataframe as a csv in the provided path
    if data_folder_name:
      filepath = data_folder_name + '/ff_dataframe.csv'
      os.makedirs(data_folder_name, exist_ok = True)
      ff_dataframe.to_csv(filepath)


    ff_dataframe = furnish_ff_dataframe(ff_dataframe, ff_real_position_sorted, monkey_information, ff_caught_T_sorted, ff_life_sorted)


    return ff_dataframe





def furnish_ff_dataframe(ff_dataframe, ff_real_position_sorted, monkey_information, ff_caught_T_sorted, ff_life_sorted):
    # Add some columns (they shall not be saved in csv for the sake of saving space)
    ff_dataframe['target_x'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:, 0]
    ff_dataframe['target_y'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:, 1]
    ff_dataframe['ffdistance2target'] = LA.norm(np.array(ff_dataframe[['ff_x', 'ff_y']])-np.array(ff_dataframe[['target_x', 'target_y']]), axis = 1)

    # Analyze whether ffangle is decreasing as the monkey moves
    abs_ffangle_decreasing = -np.sign(ff_dataframe['abs_delta_ff_angle'])
    ff_dataframe["abs_ffangle_decreasing"] = abs_ffangle_decreasing

    abs_ffangle_boundary_decreasing = -np.sign(ff_dataframe['abs_delta_ff_angle_boundary'])
    ff_dataframe["abs_ffangle_boundary_decreasing"] = abs_ffangle_boundary_decreasing

    # Analyze whether dw is the same direction as ffangle
    ff_dataframe["IS_TARGET"] = (ff_dataframe["ff_index"] == ff_dataframe["target_index"])
    ff_dataframe['monkey_dw'] = np.array(monkey_information['monkey_dw'].loc[np.array(ff_dataframe['point_index'])])
    ff_dataframe['ff_index_string'] = ff_dataframe['ff_index'].astype('str')
    dw_same_sign_as_ffangle = np.sign(np.multiply(np.array(ff_dataframe["monkey_dw"]), np.array(ff_dataframe["ff_angle"])))
    ff_dataframe["dw_same_sign_as_ffangle"] = dw_same_sign_as_ffangle

    dw_same_sign_as_ffangle_boundary = np.sign(np.multiply(np.array(ff_dataframe["monkey_dw"]), np.array(ff_dataframe["ff_angle_boundary"])))
    ff_dataframe["dw_same_sign_as_ffangle_boundary"] = dw_same_sign_as_ffangle_boundary

    ff_dataframe = add_caught_time_and_whether_caught_to_ff_dataframe(ff_dataframe, ff_caught_T_sorted, ff_life_sorted)

    return ff_dataframe


def add_caught_time_and_whether_caught_to_ff_dataframe(ff_dataframe, ff_caught_T_sorted, ff_life_sorted, dt=0.016):
    env_end_time = ff_life_sorted[-1, -1] + 100
    num_ff_to_be_added = len(ff_life_sorted) - len(ff_caught_T_sorted)
    ff_caught_T_sorted_extended = np.append(ff_caught_T_sorted, np.repeat(env_end_time, num_ff_to_be_added))
    ff_dataframe['caught_time'] = ff_caught_T_sorted_extended[np.array(ff_dataframe['ff_index'])]
    ff_dataframe['whether_caught'] = (np.abs(ff_dataframe['caught_time'] - ff_dataframe['time']) < 0.016+0.01).astype('int')
    return ff_dataframe


def make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_sorted, ff_flash_sorted,  
                    ff_real_position_sorted, ff_life_sorted, max_distance = 400, ff_radius = 10, 
                    data_folder_name = None, print_progress = True):

    """
    Make a dataframe called ff_dataframe that contains various information about all visible fireflies or fireflies that are invisible but alive and 
    within max distance to the monkey at each time point in the given duration. Here we assume that the player is the monkey. 
    If RL agents are used, the algorithm needs to be modified.


    Difference between ff_dataframe and ff_dataframe_v2:
    the former contains the information of fireflies when they are either visible or in memory (and also alive), while the latter contains 
    the information of fireflies when they are either visible or invisible but alive. Thus, within the same duration, ff_dataframe_v2 
    almost always contains much more information than ff_dataframe. On the other hand, ff_dataframe_v2 only collects information within a given duration,
    but ff_dataframe uses all available data.

    Parameters
    ----------
    duration: list, (2,)
      containing a starting time and and an ending time; only the time points within the duration will be evaluated
    monkey_information: df
      containing the speed, angle, and location of the monkey at various points of time
    ff_caught_T_sorted: np.array
      containing the time when each captured firefly gets captured
    ff_flash_sorted: list
      containing the time that each firefly flashes on and off
    ff_real_position_sorted: np.array
      containing the real locations of the fireflies
    ff_life_sorted: np.array
      containing the time that each firefly comes into being and gets captured 
      (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    max_distance: num
      the distance beyond which the firefly cannot be considered visible
    ff_radius: num
      the reward boundary of a firefly; the current setting of the game sets it to be 10
    data_folder_name: str, default is None
      the place to store the output as a csv
    print_progress: bool
      whether to print the progress of making ff_dataframe


    Returns
    -------
    ff_dataframe_v2: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point
      
    """   
    ff_index = []
    point_index = []
    time = []
    target_index = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    visible = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []

    # First, find all fireflies that are alive in this duration
    alive_ffs = np.array([index for index, life in enumerate(ff_life_sorted) if (life[1] >= duration[0]) and (life[0] < duration[1])])  
    for i in alive_ffs:
        # Find the corresponding information in monkey_information in the given duration:
        cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
        cum_t = np.array(monkey_information['monkey_t'].iloc[cum_indices])
        cum_mx, cum_my, cum_angle = np.array(monkey_information['monkey_x'].iloc[cum_indices]), np.array(monkey_information['monkey_y'].iloc[cum_indices]), np.array(monkey_information['monkey_angles'].iloc[cum_indices])
        
        # Find distances to ff
        distances_to_ff = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis = 1)
        valid_distance_indices = np.where(distances_to_ff < max_distance)[0]   
        angles_to_ff = basic_func.calculate_angles_to_ff_centers(ff_x=ff_real_position_sorted[i, 0], ff_y=ff_real_position_sorted[i, 1], mx=cum_mx, my=cum_my, m_angle=cum_angle)
        angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angles_to_ff, distances_to_ff=distances_to_ff)
        # Find the indices of the points where the ff is both within a max_distance and valid angles
        ff_within_range_indices = np.where((np.absolute(angles_to_boundaries) <= 2*pi/9) & (distances_to_ff < max_distance))[0]

        # Find indicies of fireflies that have been on at this time point
        ff_flash = ff_flash_sorted[i]
        overall_visible_indices = [index for index in ff_within_range_indices if (len(np.where(np.logical_and(ff_flash[:,0] <= cum_t[index], ff_flash[:,1] >= cum_t[index]))[0]) > 0)]
        # Also make sure that all these indices are within the ff's lifetime
        corresponding_time = cum_t[overall_visible_indices]
        alive_ff_indices = np.where((corresponding_time >= ff_life_sorted[i, 0]) & (corresponding_time <= ff_life_sorted[i, 1]))[0]
        overall_visible_indices = np.array(overall_visible_indices)[alive_ff_indices].tolist()


        # Append the values for this ff; Using list operations is faster than np.append here
        ff_index = ff_index + [i] * len(valid_distance_indices)
        point_index = point_index + cum_indices[valid_distance_indices].tolist()
        time = time + cum_t[valid_distance_indices].tolist()
        target_index = target_index + np.digitize(cum_t[valid_distance_indices], ff_caught_T_sorted).tolist()
        ff_x = ff_x + [(ff_real_position_sorted[i, 0])]*len(valid_distance_indices)
        ff_y = ff_y + [(ff_real_position_sorted[i, 1])]*len(valid_distance_indices)
        monkey_x = monkey_x + cum_mx[valid_distance_indices].tolist()
        monkey_y = monkey_y + cum_my[valid_distance_indices].tolist()
        visible_points_for_this_ff = np.zeros(len(cum_t))
        visible_points_for_this_ff[overall_visible_indices] = 1
        visible = visible + visible_points_for_this_ff[valid_distance_indices].astype('int').tolist()
        ff_distance = ff_distance + distances_to_ff[valid_distance_indices].tolist()
        ff_angle = ff_angle + angles_to_ff[valid_distance_indices].tolist()
        ff_angle_boundary = ff_angle_boundary + angles_to_boundaries[valid_distance_indices].tolist()

        if i % 100 == 0:
            if print_progress:
              print(i, " out of ", len(alive_ffs))

    # Now let's create a dictionary from the lists
    ff_dict = {'ff_index': ff_index, 'point_index': point_index, 'time': time, 'target_index': target_index,
              'ff_x': ff_x, 'ff_y': ff_y, 'monkey_x': monkey_x, 'monkey_y': monkey_y, 'visible': visible,
              'ff_distance': ff_distance, 'ff_angle': ff_angle, 'ff_angle_boundary': ff_angle_boundary}
      
    ff_dataframe_v2 = pd.DataFrame(ff_dict)

      
    # Add some columns
    ff_dataframe_v2['target_x'] = ff_real_position_sorted[np.array(ff_dataframe_v2['target_index'])][:, 0]
    ff_dataframe_v2['target_y'] = ff_real_position_sorted[np.array(ff_dataframe_v2['target_index'])][:, 1]
    ff_dataframe_v2['ffdistance2target'] = LA.norm(np.array(ff_dataframe_v2[['ff_x', 'ff_y']])-np.array(ff_dataframe_v2[['target_x', 'target_y']]), axis = 1)
    ff_dataframe_v2['point_index_in_duration'] = ff_dataframe_v2['point_index'] - cum_indices[0]
    ff_dataframe_v2['being_target'] = (ff_dataframe_v2['ff_index'] == ff_dataframe_v2['target_index']).astype('int')

    # Also to show whether each ff has been caught;
    # Since when using the agent, ff_caught_T_sorted does not contain information for all fireflies, we need to fill out the information for the ff not included,
    # To do so, we use the latest time in the environment plus 100s.
    ff_dataframe_v2 = add_caught_time_and_whether_caught_to_ff_dataframe(ff_dataframe_v2, ff_caught_T_sorted, ff_life_sorted)



    # if a path is provided, then we will store the dataframe as a csv in the provided path
    if data_folder_name:
        filepath = data_folder_name + '/ff_dataframe_v2.csv'
        os.makedirs(data_folder_name, exist_ok = True)
        ff_dataframe_v2.to_csv(filepath)
    return ff_dataframe_v2




    