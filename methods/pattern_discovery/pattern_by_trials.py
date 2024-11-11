import sys
from data_wrangling import basic_func
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials

import os
import numpy as np
import pandas as pd
import collections
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

    



def n_ff_in_a_row_func(ff_believed_position_sorted, distance_between_ff = 50):
	"""
  For each captured firefly, find how many fireflies have been caught in a row.
  For every two consequtive fireflies to be considered caught in a row, 
  they should not be more than 50 cm (or "distance_between_ff" cm) apart


  Parameters
  ----------
  caught_ff_num: numeric
  	total number of caught firefies
  ff_believed_position_sorted: np.array
    containing the locations of the monkey (or agent) when each captured firefly was captured 
  distance_between_ff: numeric
  	the maximum distance between two consecutive fireflies for them to be considered as caught in a row
  
  Returns
  -------
  n_ff_in_a_row: array
    containing one integer for each captured firefly to indicate how many fireflies have been caught in a row.
    n_ff_in_a_row[k] will denote the number of ff that the monkey has captured in a row at trial k

  """
  # For the first caught firefly, it is apparent that only 1 firefly has been caught in a row
	n_ff_in_a_row = [1]
	# Keep a count of how many fireflies have been caught in a row
	count = 1
	caught_ff_num = len(ff_believed_position_sorted)
	for i in range(1, caught_ff_num):
	  if LA.norm(ff_believed_position_sorted[i]-ff_believed_position_sorted[i-1]) < distance_between_ff:
	    count += 1
	  else:
	  	# Restarting from 1
	    count = 1
	  n_ff_in_a_row.append(count)
	n_ff_in_a_row = np.array(n_ff_in_a_row)
	return n_ff_in_a_row



def on_before_last_one_func(ff_flash_end_sorted, ff_caught_T_sorted, caught_ff_num):
	"""
  Find the trials where the current target has only flashed on before the capture of the previous target;
  In other words, the target hasn’t flashed on during the trial

  Parameters
  ----------
  ff_flash_end_sorted: np.array
      containing the last moment that each firefly flashes on
  ff_caught_T_sorted: np.array
      containing the time when each captured firefly gets captured
  caught_ff_num: numeric
      total number of caught firefies

  
  Returns
  -------
  on_before_last_one_trials: array
      trial numbers that can be categorized as "on before last one"

  """
	on_before_last_one_trials = [] 
	for i in range(1, caught_ff_num):
	  # Evaluate whether the last flash of the current ff finishes before the capture of the previous ff
	  if ff_flash_end_sorted[i] < ff_caught_T_sorted[i-1]:
	    # If the monkey captures 2 fireflies at the same time, then the trial does not count as "on_before_last_one"
	    if ff_caught_T_sorted[i] == ff_caught_T_sorted[i-1]:
	       continue
	    # Otherwise, append the trial number into the list
	    on_before_last_one_trials.append(i)
	on_before_last_one_trials = np.array(on_before_last_one_trials)
	return on_before_last_one_trials



def visible_before_last_one_func(ff_dataframe):
	"""
  Find the trials where the current target has only been visible on before the capture of the previous target;
  In other words, the target hasn’t been visible during the trial;
  Here, a firefly is considered visible if it satisfies: (1) flashes on, (2) Within 40 degrees to the left and right,
  (3) Within 400 cm to the monkey (the distance can be updated when the information of the actual experiment is available)

  Parameters
  ----------
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point

  Returns
  -------
  visible_before_last_one_trials: array
      trial numbers that can be categorized as "visible before last one"

  """
  # We first take out the trials that cannot be categorized as "visible before last one";
  # For these trials, the target has been visible for at least one time point during the trial 
	temp_dataframe = ff_dataframe[(ff_dataframe['target_index'] == ff_dataframe['ff_index']) & (ff_dataframe['visible'] == 1)]
	trials_not_to_select = np.unique(np.array(temp_dataframe['target_index']))
	# Get the numbers for all trials
	all_trials = np.unique(np.array(ff_dataframe['target_index']))
	# Using the difference to get the trials of interest
	visible_before_last_one_trials = np.setdiff1d(all_trials, trials_not_to_select)
	return visible_before_last_one_trials



def find_target_cluster_visible_before_last_one(target_cluster_df, ff_caught_T_sorted):
    target_cluster_df['caught_time'] = ff_caught_T_sorted[target_cluster_df.target_index]
    target_cluster_df['prev_caught_time'] = ff_caught_T_sorted[target_cluster_df.target_index-1]
    target_cluster_df.loc[0, 'prev_caught_time'] = 0
    target_cluster_df['last_visible_time'] = target_cluster_df['caught_time'] - target_cluster_df['time_since_last_visible']
    target_cluster_df['trial_duration'] = target_cluster_df['caught_time'] - target_cluster_df['prev_caught_time']

    target_cluster_VBLO = target_cluster_df[target_cluster_df['last_visible_time'] < target_cluster_df['prev_caught_time']-0.1]
    target_cluster_VBLO = target_cluster_VBLO[target_cluster_VBLO['caught_time'] != target_cluster_VBLO['prev_caught_time']]
    target_cluster_VBLO = target_cluster_VBLO[target_cluster_VBLO['trial_duration'] < 25]

    #target_cluster_VBLO[['target_index', 'time_since_last_visible', 'last_visible_time', 'caught_time', 'prev_caught_time']]
    return target_cluster_VBLO


    


def cluster_around_target_func(ff_dataframe, caught_ff_num, ff_caught_T_sorted, ff_real_position_sorted, 
            max_time_apart = 1.25, max_ff_distance_from_monkey = 250, max_ff_distance_from_target = 50):
  """
  Find the trials where the target is within a cluster, as well as the locations of the fireflies in the cluster


  Parameters
  ----------
  ff_dataframe: pd.dataframe
    containing various information about all visible or "in-memory" fireflies at each time point
  caught_ff_num: numeric
    total number of caught firefies
  ff_caught_T_sorted: np.array
    containing the time when each captured firefly gets captured
  ff_real_position_sorted: np.array
    containing the real locations of the fireflies
  max_time_apart: numeric
    how long a firefly can be stored in memomry after it becomes invisible for it to be included in the consideration of whether it belongs to a cluster
  max_ff_distance_from_monkey: numeric
    the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to the cluster of the target
  max_ff_distance_from_target: numeric
    the maximum distance a firefly can be from the target to be included in the consideration of whether it belongs to the cluster 

  Returns
  -------
  cluster_around_target_trials: array
    trials where a cluster exists around the target about the time of capture
  cluster_around_target_indices: list
    for each trial, it contains the indices of fireflies around the target; 
    it contains an empty array when there is no firefly around the target
  cluster_around_target_positions: list
    positions of the fireflies around the target for each trial 
    (similarly, if there's none for trial i, then cluster_around_target_positions[i] is an empty numpy array)

  """
  cluster_around_target = []
  cluster_around_target_indices = []
  cluster_around_target_positions = [] 
  temp_frame = ff_dataframe[['ff_index', 'target_index', 'ff_distance', 'visible', 'time']]

  # For each trial
  for i in range(caught_ff_num):
    # Take out the time of that the target is captured
    time = ff_caught_T_sorted[i]
    # Set the duration such that only fireflies visible in this duration will be included in the consideration of 
    # whether it belongs to the same cluster as the target
    duration = [time-max_time_apart, time+max_time_apart]
    temp_frame2 = temp_frame[(temp_frame['time'] > duration[0])&(temp_frame['time'] < duration[1])]
    # We also don't want to include the previous target and the next target into the consideration
    target_nums = np.arange(i-1, i+2)
    temp_frame2 = temp_frame2[~temp_frame2['ff_index'].isin(target_nums)]
    # Lastly, we want to make sure that these fireflies are visible
    temp_frame2 = temp_frame2[(temp_frame2['visible'] == 1)]
    temp_frame2 = temp_frame2[temp_frame2['ff_distance'] < max_ff_distance_from_monkey]
    # Take out the indices
    past_visible_ff_indices = np.unique(np.array(temp_frame2.ff_index))
    # Get positions of these ffs
    if len(past_visible_ff_indices) == 0:
      cluster_around_target.append(0)
      cluster_around_target_indices.append(np.array([]))
      cluster_around_target_positions.append(np.array([]))
      continue
    
    past_visible_ff_positions = ff_real_position_sorted[past_visible_ff_indices]
    # See if any one of it is within max_ff_distance_from_target (50 cm by default) of the target
    distance2target = LA.norm(past_visible_ff_positions - ff_real_position_sorted[i], axis=1)
    close_ff_indices = np.where(distance2target < max_ff_distance_from_target)[0]
    num_alive_ff = len(close_ff_indices)
    cluster_around_target.append(num_alive_ff)
    if num_alive_ff > 0:
      cluster_around_target_positions.append(past_visible_ff_positions[close_ff_indices])
      cluster_around_target_indices.append(past_visible_ff_indices[close_ff_indices])
    else:
      cluster_around_target_positions.append(np.array([]))
      cluster_around_target_indices.append(np.array([]))
  cluster_around_target = np.array(cluster_around_target)
  cluster_around_target_trials = np.where(cluster_around_target > 0)[0]
  return cluster_around_target_trials, cluster_around_target_indices, cluster_around_target_positions





def disappear_latest_func(ff_dataframe):
	"""
  Find trials where the target has disappeared the latest among all visible fireflies during a trial

  Parameters
  ----------
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point
  
  Returns
  -------
  disappear_latest_trials: array
      trial numbers that can be categorized as "disappear latest"

  """
	ff_dataframe_visible = ff_dataframe[(ff_dataframe['visible'] == 1)]
	# For each trial, find out the point index where the monkey last sees a ff
	last_visible_index = ff_dataframe_visible[['point_index', 'target_index']].groupby('target_index').max()
	# Take out all the rows corresponding to these points
	last_visible_ffs = pd.merge(last_visible_index, ff_dataframe_visible, how="left")
	# Select trials where the target disappears the latest
	disappear_latest_trials = np.array(last_visible_ffs[last_visible_ffs['target_index']==last_visible_ffs['ff_index']]['target_index'])
	return disappear_latest_trials


def ignore_sudden_flash_func(ff_dataframe, ff_real_position_sorted, max_point_index, max_ff_distance_from_monkey = 50):
  """
  Find the trials where a firefly other than the target or the next target suddenly becomes visible, is within in 
  50 cm (or max_ff_distance_from_monkey) of the monkey, and is closer than the target.


  Parameters
  ----------
  ff_dataframe: pd.dataframe
    containing various information about all visible or "in-memory" fireflies at each time point
  ff_real_position_sorted: np.array
    containing the real locations of the fireflies
  max_point_index: numeric
    the maximum point_index in ff_dataframe      
  max_ff_distance_from_monkey: numeric
    the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to the cluster of the target

  Returns
  -------
  ignore_sudden_flash_trials: array
    trials that can be categorized as "ignore sudden flash"
  ignore_sudden_flash_indices: array
    indices of ff_dataframe that can be categorized as "ignore sudden flash"
  ignore_sudden_flash_indices_for_anim: array
    indices of monkey_information that can be annotated as "ignore sudden flash" during animation; the difference between this variable and the previous one 
    is that the the current variable supplies 121 points (2s in the original dataset) after each intervals to make the annotations last longer and easier to read


  """
  # These are the indices in ff_dataframe where a ff changes from being invisible to become visible
  start_index1 = np.where(np.ediff1d(np.array(ff_dataframe['visible'])) == 1)[0]+1
  # These are the indices in ff_dataframe where the ff_index has changed, meaning that an invisible firefly has become visible
  start_index2 = np.where(np.ediff1d(np.array(ff_dataframe['ff_index']))!= 0)[0]+1
  # Combine the two to get the indices in ff_dataframe where a ff suddenly becomes visible
  start_index3 = np.concatenate((start_index1, start_index2))
  start_index = np.unique(start_index3)

  # Among those points, take out those where the distance between the monkey and the firefly is smaller than max_ff_distance_from_monkey
  df_ffdistance = np.array(ff_dataframe['ff_distance'])
  sudden_flash_index = start_index[np.where(df_ffdistance[start_index] < 50)]



  # On the other hand, we find the indices of ff_dataframe where the suddenly visible ff is the target or next target
  condition = np.logical_or((np.array(ff_dataframe['ff_index'])[sudden_flash_index] == np.array(ff_dataframe['target_index'])[sudden_flash_index]), 
                (np.array(ff_dataframe['ff_index'])[sudden_flash_index] == np.array(ff_dataframe['target_index']+1)[sudden_flash_index]))



  # # If we're interested in finding the cases where the suddenly visible fireflies are captured in the current or the next trial:
  # capture_sudden_flash = sudden_flash_index[condition]
  # capture_sudden_flash_trials = np.array(ff_dataframe['target_index'])[capture_sudden_flash]
  # capture_sudden_flash_trials = np.unique(capture_sudden_flash_trials)


  # Thus, we can find the indices of ff_dataframe where the suddenly visible ff is not the target
  ignore_sudden_flash = sudden_flash_index[~condition]

  # Find the distance from the monkey to the target at these points
  cum_target_distances = np.array(ff_dataframe['ffdistance2target'])[ignore_sudden_flash]
  # And find the distance from the monkey to the suddenly visible fireflies
  cum_ff_distance = df_ffdistance[ignore_sudden_flash]

  valid_indices = np.where(cum_target_distances > cum_ff_distance)

  # Only keep the trials where the suddenly visible firefly is closer than the target
  cum_target_indices = np.array(ff_dataframe['target_index'])[ignore_sudden_flash]
  ignore_sudden_flash_trials = np.unique(cum_target_indices[valid_indices])


  # Find both the target index (which is the trial number) and the corresponding ff index
  # This can be used if I want to color the ignored ffs when making visualizations or animations
  ignored_ff_target_pairs = ff_dataframe.iloc[ignore_sudden_flash][['ff_index', 'target_index', 'ff_distance', 'ff_angle', 'ff_angle_boundary']].drop_duplicates()
  ignored_ff_target_pairs = ignored_ff_target_pairs.set_index('target_index')


  # Find the indices in ff_dataframe corresponding to such a sudden flash (only storing the suddenly flashing moments)
  ignore_sudden_flash_indices = np.array(ff_dataframe['point_index'])[ignore_sudden_flash[valid_indices]]


  ignore_sudden_flash_indices_for_anim = []
  for i in ignore_sudden_flash_indices:
    # Append each point into a list and the following n points so that the message can be visible for 2 seconds
    ignore_sudden_flash_indices_for_anim = ignore_sudden_flash_indices_for_anim+list(range(i, i+121))
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  ignore_sudden_flash_indices = ignore_sudden_flash_indices[ignore_sudden_flash_indices < max_point_index]
  ignore_sudden_flash_indices_for_anim = np.array(ignore_sudden_flash_indices_for_anim)
  ignore_sudden_flash_indices_for_anim = ignore_sudden_flash_indices_for_anim[np.where(ignore_sudden_flash_indices_for_anim <= max_point_index)]

  return ignore_sudden_flash_trials, ignore_sudden_flash_indices, ignore_sudden_flash_indices_for_anim, ignored_ff_target_pairs




def whether_current_and_last_targets_are_captured_simultaneously(trial_number_arrays, ff_caught_T_sorted):
    if len(trial_number_arrays) > 0:
        dif_time = ff_caught_T_sorted[trial_number_arrays] - ff_caught_T_sorted[trial_number_arrays-1]
        trial_number_arrays_simul = trial_number_arrays[np.where(dif_time <= 0.1)[0]]
        trial_number_arrays_non_simul = trial_number_arrays[np.where(dif_time > 0.1)[0]]
    else:
        trial_number_arrays_simul = np.array([])
        trial_number_arrays_non_simul = np.array([])
    return trial_number_arrays_simul, trial_number_arrays_non_simul
  









