from multiff_analysis.functions.data_wrangling import basic_func
from multiff_analysis.functions.data_visualization import plot_behaviors
import os
import numpy as np
import pandas as pd
import math
import collections
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
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


    



def make_point_vs_cluster(ff_dataframe, max_ff_distance_from_monkey = 250, max_cluster_distance = 100, max_time_past = 1, 
						  print_progress = True, data_folder_name = None):
    """
    Find trials where the target has disappeared the latest among all visible fireflies during a trial


    Parameters
    ----------
    ff_dataframe: pd.dataframe
    	containing various information about all visible or "in-memory" fireflies at each time point
  	max_ff_distance_from_monkey: numeric
  		the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to a cluster 
  	max_cluster_distance: numeric
  		the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster
  	max_time_past: numeric
  		how long a firefly can be stored in memomry after it becomes invisible for it to be included in the consideration of whether it belongs to a cluster
  	print_progress: bool
  		whether to print the progress of making point_vs_cluster

    Returns
    -------
    point_vs_cluster: array 
    	contains indices of fireflies belonging to a cluster at each time point
    	structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
        

    """

    # Find the beginning and ending of the indices of all steps
    min_point_index = np.min(np.array(ff_dataframe['point_index']))
    max_point_index = np.max(np.array(ff_dataframe['point_index']))
    
    # Convert max_time_past to max_steps_past
    duration_per_step = ff_dataframe['time'][100]-ff_dataframe['time'][99]
    max_steps_past = math.floor(max_time_past/duration_per_step)

    # Since the "memory" of a firefly is 100 when it is visible, and decrease by 1 for each step that it is not visible,
    # thus, the minimum "memory" of a firefly to be included in the consideration of whether it belongs to a cluster is
    # 100 - max_steps_apart 
    min_memory_of_ff = 100-max_steps_past

    # Initiate a list to store the result
    # Structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
    point_vs_cluster = []         

    ff_dataframe_subset = ff_dataframe[(ff_dataframe['memory']> (min_memory_of_ff))& (ff_dataframe['ff_distance']<max_ff_distance_from_monkey)][['point_index', 'ff_x', 'ff_y', 'ff_index']]

    for i in range(min_point_index, max_point_index+1):
      # Take out fireflies that meet the criteria
      selected_ff = ff_dataframe_subset[ff_dataframe_subset['point_index']==i][['ff_x', 'ff_y', 'ff_index']]
      # Put their x, y coordinates into an array
      ffxy_array = selected_ff[['ff_x', 'ff_y']].to_numpy()
      if len(ffxy_array) > 1:
      	# Find their indices
        ff_indices = selected_ff[['ff_index']].to_numpy()

        # Use the function scipy.cluster.hierarchy.linkage
        # method = "single" means that we're using the Nearest Point Algorithm
        # In the returned array, each row is a procedure of combining two components into one cluster (the closer two components are, the sooner they'll be combined); 
        # there are n-1 rows in total in the returned variable
        linked = linkage(ffxy_array, method='single')

        # In order to find the number of clusters that satisfy our requirements, we'll first find the number of rows from linked
        # where the two components of the cluster are greater than max_cluster_distance (100 cm as default). 
        # If this number is n, then it means that at one point in the procedure, there are n+1 clusters that all satisfy our requirements, 
        # and then these clusters are condensed into 1 big cluster through combining two clusters into one and repearting the procedure;
        # However, these final steps of combining are invalid for our use since all the n+1 clusters are further apart than max_cluster_distance;
        # Thus, we know that we can separate the fireflies into at most n+1 clusters that satisfy our requirement
        # that each firefly is within 50 cm of its nearest firefly in the cluster.
        num_clusters = sum(linked[:, 2] > max_cluster_distance)+1  

        # If the number of clusters is smaller than the number of fireflies, then there is at least one cluster that has two or more fireflies
        if num_clusters < len(ff_indices):
          # This time, we assign the fireflies into the number of clusters we just found, so that we know which cluster each firefly belongs to.
          # The variable "cluster_labels" contains the cluster label for each firefly
          cluster_labels = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single').fit_predict(ffxy_array)
          # Find the clusters that have more than one firefly
          unique_cluster_labels, ff_counts = np.unique(cluster_labels, return_counts=True)
          clusters_w_more_than_1_ff = unique_cluster_labels[ff_counts > 1]
          # Find the indices of the fireflies belonging to those clusters
          for index in np.isin(cluster_labels, clusters_w_more_than_1_ff).nonzero()[0]:
          	# Store the firefly index with its cluster label along with the point index i
            point_vs_cluster.append([i, ff_indices[index].item(), cluster_labels[index]])
      
      if i % 1000 == 0:
        if print_progress == True:
          print("Progress of making point_vs_cluster: ", i, " out of ", max_point_index)
    
    point_vs_cluster = np.array(point_vs_cluster)

    filepath = data_folder_name + '/point_vs_cluster.csv'
    os.makedirs(data_folder_name, exist_ok = True)
    np.savetxt(filepath, point_vs_cluster, delimiter=',')
    return point_vs_cluster




def clusters_of_ffs_func(point_vs_cluster, monkey_information, ff_caught_T_sorted):
	"""
  Find clusters of fireflies that appear during a trial based on point_vs_cluster

  Parameters
  ----------
  point_vs_cluster: array 
  	contains indices of fireflies belonging to a cluster at each time point
  	structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
  monkey_information: df
    containing the speed, angle, and location of the monkey at various points of time
  ff_caught_T_sorted: np.array
    containing the time when each captured firefly gets captured

  Returns
  -------
	cluster_exist_trials: array
		trial numbers of the trials where at least one cluster exists
	cluster_dataframe_point: dataframe
		information of the clusters for each time point that has at least one cluster
	cluster_dataframe_trial: dataframe
		information of the clusters for each trial that has at least one cluster
  """

  # Turn point_vs_cluster from np.array into a dataframe
	temp_dataframe1 = pd.DataFrame(point_vs_cluster, columns=['point_index', 'ff_index', 'cluster_label'])
	# Find indices of unique points and their counts and make them into a dataframe as well
	unique_points, counts = np.unique(point_vs_cluster[:, 0], return_counts=True)
	temp_dataframe2 = pd.DataFrame(np.concatenate([unique_points.reshape(-1, 1), counts.reshape(-1, 1)], axis=1),
								   columns=['point_index', 'num_ff_at_point'])
	# Combine the information of the above 2 dataframes
	temp_dataframe3 = temp_dataframe1.merge(temp_dataframe2, how="left", on="point_index")
	# Find the corresponding time to all the points
	corresponding_t = monkey_information['monkey_t'].values[np.array(temp_dataframe3['point_index'])]
	temp_dataframe3['time'] = corresponding_t
	# From the time of each point, find the target index that corresponds to that point
	temp_dataframe3['target_index'] = np.digitize(corresponding_t, ff_caught_T_sorted)
	# Only keep the part of the data up to the capture of the last firefly
	temp_dataframe3 = temp_dataframe3[temp_dataframe3['target_index'] < len(ff_caught_T_sorted)]
	# Thus we have the information of the clusters for each time point that has at least one cluster
	cluster_dataframe_point = temp_dataframe3
	# By grouping the information into trials, we can have the information of the clusters for each trial that has at least one cluster;
	# For each trial, we'll have the maximum number of fireflies in all clusters as well as the number of clusters
	cluster_dataframe_trial = cluster_dataframe_point[['target_index', 'num_ff_at_point']].groupby('target_index',
									as_index=True).agg({'num_ff_at_point': ['max', 'count']})
	cluster_dataframe_trial.columns = ["max_ff_in_cluster", "num_points_w_cluster"]
	# We can also take out the trials during which at least one cluster exists
	cluster_exist_trials = np.array(cluster_dataframe_point.target_index.unique())
	return cluster_exist_trials, cluster_dataframe_point, cluster_dataframe_trial



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
    num_ff = len(close_ff_indices)
    cluster_around_target.append(num_ff)
    if num_ff > 0:
      cluster_around_target_positions.append(past_visible_ff_positions[close_ff_indices])
      cluster_around_target_indices.append(past_visible_ff_indices[close_ff_indices])
    else:
      cluster_around_target_positions.append(np.array([]))
      cluster_around_target_indices.append(np.array([]))
  cluster_around_target = np.array(cluster_around_target)
  cluster_around_target_trials = np.where(cluster_around_target > 0)[0]
  return cluster_around_target_trials, cluster_around_target_indices, cluster_around_target_positions




def try_a_few_times_func(ff_caught_T_sorted, monkey_information, ff_believed_position_sorted, 
						 player, max_point_index, max_cluster_distance = 50):
  """
  Find the trials where the monkey has stopped more than one times to catch a target

  Parameters
  ----------
  ff_caught_T_sorted: np.array
    containing the time when each captured firefly gets captured
  monkey_information: df
    containing the speed, angle, and location of the monkey at various points of time
  ff_believed_position_sorted: np.array
    containing the locations of the monkey (or agent) when each captured firefly was captured 
  player: str
    "monkey" or "agent"  
  max_point_index: numeric
  	the maximum point_index in ff_dataframe      
  max_cluster_distance: numeric
  	the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster

    
  Returns
  -------
  try_a_few_times_trials: array
    trials that can be categorized as "try a few times"
  try_a_few_times_indices: array
  	indices of moments that can be categorized as "try a few times"
  try_a_few_times_indices_for_anim: array
  	indices of monkey_information that can be annotated as "try a few times" during animation; the difference between this variable and give_up_after_trying_indices 
  	is that the this variable supplies 20 points before and after each intervals to make the annotations last longer and easier to read

  """
  try_a_few_times_trials = []
  try_a_few_times_indices = []
  try_a_few_times_indices_for_anim = []
  caught_ff_num = len(ff_caught_T_sorted)
  for i in range(1, caught_ff_num): 
    # Find clusters based on a distance of max_cluster_distance (default is 50 cm)
    clusters = basic_func.put_stops_into_clusters(i, max_cluster_distance, ff_caught_T_sorted, monkey_information, player = player)
    # if there is at least one cluster:
    if len(clusters) > 0:
    	# We are only interested in the last cluster 
      label_of_last_cluster = clusters[-1]
      # If the last cluster has more than 2 stops
      if clusters.count(label_of_last_cluster) > 1:
        # Find the locations of all the stops during the trial
        distinct_stops = basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player)
        # Also find the indices (in monkey_information) of these stops
        distinct_stops_indices = basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player, return_index = True)
        # If the last stop is close enough to the believed position of the target, then we know that the last cluster of stops 
        # are likely aimed towards the target; then, the trial can be categorized as "try a few times"
        if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < max_cluster_distance:
          # Store by trials
          try_a_few_times_trials.append(i)
          # Store by indices (in regards to monkey_information)
          num_stops_in_last_cluster = clusters.count(label_of_last_cluster)
          min_index = distinct_stops_indices[-num_stops_in_last_cluster]
          max_index = distinct_stops_indices[-1]
          try_a_few_times_indices = try_a_few_times_indices + list(range(min_index, max_index+1))
          try_a_few_times_indices_for_anim = try_a_few_times_indices_for_anim + list(range(min_index-20, max_index+21))

  try_a_few_times_trials = np.array(try_a_few_times_trials)
  try_a_few_times_indices = np.array(try_a_few_times_indices)
  try_a_few_times_indices_for_anim = np.array(try_a_few_times_indices_for_anim)
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  try_a_few_times_indices = try_a_few_times_indices[try_a_few_times_indices < max_point_index]
  try_a_few_times_indices_for_anim = try_a_few_times_indices_for_anim[try_a_few_times_indices_for_anim < max_point_index]

  return try_a_few_times_trials, try_a_few_times_indices, try_a_few_times_indices_for_anim



def give_up_after_trying_func(ff_caught_T_sorted, monkey_information, ff_believed_position_sorted, player, max_point_index, max_cluster_distance = 50):
  """
  Find the trials where the monkey has stopped more than once to catch a firefly but failed to succeed, and the monkey gave up

  Parameters
  ----------
  ff_caught_T_sorted: np.array
      containing the time when each captured firefly gets captured
  monkey_information: df
      containing the speed, angle, and location of the monkey at various points of time
  ff_believed_position_sorted: np.array
      containing the locations of the monkey (or agent) when each captured firefly was captured 
  player: str
      "monkey" or "agent"  
  max_point_index: numeric
	   the maximum point_index in ff_dataframe      
  max_cluster_distance: numeric
	   the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster

  
  Returns
  -------
  give_up_after_trying_trials: array
      trials that can be categorized as "give up after trying"
  give_up_after_trying_indices: array
      indices in monkey_information that can be categorized as "give up after trying"
  give_up_after_trying_indices_for_anim: array
      indices of monkey_information that can be annotated as "give up after trying" during animation; the difference between this variable and give_up_after_trying_indices 
      is that the this variable supplies 20 points before and after each intervals to make the annotations last longer and easier to read

  """

  # General strategy: find the trials where there at least one cluster has more than 2 stops, and this cluster is neither at the beginning nor at the end
  give_up_after_trying_trials = []
  give_up_after_trying_indices = []
  give_up_after_trying_indices_for_anim = []
  caught_ff_num = len(ff_caught_T_sorted)
  for i in range(1, caught_ff_num):
    # Find clusters based on a distance of max_cluster_distance
    clusters = basic_func.put_stops_into_clusters(i, max_cluster_distance, ff_caught_T_sorted, monkey_information, player = player)
    # if clusters is not empty:
    if len(clusters) > 0:
      clusters_counts = collections.Counter(clusters) # count the number of elements in each cluster
      distinct_stop_positions = basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player)
      distinct_stops_indices = basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player, return_index = True)
      last_cluster_label = clusters[-1]
      for k in range(1, last_cluster_label + 1):  # for each cluster
        # if the cluster has more than one element:
        if clusters_counts[k] > 1:
          # Get positions of these points
          stop_positions = [distinct_stop_positions[index] for index, value in enumerate(clusters) if value == k]
          # If the first stop is not close to beginning, and the last stop is not too close to the end:
          if LA.norm(stop_positions[0]-ff_believed_position_sorted[i-1]) > max_cluster_distance and LA.norm(stop_positions[-1]-ff_believed_position_sorted[i]) > max_cluster_distance:
            # Store the trial number
            give_up_after_trying_trials.append(i)
            # Store indices
            stop_positions_indices = [distinct_stops_indices[index] for index, value in enumerate(clusters) if value == k]
            #give_up_after_trying_indices = give_up_after_trying_indices + stop_positions_indices
            give_up_after_trying_indices = give_up_after_trying_indices + list(range(stop_positions_indices[0], stop_positions_indices[1]))
            give_up_after_trying_indices_for_anim = give_up_after_trying_indices_for_anim + list(range(min(stop_positions_indices)-20, max(stop_positions_indices)+21))
            	
  give_up_after_trying_trials = np.unique(np.array(give_up_after_trying_trials))
  give_up_after_trying_indices = np.array(give_up_after_trying_indices)
  give_up_after_trying_indices_for_anim = np.array(give_up_after_trying_indices_for_anim)
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  give_up_after_trying_indices = give_up_after_trying_indices[give_up_after_trying_indices < max_point_index]
  give_up_after_trying_indices_for_anim = give_up_after_trying_indices_for_anim[give_up_after_trying_indices_for_anim < max_point_index]

  return give_up_after_trying_trials, give_up_after_trying_indices, give_up_after_trying_indices_for_anim



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
  cum_ff_distances = df_ffdistance[ignore_sudden_flash]

  valid_indices = np.where(cum_target_distances > cum_ff_distances)

  # Only keep the trials where the suddenly visible firefly is closer than the target
  cum_target_indices = np.array(ff_dataframe.target_index)[ignore_sudden_flash]
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
  






def make_target_closest_or_target_angle_smallest(ff_dataframe, max_point_index, column='ff_distance'):
    # Here we use aim to make target_closest, but the same function can be applied to make target_angle_smallest
    # The algorithm has been verified by seeing if the individual components (subsets of indices)can add up to the whole (all indices)
    # Starting from the first point index

    # make an array such as target_closest:
    # 2 means target is the closest ff at that point (visible or in memory)
    # 1 means the target is not the closest. In the subset of 1:
        # 1 means both the target and a non-target are visible or in memory (which we call present)
        # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
        # -1 means both the target and other ff are neither visible or in memory

    key_columns = ff_dataframe[['point_index', 'ff_index', 'target_index', column]]
    min_distance_subset = ff_dataframe[['point_index', column]].groupby('point_index').min().reset_index()
    min_distance_subset = pd.merge(min_distance_subset, key_columns, on=['point_index', column], how="left")

    # From min_distance_subset, find the rows that belong to targets and non-targets respectively.
    # Below are point indices where the target is the closest; they shall be denoted as 2
    target_closest_point_indices = np.array(min_distance_subset[min_distance_subset['ff_index'] == min_distance_subset['target_index']].point_index)
    # Below are point indices where the target is not the closest; they the complement of target_closest_point_indices
    target_not_closest_point_indices = np.delete(np.arange(max_point_index+1), target_closest_point_indices)
    # Also find the indices where the non-target is both present and closest
    non_target_present_and_closest_indices = np.array(min_distance_subset[min_distance_subset['ff_index'] != min_distance_subset['target_index']].point_index)

    # Among target_not_closest_point_indices: 
    # Find the indices where both the target and other ff are neither visible or in memory.
    # They shall be denoted -1
    both_absent_indices = np.delete(np.arange(max_point_index+1), np.array(min_distance_subset.point_index))
    # Then, among target_not_closest_point_indices, find the ones that the non-target is present and closest to the monkey, while target is also present;
    # First find all the indices in ff_dataframe where target is present; and then use intersection 
    # They shall be denoted as 1
    target_present_indices = np.array(ff_dataframe[ff_dataframe['ff_index']==ff_dataframe['target_index']]['point_index'])
    both_present_and_non_target_closest_indices = np.intersect1d(non_target_present_and_closest_indices, target_present_indices)
    # Finally, find the indices where where non-target is present and closest to the monkey, while target is absent; 
    # They shall be denoted as 0
    only_non_target_present_indices = np.setdiff1d(non_target_present_and_closest_indices, both_present_and_non_target_closest_indices)

    # Lastly, make result array (such as target_closest)
    result = np.arange(max_point_index+1)
    result[target_closest_point_indices] = 2
    result[both_present_and_non_target_closest_indices] = 1
    result[only_non_target_present_indices] = 0
    result[both_absent_indices] = -1

    return result



def make_target_closest(ff_dataframe, max_point_index, data_folder_name=None):
    # make target_closest:
    # 2 means target is the closest ff at that point (visible or in memory)
    # 1 means the target is not the closest. In the subset of 1:
        # 1 means both the target and a non-target are visible or in memory (which we call present)
        # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
        # -1 means both the target and other ff are neither visible or in memory

    target_closest = make_target_closest_or_target_angle_smallest(ff_dataframe, max_point_index, column='ff_distance')
    if data_folder_name:
        np.savetxt(data_folder_name + '/target_closest.csv', target_closest.tolist(), delimiter=',')
    return target_closest



def make_target_angle_smallest(ff_dataframe, max_point_index, data_folder_name=None):
    # make target_angle_smallest:
    # 2 means target is has the smallest absolute angle at that point (visible or in memory)
    # 1 means the target does not have the smallest absolute angle. In the subset of 1:
        # 1 means both the target and a non-target are visible or in memory (which we call present)
        # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
        # -1 means both the target and other ff are neither visible or in memory

    ff_dataframe['ff_angle_boundary_abs'] = np.abs(np.array(ff_dataframe['ff_angle_boundary']))
    target_angle_smallest = make_target_closest_or_target_angle_smallest(ff_dataframe, max_point_index, column='ff_angle_boundary_abs')
    if data_folder_name:
        np.savetxt(data_folder_name + '/target_angle_smallest.csv', target_angle_smallest.tolist(), delimiter=',')
    return target_angle_smallest




def find_cluster_surrounding_targets(ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, ff_dataframe, max_distance=50, return_index_only=False):
    print("Finding clusters surrounding targets...")
    target_indices = []
    list_of_alive_ff_indices_around_targets = [] # this is useful if return_index_only is True
    alive_ff_indices_around_targets = []
    target_cluster_last_visible_time = []
    target_cluster_last_visible_distances = []
    target_cluster_last_visible_angles = []
    target_cluster_last_visible_angles_to_boundary = []
    caught_ff_num = len(ff_caught_T_sorted)
    for i in range(caught_ff_num):
        target_position = ff_real_position_sorted[i]
        target_caught_T = ff_caught_T_sorted[i]
        duration = [target_caught_T-30, target_caught_T+5]
        alive_ff_indices, alive_ff_position = plot_behaviors.find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted)
        alive_ff_indices_close_to_target = np.where(LA.norm(alive_ff_position.T - target_position, axis=1) < max_distance)[0] 
        ff_indices_close_to_target = alive_ff_indices[alive_ff_indices_close_to_target]
        if return_index_only:
            list_of_alive_ff_indices_around_targets.append(ff_indices_close_to_target)
        else:
            ff_dataframe_subset = ff_dataframe[ff_dataframe['visible']==1].copy()
            ff_dataframe_subset = ff_dataframe_subset[(ff_dataframe_subset['time'] <= target_caught_T)]
            ff_dataframe_subset = ff_dataframe_subset[ff_dataframe_subset['ff_index'].isin(ff_indices_close_to_target)]
            
            if len(ff_dataframe_subset) > 0:
                target_indices.append(i)
                alive_ff_indices_around_targets.append(ff_indices_close_to_target)
                latest_visible_ff = ff_dataframe_subset.loc[ff_dataframe_subset['time'].idxmax()]
                target_cluster_last_visible_time.append(target_caught_T-latest_visible_ff['time'])
                target_cluster_last_visible_distances.append(latest_visible_ff['ff_distance'])
                target_cluster_last_visible_angles.append(latest_visible_ff['ff_angle'])
                target_cluster_last_visible_angles_to_boundary.append(latest_visible_ff['ff_angle_boundary'])

        if i%100==0:
            print(i, 'out of', caught_ff_num)

    if return_index_only:
        return list_of_alive_ff_indices_around_targets

    # make the lists above into a dataframe
    target_cluster_df = pd.DataFrame({'target_index': target_indices, 
                                      'alive_ff_indices_around_targets': alive_ff_indices_around_targets, 
                                      'time_since_last_visible': target_cluster_last_visible_time,
                                      'last_visible_distances': target_cluster_last_visible_distances,
                                      'last_visible_angles': target_cluster_last_visible_angles,
                                      'last_visible_angles_to_boundary': target_cluster_last_visible_angles_to_boundary})
    
    target_cluster_df['abs_last_visible_angles'] = np.abs(target_cluster_df['last_visible_angles'])
    target_cluster_df['abs_last_visible_angles_to_boundary'] = np.abs(target_cluster_df['last_visible_angles_to_boundary'])    
        
    return target_cluster_df



def find_points_w_more_than_n_ff(ff_dataframe, monkey_information, ff_caught_T_sorted, n=2, n_max=None):
    # Pull out all the segments where we can see aligned ff vs non-aligned ff
    # Find segments involving the flashing on of >= 3 firefly (preferably 2ff vs 2ff on each side) and until 1.6s after.
    point_vs_num_ff = ff_dataframe[['point_index', 'ff_index']].groupby('point_index').nunique()
    point_vs_num_ff = point_vs_num_ff.rename(columns={'ff_index':'num_ff'})
    point_vs_num_ff.loc[:, 'point_index'] = point_vs_num_ff.index
    points_w_more_than_n_ff = point_vs_num_ff[point_vs_num_ff['num_ff'] > n].copy()
    if n_max is not None:
        points_w_more_than_n_ff = points_w_more_than_n_ff[points_w_more_than_n_ff['num_ff'] <= n_max].copy()
    # Eliminate the points before the capture of the 1st firefly
    valid_earliest_point = np.where(monkey_information['monkey_t'] > ff_caught_T_sorted[0])[0][0]
    points_w_more_than_n_ff = points_w_more_than_n_ff[points_w_more_than_n_ff['point_index'] >= valid_earliest_point].copy()
    diff = np.diff(points_w_more_than_n_ff['point_index'])
    points_w_more_than_n_ff['diff'] = np.append(0, diff) # in the array, numbers not equal to 1 are starts of a chunk
    points_w_more_than_n_ff['diff_2'] = np.append(diff, 0) # in the array, numbers not equal to 1 are ends of a chunk
    points_w_more_than_n_ff['diff'] = points_w_more_than_n_ff['diff'].astype(int)
    points_w_more_than_n_ff['diff_2'] = points_w_more_than_n_ff['diff_2'].astype(int)
    points_w_more_than_n_ff['chunk'] = (points_w_more_than_n_ff['diff'] != 1).cumsum()
    points_w_more_than_n_ff['chunk'] = points_w_more_than_n_ff['chunk']-1

    return points_w_more_than_n_ff


def increase_durations_between_points(df, min_duration=5):
    new_df = pd.DataFrame(columns=df.columns)
    prev_row = df.iloc[0]
    for index, row in df.iterrows():
      if (index==0) or (row.time - prev_row.time) > min_duration:
        new_df = pd.concat([new_df, pd.DataFrame(row).T], ignore_index=True)
        prev_row = row
    new_df = new_df.reset_index(drop=True)
    return new_df

def find_changing_dw_points(chunk_df, monkey_information, ff_caught_T_sorted, chunk_interval=10, minimum_time_before_capturing=0.5):
    first_point = chunk_df['point_index'].min()
    duration = [monkey_information['monkey_t'][first_point], monkey_information['monkey_t'][first_point]+chunk_interval]
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0] 

    # Take out the part right before catching a ff
    # First find ff caught in the interval and a little beyond the interval
    relevant_ff_caught_T = ff_caught_T_sorted[(ff_caught_T_sorted >= duration[0]) & (ff_caught_T_sorted <= duration[1]+minimum_time_before_capturing)]
    for time in relevant_ff_caught_T:
        duration_to_take_out = [time-minimum_time_before_capturing, time]        
        # Take out corresponding indices from cum_indices
        cum_indices = cum_indices[~((cum_indices >= duration_to_take_out[0]) & (cum_indices <= duration_to_take_out[1]))]

    cum_t = np.array(monkey_information['monkey_t'].iloc[cum_indices])
    cum_dw, cum_ddw = np.array(monkey_information['monkey_dw'].iloc[cum_indices]), np.array(monkey_information['monkey_ddw'].iloc[cum_indices])
    cum_abs_ddw = np.abs(cum_ddw)
    changing_dw_info = pd.DataFrame({'relative_point_index': np.where(cum_abs_ddw > 0.15)[0]})
    # find the first point of each sequence of consecutive points
    changing_dw_info['group'] = np.append(0, (np.diff(changing_dw_info['relative_point_index'])!=1).cumsum())
    changing_dw_info = changing_dw_info.groupby('group').min()
    relative_point_index = changing_dw_info['relative_point_index'].astype(int)
    changing_dw_info['point_index'] = cum_indices[relative_point_index]
    changing_dw_info['time'] = cum_t[relative_point_index]
    changing_dw_info['dw'] = cum_dw[relative_point_index]
    changing_dw_info['ddw'] = cum_ddw[relative_point_index]
    return changing_dw_info

def decrease_overlaps_between_chunks(points_w_more_than_n_ff, monkey_information, min_interval_between_chunks):
    temp_df = points_w_more_than_n_ff.groupby('chunk').min()
    temp_df['time'] = monkey_information['monkey_t'].values[temp_df['point_index'].values]
    temp_df = temp_df.reset_index()[['chunk', 'time']]
    new_df = increase_durations_between_points(temp_df)
    new_chunks = new_df['chunk'].astype('int')
    points_w_more_than_n_ff = points_w_more_than_n_ff[points_w_more_than_n_ff['chunk'].isin(new_chunks)].copy()

    # reset the chunk number so it starts from 0 again
    points_w_more_than_n_ff['chunk'] = (points_w_more_than_n_ff['diff'] != 1).cumsum()
    points_w_more_than_n_ff['chunk'] = points_w_more_than_n_ff['chunk']-1

    return points_w_more_than_n_ff





