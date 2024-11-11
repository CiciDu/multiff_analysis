import sys
from visualization import plot_behaviors_utils
from pattern_discovery import ff_dataframe_utils
from data_wrangling import basic_func


import os
import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

    



def make_point_vs_cluster(ff_dataframe, max_ff_distance_from_monkey = 500, max_cluster_distance = 100, max_time_past = 1, 
						  print_progress = True, data_folder_name = None, duration_per_step = None):
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
    if duration_per_step is None:
        duration_per_step = ff_dataframe['time'].iloc[100]-ff_dataframe['time'].iloc[99]
    max_steps_past = math.floor(max_time_past/duration_per_step)

    # Since the "memory" of a firefly is 100 when it is visible, and decrease by 1 for each step that it is not visible,
    # thus, the minimum "memory" of a firefly to be included in the consideration of whether it belongs to a cluster is
    # 100 - max_steps_apart 
    min_memory_of_ff = 100-max_steps_past

    # Initiate a list to store the result
    # Structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
    point_vs_cluster = []         

    ff_dataframe_subset = ff_dataframe[(ff_dataframe['memory']>= (min_memory_of_ff))& (ff_dataframe['ff_distance']<= max_ff_distance_from_monkey)][['point_index', 'ff_x', 'ff_y', 'ff_index', 'target_index']]


    for i in range(min_point_index, max_point_index+1):
      # Take out fireflies that meet the criteria
      selected_ff = ff_dataframe_subset[ff_dataframe_subset['point_index']==i]
      # Put their x, y coordinates into an array
      ffxy_array = selected_ff[['ff_x', 'ff_y']].to_numpy()
         
      if ffxy_array.shape[0] == 2:
         if LA.norm(ffxy_array[0,:]-ffxy_array[1,:]) <= max_cluster_distance:
            ff_indices = selected_ff[['ff_index']].to_numpy()
            trial = selected_ff[['target_index']].iloc[0].item()
            for index in ff_indices:
                # Store the firefly index with its cluster label along with the point index i and trial number 
                point_vs_cluster.append([i, int(index[0]), 0, trial])

      elif ffxy_array.shape[0] > 2:
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
          cluster_labels = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='single').fit_predict(ffxy_array)
          # Find the clusters that have more than one firefly
          unique_cluster_labels, ff_counts = np.unique(cluster_labels, return_counts=True)
          clusters_w_more_than_1_ff = unique_cluster_labels[ff_counts > 1]
          # Find the indices of the fireflies belonging to those clusters
          for index in np.isin(cluster_labels, clusters_w_more_than_1_ff).nonzero()[0]:
          	# Store the firefly index with its cluster label along with the point index i and trial number
            trial = selected_ff[['target_index']].iloc[0].item()
            point_vs_cluster.append([i, ff_indices[index].item(), cluster_labels[index], trial])

      if i % 1000 == 0:
        if print_progress == True:
          print("Progress of making point_vs_cluster: ", i, " out of ", max_point_index)


    point_vs_cluster = np.array(point_vs_cluster)

    point_vs_cluster = pd.DataFrame(point_vs_cluster, columns=['point_index', 'ff_index', 'cluster_label', 'target_index'])

    if data_folder_name is not None:
      filepath = os.path.join(data_folder_name, 'point_vs_cluster.csv')
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


def find_ff_clusters(ff_positions, ff_real_position_sorted, array_of_start_time_of_evaluation, array_of_end_time_of_evaluation, ff_life_sorted, max_distance=50,
                     empty_cluster_ok=False):
    ff_indices_of_each_cluster = []
    ff_num = len(array_of_start_time_of_evaluation)
    for i in range(ff_num):
        ff_of_interet_position = ff_positions[i]
        duration = [array_of_start_time_of_evaluation[i], array_of_end_time_of_evaluation[i]]
        alive_ff_indices, alive_ff_position = plot_behaviors_utils.find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted)
        alive_ff_indices_close_to_ff_of_interet = np.where(LA.norm(alive_ff_position.T - ff_of_interet_position, axis=1) < max_distance)[0] 
        ff_indices_close_to_ff_of_interet = alive_ff_indices[alive_ff_indices_close_to_ff_of_interet]
        if len(ff_indices_close_to_ff_of_interet) == 0:
            if not empty_cluster_ok:
                raise ValueError(f'No firefly is found within the distance of {max_distance} cm from the firefly of interest at time {array_of_start_time_of_evaluation[i]} and {array_of_end_time_of_evaluation[i]}')
        ff_indices_of_each_cluster.append(ff_indices_close_to_ff_of_interet)
    return ff_indices_of_each_cluster


def turn_list_of_ff_clusters_info_into_dataframe(ff_clusters, all_point_index):
    ff_cluster_df = pd.DataFrame({'point_index': [], 'ff_index': []})
    for i in range(len(all_point_index)):
        point_index = all_point_index[i]
        for ff in ff_clusters[i]:
            ff_cluster_df = pd.concat([ff_cluster_df, pd.DataFrame({'point_index': point_index, 'ff_index': ff}, index=[0])], ignore_index=True)
    ff_cluster_df['point_index'] = ff_cluster_df['point_index'].astype(int)
    ff_cluster_df['ff_index'] = ff_cluster_df['ff_index'].astype(int)
    return ff_cluster_df


def find_target_clusters(ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, max_distance=50):
    ff_indices_of_each_cluster = find_ff_clusters(ff_real_position_sorted, ff_real_position_sorted, ff_caught_T_sorted-10, ff_caught_T_sorted+10, ff_life_sorted, max_distance=max_distance)
    return ff_indices_of_each_cluster




def find_ff_cluster_df(ff_indices_of_each_cluster, time_of_evaluation_for_each_cluster, ff_dataframe, cluster_identifiers=None):
    # ff_indices_of_each_cluster: a list; each item in ff_indices_of_each_cluster should contain at least one ff index
    
    print("Finding information of clusters ...")   
    if cluster_identifiers is None:
        cluster_identifiers = list(range(len(time_of_evaluation_for_each_cluster)))
    row_indices = []
    nearby_alive_ff_indices = []
    ff_cluster_last_visible_time = []
    ff_cluster_last_visible_distances = []
    ff_cluster_last_visible_angles = []
    ff_cluster_last_visible_angles_to_boundary = []
    latest_visible_ff_indices = []
    ff_dataframe = ff_dataframe[ff_dataframe['visible']==1].copy()
    for i in range(len(time_of_evaluation_for_each_cluster)):
        time_of_evaluation = time_of_evaluation_for_each_cluster[i]
        ff_indices_in_a_cluster = ff_indices_of_each_cluster[i]
        ff_dataframe_subset = ff_dataframe[(ff_dataframe['time'] <= time_of_evaluation)].copy()
        ff_dataframe_subset = ff_dataframe_subset[ff_dataframe_subset['ff_index'].isin(ff_indices_in_a_cluster)]

        if len(ff_dataframe_subset) > 0:
            row_indices.append(cluster_identifiers[i])
            nearby_alive_ff_indices.append(ff_indices_in_a_cluster.tolist())
            latest_visible_ff = ff_dataframe_subset.loc[ff_dataframe_subset['time'].idxmax()]
            latest_visible_ff_indices.append(latest_visible_ff['ff_index'])
            ff_cluster_last_visible_time.append(time_of_evaluation - latest_visible_ff['time'])
            ff_cluster_last_visible_distances.append(latest_visible_ff['ff_distance'])
            ff_cluster_last_visible_angles.append(latest_visible_ff['ff_angle'])
            ff_cluster_last_visible_angles_to_boundary.append(latest_visible_ff['ff_angle_boundary'])

        if i%100==0:
            print(i, 'out of', len(time_of_evaluation_for_each_cluster))


    # make the lists above into a dataframe
    ff_cluster_df = pd.DataFrame({'cluster_identifier': row_indices, 
                                'latest_visible_ff_indices': latest_visible_ff_indices,
                                'nearby_alive_ff_indices': nearby_alive_ff_indices, 
                                'time_since_last_visible': ff_cluster_last_visible_time,
                                'last_visible_distances': ff_cluster_last_visible_distances,
                                'last_visible_angles': ff_cluster_last_visible_angles,
                                'last_visible_angles_to_boundary': ff_cluster_last_visible_angles_to_boundary})

    ff_cluster_df['abs_last_visible_angles'] = np.abs(ff_cluster_df['last_visible_angles'])
    ff_cluster_df['abs_last_visible_angles_to_boundary'] = np.abs(ff_cluster_df['last_visible_angles_to_boundary'])    
        
    return ff_cluster_df


# The function below is similar as the function above
def find_last_visible_time_of_a_cluster_before_a_time(list_of_ff_clusters, list_of_time, ff_dataframe):
    # relevant_indices indicate
    list_of_last_visible_time = []
    for i in range(len(list_of_time)):
        indices_of_ff = list_of_ff_clusters[i]
        latest_time = list_of_time[i]
        # for each set of ff_near_stops
        ff_info = ff_dataframe.loc[ff_dataframe['time'] <= latest_time]
        ff_info = ff_info[ff_info['ff_index'].isin(indices_of_ff)]
        ff_info = ff_info[ff_info['visible']==1]
        last_visible_time = ff_info.time.max()
        list_of_last_visible_time.append(last_visible_time)
    list_of_last_visible_time = np.array(list_of_last_visible_time)
    return list_of_last_visible_time



def find_target_cluster_df(ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, ff_dataframe, max_distance=50):
    
    ff_indices_of_each_cluster = find_target_clusters(ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, max_distance=max_distance)
    target_cluster_df = find_ff_cluster_df(ff_indices_of_each_cluster, time_of_evaluation_for_each_cluster=ff_caught_T_sorted, ff_dataframe=ff_dataframe, cluster_identifiers=None)
    target_cluster_df['target_index'] = target_cluster_df['cluster_identifier']
    target_cluster_df.drop(columns=['cluster_identifier'], inplace=True)
    
    return target_cluster_df
