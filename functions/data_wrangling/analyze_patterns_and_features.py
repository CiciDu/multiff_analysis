from multiff_analysis.functions.data_wrangling import basic_func, find_patterns
import os
import numpy as np
import pandas as pd
from numpy import linalg as LA
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)




def make_all_trial_patterns(caught_ff_num, n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials, ignore_sudden_flash_trials,
                            try_a_few_times_trials, give_up_after_trying_trials, cluster_around_target_trials, cluster_around_target_indices, 
                            waste_cluster_around_target_trials, data_folder_name=None):
    

		zero_array = np.zeros(caught_ff_num+1, dtype=int)

		multiple_in_a_row = np.where(n_ff_in_a_row >= 2)[0]
		# multiple_in_a_row_all means it also includes the first ff that's caught in a cluster
		multiple_in_a_row_all = np.union1d(multiple_in_a_row, multiple_in_a_row-1)
		multiple_in_a_row2 = zero_array.copy()
		multiple_in_a_row_all2 = zero_array.copy()
		multiple_in_a_row2[multiple_in_a_row] = 1
		multiple_in_a_row_all2[multiple_in_a_row_all] = 1


		two_in_a_row = np.where(n_ff_in_a_row == 2)[0]
		three_in_a_row = np.where(n_ff_in_a_row == 3)[0]
		four_in_a_row = np.where(n_ff_in_a_row == 4)[0]


		two_in_a_row2 = zero_array.copy()
		if len(two_in_a_row) > 0:
			two_in_a_row2[two_in_a_row] = 1 

		three_in_a_row2 = zero_array.copy()
		if len(three_in_a_row) > 0:
			three_in_a_row2[three_in_a_row] = 1

		four_in_a_row2 = zero_array.copy()
		if len(four_in_a_row) > 0:
			four_in_a_row2[four_in_a_row] = 1

		one_in_a_row = np.where(n_ff_in_a_row < 2)[0]
		one_in_a_row2 = zero_array.copy()
		if len(one_in_a_row) > 0:
			one_in_a_row2[one_in_a_row] = 1

		visible_before_last_one2 = zero_array.copy()
		if len(visible_before_last_one_trials) > 0:
			visible_before_last_one2[visible_before_last_one_trials] = 1

		disappear_latest2 = zero_array.copy()
		if len(disappear_latest_trials) > 0:
			disappear_latest2[disappear_latest_trials] = 1 

		ignore_sudden_flash2 = zero_array.copy()
		if len(ignore_sudden_flash_trials) > 0:
			ignore_sudden_flash2[ignore_sudden_flash_trials] = 1

		try_a_few_times2 = zero_array.copy()
		if len(try_a_few_times_trials) > 0:
			try_a_few_times2[try_a_few_times_trials] = 1

		give_up_after_trying2 = zero_array.copy()
		if len(give_up_after_trying_trials) > 0:
			give_up_after_trying2[give_up_after_trying_trials] = 1

		cluster_around_target2 = zero_array.copy()
		if len(cluster_around_target_trials) > 0:
			cluster_around_target2[cluster_around_target_trials] = 1 

		waste_cluster_around_target2 = zero_array.copy()
		if len(waste_cluster_around_target_trials) > 0:
			waste_cluster_around_target2[waste_cluster_around_target_trials] = 1


		all_trial_patterns_dict = {
		# bool
		'two_in_a_row': two_in_a_row2,
		'three_in_a_row': three_in_a_row2,
		'four_in_a_row': four_in_a_row2,
		'one_in_a_row': one_in_a_row2,
		'multiple_in_a_row': multiple_in_a_row2,
		'multiple_in_a_row_all': multiple_in_a_row_all2,
		'visible_before_last_one': visible_before_last_one2,
		'disappear_latest': disappear_latest2,
		'ignore_sudden_flash': ignore_sudden_flash2,
		'try_a_few_times': try_a_few_times2,
		'give_up_after_trying': give_up_after_trying2,
		'cluster_around_target': cluster_around_target2,
		'waste_cluster_around_target': waste_cluster_around_target2}

		for key, value in all_trial_patterns_dict.items():
			all_trial_patterns_dict[key] = value[:-1]

		                                  
		all_trial_patterns = pd.DataFrame(all_trial_patterns_dict) 

		if data_folder_name:
			basic_func.save_df_to_csv(all_trial_patterns, 'all_trial_patterns', data_folder_name)


		return all_trial_patterns






def make_pattern_frequencies(all_trial_patterns, ff_caught_T_sorted, monkey_information, data_folder_name=None):
		pattern_frequencies = pd.DataFrame([], columns=['Item', 'Frequency', 'N_total', 'Rate', 'Group'])
		n_trial_counted = len(all_trial_patterns)-1
		caught_ff_num = len(ff_caught_T_sorted)
		n_ff_counted = len(ff_caught_T_sorted) - 1

		group = 1
		for item in all_trial_patterns.columns:
		    frequency = all_trial_patterns[item].sum()
		    if item == 'three_in_a_row':
		    	n_trial = n_trial_counted-1
		    elif item == 'four_in_a_row':
		    	n_trial = n_trial_counted-2
		    elif item == 'cluster_around_target': # cause for this category, we actually didn't count the 1st ff (when time = ff_caught_T_sorted[0])
		    	n_trial = n_trial_counted+1
		    else:
		    	n_trial = n_trial_counted
		    		
		    new_row = pd.DataFrame({'Item': item, 'Frequency': frequency,
		                            'N_total': n_trial, 'Rate': frequency/n_trial, 'Group': group}, index=[0])
		    pattern_frequencies = pd.concat([pattern_frequencies, new_row])

		

		total_duration = (ff_caught_T_sorted[-1]-ff_caught_T_sorted[0]) # we don't consider the duration before the first ff was caught
		ff_capture_rate = n_ff_counted/total_duration
		new_row = pd.DataFrame({'Item': 'ff_capture_rate', 'Frequency': n_ff_counted, 'N_total': total_duration, 
													  'Rate': ff_capture_rate, 'Group': 2}, index=[0])
		pattern_frequencies = pd.concat([pattern_frequencies, new_row])



		num_stops_array = [len(basic_func.find_stops(i, ff_caught_T_sorted, monkey_information)) for i in range(1, caught_ff_num)]
		total_number_of_stops = sum(num_stops_array)
		stop_success_rate = n_ff_counted/total_number_of_stops
		new_row = pd.DataFrame({'Item': 'stop_success_rate', 'Frequency': n_ff_counted, 'N_total': total_number_of_stops, 
														'Rate': stop_success_rate, 'Group': 2}, index=[0])
		pattern_frequencies = pd.concat([pattern_frequencies, new_row]).reset_index(drop=True)
		pattern_frequencies = pattern_frequencies.reset_index(drop=True)


		pattern_frequencies['Label'] = 'Missing'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'two_in_a_row', 'Label'] = 'Two in a row'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'visible_before_last_one', 'Label'] = 'Visible before last capture'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'disappear_latest', 'Label'] = 'Target disappears latest'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'waste_cluster_around_target', 'Label'] = 'Waste cluster around last target'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'ignore_sudden_flash', 'Label'] = 'Ignore sudden flash'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'give_up_after_trying', 'Label'] = 'Give up after trying'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'try_a_few_times', 'Label'] = 'Try a few times'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'ff_capture_rate', 'Label'] = 'Firefly capture rate (per s)'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'stop_success_rate', 'Label'] = 'Stop success rate'
		
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'three_in_a_row', 'Label'] = 'Three in a row'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'four_in_a_row', 'Label'] = 'Four in a row'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'one_in_a_row', 'Label'] = 'One in a row'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'multiple_in_a_row', 'Label'] = 'Multiple in a row'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'multiple_in_a_row_all', 'Label'] = 'Multiple in a row including 1st one'
		pattern_frequencies.loc[pattern_frequencies['Item'] == 'cluster_around_target', 'Label'] = 'Cluster exists around last target'


		pattern_frequencies['Percentage'] = pattern_frequencies['Rate']*100

		if data_folder_name:
			basic_func.save_df_to_csv(pattern_frequencies, 'pattern_frequencies', data_folder_name)

		return pattern_frequencies





def make_all_trial_features(ff_dataframe, monkey_information, ff_caught_T_sorted, cluster_around_target_indices, ff_believed_position_sorted, player = "monkey", max_cluster_distance = 50, data_folder_name = None):
		"""
	  Make a dataframe called all_trial_features that includes some information about each trial

	  Parameters
	  ---------- 
	  ff_dataframe: pd.dataframe
	  		containing various information about all visible or "in-memory" fireflies at each time point
	  monkey_information: df
	      containing the speed, angle, and location of the monkey at various points of time
	  ff_caught_T_sorted: np.array
	      containing the time when each captured firefly gets captured
	  ff_believed_position_sorted: np.array
	      containing the locations of the monkey (or agent) when each captured firefly was captured    
	  max_cluster_distance: numeric
	      the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster
	  player: str
	      "monkey" or "agent" 
	  data_folder_name: str
	      name or path of the folder to store the output as csv
	  
	  Returns
	  -------
	  all_trial_features: dataframe containing various characteristics of each trial, with the following columns:
	      'trial': trial number
	      't': duration of a trial 
	      't_last_visible': duration since the target or a firefly near the target (within 25 cm) is last seen
	      'd_last_visible': distance from the target since the target or a nearby firefly is last seen
	      'abs_angle_last_visible': angle to the reward boundary of the target since the target or a nearby firefly is last seen
	      'hitting_arena_edge': whether the monkey/agent hits the boundary during a trial
	      'num_stops': number of stops made during a trial 
	      'num_stops_since_last_visible': number of stops made since the target or a nearby firefly is last seen
	      'num_stops_near_target': number of stops made made near the target (the closest stop should be within 50 cm, or max_cluster_distance, of the target)
	      'n_ff_in_a_row': number of fireflies the monkey/agent has caught in a row after catching the current target

	  """

		caught_ff_num = len(ff_caught_T_sorted)
		# Make an array of trial numbers. Trial number starts at 1 and is named after the index of the target. 
		trial_array = [i for i in range(1, caught_ff_num)]

		# Find the duration of each trial
		t_array = ff_caught_T_sorted[1:caught_ff_num] - ff_caught_T_sorted[:caught_ff_num-1]
		t_array = t_array.tolist()

		# Question: How long can the monkey remember a target?
		# For each trial, find the time that elapses between the target last being visible and its capture.
		# Also find the distance and angle of the target when the target is last visible.
		t_last_visible = []
		d_last_visible = []
		abs_angle_last_visible = []
		# Take out the information about visible fireflies
		visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]
		for i in range(1, caught_ff_num):
		  # Because sometimes the monkey aims for a firefly near the target but ends up catching the target, 
		  # we also consider fireflies near the targets along with the targets
		  info_of_nearby_ff = ((visible_ff['target_index']==i) & (visible_ff['ffdistance2target'] < 25))
		  info_of_target = (visible_ff['ff_index']==i)
		  relevant_df = visible_ff[ info_of_nearby_ff | info_of_target]
		  if len(relevant_df) > 0:
		    t_last_visible.append(ff_caught_T_sorted[i] - max(np.array(relevant_df.time)))
		    d_last_visible.append(max(np.array(relevant_df.ff_distance)))
		    abs_angle_last_visible.append(max(np.absolute(np.array(relevant_df.ff_angle_boundary))))
		  else:
		    t_last_visible.append(9999)
		    d_last_visible.append(9999)
		    abs_angle_last_visible.append(9999)



		# Create an array that shows whether the monkey has hit the boundary at least once during each trial
		hitting_arena_edge = []
		for i in range(1, caught_ff_num):
		  duration = [ff_caught_T_sorted[i-1], ff_caught_T_sorted[i]]
		  cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
		  if len(cum_indices) > 1:
		    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_indices]), np.array(monkey_information['monkey_y'].iloc[cum_indices])
		    if np.any(cum_mx[1:]-cum_mx[:-1] > 55) or np.any(cum_my[1:]-cum_my[:-1] > 55):
		      hitting_arena_edge.append(1)
		    else:
		      hitting_arena_edge.append(0)
		  else:
		    hitting_arena_edge.append(0)


		# Create arrays about the number of stops made by the agent during each trial
		num_stops_array = [len(basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player)) for i in range(1, caught_ff_num)]
		# Also count the number of stops since the target (or a nearby firefly) is last seen
		num_stops_since_last_visible = [len(basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player, since_target_last_seen = True, t_last_visible = t_last_visible)) for i in range(1, caught_ff_num)]


		num_stops_near_target = []
		for i in range(1, caught_ff_num):
		  clusters = basic_func.put_stops_into_clusters(i, max_cluster_distance, ff_caught_T_sorted, monkey_information, player = player)
		  # Find the locations of the stops
		  distinct_stops = basic_func.find_stops(i, ff_caught_T_sorted, monkey_information, player = player)
		  if len(distinct_stops) > 0:
		    # If the last stop is close enough to the believed position of the target
		    if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < max_cluster_distance:
		      # Append the number of stops in the last cluster
		      num_stops_near_target.append(clusters.count(clusters[-1]))
		    else: 
		      num_stops_near_target.append(0)
		  else: 
		    num_stops_near_target.append(0)

		n_ff_in_a_row = find_patterns.n_ff_in_a_row_func(ff_believed_position_sorted[1:caught_ff_num], distance_between_ff = 50)

		num_ff_around_target = np.array([len(unit) for unit in cluster_around_target_indices[1:caught_ff_num]])

		# Put all the information first into a dictionary and then into a dataframe
		trials_dict = {'trial': trial_array, 
		               't': t_array,  
		               't_last_visible': t_last_visible,
		               'd_last_visible': d_last_visible,
		               'abs_angle_last_visible': abs_angle_last_visible, 
		               'hitting_arena_edge': hitting_arena_edge,
		               'num_stops': num_stops_array, 
		               'num_stops_since_last_visible': num_stops_since_last_visible,
		               'num_stops_near_target': num_stops_near_target,
		               'num_ff_around_target': num_ff_around_target.tolist(),
		               'n_ff_in_a_row': n_ff_in_a_row[:len(trial_array)].tolist()                            
		                      }
		all_trial_features = pd.DataFrame(trials_dict)
		
		if data_folder_name:
			basic_func.save_df_to_csv(all_trial_features, 'all_trial_features', data_folder_name)
		return all_trial_features





def make_feature_statistics(all_trial_features, data_folder_name=None):
		feature_statistics = pd.DataFrame([], columns=['Item', 'Median', 'Mean', 'N_trial'])
		all_trial_features_valid = all_trial_features[(all_trial_features['t_last_visible'] < 50) & (all_trial_features['hitting_arena_edge']==False)].reset_index()
		all_trial_features_valid = all_trial_features_valid.drop(columns={'index', 'trial'})
		median_values = all_trial_features_valid.median(axis=0)
		mean_values = all_trial_features_valid.mean(axis=0)
		n_trial = len(all_trial_features_valid)
		for item in median_values.index:
		    median = median_values[item]
		    mean = mean_values[item]
		    new_row = pd.DataFrame({'Item': item, 'Median': median, 'Mean': mean, 'N_trial': n_trial}, index=[0])
		    feature_statistics = pd.concat([feature_statistics, new_row])

		feature_statistics = feature_statistics.reset_index(drop=True)


		feature_statistics['Label'] = 'Missing'
		feature_statistics.loc[feature_statistics['Item'] == 't', 'Label'] = 'time'
		feature_statistics.loc[feature_statistics['Item'] == 't_last_visible', 'Label'] = 'time target last seen'
		feature_statistics.loc[feature_statistics['Item'] == 'd_last_visible', 'Label'] = 'distance target last seen'
		feature_statistics.loc[feature_statistics['Item'] == 'abs_angle_last_visible', 'Label'] = 'abs angle target last seen'
		feature_statistics.loc[feature_statistics['Item'] == 'num_stops', 'Label'] = 'num stops'
		feature_statistics.loc[feature_statistics['Item'] == 'num_stops_near_target', 'Label'] = 'num stops near target'
		
		feature_statistics.loc[feature_statistics['Item'] == 'hitting_arena_edge', 'Label'] = 'hitting arena edge'
		feature_statistics.loc[feature_statistics['Item'] == 'num_stops_since_last_visible', 'Label'] = 'num stops target last seen'
		feature_statistics.loc[feature_statistics['Item'] == 'n_ff_in_a_row', 'Label'] = 'num ff caught in a row'
		feature_statistics.loc[feature_statistics['Item'] == 'num_ff_around_target', 'Label'] = 'num ff around target'


		feature_statistics['Label for median'] = 'Median ' + feature_statistics['Label']
		feature_statistics['Label for mean'] = 'Mean ' + feature_statistics['Label']


		if data_folder_name:
			basic_func.save_df_to_csv(feature_statistics, 'feature_statistics', data_folder_name)
		  
		return feature_statistics




def combine_df_of_agent_and_monkey(df_m, df_a, agent_names = ["Agent", "Agent2", "Agent3"], df_a2 = None, df_a3 = None):
    """
    Make a dataframe that combines df such as df from the monkey and the agent(s);
    This function can include df from up to three agents. 

    Parameters
    ---------- 
    df_a: dict
    containing a df derived from the agent data
    df_m: dict
    containing a df derived from themonkey data
    agent_names: list, optional
    names of the agents used to identify the agents, if more than one agent is used
    df_a: dict, optional
    containing a df derived from the 2nd agent's data
    df_a: dict, optional
    containing a df derived from the 3rd agent's data

    Returns
    -------
    merged_df: dataframe that combines df from the monkey and the agent(s)

    """   


    df_a['Player'] = agent_names[0]
    df_m['Player'] = 'Monkey'

    if df_a3:
        # Then a 2nd agent and a 3rd agent are used
        df_a2['Player'] = agent_names[1]
        df_a3['Player'] = agent_names[2]
        merged_df = pd.concat([df_a, df_m, df_a2, df_a3], axis=0)
    elif df_a2:
        # Then a 2nd agent is used
        df_a2['Player'] = agent_names[1]
        merged_df = pd.concat([df_a, df_m, df_a2], axis=0)
    else:
        merged_df = pd.concat([df_a, df_m], axis=0)

    merged_df = merged_df.reset_index()

    return merged_df





def add_dates_based_on_data_names(df):
    all_dates = [int(date[-4:]) for date in df['Data'].tolist()]
    all_dates = [datetime.strptime(str(date/100), '%m.%d').date() for date in all_dates]
    df['Date'] = all_dates
    df = df.sort_values(by='Date')
    return df








# maybe instead of doing it time point by time point, one can do it trial by trial

def add_target_last_seen_info_to_target_df(target_df, ff_dataframe, alive_ff_indices_around_targets=None, use_target_cluster=False,
                                              include_frozen_info=False):
    target_df['target_last_seen_time'] = 100
    target_df['target_last_seen_distance'] = 400
    target_df['target_last_seen_angle'] = 0
    target_df['target_last_seen_angle_to_boundary'] = 0

    if include_frozen_info:
        target_df['target_last_seen_distance_frozen'] = 400
        target_df['target_last_seen_angle_frozen'] = 0
        target_df['target_last_seen_angle_to_boundary_frozen'] = 0

    if use_target_cluster:
        print("Adding target-cluster-last-seen info to target_df...")
        if alive_ff_indices_around_targets is None:
            raise ValueError("alive_ff_indices_around_targets is None, but use_target_cluster is True")

    for target_index in np.sort(ff_dataframe['target_index'].unique()):
        print('target_index = %d' % target_index, end='\r')
        # get target_info which is about the current target_index
        if use_target_cluster:
            target_cluster_indices = alive_ff_indices_around_targets[target_index] 
            target_info = ff_dataframe[(ff_dataframe['ff_index'].isin(target_cluster_indices)) & (ff_dataframe['visible'] == 1)].copy()
        else:  
            target_info = ff_dataframe[(ff_dataframe['ff_index'] == target_index) & (ff_dataframe['visible'] == 1)].copy()

        # also get target_df_sub which is the subset in target_df associated with the current target_index
        target_df_idx = np.array((target_df['target_index'] == target_index)).nonzero()[0]
        target_df_sub = target_df.iloc[target_df_idx].copy()



        if len(target_df_sub) > 0:
            unique_points = target_df_sub[['point_index', 'time', 'monkey_x', 'monkey_y', 'monkey_angle']].drop_duplicates().sort_values(by='point_index')
            min_point = unique_points.point_index.min()
            target_info = target_info.sort_values(by='point_index')
            if len(target_info) > 0:
                target_info_before = target_info[target_info['point_index'] <= min_point]
                # if there's information about target before unique_points, then attach the last row to the beginning of unique_points
                if len(target_info_before) > 0:
                    target_info = target_info[target_info['point_index'] >= target_info_before.iloc[-1].point_index] 
                    starting_info = target_info_before.iloc[[-1]][['point_index', 'time', 'monkey_x', 'monkey_y', 'monkey_angle']]
                    unique_points = pd.concat([starting_info, unique_points])
                    # but also make sure there's no duplicate
                    unique_points = unique_points.drop_duplicates(subset=['point_index'], keep='first')
                else: # we eliminate the part in unique_points that don't have any target info preceding it or at the same time
                    # those will stay in the original dataframe as default values
                    valid_points = np.where(np.array(unique_points['point_index']) >= target_info.point_index.min())[0]
                    unique_points = unique_points.iloc[valid_points]
                    target_df_idx = target_df_idx[valid_points]
                    target_df_sub = target_df_sub.iloc[valid_points]
                    #print("unique_points truncated: ", unique_points)
                
                # we need to make sure that for every point_index, there's no duplicate value
                target_info = target_info.sort_values(by=['point_index', 'ff_index']).drop_duplicates(subset=['point_index'], keep='first')
                target_info = target_info[['point_index', 'time', 'ff_x', 'ff_y', 'monkey_x', 'monkey_y', 'monkey_angle']]
                target_info = target_info.rename(columns={'time':'target_time', 'ff_x':'target_x', 'ff_y':'target_y',
                                                        'monkey_x': 'frozen_monkey_x', 'monkey_y': 'frozen_monkey_y', 'monkey_angle': 'frozen_monkey_angle'})


                # merge the dataframes 
                unique_points = unique_points.merge(target_info, how='left', on='point_index')
                unique_points.fillna(method='ffill', inplace=True)
                # calculate the desired values
                target_x, target_y = unique_points.target_x, unique_points.target_y
                monkey_x, monkey_y, monkey_angle = unique_points.monkey_x, unique_points.monkey_y, unique_points.monkey_angle
                target_angle = basic_func.calculate_angles_to_ff_centers(ff_x=target_x, ff_y=target_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
                target_distance = np.sqrt((target_x - monkey_x)**2 + (target_y - monkey_y)**2)
                unique_points['target_last_seen_time'] = unique_points.time - unique_points.target_time
                unique_points['target_last_seen_distance'] = target_distance
                unique_points['target_last_seen_angle'] = target_angle
                unique_points['target_last_seen_angle_to_boundary'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=target_angle, distances_to_ff=target_distance)         
                if include_frozen_info:
                    monkey_x, monkey_y, monkey_angle = unique_points.frozen_monkey_x, unique_points.frozen_monkey_y, unique_points.frozen_monkey_angle
                    target_angle = basic_func.calculate_angles_to_ff_centers(ff_x=target_x, ff_y=target_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
                    target_distance = np.sqrt((target_x - monkey_x)**2 + (target_y - monkey_y)**2)
                    unique_points['target_last_seen_distance_frozen'] = target_distance
                    unique_points['target_last_seen_angle_frozen'] = target_angle
                    unique_points['target_last_seen_angle_to_boundary_frozen'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=target_angle, distances_to_ff=target_distance)  
                    essential_columns = ['point_index', 'target_last_seen_time', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary', \
                                        'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen']
                else:
                    essential_columns = ['point_index', 'target_last_seen_time', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary']
                unique_points = unique_points[essential_columns]

                target_df_sub_new = target_df_sub[['point_index']].merge(unique_points, on='point_index', how='left')
                target_df.iloc[target_df_idx, target_df.columns.get_indexer(target_df_sub_new.columns)]  = target_df_sub_new.values


    return target_df