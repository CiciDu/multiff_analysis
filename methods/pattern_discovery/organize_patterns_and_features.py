from data_wrangling import specific_utils, general_utils
from pattern_discovery import pattern_by_trials
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from pattern_discovery import pattern_by_trials, cluster_analysis
from planning_analysis.show_planning import alt_ff_utils, show_planning_utils

import os
import numpy as np
import pandas as pd
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



def make_pattern_frequencies(all_trial_patterns, ff_caught_T_new, monkey_information, 
                             GUAT_w_ff_frequency, one_stop_w_ff_frequency,
                             data_folder_name=None):
    pattern_frequencies = pd.DataFrame([], columns=['Item', 'Frequency', 'N_total', 'Rate', 'Group'])
    n_ff_counted = len(ff_caught_T_new) - 1

    for item in all_trial_patterns.columns:
        frequency = all_trial_patterns[item].sum()
        new_row = pd.DataFrame({'Item': item,
                                'Frequency': frequency}, index=[0])
        pattern_frequencies = pd.concat([pattern_frequencies, new_row])

    # Customize n_trials for each pattern
    pattern_frequencies['N_total'] = len(all_trial_patterns) - 1

    # Specific customizations
    pattern_frequencies.loc[pattern_frequencies['Item'].isin(['cluster_around_target', 'disappear_latest']), 'N_total'] = len(all_trial_patterns)
    pattern_frequencies.loc[pattern_frequencies['Item'] == 'three_in_a_row', 'N_total'] = len(all_trial_patterns) - 2
    pattern_frequencies.loc[pattern_frequencies['Item'] == 'four_in_a_row', 'N_total'] = len(all_trial_patterns) - 3
    pattern_frequencies.loc[pattern_frequencies['Item'].isin(['waste_cluster_around_target', 'use_cluster']), 'N_total'] = all_trial_patterns['cluster_around_target'].sum()
    pattern_frequencies.loc[pattern_frequencies['Item'] == 'ignore_sudden_flash', 'N_total'] = all_trial_patterns['sudden_flash'].sum()

    # Adjust GUAT and TAFT
    n_total = GUAT_w_ff_frequency + one_stop_w_ff_frequency + all_trial_patterns['try_a_few_times'].sum()
    pattern_frequencies.loc[pattern_frequencies['Item'] == 'give_up_after_trying', 'Frequency'] = GUAT_w_ff_frequency
    pattern_frequencies.loc[pattern_frequencies['Item'].isin(['give_up_after_trying', 'try_a_few_times']), 'N_total'] = n_total

    # Calculate Rate
    pattern_frequencies['Rate'] = pattern_frequencies['Frequency'] / pattern_frequencies['N_total']
    pattern_frequencies['Percentage'] = pattern_frequencies['Rate']*100
    pattern_frequencies['Group'] = 1

    # Calculate Firefly capture rate
    total_duration = (ff_caught_T_new[-1]-ff_caught_T_new[0]) # we don't consider the duration before the first ff was caught
    ff_capture_rate = n_ff_counted/total_duration
    new_row = pd.DataFrame({'Item': 'ff_capture_rate', 'Frequency': n_ff_counted, 'N_total': total_duration, 
                             'Rate': ff_capture_rate, 'Group': 2}, index=[0])
    pattern_frequencies = pd.concat([pattern_frequencies, new_row])

    # Calculate Stop success rate
    monkey_sub = monkey_information = monkey_information[monkey_information['whether_new_distinct_stop'] == True]
    monkey_sub = monkey_sub[monkey_sub['time'].between(ff_caught_T_new[0], ff_caught_T_new[-1])]
    total_number_of_stops = len(monkey_sub)
    stop_success_rate = n_ff_counted/total_number_of_stops
    new_row = pd.DataFrame({'Item': 'stop_success_rate', 'Frequency': n_ff_counted, 'N_total': total_number_of_stops, 
                            'Rate': stop_success_rate, 'Group': 2}, index=[0])
    pattern_frequencies = pd.concat([pattern_frequencies, new_row]).reset_index(drop=True)

    # Calculate GUAT/TAFT rate
    GUAT_over_TAFT = GUAT_w_ff_frequency / all_trial_patterns['try_a_few_times'].sum()
    new_row = pd.DataFrame({'Item': 'GUAT_over_TAFT', 'Frequency': GUAT_w_ff_frequency, 'N_total': all_trial_patterns['try_a_few_times'].sum(), 
                             'Rate': GUAT_over_TAFT, 'Group': 2}, index=[0])
    pattern_frequencies = pd.concat([pattern_frequencies, new_row]).reset_index(drop=True)


    # Define the mapping of Item to Label
    item_to_label = {
        'two_in_a_row': 'Two in a row',
        'visible_before_last_one': 'Visible before last capture',
        'disappear_latest': 'Target disappears latest',
        'use_cluster': 'Use cluster near target',
        'waste_cluster_around_target': 'Waste cluster around target',
        'ignore_sudden_flash': 'Ignore sudden flash',
        'sudden_flash': 'Sudden flash',
        'give_up_after_trying': 'Give up after trying',
        'try_a_few_times': 'Try a few times',
        'ff_capture_rate': 'Firefly capture rate (per s)',
        'stop_success_rate': 'Stop success rate',
        'three_in_a_row': 'Three in a row',
        'four_in_a_row': 'Four in a row',
        'one_in_a_row': 'One in a row',
        'multiple_in_a_row': 'Multiple in a row',
        'multiple_in_a_row_all': 'Multiple in a row including 1st one',
        'cluster_around_target': 'Cluster exists around target',
        'GUAT_over_TAFT': 'GUAT over TAFT'
    }

    # Apply the mapping to the Label column
    pattern_frequencies['Label'] = pattern_frequencies['Item'].map(item_to_label).fillna('Missing')

    pattern_frequencies['N_total'] = pattern_frequencies['N_total'].astype(int)


    if data_folder_name:
        general_utils.save_df_to_csv(pattern_frequencies, 'pattern_frequencies', data_folder_name)


    return pattern_frequencies


def get_num_stops_array(monkey_information, array_of_trials):
    monkey_information = monkey_information[monkey_information['whether_new_distinct_stop'] == True].copy()
    monkey_sub = monkey_information[['trial', 'point_index']].groupby('trial').count().reset_index().rename(columns={'point_index': 'num_stops'})
    monkey_sub = monkey_sub.merge(pd.DataFrame({'trial': array_of_trials}), on='trial', how='right')
    num_stops_array = monkey_sub['num_stops'].fillna(0).values.astype(int)
    return num_stops_array


def _calculate_trial_durations(ff_caught_T_new):
    """
    Calculate the durations of trials based on the sorted capture times.

    Parameters:
    ff_caught_T_new (list or array-like): Sorted capture times.

    Returns:
    tuple: A tuple containing the trial array and the duration array.
    """
    caught_ff_num = len(ff_caught_T_new)
    trial_array = list(range(caught_ff_num))
    
    # Calculate the differences between consecutive capture times
    t_array = (ff_caught_T_new[1:caught_ff_num] - ff_caught_T_new[:caught_ff_num-1]).tolist()
    
    # Add the first capture time to the beginning of the duration array
    t_array.insert(0, ff_caught_T_new[0])
    
    return trial_array, t_array


def _calculate_hitting_arena_edge(monkey_information, ff_caught_T_new):
    # Group by 'trial' and get the maximum 'crossing_boundary' value for each trial
    cb_df = monkey_information[['trial', 'crossing_boundary']].groupby('trial').max().reset_index()
    # Create a DataFrame with 'trial' values ranging from 0 to the length of 'ff_caught_T_new'
    trial_df = pd.DataFrame({'trial': np.arange(len(ff_caught_T_new))})
    # Merge the DataFrames and fill missing values with 0
    merged_df = cb_df.merge(trial_df, on='trial', how='right').fillna(0)
    hitting_arena_edge = merged_df['crossing_boundary'].values.astype(int)
    return hitting_arena_edge


def _calculate_num_stops_since_last_vis(monkey_information, caught_ff_num, t_last_vis):
    # note that t_last_vis starts with trial = 1
    t_last_vis = np.array(t_last_vis)
    monkey_sub = monkey_information[monkey_information['whether_new_distinct_stop']==True]
    monkey_sub = monkey_sub[monkey_sub['trial'] >= 1].copy()
    monkey_sub['t_last_vis'] = t_last_vis[monkey_sub['trial'].values - 1]
    monkey_sub = monkey_sub[monkey_sub['time'] >= monkey_sub['t_last_vis']]
    num_stops_since_last_vis = _use_merge_to_get_num_stops_for_each_trial(monkey_sub, caught_ff_num)

    return num_stops_since_last_vis


def _calculate_num_stops_near_target(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance):
    caught_ff_num = len(ff_caught_T_new)
    monkey_sub = find_GUAT_or_TAFT_trials._take_out_monkey_subset_to_get_num_stops_near_target(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)
    num_stops_near_target = _use_merge_to_get_num_stops_for_each_trial(monkey_sub, caught_ff_num)
    return num_stops_near_target


def _use_merge_to_get_num_stops_for_each_trial(monkey_sub, caught_ff_num):
    monkey_sub = monkey_sub[['trial', 'point_index']].groupby('trial').count()
    monkey_sub = monkey_sub.merge(pd.DataFrame({'trial': np.arange(caught_ff_num)}), how='right', on='trial')
    monkey_sub = monkey_sub.fillna(0)
    num_stops_near_target = monkey_sub['point_index'].values
    return num_stops_near_target


def make_all_trial_features(ff_dataframe, monkey_information, ff_caught_T_new, cluster_around_target_indices, ff_real_position_sorted, ff_believed_position_sorted, max_cluster_distance = 75, data_folder_name = None):
    # Note that we start from trial = 1
    caught_ff_num = len(ff_caught_T_new)
    visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]
    trial_array, t_array = _calculate_trial_durations(ff_caught_T_new)
    target_clust_last_vis_df = cluster_analysis.get_target_clust_last_vis_df(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted)
                                                                               
    hitting_arena_edge = _calculate_hitting_arena_edge(monkey_information, ff_caught_T_new)
    num_stops_array = get_num_stops_array(monkey_information, np.arange(len(ff_caught_T_new)))
    num_stops_since_last_vis = _calculate_num_stops_since_last_vis(monkey_information, caught_ff_num, target_clust_last_vis_df['time_since_last_vis'].values)
    num_stops_near_target = _calculate_num_stops_near_target(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)
    n_ff_in_a_row = pattern_by_trials.n_ff_in_a_row_func(ff_believed_position_sorted, distance_between_ff = 50)
    num_ff_around_target = np.array([len(unit) for unit in cluster_around_target_indices])

    trials_dict = {'trial': trial_array, 
                   't': t_array,  
                   't_last_vis': target_clust_last_vis_df['time_since_last_vis'].values,
                   'd_last_vis': target_clust_last_vis_df['last_vis_dist'].values,
                   'abs_angle_last_vis': np.abs(target_clust_last_vis_df['last_vis_ang'].values), 
                   'hitting_arena_edge': hitting_arena_edge,
                   'num_stops': num_stops_array, 
                   'num_stops_since_last_vis': num_stops_since_last_vis,
                   'num_stops_near_target': num_stops_near_target,
                   'num_ff_around_target': num_ff_around_target.tolist(),
                   'n_ff_in_a_row': n_ff_in_a_row[:len(trial_array)].tolist()                            
                          }
    all_trial_features = pd.DataFrame(trials_dict)

    # drop trial=0
    all_trial_features = all_trial_features[all_trial_features['trial'] > 0].reset_index(drop=True)
    
    if data_folder_name:
        general_utils.save_df_to_csv(all_trial_features, 'all_trial_features', data_folder_name)
    return all_trial_features


def make_feature_statistics(all_trial_features, data_folder_name=None):
	
	all_trial_features_valid = all_trial_features[(all_trial_features['t_last_vis'] < 50) & (all_trial_features['hitting_arena_edge']==False)].reset_index()
	all_trial_features_valid = all_trial_features_valid.drop(columns={'index', 'trial'})
	median_values = all_trial_features_valid.median(axis=0)
	mean_values = all_trial_features_valid.mean(axis=0)
	n_trial = len(all_trial_features_valid)
	for i, item in enumerate(median_values.index):
		median = median_values[item]
		mean = mean_values[item]
		new_row = pd.DataFrame({'Item': item, 'Median': median, 'Mean': mean, 'N_trial': n_trial}, index=[0])
		if i == 0:
			feature_statistics = new_row
		else:
			feature_statistics = pd.concat([feature_statistics, new_row])

	feature_statistics = feature_statistics.reset_index(drop=True)

	feature_statistics['Label'] = 'Missing'
	feature_statistics.loc[feature_statistics['Item'] == 't', 'Label'] = 'time'
	feature_statistics.loc[feature_statistics['Item'] == 't_last_vis', 'Label'] = 'time target last seen'
	feature_statistics.loc[feature_statistics['Item'] == 'd_last_vis', 'Label'] = 'distance target last seen'
	feature_statistics.loc[feature_statistics['Item'] == 'abs_angle_last_vis', 'Label'] = 'abs angle target last seen'
	feature_statistics.loc[feature_statistics['Item'] == 'num_stops', 'Label'] = 'num stops'
	feature_statistics.loc[feature_statistics['Item'] == 'num_stops_near_target', 'Label'] = 'num stops near target'
	
	feature_statistics.loc[feature_statistics['Item'] == 'hitting_arena_edge', 'Label'] = 'hitting arena edge'
	feature_statistics.loc[feature_statistics['Item'] == 'num_stops_since_last_vis', 'Label'] = 'num stops since target last seen'
	feature_statistics.loc[feature_statistics['Item'] == 'n_ff_in_a_row', 'Label'] = 'num ff caught in a row'
	feature_statistics.loc[feature_statistics['Item'] == 'num_ff_around_target', 'Label'] = 'num ff around target'


	feature_statistics['Label for median'] = 'Median ' + feature_statistics['Label']
	feature_statistics['Label for mean'] = 'Mean ' + feature_statistics['Label']


	if data_folder_name:
		general_utils.save_df_to_csv(feature_statistics, 'feature_statistics', data_folder_name)
		
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


def _add_dates_based_on_data_names(df):
    df['data'] = df['data_name'].apply(lambda x: x.split('_')[1])
    all_dates = [int(date[-4:]) for date in df['data'].tolist()]
    all_dates = [datetime.strptime(str(date/100), '%m.%d').date() for date in all_dates]
    df['Date'] = all_dates
    df.sort_values(by='Date', inplace=True)



def add_dates_and_sessions(df):
    # organize_patterns_and_features._add_dates_based_on_data_names(combd_feature_statistics)
    _add_dates_based_on_data_names(df)

    # Create a mapping of unique data_name to unique sessions
    unique_sessions = {name: i for i, name in enumerate(df['Date'].unique())}

    # Map the unique sessions to the Data column
    df['Session'] = df['Date'].map(unique_sessions)

    # Sort the DataFrame by data_name
    df.sort_values(by='Session', inplace=True)


def make_distance_df(ff_caught_T_new, monkey_information, ff_believed_position_sorted):
    cum_distance_array = []
    distance_array = []
    for i in range(len(ff_caught_T_new)-1):
        cum_distance_array.append(specific_utils.get_cum_distance_traveled(i, ff_caught_T_new, monkey_information))
        distance_array.append(specific_utils.get_distance_between_two_points(i, ff_caught_T_new, monkey_information, ff_believed_position_sorted))
    cum_distance_array = np.array(cum_distance_array)
    distance_array = np.array(distance_array)
    distance_df = pd.DataFrame({'cum_distance': cum_distance_array, 'distance': distance_array})
    distance_df['trial'] = np.arange(len(distance_df))
    return distance_df


def make_num_stops_df(distance_df, closest_stop_to_capture_df, ff_caught_T_new, monkey_information):
    num_stops_df = alt_ff_utils.drop_rows_where_stop_is_not_inside_reward_boundary(closest_stop_to_capture_df)
    num_stops_df = num_stops_df.rename(columns={'time': 'stop_time',
                                                    'stop_ff_index': 'trial'})
    num_stops_df['num_stops'] = get_num_stops_array(monkey_information, 
                                                                                num_stops_df['trial'].values)

    num_stops_df['current_capture_time'] = ff_caught_T_new[num_stops_df['trial']]
    num_stops_df['prev_capture_time'] = ff_caught_T_new[num_stops_df['trial'] - 1]

    # Add distance information
    num_stops_df = num_stops_df.merge(distance_df, on='trial', how='left').dropna()

    # # Filter out the outliers
    # original_length = len(num_stops_df)
    # num_stops_df = num_stops_df[(num_stops_df['distance'] < 2000) & (num_stops_df['cum_distance'] < 2000)]
    # print(f'Filtered out {original_length - len(num_stops_df)} outliers out of {original_length} trials, since they have distance ' + 
    #       f'or displacement greater than 2000, which is {round(100*(original_length - len(num_stops_df))/original_length, 2)}% of the data')
    return num_stops_df