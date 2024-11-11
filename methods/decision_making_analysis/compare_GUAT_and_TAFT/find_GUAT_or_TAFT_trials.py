import sys
from data_wrangling import basic_func
from planning_analysis.show_planning import alt_ff_utils

import os
import numpy as np
import pandas as pd
import collections
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



def give_up_after_trying_func(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_point_index=None, max_cluster_distance=75):
    """
    Find the trials where the monkey has stopped more than once to catch a firefly but failed to succeed, and the monkey gave up.
    """

    GUAT_trials_df = make_GUAT_trials_df(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance)
    give_up_after_trying_trials = GUAT_trials_df['trial'].unique()
    GUAT_indices_df, GUAT_point_indices_for_anim = _get_more_or_TAFT_info(GUAT_trials_df, monkey_information, max_point_index=max_point_index)

    return give_up_after_trying_trials, GUAT_indices_df, GUAT_trials_df, GUAT_point_indices_for_anim


def try_a_few_times_func(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_point_index=None, max_cluster_distance=75):
    """
    Find the trials where the monkey has stopped more than one times to catch a target
    """

    TAFT_trials_df = make_TAFT_trials_df(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)
    try_a_few_times_trials = TAFT_trials_df['trial'].unique()
    TAFT_indices_df, try_a_few_times_indices_for_anim = _get_more_or_TAFT_info(TAFT_trials_df, monkey_information, max_point_index=max_point_index)
    return try_a_few_times_trials, TAFT_indices_df, TAFT_trials_df, try_a_few_times_indices_for_anim


def _get_more_or_TAFT_info(trials_df, monkey_information, max_point_index=None):

    # Initialize lists to store indices
    point_indices = []
    indices_corr_trials = []
    indices_corr_clusters = []
    point_indices_for_anim = []

    # Iterate over the rows of trials_df
    for _, row in trials_df.iterrows():
        first_stop_point_index = row['first_stop_point_index']
        last_stop_point_index = row['last_stop_point_index']
        indices_to_add = list(range(first_stop_point_index, last_stop_point_index))
        
        point_indices.extend(indices_to_add)
        indices_corr_trials.extend([row['trial']] * len(indices_to_add))
        indices_corr_clusters.extend([row['cluster_index']] * len(indices_to_add))
        point_indices_for_anim.extend(range(first_stop_point_index - 20, last_stop_point_index + 21))

    if max_point_index is None:
        max_point_index = monkey_information['point_index'].max()

    # Convert lists to numpy arrays
    point_indices = np.array(point_indices)
    indices_corr_trials = np.array(indices_corr_trials)
    indices_corr_clusters = np.array(indices_corr_clusters)
    point_indices_for_anim = np.array(point_indices_for_anim)

    # Filter indices based on max_point_index
    indices_to_keep = point_indices < max_point_index
    indices_df = pd.DataFrame({
        'point_index': point_indices[indices_to_keep],
        'trial': indices_corr_trials[indices_to_keep],
        'cluster_index': indices_corr_clusters[indices_to_keep]
    })

    point_indices_for_anim = point_indices_for_anim[point_indices_for_anim < max_point_index]
    return indices_df, point_indices_for_anim



    
def make_GUAT_trials_df(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=75):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_GUAT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance)

    GUAT_trials_df = _make_trials_df(monkey_sub)

    return GUAT_trials_df


def make_TAFT_trials_df(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=75):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_TAFT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance)
    
    # make all the stops belonging to the same stop cluster id to be in the same trial (using the trial number of the first stop in the cluster)
    monkey_sub['trial'] = monkey_sub.groupby('stop_cluster_id')['trial'].transform('first')
    
    TAFT_trials_df = _make_trials_df(monkey_sub)

    return TAFT_trials_df


def _make_trials_df(monkey_sub):

    trials_df = monkey_sub[['stop_cluster_id', 'trial']].drop_duplicates().reset_index(drop=True)

    # Calculate the number of stops for each cluster
    trials_df['num_stops'] = monkey_sub.groupby(['stop_cluster_id', 'trial']).size().values

    # Get stop indices for each cluster
    trials_df['stop_indices'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])['point_index'].apply(list).values

    # Assign cluster indices
    trials_df['cluster_index'] = np.arange(len(trials_df))

    # Get first, second, and last stop point indices for each cluster
    trials_df['first_stop_point_index'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])['point_index'].first().values
    trials_df['second_stop_point_index'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])['point_index'].nth(1).values
    trials_df['last_stop_point_index'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])['point_index'].last().values

    # Get stop times for first, second, and last stops
    monkey_sub.set_index('point_index', inplace=True)
    trials_df['first_stop_time'] = monkey_sub.loc[trials_df['first_stop_point_index'], 'monkey_t'].values
    trials_df['second_stop_time'] = monkey_sub.loc[trials_df['second_stop_point_index'], 'monkey_t'].values
    trials_df['last_stop_time'] = monkey_sub.loc[trials_df['last_stop_point_index'], 'monkey_t'].values

    return trials_df



def _take_out_monkey_subset_for_GUAT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=75):
    """
    Extract a subset of monkey information for GUAT analysis.

    Parameters:
    monkey_information (pd.DataFrame): DataFrame containing monkey movement data.
    ff_caught_T_sorted (np.array): Array of times when fireflies were caught.
    ff_real_position_sorted (np.array): Array of real positions of fireflies.
    max_cluster_distance (float): Maximum distance for clustering stops.

    Returns:
    pd.DataFrame: Subset of monkey information for GUAT analysis.
    """
    # Add stop cluster IDs
    monkey_information = add_stop_cluster_id(monkey_information, max_cluster_distance)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)

    # Calculate distances to last targets
    monkey_sub[['last_target_x', 'last_target_y']] = ff_real_position_sorted[monkey_sub['trial'].values - 1]
    monkey_sub['distance_to_last_target'] = np.sqrt((monkey_sub['monkey_x'] - monkey_sub['last_target_x'])**2 + (monkey_sub['monkey_y'] - monkey_sub['last_target_y'])**2)

    # Filter out clusters too close to the targets
    too_close_clusters = monkey_sub[(monkey_sub['distance_to_target'] < max_cluster_distance) | 
                                    (monkey_sub['distance_to_last_target'] < max_cluster_distance)]['stop_cluster_id'].unique()
    print(f'{len(too_close_clusters)} clusters out of {len(monkey_sub["stop_cluster_id"].unique())} are too close to the target or the last target')
    monkey_sub = monkey_sub[~monkey_sub['stop_cluster_id'].isin(too_close_clusters)].copy()

    # Also filter out clusters that span multiple trials
    monkey_sub2 = monkey_sub[['stop_cluster_id', 'trial']].drop_duplicates()
    monkey_sub2 = monkey_sub2.groupby('stop_cluster_id').size()
    monkey_sub2 = monkey_sub2[monkey_sub2 == 1].reset_index(drop=False)
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(monkey_sub2['stop_cluster_id'])].copy()

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)

    return monkey_sub


def _take_out_monkey_subset_for_TAFT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=75):

    monkey_information = add_stop_cluster_id(monkey_information, max_cluster_distance, use_ff_capture_time_to_separate_clusters=True, 
                                                        ff_caught_T_sorted=ff_caught_T_sorted)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)

    # Keep clusters that are close to the targets
    too_close_clusters = monkey_sub[(monkey_sub['distance_to_target'] < max_cluster_distance)]['stop_cluster_id'].unique()
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(too_close_clusters)].copy()

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)
    
    return monkey_sub


def _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_sorted, ff_real_position_sorted, max_cluster_distance=75):

    # Filter for new distinct stops within the time range
    monkey_sub = monkey_information[monkey_information['whether_new_distinct_stop']].copy()
    monkey_sub = monkey_sub[monkey_sub['time'].between(ff_caught_T_sorted[0], ff_caught_T_sorted[-1])]

    # Find clusters with more than one stop
    cluster_counts = monkey_sub['stop_cluster_id'].value_counts()
    valid_clusters = cluster_counts[cluster_counts > 1].index
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(valid_clusters)]

    # Assign trial numbers and target positions
    monkey_sub['trial'] = np.searchsorted(ff_caught_T_sorted, monkey_sub['time'].values)
    monkey_sub[['target_x', 'target_y']] = ff_real_position_sorted[monkey_sub['trial'].values]

    # Calculate distances to targets
    monkey_sub['distance_to_target'] = np.sqrt((monkey_sub['monkey_x'] - monkey_sub['target_x'])**2 + (monkey_sub['monkey_y'] - monkey_sub['target_y'])**2)

    return monkey_sub


def add_stop_cluster_id(monkey_information, max_cluster_distance, use_ff_capture_time_to_separate_clusters=False, 
                        ff_caught_T_sorted=None):
    # note, in addition to stop_cluster_id, we also add stop_cluster_start_point and stop_cluster_end_point

    stop_points_df = monkey_information[monkey_information['whether_new_distinct_stop']].copy()

    # take out stops that are not too close to previous stop points
    stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance'].diff()
    stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance_from_last_stop_point'].fillna(100)

    stop_points_df['cum_distance_from_next_stop_point'] = (-1) * stop_points_df['cum_distance'].diff(-1)
    stop_points_df['cum_distance_from_next_stop_point'] = stop_points_df['cum_distance_from_next_stop_point'].fillna(100)

    stop_points_df['cluster_start'] = stop_points_df['cum_distance_from_last_stop_point'] > max_cluster_distance
    stop_points_df['cluster_end'] = stop_points_df['cum_distance_from_next_stop_point'] > max_cluster_distance

    if use_ff_capture_time_to_separate_clusters:
        if ff_caught_T_sorted is None:
            raise ValueError('If use_ff_capture_time_to_separate_clusters is True, ff_caught_T_sorted needs to be provided')
        stop_points_df = further_identify_cluster_start_and_end_based_on_ff_capture_time(stop_points_df, ff_caught_T_sorted)


    stop_points_df_sub = stop_points_df.copy()

    # now take out only the rows that are cluster_start or cluster_stop
    stop_points_df_sub = stop_points_df_sub[(stop_points_df_sub['cluster_start']) | (stop_points_df_sub['cluster_end'])].copy()

    all_start_stop_points = stop_points_df_sub.loc[stop_points_df_sub['cluster_start'], 'point_index'].values
    all_end_stop_points = stop_points_df_sub.loc[stop_points_df_sub['cluster_end'], 'point_index'].values
    if len(all_start_stop_points) != len(all_end_stop_points):
        raise ValueError('The number of start and end stop points are not the same')

    # now, assign each point in monkey_information to a stop cluster
    monkey_information['stop_cluster_id'] = np.searchsorted(all_start_stop_points, monkey_information['point_index'].values, side='right') - 1
    monkey_information.loc[monkey_information['stop_cluster_id'] < 0, 'stop_cluster_id'] = 0 
    monkey_information.loc[monkey_information['stop_cluster_id'] >= len(all_start_stop_points), 'stop_cluster_id'] = len(all_start_stop_points) - 1
    monkey_information['stop_cluster_start_point'] = all_start_stop_points[monkey_information['stop_cluster_id']]
    monkey_information['stop_cluster_end_point'] = all_end_stop_points[monkey_information['stop_cluster_id']]

    # for the rows that are not stop points, set the stop_cluster_id to nan
    monkey_information.loc[monkey_information['monkey_speeddummy'] == 1, 'stop_cluster_id'] = np.nan
    # for the rows that are not in between the start and end points, set the stop_cluster_id to nan
    monkey_information.loc[~monkey_information['point_index'].between(monkey_information['stop_cluster_start_point'], monkey_information['stop_cluster_end_point']), 
                        'stop_cluster_id'] = np.nan
    
    return monkey_information


def further_identify_cluster_start_and_end_based_on_ff_capture_time(stop_points_df, ff_caught_T_sorted):
    # Mark the stops that are near the capture time of a firefly
    all_closest_point_to_capture_df = alt_ff_utils.get_closest_stop_time_to_all_capture_time(ff_caught_T_sorted, stop_points_df, stop_ff_index_array=np.arange(len(ff_caught_T_sorted)))
    all_closest_point_to_capture_df['near_capture'] = 1
    stop_points_df = stop_points_df.merge(all_closest_point_to_capture_df, on='point_index', how='left')
    stop_points_df['near_capture'] = stop_points_df['near_capture'].fillna(0)
    if stop_points_df['near_capture'].sum() != len(all_closest_point_to_capture_df):
        raise ValueError('The number of near capture points is not the same as the number of capture points')

    # Mark those points as cluster_start, and the points after as cluster_end
    stop_points_df.reset_index(drop=True, inplace=True)
    index_to_mark_as_stop = stop_points_df[stop_points_df['near_capture']==1].index.values
    stop_points_df.loc[index_to_mark_as_stop, 'cluster_end'] = True
    index_to_mark_as_start = index_to_mark_as_stop + 1
    index_to_mark_as_start = index_to_mark_as_start[index_to_mark_as_start < len(stop_points_df)]
    stop_points_df.loc[index_to_mark_as_start, 'cluster_start'] = True

    # check correctness
    if (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() > 1) | \
        (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() < 0):
        raise ValueError('The number of cluster start and end points are not the same')

    return stop_points_df