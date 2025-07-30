from decision_making_analysis.GUAT import GUAT_utils

import os
import sys
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def give_up_after_trying_func(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, max_point_index=None, max_cluster_distance=75):
    """
    Find the trials where the monkey has stopped more than once to catch a firefly but failed to succeed, and the monkey gave up.
    """

    GUAT_trials_df = make_GUAT_trials_df(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)
    GUAT_indices_df, GUAT_point_indices_for_anim = _get_more_or_GUAT_or_TAFT_info(
        GUAT_trials_df, monkey_information, max_point_index=max_point_index)

    GUAT_w_ff_df, GUAT_expanded_trials_df = GUAT_utils.get_GUAT_w_ff_df(GUAT_indices_df,
                                                                        GUAT_trials_df,
                                                                        ff_dataframe,
                                                                        monkey_information,
                                                                        ff_real_position_sorted,
                                                                        )
    give_up_after_trying_trials = GUAT_expanded_trials_df['trial'].values

    # only keep the GUAT_indices_df in GUAT_w_ff_df
    GUAT_trials_df = GUAT_trials_df[GUAT_trials_df['cluster_index'].isin(
        GUAT_w_ff_df['cluster_index'].unique())]
    GUAT_indices_df = GUAT_indices_df[GUAT_indices_df['cluster_index'].isin(
        GUAT_w_ff_df['cluster_index'].unique())]
    GUAT_point_indices_for_anim = only_get_point_indices_for_anim(
        GUAT_trials_df, monkey_information, max_point_index=None)

    return give_up_after_trying_trials, GUAT_indices_df, GUAT_trials_df, GUAT_point_indices_for_anim, GUAT_w_ff_df


def try_a_few_times_func(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_point_index=None, max_cluster_distance=75):
    """
    Find the trials where the monkey has stopped more than one times to catch a target
    """

    TAFT_trials_df = make_TAFT_trials_df(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=max_cluster_distance)
    try_a_few_times_trials = TAFT_trials_df['trial'].unique()
    TAFT_indices_df, try_a_few_times_indices_for_anim = _get_more_or_GUAT_or_TAFT_info(
        TAFT_trials_df, monkey_information, max_point_index=max_point_index)
    return try_a_few_times_trials, TAFT_indices_df, TAFT_trials_df, try_a_few_times_indices_for_anim


def _get_more_or_GUAT_or_TAFT_info(trials_df, monkey_information, max_point_index=None):

    # Initialize lists to store indices
    point_indices = []
    indices_corr_trials = []
    indices_corr_clusters = []
    point_indices_for_anim = []

    # Iterate over the rows of trials_df
    for _, row in trials_df.iterrows():
        first_stop_point_index = row['first_stop_point_index']
        last_stop_point_index = row['last_stop_point_index']
        indices_to_add = list(
            range(first_stop_point_index, last_stop_point_index))

        point_indices.extend(indices_to_add)
        indices_corr_trials.extend([row['trial']] * len(indices_to_add))
        indices_corr_clusters.extend(
            [row['cluster_index']] * len(indices_to_add))
        point_indices_for_anim.extend(
            range(first_stop_point_index - 20, last_stop_point_index + 21))

    if max_point_index is None:
        max_point_index = monkey_information['point_index'].max()

    # Convert lists to numpy arrays
    point_indices = np.array(point_indices)
    indices_corr_trials = np.array(indices_corr_trials)
    indices_corr_clusters = np.array(indices_corr_clusters)
    point_indices_for_anim = np.unique(np.array(point_indices_for_anim))

    # Filter indices based on max_point_index
    indices_to_keep = point_indices < max_point_index
    indices_df = pd.DataFrame({
        'point_index': point_indices[indices_to_keep],
        'trial': indices_corr_trials[indices_to_keep],
        'cluster_index': indices_corr_clusters[indices_to_keep]
    })

    point_indices_for_anim = point_indices_for_anim[point_indices_for_anim < max_point_index]
    return indices_df, point_indices_for_anim


def only_get_point_indices_for_anim(trials_df, monkey_information, max_point_index=None):
    point_indices_for_anim = []
    for _, row in trials_df.iterrows():
        first_stop_point_index = row['first_stop_point_index']
        last_stop_point_index = row['last_stop_point_index']
        point_indices_for_anim.extend(
            range(first_stop_point_index - 20, last_stop_point_index + 21))

    if max_point_index is None:
        max_point_index = monkey_information['point_index'].max()

    point_indices_for_anim = np.unique(np.array(point_indices_for_anim))
    point_indices_for_anim = point_indices_for_anim[point_indices_for_anim < max_point_index]
    return point_indices_for_anim


def make_GUAT_trials_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=75):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_GUAT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)

    GUAT_trials_df = _make_trials_df(monkey_sub)

    GUAT_trials_df.reset_index(drop=True, inplace=True)

    return GUAT_trials_df


def make_TAFT_trials_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=75):

    # Extract a subset of monkey information that is relevant for GUAT analysis
    monkey_sub = _take_out_monkey_subset_for_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance)

    TAFT_trials_df = _make_trials_df(monkey_sub)

    TAFT_trials_df.reset_index(drop=True, inplace=True)

    return TAFT_trials_df


def _make_trials_df(monkey_sub):

    trials_df = monkey_sub[['stop_cluster_id', 'trial']
                           ].drop_duplicates().reset_index(drop=True)

    # Calculate the number of stops for each cluster
    trials_df['num_stops'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial']).size().values

    # only keep the trials with more than one stop
    trials_df = trials_df[trials_df['num_stops'] > 1].copy()
    monkey_sub = monkey_sub.merge(trials_df[['stop_cluster_id', 'trial']], on=[
                                  'stop_cluster_id', 'trial'], how='inner')

    # Get stop indices for each cluster
    trials_df['stop_indices'] = monkey_sub.groupby(['stop_cluster_id', 'trial'])[
        'point_index'].apply(list).values

    # Assign cluster indices
    trials_df['cluster_index'] = np.arange(len(trials_df))

    # Get first, second, and last stop point indices for each cluster
    trials_df['first_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].first().values
    trials_df['second_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].nth(1).values
    trials_df['last_stop_point_index'] = monkey_sub.groupby(
        ['stop_cluster_id', 'trial'])['point_index'].last().values

    # Get stop times for first, second, and last stops
    monkey_sub.set_index('point_index', inplace=True)
    trials_df['first_stop_time'] = monkey_sub.loc[trials_df['first_stop_point_index'], 'time'].values
    trials_df['second_stop_time'] = monkey_sub.loc[trials_df['second_stop_point_index'], 'time'].values
    trials_df['last_stop_time'] = monkey_sub.loc[trials_df['last_stop_point_index'], 'time'].values

    return trials_df


def _take_out_monkey_subset_for_GUAT(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=75):
    """
    Extract a subset of monkey information for GUAT analysis.

    Parameters:
    monkey_information (pd.DataFrame): DataFrame containing monkey movement data.
    ff_caught_T_new (np.array): Array of times when fireflies were caught.
    ff_real_position_sorted (np.array): Array of real positions of fireflies.
    max_cluster_distance (float): Maximum distance for clustering stops.

    Returns:
    pd.DataFrame: Subset of monkey information for GUAT analysis.
    """
    # Add stop cluster IDs
    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted)

    # Calculate distances to last targets
    monkey_sub[['last_target_x', 'last_target_y']
               ] = ff_real_position_sorted[monkey_sub['trial'].values - 1]
    monkey_sub['distance_to_last_target'] = np.sqrt(
        (monkey_sub['monkey_x'] - monkey_sub['last_target_x'])**2 + (monkey_sub['monkey_y'] - monkey_sub['last_target_y'])**2)

    # Filter out clusters too close to the targets
    close_to_target_clusters = monkey_sub[(monkey_sub['distance_to_target'] < max_cluster_distance) |
                                          (monkey_sub['distance_to_last_target'] < max_cluster_distance)]['stop_cluster_id'].unique()
    print(f'When take out monkey subset for GUAT, {len(close_to_target_clusters)} clusters out of {len(monkey_sub["stop_cluster_id"].unique())} are'
          ' too close to the target or the last target. Those clusters are filtered out.')
    monkey_sub = monkey_sub[~monkey_sub['stop_cluster_id'].isin(
        close_to_target_clusters)].copy()

    # Also filter out clusters that span multiple trials
    monkey_sub2 = monkey_sub[['stop_cluster_id', 'trial']].drop_duplicates()
    monkey_sub2 = monkey_sub2.groupby('stop_cluster_id').size()
    monkey_sub2 = monkey_sub2[monkey_sub2 == 1].reset_index(drop=False)
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(
        monkey_sub2['stop_cluster_id'])].copy()

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)

    return monkey_sub


def _take_out_monkey_subset_to_get_num_stops_near_target(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=75):
    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance, use_ff_caught_time_new_to_separate_clusters=True)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                                          min_stop_per_cluster=1)
    # Keep clusters that are close to the targets
    monkey_sub = _keep_clusters_close_to_target(
        monkey_sub, max_cluster_distance)

    # For each trial, keep the latest stop cluster
    monkey_sub = _keep_latest_cluster_for_each_trial(monkey_sub)

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)
    return monkey_sub


def _take_out_monkey_subset_for_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted, max_cluster_distance=75):

    monkey_information = add_stop_cluster_id(
        monkey_information, max_cluster_distance, use_ff_caught_time_new_to_separate_clusters=True)

    monkey_sub = _take_out_monkey_subset_for_GUAT_or_TAFT(
        monkey_information, ff_caught_T_new, ff_real_position_sorted)

    # Keep clusters that are close to the targets
    monkey_sub = _keep_clusters_close_to_target(
        monkey_sub, max_cluster_distance)

    # For each trial, keep the latest stop cluster
    monkey_sub = _keep_latest_cluster_for_each_trial(monkey_sub)

    # if two trials share the same stop cluster, then keep the trial with the smaller trial number
    monkey_sub.sort_values(by=['stop_cluster_id', 'trial'], inplace=True)
    unique_combo_to_keep = monkey_sub.groupby(
        'stop_cluster_id')['trial'].first().reset_index(drop=False)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'stop_cluster_id', 'trial'], how='inner')

    # Sort and reset index
    monkey_sub.sort_values(by='point_index', inplace=True)
    monkey_sub.reset_index(drop=True, inplace=True)

    return monkey_sub


def _keep_clusters_close_to_target(monkey_sub, max_cluster_distance=75):
    close_to_target_clusters = monkey_sub[(
        monkey_sub['distance_to_target'] < max_cluster_distance)]['stop_cluster_id'].unique()
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(
        close_to_target_clusters)].copy()
    return monkey_sub


def _keep_latest_cluster_for_each_trial(monkey_sub):
    monkey_sub.sort_values(by=['trial', 'stop_cluster_id'], inplace=True)
    unique_combo_to_keep = monkey_sub[[
        'trial', 'stop_cluster_id']].groupby('trial').tail(1)
    monkey_sub = monkey_sub.merge(unique_combo_to_keep, on=[
                                  'trial', 'stop_cluster_id'], how='inner')
    return monkey_sub


def _take_out_monkey_subset_for_GUAT_or_TAFT(monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                             min_stop_per_cluster=2):

    # Filter for new distinct stops within the time range
    monkey_sub = monkey_information[monkey_information['whether_new_distinct_stop'] == True].copy(
    )
    monkey_sub = monkey_sub[monkey_sub['time'].between(
        ff_caught_T_new[0], ff_caught_T_new[-1])]
    # Assign trial numbers and target positions
    monkey_sub[['target_x', 'target_y']
               ] = ff_real_position_sorted[monkey_sub['trial'].values]
    # Calculate distances to targets
    monkey_sub['distance_to_target'] = np.sqrt(
        (monkey_sub['monkey_x'] - monkey_sub['target_x'])**2 + (monkey_sub['monkey_y'] - monkey_sub['target_y'])**2)

    # Find clusters with more than one stop
    cluster_counts = monkey_sub['stop_cluster_id'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >=
                                    min_stop_per_cluster].index
    monkey_sub = monkey_sub[monkey_sub['stop_cluster_id'].isin(valid_clusters)]

    return monkey_sub


def add_stop_cluster_id(monkey_information, max_cluster_distance, use_ff_caught_time_new_to_separate_clusters=False):
    # note, in addition to stop_cluster_id, we also add stop_cluster_start_point and stop_cluster_end_point
    monkey_information = monkey_information.copy()
    stop_points_df = monkey_information[monkey_information['whether_new_distinct_stop']].copy(
    )

    # take out stops that are not too close to previous stop points
    stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance'].diff()
    stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance_from_last_stop_point'].fillna(
        100)

    stop_points_df['cum_distance_from_next_stop_point'] = (
        -1) * stop_points_df['cum_distance'].diff(-1)
    stop_points_df['cum_distance_from_next_stop_point'] = stop_points_df['cum_distance_from_next_stop_point'].fillna(
        100)

    stop_points_df['cluster_start'] = stop_points_df['cum_distance_from_last_stop_point'] > max_cluster_distance
    stop_points_df['cluster_end'] = stop_points_df['cum_distance_from_next_stop_point'] > max_cluster_distance

    if use_ff_caught_time_new_to_separate_clusters:
        stop_points_df = further_identify_cluster_start_and_end_based_on_ff_capture_time(
            stop_points_df)

    stop_points_df_sub = stop_points_df.copy()

    # now take out only the rows that are cluster_start or cluster_stop
    stop_points_df_sub = stop_points_df_sub[(stop_points_df_sub['cluster_start']) | (
        stop_points_df_sub['cluster_end'])].copy()

    all_start_stop_points = stop_points_df_sub.loc[stop_points_df_sub['cluster_start'],
                                                   'point_index'].values
    all_end_stop_points = stop_points_df_sub.loc[stop_points_df_sub['cluster_end'],
                                                 'point_index'].values
    if len(all_start_stop_points) != len(all_end_stop_points):
        raise ValueError(
            'The number of start and end stop points are not the same')

    # now, assign each point in monkey_information to a stop cluster
    monkey_information['stop_cluster_id'] = np.searchsorted(
        all_start_stop_points, monkey_information['point_index'].values, side='right') - 1
    monkey_information.loc[monkey_information['stop_cluster_id']
                           < 0, 'stop_cluster_id'] = 0
    monkey_information.loc[monkey_information['stop_cluster_id'] >= len(
        all_start_stop_points), 'stop_cluster_id'] = len(all_start_stop_points) - 1
    monkey_information['stop_cluster_start_point'] = all_start_stop_points[monkey_information['stop_cluster_id']]
    monkey_information['stop_cluster_end_point'] = all_end_stop_points[monkey_information['stop_cluster_id']]

    # for the rows that are not stop points, set the stop_cluster_id to nan
    monkey_information.loc[monkey_information['monkey_speeddummy']
                           == 1, 'stop_cluster_id'] = np.nan
    # for the rows that are not in between the start and end points, set the stop_cluster_id to nan
    monkey_information.loc[~monkey_information['point_index'].between(monkey_information['stop_cluster_start_point'], monkey_information['stop_cluster_end_point']),
                           'stop_cluster_id'] = np.nan

    # assign the same stop_cluster_id to all points with the same stop_id
    sub_w_stop_cluster_id = monkey_information[[
        'stop_id', 'stop_cluster_id']].drop_duplicates().dropna()
    monkey_information.drop(columns='stop_cluster_id', inplace=True)
    monkey_information = monkey_information.merge(
        sub_w_stop_cluster_id, on='stop_id', how='left')
    return monkey_information


def further_identify_cluster_start_and_end_based_on_ff_capture_time(stop_points_df):

    stop_points_df = stop_points_df.sort_values(by='point_index')
    # find the point index that has marked a new trial compared to previous point idnex
    stop_points_df['new_trial'] = stop_points_df['trial'].diff().fillna(1)

    # print the number of new trials
    print(
        f'The number of new trials that are used to separate stop clusters is {stop_points_df["new_trial"].sum().astype(int)}')

    # Mark those points as cluster_start, and the points after as cluster_end
    stop_points_df.reset_index(drop=True, inplace=True)
    index_to_mark_as_end = stop_points_df[stop_points_df['new_trial']
                                          == 1].index.values
    stop_points_df.loc[index_to_mark_as_end, 'cluster_end'] = True
    index_to_mark_as_start = index_to_mark_as_end + 1
    index_to_mark_as_start = index_to_mark_as_start[index_to_mark_as_start < len(
        stop_points_df)]
    stop_points_df.loc[index_to_mark_as_start, 'cluster_start'] = True

    # check correctness
    if (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() > 1) | \
            (stop_points_df['cluster_start'].sum() - stop_points_df['cluster_end'].sum() < 0):
        raise ValueError(
            'The number of cluster start and end points are not the same')

    return stop_points_df
