from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.GUAT import add_features_GUAT_and_TAFT
from decision_making_analysis import free_selection, replacement, trajectory_info
from pattern_discovery import cluster_analysis

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def find_df_related_to_cluster_replacement(replacement_df, prior_to_replacement_df, non_chosen_df, manual_anno, ff_dataframe, monkey_information, ff_real_position_sorted, ff_life_sorted, sample_size=None, equal_sample_from_two_cases=False):
    """
    Given dataframes containing information about fireflies and their clusters, this function selects a sample of fireflies
    and their clusters, and returns dataframes containing information about these fireflies and clusters, with some additional
    columns marking the intended target fireflies.

    Args:
        replacement_df (pandas.DataFrame): A dataframe containing information about fireflies that have just replaced some
            other fireflies.
        prior_to_replacement_df (pandas.DataFrame): A dataframe containing information about the fireflies that were replaced.
        non_chosen_df (pandas.DataFrame): A dataframe containing information about fireflies that were not chosen as the
            intended targets.
        manual_anno (pandas.DataFrame): A dataframe containing manual annotations for some fireflies.
        ff_dataframe (pandas.DataFrame): A dataframe containing information about all fireflies.
        monkey_information (pandas.DataFrame): A dataframe containing information about the monkey's behavior.
        ff_real_position_sorted (numpy.ndarray): An array containing the real positions of all fireflies.
        ff_life_sorted (numpy.ndarray): An array containing the life values of all fireflies.
        sample_size (int, optional): The number of fireflies to sample from each of the two cases (new fireflies and non-chosen
            fireflies). If None, all fireflies will be used. Defaults to None.
        equal_sample_from_two_cases (bool, optional): Whether to sample the same number of fireflies from each of the two cases.
            If False, the number of sampled fireflies from each case will be the same as the sample_size. Defaults to False.

    Returns:
        tuple: A tuple containing four dataframes, each containing information about the clusters of fireflies in one of the
        four cases: old fireflies, new fireflies, parallel old fireflies, and non-chosen fireflies. The dataframes have columns
        indicating the positions, times, and other properties of the fireflies in each cluster, as well as additional columns
        marking the intended target fireflies.
    """

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]

    # find information of old_ff and new_ff
    old_ff_info, new_ff_info, all_point_index, all_time = replacement.get_old_and_new_ff_info(
        replacement_df, prior_to_replacement_df, ff_dataframe, monkey_information, ff_real_position_sorted)
    old_ff_info, new_ff_info, old_ff_positions, new_ff_positions, all_point_index_1, all_time_1 = eliminate_close_by_pairs_between_old_and_new_ff_info(
        old_ff_info, new_ff_info, all_time, ff_real_position_sorted)

    # also find information of non_chosen_ff and parallel_old_ff
    non_chosen_ff_info, parallel_old_ff_info, non_chosen_ff_positions, parallel_old_ff_positions, all_point_index_2, all_time_2\
        = find_non_chosen_ff_info_and_parallel_old_ff_info_with_no_close_by_pairs(non_chosen_df, manual_anno, ff_real_position_sorted, ff_life_sorted, ff_dataframe, monkey_information)

    if sample_size is None:
        if equal_sample_from_two_cases:
            sample_size = min(len(new_ff_info), len(non_chosen_ff_info))

    if sample_size is not None:
        sampled_indices_1 = np.random.choice(
            len(new_ff_info), sample_size, replace=False)
        sampled_indices_2 = np.random.choice(
            len(non_chosen_ff_info), sample_size, replace=False)
        old_ff_info, new_ff_info = general_utils.take_out_a_sample_from_df(
            sampled_indices_1, old_ff_info, new_ff_info)
        old_ff_positions, new_ff_positions, all_point_index_1, all_time_1 = general_utils.take_out_a_sample_from_arrays(
            sampled_indices_1, old_ff_positions, new_ff_positions, all_point_index_1, all_time_1)
        non_chosen_ff_info, parallel_old_ff_info = general_utils.take_out_a_sample_from_df(
            sampled_indices_2, non_chosen_ff_info, parallel_old_ff_info)
        non_chosen_ff_positions, parallel_old_ff_positions, all_point_index_2, all_time_2 = general_utils.take_out_a_sample_from_arrays(
            sampled_indices_2, non_chosen_ff_positions, parallel_old_ff_positions, all_point_index_2, all_time_2)

    # find information of clusters of these ff
    max_cluster_distance = 50
    old_ff_cluster = cluster_analysis.find_alive_ff_clusters(
        old_ff_positions, ff_real_position_sorted, all_time_1-10, all_time_1+10, ff_life_sorted, max_distance=max_cluster_distance)
    new_ff_cluster = cluster_analysis.find_alive_ff_clusters(
        new_ff_positions, ff_real_position_sorted, all_time_1-10, all_time_1+10, ff_life_sorted, max_distance=max_cluster_distance)
    parallel_old_ff_cluster = cluster_analysis.find_alive_ff_clusters(
        parallel_old_ff_positions, ff_real_position_sorted, all_time_2-10, all_time_2+10, ff_life_sorted, max_distance=max_cluster_distance)
    non_chosen_ff_cluster = cluster_analysis.find_alive_ff_clusters(
        non_chosen_ff_positions, ff_real_position_sorted, all_time_2-10, all_time_2+10, ff_life_sorted, max_distance=max_cluster_distance)

    # turn them into df and find corresponding information
    old_ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        old_ff_cluster, all_point_index_1)
    old_ff_cluster_df = decision_making_utils.find_many_ff_info_anew(
        old_ff_cluster_df['ff_index'].values, old_ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    new_ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        new_ff_cluster, all_point_index_1)
    new_ff_cluster_df = decision_making_utils.find_many_ff_info_anew(
        new_ff_cluster_df['ff_index'].values, new_ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    parallel_old_ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        parallel_old_ff_cluster, all_point_index_2)
    parallel_old_ff_cluster_df = decision_making_utils.find_many_ff_info_anew(
        parallel_old_ff_cluster_df['ff_index'].values, parallel_old_ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    non_chosen_ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        non_chosen_ff_cluster, all_point_index_2)
    non_chosen_ff_cluster_df = decision_making_utils.find_many_ff_info_anew(
        non_chosen_ff_cluster_df['ff_index'].values, non_chosen_ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    # eliminate ff whose time_since_last_vis is too large
    old_ff_cluster_df, new_ff_cluster_df = eliminate_rows_with_large_value_in_shared_column_between_df(
        'time_since_last_vis', 3, old_ff_cluster_df, new_ff_cluster_df)
    parallel_old_ff_cluster_df, non_chosen_ff_cluster_df = eliminate_rows_with_large_value_in_shared_column_between_df(
        'time_since_last_vis', 3, parallel_old_ff_cluster_df, non_chosen_ff_cluster_df)

    # mark the current intended_target in old_ff_cluster_df and the next intended target in new_ff_cluster_df
    old_ff_cluster_df = mark_intended_target_in_df(
        old_ff_cluster_df, old_ff_info)
    new_ff_cluster_df = mark_intended_target_in_df(
        new_ff_cluster_df, new_ff_info)

    # mark the current intended_target in parallel_old_ff_cluster_df
    parallel_old_ff_cluster_df = mark_intended_target_in_df(
        parallel_old_ff_cluster_df, parallel_old_ff_info)
    non_chosen_ff_cluster_df['whether_intended_target'] = False

    # mark whether the ff is changed
    old_ff_cluster_df['whether_changed'] = True
    new_ff_cluster_df['whether_changed'] = True
    parallel_old_ff_cluster_df['whether_changed'] = False
    non_chosen_ff_cluster_df['whether_changed'] = False

    return old_ff_cluster_df, new_ff_cluster_df, parallel_old_ff_cluster_df, non_chosen_ff_cluster_df


def further_process_df_related_to_cluster_replacement(joined_old_ff_cluster_df, joined_new_ff_cluster_df, num_old_ff_per_row=3, num_new_ff_per_row=3, selection_criterion_if_too_many_ff='time_since_last_vis', sorting_criterion=None):
    '''
    Further process the dataframes related to cluster replacement, including:
    1. Guarantee that there are num_old_ff_per_row or num_new_ff_per_row of ff for each point_index
    2. Make sure that the intended target will not be removed when there are too many ff
    3. Sort the remaining ff by sorting_criterion, but make sure that the intended target in the old ff cluster will be the first one
    '''

    # check if the "selection_criterion_if_too_many_ff" column contains NA values
    if joined_old_ff_cluster_df[selection_criterion_if_too_many_ff].isnull().values.any():
        raise ValueError(
            'The column "selection_criterion_if_too_many_ff" contains NA values in joined_old_ff_cluster_df')
    elif joined_new_ff_cluster_df[selection_criterion_if_too_many_ff].isnull().values.any():
        raise ValueError(
            'The column "selection_criterion_if_too_many_ff" contains NA values in joined_new_ff_cluster_df')

    joined_old_ff_cluster_df['selection_criterion'] = joined_old_ff_cluster_df[selection_criterion_if_too_many_ff]
    joined_new_ff_cluster_df['selection_criterion'] = joined_new_ff_cluster_df[selection_criterion_if_too_many_ff]
    # make sure that the intended target will not be removed when there are too many ff
    joined_old_ff_cluster_df.loc[joined_old_ff_cluster_df['whether_intended_target']
                                 == True, 'selection_criterion'] = -9999

    # make sure that there are num_old_ff_per_row or num_new_ff_per_row of ff for each point_index
    original_joined_old_ff_cluster_df = joined_old_ff_cluster_df.copy()
    joined_old_ff_cluster_df = free_selection.guarantee_n_ff_per_point_index_in_ff_dataframe(joined_old_ff_cluster_df, np.unique(
        joined_old_ff_cluster_df.point_index.values), num_ff_per_row=num_old_ff_per_row)

    # Compute leftovers from OLD via proper anti-join on (ff_index, point_index)
    chosen_keys = joined_old_ff_cluster_df[[
        'ff_index', 'point_index']].drop_duplicates()
    leftover_old_ff_cluster_df = (
        original_joined_old_ff_cluster_df
        .merge(chosen_keys, on=['ff_index', 'point_index'], how='left', indicator=True)
        .loc[lambda d: d['_merge'] == 'left_only']
        .drop(columns=['_merge'])
    )

    # Seed NEW with its own candidates + OLD leftovers
    joined_new_ff_cluster_df = pd.concat(
        [joined_new_ff_cluster_df, leftover_old_ff_cluster_df],
        axis=0, ignore_index=True
    ).reset_index(drop=True)

    joined_new_ff_cluster_df = add_features_GUAT_and_TAFT.retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        joined_new_ff_cluster_df, joined_old_ff_cluster_df, columns=['point_index', 'ff_index'], whether_share=False)
    joined_new_ff_cluster_df = free_selection.guarantee_n_ff_per_point_index_in_ff_dataframe(joined_new_ff_cluster_df, np.unique(
        joined_old_ff_cluster_df.point_index.values), num_ff_per_row=num_new_ff_per_row)

    # fill out NAs
    joined_old_ff_cluster_df['whether_intended_target'] = joined_old_ff_cluster_df['whether_intended_target'].fillna(
        False).infer_objects()
    joined_new_ff_cluster_df['whether_intended_target'] = joined_new_ff_cluster_df['whether_intended_target'].fillna(
        False).infer_objects()
    joined_old_ff_cluster_df['whether_changed'] = joined_old_ff_cluster_df['whether_changed'].fillna(
        False).infer_objects()
    joined_new_ff_cluster_df['whether_changed'] = joined_new_ff_cluster_df['whether_changed'].fillna(
        False).infer_objects()

    # but whether_changed should depend on the point_index
    _point_index = joined_old_ff_cluster_df[joined_old_ff_cluster_df['whether_changed']
                                            == True]['point_index'].values
    joined_old_ff_cluster_df.loc[joined_old_ff_cluster_df['point_index'].isin(
        _point_index), 'whether_changed'] = True
    joined_new_ff_cluster_df.loc[joined_new_ff_cluster_df['point_index'].isin(
        _point_index), 'whether_changed'] = True

    if sorting_criterion is None:
        joined_old_ff_cluster_df.sort_values(
            by=['point_index', 'selection_criterion'], inplace=True)
        joined_new_ff_cluster_df.sort_values(
            by=['point_index', 'selection_criterion'], inplace=True)
    else:
        # sort the remaining ff by sorting_criterion, but make sure that the intended target in the old ff cluster will be the first one
        joined_old_ff_cluster_df['sorting_criterion'] = joined_old_ff_cluster_df[sorting_criterion]
        joined_new_ff_cluster_df['sorting_criterion'] = joined_new_ff_cluster_df[sorting_criterion]
        joined_old_ff_cluster_df.loc[joined_old_ff_cluster_df['whether_intended_target']
                                     == True, 'sorting_criterion'] = -9999
        joined_old_ff_cluster_df.sort_values(
            by=['point_index', 'sorting_criterion'], inplace=True)
        joined_new_ff_cluster_df.sort_values(
            by=['point_index', 'sorting_criterion'], inplace=True)

    # add order column
    joined_old_ff_cluster_df['order'] = np.tile(range(num_old_ff_per_row), int(
        len(joined_old_ff_cluster_df)/num_old_ff_per_row))
    joined_new_ff_cluster_df['order'] = np.tile(range(num_new_ff_per_row), int(
        len(joined_new_ff_cluster_df)/num_new_ff_per_row))

    # reset index
    joined_old_ff_cluster_df.reset_index(drop=True, inplace=True)
    joined_new_ff_cluster_df.reset_index(drop=True, inplace=True)

    return joined_old_ff_cluster_df, joined_new_ff_cluster_df


def eliminate_close_by_pairs_between_old_and_new_ff_info(old_ff_info, new_ff_info, all_time, ff_real_position_sorted, min_distance_between_old_and_new_ff=50):
    # Among replacement rows, find ones where the two ff (before and after) are not considered to be in the same cluster.

    # find the distance between the old ff and the new ff
    old_ff_positions = ff_real_position_sorted[old_ff_info.ff_index.values]
    new_ff_positions = ff_real_position_sorted[new_ff_info.ff_index.values]
    old_ff_to_new_ff_distance = np.linalg.norm(
        old_ff_positions - new_ff_positions, axis=1)
    in_same_cluster = np.where(
        old_ff_to_new_ff_distance <= min_distance_between_old_and_new_ff)[0]
    not_in_same_cluster = np.where(
        old_ff_to_new_ff_distance > min_distance_between_old_and_new_ff)[0]
    # print('The percentage of new ff that are in the same cluster as the old ff is', round(len(in_same_cluster)/len(old_ff_info)*100, 3), '%')

    # remove rows where the old ff and the new ff are in the same cluster
    old_ff_info = old_ff_info.iloc[not_in_same_cluster]
    new_ff_info = new_ff_info.iloc[not_in_same_cluster]
    old_ff_positions = old_ff_positions[not_in_same_cluster]
    new_ff_positions = new_ff_positions[not_in_same_cluster]
    all_point_index = old_ff_info['point_index'].values
    all_time = all_time[not_in_same_cluster]

    return old_ff_info, new_ff_info, old_ff_positions, new_ff_positions, all_point_index, all_time


def eliminate_rows_with_large_value_in_shared_column_between_df(shared_column, max_value, df_1, df_2):
    # First, we remove specific rows where the shared_column is too large
    df_1 = df_1[df_1[shared_column] < max_value]
    df_2 = df_2[df_2[shared_column] < max_value]

    # Then, we need to make sure that there is at least one valid row for each point_index in both df.
    # Otherwise, the rows associated with the point_index will all be removed
    df_1_valid_point_index = df_1.point_index.values
    df_2_valid_point_index = df_2.point_index.values
    shared_valid_point_index = np.intersect1d(
        df_1_valid_point_index, df_2_valid_point_index)
    df_1 = df_1[df_1['point_index'].isin(shared_valid_point_index)]
    df_2 = df_2[df_2['point_index'].isin(shared_valid_point_index)]
    return df_1, df_2


def mark_intended_target_in_df(df, intended_target_df):
    intended_target_df_sub = intended_target_df[[
        'ff_index', 'point_index']].copy()
    intended_target_df_sub['whether_intended_target'] = True
    if 'whether_intended_target' in df.columns:
        df.drop(['whether_intended_target'], axis=1, inplace=True)
    df = pd.merge(df, intended_target_df_sub, on=[
                  'ff_index', 'point_index'], how='left')
    df['whether_intended_target'].fillna(False, inplace=True)
    return df


def find_non_chosen_ff_info_and_parallel_old_ff_info_with_no_close_by_pairs(non_chosen_df, manual_anno, ff_real_position_sorted, ff_life_sorted, ff_dataframe, monkey_information):
    non_chosen_ff_info, all_point_index, all_time = replacement.get_non_chosen_ff_info(
        non_chosen_df, selection_criterion='abs_curv_diff', one_row_per_group=False)
    parallel_old_ff_info, position_indices_with_valid_ff = replacement.get_parallel_old_ff_info(
        all_point_index, all_time, manual_anno, ff_real_position_sorted, ff_dataframe, monkey_information)
    # update the variables based on valid ff_index
    all_point_index = all_point_index[position_indices_with_valid_ff]
    all_time = all_time[position_indices_with_valid_ff]
    non_chosen_ff_info = non_chosen_ff_info.iloc[position_indices_with_valid_ff].copy(
    )

    # eliminate rows where the non-chosen ff are in the same cluster as the old ff.
    parallel_old_ff_info, non_chosen_ff_info, parallel_old_ff_positions, non_chosen_ff_positions, all_point_index, all_time = eliminate_close_by_pairs_between_old_and_new_ff_info(
        parallel_old_ff_info, non_chosen_ff_info, all_time, ff_real_position_sorted)

    # use merge to preserve only rows where the non-chosen ff and the old ff are not in the same cluster
    non_chosen_df = pd.merge(non_chosen_df, non_chosen_ff_info[[
                             'ff_index', 'point_index']], on=['ff_index', 'point_index'], how='inner')
    non_chosen_df['group'] = non_chosen_df['time'].apply(lambda x: int(x/0.35))

    # preserve only one row per group, sort by the selection criterion
    non_chosen_df = non_chosen_df.sort_values(
        ['abs_curv_diff', 'time_since_last_vis'], ascending=[True, True])
    non_chosen_df = non_chosen_df.groupby(['group']).first().reset_index()
    non_chosen_ff_info = non_chosen_df[[
        'ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']].copy()

    # get the parallel_old_ff_info
    all_point_index = non_chosen_df['point_index'].values
    all_time = non_chosen_df['time'].values
    parallel_old_ff_info, position_indices_with_valid_ff = replacement.get_parallel_old_ff_info(
        all_point_index, all_time, manual_anno, ff_real_position_sorted, ff_dataframe, monkey_information)
    # update the variables based on valid ff_index
    all_point_index = all_point_index[position_indices_with_valid_ff]
    all_time = all_time[position_indices_with_valid_ff]
    non_chosen_ff_info = non_chosen_ff_info.iloc[position_indices_with_valid_ff].copy(
    )

    # calculate the positions again
    non_chosen_ff_positions = ff_real_position_sorted[non_chosen_ff_info.ff_index.values]
    parallel_old_ff_positions = ff_real_position_sorted[parallel_old_ff_info.ff_index.values]

    return non_chosen_ff_info, parallel_old_ff_info, non_chosen_ff_positions, parallel_old_ff_positions, all_point_index, all_time


def find_more_ff_inputs_for_plotting(point_index_all, sequence_of_obs_ff_indices, ff_dataframe, ff_real_position_sorted, monkey_information, all_available_ff_in_near_future=None,
                                     attributes_for_plotting=['ff_distance', 'ff_angle', 'time_since_last_vis'], return_all_attributes=False):
    more_ff_df = find_more_ff_df(point_index_all, ff_dataframe, ff_real_position_sorted, monkey_information,
                                 all_available_ff_in_near_future=all_available_ff_in_near_future, attributes_for_plotting=attributes_for_plotting)
    more_ff_df = eliminate_part_of_more_ff_inputs_already_in_observation(
        more_ff_df, sequence_of_obs_ff_indices, point_index_all)
    more_ff_inputs_df_for_plotting = turn_more_ff_df_into_free_selection_x_df_for_plotting(
        more_ff_df, point_index_all, return_all_attributes=return_all_attributes, attributes_for_plotting=attributes_for_plotting)
    return more_ff_df, more_ff_inputs_df_for_plotting


def find_more_ff_df(point_index_all, ff_dataframe, ff_real_position_sorted, monkey_information, all_available_ff_in_near_future=None, attributes_for_plotting=['ff_distance', 'ff_angle', 'time_since_last_vis']):
    more_ff_df = ff_dataframe[ff_dataframe['point_index'].isin(
        point_index_all)].sort_values(by='point_index').copy()
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()
    all_possible_ff = more_ff_df[['ff_index', 'point_index']].copy()
    if all_available_ff_in_near_future is not None:
        all_possible_ff = pd.concat([more_ff_df[['ff_index', 'point_index']],
                                    all_available_ff_in_near_future], axis=0).reset_index(drop=True)

    if 'time_till_next_visible' in attributes_for_plotting:
        more_ff_df = decision_making_utils.find_many_ff_info_anew(
            all_possible_ff.ff_index.values, all_possible_ff.point_index.values, ff_real_position_sorted, ff_dataframe_visible, monkey_information, add_time_till_next_visible=True)
        more_ff_df['time_till_next_visible'] = more_ff_df['time_till_next_visible'].clip(
            upper=10)
        more_ff_df = more_ff_df[(more_ff_df['time_since_last_vis'] <= 3) | (
            more_ff_df['time_till_next_visible'] <= 2)].copy()
    else:
        more_ff_df = decision_making_utils.find_many_ff_info_anew(
            all_possible_ff.ff_index.values, all_possible_ff.point_index.values, ff_real_position_sorted, ff_dataframe_visible, monkey_information, add_time_till_next_visible=False)
        more_ff_df = more_ff_df[more_ff_df['time_since_last_vis'] <= 3].copy(
        )

    more_ff_df.sort_values(by='point_index', inplace=True)
    more_ff_df.drop_duplicates(
        subset=['ff_index', 'point_index'], inplace=True)
    more_ff_df['ff_number'] = more_ff_df.groupby('point_index').cumcount()+201

    return more_ff_df


def eliminate_part_of_more_ff_inputs_already_in_observation(more_ff_inputs, sequence_of_obs_ff_indices, point_index_all):
    ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        sequence_of_obs_ff_indices, point_index_all)
    ff_cluster_df = ff_cluster_df[ff_cluster_df['ff_index'] >= 0].copy()
    ff_cluster_df['in_obs'] = True
    # eliminate ff in more_ff_inputs that's already in the observation at the same point.
    more_ff_inputs = pd.merge(more_ff_inputs, ff_cluster_df[[
                              'ff_index', 'point_index', 'in_obs']], on=['ff_index', 'point_index'], how='left')
    more_ff_inputs = more_ff_inputs[more_ff_inputs['in_obs'] != True].copy()
    more_ff_inputs.drop('in_obs', axis=1, inplace=True)
    return more_ff_inputs


def turn_more_ff_df_into_free_selection_x_df_for_plotting(more_ff_df, point_index_all, return_all_attributes=False, attributes_for_plotting=['ff_distance', 'ff_angle', 'time_since_last_vis']):
    # turn more_ff_df into a format that can be used to produce free_selection_x
    max_num_ff = more_ff_df.groupby('point_index').count().max().iloc[0]
    num_ff_per_row = max_num_ff
    more_ff_df = free_selection.guarantee_n_ff_per_point_index_in_ff_dataframe(
        more_ff_df, point_index_all, num_ff_per_row=num_ff_per_row)
    more_ff_df = more_ff_df.sort_values(by=['point_index']).copy()
    more_ff_df['order'] = np.tile(
        range(num_ff_per_row), int(len(more_ff_df)/num_ff_per_row))
    free_selection_x_df, free_selection_x_df_for_plotting, sequence_of_obs_ff_indices, point_index_array, pred_var = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(more_ff_df,
                                                                                                                                                                                      attributes_for_plotting=attributes_for_plotting, num_ff_per_row=num_ff_per_row)
    if return_all_attributes:
        return free_selection_x_df
    else:
        return free_selection_x_df_for_plotting


def add_time_till_next_visible(df, ff_dataframe_visible, monkey_information):
    all_current_time = monkey_information.loc[df['point_index'].values, 'time'].values
    df['time_till_next_visible'] = decision_making_utils.find_time_since_last_vis_OR_time_till_next_visible(
        df.ff_index.values, all_current_time, ff_dataframe_visible, time_since_last_vis=False)


def supply_info_of_ff_last_seen_and_next_seen_to_df(df, ff_dataframe, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=None,
                                                    attributes_to_add=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y']):
    if curv_of_traj_df is None:
        raise ValueError('curv_of_traj_df cannot be None')

    # we also want to find distance_from_monkey_now_to_monkey_when_ff_last_seen and angle_from_monkey_now_to_monkey_when_ff_last_seen
    if 'monkey_x' not in attributes_to_add:
        attributes_to_add.append('monkey_x')
    if 'monkey_y' not in attributes_to_add:
        attributes_to_add.append('monkey_y')
    df = decision_making_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(
        df, ff_dataframe, attributes=attributes_to_add)
    df = trajectory_info.add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_dataframe, monkey_xy_from_other_time=df[['last_seen_monkey_x', 'last_seen_monkey_y']].values)
    df = decision_making_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(
        df, ff_dataframe, attributes=attributes_to_add, use_last_seen=False)
    df = trajectory_info.add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_dataframe, monkey_xy_from_other_time=df[['next_seen_monkey_x', 'next_seen_monkey_y']].values, use_last_seen=False)

    df = trajectory_info.add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, ff_dataframe, monkey_information)
    df = trajectory_info.add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, ff_dataframe, monkey_information, use_last_seen=False)
    df = trajectory_info.add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
    df = trajectory_info.add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=curv_of_traj_df, use_last_seen=False)

    return df
