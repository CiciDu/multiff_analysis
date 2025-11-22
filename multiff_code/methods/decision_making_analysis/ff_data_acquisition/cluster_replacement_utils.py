from decision_making_analysis.ff_data_acquisition import ff_data_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data
from decision_making_analysis.ff_data_acquisition import free_selection
from decision_making_analysis.data_enrichment import trajectory_utils
from pattern_discovery import cluster_analysis
from data_wrangling import general_utils

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

    joined_new_ff_cluster_df = get_missed_ff_data.retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        joined_new_ff_cluster_df, joined_old_ff_cluster_df, columns=['point_index', 'ff_index'], whether_share=False)
    joined_new_ff_cluster_df = free_selection.guarantee_n_ff_per_point_index_in_ff_dataframe(joined_new_ff_cluster_df, np.unique(
        joined_old_ff_cluster_df.point_index.values), num_ff_per_row=num_new_ff_per_row)

    # fill out NAs
    joined_old_ff_cluster_df['whether_intended_target'] = joined_old_ff_cluster_df['whether_intended_target'] == True
    joined_new_ff_cluster_df['whether_intended_target'] = joined_new_ff_cluster_df['whether_intended_target'] == True
    joined_old_ff_cluster_df['whether_changed'] = joined_old_ff_cluster_df['whether_changed'] == True
    joined_new_ff_cluster_df['whether_changed'] = joined_new_ff_cluster_df['whether_changed'] == True

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


def supply_info_of_ff_last_seen_and_next_seen_to_df(df, ff_dataframe, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=None,
                                                    attributes_to_add=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y']):
    if curv_of_traj_df is None:
        raise ValueError('curv_of_traj_df cannot be None')

    # we also want to find distance_from_monkey_now_to_monkey_when_ff_last_seen and angle_from_monkey_now_to_monkey_when_ff_last_seen
    if 'monkey_x' not in attributes_to_add:
        attributes_to_add.append('monkey_x')
    if 'monkey_y' not in attributes_to_add:
        attributes_to_add.append('monkey_y')
    df = ff_data_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(
        df, ff_dataframe, attributes=attributes_to_add)
    df = trajectory_utils.add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_dataframe, monkey_xy_from_other_time=df[['last_seen_monkey_x', 'last_seen_monkey_y']].values)
    df = ff_data_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(
        df, ff_dataframe, attributes=attributes_to_add, use_last_seen=False)
    df = trajectory_utils.add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_dataframe, monkey_xy_from_other_time=df[['next_seen_monkey_x', 'next_seen_monkey_y']].values, use_last_seen=False)

    df = trajectory_utils.add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, ff_dataframe, monkey_information)
    df = trajectory_utils.add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, ff_dataframe, monkey_information, use_last_seen=False)
    df = trajectory_utils.add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
    df = trajectory_utils.add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(
        df, monkey_information, ff_real_position_sorted, ff_caught_T_new, curv_of_traj_df=curv_of_traj_df, use_last_seen=False)

    return df
