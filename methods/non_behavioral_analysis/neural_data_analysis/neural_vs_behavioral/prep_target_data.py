import sys
from data_wrangling import process_monkey_information, specific_utils
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_target_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_dataframe,
                   include_frozen_info=True):
    target_df = _initialize_target_df(monkey_information, ff_caught_T_new)

    target_df = _add_target_df_info(target_df, ff_real_position_sorted, ff_dataframe, ff_caught_T_new, include_frozen_info=include_frozen_info)
    
    return target_df


def _add_target_df_info(target_df, ff_real_position_sorted, ff_dataframe, ff_caught_T_new, include_frozen_info=True):
    target_df = _calculate_target_distance_and_angle(
        target_df, ff_real_position_sorted)

    # Add target last seen info
    print("Calculating target-last-seen info.")
    target_df = _add_target_last_seen_info(
        target_df, ff_dataframe, include_frozen_info=include_frozen_info)

    target_df = _add_target_disappeared_for_last_time_dummy(
        target_df, ff_caught_T_new, ff_dataframe)

    target_df = _add_target_visible_dummy(target_df)

    target_df = _find_time_since_last_capture(target_df, ff_caught_T_new)
    
    target_df = _add_target_rel_x_and_y(target_df)
    
    return target_df


def _add_target_rel_x_and_y(target_df):
    target_df['target_rel_y'] = target_df['target_distance'] * \
        np.cos(target_df['target_angle'])
    target_df['target_rel_x'] = - target_df['target_distance'] * \
        np.sin(target_df['target_angle'])
    return target_df     
            
def make_target_cluster_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_dataframe, ff_life_sorted, include_frozen_info=True):
    target_clust_df = _initialize_target_df(
        monkey_information, ff_caught_T_new)

    target_clust_df, nearby_alive_ff_indices = _add_target_cluster_last_seen_info(
        target_clust_df, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe, include_frozen_info=include_frozen_info)

    target_clust_df = _add_target_cluster_disappeared_for_last_time_dummy(
        target_clust_df, ff_caught_T_new, ff_dataframe, nearby_alive_ff_indices)

    target_clust_df = _add_target_cluster_visible_dummy(target_clust_df)

    target_clust_df = _add_while_last_seeing_target_cluster_dummy(
        target_clust_df)

    return target_clust_df


def get_max_min_and_avg_info_from_target_df(target_df):
    target_average_info = _calculate_average_info(target_df)
    target_min_info = _calculate_min_info(target_df)
    target_max_info = _calculate_max_info(target_df)
    return target_average_info, target_min_info, target_max_info


def _initialize_target_df(monkey_information, ff_caught_T_new):
    """
    Create a DataFrame with target information.
    """
    target_df = monkey_information[[
        'point_index', 'time', 'monkey_x', 'monkey_y', 'monkey_angle', 'cum_distance']].copy()
    target_df['target_index'] = np.searchsorted(
        ff_caught_T_new, target_df['time'])
    return target_df


def _calculate_target_distance_and_angle(target_df, ff_real_position_sorted):
    """
    Calculate target distance and angle.
    """
    target_df['target_x'] = ff_real_position_sorted[target_df['target_index'].values, 0]
    target_df['target_y'] = ff_real_position_sorted[target_df['target_index'].values, 1]
    target_distance = np.sqrt((target_df['target_x'] - target_df['monkey_x'])**2 + (
        target_df['target_y'] - target_df['monkey_y'])**2)
    target_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=target_df['target_x'], ff_y=target_df['target_y'], mx=target_df['monkey_x'], my=target_df['monkey_y'], m_angle=target_df['monkey_angle'])
    target_df['target_distance'] = target_distance
    target_df['target_angle'] = target_angle
    target_df['target_angle_to_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=target_angle, distances_to_ff=target_distance)
    return target_df


def _add_target_cluster_last_seen_info(target_df, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe, include_frozen_info=True):
    if 'target_cluster_last_seen_time' not in target_df.columns:
        nearby_alive_ff_indices = cluster_analysis.find_alive_target_clusters(
            ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, max_distance=50)
        print("Calculating target-cluster-last-seen info.")
        target_df = _add_target_last_seen_info(
            target_df, ff_dataframe, nearby_alive_ff_indices, use_target_cluster=True, include_frozen_info=include_frozen_info)
        target_df = target_df.rename(columns={'time_since_target_last_seen': 'target_cluster_last_seen_time',
                                              'target_last_seen_distance': 'target_cluster_last_seen_distance',
                                              'target_last_seen_angle': 'target_cluster_last_seen_angle',
                                              'target_last_seen_angle_to_boundary': 'target_cluster_last_seen_angle_to_boundary',
                                              'target_last_seen_distance_frozen': 'target_cluster_last_seen_distance_frozen',
                                              'target_last_seen_angle_frozen': 'target_cluster_last_seen_angle_frozen',
                                              'target_last_seen_angle_to_boundary_frozen': 'target_cluster_last_seen_angle_to_boundary_frozen',
                                              'monkey_x_target_last_seen_frozen': 'monkey_x_target_cluster_last_seen_frozen',
                                              'monkey_y_target_last_seen_frozen': 'monkey_y_target_cluster_last_seen_frozen',
                                              'monkey_angle_target_last_seen_frozen': 'monkey_angle_target_cluster_last_seen_frozen',
                                              'cum_distance_target_last_seen_frozen': 'cum_distance_target_cluster_last_seen_frozen',
                                              })
    return target_df, nearby_alive_ff_indices


def _add_target_disappeared_for_last_time_dummy(target_df, ff_caught_T_new, ff_dataframe):
    """
    Add target_has_disappeared_for_last_time_dummy to target_df
    """
    target_df['target_has_disappeared_for_last_time_dummy'] = 0
    for target in range(len(ff_caught_T_new)):
        target_visible_info = ff_dataframe[(
            ff_dataframe['ff_index'] == target) & (ff_dataframe['visible'] == 1)]
        target_last_vis_time = target_visible_info['time'].max()
        target_df.loc[(target_df['target_index'] == target) & (
            target_df['time'] > target_last_vis_time), 'target_has_disappeared_for_last_time_dummy'] = 1

    return target_df


def _add_target_cluster_disappeared_for_last_time_dummy(target_df, ff_caught_T_new, ff_dataframe, nearby_alive_ff_indices):
    target_df['target_cluster_has_disappeared_for_last_time_dummy'] = 0
    for target in range(len(ff_caught_T_new)):
        target_cluster_indices = nearby_alive_ff_indices[target]
        target_visible_info = ff_dataframe[(ff_dataframe['ff_index'].isin(
            target_cluster_indices)) & (ff_dataframe['visible'] == 1)]
        target_last_vis_time = target_visible_info['time'].max()
        target_df.loc[(target_df['target_index'] == target) & (
            target_df['time'] > target_last_vis_time), 'target_cluster_has_disappeared_for_last_time_dummy'] = 1
    return target_df


def _add_target_visible_dummy(target_df):
    """
    Add dummy variable of target being visible
    """
    target_df[['target_visible_dummy']] = 1
    target_df.loc[target_df['time_since_target_last_seen']
                  > 0, 'target_visible_dummy'] = 0
    return target_df


def _add_target_cluster_visible_dummy(target_df):
    """
    Add dummy variable of target cluster being visible
    """
    target_df[['target_cluster_visible_dummy']] = 1
    target_df.loc[target_df['target_cluster_last_seen_time']
                  > 0, 'target_cluster_visible_dummy'] = 0
    return target_df


def _find_time_since_last_capture(target_df, ff_caught_T_new):
    """
    Find time_since_last_capture
    """
    if target_df.target_index.unique().max() >= len(ff_caught_T_new)-1:
        num_exceeding_target = target_df.target_index.unique().max() - \
            (len(ff_caught_T_new)-1)
        ff_caught_T_new_temp = np.concatenate(
            (ff_caught_T_new, np.repeat(target_df.time.max(), num_exceeding_target)))
    else:
        ff_caught_T_new_temp = ff_caught_T_new.copy()
    target_df['current_target_caught_time'] = ff_caught_T_new_temp[target_df['target_index']]
    target_df['last_target_caught_time'] = ff_caught_T_new_temp[target_df['target_index']-1]
    target_df.loc[target_df['target_index']
                  == 0, 'last_target_caught_time'] = 0
    target_df['time_since_last_capture'] = target_df['time'] - \
        target_df['last_target_caught_time']
    return target_df


def _add_while_last_seeing_target_cluster_dummy(target_df):
    """
    Add to target_df dummy variable of being in the last duration of seeing the target cluster
    """
    target_df['while_last_seeing_target_cluster'] = 0
    for target in target_df.target_index.unique():
        target_subset = target_df[target_df.target_index == target]
        if len(target_subset) > 0:
            dif = np.diff(target_subset['target_cluster_visible_dummy'])
            becoming_visible_points = np.where(dif == 1)[0]
            if len(becoming_visible_points) > 0:
                starting_index = becoming_visible_points[-1]+1
            elif target_subset['target_cluster_visible_dummy'].iloc[0] == 1:
                starting_index = 0
            else:
                continue
            stop_being_visible_points = np.where(dif == -1)[0]
            if len(stop_being_visible_points) > 0:
                ending_index = stop_being_visible_points[-1]+1
                if ending_index < starting_index:
                    ending_index = len(target_subset)
            else:
                ending_index = len(target_subset)
            target_df.loc[target_subset.iloc[starting_index:ending_index].index,
                          'while_last_seeing_target_cluster'] = 1
    return target_df


def _calculate_average_info(target_df):
    """
    Calculate average information for each bin in target_df
    """
    target_average_info = target_df[['bin', 'target_distance', 'target_angle', 'target_angle_to_boundary',
                                    'time_since_target_last_seen', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary',
                                     'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
                                     'target_cluster_last_seen_time', 'target_cluster_last_seen_distance', 'target_cluster_last_seen_angle', 'target_cluster_last_seen_angle_to_boundary',
                                     'target_cluster_last_seen_distance_frozen', 'target_cluster_last_seen_angle_frozen', 'target_cluster_last_seen_angle_to_boundary_frozen',]].copy()

    target_average_info = target_average_info.groupby(
        'bin').mean().reset_index(drop=False)
    target_average_info.rename(columns={'target_distance': 'avg_target_distance',
                                        'target_angle': 'avg_target_angle',
                                        'target_angle_to_boundary': 'avg_target_angle_to_boundary',
                                        'time_since_target_last_seen': 'avg_target_last_seen_time',
                                        'target_last_seen_distance': 'avg_target_last_seen_distance',
                                        'target_last_seen_angle': 'avg_target_last_seen_angle',
                                        'target_last_seen_angle_to_boundary': 'avg_target_last_seen_angle_to_boundary',
                                        'target_last_seen_distance_frozen': 'avg_target_last_seen_distance_frozen',
                                        'target_last_seen_angle_frozen': 'avg_target_last_seen_angle_frozen',
                                        'target_last_seen_angle_to_boundary_frozen': 'avg_target_last_seen_angle_to_boundary_frozen',
                                        'target_cluster_last_seen_time': 'avg_target_cluster_last_seen_time',
                                        'target_cluster_last_seen_distance': 'avg_target_cluster_last_seen_distance',
                                        'target_cluster_last_seen_angle': 'avg_target_cluster_last_seen_angle',
                                        'target_cluster_last_seen_angle_to_boundary': 'avg_target_cluster_last_seen_angle_to_boundary',
                                        'target_cluster_last_seen_distance_frozen': 'avg_target_cluster_last_seen_distance_frozen',
                                        'target_cluster_last_seen_angle_frozen': 'avg_target_cluster_last_seen_angle_frozen',
                                        'target_cluster_last_seen_angle_to_boundary_frozen': 'avg_target_cluster_last_seen_angle_to_boundary_frozen'
                                        }, inplace=True)
    return target_average_info


def _calculate_min_info(target_df):
    """
    Calculate minimum information for each bin in target_df
    """
    target_min_info = target_df[['bin', 'target_has_disappeared_for_last_time_dummy',
                                 'target_cluster_has_disappeared_for_last_time_dummy']].copy()
    target_min_info = target_min_info.groupby(
        'bin').min().reset_index(drop=False)
    target_min_info.rename(columns={'target_has_disappeared_for_last_time_dummy': 'min_target_has_disappeared_for_last_time_dummy',
                                    'target_cluster_has_disappeared_for_last_time_dummy': 'min_target_cluster_has_disappeared_for_last_time_dummy',
                                    }, inplace=True)
    return target_min_info


def _calculate_max_info(target_df):
    """
    Calculate maximum information for each bin in target_df
    """
    target_max_info = target_df[[
        'bin', 'target_visible_dummy', 'target_cluster_visible_dummy']].copy()
    target_max_info = target_max_info.groupby(
        'bin').max().reset_index(drop=False)
    target_max_info.rename(columns={'target_visible_dummy': 'max_target_visible_dummy',
                                    'target_cluster_visible_dummy': 'max_target_cluster_visible_dummy'}, inplace=True)
    return target_max_info


def _add_target_last_seen_info(target_df, ff_dataframe, nearby_alive_ff_indices=None, use_target_cluster=False, include_frozen_info=False):
    """
    Add target last seen information to the target DataFrame.

    Parameters:
    - target_df: DataFrame containing target data.
    - ff_dataframe: DataFrame containing firefly data.
    - nearby_alive_ff_indices: Indices of nearby alive fireflies (optional).
    - use_target_cluster: Boolean indicating whether to use target cluster information.
    - include_frozen_info: Boolean indicating whether to include frozen information.

    Returns:
    - Updated target_df with last seen information.
    """
    # Define constants
    DEFAULT_LAST_SEEN_TIME = 100
    DEFAULT_LAST_SEEN_DISTANCE = 400
    DEFAULT_LAST_SEEN_ANGLE = 0

    # Initialize columns with default values
    _add_placeholder_last_seen_values(target_df, DEFAULT_LAST_SEEN_TIME,
                                      DEFAULT_LAST_SEEN_DISTANCE, DEFAULT_LAST_SEEN_ANGLE, include_frozen_info)

    if use_target_cluster:
        if nearby_alive_ff_indices is None:
            raise ValueError(
                "nearby_alive_ff_indices is None, but use_target_cluster is True")

    # Process each target index and add the last-seen info to target_df
    sorted_target_index = np.sort(ff_dataframe['target_index'].unique())
    for i in range(len(sorted_target_index)):
        target_index = sorted_target_index[i]
        if i < len(sorted_target_index) - 1:
            print(
                f'Adding last seen info: target_index = {target_index}', end='\r')
        else:
            print(f'Adding last seen info: target_index = {target_index}')
        target_df = _update_info_for_current_target_index(
            target_df, ff_dataframe, target_index, nearby_alive_ff_indices, use_target_cluster=use_target_cluster, include_frozen_info=include_frozen_info)

    target_df['point_index'] = target_df['point_index'].astype(int)

    return target_df


def _add_placeholder_last_seen_values(target_df, last_seen_time, last_seen_distance, last_seen_angle, include_frozen_info):
    """
    Initialize columns with default values and set their dtype to float.
    """
    columns = ['time_since_target_last_seen', 'target_last_seen_distance',
               'target_last_seen_angle', 'target_last_seen_angle_to_boundary']
    target_df[columns] = [last_seen_time,
                          last_seen_distance, last_seen_angle, last_seen_angle]
    target_df[columns] = target_df[columns].astype(float)

    if include_frozen_info:
        frozen_columns = ['target_last_seen_distance_frozen',
                          'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen']
        target_df[frozen_columns] = [
            last_seen_distance, last_seen_angle, last_seen_angle]
        target_df[frozen_columns] = target_df[frozen_columns].astype(float)

        target_df[['monkey_x_target_last_seen_frozen', 'monkey_y_target_last_seen_frozen',
                   'monkey_angle_target_last_seen_frozen', 'cum_distance_target_last_seen_frozen']] = np.nan


def _update_info_for_current_target_index(target_df, ff_dataframe, target_index, nearby_alive_ff_indices, use_target_cluster=False, include_frozen_info=True):
    """
    Get target_visible_info for the current target index and update the target DataFrame with last seen information.
    """
    # get subset of target_df associated with the current target_index, as well as the corresponding row indices in the original target_df
    target_sub_row_indices = np.array(
        (target_df['target_index'] == target_index)).nonzero()[0]
    target_df_sub = target_df.iloc[target_sub_row_indices].copy()

    # get target_visible_info for the current target_index anew from ff_dataframe, in case that info in target_df_sub is incomplete
    target_visible_info = _get_target_info_when_visible_for_current_target_index(
        ff_dataframe, target_index, nearby_alive_ff_indices, use_target_cluster)

    if len(target_df_sub) > 0:
        # In case that target_df can contain multiple rows for each time point, we can take out unique time points
        unique_time_points = target_df_sub[['point_index', 'time', 'monkey_x',
                                            'monkey_y', 'monkey_angle', 'cum_distance']].drop_duplicates().sort_values(by='point_index')
        # Adjust the duration contained in unique_time_points
        unique_time_points, target_visible_info, target_df_sub, target_sub_row_indices = _adjust_duration_of_unique_time_points(
            unique_time_points, target_df_sub, target_visible_info, target_sub_row_indices)
        target_visible_info = _polish_target_info(target_visible_info)

        if len(target_visible_info) > 0:
            unique_time_points = _get_last_seen_info(
                unique_time_points, target_visible_info, include_frozen_info=include_frozen_info)
            # put back the updated part of target_df
            target_df = _update_target_df_for_current_target_index(
                target_df, target_df_sub, target_sub_row_indices, unique_time_points)
    return target_df


def _get_target_info_when_visible_for_current_target_index(ff_dataframe, target_index, nearby_alive_ff_indices, use_target_cluster):
    if use_target_cluster:
        target_cluster_indices = nearby_alive_ff_indices[target_index]
        target_visible_info = ff_dataframe[(ff_dataframe['ff_index'].isin(
            target_cluster_indices)) & (ff_dataframe['visible'] == 1)].copy()
    else:
        target_visible_info = ff_dataframe[(ff_dataframe['ff_index'] == target_index) & (
            ff_dataframe['visible'] == 1)].copy()
    return target_visible_info


def _adjust_duration_of_unique_time_points(unique_time_points, target_df_sub, target_visible_info, target_sub_row_indices):
    """
    If there's information about target before unique_time_points, then attach the last row of that information to the beginning of unique_time_points.
    Otherwise, only keep the part in unique_time_points that has time points later than when target_visible_info starts.
    """

    # get the minimum point_index for the current target_index in target_df
    min_point = unique_time_points.point_index.min()

    if len(target_visible_info) > 0:
        # divide target_visible_info into two parts: before and after min_point
        target_info_before = target_visible_info[target_visible_info['point_index'] <= min_point]
        # if there's information about target before unique_time_points, then attach the last row of that information to the beginning of unique_time_points
        if len(target_info_before) > 0:
            target_visible_info = target_visible_info[target_visible_info['point_index']
                                                      >= target_info_before.iloc[-1].point_index]
            starting_info = target_info_before.iloc[[-1]][[
                'point_index', 'time', 'monkey_x', 'monkey_y', 'monkey_angle']]
            unique_time_points = pd.concat([starting_info, unique_time_points])
            unique_time_points = unique_time_points.drop_duplicates(
                subset=['point_index'], keep='first')
        else:  # We eliminate the part in unique_time_points that don't have any target info preceding it or at the same time.
            # These will stay in the original dataframe with default values (a.k.a. we will update target_df)
            valid_points = np.where(np.array(
                unique_time_points['point_index']) >= target_visible_info.point_index.min())[0]
            unique_time_points = unique_time_points.iloc[valid_points]
            target_sub_row_indices = target_sub_row_indices[valid_points]
            target_df_sub = target_df_sub.iloc[valid_points]
    return unique_time_points, target_visible_info, target_df_sub, target_sub_row_indices


def _polish_target_info(target_visible_info):
    target_visible_info = target_visible_info.sort_values(
        by=['point_index', 'ff_index']).drop_duplicates(subset=['point_index'], keep='first')
    target_visible_info = target_visible_info[[
        'point_index', 'time', 'ff_x', 'ff_y', 'monkey_x', 'monkey_y', 'monkey_angle', 'cum_distance']]
    target_visible_info = target_visible_info.rename(columns={'time': 'target_time', 'ff_x': 'target_x', 'ff_y': 'target_y',
                                                              'monkey_x': 'frozen_monkey_x', 'monkey_y': 'frozen_monkey_y',
                                                              'monkey_angle': 'frozen_monkey_angle', 'cum_distance': 'frozen_cum_distance'})
    return target_visible_info


def _get_last_seen_info(unique_time_points, target_visible_info, include_frozen_info=True):
    """
    Merge unique_time_points (which has one row per time point) with target_visible_info (which has multiple rows per time point) to get last seen information.
    """
    unique_time_points = unique_time_points.merge(
        target_visible_info, how='left', on='point_index')
    unique_time_points.ffill(inplace=True)

    target_x, target_y = unique_time_points.target_x, unique_time_points.target_y
    monkey_x, monkey_y, monkey_angle = unique_time_points.monkey_x, unique_time_points.monkey_y, unique_time_points.monkey_angle
    target_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=target_x, ff_y=target_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
    target_distance = np.sqrt((target_x - monkey_x) **
                              2 + (target_y - monkey_y) ** 2)
    unique_time_points['time_since_target_last_seen'] = unique_time_points.time - \
        unique_time_points.target_time
    unique_time_points['target_last_seen_distance'] = target_distance
    unique_time_points['target_last_seen_angle'] = target_angle
    unique_time_points['target_last_seen_angle_to_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=target_angle, distances_to_ff=target_distance)

    if include_frozen_info:
        monkey_x, monkey_y, monkey_angle = unique_time_points.frozen_monkey_x, unique_time_points.frozen_monkey_y, unique_time_points.frozen_monkey_angle
        target_angle = specific_utils.calculate_angles_to_ff_centers(
            ff_x=target_x, ff_y=target_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
        target_distance = np.sqrt(
            (target_x - monkey_x) ** 2 + (target_y - monkey_y) ** 2)
        unique_time_points['target_last_seen_distance_frozen'] = target_distance
        unique_time_points['target_last_seen_angle_frozen'] = target_angle
        unique_time_points['target_last_seen_angle_to_boundary_frozen'] = specific_utils.calculate_angles_to_ff_boundaries(
            angles_to_ff=target_angle, distances_to_ff=target_distance)

        unique_time_points['monkey_x_target_last_seen_frozen'] = unique_time_points.frozen_monkey_x
        unique_time_points['monkey_y_target_last_seen_frozen'] = unique_time_points.frozen_monkey_y
        unique_time_points['monkey_angle_target_last_seen_frozen'] = unique_time_points.frozen_monkey_angle
        unique_time_points['cum_distance_target_last_seen_frozen'] = unique_time_points.frozen_cum_distance

        essential_columns = ['point_index', 'time_since_target_last_seen', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary',
                             'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
                             'monkey_x_target_last_seen_frozen', 'monkey_y_target_last_seen_frozen', 'monkey_angle_target_last_seen_frozen', 'cum_distance_target_last_seen_frozen']
    else:
        essential_columns = ['point_index', 'time_since_target_last_seen', 'target_last_seen_distance',
                             'target_last_seen_angle', 'target_last_seen_angle_to_boundary',
                             'monkey_x_target_last_seen', 'monkey_y_target_last_seen', 'monkey_angle_target_last_seen',
                             ]

    unique_time_points = unique_time_points[essential_columns]

    return unique_time_points


def _update_target_df_for_current_target_index(target_df, target_df_sub, target_sub_row_indices, unique_time_points):
    """
    Update the target DataFrame with the calculated values.
    """
    target_df_sub_new = target_df_sub[['point_index']].merge(
        unique_time_points, on='point_index', how='left')
    column_indexes = target_df.columns.get_indexer(
        target_df_sub_new.columns)

    if len(column_indexes[column_indexes < 0]) > 0:
        raise ValueError(
            "Some columns in target_df_sub_new do not exist in target_df; updating failed.")

    target_df.iloc[target_sub_row_indices,
                   column_indexes] = target_df_sub_new.values
    return target_df


def add_num_stops_to_target_last_vis_df(target_last_vis_df, ff_caught_T_new, num_stops, num_stops_near_target, num_stops_since_last_vis):
    """
    Add the number of stops information to the target last visit DataFrame.

    Parameters:
    - target_last_vis_df: DataFrame containing target last visit data.
    - ff_caught_T_new: Array of caught fireflies.
    - num_stops: Number of stops.
    - num_stops_near_target: Number of stops near the target.
    - num_stops_since_last_vis: Number of stops since the last visit.

    Returns:
    - Updated target_last_vis_df with the number of stops information.
    """
    all_trial_df = pd.DataFrame(
        {'target_index': np.arange(len(ff_caught_T_new))})
    target_last_vis_df = target_last_vis_df.merge(
        all_trial_df, on='target_index', how='right')

    target_last_vis_df.sort_values(by='target_index', inplace=True)
    target_last_vis_df['num_stops'] = num_stops
    target_last_vis_df['num_stops_near_target'] = num_stops_near_target
    target_last_vis_df['num_stops_since_last_vis'] = num_stops_since_last_vis
    target_last_vis_df.dropna(inplace=True)
    target_last_vis_df = target_last_vis_df[target_last_vis_df['last_vis_dist'] != 9999]
    return target_last_vis_df
