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

    target_df = _add_target_df_info(target_df, ff_real_position_sorted,
                                    ff_dataframe, ff_caught_T_new, include_frozen_info=include_frozen_info,
                                    )
    return target_df


def _add_target_df_info(target_df, ff_real_position_sorted, ff_dataframe, ff_caught_T_new, include_frozen_info=True,
                        ):
    target_df = _calculate_target_distance_and_angle(
        target_df, ff_real_position_sorted)

    # Add target last seen info
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


def fill_na_in_target_df(target_df, na_fill_method_for_target_vars='ffill'):
    if na_fill_method_for_target_vars is not None:
        na_sum = target_df.isna().sum()
        na_df = na_sum[na_sum > 0]
        if len(na_df) > 0:
            na_vars = na_df.index
            num_rows = len(target_df)

            # Print header with separator
            print("\n" + "="*80)
            print(f"NA Values Analysis for target_df ({num_rows:,} rows)")
            print("="*80)

            # Print NA summary in a table format
            print("\nColumns with NA values:")
            print("-"*60)
            for col, count in na_df.items():
                percentage = (count / num_rows) * 100
                print(f"{col:<40} {count:>8,} ({percentage:>6.1f}%)")
            print("-"*60)

            # Print fill method
            print(
                f"\nFilling NA values using method: {na_fill_method_for_target_vars}")

            # Apply fill method
            if na_fill_method_for_target_vars == 'ffill':
                target_df[na_vars] = target_df[na_vars].ffill()
            elif na_fill_method_for_target_vars == 'bfill':
                target_df[na_vars] = target_df[na_vars].bfill()
            else:
                raise ValueError(
                    f"Invalid method to address NA: {na_fill_method_for_target_vars}")

            # Check and print results after filling
            na_sum = target_df.isna().sum()
            na_df = na_sum[na_sum > 0]
            print("\nResults after filling:")
            print("-"*60)
            if len(na_df) > 0:
                print("Remaining NA values:")
                for col, count in na_df.items():
                    percentage = (count / num_rows) * 100
                    print(f"{col:<40} {count:>8,} ({percentage:>6.1f}%)")
            else:
                print("✓ All NA values have been successfully filled")
            print("="*80 + "\n")

    return target_df


def make_target_cluster_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_dataframe, ff_life_sorted, include_frozen_info=True,
                           ):
    target_clust_df = _initialize_target_df(
        monkey_information, ff_caught_T_new)

    target_clust_df, nearby_alive_ff_indices = _add_target_cluster_last_seen_info(
        target_clust_df, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe, include_frozen_info=include_frozen_info,
    )

    target_clust_df = _add_target_cluster_disappeared_for_last_time_dummy(
        target_clust_df, ff_caught_T_new, ff_dataframe, nearby_alive_ff_indices)

    target_clust_df = _add_target_cluster_visible_dummy(target_clust_df)

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


def _add_target_cluster_last_seen_info(target_df, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe, include_frozen_info=True,
                                       ):
    if 'target_cluster_last_seen_time' not in target_df.columns:
        nearby_alive_ff_indices = cluster_analysis.find_alive_target_clusters(
            ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, max_distance=50)
        print("\n" + "="*80)
        print("Calculating target-cluster-last-seen info...")
        print("="*80)
        target_df = _add_target_last_seen_info(
            target_df, ff_dataframe, nearby_alive_ff_indices, use_target_cluster=True, include_frozen_info=include_frozen_info)
        target_df = target_df.rename(columns={'time_since_target_last_seen': 'target_cluster_last_seen_time',
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
    # Get the last visibility time for each target
    target_last_vis_times = ff_dataframe[ff_dataframe['visible'] == 1].groupby('ff_index')[
        'time'].max()

    # Create a mapping of target_index to last visibility time
    target_df['target_has_disappeared_for_last_time_dummy'] = (
        target_df['time'] > target_df['target_index'].map(
            target_last_vis_times)
    ).astype(int)

    # Print warning about segments between last-seen time and capture time
    total_rows = len(target_df)
    preserved_rows = len(
        target_df[target_df['target_has_disappeared_for_last_time_dummy'] == 0])
    percentage = (preserved_rows / total_rows) * 100

    print("\n" + "="*80)
    print("Target Visibility Analysis")
    print("="*80)
    print(f"Total rows: {total_rows:,}")
    print(f"Preserved rows: {preserved_rows:,} ({percentage:.1f}%)")
    print("="*80 + "\n")

    return target_df


def _add_target_cluster_disappeared_for_last_time_dummy(target_df, ff_caught_T_new, ff_dataframe, nearby_alive_ff_indices):
    """
    Add target_cluster_has_disappeared_for_last_time_dummy to target_df using vectorized operations
    """
    # Get visible fireflies and their last visibility times
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    ff_last_vis_times = ff_dataframe_visible.groupby('ff_index')['time'].max()

    # Create a mapping of target_index to cluster indices
    cluster_indices = pd.DataFrame({
        'target_index': np.repeat(range(len(ff_caught_T_new)),
                                  [len(indices) for indices in nearby_alive_ff_indices]),
        'ff_index': np.concatenate(nearby_alive_ff_indices)
    })

    # Get last visibility time for each cluster
    cluster_last_vis_times = (cluster_indices
                              .merge(ff_last_vis_times.reset_index(), on='ff_index', how='left')
                              .groupby('target_index')['time']
                              .max()
                              .reset_index()
                              .rename(columns={'time': 'last_vis_time'}))

    # Create the dummy variable using vectorized operations
    target_df['target_cluster_has_disappeared_for_last_time_dummy'] = (
        target_df.merge(cluster_last_vis_times, on='target_index', how='left')['time'] >
        target_df.merge(cluster_last_vis_times, on='target_index', how='left')[
            'last_vis_time']
    ).astype(int)

    # Print warning about targets not in visible clusters
    total_targets = len(target_df['target_index'].unique())
    targets_not_in_cluster = total_targets - \
        len(cluster_indices['target_index'].unique())
    percentage = (targets_not_in_cluster / total_targets) * 100

    print("\n" + "="*80)
    print("Target Cluster Visibility Analysis")
    print("="*80)
    print(f"Total targets: {total_targets:,}")
    print(
        f"Targets not in visible clusters: {targets_not_in_cluster:,} ({percentage:.1f}%)")
    print("="*80 + "\n")

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


def _calculate_average_info(target_df):
    """
    Calculate average information for each bin in target_df
    """
    target_average_info = target_df[['bin', 'target_distance', 'target_angle', 'target_angle_to_boundary',
                                    'time_since_target_last_seen', 'target_cluster_last_seen_time',
                                     'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
                                     'target_cluster_last_seen_distance_frozen', 'target_cluster_last_seen_angle_frozen', 'target_cluster_last_seen_angle_to_boundary_frozen',]].copy()

    target_average_info = target_average_info.groupby(
        'bin').mean().reset_index(drop=False)
    target_average_info.rename(columns={'target_distance': 'avg_target_distance',
                                        'target_angle': 'avg_target_angle',
                                        'target_angle_to_boundary': 'avg_target_angle_to_boundary',
                                        'time_since_target_last_seen': 'avg_target_last_seen_time',
                                        'target_last_seen_distance_frozen': 'avg_target_last_seen_distance_frozen',
                                        'target_last_seen_angle_frozen': 'avg_target_last_seen_angle_frozen',
                                        'target_last_seen_angle_to_boundary_frozen': 'avg_target_last_seen_angle_to_boundary_frozen',
                                        'target_cluster_last_seen_time': 'avg_target_cluster_last_seen_time',
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


def _add_target_last_seen_info(target_df, ff_dataframe, nearby_alive_ff_indices=None, use_target_cluster=False, include_frozen_info=False,
                               ):
    """
    Add target last seen information to the target DataFrame using vectorized operations.

    Parameters:
    - target_df: DataFrame containing target data.
    - ff_dataframe: DataFrame containing firefly data.
    - nearby_alive_ff_indices: Indices of nearby alive fireflies (optional).
    - use_target_cluster: Boolean indicating whether to use target cluster information.
    - include_frozen_info: Boolean indicating whether to include frozen information.

    Returns:
    - Updated target_df with last seen information.
    """

    # ================================================
    # Currently we decided not to use placeholder but instead use np.nan to initialize the last seen columns
    # Define constants
    # DEFAULT_LAST_SEEN_TIME = 100
    # DEFAULT_LAST_SEEN_DISTANCE = 400
    # DEFAULT_LAST_SEEN_ANGLE = 0

    # Initialize columns with default values
    # _add_placeholder_last_seen_values(target_df, DEFAULT_LAST_SEEN_TIME,
    #                                   DEFAULT_LAST_SEEN_DISTANCE, DEFAULT_LAST_SEEN_ANGLE, include_frozen_info)
    # ================================================

    target_df = _initialize_last_seen_columns(target_df, include_frozen_info)

    if use_target_cluster and nearby_alive_ff_indices is None:
        raise ValueError(
            "nearby_alive_ff_indices is None, but use_target_cluster is True")

    # Get unique target indices
    sorted_target_index = np.sort(ff_dataframe['target_index'].unique())

    # Create a mask for visible fireflies
    if use_target_cluster:
        # Create a mapping of target_index to cluster indices
        cluster_indices = pd.DataFrame({
            'target_index': np.repeat(range(len(sorted_target_index)),
                                      [len(indices) for indices in nearby_alive_ff_indices]),
            'ff_index': np.concatenate(nearby_alive_ff_indices)
        })
        visible_mask = (ff_dataframe['ff_index'].isin(
            cluster_indices['ff_index'])) & (ff_dataframe['visible'] == 1)
    else:
        visible_mask = (ff_dataframe['visible'] == 1)

    # Get visible firefly information and sort
    visible_info = ff_dataframe[visible_mask].sort_values(
        ['target_index', 'point_index'])

    # Create a mapping of target_index to time for efficient lookup
    target_times = target_df.groupby('target_index')['time'].first()

    # Process all visible information at once
    last_visible = visible_info.groupby(
        ['target_index', 'point_index']).last().reset_index()

    # Calculate time differences vectorized
    last_visible['time_since_target_last_seen'] = last_visible.apply(
        lambda x: x['time'] - target_times[x['target_index']], axis=1
    )

    # Create the base group_info DataFrame
    group_info = last_visible[[
        'point_index', 'target_index', 'time_since_target_last_seen']].copy()

    if include_frozen_info:
        group_info = _add_frozen_info(last_visible, group_info)

    # update the original target_df with the group_info
    group_info.set_index(['point_index', 'target_index'], inplace=True)
    target_df.set_index(['point_index', 'target_index'], inplace=True)
    target_df.update(group_info)
    target_df.reset_index(inplace=True)

    # Convert point_index to integer
    target_df['point_index'] = target_df['point_index'].astype(int)

    return target_df


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


def _add_frozen_info(last_visible, group_info):
    # Calculate frozen distances and angles vectorized
    target_x, target_y = last_visible['ff_x'], last_visible['ff_y']
    monkey_x, monkey_y = last_visible['monkey_x'], last_visible['monkey_y']
    monkey_angle = last_visible['monkey_angle']

    target_distance = np.sqrt(
        (target_x - monkey_x)**2 + (target_y - monkey_y)**2)
    target_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=target_x, ff_y=target_y,
        mx=monkey_x, my=monkey_y,
        m_angle=monkey_angle
    )

    # Add frozen metrics
    group_info['target_last_seen_distance_frozen'] = target_distance
    group_info['target_last_seen_angle_frozen'] = target_angle
    group_info['target_last_seen_angle_to_boundary_frozen'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=target_angle,
        distances_to_ff=target_distance
    )

    # Add frozen position information
    frozen_cols = ['monkey_x', 'monkey_y', 'monkey_angle', 'cum_distance']
    group_info = pd.concat([
        group_info,
        last_visible[frozen_cols].rename(columns={
            'monkey_x': 'monkey_x_target_last_seen_frozen',
            'monkey_y': 'monkey_y_target_last_seen_frozen',
            'monkey_angle': 'monkey_angle_target_last_seen_frozen',
            'cum_distance': 'cum_distance_target_last_seen_frozen'
        })
    ], axis=1)
    return group_info


def _initialize_last_seen_columns(target_df, include_frozen_info):
    """
    Initialize columns with default values and set their dtype to float.
    """

    target_df['time_since_target_last_seen'] = np.nan

    if include_frozen_info:
        frozen_columns = ['target_last_seen_distance_frozen',
                          'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
                          'monkey_x_target_last_seen_frozen', 'monkey_y_target_last_seen_frozen',
                          'monkey_angle_target_last_seen_frozen', 'cum_distance_target_last_seen_frozen']
        target_df[frozen_columns] = np.nan
    return target_df


# def _add_placeholder_last_seen_values(target_df, last_seen_time, last_seen_distance, last_seen_angle, include_frozen_info):
#     """
#     Initialize columns with default values and set their dtype to float.
#     """
#     columns = ['time_since_target_last_seen']
#     target_df[columns] = [last_seen_time]
#     target_df[columns] = target_df[columns].astype(float)

#     if include_frozen_info:
#         frozen_columns = ['target_last_seen_distance_frozen',
#                           'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen']
#         target_df[frozen_columns] = [
#             last_seen_distance, last_seen_angle, last_seen_angle]
#         target_df[frozen_columns] = target_df[frozen_columns].astype(float)

#         target_df[['monkey_x_target_last_seen_frozen', 'monkey_y_target_last_seen_frozen',
#                    'monkey_angle_target_last_seen_frozen', 'cum_distance_target_last_seen_frozen']] = np.nan


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
