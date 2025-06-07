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


def make_target_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_dataframe, max_visibility_window=10):
    target_df = _initialize_target_df(monkey_information, ff_caught_T_new)

    target_df = _add_target_df_info(target_df, monkey_information, ff_real_position_sorted,
                                    ff_dataframe, ff_caught_T_new, max_visibility_window=max_visibility_window
                                    )
    return target_df


def _add_target_df_info(target_df, monkey_information, ff_real_position_sorted, ff_dataframe, ff_caught_T_new,
                        max_visibility_window=10):
    target_df = _calculate_target_distance_and_angle(
        target_df, ff_real_position_sorted)

    target_df = _add_target_disappeared_for_last_time_dummy(
        target_df, ff_dataframe)

    target_df = _add_target_rel_x_and_y(target_df)

    target_df = _find_time_since_last_capture(target_df, ff_caught_T_new)

    # Add target last seen info
    target_df = _add_target_last_seen_info(
        target_df, ff_dataframe, ff_caught_T_new, monkey_information, max_visibility_window=max_visibility_window
    )

    target_df = _add_target_visible_dummy(target_df)

    return target_df


def add_columns_to_target_df(target_df):
    # Add distance from monkey position at target last seen
    target_df['distance_from_monkey_pos_target_last_seen'] = np.sqrt(
        (target_df['monkey_x'] - target_df['monkey_x_target_last_seen'])**2 +
        (target_df['monkey_y'] -
         target_df['monkey_y_target_last_seen'])**2
    )

    # Add cumulative distance since target last seen
    target_df['cum_distance_since_target_last_seen'] = target_df['cum_distance'] - \
        target_df['cum_distance_when_target_last_seen']

    # Add heading difference since target last seen
    target_df['d_heading_since_target_last_seen'] = target_df['monkey_angle'] - \
        target_df['monkey_angle_target_last_seen']

    # make sure d_heading_since_target_last_seen is an acute angle
    target_df['d_heading_since_target_last_seen'] = target_df['d_heading_since_target_last_seen'] % (
        2 * np.pi)
    target_df.loc[target_df['d_heading_since_target_last_seen']
                  > np.pi, 'd_heading_since_target_last_seen'] -= 2 * np.pi
    target_df.loc[target_df['d_heading_since_target_last_seen']
                  < -np.pi, 'd_heading_since_target_last_seen'] += 2 * np.pi

    return target_df


def _add_target_rel_x_and_y(target_df):
    target_df['target_rel_y'] = target_df['target_distance'] * \
        np.cos(target_df['target_angle'])
    target_df['target_rel_x'] = - target_df['target_distance'] * \
        np.sin(target_df['target_angle'])
    return target_df


def fill_na_in_target_df(target_df):
    na_sum = target_df.isna().sum()
    na_df = na_sum[na_sum > 0]

    if len(na_df) > 0:
        # na_rows = target_df.loc[target_df.isna().any(axis=1), na_vars]
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

        # find columns that are not 'last_seen'
        na_cols = na_df.index.values
        not_last_seen_cols = [col for col in na_cols if 'last_seen' not in col]
        # if there's any such column, raise a warning
        if len(not_last_seen_cols) > 0:
            print('Warning: there are columns that are not "last_seen" but contain NA')
            print('not_last_seen_cols', not_last_seen_cols)

        # Sort the DataFrame
        target_df.sort_values(by=['target_index', 'point_index'], inplace=True)

        # Forward fill NA values within each target_index group
        target_df[na_cols] = target_df.groupby(
            'target_index')[na_cols].transform(lambda g: g.ffill())

        # Backward fill any remaining NA values within each target_index group
        target_df[na_cols] = target_df.groupby(
            'target_index')[na_cols].transform(lambda g: g.bfill())

        # Check and print results after filling
        na_sum = target_df.isna().sum()
        na_df = na_sum[na_sum > 0]
        print(f"\nResults after fill NA:")
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


def make_target_cluster_df(monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_dataframe, ff_life_sorted,
                           max_visibility_window=10):
    target_clust_df = _initialize_target_df(
        monkey_information, ff_caught_T_new)

    target_clust_df, nearby_alive_ff_indices = _add_target_cluster_last_seen_info(
        target_clust_df, monkey_information, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe,
        max_visibility_window=max_visibility_window
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

    # Check for warnings
    left_ff_mask = target_df['target_angle'] > 90
    if left_ff_mask.any():
        max_angle = target_df.loc[left_ff_mask, 'target_angle'].max()
        num_arcs = left_ff_mask.sum()
        total_arcs = len(target_df)
        percentage = (num_arcs / total_arcs) * 100

        print("\n" + "="*80)
        print("⚠️  Angle Analysis Warning")
        print("="*80)
        print(
            f"Found {num_arcs:,} arcs ({percentage:.1f}%) where firefly is to the left of the monkey")
        print(f"Maximum angle: {max_angle:.2f}°")
        print("Action: Adjusting angles to be less than 90°")
        print("="*80 + "\n")

        # Adjust angles
        target_df.loc[left_ff_mask, 'target_angle'] = 89.9

    return target_df


def _add_target_cluster_last_seen_info(target_df, monkey_information, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, ff_dataframe,
                                       max_visibility_window=10):
    if 'target_cluster_last_seen_time' not in target_df.columns:
        nearby_alive_ff_indices = cluster_analysis.find_alive_target_clusters(
            ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, max_distance=50)
        print("\n" + "="*80)
        print("🔄 Calculating target-cluster-last-seen info...")
        print("="*80)
        target_df = _add_target_last_seen_info(
            target_df, ff_dataframe, ff_caught_T_new, monkey_information, nearby_alive_ff_indices=nearby_alive_ff_indices, use_target_cluster=True, max_visibility_window=max_visibility_window)
        target_df = target_df.rename(columns={'time_since_target_last_seen': 'target_cluster_last_seen_time',
                                              'target_last_seen_distance': 'target_cluster_last_seen_distance_frozen',
                                              'target_last_seen_angle': 'target_cluster_last_seen_angle_frozen',
                                              'target_last_seen_angle_to_boundary': 'target_cluster_last_seen_angle_to_boundary_frozen',
                                              'monkey_x_target_last_seen': 'monkey_x_target_cluster_last_seen_frozen',
                                              'monkey_y_target_last_seen': 'monkey_y_target_cluster_last_seen_frozen',
                                              'monkey_angle_target_last_seen': 'monkey_angle_target_cluster_last_seen_frozen',
                                              'cum_distance_when_target_last_seen': 'cum_distance_target_cluster_last_seen_frozen',
                                              })

        # Print warning about targets not in visible clusters
        total_targets = len(target_df['target_index'].unique())
        targets_not_in_cluster = total_targets - len(nearby_alive_ff_indices)
        percentage = (targets_not_in_cluster / total_targets) * 100

        print("\n" + "="*80)
        print("📊 Target Cluster Visibility Analysis")
        print("="*80)
        print(f"Total targets: {total_targets:,}")
        print(
            f"Targets not in visible clusters: {targets_not_in_cluster:,} ({percentage:.1f}%)")
        print("="*80 + "\n")

    return target_df, nearby_alive_ff_indices


def _add_target_disappeared_for_last_time_dummy(target_df, ff_dataframe):
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
    print("📊 Target Cluster Visibility Analysis")
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
                                     'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary',
                                     'target_cluster_last_seen_distance_frozen', 'target_cluster_last_seen_angle_frozen', 'target_cluster_last_seen_angle_to_boundary_frozen',]].copy()

    target_average_info = target_average_info.groupby(
        'bin').mean().reset_index(drop=False)
    target_average_info.rename(columns={'target_distance': 'avg_target_distance',
                                        'target_angle': 'avg_target_angle',
                                        'target_angle_to_boundary': 'avg_target_angle_to_boundary',
                                        'time_since_target_last_seen': 'avg_target_last_seen_time',
                                        'target_last_seen_distance': 'avg_target_last_seen_distance_frozen',
                                        'target_last_seen_angle': 'avg_target_last_seen_angle_frozen',
                                        'target_last_seen_angle_to_boundary': 'avg_target_last_seen_angle_to_boundary_frozen',
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


def _add_target_last_seen_info(target_df, ff_dataframe, ff_caught_T_new, monkey_information, nearby_alive_ff_indices=None, use_target_cluster=False,
                               max_visibility_window=10):
    """
    Add target last seen information to the target DataFrame using vectorized operations.

    Parameters:
    - target_df: DataFrame containing target data.
    - ff_dataframe: DataFrame containing firefly data.
    - nearby_alive_ff_indices: Indices of nearby alive fireflies (optional).
    - use_target_cluster: Boolean indicating whether to use target cluster information.

    Returns:
    - Updated target_df with last seen information.
    """

    target_df = _initialize_last_seen_columns(target_df)

    if use_target_cluster and nearby_alive_ff_indices is None:
        raise ValueError(
            "nearby_alive_ff_indices is None, but use_target_cluster is True")

    target_long_df = get_target_long_df(
        ff_dataframe, ff_caught_T_new, monkey_information, max_visibility_window=max_visibility_window,
        nearby_alive_ff_indices=nearby_alive_ff_indices, use_target_cluster=use_target_cluster)

    # Get the key columns in target_df
    target_df2 = target_df[['point_index', 'target_index', 'time', 'monkey_x',
                           'monkey_y', 'monkey_angle', 'cum_distance']].drop_duplicates()

    # Merge with target_long_df
    target_df2 = target_df2.merge(
        target_long_df, on=['point_index', 'target_index'], how='left')

    target_df2 = _calculate_last_seen_info(target_df2)

    # forward fill last seen columns
    last_seen_columns = [
        col for col in target_df2.columns if 'last_seen' in col]
    target_df2[last_seen_columns] = target_df2.groupby(
        'target_index')[last_seen_columns].transform(lambda g: g.ffill())

    # Calculate time since last seen
    target_df2['time_since_target_last_seen'] = target_df2['time'] - \
        target_df2['time_target_last_seen']

    # if time_since_target_last_seen > max_visibility_window, then make all the last_seen columns NA
    last_seen_columns.append('time_since_target_last_seen')
    target_df2.loc[target_df2['time_since_target_last_seen'] >
                   max_visibility_window, last_seen_columns] = np.nan

    # Select essential columns for update
    essential_columns = ['point_index', 'target_index', 'time_since_target_last_seen', 'time_target_last_seen',
                         'target_last_seen_distance', 'target_last_seen_angle',
                         'target_last_seen_angle_to_boundary',
                         'monkey_x_target_last_seen', 'monkey_y_target_last_seen',
                         'monkey_angle_target_last_seen', 'cum_distance_when_target_last_seen']

    # Update the original target_df
    target_df2 = target_df2[essential_columns].copy()
    target_df2.set_index(['point_index', 'target_index'], inplace=True)
    target_df.set_index(['point_index', 'target_index'], inplace=True)
    target_df.update(target_df2)
    target_df.reset_index(inplace=True)

    # Convert point_index to integer
    target_df[['point_index', 'target_index']] = target_df[[
        'point_index', 'target_index']].astype(int)

    return target_df


def get_target_long_df(ff_dataframe, ff_caught_T_new, monkey_information, max_visibility_window=10,
                       nearby_alive_ff_indices=None, use_target_cluster=False):
    """
    Create a DataFrame containing target visibility information for each point index.

    This function processes firefly data to create a mapping between target indices and
    their visibility at different point indices. It handles both individual targets and
    target clusters (groups of nearby fireflies).

    If the target (cluster) is not visible at a particular point_index, then we use the info
    at the last visible point through using ffill()

    Parameters:
    -----------
    nearby_alive_ff_indices : list of lists, optional
        List where each element is a list of firefly indices that form a cluster.
        Required if use_target_cluster is True..

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing target visibility information with columns:
        - point_index: Index of the time point
        - target_index: Index of the target (or cluster)
        - monkey_x_target_last_seen: Monkey's x position when target was last seen
        - monkey_y_target_last_seen: Monkey's y position when target was last seen
        - monkey_angle_target_last_seen: Monkey's angle when target was last seen
        - cum_distance_when_target_last_seen: Cumulative distance when target was last seen
        - time_target_last_seen: Time when target was last seen

    Notes:
    ------
    - For target clusters (use_target_cluster=True), each target_index can have multiple ff index
    - For individual targets (use_target_cluster=False), target_index equals ff_index.

    """

    # Create target_long_df based on whether we're using target clusters
    if use_target_cluster:
        # Create target_long_df from nearby_alive_ff_indices
        target_long_df = pd.DataFrame({
            'target_index': np.repeat(range(len(nearby_alive_ff_indices)),
                                      [len(indices) for indices in nearby_alive_ff_indices]),
            'ff_index': np.concatenate(nearby_alive_ff_indices)
        })
    else:
        # If not using target clusters, ff_index equals target_index
        unique_target_indices = ff_dataframe['target_index'].unique()
        target_long_df = pd.DataFrame({
            'target_index': unique_target_indices,
            'ff_index': unique_target_indices
        })

    # Get visible firefly information
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]

    # Merge to get point_index and ff_index combinations that are visible
    target_long_df = target_long_df.merge(
        ff_dataframe_visible[['point_index', 'ff_index', 'ff_x', 'ff_y', 'time', 'monkey_x', 'monkey_y', 'cum_distance',
                              'monkey_angle']], on='ff_index', how='inner')

    # We just only to preserve one row for each target_index & point_index,
    # and the existence of the row indicates that target (cluster) is visible at that point_index
    target_long_df = target_long_df.drop(columns=['ff_index']).drop_duplicates(
        subset=['point_index', 'target_index'])

    # rename columns; note: we use ff_x and ff_y instead of target_x and target_y; when we don't use target cluster, target_x and target_y are the same as ff_x and ff_y
    target_long_df.rename(columns={'ff_x': 'target_x',
                                   'ff_y': 'target_y',
                                   'monkey_x': 'monkey_x_target_last_seen',
                                   'monkey_y': 'monkey_y_target_last_seen',
                                   'monkey_angle': 'monkey_angle_target_last_seen',
                                   'cum_distance': 'cum_distance_when_target_last_seen',
                                   'time': 'time_target_last_seen'}, inplace=True)

    target_long_df = _furnish_target_long_df(
        target_long_df, ff_caught_T_new, monkey_information, max_visibility_window=max_visibility_window)

    return target_long_df


def _furnish_target_long_df(target_long_df, ff_caught_T_new, monkey_information, max_visibility_window=10):
    """
    Furnish target_long_df with additional time points before capture for each target.

    This function adds time points between (capture_time - max_visibility_window) 
    and capture_time for each target, then forward fills the values for all columns.
    Note that if the trial length is greater than max_visibility_window, then visible 
    information from the whole trial is used

    Purpose:
    --------
    This function ensures that every point index has last-seen information by:
    1. Adding all time points between (capture_time - max_visibility_window) and capture_time
    2. Forward-filling the last-seen information for each target/cluster

    This creates a continuous record of last-seen information for each target up to its capture time,
    making it easier to analyze target visibility patterns.

    Parameters:
    -----------
    target_long_df : pandas.DataFrame
        DataFrame containing target visibility information
    ff_caught_T_new : numpy.ndarray
        Array of capture times for each target
    monkey_information : pandas.DataFrame
        DataFrame containing monkey movement information with 'time' column
    max_visibility_window : float, default=7
        Number of time units to look back from capture time

    Returns:
    --------
    pandas.DataFrame
        Updated target_long_df with additional time points and forward-filled values

    Raises:
    -------
    ValueError
        If required columns are missing, if max_visibility_window is negative,
        or if ff_caught_T_new is empty
    """
    # Input validation (streamlined)
    if max_visibility_window < 0:
        raise ValueError(
            "max_visibility_window must be non-negative")

    if len(ff_caught_T_new) == 0:
        raise ValueError("ff_caught_T_new cannot be empty")

    required_cols = {'target_index', 'point_index'}
    if not required_cols.issubset(target_long_df.columns):
        raise ValueError(
            f"target_long_df must contain columns: {list(required_cols)}")

    if 'time' not in monkey_information.columns:
        raise ValueError("monkey_information must contain 'time' column")

    # Vectorized operations for time windows
    ff_caught_T_new = np.asarray(ff_caught_T_new)
    start_times = ff_caught_T_new - max_visibility_window
    end_times = ff_caught_T_new

    # Pre-sort monkey_information by time for faster searching
    monkey_times = monkey_information['time'].values
    monkey_indices = monkey_information.index.values

    # Use searchsorted for efficient range finding
    target_indices_list = []
    point_indices_list = []

    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        # Find indices within time range using binary search
        left_idx = np.searchsorted(monkey_times, start_time, side='left')
        right_idx = np.searchsorted(monkey_times, end_time, side='right')

        # Get point indices in range
        points_in_range = monkey_indices[left_idx:right_idx]
        n_points = len(points_in_range)

        if n_points > 0:
            target_indices_list.append(np.full(n_points, i))
            point_indices_list.append(points_in_range)

    # Create new points DataFrame efficiently
    if target_indices_list:
        all_target_indices = np.concatenate(target_indices_list)
        all_point_indices = np.concatenate(point_indices_list)

        new_points_df = pd.DataFrame({
            'target_index': all_target_indices,
            'point_index': all_point_indices
        })

        # Use merge instead of concat for better memory efficiency
        combined_df = target_long_df.merge(
            new_points_df,
            on=['target_index', 'point_index'],
            how='outer'
        )
    else:
        combined_df = target_long_df.copy()

    # Sort and forward fill
    combined_df.sort_values(['target_index', 'point_index'], inplace=True)

    # Get fill columns once
    fill_cols = combined_df.columns.difference(
        ['target_index', 'point_index']).tolist()

    if fill_cols:
        # Forward fill within each target group
        combined_df[fill_cols] = combined_df.groupby('target_index')[
            fill_cols].ffill()

    return combined_df.reset_index(drop=True)


def _calculate_last_seen_info(target_df2):

    # Calculate last-seen distances and angles
    target_x, target_y = target_df2['target_x'], target_df2['target_y']
    monkey_x, monkey_y = target_df2['monkey_x_target_last_seen'], target_df2['monkey_y_target_last_seen']
    monkey_angle = target_df2['monkey_angle_target_last_seen']

    target_distance = np.sqrt(
        (target_x - monkey_x)**2 + (target_y - monkey_y)**2)

    target_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=target_x, ff_y=target_y,
        mx=monkey_x, my=monkey_y,
        m_angle=monkey_angle
    )

    # Add frozen metrics
    target_df2['target_last_seen_distance'] = target_distance
    target_df2['target_last_seen_angle'] = target_angle
    target_df2['target_last_seen_angle_to_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=target_angle,
        distances_to_ff=target_distance
    )

    return target_df2


def _initialize_last_seen_columns(target_df):
    """
    Initialize columns with default values and set their dtype to float.
    """

    columns = ['time_since_target_last_seen', 'target_last_seen_distance', 'time_target_last_seen',
               'target_last_seen_angle', 'target_last_seen_angle_to_boundary',
               'monkey_x_target_last_seen', 'monkey_y_target_last_seen',
               'monkey_angle_target_last_seen', 'cum_distance_when_target_last_seen']
    target_df[columns] = np.nan
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


def _deal_with_delta_angles_greater_than_90_degrees(df, reward_boundary_radius=25, ignore_error=False):
    """
    Adjust arc angles that are greater than 90 degrees to be within valid bounds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing arc information with columns:
        - delta_angle: The angle between arc start and end
        - ff_distance: Distance to the firefly
        - arc_end_direction: Direction of the arc end
        - arc_starting_angle: Starting angle of the arc
        - arc_ending_angle: Ending angle of the arc
    reward_boundary_radius : float, default=25
        Maximum distance within which adjustments are made
    ignore_error : bool, default=False
        If True, prints warning instead of raising error for large angles

    Returns:
    --------
    pandas.DataFrame
        DataFrame with adjusted arc angles
    """
    # Constants
    MAX_ANGLE = math.pi/2
    ANGLE_ADJUSTMENT = 0.00001
    CRITICAL_ANGLE = 150  # degrees

    # Reset index for clean operations
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    # Create masks for different conditions
    too_big_angle = df['delta_angle'] > MAX_ANGLE
    within_reward_boundary = df['ff_distance'] <= reward_boundary_radius
    ff_at_left = df['arc_end_direction'] >= 0
    ff_at_right = df['arc_end_direction'] < 0

    if not too_big_angle.any():
        return df

    # First attempt: Adjust angles within reward boundary
    left_mask = too_big_angle & ff_at_left & within_reward_boundary
    right_mask = too_big_angle & ff_at_right & within_reward_boundary

    df.loc[left_mask, 'arc_ending_angle'] = df.loc[left_mask,
                                                   'arc_starting_angle'] + MAX_ANGLE - ANGLE_ADJUSTMENT
    df.loc[right_mask, 'arc_ending_angle'] = df.loc[right_mask,
                                                    'arc_starting_angle'] - (MAX_ANGLE - ANGLE_ADJUSTMENT)

    # Recalculate delta angles and check if any are still too big
    df['delta_angle'] = np.abs(
        df['arc_ending_angle'] - df['arc_starting_angle'])
    too_big_angle = df['delta_angle'] > MAX_ANGLE

    if too_big_angle.any():
        # Calculate statistics for both left and right
        left_big_angles = too_big_angle & ff_at_left
        right_big_angles = too_big_angle & ff_at_right

        max_left_angle = df.loc[left_big_angles, 'delta_angle'].max(
        ) * 180/math.pi if left_big_angles.any() else 0
        max_right_angle = df.loc[right_big_angles, 'delta_angle'].max(
        ) * 180/math.pi if right_big_angles.any() else 0
        num_affected_arcs = too_big_angle.sum()
        total_arcs = len(df)

        # Print warning message
        print("\n" + "="*80)
        print("⚠️  Arc Angle Analysis Warning")
        print("="*80)
        print(
            f"Found {num_affected_arcs:,} arcs ({num_affected_arcs/total_arcs*100:.1f}%) with angles > 90°")

        if left_big_angles.any():
            print(
                f"FF at left side: {left_big_angles.sum():,} arcs, max angle: {max_left_angle:.2f}°")
        if right_big_angles.any():
            print(
                f"FF at right side: {right_big_angles.sum():,} arcs, max angle: {max_right_angle:.2f}°")

        # Handle critical angles for both sides
        if max_left_angle > CRITICAL_ANGLE or max_right_angle > CRITICAL_ANGLE:
            error_msg = "Critical angles detected:\n"
            if max_left_angle > CRITICAL_ANGLE:
                error_msg += f"- FF at left side: {max_left_angle:.2f}°\n"
            if max_right_angle > CRITICAL_ANGLE:
                error_msg += f"- FF at right side: {max_right_angle:.2f}°"

            if ignore_error:
                print(f"Warning: {error_msg}")
                print("Action: Adjusting angles to be less than 90°")
            else:
                raise ValueError(error_msg)

        # Second attempt: Adjust all remaining large angles
        left_mask = too_big_angle & ff_at_left
        right_mask = too_big_angle & ff_at_right

        df.loc[left_mask, 'arc_ending_angle'] = df.loc[left_mask,
                                                       'arc_starting_angle'] + MAX_ANGLE - ANGLE_ADJUSTMENT
        df.loc[right_mask, 'arc_ending_angle'] = df.loc[right_mask,
                                                        'arc_starting_angle'] - (MAX_ANGLE - ANGLE_ADJUSTMENT)

        print("Action: Adjusted all angles to be less than 90°")
        print("="*80 + "\n")

    return df
