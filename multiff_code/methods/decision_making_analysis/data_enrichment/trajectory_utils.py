# Imports
import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

from data_wrangling import specific_utils
from null_behaviors import curv_of_traj_utils, opt_arc_utils
from decision_making_analysis.ff_data_acquisition import ff_data_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data

# Matplotlib and pandas setup
plt.rcParams['animation.html'] = 'html5'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
np.set_printoptions(suppress=True)


# ============================================================
#  Curvature-related utilities
# ============================================================

def find_trajectory_arc_info(point_index_array, curv_of_traj_df, ff_caught_T_new=None, monkey_information=None,
                             window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance',
                             truncate_curv_of_traj_by_time_of_capture=False):
    """
    Retrieve or compute trajectory curvature values for a set of points.
    """
    curv_of_traj_df_temp = curv_of_traj_df.groupby(
        'point_index').first().reset_index().set_index('point_index')

    try:
        curv_of_traj = curv_of_traj_df_temp.loc[point_index_array,
                                                'curv_of_traj'].values
    except KeyError:
        if ff_caught_T_new is None:
            raise ValueError(
                'ff_caught_T_new must be provided when curvature data are missing.')

        missing_point_index = np.setdiff1d(
            point_index_array, curv_of_traj_df_temp.index.values)
        print('Missing point indices:', missing_point_index)
        print('Recomputing curvature due to insufficient existing data...')

        curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
            window_for_curv_of_traj, monkey_information, ff_caught_T_new,
            curv_of_traj_mode=curv_of_traj_mode,
            truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture
        )
        curv_of_traj = curv_of_traj_df.loc[point_index_array,
                                           'curv_of_traj'].values
    except Exception as e:
        print('Unexpected error while retrieving curvature:', e)
        raise

    return opt_arc_utils.winsorize_curv(curv_of_traj)


# ============================================================
#  Trajectory data utilities
# ============================================================

def find_trajectory_data(time, monkey_information, time_range_of_trajectory=[-0.5, 0.5],
                         num_time_points_for_trajectory=10):
    """
    Extract monkey trajectory data around specified time points.
    """
    traj_time_2d = np.tile(np.array(time).reshape(-1, 1),
                           (1, num_time_points_for_trajectory))
    traj_time_2d += np.linspace(time_range_of_trajectory[0],
                                time_range_of_trajectory[1], num_time_points_for_trajectory)

    monkey_indices = np.searchsorted(
        monkey_information['time'], traj_time_2d).ravel()
    monkey_indices = np.clip(monkey_indices, 0, len(monkey_information) - 1)

    trajectory_data_dict = {
        col: monkey_information[col].values[monkey_indices].reshape(
            -1, num_time_points_for_trajectory)
        for col in monkey_information.columns
    }

    return traj_time_2d, trajectory_data_dict


def find_monkey_info_on_trajectory_relative_to_origin(monkey_indices, monkey_information,
                                                      traj_x_2d, traj_y_2d, monkey_angle_2d,
                                                      num_time_points_for_trajectory=10):
    """
    Compute distances and angles of monkey trajectory relative to its origin.
    """
    monkey_indices = np.clip(monkey_indices, 0, len(monkey_information) - 1)

    monkey_xy = monkey_information[[
        'monkey_x', 'monkey_y']].values[monkey_indices]
    monkey_angle = monkey_information['monkey_angle'].values[monkey_indices]

    monkey_angle_on_traj_rel_to_north = monkey_angle_2d - monkey_angle[:, None]
    monkey_xy_tiled = np.tile(
        monkey_xy, (1, num_time_points_for_trajectory)).reshape(-1, 2)
    monkey_angle_tiled = np.tile(
        monkey_angle, (1, num_time_points_for_trajectory)).reshape(-1)

    traj_xy_2d = np.column_stack((traj_x_2d.ravel(), traj_y_2d.ravel()))
    traj_distances = np.linalg.norm(traj_xy_2d - monkey_xy_tiled, axis=1)
    traj_angles = specific_utils.calculate_angles_to_ff_centers(
        ff_x=traj_x_2d.ravel(), ff_y=traj_y_2d.ravel(),
        mx=monkey_xy_tiled[:, 0], my=monkey_xy_tiled[:,
                                                     1], m_angle=monkey_angle_tiled
    )

    return (
        traj_distances.reshape(-1, num_time_points_for_trajectory),
        traj_angles.reshape(-1, num_time_points_for_trajectory),
        monkey_angle_on_traj_rel_to_north
    )


def generate_feature_names_given_relative_time_points(relative_time_points, num_time_points,
                                                      original_feature_names='monkey_distance'):
    """
    Generate feature names based on relative time points.
    """
    if isinstance(original_feature_names, str):
        original_feature_names = [original_feature_names]

    new_feature_names = [
        f'{feature}_{round(relative_time_points[i], 2)}s'
        for feature in original_feature_names
        for i in range(num_time_points)
    ]
    return new_feature_names


# ============================================================
#  Machine-learning data augmentation
# ============================================================

def furnish_machine_learning_data_with_trajectory_data_func(
        X_all, time_all, monkey_information, trajectory_data_kind='position',
        time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10, add_traj_stops=True):
    """
    Augment machine learning features with trajectory and stopping information.
    """
    if trajectory_data_kind == 'position':
        traj_points, trajectory_feature_names = generate_trajectory_position_data(
            time_all, monkey_information, time_range_of_trajectory, num_time_points_for_trajectory
        )
        X_all = np.concatenate([X_all, traj_points], axis=1)

    elif trajectory_data_kind == 'velocity':
        traj_points, trajectory_feature_names = generate_trajectory_velocity_data(
            time_all, monkey_information, time_range_of_trajectory, num_time_points_for_trajectory
        )
        X_all = np.concatenate([X_all, traj_points], axis=1)

    else:
        raise ValueError(
            "trajectory_data_kind must be either 'position' or 'velocity'")

    traj_stops = np.array([])
    if add_traj_stops:
        traj_stops, stop_feature_names = generate_stops_info(
            time_all, monkey_information, time_range_of_trajectory, num_time_points_for_trajectory
        )
        X_all = np.concatenate([X_all, traj_stops], axis=1)
        trajectory_feature_names.extend(stop_feature_names)

    return X_all, traj_points, traj_stops, trajectory_feature_names


# ============================================================
#  Stop information utilities
# ============================================================

def generate_stops_info(time_all, monkey_information,
                        time_range_of_trajectory=[-0.5, 0.5],
                        num_time_points_for_trajectory=10):
    """
    Generate stopping (1 = stop, 0 = no stop) information across trajectory time bins.
    """
    relative_time_points = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1],
                                       num_time_points_for_trajectory)

    traj_time_2d = np.tile(np.array(time_all).reshape(-1, 1),
                           (1, num_time_points_for_trajectory))
    traj_time_2d += relative_time_points

    bin_width = (time_range_of_trajectory[1] - time_range_of_trajectory[0]) / (
        num_time_points_for_trajectory - 1)
    monkey_dt = (monkey_information['time'].iloc[-1] -
                 monkey_information['time'].iloc[0]) / (len(monkey_information) - 1)
    num_points_in_window = math.ceil(bin_width / monkey_dt + 1)
    if num_points_in_window % 2 == 0:
        num_points_in_window += 1

    # Smooth stop indicator (1 = stop, 0 = move)
    convolve_pattern = np.ones(num_points_in_window) / num_points_in_window
    monkey_stops = (monkey_information['monkey_speeddummy'].values - 1) * -1
    stops_convolved = np.convolve(monkey_stops, convolve_pattern, 'same')
    stops_convolved = (stops_convolved > 0).astype(int)

    indices = np.searchsorted(monkey_information['time'].values, traj_time_2d)
    indices = np.clip(indices, 0, len(monkey_information) - 1)
    traj_stops = stops_convolved[indices]

    feature_names = generate_feature_names_given_relative_time_points(
        relative_time_points, num_time_points_for_trajectory, original_feature_names='whether_stopped'
    )

    return traj_stops, feature_names


def add_stops_info_to_one_row_of_trajectory_info(traj_time_1d, monkey_information):
    """
    Generate binary stop info (1 = stop) for a single trajectory row, based on monkey_speeddummy.
    """
    bin_width = (traj_time_1d[-1] - traj_time_1d[0]) / (len(traj_time_1d) - 1)
    time_bins = np.arange(
        traj_time_1d[0] - bin_width / 2, traj_time_1d[-1] + bin_width, bin_width)
    monkey_information.loc[:, 'corresponding_bins'] = np.searchsorted(
        time_bins, monkey_information['time'].values)
    monkey_sub = (monkey_information[monkey_information['corresponding_bins'].between(1, len(time_bins) - 1)]
                  [['corresponding_bins', 'monkey_speeddummy']]
                  .groupby('corresponding_bins')
                  .min())
    stopping_info = -(monkey_sub['monkey_speeddummy'].values - 1)
    return stopping_info


def add_stops_info_to_monkey_information(traj_time, monkey_information):
    """
    Add stop indicators (1 = stop) to the monkey_information DataFrame, aligned with trajectory bins.
    """
    bin_width = traj_time[1] - traj_time[0]
    time_bins = np.arange(
        traj_time[0] - bin_width / 2, traj_time[-1] + bin_width, bin_width)
    monkey_information.loc[:, 'corresponding_bins'] = np.searchsorted(
        time_bins, monkey_information['time'].values)
    monkey_sub = (monkey_information[monkey_information['corresponding_bins'].between(1, len(time_bins) - 1)]
                  [['corresponding_bins', 'monkey_speeddummy']]
                  .groupby('corresponding_bins')
                  .min())
    stopping_info = -(monkey_sub['monkey_speeddummy'].values - 1)
    monkey_information.loc[:, 'monkey_stops_based_on_bins'] = stopping_info


def combine_trajectory_and_stop_info_and_curvature_info(traj_points_df, traj_stops_df, relevant_curv_of_traj_df,
                                                        use_more_as_prefix=False):
    """
    Combine trajectory, stop, and curvature data into a single DataFrame.
    """
    for df_a, df_b, name in [(traj_points_df, traj_stops_df, 'stops'),
                             (traj_points_df, relevant_curv_of_traj_df, 'curvature')]:
        if not np.array_equal(df_a['point_index'].values, df_b['point_index'].values):
            raise ValueError(
                f'Point indices of trajectory and {name} data do not match.')

    traj_points_df = traj_points_df.drop(columns=['point_index'])
    traj_stops_df = traj_stops_df.drop(columns=['point_index'])
    relevant_curv_of_traj_df = relevant_curv_of_traj_df.drop(columns=[
                                                             'point_index'])

    key_prefix = 'more_' if use_more_as_prefix else ''
    feature_names = {
        f'{key_prefix}traj_points': traj_points_df.columns.values,
        f'{key_prefix}traj_stops': traj_stops_df.columns.values,
        f'{key_prefix}relevant_curv_of_traj': relevant_curv_of_traj_df.columns.values
    }

    combined_df = pd.concat(
        [traj_points_df, traj_stops_df, relevant_curv_of_traj_df], axis=1)
    return combined_df, feature_names


def make_traj_data_feature_names(time_range_of_trajectory, num_time_points_for_trajectory,
                                 use_more_as_prefix=False, traj_point_features=['monkey_distance', 'monkey_angle'],
                                 relevant_curv_of_traj_feature_names=['curv_of_traj']):
    """
    Construct dictionary of trajectory-related feature names.
    """
    rel_time_points = np.linspace(
        time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    key_prefix = 'more_' if use_more_as_prefix else ''

    return {
        f'{key_prefix}traj_points': generate_feature_names_given_relative_time_points(
            rel_time_points, num_time_points_for_trajectory, traj_point_features),
        f'{key_prefix}traj_stops': generate_feature_names_given_relative_time_points(
            rel_time_points, num_time_points_for_trajectory, 'whether_stopped'),
        f'{key_prefix}relevant_curv_of_traj': relevant_curv_of_traj_feature_names
    }


def make_all_traj_feature_names(time_range_of_trajectory, num_time_points_for_trajectory,
                                time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot,
                                traj_point_features=['monkey_distance', 'monkey_angle']):
    """
    Combine standard and plotting-time trajectory feature names.
    """
    base_features = make_traj_data_feature_names(time_range_of_trajectory, num_time_points_for_trajectory,
                                                 traj_point_features=traj_point_features)
    if time_range_of_trajectory_to_plot is not None and num_time_points_for_trajectory_to_plot is not None:
        extra_features = make_traj_data_feature_names(time_range_of_trajectory_to_plot,
                                                      num_time_points_for_trajectory_to_plot,
                                                      use_more_as_prefix=True,
                                                      traj_point_features=traj_point_features)
        base_features |= extra_features
    return base_features


def retrieve_or_make_all_traj_feature_names(raw_data_dir_name, monkey_name, exists_ok=True, save=True,
                                            time_range_of_trajectory=None, num_time_points_for_trajectory=None,
                                            time_range_of_trajectory_to_plot=None, num_time_points_for_trajectory_to_plot=None,
                                            traj_point_features=['monkey_distance', 'monkey_angle']):
    """
    Retrieve cached trajectory feature names, or generate and optionally save them.
    """
    file_path = os.path.join(
        raw_data_dir_name, monkey_name, 'all_traj_feature_names.pkl')

    if os.path.exists(file_path) and exists_ok:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    if None in [time_range_of_trajectory, num_time_points_for_trajectory,
                time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot]:
        raise ValueError(
            'If retrieval fails, all keyword arguments must be provided.')

    all_traj_feature_names = make_all_traj_feature_names(
        time_range_of_trajectory, num_time_points_for_trajectory,
        time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot,
        traj_point_features=traj_point_features
    )

    if save:
        os.makedirs(os.path.join(raw_data_dir_name,
                    monkey_name), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(all_traj_feature_names, f)

    return all_traj_feature_names

# ============================================================
#  Angle and distance utilities
# ============================================================


def calculate_monkey_angle_and_distance_from_now_to_other_time(all_current_point_indices, monkey_information,
                                                               monkey_xy_from_other_time):
    """
    Compute distance and heading difference between current monkey position and another time point.
    """
    monkey_xy_now = monkey_information.loc[all_current_point_indices, [
        'monkey_x', 'monkey_y']].values
    monkey_angle_now = monkey_information.loc[all_current_point_indices,
                                              'monkey_angle'].values

    distances = np.linalg.norm(
        monkey_xy_from_other_time - monkey_xy_now, axis=1)
    angles = specific_utils.calculate_angles_to_ff_centers(
        ff_x=monkey_xy_from_other_time[:,
                                       0], ff_y=monkey_xy_from_other_time[:, 1],
        mx=monkey_xy_now[:, 0], my=monkey_xy_now[:,
                                                 1], m_angle=monkey_angle_now
    )
    return distances, angles


def add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(df, monkey_information, ff_dataframe,
                                                                                    monkey_xy_from_other_time=None,
                                                                                    use_last_seen=True):
    """
    Add features: distance and angle from current position to when FF was last or next seen.
    """
    prefix = 'last_seen' if use_last_seen else 'next_seen'
    df = df.copy()

    if monkey_xy_from_other_time is None:
        placeholder = {'monkey_x': [9999, False], 'monkey_y': [False, False]}
        df = ff_data_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(
            df, ff_dataframe, attributes=['monkey_x', 'monkey_y'],
            additional_placeholder_mapping=placeholder, use_last_seen=use_last_seen
        )
        monkey_xy_from_other_time = df[[
            f'{prefix}_monkey_x', f'{prefix}_monkey_y']].values

    all_current_point_indices = df['point_index'].values
    distances, angles = calculate_monkey_angle_and_distance_from_now_to_other_time(
        all_current_point_indices, monkey_information, monkey_xy_from_other_time
    )

    df[f'distance_from_monkey_now_to_monkey_when_ff_{prefix}'] = distances
    df[f'angle_from_monkey_now_to_monkey_when_ff_{prefix}'] = angles

    placeholder_idx = np.where(monkey_xy_from_other_time[:, 0] == 9999)[0]
    df.iloc[placeholder_idx, df.columns.get_indexer(
        [f'distance_from_monkey_now_to_monkey_when_ff_{prefix}'])] = 400
    df.iloc[placeholder_idx, df.columns.get_indexer(
        [f'angle_from_monkey_now_to_monkey_when_ff_{prefix}'])] = 0

    return df


def add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(df, ff_dataframe, monkey_information,
                                                                                use_last_seen=True):
    """
    Add features: distance and angle from monkey’s current position to FF when it was last or next seen.
    """
    df = df.copy()
    suffix = '_last_seen' if use_last_seen else '_next_seen'

    all_ff_index = df['ff_index'].values
    all_current_point_indices = df['point_index'].values
    monkey_xy_now = monkey_information.loc[all_current_point_indices, [
        'monkey_x', 'monkey_y']].values
    monkey_angle_now = monkey_information.loc[all_current_point_indices,
                                              'monkey_angle'].values
    all_current_time = monkey_information.loc[all_current_point_indices, 'time'].values

    placeholder = {'ff_x': [9999, False], 'ff_y': [9999, False]}
    ff_info = ff_data_utils.find_attributes_of_ff_when_last_vis_OR_next_visible(
        all_ff_index, all_current_time, ff_dataframe,
        use_last_seen=use_last_seen, attributes=['ff_x', 'ff_y'],
        additional_placeholder_mapping=placeholder
    )

    ff_xy = ff_info[['ff_x', 'ff_y']].values
    df[['ff_x', 'ff_y']] = ff_xy

    df[f'distance_from_monkey_now_to_ff_when_ff{suffix}'] = np.linalg.norm(
        ff_xy - monkey_xy_now, axis=1)
    df[f'angle_from_monkey_now_to_ff_when_ff{suffix}'] = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ff_xy[:, 0], ff_y=ff_xy[:, 1], mx=monkey_xy_now[:, 0],
        my=monkey_xy_now[:, 1], m_angle=monkey_angle_now
    )

    df.loc[df['ff_x'] == 9999,
           f'distance_from_monkey_now_to_ff_when_ff{suffix}'] = 1000
    df.loc[df['ff_x'] == 9999,
           f'angle_from_monkey_now_to_ff_when_ff{suffix}'] = 0

    df.drop(columns=['ff_x', 'ff_y'], inplace=True)
    return df


def add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(df, monkey_information,
                                                                       ff_real_position_sorted, ff_caught_T_new,
                                                                       use_last_seen=True, curv_of_traj_df=None):
    """
    Add curvature difference between monkey’s current position and FF when last or next seen.
    """
    suffix = '_last_seen' if use_last_seen else '_next_seen'
    df = df.copy()
    df, _ = get_missed_ff_data.find_curv_diff_for_ff_info(
        df, monkey_information, ff_real_position_sorted, curv_of_traj_df=curv_of_traj_df
    )
    df.rename(columns={
              'curv_diff': f'curv_diff_from_monkey_now_to_ff_when_ff{suffix}'}, inplace=True)
    return df

# ============================================================
#  Trajectory feature generation
# ============================================================


def generate_trajectory_position_data(time_all, monkey_information,
                                      time_range_of_trajectory=[-0.5, 0.5],
                                      num_time_points_for_trajectory=10,
                                      trajectory_features=['monkey_distance', 'monkey_angle_to_origin']):
    """
    Generate 2D trajectory features (e.g., distance, angle to origin) for each sample.

    Parameters
    ----------
    time_all : array-like
        Array of time points corresponding to each sample.
    monkey_information : pandas.DataFrame
        DataFrame with columns ['time', 'monkey_x', 'monkey_y', 'monkey_angle', ...].
    time_range_of_trajectory : list of float, optional
        Time window [start, end] (in seconds) around each point to extract trajectory info.
    num_time_points_for_trajectory : int, optional
        Number of time samples to extract within the window.
    trajectory_features : list of str, optional
        Which trajectory features to include (e.g., ['monkey_distance', 'monkey_angle_to_origin']).

    Returns
    -------
    traj_points : np.ndarray
        Concatenated trajectory features for each time point (shape: n_samples × (features × time_points)).
    trajectory_feature_names : list of str
        Names of the trajectory features corresponding to traj_points columns.
    """
    relative_times = np.linspace(time_range_of_trajectory[0],
                                 time_range_of_trajectory[1],
                                 num_time_points_for_trajectory)

    traj_time_2d, traj_data = find_trajectory_data(
        time_all, monkey_information,
        time_range_of_trajectory=time_range_of_trajectory,
        num_time_points_for_trajectory=num_time_points_for_trajectory
    )

    traj_x_2d = traj_data['monkey_x']
    traj_y_2d = traj_data['monkey_y']
    monkey_angle_2d = traj_data['monkey_angle']

    monkey_indices = np.searchsorted(monkey_information['time'], time_all)

    traj_distances, traj_angles, monkey_angle_rel_north = find_monkey_info_on_trajectory_relative_to_origin(
        monkey_indices, monkey_information, traj_x_2d, traj_y_2d, monkey_angle_2d,
        num_time_points_for_trajectory=num_time_points_for_trajectory
    )

    traj_data['monkey_distance'] = traj_distances
    traj_data['monkey_angle_to_origin'] = traj_angles
    traj_data['monkey_angle_on_trajectory_relative_to_the_current_north'] = monkey_angle_rel_north

    # Verify requested features exist
    for feature in trajectory_features:
        if feature not in traj_data:
            raise ValueError(
                f"Feature '{feature}' not found in trajectory data keys.")

    traj_points = np.concatenate([traj_data[f]
                                 for f in trajectory_features], axis=1)

    feature_names = generate_feature_names_given_relative_time_points(
        relative_times, num_time_points_for_trajectory, original_feature_names=trajectory_features
    )

    return traj_points, feature_names


def generate_trajectory_velocity_data(time_all, monkey_information,
                                      time_range_of_trajectory=[-0.5, 0.5],
                                      num_time_points_for_trajectory=10):
    """
    Generate 2D velocity features (speed, angular speed) for each sample.

    Parameters
    ----------
    time_all : array-like
        Array of time points corresponding to each sample.
    monkey_information : pandas.DataFrame
        DataFrame containing 'speed' and 'ang_speed' columns.
    time_range_of_trajectory : list of float, optional
        Time window [start, end] (in seconds) around each point to extract velocity info.
    num_time_points_for_trajectory : int, optional
        Number of time samples to extract within the window.

    Returns
    -------
    monkey_dvdw : np.ndarray
        Concatenated linear and angular velocity features.
    trajectory_feature_names : list of str
        Names of the corresponding features.
    """
    relative_times = np.linspace(time_range_of_trajectory[0],
                                 time_range_of_trajectory[1],
                                 num_time_points_for_trajectory)

    _, traj_data = find_trajectory_data(
        time_all, monkey_information,
        time_range_of_trajectory=time_range_of_trajectory,
        num_time_points_for_trajectory=num_time_points_for_trajectory
    )

    traj_dv_2d = traj_data['speed']
    traj_dw_2d = traj_data['ang_speed']

    monkey_dvdw = np.concatenate([traj_dv_2d, traj_dw_2d], axis=1)

    feature_names = generate_feature_names_given_relative_time_points(
        relative_times, num_time_points_for_trajectory,
        original_feature_names=['monkey_dv', 'ang_speed']
    )

    return monkey_dvdw, feature_names
