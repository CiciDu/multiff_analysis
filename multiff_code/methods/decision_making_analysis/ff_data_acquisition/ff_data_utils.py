
from data_wrangling import specific_utils
from null_behaviors import curvature_utils, curv_of_traj_utils
import math

import os
import torch
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import warnings

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def find_time_since_last_vis(ff_indices, all_current_time, ff_dataframe, return_duration_of_last_vis_period=False):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    current_time_df = pd.DataFrame({'unique_id': np.arange(
        len(ff_indices)), 'ff_index': ff_indices, 'current_time': all_current_time})

    ff_visible_df = ff_dataframe_visible[(
        ff_dataframe_visible['ff_index'].isin(ff_indices))]
    ff_visible_df = ff_visible_df[['ff_index', 'time', 'point_index']].copy()
    # make sure that for each ff, the time in dataframe doesn't exceed the max time for that ff
    ff_visible_df = pd.merge(
        ff_visible_df, current_time_df, how='left', on='ff_index')
    ff_visible_df = ff_visible_df[ff_visible_df['time']
                                  <= ff_visible_df['current_time']]
    # for each ff, get the last visible time before max time
    ff_visible_df.sort_values(['ff_index', 'time'], ascending=[
                              True, False], inplace=True)
    ff_visible_df = add_time_when_last_vis_period_began_when_calculating_time_since_last_vis(
        ff_visible_df)

    ff_visible_df = ff_visible_df.groupby(
        ['ff_index', 'current_time']).first().reset_index(drop=False)
    # now, make sure we have info of all the ff's
    ff_visible_df = ff_visible_df.drop(columns=['ff_index', 'current_time'])
    ff_visible_df = pd.merge(
        ff_visible_df, current_time_df, how='right', on='unique_id').fillna(0)
    ff_visible_df['time_since_last_vis'] = ff_visible_df['current_time'].values - \
        ff_visible_df['time'].values
    if return_duration_of_last_vis_period:
        moment_when_ff_last_vis = ff_visible_df['current_time'] - \
            ff_visible_df['time_since_last_vis']
        ff_visible_df['duration_of_last_vis_period'] = moment_when_ff_last_vis - \
            ff_visible_df['time_when_last_vis_period_began']
        return ff_visible_df['time_since_last_vis'].values, ff_visible_df['duration_of_last_vis_period'].values
    else:
        return ff_visible_df['time_since_last_vis'].values


def add_time_when_last_vis_period_began_when_calculating_time_since_last_vis(ff_visible_df):
    # oh shoot...i only want df within ff_index though
    ff_visible_df2 = ff_visible_df.copy()
    ff_visible_df2.sort_values(['ff_index', 'current_time', 'time'], ascending=[
                               True, True, True], inplace=True)
    ff_visible_df2['point_index_diff'] = ff_visible_df2.groupby(
        ['ff_index', 'current_time'])['point_index'].diff(periods=1).fillna(100)
    # only keep the rows that mark the beginning of a new visible period
    ff_visible_df2 = ff_visible_df2[ff_visible_df2['point_index_diff'] > 1]
    ff_visible_df2 = ff_visible_df2.groupby(
        ['ff_index', 'current_time']).last().reset_index(drop=False)
    ff_visible_df2.rename(
        columns={'time': 'time_when_last_vis_period_began'}, inplace=True)
    ff_visible_df2 = ff_visible_df2[[
        'ff_index', 'current_time', 'time_when_last_vis_period_began']].copy()

    ff_visible_df = ff_visible_df.merge(ff_visible_df2, how='left', on=[
                                        'ff_index', 'current_time'])
    return ff_visible_df


def find_attributes_of_ff_when_last_vis_OR_next_visible(ff_indices, all_current_time, ff_dataframe, use_last_seen=True, attributes=['ff_distance', 'ff_angle', 'curv_diff'],
                                                        additional_placeholder_mapping=None):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    current_time_df = pd.DataFrame({'unique_id': np.arange(
        len(ff_indices)), 'ff_index': ff_indices, 'current_time': all_current_time})

    ff_visible_df = ff_dataframe_visible[(
        ff_dataframe_visible['ff_index'].isin(ff_indices))]
    # make sure that for each ff, the time in dataframe doesn't exceed the max time for that ff
    ff_visible_df = pd.merge(
        ff_visible_df, current_time_df, how='left', on='ff_index')
    if use_last_seen:
        ff_visible_df = ff_visible_df[ff_visible_df['time']
                                      <= ff_visible_df['current_time']]
    else:
        ff_visible_df = ff_visible_df[ff_visible_df['time']
                                      > ff_visible_df['current_time']]
    # for each ff, get the last visible time before max time
    whether_time_should_be_ascending = (use_last_seen == False)
    ff_visible_df.sort_values(['ff_index', 'time'], ascending=[
                              True, whether_time_should_be_ascending], inplace=True)
    ff_visible_df = ff_visible_df.groupby(
        ['ff_index', 'current_time']).first().reset_index(drop=False)
    # now, make sure we have info of all the ff's
    ff_info = ff_visible_df[['unique_id'] + attributes].copy()
    ff_info = pd.merge(ff_info, current_time_df, how='right',
                       on='unique_id').drop(columns=['unique_id'])

    ff_info = fill_NA_in_columns(
        ff_info, columns=attributes, additional_placeholder_mapping=additional_placeholder_mapping)

    return ff_info


def find_placeholder_values_for_attributes(attributes=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'curv_diff'], additional_placeholder_mapping=None):
    placeholder_mapping = {'ff_distance': [400, False],
                           'ff_angle': [-math.pi/4, True],
                           'ff_angle_boundary': [-math.pi/4, True],
                           'curv_diff': [-0.6, True],
                           'abs_curv_diff': [0.6, False],
                           'monkey_x': [9999, False],
                           'monkey_y': [9999, False]
                           }

    if additional_placeholder_mapping is not None:
        for key, value in additional_placeholder_mapping.items():
            placeholder_mapping[key] = value

    placeholder_values = []
    whether_placeholder_using_both_signs = []
    for attribute in attributes:
        placeholder_values.append(placeholder_mapping[attribute][0])
        whether_placeholder_using_both_signs.append(
            placeholder_mapping[attribute][1])

    return placeholder_values, whether_placeholder_using_both_signs


def fill_NA_in_columns(df, columns=['ff_distance', 'ff_angle'], additional_placeholder_mapping=None):
    df = df.copy()
    placeholder_values, placeholder_using_both_signs = find_placeholder_values_for_attributes(
        attributes=columns, additional_placeholder_mapping=additional_placeholder_mapping)

    for i in range(len(columns)):
        na_index = df[columns[i]].isna()
        if placeholder_using_both_signs[i] == True:
            df.loc[na_index, columns[i]] = np.random.choice(
                [-1, 1], size=np.sum(na_index)) * placeholder_values[i]
        else:
            df.loc[na_index, columns[i]] = placeholder_values[i]
    return df


def fill_new_columns_with_placeholder_values(df, columns=['ff_distance', 'ff_angle'], additional_placeholder_mapping=None):

    placeholder_values, placeholder_using_both_signs = find_placeholder_values_for_attributes(
        attributes=columns, additional_placeholder_mapping=additional_placeholder_mapping)

    num_rows = len(df)
    for i in range(len(columns)):
        if placeholder_using_both_signs[i] == True:
            df[columns[i]] = np.random.choice(
                [-1, 1], size=num_rows) * placeholder_values[i]
        else:
            df[columns[i]] = placeholder_values[i]
    return df


def add_attributes_last_seen_or_next_seen_for_each_ff_in_df(df, ff_dataframe, attributes=['ff_distance', 'ff_angle', 'curv_diff'], use_last_seen=True, additional_placeholder_mapping=None):

    df = df.copy()
    ff_info = find_attributes_of_ff_when_last_vis_OR_next_visible(df.ff_index.values, df.time.values, ff_dataframe, use_last_seen=use_last_seen, attributes=attributes,
                                                                  additional_placeholder_mapping=additional_placeholder_mapping)

    ff_info = ff_info[attributes].copy()
    prefix = 'last_seen_' if use_last_seen else 'next_seen_'
    ff_info.columns = [prefix + x for x in attributes]
    for column in ff_info.columns:
        df[column] = ff_info[column].values

    return df


def add_curv_diff_to_df(df, monkey_information, curv_of_traj_df, ff_real_position_sorted=None, ff_radius_for_opt_arc=10):
    # Note: df should at least contain ff_index, point_index, ff_distance, ff_angle, ff_angle_boundary, monkey_x, monkey_y, monkey_angle
    df = df.copy()
    if 'monkey_x' not in df.columns:
        df['monkey_x'] = monkey_information.loc[df['point_index'].values,
                                                'monkey_x'].values
        df['monkey_y'] = monkey_information.loc[df['point_index'].values,
                                                'monkey_y'].values
        df['monkey_angle'] = monkey_information.loc[df['point_index'].values,
                                                    'monkey_angle'].values
    if 'ff_x' not in df.columns:
        if ff_real_position_sorted is None:
            raise ValueError(
                'ff_real_position_sorted should be provided if ff_x and ff_y are not in df')
        df['ff_x'] = ff_real_position_sorted[df['ff_index'].values, 0]
        df['ff_y'] = ff_real_position_sorted[df['ff_index'].values, 1]
    curvature_df = curvature_utils.make_curvature_df(
        df, curv_of_traj_df, ff_radius_for_opt_arc=ff_radius_for_opt_arc)
    curvature_df = curvature_df[[
        'ff_index', 'point_index', 'curv_diff', 'abs_curv_diff']].drop_duplicates()
    if 'curv_diff' in df.columns:
        df.drop(columns=['curv_diff'], inplace=True)
    if 'abs_curv_diff' in df.columns:
        df.drop(columns=['abs_curv_diff'], inplace=True)
    df = pd.merge(df, curvature_df, how='left', on=['ff_index', 'point_index'])
    # fill NAs
    df = fill_NA_in_columns(df, columns=['curv_diff', 'abs_curv_diff'])
    return df


def find_many_ff_info_anew(ff_indices, point_index, ff_real_position_sorted, ff_dataframe_visible, monkey_information, add_curv_diff=False,
                           ff_caught_T_new=None, window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False,
                           ff_radius=10, curv_of_traj_df=None, add_duration_of_last_vis_period=True):
    """
    Computes various information about multiple fireflies and their relationship to a monkey, given their indices and positions.

    Parameters:
    -----------
    ff_indices : array-like of int
        Indices of the fireflies to analyze.
    point_index : array-like of int
        Indices of the corresponding monkey positions.
    ff_real_position_sorted : array-like of float
        Positions of all fireflies, sorted by their capture time.
    ff_dataframe_visible : pandas DataFrame
        DataFrame containing information about visible fireflies.
    monkey_information : pandas DataFrame
        DataFrame containing information about monkey such as positions and angles.
    # time till next visible: no longer supported

    Returns:
    --------
    ff_info : pandas DataFrame
        DataFrame containing various information about the fireflies and their relationship to the monkey.
    """

    if np.any(ff_indices < 0):
        warnings.warn(
            'ff_indices should not contain negative values. Negative ff_indices are likely to be placeholders. They will be ignored.')
        point_index = point_index[ff_indices >= 0]
        ff_indices = ff_indices[ff_indices >= 0]
    if len(ff_indices) != len(point_index):
        raise ValueError(
            'ff_indices and point_index should have the same length')
    ff_xy = ff_real_position_sorted[ff_indices, :]
    monkey_info = monkey_information.loc[point_index]
    monkey_xy = monkey_info[['monkey_x', 'monkey_y']].values
    monkey_angle = monkey_info['monkey_angle'].values
    all_current_time = monkey_info['time'].values
    ff_distance = np.linalg.norm(ff_xy-monkey_xy, axis=1)
    ff_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ff_xy[:, 0], ff_y=ff_xy[:, 1], mx=monkey_xy[:, 0], my=monkey_xy[:, 1], m_angle=monkey_angle)
    ff_angle_boundary = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_angle, distances_to_ff=ff_distance)
    # find time since last visible
    time_since_last_vis, duration_of_last_vis_period = find_time_since_last_vis(
        ff_indices, all_current_time, ff_dataframe_visible, return_duration_of_last_vis_period=True)
    # ff_info = pd.DataFrame([ff_distance, ff_angle, ff_angle_boundary, time_since_last_vis], columns=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis'])
    ff_info = pd.DataFrame({'ff_index': ff_indices.astype(int),
                            'point_index': point_index.astype(int),
                            'time': all_current_time,
                            'ff_distance': ff_distance,
                            'ff_angle': ff_angle,
                            'ff_angle_boundary': ff_angle_boundary,
                            'abs_ff_angle': np.abs(ff_angle),
                            'abs_ff_angle_boundary': np.abs(ff_angle_boundary),
                            'time_since_last_vis': time_since_last_vis,
                            })
    # time till next visible removed

    if add_curv_diff:
        if curv_of_traj_df is None:
            curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, monkey_information, ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode,
                                                                                                                  truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        if (ff_caught_T_new is None):
            raise ValueError(
                'ff_caught_T_newshould be provided if add_curv_diff is True')
        ff_info = add_curv_diff_to_df(ff_info, monkey_information, curv_of_traj_df,
                                      ff_real_position_sorted=ff_real_position_sorted, ff_radius_for_opt_arc=ff_radius)

    if add_duration_of_last_vis_period:
        ff_info['duration_of_last_vis_period'] = duration_of_last_vis_period
    return ff_info


def find_one_ff_info_anew_at_one_point(ff_index, point_index, ff_real_position_sorted, ff_dataframe, monkey_information):
    # This function is only intended to use on a single ff
    ff_xy = ff_real_position_sorted[ff_index, :]
    monkey_info = monkey_information.loc[monkey_information['point_index'] == point_index]
    monkey_xy = monkey_info[['monkey_x', 'monkey_y']].values.reshape(-1)
    monkey_angle = monkey_info['monkey_angle'].item()
    time = monkey_info['time'].item()
    ff_distance = np.linalg.norm(ff_xy-monkey_xy)
    ff_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ff_xy[0], ff_y=ff_xy[1], mx=monkey_xy[0], my=monkey_xy[1], m_angle=monkey_angle)
    ff_angle_boundary = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_angle, distances_to_ff=ff_distance)
    # find time since last visible
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    ff_visible_df = ff_dataframe_visible[(ff_dataframe_visible['ff_index'] == ff_index) & (
        ff_dataframe_visible['time'] <= time)]
    if len(ff_visible_df) > 0:
        ff_last_vis = ff_visible_df.time.max()
    else:
        ff_last_vis = 0

    time_since_last_vis = time - ff_last_vis

    # we want to make sure that we're just using float or int, and there's no array
    if (isinstance(ff_angle, float) is False) & (isinstance(ff_angle, int) is False):
        ff_angle = ff_angle[0]
    if (isinstance(ff_angle_boundary, float) is False) & (isinstance(ff_angle_boundary, int) is False):
        ff_angle_boundary = ff_angle_boundary[0]

    return ff_distance, ff_angle, ff_angle_boundary, time_since_last_vis


def find_one_ff_info_at_one_point(ff_index, point_index, ff_real_position_sorted, ff_dataframe, monkey_information):
    # This function is only intended to use on a single ff

    raw_ff_info = ff_dataframe[(ff_dataframe['ff_index'] == ff_index) & (
        ff_dataframe['point_index'] == point_index)].copy()
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    # if there's no corresponding info, then calculate it now.
    if len(raw_ff_info) == 0:
        ff_distance, ff_angle, ff_angle_boundary, time_since_last_vis = find_one_ff_info_anew_at_one_point(int(
            ff_index), point_index, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
        # ff_info = pd.DataFrame([[ff_distance, ff_angle, ff_angle_boundary, time_since_last_vis]], columns=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis'])
    else:
        ff_info = raw_ff_info[['ff_distance', 'ff_angle',
                               'ff_angle_boundary', 'time_since_last_vis']]
    return ff_info


def get_distance_and_angle_from_previous_target(current_ff_positions, prev_target_caught_T, monkey_information):

    target_distances = []
    target_angles = []
    for i in range(len(prev_target_caught_T)):
        current_ff_position = current_ff_positions[i]
        monkey_info = monkey_information[monkey_information['time']
                                         >= prev_target_caught_T[i]].iloc[0]
        monkey_xy = monkey_info[['monkey_x', 'monkey_y']].values.reshape(-1)
        ffxy = current_ff_position
        target_distances.append(np.linalg.norm(monkey_xy - ffxy))
        target_angles.append(specific_utils.calculate_angles_to_ff_centers(
            ff_x=ffxy[0], ff_y=ffxy[1], mx=monkey_xy[0], my=monkey_xy[1], m_angle=monkey_info['monkey_angle']))
    target_distances = np.array(target_distances)
    target_angles = np.array(target_angles)
    return target_distances, target_angles


def find_crossing_boundary_trials(original_trials, trial_ending_time, monkey_information, min_time_no_crossing_boundary):
    crossing_boundary_trials = []
    for i in range(len(original_trials)):
        trial = original_trials[i]
        duration = [trial_ending_time[i] -
                    min_time_no_crossing_boundary, trial_ending_time[i]]
        cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & (
            monkey_information['time'] <= duration[1]))[0]
        cum_t, cum_crossing_boundary = np.array(monkey_information['time'].iloc[cum_pos_index]), np.array(
            monkey_information['crossing_boundary'].iloc[cum_pos_index])
        cross_boundary_points = np.where(cum_crossing_boundary == 1)[0]
        if len(cross_boundary_points) > 0:
            crossing_boundary_trials.append(trial)
    crossing_boundary_trials = np.array(crossing_boundary_trials)
    return crossing_boundary_trials


def find_time_points_that_are_within_n_seconds_after_crossing_boundary(input_time, crossing_boundary_time, n_seconds_after_crossing_boundary=2, n_seconds_before_crossing_boundary=0):

    if len(crossing_boundary_time) == 0:
        crossing_boundary_time = np.array([-999])

    # make sure n_seconds_after_crossing_boundary is positive
    n_seconds_before_crossing_boundary = abs(
        n_seconds_before_crossing_boundary)

    corresponding_CB_time = crossing_boundary_time[np.searchsorted(
        crossing_boundary_time, input_time)-1]
    time_since_CB = input_time - corresponding_CB_time
    # if any element in time_since_CB is negative, then it should be changed to a big number because it means the monkey hasn't crossed the boundary prior to that point
    time_since_CB[time_since_CB < 0] = 1000
    CB_indices = np.where(
        time_since_CB <= n_seconds_after_crossing_boundary)[0]

    if n_seconds_before_crossing_boundary > 0:
        corresponding_CB_time_indices = np.searchsorted(
            crossing_boundary_time, input_time)
        # make sure the indices for corresponding_CB_time don't go out of bound
        corresponding_CB_time_indices[corresponding_CB_time_indices >= len(
            crossing_boundary_time)] = len(crossing_boundary_time) - 1
        corresponding_CB_time = crossing_boundary_time[corresponding_CB_time_indices]
        time_before_CB = corresponding_CB_time - input_time
        CB_indices = np.append(CB_indices, np.where(
            time_before_CB <= n_seconds_before_crossing_boundary)[0])

    CB_indices = np.unique(CB_indices)
    non_CB_indices = np.setdiff1d(np.arange(len(input_time)), CB_indices)
    input_time = np.array(input_time)
    remaining_input_time = input_time[non_CB_indices]

    return CB_indices, non_CB_indices, remaining_input_time


def make_pred_ff_indices_dict(moit2, y_pred_all):

    pred_ff_indices_dict = {}

    if y_pred_all is None:
        nn_model = moit2.nn_model
        nn_model.eval()
        y_pred_all = nn_model(torch.tensor(
            moit2.X_all_sc, dtype=torch.float32))
        y_pred_all = y_pred_all > 0.5

    for i in range(len(moit2.chosen_rows_of_df)):
        starting_point_index = int(
            moit2.chosen_rows_of_df.iloc[i]['starting_point_index'].item())
        obs_ff_indices = moit2.sequence_of_obs_ff_indices[i]

        if y_pred_all.ndim > 1:
            pred_ff = obs_ff_indices[np.argmax(y_pred_all[i])]
        else:
            if y_pred_all[i] < len(obs_ff_indices):
                pred_ff = obs_ff_indices[y_pred_all[i]]
            else:
                pred_ff = np.array([])
        pred_ff_indices_dict[starting_point_index] = pred_ff

    return pred_ff_indices_dict
