import sys
from decision_making_analysis.decision_making import decision_making_utils, plot_decision_making
from decision_making_analysis import free_selection, replacement, trajectory_info
from data_wrangling import specific_utils
from pattern_discovery import cluster_analysis
from visualization.plotly_polar_tools import plotly_for_trajectory_polar
from visualization import plot_behaviors_utils
from null_behaviors import show_null_trajectory, curvature_utils, curv_of_traj_utils

import ast
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import math

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def prepare_to_plot_a_planning_instance_in_plotly(row,
                                                  PlotTrials_args,
                                                  monkey_plot_params={}):

    default_params = {'rotation_matrix': None,
                      'show_alive_fireflies': False,
                      'show_visible_fireflies': False,
                      'show_in_memory_fireflies': False,
                      'show_connect_path_ff': False,
                      'show_visible_segments': True,
                      'connect_path_ff_max_distance': 500,
                      'eliminate_irrelevant_points_beyond_boundaries': True,
                      'stop_point_index': None}

    default_params.update(monkey_plot_params)
    monkey_plot_params = default_params

    rotation_matrix = monkey_plot_params['rotation_matrix']
    show_alive_fireflies = monkey_plot_params['show_alive_fireflies']
    show_visible_fireflies = monkey_plot_params['show_visible_fireflies']
    show_in_memory_fireflies = monkey_plot_params['show_in_memory_fireflies']
    show_connect_path_ff = monkey_plot_params['show_connect_path_ff']
    show_visible_segments = monkey_plot_params['show_visible_segments']
    connect_path_ff_max_distance = monkey_plot_params['connect_path_ff_max_distance']
    eliminate_irrelevant_points_beyond_boundaries = monkey_plot_params[
        'eliminate_irrelevant_points_beyond_boundaries']
    stop_point_index = monkey_plot_params['stop_point_index']

    time = row.stop_time
    monkey_information = PlotTrials_args[0]
    duration_to_plot = [time-4, max(time+2.5, row.next_stop_time+1.5)]
    show_connect_path_ff_specific_indices = [
        int(row.cur_ff_index), int(row.nxt_ff_index)]

    if eliminate_irrelevant_points_beyond_boundaries:
        relevant_point_indices = [
            row.stop_point_index, row.next_stop_point_index]
        duration_to_plot = show_null_trajectory.eliminate_irrelevant_points_before_or_after_crossing_boundary(
            duration_to_plot, relevant_point_indices, monkey_information, verbose=False)
    print('duration_to_plot:', duration_to_plot)

    monkey_information, ff_dataframe, ff_life_sorted, ff_real_position_sorted, _, _, ff_caught_T_new = PlotTrials_args
    trajectory_df, R = make_trajectory_df(
        PlotTrials_args, row=row, duration_to_plot=duration_to_plot, rotation_matrix=rotation_matrix)

    # then find ff to be plotted
    ff_dataframe_in_duration = ff_dataframe[(ff_dataframe['time'] >= duration_to_plot[0]) & (
        ff_dataframe['time'] <= duration_to_plot[1])].copy()
    ff_dataframe_in_duration.sort_values(
        by='ff_distance', ascending=False, inplace=True)
    ff_dataframe_in_duration_visible = ff_dataframe_in_duration.loc[ff_dataframe_in_duration['visible'] == 1].copy(
    )
    ff_dataframe_in_duration_in_memory = ff_dataframe_in_duration.loc[(ff_dataframe_in_duration['visible'] == 0) &
                                                                      (ff_dataframe_in_duration['ff_distance'] <= 400)].copy()

    shown_ff_indices = []
    if show_alive_fireflies:
        alive_ff_indices, alive_ff_position_rotated = plot_behaviors_utils.find_alive_ff(
            duration_to_plot, ff_life_sorted, ff_real_position_sorted, rotation_matrix=R)
        shown_ff_indices.extend(alive_ff_indices)
    if show_visible_fireflies:
        shown_ff_indices.extend(
            ff_dataframe_in_duration_visible.ff_index.unique())
    if show_in_memory_fireflies:
        shown_ff_indices.extend(
            ff_dataframe_in_duration_in_memory.ff_index.unique())

    # create connect_path_ff_df and add to shown_ff_indices if necessary
    connect_path_ff_df = None
    if show_connect_path_ff:
        connect_path_ff_df, shown_ff_indices = make_connect_path_ff_df(
            row, shown_ff_indices, show_connect_path_ff, show_connect_path_ff_specific_indices, ff_dataframe_in_duration_visible, R, connect_path_ff_max_distance=connect_path_ff_max_distance)

    # create ff_df
    shown_ff_indices = np.unique(np.array(shown_ff_indices)).astype(int)
    ff_positions_rotated = np.matmul(
        R, ff_real_position_sorted[shown_ff_indices].T)
    ff_df = pd.DataFrame(
        {'ff_x': ff_positions_rotated[0], 'ff_y': ff_positions_rotated[1], 'ff_index': shown_ff_indices})

    # add ff_number to ff_df
    show_visible_segments_ff_specific_indices = pd.unique(np.concatenate(
        [show_connect_path_ff_specific_indices, ff_dataframe_in_duration_visible.ff_index.unique()])).astype(int)
    ff_number_df = pd.DataFrame({'ff_index': show_visible_segments_ff_specific_indices,
                                 'ff_number': np.arange(1, len(show_visible_segments_ff_specific_indices) + 1)})
    ff_df = ff_df.merge(ff_number_df, on='ff_index', how='left')
    ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible.merge(
        ff_number_df, on='ff_index', how='left')

    current_plotly_plot_key_comp = {'duration_to_plot': duration_to_plot,
                                    'trajectory_df': trajectory_df,
                                    'ff_df': ff_df,
                                    'connect_path_ff_df': connect_path_ff_df,
                                    'R': R,
                                    'row': row,
                                    'stop_point_index': stop_point_index,
                                    'show_visible_segments': show_visible_segments}

    # modify current_plotly_plot_key_comp based on whether show_visible_segments
    current_plotly_plot_key_comp = _modify_current_plotly_plot_key_comp_based_on_whether_show_visible_segments(show_visible_segments, current_plotly_plot_key_comp, monkey_information, ff_dataframe_in_duration_visible_qualified,
                                                                                                               show_visible_segments_ff_specific_indices)
    return current_plotly_plot_key_comp


def _modify_current_plotly_plot_key_comp_based_on_whether_show_visible_segments(show_visible_segments, current_plotly_plot_key_comp, monkey_information, ff_dataframe_in_duration_visible_qualified,
                                                                                show_connect_path_ff_specific_indices):
    if show_visible_segments:
        if show_connect_path_ff_specific_indices is not None:
            ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible_qualified.loc[
                ff_dataframe_in_duration_visible_qualified['ff_index'].isin(show_connect_path_ff_specific_indices)]

        current_plotly_plot_key_comp['ff_dataframe_in_duration_visible_qualified'] = ff_dataframe_in_duration_visible_qualified.copy(
        )
        current_plotly_plot_key_comp['monkey_information'] = monkey_information.copy(
        )
    return current_plotly_plot_key_comp


def make_trajectory_df(PlotTrials_args,
                       row=None,
                       duration_to_plot=None,
                       rotation_matrix=None):

    monkey_information = PlotTrials_args[0]

    cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
        duration_to_plot, monkey_information)
    cum_point_indices = np.array(monkey_information['point_index'])[
        cum_pos_index]
    cum_distance = np.array(monkey_information['cum_distance'])[cum_pos_index]

    if rotation_matrix is None:
        R = plot_behaviors_utils.find_rotation_matrix(cum_mx, cum_my)
        rotation_matrix = R
    else:
        R = rotation_matrix

    # create trajectory_df
    cum_mxy_rotated = np.matmul(R, np.stack((cum_mx, cum_my)))
    trajectory_df = pd.DataFrame({'monkey_x': cum_mxy_rotated[0], 'monkey_y': cum_mxy_rotated[1],
                                  'point_index': cum_point_indices, 'time': cum_t, 'monkey_angle': cum_angle,
                                  'monkey_speed': cum_speed, 'monkey_speeddummy': cum_speeddummy, 'cum_distance': cum_distance})

    # in case we need to plot eye positions later
    eye_positions_columns = ['point_index', 'gaze_world_x', 'gaze_world_y',
                             'gaze_monkey_view_x', 'gaze_monkey_view_y', 'LDz', 'RDz']
    if 'gaze_world_x_l' in monkey_information.columns:
        eye_positions_columns.extend(['gaze_world_x_l', 'gaze_world_y_l', 'gaze_monkey_view_x_l', 'gaze_monkey_view_y_l',
                                      'gaze_world_x_r', 'gaze_world_y_r', 'gaze_monkey_view_x_r', 'gaze_monkey_view_y_r'])
    try:
        trajectory_df = trajectory_df.merge(
            monkey_information[eye_positions_columns], on='point_index', how='left')
    except KeyError:
        pass

    if row is not None:
        rel_time = np.round(cum_t - row.stop_time, 2)
        rel_distance = np.round(cum_distance - row.stop_cum_distance, 2)
        trajectory_df['rel_distance'] = rel_distance
        trajectory_df['rel_time'] = rel_time

    return trajectory_df, R


def make_connect_path_ff_df(row,
                            shown_ff_indices,
                            show_connect_path_ff,
                            show_connect_path_ff_specific_indices,
                            ff_dataframe_in_duration_visible,
                            R,
                            connect_path_ff_max_distance=500):

    ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible.loc[
        ff_dataframe_in_duration_visible['ff_distance'] <= connect_path_ff_max_distance]
    if show_connect_path_ff_specific_indices is not None:
        ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible_qualified.loc[
            ff_dataframe_in_duration_visible_qualified['ff_index'].isin(show_connect_path_ff_specific_indices)]

    connect_path_ff_df = ff_dataframe_in_duration_visible_qualified[[
        'ff_x', 'ff_y', 'monkey_x', 'monkey_y', 'point_index', 'time', 'cum_distance']].copy()
    connect_path_ff_df['counter'] = np.arange(connect_path_ff_df.shape[0])
    connect_path_ff_df['rel_time'] = np.round(
        connect_path_ff_df['time'] - row.stop_time, 2)
    connect_path_ff_df['rel_distance'] = np.round(
        connect_path_ff_df['cum_distance'] - row.stop_cum_distance, 2)
    # rotated ff_x and ff_y
    ff_positions_rotated = np.matmul(
        R, connect_path_ff_df[['ff_x', 'ff_y']].T.values)
    connect_path_ff_df[['ff_x', 'ff_y']] = np.vstack(
        [ff_positions_rotated[0], ff_positions_rotated[1]]).T
    # rotated monkey_x and monkey_y
    monkey_positions_rotated = np.matmul(
        R, connect_path_ff_df[['monkey_x', 'monkey_y']].T.values)
    connect_path_ff_df[['monkey_x', 'monkey_y']] = np.vstack(
        [monkey_positions_rotated[0], monkey_positions_rotated[1]]).T
    shown_ff_indices.extend(
        ff_dataframe_in_duration_visible_qualified.ff_index.unique())
    return connect_path_ff_df, shown_ff_indices


def find_traj_portion_for_traj_curv(trajectory_df, curv_of_traj_current_row):
    traj_portion = trajectory_df[(trajectory_df.point_index >= curv_of_traj_current_row['min_point_index'].item()) & (
        trajectory_df.point_index <= curv_of_traj_current_row['max_point_index'].item())]
    traj_length = traj_portion['rel_distance'].iloc[-1] - \
        traj_portion['rel_distance'].iloc[0]
    return traj_portion, traj_length


def find_nxt_ff_curv_df(current_plotly_plot_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    nxt_ff_curv_df = find_ff_curv_df(current_plotly_plot_key_comp['row'].nxt_ff_index, current_plotly_plot_key_comp,
                                     ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new)
    return nxt_ff_curv_df


def find_cur_ff_curv_df(current_plotly_plot_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    cur_ff_curv_df = find_ff_curv_df(current_plotly_plot_key_comp['row'].cur_ff_index, current_plotly_plot_key_comp,
                                     ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new)
    return cur_ff_curv_df


def find_ff_curv_df(ff_index, current_plotly_plot_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    duration_to_plot = current_plotly_plot_key_comp['duration_to_plot']
    row = current_plotly_plot_key_comp['row']
    ff_curv_df = curvature_utils.find_curvature_df_for_ff_in_duration(
        ff_dataframe, ff_index, duration_to_plot, monkey_information, curv_of_traj_df,  ff_caught_T_new=ff_caught_T_new, clean=False)
    ff_curv_df['rel_time'] = ff_curv_df['time'] - row.stop_time
    ff_curv_df = ff_curv_df.merge(
        monkey_information[['point_index', 'cum_distance']], on='point_index', how='left')
    ff_curv_df['rel_distance'] = np.round(
        ff_curv_df['cum_distance'] - row.stop_cum_distance, 2)
    ff_curv_df['curv_to_ff_center'] = ff_curv_df['curv_to_ff_center'] * \
        180/np.pi * 100  # convert to degree/cm
    ff_curv_df['optimal_curvature'] = ff_curv_df['optimal_curvature'] * \
        180/np.pi * 100  # convert to degree/cm
    ff_curv_df_sub = ff_curv_df[['point_index', 'rel_time',
                                 'rel_distance', 'curv_to_ff_center', 'optimal_curvature']].copy()
    return ff_curv_df_sub
