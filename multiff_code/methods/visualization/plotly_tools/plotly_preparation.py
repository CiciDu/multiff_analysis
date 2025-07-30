import sys
from decision_making_analysis.decision_making import decision_making_utils, plot_decision_making
from decision_making_analysis import free_selection, replacement, trajectory_info
from data_wrangling import specific_utils
from pattern_discovery import cluster_analysis
from visualization.plotly_polar_tools import plotly_for_trajectory_polar
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, curvature_utils, curv_of_traj_utils

import ast
import os
import sys
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


def prepare_to_plot_a_planning_instance_in_plotly(row, PlotTrials_args, monkey_plot_params=None):
    if monkey_plot_params is None:
        monkey_plot_params = {}

    # Define default parameters
    default_params = {
        'rotation_matrix': None,
        'show_alive_fireflies': False,
        'show_visible_fireflies': False,
        'show_in_memory_fireflies': False,
        'show_connect_path_ff': False,
        'show_visible_segments': True,
        'connect_path_ff_max_distance': 500,
        'eliminate_irrelevant_points_beyond_boundaries': True,
        'stop_point_index': None
    }

    # Update default parameters with provided parameters
    monkey_plot_params = {
        **default_params,
        **monkey_plot_params
    }

    rotation_matrix = monkey_plot_params['rotation_matrix']

    # Unpack PlotTrials arguments
    monkey_information, ff_dataframe, ff_life_sorted, ff_real_position_sorted, _, _, _ = PlotTrials_args

    # Find the duration to plot
    duration_to_plot = _find_duration_to_plot(
        row, monkey_information, monkey_plot_params['eliminate_irrelevant_points_beyond_boundaries']
    )

    # Create trajectory DataFrame and update rotation matrix
    trajectory_df, rotation_matrix = _make_trajectory_df(
        PlotTrials_args, row=row, duration_to_plot=duration_to_plot, rotation_matrix=rotation_matrix
    )

    # Filter firefly data within the duration
    ff_dataframe_in_duration = ff_dataframe[
        (ff_dataframe['time'].between(duration_to_plot[0], duration_to_plot[1]))].sort_values(by='ff_distance', ascending=False).copy()
    ff_dataframe_in_duration_visible = ff_dataframe_in_duration[ff_dataframe_in_duration['visible'] == 1].copy(
    )

    # Create connect path DataFrame and find shown firefly indices
    show_connect_path_ff_specific_indices = [
        int(row.cur_ff_index), int(row.nxt_ff_index)]
    if monkey_plot_params['show_connect_path_ff']:
        connect_path_ff_df, shown_ff_indices = _make_connect_path_ff_df(
            row, monkey_plot_params['show_connect_path_ff_specific_indices'], show_connect_path_ff_specific_indices, ff_dataframe_in_duration_visible,
            rotation_matrix, connect_path_ff_max_distance=monkey_plot_params[
                'connect_path_ff_max_distance']
        )
    else:
        connect_path_ff_df = None

    shown_ff_indices = _find_ff_to_be_plotted(
        monkey_plot_params, row, rotation_matrix, ff_dataframe_in_duration_visible, duration_to_plot, ff_life_sorted, ff_real_position_sorted,
    )

    # Create firefly DataFrame
    ff_df = _make_ff_df(
        shown_ff_indices, ff_real_position_sorted, rotation_matrix)

    # Add firefly number to DataFrame
    show_visible_segments_ff_specific_indices = pd.unique(np.concatenate(
        [show_connect_path_ff_specific_indices, ff_dataframe_in_duration_visible.ff_index.unique()])
    ).astype(int)
    ff_df, ff_number_df = _add_ff_number_to_ff_df(
        ff_df, show_visible_segments_ff_specific_indices)

    # make sure row is a Series
    if isinstance(row, pd.DataFrame) and row.shape[0] == 1:
        row = row.iloc[0]

    # Create current plotly key components
    current_plotly_key_comp = {
        'duration_to_plot': duration_to_plot,
        'trajectory_df': trajectory_df,
        'ff_df': ff_df,
        'connect_path_ff_df': connect_path_ff_df,
        'rotation_matrix': rotation_matrix,
        'row': row,
        'stop_point_index': monkey_plot_params['stop_point_index']
    }

    # Modify current plotly key components based on whether to show visible segments
    ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible.merge(
        ff_number_df, on='ff_index', how='left')
    current_plotly_key_comp = _modify_current_plotly_key_comp_based_on_whether_show_visible_segments(
        monkey_plot_params['show_visible_segments'], current_plotly_key_comp, monkey_information,
        ff_dataframe_in_duration_visible_qualified, show_visible_segments_ff_specific_indices
    )

    return current_plotly_key_comp


def _find_duration_to_plot(row, monkey_information, eliminate_irrelevant_points_beyond_boundaries=False):
    # if row is a one-row df, then make it a series
    if isinstance(row, pd.DataFrame) and row.shape[0] == 1:
        row = row.iloc[0]
    time = row.stop_time
    duration_to_plot = [time-4, max(time+2.5, row.next_stop_time+1.5)]
    if eliminate_irrelevant_points_beyond_boundaries:
        relevant_point_indices = [
            row.stop_point_index, row.next_stop_point_index]
        duration_to_plot = show_null_trajectory.eliminate_irrelevant_points_before_or_after_crossing_boundary(
            duration_to_plot, relevant_point_indices, monkey_information, verbose=False)
    print('duration_to_plot:', duration_to_plot)
    return duration_to_plot


def _make_trajectory_df(PlotTrials_args,
                        row=None,
                        duration_to_plot=None,
                        rotation_matrix=None):

    monkey_information = PlotTrials_args[0]

    cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
        duration_to_plot, monkey_information)
    cum_distance = np.array(monkey_information['cum_distance'])[cum_pos_index]

    if rotation_matrix is None:
        rotation_matrix = plot_behaviors_utils.find_rotation_matrix(
            cum_mx, cum_my)

    # create trajectory_df
    cum_mxy_rotated = np.matmul(rotation_matrix, np.stack((cum_mx, cum_my)))
    trajectory_df = pd.DataFrame({'monkey_x': cum_mxy_rotated[0], 'monkey_y': cum_mxy_rotated[1],
                                  'point_index': cum_point_index, 'time': cum_t, 'monkey_angle': cum_angle,
                                  'monkey_speed': cum_speed, 'monkey_speeddummy': cum_speeddummy, 'cum_distance': cum_distance})

    trajectory_df = _add_eye_positions_columns(
        trajectory_df, monkey_information)

    if row is not None:
        if isinstance(row, pd.DataFrame) and row.shape[0] == 1:
            row = row.iloc[0]
        rel_time = np.round(cum_t - row.stop_time, 2)
        rel_distance = np.round(cum_distance - row.stop_cum_distance, 2)
        trajectory_df['rel_distance'] = rel_distance
        trajectory_df['rel_time'] = rel_time

    return trajectory_df, rotation_matrix


def _add_eye_positions_columns(trajectory_df, monkey_information):
    # in case we need to plot eye positions later
    eye_positions_columns = ['point_index', 'gaze_world_x', 'gaze_world_y',
                             'gaze_mky_view_x', 'gaze_mky_view_y', 'LDz', 'RDz', 'valid_view_point']
    if 'gaze_world_x_l' in monkey_information.columns:
        eye_positions_columns.extend(['gaze_world_x_l', 'gaze_world_y_l', 'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'valid_view_point_l',
                                      'gaze_world_x_r', 'gaze_world_y_r', 'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'valid_view_point_r'])
    try:
        trajectory_df = trajectory_df.merge(
            monkey_information[eye_positions_columns], on='point_index', how='left')
    except KeyError:
        pass
    return trajectory_df


def _find_ff_to_be_plotted(monkey_plot_params, row, rotation_matrix, ff_dataframe_in_duration, duration_to_plot, ff_life_sorted, ff_real_position_sorted):
    shown_ff_indices = []

    ff_dataframe_in_duration_visible = ff_dataframe_in_duration.loc[ff_dataframe_in_duration['visible'] == 1].copy(
    )
    ff_dataframe_in_duration_in_memory = ff_dataframe_in_duration.loc[(ff_dataframe_in_duration['visible'] == 0) &
                                                                      (ff_dataframe_in_duration['ff_distance'] <= 400)].copy()

    if monkey_plot_params['show_alive_fireflies']:
        alive_ff_indices, alive_ff_position_rotated = plot_behaviors_utils.find_alive_ff(
            duration_to_plot, ff_life_sorted, ff_real_position_sorted, rotation_matrix=rotation_matrix)
        shown_ff_indices.extend(alive_ff_indices)
    if monkey_plot_params['show_visible_fireflies']:
        shown_ff_indices.extend(
            ff_dataframe_in_duration_visible.ff_index.unique())
    if monkey_plot_params['show_in_memory_fireflies']:
        shown_ff_indices.extend(
            ff_dataframe_in_duration_in_memory.ff_index.unique())

    if monkey_plot_params['show_cur_ff']:
        shown_ff_indices.append(int(row.cur_ff_index))
    if monkey_plot_params['show_nxt_ff']:
        shown_ff_indices.append(int(row.nxt_ff_index))

    return shown_ff_indices


def _make_connect_path_ff_df(row,
                             shown_ff_indices,
                             show_connect_path_ff_specific_indices,
                             ff_dataframe_in_duration_visible,
                             rotation_matrix,
                             connect_path_ff_max_distance=500):
    # Filter fireflies within the maximum distance
    ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible[
        ff_dataframe_in_duration_visible['ff_distance'] <= connect_path_ff_max_distance
    ]

    # Filter specific fireflies if provided
    if show_connect_path_ff_specific_indices is not None:
        ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible_qualified[
            ff_dataframe_in_duration_visible_qualified['ff_index'].isin(
                show_connect_path_ff_specific_indices)
        ]

    # Create DataFrame for connecting path
    connect_path_ff_df = ff_dataframe_in_duration_visible_qualified[[
        'ff_x', 'ff_y', 'monkey_x', 'monkey_y', 'point_index', 'time', 'cum_distance'
    ]].copy()

    # Add additional columns
    connect_path_ff_df['counter'] = np.arange(connect_path_ff_df.shape[0])
    connect_path_ff_df['rel_time'] = np.round(
        connect_path_ff_df['time'] - row.stop_time, 2)
    connect_path_ff_df['rel_distance'] = np.round(
        connect_path_ff_df['cum_distance'] - row.stop_cum_distance, 2)

    connect_path_ff_df = _rotate_positions_in_connect_path_ff_df(
        connect_path_ff_df, rotation_matrix)

    # Update shown firefly indices
    shown_ff_indices.extend(
        ff_dataframe_in_duration_visible_qualified['ff_index'].unique())

    return connect_path_ff_df, shown_ff_indices


def _rotate_positions_in_connect_path_ff_df(connect_path_ff_df, rotation_matrix):
    # Rotate firefly positions
    ff_positions_rotated = np.matmul(
        rotation_matrix, connect_path_ff_df[['ff_x', 'ff_y']].T.values)
    connect_path_ff_df[['ff_x', 'ff_y']] = ff_positions_rotated.T

    # Rotate monkey positions
    monkey_positions_rotated = np.matmul(
        rotation_matrix, connect_path_ff_df[['monkey_x', 'monkey_y']].T.values)
    connect_path_ff_df[['monkey_x', 'monkey_y']] = monkey_positions_rotated.T

    return connect_path_ff_df


def _make_ff_df(shown_ff_indices, ff_real_position_sorted, rotation_matrix):
    shown_ff_indices = np.unique(np.array(shown_ff_indices)).astype(int)
    ff_positions_rotated = np.matmul(
        rotation_matrix, ff_real_position_sorted[shown_ff_indices].T)
    ff_df = pd.DataFrame(
        {'ff_x': ff_positions_rotated[0], 'ff_y': ff_positions_rotated[1], 'ff_index': shown_ff_indices})
    return ff_df


def _add_ff_number_to_ff_df(ff_df, show_visible_segments_ff_specific_indices):
    ff_number_df = pd.DataFrame({'ff_index': show_visible_segments_ff_specific_indices,
                                 'ff_number': np.arange(1, len(show_visible_segments_ff_specific_indices) + 1)})
    ff_df = ff_df.merge(ff_number_df, on='ff_index', how='left')
    return ff_df, ff_number_df


def _modify_current_plotly_key_comp_based_on_whether_show_visible_segments(show_visible_segments, current_plotly_key_comp, monkey_information, ff_dataframe_in_duration_visible_qualified,
                                                                           show_connect_path_ff_specific_indices):
    current_plotly_key_comp['show_visible_segments'] = show_visible_segments
    if show_visible_segments:
        if show_connect_path_ff_specific_indices is not None:
            ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible_qualified.loc[
                ff_dataframe_in_duration_visible_qualified['ff_index'].isin(show_connect_path_ff_specific_indices)]

        current_plotly_key_comp['ff_dataframe_in_duration_visible_qualified'] = ff_dataframe_in_duration_visible_qualified.copy(
        )
        current_plotly_key_comp['monkey_information'] = monkey_information.copy(
        )
    return current_plotly_key_comp


def find_show_stop_point_indices(monkey_plot_params, current_plotly_key_comp):
    show_stop_point_indices = monkey_plot_params.get(
        'show_stop_point_indices')

    if show_stop_point_indices is None:
        if monkey_plot_params['show_stops']:
            trajectory_df = current_plotly_key_comp['trajectory_df']
            show_stop_point_indices = trajectory_df[
                trajectory_df['monkey_speeddummy'] == 0]['point_index'].values

    show_stop_point_indices = np.array(show_stop_point_indices).reshape(-1)

    return show_stop_point_indices
