import sys
from data_wrangling import specific_utils
from visualization import plot_polar, plot_trials
from non_behavioral_analysis import eye_positions

import os
import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

connection_linewidth = {"agent": 0.25, "monkey": 0.5, "combined": 0.3}
connection_alpha = {"agent": 0.6, "monkey": 0.7, "combined": 0.8}


def plot_a_trial(currentTrial, num_trials, ff_caught_T_new, PlotTrials_args, additional_kwargs=None, images_dir=None):

    plot_kwargs = {'player': 'monkey',
                   'show_stops': True,
                   'show_believed_target_positions': True,
                   'show_reward_boundary': True,
                   'show_scale_bar': True,
                   'show_eye_positions': False,
                   'show_eye_positions_on_the_right': False,
                   'show_connect_path_eye_positions': True,
                   'hitting_arena_edge_ok': True,
                   'trial_too_short_ok': False,
                   'images_dir': None,
                   'show_connect_path_ff': True}

    if additional_kwargs is not None:
        for key, value in additional_kwargs.items():
            plot_kwargs[key] = value

    duration = [ff_caught_T_new[currentTrial-num_trials],
                ff_caught_T_new[currentTrial]]
    returned_info = plot_trials.PlotTrials(duration,
                                           *PlotTrials_args,
                                           **plot_kwargs,
                                           currentTrial=currentTrial,
                                           num_trials=num_trials,
                                           )

    if returned_info['whether_plotted']:
        plt.show()


def plot_a_trial_from_a_category(category_name, currentTrial, num_trials, ff_caught_T_new, PlotTrials_args, all_category_kwargs, additional_kwargs=None, images_dir=None):
    PlotTrials_kargs = all_category_kwargs[category_name]
    PlotTrials_kargs['images_dir'] = images_dir
    if additional_kwargs is not None:
        for key, value in additional_kwargs.items():
            PlotTrials_kargs[key] = value

    duration = [ff_caught_T_new[currentTrial-num_trials],
                ff_caught_T_new[currentTrial]]
    returned_info = plot_trials.PlotTrials(duration,
                                           *PlotTrials_args,
                                           **PlotTrials_kargs,
                                           currentTrial=currentTrial,
                                           num_trials=num_trials,
                                           )
    return returned_info


def plot_trials_from_a_category(category, category_name, max_trial_to_plot, PlotTrials_args, all_category_kwargs,
                                ff_caught_T_new, trials=None, additional_kwargs=None, images_dir=None, using_subplots=False,
                                figsize=(10, 10)):
    # category is an array of trial indices belonging to a category
    # trials is an optional array of trial indices to be plotted; if trials is None, then trials in the category will be plotted based on max_trial_to_plot

    num_trials = 2
    if trials is not None:
        category = trials
        if isinstance(category, int):
            category = np.array([category])
    category = np.array(category)

    category = category[category > num_trials]
    k = 1  # only useful when using_subplots is True
    if category_name == 'disappear_latest' or category_name == 'ignore_sudden_flash':
        num_trials = 1

    num_trial_plotted = 0
    if len(category) > 0:
        with general_utils.initiate_plot(figsize[0], figsize[1], 100):
            if using_subplots:
                fig = plt.figure()
            for currentTrial in category:
                if using_subplots:
                    axes = fig.add_subplot(2, 2, k)
                    additional_kwargs = {'fig': fig,
                                         'axes': axes, 'subplots': True}
                returned_info = plot_a_trial_from_a_category(category_name, currentTrial, num_trials, ff_caught_T_new, PlotTrials_args,
                                                             all_category_kwargs, additional_kwargs=additional_kwargs, images_dir=images_dir)
                if returned_info['whether_plotted'] is True:
                    if using_subplots:
                        k += 1
                        if k == 5:
                            plt.show()

                            return
                    else:
                        plt.show()

                        num_trial_plotted += 1
                        if num_trial_plotted >= max_trial_to_plot:
                            break


def plot_behaviors_in_clusters(points_w_more_than_2_ff, chunk_numbers, monkey_information, ff_dataframe,
                               ff_life_sorted, ff_real_position_sorted, ff_caught_T_new, ff_flash_sorted):
    # for each chunk, finds indices of points where ddw > 0.15
    for chunk in chunk_numbers:
        chunk_df = points_w_more_than_2_ff[points_w_more_than_2_ff['chunk'] == chunk]
        duration_points = [
            chunk_df['point_index'].min(), chunk_df['point_index'].max()]
        duration = [monkey_information['time'][duration_points[0]],
                    monkey_information['time'][duration_points[0]]+10]
        cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & (
            monkey_information['time'] <= duration[1]))[0]
        cum_point_index = np.array(
            monkey_information['point_index'].iloc[cum_pos_index])
        cum_ddw = np.array(
            monkey_information['monkey_ddw'].iloc[cum_pos_index])
        cum_abs_ddw = np.abs(cum_ddw)
        changing_dw_info = pd.DataFrame(
            {'relative_pos_index': np.where(cum_abs_ddw > 0.15)[0]})
        # find the first point of each sequence of consecutive points
        changing_dw_info['group'] = np.append(
            0, (np.diff(changing_dw_info['relative_pos_index']) != 1).cumsum())
        changing_dw_info_short = changing_dw_info.groupby('group').min()
        changing_dw_info_short['relative_pos_index'] = changing_dw_info_short['relative_pos_index'].astype(
            int)
        changing_dw_info_short['point_index'] = cum_point_index[changing_dw_info_short['relative_pos_index']]
        for point_index in changing_dw_info_short['point_index']:
            duration = [monkey_information['time'][point_index] -
                        2, monkey_information['time'][point_index]]
            ff_dataframe_sub = ff_dataframe[ff_dataframe['time'].between(
                duration[0], duration[1], inclusive='both')]
            # Make a polar plot from the monkey's perspective in the duration
            plot_polar.PlotPolar(duration,
                                 monkey_information,
                                 ff_dataframe_sub,
                                 ff_life_sorted,
                                 ff_real_position_sorted,
                                 ff_caught_T_new,
                                 ff_flash_sorted,
                                 rmax=400,
                                 currentTrial=None,
                                 num_trials=None,
                                 show_visible_ff=True,
                                 show_visible_target=True,
                                 show_ff_in_memory=True,
                                 show_target_in_memory=True,
                                 show_alive_ff=True
                                 )


def show_trajectory_func(axes, player, cum_pos_index, cum_mxy_rotated, cum_t, cum_speed, monkey_information,
                         x0, y0, trail_color_var, show_eye_positions, subplots, hitting_arena_edge):
    trail_size = {"agent": 70, "monkey": 2, "combined": 2}
    trail_alpha = {"agent": 1, "monkey": 0.9, "combined": 0.5}
    if subplots == True:
        trail_size = {"agent": 10, "monkey": 5}
    if show_eye_positions:
        axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o',
                     s=trail_size[player], alpha=trail_alpha[player], c=cum_t, cmap='gist_ncar', zorder=3)
    elif trail_color_var == 'speed':  # the color of the path will vary by speed
        axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o',
                     s=trail_size[player], alpha=trail_alpha[player], c=cum_speed, cmap='viridis', zorder=3)
    elif trail_color_var == 'abs_ddw':
        cum_abs_ddw = np.abs(
            np.array(monkey_information['monkey_ddw'].iloc[cum_pos_index]))
        axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o', s=trail_size[player],
                     alpha=trail_alpha[player], c=cum_abs_ddw, cmap='viridis_r', zorder=3)
        # To mark the points where high abs_ddw occur:
        points_to_mark = np.where(cum_abs_ddw > 0.1)[0]
        axes.scatter(cum_mxy_rotated[0, points_to_mark]-x0, cum_mxy_rotated[1, points_to_mark] -
                     y0, marker='*', s=160, alpha=trail_alpha[player], c="orange", zorder=1)
    elif trail_color_var == 'target_visibility':
        axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o',
                     s=trail_size[player], alpha=trail_alpha[player], c="orange", zorder=1)
    elif trail_color_var == 'time':
        axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o', s=trail_size[player],
                     alpha=trail_alpha[player], c=cum_t-min(cum_t), cmap='viridis', zorder=3)
    elif trail_color_var is None:
        if not hitting_arena_edge:
            axes.plot(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0,
                      alpha=trail_alpha[player], c="black", zorder=3, linewidth=2)
        else:
            axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0,
                         marker='o', s=2, alpha=trail_alpha[player], c="black", zorder=3)
    else:  # then trail_color_var will be considered as a color
        if not hitting_arena_edge:
            axes.plot(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0,
                      alpha=trail_alpha[player], c=trail_color_var, zorder=3, linewidth=2)
        else:
            axes.scatter(cum_mxy_rotated[0]-x0, cum_mxy_rotated[1]-y0, marker='o',
                         s=2, alpha=trail_alpha[player], c=trail_color_var, zorder=3)
    return axes


def customize_kwargs_by_category(classic_plot_kwargs, images_dir=None):
    classic_plot_kwargs['images_dir'] = images_dir

    visible_before_last_one_kwargs = classic_plot_kwargs.copy()
    disappear_latest_kwargs = classic_plot_kwargs.copy()
    two_in_a_row_kwargs = classic_plot_kwargs.copy()
    waste_cluster_around_target_kwargs = classic_plot_kwargs.copy()
    try_a_few_times_kwargs = classic_plot_kwargs.copy()
    give_up_after_trying_kwargs = classic_plot_kwargs.copy()
    ignore_sudden_flash_kwargs = classic_plot_kwargs.copy()

    visible_before_last_one_kwargs['show_connect_path_ff_except_targets'] = True
    visible_before_last_one_kwargs['show_path_when_target_visible'] = True
    disappear_latest_kwargs['show_connect_path_ff'] = True
    two_in_a_row_kwargs['show_connect_path_ff_except_targets'] = True
    two_in_a_row_kwargs['show_path_when_target_visible'] = True
    waste_cluster_around_target_kwargs['show_connect_path_ff'] = True
    waste_cluster_around_target_kwargs['trial_to_show_cluster_around_target'] = 'previous'
    try_a_few_times_kwargs['show_connect_path_ff'] = True
    give_up_after_trying_kwargs['show_connect_path_ff'] = True
    ignore_sudden_flash_kwargs['show_connect_path_ff'] = True

    all_category_kwargs = {'visible_before_last_one': visible_before_last_one_kwargs,
                           'disappear_latest': disappear_latest_kwargs,
                           'two_in_a_row': two_in_a_row_kwargs,
                           'waste_cluster_around_target': waste_cluster_around_target_kwargs,
                           'try_a_few_times': try_a_few_times_kwargs,
                           'give_up_after_trying': give_up_after_trying_kwargs,
                           'ignore_sudden_flash': ignore_sudden_flash_kwargs}
    return all_category_kwargs


def connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color, alpha, linewidth, show_dots=True, dot_color="brown"):
    if temp_ff_positions.shape[1] > 0:
        for j in range(temp_ff_positions.shape[1]):
            axes.plot(np.stack([temp_ff_positions[0, j]-x0, temp_monkey_positions[0, j]-x0]),
                      np.stack([temp_ff_positions[1, j]-y0,
                               temp_monkey_positions[1, j]-y0]),
                      '-', alpha=alpha, linewidth=linewidth, c=color)
            show_dots_marker = None
            if show_dots:
                # to mark the connected fireflies as brown circles
                axes.scatter(temp_ff_positions[0, j]-x0, temp_ff_positions[1,
                             j]-y0, alpha=0.15, s=15, color=dot_color, zorder=3)
    return axes


def find_monkey_information_in_the_duration(duration, monkey_information):
    cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & (
        monkey_information['time'] <= duration[1]))[0]
    cum_point_index = np.array(
        monkey_information['point_index'].iloc[cum_pos_index])
    cum_t, cum_angle = np.array(monkey_information['time'].iloc[cum_pos_index]), np.array(
        monkey_information['monkey_angle'].iloc[cum_pos_index])
    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_pos_index]), np.array(
        monkey_information['monkey_y'].iloc[cum_pos_index])
    cum_speed, cum_speeddummy = np.array(monkey_information['monkey_speed'].iloc[cum_pos_index]), np.array(
        monkey_information['monkey_speeddummy'].iloc[cum_pos_index])
    return cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy


def find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted, rotation_matrix=None):
    alive_ff_indices = np.array([ff_index for ff_index, life in enumerate(
        ff_life_sorted) if (life[-1] >= duration[0]) and (life[0] <= duration[1])])
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]

    alive_ff_position_rotated = np.stack(
        (alive_ff_positions.T[0], alive_ff_positions.T[1]))
    if (rotation_matrix is not None) & (alive_ff_position_rotated.shape[1] > 0):
        alive_ff_position_rotated = np.matmul(
            rotation_matrix, alive_ff_position_rotated)

    return alive_ff_indices, alive_ff_position_rotated


def find_believed_target_positions(ff_believed_position_sorted, currentTrial, num_trials, rotation_matrix=None):
    believed_target_positions = ff_believed_position_sorted[currentTrial -
                                                            num_trials + 1:currentTrial + 1]
    believed_target_positions_rotated = np.stack(
        (believed_target_positions.T[0], believed_target_positions.T[1]))
    if (rotation_matrix is not None) & (believed_target_positions_rotated.shape[1] > 0):
        believed_target_positions_rotated = np.matmul(
            rotation_matrix, believed_target_positions_rotated)
    return believed_target_positions_rotated


def find_stops_for_plotting(cum_mx, cum_my, cum_speeddummy, rotation_matrix=None):
    zerospeed_index = np.where(cum_speeddummy == 0)
    zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
    zerospeed_rotated = np.stack((zerospeedx, zerospeedy))
    if (rotation_matrix is not None) & (zerospeed_rotated.shape[1] > 0):
        zerospeed_rotated = np.matmul(rotation_matrix, zerospeed_rotated)

    return zerospeed_rotated


def find_rotation_matrix(cum_mx, cum_my, also_return_angle=False):
    # Find the angle from the starting point to the ending point
    theta = np.arctan2(cum_my[-1]-cum_my[0], cum_mx[-1]-cum_mx[0])
    # let 0 be pointing to the north
    theta = theta - pi/2
    # since we want to rotated back this angle, we use its negative
    theta = -theta
    c, s = np.cos(theta), np.sin(theta)
    # Rotation matrix
    R = np.array(((c, -s), (s, c)))
    if also_return_angle:
        return R, theta
    else:
        return R


def find_triangles_to_show_monkey_angles(cum_mx, cum_my, cum_angle, rotation_matrix=None):
    left_end_x = cum_mx + 30 * np.cos(cum_angle + 2*pi/9)
    left_end_y = cum_my + 30 * np.sin(cum_angle + 2*pi/9)
    right_end_x = cum_mx + 30 * np.cos(cum_angle - 2*pi/9)
    right_end_y = cum_my + 30 * np.sin(cum_angle - 2*pi/9)

    left_end_xy = np.stack((left_end_x, left_end_y), axis=1).T
    right_end_xy = np.stack((right_end_x, right_end_y), axis=1).T

    if (rotation_matrix is not None) & (left_end_xy.shape[1] > 0):
        left_end_xy = np.matmul(rotation_matrix, left_end_xy)
        right_end_xy = np.matmul(rotation_matrix, right_end_xy)

    return left_end_xy, right_end_xy


def find_path_when_ff_visible(ff_index, ff_dataframe_in_duration, cum_point_index, visible_distance, rotation_matrix=None):
    temp_df = ff_dataframe_in_duration
    temp_df = temp_df.loc[(temp_df['ff_index'] == ff_index) & (
        temp_df['visible'] == 1) & (temp_df['ff_distance'] <= visible_distance)]
    temp_df = temp_df[(temp_df['point_index'] >= np.min(cum_point_index)) & (
        temp_df['point_index'] <= np.max(cum_point_index))]
    ff_visible_path_rotated = np.array(temp_df[['monkey_x', 'monkey_y']]).T
    if (rotation_matrix is not None) & (ff_visible_path_rotated.shape[1] > 0):
        ff_visible_path_rotated = np.matmul(
            rotation_matrix, ff_visible_path_rotated)
    return ff_visible_path_rotated


def find_lines_to_connect_path_ff(ff_dataframe_in_duration, target_indices, rotation_matrix=None, target_excluded=False):
    temp_df = ff_dataframe_in_duration

    if target_excluded:
        # if the player is monkey, then the following code is used to avoid the lines between the monkey's position and the target since the lines might obscure the path
        temp_df = temp_df.loc[~temp_df['ff_index'].isin(target_indices)]
    temp_array = temp_df[['ff_x', 'ff_y', 'monkey_x', 'monkey_y']].to_numpy()

    temp_ff_positions_rotated = temp_array[:, :2].T
    temp_monkey_positions_rotated = temp_array[:, 2:4].T
    if (rotation_matrix is not None) & (temp_ff_positions_rotated.shape[1] > 0):
        temp_ff_positions_rotated = np.matmul(
            rotation_matrix, temp_ff_positions_rotated)
        temp_monkey_positions_rotated = np.matmul(
            rotation_matrix, temp_monkey_positions_rotated)

    return temp_ff_positions_rotated, temp_monkey_positions_rotated


def plot_lines_to_connect_path_and_ff(axes, ff_info, rotation_matrix, x0, y0, linewidth, alpha, vary_color_for_connecting_path_ff=False, line_color="#a940f5", show_connect_path_ff_except_targets=False,
                                      target_indices=None, show_points_when_ff_stop_being_visible=False, show_points_when_ff_start_being_visible=False, legend_markers=[], legend_names=[]):
    # if vary_color_for_connecting_path_ff is True, then line_color is useless
    # if show_connect_path_ff_except_targets is False, then target_indices is useless

    legend_flag = False
    temp_ff_positions = np.array([])
    temp_monkey_positions = np.array([])
    if vary_color_for_connecting_path_ff:
        unique_ff_index = ff_info.ff_index.unique()
        varying_colors = np.concatenate(
            [sns.color_palette("tab10", 3), sns.color_palette("tab10", 14)[4:]], axis=0)
        # varying_colors = sns.color_palette("tab10", 10)
        # varying_colors = np.delete(varying_colors, 2, 0)   # take out the 3rd color (green from varying_colors)
        for i in range(len(unique_ff_index)):
            ff_index = unique_ff_index[i]
            # take out a color from Set2
            color = np.append(varying_colors[i % 9], 0.5)
            temp_df = ff_info[ff_info['ff_index'] == ff_index]
            temp_ff_positions, temp_monkey_positions = find_lines_to_connect_path_ff(
                temp_df, target_indices, rotation_matrix=rotation_matrix, target_excluded=show_connect_path_ff_except_targets)
            if temp_monkey_positions.shape[1] > 0:
                axes = connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color=color, alpha=alpha,
                                                linewidth=linewidth, show_dots=True, dot_color="brown")

                if show_points_when_ff_start_being_visible or show_points_when_ff_stop_being_visible:
                    if show_points_when_ff_start_being_visible:
                        marker = axes.scatter(
                            temp_monkey_positions[0, 0]-x0, temp_monkey_positions[1, 0]-y0, alpha=0.7, marker="X", s=80, color=color, zorder=4)
                        marker_name = 'Points when fireflies start being visible'
                        # x_point, y_point = temp_monkey_positions[0, 0]-x0, temp_monkey_positions[1, 0]-y0
                        # marker = axes.plot([x_point-30, x_point+30], [y_point, y_point], ls=':', color=color, lw=2.5)
                        # marker_name = 'Points when fireflies stop being visible'
                    if show_points_when_ff_stop_being_visible:
                        marker = axes.scatter(
                            temp_monkey_positions[0, -1]-x0, temp_monkey_positions[1, -1]-y0, alpha=0.7, marker="X", s=80, color=color, zorder=4)
                        marker_name = 'Points when fireflies stop being visible'
                        # x_point, y_point = temp_monkey_positions[0, -1]-x0, temp_monkey_positions[1, -1]-y0
                        # marker = axes.plot([x_point-30, x_point+30], [y_point, y_point], color=color, lw=2.5)
                        # marker_name = 'Points when fireflies stop being visible'
                    if legend_flag == False:
                        if show_points_when_ff_start_being_visible and show_points_when_ff_stop_being_visible:
                            marker_name = 'Points when fireflies start or stop being visible'
                        legend_markers.append(marker)
                        legend_names.append(marker_name)
                        legend_flag = True    # because we only have to append this legend once

    else:
        temp_ff_positions, temp_monkey_positions = find_lines_to_connect_path_ff(
            ff_info, target_indices, rotation_matrix=rotation_matrix, target_excluded=show_connect_path_ff_except_targets)
        axes = connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color=line_color, alpha=alpha,
                                        linewidth=linewidth, show_dots=True, dot_color="brown")

    return axes, legend_markers, legend_names, temp_ff_positions, temp_monkey_positions


def find_dict_of_perpendicular_lines_to_monkey_trajectory_at_certain_points(all_point_index, all_breaking_points, monkey_information, rotation_matrix):
    all_starting_points = np.insert(
        all_point_index[all_breaking_points], 0, all_point_index[0])
    all_ending_points = np.append(
        all_point_index[all_breaking_points-1], all_point_index[-1])

    starting_left_xy_rotated, starting_right_xy_rotated = find_perpendicular_lines_to_monkey_trajectory_at_certain_points(
        all_starting_points, monkey_information, rotation_matrix)
    ending_left_xy_rotated, ending_right_xy_rotated = find_perpendicular_lines_to_monkey_trajectory_at_certain_points(
        all_ending_points, monkey_information, rotation_matrix)
    perp_dict = {'starting_left_xy_rotated': starting_left_xy_rotated,
                 'starting_right_xy_rotated': starting_right_xy_rotated,
                 'ending_left_xy_rotated': ending_left_xy_rotated,
                 'ending_right_xy_rotated': ending_right_xy_rotated}
    return perp_dict


def find_perpendicular_lines_to_monkey_trajectory_at_certain_points(certain_points, monkey_information, rotation_matrix, line_half_length=50):
    # find monkey xy and angles at all_points and all_ending_points
    monkey_x = monkey_information.loc[certain_points, 'monkey_x'].values
    monkey_y = monkey_information.loc[certain_points, 'monkey_y'].values
    monkey_angle = monkey_information.loc[certain_points,
                                          'monkey_angle'].values

    # find lines perpendicular to monkey_angles at these points
    angles_to_left = monkey_angle + pi/2
    angles_to_right = monkey_angle - pi/2
    left_x = monkey_x + line_half_length * np.cos(angles_to_left)
    left_y = monkey_y + line_half_length * np.sin(angles_to_left)
    right_x = monkey_x + line_half_length * np.cos(angles_to_right)
    right_y = monkey_y + line_half_length * np.sin(angles_to_right)

    left_xy_rotated = np.matmul(rotation_matrix, np.vstack((left_x, left_y)))
    right_xy_rotated = np.matmul(
        rotation_matrix, np.vstack((right_x, right_y)))

    return left_xy_rotated, right_xy_rotated


def find_one_pair_of_perpendicular_lines(perp_dict, j, x0, y0):
    one_perp_dict = {'starting_left_x': perp_dict['starting_left_xy_rotated'][0, j]-x0,
                     'starting_left_y': perp_dict['starting_left_xy_rotated'][1, j]-y0,
                     'starting_right_x': perp_dict['starting_right_xy_rotated'][0, j]-x0,
                     'starting_right_y': perp_dict['starting_right_xy_rotated'][1, j]-y0,
                     'ending_left_x': perp_dict['ending_left_xy_rotated'][0, j]-x0,
                     'ending_left_y': perp_dict['ending_left_xy_rotated'][1, j]-y0,
                     'ending_right_x': perp_dict['ending_right_xy_rotated'][0, j]-x0,
                     'ending_right_y': perp_dict['ending_right_xy_rotated'][1, j]-y0}

    return one_perp_dict


def plot_horizontal_lines_to_show_ff_visible_segments(axes, ff_info, monkey_information, rotation_matrix, x0, y0, legend_markers=[], legend_names=[],
                                                      how_to_show_ff='square', unique_ff_indices=None):
    """
    This function plots horizontal lines to show visible segments of fireflies (ff).
    It also shows the ff position as a square or a circle.

    Parameters:
    axes (object): The axes object to plot on.
    ff_info (DataFrame): The DataFrame containing firefly information.
    monkey_information (dict): The dictionary containing monkey information.
    rotation_matrix (ndarray): The rotation matrix to use.
    x0, y0 (float): The coordinates of the origin.
    legend_markers (list): The list of legend markers. Default is an empty list.
    legend_names (list): The list of legend names. Default is an empty list.
    how_to_show_ff (str): How to show the firefly. Options are 'square' or 'circle'. Default is 'square'.
    unique_ff_indices (list): The list of unique firefly indices. If None, it will be set to all unique indices in ff_info. Default is None.

    Returns:
    axes (object): The axes object with the plot.
    legend_markers (list): The updated list of legend markers.
    legend_names (list): The updated list of legend names.
    show_visible_segments_of_ff_dict (dict): A dictionary with firefly indices as keys and colors as values.
    """

    # Define threshold for separating visible intervals
    point_index_gap_threshold_to_sep_vis_intervals = 12

    # Set unique_ff_indices to all unique indices in ff_info if not provided
    unique_ff_indices = ff_info.ff_index.unique(
    ) if unique_ff_indices is None else np.array(unique_ff_indices)

    # Define color palette, avoiding red
    varying_colors = np.concatenate(
        [sns.color_palette("tab10", 3), sns.color_palette("tab10", 14)[4:]], axis=0)

    # Initialize dictionary to store visible segments of fireflies
    show_visible_segments_of_ff_dict = dict()

    # Iterate over unique firefly indices
    for i, ff_index in enumerate(unique_ff_indices):
        # Define color for current firefly
        color = np.append(varying_colors[i % 9], 0.5)

        # Extract and sort data for current firefly
        temp_df = ff_info[ff_info['ff_index'] == ff_index].copy().sort_values(by=[
            'point_index'])

        if len(temp_df) == 0:
            continue
        # rotated firefly position
        ff_position_rotated = np.matmul(
            rotation_matrix, temp_df[['ff_x', 'ff_y']].drop_duplicates().values.T)

        # Find breaking points of visible segments
        all_point_index = temp_df.point_index.values
        all_breaking_points = np.where(np.diff(
            all_point_index) >= point_index_gap_threshold_to_sep_vis_intervals)[0] + 1

        # Find positions of the ends of the perpendicular lines at starting and ending points
        perp_dict = find_dict_of_perpendicular_lines_to_monkey_trajectory_at_certain_points(
            all_point_index, all_breaking_points, monkey_information, rotation_matrix)

        # Show firefly position
        if how_to_show_ff == 'square':
            axes.scatter(ff_position_rotated[0]-x0, ff_position_rotated[1]-y0,
                         marker='s', alpha=0.75, s=170, color=color, zorder=3)
        elif how_to_show_ff == 'circle':
            circle = plt.Circle((ff_position_rotated[0]-x0, ff_position_rotated[1]-y0),
                                25, facecolor=color, edgecolor=None, alpha=0.75, zorder=1)
            axes.add_patch(circle)

        # Find and plot beginning and end of each visible segment
        for j in range(len(all_breaking_points)+1):
            one_perp_dict = find_one_pair_of_perpendicular_lines(
                perp_dict, j, x0, y0)

            # Plot points when firefly starts being visible
            marker1 = axes.plot([one_perp_dict['starting_left_x'], one_perp_dict['starting_right_x']],
                                [one_perp_dict['starting_left_y'], one_perp_dict['starting_right_y']], color=color, lw=2.5)

            # Plot points when firefly stops being visible
            marker2 = axes.plot([one_perp_dict['ending_left_x'], one_perp_dict['ending_right_x']],
                                [one_perp_dict['ending_left_y'], one_perp_dict['ending_right_y']], ls=':', color=color, lw=2.5)

            # Store color of visible segments for current firefly
            show_visible_segments_of_ff_dict[ff_index] = color

            # Update legend markers and names if not provided
            if not legend_markers:
                legend_markers.extend([marker1, marker2])
                legend_names.extend(
                    ['Points when fireflies start being visible', 'Points when fireflies stop being visible'])

    return axes, legend_markers, legend_names, show_visible_segments_of_ff_dict


def find_ff_in_cluster(cluster_dataframe_point, ff_real_position_sorted, currentTrial, rotation_matrix=None):
    # Find the indices of ffs in the cluster
    cluster_indices = cluster_dataframe_point[cluster_dataframe_point['target_index']
                                              == currentTrial].ff_index
    cluster_indices = np.unique(cluster_indices.to_numpy())
    cluster_ff_positions = ff_real_position_sorted[cluster_indices]
    cluster_ff_rotated = np.stack(
        (cluster_ff_positions.T[0], cluster_ff_positions.T[1]))
    if (rotation_matrix is not None) & (cluster_ff_rotated.shape[1] > 0):
        cluster_ff_rotated = np.matmul(rotation_matrix, cluster_ff_rotated)
    return cluster_ff_rotated


def find_ff_in_cluster_around_target(cluster_around_target_indices, ff_real_position_sorted, currentTrial, rotation_matrix=None):
    cluster_ff_indices = cluster_around_target_indices[currentTrial]
    cluster_ff_positions = ff_real_position_sorted[cluster_ff_indices]
    cluster_around_target_rotated = np.stack(
        (cluster_ff_positions.T[0], cluster_ff_positions.T[1]))
    if (rotation_matrix is not None) & (cluster_around_target_rotated.shape[1] > 0):
        cluster_around_target_rotated = np.matmul(
            rotation_matrix, cluster_around_target_rotated)
    return cluster_ff_indices, cluster_around_target_rotated


def find_path_when_ff_in_cluster_visible(ff_dataframe_in_duration, ff_index, rotation_matrix=None):
    temp_df = ff_dataframe_in_duration
    temp_df = temp_df.loc[(temp_df['ff_index'] == ff_index)
                          & (temp_df['visible'] == 1)]
    monkey_xy_rotated = np.array(temp_df[['monkey_x', 'monkey_y']]).T
    ff_position_rotated = np.array(temp_df[['ff_x', 'ff_y']]).T
    if (rotation_matrix is not None) & (monkey_xy_rotated.shape[1] > 0):
        monkey_xy_rotated = np.matmul(rotation_matrix, monkey_xy_rotated)
        ff_position_rotated = np.matmul(rotation_matrix, ff_position_rotated)
    return monkey_xy_rotated, ff_position_rotated


def visualize_monkey_angles_using_triangles(axes, cum_mxy_rotated, left_end_xy_rotated, right_end_xy_rotated, linewidth=0.5):
    for point in range(cum_mxy_rotated.shape[1]):
        middle = cum_mxy_rotated[:, point]
        left_end = left_end_xy_rotated[:, point]
        right_end = right_end_xy_rotated[:, point]
        # Only show the left side of the triangle
        axes.plot(np.array([middle[0], left_end[0]]), np.array(
            [middle[1], left_end[1]]), linewidth=linewidth)
        axes.plot(np.array([middle[0], right_end[0]]), np.array(
            [middle[1], right_end[1]]), linewidth=linewidth)
    return axes


def plot_scale_bar(axes):
    scale = ScaleBar(dx=1, units='cm', length_fraction=0.2, fixed_value=100,
                     location='upper left', label_loc='left', scale_loc='bottom')
    axes.add_artist(scale)
    return axes


def plot_colorbar_for_trials(fig, axes, trail_color_var, duration, show_eye_positions=False, show_eye_positions_on_the_right=False, max_value=None):

    width = {True: 0.025, False: 0.05}
    bottom = {True: 0.47, False: 0.4}

    cmap = (matplotlib.colors.ListedColormap(['black', 'red']))
    bounds = [0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    labels = np.array(["No Reward", "Reward"])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        # left, bottom, width, height
        cax=fig.add_axes(
            [0.95, 0.12, width[show_eye_positions_on_the_right], 0.2]),
        ticks=[1, 2],
        spacing='uniform',
        orientation='vertical',
    )

    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(size=0, color='white')
    cbar.ax.set_title('Stopping Points', ha='left', y=1.06)

    # Then make the colorbar to show the meaning of color of the monkey/agent's path
    if show_eye_positions or (trail_color_var == 'speed') or (trail_color_var == 'abs_ddw'):
        cmap = cm.viridis
        if show_eye_positions:
            vmax = duration[1]-duration[0]
            title = 'Time (s)'
        elif trail_color_var == 'speed':
            vmax = 200
            title = 'Speed(cm/s)'
        else:
            cmap = cm.viridis_r
            vmax = max_value
            title = 'Angular cceleration (radians/s^2)'
        norm2 = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=fig.add_axes(
            [0.95, bottom[show_eye_positions_on_the_right], width[show_eye_positions_on_the_right], 0.43]), orientation='vertical')
        cbar2.outline.set_visible(False)
        cbar2.ax.tick_params(axis='y', color='lightgrey',
                             direction="in", right=True, length=5, width=1.5)
        cbar2.ax.set_title(title, ha='left', y=1.04)

    elif trail_color_var == "target_visibility":
        cmap = (matplotlib.colors.ListedColormap(['green', 'orange']))
        bounds = [0.5, 1.5, 2.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        labels = np.array(["Top Target Visible", "Top Target Not Visible"])
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
            # left, bottom, width, height
            cax=fig.add_axes([0.95, 0.5, 0.05, 0.2]),
            ticks=[1, 2],
            spacing='uniform',
            orientation='vertical',
        )
        cbar.ax.set_yticklabels(labels)
        cbar.ax.tick_params(size=0, color='white')
        cbar.ax.set_title('Path Color', ha='left', y=1.06)
    return fig, axes


def find_xy_min_max_for_plots(cum_mxy_rotated, x0, y0, temp_ff_positions=None):
    mx_min, mx_max = min(cum_mxy_rotated[0])-x0, max(cum_mxy_rotated[0])-x0
    my_min, my_max = min(cum_mxy_rotated[1])-y0, max(cum_mxy_rotated[1])-y0
    if temp_ff_positions is not None:
        if temp_ff_positions.shape[1] > 0:
            mx_min, mx_max = min(mx_min, min(
                temp_ff_positions[0])-x0), max(mx_max, max(temp_ff_positions[0])-x0)
            my_min, my_max = min(my_min, min(
                temp_ff_positions[1])-y0), max(my_max, max(temp_ff_positions[1])-y0)
    return mx_min, mx_max, my_min, my_max


def set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, minimal_margin=50, max_margin=200, zoom_in=False):
    bigger_width = max(mx_max - mx_min, my_max - my_min)

    # if the full arena is shown, then we'll decrease the margin
    if (mx_max - mx_min > 1800) and (my_max - my_min > 1800):
        minimal_margin = 50
        margin = 50
    else:
        margin = max(bigger_width/20, minimal_margin)
        margin = min(margin, max_margin)

    xmiddle, ymiddle = (mx_min + mx_max) / 2, (my_min + my_max) / 2
    xmin, xmax = xmiddle - bigger_width/2, xmiddle + bigger_width/2
    ymin, ymax = ymiddle - bigger_width/2, ymiddle + bigger_width/2

    if zoom_in is True:
        if minimal_margin <= 40:
            axes.set_xlim((xmin - 40, xmax + 40))
            axes.set_ylim((ymin - 20, ymax + 60))
        else:
            axes.set_xlim((xmin - minimal_margin, xmax + minimal_margin))
            axes.set_ylim((ymin - minimal_margin*2/3,
                          ymax + minimal_margin*4/3))
    else:
        axes.set_xlim((xmin - margin, xmax + margin))
        axes.set_ylim((ymin - margin*2/3, ymax + margin*4/3))
    return axes


def readjust_xy_limits_for_axes(axes, cum_mxy_rotated_1, cum_mxy_rotated_2, shown_ff_indices_1, shown_ff_indices_2, R, ff_real_position_sorted, minimal_margin=50):
    cum_mxy_rotated_all = np.concatenate(
        (cum_mxy_rotated_1, cum_mxy_rotated_2), axis=1)
    shown_ff_positions_rotated_1 = ff_real_position_sorted[shown_ff_indices_1].T
    shown_ff_positions_rotated_1 = np.matmul(R, shown_ff_positions_rotated_1)
    shown_ff_positions_rotated_2 = ff_real_position_sorted[shown_ff_indices_2].T
    shown_ff_positions_rotated_2 = np.matmul(R, shown_ff_positions_rotated_2)
    shown_ff_positions_rotated_all = np.concatenate(
        (shown_ff_positions_rotated_1, shown_ff_positions_rotated_2), axis=1)
    mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(
        cum_mxy_rotated_all, x0=0, y0=0, temp_ff_positions=shown_ff_positions_rotated_all)
    axes = set_xy_limits_for_axes(
        axes, mx_min, mx_max, my_min, my_max, minimal_margin=minimal_margin)
    return axes


def save_image(filename, images_dir):
    CHECK_FOLDER = os.path.isdir(images_dir)
    if not CHECK_FOLDER:
        os.makedirs(images_dir)
    plt.savefig(f"{images_dir}/{filename}.png")


def plot_ff_distribution_in_arena(ff_real_position_sorted, ff_life_sorted, ff_caught_T_new, images_dir=None):
    # divide total time length (or valid point_index) into 9 parts
    # since some fireflies might not be flashing at the beginnin or end of the period, their "life" information at the beginning
    # and end are not complete. Thus, we chop off the beginning and the end of the period by 50s.
    max_time = ff_caught_T_new[-1] - 50
    min_time = ff_caught_T_new[0] + 50
    time_intervals = np.linspace(min_time, max_time, 9)
    num_ff_for_each_plot = []

    # plot a 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(9):
        time_point = time_intervals[i]
        duration = [time_point-0.1, time_point+0.1]
        # for each subplot, plot the ff distribution at a time point
        alive_ff_indices, alive_ff_position = find_alive_ff(
            duration, ff_life_sorted, ff_real_position_sorted)
        # plot alive_ff_position
        axes[i].scatter(alive_ff_position[0], alive_ff_position[1],
                        marker='o', s=10, color="grey", zorder=2)
        num_ff_for_each_plot.append(len(alive_ff_indices))
    plt.show()

    print(num_ff_for_each_plot)


def get_overall_lim(axes, axes2):
    """
    Get the x-limits and y-limits of the plots based on both the monkey data and the agent data


    Parameters
    ----------
    axes: obj
        axes for one plot (e.g. for the monkey data)
    axes2: obj
        axes for another plot (e.g. for the agent data)


    Returns
    -------
    overall_xmin: num
        the minimum value of the x-axis that will be shared by both plots
    overall_xmax: num
        the maximum value of the x-axis that will be shared by both plots     
    overall_ymin: num
        the minimum value of the y-axis that will be shared by both plots
    overall_ymax: num
        the maximum value of the y-axis that will be shared by both plots

    """

    monkey_xmin, monkey_xmax = axes.get_xlim()
    monkey_ymin, monkey_ymax = axes.get_ylim()
    agent_xmin, agent_xmax = axes2.get_xlim()
    agent_ymin, agent_ymax = axes2.get_ylim()

    overall_xmin = min(monkey_xmin, agent_xmin)
    overall_xmax = max(monkey_xmax, agent_xmax)
    overall_ymin = min(monkey_ymin, agent_ymin)
    overall_ymax = max(monkey_ymax, agent_ymax)
    return overall_xmin, overall_xmax, overall_ymin, overall_ymax


def update_plot_limits(xmin, ymin, xmax, ymax, cum_mxy_rotated):
    """
    Update the limits of the plot; usually used when multiple trajectories are plotted on the same plot

    Parameters
    ----------
    xmin: num
        the minimum of the x-axis
    xmax: num
        the maximum of the x-axis
    ymin: num
        the minimum of the y-axis
    ymax: num
        the maximum of the y-axis
    cum_mxy_rotated: array, with shape (2, n)
        contains the x, y coordinates of the monkey's positions on the trajectory after rotation of the plot.

    Returns
    ----------
    xmin: num
        the updated minimum of the x-axis
    xmax: num
        the updated maximum of the x-axis
    ymin: num
        the updatedminimum of the y-axis
    ymax: num
        the updated maximum of the y-axis

    """

    x0, y0 = cum_mxy_rotated[0][0], cum_mxy_rotated[1][0]
    temp_xmin, temp_xmax = np.min(
        cum_mxy_rotated[0])-x0, np.max(cum_mxy_rotated[0])-x0
    temp_ymin, temp_ymax = np.min(
        cum_mxy_rotated[1])-y0, np.max(cum_mxy_rotated[1])-y0
    xmin, xmax = min(xmin, temp_xmin), max(xmax, temp_xmax)
    ymin, ymax = min(ymin, temp_ymin), max(ymax, temp_ymax)
    return xmin, ymin, xmax, ymax


def set_polar_background(ax, rmax, color_visible_area_in_background=True):
    """
    Decorate the canvas to set up the background for a polar plot


    Parameters
    ----------
    ax: obj
        the matplotlib axes object
    rmax: num
        radius of the polar plot

    Returns
    -------
    ax: obj
        the matplotlib axes object
    """

    ax.set_theta_zero_location("N")
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(275)
    # make r labels (the radius) to be integers for every 100 cm
    ax.set_rticks(range(100, rmax+1, 100))
    ax.set_xticks(ax.get_xticks())  # This is to prevent a warning
    # Draw the boundary of the monkey's vision (use width=np.pi*4/9 for 40 degrees of vision)
    if color_visible_area_in_background:
        ax.bar(0, rmax, width=np.pi*4/9, bottom=0.0, color="grey", alpha=0.1)
    return ax


def set_polar_background_for_plotting(ax, rmax, color_visible_area_in_background=True):
    """
    Set up certain parameters for plotting in the polar coordinates

    Parameters
    ----------
    ax: obj
        a matplotlib axes object 
    rmax: numeric
        the radius of the polar plot

    Returns
    -------
    ax: obj
        a matplotlib axes object 
    """
    if rmax < 150:
        ax.set_rticks(range(25, rmax+1, 25))
    ax.set_rlabel_position(292.5)
    ax = set_polar_background(ax, rmax, color_visible_area_in_background)
    labels = list(ax.get_xticks())
    labels[5], labels[6], labels[7] = -labels[3], -labels[2], -labels[1]
    labels_in_degrees = [str(int(math.degrees(label))) + chr(176)
                         for label in labels]
    ax.set_xticklabels(labels_in_degrees)
    return ax


def set_polar_background_for_animation(ax, rmax, color_visible_area_in_background=True):
    """
    Set up certain parameters for each frame for animation in the polar coordinates

    Parameters
    ----------
    ax: obj
        a matplotlib axes object 
    rmax: numeric
        the radius of the polar plot

    Returns
    -------
    ax: obj
        a matplotlib axes object 
    """
    ax = set_polar_background(ax, rmax, color_visible_area_in_background)
    ax.set_thetamin(-45)
    ax.set_thetamax(45)
    labels_in_degrees = [str(int(math.degrees(label))) + chr(176)
                         for label in list(ax.get_xticks())]
    ax.set_xticklabels(labels_in_degrees)
    return ax


def find_ff_distance_and_angles(ff_index, duration, ff_real_position_sorted, monkey_information, ff_radius=10):
    """
    Given the index of a firefly and a duration, find the corresponding distances (to the monkey/agent) and angles (to the monkey/agent)

    Parameters
    ----------
    ff_index: num
        index of the firefly 
    duration: list
        containing the start time and the end time
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies 
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_radius: num
        the reward boundary of each firefly


    Returns
    ----------
    ff_distance_and_angles: pd.Dataframe
        containing the distances of angles (to the boundary or to the center) of the fireflies in the duration
    """
    # Find the indices in monkey information:
    cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = find_monkey_information_in_the_duration(
        duration, monkey_information)

    distances_to_ff = LA.norm(
        np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[ff_index], axis=1)
    angles_to_ff = np.arctan2(
        ff_real_position_sorted[ff_index, 1]-cum_my, ff_real_position_sorted[ff_index, 0]-cum_mx)-cum_angle
    angles_to_ff = np.remainder(angles_to_ff, 2*pi)
    angles_to_ff[angles_to_ff > pi] = angles_to_ff[angles_to_ff > pi] - 2*pi
    # Adjust the angles according to the reward boundary
    angles_to_boundaries = np.absolute(
        angles_to_ff)-np.abs(np.arcsin(np.divide(ff_radius, np.maximum(distances_to_ff, ff_radius))))
    angles_to_boundaries = np.sign(
        angles_to_ff) * np.clip(angles_to_boundaries, 0, pi)

    ff_distance_and_angles = {}
    ff_distance_and_angles['ff_distance'] = distances_to_ff
    ff_distance_and_angles['ff_angle'] = angles_to_ff
    ff_distance_and_angles['ff_angle_boundary'] = angles_to_boundaries
    ff_distance_and_angles['point_index'] = cum_point_index
    ff_distance_and_angles = pd.DataFrame.from_dict(ff_distance_and_angles)
    return ff_distance_and_angles


def add_legend_for_polar_plot(ax,
                              show_visible_ff,
                              show_ff_in_memory,
                              show_alive_ff,
                              show_visible_target,
                              show_target_in_memory,
                              show_target_throughout_duration,
                              colors_show_overall_time):

    # colors = ['green', 'red']
    # labels = ['Captured firefly', 'Other fireflies']

    colors = []
    labels = []
    if show_visible_target or show_target_in_memory or show_target_throughout_duration:
        colors.append('green')
        labels.append('Target firefly')
    if show_visible_ff or show_ff_in_memory:
        colors.append('red')
        labels.append('Non-target fireflies')
    # if colors_show_overall_time:
    #     colors.append('blue')
    #     labels.append('Target firefly in memory')
    #     colors.append('purple')
    #     labels.append('Non-target fireflies in memory')
    if show_alive_ff:
        colors.append('grey')
        labels.append('Alive but invisible fireflies')

    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted')
             for c in colors]
    ax.legend(lines, labels, loc='lower right')
    return ax


def add_colorbar_for_polar_plot(fig, duration, show_ff_in_memory, show_target_in_memory, ff_colormap, target_colormap, colors_show_overall_time, show_all_positions_of_all_fireflies):

    if colors_show_overall_time:
        vmax = duration[1]-duration[0]
        title = 'Time into the Past (s)'
        # cbar_xticks = [0, vmax]
        # cbar_labels = ['Least recent', 'Most recent']
    else:
        vmax = 100  # the maximum value of memory is 100
        title = 'Time since firefly visible'
        cbar_xticks = [0, 20, 40, 60, 80, 100]
        cbar_labels = ['1.67s since visible', '1.33s since visible',
                       '1s since visible', '0.67s since visible', '0.33s since visible', 'Visible']
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    plotted_colorbar_for_ff = False
    if show_ff_in_memory or colors_show_overall_time or show_all_positions_of_all_fireflies:
        plotted_colorbar_for_ff = True
        # [left, bottom, width, height]
        cax = fig.add_axes([0.95, 0.05, 0.05, 0.4])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(
            ff_colormap)), cax=cax, orientation='vertical')
        cbar.ax.tick_params(axis='y', color='lightgrey',
                            direction="in", right=True, length=5, width=1.5)
        cbar.ax.set_title(title, ha='left', y=1.04)
        if ff_colormap == 'viridis':
            cbar.outline.set_visible(False)
        if not colors_show_overall_time:
            cbar.ax.set_yticks(cbar_xticks)  # This is to prevent a warning
            cbar.ax.set_yticklabels(cbar_labels)

    # adding a second colorbar if needed
    if show_target_in_memory or colors_show_overall_time or show_all_positions_of_all_fireflies:
        if (ff_colormap == target_colormap) & plotted_colorbar_for_ff:
            # If the colormap for ff and target are the same, then there's no need to plot another colorbar
            pass
        else:
            cax2 = fig.add_axes([0.88, 0.05, 0.05, 0.4])
            cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(
                target_colormap)), cax=cax2, orientation='vertical')
            cbar2.ax.set_yticks([])
            if target_colormap == 'viridis':
                cbar2.outline.set_visible(False)
            if not (plotted_colorbar_for_ff):
                # Put labels at the right side of the colorbar for the target since there's no colorbar for the non-target fireflies
                cbar2.ax.tick_params(
                    axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
                cbar2.ax.set_title(title, ha='left', y=1.04)
                if not colors_show_overall_time:
                    # This is to prevent a warning
                    cbar2.ax.set_yticks(cbar_xticks)
                    cbar2.ax.set_yticklabels(cbar_labels)

    return fig


def plot_change_in_ff_angle(ff_dataframe, trial_numbers, var_of_interest="abs_ffangle_decreasing"):
    sns.set_style(style="white")
    var = var_of_interest
    var_str = var + "_str"
    ff_dataframe[var_str] = "No Change"
    ff_dataframe.loc[ff_dataframe[var] > 0, var_str] = "Yes"
    ff_dataframe.loc[ff_dataframe[var] < 0, var_str] = "No"
    hue_order = ['Yes', 'No', 'No Change']
    plt.rcdefaults()
    for currentTrial in trial_numbers:
        info_for_currentTrial = ff_dataframe[ff_dataframe['target_index']
                                             == currentTrial]

        sns.stripplot(data=info_for_currentTrial, x="ff_index_string", y="time", hue_order=hue_order,
                      hue=var_str, jitter=False, palette="deep")

        # Mark the xticklabel for the target red
        which_ff_is_target = np.where(
            info_for_currentTrial["ff_index"].unique() == currentTrial)[0]
        if len(which_ff_is_target) > 0:
            plt.gca().get_xticklabels()[which_ff_is_target[0]].set_color('red')

        plt.xlabel("Firefly index")
        plt.ylabel("Time (s)")
        if var_of_interest == "abs_ffangle_decreasing":
            plt.title("Whether Absolute Firefly's Angle is Decreasing")
            plt.legend(title="Whether decreasing")
        if var_of_interest == "abs_ffangle_boundary_decreasing":
            plt.title(
                "Whether Absolute Firefly's Angle to Boundary is Decreasing")
            plt.legend(title="Whether decreasing")
        if var_of_interest == "dw_same_sign_as_ffangle":
            plt.title("Whether Angular V is Same Direction as fireflies")
            plt.legend(title="Whether same direction")
        if var_of_interest == "dw_same_sign_as_ffangle_boundary":
            plt.title(
                "Whether Angular V is Same Direction as fireflies (Using Angle to Boundary)")
            plt.legend(title="Whether same direction")
        plt.show()


def _show_eye_positions(monkey_sub, axes, x0, y0, marker, eye_col_suffix=''):
    monkey_sub2 = monkey_sub[monkey_sub[f'valid_view_point{eye_col_suffix}'] == True]
    # the below is the same as used above
    axes.scatter(monkey_sub2['gaze_world_x_rotated'].values - x0,
                 monkey_sub2['gaze_world_y_rotated'].values - y0,
                 c=monkey_sub2['time'].values - monkey_sub2['time'].values[0],
                 marker=marker, s=7, zorder=2, cmap='gist_ncar')
    return axes


def _show_connect_path_eye_positions(monkey_sub, axes, x0, y0, player, sample_ratio=4):
    sample = np.arange(1, monkey_sub.shape[0], sample_ratio)
    gaze_world_xy_rotated = np.vstack(
        (monkey_sub['gaze_world_x_rotated'].values, monkey_sub['gaze_world_y_rotated'].values))[:, sample]
    cum_mxy_rotated = np.vstack(
        (monkey_sub['monkey_x_rotated'].values, monkey_sub['monkey_y_rotated'].values))[:, sample]
    axes = connect_points_to_points(axes, gaze_world_xy_rotated, cum_mxy_rotated,
                                    x0, y0, color="black", alpha=connection_alpha[player], linewidth=connection_linewidth[player], show_dots=False)
    return axes


def plot_eye_positions(axes, monkey_information, duration, cum_mxy_rotated, x0, y0,
                       rotation_matrix, player, eye_col_suffix='', marker='o', sample_ratio=6,
                       show_connect_path_eye_positions=False):

    monkey_sub = eye_positions.find_eye_positions_rotated_in_world_coordinates(
        monkey_information, duration, rotation_matrix=rotation_matrix, eye_col_suffix=eye_col_suffix
    )
    axes = _show_eye_positions(monkey_sub, axes, x0, y0, marker, eye_col_suffix=eye_col_suffix)
    if show_connect_path_eye_positions:
        monkey_sub = monkey_sub.assign(
            monkey_x_rotated=cum_mxy_rotated[0],
            monkey_y_rotated=cum_mxy_rotated[1]
        )
        axes = _show_connect_path_eye_positions(
            monkey_sub, axes, x0, y0, player, sample_ratio=sample_ratio)

    return axes
