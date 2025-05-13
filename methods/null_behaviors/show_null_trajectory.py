
from matplotlib import cm
import math
import pandas as pd
import sys
from data_wrangling import specific_utils
from visualization.matplotlib_tools import plot_behaviors_utils, plot_trials
from visualization.animation import animation_func, animation_utils
from null_behaviors import find_best_arc, curvature_utils, optimal_arc_utils
from pattern_discovery import ff_dataframe_utils

import os
import warnings
import numpy as np
import numbers
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def find_relative_xy_positions(ff_x, ff_y, monkey_x, monkey_y, monkey_angle):
    # assume monkey is at origin and heading to the north
    ff_xy = np.stack((ff_x, ff_y), axis=1)
    monkey_xy = np.stack([monkey_x, monkey_y]).T
    ff_distance = LA.norm(ff_xy-monkey_xy, axis=1)
    ff_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ff_x, ff_y=ff_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
    ff_x_relative = np.cos(ff_angle+pi/2)*ff_distance
    ff_y_relative = np.sin(ff_angle+pi/2)*ff_distance
    return ff_x_relative, ff_y_relative


def turn_relative_xy_positions_to_absolute_xy_positions(ff_x_relative, ff_y_relative, monkey_x, monkey_y, monkey_angle):
    ff_distance = LA.norm(np.stack([ff_x_relative, ff_y_relative]).T, axis=1)
    ff_angle = np.arctan2(ff_y_relative, ff_x_relative) - math.pi/2
    ff_angle_in_world_coord = ff_angle + monkey_angle

    ff_x = np.cos(ff_angle_in_world_coord)*ff_distance + monkey_x
    ff_y = np.sin(ff_angle_in_world_coord)*ff_distance + monkey_y
    return ff_x, ff_y


def find_shortest_arc_among_all_available_ff(ff_x, ff_y, monkey_x, monkey_y, monkey_angle, verbose=True):
    '''
    find the shortest arc among all available ff
    '''

    # need to eliminate the ff whose relative y positive to the monkey is negative (a.k.a. behind the monkey)
    ff_xy, ff_distance, ff_angle, ff_angle_boundary, arc_length, arc_radius = find_arc_length_and_radius(
        ff_x, ff_y, monkey_x, monkey_y, monkey_angle, verbose=verbose)

    # Find the shortest arc length
    min_arc_length = min(arc_length)
    rel_index = np.where(arc_length == min_arc_length)[0][0]
    min_arc_radius = arc_radius[rel_index]
    min_arc_ff_xy = ff_xy[rel_index].reshape(-1)
    min_arc_ff_distance = ff_distance[rel_index]
    min_arc_ff_angle = ff_angle[rel_index]
    min_arc_ff_angle_boundary = ff_angle_boundary[rel_index]

    return min_arc_length, min_arc_radius, min_arc_ff_xy, min_arc_ff_distance, min_arc_ff_angle, min_arc_ff_angle_boundary


def find_arc_length_and_radius(ff_x, ff_y, monkey_x, monkey_y, monkey_angle, verbose=True, ignore_error=False):

    monkey_xy = np.stack([monkey_x, monkey_y]).T
    ff_xy = np.stack([ff_x, ff_y]).T

    ff_distance = LA.norm(ff_xy-monkey_xy, axis=1)
    ff_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ff_x, ff_y=ff_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
    ff_angle_boundary = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_angle, distances_to_ff=ff_distance)
    # ff_x_relative = np.cos(ff_angle+pi/2)*ff_distance
    # we used +pi/2 here because the ff_angle is counted as starting from 0 which is to the north of the monkey
    ff_y_relative = np.sin(ff_angle+pi/2)*ff_distance

    # if any abs ff angle is greater pi/4, then raise an error
    if np.any(np.abs(ff_angle) > math.pi/4):
        max_angle = max(np.abs(ff_angle)) * 180/pi
        # if verbose:
        #     warnings.warn("At least one ff is more than 45 degrees away from the monkey. If this is not desired, please eliminate them before calling this function.")
        if not ignore_error:
            raise ValueError(
                f"At least one ff is more than 45 degrees away from the monkey. The max is {max_angle}. If this is not desired, please eliminate them before calling this function.")
        else:
            if verbose:
                warnings.warn(
                    f"At least one ff is more than 45 degrees away from the monkey. The max is {max_angle}. If this is not desired, please eliminate them before calling this function.")

     # supress warnings because invalid values might occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # result will be between (0, pi)
        arc_angle = pi - 2 * np.arcsin(np.abs(ff_y_relative/ff_distance))
        arc_angle[arc_angle == 0] = 0.000001  # to avoid division by zero
        arc_radius = ff_y_relative/np.sin(arc_angle)
        arc_length = arc_radius*arc_angle

    na_index = np.where(np.isnan(arc_length))[0]
    arc_radius[na_index] = 0
    # because na occurs when the monkey can just go a straight line to the ff, instead of an arc
    arc_length[na_index] = ff_distance[na_index]

    if np.any(arc_radius < 0):
        if verbose:
            warnings.warn("At least one arc has a negative radius here, because its relative_y to the monkey is negative. In other words, the ff is behind the monkey. If a negative radius is not desired, please eliminate ff behind the monkey before calling this function.")

    return ff_xy, ff_distance, ff_angle, ff_angle_boundary, arc_length, arc_radius


def find_and_package_optimal_arc_info_for_plotting(best_arc_df, monkey_information=None, column_for_color=None, mode='cartesian',
                                                   ignore_error=False):
    arc_radius = best_arc_df.optimal_arc_radius.values
    arc_end_direction = best_arc_df.optimal_arc_end_direction.values
    arc_point_index = best_arc_df.point_index.values
    arc_measure = best_arc_df.optimal_arc_measure.values
    arc_ff_index = best_arc_df.ff_index.values
    # Note, here arc_ff_xy is replaced by arc ending xy, which produces the same result if the normal optimal arc is used;
    # but if opt_arc_stop_closest was True (optimal arc stop at closest point to monkey stop), then arc ending xy has to be used to mimic a new ff center
    arc_end_xy = best_arc_df[['optimal_arc_end_x', 'optimal_arc_end_y']].values
    ff_distance = best_arc_df.ff_distance.values
    ff_angle = best_arc_df.ff_angle.values
    # arc_ff_xy = ff_real_position_sorted[arc_ff_index] # the same arc ff xy can be used to calculate arc center and angle for both optimal arc and arc to ff center
    if mode == 'cartesian':
        if 'whether_ff_behind' in best_arc_df.columns:
            whether_ff_behind = best_arc_df.whether_ff_behind.values
        else:
            whether_ff_behind = (np.abs(best_arc_df['ff_angle']) > math.pi/2)
        monkey_xy = monkey_information.loc[arc_point_index, [
            'monkey_x', 'monkey_y']].values
        monkey_angle = monkey_information.loc[arc_point_index,
                                              'monkey_angle'].values
        center_x, center_y, arc_starting_angle, arc_ending_angle = optimal_arc_utils.find_cartesian_arc_center_and_angle_for_optimal_arc_to_arc_end(arc_end_xy, monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius,
                                                                                                                                                    arc_end_direction, whether_ff_behind=whether_ff_behind,
                                                                                                                                                    ignore_error=ignore_error)
    elif mode == 'polar':
        center_x, center_y, arc_starting_angle, arc_ending_angle = curvature_utils.find_polar_arc_center_and_angle(
            arc_radius, arc_measure, arc_end_direction)
    else:
        raise ValueError("Invalid mode specified. Use 'cartesian' or 'polar'.")

    null_arc_info_for_plotting = {'arc_point_index': arc_point_index, 'arc_ff_index': arc_ff_index, 'center_x': center_x, 'center_y': center_y, 'arc_starting_angle': arc_starting_angle,
                                  'arc_ending_angle': arc_ending_angle, 'all_arc_radius': arc_radius, 'all_arc_end_direction': arc_end_direction}
    if column_for_color is not None:
        null_arc_info_for_plotting['values_for_color'] = best_arc_df[column_for_color].values
    else:
        null_arc_info_for_plotting['values_for_color'] = None
    null_arc_info_for_plotting = pd.DataFrame(null_arc_info_for_plotting)

    return null_arc_info_for_plotting


def find_and_package_arc_to_center_info_for_plotting(all_point_index, all_ff_index, monkey_information, ff_real_position_sorted, verbose=True,
                                                     ff_x=None, ff_y=None, ignore_error=False):

    if isinstance(all_point_index, int):
        all_point_index = [all_point_index]

    if ff_x is None:
        ff_x = ff_real_position_sorted[all_ff_index, 0]
    if ff_y is None:
        ff_y = ff_real_position_sorted[all_ff_index, 1]
    ff_xy = np.stack((ff_x, ff_y))
    monkey_x = monkey_information['monkey_x'].loc[all_point_index].values
    monkey_y = monkey_information['monkey_y'].loc[all_point_index].values
    monkey_xy = np.stack((monkey_x, monkey_y))
    monkey_angle = monkey_information['monkey_angle'].loc[all_point_index].values
    ff_xy, ff_distance, ff_angle, ff_angle_boundary, arc_length, arc_radius = find_arc_length_and_radius(
        ff_x, ff_y, monkey_x, monkey_y, monkey_angle, verbose=verbose, ignore_error=ignore_error)
    arc_end_direction = np.sign(ff_angle)
    center_x, center_y, arc_starting_angle, arc_ending_angle = curvature_utils.find_cartesian_arc_center_and_angle_for_arc_to_center(monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, ff_xy, arc_end_direction,
                                                                                                                                     ignore_error=ignore_error)
    arc_measure = np.abs(arc_ending_angle-arc_starting_angle)
    null_arc_info_for_plotting = {'arc_point_index': all_point_index, 'arc_ff_index': all_ff_index, 'all_arc_measure': arc_measure, 'center_x': center_x, 'center_y': center_y, 'arc_starting_angle': arc_starting_angle,
                                  'arc_ending_angle': arc_ending_angle, 'all_arc_radius': arc_radius, 'all_arc_end_direction': arc_end_direction}
    null_arc_info_for_plotting = pd.DataFrame(null_arc_info_for_plotting)

    # eliminate the rows where the ff is more than 45 degrees away from the monkey
    rows_to_keep = np.where(np.abs(ff_angle) <= math.pi/4)
    null_arc_info_for_plotting = null_arc_info_for_plotting.iloc[rows_to_keep]

    return null_arc_info_for_plotting


def update_null_arc_info_based_on_fixed_arc_length(fixed_arc_length, null_arc_info):
    null_arc_info['all_arc_measure'] = fixed_arc_length / \
        null_arc_info['all_arc_radius']

    ff_to_the_left = null_arc_info['all_arc_end_direction'] == 1
    null_arc_info.loc[ff_to_the_left, 'arc_ending_angle'] = null_arc_info.loc[ff_to_the_left,
                                                                              'arc_starting_angle'] + null_arc_info.loc[ff_to_the_left, 'all_arc_measure']

    ff_to_the_right = null_arc_info['all_arc_end_direction'] == -1
    null_arc_info.loc[ff_to_the_right, 'arc_ending_angle'] = null_arc_info.loc[ff_to_the_right,
                                                                               'arc_starting_angle'] - null_arc_info.loc[ff_to_the_right, 'all_arc_measure']

    return null_arc_info


def find_monkey_angle_after_an_arc(ff_xy, min_arc_ff_angle, center_x, center_y, monkey_angle):
    angle_from_center_to_stop = np.arctan2(
        ff_xy[1]-center_y, ff_xy[0]-center_x)
    # if angle_from_center_to_stop is the starting angle, then monkey_angle at the end of the trajectory should be angle_from_center_to_stop - pi/2
    if min_arc_ff_angle < 0:  # ff is to the right of the monkey
        monkey_angle = angle_from_center_to_stop - pi/2
    # otherwise, it should be angle_from_center_to_stop + pi/2
    else:
        monkey_angle = angle_from_center_to_stop + pi/2
    return monkey_angle


def find_arc_xy(center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, num_points=100):
    # Plot an arc from arc_starting_angle to arc_ending_angle with arc_radius
    angle_array = np.linspace(arc_starting_angle, arc_ending_angle, num_points)
    # see the arc as part of a circle
    # suppress warning for the code bellow:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # the x and y coordinates of points on the arc
        arc_x_array = np.cos(angle_array)*min_arc_radius + center_x
        arc_y_array = np.sin(angle_array)*min_arc_radius + center_y
    return arc_x_array, arc_y_array


def find_arc_xy_rotated(center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, rotation_matrix=None, num_points=100):
    arc_x_array, arc_y_array = find_arc_xy(
        center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, num_points=num_points)
    arc_xy_rotated = np.stack((arc_x_array, arc_y_array))
    if rotation_matrix is not None:
        if arc_xy_rotated.ndim == 3:
            arc_xy_rotated_T = np.transpose(arc_xy_rotated, (2, 0, 1))
            arc_xy_rotated = np.transpose(
                np.matmul(rotation_matrix, arc_xy_rotated_T), (1, 2, 0))
        else:  # we assume arc_xy_rotated is a 2d array
            arc_xy_rotated = np.matmul(rotation_matrix, arc_xy_rotated)

    return arc_xy_rotated


def find_most_recent_monkey_information(monkey_information, current_moment):
    duration = [current_moment-2, current_moment]
    cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
        duration, monkey_information)
    monkey_x, monkey_y, monkey_angle = cum_mx[-1], cum_my[-1], cum_angle[-1]
    monkey_xy = np.stack([monkey_x, monkey_y]).T
    point_index = monkey_information.iloc[cum_pos_index[-1]]['point_index']
    return monkey_xy, monkey_angle, point_index


def show_null_agent_trajectory_func(duration, null_agent_starting_time, monkey_information, ff_dataframe, ff_caught_T_new,
                                    axes, legend_markers, legend_names, R, assumed_memory_duration_of_agent, show_null_agent_trajectory_2nd_time=False,
                                    show_ff_to_be_considered_by_first_null_trajectory=True, show_starting_point_of_show_null_trajectory=True, show_landing_point_of_show_null_trajectory=True,
                                    reaching_boundary_ok=True, type='most_aligned', null_arc_info_for_plotting=None):
    # This function is based on the assumption that the monkey will choose the shortest arc

    if (type != 'shortest') and (type != 'most_aligned'):
        raise ValueError('type can only be \'shortest\' or \'most_aligned\'.')

    if null_agent_starting_time is None:
        current_moment = duration[1]
    else:
        current_moment = null_agent_starting_time
    monkey_xy, monkey_angle, point_index = find_most_recent_monkey_information(
        monkey_information, current_moment)
    print('1st arch:')
    if type == 'most_aligned':
        if null_arc_info_for_plotting is None:
            raise ValueError(
                'null_arc_info_for_plotting must be provided if type is \'most_aligned\'.')
        # index = np.where(null_arc_info_for_plotting['arc_point_index'].values == point_index)[0]
        # arc_xy_to_plot = find_arc_xy_rotated(null_arc_info_for_plotting['center_x'].values[index], null_arc_info_for_plotting['center_y'].values[index], null_arc_info_for_plotting['all_arc_radius'].values[index], \
        #                                     null_arc_info_for_plotting['arc_starting_angle'].values[index], null_arc_info_for_plotting['arc_ending_angle'].values[index], rotation_matrix=R).reshape(2, -1)
        # axes.scatter(arc_xy_to_plot[0], arc_xy_to_plot[1], s=1, alpha=0.8, zorder=2, color='lime')
        # min_arc_ff_xy_rotated = np.array([arc_xy_to_plot[0, -1], arc_xy_to_plot[1, -1]])
        axes, whether_plotted = plot_null_arcs_from_best_arc_df(
            axes, point_index, null_arc_info_for_plotting, x0=0, y0=0, rotation_matrix=R, polar=False, zorder=6, alpha=0.8, color='lime', marker_size=1)
    elif type == 'shortest':
        axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y, ff_xy_to_be_considered \
            = plot_shortest_arc_from_null_condition(axes, current_moment, ff_dataframe, ff_caught_T_new, monkey_xy, monkey_angle, rotation_matrix=R, assumed_memory_duration=assumed_memory_duration_of_agent,
                                                    arc_color="lime", reaching_boundary_ok=reaching_boundary_ok)
        min_arc_ff_xy_rotated = np.matmul(R, min_arc_ff_xy.reshape(2, 1))
    line = Line2D([0], [0], linestyle="-", alpha=0.7,
                  linewidth=2, color="lime")
    legend_markers.append(line)
    legend_names.append('Null trajectory')

    if show_starting_point_of_show_null_trajectory:
        monkey_xy_rotated = np.matmul(R, monkey_xy)
        marker = axes.scatter(
            monkey_xy_rotated[0], monkey_xy_rotated[1], s=50, zorder=5, color='darkorange')
        legend_markers.append(marker)
        legend_names.append('Starting point of null trajectory')

    if show_landing_point_of_show_null_trajectory & (type == 'shortest'):
        if min_arc_ff_xy_rotated is not None:
            marker = axes.scatter(
                min_arc_ff_xy_rotated[0], min_arc_ff_xy_rotated[1], s=40, zorder=5, color='blue')
            legend_markers.append(marker)
            legend_names.append('Landing point of null trajectory')

    if show_null_agent_trajectory_2nd_time:
        if type == 'most_aligned':
            raise ValueError(
                'If type = \'most_alignt\', show_null_agent_trajectory_2nd_time cannot be True')
        elif type == 'shortest':
            print('2nd arch:')
            if min_arc_ff_xy is not None:
                min_time = min_arc_length/200
                current_moment = duration[1]+min_time
                monkey_xy = min_arc_ff_xy
                if center_x is not None:
                    monkey_angle = find_monkey_angle_after_an_arc(
                        min_arc_ff_xy, min_arc_ff_angle, center_x, center_y, monkey_angle)
                # eliminate the caught ff from ff_dataframe
                ff_dataframe_sub = ff_dataframe[(ff_dataframe['ff_x'] != min_arc_ff_center_xy[0]) | (
                    ff_dataframe['ff_y'] != min_arc_ff_center_xy[1])]
            else:
                current_moment = duration[1] + 2
                ff_dataframe_sub = ff_dataframe.copy()

                axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y, ff_xy_to_be_considered_2nd \
                    = plot_shortest_arc_from_null_condition(axes, current_moment, ff_dataframe_sub, ff_caught_T_new, monkey_xy, monkey_angle, rotation_matrix=R, assumed_memory_duration=2,
                                                            arc_color="navy", reaching_boundary_ok=reaching_boundary_ok, zorder=4)
                line = Line2D([0], [0], linestyle="-",
                              alpha=0.7, linewidth=2, color="navy")
                legend_markers.append(line)
                legend_names.append('2nd Null trajectory')

                if show_landing_point_of_show_null_trajectory:
                    if min_arc_ff_xy is not None:
                        min_arc_ff_xy_rotated = np.matmul(
                            R, min_arc_ff_xy.reshape(2, 1))
                        marker = axes.scatter(
                            min_arc_ff_xy_rotated[0], min_arc_ff_xy_rotated[1], s=40, zorder=3, color='darkorange')

    if show_ff_to_be_considered_by_first_null_trajectory & (type == 'shortest'):
        if ff_xy_to_be_considered is not None:
            marker = axes.scatter(
                ff_xy_to_be_considered[0], ff_xy_to_be_considered[1], s=60, zorder=2, color='gold')
            legend_markers.append(marker)
            legend_names.append(
                'FFs to be considered by\nthe first null trajectory')

    return axes, legend_markers, legend_names


def plot_shortest_arc_from_null_condition(axes, current_moment, ff_dataframe, ff_caught_T_new, monkey_xy, monkey_angle, rotation_matrix=None, assumed_memory_duration=2, arc_color='black',
                                          reaching_boundary_ok=False):
    R = rotation_matrix
    duration = [current_moment-assumed_memory_duration, current_moment]
    # monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    if monkey_xy.ndim == 1:
        monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    elif monkey_xy.shape[1] == 2:
        monkey_x, monkey_y = monkey_xy[:, 0], monkey_xy[:, 1]
    else:
        monkey_x, monkey_y = monkey_xy[0, :], monkey_xy[1, :]
    print("duration for plotting an arc: ", duration)
    ff_dataframe_temp = ff_dataframe[ff_dataframe['time'].between(
        duration[0], duration[1])]
    ff_dataframe_temp = ff_dataframe_temp[ff_dataframe_temp['visible'] == 1]
    if len(ff_dataframe_temp) < 1:
        print("No firefly was seen in the last 2 seconds")
        return axes, None, None, None, None, None, None, None

    # eliminate the information of ff in ff_dataframe that have already been caught prior to current_moment
    current_trial_number = np.where(ff_caught_T_new > current_moment)[0][0]
    ff_dataframe_temp = ff_dataframe_temp[ff_dataframe_temp['ff_index']
                                          >= current_trial_number]

    ff_xy_and_angle = ff_dataframe_temp[[
        'ff_x', 'ff_y', 'ff_angle', 'ff_distance']].drop_duplicates().values
    ff_xy = ff_xy_and_angle[:, :2]
    ff_angle = ff_xy_and_angle[:, 2]
    ff_distance = ff_xy_and_angle[:, 3]
    ff_center_xy = ff_xy

    if reaching_boundary_ok:
        ff_xy = find_best_arc.find_point_on_ff_boundary_with_smallest_angle_to_monkey(
            ff_xy[:, 0], ff_xy[:, 1], monkey_xy[0], monkey_xy[1], monkey_angle)

    # ff_x_relative, ff_y_relative = find_relative_xy_positions(ff_xy[:,0], ff_xy[:,1], monkey_x, monkey_y, monkey_angle)
    # ff_to_be_considered = np.where(ff_y_relative > 0)[0]
    ff_to_be_considered = np.where(np.abs(ff_angle)*180/pi <= 45)[0]
    if len(ff_to_be_considered) == 0:
        print("No firefly was in front of the monkey and visible within the last 2s.")
        return axes, None, None, None, None, None, None, None

    min_arc_length, min_arc_radius, min_arc_ff_xy, min_arc_ff_distance, min_arc_ff_angle, \
        min_arc_ff_angle_boundary = find_shortest_arc_among_all_available_ff(
            ff_xy[ff_to_be_considered, 0], ff_xy[ff_to_be_considered, 1], monkey_x, monkey_y, monkey_angle)
    min_arc_relative_index = np.where(
        (ff_xy[:, 0] == min_arc_ff_xy[0]) & (ff_xy[:, 1] == min_arc_ff_xy[1]))[0]
    min_arc_ff_center_xy = ff_center_xy[min_arc_relative_index].reshape(-1)
    if min_arc_radius > 0:
        whether_ff_behind = (np.abs(min_arc_ff_angle) > math.pi/2)
        center_x, center_y, arc_starting_angle, arc_ending_angle = curvature_utils.find_cartesian_arc_center_and_angle(
            monkey_xy, monkey_angle, ff_distance, ff_angle, min_arc_radius, min_arc_ff_xy, np.sign(min_arc_ff_angle), whether_ff_behind=whether_ff_behind)
        arc_xy_rotated = find_arc_xy_rotated(
            center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, rotation_matrix=rotation_matrix)
        # center_xy_rotated = np.matmul(R, np.stack([center_x, center_y]))
        # axes.scatter(center_xy_rotated[0], center_xy_rotated[1], s=40, zorder=4, color='blue')
    else:
        center_x, center_y = None, None
        # plot a line from the monkey to the ff
        arc_xy_rotated = np.stack(
            (np.array([monkey_x, min_arc_ff_xy[0]]), np.array([monkey_y, min_arc_ff_xy[1]])))
        if rotation_matrix is not None:
            arc_xy_rotated = np.matmul(R, arc_xy_rotated)

    ff_xy_to_be_considered = np.matmul(R, ff_center_xy[ff_to_be_considered].T)
    axes.plot(arc_xy_rotated[0], arc_xy_rotated[1],
              linewidth=2.5, color=arc_color, zorder=6)

    return axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y, ff_xy_to_be_considered


def plot_null_arcs_from_best_arc_df(axes, relevant_point_index, null_arc_info_for_plotting, x0=0, y0=0, rotation_matrix=None, polar=False,
                                    zorder=2, alpha=None, color=None, marker_size=None):
    # To plot null arcs
    # x0, y0 = cum_mxy_rotated[0][0], cum_mxy_rotated[1][0]

    line_colors = None
    current_line_color = color
    if 'values_for_color' in null_arc_info_for_plotting.columns:
        if null_arc_info_for_plotting['values_for_color'].values[0] is not None:
            viridis = matplotlib.colormaps['viridis']
            line_colors = viridis(
                null_arc_info_for_plotting['values_for_color'].values)

    if relevant_point_index is None:
        relevant_point_index = null_arc_info_for_plotting['arc_point_index'].values

    # if relevant_point_index is a float, make it into a list
    if isinstance(relevant_point_index, numbers.Real):
        relevant_point_index = [relevant_point_index]

    for point_index in relevant_point_index:
        indices = np.where(
            null_arc_info_for_plotting['arc_point_index'].values == point_index)[0]
        if len(indices) > 0:
            for index in indices:

                if (current_line_color is None) & (line_colors is not None):
                    current_line_color = line_colors[index]

                arc_xy_rotated = find_arc_xy_rotated(null_arc_info_for_plotting['center_x'].values[index], null_arc_info_for_plotting['center_y'].values[index], null_arc_info_for_plotting['all_arc_radius'].values[index],
                                                     null_arc_info_for_plotting['arc_starting_angle'].values[index], null_arc_info_for_plotting['arc_ending_angle'].values[index], rotation_matrix=rotation_matrix)
                arc_xy_to_plot = arc_xy_rotated.reshape(2, -1)
                if polar is True:
                    r, theta = animation_utils.change_xy_to_polar(
                        arc_xy_to_plot[0]-x0, arc_xy_to_plot[1]-y0)
                    theta = theta - math.pi/2  # since 0 is to the north
                    if alpha is None:
                        alpha = 0.7
                    if marker_size is None:
                        marker_size = 2
                    axes.scatter(theta, r, s=marker_size, alpha=alpha,
                                 zorder=zorder, color=current_line_color)
                else:
                    if alpha is None:
                        alpha = 0.3
                    if marker_size is None:
                        marker_size = 0.1
                    axes.scatter(arc_xy_to_plot[0]-x0, arc_xy_to_plot[1]-y0, s=marker_size,
                                 alpha=alpha, zorder=zorder, color=current_line_color)
            whether_plotted = True
        else:
            whether_plotted = False

    return axes, whether_plotted


def find_ff_near_intended_target(intended_target_ff_index, duration, ff_dataframe, max_distance_from_intended_target):

    ff_dataframe_sub = ff_dataframe[ff_dataframe['ff_index']
                                    == intended_target_ff_index]
    if ff_dataframe_sub.shape[0] == 0:
        raise ValueError('intended_target_ff_index is not in ff_dataframe.')
    intended_target_xy = ff_dataframe_sub[['ff_x', 'ff_y']].values[0]

    ff_dataframe_sub = ff_dataframe[ff_dataframe['time'].between(
        duration[0], duration[1])]
    unique_ff = ff_dataframe_sub[[
        'ff_x', 'ff_y', 'ff_index']].drop_duplicates()
    unique_ff['distance_from_intended_target'] = unique_ff.apply(
        lambda row: np.linalg.norm(row[['ff_x', 'ff_y']].values - intended_target_xy), axis=1)
    selected_ff = unique_ff[unique_ff['distance_from_intended_target']
                            <= max_distance_from_intended_target].drop_duplicates()
    ff_near_intended_target = selected_ff['ff_index'].values

    return ff_near_intended_target


def eliminate_irrelevant_points_before_or_after_crossing_boundary(duration, relevant_point_index, monkey_information, verbose=True):
    # eliminate unnecessary parts seperated by crossing boundary
    if len(relevant_point_index) == 0:
        raise ValueError('relevant_point_index cannot be empty.')
    crossing_boundary_points = monkey_information.loc[monkey_information['time'].between(duration[0], duration[1]) &
                                                      monkey_information['crossing_boundary'] == 1, 'point_index'].values
    cb_after_ff = crossing_boundary_points[crossing_boundary_points >= max(
        relevant_point_index)]
    cb_before_ff = crossing_boundary_points[crossing_boundary_points <= min(
        relevant_point_index)]
    if len(cb_after_ff) > 0:
        duration[1] = max(monkey_information.loc[monkey_information['point_index'] < min(
            cb_after_ff), 'time'].values)
    if len(cb_before_ff) > 0:
        duration[0] = min(monkey_information.loc[monkey_information['point_index'] >= max(
            cb_before_ff), 'time'].values)

    if verbose:
        print('duration after eliminating unnecessary parts: ', duration)
    return duration


def eliminate_invalid_ff_for_null_arc(all_ff_index, all_point_index, ff_real_position_sorted, monkey_information):
    # we want to eliminate ff with negative y
    ff_x = ff_real_position_sorted[all_ff_index, 0]
    ff_y = ff_real_position_sorted[all_ff_index, 1]
    monkey_x = monkey_information['monkey_x'].loc[all_point_index].values
    monkey_y = monkey_information['monkey_y'].loc[all_point_index].values
    monkey_angle = monkey_information['monkey_angle'].loc[all_point_index].values
    ff_x_relative, ff_y_relative = find_relative_xy_positions(
        ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
    ff_angle = np.arctan2(ff_y_relative, ff_x_relative) - math.pi/2
    print(round(len(np.where(ff_y_relative < 0)[
          0])/len(ff_y_relative)*100, 2), '% of ff has negative y relative to monkey')
    print(round(len(np.where(np.abs(ff_angle) >= pi/4)
          [0])/len(ff_angle)*100, 2), '% of ff has ff angle that is too large')
    print('Both of these cases are eliminated.')

    remaining_index_of_array = np.where(
        (ff_y_relative >= 0) & (np.abs(ff_angle) >= pi/4))[0]
    remaining_all_ff_index = all_ff_index[remaining_index_of_array]
    remaining_all_point_index = all_point_index[remaining_index_of_array]

    return remaining_index_of_array, remaining_all_ff_index, remaining_all_point_index


def find_point_indices_to_plot_null_arc(duration_to_plot, monkey_information, time_to_begin_plotting_null_arc=None, time_to_end_plotting_null_arc=None, time_between_every_two_null_arcs=1):

    if time_to_begin_plotting_null_arc is None:
        time_to_begin_plotting_null_arc = duration_to_plot[0]
    elif time_to_begin_plotting_null_arc < duration_to_plot[0]:
        time_to_begin_plotting_null_arc = duration_to_plot[0]
        warnings.warn(
            'time_to_begin_plotting_null_arc cannot be smaller than duration_to_plot[0]. It is set to be duration_to_plot[0].')
    if time_to_end_plotting_null_arc is None:
        time_to_end_plotting_null_arc = duration_to_plot[1]
    elif time_to_end_plotting_null_arc > duration_to_plot[1]:
        time_to_end_plotting_null_arc = duration_to_plot[1]
        warnings.warn(
            'time_to_end_plotting_null_arc cannot be larger than duration_to_plot[1].')
    if time_to_end_plotting_null_arc <= time_to_begin_plotting_null_arc:
        raise ValueError(
            'time_to_end_plotting_null_arc cannot be smaller than or equal to time_to_begin_plotting_null_arc.')

    point_indices_to_plot_null_arc = monkey_information[monkey_information['time'].between(
        time_to_begin_plotting_null_arc, time_to_end_plotting_null_arc)].index.values
    monkey_dt = (monkey_information['time'].iloc[-1] -
                 monkey_information['time'].iloc[0])/(len(monkey_information)-1)
    num_point_index_between_every_two_null_arcs = int(
        time_between_every_two_null_arcs/monkey_dt)
    point_indices_to_plot_null_arc = point_indices_to_plot_null_arc[range(0, len(
        point_indices_to_plot_null_arc), num_point_index_between_every_two_null_arcs)]

    return point_indices_to_plot_null_arc


def make_pretty_plot_for_a_duration(duration_to_plot, best_arc_df, monkey_information, ff_dataframe, null_arc_info_for_plotting, null_arc_to_center_info_for_plotting, PlotTrials_args, pretty_null_arc_plot_kwargs, intended_target_ff_index=None,
                                    ff_max_distance_to_intended_target=2000, ff_max_distance_to_path=100, plot_null_arcs=True, time_between_every_two_null_arcs=1, time_to_begin_plotting_null_arc=None, time_to_end_plotting_null_arc=None, max_num_plot_to_make=30,
                                    show_intended_target=False, show_ff_indices=False, point_indices_to_plot_null_arc=None, ff_indices_to_plot_null_arc=None, whether_mark_path_where_intended_target_has_best_arc_among_all_ff=False,
                                    add_intended_target_ff_index_to_title=True, add_duration_to_plot_to_title=False, eliminate_irrelevant_points_for_intended_target_before_or_after_crossing_boundary=True, only_plot_when_null_arc_exists=False,
                                    additional_plotting_kwargs={}, title=None,
                                    skip_plots_with_no_null_arc=True,
                                    ):

    plot_counter = 0
    pretty_null_arc_plot_kwargs = pretty_null_arc_plot_kwargs.copy()

    if eliminate_irrelevant_points_for_intended_target_before_or_after_crossing_boundary & (intended_target_ff_index is not None):
        if intended_target_ff_index >= 0:
            best_arc_df_for_intended_target = best_arc_df[(best_arc_df.ff_index == intended_target_ff_index) & (
                best_arc_df.time.between(duration_to_plot[0], duration_to_plot[1]))]
            relevant_point_indices = best_arc_df_for_intended_target.point_index.values
            if len(relevant_point_indices) > 0:
                duration_to_plot = eliminate_irrelevant_points_before_or_after_crossing_boundary(
                    duration_to_plot, relevant_point_indices, monkey_information)

    if show_ff_indices == True:
        pretty_null_arc_plot_kwargs['show_ff_indices'] = True

    if show_intended_target == True:
        if intended_target_ff_index is not None:
            pretty_null_arc_plot_kwargs['indices_of_ff_to_mark'] = [
                intended_target_ff_index]
        else:
            raise ValueError(
                'if show_intended_target is True, then intended_target_ff_index cannot be None.')

    if plot_null_arcs:
        if point_indices_to_plot_null_arc is None:
            point_indices_to_plot_null_arc = find_point_indices_to_plot_null_arc(duration_to_plot, monkey_information, time_to_begin_plotting_null_arc=time_to_begin_plotting_null_arc,
                                                                                 time_to_end_plotting_null_arc=time_to_end_plotting_null_arc, time_between_every_two_null_arcs=time_between_every_two_null_arcs)
        else:
            point_indices_to_plot_null_arc = np.array(
                point_indices_to_plot_null_arc)
    else:
        point_indices_to_plot_null_arc = [None]

    if ff_indices_to_plot_null_arc is None:
        ff_indices_to_plot_null_arc = find_ff_indices_near_intended_target_to_plot_null_arc(
            duration_to_plot, ff_dataframe, intended_target_ff_index=intended_target_ff_index, ff_max_distance_to_intended_target=ff_max_distance_to_intended_target, ff_max_distance_to_path=ff_max_distance_to_path)

    temp_null_arc_info_df = null_arc_info_for_plotting[null_arc_info_for_plotting['arc_ff_index'].isin(
        ff_indices_to_plot_null_arc)].copy()
    temp_null_arc_to_center_info_df = null_arc_to_center_info_for_plotting[null_arc_to_center_info_for_plotting['arc_ff_index'].isin(
        ff_indices_to_plot_null_arc)].copy()
    temp_null_arc_info_df['values_for_color'] = None

    for point_index_to_plot_null_arc in point_indices_to_plot_null_arc:
        if skip_plots_with_no_null_arc:
            if (len(temp_null_arc_to_center_info_df[temp_null_arc_to_center_info_df['arc_point_index'] == point_index_to_plot_null_arc]) == 0) & \
                    (len(temp_null_arc_info_df[temp_null_arc_info_df['arc_point_index'] == point_index_to_plot_null_arc]) == 0):
                print('No null arc for point_index_to_plot_null_arc: ', point_index_to_plot_null_arc,
                      '. Skipping this plot, since skip_plots_with_no_null_arc is True.')
                continue

        if point_index_to_plot_null_arc is not None:
            if only_plot_when_null_arc_exists:
                if (point_index_to_plot_null_arc not in temp_null_arc_info_df.arc_point_index.values) &\
                        (point_index_to_plot_null_arc not in temp_null_arc_to_center_info_df.arc_point_index.values):
                    continue
            print('Current point_index_to_plot_null_arc: ',
                  point_index_to_plot_null_arc)
            point_indices_to_be_marked_3rd_kind = [
                point_index_to_plot_null_arc]
        else:
            point_indices_to_be_marked_3rd_kind = None

        for key in additional_plotting_kwargs.keys():
            pretty_null_arc_plot_kwargs[key] = additional_plotting_kwargs[key]

        returned_info = plot_trials.PlotTrials(
            duration_to_plot,
            *PlotTrials_args,
            **pretty_null_arc_plot_kwargs,
            show_visible_segments_ff_indices=ff_indices_to_plot_null_arc,
            point_indices_to_be_marked_3rd_kind=point_indices_to_be_marked_3rd_kind,
            indices_of_ff_to_be_plotted_in_a_basic_way=ff_indices_to_plot_null_arc,
        )
        R = returned_info['rotation_matrix']
        axes = returned_info['axes']

        if whether_mark_path_where_intended_target_has_best_arc_among_all_ff:
            if intended_target_ff_index is None:
                raise ValueError(
                    'if whether_mark_path_where_intended_target_has_best_arc_among_all_ff is True, then intended_target_id cannot be None.')
            elif intended_target_ff_index >= 0:
                best_arc_df_for_intended_target = best_arc_df[(best_arc_df.ff_index == intended_target_ff_index) & (
                    best_arc_df.time.between(duration_to_plot[0], duration_to_plot[1]))]
                relevant_point_indices = best_arc_df_for_intended_target.point_index.values
            axes = mark_path_where_intended_target_has_best_arc_among_all_ff_func(
                axes, best_arc_df_for_intended_target, relevant_point_indices, monkey_information, rotation_matrix=R)

        if point_index_to_plot_null_arc is not None:
            axes, whether_plotted = plot_null_arcs_from_best_arc_df(axes, [point_index_to_plot_null_arc], temp_null_arc_info_df, x0=0, y0=0,
                                                                    rotation_matrix=R, polar=False, marker_size=1.5, alpha=0.7, zorder=5, color='blue')
            axes, whether_plotted = plot_null_arcs_from_best_arc_df(axes, [point_index_to_plot_null_arc], temp_null_arc_to_center_info_df, x0=0, y0=0,
                                                                    rotation_matrix=R, polar=False, marker_size=1.2, alpha=0.7, zorder=6, color='red')

        if title is not None:
            plt.title(title, fontsize=18)
        else:
            current_title = axes.get_title()
            if add_intended_target_ff_index_to_title & (intended_target_ff_index is not None):
                current_title = current_title + ', Intended Target ff index: ' + \
                    str(intended_target_ff_index)
            if add_duration_to_plot_to_title:
                current_title = current_title + ', Duration: ' + \
                    str(round(duration_to_plot[0], 2)) + \
                    '-' + str(round(duration_to_plot[1], 2))
            plt.title(current_title, fontsize=18)

        plt.show()

        plot_counter += 1
        if plot_counter == max_num_plot_to_make:
            break
    return


def find_ff_indices_near_intended_target_to_plot_null_arc(duration_to_plot, ff_dataframe, intended_target_ff_index=None, ff_max_distance_to_intended_target=2000, ff_max_distance_to_path=100):
    ff_dataframe_in_duration = ff_dataframe[ff_dataframe['time'].between(
        duration_to_plot[0], duration_to_plot[1])]
    ff_near_trajectory = np.unique(ff_dataframe_utils.keep_only_ff_that_monkey_has_passed_by_closely(
        ff_dataframe_in_duration, max_distance_to_ff=ff_max_distance_to_path).ff_index.values)
    if intended_target_ff_index is not None:
        ff_near_intended_target = find_ff_near_intended_target(
            intended_target_ff_index, duration_to_plot, ff_dataframe_in_duration, ff_max_distance_to_intended_target)
        ff_indices_to_plot_null_arc = np.intersect1d(
            ff_near_intended_target, ff_near_trajectory)
    else:
        ff_indices_to_plot_null_arc = ff_near_trajectory
    return ff_indices_to_plot_null_arc


def mark_path_where_intended_target_has_best_arc_among_all_ff_func(axes, best_arc_df_for_one_id, relevant_point_index, monkey_information, rotation_matrix=None):
    relevant_percentile = best_arc_df_for_one_id.diff_percentile_in_decimal.values
    temp_cum_mx, temp_cum_my = np.array(monkey_information['monkey_x'].loc[relevant_point_index]), np.array(
        monkey_information['monkey_y'].loc[relevant_point_index])
    temp_cum_mxy_rotated = np.matmul(
        rotation_matrix, np.stack((temp_cum_mx, temp_cum_my)))
    axes.scatter(temp_cum_mxy_rotated[0], temp_cum_mxy_rotated[1], marker='*',
                 s=50, c=relevant_percentile, cmap='viridis', zorder=3, alpha=0.6)
    return axes


def show_percentile_by_color_func(fig, axes, best_arc_sub_info_valid_ff, R):
    # let the color be based on diff_percentile
    viridis = matplotlib.colormaps['viridis']
    line_colors = viridis(
        best_arc_sub_info_valid_ff.diff_percentile_in_decimal.values)
    temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.find_lines_to_connect_path_ff(
        best_arc_sub_info_valid_ff, target_indices=None, rotation_matrix=R, target_excluded=False)
    x0, y0 = 0, 0
    if temp_ff_positions.shape[1] > 0:
        for j in range(temp_ff_positions.shape[1]):
            axes.plot(np.stack([temp_ff_positions[0, j]-x0, temp_monkey_positions[0, j]-x0]),
                      np.stack([temp_ff_positions[1, j]-y0,
                               temp_monkey_positions[1, j]-y0]),
                      '-', alpha=0.7, linewidth=0.5, c=line_colors[j])
    # plot a colorbar for it
    cmap = cm.viridis
    vmax = 100
    title = 'Percentile of Abs Diff of Angle Over Distance'
    norm2 = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=fig.add_axes(
        [0.95, 0.4, 0.05, 0.43]), orientation='vertical')
    cbar2.outline.set_visible(False)
    cbar2.ax.tick_params(axis='y', color='lightgrey',
                         direction="in", right=True, length=5, width=1.5)
    cbar2.ax.set_title(title, ha='left', y=1.04)
    return fig, axes


def make_plots_to_show_monkey_reaction_time(curvature_df,
                                            null_arcs_plotting_kwargs,
                                            PlotTrials_args,
                                            time,
                                            show_percentile_by_color=False,
                                            additional_plotting_kwargs={},
                                            ):

    null_arcs_plotting_kwargs_temp = null_arcs_plotting_kwargs.copy()

    curvature_df_sub = curvature_df[curvature_df.time.between(time-1, time)]
    if (len(np.unique(curvature_df_sub.ff_index.values)) > 1) or (len(np.unique(curvature_df_sub.ff_index.values)) == 0):
        return

    duration_to_plot = [time-2.5, time+2.5]
    returned_info = plot_trials.PlotTrials(
        duration_to_plot,
        *PlotTrials_args,
        **null_arcs_plotting_kwargs_temp,
        **additional_plotting_kwargs,
    )

    R = returned_info['rotation_matrix']
    axes = returned_info['axes']
    whether_plotted = returned_info['whether_plotted']

    if whether_plotted == True:
        best_arc_sub_info = curvature_df[curvature_df.time.between(
            duration_to_plot[0], time)].copy()
        best_arc_sub_info_valid_ff = best_arc_sub_info[best_arc_sub_info['ff_index'] >= 0]

        if not show_percentile_by_color:
            axes, legend_markers, legend_names, temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.plot_lines_to_connect_path_and_ff(axes=axes,
                                                                                                                                                  ff_info=best_arc_sub_info_valid_ff, rotation_matrix=R, x0=0, y0=0, linewidth=0.5, alpha=0.7, vary_color_for_connecting_path_ff=True,
                                                                                                                                                  show_points_when_ff_stop_being_visible=True, show_points_when_ff_start_being_visible=True)
        else:
            fig, axes = show_percentile_by_color_func(
                fig, axes, best_arc_sub_info_valid_ff, R)

    return fig, axes
