from null_behaviors import show_null_trajectory
from data_wrangling import general_utils

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import os

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
pd.options.display.max_rows = 101


def _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted):
    arc_xy = show_null_trajectory.find_arc_xy_rotated(null_arc_info_for_the_point['center_x'].item(), null_arc_info_for_the_point['center_y'].item(),
                                                      null_arc_info_for_the_point['all_arc_radius'].item(
    ),
        null_arc_info_for_the_point['arc_starting_angle'].item(
    ), null_arc_info_for_the_point['arc_ending_angle'].item(),
        rotation_matrix=None, num_points=1000)

    # now, we want to find points in arc_xy that are within ff reward boundary
    arc_df = pd.DataFrame({'monkey_x': arc_xy[0], 'monkey_y': arc_xy[1]})
    target_xy = ff_real_position_sorted[null_arc_info_for_the_point['arc_ff_index'].item(
    )]
    arc_df[['ff_x', 'ff_y']] = target_xy
    arc_df['distance_to_ff_center'] = np.sqrt(
        (arc_df['ff_x'] - arc_df['monkey_x'])**2 + (arc_df['ff_y'] - arc_df['monkey_y'])**2)
    arc_df['id'] = np.arange(arc_df.shape[0])
    return arc_df


def _get_arc_xy_rotated(arc_df, reward_boundary_radius=25):
    target_xy = arc_df[['ff_x', 'ff_y']].iloc[0]
    arc_df_sub = arc_df[arc_df['distance_to_ff_center']
                        <= reward_boundary_radius].copy()
    rotation_matrix = general_utils.make_rotation_matrix(
        arc_df_sub['monkey_x'].iloc[0], arc_df_sub['monkey_y'].iloc[0], target_xy.iloc[0], target_xy.iloc[1])
    arc_xy_rotated = np.matmul(
        rotation_matrix, arc_df_sub[['monkey_x', 'monkey_y']].values.T)
    return arc_xy_rotated, rotation_matrix


def find_arc_xy_rotated_for_plotting(null_arc_info_for_the_point, ff_real_position_sorted, reward_boundary_radius=25):
    """
    This function is meant to return the arc info that's inside the reward boundary
    """

    arc_df = _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted)
    arc_xy_rotated, _ = _get_arc_xy_rotated(
        arc_df, reward_boundary_radius=reward_boundary_radius)

    x0 = arc_xy_rotated[0, 0]
    y0 = arc_xy_rotated[1, 0]

    return arc_xy_rotated, x0, y0


def find_arc_xy_rotated_for_plotting2(null_arc_info_for_the_point, ff_real_position_sorted, reward_boundary_radius=25):
    """
    The difference between this function and find_arc_xy_rotated_for_plotting is that this function is meant to return the arc info that's not just inside the reward boundary, 
    but also the points that are before entering the reward boundary 
    """

    arc_df = _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted)
    arc_xy_rotated, rotation_matrix = _get_arc_xy_rotated(
        arc_df, reward_boundary_radius=reward_boundary_radius)

    x0 = arc_xy_rotated[0, 0]
    y0 = arc_xy_rotated[1, 0]

    arc_df_sub = arc_df[arc_df['distance_to_ff_center']
                        <= reward_boundary_radius].copy()
    arc_df_sub2 = arc_df[arc_df['id'] <= arc_df_sub['id'].iloc[-1]].copy()

    arc_xy_rotated = np.matmul(
        rotation_matrix, arc_df_sub2[['monkey_x', 'monkey_y']].values.T)
    return arc_xy_rotated, x0, y0


def plot_null_arc_ends_in_ff(null_arc_info,
                             ff_real_position_sorted,
                             starting_trial=1,
                             max_trials=100,
                             reward_boundary_radius=25,
                             include_arc_portion_before_entering_ff=False):

    # for each point_index & corresponding ff_index, find the point where monkey first enters the reward boundary
    # then make a polar plot, using that entry point as the reference, and let the ff center to be to the north
    # then plot the monkey's trajectory into the circle

    trial_counter = 1
    fig, ax = plt.subplots()

    for i in range(starting_trial, starting_trial + null_arc_info.shape[0]):
        if include_arc_portion_before_entering_ff:
            mxy_rotated, x0, y0 = find_arc_xy_rotated_for_plotting2(
                null_arc_info.iloc[[i]], ff_real_position_sorted, reward_boundary_radius=25)
        else:
            mxy_rotated, x0, y0 = find_arc_xy_rotated_for_plotting(
                null_arc_info.iloc[[i]], ff_real_position_sorted, reward_boundary_radius=25)

        ax = show_xy_overlapped(ax, mxy_rotated, x0, y0)

        if trial_counter > max_trials:
            break
        trial_counter += 1

    ax = _make_a_circle_to_show_reward_boundary(
        ax, reward_boundary_radius=reward_boundary_radius, set_xy_limit=(not include_arc_portion_before_entering_ff), circle_center=(0, 25))

    plt.show()
    return


def _make_a_circle_to_show_reward_boundary(ax, reward_boundary_radius=25, set_xy_limit=True, color='b', circle_center=(0, 0)):
    # plot a circle with radius reward_boundary_radius that centers at (0, reward_boundary_radius)
    circle = plt.Circle(circle_center, reward_boundary_radius,
                        color=color, fill=False)
    ax.add_artist(circle)
    ax.set_aspect('equal')

    if set_xy_limit:
        ax.set_xlim(circle_center[0]-reward_boundary_radius, circle_center[0]+reward_boundary_radius)
        ax.set_ylim(circle_center[1]-reward_boundary_radius, circle_center[1]+reward_boundary_radius)
    return ax


def show_xy_overlapped(ax, mxy_rotated, x0, y0, color='blue', plot_path_to_landing=True):
    if plot_path_to_landing:
        # plot paths to landing
        ax.plot(mxy_rotated[0]-x0, mxy_rotated[1]-y0, alpha=0.2, color=color, linewidth=1)
        # # also plot individual points on the paths
        # ax.plot(mxy_rotated[0]-x0, mxy_rotated[1] -
        #         y0, 'o', markersize=1, alpha=0.2, color=color)
        # plot the ending point of the monkey's trajectory
    ax.plot(mxy_rotated[0, -1]-x0, mxy_rotated[1, -1] -
            y0, 'o', markersize=3, alpha=0.5, color=color)
    return ax
