import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import basic_func
from null_behaviors import show_null_trajectory

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from math import pi
import os, sys
import math

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
pd.options.display.max_rows = 101


def _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted):
    arc_xy = show_null_trajectory.find_arc_xy_rotated(null_arc_info_for_the_point['center_x'].item(), null_arc_info_for_the_point['center_y'].item(), 
                                                        null_arc_info_for_the_point['all_arc_radius'].item(), 
                                                        null_arc_info_for_the_point['arc_starting_angle'].item(), null_arc_info_for_the_point['arc_ending_angle'].item(), 
                                                        rotation_matrix=None, num_points=1000)

    # now, we want to find points in arc_xy that are within ff reward boundary
    arc_df = pd.DataFrame({'monkey_x': arc_xy[0], 'monkey_y': arc_xy[1]})
    target_xy = ff_real_position_sorted[null_arc_info_for_the_point['arc_ff_index'].item()]
    arc_df[['ff_x', 'ff_y']] = target_xy
    arc_df['distance_to_ff_center'] = np.sqrt((arc_df['ff_x'] - arc_df['monkey_x'])**2 + (arc_df['ff_y'] - arc_df['monkey_y'])**2)
    arc_df['id'] = np.arange(arc_df.shape[0])
    return arc_df

def _get_arc_xy_rotated(arc_df, reward_boundary_radius=25):
    target_xy = arc_df[['ff_x', 'ff_y']].iloc[0]
    arc_df_sub = arc_df[arc_df['distance_to_ff_center'] <= reward_boundary_radius].copy()
    rotation_matrix = basic_func.make_rotation_matrix(arc_df_sub['monkey_x'].iloc[0], arc_df_sub['monkey_y'].iloc[0], target_xy.iloc[0], target_xy.iloc[1])
    arc_xy_rotated = np.matmul(rotation_matrix, arc_df_sub[['monkey_x', 'monkey_y']].values.T)
    return arc_xy_rotated, rotation_matrix

def find_arc_xy_rotated_for_plotting(null_arc_info_for_the_point, ff_real_position_sorted, reward_boundary_radius=25):
    """
    This function is meant to return the arc info that's inside the reward boundary
    """

    arc_df = _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted)
    arc_xy_rotated, _ = _get_arc_xy_rotated(arc_df, reward_boundary_radius=reward_boundary_radius)

    x0 = arc_xy_rotated[0, 0]
    y0 = arc_xy_rotated[1, 0]

    return arc_xy_rotated, x0, y0


def find_arc_xy_rotated_for_plotting2(null_arc_info_for_the_point, ff_real_position_sorted, reward_boundary_radius=25):
    """
    The difference between this function and find_arc_xy_rotated_for_plotting is that this function is meant to return the arc info that's not just inside the reward boundary, 
    but also the points that are before entering the reward boundary 
    """

    arc_df = _make_arc_df(null_arc_info_for_the_point, ff_real_position_sorted)
    arc_xy_rotated, rotation_matrix = _get_arc_xy_rotated(arc_df, reward_boundary_radius=reward_boundary_radius)

    x0 = arc_xy_rotated[0, 0]
    y0 = arc_xy_rotated[1, 0]

    arc_df_sub = arc_df[arc_df['distance_to_ff_center'] <= reward_boundary_radius].copy()
    arc_df_sub2 = arc_df[arc_df['id'] <= arc_df_sub['id'].iloc[-1]].copy()

    arc_xy_rotated = np.matmul(rotation_matrix, arc_df_sub2[['monkey_x', 'monkey_y']].values.T)
    return arc_xy_rotated, x0, y0


def iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, initial_duration_before_stop=2, reward_boundary_radius=25):
    # This function will iteratively get monkey's info before the stop until it contains at least one point before entering the reward boundary
    duration_before_stop = initial_duration_before_stop
    duration_counter = 1
    monkey_sub = _get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=duration_counter)
    # if all the points are within reward_boundary, then we might not have covered all points, in which case we will use a longer time
    while monkey_sub['distance_to_ff_center'].max() < reward_boundary_radius:
        duration_counter += 1
        print(f"duration_counter: {duration_counter}")
        monkey_sub = _get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=duration_counter)
    monkey_sub = monkey_sub[monkey_sub['distance_to_ff_center'] <= reward_boundary_radius]
    return monkey_sub


def _get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=1):
    # for the 2 seconds before the stop, get the monkey's trajectory and calculate the distance to the ff center
    monkey_sub = monkey_information[monkey_information['time'].between(time-duration_before_stop * duration_counter, time)].copy()
    monkey_sub[['ff_x', 'ff_y']] = ff_real_position_sorted[ff_index]
    monkey_sub['distance_to_ff_center'] = np.sqrt((monkey_sub['ff_x'] - monkey_sub['monkey_x'])**2 + (monkey_sub['ff_y'] - monkey_sub['monkey_y'])**2)
    return monkey_sub


def find_mxy_rotated_w_ff_center_to_the_north(monkey_sub, ff_center, reward_boundary_radius=25):
    # get info to plot the monkey's trajectory into the circle. We assume the entry point at the reward boundary is the origin, 
    # and we let the ff center to toward the north

    monkey_sub2 = monkey_sub[monkey_sub['distance_to_ff_center'] <= reward_boundary_radius].copy()
    monkey_x = monkey_sub2['monkey_x'].values
    monkey_y = monkey_sub2['monkey_y'].values

    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(ff_center[1]-monkey_y[0], ff_center[0]-monkey_x[0])     
    c, s = np.cos(theta), np.sin(theta)
    # Find the rotation matrix
    rotation_matrix = np.array(((c, -s), (s, c)))

    mxy_rotated = np.matmul(rotation_matrix, np.stack((monkey_x, monkey_y)))

    x0 = mxy_rotated[0, 0]
    y0 = mxy_rotated[1, 0]

    return mxy_rotated, x0, y0


def plot_null_arc_landings_in_ff(null_arc_info,
                                ff_real_position_sorted,
                                starting_trial=1,
                                max_trials=100,
                                reward_boundary_radius=25,
                                include_arc_portion_before_entering_ff=False):

    # for each point_index & corresponding ff_index, find the point where monkey first enters the reward boundary
    # then make a polar plot, using that entry point as the reference, and let the ff center to be to the north
    # then plot the monkey's trajectory into the circle

    trial_counter=1
    fig, ax = plt.subplots()

    for i in range(starting_trial, starting_trial + null_arc_info.shape[0]):
        if include_arc_portion_before_entering_ff:
            mxy_rotated, x0, y0 = find_arc_xy_rotated_for_plotting2(null_arc_info.iloc[[i]], ff_real_position_sorted, reward_boundary_radius=25)
        else:
            mxy_rotated, x0, y0 = find_arc_xy_rotated_for_plotting(null_arc_info.iloc[[i]], ff_real_position_sorted, reward_boundary_radius=25)

        ax = show_xy_overlapped(ax, mxy_rotated, x0, y0)

        if trial_counter > max_trials:
            break
        trial_counter += 1


    ax = _make_a_circle_to_show_reward_boundary(ax, reward_boundary_radius=reward_boundary_radius, set_xy_limit=(not include_arc_portion_before_entering_ff))

    plt.show()
    return


def _make_a_circle_to_show_reward_boundary(ax, reward_boundary_radius=25, set_xy_limit=True):
    # plot a circle with radius reward_boundary_radius that centers at (0, reward_boundary_radius)
    circle = plt.Circle((0, reward_boundary_radius), reward_boundary_radius, color='b', fill=False)
    ax.add_artist(circle)
    ax.set_aspect('equal')
    
    if set_xy_limit:
        ax.set_xlim(-reward_boundary_radius, reward_boundary_radius)
        ax.set_ylim(0, reward_boundary_radius * 2)
    return ax


def plot_monkey_landings_in_ff(all_closest_point_to_capture_df,
                               monkey_information,
                               ff_real_position_sorted,
                                starting_trial=1,
                                max_trials=100,
                                reward_boundary_radius=25):

    # for each point_index & corresponding ff_index, find the point where monkey first enters the reward boundary
    # then make a polar plot, using that entry point as the reference, and let the ff center to be to the north
    # then plot the monkey's trajectory into the circle

    trial_counter=1
    fig, ax = plt.subplots()

    for index, row in all_closest_point_to_capture_df.iloc[starting_trial:].iterrows():
        #point_index = int(row.point_index)

        monkey_sub = iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, int(row.stop_ff_index), row.time, initial_duration_before_stop=2, reward_boundary_radius=reward_boundary_radius)
        if len(monkey_sub) == 0:
            print(f"no monkey_sub for ff_index {row.stop_ff_index} at time {row.time}")
            continue
        mxy_rotated, x0, y0 = find_mxy_rotated_w_ff_center_to_the_north(monkey_sub, ff_center=ff_real_position_sorted[int(row.stop_ff_index)], reward_boundary_radius=reward_boundary_radius)
        
        ax = show_xy_overlapped(ax, mxy_rotated, x0, y0)

        if trial_counter > max_trials:
            break
        trial_counter += 1

    ax = _make_a_circle_to_show_reward_boundary(ax, reward_boundary_radius=reward_boundary_radius, set_xy_limit=True)

    plt.show()
    return



def show_xy_overlapped(ax, mxy_rotated, x0, y0, reward_boundary_radius=25):
    # plot mxy_rotated
    ax.plot(mxy_rotated[0]-x0, mxy_rotated[1]-y0, alpha=0.5)
    # plot the ending point of the monkey's trajectory
    ax.plot(mxy_rotated[0, -1]-x0, mxy_rotated[1, -1]-y0, 'ro', markersize=3, alpha=0.3)
    # plot the ff center
    ax.plot(0, reward_boundary_radius, 'r*', markersize=10)
    return ax


def find_distance_and_angle_from_ff_center_to_monkey_stop(all_closest_point_to_capture_df, monkey_information, ff_real_position_sorted):
    reward_boundary_radius = 25

    list_of_distance = []
    list_of_angle = []
    for index, row in all_closest_point_to_capture_df.iterrows():

        monkey_sub = iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, int(row.stop_ff_index), row.time, initial_duration_before_stop=2, 
                                                        reward_boundary_radius=reward_boundary_radius)
        if len(monkey_sub) == 0:
            print(f"no monkey_sub for ff_index {row.stop_ff_index} at time {row.time}")
            continue
        
        mxy_rotated, x0, y0 = find_mxy_rotated_w_ff_center_to_the_north(monkey_sub, ff_center=ff_real_position_sorted[int(row.stop_ff_index)])
        stop_x = mxy_rotated[0, -1]-x0
        stop_y = mxy_rotated[1, -1]-y0
        ff_x = 0
        ff_y = reward_boundary_radius
        distance = 25 - np.sqrt((stop_x - ff_x)**2 + (stop_y - ff_y)**2)
        # calculate the angle from the ff center to the monkey's stop point ("monkey_angle" will be just to the north)
        angle = basic_func.calculate_angles_to_ff_centers(ff_x=stop_x, ff_y=stop_y, mx=ff_x, my=ff_y, m_angle=math.pi/2)
        list_of_distance.append(distance)
        list_of_angle.append(angle)
    df = pd.DataFrame({'distance': list_of_distance, 'angle': list_of_angle})
    df['angle_in_degrees'] = np.degrees(df['angle'])
    return df