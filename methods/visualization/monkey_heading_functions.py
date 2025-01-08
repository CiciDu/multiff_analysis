
import sys
from data_wrangling import specific_utils
from visualization.animation import animation_func, animation_utils
from null_behaviors import curvature_utils, curv_of_traj_utils
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.GUAT import GUAT_and_TAFT
from decision_making_analysis import trajectory_info
from visualization.plotly_tools import plotly_for_monkey

import numpy as np
import math
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from numpy import linalg as LA
import math
from math import pi
import seaborn as sns
import plotly.graph_objects as go


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





def find_mheading_in_xy(traj_point_index_2d, monkey_information):
    # if the dimension of traj_point_index_2d is 1, then make it 2d
    if len(traj_point_index_2d.shape) == 1:
        traj_point_index_2d = traj_point_index_2d.reshape(-1, 1)
    num_time_points_for_trajectory = traj_point_index_2d.shape[1]
    traj_x_2d = monkey_information['monkey_x'].values[traj_point_index_2d].reshape(-1, num_time_points_for_trajectory)
    traj_y_2d = monkey_information['monkey_y'].values[traj_point_index_2d].reshape(-1, num_time_points_for_trajectory)
    monkey_angle_2d = monkey_information['monkey_angle'].values[traj_point_index_2d].reshape(-1, num_time_points_for_trajectory)
    left_end_x, left_end_y, right_end_x, right_end_y = animation_utils.find_triangle_to_show_direction(traj_x_2d, traj_y_2d, monkey_angle_2d)

    mheading_in_xy = {
        'traj_x': traj_x_2d,
        'traj_y': traj_y_2d,
        'left_end_x': left_end_x,
        'left_end_y': left_end_y,
        'right_end_x': right_end_x,
        'right_end_y': right_end_y,
        'monkey_angle': monkey_angle_2d
    }
    return mheading_in_xy


def find_mheading_at_two_ends_of_traj_of_points(all_point_index, curv_of_traj_df, monkey_information):
    curv_of_traj_df = curv_of_traj_df.copy().set_index('point_index')
    traj_point_index_2d = curv_of_traj_df.loc[all_point_index, ['min_point_index', 'max_point_index']].values
    monkey_angle_2d = curv_of_traj_df.loc[all_point_index, ['initial_monkey_angle', 'final_monkey_angle']].values
    mheading_in_xy = find_mheading_in_xy(traj_point_index_2d, monkey_information)
    if np.any(monkey_angle_2d != mheading_in_xy['monkey_angle']):
        raise ValueError('monkey_angle_2d != mheading_in_xy["monkey_angle"].values')
    return mheading_in_xy


def find_mheading_in_xy_for_null_curv(arc_df):
    start_point_x_array = np.cos(arc_df.arc_starting_angle)*arc_df.all_arc_radius + arc_df.center_x
    start_point_y_array = np.sin(arc_df.arc_starting_angle)*arc_df.all_arc_radius + arc_df.center_y
    end_point_x_array = np.cos(arc_df.arc_ending_angle)*arc_df.all_arc_radius + arc_df.center_x
    end_point_y_array = np.sin(arc_df.arc_ending_angle)*arc_df.all_arc_radius + arc_df.center_y

    traj_x_2d = np.stack([start_point_x_array.values, end_point_x_array.values]).T
    traj_y_2d = np.stack([start_point_y_array.values, end_point_y_array.values]).T

    ff_direction = arc_df.all_arc_end_direction.values
    monkey_angle_2d = np.vstack([arc_df.arc_starting_angle.values + math.pi/2, arc_df.arc_ending_angle.values + math.pi/2]).T
    monkey_angle_2d[ff_direction < 0] = monkey_angle_2d[ff_direction < 0] - math.pi

    left_end_x, left_end_y, right_end_x, right_end_y = animation_utils.find_triangle_to_show_direction(traj_x_2d, traj_y_2d, monkey_angle_2d)

    mheading_in_xy = {
        'traj_x': traj_x_2d,
        'traj_y': traj_y_2d,
        'left_end_x': left_end_x,
        'left_end_y': left_end_y,
        'right_end_x': right_end_x,
        'right_end_y': right_end_y,
        'monkey_angle': monkey_angle_2d
    }
    return mheading_in_xy


def add_monkey_heading_info_to_curv_of_traj_df(curv_of_traj_df, monkey_information):
    traj_point_index_2d = curv_of_traj_df.loc[:, ['point_index_lower_end', 'point_index_upper_end']].values
    mheading_in_xy = find_mheading_in_xy(traj_point_index_2d, monkey_information)
    for key in mheading_in_xy.keys():
        curv_of_traj_df[[key + '_0', key + '_1']] = mheading_in_xy[key]




def find_current_mheading_for_the_point(mheading, i):
    if mheading is None:
        raise ValueError('mheading cannot be None if show_direction_of_monkey_on_trajectory is True')
    current_mheading = mheading.copy()
    # for var in ['left_end_r', 'left_end_theta', 'right_end_r', 'right_end_theta', 'traj_r', 'traj_theta']:
    #     current_mheading[var] = current_mheading[var][i]
    for var in mheading.keys():
        current_mheading[var] = current_mheading[var][i]
    return current_mheading



def find_mheading_rotated(mheading, R):
    traj_x, traj_y = mheading['traj_x'], mheading['traj_y']
    right_end_x, right_end_y = mheading['right_end_x'], mheading['right_end_y']
    left_end_x, left_end_y = mheading['left_end_x'], mheading['left_end_y']

    # if mheading is a pd.dataframe
    if isinstance(traj_x, pd.core.series.Series):
        traj_x = traj_x.values
        traj_y = traj_y.values
        right_end_x = right_end_x.values
        right_end_y = right_end_y.values
        left_end_x = left_end_x.values
        left_end_y = left_end_y.values

    traj_xy_rotated = np.matmul(R, np.stack([traj_x.reshape(-1), traj_y.reshape(-1)]))
    right_end_xy_rotated = np.matmul(R, np.stack([right_end_x.reshape(-1), right_end_y.reshape(-1)]))
    left_end_xy_rotated = np.matmul(R, np.stack([left_end_x.reshape(-1), left_end_y.reshape(-1)]))

    triangle_x_rotated = np.stack([left_end_xy_rotated[0,:], traj_xy_rotated[0,:], right_end_xy_rotated[0,:]]).transpose()
    triangle_y_rotated = np.stack([left_end_xy_rotated[1,:], traj_xy_rotated[1,:], right_end_xy_rotated[1,:]]).transpose()

    # shape of triangle_x_rotated and triangle_y_rotated: (num_points, 3)
    return triangle_x_rotated, triangle_y_rotated



def plot_triangles_to_show_monkey_heading_in_xy_in_matplotlib(ax, triangle_df, point_index=None, x0=0, y0=0, color=None, linewidth=3.2):
    if point_index is not None:
        triangle_df = triangle_df[triangle_df['point_index']==point_index]
    triangle_x_rotated = triangle_df[['x_0', 'x_1', 'x_2']].values
    triangle_y_rotated = triangle_df[['y_0', 'y_1', 'y_2']].values

    overall_color = color
    varing_colors = sns.color_palette("tab10", 11)[2:12]
    for j in range(triangle_x_rotated.shape[0]):
        if overall_color is not None:
            color = overall_color
        else:
            color = np.append(varing_colors[j % 9], 0.5)
    
        ax.plot(triangle_x_rotated[j]-x0, triangle_y_rotated[j]-y0, c=color, linewidth = linewidth, alpha=0.8, zorder=3)
    return ax


def get_triangle_df_for_the_point_from_mheading_for_all_counted_points(mheading, i, R):
    current_mheading = find_current_mheading_for_the_point(mheading, i)
    triangle_df = turn_mheading_into_triangle_df(current_mheading, R)
    return triangle_df


def plot_triangles_to_show_monkey_heading_in_xy_in_plotly(fig, triangle_df, trace_name_prefix=None, color='red', point_index=None, linewidth=2):
    if point_index is not None:
        triangle_df = triangle_df[triangle_df['point_index']==point_index]
    triangle_x_rotated = triangle_df[['x_0', 'x_1', 'x_2']].values
    triangle_y_rotated = triangle_df[['y_0', 'y_1', 'y_2']].values

    for j in range(triangle_x_rotated.shape[0]):
        if trace_name_prefix is None:
            name = 'triangle_to_show_monkey_heading_' + str(j)
        else:
            name = trace_name_prefix + str(j)
        fig.add_trace(
            go.Scatter(x=triangle_x_rotated[j], y=triangle_y_rotated[j],
                    mode='lines',
                    name=name,
                    hoverinfo='name',
                    showlegend=False,
                    line=dict(color=color,
                              width=linewidth)),
            )
    return fig



def update_monkey_heading_in_monkey_plot(fig, triangle_df, trace_name_prefix=None, point_index=None):
    if point_index is not None:
        triangle_df = triangle_df[triangle_df['point_index']==point_index]
    triangle_x_rotated = triangle_df[['x_0', 'x_1', 'x_2']].values
    triangle_y_rotated = triangle_df[['y_0', 'y_1', 'y_2']].values

    for j in range(triangle_x_rotated.shape[0]):
        if trace_name_prefix is None:
            name = 'triangle_to_show_monkey_heading_' + str(j)
        else:
            name = trace_name_prefix + str(j)
        fig.update_traces(overwrite=True,
                        x=triangle_x_rotated[j], y=triangle_y_rotated[j],
                        selector=dict(name=name))

    return fig



def find_mheading_dict_from_curv_of_traj(curv_of_traj_df):
    current_mheading = dict()
    for var in ['traj_x', 'traj_y', 'left_end_x', 'left_end_y', 'right_end_x', 'right_end_y', 'monkey_angle']:
        current_mheading[var] = curv_of_traj_df[[var + '_0', var + '_1']].values
        
    return current_mheading


def find_mheading_dict_from_mheading_before_stop(mheading_before_stop):
    current_mheading = dict()
    for var in ['traj_x', 'traj_y', 'left_end_x', 'left_end_y', 'right_end_x', 'right_end_y', 'monkey_angle']:
        current_mheading[var] = mheading_before_stop[var]
    return current_mheading


def turn_dict_into_df(dict):
    df = pd.DataFrame([])
    for key, value in dict.items():
        temp_dict = {}
        for i in range(value.shape[1]):
            temp_dict[key+'_'+str(i)] = value[:,i]
        df = pd.concat([df, pd.DataFrame(temp_dict)], axis=1)
    return df


def find_triangle_info_in_form_of_df(triangle_x_rotated, triangle_y_rotated):
    triangle_dict = dict(x=triangle_x_rotated, y=triangle_y_rotated)
    triangle_df = turn_dict_into_df(triangle_dict)
    return triangle_df


def turn_mheading_into_triangle_df(mheading, R, point_index=None):
    triangle_x_rotated, triangle_y_rotated = find_mheading_rotated(mheading, R)
    triangle_df = find_triangle_info_in_form_of_df(triangle_x_rotated, triangle_y_rotated)
    if point_index is not None:
        if len(point_index) < triangle_x_rotated.shape[0]:
            point_index = np.stack([point_index, point_index]).T.reshape(-1)
        triangle_df['point_index'] = point_index
    return triangle_df




# ==================  Below are functions for plotting in polar coordinates ==================





def find_all_mheading_components_in_polar(monkey_information, time_all, time_range_of_trajectory, num_time_points_for_trajectory, return_packed_info=False):
    # Get info to plot triangles to show monkey directions
    traj_time_2d, trajectory_data_dict = trajectory_info.find_trajectory_data(time_all, monkey_information, time_range_of_trajectory = time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
    traj_x_2d = trajectory_data_dict['monkey_x']
    traj_y_2d = trajectory_data_dict['monkey_y']
    monkey_angle_2d = trajectory_data_dict['monkey_angle']  
    monkey_indices = np.searchsorted(monkey_information['time'], time_all)  
    traj_distances, traj_angles, monkey_angle_on_trajectory_relative_to_the_current_north = trajectory_info.find_monkey_info_on_trajectory_relative_to_origin(monkey_indices, monkey_information, traj_x_2d, traj_y_2d, monkey_angle_2d, num_time_points_for_trajectory=num_time_points_for_trajectory)
    left_end_r, left_end_theta, right_end_r, right_end_theta = find_coordinates_of_triangles_for_monkey_heading_in_polar(traj_distances, traj_angles, monkey_angle_on_trajectory_relative_to_the_current_north)

    if return_packed_info:
        mheading = {'traj_r': traj_distances, 'traj_theta': traj_angles,
                    'left_end_r': left_end_r, 'left_end_theta': left_end_theta,
                    'right_end_r': right_end_r, 'right_end_theta': right_end_theta}
        return mheading
    else:
        return traj_distances, traj_angles, left_end_r, left_end_theta, right_end_r, right_end_theta



def find_coordinates_of_triangles_for_monkey_heading_in_polar(traj_distances, traj_angles, monkey_angle_on_trajectory_relative_to_the_current_north):
    traj_x, traj_y = animation_utils.change_polar_to_xy(traj_distances, traj_angles)
    left_end_x, left_end_y, right_end_x, right_end_y = animation_utils.find_triangle_to_show_direction(traj_x, traj_y, monkey_angle_on_trajectory_relative_to_the_current_north)
    # now, change the end points back to polar
    left_end_r, left_end_theta = animation_utils.change_xy_to_polar(left_end_x, left_end_y)
    right_end_r, right_end_theta = animation_utils.change_xy_to_polar(right_end_x, right_end_y)
    return left_end_r, left_end_theta, right_end_r, right_end_theta


def plot_triangles_to_show_monkey_headings_in_polar(ax, current_mheading):
    if current_mheading is None:
        raise ValueError('current_mheading cannot be None if show_direction_of_monkey_on_trajectory is True')
    current_left_end_r, current_left_end_theta = current_mheading['left_end_r'], current_mheading['left_end_theta']
    current_right_end_r, current_right_end_theta = current_mheading['right_end_r'], current_mheading['right_end_theta']
    current_traj_r, current_traj_theta = current_mheading['traj_r'], current_mheading['traj_theta']
    
    varing_colors = sns.color_palette("tab10", 11)[2:12]
    for j in range(len(current_traj_r)):
        color = np.append(varing_colors[j % 9], 0.5)
        ax.plot(np.array([current_traj_theta[j] , current_left_end_theta[j]]), np.array([current_traj_r[j], current_left_end_r[j]]), c=color, linewidth = 1.2, alpha=0.8, zorder=1)
        ax.plot(np.array([current_traj_theta[j] , current_right_end_theta[j]]), np.array([current_traj_r[j], current_right_end_r[j]]), c=color, linewidth = 1.2, alpha=0.8, zorder=1)
    return ax

