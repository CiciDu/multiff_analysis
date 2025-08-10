import sys
from turtle import fillcolor

from data_wrangling import specific_utils
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, curvature_utils, find_best_arc
from planning_analysis.show_planning.cur_vs_nxt_ff import plot_cvn_utils
from decision_making_analysis.decision_making import plot_decision_making
from visualization.plotly_tools import plotly_preparation, plotly_for_monkey
from decision_making_analysis import trajectory_info
from pattern_discovery import make_ff_dataframe
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils

import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from math import pi
from numpy import linalg as LA

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)



def find_best_arc_df_for_ff(ff_indices, point_indexes, curv_of_traj_df, monkey_information, 
                            ff_real_position_sorted, opt_arc_stop_first_vis_bdry=True):

    ff_info = find_cvn_utils.find_ff_info(
            ff_indices,
            point_indexes,
            monkey_information,
            ff_real_position_sorted)  
    
    curvature_df = curvature_utils.make_curvature_df(ff_info, curv_of_traj_df, clean=True,
                                                          opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)
    return curvature_df, ff_info
    # mini_ff_dataframe = make_mini_ff_dataframe(
    #     ff_indices, duration, monkey_information, ff_real_position_sorted, ff_radius=10, max_distance=400)
    # temp_curvature_df = curvature_utils.make_curvature_df(mini_ff_dataframe, curv_of_traj_df, ff_radius_for_opt_arc=15, clean=True,
    #                                                       opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)
    # ff_best_arc_df, best_arc_original_columns = find_best_arc.make_best_arc_df(
    #     temp_curvature_df, monkey_information, ff_real_position_sorted)
    #return ff_best_arc_df


def plot_null_arcs_in_plotly(fig, null_arc_info, x0=0, y0=0, rotation_matrix=None, linewidth=2,
                             opacity=None, color=None, trace_name='null arc'):
    
    if len(null_arc_info) == 0:
        print('Warning: No null arc info to plot because null_arc_info is empty')

    for index in null_arc_info.index:
        arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(null_arc_info.loc[index, 'center_x'], null_arc_info.loc[index, 'center_y'], null_arc_info.loc[index, 'all_arc_radius'],
                                                                  null_arc_info.loc[index, 'arc_starting_angle'], null_arc_info.loc[index, 'arc_ending_angle'], rotation_matrix=rotation_matrix)

        arc_xy_to_plot = arc_xy_rotated.reshape(2, -1)
        if opacity is None:
            opacity = 0.8
        plot_to_add = go.Scatter(x=arc_xy_to_plot[0]-x0, y=arc_xy_to_plot[1]-y0, mode='lines',
                                 line=dict(color=color, width=linewidth), opacity=opacity,
                                 name=trace_name,
                                 hoverinfo='name',
                                 showlegend=True)
        fig.add_trace(plot_to_add)

    return fig


def update_null_arcs_in_plotly(fig, null_arc_info, x0=0, y0=0, rotation_matrix=None, trace_name='null arc'):

    for index in null_arc_info.index:
        arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(null_arc_info.loc[index, 'center_x'], null_arc_info.loc[index, 'center_y'], null_arc_info.loc[index, 'all_arc_radius'],
                                                                  null_arc_info.loc[index, 'arc_starting_angle'], null_arc_info.loc[index, 'arc_ending_angle'], rotation_matrix=rotation_matrix)

        arc_xy_to_plot = arc_xy_rotated.reshape(2, -1)

        fig.update_traces(overwrite=True, selector=dict(name=trace_name),
                          x=arc_xy_to_plot[0]-x0, y=arc_xy_to_plot[1]-y0)

    return fig


def make_mini_ff_dataframe(ff_indices, duration, monkey_information, ff_real_position_sorted, ff_radius=10, max_distance=400):
    ff_index = []
    point_index = []
    time = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    monkey_angle = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []

    for i in ff_indices:
        # Find the corresponding information in monkey_information in the given duration:
        cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
            duration, monkey_information)

        # Find distances to ff
        distances_to_ff = LA.norm(
            np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis=1)
        angles_to_ff = specific_utils.calculate_angles_to_ff_centers(
            ff_x=ff_real_position_sorted[i, 0], ff_y=ff_real_position_sorted[i, 1], mx=cum_mx, my=cum_my, m_angle=cum_angle)
        angles_to_boundaries = specific_utils.calculate_angles_to_ff_boundaries(
            angles_to_ff=angles_to_ff, distances_to_ff=distances_to_ff, ff_radius=ff_radius)
        # Find the indices of the points where the ff is both within a max_distance and valid angles
        ff_within_range_indices = np.where((np.absolute(
            angles_to_boundaries) <= 2*pi/9) & (distances_to_ff < max_distance))[0]

        # Append the values for this ff; Using list operations is faster than np.append here
        ff_index = ff_index + [i] * len(ff_within_range_indices)
        point_index = point_index + \
            cum_point_index[ff_within_range_indices].tolist()
        time = time + cum_t[ff_within_range_indices].tolist()
        ff_x = ff_x + [ff_real_position_sorted[i, 0]] * \
            len(ff_within_range_indices)
        ff_y = ff_y + [ff_real_position_sorted[i, 1]] * \
            len(ff_within_range_indices)
        monkey_x = monkey_x + cum_mx[ff_within_range_indices].tolist()
        monkey_y = monkey_y + cum_my[ff_within_range_indices].tolist()
        monkey_angle = monkey_angle + \
            cum_angle[ff_within_range_indices].tolist()
        ff_distance = ff_distance + \
            distances_to_ff[ff_within_range_indices].tolist()
        ff_angle = ff_angle + angles_to_ff[ff_within_range_indices].tolist()
        ff_angle_boundary = ff_angle_boundary + \
            angles_to_boundaries[ff_within_range_indices].tolist()

    # Now let's create a dictionary from the lists
    ff_dict = {'ff_index': ff_index, 'point_index': point_index, 'time': time,
               'ff_x': ff_x, 'ff_y': ff_y, 'monkey_x': monkey_x, 'monkey_y': monkey_y, 'monkey_angle': monkey_angle,
               'ff_distance': ff_distance, 'ff_angle': ff_angle, 'ff_angle_boundary': ff_angle_boundary}
    mini_ff_dataframe = pd.DataFrame(ff_dict)
    return mini_ff_dataframe
