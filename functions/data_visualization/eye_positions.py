
from numpy import linalg as LA
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import plotly.express as px
from math import pi
from matplotlib import rc, cm
from os.path import exists
from IPython.display import HTML
from matplotlib import animation
from functools import partial
from numpy import random




def convert_eye_positions_in_monkey_information(monkey_information):
    monkey_height = -10
    body_x = np.array(monkey_information.monkey_x)
    body_y = np.array(monkey_information.monkey_y)
    body_theta = pi/2 - np.array(monkey_information.monkey_angles)

    # left eye
    ver_theta = np.array(monkey_information.LDz)
    hor_theta = np.array(monkey_information.LDy)*pi/180
    ver_theta = ver_theta*pi/180
    gaze_monkey_view_x_l, gaze_monkey_view_y_l, gaze_world_x_l, gaze_world_y_l \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, body_theta, monkey_height, body_x, body_y)
    
    # right eye
    ver_theta = np.array(monkey_information.RDz)
    hor_theta = np.array(monkey_information.RDy)*pi/180
    ver_theta = ver_theta*pi/180
    gaze_monkey_view_x_r, gaze_monkey_view_y_r, gaze_world_x_r, gaze_world_y_r \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, body_theta, monkey_height, body_x, body_y)

    # average the two eyes
    gaze_monkey_view_x = (gaze_monkey_view_x_l + gaze_monkey_view_x_r)/2
    gaze_monkey_view_y = (gaze_monkey_view_y_l + gaze_monkey_view_y_r)/2
    gaze_world_x = (gaze_world_x_l + gaze_world_x_r)/2
    gaze_world_y = (gaze_world_y_l + gaze_world_y_r)/2
    
    
    monkey_information['gaze_monkey_view_x'] = gaze_monkey_view_x
    monkey_information['gaze_monkey_view_y'] = gaze_monkey_view_y
    monkey_information['gaze_world_x'] = gaze_world_x
    monkey_information['gaze_world_y'] = gaze_world_y
    return monkey_information



def apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, body_theta, monkey_height, body_x, body_y):

    theta_to_north = hor_theta + body_theta
    theta_to_north = (theta_to_north)%(2*pi)
    # Make the range of theta_to_north between [0, 2*pi)
    theta_to_north[theta_to_north < 0] = theta_to_north[theta_to_north < 0] + 2*pi
    inside_tan = theta_to_north.copy()


    # lolll actually this part does not matter cause it's tan^2 anyways
    # 3rd quadrant
    indices = np.where((theta_to_north > pi) & (theta_to_north <= 3*pi/2))[0]
    inside_tan[indices] = - pi - inside_tan[indices]
    # 4th quadrant
    indices = np.where((theta_to_north > pi/2) & (theta_to_north <= pi))[0]
    inside_tan[indices] = pi - inside_tan[indices]


    denominator = (np.tan(inside_tan)**2 + 1)
    numerator_component = 1/np.tan(ver_theta)**2 - np.tan(inside_tan)**2
    numerator = numerator_component * monkey_height**2
    gaze_monkey_view_y = np.sqrt(numerator/denominator)

    # based on hor_theta, we can know the direction of gaze_monkey_viewy
    # if hor_theta is positive, then gaze_monkey_viewx is positive; vice versa
    gaze_monkey_view_x = np.sqrt((np.tan(inside_tan))**2*(monkey_height**2 + gaze_monkey_view_y**2))


    # 4th quadrant
    indices = np.where((theta_to_north > pi/2) & (theta_to_north <= pi))[0]
    gaze_monkey_view_y[indices] = -gaze_monkey_view_y[indices]

    # 3rd quadrant
    indices = np.where((theta_to_north > pi) & (theta_to_north <= 3*pi/2))[0]
    gaze_monkey_view_x[indices] = -gaze_monkey_view_x[indices]
    gaze_monkey_view_y[indices] = -gaze_monkey_view_y[indices]

    # 2nd quadrant
    indices = np.where(theta_to_north > 3*pi/2)[0]
    gaze_monkey_view_x[indices] = -gaze_monkey_view_x[indices]

    gaze_world_x = body_x + gaze_monkey_view_x
    gaze_world_y = body_y + gaze_monkey_view_y
    return gaze_monkey_view_x, gaze_monkey_view_y, gaze_world_x, gaze_world_y






def find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix):
    monkey_subset = monkey_information[(monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1])]
    ver_theta = np.array(monkey_subset.LDz)
    
    gaze_world_x = np.array(monkey_subset['gaze_world_x'])
    gaze_world_y = np.array(monkey_subset['gaze_world_y'])
    gaze_world_xy = np.stack((gaze_world_x, gaze_world_y), axis=1)
    gaze_world_r = LA.norm(gaze_world_xy, axis=1)
    gaze_world_xy_rotate = gaze_world_xy.T
    gaze_world_xy_rotate = np.matmul(rotation_matrix, gaze_world_xy_rotate)

    gaze_monkey_view_x = np.array(monkey_subset['gaze_monkey_view_x'])
    gaze_monkey_view_y = np.array(monkey_subset['gaze_monkey_view_y'])
    gaze_monkey_view_xy = np.stack((gaze_monkey_view_x, gaze_monkey_view_y), axis=1).T
    gaze_monkey_view_xy_rotate = np.matmul(rotation_matrix, gaze_monkey_view_xy)

    valid_ver_theta_points = np.where(ver_theta < 0)[0]
    within_arena_points = np.where(gaze_world_r < 1000)[0]
    not_nan_indices = np.where(np.isnan(gaze_world_x)==False)[0]
    overall_valid_indices = np.intersect1d(np.intersect1d(valid_ver_theta_points, within_arena_points), not_nan_indices)

    return gaze_world_xy_rotate, gaze_monkey_view_xy_rotate, overall_valid_indices




# def find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix):
#     monkey_subset = monkey_information[(monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1])]
#     ver_theta = np.array(monkey_subset.LDz)
#     hor_theta = np.array(monkey_subset.LDy)*pi/180
#     ver_theta = ver_theta*pi/180
#     monkey_height = -10
#     body_x = np.array(monkey_subset.monkey_x)
#     body_y = np.array(monkey_subset.monkey_y)
#     body_theta = pi/2 - np.array(monkey_subset.monkey_angles)

#     gaze_monkey_view_x, gaze_monkey_view_y, gaze_world_x, gaze_world_y = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, body_theta, monkey_height, body_x, body_y)
#     gaze_world_xy = np.stack((gaze_world_x, gaze_world_y), axis=1)
#     gaze_world_r = LA.norm(gaze_world_xy, axis=1)
#     gaze_world_xy = gaze_world_xy
#     gaze_world_xy_rotate = gaze_world_xy.T
#     gaze_world_xy_rotate = np.matmul(rotation_matrix, gaze_world_xy_rotate)

#     valid_ver_theta_points = np.where(ver_theta < 0)[0]
#     within_arena_points = np.where(gaze_world_r < 1000)[0]
#     not_nan_indices = np.where(np.isnan(gaze_world_x)==False)[0]
#     overall_valid_indices = np.intersect1d(np.intersect1d(valid_ver_theta_points, within_arena_points), not_nan_indices)

#     gaze_monkey_view_xy = np.stack((gaze_monkey_view_x, gaze_monkey_view_y), axis=1).T
#     gaze_monkey_view_xy_rotate = np.matmul(rotation_matrix, gaze_monkey_view_xy)

#     return gaze_world_xy_rotate, gaze_monkey_view_xy_rotate, overall_valid_indices




def find_eye_world_speed(gaze_monkey_view_xy_rotate, cum_t, overall_valid_indices):
    gaze_x = gaze_monkey_view_xy_rotate[0]
    gaze_y = gaze_monkey_view_xy_rotate[1]

    delta_x = np.diff(gaze_x[overall_valid_indices])
    delta_y = np.diff(gaze_y[overall_valid_indices])
    delta_t = np.diff(cum_t[overall_valid_indices])
    
    delta_position = LA.norm(np.array([delta_x, delta_y]), axis=0)
    eye_world_speed = delta_position/delta_t
    eye_world_speed = np.append(eye_world_speed[0], eye_world_speed)

    return eye_world_speed



def plot_eye_world_speed_vs_monkey_speed(gaze_monkey_view_xy_rotate, cum_t, overall_valid_indices, monkey_information):
    
    eye_world_speed = find_eye_world_speed(gaze_monkey_view_xy_rotate, cum_t, overall_valid_indices)
    eye_world_speed[eye_world_speed > 1000] = 1000
    cum_t_to_plot = cum_t - cum_t[0]
    relevant_monkey_info = monkey_information[monkey_information.monkey_t.isin(cum_t)]
    corresponding_monkey_speed = relevant_monkey_info.monkey_speed.values
    corresponding_monkey_dw = relevant_monkey_info.monkey_dw.values
    

    legend_labels = {1:"Eye Speed",
              2:'Monkey linear speed',
              3:'Monkey angular speed'}

    ylabels = {1:"Speed of eye position in the world",
              2:'Monkey linear speed',
              3:'Monkey angular speed'}

    colors = {1:'gold',
              2:'royalblue',
              3:'indianred'}

    fig = plt.figure(figsize=(6.5, 4), dpi=125)
    ax = fig.add_subplot()
    fig.subplots_adjust(right=0.75)

    i = 1 # eye speed
    p1, = ax.plot(cum_t_to_plot[overall_valid_indices], eye_world_speed, color=colors[i], label=legend_labels[i])
    ax.set_xlabel('Time (s)', fontsize = 12)
    ax.set_ylabel(ylabels[i], color='darkgoldenrod', fontsize=10)

    # Change the last label of the ytick labels for eye speed
    yticks = ax.get_yticks()
    yticks = [ytick for ytick in yticks if (ytick <= 1000)]
    ytick_labels = [str(int(ytick)) for ytick in yticks]
    ytick_labels[-1] = ytick_labels[-1] + '+'
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)


    i = 2 # monkey linear speed
    ax2=ax.twinx()
    p2, = ax2.plot(cum_t_to_plot, corresponding_monkey_speed, color=colors[i], label=legend_labels[i])
    ax2.set_ylabel(ylabels[i], color=colors[i], fontsize=13)


    i = 3 # monkey angular speed
    ax3=ax.twinx()
    p3, = ax3.plot(cum_t_to_plot, corresponding_monkey_dw, color=colors[i], label=legend_labels[i])
    ax3.set_ylabel(ylabels[i], color=colors[i], fontsize=13)

    # Offset the right spine of ax3
    ax3.spines.right.set_position(("axes", 1.2))


    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors='darkgoldenrod', **tkw)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], fontsize=7, bbox_to_anchor=(0., -0.35, 0.5, .102), loc="lower left")


    plt.title('Eye Speed vs. Monkey Speed', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close
