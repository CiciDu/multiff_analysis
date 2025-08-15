import sys
from tkinter import font

from visualization.matplotlib_tools import plot_behaviors_utils
from visualization.animation import animation_func, animation_utils
from null_behaviors import show_null_trajectory, find_best_arc


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def animate(frame, ax, anim_monkey_info, margin, ff_dataframe_anim, flash_on_ff_dict, alive_ff_dict,
            believed_ff_dict, in_obs_ff_dict=None, plot_flash_on_ff=False, plot_eye_position=False, set_xy_limits=True, plot_time_index=False,
            show_speed_through_path_color=False, show_in_memory_ff=True, show_ff_indices=False, show_direction_through_triangle=False,
            anno_ff_indices_dict=None, pred_ff_indices_dict=None, pred_ff_colors_dict=None,
            anno_but_not_obs_ff_indices_dict=None, plot_show_null_trajectory_for_anno_ff=False,
            show_ff_with_best_aligned_arc=False, monkey_information=None, show_full_arena=False,
            circle_anno_or_pred_ff=False):
    """
    A function to be passed into matplotlib.animation.FuncAnimation to make animation

    Parameters
    ----------
    frame:
        a necessary argument for matplotlib.animation.FuncAnimation
    ax: obj
        a matplotlib axes object  
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes 
    margin: num
        the plot margins on four sides
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_position_during_this_trial: np.array
        containing the locations of all alive fireflies in a given duration
    flash_on_ff_dict: dict
        contains the indices of the fireflies that have been flashing at each time point
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative
    plot_eye_position: bool
        whether to plot the eye position of the monkey/agent
    set_xy_limits: bool
        whether to set the x and y limits of the axes
    rotated: bool
        whether to rotated the animation information of the monkey/agent
    plot_show_null_trajectory_for_anno_ff: bool
        if True, then anno_ff_indices_dict must be provided

    Returns
    -------
    ax: obj
        a matplotlib axes object 

    """

    point_index = anim_monkey_info['anim_indices'][frame]
    alive_ff = alive_ff_dict[point_index]
    relevant_ff = ff_dataframe_anim[ff_dataframe_anim['point_index'] == point_index].copy(
    )
    visible_ffs = relevant_ff[relevant_ff['visible'] == 1]

    ax.cla()
    ax.axis('off')

    # Plot the arena
    circle_theta = np.arange(0, 2*pi, 0.01)
    ax.plot(np.cos(circle_theta)*1000 -
            anim_monkey_info['x0'], np.sin(circle_theta)*1000-anim_monkey_info['y0'])

    # Plot fireflies
    ax.scatter(alive_ff[:, 0], alive_ff[:, 1], alpha=0.7, c="gray", s=20)

    if plot_flash_on_ff:
        flashing_on_ff = flash_on_ff_dict[point_index]
        ax.scatter(flashing_on_ff[:, 0],
                   flashing_on_ff[:, 1], alpha=1, c="red", s=30)

    # Plot trajectory
    ax = animation_utils.plot_trajectory_for_animation(
        anim_monkey_info, frame, show_speed_through_path_color, ax)

    if plot_eye_position:
        if (abs(anim_monkey_info['gaze_world_x'][frame]) < 1200) & (abs(anim_monkey_info['gaze_world_y'][frame]) < 1200):
            ax.scatter(anim_monkey_info['gaze_world_x'][frame],
                       anim_monkey_info['gaze_world_y'][frame], s=30, c='darkgreen')

    # #Plot target
    # trial_num = np.where(ff_caught_T_new > anim_t[frame])[0][0]
    # ax.scatter(ff_real_position_sorted[trial_num][0], ff_real_position_sorted[trial_num][1], marker='*', c='blue', s = 130, alpha = 0.5)

    # Plot the reward boundaries of visible fireflies (if ff_x_noisy is included in the columns of visible_ffs, then plot the noisy reward boundaries as well)
    ax = animation_utils.plot_visible_ff_reward_boundary_for_animation(
        visible_ffs, ax)

    # #Plot in-memory ff
    if show_in_memory_ff:
        in_memory_ffs = relevant_ff[relevant_ff['visible'] == 0]
        # ax.scatter(in_memory_ffs.ff_x , in_memory_ffs.ff_y , alpha=1, c="green", s=30)
        # Plot reward boundaries of in-memory fireflies
        ax = animation_utils.plot_in_memory_ff_reward_boundary_for_animation(
            in_memory_ffs, ax)

    # Plot the believed positions of caught fireflies
    # if believed_ff_dict is not None:
    if len(believed_ff_dict[point_index]) > 0:
        ax.scatter(believed_ff_dict[point_index][:, 0],
                   believed_ff_dict[point_index][:, 1], alpha=1, c="magenta", s=35)

    if show_ff_indices:
        ax = animation_utils.show_ff_indices_for_animation(relevant_ff, ax)

    markers = []
    marker_labels = []
    if in_obs_ff_dict is not None:
        ax, markers, marker_labels, _ = animation_utils.plot_circles_around_ff_from_dict(
            in_obs_ff_dict, point_index, anim_monkey_info['ff_real_position_rotated'], markers, marker_labels, ax)

    # circle anno or pred ff
    if circle_anno_or_pred_ff:
        ff_real_position_rotated = anim_monkey_info['ff_real_position_rotated']
        ax, markers, marker_labels, anno_ff_non_neg_indices, anno_but_not_obs_ff_indices = animation_utils.plot_anno_ff_for_animation(
            anno_ff_indices_dict, anno_but_not_obs_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax)
        ax, markers, marker_labels = animation_utils.plot_pred_ff_for_animation(
            pred_ff_indices_dict, pred_ff_colors_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax)
        if plot_show_null_trajectory_for_anno_ff:
            ax = animation_utils.plot_show_null_trajectory_for_anno_ff_for_animation(
                ax, anno_ff_non_neg_indices, anno_but_not_obs_ff_indices, anim_monkey_info, frame, ff_real_position_rotated)

    if len(markers) > 0:
        ax.legend(markers, marker_labels, scatterpoints=1, bbox_to_anchor=(
            0.8, 0.9), loc=2, borderaxespad=0, fontsize=12)

    if show_ff_with_best_aligned_arc:
        pass

    if show_direction_through_triangle:
        ax = animation_utils.plot_triangle_to_show_direction_for_animation(
            anim_monkey_info, frame, ax)

    if plot_time_index:
        ax = animation_utils.plot_time_index_for_animation(
            anim_monkey_info, frame, ax)

    if set_xy_limits:
        if show_full_arena:
            ax.set_xlim(-1100-anim_monkey_info['x0'],
                        1100-anim_monkey_info['x0'])
            ax.set_ylim(-1100-anim_monkey_info['y0'],
                        1100-anim_monkey_info['y0'])
            # ax.set_xlim(-2000-anim_monkey_info['x0'], 2000-anim_monkey_info['x0'])
            # ax.set_ylim(-2000-anim_monkey_info['x0'], 2000-anim_monkey_info['x0'])
        else:
            mx_min, mx_max, my_min, my_max = animation_utils.find_xy_min_max_for_animation(
                anim_monkey_info, ff_dataframe_anim)
            ax = plot_behaviors_utils.set_xy_limits_for_axes(
                ax, mx_min, mx_max, my_min, my_max, margin)
    ax.set_aspect('equal')

    return ax


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


def animate_annotated(frame, ax, anim_monkey_info, margin, ff_dataframe_anim,
                      flash_on_ff_dict, alive_ff_dict, believed_ff_dict, ff_caught_T_new, annotation_info,
                      **animate_kwargs):
    """
    A function to be passed into matplotlib.animation.FuncAnimation to make animation (with annotation)

    Parameters
    ----------
    frame:
        a necessary argument for matplotlib.animation.FuncAnimation
    ax: obj
        a matplotlib axes object  
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes 
    margin: num
        the plot margins on four sides
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    flash_on_ff_dict: dict
        contains the indices of the fireflies that have been flashing at each time point
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    annotation_info: dictionary
        containing the information needed for the annotation of animation 

    Returns
    -------
    ax: obj
        a matplotlib axes object 

    """

    animate(frame, ax, anim_monkey_info, margin, ff_dataframe_anim,
            flash_on_ff_dict, alive_ff_dict, believed_ff_dict, **animate_kwargs)
    index = anim_monkey_info['anim_indices'][frame]
    time = anim_monkey_info['anim_t'][frame]
    trial_num = np.where(ff_caught_T_new > time)[0][0]
    annotation = ""
    # If the monkey has captured more than one 1 ff in a cluster
    if annotation_info['n_ff_in_a_row'][trial_num] > 1:
        annotation = annotation + \
            f"Captured {annotation_info['n_ff_in_a_row'][trial_num]} ffs in a cluster\n"
    # If the target stops being on before the monkey captures the previous firefly
    if annotation_info['visible_before_last_one_trial_dummy'][trial_num] == 1:
        annotation = annotation + "Target visible before last captre\n"
    # If the target disappears the latest among visible ffs
    if annotation_info['disappear_latest_trial_dummy'][trial_num] == 1:
        annotation = annotation + "Target disappears latest\n"
    # If the monkey ignored a closeby ff that suddenly became visible
    if annotation_info['ignore_sudden_flash_point_dummy'][index] > 0:
        annotation = annotation + "Ignored sudden flash\n"
    # If the monkey uses a few tries to capture a firefly
    if annotation_info['try_a_few_times_point_dummy'][index] > 0:
        annotation = annotation + "Try a few times to catch ff\n"
    # If during the trial, the monkey fails to capture a firefly with a few tries and moves on to capture another one
    if annotation_info['give_up_after_trying_point_dummy'][index] > 0:
        annotation = annotation + "Give up after trying\n"
    ax.text(0.5, 1.04, annotation, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
            fontsize=12, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def subset_ff_dataframe(ff_dataframe, currentTrial, num_trials):
    """
    Subset ff_dataframe into smaller dataframes that will be used in polar animation

    Parameters
    ----------
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    currentTrial: numeric
        the current trial number
    num_trials: numeric
        the number of trials to be involved

    Returns
    -------
    ff_in_time_frame: pd.dataframe
        containing various information about all visible or "in-memory" fireflies during the given trials
    ff_visible: pd.dataframe
        containing various information about all visible fireflies during the given trials
    ff_in_memory: pd.dataframe
        containing various information about all "in-memory" fireflies during the given trials

    """

    ff_in_time_frame = ff_dataframe[(ff_dataframe["target_index"] > (currentTrial-num_trials)) &
                                    (ff_dataframe["target_index"] <= currentTrial)]
    ff_in_time_frame = ff_in_time_frame[[
        'point_index', 'target_index', 'visible', 'ff_distance', 'ff_angle']]
    ff_visible = ff_in_time_frame[ff_in_time_frame['visible'] == 1][[
        'point_index', 'ff_distance', 'ff_angle']]
    ff_in_memory = ff_in_time_frame[ff_in_time_frame['visible'] == 0][[
        'point_index', 'ff_distance', 'ff_angle']]
    return ff_in_time_frame, ff_visible, ff_in_memory


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


def animate_polar(frame, ax, anim_indices, rmax, ff_in_time_frame, ff_visible, ff_in_memory):
    """
    A function to be passed into matplotlib.animation.FuncAnimation to make animation (with annotation)

    Parameters
    ----------
    frame:
        a necessary argument for matplotlib.animation.FuncAnimation
    ax: obj
        a matplotlib axes object  
    anim_indices: array
        an array of indices in reference to monkey_information, with each index corresponding to a frame in the animation
    rmax: numeric
        the radius of the polar plot
    ff_in_time_frame: pd.dataframe
        containing various information about all visible or "in-memory" fireflies during the given trials
    ff_visible: pd.dataframe
        containing various information about all visible fireflies during the given trials
    ff_in_memory: pd.dataframe
        containing various information about all "in-memory" fireflies during the given trials

    Returns
    -------
    ax: obj
        a matplotlib axes object 

    """

    ax.cla()
    ax = plot_behaviors_utils.set_polar_background_for_animation(ax, rmax)
    index = anim_indices[frame]
    all_ff_info_now = ff_in_time_frame.loc[ff_in_time_frame['point_index'] == index]
    if all_ff_info_now.shape[0] > 0:
        # Only if all_ff_info_now is not empty, can ff_visible and ff_in_memory be possibly not empty
        ff_visible_now = ff_visible.loc[ff_visible['point_index'] == index]
        if ff_visible_now.shape[0] > 0:
            ax.scatter(ff_visible_now['ff_angle'], ff_visible_now['ff_distance'],
                       marker='o', alpha=0.8, c_var=None, s=30)

        ff_in_memory_now = ff_in_memory.loc[ff_in_memory['point_index'] == index]
        if ff_in_memory_now.shape[0] > 0:
            ax.scatter(ff_in_memory_now['ff_angle'], ff_in_memory_now['ff_distance'],
                       marker='o', alpha=0.8, c="green", s=30)
    return ax
