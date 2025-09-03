
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, find_best_arc
from matplotlib.lines import Line2D
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from math import pi
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def prepare_for_animation(ff_dataframe, ff_caught_T_new, ff_life_sorted, ff_believed_position_sorted, ff_real_position_sorted,
                          ff_flash_sorted, monkey_information, duration=None, currentTrial=None, num_trials=None, k=1, rotated=True,
                          max_duration=30, min_duration=1):
    """
    Prepare for animation
    """

    if duration is None:
        if num_trials > currentTrial:
            raise ValueError("num_trials must be smaller than currentTrial")
        if currentTrial >= len(ff_caught_T_new):
            currentTrial = len(ff_caught_T_new)-1
            num_trials = min(2, len(ff_caught_T_new))
        duration = [ff_caught_T_new[currentTrial-num_trials],
                    ff_caught_T_new[currentTrial]]

    # If the duration is too long
    if max_duration is not None:
        if (duration[1]-duration[0]) > max_duration:
            duration = [duration[1]-30, duration[1]]
            print(
                "The duration is too long. The animation will be shown for the last 30 seconds.")

    # If the duration is too short
    if min_duration is not None:
        if (duration[1]-duration[0]) < min_duration:
            duration = [duration[1]-1, duration[1]]
            print(
                "The duration is too short. The animation will be shown for the last 1 second.")

    if currentTrial is None:
        try:
            earlier_trials = np.where(ff_caught_T_new <= duration[1])[0]
            if len(earlier_trials) > 0:
                currentTrial = earlier_trials[-1]
            else:
                currentTrial = 1
            num_trials = currentTrial - \
                np.where(ff_caught_T_new > duration[0])[0][0]
        except Exception as e:
            print(
                f'CurrentTrial and num_trials are set to be None because of the following error: {e}')
            currentTrial = None
            num_trials = None

    cum_pos_index = np.where((monkey_information['time'] > duration[0]) &
                             (monkey_information['time'] <= duration[1]))[0]

    anim_monkey_info = make_anim_monkey_info(
        monkey_information, cum_pos_index, k=k)
    ff_dataframe_anim = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (
        ff_dataframe['time'] <= duration[1])].copy()
    if rotated:
        R, theta = plot_behaviors_utils.find_rotation_matrix(
            anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'], also_return_angle=True)
        anim_monkey_info, x0, y0 = rotated_anim_monkey_info(
            anim_monkey_info, R)
        anim_monkey_info['anim_angle'] = anim_monkey_info['anim_angle'] + theta
        ff_dataframe_anim.loc[:, ['ff_x_rotated', 'ff_y_rotated']] = (
            R @ ff_dataframe_anim[['ff_x', 'ff_y']].values.T).T - np.array([x0, y0])
        ff_real_position_rotated = (
            R @ ff_real_position_sorted.T).T - np.array([x0, y0])
        if "ff_x_noisy" in ff_dataframe_anim.columns:
            ff_dataframe_anim.loc[:, ['ff_x_noisy_rotated', 'ff_y_noisy_rotated']] = (
                R @ ff_dataframe_anim[['ff_x_noisy', 'ff_y_noisy']].values.T).T - np.array([x0, y0])
    else:
        R = np.eye(2)
        x0, y0 = 0, 0
        anim_monkey_info['x0'], anim_monkey_info['y0'] = 0, 0
        ff_real_position_rotated = ff_real_position_sorted
    anim_monkey_info['rotation_matrix'] = R

    flash_on_ff_dict = match_points_to_flash_on_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], duration, ff_flash_sorted,
                                                             ff_life_sorted, ff_real_position_sorted, rotation_matrix=R, x0=x0, y0=y0)

    alive_ff_dict = match_points_to_alive_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], ff_caught_T_new, ff_life_sorted, ff_real_position_sorted,
                                                       rotation_matrix=R, x0=x0, y0=y0)

    # believed_ff_dict = None
    # if (currentTrial is not None) & (num_trials is not None):
    believed_ff_dict = match_points_to_believed_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], currentTrial, num_trials, ff_believed_position_sorted,
                                                             ff_caught_T_new, rotation_matrix=R, x0=x0, y0=y0)

    num_frames = anim_monkey_info['anim_t'].size

    anim_monkey_info['ff_real_position_rotated'] = ff_real_position_rotated

    # plt.rcParams['figure.figsize'] = (7, 7)
    plt.rcParams['font.size'] = 15
    plt.rcParams['savefig.dpi'] = 100

    return num_frames, anim_monkey_info, flash_on_ff_dict, alive_ff_dict, believed_ff_dict, num_trials, ff_dataframe_anim


def find_xy_min_max_for_animation(anim_monkey_info, ff_dataframe_anim):
    mx_min, mx_max = anim_monkey_info['xmin'], anim_monkey_info['xmax']
    my_min, my_max = anim_monkey_info['ymin'], anim_monkey_info['ymax']
    visible_ffs = ff_dataframe_anim[ff_dataframe_anim['visible'] == 1]
    if len(visible_ffs) > 0:
        mx_min, mx_max = min(mx_min, min(visible_ffs.ff_x)), max(
            mx_max, max(visible_ffs.ff_x))
        my_min, my_max = min(my_min, min(visible_ffs.ff_y)), max(
            my_max, max(visible_ffs.ff_y))
    return mx_min, mx_max, my_min, my_max


def make_anim_monkey_info(monkey_information, cum_pos_index, k=3):
    """
    Get information of the monkey/agent as well as the limits of the axes to be used in animation

    Parameters
    ----------
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    cum_pos_index: array
        an array of indices involved in the current trajectory, in reference to monkey_information 
    k: num
        every k point in cum_pos_index will be plotted in the animation


    Returns
    -------
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes

    """
    cum_t, cum_angle = np.array(monkey_information['time'].iloc[cum_pos_index]), np.array(
        monkey_information['monkey_angle'].iloc[cum_pos_index])
    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_pos_index]), np.array(
        monkey_information['monkey_y'].iloc[cum_pos_index])
    cum_speed = np.array(
        monkey_information['speed'].iloc[cum_pos_index])
    anim_indices = cum_pos_index[0:-1:k]
    anim_t = cum_t[0:-1:k]
    anim_mx = cum_mx[0:-1:k]
    anim_my = cum_my[0:-1:k]
    anim_angle = cum_angle[0:-1:k]
    anim_speed = cum_speed[0:-1:k]

    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)
    anim_monkey_info = {"anim_indices": anim_indices, "anim_t": anim_t, "anim_angle": anim_angle, "anim_mx": anim_mx, "anim_my": anim_my,
                        "anim_speed": anim_speed, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

    if 'gaze_world_x' in monkey_information.columns:
        gaze_world_x, gaze_world_y = np.array(monkey_information['gaze_world_x'].iloc[cum_pos_index]), np.array(
            monkey_information['gaze_world_y'].iloc[cum_pos_index])
        anim_gaze_world_x = gaze_world_x[0:-1:k]
        anim_gaze_world_y = gaze_world_y[0:-1:k]
        anim_monkey_info['gaze_world_x'] = anim_gaze_world_x
        anim_monkey_info['gaze_world_y'] = anim_gaze_world_y

    return anim_monkey_info


def rotated_anim_monkey_info(anim_monkey_info, R):
    """
    rotated the animation information of the monkey/agent

    Parameters
    ----------
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes 
    R: np.array
        a rotation matrix


    Returns
    -------
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes

    """
    anim_mx, anim_my = np.array(anim_monkey_info['anim_mx']), np.array(
        anim_monkey_info['anim_my'])
    anim_mx, anim_my = R @ np.array([anim_mx, anim_my])
    x0, y0 = anim_mx[0], anim_my[0]
    anim_monkey_info['x0'], anim_monkey_info['y0'] = x0, y0
    anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'] = anim_mx-x0, anim_my-y0

    anim_monkey_info['xmin'], anim_monkey_info['xmax'] = np.min(
        anim_mx)-x0, np.max(anim_mx)-x0
    anim_monkey_info['ymin'], anim_monkey_info['ymax'] = np.min(
        anim_my)-y0, np.max(anim_my)-y0

    if 'gaze_world_x' in anim_monkey_info.keys():
        anim_monkey_info['gaze_world_x'], anim_monkey_info['gaze_world_y'] = R @ np.array(
            [anim_monkey_info['gaze_world_x'], anim_monkey_info['gaze_world_y']]) - np.array([x0, y0]).reshape(2, 1)

    return anim_monkey_info, x0, y0


# Create a dictionary of {time: [indices of fireflies that are visible], ...}
def match_points_to_flash_on_ff_positions(anim_t, anim_indices, duration, ff_flash_sorted,
                                          ff_life_sorted, ff_real_position_sorted, rotation_matrix=None, x0=0, y0=0):
    """
    Find the fireflies that are visible at each time point (for animation)

    Parameters
    ----------
    anim_t: array-like
        containing a list of time
    currentTrial: numeric
        the number of current trial 
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured


    Returns
    -------
    flash_on_ff_dict: dict
        contains the indices of the fireflies that have been flashing at each time point


    Examples
    -------
        flash_on_ff_dict = match_points_to_flash_on_ff_positions(anim_t, anim_indices, duration, ff_flash_sorted, ff_real_position_sorted)

    """
    # Find indices of fireflies that have been alive during the trial
    alive_ff_during_this_trial = np.where((ff_life_sorted[:, 1] > duration[0])
                                          & (ff_life_sorted[:, 0] < duration[1]))[0]
    flash_on_ff_dict = {}
    for i in range(len(anim_t)):
        time = anim_t[i]
        index = anim_indices[i]
        # Find indicies of fireflies that have been on at this time point
        visible_ff_indices = [ff_index for ff_index in alive_ff_during_this_trial
                              if len(np.where(np.logical_and(ff_flash_sorted[ff_index][:, 0] <= time,
                                                             ff_flash_sorted[ff_index][:, 1] >= time))[0]) > 0]
        # Store the ff indices into the dictionary with the time being the key
        visible_ff_positions = ff_real_position_sorted[visible_ff_indices]
        flash_on_ff_dict[index] = visible_ff_positions

    if rotation_matrix is not None:
        for key, item in flash_on_ff_dict.items():
            if len(item) > 0:
                flash_on_ff_dict[key] = (
                    rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return flash_on_ff_dict


def match_points_to_alive_ff_positions(anim_t, anim_indices, ff_caught_T_new, ff_life_sorted, ff_real_position_sorted,
                                       rotation_matrix=None, x0=0, y0=0):
    array_of_trial_nums = np.digitize(anim_t, ff_caught_T_new).tolist()
    alive_ff_dict = {}
    for i in range(len(anim_indices)):
        index = anim_indices[i]
        trial_num = array_of_trial_nums[i]
        current_t = anim_t[i]
        alive_ff_indices = np.where((ff_life_sorted[:, 1] >= current_t-0.1)
                                    & (ff_life_sorted[:, 0] <= current_t+0.1))[0]
        alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
        alive_ff_dict[index] = alive_ff_positions

    if rotation_matrix is not None:
        for key, item in alive_ff_dict.items():
            if len(item) > 0:
                alive_ff_dict[key] = (
                    rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return alive_ff_dict


# Create a dictionary of {time: [[believed_ff_position], [believed_ff_position2], ...], ...}
def match_points_to_believed_ff_positions(anim_t, anim_indices, currentTrial, num_trials, ff_believed_position_sorted,
                                          ff_caught_T_new, rotation_matrix=None, x0=0, y0=0):
    """
    Match the believed positions of the fireflies to the time when they are captured (for animation)

    Parameters
    ----------
    anim_t: array-like
        containing a list of time
    currentTrial: numeric
        the number of current trial 
    num_trials: numeric
        number of trials to span across when using this function
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured


    Returns
    -------
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative


    Examples
    -------
        believed_ff_dict = match_points_to_believed_ff_positions(anim_t, anim_indices, currentTrial, num_trials, ff_believed_position_sorted, ff_caught_T_new, ff_real_position_sorted)

    """

    if ff_caught_T_new is None:
        ff_caught_T_new = []

    believed_ff_dict = {}
    # For each time point:
    if (currentTrial is not None) & (num_trials is not None):
        relevant_catching_ff_time = ff_caught_T_new[currentTrial -
                                                    num_trials+1:currentTrial+1]
        relevant_caught_ff_positions = ff_believed_position_sorted[currentTrial -
                                                                   num_trials+1:currentTrial+1]
    else:
        try:
            relevant_catching_ff_time = ff_caught_T_new[ff_caught_T_new >= [
                anim_t[0]]]
            relevant_caught_ff_positions = ff_believed_position_sorted[ff_caught_T_new >= [
                anim_t[0]]]
        except IndexError:
            relevant_catching_ff_time = np.array([])
            relevant_caught_ff_positions = np.array([])
    for i in range(len(anim_t)):
        time = anim_t[i]
        index = anim_indices[i]
        already_caught_ff_positions = relevant_caught_ff_positions[
            relevant_catching_ff_time <= time]
        believed_ff_dict[index] = already_caught_ff_positions

    # # The last point
    # believed_ff_indices = [(ff_believed_position_sorted[ff]) for ff in range(currentTrial-num_trials+1, currentTrial+1)]
    # believed_ff_dict[len(anim_t)-1] = believed_ff_indices

    if rotation_matrix is not None:
        for key, item in believed_ff_dict.items():
            if len(item) > 0:
                believed_ff_dict[key] = (
                    rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return believed_ff_dict


def make_annotation_info(caught_ff_num, max_point_index, n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials,
                         ignore_sudden_flash_indices, give_up_after_trying_indices, try_a_few_times_indices):
    """
    Collect information for annotating the animation

    Parameters
    ----------
    caught_ff_num: num
        number of caught fireflies
    max_point_index: numeric
        the maximum point_index in ff_dataframe  
    n_ff_in_a_row: array
        containing one integer for each captured firefly to indicate how many fireflies have been caught in a row.
        n_ff_in_a_row[k] will denote the number of ff that the monkey has captured in a row at trial k
    visible_before_last_one_trials: array
        trial numbers that can be categorized as "visible before last one"
    disappear_latest_trials: array
        trial numbers that can be categorized as "disappear latest"
    ignore_sudden_flash_indices: array
        indices that can be categorized as "ignore sudden flash"
    give_up_after_trying_indices: array
        indices that can be categorized as "give up after trying"
    try_a_few_times_indices: array
        indices that can be categorized as "try a few times"      


    Returns
    -------
    annotation_info: dictionary
        containing the information needed for the annotation of animation 
    """

    # Convert the arrays of trial numbers or index numbers into arrays of dummy variables
    zero_array = np.zeros(caught_ff_num, dtype=int)

    visible_before_last_one_trial_dummy = zero_array.copy()
    if len(visible_before_last_one_trials) > 0:
        visible_before_last_one_trial_dummy[visible_before_last_one_trials] = 1

    disappear_latest_trial_dummy = zero_array.copy()
    if len(disappear_latest_trials) > 0:
        disappear_latest_trial_dummy[disappear_latest_trials] = 1

    ignore_sudden_flash_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(ignore_sudden_flash_indices) > 0:
        ignore_sudden_flash_point_dummy[ignore_sudden_flash_indices] = 1

    give_up_after_trying_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(give_up_after_trying_indices) > 0:
        give_up_after_trying_point_dummy[give_up_after_trying_indices] = 1

    try_a_few_times_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(try_a_few_times_indices) > 0:
        try_a_few_times_point_dummy[try_a_few_times_indices] = 1

    annotation_info = {"n_ff_in_a_row": n_ff_in_a_row, "visible_before_last_one_trial_dummy": visible_before_last_one_trial_dummy, "disappear_latest_trial_dummy": disappear_latest_trial_dummy,
                       "ignore_sudden_flash_point_dummy": ignore_sudden_flash_point_dummy, "try_a_few_times_point_dummy": try_a_few_times_point_dummy, "give_up_after_trying_point_dummy": give_up_after_trying_point_dummy}
    return annotation_info


def plot_trajectory_for_animation(anim_monkey_info, frame, show_speed_through_path_color, ax):
    if show_speed_through_path_color:
        color = plt.get_cmap('viridis')(
            anim_monkey_info['anim_speed'][:frame+1]/200)
    else:
        color = 'royalblue'
    ax.scatter(anim_monkey_info['anim_mx'][:frame+1],
               anim_monkey_info['anim_my'][:frame+1], s=15, c=color)
    return ax


def plot_visible_ff_reward_boundary_for_animation(visible_ffs, ax):
    if 'ff_x_rotated' in visible_ffs.columns:
        ff_x_column = 'ff_x_rotated'
        ff_y_column = 'ff_y_rotated'
    else:
        ff_x_column = 'ff_x'
        ff_y_column = 'ff_y'

    if "ff_x_noisy" in visible_ffs.columns:
        # plot both the real positions and the noisy positions
        if 'ff_x_noisy_rotated' in visible_ffs.columns:
            ff_x_noisy_column = 'ff_x_noisy_rotated'
            ff_y_noisy_column = 'ff_y_noisy_rotated'
        else:
            ff_x_noisy_column = 'ff_x_noisy'
            ff_y_noisy_column = 'ff_y_noisy'

        for k in range(len(visible_ffs)):
            circle = plt.Circle((visible_ffs[ff_x_column].iloc[k], visible_ffs[ff_y_column].iloc[k]),
                                25, facecolor='yellow', edgecolor='gray', alpha=0.7, zorder=1)
            ax.add_patch(circle)
            circle = plt.Circle((visible_ffs[ff_x_noisy_column].iloc[k], visible_ffs[ff_y_noisy_column].iloc[k]),
                                25, facecolor='gray', edgecolor='gray', alpha=0.5, zorder=1)
            ax.add_patch(circle)
    else:
        # plot the real positions only
        for k in range(len(visible_ffs)):
            circle = plt.Circle((visible_ffs[ff_x_column].iloc[k], visible_ffs[ff_y_column].iloc[k]),
                                25, facecolor='yellow', edgecolor='gray', alpha=0.7, zorder=1)
            ax.add_patch(circle)
    return ax


def plot_in_memory_ff_reward_boundary_for_animation(in_memory_ffs, ax):
    if 'ff_x_rotated' in in_memory_ffs.columns:
        ff_x_column = 'ff_x_rotated'
        ff_y_column = 'ff_y_rotated'
    else:
        ff_x_column = 'ff_x'
        ff_y_column = 'ff_y'
    for j in range(len(in_memory_ffs)):
        circle = plt.Circle((in_memory_ffs[ff_x_column].iloc[j], in_memory_ffs[ff_y_column].iloc[j]),
                            25, facecolor='purple', edgecolor='orange', alpha=0.3, zorder=1)
        ax.add_patch(circle)
    return ax


def show_ff_indices_for_animation(relevant_ff, ax):
    # selected_ffs = relevant_ff[relevant_ff['memory'] > 50]
    selected_ffs = relevant_ff.copy()
    if 'ff_x_rotated' in selected_ffs.columns:
        ff_x_column = 'ff_x_rotated'
        ff_y_column = 'ff_y_rotated'
    else:
        ff_x_column = 'ff_x'
        ff_y_column = 'ff_y'
    selected_ff_x = selected_ffs[ff_x_column].values
    selected_ff_y = selected_ffs[ff_y_column].values
    selected_ff_indices = selected_ffs['ff_index'].values
    for k in range(len(selected_ffs)):
        ax.annotate(
            str(selected_ff_indices[k]), (selected_ff_x[k], selected_ff_y[k]), fontsize=12)
    return ax


def find_triangle_to_show_direction(monkey_x, monkey_y, monkey_angle):
    # Plot a triangular shape to indicate the direction of the agent or monkey
    left_end_x = monkey_x + 30 * np.cos(monkey_angle + 2*pi/9)
    left_end_y = monkey_y + 30 * np.sin(monkey_angle + 2*pi/9)
    right_end_x = monkey_x + 30 * np.cos(monkey_angle - 2*pi/9)
    right_end_y = monkey_y + 30 * np.sin(monkey_angle - 2*pi/9)
    return left_end_x, left_end_y, right_end_x, right_end_y


def plot_a_triangle_to_show_direction(ax, monkey_x, monkey_y, monkey_angle):
    # Plot a triangular shape to indicate the direction of the agent or monkey
    left_end_x, left_end_y, right_end_x, right_end_y = find_triangle_to_show_direction(
        monkey_x, monkey_y, monkey_angle)

    ax.plot(np.array([monkey_x, left_end_x]), np.array(
        [monkey_y, left_end_y]), linewidth=2)
    ax.plot(np.array([monkey_x, right_end_x]), np.array(
        [monkey_y, right_end_y]), linewidth=2)
    return ax


def plot_triangle_to_show_direction_for_animation(anim_monkey_info, frame, ax):
    ax = plot_a_triangle_to_show_direction(
        ax, anim_monkey_info['anim_mx'][frame], anim_monkey_info['anim_my'][frame], anim_monkey_info['anim_angle'][frame])
    # # Plot a triangular shape to indicate the direction of the agent or monkey
    # left_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] + 2*pi/9)
    # left_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] + 2*pi/9)
    # right_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] - 2*pi/9)
    # right_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] - 2*pi/9)

    # ax.plot(np.array([anim_monkey_info['anim_mx'][frame], left_end_x]), np.array([anim_monkey_info['anim_my'][frame] , left_end_y]), linewidth = 2)
    # ax.plot(np.array([anim_monkey_info['anim_mx'][frame], right_end_x]), np.array([anim_monkey_info['anim_my'][frame] , right_end_y]), linewidth = 2)
    return ax


def change_polar_to_xy(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def change_xy_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def plot_time_index_for_animation(anim_monkey_info, frame, ax):
    index = anim_monkey_info['anim_indices'][frame]
    annotation = str(index)
    ax.text(0.02, 0.9, annotation, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
            fontsize=12, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def plot_circles_around_ff(ff_indices, ff_real_position_rotated, circle_size, edgecolor, ax, lw=2):
    ff_indices = np.array(ff_indices).reshape(-1)
    if len(ff_indices) > 0:
        for k in ff_indices:
            circle = plt.Circle((ff_real_position_rotated[k, 0], ff_real_position_rotated[k, 1]),
                                circle_size, facecolor='None', edgecolor=edgecolor, alpha=0.8, zorder=1, lw=lw)
            ax.add_patch(circle)
    return ax


def plot_anno_ff_for_animation(anno_ff_indices_dict, anno_but_not_obs_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax):
    anno_ff_non_neg_indices = np.array([])
    if anno_ff_indices_dict is not None:
        if point_index in anno_ff_indices_dict.keys():
            anno_indices = anno_ff_indices_dict[point_index]
            anno_ff_non_neg_indices = anno_indices[anno_indices >= 0]
            if len(anno_ff_non_neg_indices) > 0:
                ax = plot_circles_around_ff(
                    anno_ff_non_neg_indices, ff_real_position_rotated, circle_size=30, edgecolor='red', ax=ax)
            else:
                anno_ff_neg_indices = anno_indices[anno_indices < 0]
                ax.text(0.7, 0.95, str(anno_ff_neg_indices), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                        fontsize=18, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            if anno_but_not_obs_ff_indices_dict is not None:
                # otherwise, it will plotted below (in the next if else statement)
                if point_index not in anno_but_not_obs_ff_indices_dict.keys():
                    ax.text(0.7, 0.95, 'CB?', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                            fontsize=18, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    # CB means crossing boundary
    # marker1,_ = ax.plot([], [], 50, color='red', marker='o', alpha=1, zorder=1)
    marker1 = Line2D([0], [0], marker='o', color='r', label='Circle',
                     markerfacecolor='w', lw=0, markeredgewidth=2, markersize=15)
    markers.append(marker1)
    marker_labels.append('Annotated')

    ax, markers, marker_labels, anno_but_not_obs_ff_indices = plot_circles_around_ff_from_dict(
        anno_but_not_obs_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax, edgecolor='green', circle_size=33, legend_name='Annotated but not observed')

    # anno_but_not_obs_ff_indices = np.array([])
    # if anno_but_not_obs_ff_indices_dict is not None:
    #     if point_index in anno_but_not_obs_ff_indices_dict.keys():
    #         anno_but_not_obs_ff_indices = anno_but_not_obs_ff_indices_dict[point_index]
    #         ax = plot_circles_around_ff(anno_but_not_obs_ff_indices, ff_real_position_rotated, circle_size=33, edgecolor='green', ax=ax)

    #     marker3 = Line2D([0], [0], marker='o', color='green', label='Circle', markerfacecolor='w', lw=0, markeredgewidth=2, markersize=15)
    #     #marker3,_ = ax.plot([], [], 50, color='red', marker='o', alpha=1, zorder=1)
    #     markers.append(marker3)
    #     marker_labels.append('Annotated but not observed')

    return ax, markers, marker_labels, anno_ff_non_neg_indices, anno_but_not_obs_ff_indices


def plot_pred_ff_for_animation(pred_ff_indices_dict, pred_ff_colors_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax):

    if pred_ff_colors_dict is not None:
        try:
            edgecolor = pred_ff_colors_dict[point_index]
        except:
            edgecolor = 'blue'

    ax, markers, marker_labels, _ = plot_circles_around_ff_from_dict(
        pred_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax, edgecolor=edgecolor, circle_size=37, legend_name='Predicted')

    return ax, markers, marker_labels
    # if pred_ff_indices_dict is not None:
    #     if point_index in pred_ff_indices_dict.keys():
    #         pred_ff_indices = pred_ff_indices_dict[point_index]
    #         if pred_ff_colors_dict is not None:
    #             edgecolor = pred_ff_colors_dict[point_index]
    #         else:
    #             edgecolor = 'blue'
    #         ax = plot_circles_around_ff(pred_ff_indices, ff_real_position_rotated, circle_size=37, edgecolor=edgecolor, ax=ax)

    #     marker2 = Line2D([0], [0], marker='o', color='blue', label='Circle', markerfacecolor='w', lw=0, markeredgewidth=2, markersize=15)
    #     #marker2,_ = ax.plot([], [], 50, color='blue', marker='o', alpha=1, zorder=1)
    #     markers.append(marker2)
    #     marker_labels.append('Predicted')
    # return ax, markers, marker_labels


def plot_circles_around_ff_from_dict(ff_indices_dict,
                                     point_index,
                                     ff_real_position_rotated,
                                     markers,
                                     marker_labels,
                                     ax,
                                     edgecolor='green',
                                     circle_size=39,
                                     legend_name='In Obs',
                                     lw=1):

    in_obs_ff_indices = np.array([])
    if ff_indices_dict is not None:
        if point_index in ff_indices_dict.keys():
            in_obs_ff_indices = ff_indices_dict[point_index]
            ax = plot_circles_around_ff(in_obs_ff_indices, ff_real_position_rotated,
                                        circle_size=circle_size, edgecolor=edgecolor, ax=ax, lw=lw)

        marker2 = Line2D([0], [0], marker='o', color=edgecolor, label='Circle',
                         markerfacecolor='w', lw=0, markeredgewidth=2, markersize=15)
        # marker2,_ = ax.plot([], [], 50, color='blue', marker='o', alpha=1, zorder=1)
        markers.append(marker2)
        marker_labels.append(legend_name)
    return ax, markers, marker_labels, in_obs_ff_indices


def plot_show_null_trajectory_for_anno_ff_for_animation(ax, anno_ff_non_neg_indices, anno_but_not_obs_ff_indices, anim_monkey_info, frame, ff_real_position_rotated,
                                                        arc_color='black', line_color='black', reaching_boundary_ok=False):
    union_anno_ff_indices = np.union1d(
        anno_ff_non_neg_indices, anno_but_not_obs_ff_indices).astype(int)
    if len(union_anno_ff_indices) == 0:
        return ax

    # question: should I use ff_positions_rotated here or just ff_positions
    monkey_x = anim_monkey_info['anim_mx'][frame]
    monkey_y = anim_monkey_info['anim_my'][frame]
    monkey_xy = np.stack([monkey_x, monkey_y]).T
    monkey_angle = anim_monkey_info['anim_angle'][frame]
    ff_x = ff_real_position_rotated[union_anno_ff_indices, 0]
    ff_y = ff_real_position_rotated[union_anno_ff_indices, 1]

    if reaching_boundary_ok:
        ff_xy = find_best_arc.find_point_on_ff_boundary_with_smallest_angle_to_monkey(
            ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
        ff_x = ff_xy[:, 0]
        ff_y = ff_xy[:, 1]

    # get the null trajectory info
    ff_xy, ff_distance, ff_angle, ff_angle_boundary, arc_length, arc_radius = show_null_trajectory.find_arc_length_and_radius(
        ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
    whether_ff_behind = (np.abs(ff_angle) > math.pi/2)
    center_x, center_y, arc_starting_angle, arc_ending_angle = show_null_trajectory.find_cartesian_arc_center_and_angle_for_arc_to_center(
        monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, ff_xy, np.sign(ff_angle), whether_ff_behind=whether_ff_behind)

    # take out the ff with positive arc_radius; those are ff that will be reached by arcs instead of straight lines
    arc_ff = np.where(arc_radius > 0)[0]
    # we just need to put the information together; since the positions are already rotated, we can set rotation_matrix to be None
    arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(
        center_x[arc_ff], center_y[arc_ff], arc_radius[arc_ff], arc_starting_angle[arc_ff], arc_ending_angle[arc_ff], rotation_matrix=None)
    ax.plot(arc_xy_rotated[0], arc_xy_rotated[1],
            linewidth=2.5, color=arc_color, zorder=4)

    # then take out the ff that will be reached by straight lines
    line_ff = np.where(arc_radius == 0)[0]
    for ff in (line_ff):
        line_xy_rotated = np.stack(
            (np.array([monkey_x, ff_xy[ff][0]]), np.array([monkey_y, ff_xy[ff][1]])))
        ax.plot(line_xy_rotated[0], line_xy_rotated[1],
                linewidth=2.5, color=line_color, zorder=4)

    return ax
