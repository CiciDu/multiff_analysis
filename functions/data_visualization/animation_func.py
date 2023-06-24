#from multiff_analysis.functions.data_wrangling import find_patterns
from multiff_analysis.functions.data_visualization import plot_behaviors
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'






def make_anim_monkey_info(monkey_information, cum_indices, k = 3):
    """
    Get information of the monkey/agent as well as the limits of the axes to be used in animation

    Parameters
    ----------
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    cum_indices: array
        an array of indices involved in the current trajectory, in reference to monkey_information 
    k: num
        every k point in cum_indices will be plotted in the animation


    Returns
    -------
    anim_monkey_info: dict
        containing the index, time, position, angle, and location of the monkey at various points of time 
        to be plotted in animation, as well as the limits of the axes
    
    """
    cum_t, cum_angle = np.array(monkey_information['monkey_t'].iloc[cum_indices]), np.array(monkey_information['monkey_angles'].iloc[cum_indices])
    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_indices]), np.array(monkey_information['monkey_y'].iloc[cum_indices])
    anim_indices = cum_indices[0:-1:k]
    anim_t = cum_t[0:-1:k]
    anim_mx = cum_mx[0:-1:k]
    anim_my = cum_my[0:-1:k]
    anim_angle = cum_angle[0:-1:k]

    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)
    anim_monkey_info = {"anim_indices": anim_indices, "anim_t": anim_t, "anim_angle": anim_angle, "anim_mx": anim_mx, "anim_my": anim_my,
                        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    
    if 'gaze_world_x' in monkey_information.columns:
        gaze_world_x, gaze_world_y = np.array(monkey_information['gaze_world_x'].iloc[cum_indices]), np.array(monkey_information['gaze_world_y'].iloc[cum_indices])
        anim_gaze_world_x = gaze_world_x[0:-1:k]
        anim_gaze_world_y = gaze_world_y[0:-1:k]
        anim_monkey_info['gaze_world_x'] = anim_gaze_world_x
        anim_monkey_info['gaze_world_y'] = anim_gaze_world_y
       

    return anim_monkey_info



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
    ff_caught_T_sorted: np.array
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
    alive_ff_during_this_trial = np.where((ff_life_sorted[:,1] > duration[0])\
                                        & (ff_life_sorted[:,0] < duration[1]))[0]
    flash_on_ff_dict = {}
    for i in range(len(anim_t)):
        time = anim_t[i]
        index = anim_indices[i]
        # Find indicies of fireflies that have been on at this time point
        visible_ff_indices = [ff_index for ff_index in alive_ff_during_this_trial \
                              if len(np.where(np.logical_and(ff_flash_sorted[ff_index][:,0] <= time, \
                              ff_flash_sorted[ff_index][:,1] >= time))[0]) > 0]
        # Store the ff indices into the dictionary with the time being the key
        visible_ff_positions = ff_real_position_sorted[visible_ff_indices]
        flash_on_ff_dict[index] = visible_ff_positions

    if rotation_matrix is not None:
        for key, item in flash_on_ff_dict.items():
            if len(item) > 0:
                flash_on_ff_dict[key] = (rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return flash_on_ff_dict



def match_points_to_alive_ff_positions(anim_t, anim_indices, ff_caught_T_sorted, ff_life_sorted, ff_real_position_sorted, 
                                       rotation_matrix=None, x0=0, y0=0):
    array_of_trial_nums = np.digitize(anim_t, ff_caught_T_sorted).tolist()
    alive_ff_dict = {}
    for i in range(len(anim_indices)):
        index = anim_indices[i]
        trial_num = array_of_trial_nums[i]
        alive_ff_indices = np.where((ff_life_sorted[:,1] > ff_caught_T_sorted[trial_num-1])\
                                    & (ff_life_sorted[:, 0] < ff_caught_T_sorted[trial_num]))[0]
        alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
        alive_ff_dict[index] = alive_ff_positions


    if rotation_matrix is not None:
        for key, item in alive_ff_dict.items():
            if len(item) > 0:
                alive_ff_dict[key] = (rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return alive_ff_dict






# Create a dictionary of {time: [[believed_ff_position], [believed_ff_position2], ...], ...}
def match_points_to_believed_ff_positions(anim_t, anim_indices, currentTrial, num_trials, ff_believed_position_sorted, 
                                          ff_caught_T_sorted, rotation_matrix=None, x0=0, y0=0):
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
    ff_caught_T_sorted: np.array
        containing the time when each captured firefly gets captured


    Returns
    -------
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative


    Examples
    -------
        believed_ff_dict = match_points_to_believed_ff_positions(anim_t, anim_indices, currentTrial, num_trials, ff_believed_position_sorted, ff_caught_T_sorted, ff_real_position_sorted)

    """
    believed_ff_dict = {}
    # For each time point:
    relevant_catching_ff_time = ff_caught_T_sorted[currentTrial-num_trials+1:currentTrial+1]
    relevant_caught_ff_positions = ff_believed_position_sorted[currentTrial-num_trials+1:currentTrial+1]
    for i in range(len(anim_t)):
      time = anim_t[i]
      index = anim_indices[i]
      already_caught_ff_positions = relevant_caught_ff_positions[relevant_catching_ff_time <= time]
      believed_ff_dict[index] = already_caught_ff_positions

    # # The last point
    # believed_ff_indices = [(ff_believed_position_sorted[ff]) for ff in range(currentTrial-num_trials+1, currentTrial+1)]
    # believed_ff_dict[len(anim_t)-1] = believed_ff_indices

    if rotation_matrix is not None:
        for key, item in believed_ff_dict.items():
            if len(item) > 0:
                believed_ff_dict[key] = (rotation_matrix @ np.array(item).T).T - np.array([x0, y0])

    return believed_ff_dict







def make_annotation_info(caught_ff_num, max_point_index, n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials, \
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




def prepare_for_animation(ff_dataframe, ff_caught_T_sorted, ff_life_sorted, ff_believed_position_sorted, ff_real_position_sorted, 
                          ff_flash_sorted, monkey_information, duration=None, currentTrial=None, num_trials=None, k=1, rotate=True):
    """
    Prepare for animation
    """
    
    if duration is None:
        if num_trials > currentTrial:
            raise ValueError("num_trials must be smaller than currentTrial")
        if currentTrial >= len(ff_caught_T_sorted):
            currentTrial = len(ff_caught_T_sorted)-1
            num_trials = min(2, len(ff_caught_T_sorted))
        duration = [ff_caught_T_sorted[currentTrial-num_trials], ff_caught_T_sorted[currentTrial]]
        
    # If the duration is too long
    if (duration[1]-duration[0]) > 30:
        duration = [duration[1]-30, duration[1]]
        
    # If the duration is too short
    elif (duration[1]-duration[0]) < 0.1:
        duration = [duration[1]-1, duration[1]]

    if currentTrial is None:
        earlier_trials = np.where(ff_caught_T_sorted <= duration[1])[0]
        if len(earlier_trials) > 0:
            currentTrial = earlier_trials[-1]
        else:
            currentTrial = 1

    new_num_trials = currentTrial-np.where(ff_caught_T_sorted > duration[0])[0][0]

    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & 
                           (monkey_information['monkey_t'] <= duration[1]))[0]
    

 
    anim_monkey_info = make_anim_monkey_info(monkey_information, cum_indices, k = k)
    ff_dataframe_anim = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])].copy()
    if rotate:
        R, theta = plot_behaviors.find_rotation_matrix(anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'], also_return_angle=True)    
        anim_monkey_info, x0, y0 = rotate_anim_monkey_info(anim_monkey_info, R)
        anim_monkey_info['anim_angle'] = anim_monkey_info['anim_angle'] + theta
        ff_dataframe_anim.loc[:, ['ff_x', 'ff_y']] = (R @ ff_dataframe_anim[['ff_x', 'ff_y']].values.T).T - np.array([x0, y0])
        if "ff_x_noisy" in ff_dataframe_anim.columns:
            ff_dataframe_anim.loc[:, ['ff_x_noisy', 'ff_y_noisy']] = (R @ ff_dataframe_anim[['ff_x_noisy', 'ff_y_noisy']].values.T).T - np.array([x0, y0])
    else:
        R = None 
        anim_monkey_info['x0'], anim_monkey_info['y0'] = 0, 0
    
    
    flash_on_ff_dict = match_points_to_flash_on_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], duration, ff_flash_sorted, 
                                                                ff_life_sorted, ff_real_position_sorted, rotation_matrix=R, x0=x0, y0=y0)
    believed_ff_dict = match_points_to_believed_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], currentTrial, new_num_trials, ff_believed_position_sorted, 
                                                                ff_caught_T_sorted, rotation_matrix=R, x0=x0, y0=y0)
    alive_ff_dict = match_points_to_alive_ff_positions(anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], ff_caught_T_sorted, ff_life_sorted, ff_real_position_sorted,
                                                        rotation_matrix=R, x0=x0, y0=y0)
    num_frames = anim_monkey_info['anim_t'].size


    plt.rcParams['figure.figsize'] = (7, 7)
    plt.rcParams['font.size'] = 15
    plt.rcParams['savefig.dpi'] = 100

    
    return num_frames, anim_monkey_info, flash_on_ff_dict, alive_ff_dict, believed_ff_dict, new_num_trials, ff_dataframe_anim


def rotate_anim_monkey_info(anim_monkey_info, R):
    """
    Rotate the animation information of the monkey/agent

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
    anim_mx, anim_my = np.array(anim_monkey_info['anim_mx']), np.array(anim_monkey_info['anim_my'])
    anim_mx, anim_my = R @ np.array([anim_mx, anim_my])
    x0, y0 = anim_mx[0], anim_my[0]
    anim_monkey_info['x0'], anim_monkey_info['y0'] = x0, y0
    anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'] = anim_mx-x0, anim_my-y0

    anim_monkey_info['xmin'], anim_monkey_info['xmax'] = np.min(anim_mx)-x0, np.max(anim_mx)-x0
    anim_monkey_info['ymin'], anim_monkey_info['ymax'] = np.min(anim_my)-y0, np.max(anim_my)-y0
    
    if 'gaze_world_x' in anim_monkey_info.keys():
        anim_monkey_info['gaze_world_x'], anim_monkey_info['gaze_world_y'] = R @ np.array([anim_monkey_info['gaze_world_x'], anim_monkey_info['gaze_world_y']]) - np.array([x0, y0]).reshape(2,1)

    return anim_monkey_info, x0, y0





def find_xy_min_max_for_animation(anim_monkey_info, ff_dataframe_anim):
    mx_min, mx_max = anim_monkey_info['xmin'], anim_monkey_info['xmax']
    my_min, my_max = anim_monkey_info['ymin'], anim_monkey_info['ymax']
    visible_ffs = ff_dataframe_anim[ff_dataframe_anim['visible'] == 1]
    if len(visible_ffs) > 0:
        mx_min, mx_max = min(mx_min, min(visible_ffs.ff_x)), max(mx_max, max(visible_ffs.ff_x))
        my_min, my_max = min(my_min, min(visible_ffs.ff_y)), max(my_max, max(visible_ffs.ff_y))
    return mx_min, mx_max, my_min, my_max







def animate(frame, ax, anim_monkey_info, margin, ff_dataframe_anim, ff_real_position_sorted, flash_on_ff_dict, alive_ff_dict, 
            believed_ff_dict, plot_flash_on_ff=False, plot_eye_position=False, set_xy_limits=True, rotate=True): 
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
    rotate: bool
        whether to rotate the animation information of the monkey/agent

    Returns
    -------
    ax: obj
        a matplotlib axes object 

    """ 

    index = anim_monkey_info['anim_indices'][frame]
    alive_ff = alive_ff_dict[index]
    relevant_ff = ff_dataframe_anim[ff_dataframe_anim['point_index'] == index].copy()
    


    # if rotate:
    #     R, theta = plot_behaviors.find_rotation_matrix(anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'], also_return_angle=True)
    #     anim_monkey_info, x0, y0 = rotate_anim_monkey_info(anim_monkey_info, R)
    #     anim_monkey_info['anim_angle'] = anim_monkey_info['anim_angle'] + theta
    #     alive_ff = (R @ alive_ff.T).T - np.array([x0, y0])
    #     relevant_ff[['ff_x', 'ff_y']] = (R @ relevant_ff[['ff_x', 'ff_y']].values.T).T - np.array([x0, y0])
    #     if "ff_x_noisy" in relevant_ff.columns:
    #         relevant_ff[['ff_x_noisy', 'ff_y_noisy']] = (R @ relevant_ff[['ff_x_noisy', 'ff_y_noisy']].values.T).T - np.array([x0, y0])
    #     for key, item in believed_ff_dict.items():
    #         if len(item) > 0:
    #             believed_ff_dict[key] = (R @ np.array(item).T).T - np.array([x0, y0])
    # else:
    #     x0, y0 = 0, 0




    visible_ffs = relevant_ff[relevant_ff['visible'] == 1]

    ax.cla()
    ax.axis('off')
    if set_xy_limits:
        mx_min, mx_max, my_min, my_max = find_xy_min_max_for_animation(anim_monkey_info, ff_dataframe_anim)
        ax = plot_behaviors.set_xy_limits_for_axes(ax, mx_min, mx_max, my_min, my_max, margin)
        # ax.set_xlim((anim_monkey_info['xmin']-margin, anim_monkey_info['xmax']+margin))
        # ax.set_ylim((anim_monkey_info['ymin']-margin, anim_monkey_info['ymax']+margin))
    ax.set_aspect('equal')

    
    # Plot the arena
    circle_theta = np.arange(0, 2*pi, 0.01)
    ax.plot(np.cos(circle_theta)*1000-anim_monkey_info['x0'], np.sin(circle_theta)*1000-anim_monkey_info['y0'])

    # Plot fireflies
    ax.scatter(alive_ff[:, 0], alive_ff[:, 1], alpha=0.7, c="gray", s=20)

    if plot_flash_on_ff:
        flashing_on_ff = flash_on_ff_dict[index]
        ax.scatter(flashing_on_ff[:, 0], flashing_on_ff[:, 1], alpha=1, c="red", s=30)
    
    # Plot the monkey's or agent's trajectory
    ax.scatter(anim_monkey_info['anim_mx'][:frame+1], anim_monkey_info['anim_my'][:frame+1], s=15, c='royalblue')
    
    if plot_eye_position:
        ax.scatter(anim_monkey_info['gaze_world_x'][frame], anim_monkey_info['gaze_world_y'][frame], s=30, c='darkgreen')

    # #Plot in-memory ff
    # in_memory_ffs = relevant_ff[relevant_ff['visible']==0]
    # ax.scatter(in_memory_ffs.ff_x , in_memory_ffs.ff_y , alpha=1, c="green", s=30)


    # #Plot target
    # time = anim_tframe
    # trial_num = np.where(ff_caught_T_sorted > time)[0][0]
    # ax.scatter(ff_real_position_sorted[trial_num][0], ff_real_position_sorted[trial_num][1], marker='*', c='blue', s = 130, alpha = 0.5)



    # Plot the reward boundaries of visible fireflies
    if "ff_x_noisy" in relevant_ff.columns: 
        # plot both the real positions and the noisy positions
        for k in range(len(visible_ffs)):
          circle = plt.Circle((visible_ffs.ff_x.iloc[k], visible_ffs.ff_y.iloc[k]), 25, facecolor='yellow', edgecolor='gray', alpha=0.5, zorder=1)
          ax.add_patch(circle)
          circle = plt.Circle((visible_ffs.ff_x_noisy.iloc[k], visible_ffs.ff_y_noisy.iloc[k]), 25, facecolor='gray', edgecolor='gray', alpha=0.5, zorder=1)
          ax.add_patch(circle)
    else:
        # plot the real positions only
        for k in range(len(visible_ffs)):
          circle = plt.Circle((visible_ffs.ff_x.iloc[k], visible_ffs.ff_y.iloc[k]), 25, facecolor='yellow', edgecolor='gray', alpha=0.5, zorder=1)
          ax.add_patch(circle)


    ## Plot reward boundaries of in-memory fireflies
    # for j in range(len(in_memory_ffs)):
    #   circle = plt.Circle((in_memory_ffs.ff_x.iloc[j], in_memory_ffs.ff_y.iloc[j]), 25, facecolor='grey', edgecolor='orange', alpha=0.3, zorder=1)
    #   ax.add_patch(circle)

    
    # Plot the believed positions of caught fireflies  
    if len(believed_ff_dict[index]) > 0:
        ax.scatter(believed_ff_dict[index][:, 0], believed_ff_dict[index][:, 1], alpha=1, c="purple", s=30)

    # Plot a triangular shape to indicate the direction of the agent
    left_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] + 2*pi/9) 
    left_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] + 2*pi/9)
    right_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] - 2*pi/9) 
    right_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] - 2*pi/9)
    
    ax.plot(np.array([anim_monkey_info['anim_mx'][frame], left_end_x]), np.array([anim_monkey_info['anim_my'][frame] , left_end_y]), linewidth = 2)
    ax.plot(np.array([anim_monkey_info['anim_mx'][frame], right_end_x]), np.array([anim_monkey_info['anim_my'][frame] , right_end_y]), linewidth = 2)
    return ax





def animate_annotated(frame, ax, anim_monkey_info, margin, ff_dataframe, ff_real_position_sorted, ff_position_during_this_trial, \
                      flash_on_ff_dict, alive_ff_dict, believed_ff_dict, ff_caught_T_sorted, annotation_info):
    
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
    ff_position_during_this_trial: np.array
        containing the locations of all alive fireflies in a given duration
    flash_on_ff_dict: dict
        contains the indices of the fireflies that have been flashing at each time point
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative
    ff_caught_T_sorted: np.array
        containing the time when each captured firefly gets captured
    annotation_info: dictionary
        containing the information needed for the annotation of animation 

    Returns
    -------
    ax: obj
        a matplotlib axes object 

    """ 

    animate(frame, ax, anim_monkey_info, margin, ff_dataframe, ff_real_position_sorted, flash_on_ff_dict, alive_ff_dict, believed_ff_dict)
    index = anim_monkey_info['anim_indices'][frame]
    time = anim_monkey_info['anim_t'][frame]
    trial_num = np.where(ff_caught_T_sorted > time)[0][0]
    annotation = ""
    # If the monkey has captured more than one 1 ff in a cluster
    if annotation_info['n_ff_in_a_row'][trial_num] > 1:
      annotation = annotation + f"Captured {annotation_info['n_ff_in_a_row'][trial_num]} ffs in a cluster\n"
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

    ff_in_time_frame = ff_dataframe[(ff_dataframe["target_index"] > (currentTrial-num_trials)) &  \
                                    (ff_dataframe["target_index"] <= currentTrial)]
    ff_in_time_frame = ff_in_time_frame[['point_index', 'target_index', 'visible', 'ff_distance', 'ff_angle']]
    ff_visible = ff_in_time_frame[ff_in_time_frame['visible'] == 1][['point_index', 'ff_distance', 'ff_angle']]
    ff_in_memory = ff_in_time_frame[ff_in_time_frame['visible'] == 0][['point_index', 'ff_distance', 'ff_angle']]
    return ff_in_time_frame, ff_visible, ff_in_memory





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
    ax = plot_behaviors.set_polar_background_for_animation(ax, rmax)
    index = anim_indices[frame]
    all_ff_info_now = ff_in_time_frame.loc[ff_in_time_frame['point_index'] == index]
    if all_ff_info_now.shape[0] > 0:
      # Only if all_ff_info_now is not empty, can ff_visible and ff_in_memory be possibly not empty
      ff_visible_now = ff_visible.loc[ff_visible['point_index']==index]
      if ff_visible_now.shape[0] > 0:
        ax.scatter(ff_visible_now['ff_angle'], ff_visible_now['ff_distance'], marker = 'o', alpha=0.8, c_var = None, s=30)
      
      ff_in_memory_now = ff_in_memory.loc[ff_in_memory['point_index']==index]
      if ff_in_memory_now.shape[0] > 0:
        ax.scatter(ff_in_memory_now['ff_angle'], ff_in_memory_now['ff_distance'], marker = 'o', alpha=0.8, c="green", s=30)
    return ax
