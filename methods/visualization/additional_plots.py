import sys
from visualization import plot_trials, plot_behaviors_utils

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK']='True'







def PlotPoints(point, 
               duration_of_trajectory, 
               monkey_information, 
               ff_dataframe, 
               ff_caught_T_new,
               ff_life_sorted, 
               ff_real_position_sorted, 
               ff_believed_position_sorted, 
               ff_flash_sorted, 
               fig = None, 
               axes = None,
               visible_distance = 500,
               show_all_ff = True,
               show_flash_on_ff = False,
               show_visible_ff = True,
               show_in_memory_ff = True, 
               show_target = False,
               show_reward_boundary = True,
               show_legend = True,
               show_scale_bar = True,
               show_colorbar = True, 
               hitting_arena_edge_ok = False,
               trial_too_short_ok = False, 
               images_dir = None):

    """
    Visualize a time point in the game
    Note: As of now, this function is only used for monkey. This function also does not utilize rotation.


    Parameters
    ----------
    point: num
        the index of the point to visualize
    duration_of_trajectory: list
        the duration of the trajectory to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    fig: object
        the canvas of the plot
    axes: object
        The axes of the plot
    visible_distance: num
        the distance beyond which a firefly will not be considered visible; default is 250
    show_all_ff: bool
        whether to show all the fireflies that are alive at that point as grey
    show_flash_on_ff: bool
        whether to show all the fireflies that are flashing on at that point as red
    show_visible_ff: bool
        whether to show all the fireflies visible at that point as orange
    show_in_memory_ff: bool
        whether to show all the fireflies in memory at that point as orange
    show_target: bool
        whether to show the target using star shape
    show_reward_boundary: bool
        whether to show the reward boundaries of fireflies
    show_legend: bool
        whether to show a legend
    show_scale_bar: bool
        whether to show the scale bar
    show_colorbar: bool
        whether to show the color bar
    hitting_arena_edge_ok: bool
        whether to continue to plot the trial if the boundary is hit at any point
    trial_too_short_ok: bool
        whether to continue to plot the trial if the trial is very short (fewer than 5 time points)
    images_dir: str or None
        directory of the file to store the images


    """
    sns.set_style(style="white")

    time = np.array(monkey_information['monkey_t'])[point]
    duration = [time - duration_of_trajectory, time]
    cum_iloc_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))
    cum_t, cum_mx, cum_my = monkey_information['monkey_t'].iloc[cum_iloc_indices].values, monkey_information['monkey_x'].iloc[cum_iloc_indices].values, monkey_information['monkey_y'].iloc[cum_iloc_indices].values
    
    if not hitting_arena_edge_ok:
        # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_r = LA.norm(np.stack((cum_mx, cum_my)), axis = 0)
        if (np.any(cum_r > 949)):
            return
    if not trial_too_short_ok:
        # Stop plotting for the trial if the trial is too short
        if (len(cum_t) < 5):
            return

    if fig is None:
        fig, axes = plt.subplots()

    alive_ff_indices = np.array([ff_index for ff_index, life_duration in enumerate(ff_life_sorted) 
                                if (life_duration[-1] >= time) and (life_duration[0] < time)])
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
    
    if show_all_ff:
        axes.scatter(alive_ff_positions.T[0], alive_ff_positions.T[1], color="grey", s=30)

    if show_flash_on_ff:
        # Initialize a list to store the indices of the ffs that are flashing-on at this point
        flashing_ff_indices = []  
        # For each firefly in ff_flash_sorted:
        for ff_index, ff_flash_intervals in enumerate(ff_flash_sorted):
            # If the firefly has flashed during that trial:
            if ff_index in alive_ff_indices:
                # Let's see if the firefly has flashed at that exact moment
                for interval in ff_flash_intervals:
                    if interval[0] <= time <= interval[1]:
                        flashing_ff_indices.append(ff_index)
                        break
        flashing_ff_indices = np.array(flashing_ff_indices)
        flashing_ff_positions = ff_real_position_sorted[flashing_ff_indices]
        axes.scatter(flashing_ff_positions.T[0], flashing_ff_positions.T[1], color="red", s=120, marker='*', alpha=0.7)

    if show_visible_ff:
        visible_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 1) &
                                   (ff_dataframe['ff_distance'] <= visible_distance)]
        axes.scatter(visible_ffs['ff_x'], visible_ffs['ff_y'], color_var = None, s=40)

    if show_in_memory_ff:
        in_memory_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 0)]
        axes.scatter(in_memory_ffs['ff_x'], in_memory_ffs['ff_y'], color="green", s=40)

    if show_target:
        trial_num = np.digitize(time, ff_caught_T_new)
        if trial_num is None:
            raise ValueError("If show_target, then trial_num cannot be None")
        target_position = ff_real_position_sorted[trial_num]
        axes.scatter(target_position[0], target_position[1], marker='*', s=200, color="grey", alpha=0.35)

    if show_legend is True:
        # Need to consider what elements are used in the plot
        legend_names = []
        if show_all_ff:
            legend_names.append("Invisible")
        if show_flash_on_ff:
            legend_names.append("Flash On")
        if show_visible_ff:
            legend_names.append("Visible")
        if show_in_memory_ff:
            legend_names.append("In memory")
        if show_target:
            legend_names.append("Target")
        axes.legend(legend_names, loc='upper right')


    if show_reward_boundary:
        if show_all_ff: 
            for position in alive_ff_positions:
                circle = plt.Circle((position[0], position[1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
        if show_visible_ff:
            for index, row in visible_ffs.iterrows():
                circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='yellow', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            if show_in_memory_ff:
                for index, row in in_memory_ffs.iterrows():
                    circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle)
        elif show_flash_on_ff:
            for index, row in flashing_ff_positions.iterrows():
                circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='red', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            for ff in flashing_ff_positions:
                circle = plt.Circle((ff[0], ff[1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            if show_in_memory_ff:
                for index, row in in_memory_ffs.iterrows():
                    circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle)

    # Also plot the trajectory of the monkey/agent
    axes.scatter(cum_mx, cum_my, s=15, c=cum_iloc_indices, cmap="Blues")

    # Set the limits of the x-axis and y-axis
    mx_min, mx_max, my_min, my_max = plot_behaviors_utils.find_xy_min_max_for_plots(np.stack((cum_mx, cum_my)), x0=0, y0=0, temp_ff_positions=None)

    axes = plot_behaviors_utils.set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, 250, zoom_in=False)

    if show_scale_bar == True:
        scale1 = ScaleBar(dx=1, units='cm', length_fraction=0.2, fixed_value=100, location='upper left', label_loc='left', scale_loc='bottom')
        axes.add_artist(scale1)

    if show_colorbar == True:
        cmap = cm.Blues
        cax = fig.add_axes([0.95, 0.25, 0.05, 0.52])  # [left, bottom, width, height]
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 1], cax=cax, orientation='vertical')
        cbar.ax.set_title('Trajectory', ha='left', y=1.07)
        cbar.ax.tick_params(size=0)
        cbar.outline.remove()
        cbar.ax.set_yticklabels(['Least recent', 'Most recent'])

    axes.xaxis.set_major_locator(mtick.NullLocator())
    axes.yaxis.set_major_locator(mtick.NullLocator())


    if images_dir is not None:
      filename = "time_point_" + str(point)
      CHECK_FOLDER = os.path.isdir(images_dir)
      if not CHECK_FOLDER:
          os.makedirs(images_dir)
      plt.savefig(f"{images_dir}/{filename}.png")









def PlotSidebySide(plot_whole_duration,                   
                  info_of_monkey,
                  info_of_agent,
                  num_imitation_steps_monkey,
                  num_imitation_steps_agent,
                  currentTrial,
                  num_trials, 
                  rotation_matrix,
                  plotting_params = None,
                  data_folder_name = None,
                  ):
    """
    Plot the monkey's plot and the agent's plot side by side


    Parameters
    ----------
    plot_whole_duration: list of 2 elements
        containing the start time and the end time in respect to the monkey data
    info_of_monkey: dict
        contains various important arrays, dataframes, or lists derived from the real monkey data
    info_of_agent: dict
        contains various important arrays, dataframes, or lists derived from the RL environmentthe and the agent's behaviours
    num_imitation_steps_monkey: num
        the number of steps used by the monkey for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)
    num_imitation_steps_agent: num
        the number of steps used by the agent for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)
    currentTrial: num
        the current trial to be plotted
    num_trials: num
        the number of trials (counting from the currentTrial into the past) to be plotted
    rotation_matrix: np.array
        to be used to rotated the plot when plotting
    plotting_params: dict, optional
        keyword arguments to be passed into the plot_trials.PlotTrials function
    data_folder_name: str
        name or path of the folder to store the graph
    """  


    #===================================== Monkey =====================================
    sns.set_style(style="white")
    fig = plt.figure(figsize=(17, 12), dpi=125)
    axes = fig.add_subplot(121)

    plot_trials.PlotTrials(plot_whole_duration, 
            info_of_monkey['monkey_information'],
            info_of_monkey['ff_dataframe'], 
            info_of_monkey['ff_life_sorted'], 
            info_of_monkey['ff_real_position_sorted'], 
            info_of_monkey['ff_believed_position_sorted'], 
            info_of_monkey['cluster_around_target_indices'], 
            info_of_monkey['ff_caught_T_new'], 
            currentTrial = currentTrial,
            num_trials = num_trials,
            fig = fig, 
            axes = axes, 
            rotation_matrix = rotation_matrix,
            player = "monkey",
            steps_to_be_marked = num_imitation_steps_monkey,
            **plotting_params
            )
    axes.set_title(f"Monkey: Trial {currentTrial}", fontsize = 22)
    
    #===================================== Agent =====================================

    axes2 = fig.add_subplot(122)
    # Agent duration needs to start from 0, unlike the duration for the monkey, because the 
    # RL environment starts from 0 
    agent_duration = [0, plot_whole_duration[1]-plot_whole_duration[0]]      


    plot_trials.PlotTrials(agent_duration, 
              info_of_agent['monkey_information'],
              info_of_agent['ff_dataframe'], 
              info_of_agent['ff_life_sorted'], 
              info_of_agent['ff_real_position_sorted'], 
              info_of_agent['ff_believed_position_sorted'], 
              info_of_agent['cluster_around_target_indices'], 
              info_of_agent['ff_caught_T_new'], 
              currentTrial = None,
              num_trials = None,
              fig = fig, 
              axes = axes2, 
              rotation_matrix = rotation_matrix,
              player = "agent",
              steps_to_be_marked = num_imitation_steps_agent,
              **plotting_params
              )
    axes2.set_title(f"Agent: Trial {currentTrial}", fontsize = 22)

    overall_xmin, overall_xmax, overall_ymin, overall_ymax = plot_behaviors_utils.get_overall_lim(axes, axes2)
    plt.setp([axes, axes2], xlim=[overall_xmin, overall_xmax], ylim=[overall_ymin, overall_ymax])

    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        figure_name = os.path.join(data_folder_name, f"side_by_side_{currentTrial}.png")
        plt.savefig(figure_name)
    plt.tight_layout()
    plt.show()
    




