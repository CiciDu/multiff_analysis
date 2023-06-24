from multiff_analysis.functions.data_wrangling import basic_func, make_ff_dataframe
from multiff_analysis.functions.data_visualization import eye_positions
import os
import math
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'



              

def PlotTrials(duration, 
               monkey_information,
               ff_dataframe, 
               ff_life_sorted, 
               ff_real_position_sorted, 
               ff_believed_position_sorted, 
               cluster_around_target_indices, 
               ff_caught_T_sorted,
               currentTrial = None, # Can be None; then it means all trials in the duration will be plotted
               num_trials = None,
               fig = None, 
               axes = None, 
               rotation_matrix = None,
               x0 = None,
               y0 = None,
               player="monkey",
               trail_color_var = None, # None or 'speed' or 'abs_ddw' or 'target_visibility'; if not None, then the color of the path will vary by this variable
               visible_distance = 400,
               minimal_margin = 100,
               show_start=True,
               show_stops=False,
               show_trajectory = True,
               show_monkey_angles = False,
               show_alive_fireflies = True,
               show_ff_indices = False,
               show_believed_target_positions=False,
               show_reward_boundary=False,
               show_path_when_target_visible=False,  
               show_path_when_prev_target_visible=False,
               show_connect_path_ff=False, 
               show_connect_path_ff_except_targets=False,
               connect_path_ff_color = "#a940f5", # a kind of purple
               vary_color_for_connecting_path_ff = False, 
               show_connect_path_ff_memory=False,
               connect_path_ff_max_distance = None,
               show_path_when_cluster_visible=False,
               show_eye_positions = False,
               show_eye_positions_on_the_right = False, 
               show_connect_path_eye_positions = False,
               show_null_agent_trajectory = False,
               show_null_agent_trajectory_2nd_time = False, 
               null_agent_starting_time = None,
               assumed_memory_duration_of_agent = 2,
               show_scale_bar=False,
               show_colorbar=False,
               show_title = True,
               show_legend = False,
               trial_to_show_cluster=None,  # None, "current", or "previous"
               cluster_dataframe_point=None, 
               trial_to_show_cluster_around_target=None, # None, "current", or "previous"
               indices_of_ff_to_mark = None, # None or a list
               steps_to_be_marked=None,
               adjust_xy_limits = True,
               zoom_in = False, # can only be effective if adjust_xy_limits is True
               images_dir = None,
               hitting_arena_edge_ok = False,
               trial_too_short_ok = False, 
               subplots = False,
               combined_plot = False,
               as_part_of_animation = False,
               ):

    """
    Visualize a trial or a few consecutive trials


    Parameters
    ----------
    duration: list
        the duration to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    cluster_around_target_indices: list
        for each trial, it contains the indices of fireflies around the target; 
        it contains an empty array when there is no firefly around the target
    fig: object
        the canvas of the plot
    axes: object
        The axes of the plot
    rotation_matrix: array
        The matrix by which the plot will be rotated
    currentTrial: numeric
        the number of current trial
    num_trials: numeric
        the number of trials to be plotted
    player: str
        "monkey" or "agent"
    trail_color_var: str or None
        the variable that determines the color of the trajectory of the monkey/agent; can be None or 'speed' or 'ads_ddw'
    visible_distance: num
        the distance beyond which a firefly will not be considered visible; default is 250
    minimal_margin: num
        the minimal margin of the plot (e.g., to the left of xmin, to the right of xmax, to the top of ymax, to the bottom of ymin)
    show_start: bool
        whether to show the starting point of the monkey/agent
    show_trajectory: bool
        whether to show the trajectory of the monkey/agent
    show_monkey_angles: bool
        whether to show the angles of the monkey on the trajectory 
    show_stop: bool
        whether to show the stopping point of the monkey/agent
    show_alive_fireflies: bool
        whether to show all the fireflies that are alive
    show_ff_indices: bool
        whether to annotate the ff_index for each ff
    show_believed_target_positions: bool
        whether to show the believed positions of the targets
    show_reward_boundary: bool
        whether to show the reward boundaries of fireflies
    show_path_when_target_visible: bool
        whether to mark the part of the trajectory where the target is visible  
    show_path_when_prev_target_visible: bool
        whether to mark the part of the trajectory where the previous target is visible  
    show_connect_path_ff: bool
        whether to draw lines between the trajectory and fireflies to indicate the part of the trajectory where a firefly is visible
    show_connect_path_ff_except_targets: bool
        same function as show_connect_path_ff, except the targets during the trials are excluded
    connect_path_ff_max_distance: bool
        the distance beyond which a firefly will not be considered visible when drawing lines between the path and the firefly
    show_path_when_cluster_visible: bool
        whether to mark the part of the trajectory where any firefly in the cluster centered around the target is visible  
    show_scale_bar: bool
        whether to show the scale bar
    show_colorbar: bool
        whether to show the color bars
    show_title: bool
        whether to show title (which indicates the current trial number)
    trial_to_show_cluster: can be None, "current", or "previous"
        the trial for which to show clusters of fireflies 
    cluster_dataframe_point: dataframe
        information of the clusters for each time point that has at least one cluster; must not be None if trial_to_show_cluster is not None
    trial_to_show_cluster_around_target: can be None, "current", or "previous"
        the trial for which to show the cluster of fireflies centered around the target
    indices_of_ff_to_mark: None or a list
        a list of indices of fireflies that will be marked (can be used to show the ignored fireflies in "ignore sudden flash" trials) 
    steps_to_be_marked: None or a list
        indices of the points on the trajectory to be marked by a different color from the path color
    adjust_xy_limits: bool
        whether to adjust xmin, xmax, ymin, ymax
    zoom_in: bool
        whether to zoom in on the plot
    images_dir: str or None
        directory of the file to store the images
    hitting_arena_edge_ok: bool
        whether to continue to plot the trial if the boundary is hit at any point
    trial_too_short_ok: bool
        whether to continue to plot the trial if the trial is very short (fewer than 5 time points)
    subplots: bool
        whether subplots are used
    combined_plot: bool
        whether multiple trajectories are combined into one plot; if yes, then all trajectories 
        will be centered so that they start at the same point
        if True, then whether the plot is successfully made will be returned; the plot might fail to be made because the
        action sequence is too short (if trial_too_short_ok is False) or the monkey has hit the boundary at any point (if 
        hitting_arena_edge_ok is false)


    """
    sns.set(style="white")
    legend_markers = []
    legend_names = []

    if combined_plot is True:
        player = "combined"

    # If currentTrial is not given, then it will be calculated based on the duration
    currentTrial, num_trials, duration = basic_func.find_currentTrial_or_num_trials_or_duration(ff_caught_T_sorted, currentTrial, num_trials, duration)


    target_indices = np.arange(currentTrial-num_trials+1, currentTrial+1)
    cum_indices, cum_t, cum_angles, cum_mx, cum_my, cum_speed, cum_speeddummy = find_monkey_information_in_the_duration(duration, monkey_information)
    ff_dataframe_in_duration = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]

    if not hitting_arena_edge_ok:
        # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_r = LA.norm(np.stack((cum_mx, cum_my)), axis = 0)
        if (np.any(cum_r > 949)):
            return False, None, None, None # the three outputs are whether_plotted, axes, R, cum_mxy_rotate, shown_ff_indices

    if not trial_too_short_ok:
        # Stop plotting for the trial if the trial is too short
        if (len(cum_t) < 5):
            return False, None, None, None # the three outputs are whether_plotted, axes, R, cum_mxy_rotate, shown_ff_indices

    if fig is None:
        if show_eye_positions_on_the_right:
            fig = plt.figure(figsize=(16, 7))
            axes = fig.add_subplot(1, 2, 1)
        else:
            fig, axes = plt.subplots()
    elif axes is None:
        if show_eye_positions_on_the_right:
            axes = fig.add_subplot(121)
        elif as_part_of_animation:
            axes = fig.add_subplot(121)
        else:
            axes = fig.add_subplot(111)

    if rotation_matrix is None:
        R = find_rotation_matrix(cum_mx, cum_my)
        rotation_matrix = R
    else:
        R = rotation_matrix


    # Find the trajectory of the monkey
    cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
    
    # Determine whether translation is needed for the trajectory
    if (x0 is None) or (y0 is None):
        if combined_plot or show_eye_positions: 
            x0, y0 = cum_mxy_rotate[0][0], cum_mxy_rotate[1][0] 
        else:
            x0, y0 = 0, 0

           

    if show_monkey_angles:
        left_end_xy_rotate, right_end_xy_rotate = find_triangles_to_show_monkey_angles(cum_mx, cum_my, cum_angles, rotation_matrix=R)
        axes = visualize_monkey_angles_using_triangles(axes, cum_mxy_rotate, left_end_xy_rotate, right_end_xy_rotate, linewidth=0.5)

    if show_start:
        # Plot the start
        start_size = {"agent": 220, "monkey": 150}
        marker = axes.scatter(cum_mxy_rotate[0, 0]-x0, cum_mxy_rotate[1, 0]-y0, marker='s', s=start_size[player], color="green", zorder=3, alpha=0.7)
        legend_markers.append(marker)
        legend_names.append('Start new section or begin using null agent')
        

    if show_stops:
        stop_size = {"agent": 160, "monkey": 150, "combined": 40}
        zerospeed_rotate = find_stops_for_plotting(cum_mx, cum_my, cum_speeddummy, rotation_matrix=R)
        marker = axes.scatter(zerospeed_rotate[0]-x0, zerospeed_rotate[1]-y0, marker='*', s=stop_size[player], alpha=0.7, color="black", zorder=2)
        legend_markers.append(marker)
        legend_names.append('Low speed/stopping points')

    alive_ff_indices, alive_ff_position_rotate = find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted, rotation_matrix=R)
    if show_alive_fireflies:
        axes.scatter(alive_ff_position_rotate[0]-x0, alive_ff_position_rotate[1]-y0, marker='o', s=10, color="magenta", zorder=2)
 
    if show_ff_indices:
        for i, position in enumerate(alive_ff_position_rotate.T):
            ff_index = alive_ff_indices[i]
            axes.annotate(str(ff_index), (position[0], position[1]))

    shown_ff_indices = [] # a list of indices of fireflies that will be shown in the plot except alive_ff
    if show_believed_target_positions:
        target_size = {"agent": 185, "monkey": 120, "combined": 30}
        marker = {"agent": "*", "monkey": "*", "combined": "o"}
        shown_ff_indices.extend(range(currentTrial - num_trials + 1, currentTrial + 1))
        believed_target_positions_rotate = find_believed_target_positions(ff_believed_position_sorted, currentTrial, num_trials, rotation_matrix=R)
        marker = axes.scatter(believed_target_positions_rotate[0]-x0, believed_target_positions_rotate[1]-y0, marker=marker[player], s=target_size[player], color="red", alpha=0.75, zorder=2)
        legend_markers.append(marker)
        legend_names.append('Catching-firefly positions')

    if indices_of_ff_to_mark is not None:
        shown_ff_indices.extend(indices_of_ff_to_mark)
        for ff in indices_of_ff_to_mark:
            ff_position = ff_real_position_sorted[ff]
            ff_position_rotate = np.matmul(R, np.stack((ff_position[0], ff_position[1])))
            axes.scatter(ff_position_rotate[0]-x0, ff_position_rotate[1]-y0, marker='*', s=target_size[player], color="green", alpha=0.75, zorder=2)


    if steps_to_be_marked is not None:
        start_size = {"agent": 220, "monkey": 100}
        axes.scatter(cum_mxy_rotate[0, steps_to_be_marked]-x0, cum_mxy_rotate[1, steps_to_be_marked]-y0, marker='s', s=start_size[player], color="gold", zorder=3, alpha=0.7)

    if show_path_when_target_visible:
        path_size = {"agent": 50, "monkey": 30, "combined": 2}
        ff_visible_path_rotate = find_path_when_target_visible(currentTrial, ff_dataframe_in_duration, cum_indices, visible_distance, rotation_matrix=R)
        marker = axes.scatter(ff_visible_path_rotate[0]-x0, ff_visible_path_rotate[1]-y0, s=path_size[player], c="green", alpha=0.6, zorder=5)
        legend_markers.append(marker)
        legend_names.append('Path when target is visible')

    if show_path_when_prev_target_visible: # for previous target
        path_size = {"agent": 65, "monkey": 40, "combined": 2}
        ff_visible_path_rotate = find_path_when_target_visible(currentTrial-1, ff_dataframe_in_duration, cum_indices, visible_distance, rotation_matrix=R)
        marker = axes.scatter(ff_visible_path_rotate[0]-x0, ff_visible_path_rotate[1]-y0, s=path_size[player], c="aqua", alpha=0.8, zorder=3)
        legend_markers.append(marker)
        legend_names.append('Path when previous target is visible')

    if connect_path_ff_max_distance is None:
        connect_path_ff_max_distance = visible_distance


    temp_ff_positions = None
    connection_linewidth = {"agent": 0.15, "monkey": 0.2, "combined": 0.1}
    connection_alpha = {"agent": 0.6, "monkey": 0.7, "combined": 0.8}

    
    if show_connect_path_ff_memory:
        ff_dataframe_in_duration_in_memory = ff_dataframe_in_duration.loc[(ff_dataframe_in_duration['visible']==0) & (ff_dataframe_in_duration['ff_distance'] <= visible_distance)]
        shown_ff_indices.extend(ff_dataframe_in_duration_in_memory.ff_index.unique())
        temp_ff_positions, temp_monkey_positions = find_lines_to_connect_path_ff(ff_dataframe_in_duration_in_memory, target_indices, rotation_matrix=R)
        axes = connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color = "orange", alpha = 0.3, 
                                linewidth=connection_linewidth[player], show_dots=True, dot_color="brown")


    if show_connect_path_ff or show_connect_path_ff_except_targets:
        ff_dataframe_in_duration_visible = ff_dataframe_in_duration.loc[(ff_dataframe_in_duration['visible']==1) & (ff_dataframe_in_duration['ff_distance'] <= connect_path_ff_max_distance)]
        shown_ff_indices.extend(ff_dataframe_in_duration_visible.ff_index.unique())
        if vary_color_for_connecting_path_ff:
            unique_ff_index = ff_dataframe_in_duration_visible.ff_index.unique()
            varing_colors = sns.color_palette("tab10", 10)
            # varing_colors = np.delete(varing_colors, 2, 0)   # take out the 3rd color (green from varying_colors)
            for i in range(len(unique_ff_index)):
                ff_index = unique_ff_index[i]
                color = np.append(varing_colors[i % 10], 0.5)  # take out a color from Set2
                temp_df = ff_dataframe_in_duration_visible[ff_dataframe_in_duration_visible['ff_index'] == ff_index]
                temp_ff_positions, temp_monkey_positions = find_lines_to_connect_path_ff(temp_df, target_indices, rotation_matrix=R, target_excluded = show_connect_path_ff_except_targets)
                axes = connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color = color, alpha = connection_alpha[player], 
                                                linewidth=connection_linewidth[player], show_dots=True, dot_color="brown")
                marker = axes.scatter(temp_monkey_positions[0, -1]-x0, temp_monkey_positions[1, -1]-y0, alpha=0.7, marker="X", s=80, color=color, zorder=4)
                if i==0:
                    legend_markers.append(marker)
                    legend_names.append('Points when fireflies stop being visible')

        else:
            temp_ff_positions, temp_monkey_positions = find_lines_to_connect_path_ff(ff_dataframe_in_duration_visible, target_indices, rotation_matrix=R, target_excluded = show_connect_path_ff_except_targets)
            axes = connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color = connect_path_ff_color, alpha = connection_alpha[player], 
                                    linewidth=connection_linewidth[player], show_dots=True, dot_color="brown")




    if trial_to_show_cluster is not None:
        trial_conversion = {"current": 0, "previous": -1}
        cluster_ff_rotate = find_ff_in_cluster(cluster_dataframe_point, ff_real_position_sorted, currentTrial=currentTrial+trial_conversion[trial_to_show_cluster], rotation_matrix=R)
        axes.scatter(cluster_ff_rotate[0]-x0, cluster_ff_rotate[1]-y0, marker='o', c="blue", s=25, zorder=4)




    if trial_to_show_cluster_around_target is not None:
        trial_conversion = {"current": 0, "previous": -1}
        cluster_ff_indices, cluster_around_target_rotate = find_ff_in_cluster_around_target(cluster_around_target_indices, ff_real_position_sorted, rotation_matrix=R,\
            currentTrial=currentTrial+trial_conversion[trial_to_show_cluster_around_target])
        shown_ff_indices.extend(cluster_ff_indices)
        axes.scatter(cluster_around_target_rotate[0]-x0, cluster_around_target_rotate[1]-y0, marker='o', s=30, color="blue", zorder=4)
        if show_path_when_cluster_visible: # Find where on the path the monkey/agent can see any member of the cluster around the target
            list_of_colors = ["navy", "magenta", "white", "gray", "brown", "black"]
            path_size, path_alpha = {"agent": [80, 10], "monkey": [15, 3]}, {"agent": 0.8, "monkey": 0.4}
            ff_size, ff_alpha = {"agent": 140, "monkey": 100}, {"agent": 0.8, "monkey": 0.5}
            for index in cluster_ff_indices:
                monkey_xy_rotate, ff_position_rotate = find_path_when_ff_in_cluster_visible(ff_dataframe_in_duration, index, rotation_matrix=R)
                axes.scatter(monkey_xy_rotate[0]-x0, monkey_xy_rotate[1]-y0, s=path_size[player][0] - path_size[player][1] * i, color=list_of_colors[i], alpha=path_alpha[player], zorder=3+i)
                # Use a circle with the corresponding color to show that ff
                axes.scatter(ff_position_rotate[0]-x0, ff_position_rotate[1]-y0, marker='o', s=ff_size[player], alpha=ff_alpha[player], color=list_of_colors[i], zorder=3)


    shown_ff_indices = np.unique(np.array(shown_ff_indices)).astype(int)
    shown_ff_positions_rotate = ff_real_position_sorted[shown_ff_indices].T
    if (rotation_matrix is not None) & (shown_ff_positions_rotate.shape[1]>0):
        shown_ff_positions_rotate = np.matmul(rotation_matrix, shown_ff_positions_rotate)
    if show_reward_boundary:     
        if show_alive_fireflies:
            boundary_centers_rotated = alive_ff_position_rotate
        else:
            boundary_centers_rotated = shown_ff_positions_rotate
        for i in boundary_centers_rotated.T:
            circle = plt.Circle((i[0]-x0, i[1]-y0), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
            axes.add_patch(circle)



    if show_eye_positions:
        gaze_world_xy_rotate, gaze_monkey_view_xy_rotate, overall_valid_indices = eye_positions.find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix=R)
        used_cum_t = cum_indices[overall_valid_indices]
        gaze_world_xy_rotate_valid = gaze_world_xy_rotate[:, overall_valid_indices]
        axes.scatter(gaze_world_xy_rotate_valid[0]-x0, gaze_world_xy_rotate_valid[1]-y0, marker='o', c=used_cum_t, s=7, zorder=2, cmap='viridis')


        if show_connect_path_eye_positions:  
            sample = np.arange(1, gaze_world_xy_rotate_valid.shape[1], 4)
            axes = connect_points_to_points(axes, gaze_world_xy_rotate_valid[:, sample], cum_mxy_rotate[:,overall_valid_indices][:, sample], 
                                            x0, y0, color = "black", alpha = connection_alpha[player], linewidth=connection_linewidth[player], show_dots=False)


    if show_eye_positions_on_the_right:
        if gaze_monkey_view_xy_rotate is None:
            gaze_world_xy_rotate, gaze_monkey_view_xy_rotate, overall_valid_indices = eye_positions.find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix=R)
            used_cum_t = cum_indices[overall_valid_indices]
        axes2 = fig.add_subplot(1, 2, 2)
        axes2.scatter(gaze_monkey_view_xy_rotate[0, overall_valid_indices], gaze_monkey_view_xy_rotate[1, overall_valid_indices], s=7, c=used_cum_t, cmap='viridis')
        mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(gaze_world_xy_rotate[:, overall_valid_indices], x0, y0, temp_ff_positions=None)
        axes2 = set_xy_limits_for_axes(axes2, mx_min, mx_max, my_min, my_max, minimal_margin, zoom_in)
        fig.tight_layout()
        # plot a horizontal and a vertical line at origin
        axes2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes2.axvline(x=0, color='k', linestyle='--', linewidth=1)
 

    colorbar_max_value = None
    if show_trajectory:
        axes = show_trajectory_func(axes, player, cum_indices, cum_mxy_rotate, cum_t, cum_speed, monkey_information, 
                    x0, y0, trail_color_var, show_eye_positions, subplots)
        # Some other procedures
        if trail_color_var is None:
            # make a proxy to use legend
            line = Line2D([0], [0], linestyle="-", alpha=0.9, linewidth=2, color="black")
            legend_markers.append(line)
            legend_names.append('Monkey trajectory')
        elif trail_color_var == 'abs_ddw':
            cum_abs_ddw = np.abs(np.array(monkey_information['monkey_ddw'].iloc[cum_indices]))
            colorbar_max_value = max(cum_abs_ddw)


    if show_null_agent_trajectory:
        axes = show_null_agent_trajectory_func(duration, null_agent_starting_time, monkey_information, ff_dataframe, \
                                               axes, R, assumed_memory_duration_of_agent, show_null_agent_trajectory_2nd_time)
        line = Line2D([0], [0], linestyle="-", alpha=0.7, linewidth=2, color="brown")
        legend_markers.append(line)
        legend_names.append('Null trajectory')

        line = Line2D([0], [0], linestyle="-", alpha=0.7, linewidth=2, color="green")
        legend_markers.append(line)
        legend_names.append('2nd Null trajectory')

    if show_scale_bar:
        axes = plot_scale_bar(axes)

    if not show_eye_positions:
        axes.xaxis.set_major_locator(mtick.NullLocator())
        axes.yaxis.set_major_locator(mtick.NullLocator())

    if show_colorbar:
        fig, axes = plot_colorbar_for_trials(fig, axes, trail_color_var, show_eye_positions=show_eye_positions, show_eye_positions_on_the_right=show_eye_positions_on_the_right, duration=duration, max_value=colorbar_max_value)


    if show_legend:
        axes.legend(legend_markers, legend_names, scatterpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    # Set the limits of the x-axis and y-axis
    if adjust_xy_limits:
        if show_eye_positions_on_the_right:
            mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(gaze_world_xy_rotate[:, overall_valid_indices], x0, y0, temp_ff_positions=None)
            axes = set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, minimal_margin, zoom_in)
        else:
            mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(cum_mxy_rotate, x0, y0, temp_ff_positions=shown_ff_positions_rotate)
            axes = set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, minimal_margin, zoom_in)

    
    if show_title:
        axes.set_title(f"Trial {currentTrial}", fontsize = 22)

    if images_dir is not None:
        filename = "trial_" + str(currentTrial)
        save_image(filename, images_dir)

    whether_plotted = True


    return whether_plotted, axes, R, cum_mxy_rotate, shown_ff_indices





############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################




def show_trajectory_func(axes, player, cum_indices, cum_mxy_rotate, cum_t, cum_speed, monkey_information, 
                    x0, y0, trail_color_var, show_eye_positions, subplots):
    trail_size = {"agent": 70, "monkey": 8, "combined": 2}
    trail_alpha = {"agent": 1, "monkey": 0.9, "combined": 0.5}
    if subplots == True:
        trail_size = {"agent": 10, "monkey": 10}
    if show_eye_positions:
        axes.scatter(cum_mxy_rotate[0]-x0, cum_mxy_rotate[1]-y0, marker='o', s=trail_size[player], alpha=trail_alpha[player], c=cum_t, cmap='viridis', zorder=3)
    elif trail_color_var == 'speed': # the color of the path will vary by speed
        axes.scatter(cum_mxy_rotate[0]-x0, cum_mxy_rotate[1]-y0, marker='o', s=trail_size[player], alpha=trail_alpha[player], c=cum_speed, cmap='viridis', zorder=3)
    elif trail_color_var == 'abs_ddw':
        cum_abs_ddw = np.abs(np.array(monkey_information['monkey_ddw'].iloc[cum_indices]))
        axes.scatter(cum_mxy_rotate[0]-x0, cum_mxy_rotate[1]-y0, marker='o', s=trail_size[player], alpha=trail_alpha[player], c=cum_abs_ddw, cmap='viridis_r', zorder=3)            
        # To mark the points where high abs_ddw occur:
        points_to_mark = np.where(cum_abs_ddw > 0.1)[0]
        axes.scatter(cum_mxy_rotate[0, points_to_mark]-x0, cum_mxy_rotate[1, points_to_mark]-y0, marker='*', s=160, alpha=trail_alpha[player], c="orange", zorder=1)
    elif trail_color_var == 'target_visibility': 
        axes.scatter(cum_mxy_rotate[0]-x0, cum_mxy_rotate[1]-y0, marker='o', s=trail_size[player], alpha=trail_alpha[player], color_var = None, zorder=3)   
    else:
        axes.plot(cum_mxy_rotate[0]-x0, cum_mxy_rotate[1]-y0, alpha=trail_alpha[player], c="black", zorder=3, linewidth=2)
    return axes



def show_null_agent_trajectory_func(duration, null_agent_starting_time, monkey_information, ff_dataframe, 
                                    axes, R, assumed_memory_duration_of_agent, show_null_agent_trajectory_2nd_time=False):
    if null_agent_starting_time is None:
        current_moment = duration[1]
    else:
        current_moment = null_agent_starting_time
    monkey_xy, monkey_angle = find_most_recent_monkey_position(monkey_information, current_moment)
    axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y \
        = plot_arc_from_null_condition(axes, current_moment, ff_dataframe, monkey_xy, monkey_angle, rotation_matrix=R, assumed_memory_duration=assumed_memory_duration_of_agent,
                                        arc_color="brown", reaching_boundary_ok=True)

    if show_null_agent_trajectory_2nd_time:
        if min_arc_ff_xy is not None:
            min_time = min_arc_length/200
            current_moment = duration[1]+min_time
            if center_x is not None:
                monkey_x, monkey_y, monkey_angle = find_monkey_position_after_an_arc(min_arc_ff_xy, min_arc_ff_angle, center_x, center_y, monkey_angle)
                monkey_xy = np.array([monkey_x, monkey_y])
            else: # the monkey just went a straight line
                monkey_xy = min_arc_ff_xy
            ff_dataframe_sub = ff_dataframe[(ff_dataframe['ff_x'] != min_arc_ff_center_xy[0])|(ff_dataframe['ff_y'] != min_arc_ff_center_xy[1])]
        else:
            current_moment = duration[1] + 2
            ff_dataframe_sub = ff_dataframe.copy()
        # eliminate the caught ff from ff_dataframe
        axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y \
            = plot_arc_from_null_condition(axes, current_moment, ff_dataframe_sub, monkey_xy, monkey_angle, rotation_matrix=R, assumed_memory_duration=2, \
                                            arc_color="darkgreen", reaching_boundary_ok=True)
    return axes



def customize_kwargs_by_category(classic_plot_kwargs, images_dir=None):
    classic_plot_kwargs['images_dir'] = images_dir

    visible_before_last_one_kwargs = classic_plot_kwargs.copy()
    disappear_latest_kwargs = classic_plot_kwargs.copy()
    two_in_a_row_kwargs = classic_plot_kwargs.copy()
    waste_cluster_around_target_kwargs = classic_plot_kwargs.copy()
    try_a_few_times_kwargs = classic_plot_kwargs.copy()
    give_up_after_trying_kwargs = classic_plot_kwargs.copy()
    ignore_sudden_flash_kwargs = classic_plot_kwargs.copy()
    
    visible_before_last_one_kwargs['show_connect_path_ff_except_targets'] = True
    visible_before_last_one_kwargs['show_path_when_target_visible'] = True
    disappear_latest_kwargs['show_connect_path_ff'] = True
    two_in_a_row_kwargs['show_connect_path_ff_except_targets'] = True
    two_in_a_row_kwargs['show_path_when_target_visible'] = True
    waste_cluster_around_target_kwargs['show_connect_path_ff'] = True
    waste_cluster_around_target_kwargs['trial_to_show_cluster_around_target'] = 'previous'
    try_a_few_times_kwargs['show_connect_path_ff'] = True
    give_up_after_trying_kwargs['show_connect_path_ff'] = True
    ignore_sudden_flash_kwargs['show_connect_path_ff'] = True

    all_category_kwargs = {'visible_before_last_one': visible_before_last_one_kwargs,
                            'disappear_latest': disappear_latest_kwargs,
                            'two_in_a_row': two_in_a_row_kwargs,
                            'waste_cluster_around_target': waste_cluster_around_target_kwargs,
                            'try_a_few_times': try_a_few_times_kwargs,
                            'give_up_after_trying': give_up_after_trying_kwargs,
                            'ignore_sudden_flash': ignore_sudden_flash_kwargs}  
    return all_category_kwargs







              
def connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color, alpha, linewidth, show_dots=True, dot_color="brown"):
    if temp_ff_positions.shape[1] > 0:
        for j in range(temp_ff_positions.shape[1]):
            axes.plot(np.stack([temp_ff_positions[0, j]-x0, temp_monkey_positions[0, j]-x0]),
                        np.stack([temp_ff_positions[1, j]-y0, temp_monkey_positions[1, j]-y0]), 
                        '-', alpha=alpha, linewidth=linewidth, c=color)
            if show_dots:
                # to mark the connected fireflies as brown circles
                axes.plot(temp_ff_positions[0, j]-x0, temp_ff_positions[1, j]-y0, alpha=0.2, marker="o", markersize=5, color=dot_color, zorder=2)
    return axes     




def find_monkey_information_in_the_duration(duration, monkey_information):
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    cum_t, cum_angles = np.array(monkey_information['monkey_t'].iloc[cum_indices]), np.array(monkey_information['monkey_angles'].iloc[cum_indices])
    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_indices]), np.array(monkey_information['monkey_y'].iloc[cum_indices])
    cum_speed, cum_speeddummy = np.array(monkey_information['monkey_speed'].iloc[cum_indices]), np.array(monkey_information['monkey_speeddummy'].iloc[cum_indices])
    return cum_indices, cum_t, cum_angles, cum_mx, cum_my, cum_speed, cum_speeddummy
    


def find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted, rotation_matrix=None):
    alive_ff_indices = np.array([ff_index for ff_index, life in enumerate(ff_life_sorted) if (life[-1] >= duration[0]) and (life[0] < duration[1])])
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]

    alive_ff_position_rotate = np.stack((alive_ff_positions.T[0], alive_ff_positions.T[1]))
    if (rotation_matrix is not None) & (alive_ff_position_rotate.shape[1]>0):
        alive_ff_position_rotate = np.matmul(rotation_matrix, alive_ff_position_rotate)

    return alive_ff_indices, alive_ff_position_rotate


def find_believed_target_positions(ff_believed_position_sorted, currentTrial, num_trials, rotation_matrix=None):
    believed_target_positions = ff_believed_position_sorted[currentTrial - num_trials + 1:currentTrial + 1]
    believed_target_positions_rotate = np.stack((believed_target_positions.T[0], believed_target_positions.T[1]))
    if (rotation_matrix is not None) & (believed_target_positions_rotate.shape[1]>0):
        believed_target_positions_rotate = np.matmul(rotation_matrix, believed_target_positions_rotate)
    return believed_target_positions_rotate


def find_stops_for_plotting(cum_mx, cum_my, cum_speeddummy, rotation_matrix=None):
    zerospeed_index = np.where(cum_speeddummy == 0)
    zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
    zerospeed_rotate = np.stack((zerospeedx, zerospeedy))
    if (rotation_matrix is not None) & (zerospeed_rotate.shape[1]>0):
        zerospeed_rotate = np.matmul(rotation_matrix, zerospeed_rotate)
        
    return zerospeed_rotate



def find_rotation_matrix(cum_mx, cum_my, also_return_angle=False):
    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(cum_my[-1]-cum_my[0], cum_mx[-1]-cum_mx[0])     
    c, s = np.cos(theta), np.sin(theta)
    # Rotation matrix
    R = np.array(((c, -s), (s, c)))
    if also_return_angle:
        return R, theta
    else:
        return R


def find_triangles_to_show_monkey_angles(cum_mx, cum_my, cum_angles, rotation_matrix=None):
    left_end_x = cum_mx + 30 * np.cos(cum_angles + 2*pi/9) 
    left_end_y = cum_my + 30 * np.sin(cum_angles + 2*pi/9)
    right_end_x = cum_mx + 30 * np.cos(cum_angles - 2*pi/9) 
    right_end_y = cum_my + 30 * np.sin(cum_angles - 2*pi/9)

    left_end_xy = np.stack((left_end_x, left_end_y), axis=1).T
    right_end_xy = np.stack((right_end_x, right_end_y), axis=1).T

    if (rotation_matrix is not None) & (left_end_xy.shape[1]>0):
        left_end_xy = np.matmul(rotation_matrix, left_end_xy)
        right_end_xy = np.matmul(rotation_matrix, right_end_xy)

    return left_end_xy, right_end_xy



def find_path_when_target_visible(target_index, ff_dataframe_in_duration, cum_indices, visible_distance, rotation_matrix=None):
    temp_df = ff_dataframe_in_duration
    temp_df = temp_df.loc[(temp_df['ff_index'] == target_index) & (temp_df['visible'] == 1) & (temp_df['ff_distance'] <= visible_distance)]
    temp_df = temp_df[(temp_df['point_index'] >= cum_indices[0]) & (temp_df['point_index'] <= cum_indices[-1])]
    ff_visible_path_rotate = np.array(temp_df[['monkey_x', 'monkey_y']]).T
    if (rotation_matrix is not None) & (ff_visible_path_rotate.shape[1]>0):
        ff_visible_path_rotate = np.matmul(rotation_matrix, ff_visible_path_rotate)
    return ff_visible_path_rotate



def find_lines_to_connect_path_ff(ff_dataframe_in_duration, target_indices, rotation_matrix=None, target_excluded = False):
    temp_df = ff_dataframe_in_duration
    
    if target_excluded:
        # if the player is monkey, then the following code is used to avoid the lines between the monkey's position and the target since the lines might obscure the path
        temp_df = temp_df.loc[~temp_df['ff_index'].isin(target_indices)]
    temp_array = temp_df[['ff_x', 'ff_y', 'monkey_x', 'monkey_y']].to_numpy()

    temp_ff_positions_rotate = temp_array[:, :2].T
    temp_monkey_positions_rotate = temp_array[:, 2:].T
    if (rotation_matrix is not None) & (temp_ff_positions_rotate.shape[1]>0):
        temp_ff_positions_rotate = np.matmul(rotation_matrix, temp_ff_positions_rotate)
        temp_monkey_positions_rotate = np.matmul(rotation_matrix, temp_monkey_positions_rotate)

    return temp_ff_positions_rotate, temp_monkey_positions_rotate



def find_ff_in_cluster(cluster_dataframe_point, ff_real_position_sorted, currentTrial, rotation_matrix=None):
    # Find the indices of ffs in the cluster
    cluster_indices = cluster_dataframe_point[cluster_dataframe_point['target_index'] == currentTrial].ff_index
    cluster_indices = np.unique(cluster_indices.to_numpy())
    cluster_ff_positions = ff_real_position_sorted[cluster_indices]
    cluster_ff_rotate = np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1]))
    if (rotation_matrix is not None) & (cluster_ff_rotate.shape[1] > 0):
        cluster_ff_rotate = np.matmul(rotation_matrix, cluster_ff_rotate)
    return cluster_ff_rotate


def find_ff_in_cluster_around_target(cluster_around_target_indices, ff_real_position_sorted, currentTrial, rotation_matrix=None):
    cluster_ff_indices = cluster_around_target_indices[currentTrial]
    cluster_ff_positions = ff_real_position_sorted[cluster_ff_indices]
    cluster_around_target_rotate = np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1]))
    if (rotation_matrix is not None) & (cluster_around_target_rotate.shape[1] > 0):
        cluster_around_target_rotate = np.matmul(rotation_matrix, cluster_around_target_rotate)
    return cluster_ff_indices, cluster_around_target_rotate



def find_path_when_ff_in_cluster_visible(ff_dataframe_in_duration, ff_index, rotation_matrix=None):
    temp_df = ff_dataframe_in_duration
    temp_df = temp_df.loc[(temp_df['ff_index'] == ff_index) & (temp_df['visible'] == 1)]
    monkey_xy_rotate = np.array(temp_df[['monkey_x', 'monkey_y']]).T
    ff_position_rotate = np.array(temp_df[['ff_x', 'ff_y']]).T
    if (rotation_matrix is not None) & (monkey_xy_rotate.shape[1]>0):
        monkey_xy_rotate = np.matmul(rotation_matrix, monkey_xy_rotate)
        ff_position_rotate = np.matmul(rotation_matrix, ff_position_rotate)
    return monkey_xy_rotate, ff_position_rotate



def visualize_monkey_angles_using_triangles(axes, cum_mxy_rotate, left_end_xy_rotate, right_end_xy_rotate, linewidth=0.5):
    for point in range(cum_mxy_rotate.shape[1]):
        middle = cum_mxy_rotate[:, point]
        left_end = left_end_xy_rotate[:, point]
        right_end = right_end_xy_rotate[:, point]
        # Only show the left side of the triangle
        axes.plot(np.array([middle[0], left_end[0]]), np.array([middle[1], left_end[1]]), linewidth = linewidth)
        axes.plot(np.array([middle[0], right_end[0]]), np.array([middle[1], right_end[1]]), linewidth = linewidth)
    return axes

def plot_scale_bar(axes):
    scale = ScaleBar(dx=1, units='cm', length_fraction=0.2, fixed_value=100, location='upper left', label_loc='left', scale_loc='bottom') 
    axes.add_artist(scale)
    return axes



def plot_colorbar_for_trials(fig, axes, trail_color_var, duration, show_eye_positions=False, show_eye_positions_on_the_right=False, max_value=None):

    width = {True: 0.025, False: 0.05}
    bottom = {True: 0.47, False: 0.4}

    cmap = (matplotlib.colors.ListedColormap(['black', 'red']))
    bounds = [0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    labels = np.array(["No Reward", "Reward"])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=fig.add_axes([0.95, 0.12, width[show_eye_positions_on_the_right], 0.2]), # left, bottom, width, height
        ticks=[1, 2],
        spacing='uniform',
        orientation='vertical',
    )    
    
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(size=0, color='white')
    cbar.ax.set_title('Stopping Points', ha='left', y=1.06)


    # Then make the colorbar to show the meaning of color of the monkey/agent's path
    if show_eye_positions or (trail_color_var == 'speed') or (trail_color_var == 'abs_ddw'):
        cmap = cm.viridis 
        if show_eye_positions: 
            vmax = duration[1]-duration[0]
            title = 'Time (s)'
        elif trail_color_var == 'speed': 
            vmax = 200
            title = 'Speed(cm/s)'
        else:
            cmap = cm.viridis_r
            vmax = max_value
            title = 'Angular cceleration (radians/s^2)'
        norm2 = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=fig.add_axes([0.95, bottom[show_eye_positions_on_the_right], width[show_eye_positions_on_the_right], 0.43]), orientation='vertical')
        cbar2.outline.set_visible(False)
        cbar2.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
        cbar2.ax.set_title(title, ha='left', y=1.04)


    elif trail_color_var == "target_visibility":
        cmap = (matplotlib.colors.ListedColormap(['green', 'orange']))
        bounds = [0.5, 1.5, 2.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        labels = np.array(["Top Target Visible", "Top Target Not Visible"])
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=fig.add_axes([0.95, 0.5, 0.05, 0.2]), # left, bottom, width, height
            ticks=[1, 2],
            spacing='uniform',
            orientation='vertical',
        )    
        cbar.ax.set_yticklabels(labels)
        cbar.ax.tick_params(size=0, color='white')
        cbar.ax.set_title('Path Color', ha='left', y=1.06)
    return fig, axes




def find_xy_min_max_for_plots(cum_mxy_rotate, x0, y0, temp_ff_positions=None):
    mx_min, mx_max = min(cum_mxy_rotate[0])-x0, max(cum_mxy_rotate[0])-x0
    my_min, my_max = min(cum_mxy_rotate[1])-y0, max(cum_mxy_rotate[1])-y0
    if temp_ff_positions is not None:
        if temp_ff_positions.shape[1] > 0:
            mx_min, mx_max = min(mx_min, min(temp_ff_positions[0])-x0), max(mx_max, max(temp_ff_positions[0])-x0)
            my_min, my_max = min(my_min, min(temp_ff_positions[1])-y0), max(my_max, max(temp_ff_positions[1])-y0)
    return mx_min, mx_max, my_min, my_max



def set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, minimal_margin=50, zoom_in=False):
    bigger_width = max(mx_max - mx_min, my_max - my_min)
    margin = max(bigger_width/10, minimal_margin)
    xmiddle, ymiddle = (mx_min + mx_max)/ 2, (my_min + my_max) / 2
    xmin, xmax = xmiddle - bigger_width/2, xmiddle + bigger_width/2
    ymin, ymax = ymiddle - bigger_width/2, ymiddle + bigger_width/2

    axes.set_aspect('equal')
    if zoom_in is True:
        axes.set_xlim((xmin - 40, xmax + 40))
        axes.set_ylim((ymin - 20, ymax + 60))
    else:
        axes.set_xlim((xmin - margin, xmax + margin))
        axes.set_ylim((ymin - margin*2/3, ymax + margin*4/3))
    return axes



def readjust_xy_limits_for_axes(axes, cum_mxy_rotate_1, cum_mxy_rotate_2, shown_ff_indices_1, shown_ff_indices_2, R, ff_real_position_sorted, minimal_margin=50):
    cum_mxy_rotate_all = np.concatenate((cum_mxy_rotate_1, cum_mxy_rotate_2), axis=1)
    shown_ff_positions_rotate_1 = ff_real_position_sorted[shown_ff_indices_1].T
    shown_ff_positions_rotate_1 = np.matmul(R, shown_ff_positions_rotate_1)
    shown_ff_positions_rotate_2 = ff_real_position_sorted[shown_ff_indices_2].T
    shown_ff_positions_rotate_2 = np.matmul(R, shown_ff_positions_rotate_2)
    shown_ff_positions_rotate_all = np.concatenate((shown_ff_positions_rotate_1, shown_ff_positions_rotate_2), axis=1)
    mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(cum_mxy_rotate_all, x0=0, y0=0, temp_ff_positions=shown_ff_positions_rotate_all)
    axes = set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, minimal_margin=minimal_margin)
    return axes



# def set_xy_limits_for_plots(axes, cum_mxy_rotate, x0, y0, minimal_margin, zoom_in=False, temp_ff_positions=None):

#     # Set the limits of the x-axis and y-axis
#     mx_min, mx_max = min(cum_mxy_rotate[0])-x0, max(cum_mxy_rotate[0])-x0
#     my_min, my_max = min(cum_mxy_rotate[1])-y0, max(cum_mxy_rotate[1])-y0

#     if temp_ff_positions is not None:
#         mx_min, mx_max = min(mx_min, min(temp_ff_positions[0])-x0), max(mx_max, max(temp_ff_positions[0])-x0)
#         my_min, my_max = min(my_min, min(temp_ff_positions[1])-y0), max(my_max, max(temp_ff_positions[1])-y0)
    
#     bigger_width = max(mx_max - mx_min, my_max - my_min)
#     margin = max(bigger_width/10, minimal_margin)
#     xmiddle, ymiddle = (mx_min + mx_max)/ 2, (my_min + my_max) / 2
#     xmin, xmax = xmiddle - bigger_width/2, xmiddle + bigger_width/2
#     ymin, ymax = ymiddle - bigger_width/2, ymiddle + bigger_width/2

#     axes.set_aspect('equal')
#     if zoom_in is True:
#         axes.set_xlim((xmin - 40, xmax + 40))
#         axes.set_ylim((ymin - 20, ymax + 60))
#     else:
#         axes.set_xlim((xmin - margin, xmax + margin))
#         axes.set_ylim((ymin - margin, ymax + margin))

#     return axes



def save_image(filename, images_dir):
    CHECK_FOLDER = os.path.isdir(images_dir)
    if not CHECK_FOLDER:
        os.makedirs(images_dir)
    plt.savefig(f"{images_dir}/{filename}.png")





def plot_ff_distribution_in_arena(ff_real_position_sorted, ff_life_sorted, ff_caught_T_sorted, images_dir=None):
    # divide total time length (or valid point_indices) into 9 parts
    # since some fireflies might not be flashing at the beginnin or end of the period, their "life" information at the beginning
    # and end are not complete. Thus, we chop off the beginning and the end of the period by 50s.
    max_time = ff_caught_T_sorted[-1] - 50
    min_time = ff_caught_T_sorted[0] + 50
    time_intervals = np.linspace(min_time, max_time, 9)
    num_ff_for_each_plot = []

    # plot a 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(9):
        time_point = time_intervals[i]
        duration = [time_point-0.1, time_point+0.1]
        # for each subplot, plot the ff distribution at a time point
        alive_ff_indices, alive_ff_position = find_alive_ff(duration, ff_life_sorted, ff_real_position_sorted)
        # plot alive_ff_position
        axes[i].scatter(alive_ff_position[0], alive_ff_position[1], marker='o', s=10, color="grey", zorder=2)
        num_ff_for_each_plot.append(len(alive_ff_indices))
    plt.show()
    plt.close()
    print(num_ff_for_each_plot)





def plot_a_trial(category_name, currentTrial, num_trials, ff_caught_T_sorted, PlotTrials_args, all_category_kwargs, additional_kwargs=None, images_dir=None):
    PlotTrials_kargs = all_category_kwargs[category_name]
    PlotTrials_kargs['images_dir'] = images_dir
    if additional_kwargs is not None:
        for key, value in additional_kwargs.items():
            PlotTrials_kargs[key] = value
    
    duration = [ff_caught_T_sorted[currentTrial-num_trials], ff_caught_T_sorted[currentTrial]]
    whether_plotted, axes, R, cum_mxy_rotate, shown_ff_indices = PlotTrials(duration, 
                *PlotTrials_args,
                **PlotTrials_kargs,
                currentTrial = currentTrial,
                num_trials = num_trials,                   
                )
    return whether_plotted, axes, R, cum_mxy_rotate, shown_ff_indices
    
         


def plot_trials_from_a_category(category, category_name, max_trial_to_plot, PlotTrials_args, all_category_kwargs, 
                                ff_caught_T_sorted, additional_kwargs=None, images_dir=None, using_subplots=False):

    num_trials = 2 
    category = category[category > num_trials]
    k = 1 # only useful when using_subplots is True
    if category_name == 'disappear_latest' or category_name == 'ignore_sudden_flash':
        num_trials = 1
    
    num_trial_plotted = 0
    if len(category) > 0:
        with basic_func.initiate_plot(10,10,100):
            if using_subplots:
                fig = plt.figure()
            for currentTrial in category:
                if using_subplots:
                    axes = fig.add_subplot(2,2,k)
                    additional_kwargs = {'fig': fig, 'axes': axes, 'subplots': True}
                whether_plotted, _, _, _ = plot_a_trial(category_name, currentTrial, num_trials, ff_caught_T_sorted, PlotTrials_args, 
                                                        all_category_kwargs, additional_kwargs=additional_kwargs, images_dir=images_dir)
                if whether_plotted is True:
                    if using_subplots:
                        k += 1
                        if k == 5:
                            plt.show()
                            plt.close()
                            return
                    else:
                        plt.show()  
                        plt.close()
                        num_trial_plotted += 1
                        if num_trial_plotted >= max_trial_to_plot:
                            break 






def PlotPoints(point, 
               duration_of_trajectory, 
               monkey_information, 
               ff_dataframe, 
               ff_caught_T_sorted,
               ff_life_sorted, 
               ff_real_position_sorted, 
               ff_believed_position_sorted, 
               ff_flash_sorted, 
               fig = None, 
               axes = None,
               visible_distance = 250,
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
    ff_caught_T_sorted: np.array
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
    sns.set(style="white")

    time = np.array(monkey_information['monkey_t'])[point]
    duration = [time - duration_of_trajectory, time]
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))
    cum_t, cum_mx, cum_my = monkey_information['monkey_t'].iloc[cum_indices].values, monkey_information['monkey_x'].iloc[cum_indices].values, monkey_information['monkey_y'].iloc[cum_indices].values
    
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
        trial_num = np.digitize(time, ff_caught_T_sorted)
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
    axes.scatter(cum_mx, cum_my, s=15, c=cum_indices, cmap="Blues")

    # Set the limits of the x-axis and y-axis
    mx_min, mx_max, my_min, my_max = find_xy_min_max_for_plots(np.stack((cum_mx, cum_my)), x0=0, y0=0, temp_ff_positions=None)

    axes = set_xy_limits_for_axes(axes, mx_min, mx_max, my_min, my_max, 250, zoom_in=False)

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









def get_overall_lim(axes, axes2):
    """
    Get the x-limits and y-limits of the plots based on both the monkey data and the agent data


    Parameters
    ----------
    axes: obj
        axes for one plot (e.g. for the monkey data)
    axes2: obj
        axes for another plot (e.g. for the agent data)


    Returns
    -------
    overall_xmin: num
        the minimum value of the x-axis that will be shared by both plots
    overall_xmax: num
        the maximum value of the x-axis that will be shared by both plots     
    overall_ymin: num
        the minimum value of the y-axis that will be shared by both plots
    overall_ymax: num
        the maximum value of the y-axis that will be shared by both plots

    """

    monkey_xmin, monkey_xmax = axes.get_xlim()
    monkey_ymin, monkey_ymax = axes.get_ylim()
    agent_xmin, agent_xmax = axes2.get_xlim()
    agent_ymin, agent_ymax = axes2.get_ylim()

    overall_xmin = min(monkey_xmin, agent_xmin)
    overall_xmax = max(monkey_xmax, agent_xmax)
    overall_ymin = min(monkey_ymin, agent_ymin)
    overall_ymax = max(monkey_ymax, agent_ymax)
    return overall_xmin, overall_xmax, overall_ymin, overall_ymax





def update_plot_limits(xmin, ymin, xmax, ymax, cum_mxy_rotate):
    """
    Update the limits of the plot; usually used when multiple trajectories are plotted on the same plot

    Parameters
    ----------
    xmin: num
        the minimum of the x-axis
    xmax: num
        the maximum of the x-axis
    ymin: num
        the minimum of the y-axis
    ymax: num
        the maximum of the y-axis
    cum_mxy_rotate: array, with shape (2, n)
        contains the x, y coordinates of the monkey's positions on the trajectory after rotation of the plot.

    Returns
    ----------
    xmin: num
        the updated minimum of the x-axis
    xmax: num
        the updated maximum of the x-axis
    ymin: num
        the updatedminimum of the y-axis
    ymax: num
        the updated maximum of the y-axis

    """

    x0, y0 = cum_mxy_rotate[0][0], cum_mxy_rotate[1][0] 
    temp_xmin, temp_xmax = np.min(cum_mxy_rotate[0])-x0, np.max(cum_mxy_rotate[0])-x0
    temp_ymin, temp_ymax = np.min(cum_mxy_rotate[1])-y0, np.max(cum_mxy_rotate[1])-y0
    xmin, xmax = min(xmin, temp_xmin), max(xmax, temp_xmax)
    ymin, ymax = min(ymin, temp_ymin), max(ymax, temp_ymax)
    return xmin, ymin, xmax, ymax






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
        to be used to rotate the plot when plotting
    plotting_params: dict, optional
        keyword arguments to be passed into the PlotTrials function
    data_folder_name: str
        name or path of the folder to store the graph
    """  


    #===================================== Monkey =====================================
    sns.set(style="white")
    fig = plt.figure(figsize=(17, 12), dpi=125)
    axes = fig.add_subplot(121)

    PlotTrials(plot_whole_duration, 
            info_of_monkey['monkey_information'],
            info_of_monkey['ff_dataframe'], 
            info_of_monkey['ff_life_sorted'], 
            info_of_monkey['ff_real_position_sorted'], 
            info_of_monkey['ff_believed_position_sorted'], 
            info_of_monkey['cluster_around_target_indices'], 
            info_of_monkey['ff_caught_T_sorted'], 
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


    PlotTrials(agent_duration, 
              info_of_agent['monkey_information'],
              info_of_agent['ff_dataframe'], 
              info_of_agent['ff_life_sorted'], 
              info_of_agent['ff_real_position_sorted'], 
              info_of_agent['ff_believed_position_sorted'], 
              info_of_agent['cluster_around_target_indices'], 
              info_of_agent['ff_caught_T_sorted'], 
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

    overall_xmin, overall_xmax, overall_ymin, overall_ymax = get_overall_lim(axes, axes2)
    plt.setp([axes, axes2], xlim=[overall_xmin, overall_xmax], ylim=[overall_ymin, overall_ymax])

    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        plt.savefig(f"{data_folder_name}/side_by_side_{currentTrial}.png")
    plt.tight_layout()
    plt.show()
    plt.close()





def set_polar_background(ax, rmax, color_visible_area_in_background=True):
    """
    Decorate the canvas to set up the background for a polar plot


    Parameters
    ----------
    ax: obj
        the matplotlib axes object
    rmax: num
        radius of the polar plot

    Returns
    -------
    ax: obj
        the matplotlib axes object
    """  

    ax.set_theta_zero_location("N")
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(275)
    ax.set_xticks(ax.get_xticks()) # This is to prevent a warning
    # Draw the boundary of the monkey's vision (use width=np.pi*4/9 for 40 degrees of vision)
    if color_visible_area_in_background:
        ax.bar(0, rmax, width=np.pi*4/9, bottom=0.0, color="grey", alpha=0.1)
    return ax



def set_polar_background_for_plotting(ax, rmax, color_visible_area_in_background=True):
    """
    Set up certain parameters for plotting in the polar coordinates

    Parameters
    ----------
    ax: obj
        a matplotlib axes object 
    rmax: numeric
        the radius of the polar plot

    Returns
    -------
    ax: obj
        a matplotlib axes object 
    """
    if rmax < 150:
        ax.set_rticks(range(25, rmax+1, 25))
    ax.set_rlabel_position(292.5)
    ax = set_polar_background(ax, rmax, color_visible_area_in_background)
    labels = list(ax.get_xticks())
    labels[5], labels[6], labels[7] = -labels[3], -labels[2], -labels[1]
    labels_in_degrees = [str(int(math.degrees(label)))+ chr(176) for label in labels]
    ax.set_xticklabels(labels_in_degrees)
    return ax



def set_polar_background_for_animation(ax, rmax, color_visible_area_in_background=True):
    """
    Set up certain parameters for each frame for animation in the polar coordinates

    Parameters
    ----------
    ax: obj
        a matplotlib axes object 
    rmax: numeric
        the radius of the polar plot

    Returns
    -------
    ax: obj
        a matplotlib axes object 
    """
    ax = set_polar_background(ax, rmax, color_visible_area_in_background)
    ax.set_thetamin(-45)
    ax.set_thetamax(45)
    labels_in_degrees = [str(int(math.degrees(label)))+ chr(176) for label in list(ax.get_xticks())]
    ax.set_xticklabels(labels_in_degrees)
    return ax


def find_ff_distances_and_angles(ff_index, duration, ff_real_position_sorted, monkey_information, ff_radius=10):
    """
    Given the index of a firefly and a duration, find the corresponding distances (to the monkey/agent) and angles (to the monkey/agent)

    Parameters
    ----------
    ff_index: num
        index of the firefly 
    duration: list
        containing the start time and the end time
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies 
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_radius: num
        the reward boundary of each firefly


    Returns
    ----------
    ff_distances_and_angles: pd.Dataframe
        containing the distances of angles (to the boundary or to the center) of the fireflies in the duration
    """  
    # Find the indices in monkey information:
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_indices]), np.array(monkey_information['monkey_y'].iloc[cum_indices])
    cum_angle = np.array(monkey_information['monkey_angles'].iloc[cum_indices])
    
    distances_to_ff = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[ff_index], axis = 1)
    angles_to_ff = np.arctan2(ff_real_position_sorted[ff_index, 1]-cum_my, ff_real_position_sorted[ff_index, 0]-cum_mx)-cum_angle
    angles_to_ff = np.remainder(angles_to_ff, 2*pi)
    angles_to_ff[angles_to_ff > pi] = angles_to_ff[angles_to_ff > pi] - 2*pi
    # Adjust the angles according to the reward boundary
    angles_to_boundaries = np.absolute(angles_to_ff)-np.abs(np.arcsin(np.divide(ff_radius, np.maximum(distances_to_ff, ff_radius) ))) 
    angles_to_boundaries = np.sign(angles_to_ff) * np.clip(angles_to_boundaries, 0, pi)
    
    ff_distances_and_angles = {}
    ff_distances_and_angles['ff_distance'] = distances_to_ff
    ff_distances_and_angles['ff_angle'] = angles_to_ff
    ff_distances_and_angles['ff_angle_boundary'] = angles_to_boundaries
    ff_distances_and_angles['point_index'] = cum_indices
    ff_distances_and_angles = pd.DataFrame.from_dict(ff_distances_and_angles)
    return ff_distances_and_angles




def PlotPolar(duration,
              monkey_information,
              ff_dataframe, 
              ff_life_sorted,
              ff_real_position_sorted,
              ff_caught_T_sorted,
              ff_flash_sorted,
              rmax = 400,
              currentTrial = None, # Can be None; then it means all trials in the duration will be plotted
              num_trials = None,
              color_visible_area_in_background = True, 
              hitting_arena_edge_ok = False,
              show_visible_ff = False,
              show_ff_in_memory = False,
              show_alive_ff = False,
              show_visible_target = False,
              show_target_in_memory = False,
              show_target_throughout_duration = False,
              show_legend = True,
              show_colorbar = True,
              show_target_at_being_caught = True,
              colors_show_overall_time = False, # If True, then visible and invisible-but-in-memory fireflies shall use different cmap; it's recommended not to show in-memory-alone ff
              connect_dots = False,
              return_axes = False,
              show_all_positions_of_all_fireflies = False,
              ff_colormap = "Reds", # or "viridis"
              target_colormap = "Greens", # or viridis
              size_increase_for_visible_ff = 25,
              fig = None,
              ax = None,
               ):



    """
    Plot the positions of the fireflies from the monkey's perspective (the monkey is always at the origin of the polar plot)


    Parameters
    ----------
    duration: list
        the duration to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    ff_caught_T_sorted: np.array
        containing the time when each captured firefly gets captured
    rmax: num
        the radius of the polar plot
    currentTrial: numeric
        the number of current trial
    num_trials: numeric
        the number of trials to be plotted
    hitting_arena_edge_ok: bool
        whether to continue to plot the trial if the boundary is hit at any point
    show_alive_ff: bool
        whether to show fireflies (other than the target) that are alive
    show_visible_ff: bool
        whether to show fireflies (other than the target) that are visible
    show_ff_in_memory: bool
        whether to show fireflies (other than the target) that are in memory
    show_visible_target: bool
        whether to show the target when it is visible
    show_target_in_memory: bool
        whether to show the target when it is in memory
    show_target_throughout_duration: bool
        whether to show the target as grey throughout the duration whenever the target is not shown otherwise
    show_legend: bool
        whether to show a legend   
    show_colorbar: bool
        whether to show the color bars
    """  

    currentTrial, num_trials, duration = basic_func.find_currentTrial_or_num_trials_or_duration(ff_caught_T_sorted, currentTrial, num_trials, duration)
    target_indices = np.arange(currentTrial-num_trials+1, currentTrial+1)
    sns.set(style="white")
    if duration[1]-duration[0] == 0:
        return

    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if not hitting_arena_edge_ok: # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_crossing_boundary = np.array(monkey_information['crossing_boundary'].iloc[cum_indices])
        if (np.any(cum_crossing_boundary == 1)):
            print("Current plot is omitted because the monkey has crossed the boundary at some point.")
            return
        
    if fig is None:
        fig = plt.figure(figsize=(7, 7))
    if ax is None:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax = set_polar_background_for_plotting(ax, rmax, color_visible_area_in_background=color_visible_area_in_background)

    ff_dataframe = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]
    if show_all_positions_of_all_fireflies:
        # Make a new ff_dataframe using the new function make_ff_dataframe_v2_func
        ff_dataframe_v2 = make_ff_dataframe.make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_sorted, ff_flash_sorted,  
                        ff_real_position_sorted, ff_life_sorted, max_distance = 400, 
                        data_folder_name = None, print_progress = False) 
        # add the information about memory from ff_dataframe to ff_dataframe_v2 by merging the two dataframes
        ff_dataframe = ff_dataframe_v2.merge(ff_dataframe[['ff_index', 'memory']], on='ff_index', how='left')
        # fill up the na in "memory" with 0
        ff_dataframe['memory'] = ff_dataframe['memory'].fillna(0)  
      
    target_info = ff_dataframe[ff_dataframe["ff_index"].isin(target_indices)]

    if show_visible_target:
        # then we can separate out non-target fireflies
        ff_info = ff_dataframe[~ff_dataframe["ff_index"].isin(target_indices)]
    else:
        # ff_info shall include all ff
        ff_info = ff_dataframe.copy()
    
    if not show_all_positions_of_all_fireflies:
        if not show_ff_in_memory:
            ff_info = ff_info[(ff_info['visible'] != 0)]
        if not show_visible_ff:
            ff_info = ff_info[(ff_info['visible'] != 1)]
        if not show_target_in_memory:
            target_info = target_info[(target_info['visible'] != 0)]
        if not show_visible_target:
            target_info = target_info[(target_info['visible'] != 1)]


    if colors_show_overall_time:
        num_color_elements = len(cum_indices)+1
    else:
        num_color_elements = 101

    colors_ffs = plt.get_cmap(ff_colormap)(np.linspace(0, 1, num_color_elements))
    colors_target = plt.get_cmap(target_colormap)(np.linspace(0, 1, num_color_elements))


    if colors_show_overall_time:
        # color is based on time into the past
        ff_color = colors_ffs[cum_indices[-1] - np.array(ff_info['point_index'].astype('int'))]
        target_color = colors_target[cum_indices[-1] - np.array(target_info['point_index'].astype('int'))]
    else:
        # color is based on memory
        ff_color = colors_ffs[np.array(ff_info['memory'].astype('int'))]
        target_color = colors_target[np.array(target_info['memory'].astype('int'))]



    if show_alive_ff & (not show_all_positions_of_all_fireflies):
        ff_dataframe_v2 = make_ff_dataframe.make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_sorted, ff_flash_sorted,  
                                ff_real_position_sorted, ff_life_sorted, max_distance = 400, 
                                data_folder_name = None, print_progress = False) 
        ax.scatter(ff_dataframe_v2['ff_angle'], ff_dataframe_v2['ff_distance'], c='grey', s=5, alpha=0.2, zorder=1, marker='o')   

    # Visualize ff_info
    ax.scatter(ff_info['ff_angle'], ff_info['ff_distance'], c=ff_color, alpha=0.7, zorder=2, s=ff_info['visible']*size_increase_for_visible_ff+5, marker='o') # originally it was s=15

    if connect_dots:
        for ff_index in ff_dataframe.ff_index.unique():
            current_ff = ff_dataframe[ff_dataframe['ff_index']==ff_index]
            ax.plot(current_ff['ff_angle'], current_ff['ff_distance'], alpha=0.7, zorder=1)


    plotted_points = [] # store the indices of points that have been plotted so that if show_target_throughout_duration, one know which points to exclude
    if len(target_info) > 0:
        ax.scatter(target_info['ff_angle'], target_info['ff_distance'], c=target_color, alpha=0.7, s=target_info['visible']*20+5)     
        plotted_points = target_info['point_index']


    if show_target_throughout_duration:
        if not show_all_positions_of_all_fireflies:
            ff_distances_and_angles = find_ff_distances_and_angles(currentTrial, duration, ff_real_position_sorted, monkey_information)
            ff_distances_and_angles = ff_distances_and_angles[~ff_distances_and_angles['point_index'].isin(plotted_points)]
            ax.scatter(ff_distances_and_angles['ff_angle'], ff_distances_and_angles['ff_distance'], c='grey', s=15, alpha=0.2, zorder=1, marker='o')   

    if show_target_at_being_caught:
        target_info_at_being_caught = target_info[target_info['whether_caught']==1]  
        ax.scatter(target_info_at_being_caught['ff_angle'], target_info_at_being_caught['ff_distance'], alpha=0.7, marker = '*', c = 'red', s = 70)


    if show_legend:
        if show_all_positions_of_all_fireflies or (ff_colormap==target_colormap):
            markers = [[8, 'o', 'green'], [5, 'o', 'green']]
            legend_labels = ['Visible', 'Invisible']
            if show_target_at_being_caught & (len(target_info_at_being_caught) > 0):
                markers.append([15, '*', 'red'])
                legend_labels.append('Caught')
            lines = [Line2D([0], [0], marker=param[1], markersize=param[0], color='w', markerfacecolor=param[2]) for param in markers]
            ax.legend(lines, legend_labels, loc='lower right')
        else:
            ax = add_legend_for_polar_plot(ax, show_visible_ff, show_ff_in_memory, show_alive_ff, show_visible_target, show_target_in_memory, show_target_throughout_duration, colors_show_overall_time)

    if show_colorbar:
        fig = add_colorbar_for_polar_plot(fig, duration, show_ff_in_memory, show_target_in_memory, ff_colormap, target_colormap, colors_show_overall_time, show_all_positions_of_all_fireflies)         

    if return_axes:
        return fig, ax
    else:
        plt.show()
        plt.close()







def add_legend_for_polar_plot(ax,              
                              show_visible_ff,
                              show_ff_in_memory,
                              show_alive_ff,
                              show_visible_target,
                              show_target_in_memory,
                              show_target_throughout_duration,
                              colors_show_overall_time):
    

    # colors = ['green', 'red']
    # labels = ['Captured firefly', 'Other fireflies']


    colors = []
    labels = []
    if show_visible_target or show_target_in_memory or show_target_throughout_duration:
        colors.append('green')
        labels.append('Target firefly')
    if show_visible_ff or show_ff_in_memory:
        colors.append('red')
        labels.append('Non-target fireflies')
    # if colors_show_overall_time:
    #     colors.append('blue')
    #     labels.append('Target firefly in memory')
    #     colors.append('purple')
    #     labels.append('Non-target fireflies in memory')
    if show_alive_ff:
        colors.append('grey')
        labels.append('Alive but invisible fireflies')

    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    ax.legend(lines, labels, loc='lower right')
    return ax







def add_colorbar_for_polar_plot(fig, duration, show_ff_in_memory, show_target_in_memory, ff_colormap, target_colormap, colors_show_overall_time, show_all_positions_of_all_fireflies):            
    
    
    if colors_show_overall_time:
        vmax = duration[1]-duration[0]
        title = 'Time into the Past (s)'
        # cbar_xticks = [0, vmax]
        # cbar_labels = ['Least recent', 'Most recent']
    else:
        vmax = 100 # the maximum value of memory is 100
        title = 'Time since firefly visible'
        cbar_xticks = [0, 20, 40, 60, 80, 100]
        cbar_labels = ['1.67s since visible', '1.33s since visible', '1s since visible', '0.67s since visible', '0.33s since visible', 'Visible']          
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    plotted_colorbar_for_ff = False
    if show_ff_in_memory or colors_show_overall_time or show_all_positions_of_all_fireflies:
        plotted_colorbar_for_ff = True
        cax = fig.add_axes([0.95, 0.05, 0.05, 0.4]) #[left, bottom, width, height] 
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(ff_colormap)), cax=cax, orientation='vertical')
        cbar.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
        cbar.ax.set_title(title, ha='left', y=1.04)
        if ff_colormap == 'viridis':
            cbar.outline.set_visible(False)        
        if not colors_show_overall_time:
            cbar.ax.set_yticks(cbar_xticks) # This is to prevent a warning
            cbar.ax.set_yticklabels(cbar_labels) 

    
    if show_target_in_memory or colors_show_overall_time or show_all_positions_of_all_fireflies:
        if (ff_colormap == target_colormap) & plotted_colorbar_for_ff:
            # If the colormap for ff and target are the same, then there's no need to plot another colorbar
            pass
        else:
            cax2 = fig.add_axes([0.88, 0.05, 0.05, 0.4])
            cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(target_colormap)), cax=cax2, orientation='vertical')
            cbar2.ax.set_yticks([])
            if target_colormap == 'viridis':
                cbar2.outline.set_visible(False)
            if not (plotted_colorbar_for_ff):    
                # Put labels at the right side of the colorbar for the target since there's no colorbar for the non-target fireflies
                cbar2.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5) 
                cbar2.ax.set_title(title, ha='left', y=1.04)
                if not colors_show_overall_time:
                    cbar2.ax.set_yticks(cbar_xticks) # This is to prevent a warning
                    cbar2.ax.set_yticklabels(cbar_labels) 

    return fig




# def add_colorbar_for_polar_plot(fig): # A more simple one
#     cax = fig.add_axes([0.95, 0.05, 0.05, 0.4]) # [left, bottom, width, height] 
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
#     cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.Reds), cax=cax, orientation='vertical')
#     cbar.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
#     cbar_labels = ['Visible 1.67s Ago', 'Visible 1.33s Ago', 'Visible 1s Ago', 'Visible 0.67s Ago',  'Visible 0.33s Ago', 'Visible']
#     cbar.ax.set_yticks([0, 20, 40, 60, 80, 100]) # This is to prevent a warning
#     cbar.ax.set_yticklabels(cbar_labels)
#     # cbar.ax.set_title('Visibility', ha='left', x=0, y=1.05)
#     # cbar.ax.set_yticks([0,20,40,60,80,100])
    
#     cax2 = fig.add_axes([0.90, 0.05, 0.05, 0.4])
#     cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.YlGn), cax=cax2, orientation='vertical')
#     cbar2.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
#     return fig




def plot_change_in_ff_angle(ff_dataframe, trial_numbers, var_of_interest = "abs_ffangle_decreasing"):
    sns.set(style="white")
    var = var_of_interest
    var_str = var + "_str"
    ff_dataframe[var_str] = "No Change"
    ff_dataframe.loc[ff_dataframe[var] > 0, var_str] = "Yes"
    ff_dataframe.loc[ff_dataframe[var] < 0, var_str] = "No"
    hue_order = ['Yes', 'No', 'No Change']
    plt.rcdefaults()
    for currentTrial in trial_numbers:
        info_for_currentTrial = ff_dataframe[ff_dataframe['target_index'] == currentTrial]

        sns.stripplot(data=info_for_currentTrial, x="ff_index_string", y="time", hue_order=hue_order,
                          hue=var_str, jitter=False, palette="deep")

        # Mark the xticklabel for the target red
        which_ff_is_target = np.where(info_for_currentTrial["ff_index"].unique() == currentTrial)[0]
        if len(which_ff_is_target) > 0:
            plt.gca().get_xticklabels()[which_ff_is_target[0]].set_color('red') 

        plt.xlabel("Firefly index")
        plt.ylabel("Time (s)")
        if var_of_interest == "abs_ffangle_decreasing":
            plt.title("Whether Absolute Firefly's Angle is Decreasing")
            plt.legend(title="Whether decreasing")
        if var_of_interest == "abs_ffangle_boundary_decreasing":
            plt.title("Whether Absolute Firefly's Angle to Boundary is Decreasing")
            plt.legend(title="Whether decreasing")
        if var_of_interest == "dw_same_sign_as_ffangle":
            plt.title("Whether Angular V is Same Direction as fireflies")
            plt.legend(title="Whether same direction")
        if var_of_interest == "dw_same_sign_as_ffangle_boundary":
            plt.title("Whether Angular V is Same Direction as fireflies (Using Angle to Boundary)")
            plt.legend(title="Whether same direction")
        plt.show()
        plt.close()



def find_relative_xy_positions(ff_x, ff_y, monkey_x, monkey_y, monkey_angle):
    ff_xy = np.stack((ff_x, ff_y), axis=1)
    monkey_xy = np.array([monkey_x, monkey_y])
    ff_distances = LA.norm(ff_xy-monkey_xy, axis = 1)
    ff_angles = basic_func.calculate_angles_to_ff_centers(ff_x=ff_x, ff_y=ff_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
    ff_x_relative = np.cos(ff_angles+pi/2)*ff_distances
    ff_y_relative = np.sin(ff_angles+pi/2)*ff_distances
    return ff_x_relative, ff_y_relative



def find_shortest_arc(ff_x, ff_y, monkey_x, monkey_y, monkey_angle):
    # eliminate the ff whose relative y positive to the monkey is negative (a.k.a. behind the monkey)
    
    monkey_xy = np.array([monkey_x, monkey_y])
    ff_xy = np.stack([ff_x, ff_y], axis=1)

    ff_distances = LA.norm(ff_xy-monkey_xy, axis = 1)
    ff_angles = basic_func.calculate_angles_to_ff_centers(ff_x=ff_x, ff_y=ff_y, mx=monkey_x, my=monkey_y, m_angle=monkey_angle)
    ff_angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=ff_angles, distances_to_ff=ff_distances)
    #ff_x_relative = np.cos(ff_angles+pi/2)*ff_distances
    ff_y_relative = np.sin(ff_angles+pi/2)*ff_distances


     # supress warnings because invalid values might occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arc_angle = pi - 2* np.arcsin(np.abs(ff_y_relative/ff_distances))
        arc_radius = ff_y_relative/np.sin(np.minimum(arc_angle, pi-arc_angle))
        arc_length = arc_radius*arc_angle   


    na_index = np.where(np.isnan(arc_length))[0]
    arc_radius[na_index] = 0
    arc_length[na_index] = ff_distances[na_index]


    # Find the shortest arc length
    min_arc_length = min(arc_length)
    rel_index = np.where(arc_length == min_arc_length)[0]
    min_arc_radius = arc_radius[rel_index]
    min_arc_ff_xy = ff_xy[rel_index].reshape(-1)
    min_arc_ff_distance = ff_distances[rel_index]
    min_arc_ff_angle = ff_angles[rel_index]
    min_arc_ff_angle_boundary = ff_angles_to_boundaries[rel_index]

    # find elements of na in arc_length
    
    return min_arc_length, min_arc_radius, min_arc_ff_xy, min_arc_ff_distance, min_arc_ff_angle, min_arc_ff_angle_boundary


    
def find_elements_to_plot_an_arc(monkey_xy, monkey_angle, min_arc_radius, min_arc_ff_xy, min_arc_ff_angle):
    monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    angle_to_center_of_arc_from_monkey = monkey_angle + np.sign(min_arc_ff_angle)*pi/2
    center_x = np.cos(angle_to_center_of_arc_from_monkey)*min_arc_radius + monkey_x
    center_y = np.sin(angle_to_center_of_arc_from_monkey)*min_arc_radius + monkey_y
    angle_from_center_to_ff = np.arctan2(min_arc_ff_xy[1]-center_y, min_arc_ff_xy[0]-center_x)
    angle_from_center_to_monkey = np.arctan2(monkey_y-center_y, monkey_x-center_x)
    if min_arc_ff_angle > 0: # ff is to the left of the monkey
        arc_starting_angle = angle_from_center_to_monkey
        arc_ending_angle = angle_from_center_to_ff
    else:
        arc_starting_angle = angle_from_center_to_ff
        arc_ending_angle = angle_from_center_to_monkey

    while arc_ending_angle < arc_starting_angle:
        arc_ending_angle += 2*pi

    return center_x, center_y, arc_starting_angle, arc_ending_angle



def find_monkey_position_after_an_arc(min_arc_ff_xy, min_arc_ff_angle, center_x, center_y, monkey_angle):
    angle_from_center_to_ff = np.arctan2(min_arc_ff_xy[1]-center_y, min_arc_ff_xy[0]-center_x)
    # if angle_from_center_to_ff is the starting angle, then monkey_angle at the end of the trajectory should be angle_from_center_to_ff - pi/2
    if min_arc_ff_angle < 0: # ff is to the right of the monkey
        monkey_angle = angle_from_center_to_ff - pi/2
    # otherwise, it should be angle_from_center_to_ff + pi/2
    else:
        monkey_angle = angle_from_center_to_ff + pi/2
    # And monkey_xy should be the same as ff_xy
    monkey_x = min_arc_ff_xy[0]
    monkey_y = min_arc_ff_xy[1]
    return monkey_x, monkey_y, monkey_angle



def find_arc_xy_rotated(center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, rotation_matrix=None):
    # Plot an arc from arc_starting_angle to arc_ending_angle with arc_radius  
    angle_array = np.linspace(arc_starting_angle, arc_ending_angle, 100).reshape(-1)
    arc_x_array = np.cos(angle_array)*min_arc_radius + center_x
    arc_y_array = np.sin(angle_array)*min_arc_radius + center_y
    arc_xy_rotated = np.stack((arc_x_array, arc_y_array))
    if rotation_matrix is not None:
        arc_xy_rotated = np.matmul(rotation_matrix, arc_xy_rotated)
    return arc_xy_rotated




def find_most_recent_monkey_position(monkey_information, current_moment):
    duration = [current_moment-2, current_moment]
    cum_indices, cum_t, cum_angles, cum_mx, cum_my, cum_speed, cum_speeddummy = find_monkey_information_in_the_duration(duration, monkey_information)         
    monkey_x, monkey_y, monkey_angle = cum_mx[-1], cum_my[-1], cum_angles[-1]
    monkey_xy = np.array([monkey_x, monkey_y])
    return monkey_xy, monkey_angle




def plot_arc_from_null_condition(axes, current_moment, ff_dataframe, monkey_xy, monkey_angle, rotation_matrix=None, assumed_memory_duration=2, arc_color='black',
                                 reaching_boundary_ok=False):
    R = rotation_matrix
    duration = [current_moment-assumed_memory_duration, current_moment]
    monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    ff_dataframe_temp = ff_dataframe[ff_dataframe['time'].between(duration[0], duration[1])]
    ff_dataframe_temp = ff_dataframe_temp[ff_dataframe_temp['visible'] == 1]
    if len(ff_dataframe_temp) < 1:
        print("No firefly was seen in the last 2 seconds")
        return axes, None, None, None, None, None, None

    
    ff_xy = ff_dataframe_temp[['ff_x', 'ff_y']].drop_duplicates().values
    ff_center_xy = ff_xy
    if reaching_boundary_ok:
        ff_xy = find_point_on_ff_boundary_with_smallest_angle_to_monkey(ff_xy[:,0], ff_xy[:,1], monkey_xy[0], monkey_xy[1], monkey_angle)

    ff_x_relative, ff_y_relative = find_relative_xy_positions(ff_xy[:,0], ff_xy[:,1], monkey_x, monkey_y, monkey_angle)
    ff_to_be_considered = np.where(ff_y_relative > 0)[0]
    if len(ff_to_be_considered) == 0:
        print("No firefly was in front of the monkey and visible within the last 2s.")
        return axes, None, None, None, None, None, None

    min_arc_length, min_arc_radius, min_arc_ff_xy, min_arc_ff_distance, min_arc_ff_angle, \
            min_arc_ff_angle_boundary = find_shortest_arc(ff_xy[ff_to_be_considered,0], ff_xy[ff_to_be_considered,1], monkey_x, monkey_y, monkey_angle)
    min_arc_relative_index = np.where((ff_xy[:,0] == min_arc_ff_xy[0]) & (ff_xy[:,1] == min_arc_ff_xy[1]))[0]
    min_arc_ff_center_xy = ff_center_xy[min_arc_relative_index].reshape(-1)
    if min_arc_radius > 0:
        center_x, center_y, arc_starting_angle, arc_ending_angle = find_elements_to_plot_an_arc(monkey_xy, monkey_angle, min_arc_radius, min_arc_ff_xy, min_arc_ff_angle)
        arc_xy_rotated = find_arc_xy_rotated(center_x, center_y, min_arc_radius, arc_starting_angle, arc_ending_angle, rotation_matrix=rotation_matrix)  
        center_xy_rotated = np.matmul(R, np.stack([center_x, center_y]))
        #axes.scatter(center_xy_rotated[0], center_xy_rotated[1], s=40, zorder=4, color='blue') 
    else:
        center_x, center_y = None, None
        # plot a line from the monkey to the ff
        arc_xy_rotated = np.stack((np.array([monkey_x, min_arc_ff_xy[0]]), np.array([monkey_y, min_arc_ff_xy[1]])))
        if rotation_matrix is not None:
            arc_xy_rotated = np.matmul(R, arc_xy_rotated)
    
    min_arc_ff_xy_rotated = np.matmul(R, min_arc_ff_xy.reshape(2,-1))  
    ff_xy_to_be_considered = np.matmul(R, ff_center_xy[ff_to_be_considered].T) 

    axes.plot(arc_xy_rotated[0], arc_xy_rotated[1], linewidth=2, color=arc_color, zorder=5)
    axes.scatter(min_arc_ff_xy_rotated[0], min_arc_ff_xy_rotated[1], s=80, zorder=4, color='darkgreen')   
    axes.scatter(ff_xy_to_be_considered[0], ff_xy_to_be_considered[1], s=80, zorder=3, color='gold')   


    return axes, min_arc_ff_xy, min_arc_ff_center_xy, min_arc_ff_angle, min_arc_length, center_x, center_y


def find_point_on_ff_boundary_with_smallest_angle_to_monkey(ff_x, ff_y, monkey_x, monkey_y, monkey_angle):
    angles_to_ff = basic_func.calculate_angles_to_ff_centers(ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
    diff_x = ff_x - monkey_x
    diff_y = ff_y - monkey_y
    diff_xy = np.stack((diff_x, diff_y), axis=1)
    distances_to_ff = LA.norm(diff_xy, axis=1)
    angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff, distances_to_ff, ff_radius=25)
    dif_in_angles = angles_to_ff - angles_to_boundaries
    new_ff_distances = np.abs(np.cos(dif_in_angles)*distances_to_ff)
    new_ff_angles_in_world = angles_to_boundaries + monkey_angle
    ff_x = np.cos(new_ff_angles_in_world)*new_ff_distances + monkey_x
    ff_y = np.sin(new_ff_angles_in_world)*new_ff_distances + monkey_y
    ff_xy = np.stack((ff_x, ff_y), axis=1)
    return ff_xy

