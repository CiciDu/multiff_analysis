import sys
from data_wrangling import specific_utils
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, find_best_arc
from eye_position_analysis import eye_positions

import os
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def PlotTrials(duration,
               monkey_information,
               ff_dataframe,
               ff_life_sorted,
               ff_real_position_sorted,
               ff_believed_position_sorted,
               cluster_around_target_indices,
               ff_caught_T_new,
               currentTrial=None,  # Can be None; then it means all trials in the duration will be plotted
               num_trials=None,
               fig=None,
               axes=None,
               rotation_matrix=None,
               x0=None,
               y0=None,
               player="monkey",
               # None or 'speed' or 'abs_ddw' or 'time'; if not None, then the color of the path will vary by this variable
               trail_color_var=None,
               visible_distance=400,
               minimal_margin=100,
               show_start=True,
               show_stops=False,
               show_trajectory=True,
               show_monkey_angles=False,
               show_only_ff_that_monkey_has_passed_by_closely=False,
               show_alive_fireflies=True,
               show_visible_fireflies=False,  # only meaningful when show_alive_fireflies is False
               # only meaningful when show_alive_fireflies is False
               show_in_memory_fireflies=False,
               show_ff_indices=False,
               show_believed_target_positions=False,
               show_reward_boundary=False,
               show_path_when_target_visible=False,
               show_path_when_prev_target_visible=False,
               index_of_ff_to_show_path_when_ff_visible=None,
               indices_of_ff_to_be_plotted_in_a_basic_way=None,
               show_connect_path_ff=False,
               show_connect_path_ff_except_targets=False,
               show_connect_path_ff_specific_indices=None,
               connect_path_ff_color="#a940f5",  # a kind of purple
               vary_color_for_connecting_path_ff=False,
               # when this or the next parameter is True, vary_color_for_connecting_path_ff will automatically be set as True
               show_points_when_ff_start_being_visible=False,
               show_points_when_ff_stop_being_visible=False,
               show_connect_path_ff_memory=False,
               connect_path_ff_max_distance=None,
               show_visible_segments_ff_indices=None,
               how_to_show_ff_for_visible_segments='square',
               show_connect_path_ff_after_coloring_segments_ff_indices=None,
               show_path_when_cluster_visible=False,
               show_eye_positions=False,
               # this is only used if show_eye_positions = True
               show_eye_positions_for_both_eyes=False,
               show_eye_positions_on_the_right=False,
               show_eye_world_speed_vs_monkey_speed=False,
               show_connect_path_eye_positions=False,
               show_null_agent_trajectory=False,
               show_null_agent_trajectory_type='most_aligned',
               show_null_agent_trajectory_2nd_time=False,
               show_null_trajectory_reaching_boundary_ok=True,
               null_arc_info_for_plotting=None,
               show_ff_to_be_considered_by_first_null_trajectory=True,
               null_agent_starting_time=None,
               assumed_memory_duration_of_agent=2,
               show_scale_bar=False,
               show_colorbar=False,
               show_title=True,
               show_legend=False,
               trial_to_show_cluster=None,  # None, "current", or "previous"
               cluster_dataframe_point=None,
               trial_to_show_cluster_around_target=None,  # None, "current", or "previous"
               indices_of_ff_to_mark=None,  # None or a list
               indices_of_ff_to_mark_2nd_kind=None,
               steps_to_be_marked=None,
               point_indices_to_be_marked=None,
               point_indices_to_be_marked_2nd_kind=None,
               point_indices_to_be_marked_3rd_kind=None,
               adjust_xy_limits=True,
               zoom_in=False,  # can only be effective if adjust_xy_limits is True
               images_dir=None,
               hitting_arena_edge_ok=True,
               truncate_part_before_crossing_arena_edge=False,
               trial_too_short_ok=True,
               subplots=False,
               combined_plot=False,
               as_part_of_animation=False,
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
    point_indices_to_be_marked: None or a list
        similar as steps_to_be_marked, but here the absolute indices are passed, rather than relative indices
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
    # Set a more modern and attractive style
    sns.set_style(style="whitegrid", rc={
                  'axes.facecolor': '#f8f9fa', 'grid.color': '#e9ecef'})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial',
                                       'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    legend_markers = []
    legend_names = []

    if combined_plot is True:
        player = "combined"

    # If currentTrial is not given, then it will be calculated based on the duration
    currentTrial, num_trials, duration = specific_utils.find_currentTrial_or_num_trials_or_duration(
        ff_caught_T_new, currentTrial, num_trials, duration)
    print('currentTrial:', currentTrial, 'num_trials:', num_trials)

    if duration[1] <= duration[0]:
        raise ValueError("duration[1] must be greater than duration[0]")

    cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
        duration, monkey_information)
    monkey_subset_df = monkey_information.loc[cum_pos_index]

    cum_r = LA.norm(np.stack((cum_mx, cum_my)), axis=0)
    if (np.any(cum_r > 949)):
        hitting_arena_edge = True
        if not hitting_arena_edge_ok:
            # Stop plotting for the trial if the monkey/agent has gone across the edge
            # the three outputs are whether_plotted, axes, R, cum_mxy_rotated, shown_ff_indices
            return {'whether_plotted': False}
    else:
        hitting_arena_edge = False

    if truncate_part_before_crossing_arena_edge:
        cum_crossing_boundary = np.array(
            monkey_information['crossing_boundary'].iloc[cum_pos_index])
        # find the last time point of crossing boundary
        cross_boundary_points = np.where(cum_crossing_boundary == 1)[0]
        if len(cross_boundary_points) > 0:
            cross_boundary_time = cum_t[cross_boundary_points[-1]]
            if cross_boundary_time - duration[0] < (duration[1]-duration[0])*2/3:
                duration[0] = cross_boundary_time
                print("duration[0] updated to {} to truncate the part of crossing arena edge".format(
                    duration[0]))
                # need to find the new currentTrial and num_trials and related information
                currentTrial = None
                num_trials = None
                currentTrial, num_trials, duration = specific_utils.find_currentTrial_or_num_trials_or_duration(
                    ff_caught_T_new, currentTrial, num_trials, duration)
                cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
                    duration, monkey_information)
            else:
                print("The monkey crossed the arena edge at {} but the duration is long enough (more than 2/3 of the original duration) to include the crossing".format(cross_boundary_time))
    # # Make sure the duration is sufficient for assumed_memory_duration_of_agent
    # if show_null_agent_trajectory & show_ff_to_be_considered_by_first_null_trajectory:
    #     if null_agent_starting_time is None:
    #         null_agent_starting_time = duration[1]
    #     if null_agent_starting_time - duration[0] < assumed_memory_duration_of_agent:
    #         duration[0] = null_agent_starting_time - assumed_memory_duration_of_agent
    #         print("duration[0] is changed to %s to ensure that the duration is sufficient for assumed_memory_duration_of_agent" % duration[0])
    #         # recalculate currentTrial and num_trials accordingly
    #         currentTrial, num_trials, duration = specific_utils.find_currentTrial_or_num_trials_or_duration(ff_caught_T_new, currentTrial, num_trials, duration)

    target_indices = np.arange(currentTrial-num_trials+1, currentTrial+1)

    ff_dataframe_in_duration = ff_dataframe[(
        ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]
    # the below is no longer useful once we use ff_dataframe_sifted
    # if show_only_ff_that_monkey_has_passed_by_closely:
    #     ff_dataframe_in_duration = ff_dataframe_utils.keep_only_ff_that_monkey_has_passed_by_closely(ff_dataframe_in_duration, max_distance_to_ff=100)
    print('duration:', duration)
    ff_dataframe_in_duration_in_memory = ff_dataframe_in_duration.loc[(ff_dataframe_in_duration['visible'] == 0) &
                                                                      (ff_dataframe_in_duration['ff_distance'] <= 400)]  # this condition can make the plot cleaner, and it's also believeable that when the ff is more than 400 cm away, monkey wouldn't care to remember it
    ff_dataframe_in_duration_visible = ff_dataframe_in_duration.loc[
        ff_dataframe_in_duration['visible'] == 1]

    if len(cum_t) == 0:
        # the three outputs are whether_plotted, axes, R, cum_mxy_rotated, shown_ff_indices
        return {'whether_plotted': False}

    if not trial_too_short_ok:
        # Stop plotting for the trial if the trial is too short
        if (len(cum_t) < 5):
            # the three outputs are whether_plotted, axes, R, cum_mxy_rotated, shown_ff_indices
            return {'whether_plotted': False}

    if fig is None:
        if show_eye_positions_on_the_right:
            fig = plt.figure(figsize=(12, 6))
            axes = fig.add_subplot(1, 2, 1)
        else:
            fig, axes = plt.subplots(figsize=(8, 8))
    elif axes is None:
        if show_eye_positions_on_the_right:
            axes = fig.add_subplot(121)
        elif as_part_of_animation:
            axes = fig.add_subplot(121)
        else:
            axes = fig.add_subplot(111)

    if rotation_matrix is None:
        R = plot_behaviors_utils.find_rotation_matrix(cum_mx, cum_my)
        rotation_matrix = R
    else:
        R = rotation_matrix

    # Find the trajectory of the monkey
    cum_mxy_rotated = np.matmul(R, np.stack((cum_mx, cum_my)))

    # Determine whether translation is needed for the trajectory
    if (x0 is None) or (y0 is None):
        if combined_plot or show_eye_positions:
            x0, y0 = cum_mxy_rotated[0][0], cum_mxy_rotated[1][0]
        else:
            x0, y0 = 0, 0

    if show_monkey_angles:
        left_end_xy_rotated, right_end_xy_rotated = plot_behaviors_utils.find_triangles_to_show_monkey_angles(
            cum_mx, cum_my, cum_angle, rotation_matrix=R)
        axes = plot_behaviors_utils.visualize_monkey_angles_using_triangles(
            axes, cum_mxy_rotated, left_end_xy_rotated, right_end_xy_rotated, linewidth=0.5)

    if show_start:
        # Plot the start with improved styling
        start_size = {"agent": 280, "monkey": 200, "combined": 120}
        marker = axes.scatter(cum_mxy_rotated[0, 0]-x0, cum_mxy_rotated[1, 0]-y0,
                              marker='s', s=start_size[player], color="#28a745", zorder=3, alpha=0.9,
                              edgecolors='#1e7e34', linewidth=2)
        legend_markers.append(marker)
        legend_names.append('Start new section or begin using null agent')

    if show_stops:
        stop_size = {"agent": 200, "monkey": 250, "combined": 80}
        zerospeed_rotated = plot_behaviors_utils.find_stops_for_plotting(
            cum_mx, cum_my, cum_speeddummy, rotation_matrix=R)
        marker = axes.scatter(zerospeed_rotated[0]-x0, zerospeed_rotated[1]-y0,
                              marker='*', s=stop_size[player], alpha=0.8, color="#dc3545", zorder=2,
                              edgecolors='#c82333', linewidth=1)
        legend_markers.append(marker)
        legend_names.append('Low speed/stopping points')

    if indices_of_ff_to_be_plotted_in_a_basic_way is not None:
        ff_to_be_plotted_in_a_basic_way = indices_of_ff_to_be_plotted_in_a_basic_way.tolist()
    else:
        ff_to_be_plotted_in_a_basic_way = []

    alive_ff_indices, alive_ff_position_rotated = plot_behaviors_utils.find_alive_ff(
        duration, ff_life_sorted, ff_real_position_sorted, rotation_matrix=R)
    if show_alive_fireflies:
        marker = axes.scatter(
            alive_ff_position_rotated[0]-x0, alive_ff_position_rotated[1]-y0, marker='o', s=15,
            color="#e83e8c", zorder=2, alpha=0.8, edgecolors='#d63384', linewidth=1)
        marker_name = ('Centers of alive fireflies')
        legend_markers.append(marker)
        legend_names.append(marker_name)
    elif show_visible_fireflies or show_in_memory_fireflies or (len(ff_to_be_plotted_in_a_basic_way) > 0):
        if show_visible_fireflies:
            ff_to_be_plotted_in_a_basic_way.extend(
                ff_dataframe_in_duration_visible.ff_index.unique())
            marker_name = ('Centers of visible fireflies')
        if show_in_memory_fireflies:
            ff_to_be_plotted_in_a_basic_way.extend(
                ff_dataframe_in_duration_in_memory.ff_index.unique())
            marker_name = ('Centers of visible or in-memory fireflies')

        # if show_believed_target_positions:
        #     ff_to_be_plotted_in_a_basic_way.extend(range(currentTrial - num_trials + 1, currentTrial + 1))
        ff_to_be_plotted_in_a_basic_way = np.unique(
            np.array(ff_to_be_plotted_in_a_basic_way)).tolist()
        ff_positions_rotated = np.matmul(
            R, ff_real_position_sorted[ff_to_be_plotted_in_a_basic_way].T)
        marker = axes.scatter(
            ff_positions_rotated[0]-x0, ff_positions_rotated[1]-y0, marker='o', s=15,
            color="#e83e8c", zorder=3, alpha=0.8, edgecolors='#d63384', linewidth=1)
        if show_visible_fireflies or show_in_memory_fireflies:
            legend_markers.append(marker)
            legend_names.append(marker_name)

    # a list of indices of fireflies that will be shown in the plot except alive_ff
    shown_ff_indices = ff_to_be_plotted_in_a_basic_way.copy()

    if show_believed_target_positions:
        target_size = {"agent": 220, "monkey": 180, "combined": 50}
        marker = {"agent": "*", "monkey": "*", "combined": "o"}
        shown_ff_indices.extend(
            range(currentTrial - num_trials + 1, currentTrial + 1))
        believed_target_positions_rotated = plot_behaviors_utils.find_believed_target_positions(
            ff_believed_position_sorted, currentTrial, num_trials, rotation_matrix=R)
        marker = axes.scatter(believed_target_positions_rotated[0]-x0, believed_target_positions_rotated[1] -
                              y0, marker=marker[player], s=target_size[player], color="#fd7e14", alpha=0.9, zorder=4,
                              edgecolors='#e55a00', linewidth=2)
        legend_markers.append(marker)
        legend_names.append('Catching-firefly positions')

    if indices_of_ff_to_mark is not None:
        shown_ff_indices.extend(indices_of_ff_to_mark)
        if isinstance(indices_of_ff_to_mark, float):
            indices_of_ff_to_mark = np.array([indices_of_ff_to_mark])
        for ff in indices_of_ff_to_mark:
            ff_position = ff_real_position_sorted[ff]
            ff_position_rotated = np.matmul(
                R, np.stack((ff_position[0], ff_position[1])))
            axes.scatter(ff_position_rotated[0]-x0, ff_position_rotated[1]-y0, marker='*',
                         s=target_size[player]+170, color="green", alpha=0.75, zorder=4)

    if indices_of_ff_to_mark_2nd_kind is not None:
        shown_ff_indices.extend(indices_of_ff_to_mark_2nd_kind)
        if isinstance(indices_of_ff_to_mark_2nd_kind, float):
            indices_of_ff_to_mark = np.array([indices_of_ff_to_mark_2nd_kind])
        for ff in indices_of_ff_to_mark_2nd_kind:
            ff_position = ff_real_position_sorted[ff]
            ff_position_rotated = np.matmul(
                R, np.stack((ff_position[0], ff_position[1])))
            axes.scatter(ff_position_rotated[0]-x0, ff_position_rotated[1]-y0,
                         marker='s', s=target_size[player], color="purple", alpha=0.75, zorder=5)

    steps_to_be_marked_size = {"agent": 120, "monkey": 50}
    if steps_to_be_marked is not None:
        axes.scatter(cum_mxy_rotated[0, steps_to_be_marked]-x0, cum_mxy_rotated[1, steps_to_be_marked] -
                     y0, marker='s', s=steps_to_be_marked_size[player], color="gold", zorder=3, alpha=0.3)

    if point_indices_to_be_marked is not None:
        temp_cum_mx, temp_cum_my = np.array(monkey_information['monkey_x'].loc[point_indices_to_be_marked]), np.array(
            monkey_information['monkey_y'].loc[point_indices_to_be_marked])
        temp_cum_mxy_rotated = np.matmul(
            R, np.stack((temp_cum_mx, temp_cum_my)))
        axes.scatter(temp_cum_mxy_rotated[0]-x0, temp_cum_mxy_rotated[1]-y0, marker='s',
                     s=steps_to_be_marked_size[player], color="gold", zorder=3, alpha=0.7)

    if point_indices_to_be_marked_2nd_kind is not None:
        temp_cum_mx, temp_cum_my = np.array(monkey_information['monkey_x'].loc[point_indices_to_be_marked_2nd_kind]), np.array(
            monkey_information['monkey_y'].loc[point_indices_to_be_marked_2nd_kind])
        temp_cum_mxy_rotated = np.matmul(
            R, np.stack((temp_cum_mx, temp_cum_my)))
        axes.scatter(temp_cum_mxy_rotated[0]-x0, temp_cum_mxy_rotated[1] -
                     y0, marker='*', s=100, color="blue", zorder=3, alpha=0.6)

    if point_indices_to_be_marked_3rd_kind is not None:
        temp_cum_mx, temp_cum_my = np.array(monkey_information['monkey_x'].loc[point_indices_to_be_marked_3rd_kind]), np.array(
            monkey_information['monkey_y'].loc[point_indices_to_be_marked_3rd_kind])
        temp_cum_mxy_rotated = np.matmul(
            R, np.stack((temp_cum_mx, temp_cum_my)))
        axes.scatter(temp_cum_mxy_rotated[0]-x0, temp_cum_mxy_rotated[1] -
                     y0, marker='X', s=200, color="pink", zorder=4, alpha=0.8)

    if show_path_when_target_visible:
        path_size = {"agent": 50, "monkey": 30, "combined": 2}
        ff_visible_path_rotated = plot_behaviors_utils.find_path_when_ff_visible(
            currentTrial, ff_dataframe_in_duration, cum_point_index, visible_distance, rotation_matrix=R)
        marker = axes.scatter(
            ff_visible_path_rotated[0]-x0, ff_visible_path_rotated[1]-y0, s=path_size[player], c="green", alpha=0.6, zorder=5)
        legend_markers.append(marker)
        legend_names.append('Path when target is visible')

    if show_path_when_prev_target_visible:  # for previous target
        path_size = {"agent": 65, "monkey": 40, "combined": 2}
        ff_visible_path_rotated = plot_behaviors_utils.find_path_when_ff_visible(
            currentTrial-1, ff_dataframe_in_duration, cum_point_index, visible_distance, rotation_matrix=R)
        marker = axes.scatter(
            ff_visible_path_rotated[0]-x0, ff_visible_path_rotated[1]-y0, s=path_size[player], c="aqua", alpha=0.8, zorder=3)
        legend_markers.append(marker)
        legend_names.append('Path when previous target is visible')

    if index_of_ff_to_show_path_when_ff_visible is not None:
        path_size = {"agent": 40, "monkey": 20, "combined": 2}
        ff_visible_path_rotated = plot_behaviors_utils.find_path_when_ff_visible(
            index_of_ff_to_show_path_when_ff_visible, ff_dataframe_in_duration, cum_point_index, visible_distance, rotation_matrix=R)
        marker = axes.scatter(
            ff_visible_path_rotated[0]-x0, ff_visible_path_rotated[1]-y0, s=path_size[player], c="aqua", alpha=0.7, zorder=3)
        legend_markers.append(marker)
        legend_names.append('Path when ff marked by aqua square is visible')

        # also mark that ff
        ff_positions_rotated = np.matmul(
            R, ff_real_position_sorted[index_of_ff_to_show_path_when_ff_visible].T)
        marker = axes.scatter(
            ff_positions_rotated[0]-x0, ff_positions_rotated[1]-y0, marker='s', s=80, color="aqua", alpha=0.5, zorder=6)
        shown_ff_indices.append(index_of_ff_to_show_path_when_ff_visible)

    if connect_path_ff_max_distance is None:
        connect_path_ff_max_distance = visible_distance

    temp_ff_positions = None
    connection_linewidth = plot_behaviors_utils.connection_linewidth
    connection_alpha = plot_behaviors_utils.connection_alpha

    if show_connect_path_ff_memory:
        ff_dataframe_in_duration_in_memory_qualified = ff_dataframe_in_duration_in_memory.loc[
            ff_dataframe_in_duration_in_memory['ff_distance'] <= connect_path_ff_max_distance]
        shown_ff_indices.extend(
            ff_dataframe_in_duration_in_memory_qualified.ff_index.unique())

        if show_connect_path_ff_specific_indices is not None:
            ff_dataframe_in_duration_in_memory_qualified = ff_dataframe_in_duration_in_memory_qualified.loc[
                ff_dataframe_in_duration_in_memory_qualified['ff_index'].isin(show_connect_path_ff_specific_indices)]

        temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.find_lines_to_connect_path_ff(
            ff_dataframe_in_duration_in_memory_qualified, target_indices, rotation_matrix=R)
        axes = plot_behaviors_utils.connect_points_to_points(axes, temp_ff_positions, temp_monkey_positions, x0, y0, color="purple", alpha=0.3,
                                                             linewidth=connection_linewidth[player], show_dots=True, dot_color="brown")

    if show_visible_segments_ff_indices is not None:
        ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible.loc[
            ff_dataframe_in_duration_visible['ff_index'].isin(show_visible_segments_ff_indices)]
        axes, legend_markers, legend_names, show_visible_segments_of_ff_dict = plot_behaviors_utils.plot_horizontal_lines_to_show_ff_visible_segments(axes, ff_dataframe_in_duration_visible_qualified, monkey_information, rotation_matrix, x0, y0, legend_markers, legend_names,
                                                                                                                                                      how_to_show_ff=how_to_show_ff_for_visible_segments, unique_ff_indices=show_visible_segments_ff_indices)

        if show_connect_path_ff_after_coloring_segments_ff_indices is not None:
            for _ff_index in np.array(show_connect_path_ff_after_coloring_segments_ff_indices):
                ff_dataframe_visible_for_specific_ff = ff_dataframe_in_duration_visible.loc[
                    ff_dataframe_in_duration_visible['ff_index'] == _ff_index]
                axes, _, _, _, _, = plot_behaviors_utils.plot_lines_to_connect_path_and_ff(
                    axes, ff_dataframe_visible_for_specific_ff, rotation_matrix, x0, y0, 1, 0.5, vary_color_for_connecting_path_ff=False, line_color=show_visible_segments_of_ff_dict[_ff_index])

    if show_points_when_ff_start_being_visible or show_points_when_ff_stop_being_visible:
        vary_color_for_connecting_path_ff = True

    if show_connect_path_ff or show_connect_path_ff_except_targets or (show_connect_path_ff_specific_indices is not None):
        ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible.loc[
            ff_dataframe_in_duration_visible['ff_distance'] <= connect_path_ff_max_distance]
        if show_connect_path_ff_specific_indices is not None:
            ff_dataframe_in_duration_visible_qualified = ff_dataframe_in_duration_visible_qualified.loc[
                ff_dataframe_in_duration_visible_qualified['ff_index'].isin(show_connect_path_ff_specific_indices)]

        shown_ff_indices.extend(
            ff_dataframe_in_duration_visible_qualified.ff_index.unique())
        axes, legend_markers, legend_names, temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.plot_lines_to_connect_path_and_ff(axes=axes,
                                                                                                                                              ff_info=ff_dataframe_in_duration_visible_qualified, rotation_matrix=R, x0=x0, y0=y0, linewidth=connection_linewidth[
                                                                                                                                                  player], alpha=connection_alpha[player],
                                                                                                                                              vary_color_for_connecting_path_ff=vary_color_for_connecting_path_ff, line_color=connect_path_ff_color, target_indices=target_indices,
                                                                                                                                              show_connect_path_ff_except_targets=show_connect_path_ff_except_targets, show_points_when_ff_stop_being_visible=show_points_when_ff_stop_being_visible,
                                                                                                                                              show_points_when_ff_start_being_visible=show_points_when_ff_start_being_visible, legend_markers=legend_markers, legend_names=legend_names)

        show_dots_marker = Line2D(
            [0], [0], marker='o', markersize=5, color="brown", linestyle='')
        marker_name = "FFs visible"
        if show_connect_path_ff_memory:
            # As of now, ff_in_memory will not be shown unless ff_visible is shown too
            marker_name = marker_name + " or in memory"
        if show_connect_path_ff_except_targets:
            marker_name = marker_name + \
                " except targets\n(the longer the darker)"
        else:
            marker_name = marker_name + " (the longer the darker)"
        legend_markers.append(show_dots_marker)
        legend_names.append(marker_name)

    if trial_to_show_cluster is not None:
        trial_conversion = {"current": 0, "previous": -1}
        cluster_ff_rotated = plot_behaviors_utils.find_ff_in_cluster(
            cluster_dataframe_point, ff_real_position_sorted, currentTrial=currentTrial+trial_conversion[trial_to_show_cluster], rotation_matrix=R)
        axes.scatter(
            cluster_ff_rotated[0]-x0, cluster_ff_rotated[1]-y0, marker='o', c="blue", s=25, zorder=4)

    if trial_to_show_cluster_around_target is not None:
        trial_conversion = {"current": 0, "previous": -1}
        cluster_ff_indices, cluster_around_target_rotated = plot_behaviors_utils.find_ff_in_cluster_around_target(cluster_around_target_indices, ff_real_position_sorted, rotation_matrix=R,
                                                                                                                  currentTrial=currentTrial+trial_conversion[trial_to_show_cluster_around_target])
        shown_ff_indices.extend(cluster_ff_indices)
        axes.scatter(
            cluster_around_target_rotated[0]-x0, cluster_around_target_rotated[1]-y0, marker='o', s=40,
            color="#0d6efd", zorder=4, alpha=0.8, edgecolors='#0b5ed7', linewidth=1)
        if show_path_when_cluster_visible:  # Find where on the path the monkey/agent can see any member of the cluster around the target
            list_of_colors = ["#0d6efd", "#6610f2",
                              "#6f42c1", "#d63384", "#dc3545", "#fd7e14"]
            path_size, path_alpha = {"agent": [100, 15], "monkey": [
                20, 5]}, {"agent": 0.9, "monkey": 0.6}
            ff_size, ff_alpha = {"agent": 160, "monkey": 120}, {
                "agent": 0.9, "monkey": 0.7}
            for i, index in enumerate(cluster_ff_indices):
                monkey_xy_rotated, ff_position_rotated = plot_behaviors_utils.find_path_when_ff_in_cluster_visible(
                    ff_dataframe_in_duration, index, rotation_matrix=R)
                axes.scatter(monkey_xy_rotated[0]-x0, monkey_xy_rotated[1]-y0, s=path_size[player][0] -
                             path_size[player][1] * i, color=list_of_colors[i % len(list_of_colors)],
                             alpha=path_alpha[player], zorder=3+i)
                # Use a circle with the corresponding color to show that ff
                axes.scatter(ff_position_rotated[0]-x0, ff_position_rotated[1]-y0, marker='o',
                             s=ff_size[player], alpha=ff_alpha[player],
                             color=list_of_colors[i %
                                                  len(list_of_colors)], zorder=3,
                             edgecolors='white', linewidth=1)

    shown_ff_indices = np.unique(np.array(shown_ff_indices)).astype(int)
    shown_ff_positions_rotated = ff_real_position_sorted[shown_ff_indices].T
    if (rotation_matrix is not None) & (shown_ff_positions_rotated.shape[1] > 0):
        shown_ff_positions_rotated = np.matmul(
            rotation_matrix, shown_ff_positions_rotated)
    if show_reward_boundary:
        if show_alive_fireflies:
            boundary_centers_rotated = alive_ff_position_rotated
        else:
            boundary_centers_rotated = shown_ff_positions_rotated
        for i in boundary_centers_rotated.T:
            circle = plt.Circle(
                (i[0]-x0, i[1]-y0), 25, facecolor='#6c757d', edgecolor='#fd7e14',
                alpha=0.6, zorder=1, linewidth=2)
            axes.add_patch(circle)

    if show_ff_indices:
        print('shown_ff_indices: ', shown_ff_indices)
        for num in range(len(shown_ff_indices)):
            ff_pos = shown_ff_positions_rotated[:, num]
            ff_index = shown_ff_indices[num]
            axes.annotate(str(ff_index), (ff_pos[0], ff_pos[1]), fontsize=12,
                          fontweight='bold', color='#495057',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='#dee2e6'))

    if show_eye_positions:
        if not show_eye_positions_for_both_eyes:
            axes = plot_behaviors_utils.plot_eye_positions(axes, monkey_subset_df, duration, cum_mxy_rotated, x0, y0,
                                                           rotation_matrix, player, marker='o',
                                                           show_connect_path_eye_positions=show_connect_path_eye_positions
                                                           )
        else:
            for eye_col_suffix, marker in [('_r', 'o'), ('_r', 's')]:
                axes = plot_behaviors_utils.plot_eye_positions(axes, monkey_subset_df, duration, cum_mxy_rotated, x0, y0,
                                                               rotation_matrix, player, marker=marker, sample_ratio=6, eye_col_suffix=eye_col_suffix,
                                                               show_connect_path_eye_positions=show_connect_path_eye_positions
                                                               )

    if show_eye_positions_on_the_right:
        monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
            monkey_subset_df, duration, rotation_matrix=R
        )
        axes2 = fig.add_subplot(1, 2, 2)
        scatter = axes2.scatter(monkey_subset['gaze_mky_view_x'].values,
                                monkey_subset['gaze_mky_view_y'].values,
                                s=10, c=monkey_subset['time'].values, cmap='plasma', alpha=0.7)
        mx_min, mx_max, my_min, my_max = plot_behaviors_utils.find_xy_min_max_for_plots(
            monkey_subset[['gaze_world_x_rotated', 'gaze_world_y_rotated']].values.T, x0, y0, temp_ff_positions=None)
        axes2 = plot_behaviors_utils.set_xy_limits_for_axes(
            axes2, mx_min, mx_max, my_min, my_max, minimal_margin, zoom_in)
        fig.tight_layout()
        # plot a horizontal and a vertical line at origin
        axes2.axhline(y=0, color='#6c757d', linestyle='--',
                      linewidth=1.5, alpha=0.7)
        axes2.axvline(x=0, color='#6c757d', linestyle='--',
                      linewidth=1.5, alpha=0.7)

        # Improve the second subplot styling
        axes2.set_title("Eye Gaze Positions", fontsize=16,
                        fontweight='bold', color='#212529', pad=15)
        axes2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        axes2.spines['left'].set_color('#6c757d')
        axes2.spines['bottom'].set_color('#6c757d')
        axes2.spines['left'].set_linewidth(1.5)
        axes2.spines['bottom'].set_linewidth(1.5)

    colorbar_max_value = None
    if show_trajectory:
        axes = plot_behaviors_utils.show_trajectory_func(axes, player, cum_pos_index, cum_mxy_rotated, cum_t, cum_speed, monkey_information,
                                                         x0, y0, trail_color_var, show_eye_positions, subplots, hitting_arena_edge)
        # Some other procedures
        if trail_color_var is None:
            # make a proxy to use legend
            line = Line2D([0], [0], linestyle="-", alpha=0.9,
                          linewidth=3, color="#212529")
            legend_markers.append(line)
            legend_names.append('Monkey trajectory')
        elif trail_color_var == 'abs_ddw':
            cum_abs_ddw = np.abs(
                np.array(monkey_information['monkey_ddw'].iloc[cum_pos_index]))
            colorbar_max_value = max(cum_abs_ddw)

    if show_null_agent_trajectory:
        axes, legend_markers, legend_names = show_null_trajectory.show_null_agent_trajectory_func(duration, null_agent_starting_time, monkey_information, ff_dataframe, ff_caught_T_new,
                                                                                                  axes, legend_markers, legend_names, R, assumed_memory_duration_of_agent, show_null_agent_trajectory_2nd_time, show_ff_to_be_considered_by_first_null_trajectory,
                                                                                                  reaching_boundary_ok=show_null_trajectory_reaching_boundary_ok, null_arc_info_for_plotting=null_arc_info_for_plotting, type=show_null_agent_trajectory_type)

    if show_scale_bar:
        axes = plot_behaviors_utils.plot_scale_bar(axes)

    if not show_eye_positions:
        axes.xaxis.set_major_locator(mtick.NullLocator())
        axes.yaxis.set_major_locator(mtick.NullLocator())

    if show_colorbar:
        fig, axes = plot_behaviors_utils.plot_colorbar_for_trials(fig, axes, trail_color_var, show_eye_positions=show_eye_positions,
                                                                  show_eye_positions_on_the_right=show_eye_positions_on_the_right, duration=duration, max_value=colorbar_max_value)

    if show_legend:
        axes.legend(legend_markers, legend_names, scatterpoints=1,
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                    frameon=True, fancybox=True, shadow=True,
                    fontsize=10, framealpha=0.95)

    # Set the limits of the x-axis and y-axis
    if adjust_xy_limits:
        if show_eye_positions_on_the_right:
            mx_min, mx_max, my_min, my_max = plot_behaviors_utils.find_xy_min_max_for_plots(
                monkey_subset[['gaze_world_x_rotated', 'gaze_world_y_rotated']].values.T, x0, y0, temp_ff_positions=None)
            axes = plot_behaviors_utils.set_xy_limits_for_axes(
                axes, mx_min, mx_max, my_min, my_max, minimal_margin=minimal_margin, zoom_in=zoom_in)
        else:
            mx_min, mx_max, my_min, my_max = plot_behaviors_utils.find_xy_min_max_for_plots(
                cum_mxy_rotated, x0, y0, temp_ff_positions=shown_ff_positions_rotated)
            axes = plot_behaviors_utils.set_xy_limits_for_axes(
                axes, mx_min, mx_max, my_min, my_max, minimal_margin=minimal_margin, zoom_in=zoom_in)

    if show_title:
        axes.set_title(f"Trial {currentTrial}", fontsize=24, fontweight='bold',
                       color='#212529', pad=20)

    if images_dir is not None:
        filename = "trial_" + str(currentTrial)
        plot_behaviors_utils.save_image(filename, images_dir)

    whether_plotted = True
    axes.set_aspect('equal')

    # Add subtle grid styling
    axes.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Improve axis styling
    axes.spines['left'].set_color('#6c757d')
    axes.spines['bottom'].set_color('#6c757d')
    axes.spines['left'].set_linewidth(1.5)
    axes.spines['bottom'].set_linewidth(1.5)

    if show_eye_world_speed_vs_monkey_speed:
        monkey_sub = monkey_information.iloc[cum_pos_index]
        eye_positions.plot_eye_world_speed_vs_monkey_speed(monkey_sub)

    returned_info = {'whether_plotted': whether_plotted,
                     'axes': axes,
                     'fig': fig,
                     'x0': x0,
                     'y0': y0,
                     'rotation_matrix': R,
                     'cum_mxy_rotated': cum_mxy_rotated,
                     'shown_ff_indices': shown_ff_indices}
    return returned_info
