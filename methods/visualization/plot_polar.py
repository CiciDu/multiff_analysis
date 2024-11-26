import sys
from data_wrangling import basic_func
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe
from visualization import plot_behaviors_utils

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.environ['KMP_DUPLICATE_LIB_OK']='True'





def PlotPolar(duration,
              monkey_information,
              ff_dataframe, 
              ff_life_sorted,
              ff_real_position_sorted,
              ff_caught_T_new,
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
              figsize = None,
               ):



    """
    Plot the positions of the fireflies from the monkey's perspective (the monkey is always at the origin of the polar plot)


    Parameters
    ----------
    duration: list
        the duration to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    ff_caught_T_new: np.array
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

    currentTrial, num_trials, duration = basic_func.find_currentTrial_or_num_trials_or_duration(ff_caught_T_new, currentTrial, num_trials, duration)
    target_indices = np.arange(currentTrial-num_trials+1, currentTrial+1)
    sns.set_style(style="white")
    if duration[1]-duration[0] == 0:
        return

    cum_iloc_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if not hitting_arena_edge_ok: # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_crossing_boundary = np.array(monkey_information['crossing_boundary'].iloc[cum_iloc_indices])
        if (np.any(cum_crossing_boundary == 1)):
            print("Current plot is omitted because the monkey has crossed the boundary at some point.")
            return
        
    if fig is None:
        if figsize is None:
            figsize = (7,7)
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax = plot_behaviors_utils.set_polar_background_for_plotting(ax, rmax, color_visible_area_in_background=color_visible_area_in_background)

    ff_dataframe = ff_dataframe[(ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]
    if show_all_positions_of_all_fireflies:
        # Make a new ff_dataframe using the new function make_ff_dataframe.make_ff_dataframe_v2_func
        ff_dataframe_v2 = make_ff_dataframe.make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_new, ff_flash_sorted,  
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
        num_color_elements = len(cum_iloc_indices)+1
    else:
        num_color_elements = 101

    colors_ffs = plt.get_cmap(ff_colormap)(np.linspace(0, 1, num_color_elements))
    colors_target = plt.get_cmap(target_colormap)(np.linspace(0, 1, num_color_elements))


    if colors_show_overall_time:
        # color is based on time into the past
        ff_color = colors_ffs[cum_iloc_indices[-1] - np.array(ff_info['point_index'].astype('int'))]
        target_color = colors_target[cum_iloc_indices[-1] - np.array(target_info['point_index'].astype('int'))]
    else:
        # color is based on memory
        ff_color = colors_ffs[np.array(ff_info['memory'].astype('int'))]
        target_color = colors_target[np.array(target_info['memory'].astype('int'))]



    if show_alive_ff & (not show_all_positions_of_all_fireflies):
        ff_dataframe_v2 = make_ff_dataframe.make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_new, ff_flash_sorted,  
                                ff_real_position_sorted, ff_life_sorted, max_distance = 400, 
                                data_folder_name = None, print_progress = False) 
        ax.scatter(ff_dataframe_v2['ff_angle'], ff_dataframe_v2['ff_distance'], c='grey', s=5, alpha=0.2, zorder=1, marker='o')   

    # Visualize ff_info
    ax.scatter(ff_info['ff_angle'], ff_info['ff_distance'], c=ff_color, alpha=0.7, zorder=2, s=ff_info['visible']*size_increase_for_visible_ff+5, marker='o') # originally it was s=15

    if connect_dots:
        for ff_index in ff_dataframe['ff_index'].unique():
            current_ff = ff_dataframe[ff_dataframe['ff_index']==ff_index]
            ax.plot(current_ff['ff_angle'], current_ff['ff_distance'], alpha=0.7, zorder=1)


    plotted_points = [] # store the indices of points that have been plotted so that if show_target_throughout_duration, one know which points to exclude
    if len(target_info) > 0:
        ax.scatter(target_info['ff_angle'], target_info['ff_distance'], c=target_color, alpha=0.7, s=target_info['visible']*20+5)     
        plotted_points = target_info['point_index']


    if show_target_throughout_duration:
        if not show_all_positions_of_all_fireflies:
            ff_distance_and_angles = plot_behaviors_utils.find_ff_distance_and_angles(currentTrial, duration, ff_real_position_sorted, monkey_information)
            ff_distance_and_angles = ff_distance_and_angles[~ff_distance_and_angles['point_index'].isin(plotted_points)]
            ax.scatter(ff_distance_and_angles['ff_angle'], ff_distance_and_angles['ff_distance'], c='grey', s=15, alpha=0.2, zorder=1, marker='o')   

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
            ax = plot_behaviors_utils.add_legend_for_polar_plot(ax, show_visible_ff, show_ff_in_memory, show_alive_ff, show_visible_target, show_target_in_memory, show_target_throughout_duration, colors_show_overall_time)

    if show_colorbar:
        fig = plot_behaviors_utils.add_colorbar_for_polar_plot(fig, duration, show_ff_in_memory, show_target_in_memory, ff_colormap, target_colormap, colors_show_overall_time, show_all_positions_of_all_fireflies)         

    if return_axes:
        return fig, ax
    else:
        plt.show()
        
