
import sys
from visualization import plot_behaviors_utils, plot_trials
from null_behaviors import show_null_trajectory
from pattern_discovery import ff_dataframe_utils

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go




def connect_path_ff_based_on_available_null_arcs(duration_to_connect_path_ff, fig, R, curvature_df, show_percentile_of_curv_diff_by_color=True):

    axes = fig.axes[0]
    curvature_df_sub = curvature_df[curvature_df.time.between(duration_to_connect_path_ff[0], duration_to_connect_path_ff[1])]
    curvature_df_sub = curvature_df_sub[curvature_df_sub['ff_index'] >= 0]

    if not show_percentile_of_curv_diff_by_color:
        axes, legend_markers, legend_names, temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.plot_lines_to_connect_path_and_ff(axes=axes, \
            ff_info=curvature_df_sub, rotation_matrix=R, x0=0, y0=0, linewidth=0.5, alpha=0.7, vary_color_for_connecting_path_ff=True, 
            show_points_when_ff_stop_being_visible=True, show_points_when_ff_start_being_visible=True)

    else:
        # let the color be based on diff_percentile
        viridis = matplotlib.colormaps['viridis']
        line_colors = viridis(curvature_df_sub.diff_percentile_in_decimal.values)
        temp_ff_positions, temp_monkey_positions = plot_behaviors_utils.find_lines_to_connect_path_ff(curvature_df_sub, target_indices=None, rotation_matrix=R, target_excluded = False)
        x0, y0 = 0, 0
        if temp_ff_positions.shape[1] > 0:
            for j in range(temp_ff_positions.shape[1]):
                axes.plot(np.stack([temp_ff_positions[0, j]-x0, temp_monkey_positions[0, j]-x0]),
                        np.stack([temp_ff_positions[1, j]-y0, temp_monkey_positions[1, j]-y0]), 
                        '-', alpha=0.7, linewidth=0.5, c=line_colors[j], zorder=3)         
        
        # plot a colorbar for it
        cmap = cm.viridis 
        vmax = 100
        title = 'Percentile of Abs Diff of Angle Over Distance'
        norm2 = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=fig.add_axes([0.95, 0.4, 0.05, 0.43]), orientation='vertical')
        cbar2.outline.set_visible(False)
        cbar2.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
        cbar2.ax.set_title(title, ha='left', y=1.04)

    return fig


def plot_stops_near_ff_func(row, monkey_information, ff_real_position_sorted, ff_dataframe, null_arcs_plotting_kwargs, PlotTrials_args, 
                         additional_plotting_kwargs={}, eliminate_irrelevant_points_beyond_boundaries=True, 
                         ff_max_distance_to_path_to_show_visible_segments=None):

    plt.rcParams['figure.figsize'] = [7, 7]  
    time = row.stop_time
    duration_to_plot = [time-2.5, max(time+1.5, row.next_stop_time+0.5)]


    null_arcs_plotting_kwargs_temp = null_arcs_plotting_kwargs.copy()
    null_arcs_plotting_kwargs_temp['show_visible_fireflies'] = False
    null_arcs_plotting_kwargs_temp['show_in_memory_fireflies'] = False

    # null_arcs_plotting_kwargs_temp['show_connect_path_ff_specific_indices'] = [int(row.stop_ff_index), int(row.alt_ff_index)]
    # null_arcs_plotting_kwargs_temp['point_indices_to_be_marked_2nd_kind'] = int(row['earlest_point_index_when_alt_ff_and_stop_ff_have_both_been_seen_bbas'])
    # null_arcs_plotting_kwargs_temp['point_indices_to_be_marked_3rd_kind'] = [int(row.stop_point_index), int(row.next_stop_point_index)]
    
    #null_arcs_plotting_kwargs_temp['point_indices_to_be_marked_3rd_kind'] = [int(row.stop_point_index)]
    null_arcs_plotting_kwargs_temp['point_indices_to_be_marked_2nd_kind'] = [int(row.point_index_before_stop)]
    null_arcs_plotting_kwargs_temp['show_believed_target_positions'] = False

    if eliminate_irrelevant_points_beyond_boundaries:
        relevant_point_indices = [row.stop_point_index, row.next_stop_point_index]
        duration_to_plot = show_null_trajectory.eliminate_irrelevant_points_before_or_after_crossing_boundary(duration_to_plot, relevant_point_indices, monkey_information)
    
    if ff_max_distance_to_path_to_show_visible_segments is not None:
        ff_dataframe_in_duration = ff_dataframe[ff_dataframe['time'].between(duration_to_plot[0], duration_to_plot[1])]
        null_arcs_plotting_kwargs_temp['show_visible_segments_ff_indices'] = np.unique(ff_dataframe_utils.keep_only_ff_that_monkey_has_passed_by_closely(ff_dataframe_in_duration, max_distance_to_ff=ff_max_distance_to_path_to_show_visible_segments).ff_index.values)
    else:
        #show_visible_segments_ff_indices = [int(row.stop_ff_index), int(row.alt_ff_index)]
        null_arcs_plotting_kwargs_temp['show_visible_segments_ff_indices'] = [int(row.stop_ff_index), int(row.alt_ff_index)]

    for key, value in additional_plotting_kwargs.items():
        null_arcs_plotting_kwargs_temp[key] = value

    print('duration_to_plot:', duration_to_plot)
    #print('row.stop_ff_index:', row.stop_ff_index)


    returned_info = plot_trials.PlotTrials(
                duration_to_plot, 
                *PlotTrials_args,
                **null_arcs_plotting_kwargs_temp,   
                )
    

    R = returned_info['R']
    x0, y0 = returned_info['x0'], returned_info['y0']
    fig = returned_info['fig']
    axes = returned_info['axes']

    axes = fig.axes[0]
    axes = circle_a_ff(axes, int(row.stop_ff_index), ff_real_position_sorted, R, x0=x0, y0=y0, color='green')    
    axes = circle_a_ff(axes, int(row.alt_ff_index), ff_real_position_sorted, R, x0=x0, y0=y0, color='purple')   

    return fig, R, x0, y0


def circle_a_ff(axes, ff_index, ff_real_position_sorted, rotation_matrix, color='green', linewidth=3, alpha=0.65, x0=0, y0=0):    
    ff_position = ff_real_position_sorted[ff_index]
    ff_position_rotated = np.matmul(rotation_matrix, np.stack((ff_position[0], ff_position[1])))
    circle = plt.Circle((ff_position_rotated[0]-x0, ff_position_rotated[1]-y0), 27, facecolor=None, edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=1)
    axes.add_patch(circle)
    return axes


def show_null_arcs_func(axes, point_index_for_null_arc, monkey_information, R, x0=0, y0=0,
                        stop_null_arc_info_for_the_point=None,
                        alt_null_arc_info_for_the_point=None,
                        ):
    if point_index_for_null_arc is not None:
        x0, y0 = 0, 0
        temp_cum_mx, temp_cum_my = np.array(monkey_information['monkey_x'].loc[point_index_for_null_arc]), \
                                    np.array(monkey_information['monkey_y'].loc[point_index_for_null_arc])
        temp_cum_mxy_rotated = np.matmul(R, np.stack((temp_cum_mx, temp_cum_my)))
        axes.scatter(temp_cum_mxy_rotated[0]-x0, temp_cum_mxy_rotated[1]-y0, marker='^', s=130, color="cyan", zorder=5, alpha=0.6)

    if stop_null_arc_info_for_the_point is not None:
        if len(stop_null_arc_info_for_the_point) > 0:
            axes, whether_plotted = show_null_trajectory.plot_null_arcs_from_best_arc_df(axes, relevant_point_index=None, null_arc_info_for_plotting=stop_null_arc_info_for_the_point, x0=0, y0=0, 
                                                                                         rotation_matrix=R, polar=False, zorder=2, alpha=0.8, color='dodgerblue', marker_size=15)
    
    if alt_null_arc_info_for_the_point is not None:
        if len(alt_null_arc_info_for_the_point) > 0:
            axes, whether_plotted = show_null_trajectory.plot_null_arcs_from_best_arc_df(axes, relevant_point_index=None, null_arc_info_for_plotting=alt_null_arc_info_for_the_point, x0=0, y0=0, 
                                                                                         rotation_matrix=R, polar=False, zorder=2, alpha=0.5, color='orange', marker_size=10)
    return axes



def plot_alt_traj_vs_alt_stop(alt_traj, alt_stop, hue, title, slope, intercept):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=alt_traj, y=alt_stop, hue=hue, alpha=0.8)
    plt.title(title)
    plt.xlabel('Angle from monkey right before stop to alternative ff')
    plt.ylabel('Angle from stop ff landing to alternative ff')

    # plot a line of y=x that covers min(x) to max(x)
    plt.plot([min(alt_traj), max(alt_traj)], [min(alt_traj), max(alt_traj)], color='r')

    # Also plot the linear regression line
    x = [min(alt_traj), max(alt_traj)]
    y = [slope * x_elem + intercept for x_elem in x]
    plt.plot(x, y, color='g')

    # plot x=0 and y=0 lines
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--')

    # show legend
    plt.legend(['Data Points', 'y=x', 'Linear Regression'])
    plt.show()
    return



def _make_side_by_side_boxplots(current_df, order=[], title=None):

    # Filter DataFrame

    fig = px.box(current_df, x='box_name', y='slope', color='test_or_control', category_orders={'box_name': order},
                labels={'box_name': 'Type', 'slope': 'Slope', 'test_or_control': 'Test or Control'})

    fig.update_layout(
        title={
            'text': 'Slope of Linear Regression Line',
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        showlegend=True,  # Hide the legend
    )


    # Add title if title is provided
    if title is not None:
        fig.update_layout(title=title, title_x=0.5)

    # draw a line at y=1 that spans the width of the whole figure
    fig.add_shape(type='line', x0=-0.5, x1=len(order)-0.5, y0=1, y1=1, line=dict(color='red', width=2, dash='dash'))

    return fig
