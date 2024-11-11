
import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from decision_making_analysis import trajectory_info
from visualization import plot_behaviors_utils, monkey_heading_functions
from null_behaviors import show_null_trajectory


import os
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import rc, cm
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)




def make_one_polar_plot(max_time_before_next_visible_to_annotate=2.5, **current_polar_plot_kargs):
    """
    Plot a polar plot of fireflies and their behavior, including their trajectory, stops, and direction.

    Parameters:
    -----------
    input : numpy.ndarray
        A 2D array of shape (n, 4) or (n, 3) representing the fireflies' behavior. Each row contains the following information:
        - x coordinate of the firefly
        - angle of the firefly in radians
        - time since last visible
        - time till next visible
    current_traj_points : numpy.ndarray
        A 2D array of shape (m, 2) representing the trajectory of the fireflies. Each row contains the x and y coordinates of a point on the trajectory.
    current_stops : numpy.ndarray
        A 2D array of shape (k, 2) representing the stops of the fireflies. Each row contains the x and y coordinates of a stop.
    current_mheading: dict
        representing the direction of the monkey on the trajectory. 
    current_y_prob : float
        The probability of the monkey's behavior.
    current_point_index : int
        The index of the current point.
    current_more_ff_inputs : numpy.ndarray, optional
        A 2D array of shape (q, 4) representing additional fireflies' behavior.
    current_more_traj_points : numpy.ndarray, optional
        A 2D array of shape (r, 2) representing additional points on the trajectory.
    current_more_traj_stops : numpy.ndarray, optional
        A 2D array of shape (s, 2) representing additional stops.
    max_time_since_last_visible : float, optional
        The maximum time since last visible for a firefly to be plotted.
    show_reward_boundary : bool, optional
        Whether to show the reward boundary.
    show_null_arcs_from_best_arc_df : bool, optional
        Whether to show null arcs from the best arc dataframe.
    null_arc_info_for_plotting : pandas.DataFrame, optional
        A dataframe containing information about null arcs.
    show_direction_of_monkey_on_trajectory : bool, optional
        Whether to show the direction of the monkey on the trajectory.
    ff_colormap : str, optional
        The colormap to use for the fireflies.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The polar plot axes.
    markers : list
        A list of the markers used in the plot.
    marker_labels : list
        A list of the labels for the markers used in the plot.
    """
    ff_input = current_polar_plot_kargs['ff_input']
    current_y_prob = current_polar_plot_kargs['current_y_prob']
    current_traj_points = current_polar_plot_kargs['current_traj_points']
    current_stops = current_polar_plot_kargs['current_stops']
    current_point_index = current_polar_plot_kargs['current_point_index']
    current_mheading = current_polar_plot_kargs['current_mheading']
    current_more_ff_inputs = current_polar_plot_kargs['current_more_ff_inputs']
    current_more_traj_points = current_polar_plot_kargs['current_more_traj_points']
    current_more_traj_stops = current_polar_plot_kargs['current_more_traj_stops']
    max_time_since_last_visible = current_polar_plot_kargs['max_time_since_last_visible']
    show_reward_boundary = current_polar_plot_kargs['show_reward_boundary']
    show_null_arcs_from_best_arc_df = current_polar_plot_kargs['show_null_arcs_from_best_arc_df']
    null_arc_info_for_plotting = current_polar_plot_kargs['null_arc_info_for_plotting']
    show_direction_of_monkey_on_trajectory = current_polar_plot_kargs['show_direction_of_monkey_on_trajectory']
    ff_colormap = current_polar_plot_kargs['ff_colormap']



    markers = []
    marker_labels = []

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax = plot_behaviors_utils.set_polar_background_for_plotting(ax, 400, color_visible_area_in_background=True)
    
    # Plot fireflies
    ax, markers, marker_labels = plot_fireflies_from_input(ff_input, ax, markers, marker_labels, ff_colormap=ff_colormap, max_time=max_time_since_last_visible)
    if current_more_ff_inputs is not None:
        ax, markers, marker_labels = plot_fireflies_from_input(current_more_ff_inputs, ax, markers, marker_labels, ff_colormap=ff_colormap, 
                                                               max_time=max_time_since_last_visible, add_to_legend=False)

    # Plot a colorbar
    fig = plot_a_colorbar_for_time_since_last_visible(fig, ff_colormap=ff_colormap, max_time=max_time_since_last_visible)

    if show_reward_boundary:
        circle_center_x = np.cos(ff_input[:,1])*ff_input[:,0]
        circle_center_y = np.sin(ff_input[:,1])*ff_input[:,0]
        for i in range(ff_input.shape[0]):
            ax = plot_a_circle_in_polar_coordinates(ax, circle_center_x[i], circle_center_y[i], circle_r=25, color='orange')


    # Annotate the time since last visible
    for j in range(ff_input.shape[0]):
        if ff_input[j,2] < max_time_since_last_visible:
            ax.annotate(str(round(ff_input[j,2], 2)), xy=(ff_input[j,1], ff_input[j,0]), xytext=(ff_input[j,1], ff_input[j,0]+15), color='black', fontsize=11)
    if current_more_ff_inputs is not None:
        for j in range(current_more_ff_inputs.shape[0]):
            ax.annotate(str(round(current_more_ff_inputs[j,2], 2)), xy=(current_more_ff_inputs[j,1], current_more_ff_inputs[j,0]), xytext=(current_more_ff_inputs[j,1], current_more_ff_inputs[j,0]+15), color='black', fontsize=11)

    # Annotate the time till next visible, if applicable
    if ff_input.shape[1] > 3:
        for j in range(ff_input.shape[0]):
            if ff_input[j,3] < max_time_before_next_visible_to_annotate:
                ax.annotate(str(round(ff_input[j,3], 2)), xy=(ff_input[j,1], ff_input[j,0]), xytext=(ff_input[j,1], ff_input[j,0]-15), color='darkgreen', fontsize=9)
        if current_more_ff_inputs is not None:
            for j in range(current_more_ff_inputs.shape[0]):
                if current_more_ff_inputs[j,3] < max_time_before_next_visible_to_annotate:
                    ax.annotate(str(round(current_more_ff_inputs[j,3], 2)), xy=(current_more_ff_inputs[j,1], current_more_ff_inputs[j,0]), xytext=(current_more_ff_inputs[j,1], current_more_ff_inputs[j,0]-15), color='darkgreen', fontsize=9)

    # plot trajectory and stops if provided
    if current_traj_points is not None:
        ax, markers, marker_labels = plot_trajectory_and_stops(current_traj_points, current_stops, ax, markers, marker_labels)
    if current_more_traj_points is not None:
        ax, markers, marker_labels = plot_more_trajectory_and_stops(current_more_traj_points, current_more_traj_stops, ax, markers, marker_labels, color='blue', stop_color='magenta')

    ax.legend(markers, marker_labels, scatterpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    

    if current_y_prob is not None:
        ax.annotate('Probability: ' + str(round(current_y_prob,2)), xy=(pi, 200), xytext=(pi-0.3, 200), color='black', fontsize=11)


    if show_null_arcs_from_best_arc_df:
        if null_arc_info_for_plotting is None:
            raise ValueError('null_arc_info_for_plotting cannot be None if show_null_arcs_from_best_arc_df is True')
        if current_point_index is None:
            raise ValueError('all_point_index cannot be None if show_null_arcs_from_best_arc_df is True')
        ax, whether_plotted = show_null_trajectory.plot_null_arcs_from_best_arc_df(ax, [current_point_index], null_arc_info_for_plotting, x0=0, y0=0, rotation_matrix=None, polar=True)


    # plot triangle to show direction
    if show_direction_of_monkey_on_trajectory:
        ax = monkey_heading_functions.plot_triangles_to_show_monkey_headings_in_polar(ax, current_mheading)    
    
    return ax, markers, marker_labels





    
def make_one_polar_plot_for_decision_making(multi_class=False, data_kind='free selection', **current_polar_plot_kargs):

    
    real_label = current_polar_plot_kargs['real_label']
    predicted_label = current_polar_plot_kargs['predicted_label']
    ff_input = current_polar_plot_kargs['ff_input']
    
    if ff_input is None:
        raise ValueError('ff_input cannot be None')
    if real_label is None:
        raise ValueError('real_label cannot be None')
    if predicted_label is None:
        raise ValueError('predicted_label cannot be None')
    

    ax, markers, marker_labels = make_one_polar_plot(**current_polar_plot_kargs)
    
    
    # Circle the ff that the monkey pursues in reality
    # First check if the monkey is pursuing any ff in reality (based on annotation)
    no_real_ff_flag = False
    if multi_class:
        # real_label = real_label[:-1] # this is no longer necessary after eliminating the last column in self.y_all
        if np.all(real_label == False):
            no_real_ff_flag = True
    elif real_label >= ff_input.shape[0]:
        
        no_real_ff_flag = True
    ## Now, plot the elements
    if no_real_ff_flag:
        ax.annotate('Reality (based on annotation): not pursuing any ff', xy=(pi, 300), xytext=(pi-0.8, 300), color='black', fontsize=11)
    else:
        marker1 = ax.scatter(ff_input[real_label,1], ff_input[real_label,0], 220, facecolor="white", edgecolor='red', alpha=1, zorder=1)
        markers.insert(0, marker1)
        marker_labels.insert(0, 'Real')
        if not multi_class:
            ax.annotate('Real', xy=(ff_input[real_label,1], ff_input[real_label,0]), xytext=(ff_input[real_label,1], ff_input[real_label,0]+35), color='red', fontsize=11)


    # Circle the ff that the monkey pursues based on prediction
    # First check if the monkey is pursuing any ff in reality (based on annotation)
    no_predicted_ff_flag = False
    if multi_class:
        # predicted_label = predicted_label[:-1] # this is no longer necessary after eliminating the last column in self.y_all
        if np.all(predicted_label == False):
            no_predicted_ff_flag = True
    elif predicted_label >= ff_input.shape[0]:
        no_predicted_ff_flag = True

    ## Now, plot the circles to show the predicted and actual ff
    if no_predicted_ff_flag:
        ax.annotate('Prediction: not pursuing any ff', xy=(pi, 300), xytext=(pi-0.4, 300), color='black', fontsize=11)
    else:
        marker2 = ax.scatter(ff_input[predicted_label,1], ff_input[predicted_label,0], 150, facecolor="white", edgecolor='blue', alpha=1, zorder=1)
        markers.insert(0, marker2)
        marker_labels.insert(0, 'Predicted')
        if not multi_class:
            ax.annotate('Predicted', xy=(ff_input[predicted_label,1], ff_input[predicted_label,0]), xytext=(ff_input[predicted_label,1], ff_input[predicted_label,0]+50), color='blue', fontsize=11)

    ax.legend(markers, marker_labels, scatterpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Also annotate the original ff if the data kind of replacement
    print('\n')
    if data_kind == 'replacement':
        ax.annotate('original', xy=(ff_input[0,1], ff_input[0,0]), xytext=(ff_input[0,1], max(0, ff_input[0,0]-30)), color='green', fontsize=10)
        ax.annotate('alternative', xy=(ff_input[1,1], ff_input[1,0]), xytext=(ff_input[1,1], max(0, ff_input[1,0]-30)), color='green', fontsize=10)
        print('original xy:', (ff_input[0,1], ff_input[0,0]))
        print('alternative xy:', (ff_input[1,1], ff_input[1,0]))

    return ax




def make_polar_plots_for_decision_making(selected_cases=None, trials=None, max_plot_to_make=5, data_kind='free selection', 
                                        null_arcs_bundle=None, show_reward_boundary=False, **polar_plots_kwargs):
    # num_ff_per_row is only used to reshape the input to the original shape

    if polar_plots_kwargs['y_pred'].ndim > 1:
        if polar_plots_kwargs['y_pred'].shape[1] > 1:
            multi_class = True
        else:
            multi_class = False
    else:
        multi_class = False


    instance_to_plot = np.arange(polar_plots_kwargs['ff_inputs'].shape[0])[selected_cases]
    if len(instance_to_plot) > max_plot_to_make:
        instance_to_plot = instance_to_plot[:max_plot_to_make]


    for i in instance_to_plot:

        current_polar_plot_kargs = get_current_polar_plot_kargs(i, **polar_plots_kwargs, null_arcs_bundle=null_arcs_bundle, show_reward_boundary=show_reward_boundary)
                               
        ax = make_one_polar_plot_for_decision_making(multi_class=multi_class, data_kind=data_kind, 
                                                    **current_polar_plot_kargs)
        # Plot a title
        if trials is not None:
            ax.set_title('Input ' + str(i) +', Trial '+str(trials[i]), fontsize=15, pad=20)

        # print the prediction and the real label
        print('Prediction:', polar_plots_kwargs['y_pred'][selected_cases][i])
        print('Real:', polar_plots_kwargs['labels'][selected_cases][i])
            
        plt.show()
          




def plot_fireflies_from_input(input, ax, markers, marker_labels, ff_colormap='Greens', max_time=3, add_to_legend=True):
    ax.scatter(input[:,1], input[:,0], alpha=0.7, zorder=2, s=30, color=plt.get_cmap(ff_colormap)(1-input[:,2]/max_time), marker='o') # originally it was s=15
    marker0 = ax.scatter([], [], alpha=0.7, zorder=2, s=30, color="green", marker='o') # originally it was s=15
    if add_to_legend:
        markers.append(marker0)
        marker_labels.append('Fireflies')
    return ax, markers, marker_labels



def plot_a_colorbar_for_time_since_last_visible(fig, ff_colormap='Greens', max_time=3):
    title = 'Time Since FF Visible (s)'
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_time)
    cax = fig.add_axes([0.95, 0.05, 0.05, 0.4]) #[left, bottom, width, height] 
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(ff_colormap+"_r")), cax=cax, orientation='vertical')
    cbar.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
    cbar.ax.set_title(title, ha='left', y=1.04)
    cbar.ax.invert_yaxis()
    return fig



def plot_trajectory_and_stops(current_traj_points, current_stops, ax, markers, marker_labels, color='#ffc406', stop_color='purple'):
    if current_stops is None:
        marker3 = ax.scatter(current_traj_points[1,:], current_traj_points[0,:], alpha=0.7, zorder=2, s=25, marker='o', color=color)
        markers.append(marker3)
        marker_labels.append('Trajectory points in input')

    else:
        marker3_2 = ax.scatter(current_traj_points[1, np.where(current_stops==0)[0]], current_traj_points[0, np.where(current_stops==0)[0]], alpha=0.9, zorder=2, s=10, marker='o', color=color)
        markers.append(marker3_2)
        marker_labels.append('Trajectory points in input - w/o stop')
        
        marker3_1 = ax.scatter(current_traj_points[1, np.where(current_stops==1)[0]], current_traj_points[0, np.where(current_stops==1)[0]], alpha=0.8, zorder=2, s=25, marker='o', color=stop_color)
        markers.append(marker3_1)
        marker_labels.append('Trajectory points in input - with stop') # with stop means containing stops in the time bin
    return ax, markers, marker_labels


def plot_more_trajectory_and_stops(current_traj_points, current_stops, ax, markers, marker_labels, color='#ffc406', stop_color='purple'):
    if current_stops is None:
        marker3 = ax.scatter(current_traj_points[1,:], current_traj_points[0,:], alpha=0.7, zorder=2, s=25, marker='o', color=color)
        markers.append(marker3)
        marker_labels.append('Trajectory points not in input')

    else:
        marker3_2 = ax.scatter(current_traj_points[1, np.where(current_stops==0)[0]], current_traj_points[0, np.where(current_stops==0)[0]], alpha=0.9, zorder=2, s=10, marker='o', color=color)
        markers.append(marker3_2)
        marker_labels.append('Trajectory points not in input - w/o stop')
        
        marker3_1 = ax.scatter(current_traj_points[1, np.where(current_stops==1)[0]], current_traj_points[0, np.where(current_stops==1)[0]], alpha=0.8, zorder=2, s=25, marker='o', color=stop_color)
        markers.append(marker3_1)
        marker_labels.append('Trajectory points not in input - with stop')
    return ax, markers, marker_labels


def plot_a_circle_in_polar_coordinates(ax, circle_center_x, circle_center_y, circle_r=25, color='yellow'):
    theta = np.linspace(0, 2*np.pi, 10000)

    chunk = (circle_center_x * np.cos(theta) + circle_center_y * np.sin(theta)) 
    f_theta = circle_r**2 - circle_center_x**2 - circle_center_y**2 + chunk**2
    theta = theta[f_theta > 0]

    chunk = (circle_center_x * np.cos(theta) + circle_center_y * np.sin(theta)) 
    r = chunk + np.sqrt(circle_r**2 - circle_center_x**2 - circle_center_y**2 + chunk**2)
    r2 = chunk - np.sqrt(circle_r**2 - circle_center_x**2 - circle_center_y**2 + chunk**2)

    valid_indices1 = np.where(r > 0)[0]
    valid_indices2 = np.where(r2 > 0)[0]

    ax.scatter(theta[valid_indices1], r[valid_indices1], s=1, color=color)
    ax.scatter(theta[valid_indices2], r2[valid_indices2], s=1, color=color)
    return ax




def get_current_polar_plot_kargs(i, max_time_since_last_visible=2.5, show_reward_boundary=False, ff_colormap='Greens', null_arcs_bundle=None, 
                                 **polar_plots_kwargs):

    ff_inputs = polar_plots_kwargs['ff_inputs']
    labels = polar_plots_kwargs['labels']
    y_pred = polar_plots_kwargs['y_pred']
    time = polar_plots_kwargs['time']
    y_prob = polar_plots_kwargs['y_prob']
    num_ff_per_row = polar_plots_kwargs['num_ff_per_row']
    traj_points_to_plot = polar_plots_kwargs['traj_points_to_plot']
    traj_stops_to_plot = polar_plots_kwargs['traj_stops_to_plot']
    mheading = polar_plots_kwargs['mheading']
    show_direction_of_monkey_on_trajectory = polar_plots_kwargs['show_direction_of_monkey_on_trajectory']
    more_ff_inputs_to_plot = polar_plots_kwargs['more_ff_inputs_to_plot']
    more_traj_points_to_plot = polar_plots_kwargs['more_traj_points_to_plot']
    more_traj_stops_to_plot = polar_plots_kwargs['more_traj_stops_to_plot']
    num_ff_per_row = polar_plots_kwargs['num_ff_per_row']
    trajectory_features = polar_plots_kwargs['trajectory_features']

    if null_arcs_bundle is not None:
        show_null_arcs_from_best_arc_df = True
        null_arc_info_for_plotting = null_arcs_bundle['null_arc_info_for_plotting']
        all_point_index = null_arcs_bundle['all_point_index']
    else:
        show_null_arcs_from_best_arc_df = False
        null_arc_info_for_plotting = None
        all_point_index = None
    
    if ff_inputs is None:
        raise ValueError('ff_inputs cannot be None')
    if labels is None:
        raise ValueError('labels cannot be None')
    if y_pred is None:
        raise ValueError('y_pred cannot be None')
    
    # num_ff_per_row is only used to reshape the input to the original shape
    if num_ff_per_row is None:
        num_attributes_per_row = 3
    else:
        num_attributes_per_row = int(ff_inputs.shape[1]/num_ff_per_row)
    ff_input = ff_inputs[i,:].reshape(-1, num_attributes_per_row)
    real_label = labels[i]
    predicted_label = y_pred[i]
    num_trajectory_features = len(trajectory_features)

    current_y_prob = None
    current_stops = None
    current_traj_points = None
    current_more_ff_inputs = None
    current_more_traj_points = None
    current_more_traj_stops = None
    if y_prob is not None:
        current_y_prob = y_prob[i]
    if traj_stops_to_plot is not None:
        current_stops = traj_stops_to_plot[i]
    if traj_points_to_plot is not None:
        current_traj_points = traj_points_to_plot[i].reshape(num_trajectory_features,-1)
    if more_ff_inputs_to_plot is not None:
        current_more_ff_inputs = more_ff_inputs_to_plot[i].reshape(-1, num_attributes_per_row)
    if more_traj_points_to_plot is not None:
        current_more_traj_points = more_traj_points_to_plot[i].reshape(num_trajectory_features,-1)
    if more_traj_stops_to_plot is not None:
        current_more_traj_stops = more_traj_stops_to_plot[i]

    current_point_index = None
    if show_null_arcs_from_best_arc_df:
        if null_arc_info_for_plotting is None:
            raise ValueError('null_arc_info_for_plotting cannot be None if show_null_arcs_from_best_arc_df is True')
        if all_point_index is None:
            raise ValueError('all_point_index cannot be None if show_null_arcs_from_best_arc_df is True')
        current_point_index = all_point_index[i]

    # if need to plot triangle to show direction
    if show_direction_of_monkey_on_trajectory:
        current_mheading = monkey_heading_functions.find_current_mheading_for_the_point(mheading, i)
    else:
        current_mheading = None

    current_polar_plot_kargs = {'ff_input': ff_input,
                                'real_label': real_label,
                                'predicted_label': predicted_label,
                                'current_y_prob': current_y_prob,
                                'current_traj_points': current_traj_points,
                                'current_stops': current_stops,
                                'current_point_index': current_point_index,
                                'current_mheading': current_mheading,
                                'current_more_ff_inputs': current_more_ff_inputs,
                                'current_more_traj_points': current_more_traj_points,
                                'current_more_traj_stops': current_more_traj_stops,
                                'show_null_arcs_from_best_arc_df': show_null_arcs_from_best_arc_df,
                                'show_direction_of_monkey_on_trajectory': show_direction_of_monkey_on_trajectory,
                                'null_arc_info_for_plotting': null_arc_info_for_plotting,
                                'max_time_since_last_visible': max_time_since_last_visible,
                                'show_reward_boundary': show_reward_boundary,
                                'ff_colormap': ff_colormap,
                                'num_ff_per_row': num_ff_per_row,
    }


    return current_polar_plot_kargs




