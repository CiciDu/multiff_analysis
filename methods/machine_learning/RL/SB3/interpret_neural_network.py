import sys
from data_wrangling import basic_func

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'





def sample_and_visualize_from_neural_network(sac_model,
                        sample_size = 1000, 
                        full_memory = 4, 
                        ff_radius=10, 
                        invisible_distance = 400, 
                        color_variable = "dv",
                        plot_in_xy_coord = False,
                        use_angle_to_boundary = False,
                        const_distance = None, 
                        const_angle = None, 
                        const_memory = None, 
                        fill_empty_observational_space_with_placeholders = True,
                        norm_input = False, 
                        add_2nd_ff = False,
                        plot_colorbar = True,
                        const_distance2 = None,
                        const_angle2 = None,
                        const_memory2 = None,
                        axes = None,
                        fig = None,
                        return_fig_and_axes = False,
                        show_plot = True,
                        data_folder_name = None,
                        ):
  
    plt.rcdefaults()
    stacked_array, all_actions, angle2center, distances = generate_observations_and_actions(sac_model, sample_size=sample_size, full_memory=full_memory,  
                                 const_distance=const_distance, const_angle=const_angle, const_memory=const_memory, 
                                 fill_empty_observational_space_with_placeholders=fill_empty_observational_space_with_placeholders, norm_input=norm_input, add_2nd_ff=add_2nd_ff, 
                                 const_distance2=const_distance2, const_angle2=const_angle2, const_memory2=const_memory2)

    
    fig, axes = visualize_generated_actions(stacked_array, all_actions, angle2center, distances, color_variable=color_variable, use_angle_to_boundary=use_angle_to_boundary,
                                  plot_in_xy_coord=plot_in_xy_coord, plot_colorbar=plot_colorbar, const_distance=const_distance, const_angle=const_angle, 
                                  const_memory=const_memory, axes=axes, fig=fig, show_plot=show_plot, data_folder_name=data_folder_name)
    
    if return_fig_and_axes:
        return fig, axes



def visualize_generated_actions(stacked_array,
                            all_actions,
                            angle2center, 
                            distances,
                            color_variable = "dv",
                            const_distance = None, 
                            const_angle = None, 
                            const_memory = None, 
                            plot_in_xy_coord = False,
                            plot_colorbar = True,
                            use_angle_to_boundary = False,
                            fig = None,
                            axes = None,
                            return_fig_and_axes = False,
                            show_plot = True,
                            data_folder_name = None,
                            ):

    """
    Plot actions based on observations to aid the understanding of the neural network of the agent;
    Note, currently this function can only be used for the SB3 agent because the LSTM agent needs hidden outputd to generate action


    Parameters
    ----------
    sac_model: obj
        the agent
    stacked_array: np.array
        containing a stack of simulated observations
    all_actions: np.array
        containing actions conducted by the agent based on each observation in stacked_array
    angle2center: np.array
        containing the angles from the agent to the centers of fireflies
    distances: np.array
        containing the distances from the agent to the centers of fireflies   
    color_variable: str
        "dv" or "dw"; denotes whether the color signifies the linear velocity or the angular velocity
    const_distance: num or None
        if num, then it denotes the distance of the ff used by all observations; otherwise, distance will be randomly sampled;
        note that one of the three variables -- distance, angle, and memory -- needs to be constant; otherwise, no plot will be made
    const_angle: num or None
        if num, it denotes the angle of the ff used by all observations; otherwise, distance will be randomly sampled
    const_memory: num or None
        if num, it denotes the memory of the ff used by all observations; otherwise, distance will be randomly sampled
    plot_in_xy_coord: bool 
        whether to plot in the Cartesian coordinate system
    plot_colorbar: bool
        whether to plot the colorbar
    use_angle_to_boundary: bool
        whether to use the angle of the monkey to the reward boundary of the firefly instead of the center of the firefly
    ax: obj, optional
        matplotlib ax object
    fig: obj, optional
        matplotlib fig object  
    return_fig_and_axes: bool,
        whether to return fig and ax     
    show_plot: bool
        whether to show the plot  
    data_folder_name: str
        name of the data folder where the image will be stored

    
    Returns
    ----------
    stacked_array: np.array
        containing a stack of simulated observations
    all_actions: np.array
        containing actions conducted by the agent based on each observation in stacked_array    

    """

        
    if axes is None:
        fig, axes = plt.subplots(figsize=(7, 4), dpi=150)

    plt.rcParams['savefig.dpi']=300
    cmap = cm.viridis_r
    color_values = {"dv": (all_actions[:, 1]+1)*200, "dw": all_actions[:, 0]*pi/2}
    
    # When plotting, which type of angles will be chosen depends on whether use_angle_to_boundary is True
    angle_for_plotting = {True: stacked_array[:, 1], False: stacked_array[:, 0]}


    # # plot the dots
    # if const_distance:
    #     make_title_with_constant_distance(color_variable, const_distance)
    #     graph_name = color_variable + "_at_distance=" + str(round(const_distance))
    #     axes.scatter(stacked_array[:, 3], angle_for_plotting[use_angle_to_boundary], s=2, c=color_values[color_variable], cmap=cmap)
    # elif const_angle:
    #     make_title_with_constant_angle(color_variable, const_angle)
    #     graph_name = color_variable + "_at_angle=" + str(round(const_angle, 2))
    #     axes.scatter(stacked_array[:, 3], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)
    # elif const_memory:
    #     make_title_with_constant_memory(color_variable, const_memory)
    #     graph_name = color_variable + "_at_memory=" + str(round(const_memory))
    #     if plot_in_xy_coord is True:
    #         all_x, all_y, valid_indices = convert_to_xy_coord(angle2center, distances)
    #         axes.scatter(all_x[valid_indices], all_y[valid_indices], s=2, c=color_values[color_variable][valid_indices], cmap=cmap)
    #     else:
    #         axes.scatter(angle_for_plotting[use_angle_to_boundary], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)



 # plot the dots
    if const_distance:
        make_title_with_constant_distance(color_variable, const_distance, stacked_array)
        graph_name = color_variable + "_at_distance=" + str(round(const_distance))
        scatterplot = axes.scatter(stacked_array[:, 3], angle_for_plotting[use_angle_to_boundary], s=2, c=color_values[color_variable], cmap=cmap)
    elif const_angle:
        make_title_with_constant_angle(color_variable, const_angle, stacked_array)
        graph_name = color_variable + "_at_angle=" + str(round(const_angle, 2))
        scatterplot = axes.scatter(stacked_array[:, 3], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)  
    elif const_memory:
        make_title_with_constant_memory(color_variable, const_memory, stacked_array)
        graph_name = color_variable + "_at_memory=" + str(round(const_memory))
        if plot_in_xy_coord is True:
            all_x, all_y, valid_indices = convert_to_xy_coord(angle2center, distances)
            scatterplot = axes.scatter(all_x[valid_indices], all_y[valid_indices], s=2, c=color_values[color_variable][valid_indices], cmap=cmap)
        else:
            scatterplot = axes.scatter(angle_for_plotting[use_angle_to_boundary], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)



    # plot the colorbar
    if plot_colorbar:
        fig, axes = plot_colorbar_for_interpreting_neural_network(fig, axes, scatterplot, all_actions, color_variable)
        
    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        figure_name = os.path.join(data_folder_name, f"{graph_name}.png")
        plt.savefig(figure_name, bbox_inches='tight')
    
    if show_plot:
        plt.show()
        

    return fig, axes



def generate_observations_and_actions(
                      sac_model,
                      sample_size = 1000, 
                      full_memory = 4, 
                      ff_radius=10, 
                      invisible_distance = 400, 
                      const_distance = None, 
                      const_angle = None, 
                      const_memory = None, 
                      fill_empty_observational_space_with_placeholders = True,
                      norm_input = False, 
                      add_2nd_ff = False,
                      const_distance2 = None,
                      const_angle2 = None,
                      const_memory2 = None,
                      ):
    """
    Plot actions based on observations to aid the understanding of the neural network of the agent;
    Note, currently this function can only be used for the SB3 agent because the LSTM agent needs hidden outputd to generate action;
    In addition, among const_distance, const_angle, and const_memory, only one variable can be passed a value, while the rest need to remain None.


    Parameters
    ----------
    sac_model: obj
        the agent
    sample_size: num
        the number of dots to be plotted in the plot
    full_memory: num
        the value for memory when a firefly is visible; in other words, the maximum value for memory
    ff_radius: num
        the reward boundary of the firefly
    invisible_distance: num
        the distance beyond which a firefly will be invisible
    const_distance: num or None
        if num, then it denotes the distance of the ff used by all observations; otherwise, distance will be randomly sampled;
        note that one of the three variables -- distance, angle, and memory -- needs to be constant; otherwise, no plot will be made
    const_angle: num or None
        if num, it denotes the angle of the ff used by all observations; otherwise, distance will be randomly sampled
    const_memory: num or None
        if num, it denotes the memory of the ff used by all observations; otherwise, distance will be randomly sampled
    fill_empty_observational_space_with_placeholders: bool
        whether to pad the rest of the obs space with default ff (placeholders)
    norm_input: bool
        whether input is normed

    Returns
    ----------
    stacked_array: np.array
        containing a stack of simulated observations
    all_actions: np.array
        containing actions conducted by the agent based on each observation in stacked_array

    """


    # Sample for the 1st ff in the obs space
    if const_distance2 is not None:
        distance_upper_limits = const_distance2
    else:
        distance_upper_limits = invisible_distance
    angle2center, angle2boundary, distances, memories = sample_observations_for_one_ff(const_distance, const_angle, const_memory, sample_size, full_memory, distance_upper_limits)
    # stack the attributes together, so that each row is an observation 
    stacked_array = np.stack((angle2center, angle2boundary, distances, memories), axis=1)    

    # If applicable, sample for the 1st ff in the obs space
    if add_2nd_ff:
        angle2center2, angle2boundary2, distances2, memories2 = sample_observations_for_one_ff(const_distance2, const_angle2, const_memory2, sample_size, full_memory, distance_upper_limits=distances)
        stacked_array2 = np.stack((angle2center2, angle2boundary2, distances2, memories2), axis=1) 
        stacked_array = np.hstack((stacked_array, stacked_array2))
    
    angle2center = angle2center
    distances = distances    
    

    if fill_empty_observational_space_with_placeholders:
        total_ff = int(sac_model.observation_space.shape[0]/4)
        existing_ff = 2 if add_2nd_ff else 1
        num_default_ff = total_ff - existing_ff
        if num_default_ff > 0:
            print("Note: ", num_default_ff, " fireflies are added as placeholders to pad the observational space.")
            stacked_array = add_placeholders_for_observations(num_default_ff, stacked_array, sample_size, invisible_distance)

    if norm_input is True:
        stacked_array = norm_input(stacked_array, invisible_distance, full_memory)

    # for each observation, use the network to generate the agent's action
    all_actions = generate_actions(stacked_array, sample_size, sac_model)

    return stacked_array, all_actions, angle2center, distances




def sample_observations_for_one_ff(const_distance, const_angle, const_memory, sample_size, full_memory, distance_upper_limits):
    if const_distance is None:
        # Need to sample random distances
        distances = np.random.uniform(low=np.zeros([sample_size, ]), high=distance_upper_limits, size=[sample_size, ])   
    else:
        distances = np.ones(sample_size)*const_distance

    if const_angle is None:
        angle2center = np.random.uniform(low=-pi, high=pi, size=sample_size,)
        angle2boundary = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angle2center, distances_to_ff=distances) 
    else:
        angle2center = np.ones([sample_size, ]) * const_angle
        # Make sure the angles are within the range of -pi to pi.
        angle2center = np.remainder(angle2center, 2*pi)
        angle2center[angle2center > pi] = angle2center[angle2center > pi] - 2*pi
        angle2boundary = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angle2center, distances_to_ff=distances)      
    
    if const_memory is None:
        memories = np.random.randint(low=0, high=full_memory+1, size=[sample_size, ])
    else:
        memories = np.ones(sample_size)*const_memory

    return angle2center, angle2boundary, distances, memories

        
                  


def make_title_with_constant_distance(color_variable, const_distance, stacked_array):
    plt.title("Angle-to-center vs Memory, colored by " + color_variable + ", with distance = " + str(round(const_distance)), y=1.08)
    plt.xlabel("Memory", labelpad=10) #
    plt.ylabel("Angle to center (rad)", labelpad=10)
    plt.xticks(range(math.floor(min(stacked_array[:, 3])), math.ceil(max(stacked_array[:, 3]))+1))   



def make_title_with_constant_angle(color_variable, const_angle, stacked_array):      
    plt.title("Memory vs Distance, colored by " + color_variable + ", with angle = " + str(round(const_angle, 2)), y=1.08)
    plt.xlabel("Memory", labelpad=10)
    plt.ylabel("Distance (cm)", labelpad=10)
    plt.xticks(range(math.floor(min(stacked_array[:, 3])), math.ceil(max(stacked_array[:, 3]))+1))   


def make_title_with_constant_memory(color_variable, const_memory, stacked_array):       
    plt.title("Angle-to-center vs Distance, colored by " + color_variable + ", with memory = " + str(round(const_memory)), y=1.08)
    plt.xlabel("Egocentric x-coord (cm) ", labelpad=10)  
    plt.ylabel("Egocentric y-coord (cm)", labelpad=10)


def norm_input(stacked_array, invisible_distance, full_memory):
    stacked_array[0::4] = stacked_array[0::4]/pi
    stacked_array[1::4] = stacked_array[1::4]/pi
    stacked_array[2::4] = (stacked_array[2::4]/invisible_distance-0.5)*2
    stacked_array[3::4] = (stacked_array[3::4]/full_memory-0.5)*2
    return stacked_array


def generate_actions(stacked_array, sample_size, sac_model):
    # for each observation, use the network to generate the agent's action
    all_actions = np.zeros([sample_size, 2])
    for i in range(sample_size):
        obs = stacked_array[i]
        action, _ = sac_model.predict(obs, deterministic=True)
        all_actions[i] = action.copy()
    return all_actions


def add_placeholders_for_observations(num_default_ff, stacked_array, sample_size, invisible_distance):
    # add placeholders to fill up the obs space
    for i in range(num_default_ff):
        placeholders = np.tile(np.array([[0], [0], [invisible_distance], [1]]), sample_size).T
        stacked_array = np.concatenate([stacked_array, placeholders], axis=1)
    return stacked_array


# def plot_colorbar_for_interpreting_neural_network(fig, all_actions, color_variable):
#     max_value = {"dv": 200, "dw": pi}
#     colorbar_title = {"dv": 'dv(cm/s)', "dw": 'dw(rad/s)'}
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value[color_variable])
#     cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
#     cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis_r), cax=cax, orientation='vertical')
#     cbar.ax.set_title(colorbar_title[color_variable], ha='left', y=1.08)
#     cbar.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
#     cbar.outline.set_visible(False)
#     return fig


def plot_colorbar_for_interpreting_neural_network(fig, axes, scatterplot, all_actions, color_variable):
    
    colorbar_title = {"dv": 'dv(cm/s)', "dw": 'dw(rad/s)'}
    #max_value = {"dv": 200, "dw": pi}
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value[color_variable])
    cbar = fig.colorbar(scatterplot, ax=axes, orientation='vertical', shrink=0.8)

    cbar.ax.set_title(colorbar_title[color_variable], ha='left', y=1.08)
    cbar.ax.tick_params(axis='y', color='lightgrey', direction="in", right=True, length=5, width=1.5)
    cbar.outline.set_visible(False)
    return fig, axes    


def convert_to_xy_coord(angle2center, distances):
    all_x = np.multiply(np.cos(angle2center+pi/2), distances)
    all_y = np.multiply(np.sin(angle2center+pi/2), distances)
    # only plot a dot if it's not behind the agent 
    valid_indices = np.where(all_y > 0)[0]
    return all_x, all_y, valid_indices






def combine_6_plots_for_neural_network(sac_model,
                                        full_memory = 4, 
                                        ff_radius=10, 
                                        invisible_distance = 400, 
                                        const_distance = 100, 
                                        const_angle = 0.5, 
                                        const_memory = 3, 
                                        sample_size_for_const_distance = 1000,
                                        sample_size_for_const_angle = 1000,
                                        sample_size_for_const_memory = 10000,
                                        add_2nd_ff = True,
                                        plot_in_xy_coord = True,
                                        const_distance2 = None,
                                        const_angle2 = None,
                                        const_memory2 = None,
                                        data_folder_name = None,
                                        data_folder_name2 = None,
                                        file_name = None,
                                        file_name2 = None,
                                        ):
    num_rows = 3
    num_cols = 2
    counter = 0
    fig = plt.figure(figsize=(num_cols*6, num_rows*4))
    shared_kwargs = {"return_fig_and_axes": True,
                     "add_2nd_ff": add_2nd_ff, 
                     "plot_in_xy_coord" : plot_in_xy_coord, 
                     "full_memory" : full_memory, 
                     "ff_radius" : ff_radius, 
                     "invisible_distance" : invisible_distance, 
                     "const_distance2" : const_distance2,
                     "const_angle2" : const_angle2, 
                     "const_memory2" : const_memory2,
                     "plot_colorbar" : True, 
                     "show_plot" : False,
                     }

    # angle vs. distance, dv
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_memory, const_memory = const_memory, color_variable = "dv", **shared_kwargs) 
                                                                                        
    # angle vs. distance, dw
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_memory, const_memory = const_memory, color_variable = "dw", **shared_kwargs) 
                                                                                        
    # angle vs. memory, dv
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_distance, const_distance = const_distance, color_variable = "dv", **shared_kwargs) 
                                                                                                                                  
    # angle vs. memory, dw
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_distance, const_distance = const_distance, color_variable = "dw", **shared_kwargs) 
                                                                                                         
    # distance vs. memory, dv
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_angle, const_angle = const_angle, color_variable = "dv", **shared_kwargs) 

    # distance vs. memory, dw
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    fig, axes = sample_and_visualize_from_neural_network(sac_model, axes = axes, fig = fig, sample_size = sample_size_for_const_angle, const_angle = const_angle, color_variable = "dw", **shared_kwargs) 

    plt.tight_layout()
    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        if file_name is None:
            file_name = 'combined_6_plots_for_neural_network.png'
        figure_name = os.path.join(data_folder_name, f"{file_name}.png")
        plt.savefig(figure_name)
    
    if data_folder_name2 is not None:
        if not os.path.isdir(data_folder_name2):
            os.makedirs(data_folder_name2)
        if file_name2 is None:
            file_name2 = file_name
        figure_name = os.path.join(data_folder_name2, f"{file_name}.png")
        plt.savefig(figure_name)
    
        


    plt.show()
    