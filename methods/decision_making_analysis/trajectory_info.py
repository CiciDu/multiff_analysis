
import sysfrom data_wrangling import basic_func
from visualization import animation_utils
from null_behaviors import curvature_utils, curv_of_traj_utils
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.GUAT import GUAT_and_TAFT

import numpy as np
import math
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from numpy import linalg as LA
import math
import pickle

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)



def find_trajectory_arc_info(point_index_array, curv_of_traj_df, ff_caught_T_sorted=None, monkey_information=None, curv_of_traj_mode='time', window_for_curv_of_traj=[-1, 1], truncate_curv_of_traj_by_time_of_capture=False):
    # curvature_df has duplicate point_index
    curv_of_traj_df_temp = curv_of_traj_df.groupby('point_index').first().reset_index().set_index('point_index')
    try:
        curvature_of_traj = curv_of_traj_df_temp.loc[point_index_array, 'curvature_of_traj'].values
    except KeyError:
        if ff_caught_T_sorted is None:
            raise ValueError('Since add_current_curvature_of_traj is True and the current information is insufficient, ff_caught_T_sorted cannot be None')
        # see which point_index is in point_index_array but not in curv_of_traj_df_temp
        missing_point_index = np.setdiff1d(point_index_array, curv_of_traj_df_temp.index.values)
        print('missing_point_index', missing_point_index)

        print('Since add_current_curvature_of_traj is True and the information in curv_of_traj_df is insufficient, we will calculate the curvature_of_traj now')
        curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, monkey_information, ff_caught_T_sorted, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        curvature_of_traj = curv_of_traj_df.loc[point_index_array, 'curvature_of_traj'].values
    except:
        print('Other errors?')

    return curvature_of_traj


        
def find_trajectory_data(time, monkey_information, time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10):
    """
    Finds the monkey's trajectory data for a given time range.

    Args:
        time (array-like): The time points to find the trajectory data for.
        monkey_information (pandas.DataFrame): A DataFrame containing the monkey's position, speed, angular speed, and angle at different time points.
        time_range_of_trajectory (tuple, optional): The time range (in seconds) to consider for the trajectory. Defaults to [-0.5, 0.5].
        num_time_points_for_trajectory (int, optional): The number of time points to sample within the time range. Defaults to 10.
    """
    
    traj_time_2d = np.tile(np.array(time).reshape(-1,1), (1, num_time_points_for_trajectory))
    duration_to_add = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_time_2d = traj_time_2d + duration_to_add

    monkey_indices = np.searchsorted(monkey_information['monkey_t'], traj_time_2d)
    monkey_indices = monkey_indices.reshape(-1)
    # make sure that the indices are within the range of monkey_information
    monkey_indices[monkey_indices < 0] = 0
    monkey_indices[monkey_indices > len(monkey_information)-1] = len(monkey_information)-1


    trajectory_data_dict = dict()
    for column in monkey_information.columns:
        trajectory_data_dict[column] = monkey_information[column].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)

    # monkey_x_2d = monkey_information['monkey_x'].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)
    # monkey_y_2d = monkey_information['monkey_y'].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)
    # monkey_dv_2d = monkey_information['monkey_speed'].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)
    # monkey_dw_2d = monkey_information['monkey_dw'].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)
    # monkey_angle_2d = monkey_information['monkey_angles'].values[monkey_indices].reshape(-1,num_time_points_for_trajectory)

    return traj_time_2d, trajectory_data_dict



def find_monkey_info_on_trajectory_relative_to_origin(monkey_indices, monkey_information, traj_x_2d, traj_y_2d, monkey_angle_2d, num_time_points_for_trajectory=10):
    """
    Finds monkey information on trajectory relative to origin.

    Args:
    - time_all (numpy array): array of time points 
    - monkey_information (pandas dataframe): dataframe containing monkey information for all trials
    - traj_x_2d (numpy array): array of x-coordinates of trajectory points
    - traj_y_2d (numpy array): array of y-coordinates of trajectory points
    - monkey_angle_2d (numpy array): array of monkey angles for all time points
    - num_time_points_for_trajectory (int): number of time points selected from the duration of the trajectory (default=10)

    Returns:
    - traj_distances (numpy array): array of distances between trajectory points and monkey positions
    - traj_angles (numpy array): array of angles between trajectory points and monkey positions
    - traj_angles_to_boundaries (numpy array): array of angles between trajectory points and monkey boundaries
    - monkey_angle_on_trajectory_relative_to_the_current_north (numpy array): array of monkey angles on each trajectory point
    """
    # make sure that the indices are within the range of monkey_information
    monkey_indices[monkey_indices < 0] = 0
    monkey_indices[monkey_indices > len(monkey_information)-1] = len(monkey_information)-1

    monkey_xy = monkey_information[['monkey_x', 'monkey_y']].values[monkey_indices] # find the (0,0) positions of monkey for all the trials
    monkey_angle = monkey_information[['monkey_angles']].values[monkey_indices] # find the heading of monkey at "0" time for all the trials
    # let's also get the monkey angle on each trajectory point, with 0 pointing to the north at the current moment
    monkey_angle_on_trajectory_relative_to_the_current_north =  monkey_angle_2d - monkey_angle    
    # expand monkey_xy to the same shape as traj_xy_2d
    monkey_xy = np.tile(monkey_xy, (1, num_time_points_for_trajectory)).reshape(-1,2)
    monkey_angle = np.tile(monkey_angle, (1, num_time_points_for_trajectory)).reshape(-1)


    # treat traj_xy_2d as ffxy, and use the real monkey_xy to calculate distance and angle, so that we can plot trajectory points on polar plot
    traj_xy_2d = np.concatenate([traj_x_2d.reshape(-1,1), traj_y_2d.reshape(-1,1)], axis=1)        
    traj_distances = LA.norm(traj_xy_2d-monkey_xy, axis=1)
    traj_angles = basic_func.calculate_angles_to_ff_centers(ff_x=traj_x_2d.reshape(-1), ff_y=traj_y_2d.reshape(-1), mx=monkey_xy[:,0], my=monkey_xy[:,1], m_angle=monkey_angle)
    traj_distances = traj_distances.reshape(-1, num_time_points_for_trajectory)
    traj_angles = traj_angles.reshape(-1, num_time_points_for_trajectory)

    return traj_distances, traj_angles, monkey_angle_on_trajectory_relative_to_the_current_north



def generate_feature_names_given_relative_time_points(relative_time_points, num_time_points, original_feature_names='monkey_distance'):
    new_feature_names = []
    # if feature_name is a string:
    if isinstance(original_feature_names, str):
        original_feature_names = [original_feature_names]
    
    for feature_name in original_feature_names:
        for i in range(num_time_points):
            new_feature_names.append(feature_name + '_' + str(round(relative_time_points[i], 2)) + 's') 
    return new_feature_names



def generate_trajectory_position_data(time_all, monkey_information, time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10, trajectory_features=['monkey_distance', 'monkey_angle_to_origin']):
    relative_time_points_of_trajectory = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_time_2d, trajectory_data_dict = find_trajectory_data(time_all, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)

    traj_x_2d = trajectory_data_dict['monkey_x']
    traj_y_2d = trajectory_data_dict['monkey_y']
    monkey_angle_2d = trajectory_data_dict['monkey_angles']

    monkey_indices = np.searchsorted(monkey_information['monkey_t'], time_all) 
    traj_distances, traj_angles, monkey_angle_on_trajectory_relative_to_the_current_north = find_monkey_info_on_trajectory_relative_to_origin(monkey_indices, monkey_information, traj_x_2d, traj_y_2d, monkey_angle_2d, num_time_points_for_trajectory=num_time_points_for_trajectory)
    trajectory_data_dict['monkey_distance'] = traj_distances
    trajectory_data_dict['monkey_angle_to_origin'] = traj_angles
    trajectory_data_dict['monkey_angle_on_trajectory_relative_to_the_current_north'] = monkey_angle_on_trajectory_relative_to_the_current_north
    

    for feature in trajectory_features:
        if feature not in trajectory_data_dict.keys():
            raise ValueError('feature ' + feature + ' is not in trajectory_data_dict.keys()')
    traj_points = np.concatenate([trajectory_data_dict[feature] for feature in trajectory_features], axis=1)
    trajectory_feature_names = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=trajectory_features)

    # traj_points: array, containing the traj_distances and traj_angles for each row in X_all
    # trajectory_feature_names: list, containing the names of the features in traj_points

    return traj_points, trajectory_feature_names



def generate_trajectory_velocity_data(time_all, monkey_information, time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10):
    relative_time_points_of_trajectory = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_time_2d, trajectory_data_dict = find_trajectory_data(time_all, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
    traj_dv_2d = trajectory_data_dict['monkey_speed']
    traj_dw_2d = trajectory_data_dict['monkey_dw']
    monkey_dvdw = np.concatenate([traj_dv_2d, traj_dw_2d], axis=1)
    trajectory_feature_names = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=['monkey_dv', 'monkey_dw'])

    return monkey_dvdw, trajectory_feature_names



def furnish_machine_learning_data_with_trajectory_data_func(X_all, time_all, monkey_information, trajectory_data_kind="position",
                                                            time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10, add_traj_stops=True):
    """
    Augments the input features for machine learning with trajectory data and stops information.

    Parameters:
    -----------
    X_all : array-like, shape (n_samples, n_features)
        The input features for machine learning.
    time_all : array-like, shape (n_samples,)
        The time points for each sample in X_all.
    monkey_information : dict
        A dictionary containing the monkey's information, such as position and velocity.
    trajectory_data_kind : str, optional (default="position")
        The type of trajectory data to use. Either "position" or "velocity".
    time_range_of_trajectory : list, optional (default=[-0.5, 0.5])
        The time range around each time point to consider for the trajectory data.
    num_time_points_for_trajectory : int, optional (default=10)
        The number of time points to use for the trajectory data.
    add_traj_stops : bool, optional (default=True)
        Whether to add stops information to the input features.

    Returns:
    --------
    X_all : array, shape (n_samples, n_features + n_trajectory_features + n_stops_features)
        The augmented input features for machine learning.
    traj_points : array, shape (n_samples, n_trajectory_features)
        The trajectory data for each sample in X_all.
    traj_stops : array, shape (n_samples, n_stops_features)
        The stops information for each sample in X_all.
    trajectory_feature_names : list
        The names of the trajectory features added to X_all.
    """

    traj_points, trajectory_feature_names = generate_trajectory_position_data(time_all, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)

    if trajectory_data_kind == "position":  
        X_all = np.concatenate([X_all, traj_points], axis=1)      
    elif trajectory_data_kind == "velocity":
        monkey_dvdw, trajectory_feature_names = generate_trajectory_velocity_data(time_all, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
        X_all = np.concatenate([X_all, monkey_dvdw], axis=1)
    else:
        raise ValueError("trajectory_data_kind must be either 'position' or 'velocity'")

    traj_stops = np.array([])
    if add_traj_stops:
        traj_stops, temp_trajectory_feature_names = generate_stops_info(time_all, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
        X_all = np.concatenate([X_all, traj_stops], axis=1)
        trajectory_feature_names.extend(temp_trajectory_feature_names)

    # X_all: array, containing the input features for machine learning
    # traj_points: array, containing the traj_distances and traj_angles for each row in X_all

    return X_all, traj_points, traj_stops, trajectory_feature_names




def generate_stops_info(time_all, monkey_information, time_range_of_trajectory=[-0.5, 0.5], num_time_points_for_trajectory=10):
    """
    Generate stopping information for each row in X_all.

    Parameters:
    time_all (array): array of time points
    monkey_information (DataFrame): dataframe containing monkey information
    time_range_of_trajectory (list): list of two floats representing the time range of the trajectory (default [-0.5, 0.5])
    num_time_points_for_trajectory (int): number of time points for the trajectory (default 10)

    Returns:
    traj_stops (array): array containing the stopping information for each row in X_all, where 1 means there has been stops in the bin and 0 means not
    trajectory_feature_names (list): list containing the names of the features in traj_stops
    """

    relative_time_points_of_trajectory = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_time_2d = np.tile(np.array(time_all).reshape(-1,1), (1, num_time_points_for_trajectory))
    duration_to_add = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_time_2d = traj_time_2d + duration_to_add

    bin_width = (time_range_of_trajectory[1] - time_range_of_trajectory[0]) / (num_time_points_for_trajectory-1)
    monkey_dt = (monkey_information['monkey_t'].iloc[-1] - monkey_information['monkey_t'].iloc[0]) / (len(monkey_information)-1)
    num_points_in_window = math.ceil(bin_width/monkey_dt + 1)
    if num_points_in_window % 2 == 0:
        num_points_in_window += 1

    # First smooth the stopping information of the points so that if a point in the vicinity (within bin_width s) of a point containing a stop, then the point can be seen as containing a stop as well
    # The problem is that dt is not consistent ... but it shouldn't make too big of a difference, since it's only 2.6% of points are below 0.015
    convolve_pattern = np.ones(num_points_in_window)/num_points_in_window
    monkey_stops = (monkey_information['monkey_speeddummy'].values - 1) * (-1) # 1 means having a stop; 0 means not
    stops_convolved = np.convolve(monkey_stops, convolve_pattern, 'same')
    stops_convolved = (stops_convolved > 0).astype(int)
    indices = np.searchsorted(monkey_information['monkey_t'].values, traj_time_2d) 
    # make sure that indices don't exceed the range of monkey_information
    indices[indices < 0] = 0
    indices[indices > len(monkey_information)-1] = len(monkey_information)-1
    traj_stops = stops_convolved[indices]

    ## The code below has more accurate points because it's based on time alone, not point_index, but it's much slower
    # monkey_information_temp = monkey_information[monkey_information['monkey_t'].between(traj_time_2d.min(), traj_time_2d.max())].copy()
    # traj_stops = add_stops_info_to_one_row_of_trajectory_info(traj_time_2d[0,:], monkey_information_temp).reshape(1,-1)
    # for time_1d in traj_time_2d[1:,:]:
    #     stopping_info = add_stops_info_to_one_row_of_trajectory_info(time_1d, monkey_information_temp).reshape(1,-1)
    #     traj_stops = np.concatenate([traj_stops, stopping_info], axis=0)
    # print("traj_stops.shape", traj_stops.shape)

    trajectory_feature_names = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names='whether_stopped')

    # traj_stops: array, containing the stopping information for each row in X_all, where 1 means there has been stops in the bin and 0 means not; 
        # the number of points in each row is equal to the number of trajectory points for each row in X_all
    # trajectory_feature_names: list, containing the names of the features in traj_stops

    return traj_stops, trajectory_feature_names



def add_stops_info_to_one_row_of_trajectory_info(traj_time_1d, monkey_information):
    # this is only used to generate more accurate points, but it's much slower
    bin_width = (traj_time_1d[-1]-traj_time_1d[0])/(len(traj_time_1d)-1)
    min_time = traj_time_1d[0] - bin_width/2
    max_time = traj_time_1d[-1] + bin_width/2
    time_bins = np.arange(min_time, max_time+bin_width/2, bin_width)  # add bin_width/2 to max_time so that max_time will be included in the array
    num_bins = len(time_bins)-1
    monkey_information.loc[:,'corresponding_bins'] = np.searchsorted(time_bins, monkey_information.loc[:,'monkey_t'].values)
    monkey_sub = monkey_information[monkey_information['corresponding_bins'].between(1, num_bins)].copy()
    monkey_sub = monkey_sub[['corresponding_bins', 'monkey_speeddummy']].groupby('corresponding_bins').min()  
    stopping_info = monkey_sub['monkey_speeddummy'].values   # 0 means there has been stops in the bin
    stopping_info = -(stopping_info-1) # reverting 0 and 1 so 1 means there has been stops in the bin
    return stopping_info



def add_stops_info_to_monkey_information(traj_time, monkey_information):
    bin_width = traj_time[1]-traj_time[0]
    min_time = traj_time[0] - bin_width/2
    max_time = traj_time[-1] + bin_width/2
    time_bins = np.arange(min_time, max_time+bin_width/2, bin_width)  # add bin_width/2 to max_time so that max_time will be included in the array
    num_bins = len(time_bins)-1
    monkey_information.loc[:,'corresponding_bins'] = np.searchsorted(time_bins, monkey_information.loc[:,'monkey_t'].values)
    monkey_sub = monkey_information[monkey_information['corresponding_bins'].between(1, num_bins)].copy()
    monkey_sub = monkey_sub[['corresponding_bins', 'monkey_speeddummy']].groupby('corresponding_bins').min()  
    stopping_info = monkey_sub['monkey_speeddummy'].values   # 0 means there has been stops in the bin
    stopping_info = -(stopping_info-1) # reverting 0 and 1 so 1 means there has been stops in the bin
    monkey_information.loc[:,'monkey_stops_based_on_bins'] = stopping_info
    return monkey_information


def combine_trajectory_and_stop_info_and_curvature_info(traj_points_df, traj_stops_df, relevant_curv_of_traj_df, use_more_as_prefix=False):
    # first, verify that all three df share the same point_index
    if not (traj_points_df.point_index.values == traj_stops_df.point_index.values).all():
        raise ValueError('The point_index of traj_points_df and traj_stops_df are not the same')
    if not (traj_points_df.point_index.values == relevant_curv_of_traj_df.point_index.values).all():
        raise ValueError('The point_index of traj_points_df and relevant_curv_of_traj_df are not the same')
    
    # then, drop point_index from all three df
    traj_points_df = traj_points_df.copy().drop(['point_index'], axis=1)
    traj_stops_df = traj_stops_df.copy().drop(['point_index'], axis=1)
    relevant_curv_of_traj_df = relevant_curv_of_traj_df.copy().drop(['point_index'], axis=1)

    # then, save the list of feature names in a dictionary
    traj_data_feature_names = dict()
    if use_more_as_prefix:
        traj_data_feature_names['more_traj_points'] = traj_points_df.columns.values
        traj_data_feature_names['more_traj_stops'] = traj_stops_df.columns.values
        traj_data_feature_names['more_relevant_curv_of_traj'] = relevant_curv_of_traj_df.columns.values
    else:
        traj_data_feature_names['traj_points'] = traj_points_df.columns.values
        traj_data_feature_names['traj_stops'] = traj_stops_df.columns.values
        traj_data_feature_names['relevant_curv_of_traj'] = relevant_curv_of_traj_df.columns.values

    # then, concatenate the three df
    traj_data_df = pd.concat([traj_points_df, traj_stops_df, relevant_curv_of_traj_df], axis=1)
    return traj_data_df, traj_data_feature_names


def make_traj_data_feature_names(time_range_of_trajectory, num_time_points_for_trajectory, use_more_as_prefix=False, traj_point_features=['monkey_distance', 'monkey_angle'], relevant_curv_of_traj_feature_names=['curvature_of_traj']):
    relative_time_points_of_trajectory = np.linspace(time_range_of_trajectory[0], time_range_of_trajectory[1], num_time_points_for_trajectory)
    traj_data_feature_names = dict()
    if use_more_as_prefix:
        traj_data_feature_names['more_traj_points'] = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=traj_point_features)
        traj_data_feature_names['more_traj_stops'] = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=['whether_stopped'])
        traj_data_feature_names['more_relevant_curv_of_traj'] = ['curvature_of_traj']
    else:
        traj_data_feature_names['traj_points'] = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=traj_point_features)
        traj_data_feature_names['traj_stops'] = generate_feature_names_given_relative_time_points(relative_time_points_of_trajectory, num_time_points_for_trajectory, original_feature_names=['whether_stopped'])
        traj_data_feature_names['relevant_curv_of_traj'] = relevant_curv_of_traj_feature_names
    return traj_data_feature_names


def make_all_traj_feature_names(time_range_of_trajectory, num_time_points_for_trajectory, time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot,
                                traj_point_features=['monkey_distance', 'monkey_angle']):
    traj_data_feature_names = make_traj_data_feature_names(time_range_of_trajectory, num_time_points_for_trajectory, traj_point_features=traj_point_features)
    if (time_range_of_trajectory_to_plot is not None) & (num_time_points_for_trajectory_to_plot is not None):
        more_traj_data_feature_names = make_traj_data_feature_names(time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot, use_more_as_prefix=True, traj_point_features=traj_point_features)
        # combine the two dictionary
        all_traj_feature_names = traj_data_feature_names | more_traj_data_feature_names
    else:
        all_traj_feature_names = traj_data_feature_names
    return all_traj_feature_names


def retrieve_or_make_all_traj_feature_names(dir_name, monkey_name, exists_ok=True, save=True, time_range_of_trajectory=None, num_time_points_for_trajectory=None, time_range_of_trajectory_to_plot=None, num_time_points_for_trajectory_to_plot=None,
                                            traj_point_features=['monkey_distance', 'monkey_angle']):
    file_path = os.path.join(dir_name, monkey_name, 'all_traj_feature_names.pkl')
    if os.path.exists(file_path) & exists_ok:
        with open(file_path, 'rb') as f:
            all_traj_feature_names = pickle.load(f)
    else:
        if time_range_of_trajectory is None or num_time_points_for_trajectory is None or time_range_of_trajectory_to_plot is None or num_time_points_for_trajectory_to_plot is None:
            raise ValueError('Since retrieval failed, all kwargs should not be None.')
    
        all_traj_feature_names = make_all_traj_feature_names(time_range_of_trajectory, num_time_points_for_trajectory, time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot,
                                                                traj_point_features=traj_point_features)
        if save == True:
            with open(os.path.join(dir_name, monkey_name, 'all_traj_feature_names.pkl'), 'wb') as f:
                pickle.dump(all_traj_feature_names, f)

    return all_traj_feature_names



def calculate_monkey_angle_and_distance_from_now_to_other_time(all_current_point_indices, monkey_information, monkey_xy_from_other_time):
    monkey_xy_now = monkey_information.loc[all_current_point_indices, ['monkey_x', 'monkey_y']].values
    monkey_angle_now = monkey_information.loc[all_current_point_indices, 'monkey_angles'].values
    # distance_from_monkey_now_to_monkey_at_other_time means the distance between the monkey position when the ff was last seen and the monkey position now
    distance_from_monkey_now_to_monkey_at_other_time = np.linalg.norm(monkey_xy_from_other_time - monkey_xy_now, axis=1)
    angle_from_monkey_now_to_monkey_at_other_time = basic_func.calculate_angles_to_ff_centers(ff_x=monkey_xy_from_other_time[:,0], ff_y=monkey_xy_from_other_time[:,1], mx=monkey_xy_now[:,0], my=monkey_xy_now[:,1], m_angle=monkey_angle_now)
    return distance_from_monkey_now_to_monkey_at_other_time, angle_from_monkey_now_to_monkey_at_other_time



def add_distance_and_angle_from_monkey_now_to_monkey_when_ff_last_seen_or_next_seen(df, monkey_information, ff_dataframe, monkey_xy_from_other_time=None, use_last_seen=True):
    prefix = 'last_seen' if use_last_seen else 'next_seen'
    if monkey_xy_from_other_time is None:
        additional_placeholder_mapping = {'monkey_x': [9999, False], 'monkey_y': [False, False]}
        df = decision_making_utils.add_attributes_last_seen_or_next_seen_for_each_ff_in_df(df, ff_dataframe, attributes=['monkey_x', 'monkey_y'], additional_placeholder_mapping=additional_placeholder_mapping, use_last_seen=use_last_seen)
        monkey_xy_from_other_time = df[[prefix + '_monkey_x', prefix + '_monkey_y']].values
    all_current_point_indices = df['point_index'].values
    distance_from_monkey_now_to_monkey_from_other_time, angle_from_monkey_now_to_monkey_from_other_time = calculate_monkey_angle_and_distance_from_now_to_other_time(all_current_point_indices, monkey_information, monkey_xy_from_other_time)
    df['distance_from_monkey_now_to_monkey_when_ff_' + prefix] = distance_from_monkey_now_to_monkey_from_other_time
    df['angle_from_monkey_now_to_monkey_when_ff_' + prefix] = angle_from_monkey_now_to_monkey_from_other_time
    # find where the placeholder values are located, and replace them with new values
    placeholder_indices = np.where(monkey_xy_from_other_time[:,0] == 9999)[0]
    df.iloc[placeholder_indices, df.columns.get_indexer(['distance_from_monkey_now_to_monkey_when_ff_' + prefix])] = 400
    df.iloc[placeholder_indices, df.columns.get_indexer(['angle_from_monkey_now_to_monkey_when_' + prefix])] = 0
    return df



def add_distance_and_angle_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(df, ff_dataframe, monkey_information, use_last_seen=True):
    
    df = df.copy()
    all_ff_index = df['ff_index'].values
    all_current_point_indices = df['point_index'].values
    suffix = '_last_seen' if use_last_seen else '_next_seen'

    # if use_last_seen is False, then we assume we're using the information when the ff is next seen
    monkey_xy_now = monkey_information.loc[all_current_point_indices, ['monkey_x', 'monkey_y']].values
    monkey_angle_now = monkey_information.loc[all_current_point_indices, 'monkey_angles'].values
    all_current_time = monkey_information.loc[all_current_point_indices, 'monkey_t'].values

    additional_placeholder_mapping = {'ff_x': [9999, False], 'ff_y': [9999, False]}
    ff_info = decision_making_utils.find_attributes_of_ff_when_last_visible_OR_next_visible(all_ff_index, all_current_time, ff_dataframe, use_last_seen=use_last_seen, 
                                                                      attributes=['ff_x', 'ff_y'], additional_placeholder_mapping=additional_placeholder_mapping)

    ff_xy_when_ff_last_seen_or_next_seen = ff_info[['ff_x', 'ff_y']].values
    df[['ff_x', 'ff_y']] = ff_xy_when_ff_last_seen_or_next_seen
    df['distance_from_monkey_now_to_ff_when_ff' + suffix] = np.linalg.norm(ff_xy_when_ff_last_seen_or_next_seen - monkey_xy_now, axis=1)
    df['angle_from_monkey_now_to_ff_when_ff' + suffix] = basic_func.calculate_angles_to_ff_centers(ff_x=ff_xy_when_ff_last_seen_or_next_seen[:,0], ff_y=ff_xy_when_ff_last_seen_or_next_seen[:,1], mx=monkey_xy_now[:,0], my=monkey_xy_now[:,1], m_angle=monkey_angle_now)

    df.loc[df['ff_x'] == 9999, 'distance_from_monkey_now_to_ff_when_ff' + suffix] = 1000
    df.loc[df['ff_x'] == 9999, 'angle_from_monkey_now_to_ff_when_ff' + suffix] = 0
    
    df.drop(['ff_x', 'ff_y'], axis=1, inplace=True)
    return df



def add_curv_diff_from_monkey_now_to_ff_when_ff_last_seen_or_next_seen(df, monkey_information, ff_real_position_sorted, ff_caught_T_sorted, use_last_seen=True, curv_of_traj_df=None):
    df = df.copy()
    suffix = '_last_seen' if use_last_seen else '_next_seen'
    df, temp_curvature_df = GUAT_and_TAFT.find_curv_diff_for_ff_info(df, monkey_information, ff_real_position_sorted, curv_of_traj_df=curv_of_traj_df)
    df.rename(columns={'curv_diff': 'curv_diff_from_monkey_now_to_ff_when_ff' + suffix}, inplace=True)
    return df
