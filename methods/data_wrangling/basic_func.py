import sys
import os
import numpy as np
import sys
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from contextlib import contextmanager
from os.path import exists
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



class HiddenPrints:
    def __enter__(self):
        _original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = _original_stdout



def plt_config(title=None, xlim=None, ylim=None, xlabel=None, ylabel=None, colorbar=False, sci=False):
    """
    Set some parameters for plotting
    
    """
    for field in ['title', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        if eval(field) != None: getattr(plt, field)(eval(field))
    if isinstance(sci, str): plt.ticklabel_format(style='sci', axis=sci, scilimits=(0,0))
    if isinstance(colorbar,str): plt.colorbar(label=colorbar)
    elif colorbar: plt.colorbar(label = '$Number\ of\ Entries$')


@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    """
    Set some parameters for plotting
    
    """
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = "sans serif"
    global fig; fig = plt.figure(dpi=dpi)
    yield
    plt.show()



class HiddenPrints:
    """
    Hide all the printed statements while running the coded

    Parameters
    ----------
    merged_df: dataframe
        containing various characteristics of each trial for both the monkey and the agent(s)

    """

    def __enter__(self):
        _original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = _original_stdout



def find_intersection(intervals, query):
    """
    Find intersections between intervals. Intervals are open and are 
    represented as pairs (lower bound, upper bound). 
    The source of the code is:
    source: https://codereview.stackexchange.com/questions/203468/
    find-the-intervals-which-have-a-non-empty-intersection-with-a-given-interval


    Parameters
    ----------
    intervals: array_like, shape=(N, 2) 
        Array of intervals.
    query: array_like, shape=(2,) 
        Interval to query

    Returns
    -------
    indices_of_overlapped_intervals: array
        Array of indexes of intervals that overlap with query
    
    """
    intervals = np.asarray(intervals)
    lower, upper = query
    indices_of_overlapped_intervals = np.where((lower < intervals[:, 1]) & (intervals[:, 0] < upper))[0]
    return indices_of_overlapped_intervals


def flash_on_ff_in_trial(ff_flash_sorted, duration):
    """
    Find the index of the fireflies that have flashed during the trial

    Parameters
    ----------
    ff_flash_sorted: list
        contains the time that each firefly flashes on and off
    duration: array_like, shape=(2,) 
        the starting time and ending time of the trial

    Returns
    -------
    flash_index: list
        the indices of the fireflies that have flashed during the trial (among all fireflies)
    
    """
    flash_index = []
    for index in range(len(ff_flash_sorted)):
      # Take out the flashing-on and flashing-off time of this particular firefly
      ff = ff_flash_sorted[index]
      if len(find_intersection(ff, duration)) > 0:
        flash_index.append(index)
    return flash_index




def calculate_angles_to_ff_centers(ff_x, ff_y, mx, my, m_angle):
    """
    Calculate the angle to the center of a firefly or multiple fireflies from the monkey's or the agent's perspective
    Positive angle means to the left of the monkey, and negative angle means to the right of the monkey

    Parameters
    ----------
    ff_position: np.array
        containing the x-coordinates and the y-coordinates of all fireflies
    mx: np.array
        the x-coordinates of the monkey/agent
    my: np.array
        the y-coordinates of the monkey/agent
    m_angle: np.array
        the angles that the monkey/agent heads toward


    Returns
    -------
    angles_to_ff: np.array
        containing the angles of the centers of the fireflies to the monkey/agent

    """
    
    # find the angles of the given fireflies to the agent
    angles_to_ff = np.arctan2(ff_y-my, ff_x-mx)-m_angle
    # make sure that the angles are between (-pi, pi]
    angles_to_ff = np.remainder(angles_to_ff, 2*pi)

    # if distance to ff is very small, make the angle 0:
    try:
        distances_to_ff = LA.norm(np.stack([ff_x-mx, ff_y-my], axis=1), axis=1)
    except np.exceptions.AxisError:
        distances_to_ff = LA.norm(np.array([ff_x-mx, ff_y-my]), axis=0)
    if np.any(distances_to_ff < 0.001):
        print(f'Note: {np.sum(distances_to_ff < 0.001)} fireflies are too close to the monkey/agent. Their angles are set to 0.')
        angles_to_ff[distances_to_ff < 0.001] = 0

    try:
        angles_to_ff[angles_to_ff > pi] = angles_to_ff[angles_to_ff > pi] - 2*pi
    except TypeError:
        # then angles_to_ff must be a scalar
        if angles_to_ff > pi:
            angles_to_ff = angles_to_ff - 2*pi

    try:
        angles_to_ff[angles_to_ff <= -pi] = angles_to_ff[angles_to_ff <= -pi] + 2*pi
    except TypeError:
        # then angles_to_ff must be a scalar
        if angles_to_ff <= -pi:
            angles_to_ff = angles_to_ff + 2*pi

    return angles_to_ff


def calculate_angles_to_ff_boundaries(angles_to_ff, distances_to_ff, ff_radius=10):
    """
    Calculate the angle to the boundary of a firefly or multiple fireflies from the monkey's or the agent's perspective

    Parameters
    ----------
    angles_to_ff: np.array
        containing the angles of the centers of the fireflies to the monkey/agent
    distances_to_ff: np.array
        containing the distances of the fireflies to the agent
    ff_radius: num
        the radius of the visible area of each firefly  

    Returns
    -------
    angles_to_boundaries: np.array
        containing the smallest angles of the reward boundaries of the fireflies to the agent

    """

    # Adjust the angle based on reward boundary (i.e. find the smallest angle from the agent to the reward boundary)
    # using trignometry 
    side_opposite = ff_radius
    # hypotenuse cannot be smaller than side_opposite
    hypotenuse = np.clip(distances_to_ff, a_min=side_opposite, a_max=2000)
    theta = np.arcsin(np.divide(side_opposite, hypotenuse))
    # we use absolute values of angles here so that the adjustment will only make the angles smaller
    angle_adjusted_abs = np.abs(angles_to_ff) - np.abs(theta)
    # thus we can find the smallest absolute angle to the firefly, which is the absolute angle to the boundary of the firefly
    angles_to_boundaries_abs = np.clip(angle_adjusted_abs, 0, pi)
    # restore the signs of the angles
    angles_to_boundaries = np.sign(angles_to_ff) * angles_to_boundaries_abs

    # for the points where the monkey/agent is within the ff_radius, the reward boundary will be 0
    if isinstance(angles_to_boundaries, float) is False:
        angles_to_boundaries[distances_to_ff <= ff_radius] = 0
    else:
        if distances_to_ff <= ff_radius:
            angles_to_boundaries = 0
    
    return angles_to_boundaries



def calculate_change_in_abs_ff_angle(current_ff_index, angles_to_ff, angles_to_boundaries, ff_real_position_sorted, monkey_x_array, 
                                     monkey_y_array, monkey_angles_array, in_memory_indices):
    ## To also calculate delta_angles_to_ff and delta_angles_to_boundaries:
    prev_monkey_xy_relevant = np.stack([monkey_x_array[in_memory_indices-1], monkey_y_array[in_memory_indices-1]], axis=1)
    prev_ff_distance_relevant = LA.norm(prev_monkey_xy_relevant-ff_real_position_sorted[current_ff_index], axis=1)
    prev_monkey_angles_relevant = monkey_angles_array[in_memory_indices-1]    
    prev_angles_to_ff = calculate_angles_to_ff_centers(ff_x=ff_real_position_sorted[current_ff_index, 0], ff_y=ff_real_position_sorted[current_ff_index, 1], mx=prev_monkey_xy_relevant[:, 0], my=prev_monkey_xy_relevant[:, 1], m_angle=prev_monkey_angles_relevant)
    prev_angles_to_boundaries = calculate_angles_to_ff_boundaries(angles_to_ff=prev_angles_to_ff, distances_to_ff=prev_ff_distance_relevant)      
    delta_abs_angles_to_ff = np.abs(angles_to_ff) - np.abs(prev_angles_to_ff)
    delta_abs_angles_to_boundary = np.abs(angles_to_boundaries) - np.abs(prev_angles_to_boundaries)
    return delta_abs_angles_to_ff, delta_abs_angles_to_boundary


    

def get_cum_distance_traveled(currentTrial, ff_caught_T_new, monkey_information):
    """
    Find the length of the trajectory run by the monkey in the current trial

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time


    Returns
    -------
    distance: numeric
        the length of the trajectory run by the monkey in the current trial

    """
    duration = [ff_caught_T_new[currentTrial-1], ff_caught_T_new[currentTrial]]
    cum_iloc_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    distance = 0
    if len(cum_iloc_indices) > 1:
        cum_t = np.array(monkey_information['monkey_t'].iloc[cum_iloc_indices])
        cum_speed = np.array(monkey_information['monkey_speed'].iloc[cum_iloc_indices])
        distance = np.sum((cum_t[1:] - cum_t[:-1])*cum_speed[1:])
    return distance




def get_distance_between_two_points(currentTrial, ff_caught_T_new, monkey_information, ff_believed_position_sorted):
    """
    Find the absolute displacement between the target for the currentTrial and the target for currentTrial.
    Return 9999 if the monkey has hit the border at one point.

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 


    Returns
    -------
    displacement: numeric
        the distance between the starting and ending points of the monkey during a trial; 
        returns 9999 if the monkey has hit the border at any point during the trial

    """
    duration = [ff_caught_T_new[currentTrial-1], ff_caught_T_new[currentTrial]]
    cum_iloc_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    displacement = 0
    if len(cum_iloc_indices) > 1:
      cum_mx, cum_my = np.array(monkey_information['monkey_x'].iloc[cum_iloc_indices]), np.array(monkey_information['monkey_y'].iloc[cum_iloc_indices])
      # If the monkey has hit the boundary
      if np.any(cum_mx[1:]-cum_mx[:-1] > 10) or np.any(cum_my[1:]-cum_my[:-1] > 10):
        displacement = 9999
      else:
        displacement = LA.norm(ff_believed_position_sorted[currentTrial]-ff_believed_position_sorted[currentTrial-1])
    return displacement


def retrieve_and_merge_patterns_or_features(df_name='pattern_frequencies', data_directory_list=['all_monkey_data/raw_monkey_data/individual_monkey_data']):
    """
    Retrieve and merge patterns or features from CSV files in specified directories.

    Parameters:
    df_name (str): Name of the DataFrame to retrieve (default is 'pattern_frequencies').
    data_directory_list (list): List of directories to search for the CSV files.

    Returns:
    pd.DataFrame: Merged DataFrame containing the patterns or features.
    list: List of file paths from which the data was retrieved.
    """
    csv_name = df_name + '.csv'
    all_filepaths = []
    merged_df = pd.DataFrame()

    for data_directory in data_directory_list:
        for rootdir, dirs, files in os.walk(data_directory):
            if csv_name in files:
                filepath = os.path.join(rootdir, csv_name)
                df_of_interest = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)

                # Determine the current data name
                data_name_path = os.path.dirname(os.path.dirname(rootdir))
                data_name = os.path.basename(data_name_path)
                if 'data_' not in data_name:
                    raise ValueError("The data name must contain 'data_'")
                df_of_interest['Data'] = data_name

                # Determine the current monkey name
                monkey_name = os.path.basename(os.path.dirname(data_name_path))
                df_of_interest['Monkey'] = monkey_name[7:] if len(monkey_name) > 8 else monkey_name

                # Merge the DataFrame
                all_filepaths.append(filepath)
                merged_df = pd.concat([merged_df, df_of_interest], ignore_index=True)

    return merged_df, all_filepaths



# def retrieve_and_merge_patterns_or_features(df_name = 'pattern_frequencies', data_directory_list = ['all_monkey_data/raw_monkey_data/individual_monkey_data']):
#     csv_name = df_name + '.csv'
#     all_filepaths = []
#     merged_df = pd.DataFrame([])

#     FIRST_DF = True
#     for data_directory in data_directory_list:
#         for rootdir, dirs, files in os.walk(data_directory):
#             for file in files:
#                 if file == csv_name:        
#                     filepath = os.path.join(rootdir, csv_name)
#                     df_of_interest = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)

#                     # Figure out the current data name
#                     data_path = rootdir
#                     data_name = os.path.split(rootdir)[1]
#                     if data_name == 'patterns_and_features':
#                         data_path = os.path.dirname(rootdir)
#                         data_name = os.path.split(data_path)[1]
#                     df_of_interest['Data'] = data_name
                    

#                     # Figure out the current monkey name
#                     monkey_path = os.path.dirname(data_path)
#                     monkey_name = os.path.split(monkey_path)[1]
#                     if len(monkey_name) > 8:
#                         df_of_interest['Monkey'] = monkey_name[7:]
#                     else:
#                         df_of_interest['Monkey'] = monkey_name
                    

#                     # Merge the df
#                     all_filepaths.append(filepath)
#                     if FIRST_DF:
#                         merged_df = df_of_interest
#                         FIRST_DF = False
#                     else:
#                         merged_df = pd.concat([merged_df, df_of_interest], ignore_index=True)
        
#     return merged_df, all_filepaths



def save_df_to_csv(df, df_name, data_folder_name, exists_ok=False):
    if data_folder_name:
        csv_name = df_name + '.csv'
        filepath = os.path.join(data_folder_name, csv_name)
        if exists(filepath) & exists_ok:
            print(filepath, 'already exists.')
        else:
            os.makedirs(data_folder_name, exist_ok = True)
            df.to_csv(filepath)
            print("new", df_name, "is stored in ", filepath)


def find_currentTrial_or_num_trials_or_duration(ff_caught_T_new, currentTrial=None, num_trials=None, duration=None):
    # Among currentTrial, num_trials, duration, either currentTrial and num_trials must be specified, or duration must be specified
    if duration is None:
        duration = [ff_caught_T_new[currentTrial-num_trials], ff_caught_T_new[currentTrial]]
    # elif duration[1] > ff_caught_T_new[-1]:
    #    raise ValueError("The second element of duration must be smaller than the last element of ff_caught_T_new")
           
    if currentTrial is None:   
        try:
            if len(ff_caught_T_new) > 0:
                # Take the max of the results from two similar methods
                # Method 1:
                earlier_trials = np.where(ff_caught_T_new <= duration[1])[0]
                if len(earlier_trials) > 0:
                    currentTrial = earlier_trials[-1]
                else:
                    currentTrial = 1
                # Method 2:
                later_trials = np.where(ff_caught_T_new >= duration[0])[0]
                if len(later_trials) > 0:
                    currentTrial_2 = later_trials[0]
                else:
                    currentTrial_2 = 1
                currentTrial = max(currentTrial, currentTrial_2)
        except Exception as e:
            print('Finding currentTrial failed:', e, 'currentTrial is set to None')
            currentTrial = None
    if num_trials is None: 
        try:
            if len(ff_caught_T_new) > 0:
                trials_after_first_capture = np.where(ff_caught_T_new <= duration[0])[0]
                if len(trials_after_first_capture) > 0:
                    num_trials = max(1, currentTrial-trials_after_first_capture[-1])
                else:
                    num_trials = 1
        except Exception as e:
            num_trials = None
    
    return currentTrial, num_trials, duration


def take_out_a_sample_from_arrays(sampled_indices, *args):
    sampled_args = []
    for arg in args:
        sampled_arg = arg[sampled_indices]
        sampled_args.append(sampled_arg)
    return sampled_args

def take_out_a_sample_from_df(sampled_indices, *args):
    sampled_args = []
    for arg in args:
        sampled_arg = arg.iloc[sampled_indices]
        sampled_args.append(sampled_arg)
    return sampled_args


def initialize_monkey_sessions_df(dir_name='all_monkey_data/raw_monkey_data/individual_monkey_data'):
    list_of_monkey_name = []
    list_of_data_name = []
    for monkey_name in ['monkey_Bruno', 'monkey_Schro']: # 'monkey_Quigley'
        monkey_path = os.path.join(dir_name, monkey_name)
        for data_name in os.listdir(monkey_path):
            if data_name[0] == 'd':
                list_of_monkey_name.append(monkey_name)
                list_of_data_name.append(data_name)
    sessions_df = pd.DataFrame({'monkey_name': list_of_monkey_name, 'data_name': list_of_data_name})
    sessions_df['finished'] = False
    return sessions_df


def check_whether_finished(sessions_df, monkey_name, data_name):
    current_session_info = ((sessions_df['monkey_name'] == monkey_name) & (sessions_df['data_name'] == data_name))
    whether_finished = sessions_df.loc[current_session_info, 'finished'].item()
    return whether_finished


def find_outlier_position_index(data, outlier_z_score_threshold = 2):
    data = np.array(data)
    # calculate standard deviation in rel_curv_to_stop_ff_center
    std = np.std(data)
    # find z-score of each point
    z_score = (data - np.mean(data)) / std
    # find outliers
    outlier_positions = np.where(np.abs(z_score) > outlier_z_score_threshold)[0]
    return outlier_positions


def init_variations_list_func(ref_point_params_based_on_mode, folder_path=None, monkey_name=None):
    key_value_pairs = []
    for key, values in ref_point_params_based_on_mode.items():
        key_value_pairs.extend([[key, i] for i in values])
    variations_list = pd.DataFrame(key_value_pairs, columns=['ref_point_mode', 'ref_point_value'])

    variations_list['monkey_name'] = monkey_name
    variations_list['stored'] = False
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        df_path = os.path.join(folder_path, 'variations_list.csv')
        variations_list.to_csv(df_path)
    return variations_list


def make_rotation_matrix(x0, y0, x1, y1):
    # find a rotation matrix so that (x1, y1) is to the north of (x0, y0)

    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(y1 - y0, x1 - x0)     
    c, s = np.cos(theta), np.sin(theta)
    # Find the rotation matrix
    rotation_matrix = np.array(((c, -s), (s, c)))
    return rotation_matrix




