import sys
from pattern_discovery import pattern_by_trials, pattern_by_trials

import os
import math
from math import pi
import re
import os.path
import neo
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from os.path import exists
from scipy.ndimage import gaussian_filter1d


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)



class smr_extractor(object):   
    def __init__(self, raw_data_folder_path):
        folder_path = raw_data_folder_path
        files_names = [file for file in os.listdir(folder_path) if 'smr' in file]
        self.full_path_file_names = [os.path.join(folder_path,files_names[i]) 
                                     for i, value in enumerate(files_names)] # a list contains 2 file path in total
        
    def extract_data(self):
        Channel_signal_output = []
        marker_list = [] # used to store information about time of juice rewards
        smr_sampling_rate = None
        
        for index, file_name in enumerate(self.full_path_file_names):  # loop 2 files one by one
            seg_reader = neo.io.Spike2IO(filename=file_name, try_signal_grouping=False).read_segment() # read out segments
            
            # get sampling rate, only need to get it once
            if index == 0: 
                smr_sampling_rate = seg_reader.analogsignals[0].sampling_rate 
                
            analog_length = min([i.size for i in seg_reader.analogsignals]) # in case analog channels have different shape
            
            Channel_index = [] # create an empty list to store channel names

            # Get channel data from all channels except the last one
            for C_index, C_data in enumerate(seg_reader.analogsignals[:-1]): # -1 indicates we discard 'Raw' channel   
                shape = seg_reader.analogsignals[C_index].shape[1] # See how many channels are contained in each element of the list
                # append arrays of channel data into a big array
                if C_index==0:
                  Channel_signal = C_data.as_array()[:analog_length,]
                else:
                  Channel_signal = np.append(Channel_signal, C_data.as_array()[:analog_length,], axis=1)
                # append the channel name
                for i in range(shape):
                  Channel_index.append(seg_reader.analogsignals[C_index].name) 

            # Also append an array for time
            Channel_signal = np.append(Channel_signal, np.asarray(seg_reader.analogsignals[0].times[:analog_length,]).reshape(analog_length,1), axis=1)# get time stamps and put in Channel_signal
            Channel_index.append('Time') 

            Channel_signal_output.append(pd.DataFrame(Channel_signal,columns=Channel_index))

            # Figure out the time of juice rewards
            # First find 'marker' channel
            marker_channel_index = [index for index, value in enumerate(seg_reader.events) if value.name == 'marker'][0] 
            # Then find marker labels. We'll only use the markers whose label is 4
            marker_labels = seg_reader.events[marker_channel_index].get_labels().astype('int') # get 'marker' labels
            # Then get the time of the juice rewards
            marker_values = seg_reader.events[marker_channel_index].as_array() 
            marker = {'labels': marker_labels, 'values': marker_values} # arrange labels and values in a dict
            marker_list.append(marker)
            
        return Channel_signal_output, marker_list, smr_sampling_rate



# Read log file here
class log_extractor(object):
    # Modified from Ruiyi's codes
    def __init__(self, raw_data_folder_path):
        self.folder_path = raw_data_folder_path      
        files_names = [file for file in os.listdir(self.folder_path) if ('txt' in file) and ('s' in file)]
        self.full_path_file_names = os.path.join(self.folder_path, files_names[0])

        # files_names = file_name
        # full_path_file_names = os.path.join(folder_path,files_names)        


    def extract_data(self, monkey_information_exists_OK=True, min_distance_to_calculate_angle=5, interocular_dist=4):
        ffLinenumberList = []
        ff_information = []
        ffname_index = 0
        
        with open(self.full_path_file_names,'r',encoding='UTF-8') as content:
            log_content = content.readlines()
         
        for LineNumber, line in enumerate(log_content):
            key_ff = re.search('Firefly', line)
            key_monkey = re.search('Monkey', line)
            if key_ff is not None:
                ffLinenumberList.append(LineNumber)
            if key_monkey is not None:
                monkeyLineNum = LineNumber

        # get ff data
        for index, LineNumber in enumerate(ffLinenumberList):
            FF_caught_T = []
            FF_Position = []
            FF_believed_position = []
            FF_flash_T = []
            
            if index == len(ffLinenumberList)-1:
                log_content_block = log_content[ffLinenumberList[index]+1:monkeyLineNum]
            else:
                log_content_block = log_content[ffLinenumberList[index]+1:ffLinenumberList[index+1]]
            

            for line in log_content_block:
                if len(line.split(' ')) == 5:
                    if 'inf' not in line:
                        having_inf_in_line = False
                        FF_caught_T.append(float(line.split(' ')[0]))
                        FF_Position.append([float(line.split(' ')[1]),float(line.split(' ')[2])])
                        FF_believed_position.append([float(line.split(' ')[3]),float(line.split(' ')[4])])
                    else:
                        having_inf_in_line = True 
                        FF_caught_T.append(99999)
                        FF_Position.append([float(line.split(' ')[1]),float(line.split(' ')[2])])
                        FF_believed_position.append([9999, 9999])
                else:
                    # if isinstance(line.split(' ')[0], float) & isinstance(line.split(' ')[1], float):
                    #     FF_flash_T.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
                    try:
                        FF_flash_T.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
                    except:
                        1
                        
            FF_flash_T = np.array(FF_flash_T)
            seperated_ff = np.digitize(FF_flash_T.T[0], FF_caught_T)
            if not having_inf_in_line: 
                unique_ff = np.unique(seperated_ff)[:-1]
            else:
                unique_ff = np.unique(seperated_ff)
            for j in unique_ff:
                
                ff_information.append({'ff_index': ffname_index, 'ff_caught_T': FF_caught_T[j], 'ff_real_position': np.array(FF_Position[j]),
                                  'ff_believed_position': np.array(FF_believed_position[j]),
                                      'ff_flash_T': FF_flash_T[seperated_ff == j]})  
                
                ffname_index = ffname_index+1    
            # incorporate the last ff in this array of data that was not caught

        accurate_start_time, accurate_end_time = find_start_and_accurate_end_time(self.folder_path)

        # get monkey data
        self.processed_data_folder_path = self.folder_path.replace('raw_monkey_data', 'processed_data')
        monkey_information_path = os.path.join(self.processed_data_folder_path, 'monkey_information.csv')
        if exists(monkey_information_path) & monkey_information_exists_OK:
            print("Retrieved monkey_information")
            monkey_information = pd.read_csv(monkey_information_path).drop(["Unnamed: 0"], axis=1)
        else:
            #Monkey_X = []
            #Monkey_Y = []
            Monkey_Position_T = []
            
            for line in log_content[monkeyLineNum+1:]:
                #Monkey_X.append(float(line.split(' ')[0]))
                #Monkey_Y.append(float(line.split(' ')[1]))
                Monkey_Position_T.append(float(line.split(' ')[2]))
            monkey_information = {'monkey_t': np.array(Monkey_Position_T)}                
            #monkey_information = {'monkey_x': np.array(Monkey_X), 'monkey_y': np.array(Monkey_Y), 'monkey_t': np.array(Monkey_Position_T)}
            monkey_information = pd.DataFrame(monkey_information)
            monkey_information = trimming_monkey_information(monkey_information, accurate_start_time, accurate_end_time)
            monkey_information = add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path = self.folder_path,
                                                                         variables = ['LDy', 'LDz', 'RDy', 'RDz', 'MonkeyX', 'MonkeyY', 'LateralV', 'ForwardV', 'AngularV'])
            monkey_information['monkey_speed'] = LA.norm(monkey_information[['LateralV', 'ForwardV']].values, axis=1)
            #monkey_information['monkey_speed'] = gaussian_filter1d(monkey_information['monkey_speed'], 1)

            monkey_information = monkey_information.rename(columns={'MonkeyX': 'monkey_x', 'MonkeyY': 'monkey_y', 'AngularV': 'monkey_dw'})
            monkey_information['monkey_dw'] = monkey_information['monkey_dw'] * pi/180
            #monkey_information['monkey_dw'] = gaussian_filter1d(monkey_information['monkey_dw'], 1)


        ff_caught_T_new, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_sorted, \
                ff_flash_end_sorted = unpack_ff_information_of_monkey(ff_information, accurate_end_time=accurate_end_time, raw_data_folder_path = self.folder_path)


        monkey_information = process_monkey_information_after_retrieval(monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)

        return monkey_information, ff_caught_T_new, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted


def process_monkey_information_after_retrieval(monkey_information, min_distance_to_calculate_angle=5):
    monkey_information = take_out_erroneous_information_from_monkey_information(monkey_information)
    monkey_information = add_columns_to_monkey_information(monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)
    monkey_information = monkey_information.reset_index(drop=True)
    monkey_information = get_derivative_of_a_column(monkey_information, column_name='monkey_dw', derivative_name='monkey_ddw')
    monkey_information = get_derivative_of_a_column(monkey_information, column_name='monkey_speed', derivative_name='monkey_ddv')
    add_more_columns_to_monkey_information(monkey_information)

    # assign "stop_id" to each stop, with each whether_new_distinct_stop==True marking a new stop id
    monkey_information['stop_id'] = monkey_information['whether_new_distinct_stop'].cumsum() - 1
    monkey_information.loc[monkey_information['monkey_speeddummy'] == 1, 'stop_id'] = np.nan

    return monkey_information


def add_more_columns_to_monkey_information(monkey_information):
    monkey_information['monkey_angle'] = monkey_information['monkey_angles']
    monkey_information['time'] = monkey_information['monkey_t']
    monkey_information['point_index'] = monkey_information.index.values.astype(int)
    monkey_information['dt'] = monkey_information['time'].shift(-1) - monkey_information['time']
    monkey_information['dt'] = monkey_information['dt'].ffill()
    add_delta_distance_and_cum_distance_to_monkey_information(monkey_information)
    return monkey_information


def add_delta_distance_and_cum_distance_to_monkey_information(monkey_information):
    monkey_x = monkey_information['monkey_x']
    monkey_y = monkey_information['monkey_y']

    monkey_information['delta_distance'] = np.sqrt((monkey_x.diff())**2 + (monkey_y.diff())**2)
    monkey_information['delta_distance'] = monkey_information['delta_distance'].fillna(0)

    monkey_information['cum_distance'] = np.cumsum(monkey_information['delta_distance'])



def take_out_erroneous_information_from_monkey_information(monkey_information):
    # find delta_position
    delta_time = np.diff(monkey_information['monkey_t'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))

    # for the points where delta_position is greater than the threshold, max(10, np.max(delta_time)*200),
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5) # times 1.5 to make the criterion slightly looser
    above_ceiling_point_index = np.where(delta_position > ceiling_of_delta_position)[0]

    # inspect if the new point is away from the boundary
    corr_monkey_info = monkey_information.iloc[above_ceiling_point_index+1].copy()
    corr_monkey_info['distances_from_center'] = np.linalg.norm(corr_monkey_info[['monkey_x', 'monkey_y']].values, axis=1)

    # if so, delete this point
    abnormal_point_index = corr_monkey_info[corr_monkey_info['distances_from_center'] < 1000 - ceiling_of_delta_position].index
    monkey_information = monkey_information.drop(abnormal_point_index)

    # Since sometimes the erroneous points can occur several in the row, we shall repeat the procedure, until the points all come back to normal.
    # However, if the process has been repeated for more than 5 times, then we'll raise a warning.
    procedure_counter = 1

    while len(abnormal_point_index) > 0:
        # repeat the procedure above
        delta_x = np.diff(monkey_information['monkey_x'])
        delta_y = np.diff(monkey_information['monkey_y'])
        delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
        above_ceiling_point_index = np.where(delta_position > ceiling_of_delta_position)[0]
        corr_monkey_info = monkey_information.iloc[above_ceiling_point_index+1].copy()
        corr_monkey_info['distances_from_center'] = np.linalg.norm(corr_monkey_info[['monkey_x', 'monkey_y']].values, axis=1)
        abnormal_point_index = corr_monkey_info[corr_monkey_info['distances_from_center'] < 1000 - ceiling_of_delta_position].index
        monkey_information = monkey_information.drop(abnormal_point_index)
        procedure_counter += 1
        if procedure_counter == 5:
            print("Warning: there are still erroneous points in the monkey information after 5 times of correction!")

    if procedure_counter > 5:
        print("The procedure to remove erroneous points was repeated", procedure_counter, "times")

    monkey_information.reset_index(drop=True, inplace=True)
    return monkey_information



def calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points):
    num_points_past = int(np.floor(num_points / 2))
    num_points_future = num_points - num_points_past

    # make sure that the two numbers don't go out of bound
    if i - num_points_past < 0:
        num_points_past = i
        num_points_future = num_points - num_points_past
        if i + num_points_future >= total_points:
            delta_x = 0
            delta_y = 0
            current_delta_position = 0
            return delta_x, delta_y, current_delta_position, num_points_past, num_points_future
    
    if i + num_points_future >= total_points:
        num_points_future = total_points - i - 1
        num_points_past = num_points - num_points_future
        if i - num_points_past < 0:
            delta_x = 0
            delta_y = 0
            current_delta_position = 0
            return delta_x, delta_y, current_delta_position, num_points_past, num_points_future

    delta_x = monkey_information['monkey_x'].iloc[i+num_points_future] - monkey_information['monkey_x'].iloc[i-num_points_past]
    delta_y = monkey_information['monkey_y'].iloc[i+num_points_future] - monkey_information['monkey_y'].iloc[i-num_points_past]
    current_delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    return delta_x, delta_y, current_delta_position, num_points_past, num_points_future



# def calculate_delta_xy_and_current_delta_position(monkey_information, i, num_points_future, num_points_past):
#     delta_x = monkey_information['monkey_x'].iloc[i+num_points_future-1] - monkey_information['monkey_x'].iloc[i-num_points_past-1]
#     delta_y = monkey_information['monkey_y'].iloc[i+num_points_future-1] - monkey_information['monkey_y'].iloc[i-num_points_past-1]
#     current_delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
#     return delta_x, delta_y, current_delta_position



def calculate_monkey_angles(monkey_information, min_distance_to_calculate_angle=5):
    # Add angle of the monkey
    monkey_angles = [pi/2]  # The monkey is at 90 degree angle at the beginning
    list_of_num_points = [0]
    list_of_num_points_past = [0]
    list_of_num_points_future = [0]
    list_of_delta_positions = [0]
    current_angle = pi/2 # This keeps track of the current angle during the iterations
    previous_angle = pi/2
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])


    # Find the time in the data that is closest (right before) the time where we wan to know the monkey's angular position.
    total_points = len(monkey_information['monkey_t'])
    num_points = 1

    for i in range(1, total_points):

        if num_points < 1:
            num_points = 1

        if num_points >= total_points:
            # use the below so that the algorithm will not simply get stuck after num_points exceeds the total number;
            # rather, we give a num_points a chance to come down a little and re-do the calculation again.
            num_points = num_points - min(total_points - 1, 5)


        delta_x, delta_y, current_delta_position, num_points_past, num_points_future = calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        # first, let's make current_delta_position within min_distance_to_calculate_angle so that we can shed off excessive distance
        while (current_delta_position > min_distance_to_calculate_angle) and (num_points > 1):
            num_points -= 1 
            # now distribute the num_points to the past and future and calculate delta position 
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        # then, we make sure that current_delta_position is just above min_distance_to_calculate_angle
        while (current_delta_position <= min_distance_to_calculate_angle) and (num_points < total_points):
            num_points += 1 
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points)
        if current_delta_position < 50:
            # calculate the angle defined by two points
            current_angle = math.atan2(delta_y, delta_x)
        else:
            # Otherwise, most likely the monkey has crossed the boundary and come out at another place; we shall keep the current angle, and not update it
            current_angle = previous_angle
        
        monkey_angles.append(current_angle)
        previous_angle = current_angle
        list_of_num_points.append(num_points)
        list_of_num_points_past.append(num_points_past)
        list_of_num_points_future.append(num_points_future)
        list_of_delta_positions.append(current_delta_position)
    monkey_information['monkey_angles'] = np.array(monkey_angles)
    monkey_information['num_points'] = np.array(list_of_num_points)
    monkey_information['num_points_past'] = np.array(list_of_num_points_past)
    monkey_information['num_points_future'] = np.array(list_of_num_points_future)
    monkey_information['delta_position'] = np.array(list_of_delta_positions)



def add_columns_to_monkey_information(monkey_information, min_distance_to_calculate_angle=5,
                                      speed_threshold_for_distinct_stop=1):
    """
    Get linear speeds, angular speeds, and angles of the monkey based on time, x-coordinates, and y-coordinates.

    Parameters
    ----------
    monkey_information: df
        containing the time, x-coordinates, and y-coordinates of the monkey

    Returns
    -------
    monkey_information: df
        containing more information of the monkey

    """
    monkey_information = monkey_information.copy()
    
    delta_time = np.diff(monkey_information['monkey_t'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5)
    if 'crossing_boundary' not in monkey_information.columns:
        crossing_boundary = np.append(0, (delta_position > ceiling_of_delta_position).astype('int'))
        monkey_information['crossing_boundary'] = crossing_boundary


    # If the monkey's delta_position at one point exceeds 50, we replace it with the previous speed.
    # (This can happen when the monkey reaches the boundary and comes out at another place)
    while np.where(delta_position >= ceiling_of_delta_position)[0].size > 0:
        above_ceiling_point_index = np.where(delta_position>=ceiling_of_delta_position)[0]
        # find the previous speed for all those points
        delta_position_prev = np.append(np.array([0]), delta_position)
        delta_position[above_ceiling_point_index] = delta_position_prev[above_ceiling_point_index]  


    if 'monkey_speed' not in monkey_information.columns:
        monkey_speed = np.divide(delta_position, delta_time)
        monkey_speed = np.append(monkey_speed[0], monkey_speed)
        # use Gaussian filter on monkey_speed
        #monkey_speed = gaussian_filter1d(monkey_speed, 1)
    # and make sure that the monkey_speed does not exceed maximum speed
    monkey_information.loc[ monkey_information['monkey_speed'] > 200, 'monkey_speed'] = 200


    if 'monkey_speeddummy' not in monkey_information.columns:
        monkey_information['monkey_speeddummy'] = ((monkey_information['monkey_speed'] > 0.1) | \
                                                   (np.abs(monkey_information['monkey_dw']) > 0.0035)).astype(int) 
    


    if 'monkey_angles' not in monkey_information.columns:
        calculate_monkey_angles(monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)
        # The Gaussian filter shouldn't be used because there will be a problem when monkey_angle changes from 3.14 to -3.14
        #monkey_information['monkey_angles'] = gaussian_filter1d(monkey_information['monkey_angles'], 1)

    if 'monkey_dw' not in monkey_information.columns:
        # positive dw means the monkey is turning counterclockwise
        delta_angle = np.diff(monkey_information['monkey_angles'])
        delta_angle = np.remainder(delta_angle, 2*pi)
        delta_angle[delta_angle >= pi] = delta_angle[delta_angle >= pi]-2*pi
        monkey_dw = np.divide(delta_angle, delta_time)
        monkey_dw = np.append(monkey_dw[0], monkey_dw)
        monkey_information['monkey_dw'] = monkey_dw
        #monkey_information['monkey_dw'] = gaussian_filter1d(monkey_information['monkey_dw'], 1)

    if 'time_box' not in monkey_information.columns:
        monkey_information['time_box'] = np.arange(len(monkey_information))

    if 'whether_new_distinct_stop' not in monkey_information.columns:
        # then we shall find the distinct stops. The standard is that every two stops should be separated by at least one point 
        # that is than speed_threshold_for_distinct_stop cm/s.
        monkey_information['point_index'] = monkey_information.index
        monkey_df = monkey_information.copy()

        # mark whether the speed is over the threshold
        monkey_df['speed_over_threshold'] = monkey_df['monkey_speed'] > speed_threshold_for_distinct_stop
        # get the cumulative sum of the speed_over_threshold
        monkey_df['cum_speed_over_threshold'] = monkey_df['speed_over_threshold'].cumsum()
        # take out the stop points
        monkey_df = monkey_df[monkey_df['monkey_speeddummy'] == 0].copy()
        # get distinct stops
        monkey_df = monkey_df.groupby('cum_speed_over_threshold').first().reset_index(drop=False)
        # add back the info to monkey_information
        monkey_information['whether_new_distinct_stop'] = False
        monkey_information.loc[monkey_information['point_index'].isin(monkey_df['point_index']), 'whether_new_distinct_stop'] = True

    return monkey_information   



def add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path, variables = ['LDy', 'LDz', 'RDy', 'RDz']):
    monkey_information = monkey_information.copy()
    accurate_start_time, accurate_end_time, signal_df = find_start_and_accurate_end_time(raw_data_folder_path, exists_ok=True, return_signal_df=True)
    signal_t = np.array(signal_df.Time)
    monkey_t = monkey_information['monkey_t'].values
    time_bins = find_time_bins_for_an_array(monkey_t)

    # add time_box to monkey_information
    monkey_information.loc[:, 'time_box'] = np.arange(1, len(monkey_information)+1)
    # group signal_df.time based on intervals in monkey_information['monkey_t'], thus adding the column time_box to signal_df
    signal_df.loc[:, 'time_box'] = np.digitize(signal_t, time_bins)
    # use groupby and then find average for LDy, LDz, RDy, RDz
    variables.append('time_box')
    condensed_signal_df = signal_df[variables]
    condensed_signal_df = condensed_signal_df.groupby('time_box').mean().reset_index(drop=False)

    # Put these info into monkey_information
    monkey_information = monkey_information.merge(condensed_signal_df, how='left', on='time_box')
    return monkey_information





def find_time_bins_for_an_array(array_of_interest):
    # find mid-points of each interval in monkey_information['monkey_t']
    interval_lengths = np.diff(array_of_interest)
    half_interval_lengths = interval_lengths/2
    half_interval_lengths = np.append(half_interval_lengths, half_interval_lengths[-1])
    # find the boundaries of boxes that surround each element of monkey_information['monkey_t']
    time_bins = array_of_interest + half_interval_lengths
    # add the position of the leftmost boundary
    first_box_boundary_position = array_of_interest[0]-half_interval_lengths[0]
    time_bins = np.append(first_box_boundary_position, time_bins)
    return time_bins




def trimming_monkey_information(monkey_information, accurate_start_time, accurate_end_time):
    # Chop off the beginning part and the end part of monkey_information
    monkey_t = monkey_information['monkey_t']
    if monkey_information['monkey_t'][0] < accurate_start_time:
        valid_points = np.where((monkey_t >= accurate_start_time) & (monkey_t <= accurate_end_time))[0]
        monkey_information = monkey_information.iloc[valid_points]

    return monkey_information



def find_start_and_accurate_end_time(raw_data_folder_path, exists_ok=True, return_signal_df=False):
    """
    Parameters
    ----------
    raw_data_folder_path: str
        the folder name of the raw data

    Returns
    -------
    accurate_end_time: num
        the last point of time within accurate juice timestamps


    """
    filepath = os.path.join(raw_data_folder_path, 'start_and_end_of_juice_timestamps.csv')
    if exists(filepath) & exists_ok & (not return_signal_df):
        start_and_end_time = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
        accurate_start_time = start_and_end_time.iloc[0].item()
        accurate_end_time = start_and_end_time.iloc[1].item()
    else:
        Channel_signal_output, marker_list, smr_sampling_rate = smr_extractor(raw_data_folder_path = raw_data_folder_path).extract_data()
        # Considering the first smr file, use marker_list[0], Channel_signal_output[0]
        juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
        Channel_signal_smr1 = Channel_signal_output[0]
        # Set a default value of accurate_end_time in case Channel_signal_smr1 is empty
        accurate_start_time = 0
        accurate_end_time = 99998
        if len(Channel_signal_smr1) > 0:
            Channel_signal_smr1['section'] = np.digitize(Channel_signal_smr1.Time, juice_timestamp) # seperate analog signal by juice timestamps
            # Remove tail of analog data
            Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['section']<  Channel_signal_smr1['section'].unique()[-1]]
            if len(Channel_signal_smr1) > 0:
                # Since there might be very slight difference between the last recorded sampling time and juice_timestamp[-1], we replace the former with the latter
                Channel_signal_smr1.loc[Channel_signal_smr1.index[-1], 'Time'] = juice_timestamp[-1]
                # Remove head of analog data
                Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['Time'] > marker_list[0]['values'][marker_list[0]['labels']==1][0]]
                # monkey_smr_dataframe = Channel_signal_smr1[["Time", "Signal stream 1", "Signal stream 2", "Signal stream 3", "Signal stream 10"]].reset_index(drop=True)
                # monkey_smr_dataframe.columns = ['monkey_t', 'monkey_x', 'monkey_y', 'monkey_speed', 'AngularV']
                # monkey_smr = dict(zip(monkey_smr_dataframe.columns.tolist(), np.array(monkey_smr_dataframe.values.T.tolist())))
                accurate_start_time = Channel_signal_smr1.Time.values[0] # accurate_start_time is where behavioral data has event = 1.
                accurate_end_time = Channel_signal_smr1.Time.values[-1]

                # save them into a csv file
                data = [accurate_start_time, accurate_end_time]
                juice_timestamps = pd.DataFrame(data, columns=['time'])
                filepath = raw_data_folder_path + '/start_and_end_of_juice_timestamps.csv'
                juice_timestamps.to_csv(filepath)

                if return_signal_df:
                    signal_df = Channel_signal_smr1[['LateralV', 'LDy', 'LDz', 'MonkeyX', 'MonkeyY', 'RDy', 'RDz', 'AngularV', 'ForwardV', 'Time', 'section']].copy()
                    for column in signal_df.columns:
                        if column != 'section':
                            signal_df.loc[:,column] = np.array(signal_df.loc[:,column]).astype('float')
                    return accurate_start_time, accurate_end_time, signal_df

    return accurate_start_time, accurate_end_time




def unpack_ff_information_of_monkey(ff_information, accurate_end_time = None, raw_data_folder_path = None):
    """
    Get various useful lists and arrays of ff information from the raw data;
    modified from Ruiyi's codes

    Parameters
    ----------

    ff_information: list
        derived from the raw data; note that it is different from env.ff_information


    Returns
    -------
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    ff_index_sorted: np.array
        containing the sorted indices of the fireflies
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_flash_sorted: np.array
        containing the flashing-on durations of each firefly 
    ff_flash_end_sorted: np.array
        containing the end of each flash-on duration of each firefly

    """
    if accurate_end_time is None:
        if raw_data_folder_path is None:
            raise ValueError("raw_data_folder_path cannot be None if accurate_end_time is None!")
        else:
            accurate_start_time, accurate_end_time = find_start_and_accurate_end_time(raw_data_folder_path)


    ff_index = []
    ff_caught_T = []
    ff_real_position = []
    ff_believed_position = []
    ff_life = []
    ff_flash = []
    ff_flash_end = []  # This is the time that the firefly last stops flash
    for item in ff_information:
        item['Life'] = np.array([item['ff_flash_T'][0][0], item['ff_caught_T']])
        ff_index = np.hstack((ff_index, item['ff_index']))
        ff_caught_T = np.hstack((ff_caught_T, item['ff_caught_T']))
        ff_real_position.append(item['ff_real_position'])
        ff_believed_position.append(item['ff_believed_position'])
        ff_life.append(item['Life'])
        ff_flash_currrent_ff = item['ff_flash_T']
        ff_flash_currrent_ff = ff_flash_currrent_ff[ff_flash_currrent_ff[:, 0] < accurate_end_time]
        ff_flash.append(ff_flash_currrent_ff)
        ff_flash_end.append(item['ff_flash_T'][-1][-1])
    sort_index = np.argsort(ff_caught_T)
    ff_caught_T_new = ff_caught_T[sort_index]
    # Use accurate juice timestamps, ff_caught_T_new for smr1 (so that the time frame is correct)
 
    captured_ffs = np.where((ff_caught_T_new <= min(accurate_end_time, 99999)))
    ff_caught_T_new = ff_caught_T_new[captured_ffs]
    ff_index_sorted = ff_index[sort_index]
    ff_real_position_sorted = np.array(ff_real_position)[sort_index]
    ff_believed_position_sorted = np.array(ff_believed_position)[sort_index][captured_ffs]
    ff_life_sorted = np.array(ff_life)[sort_index]
    ff_flash_sorted = np.array(ff_flash, dtype=object)[sort_index].tolist()
    ff_flash_end_sorted = np.array(ff_flash_end)[sort_index]
    return ff_caught_T_new, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted





def find_monkey_angles_for_signal_df(signal_df, monkey_information):
    '''
    This function is currently not used.
    '''
    monkey_t = monkey_information['monkey_t'].values
    monkey_angles = monkey_information['monkey_angles'].values
    signal_t = signal_df.Time.values


    signal_df['MonkeyAngle'] = None
    # Merge the monkey_angles
    # for every chunk_of_time
    chunk_of_time = 30
    num_chunks = int(np.ceil(signal_t.max()/chunk_of_time))
    signal_t = np.array(signal_df.Time)
    for num in range(num_chunks):
        print(num, 'out of', num_chunks, 'chunks.')
        duration = [num*chunk_of_time, (num+1)*chunk_of_time]
        # signal_subset = signal_df[(signal_df['Time'] >= duration[0]) & (signal_df['Time'] = duration[1])]
        signal_subset_indices = np.where((signal_t >= duration[0]) & (signal_t < duration[1]))
        signal_subset_t = signal_t[signal_subset_indices]

        monkey_indices = np.where((monkey_t >= duration[0]) & (monkey_t < duration[1]))[0]
        monkey_subset_t = monkey_t[monkey_indices]
        monkey_subset_angles = monkey_angles[monkey_indices]
          
        # for each time point in subset_df, find the corresponding closest time point in monkey_subset_t
        signal_subset_t_expanded = np.repeat(signal_subset_t.reshape(-1,1), repeats=len(monkey_subset_t), axis=1)
        print(signal_subset_t_expanded.shape)
        abs_dif = np.abs(signal_subset_t_expanded - monkey_subset_t)
        closest_point_index = abs_dif.argmin(axis=1)
        closest_point_angles = monkey_subset_angles[closest_point_index]

        # put the angles back to signal_df
        signal_df.loc[:, 'MonkeyAngle'].iloc[signal_subset_indices] = closest_point_angles

    return signal_df





def make_info_of_monkey(monkey_information, ff_information, ff_dataframe,  accurate_end_time = None, raw_data_folder_path = None):
    accurate_start_time, accurate_end_time = find_start_and_accurate_end_time(raw_data_folder_path)
    ff_caught_T_new, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted\
        = unpack_ff_information_of_monkey(ff_information, accurate_end_time=accurate_end_time, raw_data_folder_path=raw_data_folder_path)
    monkey_information = add_columns_to_monkey_information(monkey_information)
    caught_ff_num = len(ff_caught_T_new)
    cluster_around_target_trials, used_cluster_trials, cluster_around_target_indices, cluster_around_target_positions = pattern_by_trials.cluster_around_target_func(ff_dataframe, caught_ff_num, ff_caught_T_new, ff_real_position_sorted)

    info_of_monkey = {
            "monkey_information": monkey_information,
            "ff_dataframe": ff_dataframe,
            "ff_caught_T_new": ff_caught_T_new,
            "ff_real_position_sorted": ff_real_position_sorted,
            "ff_believed_position_sorted": ff_believed_position_sorted,
            "ff_life_sorted": ff_life_sorted,
            "ff_flash_sorted": ff_flash_sorted,
            "ff_flash_end_sorted": ff_flash_end_sorted,
            "cluster_around_target_indices": cluster_around_target_indices}

    return info_of_monkey    



def get_derivative_of_a_column(monkey_information, column_name, derivative_name):
    dvar = np.diff(monkey_information[column_name])
    dvar1 = np.append(dvar[0], dvar)
    dvar2 = np.append(dvar, dvar[-1])
    avg_dvar = (dvar1 + dvar2)/2
    monkey_information[derivative_name] = avg_dvar
    return monkey_information

def turn_array_into_df(self, array, corresponding_t):
    info_num = [int(num) for num in array]
    info_dict = {'t': corresponding_t.tolist(), 'num': info_num}
    info_dataframe = pd.DataFrame(info_dict)
    info_dataframe['trial']=np.searchsorted(ff_caught_T_new, corresponding_t)
    return info_dataframe



    

