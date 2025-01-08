from data_wrangling import process_monkey_information, retrieve_raw_data

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



def find_smr_markers_start_and_end_time(raw_data_folder_path, exists_ok=True, save_start_and_end_time=True):
    """
    Parameters
    ----------
    raw_data_folder_path: str
        the folder name of the raw data

    Returns
    -------
    smr_markers_end_time: num
        the last point of time within accurate juice timestamps


    """
    metadata_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'metadata')
    filepath = os.path.join(metadata_folder_path, 'start_and_end_time_of_smr_markers.csv')
    if exists(filepath) & exists_ok:
        start_and_end_time = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
        smr_markers_start_time = start_and_end_time.iloc[0].item()
        smr_markers_end_time = start_and_end_time.iloc[1].item()
    else:
        Channel_signal_output, marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(raw_data_folder_path)
        juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
        smr_markers_start_time = marker_list[0]['values'][marker_list[0]['labels']==1][0]
        smr_markers_end_time =juice_timestamp[-1]
        if save_start_and_end_time:
            start_and_end_time = pd.DataFrame([smr_markers_start_time, smr_markers_end_time], columns=['time'])
            start_and_end_time.to_csv(os.path.join(metadata_folder_path, 'start_and_end_time_of_smr_markers.csv'))
            print("Saved start and end time of juice timestamps")
    return smr_markers_start_time, smr_markers_end_time
    

def make_signal_df(raw_data_folder_path):
    Channel_signal_output, marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(raw_data_folder_path)
    juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
    smr_markers_start_time = marker_list[0]['values'][marker_list[0]['labels']==1][0]
    signal_df = retrieve_raw_data.get_signal_df(Channel_signal_output, juice_timestamp, smr_markers_start_time)
    return signal_df


def get_closest_smr_t_to_txt_t(txt_t, smr_t):
    # Compute the absolute differences between each element in txt_t and all elements in smr_t
    differences = np.abs(txt_t[:, np.newaxis] - smr_t)

    # Find the indices of the minimum values along the smr_t axis
    closest_indices = np.argmin(differences, axis=1)

    # Use the indices to get the closest points in smr_t
    closest_smr_t_to_txt_t = smr_t[closest_indices]

    return closest_smr_t_to_txt_t


def make_ff_caught_times_df(neural_t, smr_t, txt_t, neural_events_start_time, smr_markers_start_time):
    # make df to compare the capture times between the files

    min_rows = min(len(txt_t), len(smr_t))
    df = pd.DataFrame({'smr_t': smr_t[:min_rows],
                        'txt_t': txt_t[:min_rows],
                        })
    
    closest_smr_t_to_txt_t = get_closest_smr_t_to_txt_t(txt_t, smr_t)
    df['closest_smr_t_to_txt_t'] = closest_smr_t_to_txt_t[:min_rows]

    # neural_t might have fewer rows than txt_t and smr_t
    df['neural_t'] = np.nan
    df['neural_t'][:len(neural_t)] = neural_t[:min_rows]

    # Note: neural_t_adj means that the offset is based on label==4; 
    # neural_t_adj_2 means that the offset is based on label==1; 
    df['neural_t_adj'] = df['neural_t'] - df['neural_t'].iloc[0] + df['smr_t'].iloc[0]
    df['neural_t_adj_2'] = df['neural_t'] - neural_events_start_time + smr_markers_start_time
    # Note: txt_t_adj means that the offset is based on the difference in first capture time between txt and smr; 
    # txt_t_adj_2 means that the offset is based on the the median of the differences between capture times of txt and closest smr
    df['txt_t_adj'] = df['txt_t'] - df['txt_t'].iloc[0] + df['smr_t'].iloc[0]
    df['txt_t_adj_2'] = df['txt_t'] - np.median(df['txt_t'] - df['closest_smr_t_to_txt_t'])

    df['diff_neural_adj_smr'] = df['neural_t_adj'] - df['smr_t']
    df['diff_neural_adj_2_smr'] = df['neural_t_adj_2'] - df['smr_t']
    
    df['diff_txt_smr_closest'] = df['txt_t'] - df['closest_smr_t_to_txt_t']
    df['diff_txt_adj_smr_closest'] = df['txt_t_adj'] - df['closest_smr_t_to_txt_t']
    df['diff_txt_adj_2_smr_closest'] = df['txt_t_adj_2'] - df['closest_smr_t_to_txt_t']

    df['diff_txt_adj_neural_adj'] = df['txt_t_adj'] - df['neural_t_adj']
    df['diff_txt_adj_neural_adj_2'] = df['txt_t_adj'] - df['neural_t_adj_2']
    df['diff_txt_adj_2_neural_adj'] = df['txt_t_adj_2'] - df['neural_t_adj']

    ff_caught_times_df = df

    return ff_caught_times_df
