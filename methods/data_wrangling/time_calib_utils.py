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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def find_smr_markers_start_and_end_time(raw_data_folder_path, ff_caught_T_sorted=None, exists_ok=True, save_start_and_end_time=True):
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
    time_calibration_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')
    filepath = os.path.join(time_calibration_folder_path, 'adj_smr_markers_start_and_end_time.csv')
    if exists(filepath) & exists_ok:
        start_and_end_time = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
        smr_markers_start_time = start_and_end_time.iloc[0].item()
        smr_markers_end_time = start_and_end_time.iloc[1].item()
    else:
        channel_signal_output, marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(raw_data_folder_path)
        if ff_caught_T_sorted is None:
            ff_caught_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, \
                ff_flash_end_sorted = retrieve_raw_data.make_or_retrieve_ff_info_from_txt_data(raw_data_folder_path)
        smr_markers_start_time, smr_markers_end_time = _get_adjusted_smr_markers_start_time_and_end_time(marker_list, ff_caught_T_sorted)
        if save_start_and_end_time:
            start_and_end_time = pd.DataFrame([smr_markers_start_time, smr_markers_end_time], columns=['time'])
            start_and_end_time.to_csv(filepath)
            print(f"Saved start and end time of juice timestamps at {filepath}")
    return smr_markers_start_time, smr_markers_end_time

def _get_adjusted_smr_markers_start_time_and_end_time(marker_list, ff_caught_T_sorted):
    juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
    unadj_smr_markers_start_time = marker_list[0]['values'][marker_list[0]['labels']==1][0]
    unadj_smr_markers_end_time = juice_timestamp[-1]
    smr_t_raw = marker_list[0]['values'][marker_list[0]['labels'] == 4]
    df, _, txt_smr_offset_2 = make_txt_and_smr_df(smr_t_raw, ff_caught_T_sorted)
    smr_markers_start_time = unadj_smr_markers_start_time + txt_smr_offset_2
    smr_markers_end_time = unadj_smr_markers_end_time + txt_smr_offset_2
    return smr_markers_start_time, smr_markers_end_time

def make_temp_txt_and_neural(raw_data_folder_path, ff_caught_T_sorted):   
    time_calibration_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')
    neural_event_time = pd.read_csv(os.path.join(time_calibration_folder_path, 'neural_event_time.txt'))
    neural_t_raw = neural_event_time.loc[neural_event_time['label'] == 4, 'time'].values
    txt_t = ff_caught_T_sorted.copy()

    df = pd.DataFrame({'txt_t': txt_t})
    offset_neural_txt = neural_t_raw[0] - txt_t[0]
    neural_t_adj = neural_t_raw - offset_neural_txt
    df = add_closest_neural_t_adj_to_txt_t(df, neural_t_adj, txt_t, new_column_name='neural_t')
    df['neural_t_raw_ext'] = df['neural_t'] + offset_neural_txt
    df['diff_txt_neural_raw'] = df['txt_t'] - df['neural_t_raw_ext'] # neural adjusted by first txt capture
    # drop NA
    df = df.dropna()
    return df


def make_txt_and_smr_df(smr_t_raw, txt_t):
    min_rows = min(len(txt_t), len(smr_t_raw))
    txt_t = txt_t[:min_rows]
    smr_t_raw = smr_t_raw[:min_rows]
    
    # adjust smr_t based on the difference in first capture time between txt and smr
    txt_smr_offset = txt_t[0] - smr_t_raw[0]
    smr_t_adj = smr_t_raw + txt_smr_offset

    smr_t_adj_ext = get_closest_t_to_txt_t(txt_t[:len(smr_t_adj)], smr_t_adj)  
    smr_t_raw_ext = smr_t_adj_ext - txt_smr_offset

    # adjust smr_t based on the the median of the differences between capture times of txt and closest smr
    txt_smr_offset_2 = np.median(txt_t - smr_t_raw_ext)
    smr_t_adj_ext_2 = smr_t_raw_ext + txt_smr_offset_2
      
    
    df = pd.DataFrame({'txt_t': txt_t, 
                       'smr_t_raw': smr_t_raw_ext,
                        'smr_t': smr_t_adj_ext, 
                        'smr_t_2': smr_t_adj_ext_2, 
                        })
    return df, txt_smr_offset, txt_smr_offset_2


def make_adjusted_ff_caught_times_df(neural_t_raw, smr_t_raw, txt_t, neural_events_start_time, smr_markers_start_time):
    # make df to compare the capture times between the files

    df, txt_smr_offset, txt_smr_offset_2 = make_txt_and_smr_df(smr_t_raw, txt_t)

    # adjust neural_t based on the difference in first capture time between txt and smr
    offset_neural_txt = neural_t_raw[0] - txt_t[0]
    neural_t_adj = neural_t_raw - offset_neural_txt
    # adjust neural_t based first on the difference in time of label==1 between neural_t and smr_t, and then on the median difference between txt_t and smr_t
    neural_t_adj_2 = neural_t_raw - neural_events_start_time + smr_markers_start_time + txt_smr_offset_2
    # adjust neural_t based first on the difference in time of label==1 between neural_t and smr_t, and then on the difference in first capture time between txt and smr
    neural_t_adj_3 = neural_t_raw - neural_events_start_time + smr_markers_start_time + txt_smr_offset
    # adjust neural_t based only on label==1 (or first label==4, for sessions without label==1)
    neural_t_adj_4 = neural_t_raw - neural_events_start_time + smr_markers_start_time

    df = add_closest_neural_t_adj_to_txt_t(df, neural_t_adj, txt_t, new_column_name='neural_t')
    df = add_closest_neural_t_adj_to_txt_t(df, neural_t_adj_2, txt_t, new_column_name='neural_t_2')
    df = add_closest_neural_t_adj_to_txt_t(df, neural_t_adj_3, txt_t, new_column_name='neural_t_3')
    df = add_closest_neural_t_adj_to_txt_t(df, neural_t_adj_4, txt_t, new_column_name='neural_t_4')

    df['neural_t_raw_ext'] = df['neural_t'] + offset_neural_txt
    calculate_offsets_in_ff_capture_time_between_data(df)
    ff_caught_times_df = df

    return ff_caught_times_df


def calculate_offsets_in_ff_capture_time_between_data(df):
    df['diff_txt_neural_raw'] = df['txt_t'] - df['neural_t_raw_ext']
    df['diff_txt_neural'] = df['txt_t'] - df['neural_t'] # neural adjusted by first txt capture
    df['diff_txt_neural_2'] = df['txt_t'] - df['neural_t_2'] # neural first adjusted to smr by label==1, then adjusted to txt by median of time difference between txt and smr
    df['diff_txt_neural_3'] = df['txt_t'] - df['neural_t_3'] # neural first adjusted to smr by label==1, then adjusted to txt by difference between first txt capture and first smr capture
    df['diff_txt_neural_4'] = df['txt_t'] - df['neural_t_4'] # neural adjusted only by label=1

    df['diff_txt_smr_raw'] = df['txt_t'] - df['smr_t_raw']
    df['diff_txt_smr'] = df['txt_t'] - df['smr_t'] # smr adjusted by first txt capture
    df['diff_txt_smr_2'] = df['txt_t'] - df['smr_t_2'] # smr adjusted by median of time difference

    df['diff_neural_smr'] = df['neural_t'] - df['smr_t'] # both adjusted by first txt capture
    df['diff_neural_2_smr_2'] = df['neural_t_2'] - df['smr_t_2'] # both adjusted by the time of label=1 (and the offset compared to txt canceled out)


def get_closest_t_to_txt_t(txt_t, smr_t):
    # Compute the absolute differences between each element in txt_t and all elements in smr_t
    differences = np.abs(txt_t[:, np.newaxis] - smr_t)

    # Find the indices of the minimum values along the smr_t axis
    closest_indices = np.argmin(differences, axis=1)

    # Use the indices to get the closest points in smr_t
    smr_t_adj_ext = smr_t[closest_indices]

    return smr_t_adj_ext


def add_closest_neural_t_adj_to_txt_t(df, neural_t_adj, txt_t, new_column_name='neural_t'):

    closest_neural_t_adj_to_txt_t = get_closest_t_to_txt_t(txt_t[:len(neural_t_adj)], neural_t_adj)

    # make sure that the length of closest_neural_t_adj_to_txt_t is the same as the length of df
    if len(closest_neural_t_adj_to_txt_t) < len(df):
        # append closest_neural_t_adj_to_txt_t with na values to match the length of df
        closest_neural_t_adj_to_txt_t = np.append(closest_neural_t_adj_to_txt_t, np.nan * np.ones(len(df) - len(closest_neural_t_adj_to_txt_t)))
    else:
        closest_neural_t_adj_to_txt_t = closest_neural_t_adj_to_txt_t[:len(df)]

    # Note: neural_t means that the offset is based on label==4, while neural_t_2 means that the offset is based on label==1; 
    df[new_column_name] = closest_neural_t_adj_to_txt_t
    return df


def make_or_retrieve_txt_smr_t_diff_via_xy_df(raw_data_folder_path, exists_ok=True):
    time_calibration_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')
    os.makedirs(time_calibration_folder_path, exist_ok=True)
    filename = 'txt_smr_t_diff_via_xy.csv'
    file_path = os.path.join(time_calibration_folder_path, filename)
    if exists(file_path) & exists_ok:
        txt_smr_t_diff_via_xy_df = pd.read_csv(file_path).drop(["Unnamed: 0"], axis=1)
        print(f'Retrieved {filename} from {file_path}')
    else:
        raw_monkey_information = retrieve_raw_data.get_raw_monkey_information_from_txt_data(raw_data_folder_path)
        smr_markers_start_time, smr_markers_end_time = find_smr_markers_start_and_end_time(raw_data_folder_path)
        monkey_information = retrieve_raw_data._trim_monkey_information(raw_monkey_information, smr_markers_start_time, smr_markers_end_time)
        process_monkey_information.add_monkey_speed_column(monkey_information)
        raw_signal_df = process_monkey_information.get_raw_signal_df(raw_data_folder_path)
        txt_smr_t_diff_via_xy_df = find_txt_smr_t_diff_via_xy_df(monkey_information, raw_signal_df, n_points=1000)
        txt_smr_t_diff_via_xy_df.to_csv(file_path)
        print(f'Saved {filename} to {file_path}')
    return txt_smr_t_diff_via_xy_df


def make_or_retrieve_txt_smr_t_linreg_df(raw_data_folder_path, ceiling_of_min_distance=2, exists_ok=True):
    time_calibration_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')
    filename = 'txt_smr_t_linreg.csv'
    file_path = os.path.join(time_calibration_folder_path, filename)
    if exists(file_path) & exists_ok:
        txt_smr_t_linreg_df = pd.read_csv(file_path).drop(["Unnamed: 0"], axis=1)
        print(f'Retrieved {filename} from {file_path}')
    else:
        txt_smr_t_diff_via_xy_df = make_or_retrieve_txt_smr_t_diff_via_xy_df(raw_data_folder_path)
        df_sub = clean_txt_smr_t_diff_via_xy_df_based_on_min_distance(txt_smr_t_diff_via_xy_df, ceiling_of_min_distance=ceiling_of_min_distance)
        _, txt_smr_t_linreg_df = get_linear_regression(df_sub['time'].values, df_sub['time_offset'].values, make_plot=False)
        txt_smr_t_linreg_df.to_csv(file_path)
        print(f'Saved {filename} to {file_path}')
    return txt_smr_t_linreg_df


def calibrate_smr_t(signal_df, txt_smr_t_linreg_df):
    signal_df['time'] = signal_df['time'] + txt_smr_t_linreg_df['intercept'].item() + signal_df['time'] * txt_smr_t_linreg_df['slope'].item()
    return signal_df

def make_or_retrieve_txt_neural_t_linreg_df(raw_data_folder_path, ff_caught_T_sorted, exists_ok=True, show_plot=False):
    time_calibration_folder_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')
    filename = 'txt_neural_t_linreg.csv'
    file_path = os.path.join(time_calibration_folder_path, filename)
    if exists(file_path) & exists_ok & (not show_plot):
        txt_neural_t_linreg_df = pd.read_csv(file_path).drop(["Unnamed: 0"], axis=1)
        print(f'Retrieved {filename} from {file_path}')
    else:
        temp_smr_and_neural = make_temp_txt_and_neural(raw_data_folder_path, ff_caught_T_sorted)
        _, txt_neural_t_linreg_df = get_linear_regression(temp_smr_and_neural['neural_t_raw_ext'].values, temp_smr_and_neural['diff_txt_neural_raw'].values,
                                                        label='adj by 1st txt t', make_plot=show_plot)
        if show_plot:
            plt.show()
        txt_neural_t_linreg_df.to_csv(file_path)
        print(f'Saved {filename} to {file_path}')
    print(txt_neural_t_linreg_df)
    return txt_neural_t_linreg_df


def find_txt_smr_t_diff_via_xy_df(raw_monkey_information, signal_df, n_points=1000):
    # This function finds the time offset between txt and smr based on the xy positions of the monkey

    # limit raw_monkey_information to the same duration as channel_signal_smr, and also truncate off the first 50s and last 50s for increased accuracy
    txt_sub = raw_monkey_information[raw_monkey_information['time'].between(signal_df['time'].iloc[0] + 50, signal_df['time'].iloc[-1] - 50)].copy()
    # take out points where monkey speed is above 10 cm/s (because otherwise when finding the closest position, there can be too much noise)
    txt_sub = txt_sub[txt_sub['monkey_speed'] > 50].copy()

    # sample n_points at nearly equal interval based on positional index
    n_points_total = len(txt_sub['point_index'].unique())
    sampled_pos_idx = np.linspace(1, n_points_total-1, n_points).astype(int)
    list_of_point_index = []
    list_of_txt_smr_offset = []
    list_of_min_distance = []
    list_of_time = []
    for i in range(len(sampled_pos_idx)):
        if i % 100 == 0:
            print(f'{i} out of {len(sampled_pos_idx)} sampled points processed')
        # get one point in raw_monkey_information, and take out a window of 1s around it in smr. Then find the smr point that's closest in distance to raw_monkey_information, and record the time offset
        txt_row = txt_sub.iloc[sampled_pos_idx[i]]
        smr_sub = signal_df[signal_df['time'].between(txt_row['time'] - 0.5, txt_row['time'] + 0.5)].copy()
        smr_sub['distance'] = np.sqrt((smr_sub['MonkeyX'] - txt_row['monkey_x'])**2 + (smr_sub['MonkeyY'] - txt_row['monkey_y'])**2)
        closest_smr_row = smr_sub.loc[smr_sub['distance'].idxmin()]
        txt_smr_offset = txt_row['time'] - closest_smr_row['time']
        list_of_point_index.append(txt_row['point_index'])
        list_of_txt_smr_offset.append(txt_smr_offset)
        list_of_min_distance.append(closest_smr_row['distance'])
        list_of_time.append(closest_smr_row['time'])

    list_of_point_index = np.array(list_of_point_index)
    list_of_txt_smr_offset = np.array(list_of_txt_smr_offset)
    list_of_min_distance = np.array(list_of_min_distance)

    txt_smr_t_diff_via_xy_df = pd.DataFrame({'point_index': list_of_point_index, 
                                          'time_offset': list_of_txt_smr_offset, 
                                          'min_distance': list_of_min_distance,
                                          'time': list_of_time
                                          })

    return txt_smr_t_diff_via_xy_df


def clean_txt_smr_t_diff_via_xy_df_based_on_min_distance(txt_smr_t_diff_via_xy_df, ceiling_of_min_distance=1):
    original_length = len(txt_smr_t_diff_via_xy_df)
    # take out subset of txt_smr_t_diff_via_xy_df where min_distance is less than 1
    txt_smr_t_diff_via_xy_df = txt_smr_t_diff_via_xy_df[txt_smr_t_diff_via_xy_df['min_distance'] < ceiling_of_min_distance].copy()
    # calculate what percentage is taken off
    perc_taken_off = (original_length - len(txt_smr_t_diff_via_xy_df)) / original_length * 100
    print(f'When making txt_smr_t_diff_via_xy_df, {perc_taken_off:.2f}% of points were taken off because min_distance is greater than {ceiling_of_min_distance} cm'
          f'{(original_length - len(txt_smr_t_diff_via_xy_df))} out of {original_length}')
    return txt_smr_t_diff_via_xy_df


def calibrate_neural_data_time(spike_times_in_s, raw_data_folder_path, ff_caught_T_sorted, show_plot=False):
    txt_neural_t_linreg_df = make_or_retrieve_txt_neural_t_linreg_df(raw_data_folder_path, ff_caught_T_sorted,
                                                                                    show_plot=show_plot)
    spike_times_in_s = spike_times_in_s + txt_neural_t_linreg_df['intercept'].item() + spike_times_in_s * txt_neural_t_linreg_df['slope'].item()
    return spike_times_in_s


def remove_outliers_func(x, y):
    # Identify and remove outliers using the IQR method
    mask = find_non_outliers_based_on_IQR(y)
    return x[mask], y[mask]

def find_non_outliers_based_on_IQR(array):
    q1, q3 = np.percentile(array, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (array >= lower_bound) & (array <= upper_bound)
    return mask

def get_linear_regression(list_of_time, list_of_time_offsets, ax=None, color='blue', label='Linear Regression',
                          remove_outliers=True, make_plot=True):

    if remove_outliers:
        cleaned_time, cleaned_time_offsets = remove_outliers_func(list_of_time, list_of_time_offsets)
    else:
        cleaned_time, cleaned_time_offsets = list_of_time, list_of_time_offsets

    # Perform linear regression on cleaned data
    slope, intercept, r_value, p_value, std_err = linregress(cleaned_time, cleaned_time_offsets)

    # Also put the stat into a df
    stat_df = pd.DataFrame({'name':[label], 'slope': [slope], 'intercept': [intercept], 
                            'r_value': [r_value], 'p_value': [p_value], 'std_err': [std_err],
                            'slope x time': [slope * (max(cleaned_time) - min(cleaned_time))],
                            'sample_size': [len(cleaned_time)]
                            })
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots()
            
        # Scatter plot of cleaned data
        ax.scatter(cleaned_time, cleaned_time_offsets, s=2, alpha=0.5, color=color)

        # Plot linear regression line
        ax.plot(cleaned_time, intercept + slope * cleaned_time, color=color, alpha=0.5, label=label)

        # Labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Time offset')
        ax.set_title('Linear Regression after Removing Outliers')
        ax.legend()

    return ax, stat_df