from data_wrangling import time_calib_utils

import os
import re
import os
import os.path
import neo
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def extract_smr_data(raw_data_folder_path):

    smr_files_names = [file for file in os.listdir(
        raw_data_folder_path) if 'smr' in file]
    smr_file_paths = [os.path.join(raw_data_folder_path, smr_files_names[i])
                      for i, value in enumerate(smr_files_names)]
    channel_signal_output = []
    marker_list = []  # used to store information about time of juice rewards
    smr_sampling_rate = None

    # loop 2 files one by one
    for index, file_name in enumerate(smr_file_paths):
        seg_reader = neo.io.Spike2IO(
            # read out segments
            filename=file_name, try_signal_grouping=False).read_segment()

        # get sampling rate, only need to get it once
        if index == 0:
            smr_sampling_rate = seg_reader.analogsignals[0].sampling_rate

        # in case analog channels have different shape
        analog_length = min([i.size for i in seg_reader.analogsignals])

        Channel_index = []  # create an empty list to store channel names
        # Get channel data from all channels except the last one
        # -1 indicates we discard 'Raw' channel
        for C_index, C_data in enumerate(seg_reader.analogsignals[:-1]):
            # See how many channels are contained in each element of the list
            shape = seg_reader.analogsignals[C_index].shape[1]
            # append arrays of channel data into a big array
            if C_index == 0:
                Channel_signal = C_data.as_array()[:analog_length,]
            else:
                Channel_signal = np.append(Channel_signal, C_data.as_array()[
                                           :analog_length,], axis=1)
            # append the channel name
            for i in range(shape):
                Channel_index.append(
                    str(seg_reader.analogsignals[C_index].name))

        # get time stamps and put in Channel_signal
        time_stamps_array = np.asarray(
            seg_reader.analogsignals[0].times[:analog_length,]).reshape(analog_length, 1)
        Channel_signal = np.append(Channel_signal, time_stamps_array, axis=1)
        Channel_index.append('Time')

        channel_signal_output.append(pd.DataFrame(
            Channel_signal, columns=Channel_index))

        # Find the time juice rewards
        marker_channel_index = [index for index, value in enumerate(
            seg_reader.events) if value.name == 'marker'][0]
        marker = {'labels': seg_reader.events[marker_channel_index].get_labels().astype('int'),
                  # get the time of the juice rewards
                  'values': seg_reader.events[marker_channel_index].as_array()
                  }  # arrange labels and values in a dict
        marker_list.append(marker)

    return channel_signal_output, marker_list, smr_sampling_rate


def get_raw_monkey_information_from_txt_data(raw_data_folder_path):
    txt_files_names = [file for file in os.listdir(
        raw_data_folder_path) if ('txt' in file) and ('s' in file)]
    txt_file_path = os.path.join(raw_data_folder_path, txt_files_names[0])

    with open(txt_file_path, 'r', encoding='UTF-8') as content:
        log_content = content.readlines()

    ffLinenumberList, monkeyLineNum = _take_out_ff_and_monkey_line_numbers(
        log_content)

    Monkey_Position_T = []
    Monkey_X = []
    Monkey_Y = []

    for line in log_content[monkeyLineNum+1:]:
        Monkey_X.append(float(line.split(' ')[0]))
        Monkey_Y.append(float(line.split(' ')[1]))
        Monkey_Position_T.append(float(line.split(' ')[2]))
    raw_monkey_information = {'monkey_x': np.array(Monkey_X), 'monkey_y': np.array(
        Monkey_Y), 'time': np.array(Monkey_Position_T)}
    raw_monkey_information = pd.DataFrame(raw_monkey_information)
    return raw_monkey_information


def get_trimmed_monkey_information(raw_data_folder_path):
    raw_monkey_information = get_raw_monkey_information_from_txt_data(
        raw_data_folder_path)
    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
        raw_data_folder_path)
    monkey_information = _trim_monkey_information(
        raw_monkey_information, smr_markers_start_time, smr_markers_end_time)
    return monkey_information


def _trim_monkey_information(raw_monkey_information, smr_markers_start_time, smr_markers_end_time):
    # Chop off the beginning part and the end part of monkey_information
    monkey_t = raw_monkey_information['time']
    if raw_monkey_information['time'][0] < smr_markers_start_time:
        valid_points = np.where((monkey_t >= smr_markers_start_time) & (
            monkey_t <= smr_markers_end_time))[0]
        monkey_information = raw_monkey_information.iloc[valid_points].copy()
    else:
        monkey_information = raw_monkey_information
    monkey_information['point_index'] = np.arange(len(monkey_information))
    monkey_information.index = monkey_information['point_index'].values
    return monkey_information


def make_or_retrieve_ff_info_from_txt_data(raw_data_folder_path, exists_ok=True, save_data=True):
    """
    Retrieve or process firefly information from text data.

    Parameters:
    - raw_data_folder_path (str): Path to the raw data folder.
    - exists_ok (bool): Flag to check if processed data exists.

    Returns:
    - Tuple containing sorted firefly information.
    """
    # Define the processed data folder path
    processed_data_folder_path = raw_data_folder_path.replace(
        'raw_data', 'processed_data')
    os.makedirs(processed_data_folder_path, exist_ok=True)

    # Attempt to retrieve processed data if it exists
    if exists_ok:
        try:
            return _retrieve_ff_info_in_npz_from_txt_data(processed_data_folder_path)
        except FileNotFoundError as e:
            print(
                f"Failed to retrieve ff info from txt data. Will get the data anew. Error: {e}")

    # Process raw data to get firefly information
    return _get_ff_pos_from_txt_data(raw_data_folder_path, save_data=save_data)


def make_or_retrieve_ff_flash_sorted_from_txt_data(raw_data_folder_path, exists_ok=True, save_data=True):
    processed_data_folder_path = raw_data_folder_path.replace(
        'raw_data', 'processed_data')
    os.makedirs(processed_data_folder_path, exist_ok=True)
    if exists_ok:
        try:
            ff_flash_sorted = _retrieve_ff_flash_sorted_in_npz(
                processed_data_folder_path)
        except FileNotFoundError as e:
            print(
                "Failed to retrieve ff_flash_sorted from txt data. Will get the data anew. Error: ", e)
            ff_flash_sorted = _get_ff_flash_sorted_from_txt_data(
                raw_data_folder_path, save_data=save_data)
    else:
        ff_flash_sorted = _get_ff_flash_sorted_from_txt_data(
            raw_data_folder_path, save_data=save_data)
    return ff_flash_sorted


def _get_ff_pos_from_txt_data(raw_data_folder_path, save_data=True):

    raw_ff_information = _get_raw_ff_information(raw_data_folder_path)

    ff_caught_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, \
        ff_flash_end_sorted = _unpack_raw_ff_information(raw_ff_information)

    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
        raw_data_folder_path, ff_caught_T_sorted=ff_caught_T_sorted)
    # for ff_caught_T and ff_believed_position, only keep the fireflies that are captured before the end of the smr data
    captured_ffs = np.where(
        (ff_caught_T_sorted <= min(smr_markers_end_time, 99999)))
    ff_caught_T_sorted = ff_caught_T_sorted[captured_ffs]
    ff_believed_position_sorted = np.array(
        ff_believed_position_sorted)[captured_ffs]

    if save_data:
        processed_data_folder_path = raw_data_folder_path.replace(
            'raw_data', 'processed_data')
        npz_file = os.path.join(os.path.join(
            processed_data_folder_path, 'ff_basic_info.npz'))
        np.savez(npz_file,
                 ff_life_sorted=ff_life_sorted,
                 ff_caught_T_sorted=ff_caught_T_sorted,
                 ff_index_sorted=ff_index_sorted,
                 ff_real_position_sorted=ff_real_position_sorted,
                 ff_believed_position_sorted=ff_believed_position_sorted,
                 ff_flash_end_sorted=ff_flash_end_sorted)
        print(f"Saved ff information to {npz_file}")

    return ff_caught_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, \
        ff_life_sorted, ff_flash_end_sorted


def _get_ff_flash_sorted_from_txt_data(raw_data_folder_path, save_data=True):
    raw_ff_information = _get_raw_ff_information(raw_data_folder_path)
    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
        raw_data_folder_path)
    ff_caught_T = []
    ff_flash = []
    for item in raw_ff_information:
        item['Life'] = np.array(
            [item['ff_flash_T'][0][0], item['ff_caught_T']])
        ff_caught_T = np.hstack((ff_caught_T, item['ff_caught_T']))
        ff_flash_currrent_ff = item['ff_flash_T']
        ff_flash_currrent_ff = ff_flash_currrent_ff[ff_flash_currrent_ff[:, 0]
                                                    < smr_markers_end_time]
        ff_flash.append(ff_flash_currrent_ff)
    sort_index = np.argsort(ff_caught_T)
    ff_flash_sorted = np.array(ff_flash, dtype=object)[sort_index].tolist()

    if save_data:
        processed_data_folder_path = raw_data_folder_path.replace(
            'raw_data', 'processed_data')
        npz_flash = os.path.join(os.path.join(
            processed_data_folder_path, 'ff_flash_sorted.npz'))
        np.savez(npz_flash, *ff_flash_sorted)
        print(f"Saved ff_flash_sorted to {npz_flash}")
    return ff_flash_sorted


def _retrieve_ff_info_in_npz_from_txt_data(processed_data_folder_path):
    npz_file = os.path.join(os.path.join(
        processed_data_folder_path, 'ff_basic_info.npz'))
    loaded_arrays = np.load(npz_file, allow_pickle=True)
    ff_life_sorted = loaded_arrays['ff_life_sorted']
    ff_caught_T_sorted = loaded_arrays['ff_caught_T_sorted']
    ff_index_sorted = loaded_arrays['ff_index_sorted']
    ff_real_position_sorted = loaded_arrays['ff_real_position_sorted']
    ff_believed_position_sorted = loaded_arrays['ff_believed_position_sorted']
    ff_flash_end_sorted = loaded_arrays['ff_flash_end_sorted']
    return ff_caught_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_end_sorted


def _retrieve_ff_flash_sorted_in_npz(processed_data_folder_path):
    npz_file_for_flash = os.path.join(os.path.join(
        processed_data_folder_path, 'ff_flash_sorted.npz'))
    loaded_ff_flash_sorted = np.load(npz_file_for_flash, allow_pickle=True)
    ff_flash_sorted = []
    # for keys, values in loaded_ff_flash_sorted.items():
    #     ff_flash_sorted.append(values)
    for file in loaded_ff_flash_sorted.files:
        ff_flash_sorted.append(loaded_ff_flash_sorted[file])
    print(f"Loaded ff_flash_sorted from {npz_file_for_flash}")
    return ff_flash_sorted


def _take_out_ff_and_monkey_line_numbers(log_content):
    ffLinenumberList = []
    for LineNumber, line in enumerate(log_content):
        key_ff = re.search('Firefly', line)
        key_monkey = re.search('monkey', line)
        if key_ff is not None:
            ffLinenumberList.append(LineNumber)
        if key_monkey is not None:
            monkeyLineNum = LineNumber
    return ffLinenumberList, monkeyLineNum


def _get_raw_ff_information(raw_data_folder_path):

    txt_files_names = [file for file in os.listdir(
        raw_data_folder_path) if ('txt' in file) and ('s' in file)]
    txt_file_path = os.path.join(raw_data_folder_path, txt_files_names[0])

    with open(txt_file_path, 'r', encoding='UTF-8') as content:
        log_content = content.readlines()

    ffLinenumberList, monkeyLineNum = _take_out_ff_and_monkey_line_numbers(
        log_content)
    raw_ff_information = []
    ffname_index = 0
    for index, LineNumber in enumerate(ffLinenumberList):
        FF_caught_T = []
        FF_Position = []
        FF_believed_position = []
        FF_flash_T = []

        if index == len(ffLinenumberList)-1:
            log_content_block = log_content[ffLinenumberList[index]+1:monkeyLineNum]
        else:
            log_content_block = log_content[ffLinenumberList[index] +
                                            1:ffLinenumberList[index+1]]

        for line in log_content_block:
            if len(line.split(' ')) == 5:
                if 'inf' not in line:
                    having_inf_in_line = False
                    FF_caught_T.append(float(line.split(' ')[0]))
                    FF_Position.append(
                        [float(line.split(' ')[1]), float(line.split(' ')[2])])
                    FF_believed_position.append(
                        [float(line.split(' ')[3]), float(line.split(' ')[4])])
                else:
                    having_inf_in_line = True
                    FF_caught_T.append(99999)
                    FF_Position.append(
                        [float(line.split(' ')[1]), float(line.split(' ')[2])])
                    FF_believed_position.append([9999, 9999])
            else:
                # if isinstance(line.split(' ')[0], float) & isinstance(line.split(' ')[1], float):
                #     FF_flash_T.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
                try:
                    FF_flash_T.append(
                        [float(line.split(' ')[0]), float(line.split(' ')[1])])
                except:
                    1

        FF_flash_T = np.array(FF_flash_T)
        seperated_ff = np.digitize(FF_flash_T.T[0], FF_caught_T)
        if not having_inf_in_line:
            unique_ff = np.unique(seperated_ff)[:-1]
        else:
            unique_ff = np.unique(seperated_ff)
        for j in unique_ff:
            raw_ff_information.append({'ff_index': ffname_index, 'ff_caught_T': FF_caught_T[j], 'ff_real_position': np.array(FF_Position[j]),
                                       'ff_believed_position': np.array(FF_believed_position[j]),
                                       'ff_flash_T': FF_flash_T[seperated_ff == j]})

            ffname_index = ffname_index+1
    return raw_ff_information


def _unpack_raw_ff_information(raw_ff_information):
    """
    Get various useful lists and arrays of ff information from the raw data;
    modified from Ruiyi's codes

    Parameters
    ----------

    raw_ff_information: list
        derived from the raw data; note that it is different from env.raw_ff_information

    Returns
    -------
    ff_caught_T_sorted: np.array
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
    ff_index = []
    ff_caught_T = []
    ff_real_position = []
    ff_believed_position = []
    ff_life = []
    ff_flash_end = []  # This is the time that the firefly last stops flash
    for item in raw_ff_information:
        item['Life'] = np.array(
            [item['ff_flash_T'][0][0], item['ff_caught_T']])
        ff_index = np.hstack((ff_index, item['ff_index']))
        ff_caught_T = np.hstack((ff_caught_T, item['ff_caught_T']))
        ff_real_position.append(item['ff_real_position'])
        ff_believed_position.append(item['ff_believed_position'])
        ff_life.append(item['Life'])
        ff_flash_end.append(item['ff_flash_T'][-1][-1])
    sort_index = np.argsort(ff_caught_T)

    ff_caught_T_sorted = ff_caught_T[sort_index]
    ff_believed_position_sorted = np.array(ff_believed_position)[sort_index]
    ff_index_sorted = ff_index[sort_index]
    ff_real_position_sorted = np.array(ff_real_position)[sort_index]
    ff_life_sorted = np.array(ff_life)[sort_index]
    ff_flash_end_sorted = np.array(ff_flash_end)[sort_index]

    return ff_caught_T_sorted, ff_index_sorted, ff_real_position_sorted, ff_believed_position_sorted, ff_life_sorted, ff_flash_end_sorted
