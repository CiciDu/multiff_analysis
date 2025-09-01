from data_wrangling import time_calib_utils, retrieve_raw_data, general_utils

import os
import math
from math import pi
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from eye_position_analysis import eye_positions

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_or_retrieve_monkey_information(raw_data_folder_path, interocular_dist, min_distance_to_calculate_angle=5, speed_threshold_for_distinct_stop=1,
                                        exists_ok=True, save_data=True):
    processed_data_folder_path = raw_data_folder_path.replace(
        'raw_monkey_data', 'processed_data')
    monkey_information_path = os.path.join(
        processed_data_folder_path, 'monkey_information.csv')
    if exists(monkey_information_path) & exists_ok:
        print("Retrieved monkey_information")
        monkey_information = pd.read_csv(
            monkey_information_path).drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
    else:
        raw_monkey_information = retrieve_raw_data.get_raw_monkey_information_from_txt_data(
            raw_data_folder_path)
        smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
            raw_data_folder_path)
        monkey_information = retrieve_raw_data._trim_monkey_information(
            raw_monkey_information, smr_markers_start_time, smr_markers_end_time)
        add_monkey_angle_column(
            monkey_information, min_distance_to_calculate_angle=min_distance_to_calculate_angle)
        add_monkey_speed_column(monkey_information)
        add_monkey_dw_column(monkey_information)

        monkey_information = add_smr_file_info_to_monkey_information(
            monkey_information, raw_data_folder_path)

        # convert the eye position data
        monkey_information = eye_positions.convert_eye_positions_in_monkey_information(
            monkey_information, add_left_and_right_eyes_info=True, interocular_dist=interocular_dist)
        monkey_information = eye_positions.find_valid_view_points(
            monkey_information)
        monkey_information = eye_positions.find_eye_world_speed(
            monkey_information)

        if save_data:
            monkey_information.to_csv(monkey_information_path)
            print("Saved monkey_information")

    monkey_information = _process_monkey_information_after_retrieval(
        monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
    return monkey_information


def _process_monkey_information_after_retrieval(monkey_information, speed_threshold_for_distinct_stop=1):
    monkey_information.index = monkey_information.point_index.values
    monkey_information = add_more_columns_to_monkey_information(
        monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
    monkey_information = take_out_suspicious_information_from_monkey_information(
        monkey_information)
    return monkey_information


def make_signal_df(raw_data_folder_path):
    # make signal_df with time_adjusted
    txt_smr_t_linreg_df = time_calib_utils.make_or_retrieve_txt_smr_t_linreg_df(
        raw_data_folder_path)
    signal_df = get_raw_signal_df(raw_data_folder_path)
    # adjust the time in signal_df based on the linear regression result stored in txt_smr_t_linreg_df
    signal_df = time_calib_utils.calibrate_smr_t(
        signal_df, txt_smr_t_linreg_df)
    return signal_df


def get_raw_signal_df(raw_data_folder_path):
    channel_signal_output, marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(
        raw_data_folder_path)
    signal_df = None
    # Considering the first smr file, using channel_signal_output[0]
    channel_signal_smr = channel_signal_output[0]
    juice_timestamp = marker_list[0]['values'][marker_list[0]['labels'] == 4]
    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
        raw_data_folder_path)
    if len(channel_signal_smr) > 0:
        # Seperate analog signal by juice timestamps
        channel_signal_smr['section'] = np.digitize(
            channel_signal_smr.Time, juice_timestamp)
        # Remove head and tail of analog data
        channel_signal_smr = channel_signal_smr[channel_signal_smr['Time']
                                                > smr_markers_start_time]
        # Remove tail of analog data
        channel_signal_smr = channel_signal_smr[channel_signal_smr['section']
                                                < channel_signal_smr['section'].unique()[-1]]
        if len(channel_signal_smr) > 0:
            # # Since there might be very slight difference between the last recorded sampling time and juice_timestamp[-1], we replace the former with the latter
            # channel_signal_smr.loc[channel_signal_smr.index[-1], 'Time'] = juice_timestamp[-1]
            # get the signal_df
            signal_df = channel_signal_smr[['LateralV', 'LDy', 'LDz', 'MonkeyX',
                                            'MonkeyY', 'RDy', 'RDz', 'AngularV', 'ForwardV', 'Time', 'section']].copy()
            # Convert columns to float, except for the 'section' column
            for column in signal_df.columns:
                if column != 'section':
                    signal_df.loc[:, column] = np.array(
                        signal_df.loc[:, column]).astype('float')
    signal_df.rename(columns={'Time': 'time'}, inplace=True)
    return signal_df


def get_monkey_speed_and_dw_from_smr_info(monkey_information):
    monkey_information['monkey_speed_smr'] = np.linalg.norm(
        monkey_information[['LateralV', 'ForwardV']].values, axis=1)
    monkey_information['monkey_dw_smr'] = monkey_information['monkey_dw_smr'] * pi/180

    # check if any point has a speed that's too high. Print the number of such points (and proportion of them) as well as highest speed
    too_high_speed_points = monkey_information[monkey_information['monkey_speed_smr'] > 200]
    if len(too_high_speed_points) > 0:
        print("There are", len(too_high_speed_points),
              "points with speed greater than 200 cm/s")
        print("The proportion of such points is", len(
            too_high_speed_points)/len(monkey_information))
        print("The highest speed is", np.max(
            monkey_information['monkey_speed_smr']))
        monkey_information.loc[monkey_information['monkey_speed_smr']
                               > 200, 'monkey_speed_smr'] = 200
    monkey_information['monkey_speeddummy_smr'] = ((monkey_information['monkey_speed_smr'] > 0.1) |
                                                   (np.abs(monkey_information['monkey_dw_smr']) > 0.0035)).astype(int)
    return monkey_information


def _get_derivative_of_a_column(monkey_information, column_name, derivative_name):
    if derivative_name in monkey_information.columns:
        return monkey_information
    dvar = np.diff(monkey_information[column_name])
    dvar1 = np.append(dvar[0], dvar)
    dvar2 = np.append(dvar, dvar[-1])
    avg_dvar = (dvar1 + dvar2)/2
    monkey_information[derivative_name] = avg_dvar
    return monkey_information


def add_delta_distance_and_cum_distance_to_monkey_information(monkey_information):
    if 'delta_distance' not in monkey_information.columns:
        monkey_x = monkey_information['monkey_x']
        monkey_y = monkey_information['monkey_y']

        monkey_information['delta_distance'] = np.sqrt(
            (monkey_x.diff())**2 + (monkey_y.diff())**2)
        monkey_information['delta_distance'] = monkey_information['delta_distance'].fillna(
            0)
    if 'cum_distance' not in monkey_information.columns:
        monkey_information['cum_distance'] = np.cumsum(
            monkey_information['delta_distance'])


def take_out_suspicious_information_from_monkey_information(monkey_information):
    # find delta_position
    delta_time = np.diff(monkey_information['time'].values)
    # times 1.5 to make the criterion slightly looser
    ceiling_of_delta_position = max(20, np.max(delta_time)*200*3)
    monkey_information, abnormal_point_index = _drop_rows_where_delta_position_exceeds_a_ceiling(
        monkey_information, ceiling_of_delta_position)

    print('The number of points that were removed due to delta_position exceeding the ceiling is', len(
        abnormal_point_index))

    # Since sometimes the erroneous points can occur several in the row, we shall repeat the procedure, until the points all come back to normal.
    # However, if the process has been repeated for more than 5 times, then we'll raise a warning.
    procedure_counter = 1
    while len(abnormal_point_index) > 0:
        # repeat the procedure above
        monkey_information, abnormal_point_index = _drop_rows_where_delta_position_exceeds_a_ceiling(
            monkey_information, ceiling_of_delta_position)
        procedure_counter += 1
        if procedure_counter == 5:
            print("Warning: there are still erroneous points in the monkey information after 5 times of correction!")

    if procedure_counter > 1:
        print("The procedure to remove erroneous points was repeated",
              procedure_counter, "times")

    return monkey_information


def _drop_rows_where_delta_position_exceeds_a_ceiling(monkey_information, ceiling_of_delta_position):
    delta_x = np.diff(monkey_information['monkey_x'].values)
    delta_y = np.diff(monkey_information['monkey_y'].values)
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    # for the points where delta_position is greater than the ceiling
    above_ceiling_point_index = np.where(
        delta_position > ceiling_of_delta_position)[0]
    # inspect if points are away from the boundary (because if it's near the boundary, then we accept the big delta position)
    corr_monkey_info = monkey_information.iloc[above_ceiling_point_index+1].copy()
    corr_monkey_info['distances_from_center'] = np.linalg.norm(
        corr_monkey_info[['monkey_x', 'monkey_y']].values, axis=1)
    # if so, delete these points
    abnormal_point_index = corr_monkey_info[corr_monkey_info['distances_from_center']
                                            < 1000 - ceiling_of_delta_position].index
    monkey_information = monkey_information.drop(abnormal_point_index)
    return monkey_information, abnormal_point_index


def _calculate_delta_xy_and_current_delta_position_given_num_points(monkey_information, i, num_points, total_points):
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

    delta_x = monkey_information['monkey_x'].values[i+num_points_future] - \
        monkey_information['monkey_x'].values[i-num_points_past]
    delta_y = monkey_information['monkey_y'].values[i+num_points_future] - \
        monkey_information['monkey_y'].values[i-num_points_past]
    current_delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    return delta_x, delta_y, current_delta_position, num_points_past, num_points_future


def add_more_columns_to_monkey_information(monkey_information, speed_threshold_for_distinct_stop=1):
    monkey_information = _get_derivative_of_a_column(
        monkey_information, column_name='monkey_dw', derivative_name='monkey_ddw')
    monkey_information = _get_derivative_of_a_column(
        monkey_information, column_name='monkey_speed', derivative_name='monkey_ddv')
    monkey_information = add_monkey_speeddummy_column(monkey_information)
    add_crossing_boundary_column(monkey_information)
    add_delta_distance_and_cum_distance_to_monkey_information(
        monkey_information)
    monkey_information = add_whether_new_distinct_stop_and_stop_id(
        monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
    # # assign "stop_id" to each stop, with each whether_new_distinct_stop==True marking a new stop id
    # monkey_information['stop_id'] = monkey_information['whether_new_distinct_stop'].cumsum(
    # ) - 1
    # monkey_information.loc[monkey_information['monkey_speeddummy']
    #                        == 1, 'stop_id'] = np.nan
    monkey_information['dt'] = (
        monkey_information['time'].shift(-1) - monkey_information['time']).ffill()
    # add turning right column
    monkey_information['turning_right'] = 0
    monkey_information.loc[monkey_information['monkey_dw']
                           < 0, 'turning_right'] = 1

    return monkey_information


def add_smr_file_info_to_monkey_information(monkey_information, raw_data_folder_path):
    signal_df = make_signal_df(raw_data_folder_path)
    monkey_information = _add_smr_file_info_to_monkey_information(
        monkey_information, signal_df)
    monkey_information.rename(columns={
                              'MonkeyX': 'monkey_x_smr', 'MonkeyY': 'monkey_y_smr', 'AngularV': 'monkey_dw_smr'}, inplace=True)
    monkey_information = get_monkey_speed_and_dw_from_smr_info(
        monkey_information)
    monkey_information.drop(columns=['LateralV', 'ForwardV'], inplace=True)
    return monkey_information


def _add_smr_file_info_to_monkey_information(monkey_information, signal_df,
                                             variables=['LDy', 'LDz', 'RDy', 'RDz', 'MonkeyX', 'MonkeyY', 'LateralV', 'ForwardV', 'AngularV']):
    monkey_information = monkey_information.copy()
    time_bins = general_utils.find_time_bins_for_an_array(
        monkey_information['time'].values)
    # add time_box to monkey_information
    monkey_information.loc[:, 'time_box'] = np.arange(
        1, len(monkey_information)+1)
    # group signal_df.time based on intervals in monkey_information['time'], thus adding the column time_box to signal_df
    signal_df['time_box'] = np.digitize(signal_df['time'].values, time_bins)
    # use groupby and then find average for LDy, LDz, RDy, RDz
    variables.append('time_box')
    # treat variables as a set, and then convert it back to a list
    variables = list(set(variables))
    condensed_signal_df = signal_df[variables]
    condensed_signal_df = condensed_signal_df.groupby(
        'time_box').median().reset_index(drop=False)

    # Put these info into monkey_information
    monkey_information = monkey_information.merge(
        condensed_signal_df, how='left', on='time_box')
    monkey_information.drop(columns=['time_box'], inplace=True)
    return monkey_information


def _trim_monkey_information(monkey_information, smr_markers_start_time, smr_markers_end_time):
    # Chop off the beginning part and the end part of monkey_information
    time = monkey_information['time'].values
    if monkey_information['time'][0] < smr_markers_start_time:
        valid_points = np.where((time >= smr_markers_start_time) & (
            time <= smr_markers_end_time))[0]
        monkey_information = monkey_information.iloc[valid_points]

    return monkey_information


def add_crossing_boundary_column(monkey_information):
    if 'crossing_boundary' in monkey_information.columns:
        return
    delta_time = np.diff(monkey_information['time'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5)
    crossing_boundary = np.append(
        0, (delta_position > ceiling_of_delta_position).astype('int'))
    monkey_information['crossing_boundary'] = crossing_boundary


def add_monkey_angle_column(monkey_information, min_distance_to_calculate_angle=5):
    # Add angle of the monkey
    monkey_angles = [pi/2]  # The monkey is at 90 degree angle at the beginning
    list_of_num_points = [0]
    list_of_num_points_past = [0]
    list_of_num_points_future = [0]
    list_of_delta_positions = [0]
    current_angle = pi/2  # This keeps track of the current angle during the iterations
    previous_angle = pi/2

    # Find the time in the data that is closest (right before) the time where we wan to know the monkey's angular position.
    total_points = len(monkey_information['time'])
    num_points = 1

    for i in range(1, total_points):

        if num_points < 1:
            num_points = 1

        if num_points >= total_points:
            # use the below so that the algorithm will not simply get stuck after num_points exceeds the total number;
            # rather, we give a num_points a chance to come down a little and re-do the calculation again.
            num_points = num_points - min(total_points - 1, 5)

        delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(
            monkey_information, i, num_points, total_points)
        # first, let's make current_delta_position within min_distance_to_calculate_angle so that we can shed off excessive distance
        while (current_delta_position > min_distance_to_calculate_angle) and (num_points > 1):
            num_points -= 1
            # now distribute the num_points to the past and future and calculate delta position
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(
                monkey_information, i, num_points, total_points)
        # then, we make sure that current_delta_position is just above min_distance_to_calculate_angle
        while (current_delta_position <= min_distance_to_calculate_angle) and (num_points < total_points):
            num_points += 1
            delta_x, delta_y, current_delta_position, num_points_past, num_points_future = _calculate_delta_xy_and_current_delta_position_given_num_points(
                monkey_information, i, num_points, total_points)
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
    monkey_information['monkey_angle'] = np.array(monkey_angles)


def add_monkey_speed_column(monkey_information):
    delta_time = np.diff(monkey_information['time'])
    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    ceiling_of_delta_position = max(10, np.max(delta_time)*200*1.5)

    # If the monkey's delta_position at one point exceeds 50, we replace it with the previous speed.
    # (This can happen when the monkey reaches the boundary and comes out at another place)
    while np.where(delta_position >= ceiling_of_delta_position)[0].size > 0:
        above_ceiling_point_index = np.where(
            delta_position >= ceiling_of_delta_position)[0]
        # find the previous speed for all those points
        delta_position_prev = np.append(np.array([0]), delta_position)
        delta_position[above_ceiling_point_index] = delta_position_prev[above_ceiling_point_index]

    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(monkey_speed[0], monkey_speed)
    monkey_information['monkey_speed'] = monkey_speed
    # and make sure that the monkey_speed does not exceed maximum speed
    monkey_information.loc[monkey_information['monkey_speed']
                           > 200, 'monkey_speed'] = 200


def add_monkey_dw_column(monkey_information):
    # positive dw means the monkey is turning counterclockwise
    delta_time = np.diff(monkey_information['time'])
    delta_angle = np.diff(monkey_information['monkey_angle'])
    delta_angle = np.remainder(delta_angle, 2*pi)
    delta_angle[delta_angle >= pi] = delta_angle[delta_angle >= pi]-2*pi
    monkey_dw = np.divide(delta_angle, delta_time)
    monkey_dw = np.append(monkey_dw[0], monkey_dw)
    monkey_information['monkey_dw'] = monkey_dw
    # monkey_information['monkey_dw'] = gaussian_filter1d(monkey_information['monkey_dw'], 1)


def add_monkey_speeddummy_column(monkey_information):
    if 'monkey_speeddummy' not in monkey_information.columns:
        monkey_information['monkey_speeddummy'] = ((monkey_information['monkey_speed'] > 0.1) |
                                                   (np.abs(monkey_information['monkey_dw']) > 0.0035)).astype(int)
    if 'monkey_speed_smr' in monkey_information.columns:
        monkey_information['monkey_speeddummy_smr'] = ((monkey_information['monkey_speed_smr'] > 0.1) |
                                                       (np.abs(monkey_information['monkey_dw_smr']) > 0.0035)).astype(int)
        # now, make monkey_speeddummy 0 if it's 0 in either monkey_speeddummy or monkey_speeddummy_smr
        monkey_information['monkey_speeddummy'] = monkey_information['monkey_speeddummy'] & monkey_information['monkey_speeddummy_smr']
        monkey_information.drop(
            columns=['monkey_speeddummy_smr'], inplace=True)
    return monkey_information

def add_whether_new_distinct_stop_and_stop_id(
    monkey_information: pd.DataFrame,
    speed_threshold_for_distinct_stop: float = 1.0,
    close_gap_seconds: float = 0.2,
    min_stop_duration: float = 0.05,
    initial_long_stop_threshold: float | None = 3.0,
    max_initial_long_stops: int = 5,
) -> pd.DataFrame:
    df = monkey_information.copy()

    # required columns
    for col in ["time", "monkey_speed", "monkey_speeddummy"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not in dataframe")

    t = df["time"].to_numpy()
    speed = df["monkey_speed"].to_numpy()
    is_stop = (df["monkey_speeddummy"].to_numpy() == 0)

    n = len(df)
    df["stop_id"] = pd.array([pd.NA]*n, dtype="Int64")
    df["whether_new_distinct_stop"] = False
    df["stop_id_duration"] = np.nan
    df["stop_id_start_time"] = np.nan
    df["stop_id_end_time"] = np.nan

    if n == 0 or not is_stop.any():
        return df

    s = is_stop.astype(np.int8)
    starts_mask = (s == 1) & (np.r_[0, s[:-1]] == 0)
    ends_mask   = (s == 1) & (np.r_[s[1:], 0] == 0)

    start_idx = np.flatnonzero(starts_mask)
    end_idx   = np.flatnonzero(ends_mask)
    n_runs = min(len(start_idx), len(end_idx))
    if n_runs == 0:
        return df
    start_idx = start_idx[:n_runs]
    end_idx   = end_idx[:n_runs]

    start_t = t[start_idx]
    end_t   = t[end_idx]
    run_dur = end_t - start_t

    keep = run_dur >= float(min_stop_duration)
    if not np.any(keep):
        # no kept runs — nothing to paint
        return df
    start_idx, end_idx = start_idx[keep], end_idx[keep]
    start_t, end_t     = start_t[keep], end_t[keep]
    n_runs = len(start_idx)

    # Merge rule
    if n_runs > 1:
        fast = speed > speed_threshold_for_distinct_stop
        cfast = np.cumsum(fast, dtype=np.int64)
        L  = end_idx[:-1]
        R  = start_idx[1:]
        Lm1 = np.maximum(L - 1, -1)
        cL  = np.where(Lm1 >= 0, cfast[Lm1], 0)
        has_fast_sep = (cfast[R] - cL) > 0
        gap_ok = (start_t[1:] - end_t[:-1]) > close_gap_seconds
        new_run_flag = has_fast_sep & gap_ok
    else:
        new_run_flag = np.array([], dtype=bool)

    run_starts_increment = np.r_[True, new_run_flag]
    merged_ids = np.cumsum(run_starts_increment.astype(np.int64)) - 1  # 0..M-1

    df_runs = pd.DataFrame({
        "run_start_idx": start_idx,
        "run_end_idx": end_idx,
        "merged_id": merged_ids
    })
    merged_span = df_runs.groupby("merged_id", sort=True).agg(
        merged_start_idx=("run_start_idx", "min"),
        merged_end_idx=("run_end_idx", "max"),
    ).reset_index()

    # turn idx -> time
    merged_span["merged_start_time"] = t[merged_span["merged_start_idx"].to_numpy()]
    merged_span["merged_end_time"]   = t[merged_span["merged_end_idx"].to_numpy()]
    merged_span["merged_duration"]   = merged_span["merged_end_time"] - merged_span["merged_start_time"]

    # drop leading long merged-stops if requested
    if initial_long_stop_threshold is not None and max_initial_long_stops > 0 and len(merged_span) > 0:
        cutoff = float(initial_long_stop_threshold)
        drop_ids = []
        for k in range(min(max_initial_long_stops, len(merged_span))):
            if merged_span.loc[k, "merged_duration"] >= cutoff:
                drop_ids.append(merged_span.loc[k, "merged_id"])
            else:
                break
        if drop_ids:
            merged_span = merged_span[~merged_span["merged_id"].isin(drop_ids)].reset_index(drop=True)

    if merged_span.empty:
        # everything filtered away
        return df

    # repaint
    for new_sid, row in enumerate(merged_span.itertuples(index=False)):
        i0 = int(row.merged_start_idx)
        i1 = int(row.merged_end_idx)
        # guard against bad ranges
        if i1 < i0:
            continue

        df.iloc[i0:i1+1, df.columns.get_loc("stop_id")] = new_sid
        df.iloc[i0,       df.columns.get_loc("whether_new_distinct_stop")] = True
        dur = float(row.merged_duration)
        st  = float(row.merged_start_time)
        et  = float(row.merged_end_time)
        df.iloc[i0:i1+1, df.columns.get_loc("stop_id_duration")]   = dur
        df.iloc[i0:i1+1, df.columns.get_loc("stop_id_start_time")] = st
        df.iloc[i0:i1+1, df.columns.get_loc("stop_id_end_time")]   = et

    return df



# ------------------------------------------------------------------------------
# -------------------------- process speed --------------------------
# ------------------------------------------------------------------------------

import numpy as np

import numpy as np
try:
    from numba import njit
except Exception:
    # If Numba isn't available, make a no-op decorator
    print("Numba isn't available, making a no-op decorator")
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

@njit(cache=True, fastmath=True)
def _tricube_weight(u):
    # u in [0, 1]
    one_minus = 1.0 - u*u*u
    return one_minus*one_minus*one_minus

@njit(cache=True, fastmath=True)
def _local_linear_slopes_multi(time, Z, window_s=0.040, min_pts=3, ridge=1e-8, grow=1.5, max_grow=10.0):
    """
    Numba-accelerated local linear slope for multiple signals at once.
    time: (n,), strictly increasing
    Z:    (n, m), each column is a signal sampled at 'time'
    Returns slopes: (n, m)
    """
    n = time.size
    m = Z.shape[1]
    slopes = np.zeros((n, m), np.float64)
    half0 = 0.5 * window_s

    for k in range(n):
        # adapt radius until we have enough points
        R = half0
        t0 = time[k] - R
        t1 = time[k] + R
        i0 = np.searchsorted(time, t0, side='left')
        i1 = np.searchsorted(time, t1, side='right')

        g = 1.0
        while (i1 - i0) < min_pts and g <= max_grow:
            g *= grow
            R = half0 * g
            t0 = time[k] - R
            t1 = time[k] + R
            i0 = np.searchsorted(time, t0, side='left')
            i1 = np.searchsorted(time, t1, side='right')
        if i1 <= i0:
            # fallback: derivative zero if we found nothing
            continue

        # Accumulate weighted moments for all signals
        # Scalar moments
        S_w = 0.0
        S_wt = 0.0
        S_wtt = 0.0
        # Vector moments per column
        S_wz  = np.zeros(m, np.float64)
        S_wtz = np.zeros(m, np.float64)

        tk = time[k]
        invR = 1.0 / R

        for i in range(i0, i1):
            tc = time[i] - tk
            u = tc * invR
            if u < 0.0:
                u = -u
            if u > 1.0:
                # outside tricube support
                continue
            w = _tricube_weight(u)
            wt = w * tc
            wtt = wt * tc

            S_w  += w
            S_wt += wt
            S_wtt += wtt

            zi = Z[i, :]
            for j in range(m):
                z = zi[j]
                S_wz[j]  += w * z
                S_wtz[j] += wt * z

        denom = (S_w * S_wtt - S_wt * S_wt) + ridge
        if denom == 0.0:
            continue

        for j in range(m):
            num = (S_w * S_wtz[j] - S_wt * S_wz[j])
            slopes[k, j] = num / denom

    return slopes

def _ensure_np1d(a):
    return np.asarray(a, dtype=float).reshape(-1)

def local_linear_velocity(time, x, y, window_s=0.040, min_pts=3, ridge=1e-8):
    t = _ensure_np1d(time)
    Z = np.column_stack(( _ensure_np1d(x), _ensure_np1d(y) ))
    dZdt = _local_linear_slopes_multi(t, Z, window_s=window_s, min_pts=min_pts, ridge=ridge)
    vx = dZdt[:, 0]
    vy = dZdt[:, 1]
    speed = np.hypot(vx, vy)
    return vx, vy, speed

def compute_kinematics_loclin(df, time_col='time', x_col='x', y_col='y', theta_col='theta',
                              window_s=0.040, use_theta='auto', min_pts=3, ridge=1e-8):
    out = df.copy()
    t = out[time_col].to_numpy(dtype=float)
    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)

    # 1) velocity
    vx, vy, speed = local_linear_velocity(t, x, y, window_s=window_s, min_pts=min_pts, ridge=ridge)
    out['vx'] = vx
    out['vy'] = vy
    out['speed'] = speed

    # 2) tangential acceleration from (vx, vy)
    dV = _local_linear_slopes_multi(t, np.column_stack((vx, vy)), window_s=window_s, min_pts=min_pts, ridge=ridge)
    ax = dV[:, 0]; ay = dV[:, 1]
    out['accel'] = (vx*ax + vy*ay) / np.maximum(speed, 1e-9)

    # 3) heading
    heading_from_v = np.arctan2(vy, vx)
    heading_src = 'velocity'
    if use_theta in ('auto', 'measured') and theta_col in out.columns:
        theta = out[theta_col].to_numpy(dtype=float)
        if np.isfinite(theta).any():
            heading = np.unwrap(theta)
            heading_src = 'measured'
        else:
            print(f"{theta_col} is not in the dataframe. Using heading from velocity.")
            heading = np.unwrap(heading_from_v)
    elif use_theta == 'velocity':
        heading = np.unwrap(heading_from_v)
    else:
        heading = np.unwrap(heading_from_v)

    # 4) angular kinematics
    ang_speed = _local_linear_slopes_multi(t, heading.reshape(-1, 1), window_s=window_s, min_pts=min_pts, ridge=ridge)[:, 0]
    ang_accel = _local_linear_slopes_multi(t, ang_speed.reshape(-1, 1), window_s=window_s, min_pts=min_pts, ridge=ridge)[:, 0]

    out['ang_speed'] = ang_speed
    out['ang_accel'] = ang_accel
    out['heading_source'] = heading_src
    return out
