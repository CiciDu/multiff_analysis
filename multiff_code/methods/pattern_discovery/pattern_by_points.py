import sys
import os
import sys
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def find_points_w_more_than_n_ff(ff_dataframe, monkey_information, ff_caught_T_new, n=2, n_max=None):
    """
    Identify points in time where a minimum number of fireflies are visible/alive 
    and group consecutive points into chunks.

    Parameters
    ----------
    ff_dataframe : pd.DataFrame
        DataFrame containing firefly data, must include:
        - 'point_index': frame/time index
        - 'ff_index': firefly identity
    monkey_information : pd.DataFrame
        DataFrame containing monkey time series, must include 'time'.
        Used to filter points based on firefly capture times.
    ff_caught_T_new : array-like
        Array of timestamps when fireflies were caught.
    n : int, optional
        Minimum number of alive fireflies for a point to be included. Default is 2.
    n_max : int, optional
        Maximum number of alive fireflies for a point to be included. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - 'point_index': the frame/time index
        - 'num_alive_ff': number of fireflies alive at that point
        - 'diff', 'diff_2': differences between consecutive points (helper columns)
        - 'chunk': ID of consecutive segments (chunks) where enough fireflies are alive
    """

    # Count the number of unique fireflies at each point
    point_vs_num_ff = ff_dataframe[['point_index', 'ff_index']].groupby('point_index').nunique()
    point_vs_num_ff = point_vs_num_ff.rename(columns={'ff_index': 'num_alive_ff'})
    point_vs_num_ff.loc[:, 'point_index'] = point_vs_num_ff.index

    # Select points where the number of alive fireflies is greater than n
    points_w_more_than_n_ff = point_vs_num_ff[point_vs_num_ff['num_alive_ff'] > n].copy()
    
    # Optionally filter points exceeding n_max fireflies
    if n_max is not None:
        points_w_more_than_n_ff = points_w_more_than_n_ff[
            points_w_more_than_n_ff['num_alive_ff'] <= n_max].copy()

    # Exclude points before the first firefly was captured
    valid_earliest_point = np.where(monkey_information['time'] > ff_caught_T_new[0])[0][0]
    points_w_more_than_n_ff = points_w_more_than_n_ff[
        points_w_more_than_n_ff['point_index'] >= valid_earliest_point].copy()

    # Compute differences between consecutive point indices
    diff = np.diff(points_w_more_than_n_ff['point_index'])
    
    # 'diff' indicates gaps before a point (0 for first point)
    points_w_more_than_n_ff['diff'] = np.append(0, diff)
    # 'diff_2' indicates gaps after a point (0 for last point)
    points_w_more_than_n_ff['diff_2'] = np.append(diff, 0)
    
    points_w_more_than_n_ff['diff'] = points_w_more_than_n_ff['diff'].astype(int)
    points_w_more_than_n_ff['diff_2'] = points_w_more_than_n_ff['diff_2'].astype(int)

    # Label consecutive points as chunks
    points_w_more_than_n_ff['chunk'] = (points_w_more_than_n_ff['diff'] != 1).cumsum()
    # Start chunk numbering at 0
    points_w_more_than_n_ff['chunk'] = points_w_more_than_n_ff['chunk'] - 1

    return points_w_more_than_n_ff



def find_changing_dw_info(chunk_df, monkey_information, ff_caught_T_new, 
                                 chunk_interval=10, minimum_time_before_capturing=0.5):
    """
    Identify time points in a monkey tracking experiment where the monkey's 
    angular velocity (dw) changes significantly, excluding times immediately 
    before the monkey catches a firefly (ff).

    Parameters
    ----------
    chunk_df : pd.DataFrame
        DataFrame containing a chunk of points with at least a 'point_index' column.
    monkey_information : pd.DataFrame
        DataFrame with time series data for the monkey, must include:
        - 'time': timestamps
        - 'point_index': index of each point
        - 'monkey_dw': angular velocity
        - 'monkey_ddw': angular acceleration
    ff_caught_T_new : array-like
        Timestamps when the monkey caught a firefly.
    chunk_interval : float, optional
        Duration (in seconds) of the chunk to analyze from the first point. Default is 10.
    minimum_time_before_capturing : float, optional
        Time (in seconds) before a firefly catch to exclude from analysis. Default is 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - 'point_index': the global point index
        - 'time': timestamp of the significant angular change
        - 'dw': angular velocity at that time
        - 'ddw': angular acceleration at that time
    """

    # Get the first point in the chunk and define the time interval to analyze
    first_point = chunk_df['point_index'].min()
    duration = [monkey_information['time'][first_point],
                monkey_information['time'][first_point] + chunk_interval]

    # Find indices in monkey_information that fall within the interval
    cum_pos_index = np.where((monkey_information['time'] >= duration[0]) & 
                             (monkey_information['time'] <= duration[1]))[0]
    cum_point_index = np.array(monkey_information['point_index'].iloc[cum_pos_index])

    # Exclude periods right before firefly captures
    relevant_ff_caught_T = ff_caught_T_new[(ff_caught_T_new >= duration[0]) & 
                                           (ff_caught_T_new <= duration[1] + minimum_time_before_capturing)]
    for time in relevant_ff_caught_T:
        duration_to_take_out = [time - minimum_time_before_capturing, time]
        cum_pos_index = cum_pos_index[~((cum_pos_index >= duration_to_take_out[0]) & 
                                        (cum_pos_index <= duration_to_take_out[1]))]

    # Extract the time, angular velocity, and acceleration for remaining indices
    cum_t = np.array(monkey_information['time'].iloc[cum_pos_index])
    cum_dw = np.array(monkey_information['monkey_dw'].iloc[cum_pos_index])
    cum_ddw = np.array(monkey_information['monkey_ddw'].iloc[cum_pos_index])
    cum_abs_ddw = np.abs(cum_ddw)

    # Identify indices where the angular acceleration exceeds a threshold
    changing_dw_info = pd.DataFrame({'relative_pos_index': np.where(cum_abs_ddw > 0.15)[0]})

    # Group consecutive points to avoid double-counting rapid sequences
    changing_dw_info['group'] = np.append(
        0, (np.diff(changing_dw_info['relative_pos_index']) != 1).cumsum())
    changing_dw_info = changing_dw_info.groupby('group').min()
    relative_pos_index = changing_dw_info['relative_pos_index'].astype(int)

    # Map relative indices to global point indices and extract motion data
    changing_dw_info['point_index'] = cum_pos_index[relative_pos_index]
    changing_dw_info['time'] = cum_t[relative_pos_index]
    changing_dw_info['dw'] = cum_dw[relative_pos_index]
    changing_dw_info['ddw'] = cum_ddw[relative_pos_index]

    return changing_dw_info



def increase_durations_between_points(df, min_duration=5):
    new_df = pd.DataFrame(columns=df.columns)
    prev_row = df.iloc[0]
    for index, row in df.iterrows():
        if (index == 0) or (row.time - prev_row.time) > min_duration:
            new_df = pd.concat(
                [new_df, pd.DataFrame(row).T], ignore_index=True)
            prev_row = row
    new_df = new_df.reset_index(drop=True)
    return new_df


def decrease_overlaps_between_chunks(points_w_more_than_n_ff, monkey_information, min_interval_between_chunks):
    temp_df = points_w_more_than_n_ff.groupby('chunk').min()
    temp_df['time'] = monkey_information['time'].values[temp_df['point_index'].values]
    temp_df = temp_df.reset_index()[['chunk', 'time']]
    new_df = increase_durations_between_points(temp_df)
    new_chunks = new_df['chunk'].astype('int')
    points_w_more_than_n_ff = points_w_more_than_n_ff[points_w_more_than_n_ff['chunk'].isin(
        new_chunks)].copy()

    # reset the chunk number so it starts from 0 again
    points_w_more_than_n_ff['chunk'] = (
        points_w_more_than_n_ff['diff'] != 1).cumsum()
    points_w_more_than_n_ff['chunk'] = points_w_more_than_n_ff['chunk']-1

    return points_w_more_than_n_ff


def find_number_of_visible_or_in_memory_ff_at_beginning_of_trials(prev_trials, ff_caught_T_new, ff_dataframe):
    list_of_num_visible_ff = []
    list_of_num_in_memory_ff = []
    for trial in prev_trials:
        duration = [ff_caught_T_new[trial]-0.1, ff_caught_T_new[trial]+0.1]
        ff_dataframe_sub = ff_dataframe[ff_dataframe['time'].between(
            *duration)]
        ff_dataframe_sub = ff_dataframe_sub.groupby(['ff_index']).max()
        visible_ff = ff_dataframe_sub[ff_dataframe_sub['memory'] == 100]
        in_memory_ff = ff_dataframe_sub[ff_dataframe_sub['memory'] < 100]
        list_of_num_visible_ff.append(len(visible_ff))
        list_of_num_in_memory_ff.append(len(in_memory_ff))
        num_visible_ff = np.array(list_of_num_visible_ff)
        num_in_memory_ff = np.array(list_of_num_in_memory_ff)
    return num_visible_ff, num_in_memory_ff


def make_target_closest_or_target_angle_smallest(ff_dataframe, max_point_index, column='ff_distance'):
    # Here we use aim to make target_closest, but the same function can be applied to make target_angle_smallest
    # The algorithm has been verified by seeing if the individual components (subsets of indices)can add up to the whole (all indices)
    # Starting from the first point index

    # make an array such as target_closest:
    # 2 means target is the closest ff at that point (visible or in memory)
    # 1 means the target is not the closest. In the subset of 1:
    # 1 means both the target and a non-target are visible or in memory (which we call present)
    # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
    # -1 means both the target and other ff are neither visible or in memory

    key_columns = ff_dataframe[['point_index',
                                'ff_index', 'target_index', column]]
    min_distance_subset = ff_dataframe[['point_index', column]].groupby(
        'point_index').min().reset_index()
    min_distance_subset = pd.merge(min_distance_subset, key_columns, on=[
                                   'point_index', column], how="left")

    # From min_distance_subset, find the rows that belong to targets and non-targets respectively.
    # Below are point indices where the target is the closest; they shall be denoted as 2
    target_closest_point_index = np.array(
        min_distance_subset[min_distance_subset['ff_index'] == min_distance_subset['target_index']].point_index)
    # Below are point indices where the target is not the closest; they the complement of target_closest_point_index
    target_not_closest_point_index = np.delete(
        np.arange(max_point_index+1), target_closest_point_index)
    # Also find the indices where the non-target is both present and closest
    non_target_present_and_closest_indices = np.array(
        min_distance_subset[min_distance_subset['ff_index'] != min_distance_subset['target_index']].point_index)

    # Among target_not_closest_point_index:
    # Find the indices where both the target and other ff are neither visible or in memory.
    # They shall be denoted -1
    both_absent_indices = np.delete(
        np.arange(max_point_index+1), np.array(min_distance_subset.point_index))
    # Then, among target_not_closest_point_index, find the ones that the non-target is present and closest to the monkey, while target is also present;
    # First find all the indices in ff_dataframe where target is present; and then use intersection
    # They shall be denoted as 1
    target_present_indices = np.array(
        ff_dataframe[ff_dataframe['ff_index'] == ff_dataframe['target_index']]['point_index'])
    both_present_and_non_target_closest_indices = np.intersect1d(
        non_target_present_and_closest_indices, target_present_indices)
    # Finally, find the indices where where non-target is present and closest to the monkey, while target is absent;
    # They shall be denoted as 0
    only_non_target_present_indices = np.setdiff1d(
        non_target_present_and_closest_indices, both_present_and_non_target_closest_indices)

    # Lastly, make result array (such as target_closest)
    result = np.arange(max_point_index+1)
    result[target_closest_point_index] = 2
    result[both_present_and_non_target_closest_indices] = 1
    result[only_non_target_present_indices] = 0
    result[both_absent_indices] = -1

    return result


def make_target_closest(ff_dataframe, max_point_index, data_folder_name=None):
    # make target_closest:
    # 2 means target is the closest ff at that point (visible or in memory)
    # 1 means the target is not the closest. In the subset of 1:
    # 1 means both the target and a non-target are visible or in memory (which we call present)
    # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
    # -1 means both the target and other ff are neither visible or in memory

    target_closest = make_target_closest_or_target_angle_smallest(
        ff_dataframe, max_point_index, column='ff_distance')
    if data_folder_name:
        np.savetxt(os.path.join(data_folder_name, 'target_closest.csv'),
                   target_closest.tolist(), delimiter=',')
    return target_closest


def make_target_angle_smallest(ff_dataframe, max_point_index, data_folder_name=None):
    # make target_angle_smallest:
    # 2 means target is has the smallest absolute angle at that point (visible or in memory)
    # 1 means the target does not have the smallest absolute angle. In the subset of 1:
    # 1 means both the target and a non-target are visible or in memory (which we call present)
    # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
    # -1 means both the target and other ff are neither visible or in memory

    ff_dataframe['ff_angle_boundary_abs'] = np.abs(
        np.array(ff_dataframe['ff_angle_boundary']))
    target_angle_smallest = make_target_closest_or_target_angle_smallest(
        ff_dataframe, max_point_index, column='ff_angle_boundary_abs')
    if data_folder_name:
        np.savetxt(os.path.join(data_folder_name, 'target_angle_smallest.csv'),
                   target_angle_smallest.tolist(), delimiter=',')
    return target_angle_smallest
