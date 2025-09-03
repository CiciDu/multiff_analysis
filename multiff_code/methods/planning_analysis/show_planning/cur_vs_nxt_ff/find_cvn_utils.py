from decision_making_analysis.decision_making import decision_making_utils
from data_wrangling import specific_utils, general_utils
from planning_analysis.show_planning import nxt_ff_utils
from null_behaviors import show_null_trajectory
from pattern_discovery import cluster_analysis
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
from math import pi
import warnings


def _add_basic_monkey_info(stops_near_ff_df, monkey_information):
    stops_near_ff_df[['stop_x', 'stop_y', 'monkey_angle', 'stop_time', 'stop_cum_distance']] = \
        monkey_information.loc[stops_near_ff_df['stop_point_index'],
                               ['monkey_x', 'monkey_y', 'monkey_angle', 'time', 'cum_distance']].values
    return stops_near_ff_df


def _add_ff_xy(stops_near_ff_df, ff_real_position_sorted):
    stops_near_ff_df[['cur_ff_x', 'cur_ff_y']
                     ] = ff_real_position_sorted[stops_near_ff_df['cur_ff_index'].values]
    if 'nxt_ff_index' in stops_near_ff_df.columns:
        stops_near_ff_df[['nxt_ff_x', 'nxt_ff_y']
                         ] = ff_real_position_sorted[stops_near_ff_df['nxt_ff_index'].values]
    return stops_near_ff_df


def _add_distance_info(stops_near_ff_df, monkey_information, ff_real_position_sorted):
    stops_near_ff_df['next_stop_cum_distance'] = monkey_information.loc[
        stops_near_ff_df['next_stop_point_index'], 'cum_distance'].values

    stops_near_ff_df['cum_distance_between_two_stops'] = stops_near_ff_df['next_stop_cum_distance'] - \
        stops_near_ff_df['stop_cum_distance']

    stops_near_ff_df['d_from_cur_ff_to_stop'] = np.linalg.norm(
        stops_near_ff_df[['stop_x', 'stop_y']].values -
        stops_near_ff_df[['cur_ff_x', 'cur_ff_y']].values,
        axis=1
    )
    stops_near_ff_df['d_from_cur_ff_to_nxt_ff'] = np.linalg.norm(ff_real_position_sorted[stops_near_ff_df['cur_ff_index'].values] -
                                                                 ff_real_position_sorted[stops_near_ff_df['nxt_ff_index'].values], axis=1)

    return stops_near_ff_df


def make_shared_stops_near_ff_df_and_all_nxt_ff_df(
    monkey_information, ff_dataframe_visible,
    closest_stop_to_capture_df,
    ff_real_position_sorted, ff_caught_T_new,
    ff_flash_sorted, ff_life_sorted,
    remove_cases_where_monkey_too_close_to_edge=False,
    stop_period_duration=2,
    min_time_between_cur_and_nxt_ff_caught_time=0.1,
    min_distance_between_cur_and_nxt_ff=25,
    max_distance_between_cur_and_nxt_ff=500,
    min_time_between_cur_ff_first_seen_time_and_stop=0.2,
):
    print('Making shared_stops_near_ff_df...')

    shared_stops_near_ff_df = find_captured_ff_info_for_making_stops_near_ff_df(
        monkey_information, ff_dataframe_visible,
        closest_stop_to_capture_df,
        stop_period_duration=stop_period_duration
    )

    shared_stops_near_ff_df = _add_basic_monkey_info(
        shared_stops_near_ff_df, monkey_information)

    shared_stops_near_ff_df = nxt_ff_utils.rename_first_and_last_seen_info_columns(
        shared_stops_near_ff_df, prefix='CUR_'
    )

    shared_stops_near_ff_df.rename(columns={
        'ff_index': 'cur_ff_index',
        'monkey_angle': 'stop_monkey_angle',
    }, inplace=True)

    shared_stops_near_ff_df['cur_ff_capture_time'] = ff_caught_T_new[
        shared_stops_near_ff_df['cur_ff_index'].values
    ]

    # Add next ff info
    all_nxt_ff_df = nxt_ff_utils.get_all_nxt_ff_df_from_ff_dataframe(
        shared_stops_near_ff_df, ff_dataframe_visible,
        closest_stop_to_capture_df, ff_real_position_sorted,
        ff_caught_T_new, ff_life_sorted, monkey_information,
        min_time_between_cur_and_nxt_ff_caught_time=min_time_between_cur_and_nxt_ff_caught_time,
        min_distance_between_cur_and_nxt_ff=min_distance_between_cur_and_nxt_ff,
        max_distance_between_cur_and_nxt_ff=max_distance_between_cur_and_nxt_ff
    )

    original_length = len(shared_stops_near_ff_df)
    shared_stops_near_ff_df = shared_stops_near_ff_df.merge(
        all_nxt_ff_df.drop(columns=['stop_point_index']),
        on='cur_ff_index',
        how='inner'
    )

    # the printed statement below might be unnecessary because when calling all_nxt_ff_df, info of removed rows are already printed
    # print(
    #     f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} '
    #     f'rows in shared_stops_near_ff_df have been removed because the nxt_ff was not found.'
    # )

    # shared_stops_near_ff_df['next_stop_cum_distance'] = monkey_information.loc[
    #     shared_stops_near_ff_df['next_stop_point_index'], 'cum_distance'
    # ].values

    # shared_stops_near_ff_df['cum_distance_between_two_stops'] = (
    #     shared_stops_near_ff_df['next_stop_cum_distance'] -
    #     shared_stops_near_ff_df['stop_cum_distance']
    # )

    shared_stops_near_ff_df = _add_ff_xy(
        shared_stops_near_ff_df, ff_real_position_sorted)

    shared_stops_near_ff_df = _add_distance_info(
        shared_stops_near_ff_df, monkey_information, ff_real_position_sorted)

    if len(shared_stops_near_ff_df['cur_ff_index'].unique()) != len(shared_stops_near_ff_df):
        warnings.warn(
            'There are duplicated cur_ff_index in shared_stops_near_ff_df. This should not happen.'
        )

    shared_stops_near_ff_df, all_nxt_ff_df = process_instances_where_monkey_is_too_close_to_edge(
        shared_stops_near_ff_df, all_nxt_ff_df, monkey_information,
        remove_cases_where_monkey_too_close_to_edge=remove_cases_where_monkey_too_close_to_edge
    )

    shared_stops_near_ff_df = add_monkey_info_before_stop(
        monkey_information, shared_stops_near_ff_df
    )

    shared_stops_near_ff_df = nxt_ff_utils.add_if_nxt_ff_and_nxt_ff_cluster_flash_bbas(
        shared_stops_near_ff_df, ff_real_position_sorted,
        ff_flash_sorted, ff_life_sorted,
        stop_period_duration=stop_period_duration
    )

    shared_stops_near_ff_df = nxt_ff_utils.add_if_nxt_ff_and_nxt_ff_cluster_flash_bsans(
        shared_stops_near_ff_df, ff_real_position_sorted,
        ff_flash_sorted, ff_life_sorted
    )

    shared_stops_near_ff_df = shared_stops_near_ff_df.sort_values(
        by='stop_point_index')

    add_cur_ff_cluster_50_size(
        shared_stops_near_ff_df, ff_real_position_sorted, ff_life_sorted)

    len_before = len(shared_stops_near_ff_df)
    shared_stops_near_ff_df = shared_stops_near_ff_df[
        (shared_stops_near_ff_df['stop_time'] - shared_stops_near_ff_df['CUR_time_ff_first_seen_bbas']) >=
        min_time_between_cur_ff_first_seen_time_and_stop
    ].copy().reset_index(drop=True)

    print(
        f'{len_before - len(shared_stops_near_ff_df)} rows out of {len_before} '
        f'rows in shared_stops_near_ff_df have been removed because the time between stop_time '
        f'and CUR_time_ff_first_seen_bbas is less than {min_time_between_cur_ff_first_seen_time_and_stop} seconds.'
    )

    return shared_stops_near_ff_df, all_nxt_ff_df


def extract_key_info_from_data_item_for_stops_near_ff_class(data_item):
    data_item_info = {'monkey_information': data_item.monkey_information,
                      'ff_dataframe': data_item.ff_dataframe,
                      'ff_caught_T_new': data_item.ff_caught_T_new,
                      'ff_real_position_sorted': data_item.ff_real_position_sorted,
                      'ff_life_sorted': data_item.ff_life_sorted,
                      'PlotTrials_args': data_item.PlotTrials_args,
                      'monkey_name': data_item.monkey_name,
                      'data_name': data_item.data_name}
    return data_item_info


def find_captured_ff_info_for_making_stops_near_ff_df(
    monkey_information,
    ff_dataframe_visible,
    closest_stop_to_capture_df,
    stop_period_duration=2,
    max_diff_between_caught_time_and_stop_time=0.2
):
    # Drop stops that are not inside the reward boundary
    closest_stop_to_capture_df = nxt_ff_utils.drop_rows_where_stop_is_not_inside_reward_boundary(
        closest_stop_to_capture_df
    )

    print('finding captured_ff_info...')

    # Get capture-related ff information (first seen, last seen, etc.)
    captured_ff_info = nxt_ff_utils.get_all_captured_ff_first_seen_and_last_seen_info(
        closest_stop_to_capture_df,
        stop_period_duration,
        ff_dataframe_visible,
        monkey_information,
        drop_na=True
    )

    # Drop rows where stop_time and capture_time are too far apart
    captured_ff_info = drop_rows_where_stop_and_capture_time_are_too_far_apart(
        captured_ff_info,
        closest_stop_to_capture_df,
        max_diff_between_caught_time_and_stop_time=max_diff_between_caught_time_and_stop_time
    )

    # Eliminate boundary cases (within n seconds before or after crossing)
    selected_point_index = captured_ff_info.stop_point_index.values
    time_of_stops = monkey_information.loc[selected_point_index, 'time'].values

    crossing_boundary_time = monkey_information.loc[
        monkey_information['crossing_boundary'] == 1, 'time'
    ].values

    n_seconds_before_crossing_boundary = 0.2
    n_seconds_after_crossing_boundary = stop_period_duration + 0.2
    CB_indices, non_CB_indices, _ = decision_making_utils.find_time_points_that_are_within_n_seconds_after_crossing_boundary(
        time_of_stops,
        crossing_boundary_time,
        n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
        n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary
    )

    selected_point_index = selected_point_index[non_CB_indices]
    print(
        f'{len(CB_indices)} rows out of {len(captured_ff_info)} rows in captured_ff_info were '
        f'dropped because they are within {n_seconds_before_crossing_boundary} seconds before or {n_seconds_after_crossing_boundary} seconds after crossing boundary.'
    )

    # Ensure the 'stop_point_index' column exists
    if 'stop_point_index' not in captured_ff_info.columns:
        captured_ff_info['stop_point_index'] = captured_ff_info['point_index']

    # Filter by non-boundary stop points
    captured_ff_info = captured_ff_info[
        captured_ff_info['stop_point_index'].isin(selected_point_index)
    ].copy()

    return captured_ff_info


def drop_rows_where_stop_and_capture_time_are_too_far_apart(
    captured_ff_info,
    closest_stop_to_capture_df,
    max_diff_between_caught_time_and_stop_time=0.2
):
    """
    Drops rows from captured_ff_info where the time between the stop and firefly capture
    exceeds a threshold (max_diff_between_caught_time_and_stop_time).
    """
    # Align column names and merge on ff_index
    merged_stop_df = closest_stop_to_capture_df.rename(
        columns={'cur_ff_index': 'ff_index'})
    captured_ff_info = captured_ff_info.merge(
        merged_stop_df[['ff_index', 'caught_time', 'diff_from_caught_time']],
        on='ff_index',
        how='left'
    )

    # Raise error if NA values are found
    if captured_ff_info['diff_from_caught_time'].isna().any():
        raise ValueError(
            "NA values found in 'diff_from_caught_time'. This should not happen.")

    # Check and remove rows exceeding max_diff_between_caught_time_and_stop_time
    too_far_mask = captured_ff_info['diff_from_caught_time'].abs(
    ) >= max_diff_between_caught_time_and_stop_time
    if too_far_mask.any():
        max_diff = captured_ff_info['diff_from_caught_time'].abs().max()
        num_exceeding = too_far_mask.sum()
        positive_diff_ratio = (
            merged_stop_df['diff_from_caught_time'] > 0
        ).sum() / len(merged_stop_df)

        print(
            f"[Warning] {num_exceeding} rows removed due to 'diff_from_caught_time' ≥ {max_diff_between_caught_time_and_stop_time} sec. "
            f"Max difference: {max_diff:.3f} sec. "
            f"Positive time diffs: {positive_diff_ratio*100:.2f}%"
        )

        captured_ff_info = captured_ff_info[~too_far_mask].copy()

    return captured_ff_info


def compute_distances_to_stop(shared_stops_near_ff_df, monkey_info):
    shared_stops_near_ff_df['next_stop_cum_distance'] = monkey_info.loc[
        shared_stops_near_ff_df['next_stop_point_index'], 'cum_distance'
    ].values
    shared_stops_near_ff_df['cum_distance_between_two_stops'] = (
        shared_stops_near_ff_df['next_stop_cum_distance'] -
        shared_stops_near_ff_df['stop_cum_distance']
    )


def process_instances_where_monkey_is_too_close_to_edge(shared_stops_near_ff_df, all_nxt_ff_df, monkey_information, remove_cases_where_monkey_too_close_to_edge=False):
    if remove_cases_where_monkey_too_close_to_edge is True:
        original_length = len(shared_stops_near_ff_df)
        shared_stops_near_ff_df = remove_cases_where_monkey_too_close_to_edge_func(
            shared_stops_near_ff_df, monkey_information)
        print(f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} rows in shared_stops_near_ff_df have been removed because the monkey was too close to the edge.')
        all_nxt_ff_df = all_nxt_ff_df[all_nxt_ff_df['stop_point_index'].isin(
            shared_stops_near_ff_df['stop_point_index'])].copy().sort_values(by='stop_point_index').reset_index(drop=True)
    else:
        # if there's crossing boundary between the stop and the next stop, we shall remove the row
        original_length = len(shared_stops_near_ff_df)
        crossing_boundary_points = monkey_information.loc[
            monkey_information['crossing_boundary'] == 1, 'point_index'].values
        shared_stops_near_ff_df['crossing_boundary'] = np.diff(np.searchsorted(crossing_boundary_points, shared_stops_near_ff_df[[
                                                               'stop_point_index', 'next_stop_point_index']].values), axis=1).flatten()
        shared_stops_near_ff_df = shared_stops_near_ff_df[shared_stops_near_ff_df['crossing_boundary'] == 0].copy(
        ).reset_index(drop=True)
        all_nxt_ff_df = all_nxt_ff_df[all_nxt_ff_df['stop_point_index'].isin(
            shared_stops_near_ff_df['stop_point_index'])].copy().sort_values(by='stop_point_index').reset_index(drop=True)
        print(f'{original_length - len(shared_stops_near_ff_df)} rows out of {original_length} rows in shared_stops_near_ff_df have been removed because the monkey has crossed boundary between two stops.')
    return shared_stops_near_ff_df, all_nxt_ff_df


def add_cur_ff_cluster_50_size(shared_stops_near_ff_df, ff_real_position_sorted, ff_life_sorted, empty_cluster_ok=False):
    ff_positions = shared_stops_near_ff_df[['cur_ff_x', 'cur_ff_y']].values
    if 'cur_ff_capture_time' in shared_stops_near_ff_df.columns:
        array_of_end_time_of_evaluation = shared_stops_near_ff_df['cur_ff_capture_time'].values
    else:
        array_of_end_time_of_evaluation = shared_stops_near_ff_df['stop_time'].values
    ff_indices_of_each_cluster = cluster_analysis.find_alive_ff_clusters(ff_positions, ff_real_position_sorted, shared_stops_near_ff_df['beginning_time'].values,
                                                                         array_of_end_time_of_evaluation, ff_life_sorted, max_distance=50, empty_cluster_ok=empty_cluster_ok)
    all_cluster_size = np.array([len(array)
                                for array in ff_indices_of_each_cluster])
    shared_stops_near_ff_df['cur_ff_cluster_50_size'] = all_cluster_size


def check_for_unique_stop_point_index_or_ff_index(shared_stops_near_ff_df):
    shared_stops_near_ff_df = shared_stops_near_ff_df.copy()
    shared_stops_near_ff_df.drop_duplicates(
        subset=[['stop_point_index', 'cur_ff_index', 'nxt_ff_index']], inplace=True)
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['stop_point_index'].unique()):
        raise ValueError(
            'There are duplicated stop_point_index in shared_stops_near_ff_df for the same cur_ff_index or nxt_ff_index')
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['cur_ff_index'].unique()):
        raise ValueError(
            'There are duplicated cur_ff_index in shared_stops_near_ff_df for the same stop_point_index or nxt_ff_index')
    if len(shared_stops_near_ff_df) != len(shared_stops_near_ff_df['nxt_ff_index'].unique()):
        raise ValueError(
            'There are duplicated cur_ff_index in shared_stops_near_ff_df for the same stop_point_index or cur_ff_index')


def remove_cases_where_monkey_too_close_to_edge_func(stops_near_ff_df, monkey_information, distance_to_area_edge=50):
    monkey_information['monkey_r'] = np.sqrt(
        monkey_information['monkey_x']**2 + monkey_information['monkey_y']**2)
    # eliminate cases where the monkey has been within 50cm of the area edge
    edge_points = monkey_information.loc[monkey_information['monkey_r']
                                         > 1000 - distance_to_area_edge, 'point_index'].values
    stops_near_ff_df['edge'] = np.diff(np.searchsorted(edge_points, stops_near_ff_df[[
                                       'stop_point_index', 'next_stop_point_index']].values), axis=1).flatten()
    stops_near_ff_df = stops_near_ff_df[stops_near_ff_df['edge'] == 0].copy(
    ).reset_index(drop=True)
    return stops_near_ff_df


def calculate_info_based_on_monkey_angles(stops_near_ff_df, monkey_angles):
    info_df = stops_near_ff_df[[
        'stop_point_index', 'd_from_cur_ff_to_stop', 'd_from_cur_ff_to_nxt_ff']].copy()
    info_df['monkey_angle'] = monkey_angles
    info_df['angle_from_cur_ff_to_stop'] = specific_utils.calculate_angles_to_ff_centers(ff_x=stops_near_ff_df['stop_x'].values, ff_y=stops_near_ff_df['stop_y'],
                                                                                         mx=stops_near_ff_df['cur_ff_x'].values, my=stops_near_ff_df['cur_ff_y'], m_angle=monkey_angles)
    info_df['angle_from_cur_ff_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(ff_x=stops_near_ff_df['nxt_ff_x'].values, ff_y=stops_near_ff_df['nxt_ff_y'],
                                                                                           mx=stops_near_ff_df['cur_ff_x'].values, my=stops_near_ff_df['cur_ff_y'], m_angle=monkey_angles)
    info_df['dir_from_cur_ff_to_stop'] = np.sign(
        info_df['angle_from_cur_ff_to_stop'])
    info_df['dir_from_cur_ff_to_nxt_ff'] = np.sign(
        info_df['angle_from_cur_ff_to_nxt_ff'])
    return info_df


def add_monkey_info_before_stop(monkey_information, stops_near_ff_df):
    # add the info about the monkey before the stop; the time is the most recent time when the speed was greater than 20 cm/s
    stops_near_ff_df = stops_near_ff_df.copy()
    stops_near_ff_df['stop_counter'] = np.arange(len(stops_near_ff_df))

    # we take out all the potential information from monkey_information first
    monkey_info_sub = monkey_information[monkey_information['speed'] > 20].copy(
    )
    monkey_info_sub['closest_future_stop'] = np.searchsorted(
        stops_near_ff_df['stop_point_index'].values, monkey_info_sub.index.values)
    monkey_info_sub.sort_values(by='time', inplace=True)
    # then, for each stop, we only keep the most recent monkey info before the stop
    monkey_info_sub = monkey_info_sub.groupby(
        'closest_future_stop').tail(1).reset_index(drop=False)
    monkey_info_sub.rename(columns={'closest_future_stop': 'stop_counter',
                                    'monkey_angle': 'monkey_angle_before_stop',
                                    'point_index': 'point_index_before_stop'}, inplace=True)

    # then, we furnish stops_near_ff_df with the monkey info before the stop by merging
    stops_near_ff_df = pd.merge(stops_near_ff_df, monkey_info_sub[[
                                'stop_counter', 'point_index_before_stop', 'monkey_angle_before_stop']], on='stop_counter', how='left')
    # forward fill the nan values; this is necessary because sometimes we don't have any point with speed > 20 between two stops, in which case the two stops share the same monkey info before stop
    stops_near_ff_df['monkey_angle_before_stop'] = stops_near_ff_df['monkey_angle_before_stop'].ffill()
    stops_near_ff_df['point_index_before_stop'] = stops_near_ff_df['point_index_before_stop'].ffill()
    # drop rows with na in 'point_index_before_stop'
    stops_near_ff_df.dropna(subset=['point_index_before_stop'], inplace=True)
    stops_near_ff_df['point_index_before_stop'] = stops_near_ff_df['point_index_before_stop'].astype(
        int)
    stops_near_ff_df['distance_before_stop'] = monkey_information.loc[
        stops_near_ff_df['point_index_before_stop'].values, 'cum_distance'].values
    stops_near_ff_df['time_before_stop'] = monkey_information.loc[stops_near_ff_df['point_index_before_stop'].values, 'time'].values
    stops_near_ff_df.drop(columns=['stop_counter'], inplace=True)

    return stops_near_ff_df


def modify_position_of_ff_with_big_angle_for_finding_null_arc(
    ff_df,
    remove_i_o_modify_rows_with_big_ff_angles=True,
    verbose=True
):
    """
    Filters or modifies firefly (ff) positions with undesirable relative angles
    or distances for null arc analysis.

    Args:
        ff_df (pd.DataFrame): DataFrame with firefly and monkey position data.
        remove_i_o_modify_rows_with_big_ff_angles (bool): Whether to drop or modify rows where
                                                ff_x_relative > ff_y_relative.
        verbose (bool): Whether to print warnings and logs.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Filtered/modified DataFrame and indices of kept rows.
    """

    ff_df = ff_df.copy()
    original_ff_df = ff_df.copy()

    if not remove_i_o_modify_rows_with_big_ff_angles and verbose:
        print(
            "[Note] When calling modify_position_of_ff_with_big_angle_for_finding_null_arc, "
            "Even with `remove_i_o_modify_rows_with_big_ff_angles=True`, "
            "some rows may still be removed (e.g., negative ff_y_relative). "
            "Use this function mainly for curvature calculation."
        )

    # Step 1: Remove rows where monkey is within the firefly radius
    outside_ff_mask = ff_df['ff_distance'].values > 25
    ff_df = ff_df[outside_ff_mask].copy()
    indices_of_kept_rows = np.where(outside_ff_mask)[0]

    # Step 2: Compute relative positions
    ff_x_rel, ff_y_rel = show_null_trajectory.find_relative_xy_positions(
        ff_df['ff_x'].values,
        ff_df['ff_y'].values,
        ff_df['monkey_x'].values,
        ff_df['monkey_y'].values,
        ff_df['monkey_angle'].values
    )

    # Step 3: Remove rows with negative y-relative values
    valid_y_mask = ff_y_rel > 0
    if not np.all(valid_y_mask):
        if verbose:
            pct_neg = 100 * np.sum(~valid_y_mask) / len(ff_y_rel)
            print(
                f"[Warning] {pct_neg:.3f}% of ff_y_relative values are negative and will be removed.")
        indices_of_kept_rows = indices_of_kept_rows[valid_y_mask]
        ff_df = original_ff_df.iloc[indices_of_kept_rows].copy()
        ff_x_rel, ff_y_rel = show_null_trajectory.find_relative_xy_positions(
            ff_df['ff_x'].values,
            ff_df['ff_y'].values,
            ff_df['monkey_x'].values,
            ff_df['monkey_y'].values,
            ff_df['monkey_angle'].values
        )

    # Step 4: Handle big angle cases (|x| > y)
    big_angle_mask = np.abs(ff_x_rel) > ff_y_rel
    ff_df['valid_null_arc'] = 1
    if np.any(big_angle_mask):
        pct_big = 100 * np.sum(big_angle_mask) / len(ff_x_rel)

        if remove_i_o_modify_rows_with_big_ff_angles:
            if verbose:
                print(
                    f"[Warning] {pct_big:.3f}% of ff_x_relative > ff_y_relative. Rows will be removed.")
            keep_mask = ~big_angle_mask
            indices_of_kept_rows = indices_of_kept_rows[keep_mask]
            ff_df = original_ff_df.iloc[indices_of_kept_rows].copy()
        else:
            if verbose:
                print(
                    f"[Warning] {pct_big:.3f}% of ff_x_relative > ff_y_relative. Will be modified instead.")
            # Clip x to match y magnitude
            ff_x_rel[big_angle_mask] = ff_y_rel[big_angle_mask] * \
                np.sign(ff_x_rel[big_angle_mask])

            # Recompute angle and distance
            ff_angle = np.arctan2(ff_y_rel, ff_x_rel) - np.pi / 2
            ff_distance = np.sqrt(ff_x_rel ** 2 + ff_y_rel ** 2)

            ff_df['ff_angle'] = ff_angle
            ff_df['ff_distance'] = ff_distance
            ff_df['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
                angles_to_ff=ff_angle,
                distances_to_ff=ff_distance,
                ff_radius=10
            )

            # Convert back to absolute positions
            ff_x, ff_y = show_null_trajectory.turn_relative_xy_positions_to_absolute_xy_positions(
                ff_x_rel, ff_y_rel,
                ff_df['monkey_x'], ff_df['monkey_y'], ff_df['monkey_angle']
            )
            ff_df['ff_x'] = ff_x
            ff_df['ff_y'] = ff_y

            ff_df.loc[big_angle_mask, 'valid_null_arc'] = 0

    return ff_df, indices_of_kept_rows


def find_ff_info_n_seconds_ago(ff_df, monkey_information, ff_real_position_sorted, n_seconds=-1):
    # new_ff_df = ff_df[['ff_index']].copy()
    ff_df = ff_df.copy()
    ff_df['time'] = ff_df['stop_time'] + n_seconds
    ff_df['point_index'] = monkey_information['point_index'].values[np.searchsorted(
        monkey_information['time'].values, ff_df['time'].values, side='right') - 1]
    ff_df['point_index'] = np.clip(ff_df['point_index'], monkey_information['point_index'].min(
    ), monkey_information['point_index'].max())
    new_ff_df = find_ff_info(ff_df.ff_index.values, ff_df.point_index.values,
                             monkey_information, ff_real_position_sorted)
    new_ff_df['stop_point_index'] = ff_df['stop_point_index'].values
    return new_ff_df


def find_ff_info_n_cm_ago(ff_df, monkey_information, ff_real_position_sorted, n_cm=-50):
    # new_ff_df = ff_df[['ff_index']].copy()
    ff_df = ff_df.copy()
    ff_df['cum_distance'] = ff_df['stop_cum_distance'] + n_cm
    ff_df['point_index'] = monkey_information['point_index'].values[np.searchsorted(
        monkey_information['cum_distance'].values, ff_df['cum_distance'].values, side='right') - 1]
    ff_df['point_index'] = np.clip(ff_df['point_index'], monkey_information['point_index'].min(
    ), monkey_information['point_index'].max())
    new_ff_df = find_ff_info(ff_df.ff_index.values, ff_df.point_index.values,
                             monkey_information, ff_real_position_sorted)
    new_ff_df['stop_point_index'] = ff_df['stop_point_index'].values
    return new_ff_df


def find_ff_info(all_ff_index, all_point_index, monkey_information, ff_real_position_sorted):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors

        ff_df = pd.DataFrame(
            {'ff_index': np.array(all_ff_index).astype(int), 'point_index': np.array(all_point_index).astype(int)})

    ff_df[['ff_x', 'ff_y']] = ff_real_position_sorted[ff_df['ff_index'].values]
    ff_df[['monkey_x', 'monkey_y', 'monkey_angle']] = monkey_information.loc[ff_df['point_index'], [
        'monkey_x', 'monkey_y', 'monkey_angle']].values
    ff_df['ff_distance'] = np.linalg.norm(
        ff_df[['monkey_x', 'monkey_y']].values - ff_real_position_sorted[ff_df['ff_index'].values], axis=1)
    ff_df['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(ff_x=ff_df['ff_x'].values, ff_y=ff_df['ff_y'].values,
                                                                      mx=ff_df['monkey_x'].values, my=ff_df['monkey_y'].values, m_angle=ff_df['monkey_angle'].values)
    ff_df['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_df.ff_angle.values, distances_to_ff=ff_df.ff_distance.values, ff_radius=10)
    return ff_df


def normalize(array):
    array = (array - array.mean()) / array.std()
    return array


def plot_relationship(nxt_curv_counted, traj_curv_counted, slope=None, show_plot=True, change_units_to_degrees_per_m=True):

    nxt_curv_counted = nxt_curv_counted.copy()
    traj_curv_counted = traj_curv_counted.copy()

    if change_units_to_degrees_per_m:
        nxt_curv_counted = nxt_curv_counted * (180/np.pi) * 100
        traj_curv_counted = traj_curv_counted * (180/np.pi) * 100

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        nxt_curv_counted, traj_curv_counted)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(nxt_curv_counted, traj_curv_counted)
    # calculate and plot linear correlation
    x_min = min(nxt_curv_counted)
    x_max = max(nxt_curv_counted)
    ax.plot(np.array([x_min, x_max]), np.array(
        [x_min, x_max])*slope+intercept, color='red')
    plt.ylabel('curv_of_traj - curv_to_cur_ff')
    plt.xlabel('curv_to_nxt_ff - curv_to_cur_ff')
    plt.title('r_value = %f' % r_value + ', slope = %f' % slope)
    ax.grid()
    ax.axvline(x=0, color='black', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--')

    if show_plot:
        plt.show()

    return ax


def find_relative_curvature(nxt_ff_counted_df, cur_ff_counted_df, curv_of_traj_counted, use_curv_to_ff_center):
    if curv_of_traj_counted is None:
        raise ValueError('curv_of_traj_counted cannot be None')

    if use_curv_to_ff_center:
        curv_var = 'cntr_arc_curv'
    else:
        curv_var = 'opt_arc_curv'

    nxt_ff_counted_df = nxt_ff_counted_df.copy()
    cur_ff_counted_df = cur_ff_counted_df.copy()

    traj_curv_counted = curv_of_traj_counted - cur_ff_counted_df[curv_var]
    nxt_curv_counted = nxt_ff_counted_df[curv_var] - \
        cur_ff_counted_df[curv_var]

    traj_curv_counted = traj_curv_counted.values
    nxt_curv_counted = nxt_curv_counted.values
    return traj_curv_counted, nxt_curv_counted


def find_outliers_in_a_column(df, column, outlier_z_score_threshold=2):
    outlier_positions = general_utils.find_time_bins_for_an_array(
        df[column].values, outlier_z_score_threshold=outlier_z_score_threshold)
    non_outlier_positions = np.setdiff1d(range(len(df)), outlier_positions)
    return outlier_positions, non_outlier_positions


def confine_angle_to_within_one_pie(angle_array):
    angle_array = angle_array % (2*math.pi)
    while np.any(angle_array > math.pi):
        angle_array[angle_array >
                    math.pi] = angle_array[angle_array > math.pi] - 2*math.pi
    while np.any(angle_array < -math.pi):
        angle_array[angle_array < -
                    math.pi] = angle_array[angle_array < -math.pi] + 2*math.pi
    return angle_array


def confine_angle_to_within_180(angle_array):
    angle_array = angle_array % 360
    while np.any(angle_array > 180):
        angle_array[angle_array > 180] = angle_array[angle_array > 180] - 2*180
    while np.any(angle_array < -180):
        angle_array[angle_array < -
                    180] = angle_array[angle_array < -180] + 2*180
    return angle_array


def organize_snf_streamline_organizing_info_kwargs(ref_point_params, curv_of_traj_params, overall_params):
    snf_streamline_organizing_info_kwargs = {
        'ref_point_mode': ref_point_params['ref_point_mode'],
        'ref_point_value': ref_point_params['ref_point_value'],
        'eliminate_outliers': overall_params['eliminate_outliers'],
        'curv_of_traj_mode': curv_of_traj_params['curv_of_traj_mode'],
        'window_for_curv_of_traj': curv_of_traj_params['window_for_curv_of_traj'],
        'truncate_curv_of_traj_by_time_of_capture': curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'],
        'remove_i_o_modify_rows_with_big_ff_angles': overall_params['remove_i_o_modify_rows_with_big_ff_angles'],
        'use_curv_to_ff_center': overall_params['use_curv_to_ff_center']}
    return snf_streamline_organizing_info_kwargs


def add_instances_to_polar_plot(axes, stops_near_ff_df, nxt_ff_df_from_ref, monkey_information, max_instances, color='green',
                                start='stop_point_index', end='next_stop_point_index'):

    traj_df, stop_point_df, next_stop_point_df = _get_important_df_for_polar_plot(
        stops_near_ff_df, nxt_ff_df_from_ref, monkey_information, max_instances, start, end)

    # Visualize ff_info
    axes.scatter(traj_df['monkey_angle_from_ref'].values, traj_df['monkey_distance_from_ref'].values,
                 c=color, alpha=0.3, zorder=2, marker='o', s=1)  # originally it was s=15
    if start == 'ref_point_index':
        axes.scatter(stop_point_df['monkey_angle_from_ref'].values, stop_point_df['monkey_distance_from_ref'].values,
                     c='red', alpha=0.5, zorder=3, marker='s', s=3)
    if end == 'next_stop_point_index':
        axes.scatter(next_stop_point_df['monkey_angle_from_ref'].values, next_stop_point_df['monkey_distance_from_ref'].values,
                     c='blue', alpha=0.5, zorder=3, marker='*', s=3)
    return axes


def add_instances_to_plotly_polar_plot(fig, stops_near_ff_df, nxt_ff_df_from_ref, monkey_information, max_instances, color='green', point_color='blue',
                                       start='stop_point_index', end='next_stop_point_index', legendgroup='Test data'):

    traj_df, stop_point_df, next_stop_point_df = _get_important_df_for_polar_plot(
        stops_near_ff_df, nxt_ff_df_from_ref, monkey_information, max_instances, start, end)

    # Main scatter plot for each subset
    fig.add_trace(go.Scatterpolar(
        r=traj_df['monkey_distance_from_ref'].values,
        theta=traj_df['monkey_angle_from_ref'].values * 180/pi,
        mode='markers',
        marker=dict(color=color, size=2, opacity=0.5),
        name=legendgroup,
        legendgroup=legendgroup,
    ))

    # Additional markers for start and end points
    if start == 'ref_point_index':
        fig.add_trace(go.Scatterpolar(
            r=stop_point_df['monkey_distance_from_ref'],
            theta=stop_point_df['monkey_angle_from_ref'] * 180/pi,
            mode='markers',
            marker=dict(color=point_color, size=4, opacity=0.7),
            name='stop Points',
            legendgroup=legendgroup,
        ))
    if end == 'next_stop_point_index':
        fig.add_trace(go.Scatterpolar(
            r=next_stop_point_df['monkey_distance_from_ref'],
            theta=next_stop_point_df['monkey_angle_from_ref'] * 180/pi,
            mode='markers',
            marker=dict(color=point_color, size=4, opacity=0.7),
            name='next Stop Points',
            legendgroup=legendgroup,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction="clockwise")
        ),
        title="Monkey Movement Polar Plot"
    )

    return fig


def _get_important_df_for_polar_plot(stops_near_ff_df, nxt_ff_df_from_ref, monkey_information, max_instances, start, end):
    traj_df = pd.DataFrame()
    stop_point_df = pd.DataFrame()
    next_stop_point_df = pd.DataFrame()

    for index, row in stops_near_ff_df.iterrows():
        if index >= max_instances:
            break

        monkey_sub = _get_monkey_sub_for_polar_plot(
            monkey_information, row, nxt_ff_df_from_ref, start, end)
        traj_df = pd.concat(
            [traj_df, monkey_sub[['monkey_angle_from_ref', 'monkey_distance_from_ref']]], axis=0)
        if start == 'ref_point_index':
            stop_point_df = pd.concat(
                [stop_point_df, monkey_sub.loc[[row['stop_point_index']]]], axis=0)
        if end == 'next_stop_point_index':
            next_stop_point_df = pd.concat(
                [next_stop_point_df, monkey_sub.loc[[row['next_stop_point_index']]]], axis=0)
    return traj_df, stop_point_df, next_stop_point_df


def _get_monkey_sub_for_polar_plot(monkey_information, row, nxt_ff_df_from_ref, start, end):
    point_index_dict = {'stop_point_index': row['stop_point_index'],
                        'next_stop_point_index': row['next_stop_point_index'],
                        'ref_point_index': nxt_ff_df_from_ref.loc[nxt_ff_df_from_ref['ff_index'] == row['nxt_ff_index'], 'point_index'].item()
                        }

    monkey_sub = monkey_information.loc[point_index_dict[start]
        : point_index_dict[end] + 1].copy()

    # rotated monkey_x and monkey_y in reference to monkey angle at the reference point
    monkey_ref_xy = monkey_sub.loc[point_index_dict[start], [
        'monkey_x', 'monkey_y']].values
    monkey_ref_angle = monkey_sub.loc[point_index_dict[start], 'monkey_angle'].item(
    )
    monkey_sub['monkey_distance_from_ref'] = np.linalg.norm(
        monkey_sub[['monkey_x', 'monkey_y']].values - monkey_ref_xy, axis=1)
    monkey_sub['monkey_angle_from_ref'] = np.arctan2(
        monkey_sub['monkey_y'] - monkey_ref_xy[1], monkey_sub['monkey_x'] - monkey_ref_xy[0]) - monkey_ref_angle
    return monkey_sub


def check_ff_vs_cluster(df, ff_column, cluster_column):
    # check for na in both columns
    print(
        f'There are {df[ff_column].isnull().sum()} rows where {ff_column} is null')
    print(
        f'There are {df[cluster_column].isnull().sum()} rows where {cluster_column} is null')
    print('===============================================')

    len_subset = len(df[df[ff_column] < df[cluster_column]])
    print(
        f'There are {len_subset} rows where {ff_column} < {cluster_column} (out of {len(df)} rows)')
    len_subset = len(df[df[ff_column] > df[cluster_column]])
    print(
        f'There are {len_subset} rows where {ff_column} > {cluster_column} (out of {len(df)} rows)')
    len_subset = len(df[df[ff_column] == df[cluster_column]])
    print(
        f'There are {len_subset} rows where {ff_column} == {cluster_column} (out of {len(df)} rows)')
    print('===============================================')
    len_subset = len(df[(df[ff_column].isnull()) &
                     (~df[cluster_column].isnull())])
    print(
        f'There are {len_subset} rows where {ff_column} is null but {cluster_column} is not null (out of {len(df)} rows)')
    len_subset = len(df[(~df[ff_column].isnull()) &
                     (df[cluster_column].isnull())])
    print(
        f'There are {len_subset} rows where {ff_column} is not null but {cluster_column} is null (out of {len(df)} rows)')


def get_df_name_by_ref(monkey_name, ref_point_mode, ref_point_value):
    if ref_point_mode == 'time':
        ref_point_mode_name = 'time'
    elif ref_point_mode == 'distance':
        ref_point_mode_name = 'dist'
        ref_point_value = int(ref_point_value)
    elif ref_point_mode == 'time after cur ff visible':
        ref_point_mode_name = 'cur_vis'
    else:
        ref_point_mode_name = 'special'

    if monkey_name is not None:
        if len(monkey_name.split('_')) > 1:
            df_name = monkey_name.split(
                '_')[1] + '_' + ref_point_mode_name + '_' + str((ref_point_value))
            df_name = df_name.replace('.', '_')
            return df_name

    # otherwise
    df_name = ref_point_mode_name + '_' + str(ref_point_value)
    df_name = df_name.replace('.', '_')
    return df_name


def find_diff_in_curv_df_name(ref_point_mode=None, ref_point_value=None, curv_traj_window_before_stop=[-25, 0]):
    if (ref_point_mode is not None) & (ref_point_value is not None):
        ref_df_name = get_df_name_by_ref(None, ref_point_mode, ref_point_value)
        ref_df_name = ref_df_name + '_'
    else:
        ref_df_name = ''
    df_name = ref_df_name + \
        f'window_{curv_traj_window_before_stop[0]}cm_{curv_traj_window_before_stop[1]}cm'
    return df_name


def find_ff_info_based_on_ref_point(ff_info, monkey_information, ff_real_position_sorted, ref_point_mode='distance', ref_point_value=-150,
                                    point_index_cur_ff_first_seen=None,
                                    # Note: ref_point_mode can be 'time', 'distance', or ‘time after cur ff visible’
                                    ):
    if ref_point_mode == 'time':
        if ref_point_value >= 0:
            raise ValueError(
                'ref_point_value must be negative for ref_point_mode = "time"')
        ff_info2 = find_ff_info_n_seconds_ago(
            ff_info, monkey_information, ff_real_position_sorted, n_seconds=ref_point_value)
    elif ref_point_mode == 'distance':
        if ref_point_value >= 0:
            raise ValueError(
                'ref_point_value must be negative for ref_point_mode = "distance"')
        if 'stop_cum_distance' not in ff_info.columns:
            ff_info['stop_cum_distance'] = monkey_information.loc[ff_info['stop_point_index'].values,
                                                                  'cum_distance'].values
        ff_info2 = find_ff_info_n_cm_ago(
            ff_info, monkey_information, ff_real_position_sorted, n_cm=ref_point_value)
    elif ref_point_mode == 'time after cur ff visible':
        if point_index_cur_ff_first_seen is None:
            point_index_cur_ff_first_seen = ff_info['point_index_ff_first_seen'].values

        arr = np.asarray(point_index_cur_ff_first_seen)
        if np.isnan(arr).any():
            raise ValueError(
                "NaN found in point_index_cur_ff_first_seen. Consider using a different ref_point_mode.")

        all_time = monkey_information.loc[point_index_cur_ff_first_seen.astype(int),
                                          'time'].values + ref_point_value
        new_point_index = monkey_information['point_index'].values[np.searchsorted(
            monkey_information['time'].values, all_time, side='right') - 1]
        new_point_index = np.clip(new_point_index, monkey_information['point_index'].min(
        ), monkey_information['point_index'].max())
        ff_info2 = find_ff_info(
            ff_info.ff_index.values, new_point_index, monkey_information, ff_real_position_sorted)
        ff_info2['stop_point_index'] = ff_info['stop_point_index'].values
    else:
        raise ValueError('ref_point_mode not recognized')
    ff_info2 = ff_info2.sort_values(
        by='stop_point_index').reset_index(drop=True)
    return ff_info2


def process_shared_stops_near_ff_df(shared_stops_near_ff_df):
    shared_stops_near_ff_df['temp_id'] = np.arange(
        len(shared_stops_near_ff_df))
    original_len = len(shared_stops_near_ff_df)
    stop_periods_cur_ff_not_visible = shared_stops_near_ff_df[shared_stops_near_ff_df[
        'CUR_point_index_ff_first_seen_bbas'].isnull()].temp_id.values
    stop_periods_nxt_ff_not_visible = shared_stops_near_ff_df[shared_stops_near_ff_df[[
        'NXT_time_ff_first_seen_bbas', 'NXT_time_ff_first_seen_bsans']].isnull().all(axis=1) == True].temp_id.values
    stop_periods_to_remove = np.concatenate(
        [stop_periods_nxt_ff_not_visible, stop_periods_cur_ff_not_visible])
    shared_stops_near_ff_df = shared_stops_near_ff_df[~shared_stops_near_ff_df['temp_id'].isin(
        stop_periods_to_remove)].copy()
    shared_stops_near_ff_df.drop(columns=['temp_id'], inplace=True)
    print(f'Removed {original_len - len(shared_stops_near_ff_df)} rows out of {original_len} rows where cur_ff was not visible bbas or nxt_ff was not visible both bbas and bsans')
    print(f'shared_stops_near_ff_df has {len(shared_stops_near_ff_df)} rows')
    return shared_stops_near_ff_df


def get_ref_point_descr_and_column(ref_point_mode, ref_point_value):
    if ref_point_mode == 'time':
        if ref_point_value >= 0:
            raise ValueError(
                'ref_point_value must be negative for ref_point_mode = "time"')
        ref_point_descr = 'based on %d s into past' % ref_point_value
        ref_point_column = 'rel_time'
        used_points_n_seconds_or_cm_ago = True
    elif ref_point_mode == 'distance':
        if ref_point_value >= 0:
            raise ValueError(
                'ref_point_value must be negative for ref_point_mode = "distance"')
        ref_point_descr = 'based on %d cm into past' % ref_point_value
        # ref_point_column = 'rel_distance'
        # now, for the sake of the neural plots, we'll just use 'rel_time'
        ref_point_column = 'rel_time'
        used_points_n_seconds_or_cm_ago = True
    elif ref_point_mode == 'time after cur ff visible':
        ref_point_descr = 'based on %d s ' % ref_point_value + \
            ref_point_mode[5:]
        ref_point_column = 'rel_time'
        used_points_n_seconds_or_cm_ago = True
    else:
        raise ValueError(
            'ref_point_mode must be either "time" or "distance" or "time after cur ff visible"')
    return ref_point_descr, ref_point_column, used_points_n_seconds_or_cm_ago
