from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils

import os
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


def streamline_getting_one_stop_df(monkey_information, ff_dataframe, ff_caught_T_new, min_distance_from_adjacent_stops=75,
                                   min_cum_distance_to_ff_capture=100, min_distance_to_ff=25, max_distance_to_ff=75):
    distinct_stops_df = monkey_information[monkey_information['whether_new_distinct_stop'] == 1].copy(
    )
    filtered_stops_df = get_filtered_stops_df(
        distinct_stops_df, min_distance_from_adjacent_stops)
    filtered_stops_df = filter_stops_based_on_distance_to_ff_capture(
        filtered_stops_df, monkey_information, ff_caught_T_new, min_cum_distance_to_ff_capture)
    one_stop_df = get_one_stop_df(
        filtered_stops_df, ff_dataframe, min_distance_to_ff, max_distance_to_ff)

    return one_stop_df


def make_one_stop_w_ff_df(one_stop_df):

    # group one_stop_df by 'point_index' and make a list out of all ff_indices for each point_index
    one_stop_df.sort_values(by=['point_index', 'time_since_last_vis'], ascending=[
                            True, True], inplace=True)
    ff_around_stop_df = one_stop_df[['point_index', 'ff_index']].groupby(
        'point_index')['ff_index'].apply(list).reset_index(drop=False)
    ff_around_stop_df['latest_visible_ff'] = one_stop_df.groupby('point_index')[
        'ff_index'].first().values
    one_stop_w_ff_df = ff_around_stop_df.merge(one_stop_df[['target_index', 'time', 'point_index', 'ff_distance',
                                                            'closest_cum_distance_to_ff_capture', 'min_distance_from_adjacent_stops',
                                                            ]].drop_duplicates(), on='point_index', how='left')
    one_stop_w_ff_df[['num_stops', 'whether_w_ff_near_stops']] = 1
    one_stop_w_ff_df['trial'] = one_stop_w_ff_df['target_index']
    one_stop_w_ff_df['stop_indices'] = one_stop_w_ff_df['point_index'].apply(lambda x: [
                                                                             x])
    one_stop_w_ff_df.rename(columns={'ff_index': 'nearby_alive_ff_indices',
                                     'point_index': 'first_stop_point_index',
                                     'time': 'first_stop_time'}, inplace=True)

    for col in one_stop_w_ff_df.columns:
        if '_index' in col:
            one_stop_w_ff_df[col] = one_stop_w_ff_df[col].astype('int64')

    return one_stop_w_ff_df


# def get_distinct_stops_df(monkey_information, min_distance_between_distinct_stops=15):
#     # we need to get distinct stop point_index because sometimes a few point index can indicate the same stop

#     stop_points_df = monkey_information[monkey_information['monkey_speeddummy'] == 0].copy()

#     # take out stops that are not too close to previous stop points
#     stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance'].diff()
#     stop_points_df['cum_distance_from_last_stop_point'] = stop_points_df['cum_distance_from_last_stop_point'].fillna(100)

#     distinct_stops_df = stop_points_df[stop_points_df['cum_distance_from_last_stop_point'] > min_distance_between_distinct_stops]

#     return distinct_stops_df


def get_filtered_stops_df(distinct_stops_df, min_distance_from_adjacent_stops):
    # take out stops that are not within min_distance_from_adjacent_stops of any other stop
    delta_x_from_last_stop = distinct_stops_df['monkey_x'].diff().fillna(
        min_distance_from_adjacent_stops * 2)
    delta_y_from_last_stop = distinct_stops_df['monkey_y'].diff().fillna(
        min_distance_from_adjacent_stops * 2)
    delta_x_from_next_stop = - \
        distinct_stops_df['monkey_x'].diff(-1).fillna(
            min_distance_from_adjacent_stops * 2)
    delta_y_from_next_stop = - \
        distinct_stops_df['monkey_y'].diff(-1).fillna(
            min_distance_from_adjacent_stops * 2)
    distinct_stops_df['distance_from_last_stop'] = np.sqrt(
        delta_x_from_last_stop ** 2 + delta_y_from_last_stop ** 2)
    distinct_stops_df['distance_from_next_stop'] = np.sqrt(
        delta_x_from_next_stop ** 2 + delta_y_from_next_stop ** 2)
    distinct_stops_df['min_distance_from_adjacent_stops'] = distinct_stops_df[[
        'distance_from_last_stop', 'distance_from_next_stop']].min(axis=1)
    filtered_stops_df = distinct_stops_df[(
        distinct_stops_df['min_distance_from_adjacent_stops'] > min_distance_from_adjacent_stops)].copy()

    return filtered_stops_df


def filter_stops_based_on_distance_to_ff_capture(filtered_stops_df, monkey_information, ff_caught_T_new, min_cum_distance_to_ff_capture):
    # eliminate the stops that are too close to a ff capture (within min_cum_distance_to_ff_capture)

    # first find the corresponding point index of each time point in ff_caught_T_new
    ff_caught_points_sorted = np.searchsorted(
        monkey_information['time'].values, ff_caught_T_new)
    ff_caught_points_df = monkey_information.iloc[ff_caught_points_sorted].copy(
    )

    # for each value in filtered_stops_df's cum_distance column, find the closest cum_distance in ff_caught_points
    filtered_stops_df['closest_cum_distance_to_ff_capture'] = filtered_stops_df['cum_distance'].apply(
        lambda x: np.abs(ff_caught_points_df['cum_distance'].values - x).min())
    # then, eliminate the stops that are too close to a capture
    filtered_stops_df = filtered_stops_df[filtered_stops_df['closest_cum_distance_to_ff_capture']
                                          > min_cum_distance_to_ff_capture].copy()

    return filtered_stops_df


def get_one_stop_df(filtered_stops_df, ff_dataframe, min_distance_to_ff=25, max_distance_to_ff=50):
    # one_stop_df is used as a comparison to GUAT where there are at least 2 stops beside a missed ff.
    # one_stop_df contains instances where there is one stop beside a missed ff.

    # from the filtered stops, take out the stops that are within n cm of a ff that's visible or in-memory (use ff_dataframe to check)

    # drop the columns in ff_dataframe that are also in one_stop_df except for 'point_index'
    ff_dataframe_temp = ff_dataframe.drop(columns=ff_dataframe.columns.intersection(
        filtered_stops_df.columns).difference(['point_index']))

    one_stop_df = filtered_stops_df.merge(
        ff_dataframe_temp, on='point_index', how='left')

    # eliminate point_index in one_stop_df if there's at least one row in one_stop_df that has min ff_distance < min_distance_to_ff
    grouped_df = one_stop_df.groupby('point_index').min()
    point_index_to_eliminate = grouped_df[grouped_df['ff_distance']
                                          < min_distance_to_ff].index.values
    one_stop_df = one_stop_df[~one_stop_df['point_index'].isin(
        point_index_to_eliminate)].copy()

    # eliminate rows where ff is too far away from the stop (thus, some point indices will be completely eliminated, while others that have at least one row with ff_distance < max_distance_to_ff will remain)
    one_stop_df = one_stop_df[one_stop_df['ff_distance']
                              <= max_distance_to_ff].copy()
    return one_stop_df


def get_GUAT_w_ff_df(GUAT_indices_df,
                     GUAT_trials_df,
                     ff_dataframe,
                     monkey_information,
                     ff_real_position_sorted,
                     max_distance_to_stop_for_GUAT_target=50,
                     max_allowed_time_since_last_vis=2.5):

    GUAT_df = GUAT_indices_df[['point_index', 'cluster_index']].copy()

    GUAT_ff_info = GUAT_df.merge(ff_dataframe, on='point_index', how='left')
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['time_since_last_vis']
                                <= max_allowed_time_since_last_vis]
    # among them, find ff close to monkey's position (within max_distance_to_stop_for_GUAT_target to the center of the ff), all of them can be possible targets
    GUAT_ff_info = GUAT_ff_info[GUAT_ff_info['ff_distance']
                                < max_distance_to_stop_for_GUAT_target].copy()

    # group GUAT_ff_info by cluster_index so that ff_index becomes a list of ff_indices for each cluster
    GUAT_ff_info2 = GUAT_ff_info[['cluster_index', 'ff_index']].drop_duplicates(
    ).groupby('cluster_index')['ff_index'].apply(list).reset_index(drop=False)
    GUAT_ff_info2.rename(
        columns={'ff_index': 'nearby_alive_ff_indices'}, inplace=True)

    # also get latest visible ff
    GUAT_ff_info.sort_values(by=['cluster_index', 'time_since_last_vis'], ascending=[
                             True, True], inplace=True)
    GUAT_ff_info2['latest_visible_ff'] = GUAT_ff_info.groupby('cluster_index')[
        'ff_index'].first().values

    GUAT_expanded_trials_df = GUAT_trials_df.merge(
        GUAT_ff_info2, on='cluster_index', how='left')
    GUAT_expanded_trials_df.sort_values(by='cluster_index', inplace=True)
    # mark whether_w_ff_near_stops as 1 if nearby_alive_ff_indices is not NA
    GUAT_expanded_trials_df['whether_w_ff_near_stops'] = (
        ~GUAT_expanded_trials_df['nearby_alive_ff_indices'].isna()).values.astype(int)
    GUAT_w_ff_df = GUAT_expanded_trials_df[GUAT_expanded_trials_df['whether_w_ff_near_stops'] == 1].reset_index(
        drop=True)
    GUAT_w_ff_df['target_index'] = GUAT_w_ff_df['trial']
    GUAT_w_ff_df['latest_visible_ff'] = GUAT_w_ff_df['latest_visible_ff'].astype(
        'int64')

    GUAT_w_ff_df.sort_values(by=['trial', 'first_stop_time'], inplace=True)
    GUAT_w_ff_df['ff_index'] = GUAT_w_ff_df['latest_visible_ff']
    GUAT_vs_TAFT_utils.add_stop_point_index(
        GUAT_w_ff_df, monkey_information, ff_real_position_sorted)
    GUAT_w_ff_df = GUAT_vs_TAFT_utils.deal_with_duplicated_stop_point_index(
        GUAT_w_ff_df)

    return GUAT_w_ff_df, GUAT_expanded_trials_df
