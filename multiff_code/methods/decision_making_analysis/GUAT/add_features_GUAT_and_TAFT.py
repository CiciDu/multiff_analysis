from decision_making_analysis.decision_making import decision_making_utils

from pattern_discovery import cluster_analysis
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import curvature_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def retain_rows_in_df1_that_share_or_not_share_columns_with_df2(df1, df2, columns, whether_share=True):
    temp_df = df2[columns].copy()
    temp_df['share'] = True
    temp_df.drop_duplicates(inplace=True)
    df1 = pd.merge(df1, temp_df, on=columns, how='left')
    df1['share'] = df1['share'].fillna(False)
    if whether_share:
        df1 = df1[df1['share'] == True].drop(['share'], axis=1).copy()
    else:
        df1 = df1[df1['share'] != True].drop(['share'], axis=1).copy()
    return df1


def take_optimal_row_per_group_based_on_columns(df, columns, groupby_column='point_index'):
    '''
    This function takes the optimal row per group based on the columns specified.
    '''
    df = df.sort_values(columns, ascending=True)
    df = df.groupby(groupby_column).first().reset_index(drop=False)
    return df


def supply_info_of_cluster_to_df(df, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=50):

    ff_time = monkey_information.loc[df['point_index'].values, 'time'].values
    ff_cluster = cluster_analysis.find_alive_ff_clusters(ff_real_position_sorted[df['ff_index'].values], ff_real_position_sorted, ff_time-10, ff_time+10,
                                                         ff_life_sorted, max_distance=max_cluster_distance)
    ff_cluster_df = cluster_analysis.turn_list_of_ff_clusters_info_into_dataframe(
        ff_cluster, df['point_index'].values)
    # new_df = decision_making_utils.find_many_ff_info_anew(ff_cluster_df['ff_index'].values, ff_cluster_df['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    return ff_cluster_df


def find_GUAT_cur_ff_info(GUAT_w_ff_df, ff_real_position_sorted, ff_life_sorted, ff_dataframe, monkey_information, include_ff_in_near_future=False,
                          max_time_since_last_vis=2.5,
                          duration_into_future=0.5,
                          max_cluster_distance=50,
                          max_distance_to_stop=400,):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == True].copy()
    list_of_target_index = []
    list_of_ff_index = []
    list_of_point_index = []

    GUAT_w_ff_df = GUAT_w_ff_df.rename(columns={'point_index_of_eval': 'point_index',
                                                'time_of_eval': 'time', })

    for index, row in GUAT_w_ff_df.iterrows():
        if len(row.nearby_alive_ff_indices) > 0:
            for ff_index in row.nearby_alive_ff_indices:
                list_of_ff_index.append(ff_index)
                list_of_target_index.append(row.target_index)
                list_of_point_index.append(row.point_index)

    GUAT_cur_ff = pd.DataFrame({'ff_index': list_of_ff_index,
                                'point_index': list_of_point_index,
                                'target_index': list_of_target_index
                                })

    GUAT_cur_ff = GUAT_cur_ff.astype(int)

    GUAT_cur_ff = GUAT_cur_ff.merge(GUAT_w_ff_df[['target_index', 'point_index', 'time', 'num_stops', 'first_stop_point_index', 'total_stop_time']],
                                    on=['target_index', 'point_index'], how='left')

    GUAT_cur_ff_info = decision_making_utils.find_many_ff_info_anew(
        GUAT_cur_ff['ff_index'].values, GUAT_cur_ff['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    GUAT_cur_ff_info = GUAT_cur_ff_info.drop_duplicates(
        subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)
    GUAT_cur_ff_info = GUAT_cur_ff_info[GUAT_cur_ff_info['time_since_last_vis']
                                        <= max_time_since_last_vis]

    if include_ff_in_near_future:
        GUAT_cur_ff_info = supply_info_of_cluster_to_df(
            GUAT_cur_ff_info, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=max_cluster_distance)
        GUAT_cur_ff_info = decision_making_utils.find_many_ff_info_anew(
            GUAT_cur_ff_info['ff_index'].values, GUAT_cur_ff_info['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information, add_time_till_next_visible=True)

        # now we need to add back some columns
        columns_to_add_back = GUAT_cur_ff[[
            'point_index', 'num_stops', 'total_stop_time']].drop_duplicates()
        GUAT_cur_ff_info = pd.merge(GUAT_cur_ff_info, columns_to_add_back, on=[
            'point_index'], how='left')

        # either the ff has appeared lately, or will appear shortly
        GUAT_cur_ff_info = GUAT_cur_ff_info[(GUAT_cur_ff_info['time_since_last_vis'] <= max_time_since_last_vis) |
                                            (GUAT_cur_ff_info['time_till_next_visible'] <= GUAT_cur_ff_info['total_stop_time'] + duration_into_future)]

    # get distance_to_monkey
    GUAT_cur_ff_info = GUAT_cur_ff_info.merge(
        GUAT_cur_ff[['point_index', 'first_stop_point_index']], on='point_index', how='left')
    ff_x, ff_y = ff_real_position_sorted[GUAT_cur_ff_info['ff_index'].values].T
    monkey_x, monkey_y = monkey_information.loc[GUAT_cur_ff_info['first_stop_point_index'].values, [
        'monkey_x', 'monkey_y']].values.T
    GUAT_cur_ff_info['distance_to_monkey'] = np.sqrt(
        (ff_x - monkey_x)**2 + (ff_y - monkey_y)**2)
    # since distance_to_monkey's reference point is the stop
    GUAT_cur_ff_info['distance_to_stop'] = GUAT_cur_ff_info['distance_to_monkey']
    GUAT_cur_ff_info = GUAT_cur_ff_info[GUAT_cur_ff_info['distance_to_stop'] < max_distance_to_stop].copy(
    )

    GUAT_cur_ff_info = GUAT_cur_ff_info.drop_duplicates(
        subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)

    return GUAT_cur_ff_info


def find_GUAT_nxt_ff_info(GUAT_cur_ff_info, ff_dataframe, ff_real_position_sorted, monkey_information,
                          max_time_since_last_vis=2.5,
                          max_distance_to_stop=400,
                          duration_into_future=0.5,
                          include_ff_in_near_future=True):

    # find the info of the alternative ff that's not within the same cluster as the current ff
    GUAT_nxt_ff_info = ff_dataframe[ff_dataframe['point_index'].isin(
        GUAT_cur_ff_info['point_index'].values)].copy()
    GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['time_since_last_vis']
                                        <= max_time_since_last_vis]
    GUAT_nxt_ff_info = GUAT_nxt_ff_info[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary',
                                         'abs_ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index', 'abs_curv_diff']]
    GUAT_nxt_ff_info['distance_to_monkey'] = GUAT_nxt_ff_info['ff_distance']
    GUAT_nxt_ff_info['distance_to_stop'] = GUAT_nxt_ff_info['ff_distance']
    GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['distance_to_stop']
                                        < max_distance_to_stop].copy()

    # Also add more ff for ff that become available in the near future of each point_index
    if include_ff_in_near_future:
        unique_point_index_and_time_df = GUAT_cur_ff_info[[
            'point_index', 'time', 'total_stop_time']].drop_duplicates()
        ff_info, all_available_ff_in_near_future = find_additional_ff_info_for_near_future(unique_point_index_and_time_df, ff_dataframe, ff_real_position_sorted, monkey_information,
                                                                                           duration_into_future=duration_into_future)
        GUAT_nxt_ff_info = pd.concat(
            [GUAT_nxt_ff_info, ff_info], axis=0).reset_index(drop=True)

        # calculate the distance to the stop
        add_distance_to_stop(
            GUAT_nxt_ff_info, monkey_information, ff_real_position_sorted)
        GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['distance_to_stop']
                                            < max_distance_to_stop].copy()

        # use merge to add 'total_stop_time' to GUAT_nxt_ff_info
        GUAT_nxt_ff_info = GUAT_nxt_ff_info.merge(
            GUAT_cur_ff_info[['point_index', 'total_stop_time']], on='point_index', how='left')

        GUAT_nxt_ff_info.sort_values(
            by=['point_index', 'ff_index', 'distance_to_monkey'], inplace=True)
        GUAT_nxt_ff_info = GUAT_nxt_ff_info.drop_duplicates(
            subset=['point_index', 'ff_index'], keep='first').reset_index(drop=True)

    return GUAT_nxt_ff_info


def add_distance_to_stop(df, monkey_information, ff_real_position_sorted):
    stop_monkey_x, stop_monkey_y = monkey_information.loc[df['point_index'].values, [
        'monkey_x', 'monkey_y']].values.T
    ff_x, ff_y = ff_real_position_sorted[df['ff_index'].values].T
    df['distance_to_stop'] = np.sqrt(
        (stop_monkey_x - ff_x)**2 + (stop_monkey_y - ff_y)**2)


def retain_useful_cur_and_nxt_info(GUAT_cur_ff_info, GUAT_nxt_ff_info, eliminate_cases_with_close_nxt_ff=True,
                                   min_nxt_ff_distance_to_stop=0):
    # we need to eliminate the info of the ff in GUAT_nxt_ff_info that's also in GUAT_cur_ff_info at the same point indices
    # GUAT_nxt_ff_info = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(GUAT_nxt_ff_info, GUAT_cur_ff_info, columns=['point_index', 'ff_index'], whether_share=False)

    # then, we eliminate the cases where GUAT_nxt_ff_info has at least one ff that's within 150 to the current point, because that creates confounding factors
    if eliminate_cases_with_close_nxt_ff:
        GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['ff_distance']
                                            > min_nxt_ff_distance_to_stop].copy()

    # also eliminate nxt_ff if it's at the back of the monkey
    GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['ff_angle_boundary'].between(
        -90*math.pi/180, 90*math.pi/180)].copy()

    # then, we eliminate the info in GUAT_cur_ff_info that does not have corresponding info in GUAT_nxt_ff_info (with the same point_index)
    GUAT_cur_ff_info = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(
        GUAT_cur_ff_info, GUAT_nxt_ff_info, columns=['point_index'], whether_share=True)
    return GUAT_cur_ff_info, GUAT_nxt_ff_info


def make_sure_GUAT_nxt_ff_info_and_GUAT_cur_ff_info_have_the_same_point_indices(GUAT_cur_ff_info, GUAT_nxt_ff_info):
    GUAT_cur_ff_info = GUAT_cur_ff_info[GUAT_cur_ff_info['point_index'].isin(
        GUAT_nxt_ff_info['point_index'].values)].copy()
    GUAT_nxt_ff_info = GUAT_nxt_ff_info[GUAT_nxt_ff_info['point_index'].isin(
        GUAT_cur_ff_info['point_index'].values)].copy()
    return GUAT_cur_ff_info, GUAT_nxt_ff_info


def polish_GUAT_cur_ff_info(GUAT_cur_ff_info):
    GUAT_cur_ff_info = GUAT_cur_ff_info.sort_values(
        by=['point_index']).reset_index(drop=True)
    GUAT_cur_ff_info = GUAT_cur_ff_info[['num_stops', 'ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary', 'time_since_last_vis',
                                         'time_till_next_visible', 'duration_of_last_vis_period', 'point_index', 'ff_index', 'distance_to_monkey', 'total_stop_time']].copy()
    GUAT_cur_ff_info.sort_values(by=['point_index'], inplace=True)

    GUAT_cur_ff_info = _clip_time_since_last_vis_and_time_till_next_visible(
        GUAT_cur_ff_info)

    GUAT_cur_ff_info = _add_num_ff_in_cluster(GUAT_cur_ff_info)

    return GUAT_cur_ff_info


def polish_GUAT_nxt_ff_info(GUAT_nxt_ff_info, GUAT_cur_ff_info, ff_real_position_sorted, ff_life_sorted, ff_dataframe, monkey_information,
                            columns_to_sort_nxt_ff_by=['abs_curv_diff', 'time_since_last_vis'], max_cluster_distance=50,
                            max_time_since_last_vis=2.5, duration_into_future=0.5,
                            take_one_row_for_each_point_and_find_cluster=False):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == True].copy()
    if take_one_row_for_each_point_and_find_cluster:
        # take the optimal ff from GUAT_nxt_ff_info based on columns_to_sort_nxt_ff_by
        GUAT_nxt_ff_info = take_optimal_row_per_group_based_on_columns(
            GUAT_nxt_ff_info, columns_to_sort_nxt_ff_by, groupby_column='point_index')

        # now, let's re-find GUAT_nxt_ff_info by considering clusters
        GUAT_nxt_ff_info_old = GUAT_nxt_ff_info.copy()
        GUAT_nxt_ff_info = supply_info_of_cluster_to_df(
            GUAT_nxt_ff_info, ff_real_position_sorted, ff_life_sorted, monkey_information, max_cluster_distance=max_cluster_distance)
    else:
        GUAT_nxt_ff_info_old = GUAT_nxt_ff_info.copy()

    # find the info of additional columnsfor GUAT_nxt_ff_info
    GUAT_nxt_ff_info = decision_making_utils.find_many_ff_info_anew(
        GUAT_nxt_ff_info['ff_index'].values, GUAT_nxt_ff_info['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information, add_time_till_next_visible=True)

    GUAT_nxt_ff_info = GUAT_nxt_ff_info[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary',
                                         'time_since_last_vis', 'time_till_next_visible', 'duration_of_last_vis_period', 'point_index', 'ff_index']].copy()
    GUAT_nxt_ff_info = GUAT_nxt_ff_info.merge(GUAT_nxt_ff_info_old[[
                                              'point_index', 'ff_index', 'distance_to_monkey', 'total_stop_time']], on=['point_index', 'ff_index'], how='left')

    GUAT_nxt_ff_info = GUAT_nxt_ff_info[(GUAT_nxt_ff_info['time_since_last_vis'] <= max_time_since_last_vis) |
                                        (GUAT_nxt_ff_info['time_till_next_visible'] <= GUAT_nxt_ff_info['total_stop_time'] + duration_into_future)]  # either the ff has appeared lately, or will appear shortly

    # # since we just added cluster ff, once again we need to eliminate the info of the ff in GUAT_nxt_ff_info that's also in GUAT_cur_ff_info at the same point indices
    # GUAT_nxt_ff_info = retain_rows_in_df1_that_share_or_not_share_columns_with_df2(GUAT_nxt_ff_info, GUAT_cur_ff_info, columns=['point_index', 'ff_index'], whether_share=False)

    GUAT_nxt_ff_info.sort_values(by=['point_index'], inplace=True)

    GUAT_nxt_ff_info = _clip_time_since_last_vis_and_time_till_next_visible(
        GUAT_nxt_ff_info)

    GUAT_nxt_ff_info = _add_num_ff_in_cluster(GUAT_nxt_ff_info)

    return GUAT_nxt_ff_info


def _clip_time_since_last_vis_and_time_till_next_visible(df, max_time_since_last_vis=5, max_time_till_next_visible=5):
    df.loc[df['time_since_last_vis'] > max_time_since_last_vis,
           'time_since_last_vis'] = max_time_since_last_vis
    df.loc[df['time_till_next_visible'] > max_time_till_next_visible,
           'time_till_next_visible'] = max_time_till_next_visible
    return df


def _add_num_ff_in_cluster(df):
    num_ff_in_cluster = df.groupby(
        'point_index').size().reset_index(drop=False)
    num_ff_in_cluster.columns = ['point_index', 'num_ff_in_cluster']
    df = df.merge(num_ff_in_cluster, on='point_index', how='left')
    return df


def add_curv_diff_and_ff_number_to_GUAT_cur_ff_info_and_GUAT_nxt_ff_info(GUAT_cur_ff_info, GUAT_nxt_ff_info, ff_caught_T_new, ff_real_position_sorted, monkey_information, curv_of_traj_df=None,
                                                                         ff_priority_criterion='abs_curv_diff'):

    # Note: in order not to feed the input with additional data, we will let curv_of_traj_df = curv_of_traj_df
    GUAT_cur_ff_info = decision_making_utils.add_curv_diff_to_df(
        GUAT_cur_ff_info, monkey_information, curv_of_traj_df, ff_real_position_sorted=ff_real_position_sorted)
    GUAT_nxt_ff_info = decision_making_utils.add_curv_diff_to_df(
        GUAT_nxt_ff_info, monkey_information, curv_of_traj_df, ff_real_position_sorted=ff_real_position_sorted)

    GUAT_cur_ff_info.sort_values(
        by=['point_index', ff_priority_criterion], inplace=True)
    GUAT_nxt_ff_info.sort_values(
        by=['point_index', ff_priority_criterion], inplace=True)

    # assign a ff_number to each ff within each point_index
    GUAT_cur_ff_info['ff_number'] = GUAT_cur_ff_info.groupby(
        'point_index').cumcount()+1
    GUAT_nxt_ff_info['ff_number'] = GUAT_nxt_ff_info.groupby(
        'point_index').cumcount()+101

    return GUAT_cur_ff_info, GUAT_nxt_ff_info


def find_additional_ff_info_for_near_future(unique_point_index_and_time_df, ff_dataframe, ff_real_position_sorted, monkey_information, duration_into_future=0.5,
                                            add_distance_to_monkey=True):
    all_available_ff_in_near_future = find_available_ff_in_near_future(
        unique_point_index_and_time_df, ff_dataframe, duration_into_future=duration_into_future)

    if add_distance_to_monkey:
        original_all_available_ff_in_near_future = all_available_ff_in_near_future.copy()

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == True].copy()
    ff_info = decision_making_utils.find_many_ff_info_anew(
        all_available_ff_in_near_future.ff_index.values, all_available_ff_in_near_future.point_index.values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    ff_info = ff_info[ff_info.ff_angle_boundary.between(
        -90*math.pi/180, 90*math.pi/180)]
    ff_info = ff_info[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary',
                       'abs_ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']]

    if add_distance_to_monkey:
        ff_info = ff_info.merge(original_all_available_ff_in_near_future[[
                                'point_index', 'ff_index', 'distance_to_monkey']], on=['point_index', 'ff_index'], how='left')
    return ff_info, all_available_ff_in_near_future


def find_available_ff_in_near_future(unique_point_index_and_time_df, ff_dataframe, duration_into_future=0.5):
    ff_dataframe = ff_dataframe.copy()
    ff_dataframe['ff_starting_visible_time'] = ff_dataframe['time'] + \
        ff_dataframe['time_since_last_vis']
    all_available_ff_in_near_future = pd.DataFrame(
        [], columns=['ff_index', 'point_index'])
    for index, row in unique_point_index_and_time_df.iterrows():
        # find a duration of N seconds starting from the point_index and take out the subset from ff_dataframe_visible
        point_index = row['point_index']
        time = row['time']
        duration = [time, time + row['total_stop_time'] + duration_into_future]
        # take out the ff that are visible during this duration
        ff_dataframe_subset0 = ff_dataframe[(ff_dataframe['ff_starting_visible_time'] >= duration[0]) & (
            ff_dataframe['ff_starting_visible_time'] <= duration[1])].copy()
        selected_ff = ff_dataframe_subset0['ff_index'].unique()
        # for these ff, take out the info between the duration and for 5s more beyond the duration, to calculate their closest distance to monkey's trajectory
        ff_dataframe_subset = ff_dataframe[ff_dataframe['time'].between(
            duration[0], duration[1] + 5)].copy()
        ff_dataframe_subset = ff_dataframe_subset[ff_dataframe_subset['ff_index'].isin(
            selected_ff)].copy()
        ff_dataframe_subset['distance_to_monkey'] = ff_dataframe_subset['ff_distance']
        ff_dataframe_subset.sort_values(
            by=['ff_index', 'distance_to_monkey'], inplace=True)
        available_ff = ff_dataframe_subset.groupby(
            'ff_index').first().reset_index()
        if len(available_ff) > 0:
            available_ff_sub = available_ff[['ff_index', 'ff_distance']].copy()
            available_ff_sub['point_index'] = point_index
            all_available_ff_in_near_future = pd.concat(
                [all_available_ff_in_near_future, available_ff_sub], axis=0)
    all_available_ff_in_near_future['ff_index'] = all_available_ff_in_near_future['ff_index'].astype(
        int)
    all_available_ff_in_near_future['point_index'] = all_available_ff_in_near_future['point_index'].astype(
        int)
    all_available_ff_in_near_future['distance_to_monkey'] = all_available_ff_in_near_future['ff_distance']
    return all_available_ff_in_near_future


def find_curv_diff_for_ff_info(ff_info, monkey_information, ff_real_position_sorted, curv_of_traj_df=None):
    if curv_of_traj_df is None:
        raise ValueError(
            'curv_of_traj_df is None. Please provide curv_of_traj_df.')
    ff_info_temp = ff_info.groupby(
        ['point_index', 'ff_index']).first().reset_index(drop=False).copy()
    ff_info_temp['monkey_x'] = monkey_information.loc[ff_info_temp['point_index'], 'monkey_x'].values
    ff_info_temp['monkey_y'] = monkey_information.loc[ff_info_temp['point_index'], 'monkey_y'].values
    ff_info_temp['monkey_angle'] = monkey_information.loc[ff_info_temp['point_index'],
                                                          'monkey_angle'].values
    ff_info_temp['ff_x'] = ff_real_position_sorted[ff_info_temp['ff_index'].values, 0]
    ff_info_temp['ff_y'] = ff_real_position_sorted[ff_info_temp['ff_index'].values, 1]
    temp_curvature_df = curvature_utils.make_curvature_df(
        ff_info_temp, curv_of_traj_df, ff_radius_for_opt_arc=10)
    temp_curvature_df.loc[:, 'curv_diff'] = temp_curvature_df['opt_arc_curv'].values - \
        temp_curvature_df['curv_of_traj'].values
    # temp_curvature_df.loc[:,'abs_curv_diff'] = np.abs(temp_curvature_df.loc[:,'curv_diff'])
    if 'curv_diff' in ff_info.columns:
        ff_info.drop(['curv_diff'], axis=1, inplace=True)
    ff_info = ff_info.merge(temp_curvature_df[['ff_index', 'point_index', 'curv_diff']].drop_duplicates(), on=[
                            'ff_index', 'point_index'], how='left')
    ff_info['abs_curv_diff'] = np.abs(ff_info['curv_diff'].values)
    return ff_info, temp_curvature_df


def update_point_index_of_important_df_in_important_info_func(important_info, new_point_index_start, point_index_column_name='point_index'):
    # take one random df from important_info to be used as a reference
    for df_name, df in important_info.items():
        break

    for df_name, df in important_info.items():
        print(df_name)
        print(df['point_index'].unique().shape)

    old_point_index_column_name = 'old_' + point_index_column_name
    if old_point_index_column_name in df.columns:
        point_index_all = df[old_point_index_column_name].values
    else:
        point_index_all = df[point_index_column_name].values
    unique_point_index = np.unique(point_index_all)

    point_index_to_new_number_df = pd.DataFrame({point_index_column_name: unique_point_index, 'new_number': range(
        new_point_index_start, new_point_index_start+len(unique_point_index))})
    # print('The following are the point indices and their new numbers:')
    # print(point_index_to_new_number_df)
    point_index_to_new_number_df.set_index(
        point_index_column_name, inplace=True)
    for df_name, df in important_info.items():
        # then the df has not been updated before
        if old_point_index_column_name not in important_info[df_name].columns:
            important_info[df_name][old_point_index_column_name] = df[point_index_column_name].copy(
            )
        else:  # the df has been updated before, so we need to restore the original state first in order to reapply the update with the current new_point_index_start
            important_info[df_name][point_index_column_name] = important_info[df_name][old_point_index_column_name].copy()
        important_info[df_name][point_index_column_name] = point_index_to_new_number_df.loc[df[point_index_column_name].values, 'new_number'].values
    return important_info, point_index_to_new_number_df


def find_possible_objects_of_pursuit(all_relevant_indices, ff_dataframe, max_distance_to_stop_for_GUAT_target=50,
                                     max_allowed_time_since_last_vis=2.5):
    # find corresponding info in ff_dataframe at time (in-memory ff and visible ff)
    ff_info = ff_dataframe.loc[ff_dataframe['point_index'].isin(
        all_relevant_indices)].copy()
    ff_info = ff_info[ff_info['time_since_last_vis']
                      <= max_allowed_time_since_last_vis]

    # among them, find ff close to monkey's position (within max_distance_to_stop_for_GUAT_target to the center of the ff), all of them can be possible targets
    ff_info = ff_info[ff_info['ff_distance'] <
                      max_distance_to_stop_for_GUAT_target].copy()

    return ff_info


def find_ff_aimed_at_through_manual_annotation(relevant_indices, monkey_information, manual_anno):
    # try to find confirmation from manual_anno by finding monkey's target at that period
    # see if for 2.5 s before the earliest stop was made, monkey was aiming at the ff
    manual_anno_sub = manual_anno[manual_anno['starting_point_index'] <= max(
        relevant_indices)]
    manual_anno_sub_during_interval = manual_anno_sub[manual_anno_sub['starting_point_index'].between(
        min(relevant_indices), max(relevant_indices))]
    # since monkey might have been continuously chasing some ff at the beginning of the interval defined by relevant_indices, we need to find the last ff that monkey was aiming at before the interval
    # take the row with the largest starting_point_index
    manual_anno_sub_earlier = manual_anno_sub[manual_anno_sub['starting_point_index'] < min(
        relevant_indices)]
    if len(manual_anno_sub_earlier) > 0:
        manual_anno_sub_earlier = manual_anno_sub_earlier[manual_anno_sub_earlier['starting_point_index'] == max(
            manual_anno_sub_earlier.starting_point_index.values)]
        manual_anno_sub = pd.concat(
            [manual_anno_sub_earlier, manual_anno_sub_during_interval])
    else:
        manual_anno_sub = manual_anno_sub_during_interval

    # make sure the info is not too old
    min_time_from_relevant_indices = monkey_information.time.values[relevant_indices.min(
    )]
    if manual_anno_sub.time.max() > min_time_from_relevant_indices - 4:
        ff_aimed_at = manual_anno_sub.ff_index.values
    else:
        ff_aimed_at = np.array([])

    return ff_aimed_at


def find_alive_ff_close_to_monkey(relevant_indices, monkey_information, ff_life_sorted, ff_real_position_sorted, max_ff_distance_to_include=75):
    # find alive ff that are close to monkey at certain indices
    close_ff_indices = np.array([], dtype=int)
    monkey_xy = monkey_information[['monkey_x', 'monkey_y']].values
    duration = [monkey_information['time'].loc[min(relevant_indices)].item(
    ), monkey_information['time'].loc[max(relevant_indices)].item()]

    # take out alive ff during the trial
    alive_ff_indices, alive_ff_positions = plot_behaviors_utils.find_alive_ff(
        duration, ff_life_sorted, ff_real_position_sorted, rotation_matrix=None)
    alive_ff_positions = alive_ff_positions.T

    # sample 5 indices from relevant_indices using linspace
    sampled_indices = relevant_indices[np.linspace(
        0, len(relevant_indices)-1, 5, dtype=int, endpoint=True)]

    # find monkey postions of these 5 indices
    sampled_monkey_positions = monkey_xy[sampled_indices, :]

    # find their distance to each stop
    for position in sampled_monkey_positions:
        ff_distance = np.linalg.norm(alive_ff_positions-position, axis=1)
        close_ff_indices = np.concatenate(
            [close_ff_indices, alive_ff_indices[ff_distance < max_ff_distance_to_include]])

    # take out those that are close to at least one stop
    close_ff_indices = np.unique(close_ff_indices)
    return close_ff_indices
