from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.get_cur_vs_nxt_ff_data import find_cvn_utils
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils
from data_wrangling import specific_utils
from planning_analysis.plan_factors import build_factor_comp_utils, build_factor_comp, feature_lists
import numpy as np
import pandas as pd
import math


def make_plan_y_df(heading_info_df, curv_of_traj_df, curv_of_traj_df_w_one_sided_window):

    plan_y_df = build_factor_comp.process_heading_info_df(
        heading_info_df)

    curv_of_traj_stat_df = build_factor_comp.find_curv_of_traj_stat_df(
        heading_info_df, curv_of_traj_df)
    plan_y_df = build_factor_comp_utils._add_stat_columns_to_df(
        curv_of_traj_stat_df, plan_y_df, ['curv'], 'stop_point_index')

    if curv_of_traj_df_w_one_sided_window is not None:
        plan_y_df = build_factor_comp.add_column_curv_of_traj_before_stop(
            plan_y_df, curv_of_traj_df_w_one_sided_window)

    build_factor_comp.add_dir_from_cur_ff_same_side(plan_y_df)

    plan_y_df = plan_y_df.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_y_df


def make_plan_x_df(stops_near_ff_df, heading_info_df, both_ff_at_ref_df, ff_dataframe, monkey_information, ff_real_position_sorted,
                   stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150, ff_radius=10,
                   list_of_cur_ff_cluster_radius=[100, 200, 300],
                   list_of_nxt_ff_cluster_radius=[100, 200, 300],
                   use_speed_data=False, use_eye_data=False,
                   guarantee_cur_ff_info_for_cluster=False,
                   guarantee_nxt_ff_info_for_cluster=False,
                   columns_not_to_include=[],
                   flash_or_vis='vis',
                   ):

    plan_x_df = both_ff_at_ref_df.copy()

    # get information between two stops
    info_between_two_stops = build_factor_comp.get_info_between_two_stops(
        stops_near_ff_df)
    plan_x_df = plan_x_df.merge(
        info_between_two_stops, on='stop_point_index', how='left')

    # get distance_between_stop_and_arena_edge
    plan_x_df['distance_between_stop_and_arena_edge'] = build_factor_comp.get_distance_between_stop_and_arena_edge(
        stops_near_ff_df)

    # Get cluster information for plan_x. Example output columns: cur_ff_cluster_100_EARLIEST_VIS_ff_distance
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    cluster_df = build_factor_comp.make_cluster_df_as_part_of_plan_factors(stops_near_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted,
                                                                           stop_period_duration=stop_period_duration, ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, ff_radius=ff_radius,
                                                                           list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                                           columns_not_to_include=columns_not_to_include,
                                                                           guarantee_cur_ff_info_for_cluster=guarantee_cur_ff_info_for_cluster,
                                                                           guarantee_nxt_ff_info_for_cluster=guarantee_nxt_ff_info_for_cluster,
                                                                           flash_or_vis=flash_or_vis,
                                                                           )
    plan_x_df = plan_x_df.merge(
        cluster_df, on='stop_point_index', how='left').reset_index(drop=True)

    # nxt_ff_last_seen_info = build_factor_comp.get_nxt_ff_last_seen_info_before_next_stop(nxt_ff_df_from_ref, ff_dataframe_visible, monkey_information,
    #                                                                     stops_near_ff_df, ff_real_position_sorted)
    # plan_x_df = pd.concat([plan_x_df, nxt_ff_last_seen_info])

    if use_speed_data:
        plan_x_df = build_factor_comp.add_monkey_speed_stats_to_df(
            plan_x_df, stops_near_ff_df, monkey_information)

    if use_eye_data:
        plan_x_df = build_factor_comp.add_monkey_eye_stats_to_df(
            plan_x_df, stops_near_ff_df, monkey_information)

    # only keep the rows with stop_point_index that are in heading_info_df
    plan_x_df = plan_x_df[plan_x_df['stop_point_index'].isin(
        heading_info_df['stop_point_index'])].copy()
    plan_x_df = plan_x_df.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_x_df


def drop_columns_that_contain_both_nxt_and_bbas(plan_y_tc):
    # Drop columns in self.plan_y_tc that contain both 'nxt'/'NXT' and 'bbas'
    columns_to_drop = [
        col for col in plan_y_tc.columns
        if 'bbas' in col.lower() and ('nxt' in col.lower())
    ]

    if columns_to_drop:
        plan_y_tc.drop(columns=columns_to_drop, inplace=True)
        print(
            f"Dropped {len(columns_to_drop)} columns containing both 'nxt'/'NXT' and 'bbas': {columns_to_drop}")


def process_plan_x_to_predict_monkey_info(plan_x, for_classification=False):
    plan_x['curv_range'] = plan_x['curv_max'] - plan_x['curv_min']
    plan_x['curv_iqr'] = plan_x['curv_Q3'] - plan_x['curv_Q1']

    non_cluster_columns_to_save = [
        'distance_between_stop_and_arena_edge'] + feature_lists.cur_ff_at_ref_columns + feature_lists.nxt_ff_at_ref_columns \
        + feature_lists.all_eye_features + feature_lists.trajectory_features

    cluster_columns_to_save = [col for col in plan_x.columns if (
        col in non_cluster_columns_to_save) | ('cluster' in col)]

    if for_classification:
        if 'dir_from_cur_ff_to_nxt_ff' in plan_x.columns:
            cluster_columns_to_save.append('dir_from_cur_ff_to_nxt_ff')

    plan_x = plan_x[cluster_columns_to_save].copy()
    return plan_x


def process_plan_x_to_predict_ff_info(plan_x, plan_y):

    non_cluster_columns_to_save = ['distance_between_stop_and_arena_edge'] + feature_lists.cur_ff_at_ref_columns + feature_lists.nxt_ff_at_ref_columns \
        + feature_lists.all_eye_features + feature_lists.trajectory_features + \
        feature_lists.traj_to_cur_ff_features

    # delete 'curv_range' and 'curv_iqr' from non_cluster_columns_to_save
    non_cluster_columns_to_save.remove('curv_range')
    non_cluster_columns_to_save.remove('curv_iqr')

    cluster_columns_to_save = [col for col in plan_x.columns if (
        col in non_cluster_columns_to_save) | ('cluster' in col)]
    # delete any column that contains 'nxt'
    cluster_columns_to_save = [
        col for col in cluster_columns_to_save if 'nxt' not in col]

    plan_x = plan_x[cluster_columns_to_save].copy()

    # add columns
    columns_from_plan_y_to_add = ['angle_from_cur_ff_to_stop',
                                  'diff_in_d_heading_to_cur_ff',
                                  'curv_of_traj_before_stop',
                                  'dir_from_cur_ff_to_stop']

    plan_x[columns_from_plan_y_to_add] = plan_y[columns_from_plan_y_to_add].values.copy()

    plan_x = plan_x[cluster_columns_to_save].copy()

    return plan_x


def delete_monkey_info_in_plan_x(plan_x):
    columns_to_drop = feature_lists.all_eye_features + \
        feature_lists.trajectory_features + feature_lists.traj_to_cur_ff_features
    # delete 'curv_range'
    columns_to_drop.remove('curv_range')
    plan_x = plan_x.drop(columns=columns_to_drop, errors='ignore')

    # columns_to_preserve = [col for col in plan_x.columns if ('cluster' in col) | ('ref' in col)]
    # plan_x = plan_x[columns_to_preserve].copy()
    return plan_x


def make_plan_xy_test_and_plan_xy_ctrl(plan_x_tc, plan_y_tc):
    # concat plan_x_tc and plan_y_tc but drop duplicated columns
    plan_xy = pd.concat([plan_x_tc, plan_y_tc], axis=1)
    # drop duplicated columns
    plan_xy = plan_xy.loc[:, ~plan_xy.columns.duplicated()]

    plan_xy_test = plan_xy[plan_xy['whether_test']
                           == 1].reset_index(drop=True).copy()
    plan_xy_ctrl = plan_xy[plan_xy['whether_test']
                           == 0].reset_index(drop=True).copy()
    return plan_xy_test, plan_xy_ctrl


def quickly_process_plan_xy_test_and_ctrl(plan_xy_test, plan_xy_ctrl, column_for_split, whether_filter_info, finalized_params):
    test_and_ctrl_df = pd.concat([plan_xy_test, plan_xy_ctrl], axis=0)
    ctrl_df = test_and_ctrl_df[test_and_ctrl_df[column_for_split].isnull()].copy(
    )
    test_df = test_and_ctrl_df[~test_and_ctrl_df[column_for_split].isnull()].copy(
    )

    if whether_filter_info:
        test_df, ctrl_df = test_vs_control_utils.filter_both_df(
            test_df, ctrl_df, **finalized_params)

    return test_df, ctrl_df


def add_d_heading_of_traj_to_df(df):
    df['d_heading_of_traj'] = df['monkey_angle_before_stop'] - df['monkey_angle']
    df['d_heading_of_traj'] = find_cvn_utils.confine_angle_to_within_one_pie(
        df['d_heading_of_traj'].values)
    return df
