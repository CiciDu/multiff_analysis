
from data_wrangling import specific_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import plot_cvn_utils, find_cvn_utils
from null_behaviors import curv_of_traj_utils, curvature_utils

import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
import math
import os
import sys


def compute_cur_end_to_next_ff_curv(nxt_ff_df_modified, heading_info_df,
                                    use_curv_to_ff_center=False,
                                    ff_radius_for_opt_arc=10):

    df = _prepare_cur_end_to_next_ff_data(
        heading_info_df, nxt_ff_df_modified)
    mock_monkey_info = _build_mock_monkey_info(
        df, use_curv_to_ff_center=use_curv_to_ff_center)
    cur_end_to_next_ff_curv = _compute_curv_from_cur_end(mock_monkey_info,
                                                         ff_radius_for_opt_arc=ff_radius_for_opt_arc)
    assert np.all(
        cur_end_to_next_ff_curv['nxt_ff_index'] == heading_info_df['nxt_ff_index'])
    cur_end_to_next_ff_curv['ref_point_index'] = heading_info_df['ref_point_index'].values
    return cur_end_to_next_ff_curv


def _prepare_cur_end_to_next_ff_data(heading_info_df, nxt_ff_df_modified):
    # Select relevant columns from heading_info_df
    heading_cols = [
        'ref_point_index', 'nxt_ff_index',
        'cur_cntr_arc_end_x', 'cur_cntr_arc_end_y', 'cur_cntr_arc_end_heading', 'cur_cntr_arc_curv',
        'cur_opt_arc_end_x', 'cur_opt_arc_end_y', 'cur_opt_arc_end_heading', 'cur_opt_arc_curv'
    ]
    df = heading_info_df[heading_cols].copy()
    df.rename(columns={'ref_point_index': 'point_index'}, inplace=True)

    # Select and rename relevant columns from nxt_ff_df_modified
    nxt_ff_cols = [
        'point_index',
        'ff_index', 'ff_x', 'ff_y', 'ff_distance', 'ff_angle', 'ff_angle_boundary'
    ]
    nxt_ff_df = nxt_ff_df_modified[nxt_ff_cols].copy()
    nxt_ff_df.rename(columns={
        'ff_index': 'nxt_ff_index',
        'ff_x': 'nxt_ff_x',
        'ff_y': 'nxt_ff_y',
        'ff_distance': 'nxt_ff_distance',
        'ff_angle': 'nxt_ff_angle',
        'ff_angle_boundary': 'nxt_ff_angle_boundary',
    }, inplace=True)

    # Merge on 'ref_point_index' and rename it to 'point_index'
    merged_df = df.merge(
        nxt_ff_df, on=['point_index', 'nxt_ff_index'], how='left')

    return merged_df


def _build_mock_monkey_info(df, use_curv_to_ff_center=False):
    mock_monkey_info = df.copy()

    if use_curv_to_ff_center:
        mock_monkey_info.rename(columns={'cur_cntr_arc_end_x': 'monkey_x',
                                         'cur_cntr_arc_end_y': 'monkey_y',
                                         'cur_cntr_arc_end_heading': 'monkey_angle',
                                         'cur_cntr_arc_curv': 'curv_of_traj'
                                         }, inplace=True)
    else:
        mock_monkey_info.rename(columns={'cur_opt_arc_end_x': 'monkey_x',
                                         'cur_opt_arc_end_y': 'monkey_y',
                                         'cur_opt_arc_end_heading': 'monkey_angle',
                                         'cur_opt_arc_curv': 'curv_of_traj'
                                         }, inplace=True)

    mock_monkey_info.rename(columns={'nxt_ff_index': 'ff_index',
                                     'nxt_ff_x': 'ff_x',
                                     'nxt_ff_y': 'ff_y'}, inplace=True)

    mock_monkey_info['ff_distance'] = np.sqrt(
        (mock_monkey_info['monkey_x'] - mock_monkey_info['ff_x'])**2 + (mock_monkey_info['monkey_y'] - mock_monkey_info['ff_y'])**2)
    mock_monkey_info['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(ff_x=mock_monkey_info['ff_x'].values, ff_y=mock_monkey_info['ff_y'].values, mx=mock_monkey_info['monkey_x'].values,
                                                                                 my=mock_monkey_info['monkey_y'].values, m_angle=mock_monkey_info['monkey_angle'].values)
    mock_monkey_info['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=mock_monkey_info['ff_angle'].values, distances_to_ff=mock_monkey_info['ff_distance'].values)

    return mock_monkey_info


def _compute_curv_from_cur_end(mock_monkey_info,
                               ff_radius_for_opt_arc=10,
                               ):
    '''
    df needs to contain:
    point_index, nxt_ff_x, nxt_ff_y, nxt_ff_distance, nxt_ff_angle, nxt_ff_angle_boundary
    cur_opt_arc_end_x, cur_opt_arc_end_y, cur_opt_arc_end_heading, cur_opt_arc_curv (or cur_cntr_arc_end_x, cur_cntr_arc_end_y, cur_cntr_arc_end_heading, cur_cntr_arc_curv if using arc to ff center)

    '''

    null_arc_curv_df = curvature_utils._make_curvature_df(
        mock_monkey_info,
        mock_monkey_info['curv_of_traj'].values,
        ff_radius_for_opt_arc=ff_radius_for_opt_arc,
        clean=False,
        invalid_curvature_ok=True,
        ignore_error=True,
        include_cntr_arc_curv=False,
        # this doesn't matter since we only care about the curvature to nxt ff, not the null arc landing point inside nxt ff;
        opt_arc_stop_first_vis_bdry=False,
        # and whether the arc ends at start of the visible boundary doesn't change the curvature to nxt ff
    )

    result_df = null_arc_curv_df[['point_index', 'ff_angle_boundary']].copy()
    result_df['opt_curv_to_cur_ff'] = mock_monkey_info['curv_of_traj'].values
    result_df['nxt_ff_index'] = mock_monkey_info['ff_index'].values
    result_df['curv_from_cur_end_to_nxt_ff'] = null_arc_curv_df['opt_arc_curv'].values
    return result_df


def compute_prev_stop_to_next_ff_curv(nxt_ff_indexes, point_indexes_before_stop, monkey_information, ff_real_position_sorted, ff_caught_T_new,
                                      curv_of_traj_mode='distance', curv_traj_window_before_stop=[-25, 0]):
    df = _prepare_prev_stop_to_next_ff_data(nxt_ff_indexes, point_indexes_before_stop, monkey_information, ff_real_position_sorted, ff_caught_T_new,
                                            curv_of_traj_mode=curv_of_traj_mode, curv_traj_window_before_stop=curv_traj_window_before_stop)
    prev_stop_to_next_ff_curv = _compute_curv_from_prev_stop(
        df)
    return prev_stop_to_next_ff_curv


def _prepare_prev_stop_to_next_ff_data(nxt_ff_indexes, point_indexes_before_stop, monkey_information, ff_real_position_sorted, ff_caught_T_new,
                                       curv_of_traj_mode='distance', curv_traj_window_before_stop=[-25, 0]):
    df = find_cvn_utils.find_ff_info(
        nxt_ff_indexes, point_indexes_before_stop, monkey_information, ff_real_position_sorted)

    curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(curv_traj_window_before_stop, monkey_information, ff_caught_T_new,
                                                                                            curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)
    curv_of_traj_df.set_index('point_index', inplace=True)
    monkey_curv_before_stop = curv_of_traj_df.loc[point_indexes_before_stop,
                                                  'curv_of_traj'].values
    df['curv_of_traj'] = monkey_curv_before_stop
    return df


def _compute_curv_from_prev_stop(monkey_info):
    monkey_curv_df = curvature_utils._make_curvature_df(
        monkey_info,
        monkey_info['curv_of_traj'].values,
        ff_radius_for_opt_arc=10,
        clean=False,
        invalid_curvature_ok=True,
        ignore_error=True,
        include_cntr_arc_curv=False,
        opt_arc_stop_first_vis_bdry=False,
    )

    result_df = monkey_curv_df[['point_index', 'ff_angle_boundary']].copy()
    result_df['ff_angle_boundary'] = monkey_info['ff_angle_boundary'].values
    result_df['curv_from_stop_to_nxt_ff'] = monkey_curv_df['opt_arc_curv'].values
    result_df['traj_curv_to_stop'] = monkey_info['curv_of_traj'].values
    return result_df


def make_diff_in_curv_df(prev_stop_to_next_ff_curv, cur_end_to_next_ff_curv):
    """
    Calculate the difference in curvature between null arc and monkey data, 
    excluding rows where ff_angle_boundary is outside of [-45, 45] degrees.
    """

    # Define the angle boundary
    angle_boundary = [-math.pi/4, math.pi/4]

    # Find rows where ff_angle_boundary is outside of [-45, 45] degrees for both DataFrames
    null_arc_outside_boundary = cur_end_to_next_ff_curv[
        (cur_end_to_next_ff_curv['ff_angle_boundary'] < angle_boundary[0]) |
        (cur_end_to_next_ff_curv['ff_angle_boundary'] > angle_boundary[1])
    ]

    monkey_outside_boundary = prev_stop_to_next_ff_curv[
        (prev_stop_to_next_ff_curv['ff_angle_boundary'] < angle_boundary[0]) |
        (prev_stop_to_next_ff_curv['ff_angle_boundary'] > angle_boundary[1])
    ]

    # Get the union of the indices of the rows outside the boundary
    union_indices = null_arc_outside_boundary.index.union(
        monkey_outside_boundary.index)

    # Calculate the percentage of these rows out of all rows for both DataFrames
    total_rows = len(cur_end_to_next_ff_curv) + \
        len(prev_stop_to_next_ff_curv)
    percentage_outside_boundary = len(union_indices) / total_rows * 100

    # Print the percentage
    print(
        f"Percentage of rows outside of [-45, 45]: {percentage_outside_boundary:.2f}%")

    # Drop the union of these rows from both DataFrames
    monkey_curv_df = prev_stop_to_next_ff_curv.drop(
        monkey_outside_boundary.index)
    null_arc_curv_df = cur_end_to_next_ff_curv.drop(
        null_arc_outside_boundary.index)

    monkey_curv_df = monkey_curv_df[[
        'ref_point_index', 'traj_curv_to_stop', 'curv_from_stop_to_nxt_ff']]
    null_arc_curv_df = null_arc_curv_df[[
        'ref_point_index', 'opt_curv_to_cur_ff', 'curv_from_cur_end_to_nxt_ff']]

    diff_in_curv_df = monkey_curv_df.merge(
        null_arc_curv_df, how='outer', on='ref_point_index')

    diff_in_curv_df = furnish_diff_in_curv_df(diff_in_curv_df)

    return diff_in_curv_df


def furnish_diff_in_curv_df(diff_in_curv_df):
    # we need the df to have the following columns:
    # curv_from_cur_end_to_nxt_ff, curv_from_stop_to_nxt_ff, traj_curv_to_stop, opt_curv_to_cur_ff

    diff_in_curv_df['d_curv_null_arc'] = (180/math.pi * 100) * (
        diff_in_curv_df['curv_from_cur_end_to_nxt_ff'] - diff_in_curv_df['opt_curv_to_cur_ff'])
    diff_in_curv_df['d_curv_monkey'] = (180/math.pi * 100) * (
        diff_in_curv_df['curv_from_stop_to_nxt_ff'] - diff_in_curv_df['traj_curv_to_stop'])

    diff_in_curv_df['abs_d_curv_null_arc'] = np.abs(
        diff_in_curv_df['d_curv_null_arc'])
    diff_in_curv_df['abs_d_curv_monkey'] = np.abs(
        diff_in_curv_df['d_curv_monkey'])

    diff_in_curv_df['diff_in_d_curv'] = diff_in_curv_df['d_curv_null_arc'] - \
        diff_in_curv_df['d_curv_monkey']
    diff_in_curv_df['diff_in_abs_d_curv'] = np.abs(
        diff_in_curv_df['d_curv_null_arc']) - np.abs(diff_in_curv_df['d_curv_monkey'])

    # # The following 2 vars are currently unused
    # diff_in_curv_df['diff_in_curv_to_cur_ff'] = diff_in_curv_df['opt_curv_to_cur_ff'] - \
    #     diff_in_curv_df['traj_curv_to_stop']
    # diff_in_curv_df['diff_in_curv_to_nxt_ff'] = diff_in_curv_df['curv_from_cur_end_to_nxt_ff'] - \
    #     diff_in_curv_df['curv_from_stop_to_nxt_ff']

    return diff_in_curv_df


def get_diff_in_curv_df_from_heading_info_df(heading_info_df):
    diff_in_curv_df = heading_info_df[['stop_point_index', 'ref_point_index', 'd_curv_null_arc', 'd_curv_monkey',
                                       'abs_d_curv_null_arc', 'abs_d_curv_monkey', 'diff_in_abs_d_curv']].copy()
    return diff_in_curv_df
