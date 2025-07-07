
from data_wrangling import specific_utils
from planning_analysis.show_planning.get_stops_near_ff import plot_stops_near_ff_utils, find_stops_near_ff_utils
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


def _extend_arc_length_by_increasing_arc_ending_angle(null_arc_info):
    abs_delta_angle = abs(
        null_arc_info['arc_ending_angle'] - null_arc_info['arc_starting_angle'])
    # clip angle_to_add so that its absolute value is within 45 degrees
    angle_to_add = np.clip(abs_delta_angle * 3, 0, math.pi/2 - 0.00001)
    angle_to_add = angle_to_add * \
        np.sign(null_arc_info['arc_ending_angle'] -
                null_arc_info['arc_starting_angle'])
    null_arc_info['arc_ending_angle'] = null_arc_info['arc_starting_angle'] + angle_to_add
    return null_arc_info


def get_points_on_each_arc(null_arc_info, num_points_on_each_arc=2000, extend_arc_angle=False):

    # Generate angle array
    if extend_arc_angle:
        null_arc_info = _extend_arc_length_by_increasing_arc_ending_angle(
            null_arc_info)

    angle_array = np.linspace(null_arc_info['arc_starting_angle'].values,
                              null_arc_info['arc_ending_angle'].values, num_points_on_each_arc).T.reshape(-1)

    # Repeat necessary values to match the length of angle_array
    repeated_values = {col: np.repeat(null_arc_info[col].values, num_points_on_each_arc) for col in ['arc_ff_index', 'all_arc_radius', 'center_x', 'center_y',
                                                                                                     'arc_ff_x', 'arc_ff_y', 'arc_starting_angle']}

    # Create DataFrame for arc points
    arc_df = pd.DataFrame({
        'cur_ff_index': repeated_values['arc_ff_index'],
        'radius': repeated_values['all_arc_radius'],
        'center_x': repeated_values['center_x'],
        'center_y': repeated_values['center_y'],
        'arc_ff_x': repeated_values['arc_ff_x'],
        'arc_ff_y': repeated_values['arc_ff_y'],
        'arc_starting_angle': repeated_values['arc_starting_angle'],
        'angle': angle_array,
        'point_id_on_arc': np.tile(np.arange(num_points_on_each_arc), len(null_arc_info))
    })

    # Calculate x and y coordinates of arc points
    arc_df['x'] = arc_df['center_x'] + \
        arc_df['radius'] * np.cos(arc_df['angle'])
    arc_df['y'] = arc_df['center_y'] + \
        arc_df['radius'] * np.sin(arc_df['angle'])

    # Calculate distance to firefly
    arc_df['distance_to_ff'] = np.sqrt(
        (arc_df['x'] - arc_df['arc_ff_x'])**2 + (arc_df['y'] - arc_df['arc_ff_y'])**2)

    arc_df['delta_angle_from_starting_angle'] = np.abs(
        arc_df['angle'] - arc_df['arc_starting_angle'])
    return arc_df


def get_optimal_arc_landing_points_closest_to_stop(null_arc_info, stops_near_ff_df, reward_boundary_radius=25):
    if len(null_arc_info) != len(stops_near_ff_df):
        raise ValueError(
            'The number of rows in null_arc_info and stops_near_ff_df do not match.')

    null_arc_info = null_arc_info.merge(stops_near_ff_df[[
                                        'cur_ff_index', 'cur_ff_x', 'cur_ff_y']], left_on='arc_ff_index', right_on='cur_ff_index', how='left')
    null_arc_info.rename(
        columns={'cur_ff_x': 'arc_ff_x', 'cur_ff_y': 'arc_ff_y'}, inplace=True)

    arc_df = get_points_on_each_arc(null_arc_info, extend_arc_angle=True)
    # Filter points within 25 units of the firefly
    arc_df = arc_df[arc_df['distance_to_ff'] <= reward_boundary_radius].copy()

    # Merge with stops data
    arc_df = arc_df.merge(stops_near_ff_df[[
                          'cur_ff_index', 'stop_x', 'stop_y']], on='cur_ff_index', how='left')
    # Calculate distance to stop
    arc_df['distance_to_stop'] = np.sqrt(
        (arc_df['x'] - arc_df['stop_x'])**2 + (arc_df['y'] - arc_df['stop_y'])**2)

    # Find the arc points closest to each stop
    index_of_arc_rows_closest_to_stop = arc_df.groupby(
        'cur_ff_index')['distance_to_stop'].idxmin()
    arc_rows_closest_to_stop = arc_df.loc[index_of_arc_rows_closest_to_stop].reset_index(
        drop=True)

    return arc_rows_closest_to_stop


def get_optimal_arc_landing_points_when_first_reaching_visible_boundary(null_arc_info,
                                                                        visible_boundary_radius=10,
                                                                        reward_boundary_radius=25):

    for i in range(2):  # do the following twice, just in case
        # Adjust 'arc_ending_angle' to ensure it is within 180 degrees of 'arc_starting_angle'
        # If 'arc_ending_angle' is more than 180 degrees greater than 'arc_starting_angle', subtract 2*pi from 'arc_ending_angle'
        greater_than_pi = null_arc_info['arc_ending_angle'] - \
            null_arc_info['arc_starting_angle'] > math.pi
        null_arc_info.loc[greater_than_pi, 'arc_ending_angle'] -= 2 * math.pi

        # If 'arc_ending_angle' is more than 180 degrees less than 'arc_starting_angle', add 2*pi to 'arc_ending_angle'
        less_than_minus_pi = null_arc_info['arc_starting_angle'] - \
            null_arc_info['arc_ending_angle'] > math.pi
        null_arc_info.loc[less_than_minus_pi,
                          'arc_ending_angle'] += 2 * math.pi

    arc_df_original = get_points_on_each_arc(
        null_arc_info, extend_arc_angle=True)

    # Find the arc points that first reach the visible boundary of the firefly
    arc_df = arc_df_original[arc_df_original['distance_to_ff']
                             < visible_boundary_radius + 0.1].copy()
    arc_df = arc_df.sort_values(
        by=['cur_ff_index', 'point_id_on_arc']).reset_index(drop=True)
    arc_rows_to_first_reach_boundary = arc_df.groupby(
        'cur_ff_index').first().reset_index(drop=False)

    too_big_angle_rows = arc_rows_to_first_reach_boundary[
        arc_rows_to_first_reach_boundary['delta_angle_from_starting_angle'] > math.pi/2].copy()
    if len(too_big_angle_rows) > 0:
        print(f'Note: When calling get_optimal_arc_landing_points_when_first_reaching_visible_boundary, there are {len(too_big_angle_rows)} points that are more than 90 degrees away from the starting angle of the arc.' +
              'They will be changed to the closest point to the ff center that are still within the reward boundary.')
        arc_df_sub = arc_df_original[arc_df_original['cur_ff_index'].isin(
            too_big_angle_rows['cur_ff_index'])].copy()
        arc_df_sub = arc_df_sub[arc_df_sub['distance_to_ff']
                                <= reward_boundary_radius].copy()
        arc_df_sub = arc_df_sub[arc_df_sub['delta_angle_from_starting_angle']
                                <= math.pi/2].copy()
        arc_df_sub = arc_df_sub.sort_values(by=['cur_ff_index', 'distance_to_ff'], ascending=[
                                            True, True]).reset_index(drop=True)
        new_too_big_arc_rows = arc_df_sub.groupby(
            'cur_ff_index').first().reset_index(drop=False)
        if len(new_too_big_arc_rows) != len(too_big_angle_rows):
            raise ValueError(
                'The number of rows in new_too_big_arc_rows and too_big_angle_rows do not match.')
        # else, let arc_rows_to_first_reach_boundary drop the old rows and concatenate the new rows, and then sort by cur_ff_index
        arc_rows_to_first_reach_boundary = pd.concat([arc_rows_to_first_reach_boundary[~arc_rows_to_first_reach_boundary['cur_ff_index'].isin(
            too_big_angle_rows['cur_ff_index'])], new_too_big_arc_rows], axis=0)
        arc_rows_to_first_reach_boundary = arc_rows_to_first_reach_boundary.sort_values(
            by='cur_ff_index').reset_index(drop=True)

    if len(arc_rows_to_first_reach_boundary) != len(null_arc_info):
        # arc_rows_to_first_reach_boundary = _get_missed_arc_info2(null_arc_info, arc_rows_to_first_reach_boundary, reward_boundary_radius=reward_boundary_radius)
        arc_rows_to_first_reach_boundary = _get_missed_arc_info(
            null_arc_info, arc_df_original, arc_rows_to_first_reach_boundary, reward_boundary_radius=reward_boundary_radius)

    return arc_rows_to_first_reach_boundary


def _get_missed_arc_info2(null_arc_info, arc_rows_to_first_reach_boundary, reward_boundary_radius=25):
    # compared to the first version, this version will extend the arc
    null_arc_info_sub = null_arc_info[~null_arc_info['arc_ff_index'].isin(
        arc_rows_to_first_reach_boundary['cur_ff_index'].values)].copy()
    arc_df_extended = get_points_on_each_arc(
        null_arc_info_sub, extend_arc_angle=True)
    missed_arc = arc_df_extended[(
        arc_df_extended['distance_to_ff'] <= reward_boundary_radius)].copy()
    arc_rows_to_first_reach_boundary = _add_info_to_arc_rows_to_first_reach_boundary(
        arc_rows_to_first_reach_boundary, missed_arc, null_arc_info)
    return arc_rows_to_first_reach_boundary


def _get_missed_arc_info(null_arc_info, arc_df_original, arc_rows_to_first_reach_boundary, reward_boundary_radius=25):
    # try to find the missed arc points that are within 90 degrees of the starting angle of the arc, even if they are outside of visible boundary (as long as they are inside the reward boundary)
    missed_arc = arc_df_original[~arc_df_original['cur_ff_index'].isin(
        arc_rows_to_first_reach_boundary['cur_ff_index'].values)].copy()
    missed_arc = missed_arc[(missed_arc['delta_angle_from_starting_angle'] <= math.pi/2) &
                            (missed_arc['distance_to_ff'] <= reward_boundary_radius)].copy()

    arc_rows_to_first_reach_boundary = _add_info_to_arc_rows_to_first_reach_boundary(
        arc_rows_to_first_reach_boundary, missed_arc, null_arc_info)
    return arc_rows_to_first_reach_boundary


def _add_info_to_arc_rows_to_first_reach_boundary(arc_rows_to_first_reach_boundary, missed_arc, null_arc_info):
    print(f'Note: When calling get_optimal_arc_landing_points_when_first_reaching_visible_boundary, there are {len(null_arc_info) - len(arc_rows_to_first_reach_boundary)} points out of {len(null_arc_info)} points that are not within the visible boundary of the firefly.' +
          'They will be changed to the closest point to the ff center that are still within the reward boundary.')

    missed_arc.sort_values(by=['cur_ff_index', 'distance_to_ff'], ascending=[
                           True, True], inplace=True)
    missed_arc = missed_arc.groupby(
        'cur_ff_index').first().reset_index(drop=False)
    arc_rows_to_first_reach_boundary = pd.concat(
        [arc_rows_to_first_reach_boundary, missed_arc], axis=0)
    arc_rows_to_first_reach_boundary.sort_values(
        by='cur_ff_index', inplace=True)
    if len(arc_rows_to_first_reach_boundary) != len(null_arc_info):
        raise ValueError(
            'The number of rows in arc_rows_to_first_reach_boundary and stops_near_ff_df do not match.')
    return arc_rows_to_first_reach_boundary


def make_new_ff_at_monkey_xy_if_within_1_cm(new_ff_x, new_ff_y, monkey_x, monkey_y):
    # Calculate distance between new ff and monkey
    distance = np.sqrt((new_ff_x - monkey_x)**2 + (new_ff_y - monkey_y)**2)

    # If distance is less than 1 cm, set new ff to monkey's position
    if np.where(distance < 1)[0].size > 0:
        print(
            f'Number of new ff xy within 1 cm of monkey: {np.where(distance < 1)[0].size} out of {len(distance)}. Setting them to monkey position.')
        new_ff_x[distance < 1] = monkey_x[distance < 1]
        new_ff_y[distance < 1] = monkey_y[distance < 1]

    return new_ff_x, new_ff_y


def make_cur_and_nxt_ff_df(nxt_ff_final_df, cur_ff_final_df):
    # Define shared and relevant columns
    shared_columns = ['monkey_x', 'monkey_y', 'monkey_angle', 'curv_of_traj', 'point_index',
                      'stop_point_index', 'monkey_angle_before_stop', 'd_heading_of_traj']

    relevant_columns = ['ff_index', 'ff_x', 'ff_y', 'ff_distance', 'ff_angle', 'ff_angle_boundary',
                        'optimal_curvature', 'optimal_arc_measure', 'optimal_arc_radius', 'optimal_arc_end_direction',
                        'curv_to_ff_center', 'arc_radius_to_ff_center', 'd_heading_of_arc', 'arc_end_x', 'arc_end_y']

    # Create a copy of the shared columns from nxt_ff_final_df and rename them
    cur_and_nxt_ff_df = pd.concat(
        [nxt_ff_final_df[shared_columns], cur_ff_final_df[shared_columns]], axis=0)
    # drop duplicate rows
    cur_and_nxt_ff_df = cur_and_nxt_ff_df.drop_duplicates(
        subset=['stop_point_index']).reset_index(drop=True)

    cur_and_nxt_ff_df.rename(columns={'point_index': 'ref_point_index',
                                      'monkey_x': 'ref_monkey_x',
                                      'monkey_y': 'ref_monkey_y',
                                      'monkey_angle': 'ref_monkey_angle',
                                      'curv_of_traj': 'ref_curv_of_traj'}, inplace=True)

    relevant_columns = [
        col for col in relevant_columns if col in nxt_ff_final_df.columns]
    # Create copies of the relevant columns from nxt_ff_final_df2 and cur_ff_final_df2 and rename them
    nxt_ff_final_df2 = nxt_ff_final_df[relevant_columns].copy()
    nxt_ff_final_df2.columns = [
        'nxt_'+col for col in nxt_ff_final_df2.columns.tolist()]
    nxt_ff_final_df2['stop_point_index'] = nxt_ff_final_df['stop_point_index']

    cur_ff_final_df2 = cur_ff_final_df[relevant_columns].copy()
    cur_ff_final_df2.columns = [
        'cur_'+col for col in cur_ff_final_df2.columns.tolist()]
    cur_ff_final_df2['stop_point_index'] = cur_ff_final_df['stop_point_index']

    # Merge cur_and_nxt_ff_df, nxt_ff_final_df2, and cur_ff_final_df2
    cur_and_nxt_ff_df = cur_and_nxt_ff_df.merge(
        nxt_ff_final_df2, how='left', on='stop_point_index')
    cur_and_nxt_ff_df = cur_and_nxt_ff_df.merge(
        cur_ff_final_df2, how='left', on='stop_point_index')

    # Calculate landing headings
    if 'cur_d_heading_of_arc' in cur_and_nxt_ff_df.columns:
        cur_and_nxt_ff_df['cur_arc_end_heading'] = cur_and_nxt_ff_df['ref_monkey_angle'] + \
            cur_and_nxt_ff_df['cur_d_heading_of_arc']
        cur_and_nxt_ff_df['nxt_arc_end_heading'] = cur_and_nxt_ff_df['ref_monkey_angle'] + \
            cur_and_nxt_ff_df['nxt_d_heading_of_arc']

    return cur_and_nxt_ff_df


def make_heading_info_df(cur_and_nxt_ff_df, stops_near_ff_df, monkey_information, ff_real_position_sorted):
    # Select relevant columns from stops_near_ff_df
    heading_info_df = stops_near_ff_df[['stop_point_index', 'stop_x', 'stop_y', 'stop_time',
                                        'cur_ff_index', 'cur_ff_x', 'cur_ff_y', 'cur_ff_cluster_50_size',
                                        'point_index_before_stop',  'monkey_angle_before_stop',
                                        'next_stop_point_index', 'next_stop_time', 'cum_distance_between_two_stops',
                                        'curv_range', 'curv_iqr', 'nxt_ff_index', 'nxt_ff_x', 'nxt_ff_y',
                                        'NXT_time_ff_last_seen_bbas',
                                        'NXT_time_ff_last_seen_bsans',
                                        'nxt_ff_last_flash_time_bbas',
                                        'nxt_ff_last_flash_time_bsans',
                                        'nxt_ff_cluster_last_seen_time_bbas',
                                        'nxt_ff_cluster_last_seen_time_bsans',
                                        'nxt_ff_cluster_last_flash_time_bbas',
                                        'nxt_ff_cluster_last_flash_time_bsans']].copy()

    # Add monkey's position before stop from monkey_information
    heading_info_df['mx_before_stop'], heading_info_df['my_before_stop'] = monkey_information.loc[heading_info_df['point_index_before_stop'], [
        'monkey_x', 'monkey_y']].values.T

    # Add alternative ff position from ff_real_position_sorted
    heading_info_df[['nxt_ff_x', 'nxt_ff_y']
                    ] = ff_real_position_sorted[heading_info_df['nxt_ff_index']]

    # Merge with cur_and_nxt_ff_df to get landing headings
    columns_to_keep = ['stop_point_index', 'cur_arc_end_heading', 'nxt_arc_end_heading',
                       'cur_arc_end_x', 'cur_arc_end_y',
                       'd_heading_of_traj', 'cur_d_heading_of_arc', 'ref_monkey_angle', 'ref_curv_of_traj', 'nxt_d_heading_of_arc',
                       ]
    columns_to_keep = [
        col for col in columns_to_keep if col in cur_and_nxt_ff_df.columns]
    heading_info_df = heading_info_df.merge(
        cur_and_nxt_ff_df[columns_to_keep], how='left', on='stop_point_index')

    # Calculate angles from monkey before stop to nxt ff and from cur ff null arc landing position to alternative ff
    heading_info_df['angle_from_m_before_stop_to_cur_ff'] = specific_utils.calculate_angles_to_ff_centers(
        heading_info_df['cur_ff_x'], heading_info_df['cur_ff_y'], heading_info_df['mx_before_stop'], heading_info_df['my_before_stop'], heading_info_df['monkey_angle_before_stop'])
    heading_info_df['angle_from_m_before_stop_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
        heading_info_df['nxt_ff_x'], heading_info_df['nxt_ff_y'], heading_info_df['mx_before_stop'], heading_info_df['my_before_stop'], heading_info_df['monkey_angle_before_stop'])
    if 'cur_arc_end_x' in heading_info_df:
        heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
            heading_info_df['nxt_ff_x'], heading_info_df['nxt_ff_y'], heading_info_df['cur_arc_end_x'], heading_info_df['cur_arc_end_y'], heading_info_df['cur_arc_end_heading'])

    # The following two columns are originally from calculate_info_based_on_monkey_angles
    heading_info_df['angle_from_cur_ff_to_stop'] = specific_utils.calculate_angles_to_ff_centers(ff_x=heading_info_df['stop_x'].values, ff_y=heading_info_df['stop_y'],
                                                                                                 mx=heading_info_df['cur_ff_x'].values, my=heading_info_df['cur_ff_y'], m_angle=heading_info_df['monkey_angle_before_stop'])
    heading_info_df['angle_from_cur_ff_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(ff_x=heading_info_df['nxt_ff_x'].values, ff_y=heading_info_df['nxt_ff_y'],
                                                                                                   mx=heading_info_df['cur_ff_x'].values, my=heading_info_df['cur_ff_y'], m_angle=heading_info_df['monkey_angle_before_stop'])

    return heading_info_df


def get_ang_traj_nxt_and_ang_cur_nxt(heading_info_df):
    heading_info_df = heading_info_df.copy()
    # print the number of rows in heading_info_df that contain NaN values
    print(
        f'Number of rows with NaN values in heading_info_df: {heading_info_df.isnull().any(axis=1).sum()} out of {heading_info_df.shape[0]} rows, but they are not dropped. The columns with NaN values are:')
    # print columns with NaN values and number of NaN values in each column
    print(heading_info_df.isnull().sum()[heading_info_df.isnull().sum() > 0])

    # heading_info_df.dropna(inplace=True)
    ang_traj_nxt = heading_info_df['angle_from_m_before_stop_to_nxt_ff'].values.reshape(
        -1)
    ang_cur_nxt = heading_info_df['angle_from_cur_ff_landing_to_nxt_ff'].values.reshape(
        -1)

    # heading_info_df_no_na = heading_info_df.copy()
    return ang_traj_nxt, ang_cur_nxt, heading_info_df


def conduct_linear_regression(ang_traj_nxt, ang_cur_nxt, fit_intercept=True):
    # calculate r and slope of ang_traj_nxt and ang_cur_nxt
    if fit_intercept:
        X = sm.add_constant(ang_traj_nxt)
        model = sm.OLS(ang_cur_nxt, X)
        results = model.fit()
        slope = results.params[1]
        p_value = results.pvalues[1]
        intercept = results.params[0]
    else:
        model = sm.OLS(ang_cur_nxt, ang_traj_nxt)
        results = model.fit()
        slope = results.params[0]
        p_value = results.pvalues[0]
        intercept = 0
    r_value = results.rsquared
    return slope, intercept, r_value, p_value, results


def omit_outliers_from_linear_regression_results(ang_traj_nxt, ang_cur_nxt, results):
    # Calculate residuals
    residuals = results.resid
    # Identify outliers: those points where the residual is more than 3 standard deviations away from the mean
    outliers = np.abs(residuals) > 3 * np.std(residuals)
    # Remove outliers
    ang_traj_nxt_no_outliers = ang_traj_nxt[~outliers]
    ang_cur_nxt_no_outliers = ang_cur_nxt[~outliers]
    return ang_traj_nxt_no_outliers, ang_cur_nxt_no_outliers


def conduct_linear_regression_to_show_planning(ang_traj_nxt, ang_cur_nxt, use_abs_values=False, fit_intercept=True, omit_outliers=False, q13_only=False, show_plot=True,
                                               hue=None):
    if q13_only:
        # retain only the values in the first and third quadrants
        same_sign = np.sign(ang_traj_nxt) == np.sign(ang_cur_nxt)
        ang_traj_nxt = ang_traj_nxt[same_sign]
        ang_cur_nxt = ang_cur_nxt[same_sign]

    if use_abs_values:
        ang_traj_nxt = np.abs(ang_traj_nxt)
        ang_cur_nxt = np.abs(ang_cur_nxt)

    # calculate r and slope of ang_traj_nxt and ang_cur_nxt
    slope, intercept, r_value, p_value, results = conduct_linear_regression(
        ang_traj_nxt, ang_cur_nxt, fit_intercept=fit_intercept)
    sample_size = len(ang_cur_nxt)

    if omit_outliers:
        ang_traj_nxt, ang_cur_nxt = omit_outliers_from_linear_regression_results(
            ang_traj_nxt, ang_cur_nxt, results)
        slope, intercept, r_value, p_value, results = conduct_linear_regression(
            ang_traj_nxt, ang_cur_nxt, fit_intercept=fit_intercept)
        title = f'Outliers Omitted: slope: {round(slope, 2)}, intercept: {round(intercept, 2)}, r value: {round(r_value, 2)}, p value: {round(p_value, 4)}, sample size: {sample_size}'
    else:
        title = f'slope: {round(slope, 2)}, intercept: {round(intercept, 2)}, r value: {round(r_value, 2)}, p value: {round(p_value, 4)}, sample size: {sample_size}'

    if show_plot:
        plot_stops_near_ff_utils.plot_ang_traj_nxt_vs_ang_cur_nxt(
            ang_traj_nxt, ang_cur_nxt, hue, title, slope, intercept)

    # print(results.summary())
    return slope, intercept, r_value, p_value, results


def make_diff_and_ratio_stat_df(test_df, ctrl_df):

    columns_to_describe = ['diff_in_angle_to_nxt_ff',
                           'ratio_of_angle_to_nxt_ff', 'diff_in_abs_angle_to_nxt_ff', 'diff_in_abs_d_curv']

    test_stat = test_df[columns_to_describe].describe().rename(
        index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
    test_stat.columns = ['test_' + col for col in test_stat.columns]

    ctrl_stat = ctrl_df[columns_to_describe].describe().rename(
        index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
    ctrl_stat.columns = ['ctrl_' + col for col in ctrl_stat.columns]

    diff_and_ratio_stat_df = pd.concat([test_stat, ctrl_stat], axis=1)

    return diff_and_ratio_stat_df


def remove_outliers(x_var_df, y_var):
    mean_y = y_var.mean()
    std_y = y_var.std()

    # Step 2: Identify rows where values are more than 3 std dev above the mean
    outliers = (abs(y_var) > abs(mean_y + 3 * std_y))
    non_outlier_index = np.where(np.array(outliers) == False)[0]

    # Step 3: Drop these rows from both DataFrames
    y_var = y_var.iloc[non_outlier_index]
    x_var_df = x_var_df.iloc[non_outlier_index]
    print(
        f'Number of outliers dropped before train_test_split: {len(outliers) - len(non_outlier_index)} out of {len(outliers)} samples.')
    return x_var_df, y_var


def make_nxt_ff_info_for_null_arc(nxt_ff_df_modified, cur_ff_final_df, heading_info_df):
    # To get # curv of null arc from monkey stop to nxt ff

    # use 'cur_arc_end_x', 'cur_arc_end_y', 'cur_arc_end_heading'
    # to replace 'monkey_x', 'monkey_y', 'monkey_angle'
    nxt_ff_info_for_null_arc = nxt_ff_df_modified.copy()
    nxt_ff_info_for_null_arc['monkey_x'] = heading_info_df['cur_arc_end_x'].values
    nxt_ff_info_for_null_arc['monkey_y'] = heading_info_df['cur_arc_end_y'].values
    nxt_ff_info_for_null_arc['monkey_angle'] = heading_info_df['cur_arc_end_heading'].values

    # then calculate ff_distance', 'ff_angle', 'ff_angle_boundary'
    nxt_ff_info_for_null_arc['ff_distance'] = np.sqrt((nxt_ff_info_for_null_arc['monkey_x'] - nxt_ff_info_for_null_arc['ff_x'])**2 + (
        nxt_ff_info_for_null_arc['monkey_y'] - nxt_ff_info_for_null_arc['ff_y'])**2)
    nxt_ff_info_for_null_arc['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(ff_x=nxt_ff_info_for_null_arc['ff_x'].values, ff_y=nxt_ff_info_for_null_arc['ff_y'].values, mx=nxt_ff_info_for_null_arc['monkey_x'].values,
                                                                                         my=nxt_ff_info_for_null_arc['monkey_y'].values, m_angle=nxt_ff_info_for_null_arc['monkey_angle'].values)
    nxt_ff_info_for_null_arc['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=nxt_ff_info_for_null_arc['ff_angle'].values, distances_to_ff=nxt_ff_info_for_null_arc['ff_distance'].values)

    # make the point index as point index right before stop
    nxt_ff_info_for_null_arc['point_index'] = heading_info_df['point_index_before_stop'].values

    # curv_of_traj will be cur null curv's optimal_curvature
    nxt_ff_info_for_null_arc['curv_of_traj'] = cur_ff_final_df['optimal_curvature'].values

    return nxt_ff_info_for_null_arc


def make_nxt_ff_info_for_monkey(nxt_ff_df_modified, heading_info_df, monkey_information, ff_real_position_sorted, ff_caught_T_new,
                                curv_traj_window_before_stop=[-50, 0]):
    nxt_ff_info_for_monkey = find_stops_near_ff_utils.find_ff_info(
        nxt_ff_df_modified.ff_index.values, heading_info_df.point_index_before_stop.values, monkey_information, ff_real_position_sorted)
    nxt_ff_info_for_monkey['stop_point_index'] = heading_info_df['stop_point_index'].values

    curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(curv_traj_window_before_stop, monkey_information, ff_caught_T_new,
                                                                                            curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False)
    curv_of_traj_df.set_index('point_index', inplace=True)
    monkey_curv_before_stop = curv_of_traj_df.loc[heading_info_df[
        'point_index_before_stop'].values, 'curv_of_traj'].values

    nxt_ff_info_for_monkey['curv_of_traj'] = monkey_curv_before_stop

    return nxt_ff_info_for_monkey


def make_diff_in_curv_df(nxt_ff_info_for_monkey, nxt_ff_info_for_null_arc):
    """
    Calculate the difference in curvature between null arc and monkey data, 
    excluding rows where ff_angle_boundary is outside of [-45, 45] degrees.
    """

    # Define the angle boundary
    angle_boundary = [-math.pi/4, math.pi/4]

    # Find rows where ff_angle_boundary is outside of [-45, 45] degrees for both DataFrames
    null_arc_outside_boundary = nxt_ff_info_for_null_arc[
        (nxt_ff_info_for_null_arc['ff_angle_boundary'] < angle_boundary[0]) |
        (nxt_ff_info_for_null_arc['ff_angle_boundary'] > angle_boundary[1])
    ]

    monkey_outside_boundary = nxt_ff_info_for_monkey[
        (nxt_ff_info_for_monkey['ff_angle_boundary'] < angle_boundary[0]) |
        (nxt_ff_info_for_monkey['ff_angle_boundary'] > angle_boundary[1])
    ]

    # Get the union of the indices of the rows outside the boundary
    union_indices = null_arc_outside_boundary.index.union(
        monkey_outside_boundary.index)

    # Calculate the percentage of these rows out of all rows for both DataFrames
    total_rows = len(nxt_ff_info_for_null_arc) + len(nxt_ff_info_for_monkey)
    percentage_outside_boundary = len(union_indices) / total_rows * 100

    # Print the percentage
    print(
        f"Percentage of rows outside of [-45, 45]: {percentage_outside_boundary:.2f}%")

    # Drop the union of these rows from both DataFrames
    nxt_ff_info_for_null_arc_cleaned = nxt_ff_info_for_null_arc.drop(
        null_arc_outside_boundary.index)
    nxt_ff_info_for_monkey_cleaned = nxt_ff_info_for_monkey.drop(
        monkey_outside_boundary.index)

    # Generate curvature DataFrames
    null_arc_curv_df = curvature_utils._make_curvature_df(
        nxt_ff_info_for_null_arc_cleaned,
        nxt_ff_info_for_null_arc_cleaned['curv_of_traj'].values,
        ff_radius_for_optimal_arc=15,
        clean=False,
        invalid_curvature_ok=False,
        include_curv_to_ff_center=False,
        # this doesn't matter since we only care about the curvature to nxt ff, not the null arc landing point inside nxt ff
        opt_arc_stop_first_vis_bdry=False,
    )

    monkey_curv_df = curvature_utils._make_curvature_df(
        nxt_ff_info_for_monkey_cleaned,
        nxt_ff_info_for_monkey_cleaned['curv_of_traj'].values,
        ff_radius_for_optimal_arc=15,
        clean=False,
        invalid_curvature_ok=True,
        include_curv_to_ff_center=False,
        opt_arc_stop_first_vis_bdry=False,
    )

    null_arc_curv_df = null_arc_curv_df[[
        'optimal_curvature']].reset_index(drop=True)
    null_arc_curv_df[['stop_point_index', 'curv_of_traj']] = nxt_ff_info_for_null_arc_cleaned[[
        'stop_point_index', 'curv_of_traj']].values
    null_arc_curv_df.rename(columns={'curv_of_traj': 'null_arc_curv_to_cur_ff',
                                     'optimal_curvature': 'null_arc_curv_to_nxt_ff'}, inplace=True)

    monkey_curv_df = monkey_curv_df[[
        'optimal_curvature']].reset_index(drop=True)
    monkey_curv_df[['stop_point_index', 'curv_of_traj']] = nxt_ff_info_for_monkey_cleaned[[
        'stop_point_index', 'curv_of_traj']].values
    monkey_curv_df.rename(columns={'curv_of_traj': 'monkey_curv_to_cur_ff',
                                   'optimal_curvature': 'monkey_curv_to_nxt_ff'}, inplace=True)

    diff_in_curv_df = monkey_curv_df.merge(
        null_arc_curv_df, how='outer', on='stop_point_index')

    return diff_in_curv_df


def furnish_diff_in_curv_df(diff_in_curv_df):
    diff_in_curv_df['d_curv_null_arc'] = (180/math.pi * 100) * (
        diff_in_curv_df['null_arc_curv_to_nxt_ff'] - diff_in_curv_df['null_arc_curv_to_cur_ff'])
    diff_in_curv_df['d_curv_monkey'] = (180/math.pi * 100) * (
        diff_in_curv_df['monkey_curv_to_nxt_ff'] - diff_in_curv_df['monkey_curv_to_cur_ff'])
    diff_in_curv_df['abs_d_curv_null_arc'] = np.abs(
        diff_in_curv_df['d_curv_null_arc'])
    diff_in_curv_df['abs_d_curv_monkey'] = np.abs(
        diff_in_curv_df['d_curv_monkey'])

    diff_in_curv_df['diff_in_curv_to_stop'] = diff_in_curv_df['null_arc_curv_to_cur_ff'] - \
        diff_in_curv_df['monkey_curv_to_cur_ff']
    diff_in_curv_df['diff_in_curv_to_alt'] = diff_in_curv_df['null_arc_curv_to_nxt_ff'] - \
        diff_in_curv_df['monkey_curv_to_nxt_ff']
    diff_in_curv_df['diff_in_d_curv'] = diff_in_curv_df['d_curv_null_arc'] - \
        diff_in_curv_df['d_curv_monkey']
    diff_in_curv_df['diff_in_abs_d_curv'] = np.abs(
        diff_in_curv_df['d_curv_null_arc']) - np.abs(diff_in_curv_df['d_curv_monkey'])

    return diff_in_curv_df


def retrieve_df_based_on_ref_point(ref_point_mode, ref_point_value, test_or_control, data_folder_name, df_partial_path, monkey_name):
    df_path = os.path.join(data_folder_name, df_partial_path, test_or_control)
    os.makedirs(df_path, exist_ok=True)
    df_name = find_stops_near_ff_utils.get_df_name_by_ref(
        monkey_name, ref_point_mode, ref_point_value)
    retrieved_df = pd.read_csv(os.path.join(df_path, df_name), index_col=0)
    print(
        f'Retrieving {df_name} from {os.path.join(df_path, df_name)} succeeded')
    return retrieved_df


def get_diff_in_curv_df_from_heading_info_df(heading_info_df):
    diff_in_curv_df = heading_info_df[['stop_point_index', 'd_curv_null_arc', 'd_curv_monkey',
                                       'abs_d_curv_null_arc', 'abs_d_curv_monkey', 'diff_in_abs_d_curv']].copy()
    return diff_in_curv_df
