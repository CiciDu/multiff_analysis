
from data_wrangling import specific_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import plot_cvn_utils, find_cvn_utils
from planning_analysis.plan_factors import build_factor_comp

import statsmodels.api as sm
import pandas as pd
import numpy as np
import math
import os


def get_points_on_each_arc(null_arc_info, num_points_on_each_arc=2000, extend_arc_angle=False):

    # Generate angle array
    if extend_arc_angle:
        null_arc_info = _extend_arc_length_by_increasing_arc_ending_angle(
            null_arc_info)

    angle_array = np.linspace(null_arc_info['arc_starting_angle'].values,
                              null_arc_info['arc_ending_angle'].values, num_points_on_each_arc).T.reshape(-1)

    # Repeat necessary values to match the length of angle_array
    repeated_values = {col: np.repeat(null_arc_info[col].values, num_points_on_each_arc) for col in ['arc_point_index', 'arc_ff_index', 'all_arc_radius', 'center_x', 'center_y',
                                                                                                     'arc_ff_x', 'arc_ff_y', 'arc_starting_angle']}

    # Create DataFrame for arc points
    arc_df = pd.DataFrame({
        'arc_point_index': repeated_values['arc_point_index'],
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


def get_opt_arc_end_points_closest_to_stop(null_arc_info, stop_and_ref_point_info, reward_boundary_radius=25):
    if len(null_arc_info) != len(stop_and_ref_point_info):
        print('When calling get_opt_arc_end_points_closest_to_stop, the number of rows in null_arc_info and stop_and_ref_point_info do not match.'
              'Some points do not have a valid null arc and might be missed.')

    # if cur_ff_index don't exist, then try renaming ff_index, ff_x, ff_y; if the latter don't exist, raise an error
    if 'cur_ff_index' not in stop_and_ref_point_info.columns:
        try:
            stop_and_ref_point_info.rename(columns={'ff_index': 'cur_ff_index',
                                                    'ff_x': 'cur_ff_x',
                                                    'ff_y': 'cur_ff_y'}, inplace=True)
        except:
            raise ValueError(
                'cur_ff_index, cur_ff_x, cur_ff_y must exist in get_opt_arc_end_points_closest_to_stop. If not, at least ff_index, ff_x, ff_y must exist.')

    null_arc_info = null_arc_info.merge(stop_and_ref_point_info[[
                                        'cur_ff_index', 'cur_ff_x', 'cur_ff_y']].drop_duplicates(), left_on='arc_ff_index', right_on='cur_ff_index', how='left')
    null_arc_info.rename(
        columns={'cur_ff_x': 'arc_ff_x', 'cur_ff_y': 'arc_ff_y'}, inplace=True)

    arc_df = get_points_on_each_arc(null_arc_info, extend_arc_angle=True)
    arc_df.rename(columns={'arc_point_index': 'point_index'}, inplace=True)
    # Filter points within 25 units of the firefly
    arc_df = arc_df[arc_df['distance_to_ff'] <= reward_boundary_radius].copy()

    # Merge with stops data
    arc_df = arc_df.merge(stop_and_ref_point_info[[
                          'cur_ff_index', 'point_index', 'stop_x', 'stop_y']].drop_duplicates(), on=['cur_ff_index', 'point_index'], how='left')
    # Calculate distance to stop
    arc_df['distance_to_stop'] = np.sqrt(
        (arc_df['x'] - arc_df['stop_x'])**2 + (arc_df['y'] - arc_df['stop_y'])**2)

    # Find the arc points closest to each stop
    index_of_arc_rows_closest_to_stop = arc_df.groupby(
        ['cur_ff_index', 'point_index'])['distance_to_stop'].idxmin()
    arc_rows_closest_to_stop = arc_df.loc[index_of_arc_rows_closest_to_stop].reset_index(
        drop=True)

    return arc_rows_closest_to_stop


def get_opt_arc_end_points_when_first_reaching_visible_boundary(null_arc_info,
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
        print(f'Note: When calling get_opt_arc_end_points_when_first_reaching_visible_boundary, there are {len(too_big_angle_rows)} points that are more than 90 degrees away from the starting angle of the arc.' +
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
    print(f'Note: When calling get_opt_arc_end_points_when_first_reaching_visible_boundary, there are {len(null_arc_info) - len(arc_rows_to_first_reach_boundary)} points out of {len(null_arc_info)} points that are not within the visible boundary of the firefly.' +
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


def make_cur_and_nxt_ff_from_ref_df(nxt_ff_df_final, cur_ff_df_final, include_arc_info=True):
    # Define shared and relevant columns
    shared_columns = ['monkey_x', 'monkey_y', 'monkey_angle', 'curv_of_traj', 'point_index',
                      'stop_point_index', 'monkey_angle_before_stop', 'd_heading_of_traj']

    relevant_columns = ['ff_index', 'ff_x', 'ff_y', 'ff_distance', 'ff_angle', 'ff_angle_boundary',
                        'opt_arc_curv', 'opt_arc_measure', 'opt_arc_radius', 'opt_arc_end_direction', 'opt_arc_d_heading', 'cntr_arc_end_x', 'cntr_arc_end_y',
                        'cntr_arc_curv', 'cntr_arc_measure', 'cntr_arc_radius', 'cntr_arc_end_direction', 'cntr_arc_d_heading', 'opt_arc_end_x', 'opt_arc_end_y',
                        'valid_null_arc'
                        ]

    if not include_arc_info:
        # remove arc related columns
        shared_columns = [col for col in shared_columns if col not in [
            'd_heading_of_traj', 'curv_of_traj']]
        relevant_columns = [
            col for col in relevant_columns if ('arc' not in col) & (col != 'valid_null_arc')]

    if 'valid_null_arc' not in cur_ff_df_final.columns:
        # valid_null_arc is added when modifying rows with big ff angles, so if no modification occurred, then it might be missing.
        cur_ff_df_final['valid_null_arc'] = 1
    if 'valid_null_arc' not in nxt_ff_df_final.columns:
        nxt_ff_df_final['valid_null_arc'] = 1

    # Create a copy of the shared columns from nxt_ff_df_final and rename them
    cur_and_nxt_ff_from_ref_df = cur_ff_df_final[shared_columns].copy()

    cur_and_nxt_ff_from_ref_df.rename(columns={'point_index': 'ref_point_index',
                                               'monkey_x': 'ref_monkey_x',
                                               'monkey_y': 'ref_monkey_y',
                                               'monkey_angle': 'ref_monkey_angle',
                                               'curv_of_traj': 'ref_curv_of_traj'}, inplace=True, errors='ignore')

    relevant_columns = [
        col for col in relevant_columns if col in nxt_ff_df_final.columns]

    # Create copies of the relevant columns from nxt_ff_df_final2 and cur_ff_df_final2 and rename them
    nxt_ff_df_final2 = nxt_ff_df_final[relevant_columns].copy()
    nxt_ff_df_final2.columns = [
        'nxt_'+col for col in nxt_ff_df_final2.columns.tolist()]
    nxt_ff_df_final2['stop_point_index'] = nxt_ff_df_final['stop_point_index']

    cur_ff_df_final2 = cur_ff_df_final[relevant_columns].copy()
    cur_ff_df_final2.columns = [
        'cur_'+col for col in cur_ff_df_final2.columns.tolist()]
    cur_ff_df_final2['stop_point_index'] = cur_ff_df_final['stop_point_index']

    # Merge cur_and_nxt_ff_from_ref_df, nxt_ff_df_final2, and cur_ff_df_final2
    cur_and_nxt_ff_from_ref_df = cur_and_nxt_ff_from_ref_df.merge(
        nxt_ff_df_final2, how='left', on='stop_point_index')
    cur_and_nxt_ff_from_ref_df = cur_and_nxt_ff_from_ref_df.merge(
        cur_ff_df_final2, how='left', on='stop_point_index')

    # Calculate landing headings
    if include_arc_info:
        if 'cur_cntr_arc_d_heading' in cur_and_nxt_ff_from_ref_df.columns:
            cur_and_nxt_ff_from_ref_df['cur_cntr_arc_end_heading'] = cur_and_nxt_ff_from_ref_df['ref_monkey_angle'] + \
                cur_and_nxt_ff_from_ref_df['cur_cntr_arc_d_heading']
            cur_and_nxt_ff_from_ref_df['cur_opt_arc_end_heading'] = cur_and_nxt_ff_from_ref_df['ref_monkey_angle'] + \
                cur_and_nxt_ff_from_ref_df['cur_opt_arc_d_heading']

            cur_and_nxt_ff_from_ref_df['nxt_cntr_arc_end_heading'] = cur_and_nxt_ff_from_ref_df['ref_monkey_angle'] + \
                cur_and_nxt_ff_from_ref_df['nxt_cntr_arc_d_heading']
            cur_and_nxt_ff_from_ref_df['nxt_opt_arc_end_heading'] = cur_and_nxt_ff_from_ref_df['ref_monkey_angle'] + \
                cur_and_nxt_ff_from_ref_df['nxt_opt_arc_d_heading']

    return cur_and_nxt_ff_from_ref_df


def make_heading_info_df(cur_and_nxt_ff_from_ref_df, stops_near_ff_df, monkey_information, ff_real_position_sorted):
    # Select relevant columns from stops_near_ff_df
    heading_info_df = stops_near_ff_df[['stop_point_index', 'stop_x', 'stop_y', 'stop_time',
                                        'cur_ff_index', 'cur_ff_x', 'cur_ff_y', 'cur_ff_cluster_50_size',
                                        'point_index_before_stop',  'monkey_angle_before_stop',
                                        'next_stop_point_index', 'next_stop_time', 'cum_distance_between_two_stops',
                                        'curv_range', 'curv_iqr', 'nxt_ff_index', 'nxt_ff_x', 'nxt_ff_y',
                                        'CUR_time_ff_first_seen_bbas', 'CUR_time_ff_last_seen_bbas',
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

    # Merge with cur_and_nxt_ff_from_ref_df to get landing headings
    columns_to_keep = ['stop_point_index', 'ref_point_index',
                       'cur_cntr_arc_d_heading', 'nxt_cntr_arc_d_heading',
                       'cur_opt_arc_d_heading', 'nxt_opt_arc_d_heading',
                       'cur_cntr_arc_end_heading', 'nxt_cntr_arc_end_heading',
                       'cur_opt_arc_end_heading', 'nxt_opt_arc_end_heading',
                       'cur_cntr_arc_curv', 'cur_opt_arc_curv', 'nxt_cntr_arc_curv', 'nxt_opt_arc_curv',
                       'cur_cntr_arc_end_x', 'cur_cntr_arc_end_y', 'nxt_cntr_arc_end_x', 'nxt_cntr_arc_end_y',
                       'cur_opt_arc_end_x', 'cur_opt_arc_end_y', 'nxt_opt_arc_end_x', 'nxt_opt_arc_end_y',
                       'd_heading_of_traj', 'ref_monkey_angle', 'ref_curv_of_traj',
                       'cur_valid_null_arc', 'nxt_valid_null_arc',
                       ]
    columns_to_keep = [
        col for col in columns_to_keep if col in cur_and_nxt_ff_from_ref_df.columns]
    heading_info_df = heading_info_df.merge(
        cur_and_nxt_ff_from_ref_df[columns_to_keep], how='left', on='stop_point_index')

    # Calculate angles from monkey before stop to nxt ff and from cur ff null arc landing position to alternative ff
    heading_info_df['angle_from_m_before_stop_to_cur_ff'] = specific_utils.calculate_angles_to_ff_centers(
        heading_info_df['cur_ff_x'], heading_info_df['cur_ff_y'], heading_info_df['mx_before_stop'], heading_info_df['my_before_stop'], heading_info_df['monkey_angle_before_stop'])
    heading_info_df['angle_from_stop_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
        heading_info_df['nxt_ff_x'], heading_info_df['nxt_ff_y'], heading_info_df['mx_before_stop'], heading_info_df['my_before_stop'], heading_info_df['monkey_angle_before_stop'])

    if 'cur_opt_arc_end_x' in heading_info_df.columns:
        heading_info_df['angle_opt_cur_end_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
            heading_info_df['nxt_ff_x'], heading_info_df['nxt_ff_y'], heading_info_df['cur_opt_arc_end_x'], heading_info_df['cur_opt_arc_end_y'], heading_info_df['cur_opt_arc_end_heading'])
        heading_info_df['angle_cntr_cur_end_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(
            heading_info_df['nxt_ff_x'], heading_info_df['nxt_ff_y'], heading_info_df['cur_cntr_arc_end_x'], heading_info_df['cur_cntr_arc_end_y'], heading_info_df['cur_cntr_arc_end_heading'])

    # The following two columns are originally from calculate_info_based_on_monkey_angles
    heading_info_df['angle_from_cur_ff_to_stop'] = specific_utils.calculate_angles_to_ff_centers(ff_x=heading_info_df['stop_x'].values, ff_y=heading_info_df['stop_y'],
                                                                                                 mx=heading_info_df['cur_ff_x'].values, my=heading_info_df['cur_ff_y'], m_angle=heading_info_df['monkey_angle_before_stop'])
    heading_info_df['angle_from_cur_ff_to_nxt_ff'] = specific_utils.calculate_angles_to_ff_centers(ff_x=heading_info_df['nxt_ff_x'].values, ff_y=heading_info_df['nxt_ff_y'],
                                                                                                   mx=heading_info_df['cur_ff_x'].values, my=heading_info_df['cur_ff_y'], m_angle=heading_info_df['monkey_angle_before_stop'])
    
    heading_info_df = build_factor_comp.process_heading_info_df(
        heading_info_df)
    return heading_info_df


def get_ang_traj_nxt_and_ang_cur_nxt(heading_info_df):
    heading_info_df = heading_info_df.copy()
    # print the number of rows in heading_info_df that contain NaN values
    print(
        f'Number of rows with NaN values in heading_info_df: {heading_info_df.isnull().any(axis=1).sum()} out of {heading_info_df.shape[0]} rows, but they are not dropped. The columns with NaN values are:')
    # print columns with NaN values and number of NaN values in each column
    print(heading_info_df.isnull().sum()[heading_info_df.isnull().sum() > 0])

    # heading_info_df.dropna(inplace=True)
    ang_traj_nxt = heading_info_df['angle_from_stop_to_nxt_ff'].values.reshape(
        -1)
    ang_cur_nxt = heading_info_df['angle_opt_cur_end_to_nxt_ff'].values.reshape(
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
        plot_cvn_utils.plot_ang_traj_nxt_vs_ang_cur_nxt(
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


def retrieve_df_based_on_ref_point(monkey_name, ref_point_mode, ref_point_value, test_or_control, data_folder_name, df_partial_path,
                                   target_var_name=None):
    df_path = os.path.join(data_folder_name, df_partial_path, test_or_control)
    os.makedirs(df_path, exist_ok=True)
    df_name = find_cvn_utils.get_df_name_by_ref(
        monkey_name, ref_point_mode, ref_point_value)
    retrieved_df = pd.read_csv(os.path.join(df_path, df_name), index_col=0)
    if target_var_name is None:
        target_var_name = df_name
    print(
        f'Successfully retrieved {target_var_name} from {os.path.join(df_path, df_name)}')
    return retrieved_df
