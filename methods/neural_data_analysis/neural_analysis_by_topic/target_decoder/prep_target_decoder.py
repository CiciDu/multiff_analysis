import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
import warnings
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor


def add_lagged_target_columns(y_var_lags, y_var, target_df_lags, max_y_lag_number, target_columns=None):
    """
    Process target columns in lagged data by matching with temp_target_df based on point_index and target_index.
    """

    if 'target_index' not in y_var_lags.columns:
        y_var_lags = y_var_lags.merge(
            y_var[['bin', 'target_index']], on='bin', how='left')

    if target_columns is None:
        target_columns = [
            col for col in target_df_lags.columns if 'target' in col]

    # Create a list to store all the new columns
    all_new_columns = []

    for lag_number in range(-max_y_lag_number, max_y_lag_number + 1):
        lag_suffix = f'_{lag_number}'
        point_index_col = f'point_index{lag_suffix}'

        if point_index_col not in y_var_lags.columns:
            raise KeyError(f"Warning: {point_index_col} not found in columns")

        # retrieve target info from target_df_lags
        temp_target_df = y_var_lags[[
            point_index_col, 'target_index', 'bin']].copy()
        temp_target_df.rename(
            columns={point_index_col: 'point_index'}, inplace=True)

        temp_target_df = temp_target_df.merge(
            target_df_lags, on=['point_index', 'target_index'], how='left')

        # take out y_var_lags's columns related to target from the new target_df
        temp_target_df2 = temp_target_df[target_columns]

        # Add lag_number suffix to all columns
        rename_dict = {
            col: f"{col}{lag_suffix}" for col in temp_target_df2.columns}
        temp_target_df2 = temp_target_df2.rename(columns=rename_dict)

        # Add to our list of new columns
        all_new_columns.append(temp_target_df2)

    # Concatenate all
    all_new_columns = pd.concat(all_new_columns, axis=1)
    # Remove any columns from y_var_lags that are in all_new_columns
    columns_for_y_var = [
        col for col in y_var_lags.columns if col not in all_new_columns.columns]
    y_var_lags = y_var_lags[columns_for_y_var].copy()

    y_var_lags = pd.concat([y_var_lags, all_new_columns], axis=1).copy()

    return y_var_lags


def initialize_target_df_lags(y_var, max_y_lag_number, bin_width):
    # for each target index and its point index, get the point index between point_index_min - lag_number and point_index_max + lag_number
    # then get target info for all the 'target_index' & 'point_index' pair
    point_duration = 0.017
    # use times 1.5 here to make sure that all bins will be covered)
    point_index_per_lag = math.ceil(bin_width / point_duration * 1.5)

    unique_targets = y_var['target_index'].unique()
    min_index = y_var.groupby('target_index').min()['point_index']
    max_index = y_var.groupby('target_index').max()['point_index']

    point_index_list = []
    target_list = []
    for target in unique_targets:
        current_point_index_list = list(range(
            min_index[target] - max_y_lag_number * point_index_per_lag, max_index[target] + max_y_lag_number * point_index_per_lag + 1))
        point_index_list.extend(current_point_index_list)
        target_list.extend([target] * len(current_point_index_list))

    target_df_lags = pd.DataFrame(
        {'point_index': point_index_list, 'target_index': target_list})

    target_df_lags['target_index'] = target_df_lags['target_index'].astype(int)

    return target_df_lags


def fill_na_in_last_seen_columns(target_df_lags):
    # ffill and bfill on last_seen_columns in target_df_lags
    last_seen_columns = [
        col for col in target_df_lags.columns if 'last_seen' in col]
    target_df_lags.sort_values(
        by=['target_index', 'point_index'], inplace=True)
    target_df_lags[last_seen_columns] = target_df_lags.groupby(
        'target_index', as_index=False)[last_seen_columns].ffill()
    target_df_lags[last_seen_columns] = target_df_lags.groupby(
        'target_index', as_index=False)[last_seen_columns].bfill()
    return target_df_lags


def add_target_info_based_on_target_index_and_point_index(target_df, monkey_information, ff_real_position_sorted, ff_dataframe, ff_caught_T_new, curv_of_traj_df):

    assert 'point_index' in target_df.columns, "point_index column is required"
    assert 'target_index' in target_df.columns, "target_index column is required"

    target_df = target_df.merge(monkey_information[[
        'point_index', 'time', 'monkey_x', 'monkey_y', 'monkey_angle', 'cum_distance']],
        on='point_index', how='left')
    target_df = prep_target_data._add_target_df_info(
        target_df, monkey_information, ff_real_position_sorted, ff_dataframe, ff_caught_T_new)

    target_df = prep_target_data.add_columns_to_target_df(target_df)

    target_df = add_target_opt_arc_dheading_to_target_df(
        target_df, curv_of_traj_df, monkey_information, ff_caught_T_new)
    return target_df


def add_target_opt_arc_dheading_to_target_df(target_df, curv_of_traj_df, monkey_information, ff_caught_T_new):

    if 'target_opt_arc_dheading' not in target_df.columns:

        ff_df = target_df[['point_index', 'target_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                           'target_x', 'target_y', 'target_distance', 'target_angle', 'target_angle_to_boundary']].drop_duplicates()

        ff_df = ff_df.rename(columns={'target_x': 'ff_x', 'target_y': 'ff_y', 'target_angle': 'ff_angle',
                                      'target_index': 'ff_index', 'target_distance': 'ff_distance', 'target_angle_to_boundary': 'ff_angle_boundary'})

        curv_df = curvature_utils.make_curvature_df(ff_df, curv_of_traj_df, clean=False,
                                                    remove_invalid_rows=False,
                                                    invalid_curvature_ok=True,
                                                    ignore_error=True,
                                                    monkey_information=monkey_information,
                                                    ff_caught_T_new=ff_caught_T_new)

        curv_df = curv_df.rename(columns={'opt_arc_d_heading': 'target_opt_arc_dheading',
                                          'ff_index': 'target_index'})

        target_df = target_df.merge(curv_df[['point_index', 'target_index', 'target_opt_arc_dheading']],
                                    on=['point_index', 'target_index'], how='left')
    return target_df


def find_single_vis_target_df(target_clust_last_vis_df, monkey_information, ff_caught_T_new, max_visibility_window=10):
    # check if target_clust_last_vis_df['nearby_vis_ff_indices'] is a string
    if isinstance(target_clust_last_vis_df['nearby_vis_ff_indices'].iloc[0], str):
        target_clust_last_vis_df['nearby_vis_ff_indices'] = target_clust_last_vis_df['nearby_vis_ff_indices'].apply(
            lambda x: [int(i) for i in x.strip('[]').split(',') if i.strip().isdigit()])

    target_clust_last_vis_df['num_nearby_vis_ff'] = target_clust_last_vis_df['nearby_vis_ff_indices'].apply(
        lambda x: len(x))

    # add ff_caught_time and ff_caught_point_index
    target_clust_last_vis_df['ff_caught_time'] = ff_caught_T_new[target_clust_last_vis_df['target_index'].values]
    target_clust_last_vis_df['ff_caught_point_index'] = np.searchsorted(
        monkey_information['time'], target_clust_last_vis_df['ff_caught_time'].values)

    # drop the rows where target is in a cluster (we want to preserve cases where monkey is going toward a single target, not a cluster)
    single_vis_target_df = target_clust_last_vis_df[
        target_clust_last_vis_df['num_nearby_vis_ff'] == 1]

    # also drop the rows where the ff_caught_time is within 5s of either the min or the max time in monkey_information, so that when getting lagged columns, there is enough information
    max_time_in_ff_dataframe = monkey_information['time'].max()
    min_time_in_ff_dataframe = monkey_information['time'].min()
    single_vis_target_df = single_vis_target_df[(single_vis_target_df['ff_caught_time']
                                                < max_time_in_ff_dataframe - 5) & (single_vis_target_df['ff_caught_time']
                                                > min_time_in_ff_dataframe + 5)]

    # also drop the rows where the last visible ff in the target cluster is not the target itself
    single_vis_target_df = single_vis_target_df[single_vis_target_df['last_vis_ff_index']
                                                == single_vis_target_df['target_index']].copy()

    single_vis_target_df['last_vis_time'] = monkey_information.loc[
        single_vis_target_df['last_vis_point_index'].values, 'time'].values

    # drop the rows where last_vis_time is less than ff_caught_time - max_visibility_window
    single_vis_target_df = single_vis_target_df[single_vis_target_df['last_vis_time']
                                                >= single_vis_target_df['ff_caught_time'] - max_visibility_window]

    # print percentage of single_vis_target_df
    print("Percentage of targets not in a visible cluster out of all targets", len(
        single_vis_target_df) / len(target_clust_last_vis_df) * 100)
    return single_vis_target_df


def add_target_info_to_behav_data_by_point(behav_data_by_bin, target_df):
    # drop columns in target_df that are duplicated in behav_data_by_bin
    columns_to_drop = [
        col for col in target_df.columns if col in behav_data_by_bin.columns]
    columns_to_drop.remove('point_index')
    target_df = target_df.drop(columns=columns_to_drop)

    behav_data_by_bin = behav_data_by_bin.merge(
        target_df, on='point_index', how='left')

    return behav_data_by_bin


def make_pursuit_data_all(single_vis_target_df, behav_data_by_bin):
    point_index_list = []
    segment_list = []
    target_list = []

    for index, row in single_vis_target_df.iterrows():
        point_index = range(row['last_vis_point_index'],
                            row['ff_caught_point_index'])
        point_index_list.extend(point_index)
        segment_list.extend([index] * len(point_index))
        target_list.extend([row['target_index']] * len(point_index))

    point_index_df = pd.DataFrame(
        {'point_index': point_index_list,
         'segment': segment_list,
         'target_index': target_list,
         })

    pursuit_data_all = behav_data_by_bin[behav_data_by_bin['point_index'].isin(
        point_index_list)].copy()

    pursuit_data_all = pursuit_data_all.merge(
        point_index_df, on=['point_index', 'target_index'], how='left')
    
    pursuit_data_all.sort_values(by=['segment', 'bin'], inplace=True)
    
    # for each segment, assign the first point's segment_start_dummy to 1 and the last point's segment_end_dummy to 1
    pursuit_data_all['segment_start_dummy'] = 0
    pursuit_data_all['segment_end_dummy'] = 0
    # Set start and end flags for each segment
    for segment, group in pursuit_data_all.groupby('segment'):
        first_idx = group.index[0]
        last_idx = group.index[-1]
        pursuit_data_all.loc[first_idx, 'segment_start_dummy'] = 1
        pursuit_data_all.loc[last_idx, 'segment_end_dummy'] = 1

    # # drop NA if any
    # if pursuit_data_all.isnull().any(axis=1).sum() > 0:
    #     print(
    #         f'Number of rows with NaN values in pursuit_data_all: {pursuit_data_all.isnull().any(axis=1).sum()} out of {pursuit_data_all.shape[0]} rows. The rows with NaN values will be dropped.')
    #     # drop rows with NA in x_var_df
    #     pursuit_data_all = pursuit_data_all.dropna()

    # add segment info
    org_len = len(behav_data_by_bin)
    pursuit_data_all = add_seg_info_to_pursuit_data_all_col(pursuit_data_all)
    new_len = len(pursuit_data_all)
    print(f'{new_len} rows of {org_len} rows ({round(new_len/org_len * 100, 1)}%) of behav_data_by_bin are preserved after taking out chunks between target last-seen time and capture time')

    return pursuit_data_all


def add_seg_info_to_pursuit_data_all_col(pursuit_data_all):

    # get seg_start_time as min time of segment
    pursuit_data_all['seg_start_time'] = pursuit_data_all.groupby('segment')[
        'bin_start_time'].transform('min')

    # get seg_end_time as max time of segment
    pursuit_data_all['seg_end_time'] = pursuit_data_all.groupby('segment')[
        'bin_end_time'].transform('max')

    # get seg_duration as seg_end_time - seg_start_time
    pursuit_data_all['seg_duration'] = pursuit_data_all['seg_end_time'] - \
        pursuit_data_all['seg_start_time']

    return pursuit_data_all


def _add_curv_info_to_behav_data_by_point(behav_data_by_bin, curv_of_traj_df, monkey_information, ff_caught_T_new):
    ff_df = behav_data_by_bin[['point_index', 'target_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                               'target_x', 'target_y', 'target_distance', 'target_angle', 'target_angle_to_boundary']]
    ff_df = ff_df.rename(columns={'target_x': 'ff_x', 'target_y': 'ff_y', 'target_angle': 'ff_angle',
                                  'target_index': 'ff_index', 'target_distance': 'ff_distance', 'target_angle_to_boundary': 'ff_angle_boundary'})

    curv_df = curvature_utils.make_curvature_df(ff_df, curv_of_traj_df, clean=False,
                                                remove_invalid_rows=False,
                                                invalid_curvature_ok=True,
                                                ignore_error=True,
                                                monkey_information=monkey_information,
                                                ff_caught_T_new=ff_caught_T_new)
    behav_data_by_bin = behav_data_by_bin.merge(curv_df[[
        'point_index', 'curv_of_traj', 'opt_arc_d_heading']].drop_duplicates(), on='point_index', how='left')
    behav_data_by_bin.rename(columns={
        'curv_of_traj': 'traj_curv', 'opt_arc_d_heading': 'target_opt_arc_dheading'}, inplace=True)

    return behav_data_by_bin


def _process_na(behav_data_by_bin):
    # forward fill gaze columns
    gaze_columns = [
        'gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_mky_view_angle', 'gaze_world_x', 'gaze_world_y',
        'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'gaze_mky_view_angle_l',
        'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'gaze_mky_view_angle_r',
        'gaze_world_x_l', 'gaze_world_y_l', 'gaze_world_x_r', 'gaze_world_y_r'
    ]
    # Convert inf values to NA for gaze columns
    behav_data_by_bin[gaze_columns] = behav_data_by_bin[gaze_columns].replace(
        [np.inf, -np.inf], np.nan)
    behav_data_by_bin[gaze_columns] = behav_data_by_bin[gaze_columns].ffill(
    )

    # Check for any remaining NA values
    na_rows, na_cols = general_utils.check_na_in_df(
        behav_data_by_bin, 'behav_data_by_bin')
    return na_rows, na_cols


def _get_subset_key_words_and_all_column_subsets_for_corr(y_var_lags):
    subset_key_words = ['_x',
                        '_y',
                        'angle_OR_curv_OR_dw',
                        'distance_OR_dv_OR_visible_OR_rel_y_OR_valid_view_OR_time_since_OR_gaze_world_y_OR_Dz',
                        'speed_OR_dw_OR_delta_OR_traj_OR_dv_OR_stop_OR_catching_ff',
                        'x_r_OR_x_l_OR_y_r_OR_y_l',
                        'LD_or_RD_or_gaze_or_view',
                        'ff_or_target']
    all_column_subsets = [
        [col for col in y_var_lags.columns if '_x' in col],
        [col for col in y_var_lags.columns if '_y' in col],
        [col for col in y_var_lags.columns if (
            'angle' in col) or ('curv' in col) or ('dw' in col)],
        [col for col in y_var_lags.columns if ('distance' in col) or ('dv' in col) or ('visible' in col) or (
            'rel_y' in col) or ('valid_view' in col) or ('time_since' in col) or ('gaze_world_y' in col) or ('Dz' in col)],
        [col for col in y_var_lags.columns if ('speed' in col) or (
            'dw' in col) or ('delta' in col) or ('traj' in col) or ('dv' in col) or ('stop' in col) or ('catching_ff' in col)],
        [col for col in y_var_lags.columns if ('x_r' in col) or (
            'y_r' in col) or ('x_l' in col) or ('y_l' in col)],
        [col for col in y_var_lags.columns if ('LD' in col) or (
            'RD' in col) or ('gaze' in col) or ('view' in col)],
        [col for col in y_var_lags.columns if ('ff' in col) or (
            'target' in col)],
    ]
    return subset_key_words, all_column_subsets


def _get_subset_key_words_and_all_column_subsets_for_vif(y_var_lags):
    subset_key_words = ['target_x_OR_monkey_x',
                        'target_y_OR_monkey_y',
                        'distance_OR_visible_OR_rel_y_OR_valid_view_OR_time_since_OR_gaze_world_y_OR_Dz',
                        'target_angle_OR_monkey_angle_OR_ff_angle_OR_last_seen_angle',
                        'dw',
                        'curv_or_dw_or_heading',
                        'speeddummy_OR_delta_OR_traj_OR_catching_ff',
                        'speed_OR_ddv_OR_dw_OR_stop',
                        'gaze_mky_view_angle',
                        'LD_or_RD_or_gaze_mky_view_angle',
                        'x_r_or_y_r_or_RD',
                        'x_l_or_y_l_or_LD',
                        'ff_or_target_Except_catching_ff']
    all_column_subsets = [
        [col for col in y_var_lags.columns if (
            'target_x' in col) or ('monkey_x' in col)],
        [col for col in y_var_lags.columns if (
            'target_y' in col) or ('monkey_y' in col)],
        [col for col in y_var_lags.columns if ('distance' in col) or ('visible' in col) or (
            'rel_y' in col) or ('valid_view' in col) or ('time_since' in col) or ('gaze_world_y' in col) or ('Dz' in col)],
        [col for col in y_var_lags.columns if ('target_angle' in col) or (
            'monkey_angle' in col) or ('ff_angle' in col) or ('last_seen_angle' in col)],
        [col for col in y_var_lags.columns if ('dw' in col)],
        [col for col in y_var_lags.columns if ('curv' in col) or (
            'dw' in col) or ('heading' in col)],
        [col for col in y_var_lags.columns if ('speeddummy' in col) or (
            'delta' in col) or ('traj' in col) or ('catching_ff' in col)],
        [col for col in y_var_lags.columns if ('speed' in col) or (
            'ddv' in col) or ('dw' in col) or ('stop' in col)],
        [col for col in y_var_lags.columns if (
            'gaze_mky_view_angle' in col)],
        [col for col in y_var_lags.columns if ('LD' in col) or (
            'RD' in col) or ('gaze_mky_view_angle' in col)],
        [col for col in y_var_lags.columns if (
            'x_r' in col) or ('y_r' in col) or ('RD' in col)],
        [col for col in y_var_lags.columns if (
            'x_l' in col) or ('y_l' in col) or ('LD' in col)],
        [col for col in y_var_lags.columns if (('ff' in col) or (
            'target' in col)) and ('catching_ff' not in col)],
    ]
    return subset_key_words, all_column_subsets


def remove_zero_var_cols(data):
    # if data is a df, remove columns with zero variance
    if isinstance(data, pd.DataFrame):
        zero_var_cols = data.columns[data.var() == 0]
        if len(zero_var_cols) > 0:
            print(
                f"Removing {len(zero_var_cols)} columns with zero variance: {zero_var_cols.tolist()}")
            data = data.drop(columns=zero_var_cols)
    elif isinstance(data, np.ndarray):
        zero_var_cols = np.where(np.var(data, axis=0) == 0)[0]
        if len(zero_var_cols) > 0:
            print(
                f"Removing {len(zero_var_cols)} columns with zero variance: {zero_var_cols.tolist()}")
            data = np.delete(data, zero_var_cols, axis=1)
    else:
        raise ValueError(
            f"Data is not a pandas DataFrame or numpy array: {type(data)}")

    return data
