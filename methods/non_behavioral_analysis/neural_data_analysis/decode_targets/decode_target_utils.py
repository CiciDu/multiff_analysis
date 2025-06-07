import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
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


def process_target_columns_in_lags(y_var_lags, target_df_lags, max_y_lag_number):
    """
    Process target columns in lagged data by matching with temp_target_df based on point_index and target_index.
    """

    for lag_number in range(-max_y_lag_number, max_y_lag_number + 1):
        if lag_number == 0:
            continue
        else:
            lag_suffix = f'_{lag_number}'
            point_index_col = f'point_index{lag_suffix}'

            if point_index_col not in y_var_lags.columns:
                raise KeyError(
                    f"Warning: {point_index_col} not found in columns")

            # calculate target info anew
            temp_target_df = y_var_lags[[
                point_index_col, 'target_index', 'bin']].copy()

            temp_target_df.rename(columns={
                point_index_col: 'point_index'}, inplace=True)

            temp_target_df = temp_target_df.merge(
                target_df_lags, on=['point_index', 'target_index'], how='left')

            # take out y_var_lags's columns related to target from the new target_df; raise an error if any target_cols are not in temp_target_df
            target_cols = [
                col for col in y_var_lags.columns if (col.endswith(lag_suffix) & ('target' in col))]
            # strip off suffix
            target_cols_stripped = [col.replace(
                lag_suffix, '') for col in target_cols]

            temp_target_df = temp_target_df[target_cols_stripped]

            # Add lag_number suffix to all columns
            rename_dict = {col: f"{col}{lag_suffix}"
                           for col in temp_target_df.columns}
            temp_target_df = temp_target_df.rename(columns=rename_dict)

            # update y_var_lags with the new temp_target_df
            y_var_lags[target_cols_stripped] = temp_target_df[target_cols_stripped]

    return y_var_lags


def make_target_df_lags(y_var, max_y_lag_number):
    # for each target index and its point index, get the point index between point_index_min - lag_number and point_index_max + lag_number
    # then get target info for all the 'target_index' & 'point_index' pair

    unique_targets = y_var['target_index'].unique()
    min_index = y_var.groupby('target_index').min()['point_index']
    max_index = y_var.groupby('target_index').max()['point_index']

    point_index_list = []
    target_list = []
    for target in unique_targets:
        current_point_index_list = list(range(
            min_index[target] - max_y_lag_number, max_index[target] + max_y_lag_number + 1))
        point_index_list.extend(current_point_index_list)
        target_list.extend([target] * len(current_point_index_list))

    target_df_lags = pd.DataFrame(
        {'point_index': point_index_list, 'target_index': target_list})
    return target_df_lags

# def process_target_columns_in_lags(y_var_lags, lag_number, monkey_information, ff_real_position_sorted, ff_dataframe, ff_caught_T_new, curv_of_traj_df):
#     """
#     Process target columns in lagged data by matching with temp_target_df based on point_index and target_index.
#     """

#     # Get columns for this lag_number
#     lag_suffix = f'_{lag_number}'
#     point_index_col = f'point_index{lag_suffix}'

#     if point_index_col not in y_var_lags.columns:
#         raise KeyError(f"Warning: {point_index_col} not found in columns")

#     # calculate target info anew
#     temp_target_df = y_var_lags[[
#         point_index_col, 'target_index', 'bin']].copy()
#     temp_target_df.rename(columns={
#                           point_index_col: 'point_index'}, inplace=True)

#     temp_target_df = add_target_info_based_on_target_index_and_point_index(temp_target_df, monkey_information, ff_real_position_sorted,
#                                                                            ff_dataframe, ff_caught_T_new, curv_of_traj_df)

#     # take out y_var_lags's columns related to target from the new target_df; raise an error if any target_cols are not in temp_target_df
#     target_cols = [
#         col for col in y_var_lags.columns if (col.endswith(lag_suffix) & ('target' in col))]
#     # strip off suffix
#     target_cols = [col.replace(lag_suffix, '') for col in target_cols]

#     temp_target_df = temp_target_df[target_cols]

#     # Add lag_number suffix to all columns
#     rename_dict = {col: f"{col}{lag_suffix}"
#                    for col in temp_target_df.columns}
#     temp_target_df = temp_target_df.rename(columns=rename_dict)

#     return temp_target_df


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

        curv_df = curv_df.rename(columns={'optimal_arc_d_heading': 'target_opt_arc_dheading',
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


def add_target_info_to_behav_data_all(behav_data_all, target_df):
    # drop columns in target_df that are duplicated in behav_data_all
    columns_to_drop = [
        col for col in target_df.columns if col in behav_data_all.columns]
    columns_to_drop.remove('point_index')
    target_df = target_df.drop(columns=columns_to_drop)

    behav_data_all = behav_data_all.merge(
        target_df, on='point_index', how='left')

    return behav_data_all


def make_pursuit_data_all(single_vis_target_df, behav_data_all):
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
         'target_index': target_list
         })

    pursuit_data_all = behav_data_all[behav_data_all['point_index'].isin(
        point_index_list)].copy()

    pursuit_data_all = pursuit_data_all.merge(
        point_index_df, on=['point_index', 'target_index'], how='left')

    # # drop NA if any
    # if pursuit_data_all.isnull().any(axis=1).sum() > 0:
    #     print(
    #         f'Number of rows with NaN values in pursuit_data_all: {pursuit_data_all.isnull().any(axis=1).sum()} out of {pursuit_data_all.shape[0]} rows. The rows with NaN values will be dropped.')
    #     # drop rows with NA in x_var_df
    #     pursuit_data_all = pursuit_data_all.dropna()

    # add segment info
    org_len = len(behav_data_all)
    pursuit_data_all = add_seg_info_to_pursuit_data_all_col(pursuit_data_all)
    new_len = len(pursuit_data_all)
    print(f'{new_len} rows of {org_len} rows ({round(new_len/org_len * 100, 1)}%) of behav_data_all are preserved after taking out chunks between target last-seen time and capture time')

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
