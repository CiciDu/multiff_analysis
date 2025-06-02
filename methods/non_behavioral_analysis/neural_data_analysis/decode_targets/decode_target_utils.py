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


def find_single_vis_target_df(target_clust_last_vis_df, monkey_information, ff_caught_T_new):
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

    # Add time since target last seen
    behav_data_all['target_last_seen_time_abs'] = behav_data_all['time'] + \
        behav_data_all['time_since_target_last_seen']

    # Add distance from monkey position at target last seen
    behav_data_all['distance_from_monkey_pos_target_last_seen'] = np.sqrt(
        (behav_data_all['monkey_x'] - behav_data_all['monkey_x_target_last_seen_frozen'])**2 +
        (behav_data_all['monkey_y'] -
         behav_data_all['monkey_y_target_last_seen_frozen'])**2
    )

    # Add cumulative distance since target last seen
    behav_data_all['cum_distance_since_target_last_seen'] = behav_data_all['cum_distance'] - \
        behav_data_all['cum_distance_target_last_seen_frozen']

    # Add heading difference since target last seen
    behav_data_all['d_heading_since_target_last_seen'] = behav_data_all['monkey_angle'] - \
        behav_data_all['monkey_angle_target_last_seen_frozen']

    # make sure d_heading_since_target_last_seen is an acute angle
    behav_data_all['d_heading_since_target_last_seen'] = behav_data_all['d_heading_since_target_last_seen'] % (
        2 * np.pi)
    behav_data_all.loc[behav_data_all['d_heading_since_target_last_seen']
                       > np.pi, 'd_heading_since_target_last_seen'] -= 2 * np.pi
    behav_data_all.loc[behav_data_all['d_heading_since_target_last_seen']
                       < -np.pi, 'd_heading_since_target_last_seen'] += 2 * np.pi

    return behav_data_all


def make_pursuit_data_all(single_vis_target_df, behav_data_all):
    point_index_list = []
    segment_list = []
    for index, row in single_vis_target_df.iterrows():
        point_index = range(row['last_vis_point_index'],
                            row['ff_caught_point_index'])
        point_index_list.extend(point_index)
        segment_list.extend([index] * len(point_index))

    point_index_df = pd.DataFrame(
        {'point_index': point_index_list, 'segment': segment_list})

    pursuit_data_all = behav_data_all[behav_data_all['point_index'].isin(
        point_index_list)].copy()

    pursuit_data_all = pursuit_data_all.merge(
        point_index_df, on='point_index', how='left')

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
