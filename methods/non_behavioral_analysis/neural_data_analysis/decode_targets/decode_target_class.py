import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, reduce_multicollinearity
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from non_behavioral_analysis.neural_data_analysis.decode_targets import decode_target_utils, behav_features_to_keep
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
import re


class DecodeTargetClass(neural_vs_behavioral_class.NeuralVsBehavioralClass):
    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.1,
                 window_width=0.25,
                 one_behav_idx_per_bin=True):

        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         window_width=window_width,
                         one_behav_idx_per_bin=one_behav_idx_per_bin
                         )
        self.get_basic_data()

    def streamline_making_behav_and_neural_data(self):
        self.get_behav_data()
        self.get_pursuit_data()
        self.max_bin = self.behav_data.bin.max()
        self.retrieve_neural_data()

    def get_behav_data(self):

        _, self.time_bins = prep_monkey_data.initialize_binned_features(
            self.monkey_information, self.bin_width)
        self.behav_data_all = prep_monkey_data.bin_monkey_information(
            self.monkey_information, self.time_bins, one_behav_idx_per_bin=self.one_behav_idx_per_bin)
        # drop 'stop_id' column
        self.behav_data_all = self.behav_data_all.drop(columns=['stop_id'])
        self.behav_data_all = self._add_ff_info(self.behav_data_all)
        self._add_all_target_info()
        self._add_curv_info()
        self._process_na()
        self._clip_values()
        self.behav_data = self.behav_data_all[behav_features_to_keep.shared_columns_to_keep +
                                              behav_features_to_keep.extra_columns_for_concat_trials]

    def get_pursuit_data(self):
        # Extract behavioral data for periods between target last visibility and capture
        self._find_single_vis_ff_targets()
        self._take_out_pursuit_data()

    def get_x_and_y_var(self, max_x_lag_number=5, max_y_lag_number=5):
        self._get_x_var()
        self._get_y_var()
        self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                             continuous_data=self.binned_spikes_df)
        self._get_y_var_lags(max_lag_number=max_y_lag_number, continuous_data=self.behav_data.drop(
            columns=['point_index'] + self.y_columns_to_drop))

    def _get_x_var(self):
        _, self.binned_spikes_df = neural_data_processing.prepare_binned_spikes_matrix_and_df(
            self.all_binned_spikes, max_bin=self.max_bin)
        self.binned_spikes_df['bin'] = np.arange(
            self.binned_spikes_df.shape[0])
        binned_spikes_sub = self.binned_spikes_df[self.binned_spikes_df['bin'].isin(
            self.pursuit_data['bin'].values)]
        self.x_var = binned_spikes_sub.drop(
            columns=['bin']).reset_index(drop=True)

    def _get_y_var(self):
        self.y_var = self.pursuit_data.drop(
            columns="point_index").reset_index(drop=True)
        # Convert bool columns to int
        bool_columns = self.y_var.select_dtypes(include=['bool']).columns
        self.y_var[bool_columns] = self.y_var[bool_columns].astype(int)

        # Drop the columns that cause multicollinearity
        self.y_columns_to_drop = ['time', 'cum_distance', 'target_index']
        self.y_var_reduced = self.y_var.drop(columns=self.y_columns_to_drop)

    def _process_na(self):
        # forward fill gaze columns
        gaze_columns = [
            'gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_mky_view_angle', 'gaze_world_x', 'gaze_world_y',
            'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'gaze_mky_view_angle_l',
            'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'gaze_mky_view_angle_r',
            'gaze_world_x_l', 'gaze_world_y_l', 'gaze_world_x_r', 'gaze_world_y_r'
        ]
        # Convert inf values to NA for gaze columns
        self.behav_data_all[gaze_columns] = self.behav_data_all[gaze_columns].replace(
            [np.inf, -np.inf], np.nan)
        self.behav_data_all[gaze_columns] = self.behav_data_all[gaze_columns].ffill(
        )

        # Check for any remaining NA values
        sum_na = self.behav_data_all.isna().sum()
        if len(sum_na[sum_na > 0]) > 0:
            print('Warning: There are columns with NAs: ', sum_na[sum_na > 0])
        # drop rows with na in any column

    def _clip_values(self):
        # clip values in some columns
        for column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
            self.behav_data_all.loc[:, column] = np.clip(
                self.behav_data_all.loc[:, column], -1000, 1000)

    def _add_curv_info(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        ff_df = self.behav_data_all[['point_index', 'target_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                                     'target_x', 'target_y', 'target_distance', 'target_angle', 'target_angle_to_boundary']]
        ff_df = ff_df.rename(columns={'target_x': 'ff_x', 'target_y': 'ff_y', 'target_angle': 'ff_angle',
                             'target_index': 'ff_index', 'target_distance': 'ff_distance', 'target_angle_to_boundary': 'ff_angle_boundary'})

        self.curv_of_traj_df = self.get_curv_of_traj_df(
            window_for_curv_of_traj=window_for_curv_of_traj,
            curv_of_traj_mode=curv_of_traj_mode,
            truncate_curv_of_traj_by_time_of_capture=False
        )

        self.curv_df = curvature_utils.make_curvature_df(ff_df, self.curv_of_traj_df, clean=False,
                                                         remove_invalid_rows=False,
                                                         invalid_curvature_ok=True,
                                                         ignore_error=True,
                                                         monkey_information=self.monkey_information,
                                                         ff_caught_T_new=self.ff_caught_T_new)
        self.behav_data_all = self.behav_data_all.merge(self.curv_df[[
            'point_index', 'curv_of_traj', 'optimal_arc_d_heading']], on='point_index', how='left')
        self.behav_data_all.rename(columns={
            'curv_of_traj': 'traj_curv', 'optimal_arc_d_heading': 'target_opt_arc_dheading'}, inplace=True)

    def _add_all_target_info(self):

        self._make_or_retrieve_target_df()

        # drop columns in target_df that are duplicated in behav_data_all
        columns_to_drop = [
            col for col in self.target_df.columns if col in self.behav_data_all.columns]
        columns_to_drop.remove('point_index')
        target_df = self.target_df.drop(columns=columns_to_drop)

        self.behav_data_all = self.behav_data_all.merge(
            target_df, on='point_index', how='left')

    def _find_single_vis_ff_targets(self, target_clust_last_vis_df_exists_ok=True):
        self.make_or_retrieve_target_clust_last_vis_df(
            exists_ok=target_clust_last_vis_df_exists_ok)
        self.single_vis_ff_targets = decode_target_utils.find_single_vis_ff_targets(
            self.target_clust_last_vis_df, self.monkey_information, self.ff_caught_T_new)

    def _take_out_pursuit_data(self):
        point_index_list = []
        for index, row in self.single_vis_ff_targets.iterrows():
            point_index_list.extend(
                range(row['last_vis_point_index'], row['ff_caught_point_index']))

        self.pursuit_data = self.behav_data[self.behav_data['point_index'].isin(
            point_index_list)].copy()
        org_len = len(self.pursuit_data)
        new_len = len(self.behav_data)
        print(f'{org_len} rows of {new_len} rows ({round(org_len/new_len * 100, 1)}%) of behav_data_all are preserved after taking out chunks between target last-seen time and capture time')

    def _get_x_var_lags(self, max_x_lag_number=5, continuous_data=None):
        # Find columns that start with 'bin_' and end with a single number (positive or negative)
        bin_columns = [col for col in self.x_var.columns if re.match(
            r'^bin_-?\d+(?!_\d+)$', col)]
        # Sort by absolute value of the number after 'bin_'
        bin_columns.sort(key=lambda x: abs(int(x.split('_')[1])))
        self.bin_columns = bin_columns

        # Get lagged features for each bin column
        for feature in bin_columns:
            feature_lags = [col for col in continuous_data.columns if re.match(
                rf'^{feature}_\d+$', col)]
            if len(feature_lags) > 0:
                print(
                    f'Found {len(feature_lags)} lagged columns for {feature}')
