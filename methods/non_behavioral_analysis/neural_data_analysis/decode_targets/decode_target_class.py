import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from non_behavioral_analysis.neural_data_analysis.decode_targets import decode_target_utils, behav_features_to_keep
from null_behaviors import curvature_utils, curv_of_traj_utils
from non_behavioral_analysis.neural_data_analysis.decode_targets import behav_features_to_keep, plot_gpfa_utils, decode_target_utils, fit_gpfa_utils, gpfa_regression_utils
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
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elephant.gpfa import gpfa_core, gpfa_util


class DecodeTargetClass(neural_vs_behavioral_class.NeuralVsBehavioralClass):

    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.02,
                 window_width=0.25,
                 one_behav_idx_per_bin=True):

        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         window_width=window_width,
                         one_behav_idx_per_bin=one_behav_idx_per_bin
                         )
        self.get_basic_data()
        self.bin_width_w_unit = self.bin_width * pq.s

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
        self._find_single_vis_target_df()
        self._take_out_pursuit_data()

    def get_x_and_y_var(self, max_x_lag_number=5, max_y_lag_number=5):
        self._get_x_var()
        self._get_y_var()
        self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                             continuous_data=self.binned_spikes_df)
        self._get_y_var_lags(max_lag_number=max_y_lag_number, continuous_data=self.behav_data.drop(
            columns=['point_index'] + self.y_columns_to_drop))

    def reduce_y_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85,
                          vif_threshold_for_initial_subset=5,
                          vif_threshold=5,
                          verbose=True,
                          filter_corr_by_feature=True,
                          filter_corr_by_subsets=False,
                          filter_corr_by_all_columns=True,
                          filter_vif_by_feature=True,
                          filter_vif_by_subsets=False,
                          filter_vif_by_all_columns=False
                          ):

        super().reduce_y_var_lags(corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                                  vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                  vif_threshold=vif_threshold,
                                  verbose=verbose,
                                  filter_corr_by_feature=filter_corr_by_feature,
                                  filter_corr_by_subsets=filter_corr_by_subsets,
                                  filter_corr_by_all_columns=filter_corr_by_all_columns,
                                  filter_vif_by_feature=filter_vif_by_feature,
                                  filter_vif_by_subsets=filter_vif_by_subsets,
                                  filter_vif_by_all_columns=filter_vif_by_all_columns)

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
        # note that this is for the continuous case (a.k.a. all selected time points are used together, instead of being separated into trials)

        self.y_var = self.pursuit_data.drop(
            columns="point_index").reset_index(drop=True)
        # Convert bool columns to int
        bool_columns = self.y_var.select_dtypes(include=['bool']).columns
        self.y_var[bool_columns] = self.y_var[bool_columns].astype(int)

        # To prevent multicollinearity, these columns need to be dropped from behavioral data
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

        self.target_df['target_rel_y'] = self.target_df['target_distance'] * \
            np.cos(self.target_df['target_angle'])
        self.target_df['target_rel_x'] = - self.target_df['target_distance'] * \
            np.sin(self.target_df['target_angle'])
        self.behav_data_all = decode_target_utils.add_target_info_to_behav_data_all(
            self.behav_data_all, self.target_df)

    def _find_single_vis_target_df(self, target_clust_last_vis_df_exists_ok=True):
        self.make_or_retrieve_target_clust_last_vis_df(
            exists_ok=target_clust_last_vis_df_exists_ok)

        # in the function, we'll drop the rows where target is in a cluster, because we want to preserve cases where monkey is going toward a single target, not a cluster
        self.single_vis_target_df = decode_target_utils.find_single_vis_target_df(
            self.target_clust_last_vis_df, self.monkey_information, self.ff_caught_T_new)

    def _take_out_pursuit_data(self):

        self.pursuit_data_all = decode_target_utils.make_pursuit_data_all(
            self.single_vis_target_df, self.behav_data_all)

        # add the segment info back to single_vis_target_df
        self.single_vis_target_df['segment'] = np.arange(
            len(self.single_vis_target_df))
        self.single_vis_target_df = self.single_vis_target_df.merge(self.pursuit_data_all[[
                                                                    'segment', 'seg_start_time', 'seg_end_time', 'seg_duration']].drop_duplicates(), on='segment', how='left')

        # drop the segments with 0 duration from pursuit_data_all
        num_segments_with_0_duration = len(
            self.pursuit_data_all[self.pursuit_data_all['seg_duration'] == 0])
        print(f'{num_segments_with_0_duration} segments ({round(num_segments_with_0_duration/len(self.single_vis_target_df) * 100, 1)}%) out of {len(self.single_vis_target_df)} segments have 0 duration. They are dropped from pursuit data')

        # drop segments in pursuit data that has 0 duration
        self.pursuit_data_all = self.pursuit_data_all[self.pursuit_data_all['seg_duration'] > 0].copy(
        )

        self.pursuit_data = self.pursuit_data_all[behav_features_to_keep.shared_columns_to_keep +
                                                  behav_features_to_keep.extra_columns_for_concat_trials]

        self.pursuit_data_by_trial = self.pursuit_data_all[behav_features_to_keep.shared_columns_to_keep + [
            'segment']]

    @staticmethod
    def get_subset_key_words_and_all_column_subsets(y_var_lags):
        subset_key_words = ['_x', '_y', 'angle', 'ff_or_target']
        all_column_subsets = [
            [col for col in y_var_lags.columns if '_x' in col],
            [col for col in y_var_lags.columns if '_y' in col],
            [col for col in y_var_lags.columns if 'angle' in col],
            [col for col in y_var_lags.columns if (
                'ff' in col) or ('target' in col)]
        ]
        return subset_key_words, all_column_subsets

    def prepare_spikes_for_gpfa(self, align_at_beginning=False):

        self.align_at_beginning = align_at_beginning
        self.spike_segs_df = fit_gpfa_utils.make_spike_segs_df(
            self.spike_df, self.single_vis_target_df)

        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + self.bin_width
        self.spiketrains, self.spiketrain_corr_segs = fit_gpfa_utils.turn_spike_segs_df_into_spiketrains(
            self.spike_segs_df, common_t_stop=self.common_t_stop, align_at_beginning=self.align_at_beginning)

    def get_gpfa_trahjectories(self, latent_dimensionality=10):

        gpfa_3dim = GPFA(bin_width_w_unit=self.bin_width_w_unit,
                         x_dim=latent_dimensionality)
        self.trajectories = gpfa_3dim.fit_transform(self.spiketrains)

    def get_latent_data_and_behav_data_for_all_trials(self):

        self.behavior_trials = []
        self.latent_trials = []

        segments_behav = self.pursuit_data_by_trial['segment'].unique()
        segments_neural = self.spiketrain_corr_segs
        shared_segments = [seg for seg in segments_behav.tolist()
                           if seg in segments_neural.tolist()]

        for seg in shared_segments:
            pursuit_sub = self.pursuit_data_by_trial[self.pursuit_data_by_trial['segment'] == seg]
            behav_data_of_trial = pursuit_sub.drop(columns=['segment']).values
            self.behavior_trials.append(behav_data_of_trial)

            trial_length = behav_data_of_trial.shape[0]
            latent_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
            self.latent_trials.append(latent_trial)
