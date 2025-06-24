import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import ml_decoder_class
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from non_behavioral_analysis.neural_data_analysis.decode_targets import decode_target_utils, behav_features_to_keep
from null_behaviors import curvature_utils, curv_of_traj_utils
from non_behavioral_analysis.neural_data_analysis.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils
from non_behavioral_analysis.neural_data_analysis.decode_targets import behav_features_to_keep, decode_target_utils
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

        self.bin_width_w_unit = self.bin_width * pq.s
        self.max_visibility_window = 10

        self.decoding_targets_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'decoding_targets')
        os.makedirs(self.decoding_targets_folder_path, exist_ok=True)

        # Initialize ML decoder
        self.ml_decoder = ml_decoder_class.MLBehavioralDecoder()

    def streamline_making_behav_and_neural_data(self, exists_ok=True):
        self.get_all_behav_data(exists_ok=exists_ok)
        self.max_bin = self.behav_data.bin.max()
        self.retrieve_neural_data()

    def get_all_behav_data(self, exists_ok=True):
        self.get_basic_data()
        self.get_behav_data(exists_ok=exists_ok)
        self.get_pursuit_data()

    def get_basic_data(self):
        super().get_basic_data()
        self._get_curv_of_traj_df()
        self._make_or_retrieve_target_df(
            exists_ok=True,
            fill_na=False)

    def directly_retrieve_data_for_modeling(self, exists_ok=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self.reduce_y_var_lags(exists_ok=exists_ok)

    def get_behav_data(self, exists_ok=True, save_data=True):
        behav_data_all_path = os.path.join(
            self.decoding_targets_folder_path, 'behav_data_all.csv')

        if exists_ok & os.path.exists(behav_data_all_path):
            self.behav_data_all = pd.read_csv(behav_data_all_path)
            print(f'Loaded behav_data_all from {behav_data_all_path}')
        else:
            basic_data_present = hasattr(self, 'monkey_information')
            # check if basic data is present
            if not basic_data_present:
                self.get_basic_data()
            _, self.time_bins = prep_monkey_data.initialize_binned_features(
                self.monkey_information, self.bin_width)
            self.behav_data_all = prep_monkey_data.bin_monkey_information(
                self.monkey_information, self.time_bins, one_behav_idx_per_bin=self.one_behav_idx_per_bin)
            self._add_or_drop_columns()

            self.behav_data_all = self._add_ff_info(self.behav_data_all)
            self._add_all_target_info()
            self._add_curv_info()
            self._process_na()
            self._clip_values()

            if not basic_data_present:
                # free up memory if basic data is not present before calling the function
                self._free_up_memory()

            if save_data:
                self.behav_data_all.to_csv(behav_data_all_path, index=False)

        self.behav_data = self.behav_data_all[behav_features_to_keep.shared_columns_to_keep +
                                              behav_features_to_keep.extra_columns_for_concat_trials]
        self._get_single_vis_target_df()

    def get_pursuit_data(self):
        # Extract behavioral data for periods between target last visibility and capture
        pursuit_data_all = decode_target_utils.make_pursuit_data_all(
            self.single_vis_target_df, self.behav_data_all)

        # add the segment info back to single_vis_target_df
        self.single_vis_target_df['segment'] = np.arange(
            len(self.single_vis_target_df))
        self.single_vis_target_df = self.single_vis_target_df.merge(pursuit_data_all[[
                                                                    'segment', 'seg_start_time', 'seg_end_time', 'seg_duration']].drop_duplicates(), on='segment', how='left')

        # drop the segments with 0 duration from pursuit_data_all
        num_segments_with_0_duration = len(
            pursuit_data_all[pursuit_data_all['seg_duration'] == 0])
        print(f'{num_segments_with_0_duration} segments ({round(num_segments_with_0_duration/len(self.single_vis_target_df) * 100, 1)}%) out of {len(self.single_vis_target_df)} segments have 0 duration. They are dropped from pursuit data')

        # drop segments in pursuit data that has 0 duration
        pursuit_data_all = pursuit_data_all[pursuit_data_all['seg_duration'] > 0].copy(
        )

        seg_vars = ['segment', 'segment_start_dummy', 'segment_end_dummy']

        self.pursuit_data = pursuit_data_all[behav_features_to_keep.shared_columns_to_keep +
                                             behav_features_to_keep.extra_columns_for_concat_trials + seg_vars]

        self.pursuit_data_by_trial = pursuit_data_all[behav_features_to_keep.shared_columns_to_keep + seg_vars]

        # check for NA; if there is any, raise a warning
        na_rows, na_cols = general_utils.find_rows_with_na(
            self.pursuit_data, 'pursuit_data')

    def _select_behav_features(self):
        self.behav_data = self.behav_data_all[behav_features_to_keep.shared_columns_to_keep +
                                              behav_features_to_keep.extra_columns_for_concat_trials]

        # Now, as a sanity check, see if the differences between behav_data and behav_data_all are all contained in
        # behav_features_to_keep.behav_features_to_drop. If not, raise a warning
        diff_columns = set(self.behav_data_all.columns) - \
            set(behav_features_to_keep.behav_features_to_drop)
        if diff_columns:
            print(
                f'The following columns are not accounted for in behav_features_to_keep: {diff_columns}.')

    def _add_or_drop_columns(self):
        self.behav_data_all = self.behav_data_all.drop(columns=['stop_id'])

    def _free_up_memory(self):
        vars_deleted = []
        for var in ['ff_dataframe', 'monkey_information', 'target_df', 'curv_of_traj_df', 'curv_df']:
            if hasattr(self, var):
                vars_deleted.append(var)
                delattr(self, var)
        print(
            f'Deleted instance attributes {vars_deleted} to free up memory')

    def get_x_and_y_var(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):
        self._get_x_var(exists_ok=exists_ok)
        self._get_y_var(exists_ok=exists_ok)
        self.get_x_and_y_var_lags(max_x_lag_number=max_x_lag_number,
                                  max_y_lag_number=max_y_lag_number,
                                  exists_ok=exists_ok)

    def get_x_and_y_var_lags(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):

        x_var_lags_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_x_var_lags.csv')
        y_var_lags_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_lags.csv')

        if exists_ok & os.path.exists(x_var_lags_path) & os.path.exists(y_var_lags_path):
            self.x_var_lags = pd.read_csv(x_var_lags_path)
            self.y_var_lags = pd.read_csv(y_var_lags_path)
            print(
                f'Loaded x_var_lags and y_var_lags from {x_var_lags_path} and {y_var_lags_path}')
        else:
            self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                                 continuous_data=self.binned_spikes_df)

            self._get_y_var_lags_with_target_info(
                max_y_lag_number=max_y_lag_number)

            self.x_var_lags = self.x_var_lags[self.x_var_lags['bin'].isin(
                self.pursuit_data['bin'].values)]
            self.y_var_lags = self.y_var_lags[self.y_var_lags['bin'].isin(
                self.pursuit_data['bin'].values)]
            self.x_var_lags.to_csv(x_var_lags_path, index=False)
            self.y_var_lags.to_csv(y_var_lags_path, index=False)
            print(
                f'Saved x_var_lags and y_var_lags to {x_var_lags_path} and {y_var_lags_path}')

    def _get_y_var_lags_with_target_info(self, max_y_lag_number=5):
      # we'll drop columns on target for now because we'll make them separately
        self.target_columns = [
            col for col in self.y_var.columns if 'target' in col]
        # make y_columns_to_drop to be the set of self.y_columns_to_drop and self.target_columns
        continuous_data = self.behav_data.drop(
            columns=self.target_columns)
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=continuous_data)

        self.y_var_lags = self.y_var_lags[self.y_var_lags['bin'].isin(
            self.y_var['bin'].values)]

        self._add_target_info_to_y_var_lags()

    def _add_target_info_to_y_var_lags(self):

        basic_data_present = hasattr(self, 'monkey_information')
        if not basic_data_present:
            self.get_basic_data()
            self._get_curv_of_traj_df()

        # first get info for pairs of target_index and point_index that the lagged columns will use
        target_df_lags = decode_target_utils.initialize_target_df_lags(
            self.y_var, self.max_y_lag_number, self.bin_width)
        target_df_lags = decode_target_utils.add_target_info_based_on_target_index_and_point_index(target_df_lags, self.monkey_information, self.ff_real_position_sorted,
                                                                                                   self.ff_dataframe, self.ff_caught_T_new, self.curv_of_traj_df)
        target_df_lags = decode_target_utils.fill_na_in_last_seen_columns(
            target_df_lags)

        # Now, put the lagged target columns into y_var_lags
        if 'target_index' not in self.y_var_lags.columns:
            self.y_var_lags = self.y_var_lags.merge(
                self.y_var[['bin', 'target_index']], on='bin', how='left')
        self.y_var_lags = decode_target_utils.add_lagged_target_columns(
            self.y_var_lags, target_df_lags, self.max_y_lag_number, target_columns=self.target_columns)

        if not basic_data_present:
            # free up memory if basic data is not present before calling the function
            self._free_up_memory()

    def reduce_y_var(self, corr_threshold_for_lags_of_a_feature=0.98,
                     vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                     filter_corr_by_all_columns=False,
                     filter_vif_by_subsets=True,
                     filter_vif_by_all_columns=True,
                     exists_ok=True,
                     ):
        df_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_reduced.csv')
        if exists_ok and os.path.exists(df_path):
            self.y_var_reduced = pd.read_csv(df_path)
            print(f'Loaded y_var_reduced from {df_path}')
        else:
            # drop columns with std less than 0.001
            columns_w_small_std = self.y_var.std(
            )[self.y_var.std() < 0.001].index.tolist()

            self.y_var_reduced = self.y_var.drop(columns=columns_w_small_std)
            self._reduce_y_var(self.y_var_reduced,
                               filter_corr_by_all_columns=filter_corr_by_all_columns,
                               corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                               vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                               vif_threshold=vif_threshold,
                               verbose=verbose,
                               filter_vif_by_subsets=filter_vif_by_subsets,
                               filter_vif_by_all_columns=filter_vif_by_all_columns)
            self.y_var_reduced.to_csv(df_path, index=False)
            print(f'Saved y_var_reduced to {df_path}')

    def reduce_y_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85,
                          vif_threshold_for_initial_subset=5,
                          vif_threshold=5,
                          verbose=True,
                          filter_corr_by_feature=True,
                          filter_corr_by_subsets=True,
                          filter_corr_by_all_columns=True,
                          filter_vif_by_feature=True,
                          filter_vif_by_subsets=True,
                          filter_vif_by_all_columns=False,
                          exists_ok=True):
        """Reduce y_var_lags by removing highly correlated and high VIF features.

        Parameters are passed to the parent class's reduce_y_var_lags method.
        Results are cached to avoid recomputation.
        """
        df_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_lags_reduced.csv')

        # Try to load cached results if allowed
        if exists_ok and os.path.exists(df_path):
            try:
                cached_data = pd.read_csv(df_path)
                if len(cached_data) == len(self.y_var_lags):
                    self.y_var_lags_reduced = cached_data
                    if verbose:
                        print(
                            f'Loaded y_var_lags_reduced from {df_path}')
                    return
            except (pd.errors.EmptyDataError, ValueError) as e:
                if verbose:
                    print(f'Failed to load cached data: {str(e)}')

        # If we get here, we need to recompute
        if verbose:
            print('Computing reduced y_var_lags...')

        # Call parent class method to do the actual reduction
        super().reduce_y_var_lags(
            corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
            vif_threshold=vif_threshold,
            verbose=verbose,
            filter_corr_by_feature=filter_corr_by_feature,
            filter_corr_by_subsets=filter_corr_by_subsets,
            filter_corr_by_all_columns=filter_corr_by_all_columns,
            filter_vif_by_feature=filter_vif_by_feature,
            filter_vif_by_subsets=filter_vif_by_subsets,
            filter_vif_by_all_columns=filter_vif_by_all_columns
        )

        # Cache the results
        try:
            self.y_var_lags_reduced.to_csv(df_path, index=False)
            if verbose:
                print(f'Saved reduced y_var_lags to {df_path}')
        except Exception as e:
            if verbose:
                print(f'Warning: Failed to cache results: {str(e)}')

    def _get_x_var(self, exists_ok=True):
        x_var_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_x_var.csv')
        if exists_ok and os.path.exists(x_var_path):
            self.x_var = pd.read_csv(x_var_path)
            print(f'Loaded x_var from {x_var_path}')
        else:
            neural_data_present = hasattr(self, 'binned_spikes_df')
            # check if basic data is present
            if not neural_data_present:
                self.get_all_behav_data()
                self.max_bin = self.behav_data.bin.max()
                self.retrieve_neural_data()
                # self._free_up_memory()
            binned_spikes_sub = self.binned_spikes_df[self.binned_spikes_df['bin'].isin(
                self.pursuit_data['bin'].values)]
            self.x_var = binned_spikes_sub.drop(
                columns=['bin']).reset_index(drop=True)
            self.x_var.to_csv(x_var_path, index=False)
            print(f'Saved x_var to {x_var_path}')

    def _get_y_var(self, exists_ok=True):
        # note that this is for the continuous case (a.k.a. all selected time points are used together, instead of being separated into trials)

        y_var_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var.csv')
        if exists_ok and os.path.exists(y_var_path):
            self.y_var = pd.read_csv(y_var_path)
            print(f'Loaded y_var from {y_var_path}')
        else:
            self.y_var = self.pursuit_data.reset_index(drop=True)
            # Convert bool columns to int
            bool_columns = self.y_var.select_dtypes(include=['bool']).columns
            self.y_var[bool_columns] = self.y_var[bool_columns].astype(int)
            self.y_var.to_csv(y_var_path, index=False)
            print(f'Saved y_var to {y_var_path}')

        self.reduce_y_var(exists_ok=exists_ok)

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
        na_rows, na_cols = general_utils.find_rows_with_na(
            self.behav_data_all, 'behav_data_all')

    def _clip_values(self):
        # clip values in some columns
        for column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
            self.behav_data_all.loc[:, column] = np.clip(
                self.behav_data_all.loc[:, column], -1000, 1000)

    def _get_curv_of_traj_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        self.curv_of_traj_df = self.get_curv_of_traj_df(
            window_for_curv_of_traj=window_for_curv_of_traj,
            curv_of_traj_mode=curv_of_traj_mode,
            truncate_curv_of_traj_by_time_of_capture=False
        )

    def _add_curv_info(self):
        ff_df = self.behav_data_all[['point_index', 'target_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                                     'target_x', 'target_y', 'target_distance', 'target_angle', 'target_angle_to_boundary']]
        ff_df = ff_df.rename(columns={'target_x': 'ff_x', 'target_y': 'ff_y', 'target_angle': 'ff_angle',
                             'target_index': 'ff_index', 'target_distance': 'ff_distance', 'target_angle_to_boundary': 'ff_angle_boundary'})

        self.curv_df = curvature_utils.make_curvature_df(ff_df, self.curv_of_traj_df, clean=False,
                                                         remove_invalid_rows=False,
                                                         invalid_curvature_ok=True,
                                                         ignore_error=True,
                                                         monkey_information=self.monkey_information,
                                                         ff_caught_T_new=self.ff_caught_T_new)
        self.behav_data_all = self.behav_data_all.merge(self.curv_df[[
            'point_index', 'curv_of_traj', 'optimal_arc_d_heading']].drop_duplicates(), on='point_index', how='left')
        self.behav_data_all.rename(columns={
            'curv_of_traj': 'traj_curv', 'optimal_arc_d_heading': 'target_opt_arc_dheading'}, inplace=True)

    def _add_all_target_info(self):

        self.behav_data_all = decode_target_utils.add_target_info_to_behav_data_all(
            self.behav_data_all, self.target_df)

    def _get_single_vis_target_df(self, single_vis_target_df_exists_ok=True, target_clust_last_vis_df_exists_ok=True):

        df_path = os.path.join(
            self.decoding_targets_folder_path, 'single_vis_target_df.csv')
        if single_vis_target_df_exists_ok and os.path.exists(df_path):
            try:
                self.single_vis_target_df = pd.read_csv(df_path)
                print(f'Loaded single_vis_target_df from {df_path}')
            except (pd.errors.EmptyDataError, ValueError) as e:
                print(f'Failed to load single_vis_target_df: {str(e)}')
        else:
            self.make_or_retrieve_target_clust_last_vis_df(
                exists_ok=target_clust_last_vis_df_exists_ok)
            # in the function, we'll drop the rows where target is in a cluster, because we want to preserve cases where monkey is going toward a single target, not a cluster
            self.single_vis_target_df = decode_target_utils.find_single_vis_target_df(
                self.target_clust_last_vis_df, self.monkey_information, self.ff_caught_T_new, max_visibility_window=self.max_visibility_window)
            self.single_vis_target_df.to_csv(df_path, index=False)
            print(f'Saved single_vis_target_df to {df_path}')

    # 'visible', 'rel_y', 'valid_view', 'time_since', 'gaze_world_y', 'Dz'

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_corr(y_var_lags):
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

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_vif(y_var_lags):
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

    def prepare_spikes_for_gpfa(self, align_at_beginning=False):

        self.align_at_beginning = align_at_beginning

        spike_df = neural_data_processing.make_spike_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                        sampling_rate=self.sampling_rate)

        self.spike_segs_df = fit_gpfa_utils.make_spike_segs_df(
            spike_df, self.single_vis_target_df)

        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + self.bin_width
        self.spiketrains, self.spiketrain_corr_segs = fit_gpfa_utils.turn_spike_segs_df_into_spiketrains(
            self.spike_segs_df, common_t_stop=self.common_t_stop, align_at_beginning=self.align_at_beginning)

    def get_gpfa_traj(self, latent_dimensionality=10, exists_ok=True):
        """
        Compute or load GPFA trajectories.

        Parameters:
        -----------
        latent_dimensionality : int
            Number of latent dimensions for GPFA
        exists_ok : bool
            Whether to load existing trajectories if available
        """
        import pickle

        alignment = 'segStart' if self.align_at_beginning else 'segEnd'
        file_name = f'gpfa_neural_aligned_{alignment}_d{latent_dimensionality}.pkl'

        # Create filename with latent dimensionality to avoid conflicts
        trajectories_path = os.path.join(
            self.decoding_targets_folder_path, file_name)

        if exists_ok and os.path.exists(trajectories_path):
            try:
                with open(trajectories_path, 'rb') as f:
                    self.trajectories = pickle.load(f)
                print(f'Loaded GPFA trajectories from {trajectories_path}')
                return
            except Exception as e:
                print(f'Failed to load trajectories: {str(e)}. Recomputing...')

        # Compute trajectories if not loaded
        print(
            f'Computing GPFA trajectories with {latent_dimensionality} dimensions...')
        gpfa_3dim = GPFA(bin_size=self.bin_width_w_unit,
                         x_dim=latent_dimensionality)
        self.trajectories = gpfa_3dim.fit_transform(self.spiketrains)

        # Save trajectories
        try:
            with open(trajectories_path, 'wb') as f:
                pickle.dump(self.trajectories, f)
            print(f'Saved GPFA trajectories to {trajectories_path}')
        except Exception as e:
            print(f'Warning: Failed to save trajectories: {str(e)}')


    def get_gpfa_and_behav_data_for_all_trials(self, use_lags=False):

        self.behav_trials = []
        self.gpfa_trials = []

        if use_lags:
            y_var = self.y_var_lags_reduced
            if 'segment_0' in y_var.columns:
                y_var.drop(columns=['segment_0'], inplace=True)
        else:
            y_var = self.y_var_reduced
        
        y_var['segment'] = self.y_var['segment'].values

        segments_behav = y_var['segment'].unique()
        segments_behav = segments_behav[segments_behav != '']
        segments_neural = self.spiketrain_corr_segs
        shared_segments = [seg for seg in segments_behav.tolist()
                           if seg in segments_neural.tolist()]

        for seg in shared_segments:
            y_var_sub = y_var[y_var['segment'] == seg]
            behav_data_of_trial = y_var_sub.drop(columns=['segment']).values
            self.behav_trials.append(behav_data_of_trial)

            trial_length = behav_data_of_trial.shape[0]
            gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
            self.gpfa_trials.append(gpfa_trial)
            
            
    # def get_gpfa_and_behav_data_for_all_trials(self):

    #     self.behav_trials = []
    #     self.gpfa_trials = []

    #     segments_behav = self.pursuit_data_by_trial['segment'].unique()
    #     segments_neural = self.spiketrain_corr_segs
    #     shared_segments = [seg for seg in segments_behav.tolist()
    #                        if seg in segments_neural.tolist()]

    #     for seg in shared_segments:
    #         pursuit_sub = self.pursuit_data_by_trial[self.pursuit_data_by_trial['segment'] == seg]
    #         behav_data_of_trial = pursuit_sub.drop(columns=['segment']).values
    #         self.behav_trials.append(behav_data_of_trial)

    #         trial_length = behav_data_of_trial.shape[0]
    #         gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
    #             self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
    #         self.gpfa_trials.append(gpfa_trial)

    def decode_one_var_with_ml(self, target_variable='target_distance', test_size=0.2,
                               models_to_use=['rf', 'nn', 'lr'], cv_folds=5):
        """
        Decode target representation using machine learning approaches.

        Parameters:
        -----------
        target_variable : str or list
            Target variable(s) to predict
        test_size : float
            Proportion of data to use for testing
        models_to_use : list
            List of models to use: 'rf', 'svm', 'nn', 'lr'
        cv_folds : int
            Number of cross-validation folds

        Returns:
        --------
        dict : ML results including model performance and predictions
        """
        # Use the ML decoder to perform the decoding
        ml_results = self.ml_decoder.decode_targets(
            neural_data=self.neural_data,
            target_data=self.target_data,
            target_variable=target_variable,
            test_size=test_size,
            models_to_use=models_to_use,
            cv_folds=cv_folds
        )

        # Store results in main class for compatibility
        if ml_results is not None:
            self.models[f'ml_{target_variable}'] = ml_results
            self.results[f'ml_{target_variable}'] = ml_results

        return ml_results
