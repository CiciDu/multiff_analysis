import sys
import os
import warnings
import math
import re
import logging

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import colorcet
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Scientific computing imports
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA, gpfa_core, gpfa_util

# Local imports
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, ml_decoder_class, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_by_topic.decode_targets import prep_target_decoder, behav_features_to_keep
from null_behaviors import curvature_utils, curv_of_traj_utils
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils, gpfa_helper_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class


class TargetDecoderClass(base_neural_class.NeuralBaseClass, gpfa_helper_class.GPFAHelperClass):

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

    def get_x_and_y_data_for_modeling(self, exists_ok=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self._reduce_x_var_lags()
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
            self.behav_data_all = self._add_ff_info(self.behav_data_all)

            self._add_or_drop_columns()
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
        pursuit_data_all = prep_target_decoder.make_pursuit_data_all(
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

    def reduce_y_var(self,
                     save_data=True,
                     corr_threshold_for_lags_of_a_feature=0.98,
                     vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                     filter_corr_by_all_columns=False,
                     filter_vif_by_subsets=True,
                     filter_vif_by_all_columns=True,
                     exists_ok=True,
                     ):
        df_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_reduced.csv')

        self._reduce_y_var(df_path=df_path,
                           save_data=save_data,
                           corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                           vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                           vif_threshold=vif_threshold,
                           verbose=verbose,
                           filter_corr_by_all_columns=filter_corr_by_all_columns,
                           filter_vif_by_subsets=filter_vif_by_subsets,
                           filter_vif_by_all_columns=filter_vif_by_all_columns,
                           exists_ok=exists_ok)

    def reduce_y_var_lags(self,
                          df_path=None,
                          save_data=True,
                          corr_threshold_for_lags_of_a_feature=0.85,
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
        self._reduce_y_var_lags(df_path=df_path,
                                save_data=save_data,
                                corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                vif_threshold=vif_threshold,
                                verbose=verbose,
                                filter_corr_by_feature=filter_corr_by_feature,
                                filter_corr_by_subsets=filter_corr_by_subsets,
                                filter_corr_by_all_columns=filter_corr_by_all_columns,
                                filter_vif_by_feature=filter_vif_by_feature,
                                filter_vif_by_subsets=filter_vif_by_subsets,
                                filter_vif_by_all_columns=filter_vif_by_all_columns,
                                exists_ok=exists_ok)

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
        target_df_lags = prep_target_decoder.initialize_target_df_lags(
            self.y_var, self.max_y_lag_number, self.bin_width)
        target_df_lags = prep_target_decoder.add_target_info_based_on_target_index_and_point_index(target_df_lags, self.monkey_information, self.ff_real_position_sorted,
                                                                                                   self.ff_dataframe, self.ff_caught_T_new, self.curv_of_traj_df)
        target_df_lags = prep_target_decoder.fill_na_in_last_seen_columns(
            target_df_lags)

        # Now, put the lagged target columns into y_var_lags
        self.y_var_lags = prep_target_decoder.add_lagged_target_columns(
            self.y_var_lags, self.y_var, target_df_lags, self.max_y_lag_number, target_columns=self.target_columns)

        if not basic_data_present:
            # free up memory if basic data is not present before calling the function
            self._free_up_memory()

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

        self._reduce_x_var()

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
        na_rows, na_cols = prep_target_decoder._process_na(
            self.behav_data_all)

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
        self.behav_data_all = prep_target_decoder._add_curv_info_to_behav_data_all(
            self.behav_data_all, self.curv_of_traj_df, self.monkey_information, self.ff_caught_T_new)

    def _add_all_target_info(self):

        self.behav_data_all = prep_target_decoder.add_target_info_to_behav_data_all(
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
            self.single_vis_target_df = prep_target_decoder.find_single_vis_target_df(
                self.target_clust_last_vis_df, self.monkey_information, self.ff_caught_T_new, max_visibility_window=self.max_visibility_window)
            self.single_vis_target_df.to_csv(df_path, index=False)
            print(f'Saved single_vis_target_df to {df_path}')

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_corr(y_var_lags):
        subset_key_words, all_column_subsets = prep_target_decoder._get_subset_key_words_and_all_column_subsets_for_corr(
            y_var_lags)
        return subset_key_words, all_column_subsets

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_vif(y_var_lags):
        subset_key_words, all_column_subsets = prep_target_decoder._get_subset_key_words_and_all_column_subsets_for_vif(
            y_var_lags)
        return subset_key_words, all_column_subsets
