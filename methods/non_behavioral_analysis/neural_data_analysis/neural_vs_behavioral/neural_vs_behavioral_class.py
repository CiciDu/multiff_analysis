import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
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


class NeuralVsBehavioralClass(further_processing_class.FurtherProcessing):
    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.02,
                 window_width=0.25,
                 one_behav_idx_per_bin=True):

        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.bin_width = bin_width
        self.window_width = window_width
        self.one_behav_idx_per_bin = one_behav_idx_per_bin
        self.max_bin = None
        self.max_visibility_window = 10

    def get_basic_data(self):
        self.retrieve_or_make_monkey_data(already_made_ok=True)
        self.make_or_retrieve_ff_dataframe(
            already_made_ok=True, exists_ok=True)
        self.make_relevant_paths()

    def make_relevant_paths(self):
        self.y_var_lags_path = os.path.join(
            self.processed_neural_data_folder_path, 'y_var_lags')
        self.vif_df_path = os.path.join(
            self.processed_neural_data_folder_path, 'vif_df')
        self.lr_result_df_path = os.path.join(
            self.processed_neural_data_folder_path, 'lr_result_df')
        os.makedirs(self.y_var_lags_path, exist_ok=True)
        os.makedirs(self.vif_df_path, exist_ok=True)
        os.makedirs(self.lr_result_df_path, exist_ok=True)

    def streamline_preparing_neural_and_behavioral_data(self, max_y_lag_number=3):
        self.get_basic_data()
        self._prepare_to_find_patterns_and_features()
        self.make_df_related_to_patterns_and_features()
        self.prep_behavioral_data_for_neural_data_modeling(
            max_y_lag_number=max_y_lag_number)
        self.max_bin = self.final_behavioral_data['bin'].max()
        self.retrieve_neural_data()
        self._get_x_and_y_var()

    def retrieve_neural_data(self):
        self.sampling_rate = 20000 if 'Bruno' in self.raw_data_folder_path else 30000
        spike_df = neural_data_processing.make_spike_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                        sampling_rate=self.sampling_rate)
        # get convolution data
        self.window_width, self.num_bins_in_window, self.convolve_pattern = neural_data_processing.calculate_window_parameters(
            window_width=self.window_width, bin_width=self.bin_width)
        self.time_bins, self.binned_spikes_df = neural_data_processing.prepare_binned_spikes_df(
            spike_df, bin_width=self.bin_width, max_bin=self.max_bin)

    def prep_behavioral_data_for_neural_data_modeling(self, max_y_lag_number=3):
        self.binned_features, self.time_bins = prep_monkey_data.initialize_binned_features(
            self.monkey_information, self.bin_width)
        self.binned_features = self._add_ff_info(self.binned_features)
        self._add_monkey_info()
        self._add_all_target_and_target_cluster_info()
        self._add_pattern_info_based_on_points_and_trials()
        self._make_final_behavioral_data()
        self._get_index_of_bins_in_valid_intervals()
        self._get_x_and_y_var()
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=self.final_behavioral_data)

    def make_or_retrieve_y_var_lr_result_df(self, exists_ok=True):
        df_path = os.path.join(self.lr_result_df_path,
                               'y_var_lr_result_df.csv')
        if exists_ok & exists(df_path):
            self.y_var_lr_result_df = pd.read_csv(df_path)
        else:
            self.y_var_lr_result_df = neural_data_modeling.get_y_var_lr_result_df(
                self.x_var, self.y_var)
            self.y_var_lr_result_df.to_csv(df_path, index=False)
            print('Made new y_var_lr_result_df')

    def reduce_y_var(self, vif_df_exists_ok=True, vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True):
        # delete columns with high VIF
        if not hasattr(self, 'vif_df'):
            self.make_or_retrieve_y_var_vif_df(exists_ok=vif_df_exists_ok)
        self.y_var_reduced, self.columns_dropped_from_y_var, self.vif_of_y_var_reduced = drop_high_vif_vars.take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(
            self.y_var,
            initial_vif=self.y_var_vif_df,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
            vif_threshold=vif_threshold,
            verbose=verbose,
            get_final_vif=True,
        )
        # # manually dropped some more columns
        # self.y_var_reduced.drop(columns=['bin'], inplace=True, errors='ignore')

    def reduce_y_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85,
                          vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                          filter_corr_by_feature=True,
                          filter_corr_by_subsets=True,
                          filter_corr_by_all_columns=True,
                          filter_vif_by_feature=True,
                          filter_vif_by_subsets=True,
                          filter_vif_by_all_columns=True,
                          ):

        # Call the function to iteratively drop lags with high correlation for each feature
        self.y_var_lags_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(self.y_var, self.y_var_lags,
                                                                                       corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                                       verbose=verbose,
                                                                                       filter_by_feature=filter_corr_by_feature,
                                                                                       filter_by_subsets=filter_corr_by_subsets,
                                                                                       filter_by_all_columns=filter_corr_by_all_columns,
                                                                                       get_column_subsets_func=self.get_subset_key_words_and_all_column_subsets)

        self.y_var_lags_reduced = drop_high_vif_vars.drop_columns_with_high_vif(self.y_var, self.y_var_lags_reduced_corr,
                                                                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                                                                vif_threshold=vif_threshold,
                                                                                verbose=verbose,
                                                                                filter_by_feature=filter_vif_by_feature,
                                                                                filter_by_subsets=filter_vif_by_subsets,
                                                                                filter_by_all_columns=filter_vif_by_all_columns,
                                                                                get_column_subsets_func=self.get_subset_key_words_and_all_column_subsets)

    def reduce_x_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85, vif_threshold=5, verbose=True):

        # Call the function to iteratively drop lags with high correlation for each feature
        self.x_var_lags_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(self.x_var, self.x_var_lags,
                                                                                       corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                                       verbose=verbose,
                                                                                       filter_by_feature=True,
                                                                                       filter_by_subsets=False,
                                                                                       filter_by_all_columns=False)

        self.x_var_lags_reduced = drop_high_vif_vars.drop_columns_with_high_vif(self.x_var, self.x_var_lags_reduced_corr,
                                                                                vif_threshold=vif_threshold,
                                                                                verbose=verbose,
                                                                                filter_by_feature=False,
                                                                                filter_by_subsets=False,
                                                                                filter_by_all_columns=False)

    def make_or_retrieve_y_var_lags_reduced(self, exists_ok=True):
        df_path = os.path.join(self.y_var_lags_path,
                               f'y_var_lags_{self.max_y_lag_number}_reduced.csv')
        if exists(df_path) & exists_ok:
            self.y_var_lags_reduced = pd.read_csv(df_path)
        else:
            if not hasattr(self, 'y_var_lags'):
                self._get_y_var_lags(
                    max_y_lag_number=self.max_y_lag_number, continuous_data=self.final_behavioral_data)
            self.reduce_y_var_lags()
            self.y_var_lags_reduced.to_csv(df_path, index=False)

    def make_or_retrieve_x_var_vif_df(self, exists_ok=True):
        self.x_var_vif_df = drop_high_vif_vars.make_or_retrieve_vif_df(self.x_var, self.vif_df_path,
                                                                       vif_df_name='x_var_vif_df', exists_ok=exists_ok
                                                                       )

    def make_or_retrieve_y_var_vif_df(self, exists_ok=True):
        self.y_var_vif_df = drop_high_vif_vars.make_or_retrieve_vif_df(self.y_var, self.vif_df_path,
                                                                       vif_df_name='y_var_vif_df', exists_ok=exists_ok
                                                                       )

    def make_or_retrieve_y_var_reduced_vif_df(self, exists_ok=True):
        self.y_var_reduced_vif_df = drop_high_vif_vars.make_or_retrieve_vif_df(self.y_var_reduced, self.vif_df_path,
                                                                               vif_df_name='y_var_reduced_vif_df', exists_ok=exists_ok
                                                                               )

    def make_or_retrieve_y_var_lags_vif_df(self, exists_ok=True):
        self.y_var_lags_vif_df = drop_high_vif_vars.make_or_retrieve_vif_df(self.y_var_lags, self.vif_df_path,
                                                                            vif_df_name=f'y_var_lags_{self.max_y_lag_number}_vif_df', exists_ok=exists_ok
                                                                            )

    def make_or_retrieve_y_var_lags_reduced_vif_df(self, exists_ok=True):
        self.y_var_lags_reduced_vif_df = drop_high_vif_vars.make_or_retrieve_vif_df(self.y_var_lags_reduced, self.vif_df_path,
                                                                                    vif_df_name=f'y_var_lags_{self.max_y_lag_number}_reduced_vif_df', exists_ok=exists_ok
                                                                                    )

    def _add_monkey_info(self):
        self.monkey_info_in_bins = prep_monkey_data.bin_monkey_information(
            self.monkey_information, self.time_bins, one_behav_idx_per_bin=self.one_behav_idx_per_bin)
        self.monkey_info_in_bins_ess = prep_monkey_data.make_monkey_info_in_bins_essential(
            self.monkey_info_in_bins, self.time_bins, self.ff_caught_T_new, self.convolve_pattern, self.window_width)
        self.binned_features = self.binned_features.merge(
            self.monkey_info_in_bins_ess, how='left', on='bin')

    def _add_ff_info(self, binned_features):
        ff_info = prep_monkey_data.get_ff_info_for_bins(
            binned_features[['bin']], self.ff_dataframe, self.ff_caught_T_new, self.time_bins)
        # delete columns in ff_info that are duplicated in behav_data except for bin
        columns_to_drop = [
            col for col in ff_info.columns if col in binned_features.columns and col != 'bin']
        ff_info = ff_info.drop(columns=columns_to_drop)
        binned_features = binned_features.merge(
            ff_info, on='bin', how='left')
        return binned_features

    def _add_all_target_and_target_cluster_info(self):
        self._make_or_retrieve_target_df()
        self._make_or_retrieve_target_cluster_df()
        self._make_cmb_target_df()

        if self.one_behav_idx_per_bin:
            self.target_df_to_use = self.cmb_target_df[self.cmb_target_df['point_index'].isin(
                self.monkey_info_in_bins['point_index'].values)]
        else:
            self.target_df_to_use = self.cmb_target_df

        self.target_average_info, self.target_min_info, self.target_max_info = prep_target_data.get_max_min_and_avg_info_from_target_df(
            self.target_df_to_use)
        for df in [self.target_average_info, self.target_min_info, self.target_max_info]:
            self.binned_features = self.binned_features.merge(
                df, how='left', on='bin')

    def _make_cmb_target_df(self):
        # merge target df and target cluster df based on point_index; make sure no other columns are duplicated
        columns_to_drop = [
            col for col in self.target_cluster_df.columns if col in self.target_df.columns]
        columns_to_drop.remove('point_index')
        target_cluster_df = self.target_cluster_df.drop(
            columns=columns_to_drop)
        self.cmb_target_df = pd.merge(
            self.target_df, target_cluster_df, on='point_index', how='left')

        # add bin column to the target_df
        self.cmb_target_df = self.cmb_target_df.merge(self.monkey_information[[
            'point_index', 'bin']].copy(), on='point_index', how='left')

    def _make_or_retrieve_target_df(self, exists_ok=True, fill_na=False):
        target_df_filepath = os.path.join(
            self.patterns_and_features_data_folder_path, 'target_df.csv')
        if exists(target_df_filepath) & exists_ok:
            self.target_df = pd.read_csv(target_df_filepath)
            print("Retrieved target_df")
        else:
            self.target_df = prep_target_data.make_target_df(
                self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, 
                self.ff_dataframe, max_visibility_window=self.max_visibility_window)
            self.target_df.to_csv(target_df_filepath, index=False)
            print("Made new target_df")

        if fill_na:
            self.target_df = prep_target_data.fill_na_in_target_df(
                self.target_df)

        self.target_df = prep_target_data.add_columns_to_target_df(
            self.target_df)

    def _make_or_retrieve_target_cluster_df(self, exists_ok=True):
        target_cluster_df_filepath = os.path.join(
            self.patterns_and_features_data_folder_path, 'target_cluster_df.csv')
        if exists(target_cluster_df_filepath) & exists_ok:
            self.target_cluster_df = pd.read_csv(target_cluster_df_filepath)
            print("Retrieved target_cluster_df")
        else:
            self.target_cluster_df = prep_target_data.make_target_cluster_df(
                self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, self.ff_dataframe,
                self.ff_life_sorted, max_visibility_window=self.max_visibility_window)
            self.target_cluster_df.to_csv(
                target_cluster_df_filepath, index=False)
            print("Made new target_cluster_df")

    def _add_pattern_info_based_on_points_and_trials(self):
        self.binned_features = prep_monkey_data.add_pattern_info_base_on_points(self.binned_features, self.monkey_info_in_bins, self.monkey_information,
                                                                                self.try_a_few_times_indices_for_anim, self.GUAT_point_indices_for_anim,
                                                                                self.ignore_sudden_flash_indices_for_anim)
        self.binned_features = prep_monkey_data.add_pattern_info_based_on_trials(
            self.binned_features, self.ff_caught_T_new, self.all_trial_patterns, self.time_bins)

    def _make_final_behavioral_data(self):
        self.final_behavioral_data = prep_monkey_data._make_final_behavioral_data(
            self.monkey_info_in_bins_ess, self.binned_features)
        # take out column that has angle_to_boundary
        columns_to_drop = [col for col in self.final_behavioral_data.columns if (
            'angle_to_boundary' in col) or ('angle_boundary' in col)]
        self.final_behavioral_data = self.final_behavioral_data.drop(
            columns=columns_to_drop)
        # also drop the following columns since they have high correlation with other columns
        columns_to_drop = ['avg_target_cluster_last_seen_distance', 'avg_target_cluster_last_seen_time', 'avg_target_cluster_last_seen_angle',
                           'avg_target_last_seen_distance', 'avg_target_last_seen_angle']
        self.final_behavioral_data = self.final_behavioral_data.drop(
            columns=columns_to_drop)

    def _get_index_of_bins_in_valid_intervals(self, gap_too_large_threshold=100, min_combined_valid_interval_length=50):
        """
        Calculate the midpoints of the time bins and get the indices of bins that fall within valid intervals.
        """

        self.valid_intervals_df = specific_utils.take_out_valid_intervals_based_on_ff_caught_time(
            self.ff_caught_T_new, gap_too_large_threshold=gap_too_large_threshold,
            min_combined_valid_interval_length=min_combined_valid_interval_length
        )

        # Calculate the midpoints of the time bins
        mid_bin_time = (self.time_bins[1:] + self.time_bins[:-1]) / 2

        # Get the indices of bins that fall within valid intervals
        self.valid_bin_mid_time, self.valid_bin_index = general_utils.take_out_data_points_in_valid_intervals(
            mid_bin_time, self.valid_intervals_df
        )

        # # print the number of bins out of total numbers that are in valid intervals
        # print(f"Number of bins in valid intervals based on ff caught time: {len(self.valid_bin_index)} out of {len(mid_bin_time)}"
        #       f" ({len(self.valid_bin_index)/len(mid_bin_time)*100:.2f}%)")

    def _get_x_and_y_var(self):
        self.x_var = self.binned_spikes_df.set_index(
            'bin').loc[self.valid_bin_index].reset_index(drop=True)
        self.y_var = self.final_behavioral_data.set_index(
            'bin').loc[self.valid_bin_index].reset_index(drop=False)

    def _get_y_var_lags(self, max_y_lag_number, continuous_data):
        self.max_y_lag_number = max_y_lag_number
        self.y_var_lags, self.lag_numbers = self._get_lags(
            max_y_lag_number, continuous_data)
        if 'bin_0' in self.y_var_lags.columns:
            self.y_var_lags['bin'] = self.y_var_lags['bin_0'].astype(int)
            self.y_var_lags = self.y_var_lags.drop(
                columns=[col for col in self.y_var_lags.columns if 'bin_' in col])

    def _get_x_var_lags(self, max_x_lag_number, continuous_data):
        self.max_x_lag_number = max_x_lag_number
        self.x_var_lags, self.x_lag_numbers = self._get_lags(
            max_x_lag_number, continuous_data)
        # drop all columns in x_var_lags that has bin_
        if 'bin_0' in self.x_var_lags.columns:
            self.x_var_lags['bin'] = self.x_var_lags['bin_0'].astype(int)
            self.x_var_lags = self.x_var_lags.drop(
                columns=[col for col in self.x_var_lags.columns if 'bin_' in col])

    def _get_lags(self, max_lag_number, continuous_data):
        lag_numbers = np.arange(-max_lag_number, max_lag_number+1)
        var_lags = neural_data_processing.add_lags_to_each_feature(
            continuous_data, lag_numbers)
        if hasattr(self, 'valid_bin_index'):
            var_lags = var_lags.set_index(
                'bin_0').loc[self.valid_bin_index].reset_index(drop=False)
        return var_lags, lag_numbers

    @staticmethod
    def get_subset_key_words_and_all_column_subsets(y_var_lags):
        subset_key_words = ['stop', 'speed_or_ddv', 'dw', 'LD_or_RD_or_gaze',
                            'distance', 'angle', 'frozen', 'dummy', 'num_or_any_or_rate']

        all_column_subsets = [
            [col for col in y_var_lags.columns if 'stop' in col],
            [col for col in y_var_lags.columns if (
                'speed' in col) or ('ddv' in col)],
            [col for col in y_var_lags.columns if ('dw' in col)],
            [col for col in y_var_lags.columns if (
                'LD' in col) or ('RD' in col) or ('gaze' in col)],
            [col for col in y_var_lags.columns if ('distance' in col)],
            [col for col in y_var_lags.columns if ('angle' in col)],
            [col for col in y_var_lags.columns if ('frozen' in col)],
            [col for col in y_var_lags.columns if ('dummy' in col)],
            [col for col in y_var_lags.columns if (
                'num' in col) or ('any' in col) or ('rate' in col)],
        ]
        return subset_key_words, all_column_subsets
