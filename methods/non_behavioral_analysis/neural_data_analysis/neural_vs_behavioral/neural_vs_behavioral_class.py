import sys
from data_wrangling import process_raw_data, basic_func, further_processing_class
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, reduce_multicollinearity
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
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
    def __init__(self, raw_data_folder_path=None):
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.retrieve_or_make_monkey_data(already_made_ok=True)
        self.make_or_retrieve_ff_dataframe(already_made_ok=True, exists_ok=True)
        self.make_relevant_paths()

    def make_relevant_paths(self):
        self.y_var_lags_path = os.path.join(self.processed_neural_data_folder_path, 'y_var_lags')
        self.vif_df_path = os.path.join(self.processed_neural_data_folder_path, 'vif_df')
        self.lr_result_df_path = os.path.join(self.processed_neural_data_folder_path, 'lr_result_df')
        os.makedirs(self.y_var_lags_path, exist_ok=True)
        os.makedirs(self.vif_df_path, exist_ok=True)
        os.makedirs(self.lr_result_df_path, exist_ok=True)

    def streamline_preparing_neural_and_behavioral_data(self):
        self.prepare_to_find_patterns_and_features()
        self.make_df_related_to_patterns_and_features()
        self.retrieve_neural_data()
        self.prep_behavioral_data_for_neural_data_modeling()
        self._get_x_and_y_var()

    def retrieve_neural_data(self, bin_width=0.25, window_width=1):
        self.bin_width = bin_width
        self.window_width = window_width
        self.sampling_rate = 20000 if 'Bruno' in self.raw_data_folder_path else 30000
        self.spike_df = neural_data_processing.make_spike_df(raw_data_folder_path=self.raw_data_folder_path,
                                                             sampling_rate=self.sampling_rate)
        self.time_bins, self.all_binned_spikes = neural_data_processing.bin_spikes(self.spike_df, bin_width=bin_width)
        self.num_bins = self.all_binned_spikes.shape[0]
        self.unique_clusters = np.sort(self.spike_df.cluster.unique())

        # get convolution data
        self.window_width, self.num_bins_in_window, self.convolve_pattern = neural_data_processing.calculate_window_parameters(window_width=self.window_width, bin_width=self.bin_width)
        print("Updated window width (to get convolved data): ", window_width)

    def prep_behavioral_data_for_neural_data_modeling(self, max_lag_number=3):
        self.binned_features = prep_monkey_data.make_binned_features(self.monkey_information, self.bin_width, self.ff_dataframe, self.ff_caught_T_new)
        self._add_monkey_info()   
        self._add_all_target_info()   
        self._add_pattern_info_base_on_points_and_trials()
        self._make_final_behavioral_data()
        self._match_binned_spikes_to_range_of_behavioral_data()
        self._get_x_and_y_var()
        self._get_x_and_y_var_lags(max_lag_number=max_lag_number)

    def make_or_retrieve_y_var_lr_result_df(self, exists_ok=True):
        df_path = os.path.join(self.lr_result_df_path, 'y_var_lr_result_df.csv')
        if exists_ok & exists(df_path):
            self.y_var_lr_result_df = pd.read_csv(df_path)
        else:
            self.y_var_lr_result_df = neural_data_modeling.get_y_var_lr_result_df(self.x_var, self.y_var)
            self.y_var_lr_result_df.to_csv(df_path, index=False)
            print('Made new y_var_lr_result_df')

    def reduce_y_var(self, vif_df_exists_ok=True, vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True):
        # delete columns with high VIF
        if not hasattr(self, 'vif_df'):
            self.make_or_retrieve_y_var_vif_df(exists_ok=vif_df_exists_ok)
        self.y_var_reduced, self.dropped_columns_from_y_var, self.vif_of_y_var_reduced = reduce_multicollinearity.take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(
            self.y_var,
            initial_vif=self.y_var_vif_df,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
            vif_threshold=vif_threshold,
            verbose=verbose,
            get_final_vif=True,
            )
        # # manually dropped some more columns
        # self.y_var_reduced.drop(columns=['bin'], inplace=True, errors='ignore')

    def reduce_y_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85, vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True):

        # Call the function to iteratively drop lags with high correlation for each feature
        y_var_lags_reduced0, all_r_of_dropped_features = reduce_multicollinearity.iteratively_drop_lags_with_high_corr_for_each_feature(
            self.y_var, self.y_var_lags, lag_numbers=np.array(self.lag_numbers),
            corr_threshold=corr_threshold_for_lags_of_a_feature,
        )

        y_var_lags_reduced01, dropped_columns = reduce_multicollinearity.filter_specific_subset_of_y_var_lags_by_vif(y_var_lags_reduced0, vif_threshold=vif_threshold, verbose=True)

        verbose = True
        self.y_var_lags_reduced, dropped_columns_from_y_var_lags_reduced, vif_of_y_var_lags_reduced = reduce_multicollinearity.take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(
            y_var_lags_reduced01, initial_vif=None,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset, vif_threshold=vif_threshold,
            verbose=verbose, get_final_vif=False,
        )

    def make_or_retrieve_y_var_lags_reduced(self, exists_ok=True):
        df_path = os.path.join(self.y_var_lags_path, f'y_var_lags_{self.max_lag_number}_reduced.csv')
        if exists(df_path) & exists_ok:
            self.y_var_lags_reduced = pd.read_csv(df_path)
        else:
            if not hasattr(self, 'y_var_lags'):
                self._get_x_and_y_var_lags()
            self.reduce_y_var_lags()
            self.y_var_lags_reduced.to_csv(df_path, index=False)

    def make_or_retrieve_x_var_vif_df(self, exists_ok=True):
        self.x_var_vif_df = reduce_multicollinearity.make_or_retrieve_vif_df(self.x_var, self.vif_df_path, 
            vif_df_name='x_var_vif_df', exists_ok=exists_ok
        )

    def make_or_retrieve_y_var_vif_df(self, exists_ok=True):
        self.y_var_vif_df = reduce_multicollinearity.make_or_retrieve_vif_df(self.y_var, self.vif_df_path, 
            vif_df_name='y_var_vif_df', exists_ok=exists_ok
        )

    def make_or_retrieve_y_var_reduced_vif_df(self, exists_ok=True):
        self.y_var_reduced_vif_df = reduce_multicollinearity.make_or_retrieve_vif_df(self.y_var_reduced, self.vif_df_path,
            vif_df_name='y_var_reduced_vif_df', exists_ok=exists_ok
        )

    def make_or_retrieve_y_var_lags_vif_df(self, exists_ok=True):
        self.y_var_lags_vif_df = reduce_multicollinearity.make_or_retrieve_vif_df(self.y_var_lags, self.vif_df_path, 
            vif_df_name=f'y_var_lags_{self.max_lag_number}_vif_df', exists_ok=exists_ok
        )

    def make_or_retrieve_y_var_lags_reduced_vif_df(self, exists_ok=True):
        self.y_var_lags_reduced_vif_df = reduce_multicollinearity.make_or_retrieve_vif_df(self.y_var_lags_reduced, self.vif_df_path,
            vif_df_name=f'y_var_lags_{self.max_lag_number}_reduced_vif_df', exists_ok=exists_ok
        )

    def _add_monkey_info(self):
        self.rebinned_monkey_info_essential, self.monkey_information = prep_monkey_data.make_rebinned_monkey_info_essential(
                self.monkey_information, self.time_bins, self.ff_caught_T_new, self.convolve_pattern, self.window_width)
        self.binned_features = self.binned_features.merge(self.rebinned_monkey_info_essential, how='left', on='bin')


    def _add_all_target_info(self):
        self._make_or_retrieve_target_df()
        self.target_average_info, self.target_min_info, self.target_max_info = prep_target_data.get_max_min_and_avg_info_from_target_df(self.target_df)
        for df in [self.target_average_info, self.target_min_info, self.target_max_info]:
            self.binned_features = self.binned_features.merge(df, how='left', on='bin')


    def _make_or_retrieve_target_df(self, exists_ok=True):
        filepath = os.path.join(self.patterns_and_features_data_folder_path, 'target_df.csv')
        if exists(filepath) & exists_ok:
            self.target_df = pd.read_csv(filepath)
        else:
            self.target_df = prep_target_data.make_target_df(self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, self.ff_life_sorted, self.ff_dataframe)
            self.target_df.to_csv(filepath, index=False)
            print("Made new target_df")


    def _add_pattern_info_base_on_points_and_trials(self):
        self.binned_features = prep_monkey_data.add_pattern_info_base_on_points(self.binned_features, self.monkey_information,
                                            self.try_a_few_times_indices_for_anim, self.GUAT_point_indices_for_anim,
                                            self.ignore_sudden_flash_indices_for_anim)
        self.binned_features = prep_monkey_data.add_pattern_info_based_on_trials(self.binned_features, self.ff_caught_T_new, self.all_trial_patterns, self.time_bins)


    def _make_final_behavioral_data(self):
        self.final_behavioral_data = prep_monkey_data._make_final_behavioral_data(self.rebinned_monkey_info_essential, self.binned_features)
        # take out column that has angle_to_boundary
        columns_to_drop = [col for col in self.final_behavioral_data.columns if ('angle_to_boundary' in col) or ('angle_boundary' in col)]
        self.final_behavioral_data = self.final_behavioral_data.drop(columns=columns_to_drop)
        # also drop the following columns since they have high correlation with other columns
        columns_to_drop = ['avg_target_cluster_last_seen_distance', 'avg_target_cluster_last_seen_time', 'avg_target_cluster_last_seen_angle',
                           'avg_target_last_seen_distance', 'avg_target_last_seen_angle']
        self.final_behavioral_data = self.final_behavioral_data.drop(columns=columns_to_drop)

    def _match_binned_spikes_to_range_of_behavioral_data(self):
        self.max_bin = self.final_behavioral_data['bin'].max()
        self.binned_spikes_matrix, self.binned_spikes_df = neural_data_processing.prepare_binned_spikes_matrix_and_df(self.all_binned_spikes, self.max_bin)

    def _get_x_and_y_var(self):
        self.x_var = self.binned_spikes_df.copy()
        self.y_var = self.final_behavioral_data.copy()

    def _get_x_and_y_var_lags(self, max_lag_number):
        self.max_lag_number = max_lag_number
        self.lag_numbers = np.arange(-max_lag_number, max_lag_number+1)
        if not hasattr(self, 'x_var'):
            self._get_x_and_y_var()
        self.y_var_lags = neural_data_processing.add_lags_to_each_feature(self.y_var, self.lag_numbers)
