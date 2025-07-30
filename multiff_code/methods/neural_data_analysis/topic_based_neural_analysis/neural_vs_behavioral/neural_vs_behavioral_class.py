import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
import os
import sys
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


class NeuralVsBehavioralClass(base_neural_class.NeuralBaseClass):
    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.02,
                 one_point_index_per_bin=True):

        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin)

    def streamline_preparing_neural_and_behavioral_data(self, max_y_lag_number=3):
        self.get_basic_data()
        self.make_relevant_paths()
        self._prepare_to_find_patterns_and_features()
        self.make_df_related_to_patterns_and_features()
        self.prep_behavioral_data_for_neural_data_modeling()
        self.max_bin = self.final_behavioral_data['bin'].max()
        self.retrieve_neural_data()
        self._get_x_and_y_var()
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=self.final_behavioral_data)

    def make_relevant_paths(self):
        self.vif_df_path = os.path.join(
            self.processed_neural_data_folder_path, 'vif_df')
        self.lr_df_path = os.path.join(
            self.processed_neural_data_folder_path, 'lr_df')
        os.makedirs(self.vif_df_path, exist_ok=True)
        os.makedirs(self.lr_df_path, exist_ok=True)

    def prep_behavioral_data_for_neural_data_modeling(self):
        self.binned_features, self.time_bins = prep_monkey_data.initialize_binned_features(
            self.monkey_information, self.bin_width)
        self.binned_features = self._add_ff_info(self.binned_features)
        self._add_monkey_info()
        self._add_all_target_and_target_cluster_info()
        self._add_pattern_info_based_on_points_and_trials()
        self._make_final_behavioral_data()
        self._get_index_of_bins_in_valid_intervals()

    def make_or_retrieve_y_var_lr_df(self, exists_ok=True):
        df_path = os.path.join(self.lr_df_path,
                               'y_var_lr_df.csv')
        if exists_ok & exists(df_path):
            self.y_var_lr_df = pd.read_csv(df_path)
        else:
            self.y_var_lr_df = neural_data_modeling.get_y_var_lr_df(
                self.x_var, self.y_var)
            self.y_var_lr_df.to_csv(df_path, index=False)
            print('Made new y_var_lr_df')

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
            self.monkey_information, self.time_bins, one_point_index_per_bin=self.one_point_index_per_bin)
        self.monkey_info_in_bins_ess = prep_monkey_data.make_monkey_info_in_bins_essential(
            self.monkey_info_in_bins, self.time_bins, self.ff_caught_T_new)
        self.binned_features = self.binned_features.merge(
            self.monkey_info_in_bins_ess, how='left', on='bin')

    def _add_all_target_and_target_cluster_info(self):
        self._make_or_retrieve_target_df()
        self._make_or_retrieve_target_cluster_df()
        self._make_cmb_target_df()

        if self.one_point_index_per_bin:
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
        # columns_to_drop = [col for col in self.final_behavioral_data.columns if (
        #     'angle_to_boundary' in col) or ('angle_boundary' in col)]
        # self.final_behavioral_data = self.final_behavioral_data.drop(
        #     columns=columns_to_drop)

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

        # make sure that valid_bin_index doesn't exceed the max index of the final_behavioral_data
        mask = self.valid_bin_index <= self.final_behavioral_data.index.max()
        self.valid_bin_index = self.valid_bin_index[mask]
        self.valid_bin_mid_time = self.valid_bin_mid_time[mask]

        # # print the number of bins out of total numbers that are in valid intervals
        # print(f"Number of bins in valid intervals based on ff caught time: {len(self.valid_bin_index)} out of {len(mid_bin_time)}"
        #       f" ({len(self.valid_bin_index)/len(mid_bin_time)*100:.2f}%)")
