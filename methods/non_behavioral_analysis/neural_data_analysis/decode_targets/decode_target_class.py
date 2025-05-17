import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, reduce_multicollinearity
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

    def streamline_making_behav_data(self):
        self.get_basic_data()
        self.retrieve_neural_data()
        self.get_behav_data()

    def get_behav_data(self):

        _, self.time_bins = prep_monkey_data.initialize_binned_features(
            self.monkey_information, self.bin_width)
        self.behav_data = prep_monkey_data.bin_monkey_information(
            self.monkey_information, self.time_bins, one_behav_idx_per_bin=self.one_behav_idx_per_bin)

        self.behav_data = self._add_ff_info(self.behav_data)
        self._add_all_target_info()
        self._add_curv_info()

        # clip values in some columns
        for column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
            self.behav_data.loc[:, column] = np.clip(
                self.behav_data.loc[:, column], -1000, 1000)

    def _add_curv_info(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        ff_df = self.behav_data[['point_index', 'target_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                                 'target_x', 'target_y', 'target_distance', 'target_angle', 'target_angle_to_boundary']]
        ff_df = ff_df.rename(columns={'target_x': 'ff_x', 'target_y': 'ff_y', 'target_angle': 'ff_angle',
                             'target_index': 'ff_index', 'target_distance': 'ff_distance', 'target_angle_to_boundary': 'ff_angle_boundary'})

        self.curv_of_traj_df = self.get_curv_of_traj_df(
            window_for_curv_of_traj=window_for_curv_of_traj,
            curv_of_traj_mode=curv_of_traj_mode,
            truncate_curv_of_traj_by_time_of_capture=False
        )

        self.curv_df = curvature_utils.make_curvature_df(ff_df, self.curv_of_traj_df, clean=False,
                                                         remove_invalid_rows=True,
                                                         monkey_information=self.monkey_information,
                                                         ff_caught_T_new=self.ff_caught_T_new,
                                                         ignore_error=True)
        self.behav_data = self.behav_data.merge(self.curv_df[[
                                                'point_index', 'curv_of_traj', 'optimal_arc_d_heading']], on='point_index', how='left')
        self.behav_data.rename(columns={
                               'curv_of_traj': 'traj_curv', 'optimal_arc_d_heading': 'target_opt_arc_dheading'}, inplace=True)

    def _add_all_target_info(self):

        self._make_or_retrieve_target_df()

        # drop columns in target_df that are duplicated in behav_data
        columns_to_drop = [
            col for col in self.target_df.columns if col in self.behav_data.columns]
        columns_to_drop.remove('point_index')
        target_df = self.target_df.drop(columns=columns_to_drop)

        self.behav_data = self.behav_data.merge(
            target_df, on='point_index', how='left')
