import sys
from data_wrangling import general_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.target_decoder import prep_target_decoder, behav_features_to_keep, target_decoder_class
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import pn_utils, pn_helper_class, planning_neural_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils, gpfa_helper_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
import numpy as np
import pandas as pd
import os


class PlanningAndNeuralEventAligned(planning_neural_class.PlanningAndNeural, gpfa_helper_class.GPFAHelperClass):

    def __init__(self, raw_data_folder_path=None,
                 bin_width=0.1,
                 one_point_index_per_bin=False):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin)

    def rebin_data_in_new_segments(self, segment_duration=2, rebinned_max_x_lag_number=2):
        self._get_new_seg_info(segment_duration=segment_duration)

        # rebin y_var (behavioral data)
        self.rebinned_y_var = pn_utils.rebin_segment_data(
            self.planning_data_by_point, self.new_seg_info, bin_width=self.bin_width)

        # drop columns with na
        self.rebinned_y_var = general_utils.drop_na_cols(
            self.rebinned_y_var, df_name='rebinned_y_var')

        # make new_segment, new_bin, and target_index all integers
        self.rebinned_y_var[['new_segment', 'new_bin', 'target_index']] = self.rebinned_y_var[[
            'new_segment', 'new_bin', 'target_index']].astype(int)

        # rebin x_var (neural data)
        spike_df = neural_data_processing.make_spike_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                        sampling_rate=self.sampling_rate)

        self.rebinned_x_var = pn_utils.rebin_spike_data(
            spike_df, self.new_seg_info, bin_width=self.bin_width)

        self._get_rebinned_x_var_lags(
            rebinned_max_x_lag_number=rebinned_max_x_lag_number)

    def prepare_spikes_for_gpfa(self, align_at_beginning=False):
        if not hasattr(self, 'new_seg_info'):
            raise ValueError(
                'new_seg_info not found. Please run rebin_data_in_new_segments first.')

        gpfa_helper_class.GPFAHelperClass._prepare_spikes_for_gpfa(
            self, self.new_seg_info, align_at_beginning=align_at_beginning)

    def get_rebinned_behav_data(self):
        self.rebinned_behav_data = self.rebinned_y_var.sort_values(
            by=['new_segment', 'new_bin'])

    def _get_new_seg_info(self, segment_duration=2):
        self.segment_duration = segment_duration

        # Take out segments where segment duration is greater than the specified segment_duration
        # This assumes that n_seconds_before_stop was greater than segment_duration when creating planning_data_by_point
        planning_data_sub = self.planning_data_by_bin[self.planning_data_by_bin['segment_duration'] > 2].copy(
        )
        # for each segment, we want to only take out the time points that are within the segment duration (aligned to a reference point such as stop_time)
        planning_data_sub['new_seg_end_time'] = planning_data_sub['stop_time']
        planning_data_sub['new_seg_start_time'] = planning_data_sub['new_seg_end_time'] - \
            segment_duration
        planning_data_sub['new_seg_duration'] = segment_duration
        self.new_seg_info = planning_data_sub[[
            'segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']].drop_duplicates()
        self.new_seg_info['new_segment'] = pd.factorize(
            self.new_seg_info['segment'])[0]

    def _get_rebinned_x_var_lags(self, rebinned_max_x_lag_number=2):
        trial_vector = self.rebinned_x_var['new_segment'].values
        self.rebinned_max_x_lag_number = rebinned_max_x_lag_number
        self.rebinned_x_var_lags, self.rebinned_x_lag_numbers = self._get_lags(
            self.rebinned_max_x_lag_number, self.rebinned_x_var, trial_vector=trial_vector)

        if 'new_bin_0' in self.rebinned_x_var_lags.columns:
            self.rebinned_x_var_lags['new_bin'] = self.rebinned_x_var_lags['new_bin_0'].astype(
                int)
            self.rebinned_x_var_lags = self.rebinned_x_var_lags.drop(
                columns=[col for col in self.rebinned_x_var_lags.columns if 'new_bin_' in col])
        if 'new_segment_0' in self.rebinned_x_var_lags.columns:
            self.rebinned_x_var_lags['new_segment'] = self.rebinned_x_var_lags['new_segment_0'].astype(
                int)
            self.rebinned_x_var_lags = self.rebinned_x_var_lags.drop(
                columns=[col for col in self.rebinned_x_var_lags.columns if 'new_segment_' in col])

        assert self.rebinned_x_var_lags['new_bin'].equals(
            self.rebinned_x_var['new_bin'])

    def get_raw_spikes_for_regression(self):
        if self.use_lagged_raw_spike_data:
            x_var_df = self.rebinned_x_var_lags.copy()
        else:
            x_var_df = self.rebinned_x_var.copy()

        return x_var_df

    def separate_test_and_control_data(self):
        test_data_mask = self.concat_behav_trials['whether_test'] == 1

        self.test_concat_behav_trials = self.concat_behav_trials[test_data_mask]
        self.control_concat_behav_trials = self.concat_behav_trials[~test_data_mask]

        self.test_concat_neural_trials = self.concat_neural_trials[test_data_mask]
        self.control_concat_neural_trials = self.concat_neural_trials[~test_data_mask]

    def get_concat_and_y_var_for_lr(self, test_or_control='both'):
        if test_or_control == 'test':
            x_var = self.test_concat_neural_trials.drop(
                columns=['new_segment', 'new_bin'], errors='ignore')
            y_var = self.test_concat_behav_trials
        elif test_or_control == 'control':
            x_var = self.control_concat_neural_trials.drop(
                columns=['new_segment', 'new_bin'], errors='ignore')
            y_var = self.control_concat_behav_trials
        elif test_or_control == 'both':
            x_var = self.concat_neural_trials.drop(
                columns=['new_segment', 'new_bin'], errors='ignore')
            y_var = self.concat_behav_trials
        else:
            raise ValueError(
                f'test_or_control must be "test", "control", or "both". Got {test_or_control}')

        # print dimensions of x_var and y_var
        print('test_or_control:', test_or_control)
        print(f'x_var dimensions: {x_var.shape}')
        print(f'y_var dimensions: {y_var.shape}')

        return x_var, y_var
