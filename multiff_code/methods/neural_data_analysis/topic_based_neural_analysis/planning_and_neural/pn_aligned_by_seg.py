from data_wrangling import general_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils, planning_and_neural_class
from neural_data_analysis.neural_analysis_tools.gpfa_methods import gpfa_helper_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.align_trials import align_trial_utils

import pandas as pd
import os


class PlanningAndNeuralSegmentAligned(planning_and_neural_class.PlanningAndNeural, gpfa_helper_class.GPFAHelperClass):

    def __init__(self, raw_data_folder_path=None,
                 bin_width=0.05,
                 one_point_index_per_bin=False):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin)

        self.gpfa_data_folder_path = os.path.join(
            self.planning_and_neural_folder_path, 'seg_aligned')
        os.makedirs(self.gpfa_data_folder_path, exist_ok=True)

    def prepare_seg_aligned_data(self, segment_duration=2, rebinned_max_x_lag_number=2):
        self.rebin_data_in_new_segments(segment_duration=segment_duration,
                                        rebinned_max_x_lag_number=rebinned_max_x_lag_number)
        self.prepare_spikes_for_gpfa(align_at_beginning=False)

    def get_concat_data_for_regression(self, use_raw_spike_data_instead=False,
                                       apply_pca_on_raw_spike_data=False,
                                       use_lagged_raw_spike_data=False,
                                       use_lagged_rebinned_behav_data=False,
                                       num_pca_components=7):
        self.get_rebinned_behav_data(
            use_lagged_rebinned_behav_data=use_lagged_rebinned_behav_data)
        self._get_concat_data_for_regression(use_raw_spike_data_instead=use_raw_spike_data_instead,
                                             apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                             use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                             num_pca_components=num_pca_components)
        self.separate_test_and_control_data()

    def rebin_data_in_new_segments(self, segment_duration=2, rebinned_max_x_lag_number=2):
        self.retrieve_or_make_monkey_data()
        self.get_new_seg_info(segment_duration=segment_duration)
        self._rebin_data_in_new_segments(
            rebinned_max_x_lag_number=rebinned_max_x_lag_number)
        print('Made rebinned_x_var, rebinned_y_var, rebinned_x_var_lags, and rebinned_y_var_lags.')

    def _rebin_data_in_new_segments(self, rebinned_max_x_lag_number=2):
        # rebin y_var (behavioral data)
        self.rebinned_y_var = pn_utils.rebin_segment_data(
            self.planning_data_by_point, self.new_seg_info, bin_width=self.bin_width)

        # drop columns with na
        self.rebinned_y_var = general_utils.drop_na_cols(
            self.rebinned_y_var, df_name='rebinned_y_var')

        # make new_segment, new_bin, and target_index all integers
        self.rebinned_y_var[['new_segment', 'new_bin', 'target_index']] = self.rebinned_y_var[[
            'new_segment', 'new_bin', 'target_index']].astype(int)

        # # rebin x_var (neural data)
        if not hasattr(self, 'spikes_df'):
            self.retrieve_or_make_monkey_data()
            self.spikes_df = neural_data_processing.make_spikes_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                                   sampling_rate=self.sampling_rate)

        self.rebinned_x_var = pn_utils.rebin_spike_data(
            self.spikes_df, self.new_seg_info, bin_width=self.bin_width)

        # only keep the combination of ['new_segment', 'new_bin'] in self.rebinned_x_var
        self.rebinned_x_var = self.rebinned_x_var.merge(self.rebinned_y_var[[
                                                        'new_segment', 'new_bin']], on=['new_segment', 'new_bin'], how='right')

        # assert that rebinned_y_var's [['new_segment', 'new_bin']] are the same as rebinned_x_var's [['new_segment', 'new_bin']]
        assert self.rebinned_y_var[['new_segment', 'new_bin']].equals(
            self.rebinned_x_var[['new_segment', 'new_bin']])

        self._get_rebinned_x_var_lags(
            rebinned_max_x_lag_number=rebinned_max_x_lag_number)

    def prepare_spikes_for_gpfa(self, align_at_beginning=False):
        if not hasattr(self, 'new_seg_info'):
            raise ValueError(
                'new_seg_info not found. Please run rebin_data_in_new_segments first.')

        gpfa_helper_class.GPFAHelperClass._prepare_spikes_for_gpfa(
            self, self.new_seg_info, align_at_beginning=align_at_beginning)

    def get_rebinned_behav_data(self, use_lagged_rebinned_behav_data=False):
        if use_lagged_rebinned_behav_data:
            if not hasattr(self, 'rebinned_y_var_lags'):
                self._get_rebinned_y_var_lags()
            self.rebinned_behav_data = self.rebinned_y_var_lags.sort_values(
                by=['new_segment', 'new_bin'])
            self.use_lagged_rebinned_behav_data = True
        else:
            self.rebinned_behav_data = self.rebinned_y_var.sort_values(
                by=['new_segment', 'new_bin'])

    def get_new_seg_info(self, segment_duration=2):
        self.new_seg_duration = segment_duration

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
            'segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration',
            'stop_time', 'cur_ff_index', 'nxt_ff_index']].drop_duplicates()
        self.new_seg_info['new_segment'] = pd.factorize(
            self.new_seg_info['segment'])[0]

        self.new_seg_info['prev_ff_caught_time'] = self.ff_caught_T_new[self.new_seg_info['cur_ff_index'].values-1]

    def _get_rebinned_x_var_lags(self, rebinned_max_x_lag_number=3):
        trial_vector = self.rebinned_x_var['new_segment'].values
        self.rebinned_max_x_lag_number = rebinned_max_x_lag_number
        self.rebinned_x_var_lags = align_trial_utils.get_rebinned_var_lags(
            self.rebinned_x_var, trial_vector, rebinned_max_lag_number=rebinned_max_x_lag_number)

    def _get_rebinned_y_var_lags(self, rebinned_max_y_lag_number=3):
        trial_vector = self.rebinned_y_var['new_segment'].values
        self.rebinned_max_y_lag_number = rebinned_max_y_lag_number
        self.rebinned_y_var_lags = align_trial_utils.get_rebinned_var_lags(
            self.rebinned_y_var, trial_vector, rebinned_max_lag_number=rebinned_max_y_lag_number)

    def get_raw_spikes_for_regression(self):
        if self.use_lagged_raw_spike_data:
            x_var_df = self.rebinned_x_var_lags.copy()
        else:
            x_var_df = self.rebinned_x_var.copy()

        return x_var_df

    def separate_test_and_control_data(self):
        if 'whether_test' not in self.concat_behav_trials.columns:
            if 'whether_test_0' in self.concat_behav_trials.columns:
                self.concat_behav_trials['whether_test'] = self.concat_behav_trials['whether_test_0']
            else:
                raise ValueError(
                    'Column whether_test not found in concat_behav_trials.')

        test_data_mask = self.concat_behav_trials['whether_test'] == 1

        self.test_concat_behav_trials = self.concat_behav_trials[test_data_mask]
        self.control_concat_behav_trials = self.concat_behav_trials[~test_data_mask]

        self.test_concat_neural_trials = self.concat_neural_trials[test_data_mask]
        self.control_concat_neural_trials = self.concat_neural_trials[~test_data_mask]

    def get_concat_x_and_y_var_for_lr(self, test_or_control='both'):
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
