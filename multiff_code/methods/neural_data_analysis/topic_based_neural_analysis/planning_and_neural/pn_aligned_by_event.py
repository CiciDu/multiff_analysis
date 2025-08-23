from data_wrangling import general_utils
from planning_analysis.show_planning import nxt_ff_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_helper_class, pn_aligned_by_seg
import numpy as np
import pandas as pd
import os
import math


class PlanningAndNeuralEventAligned(pn_aligned_by_seg.PlanningAndNeuralSegmentAligned):

    def __init__(self, raw_data_folder_path=None,
                 bin_width=0.05,
                 one_point_index_per_bin=False):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin)

        self.gpfa_data_folder_path = os.path.join(
            self.planning_and_neural_folder_path, 'event_aligned')
        os.makedirs(self.gpfa_data_folder_path, exist_ok=True)

    def streamline_preparing_event_aligned_data(self, cur_or_nxt='cur', first_or_last='first',
                                                time_limit_to_count_sighting=2,
                                                pre_event_window=0.25, post_event_window=0.75,
                                                rebinned_max_x_lag_number=2,
                                                latent_dimensionality=7,
                                                use_raw_spike_data_instead=False,
                                                use_lagged_raw_spike_data=False,
                                                apply_pca_on_raw_spike_data=False):

        self.prepare_seg_aligned_data(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last,
                                      time_limit_to_count_sighting=time_limit_to_count_sighting,
                                      pre_event_window=pre_event_window, post_event_window=post_event_window,
                                      rebinned_max_x_lag_number=rebinned_max_x_lag_number)

        self.get_gpfa_traj(
            latent_dimensionality=latent_dimensionality, exists_ok=False)

        self.get_concat_data_for_regression(use_raw_spike_data_instead=use_raw_spike_data_instead,
                                            use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                            apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                            num_pca_components=7,
                                            )

        self.print_data_dimensions()

        self.make_time_resolved_cv_scores()

        self.plot_time_resolved_regression()

        self.plot_trial_counts_by_timepoint()

    def get_gpfa_traj(self, latent_dimensionality=10, exists_ok=True):
        bin_width_str = f"{self.bin_width:.4f}".rstrip(
            '0').rstrip('.').replace('.', 'p')
        file_name = f'gpfa_neural_bin{bin_width_str}_{self.cur_or_nxt}_{self.first_or_last}.pkl'

        super().get_gpfa_traj(latent_dimensionality=latent_dimensionality,
                              exists_ok=exists_ok, file_name=file_name)

    def _get_time_resolved_cv_scores_file_path(self, folder_name, file_name, cv_folds=5, latent_dimensionality=7):
        bin_width_str = f"{self.bin_width:.4f}".rstrip(
            '0').rstrip('.').replace('.', 'p')
        # file_name = f'scores_bin{bin_width_str}_{self.cur_or_nxt}_{self.first_or_last}_d{latent_dimensionality}_cv{cv_folds}.csv'

        file_name = (
            f"scores_b{bin_width_str}"
            f"_t{general_utils.clean_float(self.time_limit_to_count_sighting)}"
            f"_{self.cur_or_nxt}_{self.first_or_last}"
            f"_p{general_utils.clean_float(self.pre_event_window)}"
            f"_po{general_utils.clean_float(self.post_event_window)}"
            f"_d{latent_dimensionality}_cv{cv_folds}.csv"
        )
        time_resolved_cv_scores_path = super()._get_time_resolved_cv_scores_file_path(
            folder_name, file_name, cv_folds=cv_folds, latent_dimensionality=latent_dimensionality)

        return time_resolved_cv_scores_path

    # def retrieve_or_make_time_resolved_cv_scores_gpfa(self, cv_folds=5, latent_dimensionality=7, exists_ok=True):
    #     bin_width_str = f"{self.bin_width:.4f}".rstrip(
    #         '0').rstrip('.').replace('.', 'p')
    #     # file_name = f'scores_bin{bin_width_str}_{self.cur_or_nxt}_{self.first_or_last}_d{latent_dimensionality}_cv{cv_folds}.csv'

    #     file_name = (
    #         f"scores_bin{bin_width_str}"
    #         f"_tlim{general_utils.clean_float(self.time_limit_to_count_sighting)}"
    #         f"_{self.cur_or_nxt}_{self.first_or_last}"
    #         f"_pre{general_utils.clean_float(self.pre_event_window)}"
    #         f"_post{general_utils.clean_float(self.post_event_window)}"
    #         f"_d{latent_dimensionality}_cv{cv_folds}.csv"
    #     )

    #     super().retrieve_or_make_time_resolved_cv_scores_gpfa(latent_dimensionality=latent_dimensionality,
    #                                                           exists_ok=exists_ok, file_name=file_name)

    def prepare_seg_aligned_data(self, cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                 pre_event_window=0.25, post_event_window=0.75, rebinned_max_x_lag_number=2):

        self.cur_or_nxt = cur_or_nxt
        self.first_or_last = first_or_last
        self.time_limit_to_count_sighting = time_limit_to_count_sighting

        self.rebin_data_in_new_segments(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last,
                                        time_limit_to_count_sighting=time_limit_to_count_sighting,
                                        pre_event_window=pre_event_window, post_event_window=post_event_window,
                                        rebinned_max_x_lag_number=rebinned_max_x_lag_number)
        self.prepare_spikes_for_gpfa()

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

    def rebin_data_in_new_segments(self, cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                   pre_event_window=0.25, post_event_window=0.75, rebinned_max_x_lag_number=2):
        # time_limit_to_count_sighting: the time threshold to consider a firefly sighting valid. In other words, only the sighting between stop_time - time_limit_to_count_sighting and stop_time is considered.
        self.get_new_seg_info(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last,
                              time_limit_to_count_sighting=time_limit_to_count_sighting,
                              pre_event_window=pre_event_window, post_event_window=post_event_window)
        self._rebin_data_in_new_segments(
            rebinned_max_x_lag_number=rebinned_max_x_lag_number)
        print('Made rebinned_x_var, rebinned_y_var, rebinned_x_var_lags, and rebinned_y_var_lags.')

        # add bin_mid_time_rel_to_event
        # first use merge to get event time
        if 'event_time' not in self.rebinned_y_var.columns:
            self.rebinned_y_var = self.rebinned_y_var.merge(
                self.new_seg_info[['event_time', 'new_segment']], on='new_segment', how='left')

        self.rebinned_y_var['bin_mid_time_rel_to_event'] = self.rebinned_y_var['new_bin'] * self.bin_width + \
            self.bin_width/2 + \
            self.rebinned_y_var['new_seg_start_time'] - \
            self.rebinned_y_var['event_time']

    def get_new_seg_info(self, cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                         pre_event_window=0.25, post_event_window=0.75, exists_ok=True):

        self.new_bin_start_time = -pre_event_window
        self.event_time = 0

        folder_name = os.path.join(self.planning_and_neural_folder_path,
                                   'new_seg_info')

        os.makedirs(folder_name, exist_ok=True)

        self.fix_post_event_window_if_needed(
            pre_event_window, post_event_window)

        df_name = (
            f"tlim{general_utils.clean_float(time_limit_to_count_sighting)}"
            f"_{cur_or_nxt}_{first_or_last}"
            f"_pre{general_utils.clean_float(self.pre_event_window)}"
            f"_post{general_utils.clean_float(self.post_event_window)}.csv"
        )

        df_path = os.path.join(
            folder_name, df_name)

        if exists_ok & os.path.exists(df_path):
            self.new_seg_info = pd.read_csv(df_path, index_col=False)
            print(f'Loaded new_seg_info from {df_path}')
            return self.new_seg_info
        else:
            event_df = self.get_new_ff_first_or_last_time(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last,
                                                          time_limit_to_count_sighting=time_limit_to_count_sighting,
                                                          )
            self._get_new_seg_info(event_df)
            self.new_seg_info.to_csv(df_path, index=False)
            print(f'Made new new_seg_info and saved to {df_path}')
            return self.new_seg_info

    def get_new_ff_first_or_last_time(self, cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2):
        # Compute the first time each ff is seen after a threshold before stop, for all ff in planning_data_by_point.
        # Determine the correct index column
        self.cur_or_nxt = cur_or_nxt
        self.first_or_last = first_or_last

        ff_index_map = {
            'cur': 'cur_ff_index',
            'nxt': 'nxt_ff_index'
        }
        try:
            ff_index_column = ff_index_map[cur_or_nxt]
        except KeyError:
            raise ValueError(
                f"cur_or_nxt must be 'cur' or 'nxt', got '{cur_or_nxt}'")

        # Validate first_or_last input
        if first_or_last not in {'first', 'last'}:
            raise ValueError(
                f"first_or_last must be 'first' or 'last', got '{first_or_last}'")

        self.retrieve_or_make_monkey_data()
        self.make_or_retrieve_ff_dataframe()

        self.temp_segment_df = self.select_segments_with_enough_duration(
            time_limit_to_count_sighting)

        ff_when_seen_info = nxt_ff_utils.find_first_or_last_ff_sighting_in_stop_period(
            self.temp_segment_df, ff_index_column, self.ff_dataframe, first_or_last=first_or_last)

        event_df = ff_when_seen_info[['time', 'stop_point_index']].drop_duplicates().merge(
            self.temp_segment_df[['segment', 'stop_point_index', 'stop_time', 'prev_ff_caught_time']], on='stop_point_index', how='left')
        event_df.drop(columns=['stop_point_index'], inplace=True)
        event_df.rename(columns={'time': 'event_time'}, inplace=True)
        return event_df

    def select_segments_with_enough_duration(self, time_limit_to_count_sighting):
        # need to make sure the segment duration is long enough to include the first seen time threshold and the duration before it
        # min_segment_duration = time_limit_to_count_sighting + pre_event_window
        min_segment_duration = time_limit_to_count_sighting
        planning_data_sub = self.planning_data_by_point[
            self.planning_data_by_point['segment_duration'] >= min_segment_duration]

        # get unique segments
        segment_df = planning_data_sub[[
            'cur_ff_index', 'nxt_ff_index', 'segment', 'stop_point_index', 'stop_time']].drop_duplicates()

        # print number of segments dropped because of insufficient duration
        original_segments = len(
            self.planning_data_by_point['stop_point_index'].unique())
        dropped_segments = original_segments - len(segment_df)
        print(f"Dropped {dropped_segments} out of {original_segments} segments because of insufficient duration, "
              f"which is {round(dropped_segments/original_segments * 100, 1)}% of all segments")

        # add beginning time and prev ff caught time
        segment_df['prev_ff_caught_time'] = self.ff_caught_T_new[segment_df['cur_ff_index'].values-1]
        segment_df['beginning_time'] = np.maximum(segment_df['stop_time'].values - time_limit_to_count_sighting,
                                                  segment_df['prev_ff_caught_time'].values + 0.1)
        return segment_df

    # def _get_even_time_through_stops_near_ff_df(self, event_time_column_name='CUR_time_ff_first_seen_bbas'):
    #     # this is currently not used because it might result in insufficient data for some segments
        # event_df = self.planning_data_by_bin[[
        #     'segment', 'stop_point_index']].drop_duplicates()

        # successful_retrieval = cvn_helper_class._FindCurVsNxtFF.retrieve_shared_stops_near_ff_df(
        #     self)
        # if not successful_retrieval:
        #     print(
        #         'Failed to retrieve shared_stops_near_ff_df; will make new shared_stops_near_ff_df')
        #     self.only_make_stops_near_ff_df

        # event_df = event_df.merge(self.shared_stops_near_ff_df[[
        #                'stop_point_index', event_time_column_name]], on='stop_point_index', how='left')
        # event_df.rename(
        #     columns={event_time_column_name: 'event_time'}, inplace=True)

        # return event_df

    def _get_new_seg_info(self, event_df):
        event_df['new_segment'] = np.arange(len(event_df))
        self.new_seg_info = event_df.copy()

        self.new_seg_info['new_seg_start_time'] = self.new_seg_info['event_time'] - \
            self.pre_event_window
        self.new_seg_info['new_seg_end_time'] = self.new_seg_info['event_time'] + \
            self.post_event_window
        self.new_seg_info['new_seg_duration'] = self.new_seg_duration

    def fix_post_event_window_if_needed(self, pre_event_window, post_event_window):
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        self.new_seg_duration = pre_event_window + post_event_window
        # need to make sure that new_seg_duration is a multiple of bin_width
        if self.new_seg_duration % self.bin_width != 0:
            print(
                f"Warning: new_seg_duration {self.new_seg_duration} is not a multiple of bin_width {self.bin_width}")
            self.new_seg_duration = math.floor(
                self.new_seg_duration / self.bin_width) * self.bin_width
            # increase the precision
            self.new_seg_duration = round(self.new_seg_duration, 4)
            self.post_event_window = self.new_seg_duration - self.pre_event_window
            print(
                f"new_seg_duration is now {self.new_seg_duration}, and post_event_window is now {self.post_event_window}")

    def only_make_stops_near_ff_df(self):
        self.data_kwargs1 = {'raw_data_folder_path': self.raw_data_folder_path,
                             'one_point_index_per_bin': self.one_point_index_per_bin}

        self.test_inst = pn_helper_class.PlanningAndNeuralHelper(test_or_control='test',
                                                                 **self.data_kwargs1)
        self.test_inst._only_make_stops_near_ff_df()


# These might be useful in the future

    # def get_new_seg_info(self, event_time_column_name='CUR_time_ff_first_seen_bbas',
    #                      pre_event_window=0.25, post_event_window=0.75):

    #     # merge to get event time
    #     planning_data = self.planning_data_by_bin.merge(
    #         self.shared_stops_near_ff_df[['stop_point_index', event_time_column_name]], on='stop_point_index', how='left')
    #     planning_data.rename(
    #         columns={event_time_column_name: 'event_time'}, inplace=True)

    #     # select segment data around event time
    #     self.planning_data_sub = pn_utils.select_segment_data_around_event_time(
    #         planning_data, pre_event_window=pre_event_window, post_event_window=post_event_window)

    #     # get unique rows of segment info
    #     self.new_seg_info = pn_utils._get_new_seg_info(self.planning_data_sub)


#    def build_segment_around_event(self, row, event_time_column_name='CUR_time_ff_first_seen_bbas',
#                                    pre_event_window=0.25, post_event_window=0.75):

#         # other event_time_column_name to use: CUR_time_ff_last_seen_bbas, NXT_time_ff_first_seen_bbas, NXT_time_ff_last_seen_bbas

#         event_time = row[event_time_column_name]
#         seg_start_time = event_time - pre_event_window
#         seg_end_time = event_time + post_event_window
#         # note: segment_start = segment_end can happen if two fireflies were captured in a row.
#         info_to_add = self.monkey_information[self.monkey_information['time'].between(
#             seg_start_time, seg_end_time)].copy()

#         info_to_add.loc[len(info_to_add)] = {
#             **row[['stop_time', 'cur_ff_index', 'nxt_ff_index']].to_dict(),
#             'event_time': event_time,
#             'seg_start_time': seg_start_time,
#             'seg_end_time': seg_end_time
#         }
#         return info_to_add
