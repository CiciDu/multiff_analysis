from decision_making_analysis.data_enrichment import rsw_vs_rcap_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from data_wrangling import general_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data
from decision_making_analysis.event_detection import get_miss_to_switch_data
from pattern_discovery import cluster_analysis
from visualization.matplotlib_tools import plot_trials
from decision_making_analysis.data_enrichment import miss_events_enricher
from data_wrangling import base_processing_class, further_processing_class
from decision_making_analysis.data_enrichment import trajectory_class
from decision_making_analysis.ff_data_acquisition import missed_ff_data_class
from decision_making_analysis.ff_data_acquisition import ff_data_utils
from decision_making_analysis.modeling import ml_for_decision_making_class

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import copy


class MissEventsClass(further_processing_class.FurtherProcessing, trajectory_class.TrajectoryDataClass, missed_ff_data_class.MissedFfDataClass,
                      miss_events_enricher.MissEventsDataEnricher, ml_for_decision_making_class.MLForDecisionMakingClass):

    plotting_kwargs = {'show_stops': True,
                       'show_believed_target_positions': True,
                       'show_reward_boundary': True,
                       'show_scale_bar': True,
                       'truncate_part_before_crossing_arena_edge': True,
                       'trial_too_short_ok': True,
                       'show_connect_path_ff': False,
                       'show_visible_fireflies': True}

    def __init__(self,
                 raw_data_folder_path='all_monkey_data/raw_monkey_data/monkey_Bruno/data_0330',
                 ref_point_mode='distance',
                 ref_point_value=-150,
                 stop_period_duration=2,
                 retrieve_monkey_data=True,
                 time_range_of_trajectory=[-2.5, 0],
                 num_time_points_for_trajectory=5,
                 gc_kwargs=None,
                 new_point_index_start=0):

        # --- initialize folder structure ---
        if gc_kwargs is None:
            # use default base class initialization
            super().__init__(raw_data_folder_path=raw_data_folder_path)
        else:
            # alternative initialization route when gc_kwargs is passed directly
            base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
                self, raw_data_folder_path
            )

        # --- shared / basic setup ---
        self.curv_of_traj_df = None
        self.curv_of_traj_df_w_one_sided_window = None
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.stop_period_duration = stop_period_duration
        self.new_point_index_start = new_point_index_start

        # --- folder paths ---
        self.miss_events_folder_path = os.path.join(
            self.decision_making_folder_path, 'miss_events_info')
        self.rsw_vs_rcap_folder_path = os.path.join(
            self.decision_making_folder_path, 'rsw_vs_rcap')
        os.makedirs(self.miss_events_folder_path, exist_ok=True)
        os.makedirs(self.rsw_vs_rcap_folder_path, exist_ok=True)

        # --- trajectory & GC-related parameters ---
        self.time_range_of_trajectory = time_range_of_trajectory
        self.gc_kwargs = copy.deepcopy(miss_events_enricher.gc_kwargs)
        if gc_kwargs is not None:
            self.gc_kwargs.update(gc_kwargs)

        self.polar_plots_kwargs = {}

        # trajectory features come from gc_kwargs if available
        self.trajectory_features = (
            gc_kwargs['trajectory_features']
            if gc_kwargs and 'trajectory_features' in gc_kwargs
            else ['monkey_distance', 'monkey_angle_to_origin']
        )

        # --- optional data retrieval ---
        if retrieve_monkey_data:
            self.get_monkey_data(include_ff_dataframe=False)

    def make_decision_making_basic_ff_info(self):
        # This is another way to gather features to apply ML, just using much more simple features
        self.streamline_getting_rsw_or_rcap_df(rsw_or_rcap='rcap')
        self.streamline_getting_rsw_or_rcap_df(rsw_or_rcap='rsw')

        self.rcap_df['whether_switched'] = 0
        self.rsw_df['whether_switched'] = 1

        self.decision_making_basic_ff_info = pd.concat([self.rcap_df, self.rsw_df], axis=0).reset_index(
            drop=True)

        self.decision_making_basic_ff_info.drop(
            columns=['cur_ff_capture_time'], inplace=True)

        # drop rows with NA in decision_making_basic_ff_info
        self.decision_making_basic_ff_info_cleaned = general_utils.drop_rows_with_any_na(
            self.decision_making_basic_ff_info)

    def get_relevant_monkey_data(self,
                                 already_retrieved_ok=True,
                                 ):

        include_rcap_data = True if self.rsw_or_rcap == 'rcap' else False
        include_rsw_data = True if self.rsw_or_rcap == 'rsw' else False
        self.get_monkey_data(already_retrieved_ok=already_retrieved_ok,
                             include_rsw_data=include_rsw_data, include_rcap_data=include_rcap_data)

        if self.rsw_or_rcap == 'rcap':
            self.furnish_rcap_events_df()
            # # because we need to have nxt_ff, we will limit the max number of ff_index to len(self.ff_caught_T_new - 2)
            # self.rcap_events_df = self.rcap_events_df[self.rcap_events_df['ff_index'] < len(self.ff_caught_T_new) - 2]
        self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible'] == 1]

    def find_patterns(self):
        super().find_patterns()
        self.furnish_rcap_events_df()

    def furnish_rcap_events_df(self):
        self.rcap_events_df['stop_1_time'] = self.monkey_information.loc[
            self.rcap_events_df['stop_1_point_index'], 'time'].values
        self.rcap_events_df['ff_index'] = self.rcap_events_df['trial']
        # because we need to have nxt_ff, we will limit the max number of ff_index to len(self.ff_caught_T_new - 2)

        if 'stop_point_index' not in self.rcap_events_df.columns:
            self.rcap_events_df = rsw_vs_rcap_utils.add_stop_point_index(
                self.rcap_events_df, self.monkey_information, self.ff_real_position_sorted)

        if 'cur_ff_index' not in self.rcap_events_df.columns:
            self.rcap_events_df = rsw_vs_rcap_utils.add_cur_ff_info(
                self.rcap_events_df, self.ff_real_position_sorted)

    def streamline_getting_rsw_or_rcap_df(self, rsw_or_rcap='rsw'):
        self.rsw_or_rcap = rsw_or_rcap

        if hasattr(self, 'stops_near_ff_df'):
            del self.stops_near_ff_df
        self.get_relevant_monkey_data()
        self.get_rsw_or_rcap_df()

    def get_rsw_or_rcap_df(self):
        if self.rsw_or_rcap == 'rcap':
            self._get_rcap_df()
            self._get_rcap_df2_based_on_ref_point()
        elif self.rsw_or_rcap == 'rsw':
            self._get_rsw_df()
            self._get_rsw_df2_based_on_ref_point()
        else:
            raise ValueError('rsw_or_rcap must be either rcap or rsw')

    def make_rsw_cluster_df(self):
        # ff_indices_of_each_cluster = self.rsw_w_ff_df['nearby_alive_ff_indices'].values
        ff_indices_of_each_cluster = self.rsw_w_ff_df['ff_index'].values
        ff_indices_of_each_cluster = [[ff]
                                      for ff in ff_indices_of_each_cluster]

        rsw_last_stop_time = self.rsw_w_ff_df['last_stop_time'].values

        self.rsw_cluster_df = cluster_analysis.find_ff_cluster_last_vis_df(
            ff_indices_of_each_cluster, rsw_last_stop_time, ff_dataframe=self.ff_dataframe, cluster_identifiers=self.rsw_w_ff_df['stop_cluster_id'].values)

        self.rsw_cluster_df.rename(
            columns={'cluster_identifier': 'stop_cluster_id'}, inplace=True)

        self.rsw_cluster_df = self.rsw_cluster_df.merge(self.rsw_w_ff_df[['stop_cluster_id', 'stop_1_time', 'stop_2_time', 'last_stop_time', 'stop_1_point_index',
                                                                          'stop_2_point_index', 'last_stop_point_index', 'target_index', 'num_stops']],
                                                        on='stop_cluster_id', how='left')

        # to prepare for free selection
        self.rsw_cluster_df['latest_visible_time_before_last_stop'] = self.rsw_cluster_df['last_stop_time'] - \
            self.rsw_cluster_df['time_since_last_vis']

        # sort by last_stop_time (note that the order in rsw_cluster_df will henceforward be different from other variables)
        self.rsw_cluster_df.sort_values(by='last_stop_time', inplace=True)

    def _get_rcap_df(self):
        self.rcap_df = rsw_vs_rcap_utils.process_events_df(
            self.rcap_events_df, self.monkey_information, self.ff_dataframe, self.ff_real_position_sorted, self.stop_period_duration)
        self.rcap_df['cur_ff_capture_time'] = self.ff_caught_T_new[self.rcap_df['ff_index'].values]
        if self.rcap_df['ff_index'].max() == len(self.ff_caught_T_new) - 1:
            # remove the last row since later we need nxt_ff
            self.rcap_df = self.rcap_df[self.rcap_df['ff_index'] < len(
                self.ff_caught_T_new) - 1].reset_index(drop=True)

    def _get_rsw_df(self):
        self.rsw_df = rsw_vs_rcap_utils.process_events_df(
            self.rsw_w_ff_df, self.monkey_information, self.ff_dataframe, self.ff_real_position_sorted, self.stop_period_duration)
        if self.rsw_df['ff_index'].max() == len(self.ff_caught_T_new) - 1:
            # remove the last row since later we need nxt_ff
            self.rsw_df = self.rsw_df[self.rsw_df['ff_index'] < len(
                self.ff_caught_T_new) - 1].reset_index(drop=True)

    def get_monkey_data(self, already_retrieved_ok=True, include_monkey_information=True, include_ff_dataframe=True, include_rsw_data=False,
                        include_rcap_data=False):
        self.extract_info_from_raw_data_folder_path(self.raw_data_folder_path)
        self.retrieve_or_make_monkey_data(
            include_monkey_information=include_monkey_information)

        if include_ff_dataframe:
            if (already_retrieved_ok is False) | ((getattr(self, 'ff_dataframe', None) is None)):
                self.make_or_retrieve_ff_dataframe(
                    exists_ok=True)
                self.ff_dataframe_visible = self.ff_dataframe.loc[self.ff_dataframe['visible'] == 1].copy(
                )

        if include_rsw_data:
            if getattr(self, 'rsw_events_df', None) is None:
                if getattr(self, 'ff_dataframe', None) is None:
                    self.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                self.get_retry_switch_info()

        if include_rcap_data:
            if getattr(self, 'rcap_events_df', None) is None:
                if getattr(self, 'ff_dataframe', None) is None:
                    self.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                self.get_retry_capture_info()

        if include_ff_dataframe:
            self.cluster_around_target_indices = None
            self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted,
                                    self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)

    def determine_n_seconds_before_or_after_crossing_boundary(self, n_seconds_before_crossing_boundary=None, n_seconds_after_crossing_boundary=None):
        if n_seconds_before_crossing_boundary is None:
            n_seconds_before_crossing_boundary = max(
                0, self.time_range_of_trajectory[1])
        if n_seconds_after_crossing_boundary is None:
            n_seconds_after_crossing_boundary = max(
                0, - self.time_range_of_trajectory[0])
        return n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary

    def eliminate_crossing_boundary_cases(self, n_seconds_before_crossing_boundary=None, n_seconds_after_crossing_boundary=None):
        # Determine window sizes if not provided
        n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary = (
            self.determine_n_seconds_before_or_after_crossing_boundary(
                n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary
            )
        )

        # Extract boundary crossing times
        crossing_boundary_time = self.monkey_information.loc[
            self.monkey_information['crossing_boundary'] == 1, 'time'
        ].values

        # Helper function for filtering DataFrames
        def filter_df(df, label):
            input_time = df.time.values
            original_length = len(input_time)

            _, non_CB_indices, _ = ff_data_utils.find_time_points_that_are_within_n_seconds_after_crossing_boundary(
                input_time,
                crossing_boundary_time,
                n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary,
                n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
            )

            filtered_df = df.iloc[non_CB_indices].reset_index(drop=True)
            print(
                f'{label}: {filtered_df.shape[0]} / {original_length} rows remain after filtering')
            return filtered_df

        # Apply filtering to all relevant DataFrames, if they exist
        for attr_name in ['replacement_df', 'prior_to_replacement_df', 'free_selection_df', 'non_chosen_df', 'miss_events_df']:
            df = getattr(self, attr_name, None)
            if df is not None and hasattr(df, 'time'):
                setattr(self, attr_name, filter_df(df, attr_name))
