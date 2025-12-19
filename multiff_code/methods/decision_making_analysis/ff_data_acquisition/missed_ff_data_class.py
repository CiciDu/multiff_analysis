import os
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from os.path import exists

# ----------------------------------------------------------------------
# Internal module imports
# ----------------------------------------------------------------------
from data_wrangling import base_processing_class
from null_behaviors import curvature_utils
from decision_making_analysis.data_enrichment import trajectory_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data
from decision_making_analysis.event_detection import get_miss_to_switch_data
from decision_making_analysis.data_compilation import miss_events_class
from decision_making_analysis.ff_data_acquisition import cluster_replacement_utils
from decision_making_analysis.event_detection import detect_rsw_and_rcap
from null_behaviors import curv_of_traj_utils
from decision_making_analysis.ff_data_acquisition import free_selection


class MissedFfDataClass():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def streamline_making_current_and_alternative_ff_info(self,
                                                          add_one_stop_info=False,
                                                          time_with_respect_to_stop_1=None,
                                                          time_with_respect_to_stop_2=None,
                                                          time_with_respect_to_last_stop=None,
                                                          n_seconds_before_crossing_boundary=2,
                                                          n_seconds_after_crossing_boundary=None,
                                                          max_cluster_distance=50,
                                                          max_distance_to_ref_point=400,
                                                          max_distance_from_ref_point_to_missed_target=50,
                                                          max_time_since_last_vis=3,
                                                          columns_to_sort_alt_ff_by=(
                                                              'abs_curv_diff', 'time_since_last_vis'),
                                                          window_for_curv_of_traj=(
                                                              -25, 25),
                                                          curv_of_traj_mode='distance',
                                                          truncate_curv_of_traj_by_time_of_capture=False,
                                                          last_seen_and_next_seen_attributes_to_add=(
            'ff_distance', 'ff_angle', 'ff_angle_boundary',
            'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y'
                                                          ),
                                                          curv_of_traj_data_df_exists_ok=True,
                                                          **kwargs,
                                                          ):

        self.get_monkey_data(
            already_retrieved_ok=True)
        self.make_miss_events_df(add_one_stop_info=add_one_stop_info)

        # --- Align times ---
        self.set_point_of_eval(
            time_with_respect_to_stop_1=time_with_respect_to_stop_1,
            time_with_respect_to_stop_2=time_with_respect_to_stop_2,
            time_with_respect_to_last_stop=time_with_respect_to_last_stop,
        )
        self.time_of_eval = self.miss_events_df['time_of_eval']
        self.max_distance_from_ref_point_to_missed_target = max_distance_from_ref_point_to_missed_target

        # --- Filter and curvature computation ---
        self.eliminate_crossing_boundary_cases(
            n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
            n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary,
        )

        self.make_or_retrieve_curv_of_traj_df(
            exists_ok=curv_of_traj_data_df_exists_ok,
            curv_of_traj_mode=curv_of_traj_mode,
            window_for_curv_of_traj=window_for_curv_of_traj,
            truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
        )
        self.add_curv_of_traj_info_to_monkey_information(column_exists_ok=True)

        # --- Firefly-level info ---
        ff_sub = self.ff_dataframe.loc[
            self.ff_dataframe['point_index'].isin(
                self.miss_events_df['point_index_of_eval'].values)
        ]

        self.curvature_df = curvature_utils.make_curvature_df(
            ff_sub, self.curv_of_traj_df)
        self.add_curvature_info_to_ff_dataframe()

        # --- rsw info generation ---
        self.make_current_and_alternative_ff_info(
            max_cluster_distance=max_cluster_distance,
            max_time_since_last_vis=max_time_since_last_vis,
            columns_to_sort_alt_ff_by=columns_to_sort_alt_ff_by,
            last_seen_and_next_seen_attributes_to_add=last_seen_and_next_seen_attributes_to_add,
            max_distance_to_ref_point=max_distance_to_ref_point,
        )

    # ==================================================================
    # Streamlined pipelines
    # ==================================================================

    def streamline_process_to_collect_info_from_one_session(
        self,
        miss_events_info_exists_ok=True,
        add_one_stop_info=True,
        curv_of_traj_data_df_exists_ok=True,
        update_point_index=True,
        save_data=True,
    ):
        """Main method to streamline miss events info collection from one session."""
        try:
            if not all([miss_events_info_exists_ok, curv_of_traj_data_df_exists_ok]):
                raise Exception(
                    'One or more miss events info flags are False; regenerating miss events info.')
            self._try_retrieve_all_miss_events_info_from_one_session()

        except Exception as e:
            print(f'Abort retrieving miss events info: {e}')
            print('Proceeding to collect all miss events info from this session...')
            self.streamline_making_current_and_alternative_ff_info(
                add_one_stop_info=add_one_stop_info,
                curv_of_traj_data_df_exists_ok=curv_of_traj_data_df_exists_ok, **self.gc_kwargs
            )
            self.collect_trajectory_data_to_be_combined_across_sessions(
                **self.gc_kwargs)
            if save_data:
                self._save_important_info()

        self._compile_important_info()
        if update_point_index:
            self._update_point_index_of_important_df_in_important_info()

        return self.important_info

    # ==================================================================
    # Core Processing
    # ==================================================================
    def _try_retrieve_all_miss_events_info_from_one_session(self, df_names=None):
        """Attempt to retrieve saved rsw dataframes from disk."""
        if df_names is None:
            df_names = [
                'miss_event_alt_ff',
                'miss_event_cur_ff',
                'miss_events_df',
                'traj_data_df',
            ]
        for df_name in df_names:
            df_path = os.path.join(
                self.miss_events_folder_path, f'{df_name}.csv')
            df = pd.read_csv(df_path).drop(
                columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
            setattr(self, df_name, df)

    # ==================================================================
    # Utility / Save / Update
    # ==================================================================

    def _compile_important_info(self):
        """Bundle key session DataFrames into a dictionary."""
        self.important_info = copy.deepcopy({
            'miss_events_df': self.miss_events_df,
            'miss_event_alt_ff': self.miss_event_alt_ff,
            'miss_event_cur_ff': self.miss_event_cur_ff,
            'traj_data_df': self.traj_data_df,
        })
        print('Compiled important_info with keys:', self.important_info.keys())

        try:
            self.all_traj_feature_names = trajectory_utils.make_all_traj_feature_names(
                self.gc_kwargs['time_range_of_trajectory'],
                self.gc_kwargs['num_time_points_for_trajectory'],
                self.gc_kwargs['time_range_of_trajectory_to_plot'],
                self.gc_kwargs['num_time_points_for_trajectory_to_plot'],
                traj_point_features=self.gc_kwargs['trajectory_features'],
            )
            self.relevant_curv_of_traj_df = self.traj_data_df[self.all_traj_feature_names['relevant_curv_of_traj']].copy(
            )
            self.relevant_curv_of_traj_df['point_index'] = self.traj_data_df['point_index']

        except AttributeError:
            pass

    def _save_important_info(self, df_names=None):
        """Save key session DataFrames to disk."""
        if df_names is None:
            df_names = [
                'miss_events_df',
                'miss_event_alt_ff',
                'miss_event_cur_ff',
                'traj_data_df',
            ]
        os.makedirs(self.miss_events_folder_path, exist_ok=True)
        for df in df_names:
            getattr(self, df).to_csv(os.path.join(
                self.miss_events_folder_path, f'{df}.csv'), index=False)

    def _update_point_index_of_important_df_in_important_info(self):
        """Reassign continuous point indices for combined DataFrames."""
        self.important_info, self.point_index_map = (
            get_missed_ff_data.assign_new_point_index_to_combine_across_sessions(
                self.important_info, self.new_point_index_start
            )
        )

    def make_current_and_alternative_ff_info(
        self,
        max_cluster_distance=50,
        max_time_since_last_vis=3,
        max_distance_to_ref_point=400,
        ff_priority_criterion='abs_curv_diff',
        columns_to_sort_alt_ff_by=('abs_curv_diff', 'time_since_last_vis'),
        last_seen_and_next_seen_attributes_to_add=(
            'ff_distance', 'ff_angle', 'ff_angle_boundary',
            'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y'
        ),
    ):
        """
        Find and enrich current and alternative firefly information for RSW feature collection.

        This method:
        1. Identifies current and alternative fireflies for missed-to-switch events.
        2. Adds curvature and spatial features to each firefly.
        3. Assigns firefly numbers for consistent referencing across point indices.

        Parameters
        ----------
        max_cluster_distance : float, optional
            Maximum spatial distance (cm) for clustering missed fireflies.
        max_time_since_last_vis : float, optional
            Maximum time (s) since the last visibility of a firefly.
        max_distance_to_ref_point : float, optional
            Maximum distance (cm) between the stop position and firefly location.
        ff_priority_criterion : str, optional
            Column used for prioritizing fireflies during sorting.
        columns_to_sort_alt_ff_by : tuple of str, optional
            Columns used to sort next fireflies (alternative targets) by priority.
        last_seen_and_next_seen_attributes_to_add : tuple of str, optional
            Feature names to add to both current and alternative firefly DataFrames.

        Returns
        -------
        None
            Updates the following attributes:
            - self.miss_event_cur_ff
            - self.miss_event_alt_ff
            - self.point_index_all
            - self.time_all
        """
        print('[INFO] Generating current and alternative firefly information...')
        self.last_seen_and_next_seen_attributes_to_add = last_seen_and_next_seen_attributes_to_add
        os.makedirs(self.miss_events_folder_path, exist_ok=True)

        # Step 1: Identify current and alternative fireflies
        self.miss_event_cur_ff, self.miss_event_alt_ff = get_missed_ff_data.find_current_and_alternative_ff_info(
            self.miss_events_df,
            self.ff_real_position_sorted,
            self.ff_life_sorted,
            self.ff_dataframe,
            self.monkey_information,
            max_time_since_last_vis=max_time_since_last_vis,
            max_cluster_distance=max_cluster_distance,
            max_distance_to_ref_point=max_distance_to_ref_point,
            columns_to_sort_alt_ff_by=columns_to_sort_alt_ff_by,
        )

        # Step 2: Enrich both with trajectory and curvature features
        for label, df_attr in [('current', 'miss_event_cur_ff'), ('alternative', 'miss_event_alt_ff')]:
            print(f'[INFO] Adding features to {label} fireflies...')
            ff_df = getattr(self, df_attr)
            ff_df = get_missed_ff_data.add_features_to_miss_event_ff_info(
                ff_df,
                self.ff_dataframe,
                self.monkey_information,
                self.ff_real_position_sorted,
                self.ff_caught_T_new,
                self.curv_of_traj_df,
                self.curvature_df,
                last_seen_and_next_seen_attributes_to_add=last_seen_and_next_seen_attributes_to_add,
                ff_priority_criterion=ff_priority_criterion,
            )
            setattr(self, df_attr, ff_df)

        # Step 3: Extract and align trial indices and times
        self.point_index_all = self.miss_event_cur_ff.point_index.unique()
        self.time_all = self.monkey_information.loc[self.point_index_all, 'time'].values

        # Step 4: Sort and assign firefly numbering
        for df_attr, offset in [('miss_event_cur_ff', 1), ('miss_event_alt_ff', 101)]:
            ff_df = getattr(self, df_attr)
            ff_df.sort_values(
                by=['point_index', ff_priority_criterion], inplace=True)
            ff_df['ff_number'] = ff_df.groupby(
                'point_index').cumcount() + offset
            setattr(self, df_attr, ff_df)

        print('[INFO] Firefly data successfully processed.')

    def add_curvature_info_to_ff_dataframe(self, column_exists_ok=False):
        """Add curvature difference columns to ff_dataframe."""
        if ('curv_diff' not in self.ff_dataframe.columns) or (not column_exists_ok):
            curv_sub = self.curvature_df[[
                'ff_index', 'point_index', 'curv_diff']]
            self.ff_dataframe = pd.merge(self.ff_dataframe, curv_sub, on=[
                                         'ff_index', 'point_index'], how='left')

            na_mask = self.ff_dataframe['curv_diff'].isna()
            self.ff_dataframe.loc[na_mask, 'curv_diff'] = (
                np.random.choice([-1, 1], size=na_mask.sum()) * 0.6
            )
            self.ff_dataframe['abs_curv_diff'] = self.ff_dataframe['curv_diff'].abs(
            )

    def add_curv_of_traj_info_to_monkey_information(self, column_exists_ok=False):
        """Add curvature of trajectory (and absolute curvature) to monkey_information."""
        if ('curv_of_traj' not in self.monkey_information.columns) or (not column_exists_ok):
            curv_sub = self.curv_of_traj_df[['point_index', 'curv_of_traj']]
            self.monkey_information = pd.merge(
                self.monkey_information, curv_sub, on='point_index', how='left')
            self.monkey_information['abs_curv_of_traj'] = self.monkey_information['curv_of_traj'].abs(
            )

            if self.monkey_information['curv_of_traj'].isna().any():
                print(
                    '[WARN] Found NaNs in curv_of_traj after merge; filling with 0.')
                self.monkey_information['curv_of_traj'] = self.monkey_information['curv_of_traj'].fillna(
                    0)

    def make_or_retrieve_curv_of_traj_df(
        self,
        exists_ok=True,
        window_for_curv_of_traj=(-25, 0),
        curv_of_traj_mode='distance',
        truncate_curv_of_traj_by_time_of_capture=False,
    ):
        """Load or compute curvature of trajectory dataframe."""
        filepath = os.path.join(
            self.miss_events_folder_path, 'curv_of_traj_df.csv')

        if exists(filepath) and exists_ok:
            self.curv_of_traj_df = pd.read_csv(filepath).drop(
                columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
            self.curv_of_traj_df = self.curv_of_traj_df[[
                'point_index', 'curv_of_traj']]
            print('[INFO] Retrieved existing curv_of_traj_df')
        else:
            self.curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
                window_for_curv_of_traj,
                self.monkey_information,
                self.ff_caught_T_new,
                curv_of_traj_mode=curv_of_traj_mode,
                truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
            )
            self.curv_of_traj_df = self.curv_of_traj_df[[
                'point_index', 'curv_of_traj']]
            self.curv_of_traj_df.to_csv(filepath, index=False)
            print(
                f'[INFO] Generated and saved new curv_of_traj_df at {filepath}')

    def set_point_of_eval(self, time_with_respect_to_stop_1=None,
                          time_with_respect_to_stop_2=None,
                          time_with_respect_to_last_stop=None,
                          ):

        self.miss_events_df = get_missed_ff_data.set_point_of_eval(
            self.miss_events_df,
            self.monkey_information,
            time_with_respect_to_stop_1=time_with_respect_to_stop_1,
            time_with_respect_to_stop_2=time_with_respect_to_stop_2,
            time_with_respect_to_last_stop=time_with_respect_to_last_stop,
        )

    def make_miss_events_df(self, add_one_stop_info=False):
        if not hasattr(self, 'rcap_events_df'):
            self.get_retry_capture_info()
            self.furnish_rcap_events_df()

        self.rcap_events_df['total_stop_time'] = self.rcap_events_df['last_stop_time'] - \
            self.rcap_events_df['stop_1_time']
        self.rcap_events_df['candidate_target'] = self.rcap_events_df['cur_ff_index'].fillna(
            0)
        self.rcap_events_df['target_index'] = self.rcap_events_df['trial'].astype(
            int)

        if not hasattr(self, 'miss_to_switch_df'):
            self._get_miss_to_switch_df(add_one_stop_info=add_one_stop_info)

        self.miss_events_df = pd.concat(
            [self.rcap_events_df, self.miss_to_switch_df], axis=0).reset_index(drop=True)

    def _get_miss_to_switch_df(self, add_one_stop_info=False):
        """Combine rsw and one-stop data if requested."""
        if add_one_stop_info:
            self.make_one_stop_w_ff_df()
            if getattr(self, 'rsw_w_ff_df', None) is None:
                self.get_retry_switch_info()
            self.miss_to_switch_df = detect_rsw_and_rcap.combine_rsw_and_one_stop_to_make_miss_to_switch_df(
                self.rsw_w_ff_df, self.one_stop_w_ff_df)

        else:
            self.miss_to_switch_df = self.rsw_w_ff_df.copy()

        self.miss_to_switch_df['total_stop_time'] = self.miss_to_switch_df['last_stop_time'] - \
            self.miss_to_switch_df['stop_1_time']

    def get_free_selection_x(self, num_ff_per_row=5, select_every_nth_row=1, add_arc_info=False, arc_info_to_add=['opt_arc_curv', 'curv_diff'],
                             curvature_df=None, curv_of_traj_df=None, **kwargs):
        '''
        free_selection_x_df: df, the input features for machine learning
        free_selection_labels: array, the labels for machine learning
        cases_for_inspection: dict, the cases that are chosen for inspection
        chosen_rows_of_df: df, the rows of free_selection_df that are chosen for inspection
        sequence_of_obs_ff_indices: list, the sequence of observed ff indices
        '''

        self.num_ff_per_row = num_ff_per_row

        if select_every_nth_row > 1:
            self.free_selection_df_sample = self.free_selection_df.iloc[::select_every_nth_row]
        else:
            self.free_selection_df_sample = self.free_selection_df.copy()

        self.free_selection_x_df, self.y_value, self.cases_for_inspection, self.chosen_rows_of_df, self.sequence_of_obs_ff_indices, self.free_selection_x_df_for_plotting = free_selection.organize_free_selection_x(self.free_selection_df_sample, self.ff_dataframe, self.ff_real_position_sorted, self.monkey_information, ff_caught_T_new=self.ff_caught_T_new,
                                                                                                                                                                                                                     only_select_n_ff_case=None, num_ff_per_row=num_ff_per_row, add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df, **kwargs)
        self.free_selection_x_df_for_plotting = self.free_selection_x_df_for_plotting.drop(
            columns=['ff_angle_boundary'], errors='ignore')
        self.free_selection_time = self.chosen_rows_of_df.time.values
        self.free_selection_point_index = self.chosen_rows_of_df.starting_point_index.values.astype(
            int)
        self.cases_for_inspection_obs = self.cases_for_inspection['cases_for_inspection_obs']
        self.non_chosen_rows_of_df = self.cases_for_inspection['non_chosen_rows_of_df']
