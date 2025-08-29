from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils
from planning_analysis.plan_factors import plan_factors_helper_class, build_factor_comp_utils, build_factor_comp
from planning_analysis.only_cur_ff import only_cur_ff_utils, features_to_keep_utils
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, cvn_from_ref_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp
from null_behaviors import curv_of_traj_utils
from decision_making_analysis.decision_making import decision_making_class

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class HelperGUATavsTAFTclass(decision_making_class.DecisionMaking):

    def __init__(self, raw_data_folder_path=None):
        super().__init__(raw_data_folder_path=raw_data_folder_path, retrieve_monkey_data=False)

    def _get_stops_near_ff_df(self,
                              already_made_ok=True):
        # if already_made_ok & (getattr(self, 'stops_near_ff_df', None) is not None):
        #     return

        if already_made_ok & (getattr(self, 'stops_near_ff_df', None) is not None):
            return

        self.stops_near_ff_df = self.TAFT_df.copy() if (
            self.GUAT_or_TAFT == 'TAFT') else self.GUAT_df.copy()

        self.stops_near_ff_df.rename(
            columns={'ff_index': 'cur_ff_index'}, inplace=True)
        self.stops_near_ff_df = find_cvn_utils._add_basic_monkey_info(
            self.stops_near_ff_df, self.monkey_information)

        self.stops_near_ff_df.rename(columns={'ff_index': 'cur_ff_index',
                                              'monkey_angle': 'stop_monkey_angle', }, inplace=True)

        # note that this only works for TAFT, but doesn't work for GUAT
        self._add_nxt_ff_index()
        self.stops_near_ff_df['nxt_ff_caught_time'] = self.ff_caught_T_new[self.stops_near_ff_df['nxt_ff_index'].values]

        # add the next stop time and point index
        closest_stop_to_capture_df2 = self.closest_stop_to_capture_df[[
            'cur_ff_index', 'stop_point_index', 'stop_time']].copy()
        closest_stop_to_capture_df2.rename(columns={'cur_ff_index': 'nxt_ff_index',
                                                    'stop_time': 'next_stop_time',
                                                    'stop_point_index': 'next_stop_point_index'}, inplace=True)

        self.stops_near_ff_df = self.stops_near_ff_df.merge(
            closest_stop_to_capture_df2, on='nxt_ff_index', how='left')

        self.stops_near_ff_df = find_cvn_utils._add_ff_xy(
            self.stops_near_ff_df, self.ff_real_position_sorted)

        self._add_nxt_ff_info()

        find_cvn_utils.add_cur_ff_cluster_50_size(self.stops_near_ff_df, self.ff_real_position_sorted, self.ff_life_sorted,
                                                  empty_cluster_ok=True)

        # The below is not needed rn since it will filter out rows based on visibility of cur or nxt ff
        # self.stops_near_ff_df = find_cvn_utils.process_shared_stops_near_ff_df(self.stops_near_ff_df)

        self.stops_near_ff_df = find_cvn_utils.add_monkey_info_before_stop(
            self.monkey_information, self.stops_near_ff_df)
        self._add_curv_of_traj_stat_df()

        self.stops_near_ff_df = find_cvn_utils._add_distance_info(
            self.stops_near_ff_df, self.monkey_information, self.ff_real_position_sorted)

        self.stops_near_ff_df['data_category_by_vis'] = 'test'

    def _add_nxt_ff_info(self):
        self.stops_near_ff_df = nxt_ff_utils.add_nxt_ff_first_and_last_seen_info(
            self.stops_near_ff_df, self.ff_dataframe_visible, self.monkey_information, self.ff_real_position_sorted, self.ff_life_sorted)

        self.stops_near_ff_df = nxt_ff_utils.add_if_nxt_ff_and_nxt_ff_cluster_flash_bbas(self.stops_near_ff_df, self.ff_real_position_sorted,
                                                                                         self.ff_flash_sorted, self.ff_life_sorted, stop_period_duration=self.stop_period_duration)
        self.stops_near_ff_df = nxt_ff_utils.add_if_nxt_ff_and_nxt_ff_cluster_flash_bsans(self.stops_near_ff_df, self.ff_real_position_sorted,
                                                                                          self.ff_flash_sorted, self.ff_life_sorted)

        self.stops_near_ff_df = nxt_ff_utils._add_stop_or_nxt_ff_first_seen_and_last_seen_info_bbas(
            self.stops_near_ff_df, self.ff_dataframe_visible, self.monkey_information, cur_or_nxt='cur')

    def _make_plan_features_df(self):
        self._make_plan_features_step_1()
        self._make_plan_features_step_2(
            list_of_cur_ff_cluster_radius=[100],
            list_of_nxt_ff_cluster_radius=[200]
        )
        self.plan_features_df = plan_factors_utils.merge_plan_features1_and_plan_features2(self.plan_features1, self.plan_features2)

    def _make_plan_features_step_1(self, window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False):
        test_or_ctrl = 'test'

        self._make_heading_info_df_for_GUAT_vs_TAFT()
        setattr(self, f'{test_or_ctrl}_heading_info_df', self.heading_info_df)
        cvn_from_ref_class.CurVsNxtFfFromRefClass._make_curv_of_traj_df_if_not_already_made(
            self, window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        plan_factors_helper_class.PlanFactorsHelpClass._make_curv_of_traj_df_w_one_sided_window_if_not_already_made(
            self, window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode)

        self.plan_features1 = plan_factors_utils.make_plan_features1(
            self.heading_info_df, self.curv_of_traj_df, self.curv_of_traj_df_w_one_sided_window)

        # time_columns = ['NXT_time_ff_last_seen_bbas',
        #         'NXT_time_ff_last_seen_bsans',
        #         'nxt_ff_last_flash_time_bbas',
        #         'nxt_ff_last_flash_time_bsans',
        #         'nxt_ff_cluster_last_seen_time_bbas',
        #         'nxt_ff_cluster_last_seen_time_bsans',
        #         'nxt_ff_cluster_last_flash_time_bbas',
        #         'nxt_ff_cluster_last_flash_time_bsans']

        # for col in time_columns:
        #     self.plan_features1[f'{col}_rel_to_stop'] = self.plan_features1[col] - self.plan_features1['stop_time']

        return

    def _make_plan_features_step_2(self, use_eye_data=True, use_speed_data=True, ff_radius=10,
                                   list_of_cur_ff_cluster_radius=[
                                       100, 200, 300],
                                   list_of_nxt_ff_cluster_radius=[
                                       100, 200, 300],
                                   ):

        self._get_stops_near_ff_df(already_made_ok=True)

        self.both_ff_at_ref_df = self.get_both_ff_at_ref_df()
        self.both_ff_at_ref_df['stop_point_index'] = self.nxt_ff_df_from_ref['stop_point_index'].values
        
        if getattr(self, 'ff_dataframe', None) is None:
            cvn_from_ref_class.CurVsNxtFfFromRefClass.get_more_monkey_data(
                self)

        if getattr(self, 'nxt_ff_df', None) is None:
            self._get_stops_near_ff_df(already_made_ok=True)
            self.nxt_ff_df, self.cur_ff_df = nxt_ff_utils.get_nxt_ff_df_and_cur_ff_df(
                self.stops_near_ff_df)
            self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref = cvn_from_ref_class.CurVsNxtFfFromRefClass.find_nxt_ff_df_and_cur_ff_df_from_ref(
                self, self.ref_point_value, self.ref_point_mode)

        self.plan_features2 = plan_factors_utils.make_plan_features2(self.stops_near_ff_df, self.heading_info_df, self.both_ff_at_ref_df, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted,
                                                                     stop_period_duration=self.stop_period_duration, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value, ff_radius=ff_radius,
                                                                     list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                                     use_speed_data=use_speed_data, use_eye_data=use_eye_data,
                                                                     guarantee_cur_ff_info_for_cluster=True,
                                                                     guarantee_nxt_ff_info_for_cluster=True,
                                                                     flash_or_vis=None,
                                                                     )

    def _make_ff_info_at_start_df(self):
        df = self.TAFT_df2 if (self.GUAT_or_TAFT == 'TAFT') else self.GUAT_df2
        if len(self.stops_near_ff_df) != len(df):
            raise ValueError(
                'The length of stops_near_ff_df and df must be the same')

        self.monkey_info_in_all_stop_periods = only_cur_ff_utils.make_monkey_info_in_all_stop_periods(self.stops_near_ff_df, self.monkey_information, stop_period_duration=self.stop_period_duration,
                                                                                                      all_end_time=self.stops_near_ff_df[
                                                                                                          'next_stop_time'], all_start_time=self.stops_near_ff_df['beginning_time']
                                                                                                      )
        self.ff_info_at_start_df, self.cur_ff_info_at_start_df = only_cur_ff_utils.find_ff_info_and_cur_ff_info_at_start_df(df, self.monkey_info_in_all_stop_periods, self.ff_flash_sorted,
                                                                                                                            self.ff_real_position_sorted, self.ff_life_sorted, ff_radius=10,
                                                                                                                            guarantee_info_for_cur_ff=True, dropna=False)

    def _get_x_features_df(self,
                           list_of_cur_ff_cluster_radius=[100],
                           list_of_cur_ff_ang_cluster_radius=[20],
                           list_of_start_dist_cluster_radius=[100],
                           list_of_start_ang_cluster_radius=[20],
                           list_of_flash_cluster_period=[[1.5, 2.0]]
                           ):
        self._make_ff_info_at_start_df()

        # then I can get x_features!
        self.x_features_df, self.all_cluster_names = only_cur_ff_utils.get_x_features_df(self.ff_info_at_start_df, self.cur_ff_info_at_start_df,
                                                                                         list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius,
                                                                                         list_of_cur_ff_ang_cluster_radius=list_of_cur_ff_ang_cluster_radius,
                                                                                         list_of_start_dist_cluster_radius=list_of_start_dist_cluster_radius,
                                                                                         list_of_start_ang_cluster_radius=list_of_start_ang_cluster_radius,
                                                                                         list_of_flash_cluster_period=list_of_flash_cluster_period,
                                                                                         flash_or_vis=None,
                                                                                         )

    def _make_only_cur_ff_df(self):
        self.only_cur_ff_df = only_cur_ff_utils.get_only_cur_ff_df(self.stops_near_ff_df, self.ff_real_position_sorted, self.ff_caught_T_new, self.monkey_information,
                                                                   self.curv_of_traj_df, self.ff_dataframe_visible, stop_period_duration=self.stop_period_duration,
                                                                   ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value)

    def get_both_ff_at_ref_df(self):
        self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref = cvn_from_ref_class.CurVsNxtFfFromRefClass.find_nxt_ff_df_and_cur_ff_df_from_ref(
                self, self.ref_point_value, self.ref_point_mode)
        self.both_ff_at_ref_df = self.nxt_ff_df_from_ref[[
            'ff_distance', 'ff_angle']].copy()
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'nxt_ff_distance_at_ref',
                                               'ff_angle': 'nxt_ff_angle_at_ref'}, inplace=True)
        self.both_ff_at_ref_df = pd.concat([self.both_ff_at_ref_df.reset_index(drop=True), self.cur_ff_df_from_ref[['ff_distance', 'ff_angle',
                                                                                                                    'ff_angle_boundary']].reset_index(drop=True)], axis=1)
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'cur_ff_distance_at_ref',
                                               'ff_angle': 'cur_ff_angle_at_ref',
                                               'ff_angle_boundary': 'cur_ff_angle_boundary_at_ref'}, inplace=True)

        # self.both_ff_at_ref_df = self.heading_info_df[['nxt_ff_distance_at_ref', 'nxt_ff_angle_at_ref',
        #                                             'cur_ff_distance_at_ref', 'cur_ff_angle_at_ref',
        #                                             'cur_ff_angle_boundary_at_ref']].copy()
        return self.both_ff_at_ref_df

    def _make_heading_info_df_for_GUAT_vs_TAFT(self):

        self._get_stops_near_ff_df(already_made_ok=True)
        self._get_nxt_ff_and_cur_ff_info_based_on_ref_point_for_GUAT_vs_TAFT()
        self.cur_and_nxt_ff_from_ref_df = show_planning_utils.make_cur_and_nxt_ff_from_ref_df(
            self.nxt_ff_df_final, self.cur_ff_df_final)

        self.heading_info_df = show_planning_utils.make_heading_info_df(
            self.cur_and_nxt_ff_from_ref_df, self.stops_near_ff_df_modified, self.monkey_information, self.ff_real_position_sorted)



    def _add_nxt_ff_index(self):
        if self.GUAT_or_TAFT == 'TAFT':
            self.stops_near_ff_df['nxt_ff_index'] = self.stops_near_ff_df['cur_ff_index'] + 1
        else:
            self.stops_near_ff_df['nxt_ff_index'] = np.searchsorted(
                self.ff_caught_T_new, self.stops_near_ff_df['stop_time'].values)

            # check if the nxt_ff_index is correct (can delete the lines below later)
            self.stops_near_ff_df['nxt_ff_caught_time'] = self.ff_caught_T_new[self.stops_near_ff_df['nxt_ff_index'].values]
            # see if any element of self.stops_near_ff_df['nxt_ff_caught_time'] - self.stops_near_ff_df['stop_time'] is smaller than 0
            # if there is, then the nxt_ff_index is not correct
            if np.any(self.stops_near_ff_df['nxt_ff_caught_time'] - self.stops_near_ff_df['stop_time'] < 0):
                raise ValueError('nxt_ff_index is not correct')

    def _add_curv_of_traj_stat_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 0]):
        if self.curv_of_traj_df is None:
            self.curv_of_traj_df, self.traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new,
                                                                                                                            curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)
        self.curv_of_traj_stat_df = build_factor_comp.find_curv_of_traj_stat_df(
            self.stops_near_ff_df, self.curv_of_traj_df)
        self.stops_near_ff_df = build_factor_comp_utils._add_stat_columns_to_df(
            self.curv_of_traj_stat_df, self.stops_near_ff_df, ['curv'], 'stop_point_index')

    def _get_nxt_ff_and_cur_ff_info_based_on_ref_point_for_GUAT_vs_TAFT(self):
        self.stops_near_ff_df, self.nxt_ff_df, self.cur_ff_df = nxt_ff_utils.get_nxt_ff_df_and_cur_ff_df(
            self.stops_near_ff_df)
        self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref = cvn_from_ref_class.CurVsNxtFfFromRefClass.find_nxt_ff_df_and_cur_ff_df_from_ref(
                self, self.ref_point_value, self.ref_point_mode)
        self.nxt_ff_df_modified = self.nxt_ff_df_from_ref.copy()
        self.cur_ff_df_modified = self.cur_ff_df_from_ref.copy()

        self.stop_point_index_modified = self.nxt_ff_df_modified.stop_point_index.values.copy()
        self.stops_near_ff_df_modified = self.stops_near_ff_df.copy()

        self.nxt_ff_df_from_ref = self.nxt_ff_df_from_ref.merge(
            self.curv_of_traj_df[['point_index', 'curv_of_traj']], on='point_index', how='left')
        self.cur_ff_df_from_ref = self.cur_ff_df_from_ref.merge(
            self.curv_of_traj_df[['point_index', 'curv_of_traj']], on='point_index', how='left')

        self.nxt_ff_df_final = self.nxt_ff_df_from_ref.copy()
        self.cur_ff_df_final = self.cur_ff_df_from_ref.copy()

        def _merge_and_compute_heading(df):
            # Merge stops_near_ff_df info (stop_point_index, monkey_angle_before_stop)
            df = df.merge(
                self.stops_near_ff_df[[
                    'stop_point_index', 'monkey_angle_before_stop']],
                how='left'
            )
            # Calculate heading difference and confine angle
            df['d_heading_of_traj'] = df['monkey_angle_before_stop'] - \
                df['monkey_angle']
            df['d_heading_of_traj'] = find_cvn_utils.confine_angle_to_within_one_pie(
                df['d_heading_of_traj'].values
            )
            return df

        # Apply to both final DataFrames
        self.nxt_ff_df_final = _merge_and_compute_heading(self.nxt_ff_df_final)
        self.cur_ff_df_final = _merge_and_compute_heading(self.cur_ff_df_final)


    def _get_TAFT_df(self):
        self.TAFT_df = GUAT_vs_TAFT_utils.process_trials_df(
            self.TAFT_trials_df, self.monkey_information, self.ff_dataframe, self.ff_real_position_sorted, self.stop_period_duration)
        self.TAFT_df['cur_ff_capture_time'] = self.ff_caught_T_new[self.TAFT_df['ff_index'].values]

    def _get_GUAT_df(self):
        self.GUAT_df = GUAT_vs_TAFT_utils.process_trials_df(
            self.GUAT_w_ff_df, self.monkey_information, self.ff_dataframe, self.ff_real_position_sorted, self.stop_period_duration)

    def _get_TAFT_df2_based_on_ref_point(self):
        self.TAFT_df2 = GUAT_vs_TAFT_utils.further_make_trials_df(
            self.TAFT_df, self.monkey_information, self.ff_real_position_sorted, self.stop_period_duration, self.ref_point_mode, self.ref_point_value)

    def _get_GUAT_df2_based_on_ref_point(self):
        self.GUAT_df2 = GUAT_vs_TAFT_utils.further_make_trials_df(
            self.GUAT_df, self.monkey_information, self.ff_real_position_sorted, self.stop_period_duration, self.ref_point_mode, self.ref_point_value)

    def _get_GUAT_or_TAFT_x_df(self, save_data=True):
        self.GUAT_or_TAFT_x_df = GUAT_vs_TAFT_utils.combine_relevant_features(
            self.x_features_df, self.only_cur_ff_df, self.plan_features_df)

        # add num_stops
        trials_df = self.TAFT_trials_df if self.GUAT_or_TAFT == 'TAFT' else self.GUAT_w_ff_df
        self.GUAT_or_TAFT_x_df = self.GUAT_or_TAFT_x_df.merge(
            trials_df[['stop_point_index', 'num_stops']], on='stop_point_index', how='left')

        # also clean out unnecesary columns especially in clusters
        new_df_columns = features_to_keep_utils.get_minimal_features_to_keep(
            self.GUAT_or_TAFT_x_df, for_classification=False, deal_with_combd_min_ff_distance=True)
        self.GUAT_or_TAFT_x_df = self.GUAT_or_TAFT_x_df[new_df_columns].copy()

        if self.GUAT_or_TAFT == 'TAFT':
            self.TAFT_x_df = self.GUAT_or_TAFT_x_df.copy()
            if save_data:
                self.TAFT_x_df.to_csv(os.path.join(
                    self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'), index=False)
        elif self.GUAT_or_TAFT == 'GUAT':
            self.GUAT_x_df = self.GUAT_or_TAFT_x_df.copy()
            if save_data:
                self.GUAT_x_df.to_csv(os.path.join(
                    self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'), index=False)
        else:
            raise ValueError('GUAT_or_TAFT must be either TAFT or GUAT')


gc_kwargs = {'time_with_respect_to_first_stop': -0.1,
             'time_with_respect_to_second_stop': None,
             'time_with_respect_to_last_stop': None,
             'n_seconds_before_crossing_boundary': 2.5,
             'n_seconds_after_crossing_boundary': 2.5,

             # max distance from other ff to the current ff to be considered as in the same cluster
             'max_cluster_distance': 50,
             'max_distance_to_stop': 400,
             'max_distance_to_stop_for_GUAT_target': 50,
             # originally using ['abs_curv_diff', 'time_since_last_vis'],
             'columns_to_sort_nxt_ff_by': ['distance_to_monkey'],
             'selection_criterion_if_too_many_ff': 'distance_to_monkey',

             'num_old_ff_per_row': 2,  # originally it was 2
             'num_new_ff_per_row': 2,  # originally it was 2

             'last_seen_and_next_seen_attributes_to_add': ['ff_distance', 'ff_angle', 'ff_angle_boundary', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y'],

             'curv_of_traj_mode': 'time',
             'window_for_curv_of_traj': [-1, 0],
             'truncate_curv_of_traj_by_time_of_capture': False,

             'time_range_of_trajectory': [-2.5, 0],  # original [-2, 0]
             'num_time_points_for_trajectory': 11,  # originally 8
             'time_range_of_trajectory_to_plot': [0, 10],  # original [-2, 5]
             'num_time_points_for_trajectory_to_plot': 41,
             'trajectory_features': ['monkey_distance', 'monkey_angle_to_origin', 'time', 'curv_of_traj'],

             'include_ff_in_near_future': True,
             'max_time_since_last_vis': 2.5,
             'duration_into_future': 0.5,
             }
