from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils
from null_behaviors import curv_of_traj_utils

from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, cvn_from_ref_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, test_vs_control_utils
from null_behaviors import curvature_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from data_wrangling import base_processing_class
import pandas as pd
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# note, one class instance is either for test or control, but not both


class PlanFactorsHelpClass(cvn_from_ref_class.CurVsNxtFfFromRefClass):

    def __init__(self, test_or_control, raw_data_folder_path, curv_of_traj_mode='distance',
                 window_for_curv_of_traj=[-25, 0],
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 opt_arc_type='opt_arc_stop_closest',
                 ):
        super().__init__(opt_arc_type=opt_arc_type,
                         raw_data_folder_path=None)

        # if test_or_control is not 'test' or 'control', raise an error
        if test_or_control not in ['test', 'control']:
            raise ValueError('test_or_control must be either test or control')

        self.test_or_control = test_or_control
        self.test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'

        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj

        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
            self, raw_data_folder_path)

    def _make_plan_x(self, stops_near_ff_df_exists_ok=True, save_stops_near_ff_df=True, use_eye_data=True,
                     use_speed_data=True, stop_period_duration=2, ff_radius=10,
                     list_of_cur_ff_cluster_radius=[100, 200, 300],
                     list_of_nxt_ff_cluster_radius=[100, 200, 300]):

        if getattr(self, 'monkey_dataframe', None) is None:
            self.load_raw_data(self.raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=self.curv_of_traj_mode,
                               window_for_curv_of_traj=self.window_for_curv_of_traj)

        self.make_stops_near_ff_and_ff_comparison_dfs(test_or_control=self.test_or_control,
                                                      exists_ok=stops_near_ff_df_exists_ok, save_data=save_stops_near_ff_df)

        self.both_ff_at_ref_df = self.get_both_ff_at_ref_df()
        self.both_ff_at_ref_df['stop_point_index'] = self.nxt_ff_df_from_ref['stop_point_index']

        if self.ff_dataframe is None:
            self.get_more_monkey_data()

        if not hasattr(self, 'heading_info_df'):
            self.make_heading_info_df_without_long_process(test_or_control=self.test_or_control, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                           curv_traj_window_before_stop=self.curv_traj_window_before_stop, use_curv_to_ff_center=self.use_curv_to_ff_center)

        plan_x = plan_factors_utils.make_plan_x_df(self.stops_near_ff_df, self.heading_info_df, self.both_ff_at_ref_df, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted,
                                                   stop_period_duration=stop_period_duration, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value, ff_radius=ff_radius,
                                                   list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                   use_speed_data=use_speed_data, use_eye_data=use_eye_data)

        return plan_x

    def _make_plan_y(self, heading_info_df_exists_ok=False, stops_near_ff_df_exists_ok=False, save_data=True):

        if getattr(self, 'monkey_dataframe', None) is None:
            self.load_raw_data(self.raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=self.curv_of_traj_mode,
                               window_for_curv_of_traj=self.window_for_curv_of_traj)

        self.make_heading_info_df_without_long_process(
            test_or_control=self.test_or_control, ref_point_mode=self.ref_point_mode,
            curv_traj_window_before_stop=self.curv_traj_window_before_stop,
            ref_point_value=self.ref_point_value, use_curv_to_ff_center=self.use_curv_to_ff_center,
            heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            save_data=save_data
        )
        setattr(self, f'{self.test_or_ctrl}_heading_info_df',
                self.heading_info_df)
        self._make_curv_of_traj_df_if_not_already_made()
        self._make_curv_of_traj_df_w_one_sided_window_if_not_already_made()

        # prepare to add monkey_angle_when_cur_ff_first_seen
        self.cur_ff_df_modified = self.cur_ff_df_modified.merge(self.cur_ff_df[[
                                                                'stop_point_index', 'point_index_ff_first_seen']], on='stop_point_index', how='left').sort_values(by='stop_point_index')
        self.cur_ff_df_temp = find_cvn_utils.find_ff_info(self.cur_ff_df_modified.ff_index.values, self.cur_ff_df_modified['point_index_ff_first_seen'].values,
                                                          self.monkey_information, self.ff_real_position_sorted)

        plan_y = plan_factors_utils.make_plan_y_df(
            self.heading_info_df, self.curv_of_traj_df, self.curv_of_traj_df_w_one_sided_window)

        plan_y = build_factor_comp.add_d_monkey_angle(
            plan_y, self.cur_ff_df_temp, self.stops_near_ff_df)

        return plan_y

    def _get_file_names_for_plan_x_or_y(self, plan_type):
        df_name = find_cvn_utils.find_diff_in_curv_df_name(
            ref_point_mode=self.ref_point_mode,
            ref_point_value=self.ref_point_value,
            curv_traj_window_before_stop=self.curv_traj_window_before_stop
        )

        partial_path = self.plan_x_partial_path if plan_type == 'plan_x' else self.plan_y_partial_path
        folder_name = os.path.join(
            self.planning_data_folder_path,
            partial_path,
            self.test_or_control
        )
        os.makedirs(folder_name, exist_ok=True)

        csv_path = os.path.join(folder_name, df_name)

        return df_name, csv_path

    def make_plan_x(self, exists_ok=True, already_made_ok=True, save_data=True, **make_plan_func_kwargs):
        df_name, csv_path = self._get_file_names_for_plan_x_or_y('plan_x')
        attr_name = f'plan_x_{self.test_or_ctrl}'

        if already_made_ok & (getattr(self, attr_name, None) is not None):
            return getattr(self, attr_name)

        if exists_ok and os.path.exists(csv_path):
            plan_data = pd.read_csv(csv_path).reset_index(drop=True)
            print(f'Successfully retrieved {attr_name} ({df_name})')
        else:
            print(f'Making new: {attr_name} ({df_name})')
            plan_data = self._make_plan_x(**make_plan_func_kwargs)
            if save_data:
                plan_data.to_csv(csv_path, index=False)
                print(f'Made {attr_name} and saved to {csv_path}')
        setattr(self, attr_name, plan_data)
        return plan_data

    def make_plan_y(self, exists_ok=True, already_made_ok=True, save_data=True, **make_plan_func_kwargs):

        df_name, csv_path = self._get_file_names_for_plan_x_or_y('plan_y')
        attr_name = f'plan_y_{self.test_or_ctrl}'

        if already_made_ok & (getattr(self, attr_name, None) is not None):
            return getattr(self, attr_name)

        if exists_ok and os.path.exists(csv_path):
            plan_data = pd.read_csv(csv_path).reset_index(drop=True)
            print(f'Successfully retrieved {attr_name} ({df_name})')
        else:
            print(f'Making new: {attr_name} ({df_name})')
            plan_data = self._make_plan_y(**make_plan_func_kwargs, save_data=save_data)
            if save_data:
                plan_data.to_csv(csv_path, index=False)
                print(f'Made {attr_name} and saved to {csv_path}')
        setattr(self, attr_name, plan_data)
        return plan_data

    def _make_curv_of_traj_df_w_one_sided_window_if_not_already_made(self, window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance'):
        # One-sided window: Unlike the regular curv_of_traj_df which uses a symmetric window (e.g., [-25, 25] cm), this uses an asymmetric window that only looks backward from the current point (e.g., [-25, 0] cm).
        if getattr(self, 'curv_of_traj_df_w_one_sided_window', None) is None:
            self.curv_of_traj_df_w_one_sided_window, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new,
                                                                                                                            curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)
