import sys
from data_wrangling import base_processing_class
from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, build_factor_comp_utils
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs, plotly_plot_class
import numpy as np
import pandas as pd
import os
import sys


class _FindCurVsNxtFF(base_processing_class.BaseProcessing):

    def __init__(self,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
                 opt_arc_type='opt_arc_stop_closest',
                 ):
        super().__init__()
        self._update_opt_arc_type_and_related_paths(
            opt_arc_type=opt_arc_type)
        self.default_monkey_plot_params = plotly_plot_class.PlotlyPlotter.default_monkey_plot_params

    def make_stops_near_ff_and_ff_comparison_dfs(self, test_or_control='test', exists_ok=True, save_data=True):
        self.test_or_control = test_or_control
        if test_or_control == 'test':
            self.make_stops_near_ff_and_ff_comparison_dfs_test(
                exists_ok=exists_ok, save_data=save_data)
        elif test_or_control == 'control':
            self.make_stops_near_ff_and_ff_comparison_dfs_ctrl(
                exists_ok=exists_ok, save_data=save_data)
        else:
            raise ValueError(
                'test_or_control should be either test or control')

    def make_stops_near_ff_and_ff_comparison_dfs_test(self, shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        self._make_stops_near_ff_and_ff_comparison_dfs_test_or_ctrl(
            test_or_control='test', shared_stops_near_ff_df_already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)

    def make_stops_near_ff_and_ff_comparison_dfs_ctrl(self, shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        self._make_stops_near_ff_and_ff_comparison_dfs_test_or_ctrl(
            test_or_control='control', shared_stops_near_ff_df_already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)

    def retrieve_shared_stops_near_ff_df(self):
        try:
            self.shared_stops_near_ff_df = pd.read_csv(os.path.join(
                self.planning_data_folder_path, 'shared_stops_near_ff_df.csv')).drop(["Unnamed: 0"], axis=1)
            self.shared_stops_near_ff_df = find_cvn_utils.process_shared_stops_near_ff_df(
                self.shared_stops_near_ff_df)
            print('Retrieving shared_stops_near_ff_df succeeded')
            successful_retrieval = True
        except Exception as e:
            print(
                'Failed to retrieve shared_stops_near_ff_df; will make new shared_stops_near_ff_df')
            successful_retrieval = False
        return successful_retrieval

    def _make_shared_stops_near_ff_df_if_not_already_made(self, remove_cases_where_monkey_too_close_to_edge=False,
                                                          max_distance_between_cur_and_nxt_ff=500, min_distance_between_cur_and_nxt_ff=25,
                                                          stop_period_duration=2,
                                                          already_made_ok=True, exists_ok=True, save_data=True):

        if already_made_ok & (getattr(self, 'shared_stops_near_ff_df', None) is not None):
            return

        if exists_ok:
            successful_retrieval = self.retrieve_shared_stops_near_ff_df()
            if successful_retrieval:
                return

        if self.monkey_information is None:
            self.load_raw_data(self.raw_data_folder_path, monkey_data_exists_ok=True,
                               curv_of_traj_mode=self.curv_of_traj_params['curv_of_traj_mode'],
                               window_for_curv_of_traj=self.curv_of_traj_params[
                                   'window_for_curv_of_traj'],
                               truncate_curv_of_traj_by_time_of_capture=self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'])

        if self.ff_dataframe is None:
            self.get_more_monkey_data()

        self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible'] == 1].copy(
        )
        self.shared_stops_near_ff_df, self.all_nxt_ff_df = find_cvn_utils.make_shared_stops_near_ff_df_and_all_nxt_ff_df(self.monkey_information, self.ff_dataframe_visible, self.closest_stop_to_capture_df, self.ff_real_position_sorted,
                                                                                                                         self.ff_caught_T_new, self.ff_flash_sorted, self.ff_life_sorted, min_distance_between_cur_and_nxt_ff=min_distance_between_cur_and_nxt_ff,
                                                                                                                         max_distance_between_cur_and_nxt_ff=max_distance_between_cur_and_nxt_ff, stop_period_duration=stop_period_duration,
                                                                                                                         remove_cases_where_monkey_too_close_to_edge=remove_cases_where_monkey_too_close_to_edge
                                                                                                                         )

        self.shared_stops_near_ff_df = find_cvn_utils.process_shared_stops_near_ff_df(
            self.shared_stops_near_ff_df)
        self._add_curv_of_traj_stat_df()
        if save_data:
            self.shared_stops_near_ff_df.to_csv(os.path.join(
                self.planning_data_folder_path, 'shared_stops_near_ff_df.csv'))
            print(
                f'Stored shared_stops_near_ff_df ({len(self.shared_stops_near_ff_df)} rows) in {os.path.join(self.planning_data_folder_path, "shared_stops_near_ff_df.csv")}')

    def _make_stops_near_ff_and_ff_comparison_dfs_test_or_ctrl(self, test_or_control='test', shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        if (test_or_control != 'test') and (test_or_control != 'control'):
            raise ValueError('test_or_ctrl must be either "test" or "control"')
        test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'

        self._make_shared_stops_near_ff_df_if_not_already_made(
            already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)
        self.stops_near_ff_df = self.shared_stops_near_ff_df[
            self.shared_stops_near_ff_df['data_category_by_vis'] == test_or_control].copy()
        self.stops_near_ff_df.reset_index(drop=True, inplace=True)
        self.stops_near_ff_df, self.nxt_ff_df, self.cur_ff_df = self._make_nxt_ff_df_and_cur_ff_df(
            self.stops_near_ff_df)
        setattr(
            self, f'stops_near_ff_df_{test_or_ctrl}', self.stops_near_ff_df)
        # print(f'Made stops_near_ff_df_test, which has {len(self.stops_near_ff_df_test)} rows')

    def _add_curv_of_traj_stat_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        if self.curv_of_traj_df is None:
            self.curv_of_traj_df = self.get_curv_of_traj_df(window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode,
                                                            truncate_curv_of_traj_by_time_of_capture=False)
        self.curv_of_traj_stat_df = build_factor_comp.find_curv_of_traj_stat_df(
            self.shared_stops_near_ff_df, self.curv_of_traj_df)
        self.shared_stops_near_ff_df = build_factor_comp_utils._add_stat_columns_to_df(
            self.curv_of_traj_stat_df, self.shared_stops_near_ff_df, ['curv'], 'stop_point_index')

    def _make_nxt_ff_df_and_cur_ff_df(self, stops_near_ff_df):
        self.stops_near_ff_df, self.nxt_ff_df, self.cur_ff_df = nxt_ff_utils.get_nxt_ff_df_and_cur_ff_df(
            stops_near_ff_df)
        self.stops_near_ff_df_counted = self.stops_near_ff_df.copy()
        return self.stops_near_ff_df, self.nxt_ff_df, self.cur_ff_df
