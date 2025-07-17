import sys
from data_wrangling import combine_info_utils, base_processing_class
from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils
from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.get_cur_vs_nxt_ff_data import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_class, test_vs_control_utils, monkey_plan_factors_x_sess_class
from planning_analysis.only_cur_ff import only_cur_ff_utils, only_cur_ff_utils, features_to_keep_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from planning_analysis import ml_for_planning_class, ml_for_planning_utils
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
from os.path import exists
import gc
from scipy.stats.mstats import winsorize
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import math


class OnlyStopFF(base_processing_class.BaseProcessing):

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    ref_point_info = {
        'time after cur ff visible': {'min': 0,
                                      'max': 0.3,
                                      'step': 0.1,
                                      'values': None,
                                      'marks': None},
        'distance': {'min': -190,
                     'max': -100,
                     'step': 20,
                     'values': None,
                     'marks': None},
    }

    def __init__(self, monkey_name='monkey_Bruno',
                 raw_data_folder_path=None,
                 opt_arc_type='opt_arc_stop_closest',
                 curv_of_traj_mode='distance',
                 window_for_curv_of_traj=[-25, 25],
                 truncate_curv_of_traj_by_time_of_capture=False):

        self.monkey_information = None
        self.ff_info_at_start_df = None
        self.raw_data_folder_path = raw_data_folder_path
        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.truncate_curv_of_traj_by_time_of_capture = truncate_curv_of_traj_by_time_of_capture
        self.ref_point_mode = None
        self.ref_point_value = None
        self.monkey_name = monkey_name
        if raw_data_folder_path is not None:
            super().__init__()
            self.load_raw_data(raw_data_folder_path=raw_data_folder_path,
                               window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode)

        self.ml_inst = ml_methods_class.MlMethods()
        self._update_opt_arc_type_and_related_paths(
            opt_arc_type=opt_arc_type)

        self.ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode

    def _update_opt_arc_type_and_related_paths(self, opt_arc_type='opt_arc_stop_closest'):
        self.opt_arc_type = opt_arc_type
        self.planning_data_folder_path = self.raw_data_folder_path.replace(
            'raw_monkey_data', 'planning')
        self.only_cur_ff_folder_path = os.path.join(
            self.planning_data_folder_path, f'only_cur_ff/only_cur_ff_df/{self.opt_arc_type}')
        self.x_features_folder_path = os.path.join(
            self.planning_data_folder_path, f'only_cur_ff/x_features_df/{self.opt_arc_type}')
        os.makedirs(self.only_cur_ff_folder_path, exist_ok=True)
        os.makedirs(self.x_features_folder_path, exist_ok=True)

    def make_only_cur_ff_df(self, exists_ok=True, stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150):
        df_name = find_cvn_utils.get_df_name_by_ref(
            self.monkey_name, ref_point_mode, ref_point_value)
        df_path = os.path.join(self.only_cur_ff_folder_path, df_name + '.csv')
        if exists_ok:
            if exists(df_path):
                self.only_cur_ff_df = pd.read_csv(
                    df_path).drop(columns=['Unnamed: 0'])
                print(f'Successfully retrieved only_cur_ff_df from {df_path}')
                return
            else:
                print(
                    f'Failed to retrieve only_cur_ff_df from {df_path}; will make a new one')

        if self.monkey_information is None:
            self.load_raw_data(raw_data_folder_path=self.raw_data_folder_path, window_for_curv_of_traj=self.window_for_curv_of_traj, curv_of_traj_mode=self.curv_of_traj_mode,
                               truncate_curv_of_traj_by_time_of_capture=self.truncate_curv_of_traj_by_time_of_capture)

        self.get_more_monkey_data()
        self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible'] == 1].copy(
        )
        self.only_cur_ff_df = only_cur_ff_utils.get_only_cur_ff_df(self.closest_stop_to_capture_df, self.ff_real_position_sorted, self.ff_caught_T_new, self.monkey_information,
                                                                   self.curv_of_traj_df, self.ff_dataframe_visible, stop_period_duration=stop_period_duration,
                                                                   ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                   opt_arc_type=self.opt_arc_type)
        self.only_cur_ff_df.to_csv(df_path)
        print(f'New only_cur_ff_df was stored in {df_path}.')

    def _make_ff_info_at_start_df(self, stop_period_duration=2):
        self.monkey_info_in_all_stop_periods = only_cur_ff_utils.make_monkey_info_in_all_stop_periods(
            self.closest_stop_to_capture_df, self.monkey_information, stop_period_duration=stop_period_duration)
        self.ff_info_at_start_df, self.cur_ff_info_at_start_df = only_cur_ff_utils.find_ff_info_and_cur_ff_info_at_start_df(self.only_cur_ff_df, self.monkey_info_in_all_stop_periods, self.ff_flash_sorted,
                                                                                                                            self.ff_real_position_sorted, self.ff_life_sorted, ff_radius=10, dropna=True,
                                                                                                                            filter_out_ff_not_in_front_of_monkey_at_ref_point=False)

    def make_x_features_df(self, exists_ok=True, ref_point_mode='distance', ref_point_value=-150,
                           list_of_cur_ff_cluster_radius=[100, 200, 300],
                           list_of_cur_ff_ang_cluster_radius=[20],
                           list_of_start_dist_cluster_radius=[100, 200, 300],
                           list_of_start_ang_cluster_radius=[20],
                           list_of_flash_cluster_period=[
                               [1.0, 1.5], [1.5, 2.0]],
                           ):
        df_name = find_cvn_utils.get_df_name_by_ref(
            self.monkey_name, ref_point_mode, ref_point_value)
        df_path = os.path.join(self.x_features_folder_path, df_name + '.csv')
        if exists_ok:
            if exists(df_path):
                self.x_features_df = pd.read_csv(
                    df_path).drop(columns=['Unnamed: 0'])
                print(f'Successfully retrieved x_features_df from {df_path}')
                return
            else:
                print(
                    f'Failed to retrieve x_features_df from {df_path}; will make a new one')

        if self.monkey_information is None:
            self.load_raw_data(raw_data_folder_path=self.raw_data_folder_path, window_for_curv_of_traj=self.window_for_curv_of_traj,
                               curv_of_traj_mode=self.curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)

        if self.ff_info_at_start_df is None:
            self._make_ff_info_at_start_df()
        self.x_features_df, self.all_cluster_names = only_cur_ff_utils.get_x_features_df(self.ff_info_at_start_df, self.cur_ff_info_at_start_df,
                                                                                         list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius,
                                                                                         list_of_cur_ff_ang_cluster_radius=list_of_cur_ff_ang_cluster_radius,
                                                                                         list_of_start_dist_cluster_radius=list_of_start_dist_cluster_radius,
                                                                                         list_of_start_ang_cluster_radius=list_of_start_ang_cluster_radius,
                                                                                         list_of_flash_cluster_period=list_of_flash_cluster_period,
                                                                                         )

        self.x_features_df_w_all_columns = self.x_features_df.copy()
        self.x_features_df.to_csv(df_path)
        print(f'New x_features_df was stored in {df_path}.')

    def make_x_and_y_var_df(self, scale_x_var=True, use_pca=False, n_components_for_pca=None):
        x_features_df_temp = self.x_features_df.copy()
        for column in ['data_name', 'stop_point_index']:
            if column in x_features_df_temp.columns:
                x_features_df_temp.drop(columns=[column], inplace=True)
        self.original_x_var_df = self.x_var_df.copy()
        self.x_var_df, self.y_var_df = prep_ml_data_utils.make_x_and_y_var_df(x_features_df_temp, self.only_cur_ff_df, scale_x_var=scale_x_var,
                                                                              use_pca=use_pca, n_components_for_pca=n_components_for_pca)

    def use_ref_columns_only(self):
        self.x_features_df = self.x_features_df[[
            column for column in self.x_features_df.columns if 'ref' in column]]

    def prepare_only_cur_ff_data_for_ml(self):

        self.only_cur_ff_df_for_ml = self.only_cur_ff_df.copy()
        self.x_features_df_for_ml = self.x_features_df.copy()

        self._prepare_only_cur_ff_data_for_ml()

    def _prepare_only_cur_ff_data_for_ml(self):
        if 'data_name' not in self.only_cur_ff_df_for_ml.columns:
            # make sure that the two df share the same set of stop_point_index
            shared_stop_periods = set(self.only_cur_ff_df_for_ml['stop_point_index'].unique(
            )).intersection(set(self.x_features_df_for_ml['stop_point_index'].unique()))
            self.only_cur_ff_df_for_ml = self.only_cur_ff_df_for_ml[self.only_cur_ff_df_for_ml['stop_point_index'].isin(
                shared_stop_periods)].reset_index(drop=True)
            self.x_features_df_for_ml = self.x_features_df_for_ml[self.x_features_df_for_ml['stop_point_index'].isin(
                shared_stop_periods)].reset_index(drop=True)
            print('Note: only_cur_ff_df_for_ml and x_features_df_for_ml have been aligned to share the same set of stop_point_index.')
        else:
            # make sure that the two df share the same set of data_name + stop_point_index pairs
            self.only_cur_ff_df_for_ml, self.x_features_df_for_ml = only_cur_ff_utils.keep_same_data_name_and_stop_point_pairs(
                self.only_cur_ff_df_for_ml, self.x_features_df_for_ml)
            print('Note: only_cur_ff_df_for_ml and x_features_df_for_ml have been aligned to share the same set of data_name + stop_point_index pairs.')

        self.x_features_df_w_all_columns = self.x_features_df_for_ml.copy()

        for column in ['data_name', 'stop_point_index']:
            if column in self.x_features_df_for_ml.columns:
                self.x_features_df_for_ml.drop(columns=[column], inplace=True)

        self.only_cur_ff_df_for_ml['dir_from_cur_ff_to_stop'] = (
            (self.only_cur_ff_df_for_ml['dir_from_cur_ff_to_stop'] + 1)/2).astype(int)

    def streamline_preparing_for_ml(self, y_var_column,
                                    ref_columns_only=False,
                                    cluster_to_keep='all',
                                    cluster_for_interaction='none',
                                    add_ref_interaction=True,
                                    winsorize_angle_features=True,
                                    using_lasso=True,
                                    ensure_cur_ff_at_front=True,
                                    use_pca=False,
                                    use_combd_features_for_cluster_only=False,
                                    ):

        self.prepare_only_cur_ff_data_for_ml()

        self.x_var_df, self.y_var_df = ml_for_planning_utils.streamline_preparing_for_ml(self.x_features_df_for_ml,
                                                                                         self.only_cur_ff_df_for_ml,
                                                                                         y_var_column,
                                                                                         ref_columns_only=ref_columns_only,
                                                                                         cluster_to_keep=cluster_to_keep,
                                                                                         cluster_for_interaction=cluster_for_interaction,
                                                                                         add_ref_interaction=add_ref_interaction,
                                                                                         winsorize_angle_features=winsorize_angle_features,
                                                                                         using_lasso=using_lasso,
                                                                                         ensure_cur_ff_at_front=ensure_cur_ff_at_front,
                                                                                         use_pca=use_pca,
                                                                                         use_combd_features_for_cluster_only=use_combd_features_for_cluster_only)
