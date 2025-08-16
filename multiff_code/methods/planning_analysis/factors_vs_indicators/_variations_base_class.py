
from planning_analysis.plan_factors import test_vs_control_utils, test_vs_control_utils
from planning_analysis.factors_vs_indicators import make_variations_utils, _predict_y_values_class, _compare_y_values_class, _plot_variations_class
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp
import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class _VariationsBase(_predict_y_values_class._PredictYValues,
                      _compare_y_values_class._CompareYValues,
                      _plot_variations_class._PlotVariations):

    # class _VariationsBase:

    x_columns = ['time_when_cur_ff_last_seen_rel_to_stop',
                 'left_eye_cur_ff_time_perc',
                 'right_eye_cur_ff_time_perc',
                 'left_eye_cur_ff_time_perc_10',
                 'right_eye_cur_ff_time_perc_10',
                 'LDy_Q1',
                 'LDy_median',
                 'LDy_Q3',
                 'LDz_Q1',
                 'LDz_median',
                 'LDz_Q3',
                 'RDy_Q1',
                 'RDy_median',
                 'RDy_Q3',
                 'RDz_Q1',
                 'monkey_speed_Q1',
                 'monkey_speed_median',
                 'monkey_speed_Q3',
                 'monkey_dw_Q1',
                 'monkey_dw_median',
                 'monkey_dw_Q3',
                 # 'cur_ff_angle_when_cur_ff_last_seen',
                 # 'cur_ff_distance_when_cur_ff_last_seen',
                 # 'traj_curv_when_cur_ff_last_seen',
                 ]

    cur_ff_cluster_columns = ['cur_ff_cluster_100_size',
                              'cur_ff_cluster_100_EARLIEST_APPEAR_ff_angle',
                              'cur_ff_cluster_100_EARLIEST_APPEAR_latest_vis_time',
                              'cur_ff_cluster_100_EARLIEST_APPEAR_visible_duration_after_stop',
                              'cur_ff_cluster_100_EARLIEST_APPEAR_visible_duration_before_stop',
                              'cur_ff_cluster_100_LAST_DISP_earliest_vis_time',
                              'cur_ff_cluster_100_LAST_DISP_ff_angle',
                              'cur_ff_cluster_100_LAST_DISP_visible_duration_after_stop',
                              'cur_ff_cluster_100_LAST_DISP_visible_duration_before_stop',
                              'cur_ff_cluster_100_LONGEST_VIS_earliest_vis_time',
                              'cur_ff_cluster_100_LONGEST_VIS_ff_angle',
                              'cur_ff_cluster_100_LONGEST_VIS_latest_vis_time',
                              'cur_ff_cluster_100_LONGEST_VIS_visible_duration_after_stop',
                              'cur_ff_cluster_100_LONGEST_VIS_visible_duration_before_stop',
                              'cur_ff_cluster_100_combd_min_ff_angle',
                              'cur_ff_cluster_100_combd_max_ff_angle',
                              'cur_ff_cluster_100_combd_median_ff_angle',
                              'cur_ff_cluster_100_combd_median_ff_distance',
                              'cur_ff_cluster_100_combd_earliest_vis_time',
                              'cur_ff_cluster_100_combd_latest_vis_time',
                              'cur_ff_cluster_100_combd_visible_duration',
                              'cur_ff_cluster_100_combd_earliest_vis_time_after_stop',
                              'cur_ff_cluster_100_combd_latest_vis_time_before_stop',
                              # 'cur_ff_cluster_100_EARLIEST_APPEAR_earliest_vis_time',
                              # 'cur_ff_cluster_100_LAST_DISP_latest_vis_time',
                              ]

    curv_columns = ['ref_curv_of_traj',
                    'curv_mean',
                    'curv_std',
                    'curv_min',
                    'curv_Q1',
                    'curv_median',
                    'curv_Q3',
                    'curv_max']

    def __init__(self, opt_arc_type='opt_arc_stop_closest'):
        self.opt_arc_type = opt_arc_type

        # # Bind methods from _PredictYValues
        # for name, method in _predict_y_values_class._PredictYValues.__dict__.items():
        #     if callable(method):
        #         setattr(self, name, MethodType(method, self))

        # # Bind methods from _CompareYValues
        # for name, method in _compare_y_values_class._CompareYValues.__dict__.items():
        #     if callable(method):
        #         setattr(self, name, MethodType(method, self))

        # # Bind methods from _PlotVariations
        # for name, method in _plot_variations_class._PlotVariations.__dict__.items():
        #     if callable(method):
        #         setattr(self, name, MethodType(method, self))

    def make_key_paths(self):
        self.cur_and_nxt_data_comparison_path = os.path.join(
            self.combd_cur_and_nxt_folder_path, 'data_comparison')
        self.all_perc_info_path = os.path.join(
            self.cur_and_nxt_data_comparison_path, f'{self.opt_arc_type}/all_perc_info.csv')
        self.all_median_info_folder_path = os.path.join(
            self.cur_and_nxt_data_comparison_path, f'{self.opt_arc_type}/all_median_info')
        self.overall_median_info_path = os.path.join(
            self.cur_and_nxt_data_comparison_path, f'{self.opt_arc_type}/overall_median_info.csv')
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)

        self.cur_and_nxt_lr_df_path = os.path.join(
            self.combd_cur_and_nxt_folder_path, f'ml_results/lr_variations/{self.opt_arc_type}/all_cur_and_nxt_lr_df.csv')
        self.cur_and_nxt_lr_pred_ff_df_path = os.path.join(
            self.combd_cur_and_nxt_folder_path, f'ml_results/lr_variations/{self.opt_arc_type}/all_cur_and_nxt_lr_pred_ff_df.csv')
        os.makedirs(os.path.dirname(
            self.cur_and_nxt_lr_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(
            self.cur_and_nxt_lr_pred_ff_df_path), exist_ok=True)

    # note that the method below is only used for monkey; for agent, the method is defined in the agent class
    def get_test_and_ctrl_heading_info_df_across_sessions(self, ref_point_mode='distance', ref_point_value=-150,
                                                          curv_traj_window_before_stop=[
                                                              -25, 0],
                                                          heading_info_df_exists_ok=True, combd_heading_df_x_sessions_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                   opt_arc_type=self.opt_arc_type)
        self.test_heading_info_df, self.ctrl_heading_info_df = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value,
                                                                                                                                               curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                                                               combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                                                                                                                                               show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                                                                               stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)

    def make_or_retrieve_all_cur_and_nxt_lr_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.cur_and_nxt_lr_df_path
        if exists_ok:
            if exists(df_path):
                self.all_cur_and_nxt_lr_df = pd.read_csv(df_path)
                print('Successfully retrieved all_cur_and_nxt_lr_df from ', df_path)
                return self.all_cur_and_nxt_lr_df
            else:
                print('all_cur_and_nxt_lr_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_cur_and_nxt_lr_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_cur_and_nxt_lr_df,
                                                                                                      ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                      monkey_name=self.monkey_name,
                                                                                                      path_to_save=df_path,
                                                                                                      )
        return self.all_cur_and_nxt_lr_df

    def quickly_get_plan_features_control_and_test_data(self, ref_point_mode, ref_point_value, to_predict_ff=False, keep_monkey_info=False, for_classification=False):
        self.get_plan_features_df_across_sessions(
            ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
        self.process_combd_plan_features()
        self.plan_features_test, self.plan_features_ctrl = plan_factors_utils.make_plan_features_test_and_plan_features_ctrl(
            self.combd_plan_features_tc)
        return

    def _make_cur_and_nxt_variations_df(self, ref_point_mode, ref_point_value,
                                        agg_regrouped_info_func,
                                        agg_regrouped_info_kwargs={},
                                        to_predict_ff=False,
                                        keep_monkey_info_choices=[True],
                                        make_regrouped_info_kwargs={}):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value

        df = pd.DataFrame()
        for keep_monkey_info in keep_monkey_info_choices:

            print('keep_monkey_info:', keep_monkey_info)
            self.quickly_get_plan_features_control_and_test_data(ref_point_mode, ref_point_value, to_predict_ff=to_predict_ff,
                                                                 keep_monkey_info=keep_monkey_info)
            print('Have successfully run get_plan_features_df_across_sessions.')

            temp_df = make_variations_utils.make_regrouped_info(self.plan_features_test,
                                                                self.plan_features_ctrl,
                                                                agg_regrouped_info_func,
                                                                agg_regrouped_info_kwargs=agg_regrouped_info_kwargs,
                                                                **make_regrouped_info_kwargs)

            temp_df['keep_monkey_info'] = keep_monkey_info
            df = pd.concat([df, temp_df], axis=0)
        df.reset_index(drop=True, inplace=True)
        return df

    def separate_plan_features_test_and_plan_features_ctrl(self):
        self.plan_features_test = self.plan_features_test[self.combd_plan_features_tc.columns].copy(
        )
        self.plan_features_ctrl = self.plan_features_ctrl[self.combd_plan_features_tc.columns].copy(
        )

    def process_both_heading_info_df(self):
        self.test_heading_info_df = build_factor_comp.process_heading_info_df(
            self.test_heading_info_df)
        self.ctrl_heading_info_df = build_factor_comp.process_heading_info_df(
            self.ctrl_heading_info_df)

    def filter_both_heading_info_df(self, **kwargs):
        self.test_heading_info_df, self.ctrl_heading_info_df = test_vs_control_utils.filter_both_df(
            self.test_heading_info_df, self.ctrl_heading_info_df, **kwargs)

    def process_combd_plan_features(self):
        self.combd_plan_features_tc = test_vs_control_utils.process_combd_plan_features(
            self.combd_plan_features_tc, curv_columns=self.curv_columns)
        self.ref_columns = [column for column in self.combd_plan_features_tc.columns if (
            'ref' in column) & ('cur_ff' in column)]
        # note that it will include d_heading_of_traj

        # drop columns with NA in self.combd_plan_features_tc and print them
        columns_with_null_info = self.combd_plan_features_tc.isnull().sum(
            axis=0)[self.combd_plan_features_tc.isnull().sum(axis=0) > 0]
        if len(columns_with_null_info) > 0:
            print('Columns with nulls are dropped:')
            print(columns_with_null_info)
        self.combd_plan_features_tc.dropna(axis=1, inplace=True)

    def _use_a_method_on_test_and_ctrl_data_data_respectively(self,
                                                              plan_features_test,
                                                              plan_features_ctrl,
                                                              method,
                                                              method_kwargs={}):
        self.plan_features_test = plan_features_test.copy()
        self.plan_features_ctrl = plan_features_ctrl.copy()
        regrouped_info = pd.DataFrame()

        for test_or_control in ['control', 'test']:
            print('test_or_control:', test_or_control)
            self.test_or_control = test_or_control

            df = method(self, **method_kwargs)

            df['test_or_control'] = test_or_control
            if test_or_control == 'control':
                print('control')
            regrouped_info = pd.concat([regrouped_info, df], axis=0)
        return regrouped_info
