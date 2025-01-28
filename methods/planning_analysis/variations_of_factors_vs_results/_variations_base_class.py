
from planning_analysis.plan_factors import test_vs_control_utils, test_vs_control_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, process_variations_utils, _predict_y_values_class, _compare_y_values_class, _plot_variations_class
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
from planning_analysis import ml_methods_utils, ml_methods_class
import os
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
                      _plot_variations_class._PlotVariations, ):

    x_columns = ['time_when_stop_ff_last_seen_rel_to_stop',
                 'left_eye_stop_ff_time_perc',
                 'right_eye_stop_ff_time_perc',
                 'left_eye_stop_ff_time_perc_10',
                 'right_eye_stop_ff_time_perc_10',
                 'LDy_25%',
                 'LDy_50%',
                 'LDy_75%',
                 'LDz_25%',
                 'LDz_50%',
                 'LDz_75%',
                 'RDy_25%',
                 'RDy_50%',
                 'RDy_75%',
                 'RDz_25%',
                 'monkey_speed_25%',
                 'monkey_speed_50%',
                 'monkey_speed_75%',
                 'monkey_dw_25%',
                 'monkey_dw_50%',
                 'monkey_dw_75%',
                 # 'stop_ff_angle_when_stop_ff_last_seen',
                 # 'stop_ff_distance_when_stop_ff_last_seen',
                 # 'traj_curv_when_stop_ff_last_seen',
                 ]

    stop_ff_cluster_columns = ['stop_ff_cluster_100_size',
                               'stop_ff_cluster_100_EARLIEST_APPEAR_ff_angle',
                               'stop_ff_cluster_100_EARLIEST_APPEAR_latest_vis_time',
                               'stop_ff_cluster_100_EARLIEST_APPEAR_visible_duration_after_stop',
                               'stop_ff_cluster_100_EARLIEST_APPEAR_visible_duration_before_stop',
                               'stop_ff_cluster_100_LAST_DISP_earliest_vis_time',
                               'stop_ff_cluster_100_LAST_DISP_ff_angle',
                               'stop_ff_cluster_100_LAST_DISP_visible_duration_after_stop',
                               'stop_ff_cluster_100_LAST_DISP_visible_duration_before_stop',
                               'stop_ff_cluster_100_LONGEST_VIS_earliest_vis_time',
                               'stop_ff_cluster_100_LONGEST_VIS_ff_angle',
                               'stop_ff_cluster_100_LONGEST_VIS_latest_vis_time',
                               'stop_ff_cluster_100_LONGEST_VIS_visible_duration_after_stop',
                               'stop_ff_cluster_100_LONGEST_VIS_visible_duration_before_stop',
                               'stop_ff_cluster_100_combd_min_ff_angle',
                               'stop_ff_cluster_100_combd_max_ff_angle',
                               'stop_ff_cluster_100_combd_median_ff_angle',
                               'stop_ff_cluster_100_combd_median_ff_distance',
                               'stop_ff_cluster_100_combd_earliest_vis_time',
                               'stop_ff_cluster_100_combd_latest_vis_time',
                               'stop_ff_cluster_100_combd_visible_duration',
                               'stop_ff_cluster_100_combd_earliest_vis_time_after_stop',
                               'stop_ff_cluster_100_combd_latest_vis_time_before_stop',
                               # 'stop_ff_cluster_100_EARLIEST_APPEAR_earliest_vis_time',
                               # 'stop_ff_cluster_100_LAST_DISP_latest_vis_time',
                               ]

    curv_columns = ['ref_curv_of_traj',
                    'curv_mean',
                    'curv_std',
                    'curv_min',
                    'curv_25%',
                    'curv_50%',
                    'curv_75%',
                    'curv_max']

    def __init__(self,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 optimal_arc_type='norm_opt_arc',
                 ):

        self.optimal_arc_type = optimal_arc_type
        self.predict_inst = _predict_y_values_class._PredictYValues()
        self.compare_inst = _compare_y_values_class._CompareYValues()
        self.plot_inst = _plot_variations_class._PlotVariations()
        self.__dict__.update(self.predict_inst.__dict__)
        self.__dict__.update(self.compare_inst.__dict__)
        self.__dict__.update(self.plot_inst.__dict__)

    def make_key_paths(self):
        self.stop_and_alt_data_comparison_path = os.path.join(
            self.combd_stop_and_alt_folder_path, 'data_comparison')
        self.all_perc_info_path = os.path.join(
            self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/all_perc_info.csv')
        self.all_median_info_folder_path = os.path.join(
            self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/all_median_info')
        self.overall_median_info_path = os.path.join(
            self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/overall_median_info.csv')
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)

        self.stop_and_alt_lr_df_path = os.path.join(
            self.combd_stop_and_alt_folder_path, f'ml_results/lr_variations/{self.optimal_arc_type}/all_stop_and_alt_lr_df.csv')
        self.stop_and_alt_lr_pred_ff_df_path = os.path.join(
            self.combd_stop_and_alt_folder_path, f'ml_results/lr_variations/{self.optimal_arc_type}/all_stop_and_alt_lr_pred_ff_df.csv')
        os.makedirs(os.path.dirname(
            self.stop_and_alt_lr_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(
            self.stop_and_alt_lr_pred_ff_df_path), exist_ok=True)

    # note that the method below is only used for monkey; for agent, the method is defined in the agent class
    def get_test_and_ctrl_heading_info_df_across_sessions(self, ref_point_mode='distance', ref_point_value=-150,
                                                          curv_traj_window_before_stop=[
                                                              -50, 0],
                                                          heading_info_df_exists_ok=True, combd_heading_df_x_sessions_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                   optimal_arc_type=self.optimal_arc_type)
        self.test_heading_info_df, self.ctrl_heading_info_df = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value,
                                                                                                                                               curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                                                               combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                                                                                                                                               show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                                                                               stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)

    def make_or_retrieve_all_stop_and_alt_lr_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.stop_and_alt_lr_df_path
        if exists_ok:
            if exists(df_path):
                self.all_stop_and_alt_lr_df = pd.read_csv(df_path)
                print('Successfully retrieved all_stop_and_alt_lr_df from ', df_path)
                return self.all_stop_and_alt_lr_df
            else:
                print('all_stop_and_alt_lr_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_stop_and_alt_lr_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_stop_and_alt_lr_df,
                                                                                                       ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                       monkey_name=self.monkey_name,
                                                                                                       path_to_save=df_path,
                                                                                                       )
        return self.all_stop_and_alt_lr_df

    def quickly_get_plan_x_and_y_control_and_test_data(self, ref_point_mode, ref_point_value, to_predict_ff=False, keep_monkey_info=False, for_classification=False):
        self.get_plan_x_and_plan_y_across_sessions(
            ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
        self.process_combd_plan_x_tc_and_plan_y_tc()
        self.further_process_combd_plan_x_tc(
            to_predict_ff, for_classification=for_classification)
        if keep_monkey_info is False:
            self.combd_plan_x_tc = plan_factors_utils.delete_monkey_info_in_plan_x(
                self.combd_plan_x_tc)
        self.plan_xy_test, self.plan_xy_ctrl = plan_factors_utils.make_plan_xy_test_and_plan_xy_ctrl(
            self.combd_plan_x_tc, self.combd_plan_y_tc)
        return

    def _make_stop_and_alt_variations_df(self, ref_point_mode, ref_point_value,
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
            self.quickly_get_plan_x_and_y_control_and_test_data(ref_point_mode, ref_point_value, to_predict_ff=to_predict_ff,
                                                                keep_monkey_info=keep_monkey_info)
            print('Have successfully run get_plan_x_and_plan_y_across_sessions.')

            temp_df = make_variations_utils.make_regrouped_info(self.plan_xy_test,
                                                                self.plan_xy_ctrl,
                                                                agg_regrouped_info_func,
                                                                agg_regrouped_info_kwargs=agg_regrouped_info_kwargs,
                                                                **make_regrouped_info_kwargs)

            temp_df['keep_monkey_info'] = keep_monkey_info
            df = pd.concat([df, temp_df], axis=0)
        df.reset_index(drop=True, inplace=True)
        return df

    def separate_plan_xy_test_and_plan_xy_ctrl(self):
        self.plan_x_test = self.plan_xy_test[self.combd_plan_x_tc.columns].copy(
        )
        self.plan_y_test = self.plan_xy_test[self.combd_plan_y_tc.columns].copy(
        )
        self.plan_x_ctrl = self.plan_xy_ctrl[self.combd_plan_x_tc.columns].copy(
        )
        self.plan_y_ctrl = self.plan_xy_ctrl[self.combd_plan_y_tc.columns].copy(
        )

    def process_both_heading_info_df(self):
        self.test_heading_info_df = plan_factors_utils.process_heading_info_df(
            self.test_heading_info_df)
        self.ctrl_heading_info_df = plan_factors_utils.process_heading_info_df(
            self.ctrl_heading_info_df)

    def filter_both_heading_info_df(self, **kwargs):
        self.test_heading_info_df, self.ctrl_heading_info_df = test_vs_control_utils.filter_both_df(
            self.test_heading_info_df, self.ctrl_heading_info_df, **kwargs)

    def process_combd_plan_x_tc_and_plan_y_tc(self):
        test_vs_control_utils.process_combd_plan_x_and_y_combd(
            self.combd_plan_x_tc, self.combd_plan_y_tc, curv_columns=self.curv_columns)
        self.ref_columns = [column for column in self.combd_plan_x_tc.columns if (
            'ref' in column) & ('stop_ff' in column)]
        # note that it will include d_heading_of_traj

        # drop columns with NA in self.combd_plan_x_tc and print them
        columns_with_null_info = self.combd_plan_x_tc.isnull().sum(
            axis=0)[self.combd_plan_x_tc.isnull().sum(axis=0) > 0]
        if len(columns_with_null_info) > 0:
            print('Columns with nulls are dropped:')
            print(columns_with_null_info)
        self.combd_plan_x_tc.dropna(axis=1, inplace=True)

        # Also drop the columns that can't be put into x_var
        for column in ['data_name', 'stop_point_index']:
            if column in self.combd_plan_x_tc.columns:
                self.combd_plan_x_tc.drop(columns=[column], inplace=True)

    def further_process_combd_plan_x_tc(self, to_predict_ff, for_classification=False):
        if to_predict_ff:
            self.combd_plan_x_tc = plan_factors_utils.process_plan_x_to_predict_ff_info(
                self.combd_plan_x_tc, self.combd_plan_y_tc)
        else:
            self.combd_plan_x_tc = plan_factors_utils.process_plan_x_to_predict_monkey_info(
                self.combd_plan_x_tc, for_classification=for_classification)

    def _use_a_method_on_test_and_ctrl_data_data_respectively(self,
                                                              plan_xy_test,
                                                              plan_xy_ctrl,
                                                              method,
                                                              method_kwargs={}):
        self.plan_xy_test = plan_xy_test.copy()
        self.plan_xy_ctrl = plan_xy_ctrl.copy()
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
