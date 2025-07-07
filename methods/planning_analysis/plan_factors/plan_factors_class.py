from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, stops_near_ff_based_on_ref_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, test_vs_control_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import planning_neural_utils
from data_wrangling import base_processing_class, general_utils
from planning_analysis.plan_factors import plan_factors_helper_class
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class PlanFactors(stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef):

    def __init__(self, raw_data_folder_path=None, curv_of_traj_mode='distance',
                 window_for_curv_of_traj=[-25, 25],
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 opt_arc_type='opt_arc_stop_closest',
                 ):
        super().__init__(opt_arc_type=opt_arc_type,
                         raw_data_folder_path=None)

        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.ml_inst = ml_methods_class.MlMethods()

        if raw_data_folder_path is not None:
            # self.load_raw_data(raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=curv_of_traj_mode,
            #                    window_for_curv_of_traj=window_for_curv_of_traj)
            base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
                self, raw_data_folder_path)

    def make_test_and_ctrl_inst(self):
        if not hasattr(self, 'test_inst'):
            self.test_inst = plan_factors_helper_class.PlanFactorsHelpClass(
                'test', self.raw_data_folder_path,
                opt_arc_type=self.opt_arc_type,
                curv_of_traj_mode=self.curv_of_traj_mode,
                window_for_curv_of_traj=self.window_for_curv_of_traj)
        if not hasattr(self, 'ctrl_inst'):
            self.ctrl_inst = plan_factors_helper_class.PlanFactorsHelpClass(
                'control', self.raw_data_folder_path,
                opt_arc_type=self.opt_arc_type,
                curv_of_traj_mode=self.curv_of_traj_mode,
                window_for_curv_of_traj=self.window_for_curv_of_traj)

    def make_plan_x_and_y_for_both_test_and_ctrl(self, already_made_ok=True, plan_x_exists_ok=True, plan_y_exists_ok=True,
                                                 ref_point_mode='time after cur ff visible',
                                                 ref_point_value=0.0, curv_traj_window_before_stop=[-50, 0],
                                                 use_curvature_to_ff_center=False, heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                 use_eye_data=True, save_data=True):

        if already_made_ok:
            if hasattr(self, 'plan_x_test') & hasattr(self, 'plan_x_ctrl') & hasattr(self, 'plan_y_test') & hasattr(self, 'plan_y_ctrl'):
                self.get_plan_x_and_y_tc()
                return

        if plan_x_exists_ok & plan_y_exists_ok:
            try:
                self.retrieve_all_plan_data_for_one_session(
                    ref_point_mode, ref_point_value, curv_traj_window_before_stop)
                self.get_plan_x_and_y_tc()
                return
            except FileNotFoundError:
                pass

        self.make_test_and_ctrl_inst()

        for obj in [self, self.test_inst, self.ctrl_inst]:
            obj.ref_point_mode = ref_point_mode
            obj.ref_point_value = ref_point_value
            obj.curv_traj_window_before_stop = curv_traj_window_before_stop
            obj.use_curvature_to_ff_center = use_curvature_to_ff_center

        self.make_plan_y_test_and_ctrl(
            exists_ok=plan_y_exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
        )

        self.make_plan_x_test_and_ctrl(
            exists_ok=plan_x_exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            use_eye_data=use_eye_data,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
        )

        self.get_plan_x_and_y_tc()

    def retrieve_all_plan_data_for_one_session(self, ref_point_mode, ref_point_value, curv_traj_window_before_stop):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        for plan_type in ['plan_x', 'plan_y']:
            for test_or_ctrl in ['test', 'ctrl']:
                test_or_control = 'test' if test_or_ctrl == 'test' else 'control'
                df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                             curv_traj_window_before_stop=curv_traj_window_before_stop)

                if plan_type == 'plan_x':
                    folder_name = os.path.join(
                        self.planning_data_folder_path, self.plan_x_partial_path, test_or_control)
                elif plan_type == 'plan_y':
                    folder_name = os.path.join(
                        self.planning_data_folder_path, self.plan_y_partial_path, test_or_control)

                csv_path = os.path.join(folder_name, df_name)
                plan_data = pd.read_csv(csv_path).drop(
                    columns=['Unnamed: 0'], errors='ignore').reset_index(drop=True)
                setattr(self, f'{plan_type}_{test_or_ctrl}', plan_data)
                print(f'Retrieved {plan_type}_{test_or_ctrl} from {csv_path}')

    def make_plan_x_test_and_ctrl(self, exists_ok=True, save_data=True,
                                  already_made_ok=True, use_eye_data=True,
                                  stops_near_ff_df_exists_ok=True,
                                  **make_plan_func_kwargs):

        self.make_test_and_ctrl_inst()

        self.plan_x_test = self.test_inst.make_plan_x(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            use_eye_data=use_eye_data,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            **make_plan_func_kwargs
        )
        self.plan_x_ctrl = self.ctrl_inst.make_plan_x(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            use_eye_data=use_eye_data,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            **make_plan_func_kwargs
        )

    def make_plan_y_test_and_ctrl(self, exists_ok=True, already_made_ok=True,
                                  save_data=True,
                                  heading_info_df_exists_ok=True,
                                  stops_near_ff_df_exists_ok=True,
                                  **make_plan_func_kwargs):

        self.make_test_and_ctrl_inst()

        self.plan_y_test = self.test_inst.make_plan_y(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            **make_plan_func_kwargs
        )
        self.plan_y_ctrl = self.ctrl_inst.make_plan_y(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            **make_plan_func_kwargs
        )

    def get_test_and_ctrl_data_from_combd_data(self):
        self.plan_x_test = self.plan_x_tc[self.plan_x_tc['whether_test'] == 1].drop(
            columns='whether_test').copy()
        self.plan_x_ctrl = self.plan_x_tc[self.plan_x_tc['whether_test'] == 0].drop(
            columns='whether_test').copy()
        self.plan_y_test = self.plan_y_tc[self.plan_y_tc['whether_test'] == 1].drop(
            columns='whether_test').copy()
        self.plan_y_ctrl = self.plan_y_tc[self.plan_y_tc['whether_test'] == 0].drop(
            columns='whether_test').copy()

    def get_plan_x_and_y_tc(self):
        self.plan_x_test['whether_test'] = 1
        self.plan_x_ctrl['whether_test'] = 0
        self.plan_y_test['whether_test'] = 1
        self.plan_y_ctrl['whether_test'] = 0
        self.plan_x_tc = pd.concat(
            [self.plan_x_test, self.plan_x_ctrl], ignore_index=True)
        self.plan_y_tc = pd.concat(
            [self.plan_y_test, self.plan_y_ctrl], ignore_index=True)

        build_factor_comp.add_dir_from_cur_ff_same_side(self.plan_y_tc)
        plan_factors_utils.drop_columns_that_contain_both_nxt_and_bbas(
            self.plan_y_tc)
        general_utils.convert_bool_to_int(self.plan_y_tc)

    def change_control_data_to_conform_to_test_data(self):
        self.plan_x_ctrl, self.plan_y_ctrl = test_vs_control_utils.change_control_data_to_conform_to_test_data(
            self.plan_x_ctrl, self.plan_y_ctrl, self.plan_x_test)
        self.get_plan_x_and_y_tc()

    def make_the_distributions_of_angles_more_similar(self):
        self.plan_x_ctrl['unique_id'] = np.arange(len(self.plan_x_ctrl))
        self.plan_y_ctrl['unique_id'] = np.arange(len(self.plan_x_ctrl))

        self.plan_x_ctrl, self.plan_x_test = test_vs_control_utils.make_the_distributions_of_angles_more_similar(
            self.plan_x_ctrl, self.plan_x_test, column_name='cur_ff_angle_at_ref')
        self.plan_y_ctrl = self.plan_y_ctrl[self.plan_y_ctrl['unique_id'].isin(
            self.plan_x_ctrl['unique_id'])].sort_values(by='unique_id').drop(columns=['unique_id'])
        self.plan_x_ctrl = self.plan_x_ctrl.drop(columns=['unique_id'])

        self.get_plan_x_and_y_tc()

    def limit_curv_range(self, max_curv_range=100):
        self.plan_x_tc, self.plan_y_tc = test_vs_control_utils.prune_out_data_with_large_curv_range(
            self.plan_x_tc, self.plan_y_tc, max_curv_range=max_curv_range)
        self.get_test_and_ctrl_data_from_combd_data()

    def limit_cum_distance_between_two_stops(self, max_cum_distance_between_two_stops=400):
        self.plan_x_tc, self.plan_y_tc = test_vs_control_utils.limit_cum_distance_between_two_stops(
            self.plan_x_tc, self.plan_y_tc, max_cum_distance_between_two_stops=max_cum_distance_between_two_stops)
        self.get_test_and_ctrl_data_from_combd_data()

    def make_x_and_y_var_df(self, test_data_only=False, control_data_only=False, drop_na=False, scale_x_var=True, use_pca=False):
        if test_data_only & control_data_only:
            raise ValueError(
                'Both test_data_only and control_data_only are True')
        if test_data_only:
            x_df = self.plan_x_test
            y_df = self.plan_y_test
        elif control_data_only:
            x_df = self.plan_x_ctrl
            y_df = self.plan_y_ctrl
        else:
            x_df = self.plan_x_tc
            y_df = self.plan_y_tc

        for column in ['d_from_cur_ff_to_nxt_ff', 'time_between_two_stops']:
            if column in x_df.columns:
                x_df.drop(columns=[column], inplace=True)
        self.x_var_df, self.y_var_df = prep_ml_data_utils.make_x_and_y_var_df(
            x_df, y_df, drop_na=drop_na, scale_x_var=scale_x_var, use_pca=use_pca)

    def get_x_and_y_for_lr(self, test_or_control='test', scale_x_var=True, use_pca=False):
        if test_or_control == 'test':
            self.make_x_and_y_var_df(
                test_data_only=True, scale_x_var=scale_x_var, use_pca=use_pca)
        elif test_or_control == 'control':
            self.make_x_and_y_var_df(
                control_data_only=True, scale_x_var=scale_x_var, use_pca=use_pca)
        else:
            self.make_x_and_y_var_df(scale_x_var=scale_x_var, use_pca=use_pca)

    def run_lr(self, y_var_column, x_var_df=None, y_var_df=None, test_size=0.2):
        if x_var_df is None:
            x_var_df = self.x_var_df
        if y_var_df is None:
            y_var_df = self.y_var_df
        self.ml_inst.split_and_use_linear_regression(
            x_var_df, y_var_df[y_var_column], test_size=test_size)
        self.summary_df = self.ml_inst.summary_df

    def use_lr_on_all(self, test_or_control='test', y_var_column='d_monkey_angle_since_cur_ff_first_seen2', use_pca=False):
        self.get_x_and_y_for_lr(
            test_or_control=test_or_control, use_pca=use_pca)
        self.run_lr(y_var_column)

    def use_lr_on_specific_x_columns(self, specific_x_columns=None, test_or_control='test', y_var_column='d_monkey_angle_since_cur_ff_first_seen2'):
        if specific_x_columns is None:
            self.specific_x_columns = self.summary_df[self.summary_df['p_value']
                                                      < 0.05].index.values
        else:
            self.specific_x_columns = specific_x_columns
        self.get_x_and_y_for_lr(test_or_control=test_or_control)
        try:
            self.x_var_df = self.x_var_df[self.specific_x_columns].copy()
        except KeyError as e:
            print(e)
            return
        self.run_lr(y_var_column)
