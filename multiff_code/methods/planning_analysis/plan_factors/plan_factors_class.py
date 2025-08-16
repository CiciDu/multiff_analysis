from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, cvn_from_ref_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, test_vs_control_utils
from null_behaviors import curvature_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from data_wrangling import base_processing_class, general_utils
from planning_analysis.plan_factors import plan_factors_helper_class
import pandas as pd
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class PlanFactors(cvn_from_ref_class.CurVsNxtFfFromRefClass):

    def __init__(self, raw_data_folder_path=None, curv_of_traj_mode='distance',
                 window_for_curv_of_traj=[-25, 0],
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

    def make_plan_features_df_both_test_and_ctrl(self, already_made_ok=True, plan_features_exists_ok=True,
                                                 ref_point_mode='time after cur ff visible',
                                                 ref_point_value=0.0, curv_traj_window_before_stop=[-25, 0],
                                                 use_curv_to_ff_center=False, heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                 use_eye_data=True, save_data=True):

        if already_made_ok:
            if hasattr(self, 'plan_features_test') & hasattr(self, 'plan_features_ctrl'):
                self.get_plan_features_tc()
                return

        if plan_features_exists_ok:
            try:
                self.retrieve_all_plan_data_for_one_session(
                    ref_point_mode, ref_point_value, curv_traj_window_before_stop)
                self.get_plan_features_tc()
                return
            except FileNotFoundError:
                pass

        self.make_test_and_ctrl_inst()

        for obj in [self, self.test_inst, self.ctrl_inst]:
            obj.ref_point_mode = ref_point_mode
            obj.ref_point_value = ref_point_value
            obj.curv_traj_window_before_stop = curv_traj_window_before_stop
            obj.use_curv_to_ff_center = use_curv_to_ff_center

        self.make_plan_features_test_and_ctrl(
            exists_ok=plan_features_exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            use_eye_data=use_eye_data,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
        )

        self.get_plan_features_tc()

    # def make_plan_features_df_test_and_ctrl(self, exists_ok=True, already_made_ok=True, save_data=True, **make_plan_func_kwargs):
    #     self.make_test_and_ctrl_inst()
    #     self.plan_features_test = self.test_inst.make_plan_features_df(
    #         exists_ok=exists_ok,
    #         already_made_ok=already_made_ok,
    #         save_data=save_data,
    #         **make_plan_func_kwargs)
    #     self.plan_features_ctrl = self.ctrl_inst.make_plan_features_df(
    #         exists_ok=exists_ok,
    #         already_made_ok=already_made_ok,
    #         save_data=save_data,
    #         **make_plan_func_kwargs)
    #     self.get_plan_features_tc()

    def retrieve_all_plan_data_for_one_session(self, ref_point_mode, ref_point_value, curv_traj_window_before_stop):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        for test_or_ctrl in ['test', 'ctrl']:
            test_or_control = 'test' if test_or_ctrl == 'test' else 'control'
            df_name = find_cvn_utils.find_diff_in_curv_df_name(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                               curv_traj_window_before_stop=curv_traj_window_before_stop)

            folder_name = os.path.join(
                self.planning_data_folder_path, self.plan_features_partial_path, test_or_control)

            csv_path = os.path.join(folder_name, df_name)
            plan_data = pd.read_csv(csv_path).drop(
                columns=['Unnamed: 0'], errors='ignore').reset_index(drop=True)
            setattr(self, f'plan_features_{test_or_ctrl}', plan_data)
            print(f'Retrieved plan_features_{test_or_ctrl} from {csv_path}')

    def make_plan_features_test_and_ctrl(self, exists_ok=True, already_made_ok=True, save_data=True, **make_plan_func_kwargs):
        self.make_test_and_ctrl_inst()
        self.plan_features_test = self.test_inst.make_plan_features_df(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            **make_plan_func_kwargs)
        self.plan_features_ctrl = self.ctrl_inst.make_plan_features_df(
            exists_ok=exists_ok,
            already_made_ok=already_made_ok,
            save_data=save_data,
            **make_plan_func_kwargs)
        self.get_plan_features_tc()

    def get_test_and_ctrl_data_from_combd_data(self):
        self.plan_features_test = self.plan_features_tc[self.plan_features_tc['whether_test'] == 1].drop(
            columns='whether_test').copy()
        self.plan_features_ctrl = self.plan_features_tc[self.plan_features_tc['whether_test'] == 0].drop(
            columns='whether_test').copy()

    def get_plan_features_tc(self):
        self.plan_features_test['whether_test'] = 1
        self.plan_features_ctrl['whether_test'] = 0
        self.plan_features_tc = pd.concat(
            [self.plan_features_test, self.plan_features_ctrl], ignore_index=True)

        build_factor_comp.add_dir_from_cur_ff_same_side(self.plan_features_tc)
        plan_factors_utils.drop_columns_that_contain_both_nxt_and_bbas(
            self.plan_features_tc)
        general_utils.convert_bool_to_int(self.plan_features_tc)

    def change_control_data_to_conform_to_test_data(self):
        self.plan_features_ctrl = test_vs_control_utils.change_control_data_to_conform_to_test_data(
            self.plan_features_test, self.plan_features_ctrl)
        self.get_plan_features_tc()

    def limit_cum_distance_between_two_stops(self, max_cum_distance_between_two_stops=400):
        self.plan_features_tc = self.plan_features_tc[self.plan_features_tc['cum_distance_between_two_stops']
                                                      <= max_cum_distance_between_two_stops].copy()
        self.get_test_and_ctrl_data_from_combd_data()

    def run_lr(self, y_var_column, x_var_df=None, y_var_df=None, test_size=0.2):
        if x_var_df is None:
            x_var_df = self.x_var_df
        if y_var_df is None:
            y_var_df = self.y_var_df
        self.ml_inst.split_and_use_linear_regression(
            x_var_df, y_var_df[y_var_column], test_size=test_size)
        self.summary_df = self.ml_inst.summary_df

    def use_lr_on_all(self, test_or_control='test', y_var_column='d_monkey_angle_since_cur_ff_first_seen2', use_pca=False, pca_dim=10,
                      to_predict_ff=False, for_classification=False, drop_na_rows=False, drop_na_cols=True, scale_x_var=True,
                      selected_features=None):
        # note: self.pca will be None if use_pca is False
        self.x_var_df, self.y_var_df, self.pca = self.make_x_and_y_var_df(
            test_or_control=test_or_control, use_pca=use_pca, pca_dim=pca_dim, to_predict_ff=to_predict_ff, for_classification=for_classification,
            drop_na_rows=drop_na_rows, drop_na_cols=drop_na_cols, scale_x_var=scale_x_var, selected_features=selected_features)
        self.run_lr(y_var_column)

    def use_lr_on_specific_x_columns(self, specific_x_columns=None, test_or_control='test', y_var_column='d_monkey_angle_since_cur_ff_first_seen2',
                                     to_predict_ff=False, for_classification=False):
        if specific_x_columns is None:
            self.specific_x_columns = self.summary_df[self.summary_df['p_value']
                                                      < 0.05].index.values
        else:
            self.specific_x_columns = specific_x_columns
        # note: self.pca will be None if use_pca is False
        self.x_var_df, self.y_var_df, self.pca = self.make_x_and_y_var_df(test_or_control=test_or_control,
                                                                          to_predict_ff=to_predict_ff, for_classification=for_classification)
        try:
            self.x_var_df = self.x_var_df[self.specific_x_columns].copy()
        except KeyError as e:
            print(e)
            return
        self.run_lr(y_var_column)

    def make_x_and_y_var_df(self, test_or_control='both', drop_na_rows=False, drop_na_cols=True, scale_x_var=True, use_pca=False, pca_dim=10,
                            to_predict_ff=False, for_classification=False, selected_features=None):
        # test_or_control can be 'test', 'control', or 'both'
        pca = None

        if test_or_control == 'test':
            x_df = self.plan_features_test.copy()
            y_df = self.plan_features_test.copy()
        elif test_or_control == 'control':
            x_df = self.plan_features_ctrl.copy()
            y_df = self.plan_features_ctrl.copy()
        elif test_or_control == 'both':
            x_df = self.plan_features_tc.copy()
            y_df = self.plan_features_tc.copy()
        else:
            raise ValueError(
                f'test_or_control must be "test", "control", or "both". Got {test_or_control}')

        if selected_features is None:
            self.selected_features = plan_factors_utils.select_planning_features_for_modeling(x_df,
                                                                                              to_predict_ff=to_predict_ff, for_classification=for_classification)
        else:
            self.selected_features = selected_features

        x_df = x_df[self.selected_features].copy()

        for column in ['d_from_cur_ff_to_nxt_ff', 'time_between_two_stops']:
            if column in x_df.columns:
                x_df.drop(columns=[column], inplace=True)

        # save a copy of x_df
        self.original_x_df = x_df.copy()

        x_df, y_df, pca = prep_ml_data_utils.make_x_and_y_var_df(
            x_df, y_df, drop_na=drop_na_rows, scale_x_var=scale_x_var, use_pca=use_pca, n_components_for_pca=pca_dim)

        # drop columns with NA and print the names of these columns
        columns_with_na = x_df.columns[x_df.isna().any()].tolist()
        if drop_na_cols:
            x_df = x_df.drop(columns=columns_with_na)
            print(
                f'When preparing x_var to predict ff, there are {len(columns_with_na)} columns with NA that are dropped. {x_df.shape[1]} columns are left.')
            print('Columns with NA that are dropped:',
                  np.array(columns_with_na))
        else:
            print(
                f'When preparing x_var to predict ff, there are {len(columns_with_na)} out of {x_df.shape[1]} columns with NA.')

        return x_df, y_df, pca
