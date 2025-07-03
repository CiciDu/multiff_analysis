from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, stops_near_ff_based_on_ref_class
from planning_analysis.plan_factors import plan_factors_utils, test_vs_control_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import planning_neural_utils
from data_wrangling import base_processing_class
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class PlanFactors(stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef):

    def __init__(self, raw_data_folder_path=None, curv_of_traj_mode='distance',
                 window_for_curv_of_traj=[-25, 25],
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 optimal_arc_type='norm_opt_arc',
                 ):
        super().__init__(optimal_arc_type=optimal_arc_type,
                         raw_data_folder_path=None)

        self.test_or_control = None
        self.shared_stops_near_ff_df = None
        self.curv_of_traj_df = None
        self.curv_of_traj_df_w_one_sided_window = None
        self.stops_near_ff_df = None
        self.plan_x_test = None
        self.plan_x_ctrl = None
        self.plan_y_test = None
        self.plan_y_ctrl = None
        self.ref_point_mode = None
        self.ref_point_value = None
        self.curv_traj_window_before_stop = None
        self.ff_dataframe = None
        self.curv_of_traj_params = {
            'curv_of_traj_mode': curv_of_traj_mode, 'window_for_curv_of_traj': window_for_curv_of_traj}

        if raw_data_folder_path is not None:
            self.load_raw_data(raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=curv_of_traj_mode,
                               window_for_curv_of_traj=window_for_curv_of_traj)
            base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
                self, raw_data_folder_path)

        self.overall_params['use_curvature_to_ff_center'] = False
        self.ml_inst = ml_methods_class.MlMethods()

    def make_plan_x_and_y_for_both_test_and_ctrl(self, already_made_ok=True, plan_x_exists_ok=True, plan_y_exists_ok=True,
                                                 ref_point_mode='time after cur ff visible',
                                                 ref_point_value=0.0, curv_traj_window_before_stop=[-50, 0],
                                                 use_curvature_to_ff_center=False, heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                 use_eye_data=True, save_data=True):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        self.use_curvature_to_ff_center = use_curvature_to_ff_center

        for test_or_control in ['test', 'control']:
            self._make_plan_x_and_y_test_OR_plan_x_and_y_ctrl(test_or_control=test_or_control, already_made_ok=already_made_ok,
                                                              plan_x_exists_ok=plan_x_exists_ok, plan_y_exists_ok=plan_y_exists_ok,
                                                              heading_info_df_exists_ok=heading_info_df_exists_ok, use_eye_data=use_eye_data,
                                                              stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)
        self._get_plan_x_and_y_combd()

    def _make_plan_x_and_y_test_OR_plan_x_and_y_ctrl(self, test_or_control='test', already_made_ok=True, plan_x_exists_ok=True, plan_y_exists_ok=True, heading_info_df_exists_ok=True,
                                                     use_eye_data=True, stops_near_ff_df_exists_ok=True, save_data=True):
        test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'
        if already_made_ok:
            try:
                plan_x = getattr(self, f'plan_x_{test_or_ctrl}')
                plan_y = getattr(self, f'plan_y_{test_or_ctrl}')
                if (plan_x is not None) & (plan_y is not None):
                    print(f'{test_or_ctrl} plan_x and plan_y already exist')
                    return
            except AttributeError:
                pass
        self._make_plan_y_test_OR_plan_y_ctrl(test_or_control=test_or_control, exists_ok=plan_y_exists_ok,
                                              heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)
        self._make_plan_x_test_OR_plan_x_ctrl(test_or_control=test_or_control, exists_ok=plan_x_exists_ok,
                                              use_eye_data=use_eye_data, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)

    def _make_plan_x_tc_OR_plan_y_tc(self, x_or_y='x'):
        plan_test = getattr(self, f'plan_{x_or_y}_test').reset_index(
            drop=True).copy()
        plan_ctrl = getattr(self, f'plan_{x_or_y}_ctrl').reset_index(
            drop=True).copy()
        plan_test['whether_test'] = 1
        plan_ctrl['whether_test'] = 0
        plan_combd = pd.concat([plan_test, plan_ctrl], axis=0)
        plan_combd.reset_index(drop=True, inplace=True)
        setattr(self, f'plan_{x_or_y}_combd', plan_combd)
        return plan_combd

    def _get_plan_x_and_y_combd(self):
        self.plan_x_tc = self._make_plan_x_tc_OR_plan_y_tc(x_or_y='x')
        self.plan_y_tc = self._make_plan_x_tc_OR_plan_y_tc(x_or_y='y')

        if 'd_monkey_angle_since_cur_ff_first_seen' not in self.plan_y_tc.columns:
            self.plan_y_tc['d_monkey_angle_since_cur_ff_first_seen'] = self.plan_y_tc['d_monkey_angle']
        plan_factors_utils.add_dir_from_cur_ff_same_side(self.plan_y_tc)

    def _prepare_plan_data(self, plan_type, test_or_control, exists_ok, make_plan_func, save_data):
        test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'

        df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                                     curv_traj_window_before_stop=self.curv_traj_window_before_stop)

        if plan_type == 'plan_x':
            folder_name = os.path.join(
                self.planning_data_folder_path, self.plan_x_partial_path, test_or_control)
        elif plan_type == 'plan_y':
            folder_name = os.path.join(
                self.planning_data_folder_path, self.plan_y_partial_path, test_or_control)
        else:
            raise ValueError('plan_type must be either plan_x or plan_y')

        os.makedirs(folder_name, exist_ok=True)
        csv_path = os.path.join(folder_name, df_name)

        if exists_ok & os.path.exists(csv_path):
            plan_data = pd.read_csv(csv_path).reset_index(drop=True)
            setattr(self, f'{plan_type}_{test_or_ctrl}', plan_data)
            if plan_type == 'plan_x':
                self.plan_x_df = plan_data
            print(
                f'Successfully retrieved {plan_type}_{test_or_ctrl} ({df_name})')
        else:
            print(f'Making new: {plan_type}_{test_or_ctrl} ({df_name})')
            plan_data = make_plan_func()
            setattr(self, f'{plan_type}_{test_or_ctrl}', plan_data)

            if save_data:
                plan_data.to_csv(csv_path, index=False)
                print(
                    f'Made {plan_type}_{test_or_ctrl} and saved to {csv_path}')
        return plan_data

    def retrieve_all_plan_data_for_one_session(self):
        for plan_type in ['plan_x', 'plan_y']:
            for test_or_ctrl in ['test', 'ctrl']:
                test_or_control = 'test' if test_or_ctrl == 'test' else 'control'
                df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                                             curv_traj_window_before_stop=self.curv_traj_window_before_stop)

                if plan_type == 'plan_x':
                    folder_name = os.path.join(
                        self.planning_data_folder_path, self.plan_x_partial_path, test_or_control)
                elif plan_type == 'plan_y':
                    folder_name = os.path.join(
                        self.planning_data_folder_path, self.plan_y_partial_path, test_or_control)

                csv_path = os.path.join(folder_name, df_name)
                plan_data = pd.read_csv(csv_path).reset_index(drop=True)
                setattr(self, f'{plan_type}_{test_or_ctrl}', plan_data)

    def _make_plan_x_test_OR_plan_x_ctrl(self, test_or_control='test', exists_ok=True, use_eye_data=True,
                                         stops_near_ff_df_exists_ok=True, save_data=True):
        def make_plan_x():
            self.get_stops_near_ff_df(test_or_control=test_or_control,
                                      exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)
            return self._make_plan_x_df(use_eye_data=use_eye_data, test_or_control=test_or_control)

        self._prepare_plan_data(
            plan_type='plan_x',
            test_or_control=test_or_control,
            exists_ok=exists_ok,
            make_plan_func=make_plan_x,
            save_data=save_data
        )

    def _make_plan_y_test_OR_plan_y_ctrl(self, test_or_control='test', exists_ok=True, heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'

        def make_plan_y():
            self.make_heading_info_df_without_long_process(
                test_or_control=test_or_control, ref_point_mode=self.ref_point_mode,
                curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                ref_point_value=self.ref_point_value, use_curvature_to_ff_center=self.use_curvature_to_ff_center,
                heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                save_data=save_data
            )
            setattr(self, f'{test_or_ctrl}_heading_info_df',
                    self.heading_info_df)
            self._make_curv_of_traj_df_if_not_already_made()
            self._make_curv_of_traj_df_w_one_sided_window_if_not_already_made()

            # prepare to add monkey_angle_when_cur_ff_first_seen
            self.cur_ff_df_modified = self.cur_ff_df_modified.merge(self.cur_ff_df[[
                                                                    'stop_point_index', 'point_index_ff_first_seen']], on='stop_point_index', how='left').sort_values(by='stop_point_index')
            self.cur_ff_df_temp = find_stops_near_ff_utils.find_ff_info(self.cur_ff_df_modified.ff_index.values, self.cur_ff_df_modified['point_index_ff_first_seen'].values,
                                                                        self.monkey_information, self.ff_real_position_sorted)

            plan_y = plan_factors_utils.make_plan_y_df(
                self.heading_info_df, self.curv_of_traj_df, self.curv_of_traj_df_w_one_sided_window)

            plan_y = plan_factors_utils.add_d_monkey_angle(
                plan_y, self.cur_ff_df_temp, self.stops_near_ff_df)

            return plan_y

        self._prepare_plan_data(
            plan_type='plan_y',
            test_or_control=test_or_control,
            exists_ok=exists_ok,
            make_plan_func=make_plan_y,
            save_data=save_data
        )

    def _make_curv_of_traj_df_w_one_sided_window_if_not_already_made(self, window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance'):
        if self.curv_of_traj_df_w_one_sided_window is None:
            self.curv_of_traj_df_w_one_sided_window, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new,
                                                                                                                            curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)

    def get_test_and_ctrl_data_from_combd_data(self):
        self.plan_x_test = self.plan_x_tc[self.plan_x_tc['whether_test'] == 1].drop(
            columns='whether_test').copy()
        self.plan_x_ctrl = self.plan_x_tc[self.plan_x_tc['whether_test'] == 0].drop(
            columns='whether_test').copy()
        self.plan_y_test = self.plan_y_tc[self.plan_y_tc['whether_test'] == 1].drop(
            columns='whether_test').copy()
        self.plan_y_ctrl = self.plan_y_tc[self.plan_y_tc['whether_test'] == 0].drop(
            columns='whether_test').copy()

    def get_combd_data_from_test_and_ctrl_data(self):
        self.plan_x_test['whether_test'] = 1
        self.plan_x_ctrl['whether_test'] = 0
        self.plan_y_test['whether_test'] = 1
        self.plan_y_ctrl['whether_test'] = 0
        self.plan_x_tc = pd.concat(
            [self.plan_x_test, self.plan_x_ctrl], ignore_index=True)
        self.plan_y_tc = pd.concat(
            [self.plan_y_test, self.plan_y_ctrl], ignore_index=True)

    def change_control_data_to_conform_to_test_data(self):
        self.plan_x_ctrl, self.plan_y_ctrl = test_vs_control_utils.change_control_data_to_conform_to_test_data(
            self.plan_x_ctrl, self.plan_y_ctrl, self.plan_x_test)
        self.get_combd_data_from_test_and_ctrl_data()

    def make_the_distributions_of_angles_more_similar(self):
        self.plan_x_ctrl['unique_id'] = np.arange(len(self.plan_x_ctrl))
        self.plan_y_ctrl['unique_id'] = np.arange(len(self.plan_x_ctrl))

        self.plan_x_ctrl, self.plan_x_test = test_vs_control_utils.make_the_distributions_of_angles_more_similar(
            self.plan_x_ctrl, self.plan_x_test, column_name='cur_ff_angle_at_ref')
        self.plan_y_ctrl = self.plan_y_ctrl[self.plan_y_ctrl['unique_id'].isin(
            self.plan_x_ctrl['unique_id'])].sort_values(by='unique_id').drop(columns=['unique_id'])
        self.plan_x_ctrl = self.plan_x_ctrl.drop(columns=['unique_id'])

        self.get_combd_data_from_test_and_ctrl_data()

    def limit_curv_range(self, max_curv_range=100):
        self.plan_x_tc, self.plan_y_tc = test_vs_control_utils.prune_out_data_with_large_curv_range(
            self.plan_x_tc, self.plan_y_tc, max_curv_range=max_curv_range)
        self.get_test_and_ctrl_data_from_combd_data()

    def limit_cum_distance_between_two_stops(self, max_cum_distance_between_two_stops=400):
        self.plan_x_tc, self.plan_y_tc = test_vs_control_utils.limit_cum_distance_between_two_stops(
            self.plan_x_tc, self.plan_y_tc, max_cum_distance_between_two_stops=max_cum_distance_between_two_stops)
        self.get_test_and_ctrl_data_from_combd_data()

    def make_x_and_y_var_df(self, test_data_only=False, control_data_only=False, scale_x_var=True, use_pca=False):
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
            x_df, y_df, scale_x_var=scale_x_var, use_pca=use_pca)

    def get_x_and_y_for_lr(self, test_or_control='test', scale_x_var=True, use_pca=False):
        if test_or_control == 'test':
            self.make_x_and_y_var_df(
                test_data_only=True, scale_x_var=scale_x_var, use_pca=use_pca)
        elif test_or_control == 'control':
            self.make_x_and_y_var_df(
                control_data_only=True, scale_x_var=scale_x_var, use_pca=use_pca)
        else:
            self.make_x_and_y_var_df(scale_x_var=scale_x_var, use_pca=use_pca)

    def _run_lr(self, y_var_column, x_var_df=None, y_var_df=None):
        if x_var_df is None:
            x_var_df = self.x_var_df
        if y_var_df is None:
            y_var_df = self.y_var_df
        self.ml_inst.use_train_test_split(
            x_var_df, y_var_df, y_var_column=y_var_column)
        self.ml_inst.use_linear_regression()
        self.summary_df = self.ml_inst.summary_df

    def use_lr_on_all(self, test_or_control='test', y_var_column='d_monkey_angle2', use_pca=False):
        self.get_x_and_y_for_lr(
            test_or_control=test_or_control, use_pca=use_pca)
        self._run_lr(y_var_column)

    def use_lr_on_specific_x_columns(self, specific_x_columns=None, test_or_control='test', y_var_column='d_monkey_angle2'):
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
        self._run_lr(y_var_column)

    def get_nxt_ff_at_stop_df(self):

        self.nxt_ff_df_temp = find_stops_near_ff_utils.find_ff_info(self.nxt_ff_df_modified.ff_index.values, self.nxt_ff_df_modified['stop_point_index'].values,
                                                                    self.monkey_information, self.ff_real_position_sorted)
        self.nxt_ff_at_stop_df = self.nxt_ff_df_temp[[
            'ff_distance', 'ff_angle']].copy()
        self.nxt_ff_at_stop_df.rename(columns={'ff_distance': 'nxt_ff_distance_at_stop',
                                               'ff_angle': 'nxt_ff_angle_at_stop'}, inplace=True)

        return self.nxt_ff_at_stop_df

    def get_both_ff_when_seen_df(self, crossing_ff=False, deal_with_rows_with_big_ff_angles=False):
        # This contains the planning-related information at specific time (such as when cur_ff was last visibl)
        # If crossing_ff is true, we'll get nxt_ff_info when cur_ff was first/last seen, and vice versa

        print('Making both_ff_when_seen_df...')
        self.both_ff_when_seen_df = self.nxt_ff_df[[
            'stop_point_index']].copy().set_index('stop_point_index')
        for first_or_last in ['first', 'last']:
            for when_which_ff, ff_df in [('when_nxt_ff', self.nxt_ff_df),
                                         ('when_cur_ff', self.cur_ff_df)]:
                all_point_index = ff_df[f'point_index_ff_{first_or_last}_seen'].values
                self._find_nxt_ff_df_2_and_cur_ff_df_2_based_on_specific_point_index(
                    all_point_index=all_point_index)
                if deal_with_rows_with_big_ff_angles:
                    self._deal_with_rows_with_big_ff_angles(
                        remove_i_o_modify_rows_with_big_ff_angles=True, delete_the_same_rows=True)

                for which_ff_info in ['nxt_', 'cur_']:
                    if (when_which_ff == 'when_cur_ff') & (first_or_last == 'first') & (which_ff_info == 'cur_'):
                        continue  # because the information is already contained in cur ff info at ref point

                    if not crossing_ff:
                        if (which_ff_info == 'nxt_') & (when_which_ff == 'when_cur_ff'):
                            continue
                        if (which_ff_info == 'cur_') & (when_which_ff == 'when_nxt_ff'):
                            continue
                    if deal_with_rows_with_big_ff_angles:
                        ff_df_modified = self.nxt_ff_df_modified if which_ff_info == 'nxt_' else self.cur_ff_df_modified
                    else:
                        ff_df_modified = self.nxt_ff_df2 if which_ff_info == 'nxt_' else self.cur_ff_df2

                    opt_arc_stop_first_vis_bdry = True if (
                        self.optimal_arc_type == 'opt_arc_stop_first_vis_bdry') else False

                    curv_df = curvature_utils.make_curvature_df(ff_df_modified, self.curv_of_traj_df, clean=False,
                                                                monkey_information=self.monkey_information,
                                                                ff_caught_T_new=self.ff_caught_T_new,
                                                                remove_invalid_rows=False,
                                                                opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)
                    if len(curv_df) != len(ff_df_modified):
                        raise ValueError(
                            'The length of curv_df is not the same as the length of ff_df_modified')
                    curv_df = pd.concat([ff_df_modified.drop(columns='point_index').reset_index(
                        drop=True), curv_df.reset_index(drop=True)], axis=1)
                    # for duplicated columns in curv_df, preserve only one
                    curv_df = curv_df.loc[:, ~curv_df.columns.duplicated()]
                    planning_neural_utils.add_to_both_ff_when_seen_df(
                        self.both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df)
        self.both_ff_when_seen_df.reset_index(drop=False, inplace=True)
        return self.both_ff_when_seen_df

    def _make_plan_x_df(self, use_eye_data=True, use_speed_data=True, stop_period_duration=2, ff_radius=10,
                        list_of_cur_ff_cluster_radius=[100, 200, 300],
                        list_of_nxt_ff_cluster_radius=[100, 200, 300],
                        test_or_control='test'):

        self.both_ff_at_ref_df = self.get_both_ff_at_ref_df()
        self.both_ff_at_ref_df['stop_point_index'] = self.nxt_ff_df2['stop_point_index']
        # self.nxt_ff_at_stop_df = self.get_nxt_ff_at_stop_df()
        # self.both_ff_when_seen_df = self.get_both_ff_when_seen_df(deal_with_rows_with_big_ff_angles=False)

        if self.ff_dataframe is None:
            self.get_more_monkey_data()

        if not hasattr(self, 'heading_info_df'):
            self.make_heading_info_df_without_long_process(test_or_control=test_or_control, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                           curv_traj_window_before_stop=self.curv_traj_window_before_stop, use_curvature_to_ff_center=self.use_curvature_to_ff_center)

        self.plan_x_df = plan_factors_utils.make_plan_x_df(self.stops_near_ff_df, self.heading_info_df, self.both_ff_at_ref_df, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted,
                                                           stop_period_duration=stop_period_duration, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value, ff_radius=ff_radius,
                                                           list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                           use_speed_data=use_speed_data, use_eye_data=use_eye_data)

        return self.plan_x_df
