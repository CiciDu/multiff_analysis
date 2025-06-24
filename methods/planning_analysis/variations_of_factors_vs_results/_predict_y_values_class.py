
from planning_analysis.variations_of_factors_vs_results import make_variations_utils
from planning_analysis import ml_for_planning_class, ml_for_planning_utils
from machine_learning.ml_methods import ml_methods_class
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


class _PredictYValues:

    def __init__(self):
        pass

    def make_or_retrieve_all_cur_and_nxt_lr_pred_ff_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.cur_and_nxt_lr_pred_ff_df_path
        if exists_ok:
            if exists(df_path):
                self.all_cur_and_nxt_lr_pred_ff_df = pd.read_csv(df_path)
                print(
                    'Successfully retrieved all_cur_and_nxt_lr_pred_ff_df from ', df_path)
                return self.all_cur_and_nxt_lr_pred_ff_df
            else:
                print('all_cur_and_nxt_lr_pred_ff_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_cur_and_nxt_lr_pred_ff_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_cur_and_nxt_lr_df,
                                                                                                              ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                              variation_func_kwargs={
                                                                                                                  'to_predict_ff': True},
                                                                                                              monkey_name=self.monkey_name,
                                                                                                              path_to_save=df_path,
                                                                                                              )
        return self.all_cur_and_nxt_lr_pred_ff_df

    def make_or_retrieve_all_cur_and_nxt_clf_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = os.path.join(self.combd_cur_and_nxt_folder_path,
                               'ml_results/clf_variations/all_cur_and_nxt_clf_df')
        if exists_ok:
            if exists(df_path):
                self.all_cur_and_nxt_clf_df = pd.read_csv(df_path)
                print('Successfully retrieved all_cur_and_nxt_clf_df from ', df_path)
                return self.all_cur_and_nxt_clf_df
            else:
                print('all_cur_and_nxt_clf_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_cur_and_nxt_clf_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_cur_and_nxt_clf_df,
                                                                                                       ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                       monkey_name=self.monkey_name,
                                                                                                       path_to_save=df_path,
                                                                                                       )
        return self.all_cur_and_nxt_clf_df

    def make_cur_and_nxt_lr_df(self, ref_point_mode, ref_point_value, to_predict_ff=False,
                               keep_monkey_info_choices=[True],
                               key_for_split_choices=['ff_seen'],
                               whether_filter_info_choices=[True],
                               whether_even_out_distribution_choices=[False],
                               whether_test_nxt_ff_flash_after_stop_choices=[
                                   'flexible'],
                               whether_limit_cur_ff_cluster_50_size_choices=[
                                   False],
                               ctrl_flash_compared_to_test_choices=[
                                   'flexible'],
                               max_curv_range_choices=[200]):

        print('to_predict_ff:', to_predict_ff)
        use_lr_func = self.use_lr_to_predict_monkey_info if not to_predict_ff else self.use_lr_to_predict_ff_info

        make_regrouped_info_kwargs = dict(key_for_split_choices=key_for_split_choices,
                                          whether_filter_info_choices=whether_filter_info_choices,
                                          whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                          whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                          whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                          ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                          max_curv_range_choices=max_curv_range_choices)

        self.cur_and_nxt_lr_df = self._make_cur_and_nxt_variations_df(ref_point_mode,
                                                                      ref_point_value,
                                                                      use_lr_func,
                                                                      to_predict_ff=to_predict_ff,
                                                                      keep_monkey_info_choices=keep_monkey_info_choices,
                                                                      make_regrouped_info_kwargs=make_regrouped_info_kwargs)

        return self.cur_and_nxt_lr_df

    def make_cur_and_nxt_clf_df(self,
                                ref_point_mode,
                                ref_point_value,
                                keep_monkey_info_choices=[True],
                                key_for_split_choices=['ff_seen'],
                                whether_filter_info_choices=[True],
                                whether_even_out_distribution_choices=[False],
                                whether_test_nxt_ff_flash_after_stop_choices=[
                                    'flexible'],
                                whether_limit_cur_ff_cluster_50_size_choices=[
                                    False],
                                ctrl_flash_compared_to_test_choices=[
                                    'flexible'],
                                max_curv_range_choices=[200],
                                agg_regrouped_info_kwargs={}):

        use_clf_func = self.use_clf_to_predict_monkey_info

        make_regrouped_info_kwargs = dict(key_for_split_choices=key_for_split_choices,
                                          whether_filter_info_choices=whether_filter_info_choices,
                                          whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                          whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                          whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                          ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                          max_curv_range_choices=max_curv_range_choices)

        self.cur_and_nxt_clf_df = self._make_cur_and_nxt_variations_df(ref_point_mode,
                                                                       ref_point_value,
                                                                       use_clf_func,
                                                                       agg_regrouped_info_kwargs=agg_regrouped_info_kwargs,
                                                                       keep_monkey_info_choices=keep_monkey_info_choices,
                                                                       make_regrouped_info_kwargs=make_regrouped_info_kwargs,
                                                                       )

        return self.cur_and_nxt_clf_df

    def use_clf_to_predict_monkey_info(self, plan_xy_test, plan_xy_ctrl, **agg_regrouped_info_kwargs):

        method_kwargs = dict(y_columns_of_interest=['dir_from_cur_ff_to_stop',
                                                    'dir_from_cur_ff_same_side',
                                                    ],
                             add_ref_interaction_choices=[True],
                             clusters_to_keep_choices=['cur_ff_cluster_100',
                                                       'nxt_ff_cluster_200',
                                                       'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                                       'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                                       ],
                             clusters_for_interaction_choices=[
                                 'none', 'cur_ff_cluster_100'],
                             use_combd_features_for_cluster_only_choices=[
                                 False],
                             max_features_to_save=None,
                             ref_columns_only_choices=[True, False])
        method_kwargs.update(agg_regrouped_info_kwargs)

        self.ml_inst = ml_methods_class.MlMethods()

        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl,
                                                                                    self.ml_inst.try_different_combinations_for_classification,
                                                                                    method_kwargs=method_kwargs,
                                                                                    )

        return regrouped_info

    def use_lr_to_predict_monkey_info(self, plan_xy_test, plan_xy_ctrl):

        method_kwargs = dict(y_columns_of_interest=['diff_in_d_heading_of_traj_from_null',
                                                    'diff_in_abs_d_heading',
                                                    'diff_in_abs_d_curv',
                                                    'dir_from_cur_ff_to_stop',  # this one is classification though
                                                    'd_heading_of_traj',
                                                    'curv_of_traj_before_stop',
                                                    # 'dir_from_cur_ff_same_side',
                                                    # 'diff_in_d_heading'
                                                    ],
                             clusters_for_interaction_choices=[
            'cur_ff_cluster_100',
            # 'nxt_ff_cluster_100',
            # 'cur_ff_cluster_200',
            # 'nxt_ff_cluster_200',
            # 'cur_ff_cluster_300',
            # 'cur_ff_ang_cluster_20',
        ],
            clusters_to_keep_choices=['cur_ff_cluster_100',
                                      'nxt_ff_cluster_200',
                                      'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                      'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                      ],
            max_features_to_save=None,
            use_combd_features_for_cluster_only_choices=[False],
        )
        self.ml_inst = ml_methods_class.MlMethods()

        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl,
                                                                                    self.ml_inst.try_different_combinations_for_linear_regressions,
                                                                                    method_kwargs=method_kwargs,
                                                                                    )
        return regrouped_info

    def use_lr_to_predict_ff_info(self, plan_xy_test, plan_xy_ctrl):
        method_kwargs = dict(y_columns_of_interest=['nxt_ff_angle_at_ref',
                                                    'nxt_ff_distance_at_ref',
                                                    ],
                             clusters_to_keep_choices=['cur_ff_cluster_100',
                                                       'nxt_ff_cluster_200',
                                                       'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                                       'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                                       ],
                             clusters_for_interaction_choices=[],
                             max_features_to_save=None,
                             use_combd_features_for_cluster_only_choices=[False],)

        self.ml_inst = ml_for_planning_class.MlForPlanning()

        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl,
                                                                                    self.ml_inst.try_different_combinations_for_linear_regressions,
                                                                                    method_kwargs=method_kwargs,
                                                                                    )
        return regrouped_info

    def streamline_preparing_for_ml(self,
                                    y_var_column,
                                    ref_columns_only=False,
                                    cluster_to_keep='all',
                                    cluster_for_interaction='none',
                                    add_ref_interaction=True,
                                    winsorize_angle_features=True,
                                    using_lasso=True,
                                    use_combd_features_for_cluster_only=False,
                                    for_classification=False):

        self.separate_plan_xy_test_and_plan_xy_ctrl()

        if self.test_or_control == 'test':
            self.plan_x = self.plan_x_test.copy()
            self.plan_y = self.plan_y_test.copy()
        else:
            self.plan_x = self.plan_x_ctrl.copy()
            self.plan_y = self.plan_y_ctrl.copy()

        print('test_or_control:', self.test_or_control)

        self.x_var_df, self.y_var_df = ml_for_planning_utils.streamline_preparing_for_ml(self.plan_x,
                                                                                         self.plan_y,
                                                                                         y_var_column,
                                                                                         ref_columns_only=ref_columns_only,
                                                                                         cluster_to_keep=cluster_to_keep,
                                                                                         cluster_for_interaction=cluster_for_interaction,
                                                                                         add_ref_interaction=add_ref_interaction,
                                                                                         winsorize_angle_features=winsorize_angle_features,
                                                                                         using_lasso=using_lasso,
                                                                                         ensure_cur_ff_at_front=False,
                                                                                         use_combd_features_for_cluster_only=use_combd_features_for_cluster_only,
                                                                                         for_classification=for_classification)
