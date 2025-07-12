from machine_learning.ml_methods import regression_utils, ml_methods_class
import pandas as pd
import matplotlib.pyplot as plt
import gc


class MlForPlanning(ml_methods_class.MlMethods):

    def __init__(self,
                 x_var_df=None,
                 y_var_df=None,
                 ):
        super().__init__(x_var_df=x_var_df, y_var_df=y_var_df)

    def try_different_combinations_for_linear_regressions(self, data_source,
                                                          y_columns_of_interest=['diff_in_d_heading_to_cur_ff',
                                                                                 'diff_in_abs_angle_to_nxt_ff',
                                                                                 'diff_in_abs_d_curv',
                                                                                 'dir_from_cur_ff_to_stop',
                                                                                 'd_heading_of_traj',
                                                                                 'curv_of_traj_before_stop',
                                                                                 ],
                                                          add_ref_interaction_choices=[
                                                              True],
                                                          clusters_to_keep_choices=[
                                                              'cur_ff_cluster_100'],
                                                          clusters_for_interaction_choices=[
                                                              'none', 'cur_ff_cluster_100'],
                                                          ref_columns_only_choices=[
                                                              False, True],
                                                          use_combd_features_for_cluster_only_choices=[
                                                              False, True],
                                                          max_features_to_save=None,
                                                          add_coeff=True):

        self.lr_variations_df = pd.DataFrame()
        process_combination_kwargs = dict(
            max_features_to_save=max_features_to_save, add_coeff=add_coeff)

        self.lr_variations_df = self._try_different_combinations_for_learning(data_source,
                                                                              self._process_combination_for_lr,
                                                                              process_combination_kwargs=process_combination_kwargs,
                                                                              y_columns_of_interest=y_columns_of_interest,
                                                                              add_ref_interaction_choices=add_ref_interaction_choices,
                                                                              clusters_to_keep_choices=clusters_to_keep_choices,
                                                                              clusters_for_interaction_choices=clusters_for_interaction_choices,
                                                                              ref_columns_only_choices=ref_columns_only_choices,
                                                                              use_combd_features_for_cluster_only_choices=use_combd_features_for_cluster_only_choices,
                                                                              winsorize_angle_features_choices=[
                                                                                  True],
                                                                              using_lasso_choices=[True])
        return self.lr_variations_df

    def try_different_combinations_for_ml(self, data_source,
                                          y_columns_of_interest=['diff_in_d_heading_to_cur_ff',
                                                                 'diff_in_abs_angle_to_nxt_ff',
                                                                 'diff_in_abs_d_curv',
                                                                 'd_heading_of_traj',
                                                                 'dir_from_cur_ff_to_stop',
                                                                 'curv_of_traj_before_stop',
                                                                 ],
                                          add_ref_interaction_choices=[True],
                                          clusters_to_keep_choices=['cur_ff_cluster_100',
                                                                    'nxt_ff_cluster_200',
                                                                    'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                                                    'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                                                    ],
                                          ref_columns_only_choices=[
                                              False, True],
                                          model_names=['grad_boosting', 'rf']):

        self.ml_variations_df = pd.DataFrame()
        process_combination_kwargs = dict(model_names=model_names)

        self.ml_variations_df = self._try_different_combinations_for_learning(data_source,
                                                                              self._process_combination_for_ml,
                                                                              process_combination_kwargs=process_combination_kwargs,
                                                                              y_columns_of_interest=y_columns_of_interest,
                                                                              add_ref_interaction_choices=add_ref_interaction_choices,
                                                                              clusters_to_keep_choices=clusters_to_keep_choices,
                                                                              ref_columns_only_choices=ref_columns_only_choices,
                                                                              using_lasso_choices=[
                                                                                  False]
                                                                              )

        return self.ml_variations_df

    def try_different_combinations_for_classification(self, data_source,
                                                      y_columns_of_interest=['dir_from_cur_ff_to_stop',
                                                                             'dir_from_cur_ff_same_side',
                                                                             ],
                                                      add_ref_interaction_choices=[
                                                          True],
                                                      clusters_to_keep_choices=['cur_ff_cluster_100',
                                                                                'nxt_ff_cluster_200',
                                                                                'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                                                                'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                                                                ],
                                                      clusters_for_interaction_choices=[
                                                          'cur_ff_cluster_100'],
                                                      ref_columns_only_choices=[
                                                          False, True],
                                                      use_combd_features_for_cluster_only_choices=[
                                                          False],
                                                      max_features_to_save=None,
                                                      add_coeff=True):

        self.clf_variations_df = pd.DataFrame()
        process_combination_kwargs = dict(max_features_to_save=max_features_to_save,
                                          add_coeff=add_coeff)

        self.clf_variations_df = self._try_different_combinations_for_learning(data_source,
                                                                               self._process_combination_for_clf,
                                                                               process_combination_kwargs=process_combination_kwargs,
                                                                               y_columns_of_interest=y_columns_of_interest,
                                                                               add_ref_interaction_choices=add_ref_interaction_choices,
                                                                               clusters_to_keep_choices=clusters_to_keep_choices,
                                                                               clusters_for_interaction_choices=clusters_for_interaction_choices,
                                                                               ref_columns_only_choices=ref_columns_only_choices,
                                                                               use_combd_features_for_cluster_only_choices=use_combd_features_for_cluster_only_choices,
                                                                               using_lasso_choices=[False])

        return self.clf_variations_df

    def _process_combination_for_lr(self,
                                    max_features_to_save=None,
                                    add_coeff=True,
                                    param_info_to_record={}):

        # also work simply on train and test set
        self.use_linear_regression(show_plot=False, y_var_name=self.y_test.name)
        print('num_features:', self.X_train.shape[1])

        temp_info = {'num_features': [self.X_train.shape[1]],
                     'num_significant_features': len(self.summary_df),
                     'sample_size': [self.X_train.shape[0]],
                     'rsquared': [round(self.results.rsquared, 4)],
                     'adj_rsquared': [round(self.results.rsquared_adj, 4)],
                     'r2_test': [round(self.r2_test, 4)]}
        temp_info.update(param_info_to_record)
        temp_info = pd.DataFrame(temp_info, index=[0])

        more_temp_info = regression_utils.get_significant_features_in_one_row(
            self.summary_df, max_features_to_save=max_features_to_save, add_coeff=add_coeff)
        test_avg_r_squared, test_std_r_squared, train_avg_r_squared, train_std_r_squared = regression_utils.use_linear_regression_cv(
            self.x_var_prepared, self.y_var_prepared)
        more_temp_info['avg_r_squared'] = round(test_avg_r_squared, 4)
        more_temp_info['std_r_squared'] = round(test_std_r_squared, 4)

        temp_info = pd.concat([temp_info, more_temp_info], axis=1)
        self.lr_variations_df = pd.concat(
            [self.lr_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.lr_variations_df

    def _process_combination_for_ml(self,
                                    model_names=[
                                        'linreg', 'grad_boosting', 'rf'],
                                    param_info_to_record={},
                                    ):

        self.use_ml(model_names=model_names, use_cv=True)
        print('num_features:', self.X_train.shape[1])

        temp_info = {'num_features': [self.X_train.shape[1]],
                     'sample_size': [self.X_train.shape[0]],
                     }
        temp_info.update(param_info_to_record)
        temp_info = pd.DataFrame(temp_info, index=[0])

        # # repeat temp_info for three rows
        # temp_info = pd.concat([temp_info]*len(model_names), axis=0, ignore_index=True)
        temp_info = pd.concat(
            [self.model_comparison_df.reset_index(drop=True), temp_info], axis=1)
        self.ml_variations_df = pd.concat(
            [self.ml_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.ml_variations_df

    def _process_combination_for_clf(self,
                                     max_features_to_save=None,
                                     add_coeff=True,
                                     param_info_to_record={},
                                     ):

        self.use_logistic_regression(
            self.data_source.x_var_df, self.data_source.y_var_df)
        temp_info = regression_utils.get_significant_features_in_one_row(
            self.summary_df, max_features_to_save=max_features_to_save, add_coeff=add_coeff)

        print('num_features:', self.data_source.x_var_df.shape[1])
        print('num_selected_features:', self.num_selected_features)
        print('sample_size:', self.data_source.x_var_df.shape[0])

        more_temp_info = {'average_accuracy': self.average_accuracy,
                          'train_avg_accuracy': self.train_avg_accuracy,
                          'sample_size': self.data_source.x_var_df.shape[0],
                          'num_features': self.data_source.x_var_df.shape[1],
                          'num_selected_features': self.num_selected_features,
                          }
        more_temp_info.update(param_info_to_record)
        more_temp_info = pd.DataFrame(more_temp_info, index=[0])

        temp_info = pd.concat([more_temp_info, temp_info], axis=1)
        self.clf_variations_df = pd.concat(
            [self.clf_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.clf_variations_df

    def _try_different_combinations_for_learning(self, data_source,
                                                 process_combination_func,
                                                 y_columns_of_interest=['diff_in_d_heading_to_cur_ff',
                                                                        'diff_in_abs_angle_to_nxt_ff',
                                                                        'diff_in_abs_d_curv',
                                                                        'dir_from_cur_ff_to_stop',
                                                                        'd_heading_of_traj',
                                                                        'curv_of_traj_before_stop',
                                                                        ],
                                                 add_ref_interaction_choices=[
                                                     False],
                                                 clusters_to_keep_choices=['cur_ff_cluster_100',
                                                                           'nxt_ff_cluster_200',
                                                                           'cur_ff_cluster_100_PLUS_nxt_ff_cluster_200',
                                                                           'cur_ff_cluster_100_PLUS_cur_ff_cluster_300',
                                                                           ],
                                                 clusters_for_interaction_choices=[
                                                     'none', 'cur_ff_cluster_100'],
                                                 ref_columns_only_choices=[
                                                     False, True],
                                                 use_combd_features_for_cluster_only_choices=[
                                                     False],
                                                 winsorize_angle_features_choices=[
                                                     True],
                                                 using_lasso_choices=[True],
                                                 process_combination_kwargs={}
                                                 ):

        self.data_source = data_source

        for y_var_column in y_columns_of_interest:
            for add_ref_interaction in add_ref_interaction_choices:
                for ref_columns_only in ref_columns_only_choices:
                    gc.collect()
                    temp_clusters_to_keep_choices, temp_clusters_for_interaction_choices, temp_use_combd_features_for_cluster_only_choices = self._get_temp_choices(ref_columns_only, clusters_to_keep_choices,
                                                                                                                                                                    use_combd_features_for_cluster_only_choices, clusters_for_interaction_choices)
                    for cluster_to_keep in temp_clusters_to_keep_choices:
                        if cluster_to_keep != 'none':
                            if cluster_to_keep == 'all':
                                clusters_to_keep = cluster_to_keep.split(
                                    '_PLUS_')
                                temp_clusters_for_interaction_choices = [
                                    cluster for cluster in clusters_for_interaction_choices if cluster in clusters_to_keep]
                            else:
                                temp_clusters_for_interaction_choices = clusters_for_interaction_choices
                            temp_clusters_for_interaction_choices = [
                                'none'] + temp_clusters_for_interaction_choices
                            temp_clusters_for_interaction_choices = list(
                                set(temp_clusters_for_interaction_choices))
                        else:
                            temp_clusters_for_interaction_choices = ['none']

                        for cluster_for_interaction in temp_clusters_for_interaction_choices:
                            for use_combd_features_for_cluster_only in temp_use_combd_features_for_cluster_only_choices:
                                for winsorize_angle_features in winsorize_angle_features_choices:
                                    for using_lasso in using_lasso_choices:
                                        param_info_to_record = self._process_data(y_var_column, ref_columns_only, add_ref_interaction, cluster_to_keep, cluster_for_interaction,
                                                                                  use_combd_features_for_cluster_only, winsorize_angle_features, using_lasso)
                                        df = process_combination_func(param_info_to_record=param_info_to_record,
                                                                      **process_combination_kwargs)

        try:
            df['monkey_name'] = self.data_source.monkey_name
            df['ref_point_mode'] = self.data_source.ref_point_mode
            df['ref_point_value'] = self.data_source.ref_point_value
        except AttributeError:
            pass

        return df

    def _get_temp_choices(self, ref_columns_only, clusters_to_keep_choices, use_combd_features_for_cluster_only_choices, clusters_for_interaction_choices):
        if ref_columns_only:
            return ['none'], ['none'], [False]
        else:
            clusters_to_keep_choices = ['none'] + clusters_to_keep_choices
            clusters_to_keep_choices = list(set(clusters_to_keep_choices))
            return clusters_to_keep_choices, clusters_for_interaction_choices, use_combd_features_for_cluster_only_choices

    def _process_data(self, y_var_column, ref_columns_only, add_ref_interaction, cluster_to_keep, cluster_for_interaction, use_combd_features_for_cluster_only, winsorize_angle_features, using_lasso):

        print('   ')
        print('================================================================')
        print('ref_point_mode: ', self.data_source.ref_point_mode)
        print('ref_point_value: ', self.data_source.ref_point_value)
        print('y_var_column: ', y_var_column)
        print('ref_columns_only: ', ref_columns_only)
        print('add_ref_interaction: ', add_ref_interaction)
        print('cluster_to_keep: ', cluster_to_keep)
        print('cluster_for_interaction: ', cluster_for_interaction)
        print('use_combd_features_for_cluster_only:',
              use_combd_features_for_cluster_only)

        self.data_source.streamline_preparing_for_ml(y_var_column, cluster_to_keep=cluster_to_keep, cluster_for_interaction=cluster_for_interaction,
                                                     add_ref_interaction=add_ref_interaction, ref_columns_only=ref_columns_only,
                                                     winsorize_angle_features=winsorize_angle_features, using_lasso=using_lasso,
                                                     use_combd_features_for_cluster_only=use_combd_features_for_cluster_only)

        if (self.data_source.x_var_df.shape[0] == 0) or (self.data_source.x_var_df.shape[1] == 0):
            return {}
        self.use_train_test_split(self.data_source.x_var_df, self.data_source.y_var_df,
                                  y_var_column=y_var_column, remove_outliers=True)

        param_info_to_record = {'y_var_column': [y_var_column],
                                'add_ref_interaction': [add_ref_interaction],
                                'cluster_to_keep': [cluster_to_keep],
                                'cluster_for_interaction': [cluster_for_interaction],
                                'ref_columns_only': [ref_columns_only],
                                'use_combd_features_for_cluster_only': [use_combd_features_for_cluster_only],
                                'winsorize_angle_features': [winsorize_angle_features],
                                'using_lasso': [using_lasso]}
        return param_info_to_record
