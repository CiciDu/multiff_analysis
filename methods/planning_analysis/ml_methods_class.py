import sys
from planning_analysis.plan_factors import plan_factors_utils, plan_factors_class, test_vs_control_utils
from planning_analysis.show_planning import show_planning_utils
from planning_analysis import ml_methods_utils
from machine_learning import machine_learning_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import gc
from sklearn.metrics import accuracy_score
import warnings

class MlMethods():

    def __init__(self,
                 x_var_df=None,
                 y_var_df=None,
                 ):
        if x_var_df is not None:
            self.x_var_df = x_var_df
        if y_var_df is not None:
            self.y_var_df = y_var_df


    def use_train_test_split(self, x_var_df, y_var_df, y_var_column='d_monkey_angle_since_stop_ff_first_seen', remove_outliers=True):
        self.x_var_prepared, self.y_var_prepared = ml_methods_utils.further_prepare_x_var_and_y_var(x_var_df, y_var_df, y_var_column=y_var_column, remove_outliers=remove_outliers)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_var_prepared, self.y_var_prepared, test_size=0.2)


    def use_ml(self, model_names=['linreg', 'svr', 'dt', 'bagging', 'boosting', 'grad_boosting', 'rf'], use_cv=False):
        self.model_comparison_df, self.chosen_model_info = machine_learning_utils.use_ml_model_for_regression(self.X_train, self.y_train, self.X_test, self.y_test,
                                                                                                                             model_names=model_names, use_cv=use_cv)

    def use_ml_with_plots(self, models=None):
        # Define the models
        if models is None:
            models = {
                "Bagging": BaggingRegressor(random_state=42),
                "Boosting": AdaBoostRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42)
            }

        # Fit the models and make predictions
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # Plot the predicted results against actual values
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, y_pred)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{name}: Actual vs Predicted Values')
            # also plot a line of y=x
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
            plt.show()
            
            # Print the mean squared error
            mse = mean_squared_error(self.y_test, y_pred)
            print(f'{name} Mean Squared Error: {mse}')

            # Print feature importances for RandomForestRegressor
            if name == 'Random Forest':
                feature_results_df = pd.DataFrame({'feature': self.X_train.columns, 'importance': model.feature_importances_})
                feature_results_df.sort_values(by='importance', ascending=False, inplace=True)
                self.feature_results_df = feature_results_df
                

    def process_summary_df(self, summary_df):
        self.summary_df_all = summary_df.copy()
        self.summary_df = summary_df[summary_df['p_value'] <= 0.05].copy()
        self.summary_df['rank_by_abs_coeff'] = self.summary_df['abs_coeff'].rank(ascending=False, method='first').astype(int)
        self.summary_df.reset_index(drop=False, inplace=True)


    def use_linear_regression(self, show_plot=True):
        summary_df, self.y_pred, self.results, self.r_squared_on_test = machine_learning_utils.use_linear_regression(self.X_train, self.X_test, self.y_train,                                                                                               self.y_test, show_plot=show_plot)
        self.process_summary_df(summary_df)


    def use_logistic_regression(self, x_var_df, y_var_df):
        # suppress warnings

        warnings.filterwarnings("ignore")
        self.summary_df, self.average_accuracy, self.train_avg_accuracy, self.num_selected_features = machine_learning_utils.use_logistic_regression(x_var_df, y_var_df)

        # self.coeff = self.results.params
        # self.pvalues = self.results.pvalues  
        self.process_summary_df(self.summary_df)


    def use_ml_model_for_classification(self, x_var_df, y_var_df, model=None):
        y_var_df = np.array(y_var_df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_var_df, y_var_df, test_size=0.2) #random_state=42
        self.model, self.y_pred, self.model_comparison_df = machine_learning_utils.use_ml_model_for_classification(
            self.X_train, self.y_train, self.X_test, self.y_test, model=model,
            )

    def use_neural_network(self):
        self.model, self.predictions = machine_learning_utils.use_neural_network_on_linear_regression_func(self.X_train.values, self.y_train.values, 
                                                                                                           self.X_test.values, self.y_test.values)
        
        r_squared = r2_score(self.y_test, self.predictions)
        print("R-squared on test set:", r_squared)

    def use_vif(self):
        # Calculate VIF
        self.vif = pd.DataFrame()
        self.vif["features"] = self.x_var_df.columns
        vif_values = []
        for i in range(self.x_var_df.shape[1]):
            vif_values.append(variance_inflation_factor(self.x_var_df.values, i))
            if i % 10 == 0:
                print(f'{i} out of {self.x_var_df.shape[1]} features are processed.')
        self.vif["VIF"] = vif_values
        self.vif = self.vif.sort_values(by='VIF', ascending=False).round(1)
        print(self.vif)


    def show_correlation_heatmap(self, specific_columns=None):
        if specific_columns is None:
            specific_columns = self.vif[self.vif["VIF"] > 5].features.values[:15]
        # calculate the correlation coefficient among the columns with VIF > 5
        self.corr_coeff = self.x_var_df[specific_columns].corr()
        plt.figure(figsize=(15, 15))
        sns.heatmap(self.corr_coeff, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
        plt.show()


    def try_different_combinations_for_linear_regressions(self, data_source,
                                                        y_columns_of_interest=['curvature_of_traj_before_stop',
                                                                                'ref_d_heading_of_traj',
                                                                                'dev_d_angle_from_null',
                                                                                'dir_from_stop_ff_to_stop'],
                                                        add_ref_interaction_choices=[True],                        
                                                        clusters_to_keep_choices=['none'],  
                                                        clusters_for_interaction_choices=['none'],                                                                                                                                                  
                                                        ref_columns_only_choices=[False, True],
                                                        use_combd_features_for_cluster_only_choices=[False, True],
                                                        max_features_to_save=None,
                                                        add_coeff=True):
        
        self.lr_variations_df = pd.DataFrame()
        process_combination_kwargs = dict(max_features_to_save=max_features_to_save, add_coeff=add_coeff)
        
        self.lr_variations_df = self._try_different_combinations_for_learning(data_source,
                                                                            self._process_combination_for_lr,
                                                                            process_combination_kwargs=process_combination_kwargs,
                                                                            y_columns_of_interest=y_columns_of_interest,
                                                                            add_ref_interaction_choices=add_ref_interaction_choices,
                                                                            clusters_to_keep_choices=clusters_to_keep_choices,
                                                                            clusters_for_interaction_choices=clusters_for_interaction_choices,
                                                                            ref_columns_only_choices=ref_columns_only_choices,
                                                                            use_combd_features_for_cluster_only_choices=use_combd_features_for_cluster_only_choices,
                                                                            winsorize_angle_features_choices=[True],
                                                                            using_lasso_choices=[True])
        return self.lr_variations_df




    def try_different_combinations_for_ml(self, data_source,
                                            y_columns_of_interest=['curvature_of_traj_before_stop',
                                                                    'ref_d_heading_of_traj',
                                                                    'dev_d_angle_from_null',
                                                                    'dir_from_stop_ff_to_stop'],
                                            add_ref_interaction_choices=[True],                        
                                            clusters_to_keep_choices=['none'],                                                                                                                                                  
                                            ref_columns_only_choices=[False, True],                                       
                                            model_names=['linreg', 'grad_boosting', 'rf']):
        
        self.ml_variations_df = pd.DataFrame()
        process_combination_kwargs = dict(model_names=model_names)

        self.ml_variations_df = self._try_different_combinations_for_learning(data_source,
                                                                            self._process_combination_for_ml,
                                                                            process_combination_kwargs=process_combination_kwargs,
                                                                            y_columns_of_interest=y_columns_of_interest,
                                                                            add_ref_interaction_choices=add_ref_interaction_choices,
                                                                            clusters_to_keep_choices=clusters_to_keep_choices,
                                                                            ref_columns_only_choices=ref_columns_only_choices,
                                                                            using_lasso_choices=[False]
                                                                            )

        return self.ml_variations_df


    def try_different_combinations_for_classification(self, data_source,
                                            y_columns_of_interest=['dir_from_stop_ff_to_stop'],
                                            add_ref_interaction_choices=[True],                        
                                            clusters_to_keep_choices=['stop_ff_cluster_100_PLUS_alt_ff_cluster_100'],  
                                            clusters_for_interaction_choices=['stop_ff_cluster_100'],                                                                                                                                                  
                                            ref_columns_only_choices=[False, True],
                                            use_combd_features_for_cluster_only_choices=[False],
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


        self.use_linear_regression(show_plot=False)
        print('num_features:', self.X_train.shape[1]) 

        temp_info = ml_methods_utils.get_significant_features_in_one_row(self.summary_df, max_features_to_save=max_features_to_save, add_coeff=add_coeff)  

        avg_r_squared, std_r_squared = machine_learning_utils.use_linear_regression_cv(self.x_var_prepared, self.y_var_prepared)
        temp_info['avg_r_squared'] = round(avg_r_squared, 4)
        temp_info['std_r_squared'] = round(std_r_squared, 4)

        more_temp_info = {'num_features': [self.X_train.shape[1]],
                        'num_significant_features': len(self.summary_df),
                        'sample_size': [self.X_train.shape[0]],
                        'rsquared': [round(self.results.rsquared, 4)],
                        'adj_rsquared': [round(self.results.rsquared_adj, 4)],
                        'r_squared_on_test': [round(self.r_squared_on_test, 4)]}
        more_temp_info.update(param_info_to_record)
        more_temp_info = pd.DataFrame(more_temp_info, index=[0])

        temp_info = pd.concat([temp_info, more_temp_info], axis=1)
        self.lr_variations_df = pd.concat([self.lr_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.lr_variations_df



    def _process_combination_for_ml(self, 
                                   model_names=['linreg', 'grad_boosting', 'rf'],
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
        temp_info = pd.concat([self.model_comparison_df.reset_index(drop=True), temp_info], axis=1)
        self.ml_variations_df = pd.concat([self.ml_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.ml_variations_df
    

    def _process_combination_for_clf(self,                              
                                   max_features_to_save=None, 
                                   add_coeff=True,
                                   param_info_to_record={},
                                   ):
    

        self.use_logistic_regression(self.data_source.x_var_df, self.data_source.y_var_df)
        temp_info = ml_methods_utils.get_significant_features_in_one_row(self.summary_df, max_features_to_save=max_features_to_save, add_coeff=add_coeff)  

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
        self.clf_variations_df = pd.concat([self.clf_variations_df, temp_info], axis=0).reset_index(drop=True)
        return self.clf_variations_df


    def _try_different_combinations_for_learning(self, data_source,
                                                process_combination_func,
                                                y_columns_of_interest=['curvature_of_traj_before_stop',
                                                                        'ref_d_heading_of_traj',
                                                                        'dev_d_angle_from_null',
                                                                        'dir_from_stop_ff_to_stop'],
                                                add_ref_interaction_choices=[False],                        
                                                clusters_to_keep_choices=['none'],  
                                                clusters_for_interaction_choices=['none'],                                                                                                                                                  
                                                ref_columns_only_choices=[False, True],
                                                use_combd_features_for_cluster_only_choices=[False],
                                                winsorize_angle_features_choices=[True],   
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
                                clusters_to_keep = cluster_to_keep.split('_PLUS_')
                                temp_clusters_for_interaction_choices = [cluster for cluster in clusters_for_interaction_choices if cluster in clusters_to_keep]
                            else:
                                temp_clusters_for_interaction_choices = clusters_for_interaction_choices
                            temp_clusters_for_interaction_choices = ['none'] + temp_clusters_for_interaction_choices
                            temp_clusters_for_interaction_choices = list(set(temp_clusters_for_interaction_choices))                             
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
        print('use_combd_features_for_cluster_only:', use_combd_features_for_cluster_only)

        self.data_source.streamline_preparing_for_ml(y_var_column, cluster_to_keep=cluster_to_keep, cluster_for_interaction=cluster_for_interaction,
                                                    add_ref_interaction=add_ref_interaction, ref_columns_only=ref_columns_only,
                                                    winsorize_angle_features=winsorize_angle_features, using_lasso=using_lasso,
                                                    use_combd_features_for_cluster_only=use_combd_features_for_cluster_only)

        if (self.data_source.x_var_df.shape[0] == 0) or (self.data_source.x_var_df.shape[1] == 0):
            return
        self.use_train_test_split(self.data_source.x_var_df, self.data_source.y_var_df, y_var_column=y_var_column, remove_outliers=True)

        param_info_to_record = {'y_var_column': [y_var_column],
                                'add_ref_interaction': [add_ref_interaction],
                                'cluster_to_keep': [cluster_to_keep],
                                'cluster_for_interaction': [cluster_for_interaction],
                                'ref_columns_only': [ref_columns_only],
                                'use_combd_features_for_cluster_only': [use_combd_features_for_cluster_only],
                                'winsorize_angle_features': [winsorize_angle_features],
                                'using_lasso': [using_lasso]}
        return param_info_to_record
