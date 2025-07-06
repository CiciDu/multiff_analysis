from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import Lasso


class MlMethods():

    def __init__(self,
                 x_var_df=None,
                 y_var_df=None,
                 ):
        if x_var_df is not None:
            self.x_var_df = x_var_df
        if y_var_df is not None:
            self.y_var_df = y_var_df

    def use_train_test_split(self, x_var_df, y_var_df, y_var_column='d_monkey_angle_since_cur_ff_first_seen', remove_outliers=True):
        self.x_var_prepared, self.y_var_prepared = prep_ml_data_utils.further_prepare_x_var_and_y_var(
            x_var_df, y_var_df, y_var_column=y_var_column, remove_outliers=remove_outliers)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_var_prepared, self.y_var_prepared, test_size=0.2)

    def use_ml(self, model_names=['linreg', 'svr', 'dt', 'bagging', 'boosting', 'grad_boosting', 'rf'], use_cv=False):
        self.model_comparison_df, self.chosen_model_info = regression_utils.use_ml_model_for_regression(self.X_train, self.y_train, self.X_test, self.y_test,
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
            plt.plot([self.y_test.min(), self.y_test.max()], [
                     self.y_test.min(), self.y_test.max()], 'k--', lw=4)
            plt.show()

            # Print the mean squared error
            mse = mean_squared_error(self.y_test, y_pred)
            print(f'{name} Mean Squared Error: {mse}')

            # Print feature importances for RandomForestRegressor
            if name == 'Random Forest':
                feature_results_df = pd.DataFrame(
                    {'feature': self.X_train.columns, 'importance': model.feature_importances_})
                feature_results_df.sort_values(
                    by='importance', ascending=False, inplace=True)
                self.feature_results_df = feature_results_df

    def process_summary_df(self, summary_df):
        self.summary_df_all = summary_df.copy()
        # self.summary_df = summary_df[summary_df['p_value'] <= 0.05].copy()
        self.summary_df['significant'] = self.summary_df['p_value'] <= 0.05
        self.summary_df['rank_by_abs_coeff'] = self.summary_df['abs_coeff'].rank(
            ascending=False, method='first').astype(int)
        #self.summary_df.reset_index(drop=False, inplace=True)

    def use_linear_regression(self, show_plot=True, y_var_name=None):
        self.summary_df, y_pred, results, r2_test = regression_utils.use_linear_regression(
            self.X_train, self.X_test, self.y_train, self.y_test, show_plot=show_plot, y_var_name=y_var_name)
        self.process_summary_df(self.summary_df)

    def split_and_use_linear_regression(self, x_var_df, y_var_df, test_size=0.2):
        # if y_var_df is a series, convert it to a dataframe
        if isinstance(y_var_df, pd.Series):
            y_var_df = pd.DataFrame(y_var_df) 
        elif isinstance(y_var_df, pd.DataFrame):
            if y_var_df.shape[1] > 1:
                raise ValueError('y_var_df should contain only one column.')
        else:
            raise ValueError('y_var_df should be a series or a dataframe.')
        
        y_var_name = y_var_df.columns[0]
        
        self.temp_x_var_df, self.temp_y_var_df = prep_ml_data_utils.drop_na_in_x_and_y_var(
            x_var_df, y_var_df)        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.temp_x_var_df, self.temp_y_var_df, test_size=test_size)
        self.use_linear_regression(show_plot=True, y_var_name=y_var_name)

    def use_logistic_regression(self, x_var_df, y_var_df, select_features=True, lasso_alpha=0.01):

        # Use the Lasso model for feature selection
        if select_features:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lasso = Lasso(alpha=lasso_alpha)  # Adjust alpha as needed
                lasso.fit(x_var_df.values, y_var_df.values.reshape(-1))
                self.selected_features = x_var_df.columns[(lasso.coef_ != 0)]
                X_selected = x_var_df[self.selected_features]
                self.num_selected_features = X_selected.shape[1]
        else:
            X_selected = x_var_df

        self.summary_df, _ = classification_utils.use_logistic_regression(
            X_selected, y_var_df)
        self.process_summary_df(self.summary_df)

    def use_ml_model_for_classification(self, x_var_df, y_var_df, model=None):
        y_var_df = np.array(y_var_df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x_var_df, y_var_df, test_size=0.2)  # random_state=42
        self.model, self.y_pred, self.model_comparison_df = classification_utils.use_ml_model_for_classification(
            self.X_train, self.y_train, self.X_test, self.y_test, model=model,
        )

    def use_neural_network(self):
        self.model, self.predictions = regression_utils.use_neural_network_on_linear_regression_func(self.X_train.values, self.y_train.values,
                                                                                                     self.X_test.values, self.y_test.values)

        r_squared = r2_score(self.y_test, self.predictions)
        print("R-squared on test set:", r_squared)

    def use_vif(self, var_df):
        # Calculate VIF
        self.vif_df = pd.DataFrame()
        self.vif_df["feature"] = var_df.columns
        vif_values = []
        for i in range(var_df.shape[1]):
            vif_values.append(variance_inflation_factor(
                var_df.values, i))
            if i % 10 == 0:
                print(
                    f'{i} out of {self.x_var_df.shape[1]} features are processed.')
        self.vif_df['vif'] = vif_values
        self.vif_df = self.vif_df.sort_values(
            by='vif', ascending=False).round(1)
        print(self.vif_df)

    def show_correlation_heatmap(self, specific_columns=None):
        if specific_columns is None:
            specific_columns = self.vif_df[self.vif_df['vif']
                                           > 5].feature.values[:15]
        # calculate the correlation coefficient among the columns with VIF > 5
        self.corr_coeff = self.x_var_df[specific_columns].corr()
        plt.figure(figsize=(15, 15))
        sns.heatmap(self.corr_coeff, cmap='coolwarm',
                    annot=True, linewidths=1, vmin=-1)
        plt.show()
