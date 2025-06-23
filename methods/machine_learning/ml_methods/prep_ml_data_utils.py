import sys
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling
from planning_analysis.plan_factors import plan_factors_utils, plan_factors_class, test_vs_control_utils
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.only_cur_ff import features_to_keep_utils, only_cur_ff_utils
from planning_analysis import prep_ml_data_utils
from non_behavioral_analysis.neural_data_analysis.visualize_neural_data import plot_modeling_result
from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data.cca_methods import cca_class, cca_plotting
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
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import numpy as np
import matplotlib.pyplot as plt


def add_interaction_terms_to_df(df, specific_columns=None):

    if specific_columns is not None:
        if len(specific_columns) > 0:
            # Assuming df is your DataFrame and specific_columns is a list of column names
            df_selected = df[specific_columns]

            # Initialize PolynomialFeatures
            poly = PolynomialFeatures(
                degree=2, interaction_only=False, include_bias=False)

            # Fit and transform the selected DataFrame
            df_interactions = poly.fit_transform(df_selected)

            # Generate new column names (including original and interaction terms)
            new_column_names = poly.get_feature_names_out(
                input_features=df_selected.columns)

            # Create a new DataFrame with the interaction terms and new column names
            df_with_interactions = pd.DataFrame(
                df_interactions, columns=new_column_names)

            # df_with_interactions now contains the original columns, squared terms, and interaction terms with appropriate names

            # drop the original columns
            df_with_interactions.drop(columns=specific_columns, inplace=True)
            df_with_interactions.index = df.index

            df = pd.concat([df, df_with_interactions], axis=1)
            print('Added interaction terms.')
    return df


def make_x_and_y_var_df(x_df, y_df, drop_na=True, scale_x_var=True, use_pca=False, n_components_for_pca=None):
    x_var_df = x_df.copy()
    y_var_df = y_df.copy()

    # scale the variables
    if scale_x_var:
        sc = StandardScaler()
        sc.fit(x_var_df)
        columns = x_var_df.columns
        index = x_var_df.index
        x_var_df = sc.transform(x_var_df)
        x_var_df = pd.DataFrame(x_var_df, columns=columns, index=index)

    if drop_na:
        if len(y_var_df.shape) > 1:
            if y_var_df.shape[1] > 1:
                x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_var(
                    x_var_df, y_var_df)
            else:
                x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_and_y_var(
                    x_var_df, y_var_df)
        else:
            x_var_df, y_var_df = plan_factors_utils.drop_na_in_x_and_y_var(
                x_var_df, y_var_df)

    if use_pca:
        if n_components_for_pca is None:
            n_components_for_pca = min(x_var_df.shape[0], x_var_df.shape[1])
        # 'mle' automatically selects the number of components or choose a fixed number
        pca = PCA(n_components=n_components_for_pca)
        x_var_df = pca.fit_transform(x_var_df)
        x_var_df = pd.DataFrame(
            x_var_df, columns=[f'x{i+1}' for i in range(n_components_for_pca)])

    x_var_df = sm.add_constant(x_var_df)
    x_var_df.reset_index(drop=True, inplace=True)
    y_var_df.reset_index(drop=True, inplace=True)

    return x_var_df, y_var_df


def further_prepare_x_var_and_y_var(x_var_df, y_var_df, y_var_column='d_monkey_angle_since_cur_ff_first_seen', remove_outliers=True):

    x_var_df = x_var_df.reset_index(drop=True)
    y_var_df = y_var_df.reset_index(drop=True)
    y_var = y_var_df[y_var_column].copy()
    x_var = x_var_df.copy()

    if remove_outliers:
        # remove rows in y_var_df that are 3 std above the mean, and teh corresponding rows in x_var_df
        x_var, y_var = show_planning_utils.remove_outliers(x_var, y_var)

    return x_var, y_var
