import sys
from non_behavioral_analysis.neural_data_analysis.model_neural_data import drop_high_corr_vars
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from os.path import exists
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

import re


def drop_columns_with_high_vif(y_var_lags, vif_threshold=5, vif_threshold_for_initial_subset=5, verbose=True,
                               filter_by_feature=True,
                               filter_by_subsets=False,
                               filter_by_all_columns=False,
                               get_column_subsets_func=None):

    if (not filter_by_feature) & (not filter_by_subsets) & (not filter_by_all_columns):
        return y_var_lags

    num_init_columns = y_var_lags.shape[1]
    y_var_lags_reduced = y_var_lags.copy()

    if filter_by_feature:
        # drop all columns in y_var_lags that has 'feature' but is not 'feature'
        print('\n====================Dropping features with high VIF for each feature====================')
        y_var_lags_reduced, top_values_by_feature, columns_dropped = drop_high_corr_vars.drop_lags_with_high_corr_or_vif_for_each_feature(
            y_var_lags_reduced,
            vif_threshold=vif_threshold,
            verbose=verbose,
            use_vif_instead_of_corr=True
        )

    if filter_by_subsets:
        print('\n====================Dropping features with high VIF in subsets of features in an iterative manner====================')
        if get_column_subsets_func is not None:
            subset_key_words, all_column_subsets = get_column_subsets_func(
                y_var_lags_reduced)
        else:
            subset_key_words = None
            all_column_subsets = None
        y_var_lags_reduced, columns_dropped = filter_specific_subset_of_y_var_lags_by_vif(
            y_var_lags_reduced, vif_threshold=vif_threshold, verbose=True, subset_key_words=subset_key_words, all_column_subsets=all_column_subsets)

    if filter_by_all_columns:
        print('\n====================Dropping columns with the highest VIF in an iterative manner====================')
        y_var_lags_reduced, columns_dropped_from_y_var_lags_reduced, vif_of_y_var_lags_reduced = take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(
            y_var_lags_reduced, initial_vif=None,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset, vif_threshold=vif_threshold,
            verbose=verbose, get_final_vif=False,
        )

    num_final_columns = y_var_lags_reduced.shape[1]
    print(
        f'\n** Summary: {num_init_columns - num_final_columns} out of {num_init_columns} '
        f'({(num_init_columns - num_final_columns) / num_init_columns * 100:.2f}%) '
        f'are dropped after calling drop_columns_with_high_vif. \n** {num_final_columns} features are left **'
    )

    return y_var_lags_reduced


def get_vif_df(var_df, verbose=True):
    vif_df = pd.DataFrame()
    vif_df["feature"] = var_df.columns
    vif_values = []
    num_total_features = var_df.shape[1]
    if num_total_features > 1:
        for i in range(var_df.shape[1]):
            # check for RuntimeWarning; print the column name that causes the warning
            try:
                vif_values.append(variance_inflation_factor(
                    var_df.values, i))
            except RuntimeWarning as e:
                print(f'RuntimeWarning: {e}')
                print(f'Column {var_df.columns[i]} causes the warning')
            if verbose:
                if num_total_features > 50:
                    if i % 10 == 0:
                        print(
                            f'{i} out of {var_df.shape[1]} features are processed for VIF.')
        vif_df['vif'] = vif_values
        vif_df = vif_df.sort_values(by='vif', ascending=False).round(1)
        return vif_df
    else:
        # if num_total_features == 1:
        #     if verbose:
        #         print(
        #             f'{var_df.columns.values[0]} is the only feature in the dataframe. No VIF to calculate')
        # else:
        #     if verbose:
        #         print('The dataframe is empty. No VIF to calculate')

        vif_df['vif'] = 0
        return vif_df


def check_vif_contribution(df, target_feature, top_n=15, standardize=True):
    """
    Identifies which features contribute most to high VIF for a given feature.

    Parameters:
    - df: DataFrame of features.
    - target_feature: The name of the feature whose VIF contributors you want to check.
    - top_n: Number of top contributing features to display.
    - standardize: Whether to standardize features before analysis.

    Returns:
    - contributions: Series of absolute standardized regression coefficients sorted by importance.
    """
    if standardize:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    else:
        df_scaled = df.copy()

    X_others = df_scaled.drop(columns=[target_feature])
    y_target = df_scaled[target_feature]

    reg = LinearRegression().fit(X_others, y_target)
    contributions = pd.Series(
        reg.coef_, index=X_others.columns).abs().sort_values(ascending=False)

    print(
        f"\nTop {top_n} contributors to multicollinearity for '{target_feature}':")
    print(contributions.head(top_n))

    return contributions


def make_or_retrieve_vif_df(df, data_folder_path, vif_df_name='vif_df', exists_ok=True):
    df_path = os.path.join(data_folder_path, f'{vif_df_name}.csv')
    if exists(df_path) & exists_ok:
        vif_df = pd.read_csv(df_path)
    else:
        vif_df = get_vif_df(df)
        vif_df.to_csv(df_path, index=False)
    return vif_df


# take out a subset of columns that are related. Iteratively calculate its VIF and delete the vars with the highest VIF
def iteratively_drop_column_w_highest_vif(df, vif_threshold=5, verbose=True):
    df = df.copy()
    columns_dropped = []
    vif_df = get_vif_df(df)
    initial_columns = df.columns
    iteration_counter = 0
    while vif_df['vif'].max() > vif_threshold:
        column_to_drop = vif_df['feature'].values[0]
        iteration_counter += 1
        print(
            f'Iter {iteration_counter}: Dropped {column_to_drop} (VIF: {vif_df["vif"].max():.1f})')
        df.drop(columns=column_to_drop, inplace=True)
        columns_dropped.append(column_to_drop)
        vif_df = get_vif_df(df)
    final_vif_df = vif_df
    if len(vif_df) > 0:
        print(
            f'After iterative dropping, the column with the highest VIF of the dataframe or subset is {vif_df["feature"].values[0]} with VIF {vif_df["vif"].max()}')
    else:
        print('After iterative dropping, the dataframe is empty. No columns are dropped.')
    if verbose:
        if len(columns_dropped) > 0:
            # print('Examined columns: ', np.array(initial_columns))
            print('Dropped columns: ', np.array(columns_dropped))
            print('Kept columns: ', np.array(df.columns))
    return df, columns_dropped, final_vif_df


def take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(df,
                                                                          initial_vif=None,
                                                                          vif_threshold_for_initial_subset=5,
                                                                          vif_threshold=5,
                                                                          get_final_vif=True,
                                                                          verbose=True):
    if verbose:
        print(f'Getting VIF for all {df.shape[1]} features...')
    if initial_vif is None:
        vif_df = get_vif_df(df)
    else:
        vif_df = initial_vif
    # after calculating vif, take out the subset of features with vif greater than threshold and iterate over them
    vif_df_subset = vif_df.loc[vif_df['vif']
                               > vif_threshold_for_initial_subset]
    column_subset = vif_df_subset['feature'].values
    if verbose:
        print(
            f"Initial subset of columns with VIF > {vif_threshold_for_initial_subset}")
        print(vif_df_subset)
    _, columns_dropped, _ = iteratively_drop_column_w_highest_vif(
        df[column_subset].copy(), verbose=verbose, vif_threshold=vif_threshold)
    df_reduced = df.drop(columns=columns_dropped)
    print(f'The shape of the reduced dataframe is {df_reduced.shape}')
    if get_final_vif:
        final_vif_df = get_vif_df(df_reduced)
        if verbose:
            print(f"Final number of columns {df_reduced.shape[1]}")
            subset_above_threshold = final_vif_df.loc[final_vif_df['vif']
                                                      > vif_threshold_for_initial_subset]
            print(f"Columns still above threshold: ")
            print(subset_above_threshold)
    else:
        final_vif_df = None
    return df_reduced, columns_dropped, final_vif_df


def filter_specific_subset_of_y_var_lags_by_vif(y_var_lags, vif_threshold=5, verbose=True, subset_key_words=None, all_column_subsets=None):

    if all_column_subsets is None:
        subset_key_words = ['stop', 'speed_or_ddv', 'dw', 'LD_or_RD_or_gaze',
                            'distance', 'angle', 'frozen', 'dummy', 'num_or_any_or_rate']

        all_column_subsets = [
            [col for col in y_var_lags.columns if 'stop' in col],
            [col for col in y_var_lags.columns if (
                'speed' in col) or ('ddv' in col)],
            [col for col in y_var_lags.columns if ('dw' in col)],
            [col for col in y_var_lags.columns if (
                'LD' in col) or ('RD' in col) or ('gaze' in col)],
            [col for col in y_var_lags.columns if ('distance' in col)],
            [col for col in y_var_lags.columns if ('angle' in col)],
            [col for col in y_var_lags.columns if ('frozen' in col)],
            [col for col in y_var_lags.columns if ('dummy' in col)],
            [col for col in y_var_lags.columns if (
                'num' in col) or ('any' in col) or ('rate' in col)],
        ]

    df_reduced, columns_dropped = drop_high_corr_vars.filter_subsets_of_var_df_lags_by_corr_or_vif(
        y_var_lags, use_vif_instead_of_corr=True,
        vif_threshold=vif_threshold, verbose=verbose,
        subset_key_words=subset_key_words,
        all_column_subsets=all_column_subsets
    )

    return df_reduced, columns_dropped
