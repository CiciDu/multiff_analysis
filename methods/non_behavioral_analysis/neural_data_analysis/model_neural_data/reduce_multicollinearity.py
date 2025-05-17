import sys
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


# def get_vif_df(df):
#     vif_df = pd.DataFrame({
#         'vif': [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
#         "var": df.columns
#     }).sort_values(by='vif', ascending=False).round(1).reset_index(drop=True)
#     return vif_df


def get_vif_df(var_df, verbose=True):
    vif_df = pd.DataFrame()
    vif_df["feature"] = var_df.columns
    vif_values = []
    for i in range(var_df.shape[1]):
        vif_values.append(variance_inflation_factor(
            var_df.values, i))
        if verbose:
            if i % 10 == 0:
                print(
                    f'{i} out of {var_df.shape[1]} features are processed for VIF.')
    vif_df['vif'] = vif_values
    vif_df = vif_df.sort_values(by='vif', ascending=False).round(1)
    return vif_df


def make_or_retrieve_vif_df(df, data_folder_path, vif_df_name='vif_df', exists_ok=True):
    df_path = os.path.join(data_folder_path, f'{vif_df_name}.csv')
    if exists(df_path) & exists_ok:
        vif_df = pd.read_csv(df_path)
    else:
        vif_df = get_vif_df(df)
        vif_df.to_csv(df_path, index=False)
    return vif_df


def drop_columns_with_high_corr(y_var, y_var_lags, lag_numbers=np.array(range(-3, 4)), corr_threshold_for_lags=0.85, verbose=True):

    num_init_columns = y_var_lags.shape[1]
    # drop all columns in y_var_lags that has 'feature' but is not 'feature'
    print('\n====================Dropping lags of features with high correlation for each feature====================')
    y_var_lags_reduced0, columns_dropped = drop_lags_with_high_corr_for_each_feature(
        y_var, y_var_lags, lag_numbers=lag_numbers,
        corr_threshold=corr_threshold_for_lags,
        verbose=verbose
    )

    print('====================Dropping lags of features with high correlation in specific subsets of features====================')
    y_var_lags_reduced0, columns_dropped = filter_specific_subset_of_y_var_lags_by_corr(
        y_var_lags_reduced0, corr_threshold=corr_threshold_for_lags, verbose=verbose)

    print('====================Dropping lags of features with high correlation in all columns====================')
    y_var_lags_reduced0, columns_dropped = filter_specific_subset_of_y_var_lags_by_corr(
        y_var_lags_reduced0, corr_threshold=corr_threshold_for_lags, verbose=verbose, all_column_subsets=[y_var_lags_reduced0.columns])

    num_final_columns = y_var_lags_reduced0.shape[1]
    print(f'\n*Summary: {num_init_columns - num_final_columns} out of {num_init_columns} ({num_init_columns - num_final_columns / num_init_columns * 100:.2f}%) are dropped after calling drop_columns_with_high_corr*')

    return y_var_lags_reduced0


def drop_lags_with_high_corr_for_each_feature(df, df_with_lags, corr_threshold=0.85,
                                              lag_numbers=np.array(range(-3, 4)), verbose=True):
    """
    Iteratively drop lags with high correlation for each feature in the DataFrame.

    Returns:
    - df_reduced: DataFrame with reduced features after dropping highly correlated lags.
    """
    columns_dropped = []
    num_original_columns = len(df_with_lags.columns)

    for i, feature in enumerate(df.columns):
        if i % 10 == 0:
            print(f"Processing feature {i+1}/{len(df.columns)}")

        feature_columns_to_drop = []

        # Get the subset of df_with_lags for the current feature
        df_with_lags_sub = _find_subset_of_df_with_lags_for_current_feature(
            df_with_lags, feature, lag_numbers=lag_numbers)

        while len(df_with_lags_sub.columns) > 1:
            # Find features with correlation above the threshold
            high_corr_pair_df = get_pairs_of_high_corr_features(
                df_with_lags_sub, corr_threshold=corr_threshold)

            # Drop features with correlation greater than the threshold
            temp_columns_to_drop = high_corr_pair_df['var_1'].values.tolist()
            df_with_lags_sub.drop(columns=temp_columns_to_drop, inplace=True)

            # Save the names of all the columns that were dropped
            columns_dropped.extend(temp_columns_to_drop)
            feature_columns_to_drop.extend(temp_columns_to_drop)

            # Drop the first column of the subset
            df_with_lags_sub.drop(
                columns=df_with_lags_sub.columns[0], inplace=True)

        # print number of iteration and name of feature, as well as columns to drop, both number and names
        if verbose:
            if len(feature_columns_to_drop) > 0:
                # print(f'Feature {i+1} of {len(df.columns)}: {feature} dropped {len(feature_columns_to_drop)} columns: {feature_columns_to_drop}')
                print(
                    f'{len(feature_columns_to_drop)} columns of {feature} dropped: {feature_columns_to_drop}')

    # Drop highly correlated features from the original DataFrame
    df_reduced = df_with_lags.drop(columns=columns_dropped)

    # print the total number of columns dropped
    print(
        f'\n{len(columns_dropped)} out of {num_original_columns} ({len(columns_dropped) / num_original_columns * 100:.2f}%) are dropped after calling drop_lags_with_high_corr_for_each_feature')

    return df_reduced, columns_dropped


def filter_specific_subset_of_y_var_lags_by_corr(y_var_lags, corr_threshold=0.9, verbose=True, all_column_subsets=None):
    subset_features = ['_x', '_y', 'angle']
    if all_column_subsets is None:
        all_column_subsets = [
            [col for col in y_var_lags.columns if '_x' in col],
            [col for col in y_var_lags.columns if '_y' in col],
            [col for col in y_var_lags.columns if 'angle' in col],
        ]

    columns_dropped = []
    num_subsets = len(all_column_subsets)
    num_original_columns = len(y_var_lags.columns)
    for i, column_subset in enumerate(all_column_subsets):

        high_corr_pair_df = get_pairs_of_high_corr_features(
            y_var_lags[column_subset], corr_threshold=corr_threshold)
        features_above_corr_threshold = high_corr_pair_df['var_1'].values

        if verbose:
            if len(features_above_corr_threshold) > 0:
                if i != 0:
                    print('')
                print(
                    f'Processing subset {i+1} of {num_subsets} with features that contain "{subset_features[i]}":')

        temp_columns_to_drop = features_above_corr_threshold.tolist()
        if len(temp_columns_to_drop) > 0:
            columns_dropped.extend(temp_columns_to_drop)
            if verbose:
                print(
                    f'{len(temp_columns_to_drop)} columns of out of {len(column_subset)} dropped: {temp_columns_to_drop}')

    df_reduced = y_var_lags.drop(columns=columns_dropped)
    if verbose:
        if len(columns_dropped) > 0:
            print(
                f'\n{len(columns_dropped)} out of {num_original_columns} ({len(columns_dropped) / num_original_columns * 100:.2f}%) are dropped after calling filter_specific_subset_of_y_var_lags_by_corr')

    return df_reduced, columns_dropped


# take out a subset of columns that are related. Iteratively calculate its VIF and delete the vars with the highest VIF
def iteratively_drop_column_w_highest_vif(df, vif_threshold=5, verbose=True):
    df = df.copy()
    dropped_columns = []
    vif_df = get_vif_df(df)
    if verbose:
        print('Examined columns: ', np.array(df.columns))
    while vif_df['vif'].max() > vif_threshold:
        column_to_drop = vif_df['feature'][0]
        df.drop(columns=column_to_drop, inplace=True)
        dropped_columns.append(column_to_drop)
        vif_df = get_vif_df(df)
    final_vif_df = vif_df
    if verbose:
        print('Dropped columns: ', np.array(dropped_columns))
        print('Kept columns: ', np.array(df.columns))
    return df, dropped_columns, final_vif_df


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
    column_subset = vif_df_subset['var'].values
    if verbose:
        print(
            f"Initial subset of columns with VIF > {vif_threshold_for_initial_subset}")
        print(vif_df_subset)
    _, dropped_columns, _ = iteratively_drop_column_w_highest_vif(
        df[column_subset].copy(), verbose=verbose, vif_threshold=vif_threshold)
    columns_to_keep = [col for col in df.columns if col not in dropped_columns]
    df_reduced = df[columns_to_keep].copy()
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
    return df_reduced, dropped_columns, final_vif_df


def filter_specific_subset_of_y_var_lags_by_vif(y_var_lags, vif_threshold=5, verbose=True):
    if verbose:
        print(f'The shape of the initial dataframe is {y_var_lags.shape}')
    dropped_columns = []
    df_reduced = y_var_lags.copy()
    all_column_subsets = [
        [col for col in df_reduced.columns if 'stop' in col],
        [col for col in df_reduced.columns if (
            'speed' in col) or ('ddv' in col)],
        [col for col in df_reduced.columns if ('dw' in col)],
        [col for col in df_reduced.columns if (
            'LD' in col) or ('RD' in col) or ('gaze' in col)],
        [col for col in df_reduced.columns if ('distance' in col)],
        [col for col in df_reduced.columns if ('angle' in col)],
        [col for col in df_reduced.columns if ('frozen' in col)],
        [col for col in df_reduced.columns if ('dummy' in col)],
        [col for col in df_reduced.columns if (
            'num' in col) or ('any' in col) or ('rate' in col)],
    ]
    num_subsets = len(all_column_subsets)
    for i, column_subset in enumerate(all_column_subsets):
        if verbose:
            print(f'\nProcessing subset {i+1} of {num_subsets}...')
        _, temp_dropped_columns, _ = iteratively_drop_column_w_highest_vif(df_reduced[column_subset].copy(),
                                                                           vif_threshold=vif_threshold, verbose=verbose)
        dropped_columns.extend(temp_dropped_columns)

    columns_to_keep = [
        col for col in df_reduced.columns if col not in dropped_columns]
    df_reduced = df_reduced[columns_to_keep].copy()
    if verbose:
        print(f'The shape of the reduced dataframe is {df_reduced.shape}')
    return df_reduced, dropped_columns


def _find_subset_of_df_with_lags_for_current_feature(df_with_lags, feature, lag_numbers=np.array(range(-3, 4))):
    # sort np.array(lag_numbers) by absolute value
    sorted_lag_numbers = lag_numbers[np.argsort(np.abs(lag_numbers))].tolist()
    column_names_w_lags = [feature + "_" +
                           str(lag) for lag in sorted_lag_numbers]
    df_with_lags_sub = df_with_lags[column_names_w_lags].copy()
    return df_with_lags_sub


def get_pairs_of_high_corr_features(df, corr_threshold=0.9, verbose=False):
    # Get absolute correlation values
    corr_coeff = df.corr()
    abs_corr = np.abs(corr_coeff)

    # Find pairs of columns with correlation > 0.9 (excluding self-correlations)
    high_corr_pairs = np.where((abs_corr > corr_threshold) & (abs_corr < 1.0))

    all_corr = []
    high_cor_var1 = []
    high_cor_var2 = []

    # Print the pairs of columns with high correlation
    # Note: each pair will only appear twice because I gave the condition 'i < j'
    if verbose:
        if len(high_corr_pairs[0]) > 0:
            print(
                f"\nHighly correlated pairs (correlation > {corr_threshold}), {int(len(high_corr_pairs[0]) / 2)} in total:")
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        if i < j:  # Only print each pair once
            col1 = corr_coeff.index[i]
            col2 = corr_coeff.columns[j]
            correlation = corr_coeff.iloc[i, j]

            high_cor_var1.append(col1)
            high_cor_var2.append(col2)
            all_corr.append(correlation)

            if verbose:
                print(f"{col1} -- {col2}: {correlation:.3f}")

    # Keep only the highly correlated rows and columns
    high_corr_pair_df = pd.DataFrame({'var_1': high_cor_var1,
                                     'var_2': high_cor_var2,
                                      'corr': all_corr})

    high_corr_pair_df['abs_corr'] = high_corr_pair_df['corr'].apply(abs)
    high_corr_pair_df.sort_values(by='abs_corr', ascending=False, inplace=True)

    return high_corr_pair_df
