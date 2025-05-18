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
import re


def get_vif_df(var_df, verbose=True):
    vif_df = pd.DataFrame()
    vif_df["feature"] = var_df.columns
    vif_values = []
    num_total_features = var_df.shape[1]
    if num_total_features > 1:
        for i in range(var_df.shape[1]):
            vif_values.append(variance_inflation_factor(
                var_df.values, i))
            if verbose:
                if num_total_features > 20:
                    if i % 10 == 0:
                        print(
                            f'{i} out of {var_df.shape[1]} features are processed for VIF.')
        vif_df['vif'] = vif_values
        vif_df = vif_df.sort_values(by='vif', ascending=False).round(1)
        return vif_df
    else:
        if num_total_features == 1:
            print(
                f'{var_df.columns.values[0]} is the only feature in the dataframe. No VIF to calculate')
        else:
            print('The dataframe is empty. No VIF to calculate')
        vif_df['vif'] = 0
        return vif_df


def make_or_retrieve_vif_df(df, data_folder_path, vif_df_name='vif_df', exists_ok=True):
    df_path = os.path.join(data_folder_path, f'{vif_df_name}.csv')
    if exists(df_path) & exists_ok:
        vif_df = pd.read_csv(df_path)
    else:
        vif_df = get_vif_df(df)
        vif_df.to_csv(df_path, index=False)
    return vif_df


def drop_columns_with_high_vif(y_var, y_var_lags, vif_threshold=5, vif_threshold_for_initial_subset=5, verbose=True):

    num_init_columns = y_var_lags.shape[1]
    # drop all columns in y_var_lags that has 'feature' but is not 'feature'
    print('\n====================Dropping lags of features with high VIF for each feature====================')
    y_var_lags_reduced, top_values_by_feature, columns_dropped = drop_lags_with_high_corr_or_vif_for_each_feature(
        y_var, y_var_lags,
        vif_threshold=vif_threshold,
        verbose=verbose,
        use_vif_instead_of_corr=True
    )
    
    print('\n====================Dropping lags with high VIF in a specific subset of features====================')
    y_var_lags_reduced, dropped_columns = filter_specific_subset_of_y_var_lags_by_vif(
        y_var_lags_reduced, vif_threshold=vif_threshold, verbose=True)

    print('\n====================Iteratively dropping columns with the highest VIF====================')
    y_var_lags_reduced, dropped_columns_from_y_var_lags_reduced, vif_of_y_var_lags_reduced = take_out_subset_of_high_vif_and_iteratively_drop_column_w_highest_vif(
        y_var_lags_reduced, initial_vif=None,
        vif_threshold_for_initial_subset=vif_threshold_for_initial_subset, vif_threshold=vif_threshold,
        verbose=verbose, get_final_vif=False,
    )
    
    num_final_columns = y_var_lags_reduced.shape[1]
    print(f'\n*Summary: {num_init_columns - num_final_columns} out of {num_init_columns} ({num_init_columns - num_final_columns / num_init_columns * 100:.2f}%) are dropped after calling drop_columns_with_high_vif*')
    
    return y_var_lags_reduced
    
        

def drop_columns_with_high_corr(y_var, y_var_lags, corr_threshold_for_lags=0.85, verbose=True):

    num_init_columns = y_var_lags.shape[1]
    # drop all columns in y_var_lags that has 'feature' but is not 'feature'
    print('\n====================Dropping lags of features with high correlation for each feature====================')
    y_var_lags_reduced, top_values_by_feature, columns_dropped = drop_lags_with_high_corr_or_vif_for_each_feature(
        y_var, y_var_lags,
        corr_threshold=corr_threshold_for_lags,
        verbose=verbose
    )

    print('====================Dropping lags of features with high correlation in specific subsets of features====================')
    y_var_lags_reduced, columns_dropped = filter_subsets_of_y_var_lags_by_corr(
        y_var_lags_reduced, corr_threshold=corr_threshold_for_lags, verbose=verbose)

    print('====================Dropping lags of features with high correlation in all columns====================')
    y_var_lags_reduced, columns_dropped = filter_subsets_of_y_var_lags_by_corr(
        y_var_lags_reduced, corr_threshold=corr_threshold_for_lags, verbose=verbose, all_column_subsets=[y_var_lags_reduced.columns])

    num_final_columns = y_var_lags_reduced.shape[1]
    print(f'\n*Summary: {num_init_columns - num_final_columns} out of {num_init_columns} ({num_init_columns - num_final_columns / num_init_columns * 100:.2f}%) are dropped after calling drop_columns_with_high_corr*')

    return y_var_lags_reduced


def drop_lags_with_high_corr_or_vif_for_each_feature(df, df_with_lags,
                                                     corr_threshold=0.85,
                                                     vif_threshold=10,
                                                     verbose=True,
                                                     show_top_values_of_each_feature=False,
                                                     use_vif_instead_of_corr=False):
    """
    Iteratively drop lags with high correlation for each feature in the DataFrame.

    Returns:
    - df_reduced: DataFrame with reduced features after dropping highly correlated lags.
    """
    columns_dropped = []
    num_original_columns = len(df_with_lags.columns)
    top_values_by_feature = pd.DataFrame()

    for i, feature in enumerate(df.columns):
        if i % 10 == 0:
            print(f"Processing feature {i+1}/{len(df.columns)}")

        feature_columns_to_drop = []

        # Get the subset of df_with_lags for the current feature
        df_with_lags_sub = _find_subset_of_df_with_lags_for_current_feature(
            df_with_lags, feature)


        if df_with_lags_sub.shape[1] == 0: 
            print(f'No lags for feature {feature} found. Skipping...')
            continue
        
        if not use_vif_instead_of_corr:
            # Find features with correlation above the threshold
            high_corr_pair_df, top_values_of_feature = get_pairs_of_columns_w_high_corr(
                df_with_lags_sub, corr_threshold=corr_threshold)

            # Drop features with correlation greater than the threshold
            temp_columns_to_drop = high_corr_pair_df['var_1'].values.tolist()

            if show_top_values_of_each_feature:
                # print top pairs of df_with_lags_sub
                print(top_values_by_feature.head(3))

        else:
            _, temp_columns_to_drop, top_values_of_feature = iteratively_drop_column_w_highest_vif(df_with_lags_sub.copy(),
                                                                                                   vif_threshold=vif_threshold, 
                                                                                                   verbose=False)

        top_values_of_feature['feature'] = feature
        top_values_by_feature = pd.concat(
            [top_values_by_feature, top_values_of_feature.iloc[[0]]])

        # Save the names of all the columns that were dropped
        columns_dropped.extend(temp_columns_to_drop)
        feature_columns_to_drop.extend(temp_columns_to_drop)

        # print number of iteration and name of feature, as well as columns to drop, both number and names
        if verbose:
            if len(feature_columns_to_drop) > 0:
                # print(f'Feature {i+1} of {len(df.columns)}: {feature} dropped {len(feature_columns_to_drop)} columns: {feature_columns_to_drop}')
                print(
                    f'{len(feature_columns_to_drop)} columns of {feature} dropped: {feature_columns_to_drop}')

    # Drop highly correlated features from the original DataFrame
    df_reduced = df_with_lags.drop(columns=columns_dropped)

    value_to_sort_by = 'vif' if use_vif_instead_of_corr else 'abs_corr'

    top_values_by_feature = top_values_by_feature.sort_values(
        by=value_to_sort_by, ascending=False)

    # print the total number of columns dropped
    print(
        f'\n{len(columns_dropped)} out of {num_original_columns} ({len(columns_dropped) / num_original_columns * 100:.2f}%) are dropped after calling drop_lags_with_high_corr_or_vif_for_each_feature')

    return df_reduced, top_values_by_feature, columns_dropped


def filter_subsets_of_y_var_lags_by_corr(y_var_lags, corr_threshold=0.9, verbose=True, all_column_subsets=None):
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

        high_corr_pair_df, top_n_corr_df = get_pairs_of_columns_w_high_corr(
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
                f'\n{len(columns_dropped)} out of {num_original_columns} ({len(columns_dropped) / num_original_columns * 100:.2f}%) are dropped after calling filter_subsets_of_y_var_lags_by_corr')

    return df_reduced, columns_dropped


# take out a subset of columns that are related. Iteratively calculate its VIF and delete the vars with the highest VIF
def iteratively_drop_column_w_highest_vif(df, vif_threshold=5, verbose=True):
    df = df.copy()
    dropped_columns = []
    vif_df = get_vif_df(df)
    initial_columns = df.columns
    while vif_df['vif'].max() > vif_threshold:
        column_to_drop = vif_df['feature'][0]
        df.drop(columns=column_to_drop, inplace=True)
        dropped_columns.append(column_to_drop)
        vif_df = get_vif_df(df)
    final_vif_df = vif_df
    if verbose:
        if len(dropped_columns) > 0:
            #print('Examined columns: ', np.array(initial_columns))
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
    df_reduced = df.drop(columns=dropped_columns)
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


def _find_subset_of_df_with_lags_for_current_feature(df_with_lags, feature):
    # sort np.array(lag_numbers) by absolute value
    # sorted_lag_numbers = lag_numbers[np.argsort(np.abs(lag_numbers))].tolist()

    column_names_w_lags = [
        col for col in df_with_lags.columns if re.match(rf'^{feature}_-?\d+$', col)]
    column_names_w_lags.sort(key=lambda x: abs(int(x.split('_')[-1])))
    # column_names_w_lags = [feature + "_" +
    #                        str(lag) for lag in sorted_lag_numbers]
    df_with_lags_sub = df_with_lags[column_names_w_lags].copy()
    return df_with_lags_sub


def get_pairs_of_columns_w_high_corr(df, corr_threshold=0.9, verbose=False):
    # Get absolute correlation values
    corr_coeff = df.corr()
    abs_corr = np.abs(corr_coeff)

    high_corr_pair_df = get_high_corr_pair_df(
        corr_coeff, corr_threshold=corr_threshold, verbose=verbose)
    top_n_corr_df = get_top_n_corr_df(abs_corr)

    return high_corr_pair_df, top_n_corr_df


def get_high_corr_pair_df(corr_coeff, corr_threshold=0.9, verbose=False):
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


def get_top_n_corr_df(abs_corr, n_top=5):
    # get top N correlations from abs_corr, excluding diagonal and using only upper triangular

    # Create a mask for upper triangular matrix (excluding diagonal)
    mask = np.triu(np.ones_like(abs_corr), k=1).astype(bool)

    # Get values and indices from upper triangular matrix
    upper_tri_values = abs_corr.values[mask]
    upper_tri_indices = np.where(mask)

    # Get indices of top N values in descending order
    top_n_indices = np.argsort(upper_tri_values)[-n_top:][::-1]

    # Get corresponding row and column indices
    rows = upper_tri_indices[0][top_n_indices]
    cols = upper_tri_indices[1][top_n_indices]

    # create dataframe with top N correlations
    top_n_corr_df = pd.DataFrame({
        'var_1': [abs_corr.index[r] for r in rows],
        'var_2': [abs_corr.columns[c] for c in cols],
        'corr': [upper_tri_values[i] for i in top_n_indices]
    })

    top_n_corr_df['abs_corr'] = top_n_corr_df['corr'].apply(abs)

    return top_n_corr_df
