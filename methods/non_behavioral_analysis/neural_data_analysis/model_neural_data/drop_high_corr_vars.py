import sys
from non_behavioral_analysis.neural_data_analysis.model_neural_data import transform_vars, drop_high_vif_vars
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


def drop_columns_with_high_corr(var_df_lags, corr_threshold_for_lags=0.85, verbose=True,
                                filter_by_feature=True,
                                filter_by_subsets=False,
                                filter_by_all_columns=False,
                                get_column_subsets_func=None):

    if (not filter_by_feature) & (not filter_by_subsets) & (not filter_by_all_columns):
        return var_df_lags

    num_init_columns = var_df_lags.shape[1]
    var_df_lags_reduced = var_df_lags.copy()
    # drop all columns in var_df_lags that has 'feature' but is not 'feature'
    if filter_by_feature:
        print('\n====================Dropping features with high correlation for each feature====================')
        var_df_lags_reduced, top_values_by_feature, columns_dropped = drop_lags_with_high_corr_or_vif_for_each_feature(
            var_df_lags_reduced,
            corr_threshold=corr_threshold_for_lags,
            verbose=verbose
        )

    if filter_by_subsets:
        if get_column_subsets_func is not None:
            subset_key_words, all_column_subsets = get_column_subsets_func(
                var_df_lags_reduced)
        else:
            subset_key_words = None
            all_column_subsets = None
        print('====================Dropping features with high correlation in specific subsets of features====================')
        var_df_lags_reduced, columns_dropped = filter_subsets_of_var_df_lags_by_corr_or_vif(
            var_df_lags_reduced, corr_threshold=corr_threshold_for_lags, verbose=verbose, subset_key_words=subset_key_words, all_column_subsets=all_column_subsets)

    if filter_by_all_columns:
        print('====================Dropping features with high correlation in all columns====================')
        var_df_lags_reduced, columns_dropped = filter_subsets_of_var_df_lags_by_corr_or_vif(
            var_df_lags_reduced, corr_threshold=corr_threshold_for_lags, verbose=verbose, all_column_subsets=[var_df_lags_reduced.columns])

    num_final_columns = var_df_lags_reduced.shape[1]
    print(
        f'\nSummary: {num_init_columns - num_final_columns} out of {num_init_columns} '
        f'({(num_init_columns - num_final_columns) / num_init_columns * 100:.2f}%) '
        f'are dropped after calling drop_columns_with_high_corr. \n** {num_final_columns} features are left **'
    )

    return var_df_lags_reduced


def get_base_feature_names(df_with_lags):
    """
    Extract base feature names by removing lag numbers from column names.
    For example, 'feature_1' and 'feature_-1' will both return 'feature'.

    Args:
        df_with_lags (pd.DataFrame): DataFrame with column names that may contain lag numbers

    Returns:
        set: Set of unique base feature names without lag numbers
    """
    base_features = set()
    for col in df_with_lags.columns:
        # Match pattern of feature_name followed by _number or _-number
        match = re.match(r'^(.*?)_-?\d+$', col)
        if match:
            base_features.add(match.group(1))
    return base_features


def _drop_lags_for_feature(df_with_lags, feature, corr_threshold, vif_threshold, use_vif_instead_of_corr):
    """
    Drop lags for a single feature based on correlation or VIF.
    Returns: columns_to_drop, top_values_of_feature
    """
    from non_behavioral_analysis.neural_data_analysis.model_neural_data import transform_vars, drop_high_vif_vars
    df_with_lags_sub = _find_subset_of_df_with_lags_for_current_feature(
        df_with_lags, feature)
    if df_with_lags_sub.shape[1] == 0:
        return [], pd.DataFrame()
    if not use_vif_instead_of_corr:
        high_corr_pair_df, top_values_of_feature = get_pairs_of_columns_w_high_corr(
            df_with_lags_sub, corr_threshold=corr_threshold)
        columns_to_drop = np.unique(high_corr_pair_df['var_1'].values).tolist()
    else:
        _, columns_to_drop, top_values_of_feature = drop_high_vif_vars.iteratively_drop_column_w_highest_vif(
            df_with_lags_sub.copy(), vif_threshold=vif_threshold, verbose=False)
    return columns_to_drop, top_values_of_feature


def _print_dropped_lags(feature, temp_columns_to_drop, df_with_lags_sub, verbose):
    if verbose and len(temp_columns_to_drop) > 0:
        lag_numbers_to_drop = [int(col.split('_')[-1])
                               for col in temp_columns_to_drop]
        lag_numbers_to_drop.sort()
        print(f'{len(temp_columns_to_drop)} columns out of {len(df_with_lags_sub.columns)} of *{feature}* dropped: {lag_numbers_to_drop}')


def drop_lags_with_high_corr_or_vif_for_each_feature(
    df_with_lags,
    corr_threshold=0.85,
    vif_threshold=5,
    verbose=True,
    show_top_values_of_each_feature=False,
    use_vif_instead_of_corr=False
):
    """
    Iteratively drop lags with high correlation or VIF for each feature in the DataFrame.
    Returns:
    - df_reduced: DataFrame with reduced features after dropping highly correlated lags.
    - top_values_by_feature: DataFrame of top values by feature
    - columns_dropped: List of dropped columns
    """
    num_original_columns = len(df_with_lags.columns)
    base_features = get_base_feature_names(df_with_lags)
    columns_dropped = []
    top_values_by_feature = pd.DataFrame()
    for i, feature in enumerate(base_features):
        if i % 10 == 0:
            print(f"Processing feature {i+1}/{len(base_features)}")
        df_with_lags_sub = _find_subset_of_df_with_lags_for_current_feature(
            df_with_lags, feature)
        temp_columns_to_drop, top_values_of_feature = _drop_lags_for_feature(
            df_with_lags, feature, corr_threshold, vif_threshold, use_vif_instead_of_corr)
        if show_top_values_of_each_feature and not top_values_of_feature.empty:
            print(top_values_of_feature.head(3))
        if not top_values_of_feature.empty:
            top_values_of_feature['feature'] = feature
            top_values_by_feature = pd.concat(
                [top_values_by_feature, top_values_of_feature.iloc[[0]]])
        columns_dropped.extend(temp_columns_to_drop)
        _print_dropped_lags(feature, temp_columns_to_drop,
                            df_with_lags_sub, verbose)
    columns_dropped = list(set(columns_dropped))
    df_reduced = df_with_lags.drop(columns=columns_dropped)
    value_to_sort_by = 'vif' if use_vif_instead_of_corr else 'abs_corr'
    if not top_values_by_feature.empty:
        top_values_by_feature = top_values_by_feature.sort_values(
            by=value_to_sort_by, ascending=False)
    corr_or_vif = 'correlation' if not use_vif_instead_of_corr else 'VIF'
    print(
        f"\nDropped {len(columns_dropped)} out of {num_original_columns} columns "
        f"({len(columns_dropped) / num_original_columns * 100:.2f}%) "
        f"after removing lags of features with high {corr_or_vif}.\n"
        f"Dropped columns: {columns_dropped}"
    )
    return df_reduced, top_values_by_feature, columns_dropped


def filter_subsets_of_var_df_lags_by_corr_or_vif(var_df_lags,
                                                 use_vif_instead_of_corr=False,
                                                 corr_threshold=0.9,
                                                 vif_threshold=5,
                                                 verbose=True,
                                                 subset_key_words=None,
                                                 all_column_subsets=None):

    if all_column_subsets is None:
        subset_key_words = ['_x', '_y', 'angle']
        all_column_subsets = [
            [col for col in var_df_lags.columns if '_x' in col],
            [col for col in var_df_lags.columns if '_y' in col],
            [col for col in var_df_lags.columns if 'angle' in col],
        ]

    columns_dropped = []
    num_subsets = len(all_column_subsets)
    num_original_columns = len(var_df_lags.columns)
    for i, column_subset in enumerate(all_column_subsets):
        # now, only keep columns in column_subset that are still in var_df_lags.columns
        column_subset = [
            col for col in column_subset if col not in columns_dropped]

        if verbose:
            if i > 0:
                print('-'*100)

            if subset_key_words is not None:
                print(
                    f'Processing subset {i+1} of {num_subsets} with features that contain "{subset_key_words[i]}", {len(column_subset)} features in total.')
            else:
                print(
                    f'Processing subset {i+1} of {num_subsets}, {len(column_subset)} features in total.')

        if not use_vif_instead_of_corr:
            high_corr_pair_df, top_n_corr_df = get_pairs_of_columns_w_high_corr(
                var_df_lags[column_subset], corr_threshold=corr_threshold)
            temp_columns_to_drop = high_corr_pair_df['var_1'].values.tolist()
        else:
            _, temp_columns_to_drop, _ = drop_high_vif_vars.iteratively_drop_column_w_highest_vif(var_df_lags[column_subset].copy(),
                                                                                                  vif_threshold=vif_threshold, verbose=verbose)

        if len(temp_columns_to_drop) > 0:
            # get unique columns dropped
            temp_columns_to_drop = list(set(temp_columns_to_drop))
            columns_dropped.extend(temp_columns_to_drop)
            if verbose:
                print(
                    f'{len(temp_columns_to_drop)} columns out of {len(column_subset)} dropped: {temp_columns_to_drop}')

    df_reduced = var_df_lags.drop(columns=columns_dropped)
    if verbose:
        if len(columns_dropped) > 0:
            corr_or_vif = 'correlation' if not use_vif_instead_of_corr else 'VIF'
            print(
                f'\n{len(columns_dropped)} out of {num_original_columns} ({len(columns_dropped) / num_original_columns * 100:.2f}%) are dropped after Dropping features with high {corr_or_vif} in subsets of features')

    return df_reduced, columns_dropped


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

    # Find pairs of columns with correlation > corr_threshold
    high_corr_pairs = np.array(np.where(abs_corr > corr_threshold)).T

    # excluding self-correlations
    high_corr_pairs = high_corr_pairs[high_corr_pairs[:, 0]
                                      != high_corr_pairs[:, 1]]

    all_corr = []
    high_cor_var1 = []
    high_cor_var2 = []

    # Print the pairs of columns with high correlation
    # Note: each pair will only appear twice because I gave the condition 'i < j'
    if verbose:
        if len(high_corr_pairs) > 0:
            print(
                f"\nHighly correlated pairs (correlation > {corr_threshold}), {int(len(high_corr_pairs) / 2)} in total:")
    for pair in high_corr_pairs:
        i, j = pair
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
    # high_corr_pair_df.sort_values(by='abs_corr', ascending=False, inplace=True)
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


def _find_subset_of_df_with_lags_for_current_feature(df_with_lags, feature):
    # sort np.array(lag_numbers) by absolute value
    # sorted_lag_numbers = lag_numbers[np.argsort(np.abs(lag_numbers))].tolist()

    column_names_w_lags = [
        col for col in df_with_lags.columns if re.match(rf'^{feature}_-?\d+$', col)]
    column_names_w_lags.sort(key=lambda x: abs(
        int(x.split('_')[-1])), reverse=True)
    # column_names_w_lags = [feature + "_" +
    #                        str(lag) for lag in sorted_lag_numbers]
    df_with_lags_sub = df_with_lags[column_names_w_lags].copy()
    return df_with_lags_sub
