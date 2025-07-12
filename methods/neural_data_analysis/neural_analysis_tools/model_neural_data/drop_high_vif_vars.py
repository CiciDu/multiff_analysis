import sys
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, drop_high_corr_vars
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
            use_vif_instead_of_corr=True,
            drop_lag_0_last_in_vif=True,
        )

    if filter_by_subsets:
        print('\n====================Among subsets of features, iteratively dropping features with high VIF====================')
        if get_column_subsets_func is not None:
            subset_key_words, all_column_subsets = get_column_subsets_func(
                y_var_lags_reduced)
        else:
            subset_key_words = None
            all_column_subsets = None
        y_var_lags_reduced, columns_dropped = filter_specific_subset_of_y_var_lags_by_vif(
            y_var_lags_reduced, vif_threshold=vif_threshold, verbose=True, subset_key_words=subset_key_words, all_column_subsets=all_column_subsets)

    if filter_by_all_columns:
        print('\n====================Among all columns, iteratively dropping columns with the highest VIF====================')
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


# def get_vif_df(var_df, verbose=True):
#     vif_df = pd.DataFrame()
#     vif_df["feature"] = var_df.columns
#     vif_values = []
#     num_total_features = var_df.shape[1]
#     if num_total_features > 1:
#         for i in range(var_df.shape[1]):
#             # check for RuntimeWarning; print the column name that causes the warning
#             try:
#                 vif_values.append(variance_inflation_factor(
#                     var_df.values, i))
#             # except RuntimeWarning as e:
#             except Exception as e:
#                 print(f'Error: {e}')
#                 print(f'Column {var_df.columns[i]} causes the error')
#             if verbose:
#                 if num_total_features > 50:
#                     if i % 10 == 0:
#                         print(
#                             f'{i} out of {var_df.shape[1]} features are processed for VIF.')
#         vif_df['vif'] = vif_values
#         vif_df = vif_df.sort_values(by='vif', ascending=False).round(1)
#         return vif_df
#     else:

#         vif_df['vif'] = 0
#         return vif_df

def get_vif_df(var_df, verbose=False) -> pd.DataFrame:
    """
    Fast computation of VIFs using correlation matrix inversion.
    Mimics output of get_vif_df but much faster for large feature sets.

    Parameters:
        var_df (pd.DataFrame): DataFrame of numeric predictors.
        verbose (bool): If True, prints warnings for collinearity and errors.

    Returns:
        pd.DataFrame: VIF results with columns ['feature', 'vif'], sorted descending.
    """
    num_features = var_df.shape[1]
    if num_features <= 1:
        if verbose:
            if num_features == 1:
                print(f"Only one feature: '{var_df.columns[0]}'. VIF = 0 by definition.")
            else:
                print("DataFrame is empty. No VIFs to compute.")
        vif_df = pd.DataFrame({'feature': var_df.columns, 'vif': 0})
        return vif_df


    X = var_df.copy()
    X_std = (X - X.mean()) / X.std(ddof=0)

    # Compute correlation matrix
    corr = np.corrcoef(X_std.T)

    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError as e:
        if verbose:
            print("❌ Correlation matrix inversion failed. Using pseudo-inverse.")
            print(f"Error: {e}")
        inv_corr = np.linalg.pinv(corr)

    vif_values = np.diag(inv_corr)

    vif_df = pd.DataFrame({
        'feature': X.columns,
        'vif': np.round(vif_values, 1)
    }).sort_values(by='vif', ascending=False).reset_index(drop=True)

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


def iteratively_drop_column_w_highest_vif(df, vif_threshold=5, drop_lag_0_last=False, verbose=True):
    df = df.copy()
    columns_dropped = []
    vif_df = get_vif_df(df)
    iteration_counter = 0

    while vif_df['vif'].max() > vif_threshold:
        columns_above_threshold = vif_df[vif_df['vif'] > vif_threshold]['feature'].values
        highest_vif_row = vif_df.loc[vif_df['vif'].idxmax()]
        highest_vif_feature = highest_vif_row['feature']
        highest_vif_value = highest_vif_row['vif']

        if drop_lag_0_last:
            non_lag_0_columns_above_threshold = [col for col in columns_above_threshold if '_0' not in col]
            if len(non_lag_0_columns_above_threshold) > 0:
                columns_above_threshold = non_lag_0_columns_above_threshold

        if len(columns_above_threshold) == 0:
            print("No suitable columns left to drop without violating 'drop_lag_0_last' constraint.")
            break

        column_to_drop = columns_above_threshold[0]
        vif_value_to_drop = vif_df[vif_df['feature'] == column_to_drop]['vif'].values[0]
        iteration_counter += 1

        if drop_lag_0_last and highest_vif_feature.endswith('_0') and column_to_drop != highest_vif_feature:
            print(
                f"Iter {iteration_counter}: Dropped {column_to_drop} (VIF: {vif_value_to_drop:.1f}) "
                f"instead of {highest_vif_feature} (VIF: {highest_vif_value:.1f}) "
                f"because drop_lag_0_last=True"
            )
        else:
            print(
                f'Iter {iteration_counter}: Dropped {column_to_drop} (VIF: {vif_value_to_drop:.1f})')

        df.drop(columns=column_to_drop, inplace=True)
        columns_dropped.append(column_to_drop)
        vif_df = get_vif_df(df)

    final_vif_df = vif_df
    if len(vif_df) > 0:
        max_vif_idx = vif_df['vif'].idxmax()
        print(
            f'After iterative dropping, the column with the highest VIF is {vif_df.loc[max_vif_idx, "feature"]} '
            f'with VIF {vif_df.loc[max_vif_idx, "vif"]:.2f}')
    else:
        print('After iterative dropping, the dataframe is empty. No columns are dropped.')

    if verbose and len(columns_dropped) > 0:
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
    _, columns_dropped, final_vif_df = iteratively_drop_column_w_highest_vif(
        df.copy(), verbose=verbose, vif_threshold=vif_threshold)
    df_reduced = df.drop(columns=columns_dropped)
    print(f'The shape of the reduced dataframe is {df_reduced.shape}')

    if verbose:
        print(f"Final number of columns {df_reduced.shape[1]}")
        subset_above_threshold = final_vif_df.loc[final_vif_df['vif']
                                                  > vif_threshold_for_initial_subset]
        if len(subset_above_threshold) > 0:
            print(f"Columns still above threshold: ")
            print(subset_above_threshold)
            
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
