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


def get_vif_df(df):
    vif_df = pd.DataFrame({
        "vif": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
        "var": df.columns
    }).sort_values(by='vif', ascending=False).round(1).reset_index(drop=True)
    return vif_df

    
def make_or_retrieve_vif_df(df, data_folder_path, vif_df_name='vif_df', exists_ok=True):
    df_path = os.path.join(data_folder_path, f'{vif_df_name}.csv')
    if exists(df_path) & exists_ok:
        vif_df = pd.read_csv(df_path)
    else:
        vif_df = get_vif_df(df)
        vif_df.to_csv(df_path, index=False)
    return vif_df

# get the 2nd maximum value in the correlation matrix
def get_second_largest_abs_r_for_each_feature(corr_matrix):
    # Function to get the second maximum value in a row
    def second_max_row(row):
        return row.nlargest(2).iloc[-1]

    # Apply the function to each row
    features_2nd_largest_r = corr_matrix.apply(second_max_row, axis=1)
    features_2nd_largest_r.sort_values(ascending=False, inplace=True)
    features_2nd_largest_r = pd.DataFrame(features_2nd_largest_r, columns=['2nd_largest_r'])
    features_2nd_largest_r['2nd_largest_r'] = np.round(features_2nd_largest_r['2nd_largest_r'], 4)
    return features_2nd_largest_r

def drop_features_with_high_correlation(df, corr_threshold=0.9):
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(np.abs(upper[column]) > corr_threshold)]

    # Drop highly correlated features
    df_reduced = df.drop(columns=to_drop)

    features_2nd_largest_r = get_second_largest_abs_r_for_each_feature(corr_matrix)

    dropped_columns_2nd_largest_r = features_2nd_largest_r.loc[to_drop]
    dropped_columns_2nd_largest_r.sort_values(by='2nd_largest_r', ascending=False, inplace=True)
    return df_reduced, dropped_columns_2nd_largest_r


def iteratively_drop_lags_with_high_corr_for_each_feature(df, df_with_lags, corr_threshold=0.85, lag_numbers=np.array(range(-3, 4))):
    """
    Iteratively drop lags with high correlation for each feature in the DataFrame.

    Returns:
    - df_reduced: DataFrame with reduced features after dropping highly correlated lags.
    - all_r_of_dropped_features: DataFrame containing the correlation values of the dropped features.
    """
    columns_to_drop = []
    all_r_of_dropped_features = pd.DataFrame([])


    for i, feature in enumerate(df.columns):
        if i % 10 == 0:
            print(f"Processing feature {i+1}/{len(df.columns)}")

        # Get the subset of df_with_lags for the current feature
        df_with_lags_sub = _find_df_with_lags_sub_for_current_feature(df_with_lags, feature, lag_numbers=lag_numbers)

        while len(df_with_lags_sub.columns) > 1:
            # Find features with correlation above the threshold
            features_above_corr_threshold = _find_features_above_corr_threshold(df_with_lags_sub, corr_threshold=corr_threshold)

            # Drop features with correlation greater than the threshold
            temp_columns_to_drop = features_above_corr_threshold.index.tolist()
            df_with_lags_sub.drop(columns=temp_columns_to_drop, inplace=True)

            # Save the names of all the columns that were dropped
            columns_to_drop.extend(temp_columns_to_drop)

            # Drop the first column of the subset
            df_with_lags_sub.drop(columns=df_with_lags_sub.columns[0], inplace=True)

            # Get the correlation values for the dropped features
            r_of_dropped_features = _put_r_of_dropped_vars_into_df(features_above_corr_threshold)
            r_of_dropped_features['dropped_var_name_no_lag'] = feature
            all_r_of_dropped_features = pd.concat([all_r_of_dropped_features, r_of_dropped_features], axis=0)

    # Drop highly correlated features from the original DataFrame
    df_reduced = df_with_lags.drop(columns=columns_to_drop)

    # Sort the DataFrame containing the correlation values of the dropped features
    all_r_of_dropped_features = all_r_of_dropped_features.sort_values(by='r_value', ascending=False).reset_index(drop=True)

    return df_reduced, all_r_of_dropped_features


# take out a subset of columns that are related. Iteratively calculate its VIF and delete the vars with the highest VIF
def iteratively_drop_column_w_highest_vif(df, vif_threshold=5, verbose=True):
    df = df.copy()
    dropped_columns = []
    vif_df = get_vif_df(df)
    if verbose:
        print('Examined columns: ', np.array(df.columns))
    while vif_df['vif'].max() > vif_threshold:
        column_to_drop = vif_df['var'][0]
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
    vif_df_subset = vif_df.loc[vif_df['vif'] > vif_threshold_for_initial_subset]
    column_subset = vif_df_subset['var'].values
    if verbose:
        print(f"Initial subset of columns with VIF > {vif_threshold_for_initial_subset}")
        print(vif_df_subset)
    _, dropped_columns, _ = iteratively_drop_column_w_highest_vif(df[column_subset].copy(), verbose=verbose, vif_threshold=vif_threshold)
    columns_to_keep = [col for col in df.columns if col not in dropped_columns]
    df_reduced = df[columns_to_keep].copy()
    print(f'The shape of the reduced dataframe is {df_reduced.shape}')
    if get_final_vif:
        final_vif_df = get_vif_df(df_reduced)
        if verbose:
            print(f"Final number of columns {df_reduced.shape[1]}")
            subset_above_threshold = final_vif_df.loc[final_vif_df['vif'] > vif_threshold_for_initial_subset]
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
    for column_subset in [[col for col in df_reduced.columns if 'stop' in col],
                        [col for col in df_reduced.columns if ('speed' in col) or ('ddv' in col)],
                        [col for col in df_reduced.columns if ('dw' in col)],
                        [col for col in df_reduced.columns if ('LD' in col) or ('RD' in col) or ('gaze' in col)],
                        [col for col in df_reduced.columns if ('distance' in col)],
                        [col for col in df_reduced.columns if ('angle' in col)],
                        [col for col in df_reduced.columns if ('frozen' in col)],
                        [col for col in df_reduced.columns if ('dummy' in col)],
                        [col for col in df_reduced.columns if ('num' in col) or ('any' in col) or ('rate' in col)],
                        ]:
        _, temp_dropped_columns, _ = iteratively_drop_column_w_highest_vif(df_reduced[column_subset].copy(), 
                                                                           vif_threshold=vif_threshold, verbose=verbose)
        dropped_columns.extend(temp_dropped_columns)
    
    columns_to_keep = [col for col in df_reduced.columns if col not in dropped_columns]
    df_reduced = df_reduced[columns_to_keep].copy()
    if verbose:
        print(f'The shape of the reduced dataframe is {df_reduced.shape}')
    return df_reduced, dropped_columns

def _find_df_with_lags_sub_for_current_feature(df_with_lags, feature, lag_numbers=np.array(range(-3, 4))):
    # sort np.array(lag_numbers) by absolute value
    sorted_lag_numbers = lag_numbers[np.argsort(np.abs(lag_numbers))].tolist()
    column_names_w_lags = [feature + "_" + str(lag) for lag in sorted_lag_numbers]
    df_with_lags_sub = df_with_lags[column_names_w_lags].copy()
    return df_with_lags_sub

def _find_features_above_corr_threshold(df, corr_threshold=0.9):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Take out the first row of the upper triangle; find features with correlation greater than 0.9 and add them to columns_to_drop
    features_above_corr_threshold = upper.iloc[0][upper.iloc[0] > corr_threshold]
    return features_above_corr_threshold

def _put_r_of_dropped_vars_into_df(features_above_corr_threshold):
    other_var = features_above_corr_threshold.name
    r_df = pd.DataFrame(features_above_corr_threshold).reset_index().rename(columns={'index': 'dropped_var', 
                                                                                    other_var: 'r_value'})
    r_df['other_var'] = other_var
    return r_df
