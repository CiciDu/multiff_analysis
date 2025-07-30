from sklearn.model_selection import KFold, GroupKFold
import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_decomposition import CCA
import rcca
from sklearn.preprocessing import StandardScaler
from palettable.colorbrewer import qualitative
from sklearn.model_selection import KFold


def combine_cca_cv_results(cca_no_lag, cca_lags, n_components=7, reg=0.1, n_splits=10,
                           groups=None, random_state=42):

    combined_cross_view_df = pd.DataFrame()
    combined_can_load_df = pd.DataFrame()

    for whether_lag, _cca_inst in [('no_lag', cca_no_lag), ('lag', cca_lags)]:
        X1_df = _cca_inst.X1_sc_df
        X2_df = _cca_inst.X2_sc_df
        # X2_df = _cca_inst.X2_tf_df
        can_load_df, cross_view_df, can_corr_stats = crossvalidated_cca_analysis(
            X1_df, X2_df, n_components=n_components, reg=reg, n_splits=n_splits,
            groups=groups, random_state=random_state)

        cross_view_df['whether_lag'] = whether_lag
        can_load_df['whether_lag'] = whether_lag

        combined_cross_view_df = pd.concat(
            [combined_cross_view_df, cross_view_df])
        combined_can_load_df = pd.concat([combined_can_load_df, can_load_df])

    # rename all _0 to the original name
    combined_cross_view_df['variable'] = conditional_replace_suffix(
        combined_cross_view_df['variable'])
    combined_cross_view_df = combined_cross_view_df.sort_values(
        by='variable').reset_index(drop=True)

    combined_can_load_df['variable'] = conditional_replace_suffix(
        combined_can_load_df['variable'])
    combined_can_load_df = combined_can_load_df.sort_values(
        by='variable').reset_index(drop=True)

    return combined_cross_view_df, combined_can_load_df


def convert_stats_dict_to_df(stats_dict):
    """
    Convert stats_dict dictionary into a pandas DataFrame.

    Parameters:
    -----------
    stats_dict : dict
        Dictionary containing cross-view correlation statistics from crossvalidated_cca_analysis

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: dataset, train_or_test, mean_corr, std_corr, variable, canon_component
    """
    # Extract the data
    datasets = ['X1', 'X2']
    train_test_sets = ['train', 'test']

    # Create all combinations of dataset and train_test
    combinations = [(dataset, train_test)
                    for dataset in datasets for train_test in train_test_sets]

    # Process each combination
    dfs = []

    for dataset, train_test in combinations:
        mean_key = f'mean_{dataset}_{train_test}_corr'
        std_key = f'std_{dataset}_{train_test}_corr'

        if mean_key in stats_dict:
            mean_corrs = stats_dict[mean_key]

            # Get variable labels if available
            labels_key = f'{dataset}_labels'
            if labels_key in stats_dict:
                variables = stats_dict[labels_key]
            else:
                variables = [f'{dataset}_var_{i}' for i in range(
                    mean_corrs.shape[0])]

            if mean_corrs.ndim == 1:
                # Cross-view correlations (already flattened)
                df = pd.DataFrame({
                    'dataset': dataset,
                    'train_or_test': train_test,
                    'mean_corr': mean_corrs,
                    'variable': variables,
                })

            else:
                # Canonical correlations (2D: variables x components)
                num_vars, num_cc = mean_corrs.shape

                # Create all combinations of variables and components
                var_indices = np.repeat(range(num_vars), num_cc)
                comp_indices = np.tile(range(num_cc), num_vars) + 1

                df = pd.DataFrame({
                    'dataset': dataset,
                    'train_or_test': train_test,
                    'mean_corr': mean_corrs.flatten(),
                    'variable': np.array(variables)[var_indices],
                    'canonical_component': comp_indices
                })

            if std_key in stats_dict:
                std_corrs = stats_dict[std_key]
                df['std_corr'] = std_corrs.flatten()

            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    stats_df = pd.concat(dfs, ignore_index=True)

    return stats_df


def conditional_replace_suffix(variable_series):
    """
    Replace '_0' suffix only if the variable doesn't end with '_0_0'.

    Parameters:
    -----------
    variable_series : pd.Series
        Series of variable names

    Returns:
    --------
    pd.Series
        Series with conditional suffix replacement
    """
    def replace_if_not_double_zero(var):
        if var.endswith('_0_0'):
            return var
        elif var.endswith('_0'):
            return var[:-2]  # Remove last 2 characters ('_0')
        else:
            return var

    return variable_series.apply(replace_if_not_double_zero)


def crossvalidated_cca_analysis(
    X1_df, X2_df, n_components=10, reg=0.1, n_splits=5,
    random_state=42, groups=None,
):
    """
    Cross-validated CCA: supports KFold or GroupKFold (via `group_by_segment` flag).

    Parameters:
        X1_df, X2_df : pd.DataFrame
            Input feature DataFrames (must contain `segment_column` if group_by_segment=True)
        n_components : int
            Number of canonical components
        reg : float
            Regularization for rcca
        n_splits : int
            Number of CV folds
        random_state : int
            For reproducibility (only used in KFold)
        group_by_segment : bool
            Whether to use GroupKFold based on the `segment_column`
        segment_column : str
            Column name to group by (only required if group_by_segment=True)

    Returns:
        can_load_df, cross_view_df, can_corr_stats
    """
    # Handle group splits if requested
    if groups is not None:
        assert len(groups) == len(X1_df)
        splitter = GroupKFold(n_splits=n_splits, shuffle=True)
        print('Cross validation is conducted based on groups.')
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    X1 = X1_df.values
    X2 = X2_df.values
    X1_labels = X1_df.columns.values
    X2_labels = X2_df.columns.values

    can_load_stats = {key: []
                      for key in ['X1_train', 'X1_test', 'X2_train', 'X2_test']}
    cross_view_corr_stats = {key: [] for key in [
        'X1_train', 'X1_test', 'X2_train', 'X2_test']}
    fold_canonical_corrs = []
    fold_loadings = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X1, X2, groups=groups)):
        X1_tr, X2_tr = X1[train_idx], X2[train_idx]
        X1_te, X2_te = X1[test_idx], X2[test_idx]

        scaler1 = StandardScaler()
        X1_tr_sc = scaler1.fit_transform(X1_tr)
        X1_te_sc = scaler1.transform(X1_te)

        scaler2 = StandardScaler()
        X2_tr_sc = scaler2.fit_transform(X2_tr)
        X2_te_sc = scaler2.transform(X2_te)

        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_tr_sc, X2_tr_sc])

        canonical_corrs = cca.cancorrs
        fold_canonical_corrs.append(canonical_corrs)

        U_tr, V_tr = cca.comps
        U_te = X1_te_sc @ cca.ws[0]
        V_te = X2_te_sc @ cca.ws[1]

        fold_loadings.append({
            'X1_train': np.corrcoef(X1_tr_sc.T, U_tr.T)[:X1.shape[1], X1.shape[1]:],
            'X2_train': np.corrcoef(X2_tr_sc.T, V_tr.T)[:X2.shape[1], X2.shape[1]:],
            'X1_test': np.corrcoef(X1_te_sc.T, U_te.T)[:X1.shape[1], X1.shape[1]:],
            'X2_test': np.corrcoef(X2_te_sc.T, V_te.T)[:X2.shape[1], X2.shape[1]:]
        })

        tr_corrs, te_corrs = cca.validate(
            [X1_tr, X2_tr]), cca.validate([X1_te, X2_te])
        cross_view_corr_stats['X1_train'].append(tr_corrs[0])
        cross_view_corr_stats['X2_train'].append(tr_corrs[1])
        cross_view_corr_stats['X1_test'].append(te_corrs[0])
        cross_view_corr_stats['X2_test'].append(te_corrs[1])

    # Identify best fold
    fold_avg_corrs = [np.mean(corrs) for corrs in fold_canonical_corrs]
    best_fold_idx = np.argmax(fold_avg_corrs)
    best_fold_loadings = fold_loadings[best_fold_idx]

    can_load_stats = {
        f"mean_{k}_corr": best_fold_loadings[k]
        for k in ['X1_train', 'X1_test', 'X2_train', 'X2_test']
    }

    cross_view_corr_stats = {
        f"mean_{k}_corr": np.mean(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    } | {
        f"std_{k}_corr": np.std(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    }

    fold_canonical_corrs_array = np.array(fold_canonical_corrs)
    can_corr_stats = {
        "mean_canonical_corr": np.mean(fold_canonical_corrs_array, axis=0),
        "std_canonical_corr": np.std(fold_canonical_corrs_array, axis=0),
    }

    # Add feature labels
    can_load_stats['X1_labels'] = X1_labels
    can_load_stats['X2_labels'] = X2_labels
    cross_view_corr_stats['X1_labels'] = X1_labels
    cross_view_corr_stats['X2_labels'] = X2_labels

    can_load_df = convert_stats_dict_to_df(can_load_stats)
    cross_view_df = convert_stats_dict_to_df(cross_view_corr_stats)

    return can_load_df, cross_view_df, can_corr_stats
