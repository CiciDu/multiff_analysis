import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from non_behavioral_analysis.neural_data_analysis.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from non_behavioral_analysis.neural_data_analysis.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
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
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_decomposition import CCA
import rcca
from sklearn.preprocessing import StandardScaler
from palettable.colorbrewer import qualitative

from sklearn.model_selection import KFold


def crossvalidated_cca_analysis(
    X1_df, X2_df, n_components=10, reg=0.1, n_splits=5,
    random_state=42
):
    """
    Cross-validated CCA: compute either canonical correlations or 
    cross-view variable-to-projection correlations.

    Returns dict with mean and std stats across folds.
    """
    X1 = X1_df.values
    X2 = X2_df.values

    X1_labels = X1_df.columns.values
    X2_labels = X2_df.columns.values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    can_load_stats = {key: []
                      for key in ['X1_train', 'X1_test', 'X2_train', 'X2_test']}
    cross_view_corr_stats = {key: [] for key in [
        'X1_train', 'X1_test', 'X2_train', 'X2_test']}

    for train_idx, test_idx in kf.split(X1):
        X1_tr, X2_tr = X1[train_idx], X2[train_idx]
        X1_te, X2_te = X1[test_idx], X2[test_idx]

        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_tr, X2_tr])

        # Canonical loadings
        U_tr, V_tr = cca.comps
        U_te = X1_te @ cca.ws[0]
        V_te = X2_te @ cca.ws[1]
        can_load_stats['X1_train'].append(np.corrcoef(
            X1_tr.T, V_tr.T)[:X1.shape[1], X1.shape[1]:])
        can_load_stats['X2_train'].append(np.corrcoef(
            X2_tr.T, U_tr.T)[:X2.shape[1], X2.shape[1]:])
        can_load_stats['X1_test'].append(np.corrcoef(X1_te.T, V_te.T)[
            :X1.shape[1], X1.shape[1]:])
        can_load_stats['X2_test'].append(np.corrcoef(X2_te.T, U_te.T)[
            :X2.shape[1], X2.shape[1]:])

        # Cross-view projections
        tr_corrs, te_corrs = cca.validate(
            [X1_tr, X2_tr]), cca.validate([X1_te, X2_te])
        cross_view_corr_stats['X1_train'].append(tr_corrs[0].reshape(-1, 1))
        cross_view_corr_stats['X2_train'].append(tr_corrs[1].reshape(-1, 1))
        cross_view_corr_stats['X1_test'].append(te_corrs[0].reshape(-1, 1))
        cross_view_corr_stats['X2_test'].append(te_corrs[1].reshape(-1, 1))

    can_load_stats = {
        f"mean_{k}_corr": np.mean(v, axis=0)
        for k, v in can_load_stats.items()
    } | {
        f"std_{k}_corr": np.std(v, axis=0)
        for k, v in can_load_stats.items()
    }

    cross_view_corr_stats = {
        f"mean_{k}_corr": np.mean(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    } | {
        f"std_{k}_corr": np.std(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    }

    # add labels
    can_load_stats['X1_labels'] = X1_labels
    can_load_stats['X2_labels'] = X2_labels
    cross_view_corr_stats['X1_labels'] = X1_labels
    cross_view_corr_stats['X2_labels'] = X2_labels

    # Convert to arrays and summarize
    return can_load_stats, cross_view_corr_stats


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
    import pandas as pd
    import numpy as np

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

        if mean_key in stats_dict and std_key in stats_dict:
            mean_corrs = stats_dict[mean_key]
            std_corrs = stats_dict[std_key]

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
                    'std_corr': std_corrs,
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
                    'std_corr': std_corrs.flatten(),
                    'variable': np.array(variables)[var_indices],
                    'canonical_component': comp_indices
                })

            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    stats_df = pd.concat(dfs, ignore_index=True)

    return stats_df
