import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from non_behavioral_analysis.neural_data_analysis.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from non_behavioral_analysis.neural_data_analysis.model_neural_data.cca_methods import cca_plotting
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
from functools import partial
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def transform_behav_data(behav_data,
                         power_vars=['monkey_ddv', 'target_rel_y'],
                         log_vars=['time_since_last_capture'],
                         smooth_vars=['monkey_dw', 'target_rel_x'],
                         gaussian_smooth_vars=['monkey_ddv', 'delta_distance'],
                         powers=[0.5, 1, 2, 3],
                         smooth_window_size=[5, 7],
                         gaussian_smooth_sigma=[4],
                         **kwargs
                         ):

    smooth_func = partial(smooth_signal, window_size=smooth_window_size)
    gaussian_smooth_func = partial(
        gaussian_smooth, sigma=gaussian_smooth_sigma)
    power_func = partial(safe_power_features, powers=powers)

    # first use log transform
    column_transform_map = [
        (smooth_vars, smooth_func, 'smooth'),
        (gaussian_smooth_vars, gaussian_smooth_func, 'gaussian_smooth'),
        (log_vars, safe_signed_log1p, 'log'),
    ]

    X2_tf_df = apply_transformers_by_column_with_names(
        behav_data, column_transform_map)

    # standardize the data
    scaler = StandardScaler()
    X2_sc = scaler.fit_transform(X2_tf_df.values)
    X2_sc_df = pd.DataFrame(X2_sc, columns=X2_tf_df.columns)

    column_transform_map = [
        (power_vars, power_func, 'poly'),
    ]

    X_tf_df = apply_transformers_by_column_with_names(
        X2_sc_df, column_transform_map)
    return X_tf_df


def apply_transformers_by_column_with_names(
    df, column_transform_map, keep_original=True
):
    """
    Apply transformers to selected columns of X and track feature names.
    Allows passing specs for multi-feature expansions.

    Parameters:
    - X: (n_samples, n_features) ndarray
    - feature_names: list of str, original names of X's columns
    - column_transform_map: list of tuples:
        (column_indices, transformer_func, prefix, specs)
        specs: None (default) or list of str for each output feature per input feature
    - keep_original: if True, includes original untransformed columns

    Returns:
    - X_tf: ndarray of new features
    - new_feature_names: list of names for the new columns
    """
    transformed_parts = []
    new_feature_names = []

    X = df.values
    feature_names = df.columns

    if keep_original:
        transformed_parts.append(X)
        new_feature_names.extend(feature_names)

    for entry in column_transform_map:
        col_names, func, prefix = entry
        specs = None

        if len(col_names) == 0:
            continue

        col_idx = feat_idx(feature_names, col_names)

        X_subset = X[:, col_idx]
        transformed, specs = func(X_subset)
        if transformed.ndim == 1:
            transformed = transformed[:, np.newaxis]

        transformed_parts.append(transformed)

        n_input_cols = len(col_idx)
        n_output_cols = transformed.shape[1]

        if n_output_cols == n_input_cols:
            # 1:1 transform, simple naming
            new_feature_names.extend(
                [f"{prefix}_{name}" for name in col_names])
        else:
            # multi-feature per input column (e.g. polynomial expansion)
            # specs expected to be len = n_output_cols / n_input_cols
            n_features_per_col = n_output_cols // n_input_cols
            if specs is None or len(specs) != n_features_per_col:
                # fallback generic specs
                specs = [f"f{i+1}" for i in range(n_features_per_col)]
            for spec in specs:
                for i, name in enumerate(col_names):
                    new_feature_names.append(f"{prefix}_{spec}_{name}")

    X_tf = np.hstack(transformed_parts)
    X_tf_df = pd.DataFrame(X_tf, columns=new_feature_names)
    return X_tf_df


def feat_idx(all_feature_names, feature_names_to_find):
    for col in feature_names_to_find:
        if col not in all_feature_names:
            raise ValueError(f"Feature {col} not found in {all_feature_names}")
    return np.array([np.where(all_feature_names == col)[0][0] for col in feature_names_to_find])


def safe_signed_log1p(x):
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask = np.isfinite(x)
    result[mask] = np.log1p(np.abs(x[mask])) * np.sign(x[mask])
    return result, None


def safe_power_features(X, powers=[0.5, 1, 2, 3]):
    """Safe power transformation that handles negative values appropriately"""
    result_parts = []
    for p in powers:
        if p != int(p) and p > 0:  # Fractional positive powers
            # Use signed absolute value method for fractional powers
            transformed = np.sign(X) * np.power(np.abs(X), p)
        else:
            # Use normal power for integer powers (they handle negatives correctly)
            transformed = np.power(X, p)
        result_parts.append(transformed)

    specs = [f'p{p}' if (p == int(p)) else f'p{p}'.replace('.', '_')
             for p in powers]
    return np.hstack(result_parts), specs


def smooth_signal(x, window_size=[5, 10, 20]):
    x = np.asarray(x)
    result_parts = []

    # Handle both 1D and 2D inputs
    if x.ndim == 1:
        # 1D case - original logic
        for window in window_size:
            result_parts.append(np.convolve(
                x, np.ones(window)/window, mode='same'))
    else:
        # 2D case - use vectorized uniform_filter1d for all columns at once
        for window in window_size:
            # uniform_filter1d applies the filter along axis 0 (rows) for all columns simultaneously
            smoothed = uniform_filter1d(
                x.astype(float), size=window, axis=0, mode='constant')
            result_parts.append(smoothed)

    specs = [f'{window}' for window in window_size]
    return np.hstack(result_parts), specs


def gaussian_smooth(x, sigma=[1, 2, 3]):
    x = np.asarray(x)
    result_parts = []

    # Handle both 1D and 2D inputs
    if x.ndim == 1:
        # 1D case - original logic
        for s in sigma:
            result_parts.append(gaussian_filter1d(x, sigma=s))
    else:
        # 2D case - use vectorized gaussian_filter1d for all columns at once
        for s in sigma:
            # gaussian_filter1d can handle 2D arrays and apply along axis 0 for all columns
            smoothed = gaussian_filter1d(x.astype(float), sigma=s, axis=0)
            result_parts.append(smoothed)

    specs = [f'{s}' for s in sigma]
    return np.hstack(result_parts), specs


def run_cca(X1, X2, n_comp=5, n_splits=5, show_plots=True):

    # Drop rows with NA in either X1 or X2
    X1 = X1.copy()
    X2 = X2.copy()
    X1, X2 = plan_factors_utils.drop_na_in_x_and_y_var(X1, X2)

    # Initialize KFold
    kf = KFold(n_splits=n_splits)

    # Number of components for CCA
    n_comp = min(X1.shape[1], X2.shape[1], n_comp)

    # Store canonical correlations and loadings from each fold
    all_canon_corrs = []
    all_x_loadings = []
    all_y_loadings = []

    for train_index, test_index in kf.split(X1):
        # Split data into training and testing sets
        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
        X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]

        # Initialize and fit scaler on training data
        scaler = StandardScaler()
        X1_train_sc = scaler.fit_transform(X1_train)
        X1_test_sc = scaler.transform(X1_test)

        X2_train_sc = scaler.fit_transform(X2_train)
        X2_test_sc = scaler.transform(X2_test)

        # Define and fit CCA on scaled training data
        cca = CCA(scale=False, n_components=n_comp).fit(
            X1_train_sc, X2_train_sc)

        # Store loadings
        all_x_loadings.append(cca.x_loadings_)
        all_y_loadings.append(cca.y_loadings_)

        # Transform test datasets to obtain canonical variates
        X1_test_c, X2_test_c = cca.transform(X1_test_sc, X2_test_sc)

        # Calculate and store canonical correlations for this fold
        canon_corr = [np.corrcoef(X1_test_c[:, i], X2_test_c[:, i])[
            1][0] for i in range(n_comp)]
        all_canon_corrs.append(canon_corr)

    # Calculate average canonical correlations and loadings across all folds
    avg_canon_corrs = np.mean(all_canon_corrs, axis=0)
    avg_x_loadings = np.mean(all_x_loadings, axis=0)
    avg_y_loadings = np.mean(all_y_loadings, axis=0)

    if show_plots:
        cca_plotting.plot_correlation_coefficients(avg_canon_corrs, n_comp)
    return avg_x_loadings, avg_y_loadings, avg_canon_corrs
