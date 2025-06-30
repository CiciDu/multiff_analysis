import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.cca_methods.cca_plotting import cca_plotting
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
