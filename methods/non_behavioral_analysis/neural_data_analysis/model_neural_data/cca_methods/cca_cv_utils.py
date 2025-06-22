import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
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
    X1, X2, n_components=10, reg=0.1, n_splits=5, 
    random_state=42
):
    """
    Cross-validated CCA: compute either canonical correlations or 
    cross-view variable-to-projection correlations.

    Returns dict with mean and std stats across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    canon_corr_stats = {key: [] for key in ['X1_train', 'X1_test', 'X2_train', 'X2_test']}
    cross_view_corr_stats = {key: [] for key in ['X1_train', 'X1_test', 'X2_train', 'X2_test']}

    for train_idx, test_idx in kf.split(X1):
        X1_tr, X2_tr = X1[train_idx], X2[train_idx]
        X1_te, X2_te = X1[test_idx], X2[test_idx]

        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_tr, X2_tr])

        # Cross-view projections
        U_tr, V_tr = cca.comps
        U_te = X1_te @ cca.ws[0]
        V_te = X2_te @ cca.ws[1]
        canon_corr_stats['X1_train'].append(np.corrcoef(X1_tr.T, V_tr.T)[:X1.shape[1], X1.shape[1]:])
        canon_corr_stats['X2_train'].append(np.corrcoef(X2_tr.T, U_tr.T)[:X2.shape[1], X2.shape[1]:])
        canon_corr_stats['X1_test'].append(np.corrcoef(X1_te.T, V_te.T)[:X1.shape[1], X1.shape[1]:])
        canon_corr_stats['X2_test'].append(np.corrcoef(X2_te.T, U_te.T)[:X2.shape[1], X2.shape[1]:])

        # Canonical correlations
        tr_corrs, te_corrs = cca.validate([X1_tr, X2_tr]), cca.validate([X1_te, X2_te])
        cross_view_corr_stats['X1_train'].append(tr_corrs[0].reshape(-1, 1))
        cross_view_corr_stats['X2_train'].append(tr_corrs[1].reshape(-1, 1))
        cross_view_corr_stats['X1_test'].append(te_corrs[0].reshape(-1, 1))
        cross_view_corr_stats['X2_test'].append(te_corrs[1].reshape(-1, 1))
            
    canon_corr_stats = {
        f"mean_{k}_corr": np.mean(v, axis=0)
        for k, v in canon_corr_stats.items()
    } | {
        f"std_{k}_corr": np.std(v, axis=0)
        for k, v in canon_corr_stats.items()
    }

    cross_view_corr_stats = {
        f"mean_{k}_corr": np.mean(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    } | {
        f"std_{k}_corr": np.std(v, axis=0)
        for k, v in cross_view_corr_stats.items()
    }

    # Convert to arrays and summarize
    return canon_corr_stats, cross_view_corr_stats
