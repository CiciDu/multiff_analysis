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


import numpy as np
import matplotlib.pyplot as plt


def plot_cca_cv_results(
    stats, data_type='X1', component=0,
    use_cross_view_corr=True, filter_significant=False, sort_by_significance=False,
    significance_threshold=2, title_prefix=""
):
    """
    Plot results from cross-validated CCA:
    - If use_cross_view_corr is True: plot correlation between variables and canonical projections
    - Else: plot canonical correlations

    Parameters:
        stats: dict from crossvalidated_cca_analysis
        labels: list of variable names
        data_type: 'X1' or 'X2'
        component: canonical component index
        use_cross_view_corr: toggle between loading-style and canonical correlation plots
        filter_significant: show only significant variables (|mean| > threshold * std)
        significance_threshold: std multiplier for significance
        title_prefix: optional string to prepend to title
    """
    values_train, values_test, errors_train, errors_test, final_labels = _extract_plot_data(
        stats, data_type, component, use_cross_view_corr,
        filter_significant, significance_threshold, sort_by_significance
    )

    _plot_bars(
        values_train, values_test, errors_train, errors_test, final_labels,
        ylabel='Cross-View Variable–Component Corr' if use_cross_view_corr else 'Canonical Correlation',
        title=f"{title_prefix}{data_type} - Component {component + 1}"
    )


def _extract_plot_data(
    stats, data_type, component,
    use_cross_view_corr, filter_significant, significance_threshold, sort_by_significance
):
    """
    Extract data to be plotted, apply filtering if requested.
    """
    mean_train = stats[f'mean_{data_type}_train_corr']
    std_train = stats[f'std_{data_type}_train_corr']
    mean_test = stats[f'mean_{data_type}_test_corr']
    std_test = stats[f'std_{data_type}_test_corr']
    
    labels = stats['X1_labels'] if data_type == 'X1' else stats['X2_labels']

    if use_cross_view_corr:
        values_train = mean_train.reshape(-1)
        values_test = mean_test.reshape(-1)
        errors_train = std_train.reshape(-1)
        errors_test = std_test.reshape(-1)
    else:
        values_train = mean_train[:, component]
        values_test = mean_test[:, component]
        errors_train = std_train[:, component]
        errors_test = std_test[:, component]

    if filter_significant:
        mask = np.abs(values_test) > significance_threshold * errors_test
        if not np.any(mask):
            print(
                f"No significant {data_type} variables for component {component + 1}.")
            return [], [], [], [], []
        labels = labels[mask]
        values_train = values_train[mask]
        values_test = values_test[mask]
        errors_train = errors_train[mask]
        errors_test = errors_test[mask]

    if sort_by_significance:
        # sig_score = np.abs(values_test) / errors_test
        sig_score = np.abs(values_test)
        sorted_idx = np.argsort(-sig_score)
        labels = np.array(labels)[sorted_idx]
        values_train = values_train[sorted_idx]
        values_test = values_test[sorted_idx]
        errors_train = errors_train[sorted_idx]
        errors_test = errors_test[sorted_idx]

    return values_train, values_test, errors_train, errors_test, labels


def _plot_bars(values_train, values_test, errors_train, errors_test, labels, ylabel, title):
    """
    Bar plot with train/test mean ± std.
    """
    if len(labels) == 0:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 5))
    ax.bar(x - width/2, values_train, width, yerr=errors_train,
           label='Train', alpha=0.7, capsize=4)
    ax.bar(x + width/2, values_test, width, yerr=errors_test,
           label='Test', alpha=0.7, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
