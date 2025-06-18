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


def plot_cca_prediction_accuracy_w_scatter(testcorrsCV):
    # Plot correlations between actual test data and predictions
    # obtained by projecting the other test dataset via the CCA mapping for each dimension.
    plt.figure(figsize=(10, 6))
    nTicks = max(testcorrsCV[0].shape[0], testcorrsCV[1].shape[0])
    bmap1 = qualitative.Dark2_3
    plt.plot(np.arange(testcorrsCV[0].shape[0])+1,
             testcorrsCV[0], 'o', color=bmap1.mpl_colors[0])
    plt.plot(np.arange(testcorrsCV[1].shape[0])+1,
             testcorrsCV[1], 'o', color=bmap1.mpl_colors[1])
    plt.xlim(0.5, 0.5 + nTicks + 3)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(nTicks)+1)
    plt.xlabel('Dataset dimension')
    plt.ylabel('Prediction correlation')
    plt.title('Prediction accuracy')
    plt.legend(['Dataset 1', 'Dataset 2'])
    plt.show()


def plot_cca_prediction_accuracy_test_w_bars(testcorrsCV):
    plt.figure(figsize=(10, 6))
    # Plot canonical correlations
    plt.bar(range(len(testcorrsCV[0])),
            testcorrsCV[0], alpha=0.7, label='Set 1')
    plt.bar(range(len(testcorrsCV[1])),
            testcorrsCV[1], alpha=0.7, label='Set 2')
    plt.xlabel('Canonical component index')
    plt.ylabel('Test canonical correlation')
    plt.legend()
    plt.show()


def plot_cca_prediction_accuracy_train_test_bars(traincorrs, testcorrs):
    for i in range(2):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(testcorrs[i])),
                testcorrs[i], alpha=0.3, label='Test')
        plt.bar(range(len(traincorrs[i])),
                traincorrs[i], alpha=0.3, label='Train')
        plt.xlabel('Canonical component index')
        plt.ylabel('Prediction correlation')
        plt.title(f'Test prediction accuracy for set {i+1}')
        plt.legend()
        plt.show()


def plot_cca_prediction_accuracy_train_test_stacked_bars(traincorrs, testcorrs):
    for i in range(2):
        plt.figure(figsize=(10, 6))
        train = traincorrs[i]
        test = testcorrs[i]
        n_components = len(train)

        # Create a tidy DataFrame
        df = pd.DataFrame({
            'Component': list(range(n_components)) * 2,
            'Correlation': train.tolist() + test.tolist(),
            'Set': ['Train'] * n_components + ['Test'] * n_components
        })

        # Plot grouped bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Component',
                    y='Correlation', hue='Set', alpha=0.8)

        plt.xlabel('Canonical component index')
        plt.ylabel('Prediction correlation')
        plt.title(f'Test prediction accuracy for set {i+1}')
        plt.legend(title='Set')
        plt.tight_layout()
        plt.show()


def plot_cca_prediction_accuracy_train_test_bars_for_lags_and_no_lags(lags_testcorrs, lags_traincorrs, no_lags_testcorrs, no_lags_traincorrs):
    for i in range(2):
        plt.figure(figsize=(10, 6))
        plt.bar(range(
            len(lags_testcorrs[i])), lags_testcorrs[i], alpha=0.3, label='Test with lags')
        plt.bar(range(len(
            no_lags_testcorrs[i])), no_lags_testcorrs[i], alpha=0.3, label='Test without lags')
        plt.xlabel('Canonical component index')
        plt.ylabel('Prediction correlation')
        plt.title(f'Test prediction accuracy for set {i+1}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(range(
            len(lags_traincorrs[i])), lags_traincorrs[i], alpha=0.3, label='Train with lags')
        plt.bar(range(len(
            no_lags_traincorrs[i])), no_lags_traincorrs[i], alpha=0.3, label='Train without lags')
        plt.xlabel('Canonical component index')
        plt.ylabel('Prediction correlation')
        plt.title(f'Test prediction accuracy for set {i+1}')
        plt.legend()
        plt.show()


