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


def plot_cca_results(cca_results):

    # Plot canonical correlations
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Canonical correlations
    axes[0].bar(range(len(cca_results['canon_corr'])),
                cca_results['canon_corr'])
    axes[0].set_title('Canonical Correlations')
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Correlation')

    # Canonical variables scatter plot
    axes[1].scatter(cca_results['X1_canon_vars'][:, 0],
                    cca_results['X2_canon_vars'][:, 0], alpha=0.5, color='blue')
    axes[1].set_title('First Canonical Variables')
    axes[1].set_xlabel('Neural CV1')
    axes[1].set_ylabel('Behavioral CV1')
    # add a line of y=x (45 degrees)
    axes[1].plot([cca_results['X1_canon_vars'][:, 0].min(), cca_results['X1_canon_vars'][:, 0].max()],
                 [cca_results['X1_canon_vars'][:, 0].min(
                 ), cca_results['X1_canon_vars'][:, 0].max()],
                 color='red', linestyle='--')
    # add R & R2
    axes[1].text(0.05, 0.95, f'R={cca_results["canon_corr"][0]:.2f}, R2={cca_results["canon_corr"][0]**2:.2f}',
                 transform=axes[1].transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()


# Function to make a series of bar plots of ranked loadings
def make_a_series_of_barplots_of_ranked_loadings_or_weights(squared_loading, canon_corr, num_variates,
                                                            keep_one_value_for_each_feature=False,
                                                            max_plots_to_show=None,
                                                            max_features_to_show_per_plot=20,
                                                            horizontal_bars=True,
                                                            squared=False):
    # Get the unique feature categories
    unique_feature_category = _get_unique_feature_category(
        squared_loading, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot)
    # Generate a color dictionary for the unique feature categories
    color_dict = _get_color_dict(unique_feature_category)

    if max_plots_to_show is None:
        max_plots_to_show = num_variates
    # Iterate over the number of variates
    for variate in range(max_plots_to_show):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = squared_loading.sort_values(variate, ascending=False, key=abs).groupby(
                'feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = squared_loading.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(
            by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]

        # Create a new plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # If horizontal bars are preferred
        xlabel = "Squared Loading" if squared else "Loading"
        if horizontal_bars:
            # Create a horizontal bar plot with seaborn
            sns.barplot(data=loading_subset, x=variate, y='feature', dodge=False,
                        ax=ax, hue='feature_category', palette=color_dict, orient='h')
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel("")
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)
        else:
            # Otherwise, create a vertical bar plot
            sns.barplot(data=loading_subset, x='feature', y=variate,
                        dodge=False, ax=ax, hue='feature_category', palette=color_dict)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(xlabel, fontsize=14)
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)

        # If the flag is set to keep one value for each feature, remove the legend
        if keep_one_value_for_each_feature:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Draw a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='--')

        # Calculate the coefficient and set the title of the plot
        coefficient = np.around(np.array(canon_corr), 2)[variate]
        plt.title(
            f'Variate: {variate + 1}; canonical correlation coefficient: {coefficient}', fontsize=18)

        # Display the plot
        plt.show()

        # Close the plot to free up memory


# Function to get unique feature categories based on the given parameters
def _get_unique_feature_category(squared_loading, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot):
    unique_feature_category = np.array([])
    for variate in range(num_variates):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = squared_loading.sort_values(variate, ascending=False, key=abs).groupby(
                'feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = squared_loading.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(
            by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]
        # Update the unique feature categories
        unique_feature_category = np.unique(np.concatenate(
            [unique_feature_category, loading_subset.feature_category]))
    # Log the number of unique feature categories included in the plot
    logging.info(
        f"{len(unique_feature_category)} out of {len(squared_loading.feature_category.unique())} feature categories are included in the plot")
    return unique_feature_category


# Function to generate a color dictionary for the unique feature categories
def _get_color_dict(unique_feature_category):
    # Get the first 10 colors from the Set3 palette
    qualitative_colors = sns.color_palette("Set3", 10)
    # Get the remaining colors from the Glasbey palette
    qualitative_colors_2 = sns.color_palette(
        colorcet.glasbey, n_colors=len(unique_feature_category)-10)
    # Combine the two color palettes
    qualitative_colors.extend(qualitative_colors_2)
    # Create a dictionary mapping each feature category to a color
    color_dict = {unique_feature_category[i]: qualitative_colors[i] for i in range(
        len(unique_feature_category))}
    return color_dict


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


def plot_correlation_coefficients(avg_canon_corrs):
    # Plot average canonical correlations
    bar_names = [f'CC {i+1}' for i in range(len(avg_canon_corrs))]
    plt.bar(bar_names, avg_canon_corrs,
            color='lightgrey', width=0.8, edgecolor='k')

    # Label y value on each bar
    for i, val in enumerate(avg_canon_corrs):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.title('Average Canonical Correlations Across Folds')
    plt.show()
    return


def plot_x_loadings(avg_x_loadings, avg_canon_corrs, X1):

    squared_loading = pd.DataFrame(np.round(avg_x_loadings**2, 3))
    squared_loading['feature'] = X1.columns
    squared_loading['feature_category'] = squared_loading['feature']

    num_variates = avg_x_loadings.shape[1]
    make_a_series_of_barplots_of_ranked_loadings_or_weights(
        squared_loading, avg_canon_corrs, num_variates, keep_one_value_for_each_feature=True, max_features_to_show_per_plot=20)
    return


def plot_y_loadings(avg_y_loadings, avg_canon_corrs, X2):

    squared_loading = pd.DataFrame(np.round(avg_y_loadings**2, 3))
    squared_loading['feature'] = X2.columns
    squared_loading['feature_category'] = squared_loading['feature']

    num_variates = avg_y_loadings.shape[1]
    make_a_series_of_barplots_of_ranked_loadings_or_weights(
        squared_loading, avg_canon_corrs, num_variates, keep_one_value_for_each_feature=True, max_features_to_show_per_plot=5)
    return
