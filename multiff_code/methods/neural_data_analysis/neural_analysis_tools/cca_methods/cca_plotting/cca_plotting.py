import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils, ml_plotting_utils
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
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score


import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import AgglomerativeClustering


def plot_loading_heatmap(loadings, feature_names, canonical_corrs=None, p_values=None, matrix_label='X',
                         max_components=20, features_per_fig=30, base_width=0.75, base_height=0.3,
                         annotation_threshold=0.3, pval_threshold=0.05):
    """
    Plots a heatmap of canonical loadings with features on y-axis and components on x-axis,
    split across multiple figures if needed. Adds asterisk for significant p-values and annotates
    loadings greater than a threshold.

    Parameters:
    - loadings (np.ndarray): Feature × Component matrix (n_features × n_components)
    - feature_names (list or np.ndarray): Names of the features (length = n_features)
    - canonical_corrs (list or np.ndarray, optional): Canonical correlations for each component
    - p_values (list or np.ndarray, optional): P-values for each canonical correlation (length = n_components)
    - matrix_label (str): Label prefix for figure titles (e.g., 'X1' or 'X2')
    - max_components (int): Max number of components (columns) to plot
    - features_per_fig (int): Max number of features (rows) per figure
    - base_width (float): Width in inches per component (x-axis)
    - base_height (float): Height in inches per feature (y-axis)
    - annotation_threshold (float): Threshold above which values will be annotated on heatmap
    - pval_threshold (float): Threshold below which p-values are considered significant
    """
    # reorder loadings based on clustering
    loadings, row_order, col_order = ml_plotting_utils.reorder_based_on_clustering(
        loadings)
    feature_names = feature_names[row_order]

    max_components = min(max_components, loadings.shape[1])
    num_features = loadings.shape[0]
    feature_names = np.asarray(feature_names)

    # Determine shared color scale
    vmin = np.min(loadings[:, :max_components])
    vmax = np.max(loadings[:, :max_components])

    num_figs = math.ceil(num_features / features_per_fig)
    max_label_len = max(len(str(label)) for label in feature_names)

    for i in range(num_figs):
        start = i * features_per_fig
        end = min((i + 1) * features_per_fig, num_features)
        num_rows = end - start  # number of features (rows)

        fig_height = max(base_height * num_rows, 3)
        left_margin = 0.3 + 0.01 * max_label_len
        fig_width = max(base_width * max_components, 3) + left_margin

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        submatrix = loadings[start:end, :max_components]
        im = ax.imshow(submatrix, aspect='auto',
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)

        ax.set_title(f'{matrix_label} Loadings (Features {start}-{end})')

        # X-axis labels
        ax.set_xticks(range(max_components))
        if canonical_corrs is not None:
            labels = []
            for idx in range(max_components):
                if p_values is not None and p_values[idx] < pval_threshold:
                    label = f"{idx+1}*\n({canonical_corrs[idx]:.2f})"
                else:
                    label = f"{idx+1}\n({canonical_corrs[idx]:.2f})"
                labels.append(label)
        else:
            labels = [str(i) for i in range(1, max_components + 1)]
        ax.set_xticklabels(labels, fontsize=10)

        # Y-axis labels
        ax.set_yticks(range(num_rows))
        ax.set_yticklabels(feature_names[start:end], fontsize=8)

        # Annotate values greater than threshold
        for row in range(num_rows):
            for col in range(max_components):
                val = submatrix[row, col]
                if abs(val) >= annotation_threshold:
                    bg_color = im.cmap(
                        (val - vmin) / (vmax - vmin))  # RGBA tuple
                    brightness = 0.299 * \
                        bg_color[0] + 0.587*bg_color[1] + \
                        0.114*bg_color[2]  # luminance
                    text_color = 'white' if brightness < 0.3 else 'black'
                    ax.text(col, row, f"{val:.2f}", ha='center',
                            va='center', fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax)
        plt.subplots_adjust(left=left_margin)
        plt.tight_layout()
        plt.show()


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


def plot_cca_component_scatter(X1_c, X2_c, components, show_y_eq_x=True):
    """
    Plot scatter subplots of canonical variates for multiple components in two columns,
    with R and R² values annotated in each subplot.

    Parameters:
    - X1_c: np.ndarray, canonical variates from set 1, shape (n_samples, n_components)
    - X2_c: np.ndarray, canonical variates from set 2, shape (n_samples, n_components)
    - components: list or iterable of int, 1-based indices of components to plot
    - show_y_eq_x: bool, whether to draw y = x line in each subplot
    """

    # if components is an integer, convert to a list
    if isinstance(components, int):
        components = [components]

    n_plots = len(components)
    n_cols = 2
    n_rows = (n_plots + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, comp in enumerate(components):

        if comp < 1:
            raise ValueError(
                f"Component index must be greater than 0, but got {comp}")

        x_vals = X1_c[:, comp - 1]
        y_vals = X2_c[:, comp - 1]

        lims = [
            np.min([x_vals.min(), y_vals.min()]) * 1.1,
            np.max([x_vals.max(), y_vals.max()]) * 1.1,
        ]

        # Calculate Pearson correlation and R^2
        r, _ = pearsonr(x_vals, y_vals)
        r2 = r**2

        ax = axes[i]
        scatter = ax.scatter(
            x_vals,
            y_vals,
            alpha=0.75,
            edgecolor='black',
            s=60,
            linewidth=0.7
        )

        if show_y_eq_x:
            ax.plot(lims, lims, 'r--', linewidth=1, label='y = x')

        # Annotate R and R^2 on the plot (top-left corner)
        ax.text(0.05, 0.95, f"R = {r:.2f}\nR² = {r2:.2f}",
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        ax.set_xlabel(
            f'Canonical Variable {comp} (X)', fontsize=11, weight='bold')
        ax.set_ylabel(
            f'Canonical Variable {comp} (Y)', fontsize=11, weight='bold')
        ax.set_title(f'Component {comp}', fontsize=13, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if show_y_eq_x:
            ax.legend()

    # Hide any unused subplots if components list length is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

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
    else:
        max_plots_to_show = min(max_plots_to_show, num_variates)
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
