import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
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

from neural_data_analysis.neural_analysis_tools.cca_methods import cca_cv_utils



def plot_cca_lag_vs_nolag_and_train_vs_test(
    df, dataset_name, chunk_size=30, alpha=0.8,
    base_width=0.35, narrower_width_ratio=0.4,
    width_per_var=0.6, height=6,
    mode='train_offset'
):
    
    use_cross_view_corr = False if 'canonical_component' in df.columns else True

    vars_unique = df['variable'].unique()
    num_chunks = int(np.ceil(len(vars_unique) / chunk_size))
    train_test_statuses = ['train', 'test']
    lag_statuses = sorted(df['whether_lag'].unique())

    offset_groups, width_groups, colors = get_plot_config(
        mode, train_test_statuses, lag_statuses)

    for i in range(num_chunks):
        chunk_vars, chunk_df = get_chunk_data(df, vars_unique, i, chunk_size)
        fig, ax = create_plot_axes(
            chunk_vars, chunk_size, width_per_var, height)
        x = np.arange(len(chunk_vars))

        draw_bars(
            ax, chunk_df, chunk_vars, x, offset_groups, width_groups,
            mode, base_width, narrower_width_ratio, alpha, colors
        )

        finalize_plot(ax, x, chunk_vars, dataset_name, i, chunk_size, use_cross_view_corr)
        plt.tight_layout()
        plt.show()
        # print("=" * 150)


def get_plot_config(mode, train_test_statuses, lag_statuses):
    if mode == 'train_offset':
        offset_groups = train_test_statuses
        width_groups = lag_statuses
        colors = {
            ('train', 'no_lag'): '#1f77b4',
            ('train', 'lag'): '#aec7e8',
            ('test', 'no_lag'): '#ff7f0e',
            ('test', 'lag'): '#ffbb78',
        }
    elif mode == 'lag_offset':
        offset_groups = lag_statuses
        width_groups = train_test_statuses
        colors = {
            # ('lag', 'train'): '#1f77b4',
            # ('lag', 'test'): '#d62728',
            # ('no_lag', 'train'): '#2ca02c',
            # ('no_lag', 'test'): '#ff7f0e',
            ('lag', 'test'): '#1f77b4',
            ('lag', 'train'): '#aec7e8',
            ('no_lag', 'test'): '#ff7f0e',
            ('no_lag', 'train'): '#ffbb78',
        }
    else:
        raise ValueError("mode must be 'train_offset' or 'lag_offset'")
    return offset_groups, width_groups, colors


def get_chunk_data(df, vars_unique, chunk_idx, chunk_size):
    chunk_vars = vars_unique[chunk_idx *
                             chunk_size:(chunk_idx + 1) * chunk_size]
    chunk_df = df[df['variable'].isin(chunk_vars)].copy()
    chunk_df['variable'] = pd.Categorical(
        chunk_df['variable'], categories=chunk_vars, ordered=True)
    return chunk_vars, chunk_df


def create_plot_axes(chunk_vars, chunk_size, width_per_var, height):
    fig_width = max(chunk_size * width_per_var, 10)
    fig, ax = plt.subplots(figsize=(fig_width, height))
    return fig, ax


def draw_bars(
    ax, chunk_df, chunk_vars, x, offset_groups, width_groups,
    mode, base_width, narrower_width_ratio, alpha, colors
):

    corr_var = 'corr' if 'corr' in chunk_df.columns else 'mean_corr'

    n_offsets = len(offset_groups)
    offset_width = base_width * 1.5

    for idx, offset_group in enumerate(offset_groups):
        offset = (idx - (n_offsets - 1) / 2) * offset_width

        for width_group in width_groups:
            subset_df, width, color, alpha_adj = get_bar_attributes(
                chunk_df, chunk_vars, offset_group, width_group,
                mode, base_width, narrower_width_ratio, alpha, colors
            )

            ax.bar(
                x + offset, subset_df[corr_var],
                width=width, alpha=alpha_adj,
                label=f'{offset_group.capitalize()} - {width_group}',
                color=color
            )


def get_bar_attributes(
    chunk_df, chunk_vars, offset_group, width_group,
    mode, base_width, narrower_width_ratio, alpha, colors
):
    if mode == 'train_offset':
        subset_df = chunk_df[
            (chunk_df['train_or_test'] == offset_group) &
            (chunk_df['whether_lag'] == width_group)
        ]
        width = base_width if width_group == 'lag' else base_width * narrower_width_ratio
        alpha_adj = alpha
        key = (offset_group, width_group)
    else:  # lag_offset
        subset_df = chunk_df[
            (chunk_df['whether_lag'] == offset_group) &
            (chunk_df['train_or_test'] == width_group)
        ]
        width = base_width if width_group == 'train' else base_width * narrower_width_ratio
        alpha_adj = alpha
        # alpha_adj = alpha if width_group == 'train' else alpha * 0.7
        key = (offset_group, width_group)

    subset_df = subset_df.set_index('variable').reindex(chunk_vars)
    color = colors.get(key, 'gray')

    return subset_df, width, color, alpha_adj


def finalize_plot(ax, x, chunk_vars, dataset_name, chunk_idx, chunk_size, use_cross_view_corr):
    ax.set_title(
        f"{dataset_name} - Variables {chunk_idx * chunk_size + 1} to {(chunk_idx + 1) * chunk_size}")
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_vars, rotation=60, ha='right')
    ax.set_xlabel("Variable")
    ax.set_ylabel("Canonical Correlation")

    ax.axhline(0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    # if loading is plotted, then also plot y=-0.1
    if not use_cross_view_corr:
        ax.axhline(-0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    for grid_pos in x[:-1] + 0.5:
        ax.axvline(grid_pos, color='gray', linestyle='--',
                   linewidth=0.7, alpha=0.5)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Groups', loc='best')


def _build_corr_df(corrs, x_cols, train_or_test, whether_lag, dataset):
    return pd.DataFrame({
        'dataset': dataset,
        'train_or_test': train_or_test,
        'corr': corrs,
        'whether_lag': whether_lag,
        'variable': x_cols,
    })
