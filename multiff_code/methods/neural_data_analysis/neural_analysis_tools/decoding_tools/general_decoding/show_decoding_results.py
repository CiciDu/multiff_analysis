import os
import pandas as pd
from pathlib import Path

from data_wrangling import combine_info_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding


def collect_all_session_decoding_results(
    raw_data_dir_name,
    monkey_name,
    decode_runner_class,
    verbose=False,
    shuffle_mode='none',
):
    """
    Collect existing CV decoding results across all sessions for one monkey.

    Parameters
    ----------
    raw_data_dir_name : str
        Root directory containing raw data.
    monkey_name : str
        Name of the monkey.
    decode_runner_class : class
        Runner class (e.g., decode_stops_pipeline.StopDecodingRunner).
    verbose : bool, optional
        If True, print progress information. Default is True.
    shuffle_mode : str, optional
        Shuffle mode ('none', 'foldwise', 'groupwise', 'timeshift_fold', 'timeshift_group').
        Default is 'none'.

    Returns
    -------
    all_session_results_df : pd.DataFrame
        Concatenated decoding results across all sessions.
    """

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name,
        monkey_name,
    )

    all_session_results = []

    for _, row in sessions_df_for_one_monkey.iterrows():

        if verbose:
            print('=' * 100)
            print(row['data_name'])

        raw_data_folder_path = os.path.join(
            raw_data_dir_name,
            row['monkey_name'],
            row['data_name'],
        )

        runner = decode_runner_class(
            raw_data_folder_path=raw_data_folder_path,
        )

        save_dir = runner._get_save_dir()
        if shuffle_mode != 'none':
            save_dir = Path(save_dir) / f'shuffle_{shuffle_mode}'

        session_results_df = cv_decoding.run_cv_decoding(
            save_dir=save_dir,
            load_existing_only=True,
            verbosity=1 if verbose else 0,
        )

        session_results_df['data_name'] = row['data_name']

        all_session_results.append(session_results_df)

    all_session_results_df = pd.concat(
        all_session_results,
        axis=0,
        ignore_index=True,
    )

    return all_session_results_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


def visualize_decoding_results(
    decoding_results_df,
    model_name,
    show_plots=True,
):
    
    if len(decoding_results_df) == 0:
        print(f"No decoding results found for model: {model_name}")
        return None, None

    sns.set_theme(style='whitegrid', context='talk', font_scale=0.85)

    model_df = decoding_results_df.query(
        "model_name == @model_name"
    ).copy()

    if len(model_df) == 0:
        raise ValueError(f"No rows found for model: {model_name}")

    reg_df = model_df.query("mode == 'regression'")
    clf_df = model_df.query("mode == 'classification'")

    reg_summary = None
    clf_summary = None
    
    max_height = 20

    # ==========================================================
    # REGRESSION
    # ==========================================================
    if len(reg_df) > 0:

        reg_summary = (
            reg_df
            .groupby(['behav_feature', 'shuffle_mode'])
            .agg(
                mean_r2=('r2_cv', 'mean'),
                sem_r2=('r2_cv', lambda x: x.std() / np.sqrt(len(x))),
                n_sessions=('data_name', 'nunique')
            )
            .reset_index()
        )

        # --------------------------------------------------
        # Paired t-test: real vs shuffle
        # --------------------------------------------------
        pvals = []

        for feature in reg_df['behav_feature'].unique():

            sub = reg_df.query("behav_feature == @feature")

            pivot = sub.pivot_table(
                index='data_name',
                columns='shuffle_mode',
                values='r2_cv'
            )

            if {False, True}.issubset(pivot.columns):
                stat, p = ttest_rel(pivot[False], pivot[True])
            else:
                p = np.nan

            pvals.append((feature, p))

        pval_df = pd.DataFrame(pvals, columns=['behav_feature', 'pval'])

        if pval_df['pval'].notna().sum() > 0:
            reject, pvals_corr, _, _ = multipletests(
                pval_df['pval'].fillna(1),
                method='fdr_bh'
            )
            pval_df['pval_fdr'] = pvals_corr
            pval_df['significant'] = reject
        else:
            pval_df['pval_fdr'] = np.nan
            pval_df['significant'] = False

        reg_summary = reg_summary.merge(
            pval_df,
            on='behav_feature',
            how='left'
        )

        if show_plots:

            sort_df = reg_summary.query("shuffle_mode == 'none'") \
                if 'none' in reg_summary['shuffle_mode'].values \
                else reg_summary.copy()

            feature_order = (
                sort_df
                .sort_values('mean_r2')
                ['behav_feature']
                .tolist()
            )

            reg_summary['behav_feature'] = pd.Categorical(
                reg_summary['behav_feature'],
                categories=feature_order,
                ordered=True
            )

            height = min(0.25 * len(feature_order) + 1.5, max_height)

            fig, ax = plt.subplots(
                figsize=(10, height)
            )

            hue_levels = reg_summary['shuffle_mode'].unique()
            n_hue_levels = len(hue_levels)

            for i, hue_val in enumerate(hue_levels):

                sub = reg_summary.query(
                    "shuffle_mode == @hue_val"
                ).sort_values('behav_feature')

                y_positions = np.arange(len(sub))

                offset = (
                    (i - (n_hue_levels - 1) / 2) * 0.25
                    if n_hue_levels > 1 else 0
                )

                ax.errorbar(
                    x=sub['mean_r2'],
                    y=y_positions + offset,
                    xerr=sub['sem_r2'],
                    fmt='o',
                    capsize=4,
                    markersize=6,
                    label=f'Shuffle={hue_val}'
                )

            # Add significance stars
            sig_features = pval_df.query("significant")['behav_feature']

            for idx, feature in enumerate(feature_order):
                if feature in sig_features.values:
                    ax.text(
                        x=ax.get_xlim()[1],
                        y=idx,
                        s='*',
                        va='center',
                        fontsize=14
                    )

            ax.axvline(0, linestyle='--')
            ax.set_yticks(np.arange(len(feature_order)))
            ax.set_yticklabels(feature_order)
            ax.set_xlabel('Mean Cross-validated R²')
            ax.set_title(f'Regression Decoding Strength\n{model_name}')

            if n_hue_levels > 1:
                ax.legend(frameon=False)

            plt.tight_layout()
            plt.show()

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================
    if len(clf_df) > 0:

        clf_summary = (
            clf_df
            .groupby(['behav_feature', 'shuffle_mode'])
            .agg(
                mean_auc=('auc_mean', 'mean'),
                sem_auc=('auc_mean', lambda x: x.std() / np.sqrt(len(x))),
                n_sessions=('data_name', 'nunique')
            )
            .reset_index()
        )

        # Paired test real vs shuffle
        pvals = []

        for feature in clf_df['behav_feature'].unique():

            sub = clf_df.query("behav_feature == @feature")

            pivot = sub.pivot_table(
                index='data_name',
                columns='shuffle_mode',
                values='auc_mean'
            )

            if {False, True}.issubset(pivot.columns):
                stat, p = ttest_rel(pivot[False], pivot[True])
            else:
                p = np.nan

            pvals.append((feature, p))

        pval_df = pd.DataFrame(pvals, columns=['behav_feature', 'pval'])

        if pval_df['pval'].notna().sum() > 0:
            reject, pvals_corr, _, _ = multipletests(
                pval_df['pval'].fillna(1),
                method='fdr_bh'
            )
            pval_df['pval_fdr'] = pvals_corr
            pval_df['significant'] = reject
        else:
            pval_df['pval_fdr'] = np.nan
            pval_df['significant'] = False

        clf_summary = clf_summary.merge(
            pval_df,
            on='behav_feature',
            how='left'
        )

        if show_plots:

            sort_df = clf_summary.query("shuffle_mode == 'none'") \
                if 'none' in clf_summary['shuffle_mode'].values \
                else clf_summary.copy()

            feature_order = (
                sort_df
                .sort_values('mean_auc')
                ['behav_feature']
                .tolist()
            )

            clf_summary['behav_feature'] = pd.Categorical(
                clf_summary['behav_feature'],
                categories=feature_order,
                ordered=True
            )

            height = min(0.25 * len(feature_order) + 1.5, max_height)

            fig, ax = plt.subplots(
                figsize=(6, height)
            )
            hue_levels = clf_summary['shuffle_mode'].unique()
            n_hue_levels = len(hue_levels)

            for i, hue_val in enumerate(hue_levels):

                sub = clf_summary.query(
                    "shuffle_mode == @hue_val"
                ).sort_values('behav_feature')

                y_positions = np.arange(len(sub))

                offset = (
                    (i - (n_hue_levels - 1) / 2) * 0.25
                    if n_hue_levels > 1 else 0
                )

                ax.errorbar(
                    x=sub['mean_auc'],
                    y=y_positions + offset,
                    xerr=sub['sem_auc'],
                    fmt='o',
                    capsize=4,
                    markersize=6,
                    label=f'Shuffle={hue_val}'
                )

            sig_features = pval_df.query("significant")['behav_feature']

            for idx, feature in enumerate(feature_order):
                if feature in sig_features.values:
                    ax.text(
                        x=ax.get_xlim()[1],
                        y=idx,
                        s='*',
                        va='center',
                        fontsize=14
                    )

            ax.axvline(0.5, linestyle='--')
            ax.set_yticks(np.arange(len(feature_order)))
            ax.set_yticklabels(feature_order)
            ax.set_xlabel('Mean AUC')
            ax.set_title(f'Classification Decoding Strength\n{model_name}')

            if n_hue_levels > 1:
                ax.legend(frameon=False)

            plt.tight_layout()
            plt.show()

    return reg_summary, clf_summary


# show decoding results for one session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# ============================================================
# Helper 1: Select metric per mode
# ============================================================

def _get_metric_column(mode, regression_metric, classification_metric):
    if mode == 'regression':
        return regression_metric
    elif mode == 'classification':
        return classification_metric
    else:
        raise ValueError(f'Unknown mode: {mode}')


# ============================================================
# Helper 2: Select rows according to model strategy
# ============================================================

def _select_rows(
    df,
    model_selection='best_per_feature',
    model_name=None,
    regression_metric='r_cv',
    classification_metric='auc_mean'
):
    df = df.copy()

    # Assign unified score column
    df['score'] = np.nan

    for mode in df['mode'].unique():
        metric_col = _get_metric_column(
            mode,
            regression_metric,
            classification_metric
        )
        mask = df['mode'] == mode
        df.loc[mask, 'score'] = df.loc[mask, metric_col]

    df = df.dropna(subset=['score'])

    # -------------------------------
    # Strategy 1: best per feature
    # -------------------------------
    if model_selection == 'best_per_feature':
        idx = df.groupby(['behav_feature', 'mode'])['score'].idxmax()
        selected_df = df.loc[idx].reset_index(drop=True)

    # -------------------------------
    # Strategy 2: single model
    # -------------------------------
    elif model_selection == 'single_model':
        if model_name is None:
            raise ValueError('You must provide model_name for single_model mode.')
        selected_df = df[df['model_name'] == model_name].copy()

    else:
        raise ValueError("model_selection must be 'best_per_feature' or 'single_model'")

    return selected_df


# ============================================================
# Helper 3: Plot a single mode (with auto subplot splitting)
# ============================================================
def _plot_mode(
    df_mode,
    mode,
    max_vars_per_plot=20,
    chance_line=True,
    metric_name=None,
):
    if df_mode.empty:
        return

    df_mode = df_mode.sort_values('score', ascending=False)

    features = df_mode['behav_feature'].values
    scores = df_mode['score'].values

    n_vars = len(features)
    n_subplots = math.ceil(n_vars / max_vars_per_plot)

    # ---- Dynamic figure sizing ----
    max_vars_this_plot = min(max_vars_per_plot, n_vars)
    width = max(8, max_vars_this_plot * 0.6)   # scale width to number of bars
    height = 4.5 * n_subplots                  # scale height to subplot count

    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(width, height),
        squeeze=False
    )

    for i in range(n_subplots):

        start = i * max_vars_per_plot
        end = min((i + 1) * max_vars_per_plot, n_vars)

        ax = axes[i, 0]

        sub_features = features[start:end]
        sub_scores = scores[start:end]

        x = np.arange(len(sub_features))

        ax.bar(x, sub_scores)

        # ---- X-axis formatting ----
        ax.set_xticks(x)
        ax.set_xticklabels(sub_features, rotation=35, ha='right')
        ax.margins(x=0.02)

        # ---- Reference lines ----
        if mode == 'regression':
            ax.axhline(0, linewidth=1)
        elif mode == 'classification' and chance_line:
            ax.axhline(0.5, linestyle='--', linewidth=1)

        label = 'Cross-validated score'
        if metric_name:
            label = f"{label} ({metric_name})"

        ax.set_ylabel(label)

        # Remove extra whitespace
        ax.set_xlim(-0.6, len(sub_features) - 0.4)

        # Cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'{mode.capitalize()} decoder performance', y=0.995)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main plotting function
# ============================================================

def plot_decoder_performance(
    all_results_df,
    model_selection='best_per_feature', # 'best_per_feature' or 'single_model'
    model_name=None,
    regression_metric='r_cv',
    classification_metric='auc_mean',
    max_vars_per_plot=20
):
    """
    Flexible decoder performance plot.

    Parameters
    ----------
    model_selection : str
        'best_per_feature' or 'single_model'

    model_name : str
        Required if model_selection='single_model'

    regression_metric : str
        'r_cv' or 'r2_cv'

    classification_metric : str
        'auc_mean' or 'pr_mean'

    max_vars_per_plot : int
        Number of variables per subplot before splitting.
    """

    selected_df = _select_rows(
        all_results_df,
        model_selection=model_selection,
        model_name=model_name,
        regression_metric=regression_metric,
        classification_metric=classification_metric
    )

    # Split by mode
    reg_df = selected_df[selected_df['mode'] == 'regression']
    clf_df = selected_df[selected_df['mode'] == 'classification']

    # Plot separately
    _plot_mode(
        reg_df,
        'regression',
        max_vars_per_plot=max_vars_per_plot,
        metric_name=regression_metric,
    )
    _plot_mode(
        clf_df,
        'classification',
        max_vars_per_plot=max_vars_per_plot,
        metric_name=classification_metric,
    )

    return selected_df