from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
from matplotlib.ticker import MaxNLocator

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding, show_decoding_results


# This has yet to be verified as working
def plot_single_trial_decoding_panel(
    readout_block: Dict,
    *,
    trial_indices: Optional[Sequence[int]] = None,
    n_trials: int = 6,
    figsize: Optional[Tuple[float, float]] = None,
):
    varnames = [k for k, v in readout_block.items() if isinstance(v, dict) and ("trials" in v)]
    if len(varnames) == 0:
        print(
            "No decoded trial series found. "
            "Re-run regress_popreadout with save_predictions=True to enable trial plots."
        )
        return None, None

    first_var = varnames[0]
    n_total_trials = len(readout_block[first_var]["trials"]["true"])
    if n_total_trials == 0:
        print("Decoded trials are empty. Skipping trial decoding panel.")
        return None, None

    if trial_indices is None:
        n_show = min(n_trials, n_total_trials)
        trial_indices = np.linspace(0, n_total_trials - 1, n_show).astype(int).tolist()
    else:
        trial_indices = [int(i) for i in trial_indices if 0 <= int(i) < n_total_trials]
    if len(trial_indices) == 0:
        print("No valid trial indices to plot.")
        return None, None

    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(varnames), 1)))
    n_rows, n_cols = len(varnames), len(trial_indices)

    # Auto-size panels for readability across different row/column counts.
    if figsize is None:
        panel_w = 1.45
        panel_h = 0.9
        fig_w = max(8.0, min(20.0, 2.2 + panel_w * n_cols))
        fig_h = max(4.5, min(18.0, 1.6 + panel_h * n_rows))
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, var in enumerate(varnames):
        c_pred = tuple(palette[r][:3])
        c_true = tuple(np.clip(np.asarray(c_pred) * 0.65, 0.0, 1.0))
        t_true = readout_block[var]["trials"]["true"]
        t_pred = readout_block[var]["trials"]["pred"]

        for c, tid in enumerate(trial_indices):
            ax = axes[r, c]
            y_true = np.asarray(t_true[tid], dtype=float)
            y_pred = np.asarray(t_pred[tid], dtype=float)
            if y_true.size == 0 or y_pred.size == 0:
                ax.axis("off")
                continue
            x = np.arange(min(y_true.size, y_pred.size))
            y_true = y_true[: x.size]
            y_pred = y_pred[: x.size]
            ax.plot(x, y_true, color=c_true, lw=1.8, alpha=0.9)
            ax.plot(x, y_pred, color=c_pred, lw=2.2, alpha=0.95)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.text(
                    -0.18,
                    0.5,
                    var,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                )

    fig.suptitle("Single-trial stop decoding", fontsize=18, y=0.99)
    fig.tight_layout(rect=[0.12, 0.02, 0.995, 0.96], h_pad=0.35, w_pad=0.22)
    plt.show()


def plot_all_decoding_results(
    *,
    canoncorr_block: Optional[Dict] = None,
    readout_block: Optional[Dict] = None,
    parity_varnames: Optional[Sequence[str]] = None,
    bar_varnames: Optional[Sequence[str]] = None,
    trial_indices: Optional[Sequence[int]] = None,
    n_trials: int = 6,
):
    if canoncorr_block is not None:
        try:
            plot_one_ff_decoding.plot_canoncorr_coefficients(canoncorr_block)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip canoncorr: result doesn't exist ({e})")
    if readout_block is not None:
        try:
            plot_one_ff_decoding.plot_decoder_parity(readout_block, varnames=parity_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip parity: result doesn't exist ({e})")
        try:
            plot_one_ff_decoding.plot_decoder_correlation_bars(readout_block, varnames=bar_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip corr_bars: result doesn't exist ({e})")
        try:
            plot_single_trial_decoding_panel(
                readout_block,
                trial_indices=trial_indices,
                n_trials=n_trials,
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip single_trial_panel: result doesn't exist ({e})")


def plot_fold_tuning_info(fold_tuning_info, ax=None, show_folds=True, shade='sem'):
    """
    Visualize CV score vs filter width across folds.

    Parameters
    ----------
    fold_tuning_info : list[dict]
        One dict per fold: {filter_width: cv_value, ...}
    ax : matplotlib.axes.Axes or None
        If None, creates a new figure/axes.
    show_folds : bool
        If True, plot each fold as a faint line.
    shade : {'sem', 'std', None}
        Shading around the mean curve. 'sem' = ±SEM, 'std' = ±STD, None = no shading.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    summary : dict
        {
          'filter_widths': np.ndarray,
          'cv_matrix': np.ndarray (n_folds, n_widths),
          'mean_cv': np.ndarray,
          'std_cv': np.ndarray,
          'sem_cv': np.ndarray,
          'best_width': int/float,
          'best_mean_cv': float
        }
    """
    if not isinstance(fold_tuning_info, (list, tuple)) or len(fold_tuning_info) == 0:
        raise ValueError('fold_tuning_info must be a non-empty list of dicts.')

    # Use the union of all widths, keep sorted for x-axis
    all_widths = sorted({w for fold in fold_tuning_info for w in fold.keys()})
    n_folds = len(fold_tuning_info)
    n_widths = len(all_widths)

    # Build matrix with NaNs for missing widths (handles imperfect folds robustly)
    cv_matrix = np.full((n_folds, n_widths), np.nan, dtype=float)
    for i, fold in enumerate(fold_tuning_info):
        for j, w in enumerate(all_widths):
            if w in fold:
                cv_matrix[i, j] = fold[w]

    mean_cv = np.nanmean(cv_matrix, axis=0)
    std_cv = np.nanstd(cv_matrix, axis=0)
    sem_cv = std_cv / np.sqrt(np.sum(~np.isnan(cv_matrix), axis=0))

    best_idx = int(np.nanargmax(mean_cv))
    best_width = all_widths[best_idx]
    best_mean_cv = float(mean_cv[best_idx])

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Plot individual folds
    if show_folds:
        for i in range(n_folds):
            ax.plot(all_widths, cv_matrix[i, :], alpha=0.3)

    # Plot mean
    ax.plot(all_widths, mean_cv, linewidth=3)

    # Shading
    if shade is not None:
        shade = shade.lower()
        if shade == 'sem':
            band = sem_cv
            label = '±SEM'
        elif shade == 'std':
            band = std_cv
            label = '±STD'
        else:
            raise ValueError("shade must be one of {'sem','std',None}")

        ax.fill_between(
            all_widths,
            mean_cv - band,
            mean_cv + band,
            alpha=0.2,
            label=label
        )

    # Annotate best
    ax.scatter([best_width], [best_mean_cv], zorder=5)
    ax.axvline(best_width, alpha=0.3, linestyle='--')
    ax.set_title('CV Performance vs Filter Width (Mean ± Variability)')
    ax.set_xlabel('Filter Width')
    ax.set_ylabel('CV Score')

    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Only add legend if shading is enabled (keeps it clean)
    if shade is not None:
        ax.legend(frameon=False)

    if created_fig:
        fig.tight_layout()

    summary = {
        'filter_widths': np.asarray(all_widths),
        'cv_matrix': cv_matrix,
        'mean_cv': mean_cv,
        'std_cv': std_cv,
        'sem_cv': sem_cv,
        'best_width': best_width,
        'best_mean_cv': best_mean_cv,
    }
    return fig, ax, summary



def compare_decoding_results(
    df_compare,
    one_ff_label='one_ff',
    main_method_label='main_method',
    figsize=(8, 6),
    line_color='gray',
    line_alpha=0.4
):
    '''
    Plot shared variables from two decoder result sets, sorted by the main_method
    method's score.

    Parameters
    ----------
    names : array-like
        Variable names for the first result set.
    vals : array-like
        Scores for the first result set.
    features : array-like
        Variable names for the second result set.
    scores : array-like
        Scores for the second result set.
    one_ff_label : str, default='one_ff'
        Label for the first result set.
    main_method_label : str, default='new_method'
        Label for the second result set.
    figsize : tuple, default=(8, 6)
        Figure size.
    line_color : str, default='gray'
        Color of connecting lines.
    line_alpha : float, default=0.4
        Alpha for connecting lines.

    Returns
    -------
    df_compare : pandas.DataFrame
        Comparison table for shared variables.
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes handle.
    '''


    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(df_compare))

    ax.scatter(df_compare[one_ff_label], y_positions, label=one_ff_label)
    ax.scatter(df_compare[main_method_label], y_positions, label=main_method_label)

    for row_idx in range(len(df_compare)):
        ax.plot(
            [df_compare.loc[row_idx, one_ff_label], df_compare.loc[row_idx, main_method_label]],
            [y_positions[row_idx], y_positions[row_idx]],
            color=line_color,
            alpha=line_alpha
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_compare['variable'])
    ax.set_xlabel('score')
    ax.set_title(f'Decoder comparison (sorted by {main_method_label})')
    ax.legend()

    fig.tight_layout()


def build_shared_decoder_comparison_df(
    df_corr,
    df_reg,
    corr_col='corr',
    reg_col='score',
    corr_label='one_ff',
    reg_label='main_method'
):
    '''
    Build comparison dataframe using shared variables
    from two decoder result dataframes.

    Parameters
    ----------
    df_corr : DataFrame
        Output of extract_decoder_corr_df
    df_reg : DataFrame
        Output of extract_regression_feature_scores_df
    '''

    df_corr = df_corr.rename(columns={corr_col: corr_label})
    df_reg = df_reg.rename(columns={reg_col: reg_label})

    df_compare = pd.merge(
        df_corr[['variable', corr_label]],
        df_reg[['variable', reg_label]],
        on='variable',
        how='inner'
    )

    df_compare['difference'] = (
        df_compare[reg_label] - df_compare[corr_label]
    )

    df_compare['higher_in'] = np.where(
        df_compare['difference'] > 0,
        reg_label,
        np.where(df_compare['difference'] < 0, corr_label, 'equal')
    )

    return df_compare


import numpy as np
import matplotlib.pyplot as plt
import math

def add_score_column(results_df, regression_metric='r_cv', classification_metric='auc_mean'):
    # Assign unified score column
    results_df['score'] = np.nan

    for mode in results_df['mode'].unique():
        metric_col = show_decoding_results._get_metric_column(
            mode,
            regression_metric,
            classification_metric
        )
        mask = results_df['mode'] == mode
        results_df.loc[mask, 'score'] = results_df.loc[mask, metric_col]
    return results_df

def plot_decoder_performance(
    results_df,
    model_selection='best_per_feature',   # 'best_per_feature' or 'single_model'
    model_name=None,                      # for testing: custom row selection function
    regression_metric='r_cv',
    classification_metric='auc_mean',
    max_vars_per_plot=20,
    show_model_in_xlabel=True,
    bar_mode='feature'                    # NEW
):
    """
    Flexible decoder performance plot.

    Parameters
    ----------
    bar_mode : str
        'feature' (default): original behavior
        'model' : grouped bars comparing models per feature
    """

    # Assign unified score column
    results_df = add_score_column(results_df, regression_metric, classification_metric)
    results_df = results_df.dropna(subset=['score'])
    if bar_mode == 'feature':
        selected_df = show_decoding_results._select_rows(
            results_df,
            model_selection=model_selection,
            model_name=model_name,
            regression_metric=regression_metric,
            classification_metric=classification_metric
        )
    else:
        selected_df = results_df.copy()  # use all rows for model comparison mode

    reg_df = selected_df[selected_df['mode'] == 'regression']
    clf_df = selected_df[selected_df['mode'] == 'classification']

    _plot_mode(
        reg_df,
        'regression',
        max_vars_per_plot=max_vars_per_plot,
        metric_name=regression_metric,
        show_model_in_xlabel=show_model_in_xlabel,
        bar_mode=bar_mode
    )

    _plot_mode(
        clf_df,
        'classification',
        max_vars_per_plot=max_vars_per_plot,
        metric_name=classification_metric,
        show_model_in_xlabel=show_model_in_xlabel,
        bar_mode=bar_mode
    )

    return selected_df


def _plot_mode(
    df_mode,
    mode,
    max_vars_per_plot=20,
    chance_line=True,
    metric_name=None,
    show_model_in_xlabel=False,
    bar_mode='feature'
):

    if df_mode.empty:
        return

    df_mode = df_mode.copy()

    # ------------------------------------------------------------------
    # NEW MODE: grouped bars comparing models
    # ------------------------------------------------------------------

    if bar_mode == 'model':

        pivot = (
            df_mode
            .groupby(['behav_feature', 'model_name'])['score']
            .mean()
            .unstack()
        )

        features = pivot.index.values
        models = pivot.columns.values

        n_features = len(features)
        n_models = len(models)

        n_subplots = math.ceil(n_features / max_vars_per_plot)

        width = max(8, max_vars_per_plot * 0.6)
        height = 4.5 * n_subplots

        fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=(width, height),
            squeeze=False
        )

        for i in range(n_subplots):

            start = i * max_vars_per_plot
            end = min((i + 1) * max_vars_per_plot, n_features)

            ax = axes[i, 0]

            sub_features = features[start:end]
            sub_df = pivot.loc[sub_features]

            x = np.arange(len(sub_features))
            bar_width = 0.8 / n_models

            for j, model in enumerate(models):

                values = sub_df[model].values

                ax.bar(
                    x + j * bar_width,
                    values,
                    bar_width,
                    label=model
                )

            ax.set_xticks(x + bar_width * (n_models - 1) / 2)
            ax.set_xticklabels(sub_features, rotation=35, ha='right')

            label = 'Cross-validated score'
            if metric_name:
                label = f'{label} ({metric_name})'

            ax.set_ylabel(label)

            if mode == 'regression':
                ax.axhline(0, linewidth=1)

            elif mode == 'classification' and chance_line:
                ax.axhline(0.5, linestyle='--', linewidth=1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[0, 0].legend(title='model')

        fig.suptitle(f'{mode.capitalize()} decoder performance', y=0.995)

        plt.tight_layout()
        plt.show()

        return

    # ------------------------------------------------------------------
    # ORIGINAL FEATURE MODE (unchanged)
    # ------------------------------------------------------------------

    df_mode = df_mode.sort_values('score', ascending=False)

    features = df_mode['behav_feature'].values
    scores = df_mode['score'].values

    if show_model_in_xlabel and 'model_name' in df_mode.columns:
        models = df_mode['model_name'].values
        labels = [f'{f} ({m})' for f, m in zip(features, models)]
    else:
        labels = features

    n_vars = len(labels)
    n_subplots = math.ceil(n_vars / max_vars_per_plot)

    width = max(8, min(max_vars_per_plot, n_vars) * 0.6)
    height = 4.5 * n_subplots

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

        sub_labels = labels[start:end]
        sub_scores = scores[start:end]

        x = np.arange(len(sub_labels))

        ax.bar(x, sub_scores)

        ax.set_xticks(x)
        ax.set_xticklabels(sub_labels, rotation=35, ha='right')
        ax.margins(x=0.02)

        if mode == 'regression':
            ax.axhline(0, linewidth=1)

        elif mode == 'classification' and chance_line:
            ax.axhline(0.5, linestyle='--', linewidth=1)

        label = 'Cross-validated score'
        if metric_name:
            label = f'{label} ({metric_name})'

        ax.set_ylabel(label)

        ax.set_xlim(-0.6, len(sub_labels) - 0.4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'{mode.capitalize()} decoder performance', y=0.995)

    plt.tight_layout()
    plt.show()