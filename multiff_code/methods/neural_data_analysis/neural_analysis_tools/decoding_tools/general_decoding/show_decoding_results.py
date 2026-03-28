import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests



from data_wrangling import combine_info_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding


# ==========================================================
# VARIABLE CATEGORIES (DECODING)
# ==========================================================

DEFAULT_DECODING_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w", "accel", "ang_accel"],
    "latent_vars": ["cur_ff_distance", "cur_ff_angle", "nxt_ff_distance", "nxt_ff_angle"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["stop"],
}


def collect_all_session_decoding_results(
    raw_data_dir_name,
    monkey_name,
    decode_runner_class,
    verbose=False,
    shuffle_mode='none',
    fit_kernelwidth=True,
    detrend_spikes=True,
    use_detrend_inside_cv=None,
    detrend_per_block=None,
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
    fit_kernelwidth : bool, optional
        Whether to load nested CV (fit kernelwidth) or fixed-width results.
    use_detrend_inside_cv : bool, optional
        If True, keep only results from detrended runs. If False, keep only
        non-detrended. If None (default), keep all.
    detrend_per_block : bool, optional
        When use_detrend_inside_cv=True: if True, only per-block detrended; if False,
        only global detrended; if None, both.

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
            detrend_spikes=detrend_spikes,
        )

        save_dir = runner._get_save_dir()
        
        if shuffle_mode != 'none':
            save_dir = Path(save_dir) / f'shuffle_{shuffle_mode}'

        if fit_kernelwidth:
            models_save_dir = Path(save_dir) / "cv_decoding" / "cvnested"
        else:
            models_save_dir = Path(save_dir) / "cv_decoding" / "fixed_width"

        session_results_df = cv_decoding.run_cv_decoding(
            save_dir=models_save_dir,
            load_existing_only=True,
            verbosity=1 if verbose else 0,
            use_detrend_inside_cv=use_detrend_inside_cv,
            detrend_per_block=detrend_per_block,
        )

        session_results_df['data_name'] = row['data_name']

        all_session_results.append(session_results_df)

    all_session_results_df = pd.concat(
        all_session_results,
        axis=0,
        ignore_index=True,
    )
    
    all_session_results_df['shuffle_mode'] = shuffle_mode

    return all_session_results_df



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






# ==========================================================
# DATA PREPARATION
# ==========================================================

def _prepare_decoding_df(decoding_results_df):
    df = decoding_results_df.copy()

    if 'shuffle_mode' not in df.columns:
        df['shuffle_mode'] = 'none'

    if 'data_name' not in df.columns:
        df['data_name'] = 'one_session'

    return df


def _build_var_category_lookup(var_categories):
    lookup = {}
    for cat, vars_list in var_categories.items():
        for v in vars_list:
            lookup[v] = cat
    return lookup


def _add_var_category_column(df, var_categories, feature_col='behav_feature', category_col='var_category'):
    if var_categories is None:
        return df
    lookup = _build_var_category_lookup(var_categories)
    df = df.copy()
    df[category_col] = df[feature_col].map(lookup)
    return df


# ==========================================================
# PLOTTING
# ==========================================================

def _plot_decoding_summary(summary_df, pval_df, value_label, title, vline, hue_col=None, var_categories=None):
    """
    Plot decoding summary. Uses hue_col for grouping when provided (e.g. 'result_group'
    for multiple datasets), otherwise falls back to 'shuffle_mode'.
    """
    max_height = 20

    mean_by_feature = (
        summary_df
        .groupby('behav_feature')['mean_value']
        .mean()
    )

    if var_categories is not None and 'var_category' in summary_df.columns:
        feature_order = []
        present_categories = summary_df['var_category'].dropna().unique().tolist()
        ordered_categories = [
            c for c in var_categories.keys() if c in present_categories
        ]
        for cat in ordered_categories:
            sub = summary_df.query("var_category == @cat")
            feats = sub['behav_feature'].unique().tolist()
            feats_sorted = sorted(feats, key=lambda f: mean_by_feature[f])
            feature_order.extend(feats_sorted)
        remaining_feats = [
            f for f in mean_by_feature.index.tolist() if f not in feature_order
        ]
        if remaining_feats:
            remaining_sorted = (
                mean_by_feature.loc[remaining_feats]
                .sort_values()
                .index
                .tolist()
            )
            feature_order.extend(remaining_sorted)
    else:
        feature_order = (
            mean_by_feature
            .sort_values()
            .index
            .tolist()
        )

    summary_df['behav_feature'] = pd.Categorical(
        summary_df['behav_feature'],
        categories=feature_order,
        ordered=True
    )

    height = min(0.35 * len(feature_order) + 1.5, max_height)

    fig, ax = plt.subplots(figsize=(8, height))

    if hue_col is not None and hue_col in summary_df.columns:
        hue_levels = summary_df[hue_col].dropna().unique().tolist()
    else:
        hue_col = 'shuffle_mode'
        hue_levels = summary_df['shuffle_mode'].unique().tolist()

    n_hue_levels = len(hue_levels)
    # High-contrast palette: blue and orange for 2 groups, extend for more
    default_colors = ['#e67e22', '#2166ac', '#2ca02c', '#9467bd', '#8c564b']
    colors = [default_colors[i % len(default_colors)] for i in range(max(n_hue_levels, 2))]

    for i, hue_val in enumerate(hue_levels):

        sub = summary_df.query(
            f"{hue_col} == @hue_val"
        ).sort_values('behav_feature')

        y_positions = np.arange(len(sub))

        offset = (
            (i - (n_hue_levels - 1) / 2) * 0.25
            if n_hue_levels > 1 else 0
        )

        color = colors[i % len(colors)] if n_hue_levels > 1 else None
        label = f'Shuffle={hue_val}' if hue_col == 'shuffle_mode' else str(hue_val)

        ax.errorbar(
            x=sub['mean_value'],
            y=y_positions + offset,
            xerr=sub['sem_value'],
            fmt='o',
            capsize=4,
            markersize=6,
            label=label,
            color=color
        )

    if pval_df is not None and len(pval_df) > 0 and 'significant' in pval_df.columns:
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

    ax.axvline(vline, linestyle='--')
    ax.set_yticks(np.arange(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    
    if 'r2_cv' in value_label or 'r_cv' in value_label:
        ax.set_xlim(-0.5, 0.5)

    if n_hue_levels > 1:
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


def visualize_decoding_results(
    decoding_results_df=None,
    decoding_results_dfs=None,
    model_name='ridge_strong',
    regression_metric_col='r2_cv',
    classification_metric_col='auc_mean',
    show_plots=True,
    var_categories=None,
):
    """
    Visualize decoding results. Supports plotting multiple result DataFrames on the
    same plot, differentiated by color.

    Parameters
    ----------
    decoding_results_df : pd.DataFrame, optional
        Single decoding results DataFrame. Ignored if decoding_results_dfs is provided.
    decoding_results_dfs : dict, optional
        Dict mapping label -> DataFrame, e.g. {'all': all_results, 'subset': all_results_s}.
        When provided, all datasets are plotted on the same axes, differentiated by color.
    model_name : str
        Model name to filter (e.g. 'ridge_strong').
    var_categories : dict, optional
        Mapping from category_name -> list of variable names. Used to add a
        'var_category' column and to group variables on the y-axis. If None,
        uses DEFAULT_DECODING_VAR_CATEGORIES.
    """
    if decoding_results_dfs is not None:
        dfs = []
        for label, d in decoding_results_dfs.items():
            prep = _prepare_decoding_df(d)
            prep['result_group'] = label
            dfs.append(prep)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        hue_col = 'result_group'
    elif decoding_results_df is not None:
        df = _prepare_decoding_df(decoding_results_df)
        df['result_group'] = 'single'
        hue_col = None
    else:
        raise ValueError("Provide either decoding_results_df or decoding_results_dfs")

    if len(df) == 0:
        print(f"No decoding results found for model: {model_name}")
        return None, None

    sns.set_theme(style='whitegrid', context='talk', font_scale=0.85)

    if var_categories is None:
        var_categories = DEFAULT_DECODING_VAR_CATEGORIES

    model_df = df.query("model_name == @model_name").copy()

    model_df = _add_var_category_column(
        model_df,
        var_categories=var_categories,
        feature_col='behav_feature',
        category_col='var_category',
    )

    if len(model_df) == 0:
        raise ValueError(f"No rows found for model: {model_name}")

    reg_df = model_df.query("mode == 'regression'")
    clf_df = model_df.query("mode == 'classification'")

    reg_summary = None
    clf_summary = None

    # ======================================================
    # REGRESSION
    # ======================================================

    if len(reg_df) > 0:

        reg_summary = _compute_summary(reg_df, regression_metric_col, hue_col=hue_col)

        pval_df = _compute_shuffle_pvals(reg_df, regression_metric_col)

        reg_summary = reg_summary.merge(
            pval_df,
            on='behav_feature',
            how='left'
        )

        if show_plots:

            _plot_decoding_summary(
                reg_summary,
                pval_df,
                value_label=f'Mean {regression_metric_col}',
                title=f'Regression Decoding Strength\n{model_name}',
                vline=0,
                hue_col=hue_col,
                var_categories=var_categories,
            )

    # ======================================================
    # CLASSIFICATION
    # ======================================================

    if len(clf_df) > 0:

        clf_summary = _compute_summary(clf_df, classification_metric_col, hue_col=hue_col)

        pval_df = _compute_shuffle_pvals(clf_df, classification_metric_col)

        clf_summary = clf_summary.merge(
            pval_df,
            on='behav_feature',
            how='left'
        )

        if show_plots:

            _plot_decoding_summary(
                clf_summary,
                pval_df,
                value_label=f'Mean {classification_metric_col}',
                title=f'Classification Decoding Strength\n{model_name}',
                vline=0.5,
                hue_col=hue_col,
                var_categories=var_categories,
            )

    return reg_summary, clf_summary


def _compute_shuffle_pvals(df, value_col):

    pvals = []

    for feature in df['behav_feature'].unique():

        sub = df.query("behav_feature == @feature")

        pivot = sub.pivot_table(
            index='data_name',
            columns='shuffle_mode',
            values=value_col
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

    return pval_df


def _compute_summary(df, value_col, hue_col=None):

    group_cols = ['behav_feature']
    if 'var_category' in df.columns:
        group_cols.append('var_category')
    if hue_col is not None and hue_col in df.columns:
        group_cols = ['behav_feature', hue_col]
    elif 'shuffle_mode' in df.columns:
        group_cols.append('shuffle_mode')

    summary = (
        df
        .groupby(group_cols)
        .agg(
            mean_value=(value_col, 'mean'),
            sem_value=(value_col, lambda x: x.std() / np.sqrt(len(x))),
            n_sessions=('data_name', 'nunique')
        )
        .reset_index()
    )

    return summary


def compute_fold_diagnostics(df, score_col):

    fold_stats = (
        df
        .groupby(['behav_feature', 'fold'])
        .agg(
            mean_score=(score_col, 'mean'),
            std_score=(score_col, 'std')
        )
        .reset_index()
    )

    return fold_stats


def compute_fold_bias(df, score_col):

    df = df.copy()

    df['session_mean'] = (
        df.groupby(['data_name', 'behav_feature'])[score_col]
        .transform('mean')
    )

    df['fold_bias'] = df[score_col] - df['session_mean']

    bias = (
        df.groupby('fold')
        .agg(
            mean_bias=('fold_bias', 'mean'),
            sem_bias=('fold_bias', lambda x: x.std()/np.sqrt(len(x)))
        )
        .reset_index()
    )

    return bias


def plot_fold_session_heatmap(
    all_results,
    model_name,
    mode='regression',
    behav_feature=None,
    score_col=None,
    vmin=None,
    vmax=None,
):
    """
    Plot session × fold decoding performance heatmap.

    Parameters
    ----------
    all_results : DataFrame
        full decoding results dataframe
    model_name : str
        model to visualize
    mode : str
        'regression' or 'classification'
    behav_feature : str or None
        optional feature to filter
    score_col : str or None
        score column (auto-detected if None)
    vmin : float or None
        lower bound for colorbar (None = auto from data)
    vmax : float or None
        upper bound for colorbar (None = auto from data)
    """

    df = all_results.copy()

    # ---------------------------------------
    # filter model + mode
    # ---------------------------------------
    df = df.query(
        "model_name == @model_name and mode == @mode"
    ).copy()

    if behav_feature is not None:
        df = df.query(
            "behav_feature == @behav_feature"
        ).copy()

    if len(df) == 0:
        raise ValueError("No rows found after filtering")

    # ---------------------------------------
    # infer score column
    # ---------------------------------------
    if score_col is None:

        if mode == 'regression':
            score_col = 'r2_cv'

        elif mode == 'classification':
            score_col = 'auc_mean'

        else:
            raise ValueError("Unknown mode")

    # ---------------------------------------
    # pivot
    # ---------------------------------------
    pivot = df.pivot_table(
        index='data_name',
        columns='fold',
        values=score_col,
        aggfunc='mean'
    )

    # ---------------------------------------
    # plot
    # ---------------------------------------
    plt.figure(figsize=(6, max(4, 0.4 * len(pivot))))

    heatmap_kwargs = dict(
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': score_col},
    )
    if vmin is not None:
        heatmap_kwargs['vmin'] = vmin
    if vmax is not None:
        heatmap_kwargs['vmax'] = vmax

    if vmin is None and vmax is None:
        heatmap_kwargs['center'] = pivot.values.mean()
        
    
    sns.heatmap(pivot, **heatmap_kwargs)

    plt.xlabel('Fold')
    plt.ylabel('Session')

    title = f'{model_name} | {mode}'
    if behav_feature:
        title += f' | {behav_feature}'

    plt.title(title)

    plt.tight_layout()
    plt.show()

    return pivot