import os
import pickle
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
    detrend_spikes=False,
    use_detrend_inside_cv=None,
    detrend_per_block=None,
    smoothing_width=None,
    cv_mode=None,
    model_name=None,
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
    model_name : str, optional
        If set, keep only rows for this decoder model (matches the ``model_name``
        column in consolidated results). If None, keep all models.

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
            smoothing_width=smoothing_width,
            cv_mode=cv_mode,
        )

        save_dir = runner._get_save_dir()
        
        if (shuffle_mode is not None) and (shuffle_mode != 'none'):
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
            cv_mode=cv_mode,
        )

        if model_name is not None:
            if session_results_df.empty or 'model_name' not in session_results_df.columns:
                if verbose and not session_results_df.empty:
                    print(
                        f"Warning: no 'model_name' column in results under {models_save_dir}; "
                        f"cannot filter by model_name={model_name!r}."
                    )
            else:
                session_results_df = session_results_df[
                    session_results_df['model_name'] == model_name
                ].reset_index(drop=True)

        if 'cv_mode' not in session_results_df.columns:
            inferred_cv_mode = None
            for pkl_path in Path(models_save_dir).rglob('*.pkl'):
                try:
                    with pkl_path.open('rb') as _f:
                        _loaded = pickle.load(_f)
                    _cv_config = _loaded.get('_cv_config', {}) if isinstance(_loaded, dict) else {}
                    inferred_cv_mode = _cv_config.get('cv_mode')
                    if inferred_cv_mode is None:
                        for _candidate in cv_decoding.CV_MODE_CANDIDATES:
                            if _candidate in pkl_path.stem:
                                inferred_cv_mode = _candidate
                                break
                    if inferred_cv_mode is not None:
                        break
                except Exception:
                    continue
            if inferred_cv_mode is not None:
                session_results_df['cv_mode'] = inferred_cv_mode

        session_results_df['data_name'] = row['data_name']
        session_results_df['smoothing_width'] = smoothing_width

        all_session_results.append(session_results_df)

    all_session_results_df = pd.concat(
        all_session_results,
        axis=0,
        ignore_index=True,
    )
    
    all_session_results_df['shuffle_mode'] = 'none' if shuffle_mode is None else shuffle_mode

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
    summary_df = summary_df.copy()
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
        if 'shuffle_mode' not in summary_df.columns:
            summary_df['shuffle_mode'] = 'none'
        hue_levels = summary_df['shuffle_mode'].unique().tolist()

    n_hue_levels = len(hue_levels)
    # High-contrast palette: blue and orange for 2 groups, extend for more
    default_colors = ['#e67e22', '#2166ac', '#2ca02c', '#9467bd', '#8c564b']
    colors = [default_colors[i % len(default_colors)] for i in range(max(n_hue_levels, 2))]

    for i, hue_val in enumerate(hue_levels):

        sub = summary_df.query(
            f"{hue_col} == @hue_val"
        ).sort_values('behav_feature')

        # Align each point to the global y-axis (feature_order), not sequential 0..n-1
        # within this hue — otherwise missing/extra rows misplace markers vs tick labels.
        y_codes = sub['behav_feature'].cat.codes.to_numpy()
        valid = y_codes >= 0
        sub_plot = sub.iloc[valid]
        y_positions = y_codes[valid]

        color = colors[i % len(colors)] if n_hue_levels > 1 else None
        label = f'Shuffle={hue_val}' if hue_col == 'shuffle_mode' else str(hue_val)

        ax.errorbar(
            x=sub_plot['mean_value'],
            y=y_positions,
            xerr=sub_plot['sem_value'],
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
        ax.set_xlim(-0.75, 0.75)
    elif 'auc_mean' in value_label:
        ax.set_xlim(0.4, 1.0)

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

    if model_name is not None:
        model_df = df.query("model_name == @model_name").copy()
    else:
        model_df = df

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
        group_cols = [*group_cols, 'shuffle_mode']

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



def plot_fold_diagnostics(
    all_results,
    mode='regression',
    vmin=-0.5,
    vmax=0.5,
    compare_results=None,
    compare_label=None,
    compare_plot='difference',
    diff_vmin=None,
    diff_vmax=None,
):
    result_df = all_results.query(f"mode == '{mode}'")
    value_col = 'r2_cv' if mode == 'regression' else 'auc_mean'
    result_summary = _compute_summary(result_df, value_col)

    mean_by_feature = (
        result_summary
        .groupby('behav_feature')['mean_value']
        .mean()
    )

    result_summary = result_summary.sort_values('mean_value', ascending=False)
    

    for behav_feature in result_summary['behav_feature'].values:
        print(behav_feature)
        try:
            plot_fold_session_heatmap(
                all_results,
                model_name=None,
                mode=mode,
                behav_feature=behav_feature,
                vmin=vmin,
                vmax=vmax,
                compare_results=compare_results,
                compare_label=compare_label,
                compare_plot=compare_plot,
                diff_vmin=diff_vmin,
                diff_vmax=diff_vmax,
            )
        except Exception as e:
            print(f'Error plotting {behav_feature}: {e}')

    return result_summary, mean_by_feature

# =========================
# Helper: visualize + diagnostics
# =========================
def run_and_analyze_decoding_sweep(
    all_results_dict,
    runner,
    model_name='ridge_strong',
    mode='regression',
    vmin=-0.2,
    vmax=0.2,
    compare_keys=None,
):
    if not all_results_dict:
        return None, None, None, None

    # summary
    reg_summary, clf_summary = visualize_decoding_results(
        decoding_results_dfs=all_results_dict,
        model_name=model_name,
        var_categories=runner.var_categories,
    )

    keys = list(all_results_dict.keys())
    key_a = keys[0]
    key_b = None

    # try custom keys
    if compare_keys is not None and len(compare_keys) >= 2:
        k1, k2 = compare_keys
        if k1 in all_results_dict:
            key_a = k1
        if k1 in all_results_dict and k2 in all_results_dict:
            key_b = k2

    # fallback compare
    if key_b is None and len(keys) >= 2:
        key_b = keys[1] if key_a == keys[0] else keys[0]

    # ALWAYS call diagnostics
    kwargs = dict(
        all_results=all_results_dict[key_a],
        mode=mode,
        vmin=vmin,
        vmax=vmax,
    )

    if key_b is not None:
        kwargs.update(dict(
            compare_results=all_results_dict[key_b],
            compare_label=key_b,
            compare_plot=f'three_panel',
        ))

    reg_summary2, mean_by_feature = plot_fold_diagnostics(**kwargs)

    # return reg_summary, clf_summary, reg_summary2, mean_by_feature


def plot_fold_session_heatmap(
    all_results,
    model_name,
    mode='regression',
    behav_feature=None,
    score_col=None,
    vmin=None,
    vmax=None,
    compare_results=None,
    compare_label=None,
    compare_plot='difference',
    diff_vmin=None,
    diff_vmax=None,
):
    """
    Plot session × fold decoding performance heatmap.

    compare_plot options
    --------------------
    'difference'
        Plot only all_results - compare_results.
    'side_by_side'
        Plot raw all_results and difference side by side.
    'three_panel'
        Plot all_results, compare_results, and difference.
    """
    
    if len(all_results) == 0:
        raise ValueError('No rows found in all_results')

    def _filter_results(df_in):
        df = df_in.copy()

        if model_name is not None:
            df = df.query(
                'model_name == @model_name'
            ).copy()

        if mode is not None:
            df = df.query(
                'mode == @mode'
            ).copy()

        if behav_feature is not None:
            df = df.query(
                'behav_feature == @behav_feature'
            ).copy()

        return df

    def _make_pivot(df_in, value_col):
        return df_in.pivot_table(
            index='data_name',
            columns='fold',
            values=value_col,
            aggfunc='mean'
        )

    def _plot_single_heatmap(
        ax,
        pivot,
        title,
        cbar_label,
        plot_kind,
    ):
        heatmap_kwargs = dict(
            cmap='coolwarm',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': cbar_label},
            ax=ax,
        )

        if plot_kind == 'raw':
            if vmin is not None:
                heatmap_kwargs['vmin'] = vmin
            if vmax is not None:
                heatmap_kwargs['vmax'] = vmax

            if vmin is None and vmax is None:
                heatmap_kwargs['center'] = np.nanmean(pivot.values)

        elif plot_kind == 'difference':
            if diff_vmin is not None:
                heatmap_kwargs['vmin'] = diff_vmin
            if diff_vmax is not None:
                heatmap_kwargs['vmax'] = diff_vmax

            if diff_vmin is None and diff_vmax is None:
                max_abs = np.nanmax(np.abs(pivot.values))
                heatmap_kwargs['vmin'] = -max_abs
                heatmap_kwargs['vmax'] = max_abs

            heatmap_kwargs['center'] = 0

        sns.heatmap(pivot, **heatmap_kwargs)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Session')
        ax.set_title(title)

    # ---------------------------------------
    # infer score column
    # ---------------------------------------
    if score_col is None:
        if mode == 'regression':
            score_col = 'r2_cv'
        elif mode == 'classification':
            score_col = 'auc_mean'
        else:
            raise ValueError('Unknown mode')

    # ---------------------------------------
    # filter primary results
    # ---------------------------------------
    df = _filter_results(all_results)

    if len(df) == 0:
        raise ValueError('No rows found after filtering')

    pivot = _make_pivot(df, score_col)

    base_title = f'{mode}'
    if model_name:
        base_title += f' | {model_name}'
    if behav_feature:
        base_title += f' | {behav_feature}'

    # ---------------------------------------
    # raw only
    # ---------------------------------------
    if compare_results is None or len(compare_results) == 0:
        fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(pivot))))
        _plot_single_heatmap(
            ax=ax,
            pivot=pivot,
            title=base_title,
            cbar_label=score_col,
            plot_kind='raw',
        )
        plt.tight_layout()
        plt.show()
        return pivot




    # ---------------------------------------
    # comparison mode
    # ---------------------------------------
    compare_df = _filter_results(compare_results)

    if len(compare_df) == 0:
        raise ValueError('No rows found in compare_results after filtering')

    compare_pivot = _make_pivot(compare_df, score_col)

    pivot_aligned, compare_pivot_aligned = pivot.align(compare_pivot)
    diff_pivot = pivot_aligned - compare_pivot_aligned

    if compare_label is None:
        compare_label = 'compare_results'

    if compare_plot == 'difference':
        fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(diff_pivot))))
        _plot_single_heatmap(
            ax=ax,
            pivot=diff_pivot,
            title=f'{base_title} | minus {compare_label}',
            cbar_label=f'{score_col} difference',
            plot_kind='difference',
        )
        plt.tight_layout()
        plt.show()

    elif compare_plot == 'side_by_side':
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, max(4, 0.4 * len(pivot_aligned))),
            constrained_layout=True,
        )

        _plot_single_heatmap(
            ax=axes[0],
            pivot=pivot_aligned,
            title=f'{base_title} | primary',
            cbar_label=score_col,
            plot_kind='raw',
        )

        _plot_single_heatmap(
            ax=axes[1],
            pivot=diff_pivot,
            title=f'{base_title} | minus {compare_label}',
            cbar_label=f'{score_col} difference',
            plot_kind='difference',
        )

        plt.show()

    elif compare_plot == 'three_panel':
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, max(4, 0.4 * len(pivot_aligned))),
            constrained_layout=True,
        )

        _plot_single_heatmap(
            ax=axes[0],
            pivot=pivot_aligned,
            title=f'{base_title} | primary',
            cbar_label=score_col,
            plot_kind='raw',
        )

        _plot_single_heatmap(
            ax=axes[1],
            pivot=compare_pivot_aligned,
            title=f'{base_title} | {compare_label}',
            cbar_label=score_col,
            plot_kind='raw',
        )

        _plot_single_heatmap(
            ax=axes[2],
            pivot=diff_pivot,
            title=f'{base_title} | primary minus {compare_label}',
            cbar_label=f'{score_col} difference',
            plot_kind='difference',
        )

        plt.show()

    else:
        pass
        # raise ValueError(
        #     "compare_plot must be 'difference', 'side_by_side', or 'three_panel'"
        # )

    return {
        'raw_pivot': pivot_aligned,
        'compare_pivot': compare_pivot_aligned,
        'diff_pivot': diff_pivot,
    }