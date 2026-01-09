"""
Plotting utilities for conditional regression decoding.

Expects output from:
  conditional_decoding_reg.run_conditional_decoding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import add_interactions


# ============================================================
# Shared helpers
# ============================================================

def _pruned_or_fallback(analysis_out, x_df, y_df, *, condition_name=None, min_count=200):
    """
    Return (x_df_use, y_df_use) preferring analysis_out['*_pruned'] when present.
    If pruned data are missing, require x_df and y_df and optionally prune
    by condition_name for per-condition plots.
    """
    x_pruned = analysis_out.get('x_pruned') if isinstance(
        analysis_out, dict) else None
    y_pruned = analysis_out.get('y_pruned') if isinstance(
        analysis_out, dict) else None

    if x_pruned is not None and y_pruned is not None:
        return x_pruned, y_pruned

    assert x_df is not None and y_df is not None, (
        'x_df and y_df must be provided when analysis_out lacks pruned data'
    )
    if condition_name is not None:
        y_df, x_df = add_interactions.prune_rare_states_two_dfs(
            df_behavior=y_df,
            df_neural=x_df,
            label_col=condition_name,
            min_count=min_count,
        )
    return x_df, y_df


def summarize_bootstrap_ci(values, ci=95):
    """
    Percentile confidence interval from a bootstrap distribution.
    """
    values = np.asarray(values)
    alpha = (100 - ci) / 2

    return {
        'mean': np.mean(values),
        'ci_low': np.percentile(values, alpha),
        'ci_high': np.percentile(values, 100 - alpha),
    }


def plot_global_decoding_clf(
    *,
    interaction_summary_df,
    interaction_name,
    ax,
):
    ax.bar(
        interaction_summary_df['model'],
        interaction_summary_df['mean_bal_acc'],
        alpha=0.8,
    )

    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel('Decoder model')
    ax.set_title(f'Global decoding: {interaction_name}')
    ax.set_ylim(0, 1)


def plot_bootstrapped_delta(
    *,
    delta_bootstrap_df,
    model_type,
    target_name,
    condition_name,
    ax,
    ci=95,
):
    """
    Plot absolute conditioned decoding accuracy with bootstrapped CI
    and a dashed baseline at the global accuracy.
    Expects per-bootstrap (fold-collapsed) estimates with columns:
      - delta_balanced_accuracy
      - global_balanced_accuracy
      - cond_balanced_accuracy (optional; if absent will be derived)
    """

    df = delta_bootstrap_df.query("model == @model_type")

    condition_values = sorted(df['condition_value'].unique())

    means = []
    lowers = []
    uppers = []

    # Compute global baseline as mean across bootstraps
    if 'global_balanced_accuracy' in df.columns:
        global_vals = df[['bootstrap_id', 'global_balanced_accuracy']].drop_duplicates()[
            'global_balanced_accuracy'].values
        global_summary = summarize_bootstrap_ci(global_vals, ci=ci)
        global_baseline = global_summary['mean']
    else:
        # Fallback to zero if not provided (should not happen if upstream filled it)
        global_baseline = 0.0

    for val in condition_values:
        sub = df.query("condition_value == @val")
        if 'cond_balanced_accuracy' in sub.columns:
            vals = sub['cond_balanced_accuracy'].values
        else:
            # derive absolute conditioned accuracy = delta + global per-bootstrap
            # attempt to align by bootstrap_id
            if 'global_balanced_accuracy' in sub.columns and 'bootstrap_id' in sub.columns:
                vals = (sub['delta_balanced_accuracy'] +
                        sub['global_balanced_accuracy']).values
            else:
                # final fallback: shift deltas by global_baseline scalar
                vals = (sub['delta_balanced_accuracy'] +
                        global_baseline).values

        summary = summarize_bootstrap_ci(vals, ci=ci)

        means.append(summary['mean'])
        lowers.append(summary['ci_low'])
        uppers.append(summary['ci_high'])

    x = np.arange(len(condition_values))

    ax.errorbar(
        x,
        means,
        yerr=[
            np.array(means) - np.array(lowers),
            np.array(uppers) - np.array(means),
        ],
        fmt='o',
        capsize=4,
    )

    # Baseline at global accuracy
    ax.axhline(global_baseline, linestyle='--', color='k', alpha=0.6)

    ax.set_xticks(x)
    # If available, annotate each condition with its sample size
    if 'n_samples' in df.columns:
        n_map = (
            df[['condition_value', 'n_samples']]
            .drop_duplicates()
            .set_index('condition_value')['n_samples']
            .to_dict()
        )
        labels = [
            f'{val} (n={int(n_map.get(val, np.nan))})' for val in condition_values]
    else:
        labels = [str(val) for val in condition_values]
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel(condition_name)
    ax.set_title(f'{target_name} | {condition_name}')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))


def plot_pairwise_interaction_analysis_clf(
    *,
    analysis_out,
    interaction_name,
    var_a,
    var_b,
    model_type='logreg',
    ci=95,
    figsize=(12, 4),
):
    """
    Three-panel plot:
    A) Global interaction decoding
    B) Δ decoding: var_a | var_b
    C) Δ decoding: var_b | var_a
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_global_decoding_clf(
        interaction_summary_df=analysis_out['interaction_summary'],
        interaction_name=interaction_name,
        ax=axes[0],
    )

    plot_bootstrapped_delta(
        delta_bootstrap_df=analysis_out['cond_var_a_delta_bootstrap'],
        model_type=model_type,
        target_name=var_a,
        condition_name=var_b,
        ax=axes[1],
        ci=ci,
    )

    plot_bootstrapped_delta(
        delta_bootstrap_df=analysis_out['cond_var_b_delta_bootstrap'],
        model_type=model_type,
        target_name=var_b,
        condition_name=var_a,
        ax=axes[2],
        ci=ci,
    )

    plt.tight_layout()
    return fig


# ============================================================
# Global decoding plot
# ============================================================

def plot_global_decoding_reg(
    *,
    global_summary,
    metric='r2',
    ax=None,
    target_name=None,
):
    """
    Bar plot of global decoding performance across models.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.bar(
        global_summary['model'],
        global_summary[metric],
        alpha=0.8,
    )

    ax.set_ylabel(metric.upper())
    ax.set_xlabel('Model')
    if target_name:
        ax.set_title(f'Global decoding: {target_name}')
    else:
        ax.set_title('Global decoding')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticklabels(global_summary['model'], rotation=30)
    # Ensure upper y-limit is at least 0.1 for readability
    try:
        yvals = np.asarray(global_summary[metric].values, dtype=float)
        curr_bottom, curr_top = ax.get_ylim()
        ax.set_ylim(curr_bottom, max(0.1, np.nanmax(yvals)))
    except Exception:
        pass

    return ax


# ============================================================
# Conditioned decoding plot
# ============================================================

def plot_bootstrapped_delta_reg(
    *,
    cond_delta_bootstrap,
    model_type,
    metric='delta_score',
    ax=None,
    ci=95,
    target_name=None,
    condition_name=None,
):
    """
    Plot absolute conditioned decoding (global + delta) with CI.

    Uses fold-collapsed bootstrap estimates.
    """

    df = cond_delta_bootstrap.query('model == @model_type')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Guard: empty or missing required columns
    required_cols = {'bootstrap_id', 'condition_value',
                     'global_score', 'delta_score'}
    if len(df) == 0 or not required_cols.issubset(set(df.columns)):
        if target_name and condition_name:
            ax.set_title(f'{target_name} | {condition_name} ({model_type})')
        else:
            ax.set_title(f'Conditioned decoding ({model_type})')
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', alpha=0.6)
        ax.set_xticks([])
        ax.set_ylabel('R²')
        ax.set_xlabel('Condition')
        return ax

    condition_values = sorted(df['condition_value'].unique())

    means, lows, highs = [], [], []

    # Global baseline
    if len(df) > 0:
        global_vals = (
            df[['bootstrap_id', 'global_score']]
            .drop_duplicates()['global_score']
            .values
        )
        if global_vals.size > 0:
            global_ci = summarize_bootstrap_ci(global_vals, ci=ci)
            global_mean = global_ci['mean']
        else:
            global_mean = np.nan
    else:
        global_mean = np.nan

    for cond in condition_values:
        vals = (
            df
            .query('condition_value == @cond')
            .eval('global_score + delta_score')
            .values
        )
        if vals.size > 0:
            ci_out = summarize_bootstrap_ci(vals, ci=ci)
            means.append(ci_out['mean'])
            lows.append(ci_out['ci_low'])
            highs.append(ci_out['ci_high'])
        else:
            means.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)

    x = np.arange(len(condition_values))

    ax.errorbar(
        x,
        means,
        yerr=[
            np.array(means) - np.array(lows),
            np.array(highs) - np.array(means),
        ],
        fmt='o',
        capsize=4,
    )

    if np.isfinite(global_mean):
        ax.axhline(
            global_mean,
            linestyle='--',
            color='k',
            alpha=0.6,
            label='Global',
        )

    # Condition labels with sample sizes
    if 'n_samples' in df.columns:
        n_map = (
            df[['condition_value', 'n_samples']]
            .drop_duplicates()
            .set_index('condition_value')['n_samples']
            .to_dict()
        )
        labels = [
            f'{c} (n={int(n_map.get(c, 0))})'
            for c in condition_values
        ]
    else:
        labels = [str(c) for c in condition_values]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)

    ax.set_ylabel('R²')
    ax.set_xlabel(condition_name if condition_name else 'Condition')
    if target_name and condition_name:
        ax.set_title(f'{target_name} | {condition_name} ({model_type})')
    else:
        ax.set_title(f'Conditioned decoding ({model_type})')
    # Ensure upper y-limit is at least 0.1 for readability
    try:
        curr_bottom, curr_top = ax.get_ylim()
        # Use computed means if available; otherwise keep current top
        y_top = max([v for v in means if np.isfinite(v)] +
                    [curr_top]) if len(means) else curr_top
        ax.set_ylim(curr_bottom, max(0.1, y_top))
    except Exception:
        pass

    return ax


# ============================================================
# Combined figure
# ============================================================

def plot_pairwise_interaction_analysis_reg(
    *,
    analysis_out,
    model_type='ridge',
    metric='r2',
    ci=95,
    figsize=(10, 4),
    target_name=None,
    condition_name=None,
):
    """
    Two-panel plot:
      A) Global decoding
      B) Conditioned decoding by band
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_global_decoding_reg(
        global_summary=analysis_out['global_summary'],
        metric=metric,
        ax=axes[0],
        target_name=target_name,
    )

    plot_bootstrapped_delta_reg(
        cond_delta_bootstrap=analysis_out['cond_delta_bootstrap'],
        model_type=model_type,
        metric='delta_score',
        ax=axes[1],
        ci=ci,
        target_name=target_name,
        condition_name=condition_name,
    )

    plt.tight_layout()
    return fig


# ============================================================
# Per-condition diagnostic plots
# ============================================================

def plot_condition_scatterpanels_reg(
    *,
    analysis_out,
    target_name,
    condition_name,
    x_df=None,
    y_df=None,
    model_type='ridge',
    n_splits=5,
    random_state=0,
    max_points=5000,
):
    """
    For each condition value, make a scatter plot (CV: y_true vs y_pred).
    Uses KFold CV within each condition subset.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR

    def _make_reg(model_type):
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)
        elif model_type == 'kernel_ridge_rbf':
            model = KernelRidge(kernel='rbf', alpha=1.0, gamma=None)
        elif model_type == 'svr_rbf':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')
        return Pipeline([('scaler', StandardScaler()), ('model', model)])

    x_df, y_df = _pruned_or_fallback(
        analysis_out, x_df, y_df, condition_name=condition_name, min_count=200
    )

    condition_values = list(y_df[condition_name].cat.categories) \
        if hasattr(y_df[condition_name], 'cat') else sorted(y_df[condition_name].unique())

    n = len(condition_values)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No conditions', ha='center', va='center')
        ax.axis('off')
        return fig

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, cond in enumerate(condition_values):
        ax = axes[i]
        mask = (y_df[condition_name] == cond)
        X = x_df.loc[mask].values
        y = y_df.loc[mask, target_name].values

        if len(y) < max(3, n_splits):
            ax.text(0.5, 0.5, f'{cond}\nInsufficient samples',
                    ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        cv = KFold(n_splits=min(n_splits, len(y)),
                   shuffle=True, random_state=random_state)
        y_pred_all = np.full_like(y, np.nan, dtype=float)

        for tr, te in cv.split(X):
            model = _make_reg(model_type)
            model.fit(X[tr], y[tr])
            y_pred_all[te] = model.predict(X[te])

        # downsample points for plotting if large
        idx = np.arange(len(y))
        if len(idx) > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=max_points, replace=False)

        ax.scatter(y[idx], y_pred_all[idx], s=8, alpha=0.5)
        lo = float(np.nanmin([y[idx].min(), np.nanmin(y_pred_all[idx])]))
        hi = float(np.nanmax([y[idx].max(), np.nanmax(y_pred_all[idx])]))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(str(cond))

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'{target_name} | {condition_name}: per-condition CV scatter', y=1.02)
    fig.tight_layout()
    return fig


def plot_condition_confusion_heatmaps_clf(
    *,
    analysis_out,
    target_name,
    condition_name,
    x_df=None,
    y_df=None,
    model_type='logreg',
    n_splits=3,
    random_state=0,
    normalize=True,
):
    """
    For each condition value, make a confusion-matrix heatmap (CV within condition).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import discrete_decoders

    x_df, y_df = _pruned_or_fallback(
        analysis_out, x_df, y_df, condition_name=condition_name, min_count=200
    )

    condition_values = list(y_df[condition_name].cat.categories) \
        if hasattr(y_df[condition_name], 'cat') else sorted(y_df[condition_name].unique())

    n = len(condition_values)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No conditions', ha='center', va='center')
        ax.axis('off')
        return fig

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    im = None
    any_plotted = False

    for i, cond in enumerate(condition_values):
        ax = axes[i]
        mask = (y_df[condition_name] == cond)
        X = x_df.loc[mask].values
        y = y_df.loc[mask, target_name].astype(str).values

        if len(np.unique(y)) < 2 or len(y) < max(3, n_splits):
            ax.text(0.5, 0.5, f'{cond}\nInsufficient data',
                    ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Limit splits by the smallest class size to satisfy StratifiedKFold
        _, class_counts = np.unique(y, return_counts=True)
        min_class = int(class_counts.min()) if class_counts.size > 0 else 0
        n_splits_eff = min(n_splits, len(y), min_class)
        if n_splits_eff < 2:
            ax.text(0.5, 0.5, f'{cond}\nInsufficient per-class samples',
                    ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        skf = StratifiedKFold(
            n_splits=n_splits_eff, shuffle=True, random_state=random_state
        )
        y_true_all = []
        y_pred_all = []
        clf = discrete_decoders.make_decoder(model_type)

        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            y_pred = clf.predict(X[te])
            y_true_all.append(y[te])
            y_pred_all.append(y_pred)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        labels = sorted(np.unique(y_true_all))
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        if normalize:
            with np.errstate(invalid='ignore', divide='ignore'):
                cm = cm / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm, origin='lower', vmin=0,
                       vmax=1 if normalize else None)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(str(cond))
        any_plotted = True

    # colorbar on the right of the figure
    if any_plotted and im is not None:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cax)
    else:
        # No valid condition had enough samples → show a single message
        for ax in axes:
            ax.axis('off')
        ax0 = axes[0]
        ax0.axis('on')
        ax0.text(0.5, 0.5, 'No conditions with sufficient per-class samples',
                 ha='center', va='center')
        ax0.set_xticks([])
        ax0.set_yticks([])

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'{target_name} | {condition_name}: per-condition confusion', y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    return fig


# ============================================================
# Global diagnostic plots
# ============================================================

def plot_global_scatter_reg(
    *,
    analysis_out,
    target_name,
    model_type='ridge',
    n_splits=5,
    random_state=0,
    max_points=5000,
    x_df=None,
    y_df=None,
):
    """
    Global CV scatter (y_true vs y_pred) for a continuous target.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR

    def _make_reg(model_type):
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)
        elif model_type == 'kernel_ridge_rbf':
            model = KernelRidge(kernel='rbf', alpha=1.0, gamma=None)
        elif model_type == 'svr_rbf':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')
        return Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Prefer pruned data when available; otherwise use provided full data
    if analysis_out.get('x_pruned', None) is not None and analysis_out.get('y_pruned', None) is not None:
        X = analysis_out['x_pruned'].values
        y = analysis_out['y_pruned'][target_name].values
    else:
        assert x_df is not None and y_df is not None, 'x_df and y_df must be provided when analysis_out lacks pruned data'
        X = x_df.values
        y = y_df[target_name].values

    if len(y) < max(3, n_splits):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'Insufficient samples', ha='center', va='center')
        ax.axis('off')
        return fig

    cv = KFold(n_splits=min(n_splits, len(y)),
               shuffle=True, random_state=random_state)
    y_pred_all = np.full_like(y, np.nan, dtype=float)

    for tr, te in cv.split(X):
        model = _make_reg(model_type)
        model.fit(X[tr], y[tr])
        y_pred_all[te] = model.predict(X[te])

    # scatter (optionally downsample)
    idx = np.arange(len(y))
    if len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y[idx], y_pred_all[idx], s=8, alpha=0.5)
    lo = float(np.nanmin([y[idx].min(), np.nanmin(y_pred_all[idx])]))
    hi = float(np.nanmax([y[idx].max(), np.nanmax(y_pred_all[idx])]))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted (CV)')
    ax.set_title(f'Global: {target_name} ({model_type})')
    fig.tight_layout()
    return fig


def plot_global_confusion_heatmap_clf(
    *,
    analysis_out,
    target_name,
    model_type='logreg',
    n_splits=3,
    random_state=0,
    normalize=True,
    x_df=None,
    y_df=None,
):
    """
    Global CV confusion-matrix heatmap for a categorical target.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import discrete_decoders

    # Prefer pruned data when available; otherwise use provided full data
    if analysis_out.get('x_pruned', None) is not None and analysis_out.get('y_pruned', None) is not None:
        X = analysis_out['x_pruned'].values
        y = analysis_out['y_pruned'][target_name].astype(str).values
    else:
        assert x_df is not None and y_df is not None, 'x_df and y_df must be provided when analysis_out lacks pruned data'
        X = x_df.values
        y = y_df[target_name].astype(str).values

    if len(np.unique(y)) < 2 or len(y) < max(3, n_splits):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax.axis('off')
        return fig

    skf = StratifiedKFold(n_splits=min(n_splits, len(y)),
                          shuffle=True, random_state=random_state)
    y_true_all = []
    y_pred_all = []
    clf = discrete_decoders.make_decoder(model_type)

    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    labels = sorted(np.unique(y_true_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    if normalize:
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, origin='lower', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Global: {target_name} ({model_type})')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
