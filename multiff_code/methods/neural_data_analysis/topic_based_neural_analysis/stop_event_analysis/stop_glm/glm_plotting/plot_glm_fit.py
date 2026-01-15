from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import general_glm_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def _cluster_sort_key(label):
    # numeric-first fallback-to-string sort key
    try:
        return (0, float(label))
    except Exception:
        return (1, str(label))

def plot_coef_distributions(
    coefs_df,
    terms=None,
    rank='sig_FDR',           # 'none' | 'sig_FDR' | 'abs_med'
    rank_mode='fraction',     # when rank='sig_FDR': 'fraction' or 'count'
    ascending=True,           # False -> most significant/effectful at top
    *,
    cluster_col='cluster',    # or 'cluster_id' (auto-detected)
    show_legend=True,
    max_clusters_in_legend=20
):
    df = coefs_df.copy()

    # Resolve cluster column
    if cluster_col not in df.columns:
        if 'cluster_id' in df.columns:
            cluster_col = 'cluster_id'
        else:
            raise KeyError(f'Could not find cluster column {cluster_col!r} (or "cluster_id") in coefs_df.')

    # Candidate terms
    candidate_terms = (df['term'].unique().tolist()
                       if terms is None
                       else [t for t in terms if t in set(df['term'])])

    # Ranking
    if rank == 'sig_FDR' and 'sig_FDR' in df.columns:
        by = df.groupby('term')['sig_FDR']
        key = by.sum() if rank_mode == 'count' else by.mean()
    elif rank == 'abs_med':
        key = df.groupby('term')['coef'].apply(lambda s: float(np.median(np.abs(s))))
    else:
        key = None

    # Ordering
    if key is not None:
        ordered_terms_all = list(key.sort_values(ascending=ascending).index)
        terms_to_plot = [t for t in ordered_terms_all if t in candidate_terms]
    else:
        terms_to_plot = candidate_terms

    # Filter by significance (optional)
    if 'sig_FDR' in df.columns:
        keep_terms, skip_terms = [], []
        for t in terms_to_plot:
            n_sig = int(df.loc[df['term'] == t, 'sig_FDR'].sum())
            (keep_terms if n_sig > 0 else skip_terms).append(t)
        if skip_terms:
            print('Skipping terms with no significant clusters (sig_FDR=0):', skip_terms)
        terms_to_plot = keep_terms

    if not terms_to_plot:
        print('No terms with sig_FDR > 0 to plot.')
        return None

    # Consistent colors per cluster across all terms (sorted numerically if possible)
    clusters = df[cluster_col].astype(str).unique().tolist()
    clusters = sorted(clusters, key=_cluster_sort_key)
    base_cmap = plt.get_cmap('tab20')
    palette = [base_cmap(i % 20) for i in range(len(clusters))]
    cluster_to_color = {c: palette[i] for i, c in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=(8, 2.5 + 0.35 * len(terms_to_plot)))
    ytick, ylocs = [], []
    legend_handles = {}

    for k, term in enumerate(terms_to_plot):
        g = df[df['term'] == term].copy()
        y_base = np.full(g.shape[0], k, float)

        # Group by cluster; one scatter per cluster for stable handles
        for c_id, sub in g.groupby(g[cluster_col].astype(str)):
            idx = sub.index
            jitter = (np.random.rand(len(idx)) - 0.5) * 0.15
            xvals = sub['coef'].to_numpy()
            # map back to the y_base positions for these rows
            yvals = y_base[g.index.get_indexer(idx)] + jitter

            handle = ax.plot(
                xvals, yvals,
                'o', alpha=0.65, markersize=4,
                color=cluster_to_color.get(c_id, (0.3, 0.3, 0.3))
            )[0]

            if c_id not in legend_handles:
                legend_handles[c_id] = handle

        # # Median & IQR bars (per term)
        # med = float(np.median(g['coef']))
        # q1, q3 = np.percentile(g['coef'], [25, 75])
        # ax.plot([q1, q3], [k, k], lw=4, color='k')
        # ax.plot([med, med], [k - 0.18, k + 0.18], lw=2, color='k')

        # Right-side annotation: n_sig / total
        if 'sig_FDR' in g.columns:
            n_sig = int(np.asarray(g['sig_FDR']).sum())
            ax.text(1.005, k, f'{n_sig}/{g.shape[0]} sig', va='center', ha='left',
                    transform=ax.get_yaxis_transform())

        ytick.append(term); ylocs.append(k)

    ax.axvline(0, ls='--', lw=1, color='k')
    ax.set_yticks(ylocs); ax.set_yticklabels(ytick)
    ax.set_xlabel('Coefficient (β)')
    ax.set_title('Per-cluster coefficients by term (colored by cluster)')
    plt.tight_layout()

    # Legend (numeric-first, fallback to string; optionally truncated)
    if show_legend and legend_handles:
        items = sorted(legend_handles.items(), key=lambda kv: _cluster_sort_key(kv[0]))
        if len(items) > max_clusters_in_legend:
            shown = items[:max_clusters_in_legend]
            labels = [k for k, _ in shown]
            handles = [h for _, h in shown]
            rem = len(items) - max_clusters_in_legend
            ax.legend(handles, labels,
                      title=f'Clusters (first {max_clusters_in_legend}; +{rem} more)',
                      loc='best', fontsize=8)
        else:
            labels = [k for k, _ in items]
            handles = [h for _, h in items]
            ax.legend(handles, labels, title='Clusters', loc='best', fontsize=8)

    return fig



# ---------- plots ----------
# def plot_coef_distributions(
#     coefs_df,
#     terms=None,
#     rank='sig_FDR',           # 'none' | 'sig_FDR' | 'abs_med'
#     rank_mode='fraction',     # when rank='sig_FDR': 'fraction' or 'count'
#     ascending=True            # False -> most significant/effectful at top
# ):
#     """
#     Jittered dot + median/IQR per term across clusters.
#     If FDR info is present, hides terms with zero significant units (and prints them).
#     """
#     df = coefs_df.copy()

#     # Candidate terms
#     candidate_terms = df['term'].unique().tolist() if terms is None else [t for t in terms if t in set(df['term'])]

#     # Ranking
#     if rank == 'sig_FDR' and 'sig_FDR' in df.columns:
#         by = df.groupby('term')['sig_FDR']
#         key = by.sum() if rank_mode == 'count' else by.mean()
#     elif rank == 'abs_med':
#         key = df.groupby('term')['coef'].apply(lambda s: float(np.median(np.abs(s))))
#     else:
#         key = None

#     # Ordering
#     if key is not None:
#         ordered_terms_all = list(key.sort_values(ascending=ascending).index)
#         terms_to_plot = [t for t in ordered_terms_all if t in candidate_terms]
#     else:
#         terms_to_plot = candidate_terms

#     # Filter by significance (optional)
#     if 'sig_FDR' in df.columns:
#         keep_terms, skip_terms = [], []
#         for t in terms_to_plot:
#             n_sig = int(df.loc[df['term'] == t, 'sig_FDR'].sum())
#             (keep_terms if n_sig > 0 else skip_terms).append(t)
#         if skip_terms:
#             print('Skipping terms with no significant clusters (sig_FDR=0):', skip_terms)
#         terms_to_plot = keep_terms

#     if not terms_to_plot:
#         print('No terms with sig_FDR > 0 to plot.')
#         return None

#     fig, ax = plt.subplots(figsize=(8, 2.5 + 0.35 * len(terms_to_plot)))
#     ytick, ylocs = [], []

#     for k, term in enumerate(terms_to_plot):
#         g = df[df['term'] == term]
#         y = np.full(g.shape[0], k, float)
#         jitter = (np.random.rand(g.shape[0]) - 0.5) * 0.15
#         ax.plot(g['coef'].to_numpy(), y + jitter, 'o', alpha=0.45, markersize=4)

#         med = float(np.median(g['coef']))
#         q1, q3 = np.percentile(g['coef'], [25, 75])
#         ax.plot([q1, q3], [k, k], lw=4)
#         ax.plot([med, med], [k - 0.18, k + 0.18], lw=2)

#         if 'sig_FDR' in g.columns:
#             n_sig = int(np.asarray(g['sig_FDR']).sum())
#             ax.text(1.005, k, f'{n_sig}/{g.shape[0]} sig', va='center', ha='left', transform=ax.get_yaxis_transform())

#         ytick.append(term); ylocs.append(k)

#     ax.axvline(0, ls='--', lw=1)
#     ax.set_yticks(ylocs); ax.set_yticklabels(ytick)
#     ax.set_xlabel('Coefficient (β)')
#     ax.set_title('Per-cluster coefficients by term')
#     plt.tight_layout()
#     return fig


def plot_forest_for_term(coefs_df, term, top_n=30):
    """
    Forest plot of top-|z| units for a given term.
    Note: if SE is NaN (penalized without refit), CI bars will be NaN and omitted.
    """
    g = coefs_df[coefs_df['term'] == term].copy()
    if g.empty:
        print(f'No coefficients found for term {term}')
        return None
    g['abs_z'] = np.abs(g['z'])
    g = g.sort_values('abs_z', ascending=False).head(top_n)
    ci_lo = g['coef'] - 1.96 * g['se']
    ci_hi = g['coef'] + 1.96 * g['se']

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(g))))
    y = np.arange(len(g))
    ax.hlines(y, ci_lo, ci_hi, lw=2)
    ax.plot(g['coef'], y, 'o')
    ax.axvline(0, ls='--', lw=1)
    ax.set_yticks(y); ax.set_yticklabels([str(c) for c in g['cluster']])
    ax.set_xlabel('Coefficient (β)')
    ax.set_title(f'Forest (top {top_n} |z|) — {term}')
    plt.tight_layout()
    return fig


def plot_rate_ratio_hist(coefs_df, term, delta, bins=30, log=False, clip_q=None):
    """
    Histogram of per-unit rate ratios for a given term.
    - Drops non-finite or non-positive values before plotting.
    - Optional log scale on x and optional upper-tail clipping for readability.

    Parameters
    ----------
    coefs_df : DataFrame
        Must contain columns: ['term', 'rr'] at least.
    term : str
        Term name to filter rows.
    delta : float
        The delta used to compute the rate ratio (for labeling).
    bins : int or sequence
        Passed to matplotlib hist. Default 30.
    log : bool
        If True, use log scale on x-axis.
    clip_q : float or None
        If set (e.g., 0.995), clip RR values above this quantile to reduce the impact of outliers.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    g = coefs_df.loc[coefs_df['term'] == term]
    rr = g['rr'].to_numpy(dtype=float)

    # Keep only finite and strictly positive (log-x requires >0)
    m = np.isfinite(rr) & (rr > 0)
    rr = rr[m]

    fig, ax = plt.subplots(figsize=(6, 4))

    if rr.size == 0:
        ax.text(0.5, 0.5, 'No finite rate ratios to plot', ha='center', va='center')
        ax.axis('off')
        return fig

    # Optional gentle clipping of extreme tail for readability
    if clip_q is not None:
        hi = np.nanquantile(rr, clip_q)
        rr = np.minimum(rr, hi)

    # Optional log scaling: use log-spaced bins for a clean look
    if log:
        rr_min = rr.min()
        rr_max = rr.max()
        # Guard in case all values identical after clipping
        if rr_min == rr_max:
            rr_max = rr_min * 1.001
        # Build log-spaced edges if user passed an int for bins
        if isinstance(bins, int):
            import math
            lo = max(rr_min, np.finfo(float).tiny)
            edges = np.logspace(math.log10(lo), math.log10(rr_max), bins + 1)
        else:
            edges = bins
        ax.hist(rr, bins=edges)
        ax.set_xscale('log')
    else:
        ax.hist(rr, bins=bins)

    ax.axvline(1.0, ls='--', lw=1)
    ax.set_xlabel(f'Rate ratio for Δ{term}={delta}')
    ax.set_ylabel('Units')
    ax.set_title(f'{term}: distribution of rate ratios (n={rr.size})')
    return fig



def plot_model_quality(metrics_df):
    """
    Quick quality histograms across clusters: deviance explained and McFadden R².
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    de = metrics_df['deviance_explained'].to_numpy()
    ax[0].hist(de[np.isfinite(de)], bins=30)
    ax[0].set_xlabel('Deviance explained (1 - dev/null_dev)')
    ax[0].set_ylabel('Units')
    ax[0].set_title('Model fit (deviance explained)')
    r2 = metrics_df['mcfadden_R2'].to_numpy()
    ax[1].hist(r2[np.isfinite(r2)], bins=30)
    ax[1].set_xlabel('McFadden R²')
    ax[1].set_title('Model fit (McFadden R²)')
    plt.tight_layout()
    return fig



# ---------- optional comparison plots ----------
def plot_config_summary(summary_df):
    """
    Quick comparison bar charts (requires matplotlib). 
    Shows mean deviance explained, mean McFadden R², and sparsity per config.
    """
    if summary_df.empty:
        print('No summary to plot.')
        return None

    # order by mean deviance explained
    sdf = summary_df.copy().reset_index(drop=True)
    x = np.arange(len(sdf))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x, sdf['mean_deviance_explained'])
    ax1.set_xticks(x); ax1.set_xticklabels(sdf['config'], rotation=30, ha='right')
    ax1.set_ylabel('Mean deviance explained')
    ax1.set_title('Config comparison — deviance explained')
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x, sdf['mean_mcfadden_R2'])
    ax2.set_xticks(x); ax2.set_xticklabels(sdf['config'], rotation=30, ha='right')
    ax2.set_ylabel('Mean McFadden R²')
    ax2.set_title('Config comparison — McFadden R²')
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(x, sdf['sparsity'])
    ax3.set_xticks(x); ax3.set_xticklabels(sdf['config'], rotation=30, ha='right')
    ax3.set_ylabel('Sparsity (1 - nonzero share)')
    ax3.set_title('Config comparison — sparsity')
    plt.tight_layout()

    return {'deviance': fig1, 'mcfadden': fig2, 'sparsity': fig3}


def plot_coef_scatter(wide_coefs, cfg_a, cfg_b, term, limits=None):
    """
    Scatter plot comparing betas for one term across two configs.
    wide_coefs: from join_coefs_across_configs()
    cfg_a/cfg_b: labels to compare (must match in wide table)
    term: term name
    limits: tuple (lo, hi) to set symmetric axes, or None for auto
    """
    col_a = f'coef:{cfg_a}'
    col_b = f'coef:{cfg_b}'
    g = wide_coefs[wide_coefs['term'] == term][['cluster', col_a, col_b]].dropna()
    if g.empty:
        print(f'No overlapping coefficients for term "{term}" between {cfg_a} and {cfg_b}.')
        return None

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(g[col_a], g[col_b], 'o', alpha=0.6)
    lo = np.nanmin(np.concatenate([g[col_a].values, g[col_b].values]))
    hi = np.nanmax(np.concatenate([g[col_a].values, g[col_b].values]))
    if limits is None:
        pad = 0.05 * (hi - lo if np.isfinite(hi - lo) else 1.0)
        limits = (lo - pad, hi + pad)
    ax.plot(limits, limits, '--', lw=1)  # identity line
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel(cfg_a); ax.set_ylabel(cfg_b)
    ax.set_title(f'β comparison for term "{term}"')
    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt


def plot_insample_model_diagnostics(
    metrics_df,
    *,
    bins=20,
    dev_expl_xlim=(0.0, 0.1),
    mcfadden_xlim=(0.0, 0.08),
    ll_xlim=(-2000, 500),
    ll_per_obs_xlim=(-0.05, 0.05),
    show=True,
):
    """
    Plot IN-SAMPLE GLM performance diagnostics.

    Notes
    -----
    - Metrics reflect goodness-of-fit, not generalization.
    - Axis limits are applied for visualization only.
    """

    df = metrics_df.copy()

    # ---- derived quantities ----
    df['ll_improvement'] = df['llf'] - df['llnull']
    df['ll_improvement_per_obs'] = df['ll_improvement'] / df['n_obs']

    # ---- clipped copies for plotting only ----
    dev_expl_plot = df['deviance_explained'].clip(*dev_expl_xlim)
    mcf_plot = df['mcfadden_R2'].clip(*mcfadden_xlim)
    ll_plot = df['ll_improvement'].clip(*ll_xlim)
    ll_per_obs_plot = df['ll_improvement_per_obs'].clip(*ll_per_obs_xlim)

    # ---- figure layout ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    # 1. Deviance explained
    axes[0].hist(dev_expl_plot.dropna(), bins=bins)
    axes[0].axvline(np.nanmedian(df['deviance_explained']), linestyle='--')
    axes[0].set_xlim(dev_expl_xlim)
    axes[0].set_xlabel('Deviance explained')
    axes[0].set_ylabel('Number of neurons')
    axes[0].set_title('In-sample deviance explained')

    # 2. McFadden R²
    axes[1].hist(mcf_plot.dropna(), bins=bins)
    axes[1].axvline(np.nanmedian(df['mcfadden_R2']), linestyle='--')
    axes[1].set_xlim(mcfadden_xlim)
    axes[1].set_xlabel('McFadden $R^2$')
    axes[1].set_ylabel('Number of neurons')
    axes[1].set_title('In-sample pseudo-$R^2$')

    # 3. Consistency check
    axes[2].scatter(dev_expl_plot, mcf_plot, alpha=0.7)
    axes[2].set_xlim(dev_expl_xlim)
    axes[2].set_ylim(mcfadden_xlim)
    axes[2].set_xlabel('Deviance explained')
    axes[2].set_ylabel('McFadden $R^2$')
    axes[2].set_title('In-sample consistency check')

    # 4. Dependence on firing statistics
    axes[3].scatter(df['null_deviance'], dev_expl_plot, alpha=0.7)
    axes[3].set_ylim(dev_expl_xlim)
    axes[3].set_xlabel('Null deviance')
    axes[3].set_ylabel('Deviance explained')
    axes[3].set_title('Dependence on firing statistics')

    # 5. Log-likelihood gain
    axes[4].hist(ll_plot.dropna(), bins=bins)
    axes[4].axvline(0, linestyle='--')
    axes[4].set_xlim(ll_xlim)
    axes[4].set_xlabel('Log-likelihood gain')
    axes[4].set_ylabel('Number of neurons')
    axes[4].set_title('In-sample log-likelihood gain')

    # 6. Normalized gain
    axes[5].hist(ll_per_obs_plot.dropna(), bins=bins)
    axes[5].axvline(0, linestyle='--')
    axes[5].set_xlim(ll_per_obs_xlim)
    axes[5].set_xlabel('Δ log-likelihood per observation')
    axes[5].set_ylabel('Number of neurons')
    axes[5].set_title('In-sample normalized gain')

    plt.tight_layout()
    if show:
        plt.show()


def plot_cv_model_diagnostics(
    metrics_df,
    *,
    bins=40,
    cv_dev_xlim=(-0.1, 0.2),
    cv_ll_xlim=(-2000, 500),
    cv_ll_per_obs_xlim=(-0.05, 0.05),
    show=True,
):
    """
    Plot CROSS-VALIDATED GLM performance diagnostics.

    Notes
    -----
    - Metrics reflect generalization performance.
    - Axis limits are applied for visualization only.
    """

    df = metrics_df.copy()

    df['cv_ll_improvement'] = df['cv_loglik_improvement']
    df['cv_ll_improvement_per_obs'] = df['cv_ll_improvement'] / df['n_obs']

    cv_dev_plot = df['cv_deviance_explained'].clip(*cv_dev_xlim)
    cv_ll_plot = df['cv_ll_improvement'].clip(*cv_ll_xlim)
    cv_ll_per_obs_plot = df['cv_ll_improvement_per_obs'].clip(*cv_ll_per_obs_xlim)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    # 1. Deviance explained
    axes[0].hist(cv_dev_plot.dropna(), bins=bins)
    axes[0].axvline(np.nanmedian(df['cv_deviance_explained']), linestyle='--')
    axes[0].set_xlim(cv_dev_xlim)
    axes[0].set_xlabel('Deviance explained')
    axes[0].set_ylabel('Number of neurons')
    axes[0].set_title('CV deviance explained')

    # 2. Log-likelihood gain
    axes[1].hist(cv_ll_plot.dropna(), bins=bins)
    axes[1].axvline(0, linestyle='--')
    axes[1].set_xlim(cv_ll_xlim)
    axes[1].set_xlabel('Log-likelihood gain')
    axes[1].set_ylabel('Number of neurons')
    axes[1].set_title('CV log-likelihood gain')

    # 3. Normalized gain
    axes[2].hist(cv_ll_per_obs_plot.dropna(), bins=bins)
    axes[2].axvline(0, linestyle='--')
    axes[2].set_xlim(cv_ll_per_obs_xlim)
    axes[2].set_xlabel('Δ log-likelihood per observation')
    axes[2].set_ylabel('Number of neurons')
    axes[2].set_title('CV normalized gain')

    # 4. Dependence on firing statistics
    axes[3].scatter(df['null_deviance'], cv_dev_plot, alpha=0.7)
    axes[3].set_ylim(cv_dev_xlim)
    axes[3].set_xlabel('Null deviance')
    axes[3].set_ylabel('Deviance explained')
    axes[3].set_title('Dependence on firing statistics')

    # 5. Consistency check
    axes[4].scatter(cv_dev_plot, cv_ll_plot, alpha=0.7)
    axes[4].axhline(0, linestyle='--')
    axes[4].set_xlim(cv_dev_xlim)
    axes[4].set_ylim(cv_ll_xlim)
    axes[4].set_xlabel('Deviance explained')
    axes[4].set_ylabel('Log-likelihood gain')
    axes[4].set_title('CV consistency check')

    # 6. Effect of sparsity
    if 'zero_frac' in df.columns:
        axes[5].scatter(df['zero_frac'], cv_ll_plot, alpha=0.7)
        axes[5].axhline(0, linestyle='--')
        axes[5].set_ylim(cv_ll_xlim)
        axes[5].set_xlabel('Zero fraction')
        axes[5].set_ylabel('Log-likelihood gain')
        axes[5].set_title('Effect of sparsity')
    else:
        axes[5].axis('off')

    plt.tight_layout()
    if show:
        plt.show()
