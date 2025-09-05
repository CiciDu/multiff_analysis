import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

# ---------- stats helpers ----------
def add_fdr(coefs_df, alpha=0.05, by_term=True):
    df = coefs_df.copy()
    if by_term:
        df['q'] = np.nan
        df['sig_FDR'] = False
        for term, g in df.groupby('term', sort=False):
            p = g['p'].to_numpy()
            order = np.argsort(p)
            m = p.size
            ranks = np.empty(m, int); ranks[order] = np.arange(1, m+1)
            q = p * m / ranks
            # monotone adjustment
            q = np.minimum.accumulate(q[np.argsort(order)[::-1]])[::-1]
            df.loc[g.index, 'q'] = q
            df.loc[g.index, 'sig_FDR'] = q <= alpha
    else:
        p = df['p'].to_numpy()
        order = np.argsort(p)
        m = p.size
        ranks = np.empty(m, int); ranks[order] = np.arange(1, m+1)
        q = p * m / ranks
        q = np.minimum.accumulate(q[np.argsort(order)[::-1]])[::-1]
        df['q'] = q
        df['sig_FDR'] = df['q'] <= alpha
    return df

def add_rate_ratios(coefs_df, delta=1.0):
    df = coefs_df.copy()
    df['rr'] = np.exp(df['coef'] * delta)
    lo = df['coef'] - 1.96 * df['se']
    hi = df['coef'] + 1.96 * df['se']
    df['rr_lo'] = np.exp(lo * delta)
    df['rr_hi'] = np.exp(hi * delta)
    return df

def term_population_tests(coefs_df, terms=None):
    rows = []
    if terms is None:
        terms = coefs_df['term'].unique().tolist()
    for term in terms:
        g = coefs_df.loc[coefs_df['term'] == term]
        beta = g['coef'].to_numpy()
        beta = beta[np.isfinite(beta)]
        if beta.size == 0:
            continue
        try:
            w = stats.wilcoxon(beta, alternative='two-sided', zero_method='wilcox')
            p_w = w.pvalue
        except Exception:
            p_w = np.nan
        t = stats.ttest_1samp(beta, popmean=0.0, alternative='two-sided', nan_policy='omit')
        rows.append({
            'term': term,
            'n_units': beta.size,
            'beta_median': np.median(beta),
            'beta_mean': float(np.mean(beta)),
            'beta_std': float(np.std(beta, ddof=1)) if beta.size > 1 else np.nan,
            'p_wilcoxon': p_w,
            'p_ttest': t.pvalue
        })
    return pd.DataFrame(rows)

def fit_poisson_glm_per_cluster(
    df_X,
    df_Y,
    offset_log,
    cluster_ids=None,      # optional list of cluster IDs; defaults to df_Y columns
    cov_type='HC1',
):
    """
    Fit Poisson GLMs per cluster, using predictors from df_X and responses from df_Y.

    Parameters
    ----------
    df_X : pd.DataFrame
        Predictor DataFrame. Must include the offset column.
    df_Y : pd.DataFrame
        Response DataFrame. Each column = one cluster’s response (spike counts).
        Index must align with df_X.
    offset_log : np.ndarray
        Log-offset for each row.
    cluster_ids : list or None
        Identifiers for clusters. If None, uses df_Y.columns.
    cov_type : str
        Covariance type for GLM.


    Returns
    -------
    results, coefs_df, metrics_df
    """
    # --- Design matrix ---
    feature_names = df_X.columns

    # --- Responses ---
    if cluster_ids is None:
        cluster_ids = df_Y.columns

    results = {}
    coef_rows = []
    metrics_rows = []

    for cid in cluster_ids:
        yj = df_Y[cid].to_numpy()
        n = len(yj)

        if np.all(yj == 0):
            metrics_rows.append({
                'cluster': cid, 'n_obs': n, 'skipped_zero_unit': True,
                'deviance': np.nan, 'null_deviance': np.nan,
                'llf': np.nan, 'llnull': np.nan,
                'deviance_explained': np.nan, 'mcfadden_R2': np.nan
            })
            continue

        model = sm.GLM(yj, df_X, family=sm.families.Poisson(), offset=offset_log)
        res = model.fit(method='newton', maxiter=100, disp=False, cov_type=cov_type)
        results[cid] = res

        params = res.params.reindex(feature_names)
        ses    = res.bse.reindex(feature_names)
        pvals  = res.pvalues.reindex(feature_names)
        for name in feature_names:
            se = ses[name]
            coef_rows.append({
                'cluster': cid,
                'term': name,
                'coef': params[name],
                'se': se,
                'z': params[name] / se if np.isfinite(se) and se != 0 else np.nan,
                'p': pvals[name]
            })

        llf    = getattr(res, 'llf', np.nan)
        llnull = getattr(res, 'llnull', np.nan)
        dev    = getattr(res, 'deviance', np.nan)
        dev0   = getattr(res, 'null_deviance', np.nan)

        r2_dev = safe_deviance_explained(dev, dev0)
        r2_mcf = safe_mcfadden_r2(llf, llnull)

        metrics_rows.append({
            'cluster': cid, 'n_obs': n,
            'deviance': dev, 'null_deviance': dev0,
            'llf': llf, 'llnull': llnull,
            'deviance_explained': r2_dev, 'mcfadden_R2': r2_mcf
        })

    coefs_df = pd.DataFrame(coef_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return results, coefs_df, metrics_df


import numpy as np

def safe_deviance_explained(dev, null_dev, eps=1e-8):
    # Returns NaN if null_dev is too small or non-finite
    if not (np.isfinite(dev) and np.isfinite(null_dev)) or (abs(null_dev) < eps):
        return np.nan
    return 1.0 - (dev / null_dev)

def safe_mcfadden_r2(ll_full, ll_null, eps=1e-12):
    # ll_* should be log-likelihoods computed on the same data
    if not (np.isfinite(ll_full) and np.isfinite(ll_null)) or (abs(ll_null) < eps):
        return np.nan
    return 1.0 - (ll_full / ll_null)


# ---------- plots ----------
import numpy as np
import matplotlib.pyplot as plt

def plot_coef_distributions(
    coefs_df,
    terms=None,
    rank='sig_FDR',                 # 'none' | 'sig_FDR' | 'abs_med'
    rank_mode='fraction',        # when rank='sig_FDR': 'fraction' or 'count'
    ascending=True              # False => most significant/effectful at top
):
    """
    Plot per-cluster coefficient distributions, with optional term ranking.
    Only plots terms with sig_FDR > 0; prints the others.
    """
    df = coefs_df.copy()

    # Decide candidate terms
    if terms is None:
        candidate_terms = df['term'].unique().tolist()
    else:
        candidate_terms = [t for t in terms if t in set(df['term'])]

    # Build ranking key
    if rank == 'sig_FDR' and 'sig_FDR' in df.columns:
        by = df.groupby('term')['sig_FDR']
        if rank_mode == 'count':
            key = by.sum()
        else:  # 'fraction'
            key = by.mean()
    elif rank == 'abs_med':
        key = df.groupby('term')['coef'].apply(lambda s: float(np.median(np.abs(s))))
    else:
        key = None

    # Order terms
    if key is not None:
        ordered_terms_all = list(key.sort_values(ascending=ascending).index)
        terms_to_plot = [t for t in ordered_terms_all if t in candidate_terms]
    else:
        terms_to_plot = candidate_terms

    # Filter by sig_FDR
    if 'sig_FDR' in df.columns:
        keep_terms = []
        skip_terms = []
        for t in terms_to_plot:
            n_sig = int(df.loc[df['term'] == t, 'sig_FDR'].sum())
            if n_sig > 0:
                keep_terms.append(t)
            else:
                skip_terms.append(t)
        if skip_terms:
            print('Skipping terms with no significant clusters (sig_FDR=0):', skip_terms)
        terms_to_plot = keep_terms

    if not terms_to_plot:
        print('No terms with sig_FDR > 0 to plot.')
        return None

    fig, ax = plt.subplots(figsize=(8, 3 + 0.6*len(terms_to_plot)))
    ytick = []; ylocs = []

    for k, term in enumerate(terms_to_plot):
        g = df[df['term'] == term]
        y = np.full(g.shape[0], k, float)
        jitter = (np.random.rand(g.shape[0]) - 0.5) * 0.15
        ax.plot(g['coef'].to_numpy(), y + jitter, 'o', alpha=0.45, markersize=4)

        med = float(np.median(g['coef']))
        q1, q3 = np.percentile(g['coef'], [25, 75])
        ax.plot([q1, q3], [k, k], lw=4)
        ax.plot([med, med], [k - 0.18, k + 0.18], lw=2)

        if 'sig_FDR' in g.columns:
            n_sig = int(np.asarray(g['sig_FDR']).sum())
            ax.text(
                1.005, k, f'{n_sig}/{g.shape[0]} sig',
                va='center', ha='left', transform=ax.get_yaxis_transform()
            )

        ytick.append(term); ylocs.append(k)

    ax.axvline(0, ls='--', lw=1)
    ax.set_yticks(ylocs); ax.set_yticklabels(ytick)
    ax.set_xlabel('Coefficient (β)')
    ax.set_title('Per-cluster coefficients by term')
    plt.tight_layout()
    return fig

def plot_forest_for_term(coefs_df, term, top_n=30):
    g = coefs_df[coefs_df['term'] == term].copy()
    if g.empty:
        print(f'No coefficients found for term {term}')
        return None
    g['abs_z'] = np.abs(g['z'])
    g = g.sort_values('abs_z', ascending=False).head(top_n)
    ci_lo = g['coef'] - 1.96*g['se']
    ci_hi = g['coef'] + 1.96*g['se']

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3*len(g))))
    y = np.arange(len(g))
    ax.hlines(y, ci_lo, ci_hi, lw=2)
    ax.plot(g['coef'], y, 'o')
    ax.axvline(0, ls='--', lw=1)
    ax.set_yticks(y); ax.set_yticklabels([str(c) for c in g['cluster']])
    ax.set_xlabel('Coefficient (β)')
    ax.set_title(f'Forest (top {top_n} |z|) — {term}')
    plt.tight_layout()
    return fig

def plot_rate_ratio_hist(coefs_df, term, delta=1.0, bins=30):
    g = add_rate_ratios(coefs_df[coefs_df['term'] == term], delta=delta)
    rr = g['rr'].to_numpy()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(rr, bins=bins)
    ax.axvline(1.0, ls='--', lw=1)
    ax.set_xlabel(f'Rate ratio for Δ{term}={delta}')
    ax.set_ylabel('Units')
    ax.set_title(f'Population rate ratios — {term}')
    plt.tight_layout()
    return fig

def plot_model_quality(metrics_df):
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
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

def glm_mini_report(
    df_X,
    df_Y,
    offset_log,
    feature_names=None,         # if None, inferred from df_X
    cluster_ids=None,           # if None, uses df_Y.columns
    alpha=0.05,
    delta_for_rr=1.0,
    forest_term=None,
    forest_top_n=30,
    cov_type='HC1',
    show_plots=True,
    save_dir=None
):
    """
    Fits Poisson GLMs per cluster from (df_X, df_Y), adds FDR & rate ratios,
    runs population tests, and produces core figures. Returns dict of outputs.

    Parameters
    ----------
    df_X : pd.DataFrame
        Predictors (+ offset column). Index must align with df_Y.
    df_Y : pd.DataFrame
        Responses; each column is one cluster/unit. Index must align with df_X.
    offset_log : np.ndarray
        Log-offset for each row.
    feature_names : list[str] or None
        Predictor columns to use. If None, inferred as df_X.columns.
    cluster_ids : list or None
        Which response columns (IDs) to fit. If None, uses df_Y.columns.
    alpha : float
        FDR threshold.
    delta_for_rr : float
        Step size used when converting β to rate ratios: rr = exp(β * delta_for_rr).
    forest_term : str or None
        Which term to use for forest / rr histogram. Defaults to first feature.
    forest_top_n : int
        Top |z| coefficients to display in forest plot.
    cov_type : str
        Covariance estimator for GLM (e.g., 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3').
    show_plots : bool
        If True, display figures. If False, close figures to avoid display.
    save_dir : str or Path or None
        If provided, saves CSV tables and PNG figures here.

    Returns
    -------
    out : dict
        {
          'results': dict{cluster_id: GLMResults},
          'coefs_df': pd.DataFrame,
          'metrics_df': pd.DataFrame,
          'population_tests_df': pd.DataFrame,
          'figures': dict[str, matplotlib.figure.Figure or None]
        }
    """
    # infer feature list if not provided
    if feature_names is None:
        feature_names = df_X.columns
    # run fits
    results, coefs_df, metrics_df = fit_poisson_glm_per_cluster(
        df_X=df_X,
        df_Y=df_Y,
        offset_log=offset_log,
        cluster_ids=cluster_ids,
        cov_type=cov_type,
    )

    # add FDR & rate ratios and population tests
    coefs_df = add_fdr(coefs_df, alpha=alpha, by_term=True)
    coefs_df = add_rate_ratios(coefs_df, delta=delta_for_rr)
    pop_tests = term_population_tests(coefs_df)

    # figures
    figs = {}
    figs['coef_dists'] = plot_coef_distributions(coefs_df)
    figs['model_quality'] = plot_model_quality(metrics_df)
    if forest_term is None and len(feature_names) > 0:
        forest_term = feature_names[0]
    elif forest_term not in feature_names:
        print(f'Term {forest_term} not found in feature_names. Using {feature_names[0]} instead.')
        forest_term = feature_names[0]
    figs['forest'] = plot_forest_for_term(coefs_df, term=forest_term, top_n=forest_top_n)
    figs['rr_hist'] = plot_rate_ratio_hist(coefs_df, term=forest_term, delta=delta_for_rr)

    # save if requested
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        coefs_df.to_csv(save_dir / 'coefs.csv', index=False)
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
        pop_tests.to_csv(save_dir / 'population_tests.csv', index=False)
        for name, fig in figs.items():
            if fig is not None:
                fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')

    # show or close
    if show_plots:
        for fig in figs.values():
            if fig is not None:
                fig.show()
    else:
        for fig in figs.values():
            if fig is not None:
                plt.close(fig)

    return {
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df,
        'population_tests_df': pop_tests,
        'figures': figs
    }


# ---------- report-only (if you already have fits) ----------
def glm_report_from_tables(coefs_df, metrics_df, feature_names, alpha=0.05, delta_for_rr=1.0,
                           forest_term=None, forest_top_n=30, show_plots=True, save_dir=None):
    coefs_df = add_fdr(coefs_df, alpha=alpha, by_term=True)
    coefs_df = add_rate_ratios(coefs_df, delta=delta_for_rr)
    pop_tests = term_population_tests(coefs_df)

    figs = {}
    figs['coef_dists'] = plot_coef_distributions(coefs_df)
    figs['model_quality'] = plot_model_quality(metrics_df)
    if forest_term is None and len(feature_names) > 0:
        forest_term = feature_names[0]
    figs['forest'] = plot_forest_for_term(coefs_df, term=forest_term, top_n=forest_top_n)
    figs['rr_hist'] = plot_rate_ratio_hist(coefs_df, term=forest_term, delta=delta_for_rr)

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        coefs_df.to_csv(save_dir / 'coefs.csv', index=False)
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
        pop_tests.to_csv(save_dir / 'population_tests.csv', index=False)
        for name, fig in figs.items():
            if fig is not None:
                fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')

    if show_plots:
        for fig in figs.values():
            if fig is not None:
                fig.show()
    else:
        for fig in figs.values():
            if fig is not None:
                plt.close(fig)

    return {'coefs_df': coefs_df, 'metrics_df': metrics_df, 'population_tests_df': pop_tests, 'figures': figs}
