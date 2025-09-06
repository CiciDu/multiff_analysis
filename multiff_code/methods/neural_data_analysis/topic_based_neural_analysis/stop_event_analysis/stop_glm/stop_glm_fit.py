# -*- coding: utf-8 -*-
# Poisson GLM per cluster with optional regularization + GroupKFold CV
# -------------------------------------------------------------------
# Highlights
# - Regularization via statsmodels.GLM.fit_regularized (L1/L2/Elastic-Net).
# - Hyper-parameter tuning (alpha × l1_wt) using GroupKFold or KFold.
# - Optional L1 "refit on support" to recover SE/p-values for inference.
# - NaN-robust FDR (penalized fits may lack SE/p).
# - Progress printing per cluster and best hyper-params per cluster.
#
# API compatibility
# - Public functions preserved: add_fdr, add_rate_ratios, term_population_tests,
#   fit_poisson_glm_per_cluster, glm_mini_report
# - New knobs are optional and default to your original (no penalty).
from . import plot_glm_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold


def add_fdr(coefs_df, alpha: float = 0.05, by_term: bool = True,
            p_col: str = 'p', out_q_col: str = 'q', out_sig_col: str = 'sig_FDR'):
    """
    Add Benjamini–Hochberg FDR-adjusted q-values to a coefficient table.

    Parameters
    ----------
    coefs_df : pd.DataFrame
        Must contain a p-value column (default: 'p'). If `by_term=True`, must also
        contain a 'term' column to adjust within each term separately.
    alpha : float, default 0.05
        FDR threshold used to create the boolean significance flag.
    by_term : bool, default True
        If True, perform BH correction *within each term* (grouped by 'term').
        If False, perform BH correction across all rows at once.
    p_col : str, default 'p'
        Name of the column containing (two-sided) p-values.
    out_q_col : str, default 'q'
        Name of the output column for BH-adjusted q-values.
    out_sig_col : str, default 'sig_FDR'
        Name of the output column for the significance flag (q <= alpha).

    Returns
    -------
    pd.DataFrame
        Copy of `coefs_df` with two added columns:
          - `out_q_col`: BH-adjusted q-values (NaN where input p was NaN).
          - `out_sig_col`: boolean flag, True where q <= alpha (False for NaN p).

    Notes
    -----
    - NaN p-values are ignored in the ranking and remain NaN in q; their flag is False.
    - Implements the standard BH procedure with right-to-left monotonicity and q clipped to [0,1].
    """

    def _bh_adjust(p: np.ndarray) -> np.ndarray:
        """Vectorized BH on a 1D array of p-values (may contain NaNs)."""
        p = np.asarray(p, float)
        q = np.full_like(p, np.nan, dtype=float)

        mask = np.isfinite(p)
        m = int(mask.sum())
        if m == 0:
            return q  # nothing to adjust

        p_valid = p[mask]
        order = np.argsort(p_valid)                  # ascending p
        p_sorted = p_valid[order]

        ranks = np.arange(1, m + 1, dtype=float)
        q_sorted = p_sorted * m / ranks              # raw BH
        # enforce monotone non-decreasing q along increasing p (right-to-left cummin)
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0.0, 1.0)

        # scatter back to original positions
        inv = np.empty_like(order)
        inv[order] = np.arange(m)
        q_valid = q_sorted[inv]
        q[mask] = q_valid
        return q

    df = coefs_df.copy()

    # Prepare/initialize outputs
    df[out_q_col] = np.nan
    df[out_sig_col] = False

    if by_term:
        if 'term' not in df.columns:
            raise KeyError("add_fdr(by_term=True) requires a 'term' column.")
        for term, g in df.groupby('term', sort=False):
            pvals = g[p_col].to_numpy(dtype=float, copy=False)
            qvals = _bh_adjust(pvals)
            df.loc[g.index, out_q_col] = qvals
            df.loc[g.index, out_sig_col] = (qvals <= alpha)
    else:
        pvals = df[p_col].to_numpy(dtype=float, copy=False)
        qvals = _bh_adjust(pvals)
        df[out_q_col] = qvals
        df[out_sig_col] = (qvals <= alpha)

    # Ensure boolean dtype for the flag
    df[out_sig_col] = df[out_sig_col].astype(bool)
    return df


def add_rate_ratios(coefs_df, delta=1.0):
    """
    Convert coefficients to rate ratios for a delta-step in the predictor:
        rr = exp(beta * delta)
    Also returns Wald 95% CI on the same scale when SE is available.
    """
    df = coefs_df.copy()
    df['rr'] = np.exp(df['coef'] * delta)
    lo = df['coef'] - 1.96 * df['se']
    hi = df['coef'] + 1.96 * df['se']
    df['rr_lo'] = np.exp(lo * delta)
    df['rr_hi'] = np.exp(hi * delta)
    return df


def term_population_tests(coefs_df, terms=None):
    """
    Across clusters, test whether median/mean(beta) differs from 0 for each term.
    Uses Wilcoxon signed-rank and one-sample t-test on finite betas only.
    """
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
            w = stats.wilcoxon(beta, alternative='two-sided',
                               zero_method='wilcox')
            p_w = w.pvalue
        except Exception:
            p_w = np.nan
        t = stats.ttest_1samp(
            beta, popmean=0.0, alternative='two-sided', nan_policy='omit')
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


def safe_deviance_explained(dev, null_dev, eps=1e-8):
    """Return 1 - dev/null_dev with guards for small/NaN denominators."""
    if not (np.isfinite(dev) and np.isfinite(null_dev)) or (abs(null_dev) < eps):
        return np.nan
    return 1.0 - (dev / null_dev)


def safe_mcfadden_r2(ll_full, ll_null, eps=1e-12):
    """Return 1 - ll_full/ll_null with guards."""
    if not (np.isfinite(ll_full) and np.isfinite(ll_null)) or (abs(ll_null) < eps):
        return np.nan
    return 1.0 - (ll_full / ll_null)


# ---------- internal GLM/CV helpers ----------
def _poisson_mu_from_eta(eta):
    # Canonical link for Poisson: mu = exp(eta)
    return np.exp(eta)


def _poisson_loglik(y, mu, eps=1e-12):
    """
    Pointwise log-likelihood for Poisson(y | mu).
    We retain the full logpmf (including log(y!)) so CV scores are interpretable.
    """
    mu = np.clip(mu, eps, None)
    return stats.poisson(mu).logpmf(y)


def _poisson_deviance(y, mu, eps=1e-12):
    """
    2 * sum( y*log(y/mu) - (y - mu) ), with the convention y*log(y/.) := 0 when y=0.
    """
    mu = np.clip(mu, eps, None)
    y = np.asarray(y, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu))


def _build_folds(n, *, n_splits=5, groups=None, cv_splitter=None, random_state=0):
    """
    Return a list of (train_idx, valid_idx) pairs.
      - If cv_splitter == 'blocked_time': forward-chaining, contiguous blocks (row order = time).
      - Else if groups is not None: GroupKFold.
      - Else: KFold(shuffle=True).
    """
    idx = np.arange(n)

    if cv_splitter == 'blocked_time':
        # forward-chaining: use earlier rows to predict a later contiguous block
        # split the range [0, n) into n_splits equal-ish blocks as validation sets
        bps = np.linspace(0, n, n_splits + 1, dtype=int)
        folds = []
        for k in range(1, len(bps)):
            start, stop = bps[k-1], bps[k]
            valid = idx[start:stop]
            train = idx[:start]  # only past rows (no look-ahead)
            if len(train) == 0 or len(valid) == 0:
                continue
            folds.append((train, valid))
        return folds

    if groups is not None:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(idx, groups=groups))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(idx))


def _score_from_beta(beta, X_val, off_val, y_val, metric='loglik'):
    """
    Score coefficients on validation data.
    - 'loglik': sum of log-likelihoods (higher is better)
    - 'deviance': negative deviance (higher is better)
    """
    eta_val = X_val @ beta + off_val
    mu_val = _poisson_mu_from_eta(eta_val)
    if metric == 'deviance':
        return -_poisson_deviance(y_val, mu_val)
    return float(np.sum(_poisson_loglik(y_val, mu_val)))


def _validate_shapes(df_X, df_Y, offset_log, cluster_ids):
    """
    Ensure consistent shapes and return canonical objects.
    """
    feature_names = list(df_X.columns)
    X = df_X.to_numpy()
    off = np.asarray(offset_log).reshape(-1)
    n = X.shape[0]
    if cluster_ids is None:
        cluster_ids = list(df_Y.columns)
    if len(off) != n:
        raise ValueError(
            'offset_log length must match the number of rows in df_X/df_Y.')
    return feature_names, X, off, n, cluster_ids


def _grid_for_regularization(regularization, alpha_grid, l1_wt_grid):
    """
    Decide hyper-parameter grid for search.
    - 'none' => just (alpha=0, l1_wt=0) which means unpenalized MLE.
    - otherwise return user-provided grids.
    """
    if regularization == 'none':
        return [0.0], [0.0]
    return list(alpha_grid), list(l1_wt_grid)


def _cv_score_for_combo(y, X, off, folds, alpha, l1_wt, regularization, cv_metric):
    """
    Evaluate one (alpha, l1_wt) pair via CV; return the mean validation score.
    If any fold fails to converge, return -inf for this combo.
    """
    scores = []
    for tr_idx, va_idx in folds:
        X_tr, X_va = X[tr_idx], X[va_idx]
        off_tr, off_va = off[tr_idx], off[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = sm.GLM(y_tr, X_tr, family=sm.families.Poisson(), offset=off_tr)
        try:
            if regularization == 'none' or alpha == 0.0:
                res_k = model.fit(method='newton', maxiter=100, disp=False)
            else:
                res_k = model.fit_regularized(
                    alpha=alpha, L1_wt=l1_wt, maxiter=300)
        except Exception:
            return -np.inf  # failed combo
        beta_k = np.asarray(res_k.params)
        scores.append(_score_from_beta(
            beta_k, X_va, off_va, y_va, metric=cv_metric))
    return float(np.mean(scores)) if scores else -np.inf


def _refit_l1_support_if_needed(res, y, X, off, cov_type, regularization, alpha, l1_wt,
                                refit_on_support, use_overdispersion_scale=False):
    """
    If we chose a pure L1 model (l1_wt==1) with alpha>0 and refit_on_support=True,
    refit an unpenalized GLM on the selected columns to recover SE/p.
    """
    used_refit = False
    if (regularization != 'none') and (alpha > 0.0) and (abs(l1_wt - 1.0) < 1e-12) and refit_on_support:
        beta = np.asarray(res.params)
        support = np.isfinite(beta) & (np.abs(beta) > 0)

        # If everything survived L1 (rare), still refit to get SEs:
        if support.sum() == len(beta):
            model_full = sm.GLM(y, X, family=sm.families.Poisson(), offset=off)
            scale_arg = 'X2' if use_overdispersion_scale else None
            res_full = model_full.fit(method='newton', maxiter=400, disp=False,
                                      cov_type=cov_type, scale=scale_arg)
            # keep res_full directly
            return res_full, True

        if 0 < support.sum() < len(beta):
            Xs = X[:, support]
            model_s = sm.GLM(y, Xs, family=sm.families.Poisson(), offset=off)
            scale_arg = 'X2' if use_overdispersion_scale else None
            res_s = model_s.fit(method='newton', maxiter=400, disp=False,
                                cov_type=cov_type, scale=scale_arg)
            # Pack back to full-length arrays (NaN for dropped features)
            beta_full = np.full_like(beta, np.nan, dtype=float)
            beta_full[support] = res_s.params
            bse_full = np.full_like(beta_full, np.nan, dtype=float)
            p_full = np.full_like(beta_full, np.nan, dtype=float)
            try:
                bse_full[support] = res_s.bse
                p_full[support] = res_s.pvalues
            except Exception:
                pass

            class _Pack(object):
                ...
            res_pack = _Pack()
            res_pack.params = beta_full
            res_pack.bse = bse_full
            res_pack.pvalues = p_full
            res_pack.llf = getattr(res_s, 'llf', np.nan)
            res_pack.llnull = getattr(res_s, 'llnull', np.nan)
            res_pack.deviance = getattr(res_s, 'deviance', np.nan)
            res_pack.null_deviance = getattr(res_s, 'null_deviance', np.nan)

            res = res_pack
            used_refit = True
    return res, used_refit


def _metrics_from_result(res, y, X, off):
    """
    Try to read fit metrics from the result; if absent (common for penalized),
    compute them explicitly.
    """
    try:
        llf = float(res.llf)
        llnull = float(res.llnull)
        dev = float(res.deviance)
        dev0 = float(res.null_deviance)
        return llf, llnull, dev, dev0
    except Exception:
        beta = np.asarray(res.params)
        eta = X @ beta + off
        mu = _poisson_mu_from_eta(eta)
        llf = float(np.sum(_poisson_loglik(y, mu)))
        # Intercept-only null model for llnull/dev0
        n = X.shape[0]
        X0 = np.ones((n, 1))
        model0 = sm.GLM(y, X0, family=sm.families.Poisson(), offset=off)
        res0 = model0.fit(method='newton', maxiter=200, disp=False)
        eta0 = X0 @ res0.params + off
        mu0 = _poisson_mu_from_eta(eta0)
        llnull = float(np.sum(_poisson_loglik(y, mu0)))
        dev = float(_poisson_deviance(y, mu))
        dev0 = float(_poisson_deviance(y, mu0))
        return llf, llnull, dev, dev0


def _collect_coef_rows(feature_names, cid, res, alpha, l1_wt, used_refit):
    """
    Create tidy coefficient rows for a cluster (SE/p may be NaN for penalized fits).
    """
    params = pd.Series(np.asarray(res.params), index=feature_names)
    ses = pd.Series(getattr(res, 'bse', np.nan), index=feature_names)
    pvals = pd.Series(getattr(res, 'pvalues', np.nan), index=feature_names)
    rows = []
    for name in feature_names:
        se = ses[name]
        beta = params[name]
        z = (beta / se) if np.isfinite(se) and se != 0 else np.nan
        p = pvals[name] if np.isfinite(pvals.get(name, np.nan)) else np.nan
        rows.append({
            'cluster': cid,
            'term': name,
            'coef': float(beta),
            'se': float(se) if np.isfinite(se) else np.nan,
            'z': float(z) if np.isfinite(z) else np.nan,
            'p': float(p) if np.isfinite(p) else np.nan,
            'alpha': float(alpha),
            'l1_wt': float(l1_wt),
            'regularization': ('none' if alpha == 0.0 else 'elasticnet'),
            'refit_on_support': bool(used_refit)
        })
    return rows


def _collect_metric_row(cid, n, llf, llnull, dev, dev0, alpha, l1_wt):
    """One tidy metrics row for a cluster."""
    return {
        'cluster': cid, 'n_obs': n,
        'deviance': dev, 'null_deviance': dev0,
        'llf': llf, 'llnull': llnull,
        'deviance_explained': safe_deviance_explained(dev, dev0),
        'mcfadden_R2': safe_mcfadden_r2(llf, llnull),
        'alpha': float(alpha), 'l1_wt': float(l1_wt)
    }


# ---------- orchestration ----------


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ---------- core fitting (refactored with MLE short-circuit) ----------
def fit_poisson_glm_per_cluster(
    df_X,
    df_Y,
    offset_log,
    cluster_ids=None,          # optional list of cluster IDs; defaults to df_Y columns
    cov_type='HC1',
    # --- regularization & tuning ---
    # 'none' | 'elasticnet'  (L1/L2/EN via alpha + l1_wt)
    regularization='none',
    alpha_grid=(0.1, 0.3, 1.0),
    l1_wt_grid=(1.0, 0.5, 0.0),  # 1.0=L1, 0.0=L2, mid=EN
    n_splits=5,
    cv_metric='loglik',        # 'loglik' | 'deviance' (assumed higher=better)
    groups=None,               # group labels for GroupKFold; len == n_rows
    # if best l1_wt==1.0 and alpha>0, refit OLS-GLM on support for SE/p
    refit_on_support=True,
    return_cv_tables=True,     # if True, also return concatenated CV grid across clusters
    *,
    cv_splitter=None,          # None | 'blocked_time' (forward-chaining CV)
    # if True, unpenalized fits use scale='X2' (quasi-Poisson-like SEs)
    use_overdispersion_scale=False
):
    """
    Fit Poisson GLMs per cluster with optional Elastic-Net and CV hyperparam search.
    """

    feature_names, X, off, n, cluster_ids = _validate_shapes(df_X, df_Y, offset_log, cluster_ids)

    # -------- MLE short-circuit (no tuning, no folds, no CV tables) --------
    no_tuning = (
        regularization == 'none'
        and tuple(alpha_grid) == (0.0,)
        and tuple(l1_wt_grid) == (0.0,)
    )
    if no_tuning:
        results = {}
        coef_rows = []
        metrics_rows = []
        for i, cid in enumerate(cluster_ids, 1):
            print(f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
            y = df_Y[cid].to_numpy()

            if np.all(y == 0):
                metrics_rows.append({
                    'cluster': cid, 'n_obs': n, 'skipped_zero_unit': True,
                    'deviance': np.nan, 'null_deviance': np.nan,
                    'llf': np.nan, 'llnull': np.nan,
                    'deviance_explained': np.nan, 'mcfadden_R2': np.nan,
                    'alpha': 0.0, 'l1_wt': 0.0
                })
                continue

            # single unpenalized fit
            res = _fit_once(
                y=y, X=X, off=off, cov_type=cov_type,
                regularization='none', alpha=0.0, l1_wt=0.0,
                use_overdispersion_scale=use_overdispersion_scale
            )
            used_refit = False  # irrelevant for MLE

            # metrics
            llf, llnull, dev, dev0 = _metrics_from_result(res, y, X, off)

            # rows
            coef_rows.extend(_collect_coef_rows(feature_names, cid, res, 0.0, 0.0, used_refit))
            metrics_rows.append(_collect_metric_row(cid, n, llf, llnull, dev, dev0, 0.0, 0.0))
            results[cid] = res
            print('  done (MLE)', flush=True)

        results_dict = pd.Series(results).to_dict()
        coef_df = pd.DataFrame(coef_rows)
        metrics_df = pd.DataFrame(metrics_rows)
        # no CV tables in MLE short-circuit
        return results_dict, coef_df, metrics_df, pd.DataFrame()

    # ----------------------- regular path with tuning ------------------------
    folds = _build_folds(n, n_splits=n_splits, groups=groups, cv_splitter=cv_splitter)

    results = {}        # {cluster_id: fitted_result}
    coef_rows = []      # list of per-term rows → DataFrame
    metrics_rows = []   # list of per-cluster rows → DataFrame
    cv_tables = []      # collect per-cluster CV grids (if return_cv_tables=True)

    for i, cid in enumerate(cluster_ids, 1):
        print(f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
        y = df_Y[cid].to_numpy()

        # Degenerate all-zero unit cannot estimate meaningful Poisson GLM
        if np.all(y == 0):
            metrics_rows.append({
                'cluster': cid, 'n_obs': n, 'skipped_zero_unit': True,
                'deviance': np.nan, 'null_deviance': np.nan,
                'llf': np.nan, 'llnull': np.nan,
                'deviance_explained': np.nan, 'mcfadden_R2': np.nan,
                'alpha': np.nan, 'l1_wt': np.nan
            })
            continue

        # Grid-search via CV for best (alpha, l1_wt); then fit the winner on full data
        best, cv_table = _hyperparam_search(
            y=y, X=X, off=off, folds=folds, cov_type=cov_type,
            regularization=regularization,
            alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
            cv_metric=cv_metric,
            return_table=True,
            use_overdispersion_scale=use_overdispersion_scale
        )
        res = best['res']

        # Optional L1 refit to recover SE/p-values
        res, used_refit = _refit_l1_support_if_needed(
            res, y, X, off, cov_type,
            regularization, best['alpha'], best['l1_wt'],
            refit_on_support,
            use_overdispersion_scale=use_overdispersion_scale
        )

        # Fit metrics
        llf, llnull, dev, dev0 = _metrics_from_result(res, y, X, off)

        # Collect tables
        coef_rows.extend(_collect_coef_rows(feature_names, cid, res, best['alpha'], best['l1_wt'], used_refit))
        metrics_rows.append(_collect_metric_row(cid, n, llf, llnull, dev, dev0, best['alpha'], best['l1_wt']))
        results[cid] = res

        if return_cv_tables:
            cv_table = cv_table.copy()
            cv_table['cluster'] = cid
            cv_tables.append(cv_table)

        print(f'  done (best alpha={best["alpha"]}, l1_wt={best["l1_wt"]})', flush=True)

    # finalize outputs
    results_dict = pd.Series(results).to_dict()
    coef_df = pd.DataFrame(coef_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    if return_cv_tables:
        cv_tables_df = pd.concat(cv_tables, ignore_index=True) if len(cv_tables) else pd.DataFrame()
        return results_dict, coef_df, metrics_df, cv_tables_df

    return results_dict, coef_df, metrics_df, pd.DataFrame()



import numpy as np
import pandas as pd
import statsmodels.api as sm

def _fit_once(y, X, off, cov_type, alpha, l1_wt, regularization, *, use_overdispersion_scale=False):
    """
    Fit a single Poisson GLM.
      - Unpenalized: GLM.fit(method='newton', cov_type, scale=('X2' if QP-like))
      - Penalized:   GLM.fit_regularized(alpha, L1_wt)  (no SEs; refit later if needed)
    """
    y = np.asarray(y, dtype=float)
    off = None if off is None else np.asarray(off, dtype=float)

    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=off)

    # ------- fast path: plain MLE or alpha <= 0 or regularization='none' -------
    if (regularization == 'none') or (alpha is None) or (alpha <= 0.0):
        # Quasi-Poisson flavor: use Pearson chi^2 / dof as scale => SEs inflate with overdispersion
        scale_arg = 'X2' if use_overdispersion_scale else None
        res = model.fit(method='newton', maxiter=200, disp=False,
                        cov_type=cov_type, scale=scale_arg)
        for k, v in dict(alpha=0.0, l1_wt=0.0, regularization='none',
                         is_penalized=False, cov_type=cov_type,
                         qp_like=bool(use_overdispersion_scale)).items():
            setattr(res, k, v)
        return res

    # --------------------------- penalized (EN / L1 / L2) ---------------------------
    res = model.fit_regularized(alpha=float(alpha), L1_wt=float(l1_wt),
                                maxiter=1000, cnvrg_tol=1e-8)
    for k, v in dict(alpha=float(alpha), l1_wt=float(l1_wt), regularization=str(regularization),
                     is_penalized=True, cov_type=cov_type, qp_like=False).items():
        setattr(res, k, v)
    return res


def _hyperparam_search(
    y, X, off, folds, cov_type, regularization,
    alpha_grid, l1_wt_grid, cv_metric,
    *, return_table: bool = False, use_overdispersion_scale: bool = False
):
    """
    Grid-search over (alpha, l1_wt). **Fast-paths**:
      - If regularization='none' and grids are (0.0,), skip CV entirely and fit once (MLE).
      - If the grid contains exactly one combo, skip CV scoring and fit that combo once.
    """
    alpha_list, l1wt_list = _grid_for_regularization(regularization, alpha_grid, l1_wt_grid)

    # --------------------- fast path A: plain MLE (no tuning) ---------------------
    no_tuning = (regularization == 'none'
                 and tuple(alpha_list) == (0.0,)
                 and tuple(l1wt_list) == (0.0,))
    if no_tuning:
        res_full = _fit_once(y, X, off, cov_type, 0.0, 0.0, 'none',
                             use_overdispersion_scale=use_overdispersion_scale)
        best = {'score': 0.0, 'alpha': 0.0, 'l1_wt': 0.0, 'res': res_full}
        if return_table:
            table = pd.DataFrame({'alpha': [0.0], 'l1_wt': [0.0], 'score': [0.0],
                                  'fit_attempted': [True], 'fit_ok': [True], 'error': [None],
                                  'rank': [1], 'selected': [True]})
            return best, table
        return best

    # ----------------- fast path B: single-combo grid (no CV scoring) -----------------
    if (len(alpha_list) == 1) and (len(l1wt_list) == 1):
        a, l = float(alpha_list[0]), float(l1wt_list[0])
        try:
            res_full = _fit_once(y, X, off, cov_type, a, l, regularization,
                                 use_overdispersion_scale=use_overdispersion_scale)
            best = {'score': 0.0, 'alpha': a, 'l1_wt': l, 'res': res_full}
            if return_table:
                table = pd.DataFrame({'alpha': [a], 'l1_wt': [l], 'score': [0.0],
                                      'fit_attempted': [True], 'fit_ok': [True], 'error': [None],
                                      'rank': [1], 'selected': [True]})
                return best, table
            return best
        except Exception as e:
            # fall back to MLE if penalized fit fails
            res_fallback = _fit_once(y, X, off, cov_type, 0.0, 0.0, 'none',
                                     use_overdispersion_scale=use_overdispersion_scale)
            best = {'score': -np.inf, 'alpha': a, 'l1_wt': l, 'res': res_fallback}
            if return_table:
                table = pd.DataFrame({'alpha': [a], 'l1_wt': [l], 'score': [np.nan],
                                      'fit_attempted': [True], 'fit_ok': [False], 'error': [str(e)],
                                      'rank': [1], 'selected': [True]})
                return best, table
            return best

    # --------------------------- regular CV-scored path ---------------------------
    best = {'score': -np.inf, 'alpha': 0.0, 'l1_wt': 0.0, 'res': None}
    records = []

    for alpha in alpha_list:
        for l1_wt in l1wt_list:
            score = _cv_score_for_combo(y, X, off, folds, alpha, l1_wt, regularization, cv_metric)
            rec = {'alpha': float(alpha), 'l1_wt': float(l1_wt), 'score': float(score),
                   'fit_attempted': False, 'fit_ok': None, 'error': None}

            if score > best['score']:
                rec['fit_attempted'] = True
                try:
                    res_full = _fit_once(y, X, off, cov_type, alpha, l1_wt, regularization,
                                         use_overdispersion_scale=use_overdispersion_scale)
                    rec['fit_ok'] = True
                    best.update(score=score, alpha=float(alpha), l1_wt=float(l1_wt), res=res_full)
                except Exception as e:
                    rec['fit_ok'] = False
                    rec['error'] = str(e)
                    print(f'Could not fit full model for this combo: alpha={alpha}, l1_wt={l1_wt}: {e}')

            records.append(rec)

    if best['res'] is None:  # robust fallback to MLE
        best['alpha'], best['l1_wt'] = 0.0, 0.0
        best['res'] = _fit_once(y, X, off, cov_type, 0.0, 0.0, 'none',
                                use_overdispersion_scale=use_overdispersion_scale)

    table = pd.DataFrame.from_records(records).sort_values('score', ascending=False, kind='mergesort')
    table['rank'] = np.arange(1, len(table) + 1)
    table['selected'] = (np.isclose(table['alpha'], best['alpha'])) & (np.isclose(table['l1_wt'], best['l1_wt']))
    print(f'    best hyperparams: alpha={best["alpha"]}, l1_wt={best["l1_wt"]}, score={best["score"]:.3f}', flush=True)

    return (best, table) if return_table else best


# === Full, paste-ready code ===

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---- helpers assumed to exist elsewhere in your codebase ----
# add_fdr, add_rate_ratios, term_population_tests
# plot_glm_fit.plot_coef_distributions / plot_model_quality / plot_forest_for_term / plot_rate_ratio_hist
# safe_deviance_explained / safe_mcfadden_r2 (or inline versions below)

def _safe_deviance_explained(dev, null_dev, eps=1e-8):
    if not (np.isfinite(dev) and np.isfinite(null_dev)) or (abs(null_dev) < eps):
        return np.nan
    return 1.0 - (dev / null_dev)

def _safe_mcfadden_r2(ll_full, ll_null, eps=1e-12):
    if not (np.isfinite(ll_full) and np.isfinite(ll_null)) or (abs(ll_null) < eps):
        return np.nan
    return 1.0 - (ll_full / ll_null)


def fit_poisson_glm_per_cluster_fast_mle(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    cluster_ids=None,
    cov_type: str = 'HC1',    # use 'nonrobust' for max speed
    maxiter: int = 100,
    show_progress: bool = False,
):
    """
    Ultra-fast MLE per cluster: no CV/inference/plots. NumPy-centric inner loop.
    Returns (results_dict, coefs_df, metrics_df).
    """
    feature_names = list(df_X.columns)
    X = np.asarray(df_X, dtype=float, order='F')
    off = None if offset_log is None else np.asarray(offset_log, dtype=float)

    effective_cluster_ids = list(df_Y.columns) if cluster_ids is None else list(cluster_ids)

    coef_rows = []
    metrics_rows = []
    results = {}

    fam = sm.families.Poisson()

    for i, cid in enumerate(effective_cluster_ids, 1):
        if show_progress:
            print(f'Fitting cluster {i}/{len(effective_cluster_ids)}: {cid} ...', flush=True)

        y = np.asarray(df_Y[cid], dtype=float)
        n = y.shape[0]

        if np.all(y == 0):
            metrics_rows.append({
                'cluster': cid, 'n_obs': n, 'skipped_zero_unit': True,
                'deviance': np.nan, 'null_deviance': np.nan,
                'llf': np.nan, 'llnull': np.nan,
                'deviance_explained': np.nan, 'mcfadden_R2': np.nan
            })
            continue

        model = sm.GLM(y, X, family=fam, offset=off)
        res = model.fit(method='newton', maxiter=maxiter, disp=False, cov_type=cov_type)
        results[cid] = res

        params = np.asarray(res.params, dtype=float)
        ses    = np.asarray(res.bse, dtype=float)
        pvals  = np.asarray(res.pvalues, dtype=float)

        for j, name in enumerate(feature_names):
            se = ses[j]
            z  = params[j] / se if np.isfinite(se) and se != 0.0 else np.nan
            coef_rows.append({
                'cluster': cid,
                'term': name,
                'coef': params[j],
                'se': se,
                'z': z,
                'p': pvals[j],
            })

        llf    = getattr(res, 'llf', np.nan)
        llnull = getattr(res, 'llnull', np.nan)
        dev    = getattr(res, 'deviance', np.nan)
        dev0   = getattr(res, 'null_deviance', np.nan)

        r2_dev = _safe_deviance_explained(dev, dev0)
        r2_mcf = _safe_mcfadden_r2(llf, llnull)

        metrics_rows.append({
            'cluster': cid, 'n_obs': n,
            'deviance': dev, 'null_deviance': dev0,
            'llf': llf, 'llnull': llnull,
            'deviance_explained': r2_dev, 'mcfadden_R2': r2_mcf
        })

        if show_progress:
            print('  done (MLE)', flush=True)

    coefs_df = pd.DataFrame.from_records(coef_rows)
    metrics_df = pd.DataFrame.from_records(metrics_rows)
    return results, coefs_df, metrics_df


# ==== COMPLETE, PASTE-READY FUNCTION ====
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def glm_mini_report(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    # kept for API compatibility; df_X columns are used internally
    feature_names=None,
    cluster_ids=None,           # if None, uses df_Y.columns
    alpha: float = 0.05,
    delta_for_rr: float = 1.0,
    forest_term=None,
    forest_top_n: int = 30,
    cov_type: str = 'HC1',
    show_plots: bool = True,
    save_dir=None,
    # --- knobs for the slower/general path (assumes your fitter exists) ---
    regularization: str = 'none',      # 'none' | 'elasticnet'
    alpha_grid=(0.1, 0.3, 1.0),
    l1_wt_grid=(1.0, 0.5, 0.0),
    n_splits: int = 5,
    cv_metric: str = 'loglik',
    groups=None,
    refit_on_support: bool = True,
    cv_splitter=None,
    use_overdispersion_scale: bool = False,
    # --- fast path + toggles (assumes your fast fitter exists) ---
    fast_mle: bool = False,            # if True: use ultra-fast MLE fitter (no CV)
    make_plots: bool = True,           # if False: skip creating figure objects entirely
    do_inference: bool = True,         # if False: skip rate ratios & pop-tests (but still add FDR when plotting)
    return_cv_tables: bool = True,     # forwarded to slower path; ignored in fast path
    show_progress: bool = False,       # per-cluster printouts in fast MLE path
):
    """
    Fits per-cluster Poisson GLMs, optionally does inference and plots.

    ALWAYS behavior:
      - If make_plots=True, ensures coefs_df contains an FDR column (sig_FDR),
        even when do_inference=False, so coefficient plots can display
        'n_sig / n_total' per term.

    Returns
    -------
    out : dict
        {
          'results': dict{cluster_id: GLMResults},
          'coefs_df': pd.DataFrame,
          'metrics_df': pd.DataFrame,
          'population_tests_df': pd.DataFrame,   # empty if do_inference=False
          'figures': dict[str, matplotlib.figure.Figure or None],
          'cv_tables_df': pd.DataFrame
        }
    """
    # --- Resolve feature/cluster lists safely ---
    feature_names = list(df_X.columns) if feature_names is None else list(feature_names)
    effective_cluster_ids = list(df_Y.columns) if cluster_ids is None else list(cluster_ids)

    # --- Choose fit path (fast MLE vs general) ---
    if fast_mle:
        # You must have defined this elsewhere:
        # results, coefs_df, metrics_df = fit_poisson_glm_per_cluster_fast_mle(...)
        results, coefs_df, metrics_df = fit_poisson_glm_per_cluster_fast_mle(
            df_X=df_X,
            df_Y=df_Y,
            offset_log=offset_log,
            cluster_ids=effective_cluster_ids,
            cov_type=cov_type,
            maxiter=100,
            show_progress=show_progress,
        )
        cv_tables_df = pd.DataFrame()
    else:
        # Detect plain-MLE case to disable CV table creation
        no_tuning = (
            (regularization == 'none') and
            (tuple(alpha_grid) == (0.0,)) and
            (tuple(l1_wt_grid) == (0.0,))
        )
        rcv = bool(return_cv_tables and not no_tuning)

        # You must have defined this elsewhere:
        # results, coefs_df, metrics_df, cv_tables_df = fit_poisson_glm_per_cluster(...)
        results, coefs_df, metrics_df, cv_tables_df = fit_poisson_glm_per_cluster(
            df_X=df_X,
            df_Y=df_Y,
            offset_log=offset_log,
            cluster_ids=effective_cluster_ids,
            cov_type=cov_type,
            regularization=regularization,
            alpha_grid=alpha_grid,
            l1_wt_grid=l1_wt_grid,
            n_splits=n_splits,
            cv_metric=cv_metric,
            groups=groups,
            refit_on_support=refit_on_support,
            cv_splitter=cv_splitter,
            use_overdispersion_scale=use_overdispersion_scale,
            return_cv_tables=rcv,
        )

    # --- Inference / post-processing ---
    # 1) Full inference path
    if do_inference:
        # You must have add_fdr / add_rate_ratios / term_population_tests defined/imported
        coefs_df = add_fdr(coefs_df, alpha=alpha, by_term=True)
        coefs_df = add_rate_ratios(coefs_df, delta=delta_for_rr)
        pop_tests = term_population_tests(coefs_df)
    else:
        # 2) Lightweight path: ensure FDR exists if we're going to plot
        if make_plots and ('sig_FDR' not in coefs_df.columns):
            coefs_df = add_fdr(coefs_df, alpha=alpha, by_term=True)
        pop_tests = pd.DataFrame()

    # --- Figures (optional) ---
    figs = {}
    if make_plots:
        # these plotting helpers are assumed to exist in your codebase as plot_glm_fit.*
        figs['coef_dists'] = plot_glm_fit.plot_coef_distributions(coefs_df)
        figs['model_quality'] = plot_glm_fit.plot_model_quality(metrics_df)

        # Resolve forest term robustly
        if forest_term is None and len(feature_names) > 0:
            ft = feature_names[0]
        elif forest_term in feature_names:
            ft = forest_term
        else:
            ft = feature_names[0] if feature_names else None
            if forest_term is not None and ft is not None:
                print(f'Term {forest_term} not found in feature_names. Using {ft} instead.')
        if ft is not None:
            figs['forest'] = plot_glm_fit.plot_forest_for_term(coefs_df, term=ft, top_n=forest_top_n)
            figs['rr_hist'] = plot_glm_fit.plot_rate_ratio_hist(coefs_df, term=ft, delta=delta_for_rr)
        else:
            figs['forest'] = None
            figs['rr_hist'] = None

    # --- Save tables/figures if requested ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        coefs_df.to_csv(save_dir / 'coefs.csv', index=False)
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
        pop_tests.to_csv(save_dir / 'population_tests.csv', index=False)
        if make_plots:
            for name, fig in figs.items():
                if fig is not None:
                    fig.savefig(save_dir / f'{name}.png',
                                dpi=150, bbox_inches='tight')

    # --- Show or close figures ---
    if make_plots:
        if show_plots:
            for fig in figs.values():
                if fig is not None:
                    plt.show()
        else:
            for fig in figs.values():
                if fig is not None:
                    plt.close(fig)

    # --- Final summary print (works even when cluster_ids=None) ---
    try:
        n_clusters = int(metrics_df['cluster'].nunique()) if 'cluster' in metrics_df.columns else len(effective_cluster_ids)
    except Exception:
        n_clusters = len(effective_cluster_ids)
    if fast_mle:
        print(f'[glm_mini_report] Finished fast MLE on {n_clusters} clusters, '
              f'cov_type={cov_type}, inference={do_inference}, plots={make_plots}')

    return {
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df,
        'population_tests_df': pop_tests,
        'figures': figs,
        'cv_tables_df': cv_tables_df,
    }
