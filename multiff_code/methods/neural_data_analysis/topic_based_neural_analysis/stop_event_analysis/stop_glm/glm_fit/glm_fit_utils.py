from statsmodels.stats.multitest import multipletests
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_plotting import plot_spikes, plot_glm_fit


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


def add_fdr(coefs_df, alpha: float = 0.05, by_term: bool = True,
            p_col: str = 'p', out_q_col: str = 'q', out_sig_col: str = 'sig_FDR',
            group_col: str = 'group'):
    """
    Add Benjamini–Hochberg FDR-adjusted q-values to a coefficient table.

    Parameters
    ----------
    coefs_df : pd.DataFrame
        Must contain a p-value column (default: 'p').
    alpha : float, default 0.05
        FDR threshold used to create the boolean significance flag.
    by_term : bool, default True
        If True, perform BH correction within groups. Groups are defined by
        `group_col` if present, otherwise fall back to 'term' for compatibility.
        If False, perform BH correction across all rows at once.
    p_col : str, default 'p'
        Name of the column containing (two-sided) p-values.
    out_q_col : str, default 'q'
        Name of the output column for BH-adjusted q-values.
    out_sig_col : str, default 'sig_FDR'
        Name of the output column for the significance flag (q <= alpha).
    group_col : str, default 'group'
        Column name to use for grouping when `by_term=True`. If this column is
        not present, the function will fall back to using 'term' if available.

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
        # Prefer explicit group column; fall back to 'term' for backward compatibility
        if group_col in df.columns:
            group_by = group_col
            print(
                f'Conducting FDR adjustment by group (column="{group_col}")...')
        elif 'term' in df.columns:
            group_by = 'term'
            print('Conducting FDR adjustment by term (fallback)...')
        else:
            raise KeyError(
                "add_fdr(by_term=True) requires either a 'group' or 'term' column.")

        for term, g in df.groupby(group_by, sort=False):
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


def normalize_term_labels(term_series: pd.Series) -> pd.Series:
    """
    Normalize per-column term labels into group terms suitable for FDR grouping.
    Rules (kept intentionally simple and idempotent):
      - Basis-expanded columns use ':' separators (e.g., 'speed:b0:2', 'angle:sin1').
        The base term is everything before the first ':'  -> 'speed', 'angle'.
      - Condition one-hots created via pd.get_dummies(..., prefix='cond') look like
        'cond_x', 'cond_y'. These collapse to 'cond'.
      - Otherwise, leave the label unchanged.
    """
    s = term_series.astype(str)
    # basis-style: keep prefix before first ':'
    base = s.str.split(':', n=1, expand=True)[0]
    # cond one-hots: collapse 'cond_*' -> 'cond'
    base = base.str.replace(r'^(cond)_.+$', r'\1', regex=True)
    return base


def _rank_biserial_from_wilcoxon(beta):
    # For one-sample Wilcoxon signed-rank: r_rb = (W_plus - W_minus) / (W_plus + W_minus)
    x = np.asarray(beta, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    # split signs, rank absolute values
    abs_ranks = stats.rankdata(np.abs(x))
    r_plus = abs_ranks[x > 0].sum()
    r_minus = abs_ranks[x < 0].sum()
    denom = r_plus + r_minus
    return np.nan if denom == 0 else (r_plus - r_minus) / denom


def summarize_bootstrap_ci(a, stat_fn, n_boot=2000, alpha=0.05, random_state=0):
    rng = np.random.default_rng(random_state)
    x = np.asarray(a, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    stats_boot = []
    n = x.size
    for _ in range(n_boot):
        samp = x[rng.integers(0, n, n)]
        stats_boot.append(stat_fn(samp))
    qlo, qhi = np.quantile(stats_boot, [alpha/2, 1 - alpha/2])
    return float(qlo), float(qhi)


def term_population_tests(coefs_df, terms=None, alpha=0.05, do_fdr=True):
    """
    Across clusters, test whether mean/median(beta) differs from 0 for each term.
    Adds flags and FDR-corrected q-values for easier spotting of significance.
    """
    rows = []
    if terms is None:
        terms = coefs_df['term'].unique().tolist()

    for term in terms:
        g = coefs_df.loc[coefs_df['term'] == term, 'coef']
        beta = g.to_numpy()
        beta = beta[np.isfinite(beta)]
        if beta.size == 0:
            continue

        # Wilcoxon signed-rank
        try:
            w = stats.wilcoxon(beta, alternative='two-sided',
                               zero_method='wilcox')
            p_w = float(w.pvalue)
        except Exception:
            p_w = np.nan

        # One-sample t-test
        t = stats.ttest_1samp(
            beta, popmean=0.0, alternative='two-sided', nan_policy='omit')
        p_t = float(t.pvalue)

        rows.append({
            'term': term,
            'n_units': beta.size,
            'beta_median': np.median(beta),
            'beta_mean': float(np.mean(beta)),
            'beta_std': float(np.std(beta, ddof=1)) if beta.size > 1 else np.nan,
            'p_wilcoxon': p_w,
            'p_ttest': p_t,
        })

    out = pd.DataFrame(rows)

    # FDR correction across terms
    if do_fdr and not out.empty:
        for col in ['p_wilcoxon', 'p_ttest']:
            if out[col].notna().any():
                pvals = out[col].fillna(1.0).to_numpy()
                _, q, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                out[col.replace('p_', 'q_')] = q
                out['sig_' + col.split('_')[1]] = q < alpha
                out['stars_' + col.split('_')[1]] = pd.cut(
                    q,
                    bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                    labels=['***', '**', '*', '']
                )

    # Sort terms by min q-value (most significant first)
    if 'q_wilcoxon' in out or 'q_ttest' in out:
        qmin = out[['q_wilcoxon', 'q_ttest']].min(axis=1, skipna=True)
        out = out.assign(q_min=qmin).sort_values('q_min', ascending=True)

    return out


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


def _poisson_mu_from_eta(eta, clip=(-50.0, 50.0)):
    """
    Canonical link for Poisson: mu = exp(eta).
    """
    if clip is not None:
        lo, hi = clip
        eta = np.clip(np.asarray(eta, dtype=float), lo, hi)
    else:
        eta = np.asarray(eta, dtype=float)
    return np.exp(eta)


def _poisson_loglik(y, mu, eps=1e-12):
    """
    Pointwise log-likelihood for Poisson(y | mu).

    We return the full log pmf (including log(y!)) so CV scores are comparable.
    This function returns one value per observation, not a sum.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), eps, None)
    return stats.poisson(mu).logpmf(y)


def _poisson_deviance(y, mu, eps=1e-12):
    """
    Poisson deviance:
        D = 2 * sum( y * log(y / mu) - (y - mu) )
    with the convention y*log(y/.) := 0 when y == 0.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), eps, None)

    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y > 0.0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


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


def attach_feature_names(model, feature_names):
    # statsmodels uses model.data.xnames under the hood
    model.data.xnames = list(feature_names)
    # convenience: also keep on the model for later checks
    model.feature_names_in_ = list(feature_names)


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


# --- numerically safe mean ---
def poisson_mu_from_eta(eta, mu_max=1e6):
    # cap eta so exp() never overflows; keep mu in [1e-12, mu_max]
    eta = np.asarray(eta, float)
    eta = np.clip(eta, -20.0, np.log(mu_max))
    mu = np.exp(eta)
    return np.clip(mu, 1e-12, mu_max)


def poisson_loglik(y, mu):
    # full log pmf; require finite positive mu
    mu = np.asarray(mu, float)
    mu = np.clip(mu, 1e-12, None)
    return stats.poisson(mu).logpmf(y)


def poisson_deviance(y, mu):
    # 2 * sum( y*log(y/mu) - (y - mu) ), with y*log(y/.) = 0 at y=0
    y = np.asarray(y,  float)
    mu = np.asarray(mu, float)
    mu = np.clip(mu, 1e-12, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * float(np.sum(term - (y - mu)))


def _null_glm_stats_poisson_with_offset(y, off):
    """
    Closed-form null (intercept-only) Poisson MLE with offset:
      b0 = log( sum(y) / sum(exp(off)) )
    Returns (llnull, dev0).
    """
    y = np.asarray(y, float)
    n = y.shape[0]
    off = np.zeros(n, float) if off is None else np.asarray(
        off, float).reshape(-1)
    denom = float(np.sum(np.exp(off)))
    if denom <= 0 or not np.isfinite(denom):
        # fall back to no-offset formula
        denom = n
    lam0 = float(np.sum(y)) / max(denom, 1e-12)
    mu0 = lam0 * np.exp(off)  # exp(b0 + off)
    llnull = float(np.sum(poisson_loglik(y, mu0)))
    dev0 = poisson_deviance(y, mu0)
    return llnull, dev0


def metrics_from_result(res, y, X, off):
    """
    Robustly return (llf, llnull, dev, dev0). If attrs are missing OR NaN,
    recompute from params. Handles NaN betas (treated as 0 for dropped cols).
    """
    # First, try to read attrs
    llf = getattr(res, 'llf', np.nan)
    llnull = getattr(res, 'llnull', np.nan)
    dev = getattr(res, 'deviance', np.nan)
    dev0 = getattr(res, 'null_deviance', np.nan)

    need_recompute = not (np.isfinite(llf) and np.isfinite(dev))
    need_null = not (np.isfinite(llnull) and np.isfinite(dev0))

    # Recompute full-model metrics if needed
    if need_recompute:
        beta = np.asarray(getattr(res, 'params', np.nan), float)
        # dropped features → 0 contribution
        beta = np.nan_to_num(beta, nan=0.0)
        offv = None if off is None else np.asarray(off, float).reshape(-1)
        eta = X @ beta + (0.0 if offv is None else offv)
        mu = poisson_mu_from_eta(eta)
        llf = float(np.sum(poisson_loglik(y, mu)))
        dev = poisson_deviance(y, mu)

    # Recompute null-model metrics if needed (closed-form; no fitting)
    if need_null:
        llnull, dev0 = _null_glm_stats_poisson_with_offset(y, off)

    return float(llf), float(llnull), float(dev), float(dev0)


def collect_coef_rows(feature_names, cid, res, alpha, l1_wt, used_refit):
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
            'refit_on_support': bool(used_refit),
            'used_ridge_fallback': bool(getattr(res, 'used_ridge_fallback', False)
                                        or getattr(res, 'is_penalized', False))
        })
    return rows


def collect_metric_row(cid, n, llf, llnull, dev, dev0, alpha, l1_wt):
    """One tidy metrics row for a cluster."""
    return {
        'cluster': cid, 'n_obs': n,
        'deviance': dev, 'null_deviance': dev0,
        'llf': llf, 'llnull': llnull,
        'deviance_explained': safe_deviance_explained(dev, dev0),
        'mcfadden_R2': safe_mcfadden_r2(llf, llnull),
        'alpha': float(alpha), 'l1_wt': float(l1_wt)
    }


def _hyperparam_search(
    y, X, off, folds, cov_type, regularization,
    alpha_grid, l1_wt_grid, cv_metric,
    *, return_table: bool = False, use_overdispersion_scale: bool = False,
    feature_names=None
):
    """
    Grid-search over (alpha, l1_wt). **Fast-paths**:
      - If regularization='none' and grids are (0.0,), skip CV entirely and fit once (MLE).
      - If the grid contains exactly one combo, skip CV scoring and fit that combo once.
    """
    alpha_list, l1wt_list = _grid_for_regularization(
        regularization, alpha_grid, l1_wt_grid)

    # --------------------- fast path A: plain MLE (no tuning) ---------------------
    no_tuning = (regularization == 'none'
                 and tuple(alpha_list) == (0.0,)
                 and tuple(l1wt_list) == (0.0,))
    if no_tuning:
        res_full = _fit_once(y, X, off, cov_type, 0.0, 0.0, 'none',
                             use_overdispersion_scale=use_overdispersion_scale, feature_names=feature_names)
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
                                 use_overdispersion_scale=use_overdispersion_scale, feature_names=feature_names)
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
                                     use_overdispersion_scale=use_overdispersion_scale, feature_names=feature_names)
            best = {'score': -np.inf, 'alpha': a,
                    'l1_wt': l, 'res': res_fallback}
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
            score = _cv_score_for_combo(
                y, X, off, folds, alpha, l1_wt, regularization, cv_metric)
            rec = {'alpha': float(alpha), 'l1_wt': float(l1_wt), 'score': float(score),
                   'fit_attempted': False, 'fit_ok': None, 'error': None}

            if score > best['score']:
                rec['fit_attempted'] = True
                try:
                    res_full = _fit_once(y, X, off, cov_type, alpha, l1_wt, regularization,
                                         use_overdispersion_scale=use_overdispersion_scale, feature_names=feature_names)
                    rec['fit_ok'] = True
                    best.update(score=score, alpha=float(alpha),
                                l1_wt=float(l1_wt), res=res_full)
                except Exception as e:
                    rec['fit_ok'] = False
                    rec['error'] = str(e)
                    print(
                        f'Could not fit full model for this combo: alpha={alpha}, l1_wt={l1_wt}: {e}')

            records.append(rec)

    if best['res'] is None:  # robust fallback to MLE
        best['alpha'], best['l1_wt'] = 0.0, 0.0
        best['res'] = _fit_once(y, X, off, cov_type, 0.0, 0.0, 'none',
                                use_overdispersion_scale=use_overdispersion_scale, feature_names=feature_names)

    table = pd.DataFrame.from_records(records).sort_values(
        'score', ascending=False, kind='mergesort')
    table['rank'] = np.arange(1, len(table) + 1)
    table['selected'] = (np.isclose(table['alpha'], best['alpha'])) & (
        np.isclose(table['l1_wt'], best['l1_wt']))
    print(
        f'    best hyperparams: alpha={best["alpha"]}, l1_wt={best["l1_wt"]}, score={best["score"]:.3f}', flush=True)

    return (best, table) if return_table else best


def _fit_once(y, X, off, cov_type, alpha, l1_wt, regularization, *, use_overdispersion_scale=False, feature_names=None):
    """
    Fit a single Poisson GLM.
      - Unpenalized: GLM.fit(method='newton', cov_type, scale=('X2' if QP-like))
      - Penalized:   GLM.fit_regularized(alpha, L1_wt)  (no SEs; refit later if needed)
    """
    y = np.asarray(y, dtype=float)
    off = None if off is None else np.asarray(off, dtype=float)

    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=off)
    if feature_names is not None:
        attach_feature_names(model, feature_names)

    if (regularization == 'none') or (alpha is None) or (alpha <= 0.0):
        res = fit_with_fallback(model, cov_type=cov_type,
                                use_overdispersion_scale=use_overdispersion_scale,
                                maxiter=400, try_unpenalized_refit=True)
        for k, v in dict(alpha=0.0, l1_wt=0.0, regularization='none',
                         is_penalized=bool(
                             getattr(res, 'is_penalized', False)),
                         cov_type=cov_type,
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


def summarize_large_coeffs(coefs_df, df_X, top_n=20, beta_abs_thresh=8.0, rr_thresh=50.0):
    """
    Flag large coefficients and report more interpretable 'rate ratio per 1 SD' (RR_1sd).
    """
    # 1SD per column (binary: SD≈sqrt(p(1-p)); still fine for RR scaling)
    sd = df_X.std(axis=0, ddof=0).replace(0, np.nan)   # avoid zero division
    sd_map = sd.to_dict()

    def rr_1sd(row):
        s = sd_map.get(row['term'], np.nan)
        if not np.isfinite(s) or s == 0:
            return np.nan
        return float(np.exp(row['coef'] * s))

    df = coefs_df.copy()
    df['RR_1sd'] = df.apply(rr_1sd, axis=1)
    df['is_large_beta'] = df['coef'].abs() > beta_abs_thresh
    df['is_large_rr'] = (df['RR_1sd'] > rr_thresh) | (
        df['RR_1sd'] < (1/rr_thresh))

    cols = ['cluster', 'term', 'coef', 'se', 'p', 'RR_1sd']
    out = (df.loc[df['is_large_beta'] | df['is_large_rr'], cols]
             .assign(abs_beta=lambda d: d['coef'].abs())
             .sort_values(['abs_beta', 'RR_1sd'], ascending=[False, False])
             .head(top_n))
    return out


def has_all_finite_params(res):

    try:
        b = np.asarray(getattr(res, 'params', np.nan), float)
        return np.isfinite(b).all()
    except Exception:
        return False


def get_const_ix(model):
    names = list(getattr(model, 'exog_names', []))
    for nm in ('const', 'Intercept'):
        if nm in names:
            return names.index(nm)
    return None


def mark_result(res, *, used_ridge, msg, is_penalized, tried=None):
    setattr(res, 'used_ridge_fallback', bool(used_ridge))
    setattr(res, 'convergence_message', str(msg))
    setattr(res, 'is_penalized', bool(is_penalized))
    if tried is not None:
        setattr(res, 'ridge_alphas_tried', tuple(tried))
    return res


def ensure_fields_present(res, p):

    if not hasattr(res, 'bse'):
        res.bse = np.full(p, np.nan, float)
    if not hasattr(res, 'pvalues'):
        res.pvalues = np.full(p, np.nan, float)
    return res


def try_unpenalized(model, *, cov_type, use_overdispersion_scale, maxiter, start_params=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        try:
            res = model.fit(method='newton', maxiter=maxiter, tol=1e-8, disp=False,
                            cov_type=cov_type, start_params=start_params,
                            scale=('X2' if use_overdispersion_scale else None))
            warned = any(isinstance(wi.message, ConvergenceWarning)
                         for wi in w)
            return res if bool(getattr(res, 'converged', True)) and not warned else None
        except Exception:
            return None


def ridge_once(model, a, const_ix):

    p = int(model.exog.shape[1])
    alpha_vec = np.full(p, float(a), dtype=float)
    if const_ix is not None:
        alpha_vec[const_ix] = 0.0
    return model.fit_regularized(alpha=alpha_vec, L1_wt=0.0, maxiter=2000, cnvrg_tol=1e-8)


def inject_baseline_intercept_if_needed(res):
    """
    If params are all zeros (or intercept is non-finite) and an intercept exists,
    set it to log(mean rate) adjusted for offset so predictions are sane.
    """

    model = getattr(res, 'model', None)
    if model is None:
        return res
    const_ix = get_const_ix(model)
    if const_ix is None:
        # no intercept column to place a baseline into
        return res

    b = np.asarray(getattr(res, 'params', np.zeros(
        model.exog.shape[1], float)), float)
    needs = np.allclose(b, 0.0) or (not np.isfinite(b[const_ix]))
    if not needs:
        return res

    y = np.asarray(model.endog, float).reshape(-1)
    off = getattr(model, 'offset', None)
    offv = None if off is None else np.asarray(off, float).reshape(-1)

    if offv is not None and np.isfinite(offv).all():
        denom = float(np.sum(np.exp(offv))) or len(y)
    else:
        denom = float(len(y))
    lam0 = float(np.sum(y)) / max(denom, 1e-12)

    b[const_ix] = np.log(max(lam0, 1e-12))
    try:
        res.params = b
    except Exception:
        setattr(res, 'params', b)

    msg = getattr(res, 'convergence_message', '')
    mark_result(res,
                used_ridge=getattr(res, 'used_ridge_fallback', False),
                msg=(msg + '+baseline_const_injected'),
                is_penalized=getattr(res, 'is_penalized', False),
                tried=getattr(res, 'ridge_alphas_tried', None),
                )
    return res


def coerce_params_finite_inplace(res):
    """NaN/±inf → 0.0 (keeps vector finite)."""

    try:
        b = np.asarray(res.params, float)
    except Exception:
        return res
    mask = ~np.isfinite(b)
    if mask.any():
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            res.params = b
        except Exception:
            setattr(res, 'params', b)
        msg = getattr(res, 'convergence_message', '')
        mark_result(res,
                    used_ridge=getattr(res, 'used_ridge_fallback', False),
                    msg=(msg + '+coerced_params'),
                    is_penalized=getattr(res, 'is_penalized', False),
                    tried=getattr(res, 'ridge_alphas_tried', None),
                    )
    return res


def make_ridge_partial_result(last_ridge, last_alpha, tried):

    class _RidgeCoercedRes:
        pass
    rc = _RidgeCoercedRes()
    rc.model = getattr(last_ridge, 'model', None)
    p = int(rc.model.exog.shape[1])
    b = np.asarray(getattr(last_ridge, 'params', np.zeros(p, float)), float)
    rc.params = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    rc.llf = getattr(last_ridge, 'llf', np.nan)
    rc.llnull = getattr(last_ridge, 'llnull', np.nan)
    rc.deviance = getattr(last_ridge, 'deviance', np.nan)
    rc.null_deviance = getattr(last_ridge, 'null_deviance', np.nan)
    rc.bse = np.full(p, np.nan, float)
    rc.pvalues = np.full(p, np.nan, float)
    rc.converged = False
    mark_result(rc, used_ridge=True,
                msg=f'ridge_partial_nan_to_zero_alpha={last_alpha:g}',
                is_penalized=True, tried=tried)
    return rc


def make_baseline_result(model, const_ix, last_alpha, tried):

    class _BaselineRes:
        pass
    endog = np.asarray(model.endog, float).reshape(-1)
    off = getattr(model, 'offset', None)
    offv = None if off is None else np.asarray(off, float).reshape(-1)
    n, p = model.exog.shape
    if offv is not None and np.isfinite(offv).all():
        denom = float(np.sum(np.exp(offv))) or n
    else:
        denom = float(n)
    lam0 = float(np.sum(endog)) / max(denom, 1e-12)

    bres = _BaselineRes()
    bres.model = model
    bres.params = np.zeros(p, float)
    if const_ix is not None:
        bres.params[const_ix] = np.log(max(lam0, 1e-12))
    bres.bse = np.full(p, np.nan, float)
    bres.pvalues = np.full(p, np.nan, float)
    bres.llf = np.nan
    bres.llnull = np.nan
    bres.deviance = np.nan
    bres.null_deviance = np.nan
    bres.converged = False
    mark_result(bres, used_ridge=True,
                msg=f'baseline_only_no_finite_fit(last_ridge_alpha={last_alpha if last_alpha is not None else "n/a"})',
                is_penalized=False, tried=tried)
    return bres


def fit_with_fallback(
    model, *, cov_type, use_overdispersion_scale=False,
    maxiter=400, try_unpenalized_refit=True,
    ridge_alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
):
    """
    Always returns a result with FINITE params.
    Additionally, if the returned vector is all zeros and an intercept exists,
    we inject a baseline intercept so predictions are meaningful.
    """

    # 1) Unpenalized
    res = try_unpenalized(model, cov_type=cov_type,
                          use_overdispersion_scale=use_overdispersion_scale,
                          maxiter=maxiter, start_params=None)
    if res is not None:
        mark_result(res, used_ridge=False, msg='newton_ok', is_penalized=False)
        ensure_fields_present(res, p=int(model.exog.shape[1]))
        coerce_params_finite_inplace(res)
        res = inject_baseline_intercept_if_needed(res)
        return res

    # 2) Ridge ladder
    const_ix = get_const_ix(model)
    last_ridge, last_alpha = None, None
    for a in ridge_alphas:
        try:
            rr = ridge_once(model, a, const_ix)
            last_ridge, last_alpha = rr, a
            ensure_fields_present(rr, p=int(model.exog.shape[1]))
            if has_all_finite_params(rr):
                mark_result(rr, used_ridge=True, msg=f'ridge_only_alpha={a:g}',
                            is_penalized=True, tried=ridge_alphas)
                if try_unpenalized_refit:
                    res_unpen = try_unpenalized(
                        model, cov_type=cov_type,
                        use_overdispersion_scale=use_overdispersion_scale,
                        maxiter=maxiter, start_params=np.asarray(
                            rr.params, float)
                    )
                    if (res_unpen is not None) and has_all_finite_params(res_unpen):
                        mark_result(res_unpen, used_ridge=True,
                                    msg=f'ridge_then_unpen_ok_alpha={a:g}',
                                    is_penalized=False, tried=ridge_alphas)
                        ensure_fields_present(
                            res_unpen, p=int(model.exog.shape[1]))
                        coerce_params_finite_inplace(res_unpen)
                        res_unpen = inject_baseline_intercept_if_needed(
                            res_unpen)
                        return res_unpen
                coerce_params_finite_inplace(rr)
                rr = inject_baseline_intercept_if_needed(rr)
                return rr
        except Exception:
            continue

    # 4) Ridge-partial salvage
    if last_ridge is not None:
        rc = make_ridge_partial_result(last_ridge, last_alpha, ridge_alphas)
        rc = inject_baseline_intercept_if_needed(
            coerce_params_finite_inplace(rc))
        return rc

    # 5) Baseline fallback
    bres = make_baseline_result(model, const_ix, last_alpha, ridge_alphas)
    bres = inject_baseline_intercept_if_needed(
        coerce_params_finite_inplace(bres))
    return bres
