# -*- coding: utf-8 -*-
# Unified decoding toolkit: GLM LLR (with stable math + proxy freezing) and LR baseline with PCA.
# Functions are kept compact; comments point to key steps.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# Small utilities
# =========================

def _as_np(a, name, two_d=False):
    arr = a.to_numpy() if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)
    if two_d and arr.ndim == 1:
        arr = arr[:, None]
    return arr

def exp_diff_stable(a: np.ndarray, b: np.ndarray, clip_hi: float = 80.0) -> np.ndarray:
    """Stable exp(a)-exp(b) with clipping of the larger exponent to avoid overflow."""
    use_a = a >= b
    hi = np.where(use_a, a, b)
    lo = np.where(use_a, b, a)
    hi_c = np.clip(hi, None, clip_hi)
    scale = np.exp(hi_c)
    delta = -np.expm1(lo - hi)   # 1 - exp(lo-hi), stable near 0
    return scale * np.where(use_a, delta, -delta)

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Stable logistic transform."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def guard_mask_from_episodes(bins_2d, y, guard: float = 0.05):
    """Mask out bin centers within ±guard of any 0/1 boundary in y."""
    bins_2d = np.asarray(bins_2d, float)
    y = np.asarray(y, int).reshape(-1)
    t = (bins_2d[:, 0] + bins_2d[:, 1]) * 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, y, 0]) != 0)
    if edges.size == 0:
        return np.ones_like(y, bool)
    starts, ends = edges[::2], edges[1::2]
    bounds = np.empty(2 * starts.size, float)
    bounds[0::2] = bins_2d[starts, 0]
    bounds[1::2] = bins_2d[ends - 1, 1]
    mind = np.min(np.abs(t[:, None] - bounds[None, :]), axis=1)
    return mind >= float(guard)

# =========================
# Params table helpers
# =========================

def params_df_from_coefs_df(coefs_df: pd.DataFrame,
                            unit_col: str = 'cluster',
                            term_col: str = 'term',
                            coef_col: str = 'coef') -> pd.DataFrame:
    """Long → wide: rows=units, cols=terms."""
    return coefs_df.pivot(index=unit_col, columns=term_col, values=coef_col).sort_index()

def align_params_to_Y(params_df: pd.DataFrame, df_Y: pd.DataFrame, *, fill_missing: float = 0.0) -> pd.DataFrame:
    """
    Align params rows to df_Y columns. Missing units are filled with zeros
    so they contribute no evidence.
    """
    p = params_df.copy()
    y_idx = df_Y.columns
    p.index = p.index.astype(str)
    y_as_str = y_idx.astype(str)

    aligned = p.reindex(y_as_str)
    missing_mask = aligned.isna().all(axis=1)
    if missing_mask.any():
        missing = aligned.index[missing_mask].tolist()
        print(f'[align_params_to_Y] WARNING: {len(missing)} Y units missing; filling with {fill_missing}. Examples: {missing[:10]}'
              f'{"..." if len(missing) > 10 else ""}')
        aligned = aligned.fillna(fill_missing)

    aligned.index = y_idx
    return aligned

# =========================
# Visibility-proxy freezing
# =========================

def _find_vis_proxies(X: pd.DataFrame, vis_col: str,
                      proxy_prefixes: tuple[str, ...] = ('vis_', 'visible_', 'viswin_', 'vishist_', 'visx_', 'vis×')) -> list[str]:
    """Columns derived from visibility (except the protected vis_col). Adjust prefixes as needed."""
    return [c for c in X.columns if c != vis_col and any(c.startswith(p) for p in proxy_prefixes)]

def _assert_only_vis_changes(X0: pd.DataFrame, X1: pd.DataFrame, vis_col: str):
    diff_cols = [c for c in X0.columns if not np.allclose(X0[c].to_numpy(), X1[c].to_numpy())]
    if diff_cols != [vis_col]:
        raise AssertionError(f'Counterfactual differs in {diff_cols}, expected only {vis_col}.')

# =========================
# Core GLM LLR decoding (fitted params provided)
# =========================

def compute_decode_row_mask(df_X, df_Y, offset_log):
    """Keep rows with finite X, Y, and offset."""
    X = df_X.to_numpy() if isinstance(df_X, pd.DataFrame) else np.asarray(df_X)
    Y = df_Y.to_numpy() if isinstance(df_Y, pd.DataFrame) else np.asarray(df_Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'Row mismatch: df_X={X.shape}, df_Y={Y.shape}')
    X_ok = np.isfinite(X).all(axis=1) if X.size else np.ones(X.shape[0], bool)
    Y_ok = np.isfinite(Y).all(axis=1) if Y.size else np.ones(Y.shape[0], bool)
    if np.isscalar(offset_log):
        O_ok = np.ones(X.shape[0], bool)
    else:
        O = np.asarray(offset_log).reshape(-1)
        if O.shape[0] != X.shape[0]:
            raise ValueError('offset_log length must match rows of df_X/df_Y')
        O_ok = np.isfinite(O)
    return X_ok & Y_ok & O_ok

def decode_from_fitted_glm(
    df_X_te: pd.DataFrame,
    df_Y_te: pd.DataFrame,
    offset_log_te,                    # scalar or (T_te,)
    params_df: pd.DataFrame,          # (N x P)
    *,
    vis_col: str = 'any_ff_visible',
    intercept_names: tuple[str, ...] = ('const', 'Intercept', 'intercept'),
    proxy_prefixes: tuple[str, ...] = ('vis_', 'visible_', 'viswin_', 'vishist_', 'visx_', 'vis×'),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Toggle vis_col 0/1 and compute LLR & posterior on TEST.
    Freezes vis-derived proxies so only vis_col differs across hypotheses.
    """
    X = df_X_te.copy()
    Y = df_Y_te.copy()

    # Intercept column if model expects it
    if any(c in params_df.columns for c in intercept_names):
        ic = next(c for c in intercept_names if c in params_df.columns)
        if ic not in X.columns:
            X[ic] = 1.0

    # Align design to params columns; add missing (non-vis) as zeros
    keep = [c for c in X.columns if c in params_df.columns]
    X = X[keep]
    for c in params_df.columns:
        if c not in X.columns:
            if c == vis_col:
                raise ValueError(f"vis_col '{vis_col}' missing from test design.")
            X[c] = 0.0
    X = X[params_df.columns]

    # Offset vector
    off = np.full(len(X), float(offset_log_te), float) if np.isscalar(offset_log_te) else np.asarray(offset_log_te, float).reshape(-1)
    if off.shape[0] != len(X):
        raise ValueError('offset_log_te length must match test rows')

    # Build counterfactual designs; proxies remain observed (frozen)
    if vis_col not in X.columns:
        raise ValueError(f"'{vis_col}' not found in design/params.")
    X0 = X.copy(); X0[vis_col] = 0.0
    X1 = X.copy(); X1[vis_col] = 1.0
    # _assert_only_vis_changes(X0, X1, vis_col)  # enable while debugging

    # Matrix math
    B = params_df[X.columns].to_numpy(float)         # (N, P)
    X0m, X1m = X0.to_numpy(float), X1.to_numpy(float)
    eta0 = X0m @ B.T + off[:, None]                  # (T, N)
    eta1 = X1m @ B.T + off[:, None]

    lam_diff = exp_diff_stable(eta1, eta0)
    Yn = Y.to_numpy(float)
    llr = (Yn * (eta1 - eta0) - lam_diff).sum(axis=1, dtype=np.float64)
    p_post = sigmoid_stable(llr)
    return llr, p_post

# =========================
# GLM fit per neuron (Poisson) and CV wrapper (alt path)
# =========================

def fit_glm_poisson_per_neuron(K, X, y, offset, train_idx, alpha: float = 0.0):
    """
    Fit Poisson GLM per neuron with design [1, vis, feats], offset=log(dt).
    Returns a list of fitted results (or dummy with zero params on failure).
    """
    K = _as_np(K, 'binned_spikes', two_d=True)     # (T, N)
    X = _as_np(X, 'binned_feats',  two_d=True)     # (T, F)
    y = _as_np(y, 'y_visible').ravel()             # (T,)
    offset = _as_np(offset, 'offset').ravel()      # (T,)

    T, N = K.shape
    if not (X.shape[0] == y.shape[0] == offset.shape[0] == T):
        raise ValueError('Row mismatch among K, X, y, offset')

    Xdesign = np.column_stack([np.ones((T, 1)), y.reshape(-1, 1), X])  # [1, vis, feats]
    models = []
    for n in range(N):
        endog = K[train_idx, n]
        exog  = Xdesign[train_idx]
        off   = offset[train_idx]
        mod = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=off)
        try:
            res = (mod.fit_regularized(alpha=alpha, L1_wt=0.0) if alpha > 0 else mod.fit())
            if not np.all(np.isfinite(res.params)):
                res = mod.fit()
        except Exception:
            p = exog.shape[1]
            class _Dummy:  # zero contribution if fit fails
                params = np.zeros(p)
            res = _Dummy()
        models.append(res)
    return models

def _stack_params(models):
    P = len(models[0].params)
    M = np.empty((len(models), P), float)
    for i, m in enumerate(models):
        M[i, :] = m.params
    return M  # (N, P)

def llr_etas_from_models(models, X, offset):
    """Return eta1, eta0 (T, N) for design [1, vis, feats] under vis=1 vs 0."""
    X = _as_np(X, 'binned_feats', two_d=True)
    offset = _as_np(offset, 'offset').ravel()
    T = X.shape[0]
    X1 = np.column_stack([np.ones((T, 1)), np.ones((T, 1)), X])   # vis=1
    X0 = np.column_stack([np.ones((T, 1)), np.zeros((T, 1)), X])  # vis=0
    P = _stack_params(models)
    eta1 = X1 @ P.T + offset[:, None]
    eta0 = X0 @ P.T + offset[:, None]
    return eta1, eta0

def cv_decode_glm(
    K, X, y, groups, dt, alpha: float = 0.0, n_splits: int = 5, scale_X: bool = True, verbose: bool = False
):
    """
    Per-neuron Poisson-GLM LLR decoder with GroupKFold CV.
    Returns mean AUC, std AUC across folds.
    """
    K = _as_np(K, 'binned_spikes', two_d=True)
    X = _as_np(X, 'binned_feats',  two_d=True)
    y = _as_np(y, 'y_visible').ravel()
    g = _as_np(groups, 'groups').ravel()
    dt = _as_np(dt, 'dt').ravel() if np.ndim(dt) else np.array([float(dt)])

    T, N = K.shape
    if X.shape[0] != T or y.shape[0] != T or g.shape[0] != T:
        raise ValueError('Row mismatch among K, X, y, groups')
    if dt.size == 1:
        dt = np.full(T, float(dt[0]))
    if dt.shape[0] != T or (dt <= 0).any():
        raise ValueError('dt must be scalar or length T and > 0')

    fin_mask = np.isfinite(K).all(axis=1) & np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(g) & np.isfinite(dt)
    if not fin_mask.all():
        if verbose:
            print(f'Dropping {np.sum(~fin_mask)} rows with NaN/Inf before CV.')
        K, X, y, g, dt = K[fin_mask], X[fin_mask], y[fin_mask], g[fin_mask], dt[fin_mask]

    offset = np.log(dt)
    uniq = np.unique(g)
    if uniq.size < n_splits:
        n_splits = max(2, uniq.size)

    cv = GroupKFold(n_splits=n_splits)
    aucs = []

    for tr, te in cv.split(X, y, g):
        Xtr, Xte = X[tr], X[te]
        if scale_X and X.shape[1] > 0:
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd

        models = fit_glm_poisson_per_neuron(K, Xtr, y, offset, tr, alpha=alpha)
        eta1, eta0 = llr_etas_from_models(models, Xte, offset[te])

        Kte = K[te]
        llr = (Kte * (eta1 - eta0)).sum(axis=1) - exp_diff_stable(eta1, eta0).sum(axis=1)

        good = np.isfinite(llr)
        if not np.all(good):
            if verbose:
                print(f'Warning: {np.sum(~good)} NaN/Inf LLR on test; dropping for AUC.')
            llr, yte = llr[good], y[te][good]
        else:
            yte = y[te]

        aucs.append(roc_auc_score(yte, llr))

    return float(np.mean(aucs)), float(np.std(aucs))

# =========================
# CV orchestrator using your glm_mini_report (params_df path)
# =========================

def standardize_like_train(df_X, train_idx, exclude_cols=()):
    """Z-score columns on TRAIN only; leave excluded columns untouched."""
    X = df_X.copy()
    cols = [c for c in X.columns if c not in exclude_cols]
    if len(cols) > 0:
        mu = X.loc[train_idx, cols].mean(axis=0)
        sd = X.loc[train_idx, cols].std(axis=0).replace(0.0, 1.0)
        X.loc[:, cols] = (X.loc[:, cols] - mu) / sd
    return X, (mu if len(cols) > 0 else pd.Series(dtype=float), sd if len(cols) > 0 else pd.Series(dtype=float))

def cv_decode_with_glm_report(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,            # spikes (T x N)
    y: np.ndarray,                 # (T,) 0/1 labels (only for metrics/guards)
    groups: np.ndarray,            # (T,) group ids for CV
    offset_log,                    # scalar or (T,)
    *,
    fit_fn,                        # e.g., stop_glm_fit.glm_mini_report
    fit_kwargs: dict | None = None,
    bins_2d: np.ndarray | None = None,
    vis_col: str = 'any_ff_visible',
    n_splits: int = 5,
    standardize: bool = False,
    exclude_from_standardize: tuple = ('any_ff_visible', 'const', 'Intercept', 'intercept'),
    guard: float | None = None
):
    """GroupKFold CV around your report-based GLM; returns out-of-fold metrics & predictions."""
    if fit_kwargs is None:
        fit_kwargs = dict(cov_type='HC1', fast_mle=True, do_inference=False, make_plots=False, show_plots=False)

    row_mask = compute_decode_row_mask(df_X, df_Y, offset_log)
    X_all = df_X.loc[row_mask].reset_index(drop=True)
    Y_all = df_Y.loc[row_mask].reset_index(drop=True)
    y_all = np.asarray(y).reshape(-1)[row_mask]
    g_all = np.asarray(groups).reshape(-1)[row_mask]
    off_all = float(offset_log) if np.isscalar(offset_log) else np.asarray(offset_log, float).reshape(-1)[row_mask]
    bins_all = None if bins_2d is None else np.asarray(bins_2d, float)[row_mask]

    uniq = np.unique(g_all)
    if uniq.size < n_splits:
        n_splits = max(2, uniq.size)
    cv = GroupKFold(n_splits=n_splits)

    T = len(X_all)
    oof_llr = np.full(T, np.nan)
    oof_prob = np.full(T, np.nan)
    fold_metrics = []

    for fold, (tr, te) in enumerate(cv.split(X_all, y_all, g_all), start=1):
        Xtr, Xte = X_all.iloc[tr], X_all.iloc[te]
        Ytr, Yte = Y_all.iloc[tr], Y_all.iloc[te]
        yte = y_all[te]
        off_tr = off_all if np.isscalar(off_all) else off_all[tr]
        off_te = off_all if np.isscalar(off_all) else off_all[te]
        bins_te = None if bins_all is None else bins_all[te]

        if standardize:
            Xtr, _ = standardize_like_train(X_all, tr, exclude_cols=exclude_from_standardize)
            Xte = Xtr.iloc[te]  # reuse scaled view

        report = fit_fn(df_X=Xtr, df_Y=Ytr, offset_log=off_tr, **fit_kwargs)
        params_df = params_df_from_coefs_df(report['coefs_df'], unit_col='cluster', term_col='term', coef_col='coef')
        params_df = align_params_to_Y(params_df, Ytr)  # fills missing units with zeros

        llr_te, p_te = decode_from_fitted_glm(Xte, Yte, off_te, params_df, vis_col=vis_col)

        mask_guard = np.ones_like(yte, bool)
        if guard is not None and bins_te is not None:
            mask_guard = guard_mask_from_episodes(bins_te, yte, guard=guard)

        if mask_guard.sum() == 0 or (yte[mask_guard].min() == yte[mask_guard].max()):
            auc = np.nan; ap = np.nan
        else:
            auc = roc_auc_score(yte[mask_guard], p_te[mask_guard])
            ap = average_precision_score(yte[mask_guard], p_te[mask_guard])

        fold_metrics.append(dict(fold=fold, auc=auc, pr_auc=ap, n_test=len(te), n_kept=int(mask_guard.sum())))
        oof_llr[te] = llr_te
        oof_prob[te] = p_te

    aucs = np.array([m['auc'] for m in fold_metrics], float)
    aps  = np.array([m['pr_auc'] for m in fold_metrics], float)
    auc_mean = float(np.nanmean(aucs)) if aucs.size else np.nan
    auc_std  = float(np.nanstd(aucs, ddof=1)) if np.isfinite(aucs).sum() > 1 else np.nan
    ap_mean  = float(np.nanmean(aps)) if aps.size else np.nan
    ap_std   = float(np.nanstd(aps, ddof=1)) if np.isfinite(aps).sum() > 1 else np.nan

    return dict(
        row_mask=row_mask,
        fold_metrics=fold_metrics,
        auc_mean=auc_mean, auc_std=auc_std,
        pr_mean=ap_mean,  pr_std=ap_std,
        oof_llr=oof_llr,  oof_prob=oof_prob,
        n_splits=n_splits
    )

# =========================
# Baseline: LR pipeline on rates (with optional PCA search)
# =========================

def decode_visible_with_lr(
    K_counts,           # (T, N) spikes (counts)
    y_visible,          # (T,) 0/1
    groups,             # (T,) episode ids
    dt=None,            # scalar or (T,)
    try_pca: bool = False
):
    """Simple LR decoder on sqrt-rate features with GroupKFold + GridSearch."""
    K = _as_np(K_counts, 'K', two_d=True)
    y = _as_np(y_visible, 'y').ravel()
    g = _as_np(groups, 'groups').ravel()

    T, _ = K.shape
    if dt is None:
        dt_arr = np.ones(T, float)
    elif isinstance(dt, float) or np.ndim(dt) == 0:
        dt_arr = np.full(T, float(dt))
    else:
        dt_arr = _as_np(dt, 'dt').ravel()
        if dt_arr.shape[0] != T:
            raise ValueError('dt must be scalar or length T')

    K_feat = K / dt_arr[:, None]  # convert to rates

    pipe = Pipeline([
        ('sqrt', FunctionTransformer(lambda X: np.sqrt(X + 1e-8))),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('pca', PCA(svd_solver='auto', whiten=False)),
        ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced'))
    ])

    Cs = np.logspace(-2, 2, 7)
    param_grid = [{'pca__n_components': [None], 'clf__C': Cs}]
    if try_pca:
        param_grid.append({'pca__n_components': [0.8, 0.9, 0.95, 0.98], 'clf__C': Cs})

    cv = GroupKFold(n_splits=min(5, len(np.unique(g))))
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, refit=True, verbose=0)
    gs.fit(K_feat, y, groups=g)

    best_model = gs.best_estimator_
    best_auc = float(gs.best_score_)
    proba = best_model.predict_proba(K_feat)[:, 1]
    auc_all = float(roc_auc_score(y, proba))
    return best_model, best_auc, auc_all, gs.best_params_

# =========================
# Significance & uncertainty utilities (AUC)
# =========================

def auc_permutation_test(y, scores, groups=None, n_perm: int = 1000, rng: int = 0, mask=None,
                         progress: bool = False, progress_every: int = 200, desc: str = 'Permutations'):
    """
    Time-aware permutation test:
      - Circularly shift scores within each group (episode) to preserve autocorrelation.
    """
    y = np.asarray(y, int).reshape(-1)
    s = np.asarray(scores, float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError('y and scores must have same length')
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    if m.shape[0] != y.shape[0]:
        raise ValueError('mask length != data length')
    y, s = y[m], s[m]

    if groups is None:
        g = np.zeros_like(y)
    else:
        g_full = np.asarray(groups).reshape(-1)
        if g_full.shape[0] != m.shape[0]:
            raise ValueError('groups length != data length')
        g = g_full[m]

    if y.min() == y.max():
        raise ValueError('Only one class after masking; AUC undefined.')

    auc_obs = roc_auc_score(y, s)
    rng = np.random.default_rng(rng)
    uniq = np.unique(g)
    group_idx = [np.where(g == gg)[0] for gg in uniq]
    null = np.empty(int(n_perm), float)

    it = range(int(n_perm))
    use_tqdm = False
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=desc, leave=False)
            use_tqdm = True
        except Exception:
            use_tqdm = False

    for k in it:
        s_perm = np.empty_like(s)
        for idx in group_idx:
            if idx.size <= 1:
                s_perm[idx] = s[idx]
            else:
                shift = rng.integers(0, idx.size)
                s_perm[idx] = np.roll(s[idx], shift)
        null[k] = roc_auc_score(y, s_perm)
        if progress and not use_tqdm and ((k + 1) % max(1, progress_every) == 0 or (k + 1) == n_perm):
            print(f'{desc}: {k+1}/{n_perm}', end='\r', flush=True)
    if progress and not use_tqdm:
        print()

    p = (1 + np.sum(null >= auc_obs)) / (1 + n_perm)
    return float(auc_obs), float(p), null

def auc_block_bootstrap_ci(y, scores, groups=None, n_boot: int = 2000, conf: float = 0.95, rng: int = 0, mask=None,
                           progress: bool = False, progress_every: int = 200, desc: str = 'Bootstraps'):
    """Block/bootstrap AUC by resampling groups with replacement."""
    y = np.asarray(y, int).reshape(-1)
    s = np.asarray(scores, float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError('y and scores must have same length')
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    if m.shape[0] != y.shape[0]:
        raise ValueError('mask length != data length')
    y, s = y[m], s[m]

    if groups is None:
        g = np.arange(y.shape[0])
    else:
        g_full = np.asarray(groups).reshape(-1)
        if g_full.shape[0] != m.shape[0]:
            raise ValueError('groups length != data length')
        g = g_full[m]

    uniq = np.unique(g)
    group_idx = [np.where(g == gg)[0] for gg in uniq]

    rng = np.random.default_rng(rng)
    aucs = np.empty(int(n_boot), float)

    it = range(int(n_boot))
    use_tqdm = False
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=desc, leave=False)
            use_tqdm = True
        except Exception:
            use_tqdm = False

    for b in it:
        sel = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([group_idx[i] for i in sel]) if sel.size else np.array([], int)
        yb, sb = y[idx], s[idx]
        if yb.size == 0 or yb.min() == yb.max():
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(yb, sb)
        if progress and not use_tqdm and ((b + 1) % max(1, progress_every) == 0 or (b + 1) == n_boot):
            print(f'{desc}: {b+1}/{n_boot}', end='\r', flush=True)
    if progress and not use_tqdm:
        print()

    aucs = aucs[np.isfinite(aucs)]
    if aucs.size == 0:
        raise ValueError('All bootstrap samples degenerate (single class).')
    mean_auc = float(np.mean(aucs))
    lo = float(np.quantile(aucs, (1 - conf) / 2))
    hi = float(np.quantile(aucs, 1 - (1 - conf) / 2))
    return mean_auc, lo, hi, aucs
