import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- helpers to work with your report ----------
def params_df_from_coefs_df(coefs_df: pd.DataFrame,
                            unit_col: str = 'cluster',
                            term_col: str = 'term',
                            coef_col: str = 'coef') -> pd.DataFrame:
    """Pivot long-form coefs_df (cluster, term, coef) -> wide (rows=units, cols=terms)."""
    wide = coefs_df.pivot(index=unit_col, columns=term_col, values=coef_col).sort_index()
    return wide

def align_params_to_Y(params_df: pd.DataFrame, df_Y: pd.DataFrame, *, fill_missing: float = 0.0) -> pd.DataFrame:
    """
    Align params rows to df_Y columns (units). If some Y units are missing from params,
    fill their entire rows with `fill_missing` (default 0.0) rather than raising.
    """
    p = params_df.copy()
    y_idx = df_Y.columns
    p.index = p.index.astype(str)
    y_as_str = y_idx.astype(str)

    aligned = p.reindex(y_as_str)
    missing_mask = aligned.isna().all(axis=1)
    if missing_mask.any():
        missing = aligned.index[missing_mask].tolist()
        print(f'[align_params_to_Y] WARNING: {len(missing)} Y units missing from params (filling with {fill_missing}): '
              f'{missing[:10]}{"..." if len(missing) > 10 else ""}')
        aligned = aligned.fillna(fill_missing)

    aligned.index = y_idx
    return aligned

# ---------- numerically stable primitives ----------
def exp_diff_stable(a: np.ndarray, b: np.ndarray, clip_hi: float = 80.0) -> np.ndarray:
    """
    Compute exp(a) - exp(b) stably, elementwise.
    Clips the larger exponent to avoid overflow; this slightly underestimates magnitude
    when the true value would be astronomically large (which is fine for LLR decisions).
    """
    use_a = a >= b
    hi = np.where(use_a, a, b)
    lo = np.where(use_a, b, a)
    hi_c = np.clip(hi, None, clip_hi)
    scale = np.exp(hi_c)
    delta = -np.expm1(lo - hi)  # equals 1 - exp(lo - hi), stable near 0
    return scale * np.where(use_a, delta, -delta)

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Stable logistic: 0.5 * (1 + tanh(x/2))."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))

# ---------- visibility-proxy freezing ----------
def _find_vis_proxies(X: pd.DataFrame, vis_col: str,
                      proxy_prefixes: tuple[str, ...] = ('vis_', 'visible_', 'viswin_', 'vishist_', 'visx_', 'vis×')) -> list[str]:
    """
    Identify columns derived from visibility (except the protected vis_col).
    Adjust prefixes to match your naming.
    """
    return [c for c in X.columns if c != vis_col and any(c.startswith(p) for p in proxy_prefixes)]

def _assert_only_vis_changes(X0: pd.DataFrame, X1: pd.DataFrame, vis_col: str):
    diff_cols = [c for c in X0.columns if not np.allclose(X0[c].to_numpy(), X1[c].to_numpy())]
    if diff_cols != [vis_col]:
        raise AssertionError(f'Counterfactual differs in {diff_cols}, expected only {vis_col}.')

# ---------- guard-band mask ----------
def guard_mask_from_episodes(bins_2d, y, guard: float = 0.05):
    """Mask out bin centers within ±guard of any on/off boundary in y (1D 0/1)."""
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

# ---------- (optional) standardization without leakage ----------
def standardize_like_train(df_X, train_idx, exclude_cols=()):
    """
    Z-score columns on TRAIN only, apply to all rows.
    exclude_cols: columns to skip (e.g., 'any_ff_visible', intercept).
    Returns: X_scaled (same columns/order), fitted (mu, sd).
    """
    X = df_X.copy()
    cols = [c for c in X.columns if c not in exclude_cols]
    if len(cols) > 0:
        mu = X.loc[train_idx, cols].mean(axis=0)
        sd = X.loc[train_idx, cols].std(axis=0).replace(0.0, 1.0)
        X.loc[:, cols] = (X.loc[:, cols] - mu) / sd
    else:
        mu = pd.Series(dtype=float)
        sd = pd.Series(dtype=float)
    return X, (mu, sd)

# ---------- row mask used by decoder ----------
def compute_decode_row_mask(df_X, df_Y, offset_log):
    """Rows kept by decoder: finite X, finite Y, finite offset (no CV leakage)."""
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

# ---------- core decoder given fitted params (no CV here) ----------
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
    Toggle vis_col 0/1 and compute LLR and posterior on the test split.
    Freezes all vis-derived proxy columns so only `vis_col` differs across X0 and X1.
    Uses numerically stable exp-difference and sigmoid.
    """
    X = df_X_te.copy()
    Y = df_Y_te.copy()

    # Intercept handling
    if any(c in params_df.columns for c in intercept_names):
        ic = next(c for c in intercept_names if c in params_df.columns)
        if ic not in X.columns:
            X[ic] = 1.0

    # Keep only terms present in params; add missing (non-vis) as zeros
    keep = [c for c in X.columns if c in params_df.columns]
    X = X[keep]
    for c in params_df.columns:
        if c not in X.columns:
            if c == vis_col:
                raise ValueError(f"vis_col '{vis_col}' missing from test design.")
            X[c] = 0.0
    X = X[params_df.columns]

    # Offset vector
    if np.isscalar(offset_log_te):
        off = np.full(len(X), float(offset_log_te), dtype=float)
    else:
        off = np.asarray(offset_log_te, float).reshape(-1)
        if off.shape[0] != len(X):
            raise ValueError('offset_log_te length must match test rows')

    # Build counterfactual designs with proxies frozen (copy keeps observed values)
    X0 = X.copy()
    X1 = X.copy()
    proxies = _find_vis_proxies(X, vis_col, proxy_prefixes=proxy_prefixes)
    # Proxies remain identical in X0 and X1; only toggle the protected indicator:
    if vis_col not in X.columns:
        raise ValueError(f"'{vis_col}' not found in design/params.")
    X0[vis_col] = 0.0
    X1[vis_col] = 1.0

    # Optional safety assertion while debugging:
    # _assert_only_vis_changes(X0, X1, vis_col)

    # Matrix math
    B = params_df[X.columns].to_numpy(float)          # (N, P)
    X0m = X0.to_numpy(float)
    X1m = X1.to_numpy(float)
    eta0 = X0m @ B.T + off[:, None]                   # (T, N)
    eta1 = X1m @ B.T + off[:, None]

    # Stable Poisson term: exp(eta1) - exp(eta0)
    lam_diff = exp_diff_stable(eta1, eta0)

    Yn = Y.to_numpy(float)
    llr = (Yn * (eta1 - eta0) - lam_diff).sum(axis=1, dtype=np.float64)

    p_post = sigmoid_stable(llr)
    return llr, p_post

# ---------- main CV orchestrator ----------
def cv_decode_with_glm_report(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,            # spikes (T x N), columns = units
    y: np.ndarray,                 # (T,) 0/1 labels
    groups: np.ndarray,            # (T,) group ids (session/episode) for CV
    offset_log,                    # scalar or (T,)
    *,
    fit_fn,                        # e.g., stop_glm_fit.glm_mini_report
    fit_kwargs: dict | None = None,
    bins_2d: np.ndarray | None = None,
    vis_col: str = 'any_ff_visible',
    n_splits: int = 5,
    standardize: bool = False,
    exclude_from_standardize: tuple = ('any_ff_visible', 'const', 'Intercept', 'intercept'),
    guard: float | None = None,
    random_state: int = 0
):
    """
    GroupKFold CV where fitting is done by your glm_mini_report on TRAIN only.
    Returns dict with per-fold metrics and concatenated out-of-fold predictions.
    """
    if fit_kwargs is None:
        fit_kwargs = dict(cov_type='HC1', fast_mle=True, do_inference=False, make_plots=False, show_plots=False)

    # 0) restrict to rows the decoder would keep (finite X/Y/offset)
    row_mask = compute_decode_row_mask(df_X, df_Y, offset_log)
    X_all = df_X.loc[row_mask].reset_index(drop=True)
    Y_all = df_Y.loc[row_mask].reset_index(drop=True)
    y_all = np.asarray(y).reshape(-1)[row_mask]
    g_all = np.asarray(groups).reshape(-1)[row_mask]
    if np.isscalar(offset_log):
        off_all = float(offset_log)
    else:
        off_all = np.asarray(offset_log, float).reshape(-1)[row_mask]
    bins_all = None if bins_2d is None else np.asarray(bins_2d, float)[row_mask]

    # 1) CV setup
    uniq = np.unique(g_all)
    if uniq.size < n_splits:
        n_splits = max(2, uniq.size)
    cv = GroupKFold(n_splits=n_splits)

    # storage
    T = len(X_all)
    oof_llr = np.full(T, np.nan)
    oof_prob = np.full(T, np.nan)
    fold_metrics = []

    for fold, (tr, te) in enumerate(cv.split(X_all, y_all, g_all), start=1):
        # 2) (optional) standardize using TRAIN only (no leakage)
        if standardize:
            X_scaled, _ = standardize_like_train(X_all, tr, exclude_cols=exclude_from_standardize)
        else:
            X_scaled = X_all

        # create train/test views
        Xtr = X_scaled.iloc[tr];  Xte = X_scaled.iloc[te]
        Ytr = Y_all.iloc[tr];     Yte = Y_all.iloc[te]
        ytr = y_all[tr];          yte = y_all[te]
        off_tr = off_all if np.isscalar(off_all) else off_all[tr]
        off_te = off_all if np.isscalar(off_all) else off_all[te]
        bins_te = None if bins_all is None else bins_all[te]

        # 3) fit GLM via your pipeline on TRAIN only
        report = fit_fn(df_X=Xtr, df_Y=Ytr, offset_log=off_tr, **fit_kwargs)

        # 4) get params -> align to Y columns
        params_df = params_df_from_coefs_df(report['coefs_df'],
                                            unit_col='cluster', term_col='term', coef_col='coef')
        params_df = align_params_to_Y(params_df, Ytr)  # preserves Y order; fills missing with zeros

        # 5) decode on TEST (with proxies frozen inside)
        llr_te, p_te = decode_from_fitted_glm(Xte, Yte, off_te, params_df, vis_col=vis_col)

        # 6) evaluate on TEST (optionally guard-band near edges)
        if guard is not None and bins_te is not None:
            mask_guard = guard_mask_from_episodes(bins_te, yte, guard=guard)
        else:
            mask_guard = np.ones_like(yte, bool)

        # need both classes after mask
        if mask_guard.sum() == 0 or (yte[mask_guard].min() == yte[mask_guard].max()):
            auc = np.nan
            ap = np.nan
        else:
            auc = roc_auc_score(yte[mask_guard], p_te[mask_guard])
            ap = average_precision_score(yte[mask_guard], p_te[mask_guard])

        fold_metrics.append(dict(fold=fold, auc=auc, pr_auc=ap, n_test=len(te), n_kept=int(mask_guard.sum())))

        # store out-of-fold predictions
        oof_llr[te] = llr_te
        oof_prob[te] = p_te

    # 7) aggregate metrics (ignore NaNs)
    aucs = np.array([m['auc'] for m in fold_metrics], float)
    aps = np.array([m['pr_auc'] for m in fold_metrics], float)
    auc_mean = float(np.nanmean(aucs)) if aucs.size else np.nan
    auc_std = float(np.nanstd(aucs, ddof=1)) if np.isfinite(aucs).sum() > 1 else np.nan
    ap_mean = float(np.nanmean(aps)) if aps.size else np.nan
    ap_std = float(np.nanstd(aps, ddof=1)) if np.isfinite(aps).sum() > 1 else np.nan

    return dict(
        row_mask=row_mask,                # which global rows were used at all
        fold_metrics=fold_metrics,        # per-fold AUC/PR-AUC
        auc_mean=auc_mean, auc_std=auc_std,
        pr_mean=ap_mean, pr_std=ap_std,
        oof_llr=oof_llr,                 # concatenated out-of-fold LLR (NaN where not evaluated due to row_mask=False)
        oof_prob=oof_prob,               # concatenated out-of-fold posterior
        n_splits=n_splits
    )
