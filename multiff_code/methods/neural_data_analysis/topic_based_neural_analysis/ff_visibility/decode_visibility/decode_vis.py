import pandas as pd
from sklearn.model_selection import GroupKFold
import statsmodels.api as sm
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score


def decode_visible_with_lr(
    K_counts,           # (T, N) binned_spikes (counts)
    y_visible,          # (T,) 0/1 labels
    groups,             # (T,) episode ids for GroupKFold
    dt=None,            # (T,) bin widths; if None, assume constant
    try_pca=False        # toggle PCA search
):
    T, N = K_counts.shape
    if dt is None:
        dt = np.ones(T, dtype=float)
    elif isinstance(dt, float):
        dt = np.ones(T, dtype=float) * dt
    # 1) rates, 2) sqrt
    K_feat = (K_counts / dt[:, None])

    pipe = Pipeline([
        ('sqrt', FunctionTransformer(lambda X: np.sqrt(X + 1e-8))),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        # optional PCA slot; we’ll tune n_components
        ('pca', PCA(svd_solver='auto', whiten=False)),
        ('clf', LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=5000,
            class_weight='balanced'
        ))
    ])

    param_grid = [
        # no PCA: set n_components=None (scikit-learn treats None as “keep all” but still transforms)
        # Better: emulate “no PCA” by using very high explained variance (≈1.0) and compare.
        {'pca__n_components': [None], 'clf__C': np.logspace(-2, 2, 7)},
    ]
    if try_pca:
        param_grid.append(
            {'pca__n_components': [0.8, 0.9, 0.95, 0.98], 'clf__C': np.logspace(-2, 2, 7)})

    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc',
                      cv=cv, n_jobs=-1, refit=True, verbose=0)
    gs.fit(K_feat, y_visible, groups=groups)

    # CV performance
    best_auc = gs.best_score_
    best_params = gs.best_params_

    # Fit best on full data and return the decision values/probs if you want
    best_model = gs.best_estimator_
    proba = best_model.predict_proba(K_feat)[:, 1]
    auc_all = roc_auc_score(y_visible, proba)

    return best_model, best_auc, auc_all, best_params


# ------------ utils ------------

def _as_np(a, name, two_d=False):
    arr = a.to_numpy() if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)
    if two_d and arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _stack_params(models):
    """Stack per-neuron parameter vectors into an (N, P) matrix."""
    P = len(models[0].params)
    M = np.empty((len(models), P), float)
    for i, m in enumerate(models):
        M[i, :] = m.params
    return M  # shape (N, P)

# ------------ fit per-neuron GLMs ------------


def fit_glm_poisson_per_neuron(K, X, y, offset, train_idx, alpha=0.0):
    """
    Fit Poisson GLM per neuron with design [1, vis, feats], with offset=log(dt).
    Accepts pandas or numpy.
    Returns a list of fitted results, one per neuron.
    """
    K = _as_np(K, "binned_spikes", two_d=True)     # (T, N)
    X = _as_np(X, "binned_feats",  two_d=True)     # (T, F)
    y = _as_np(y, "y_visible").ravel()             # (T,)
    offset = _as_np(offset, "offset").ravel()      # (T,)

    T, N = K.shape
    if not (X.shape[0] == y.shape[0] == offset.shape[0] == T):
        raise ValueError("Row mismatch among K, X, y, offset")

    Xdesign = np.column_stack(
        [np.ones((T, 1)), y.reshape(-1, 1), X])  # [1, vis, feats]
    models = []
    for n in range(N):
        endog = K[train_idx, n]
        exog = Xdesign[train_idx]
        off = offset[train_idx]
        mod = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=off)
        try:
            res = (mod.fit_regularized(alpha=alpha, L1_wt=0.0)
                   if alpha > 0 else mod.fit())
            # guard: NaN params → fallback to unregularized if needed
            if not np.all(np.isfinite(res.params)):
                res = mod.fit()
        except Exception:
            # if a neuron completely fails, use a tiny model (zero effect)
            # so it contributes ~0 to LLR
            p = exog.shape[1]

            class Dummy:
                params = np.zeros(p)
            res = Dummy()
        models.append(res)
    return models

# ------------ stable LLR (no overflow) ------------


def llr_from_models(models, X, offset):
    """
    Compute per-bin LLR contributions *without* overflow:
      LLR_t = sum_n [ k*(eta1-eta0) - (exp(eta1)-exp(eta0)) ]
    but we return (eta1, eta0) stacked so caller can finish with K.
    """
    X = _as_np(X, "binned_feats", two_d=True)      # (T, F)
    offset = _as_np(offset, "offset").ravel()      # (T,)
    T = X.shape[0]

    # Build [1, vis, feats] for both vis states
    X1 = np.column_stack([np.ones((T, 1)), np.ones((T, 1)), X])   # vis=1
    X0 = np.column_stack([np.ones((T, 1)), np.zeros((T, 1)), X])  # vis=0

    P = _stack_params(models)       # (N, P)
    # eta matrices: (T, N)
    eta1 = X1 @ P.T + offset[:, None]
    eta0 = X0 @ P.T + offset[:, None]
    return eta1, eta0


def _stable_lambda_diff(eta1, eta0):
    """
    Compute exp(eta1) - exp(eta0) stably using log-sum-exp trick.
    Shapes: (T, N) -> (T, N)
    """
    m = np.maximum(eta1, eta0)
    return np.exp(m) * (np.exp(eta1 - m) - np.exp(eta0 - m))

# ------------ cross-validated decoding ------------


def cv_decode_glm(
    K, X, y, groups, dt, alpha=0.0, n_splits=5, scale_X=True, verbose=False
):
    """
    GLM-LLR decoder with GroupKFold CV.
    - K: (T,N) spikes (DF or ndarray)
    - X: (T,F) feats (DF or ndarray). F can be 0.
    - y, groups: (T,) ndarrays
    - dt: scalar or (T,) ndarray
    Returns mean AUC, std AUC.
    """
    K = _as_np(K, "binned_spikes", two_d=True)
    X = _as_np(X, "binned_feats",  two_d=True)
    y = _as_np(y, "y_visible").ravel()
    g = _as_np(groups, "groups").ravel()
    dt = _as_np(dt, "dt").ravel() if np.ndim(dt) else np.array([float(dt)])

    T, N = K.shape
    if X.shape[0] != T or y.shape[0] != T or g.shape[0] != T:
        raise ValueError("Row mismatch among K, X, y, groups")
    if dt.size == 1:
        dt = np.full(T, float(dt[0]))
    if dt.shape[0] != T:
        raise ValueError("dt must be scalar or length T")
    if (dt <= 0).any():
        raise ValueError("dt must be > 0")

    # --- drop rows with any NaN/Inf ---
    fin_mask = np.isfinite(K).all(axis=1) & np.isfinite(X).all(
        axis=1) & np.isfinite(y) & np.isfinite(g) & np.isfinite(dt)
    if not fin_mask.all():
        if verbose:
            print(f"Dropping {np.sum(~fin_mask)} rows with NaN/Inf before CV.")
        K, X, y, g, dt = K[fin_mask], X[fin_mask], y[fin_mask], g[fin_mask], dt[fin_mask]
        T = K.shape[0]

    offset = np.log(dt)

    # ensure enough groups
    uniq = np.unique(g)
    if uniq.size < n_splits:
        n_splits = max(2, uniq.size)

    cv = GroupKFold(n_splits=n_splits)
    aucs = []

    for tr, te in cv.split(X, y, g):
        Xtr, Xte = X[tr], X[te]
        if scale_X and X.shape[1] > 0:
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd

        # fit per-neuron GLMs on train split
        models = fit_glm_poisson_per_neuron(K, Xtr, y, offset, tr, alpha=alpha)

        # get eta1, eta0 on test split using SAME scaling applied above
        eta1, eta0 = llr_from_models(models, Xte, offset[te])

        # stable LLR on test split
        # LLR = sum_n [ k*(eta1-eta0) - (exp(eta1)-exp(eta0)) ]
        Kte = K[te]
        term1 = (Kte * (eta1 - eta0)).sum(axis=1)
        term2 = _stable_lambda_diff(eta1, eta0).sum(axis=1)
        llr = term1 - term2

        # guard against any residual NaN/Inf (shouldn't happen now)
        good = np.isfinite(llr)
        if not np.all(good):
            if verbose:
                bad = np.sum(~good)
                print(
                    f"Warning: {bad} NaN/Inf LLR entries on test split; dropping them for AUC.")
            llr = llr[good]
            yte = y[te][good]
        else:
            yte = y[te]

        aucs.append(roc_auc_score(yte, llr))

    return float(np.mean(aucs)), float(np.std(aucs))


# 1) Build wide params from report['coefs_df']

def params_df_from_coefs_df(coefs_df: pd.DataFrame,
                            unit_col='cluster', term_col='term', coef_col='coef') -> pd.DataFrame:
    params_df = coefs_df.pivot(
        index=unit_col, columns=term_col, values=coef_col).sort_index()
    # If duplicate (unit, term) exist, last wins automatically.
    return params_df

# 2) Align params row order to df_Y columns


def align_params_to_Y(params_df: pd.DataFrame, df_Y: pd.DataFrame) -> pd.DataFrame:
    # Try direct index match to df_Y column labels
    try:
        return params_df.loc[df_Y.columns]
    except KeyError:
        # Try string-based alignment if dtypes differ
        p_idx = params_df.index.astype(str)
        y_cols = df_Y.columns.astype(str)
        aligned = params_df.copy()
        aligned.index = p_idx
        return aligned.loc[y_cols].set_axis(df_Y.columns, axis=0)

# 3) Decode on ALL rows using fitted params (no CV)


def decode_from_fitted_glm(
    df_X: pd.DataFrame,               # design you fit with (T x P)
    # spikes (T x N) — columns must be neurons (same order as params rows)
    df_Y: pd.DataFrame,
    offset_log,                       # scalar or (T,) log(dt)
    # (N x P) — rows match df_Y columns after align
    params_df: pd.DataFrame,
    *,
    # name of your visibility term in df_X / params_df (e.g., 'vis', 'visible')
    vis_col='vis',
    intercept_names=('const', 'Intercept', 'intercept'),
    return_prob=True
):
    X = df_X.copy()
    Y = df_Y.copy()

    # Ensure intercept present if in params
    if any(c in params_df.columns for c in intercept_names):
        ic = next(c for c in intercept_names if c in params_df.columns)
        if ic not in X.columns:
            X[ic] = 1.0

    # Keep only terms that exist in params; add missing non-vis terms as 0
    keep = [c for c in X.columns if c in params_df.columns]
    X = X[keep]
    for c in params_df.columns:
        if c not in X.columns:
            if c == vis_col:
                raise ValueError(
                    f"vis_col '{vis_col}' is not in df_X; it must exist to toggle 0/1.")
            X[c] = 0.0
    # exact column order match
    X = X[params_df.columns]

    # offset vector
    off = (np.full(len(X), float(offset_log), dtype=float)
           if np.isscalar(offset_log)
           else np.asarray(offset_log, float).reshape(-1))
    if off.shape[0] != len(X):
        raise ValueError("offset_log length must match rows of df_X/df_Y")

    # drop rows with NaN/Inf anywhere
    fin = (np.isfinite(X.to_numpy()).all(axis=1)
           & np.isfinite(Y.to_numpy()).all(axis=1)
           & np.isfinite(off))
    if not fin.all():
        X = X.loc[fin].reset_index(drop=True)
        Y = Y.loc[fin].reset_index(drop=True)
        off = off[fin]

    # Toggle vis 0→1 and compute eta, then LLR
    if vis_col not in X.columns:
        raise ValueError(f"'{vis_col}' term not found in X/params. "
                         "Re-fit including a visibility regressor or pass the correct name via vis_col=.")
    X0 = X.copy()
    X0[vis_col] = 0.0
    X1 = X.copy()
    X1[vis_col] = 1.0

    B = params_df.to_numpy(float)                 # (N, P)
    X0m = X0.to_numpy(float)
    X1m = X1.to_numpy(float)
    eta0 = X0m @ B.T + off[:, None]               # (T, N)
    eta1 = X1m @ B.T + off[:, None]

    # Stable exp(eta1)-exp(eta0)
    m = np.maximum(eta1, eta0)
    lam_diff = exp_diff_stable(eta1, eta0)

    print('[debug] eta0 range:', float(
        np.nanmin(eta0)), float(np.nanmax(eta0)))
    print('[debug] eta1 range:', float(
        np.nanmin(eta1)), float(np.nanmax(eta1)))

    Yn = Y.to_numpy(float)
    llr = (Yn * (eta1 - eta0) - lam_diff).sum(axis=1, dtype=np.float64)

    if return_prob:
        p = 0.5 * (1.0 + np.tanh(0.5 * llr))
        return llr, p
    return llr


def exp_diff_stable(a, b, clip_hi=80.0):
    # computes exp(a) - exp(b) stably, elementwise
    # clip_hi ≈ 80 keeps exp(clip_hi) ~ 5.54e34 (far below overflow ~1e308 in float64)
    use_a = a >= b
    hi = np.where(use_a, a, b)
    lo = np.where(use_a, b, a)
    # exp(hi) * (1 - exp(lo-hi))  when hi=a
    # exp(hi) * (exp(lo-hi) - 1)  when hi=b  -> same with a flipped sign below
    hi_c = np.clip(hi, None, clip_hi)
    scale = np.exp(hi_c)
    # note: (lo - hi) <= 0, so expm1 stays in [-1, 0]
    delta = -np.expm1(lo - hi)        # in [0, 1]
    out = scale * np.where(use_a, delta, -delta)
    # If we clipped hi downward, we underestimated magnitude slightly.
    # That’s acceptable for LLR regularization; if you need tighter control, detect and treat as -inf LLR instead.
    return out


def align_params_to_Y(params_df: pd.DataFrame, df_Y: pd.DataFrame) -> pd.DataFrame:
    """
    Align per-unit params to the columns of df_Y (units).
    Missing units are filled with zeros so they contribute nothing to decoding.
    """
    wanted = df_Y.columns

    # 1) fast path: exact index selection
    try:
        return params_df.loc[wanted]
    except KeyError:
        pass

    # 2) permissive reindex (preserve dtype of df_Y.columns)
    aligned = params_df.reindex(wanted)

    # 3) if still missing due to dtype mismatch (int vs str), try string match
    if aligned.isna().all(axis=1).any():
        tmp = params_df.copy()
        tmp.index = tmp.index.astype(str)
        aligned2 = tmp.reindex(wanted.astype(str))
        aligned2.index = wanted  # restore original dtype/order
        aligned = aligned2

    # 4) warn & fill
    missing_mask = aligned.isna().all(axis=1)
    if missing_mask.any():
        missing = list(map(str, aligned.index[missing_mask][:10].tolist()))
        print(f"[align_params_to_Y] WARNING: {missing_mask.sum()} unit(s) missing params; "
              f"examples: {missing}. Filling zeros (those units won't affect decoding).")
    return aligned.fillna(0.0)
