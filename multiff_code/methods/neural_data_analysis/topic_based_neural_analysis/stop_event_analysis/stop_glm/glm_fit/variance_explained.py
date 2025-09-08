
"""
stop_glm_metrics.py
===================

Variance-explained metrics for stop-aligned GLMs and related models.

Implements two complementary families of evaluation metrics:

1. Single-neuron temporal variance explained (VE)
   - For each neuron, compute how much of its firing rate variance over time
     is captured by the model’s predictions.
   - Formula: VE_j = 1 - Var_t(y_j - yhat_j) / Var_t(y_j)
   - Report distribution across neurons, plus mean/median/weighted summary.

2. Population structure variance explained (VE in PC-space)
   - Assemble the population matrix (time x neurons).
   - Define a low-dimensional subspace via PCA on observed activity.
   - Project both observed and predicted activity into this subspace.
   - Compare structures via Frobenius norm similarity:
       VE_pop = 1 - ||X_proj - Y_proj||_F^2 / ||X_proj||_F^2
   - Mirrors Lakshminarasimhan et al. (Nature 2023).

Also includes:
- Per-stop breakdown (grouped by stop_id) for condition-wise VE analysis.
- Utilities for safe variance computation and shape checking.

References
----------
Lakshminarasimhan, K.J., et al. (2023).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_df_Y_pred_from_results(results, df_X, offset_log, df_Y, clip_eta=(-30.0, 30.0)):
    """
    Assemble predicted expected counts (μ) per bin × neuron, robustly.

    Strategy per cluster:
      1) Use res.fittedvalues *if* length matches and all finite.
      2) Else try res.predict on aligned X/offset.
      3) Else compute μ = exp(clip( X_aligned @ params + offset , clip_eta )).
         - Coerce any NaNs/±inf in params or offset to finite before computing.
         - Ensure 'const' column is 1.0 if the model was fit with one.

    Returns
    -------
    df_Y_pred  : (T x N) DataFrame of expected counts per bin × neuron.
    """
    import numpy as np
    import pandas as pd

    T, N = df_Y.shape
    target_index = df_Y.index
    mu_cols = {}

    non_converged = []
    used_fitted = []
    used_predict = []
    used_params = []
    had_nan_fitvals = []
    coerced_params = []
    pred_nonfinite = []

    def _align_exog_and_offset(res):
        # Columns the model was trained on (order matters)
        train_cols = list(getattr(res.model, 'exog_names', df_X.columns))
        X_aligned = df_X.reindex(columns=train_cols, fill_value=0.0)
        # If model expected an explicit constant column, set it to 1.0
        if 'const' in X_aligned.columns:
            X_aligned['const'] = 1.0
        # Align rows to df_Y index
        X_aligned = X_aligned.reindex(index=target_index, fill_value=0.0)

        # Align offset to rows; coerce to finite and clip to avoid overflow
        if hasattr(offset_log, 'reindex'):
            off = offset_log.reindex(target_index)
        else:
            off = np.asarray(offset_log, float)
            if off.shape[0] != T:
                raise ValueError("offset_log length does not match df_Y rows.")
            off = pd.Series(off, index=target_index)
        off = np.asarray(off, float)
        off = np.nan_to_num(off, nan=0.0, posinf=clip_eta[1], neginf=clip_eta[0])
        off = np.clip(off, clip_eta[0], clip_eta[1])
        return X_aligned, off

    for cid in df_Y.columns:
        res = results.get(cid, None)

        if res is None:
            # skipped cluster (e.g., all-zero unit)
            mu_cols[cid] = np.zeros(T, dtype=float)
            continue

        if getattr(res, 'converged', True) is False:
            non_converged.append(cid)

        # 1) Try cached fittedvalues if shape matches and all finite
        fv = getattr(res, 'fittedvalues', None)
        if fv is not None:
            fv = np.asarray(fv, float).reshape(-1)
            if fv.size == T and np.isfinite(fv).all():
                mu_cols[cid] = fv
                used_fitted.append(cid)
                continue
            else:
                had_nan_fitvals.append(cid)

        # Prepare aligned design/offset (used by paths 2 and 3)
        X_aligned, off = _align_exog_and_offset(res)

        # 2) Try statsmodels predict on aligned data
        try:
            mu = res.predict(exog=X_aligned.values, offset=off, linear=False)
            mu = np.asarray(mu, float).reshape(-1)
            # Guard against non-finite results
            if mu.size == T and np.isfinite(mu).all():
                mu_cols[cid] = mu
                used_predict.append(cid)
                continue
        except Exception:
            pass  # fall through to manual compute

        # 3) Last resort: compute μ = exp(clip( X @ params + offset , clip_eta ))
        params = np.asarray(getattr(res, 'params', None), float)
        if params.ndim != 1 or params.shape[0] != X_aligned.shape[1]:
            raise RuntimeError(
                f'Cannot compute predictions for cluster {cid}: '
                f'params shape {params.shape} incompatible with X {X_aligned.shape}'
            )
        # Coerce any NaNs/±inf in params to zero (neutral) before dot product
        if not np.isfinite(params).all():
            coerced_params.append(cid)
            params = np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

        eta = X_aligned.values @ params
        eta = np.nan_to_num(eta, nan=0.0, posinf=clip_eta[1], neginf=clip_eta[0])
        eta = eta + off
        eta = np.clip(eta, clip_eta[0], clip_eta[1])
        mu = np.exp(eta)

        # Final guard (should be finite after clipping)
        if not np.isfinite(mu).all():
            pred_nonfinite.append(cid)
            mu = np.nan_to_num(mu, nan=0.0, posinf=np.exp(clip_eta[1]), neginf=0.0)

        mu_cols[cid] = mu
        used_params.append(cid)

    if non_converged:
        print('Non-converged clusters:', sorted(non_converged))
    if had_nan_fitvals:
        print('Clusters had NaN/inf in fittedvalues (Recomputed):', sorted(had_nan_fitvals))
    if coerced_params:
        print('Clusters had non-finite params (coerced to 0 for prediction):', sorted(coerced_params))
    if pred_nonfinite:
        print('Clusters produced non-finite μ even after predict; coerced:', sorted(pred_nonfinite))
    # Optional: quick summary of which path was used
    # print({'fitted': len(used_fitted), 'predict': len(used_predict), 'params': len(used_params)})

    df_Y_pred = pd.DataFrame(mu_cols, index=df_Y.index, columns=df_Y.columns)
    return df_Y_pred


def _check_same_shape(X, X_hat):
    """
    Validate inputs for all VE computations.

    Ensures:
      - X (observed) and X_hat (predicted) are float arrays of the SAME shape (T, N)
      - No NaN/Inf in either array

    Why:
      - All downstream math assumes 2D time-by-neuron matrices with aligned rows.
      - Silent shape mismatches are a common source of subtle bugs (e.g., comparing
        shuffled bins or wrong session). We hard-fail early.
      - Non-finite values can poison SVD/variance computations.

    Returns
    -------
    X, X_hat : float arrays with shape (T, N)
    """
    X = np.asarray(X, float)
    X_hat = np.asarray(X_hat, float)
    if X.shape != X_hat.shape:
        raise ValueError(f'shape mismatch: X{X.shape} vs X_hat{X_hat.shape}')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(X_hat)):
        raise ValueError('X or X_hat has non-finite values.')
    return X, X_hat




def _safe_var(a, axis=0, ddof=0):
    """
    Numerically safe variance along an axis.

    Behavior:
      - Computes np.var(a, axis=axis, ddof=ddof).
      - Replaces values <= 1e-12 with 1e-12 to avoid divide-by-zero downstream.

    Why:
      - VE = 1 - Var(resid)/Var(obs) → zero (or near-zero) Var(obs) explodes.
      - Some neurons can be almost constant within a short window (low firing),
        especially with small T per stop; we clamp denominators conservatively.

    Returns
    -------
    np.ndarray variance array with minimum 1e-12
    """
    v = np.var(a, axis=axis, ddof=ddof)
    return np.where(v <= 1e-12, 1e-12, v)


# ------------------- PCA projection on observed activity -------------------
# We define PCA explicitly via SVD to keep dependencies minimal.

def _pca_project(X, k=10, var_thresh=None, center='neuron'):
    """
    Perform PCA on observed neural activity X (shape: T x N) and return
    the top principal component time-courses and loadings.

    Centering options (important):
      - 'neuron': subtract each neuron's mean over time (recommended).
                  This focuses the analysis on co-fluctuations / covariance
                  structure rather than tonic firing differences across neurons.
      - 'global': subtract one global mean (less common; treats all neurons
                  as one big vector).
      - None    : no centering (rare; only if you know you want means included).

    Dimensionality selection:
      - k: fixed number of PCs (e.g., 10 to mirror Lakshminarasimhan et al.).
      - var_thresh: choose the smallest k s.t. cumulative variance ≥ var_thresh
                    (e.g., 0.9 for 90%). If provided, overrides k.

    Numerical notes:
      - Uses economy SVD: Xc ≈ U S V^T. PCs (loadings) are columns of V.
      - Variance per PC is S^2/(T-1) when rows are observations (time).

    Returns
    -------
    X_proj : (T, k_eff)
        Scores (time-courses) of X in the top-PC space (observed-only PCA).
    Vk : (N, k_eff)
        Loadings (eigenvectors) that map neuron-space → PC-space.
    mu : (1, N)
        Mean subtracted from X prior to SVD (store to center X_hat consistently).
    k_eff : int
        The actual number of PCs retained (useful for logging/plots).
    """
    T, N = X.shape
    if center == 'neuron':
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
    elif center == 'global':
        mu = np.array([[X.mean()] * N])
        Xc = X - mu
    else:
        mu = np.zeros((1, N))
        Xc = X

    # SVD yields stable PCA without explicitly forming covariance
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # If var_thresh is requested, set k to #PCs covering that variance fraction
    if var_thresh is not None:
        var = (S ** 2) / (T - 1 + 1e-12)   # eigenvalues
        cum = np.cumsum(var) / (np.sum(var) + 1e-12)
        k = int(np.searchsorted(cum, var_thresh) + 1)  # first index meeting threshold

    # Clip k to valid range [1, rank]
    k = max(1, min(k, Vt.shape[0]))

    Vk = Vt[:k].T           # (N, k) loadings: neuron-space → PC-space
    X_proj = Xc @ Vk        # (T, k) projected time-courses
    return X_proj, Vk, mu, k


# ---------------------- VE metrics ----------------------
# Two complementary views:
#   1) Population VE in a fixed subspace (compare matrices, structure-first)
#   2) Single-neuron temporal VE (compare time-courses, unit-first)


def single_neuron_temporal_VE(X, X_hat, *, aggregate='mean'):
    """
    Single-neuron variance explained (temporal).

    For each neuron j, over time bins t:
        VE_j = 1 - Var_t( y_j(t) - yhat_j(t) ) / Var_t( y_j(t) )

    Interpretation:
      - VE_j ≈ 1   → predictions nearly match neuron's time-course
      - VE_j ≈ 0   → predictions no better than the neuron's mean over time
      - VE_j < 0   → predictions worse than baseline (can happen for small T)

    Aggregation:
      - 'mean'     : average across neurons (what many papers report)
      - 'median'   : robust to a few poor fits/outliers
      - 'weighted' : weights neurons by their temporal variance (units with
                     more dynamic range count more)

    Notes:
      - We clip VE_j to [-1, 1] purely for display stability; raw math can
        yield mild excursions outside due to finite-sample noise.

    Returns
    -------
    ve_per_neuron : (N,) array
        VE for each neuron.
    ve_agg : float
        Aggregated summary across neurons according to 'aggregate'.
    """
    X, X_hat = _check_same_shape(X, X_hat)
    resid = X - X_hat

    # Variance per neuron over time (safe-guarded)
    var_obs = _safe_var(X, axis=0)
    var_res = _safe_var(resid, axis=0)

    ve = 1.0 - (var_res / var_obs)
    ve = np.clip(ve, -1.0, 1.0)

    if aggregate == 'mean':
        agg = float(np.mean(ve))
    elif aggregate == 'median':
        agg = float(np.median(ve))
    elif aggregate == 'weighted':
        w = var_obs / (var_obs.sum() + 1e-12)
        agg = float(np.sum(w * ve))
    else:
        raise ValueError('aggregate must be mean|median|weighted')

    return ve, agg



import numpy as np
import pandas as pd

def _fro_sq(A):
    """
    Squared Frobenius norm: sum of squares over all entries.
    """
    A = np.asarray(A, float)
    return float(np.dot(A.ravel(), A.ravel()))

def population_VE_in_PCspace(X, X_hat, *, k=10, var_thresh=None, center=True, eps=1e-12):
    """
    Population variance explained by predictions Y (=X_hat) inside the k-D PC subspace
    defined from X (observed). Safe against zero-variance windows.

    Returns
    -------
    ve_pop : float in [0,1] (0 if denom ~ 0)
    k_eff  : int, number of PCs actually used
    """
    X = np.asarray(X, float)
    Y = np.asarray(X_hat, float)
    if X.shape != Y.shape:
        raise ValueError(f"X and X_hat must have same shape, got {X.shape} vs {Y.shape}")

    # Optionally center over time
    Xc = X - X.mean(axis=0, keepdims=True) if center else X.copy()
    Yc = Y - Y.mean(axis=0, keepdims=True) if center else Y.copy()

    # SVD of observed data (column/neuronal PCs)
    try:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, 0

    s2 = s**2
    tot = float(s2.sum())
    rank = int((s2 > eps).sum())

    if rank == 0 or tot <= eps:
        # no variance in this window → define VE=0, no components
        return 0.0, 0

    # Choose effective dimensionality
    if var_thresh is not None:
        cum = np.cumsum(s2) / tot
        k_eff = int(np.searchsorted(cum, float(var_thresh)) + 1)
    else:
        k_eff = int(k)

    k_eff = max(1, min(k_eff, rank))

    V_k = Vt[:k_eff, :]              # (k_eff × neurons)
    Xp = Xc @ V_k.T                  # (time × k_eff)
    Yp = Yc @ V_k.T

    den = _fro_sq(Xp)                # observed energy in subspace
    if not np.isfinite(den) or den <= eps:
        return 0.0, k_eff

    num = _fro_sq(Xp - Yp)
    ve = 1.0 - (num / den)

    # Numerical guards
    if not np.isfinite(ve):
        ve = 0.0
    ve = max(0.0, min(1.0, ve))
    return ve, k_eff

def per_stop_breakdown(X, X_hat, *, stop_ids, k=10, var_thresh=None, center=True, eps=1e-12):
    """
    Compute per-stop population VE (in PC space) and single-neuron temporal VE.
    Skips/zeros out degenerate windows safely.
    """
    # Expect X, X_hat as (T × N) arrays; stop_ids length T
    X = np.asarray(X, float)
    Y = np.asarray(X_hat, float)
    if X.shape != Y.shape:
        raise ValueError(f"X and X_hat must have same shape, got {X.shape} vs {Y.shape}")

    stop_ids = np.asarray(stop_ids)
    if stop_ids.shape[0] != X.shape[0]:
        raise ValueError("stop_ids length must match number of time bins in X/X_hat")

    rows = []
    for sid in pd.unique(stop_ids):
        m = (stop_ids == sid)
        T_win = int(m.sum())
        if T_win == 0:
            continue

        Xg, Yg = X[m], Y[m]

        # If the window is extremely short or has ~0 variance, VE_pop will be 0
        ve_pop, k_eff = population_VE_in_PCspace(
            Xg, Yg, k=k, var_thresh=var_thresh, center=center, eps=eps
        )

        # Single-neuron temporal VE (safe version)
        # VE_unit(i) = 1 - ||x_i - y_i||² / ||x_i - mean(x_i)||² (guard denom)
        Xgc = Xg - Xg.mean(axis=0, keepdims=True) if center else Xg
        denom = np.sum(Xgc**2, axis=0)
        num = np.sum((Xg - Yg)**2, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ve_unit = 1.0 - (num / np.maximum(denom, eps))
        ve_unit = np.clip(ve_unit, 0.0, 1.0)
        ve_unit_mean = float(np.nanmean(ve_unit)) if np.isfinite(ve_unit).any() else 0.0

        rows.append({
            'stop_id': sid,
            'n_bins': T_win,
            've_pop': float(ve_pop),
            'k_eff': int(k_eff),
            've_unit_mean': ve_unit_mean
        })

    return pd.DataFrame(rows, columns=['stop_id', 'n_bins', 've_pop', 'k_eff', 've_unit_mean'])



def plot_single_neuron_VE_hist(ve_per_neuron, bins=20, title='Single-neuron VE distribution'):
    """
    Plot a histogram of variance explained per neuron.
    """
    plt.figure(figsize=(5,3))
    plt.hist(ve_per_neuron, bins=bins, color='gray', edgecolor='k', alpha=0.7)
    plt.axvline(np.mean(ve_per_neuron), color='red', linestyle='--', label='mean')
    plt.xlabel('VE (per neuron)')
    plt.ylabel('# neurons')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_population_VE_bar(ve_pop, title='Population VE in PC-space'):
    """
    Plot a simple bar for population variance explained (0–1).
    """
    plt.figure(figsize=(3,4))
    plt.bar([0], [ve_pop], color='steelblue')
    plt.xticks([0], ['Population'])
    plt.ylim(0,1)
    plt.ylabel('Variance explained')
    plt.title(title)
    plt.tight_layout()
    plt.show()
