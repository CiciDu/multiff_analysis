import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ============================================================
# Utilities
# ============================================================

def _as_numpy(A):
    """
    Convert DataFrame/Series to NumPy; pass ndarray through unchanged.
    We do this early because sklearn CCA expects arrays, not labeled pandas objects.
    """
    if isinstance(A, (pd.DataFrame, pd.Series)):
        return A.to_numpy()
    return np.asarray(A)

def _standardize(X, Y):
    """
    Standardize both views (mean 0, std 1). CCA is scale-sensitive.
    We always standardize 'inside the fold' to avoid leakage.
    """
    sx, sy = StandardScaler(), StandardScaler()
    Xs, Ys = sx.fit_transform(X), sy.fit_transform(Y)
    return Xs, Ys, sx, sy

def _fit_cca_train_corr(Xtr, Ytr, max_iter=1000) -> float:
    """
    Fit 1D CCA on training data and return *training* canonical correlation.
    Useful for sanity checks (overfit vs generalization).
    """
    Xs, Ys, _, _ = _standardize(Xtr, Ytr)
    cca = CCA(n_components=1, max_iter=max_iter)
    Utr, Vtr = cca.fit_transform(Xs, Ys)
    return float(np.corrcoef(Utr[:,0], Vtr[:,0])[0,1])

def _train_then_transform_test_corr(Xtr, Ytr, Xte, Yte, max_iter=1000) -> float:
    """
    Fit CCA on train, transform test, return *test* canonical correlation.
    This is the metric we care about for incremental/unique contribution.
    """
    Xtr_s, Ytr_s, sx, sy = _standardize(Xtr, Ytr)
    Xte_s, Yte_s = sx.transform(Xte), sy.transform(Yte)
    cca = CCA(n_components=1, max_iter=max_iter)
    cca.fit(Xtr_s, Ytr_s)
    Ute, Vte = cca.transform(Xte_s, Yte_s)
    return float(np.corrcoef(Ute[:,0], Vte[:,0])[0,1])

def _residualize_multioutput(C_train, Z_train, C_test, Z_test):
    """
    Residualize a (possibly multi-output) target Z on controls C using *train-fit* LinearRegression.
    - Fit:  Z_train ~ C_train          (multi-output OK)
    - Resid: Z_res = Z - Z_hat         (for both train and test, using train-fitted model)
    Returns:
      Z_train_res, Z_test_res
    Note: if you prefer ridge for stability, swap LinearRegression() → Ridge(alpha=...).
    """
    lr = LinearRegression()
    lr.fit(C_train, Z_train)
    Ztr_hat = lr.predict(C_train)
    Zte_hat = lr.predict(C_test)
    return Z_train - Ztr_hat, Z_test - Zte_hat

def _drop_near_constant_columns(Xtr, Xte, tol=1e-10):
    """
    Sometimes after residualization, some columns are ~constant (variance ~ 0),
    which can cause numerical issues for CCA. We drop those *by train variance*,
    and apply the same mask to test.
    """
    std = Xtr.std(axis=0)
    keep = std > tol
    # If everything is dropped (pathological), just return original to avoid crashing.
    if not np.any(keep):
        return Xtr, Xte, np.ones(Xtr.shape[1], dtype=bool)
    return Xtr[:, keep], Xte[:, keep], keep

# ============================================================
# Result container
# ============================================================

@dataclass
class PartialCCAResult:
    mean_test_partial_cancorr: float         # Average test canonical correlation across folds (residual vs residual)
    fold_test_partial_cancorr: List[float]   # Per-fold test correlations
    mean_train_partial_cancorr: float        # Average train canonical correlation (for context)
    fold_train_partial_cancorr: List[float]  # Per-fold train correlations
    kept_cols_masks: List[np.ndarray]        # Which residualized lag cols were kept after variance filter (per fold)
    notes: str

# ============================================================
# Main API: Blockwise Partial CCA (Incremental/Unique test)
# ============================================================

def partial_cca_incremental_cv(
    X_full,
    Y,
    lag_block_cols: List[int],
    n_splits: int = 5,
    random_state: int = 0,
    max_iter: int = 1000
) -> PartialCCAResult:
    """
    Blockwise Partial CCA (incremental test):
    Residualize BOTH the lag block and Y against the BASE features (all other columns),
    then run CCA between the residuals inside each CV fold.

    Interpretation:
      - If the mean TEST canonical correlation of residuals is ~0,
        the lag block adds no *unique* cross-view information beyond BASE.
      - If it's > 0 (and stable across folds), the block contributes uniquely.

    Parameters
    ----------
    X_full : array-like (n, p) or DataFrame
        Full X design with ALL features, including the lag block under test.
    Y : array-like (n, q) or DataFrame
        Opposite view for CCA.
    lag_block_cols : list[int]
        Positional indices of columns in X_full that form the lag block (e.g., [20..30]).
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for KFold shuffling.
    max_iter : int
        Max iterations for the CCA solver.

    Returns
    -------
    PartialCCAResult
        Contains per-fold and mean train/test partial canonical correlations,
        plus masks indicating which residual lag columns were kept per fold
        (in case near-constant columns were removed).
    """

    # --- Prepare arrays and masks ---
    X_is_df = isinstance(X_full, pd.DataFrame)
    X_np = _as_numpy(X_full)
    Y_np = _as_numpy(Y)
    n, p = X_np.shape

    # Boolean mask for the lag block
    mask_block = np.zeros(p, dtype=bool)
    mask_block[np.asarray(lag_block_cols, dtype=int)] = True

    # Split into BASE (everything except block) and BLOCK (just the lag block)
    if X_is_df:
        X_base  = X_full.iloc[:, ~mask_block].to_numpy()
        X_block = X_full.iloc[:,  mask_block].to_numpy()
    else:
        X_base  = X_np[:, ~mask_block]
        X_block = X_np[:,  mask_block]

    # --- Cross-validation setup ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_test_r = []
    fold_train_r = []
    kept_masks_per_fold = []

    # --- Main CV loop ---
    for tr, te in kf.split(X_np):
        # Controls in this fold: BASE features
        C_tr, C_te = X_base[tr], X_base[te]

        # 1) Residualize Y on BASE (fit on train, apply to train/test)
        Ytr_res, Yte_res = _residualize_multioutput(C_tr, Y_np[tr], C_te, Y_np[te])

        # 2) Residualize lag block on BASE (fit on train, apply to train/test)
        #    Multi-output regression: each lag column acts as a target, jointly residualized.
        Xblk_tr_res, Xblk_te_res = _residualize_multioutput(C_tr, X_block[tr], C_te, X_block[te])

        # 3) Guard against numerical issues: drop near-constant residual lag columns (by train variance)
        Xblk_tr_res_f, Xblk_te_res_f, keep_mask = _drop_near_constant_columns(Xblk_tr_res, Xblk_te_res, tol=1e-12)
        kept_masks_per_fold.append(keep_mask)

        # 4) TRAIN canonical corr on residuals (sanity check)
        r_train = _fit_cca_train_corr(Xblk_tr_res_f, Ytr_res, max_iter=max_iter)
        fold_train_r.append(r_train)

        # 5) TEST canonical corr on residuals (the quantity of interest)
        r_test = _train_then_transform_test_corr(Xblk_tr_res_f, Ytr_res, Xblk_te_res_f, Yte_res, max_iter=max_iter)
        fold_test_r.append(r_test)

    # --- Summarize results across folds ---
    mean_train = float(np.mean(fold_train_r))
    mean_test  = float(np.mean(fold_test_r))

    return PartialCCAResult(
        mean_test_partial_cancorr=mean_test,
        fold_test_partial_cancorr=fold_test_r,
        mean_train_partial_cancorr=mean_train,
        fold_train_partial_cancorr=fold_train_r,
        kept_cols_masks=kept_masks_per_fold,
        notes=(
            "We residualized Y and the lag block on BASE inside each fold, "
            "then ran 1D CCA between residuals. "
            "Near-zero mean TEST canonical correlation implies no unique contribution "
            "from the lag block beyond BASE."
        ),
    )




