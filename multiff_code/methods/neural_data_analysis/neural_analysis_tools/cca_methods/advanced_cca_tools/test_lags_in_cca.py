import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import List, Optional

# ============================================================
# Utility: convert DataFrame / Series to NumPy safely
# ============================================================
def _as_numpy(A):
    """
    Convert input to a NumPy array.
    - Handles pandas DataFrame/Series by calling .to_numpy().
    - If already ndarray, returns unchanged.
    This is important because sklearn CCA expects NumPy arrays,
    not pandas objects with labels.
    """
    if isinstance(A, (pd.DataFrame, pd.Series)):
        return A.to_numpy()
    return np.asarray(A)

# ============================================================
# Utility: standardize features before CCA
# ============================================================
def _standardize(X, Y):
    """
    Standardize each variable (mean=0, std=1).
    CCA is scale-sensitive: without standardization,
    variables with large variance dominate the solution.
    """
    sx, sy = StandardScaler(), StandardScaler()
    Xs, Ys = sx.fit_transform(X), sy.fit_transform(Y)
    return Xs, Ys, sx, sy

# ============================================================
# Utility: train CCA on (Xtr,Ytr) and evaluate correlation on test set
# ============================================================
def _train_then_transform_cancorr(Xtr, Ytr, Xte, Yte, max_iter=1000):
    """
    Train CCA on training data and compute test canonical correlation.
    Steps:
      1. Standardize train and test.
      2. Fit CCA (n_components=1).
      3. Transform test data to canonical variates (U,V).
      4. Return correlation between U and V on test set.
    """
    Xtr_s, Ytr_s, sx, sy = _standardize(Xtr, Ytr)
    Xte_s, Yte_s = sx.transform(Xte), sy.transform(Yte)
    cca = CCA(n_components=1, max_iter=max_iter)
    cca.fit(Xtr_s, Ytr_s)
    Ute, Vte = cca.transform(Xte_s, Yte_s)
    return float(np.corrcoef(Ute[:, 0], Vte[:, 0])[0, 1])

# ============================================================
# Utility: partial correlation
# ============================================================
def _partial_corr_with_controls(x, u, C):
    """
    Compute partial correlation between:
      - x (one lag variable)
      - u (canonical variate from full model)
      controlling for all other (non-lag) features C.
    Idea:
      - Regress x ~ C to remove shared variance -> residual rx.
      - Regress u ~ C to remove shared variance -> residual ru.
      - Correlate rx and ru.
    This tells us the 'unique' contribution of each lagged column
    once other features are accounted for.
    """
    if C.size == 0:  # no controls
        return float(np.corrcoef(x, u)[0, 1])
    lr = LinearRegression()
    lr.fit(C, x); rx = x - lr.predict(C)
    lr.fit(C, u); ru = u - lr.predict(C)
    return float(np.corrcoef(rx, ru)[0, 1])

# ============================================================
# Result container
# ============================================================
@dataclass
class BlockInFullResult:
    test_mean_BASE: float                  # mean test cancorr with BASE model
    test_mean_BASE_plus_best1: float       # mean test cancorr with BASE + best single lag
    test_mean_BASE_plus_all: float         # mean test cancorr with BASE + all lag columns
    delta_all_vs_base: float               # improvement of full block over BASE
    delta_all_vs_best1: float              # improvement of full block over best single lag
    best_single_indices: List[int]         # chosen lag index per fold
    partial_loadings_full: Optional[np.ndarray]  # partial loadings for lag cols
    notes: str                             # explanation of what these values mean

# ============================================================
# Main function
# ============================================================
def evaluate_lag_block_in_full_model(
    X_full, Y, lag_block_cols,
    n_splits=5, random_state=0, max_iter=1000,
    compute_partial_loadings=True
) -> BlockInFullResult:
    """
    Evaluate the importance of one feature's lag block while
    keeping all other features in the model.
    
    Arguments
    ---------
    X_full : DataFrame or ndarray (n_samples, n_features)
        Full design matrix including all features and lags.
    Y : DataFrame or ndarray (n_samples, q)
        Opposite view (e.g. neural or behavioral variables).
    lag_block_cols : list of int
        Column indices of X_full corresponding to this feature's lag block.
    n_splits : int
        Number of folds for cross-validation.
    random_state : int
        Reproducibility seed for CV splits.
    max_iter : int
        Max iterations for CCA solver.
    compute_partial_loadings : bool
        Whether to compute partial correlations of lag cols with canonical variate.

    Returns
    -------
    BlockInFullResult with CV means and partial loadings.
    """

    # --- Convert inputs ---
    X_is_df = isinstance(X_full, pd.DataFrame)
    X_np = _as_numpy(X_full)
    Y_np = _as_numpy(Y)
    n, p = X_np.shape

    # --- Build mask for lag block columns ---
    mask_block = np.zeros(p, dtype=bool)
    mask_block[np.asarray(lag_block_cols, dtype=int)] = True

    # --- Slice into BASE vs BLOCK ---
    # BASE = all features except lag block
    # BLOCK = only the lag block
    if X_is_df:
        X_base  = X_full.iloc[:, ~mask_block].to_numpy()
        X_block = X_full.iloc[:,  mask_block].to_numpy()
    else:
        X_base  = X_np[:, ~mask_block]
        X_block = X_np[:,  mask_block]

    # --- CV: compare BASE, BASE+best1, BASE+all ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores_base, scores_best1, scores_all, best_idx_per_fold = [], [], [], []

    for tr, te in kf.split(X_base):
        # 1) BASE model (no lag block)
        r_base = _train_then_transform_cancorr(X_base[tr], Y_np[tr], X_base[te], Y_np[te], max_iter=max_iter)
        scores_base.append(r_base)

        # 2) BASE + best single lag (choose on train)
        best_j, best_r = 0, -np.inf
        for j in range(X_block.shape[1]):
            Xtr_single = np.column_stack([X_base[tr], X_block[tr, [j]]])
            Xte_single = np.column_stack([X_base[te], X_block[te, [j]]])
            rj = _train_then_transform_cancorr(Xtr_single, Y_np[tr], Xte_single, Y_np[te], max_iter=max_iter)
            if rj > best_r:
                best_r, best_j = rj, j
        best_idx_per_fold.append(best_j)
        scores_best1.append(best_r)

        # 3) BASE + all lag cols
        Xtr_all = np.column_stack([X_base[tr], X_block[tr]])
        Xte_all = np.column_stack([X_base[te], X_block[te]])
        r_all = _train_then_transform_cancorr(Xtr_all, Y_np[tr], Xte_all, Y_np[te], max_iter=max_iter)
        scores_all.append(r_all)

    # --- Summarize CV results ---
    mean_base  = float(np.mean(scores_base))
    mean_best1 = float(np.mean(scores_best1))
    mean_all   = float(np.mean(scores_all))

    # --- Partial structure loadings (optional) ---
    partial_loads = None
    if compute_partial_loadings:
        # Fit full model (BASE+all lags) on all data
        X_full_np = np.column_stack([X_base, X_block])
        Xs, Ys, sx, sy = _standardize(X_full_np, Y_np)
        cca = CCA(n_components=1, max_iter=max_iter)
        U, V = cca.fit_transform(Xs, Ys)
        U = U[:, 0]  # canonical variate from X side

        n_base = X_base.shape[1]
        C = Xs[:, :n_base]    # controls = non-lag features
        B = Xs[:, n_base:]    # standardized lag block
        partial_loads = np.array([_partial_corr_with_controls(B[:, j], U, C)
                                  for j in range(B.shape[1])])

    # --- Pack results ---
    return BlockInFullResult(
        test_mean_BASE=mean_base,
        test_mean_BASE_plus_best1=mean_best1,
        test_mean_BASE_plus_all=mean_all,
        delta_all_vs_base=mean_all - mean_base,
        delta_all_vs_best1=mean_all - mean_best1,
        best_single_indices=best_idx_per_fold,
        partial_loadings_full=partial_loads,
        notes=("BASE = all features except lag block. "
               "BASE+best1 = add single best lag. "
               "BASE+all = add full lag block. "
               "Partial loadings = unique correlation of each lag with canonical variate.")
    )
    


import matplotlib.pyplot as plt
import numpy as np

def plot_lag_block_loadings(partial_loadings: np.ndarray,
                            best_single_indices=None,
                            title: str = "Lag block partial loadings"):
    """
    Plot the partial structure loadings (unique corr with canonical variate per lag).
    
    Args:
    -------
    partial_loadings : np.ndarray, shape (n_lags,)
        Output from res.partial_loadings_full
    best_single_indices : list of int, optional
        Indices chosen as 'best lag' in each CV fold. Will be overlaid as vertical lines.
    title : str
        Figure title.
    """
    lags = np.arange(len(partial_loadings))
    
    plt.figure(figsize=(8,4))
    plt.plot(lags, partial_loadings, marker='o', linewidth=2, label="Partial loading")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Optionally highlight which lags were chosen in CV
    if best_single_indices is not None:
        uniq, counts = np.unique(best_single_indices, return_counts=True)
        for u, c in zip(uniq, counts):
            plt.axvline(u, color='red', linestyle=':', alpha=0.6,
                        label=f"Chosen lag {u} ({c} folds)")
    
    plt.xlabel("Lag index")
    plt.ylabel("Partial loading (corr with canonical variate | others)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

