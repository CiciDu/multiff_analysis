"""
Conditional decoding of a continuous variable y given a discrete 'band' label.

Implements three approaches:
  A) per-band decoder: fit separate model for each band
  B) pooled decoder with band main effect and X-by-band interactions
  C) residual decoding: remove band mean from y, then decode residual y'

Uses ridge regression with standardization inside each CV fold.
Outputs tidy DataFrames with R^2, MSE, Corr for each approach (and per band when applicable).

Expected inputs:
  - X: np.ndarray, shape (n_samples, n_features)
  - y: np.ndarray, shape (n_samples,)
  - band: array-like, shape (n_samples,) with discrete labels (int/str)
  - groups (optional): array-like, shape (n_samples,) for GroupKFold (e.g., session id)

Dependencies: numpy, pandas, scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _make_cv(n_splits: int, groups: np.ndarray | None, random_state: int) -> object:
    if groups is None:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return GroupKFold(n_splits=n_splits)


def _fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    model.fit(X_train_z, y_train)
    return model.predict(X_test_z)


def _band_means_from_train(y_train: np.ndarray, band_train: np.ndarray) -> dict:
    band_means = {}
    for b in np.unique(band_train):
        mask = band_train == b
        band_means[b] = float(np.mean(y_train[mask])) if np.any(mask) else 0.0
    return band_means


def _residualize_y_by_band(
    y: np.ndarray,
    band: np.ndarray,
    band_means: dict,
    global_fallback: float,
) -> np.ndarray:
    # y_res[i] = y[i] - mean_train(y | band=b_i)
    # if unseen band in test fold, subtract global mean of train fold
    y_res = np.empty_like(y, dtype=float)
    for i, b in enumerate(band):
        y_res[i] = float(y[i]) - float(band_means.get(b, global_fallback))
    return y_res


def decode_continuous_conditioned_on_band(
    X: np.ndarray,
    y: np.ndarray,
    band: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict of DataFrames:
      - 'per_band': approach A metrics per band per fold
      - 'pooled_interaction': approach B metrics per fold (overall + per band)
      - 'residual': approach C metrics per fold (overall + per band)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    band = np.asarray(band)
    if groups is not None:
        groups = np.asarray(groups)

    unique_bands = np.unique(band)
    cv = _make_cv(n_splits=n_splits, groups=groups, random_state=random_state)

    per_band_rows = []
    pooled_rows = []
    resid_rows = []

    # Precompute one-hot for band (stable ordering)
    band_to_col = {b: i for i, b in enumerate(unique_bands)}
    B = np.zeros((band.size, unique_bands.size), dtype=float)
    for i, b in enumerate(band):
        B[i, band_to_col[b]] = 1.0

    # Build pooled interaction design: [X, B, X*B]
    # X*B is block-wise: for each band column k, include X * B[:, k]
    X_by_band_blocks = []
    for k in range(B.shape[1]):
        X_by_band_blocks.append(X * B[:, [k]])
    X_int = np.concatenate(X_by_band_blocks, axis=1)  # (n, n_features*n_bands)
    X_pooled = np.concatenate([X, B, X_int], axis=1)

    split_iter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        band_train, band_test = band[train_idx], band[test_idx]

        # -------------------------
        # A) Per-band decoder
        # -------------------------
        for b in unique_bands:
            tr_mask = band_train == b
            te_mask = band_test == b
            if np.sum(tr_mask) < 5 or np.sum(te_mask) < 2:
                per_band_rows.append({
                    'approach': 'per_band',
                    'fold': fold_idx,
                    'band': b,
                    'n_train': int(np.sum(tr_mask)),
                    'n_test': int(np.sum(te_mask)),
                    'r2': np.nan,
                    'mse': np.nan,
                    'corr': np.nan,
                })
                continue

            y_pred_b = _fit_predict_ridge(
                X_train=X_train[tr_mask],
                y_train=y_train[tr_mask],
                X_test=X_test[te_mask],
                alpha=alpha,
            )
            y_true_b = y_test[te_mask]

            per_band_rows.append({
                'approach': 'per_band',
                'fold': fold_idx,
                'band': b,
                'n_train': int(np.sum(tr_mask)),
                'n_test': int(np.sum(te_mask)),
                'r2': float(r2_score(y_true_b, y_pred_b)),
                'mse': float(mean_squared_error(y_true_b, y_pred_b)),
                'corr': _safe_corr(y_true_b, y_pred_b),
            })

        # -------------------------
        # B) Pooled with band main + interactions
        # -------------------------
        Xp_train, Xp_test = X_pooled[train_idx], X_pooled[test_idx]
        y_pred_pooled = _fit_predict_ridge(Xp_train, y_train, Xp_test, alpha=alpha)

        pooled_rows.append({
            'approach': 'pooled_interaction',
            'fold': fold_idx,
            'band': 'ALL',
            'n_train': int(train_idx.size),
            'n_test': int(test_idx.size),
            'r2': float(r2_score(y_test, y_pred_pooled)),
            'mse': float(mean_squared_error(y_test, y_pred_pooled)),
            'corr': _safe_corr(y_test, y_pred_pooled),
        })
        for b in unique_bands:
            te_mask = band_test == b
            if np.sum(te_mask) < 2:
                pooled_rows.append({
                    'approach': 'pooled_interaction',
                    'fold': fold_idx,
                    'band': b,
                    'n_train': int(np.sum(band_train == b)),
                    'n_test': int(np.sum(te_mask)),
                    'r2': np.nan,
                    'mse': np.nan,
                    'corr': np.nan,
                })
                continue
            y_true_b = y_test[te_mask]
            y_pred_b = y_pred_pooled[te_mask]
            pooled_rows.append({
                'approach': 'pooled_interaction',
                'fold': fold_idx,
                'band': b,
                'n_train': int(np.sum(band_train == b)),
                'n_test': int(np.sum(te_mask)),
                'r2': float(r2_score(y_true_b, y_pred_b)),
                'mse': float(mean_squared_error(y_true_b, y_pred_b)),
                'corr': _safe_corr(y_true_b, y_pred_b),
            })

        # -------------------------
        # C) Residual decoding: y' = y - E[y|band] (estimated on train fold)
        # -------------------------
        band_means = _band_means_from_train(y_train=y_train, band_train=band_train)
        global_mean = float(np.mean(y_train))

        y_train_res = _residualize_y_by_band(y_train, band_train, band_means, global_mean)
        y_test_res = _residualize_y_by_band(y_test, band_test, band_means, global_mean)

        y_pred_res = _fit_predict_ridge(X_train, y_train_res, X_test, alpha=alpha)

        resid_rows.append({
            'approach': 'residual',
            'fold': fold_idx,
            'band': 'ALL',
            'n_train': int(train_idx.size),
            'n_test': int(test_idx.size),
            'r2': float(r2_score(y_test_res, y_pred_res)),
            'mse': float(mean_squared_error(y_test_res, y_pred_res)),
            'corr': _safe_corr(y_test_res, y_pred_res),
        })
        for b in unique_bands:
            te_mask = band_test == b
            if np.sum(te_mask) < 2:
                resid_rows.append({
                    'approach': 'residual',
                    'fold': fold_idx,
                    'band': b,
                    'n_train': int(np.sum(band_train == b)),
                    'n_test': int(np.sum(te_mask)),
                    'r2': np.nan,
                    'mse': np.nan,
                    'corr': np.nan,
                })
                continue
            y_true_b = y_test_res[te_mask]
            y_pred_b = y_pred_res[te_mask]
            resid_rows.append({
                'approach': 'residual',
                'fold': fold_idx,
                'band': b,
                'n_train': int(np.sum(band_train == b)),
                'n_test': int(np.sum(te_mask)),
                'r2': float(r2_score(y_true_b, y_pred_b)),
                'mse': float(mean_squared_error(y_true_b, y_pred_b)),
                'corr': _safe_corr(y_true_b, y_pred_b),
            })

    df_per_band = pd.DataFrame(per_band_rows)
    df_pooled = pd.DataFrame(pooled_rows)
    df_resid = pd.DataFrame(resid_rows)

    return {
        'per_band': df_per_band.sort_values(['band', 'fold']).reset_index(drop=True),
        'pooled_interaction': df_pooled.sort_values(['band', 'fold']).reset_index(drop=True),
        'residual': df_resid.sort_values(['band', 'fold']).reset_index(drop=True),
    }


# -------------------------
# Example usage (replace with your arrays)
# -------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 2000
    p = 40

    # Fake data: 3 bands, band-dependent slope
    band = rng.choice(['low', 'mid', 'high'], size=n, replace=True)
    X = rng.standard_normal((n, p))

    true_w = {
        'low': rng.standard_normal(p) * 0.2,
        'mid': rng.standard_normal(p) * 0.4,
        'high': rng.standard_normal(p) * 0.6,
    }
    y = np.zeros(n)
    for i in range(n):
        y[i] = X[i] @ true_w[band[i]] + (0.3 if band[i] == 'high' else 0.0) + rng.standard_normal() * 0.5

    # Optional grouping (e.g., session ids) to avoid leakage
    groups = rng.integers(0, 20, size=n)

    out = decode_continuous_conditioned_on_band(
        X=X,
        y=y,
        band=band,
        groups=groups,
        n_splits=5,
        alpha=10.0,
        random_state=0,
    )

    for k, df in out.items():
        print('\n===', k, '===')
        # quick summary: mean across folds for ALL + each band
        summary = (
            df.groupby(['approach', 'band'], dropna=False)[['r2', 'mse', 'corr']]
              .mean()
              .reset_index()
              .sort_values(['approach', 'band'])
        )
        print(summary.to_string(index=False))
