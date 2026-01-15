import numpy as np
import pandas as pd
from numpy.linalg import svd
from statsmodels.tools.tools import add_constant

import json
import os

# ============================================================
# Utilities
# ============================================================

def check_near_constant(df, tol=1e-12):
    variances = df.var(axis=0).astype(float)
    near_const = variances[variances <= tol]
    return near_const.index.tolist(), variances


def find_duplicate_columns(df, decimals=12):
    X = np.round(df.to_numpy(dtype=float), decimals)
    hashes = pd.util.hash_pandas_object(
        pd.DataFrame(X), index=False
    )

    groups = {}
    for col, h in zip(df.columns, hashes):
        groups.setdefault(h, []).append(col)

    return [g for g in groups.values() if len(g) > 1]


def condition_number(df):
    s = svd(df.to_numpy(dtype=float), compute_uv=False)
    if s.min() == 0:
        return np.inf, s
    return float(s.max() / s.min()), s


def rank_deficiency(df, tol=1e-10):
    x = df.to_numpy(dtype=float)
    r = np.linalg.matrix_rank(x, tol=tol)
    return r, df.shape[1], r < df.shape[1]


# ============================================================
# VIF (used only in multipass)
# ============================================================

def compute_vif(df, add_intercept=False):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df.copy()
    if add_intercept:
        X = add_constant(X, has_constant='add')

    Xv = X.to_numpy(dtype=float)
    cols = list(X.columns)

    vif_vals = [
        (col, float(variance_inflation_factor(Xv, i)))
        for i, col in enumerate(cols)
    ]

    vif_df = pd.DataFrame(vif_vals, columns=['feature', 'VIF'])

    if add_intercept and 'const' in vif_df['feature'].values:
        vif_df = vif_df[vif_df['feature'] != 'const']

    return vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)


# ============================================================
# Single-pass pruning (> 60 features)
# ============================================================
def single_pass_prune(df, corr_thresh=0.98):
    X = df.to_numpy(dtype=float)
    cols = np.array(df.columns)
    n, p = X.shape

    # ---- stats
    var = X.var(axis=0)
    near_const = var < 1e-12

    # ---- correlation
    corr = np.corrcoef(X, rowvar=False)
    high_corr = np.abs(corr) >= corr_thresh
    np.fill_diagonal(high_corr, False)

    # ---- duplicates (column-wise hashing)
    X_round = np.round(X, 12)
    hashes = np.array([
        pd.util.hash_pandas_object(
            pd.Series(X_round[:, j]), index=False
        ).sum()
        for j in range(p)
    ])

    uniq, counts = np.unique(hashes, return_counts=True)
    dup_count_map = dict(zip(uniq, counts))

    # ---- score (higher = worse)
    score = (
        10.0 * near_const +
        5.0 * np.array([dup_count_map[h] - 1 for h in hashes]) +
        high_corr.sum(axis=0) +
        np.mean(np.abs(corr), axis=0)
    )

    # ---- prune: keep best in each correlated component
    to_drop = set()
    for j in range(p):
        if high_corr[j].any():
            group = np.where(high_corr[j])[0]
            worst = max([j, *group], key=lambda k: score[k])
            to_drop.add(cols[worst])

    # ---- drop duplicate columns (keep first)
    seen = {}
    for col, h in zip(cols, hashes):
        seen.setdefault(h, []).append(col)
    for g in seen.values():
        to_drop.update(g[1:])

    return sorted(to_drop)

# ============================================================
# Multi-pass pruning (≤ 60 features)
# ============================================================

def multipass_prune(df, corr_thresh=0.98, vif_thresh=30.0):
    drops = []

    # ---- duplicates
    for g in find_duplicate_columns(df):
        drops.extend(g[1:])

    df_work = df.drop(columns=drops, errors='ignore')

    # ---- correlations (cached)
    corr = df_work.corr(numeric_only=True)

    while True:
        vals = corr.values
        upper = np.triu(np.ones_like(vals, dtype=bool), k=1)
        mask = (np.abs(vals) >= corr_thresh) & upper

        if not mask.any():
            break

        mean_abs = corr.abs().mean()
        worst = mean_abs.idxmax()

        drops.append(worst)
        df_work = df_work.drop(columns=worst)
        corr = corr.drop(index=worst, columns=worst)

    # ---- VIF
    while df_work.shape[1] > 1:
        vif_df = compute_vif(df_work, add_intercept=True)
        if vif_df['VIF'].max() < vif_thresh:
            break

        worst = vif_df.iloc[0]['feature']
        drops.append(worst)
        df_work = df_work.drop(columns=worst)

    return drops


# ============================================================
# Unified interface
# ============================================================

def suggest_columns_to_drop(
    df,
    corr_thresh=0.98,
    vif_thresh=30.0,
    switch_at=60,
):
    if df.shape[1] > switch_at:
        return single_pass_prune(df, corr_thresh=corr_thresh)
    else:
        return multipass_prune(
            df,
            corr_thresh=corr_thresh,
            vif_thresh=vif_thresh,
        )


def check_design(binned_feats_sc, skip_vif=False, max_vif_cols=200):
    X = binned_feats_sc

    print('NaN/inf present?',
          ~np.isfinite(X.to_numpy(dtype=float)).all())

    near_const, _ = check_near_constant(X)
    print('Near-constant:', near_const)

    print('Duplicate column groups:',
          find_duplicate_columns(X))

    corr = X.corr(numeric_only=True)
    hits = np.where(
        np.triu(np.abs(corr.values) >= 0.98, k=1)
    )
    print('High-corr pairs:',
          [(corr.index[i], corr.columns[j],
            float(corr.iat[i, j]))
           for i, j in zip(*hits)])

    kappa, _ = condition_number(X)
    print('Condition number:', kappa)

    rank, p, deficient = rank_deficiency(X)
    print(f'rank={rank} of {p}; deficient? {deficient}')

    vif_report = None
    if skip_vif:
        print('VIF skipped (explicit)')
    elif X.shape[1] > max_vif_cols:
        print(f'VIF skipped (p={X.shape[1]} > {max_vif_cols})')
    else:
        vif_report = compute_vif(X)
        print(vif_report.head(20))

    to_drop = suggest_columns_to_drop(X)
    print('Suggested drops:', to_drop)

    return X.drop(columns=to_drop, errors='ignore'), vif_report


def load_or_compute_selected_cols(
    design_df,
    cols_path,
    *,
    skip_vif=False,
    exists_ok=True,
):
    """
    If exists_ok=True, try loading selected columns from cols_path.
    If loading fails or exists_ok=False, recompute and save.

    Parameters
    ----------
    exists_ok : bool
        Whether it is OK for the JSON file to already exist and be used.
    """

    if exists_ok:
        try:
            with open(cols_path, 'r') as f:
                selected_cols = json.load(f)
            X_pruned = design_df[selected_cols].copy()
            print('Loaded selected columns from file')
            return X_pruned, None
        except Exception:
            pass

    X_pruned, vif_report = check_design(design_df, skip_vif=skip_vif)

    os.makedirs(os.path.dirname(cols_path), exist_ok=True)
    with open(cols_path, 'w') as f:
        json.dump(X_pruned.columns.tolist(), f)

    print('Saved selected columns to file')
    return X_pruned, vif_report
