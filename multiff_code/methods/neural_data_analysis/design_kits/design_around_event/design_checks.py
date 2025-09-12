import numpy as np
import pandas as pd
from numpy.linalg import svd, matrix_rank
from statsmodels.tools.tools import add_constant

def check_near_constant(df, tol=1e-12):
    variances = df.var(axis=0).astype(float)
    near_const = variances[variances <= tol]
    return near_const.index.tolist(), variances

def pairwise_high_corr(df, thresh=0.98):
    # returns list of (col_i, col_j, r)
    corr = df.corr(numeric_only=True)
    hits = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= thresh:
                hits.append((cols[i], cols[j], float(r)))
    return hits, corr

def find_duplicate_columns(df):
    # exact duplicates
    dup_groups = {}
    seen = {}
    for c in df.columns:
        key = tuple(np.round(df[c].to_numpy(dtype=float), 12))  # robust-ish float key
        dup_groups.setdefault(key, []).append(c)
    duplicates = [cols for cols in dup_groups.values() if len(cols) > 1]
    return duplicates

def condition_number(df):
    x = df.to_numpy(dtype=float)
    u, s, vt = svd(x, full_matrices=False)
    if s.min() == 0:
        return np.inf, s
    return float(s.max() / s.min()), s

def rank_deficiency(df, tol=1e-10):
    x = df.to_numpy(dtype=float)
    r = matrix_rank(x, tol=tol)
    return r, df.shape[1], r < df.shape[1]

def compute_vif(df, add_intercept=False):
    """
    Returns a DataFrame with VIF per column (excluding the explicit constant if added).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df.copy()
    if add_intercept:
        X = add_constant(X, has_constant='add')
    Xv = X.to_numpy(dtype=float)
    cols = list(X.columns)
    vif_vals = []
    for i, col in enumerate(cols):
        vif = variance_inflation_factor(Xv, i)
        vif_vals.append((col, float(vif)))
    vif_df = pd.DataFrame(vif_vals, columns=['feature', 'VIF'])
    # If we added a constant, drop it from the report
    if add_intercept and 'const' in vif_df['feature'].values:
        vif_df = vif_df[vif_df['feature'] != 'const'].reset_index(drop=True)
    return vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)

def suggest_columns_to_drop(df, corr_thresh=0.98, vif_thresh=30.0):
    """
    Heuristic:
      1) Drop exact duplicates (keep first)
      2) For high-corr pairs, drop the one with higher mean |corr| to others
      3) If VIF still extreme, drop the worst-offender one-by-one
    Returns: ordered list of suggested drops
    """
    drops = []

    # 1) duplicates
    dups = find_duplicate_columns(df)
    for group in dups:
        # keep the first, drop the rest
        drops.extend(group[1:])

    df_work = df.drop(columns=set(drops), errors='ignore')

    # 2) high correlations
    hits, corr = pairwise_high_corr(df_work, thresh=corr_thresh)
    # Greedy removal based on average absolute correlation
    while hits:
        cols = list(df_work.columns)
        mean_abs = corr.abs().mean().reindex(cols)
        # pick the feature with highest overall correlation to others
        worst = mean_abs.idxmax()
        drops.append(worst)
        df_work = df_work.drop(columns=[worst])
        hits, corr = pairwise_high_corr(df_work, thresh=corr_thresh)

    # 3) VIF pruning
    # iterate until all VIFs below threshold or only 1 col remains
    while df_work.shape[1] > 1:
        vif_df = compute_vif(df_work, add_intercept=True)
        if vif_df['VIF'].max() < vif_thresh:
            break
        worst = vif_df.iloc[0]['feature']
        drops.append(worst)
        df_work = df_work.drop(columns=[worst])

    return drops


def check_design(binned_feats_sc):
    # 0) quick sanity: NaN/inf
    bad_any = ~np.isfinite(binned_feats_sc.to_numpy(dtype=float)).all()
    print('NaN/inf present?', bad_any)

    # 1) near-constant columns
    near_const_cols, variances = check_near_constant(binned_feats_sc, tol=1e-12)
    print('Near-constant:', near_const_cols)

    # 2) duplicates
    dups = find_duplicate_columns(binned_feats_sc)
    print('Duplicate column groups:', dups)

    # 3) correlation spikes
    hits, corr = pairwise_high_corr(binned_feats_sc, thresh=0.98)
    print('High-corr pairs (|r| >= 0.98):', hits)

    # 4) condition number
    kappa, svals = condition_number(binned_feats_sc)
    print('Condition number:', kappa)

    # 5) rank deficiency
    rank, p, is_deficient = rank_deficiency(binned_feats_sc)
    print(f'rank={rank} of {p} columns; deficient? {is_deficient}')

    # 6) VIF
    vif_report = compute_vif(binned_feats_sc)
    print(vif_report.head(10))

    # 7) Suggested columns to drop
    to_drop = suggest_columns_to_drop(binned_feats_sc, corr_thresh=0.98, vif_thresh=30.0)
    print('Suggested drops (order matters):', to_drop)

    # Optional: apply the drop list
    X_pruned = binned_feats_sc.drop(columns=to_drop, errors='ignore')
    return X_pruned