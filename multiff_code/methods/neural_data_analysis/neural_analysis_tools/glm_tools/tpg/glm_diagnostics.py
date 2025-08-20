from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
# -------- glm_design diagnostics --------


def _column_blocks(design_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group columns by prefix (before ``_rc`` if present)."""
    blocks: Dict[str, List[str]] = {}
    for c in design_df.columns:
        p = c.split('_rc')[0] if '_rc' in c else c
        blocks.setdefault(p, []).append(c)
    return blocks


def design_summary(design_df: pd.DataFrame, y: np.ndarray, *, topk: int = 10) -> pd.DataFrame:
    """Quick stats per column: variance, nonzero %, and |corr| with y (if defined)."""
    X = design_df.values
    var = X.var(axis=0)
    nz = (np.abs(X) > 0).mean(axis=0)
    y0 = (y - y.mean()) if y is not None else None
    cors = np.full(X.shape[1], np.nan)
    if y0 is not None and y0.std() > 0:
        for j in range(X.shape[1]):
            xj = X[:, j]
            if xj.std() > 0:
                cors[j] = np.corrcoef(xj, y0)[0, 1]
    df = pd.DataFrame({"col": design_df.columns, "var": var,
                      "nonzero_frac": nz, "corr_y": cors})
    df["abs_corr_y"] = np.abs(df["corr_y"])
    return df.sort_values("abs_corr_y", ascending=False).head(topk)


def block_summary(design_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Aggregate per block: ncols, mean var, max |corr| with y, zero-var count."""
    blocks = _column_blocks(design_df)
    rows = []
    X = design_df
    for p, cols in blocks.items():
        sub = X[cols]
        v = sub.var().mean()
        zero = int((sub.var() == 0).sum())
        mabs = np.nan
        if y is not None and y.std() > 0:
            cors = []
            for c in cols:
                xv = sub[c].values
                if xv.std() > 0:
                    cors.append(np.corrcoef(xv, y)[0, 1])
            mabs = float(np.nanmax(np.abs(cors))) if len(cors) else np.nan
        rows.append({"block": p, "ncols": len(cols), "mean_var": v,
                    "zero_var": zero, "max_abs_corr_y": mabs})
    return pd.DataFrame(rows).sort_values("max_abs_corr_y", ascending=False)


def constant_or_near_constant_columns(design_df: pd.DataFrame, tol: float = 1e-12) -> List[str]:
    """Return columns whose variance is ``<= tol`` (potentially redundant)."""
    v = design_df.var()
    return list(v.index[v <= tol])


def svd_report(design_df: pd.DataFrame, *, k: int = 20) -> Dict[str, object]:
    """Lightweight SVD diagnostics: rank, condition number, and top singular values."""
    X = design_df.values
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return {"rank": int((s > 1e-10).sum()), "ncols": X.shape[1], "nrows": X.shape[0], "cond_number": cond, "singular_values_top": s[:k]}


def check_param_mapping(result, design_df: pd.DataFrame) -> pd.DataFrame:
    """Return a mapping of parameter names to values after fitting.

    Useful to verify that column ordering matches intended kernel blocks.
    """
    names = getattr(result, 'exog_names', None)
    if names is None and hasattr(result.model, 'exog_names'):
        names = result.model.exog_names
    if names is None:
        names = ['const'] + list(design_df.columns)
    params = np.asarray(result.params).ravel()
    if len(params) == len(names):
        return pd.DataFrame({'name': names, 'param': params})
    return pd.DataFrame({'name': ['const'] + list(design_df.columns[:len(params)-1]), 'param': params})


def single_block_fit(prefix: str, design_df: pd.DataFrame, y: np.ndarray, dt: float, trial_ids: np.ndarray):
    """Fit a GLM using only the columns from a specific prefix block.

    Returns a dict with deviance and pseudo-R^2 vs the null model.
    """
    from glm_fit import fit_poisson_glm_trials, predict_mu, poisson_deviance
    cols = [c for c in design_df.columns if c.startswith(
        prefix + '_rc')] or [prefix]
    X = design_df[cols]
    res = fit_poisson_glm_trials(
        X, y, dt, trial_ids, add_const=True, l2=0.0, cluster_se=FalseTrue)
    mu = predict_mu(res, X, dt)
    dev = poisson_deviance(y, mu)
    null = poisson_deviance(y, np.full_like(y, y.mean()))
    return {"prefix": prefix, "deviance": dev, "null_dev": null, "pseudo_R2": 1 - dev/null}


def peth_from_onsets(onsets: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, *, window_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple PSTH around onsets within trials.

    Returns
    -------
    lags : ndarray of length ``2*window_bins+1`` (in *bins*, not seconds)
    mean_counts : ndarray of same length, averaged across all valid snippets
    """
    lags = np.arange(-window_bins, window_bins+1)
    snippets = []
    for tr in glm_bases._unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        on = np.where(onsets[idx] > 0)[0]
        for t in on:
            left = t - window_bins
            right = t + window_bins
            if left >= 0 and right < len(idx):
                seg = y[idx][left:right+1]
                snippets.append(seg)
    if not snippets:
        return lags, np.zeros_like(lags, dtype=float)
    M = np.vstack(snippets)
    return lags, M.mean(axis=0)


def debug_all_kernels_flat(res, design_df, y, trial_ids, meta, dt):
    """Run a compact suite of glm_design/model diagnostics and quick visual checks."""
    print("[DEBUG] Constant/near-constant columns:")
    print(constant_or_near_constant_columns(design_df)[:20])

    print("[DEBUG] Top-10 columns by |corr(y)|:")
    print(design_summary(design_df, y, topk=10))

    print("[DEBUG] Block summary (max |corr(y)| per block):")
    print(block_summary(design_df, y))

    print("[DEBUG] SVD report:")
    print(svd_report(design_df))

    print("[DEBUG] Param mapping (first 20):")
    pm = check_param_mapping(res, design_df)
    print(pm.head(20))

    # Quick single-block tests
    prefixes = sorted(
        set([c.split('_rc')[0] if '_rc' in c else c for c in design_df.columns]))
    tests = []
    for p in prefixes:
        tests.append(single_block_fit(p, design_df, y, dt, trial_ids))
    tests_df = pd.DataFrame(tests).sort_values('pseudo_R2', ascending=False)
    print("[DEBUG] Single-block fits (sorted by pseudo_R2):")
    print(tests_df)

    # Example PETH if cur_on is present (expects raw onset column)
    if 'cur_on' in design_df.columns:
        lags, mean_counts = peth_from_onsets(
            design_df['cur_on'].values, y, trial_ids, window_bins=40)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(lags * dt, mean_counts)
        plt.axvline(0, linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean spikes/bin')
        plt.title('PETH around cur_on onsets')
        plt.show()
