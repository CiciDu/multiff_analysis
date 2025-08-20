# =============================
# FILE: glm_fit_metrics.py
# =============================
"""Model fitting (statsmodels Poisson GLM) and common metrics/utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to a NumPy array glm_design."""
    return np.column_stack([np.ones(len(X)), X])


def fit_poisson_glm_trials(
    design_df: pd.DataFrame,
    y: np.ndarray,
    dt: float,
    trial_ids: np.ndarray,
    *,
    add_const: bool = True,
    l2: float = 0.0,
    cluster_se: bool = True,
):
    """Fit a Poisson GLM with optional L2 and cluster-robust SEs by trial.

    Parameters
    ----------
    design_df : DataFrame
        Model matrix with **column names** so coefficients map back to kernels.
    y : ndarray
        Spike counts per bin.
    dt : float
        Bin width (seconds). Supplied to ``exposure`` so rates are per second.
    trial_ids : ndarray
        For ``cov_type='cluster'`` grouping when ``cluster_se`` is True.
    add_const : bool, default=True
        If True, add an intercept column named ``'const'``.
    l2 : float, default=0.0
        If > 0, uses ``fit_regularized`` with L2 penalty (no SEs available).
    cluster_se : bool, default=True
        If True (and ``l2==0``), use cluster-robust covariance by trial.

    Returns
    -------
    result : statsmodels result object
        ``GLMResults`` when unpenalized; regularized results object otherwise.
    """
    X_df = design_df.copy()
    if add_const:
        X_df = sm.add_constant(X_df, has_constant='add')  # preserves names
    exposure = np.full_like(y, fill_value=dt, dtype=float)

    model = sm.GLM(y, X_df, family=sm.families.Poisson(), exposure=exposure)
    if l2 > 0:
        # Regularized fit (no covariance / SEs by default)
        res = model.fit_regularized(alpha=l2, L1_wt=0.0, maxiter=1000)
        return res
    else:
        if cluster_se:
            return model.fit(cov_type="cluster", cov_kwds={"groups": trial_ids})
        else:
            return model.fit()


def predict_mu(result, design_df: pd.DataFrame, dt: float, add_const: bool = True) -> np.ndarray:
    """Predict mean bin counts ``mu`` aligned to a fitted ``result``.

    Uses the same exposure ``dt`` as during fitting and ensures the intercept
    column is present if used.
    """
    X_df = design_df.copy()
    if add_const:
        X_df = sm.add_constant(X_df, has_constant='add')
    mu = result.predict(X_df, exposure=np.full(len(X_df), dt))
    return np.asarray(mu, dtype=float)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute the (twice) negative log-likelihood *deviance* for Poisson.

    Defined per bin as ``2 * [ y * log(y/mu) - (y - mu) ]`` with the convention
    that when ``y=0`` the first term is 0. Small epsilons guard ``log(0)``.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    eps = 1e-12
    term = np.where(y > 0, y * np.log((y + eps) / (mu + eps)), 0.0)
    dev = 2.0 * np.sum(term - (y - mu))
    return float(dev)


def pseudo_R2(y: np.ndarray, mu_full: np.ndarray, mu_null: np.ndarray) -> float:
    """McFadden pseudo-R^2 = ``1 - logL_full / logL_null``."""
    eps = 1e-12
    ll_full = np.sum(y * np.log(mu_full + eps) - mu_full)
    ll_null = np.sum(y * np.log(mu_null + eps) - mu_null)
    return float(1.0 - ll_full / ll_null)


def per_trial_deviance(y: np.ndarray, mu: np.ndarray, trial_ids: np.ndarray) -> pd.DataFrame:
    """Aggregate Poisson deviance per trial for diagnostics and CV.

    Returns a tidy DataFrame with columns ``['trial', 'trial_deviance']``.
    """
    dev = pd.DataFrame({"trial": trial_ids, "y": y, "mu": mu})
    eps = 1e-12
    dev["bin_dev"] = np.where(
        dev["y"] > 0,
        dev["y"] * np.log((dev["y"] + eps) / (dev["mu"] + eps)),
        0.0,
    ) - (dev["y"] - dev["mu"])
    out = dev.groupby("trial", as_index=False)["bin_dev"].sum()
    out.rename(columns={"bin_dev": "trial_deviance"}, inplace=True)
    out["trial_deviance"] *= 2.0
    return out
