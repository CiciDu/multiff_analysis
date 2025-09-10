# =============================
# FILE: glm_fit.py
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
    """Fit Poisson GLM keeping column names; robustly coerce inputs to numeric.

    - Coerces design to numeric float (object→NaN→0.0).
    - Coerces y to float.
    - Replaces inf with NaN then 0.0 in design to avoid statsmodels object-cast errors.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # --- coerce design to numeric float ---
    X_df = design_df.copy()
    # Convert any non-numeric entries to NaN, then fill with 0.0
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    # Replace inf/-inf just in case
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # optional: sanity check for remaining object dtypes (should be none)
    # assert not any(X_df.dtypes == "object"), X_df.dtypes[X_df.dtypes=="object"]

    if add_const:
        X_df = sm.add_constant(X_df, has_constant="add")

    # --- coerce y to numeric float ---
    y = np.asarray(y).astype(float)
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN/Inf after coercion; please clean spike_counts.")

    # exposure must be a float scalar per row
    dt = float(np.asarray(dt).ravel()[0])  # allow scalar-like arrays/Series
    exposure = np.full_like(y, fill_value=dt, dtype=float)

    model = sm.GLM(y, X_df, family=sm.families.Poisson(), exposure=exposure)

    if l2 > 0:
        res = model.fit_regularized(alpha=l2, L1_wt=0.0, maxiter=1000)
        return res
    else:
        if cluster_se:
            return model.fit(cov_type="cluster", cov_kwds={"groups": trial_ids})
        else:
            return model.fit()


def predict_mu(result, design_df: pd.DataFrame, dt: float, add_const: bool = True) -> np.ndarray:
    """Predict mean bin counts (mu) aligned with a fitted statsmodels GLM result.

    - Coerces design to numeric float (object→NaN→0.0)
    - Coerces dt to a scalar float (accepts Series/array if constant)
    - Re-adds intercept if needed to match the fitted model's columns
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # --- coerce dt to a scalar float ---
    def _coerce_dt_scalar(dt_val):
        if np.isscalar(dt_val):
            return float(dt_val)
        arr = np.asarray(dt_val).ravel()
        if arr.size == 1:
            return float(arr[0])
        if np.allclose(arr, arr[0]):
            return float(arr[0])
        raise ValueError(f"dt must be a scalar or constant array; got min={arr.min()}, max={arr.max()}")
    dt = _coerce_dt_scalar(dt)

    # --- coerce design to numeric float ---
    X_df = design_df.copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")           # non-numeric -> NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # sanitize
    X_df = X_df.astype(float)

    # --- align columns & add const if needed to match the fitted model ---
    if add_const:
        X_df = sm.add_constant(X_df, has_constant="add")

    # Ensure we only pass the columns the model expects (and in the same order)
    model_names = list(getattr(result.model, "exog_names", [])) or list(getattr(result, "exog_names", []))
    if model_names:
        # Some results include 'const' in exog_names; select strictly by that order
        missing = [c for c in model_names if c not in X_df.columns]
        if missing:
            # If some FE columns aren't present at prediction time, create zeros for them
            for c in missing:
                X_df[c] = 0.0
        X_df = X_df[model_names]

    # --- predict with stable numeric exposure ---
    exposure = np.full(X_df.shape[0], dt, dtype=float)
    mu = result.predict(X_df, exposure=exposure)
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
