# --- Imports (duplicates kept intentionally to preserve your original layout) ---
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from typing import Tuple, Iterable, List, Dict

# ======================================================================
# Public API: glm_tuning_curve
# ======================================================================

def glm_tuning_curve(model_res,
                     X_df: pd.DataFrame,
                     var: str,
                     grid: np.ndarray | None = None,
                     offset_log: np.ndarray | float | None = None,
                     average: str = 'marginal',
                     weights: np.ndarray | None = None,
                     return_ci: bool = True,
                     ci_level: float = 0.95) -> pd.DataFrame:
    """
    Compute a (partial dependence) tuning curve for one predictor `var` from a
    Poisson GLM with log link.

    The function supports both:
      - numeric predictors (including binary 0/1 dummies living directly in X_df), and
      - categorical predictors encoded as one-hot columns like 'var[T.level]'.

    It produces the *predicted firing rate (Hz)* as you sweep `var` over `grid`,
    averaging either:
      - 'marginal': over *rows* of X_df (optionally weighted by `weights`), or
      - 'at_means': at the mean of other covariates (linear on the design).

    If `return_ci=True` and the model covariance is available, standard errors
    and symmetric normal CIs are added via the delta method.

    Parameters
    ----------
    model_res : statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted GLM results object (Poisson, log link).
    X_df : pd.DataFrame
        The *exact* design frame used for fitting (same column set / order).
    var : str
        Name of the predictor to sweep. For categoricals, pass the *base prefix*
        before one-hots (e.g., 'cond' for 'cond[T.A]', 'cond[T.B]', ...).
    grid : array-like or None
        Values/levels to sweep. If None:
          - numeric   -> 50-point grid from 1st–99th percentile (or [0,1] if binary),
          - categorical -> discovered levels (+ '<base>' if no explicit base column).
    offset_log : array-like, scalar, or None
        Log(exposure) per row. If None, we try to recover offset/exposure from
        the fitted model; otherwise default to 0.
    average : {'marginal','at_means'}
        How to average other regressors while sweeping `var`.
    weights : array-like or None
        Optional per-row weights for the marginal average (e.g., exposure seconds).
    return_ci : bool
        If True, include SEs and CIs when covariance is available.
    ci_level : float
        Confidence level for the CI (e.g., 0.95).

    Returns
    -------
    pd.DataFrame
        Columns:
          - numeric:   [var, 'rate_hz'] (+ 'se_rate','ci_lo','ci_hi' optionally)
          - categorical: [var (level or '<base>'), 'rate_hz'] (+ SE/CI optionally)
    """
    # Align params to X_df columns and extract numeric arrays
    beta_s, X_df = _coerce_params_to_series(model_res, X_df)
    beta = beta_s.to_numpy()
    X = X_df.to_numpy()

    # Pull covariance if requested and available
    cov = _get_cov(model_res, X.shape[1]) if return_ci else None

    # Resolve scalar or per-row log-offset (or fallback to model/exposure/0)
    off = _resolve_offset(model_res, X_df, offset_log)

    # Decide whether `var` is numeric (incl. binary) or categorical one-hot family
    is_cat, grid, xcol, fam_cols = _build_grid(var, X_df, grid)

    # Sweep the grid and compute rates (+ SE) using the appropriate strategy
    rows = (_sweep_categorical(grid, fam_cols, X, X_df, beta, cov, off, average, weights, var)
            if is_cat else
            _sweep_numeric(grid, X, X_df, xcol, beta, cov, off, average, weights, var))

    # Assemble output frame and attach CIs if requested
    df = pd.DataFrame(rows)
    if return_ci:
        df = _attach_ci(df, ci_level)
    return df

# ======================================================================
# Model/Design alignment helpers
# ======================================================================

def _coerce_params_to_series(model_res, X_df):
    """
    Ensure we have a pandas Series of coefficients aligned to X_df's columns.
    - If statsmodels stored exog_names, prefer those.
    - If the model had an intercept but X_df lacks 'const', prepend it.
    - If names are unavailable, fall back to X_df columns (validated by length).
    """
    beta_arr = np.asarray(model_res.params).reshape(-1)

    exog_names = None
    if hasattr(model_res, 'model') and hasattr(model_res.model, 'exog_names'):
        exog_names = list(model_res.model.exog_names)

    # If model had an intercept, ensure X_df has it too
    if exog_names and 'const' in exog_names and 'const' not in X_df.columns:
        X_df = X_df.copy()
        X_df.insert(0, 'const', 1.0)

    # Decide which index to use for params
    if exog_names is not None and len(exog_names) == beta_arr.size:
        idx = exog_names
    elif X_df.shape[1] == beta_arr.size:
        idx = list(X_df.columns)
    else:
        # Mismatch: user likely passed a different design than was fit
        raise ValueError(
            f'Parameter length {beta_arr.size} does not match model.exog_names '
            f'({None if exog_names is None else len(exog_names)}) or X_df columns ({X_df.shape[1]}). '
            'Check that X_df has the same columns (and intercept) used for fitting.'
        )

    beta_s = pd.Series(beta_arr, index=idx, dtype=float)
    # Reindex just in case column order differs; unseen cols get 0
    beta_s = beta_s.reindex(X_df.columns, fill_value=0.0)
    return beta_s, X_df

def _is_binary_series(s: pd.Series) -> bool:
    """
    Return True if a Series is boolean or strictly {0,1} (ignoring NaNs).
    Used to decide that the default grid should be [0, 1] rather than a linspace.
    """
    if s.dtype == bool:
        return True
    u = pd.unique(s.dropna())
    return set(u.tolist()) <= {0, 1}

def _get_cov(model_res, p: int) -> np.ndarray | None:
    """
    Try to fetch the parameter covariance matrix; return None if missing or wrong shape.
    """
    try:
        cov = np.asarray(model_res.cov_params(), dtype=float)
        return cov if cov.shape == (p, p) else None
    except Exception:
        return None

def _resolve_offset(model_res, X_df: pd.DataFrame,
                    offset_log: np.ndarray | float | None) -> np.ndarray | float:
    """
    Resolve the log-offset:
      1) use user-provided `offset_log` if given (scalar or vector),
      2) else try model_res.model.offset,
      3) else try log(model_res.model.exposure),
      4) else default to scalar 0.0.
    """
    n = len(X_df)
    if offset_log is not None:
        off = np.asarray(offset_log, dtype=float)
        return float(off) if off.ndim == 0 else off
    try:
        off_model = getattr(model_res.model, 'offset', None)
        if off_model is not None:
            off = np.asarray(off_model, dtype=float)
            if off.size == n:
                return off
        exp_model = getattr(model_res.model, 'exposure', None)
        if exp_model is not None:
            exp = np.asarray(exp_model, dtype=float)
            if exp.size == n:
                return np.log(exp)
    except Exception:
        pass
    return 0.0

def _numeric_default_grid(s: pd.Series) -> np.ndarray:
    """
    Default sweep for numeric columns:
      - if binary/boolean -> [0, 1]
      - else 50-point grid from 1st–99th percentile, with fallbacks if degenerate
    """
    if _is_binary_series(s):
        return np.array([0, 1], dtype=int if s.dtype != bool else bool)
    x = s.to_numpy()
    finite = np.isfinite(x)
    lo, hi = np.percentile(x[finite], [1, 99]) if finite.any() else (np.nanmin(x), np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Fallbacks if data are constant/heavy-NaN
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(x.min()), float(x.max())
    return np.linspace(lo, hi, 50)

def _categorical_family(var: str, X_df: pd.DataFrame) -> Tuple[List[str], List[str], bool]:
    """
    Discover the one-hot columns for a categorical family with prefix `var`
    (columns named like 'var[T.level]'). Also report whether an explicit base
    column `var` exists.
    """
    fam_cols = [c for c in X_df.columns if c.startswith(var + '[T.')]
    if not fam_cols:
        raise ValueError(f'Variable "{var}" not found as numeric or one-hot.')
    levels = [c[len(var) + 3:-1] for c in fam_cols]  # strip 'var[T.' and trailing ']'
    base_present = any(c == var for c in X_df.columns)
    return levels, fam_cols, base_present

def _build_grid(var: str, X_df: pd.DataFrame, grid) -> Tuple[bool, np.ndarray, int | None, List[str] | None]:
    """
    Determine whether `var` is numeric or categorical, and build the appropriate grid.
    Returns:
      - is_categorical (bool),
      - grid (np.ndarray of values/levels),
      - xcol (int index into X for numeric vars; None if categorical),
      - fam_cols (list of dummy column names for categorical; None if numeric).
    """
    if var in X_df.columns:
        g = _numeric_default_grid(X_df[var]) if grid is None else np.asarray(grid)
        return False, g, X_df.columns.get_loc(var), None
    levels, fam_cols, base_present = _categorical_family(var, X_df)
    if grid is None:
        grid = np.array(levels if base_present else levels + ['<base>'], dtype=object)
    else:
        grid = np.asarray(grid, dtype=object)
    return True, grid, None, fam_cols

def _mean_row(X_df: pd.DataFrame) -> np.ndarray:
    """
    Column means of X_df as a numeric row vector (for 'at_means' predictions).
    """
    return X_df.mean(axis=0).to_numpy()

def _off_bar(off: np.ndarray | float) -> float:
    """
    Scalar summary of offset for 'at_means' predictions.
    If per-row, use the simple mean; if scalar already, return as float.
    """
    return float(np.mean(off)) if np.ndim(off) == 1 else float(off)

# ======================================================================
# Prediction cores (delta-method SEs)
# ======================================================================
def _avg_and_se(Xg: np.ndarray,
                beta: np.ndarray,
                cov: np.ndarray | None,
                off: np.ndarray | float,
                weights: np.ndarray | None) -> Tuple[float, float | None]:
    """
    Compute an average **rate (Hz)** over rows of Xg.

    Let μ_i = exp(off_i + x_i^T β) = expected count per bin i.
    Let t_i = exp(off_i) = bin duration (seconds).
    Then rate_i = μ_i / t_i = exp(x_i^T β).

    Averaging modes:
      - If `weights is None`: simple mean of per-row rates, r = (1/n) Σ rate_i.
      - If `weights` provided (expected to be exposure seconds per row): 
          time-weighted mean rate, r = (Σ μ_i) / (Σ weights) = (Σ t_i * exp(xβ)) / (Σ weights).

    Delta-method SE:
      - Unweighted mean:   ∂r/∂β = (1/n) Σ rate_i * x_i
      - Time-weighted mean (weights = exposures): ∂r/∂β = (1/Σw) Σ μ_i * x_i
    """
    # linear predictor and expected counts per bin
    if np.ndim(off):
        eta = off + Xg @ beta           # shape (n,)
        t = np.exp(off)                 # seconds per bin (vector)
    else:
        eta = float(off) + Xg @ beta
        t = np.exp(float(off))          # scalar seconds

    mu = np.exp(eta)                    # expected counts per bin
    # per-row rate in Hz
    rate_i = mu / t                     # equals exp(Xβ) when off varies per-row

    # choose averaging rule
    if weights is None:
        # simple mean of per-row rates
        r = float(rate_i.mean())
        agg_grad = (rate_i[:, None] * Xg).mean(axis=0) if cov is not None else None
    else:
        # time-weighted mean rate using weights (exposures) in seconds
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != Xg.shape[0]:
            raise ValueError('weights length must match X_df rows for marginal averaging.')
        W = w.sum()
        if W <= 0 or not np.isfinite(W):
            raise ValueError('weights must sum to a positive finite value.')
        r = float(mu.sum() / W)         # == (Σ t_i * exp(Xβ)) / Σ weights
        agg_grad = (mu[:, None] * Xg).sum(axis=0) / W if cov is not None else None

    if cov is None:
        return r, None

    var_r = float(agg_grad @ cov @ agg_grad)
    return r, np.sqrt(max(var_r, 0.0))


def _at_means_and_se(xmean: np.ndarray,
                     beta: np.ndarray,
                     cov: np.ndarray | None,
                     off_bar: float) -> Tuple[float, float | None]:
    """
    At-means **rate (Hz)**.

    μ_ref = exp(off_bar + x̄^T β);  t_ref = exp(off_bar)
    rate_at_means = μ_ref / t_ref = exp(x̄^T β)  (independent of off_bar)

    Delta-method SE:
      ∂r/∂β = rate_at_means * x̄
    """
    # rate in Hz does not depend on off_bar
    eta_no_off = float(xmean @ beta)
    r = float(np.exp(eta_no_off))
    if cov is None:
        return r, None
    grad = r * xmean
    var_r = float(grad @ cov @ grad)
    return r, np.sqrt(max(var_r, 0.0))

# ======================================================================
# Grid sweep strategies (numeric vs categorical)
# ======================================================================

def _sweep_numeric(grid: Iterable,
                   X: np.ndarray,
                   X_df: pd.DataFrame,
                   xcol: int,
                   beta: np.ndarray,
                   cov: np.ndarray | None,
                   off: np.ndarray | float,
                   average: str,
                   weights: np.ndarray | None,
                   var: str) -> List[Dict]:
    """
    Sweep numeric `var` over `grid`, predicting rate under the chosen averaging mode.
    """
    rows = []
    for g in np.atleast_1d(grid):
        if average == 'marginal':
            Xg = X.copy()       # change only the target column
            Xg[:, xcol] = g
            rate, se = _avg_and_se(Xg, beta, cov, off, weights)
        elif average == 'at_means':
            xm = _mean_row(X_df)  # plug everything else at column means
            xm[xcol] = g
            rate, se = _at_means_and_se(xm, beta, cov, _off_bar(off))
        else:
            raise ValueError('average must be "marginal" or "at_means"')
        rows.append({var: float(g), 'rate_hz': rate, 'se_rate': se})
    return rows

def _sweep_categorical(grid: Iterable,
                       fam_cols: List[str],
                       X: np.ndarray,
                       X_df: pd.DataFrame,
                       beta: np.ndarray,
                       cov: np.ndarray | None,
                       off: np.ndarray | float,
                       average: str,
                       weights: np.ndarray | None,
                       var: str) -> List[Dict]:
    """
    Sweep categorical `var` over its levels.
    For each g:
      - zero the whole one-hot family,
      - set the chosen level (unless g == '<base>'),
      - predict under the chosen averaging mode.
    """
    rows = []
    fam_idx = [X_df.columns.get_loc(c) for c in fam_cols]
    for g in np.atleast_1d(grid):
        if average == 'marginal':
            Xg = X.copy()
            Xg[:, fam_idx] = 0.0
            if g != '<base>':
                cname = f'{var}[T.{g}]'
                if cname not in X_df.columns:
                    raise ValueError(f'Level "{g}" not found among one-hots for {var}.')
                Xg[:, X_df.columns.get_loc(cname)] = 1.0
            rate, se = _avg_and_se(Xg, beta, cov, off, weights)
        elif average == 'at_means':
            xm = _mean_row(X_df)
            xm[fam_idx] = 0.0
            if g != '<base>':
                xm[X_df.columns.get_loc(f'{var}[T.{g}]')] = 1.0
            rate, se = _at_means_and_se(xm, beta, cov, _off_bar(off))
        else:
            raise ValueError('average must be "marginal" or "at_means"')
        rows.append({var: g, 'rate_hz': rate, 'se_rate': se})
    return rows

def _attach_ci(df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    """
    Attach symmetric normal-approximation confidence intervals using `se_rate`.
    If SEs are all missing/NaN, drop the column quietly.
    """
    from scipy.stats import norm
    z = float(norm.ppf(1 - (1 - ci_level) / 2))
    if 'se_rate' not in df or df['se_rate'].isna().all():
        return df.drop(columns=['se_rate'], errors='ignore')
    df = df.copy()
    df['ci_lo'] = df['rate_hz'] - z * df['se_rate']
    df['ci_hi'] = df['rate_hz'] + z * df['se_rate']
    return df

# ---------- thin public API ----------
# (glm_tuning_curve is already defined above)

# ======================================================================
# Empirical tuning (data-binned reference)
# ======================================================================

import numpy as np
import pandas as pd

def empirical_tuning_curve(binned_spikes,      # 1D array of counts for one unit
                           predictor_vals,     # 1D array aligned to bins
                           exposure_s,         # 1D array of bin durations in seconds
                           nbins=20,           # used only for continuous
                           bin_edges=None,
                           drop_empty=True):
    """
    Compute empirical tuning from binned data.

    Continuous predictor:
      - bin `x` (equal-width by default),
      - compute rate per bin as sum(spikes)/sum(time).

    Categorical/binary predictor:
      - compute one rate per unique level.

    Returns
    -------
    pd.DataFrame
      - Continuous: ['bin_left','bin_right','bin_center','n_bins','spikes','time_s','rate_hz']
      - Categorical: ['level','n_bins','spikes','time_s','rate_hz']
    """
    y = np.asarray(binned_spikes).reshape(-1)
    x = np.asarray(predictor_vals).reshape(-1)
    t = np.asarray(exposure_s).reshape(-1)

    # Basic alignment/validity checks
    assert y.size == x.size == t.size, 'Inputs must be aligned per bin.'
    if np.any(t <= 0):
        raise ValueError('exposure_s must be > 0 for all bins.')

    # Drop NaN/inf rows consistently across all arrays
    m_valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(t)
    y, x, t = y[m_valid], x[m_valid], t[m_valid]

    # Heuristic: treat as categorical if small number of unique values or object/bool dtype
    unique_vals = pd.unique(x)
    is_categorical = (x.dtype.kind in 'Ob' or unique_vals.size <= 6)

    if is_categorical:
        # ----- Categorical path -----
        rows = []
        for lvl in np.sort(unique_vals):
            mask = (x == lvl)
            if drop_empty and mask.sum() == 0:
                continue
            spikes = int(y[mask].sum())
            time_s = float(t[mask].sum())
            rate = spikes / time_s if time_s > 0 else np.nan
            rows.append({
                'level': lvl,
                'n_bins': int(mask.sum()),
                'spikes': spikes,
                'time_s': time_s,
                'rate_hz': float(rate),
            })
        return pd.DataFrame(rows)
    else:
        # ----- Continuous path -----
        if bin_edges is None:
            # Robust to heavy tails: 1st–99th percentiles; widen edges a touch
            lo, hi = np.nanpercentile(x, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = np.nanmin(x), np.nanmax(x)
            if hi <= lo:
                # Degenerate case: all x nearly equal → single tiny bin
                bin_edges = np.array([lo, hi if hi > lo else lo + 1e-12])
            else:
                bin_edges = np.linspace(lo, hi, nbins + 1)

            # Expand extremes so boundary values are captured
            lo = np.nextafter(bin_edges[0], -np.inf)
            hi = np.nextafter(bin_edges[-1], np.inf)
            bin_edges = bin_edges.copy()
            bin_edges[0] = lo
            bin_edges[-1] = hi

        # Assign each observation to a bin
        idx = np.digitize(x, bin_edges) - 1
        k = len(bin_edges) - 1

        rows = []
        for b in range(k):
            mask = (idx == b)
            if drop_empty and mask.sum() == 0:
                continue
            spikes = int(y[mask].sum())
            time_s = float(t[mask].sum())
            rate = spikes / time_s if time_s > 0 else np.nan
            rows.append({
                'bin_left': bin_edges[b],
                'bin_right': bin_edges[b+1],
                'bin_center': 0.5*(bin_edges[b] + bin_edges[b+1]),
                'n_bins': int(mask.sum()),
                'spikes': spikes,
                'time_s': time_s,
                'rate_hz': float(rate),
            })
        return pd.DataFrame(rows)

# ======================================================================
# Plotting helpers for tuning curves
# ======================================================================

def plot_tuning_curve(df, xcol, ycol='rate_hz', title=None):
    """
    Simple line plot of a single tuning curve DataFrame (continuous or ordinal x).
    """
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(df[xcol], df[ycol], marker='o')
    plt.xlabel(xcol)
    plt.ylabel('rate (Hz)')
    if title:
        plt.title(title)
    plt.tight_layout()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _extract_x_y(df, xcol_hint, ycol):
    """
    Normalize (x, y) for plotting from either continuous- or categorical-style frames.

    Returns
    -------
    x : np.ndarray
        x-values to plot (bin centers or categorical levels mapped to values)
    y : np.ndarray
        y-values to plot
    is_categorical : bool
        Whether x represents unordered levels (affects tick handling in overlay)
    """
    # Prefer explicit continuous structure if available
    if 'bin_center' in df.columns:
        x = df['bin_center'].to_numpy()
        y = df[ycol].to_numpy()
        return x, y, False

    # Categorical path: expect a 'level' column
    if 'level' in df.columns:
        # Keep stable order: numeric levels sorted; strings preserve row order
        x_levels = df['level']
        if np.issubdtype(x_levels.dtype, np.number):
            order = np.argsort(x_levels.to_numpy())
        else:
            order = np.arange(len(x_levels))
        x = x_levels.to_numpy()[order]
        y = df[ycol].to_numpy()[order]
        return x, y, True

    # Fallback: use provided xcol_hint, infer categorical by few uniques or object dtype
    if xcol_hint in df.columns:
        x = df[xcol_hint].to_numpy()
        y = df[ycol].to_numpy()
        is_cat = pd.unique(x).size <= 6 and (df[xcol_hint].dtype.kind in 'Obifc')
        return x, y, is_cat

    raise KeyError(f'Could not find x in columns: {list(df.columns)}')

def overlay_tuning_curves(emp_df,
                          model_df,
                          xcol,
                          ycol='rate_hz',
                          title=None,
                          label_emp='empirical',
                          label_model='GLM'):
    """
    Overlay empirical (binned) and model-predicted tuning curves.
    Handles both:
      - continuous curves (line vs line), and
      - categorical curves (discrete ticks with matched level order).
    """
    x_e, y_e, cat_e = _extract_x_y(emp_df, xcol, ycol)
    x_m, y_m, cat_m = _extract_x_y(model_df, xcol, ycol)

    plt.figure(figsize=(4,3))

    # Categorical overlay: map levels to integer ticks and align both series
    if cat_e or cat_m:
        levels = np.unique(np.concatenate([np.atleast_1d(x_e), np.atleast_1d(x_m)]))
        level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        xe = np.array([level_to_idx[l] for l in x_e])
        xm = np.array([level_to_idx[l] for l in x_m])

        plt.plot(xe, y_e, 'o-', label=label_emp)
        plt.plot(xm, y_m, '-', lw=2, label=label_model)

        # Human-friendly categorical ticks
        plt.xticks(range(len(levels)), [str(l) for l in levels])
        plt.xlabel(xcol)
    else:
        # Continuous overlay: sort for monotone x progression
        se = np.argsort(x_e)
        sm = np.argsort(x_m)
        plt.plot(x_e[se], y_e[se], 'o-', label=label_emp)
        plt.plot(x_m[sm], y_m[sm], '-', lw=2, label=label_model)
        plt.xlabel(xcol)

    plt.ylabel('rate (Hz)')
    if title:
        plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_tuning_with_ci(df: pd.DataFrame,
                        xcol: str,
                        ycol: str = 'rate_hz',
                        ci_lo: str = 'ci_lo',
                        ci_hi: str = 'ci_hi',
                        title: str | None = None,
                        ax=None):
    """
    Plot one tuning curve with an optional shaded CI band.

    Accepts DataFrames produced by `glm_tuning_curve` (numeric or categorical)
    or by your empirical function (if it contains matching columns).

    Parameters
    ----------
    df : DataFrame
        Must contain either:
          - continuous: 'bin_center' (preferred) or a numeric `xcol`
          - categorical: 'level' or a discrete `xcol`
        And a y column (default 'rate_hz').
        CI columns are optional; provide names via `ci_lo`, `ci_hi`.
    xcol : str
        Name of x column to use when neither 'bin_center' nor 'level' is present.
    ycol : str
        Name of y column (default 'rate_hz').
    ci_lo, ci_hi : str
        Column names for CI bounds, if present.
    title : str | None
        Figure title.
    ax : matplotlib Axes | None
        Existing axes to draw on; creates a new one if None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Decide x source and categorical status
    if 'bin_center' in df.columns:
        x = df['bin_center'].to_numpy()
        y = df[ycol].to_numpy()
        order = np.argsort(x)
        x, y = x[order], y[order]
        has_ci = (ci_lo in df.columns) and (ci_hi in df.columns)
        if has_ci:
            lo = df[ci_lo].to_numpy()[order]
            hi = df[ci_hi].to_numpy()[order]
        # plot
        ax.plot(x, y, marker='o')
        if has_ci and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xlabel('bin_center')
    elif 'level' in df.columns:
        # treat as categorical; map to integer ticks
        levels = df['level'].to_numpy()
        y = df[ycol].to_numpy()
        # keep numeric levels sorted; otherwise preserve row order
        if np.issubdtype(df['level'].dtype, np.number):
            order = np.argsort(levels)
            levels, y = levels[order], y[order]
            df_ord = df.iloc[order]
        else:
            df_ord = df
        idx = np.arange(len(df_ord))
        ax.plot(idx, df_ord[ycol].to_numpy(), marker='o')
        has_ci = (ci_lo in df_ord.columns) and (ci_hi in df_ord.columns)
        if has_ci:
            lo = df_ord[ci_lo].to_numpy()
            hi = df_ord[ci_hi].to_numpy()
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                ax.fill_between(idx, lo, hi, alpha=0.2)
        ax.set_xticks(idx)
        ax.set_xticklabels([str(v) for v in df_ord['level'].to_numpy()])
        ax.set_xlabel('level')
    else:
        # fallback to provided xcol—decide if categorical by few uniques
        x = df[xcol].to_numpy()
        y = df[ycol].to_numpy()
        uniques = pd.unique(x)
        is_categorical = (df[xcol].dtype.kind in 'Ob') or (uniques.size <= 6)
        if is_categorical:
            levels = np.asarray(uniques)
            # build index order matching first occurrence of each level
            level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
            idx = np.array([level_to_idx[v] for v in x])
            order = np.argsort(idx)
            idx, y = idx[order], y[order]
            lo = df.get(ci_lo, pd.Series(index=df.index, dtype=float)).to_numpy()
            hi = df.get(ci_hi, pd.Series(index=df.index, dtype=float)).to_numpy()
            lo = lo[order] if lo.size else None
            hi = hi[order] if hi.size else None

            ax.plot(idx, y, marker='o')
            if lo is not None and hi is not None and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                ax.fill_between(idx, lo, hi, alpha=0.2)
            ax.set_xticks(np.arange(len(levels)))
            ax.set_xticklabels([str(v) for v in levels])
        else:
            order = np.argsort(x)
            x, y = x[order], y[order]
            ax.plot(x, y, marker='o')
            if (ci_lo in df.columns) and (ci_hi in df.columns):
                lo = df[ci_lo].to_numpy()[order]
                hi = df[ci_hi].to_numpy()[order]
                if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                    ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xlabel(xcol)

    ax.set_ylabel('rate (Hz)')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
