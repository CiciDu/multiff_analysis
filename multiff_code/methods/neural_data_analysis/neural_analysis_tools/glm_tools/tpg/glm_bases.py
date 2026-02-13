"""Utility functions and basis construction for trial-aware Poisson GLMs.

This module includes:
- Trial helpers (``_unique_trials``) that explicitly avoid cross-trial leakage.
- Time-basis construction via *causal* raised-cosine functions (unit area).
- Safe intensity mapping ``safe_poisson_lambda`` to cap Poisson rates.
- Angle helpers and trial-aware onset/offset detectors for binary masks.

glm_design principles
-----------------
1) **Causality.** Kernels are zero for negative lags; history uses *strictly past* bins.
2) **Trial isolation.** Convolutions and differencing restart at each trial, so no
   information leaks across trial boundaries.
3) **Numerical stability.** Unit-area basis columns; log/exp clamps; robust angle wrap.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.interpolate import BSpline

# -----------------
# Trial helpers
# -----------------


def _unique_trials(trial_ids: np.ndarray) -> np.ndarray:
    """Return unique trial labels as a NumPy array.

    Parameters
    ----------
    trial_ids : array-like of shape (T,)
        Trial identifier per time bin. Can be int or str; will be coerced to ndarray.

    Notes
    -----
    Using ``np.unique`` here is fine because we only need the set of unique labels.
    We keep this in its own function to centralize any future changes (e.g., ordering).
    """
    return np.unique(np.asarray(trial_ids))


import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import BSpline


def _normalize_columns(B: np.ndarray, dt: float, normalize: Optional[str]) -> np.ndarray:
    if normalize is None:
        return B

    B = B.copy()

    if normalize == 'area':
        col_sums = B.sum(axis=0) * dt + 1e-12
        B /= col_sums

    elif normalize == 'l2':
        norms = np.sqrt((B ** 2).sum(axis=0)) + 1e-12
        B /= norms

    elif normalize == 'peak':
        peaks = B.max(axis=0) + 1e-12
        B /= peaks

    else:
        raise ValueError("normalize must be None, 'area', 'l2', or 'peak'")

    return B


def raised_cosine_basis(
    n_basis: int,
    t_min: float,
    t_max: float,
    dt: float,
    *,
    normalize: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear-time raised cosine basis.

    Parameters
    ----------
    normalize : {None, 'area', 'l2', 'peak'}
        Column normalization type.
    """

    if t_max <= t_min:
        raise ValueError('t_max must be greater than t_min')

    lags = np.arange(t_min, t_max + 1e-12, dt)
    nbins = lags.size
    K = int(n_basis)

    if K < 1:
        return lags, np.zeros((nbins, 0))

    dbcenter = nbins / (3 + K)
    width = 4 * dbcenter
    centers = 2 * dbcenter + dbcenter * np.arange(K) - 1

    B = np.zeros((nbins, K))
    grid = np.arange(nbins)

    for k, c in enumerate(centers):
        x = grid - c
        mask = np.abs(x / width) < 0.5
        B[mask, k] = 0.5 * (np.cos(x[mask] * 2 * np.pi / width) + 1.0)

    B = _normalize_columns(B, dt, normalize)

    return lags, B


import numpy as np
from typing import Tuple


def raised_log_cosine_basis(
    n_bases: int,
    end_points: Tuple[float, float],
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact Python replication of the provided MATLAB log raised cosine basis code.

    Parameters
    ----------
    n_bases : int
        Number of basis functions.
    end_points : (float, float)
        Time range for basis placement (same as MATLAB `endPoints`).
    bin_size : float
        Time bin size (same as MATLAB `binSize`).

    Returns
    -------
    iht : (T,) array
        Time bins.
    ihbasis : (T, n_bases) array
        Basis matrix.
    ihctrs : (n_bases,) array
        Basis centers in original (unwarped) time.
    """

    # --- nonlinearity and its inverse (exact epsilon) ---
    eps = 1e-10
    nlin = lambda x: np.log(x + eps)
    invnl = lambda x: np.exp(x) - eps

    # --- warped range ---
    yrnge = nlin(np.array(end_points) + bin_size / 3.0)

    # spacing between raised cosine peaks
    db = (yrnge[1] - yrnge[0]) / (n_bases - 1)

    # centers in warped space
    ctrs = np.arange(yrnge[0], yrnge[1] + 1e-12, db)

    # extend maximum time so last basis decays to zero
    mxt = invnl(yrnge[1] + 2 * db) - bin_size / 3.0

    # time bins
    iht = np.arange(0.0, mxt + 1e-12, bin_size)

    # --- compute basis ---
    warped_time = nlin(iht + bin_size / 3.0)

    # replicate MATLAB repmat behavior
    X = np.tile(warped_time[:, None], (1, n_bases))
    C = np.tile(ctrs[None, :], (len(iht), 1))

    # cosine argument (exact MATLAB scaling)
    arg = (X - C) * np.pi / (2.0 * db)

    # hard clipping to [-pi, pi]
    arg = np.maximum(-np.pi, np.minimum(np.pi, arg))

    ihbasis = (np.cos(arg) + 1.0) / 2.0

    # centers in original time
    ihctrs = invnl(ctrs)

    return iht, ihbasis, ihctrs


def spline_basis(
    n_basis: int,
    t_max: float,
    dt: float,
    *,
    t_min: float = 0.0,
    degree: int = 3,
    log_spaced: bool = True,
    eps: float = 1e-3,
    normalize: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal clamped B-spline basis.

    Parameters
    ----------
    normalize : {None, 'area', 'l2', 'peak'}
        Column normalization type.
    """

    lags = np.arange(t_min, t_max + 1e-12, dt)
    L = lags.size
    K = int(n_basis)

    if K < 1:
        raise ValueError('n_basis must be >= 1')

    deg = int(min(max(0, degree), K - 1))

    def warp(x):
        return np.log(x + eps) if log_spaced else x

    W_min, W_max = warp(t_min), warp(t_max + 1e-12)

    M = max(0, K - deg - 1)

    if M > 0:
        internal = np.linspace(W_min, W_max, M + 2)[1:-1]
        knots = np.concatenate([
            np.full(deg + 1, W_min),
            internal,
            np.full(deg + 1, W_max)
        ])
    else:
        knots = np.concatenate([
            np.full(deg + 1, W_min),
            np.full(deg + 1, W_max)
        ])

    W_eval = warp(lags)
    B = np.empty((L, K))

    for k in range(K):
        coeff = np.zeros(K)
        coeff[k] = 1.0
        spline_k = BSpline(knots, coeff, deg, extrapolate=False)
        col = spline_k(W_eval)
        B[:, k] = np.nan_to_num(col, nan=0.0)

    B[B < 0] = 0.0

    B = _normalize_columns(B, dt, normalize)

    return lags, B

def safe_poisson_lambda(
    eta: float | np.ndarray,
    dt: float,
    *,
    max_rate_hz: float = 200.0,
) -> np.ndarray:
    """Map log-rate ``eta`` (units of *per second*) to Poisson mean per bin.

    The transformation ``exp(eta) * dt`` is *clipped* to avoid unstable simulation
    or optimization when rates become extreme.

    Parameters
    ----------
    eta : float or ndarray
        Log-rate (per-second).
    dt : float
        Bin width in seconds.
    max_rate_hz : float, default=200.0
        Maximum allowed firing rate in Hz before clipping (on the log scale).

    Returns
    -------
    lam : ndarray
        Expected counts in each bin (Poisson mean), with clipping applied.
    """
    log_min = np.log(1e-6)          # effectively zero rate
    log_max = np.log(max_rate_hz)    # hard cap for stability
    eta_clipped = np.clip(eta, log_min, log_max)
    rate_hz = np.exp(eta_clipped)
    return rate_hz * dt


# -----------------
# Angle helpers
# -----------------

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to ``(-pi, pi]`` for consistent trig computations."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_sin_cos(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(sin(theta), cos(theta))`` after wrapping to ``(-pi, pi]``."""
    th = wrap_angle(theta)
    return np.sin(th), np.cos(th)


# -----------------
# Onset/offset utilities (trial-aware)
# -----------------

def onset_from_mask_trials(mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Detect onsets (0→1 transitions) within each trial for a binary mask.

    Parameters
    ----------
    mask : array-like of shape (T,)
        Binary (0/1) vector marking event visibility/occurrence.
    trial_ids : array-like of shape (T,)
        Trial label per time bin. Transitions are computed *within* each trial.

    Returns
    -------
    on : ndarray of shape (T,)
        1.0 at time bins where a 0→1 transition occurs *within the same trial*;
        0.0 otherwise.
    """
    mask = (mask > 0).astype(int)
    on = np.zeros_like(mask, dtype=float)
    for tr in _unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        m = mask[idx]
        d = np.diff(np.r_[0, m])  # prepend 0 so t=0 can be an onset
        on[idx] = (d == 1).astype(float)
    return on


def offset_from_mask_trials(mask: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Detect offsets (1→0 transitions) within each trial for a binary mask.

    Same semantics as :func:`onset_from_mask_trials`, but for falling edges.
    """
    mask = (mask > 0).astype(int)
    off = np.zeros_like(mask, dtype=float)
    for tr in _unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        m = mask[idx]
        d = np.diff(np.r_[m, 0])  # append 0 so last bin can be an offset
        off[idx] = (d == 1).astype(float)
    return off
