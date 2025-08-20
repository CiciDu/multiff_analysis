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

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

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


# -----------------
# Basis construction
# -----------------

def raised_cosine_basis(
    n_basis: int,
    t_max: float,
    dt: float,
    *,
    t_min: float = 0.0,
    log_spaced: bool = True,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a *causal* raised-cosine basis that tiles ``[t_min, t_max]``.

    Each column integrates to 1 (unit area), so a weight directly scales the
    *average* contribution over its temporal support. This improves interpretability
    and keeps units consistent across different ``dt``.

    Parameters
    ----------
    n_basis : int
        Number of basis functions (columns).
    t_max : float
        Maximum lag in seconds included in the basis support.
    dt : float
        Bin width in seconds; determines temporal resolution.
    t_min : float, default=0.0
        Minimum lag in seconds for the basis support (\>= 0 for causality).
    log_spaced : bool, default=True
        If True, centers are uniformly spaced in a *log-warped* domain which
        concentrates resolution near 0 (useful for fast onsets and histories).
    eps : float, default=1e-3
        Small positive constant to avoid ``log(0)`` when ``log_spaced`` is True.

    Returns
    -------
    lags : ndarray of shape (L,)
        Nonnegative time lags in seconds.
    B : ndarray of shape (L, K)
        Basis matrix with K = ``n_basis`` columns (unit-area, nonnegative, causal).
    """
    # Discrete lag grid; include t_max by epsilon to account for floating error.
    lags = np.arange(0.0, t_max + 1e-12, dt)
    K = int(n_basis)

    def warp(x: np.ndarray | float):
        # Log-warp puts more centers near 0, capturing sharp transients.
        return np.log(x + eps) if log_spaced else x

    W = warp(lags)
    W_min, W_max = warp(t_min), warp(t_max)

    # Evenly space centers *in the warped domain*; width slightly overlaps for tiling.
    centers = np.linspace(W_min, W_max, K) if K > 0 else np.array([])
    delta = centers[1] - centers[0] if K > 1 else (W_max - W_min + 1e-12)
    width = delta * 1.5  # generous overlap to avoid gaps between adjacent bumps

    B_cols: List[np.ndarray] = []
    for c in centers:
        arg = (W - c) / width  # normalized displacement in warped domain
        # Raised cosine (clipped to [-pi, pi] so tails are exactly zero)
        bk = np.cos(np.clip(arg, -np.pi, np.pi))
        bk[np.abs(arg) > np.pi] = 0.0
        bk = np.maximum(bk, 0.0)
        # Enforce causality window [t_min, t_max]
        bk[lags < t_min] = 0.0
        # Normalize to unit area
        area = bk.sum() * dt
        if area > 0:
            bk = bk / area
        B_cols.append(bk)

    B = np.column_stack(B_cols) if B_cols else np.zeros((len(lags), 0))
    return lags, B


# -----------------
# Link / intensity utilities
# -----------------

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
