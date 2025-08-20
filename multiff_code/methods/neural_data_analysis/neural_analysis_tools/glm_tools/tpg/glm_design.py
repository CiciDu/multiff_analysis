
# =============================
# FILE: design.py
# =============================
"""Design-matrix builders that *respect trial boundaries*.

Key routines:
- ``lagged_design_from_signal_trials``: convolve a scalar regressor with a
  basis within each trial (causal, no leakage across trials).
- ``spike_history_design_with_trials``: build a strictly *past* spike-history
  design using a Toeplitz trick, restarted per trial.
- ``build_glm_design_with_trials``: orchestrate stimulus, history, and extras
  into a single ``pandas.DataFrame`` ready for ``statsmodels.GLM``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import toeplitz

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases


def lagged_design_from_signal_trials(x: np.ndarray, basis: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Convolve ``x`` with each basis column *within each trial*.

    Parameters
    ----------
    x : ndarray of shape (T,)
        Input scalar regressor (e.g., event onsets, gated distance).
    basis : ndarray of shape (L, K)
        Causal basis matrix; each column represents a kernel shape.
    trial_ids : ndarray of shape (T,)
        Trial IDs; convolution restarts at the first bin of each trial.

    Returns
    -------
    Xk : ndarray of shape (T, K)
        Trial-aware, causal lagged design (no future leakage by construction).
    """
    T = len(x)
    L, K = basis.shape
    Xk = np.zeros((T, K))
    for tr in glm_bases._unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        xt = x[idx]
        # Causal convolution per basis column, truncated to trial length.
        for k in range(K):
            y = signal.convolve(xt, basis[:, k], mode="full")[: len(idx)]
            Xk[idx, k] = y
    return Xk


def spike_history_design_with_trials(y_counts: np.ndarray, basis: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Build *strictly past* spike-history design per trial using a Toeplitz map.

    For a history basis ``B`` (L×K), we want for each time ``t`` the vector
    ``[y_{t-1}, y_{t-2}, ..., y_{t-L}]`` multiplied by ``B``. The Toeplitz
    construction gives an efficient linear operator for all lags at once.

    Parameters
    ----------
    y_counts : ndarray of shape (T,)
        Observed spike counts per bin.
    basis : ndarray of shape (L, K)
        Causal basis (no support at lag 0 for history to ensure "strictly past").
    trial_ids : ndarray of shape (T,)
        Trial IDs; the history buffer is cleared at the start of each trial.

    Returns
    -------
    Xh : ndarray of shape (T, K)
        Spike-history design. Row ``t`` uses only ``y`` from the *same trial*
        and strictly earlier bins.
    """
    T = len(y_counts)
    L, K = basis.shape
    Xh = np.zeros((T, K))
    for tr in glm_bases._unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        y_tr = y_counts[idx]
        # First column is y_{t-1}, so prepend 0 to enforce "strictly past".
        col0 = np.r_[0, y_tr[:-1]]
        toepl = toeplitz(col0, np.zeros(L))  # (T_tr × L)
        Xh[idx, :] = toepl @ basis
    return Xh


def build_glm_design_with_trials(
    dt: float,
    trial_ids: np.ndarray,
    stimulus_dict: Optional[Dict[str, np.ndarray]] = None,
    stimulus_basis_dict: Optional[Dict[str, np.ndarray]] = None,
    spike_counts: Optional[np.ndarray] = None,
    history_basis: Optional[np.ndarray] = None,
    extra_covariates: Optional[Dict[str, np.ndarray]] = None,
    use_trial_FE: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Combine stimulus, history, and extra covariates into a DataFrame design.

    "Trial fixed effects" can be added as one-hot columns with ``use_trial_FE``.

    Parameters
    ----------
    dt : float
        Bin width (seconds). Included for completeness and future checks.
    trial_ids : ndarray of shape (T,)
        Trial index per bin.
    stimulus_dict : dict[str, ndarray], optional
        Map from *stimulus name* → raw regressor (length T). If a name appears
        in ``stimulus_basis_dict``, a lagged design is constructed using that basis.
        Otherwise the raw column is included.
    stimulus_basis_dict : dict[str, ndarray], optional
        Map stimulus name → basis (L×K). Only names present here are expanded.
    spike_counts : ndarray, optional
        Spike counts; required if ``history_basis`` is provided.
    history_basis : ndarray, optional
        Basis for spike-history; if provided with ``spike_counts``, adds columns
        ``hist_rc1, hist_rc2, ...``.
    extra_covariates : dict[str, ndarray], optional
        Additional columns (no lagging) such as speed/curvature.
    use_trial_FE : bool, default=False
        Whether to append one-hot trial indicators (drop_first=True to avoid
        perfect collinearity with intercept).

    Returns
    -------
    design_df : pandas.DataFrame
        Model matrix with named columns, suitable for ``statsmodels``.
    y : ndarray or None
        ``spike_counts`` if provided, else ``None``.
    """
    T = len(trial_ids)
    cols: List[np.ndarray] = []
    names: List[str] = []

    # Stimuli (possibly lagged via basis)
    if stimulus_dict is not None:
        for name, x in stimulus_dict.items():
            if stimulus_basis_dict is not None and name in stimulus_basis_dict:
                B = stimulus_basis_dict[name]
                Xk = lagged_design_from_signal_trials(x, B, trial_ids)
                for k in range(Xk.shape[1]):
                    cols.append(Xk[:, k])
                    names.append(f"{name}_rc{k+1}")
            else:
                cols.append(x)
                names.append(name)

    # Spike history
    if spike_counts is not None and history_basis is not None:
        Xh = spike_history_design_with_trials(
            spike_counts, history_basis, trial_ids)
        for k in range(Xh.shape[1]):
            cols.append(Xh[:, k])
            names.append(f"hist_rc{k+1}")

    # Extras
    if extra_covariates is not None:
        for n, v in extra_covariates.items():
            cols.append(v)
            names.append(n)

    X = np.column_stack(cols) if cols else np.zeros((T, 0))
    design_df = pd.DataFrame(X, columns=names)

    # Optionally add trial fixed effects (drop_first avoids dummy trap with intercept)
    if use_trial_FE:
        trial_FE = pd.get_dummies(trial_ids, prefix="trial", drop_first=True)
        design_df = pd.concat([design_df, trial_FE], axis=1)

    y = spike_counts if spike_counts is not None else None
    return design_df, y
