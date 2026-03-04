"""
Parameters for stop-aligned GAM / PAM analyses.

This mirrors the structure of `one_ff_parameters.Params` but is scoped to
stop-event encoding:

- Default bin width and temporal window around each stop.
- Default numbers of basis functions for temporal kernels and tuning.
- Bin ranges (`binrange`) for kinematic and firefly-related variables, so
  that tuning designs (e.g. `build_tuning_design_for_continuous_vars`) are consistent
  across sessions and matched to the one_ff configuration.
- Default penalties (`lam_f`, `lam_g`, `lam_h`, `lam_p`) to keep the
  stop_GAM regularization in the same order as one_ff_GAM.
"""

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class StopParams:
    """
    Parameter bundle for stop-aligned GAM / PAM.

    Notes
    -----
    - `bin_width` should match the binning used when assembling the stop design
      (e.g. `bin_width` argument to `StopEncodingRunner` and
      `build_stop_encoding_design`).
    - `binrange` is passed as `binrange_dict` to `build_tuning_design_for_continuous_vars`
      so that tuning boxcars for v/w/d/phi/... match one_ff.
    - `lam_*` match the semantics in `one_ff_gam.build_group_specs`:
        lam_f: tuning (1D curves over continuous covariates)
        lam_g: temporal/event kernels (multi-column raised-cosine basis)
        lam_h: self spike-history kernel
        lam_p: coupling kernels
    """

    # =========================
    # Sampling & timing
    # =========================
    bin_width: float = 0.04          # seconds, typical stop-aligned binning
    dt: float = 0.04                 # alias for temporal bin width
    pre_event: float = 0.3           # seconds before stop
    post_event: float = 0.3          # seconds after stop

    # =========================
    # GAM configuration
    # =========================
    default_n_basis: int = 20        # raised-cosine basis functions for time
    tuning_n_bins: int = 10          # boxcar bins per tuning covariate
    # raw_only | boxcar_only | raw_plus_boxcar
    tuning_feature_mode: str = 'boxcar_only'

    # Penalties (match one_ff_gam defaults in scale)
    lam_f: float = 100.0             # tuning curves
    lam_g: float = 10.0              # temporal event kernels
    lam_h: float = 10.0              # spike history
    lam_p: float = 10.0              # coupling

    # =========================
    # Bin ranges for tuning / temporal
    # =========================
    binrange: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            # Kinematics / integrated motion (copied from one_ff_parameters)
            "v": np.array([0, 200]),            # cm/s
            "w": np.array([-90, 90]),           # deg/s
            # "d": np.array([0, 400]),            # cm
            "phi": np.array([-180, 180]),         # deg
            "accel": np.array([-1000, 1000]),  # cm/s^2
            "ang_accel_deg": np.array([-600, 600]),  # rad/s^2
            # Target in egocentric polar coordinates
            "r_targ": np.array([0, 400]),       # cm
            "theta_targ": np.array([-60, 60]),  # deg
            # Eye position
            "eye_ver": np.array([-25, 0]),      # deg
            "eye_hor": np.array([-40, 40]),     # deg
            # Temporal windows (stop-aligned); kept symmetric for simplicity
            "t_stop": np.array([-0.36, 0.36]),
            # Spike history (causal window, copied from one_ff_parameters)
            "spike_hist": np.array([0.006, 0.246]),
        }
    )


def estimate_stop_binrange_from_binned_feats(
    binned_feats,
    *,
    vars_to_include: Sequence[str] = (
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor',
    ),
    percentiles: Tuple[float, float] = (1.0, 99.0),
    pad_frac: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Estimate a reasonable `binrange_dict` from an existing stop design DataFrame.

    Uses robust percentiles (default 1–99%) and applies simple constraints:
    - nonnegative vars: lower bound clipped to 0
    - signed vars: symmetric bounds around 0 using max(|low|, |high|)

    Returns a dict var -> np.array([min, max]) for vars present in binned_feats.
    """
    import pandas as pd  # type: ignore

    if not isinstance(binned_feats, pd.DataFrame):
        raise TypeError('binned_feats must be a pandas DataFrame')

    p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError('percentiles must satisfy 0 <= low < high <= 100')

    nonneg_vars = {'v', 'd', 'r_targ'}
    signed_vars = {'w', 'phi', 'theta_targ', 'eye_ver', 'eye_hor'}

    out: Dict[str, np.ndarray] = {}
    for var in vars_to_include:
        if var not in binned_feats.columns:
            continue
        x = pd.to_numeric(binned_feats[var],
                          errors='coerce').to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        lo = float(np.percentile(x, p_lo))
        hi = float(np.percentile(x, p_hi))

        if var in signed_vars:
            m = float(max(abs(lo), abs(hi)))
            lo, hi = -m, m
        if var in nonneg_vars:
            lo = max(0.0, lo)
            hi = max(lo + 1e-6, hi)

        width = hi - lo
        lo = lo - pad_frac * width
        hi = hi + pad_frac * width
        if var in nonneg_vars:
            lo = max(0.0, lo)

        out[var] = np.array([lo, hi], dtype=float)

    return out


def default_prs() -> StopParams:
    """
    Return a default StopParams object for stop-aligned analyses.

    Example
    -------
    >>> from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
    >>> prs = multiff_encoding_params.default_prs()
    >>> binrange_dict = prs.binrange
    """

    return StopParams()
