# =============================
# FILE: multiff.py
# =============================
"""MultiFF adapter, simulators, and visualization/debug helpers.

This ties together the glm_bases/glm_design/fit modules for the multi-firefly task:
- Builds gated stimulus features (visible-only distance/angle, alignments).
- Expands them with raised-cosine glm_bases (separate for events vs. short effects).
- Optionally adds speed/curvature and spike history.
- Provides end-to-end fit convenience and synthetic-data simulators.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_bases import raised_cosine_basis, onset_from_mask_trials, angle_sin_cos, wrap_angle, _unique_trials, safe_poisson_lambda
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_design import build_glm_design_with_trials, lagged_design_from_signal_trials
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_fit import fit_poisson_glm_trials, predict_mu, poisson_deviance, pseudo_R2, per_trial_deviance


# -------- MultiFF glm_design builder --------

def build_multiff_design(
    *,
    dt: float,
    trial_ids: np.ndarray,
    cur_vis: np.ndarray,
    nxt_vis: np.ndarray,
    cur_dist: np.ndarray,
    nxt_dist: np.ndarray,
    cur_angle: np.ndarray,
    nxt_angle: np.ndarray,
    heading: Optional[np.ndarray] = None,
    speed: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    spike_counts: Optional[np.ndarray] = None,
    use_trial_FE: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Construct the MultiFF GLM glm_design and return ``(design_df, y, meta)``.

    Gating rules
    ------------
    - Distance and angle terms are *gated by visibility*. When a target is not
      visible, its distance/angle columns are zero so they cannot drive the GLM.
    - Alignments (if heading provided) are also visible-only.

    glm_bases
    -----
    - ``B_event``: slightly longer for event onsets (e.g., visibility onsets).
    - ``B_short``: shorter for distance/angle/align effects.
    - ``B_hist``: history basis (starts at ``t=dt`` to enforce strictly past).
    """
    T = len(trial_ids)

    # --- glm_bases ---
    _, B_event = raised_cosine_basis(
        n_basis=6, t_max=0.60, dt=dt, t_min=0.0, log_spaced=True)
    _, B_short = raised_cosine_basis(
        n_basis=5, t_max=0.30, dt=dt, t_min=0.0, log_spaced=True)
    _, B_hist = raised_cosine_basis(
        n_basis=5, t_max=0.20, dt=dt, t_min=dt,  log_spaced=True)

    # --- Onsets and gated features ---
    cur_on = onset_from_mask_trials(cur_vis, trial_ids)
    nxt_on = onset_from_mask_trials(nxt_vis, trial_ids)

    cur_dist_g = cur_dist * (cur_vis > 0)
    nxt_dist_g = nxt_dist * (nxt_vis > 0)

    cur_sin, cur_cos = angle_sin_cos(cur_angle)
    nxt_sin, nxt_cos = angle_sin_cos(nxt_angle)
    cur_sin *= (cur_vis > 0)
    cur_cos *= (cur_vis > 0)
    nxt_sin *= (nxt_vis > 0)
    nxt_cos *= (nxt_vis > 0)

    stimulus_dict: Dict[str, np.ndarray] = {
        "cur_on": cur_on,
        "nxt_on": nxt_on,
        "cur_dist": cur_dist_g,
        "nxt_dist": nxt_dist_g,
        "cur_angle_sin": cur_sin,
        "cur_angle_cos": cur_cos,
        "nxt_angle_sin": nxt_sin,
        "nxt_angle_cos": nxt_cos,
    }

    extra_covariates: Dict[str, np.ndarray] = {}
    if heading is not None:
        cur_align = np.cos(wrap_angle(heading - cur_angle)) * (cur_vis > 0)
        nxt_align = np.cos(wrap_angle(heading - nxt_angle)) * (nxt_vis > 0)
        stimulus_dict["cur_align"] = cur_align
        stimulus_dict["nxt_align"] = nxt_align

    if speed is not None:
        extra_covariates["speed"] = speed
    if curvature is not None:
        extra_covariates["curvature"] = curvature

    stimulus_basis_dict: Dict[str, np.ndarray] = {
        "cur_on": B_event, "nxt_on": B_event,
        "cur_dist": B_short, "nxt_dist": B_short,
        "cur_angle_sin": B_short, "cur_angle_cos": B_short,
        "nxt_angle_sin": B_short, "nxt_angle_cos": B_short,
        **({"cur_align": B_short, "nxt_align": B_short} if heading is not None else {}),
    }

    design_df, y = build_glm_design_with_trials(
        dt=dt,
        trial_ids=trial_ids,
        stimulus_dict=stimulus_dict,
        stimulus_basis_dict=stimulus_basis_dict,
        spike_counts=spike_counts,
        history_basis=B_hist,
        extra_covariates=extra_covariates,
        use_trial_FE=use_trial_FE,
    )

    meta = {"B_event": B_event, "B_short": B_short, "B_hist": B_hist}
    return design_df, y, meta


def fit_multiff_glm(
    *,
    dt: float,
    trial_ids: np.ndarray,
    cur_vis: np.ndarray,
    nxt_vis: np.ndarray,
    cur_dist: np.ndarray,
    nxt_dist: np.ndarray,
    cur_angle: np.ndarray,
    nxt_angle: np.ndarray,
    heading: Optional[np.ndarray] = None,
    speed: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    spike_counts: np.ndarray,
    l2: float = 0.0,
    use_trial_FE: bool = True,
    cluster_se: bool = False,
):
    """End-to-end: build MultiFF glm_design, fit GLM, and compute basic metrics."""
    design_df, y, meta = build_multiff_design(
        dt=dt, trial_ids=trial_ids,
        cur_vis=cur_vis, nxt_vis=nxt_vis,
        cur_dist=cur_dist, nxt_dist=nxt_dist,
        cur_angle=cur_angle, nxt_angle=nxt_angle,
        heading=heading, speed=speed, curvature=curvature,
        spike_counts=spike_counts, use_trial_FE=use_trial_FE,
    ) 
    res = fit_poisson_glm_trials(
        design_df, y, dt, trial_ids, add_const=True, l2=l2, cluster_se=False)
    mu = predict_mu(res, design_df, dt)
    mu_null = np.full_like(y, y.mean(), dtype=float)
    metrics = {
        "deviance": poisson_deviance(y, mu),
        "pseudo_R2": pseudo_R2(y, mu, mu_null),
        "per_trial_deviance": per_trial_deviance(y, mu, trial_ids),
    }
    return res, design_df, metrics, meta


# -------- Simple simulators (stable via capped intensity) --------

def simulate_spikes_with_trials(
    *, n_trials: int = 20, trial_len: int = 300, dt: float = 0.01, seed: int = 1,
    max_rate_hz: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Simulate generic event+speed GLM data with per-trial baselines and history."""
    rng = np.random.default_rng(seed)
    T = n_trials * trial_len
    trial_ids = np.repeat(np.arange(n_trials), trial_len)

    _, B_stim = raised_cosine_basis(
        n_basis=5, t_max=0.300, dt=dt, t_min=0.0, log_spaced=True)
    _, B_hist = raised_cosine_basis(
        n_basis=4, t_max=0.200, dt=dt, t_min=dt, log_spaced=True)

    event = np.concatenate(
        [(rng.random(trial_len) < 0.02).astype(float) for _ in range(n_trials)])
    # AR(1)-like speed process; then standardize
    from scipy import signal as _sig
    speed = np.concatenate([_sig.lfilter(
        [1.0], [1.0, -0.7], rng.normal(0.0, 1.0, size=trial_len)) for _ in range(n_trials)])
    speed = (speed - speed.mean()) / (speed.std() + 1e-12)

    beta_event = np.array([0.8, 0.5, 0.2, -0.1, -0.2])
    beta_speed = np.array([0.4, 0.2, 0.0, -0.1, -0.2])
    beta_hist = np.array([-1.2, -0.8, -0.5, -0.2])

    X_event = lagged_design_from_signal_trials(event, B_stim, trial_ids)
    X_speed = lagged_design_from_signal_trials(speed, B_stim, trial_ids)

    h_hist = B_hist @ beta_hist
    Lh = len(h_hist)

    baseline_per_trial = -2.8 + 0.4 * rng.standard_normal(n_trials)

    y = np.zeros(T, dtype=int)
    for tr in _unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        past = np.zeros(Lh)
        b0 = baseline_per_trial[tr]
        for t in idx:
            stim_drive = X_event[t] @ beta_event + X_speed[t] @ beta_speed
            hist_drive = np.dot(past, h_hist[::-1])
            eta = b0 + stim_drive + hist_drive
            lam = safe_poisson_lambda(eta, dt, max_rate_hz=max_rate_hz)
            y[t] = np.random.poisson(lam)
            past = np.roll(past, 1)
            past[0] = y[t]

    stimulus_dict = {"event": event, "speed": speed}
    stimulus_basis_dict = {"event": B_stim, "speed": B_stim}
    return trial_ids, y, stimulus_dict, stimulus_basis_dict, B_hist


def simulate_multiff_trials(
    *, n_trials: int = 12, trial_len: int = 400, dt: float = 0.01, seed: int = 3,
    max_rate_hz: float = 200.0,
):
    """Simulate MultiFF-like data with visibility-gated signals and history.

    Produces a dictionary you can pass directly into ``fit_multiff_glm``.
    """
    rng = np.random.default_rng(seed)
    T = n_trials * trial_len
    trial_ids = np.repeat(np.arange(n_trials), trial_len)

    def random_vis_mask():
        m = np.zeros(trial_len, dtype=int)
        t = 0
        while t < trial_len:
            off = rng.integers(20, 80)
            on = rng.integers(30, 120)
            t += off
            if t >= trial_len:
                break
            m[t: min(trial_len, t + on)] = 1
            t += on
        return m

    cur_vis = np.concatenate([random_vis_mask() for _ in range(n_trials)])
    nxt_vis = np.concatenate([np.r_[np.zeros(rng.integers(
        40, 120)), random_vis_mask()][:trial_len] for _ in range(n_trials)])

    def distance_from_vis(m):
        d = np.zeros_like(m, dtype=float)
        run = 0.0
        for i, v in enumerate(m):
            if v:
                run = 1.0 if run == 0 else max(0.0, run - 0.02)
                d[i] = run
            else:
                run = 0.0
                d[i] = 0.0
        return d

    cur_dist = np.concatenate([distance_from_vis(
        cur_vis[i*trial_len:(i+1)*trial_len]) for i in range(n_trials)])
    nxt_dist = np.concatenate([distance_from_vis(
        nxt_vis[i*trial_len:(i+1)*trial_len]) for i in range(n_trials)])

    def rand_angle_series(L):
        a = np.cumsum(rng.normal(0, 0.05, size=L))
        return ((a + np.pi) % (2*np.pi)) - np.pi

    cur_angle = np.concatenate([rand_angle_series(trial_len)
                               for _ in range(n_trials)])
    nxt_angle = np.concatenate([rand_angle_series(trial_len)
                               for _ in range(n_trials)])
    heading = np.concatenate([rand_angle_series(trial_len)
                             for _ in range(n_trials)])

    from scipy import signal as _sig
    speed = np.concatenate([_sig.lfilter(
        [1.0], [1.0, -0.8], rng.normal(0, 1.0, size=trial_len)) for _ in range(n_trials)])
    speed = (speed - speed.mean()) / (speed.std() + 1e-12)
    curvature = np.concatenate(
        [rng.normal(0.0, 0.4, size=trial_len) for _ in range(n_trials)])

    # Build glm_design (no history yet) to compute stim_drive for simulation
    design_nohist, _, meta = build_multiff_design(
        dt=dt, trial_ids=trial_ids,
        cur_vis=cur_vis, nxt_vis=nxt_vis,
        cur_dist=cur_dist, nxt_dist=nxt_dist,
        cur_angle=cur_angle, nxt_angle=nxt_angle,
        heading=heading, speed=speed, curvature=curvature,
        spike_counts=None, use_trial_FE=True,
    )

    cols = list(design_nohist.columns)

    def idxs(prefix):
        return [i for i, c in enumerate(cols) if c.startswith(prefix)]

    beta = np.zeros(design_nohist.shape[1])
    for p, gain in [("cur_on", 0.9), ("nxt_on", 0.5)]:
        j = idxs(p)
        if j:
            beta[j] = np.linspace(gain, 0.0, num=len(j))
    for p, gain in [("cur_dist", -0.7), ("nxt_dist", -0.5)]:
        j = idxs(p)
        if j:
            beta[j] = np.linspace(gain, 0.0, num=len(j))
    for p, gain in [("cur_angle_sin", 0.15), ("cur_angle_cos", 0.15), ("nxt_angle_sin", 0.10), ("nxt_angle_cos", 0.10)]:
        j = idxs(p)
        if j:
            beta[j] = gain / max(1, len(j))
    for p, gain in [("cur_align", 0.35), ("nxt_align", 0.20)]:
        j = idxs(p)
        if j:
            beta[j] = np.linspace(gain, 0.0, num=len(j))
    if "speed" in cols:
        beta[cols.index("speed")] = 0.2
    if "curvature" in cols:
        beta[cols.index("curvature")] = -0.25

    stim_drive = design_nohist.values @ beta

    B_hist = meta["B_hist"]
    h_hist = B_hist @ np.array([-1.2, -0.8, -0.5, -0.3, -0.2])
    Lh = len(h_hist)

    baseline = -3.0 + 0.3 * rng.standard_normal(n_trials)

    y = np.zeros(T, dtype=int)
    for tr in _unique_trials(trial_ids):
        idx = np.where(trial_ids == tr)[0]
        past = np.zeros(Lh)
        b0 = baseline[tr]
        for t in idx:
            eta = b0 + stim_drive[t] + np.dot(past, h_hist[::-1])
            lam = safe_poisson_lambda(eta, dt, max_rate_hz=max_rate_hz)
            y[t] = np.random.poisson(lam)
            past = np.roll(past, 1)
            past[0] = y[t]

    return {
        "trial_ids": trial_ids,
        "cur_vis": cur_vis,
        "nxt_vis": nxt_vis,
        "cur_dist": cur_dist,
        "nxt_dist": nxt_dist,
        "cur_angle": cur_angle,
        "nxt_angle": nxt_angle,
        "heading": heading,
        "speed": speed,
        "curvature": curvature,
        "spike_counts": y,
        "dt": dt,
    }
