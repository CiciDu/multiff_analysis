"""
population_analysis_utils.py

General-purpose utilities for population neural data analysis.
Extracted from the Python replication of AnalysePopulation.m.

Includes:
- spike binning
- trial concatenation
- smoothing
- trajectory reconstruction
- basic kinematic transforms

Author: you
"""

import numpy as np


# =========================
# Spike binning
# =========================

import numpy as np


def bin_spikes(spike_times, ts):
    """
    MATLAB-compatible spike binning equivalent to:
    hist(spike_times, ts)

    Parameters
    ----------
    spike_times : (S,)
    ts          : (T,)  bin centers

    Returns
    -------
    counts : (T,)
    """
    if len(ts) < 2:
        return np.zeros(len(ts))

    # Construct bin edges from bin centers (MATLAB hist behavior)
    dt = np.median(np.diff(ts))
    edges = np.concatenate([
        [ts[0] - dt / 2],
        ts + dt / 2
    ])

    counts, _ = np.histogram(spike_times, bins=edges)
    return counts.astype(float)

# =========================
# Smoothing
# =========================

def gaussian_kernel(width):
    """
    Create a normalized Gaussian kernel.
    """
    t = np.arange(-2 * width, 2 * width + 1)
    kernel = np.exp(-t ** 2 / (2 * width ** 2))
    return kernel / kernel.sum()


def smooth_signal(x, width):
    """
    Smooth signal(s) with a Gaussian kernel.

    x can be:
    - (T,)
    - (T, N)
    """
    if width <= 0:
        return x

    kernel = gaussian_kernel(width)

    if x.ndim == 1:
        return np.convolve(x, kernel, mode='same')

    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=x
    )



def deconcatenate_trials(signal_concat, trial_lengths):
    """
    Split a concatenated signal back into trial-wise segments.
    """
    output = []
    cursor = 0

    for length in trial_lengths:
        output.append(signal_concat[cursor:cursor + length])
        cursor += length

    return output


# =========================
# Kinematics
# =========================

def integrate_velocity(v, ts, dt):
    """
    Integrate forward velocity to get distance (D).
    """
    distance = np.zeros_like(v)
    valid = ts > 0
    distance[valid] = np.cumsum(v[valid]) * dt
    return distance


def integrate_angular_velocity(w, ts, dt):
    """
    Integrate angular velocity to get heading angle (Phi).
    """
    return np.cumsum(w * (ts > 0)) * dt


def reconstruct_trajectory(w, v, dt):
    """
    Reconstruct (x, y) trajectory from angular and linear velocity.
    """
    x = np.zeros(len(v))
    y = np.zeros(len(v))

    for i in range(1, len(v)):
        x[i] = x[i - 1] + v[i] * np.cos(w[i]) * dt
        y[i] = y[i - 1] + v[i] * np.sin(w[i]) * dt

    return x, y


def event_impulse(tr, tid, event_name):
    ts = tr.continuous.ts
    evt_t = getattr(tr.events, event_name)

    ev = np.zeros_like(ts)
    idx = np.searchsorted(ts, evt_t)
    if 0 <= idx < len(ts):
        ev[idx] = 1.0
    return ev


def concatenate_trials_with_trial_id(
    trials,
    trial_indices,
    signal_fn,
    time_window_fn
):
    """
    MATLAB-compatible trial concatenation.

    Replicates:
    - Remove first and last bin (2:end-1)
    - Strict windowing using trimmed time vector
    """

    signal_list = []
    trial_id_list = []

    for tid in trial_indices:
        tr = trials[tid]

        ts = tr.continuous.ts
        signal = signal_fn(tr, tid)

        # -----------------------------------------
        # 1) Remove first and last bin
        # -----------------------------------------
        ts_trim = ts[1:-1]
        signal_trim = signal[1:-1]

        # -----------------------------------------
        # 2) Compute strict time window on trimmed ts
        # -----------------------------------------
        t_start, t_stop = time_window_fn(tr)

        mask = (ts_trim > t_start) & (ts_trim < t_stop)

        signal_final = signal_trim[mask]

        signal_list.append(signal_final)
        trial_id_list.append(
            np.full(len(signal_final), tid, dtype=int)
        )

    return (
        np.concatenate(signal_list),
        np.concatenate(trial_id_list)
    )
    
    
def full_time_window(tr, pretrial=0.5, posttrial=0.5):
    """
    MATLAB-compatible full time window.

    Returns
    -------
    (t_start, t_stop)
    """

    t_start = min(tr.events.t_move, tr.events.t_targ) - pretrial
    t_stop  = tr.events.t_end + posttrial

    return t_start, t_stop


import numpy as np


def concatenate_covariates_with_trial_id(
    trials,
    trial_indices,
    covariate_fn,
    time_window_fn,
    covariate_names,
    duration_zeropad=None,
    duration_nanpad=None
):
    """
    MATLAB-equivalent concatenation with optional padding.

    Replicates ConcatenateTrials.m behavior:
    - Remove first and last bin (2:end-1)
    - Strict windowing (> and <)
    - Optional zero OR NaN padding per trial
    """

    if duration_zeropad is not None and duration_nanpad is not None:
        raise ValueError("Use either zero padding OR NaN padding, not both.")

    covariates_concat = {name: [] for name in covariate_names}
    trial_id_list = []

    for tid in trial_indices:
        tr = trials[tid]

        ts = tr.continuous.ts
        dt = np.median(np.diff(ts))

        t_start, t_stop = time_window_fn(tr)

        # --------------------------------------------------
        # 1) Remove first and last bin (2:end-1)
        # --------------------------------------------------
        ts_trim = ts[1:-1]
        mask = (ts_trim > t_start) & (ts_trim < t_stop)

        covs = covariate_fn(tr)

        # --------------------------------------------------
        # 2) Compute padding (in samples)
        # --------------------------------------------------
        if duration_zeropad is not None:
            pad_len = int(round(duration_zeropad / dt))
            pad_value = 0.0
        elif duration_nanpad is not None:
            pad_len = int(round(duration_nanpad / dt))
            pad_value = np.nan
        else:
            pad_len = 0

        # --------------------------------------------------
        # 3) Process each covariate
        # --------------------------------------------------
        trial_length_after_mask = None

        for name in covariate_names:
            signal = covs[name]

            signal_trim = signal[1:-1]
            signal_final = signal_trim[mask]

            if pad_len > 0:
                padding = np.full(pad_len, pad_value)
                signal_final = np.concatenate([padding, signal_final])

            covariates_concat[name].append(signal_final)

            if trial_length_after_mask is None:
                trial_length_after_mask = len(signal_final)

        trial_id_list.append(
            np.full(trial_length_after_mask, tid, dtype=int)
        )

    # --------------------------------------------------
    # 4) Concatenate across trials
    # --------------------------------------------------
    for name in covariate_names:
        covariates_concat[name] = np.concatenate(
            covariates_concat[name]
        )

    trial_id_vec = np.concatenate(trial_id_list)

    # --------------------------------------------------
    # 5) MATLAB NaN-padding adds trailing NaNs once
    # --------------------------------------------------
    if duration_nanpad is not None:
        pad_len = int(round(duration_nanpad / dt))
        padding = np.full(pad_len, np.nan)

        for name in covariate_names:
            covariates_concat[name] = np.concatenate(
                [covariates_concat[name], padding]
            )

        trial_id_vec = np.concatenate(
            [trial_id_vec, np.full(pad_len, -1)]
        )

    return covariates_concat, trial_id_vec