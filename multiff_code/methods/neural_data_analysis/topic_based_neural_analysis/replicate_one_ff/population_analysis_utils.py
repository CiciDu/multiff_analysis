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

def bin_spikes(spike_times, ts):
    """
    Bin spike times into a per-sample spike count vector.
    """
    counts = np.zeros(len(ts), dtype=float)
    idx = np.searchsorted(ts, spike_times)
    idx = idx[(idx >= 0) & (idx < len(ts))]
    np.add.at(counts, idx, 1)
    return counts


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


# =========================
# Trial concatenation
# =========================

def concatenate_trials_with_trial_id(trials, trial_indices, signal_fn, time_window_fn):
    """
    Concatenate trial-wise signals into a single vector,
    while keeping an explicit trial-id vector.

    Returns:
    signal_concat   : (T,)
    trial_id_vec    : (T,)
    """
    signal_list = []
    trial_id_list = []

    for trial_index in trial_indices:
        trial = trials[trial_index]
        mask = time_window_fn(trial)

        signal = signal_fn(trial, trial_index)[mask]
        signal_list.append(signal)

        trial_id_list.append(
            np.full(len(signal), trial_index, dtype=int)
        )

    return (
        np.concatenate(signal_list),
        np.concatenate(trial_id_list)
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


def concatenate_covariates_with_trial_id(
    trials,
    trial_indices,
    covariate_fn,
    time_window_fn,
    covariate_names
):
    """
    Concatenate multiple covariates across trials using a shared trial_id_vec.

    Parameters
    ----------
    covariate_fn : function(trial) -> dict
        Returns a dict of covariates for ONE trial.
    covariate_names : list of str
        Keys to extract and concatenate.

    Returns
    -------
    covariates_concat : dict[str, np.ndarray]
    trial_id_vec      : np.ndarray
    """
    covariates_concat = {name: [] for name in covariate_names}
    trial_id_vec = []

    for tid in trial_indices:
        tr = trials[tid]
        mask = time_window_fn(tr)

        covs = covariate_fn(tr)

        for name in covariate_names:
            covariates_concat[name].append(covs[name][mask])

        trial_id_vec.append(
            np.full(np.sum(mask), tid, dtype=int)
        )

    for name in covariate_names:
        covariates_concat[name] = np.concatenate(covariates_concat[name])

    trial_id_vec = np.concatenate(trial_id_vec)

    return covariates_concat, trial_id_vec


def full_time_window(tr, pretrial=0.5, posttrial=0.5):
    """
    Time window for concatenating trial data.

    Includes:
    - pretrial buffer before target onset or movement onset (whichever is earlier)
    - posttrial buffer after end of trial

    Matches AnalysePopulation.m:
    timewindow_full = [
        min(t_move, t_targ) - pretrial,
        t_end + posttrial
    ]
    """
    t_start = min(tr.events.t_move, tr.events.t_targ) - pretrial
    t_stop  = tr.events.t_end + posttrial

    ts = tr.continuous.ts
    return (ts >= t_start) & (ts <= t_stop)
