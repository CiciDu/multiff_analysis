"""
Population analysis pipeline
Python replication of AnalysePopulation.m (core functionality)

Author: you + ChatGPT
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import gaussian, hilbert
from scipy.linalg import lstsq
from sklearn.cross_decomposition import CCA
from dataclasses import dataclass
from typing import List, Dict


# =========================
# Parameters (prs struct)
# =========================

@dataclass
class Params:
    dt: float = 0.01
    neural_filtwidth: int = 5
    pretrial: float = 0.5
    posttrial: float = 0.5
    cca_vars: List[str] = None
    decode_vars: List[str] = None


prs = Params(
    dt=0.01,
    neural_filtwidth=5,
    cca_vars=['v', 'w'],
    decode_vars=['v', 'w']
)


# =========================
# Utilities
# =========================

def gaussian_kernel(width):
    t = np.arange(-2 * width, 2 * width + 1)
    h = np.exp(-t ** 2 / (2 * width ** 2))
    return h / h.sum()


def smooth_signal(x, width):
    if width <= 0:
        return x
    h = gaussian_kernel(width)
    return np.apply_along_axis(lambda m: np.convolve(m, h, mode='same'), 0, x)


def bin_spikes(spike_times, ts):
    counts = np.zeros(len(ts))
    idx = np.searchsorted(ts, spike_times)
    idx = idx[(idx >= 0) & (idx < len(ts))]
    np.add.at(counts, idx, 1)
    return counts


def concatenate_trials(trials, trial_ids, signal_fn, time_window_fn):
    X = []
    trial_lengths = []

    for tid in trial_ids:
        tr = trials[tid]
        mask = time_window_fn(tr)
        sig = signal_fn(tr)[mask]
        X.append(sig)
        trial_lengths.append(len(sig))

    return np.concatenate(X), trial_lengths


def deconcatenate(x, trial_lengths):
    out = []
    idx = 0
    for L in trial_lengths:
        out.append(x[idx:idx+L])
        idx += L
    return out


def compute_d(v, ts, dt):
    d = np.zeros_like(v)
    valid = ts > 0
    d[valid] = np.cumsum(v[valid]) * dt
    return d


def compute_phi(w, ts, dt):
    phi = np.zeros_like(w)
    valid = ts > 0
    phi[valid] = np.cumsum(w[valid]) * dt
    return phi


# =========================
# Load data
# =========================

data = loadmat(
    'all_monkey_data/one_ff_data/sessions_python.mat',
    squeeze_me=True,
    struct_as_record=False
)

sessions = data['sessions_out']
session = sessions[0]

trials = session.behaviour_trials
units = session.units

n_trials = len(trials)
n_units = len(units)


# =========================
# Build trial index (all trials)
# =========================

trial_ids = np.arange(n_trials)


# =========================
# Time window helper
# =========================

def full_time_window(tr):
    t0 = min(tr.events.t_move, tr.events.t_targ) - prs.pretrial
    t1 = tr.events.t_end + prs.posttrial
    return (tr.continuous.ts >= t0) & (tr.continuous.ts <= t1)


# =========================
# Build stimulus matrix X
# =========================

def get_var(tr, name):
    if name == 'v':
        return tr.continuous.v
    if name == 'w':
        return tr.continuous.w
    if name == 'd':
        return compute_d(tr.continuous.v, tr.continuous.ts, prs.dt)
    if name == 'phi':
        return compute_phi(tr.continuous.w, tr.continuous.ts, prs.dt)
    raise ValueError(name)


X_list = []
trial_lengths = None

for var in prs.cca_vars:
    x, trial_lengths = concatenate_trials(
        trials,
        trial_ids,
        lambda tr, v=var: get_var(tr, v),
        full_time_window
    )
    X_list.append(x)

X = np.column_stack(X_list)
X[np.isnan(X)] = 0


# =========================
# Build population activity Y
# =========================

Y = np.zeros((X.shape[0], n_units))

for k in range(n_units):
    yk, _ = concatenate_trials(
        trials,
        trial_ids,
        lambda tr, k=k: bin_spikes(units[k].trials[tr.trial_id].tspk, tr.continuous.ts),
        full_time_window
    )
    Y[:, k] = yk

Y_smooth = smooth_signal(Y, prs.neural_filtwidth) / prs.dt


# =========================
# Canonical Correlation Analysis
# =========================

cca = CCA(n_components=min(X.shape[1], Y_smooth.shape[1]))
Xc, Yc = cca.fit_transform(X, Y_smooth)

cca_corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(Xc.shape[1])]

print('CCA correlations:', cca_corrs)


# =========================
# Linear population decoding
# =========================

decode_results = {}

for var in prs.decode_vars:
    xt, _ = concatenate_trials(
        trials,
        trial_ids,
        lambda tr, v=var: get_var(tr, v),
        full_time_window
    )
    xt[np.isnan(xt)] = 0

    Yd = smooth_signal(Y, prs.neural_filtwidth)

    wts, _, _, _ = lstsq(Yd, xt)
    pred = Yd @ wts

    corr = np.corrcoef(xt, pred)[0, 1]

    decode_results[var] = dict(
        weights=wts,
        corr=corr,
        true=deconcatenate(xt, trial_lengths),
        pred=deconcatenate(pred, trial_lengths)
    )

    print(f'Decode {var}: r = {corr:.3f}')


# =========================
# Trajectory reconstruction
# =========================

def gen_traj(w, v, ts):
    x = np.zeros(len(ts))
    y = np.zeros(len(ts))
    for i in range(1, len(ts)):
        x[i] = x[i-1] + v[i] * np.cos(w[i]) * prs.dt
        y[i] = y[i-1] + v[i] * np.sin(w[i]) * prs.dt
    return x, y


traj_pred = []

for tr_v, tr_w in zip(decode_results['v']['pred'], decode_results['w']['pred']):
    x, y = gen_traj(tr_w, tr_v, np.arange(len(tr_v)) * prs.dt)
    traj_pred.append((x, y))


print('Pipeline complete.')
