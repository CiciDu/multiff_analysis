"""
Shared-manifold analysis and visualization utilities.
Notebook-friendly, modular, optionally saving figures.
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GroupKFold


# =========================
# Basic helpers
# =========================
def zscore(A, eps=1e-12):
    mu = np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd, mu, sd


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(np.isfinite(a), axis=axis)
    n = np.maximum(n, 1)
    return np.nanstd(a, axis=axis) / np.sqrt(n)


def smooth_1d(x, k=1):
    if k <= 1:
        return x
    w = np.ones(int(k)) / k
    return np.convolve(x, w, mode='same')


def clip_quantile(y, lo=0.01, hi=0.99):
    if lo is None or hi is None:
        return y
    ql, qh = np.nanpercentile(y, [lo * 100, hi * 100])
    return np.clip(y, ql, qh)


def cumulative_shared_variance(corrs):
    pwr = np.cumsum(np.square(np.asarray(corrs, float)))
    tot = pwr[-1] if pwr[-1] > 0 else 1.0
    return pwr / tot


# =========================
# Data + model utilities
# =========================
def split_by_trial(X, Y, trial_ids, n_splits, which_fold):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=trial_ids))
    tr_idx, te_idx = splits[which_fold]
    return (X[tr_idx], Y[tr_idx]), (X[te_idx], Y[te_idx]), te_idx


def fit_cca(Xtr, Ytr, Xte, Yte, n_components):
    cca = CCA(n_components=int(n_components), max_iter=1000)
    cca.fit(Xtr, Ytr)
    Xc, Yc = cca.transform(Xte, Yte)
    return cca, Xc, Yc


def make_shared_df(Xc, Yc, trial_ids, time_idx, cond):
    dfs = []
    for view, Z in [('obs', Xc), ('pred', Yc)]:
        d = pd.DataFrame({
            'trial': trial_ids,
            'time': time_idx,
            'cond': cond,
            'view': view
        })
        for i in range(Z.shape[1]):
            d[f'can{i+1}'] = Z[:, i]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


# =========================
# Plotting primitives
# =========================
def compute_mean_sem(df, comp, smooth_bins):
    out = {}
    for view in ['obs', 'pred']:
        sub = df[df.view == view]
        g = sub.groupby('time')[comp]
        mu = smooth_1d(g.mean().values, smooth_bins)
        se = smooth_1d(g.apply(lambda x: sem(x.values)).values, smooth_bins)
        out[view] = (g.mean().index.values, mu, se)
    return out


def plot_mean_sem(ax, t, mu, se, color, label, ls='-'):
    ax.plot(t, mu, color=color, ls=ls, label=label)
    ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.18)


def plot_single_trials(
    ax, df, comp, trials, view, color,
    smooth_bins, overlay_norm, clip_quantiles,
    alpha, lw
):
    for tr in trials:
        one = df[(df.trial == tr) & (df.view == view)][['time', comp]]
        one = one.sort_values('time').values
        t, y = one[:, 0], one[:, 1]

        y = clip_quantile(y, *clip_quantiles)
        if overlay_norm in ('zscore', 'zscore_to_sem'):
            sd = np.nanstd(y)
            sd = 1.0 if sd < 1e-12 else sd
            y = (y - np.nanmean(y)) / sd

        y = smooth_1d(y, smooth_bins)
        ax.plot(t, y, color=color, alpha=alpha, lw=lw)


def set_mean_ylim(ax, mu_obs, se_obs, mu_pred, se_pred, mode):
    if mode == 'mean_sem_3sd':
        pooled = se_obs + se_pred
        lo = np.nanmin([mu_obs - 3 * pooled, mu_pred - 3 * pooled])
        hi = np.nanmax([mu_obs + 3 * pooled, mu_pred + 3 * pooled])
        pad = 0.05 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    elif isinstance(mode, tuple) and mode[0] == 'fixed':
        ax.set_ylim(*mode[1])


# =========================
# High-level plotting API
# =========================
def plot_shared_components(
    df, n_components, *,
    smooth_bins=1,
    overlay_mode='separate',
    n_trials_overlay=12,
    overlay_norm='zscore_to_sem',
    clip_quantiles=(0.01, 0.99),
    colors=('C0', 'C1'),
    alpha_trials=0.22,
    lw_trials=0.7,
    ylimit_mode='mean_sem_3sd',
    save=False,
    out_dir=None,
    prefix='shared_comp'
):
    figs = []
    rng = np.random.default_rng(0)
    all_trials = df.trial.unique()

    for ic in range(n_components):
        comp = f'can{ic+1}'
        stats = compute_mean_sem(df, comp, smooth_bins)

        if overlay_mode == 'separate':
            fig, (ax_mean, ax_trials) = plt.subplots(1, 2, figsize=(9, 3.2))
        else:
            fig, ax_mean = plt.subplots(1, 1, figsize=(5.4, 3.2))
            ax_trials = ax_mean

        for view, ls in [('obs', '-'), ('pred', '--')]:
            t, mu, se = stats[view]
            plot_mean_sem(
                ax_mean, t, mu, se,
                colors[0 if view == 'obs' else 1],
                view, ls
            )

        ax_mean.axvline(0, color='k', lw=1, alpha=0.25)
        ax_mean.set_title(f'Canonical component {ic+1}')
        ax_mean.set_xlabel('Time')
        ax_mean.set_ylabel('Canonical coordinate')
        ax_mean.legend(fontsize=8)
        ax_mean.grid(alpha=0.2, ls=':')

        set_mean_ylim(
            ax_mean,
            stats['obs'][1], stats['obs'][2],
            stats['pred'][1], stats['pred'][2],
            ylimit_mode
        )

        pick = (
            all_trials if len(all_trials) <= n_trials_overlay
            else rng.choice(all_trials, n_trials_overlay, replace=False)
        )

        for view, color in [('obs', colors[0]), ('pred', colors[1])]:
            plot_single_trials(
                ax_trials, df, comp, pick, view, color,
                smooth_bins, overlay_norm, clip_quantiles,
                alpha_trials, lw_trials
            )

        if overlay_mode == 'separate':
            ax_trials.axvline(0, color='k', lw=1, alpha=0.25)
            ax_trials.grid(alpha=0.2, ls=':')

        plt.tight_layout()

        if save and out_dir is not None:
            fig.savefig(f'{out_dir}/{prefix}_{comp}.png', dpi=200)

        figs.append(fig)

    return figs


def plot_phase_plane(
    df, *, smooth_bins=1, cmap_name='viridis',
    save=False, out_path=None
):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    norm = Normalize(df.time.min(), df.time.max())
    cmap = cm.get_cmap(cmap_name)

    for view, ls in [('obs', '-'), ('pred', '--')]:
        sub = df[df.view == view]
        m1 = smooth_1d(sub.groupby('time')['can1'].mean().values, smooth_bins)
        m2 = smooth_1d(sub.groupby('time')['can2'].mean().values, smooth_bins)
        t = sub.groupby('time').mean().index.values

        for i in range(len(t) - 1):
            ax.plot(m1[i:i+2], m2[i:i+2],
                    ls=ls, color=cmap(norm(t[i])))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label='Time')
    ax.set_xlabel('Can1')
    ax.set_ylabel('Can2')
    ax.set_title('Shared-manifold phase plane')
    ax.grid(alpha=0.2, ls=':')

    if save and out_path is not None:
        fig.savefig(out_path, dpi=200)

    return fig
