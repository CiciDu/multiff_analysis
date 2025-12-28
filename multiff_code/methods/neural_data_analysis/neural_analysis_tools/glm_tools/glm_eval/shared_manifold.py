# shared_manifold_plots.py
import numpy as np
import matplotlib.pyplot as plt


def _smooth_1d(x, k=1):
    if k <= 1:
        return x
    w = np.ones(int(k)) / k
    return np.convolve(x, w, mode='same')


def _sem(x):
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x))
    return 0.0 if n <= 1 else np.nanstd(x) / np.sqrt(n)


def plot_shared_components(
    df,
    *,
    n_components=None,
    components=None,
    smooth_bins=1,
    overlay_mode='separate',   # 'separate' | 'overlay'
    n_trials_overlay=12,
    colors=('C0', 'C1'),
    alpha_trials=0.25,
    lw_trials=0.7
):
    """
    Plot shared latent trajectories from a tidy DataFrame.

    Required columns:
        trial | time | view | can1 ... canK

    view:
        'obs'  = X-side latents
        'pred' = Y-side latents
    """
    # ---- component selection (backward compatible)
    if components is None:
        if n_components is None:
            raise ValueError('Must provide n_components or components')
        components = range(1, n_components + 1)

    figs = []
    trials = df['trial'].unique()
    rng = np.random.default_rng(0)

    for k in components:
        comp = f'can{k}'

        if overlay_mode == 'separate':
            fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(9, 3))
        else:
            fig, ax_m = plt.subplots(1, 1, figsize=(5, 3))
            ax_t = ax_m

        # ---- mean + SEM
        for view, ls, color in [('obs', '-', colors[0]),
                                 ('pred', '--', colors[1])]:
            sub = df[df['view'] == view]

            g = sub.groupby('time')[comp]
            t = g.mean().index.values
            mu = _smooth_1d(g.mean().values, smooth_bins)
            se = _smooth_1d(g.apply(_sem).values, smooth_bins)

            ax_m.plot(t, mu, ls=ls, color=color, label=view)
            ax_m.fill_between(t, mu - se, mu + se,
                              color=color, alpha=0.2)

        ax_m.axvline(0, color='k', lw=1, alpha=0.3)
        ax_m.set_title(f'Canonical component {k}')
        ax_m.set_xlabel('Time')
        ax_m.set_ylabel('Latent value')
        ax_m.legend()
        ax_m.grid(alpha=0.2, ls=':')

        # ---- single trials
        pick = trials if len(trials) <= n_trials_overlay else rng.choice(
            trials, n_trials_overlay, replace=False
        )

        for tr in pick:
            for view, color in [('obs', colors[0]), ('pred', colors[1])]:
                sub = df[(df['trial'] == tr) & (df['view'] == view)]
                y = _smooth_1d(sub[comp].values, smooth_bins)
                ax_t.plot(sub['time'].values, y,
                          color=color, alpha=alpha_trials, lw=lw_trials)

        if overlay_mode == 'separate':
            ax_t.axvline(0, color='k', lw=1, alpha=0.3)
            ax_t.set_title('Single trials')
            ax_t.grid(alpha=0.2, ls=':')

        plt.tight_layout()
        figs.append(fig)

    return figs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_shared_phase_plane(
    df,
    *,
    components=(1, 2),
    smooth_bins=1,
    cmap_name='viridis',
    colors=('C0', 'C1'),
    show_time_zero=True
):
    """
    Plot time-colored phase plane for shared CCA components.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy latent DataFrame with columns:
        trial | time | view | can1 | can2 | ...

    components : tuple[int, int]
        Which components to plot (default: (1, 2)).

    smooth_bins : int
        Temporal smoothing window.

    cmap_name : str
        Matplotlib colormap for time coloring.

    Returns
    -------
    fig : matplotlib Figure
    """
    kx, ky = components
    cx, cy = f'can{kx}', f'can{ky}'

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # time normalization
    t = df['time'].values
    norm = Normalize(vmin=np.nanmin(t), vmax=np.nanmax(t))
    cmap = cm.get_cmap(cmap_name)

    for view, ls, base_color in [('obs', '-', colors[0]),
                                 ('pred', '--', colors[1])]:
        sub = df[df['view'] == view]

        # mean trajectory over time
        m = (
            sub.groupby('time')[[cx, cy]]
            .mean()
            .sort_index()
        )

        x = _smooth_1d(m[cx].values, smooth_bins)
        y = _smooth_1d(m[cy].values, smooth_bins)
        tt = m.index.values

        # draw colored trajectory
        for i in range(len(tt) - 1):
            ax.plot(
                x[i:i+2], y[i:i+2],
                ls=ls,
                color=cmap(norm(tt[i])),
                lw=2
            )

        # mark time zero
        if show_time_zero and 0 in tt:
            i0 = np.where(tt == 0)[0][0]
            ax.scatter(
                x[i0], y[i0],
                s=40, color=base_color,
                edgecolor='k', zorder=3
            )

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time')

    ax.set_xlabel(cx)
    ax.set_ylabel(cy)
    ax.set_title(f'CCA phase plane ({cx} vs {cy})')
    ax.grid(alpha=0.2, ls=':')

    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_shared_phase_plane_by_fold(
    latents_by_fold,
    *,
    components=(1, 2),
    smooth_bins=1,
    cmap_name='viridis',
    colors=('C0', 'C1'),
    alpha_fold=0.5,
    lw=2,
    show_time_zero=True
):
    """
    Overlay fold-wise CCA phase-plane trajectories.

    Parameters
    ----------
    latents_by_fold : dict[int, pd.DataFrame]
        Output of res['parity']['latents_by_fold'].

    components : tuple[int, int]
        Which canonical components to plot (e.g. (1, 2)).

    Returns
    -------
    fig : matplotlib Figure
    """
    kx, ky = components
    cx, cy = f'can{kx}', f'can{ky}'

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    # global time normalization (shared across folds)
    all_times = np.concatenate([
        df['time'].values for df in latents_by_fold.values()
    ])
    norm = Normalize(vmin=np.min(all_times), vmax=np.max(all_times))
    cmap = cm.get_cmap(cmap_name)

    for fold, df in latents_by_fold.items():
        for view, ls, base_color in [('obs', '-', colors[0]),
                                     ('pred', '--', colors[1])]:

            sub = df[df['view'] == view]

            # mean trajectory for this fold
            m = (
                sub.groupby('time')[[cx, cy]]
                .mean()
                .sort_index()
            )

            x = _smooth_1d(m[cx].values, smooth_bins)
            y = _smooth_1d(m[cy].values, smooth_bins)
            t = m.index.values

            # colored trajectory
            for i in range(len(t) - 1):
                ax.plot(
                    x[i:i+2], y[i:i+2],
                    ls=ls,
                    color=cmap(norm(t[i])),
                    alpha=alpha_fold,
                    lw=lw
                )

            # mark time zero
            if show_time_zero and 0 in t:
                i0 = np.where(t == 0)[0][0]
                ax.scatter(
                    x[i0], y[i0],
                    s=30,
                    color=base_color,
                    alpha=alpha_fold,
                    edgecolor='k',
                    zorder=3
                )

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time')

    ax.set_xlabel(cx)
    ax.set_ylabel(cy)
    ax.set_title('CCA phase plane (fold-wise overlay)')
    ax.grid(alpha=0.2, ls=':')

    plt.tight_layout()
    return fig
