import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def _safe_div(a, b, fill=0.0):
    out = np.zeros_like(a, dtype=float)
    m = b > 0
    out[m] = a[m] / b[m]
    out[~m] = fill
    return out

def _gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins is None or sigma_bins <= 0:
        return y
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    w = np.exp(-(x**2) / (2 * sigma_bins**2))
    w /= w.sum()

    full = np.convolve(y, w, mode='full')  # length len(y) + len(w) - 1
    start = len(w) // 2
    end = start + len(y)
    return full[start:end]                 # exactly len(y)


def make_rate_df_from_binned(
    binned_spikes2: pd.DataFrame,
    unit_col: str | int,
    *,
    stop_col: str = 'stop_id',
    time_col: str = 'rel_center',
    left_col: str = 't_left',
    right_col: str = 't_right',
    keep_cols: Optional[Sequence[str]] = ('bin',)
) -> pd.DataFrame:
    df = binned_spikes2.copy()
    if unit_col not in df.columns:
        # allow int index passed in; convert to str if your columns are ints-as-names
        unit_col = int(unit_col)
        if unit_col not in df.columns:
            raise KeyError(f'Unit column {unit_col!r} not found.')

    # exposure per bin (seconds)
    exp = (df[right_col].to_numpy(dtype=float) - df[left_col].to_numpy(dtype=float))
    exp[~np.isfinite(exp)] = 0.0

    # rate in Hz
    counts = df[unit_col].to_numpy(dtype=float)
    rate = _safe_div(counts, exp, fill=0.0)

    out_cols = [stop_col, time_col]
    if keep_cols:
        for k in keep_cols:
            if k in df.columns and k not in out_cols:
                out_cols.append(k)

    out = df[out_cols].copy()
    out['exposure_s'] = exp
    out['spike_count'] = counts
    out['rate_hz'] = rate
    return out

def plot_spaghetti_per_stop(
    df_rate: pd.DataFrame,
    *,
    stop_col: str = 'stop_id',
    time_col: str = 'rel_center',
    rate_col: str = 'rate_hz',
    smooth_sigma_bins: Optional[float] = None,
    smooth_sigma_s: Optional[float] = None,
    bin_width_s_hint: Optional[float] = None,
    baseline_window: Optional[tuple[float, float]] = None,
    max_stops: Optional[int] = None,
    alpha: float = 0.3,
    lw: float = 1.2,
    show_median: bool = True,
    median_lw: float = 2.2,
    median_label: str = 'median across stops',
    title: str = 'Firing rate per stop (one line per stop)',
    xlabel: str = 'Time from stop (s)',
    ylabel: str = 'Rate (Hz)'
):
    # infer bin width if needed for smoothing in seconds
    if smooth_sigma_s is not None and smooth_sigma_bins is None:
        if bin_width_s_hint is not None:
            bw = bin_width_s_hint
        else:
            # robust guess from within-stop diffs
            tmp = df_rate.sort_values([stop_col, time_col])
            diffs = tmp.groupby(stop_col)[time_col].diff().dropna().to_numpy()
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            bw = np.median(diffs) if diffs.size else 0.04
        smooth_sigma_bins = smooth_sigma_s / max(bw, 1e-9)

    # optionally downselect stops
    stops = df_rate[stop_col].unique().tolist()
    if max_stops is not None and len(stops) > max_stops:
        stops = stops[:max_stops]
    g = df_rate[df_rate[stop_col].isin(stops)].copy()

    # plot each stop
    fig, ax = plt.subplots(figsize=(8, 5))
    lines_plotted = 0
    for sid, df_s in g.groupby(stop_col, sort=True):
        y = df_s.sort_values(time_col)
        yv = y[rate_col].to_numpy()
        if baseline_window is not None:
            t0, t1 = baseline_window
            mask = (y[time_col].to_numpy() >= t0) & (y[time_col].to_numpy() < t1)
            base = yv[mask].mean() if mask.any() else 0.0
            yv = yv - base
        if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
            yv = _gaussian_smooth_1d(yv, smooth_sigma_bins)
        ax.plot(y[time_col].to_numpy(), yv, alpha=alpha, lw=lw)
        lines_plotted += 1

    # median across stops at each time (works if time grid is common; otherwise still a useful pooled summary)
    if show_median:
        med = g.groupby(time_col)[rate_col].median().reset_index().sort_values(time_col)
        yv = med[rate_col].to_numpy()
        if baseline_window is not None:
            t0, t1 = baseline_window
            mask = (med[time_col].to_numpy() >= t0) & (med[time_col].to_numpy() < t1)
            base = yv[mask].mean() if mask.any() else 0.0
            yv = yv - base
        if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
            yv = _gaussian_smooth_1d(yv, smooth_sigma_bins)
        ax.plot(med[time_col].to_numpy(), yv, lw=median_lw, label=median_label)

    ax.axvline(0.0, ls='--', lw=1.0)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if show_median:
        ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig, ax, lines_plotted


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_observed_vs_predicted_stop(
    binned_feats_sc: pd.DataFrame,
    binned_spikes: pd.DataFrame,
    meta_used: pd.DataFrame,
    offset_log,                        # 1D np.ndarray or pd.Series aligned to rows
    model_res,                         # statsmodels GLMResults(_wrapper) for THIS cluster
    cluster_idx: int | str,            # column label or positional index in binned_spikes
    stop_id: int,
    *,
    time_col: str = 'rel_center',
    exposure_s: pd.Series | np.ndarray | None = None,  # if None, derived from offset_log
    sort_by_time: bool = True,
    title_prefix: str = 'Observed vs Predicted',
    ax: plt.Axes | None = None,
):
    """
    Plot observed vs predicted firing rate (Hz) for one cluster and one stop,
    correctly handling variable per-bin exposure.

    Assumptions
    ----------
    - Poisson GLM with log link.
    - If exposure_s is None, offset_log = log(exposure_s).

    Parameters
    ----------
    binned_feats_sc : DataFrame
        The exact feature matrix used to FIT the model (same scaling & columns).
    binned_spikes : DataFrame
        Shape (n_bins, n_clusters), integer spike counts per bin.
    meta_used : DataFrame
        Must include 'stop_id' and the time_col (e.g., 'rel_time').
    offset_log : 1D array-like
        Log-exposure aligned 1:1 with rows of binned_feats_sc/meta_used.
    model_res : GLMResults
        Fitted GLM for this cluster.
    cluster_idx : int | str
        Column index or column name in binned_spikes for the neuron.
    stop_id : int
        Stop/segment to visualize.
    time_col : str
        Column in meta_used for the x-axis time (relative to stop).
    exposure_s : array-like | None
        Per-bin exposure in seconds. If None, uses exp(offset_log).
    sort_by_time : bool
        If True, sort by time before plotting (cleaner lines).
    title_prefix : str
        Title prefix for the plot.
    ax : matplotlib.axes.Axes | None
        If provided, draw into this axes; otherwise create a new figure.

    Returns
    -------
    tidy : DataFrame
        Columns: ['time_s', 'obs_hz', 'pred_hz', 'exposure_s', 'cluster', 'stop_id'].
    """
    n = len(meta_used)
    if binned_feats_sc.shape[0] != n or binned_spikes.shape[0] != n:
        raise ValueError('Row counts must match across features, spikes, and meta.')

    if 'stop_id' not in meta_used.columns:
        raise ValueError('meta_used must contain a "stop_id" column.')
    if time_col not in meta_used.columns:
        raise ValueError(f'meta_used must contain "{time_col}".')

    offset_log = np.asarray(offset_log).reshape(-1)
    if offset_log.shape[0] != n:
        raise ValueError('offset_log length must match rows of meta_used.')

    if exposure_s is not None:
        exposure_s = np.asarray(exposure_s).reshape(-1)
        if exposure_s.shape[0] != n:
            raise ValueError('exposure_s length must match rows of meta_used.')

    # --- mask rows for this stop
    mask = (meta_used['stop_id'].to_numpy() == stop_id)
    if not np.any(mask):
        raise ValueError(f'No rows found for stop_id={stop_id}.')

    # --- slice aligned views
    X_stop_full = binned_feats_sc.loc[mask]
    if isinstance(cluster_idx, (int, np.integer)):
        y_counts = binned_spikes.loc[mask].iloc[:, int(cluster_idx)]
        cluster_label = binned_spikes.columns[int(cluster_idx)]
    else:
        y_counts = binned_spikes.loc[mask, cluster_idx]
        cluster_label = cluster_idx
    off_stop = offset_log[mask]

    # --- resolve exposure (seconds) per bin
    if exposure_s is None:
        exp_s = np.exp(off_stop)
    else:
        exp_s = exposure_s[mask]

    # --- guard against invalid exposure and keep all arrays aligned
    good = np.isfinite(exp_s) & (exp_s > 0)
    if not good.all():
        X_stop_full = X_stop_full.loc[good]
        y_counts = y_counts.loc[good]
        off_stop = off_stop[good]
        exp_s = exp_s[good]


    # --- predictions: expected COUNTS per bin
    pred_counts = model_res.predict(X_stop_full, offset=off_stop)

    # --- convert to Hz using per-bin exposure
    obs_rate_hz = y_counts.to_numpy() / exp_s
    pred_rate_hz = pred_counts.to_numpy() / exp_s

    # --- time axis (apply same "good" filter, then optional sort)
    t = meta_used.loc[mask, time_col].to_numpy()
    t = t[good]

    if sort_by_time:
        order = np.argsort(t)
        t = t[order]
        obs_rate_hz = obs_rate_hz[order]
        pred_rate_hz = pred_rate_hz[order]
        exp_s_plot = exp_s[order]
    else:
        exp_s_plot = exp_s

    # --- plot
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, obs_rate_hz, 'o-', label='Observed (Hz)', alpha=0.7)
    ax.plot(t, pred_rate_hz, '-', label='Predicted (Hz)', lw=2)
    ax.axvline(0, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Time relative to stop (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'{title_prefix} • cluster {cluster_label} • stop {stop_id}')
    ax.legend()
    plt.tight_layout()

    return pd.DataFrame({
        'time_s': t,
        'obs_hz': obs_rate_hz,
        'pred_hz': pred_rate_hz,
        'exposure_s': exp_s_plot,
        'cluster': cluster_label,
        'stop_id': stop_id,
    })
