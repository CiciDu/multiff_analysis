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
