from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_canoncorr_coefficients(
    canoncorr_block: Dict,
    *,
    figsize: Tuple[float, float] = (6.5, 3.5),
):
    coeff = np.asarray(canoncorr_block.get("coeff", []), dtype=float)
    if coeff.size == 0:
        raise ValueError("No canoncorr coefficients found.")
    x = np.arange(1, coeff.size + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, coeff, marker="o", lw=1.8)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Correlation")
    title = "Stop decoding canonical correlations"
    dim = canoncorr_block.get("dimensionality", np.nan)
    if np.isfinite(dim):
        title += f" (dim={dim:.2f})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def plot_decoder_parity(
    readout_block: Dict,
    *,
    varnames: Optional[Sequence[str]] = None,
    max_points: int = 5000,
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (3.0, 2.8),
):
    if varnames is None:
        varnames = [k for k, v in readout_block.items() if isinstance(v, dict) and ("true" in v and "pred" in v)]
    varnames = list(varnames)
    varnames = sorted(
        varnames,
        key=lambda k: np.nan_to_num(float(readout_block.get(k, {}).get("corr", np.nan)), nan=-np.inf),
        reverse=True,
    )
    if len(varnames) == 0:
        raise ValueError(
            "No decodable variables with true/pred found. "
            "Re-run regress_popreadout with save_predictions=True to enable parity plots."
        )

    n_panels = len(varnames)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for i, var in enumerate(varnames):
        ax = flat_axes[i]
        entry = readout_block[var]
        y_true = np.asarray(entry["true"], dtype=float)
        y_pred = np.asarray(entry["pred"], dtype=float)
        if y_true.size > max_points:
            idx = np.linspace(0, y_true.size - 1, max_points).astype(int)
            y_true = y_true[idx]
            y_pred = y_pred[idx]
        corr = float(entry.get("corr", np.nan))
        ax.scatter(y_true, y_pred, s=5, alpha=0.35)
        lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{var} (r={corr:.2f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(alpha=0.2)

    for j in range(n_panels, len(flat_axes)):
        fig.delaxes(flat_axes[j])
    fig.tight_layout()
    plt.show()


def plot_decoder_correlation_bars(
    readout_block: Dict,
    *,
    varnames: Optional[Sequence[str]] = None,
    sort: bool = True,
    figsize: Tuple[float, float] = (6.5, 3.5),
):
    if varnames is None:
        varnames = [k for k, v in readout_block.items() if isinstance(v, dict) and ("corr" in v)]
    varnames = list(varnames)
    if len(varnames) == 0:
        raise ValueError("No decoder correlations found in readout block.")

    pairs = [(v, float(readout_block[v].get("corr", np.nan))) for v in varnames]
    if sort:
        pairs = sorted(pairs, key=lambda x: (np.nan_to_num(x[1], nan=-np.inf)), reverse=True)
    names = [p[0] for p in pairs]
    vals = np.array([p[1] for p in pairs], dtype=float)

    # Moderate width scaling with a cap to keep plot size reasonable.
    n = len(names)
    fig_w = min(12.0, max(6.0, 5.0 + 0.30 * n))
    fig_h = max(3.2, figsize[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.bar(names, vals)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("corr(true, pred)")
    ax.set_title("Stop decoder performance by variable")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_fontsize(9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    plt.show()


def plot_single_trial_decoding_panel(
    readout_block: Dict,
    *,
    trial_indices: Optional[Sequence[int]] = None,
    n_trials: int = 6,
    figsize: Optional[Tuple[float, float]] = None,
):
    varnames = [k for k, v in readout_block.items() if isinstance(v, dict) and ("trials" in v)]
    if len(varnames) == 0:
        raise ValueError(
            "No decoded trial series found. "
            "Re-run regress_popreadout with save_predictions=True to enable trial plots."
        )

    first_var = varnames[0]
    n_total_trials = len(readout_block[first_var]["trials"]["true"])
    if n_total_trials == 0:
        raise ValueError("Decoded trials are empty.")

    if trial_indices is None:
        n_show = min(n_trials, n_total_trials)
        trial_indices = np.linspace(0, n_total_trials - 1, n_show).astype(int).tolist()
    else:
        trial_indices = [int(i) for i in trial_indices if 0 <= int(i) < n_total_trials]
    if len(trial_indices) == 0:
        raise ValueError("No valid trial indices to plot.")

    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(varnames), 1)))
    n_rows, n_cols = len(varnames), len(trial_indices)

    # Auto-size panels for readability across different row/column counts.
    if figsize is None:
        panel_w = 1.45
        panel_h = 0.9
        fig_w = max(8.0, min(20.0, 2.2 + panel_w * n_cols))
        fig_h = max(4.5, min(18.0, 1.6 + panel_h * n_rows))
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, var in enumerate(varnames):
        c_pred = tuple(palette[r][:3])
        c_true = tuple(np.clip(np.asarray(c_pred) * 0.65, 0.0, 1.0))
        t_true = readout_block[var]["trials"]["true"]
        t_pred = readout_block[var]["trials"]["pred"]

        for c, tid in enumerate(trial_indices):
            ax = axes[r, c]
            y_true = np.asarray(t_true[tid], dtype=float)
            y_pred = np.asarray(t_pred[tid], dtype=float)
            if y_true.size == 0 or y_pred.size == 0:
                ax.axis("off")
                continue
            x = np.arange(min(y_true.size, y_pred.size))
            y_true = y_true[: x.size]
            y_pred = y_pred[: x.size]
            ax.plot(x, y_true, color=c_true, lw=1.8, alpha=0.9)
            ax.plot(x, y_pred, color=c_pred, lw=2.2, alpha=0.95)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.text(
                    -0.18,
                    0.5,
                    var,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                )

    fig.suptitle("Single-trial stop decoding", fontsize=18, y=0.99)
    fig.tight_layout(rect=[0.12, 0.02, 0.995, 0.96], h_pad=0.35, w_pad=0.22)
    plt.show()


def plot_all_decoding_results(
    *,
    canoncorr_block: Optional[Dict] = None,
    readout_block: Optional[Dict] = None,
    parity_varnames: Optional[Sequence[str]] = None,
    bar_varnames: Optional[Sequence[str]] = None,
    trial_indices: Optional[Sequence[int]] = None,
    n_trials: int = 6,
):
    if canoncorr_block is not None:
        try:
            plot_canoncorr_coefficients(canoncorr_block)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip canoncorr: result doesn't exist ({e})")
    if readout_block is not None:
        try:
            plot_decoder_parity(readout_block, varnames=parity_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip parity: result doesn't exist ({e})")
        try:
            plot_decoder_correlation_bars(readout_block, varnames=bar_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip corr_bars: result doesn't exist ({e})")
        try:
            plot_single_trial_decoding_panel(
                readout_block,
                trial_indices=trial_indices,
                n_trials=n_trials,
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip single_trial_panel: result doesn't exist ({e})")
