from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding


# This has yet to be verified as working
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
            plot_one_ff_decoding.plot_canoncorr_coefficients(canoncorr_block)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip canoncorr: result doesn't exist ({e})")
    if readout_block is not None:
        try:
            plot_one_ff_decoding.plot_decoder_parity(readout_block, varnames=parity_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip parity: result doesn't exist ({e})")
        try:
            plot_one_ff_decoding.plot_decoder_correlation_bars(readout_block, varnames=bar_varnames)
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
