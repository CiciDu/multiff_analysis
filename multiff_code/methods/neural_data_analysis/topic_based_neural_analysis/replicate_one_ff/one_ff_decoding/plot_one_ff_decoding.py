from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_parameters

def _get_canoncorr_block(stats_or_block: Dict) -> Dict:
    if "coeff" in stats_or_block:
        return stats_or_block
    return stats_or_block.get("trialtype", {}).get("all", {}).get("canoncorr", {})


def plot_canoncorr_coefficients(
    stats_or_block: Dict,
    *,
    figsize: Tuple[float, float] = (6.5, 3.5),
):
    """
    Plot canonical correlations from canoncorr output.
    """
    block = _get_canoncorr_block(stats_or_block)
    coeff = np.asarray(block.get("coeff", []), dtype=float)
    if coeff.size == 0:
        raise ValueError("No canoncorr coefficients found. Run compute_canoncorr() first.")

    x = np.arange(1, coeff.size + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, coeff, marker="o", lw=1.8)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Correlation")
    title = "Canonical correlations"
    if "dimensionality" in block and np.isfinite(block["dimensionality"]):
        title += f" (dim={block['dimensionality']:.2f})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def _get_parity_varnames(readout_block: Dict) -> list:
    """Variables with true/pred for parity plot (top-level or trials)."""
    out = []
    for k, v in readout_block.items():
        if not isinstance(v, dict):
            continue
        if "true" in v and "pred" in v:
            out.append(k)
        elif "trials" in v and "true" in v.get("trials", {}) and "pred" in v.get("trials", {}):
            out.append(k)
    return out


def plot_decoder_parity(
    readout_block: Dict,
    *,
    varnames: Optional[Sequence[str]] = None,
    max_points: int = 5000,
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (3.2, 3.0),
):
    """
    Plot true vs predicted parity for decoded variables.
    Supports both top-level true/pred and trials.true/pred (concatenated).
    """
    if varnames is None:
        varnames = _get_parity_varnames(readout_block)
    varnames = list(varnames)
    if len(varnames) == 0:
        raise ValueError("No decodable variables found in readout block.")

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
        if "true" in entry and "pred" in entry:
            y_true = np.asarray(entry["true"], dtype=float).ravel()
            y_pred = np.asarray(entry["pred"], dtype=float).ravel()
        else:
            tr = entry.get("trials", {})
            y_true = np.concatenate([np.asarray(t, dtype=float).ravel() for t in tr.get("true", [])])
            y_pred = np.concatenate([np.asarray(t, dtype=float).ravel() for t in tr.get("pred", [])])
        if y_true.size > max_points:
            idx = np.linspace(0, y_true.size - 1, max_points).astype(int)
            y_true = y_true[idx]
            y_pred = y_pred[idx]
        corr = entry.get("corr", np.nan)
        ax.scatter(y_true, y_pred, s=6, alpha=0.35)
        lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{var} (r={corr:.2f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(alpha=0.2)

    for j in range(n_panels, len(flat_axes)):
        fig.delaxes(flat_axes[j])
    cv_str = _get_cv_config_str(readout_block)
    if cv_str:
        fig.suptitle("True vs predicted" + cv_str, y=1.02, fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_decoded_trajectory(
    readout_block: Dict,
    *,
    trial_idx: int = 0,
    source: str = "xt_from_vw",
    figsize: Tuple[float, float] = (4.5, 4.0),
):
    """
    Plot one decoded trajectory trial.

    source: "xt_from_vw" or "xt"
    """
    if source not in {"xt_from_vw", "xt"}:
        raise ValueError("source must be 'xt_from_vw' or 'xt'")
    x_trials = readout_block.get(source, {}).get("trials", {}).get("pred", [])
    y_key = "yt_from_vw" if source == "xt_from_vw" else "yt"
    y_trials = readout_block.get(y_key, {}).get("trials", {}).get("pred", [])
    if len(x_trials) == 0 or len(y_trials) == 0:
        raise ValueError(f"No decoded trajectories found for source='{source}'.")
    if trial_idx < 0 or trial_idx >= len(x_trials):
        raise IndexError(f"trial_idx out of bounds: {trial_idx}, n_trials={len(x_trials)}")

    x = np.asarray(x_trials[trial_idx], dtype=float)
    y = np.asarray(y_trials[trial_idx], dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, lw=2)
    ax.scatter(x[0], y[0], s=30, c="green", label="start")
    ax.scatter(x[-1], y[-1], s=30, c="red", label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cv_str = _get_cv_config_str(readout_block)
    ax.set_title(f"Decoded trajectory ({source}), trial {trial_idx}" + cv_str)
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def _get_cv_config_str(readout_block: Dict) -> str:
    """Return a short string describing cv config for plot titles."""
    cfg = readout_block.get("_cv_config", {})
    if not cfg:
        return ""
    cv = cfg.get("cv_mode", "")
    n = cfg.get("n_splits", "")
    buf = cfg.get("buffer_samples", "")
    if cv or n or buf:
        return f" [cv={cv}, n={n}, buf={buf}]"
    return ""


def plot_decoder_correlation_bars(
    readout_block: Dict,
    *,
    varnames: Optional[Sequence[str]] = None,
    sort: bool = True,
    figsize: Tuple[float, float] = (6.5, 5.5),
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
    ax.set_title("Decoder performance by variable")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_fontsize(9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    plt.show()


def plot_all_decoding_results(
    *,
    canoncorr_block: Optional[Dict] = None,
    readout_block: Optional[Dict] = None,
    parity_varnames: Optional[Sequence[str]] = None,
    bar_varnames: Optional[Sequence[str]] = None,
    trial_idx: int = 0,
    traj_source: str = "xt_from_vw",
):
    """
    Convenience function to call all decoding plotting functions.
    """
    out = {}

    if canoncorr_block is not None:
        try:
            out["canoncorr"] = plot_canoncorr_coefficients(canoncorr_block)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip canoncorr: result doesn't exist ({e})")

    if readout_block is not None:
        try:
            out["parity"] = plot_decoder_parity(readout_block, varnames=parity_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip parity: result doesn't exist ({e})")
        try:
            out["corr_bars"] = plot_decoder_correlation_bars(readout_block, varnames=bar_varnames)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip corr_bars: result doesn't exist ({e})")
        try:
            out["trajectory"] = plot_decoded_trajectory(
                readout_block,
                trial_idx=trial_idx,
                source=traj_source,
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"[plot_all_decoding_results] skip trajectory: result doesn't exist ({e})")

    return out


def plot_single_trial_decoding_panel(
    readout_block: Dict,
    *,
    trial_indices: Optional[Sequence[int]] = None,
    n_trials: int = 6,
    figsize: Tuple[float, float] = (10.0, 6.6),
):
    """
    Plot compact single-trial decoding traces for canoncorr variables.
    """
    varnames = list(one_ff_parameters.default_prs().canoncorr_varname)
    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(varnames), 1)))
    row_defs = []
    for i, var in enumerate(varnames):
        c_pred = tuple(palette[i][:3])
        c_true = tuple(np.clip(np.asarray(c_pred) * 0.65, 0.0, 1.0))
        row_defs.append((var, var, var, c_pred, c_true))
    available = [r for r in row_defs if r[0] in readout_block and "trials" in readout_block[r[0]]]
    if len(available) == 0:
        raise ValueError("No compatible decoded trial results found for panel plot.")

    # Infer valid trial range from first available variable.
    first_var = available[0][0]
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

    n_rows = len(available)
    n_cols = len(trial_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )

    # Style similar to reference panel.
    fig.patch.set_facecolor("white")

    for r, (var, _, ylabel_txt, color_pred, color_true) in enumerate(available):
        trials_true = readout_block[var]["trials"]["true"]
        trials_pred = readout_block[var]["trials"]["pred"]
        for c, tid in enumerate(trial_indices):
            ax = axes[r, c]
            y_true = np.asarray(trials_true[tid], dtype=float)
            y_pred = np.asarray(trials_pred[tid], dtype=float)
            if y_true.size == 0 or y_pred.size == 0:
                ax.axis("off")
                continue

            x = np.arange(min(y_true.size, y_pred.size))
            y_true = y_true[: x.size]
            y_pred = y_pred[: x.size]
            ax.plot(x, y_true, color=color_true, lw=2.0, alpha=0.9)
            ax.plot(x, y_pred, color=color_pred, lw=2.4, alpha=0.95)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if c == 0:
                # Left-side scale text per row.
                ax.text(
                    -0.32,
                    0.5,
                    ylabel_txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    color="black",
                )

    fig.suptitle("Single-trial decoding", fontsize=24, y=0.98)
    fig.tight_layout(rect=[0.07, 0.02, 0.99, 0.95])
    return fig, axes
