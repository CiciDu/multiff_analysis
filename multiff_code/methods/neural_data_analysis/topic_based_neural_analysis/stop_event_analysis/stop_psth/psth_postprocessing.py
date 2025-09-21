from matplotlib.colors import TwoSlopeNorm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def export_psth_to_df(
    analyzer: "PSTHAnalyzer",
    clusters: Optional[List[int]] = None,
    include_ci: bool = True,
) -> pd.DataFrame:
    """
    Flatten cached PSTH results into a tidy DataFrame.

    Parameters
    ----------
    analyzer : PSTHAnalyzer
        Instance that has already run .run_full_analysis() (will run if needed).
    clusters : list of int or None
        Optional subset of cluster indices (0-based, matching analyzer.clusters' order).
        If None, export all clusters.
    include_ci : bool
        If True, include 'sem', 'lower', 'upper' columns.

    Returns
    -------
    pd.DataFrame with columns:
        ['time', 'cluster', 'condition', 'mean']  (+ ['sem','lower','upper'] if include_ci)

    Notes
    -----
    - 'cluster' column uses the original cluster IDs (analyzer.clusters values),
      not the 0-based column indices.
    - 'mean' and 'sem' reflect whatever normalization and CI method were configured.
    """
    # Ensure results exist
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis()

    psth = analyzer.psth_data["psth"]  # type: ignore[assignment]
    time = psth["time_axis"]
    all_idx = range(len(analyzer.clusters))
    idxs = list(all_idx) if clusters is None else list(clusters)

    label_a = getattr(analyzer, "event_a_label", "event_a")
    label_b = getattr(analyzer, "event_b_label", "event_b")

    # Helper to build one condition block
    def _block(cond_key: str, label: str) -> pd.DataFrame:
        mean = psth[cond_key][:, idxs]           # (n_bins, n_sel_clusters)
        sem = psth[cond_key + "_sem"][:, idxs]   # same shape

        # Repeat time for each selected cluster
        t_rep = np.tile(time[:, None], (1, len(idxs)))
        cl_ids = analyzer.clusters[idxs]
        cl_rep = np.tile(cl_ids[None, :], (len(time), 1))

        data = {
            "time": t_rep.ravel(),
            "cluster": cl_rep.ravel(),
            "condition": np.full(t_rep.size, label, dtype=object),
            "mean": mean.ravel(),
        }
        if include_ci:
            data["sem"] = sem.ravel()
            data["lower"] = (mean - sem).ravel()
            data["upper"] = (mean + sem).ravel()

        return pd.DataFrame(data)

    out_frames: List[pd.DataFrame] = []
    # Only append if there are any events for that condition
    if analyzer.psth_data["n_events"]["event_a"] > 0:
        out_frames.append(_block("event_a", label_a))
    if analyzer.psth_data["n_events"]["event_b"] > 0:
        out_frames.append(_block("event_b", label_b))

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(
        columns=(["time", "cluster", "condition", "mean"] +
                 (["sem", "lower", "upper"] if include_ci else []))
    )


def _bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR correction.
    Returns array of booleans for which hypotheses are rejected.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.arange(1, n+1)
    thresh = alpha * ranked / n
    passed = p[order] <= thresh
    # make it monotone
    if np.any(passed):
        kmax = np.max(np.where(passed)[0])
        passed[:kmax+1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = passed
    return out


def compare_windows(analyzer, windows, alpha=0.05):
    """
    Run analyzer.statistical_comparison() on multiple windows and
    return a tidy DataFrame with FDR-corrected significance flags.

    Returns columns:
      cluster, window, p, U, cohens_d,
      event_a_mean, event_b_mean, n_event_a, n_event_b, sig_FDR
    """
    rows = []
    for win_name, (a, b) in windows.items():
        stats_out = analyzer.statistical_comparison(time_window=(a, b))
        for cl_id, d in stats_out.items():
            if "error" in d:
                rows.append({
                    "cluster": cl_id, "window": win_name, "p": np.nan,
                    "U": np.nan, "cohens_d": np.nan,
                    "event_a_mean": np.nan, "event_b_mean": np.nan,
                    "n_event_a": d.get("n_event_a", 0), "n_event_b": d.get("n_event_b", 0),
                    "sig_FDR": False
                })
            else:
                rows.append({
                    "cluster": cl_id, "window": win_name, "p": d["p_value"],
                    "U": d["statistic_U"], "cohens_d": d["cohens_d"],
                    "event_a_mean": d["event_a_mean"], "event_b_mean": d["event_b_mean"],
                    "n_event_a": d["n_event_a"], "n_event_b": d["n_event_b"],
                    "sig_FDR": False  # temp, fill later
                })
    df = pd.DataFrame(rows)

    # FDR within each window across clusters
    out = []
    for w, g in df.groupby("window", dropna=False):
        mask = g["p"].notna()
        sig = np.full(len(g), False)
        if mask.any():
            sig_indices = _bh_fdr(g.loc[mask, "p"].values, alpha=alpha)
            sig[np.where(mask)[0]] = sig_indices
        gg = g.copy()
        gg.loc[:, "sig_FDR"] = sig
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def summarize_epochs(analyzer, alpha=0.05):
    """
    Run statistical comparisons across three canonical epochs:
    - pre_bump (-0.3–0.0 s)
    - early_dip (0.0–0.3 s)
    - late_rebound (0.3–0.8 s)

    Returns
    -------
    pd.DataFrame with columns:
      cluster, window, p, cohens_d, event_a_mean, event_b_mean,
      n_event_a, n_event_b, sig_FDR
    """
    windows = {
        "pre_bump(-0.3–0.0)": (-0.3, 0.0),
        "early_dip(0.0–0.3)": (0.0, 0.3),
        "late_rebound(0.3–0.8)": (0.3, 0.8),
    }
    return compare_windows(analyzer, windows, alpha=alpha)


def plot_effect_heatmap_all(summary: pd.DataFrame,
                            analyzer,
                            title=None,
                            grey_color="lightgrey",
                            order: str = "effect"):
    """
    Heatmap of Cohen's d across clusters × epochs.
    - Significant cells (FDR) show d with a diverging colormap.
    - Non-significant cells are grey.
    - Includes ALL clusters, even if nothing is significant.

    Parameters
    ----------
    summary : DataFrame
        Output from summarize_epochs()/compare_windows().
        Must have columns ['cluster','window','cohens_d','sig_FDR'].
    title : str
        Plot title.
    grey_color : str
        Color for non-significant cells.
    order : {'effect','cluster'}
        'effect': sort rows by strongest |d| among significant cells (descending).
        'cluster': sort rows by cluster label.

    Returns
    -------
    (fig, ax)
    """
    df = summary.copy()

    # Keep all rows but mask non-significant d as NaN (so they render grey)
    d_masked = df["cohens_d"].where(df["sig_FDR"], np.nan)
    df = df.assign(d_masked=d_masked)

    # Pivot to clusters × windows (may contain NaNs)
    pivot = df.pivot_table(index="cluster",
                           columns="window",
                           values="d_masked",
                           aggfunc="mean")

    # Ensure ALL clusters/windows appear (even if all NaN)
    all_clusters = pd.Index(df["cluster"].astype(str).unique(), name="cluster")
    all_windows = pd.Index(df["window"].unique(), name="window")
    pivot = pivot.reindex(index=all_clusters, columns=all_windows)

    # Row ordering
    if order == "effect":
        strength = pivot.abs().max(axis=1).fillna(0.0)
        pivot = pivot.loc[strength.sort_values(ascending=False).index]
    elif order == "cluster":
        pivot = pivot.sort_index()
    else:
        raise ValueError("order must be 'effect' or 'cluster'")

    # Colormap with grey for NaNs
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(grey_color)

    # Symmetric color scale by max |d| among significant cells
    vmax = np.nanmax(np.abs(pivot.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0  # fallback if nothing significant at all

    event_a_label = getattr(analyzer, 'event_a_label', 'event_a')
    event_b_label = getattr(analyzer, 'event_b_label', 'event_b')

    if title is None:
        title = f"{event_a_label} − {event_b_label} effects (Cohen's d)"

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d ({event_a_label} − {event_b_label})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cluster")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax


def _pick_sample_sizes(df: pd.DataFrame) -> tuple[int | None, int | None]:
    nA = nB = None
    if 'n_event_a' in df.columns and df['n_event_a'].notna().any():
        nA = int(pd.to_numeric(df['n_event_a'],
                 errors='coerce').dropna().iloc[0])
    if 'n_event_b' in df.columns and df['n_event_b'].notna().any():
        nB = int(pd.to_numeric(df['n_event_b'],
                 errors='coerce').dropna().iloc[0])
    return nA, nB


def _shorten_labels(seq):
    # turn 'pre_bump(…)' -> 'pre bump\n(…)' to save horizontal space
    out = []
    for s in map(str, seq):
        if '(' in s and '_' in s:
            left, right = s.split('(', 1)
            out.append(left.replace('_', ' ') + '\n(' + right)
        else:
            out.append(s.replace('_', ' '))
    return out


def _extract_paren_content(seq):
    out = []
    for s in map(str, seq):
        if '(' in s and ')' in s:
            left = s.find('(')
            right = s.rfind(')')
            if right > left:
                out.append(s[left+1:right])
                continue
        out.append(s.replace('_', ' '))
    return out


def plot_sig_heatmap(summary: pd.DataFrame,
                     title: str | None = None,
                     window_order: list | None = None,
                     cmap: str = 'coolwarm',
                     drop_empty_windows: bool = False,
                     show_y_tick_label: bool = False,
                     xtick_rotation: int = 30,
                     strip_prefix: bool = False,
                     show_sample_size: bool = False,
                     dpi=300,
                     ):
    """
    Heatmap of FDR-significant Cohen's d by cluster × window.
    - Title and sample size are combined into a single two-line suptitle.
    - Explicit observed=… passed to pivot_table to silence FutureWarning.

    summary needs: ['cluster','window','cohens_d','sig_FDR'] and (optionally) n_event_a/b.
    """
    if title is None:
        title = "Significant effects (Cohen's d)"

    nA, nB = _pick_sample_sizes(summary)
    if show_sample_size:
        subtitle = f'sample sizes: {nA} and {nB}' if (
            nA is not None and nB is not None) else None
    else:
        subtitle = None

    sig = summary[summary['sig_FDR']].copy()

    # no significant results → compact info figure (with same header style)
    if sig.empty:
        fig, ax = plt.subplots(figsize=(7.5, 2.6))
        ax.axis('off')
        header = title if not subtitle else f'{title}\n{subtitle}'
        fig.suptitle(header, fontsize=13)
        ax.text(0.5, 0.58, 'No significant results to plot.',
                ha='center', va='center', fontsize=11)
        ax.text(0.5, 0.30, 'Try more trials, different windows, or relaxed corrections.',
                ha='center', va='center', fontsize=9)
        plt.subplots_adjust(top=0.78)
        return fig, ax

    if window_order is not None:
        sig['window'] = pd.Categorical(
            sig['window'], categories=window_order, ordered=True)

    # pivot clusters × windows; choose whether to keep unobserved categories
    observed_flag = True if drop_empty_windows else False
    pivot = sig.pivot_table(index='cluster',
                            columns='window',
                            values='cohens_d',
                            aggfunc='mean',
                            observed=observed_flag)

    # optionally drop windows that are all NaN
    if drop_empty_windows:
        pivot = pivot.loc[:, pivot.notna().any(axis=0)]

    # enforce column order
    if window_order is not None:
        cols = [
            w for w in window_order if w in pivot.columns] if drop_empty_windows else window_order
        pivot = pivot.reindex(columns=cols)

    # sort clusters by strongest absolute effect
    with np.errstate(invalid='ignore'):
        strength = pivot.abs().max(axis=1).to_numpy()
    order_idx = np.argsort(-np.nan_to_num(strength, nan=-np.inf))
    pivot = pivot.iloc[order_idx]

    # symmetric color scale
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # figure
    fig_h = min(max(3.2, 0.42 * max(1, len(pivot))), 5)
    fig, ax = plt.subplots(figsize=(9.0, fig_h), dpi=dpi)

    im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(pivot.shape[1]))
    if strip_prefix:
        ax.set_xticklabels(_extract_paren_content(
            pivot.columns), rotation=xtick_rotation, ha='right')
    else:
        ax.set_xticklabels(_shorten_labels(pivot.columns),
                           rotation=xtick_rotation, ha='right')

    if show_y_tick_label:
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # single two-line header avoids collisions
    header = title if not subtitle else f'{title}\n{subtitle}'
    fig.suptitle(header, fontsize=17, y=0.95)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    cbar.set_label('Standardized\nmean difference', rotation=0, labelpad=40, fontsize=12)
    # cbar.set_label("Cohen's d (event_a − event_b)")

    # reserve space for the two-line suptitle
    if subtitle:
        plt.subplots_adjust(top=0.80)
    else:
        plt.subplots_adjust(top=0.85)

    ax.set_ylabel('Neurons', fontsize=14)
    ax.set_xlabel('Window (in seconds)', fontsize=14)

    # find the column whose label spans 0
    zero_col = [i for i, w in enumerate(pivot.columns) if '0.0' in str(w)]
    if zero_col:
        ax.axvline(x=zero_col[0], color='k', linestyle='--', linewidth=1)

    return fig, ax
