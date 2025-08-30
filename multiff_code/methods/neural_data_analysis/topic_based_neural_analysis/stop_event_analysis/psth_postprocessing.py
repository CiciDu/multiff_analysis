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
        
    if title is None:
        title = f"{getattr(analyzer,'event_a_label','event_a')} − {getattr(analyzer,'event_b_label','event_b')} effects (Cohen's d)"

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d (event_a − event_b)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cluster")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax
