"""
Standalone PSTH stats utilities for event_a vs event_b comparisons.

Drop this file next to your PSTH analyzer. It operates on a ready-to-use
`PSTHAnalyzer` instance (from your existing script) without modifying it.

Provided functions:
- statistical_comparison_window(...): richer stats for a single window (abs or percent)
- statistical_comparison_by_windows(...): batch over named windows (abs + percent)
- sliding_window_stats(...): sweep fixed-width windows across time
- permutation_time_cluster_test(...): Maris–Oostenveld cluster permutation in time

All return tidy dicts/DataFrames for easy downstream analysis.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ----------------------------- Small helpers ---------------------------------
def quick_report(an, window="late_rebound(0.3–0.8)"):
    fig = an.plot_comparison()
    plt.show()
    res = statistical_comparison_window(an, window_name=window)
    # pretty-print
    for k, v in res.items():
        if "error" in v:
            print(f"Cluster {k}: {v['error']}")
        else:
            print(f"Cluster {k}: p={v['p_value']:.3g}  AUC={v['AUC']:.3f}  δ={v['cliffs_delta']:.3f}  g={v['hedges_g']:.2f}")

def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / np.sqrt(((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1)) / (n1+n2-2))
    J = 1 - (3 / (4*(n1+n2) - 9))  # small-sample correction
    return float(J * d)

def _cliffs_delta_mwu(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta via Mann–Whitney U -> AUC relation.
    δ = 2*AUC - 1. Equivalent to rank-biserial correlation.
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = U / (len(x) * len(y))
    return float(2*auc - 1)

def _fdr_bh(p: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR. Returns (reject_mask, p_adjusted)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotone nonincreasing when mapped back
    cummin_rev = np.minimum.accumulate(ranked[::-1])[::-1]
    p_adj_sorted = np.minimum(1.0, cummin_rev)
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    reject = p_adj <= alpha
    return reject, p_adj

# ------------------------- Data collectors from analyzer ---------------------

def _ensure_ready(analyzer) -> None:
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis(None)

def _collect_window_absolute(analyzer, t0: float, t1: float) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return per-trial mean rates (Hz) within [t0,t1] for each cluster and condition."""
    _ensure_ready(analyzer)
    segments = analyzer.psth_data["segments"]
    time_axis = analyzer.psth_data["psth"]["time_axis"]
    bw = analyzer.config.bin_width

    i0 = int(np.searchsorted(time_axis, t0, side="left"))
    i1 = int(np.searchsorted(time_axis, t1, side="right")) - 1
    i0 = np.clip(i0, 0, len(time_axis)-1)
    i1 = np.clip(i1, 0, len(time_axis)-1)
    if i1 < i0:
        i0, i1 = i1, i0

    a_by_c, b_by_c = {}, {}
    for ci in range(analyzer.n_clusters):
        def take(name):
            arr = segments.get(name, np.zeros((0, len(time_axis), analyzer.n_clusters), np.float32))
            if arr.shape[0] == 0:
                return np.array([], float)
            return (arr[:, i0:i1+1, ci].mean(axis=1) / bw).astype(float)
        cid = int(analyzer.clusters[ci]) if str(analyzer.clusters[ci]).isdigit() else analyzer.clusters[ci]
        a_by_c[cid] = take("event_a")
        b_by_c[cid] = take("event_b")
    return a_by_c, b_by_c

def _collect_window_percent(analyzer, window_name: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Use analyzer.window_summary to resolve percent windows per event."""
    _ensure_ready(analyzer)
    if not getattr(analyzer.config, "windows_pct", None) or window_name not in analyzer.config.windows_pct:
        raise ValueError(f"Percent window '{window_name}' not defined in config.windows_pct")
    tables = analyzer.window_summary(segments=analyzer.psth_data["segments"], use_abs=False, use_pct=True)
    a_by_c, b_by_c = {}, {}
    for ci in range(analyzer.n_clusters):
        cid = int(analyzer.clusters[ci]) if str(analyzer.clusters[ci]).isdigit() else analyzer.clusters[ci]
        def pick(df, cond):
            if df.empty:
                return np.array([], float)
            sub = df[(df["cluster"] == cid) & (df["window"] == window_name) & (df["event_type"] == cond)]
            return sub["rate_hz"].to_numpy(dtype=float)
        a_by_c[cid] = pick(tables["event_a"], "event_a")
        b_by_c[cid] = pick(tables["event_b"], "event_b")
    return a_by_c, b_by_c

# ------------------------------ Main stats API -------------------------------

def statistical_comparison_window(
    analyzer,
    time_window: Optional[Tuple[float, float]] = None,
    window_name: Optional[str] = None,
    alpha: Optional[float] = None,
    fdr_across_clusters: bool = True,
) -> Dict:
    """
    event_a vs event_b within a window (absolute or percent). Returns per-cluster dict with:
    U, p, p_fdr, AUC, rank-biserial r, Cliff's δ, Hedges' g, means, Ns, and window metadata.
    """
    _ensure_ready(analyzer)
    cfg_alpha = analyzer.config.alpha if alpha is None else alpha

    if window_name is not None:
        if getattr(analyzer.config, "windows_abs", None) and window_name in analyzer.config.windows_abs:
            t0, t1 = analyzer.config.windows_abs[window_name]
            a_by_c, b_by_c = _collect_window_absolute(analyzer, t0, t1)
            wlabel, wtype = window_name, "absolute"
        elif getattr(analyzer.config, "windows_pct", None) and window_name in analyzer.config.windows_pct:
            a_by_c, b_by_c = _collect_window_percent(analyzer, window_name)
            wlabel, wtype = window_name, "percent"
        else:
            raise ValueError(f"window_name '{window_name}' not found in absolute or percent windows.")
    else:
        if time_window is None:
            time_window = (0.0, 0.5)
        t0, t1 = time_window
        a_by_c, b_by_c = _collect_window_absolute(analyzer, t0, t1)
        wlabel, wtype = f"[{t0:.3f},{t1:.3f}]", "absolute"

    results: Dict[str, Dict] = {}
    pvals, keys = [], []

    for ci in range(analyzer.n_clusters):
        cid = int(analyzer.clusters[ci]) if str(analyzer.clusters[ci]).isdigit() else analyzer.clusters[ci]
        a = np.asarray(a_by_c[cid])
        b = np.asarray(b_by_c[cid])
        if len(a) >= analyzer.config.min_trials and len(b) >= analyzer.config.min_trials:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = float(U / (len(a) * len(b)))
            r_rb = float(2*auc - 1)
            d_h = _hedges_g(a, b)
            delta = _cliffs_delta_mwu(a, b)
            res = {
                "event_a_mean": float(np.mean(a)),
                "event_a_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "event_b_mean": float(np.mean(b)),
                "event_b_std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
                "U": float(U),
                "p_value": float(p),
                "AUC": auc,
                "rank_biserial_r": r_rb,
                "cliffs_delta": delta,
                "hedges_g": d_h,
                "n_event_a": int(len(a)),
                "n_event_b": int(len(b)),
                "window": wlabel,
                "window_type": wtype,
            }
        else:
            res = {"error": "Insufficient data", "window": wlabel, "window_type": wtype}
        results[str(cid)] = res
        if "p_value" in res:
            pvals.append(res["p_value"])
            keys.append(str(cid))

    if fdr_across_clusters and pvals:
        rej, padj = _fdr_bh(np.array(pvals, float), alpha=cfg_alpha)
        for i, cid in enumerate(keys):
            results[cid]["p_value_adj_fdr"] = float(padj[i])
            results[cid]["reject_fdr"] = bool(rej[i])

    return results

def statistical_comparison_by_windows(
    analyzer,
    windows: Optional[List[str]] = None,
    include_absolute: bool = True,
    include_percent: bool = True,
    fdr_across_all: bool = True,
) -> pd.DataFrame:
    """Batch test over named windows (absolute + percent). Returns tidy DataFrame."""
    _ensure_ready(analyzer)

    names: List[str] = []
    if include_absolute and getattr(analyzer.config, "windows_abs", None):
        abs_names = list(analyzer.config.windows_abs.keys())
        names.extend(abs_names if windows is None else [w for w in windows if w in abs_names])
    if include_percent and getattr(analyzer.config, "windows_pct", None):
        pct_names = list(analyzer.config.windows_pct.keys())
        names.extend(pct_names if windows is None else [w for w in windows if w in pct_names])

    rows = []
    for w in names:
        res = statistical_comparison_window(analyzer, window_name=w, fdr_across_clusters=False)
        for cid, d in res.items():
            d2 = d.copy()
            d2["cluster"] = int(cid) if cid.isdigit() else cid
            d2["window_name"] = w
            rows.append(d2)

    out = pd.DataFrame(rows)
    if out.empty or "p_value" not in out.columns:
        return out
    if fdr_across_all:
        rej, padj = _fdr_bh(out["p_value"].to_numpy(float), alpha=analyzer.config.alpha)
        out["p_value_adj_fdr_all"] = padj
        out["reject_fdr_all"] = rej
    return out.sort_values(["window_type", "window_name", "cluster"]).reset_index(drop=True)

def sliding_window_stats(
    analyzer,
    width_s: float = 0.1,
    step_s: float = 0.02,
    t_min: float = -0.5,
    t_max: float = 1.0,
    fdr: bool = True,
) -> pd.DataFrame:
    """Sweep event_a vs event_b over sliding windows. Returns tidy DataFrame."""
    _ensure_ready(analyzer)
    starts = np.arange(t_min, t_max - width_s + 1e-12, step_s)
    rows = []
    for t0 in starts:
        t1 = t0 + width_s
        res = statistical_comparison_window(analyzer, time_window=(t0, t1), fdr_across_clusters=False)
        for cid, d in res.items():
            d2 = d.copy()
            d2["cluster"] = int(cid) if cid.isdigit() else cid
            d2["t0"], d2["t1"] = float(t0), float(t1)
            rows.append(d2)
    out = pd.DataFrame(rows)
    if out.empty or "p_value" not in out.columns:
        return out
    if fdr:
        rej, padj = _fdr_bh(out["p_value"].to_numpy(float), alpha=analyzer.config.alpha)
        out["p_value_adj_fdr"] = padj
        out["reject_fdr"] = rej
    return out.sort_values(["cluster", "t0"]).reset_index(drop=True)

def permutation_time_cluster_test(
    analyzer,
    cluster_idx: int,
    n_perm: int = 2000,
    threshold: float = 2.0,
    tail: Literal["two", "pos", "neg"] = "two",
) -> Dict:
    """
    Maris–Oostenveld time-cluster permutation for one cluster.
    Returns dict with keys: time_axis, t_like, clusters, cluster_masses, p_values, crit_val, alpha.
    """
    _ensure_ready(analyzer)

    segments = analyzer.psth_data["segments"]
    time_axis = analyzer.psth_data["psth"]["time_axis"]
    bw = analyzer.config.bin_width

    def take(name):
        arr = segments.get(name, np.zeros((0, len(time_axis), analyzer.n_clusters), np.float32))
        return (arr[:, :, cluster_idx] / bw).astype(float)

    X = take("event_a")  # trials×time
    Y = take("event_b")
    if X.shape[0] < analyzer.config.min_trials or Y.shape[0] < analyzer.config.min_trials:
        return {"error": "Insufficient trials"}

    mx, my = X.mean(0), Y.mean(0)
    vx, vy = X.var(0, ddof=1), Y.var(0, ddof=1)
    nx, ny = X.shape[0], Y.shape[0]
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    pooled[pooled == 0] = 1.0
    t_like = (mx - my) / pooled

    def clusters_from_stat(stat, thr, tail_mode):
        if tail_mode == "two":
            mask = np.abs(stat) >= thr
            sign_use = np.sign(stat)
        elif tail_mode == "pos":
            mask = stat >= thr
            sign_use = np.ones_like(stat)
        else:
            mask = stat <= -thr
            sign_use = -np.ones_like(stat)
        clusters, masses = [], []
        i = 0
        while i < mask.size:
            if mask[i]:
                j = i
                s = 0.0
                while j < mask.size and mask[j]:
                    s += stat[j] * sign_use[j]
                    j += 1
                clusters.append((i, j-1))
                masses.append(s)
                i = j
            else:
                i += 1
        return clusters, np.array(masses, float)

    obs_clusters, obs_masses = clusters_from_stat(t_like, threshold, tail)

    rng = np.random.default_rng(1234)
    Z = np.concatenate([X, Y], axis=0)
    nX = X.shape[0]
    max_masses = np.empty(n_perm, float)
    for p in range(n_perm):
        idx = rng.permutation(Z.shape[0])
        Xp = Z[idx[:nX]]
        Yp = Z[idx[nX:]]
        mxp, myp = Xp.mean(0), Yp.mean(0)
        vxp, vyp = Xp.var(0, ddof=1), Yp.var(0, ddof=1)
        pooled_p = np.sqrt(((nX-1)*vxp + (Yp.shape[0]-1)*vyp) / (nX+Yp.shape[0]-2))
        pooled_p[pooled_p == 0] = 1.0
        t_like_p = (mxp - myp) / pooled_p
        _, masses_p = clusters_from_stat(t_like_p, threshold, tail)
        max_masses[p] = 0.0 if masses_p.size == 0 else np.max(np.abs(masses_p))

    crit = np.quantile(max_masses, 1 - analyzer.config.alpha)
    pvals = np.array([(np.sum(max_masses >= abs(m)) + 1) / (n_perm + 1) for m in obs_masses], float)

    return {
        "time_axis": time_axis,
        "t_like": t_like,
        "clusters": obs_clusters,
        "cluster_masses": obs_masses,
        "p_values": pvals,
        "crit_val": float(crit),
        "alpha": analyzer.config.alpha,
    }
