from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats


def get_event_template(
    analyzer: "PSTHAnalyzer",
    event: Literal["event_a", "event_b"] = "event_a",
    include_ci: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Return the stereotypical PSTH template for a single event type.
    The template matches analyzer normalization/smoothing settings.

    Returns
    -------
    dict with keys:
      - 'time': (n_bins,)
      - 'template': (n_bins, n_clusters)
      - 'sem': (n_bins, n_clusters)  [only if include_ci]
      - 'clusters': (n_clusters,) original cluster IDs
    """
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis()
    psth = analyzer.psth_data["psth"]
    time = psth["time_axis"]
    tmpl = psth[event]
    out: Dict[str, np.ndarray] = {
        "time": time,
        "template": tmpl,
        "clusters": analyzer.clusters,
    }
    if include_ci:
        out["sem"] = psth[event + "_sem"]
    return out


def export_template_to_df(
    analyzer: "PSTHAnalyzer",
    event: Literal["event_a", "event_b"] = "event_a",
    include_ci: bool = True,
) -> pd.DataFrame:
    """
    Flatten a single-event template into a tidy DataFrame.

    Columns: ['time','cluster','mean'] (+ ['sem','lower','upper'] if include_ci)
    """
    d = get_event_template(analyzer, event=event, include_ci=include_ci)
    time = d["time"]
    mean = d["template"]
    cl_ids = d["clusters"]
    t_rep = np.tile(time[:, None], (1, mean.shape[1]))
    cl_rep = np.tile(cl_ids[None, :], (len(time), 1))
    data = {
        "time": t_rep.ravel(),
        "cluster": cl_rep.ravel(),
        "mean": mean.ravel(),
    }
    if include_ci and "sem" in d:
        sem = d["sem"]
        data["sem"] = sem.ravel()
        data["lower"] = (mean - sem).ravel()
        data["upper"] = (mean + sem).ravel()
    return pd.DataFrame(data)



def split_half_reliability(
    analyzer,
    event: Literal["event_a", "event_b"] = "event_a",
    n_splits: int = 200,
    seed: int = 1234,
    metric: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Split-half reliability of stereotypical PSTH per cluster.
    Randomly split trials into halves repeatedly, correlate half-means across time.

    Returns
    -------
    DataFrame with columns:
      ['cluster','r_mean','r_std','n_splits','n_trials']
    """
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis(None)
    segments = analyzer.psth_data["segments"]
    time_axis = analyzer.psth_data["psth"]["time_axis"]
    pre_mask = analyzer._pre_mask  # type: ignore[attr-defined]
    arr = segments.get(event, np.zeros(
        (0, len(time_axis), analyzer.n_clusters), np.float32))

    n_trials = arr.shape[0]
    if n_trials < max(4, analyzer.config.min_trials):  # need at least 2 per half
        return pd.DataFrame({
            "cluster": analyzer.clusters,
            "r_mean": np.full(analyzer.n_clusters, np.nan),
            "r_std": np.full(analyzer.n_clusters, np.nan),
            "n_splits": np.zeros(analyzer.n_clusters, int),
            "n_trials": np.full(analyzer.n_clusters, n_trials, int),
        })

    # normalize trials to match analyzer settings
    base_mu, base_sd = analyzer._baseline_stats(
        arr, pre_mask)  # type: ignore[attr-defined]
    rates_trials = analyzer._trial_rates(
        arr.astype(np.float32))  # type: ignore[attr-defined]
    rates_trials = analyzer._normalize(rates_trials, base_mu, base_sd).astype(
        np.float32)  # type: ignore[attr-defined]

    rng = np.random.default_rng(seed)
    r_store = np.full((n_splits, analyzer.n_clusters), np.nan, dtype=float)
    half = n_trials // 2
    for s in range(n_splits):
        idx = rng.permutation(n_trials)
        A = rates_trials[idx[:half]]  # (half, n_bins, n_clusters)
        B = rates_trials[idx[half:]]  # (n_trials-half, ...)
        if A.shape[0] == 0 or B.shape[0] == 0:
            continue
        mA = A.mean(axis=0)  # (n_bins, n_clusters)
        mB = B.mean(axis=0)
        for ci in range(analyzer.n_clusters):
            x = mA[:, ci]
            y = mB[:, ci]
            if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
                r = np.nan
            else:
                if metric == "pearson":
                    r = float(np.corrcoef(x, y)[0, 1])
                else:
                    r = float(stats.spearmanr(x, y).correlation)
            r_store[s, ci] = r

    # Aggregate robustly to avoid warnings on empty/degenerate columns
    finite = np.isfinite(r_store)
    counts = finite.sum(axis=0)

    with np.errstate(invalid="ignore"):
        z = np.arctanh(np.clip(r_store, -0.999999, 0.999999))

    # Fisher z mean only where counts>0
    sum_z = np.nansum(z, axis=0)
    z_mean = np.where(counts > 0, sum_z / counts, np.nan)
    r_mean = np.tanh(z_mean)

    # Std in r-space with sample ddof=1 only where counts>=2
    sum_r = np.nansum(np.where(finite, r_store, 0.0), axis=0)
    r_bar = np.where(counts > 0, sum_r / counts, np.nan)
    diff = np.where(finite, r_store - r_bar, 0.0)
    sumsq = np.nansum(diff**2, axis=0)
    r_std = np.where(counts >= 2, np.sqrt(sumsq / (counts - 1)), np.nan)

    return pd.DataFrame({
        "cluster": analyzer.clusters,
        "r_mean": r_mean,
        "r_std": r_std,
        "n_splits": counts.astype(int),
        "n_trials": np.full(analyzer.n_clusters, n_trials, int),
    })


def residualize_segments(
    analyzer: "PSTHAnalyzer",
    events: Literal["event_a", "event_b", "both"] = "both",
    leave_one_out: bool = False,
    template_analyzer: Optional["PSTHAnalyzer"] = None,
    template_event_for: Optional[Dict[Literal["event_a","event_b"], Literal["event_a","event_b"]]] = None,
    template_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ci: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Subtract the stereotypical event PSTH template and return:
    - residuals
    - template
    - template confidence intervals (CI)
    """

    import numpy as np
    from scipy.stats import norm

    # Ensure PSTH computed
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis()

    segments = analyzer.psth_data["segments"]
    time = analyzer.psth_data["psth"]["time_axis"]
    pre_mask = analyzer._pre_mask

    # Default event mapping
    if template_event_for is None:
        template_event_for = {"event_a": "event_a", "event_b": "event_b"}  # type: ignore

    # -------------------------------------------------------
    # Utility: align external templates (time + cluster)
    # -------------------------------------------------------
    def _align_template_to_target(tmpl_time, tmpl_mat, tmpl_clusters):
        # Time alignment
        if tmpl_time.shape != time.shape or not np.allclose(tmpl_time, time):
            interp = np.zeros((len(time), tmpl_mat.shape[1]), np.float32)
            for j in range(tmpl_mat.shape[1]):
                interp[:, j] = np.interp(time, tmpl_time, tmpl_mat[:, j]).astype(np.float32)
        else:
            interp = tmpl_mat.astype(np.float32, copy=False)

        # Cluster alignment
        aligned = np.zeros((len(time), analyzer.n_clusters), np.float32)
        src_idx = {tmpl_clusters[i]: i for i in range(len(tmpl_clusters))}
        for j, c in enumerate(analyzer.clusters):
            if c in src_idx:
                aligned[:, j] = interp[:, src_idx[c]]
        return aligned

    # -------------------------------------------------------
    # Utility: fetch external template if provided
    # -------------------------------------------------------
    def _get_external_template(src_event: str):
        # From user-provided dict
        if template_dict is not None and src_event in template_dict:
            d = template_dict[src_event]
            required_keys = {"time", "template", "clusters"}
            if not required_keys.issubset(d.keys()):
                raise ValueError("template_dict entries must include 'time','template','clusters'")
            return _align_template_to_target(d["time"], d["template"], d["clusters"])

        # From template analyzer
        if template_analyzer is not None:
            if not getattr(template_analyzer, "psth_data", None):
                template_analyzer.run_full_analysis()
            d = get_event_template(template_analyzer, event=src_event, include_ci=False)
            return _align_template_to_target(d["time"], d["template"], d["clusters"])

        return None

    # -------------------------------------------------------
    # Core routine: compute residuals, template, template CI
    # -------------------------------------------------------
    def _resid_one(name: str):
        arr = segments.get(name, np.zeros((0, len(time), analyzer.n_clusters), np.float32))

        # No trials -> return empty
        if arr.shape[0] == 0:
            empty = np.zeros((len(time), analyzer.n_clusters), np.float32)
            return arr.copy(), empty, (empty.copy(), empty.copy())

        # Normalize trials (same as PSTHAnalyzer)
        base_mu, base_sd = analyzer._baseline_stats(arr, pre_mask)
        rates_trials = analyzer._trial_rates(arr.astype(np.float32))
        rates_trials = analyzer._normalize(rates_trials, base_mu, base_sd).astype(np.float32)

        # Choose template source
        src_event = template_event_for.get(name, name)  # type: ignore
        external_template = _get_external_template(src_event)

        # ---------------------------
        # Template from external source
        # ---------------------------
        if external_template is not None:
            tmpl = external_template.astype(np.float32)

            # Residuals = trial - template
            resid = rates_trials - tmpl[None, :, :]

            # CI must be computed from rates_trials of CURRENT analyzer
            z = norm.ppf(0.5 + ci / 2)
            sem = rates_trials.std(axis=0, ddof=1) / np.sqrt(rates_trials.shape[0])
            lower = tmpl - z * sem
            upper = tmpl + z * sem

            return resid.astype(np.float32), tmpl.astype(np.float32), (lower.astype(np.float32), upper.astype(np.float32))

        # ---------------------------
        # Internal template (computed from these trials)
        # ---------------------------
        tmpl = rates_trials.mean(axis=0)  # (n_bins, n_clusters)

        # CI (SEM-based)
        z = norm.ppf(0.5 + ci / 2)
        sem = rates_trials.std(axis=0, ddof=1) / np.sqrt(rates_trials.shape[0])
        lower = tmpl - z * sem
        upper = tmpl + z * sem

        # Residuals: leave-one-out or direct
        if not leave_one_out:
            resid = rates_trials - tmpl[None, :, :]
        else:
            n = rates_trials.shape[0]
            sum_all = rates_trials.sum(axis=0)
            scale = n / max(n - 1, 1)
            resid = scale * rates_trials - (sum_all / max(n - 1, 1))[None, :, :]

        return resid.astype(np.float32), tmpl.astype(np.float32), (lower.astype(np.float32), upper.astype(np.float32))

    # -------------------------------------------------------
    # Build output dict
    # -------------------------------------------------------
    out = {}

    if events in ("event_a", "both"):
        resid, tmpl, ci_out = _resid_one("event_a")
        out["event_a"] = resid
        out["template_a"] = tmpl
        out["template_ci_a"] = ci_out

    if events in ("event_b", "both"):
        resid, tmpl, ci_out = _resid_one("event_b")
        out["event_b"] = resid
        out["template_b"] = tmpl
        out["template_ci_b"] = ci_out

    return out
