"""
Behavioral variable decoding from neural population activity.
Extends neural_population.py with different grouping variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

from neural_data_analysis.topic_based_neural_analysis.conditioned_responses import trajectory_clustering, plot_trajectories, neural_conditioned, neural_population, behavioral_decoding


# =========================
# 1. LABEL CONSTRUCTION
# =========================


def make_duration_labels(entry_times, seg_ids, df, n_bins=3):
    durations = {}
    for seg_id in seg_ids:
        t0 = entry_times[seg_id]
        t_end = df[df["new_segment"] == seg_id]["t_center"].max()
        durations[seg_id] = t_end - t0
    return make_quantile_labels(durations, seg_ids, n_bins=n_bins), durations


def make_curvature_labels(entry_times, seg_ids, df, n_bins=3):
    curvatures = {}
    for seg_id in seg_ids:
        t0 = entry_times[seg_id]
        seg_df = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
        idx = (seg_df["t_center"] - t0).abs().idxmin()
        curvatures[seg_id] = seg_df.loc[idx, "curv_of_traj"]
    return make_quantile_labels(curvatures, seg_ids, n_bins=n_bins), curvatures

def make_distance_labels(entry_by_seg, seg_ids, n_bins=3):
    seg_ids_set = set(seg_ids.tolist())
    distances = {
        seg_id: np.sqrt(ex**2 + ey**2)
        for seg_id, (ex, ey) in entry_by_seg.items()
        if seg_id in seg_ids_set
    }
    return make_quantile_labels(distances, seg_ids, n_bins=n_bins), distances

def make_quantile_labels(values_dict, seg_ids, n_bins=3):
    vals = np.array([values_dict[s] for s in seg_ids])
    if np.unique(vals).size < n_bins:
        print(f"Warning: fewer unique values ({np.unique(vals).size}) than n_bins ({n_bins}), reducing.")
        n_bins = np.unique(vals).size
    quantiles = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
    quantiles[-1] += 1e-10
    labels = np.digitize(vals, quantiles[1:-1])
    return labels

# =========================
# 2. DECODE AT ENTRY ONLY
# (new — not in neural_population.py)
# =========================
# in behavioral_decoding.py, define locally
def _safe_n_splits(y, n_splits=5):
    min_count = np.bincount(y).min()
    if min_count < n_splits:
        if verbose:
            print(f"  Warning: min bin count {min_count} < n_splits {n_splits}, reducing.")
        return min_count
    return n_splits

        
def decode_at_entry(X_3d, y, t_bins, n_splits=5, method="svm"):
    """Decode label from population vector at t=0 only."""
    n_splits = _safe_n_splits(y, n_splits)
    entry_idx = np.argmin(np.abs(t_bins))
    X_entry = X_3d[:, :, entry_idx]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = neural_population._make_clf(method, n_features=X_entry.shape[1])
    chance_clf = DummyClassifier(strategy="most_frequent")
    acc = cross_val_score(clf, X_entry, y, cv=cv).mean()
    chance = cross_val_score(chance_clf, X_entry, y, cv=cv).mean()
    return acc, chance


# =========================
# 3. PLOTTING
# =========================
def plot_decoding_summary(results, t_bins, title="decoding summary"):
    n_vars = len(results)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    if n_vars == 1:
        axes = axes[None, :]

    for i, (var_name, res) in enumerate(results.items()):
        # left: timecourse
        ax = axes[i, 0]
        ax.plot(t_bins, res["acc_t"], color="steelblue", lw=2, label="decoding")
        ax.plot(t_bins, res["chance_t"], color="k", lw=1, ls="--", label="chance")
        ax.fill_between(t_bins, res["chance_t"], res["acc_t"],
                        where=res["acc_t"] > res["chance_t"],
                        alpha=0.15, color="steelblue")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("normalized time")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{var_name} — per-timepoint")
        ax.legend(fontsize=8)

        # right: side-by-side bars
        ax = axes[i, 1]
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [res["acc_entry"], res["acc_full"]],
               width=width, color="steelblue", alpha=0.8, label="decoding")
        ax.bar(x + width/2, [res["chance_entry"], res["chance_full"]],
               width=width, color="gray", alpha=0.5, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(["at entry (t=0)", "full trajectory"])
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{var_name} — scalar decoding")
        ax.legend(fontsize=8)

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


# =========================
# 4. RUN ALL
# =========================
def safe_n_splits(y, n_splits=5, verbose=True):
    min_count = np.bincount(y).min()
    if min_count < n_splits:
        if verbose:
            print(f"  Warning: min bin count {min_count} < n_splits {n_splits}, reducing.")
        return min_count
    return n_splits


def run_all(psth, trial_rates, seg_ids, labels_spatial, entry_by_seg,
            entry_times, df, t_bins, n_bins=3, method="svm", n_splits=5, verbose=False):

    neuron_ids = sorted(set(k[1] for k in psth))
    duration_labels, _ = make_duration_labels(entry_times, seg_ids, df, n_bins=n_bins)
    distance_labels, _ = make_distance_labels(entry_by_seg, seg_ids, n_bins=n_bins)
    curvature_labels, _ = make_curvature_labels(entry_times, seg_ids, df, n_bins=n_bins)

    all_labels = {
        "spatial bin": make_quantile_labels({s: int(l) for s, l in zip(seg_ids, labels_spatial)}, seg_ids, n_bins=n_bins),
        "duration":    duration_labels,
        "distance":    distance_labels,
        "curvature":   curvature_labels,
    }

    results = {}
    for var_name, lbl in all_labels.items():
        if verbose:
            print(f"\n--- {var_name} ---")
        X_3d, y = neural_population.build_population_matrix(
            trial_rates, seg_ids, lbl, neuron_ids, t_bins
        )
        if verbose:
            print(f"  trials per bin: { {l: int((y==l).sum()) for l in np.unique(y)} }")

        acc_t, chance_t = neural_population.decode_per_timepoint(
            X_3d, y, n_splits=n_splits, method=method, verbose=verbose
        )
        acc_entry, chance_entry = decode_at_entry(
            X_3d, y, t_bins, n_splits=n_splits, method=method, verbose=verbose
        )
        acc_full, chance_full = neural_population.decode_full_trajectory(
            X_3d, y, n_splits=n_splits, method=method, verbose=verbose
        )

        if verbose:
            print(f"  entry decoding: {acc_entry:.2f} (chance {chance_entry:.2f})")
            print(f"  full trajectory: {acc_full:.2f} (chance {chance_full:.2f})")

        results[var_name] = dict(
            acc_t=acc_t, chance_t=chance_t,
            acc_entry=acc_entry, chance_entry=chance_entry,
            acc_full=acc_full, chance_full=chance_full,
        )

    fig = plot_decoding_summary(results, t_bins)
    return fig, results

import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

def run_session(pn, bin_size=50, std_thresh=0.004, range_thresh=0.01,
                n_bins=20, pre=0.2, n_decode_bins=3, method="lda"):
    """Run full pipeline for one session. Returns (fig, results) or None if failed."""

    # trajectory clustering
    df = pn.rebinned_behav_data.copy()
    df, seg_df, bounds = trajectory_clustering.preprocess_df(df)
    X, seg_ids, entry_by_seg, df, stable_bounds, labels, bin_pairs = \
        trajectory_clustering.build_and_filter_stable_entries(
            df, bin_size=bin_size, std_thresh=std_thresh, range_thresh=range_thresh
        )
    if len(X) == 0:
        print("  No trajectories, skipping.")
        return None, None

    # entry times
    entry_times = {}
    for seg_id, entry_xy in entry_by_seg.items():
        seg_df_s = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
        ex, ey = entry_xy
        dists = (seg_df_s["cur_ff_rel_x"] - ex).abs() + (seg_df_s["cur_ff_rel_y"] - ey).abs()
        entry_times[seg_id] = seg_df_s.loc[dists.idxmin(), "t_center"]

    # neural conditioned
    t_bins, psth, residuals, trial_rates = neural_conditioned.compute_psth(
        pn.spikes_df, entry_times, seg_ids, labels, df, n_bins=n_bins, pre=pre
    )


    fig, results = run_all(
        psth, trial_rates, seg_ids, labels, entry_by_seg,
        entry_times, df, t_bins, n_bins=n_decode_bins, method=method, verbose=verbose
    )
    return fig, results