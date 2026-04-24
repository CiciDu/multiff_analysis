"""
Behavioral variable decoding from neural population activity.
Extends neural_population.py with different grouping variables.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.dummy import DummyClassifier

from neural_data_analysis.topic_based_neural_analysis.conditioned_responses import (
    trajectory_clustering, neural_conditioned, neural_population
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import combine_info_utils, general_utils

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
# 2. DECODING
# =========================

def _safe_n_splits(y, n_splits=5, verbose=True):
    min_count = np.bincount(y).min()
    if min_count < n_splits:
        if verbose:
            print(f"  Warning: min bin count {min_count} < n_splits {n_splits}, reducing.")
        return min_count
    return n_splits

def decode_at_entry(X_3d, y, t_bins, n_splits=5, method="svm",
                    n_permutations=200, verbose=True):
    """
    Decode label from population vector at t=0 only.
    Returns (acc, chance, acc_folds, chance_folds, p_value).
    """
    n_splits = _safe_n_splits(y, n_splits, verbose=verbose)
    entry_idx = np.argmin(np.abs(t_bins))
    X_entry = X_3d[:, :, entry_idx]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    acc_folds, chance_folds = [], []
    for train_idx, test_idx in cv.split(X_entry, y):
        clf_fold = neural_population._make_clf(method, n_features=X_entry.shape[1])
        clf_fold.fit(X_entry[train_idx], y[train_idx])
        acc_folds.append(clf_fold.score(X_entry[test_idx], y[test_idx]))

        chance_clf = DummyClassifier(strategy="most_frequent")
        chance_clf.fit(X_entry[train_idx], y[train_idx])
        chance_folds.append(chance_clf.score(X_entry[test_idx], y[test_idx]))

    acc    = np.mean(acc_folds)
    chance = np.mean(chance_folds)

    clf_perm = neural_population._make_clf(method, n_features=X_entry.shape[1])
    _, perm_scores_entry, p_value = permutation_test_score(
        clf_perm, X_entry, y, cv=cv,
        n_permutations=n_permutations, random_state=0, n_jobs=1
    )
    return acc, chance, np.array(acc_folds), np.array(chance_folds), p_value, perm_scores_entry


# =========================
# 3. PER-NEURON η² WITH SHUFFLE THRESHOLD
# =========================

def _eta2_from_groups(rates, labels):
    """Scalar η² for one neuron given its trial rates and group labels."""
    groups = [rates[labels == b] for b in np.unique(labels)]
    grand_mean = rates.mean()
    ss_total = ((rates - grand_mean) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total


def compute_eta2_per_neuron(trial_rates, seg_ids, y, neuron_ids,
                             n_shuffles=500, random_state=0):
    """
    Compute per-neuron η² with a per-neuron shuffle threshold.

    For each neuron, the significance threshold is the 95th percentile of
    that neuron's own shuffle null distribution (n_shuffles permutations of y).
    This avoids the pooled-threshold problem where high-variance neurons
    inflate the threshold for everyone.

    Returns
    -------
    eta2_per_neuron : (n_neurons,)  observed η² per neuron
    thresholds_95   : (n_neurons,)  per-neuron shuffle 95th percentile
    n_tuned         : int           neurons with η² > their own threshold
    neuron_ids_used : list
    """
    rng     = np.random.default_rng(random_state)
    seg_ids = np.asarray(seg_ids)
    y       = np.asarray(y)

    rate_rows, valid_nids = [], []
    for nid in neuron_ids:
        row = np.array([
            trial_rates[(sid, nid)].mean()
            for sid in seg_ids
            if (sid, nid) in trial_rates
        ])
        if len(row) == len(seg_ids):
            rate_rows.append(row)
            valid_nids.append(nid)

    if len(rate_rows) == 0:
        return np.array([]), np.array([]), 0, []

    rate_matrix = np.array(rate_rows)   # (n_neurons, n_trials)
    n_neurons   = len(rate_matrix)

    # observed η² per neuron
    eta2_obs = np.array([
        _eta2_from_groups(rate_matrix[i], y) for i in range(n_neurons)
    ])

    # per-neuron shuffle null → per-neuron threshold
    thresholds_95 = np.zeros(n_neurons)
    for i in range(n_neurons):
        null = np.array([
            _eta2_from_groups(rate_matrix[i], rng.permutation(y))
            for _ in range(n_shuffles)
        ])
        thresholds_95[i] = np.nanpercentile(null, 95)

    valid   = ~np.isnan(eta2_obs)
    n_tuned = int(np.sum(eta2_obs[valid] > thresholds_95[valid]))

    return eta2_obs, thresholds_95, n_tuned, valid_nids


# =========================
# 4. PLOTTING
# =========================

def _pvalue_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def plot_decoding_summary(results, t_bins, title="decoding summary"):
    """
    3-column layout per variable:
      col 0 — per-timepoint decoding timecourse
      col 1 — scalar decoding bars (entry / full) + SEM + permutation-test stars
      col 2 — shuffle null distributions for entry and full trajectory decoding
    """
    n_vars = len(results)
    fig, axes = plt.subplots(n_vars, 3, figsize=(16, 4 * n_vars))
    if n_vars == 1:
        axes = axes[None, :]

    for i, (var_name, res) in enumerate(results.items()):

        # --- col 0: per-timepoint decoding ---
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

        # --- col 1: scalar decoding bars + SEM + permutation stars ---
        ax = axes[i, 1]
        x     = np.arange(2)
        width = 0.35

        dec_means  = [res["acc_entry"],       res["acc_full"]]
        dec_sems   = [res["sem_entry"],        res["sem_full"]]
        chan_means = [res["chance_entry"],     res["chance_full"]]
        chan_sems  = [res["sem_chance_entry"], res["sem_chance_full"]]
        p_vals     = [res["p_entry"],          res["p_full"]]

        ax.bar(x - width/2, dec_means, width=width,
               color="steelblue", alpha=0.8, label="decoding")
        ax.errorbar(x - width/2, dec_means, yerr=dec_sems,
                    fmt="none", color="black", capsize=4, lw=1.5)
        ax.bar(x + width/2, chan_means, width=width,
               color="gray", alpha=0.5, label="chance")
        ax.errorbar(x + width/2, chan_means, yerr=chan_sems,
                    fmt="none", color="black", capsize=4, lw=1.5)

        for xi, (dm, ds, p) in enumerate(zip(dec_means, dec_sems, p_vals)):
            stars = _pvalue_to_stars(p)
            ax.text(xi - width / 2, dm + ds + 0.03, stars,
                    ha="center", va="bottom", fontsize=11,
                    color="black" if stars != "ns" else "gray")

        ax.set_xticks(x)
        ax.set_xticklabels(["at entry (t=0)", "full trajectory"])
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{var_name} — scalar decoding")
        ax.legend(fontsize=8)

        # --- col 2: shuffle null distributions ---
        ax = axes[i, 2]
        perm_entry = res.get("perm_scores_entry", None)
        perm_full  = res.get("perm_scores_full",  None)
        acc_entry  = res["acc_entry"]
        acc_full   = res["acc_full"]
        p_entry    = res["p_entry"]
        p_full     = res["p_full"]

        bins = np.linspace(0, 1, 40)
        if perm_entry is not None:
            ax.hist(perm_entry, bins=bins, color="steelblue", alpha=0.4,
                    label="null — entry")
            pct95_entry = np.percentile(perm_entry, 95)
            ax.axvline(pct95_entry, color="steelblue", lw=1.5, ls="--",
                       alpha=0.8, label=f"95th entry = {pct95_entry:.2f}")
            ax.axvline(acc_entry, color="steelblue", lw=2, ls="-",
                       label=f"obs entry ({_pvalue_to_stars(p_entry)})")
        if perm_full is not None:
            ax.hist(perm_full, bins=bins, color="darkorange", alpha=0.4,
                    label="null — full")
            pct95_full = np.percentile(perm_full, 95)
            ax.axvline(pct95_full, color="darkorange", lw=1.5, ls="--",
                       alpha=0.8, label=f"95th full = {pct95_full:.2f}")
            ax.axvline(acc_full, color="darkorange", lw=2, ls="-",
                       label=f"obs full ({_pvalue_to_stars(p_full)})")

        ax.set_xlabel("accuracy")
        ax.set_ylabel("count")
        ax.set_xlim(0, 1)
        ax.set_title(f"{var_name} — shuffle null")
        ax.legend(fontsize=8)

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_eta2_summary(results, title="per-neuron η²"):
    """
    Separate figure: one panel per variable.
    Shows per-neuron η² as a bar chart (neurons on x-axis) with each
    neuron's individual shuffle 95th-percentile threshold marked as a
    dot, making it easy to see which neurons exceed their own threshold.
    """
    n_vars = len(results)
    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]

    for ax, (var_name, res) in zip(axes, results.items()):
        eta2_per_neuron = res.get("eta2_per_neuron", np.array([]))
        thresholds_95   = res.get("eta2_thresholds_95", np.array([]))
        n_tuned         = res.get("eta2_n_tuned", 0)

        valid_mask = ~np.isnan(eta2_per_neuron) if len(eta2_per_neuron) > 0 else np.array([], dtype=bool)
        eta2_valid = eta2_per_neuron[valid_mask] if len(eta2_per_neuron) > 0 else np.array([])
        thr_valid  = thresholds_95[valid_mask]   if len(thresholds_95) > 0 else np.array([])
        n_total    = len(eta2_valid)

        if n_total > 0:
            x        = np.arange(n_total)
            tuned    = eta2_valid > thr_valid
            colors   = ["steelblue" if t else "lightsteelblue" for t in tuned]

            ax.bar(x, eta2_valid, color=colors, edgecolor="white", lw=0.5)
            # per-neuron threshold as red dots
            ax.scatter(x, thr_valid, color="firebrick", s=30, zorder=5,
                       label="shuffle 95th (per neuron)")

            pct_tuned = 100 * n_tuned / n_total
            ax.set_title(f"{var_name}\n{n_tuned}/{n_total} tuned ({pct_tuned:.0f}%)")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(var_name)

        ax.set_xlabel("neuron")
        ax.set_ylabel("η²")
        ax.set_xticks([])   # neuron IDs are arbitrary; suppress tick clutter

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


# =========================
# 5. RUN ALL
# =========================

def safe_n_splits(y, n_splits=5, verbose=True):
    min_count = np.bincount(y).min()
    if min_count < n_splits:
        if verbose:
            print(f"  Warning: min bin count {min_count} < n_splits {n_splits}, reducing.")
        return min_count
    return n_splits


def run_all(psth, trial_rates, seg_ids, labels_spatial, entry_by_seg,
            entry_times, df, t_bins, n_bins=3, method="svm", n_splits=5,
            n_permutations=200, n_eta2_shuffles=500, verbose=False):

    neuron_ids = sorted(set(k[1] for k in psth))
    duration_labels, _ = make_duration_labels(entry_times, seg_ids, df, n_bins=n_bins)
    distance_labels, _ = make_distance_labels(entry_by_seg, seg_ids, n_bins=n_bins)
    curvature_labels, _ = make_curvature_labels(entry_times, seg_ids, df, n_bins=n_bins)

    all_labels = {
        "spatial bin": make_quantile_labels(
            {s: int(label) for s, label in zip(seg_ids, labels_spatial)},
            seg_ids, n_bins=n_bins,
        ),
        "duration":  duration_labels,
        "distance":  distance_labels,
        "curvature": curvature_labels,
    }

    results = {}
    for var_name, lbl in all_labels.items():
        if verbose:
            print(f"\n--- {var_name} ---")

        X_3d, y = neural_population.build_population_matrix(
            trial_rates, seg_ids, lbl, neuron_ids, verbose=verbose
        )

        acc_t, chance_t = neural_population.decode_per_timepoint(
            X_3d, y, n_splits=n_splits, method=method, verbose=verbose
        )
        acc_entry, chance_entry, folds_entry, folds_chance_entry, p_entry, perm_scores_entry = \
            decode_at_entry(X_3d, y, t_bins, n_splits=n_splits, method=method,
                            n_permutations=n_permutations, verbose=verbose)
        acc_full, chance_full, folds_full, folds_chance_full, p_full, perm_scores_full = \
            neural_population.decode_full_trajectory(
                X_3d, y, n_splits=n_splits, method=method,
                n_permutations=n_permutations, verbose=verbose
            )

        # per-neuron η² with per-neuron shuffle threshold
        eta2_per_neuron, thresholds_95, n_tuned, _ = compute_eta2_per_neuron(
            trial_rates, seg_ids, lbl, neuron_ids,
            n_shuffles=n_eta2_shuffles
        )

        if verbose:
            n_total = len(eta2_per_neuron)
            print(f"  entry:  {acc_entry:.2f} ± {folds_entry.std()/np.sqrt(len(folds_entry)):.2f}  (p={p_entry:.3f})")
            print(f"  full:   {acc_full:.2f} ± {folds_full.std()/np.sqrt(len(folds_full)):.2f}  (p={p_full:.3f})")
            print(f"  η²:     {n_tuned}/{n_total} neurons tuned")

        results[var_name] = dict(
            acc_t=acc_t,
            chance_t=chance_t,
            acc_entry=acc_entry,
            chance_entry=chance_entry,
            sem_entry=folds_entry.std() / np.sqrt(len(folds_entry)),
            sem_chance_entry=folds_chance_entry.std() / np.sqrt(len(folds_chance_entry)),
            p_entry=p_entry,
            perm_scores_entry=perm_scores_entry,
            acc_full=acc_full,
            chance_full=chance_full,
            sem_full=folds_full.std() / np.sqrt(len(folds_full)),
            sem_chance_full=folds_chance_full.std() / np.sqrt(len(folds_chance_full)),
            p_full=p_full,
            perm_scores_full=perm_scores_full,
            eta2_per_neuron=eta2_per_neuron,
            eta2_thresholds_95=thresholds_95,   # array, one per neuron
            eta2_n_tuned=n_tuned,
        )

    fig_decoding = plot_decoding_summary(results, t_bins)
    fig_eta2     = plot_eta2_summary(results)
    return fig_decoding, fig_eta2, results


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
                n_bins=20, pre=0.2, n_decode_bins=3, method="lda",
                n_permutations=200, n_eta2_shuffles=500, verbose=False):
    """Run full pipeline for one session. Returns (fig, results) or None if failed."""

    df = pn.rebinned_behav_data.copy()
    df, seg_df, bounds = trajectory_clustering.preprocess_df(df)
    X, seg_ids, entry_by_seg, df, stable_bounds, labels, bin_pairs = \
        trajectory_clustering.build_and_filter_stable_entries(
            df, bin_size=bin_size, std_thresh=std_thresh, range_thresh=range_thresh
        )
    if len(X) == 0:
        print("  No trajectories, skipping.")
        return None, None

    entry_times = {}
    for seg_id, entry_xy in entry_by_seg.items():
        seg_df_s = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
        ex, ey = entry_xy
        dists = (seg_df_s["cur_ff_rel_x"] - ex).abs() + (seg_df_s["cur_ff_rel_y"] - ey).abs()
        entry_times[seg_id] = seg_df_s.loc[dists.idxmin(), "t_center"]

    t_bins, psth, residuals, trial_rates = neural_conditioned.compute_psth(
        pn.spikes_df, entry_times, seg_ids, labels, df, n_bins=n_bins, pre=pre
    )

    fig_decoding, fig_eta2, results = run_all(
        psth, trial_rates, seg_ids, labels, entry_by_seg,
        entry_times, df, t_bins, n_bins=n_decode_bins, method=method,
        n_permutations=n_permutations, n_eta2_shuffles=n_eta2_shuffles,
        verbose=verbose
    )
    return fig_decoding, fig_eta2, results


# =========================
# 6. SUMMARIZE ACROSS SESSIONS
# =========================

def summarize_results(all_results):
    """
    Flatten all_results into a tidy DataFrame.
    One row per session × variable × condition.
    """
    import pandas as pd

    rows = []
    for session_name, results in all_results.items():
        if results is None:
            continue
        for var_name, res in results.items():
            eta2_arr = res.get("eta2_per_neuron", np.array([]))
            n_total  = int(np.sum(~np.isnan(eta2_arr))) if len(eta2_arr) > 0 else 0
            n_tuned  = res.get("eta2_n_tuned", 0)

            for condition in ("entry", "full"):
                rows.append(dict(
                    session         = session_name,
                    variable        = var_name,
                    condition       = condition,
                    acc             = res[f"acc_{condition}"],
                    chance          = res[f"chance_{condition}"],
                    sem             = res[f"sem_{condition}"],
                    sem_chance      = res[f"sem_chance_{condition}"],
                    p_value         = res[f"p_{condition}"],
                    significant     = res[f"p_{condition}"] < 0.05,
                    eta2_threshold  = float(np.nanmean(res.get("eta2_thresholds_95", [np.nan]))),
                    n_tuned         = n_tuned,
                    n_neurons       = n_total,
                    pct_tuned       = 100 * n_tuned / n_total if n_total > 0 else np.nan,
                ))

    return pd.DataFrame(rows)

# =========================
# 7. CROSS-SESSION SUMMARY PLOT
# =========================

def plot_summary_across_sessions(summary_df, title="across-session decoding summary"):
    """
    Two-panel figure summarizing decoding results across sessions.

    Panel A — fraction of sessions with significant decoding (p < 0.05),
      per variable × condition. Dashed line at 0.05 (expected false positive rate).

    Panel B — % neurons tuned (η² > per-neuron shuffle threshold),
      mean ± SEM across sessions, per variable.
      Dashed line at 5% (expected by chance from the 95th-percentile threshold).
    """
    variables  = list(summary_df["variable"].unique())
    conditions = ["entry", "full"]
    n_vars     = len(variables)

    var_colors = {
        "spatial bin": "steelblue",
        "duration":    "darkorange",
        "distance":    "forestgreen",
        "curvature":   "firebrick",
    }

    fig, (ax_sig, ax_eta) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: fraction of sessions significant ----
    x_cond    = np.arange(len(conditions))
    bar_width = 0.8 / n_vars

    for vi, var in enumerate(variables):
        color  = var_colors.get(var, "steelblue")
        fracs  = []
        n_sess = []
        for cond in conditions:
            sub = summary_df[(summary_df["variable"] == var) &
                             (summary_df["condition"] == cond)]
            fracs.append(sub["significant"].mean() if not sub.empty else 0)
            n_sess.append(len(sub))

        offset = (vi - n_vars / 2 + 0.5) * bar_width
        bars = ax_sig.bar(x_cond + offset, fracs, width=bar_width,
                          color=color, alpha=0.85,
                          label=var)
        # annotate with raw count
        for xi, (frac, n) in enumerate(zip(fracs, n_sess)):
            ax_sig.text(x_cond[xi] + offset, frac + 0.02,
                        f"{int(round(frac * n))}/{n}",
                        ha="center", va="bottom", fontsize=8)

    ax_sig.axhline(0.05, color="k", lw=1, ls="--", alpha=0.6,
                   label="α = 0.05")
    ax_sig.set_xticks(x_cond)
    ax_sig.set_xticklabels(["at entry (t=0)", "full trajectory"])
    ax_sig.set_ylabel("fraction of sessions significant")
    ax_sig.set_ylim(0, 1.15)
    ax_sig.set_title("fraction of sessions with\nsignificant decoding (p < 0.05)")
    ax_sig.legend(title="variable", fontsize=8)

    # ---- Panel B: % neurons tuned ----
    eta_df = summary_df.drop_duplicates(["session", "variable"])
    x_vars = np.arange(n_vars)

    for vi, var in enumerate(variables):
        color = var_colors.get(var, "steelblue")
        sub   = eta_df[eta_df["variable"] == var]
        if sub.empty:
            continue
        mean_pct = sub["pct_tuned"].mean()
        sem_pct  = sub["pct_tuned"].sem()
        ax_eta.bar(vi, mean_pct, color=color, alpha=0.85, width=0.6)
        ax_eta.errorbar(vi, mean_pct, yerr=sem_pct,
                        fmt="none", color="black", capsize=4, lw=1.2)
        ax_eta.text(vi, mean_pct + (sem_pct or 0) + 1,
                    f"{mean_pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax_eta.axhline(5, color="k", lw=1, ls="--", alpha=0.5,
                   label="5% expected by chance")
    ax_eta.set_xticks(x_vars)
    ax_eta.set_xticklabels(variables, rotation=15, ha="right")
    ax_eta.set_ylabel("% neurons tuned (η² > shuffle 95th)")
    ax_eta.set_ylim(0, 100)
    ax_eta.set_title("per-neuron tuning\n(mean ± SEM across sessions)")
    ax_eta.legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    return fig

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_wrangling import combine_info_utils


# ── paths ────────────────────────────────────────────────────────────────────

def get_save_paths(monkey_name: str, base_dir: str = 'all_monkey_data') -> dict:
    base = os.path.join(base_dir, 'planning_and_neural', monkey_name, 'combined_data', 'conditioned_responses')
    return {
        'all_results': os.path.join(base, 'trajectory_decoding_all_results.pkl'),
        'summary_df':  os.path.join(base, 'trajectory_decoding_summary_df.csv'),
    }


# ── session loop ─────────────────────────────────────────────────────────────

def run_decoding_for_monkey(
    monkey_name:    str,
    raw_data_dir:   str  = 'all_monkey_data/raw_monkey_data',
    bin_width_pn:   float = 0.1,
    load_if_exists: bool  = True,
) -> dict:
    """Run behavioral decoding for all sessions of one monkey.

    Returns all_results dict keyed by session name.
    If load_if_exists and the pickle exists, loads and returns it instead.
    """
    paths = get_save_paths(monkey_name, base_dir=os.path.dirname(raw_data_dir))
    save_path = paths['all_results']

    if load_if_exists and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f'Loaded {len(all_results)} sessions from {save_path}')
        plot_all_results(all_results)
        return all_results

    sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(raw_data_dir, monkey_name)
    all_results = {}

    for _, row in sessions_df.iterrows():
        session_name = row['data_name']
        print('=' * 80)
        print(session_name)
        try:
            pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
                raw_data_folder_path=os.path.join(raw_data_dir, row['monkey_name'], session_name),
                bin_width=bin_width_pn,
            )
            with suppress_stdout():
                pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=True)
                pn.planning_data_by_point, _ = general_utils.drop_columns_with_many_nans(pn.planning_data_by_point)
                pn.get_x_and_y_data_for_modeling(exists_ok=True, reduce_y_var_lags=False)
                pn.prepare_seg_aligned_data(start_t_rel_event=-0.25, end_at_stop_time=True)

            fig_decoding, fig_eta2, results = run_session(pn)
            if results is None:
                continue

            fig_decoding.suptitle(f'{session_name} — decoding')
            fig_eta2.suptitle(f'{session_name} — per-neuron η²')
            plt.show()
            all_results[session_name] = results

        except Exception as e:
            print(f'  Failed: {e}')
            continue

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(all_results)} sessions to {save_path}')

    return all_results


def plot_all_results(
    all_results: dict,
    show: bool = True,
) -> dict:
    """Plot per-session decoding and eta2 summaries from precomputed all_results.

    Mirrors plotting behavior in run_decoding_for_monkey without recomputing
    session decoding.
    """
    figures_by_session = {}

    for session_name, results in all_results.items():
        if results is None:
            continue

        first_var = next(iter(results.values()), None)
        if first_var is None or 'acc_t' not in first_var:
            print(f'Skipping {session_name}: missing timecourse in results')
            continue

        n_time_bins = len(first_var['acc_t'])
        if n_time_bins == 0:
            print(f'Skipping {session_name}: empty timecourse')
            continue

        t_bins = np.linspace(0, 1, n_time_bins)
        fig_decoding = plot_decoding_summary(results, t_bins)
        fig_eta2 = plot_eta2_summary(results)

        fig_decoding.suptitle(f'{session_name} — decoding')
        fig_eta2.suptitle(f'{session_name} — per-neuron η²')

        if show:
            plt.show()

        figures_by_session[session_name] = {
            'fig_decoding': fig_decoding,
            'fig_eta2': fig_eta2,
        }

    return figures_by_session


# ── summary df ───────────────────────────────────────────────────────────────

def get_summary_df(
    all_results:    dict,
    monkey_name:    str,
    base_dir:       str  = 'all_monkey_data',
    load_if_exists: bool = True,
) -> pd.DataFrame:
    """Build (or load) summary_df from all_results.

    If load_if_exists and the CSV exists, loads and returns it instead.
    """
    paths = get_save_paths(monkey_name, base_dir=base_dir)
    save_path = paths['summary_df']

    if load_if_exists and os.path.exists(save_path):
        summary_df = pd.read_csv(save_path)
        summary_df['significant'] = summary_df['significant'].astype(bool)
        print(f'Loaded {len(summary_df)} rows from {save_path}')
        return summary_df

    summary_df = summarize_results(all_results)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f'Saved to {save_path}')

    return summary_df


# ── aggregate summaries ──────────────────────────────────────────────────────

def print_aggregate_stats(summary_df: pd.DataFrame) -> None:
    print('\n── mean accuracy per variable/condition ──')
    print(summary_df.groupby(['variable', 'condition'])[['acc', 'chance']].mean())

    print('\n── fraction of sessions with significant decoding ──')
    print(summary_df.groupby(['variable', 'condition'])['significant'].mean())

    print('\n── fraction of tuned neurons per variable ──')
    print(summary_df.groupby('variable')[['pct_tuned', 'n_tuned', 'n_neurons']].mean())


def plot_summary(summary_df: pd.DataFrame, monkey_name: str) -> plt.Figure:
    fig_summary = plot_summary_across_sessions(summary_df)
    fig_summary.suptitle(f'{monkey_name} — all sessions')

    plt.show()
    return fig_summary


