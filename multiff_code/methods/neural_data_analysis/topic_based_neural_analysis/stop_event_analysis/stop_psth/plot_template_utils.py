from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# 1) PSTH Template Plot (single cluster)
# ------------------------------------------------------


def plot_psth_template(tmpl_df, cluster, title=None):
    """
    Plot event PSTH template (mean ± CI) for a single cluster.
    tmpl_df is output from export_template_to_df(..., include_ci=True)
    """

    df = tmpl_df[tmpl_df["cluster"] == cluster]

    t = df["time"].values
    mean = df["mean"].values
    lo = df["lower"].values
    hi = df["upper"].values

    plt.figure(figsize=(6, 4))
    plt.plot(t, mean, label=f'Cluster {cluster}', lw=2)
    plt.fill_between(t, lo, hi, alpha=0.25)

    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (normalized)")
    plt.title(title or f"PSTH Template (Cluster {cluster})")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# 2) Split-Half Reliability Plot
# ------------------------------------------------------

def plot_split_half_reliability(rel_df, title=None):
    """
    Plot split-half reliability (r_mean) for each cluster.
    rel_df is output from split_half_reliability().
    """

    plt.figure(figsize=(8, 4))
    plt.bar(rel_df["cluster"], rel_df["r_mean"], color='steelblue')

    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(0.8, color='gray', linestyle='--', alpha=0.7)

    plt.xlabel("Cluster")
    plt.ylabel("Split-half reliability (r)")
    plt.title(title or "Split-Half Reliability by Cluster")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# 3) Mean Residual PSTH (single cluster)
# ------------------------------------------------------


def plot_residual_psth_trials(
    residual_array,
    time,
    cluster,
    plot_mean=False,
    alpha=0.3,
    linewidth=1.0,
    mean_lw=2.0,
    title=None
):
    """
    Plot individual trial residual PSTHs for a single cluster.

    Parameters
    ----------
    residual_array : np.ndarray
        res["event_a"] or res["event_b"],
        shape (n_trials, n_bins, n_clusters)

    time : np.ndarray
        time axis (n_bins,)

    cluster : int
        which cluster to plot

    plot_mean : bool, default False
        If True, also plot the mean residual PSTH in bold.

    alpha : float
        transparency of individual trial lines.

    linewidth : float
        width of each individual trial line.

    mean_lw : float
        linewidth of mean PSTH curve.

    title : str or None
        Plot title.

    """
    # Extract (n_trials, n_bins)
    R = residual_array[:, :, cluster]

    plt.figure(figsize=(7, 4))

    # Plot each trial
    for tr in range(R.shape[0]):
        plt.plot(time, R[tr], color='gray', alpha=alpha, linewidth=linewidth)

    # Optional: plot mean
    if plot_mean:
        mean_res = R.mean(axis=0)
        plt.plot(time, mean_res, color='black', lw=mean_lw, label='Mean')

    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Residual firing (z-normalized)")
    plt.title(title or f"Residual PSTHs (Cluster {cluster})")
    if plot_mean:
        plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# 4) Residual Heatmap (trial × time)
# ------------------------------------------------------

def plot_residual_sorted_heatmap(
    residual_trials,
    time,
    cluster,
    sort_values=None,
    smooth_sigma=None,
    title=None
):
    """
    Plot trial × time residual heatmap for one cluster.
    If sort_values is provided, trials are sorted by it.
    """

    # Extract (n_trials, n_bins)
    R = residual_trials[:, :, cluster]

    # Sorting if requested
    if sort_values is not None:
        order = np.argsort(sort_values)
        R = R[order]
    else:
        order = np.arange(R.shape[0])  # identity

    # Optional smoothing
    if smooth_sigma is not None:
        R = gaussian_filter1d(R, smooth_sigma, axis=1)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        R,
        aspect='auto',
        interpolation='nearest',
        extent=[time[0], time[-1], 0, R.shape[0]],
        cmap='coolwarm'
    )
    plt.colorbar(label="Residual firing (z)")
    plt.axvline(0, color='k', linestyle='--', alpha=0.4)

    plt.xlabel("Time (s)")
    plt.ylabel("Trial index" if sort_values is None else "Sorted trials")
    plt.title(title or f"Residual Heatmap (Cluster {cluster})")
    plt.tight_layout()
    plt.show()

    return order  # optional, useful if caller wants to know trial order





# ------------------------------------------------------
# OPTIONAL — Residual Variability Plot
# ------------------------------------------------------

def plot_residual_variability(residual_array, time, cluster, title=None):
    """
    Plot the across-trial standard deviation of residuals.
    residual_array: shape (n_trials, n_bins, n_clusters)
    """

    std_res = residual_array[:, :, cluster].std(axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(time, std_res, lw=2)

    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Residual variability (std)")
    plt.title(title or f"Residual Variability (Cluster {cluster})")
    plt.tight_layout()
    plt.show()


def compute_variance_explained(original_trials, template, residual_trials, cluster):
    """
    Compute variance explained by template for a single cluster.
    """
    O = original_trials[:, :, cluster]      # (n_trials, n_bins)
    R = residual_trials[:, :, cluster]      # (n_trials, n_bins)
    T = template[:, cluster]                # (n_bins,)

    var_orig = np.var(O)
    var_resid = np.var(R)
    var_tmpl = np.var(T)

    # Two common VE definitions
    ve_fraction = var_tmpl / var_orig
    ve_resid = 1 - (var_resid / var_orig)

    return ve_fraction, ve_resid


def plot_variance_explained_all(original_trials, template, residual_trials, n_clusters):
    """
    Plot variance explained for all clusters.
    """
    ve_frac = []
    ve_resid = []

    for ci in range(n_clusters):
        vf, vr = compute_variance_explained(
            original_trials, template, residual_trials, ci)
        ve_frac.append(vf)
        ve_resid.append(vr)

    plt.figure(figsize=(9, 4))
    plt.bar(range(n_clusters), ve_resid, color='purple')
    plt.xlabel("Cluster")
    plt.ylabel("Variance explained")
    plt.title("Variance Explained by Template (Per Cluster)")
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_residual_by_condition(
    time,
    residual_trials,
    condition,
    cond_val1,
    cond_val2,
    cluster,
    smooth_sigma=2,
    title=None
):
    """
    Overlay residual PSTHs split by condition.
    Example: retry vs switch, left vs right, etc.
    """
    R = residual_trials[:, :, cluster]

    idx1 = np.where(condition == cond_val1)[0]
    idx2 = np.where(condition == cond_val2)[0]

    R1 = R[idx1]
    R2 = R[idx2]

    if smooth_sigma:
        R1 = gaussian_filter1d(R1, smooth_sigma, axis=1)
        R2 = gaussian_filter1d(R2, smooth_sigma, axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(time, R1.mean(axis=0), color='tab:green',
             lw=2, label=str(cond_val1))
    plt.plot(time, R2.mean(axis=0), color='tab:orange',
             lw=2, label=str(cond_val2))

    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Residual firing (z)")
    plt.title(title or f"Residual PSTH by Condition (Cluster {cluster})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_psth_overlay(
    time,
    original_trials,
    template,
    residual_trials=None,
    cluster=0,
    smooth_sigma=None,
    plot_original_mean=True,
    plot_original_trials=False,
    plot_residual_mean=True,
    plot_residual_trials=False,
    template_ci=None,
    alpha_trials=0.2,
    lw_trials=1.0,
    lw_mean=2.0,
    template_color='black',
    original_color='tab:blue',
    residual_color='tab:red',
    title=None,
):
    """
    Overlay original PSTH, template, and residual PSTH for one cluster.

    Parameters
    ----------
    time : (n_bins,) array
        Time axis.
    original_trials : (n_trials, n_bins, n_clusters)
        Trial PSTHs before subtraction.
    template : (n_bins, n_clusters)
        Event template.
    residual_trials : optional (n_trials, n_bins, n_clusters)
        Residual PSTHs after subtraction.
    cluster : int
        Cluster index to plot.
    smooth_sigma : float or None
        Gaussian smoothing (in bins).
    plot_original_mean, plot_original_trials : bool
        Plot mean and/or individual original traces.
    plot_residual_mean, plot_residual_trials : bool
        Plot mean and/or individual residual traces.
    template_ci : (lower, upper) or None
        Template confidence intervals.
    """

    # Extract data for this cluster
    O = original_trials[:, :, cluster]          # (n_trials, n_bins)
    T = template[:, cluster]                    # (n_bins,)
    R = None if residual_trials is None else residual_trials[:, :, cluster]

    # Optional smoothing
    if smooth_sigma is not None:
        T = gaussian_filter1d(T, smooth_sigma)
        if template_ci is not None:
            lo = gaussian_filter1d(template_ci[0][:, cluster], smooth_sigma)
            hi = gaussian_filter1d(template_ci[1][:, cluster], smooth_sigma)
        if original_trials is not None:
            O = gaussian_filter1d(O, smooth_sigma, axis=1)
        if R is not None:
            R = gaussian_filter1d(R, smooth_sigma, axis=1)
    else:
        if template_ci is not None:
            lo = template_ci[0][:, cluster]
            hi = template_ci[1][:, cluster]

    plt.figure(figsize=(8, 5))

    # -------------------------------------------------
    # 1) Original PSTHs
    # -------------------------------------------------
    if plot_original_trials:
        for tr in range(O.shape[0]):
            plt.plot(time, O[tr], color=original_color,
                     alpha=alpha_trials, lw=lw_trials)

    if plot_original_mean:
        mean_O = O.mean(axis=0)
        plt.plot(time, mean_O, color=original_color,
                 lw=lw_mean, label='Original mean')

    # -------------------------------------------------
    # 2) Template
    # -------------------------------------------------
    plt.plot(time, T, color=template_color, lw=lw_mean-1, label='Template')

    if template_ci is not None:
        plt.fill_between(time, lo, hi, color=template_color, alpha=0.2)

    # -------------------------------------------------
    # 3) Residual PSTHs
    # -------------------------------------------------
    if R is not None:
        if plot_residual_trials:
            for tr in range(R.shape[0]):
                plt.plot(time, R[tr], color=residual_color,
                         alpha=alpha_trials, lw=lw_trials)

        if plot_residual_mean:
            mean_R = R.mean(axis=0)
            plt.plot(time, mean_R, color=residual_color,
                     lw=lw_mean, label='Residual mean')
            print(mean_R)

    # -------------------------------------------------
    # Axes formatting
    # -------------------------------------------------
    plt.axvline(0, color='k', linestyle='--', alpha=0.4)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Firing (normalized)")
    plt.title(title or f"PSTH Decomposition (Cluster {cluster})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reliability_heatmap(rel_df):
    """
    Heatmap-like visualization of reliability across clusters.
    """
    values = rel_df["r_mean"].values[np.newaxis, :]

    plt.figure(figsize=(10, 2.5))
    plt.imshow(values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Split-half reliability r')

    plt.yticks([])
    plt.xticks(range(len(values[0])), rel_df["cluster"])
    plt.title("Cluster Reliability Heatmap")
    plt.tight_layout()
    plt.show()


def get_normalized_trials(an, event):
    seg = an.psth_data["segments"][event]
    pre_mask = an._pre_mask
    base_mu, base_sd = an._baseline_stats(seg, pre_mask)
    return an._normalize(an._trial_rates(seg), base_mu, base_sd)


def plot_all_templates_grid(
    tmpl_df,
    n_clusters,
    cols=5,
    title_prefix="Cluster",
    clusters_in_order=None,
):
    """
    Plot PSTH templates for all clusters in a grid.

    Parameters
    ----------
    tmpl_df : DataFrame
        Output of export_template_to_df(include_ci=True)
    n_clusters : int
    clusters_in_order : array-like or None
        Order in which to plot clusters. If None, use 0..n_clusters-1.
    """

    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if clusters_in_order is None:
        clusters_in_order = np.arange(n_clusters)

    rows = math.ceil(n_clusters / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for idx, ci in enumerate(clusters_in_order):
        ax = axes[idx]
        d = tmpl_df[tmpl_df["cluster"] == ci]

        t = d["time"].values
        m = d["mean"].values
        lo = d["lower"].values
        hi = d["upper"].values

        ax.plot(t, m, lw=1.5)
        ax.fill_between(t, lo, hi, alpha=0.2)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_title(f"{title_prefix} {ci}")

    # Hide unused axes
    for i in range(len(clusters_in_order), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
