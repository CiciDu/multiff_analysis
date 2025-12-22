import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import bootstrap_ci

from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import (
    compute_timepoint_pvals,
    cluster_permutation_test
)

# ----------------------------------------------------------
# Projection over time
# ----------------------------------------------------------


def plot_projection(projection, time_axis, title=None):
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, projection)
    plt.xlabel("Time (s)")
    plt.ylabel("Projection")
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# A/B event projection scatter
# ----------------------------------------------------------
def plot_event_projection_scatter(projection, a_bins, b_bins, label_a="A", label_b="B"):
    proj_A = projection[a_bins]
    proj_B = projection[b_bins]

    plt.figure(figsize=(6, 4))
    plt.scatter(np.zeros_like(proj_A), proj_A,
                alpha=0.6, color='blue', label=label_a)
    plt.scatter(np.ones_like(proj_B), proj_B,
                alpha=0.6, color='red', label=label_b)
    plt.xticks([0, 1], [label_a, label_b])
    plt.ylabel("Projection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Projection hist (A vs B)
# ----------------------------------------------------------
def plot_event_projection_hist(projection, a_bins, b_bins, label_a="A", label_b="B", bins=40):
    plt.figure(figsize=(6, 4))
    plt.hist(projection[a_bins], bins=bins, alpha=0.5,
             density=True, color='blue', label=label_a)
    plt.hist(projection[b_bins], bins=bins, alpha=0.5,
             density=True, color='red',  label=label_b)
    plt.xlabel("Projection")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


# ----------------------------------------------------------
# Projection over time with event overlay
# ----------------------------------------------------------
def plot_projection_with_events(projection, time_axis, a_bins, b_bins, label_a="A", label_b="B", title=None):
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, projection, linewidth=0.8, alpha=0.5)
    plt.scatter(time_axis[a_bins], projection[a_bins],
                color='blue', s=15, label=label_a)
    plt.scatter(time_axis[b_bins], projection[b_bins],
                color='red',  s=15, label=label_b)
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Diagnostic plot (hist + scatter + CV summary)
# Supports either (projection + index bins) or direct value arrays.
# ----------------------------------------------------------
def diagnose_axis(projection=None,
                  a_bins=None,
                  b_bins=None,
                  label_a="A",
                  label_b="B",
                  cv_results=None,
                  a_values=None,
                  b_values=None):
    if (a_values is not None) and (b_values is not None):
        proj_A = np.asarray(a_values, float)
        proj_B = np.asarray(b_values, float)
    else:
        if projection is None or a_bins is None or b_bins is None:
            raise ValueError(
                "Provide either (projection, a_bins, b_bins) or (a_values, b_values).")
        proj_A = projection[a_bins]
        proj_B = projection[b_bins]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # Hist
    axs[0].hist(proj_A, bins=40, density=True, alpha=0.5, color='blue')
    axs[0].hist(proj_B, bins=40, density=True, alpha=0.5, color='red')
    axs[0].set_title("Projection Distribution")

    # Scatter
    axs[1].scatter(np.zeros_like(proj_A), proj_A, alpha=0.6, color='blue')
    axs[1].scatter(np.ones_like(proj_B), proj_B, alpha=0.6, color='red')
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels([label_a, label_b])
    axs[1].set_title("Event Projection")

    # CV Summary
    if cv_results:
        summary = (
            f"AUC={cv_results['mean_auc']:.3f}\n"
            f"Acc={cv_results['mean_accuracy']:.3f}\n"
            f"Cos={cv_results['axis_cosine_similarity']:.3f}"
        )
    else:
        summary = "(no CV)"

    axs[2].text(0.1, 0.5, summary, fontsize=16)
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 3D projection
# ----------------------------------------------------------
def plot_3d_projection(proj_retry, proj_reward, proj_stop, title="3D Projection"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(proj_retry, proj_reward, proj_stop)
    ax.set_xlabel("Retry")
    ax.set_ylabel("Reward")
    ax.set_zlabel("Stop")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Heatmap for window search results
# ----------------------------------------------------------
def plot_window_heatmap(df, value='mean_auc', cmap='viridis'):
    """
    Plot a heatmap of window performance.

    Parameters
    ----------
    df : DataFrame returned by scan_windows()
    value : str
        Column to plot ('mean_auc', 'mean_accuracy', or 'axis_cosine_similarity')
    """

    if value not in df.columns:
        raise ValueError(f"Column {value} not found in DataFrame.")

    # Correct pivot syntax
    pivot = df.pivot(index="start_ms", columns="end_ms", values=value)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max()
        ]
    )
    plt.colorbar(label=value)
    plt.xlabel("Window end (ms)")
    plt.ylabel("Window start (ms)")
    plt.title(f"Window Performance Heatmap ({value})")
    plt.tight_layout()
    plt.show()


def plot_event_aligned_projection(
    aligned_A,
    aligned_B,
    time_axis,
    label_a="A",
    label_b="B",
    title=None,
    ci="sem",
    n_boot=500,
    alpha=0.05,             # NEW: significance threshold
    show_significance=False,  # NEW: show shading
    # (start_time, end_time) in same units as time_axis
    train_window=None
):
    """
    Plot event-aligned projection with SEM/CI/bootstrap and 
    significance shading across time where A and B differ.

    Parameters
    ----------
    aligned_A, aligned_B : np.ndarray
        Trial x Time arrays of event-aligned projections for conditions A and B.
    time_axis : np.ndarray
        1D array of times (in seconds) corresponding to the columns of aligned_*.
    label_a, label_b : str
        Labels for A and B conditions.
    title : str or None
        Optional title.
    ci : {"sem", "ci95", "bootstrap"}
        How to compute uncertainty bounds.
    n_boot : int
        Number of bootstrap samples if ci == "bootstrap".
    alpha : float
        Significance threshold used for cluster-based permutation test.
    show_significance : bool
        If True, shade timepoints where A and B differ significantly.
    train_window : tuple or None
        Optional (start_time, end_time) to mark the window used for training,
        in the same units as time_axis (typically seconds relative to event).
    """

    # -------------------------------
    # Compute mean and CI
    # -------------------------------
    mean_A = aligned_A.mean(axis=0)
    mean_B = aligned_B.mean(axis=0)

    if ci == "sem":
        err_A = aligned_A.std(axis=0, ddof=1) / np.sqrt(len(aligned_A))
        err_B = aligned_B.std(axis=0, ddof=1) / np.sqrt(len(aligned_B))
        lower_A, upper_A = mean_A - err_A, mean_A + err_A
        lower_B, upper_B = mean_B - err_B, mean_B + err_B

    elif ci == "ci95":
        sem_A = aligned_A.std(axis=0, ddof=1) / np.sqrt(len(aligned_A))
        sem_B = aligned_B.std(axis=0, ddof=1) / np.sqrt(len(aligned_B))
        lower_A, upper_A = mean_A - 1.96*sem_A, mean_A + 1.96*sem_A
        lower_B, upper_B = mean_B - 1.96*sem_B, mean_B + 1.96*sem_B

    elif ci == "bootstrap":
        lower_A, upper_A = bootstrap_ci(aligned_A, ci=95, n_boot=n_boot)
        lower_B, upper_B = bootstrap_ci(aligned_B, ci=95, n_boot=n_boot)

    # -------------------------------
    # Significance computation
    # -------------------------------
    if show_significance:
        corrected_mask = cluster_permutation_test(
            aligned_A,
            aligned_B,
            alpha=alpha,
            cluster_alpha=0.05,
            n_perm=1000   # adjust if too slow
        )
    else:
        corrected_mask = np.zeros_like(mean_A, dtype=bool)

    # -------------------------------
    # PLOT
    # -------------------------------
    plt.figure(figsize=(9, 5))

    # --- Optional training window shading (behind traces) ---
    if train_window is not None:
        try:
            x0, x1 = float(train_window[0]), float(train_window[1])
        except Exception as exc:
            raise ValueError("train_window must be a tuple/list (start_time, end_time) "
                             "in the same units as time_axis.") from exc
        if x0 > x1:
            x0, x1 = x1, x0
        # If train_window appears to be in ms while time_axis is in s, auto-convert
        try:
            ta_abs_max = float(np.nanmax(np.abs(time_axis)))
            ta_span = float(np.nanmax(time_axis) - np.nanmin(time_axis))
            tw_abs_max = max(abs(x0), abs(x1))
            tw_span = x1 - x0
            # Heuristic: if train window scale is much larger than time_axis scale, assume ms
            if ta_abs_max > 0 and ((tw_abs_max > 10.0 * ta_abs_max) or (tw_span > 10.0 * ta_span)):
                x0 /= 1000.0
                x1 /= 1000.0
        except Exception:
            # If any issue in heuristic, fall back to given units
            pass
        # Default aesthetics for training window shading
        train_color = "gray"
        train_alpha = 0.12
        train_label = "Train window"
        train_hatch = None
        span_kwargs = dict(color=train_color, alpha=train_alpha,
                           label=train_label, zorder=0)
        # Only pass hatch if explicitly provided to avoid backend-specific warnings
        if train_hatch is not None:
            span_kwargs["hatch"] = train_hatch
            span_kwargs["edgecolor"] = train_color
        plt.axvspan(x0, x1, **span_kwargs)

    # --- Plot means and error ---
    plt.plot(time_axis, mean_A, color="blue", label=label_a)
    plt.fill_between(time_axis, lower_A, upper_A, alpha=0.2, color="blue")

    plt.plot(time_axis, mean_B, color="red", label=label_b)
    plt.fill_between(time_axis, lower_B, upper_B, alpha=0.2, color="red")

    # --- Event time ---
    plt.axvline(0, linestyle="--", color="black", alpha=0.7)

    # --- Optional significance shading ---
    if show_significance:
        ymin, ymax = plt.ylim()  # current limits from the projection data
        plt.fill_between(
            time_axis,
            ymin, ymax,
            where=corrected_mask,
            color="yellow",
            alpha=0.15,
            label=f"Cluster-corrected (p < {alpha:.2f})"
        )

    plt.xlabel("Time relative to event (s)")
    plt.ylabel("Projection")
    n_A = len(aligned_A)
    n_B = len(aligned_B)
    if title:
        plt.title(f"{title} ({label_a}={n_A}, {label_b}={n_B})", fontsize=14)
    else:
        plt.title(
            f"Event-Aligned Projection with Significance Shading ({label_a}={n_A}, {label_b}={n_B})", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
