

import numpy as np
import matplotlib.pyplot as plt

from neural_data_analysis.topic_based_neural_analysis.conditioned_responses import trajectory_clustering


def _mark_axes_origin(ax=None, *, zorder=12, color="k", markersize=9):
    """Mark data coordinates (0, 0) on the current or given axes."""
    kw = dict(
        marker="+",
        color=color,
        markersize=markersize,
        markeredgewidth=1.8,
        linestyle="None",
        zorder=zorder,
    )
    if ax is None:
        plt.plot(0.0, 0.0, **kw)
    else:
        ax.plot(0.0, 0.0, **kw)


def plot_clusters(
    df,
    seg_ids,
    labels,
    bounds=None,
    bin_size=50,
    align=True,
    show_grid=True,
    entry_by_seg=None,
):
    """
    Plot trajectories colored by cluster/bin label.

    ``entry_by_seg``: required map ``new_segment`` -> (x, y) for stable-phase
    entry. Trajectories are truncated from this entry point before plotting.
    """
    if entry_by_seg is None:
        raise ValueError("plot_clusters requires entry_by_seg (stable-entry map).")

    mapping = dict(zip(seg_ids, labels))
    unique_labels = np.unique(labels)
    label_to_color_idx = {lab: i for i, lab in enumerate(unique_labels)}

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]

    plt.figure(figsize=(8, 10))

    cluster_trajs = {k: [] for k in unique_labels}

    def _to_plot_xy(traj):
        if align:
            return traj - traj[0]
        return traj

    def _entry_xy(seg_id, traj):
        if seg_id in entry_by_seg:
            return np.array(entry_by_seg[seg_id], dtype=float)
        return traj[0].astype(float)

    def _truncate_from_entry(seg_id, traj):
        if entry_by_seg is None or seg_id not in entry_by_seg:
            return traj
        entry_xy = np.array(entry_by_seg[seg_id], dtype=float)
        d2 = np.sum((traj - entry_xy) ** 2, axis=1)
        start = int(np.argmin(d2))
        return traj[start:]

    # Plot individual trajectories (colored, light)
    for seg_id, seg_df in df.groupby("new_segment"):
        if seg_id not in mapping:
            continue

        label = mapping[seg_id]
        color = colors[label_to_color_idx[label]]

        traj = seg_df.sort_values("bin_in_new_seg")[
            ["cur_ff_rel_x", "cur_ff_rel_y"]
        ].values
        traj = _truncate_from_entry(seg_id, traj)

        cluster_trajs[label].append(traj)

        xy = _to_plot_xy(traj)
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            alpha=0.45,
            linewidth=1.5,
        )

        ex, ey = _entry_xy(seg_id, traj)
        if align:
            ex, ey = ex - traj[0, 0], ey - traj[0, 1]
        plt.scatter(ex, ey, color=color, alpha=0.7, s=20)

    # Plot mean trajectories (bold)
    for i, label in enumerate(unique_labels):
        trajs = cluster_trajs[label]
        if len(trajs) == 0:
            continue

        trajs_to_avg = [(t - t[0]) if align else t for t in trajs]
        mean_traj = mean_traj_arclength(trajs_to_avg, n_points=50)
        if mean_traj is None:
            continue  # skip this bin if resampling failed

        plt.plot(
            mean_traj[:, 0],
            mean_traj[:, 1],
            color=colors[i],
            linewidth=2,
            label=f"cluster {label}",
        )

    if show_grid and bounds is not None:
        x_min, x_max, y_min, y_max = bounds

        for x in np.arange(x_min, x_max + bin_size, bin_size):
            plt.axvline(x, linestyle="--", alpha=0.15)

        for y in np.arange(y_min, y_max + bin_size, bin_size):
            plt.axhline(y, linestyle="--", alpha=0.15)

    all_x, all_y = [], []

    for trajs in cluster_trajs.values():
        for t in trajs:
            if align:
                t = t - t[0]
            all_x.append(t[:, 0])
            all_y.append(t[:, 1])

    if len(all_x) == 0:
        _mark_axes_origin()
        plt.show()
        return

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min, x_max = np.percentile(all_x, [0.5, 99.5])
    y_min, y_max = np.percentile(all_y, [0.5, 99.5])

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0:
        x_span = 1.0
    if y_span <= 0:
        y_span = 1.0
    x_pad = 0.05 * x_span
    y_pad = 0.05 * y_span

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    _mark_axes_origin()

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("rel_x (target-centered)")
    plt.ylabel("rel_y (target-centered)")
    plt.title("Trajectory clusters by stable-phase entry bin")

    plt.legend()
    plt.show()


def plot_clusters_grid_subplots(
    df,
    seg_ids,
    labels,
    bounds=None,
    bin_pairs=None,
    bin_size=50,
    align=True,
    show_grid=True,
    min_trajs=2,
    entry_by_seg=None,
):
    """
    One subplot per grid (cluster label) that has at least ``min_trajs`` trajectories.

    ``seg_ids`` and ``labels`` must align (one label per segment, same order as from
    ``build_traj_dataset`` / ``assign_stable_bins``). If ``bin_pairs`` is given, it must
    have the same length as ``labels``; each row is ``(bin_x, bin_y)`` for that sample,
    used for subplot titles. Pass ``bounds`` as the fourth argument, or use
    ``bounds=...`` / ``bin_pairs=...`` keywords.
    """
    if entry_by_seg is None:
        raise ValueError("plot_clusters_grid_subplots requires entry_by_seg (stable-entry map).")

    mapping = dict(zip(seg_ids, labels))
    unique_labels = np.unique(labels)

    def _truncate_from_entry(seg_id, traj):
        if entry_by_seg is None or seg_id not in entry_by_seg:
            return traj
        entry_xy = np.array(entry_by_seg[seg_id], dtype=float)
        d2 = np.sum((traj - entry_xy) ** 2, axis=1)
        start = int(np.argmin(d2))
        return traj[start:]

    cluster_trajs = {k: [] for k in unique_labels}
    for seg_id, seg_df in df.groupby("new_segment"):
        if seg_id not in mapping:
            continue
        label = mapping[seg_id]
        traj = seg_df.sort_values("bin_in_new_seg")[
            ["cur_ff_rel_x", "cur_ff_rel_y"]
        ].values
        traj = _truncate_from_entry(seg_id, traj)
        cluster_trajs[label].append((seg_id, traj))

    label_to_bin = {}
    if bin_pairs is not None:
        bin_pairs = np.asarray(bin_pairs)
        for lab, bp in zip(labels, bin_pairs):
            lab = int(lab)
            if lab not in label_to_bin:
                label_to_bin[lab] = (int(bp[0]), int(bp[1]))

    grid_labels = [
        lab for lab in unique_labels if len(cluster_trajs[lab]) >= min_trajs
    ]
    n = len(grid_labels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(
            0.5,
            0.5,
            f"No grid with ≥{min_trajs} trajectories",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.show()
        return

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    cmap = plt.get_cmap("tab20")

    def _to_plot_xy(traj):
        if align:
            return traj - traj[0]
        return traj

    def _entry_xy(seg_id, traj):
        if seg_id in entry_by_seg:
            return np.array(entry_by_seg[seg_id], dtype=float)
        return traj[0].astype(float)

    for ax_idx, label in enumerate(grid_labels):
        ax = axes_flat[ax_idx]
        pairs = cluster_trajs[label]
        color = cmap(ax_idx % cmap.N)

        for seg_id, traj in pairs:
            xy = _to_plot_xy(traj)
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                alpha=0.45,
                linewidth=1.5,
            )
            ex, ey = _entry_xy(seg_id, traj)
            if align:
                ex, ey = ex - traj[0, 0], ey - traj[0, 1]
            ax.scatter(ex, ey, color=color, alpha=0.7, s=16)

        trajs = [t for _, t in pairs]
        trajs_to_avg = [(t - t[0]) if align else t for t in trajs]
        mean_traj = mean_traj_arclength(trajs_to_avg, n_points=50)
        if mean_traj is not None:
            ax.plot(
                mean_traj[:, 0],
                mean_traj[:, 1],
                color=color,
                linewidth=2,
            )

        if show_grid and bounds is not None:
            x_min_b, x_max_b, y_min_b, y_max_b = bounds
            for x in np.arange(x_min_b, x_max_b + bin_size, bin_size):
                ax.axvline(x, linestyle="--", alpha=0.15)
            for y in np.arange(y_min_b, y_max_b + bin_size, bin_size):
                ax.axhline(y, linestyle="--", alpha=0.15)

        all_x, all_y = [], []
        for t in trajs:
            tt = t - t[0] if align else t
            all_x.append(tt[:, 0])
            all_y.append(tt[:, 1])
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        xm0, xm1 = np.percentile(all_x, [0.5, 99.5])
        ym0, ym1 = np.percentile(all_y, [0.5, 99.5])
        x_span = xm1 - xm0
        y_span = ym1 - ym0
        if x_span <= 0:
            x_span = 1.0
        if y_span <= 0:
            y_span = 1.0
        pad_x = 0.05 * x_span
        pad_y = 0.05 * y_span
        ax.set_xlim(xm0 - pad_x, xm1 + pad_x)
        ax.set_ylim(ym0 - pad_y, ym1 + pad_y)
        ax.set_aspect("equal", adjustable="box")
        _mark_axes_origin(ax, markersize=8)

        if int(label) in label_to_bin:
            bx, by = label_to_bin[int(label)]
            ax.set_title(f"bin ({bx}, {by}), n={len(trajs)}")
        else:
            ax.set_title(f"grid {label}, n={len(trajs)}")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle("Trajectories per spatial bin (stable-phase entry)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def collect_rolling_stds_and_ranges(df, window=5):
    all_stds, all_ranges = [], []
    for seg_id, seg_df in df.groupby("new_segment"):
        curv = seg_df.sort_values("bin_in_new_seg")["curv_of_traj"].values
        for i in range(len(curv) - window + 1):
            w = curv[i : i + window]
            if np.all(np.isnan(w)):
                continue
            all_stds.append(np.nanstd(w))
            all_ranges.append(np.nanmax(w) - np.nanmin(w))
    return np.array(all_stds), np.array(all_ranges)


def collect_entry_window_stats(df, window=5, std_thresh=0.002, range_thresh=None, min_len=5, n_persist=1):
    entry_stds, entry_ranges = [], []

    for seg_id, seg_df in df.groupby("new_segment"):
        seg_df = seg_df.sort_values("bin_in_new_seg")
        curv = seg_df["curv_of_traj"].values

        start = trajectory_clustering.find_stable_phase_entry_index(
            curv, std_thresh=std_thresh,
            range_thresh=range_thresh, min_points_after_entry=min_len,
            n_persist=n_persist,
        )

        if start is None:
            continue

        w = curv[start : start + window]
        entry_stds.append(np.nanstd(w))
        entry_ranges.append(np.nanmax(w) - np.nanmin(w))

    return np.array(entry_stds), np.array(entry_ranges)

def plot_single_trajectory(df, seg_id, entry_by_seg, window=5, std_thresh=0.002, range_thresh=None):
    seg_df = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
    curv = seg_df["curv_of_traj"].values
    xy = seg_df[["cur_ff_rel_x", "cur_ff_rel_y"]].values

    start = trajectory_clustering.find_stable_phase_entry_index(
        curv, std_thresh=std_thresh,
        range_thresh=range_thresh, min_points_after_entry=5,
    )

    traj_std = np.nanstd(curv)
    traj_range = np.nanmax(curv) - np.nanmin(curv)

    entry_std = np.nan
    entry_range = np.nan
    if start is not None and start + window <= len(curv):
        entry_window = curv[start:start + window]
        entry_std = np.nanstd(entry_window)
        entry_range = np.nanmax(entry_window) - np.nanmin(entry_window)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: spatial trajectory
    axes[0].plot(xy[:, 0], xy[:, 1], 'b-', alpha=0.6)
    if start is not None:
        axes[0].scatter(*xy[start], color='red', s=60, zorder=5, label=f'entry (t={start})')
        axes[0].plot(xy[start:, 0], xy[start:, 1], 'r-', linewidth=2, alpha=0.8)
    if seg_id in entry_by_seg:
        ex, ey = entry_by_seg[seg_id]
        axes[0].scatter(ex, ey, color='orange', s=80, zorder=6, label='entry_by_seg')
    _mark_axes_origin(axes[0])
    axes[0].set(title=f"seg {seg_id} trajectory", xlabel="rel_x", ylabel="rel_y")
    axes[0].set_aspect("equal")
    axes[0].legend()

    # right: curvature over time
    t = np.arange(len(curv))
    axes[1].plot(t, curv, 'k-', alpha=0.7, label='curvature')
    
    # rolling std and range
    roll_std = [np.nanstd(curv[i:i+window]) for i in range(len(curv)-window+1)]
    roll_range = [np.nanmax(curv[i:i+window]) - np.nanmin(curv[i:i+window]) 
                  for i in range(len(curv)-window+1)]
    axes[1].plot(t[:len(roll_std)], roll_std, 'b--', alpha=0.7, label='rolling std')
    axes[1].plot(t[:len(roll_range)], roll_range, 'g--', alpha=0.7, label='rolling range')
    axes[1].axhline(std_thresh, color='b', linestyle=':', label=f'std_thresh={std_thresh}')
    if range_thresh:
        axes[1].axhline(range_thresh, color='g', linestyle=':', label=f'range_thresh={range_thresh}')
    if start is not None:
        axes[1].axvline(start, color='red', linestyle='--', label=f'entry t={start}')
    stats_text = (
        f"traj std={traj_std:.4f}\n"
        f"traj range={traj_range:.4f}\n"
        f"entry std={entry_std:.4f}\n"
        f"entry range={entry_range:.4f}"
    )
    axes[1].text(
        0.02, 0.98, stats_text,
        transform=axes[1].transAxes,
        va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )
    _mark_axes_origin(axes[1], markersize=8)
    axes[1].set(title="curvature + rolling stats", xlabel="bin_in_new_seg")
    axes[1].legend(fontsize=8)

    plt.suptitle(f"seg {seg_id}")
    plt.tight_layout()
    plt.show()
    
    
def inspect_bin_trajectories(df, seg_ids, labels, entry_by_seg, bin_idx=0):
    """Print per-trajectory info for a given bin label."""
    mapping = dict(zip(seg_ids, labels))
    
    for seg_id, seg_df in df.groupby("new_segment"):
        if mapping.get(seg_id) != bin_idx:
            continue
        
        seg_df = seg_df.sort_values("bin_in_new_seg")
        curv = seg_df["curv_of_traj"].values
        xy = seg_df[["cur_ff_rel_x", "cur_ff_rel_y"]].values
        entry = entry_by_seg.get(seg_id)
        
        print(f"\nseg {seg_id} | n_bins={len(seg_df)} | entry={entry}")
        print(f"  curv: min={np.nanmin(curv):.4f} max={np.nanmax(curv):.4f} std={np.nanstd(curv):.4f}")
        print(f"  x range: [{xy[:,0].min():.1f}, {xy[:,0].max():.1f}]")
        print(f"  y range: [{xy[:,1].min():.1f}, {xy[:,1].max():.1f}]")
        
def find_segs_in_bin(entry_by_seg, seg_ids, labels, bin_idx=0, 
                     x_range=None, y_range=None):
    """
    List seg_ids in a given bin, optionally filtered by entry coordinate range.
    x_range / y_range: (min, max) tuples to zoom in on suspicious dots.
    """
    mapping = dict(zip(seg_ids, labels))
    
    for seg_id, (ex, ey) in entry_by_seg.items():
        if mapping.get(seg_id) != bin_idx:
            continue
        if x_range is not None and not (x_range[0] <= ex <= x_range[1]):
            continue
        if y_range is not None and not (y_range[0] <= ey <= y_range[1]):
            continue
        print(f"seg_id={seg_id}  entry=({ex:.1f}, {ey:.1f})")
        
        
def inspect_bin_one_by_one(df, seg_ids, labels, entry_by_seg,
                            bin_idx=0, window=5,
                            std_thresh=0.002, range_thresh=0.008):
    """Step through each trajectory in a bin interactively."""
    mapping = dict(zip(seg_ids, labels))
    segs_in_bin = [s for s in seg_ids if mapping.get(s) == bin_idx]

    for seg_id in segs_in_bin:
        plot_single_trajectory(df, seg_id, entry_by_seg,
                                window=window, std_thresh=std_thresh,
                                range_thresh=range_thresh)
        resp = input(f"seg_id={seg_id} | next? [enter] or quit [q]: ")
        if resp.strip().lower() == 'q':
            break


def inspect_bin_grid(
    df,
    seg_ids,
    labels,
    entry_by_seg,
    bin_idx=0,
    ncols=3,
    align=False,
    std_thresh=0.002,
    range_thresh=0.008,
    min_len=5,
):
    """
    Plot all trajectories in a bin in a grid, each annotated with seg_id and
    entry point. Entry points are taken from ``entry_by_seg`` when available;
    otherwise we infer them using stable-phase thresholds.
    """
    mapping = dict(zip(seg_ids, labels))
    segs_in_bin = [s for s in seg_ids if mapping.get(s) == bin_idx]

    n = len(segs_in_bin)
    if n == 0:
        print(f"No trajectories in bin {bin_idx}")
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.ravel()

    for idx, seg_id in enumerate(segs_in_bin):
        ax = axes_flat[idx]
        seg_df = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
        xy = seg_df[["cur_ff_rel_x", "cur_ff_rel_y"]].values
        curv = seg_df["curv_of_traj"].values

        if align:
            xy_plot = xy - xy[0]
        else:
            xy_plot = xy

        ax.plot(xy_plot[:, 0], xy_plot[:, 1], 'b-', alpha=0.6, linewidth=1)
        traj_std = np.nanstd(curv)
        traj_range = np.nanmax(curv) - np.nanmin(curv)
        start = trajectory_clustering.find_stable_phase_entry_index(
            curv,
            std_thresh=std_thresh,
            range_thresh=range_thresh,
            min_points_after_entry=min_len,
        )

        entry_xy_raw = None
        entry_std = np.nan
        entry_range = np.nan
        if start is not None and start + min_len <= len(curv):
            entry_window = curv[start:start + min_len]
            entry_std = np.nanstd(entry_window)
            entry_range = np.nanmax(entry_window) - np.nanmin(entry_window)
        if seg_id in entry_by_seg:
            entry_xy_raw = np.array(entry_by_seg[seg_id], dtype=float)
        else:
            if start is not None and 0 <= start < len(xy):
                entry_xy_raw = xy[start].astype(float)

        # mark entry point
        if entry_xy_raw is not None:
            ex, ey = entry_xy_raw
            if align:
                ex, ey = ex - xy[0, 0], ey - xy[0, 1]
            ax.scatter(ex, ey, color='red', s=40, zorder=5)

        _mark_axes_origin(ax, markersize=7)
        ax.set_aspect("equal")
        ax.set_title(f"seg {seg_id}", fontsize=8)
        ax.tick_params(labelsize=7)

        # annotate entry coords outside the plotting area to avoid obscuring traces
        if entry_xy_raw is not None:
            ex_raw, ey_raw = entry_xy_raw
            ax.text(
                1.02, 0.98,
                f"entry: ({ex_raw:.0f}, {ey_raw:.0f})",
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=6, color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75),
                clip_on=False,
            )

        stats_text = (
            f"std={traj_std:.4f}\n"
            f"range={traj_range:.4f}\n"
            f"entry_std={entry_std:.4f}\n"
            f"entry_range={entry_range:.4f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=6,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75),
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(f"Bin {bin_idx} — {n} trajectories", y=1.01)
    plt.tight_layout()
    plt.show()
    
def plot_rolling_std_range_dist(df, window=5, std_thresh=0.002, range_thresh=0.01):
    stds, ranges = collect_rolling_stds_and_ranges(df, window=window)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(stds, bins=100)
    axes[0].axvline(std_thresh, color='r', linestyle='--', label=f'std_thresh={std_thresh}')
    axes[0].set(xlabel="rolling std", ylabel="count", yscale="log", title="Std distribution")
    axes[0].legend()

    axes[1].hist(ranges, bins=100)
    axes[1].axvline(range_thresh, color='r', linestyle='--', label=f'range_thresh={range_thresh}')
    axes[1].set(xlabel="rolling range", yscale="log", title="Range distribution")
    axes[1].legend()

    axes[2].scatter(stds, ranges, alpha=0.05, s=2)
    axes[2].axvline(std_thresh, color='r', linestyle='--', label=f'std_thresh={std_thresh}')
    axes[2].axhline(range_thresh, color='b', linestyle='--', label=f'range_thresh={range_thresh}')
    axes[2].set(xlabel="rolling std", ylabel="rolling range", title="Std vs Range (joint)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_curvature_diagnostics(df, window=5, std_thresh=None, range_thresh=None,
                                min_len=5, n_persist=1):
    """All-windows distribution (top row) vs entry-window distribution (bottom row).
    std_thresh and range_thresh can be None to disable the threshold line."""
    stds, ranges = collect_rolling_stds_and_ranges(df, window=window)
    entry_stds, entry_ranges = collect_entry_window_stats(
        df, window=window, std_thresh=std_thresh,
        range_thresh=range_thresh, min_len=min_len, n_persist=n_persist,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (s, r, title_prefix, n) in enumerate([
        (stds, ranges, "All windows", len(stds)),
        (entry_stds, entry_ranges, "Entry window", len(entry_stds)),
    ]):
        axes[row, 0].hist(s, bins=100 if row == 0 else 50)
        if std_thresh is not None:
            axes[row, 0].axvline(std_thresh, color='r', linestyle='--',
                                 label=f'std_thresh={std_thresh}')
            axes[row, 0].legend()
        axes[row, 0].set(xlabel="rolling std", ylabel="count", yscale="log",
                         title=f"{title_prefix} std (n={n})")

        axes[row, 1].hist(r, bins=100 if row == 0 else 50)
        if range_thresh is not None:
            axes[row, 1].axvline(range_thresh, color='r', linestyle='--',
                                 label=f'range_thresh={range_thresh}')
            axes[row, 1].legend()
        axes[row, 1].set(xlabel="rolling range", yscale="log",
                         title=f"{title_prefix} range")

        axes[row, 2].scatter(s, r, alpha=0.05 if row == 0 else 0.4,
                             s=2 if row == 0 else 10)
        if std_thresh is not None:
            axes[row, 2].axvline(std_thresh, color='r', linestyle='--',
                                 label=f'std={std_thresh}')
        if range_thresh is not None:
            axes[row, 2].axhline(range_thresh, color='b', linestyle='--',
                                 label=f'range={range_thresh}')
        if std_thresh is not None or range_thresh is not None:
            axes[row, 2].legend()
        axes[row, 2].set(xlabel="std", ylabel="range", title=f"{title_prefix} joint")

    fig.suptitle(
        f"Curvature diagnostics | window={window}"
        + (f" | std_thresh={std_thresh}" if std_thresh is not None else "")
        + (f" | range_thresh={range_thresh}" if range_thresh is not None else "")
    )
    plt.tight_layout()
    plt.show()
    
    
# =========================
# ARC-LENGTH RESAMPLING
# =========================
def arclength_resample(traj, n_points=50):
    """Resample trajectory to n_points evenly spaced by arc length."""
    from scipy.interpolate import interp1d

    traj = np.asarray(traj, dtype=float)
    if len(traj) < 2:
        return None

    diffs = np.diff(traj, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])

    if cumlen[-1] == 0:  # degenerate trajectory (all same point)
        return None

    cumlen /= cumlen[-1]  # normalize to [0, 1]
    t_new = np.linspace(0, 1, n_points)

    resampled = np.stack(
        [interp1d(cumlen, traj[:, i])(t_new) for i in range(traj.shape[1])],
        axis=1,
    )
    return resampled


def mean_traj_arclength(trajs, n_points=50):
    """
    Compute mean trajectory using arc-length resampling.
    Handles variable-length trajectories without truncation bias.
    """
    resampled = [arclength_resample(t, n_points=n_points) for t in trajs]
    resampled = [r for r in resampled if r is not None]
    if len(resampled) == 0:
        return None
    return np.stack(resampled).mean(axis=0)