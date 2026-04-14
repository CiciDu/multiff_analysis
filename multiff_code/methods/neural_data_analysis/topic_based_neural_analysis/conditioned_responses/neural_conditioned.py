import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def compute_psth(spikes_df, entry_times, seg_ids, labels, df,
                 n_bins=20, bin_width=0.05, pre=0.2):
    neuron_ids = spikes_df["cluster"].unique()
    trial_rates = {}
    n_pre_bins = int(pre / bin_width)

    for seg_id, label in zip(seg_ids, labels):
        t0 = entry_times.get(seg_id)
        if t0 is None:
            continue

        seg_df = df[df["new_segment"] == seg_id].sort_values("bin_in_new_seg")
        t_end = seg_df["t_center"].max()
        duration = t_end - t0

        if duration <= 0:
            continue

        pre_edges = np.linspace(t0 - pre, t0, n_pre_bins + 1)
        post_edges = np.arange(t0, t_end + bin_width, bin_width)

        if len(post_edges) < 2:
            continue

        seg_spikes = spikes_df[
            (spikes_df["time"] >= t0 - pre) &
            (spikes_df["time"] <  t_end)
        ].copy()

        for nid in neuron_ids:
            spike_times = seg_spikes[seg_spikes["cluster"] == nid]["time"].values

            pre_counts, _ = np.histogram(spike_times, bins=pre_edges)
            pre_rate = pre_counts / bin_width  # (n_pre_bins,)

            post_counts, _ = np.histogram(spike_times, bins=post_edges)
            post_rate = post_counts / bin_width
            t_old = np.linspace(0, 1, len(post_rate))
            t_new = np.linspace(0, 1, n_bins)
            post_rate_resampled = np.interp(t_new, t_old, post_rate)

            trial_rates[(seg_id, nid)] = np.concatenate([pre_rate, post_rate_resampled])

    t_bins_pre = np.linspace(-pre, 0, n_pre_bins, endpoint=False)
    t_bins_post = np.linspace(0, 1, n_bins)
    t_bins = np.concatenate([t_bins_pre, t_bins_post])

    psth, residuals = {}, {}
    for lbl in np.unique(labels):
        segs_in_bin = [s for s, l in zip(seg_ids, labels) if l == lbl]
        for nid in neuron_ids:
            R = np.array([trial_rates[(s, nid)] for s in segs_in_bin
                          if (s, nid) in trial_rates])
            if len(R) == 0:
                continue
            mean_r = R.mean(axis=0)
            delta_r = R - mean_r
            psth[(lbl, nid)] = mean_r
            residuals[(lbl, nid)] = delta_r

    return t_bins, psth, residuals, trial_rates


def compute_eta2_timecourse(psth, residuals, neuron_id, t_bins):
    """
    Time-resolved η²(t) via one-way ANOVA across bins at each timepoint.
    η²(t) = SS_between(t) / SS_total(t)
    """
    bins = sorted(set(k[0] for k in psth if k[1] == neuron_id))
    groups = []
    for b in bins:
        key = (b, neuron_id)
        if key not in residuals:
            continue
        R = residuals[key] + psth[key]  # reconstruct individual trials
        groups.append(R)

    T = len(t_bins)
    eta2_t = np.zeros(T)

    for t in range(T):
        samples = [g[:, t] for g in groups if g.shape[0] > 1]
        if len(samples) < 2:
            continue
        all_vals = np.concatenate(samples)
        grand_mean = all_vals.mean()
        ss_total = ((all_vals - grand_mean)**2).sum()
        ss_between = sum(len(s) * (s.mean() - grand_mean)**2 for s in samples)
        if ss_total > 0:
            eta2_t[t] = ss_between / ss_total

    return eta2_t


def compute_eta2_scalar(psth, residuals, trial_rates, seg_ids, neuron_id):
    """
    Scalar η² via ANOVA on time-averaged firing rate per trial.
    More robust than averaging η²(t) over time.
    """
    bins = sorted(set(k[0] for k in psth if k[1] == neuron_id))
    groups = []
    for b in bins:
        key = (b, neuron_id)
        if key not in residuals:
            continue
        R = (residuals[key] + psth[key]).mean(axis=1)  # mean over time -> (n_trials,)
        if len(R) > 1:
            groups.append(R)

    if len(groups) < 2:
        return np.nan

    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total = ((all_vals - grand_mean)**2).sum()
    ss_between = sum(len(s) * (s.mean() - grand_mean)**2 for s in groups)

    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def plot_residual_var_heatmap(residuals, neuron_ids=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    all_bins = sorted(set(k[0] for k in residuals))
    if neuron_ids is None:
        neuron_ids = sorted(set(k[1] for k in residuals))

    mat = np.full((len(all_bins), len(neuron_ids)), np.nan)
    for i, b in enumerate(all_bins):
        for j, nid in enumerate(neuron_ids):
            key = (b, nid)
            if key in residuals:
                mat[i, j] = residuals[key].var()

    im = ax.imshow(mat, aspect="auto", cmap="hot_r")
    ax.set_xticks(range(len(neuron_ids)))
    ax.set_xticklabels(neuron_ids, fontsize=8)
    ax.set_yticks(range(len(all_bins)))
    ax.set_yticklabels([f"bin {b}" for b in all_bins], fontsize=8)
    ax.set_xlabel("neuron")
    ax.set_ylabel("trajectory bin")
    plt.colorbar(im, ax=ax, label="residual var (Hz²)")
    return ax


def plot_fraction_explained(psth, residuals, trial_rates, seg_ids, neuron_ids=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    if neuron_ids is None:
        neuron_ids = sorted(set(k[1] for k in psth))

    eta2 = [compute_eta2_scalar(psth, residuals, trial_rates, seg_ids, nid)
            for nid in neuron_ids]
    eta2 = np.array(eta2)

    colors = ["steelblue" if not np.isnan(e) else "lightgray" for e in eta2]
    ax.bar(range(len(neuron_ids)), np.nan_to_num(eta2), color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(neuron_ids)))
    ax.set_xticklabels(neuron_ids, fontsize=8)
    ax.set_ylabel("η² (fraction explained)")
    ax.set_xlabel("neuron")
    ax.set_ylim(0, 1.0)
    return ax


def plot_psth_by_bin(t_bins, psth, residuals, trial_rates, seg_ids, neuron_id, axes=None):
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    ax_mean, ax_std, ax_eta = axes

    bins = sorted(set(k[0] for k in psth if k[1] == neuron_id))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(bins)))

    for b, color in zip(bins, cmap):
        key = (b, neuron_id)
        if key not in psth:
            continue
        mean_r = psth[key]
        ax_mean.plot(t_bins, mean_r, color=color, lw=2, label=f"bin {b}")

        if key in residuals:
            dr = residuals[key]
            sem = dr.std(axis=0) / np.sqrt(dr.shape[0])
            ax_mean.fill_between(t_bins, mean_r - sem, mean_r + sem,
                                 color=color, alpha=0.08)
            ax_std.plot(t_bins, dr.std(axis=0), color=color, lw=1.5)

    # single ANOVA-based η²(t) curve
    eta2_t = compute_eta2_timecourse(psth, residuals, neuron_id, t_bins)
    ax_eta.plot(t_bins, eta2_t, color="k", lw=2)

    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")

    ax_mean.set_ylabel("⟨r|s⟩ (Hz)")
    ax_mean.legend(fontsize=7, ncol=2)
    ax_std.set_ylabel("std(δr) (Hz)")
    ax_eta.set_ylabel("η²(t)")
    ax_eta.set_ylim(0, 1.0)
    ax_eta.set_xlabel("normalized time")
    ax_mean.set_title(f"neuron {neuron_id}")

    return axes


def plot_summary(t_bins, psth, residuals, trial_rates, seg_ids, neuron_ids=None):
    if neuron_ids is None:
        neuron_ids = sorted(set(k[1] for k in psth))

    top_nid = max(neuron_ids, key=lambda n: np.nanmean([
        residuals[(b, n)].var() for b in set(k[0] for k in residuals) if (b, n) in residuals
    ]))

    eta2_per_nid = {
        nid: compute_eta2_scalar(psth, residuals, trial_rates, seg_ids, nid)
        for nid in neuron_ids
    }
    best_nid = max(eta2_per_nid, key=lambda n: eta2_per_nid[n] if not np.isnan(eta2_per_nid[n]) else -1)

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    axes0 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    plot_psth_by_bin(t_bins, psth, residuals, trial_rates, seg_ids, top_nid, axes=axes0)
    axes0[0].set_title(f"neuron {top_nid} (highest residual var)")

    axes1 = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    plot_psth_by_bin(t_bins, psth, residuals, trial_rates, seg_ids, best_nid, axes=axes1)
    axes1[0].set_title(f"neuron {best_nid} (highest η²={eta2_per_nid[best_nid]:.2f})")

    # ax2 = fig.add_subplot(gs[3, 0])
    # plot_residual_var_heatmap(residuals, neuron_ids=neuron_ids, ax=ax2)

    ax2 = fig.add_subplot(gs[3, 0])
    plot_eta2_heatmap(psth, residuals, t_bins, neuron_ids=neuron_ids, ax=ax2)


    ax3 = fig.add_subplot(gs[3, 1])
    plot_fraction_explained(psth, residuals, trial_rates, seg_ids, neuron_ids=neuron_ids, ax=ax3)

    return fig

def plot_eta2_heatmap(psth, residuals, t_bins, neuron_ids=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if neuron_ids is None:
        neuron_ids = sorted(set(k[1] for k in psth))

    mat = np.full((len(neuron_ids), len(t_bins)), np.nan)
    for i, nid in enumerate(neuron_ids):
        eta2_t = compute_eta2_timecourse(psth, residuals, nid, t_bins)
        mat[i] = eta2_t

    vmax = np.nanpercentile(mat, 95)  # robust max
    im = ax.imshow(mat, aspect="auto", cmap="hot", vmin=0, vmax=vmax,
                   extent=[t_bins[0], t_bins[-1], len(neuron_ids) - 0.5, -0.5])
    ax.set_yticks(range(len(neuron_ids)))
    ax.set_yticklabels(neuron_ids, fontsize=8)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("neuron")
    ax.axvline(0, color="w", lw=0.8, ls="--")
    plt.colorbar(im, ax=ax, label="η²(t)")
    return ax