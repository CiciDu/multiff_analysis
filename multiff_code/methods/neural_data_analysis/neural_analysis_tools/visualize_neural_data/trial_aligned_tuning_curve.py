from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data, plot_modeling_result, tuning_curve, raster_plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_trial_aligned_tuning_curve_data(aligned_spike_trains):
    aligned_spike_trains = raster_plot.add_relative_times(
        aligned_spike_trains, reference_time_col='new_seg_start_time')
    return aligned_spike_trains


def compute_bin_edges(new_seg_info, n_bins):
    durations = (new_seg_info['new_seg_end_time'] -
                 new_seg_info['new_seg_start_time']).abs()
    bin_widths = durations / n_bins
    return durations, bin_widths


def bin_spikes(spikes_and_durations, n_bins):
    if spikes_and_durations.shape[0] == 0:
        return np.zeros(n_bins)
    # make sure that all spikes are between 0 and duration
    assert np.all(spikes_and_durations[:, 0] >= 0), 'spikes must be positive'
    assert np.all(
        spikes_and_durations[:, 0] <= spikes_and_durations[:, 1]), 'spikes must be less than duration'
    scaled_spikes_in_bins = np.floor(
        spikes_and_durations[:, 0]/spikes_and_durations[:, 1] * n_bins).astype(int)
    return np.bincount(scaled_spikes_in_bins, minlength=n_bins)


def compute_firing_records(clusters, segments, grouped, n_bins):
    """
    Computes per-bin firing rates for each (cluster_id, segment_id) pair.

    Parameters:
        clusters (iterable): List or array of cluster IDs.
        segments (iterable): List or array of segment IDs.
        grouped (pd.core.groupby.generic.DataFrameGroupBy): Grouped spike data by (cluster_id, segment_id).
        n_bins (int): Number of time bins to divide each segment into.

    Returns:
        pd.DataFrame: A DataFrame with columns: ['cluster_id', 'segment_id', 'time_bin', 'firing_rate']
    """
    records = []

    for cluster_id in clusters:
        for seg_id in segments:
            key = (cluster_id, seg_id)

            # Get spike data if available
            if key not in grouped.groups:
                continue

            spikes_and_durations = grouped.get_group(key).values
            if spikes_and_durations.size == 0:
                continue

            # Assumes duration is stored in first column
            duration = spikes_and_durations[0, 1]
            if duration == 0:
                continue

            counts = bin_spikes(spikes_and_durations, n_bins)
            bin_duration = duration / n_bins

            for bin_idx, count in enumerate(counts):
                records.append({
                    'cluster_id': cluster_id,
                    'segment_id': seg_id,
                    'time_bin': bin_idx,
                    'firing_rate': count / bin_duration
                })

    return pd.DataFrame(records)


def scale_firing_rates(df):
    def _scale(group):
        min_val = group['firing_rate'].min()
        max_val = group['firing_rate'].max()
        scale = max_val - min_val + 1e-8
        group['firing_rate'] = (group['firing_rate'] - min_val) / scale
        group['sem'] = group['sem'] / scale
        return group

    return df.groupby('cluster_id', group_keys=False).apply(_scale)


def plot_avg_firing(
    avg_df,
    title='Trial-Averaged Firing',
    ylabel='Avg Firing Rate',
    clusters_per_fig=8,
    event_x_position=None,
    show_se=True
):
    """
    Plot trial-averaged firing rates per cluster. Optionally adds shaded SE regions.

    Parameters:
    - avg_df (pd.DataFrame): DataFrame with 'cluster_id', 'time_bin', 'firing_rate', and optionally 'sem'.
    - title (str): Title prefix for the plots.
    - ylabel (str): Y-axis label.
    - clusters_per_fig (int): Number of clusters to show per figure.
    - event_x_position (float or list): Optional vertical line(s) marking event times.
    - show_se (bool): Whether to show shaded standard error region (±2*SEM).
    """
    unique_clusters = sorted(avg_df['cluster_id'].unique())
    total_clusters = len(unique_clusters)
    n_bins = avg_df['time_bin'].max() + 1  # assuming zero-indexed bins

    for i in range(0, total_clusters, clusters_per_fig):
        fig_clusters = unique_clusters[i:i + clusters_per_fig]
        fig, ax = plt.subplots(figsize=(8, 4))

        for cluster_id in fig_clusters:
            cluster_data = avg_df[avg_df['cluster_id'] == cluster_id]
            line, = ax.plot(
                cluster_data['time_bin'],
                cluster_data['firing_rate'],
                label=f'Cluster {cluster_id}'
            )

            if show_se and 'sem' in cluster_data:
                ax.fill_between(
                    cluster_data['time_bin'],
                    cluster_data['firing_rate'] - 2 * cluster_data['sem'],
                    cluster_data['firing_rate'] + 2 * cluster_data['sem'],
                    color=line.get_color(),
                    alpha=0.3
                )

        ax.set_xlabel('Normalized Time Bin')
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title} (Clusters {fig_clusters[0]}–{fig_clusters[-1]})")
        ax.set_xticks(np.arange(n_bins))

        if event_x_position is not None:
            if isinstance(event_x_position, (list, np.ndarray)):
                for pos in event_x_position:
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
            else:
                ax.axvline(x=event_x_position, color='red', linestyle='--', alpha=0.7)

        ax.legend(
            title="Clusters",
            ncol=2,
            fontsize='small',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0
        )
        fig.tight_layout()
        plt.show()


def trial_averaged_time_normalized_firing(
    aligned_spike_trains,
    n_bins=20,
    clusters_per_fig=8,
    max_clusters_to_plot=None,
    rescale_avg_firing_rate=True,
    event_x_position=None,
    show_se=True,
):
    """
    Computes and plots trial-averaged, time-normalized firing rates across segments.

    Parameters:
        aligned_spike_trains (pd.DataFrame): Contains spike data, including 'cluster', 'new_segment',
                                        'rel_spike_time', and 'new_seg_duration'.
        n_bins (int): Number of time bins to divide each segment into (default: 20).
        max_clusters_to_plot (int or None): If specified, limits number of clusters to include.
        rescale_avg_firing_rate (bool): Whether to normalize firing rates before plotting.

    Returns:
        None
    """
    # make sure all rel_spike_time are positive
    assert np.all(
        aligned_spike_trains['rel_spike_time'] >= 0), 'rel_spike_time must be positive'

    segments = aligned_spike_trains['new_segment'].unique()

    clusters = raster_plot._prepare_clusters(
        aligned_spike_trains, cluster_col='cluster',
        max_clusters_to_plot=max_clusters_to_plot
    )

    grouped = aligned_spike_trains.groupby(['cluster', 'new_segment'])[
        ['rel_spike_time', 'new_seg_duration']
    ]

    # Compute firing rate for each bin of each (cluster, segment)
    firing_df = compute_firing_records(
        clusters, segments, grouped, n_bins=n_bins)

    # Trial-averaged (across segments) firing rate for each cluster and time bin
    avg_firing_df = (
        firing_df
        .groupby(['cluster_id', 'time_bin'], sort=False)['firing_rate']
        .agg(['mean', 'sem'])
        .reset_index()
        .rename(columns={'mean': 'firing_rate', 'sem': 'sem'})
    )

    if rescale_avg_firing_rate:
        avg_firing_df = scale_firing_rates(avg_firing_df)

    plot_avg_firing(avg_firing_df, clusters_per_fig=clusters_per_fig, event_x_position=event_x_position, show_se=show_se)

    return firing_df, avg_firing_df
