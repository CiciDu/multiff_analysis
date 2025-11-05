from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import rc

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def prepare_aligned_spike_trains(new_seg_info, spikes_df,
                                 columns_to_preserve=['event_time', 'stop_time', 'prev_ff_caught_time']):
    """
    Aligns spike times to new segment information and appends event-related timing columns.

    Parameters:
        new_seg_info (DataFrame): Segment information containing at least 'new_segment', 'event_time', 'stop_time', and 'prev_ff_caught_time'.
        spikes_df (DataFrame): Raw spike times with segment IDs.
        bin_width (float): Width of the bin used for alignment in `concat_new_seg_info`.

    Returns:
        DataFrame: A DataFrame of aligned spike times with additional timing context.
    """
    # drop rows where new_seg_start_time or new_seg_send_time is NA, and print the number of rows dropped
    original_len = len(new_seg_info)
    new_seg_info = new_seg_info.dropna(
        subset=['new_seg_start_time', 'new_seg_end_time']).reset_index(drop=True)
    print(f'Dropped {original_len - len(new_seg_info)} rows out of {original_len} due to NA in new_seg_start_time or new_seg_end_time, '
          f'which is {(original_len - len(new_seg_info))/original_len*100:.2f}% of the original data')

    # needs to update new_seg_duration in case of new_seg_start_time and new_seg_end_time have been changed
    new_seg_info['new_seg_duration'] = new_seg_info['new_seg_end_time'] - \
        new_seg_info['new_seg_start_time']

    # assert that new_seg_duration is positive
    assert np.all(new_seg_info['new_seg_duration'] >
                  0), 'new_seg_duration must be positive'

    # Align spikes to new segment info
    aligned_spike_trains = pn_utils.concat_new_seg_info(
        spikes_df, new_seg_info,
    )

    cols_to_merge = [col for col in columns_to_preserve if (col in new_seg_info.columns) & (col not in aligned_spike_trains.columns)
                     ] + ['new_segment']

    aligned_spike_trains = aligned_spike_trains.merge(
        new_seg_info[cols_to_merge],
        on='new_segment',
        how='left'
    )

    # Rename time column for clarity
    aligned_spike_trains.rename(columns={'time': 'spike_time'}, inplace=True)

    return aligned_spike_trains


def add_relative_times(aligned_spike_trains, reference_time_col):
    """
    Adds columns for time values relative to a specified reference time.

    Parameters:
        aligned_spike_trains (DataFrame): DataFrame with absolute time columns.
        reference_time_col (str): Column name to use as the reference time.

    Returns:
        DataFrame: The original DataFrame with added relative time columns.
    """
    df = aligned_spike_trains

    # Set reference time
    df['reference_time'] = df[reference_time_col]

    # Columns to convert to relative times
    time_columns = [
        'new_seg_start_time',
        'new_seg_end_time',
        'spike_time',
        'prev_ff_caught_time',
        'stop_time',
        'event_time'
    ]

    for col in time_columns:
        if col not in df.columns:
            continue
        rel_col = f"rel_{col}"
        df[rel_col] = df[col] - df['reference_time']

    return df


def add_scaling_info(
    aligned_spike_trains,
    scale_anchor_col,
    scale_factor_upper_col='new_seg_end_time',
    scale_factor_lower_col='scale_anchor'
):
    """
    Adds scaled timing columns to `aligned_spike_trains` based on a scaling anchor and factor.

    Parameters:
        aligned_spike_trains (DataFrame): DataFrame containing spike timing and alignment information.
        scale_anchor_col (str): Column to use as the scale anchor.
        scale_factor_upper_col (str): Column for the upper bound of the scaling factor.
        scale_factor_lower_col (str): Column for the lower bound of the scaling factor.

    Returns:
        DataFrame: The original DataFrame with added scaling-related columns.
    """
    df = aligned_spike_trains

    # Define the anchor and scaling factor
    df['scale_anchor'] = df[scale_anchor_col]

    # make sure that scale_factor_upper_col is greater than scale_factor_lower_col
    assert np.all(df[scale_factor_upper_col] > df[scale_factor_lower_col]
                  ), 'scale_factor_upper_col must be greater than scale_factor_lower_col'

    df['scale_factor'] = df[scale_factor_upper_col] - df[scale_factor_lower_col]

    # Calculate relative anchor position (zero-point for scaling)
    df['rel_scale_anchor'] = df['scale_anchor'] - df['reference_time']

    # Columns to scale
    rel_time_cols = ['rel_new_seg_start_time',
                     'rel_new_seg_end_time', 'rel_spike_time']
    for col in rel_time_cols:
        scaled_col = f"sc_{col}"
        df[scaled_col] = (df[col] - df['rel_scale_anchor']) / \
            df['scale_factor']

    return df


def plot_rasters_and_fr(
    aligned_spike_trains, new_seg_info, binned_spikes_df, bin_width,
    cluster_col='cluster',
    bins_per_aggregate=1, plot_mean=True,
    max_clusters_to_plot=None, max_segments_to_plot=None, max_time=None
):
    segments = new_seg_info['new_segment'].unique()
    segments = segments[:max_segments_to_plot] if max_segments_to_plot else segments
    clusters = _prepare_clusters(
        aligned_spike_trains, cluster_col, max_clusters_to_plot)
    if max_time is None:
        max_time = new_seg_info['new_seg_end_time'].max()

    fr_df = plot_neural_data._prepare_fr_data(
        binned_spikes_df, bin_width, bins_per_aggregate, max_time)
    cluster_cols = [col for col in fr_df.columns if col.startswith('cluster_')]
    grouped_spikes = aligned_spike_trains.groupby([cluster_col, 'new_segment'])[
        'rel_spike_time']
    cluster_to_frcols = {int(
        col.split('_')[1]): col for col in cluster_cols if col.split('_')[1].isdigit()}

    for cluster_id in clusters:
        spike_data = {
            seg: grouped_spikes.get_group((cluster_id, seg)).values
            for seg in segments if (cluster_id, seg) in grouped_spikes.groups
        }
        fr_col = cluster_to_frcols.get(cluster_id)
        if fr_col:
            _plot_cluster_raster_and_fr(
                spike_data, segments, fr_df[fr_col], fr_df['time'], cluster_id, plot_mean)

    mean_fr_all = fr_df.mean(axis=1)
    slope = np.polyfit(fr_df['time'], mean_fr_all, 1)[0]
    total_change = mean_fr_all.iloc[-1] - mean_fr_all.iloc[0]
    print(f'Slope: {slope:.4f}, Total change: {total_change:.4f}')


def _plot_cluster_raster_and_fr(spike_data, segments, fr_curve, time, cluster_id, plot_mean):
    fig, (ax_raster, ax_fr) = plt.subplots(1, 2, figsize=(
        12, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})
    num_segments = len(segments)

    for i, seg in enumerate(segments):
        y_pos = num_segments - i
        ax_raster.vlines(spike_data.get(seg, []), y_pos -
                         0.5, y_pos + 0.5, color='black')
    ax_raster.axvline(0, color='red', linestyle='--', alpha=0.2, linewidth=2)
    step = 50
    ticks = np.arange(0, num_segments, step)
    ax_raster.set_yticks(num_segments - ticks)
    ax_raster.set_yticklabels(ticks)
    ax_raster.set_ylim(0.5, num_segments + 0.5)
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Segments")
    ax_raster.set_title(f"Raster: Cluster {cluster_id}")

    ax_fr.plot(time, fr_curve, label=f"Cluster {cluster_id}")
    if plot_mean:
        ax_fr.plot(time, fr_curve, color='black', linewidth=2, alpha=0.6)
    ax_fr.set_xlabel("Time (s)")
    ax_fr.set_ylabel("Firing Rate")
    ax_fr.set_title(f"FR: Cluster {cluster_id}")
    ax_fr.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def _prepare_segments(new_seg_info, max_segments_to_plot):
    """Get unique segment identifiers, truncated to max_segments_to_plot if specified."""
    segments = new_seg_info['new_segment'].unique()
    return segments[:max_segments_to_plot] if max_segments_to_plot else segments


def _prepare_clusters(aligned_spike_trains, cluster_col, max_clusters_to_plot):
    clusters = np.sort(aligned_spike_trains[cluster_col].unique())
    return clusters[:max_clusters_to_plot] if max_clusters_to_plot else clusters


def _plot_segment_event_lines(aligned_spike_trains, column, segments, color, label=None, scale_spike_times=False):
    if column not in aligned_spike_trains.columns:
        # see if the column without 'rel_' exists
        base_column = column.replace('rel_', '')
        if base_column in aligned_spike_trains.columns:
            aligned_spike_trains[f'rel_{base_column}'] = aligned_spike_trains[base_column] - aligned_spike_trains['reference_time']
        else:
            print(f"Warning: Column '{column}' not found in aligned_spike_trains; skipping event line plot.")
            return
    unique_events = aligned_spike_trains[[
        'new_segment', column]].drop_duplicates().copy()
    event_values = unique_events.set_index(
        'new_segment').reindex(segments)[column].values
    num_segments = len(segments)
    y_positions = num_segments - np.arange(len(segments))
    if scale_spike_times and label:
        label = f"Scaled {label.lower()}"
    plt.plot(event_values, y_positions, color=color, label=label)


def _set_xlim(aligned_spike_trains, xmin=None, xmax=None):
    if xmin is None:
        xmin = aligned_spike_trains['rel_new_seg_start_time'].min() - 0.25
    if xmax is None:
        xmax = aligned_spike_trains['rel_new_seg_end_time'].max() + 0.25

    plt.xlim(xmin, xmax)


def _finalize_legend_and_layout(x_lim=None):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize='small', borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.show()


def _plot_raster(spike_data, segments, cluster_id=None, title_prefix="Raster Plot"):
    num_segments = len(segments)

    for i, seg in enumerate(segments):
        spikes = spike_data.get(seg, [])
        y_pos = num_segments - i
        plt.vlines(spikes, y_pos - 0.5, y_pos + 0.5, color='black')

    step = 50
    ytick_positions = np.arange(0, num_segments, step)
    if len(ytick_positions) == 0:
        ytick_positions = [num_segments // 2]
    ytick_labels = ytick_positions
    ytick_positions = num_segments - np.array(ytick_positions)
    plt.yticks(ytick_positions, ytick_labels)

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Event')
    plt.xlabel("Time (s)")
    plt.ylabel("Segments")
    title = f"{title_prefix} {cluster_id}" if cluster_id is not None else title_prefix
    plt.title(title)
    plt.ylim(0.5, num_segments + 0.5)


def _plot_events(aligned_spike_trains, events_to_plot, segments, scale_spike_times):
    events_to_plot_dict = {
        'rel_new_seg_start_time': ['Segment start time', 'blue'],
        'rel_new_seg_end_time': ['Segment end time', 'purple'],
        'rel_stop_time': ['Stop time', 'green'],
        'rel_prev_ff_caught_time': ['Prev FF caught time', 'orange'],
        'rel_event_time': ['Event time', 'red'],
        'rel_stop_id_end_time': ['Stop end time', 'brown'],
        'rel_next_stop_time': ['Next stop time', 'cyan'],
    }
    for event in events_to_plot:
        if event in events_to_plot_dict:
            label, color = events_to_plot_dict[event]
            _plot_segment_event_lines(aligned_spike_trains, event, segments,
                                      color=color, label=label, scale_spike_times=scale_spike_times)
        else:
            print(f"Warning: Event '{event}' not recognized and will be skipped.")


def _rearrange_segments(aligned_spike_trains, col_to_rearrange_segments):
    assert col_to_rearrange_segments in aligned_spike_trains, f'{col_to_rearrange_segments} is required in aligned_spike_trains to rearrange by {col_to_rearrange_segments}'
    seg_info = aligned_spike_trains[[
        'new_segment', col_to_rearrange_segments]].drop_duplicates()
    seg_info.sort_values(by=col_to_rearrange_segments, inplace=True)
    segments = seg_info['new_segment'].unique()
    return segments


def _get_segments(aligned_spike_trains, col_to_rearrange_segments):
    if col_to_rearrange_segments is not None:
        segments = _rearrange_segments(
            aligned_spike_trains, col_to_rearrange_segments)
    else:
        segments = aligned_spike_trains['new_segment'].unique()
    return segments


def _scale_rel_times(aligned_spike_trains):
    # assert that all scale_factor are positive
    assert np.all(
        aligned_spike_trains['scale_factor'] > 0), 'scale_factor must be positive'
    # transform all rel times to be normalized by scale_factor
    rel_time_columns = [
        col for col in aligned_spike_trains.columns if col.startswith('rel_')]
    # make a copy to avoid modifying the original dataframe
    aligned_spike_trains = aligned_spike_trains.copy()
    for col in rel_time_columns:
        aligned_spike_trains[col] = (
            aligned_spike_trains[col] - aligned_spike_trains['rel_scale_anchor']) / aligned_spike_trains['scale_factor']
    return aligned_spike_trains


def plot_rasters(
    aligned_spike_trains,
    cluster_col='cluster',
    title_prefix="Raster Plot for Cluster",
    xmin=None,
    xmax=None,
    col_to_rearrange_segments=None,
    scale_spike_times=False,
    max_clusters_to_plot=None,
    max_segments_to_plot=None,
    events_to_plot=(
        'rel_new_seg_start_time',
        'rel_new_seg_end_time',
        'rel_stop_time',
        'rel_prev_ff_caught_time'
    )
):
    """
    Plot raster plots of spike times grouped by cluster and segment.

    Parameters
    ----------
    aligned_spike_trains : pd.DataFrame
        A DataFrame containing spike times and metadata, including relative spike times,
        cluster assignments, and segment identifiers.

    cluster_col : str, default='cluster'
        The name of the column indicating cluster IDs.

    title_prefix : str, default="Raster Plot for Cluster"
        A prefix for the plot title; the cluster ID will be appended.

    xmin : float or None, optional
        Lower limit of the x-axis. If None, determined automatically.

    xmax : float or None, optional
        Upper limit of the x-axis. If None, determined automatically.

    col_to_rearrange_segments : str, default='rel_stop_time'
        The column used to order segments on the y-axis.

    scale_spike_times : bool, default=False
        If True, normalize spike times and event times within each segment.

    max_clusters_to_plot : int or None, optional
        Maximum number of clusters to include in the plot. If None, all clusters are used.

    max_segments_to_plot : int or None, optional
        Maximum number of segments to plot per cluster. If None, all segments are used.

    events_to_plot : list of str, optional
        Column names of events (e.g., start/end times, behavioral markers) to be drawn
        as vertical lines on the plots.

    Returns
    -------
    None
        Displays one raster plot per cluster using matplotlib.
    """
    # Normalize spike times if requested
    if scale_spike_times:
        aligned_spike_trains = _scale_rel_times(aligned_spike_trains)

    # Get and optionally truncate list of segments
    segments = _get_segments(aligned_spike_trains, col_to_rearrange_segments)
    if max_segments_to_plot:
        segments = segments[:max_segments_to_plot]

    # Identify clusters to plot
    clusters = _prepare_clusters(
        aligned_spike_trains,
        cluster_col,
        max_clusters_to_plot
    )

    # Group data by cluster and segment
    grouped_spikes = aligned_spike_trains.groupby(
        [cluster_col, 'new_segment']
    )['rel_spike_time']

    # Create one plot per cluster
    for cluster_id in clusters:
        plt.figure(figsize=(8, 4))

        spike_data = {
            seg: grouped_spikes.get_group((cluster_id, seg)).values
            for seg in segments
            if (cluster_id, seg) in grouped_spikes.groups
        }

        _plot_raster(spike_data, segments, cluster_id, title_prefix)
        _plot_events(aligned_spike_trains, events_to_plot,
                     segments, scale_spike_times)
        _set_xlim(aligned_spike_trains, xmin, xmax)
        _finalize_legend_and_layout()
