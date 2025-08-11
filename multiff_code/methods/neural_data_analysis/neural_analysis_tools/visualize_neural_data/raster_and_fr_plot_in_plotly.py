import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data


def create_raster_plot_for_one_duration_in_plotly(spikes_df, reference_time, start_time, end_time,
                                                  max_clusters_to_plot=10, rel_hover_time=None):
    try:
        window_spikes, selected_clusters = _select_clusters_and_spikes_in_time_window(
            spikes_df, start_time, end_time, max_clusters_to_plot)

        if window_spikes.empty:
            return go.Figure()

        window_spikes = window_spikes.copy()
        window_spikes['rel_spike_time'] = window_spikes['time'] - \
            reference_time

        fig = go.Figure()
        for cluster_id, group in window_spikes.groupby('cluster'):
            fig.add_trace(go.Scatter(
                x=group['rel_spike_time'],
                y=[cluster_id] * len(group),
                mode='markers',
                marker=dict(size=3, color='black'),
                showlegend=False
            ))

        y_min, y_max = window_spikes['cluster'].min(
        ), window_spikes['cluster'].max()

        _add_vertical_line(fig, 0, y_min, y_max, color='red',
                           name='Reference Time', showlegend=False)

        if rel_hover_time is not None:
            _add_vertical_line(fig, rel_hover_time, y_min, y_max, color='blue',
                               dash='dash', showlegend=False, name='rel_hover_time')

        fig.update_xaxes(
            range=[start_time - reference_time, end_time - reference_time])
        fig.update_layout(
            title='Raster Plot (Neurons in Duration)',
            xaxis_title='Relative Spike Time (s)',
            yaxis_title='Cluster',
            height=400,
            showlegend=True
        )
        return fig

    except Exception as e:
        logging.error(f"Error creating raster plot: {e}", exc_info=True)
        return go.Figure()


def create_firing_rate_plot_for_one_duration_in_plotly(
        spikes_df, reference_time, start_time, end_time,
        bin_width=0.1, bins_per_aggregate=3,
        max_clusters_to_plot=10, rel_hover_time=None):
    """
    Create a firing rate plot over a given time window.

    Args:
        spikes_df: DataFrame with 'time' and 'cluster' columns.
        reference_time: Time for vertical reference line.
        start_time: Start time of the window.
        end_time: End time of the window.
        bin_width: Width of time bins for spike counting.
        bins_per_aggregate: Number of bins to average (downsampling).
        max_clusters_to_plot: Maximum number of clusters to plot.
        rel_hover_time: Optional vertical line time (blue line).

    Returns:
        Plotly Figure with firing rate curves.
    """
    try:
        # Select spikes and clusters within time window
        window_spikes, selected_clusters = _select_clusters_and_spikes_in_time_window(
            spikes_df, start_time, end_time, max_clusters_to_plot)

        if window_spikes.empty or len(selected_clusters) == 0:
            return go.Figure()

        # Calculate relative spike times to reference_time
        window_spikes = window_spikes.copy()
        window_spikes['rel_spike_time'] = window_spikes['time'] - \
            reference_time

        # Get binned spike counts per cluster
        time_bins, binned_df = neural_data_processing.prepare_binned_spikes_df(
            window_spikes, bin_width=bin_width)
        time_array = (time_bins[:-1] + time_bins[1:]) / 2

        # Aggregate firing rates over bins_per_aggregate bins
        fr_df = _prepare_fr_data(binned_df, bin_width,
                                 bins_per_aggregate, time_array=time_array)

        # Map cluster IDs to column names in binned_df/fr_df
        selected_cluster_cols = [f"cluster_{c}" for c in selected_clusters]

        # Sanity check for required columns
        if 'time' not in fr_df.columns or not set(selected_cluster_cols).intersection(fr_df.columns):
            logging.warning(
                "Prepared firing rate DataFrame missing expected columns.")
            return go.Figure()

        fig = go.Figure()

        # Plot each cluster's firing rate
        for cluster_col in selected_cluster_cols:
            if cluster_col in fr_df.columns:
                fig.add_trace(go.Scatter(
                    x=fr_df['time'],
                    y=fr_df[cluster_col],
                    mode='lines',
                    name=cluster_col,
                    line=dict(width=1)
                ))

        # Plot mean firing rate if multiple clusters
        if len(selected_cluster_cols) > 1:
            mean_fr = fr_df[selected_cluster_cols].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=fr_df['time'],
                y=mean_fr,
                mode='lines',
                name='Mean',
                line=dict(color='red', width=2, dash='dash')
            ))

        # Set y-axis limits
        y_min = 0
        y_max = fr_df[selected_cluster_cols].max().max()
        if y_max == 0:
            y_max = 1

        # Add vertical reference line at 0 (relative time)
        _add_vertical_line(fig, 0, y_min, y_max, color='red',
                           name='Reference Time', showlegend=False)

        # Add optional hover vertical line (relative to reference_time)
        if rel_hover_time is not None:
            _add_vertical_line(fig, rel_hover_time, y_min, y_max, color='blue',
                               dash='dash', showlegend=False, name='rel_hover_time')

        fig.update_layout(
            title='Firing Rate Over Time',
            xaxis_title='Time (s) relative to reference',
            yaxis_title='Firing Rate (Hz)',
            height=400,
            showlegend=True
        )

        return fig

    except Exception as e:
        logging.error(f"Error creating firing rate plot: {e}", exc_info=True)
        return go.Figure()


def _select_clusters_and_spikes_in_time_window(spikes_df, start_time, end_time, max_clusters_to_plot=10):
    """
    Select spikes within a time window and limit clusters.

    Args:
        spikes_df: DataFrame with spike data containing 'cluster' and 'time'.
        start_time: Start time of the window.
        end_time: End time of the window.
        max_clusters_to_plot: Max number of clusters to include.

    Returns:
        Tuple of (filtered spikes DataFrame, array of selected clusters).
    """
    unique_clusters = spikes_df['cluster'].unique()
    if max_clusters_to_plot and len(unique_clusters) > max_clusters_to_plot:
        selected_clusters = np.random.choice(
            unique_clusters, max_clusters_to_plot, replace=False)
    else:
        selected_clusters = unique_clusters

    filtered_spikes = spikes_df[
        (spikes_df['cluster'].isin(selected_clusters)) &
        (spikes_df['time'] >= start_time) &
        (spikes_df['time'] <= end_time)
    ].copy()

    return filtered_spikes, selected_clusters


def _add_vertical_line(fig, x_val, y_min, y_max, color, width=2, dash=None, name='', showlegend=True):
    """
    Helper to add vertical lines to plotly figures.

    Args:
        fig: Plotly figure object.
        x_val: X coordinate for the vertical line.
        y_min: Minimum y-value for line span.
        y_max: Maximum y-value for line span.
        color: Line color.
        width: Line width.
        dash: Line dash style ('dash', 'dot', etc.).
        name: Legend name.
        showlegend: Whether to show in legend.
    """

    fig.add_trace(go.Scatter(
        x=[x_val, x_val],
        y=[y_min, y_max],
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        showlegend=showlegend
    ))


def _prepare_fr_data(binned_df, bin_width, bins_per_aggregate, time_array=None, max_time=None):
    """
    Aggregate binned spike counts into firing rates averaged over bins_per_aggregate bins.

    Args:
        binned_df: DataFrame where each column is cluster spike counts per bin, and index corresponds to bins.
        bin_width: Width of each bin in seconds.
        bins_per_aggregate: Number of bins to average (downsample factor).
        time_array: Optional array of times corresponding to bins.
        max_time: Optional max time to filter data.

    Returns:
        DataFrame with averaged firing rates and 'time' column.
    """
    df = binned_df.copy()

    # Assign time column
    if time_array is None:
        df['time'] = df.index * bin_width
    else:
        if len(time_array) != len(df):
            logging.warning(
                f"Length of time_array ({len(time_array)}) does not match binned_df ({len(df)}), adjusting time_array.")
            df['time'] = np.linspace(
                time_array[0], time_array[-1], num=len(df), endpoint=False)
        else:
            df['time'] = time_array

    # Filter by max_time if specified
    if max_time is not None:
        df = df[df['time'] <= max_time]

    # Group bins to aggregate over bins_per_aggregate
    df['agg_bin'] = np.arange(len(df)) // bins_per_aggregate

    # Prepare aggregation dict: average all spike count columns plus time
    agg_dict = {col: 'mean' for col in df.columns if col not in ['agg_bin']}
    df_agg = df.groupby('agg_bin').agg(agg_dict).reset_index(drop=True)

    return df_agg
