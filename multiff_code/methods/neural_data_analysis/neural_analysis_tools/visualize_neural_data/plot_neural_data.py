from neural_data_analysis.neural_analysis_tools.model_neural_data import neural_data_modeling

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib import rc


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def plot_spike_times(ax, spikes_df, duration, unique_clusters, x_values_for_vline=[1], marker_size=8):
    spike_subset = spikes_df[(spikes_df['time'] >= duration[0]) & (
        spikes_df['time'] <= duration[1])]
    spike_time = spike_subset.time - duration[0]

    ax.scatter(spike_time, spike_subset.cluster, s=marker_size)
    for x in x_values_for_vline:
        ax.axvline(x=x, color='r', linestyle='--')
    ax.set_xlim([0, duration[1]-duration[0]])
    if len(unique_clusters) < 30:
        ax.set_yticks(unique_clusters)
        ax.set_yticklabels(unique_clusters)
    else:
        # take out part of unique clusters to label
        factor_to_take_out = math.ceil(len(unique_clusters) / 30)
        ax.set_yticks(unique_clusters[::factor_to_take_out])
        ax.set_yticklabels(unique_clusters[::factor_to_take_out])
    ax.set_title("Spikes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cluster")
    return ax


def make_overlaid_spike_plot(time_to_sample_from, spikes_df, unique_clusters, interval_half_length=1, max_rows_to_plot=2, marker_size=8):
    random_sample = np.random.choice(time_to_sample_from, max_rows_to_plot)
    ax = None
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length]
        ax = _add_to_overlaid_spike_plot(ax, spikes_df, duration, unique_clusters, x_values_for_vline=[
                                         interval_half_length], marker_size=marker_size)
        if i == max_rows_to_plot:
            break
    plt.show()


def _add_to_overlaid_spike_plot(ax, spikes_df, duration, unique_clusters, x_values_for_vline=[], marker_size=8):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spikes_df, duration, unique_clusters,
                              x_values_for_vline=x_values_for_vline, marker_size=marker_size)
    else:
        spike_subset = spikes_df[(spikes_df['time'] >= duration[0]) & (
            spikes_df['time'] <= duration[1])]
        spike_time = spike_subset.time - duration[0]
        ax.scatter(spike_time, spike_subset.cluster, s=marker_size)
    return ax


def make_individual_spike_plots(time_to_sample_from, spikes_df, unique_clusters, interval_half_length=1, max_plots=2):
    random_sample = np.random.choice(
        time_to_sample_from, size=200, replace=False)
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spikes_df, duration, unique_clusters,
                              x_values_for_vline=[interval_half_length])
        plt.show()
        plt.close(fig)

        if i == max_plots:
            break


def make_individual_spike_plot_from_target_cluster_VBLO(target_cluster_VBLO, spikes_df, unique_clusters=1, starting_row=100, max_plots=2):
    target_cluster_VBLO = target_cluster_VBLO.copy()
    subset = target_cluster_VBLO.iloc[starting_row:starting_row+max_plots]
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        # if the time between last_vis_time and caught_time is more than 5 seconds, then don't plot last visible time
        if row.caught_time - row.last_vis_time < 5:
            duration = [row.last_vis_time - 1, row.caught_time + 1]
        else:
            duration = [row.caught_time - 2, row.caught_time + 1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spikes_df, duration, unique_clusters, x_values_for_vline=[
                              row.last_vis_time-duration[0], row.caught_time-duration[0]])

        # annotate at row.last_vis_time-duration[0] as "last visible time"
        if row.caught_time - row.last_vis_time < 5:
            rel_last_visible_time = row.last_vis_time-duration[0]
            ax.annotate('last visible time', xy=(rel_last_visible_time, 0), xytext=(
                rel_last_visible_time + 0.01, len(unique_clusters) - 1))

        # annotate at row.caught_time-duration[0] as "caught time"
        rel_last_caught_time = row.caught_time-duration[0]
        ax.annotate('caught time', xy=(rel_last_caught_time, 0), xytext=(
            rel_last_caught_time + 0.01, len(unique_clusters) - 1))
        ax.set_title(f'Trial {row.target_index}')

        plt.show()
        plt.close(fig)

        if i == max_plots:
            break
    return


def plot_regression(final_behavioral_data, column, x_var, bins_to_plot=None, min_r_squared_to_plot=0.1):

    # # drop rows where either x_var or y_var is nan, and print the number of dropped rows
    # n_rows = len(x_var)
    # dropped_rows = x_var[np.isnan(x_var) | np.isnan(y_var)]
    # x_var = x_var[~np.isnan(x_var) & ~np.isnan(y_var)]
    # y_var = y_var[~np.isnan(x_var) & ~np.isnan(y_var)]
    # print(f"Dropped {len(dropped_rows)} rows out of {n_rows} rows for {column} due to nan values.")

    if bins_to_plot is None:
        bins_to_plot = np.arange(final_behavioral_data.shape[0])

    y_var = final_behavioral_data[column].values
    slope, intercept, r_value, r_squared, p_values, f_p_value, y_pred = neural_data_modeling.conduct_linear_regression(
        x_var, y_var)
    title_str = f"{column}, R: {round(r_value, 2)}, R^2: {round(r_squared, 3)}, overall_p: {round(f_p_value, 3)}"

    # if r_squared < min_r_squared_to_plot:
    #     print(title_str)
    #     return
    if f_p_value > 0.05:
        print(f"Warning: {column} is not significant")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Fit plot
    ax1.scatter(range(len(bins_to_plot)),
                y_var[bins_to_plot], s=3, label='true')
    ax1.plot(range(len(bins_to_plot)),
             y_pred[bins_to_plot], color='red', linewidth=0.3, alpha=0.8, label='fit')

    ax1.set_title(title_str, fontsize=13)
    ax1.set_xlabel("bin", fontsize=14)
    ax1.set_ylabel(column, fontsize=12)
    ax1.legend(loc='upper left', fontsize=14)

    # Plot 2: Pred vs True plot
    ax2.scatter(y_var[bins_to_plot], y_pred[bins_to_plot], s=5, label='pred')
    min_val = min(min(y_var[bins_to_plot]), min(y_pred[bins_to_plot]))
    max_val = max(max(y_var[bins_to_plot]), max(y_pred[bins_to_plot]))
    ax2.plot([min_val, max_val], [min_val, max_val],
             color='red', linewidth=1, label='y = x line')

    ax2.set_title(
        f"{column}, R: {round(r_value, 2)}, R^2: {round(r_squared, 3)}", fontsize=14)
    ax2.set_xlabel("True value", fontsize=14)
    ax2.set_ylabel("Pred value", fontsize=14)
    ax2.legend(loc='upper left', fontsize=14)

    if column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
        ax2.set_xlim(-1000, 1000)

    plt.tight_layout()
    plt.show()


def plot_fr_over_time(
    binned_spikes_df, bin_width, max_time=None,
    bins_per_aggregate=1, num_clusters_per_plot=5, plot_mean=True
):
    fr_df = _prepare_fr_data(
        binned_spikes_df, bin_width, bins_per_aggregate, max_time)
    cluster_cols = [col for col in fr_df.columns if col.startswith('cluster_')]
    _plot_fr_curves(
        fr_df, cluster_cols, num_clusters_per_plot, plot_mean)

    fr_mean = fr_df[cluster_cols].mean(axis=1)
    slope = np.polyfit(fr_df['time'], fr_mean, 1)[0]
    total_change = fr_mean.iloc[-1] - fr_mean.iloc[0]
    print(f'Slope: {slope}')
    print(f'Total change: {round(total_change, 4)}')


def _prepare_fr_data(binned_df, bin_width, bins_per_aggregate, time_array=None, max_time=None):
    df = binned_df.copy()
    if time_array is None:
        df['time'] = df['bin'] * bin_width
    else:
        df['time'] = time_array
    if max_time is not None:
        df = df[df['time'] <= max_time]

    df['new_bin2'] = np.arange(len(df)) // bins_per_aggregate
    df_agg = df.groupby('new_bin2').mean().reset_index(drop=False)
    return df_agg


def _plot_fr_curves(fr_df, cluster_cols, num_clusters_per_plot, plot_mean=True):
    """
    Plot firing rate (FR) curves without downsampling.

    Args:
        fr_df (pd.DataFrame): DataFrame with firing rates for clusters, must include 'time' column.
        cluster_cols (list): List of cluster column names to plot.
        num_clusters_per_plot (int): How many clusters to plot per figure.
        plot_mean (bool): Whether to plot mean firing rate across clusters.
        cluster_id: Unused in this function.

    Returns:
        None
    """
    time = fr_df['time']
    mean_fr = fr_df[cluster_cols].mean(axis=1)

    num_plots = int(np.ceil(len(cluster_cols) / num_clusters_per_plot))

    for i in range(num_plots):
        plt.figure(figsize=(8, 3.5))

        cluster_subset = cluster_cols[i *
                                      num_clusters_per_plot:(i + 1) * num_clusters_per_plot]

        for col in cluster_subset:
            plt.plot(time, fr_df[col], alpha=0.6, label=col)

        if plot_mean and len(cluster_subset) > 1:
            plt.plot(time, mean_fr, color='black', linewidth=2, label='mean')

        # Extract cluster numbers for the title
        cluster_numbers = [int(col.split('_')[1]) for col in cluster_subset]
        if len(cluster_subset) == 1:
            title = f'Firing Rate for Cluster {cluster_numbers[0]}'
        else:
            title = f'Firing Rate for Clusters {cluster_numbers[0]} to {cluster_numbers[-1]}'

        plt.title(title)
        plt.ylabel('FR')
        plt.xlabel('Time (s)')
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()
