from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
import statsmodels.api as sm
from matplotlib import rc
import scipy.interpolate as interpolate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=[1], marker_size=8):
    spike_subset = spike_df[(spike_df['time'] >= duration[0]) & (
        spike_df['time'] <= duration[1])]
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


def make_overlaid_spike_plot(time_to_sample_from, spike_df, unique_clusters, interval_half_length=1, max_rows_to_plot=2, marker_size=8):
    random_sample = np.random.choice(time_to_sample_from, max_rows_to_plot)
    ax = None
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length]
        ax = _add_to_overlaid_spike_plot(ax, spike_df, duration, unique_clusters, x_values_for_vline=[
                                         interval_half_length], marker_size=marker_size)
        if i == max_rows_to_plot:
            break
    plt.show()


def _add_to_overlaid_spike_plot(ax, spike_df, duration, unique_clusters, x_values_for_vline=[], marker_size=8):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters,
                              x_values_for_vline=x_values_for_vline, marker_size=marker_size)
    else:
        spike_subset = spike_df[(spike_df['time'] >= duration[0]) & (
            spike_df['time'] <= duration[1])]
        spike_time = spike_subset.time - duration[0]
        ax.scatter(spike_time, spike_subset.cluster, s=marker_size)
    return ax


def make_individual_spike_plots(time_to_sample_from, spike_df, unique_clusters, interval_half_length=1, max_plots=2):
    random_sample = np.random.choice(
        time_to_sample_from, size=200, replace=False)
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters,
                              x_values_for_vline=[interval_half_length])
        plt.show()
        plt.close(fig)

        if i == max_plots:
            break


def make_individual_spike_plot_from_target_cluster_VBLO(target_cluster_VBLO, spike_df, unique_clusters=1, starting_row=100, max_plots=2):
    target_cluster_VBLO = target_cluster_VBLO.copy()
    subset = target_cluster_VBLO.iloc[starting_row:starting_row+max_plots]
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        # if the time between last_vis_time and caught_time is more than 5 seconds, then don't plot last visible time
        if row.caught_time - row.last_vis_time < 5:
            duration = [row.last_vis_time - 1, row.caught_time + 1]
        else:
            duration = [row.caught_time - 2, row.caught_time + 1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=[
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
    title_str = f"{column}, R: {round(r_value, 2)}, R^2: {round(r_squared, 3)}, overall_p: {f_p_value}"

    if r_squared < min_r_squared_to_plot:
        print(title_str)
        return

    # plot fit
    plt.figure()
    plt.title(title_str, fontsize=20)
    plt.scatter(range(len(bins_to_plot)), y_var[bins_to_plot], s=3)
    plt.plot(range(len(bins_to_plot)),
             y_pred[bins_to_plot], color='red', linewidth=0.3, alpha=0.8)
    plt.xlabel("bin", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.show()

    # plot pred against true
    plt.figure()
    plt.scatter(y_var[bins_to_plot], y_pred[bins_to_plot], s=5)
    plt.title(
        f"{column}, R: {round(r_value, 2)}, R^2: {round(r_squared, 3)}", fontsize=20)
    min_val = min(min(y_var[bins_to_plot]), min(y_pred[bins_to_plot]))
    max_val = max(max(y_var[bins_to_plot]), max(y_pred[bins_to_plot]))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=1)
    plt.xlabel("True value", fontsize=14)
    plt.ylabel("Pred value", fontsize=14)
    if column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
        plt.xlim(-1000, 1000)
    plt.show()
