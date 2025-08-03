from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data, plot_modeling_result

import sys
import os
import sys
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


def _compute_tuning_curves_pooled(
    flat_spikes,
    flat_stimulus_values,
    n_stimulus_bins=10,
    variable_type='continuous',
    binning_method='equal',
    exclude_edge_bins=False
):
    """
    Computes tuning curves (mean, SEM, count) for neurons over stimulus bins.

    Parameters:
    - flat_spikes: array (n_samples x n_neurons) of firing/spike data
    - flat_stimulus_values: array (n_samples,) of stimulus values (e.g. angle)
    - n_stimulus_bins: number of bins for the tuning curve
    - variable_type: 'continuous' or 'categorical'
    - binning_method: 'equal' or 'quantile'
    - exclude_edge_bins: whether to exclude the first and last bins (default: False)

    Returns:
    - tuning_curves: dict of neuron_idx: (bin_centers, means, sems, counts)
    """
    if variable_type == 'continuous':
        if binning_method == 'equal':
            bins = np.linspace(np.min(flat_stimulus_values), np.max(flat_stimulus_values), n_stimulus_bins + 1)
        elif binning_method == 'quantile':
            bins = np.quantile(flat_stimulus_values, np.linspace(0, 1, n_stimulus_bins + 1))
        else:
            raise ValueError(f"Unknown binning_method '{binning_method}'")

        bin_indices = np.digitize(flat_stimulus_values, bins) - 1
        bin_indices[bin_indices == n_stimulus_bins] = n_stimulus_bins - 1  # edge case
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Exclude edges if needed
        if exclude_edge_bins:
            valid_bins = range(1, n_stimulus_bins - 1)
            bin_centers = bin_centers[1:-1]
        else:
            valid_bins = range(n_stimulus_bins)

    else:  # categorical
        unique_vals = np.unique(flat_stimulus_values)
        bin_indices = np.searchsorted(unique_vals, flat_stimulus_values)
        bin_centers = unique_vals
        if exclude_edge_bins:
            valid_bins = range(1, len(unique_vals) - 1)
            bin_centers = bin_centers[1:-1]
        else:
            valid_bins = range(len(unique_vals))

    tuning_curves = {}
    for neuron_idx in range(flat_spikes.shape[1]):
        means = []
        sems = []
        counts = []
        for b in valid_bins:
            mask = (bin_indices == b)
            data = flat_spikes[mask, neuron_idx]
            counts.append(data.size)
            if data.size > 0:
                means.append(np.mean(data))
                sems.append(np.std(data, ddof=1) / np.sqrt(data.size))
            else:
                means.append(np.nan)
                sems.append(np.nan)
        tuning_curves[neuron_idx] = (bin_centers, np.array(means), np.array(sems), np.array(counts))
    return tuning_curves


def compute_and_plot_tuning_curves_pooled(concat_neural_trials, concat_behav_trials, var_of_interest, **kwargs):
    flat_spikes = concat_neural_trials.filter(regex='cluster_').values
    flat_stimulus_values = concat_behav_trials[var_of_interest].values.flatten()
    tuning_curves = _compute_tuning_curves_pooled(flat_spikes, flat_stimulus_values, 
                                                            **kwargs)
    plot_tuning_curves(tuning_curves)


def plot_tuning_curves(tuning_curves, r2_scores=None, max_neurons_to_plot=None, max_per_fig=16, show_bin_counts=True):
    neurons = list(tuning_curves.items())
    total = len(neurons)
    if max_neurons_to_plot:
        neurons = neurons[:max_neurons_to_plot]
        total = len(neurons)

    for start in range(0, total, max_per_fig):
        chunk = neurons[start:start+max_per_fig]
        n = len(chunk)
        cols = 4
        rows = int(np.ceil(n / cols))

        scale = 3.5
        fig, axs = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
        axs = np.array([axs]).flatten() if n == 1 else axs.flatten()
        for ax in axs[n:]:
            ax.set_visible(False)

        for i, (neuron, data) in enumerate(chunk):
            ax = axs[i]

            # Unpack depending on tuple length
            if len(data) == 4:
                x, y, err, counts = data
            elif len(data) == 2:
                x, y = data
                err = np.zeros_like(y)
                counts = None
            else:
                raise ValueError(f"Unsupported data tuple length {len(data)} for neuron {neuron}")

            line_color = '#1f77b4'  # nice blue
            ax.plot(x, y, '-o', color=line_color, markerfacecolor='white', markeredgecolor=line_color, linewidth=2, markersize=6)

            if err is not None and len(err) == len(y):
                upper = y + err
                lower = y - err
                ax.fill_between(x, lower, upper, color=line_color, alpha=0.2)
                ax.errorbar(x, y, yerr=err, fmt='none', ecolor=line_color, elinewidth=1.2, capsize=4, alpha=0.6)

            ax.grid(True, linestyle='--', alpha=0.25)

            if show_bin_counts and counts is not None:
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                for idx, (xi, yi, ei, c) in enumerate(zip(x, y, err, counts)):
                    if np.isnan(yi) or np.isnan(ei):
                        continue
                    offset = 0.03 * y_range
                    y_pos = yi + ei + offset if idx % 2 else yi - ei - offset
                    va = 'bottom' if idx % 2 else 'top'
                    ax.text(xi, y_pos, str(c), ha='center', va=va, fontsize=8, color='gray')

            title=f'Neuron {neuron}'
            if r2_scores and neuron in r2_scores:
                title += f' | R² = {r2_scores[neuron]:.3f}'
                
            ax.set(title=title, xlabel='Stimulus', ylabel='Mean Firing Rate')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_facecolor('#fafafa')

        plt.tight_layout()
        plt.show()


