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


def compute_tuning_curves_pooled(flat_spikes, flat_stimulus_values, n_stimulus_bins=10, variable_type='continuous'):
    """
    Returns tuning curves with mean, SEM, and counts per bin.
    """
    if variable_type == 'continuous':
        bins = np.linspace(np.min(flat_stimulus_values), np.max(flat_stimulus_values), n_stimulus_bins + 1)
        bin_indices = np.digitize(flat_stimulus_values, bins) - 1
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    else:
        unique_vals = np.unique(flat_stimulus_values)
        bin_indices = np.searchsorted(unique_vals, flat_stimulus_values)
        bin_centers = unique_vals

    tuning_curves = {}
    for neuron_idx in range(flat_spikes.shape[1]):
        means = []
        sems = []
        counts = []
        for b in range(len(bin_centers)):
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


def plot_tuning_curves(tuning_curves, max_neurons_to_plot=None, max_per_fig=16, show_bin_counts=True):
    neurons = list(tuning_curves.items())
    total = len(neurons)
    if max_neurons_to_plot:
        neurons = neurons[:max_neurons_to_plot]
        total = len(neurons)
    for start in range(0, total, max_per_fig):
        chunk = neurons[start:start+max_per_fig]
        n = len(chunk)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axs = np.array([axs]).flatten() if n == 1 else axs.flatten()
        for ax in axs[n:]:
            ax.set_visible(False)
        
        for i, (neuron, (x, y, err, counts)) in enumerate(chunk):
            ax = axs[i]
            
            # Main tuning curve line
            line_color = '#1f77b4'  # nice blue
            ax.plot(x, y, '-o', color=line_color, markerfacecolor='white', markeredgecolor=line_color, linewidth=2, markersize=6)
            
            # Shaded error band
            upper = y + err
            lower = y - err
            ax.fill_between(x, lower, upper, color=line_color, alpha=0.2)
            
            # Optional: subtle error bar caps (without connecting lines)
            ax.errorbar(x, y, yerr=err, fmt='none', ecolor=line_color, elinewidth=1.2, capsize=4, alpha=0.6)

            ax.grid(True, linestyle='--', alpha=0.25)

            if show_bin_counts:
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                for idx, (xi, yi, ei, c) in enumerate(zip(x, y, err, counts)):
                    if np.isnan(yi) or np.isnan(ei):
                        continue
                    offset = 0.03 * y_range
                    y_pos = yi + ei + offset if idx % 2 else yi - ei - offset
                    va = 'bottom' if idx % 2 else 'top'
                    ax.text(xi, y_pos, str(c), ha='center', va=va, fontsize=8, color='gray')

            ax.set(title=f'Neuron {neuron}', xlabel='Stimulus', ylabel='Mean Firing Rate')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.show()
