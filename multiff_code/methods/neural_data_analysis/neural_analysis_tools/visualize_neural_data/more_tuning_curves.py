from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data, plot_modeling_result, find_tuning_curves, raster_plot, trial_aligned_tuning_curves


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
from sklearn.linear_model import LinearRegression

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def _fit_cosine_tuning_curves(flat_spikes, flat_angles_deg, n_points=100):
    """
    Fits a cosine/sine regression model to each neuron's response over circular angles.

    Parameters:
    - flat_spikes: array (n_samples x n_neurons) of firing/spike data
    - flat_angles_deg: array (n_samples,) of angle in degrees (0-360)
    - n_points: number of points for smooth prediction (default: 100)

    Returns:
    - tuning_curves: dict of neuron_idx: (pred_angles_deg, predicted_rates)
    """
    theta_rad = np.deg2rad(flat_angles_deg)
    X = np.column_stack([np.cos(theta_rad), np.sin(theta_rad)])

    pred_angles = np.linspace(0, 360, n_points)
    pred_rad = np.deg2rad(pred_angles)
    X_pred = np.column_stack([np.cos(pred_rad), np.sin(pred_rad)])

    tuning_curves = {}
    r2_scores = {}

    for neuron_idx in range(flat_spikes.shape[1]):
        y = flat_spikes[:, neuron_idx]

        # Fit model
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X_pred)
        y_fit_train = model.predict(X)

        # Compute R² on training data
        r2 = r2_score(y, y_fit_train)

        tuning_curves[neuron_idx] = (pred_angles, y_pred)
        r2_scores[neuron_idx] = r2

    return tuning_curves, r2_scores

def fit_cosine_tuning_curves(concat_neural_trials, concat_behav_trials, var_of_interest, **kwargs):
    flat_spikes = concat_neural_trials.filter(regex='cluster_').values
    flat_stimulus_values = concat_behav_trials[var_of_interest].values.flatten()
    tuning_curves, r2_scores = _fit_cosine_tuning_curves(flat_spikes, flat_stimulus_values, **kwargs)
    find_tuning_curves.plot_tuning_curves(tuning_curves, r2_scores=r2_scores)