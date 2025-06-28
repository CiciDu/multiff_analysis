import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
from non_behavioral_analysis.neural_data_analysis.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils

import warnings
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process


from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, cpu_count
import numpy as np
import itertools
from scipy.ndimage import uniform_filter1d


def run_time_resolved_baseline(X_trials, Y_trials, regression_type='ridge', ridge_alpha=1.0, n_components=None):
    # Optionally apply PCA
    if n_components is not None:
        pca = PCA(n_components=n_components)
        X_trials = [pca.fit_transform(trial) for trial in X_trials]
    # Use your time-resolved regression function
    from methods.non_behavioral_analysis.neural_data_analysis.gpfa_methods import gpfa_regression_utils
    scores_by_time, times = gpfa_regression_utils.run_time_resolved_regression_variable_length_trials(
        X_trials, Y_trials, time_step=0.02, cv_folds=5, max_timepoints=75, align_at_beginning=True
    )
    mean_r2 = np.nanmean(scores_by_time)
    return mean_r2, scores_by_time, times


def run_gpfa_experiment_time_resolved(
    dec, smoothing, sqrt, gpfa_dim, bin_width, ridge_alpha, regression_type, align_at_beginning, baseline=None, pca_components=None
):
    # Preprocess neural data
    neural_trials = [trial.copy() for trial in dec.gpfa_neural_trials]
    if smoothing > 1:
        neural_trials = [uniform_filter1d(
            trial, size=smoothing, axis=0) for trial in neural_trials]
    if sqrt:
        neural_trials = [np.sqrt(trial) for trial in neural_trials]
    # Standardize
    X_trials = [StandardScaler().fit_transform(trial)
                for trial in neural_trials]
    Y_trials = [StandardScaler().fit_transform(trial)
                for trial in dec.behav_trials]

    # Baseline: raw or PCA
    if baseline == 'raw':
        mean_r2, scores_by_time, times = run_time_resolved_baseline(
            X_trials, Y_trials, regression_type, ridge_alpha)
        return {
            'model': 'raw',
            'smoothing': smoothing,
            'sqrt': sqrt,
            'gpfa_dim': None,
            'bin_width': bin_width,
            'ridge_alpha': ridge_alpha,
            'regression_type': regression_type,
            'align_at_beginning': align_at_beginning,
            'mean_r2': mean_r2,
            'r2_by_time': scores_by_time.tolist(),
            'times': times.tolist()
        }
    elif baseline == 'pca':
        mean_r2, scores_by_time, times = run_time_resolved_baseline(
            X_trials, Y_trials, regression_type, ridge_alpha, n_components=pca_components)
        return {
            'model': f'pca_{pca_components}',
            'smoothing': smoothing,
            'sqrt': sqrt,
            'gpfa_dim': None,
            'bin_width': bin_width,
            'ridge_alpha': ridge_alpha,
            'regression_type': regression_type,
            'align_at_beginning': align_at_beginning,
            'mean_r2': mean_r2,
            'r2_by_time': scores_by_time.tolist(),
            'times': times.tolist()
        }
    # GPFA pipeline
    dec.get_gpfa_traj(latent_dimensionality=gpfa_dim, exists_ok=False)
    dec.get_trialwise_gpfa_and_behav_data()
    X_trials_gpfa = [StandardScaler().fit_transform(trial)
                     for trial in dec.gpfa_neural_trials]
    Y_trials_gpfa = [StandardScaler().fit_transform(trial)
                     for trial in dec.behav_trials]
    scores_by_time, times = gpfa_regression_utils.run_time_resolved_regression_variable_length_trials(
        X_trials_gpfa, Y_trials_gpfa, time_step=bin_width, cv_folds=5, max_timepoints=75, align_at_beginning=align_at_beginning
    )
    mean_r2 = np.nanmean(scores_by_time)
    return {
        'model': f'gpfa_{gpfa_dim}',
        'smoothing': smoothing,
        'sqrt': sqrt,
        'gpfa_dim': gpfa_dim,
        'bin_width': bin_width,
        'ridge_alpha': ridge_alpha,
        'regression_type': regression_type,
        'align_at_beginning': align_at_beginning,
        'mean_r2': mean_r2,
        'r2_by_time': scores_by_time.tolist(),
        'times': times.tolist()
    }
