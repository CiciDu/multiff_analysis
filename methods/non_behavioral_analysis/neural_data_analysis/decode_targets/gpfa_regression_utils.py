import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
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


# Assuming you've run:
# scores_by_time, times = time_resolved_regression_variable_length(...)

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score


def get_latent_neural_data_for_trial(trajectories, current_seg, trial_length, spiketrain_corr_segs, align_at_beginning=True):

    traj_index = np.where(spiketrain_corr_segs == current_seg)[0][0]

    if align_at_beginning:
        gpfa_trial = trajectories[traj_index].T[:trial_length, :]
    else:
        gpfa_trial = trajectories[traj_index].T[-trial_length:, :]
    return gpfa_trial


def time_resolved_regression_variable_length(gpfa_trials, behav_trials, time_step=0.02, cv_folds=5, max_timepoints=None):
    """
    Perform time-resolved regression with variable-length trials.

    Parameters:
    - gpfa_trials: list of arrays, each shape (trial_length, n_latent_dims)
    - behav_trials: list of arrays, each shape (trial_length, n_behaviors)
    - time_step: float, time bin size in seconds
    - cv_folds: int, cross-validation folds

    Returns:
    - scores_by_time: np.array, shape (max_timepoints, n_behaviors)
    - times: np.array, time vector for max trial length
    """

    n_latent_dims = gpfa_trials[0].shape[1]

    # Find max trial length
    if max_timepoints is None:
        max_timepoints = max(trial.shape[0] for trial in gpfa_trials)
    n_behaviors = behav_trials[0].shape[1]
    scores_by_time = np.full((max_timepoints, n_behaviors), np.nan)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for t in range(max_timepoints):
        # Collect all trials with data at time t

        X_t = []
        Y_t = []
        for latent, behavior in zip(gpfa_trials, behav_trials):
            if latent.shape[0] > t:  # trial has data at time t
                X_t.append(latent[t])
                Y_t.append(behavior[t])
        if len(X_t) <= max(cv_folds, n_latent_dims):
            # Not enough trials for cross-validation at this timepoint
            continue

        X_t = np.vstack(X_t)
        Y_t = np.vstack(Y_t)

        for b in range(n_behaviors):
            y = Y_t[:, b]
            model = RidgeCV(alphas=np.logspace(-6, 6, 13))
            try:
                scores = cross_val_score(model, X_t, y, cv=kf, scoring='r2')
                scores_by_time[t, b] = scores.mean()
            except ValueError:
                # Sometimes regression fails if data is degenerate at this t
                scores_by_time[t, b] = np.nan

    times = np.arange(max_timepoints) * time_step
    return scores_by_time, times


def plot_time_resolved_scores(scores_by_time, times, behavior_labels=None):
    """
    Plot time-resolved regression R² scores over time for each behavior.

    Parameters:
    - scores_by_time: np.array, shape (time, behaviors)
    - times: np.array, time vector
    - behavior_labels: list of str, optional behavior names
    """
    n_behaviors = scores_by_time.shape[1]

    n_behaviors_per_plot = 4

    for b in range(n_behaviors):

        if b == 0:
            plt.figure(figsize=(10, 5))

        elif b % n_behaviors_per_plot == 0:
            plt.plot(times, [1] * len(times), lw=2)
            plt.plot(times, [0] * len(times), lw=2)
            # plt.plot(0, )
            plt.xlabel('Time (s)')
            plt.ylabel('Cross-validated $R^2$')
            plt.title('Time-Resolved Regression Performance')
            plt.ylim(-2, 1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # new figure
            plt.figure(figsize=(10, 5))

        label = behavior_labels[b] if (
            behavior_labels is not None) else f'Behavior {b + 1}'
        plt.plot(times, scores_by_time[:, b], label=label)

    plt.plot(times, [1] * len(times), lw=2)
    plt.plot(times, [0] * len(times), lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-validated $R^2$')
    plt.title('Time-Resolved Regression Performance')
    plt.legend()
    plt.ylim(-2, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return
