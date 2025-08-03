from contextlib import contextmanager
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
import warnings
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
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor
from elephant.gpfa import GPFA
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


def plot_trial_counts_by_timepoint(time_resolved_cv_scores, trial_column='trial_count'):
    # make sure that y axis starts from 0
    trial_counts = time_resolved_cv_scores[['bin_mid_time', trial_column]].drop_duplicates().sort_values(by='bin_mid_time')
    plt.plot(trial_counts['bin_mid_time'],
             trial_counts[trial_column], color='black', marker='o')
    plt.ylim(0, max(trial_counts[trial_column]) + 10)
    plt.xlabel("Time (s)")
    plt.ylabel("Trials with data")
    plt.title("Number of trials with data at each timepoint")
    plt.show()


def _plot_time_resolved_regression(time_resolved_cv_scores, show_counts_on_xticks=True,
                                   event_time=None, features_to_plot=None, features_not_to_plot=None,
                                   rank_by_max_score=True,
                                   score_threshold_to_plot=None):
    """
    Plot time-resolved regression R² scores over time for each behavior.

    Parameters:
    - time_resolved_cv_scores: pd.DataFrame with columns like 'bin_mid_time', 'trial_count', behavior scores
    - show_counts_on_xticks: bool, whether to show trial counts on x-tick labels
    - event_time: float or None, vertical line at event (e.g., stimulus onset)
    - features_not_to_plot: list of str, columns to exclude from plotting
    - score_threshold_to_plot: float or None, threshold to plot only behaviors with scores (for at least one timepoint) above this threshold
    """
    
    print('time_resolved_cv_scores.shape', time_resolved_cv_scores.shape)
    time_resolved_cv_scores = time_resolved_cv_scores.groupby(['behavior', 'new_bin']).mean().reset_index(drop=False)
    print('time_resolved_cv_scores.shape', time_resolved_cv_scores.shape)

    behaviorals = time_resolved_cv_scores['behavior'].unique()
    max_values_by_behavior = time_resolved_cv_scores.groupby('behavior').max()
    
    if score_threshold_to_plot is not None:
        good_behaviors = max_values_by_behavior['r2'] >= score_threshold_to_plot
        behaviorals = max_values_by_behavior[good_behaviors].index.values

    if rank_by_max_score:
        behaviorals = max_values_by_behavior.loc[behaviorals].sort_values(by='r2', ascending=False).index.tolist()
    else:
        behaviorals = list(behaviorals)
    
    n_behaviors_per_plot = 4
    xticks = None
    xtick_labels = None

    if show_counts_on_xticks:
        unique_bins = time_resolved_cv_scores[['bin_mid_time', 'trial_count']].drop_duplicates().sort_values('bin_mid_time')
        xticks = unique_bins['bin_mid_time']
        xtick_labels = [
            f"{row.bin_mid_time:.2f}\n({int(row.trial_count)})" if not np.isnan(row.trial_count)
            else f"{row.bin_mid_time:.2f}\n(n/a)"
            for row in unique_bins.itertuples()
        ]

    def finalize_plot():
        plt.axhline(0, color='gray', lw=2)
        plt.xlabel(
            'Time (s)' + ('\nTrial count' if show_counts_on_xticks else ''))
        plt.ylabel('Cross-validated $R^2$')
        plt.title('Time-Resolved Regression Performance')
        plt.ylim(-2, 1.03)
        plt.legend(fontsize=10, loc='lower left')
        plt.grid(True)
        if xtick_labels is not None:
            plt.xticks(xticks,
                       xtick_labels, ha='right', rotation=0)
        if event_time is not None:
            plt.axvline(event_time, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    if features_not_to_plot is None:
        features_not_to_plot = [
            'new_bin', 'new_seg_duration', 'trial_count', 'bin_mid_time']
    else:
        features_not_to_plot = set(features_not_to_plot)

    if features_to_plot is None:
        features_to_plot = [
            feat for feat in behaviorals if feat not in features_not_to_plot]

    any_plots = False
    for b, behavior in enumerate(features_to_plot):
        if b % n_behaviors_per_plot == 0:
            if any_plots:
                finalize_plot()
            plt.figure(figsize=(8, 4))
            any_plots = True

        df_b = time_resolved_cv_scores[time_resolved_cv_scores['behavior'] == behavior]
        plt.plot(df_b['bin_mid_time'], df_b['r2'], label=behavior)

    if any_plots:
        finalize_plot()

# below are suggesed functions that i have yet to try


def plot_trial_point_distribution(pursuit_data):
    trial_points = pursuit_data.groupby('segment').count()['bin'].values
    # Compute bin edges for width = 1
    min_val = min(trial_points)
    max_val = max(trial_points)
    bins = np.arange(min_val, max_val + 2)  # +2 to include the last value

    plt.hist(trial_points, bins=bins, edgecolor='black')
    plt.title('Number of points of the trials')
    plt.xlabel('Number of points')
    plt.ylabel('Number of trials')
    plt.show()

    print('Number of trials:', len(trial_points))
    print('Number of points of the trials:', trial_points)


def print_trials_per_timepoint(gpfa_neural_trials, max_timepoints=None):
    if max_timepoints is None:
        max_timepoints = max(trial.shape[0] for trial in gpfa_neural_trials)
    counts = np.zeros(max_timepoints, dtype=int)
    for t in range(max_timepoints):
        for latent in gpfa_neural_trials:
            if latent.shape[0] > t:
                counts[t] += 1
    # print('Trials per timepoint:', counts)
    plt.figure(figsize=(10, 3))
    plt.plot(counts)
    plt.xlabel('Timepoint (no unit, aligned at beginning)')
    plt.ylabel('Number of trials')
    plt.title('Number of trials at each timepoint')
    plt.show()
    return counts


# def try_multiple_latent_dims_and_plot(dec, behav_trials, dims=[3, 5, 10, 15], time_step=0.02, cv_folds=5, max_timepoints=None,
#                                       ):
#     """Try multiple latent dimensionalities and plot R^2 curves."""
#     results = {}
#     for d in dims:
#         dec.get_gpfa_traj(latent_dimensionality=d, exists_ok=False)

#         dec.get_rebinned_behav_data(
#         )
#         dec.get_concat_data_for_regression()
#         scores_by_time, times, trial_counts = time_resolved_regression_cv(
#             dec.gpfa_neural_trials, behav_trials, time_step=time_step, cv_folds=cv_folds, max_timepoints=max_timepoints)
#         results[d] = (scores_by_time, times)
#         time_resolved_cv_scores = pd.DataFrame(
#             scores_by_time, columns=dec.rebinned_behav_data_columns)
#         _plot_time_resolved_regression(
#             scores_by_time, times, behavior_labels=time_resolved_cv_scores.columns, trial_counts=trial_counts)

#     for k, v in results.items():
#         scores_by_time, times = v
#         plt.plot(times, np.nanmean(scores_by_time, axis=1), label=f'dim={k}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Mean R^2')
#     plt.title('GPFA Regression Performance vs. Latent Dimensionality')
#     plt.legend()
#     plt.show()
#     return results


def plot_latents_and_behav_trials(gpfa_neural_trials, behav_trials, bin_width, n_trials=5):
    """Plot latent trajectories and behavioral variables for a few trials."""
    for i in range(min(n_trials, len(gpfa_neural_trials))):
        time_points = np.arange(gpfa_neural_trials[i].shape[0]) * bin_width
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        axs[0].plot(time_points, gpfa_neural_trials[i])
        axs[0].set_title(f'Latent Trajectory Trial {i}')
        axs[1].plot(time_points, behav_trials[i])
        axs[1].set_title(f'Behavioral Variables Trial {i}')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()


def check_for_nans_in_trials(trials, name='trials'):
    for i, trial in enumerate(trials):
        if np.isnan(trial).any():
            print(f'NaNs found in {name} trial {i}')
