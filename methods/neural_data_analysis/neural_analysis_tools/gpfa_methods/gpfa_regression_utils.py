from contextlib import contextmanager
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
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
from elephant.gpfa import GPFA


import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


def get_latent_neural_data_for_trial(trajectories, current_seg, trial_length, spiketrain_corr_segs, align_at_beginning=True):

    traj_index = np.where(spiketrain_corr_segs == current_seg)[0][0]

    if align_at_beginning:
        gpfa_trial = trajectories[traj_index].T[:trial_length, :]
    else:
        gpfa_trial = trajectories[traj_index].T[-trial_length:, :]
    return gpfa_trial


def _regress_at_timepoint(X_t, Y_t, n_behaviors, kf, alphas):
    r2s = np.full(n_behaviors, np.nan)
    for b in range(n_behaviors):
        y = Y_t[:, b]
        model = RidgeCV(alphas=alphas)
        try:
            score = cross_val_score(
                model, X_t, y, cv=kf, scoring='r2', n_jobs=1)
            r2s[b] = score.mean()
        except Exception:
            pass
    return r2s


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def run_time_resolved_regression_variable_length_trials(
    gpfa_neural_trials, behav_trials, time_step=0.02, cv_folds=5,
    max_timepoints=None, align_at_beginning=True, n_jobs=-1
):
    """
    Run time-resolved regression for variable length trials.
    Returns:
        scores_by_time: np.ndarray, shape (max_timepoints, n_behaviors)
        times: np.ndarray, time vector
        trial_counts: np.ndarray, number of trials used at each timepoint (NaN for unused timepoints)
    """
    assert len(gpfa_neural_trials) == len(
        behav_trials), "Mismatch in number of trials"

    n_latent_dims = gpfa_neural_trials[0].shape[1]
    n_behaviors = behav_trials[0].shape[1]
    if max_timepoints is None:
        max_timepoints = max(trial.shape[0] for trial in gpfa_neural_trials)

    scores_by_time = np.full((max_timepoints, n_behaviors), np.nan)
    trial_counts = np.full(max_timepoints, np.nan)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    alphas = np.logspace(-6, 6, 13)

    valid_ts, XYs = [], []
    min_samples = max(cv_folds, n_latent_dims)
    for t_idx in range(max_timepoints):
        X_t, Y_t = [], []
        for latent, behavior in zip(gpfa_neural_trials, behav_trials):
            if latent.shape[0] > t_idx:
                if align_at_beginning:
                    t = t_idx
                    X_t.append(latent[t])
                    Y_t.append(behavior[t])
                else:
                    t = max_timepoints - 1 - t_idx
                    X_t.append(latent[- 1 - t_idx])
                    Y_t.append(behavior[- 1 - t_idx])
        if len(X_t) > min_samples:
            valid_ts.append(t)
            XYs.append((np.vstack(X_t), np.vstack(Y_t)))
            trial_counts[t] = len(X_t)

    # 🧠 Show progress bar while running regression in parallel
    with tqdm_joblib(tqdm(total=len(XYs), desc="Timepoints")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_regress_at_timepoint)(X_t, Y_t, n_behaviors, kf, alphas)
            for X_t, Y_t in XYs
        )

    for t, r2s in zip(valid_ts, results):
        scores_by_time[t, :] = r2s

    times = np.arange(max_timepoints) * time_step

    return scores_by_time, times, trial_counts


def plot_trial_counts_by_timepoint(gpfa_neural_trials, times, align_at_beginning=True):
    n_timepoints = len(times)
    trial_counts = np.zeros(n_timepoints, dtype=int)
    for t in range(n_timepoints):
        trial_counts[t] = sum(
            latent.shape[0] > t for latent in gpfa_neural_trials)
    if not align_at_beginning:
        # Reverse for segment-end alignment so time 0 is segment start
        trial_counts = trial_counts[::-1]
    plt.plot(times, trial_counts, color='black', marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Trials with data")
    plt.title("Number of trials with data at each timepoint")
    plt.show()


def plot_time_resolved_scores(scores_by_time, times, behavior_labels=None, trial_counts=None, show_counts_on_xticks=True):
    """
    Plot time-resolved regression R² scores over time for each behavior.

    Parameters:
    - scores_by_time: np.ndarray, shape (time, behaviors)
    - times: np.ndarray, time vector
    - behavior_labels: list of str, optional behavior names
    - trial_counts: np.ndarray, number of trials per timepoint
    - show_counts_on_xticks: bool, whether to show trial counts on x-tick labels
    """
    n_behaviors = scores_by_time.shape[1]
    n_behaviors_per_plot = 4
    xtick_labels = None

    if show_counts_on_xticks and trial_counts is not None:
        xtick_labels = [f"{t:.2f}\n({int(n)})" if not np.isnan(n) else f"{t:.2f}\n(n/a)"
                        for t, n in zip(times, trial_counts)]

    def finalize_plot():
        plt.axhline(0, color='gray', lw=3)
        plt.xlabel(
            'Time (s)' + (' and trial count' if show_counts_on_xticks else ''))
        plt.ylabel('Cross-validated $R^2$')
        plt.title('Time-Resolved Regression Performance')
        plt.ylim(-2, 1.03)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if xtick_labels is not None:
            plt.xticks(times, xtick_labels, ha='right', rotation=0)
        plt.show()

    for b in range(n_behaviors):
        if b % n_behaviors_per_plot == 0:
            if b > 0:
                finalize_plot()
            plt.figure(figsize=(10, 5))

        label = behavior_labels[b] if (
            behavior_labels is not None) else f'Behavior {b + 1}'
        plt.plot(times, scores_by_time[:, b], label=label)

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


def standardize_trials(trials):
    """Standardize each trial (list of arrays) independently."""
    scaler = StandardScaler()
    return [scaler.fit_transform(trial) for trial in trials]


def try_multiple_latent_dims_and_plot(dec, behav_trials, dims=[3, 5, 10, 15], time_step=0.02, cv_folds=5, max_timepoints=None,
                                      ):
    """Try multiple latent dimensionalities and plot R^2 curves."""
    results = {}
    for d in dims:
        dec.get_gpfa_traj(latent_dimensionality=d, exists_ok=False)

        dec.get_rebinned_behav_data(
        )
        dec.get_trialwise_gpfa_and_behav_data()
        scores_by_time, times, trial_counts = run_time_resolved_regression_variable_length_trials(
            dec.gpfa_neural_trials, behav_trials, time_step=time_step, cv_folds=cv_folds, max_timepoints=max_timepoints)
        results[d] = (scores_by_time, times)
        scores_by_time_df = pd.DataFrame(
            scores_by_time, columns=dec.rebinned_behav_data_columns)
        plot_time_resolved_scores(
            scores_by_time, times, behavior_labels=scores_by_time_df.columns, trial_counts=trial_counts)

    for k, v in results.items():
        scores_by_time, times = v
        plt.plot(times, np.nanmean(scores_by_time, axis=1), label=f'dim={k}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean R^2')
    plt.title('GPFA Regression Performance vs. Latent Dimensionality')
    plt.legend()
    plt.show()
    return results


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
