import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


def plot_best_alpha_counts(df, title='Counts of Best Alpha Values'):
    counts = df['best_alpha'].value_counts()
    # Sort by index (best_alpha) ascending
    counts = counts.sort_index()

    plt.figure(figsize=(6, 3.5))
    plt.bar(counts.index.astype(str), counts.values, edgecolor='black')
    plt.xlabel('Best Alpha')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_trial_counts_by_timepoint(time_resolved_cv_scores, trial_column='trial_count'):
    # make sure that y axis starts from 0
    trial_counts = time_resolved_cv_scores[[
        'bin_mid_time', trial_column]].drop_duplicates().sort_values(by='bin_mid_time')
    plt.plot(trial_counts['bin_mid_time'],
             trial_counts[trial_column], color='black', marker='o')
    plt.ylim(0, max(trial_counts[trial_column]) + 10)
    plt.xlabel("Time (s)")
    plt.ylabel("Trials with data")
    plt.title("Number of trials with data at each timepoint")
    plt.show()


def _agg_info_across_folds(time_resolved_cv_scores, group_cols=['feature', 'new_bin']):
    group_cols = ['feature', 'new_bin']

    # Dynamically construct the aggregation dictionary
    agg_dict = {
        col: ['mean', 'std'] if col == 'r2' else 'mean'
        for col in ['r2', 'bin_mid_time', 'trial_count', 'train_trial_count', 'test_trial_count']
    }

    # Group and aggregate
    agg_df = (
        time_resolved_cv_scores
        .groupby(group_cols)
        .agg(agg_dict)
        .reset_index()
    )

    # Flatten columns
    agg_df.columns = [
        'r2_std' if col == ('r2', 'std')
        else 'r2' if col == ('r2', 'mean')
        else col[0] if col[1] == 'mean'
        else ''.join(col)  # fallback for anything unexpected
        for col in agg_df.columns
    ]

    return agg_df


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

    agg_df = _agg_info_across_folds(time_resolved_cv_scores)

    behaviorals = agg_df['feature'].unique()
    max_values_by_behavior = agg_df.groupby('feature').max()

    if score_threshold_to_plot is not None:
        good_behaviors = max_values_by_behavior['r2'] >= score_threshold_to_plot
        behaviorals = max_values_by_behavior[good_behaviors].index.values

    if rank_by_max_score:
        behaviorals = max_values_by_behavior.loc[behaviorals].sort_values(
            by='r2', ascending=False).index.tolist()
    else:
        behaviorals = list(behaviorals)

    n_behaviors_per_plot = 4
    xticks = None
    xtick_labels = None

    if show_counts_on_xticks:
        min_trial_counts = agg_df[['bin_mid_time', 'trial_count']].groupby(
            'bin_mid_time').min().reset_index()
        xticks = min_trial_counts['bin_mid_time']
        xtick_labels = [
            f"{row.bin_mid_time:.2f}\n({int(row.trial_count)})" if not np.isnan(row.trial_count)
            else f"{row.bin_mid_time:.2f}\n(n/a)"
            for row in min_trial_counts.itertuples()
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

        df_b = agg_df[agg_df['feature'] == behavior]
        plt.plot(df_b['bin_mid_time'], df_b['r2'], label=behavior)

    if any_plots:
        finalize_plot()


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
