




# def run_time_resolved_baseline(X_trials, Y_trials, regression_type='ridge', ridge_alpha=1.0, n_components=None):
#     # Optionally apply PCA
#     if n_components is not None:
#         pca = PCA(n_components=n_components)
#         X_trials = [pca.fit_transform(trial) for trial in X_trials]
#     # Use your time-resolved regression function
#     from methods.neural_data_analysis.neural_analysis_tools.gpfa_methods import time_resolved_regression
#     scores_by_time, times, trial_counts = time_resolved_regression.time_resolved_regression_cv(
#         X_trials, Y_trials, time_step=0.02, cv_folds=5, max_timepoints=75, align_at_beginning=True
#     )
#     mean_r2 = np.nanmean(scores_by_time)
#     return mean_r2, scores_by_time, times

# The function above, if needs to be used, needs to be fixed again because we have changed behav_trials
# def run_gpfa_experiment_time_resolved(
#     dec, smoothing, sqrt, gpfa_dim, bin_width, ridge_alpha, regression_type, align_at_beginning, baseline=None, pca_components=None,
# ):
#     # Preprocess neural data
#     neural_trials = [trial.copy() for trial in dec.gpfa_neural_trials]
#     if smoothing > 1:
#         neural_trials = [uniform_filter1d(
#             trial, size=smoothing, axis=0) for trial in neural_trials]
#     if sqrt:
#         neural_trials = [np.sqrt(trial) for trial in neural_trials]
#     # Standardize
#     X_trials = [StandardScaler().fit_transform(trial)
#                 for trial in neural_trials]
#     Y_trials = [StandardScaler().fit_transform(trial)
#                 for trial in dec.behav_trials]

#     # Baseline: raw or PCA
#     if baseline == 'raw':
#         mean_r2, scores_by_time, times = run_time_resolved_baseline(
#             X_trials, Y_trials, regression_type, ridge_alpha)
#         return {
#             'model': 'raw',
#             'smoothing': smoothing,
#             'sqrt': sqrt,
#             'gpfa_dim': None,
#             'bin_width': bin_width,
#             'ridge_alpha': ridge_alpha,
#             'regression_type': regression_type,
#             'align_at_beginning': align_at_beginning,
#             'mean_r2': mean_r2,
#             'r2_by_time': scores_by_time.tolist(),
#             'times': times.tolist()
#         }
#     elif baseline == 'pca':
#         mean_r2, scores_by_time, times = run_time_resolved_baseline(
#             X_trials, Y_trials, regression_type, ridge_alpha, n_components=pca_components)
#         return {
#             'model': f'pca_{pca_components}',
#             'smoothing': smoothing,
#             'sqrt': sqrt,
#             'gpfa_dim': None,
#             'bin_width': bin_width,
#             'ridge_alpha': ridge_alpha,
#             'regression_type': regression_type,
#             'align_at_beginning': align_at_beginning,
#             'mean_r2': mean_r2,
#             'r2_by_time': scores_by_time.tolist(),
#             'times': times.tolist()
#         }
#     # GPFA pipeline
#     dec.get_gpfa_traj(latent_dimensionality=gpfa_dim, exists_ok=False)
#     dec.get_rebinned_behav_data(
#     )
#     dec.get_concat_data_for_regression()
#     X_trials_gpfa = [StandardScaler().fit_transform(trial)
#                      for trial in dec.gpfa_neural_trials]
#     Y_trials_gpfa = [StandardScaler().fit_transform(trial)
#                      for trial in dec.behav_trials]
#     scores_by_time, times, trial_counts = time_resolved_regression.time_resolved_regression_cv(
#         X_trials_gpfa, Y_trials_gpfa, time_step=bin_width, cv_folds=5, max_timepoints=75, align_at_beginning=align_at_beginning
#     )
#     mean_r2 = np.nanmean(scores_by_time)
#     return {
#         'model': f'gpfa_{gpfa_dim}',
#         'smoothing': smoothing,
#         'sqrt': sqrt,
#         'gpfa_dim': gpfa_dim,
#         'bin_width': bin_width,
#         'ridge_alpha': ridge_alpha,
#         'regression_type': regression_type,
#         'align_at_beginning': align_at_beginning,
#         'mean_r2': mean_r2,
#         'r2_by_time': scores_by_time.tolist(),
#         'times': times.tolist()
#     }
