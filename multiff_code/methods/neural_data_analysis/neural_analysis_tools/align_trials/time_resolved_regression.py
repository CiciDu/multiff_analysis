from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
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
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils
import warnings
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
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


def time_resolved_regression_cv(
    concat_neural_trials, concat_behav_trials, cv_folds=5, n_jobs=-1,
    alphas=np.logspace(-6, 6, 13), nested=False,
):
    assert concat_neural_trials[['new_segment', 'new_bin']].equals(
        concat_behav_trials[['new_segment', 'new_bin']]
    ), "Mismatch in data dimensions"

    n_behaviors = concat_behav_trials.shape[1]

    new_bins = np.sort(concat_neural_trials['new_bin'].unique())
    kf = KFold(n_splits=cv_folds, shuffle=True)

    neural_data_only = concat_neural_trials[
        [col for col in concat_neural_trials.columns if col.startswith(
            'dim_') or col == 'new_bin']
    ]

    neural_grouped = neural_data_only.groupby('new_bin')
    behav_grouped = concat_behav_trials.groupby('new_bin')

    used_new_bins = []
    XYs = []
    trial_counts = []

    for new_bin in new_bins:
        try:
            X_df = neural_grouped.get_group(new_bin)
            Y_df = behav_grouped.get_group(new_bin)
        except KeyError:
            continue

        X = X_df.values
        Y = Y_df.values

        if X.shape[0] < cv_folds:
            continue

        used_new_bins.append(new_bin)
        trial_counts.append(len(X))
        XYs.append((X, Y))

    if not XYs:
        raise RuntimeError("No time bins had sufficient data for regression.")

    with tqdm_joblib(tqdm(total=len(XYs), desc="Timepoints")):
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_regress_at_timepoint)(
                X_t, Y_t, n_behaviors, kf, alphas, nested)
            for X_t, Y_t in XYs
        )

    r2s, all_best_alphas = zip(*results)
    behavior_names = concat_behav_trials.columns

    r2s_flat, alphas_flat = [], []
    all_new_bins, all_features, all_folds, trial_count_repeated = [], [], [], []

    for i, (new_bin, r2_mat, alpha_mat) in enumerate(zip(used_new_bins, r2s, all_best_alphas)):
        for b, feature in enumerate(behavior_names):
            n_folds_actual = len(r2_mat[b])
            for fold in range(n_folds_actual):
                r2s_flat.append(r2_mat[b][fold])
                alphas_flat.append(alpha_mat[b][fold])
                all_new_bins.append(new_bin)
                all_features.append(feature)
                all_folds.append(fold)
                trial_count_repeated.append(trial_counts[i])

    time_resolved_cv_scores = pd.DataFrame({
        'feature': all_features,
        'fold': all_folds,
        'new_bin': all_new_bins,
        'trial_count': np.repeat(trial_counts, n_behaviors * cv_folds),
        'r2': r2s_flat,
        'best_alpha': alphas_flat
    })

    return time_resolved_cv_scores


def _regress_at_timepoint(X_t, Y_t, n_behaviors, kf, alphas, nested=False):
    r2s = []
    best_alphas = []

    for b in range(n_behaviors):
        y = Y_t[:, b]

        if nested:
            # --- Nested CV ---
            inner_r2s = []
            inner_best_alphas = []

            for train_idx, test_idx in kf.split(X_t):
                X_train, X_test = X_t[train_idx], X_t[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                ridge_cv = RidgeCV(alphas=alphas, scoring='r2')
                ridge_cv.fit(X_train, y_train)
                best_alpha = ridge_cv.alpha_
                inner_best_alphas.append(best_alpha)

                model = Ridge(alpha=best_alpha)
                model.fit(X_train, y_train)
                inner_r2s.append(r2_score(y_test, model.predict(X_test)))

            r2s.append(inner_r2s)
            best_alphas.append(inner_best_alphas)

        else:
            # --- Non-nested CV (less accurate) ---
            try:
                ridge_cv = RidgeCV(alphas=alphas)
                ridge_cv.fit(X_t, y)  # <--- This is what was missing
                scores = cross_val_score(
                    Ridge(alpha=ridge_cv.alpha_), X_t, y, cv=kf, scoring='r2', n_jobs=1)
                r2s.append(scores)
                best_alphas.append([ridge_cv.alpha_] * len(scores))
            except Exception as e:
                print('Error in ridge_cv.fit(X_t, y):', e)
                r2s.append([np.nan] * kf.get_n_splits())
                best_alphas.append([np.nan] * kf.get_n_splits())

    return np.array(r2s), np.array(best_alphas)


def nested_cv_r2(X, y, outer_cv, inner_alphas):
    r2_scores = []
    best_alphas = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ridge_cv = RidgeCV(alphas=inner_alphas, scoring='r2')
        ridge_cv.fit(X_train, y_train)

        best_alpha = ridge_cv.alpha_
        best_alphas.append(best_alpha)

        # Train on train set with best alpha, evaluate on test set
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        r2_scores.append(r2_score(y_test, model.predict(X_test)))

    return r2_scores, best_alphas


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


def standardize_trials(trials):
    """Standardize each trial (list of arrays) independently."""
    scaler = StandardScaler()
    return [scaler.fit_transform(trial) for trial in trials]
