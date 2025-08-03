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



def _regress_at_timepoint(X_t, Y_t, n_behaviors, kf, alphas):
    r2s = []
    best_alphas = []
    for b in range(n_behaviors):
        y = Y_t[:, b]
        r2_scores, b_alphas = nested_cv_r2(X_t, y, kf, alphas)
        r2s.append(r2_scores)
        best_alphas.append(b_alphas)
    return np.array(r2s), np.array(best_alphas)


def nested_cv_r2(X, y, outer_cv, inner_alphas):
    r2_scores = []
    best_alphas = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV: Find best alpha
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        ridge_cv = RidgeCV(alphas=inner_alphas, cv=inner_cv, scoring='r2')
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


def time_resolved_regression_cv(
    concat_neural_trials, concat_behav_trials, cv_folds=5, n_jobs=-1, alphas = np.logspace(-6, 6, 13)
):
    """
    Run time-resolved regression for variable-length trials.

    Returns:
        time_resolved_cv_scores: pd.DataFrame with columns: ['new_bin', 'trial_count', behavior columns...]
    """
    assert len(concat_neural_trials) == len(
        concat_behav_trials), "Mismatch in data dimensions"
    n_behaviors = concat_behav_trials.shape[1]

    # Ensure required columns
    for df in (concat_neural_trials, concat_behav_trials):
        if 'new_bin' not in df.columns:
            raise ValueError("'new_bin' column is required in both inputs.")

    new_bins = np.sort(concat_neural_trials['new_bin'].unique())
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Select only 'dim_*' columns and 'new_bin' for grouping
    neural_data_only = concat_neural_trials[
        [col for col in concat_neural_trials.columns if col.startswith(
            'dim_') or col == 'new_bin']
    ]

    # Group by time bin
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
            continue  # Skip bins with missing data

        X = X_df.values
        Y = Y_df.values

        if X.shape[0] < cv_folds:
            continue  # Not enough samples

        used_new_bins.append(new_bin)
        trial_counts.append(len(X))
        XYs.append((X, Y))

    if not XYs:
        raise RuntimeError("No time bins had sufficient data for regression.")

    # Parallel regression with progress
    with tqdm_joblib(tqdm(total=len(XYs), desc="Timepoints")):
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(# The `_regress_at_timepoint` function is performing time-resolved regression at a
            # specific time point. Here is a breakdown of what it does:
            _regress_at_timepoint)(X_t, Y_t, n_behaviors, kf, alphas)
            for X_t, Y_t in XYs
        )
        
    r2s, all_best_alphas = zip(*results)

    # Assuming r2s and alphas are shape (n_behaviors, n_folds) each timepoint
    r2s = np.stack(r2s)  # shape (n_timepoints, n_behaviors, n_folds)
    all_best_alphas = np.stack(all_best_alphas)

    n_bins = len(used_new_bins)
    r2s_flat = r2s.transpose(1, 0, 2).reshape(-1)  # behavior-major flattening
    alphas_flat = all_best_alphas.transpose(1, 0, 2).reshape(-1)

    time_resolved_cv_scores = pd.DataFrame({
        'behavior': np.tile(np.repeat(concat_behav_trials.columns, cv_folds), n_bins),
        'fold': np.tile(np.arange(cv_folds), n_behaviors * n_bins),
        'new_bin': np.repeat(used_new_bins, n_behaviors * cv_folds),
        'trial_count': np.repeat(trial_counts, n_behaviors * cv_folds),
        'r2': r2s_flat,
        'best_alpha': alphas_flat
    })
    return time_resolved_cv_scores


def standardize_trials(trials):
    """Standardize each trial (list of arrays) independently."""
    scaler = StandardScaler()
    return [scaler.fit_transform(trial) for trial in trials]



def streamline_getting_time_resolved_cv_scores(pn, 
                                               planning_data_by_point_exists_ok=True,
                                               latent_dimensionality=7,
                                               cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                               pre_event_window=0.25, post_event_window=0.75,
                                               cv_folds=5):
    # get data
    pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
    pn.planning_data_by_point, cols_to_drop = general_utils.drop_columns_with_many_nans(
        pn.planning_data_by_point)
    pn.prepare_seg_aligned_data(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last, time_limit_to_count_sighting=time_limit_to_count_sighting,
                                pre_event_window=pre_event_window, post_event_window=post_event_window)
    
    # time_resolved_cv_scores_gpfa
    pn.get_concat_data_for_regression(use_raw_spike_data_instead=True) 
    pn.retrieve_or_make_time_resolved_cv_scores_gpfa(latent_dimensionality=latent_dimensionality, cv_folds=cv_folds)
    
    # time_resolved_cv_scores
    pn.get_gpfa_traj(latent_dimensionality=latent_dimensionality, exists_ok=True)
    pn.get_concat_data_for_regression(use_raw_spike_data_instead=False) 
    pn.retrieve_or_make_time_resolved_cv_scores(cv_folds=cv_folds)