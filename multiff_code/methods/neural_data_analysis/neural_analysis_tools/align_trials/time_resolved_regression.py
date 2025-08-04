# old

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


def time_resolved_regression_cv(
    concat_neural_trials, concat_behav_trials, cv_folds=5, n_jobs=-1
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
    alphas = np.logspace(-6, 6, 13)
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
            delayed(_regress_at_timepoint)(X_t, Y_t, n_behaviors, kf, alphas)
            for X_t, Y_t in XYs
        )

    # Build results DataFrame
    scores_by_time = np.array(results)
    time_resolved_cv_scores = pd.DataFrame(
        scores_by_time, columns=concat_behav_trials.columns)
    time_resolved_cv_scores['new_bin'] = used_new_bins
    time_resolved_cv_scores['trial_count'] = trial_counts

    return time_resolved_cv_scores


def standardize_trials(trials):
    """Standardize each trial (list of arrays) independently."""
    scaler = StandardScaler()
    return [scaler.fit_transform(trial) for trial in trials]

