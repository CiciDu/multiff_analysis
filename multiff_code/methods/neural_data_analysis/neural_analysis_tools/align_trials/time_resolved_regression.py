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


# ================================
# ================================
# fit gpfa on train set and test on test set


def time_resolved_gpfa_regression_cv(
    concat_behav_trials, spiketrains, spiketrain_corr_segs, bin_bounds, bin_width_w_unit,
    cv_folds=5, n_jobs=-1, latent_dimensionality=7
):
    all_segments = list(concat_behav_trials['new_segment'].unique())
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    all_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(all_segments)):
        train_segments = [all_segments[i] for i in train_idx]
        test_segments = [all_segments[i] for i in test_idx]

        print(
            f"Fold {fold_idx + 1} | Train segments: {len(train_segments)} | Test segments: {len(test_segments)}")

        # GPFA step
        gpfa_train, gpfa_test= fit_gpfa(train_segments, test_segments, spiketrains,
                        spiketrain_corr_segs, bin_bounds, latent_dimensionality, bin_width_w_unit)
        
        # Prepare behavior data subsets
        behav_train = concat_behav_trials[concat_behav_trials['new_segment'].isin(
            train_segments)].reset_index(drop=True)
        behav_test = concat_behav_trials[concat_behav_trials['new_segment'].isin(
            test_segments)].reset_index(drop=True)

        assert np.all(gpfa_train[['new_segment', 'new_bin']].values == behav_train[['new_segment', 'new_bin']].values)
        assert np.all(gpfa_test[['new_segment', 'new_bin']].values == behav_test[['new_segment', 'new_bin']].values)
 
        gpfa_train = gpfa_train.drop(columns=['new_segment', 'new_bin']).reset_index(drop=True)
        gpfa_test = gpfa_test.drop(columns=['new_segment', 'new_bin']).reset_index(drop=True)

        scores_df = run_time_resolved_regression_train_test(
            gpfa_train, behav_train,
            gpfa_test, behav_test,
            n_jobs=n_jobs
        )

        scores_df['fold'] = fold_idx
        all_scores.append(scores_df)

    all_scores_df = pd.concat(all_scores, ignore_index=True)

    return all_scores_df


def fit_gpfa(train_segments, test_segments, spiketrains, spiketrain_corr_segs, bin_bounds, latent_dimensionality, bin_width_w_unit):
    """
    Fit GPFA model on training segments, transform test segments,
    and prepare concatenated data for regression.
    """
    gpfa = GPFA(x_dim=latent_dimensionality,
                bin_size=bin_width_w_unit, verbose=False)
    train_trajectories = gpfa.fit_transform(
        [spiketrains[seg] for seg in train_segments])
    test_trajectories = gpfa.transform(
        [spiketrains[seg] for seg in test_segments])

    gpfa_train = fit_gpfa_utils._get_concat_gpfa_data(
        train_trajectories, spiketrain_corr_segs[train_segments], bin_bounds,
        new_segments_for_gpfa=train_segments
    )
    gpfa_test = fit_gpfa_utils._get_concat_gpfa_data(
        test_trajectories, spiketrain_corr_segs[test_segments], bin_bounds,
        new_segments_for_gpfa=test_segments
    )

    return gpfa_train, gpfa_test


def run_time_resolved_regression_train_test(
    neural_train, behav_train, neural_test, behav_test, n_jobs=-1
):
    """
    Train time-resolved Ridge regression models on train set and evaluate on test set,
    selecting the best alpha per time bin via cross-validation on the train set.

    Args:
        neural_train, behav_train: training data (dataframes)
        neural_test, behav_test: test data (dataframes)
        alphas: list or array of alpha values to try. If None, use default.
        n_jobs: parallel jobs

    Returns:
        DataFrame with R² scores per behavioral variable, new_bin, and selected alpha.
    """
    behavior_cols = behav_train.select_dtypes(include=[float, int]).columns

    def fit_and_score_bin(new_bin):
        try:
            train_mask = behav_train['new_bin'] == new_bin
            test_mask = behav_test['new_bin'] == new_bin
            X_train = neural_train[train_mask].select_dtypes(include=[float, int]).values
            X_test = neural_test[test_mask].select_dtypes(include=[float, int]).values
            Y_train = behav_train[train_mask]
            Y_test = behav_test[test_mask][behavior_cols].values
        except KeyError:
            return None

        if len(X_train) < 2 or len(X_test) < 1:
            return None


        # model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
        alphas = np.logspace(-6, 6, 13)
        model = make_pipeline(StandardScaler(),RidgeCV(alphas=alphas, scoring='r2', cv=3, n_jobs=n_jobs)))
        
        model.fit(X_train, Y_train)
        # best_alpha = model.alpha_

        Y_pred = model.predict(X_test)
        scores = [r2_score(Y_test[:, i], Y_pred[:, i])
                  for i in range(Y_test.shape[1])]

        row = dict(zip(behavior_cols, scores))
        row['new_bin'] = new_bin
        row['train_trial_count'] = len(X_train)
        row['test_trial_count'] = len(X_test)
        # row['best_alpha'] = best_alpha
        return row
    
    new_bins = behav_train['new_bin'].unique()

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score_bin)(nb) for nb in new_bins
    )
    results = [r for r in results if r is not None]

    return pd.DataFrame(results)

def streamline_getting_time_resolved_cv_scores(pn, 
                                               planning_data_by_point_exists_ok=True,
                                               latent_dimensionality=7,
                                               cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                               pre_event_window=0.25, post_event_window=0.75):
    # get data
    pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
    pn.planning_data_by_point, cols_to_drop = general_utils.drop_columns_with_many_nans(
        pn.planning_data_by_point)
    pn.prepare_seg_aligned_data(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last, time_limit_to_count_sighting=time_limit_to_count_sighting,
                                pre_event_window=pre_event_window, post_event_window=post_event_window)
    
    # time_resolved_cv_scores_gpfa
    pn.get_concat_data_for_regression(use_raw_spike_data_instead=True) 
    pn.retrieve_or_make_time_resolved_cv_scores_gpfa(latent_dimensionality=latent_dimensionality)
    
    # time_resolved_cv_scores
    pn.get_gpfa_traj(latent_dimensionality=latent_dimensionality, exists_ok=True)
    pn.get_concat_data_for_regression(use_raw_spike_data_instead=False) 
    pn.retrieve_or_make_time_resolved_cv_scores()