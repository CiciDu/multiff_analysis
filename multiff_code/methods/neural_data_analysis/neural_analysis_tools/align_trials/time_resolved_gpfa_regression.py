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
        model = make_pipeline(StandardScaler(),RidgeCV(alphas=alphas, scoring='r2', cv=3, n_jobs=n_jobs))
        
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

