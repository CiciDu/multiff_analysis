# gpfa new

import os
import numpy as np
import pandas as pd
import logging
from typing import Optional

from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from elephant.gpfa import GPFA

from neural_data_analysis.neural_analysis_tools.gpfa_methods import fit_gpfa_utils

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
logging.basicConfig(level=logging.INFO)


def time_resolved_gpfa_regression_cv(
    concat_behav_trials, spiketrains, spiketrain_corr_segs, bin_bounds, bin_width_w_unit,
    cv_folds=5, n_jobs=-1, latent_dimensionality=7, alphas=np.logspace(-6, 6, 13)
):
    all_segments = list(concat_behav_trials['new_segment'].unique())
    splitter = KFold(n_splits=cv_folds, shuffle=True)

    all_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(all_segments)):
        train_segments = [all_segments[i] for i in train_idx]
        test_segments = [all_segments[i] for i in test_idx]

        logging.info(
            f"Fold {fold_idx + 1} | Train segments: {len(train_segments)} | Test segments: {len(test_segments)}"
        )

        gpfa_train, gpfa_test = fit_gpfa(
            train_segments, test_segments, spiketrains,
            spiketrain_corr_segs, bin_bounds, latent_dimensionality, bin_width_w_unit
        )

        behav_train = concat_behav_trials.loc[concat_behav_trials['new_segment'].isin(
            train_segments)].reset_index(drop=True)
        behav_test = concat_behav_trials.loc[concat_behav_trials['new_segment'].isin(
            test_segments)].reset_index(drop=True)

        # Check row alignment
        if not gpfa_train[['new_segment', 'new_bin']].equals(behav_train[['new_segment', 'new_bin']]):
            raise ValueError("GPFA train and behavior train are misaligned.")
        if not gpfa_test[['new_segment', 'new_bin']].equals(behav_test[['new_segment', 'new_bin']]):
            raise ValueError("GPFA test and behavior test are misaligned.")

        gpfa_train = gpfa_train.drop(
            columns=['new_segment', 'new_bin']).reset_index(drop=True)
        gpfa_test = gpfa_test.drop(
            columns=['new_segment', 'new_bin']).reset_index(drop=True)

        scores_df = run_time_resolved_regression_train_test(
            gpfa_train, behav_train,
            gpfa_test, behav_test,
            n_jobs=n_jobs, alphas=alphas
        )
        scores_df['fold'] = fold_idx
        all_scores.append(scores_df)

    return pd.concat(all_scores, ignore_index=True)


def fit_gpfa(train_segments, test_segments, spiketrains, spiketrain_corr_segs, bin_bounds, latent_dimensionality, bin_width_w_unit):
    """
    Fit GPFA model on training segments and transform both train and test data.
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
    neural_train, behav_train, neural_test, behav_test, alphas=np.logspace(-6, 6, 13), n_jobs=-1
):
    """
    Perform ridge regression at each time bin to predict behavioral variables from GPFA neural data.

    Parameters:
        - neural_train/test: DataFrames with latent neural trajectories
        - behav_train/test: DataFrames with behavioral variables
        - global_alphas, per_fold_alphas: Optional dictionaries of pre-selected alphas
        - n_jobs: parallel workers

    Returns:
        - DataFrame with R² scores and best alphas per behavior per time bin
    """
    behavior_cols = behav_train.select_dtypes(include=[float, int]).columns

    def fit_and_score_bin(new_bin):
        train_mask = behav_train['new_bin'] == new_bin
        test_mask = behav_test['new_bin'] == new_bin

        X_train = neural_train.loc[train_mask].select_dtypes(
            include=[float, int]).values
        X_test = neural_test.loc[test_mask].select_dtypes(
            include=[float, int]).values
        Y_train_all = behav_train.loc[train_mask, behavior_cols].values
        Y_test_all = behav_test.loc[test_mask, behavior_cols].values

        if len(X_train) < 2 or len(X_test) < 1:
            return None

        n_behaviors = Y_train_all.shape[1]
        r2s = np.full(n_behaviors, np.nan)
        all_best_alphas = np.full(n_behaviors, np.nan)

        for b in range(n_behaviors):
            Y_train = Y_train_all[:, b]
            Y_test = Y_test_all[:, b]

            ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=3)
            model = make_pipeline(StandardScaler(), ridge_cv)
            model.fit(X_train, Y_train)

            best_alpha = model.named_steps['ridgecv'].alpha_
            Y_pred = model.predict(X_test)

            r2s[b] = r2_score(Y_test, Y_pred)
            all_best_alphas[b] = best_alpha

        return pd.DataFrame({
            'feature': behavior_cols,
            'r2': r2s,
            'best_alpha': all_best_alphas,
            'new_bin': new_bin,
            'train_trial_count': len(X_train),
            'test_trial_count': len(X_test)
        })

    new_bins = behav_train['new_bin'].unique()
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score_bin)(nb) for nb in new_bins)
    results = [r for r in results if r is not None]

    return pd.concat(results, ignore_index=True)
