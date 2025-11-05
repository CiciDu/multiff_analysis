#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible neural decoding with optional hyperparameter tuning and user-defined model parameters.

Supports:
    - Logistic Regression (linear)
    - Elastic Net Logistic Regression (sparse)
    - SVM (RBF)
    - Random Forest
    - MLP (neural net)

Also includes:
    - Sliding-window (time-resolved) decoding

Usage:
    from decoding import run_decoding, run_time_resolved_decoding
"""

from tqdm import tqdm  # make sure to install with: pip install tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.utils import shuffle
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional, Dict, Any

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# classifiers
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------
# 1) BUILD FEATURE & LABEL MATRICES
# ---------------------------------------------------------------------
def build_Xy(an, window, units: Optional[Sequence] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y) matrices for decoding."""
    seg = an.psth_data['segments']
    time = an.psth_data['psth']['time_axis']
    bw = an.config.bin_width

    # make sure that window[0] and window[1] are numerical, and window[0] < window[1]
    if not isinstance(window[0], (int, float)) or not isinstance(window[1], (int, float)):
        raise ValueError('window[0] and window[1] must be numerical')
    if window[0] >= window[1]:
        raise ValueError('window[0] must be less than window[1]')

    n_time = len(time)
    i0 = int(np.searchsorted(time, window[0], side='left'))
    i1 = int(np.searchsorted(time, window[1], side='right'))
    print('window[0]:', window[0], 'window[1]:', window[1])
    i0 = max(0, min(n_time - 1, i0))
    i1 = max(i0 + 1, min(n_time, i1))

    Xa = seg['event_a'][:, i0:i1, :].mean(axis=1) / bw
    Xb = seg['event_b'][:, i0:i1, :].mean(axis=1) / bw
    X = np.vstack([Xa, Xb])
    y = np.r_[np.ones(len(Xa), dtype=int), np.zeros(len(Xb), dtype=int)]
    unit_ids = np.array(an.clusters)

    if units is not None:
        if all(isinstance(u, (int, np.integer)) for u in units) and set(units).issubset(set(range(X.shape[1]))):
            cols = np.array(list(units), int)
        else:
            col_map = {lab: j for j, lab in enumerate(unit_ids)}
            cols = np.array([col_map[u] for u in units], int)
        X = X[:, cols]
        unit_ids = unit_ids[cols]

    # print('y.shape:', y.shape)
    print('X.shape:', X.shape)

    return X, y, unit_ids


# ---------------------------------------------------------------------
# 2) DEFINE DECODER LIBRARY
# ---------------------------------------------------------------------
def get_decoder(name: str,
                seed: int = 0,
                n_samples: int = 300,
                n_features: int = 50,
                model_kwargs: Optional[Dict[str, Any]] = None
                ) -> Tuple[Pipeline, Dict[str, Any]]:
    """Return classifier pipeline and parameter grid tailored to data size."""
    model_kwargs = model_kwargs or {}
    name = name.lower()

    small = n_samples <= 200
    medium = 200 < n_samples <= 500
    large = n_samples > 500

    # helper: smaller grids when data is small
    def choose(vals_small, vals_medium, vals_large):
        if small:
            return vals_small
        if medium:
            return vals_medium
        return vals_large

    if name == 'logreg':
        base = LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=seed,
            class_weight='balanced', **model_kwargs
        )
        # Keep C modest at small N
        Cs = choose([0.01, 0.1, 1], [0.03, 0.3, 1, 3], [0.03, 0.3, 1, 3, 10])
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': Cs}

    elif name == 'logreg_elasticnet':
        base = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 7),
            solver='saga', penalty='elasticnet',
            l1_ratios=choose([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], [
                             0.0, 0.1, 0.5, 0.9]),
            scoring='roc_auc', max_iter=1000,
            class_weight='balanced',
            random_state=seed, **model_kwargs
        )
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {}  # CV internal

    elif name == 'svm_linear':
        # Linear SVM is great when p is not tiny and N is small
        base = LinearSVC(
            random_state=seed, class_weight='balanced', **model_kwargs
        )
        Cs = choose([0.01, 0.1, 1], [0.03, 0.3, 1, 3], [0.03, 0.3, 1, 3, 10])
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': Cs}
        # NOTE: LinearSVC has no predict_proba; if you need probabilities, wrap in CalibratedClassifierCV externally.

    elif name == 'svm':  # RBF SVM
        base = SVC(probability=True, kernel='rbf',
                   random_state=seed, class_weight='balanced', **model_kwargs)
        # Keep gamma mostly 'scale'; only add a tiny perturbation at larger N
        gamma_grid = choose(['scale'], ['scale', 0.01],
                            ['scale', 0.03, 0.01])
        C_grid = choose([0.1, 1], [0.1, 1, 3], [0.1, 1, 3, 10])
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': C_grid, 'clf__gamma': gamma_grid}

    elif name == 'rf':
        base = RandomForestClassifier(random_state=seed, **model_kwargs)
        # Shallow trees for small N; grow a bit with more data
        n_estimators = choose([100], [100, 200], [200, 400])
        max_depth = choose([3, 5], [5, 10], [None, 10])
        min_split = choose([5, 10], [5, 10], [2, 5, 10])
        param_grid = {
            'clf__n_estimators': n_estimators,
            'clf__max_depth': max_depth,
            'clf__min_samples_split': min_split
        }
        clf = Pipeline([('clf', base)])  # no scaling needed

    elif name == 'mlp':
        base = MLPClassifier(
            max_iter=1000, random_state=seed,
            early_stopping=True, n_iter_no_change=10, **model_kwargs
        )
        # Keep networks tiny; increase slightly with data
        hls = choose([(16,), (32,)],
                     [(32,), (32, 16)],
                     [(32,), (64,), (64, 32)])
        alphas = choose([1e-3, 1e-2, 1e-1],
                        [1e-3, 1e-2, 1e-1],
                        [1e-4, 1e-3, 1e-2])
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__hidden_layer_sizes': hls,
                      'clf__alpha': alphas}

    else:
        raise ValueError(
            "Unknown decoder. Choose from: logreg, logreg_elasticnet, svm_linear, svm, rf, mlp")

    return clf, param_grid


# ---------------------------------------------------------------------
# 3) CROSS-VALIDATION DECODING
# ---------------------------------------------------------------------


def decode_auc_cv(X: np.ndarray, y: np.ndarray,
                  model_name: str = 'logreg',
                  k: int = 5, seed: int = 0, tune: bool = True,
                  search: str = 'grid', n_iter: int = 10,
                  model_kwargs: Optional[Dict[str, Any]] = None
                  ) -> Tuple[float, float, Dict[str, Any]]:
    """
    Cross-validated decoding with ROC-AUC scoring and optional hyperparameter tuning.
    Automatically handles models with or without predict_proba.
    """
    n_samples, n_features = X.shape
    clf, param_grid = get_decoder(
        model_name, seed, n_samples, n_features, model_kwargs)

    # Base classifier
    base_clf = clf.named_steps['clf'] if 'clf' in clf.named_steps else clf

    # Imputation only; scaling handled in get_decoder(). Build pipeline after
    # potential calibration so it always contains the correct classifier.

    # Handle models without predict_proba (e.g., LinearSVC)
    needs_calibration = not hasattr(base_clf, "predict_proba") and hasattr(
        base_clf, "decision_function")
    if needs_calibration:
        base_clf = CalibratedClassifierCV(base_clf, cv=3)

    steps = [('imputer', SimpleImputer(strategy='mean')), ('clf', base_clf)]
    pipeline = Pipeline(steps)

    # Use response_method for compatibility across sklearn versions
    scorer = make_scorer(roc_auc_score, response_method='predict_proba')
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)

    # Elastic-net logreg handles its own CV internally
    if model_name == 'logreg_elasticnet':
        model = pipeline.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        return float(auc), 0.0, model.get_params()['clf'].get_params()

    # Hyperparameter tuning
    if tune:
        if search == 'grid':
            searcher = GridSearchCV(
                pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1
            )
        elif search == 'random':
            searcher = RandomizedSearchCV(
                pipeline, param_grid, scoring=scorer, cv=cv,
                n_jobs=-1, n_iter=n_iter, random_state=seed
            )
        else:
            raise ValueError("search must be 'grid' or 'random'")

        searcher.fit(X, y)
        aucs = searcher.cv_results_['mean_test_score']
        aucs_sd = searcher.cv_results_['std_test_score']
        best_idx = np.argmax(aucs)
        return float(aucs[best_idx]), float(aucs_sd[best_idx]), searcher.best_params_

    # No tuning: just k-fold CV
    aucs = []
    for tr, te in cv.split(X, y):
        model = pipeline.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X[te])[:, 1]
        elif hasattr(model, "decision_function"):
            # decision_function may output signed scores
            p = model.decision_function(X[te])
        else:
            raise ValueError(
                f"Model {model_name} lacks predict_proba and decision_function.")
        aucs.append(roc_auc_score(y[te], p))

    aucs = np.asarray(aucs)
    return float(np.mean(aucs)), float(np.std(aucs)), model.get_params()['clf'].get_params()


# ---------------------------------------------------------------------
# 4) MAIN WRAPPER FUNCTION
# ---------------------------------------------------------------------
def run_decoding(an, window=(0.0, 0.2), units: Optional[Sequence] = None,
                 model_name='logreg', k=5, seed=0, tune=True, search='grid',
                 n_iter=10, model_kwargs: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Run full decoding pipeline for one window."""
    X, y, unit_ids = build_Xy(an, window, units=units)
    mean_auc, sd_auc, best_params = decode_auc_cv(
        X, y, model_name=model_name, k=k, seed=seed,
        tune=tune, search=search, n_iter=n_iter,
        model_kwargs=model_kwargs
    )
    return {
        'model': model_name,
        'tuned': tune,
        'window': window,
        'n_units': X.shape[1],
        'mean_auc': mean_auc,
        'sd_auc': sd_auc,
        'best_params': best_params
    }


# ---------------------------------------------------------------------
# 5) SLIDING-WINDOW DECODING
# ---------------------------------------------------------------------
def run_time_resolved_decoding(an, window_size=0.2, step=0.05, tmin=None, tmax=None,
                               model_name='logreg', k=5, seed=0, tune=False,
                               model_kwargs=None) -> pd.DataFrame:
    """
    Perform sliding-window decoding to estimate AUC over time.
    Returns a DataFrame with columns ['t_center', 'mean_auc', 'sd_auc'].
    """
    time = an.psth_data['psth']['time_axis']
    if tmin is None:
        tmin = time[0]
    if tmax is None:
        tmax = time[-1]

    # Define overlapping windows
    starts = np.arange(tmin, tmax - window_size, step)
    results = []
    for t0 in starts:
        window = (t0, t0 + window_size)
        res = run_decoding(an, window=window, model_name=model_name,
                           k=k, seed=seed, tune=tune,
                           model_kwargs=model_kwargs)
        res['t_center'] = np.mean(window)
        results.append(res)

    df = pd.DataFrame(results)
    return df[['t_center', 'mean_auc', 'sd_auc']]


# ---------------------------------------------------------------------
# 6) SIGNIFICANCE TESTING HELPERS
# ---------------------------------------------------------------------


def permutation_test_auc(
    X, y, model_name='logreg', k=5, n_perm=1000, seed=0,
    tune=False, model_kwargs=None, show_progress=True,
    plot=False
):
    """
    Permutation test for decoding significance with optional progress bar.

    Parameters
    ----------
    X, y : np.ndarray
        Features and labels.
    model_name : str
        Decoder model name ('logreg', 'svm', etc.).
    k : int
        Number of CV folds for decoding.
    n_perm : int
        Number of label shuffles.
    seed : int
        Random seed for reproducibility.
    tune : bool
        Whether to tune hyperparameters inside each permutation (slow).
    model_kwargs : dict or None
        Model parameters.
    show_progress : bool
        If True, display tqdm progress bar.

    Returns
    -------
    real_auc : float
        Observed decoding AUC.
    auc_null : np.ndarray
        Distribution of null AUCs from permutations.
    pval : float
        Empirical one-tailed p-value (AUC > chance).
    """
    rng = np.random.default_rng(seed)

    # 1. Compute observed AUC
    real_auc, _, _ = decode_auc_cv(
        X, y, model_name=model_name, k=k, seed=seed,
        tune=tune, model_kwargs=model_kwargs
    )

    # 2. Permutation null distribution
    auc_null = np.zeros(n_perm)
    iterator = range(n_perm)
    if show_progress:
        iterator = tqdm(iterator, desc=f'Permuting ({n_perm}x)', ncols=80)

    try:
        for i in iterator:
            y_perm = shuffle(y, random_state=rng.integers(1e6))
            auc_null[i], _, _ = decode_auc_cv(
                X, y_perm, model_name=model_name, k=k, seed=seed,
                tune=False, model_kwargs=model_kwargs
            )
    except KeyboardInterrupt:
        print(f'\nStopped early at permutation {i+1}/{n_perm}')

    # 3. Empirical one-tailed p-value
    pval = (np.sum(auc_null >= real_auc) + 1) / (len(auc_null) + 1)

    # 4. Plot null distribution
    if plot:
        plt.figure(figsize=(5, 3))
        plt.hist(auc_null, bins=30, alpha=0.7, label='null')
        plt.axvline(real_auc, color='r', lw=2,
                    label=f'observed={real_auc:.3f}')
        plt.title(f'Permutation test p={pval:.4f}')
        plt.xlabel('AUC')
        plt.ylabel('count')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return real_auc, auc_null, pval


def ttest_auc_folds(X, y, model_name='logreg', k=5, seed=0,
                    tune=False, model_kwargs=None):
    """
    One-sample t-test on CV fold AUCs vs 0.5 chance level.
    Returns mean AUC, std, t, and p-value (one-sided).
    """
    n_samples, n_features = X.shape
    clf, _ = get_decoder(model_name, seed, n_samples, n_features, model_kwargs)
    base_clf = clf.named_steps['clf'] if 'clf' in clf.named_steps else clf
    steps = [('imputer', SimpleImputer(strategy='mean')), ('clf', base_clf)]
    pipeline = Pipeline(steps)
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        model = pipeline.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    aucs = np.array(aucs)
    tstat, pval = stats.ttest_1samp(aucs, 0.5, alternative='greater')
    return np.mean(aucs), np.std(aucs), tstat, pval, aucs


# ---------------------------------------------------------------------
# 7) POPULATION-LEVEL AND TIME-RESOLVED TESTS
# ---------------------------------------------------------------------
def population_decoding_test(auc_per_neuron: np.ndarray, method='t', alpha=0.05):
    """
    Test whether a population of neurons decodes above chance (AUC>0.5).
    """
    auc_per_neuron = np.asarray(auc_per_neuron)
    if method == 't':
        t, p = stats.ttest_1samp(auc_per_neuron, 0.5, alternative='greater')
    elif method == 'wilcoxon':
        p = stats.wilcoxon(auc_per_neuron - 0.5, alternative='greater').pvalue
        t = np.nan
    else:
        raise ValueError("method must be 't' or 'wilcoxon'")
    sig = p < alpha
    return {'mean_auc': np.mean(auc_per_neuron), 't': t, 'p': p, 'sig': sig}


def time_resolved_significance(df_time, alpha=0.05, method='t', fdr=True):
    """
    Given a DataFrame from run_time_resolved_decoding (with fold AUCs per bin),
    compute per-bin p-values and FDR correction.
    """
    pvals = []
    for aucs in df_time['fold_aucs']:
        if method == 't':
            _, p = stats.ttest_1samp(aucs, 0.5, alternative='greater')
        elif method == 'wilcoxon':
            p = stats.wilcoxon(np.array(aucs) - 0.5,
                               alternative='greater').pvalue
        else:
            raise ValueError
        pvals.append(p)
    df_time['pval'] = pvals
    if fdr:
        df_time['sig_FDR'] = multipletests(
            df_time['pval'], alpha=alpha, method='fdr_bh')[0]
    else:
        df_time['sig_FDR'] = df_time['pval'] < alpha
    return df_time


def get_auc_per_neuron(an, window=(0, 0.2),
                       model_name='logreg', k=5, seed=0, tune=False,
                       model_kwargs=None):
    """
    Compute cross-validated AUC per individual neuron.
    Returns:
        auc_per_neuron : np.ndarray of shape (n_units,)
        unit_ids : list of neuron IDs
    """
    X, y, unit_ids = build_Xy(an, window)
    n_units = X.shape[1]
    aucs = np.zeros(n_units)

    for i in range(n_units):
        Xi = X[:, [i]]  # use only one column
        auc_mean, _, _ = decode_auc_cv(
            Xi, y, model_name=model_name, k=k, seed=seed,
            tune=tune, model_kwargs=model_kwargs
        )
        aucs[i] = auc_mean
        print(f'Neuron {i+1}/{n_units} (ID={unit_ids[i]}): AUC={auc_mean:.3f}')

    return aucs, unit_ids
