from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, Optional
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
            scoring='roc_auc', max_iter=5000, n_jobs=1,
            class_weight='balanced',
            random_state=seed, **model_kwargs,
            tol=1e-3
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

    # Build fully prepared pipeline (imputer + calibration if needed)
    pipeline = make_calibrated_pipeline(clf)

    # Use response_method for compatibility across sklearn versions
    scorer = make_scorer(roc_auc_score, response_method='predict_proba')
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)

    # Elastic-net logreg handles its own CV internally
    if model_name == 'logreg_elasticnet':
        model = pipeline.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        # Extract single best hyperparameters from LogisticRegressionCV
        try:
            lr_cv = model.named_steps['clf']
        except Exception:
            # Fallback: access via get_params (should not normally happen)
            lr_cv = model.get_params().get('clf', None)
        best_params = {}
        if lr_cv is not None:
            # Best C can be per-class; pick the first (one-vs-rest) selection
            try:
                c_attr = getattr(lr_cv, 'C_', None)
                if c_attr is not None:
                    c_vals = np.ravel(c_attr)
                    best_C = float(c_vals[0])
                else:
                    best_C = 1.0
            except Exception:
                best_C = 1.0
            # l1_ratio_ may be scalar or array depending on sklearn version
            try:
                lr_attr = getattr(lr_cv, 'l1_ratio_', None)
                if lr_attr is None:
                    # Fall back to default if attribute missing
                    best_l1 = 0.5
                else:
                    best_l1 = float(np.ravel(lr_attr)[0])
            except Exception:
                best_l1 = 0.5
            # Return a clean, frozen configuration for plain LogisticRegression
            best_params = {
                'C': best_C,
                'l1_ratio': best_l1,
                'penalty': 'elasticnet',
                'solver': 'saga',
                'class_weight': getattr(lr_cv, 'class_weight', 'balanced'),
                'max_iter': getattr(lr_cv, 'max_iter', 5000),
                'random_state': getattr(lr_cv, 'random_state', 0),
            }
        return float(auc), 0.0, best_params

    # Hyperparameter tuning
    if tune:
        if search == 'grid':
            searcher = GridSearchCV(
                pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=1)
        elif search == 'random':
            searcher = RandomizedSearchCV(pipeline, param_grid, scoring=scorer,
                                          cv=cv, n_jobs=1, n_iter=n_iter, random_state=seed)
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
            p = model.decision_function(X[te])
        else:
            raise ValueError(
                f"Model {model_name} lacks predict_proba and decision_function.")
        aucs.append(roc_auc_score(y[te], p))

    aucs = np.asarray(aucs)
    return float(np.mean(aucs)), float(np.std(aucs)), model.get_params()['clf'].get_params()

# ---------------------------------------------------------------------
# Helper: CV AUC with fixed (pre-tuned) params
# ---------------------------------------------------------------------


def _cv_auc_with_fixed_params(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "logreg",
    k: int = 5,
    seed: int = 0,
    model_kwargs: Optional[Dict[str, Any]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute k-fold CV AUC using a fixed set of pipeline params.
    Uses make_calibrated_pipeline() for consistent preprocessing/calibration.
    """
    n_samples, n_features = X.shape
    clf, _ = get_decoder(model_name, seed, n_samples, n_features, model_kwargs)

    # Unified pipeline creation
    pipeline = make_calibrated_pipeline(clf)

    # Apply fixed parameters if provided
    if fixed_params:
        # Special-case: freeze LogisticRegression (not CV) for elastic-net permutations
        if (model_name == 'logreg_elasticnet'
                and 'C' in fixed_params and 'l1_ratio' in fixed_params):
            try:
                frozen_lr = LogisticRegression(
                    penalty=str(fixed_params.get('penalty', 'elasticnet')),
                    solver=str(fixed_params.get('solver', 'saga')),
                    C=float(fixed_params['C']),
                    l1_ratio=float(fixed_params['l1_ratio']),
                    class_weight=fixed_params.get('class_weight', 'balanced'),
                    max_iter=int(fixed_params.get('max_iter', 5000)),
                    random_state=int(fixed_params.get('random_state', 0)),
                )
                if isinstance(pipeline.named_steps.get("clf"), CalibratedClassifierCV):
                    pipeline.named_steps["clf"].base_estimator = frozen_lr
                else:
                    pipeline.set_params(clf=frozen_lr)
                # Do not also try to set_params with CV grids
                fixed_params = {}
            except Exception as e:
                print(
                    f"[warning] Could not freeze LogisticRegression with fixed params: {e}")

        def _prefix_params(params: Dict[str, Any], step: str) -> Dict[str, Any]:
            """Prefix bare params with <step>__ for sklearn pipeline compatibility."""
            return {key if "__" in key else f"{step}__{key}": val for key, val in params.items()}

        clf_step = "clf__base_estimator" if isinstance(
            pipeline.named_steps["clf"], CalibratedClassifierCV) else "clf"

        if fixed_params:
            params_to_apply = _prefix_params(fixed_params, clf_step)
            try:
                #print(f"[info] Applying fixed_params: {params_to_apply}")
                pipeline.set_params(**params_to_apply)
            except ValueError as e:
                #print(f"[warning] Could not apply fixed_params: {e}")
                pass
    # Cross-validation loop
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        model = pipeline.fit(X[train_idx], y[train_idx])
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X[test_idx])[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred = model.decision_function(X[test_idx])
        else:
            raise ValueError(
                f"Model {model_name} lacks a probability-like output.")
        aucs.append(roc_auc_score(y[test_idx], y_pred))

    return float(np.mean(aucs))

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
# 6) SIGNIFICANCE TESTING HELPERS
# ---------------------------------------------------------------------


def permutation_test_auc(
    X, y, model_name='logreg', k=5, n_perm=1000, seed=0,
    tune=False, model_kwargs=None, show_progress=True,
    plot=False, fixed_params: Optional[Dict[str, Any]] = None,
    real_auc: Optional[float] = None
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
    tuned_params = {}
    if real_auc is not None:
        # Caller provided observed AUC already
        pass
    elif fixed_params is not None:
        # Use caller-provided fixed (tuned) params to score observed AUC
        real_auc = _cv_auc_with_fixed_params(
            X, y, model_name=model_name, k=k, seed=seed,
            model_kwargs=model_kwargs, fixed_params=fixed_params
        )
    else:
        # Fall back to internal CV (optionally tuned)
        real_auc, _, tuned_params = decode_auc_cv(
            X, y, model_name=model_name, k=k, seed=seed,
            tune=tune, model_kwargs=model_kwargs
        )

    # 2. Announce parameter strategy for permutations
    if fixed_params is not None:
        print(
            f"[permutation_test_auc] Using pre-determined params from caller for permutations: {fixed_params}")
    elif isinstance(tuned_params, dict) and len(tuned_params) > 0:
        print("[permutation_test_auc] Using tuned best params from observed-label CV for permutations.")
    else:
        print(
            "[permutation_test_auc] No pre-computed params; using untuned CV per permutation.")

    # 3. Permutation null distribution
    auc_null = np.zeros(n_perm)
    iterator = range(n_perm)
    if show_progress:
        iterator = tqdm(iterator, desc=f'Permuting ({n_perm}x)', ncols=80)

    try:
        for i in iterator:
            y_perm = shuffle(y, random_state=rng.integers(1e6))
            # Priority: use provided fixed_params; else reuse tuned_params if available;
            # else compute non-tuned CV per permutation.
            if fixed_params is not None:
                auc_null[i] = _cv_auc_with_fixed_params(
                    X, y_perm, model_name=model_name, k=k, seed=seed,
                    model_kwargs=model_kwargs, fixed_params=fixed_params
                )
            elif tune and isinstance(tuned_params, dict) and len(tuned_params) > 0:
                auc_null[i] = _cv_auc_with_fixed_params(
                    X, y_perm, model_name=model_name, k=k, seed=seed,
                    model_kwargs=model_kwargs, fixed_params=tuned_params
                )
            else:
                auc_null[i], _, _ = decode_auc_cv(
                    X, y_perm, model_name=model_name, k=k, seed=seed,
                    tune=False, model_kwargs=model_kwargs
                )
    except KeyboardInterrupt:
        print(f'\nStopped early at permutation {i+1}/{n_perm}')

    # 4. Empirical one-tailed p-value
    pval = (np.sum(auc_null >= real_auc) + 1) / (len(auc_null) + 1)

    # 5. Plot null distribution
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
    # Preserve scaler from get_decoder by prepending imputer
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))] +
                        (clf.steps if isinstance(clf, Pipeline) else [('clf', clf)]))
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


def make_calibrated_pipeline(clf: Pipeline) -> Pipeline:
    """
    Wrap a classifier (or pipeline) with calibration and imputation if needed.

    - Preserves any scaler or preprocessing in the given pipeline.
    - Adds mean imputation at the start.
    - Wraps the final estimator in CalibratedClassifierCV if it lacks predict_proba.

    Returns
    -------
    pipeline : sklearn.Pipeline
        A fully prepared and calibrated pipeline.
    """
    # Extract base estimator
    if isinstance(clf, Pipeline):
        base_clf = clf.named_steps.get('clf', clf)
    else:
        base_clf = clf

    # Avoid double calibration
    if isinstance(base_clf, CalibratedClassifierCV):
        needs_calibration = False
    else:
        needs_calibration = not hasattr(base_clf, 'predict_proba') and hasattr(
            base_clf, 'decision_function')

    # Wrap with calibration if necessary
    if needs_calibration:
        if isinstance(clf, Pipeline) and 'clf' in clf.named_steps:
            new_steps = []
            for name, step in clf.steps:
                if name == 'clf':
                    step = CalibratedClassifierCV(step, cv=3)
                new_steps.append((name, step))
            clf = Pipeline(new_steps)
        else:
            base_clf = CalibratedClassifierCV(base_clf, cv=3)
            clf = Pipeline([('clf', base_clf)])

    # Always prepend imputer
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))] +
                        (clf.steps if isinstance(clf, Pipeline) else [('clf', clf)]))
    return pipeline
