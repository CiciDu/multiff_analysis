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
from sklearn.base import clone
import itertools


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

    name = name.lower()

    default_model_kwargs = {
        "svm": {"C": 0.5, "gamma": 'scale'},
        "svm_linear": {"C": 0.03},
        "logreg": {"C": 0.03},
        "logreg_elasticnet": {"C": 0.1, "l1_ratio": 0.5},
        "rf": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
        "mlp": {"hidden_layer_sizes": (32,), "alpha": 1e-3},
    }

    model_kwargs_to_use = default_model_kwargs.get(name, {})
    model_kwargs_to_use.update(model_kwargs or {})

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
            class_weight='balanced', **model_kwargs_to_use
        )
        # Keep C modest at small N
        # Cs = choose([0.01, 0.1, 1], [0.03, 0.3, 1, 3], [0.03, 0.3, 1, 3, 10])
        Cs = [0.03]
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': Cs}

    elif name == 'logreg_elasticnet':
        # base = LogisticRegressionCV(
        #     Cs=np.logspace(-3, 0, num=4),
        #     solver='saga', penalty='elasticnet',
        #     l1_ratios=[0.01, 0.1, 0.5, 0.9],
        #     scoring='roc_auc', max_iter=5000, n_jobs=1,
        #     class_weight='balanced',
        #     random_state=seed, **model_kwargs_to_use,
        #     tol=1e-3
        # )
        # clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        # param_grid = {}  # CV internal

        base = LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            C=0.1,                  # fixed choice
            l1_ratio=0.5,             # fixed choice
            max_iter=5000,
            n_jobs=1,
            class_weight='balanced',
            random_state=seed,
            tol=1e-3,
            **model_kwargs_to_use
        )

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base)
        ])

        param_grid = {}

    elif name == 'svm_linear':
        # Linear SVM is great when p is not tiny and N is small
        base = LinearSVC(
            random_state=seed, class_weight='balanced', **model_kwargs_to_use
        )
        Cs = choose([0.01, 0.1, 1], [0.03, 0.3, 1, 3], [0.03, 0.3, 1, 3, 10])
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': Cs}
        # NOTE: LinearSVC has no predict_proba; if you need probabilities, wrap in CalibratedClassifierCV externally.

    elif name == 'svm':  # RBF SVM
        base = SVC(probability=True, kernel='rbf',
                   random_state=seed, class_weight='balanced', **model_kwargs_to_use)
        # Keep gamma mostly 'scale'; only add a tiny perturbation at larger N
        # gamma_grid = choose(['scale'], ['scale', 0.01],
        #                     ['scale', 0.03, 0.01])
        # C_grid = choose([0.1, 1], [0.1, 1, 3], [0.1, 1, 3, 10])
        gamma_grid = ['scale']
        C_grid = [0.1, 1]
        clf = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        param_grid = {'clf__C': C_grid, 'clf__gamma': gamma_grid}

    elif name == 'rf':
        base = RandomForestClassifier(random_state=seed, **model_kwargs_to_use)
        # Shallow trees for small N; grow a bit with more data
        # n_estimators = choose([100], [100, 200], [200, 400])
        # max_depth = choose([3, 5], [5, 10], [None, 10])
        # min_split = choose([5, 10], [5, 10], [2, 5, 10])
        n_estimators = [200, 400]
        max_depth = [None, 10]
        min_samples_leaf = [2, 5, 10]
        param_grid = {
            'clf__n_estimators': n_estimators,
            'clf__max_depth': max_depth,
            'clf__min_samples_leaf': min_samples_leaf
        }
        clf = Pipeline([('clf', base)])  # no scaling needed

    elif name == 'mlp':
        base = MLPClassifier(
            max_iter=1000, random_state=seed,
            early_stopping=True, n_iter_no_change=10, **model_kwargs_to_use
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
                  search: str = 'grid',
                  model_kwargs: Optional[Dict[str, Any]] = None,
                  param_grid_override: Optional[Dict[str, Any]] = None
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

    # If caller provides a param grid, use it instead of the default
    if param_grid_override is not None:
        param_grid = dict(param_grid_override)  # shallow copy

    # Normalize/retarget grid param keys to match pipeline structure
    def _normalize_grid_keys(grid: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(grid, dict) or len(grid) == 0:
            return grid
        try:
            clf_step = pipeline.named_steps.get('clf', None)
            calibrated = isinstance(clf_step, CalibratedClassifierCV)
        except Exception:
            calibrated = False
        norm_grid: Dict[str, Any] = {}
        for key, vals in grid.items():
            # If user already provided a nested path, keep as is, but fix common case
            if '__' in key:
                if calibrated and key.startswith('clf__') and not key.startswith('clf__base_estimator__'):
                    # Redirect to base_estimator inside CalibratedClassifierCV
                    key = 'clf__base_estimator__' + key[len('clf__'):]
                norm_grid[key] = vals
            else:
                # Bare param name -> prefix with appropriate step
                prefix = 'clf__base_estimator__' if calibrated else 'clf__'
                norm_grid[prefix + key] = vals
        return norm_grid

    param_grid = _normalize_grid_keys(param_grid)

    cv = StratifiedKFold(k, shuffle=True, random_state=seed)

    # Elastic-net logreg: if using LogisticRegressionCV, extract tuned params;
    # if using plain LogisticRegression, propagate provided params instead of placeholders.
    if model_name == 'logreg_elasticnet':
        # 1) Fit once on full data to extract parameters
        model_full = pipeline.fit(X, y)
        # 2) Compute out-of-sample AUC via k-fold CV
        aucs = []
        for tr, te in cv.split(X, y):
            model_cv = pipeline.fit(X[tr], y[tr])
            if hasattr(model_cv, "predict_proba"):
                p = model_cv.predict_proba(X[te])[:, 1]
            elif hasattr(model_cv, "decision_function"):
                p = model_cv.decision_function(X[te])
            else:
                raise ValueError(
                    "Model logreg_elasticnet lacks predict_proba and decision_function.")
            aucs.append(roc_auc_score(y[te], p))
        aucs = np.asarray(aucs)
        # Extract underlying classifier (unwrap calibration if present) from full-data fit
        try:
            clf_step = model_full.named_steps['clf']
        except Exception:
            clf_step = model_full.get_params().get('clf', None)
        if isinstance(clf_step, CalibratedClassifierCV):
            base_estimator = clf_step.base_estimator
        else:
            base_estimator = clf_step
        best_params = {}
        if base_estimator is not None:
            # Case 1: LogisticRegressionCV (has C_ and l1_ratio_)
            c_attr = getattr(base_estimator, 'C_', None)
            l1_attr = getattr(base_estimator, 'l1_ratio_', None)
            if c_attr is not None or l1_attr is not None:
                try:
                    c_vals = np.ravel(
                        c_attr) if c_attr is not None else np.array([1.0])
                    best_C = float(c_vals[0])
                except Exception:
                    best_C = 1.0
                try:
                    if l1_attr is None:
                        best_l1 = 0.5
                    else:
                        best_l1 = float(np.ravel(l1_attr)[0])
                except Exception:
                    best_l1 = 0.5
                best_params = {
                    'C': best_C,
                    'l1_ratio': best_l1,
                    'penalty': 'elasticnet',
                    'solver': getattr(base_estimator, 'solver', 'saga'),
                    'class_weight': getattr(base_estimator, 'class_weight', 'balanced'),
                    'max_iter': getattr(base_estimator, 'max_iter', 5000),
                    'random_state': getattr(base_estimator, 'random_state', 0),
                }
            else:
                # Case 2: Plain LogisticRegression -> propagate its own configured params
                try:
                    params = base_estimator.get_params()
                except Exception:
                    params = {}
                best_params = {
                    'C': float(params.get('C', 1.0)),
                    'l1_ratio': float(params.get('l1_ratio', 0.5)),
                    'penalty': params.get('penalty', 'elasticnet'),
                    'solver': params.get('solver', 'saga'),
                    'class_weight': params.get('class_weight', 'balanced'),
                    'max_iter': int(params.get('max_iter', 5000)),
                    'random_state': int(params.get('random_state', 0)),
                }
        return float(np.mean(aucs)), float(np.std(aucs)), best_params

    # Hyperparameter tuning
    if tune:
        if search == 'grid':
            searcher = GridSearchCV(
                pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=1)
        elif search == 'random':
            searcher = RandomizedSearchCV(pipeline, param_grid, scoring='roc_auc',
                                          cv=cv, n_jobs=1, n_iter=10, random_state=seed)
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
    # Extract params from final classifier; unwrap calibration if present
    try:
        clf_step = model.named_steps['clf']
    except Exception:
        clf_step = model.get_params().get('clf', None)
    if isinstance(clf_step, CalibratedClassifierCV):
        base_estimator = clf_step.base_estimator
    else:
        base_estimator = clf_step
    best_params = {}
    if base_estimator is not None:
        try:
            best_params = base_estimator.get_params()
        except Exception:
            best_params = {}
    return float(np.mean(aucs)), float(np.std(aucs)), best_params

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
                # print(f"[info] Applying fixed_params: {params_to_apply}")
                pipeline.set_params(**params_to_apply)
            except ValueError as e:
                # print(f"[warning] Could not apply fixed_params: {e}")
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
                 model_kwargs: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Run full decoding pipeline for one window."""
    X, y, unit_ids = build_Xy(an, window, units=units)
    mean_auc, sd_auc, best_params = decode_auc_cv(
        X, y, model_name=model_name, k=k, seed=seed,
        tune=tune, search=search,
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
    real_auc: Optional[float] = None,
    param_grid: Optional[Dict[str, Any]] = None,
    perm_search: Optional[str] = 'grid',
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
    fixed_params : dict or None
        If provided, use these frozen parameters for all permutations (fast).
    real_auc : float or None
        Observed AUC to reuse (skip recompute).
    param_grid : dict or None
        Optional hyperparameter grid to use during permutations (if perm_search is set).
        Keys may be bare (e.g., 'C') or pipeline-style; they will be normalized.
    perm_search : {'grid','random',None}
        If 'grid' or 'random', perform hyperparameter search on each permutation
        using param_grid (or the model's default grid if None). If None, fall back to
        fixed_params -> tuned_params -> untuned per original behavior.

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
            tune=tune, model_kwargs=model_kwargs,
            param_grid_override=param_grid
        )

    # 2. Prepare reusable pipeline template and CV splits to avoid rebuilds per permutation
    n_samples, n_features = X.shape
    base_clf, default_grid = get_decoder(
        model_name, seed, n_samples, n_features, model_kwargs
    )
    pipeline_template = make_calibrated_pipeline(base_clf)

    # Precompute CV splits once and reuse across permutations
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))

    # Identify preprocessing steps (everything before final 'clf') and cache transformed folds
    preproc_steps = []
    for name, step in pipeline_template.steps:
        if name == 'clf':
            break
        preproc_steps.append((name, step))
    preproc_pipeline = Pipeline(preproc_steps) if preproc_steps else None

    Xtr_list = []
    Xte_list = []
    for tr, te in splits:
        if preproc_pipeline is None:
            Xtr_list.append(X[tr])
            Xte_list.append(X[te])
        else:
            pp = clone(preproc_pipeline)
            Xtr_list.append(pp.fit_transform(X[tr]))
            Xte_list.append(pp.transform(X[te]))

    # Helpers to normalize grid keys and to apply fixed params on a cloned estimator/classifier
    def _normalize_grid_keys_for_pipeline(pipeline, grid: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(grid, dict) or len(grid) == 0:
            return grid
        try:
            clf_step = pipeline.named_steps.get('clf', None)
            calibrated = isinstance(clf_step, CalibratedClassifierCV)
        except Exception:
            calibrated = False
        norm_grid: Dict[str, Any] = {}
        for key, vals in grid.items():
            if '__' in key:
                if calibrated and key.startswith('clf__') and not key.startswith('clf__base_estimator__'):
                    key = 'clf__base_estimator__' + key[len('clf__'):]
                norm_grid[key] = vals
            else:
                prefix = 'clf__base_estimator__' if calibrated else 'clf__'
                norm_grid[prefix + key] = vals
        return norm_grid

    def _to_estimator_param_grid(grid: Dict[str, Any]) -> Dict[str, Any]:
        """Convert pipeline-style keys to bare estimator keys for the clf step."""
        if not isinstance(grid, dict) or len(grid) == 0:
            return grid
        est_grid: Dict[str, Any] = {}
        for key, vals in grid.items():
            if '__' in key:
                if key.startswith('clf__base_estimator__'):
                    est_key = key[len('clf__base_estimator__'):]
                elif key.startswith('clf__'):
                    est_key = key[len('clf__'):]
                else:
                    est_key = key.split('__')[-1]
            else:
                est_key = key
            est_grid[est_key] = vals
        return est_grid

    def _classifier_with_params(params: Optional[Dict[str, Any]]):
        """Clone only the classifier step from the template and set params on it."""
        est = clone(pipeline_template.named_steps['clf'])
        if not params:
            return est

        # Special handling: freeze LogisticRegression for elasticnet if requested
        params_local = dict(params)
        try:
            if (model_name == 'logreg_elasticnet'
                    and 'C' in params_local and 'l1_ratio' in params_local):
                frozen_lr = LogisticRegression(
                    penalty=str(params_local.get('penalty', 'elasticnet')),
                    solver=str(params_local.get('solver', 'saga')),
                    C=float(params_local['C']),
                    l1_ratio=float(params_local['l1_ratio']),
                    class_weight=params_local.get('class_weight', 'balanced'),
                    max_iter=int(params_local.get('max_iter', 5000)),
                    random_state=int(params_local.get('random_state', 0)),
                )
                if isinstance(est, CalibratedClassifierCV):
                    est.base_estimator = frozen_lr
                else:
                    est = frozen_lr
                # Remove keys consumed by the frozen estimator construction
                for kk in ('C', 'l1_ratio', 'penalty', 'solver', 'class_weight', 'max_iter', 'random_state'):
                    params_local.pop(kk, None)
        except Exception:
            # If anything goes wrong, fall back to generic param setting below
            pass

        if params_local:
            # Strip pipeline prefixes if present and apply to appropriate object
            bare_params = _to_estimator_param_grid(params_local)
            try:
                if isinstance(est, CalibratedClassifierCV):
                    est.base_estimator.set_params(**bare_params)
                else:
                    est.set_params(**bare_params)
            except ValueError:
                # Silently ignore params that do not match
                pass
        return est

    def _cv_mean_auc_pretransformed(y_labels: np.ndarray, params: Optional[Dict[str, Any]] = None) -> float:
        aucs = []
        for (tr, te), Xtr, Xte in zip(splits, Xtr_list, Xte_list):
            clf = _classifier_with_params(params)
            clf.fit(Xtr, y_labels[tr])
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(Xte)[:, 1]
            elif hasattr(clf, "decision_function"):
                p = clf.decision_function(Xte)
            else:
                raise ValueError(
                    f"Model {model_name} lacks probability-like output.")
            aucs.append(roc_auc_score(y_labels[te], p))
        return float(np.mean(aucs))

    def _iter_param_combinations(est_grid: Dict[str, Any], n_random: Optional[int] = None, rng_seed: int = 0):
        keys = list(est_grid.keys())
        values = [list(v) if isinstance(v, (list, tuple, np.ndarray)) else [
            v] for v in est_grid.values()]
        rng_local = np.random.default_rng(rng_seed)
        if n_random is None:
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))
        else:
            # Randomly sample with replacement
            for _ in range(int(n_random)):
                combo = [rng_local.choice(vals) for vals in values]
                yield dict(zip(keys, combo))

    # 3. Permutation null distribution
    auc_null = np.zeros(n_perm)
    iterator = range(n_perm)
    if show_progress:
        iterator = tqdm(iterator, desc=f'Permuting ({n_perm}x)', ncols=80)

    try:
        grid = param_grid if param_grid is not None else default_grid
        norm_grid = _normalize_grid_keys_for_pipeline(
            pipeline_template, grid)
        est_grid = _to_estimator_param_grid(norm_grid)
        for i in iterator:
            y_perm = shuffle(y, random_state=rng.integers(1e6))
            # Priority: use provided fixed_params; else reuse tuned_params if available;
            # else compute non-tuned CV per permutation.
            if perm_search in ('grid', 'random'):
                n_iter = None if perm_search == 'grid' else 10
                best_score = -np.inf
                for params_candidate in _iter_param_combinations(est_grid, n_random=n_iter, rng_seed=seed):
                    score = _cv_mean_auc_pretransformed(
                        y_perm, params_candidate)
                    if score > best_score:
                        best_score = score
                auc_null[i] = float(best_score)
            elif fixed_params is not None:
                auc_null[i] = _cv_mean_auc_pretransformed(y_perm, fixed_params)
            elif tune and isinstance(tuned_params, dict) and len(tuned_params) > 0:
                auc_null[i] = _cv_mean_auc_pretransformed(y_perm, tuned_params)
            else:
                auc_null[i] = _cv_mean_auc_pretransformed(y_perm, None)
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
    # Use unified calibrated pipeline to ensure probability-like output
    pipeline = make_calibrated_pipeline(clf)
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        model = pipeline.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X[te])[:, 1]
        elif hasattr(model, "decision_function"):
            p = model.decision_function(X[te])
        else:
            raise ValueError(
                f"Model {model_name} lacks a probability-like output.")
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
