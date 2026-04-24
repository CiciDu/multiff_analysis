# ============================================================
# Imports
# ============================================================

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostError
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# Config
# ============================================================

# CV modes that use blocks (time or group); per_block detrending is default for these
BLOCK_CV_MODES = ('blocked_time_buffered', 'blocked_time')

# Default when run_cv_decoding(..., cv_mode=None)
# Other options: 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold' (shuffled KFold)
DEFAULT_CV_MODE = 'blocked_time_buffered'

# For inferring cv_mode from pkl stems; longer names first so e.g. 'group_kfold' wins over 'kfold'
CV_MODE_CANDIDATES = (
    'blocked_time_buffered',
    'blocked_time',
    'group_kfold',
    'kfold',
)


@dataclass
class DecodingRunConfig:
    buffer_samples: int = 20
    use_early_stopping: bool = False

    # Detrending: when detrend_covariates is provided, regress them out of neural X before scaling
    detrend_degree: int = 1  # polynomial degree when covariate is 1D
    # When True and groups is provided, fit detrend separately within each block (group).
    # Default True for block-based CV (group_kfold, blocked_time*) to handle block-specific drift.
    detrend_per_block: bool = True

    # Regression
    regression_model_class: Optional[Type] = None
    regression_model_kwargs: dict = field(default_factory=dict)

    # Classification
    classification_model_class: Optional[Type] = None
    classification_model_kwargs: dict = field(
        default_factory=lambda: {
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 2000,
            'class_weight': 'balanced',
        }
    )


def run_cv_decoding(
    X=None,
    y_df=None,
    behav_features=None,
    groups=None,
    n_splits=5,
    config: Optional[DecodingRunConfig] = None,
    context_label=None,
    verbosity: int = 1,
    # can be 'none', 'foldwise', 'groupwise', 'timeshift_fold', 'timeshift_group'
    shuffle_mode: str = 'none',
    shuffle_seed: int = 0,
    save_dir: Optional[Union[str, Path]] = None,
    load_existing_only=False,
    exists_ok=True,
    model_name: Optional[str] = None,
    detrend_covariates=None,
    use_detrend_inside_cv=None,
    detrend_per_block=None,
    cv_mode=None,  # 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold'; None -> DEFAULT_CV_MODE
    force_reconsolidate=True,
):

    if config is None:
        config = DecodingRunConfig()

    # If load_existing_only is True, just load and return consolidated results
    if load_existing_only:
        if save_dir is None:
            raise ValueError(
                "save_dir must be provided when load_existing_only=True")
        return load_consolidated_results(
            save_dir,
            use_detrend_inside_cv=use_detrend_inside_cv,
            detrend_per_block=detrend_per_block,
            cv_mode=cv_mode,
            force_reconsolidate=force_reconsolidate,
            verbosity=verbosity,
        )

    if cv_mode is None:
        cv_mode = DEFAULT_CV_MODE
        print(f'cv_mode is None, using default: {cv_mode}')

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    X, groups, rng = _prepare_inputs(X, groups, shuffle_seed)

    if detrend_covariates is not None:
        if isinstance(detrend_covariates, pd.DataFrame):
            if len(detrend_covariates) != len(X):
                raise ValueError(
                    f"detrend_covariates rows ({len(detrend_covariates)}) must match X rows ({len(X)})"
                )
        else:
            arr = np.asarray(detrend_covariates)
            n = arr.shape[0] if arr.ndim > 1 else len(arr)
            if n != len(X):
                raise ValueError(
                    f"detrend_covariates length ({n}) must match X rows ({len(X)})"
                )

    if behav_features is None:
        behav_features = y_df.columns.tolist()

    do_detrend = detrend_covariates is not None
    existing_lookup = (
        _load_existing_results(
            out_dir,
            config,
            n_splits,
            shuffle_mode,
            context_label,
            verbosity,
            model_name,
            do_detrend=do_detrend,
            cv_mode=cv_mode,
        )
        if exists_ok
        else {}
    )

    results = []
    results_by_model = {}

    for feature in tqdm(behav_features, desc='Decoding features', mininterval=5):

        if verbosity >= 2:
            print(f'\nDecoding feature: {feature}')
            print(f'Existing results for this feature: {sum(1 for key in existing_lookup if key[0] == feature)}')

        y = y_df[feature].to_numpy().ravel()
        X_ok, y_ok, _, _ = filter_valid_rows(X, y, groups, detrend_covariates)
        mode = infer_decoding_type(y_ok)
        if mode == 'skip':
            continue

        use_per_block = (
            getattr(config, 'detrend_per_block', True) or cv_mode in BLOCK_CV_MODES
        ) if do_detrend else False
        lookup_model_name = (
            model_name if model_name is not None
            else get_model_name(mode, config, do_detrend=do_detrend, use_per_block=use_per_block)
        )
        lookup_key = (feature, lookup_model_name)

        if lookup_key in existing_lookup:
            row = existing_lookup[lookup_key].copy()
            row['model_name'] = lookup_model_name
            if row.get('detrend_degree') is None and '_detrend' in str(row.get('model_name', '')):
                row['detrend_degree'] = 1
            results.append(row)
            _aggregate_results(results_by_model, row, lookup_model_name, mode)
            continue

        computed = _compute_single_feature(
            feature,
            X,
            y_df,
            groups,
            config,
            n_splits,
            shuffle_mode,
            context_label,
            rng,
            model_name,
            detrend_covariates=detrend_covariates,
            cv_mode=cv_mode,
        )

        if computed is None:
            continue

        row, result_model_name, mode = computed
        results.append(row)
        _aggregate_results(results_by_model, row, result_model_name, mode)

    _save_results(out_dir, results_by_model, verbosity)

    return pd.DataFrame(results)


def infer_decoding_type(y):
    y = y[np.isfinite(y)]
    n_unique = np.unique(y).size
    if n_unique <= 1:
        return 'skip'
    if n_unique == 2:
        return 'classification'
    return 'regression'


def _encode_binary_classification_labels(y):
    """Map distinct label values to 0..K-1 (sorted uniques).

    ``y.astype(int)`` must not be used for sklearn: it truncates floats (e.g. 0.2 and 0.8
    both become 0), so folds can pass a two-value ``np.unique`` check yet fail in ``fit``.
    """
    y_arr = np.asarray(y).ravel()
    _, inv = np.unique(y_arr, return_inverse=True)
    return inv.astype(np.int64)


def _build_detrend_design_matrix(covariates, degree=1):
    """Build design matrix for detrending from covariates.

    Parameters
    ----------
    covariates : None, np.ndarray (1D or 2D), or pd.DataFrame
        - None: return None
        - 1D array (n_samples,) or (n_samples, 1): polynomial [t, t^2, ..., t^degree]
        - 2D array (n_samples, n_cov): use as-is (linear terms)
        - DataFrame (n_samples, n_cols): use .values as-is
    degree : int
        Polynomial degree for 1D/single-column input only.

    Returns
    -------
    np.ndarray or None
        Design matrix (n_samples, n_features) or None
    """
    if covariates is None:
        return None
    if isinstance(covariates, pd.DataFrame):
        T = np.asarray(covariates.values, dtype=float)
    else:
        T = np.asarray(covariates, dtype=float)

    if T.ndim == 1:
        T = T.reshape(-1, 1)
    if T.shape[1] == 1:
        T = np.hstack([T ** d for d in range(1, degree + 1)])
    return np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)


def filter_valid_rows(X, y, groups, detrend_covariates=None):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    dc_ok = None
    if detrend_covariates is not None:
        if isinstance(detrend_covariates, pd.DataFrame):
            dc_ok = np.asarray(detrend_covariates.iloc[ok].values, dtype=float)
        else:
            arr = np.asarray(detrend_covariates)
            dc_ok = arr[ok] if arr.ndim == 1 else arr[ok, :]

    if groups is not None:
        groups = groups[ok]
    return X[ok], y[ok], groups, dc_ok


def shuffle_y_groupwise(y, groups, rng):
    y_shuf = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_shuf[m] = rng.permutation(y_shuf[m])
    return y_shuf


def _shuffle_y_for_fold(y_tr, groups_tr, shuffle_mode, rng, buffer_samples=20):
    """Apply shuffle to training labels based on shuffle_mode. Returns shuffled y_tr."""
    min_shift = max(1, int(buffer_samples) + 1)
    if shuffle_mode == 'none':
        return y_tr
    if shuffle_mode == 'foldwise':
        return rng.permutation(y_tr)
    if shuffle_mode == 'groupwise' and groups_tr is not None:
        return shuffle_y_groupwise(y_tr, groups_tr, rng)
    if shuffle_mode == 'timeshift_fold':
        return _timeshift_1d(y_tr, rng, min_shift=min_shift)
    if shuffle_mode == 'timeshift_group' and groups_tr is not None:
        return shuffle_y_timeshift(y_tr, groups_tr, rng, min_shift=min_shift, within_groups=True)
    raise ValueError(f'Unknown shuffle_mode: {shuffle_mode}')


def _maybe_detrend_neural(
    X_tr,
    X_te,
    detrend_cov_tr,
    detrend_cov_te,
    degree=1,
    groups_tr=None,
    groups_te=None,
    per_block=False,
):
    """
    Regress detrend covariates out of neural X.

    If per_block=True and (groups_tr, groups_te) are provided:
        Fit detrend separately within each block (group). Each block's trend is
        fit on that block's data only and applied to that block. Use when drift
        varies by block (e.g., group_kfold, blocked CV with trial blocks).
    Else:
        Fit on all train data, apply to train and test (original behavior).
    Returns (X_tr, X_te) unchanged if either cov is None.
    """
    if detrend_cov_tr is None or detrend_cov_te is None:
        return X_tr.copy(), X_te.copy()
    from neural_data_analysis.neural_analysis_tools.get_neural_data.neural_drift import (
        detrend_features_cv_covariates,
    )
    T_tr = _build_detrend_design_matrix(detrend_cov_tr, degree=degree)
    T_te = _build_detrend_design_matrix(detrend_cov_te, degree=degree)

    if T_tr is None or T_te is None:
        return X_tr.copy(), X_te.copy()

    if per_block and groups_tr is not None and groups_te is not None:
        #print('detrended X_train and X_test by block')
        return _detrend_neural_per_block(X_tr, X_te, T_tr, T_te, groups_tr, groups_te,
                                        detrend_features_cv_covariates)
                
    #print('detrended X_train and X_test')
    return detrend_features_cv_covariates(X_tr, X_te, T_tr, T_te)


def _detrend_neural_per_block(X_tr, X_te, T_tr, T_te, groups_tr, groups_te, detrend_fn):
    """
    Detrend each block (group) separately. Fit on that block's data, apply to that block.
    """
    groups_tr = np.asarray(groups_tr)
    groups_te = np.asarray(groups_te)
    X_tr_out = X_tr.copy()
    X_te_out = X_te.copy()

    for g in np.unique(np.concatenate([groups_tr, groups_te])):
        m_tr = groups_tr == g
        m_te = groups_te == g
        if m_tr.any() and m_te.any():
            # Block appears in both train and test: fit on train, apply to both
            X_tr_out[m_tr], X_te_out[m_te] = detrend_fn(
                X_tr[m_tr], X_te[m_te], T_tr[m_tr], T_te[m_te]
            )
        elif m_tr.any():
            # Block only in train: fit and apply on train block
            dt_tr, _ = detrend_fn(X_tr[m_tr], X_tr[m_tr], T_tr[m_tr], T_tr[m_tr])
            X_tr_out[m_tr] = dt_tr
        elif m_te.any():
            # Block only in test: fit and apply on test block
            _, dt_te = detrend_fn(X_te[m_te], X_te[m_te], T_te[m_te], T_te[m_te])
            X_te_out[m_te] = dt_te
    return X_tr_out, X_te_out


def build_group_kfold_splits(X, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, groups=groups))


def _build_folds(
    n,
    *,
    n_splits=5,
    groups=None,
    cv_splitter=None,
    random_state=0,
    buffer_samples=20
):
    """
    Return a list of (train_idx, valid_idx) pairs.

    cv_splitter options:
      - 'blocked_time_buffered': contiguous time blocks with buffers on both sides
      - 'blocked_time': forward-chaining (past → future)
      - 'group_kfold': GroupKFold (requires ``groups``)
      - 'kfold': shuffled KFold (ignores ``groups`` for splitting)
      - else if ``groups`` is not None: GroupKFold
      - default: shuffled KFold (same as 'kfold')
    """
    idx = np.arange(n)

    # -------- BLOCKED TIME + BUFFER (recommended) --------
    if cv_splitter == 'blocked_time_buffered':
        # contiguous blocks
        edges = np.linspace(0, n, n_splits + 1, dtype=int)
        folds = []

        for k in range(n_splits):
            test_start, test_end = edges[k], edges[k + 1]

            test_idx = idx[test_start:test_end]

            # buffer region in time
            buf_start = max(0, test_start - buffer_samples)
            buf_end = min(n, test_end + buffer_samples)
            buffer_idx = idx[buf_start:buf_end]

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_mask[buffer_idx] = False

            train_idx = idx[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            folds.append((train_idx, test_idx))

        # print('cv_splitter = blocked_time_buffered: Split into contiguous blocks with buffer region')
        return folds

    # -------- FORWARD-CHAINING (causal CV) --------
    if cv_splitter == 'blocked_time':
        bps = np.linspace(0, n, n_splits + 1, dtype=int)
        folds = []
        for k in range(1, len(bps)):
            start, stop = bps[k-1], bps[k]
            train = idx[:start]
            valid = idx[start:stop]
            if len(train) and len(valid):
                folds.append((train, valid))
        # print('cv_splitter = blocked_time: Forward-chaining (past → future)')
        return folds

    # -------- GROUPED CV --------
    if cv_splitter == 'group_kfold':
        if groups is None:
            raise ValueError(
                "cv_splitter='group_kfold' requires non-None `groups` "
                "(length n_samples vector of trial / segment ids)."
            )
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(idx, groups=groups))

    # -------- SHUFFLED K-FOLD (explicit; ignores groups) --------
    if cv_splitter == 'kfold':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(idx))

    if groups is not None:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(idx, groups=groups))

    # -------- DEFAULT (NOT recommended for time series) --------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(idx))


def _infer_block_ids_from_splits(n, splits):
    """
    Infer a block id (fold id) for each sample index from CV splits.

    Each sample is assigned the index of the fold in which it appears
    as a test sample. This matches the notion that "each fold uses one block"
    when using time- or group-blocked CV.
    """
    block_ids = -np.ones(n, dtype=int)
    for k, (_, te) in enumerate(splits):
        block_ids[te] = k
    # Some CV schemes might leave a few points never in any test set;
    # assign them to block 0 so they still get detrended.
    if (block_ids < 0).any():
        block_ids[block_ids < 0] = 0
    return block_ids


def make_feature_hash(feature, mode, n_splits, shuffle_mode, context, config, cv_mode):
    params_hash = hashlib.sha1(
        json.dumps(
            dict(
                feature=feature,
                mode=mode,
                n_splits=n_splits,
                shuffle_mode=shuffle_mode,
                context=context,
                config=serialize_decoding_config(config, cv_mode),
            ),
            sort_keys=True,
            default=str,
        ).encode('utf-8')
    ).hexdigest()[:10]
    return params_hash


def get_feature_csv_path(out_dir, feature, mode, params_hash):
    if out_dir is None:
        return None
    out_dir = Path(out_dir)  # Ensure out_dir is a Path object
    tag = ''.join(c if c.isalnum() else '_' for c in feature)[:24]
    return out_dir / f'{tag}_{mode}_{params_hash}.csv'


def make_mode_hash(mode, n_splits, shuffle_mode, context, config, cv_mode):
    """Create a hash based on mode and parameters (excluding feature)."""
    params_hash = hashlib.sha1(
        json.dumps(
            dict(
                mode=mode,
                n_splits=n_splits,
                shuffle_mode=shuffle_mode,
                context=context,
                config=serialize_decoding_config(config, cv_mode),
            ),
            sort_keys=True,
            default=str,
        ).encode('utf-8')
    ).hexdigest()[:10]
    return params_hash


def get_mode_csv_path(out_dir, mode, params_hash):
    """Get the path for a mode-level results file."""
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    return out_dir / f'{mode}_{params_hash}.csv'


def _normalize_detrend_model_name(name):
    """If name ends with _detrend or _detrend_perblock (no _deg suffix), treat as degree 1."""
    if name is None or name == '':
        return name
    if '_deg' in name:
        return name
    if name.endswith('_detrend_perblock'):
        return name + '_deg1'
    if name.endswith('_detrend'):
        return name + '_deg1'
    return name


def get_model_name(mode, config, *, do_detrend=False, use_per_block=False):
    """Get the model name for a given mode and config.
    When do_detrend=True, appends _detrend[_perblock]_deg{N} to the base name.
    """
    if mode == 'regression':
        model_class = config.regression_model_class or CatBoostRegressor
    else:
        model_class = config.classification_model_class or LogisticRegression
    base = model_class.__name__
    if not do_detrend:
        return base
    detrend_degree = getattr(config, 'detrend_degree', 1)
    suffix = '_perblock' if use_per_block else ''
    return f"{base}_detrend{suffix}_deg{detrend_degree}"


def get_model_csv_path(out_dir, model_name):
    """Get the path for a model-level results file (by model name key)."""
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    return out_dir / f'{model_name}.csv'


def config_matches(row, n_splits, shuffle_mode, context_label, config, cv_mode, model_name=None, do_detrend=False):

    if row.get('n_splits') != n_splits:
        return False
    if row.get('shuffle_mode') != shuffle_mode:
        return False
    if row.get('context') != context_label:
        return False

    if row.get('cv_mode') != cv_mode:
        return False
    if row.get('buffer_samples') != config.buffer_samples:
        return False

    if model_name is not None:
        return row.get('model_name') == model_name

    mode = row.get('mode')
    use_per_block = (
        getattr(config, 'detrend_per_block', True) or cv_mode in BLOCK_CV_MODES
    ) if do_detrend else False
    expected_model_name = get_model_name(mode, config, do_detrend=do_detrend, use_per_block=use_per_block)
    row_model_name = _normalize_detrend_model_name(row.get('model_name'))
    return row_model_name == expected_model_name
# ============================================================
# Main (slim orchestration)
# ============================================================


def serialize_decoding_config(config, cv_mode):
    return dict(
        cv_mode=cv_mode,
        buffer_samples=config.buffer_samples,
        use_early_stopping=config.use_early_stopping,
        detrend_degree=getattr(config, 'detrend_degree', 1),
        detrend_per_block=getattr(config, 'detrend_per_block', True),

        regression_model_class=(
            config.regression_model_class.__name__
            if config.regression_model_class is not None else None
        ),
        regression_model_kwargs=config.regression_model_kwargs,

        classification_model_class=(
            config.classification_model_class.__name__
            if config.classification_model_class is not None else None
        ),
        classification_model_kwargs=config.classification_model_kwargs,
    )

# ============================================================
# CV Decoding Orchestration (Refactored)
# ============================================================


def _prepare_inputs(X, groups, shuffle_seed):
    X = np.asarray(X)
    groups = np.asarray(groups)
    rng = np.random.default_rng(shuffle_seed)

    return X, groups, rng


def _load_existing_results(
    out_dir,
    config,
    n_splits,
    shuffle_mode,
    context_label,
    verbosity,
    model_name=None,
    do_detrend=False,
    cv_mode=None,
):
    """
    Load existing results matching current configuration from pickle files.
    Scans for *.pkl under out_dir, unpickles each and extracts results_df.
    Returns dict keyed by (feature, model_name).
    """
    existing_lookup = {}

    if out_dir is None:
        return existing_lookup

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return existing_lookup

    pkl_paths = list(out_dir.rglob('*.pkl'))

    if verbosity >= 2:
        print(f'Checking for existing results in {len(pkl_paths)} pickle file(s) under {out_dir}')

    for p in pkl_paths:
        # Fast pre-filter: if cv_mode is encoded in the filename and doesn't match, skip
        stem = p.stem
        stem_cv_mode = next((c for c in CV_MODE_CANDIDATES if c in stem), None)
        if stem_cv_mode is not None and stem_cv_mode != cv_mode:
            if verbosity >= 2:
                print(f'Skipping {p.name}: cv_mode {stem_cv_mode!r} != {cv_mode!r}')
            continue

        try:
            with p.open('rb') as f:
                loaded = pickle.load(f)
        except Exception as e:
            if verbosity >= 2:
                print(f'Skipping {p}: failed to unpickle ({e})')
            continue

        if not (isinstance(loaded, dict) and 'results_df' in loaded):
            if verbosity >= 2:
                print(f'No results_df key in {p.name}, skipping')
            continue

        # Confirm cv_mode from _cv_config (authoritative) after loading
        pkl_cv_mode = loaded.get('_cv_config', {}).get('cv_mode')
        if pkl_cv_mode is not None and pkl_cv_mode != cv_mode:
            if verbosity >= 2:
                print(f'Skipping {p.name}: _cv_config cv_mode {pkl_cv_mode!r} != {cv_mode!r}')
            continue

        df = loaded['results_df']
        if not (isinstance(df, pd.DataFrame) and len(df) > 0):
            continue

        if 'cv_mode' not in df.columns:
            inferred_cv_mode = pkl_cv_mode or stem_cv_mode
            if inferred_cv_mode is not None:
                df = df.copy()
                df['cv_mode'] = inferred_cv_mode

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            if config_matches(row_dict, n_splits, shuffle_mode, context_label, config, cv_mode, model_name, do_detrend=do_detrend):
                key = (row_dict['behav_feature'], _normalize_detrend_model_name(row_dict.get('model_name', '')))
                existing_lookup[key] = row_dict

    if verbosity >= 2:
        print(f'Loaded {len(existing_lookup)} matching rows from pickles')

    return existing_lookup


def _aggregate_results(results_by_model, row, model_name, mode):
    results_by_model.setdefault(model_name, {
        'mode': mode,
        'rows': [],
    })['rows'].append(row)


def _save_results(
    out_dir,
    results_by_model,
    verbosity,
):
    if out_dir is None:
        return

    dedup_cols = [
        'behav_feature',
        'model_name',
        'n_splits',
        'shuffle_mode',
        'cv_mode',
        'buffer_samples',
        'context',
        'detrend_degree',
    ]

    for model_name, data in results_by_model.items():
        rows = data['rows']
        if not rows:
            continue

        csv_path = get_model_csv_path(out_dir, model_name)

        new_df = pd.DataFrame(rows)

        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            existing_len = len(existing_df)

            combined_df = pd.concat(
                [existing_df, new_df],
                ignore_index=True,
            )

            # Only use dedup columns that exist in the DataFrame
            actual_dedup_cols = [
                col for col in dedup_cols if col in combined_df.columns]
            combined_df = combined_df.drop_duplicates(
                subset=actual_dedup_cols,
                keep='first',
            )

            if len(combined_df) > existing_len:
                combined_df.to_csv(csv_path, index=False)
                if verbosity > 0:
                    print(
                        f'Saved {len(combined_df) - existing_len} '
                        f'new rows to {csv_path.name} '
                        f'({len(combined_df)} total)'
                    )
            elif verbosity > 1:
                print(f'No new rows to save for {csv_path.name}')

        else:
            new_df.to_csv(csv_path, index=False)
            if verbosity > 0:
                print(
                    f'Saved {len(new_df)} new results '
                    f'to {csv_path.name}'
                )


def load_consolidated_results(
    out_dir,
    filename='all_models_results.csv',
    use_detrend_inside_cv=None,
    detrend_per_block=None,
    cv_mode=None,
    force_reconsolidate=True,
    verbosity: int = 0,
):
    """
    Load the consolidated results CSV file.

    Parameters
    ----------
    out_dir : str or Path
        Directory containing the consolidated results file.
    filename : str, optional
        Name of the consolidated results CSV file. Default is 'all_models_results.csv'.
    use_detrend_inside_cv : bool, optional
        If True, keep only detrended results. If False, keep only non-detrended.
        If None, keep all. When loading from CSV, filters by 'use_detrend_inside_cv' column if present.
    detrend_per_block : bool, optional
        When use_detrend_inside_cv=True: if True, keep only per-block detrended; if False,
        keep only global detrended; if None, keep both. Ignored when use_detrend_inside_cv is False.
    cv_mode : str, optional
        If provided, keep only rows whose 'cv_mode' matches this value. If None, keep all.

    Returns
    -------
    pd.DataFrame
        Consolidated results DataFrame, or empty DataFrame if file doesn't exist.
    """
    out_dir = Path(out_dir)
    csv_path = out_dir / filename

    if not csv_path.exists() or force_reconsolidate:
        consolidated_df = consolidate_results_across_models(
            out_dir, filename,
            use_detrend_inside_cv=use_detrend_inside_cv,
            detrend_per_block=detrend_per_block,
            cv_mode=cv_mode,
            verbosity=verbosity,
        )
        df = consolidated_df
    else:
        df = pd.read_csv(csv_path)
        if 'cv_mode' not in df.columns:
            # CSV predates cv_mode column — re-consolidate from pkls to recover it
            df = consolidate_results_across_models(
                out_dir, filename,
                use_detrend_inside_cv=use_detrend_inside_cv,
                detrend_per_block=detrend_per_block,
                cv_mode=cv_mode,
                verbosity=verbosity,
            )
        if use_detrend_inside_cv is not None and 'use_detrend_inside_cv' in df.columns:
            df = df[df['use_detrend_inside_cv'] == use_detrend_inside_cv].reset_index(drop=True)
        if use_detrend_inside_cv and detrend_per_block is not None and 'detrend_per_block' in df.columns:
            df = df[df['detrend_per_block'] == detrend_per_block].reset_index(drop=True)

    if cv_mode is not None and 'cv_mode' in df.columns:
        df = df[df['cv_mode'] == cv_mode].reset_index(drop=True)

    return df


def consolidate_results_across_models(
    out_dir,
    output_filename='all_models_results.csv',
    model_names=None,
    verbosity=0,
    save_output=False,
    use_detrend_inside_cv=None,
    detrend_per_block=None,
    cv_mode=None,
):
    """
    Consolidate results by scanning for pickled model result files (*.pkl)
    under `out_dir` (recursively), unpickling each and extracting the
    `loaded['results_df']` object when present. Concatenates all found
    DataFrames, deduplicates and optionally saves to CSV.

    Parameters
    ----------
    out_dir : str or Path
        Directory containing pickled model result files.
    output_filename : str, optional
        Name of the consolidated output CSV file. Default is 'all_models_results.csv'.
    model_names : list of str, optional
        Unused for pickle-based consolidation (kept for backward compatibility).
    verbosity : int, optional
        Verbosity level.
    save_output : bool, optional
        Whether to save the consolidated CSV to `out_dir`.
    use_detrend_inside_cv : bool, optional
        If True, include only detrended runs. If False, include only non-detrended.
        If None, include all. When _cv_config lacks use_detrend_inside_cv, infers from
        filename (*_detrend*.pkl = detrended).
    detrend_per_block : bool, optional
        When use_detrend_inside_cv=True: if True, only *_detrend_perblock*.pkl; if False,
        only *_detrend*.pkl (global, not per block); if None, both. Ignored when
        use_detrend_inside_cv is False.
    cv_mode : str, optional
        If provided, only include pkls whose cv_mode matches this value (checked via
        filename stem first, then _cv_config after loading). If None, include all.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with all results.
    """
    out_dir = Path(out_dir)

    if not out_dir.exists():
        if verbosity > 0:
            print(f'Directory does not exist: {out_dir}')
        return pd.DataFrame()

    pkl_paths = list(out_dir.rglob('*.pkl'))

    if verbosity > 0:
        print(f'Found {len(pkl_paths)} pkl files under {out_dir}: {pkl_paths}')

    all_dfs = []

    for p in pkl_paths:
        # Skip loading when filter is set and filename indicates mismatch
        stem = p.stem
        if use_detrend_inside_cv is False and '_detrend' in stem:
            continue
        if use_detrend_inside_cv is True:
            if '_detrend' not in stem:
                continue
            if detrend_per_block is True and '_detrend_perblock' not in stem:
                continue
            if detrend_per_block is False and '_detrend_perblock' in stem:
                continue

        # Fast cv_mode pre-filter from filename (no I/O)
        if cv_mode is not None:
            stem_cv_mode = next((c for c in CV_MODE_CANDIDATES if c in stem), None)
            if stem_cv_mode is not None and stem_cv_mode != cv_mode:
                if verbosity > 1:
                    print(f'Skipping {p.name}: cv_mode {stem_cv_mode!r} != {cv_mode!r}')
                continue

        try:
            with p.open('rb') as f:
                loaded = pickle.load(f)
                if verbosity > 1:
                    print('Loaded pickle file:', p)
        except Exception as e:
            if verbosity > 1:
                print(f'Skipping {p}: failed to unpickle ({e})')
            continue

        # Expect a dict with key 'results_df'
        if isinstance(loaded, dict) and 'results_df' in loaded:
            df = loaded['results_df']
            try:
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    cv_config = loaded.get('_cv_config', {})

                    # Confirm cv_mode from _cv_config after loading
                    pkl_cv_mode = cv_config.get('cv_mode')
                    if pkl_cv_mode is None:
                        pkl_cv_mode = next((c for c in CV_MODE_CANDIDATES if c in stem), None)
                    if cv_mode is not None and pkl_cv_mode is not None and pkl_cv_mode != cv_mode:
                        if verbosity > 1:
                            print(f'Skipping {p.name}: _cv_config cv_mode {pkl_cv_mode!r} != {cv_mode!r}')
                        continue

                    # Infer use_detrend_inside_cv and detrend_per_block from _cv_config or filename
                    pkl_use_detrend = cv_config.get('use_detrend_inside_cv')
                    if pkl_use_detrend is None:
                        pkl_use_detrend = '_detrend' in stem
                    pkl_detrend_per_block = cv_config.get('detrend_per_block')
                    if pkl_detrend_per_block is None:
                        pkl_detrend_per_block = '_detrend_perblock' in stem if pkl_use_detrend else False
                    if use_detrend_inside_cv is not None and pkl_use_detrend != use_detrend_inside_cv:
                        continue
                    if use_detrend_inside_cv and detrend_per_block is not None and pkl_detrend_per_block != detrend_per_block:
                        continue
                    df = df.copy()
                    df['use_detrend_inside_cv'] = pkl_use_detrend
                    df['detrend_per_block'] = pkl_detrend_per_block
                    if 'detrend_degree' not in df.columns:
                        df['detrend_degree'] = cv_config.get('detrend_degree', 1 if pkl_use_detrend else None)
                    if 'cv_mode' not in df.columns and pkl_cv_mode is not None:
                        df['cv_mode'] = pkl_cv_mode
                    all_dfs.append(df)
                    if verbosity > 1:
                        print(f'Loaded {len(df)} rows from {p.name}')
            except Exception:
                if verbosity > 1:
                    print(f'Invalid results_df in {p.name}, skipping')
        else:
            if verbosity > 1:
                print(f'No results_df key in {p.name}, skipping')

    if not all_dfs:
        if verbosity > 0:
            print('No data found to consolidate from pickles')
        return pd.DataFrame()

    consolidated_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by model_name and behav_feature for cleaner output
    sort_cols = []
    if 'model_name' in consolidated_df.columns:
        sort_cols.append('model_name')
    if 'behav_feature' in consolidated_df.columns:
        sort_cols.append('behav_feature')

    if sort_cols:
        consolidated_df = consolidated_df.sort_values(
            sort_cols).reset_index(drop=True)

    # Save consolidated results (optional)
    if save_output:
        output_path = out_dir / output_filename
        consolidated_df.to_csv(output_path, index=False)

        if verbosity > 0:
            print(
                f'Saved {len(consolidated_df)} total rows to {output_path.name}')
    else:
        if verbosity > 0:
            print(
                f'Consolidated {len(consolidated_df)} total rows')

    return consolidated_df


def _timeshift_1d(y, rng, min_shift):
    """
    Circularly shift a 1D array by a random offset in [min_shift, n - min_shift].
    Ensures shift != 0 and avoids tiny shifts.
    """
    y = np.asarray(y)
    n = y.size
    if n < 2 * min_shift + 1:
        # Not enough points to do a meaningful shift; fall back to permutation
        print('Not enough points to do a meaningful circle time shift; fall back to permutation')
        return rng.permutation(y)

    shift = rng.integers(min_shift, n - min_shift + 1)
    #print('Doing a circle time shift with shift:', shift)
    return np.roll(y, shift)


def shuffle_y_timeshift(y, groups, rng, *, min_shift, within_groups=True):
    """
    Time-shift y using circular shift.

    If within_groups=True and groups is not None:
      - perform an independent circular shift *within each group*
        (prevents mixing across trials/blocks)
    Else:
      - circular shift across the entire provided y array
    """
    y = np.asarray(y)
    if (groups is None) or (not within_groups):
        return _timeshift_1d(y, rng, min_shift=min_shift)

    groups = np.asarray(groups)
    y_out = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_out[m] = _timeshift_1d(y_out[m], rng, min_shift=min_shift)
    return y_out


# ============================================================
# Main Entry Point
# ============================================================


def _compute_single_feature(
    feature,
    X,
    y_df,
    groups,
    config,
    n_splits,
    shuffle_mode,
    context_label,
    rng,
    model_name=None,
    detrend_covariates=None,
    cv_mode=None,
):

    y = y_df[feature].to_numpy().ravel()
    X_ok, y_ok, g_ok, detrend_cov_ok = filter_valid_rows(X, y, groups, detrend_covariates)

    mode = infer_decoding_type(y_ok)
    if mode == 'skip':
        return None

    do_detrend = detrend_covariates is not None
    detrend_degree = getattr(config, 'detrend_degree', 1) if do_detrend else None
    use_per_block = (
        getattr(config, 'detrend_per_block', True) or cv_mode in BLOCK_CV_MODES
    ) if do_detrend else False
    if model_name is None:
        model_name = get_model_name(mode, config, do_detrend=do_detrend, use_per_block=use_per_block)

    splits = _build_folds(
        len(X_ok),
        n_splits=n_splits,
        groups=g_ok,
        cv_splitter=cv_mode,
        buffer_samples=config.buffer_samples,
        random_state=0,
    )

    if mode == 'regression':
        metrics = run_regression_cv(
            X_ok,
            y_ok,
            g_ok,
            splits,
            config,
            rng,
            shuffle_mode=shuffle_mode,
            detrend_covariates=detrend_cov_ok,
            cv_mode=cv_mode,
        )
    else:
        metrics = run_classification_cv(
            X_ok,
            y_ok,
            splits,
            config,
            rng,
            groups=g_ok,
            shuffle_mode=shuffle_mode,
            detrend_covariates=detrend_cov_ok,
            cv_mode=cv_mode,
        )

    row = dict(
        behav_feature=feature,
        mode=mode,
        model_name=model_name,
        n_splits=n_splits,
        shuffle_mode=shuffle_mode,
        cv_mode=cv_mode,
        buffer_samples=config.buffer_samples,
        context=context_label,
        n_samples=len(y_ok),
        detrend_degree=detrend_degree,
        **metrics,
    )

    return row, model_name, mode


def run_regression_cv(
    X,
    y,
    groups,
    splits,
    config,
    rng,
    *,
    # can be 'none', 'foldwise', 'groupwise', 'timeshift_fold', 'timeshift_group'
    shuffle_mode='none',
    detrend_covariates=None,
    cv_mode=None,
):

    y_pred = np.full_like(y, np.nan, float)

    n_total_folds = len(splits)
    n_valid_folds = 0

    model_class = config.regression_model_class or CatBoostRegressor
    model_kwargs = config.regression_model_kwargs or dict(verbose=False)

    do_detrend = detrend_covariates is not None
    detrend_degree = getattr(config, 'detrend_degree', 1)
    buffer_samples = getattr(config, 'buffer_samples', 20)
    if do_detrend:
        T_full = _build_detrend_design_matrix(detrend_covariates, degree=detrend_degree)
        use_per_block = getattr(config, 'detrend_per_block', True) or (
            cv_mode in BLOCK_CV_MODES
        )
        block_ids = _infer_block_ids_from_splits(len(y), splits) if use_per_block else None

    for tr, te in splits:
        y_tr = _shuffle_y_for_fold(y[tr], groups[tr], shuffle_mode, rng, buffer_samples)

        X_tr = X[tr].copy()
        X_te = X[te].copy()
        if do_detrend:
            if use_per_block and block_ids is not None:
                groups_tr = block_ids[tr]
                groups_te = block_ids[te]
            else:
                groups_tr = None
                groups_te = None
            X_tr, X_te = _maybe_detrend_neural(
                X_tr,
                X_te,
                T_full[tr],
                T_full[te],
                degree=detrend_degree,
                groups_tr=groups_tr,
                groups_te=groups_te,
                per_block=use_per_block if do_detrend else False,
            )

        if np.unique(y_tr).size <= 1:
            # Not enough variance in this training fold to fit a model; fill with constant prediction
            y_pred[te] = y_tr[0]
            # treat as skipped fold (no model trained)
            continue

        # This fold will be used for evaluation (a model will be trained)
        n_valid_folds += 1

        # Scale features for regression (helps L-BFGS/Huber converge; matches classification path)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model = model_class(**model_kwargs)

        try:
            if config.use_early_stopping:
                uniq = np.unique(groups[tr])
                val_groups = rng.choice(
                    uniq,
                    size=max(1, int(0.2 * len(uniq))),
                    replace=False,
                )
                val_mask = np.isin(groups[tr], val_groups)

                # Check if train or validation subset has no variance
                if np.unique(y_tr[~val_mask]).size <= 1 or np.unique(y_tr[val_mask]).size <= 1:
                    # Fall back to fitting without early stopping
                    model.fit(X_tr, y_tr)
                else:
                    model.fit(
                        X_tr[~val_mask],
                        y_tr[~val_mask],
                        eval_set=(X_tr[val_mask], y_tr[val_mask]),
                        use_best_model=True,
                    )
            else:
                model.fit(X_tr, y_tr)
        except CatBoostError as e:
            msg = str(e)
            if 'All train targets are equal' in msg or 'All targets are equal' in msg:
                # Gracefully handle folds with no variance in targets despite checks
                y_pred[te] = np.mean(y_tr)
                continue
            raise

        y_pred[te] = model.predict(X_te)

    return dict(
        r2_cv=r2_score(y, y_pred),
        rmse_cv=np.sqrt(mean_squared_error(y, y_pred)),
        r_cv=np.corrcoef(y, y_pred)[0, 1],
        n_total_folds=n_total_folds,
        n_valid_folds=n_valid_folds,
        n_skipped_folds=n_total_folds - n_valid_folds,
    )


def run_classification_cv(
    X,
    y,
    splits,
    config,
    rng,
    *,
    groups=None,
    shuffle_mode='none',
    detrend_covariates=None,
    cv_mode=None,
):

    aucs, pr_aucs = [], []
    n_total_folds = len(splits)
    n_valid_folds = 0

    y_enc = _encode_binary_classification_labels(y)

    model_class = config.classification_model_class or LogisticRegression
    model_kwargs = config.classification_model_kwargs or {}

    do_detrend = detrend_covariates is not None
    detrend_degree = getattr(config, 'detrend_degree', 1)
    buffer_samples = getattr(config, 'buffer_samples', 20)
    if do_detrend:
        T_full = _build_detrend_design_matrix(detrend_covariates, degree=detrend_degree)
        use_per_block = getattr(config, 'detrend_per_block', True) or (
            cv_mode in BLOCK_CV_MODES
        )
        block_ids = _infer_block_ids_from_splits(len(y), splits) if use_per_block else None

    for tr, te in splits:
        y_tr = _shuffle_y_for_fold(y_enc[tr], groups[tr], shuffle_mode, rng, buffer_samples)

        if np.unique(y_tr).size < 2 or np.unique(y_enc[te]).size < 2:
            continue

        X_tr = X[tr].copy()
        X_te = X[te].copy()
        if do_detrend:
            if use_per_block and block_ids is not None:
                groups_tr = block_ids[tr]
                groups_te = block_ids[te]
            else:
                groups_tr = None
                groups_te = None
            X_tr, X_te = _maybe_detrend_neural(
                X_tr,
                X_te,
                T_full[tr],
                T_full[te],
                degree=detrend_degree,
                groups_tr=groups_tr,
                groups_te=groups_te,
                per_block=use_per_block if do_detrend else False,
            )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = model_class(**model_kwargs)
        try:
            clf.fit(X_tr, y_tr)
        except ValueError as e:
            # Rare folds (e.g. label quirks); encoding above should prevent sklearn's
            # "needs samples of at least 2 classes" in normal binary runs.
            if 'least 2 classes' in str(e):
                continue
            raise

        p_te = clf.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_enc[te], p_te))
        pr_aucs.append(average_precision_score(y_enc[te], p_te))
        n_valid_folds += 1

    # Handle pathological case
    if n_valid_folds == 0:
        return dict(
            auc_mean=np.nan,
            auc_std=np.nan,
            pr_mean=np.nan,
            pr_std=np.nan,
            n_total_folds=n_total_folds,
            n_valid_folds=0,
            n_skipped_folds=n_total_folds,
        )

    return dict(
        auc_mean=np.mean(aucs),
        auc_std=np.std(aucs),
        pr_mean=np.mean(pr_aucs),
        pr_std=np.std(pr_aucs),
        n_total_folds=n_total_folds,
        n_valid_folds=n_valid_folds,
        n_skipped_folds=n_total_folds - n_valid_folds,
    )
