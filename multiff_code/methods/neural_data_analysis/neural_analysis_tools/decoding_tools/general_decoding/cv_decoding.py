# ============================================================
# Imports
# ============================================================

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Type
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor


# ============================================================
# Config
# ============================================================

@dataclass
class DecodingRunConfig:
    # CV
    cv_mode: str = 'group_kfold'
    buffer_samples: int = 0
    use_early_stopping: bool = True

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


# ============================================================
# Helpers
# ============================================================

def infer_decoding_type(y):
    y = y[np.isfinite(y)]
    n_unique = np.unique(y).size
    if n_unique <= 1:
        return 'skip'
    if n_unique == 2:
        return 'classification'
    return 'regression'


def filter_valid_rows(X, y, groups):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok], groups[ok]


def shuffle_y_groupwise(y, groups, rng):
    y_shuf = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_shuf[m] = rng.permutation(y_shuf[m])
    return y_shuf


def build_group_kfold_splits(X, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, groups=groups))


def make_feature_hash(feature, mode, n_splits, shuffle_y, context, config):
    params_hash = hashlib.sha1(
        json.dumps(
            dict(
                feature=feature,
                mode=mode,
                n_splits=n_splits,
                shuffle_y=shuffle_y,
                context=context,
                config=serialize_decoding_config(config),
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


def make_mode_hash(mode, n_splits, shuffle_y, context, config):
    """Create a hash based on mode and parameters (excluding feature)."""
    params_hash = hashlib.sha1(
        json.dumps(
            dict(
                mode=mode,
                n_splits=n_splits,
                shuffle_y=shuffle_y,
                context=context,
                config=serialize_decoding_config(config),
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


def get_model_name(mode, config):
    """Get the model name for a given mode and config."""
    if mode == 'regression':
        model_class = config.regression_model_class or CatBoostRegressor
    else:
        model_class = config.classification_model_class or LogisticRegression
    return model_class.__name__


def get_model_csv_path(out_dir, model_name):
    """Get the path for a model-level results file (by model name key)."""
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    return out_dir / f'{model_name}.csv'


def config_matches(row, n_splits, shuffle_y, context_label, config, model_name=None):
    """Check if a result row matches the given configuration."""
    # Check basic parameters
    if row.get('n_splits') != n_splits:
        return False
    if row.get('shuffle_y') != shuffle_y:
        return False
    if row.get('context') != context_label:
        return False
    
    # Check model_name if provided
    if model_name is not None:
        if row.get('model_name') != model_name:
            return False
    else:
        # Fallback: infer expected model_name from config
        mode = row.get('mode')
        expected_model_name = get_model_name(mode, config)
        if row.get('model_name') != expected_model_name:
            return False
    
    return True


# ============================================================
# Decoders
# ============================================================

def run_regression_cv(X, y, groups, splits, config, rng):
    y_pred = np.full_like(y, np.nan, float)

    model_class = config.regression_model_class or CatBoostRegressor
    model_kwargs = config.regression_model_kwargs or dict(verbose=False)

    for tr, te in splits:
        X_tr, y_tr = X[tr], y[tr]

        if np.unique(y_tr).size <= 1:
            y_pred[te] = y_tr[0]
            continue

        model = model_class(**model_kwargs)

        if config.use_early_stopping:
            uniq = np.unique(groups[tr])
            val_groups = rng.choice(
                uniq, size=max(1, int(0.2 * len(uniq))), replace=False
            )
            val_mask = np.isin(groups[tr], val_groups)

            model.fit(
                X_tr[~val_mask],
                y_tr[~val_mask],
                eval_set=(X_tr[val_mask], y_tr[val_mask]),
                use_best_model=True,
            )
        else:
            model.fit(X_tr, y_tr)

        y_pred[te] = model.predict(X[te])

    return dict(
        r2_cv=r2_score(y, y_pred),
        rmse_cv=np.sqrt(mean_squared_error(y, y_pred)),
        r_cv=np.corrcoef(y, y_pred)[0, 1],
    )


def run_classification_cv(X, y, splits, config):
    aucs, pr_aucs = [], []

    model_class = config.classification_model_class or LogisticRegression
    model_kwargs = config.classification_model_kwargs or {}

    for tr, te in splits:
        if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
            continue
                
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        clf = model_class(**model_kwargs)
        clf.fit(X_tr, y[tr].astype(int))

        p_te = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y[te], p_te))
        pr_aucs.append(average_precision_score(y[te], p_te))

    return dict(
        auc_mean=np.mean(aucs),
        auc_std=np.std(aucs),
        pr_mean=np.mean(pr_aucs),
        pr_std=np.std(pr_aucs),
    )


# ============================================================
# Main (slim orchestration)
# ============================================================
def serialize_decoding_config(config):
    return dict(
        cv_mode=config.cv_mode,
        buffer_samples=config.buffer_samples,
        use_early_stopping=config.use_early_stopping,

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
    shuffle_y,
    context_label,
    verbosity,
    model_name=None,
):
    """
    Load existing results matching current configuration.
    Returns dict keyed by (feature, model_name).
    """
    existing_lookup = {}

    if out_dir is None:
        return existing_lookup

    # If model_name is provided, only check that one file
    # Otherwise check both regression and classification
    if model_name is not None:
        csv_path = get_model_csv_path(out_dir, model_name)
        
        if verbosity >= 1:
            print(f'Checking for existing results in {csv_path}')
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                if config_matches(row, n_splits, shuffle_y, context_label, config, model_name):
                    key = (row['behav_feature'], row['model_name'])
                    existing_lookup[key] = row.to_dict()
            
            if verbosity >= 1:
                print(f'Loaded {len(existing_lookup)} matching rows')
        else:
            if verbosity >= 1:
                print(f'No results found for {csv_path}')
    else:
        # Fallback: scan for files by inferring model_name from config
        for mode in ['regression', 'classification']:
            inferred_model_name = get_model_name(mode, config)
            csv_path = get_model_csv_path(out_dir, inferred_model_name)

            if verbosity >= 1:
                print(f'Checking for existing results in {csv_path}')

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                if config_matches(row, n_splits, shuffle_y, context_label, config, None):
                    # Use model_name from the row, or fall back to inferred name
                    mname = row.get('model_name', inferred_model_name)
                    key = (row['behav_feature'], mname)
                    existing_lookup[key] = row.to_dict()

            if verbosity >= 1:
                print(f'Loaded {len(existing_lookup)} matching rows')

    return existing_lookup


def _compute_single_feature(
    feature,
    X,
    y_df,
    groups,
    config,
    n_splits,
    shuffle_y,
    context_label,
    rng,
    model_name=None,
):
    """
    Compute decoding for a single feature.
    Returns (row_dict, model_name, mode) or None.
    
    Parameters
    ----------
    model_name : str, optional
        The model name key (from model_specs). If None, uses the model class name.
    """
    y = y_df[feature].to_numpy().ravel()
    X_ok, y_ok, g_ok = filter_valid_rows(X, y, groups)

    mode = infer_decoding_type(y_ok)
    if mode == 'skip':
        return None

    # Use provided model_name or fall back to class name
    if model_name is None:
        model_name = get_model_name(mode, config)

    if shuffle_y:
        y_ok = shuffle_y_groupwise(y_ok, g_ok, rng)

    splits = build_group_kfold_splits(X_ok, g_ok, n_splits)

    if mode == 'regression':
        metrics = run_regression_cv(
            X_ok, y_ok, g_ok, splits, config, rng
        )
    else:
        metrics = run_classification_cv(
            X_ok, y_ok, splits, config
        )

    row = dict(
        behav_feature=feature,
        mode=mode,
        model_name=model_name,
        n_splits=n_splits,
        shuffle_y=shuffle_y,
        context=context_label,
        n_samples=len(y_ok),
        **metrics,
    )

    return row, model_name, mode


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
        'shuffle_y',
        'context',
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
            actual_dedup_cols = [col for col in dedup_cols if col in combined_df.columns]
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
):
    """
    Load the consolidated results CSV file.
    
    Parameters
    ----------
    out_dir : str or Path
        Directory containing the consolidated results file.
    filename : str, optional
        Name of the consolidated results CSV file. Default is 'all_models_results.csv'.
    
    Returns
    -------
    pd.DataFrame
        Consolidated results DataFrame, or empty DataFrame if file doesn't exist.
    """
    out_dir = Path(out_dir)
    csv_path = out_dir / filename
    
    if not csv_path.exists():
        print(f'Consolidated results file not found: {csv_path}')
        print('Run consolidate_results_across_models() first to create it.')
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows from {csv_path.name}')
    
    return df


def consolidate_results_across_models(
    out_dir,
    output_filename='all_models_results.csv',
    model_names=None,
    verbosity=1,
    save_output=False,
):
    """
    Consolidate all results from different model CSV files into one combined CSV.
    
    Parameters
    ----------
    out_dir : str or Path
        Directory containing individual model CSV files.
    output_filename : str, optional
        Name of the consolidated output CSV file. Default is 'all_models_results.csv'.
        Only used if save_output=True.
    model_names : list of str, optional
        List of specific model names to consolidate. If None, will scan for all CSV files
        in the directory and attempt to consolidate all of them.
    verbosity : int, optional
        Verbosity level. 0 = silent, 1 = basic info, 2 = detailed info.
    save_output : bool, optional
        Whether to save the consolidated results to a CSV file. Default is True.
        If False, only returns the DataFrame without saving.
    
    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with all results from all models.
    """
    out_dir = Path(out_dir)
    
    if not out_dir.exists():
        if verbosity > 0:
            print(f'Directory does not exist: {out_dir}')
        return pd.DataFrame()
    
    all_dfs = []
    
    if model_names is not None:
        # Load specific model files
        csv_paths = [get_model_csv_path(out_dir, model_name) for model_name in model_names]
    else:
        # Scan directory for all CSV files, excluding:
        # 1. The consolidated output file itself (to avoid circular inclusion)
        # 2. Any other non-model result files
        csv_paths = [
            p for p in out_dir.glob('*.csv') 
            if p.name != output_filename  # Exclude the consolidated file
        ]
    
    if verbosity > 0:
        print(f'Found {len(csv_paths)} CSV files to consolidate')
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            if verbosity > 1:
                print(f'Skipping non-existent file: {csv_path}')
            continue
        
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_dfs.append(df)
                if verbosity > 1:
                    print(f'Loaded {len(df)} rows from {csv_path.name}')
        except Exception as e:
            if verbosity > 0:
                print(f'Error loading {csv_path.name}: {e}')
    
    if not all_dfs:
        if verbosity > 0:
            print('No data found to consolidate')
        return pd.DataFrame()
    
    # Concatenate all dataframes
    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates based on key columns
    dedup_cols = [
        'behav_feature',
        'model_name',
        'n_splits',
        'shuffle_y',
        'context',
    ]
    actual_dedup_cols = [col for col in dedup_cols if col in consolidated_df.columns]
    
    if actual_dedup_cols:
        before_dedup = len(consolidated_df)
        consolidated_df = consolidated_df.drop_duplicates(
            subset=actual_dedup_cols,
            keep='first',
        )
        if verbosity > 1:
            print(f'Removed {before_dedup - len(consolidated_df)} duplicate rows')
    
    # Sort by model_name and behav_feature for cleaner output
    sort_cols = []
    if 'model_name' in consolidated_df.columns:
        sort_cols.append('model_name')
    if 'behav_feature' in consolidated_df.columns:
        sort_cols.append('behav_feature')
    
    if sort_cols:
        consolidated_df = consolidated_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Save consolidated results (optional)
    if save_output:
        output_path = out_dir / output_filename
        consolidated_df.to_csv(output_path, index=False)
        
        if verbosity > 0:
            print(f'Saved {len(consolidated_df)} total rows to {output_path.name}')
    else:
        if verbosity > 0:
            print(f'Consolidated {len(consolidated_df)} total rows (not saved to file)')
    
    return consolidated_df


# ============================================================
# Main Entry Point
# ============================================================

def run_cv_decoding(
    X=None,
    y_df=None,
    behav_features=None,
    groups=None,
    n_splits=5,
    config: Optional[DecodingRunConfig] = None,
    context_label=None,
    verbosity: int = 1,
    shuffle_y: bool = False,
    shuffle_seed: int = 0,
    save_dir: Optional[str | Path] = None,
    load_existing_only=False,
    exists_ok=True,
    model_name: Optional[str] = None,
):
    """
    Cross-validated decoding over multiple behavioral features.
    
    Parameters
    ----------
    load_existing_only : bool, optional
        If True, only loads existing results using consolidate_results_across_models.
        Does not compute any new results. Default is False.
    """

    if config is None:
        config = DecodingRunConfig()

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # If only loading existing results, use consolidate_results_across_models
    if load_existing_only:
        if out_dir is None:
            if verbosity > 0:
                print('Cannot load existing results: save_dir is None')
            return pd.DataFrame()
        
        if verbosity > 0:
            print('Loading existing results only...')
        
        # Consolidate all results across models
        all_results = consolidate_results_across_models(
            out_dir=out_dir,
            model_names=[model_name] if model_name is not None else None,
            verbosity=verbosity,
            save_output=False,  # Don't save, just return
        )

        return all_results

    X, groups, rng = _prepare_inputs(
        X, groups, shuffle_seed
    )
    
    if behav_features is None:
        behav_features = y_df.columns.tolist()

    existing_lookup = (
        _load_existing_results(
            out_dir,
            config,
            n_splits,
            shuffle_y,
            context_label,
            verbosity,
            model_name,
        )
        if exists_ok
        else {}
    )

    results = []
    results_by_model = {}

    for feature in tqdm(behav_features, desc='Decoding features'):

        # Determine mode first for lookup
        y = y_df[feature].to_numpy().ravel()
        X_ok, y_ok, _ = filter_valid_rows(X, y, groups)
        mode = infer_decoding_type(y_ok)
        if mode == 'skip':
            continue

        # Determine lookup model name
        # Use provided model_name or fall back to class name
        lookup_model_name = model_name if model_name is not None else get_model_name(mode, config)
        lookup_key = (feature, lookup_model_name)

        if lookup_key in existing_lookup:
            row = existing_lookup[lookup_key]
            results.append(row)
            _aggregate_results(results_by_model, row, lookup_model_name, mode)
            continue

        if load_existing_only:
            continue

        computed = _compute_single_feature(
            feature,
            X,
            y_df,
            groups,
            config,
            n_splits,
            shuffle_y,
            context_label,
            rng,
            model_name,
        )

        if computed is None:
            continue

        row, result_model_name, mode = computed
        results.append(row)
        _aggregate_results(results_by_model, row, result_model_name, mode)

    _save_results(out_dir, results_by_model, verbosity)

    return pd.DataFrame(results)