# ============================================================
# Imports
# ============================================================

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Type
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
    tag = ''.join(c if c.isalnum() else '_' for c in feature)[:24]
    return out_dir / f'{tag}_{mode}_{params_hash}.csv'


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


def run_cv_decoding(
    X,
    y_df,
    behav_features,
    groups,
    n_splits=5,
    config: Optional[DecodingRunConfig] = None,
    context_label=None,
    verbosity: int = 1,
    shuffle_y: bool = False,
    shuffle_seed: int = 0,
    save_dir: Optional[str | Path] = None,
    load_existing_only = False,
):
    if config is None:
        config = DecodingRunConfig()

    X = np.asarray(X)
    groups = np.asarray(groups)
    rng = np.random.default_rng(shuffle_seed)

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    
    if behav_features is None:
        behav_features = y_df.columns.tolist()

    for feature in tqdm(behav_features, desc='Decoding features'):
        y = y_df[feature].to_numpy().ravel()
        X_ok, y_ok, g_ok = filter_valid_rows(X, y, groups)

        mode = infer_decoding_type(y_ok)
        if mode == 'skip':
            continue

        if shuffle_y:
            y_ok = shuffle_y_groupwise(y_ok, g_ok, rng)

        params_hash = make_feature_hash(
            feature, mode, n_splits, shuffle_y, context_label, config
        )
        csv_path = get_feature_csv_path(
            out_dir, feature, mode, params_hash
        )

        if csv_path is not None: 
            if csv_path.exists():
                results.append(pd.read_csv(csv_path).iloc[0].to_dict())
                if verbosity > 0:
                    print(f'Loaded results from {csv_path}')
                continue
            else:
                if verbosity > 0:
                    print(f'No results found for {csv_path}')
                
        if load_existing_only:
            continue

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
            n_samples=len(y_ok),
            context=context_label,
            **metrics,
        )

        results.append(row)

        if csv_path is not None:
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            print(f'Saved results to {csv_path}')


    return pd.DataFrame(results)
