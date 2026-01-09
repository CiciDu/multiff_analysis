"""
Conditional decoding (regression):
Decode a continuous target variable conditioned on a categorical variable.

Supports multiple regression models:
  - ridge
  - lasso
  - elasticnet
  - kernel_ridge_rbf
  - svr_rbf

Implements:
  1) Global decoding:        y ~ X
  2) Conditioned decoding:   y ~ X | condition
  3) Bootstrapped delta:     conditioned − global
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from pathlib import Path
from datetime import datetime
import hashlib
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import (
    add_interactions, interaction_decoding
)


# ============================================================
# Model factory
# ============================================================

def make_regressor(model_type, random_state=0):
    """
    Returns a standardized regression pipeline.
    """

    if model_type == 'ridge':
        model = Ridge(alpha=1.0)

    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=5000)

    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)

    elif model_type == 'kernel_ridge_rbf':
        model = KernelRidge(
            kernel='rbf',
            alpha=1.0,
            gamma=None,  # uses 1 / n_features by default
        )

    elif model_type == 'svr_rbf':
        model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
        )

    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])


def regression_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
    }


# ============================================================
# Bootstrap: conditioned − global
# ============================================================

def bootstrap_conditioned_minus_global(
    *,
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type,
    n_splits=3,
    n_bootstraps=100,
    min_samples=50,
    random_state=0,
):
    # Guard: insufficient total samples
    if len(y_df) < 2 or n_splits < 2:
        return pd.DataFrame(columns=[
            'bootstrap_id', 'fold', 'condition_value',
            'global_score', 'cond_score', 'delta_score'
        ])

    rng = np.random.default_rng(random_state)

    X = x_df.values
    y = y_df[target_col].values
    c = y_df[condition_col].values

    # Adjust splits to available samples
    n_splits_eff = min(max(2, n_splits), len(y))
    if n_splits_eff < 2:
        return pd.DataFrame(columns=[
            'bootstrap_id', 'fold', 'condition_value',
            'global_score', 'cond_score', 'delta_score'
        ])

    cv = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    rows = []

    for b in range(n_bootstraps):
        idx = rng.choice(len(y), size=len(y), replace=True)

        Xb, yb, cb = X[idx], y[idx], c[idx]

        for fold, (tr, te) in enumerate(cv.split(Xb)):
            Xtr, Xte = Xb[tr], Xb[te]
            ytr, yte = yb[tr], yb[te]
            ctr, cte = cb[tr], cb[te]

            # ----------------------------
            # Global decoding
            # ----------------------------
            model = make_regressor(model_type, random_state)
            model.fit(Xtr, ytr)
            yhat_global = model.predict(Xte)
            global_metrics = regression_metrics(yte, yhat_global)

            # ----------------------------
            # Conditioned decoding
            # ----------------------------
            for cond_val in np.unique(cte):
                tr_mask = ctr == cond_val
                te_mask = cte == cond_val

                if (
                    np.sum(tr_mask) < min_samples or
                    np.sum(te_mask) < min_samples
                ):
                    continue

                model_c = make_regressor(model_type, random_state)
                model_c.fit(Xtr[tr_mask], ytr[tr_mask])
                yhat_cond = model_c.predict(Xte[te_mask])

                cond_metrics = regression_metrics(
                    yte[te_mask], yhat_cond
                )

                rows.append({
                    'bootstrap_id': b,
                    'fold': fold,
                    'condition_value': cond_val,

                    'global_score': global_metrics['r2'],
                    'cond_score': cond_metrics['r2'],
                    'delta_score': (
                        cond_metrics['r2'] - global_metrics['r2']
                    ),
                })

    return pd.DataFrame(rows)


# ============================================================
# Public API
# ============================================================

def run_conditional_decoding(
    *,
    x_df,
    y_df,
    target_col,
    condition_col,
    model_types=('ridge',),
    min_count=200,
    n_splits=5,
    n_bootstraps=200,
    random_state=0,
    save_path=None,
    load_if_exists=True,
    overwrite=False,
    verbosity: int = 1,
    processing_flags: dict | None = None,
    cache_tag: str | None = None,
):
    """
    Decode a continuous variable conditioned on a categorical variable.
    """

    # --------------------------------------------------------
    # 1. Prune rare condition values
    # --------------------------------------------------------
    y_pruned, x_pruned = add_interactions.prune_rare_states_two_dfs(
        df_behavior=y_df,
        df_neural=x_df,
        label_col=condition_col,
        min_count=min_count,
    )

    # --------------------------------------------------------
    # 2. Global decoding (no bootstrap)
    # --------------------------------------------------------
    X = x_pruned.values
    y = y_pruned[target_col].values

    # --------------------------------------------------------
    # 2b. Optional: cache load/save setup
    # --------------------------------------------------------
    def _log(msg, level=1):
        if verbosity >= level:
            print(msg)

    def _build_cache_paths(sp, params_hash):
        """
        Returns dict of paths:
          - meta: Path to metadata json
          - global, global_summary, cond_raw, cond_boot, cond_summary: CSV paths
        """
        sp = Path(sp)
        # Optional processing-aware subdirectory
        tag = _make_processing_tag(processing_flags, cache_tag)
        if sp.suffix.lower() != '.csv' and tag:
            sp = sp / tag
        if sp.suffix.lower() == '.csv':
            prefix = sp.with_suffix('')
        else:
            sp.mkdir(parents=True, exist_ok=True)
            prefix = sp / f'cond_{params_hash}'
        paths = {
            'meta': Path(str(prefix) + '_meta.json'),
            'global': Path(str(prefix) + '_global.csv'),
            'global_summary': Path(str(prefix) + '_global_summary.csv'),
            'cond_raw': Path(str(prefix) + '_cond_raw.csv'),
            'cond_boot': Path(str(prefix) + '_cond_boot.csv'),
            'cond_summary': Path(str(prefix) + '_cond_summary.csv'),
        }
        return paths

    def _save_outputs(paths, metadata, outs):
        try:
            outs['global_results'].to_csv(paths['global'], index=False)
            outs['global_summary'].to_csv(paths['global_summary'], index=False)
            outs['cond_delta_raw'].to_csv(paths['cond_raw'], index=False)
            outs['cond_delta_bootstrap'].to_csv(
                paths['cond_boot'], index=False)
            outs['cond_delta_summary'].to_csv(
                paths['cond_summary'], index=False)
            with open(paths['meta'], 'w') as f:
                json.dump(metadata, f, indent=2)
            _log(f"Saved results to {paths['global']}", level=1)
        except Exception as e:
            _log(f'Warning: failed saving cached results: {e}', level=1)

    def _load_outputs(paths):
        try:
            out = {
                'x_pruned': None,
                'y_pruned': None,
                'global_results': pd.read_csv(paths['global']),
                'global_summary': pd.read_csv(paths['global_summary']),
                'cond_delta_raw': pd.read_csv(paths['cond_raw']),
                'cond_delta_bootstrap': pd.read_csv(paths['cond_boot']),
                'cond_delta_summary': pd.read_csv(paths['cond_summary']),
            }
            return out
        except Exception as e:
            _log(f'Warning: failed loading cached results: {e}', level=1)
            return None

    save_paths = None
    metadata = None

    def _make_processing_tag(proc_flags, explicit_tag):
        if explicit_tag:
            return str(explicit_tag)
        if not proc_flags:
            return None
        # Compact, stable tag emphasizing known keys
        key_map = {
            'use_raw_spike_data_instead': 'raw',
            'apply_pca_on_raw_spike_data': 'pca',
            'use_lagged_raw_spike_data': 'lag',
        }
        parts = []
        for k in ['use_raw_spike_data_instead',
                  'apply_pca_on_raw_spike_data',
                  'use_lagged_raw_spike_data']:
            if k in proc_flags:
                v = proc_flags.get(k)
                try:
                    val = 1 if bool(v) else 0
                except Exception:
                    val = v
                parts.append(f"{key_map[k]}{val}")
        # Include any other flags deterministically
        for k in sorted(k for k in proc_flags.keys() if k not in key_map):
            v = proc_flags[k]
            if isinstance(v, (bool, int)):
                v = int(bool(v))
            parts.append(f"{k}={v}")
        return '_'.join(parts) if parts else None

    if save_path is not None:
        # Prepare metadata for stable hashing
        n_units_meta = int(X.shape[1]) if X.ndim >= 2 else 1
        n_samples_meta = int(X.shape[0])
        metadata = {
            'version': 1,
            'target_col': str(target_col),
            'condition_col': str(condition_col),
            'model_types': list(model_types),
            'min_count': int(min_count),
            'n_splits': int(n_splits),
            'n_bootstraps': int(n_bootstraps),
            'random_state': int(random_state),
            'n_units': n_units_meta,
            'n_samples': n_samples_meta,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_flags': dict(processing_flags) if processing_flags else None,
            'cache_tag': cache_tag,
        }
        hash_payload = {
            'target_col': metadata['target_col'],
            'condition_col': metadata['condition_col'],
            'model_types': tuple(sorted(metadata['model_types'])),
            'min_count': metadata['min_count'],
            'n_splits': metadata['n_splits'],
            'n_bootstraps': metadata['n_bootstraps'],
            'random_state': metadata['random_state'],
            'n_units': metadata['n_units'],
            'processing_flags': metadata['processing_flags'],
            'cache_tag': metadata['cache_tag'],
        }
        params_hash = hashlib.sha1(
            json.dumps(hash_payload, sort_keys=True,
                       default=str).encode('utf-8')
        ).hexdigest()[:10]
        metadata['params_hash'] = params_hash

        save_paths = _build_cache_paths(save_path, params_hash)

        # Try to load cached results (all files must exist and meta hash match)
        if load_if_exists and all(p.exists() for p in save_paths.values()):
            try:
                with open(save_paths['meta'], 'r') as f:
                    existing_meta = json.load(f)
                if existing_meta.get('params_hash') == params_hash:
                    _log(
                        f"Loaded cached results (hash={params_hash})", level=1)
                    loaded = _load_outputs(save_paths)
                    if loaded is not None:
                        return loaded
                else:
                    _log('Cached metadata mismatch. Recomputing.', level=1)
            except Exception as e:
                _log(f'Error reading cache metadata: {e}', level=1)
        else:
            _log('Computing new results...')

    # Handle insufficient samples early
    n_samples = len(y)
    if n_samples == 0:
        warnings.warn(
            f'run_conditional_decoding: no samples after pruning '
            f'({condition_col}, min_count={min_count}); returning empty results.'
        )
        empty_global = pd.DataFrame(columns=['model', 'fold', 'r2', 'mse'])
        empty_cond_raw = pd.DataFrame(columns=[
            'bootstrap_id', 'fold', 'condition_value',
            'global_score', 'cond_score', 'delta_score', 'model'
        ])
        collapsed, summary = summarize_bootstrap_deltas_reg(
            empty_cond_raw.copy()
        )
        outs = {
            'x_pruned': x_pruned,
            'y_pruned': y_pruned,
            'global_results': empty_global,
            'global_summary': pd.DataFrame(columns=['model', 'r2', 'mse']),
            'cond_delta_raw': empty_cond_raw,
            'cond_delta_bootstrap': collapsed,
            'cond_delta_summary': summary,
        }
        if save_paths is not None and (overwrite or not all(p.exists() for p in save_paths.values())):
            _save_outputs(save_paths, metadata, outs)
        return outs

    # Adjust CV splits to available samples
    n_splits_eff = min(max(2, n_splits), n_samples)
    if n_splits_eff < 2:
        warnings.warn(
            f'run_conditional_decoding: insufficient samples for CV '
            f'(n_samples={n_samples}); returning empty results.'
        )
        empty_global = pd.DataFrame(columns=['model', 'fold', 'r2', 'mse'])
        empty_cond_raw = pd.DataFrame(columns=[
            'bootstrap_id', 'fold', 'condition_value',
            'global_score', 'cond_score', 'delta_score', 'model'
        ])
        collapsed, summary = summarize_bootstrap_deltas_reg(
            empty_cond_raw.copy()
        )
        outs = {
            'x_pruned': x_pruned,
            'y_pruned': y_pruned,
            'global_results': empty_global,
            'global_summary': pd.DataFrame(columns=['model', 'r2', 'mse']),
            'cond_delta_raw': empty_cond_raw,
            'cond_delta_bootstrap': collapsed,
            'cond_delta_summary': summary,
        }
        if save_paths is not None and (overwrite or not all(p.exists() for p in save_paths.values())):
            _save_outputs(save_paths, metadata, outs)
        return outs

    cv = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    global_rows = []

    for model_type in model_types:
        for fold, (tr, te) in enumerate(cv.split(X)):
            model = make_regressor(model_type, random_state)
            model.fit(X[tr], y[tr])
            yhat = model.predict(X[te])

            metrics = regression_metrics(y[te], yhat)

            global_rows.append({
                'model': model_type,
                'fold': fold,
                **metrics,
            })

    # Ensure consistent columns even if no rows
    global_results = pd.DataFrame(
        global_rows, columns=['model', 'fold', 'r2', 'mse'])

    global_summary = (
        global_results
        .groupby('model', as_index=False)
        .mean()
    )

    # --------------------------------------------------------
    # 3. Bootstrapped conditioned − global
    # --------------------------------------------------------
    all_models = []

    for model_type in model_types:
        df = bootstrap_conditioned_minus_global(
            x_df=x_pruned,
            y_df=y_pruned,
            target_col=target_col,
            condition_col=condition_col,
            model_type=model_type,
            n_splits=n_splits_eff,
            n_bootstraps=n_bootstraps,
            min_samples=min_count,
            random_state=random_state,
        )
        df['model'] = model_type
        all_models.append(df)

    cond_raw = (
        pd.concat(all_models, ignore_index=True)
        if len(all_models) and sum(len(df) for df in all_models) > 0
        else pd.DataFrame(columns=[
            'bootstrap_id', 'fold', 'condition_value',
            'global_score', 'cond_score', 'delta_score', 'model'
        ])
    )

    # --------------------------------------------------------
    # 4. Collapse → CI
    # --------------------------------------------------------
    collapsed, summary = summarize_bootstrap_deltas_reg(cond_raw)

    # --------------------------------------------------------
    # 5. Attach condition sample sizes
    # --------------------------------------------------------
    counts = (
        y_pruned[condition_col]
        .value_counts()
        .rename_axis('condition_value')
        .reset_index(name='n_samples')
    )

    collapsed = collapsed.merge(
        counts, on='condition_value', how='left'
    )

    outs = {
        'x_pruned': x_pruned,
        'y_pruned': y_pruned,

        'global_results': global_results,
        'global_summary': global_summary,

        'cond_delta_raw': cond_raw,
        'cond_delta_bootstrap': collapsed,
        'cond_delta_summary': summary,
    }
    if save_paths is not None and (overwrite or not all(p.exists() for p in save_paths.values())):
        _save_outputs(save_paths, metadata, outs)
    return outs


def summarize_bootstrap_deltas_reg(df):
    """
    Collapse CV folds within each bootstrap, then compute CI across bootstraps
    for regression decoding.

    Expects columns:
      - bootstrap_id
      - condition_value
      - model
      - delta_score
      - global_score
      - cond_score
    """

    # Ensure required columns exist; handle empty/missing gracefully
    required_cols = [
        'bootstrap_id', 'condition_value', 'model',
        'delta_score', 'global_score', 'cond_score'
    ]
    df = df.copy()
    for col in required_cols:
        if col not in df.columns:
            if col in ('condition_value', 'model'):
                df[col] = pd.Series(dtype=object)
            elif col == 'bootstrap_id':
                df[col] = pd.Series(dtype='Int64')
            else:
                df[col] = pd.Series(dtype=float)

    if len(df) == 0:
        collapsed = pd.DataFrame(columns=[
            'bootstrap_id', 'condition_value', 'model',
            'delta_score', 'global_score', 'cond_score'
        ])
        summary = pd.DataFrame(columns=[
            'condition_value', 'model',
            'mean_delta_score', 'ci_low', 'ci_high'
        ])
        return collapsed, summary

    # ------------------------------------------------------------
    # Collapse folds within each bootstrap
    # ------------------------------------------------------------
    collapsed = (
        df
        .groupby(
            ['bootstrap_id', 'condition_value', 'model'],
            as_index=False,
        )
        .agg(
            delta_score=('delta_score', 'mean'),
            global_score=('global_score', 'mean'),
            cond_score=('cond_score', 'mean'),
        )
    )

    # ------------------------------------------------------------
    # CI across bootstraps
    # ------------------------------------------------------------
    summary = (
        collapsed
        .groupby(['condition_value', 'model'], as_index=False)
        .agg(
            mean_delta_score=('delta_score', 'mean'),
            ci_low=('delta_score',
                    lambda x: np.percentile(x, 2.5)),
            ci_high=('delta_score',
                     lambda x: np.percentile(x, 97.5)),
        )
    )

    return collapsed, summary




def run_band_conditioned_reg_decoding(
    df,
    concat_neural_trials,
    CONTINUOUS_INTERACTIONS,
    conditional_decoding_reg,
    conditional_decoding_plots,
    flags,
    max_pairs=100,
    save_path=None,
):
    """
    Run conditional decoding regression analyses and plotting for continuous interaction pairs.

    Parameters
    ----------
    pn : object
        Planning/neural data container with attributes:
        - planning_and_neural_folder_path
        - concat_neural_trials
    df : pd.DataFrame
        Behavioral or target dataframe.
    CONTINUOUS_INTERACTIONS : iterable of (str, str)
        Pairs of (target_col, condition_col).
    conditional_decoding_reg : module
        Module providing run_conditional_decoding.
    conditional_decoding_plots : module
        Module providing plotting functions.
    flags : dict
        Processing flags passed to decoding.
    max_pairs : int, optional
        Maximum number of pairs to process.
    """



    counter = 0
    for var_a, var_b in CONTINUOUS_INTERACTIONS:
        print(f'target_col: {var_a}, condition_col: {var_b}')

        out = conditional_decoding_reg.run_conditional_decoding(
            x_df=concat_neural_trials,
            y_df=df,
            target_col=var_a,
            condition_col=var_b,
            model_types=('ridge', 'elasticnet'),
            save_path=save_path,
            processing_flags=flags
        )

        fig = conditional_decoding_plots.plot_pairwise_interaction_analysis_reg(
            analysis_out=out,
            model_type='ridge',
            target_name=var_a,
            condition_name=var_b,
        )
        plt.show()

        fig = conditional_decoding_plots.plot_condition_scatterpanels_reg(
            analysis_out=out,
            target_name=var_a,
            condition_name=var_b,
            x_df=concat_neural_trials,
            y_df=df,
            model_type='ridge',
            n_splits=5,
        )
        plt.show()

        fig = conditional_decoding_plots.plot_global_scatter_reg(
            analysis_out=out,
            target_name=var_a,
            x_df=concat_neural_trials,
            y_df=df,
            model_type='ridge',
            n_splits=5,
        )
        plt.show()

        if counter >= max_pairs:
            break
        counter += 1