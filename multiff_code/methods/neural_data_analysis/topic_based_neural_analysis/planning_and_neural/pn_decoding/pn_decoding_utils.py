import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import json
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from itertools import product
from data_wrangling import general_utils


def add_interaction_terms_and_features(concat_behav_trials):
    cols_added = []
    for var_a, var_b in product(
            ['cur_vis', 'nxt_vis', 'nxt_in_memory'],
            ['cur_ff_distance', 'nxt_ff_distance']):
        col_name = f'{var_a}*{var_b}'
        concat_behav_trials[col_name] = (
            concat_behav_trials[var_a] * concat_behav_trials[var_b]
        )
        cols_added.append(col_name)

    return concat_behav_trials, cols_added


def prep_behav(df,
               cont_cols=('cur_ff_distance', 'nxt_ff_distance',
                          'time_since_last_capture', 'speed', 'accel'),
               cat_vars=('cur_vis', 'nxt_vis', 'nxt_in_memory', 'log1p_num_ff_visible')):
    # keep only requested features (that exist), copy to avoid side effects
    out = df.copy()
    added_cols = []

    # add log1p features (clip negatives to 0 to keep log1p valid)
    for c in cont_cols:
        if c in out.columns:
            out[f'log1p_{c}'] = np.log1p(pd.to_numeric(
                out[c], errors='coerce').clip(lower=0))
            added_cols.append(f'log1p_{c}')

    # binarize categorical indicators ( > 0 → 1; else 0 )
    for v in cat_vars:
        if v in out.columns:
            x = pd.to_numeric(out[v], errors='coerce')
            out[v] = (x.fillna(0) > 0).astype('int8')
            added_cols.append(v)
    return out, added_cols


def eval_decoder(y_true, y_pred, title='Decoder fit', sample_ts=None, n_show=500):
    """
    Back-compatible evaluation (no need for sklearn>=0.22).
    - y_true, y_pred: 1D array-like
    - sample_ts: optional 1D times for the overlay plot
    - n_show: how many samples to show in the overlay
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # drop NaNs in a paired way
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    if sample_ts is not None:
        sample_ts = np.asarray(sample_ts).ravel()[ok]

    # metrics (old sklearn safe)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Pearson R (scale-invariant)
    if y_true.std() == 0 or y_pred.std() == 0:
        R = np.nan
    else:
        R = np.corrcoef(y_true, y_pred)[0, 1]

    print(f'R = {R:.3f} | R^2 = {r2:.3f} | RMSE = {rmse:.3f}')

    # scatter with unity line
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'{title}  (R={R:.2f}, R^2={r2:.2f}, RMSE={rmse:.2f})')
    plt.tight_layout()

    # short time overlay
    plt.figure(figsize=(9, 3))
    idx = slice(0, min(n_show, len(y_true)))
    if sample_ts is not None:
        plt.plot(sample_ts[idx], y_true[idx], label='True', linewidth=1.2)
        plt.plot(sample_ts[idx], y_pred[idx], label='Pred', linewidth=1.2)
        plt.xlabel('Time')
    else:
        plt.plot(y_true[idx], label='True', linewidth=1.2)
        plt.plot(y_pred[idx], label='Pred', linewidth=1.2)
        plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.title('First segment: true vs predicted')
    plt.legend()
    plt.tight_layout()

    return {'R': R, 'R2': r2, 'RMSE': rmse}


@dataclass
class DecodingRunConfig:
    fast_mode: bool = False
    make_plots: bool = True
    n_jobs: int = -1
    use_early_stopping: bool = False
    od_wait: int = 20
    # CV options
    cv_mode: str = 'group_kfold'  # reserved for future extensibility
    buffer_samples: int = 0       # if > 0, drop ±buffer around test indices from train

    model_class: type | None = None
    model_kwargs: dict | None = None


def run_cv_decoding(
    X,
    y_df,
    behav_features,
    groups,
    n_splits=5,
    config: DecodingRunConfig | None = None,
    context_label=None,
    verbosity: int = 1,
    shuffle_y: bool = False,
    shuffle_seed: int = 0,
    save_dir: str | Path | None = None,
):
    if config is None:
        config = DecodingRunConfig()

    X = np.asarray(X)
    groups_arr = np.asarray(groups)

    results = []
    rng_global = np.random.default_rng(shuffle_seed)

    # ---------------- helpers ----------------

    def log_msg(msg, level=1):
        log(msg, verbosity=verbosity, level=level)

    def shuffle_y_groupwise(y, groups, rng):
        y_shuf = y.copy()
        for g in np.unique(groups):
            m = groups == g
            y_shuf[m] = rng.permutation(y_shuf[m])
        return y_shuf

    def group_train_val_split(X, y, groups, val_frac=0.2, rng=None):
        unique_groups = np.unique(groups)
        n_val = max(1, int(np.floor(len(unique_groups) * val_frac)))
        val_groups = rng.choice(unique_groups, size=n_val, replace=False)

        val_mask = np.isin(groups, val_groups)
        tr_mask = ~val_mask

        return (
            X[tr_mask], X[val_mask],
            y[tr_mask], y[val_mask],
        )

    # -----------------------------------------

    # Prepare optional per-feature save directory and constants
    out_dir = None
    n_units = int(X.shape[1]) if X.ndim >= 2 else 1
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    total_features = len(behav_features)

    for feat_i, behav_feature in enumerate(
            tqdm(behav_features, total=total_features,
                 desc='Processing features'),
            start=1):
        if y_df[behav_feature].nunique() <= 1:
            continue

        log_msg(behav_feature, level=2)

        y = y_df[behav_feature].to_numpy().ravel()

        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_ok = X[ok]
        y_ok = y[ok]
        groups_ok = groups_arr[ok]

        # ---- assert temporal order preserved ----
        orig_idx = np.where(ok)[0]
        if not np.all(np.diff(orig_idx) > 0):
            raise RuntimeError('Temporal order violated after filtering')

        # ---- structure-aware y shuffle ----
        if shuffle_y:
            if getattr(config, 'cv_mode', 'group_kfold') == 'group_kfold':
                y_ok = shuffle_y_groupwise(y_ok, groups_ok, rng_global)
            else:
                y_ok = rng_global.permutation(y_ok)

        # ---- build CV splits ----
        n = X_ok.shape[0]

        if getattr(config, 'cv_mode', 'group_kfold') == 'sequential_block':
            print('cv_mode = sequential_block')
            all_indices = np.arange(n, dtype=int)
            blocks = np.array_split(all_indices, n_splits)
            buf = int(getattr(config, 'buffer_samples', 0) or 0)

            cv_splits_ok = []
            for block in blocks:
                te = block
                if te.size == 0:
                    cv_splits_ok.append((np.array([], dtype=int), te))
                    continue

                if buf > 0:
                    start = max(0, te[0] - buf)
                    end = min(n, te[-1] + buf + 1)
                    forbidden = np.arange(start, end, dtype=int)
                    tr = np.setdiff1d(all_indices, forbidden)
                    if tr.size == 0:
                        tr = np.setdiff1d(all_indices, te)
                else:
                    tr = np.setdiff1d(all_indices, te)

                cv_splits_ok.append((tr, te))
        else:
            gcv_ok = GroupKFold(n_splits=n_splits)
            cv_splits_ok = list(gcv_ok.split(X_ok, groups=groups_ok))

        # ---- model ----
        if config.model_class is None:
            model_class = CatBoostRegressor
            cb_verbose = False if verbosity == 0 else (
                100 if verbosity >= 2 else False)
            model_kwargs = dict(
                loss_function='RMSE',
                verbose=cb_verbose,
                random_seed=0,
                od_type='Iter' if config.use_early_stopping else None,
                od_wait=config.od_wait if config.use_early_stopping else None,
            )
        else:
            model_class = config.model_class
            model_kwargs = config.model_kwargs or {}

        # ---- CV prediction ----
        y_cv_pred = np.full_like(y_ok, np.nan, dtype=float)

        for fold_i, (train_idx, test_idx) in enumerate(cv_splits_ok):
            X_tr_full = X_ok[train_idx]
            y_tr_full = y_ok[train_idx]
            X_te = X_ok[test_idx]

            if np.unique(y_tr_full).size <= 1:
                log_msg(
                    f'{behav_feature} | fold {fold_i}: constant y, using mean',
                    level=2,
                )
                y_cv_pred[test_idx] = y_tr_full[0]
                continue

            model = model_class(**model_kwargs)
            fit_kwargs = {}

            if config.use_early_stopping:
                if getattr(config, 'cv_mode', 'group_kfold') == 'group_kfold':
                    X_tr, X_val, y_tr, y_val = group_train_val_split(
                        X_tr_full,
                        y_tr_full,
                        groups_ok[train_idx],
                        val_frac=0.2,
                        rng=rng_global,
                    )
                else:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_tr_full,
                        y_tr_full,
                        test_size=0.2,
                        shuffle=True,
                        random_state=shuffle_seed,
                    )

                if X_val is not None and np.unique(y_val).size > 1:
                    fit_verbose = False if verbosity == 0 else (
                        100 if verbosity >= 2 else False)
                    fit_kwargs.update(
                        dict(
                            eval_set=(X_val, y_val),
                            use_best_model=True,
                            verbose=fit_verbose,
                        )
                    )
                else:
                    X_tr, y_tr = X_tr_full, y_tr_full
            else:
                X_tr, y_tr = X_tr_full, y_tr_full

            model.fit(X_tr, y_tr, **fit_kwargs)
            y_cv_pred[test_idx] = model.predict(X_te)

        # ---- metrics ----
        if np.isnan(y_cv_pred).any():
            raise RuntimeError(f'NaNs in CV predictions for {behav_feature}')

        row = {
            'behav_feature': behav_feature,
            'n_samples': len(y_ok),
            'r2_cv': r2_score(y_ok, y_cv_pred),
            'rmse_cv': np.sqrt(mean_squared_error(y_ok, y_cv_pred)),
            'r_cv': np.corrcoef(y_ok, y_cv_pred)[0, 1],
            'context': context_label,
            'model_name': model_class.__name__,
        }
        results.append(row)

        # Optional per-feature save (short filenames)
        if out_dir is not None:
            feat_tag = ''.join(ch if ch.isalnum()
                               else '_' for ch in behav_feature)[:24]
            model_name = (config.model_class.__name__
                          if (config is not None and config.model_class is not None)
                          else CatBoostRegressor.__name__)
            hash_payload = {
                'feat': behav_feature,
                'k': int(n_splits),
                'model': model_name,
                'ctx': context_label,
                'shuf': bool(shuffle_y),
            }
            params_hash = hashlib.sha1(
                json.dumps(hash_payload, sort_keys=True,
                           default=str).encode('utf-8')
            ).hexdigest()[:8]
            fname = f"{feat_tag}_{params_hash}.csv"
            csv_path = out_dir / fname
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            meta = {
                'params_hash': params_hash,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'behav_feature': behav_feature,
                'n_units': n_units,
                'n_splits': int(n_splits),
                'context': context_label,
                'model_name': model_name,
                'shuffle_y': bool(shuffle_y),
            }
            try:
                with open(csv_path.with_suffix('.json'), 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

    results_df = pd.DataFrame(results)

    # optional saving if directory provided
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        X_arr = np.asarray(X)
        n_units = int(X_arr.shape[1]) if X_arr.ndim >= 2 else 1
        model_name = (config.model_class.__name__
                      if (config is not None and config.model_class is not None)
                      else CatBoostRegressor.__name__)
        model_kwargs = (config.model_kwargs or {}
                        ) if config is not None else {}

        hash_payload = {
            'behav_features': sorted(list(behav_features)),
            'n_splits': int(n_splits),
            'model_name': model_name,
            'model_kwargs': model_kwargs,
            'context': context_label,
            'shuffle_y': bool(shuffle_y),
        }
        params_hash = hashlib.sha1(
            json.dumps(hash_payload, sort_keys=True,
                       default=str).encode('utf-8')
        ).hexdigest()[:10]

        fname = f"cv_u{n_units}_k{n_splits}_{model_name}_{params_hash}.csv"
        csv_path = out_dir / fname
        results_df.to_csv(csv_path, index=False)

        meta = {
            'params_hash': params_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'context': context_label,
            'n_units': n_units,
            'n_splits': int(n_splits),
            'behav_features': list(behav_features),
            'model_name': model_name,
            'model_kwargs': model_kwargs,
            'shuffle_y': bool(shuffle_y),
        }
        try:
            with open(csv_path.with_suffix('.json'), 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    return results_df


def _normalize_num_ff(v):
    # numeric → int
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return np.nan
        return int(np.ceil(v))
    # everything else (e.g. 'any')
    return v


def log(msg: str, verbosity: int = 1, level: int = 1):
    if verbosity >= level:
        print(msg)


def decode_by_num_ff_visible_or_in_memory(
    x_var,
    y_var,
    behav_features,
    ff_visibility_col='num_ff_visible',  # 'num_ff_visible' or 'num_ff_in_memory'
    group_col='new_segment',
    n_splits=5,
    config: DecodingRunConfig | None = None,
    save_path=None,
    include_pooled=True,
    load_if_exists=True,
    overwrite=False,
    verbosity: int = 1,
    shuffle_y: bool = False,
    shuffle_seed: int = 0,
):
    """
    Decode behavioral features stratified by num_ff_visible or num_ff_in_memory,
    with optional pooled ('any') decoding.

    Caching logic:
    - If CSV + JSON exist and metadata matches → load
    - If CSV exists without JSON → load only if overwrite=False
    - Otherwise recompute and overwrite if allowed
    """

    if config is None:
        config = DecodingRunConfig()

    assert ff_visibility_col in [
        'num_ff_visible', 'num_ff_in_memory'], "ff_visibility_col must be 'num_ff_visible' or 'num_ff_in_memory'"

    num_ff_visible_or_in_memory_raw = y_var[ff_visibility_col]
    num_ff_visible_or_in_memory_clean = num_ff_visible_or_in_memory_raw.apply(
        _normalize_num_ff)
    # ---------- Prepare metadata & potential cache paths ----------
    X_arr = np.asarray(x_var)
    n_units = int(X_arr.shape[1]) if X_arr.ndim >= 2 else 1
    model_name = (config.model_class.__name__
                  if (config is not None and config.model_class is not None)
                  else CatBoostRegressor.__name__)
    model_kwargs = (config.model_kwargs or {}) if config is not None else {}

    metadata = {
        'version': 1,
        'ff_visibility_col': ff_visibility_col,
        'n_units': n_units,
        'n_samples': int(X_arr.shape[0]),
        'behav_features': list(behav_features),
        'group_col': group_col,
        'n_splits': int(n_splits),
        'include_pooled': bool(include_pooled),
        'model_name': model_name,
        'model_kwargs': model_kwargs,
        'config': {
            'fast_mode': bool(config.fast_mode),
            'make_plots': bool(config.make_plots),
            'n_jobs': int(config.n_jobs),
            'use_early_stopping': bool(config.use_early_stopping),
            'od_wait': int(config.od_wait),
            'cv_mode': getattr(config, 'cv_mode', 'group_kfold'),
            'buffer_samples': int(getattr(config, 'buffer_samples', 0)),
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Stable, short hash to key the run (avoid large filenames)
    hash_payload = {
        'ff_visibility_col': metadata['ff_visibility_col'],
        'n_units': metadata['n_units'],
        'n_splits': metadata['n_splits'],
        'include_pooled': metadata['include_pooled'],
        'behav_features': sorted(metadata['behav_features']),
        'model_name': metadata['model_name'],
        'model_kwargs': metadata['model_kwargs'],
        'config': metadata['config'],
    }
    params_hash = hashlib.sha1(
        json.dumps(hash_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:10]

    metadata['params_hash'] = params_hash

    csv_path = None
    json_path = None
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() == '.csv':
            # Exact file specified
            csv_path = save_path
            json_path = csv_path.with_suffix('.json')
        else:
            # Treat as directory; keep filenames short (user preference)
            # Example: dec_numff_vis_u64_k5_CatBoost_abc123def4.csv
            tag = 'vis' if ff_visibility_col == 'num_ff_visible' else 'mem'
            fname = f"{tag}_u{n_units}_k{n_splits}_{model_name}_{params_hash}"
            if shuffle_y:
                fname += '_shuffled'
            fname += '.csv'
            save_path.mkdir(parents=True, exist_ok=True)
            csv_path = save_path / fname
            json_path = csv_path.with_suffix('.json')

        # Try to load if exists and matching metadata requested
        if load_if_exists and csv_path.exists():
            try:
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        existing_meta = json.load(f)
                    if existing_meta.get('params_hash') == params_hash:
                        log(
                            f'Loaded cached results from {csv_path} (params_hash matches)',
                            verbosity=verbosity,
                            level=1,
                        )
                        return pd.read_csv(csv_path)
                    else:
                        log(f'Cached results do not match metadata. Recomputing.',
                            verbosity=verbosity,
                            level=1,
                            )
                else:
                    # No metadata sidecar; conservatively load only if allowed
                    if not overwrite:
                        log(
                            f'No metadata sidecar. Loaded cached results from {csv_path} (overwrite=False)',
                            verbosity=verbosity,
                            level=1,
                        )
                        return pd.read_csv(csv_path)
                    else:
                        log(f'No metadata sidecar. Recomputing. (overwrite=True)',
                            verbosity=verbosity,
                            level=1,
                            )

            except Exception as e:
                log(f'Error loading cached results from {csv_path}: {e}',
                    verbosity=verbosity,
                    level=1,
                    )
                pass
    else:
        log('No save_path provided. Recomputing.',
            verbosity=verbosity,
            level=1,
            )

    all_results = []

    # ---------- Helper to run one decoding pass ----------
    def _run_one_pass(X_sub, y_sub, groups_sub, context_label, num_ff_label):

        if verbosity >= 1:
            if config.model_class is not None:
                log(f'Fitting: {config.model_class.__name__}',
                    verbosity=verbosity,
                    level=2,
                    )
            else:
                log('Fitting: CatBoostRegressor',
                    verbosity=verbosity,
                    level=2,
                    )

        results_df = run_cv_decoding(
            X=X_sub,
            y_df=y_sub,
            behav_features=behav_features,
            groups=groups_sub,
            n_splits=n_splits,
            config=config,
            context_label=context_label,
            verbosity=verbosity,
            shuffle_y=shuffle_y,
            shuffle_seed=shuffle_seed,
        )

        results_df[ff_visibility_col] = num_ff_label
        return results_df

    # ---------- Per-num_ff decoding ----------
    for num_ff in np.sort(num_ff_visible_or_in_memory_clean.unique()):
        log(f'\nnum_ff: {num_ff}',
            verbosity=verbosity,
            level=1,
            )

        mask = num_ff_visible_or_in_memory_clean == num_ff
        X_sub = x_var[mask]
        y_sub = y_var.loc[mask]
        groups_sub = y_sub[group_col].values

        if len(y_sub) < 50:
            log(f'Skipping num_ff={num_ff}: too few samples',
                verbosity=verbosity,
                level=1,
                )
            continue

        results_df = _run_one_pass(
            X_sub=X_sub,
            y_sub=y_sub,
            groups_sub=groups_sub,
            context_label=f'num_ff={num_ff}',
            num_ff_label=None,  # handled below
        )

        # results_df[ff_visibility_col] = int(num_ff)
        results_df[ff_visibility_col] = num_ff
        results_df['num_ff_group'] = 'stratified'
        all_results.append(results_df)

    # ---------- Pooled decoding ----------
    if include_pooled:
        log('\nnum_ff: any (pooled)',
            verbosity=verbosity,
            level=1,
            )

        results_df = _run_one_pass(
            X_sub=x_var,
            y_sub=y_var,
            groups_sub=y_var[group_col].values,
            context_label='num_ff=any',
            num_ff_label=None,  # handled below
        )

        results_df[ff_visibility_col] = -1
        results_df['num_ff_group'] = 'pooled'
        all_results.append(results_df)

    # ---------- Final table ----------
    results_df = pd.concat(all_results, ignore_index=True)

    if csv_path is not None:
        if overwrite or (not csv_path.exists()):
            results_df.to_csv(csv_path, index=False)
            # Save sidecar metadata for robust retrieval
            print(f'Saving results to {csv_path}')
            try:
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f'Saving metadata to {json_path}')
            except Exception:
                pass

    return results_df


def decode_cur_ff_only(
    x_var,
    y_var,
    behav_features,
    ff_visibility_col='num_ff_visible',  # 'num_ff_visible' or 'num_ff_in_memory'
    group_col='new_segment',
    n_splits=5,
    verbosity: int = 1,
    config: DecodingRunConfig | None = None,
    save_path=None,
):
    """
    Decode behavioral features when the current firefly is the only visible one:
    ff_visibility_col == 1 AND cur_vis == True.

    Returns a results DataFrame with the same format as decode_by_num_ff_visible_or_in_memory.
    """

    if config is None:
        config = DecodingRunConfig()

    mask = (y_var[ff_visibility_col] == 1) & (y_var['cur_vis'] == True)

    if mask.sum() == 0:
        raise ValueError('No samples satisfy cur_ff_only condition.')

    X_sub = x_var[mask]
    y_sub = y_var.loc[mask]
    groups_sub = y_sub[group_col].values

    log(f'\ncur_ff_only: n_samples={mask.sum()}',
        verbosity=verbosity,
        level=1,
        )

    # --- DO NOT precompute CV splits here ---
    # run_cv_decoding will build GroupKFold on the filtered data

    context_label = 'cur_ff_visible_only' if ff_visibility_col == 'num_ff_visible' else 'cur_ff_in_memory_only'

    results_df = run_cv_decoding(
        X=X_sub,
        y_df=y_sub,
        behav_features=behav_features,
        groups=groups_sub,
        n_splits=n_splits,
        config=config,
        context_label=context_label,
    )

    # ---- match decode_by_num_ff_visible_or_in_memory output schema ----
    results_df[ff_visibility_col] = -2
    results_df['num_ff_group'] = 'cur_only'

    if save_path is not None:
        results_df.to_csv(save_path, index=False)

    return results_df


def make_raw_neural_data_processing_tag(pn):
    parts = []
    if not pn.use_raw_spike_data_instead:
        parts.extend(['raw0', 'pca0', 'lag0'])
    else:
        parts.append("raw1")
        if pn.apply_pca_on_raw_spike_data:
            parts.append(f"pca{pn.num_pca_components}")
        else:
            parts.append("pca0")
        if pn.use_lagged_raw_spike_data:
            parts.append(f"lag{pn.rebinned_max_x_lag_number}")
        else:
            parts.append("lag0")
    return '_'.join(parts) if parts else None


def get_band_conditioned_save_path(pn, reg_or_clf):
    neural_data_tag = make_raw_neural_data_processing_tag(pn)
    bin_width_str = f"{pn.bin_width:.4f}".rstrip(
        '0').rstrip('.').replace('.', 'p')
    seg_str = f'bin{bin_width_str}_{pn.cur_or_nxt}_{pn.first_or_last}_st{general_utils.clean_float(pn.start_t_rel_event)}_et{general_utils.clean_float(pn.end_t_rel_event)}'
    if reg_or_clf == 'reg':
        save_path = os.path.join(pn.planning_and_neural_folder_path,
                                 'pn_decoding', 'band_conditioned_reg', neural_data_tag, seg_str)
    elif reg_or_clf == 'clf':
        save_path = os.path.join(pn.planning_and_neural_folder_path,
                                 'pn_decoding', 'band_conditioned_clf', neural_data_tag, seg_str)
    else:
        raise ValueError(f'Invalid reg_or_clf: {reg_or_clf}')
    return save_path
