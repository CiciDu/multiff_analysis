import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding


# ============================================================
# Small utilities
# ============================================================

def _normalize_num_ff(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return np.nan if np.isnan(v) else int(np.ceil(v))
    return v


def log(msg: str, verbosity: int = 1, level: int = 1):
    if verbosity >= level:
        print(msg)


def _run_decoding_pass(
    X_sub,
    y_sub,
    behav_features,
    groups_sub,
    n_splits,
    config,
    context_label,
    verbosity,
    shuffle_y,
    shuffle_seed,
):
    return cv_decoding.run_cv_decoding(
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


# ============================================================
# Main: stratified decoding by num_ff
# ============================================================

def decode_by_num_ff_visible_or_in_memory(
    x_var,
    y_var,
    behav_features,
    ff_visibility_col='num_ff_visible',
    group_col='new_segment',
    n_splits=5,
    config: Optional[cv_decoding.DecodingRunConfig] = None,
    save_path=None,
    include_pooled=True,
    load_if_exists=True,
    overwrite=False,
    verbosity: int = 1,
    shuffle_y: bool = False,
    shuffle_seed: int = 0,
):
    if config is None:
        config = cv_decoding.DecodingRunConfig()

    assert ff_visibility_col in (
        'num_ff_visible',
        'num_ff_in_memory',
    )

    num_ff_clean = y_var[ff_visibility_col].apply(_normalize_num_ff)

    # ---------- caching (group-level only) ----------
    csv_path = None
    json_path = None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        csv_path = save_path / f'decode_by_{ff_visibility_col}.csv'
        json_path = csv_path.with_suffix('.json')

        if load_if_exists and csv_path.exists() and not overwrite:
            log(f'Loaded cached results from {csv_path}',
                verbosity=verbosity, level=1)
            return pd.read_csv(csv_path)

    all_results = []

    # ---------- stratified ----------
    for num_ff in np.sort(num_ff_clean.dropna().unique()):
        mask = num_ff_clean == num_ff
        if mask.sum() < 50:
            log(f'Skipping num_ff={num_ff}: too few samples',
                verbosity=verbosity, level=1)
            continue

        log(f'\nnum_ff={num_ff}', verbosity=verbosity, level=1)

        X_sub = x_var[mask]
        y_sub = y_var.loc[mask]
        groups_sub = y_sub[group_col].values

        results_df = _run_decoding_pass(
            X_sub=X_sub,
            y_sub=y_sub,
            behav_features=behav_features,
            groups_sub=groups_sub,
            n_splits=n_splits,
            config=config,
            context_label=f'num_ff={num_ff}',
            verbosity=verbosity,
            shuffle_y=shuffle_y,
            shuffle_seed=shuffle_seed,
        )

        results_df[ff_visibility_col] = num_ff
        results_df['num_ff_group'] = 'stratified'
        all_results.append(results_df)

    # ---------- pooled ----------
    if include_pooled:
        log('\nnum_ff=any (pooled)', verbosity=verbosity, level=1)

        results_df = _run_decoding_pass(
            X_sub=x_var,
            y_sub=y_var,
            behav_features=behav_features,
            groups_sub=y_var[group_col].values,
            n_splits=n_splits,
            config=config,
            context_label='num_ff=any',
            verbosity=verbosity,
            shuffle_y=shuffle_y,
            shuffle_seed=shuffle_seed,
        )

        results_df[ff_visibility_col] = -1
        results_df['num_ff_group'] = 'pooled'
        all_results.append(results_df)

    results_df = pd.concat(all_results, ignore_index=True)

    if csv_path is not None:
        results_df.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(
                dict(
                    ff_visibility_col=ff_visibility_col,
                    behav_features=list(behav_features),
                    n_splits=n_splits,
                    include_pooled=include_pooled,
                    shuffle_y=shuffle_y,
                    timestamp=datetime.now().isoformat(),
                ),
                f,
                indent=2,
            )

    return results_df


# ============================================================
# Current-FF-only decoding
# ============================================================

def decode_cur_ff_only(
    x_var,
    y_var,
    behav_features,
    ff_visibility_col='num_ff_visible',
    group_col='new_segment',
    n_splits=5,
    verbosity: int = 1,
    config: Optional[cv_decoding.DecodingRunConfig] = None,
    save_path=None,
):
    if config is None:
        config = cv_decoding.DecodingRunConfig()

    mask = (y_var[ff_visibility_col] == 1) & (y_var['cur_vis'] == True)
    if mask.sum() == 0:
        raise ValueError('No samples satisfy cur_ff_only condition.')

    log(f'\ncur_ff_only: n_samples={mask.sum()}',
        verbosity=verbosity, level=1)

    X_sub = x_var[mask]
    y_sub = y_var.loc[mask]
    groups_sub = y_sub[group_col].values

    context = (
        'cur_ff_visible_only'
        if ff_visibility_col == 'num_ff_visible'
        else 'cur_ff_in_memory_only'
    )

    results_df = _run_decoding_pass(
        X_sub=X_sub,
        y_sub=y_sub,
        behav_features=behav_features,
        groups_sub=groups_sub,
        n_splits=n_splits,
        config=config,
        context_label=context,
        verbosity=verbosity,
        shuffle_y=False,
        shuffle_seed=0,
    )

    results_df[ff_visibility_col] = -2
    results_df['num_ff_group'] = 'cur_only'

    if save_path is not None:
        results_df.to_csv(save_path, index=False)

    return results_df


# ============================================================
# Tag helper (unchanged, already slim)
# ============================================================

def make_raw_neural_data_processing_tag(pn):
    parts = []
    if not pn.use_raw_spike_data_instead:
        parts.extend(['raw0', 'pca0', 'lag0'])
    else:
        parts.append('raw1')
        parts.append(
            f'pca{pn.num_pca_components}'
            if pn.apply_pca_on_raw_spike_data else 'pca0'
        )
        parts.append(
            f'lag{pn.rebinned_max_x_lag_number}'
            if pn.use_lagged_raw_spike_data else 'lag0'
        )
    return '_'.join(parts) if parts else None
