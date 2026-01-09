import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
from datetime import datetime
import hashlib
import json
from typing import Dict, Any, Optional

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import add_interactions, discrete_decoders
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions.band_conditioned import conditional_decoding_clf


def summarize_bootstrap_deltas(df):
    """
    Collapse CV folds within each bootstrap, then compute CI across bootstraps.
    """

    collapsed = (
        df
        .groupby(['bootstrap_id', 'condition_value', 'model'], as_index=False)
        .agg(
            delta_balanced_accuracy=('delta_balanced_accuracy', 'mean'),
            global_balanced_accuracy=('global_balanced_accuracy', 'mean'),
            cond_balanced_accuracy=('cond_balanced_accuracy', 'mean'),
        )
    )

    summary = (
        collapsed
        .groupby(['condition_value', 'model'], as_index=False)
        .agg(
            mean_delta_bal_acc=('delta_balanced_accuracy', 'mean'),
            ci_low=('delta_balanced_accuracy',
                    lambda x: np.percentile(x, 2.5)),
            ci_high=('delta_balanced_accuracy',
                     lambda x: np.percentile(x, 97.5)),
        )
    )

    return collapsed, summary


def run_pairwise_interaction_analysis(
    *,
    x_df,
    y_df,
    var_a,
    var_b,
    interaction_col,
    unconditioned_model_types=('logreg', 'svm', 'ridge'),
    conditioned_model_types=('logreg', 'svm', 'ridge'),
    min_count=200,
    n_splits=3,
    n_bootstraps=100,
    random_state=0,
    save_path=None,
    load_if_exists=True,
    overwrite=False,
    verbosity: int = 1,
    processing_flags: Optional[Dict[str, Any]] = None,
    cache_tag: Optional[str] = None,
):
    """
    Pairwise interaction decoding with paired, trial-level bootstrap
    of (conditioned − global), collapsing CV folds within each bootstrap.
    """

    def _log(msg, level=1):
        if verbosity >= level:
            print(msg)

    def _build_cache_paths(sp, params_hash):
        sp = Path(sp)
        # Optional processing-aware subdirectory
        tag = _make_processing_tag(processing_flags, cache_tag)
        if sp.suffix.lower() != '.csv' and tag:
            sp = sp / tag
        if sp.suffix.lower() == '.csv':
            prefix = sp.with_suffix('')
        else:
            sp.mkdir(parents=True, exist_ok=True)
            prefix = sp / f'pair_{params_hash}'
        paths = {
            'meta': Path(str(prefix) + '_meta.json'),
            'interaction_results': Path(str(prefix) + '_interaction_results.csv'),
            'interaction_summary': Path(str(prefix) + '_interaction_summary.csv'),
            'cond_a_raw': Path(str(prefix) + '_cond_a_raw.csv'),
            'cond_a_boot': Path(str(prefix) + '_cond_a_boot.csv'),
            'cond_a_summary': Path(str(prefix) + '_cond_a_summary.csv'),
            'cond_b_raw': Path(str(prefix) + '_cond_b_raw.csv'),
            'cond_b_boot': Path(str(prefix) + '_cond_b_boot.csv'),
            'cond_b_summary': Path(str(prefix) + '_cond_b_summary.csv'),
        }
        return paths

    def _save_outputs(paths, metadata, outs):
        try:
            outs['interaction_results'].to_csv(
                paths['interaction_results'], index=False)
            outs['interaction_summary'].to_csv(
                paths['interaction_summary'], index=False)
            outs['cond_var_a_delta_raw'].to_csv(
                paths['cond_a_raw'], index=False)
            outs['cond_var_a_delta_bootstrap'].to_csv(
                paths['cond_a_boot'], index=False)
            outs['cond_var_a_delta_summary'].to_csv(
                paths['cond_a_summary'], index=False)
            outs['cond_var_b_delta_raw'].to_csv(
                paths['cond_b_raw'], index=False)
            outs['cond_var_b_delta_bootstrap'].to_csv(
                paths['cond_b_boot'], index=False)
            outs['cond_var_b_delta_summary'].to_csv(
                paths['cond_b_summary'], index=False)
            with open(paths['meta'], 'w') as f:
                json.dump(metadata, f, indent=2)
            _log(
                f"Saved pairwise interaction results to {paths['interaction_results'].parent}", level=1)
        except Exception as e:
            _log(f'Warning: failed saving cached results: {e}', level=1)

    def _load_outputs(paths):
        try:
            out = {
                'x_pruned': None,
                'y_pruned': None,
                'interaction_results': pd.read_csv(paths['interaction_results']),
                'interaction_summary': pd.read_csv(paths['interaction_summary']),
                'cond_var_a_delta_raw': pd.read_csv(paths['cond_a_raw']),
                'cond_var_a_delta_bootstrap': pd.read_csv(paths['cond_a_boot']),
                'cond_var_a_delta_summary': pd.read_csv(paths['cond_a_summary']),
                'cond_var_b_delta_raw': pd.read_csv(paths['cond_b_raw']),
                'cond_var_b_delta_bootstrap': pd.read_csv(paths['cond_b_boot']),
                'cond_var_b_delta_summary': pd.read_csv(paths['cond_b_summary']),
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
        for k in sorted(k for k in proc_flags.keys() if k not in key_map):
            v = proc_flags[k]
            if isinstance(v, (bool, int)):
                v = int(bool(v))
            parts.append(f"{k}={v}")
        return '_'.join(parts) if parts else None

    if save_path is not None:
        n_units_meta = int(x_df.shape[1]) if hasattr(
            x_df, 'shape') and len(x_df.shape) >= 2 else None
        n_samples_meta = int(len(x_df))
        metadata = {
            'version': 1,
            'var_a': str(var_a),
            'var_b': str(var_b),
            'interaction_col': str(interaction_col),
            'unconditioned_model_types': list(unconditioned_model_types),
            'conditioned_model_types': list(conditioned_model_types),
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
            'var_a': metadata['var_a'],
            'var_b': metadata['var_b'],
            'interaction_col': metadata['interaction_col'],
            'unconditioned_model_types': tuple(sorted(metadata['unconditioned_model_types'])),
            'conditioned_model_types': tuple(sorted(metadata['conditioned_model_types'])),
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
        if load_if_exists and all(p.exists() for p in save_paths.values()):
            try:
                with open(save_paths['meta'], 'r') as f:
                    existing_meta = json.load(f)
                if existing_meta.get('params_hash') == params_hash:
                    _log(
                        f"Loaded cached pairwise interaction results (hash={params_hash})", level=1)
                    loaded = _load_outputs(save_paths)
                    if loaded is not None:
                        return loaded
                else:
                    _log('Cached metadata mismatch. Recomputing.', level=1)
            except Exception as e:
                _log(f'Error reading cache metadata: {e}', level=1)
        else:
            _log('Computing new results...', level=1)

    # ============================================================
    # 1. Interaction label
    # ============================================================
    y_df = add_interactions.add_pairwise_interaction(
        df=y_df,
        var_a=var_a,
        var_b=var_b,
        new_col=interaction_col,
    )

    # ============================================================
    # 2. Prune rare interaction states
    # ============================================================
    y_pruned, x_pruned = add_interactions.prune_rare_states_two_dfs(
        df_behavior=y_df,
        df_neural=x_df,
        label_col=interaction_col,
        min_count=min_count,
    )

    # ============================================================
    # 3. Global interaction decoding (model sweep, no bootstrap)
    # ============================================================
    interaction_results = discrete_decoders.sweep_decoders_xy(
        x_df=x_pruned,
        y_df=y_pruned,
        label_col=interaction_col,
        model_types=list(unconditioned_model_types),
        n_splits=n_splits,
    )

    interaction_summary = (
        interaction_results
        .groupby('model', as_index=False)
        .agg(mean_bal_acc=('balanced_accuracy', 'mean'))
    )

    # ============================================================
    # 4–5. Bootstrapped Δ decoding (shared logic)
    # ============================================================
    def run_conditioned_delta(target_col, condition_col):
        all_models = []

        for model_type in conditioned_model_types:
            df = conditional_decoding_clf.bootstrap_conditioned_minus_global(
                x_df=x_pruned,
                y_df=y_pruned,
                target_col=target_col,
                condition_col=condition_col,
                model_type=model_type,
                n_splits=n_splits,
                n_bootstraps=n_bootstraps,
                min_samples=min_count,
                random_state=random_state,
            )
            df['model'] = model_type
            df['target'] = target_col
            df['condition'] = condition_col
            all_models.append(df)

        raw = pd.concat(all_models, ignore_index=True)
        collapsed, summary = summarize_bootstrap_deltas(raw)

        # ------------------------------------------------------------
        # Attach per-condition sample sizes for labeling in plots
        # Use the pruned behavioral dataframe counts for the condition
        # (constant across bootstraps; repeated per-row after merge)
        # ------------------------------------------------------------
        condition_counts = (
            y_pruned[condition_col]
            .value_counts()
            .rename_axis('condition_value')
            .reset_index(name='n_samples')
        )
        present_vals = collapsed['condition_value'].unique()
        condition_counts = condition_counts[
            condition_counts['condition_value'].isin(present_vals)
        ]
        collapsed = collapsed.merge(
            condition_counts, on='condition_value', how='left')

        return raw, collapsed, summary

    # var_a | var_b
    (
        cond_a_raw,
        cond_a_collapsed,
        cond_a_summary,
    ) = run_conditioned_delta(var_a, var_b)

    # var_b | var_a
    (
        cond_b_raw,
        cond_b_collapsed,
        cond_b_summary,
    ) = run_conditioned_delta(var_b, var_a)

    # ============================================================
    # 6. Return
    # ============================================================
    outs = {
        'x_pruned': x_pruned,
        'y_pruned': y_pruned,

        # Interaction decoding
        'interaction_results': interaction_results,
        'interaction_summary': interaction_summary,

        # Conditioned − global (raw per fold)
        'cond_var_a_delta_raw': cond_a_raw,
        'cond_var_b_delta_raw': cond_b_raw,

        # Collapsed per bootstrap
        'cond_var_a_delta_bootstrap': cond_a_collapsed,
        'cond_var_b_delta_bootstrap': cond_b_collapsed,

        # Final CI summaries
        'cond_var_a_delta_summary': cond_a_summary,
        'cond_var_b_delta_summary': cond_b_summary,
    }
    if save_paths is not None and (overwrite or not all(p.exists() for p in save_paths.values())):
        _save_outputs(save_paths, metadata, outs)
    return outs
