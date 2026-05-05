import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import \
    _build_folds
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec, _extract_lambda_config, _format_lambda, cross_validated_ll,
    cross_validated_ll_and_null)
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Module-level candidate evaluator — must be top-level for joblib to pickle
# ---------------------------------------------------------------------------

def _eval_candidate(
    design_df,
    y,
    kept,
    group_name,
    ll_full,
    folds,
    warm_start_coefs,
    n_folds,
    cv_mode,
    buffer_samples,
):
    """
    Evaluate removing one group from 'kept'.

    Returns (mean_delta, p_val, ll_reduced, coef_reduced).
      mean_delta > 0  →  reduced model is worse (higher NLL)
      p_val is from Wilcoxon signed-rank, left-tailed (matches neuroGAM)
    """
    reduced = [g for g in kept if g.name != group_name]
    ll_reduced, coef_reduced = cross_validated_ll(
        design_df, y, reduced,
        n_folds=n_folds,
        cv_mode=cv_mode,
        buffer_samples=buffer_samples,
        folds=folds,
        warm_start_coefs=warm_start_coefs,
        return_coefs=True,
    )
    delta = ll_reduced - ll_full
    mean_delta = float(np.nanmean(delta))
    valid = np.isfinite(ll_full) & np.isfinite(ll_reduced)
    try:
        _, p_val = wilcoxon(ll_full[valid], ll_reduced[valid], alternative='less')
    except ValueError:
        p_val = 1.0
    return mean_delta, p_val, ll_reduced, coef_reduced


# ---------------------------------------------------------------------------
# Main backward-elimination function
# ---------------------------------------------------------------------------

def backward_elimination_gam(
    design_df,
    y,
    groups,
    *,
    alpha=0.05,
    n_folds=20,
    cv_mode='blocked_time_buffered',
    buffer_samples=20,
    trial_ids=None,
    n_jobs=1,
    verbose=True,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    save_metadata: Optional[Dict] = None,
):
    """
    Backward elimination for one neuron, matching neuroGAM's algorithm.

    Parameters
    ----------
    design_df : pd.DataFrame
    y : np.ndarray
    groups : List[GroupSpec]
    alpha : float
        Significance level (default 0.05).
    n_folds : int
        Number of CV folds (default 20).
    cv_mode : str
        CV strategy passed to _build_folds.
        Options: 'blocked_time_buffered' (default), 'blocked_time',
        'group_kfold', 'kfold', 'matlab_interleaved'.
    buffer_samples : int
        Buffer size for 'blocked_time_buffered' mode (default 20).
    trial_ids : np.ndarray, optional
        Per-sample group labels (length == len(y)). Required when
        cv_mode='group_kfold'; ignored otherwise.
    n_jobs : int
        Number of parallel jobs for evaluating candidates within each step.
        1 = sequential (default). -1 = use all available threads.
        Uses threading (prefer='threads') to avoid data-copy overhead.
    verbose : bool
    save_path : str, optional
    load_if_exists : bool
    save_metadata : dict, optional

    Returns
    -------
    kept_groups : list of GroupSpec
    history : list of dict
    """
    start_time = time.time()

    # Save initial groups for lambda validation throughout
    initial_groups = groups.copy()

    # Try to load existing results
    if save_path is not None and load_if_exists:
        saved_data = _load_elimination_results(save_path, initial_groups)
        if saved_data is not None and saved_data.get('completed', False):
            kept = [
                GroupSpec(
                    name=g['name'],
                    cols=g['cols'],
                    vartype=g['vartype'],
                    lam=g['lam'],
                )
                for g in saved_data['kept_groups']
            ]
            return kept, saved_data['history']

    # Initialize
    kept = groups.copy()
    history = []
    step = 0

    # Prepare save path
    save_path_obj = None
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # (3) Pre-build folds once — reused across all steps and all candidates.
    #     random_state=0 matches the hardcoded default in cross_validated_ll.
    # -----------------------------------------------------------------------
    folds = _build_folds(
        len(y),
        n_splits=n_folds,
        groups=trial_ids,
        cv_splitter=cv_mode,
        random_state=0,
        buffer_samples=buffer_samples,
    )

    # -----------------------------------------------------------------------
    # Baseline CV NLL for the full model, plus the mean-rate null.
    # Also collect per-fold coefficients for warm-starting candidates.
    # -----------------------------------------------------------------------
    ll_full, ll_null, warm_start_coefs = cross_validated_ll_and_null(
        design_df, y, kept,
        n_folds=n_folds,
        cv_mode=cv_mode,
        buffer_samples=buffer_samples,
        folds=folds,
        return_coefs=True,
    )

    if verbose:
        _print_elimination_header(initial_groups, n_folds, alpha, ll_full.mean())

    improved = True

    while improved and len(kept) > 1:
        step += 1
        improved = False

        if verbose:
            _print_elimination_step_header(step, len(kept))

        # -------------------------------------------------------------------
        # (1) Evaluate all candidates — parallel or sequential
        # -------------------------------------------------------------------
        if n_jobs == 1:
            candidates = _evaluate_candidates_sequential(
                design_df, y, kept, ll_full, folds,
                warm_start_coefs, n_folds, cv_mode, buffer_samples, verbose,
            )
        else:
            candidates = _evaluate_candidates_parallel(
                design_df, y, kept, ll_full, folds,
                warm_start_coefs, n_folds, cv_mode, buffer_samples,
                n_jobs, verbose,
            )

        # Pick candidate whose removal hurts least (smallest NLL increase)
        candidates.sort(key=lambda x: x['mean_delta'])
        best = candidates[0]

        alpha_threshold = alpha
        will_remove = best['p_value'] >= alpha_threshold

        if will_remove:
            removed_group = best['group']
            kept = [g for g in kept if g.name != removed_group.name]
            ll_full = best['ll_reduced']
            # (2) Update warm start to the coefficients of the new best model
            warm_start_coefs = best['coef_reduced']
            improved = True

            history.append({
                'step': step,
                'removed': removed_group.name,
                'delta_ll': best['mean_delta'],
                'p_value': best['p_value'],
                'alpha_step': alpha_threshold,
            })

        if verbose:
            kept_names = [g.name for g in kept]
            _print_elimination_decision(best, will_remove, kept_names, alpha_threshold)

        if save_path_obj is not None:
            _save_elimination_state(
                save_path_obj, kept, history,
                completed=not improved,
                current_step=step,
                save_metadata=save_metadata,
                initial_groups=initial_groups,
            )

    # -----------------------------------------------------------------------
    # Null-model check (matches neuroGAM)
    # -----------------------------------------------------------------------
    alpha_null = alpha
    null_rejected = False
    if kept:
        improvement = ll_null - ll_full  # positive when model beats null
        valid_null = np.isfinite(improvement)
        try:
            _, p_null = wilcoxon(improvement[valid_null], alternative='greater')
            null_rejected = p_null < alpha_null
        except ValueError:
            null_rejected = False

        if not null_rejected:
            if verbose:
                print('\n' + '─' * 80)
                print('NULL CHECK: best model not significantly better than mean-rate null '
                      f'(p={p_null:.4f} >= α={alpha_null})')
                print('→ Setting kept = [] (no variable survives)')
                print('─' * 80)
            kept = []
        else:
            if verbose:
                print('\n' + '─' * 80)
                print('NULL CHECK: best model significantly better than mean-rate null '
                      f'(p={p_null:.4f} < α={alpha_null})')
                print('→ Variable survives')
                print('─' * 80)

    total_time = time.time() - start_time
    if verbose:
        _print_elimination_summary(kept, history, total_time)

    if save_path_obj is not None:
        _save_elimination_state(
            save_path_obj, kept, history,
            completed=True,
            current_step=step,
            save_metadata=save_metadata,
            initial_groups=initial_groups,
        )
        if verbose:
            print(f'\n✓ Results saved to: {save_path}')

    return kept, history


# ---------------------------------------------------------------------------
# Sequential and parallel inner-loop helpers
# ---------------------------------------------------------------------------

def _evaluate_candidates_sequential(
    design_df, y, kept, ll_full, folds,
    warm_start_coefs, n_folds, cv_mode, buffer_samples, verbose,
):
    """Evaluate all candidates sequentially, printing live progress."""
    candidates = []
    candidate_times = []

    for i, g in enumerate(kept, start=1):
        cand_start = time.time()

        mean_delta, p_val, ll_reduced, coef_reduced = _eval_candidate(
            design_df, y, kept, g.name,
            ll_full, folds, warm_start_coefs,
            n_folds, cv_mode, buffer_samples,
        )

        candidates.append({
            'group': g,
            'mean_delta': mean_delta,
            'p_value': p_val,
            'll_reduced': ll_reduced,
            'coef_reduced': coef_reduced,
        })

        cand_time = time.time() - cand_start
        candidate_times.append(cand_time)

        if verbose:
            avg_time = np.mean(candidate_times)
            _print_candidate_progress(
                i, len(kept), g.name, mean_delta, p_val, cand_time, avg_time,
            )

    return candidates


def _evaluate_candidates_parallel(
    design_df, y, kept, ll_full, folds,
    warm_start_coefs, n_folds, cv_mode, buffer_samples,
    n_jobs, verbose,
):
    """Evaluate all candidates in parallel using threads."""
    step_start = time.time()
    if verbose:
        effective = n_jobs if n_jobs > 0 else 'all available'
        print(f'  [parallel n_jobs={effective}] evaluating {len(kept)} candidates ...')

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_eval_candidate)(
            design_df, y, kept, g.name,
            ll_full, folds, warm_start_coefs,
            n_folds, cv_mode, buffer_samples,
        )
        for g in kept
    )

    step_time = time.time() - step_start
    candidates = [
        {
            'group': g,
            'mean_delta': mean_delta,
            'p_value': p_val,
            'll_reduced': ll_reduced,
            'coef_reduced': coef_reduced,
        }
        for g, (mean_delta, p_val, ll_reduced, coef_reduced) in zip(kept, results)
    ]

    if verbose:
        _print_candidates_parallel(candidates, step_time)

    return candidates


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def _load_elimination_results(save_path, current_groups):
    save_path_obj = Path(save_path)
    if not save_path_obj.exists():
        print('='*80)
        print(f'ℹ No existing results found at: {save_path}')
        print('ℹ Starting fresh backward elimination...')
        print('='*80)
        return None

    print('='*80)
    print(f'Loading existing results from: {save_path}')

    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)

    if 'lambda_config' in saved_data:
        current_lambdas = _extract_lambda_config(current_groups)
        matches, msg = _validate_lambda_match(saved_data['lambda_config'], current_lambdas)

        if not matches:
            print('⚠ WARNING: Lambda mismatch detected!')
            print(f'⚠ {msg}')
            print('⚠ Ignoring cached results due to lambda mismatch.')
            print('='*80)
            return None
        else:
            print('✓ Lambda values validated - match saved configuration')
    else:
        print('⚠ Warning: No lambda information in saved file (old format)')

    if saved_data.get('completed', False):
        print('✓ Backward elimination already complete.')
        print(f'✓ Final model has {len(saved_data["kept_groups"])} groups')
        print(f'✓ Groups retained: {[g["name"] for g in saved_data["kept_groups"]]}')
        print('='*80)
        return saved_data

    print(f'⚠ Found partial results: {saved_data["current_step"]} steps completed')
    print(f'⚠ Current model has {len(saved_data["kept_groups"])} groups')
    print('⚠ Resuming elimination...')
    print('='*80)
    return saved_data


def _save_elimination_state(save_path_obj, kept, history, completed, current_step,
                            save_metadata, initial_groups):
    kept_dicts = [
        {'name': g.name, 'cols': g.cols, 'vartype': g.vartype, 'lam': g.lam}
        for g in kept
    ]
    lambda_config = _extract_lambda_config(initial_groups)

    save_dict = {
        'kept_groups': kept_dicts,
        'history': history,
        'completed': completed,
        'current_step': current_step,
        'lambda_config': lambda_config,
    }
    if save_metadata is not None:
        save_dict['metadata'] = save_metadata

    with open(save_path_obj, 'wb') as f:
        pickle.dump(save_dict, f)


# ---------------------------------------------------------------------------
# Verbose printing helpers
# ---------------------------------------------------------------------------

def _print_elimination_header(groups, n_folds, alpha, ll_initial):
    print('='*80)
    print('BACKWARD ELIMINATION - GAM')
    print('='*80)
    print(f'Initial groups: {len(groups)}')
    print(f'CV folds: {n_folds}')
    print(f'Significance level (α): {alpha}')
    print(f'Initial model LL: {ll_initial:.6f}')
    print('\nPenalty (λ) configuration:')
    lambda_config = _extract_lambda_config(groups)
    for key in ['lam_f', 'lam_g', 'lam_h', 'lam_p']:
        if key in lambda_config:
            print(f'  {key}: λ={_format_lambda(lambda_config[key])}')
    print('='*80)


def _print_elimination_step_header(step, n_kept):
    print(f'\n{"="*80}')
    print(f'STEP {step}: Evaluating {n_kept} groups for removal')
    print(f'{"="*80}')


def _print_candidate_progress(idx, total, group_name, mean_delta, p_val, elapsed, avg_time):
    progress = idx / total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f'  [{idx}/{total}] {bar} {100*progress:.0f}%')
    print(f'    Group: {group_name}')
    print(f'    ΔLL = {mean_delta:+.6f}, p = {p_val:.4f}')
    print(f'    Time: {elapsed:.1f}s (avg: {avg_time:.1f}s/candidate)')


def _print_candidates_parallel(candidates, step_time):
    n = len(candidates)
    print(f'  Parallel evaluation complete: {step_time:.1f}s wall-clock ({n} candidates)')
    print(f'  {"#":>4}  {"Group":<35}  {"ΔLL":>12}  {"p-value":>10}')
    print(f'  {"─"*4}  {"─"*35}  {"─"*12}  {"─"*10}')
    for i, c in enumerate(candidates, start=1):
        print(f'  {i:>4}  {c["group"].name:<35}  {c["mean_delta"]:>+12.6f}  {c["p_value"]:>10.4f}')


def _print_elimination_decision(best, removed, kept_names, alpha_threshold):
    print('\n' + '─'*80)
    print('DECISION:')
    if removed:
        print(f'  ✓ REMOVED: {best["group"].name}')
        print(f'    ΔLL = {best["mean_delta"]:+.6f}, p = {best["p_value"]:.4f} (α = {alpha_threshold:.4f})')
        print(f'  → Remaining groups ({len(kept_names)}): {kept_names}')
    else:
        print(f'  ✗ STOP: {best["group"].name} is significant (p = {best["p_value"]:.4f} < α = {alpha_threshold:.4f})')
        print('  → All remaining groups are necessary')
        print(f'  → Final groups ({len(kept_names)}): {kept_names}')
    print('─'*80)


def _print_elimination_summary(kept, history, total_time):
    print('\n' + '='*80)
    print('BACKWARD ELIMINATION COMPLETE')
    print('='*80)
    print(f'✓ Total steps: {len(history)}')
    print(f'✓ Variables removed: {len(history)}')
    print(f'✓ Variables retained: {len(kept)}')
    print(f'✓ Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')

    print(f'\n✓ FINAL MODEL ({len(kept)} groups):')
    for g in kept:
        print(f'    • {g.name}')

    if history:
        print('\n✓ ELIMINATION HISTORY:')
        for h in history:
            print(f'    Step {h["step"]}: Removed {h["removed"]} '
                  f'(ΔLL={h["delta_ll"]:+.4f}, p={h["p_value"]:.4f}, α={h.get("alpha_step", float("nan")):.4f})')

    print('='*80)


# ---------------------------------------------------------------------------
# Lambda validation helper
# ---------------------------------------------------------------------------

def _validate_lambda_match(saved_lambdas, current_lambdas):
    """
    Check if lambda parameters match for keys present in both dictionaries.
    Returns (bool, message).
    """
    common_keys = set(saved_lambdas.keys()) & set(current_lambdas.keys())

    if not common_keys:
        return False, "No common lambda keys to compare - treating as mismatch"

    mismatches = []
    for key in sorted(common_keys):
        if abs(saved_lambdas[key] - current_lambdas[key]) > 1e-10:
            mismatches.append(
                f"{key}: saved={saved_lambdas[key]:.6e}, current={current_lambdas[key]:.6e}"
            )

    if mismatches:
        return False, "Lambda values don't match:\n  " + "\n  ".join(mismatches)

    compared_keys = ', '.join(sorted(common_keys))
    return True, f"Lambda values match for keys: {compared_keys}"
