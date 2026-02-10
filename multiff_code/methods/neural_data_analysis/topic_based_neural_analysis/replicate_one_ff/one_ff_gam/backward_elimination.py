import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import ttest_1samp


from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import GroupSpec, cross_validated_ll, _extract_lambda_config, _format_lambda


def backward_elimination_gam(
    design_df,
    y,
    groups,
    *,
    alpha=0.05,
    n_folds=10,
    verbose=True,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    save_metadata: Optional[Dict] = None,
):
    """
    Paper-faithful backward elimination for one neuron with incremental saving.

    Parameters
    ----------
    design_df : pd.DataFrame
        Design matrix
    y : np.ndarray
        Response variable
    groups : List[GroupSpec]
        Initial group specifications with penalty (lambda) values
    alpha : float, optional
        Significance level for retention, by default 0.05
    n_folds : int, optional
        Number of CV folds, by default 10
    verbose : bool, optional
        Print progress information, by default True
    save_path : Optional[str], optional
        Path to save elimination results. If provided, results are saved after
        each step, by default None
    load_if_exists : bool, optional
        If True and save_path exists, load and resume from saved state. 
        Lambda values are validated against saved configuration - if they don't
        match, cached results are ignored, by default True
    save_metadata : Optional[Dict], optional
        Additional metadata to save with results, by default None

    Returns
    -------
    kept_groups : list of GroupSpec
        Variables retained in the final model.
    history : list of dict
        Elimination steps and statistics.
        
    Notes
    -----
    Lambda (penalty) values from the input groups are saved with the results.
    When loading cached results, the function validates that lambda values match.
    This ensures that elimination results are only reused when penalty settings
    are identical.
    
    For convenience, use `generate_lambda_suffix(groups)` to create descriptive
    filenames that include lambda configuration, e.g.:
        suffix = generate_lambda_suffix(groups)
        save_path = f'results/elimination_{suffix}.pkl'
    
    Examples
    --------
    >>> # Basic usage
    >>> kept, history = backward_elimination_gam(
    ...     design_df, y, groups,
    ...     save_path='results/elim.pkl'
    ... )
    
    >>> # With lambda-based filename
    >>> suffix = generate_lambda_suffix(groups)
    >>> kept, history = backward_elimination_gam(
    ...     design_df, y, groups,
    ...     save_path=f'results/elim_{suffix}.pkl'
    ... )
    """
    import time
    start_time = time.time()
    
    # Save initial groups for lambda validation throughout
    initial_groups = groups.copy()
    
    # Try to load existing results
    if save_path is not None and load_if_exists:
        saved_data = _load_elimination_results(save_path, initial_groups)
        if saved_data is not None and saved_data.get('completed', False):
            # Reconstruct GroupSpec objects
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
    
    # Initialize or resume
    kept = groups.copy()
    history = []
    step = 0
    
    # Prepare save path if needed
    save_path_obj = None
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Compute baseline CV LL for full model
    ll_full = cross_validated_ll(
        design_df, y, kept, n_folds=n_folds
    )
    
    if verbose:
        _print_elimination_header(initial_groups, n_folds, alpha, ll_full.mean())

    improved = True

    while improved and len(kept) > 1:
        step += 1
        improved = False
        candidates = []
        
        if verbose:
            _print_elimination_step_header(step, len(kept))
        
        # Timing for candidate evaluation
        candidate_times = []

        for i, g in enumerate(kept, start=1):
            cand_start = time.time()
            
            reduced = [gg for gg in kept if gg.name != g.name]

            ll_reduced = cross_validated_ll(
                design_df, y, reduced, n_folds=n_folds
            )

            delta = ll_reduced - ll_full  # per fold
            mean_delta = delta.mean()

            # Paired test: is removal NOT harmful?
            t_stat, p_val = ttest_1samp(delta, 0.0)

            candidates.append({
                'group': g,
                'mean_delta': mean_delta,
                'p_value': p_val,
                'll_reduced': ll_reduced,
            })
            
            cand_time = time.time() - cand_start
            candidate_times.append(cand_time)

            if verbose:
                avg_time = np.mean(candidate_times)
                _print_candidate_progress(
                    i, len(kept), g.name, mean_delta, p_val, cand_time, avg_time
                )

        # Choose the variable whose removal hurts least
        candidates.sort(key=lambda x: x['mean_delta'], reverse=True)
        best = candidates[0]

        # Decision rule:
        # If removal does NOT significantly decrease LL → remove it
        will_remove = best['p_value'] > alpha and best['mean_delta'] >= 0
        
        if will_remove:
            removed_group = best['group']
            kept = [g for g in kept if g.name != removed_group.name]
            ll_full = best['ll_reduced']
            improved = True

            history.append({
                'step': step,
                'removed': removed_group.name,
                'delta_ll': best['mean_delta'],
                'p_value': best['p_value'],
            })
        
        if verbose:
            kept_names = [g.name for g in kept]
            _print_elimination_decision(best, will_remove, kept_names)
        
        # Save incrementally after each step
        if save_path_obj is not None:
            _save_elimination_state(
                save_path_obj, kept, history, 
                completed=not improved,  # Mark complete if we're stopping
                current_step=step,
                save_metadata=save_metadata,
                initial_groups=initial_groups
            )

    # Print final summary
    total_time = time.time() - start_time
    if verbose:
        _print_elimination_summary(kept, history, total_time)
    
    # Final save marking completion
    if save_path_obj is not None:
        _save_elimination_state(
            save_path_obj, kept, history,
            completed=True,
            current_step=step,
            save_metadata=save_metadata,
            initial_groups=initial_groups
        )
        if verbose:
            print(f'\n✓ Results saved to: {save_path}')

    return kept, history



def _load_elimination_results(save_path, current_groups):
    """Load existing backward elimination results and validate lambda values."""
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
    
    # Validate lambda configuration
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
    """Save the current backward elimination state with lambda configuration."""
    # Convert GroupSpec objects to dicts for serialization
    kept_dicts = [
        {
            'name': g.name,
            'cols': g.cols,
            'vartype': g.vartype,
            'lam': g.lam,
        }
        for g in kept
    ]
    
    # Extract lambda configuration from initial groups for validation
    lambda_config = _extract_lambda_config(initial_groups)
    
    save_dict = {
        'kept_groups': kept_dicts,
        'history': history,
        'completed': completed,
        'current_step': current_step,
        'lambda_config': lambda_config,  # Save lambda values for validation
    }
    
    if save_metadata is not None:
        save_dict['metadata'] = save_metadata
    
    with open(save_path_obj, 'wb') as f:
        pickle.dump(save_dict, f)


def _print_elimination_header(groups, n_folds, alpha, ll_initial):
    """Print header for backward elimination."""
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
    """Print header for elimination step."""
    print(f'\n{"="*80}')
    print(f'STEP {step}: Evaluating {n_kept} groups for removal')
    print(f'{"="*80}')


def _print_candidate_progress(idx, total, group_name, mean_delta, p_val, elapsed, avg_time):
    """Print progress for candidate evaluation."""
    progress = idx / total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'  [{idx}/{total}] {bar} {100*progress:.0f}%')
    print(f'    Group: {group_name}')
    print(f'    ΔLL = {mean_delta:+.6f}, p = {p_val:.4f}')
    print(f'    Time: {elapsed:.1f}s (avg: {avg_time:.1f}s/candidate)')


def _print_elimination_decision(best, removed, kept_names):
    """Print decision after evaluating all candidates."""
    print('\n' + '─'*80)
    print('DECISION:')
    if removed:
        print(f'  ✓ REMOVED: {best["group"].name}')
        print(f'    ΔLL = {best["mean_delta"]:+.6f}, p = {best["p_value"]:.4f}')
        print(f'  → Remaining groups ({len(kept_names)}): {kept_names}')
    else:
        print(f'  ✗ STOP: {best["group"].name} is significant (p = {best["p_value"]:.4f})')
        print('  → All remaining groups are necessary')
        print(f'  → Final groups ({len(kept_names)}): {kept_names}')
    print('─'*80)


def _print_elimination_summary(kept, history, total_time):
    """Print final summary of backward elimination."""
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
                  f'(ΔLL={h["delta_ll"]:+.4f}, p={h["p_value"]:.4f})')
    
    print('='*80)



def _validate_lambda_match(saved_lambdas, current_lambdas):
    """
    Check if lambda parameters match for keys present in both dictionaries.
    
    Only validates keys that exist in both saved and current lambdas.
    Keys that exist in only one dictionary are ignored.
    
    Args:
        saved_lambdas: Dict with lambda parameters (e.g., {'lam_f': 100, 'lam_g': 10, ...})
        current_lambdas: Dict with lambda parameters
    
    Returns:
        tuple: (bool, str) - (match status, message)
    """
    # Find common keys between saved and current
    common_keys = set(saved_lambdas.keys()) & set(current_lambdas.keys())
    
    if not common_keys:
        return False, "No common lambda keys to compare - treating as mismatch"
    
    # Check if values match for common keys
    mismatches = []
    for key in sorted(common_keys):
        if abs(saved_lambdas[key] - current_lambdas[key]) > 1e-10:
            mismatches.append(
                f"{key}: saved={saved_lambdas[key]:.6e}, current={current_lambdas[key]:.6e}"
            )
    
    if mismatches:
        return False, "Lambda values don't match:\n  " + "\n  ".join(mismatches)
    
    # Report which keys were compared
    compared_keys = ', '.join(sorted(common_keys))
    return True, f"Lambda values match for keys: {compared_keys}"
