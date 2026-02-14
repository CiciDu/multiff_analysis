import itertools
import pickle
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import KFold

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import one_ff_gam_fit


def _format_lambda(lam):
    """Format lambda value concisely for display."""
    if lam == 0:
        return '0'
    elif lam >= 1000:
        return f'{lam:.2e}'
    elif lam >= 1:
        return f'{lam:.1f}'
    else:
        return f'{lam:.2e}'


def generate_tuning_filename(lam_grid, base_name='tuning', suffix='.pkl'):
    """
    Generate a filename for penalty tuning based on lambda grid.

    Parameters
    ----------
    lam_grid : dict
        Lambda grid like {'lam_f': [1, 10, 100], 'lam_g': [1, 10]}
    base_name : str, optional
        Base name for file, by default 'tuning'
    suffix : str, optional
        File suffix, by default '.pkl'

    Returns
    -------
    str
        Filename like 'tuning_lamf-1-10-100_lamg-1-10.pkl'

    Examples
    --------
    >>> lam_grid = {'lam_f': [1, 10, 100], 'lam_g': [1, 10]}
    >>> generate_tuning_filename(lam_grid)
    'tuning_lamf-1-10-100_lamg-1-10.pkl'
    """
    parts = []
    for key, values in lam_grid.items():
        # Clean key name (remove underscores)
        key_clean = key.replace('_', '').replace('lam', 'lam')
        # Format values
        val_strs = [_format_lambda(v).replace('.', 'p') for v in values]
        parts.append(f'{key_clean}-{"-".join(val_strs)}')

    return f'{base_name}_{"_".join(parts)}{suffix}'


def load_tuning_results(save_path: str) -> Dict:
    """
    Load saved penalty tuning results from pickle file.

    Parameters
    ----------
    save_path : str
        Path to the saved pickle file

    Returns
    -------
    Dict
        Dictionary containing:
        - best_lams: Dictionary of best lambda values (best so far if partial)
        - best_score: Best CV score achieved (best so far if partial)
        - results: List of (lam_setting, score) tuples for all tested combinations
        - lam_grid: The lambda grid used for search
        - group_name_map: The group name mapping used
        - n_folds: Number of CV folds used
        - n_completed: Number of combinations tested so far
        - n_total: Total number of combinations to test
        - metadata: Additional metadata (if provided during tuning)
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def poisson_log_likelihood(y, rate, eps=1e-12):
    """
    Mean Poisson log-likelihood per sample (dropping constants).
    """
    rate = np.maximum(rate, eps)
    return np.mean(y * np.log(rate) - rate)


def clone_groups_with_lams(groups, lam_map):
    """
    lam_map: dict {group_name: lambda}
    """
    new_groups = []
    for g in groups:
        lam = lam_map.get(g.name, g.lam)
        new_groups.append(
            one_ff_gam_fit.GroupSpec(
                name=g.name,
                cols=g.cols,
                vartype=g.vartype,
                lam=lam,
            )
        )
    return new_groups


def cv_score_gam(
    design_df,
    y,
    groups,
    l1_groups,
    n_folds=5,
    random_state=0,
    verbose_errors=False,
):
    """
    Returns mean held-out Poisson log-likelihood.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    ll_vals = []
    failed_folds = []

    X = design_df.to_numpy()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = design_df.iloc[train_idx]
        y_train = y[train_idx]

        X_test = design_df.iloc[test_idx]
        y_test = y[test_idx]

        fit_res = one_ff_gam_fit.fit_poisson_gam(
            design_df=X_train,
            y=y_train,
            groups=groups,
            l1_groups=l1_groups,
            tol=1e-6,
            verbose=False,
        )

        if not fit_res.success:
            # Hard fail → terrible score
            ll_vals.append(-np.inf)
            failed_folds.append((fold_idx, fit_res.message, fit_res.grad_norm))
            continue

        beta = fit_res.coef.values
        rate_test = np.exp(X_test.to_numpy() @ beta)
        ll_vals.append(poisson_log_likelihood(y_test, rate_test))

    # If verbose and all folds failed, print diagnostic info
    if verbose_errors and len(failed_folds) == n_folds:
        print(f'\n  ⚠ ALL {n_folds} FOLDS FAILED!')
        for fold_idx, msg, grad_norm in failed_folds[:2]:  # Show first 2
            print(f'    Fold {fold_idx}: {msg} (|grad|={grad_norm:.3e})')

    return np.mean(ll_vals)


# ========== Helper functions for tune_penalties ==========

def _load_existing_results(save_path, lam_grid):
    """Load existing results and determine if search is complete or partial."""
    save_path_obj = Path(save_path)
    if not save_path_obj.exists():
        return None, set(), -np.inf, None, []

    print('='*80)
    print(f'Loading existing results from: {save_path}')

    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)

    n_completed = len(saved_data['results'])
    keys = list(lam_grid.keys())
    n_total = int(np.prod([len(lam_grid[k]) for k in keys]))

    if n_completed >= n_total:
        print(f'✓ All {n_total} combinations already tested.')
        print(f'✓ Best lambdas: {saved_data["best_lams"]}')
        print(f'✓ Best score: {saved_data["best_score"]:.6f}')
        print('='*80)
        return saved_data, None, None, None, None

    print(
        f'⚠ Found partial results: {n_completed}/{n_total} combinations tested ({100*n_completed/n_total:.1f}%)')
    print(
        f'⚠ Best so far: {saved_data["best_lams"]} (LL={saved_data["best_score"]:.6f})')
    print('⚠ Resuming from where it left off...')
    print('='*80)

    tested_combos = {
        tuple(sorted(lam_setting.items()))
        for lam_setting, _ in saved_data['results']
    }

    return None, tested_combos, saved_data['best_score'], saved_data['best_lams'], saved_data['results']


def _print_grid_search_setup(lam_grid, group_name_map, n_combinations, n_folds, save_path):
    """Print the initial grid search configuration."""
    print('='*80)
    print('PENALTY TUNING - GRID SEARCH')
    print('='*80)
    print('Parameter grid:')
    for k, values in lam_grid.items():
        print(f'  {k}: {values}')
    print('\nGroup mapping:')
    for k, gnames in group_name_map.items():
        print(f'  {k} → {gnames}')
    print(f'\nTotal combinations: {n_combinations}')
    print(f'CV folds: {n_folds}')
    if save_path is not None:
        print(f'Save path: {save_path}')
        print('Incremental saving: ENABLED')
    print('='*80)


def _format_time(seconds):
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds/60:.1f}m'
    else:
        return f'{seconds/3600:.1f}h'


def _print_progress(lam_setting, score, best_score, is_new_best,
                    combo_time, combo_times, n_completed, n_total, start_time):
    """Print progress information for the current combination."""
    elapsed = time.time() - start_time
    # Use last 10 for more recent estimate
    avg_time = np.mean(combo_times[-10:])
    remaining = n_total - n_completed
    eta_seconds = avg_time * remaining
    eta_str = _format_time(eta_seconds)

    # Progress bar
    progress = n_completed / n_total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    # Print status
    best_marker = ' ⭐ NEW BEST!' if is_new_best else ''
    print(f'[{n_completed}/{n_total}] {bar} {100*progress:.1f}%')
    print(f'  Params: {lam_setting}')
    print(f'  Score:  LL={score:.6f} | Best={best_score:.6f}{best_marker}')
    print(
        f'  Time:   {combo_time:.1f}s | Avg={avg_time:.1f}s | ETA={eta_str} | Elapsed={elapsed/60:.1f}m')
    print()


def _save_tuning_state(save_path_obj, best_lams, best_score, results,
                       lam_grid, group_name_map, n_folds, n_combinations, save_metadata):
    """Save the current tuning state to disk."""
    save_dict = {
        'best_lams': best_lams,
        'best_score': best_score,
        'results': results,
        'lam_grid': lam_grid,
        'group_name_map': group_name_map,
        'n_folds': n_folds,
        'n_completed': len(results),
        'n_total': n_combinations,
    }

    if save_metadata is not None:
        save_dict['metadata'] = save_metadata

    with open(save_path_obj, 'wb') as f:
        pickle.dump(save_dict, f)


def _print_final_summary(results, n_combinations, combo_times, best_lams, best_score,
                         start_time, save_path):
    """Print the final summary after grid search completion."""
    total_time = time.time() - start_time
    print('='*80)
    print('GRID SEARCH COMPLETE')
    print('='*80)
    print(f'✓ Tested {len(results)}/{n_combinations} combinations')
    print(
        f'✓ Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
    if combo_times:
        print(f'✓ Average time per combination: {np.mean(combo_times):.1f}s')

    if best_lams is not None:
        print('\n✓ BEST PARAMETERS:')
        for k, v in best_lams.items():
            print(f'    {k} = {v}')
        print(f'✓ BEST SCORE: {best_score:.6f}')
    else:
        print('\n⚠ WARNING: No valid model fits found!')
        print('⚠ All cross-validation fits failed (all scores were -inf)')
        print('⚠ This may indicate:')
        print('    - Numerical instability in the model')
        print('    - Inappropriate penalty values')
        print('    - Issues with the data or design matrix')

    if save_path is not None:
        print(f'\n✓ Results saved to: {save_path}')
    print('='*80)


def _validate_data(design_df, y, base_groups):
    """Validate data before running grid search."""
    print('Validating data...')

    # Check for NaN/Inf in design matrix
    X = design_df.to_numpy()
    if np.any(np.isnan(X)):
        n_nan = np.sum(np.isnan(X))
        print(f'  ⚠ WARNING: Design matrix contains {n_nan} NaN values!')
        return False
    if np.any(np.isinf(X)):
        n_inf = np.sum(np.isinf(X))
        print(f'  ⚠ WARNING: Design matrix contains {n_inf} Inf values!')
        return False

    # Check for NaN/Inf in response
    if np.any(np.isnan(y)):
        print('  ⚠ WARNING: Response variable contains NaN values!')
        return False
    if np.any(np.isinf(y)):
        print('  ⚠ WARNING: Response variable contains Inf values!')
        return False

    # Check for negative spike counts
    if np.any(y < 0):
        print('  ⚠ WARNING: Response variable contains negative values!')
        return False

    # Check data dimensions
    print(f'  ✓ Design matrix shape: {X.shape}')
    print(f'  ✓ Response shape: {y.shape}')
    print(f'  ✓ Mean spike count: {y.mean():.3f}')
    print(f'  ✓ Spike count range: [{y.min():.1f}, {y.max():.1f}]')
    print(
        f'  ✓ Non-zero spike bins: {np.sum(y > 0)} / {len(y)} ({100*np.mean(y > 0):.1f}%)')
    print(f'  ✓ Number of groups: {len(base_groups)}')

    # Check for extreme values in design matrix
    X_abs_max = np.abs(X).max()
    X_abs_mean = np.abs(X).mean()
    print(
        f'  ✓ Design matrix |values|: max={X_abs_max:.2e}, mean={X_abs_mean:.2e}')
    if X_abs_max > 1e3:
        print(
            f'  ⚠ WARNING: Design matrix has very large values (max={X_abs_max:.2e})')
        print('  ⚠ This may cause numerical instability. Consider standardizing features.')

    # Check group specifications
    col_index = {c: i for i, c in enumerate(design_df.columns)}
    for g in base_groups:
        missing_cols = [c for c in g.cols if c not in col_index]
        if missing_cols:
            print(
                f'  ⚠ WARNING: Group {g.name} has missing columns: {missing_cols}')
            return False

    # Check for constant columns (besides intercept)
    for col in design_df.columns:
        if col == 'const':
            continue
        col_vals = design_df[col].values
        if np.std(col_vals) < 1e-10:
            print(f'  ⚠ WARNING: Column {col} appears to be constant!')

    print('  ✓ All data validation checks passed\n')
    return True


def tune_penalties(
    design_df,
    y,
    base_groups,
    l1_groups,
    lam_grid,
    group_name_map,
    n_folds=5,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    save_metadata: Optional[Dict] = None,
    retrieve_only: bool = False,
):
    """
    Grid search over penalty parameters using cross-validation.

    Results are saved incrementally after each lambda combination is tested,
    allowing for safe interruption and resumption of long-running searches.

    Parameters
    ----------
    design_df : pd.DataFrame
        Design matrix
    y : np.ndarray
        Response variable
    base_groups : List[GroupSpec]
        Base group specifications (lambdas will be overridden)
    l1_groups : List[GroupSpec]
        L1 penalty groups
    lam_grid : dict
        Dictionary of {key: list of lambdas}
        e.g. {'lam_f': [1,10,100], 'lam_g': [1,10]}
    group_name_map : dict
        Dictionary of {key: [group_names]}
        e.g. {'lam_f': ['pos','vel'], 'lam_g': ['t_targ','t_move','t_rew']}
    n_folds : int, optional
        Number of CV folds, by default 5
    save_path : Optional[str], optional
        Path to save tuning results. If provided, results are saved after
        each combination is tested, by default None
    load_if_exists : bool, optional
        If True and save_path exists, load cached results. If all combinations
        are complete, returns immediately. If partial, resumes from where it
        left off, by default True
    save_metadata : Optional[Dict], optional
        Additional metadata to save with results, by default None
    retrieve_only : bool, optional
        If True, only retrieve existing results without running any new fits.
        Returns partial results if available. Useful for checking progress or
        retrieving results from interrupted runs. Requires save_path to be set
        and existing results file to exist, by default False

    Returns
    -------
    Tuple[Optional[Dict], List[Tuple]]
        best_lams: Dictionary of best lambda values, or None if all fits failed
        results: List of (lam_setting, score) tuples for all combinations

    Notes
    -----
    For convenience, use `generate_tuning_filename(lam_grid)` to create 
    descriptive filenames that include the lambda grid configuration.

    Examples
    --------
    >>> # Basic usage
    >>> best_lams, results = tune_penalties(
    ...     design_df, y, base_groups, l1_groups,
    ...     lam_grid={'lam_f': [1, 10, 100], 'lam_g': [1, 10]},
    ...     group_name_map={'lam_f': ['pos', 'vel'], 'lam_g': ['t_targ']},
    ...     save_path='results/tuning.pkl'
    ... )

    >>> # With lambda-based filename
    >>> filename = generate_tuning_filename(lam_grid, base_name='neuron_123')
    >>> best_lams, results = tune_penalties(
    ...     design_df, y, base_groups, l1_groups,
    ...     lam_grid=lam_grid,
    ...     group_name_map=group_name_map,
    ...     save_path=f'results/{filename}'
    ... )

    >>> # Retrieve existing results without running new fits
    >>> best_lams, results = tune_penalties(
    ...     design_df, y, base_groups, l1_groups,
    ...     lam_grid=lam_grid,
    ...     group_name_map=group_name_map,
    ...     save_path='results/tuning.pkl',
    ...     retrieve_only=True
    ... )
    """
    # Handle retrieve_only mode
    if retrieve_only:
        if save_path is None:
            raise ValueError(
                "retrieve_only=True requires save_path to be specified")
        if not Path(save_path).exists():
            raise FileNotFoundError(f"No results file found at {save_path}")

        # Load whatever results exist
        saved_data, tested_combos, best_score, best_lams, results = _load_existing_results(
            save_path, lam_grid
        )
        if saved_data is not None:
            # Complete results found
            print(f"✓ Retrieved complete results: {len(results)} combinations")
            return saved_data['best_lams'], saved_data['results']
        elif results:
            # Partial results found
            print(
                f"✓ Retrieved partial results: {len(results)} combinations tested")
            if best_lams is not None:
                print(f"  Current best score: {best_score:.6f}")
                print(f"  Current best lambdas: {best_lams}")
            return best_lams, results
        else:
            raise ValueError(
                f"Results file exists at {save_path} but contains no valid results")

    # Validate data before proceeding with new fits
    if not _validate_data(design_df, y, base_groups):
        raise ValueError('Data validation failed! See warnings above.')

    # Try to load existing results for resumption
    if save_path is not None and load_if_exists:
        saved_data, tested_combos, best_score, best_lams, results = _load_existing_results(
            save_path, lam_grid
        )
        if saved_data is not None:
            # Complete results found, return immediately
            return saved_data['best_lams'], saved_data['results']
    else:
        tested_combos = set()
        best_score = -np.inf
        best_lams = None
        results = []

    # Calculate grid dimensions
    keys = list(lam_grid.keys())
    n_combinations = int(np.prod([len(lam_grid[k]) for k in keys]))

    # Print setup information
    _print_grid_search_setup(lam_grid, group_name_map,
                             n_combinations, n_folds, save_path)

    # Prepare save path if needed
    save_path_obj = None
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Initialize timing
    start_time = time.time()
    combo_times = []
    n_tested = 0  # Track how many we've actually tested (not resumed)

    # Main grid search loop
    for idx, values in enumerate(itertools.product(*[lam_grid[k] for k in keys]), 1):
        lam_setting = dict(zip(keys, values))

        # Skip if already tested (in case of resume)
        lam_tuple = tuple(sorted(lam_setting.items()))
        if lam_tuple in tested_combos:
            continue

        # Time this combination
        combo_start = time.time()

        # Build lambda mapping for groups
        lam_map = {}
        for k, lam in lam_setting.items():
            for gname in group_name_map[k]:
                lam_map[gname] = lam

        # Fit and score (show errors for first few combinations)
        groups = clone_groups_with_lams(base_groups, lam_map)
        # Show detailed errors for first 3 combinations
        verbose_errors = (n_tested < 3)
        score = cv_score_gam(design_df, y, groups, l1_groups,
                             n_folds, verbose_errors=verbose_errors)
        n_tested += 1

        # Track timing
        combo_time = time.time() - combo_start
        combo_times.append(combo_time)

        # Update results
        results.append((lam_setting, score))
        is_new_best = score > best_score
        if is_new_best:
            best_score = score
            best_lams = lam_setting

        # Print progress
        _print_progress(
            lam_setting, score, best_score, is_new_best,
            combo_time, combo_times, len(results), n_combinations, start_time
        )

        # Save incrementally
        if save_path_obj is not None:
            _save_tuning_state(
                save_path_obj, best_lams, best_score, results,
                lam_grid, group_name_map, n_folds, n_combinations, save_metadata
            )

    # Print final summary
    _print_final_summary(
        results, n_combinations, combo_times, best_lams, best_score, start_time, save_path
    )

    return best_lams, results
