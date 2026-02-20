
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import _build_folds
import pickle
from pathlib import Path
from typing import Optional, Dict
from dataclasses import asdict

from scipy.signal import convolve

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import one_ff_gam_fit


# ------------------------------------------------------------------
# Gaussian smoothing kernel (MATLAB equivalent)
# ------------------------------------------------------------------

def make_gaussian_kernel(filtwidth):
    t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
    h = np.exp(-(t ** 2) / (2 * filtwidth ** 2))
    h = h / np.sum(h)
    return h


def smooth_with_kernel(signal, kernel):
    return convolve(signal, kernel, mode='same')


# ------------------------------------------------------------------
# Classical MATLAB-style R² on smoothed firing rate
# ------------------------------------------------------------------

def classical_r2(design_df, beta, spikes, dt, filtwidth=2, invlink=np.exp):

    linear_term = design_df.values @ beta
    r_hat = invlink(linear_term)
    fr_hat = r_hat / dt

    kernel = make_gaussian_kernel(filtwidth)

    smooth_spikes = smooth_with_kernel(spikes, kernel)
    smooth_fr = smooth_spikes / dt

    smooth_fr_hat = smooth_with_kernel(fr_hat, kernel)

    sse = np.sum((smooth_fr_hat - smooth_fr) ** 2)
    sst = np.sum((smooth_fr - np.mean(smooth_fr)) ** 2)

    r2 = 1.0 - (sse / sst)

    return r2


# ------------------------------------------------------------------
# Unified evaluation
# ------------------------------------------------------------------

def evaluate_model(design_df, beta, spikes, dt, filtwidth=2, invlink=np.exp):
    """
    Returns:
        dict with:
            - pseudo_r2
            - classical_r2
            - ll_model
            - ll_mean
            - ll_saturated
    """

    ll_model, ll_mean, ll_sat = one_ff_gam_fit.compute_likelihoods(
        design_df, beta, spikes)

    r2_pseudo = (ll_mean - ll_model) / (ll_mean - ll_sat)

    r2_classical = classical_r2(
        design_df,
        beta,
        spikes,
        dt,
        filtwidth=filtwidth,
        invlink=invlink
    )

    return {
        'pseudo_r2': r2_pseudo,
        'classical_r2': r2_classical,
        'll_model': ll_model,
        'll_mean': ll_mean,
        'll_saturated': ll_sat
    }


# ------------------------------------------------------------------
# Save/load utilities for cross-validation results
# ------------------------------------------------------------------

def _save_crossval_results(
    *,
    save_path,
    cv_results,
    groups,
    dt,
    n_folds,
    random_state,
    fit_kwargs,
    save_metadata,
    verbose,
    cv_mode='blocked_time_buffered',
    buffer_samples=20,
):
    """Save cross-validation variance explained results to pickle file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        'dt': dt,
        'n_folds': n_folds,
        'random_state': random_state,
        'fit_kwargs': fit_kwargs,
    }
    if cv_mode is not None:
        hyperparams['cv_mode'] = cv_mode
    if buffer_samples != 0:
        hyperparams['buffer_samples'] = buffer_samples

    save_dict = {
        'cv_results': cv_results,
        'groups': [asdict(g) for g in groups],
        'hyperparameters': hyperparams,
    }

    if save_metadata is not None:
        save_dict['metadata'] = save_metadata

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    if verbose:
        print(f'\nCross-validation results saved to: {save_path}')


def _maybe_load_saved_crossval(save_path, load_if_exists, verbose):
    """Load saved cross-validation results if they exist."""
    if save_path is None or not load_if_exists:
        return None

    save_path = Path(save_path)
    if not save_path.exists():
        return None

    if verbose:
        print('=' * 80)
        print(f'Loading existing cross-validation results from: {save_path}')
        print('=' * 80)

    with open(save_path, 'rb') as f:
        saved = pickle.load(f)

    cv_results = saved['cv_results']

    if verbose:
        print('✓ Loaded saved cross-validation results:')
        print(f'  Mean Classical R²: {cv_results["mean_classical_r2"]:.4f}')
        print(f'  Mean Pseudo R²: {cv_results["mean_pseudo_r2"]:.4f}')
        print(f'  Number of folds: {len(cv_results["fold_classical_r2"])}')
        print('=' * 80)

    return cv_results


def load_crossval_results(save_path: str) -> Dict:
    """
    Load saved cross-validation variance explained results from pickle file.

    Parameters
    ----------
    save_path : str
        Path to the saved pickle file

    Returns
    -------
    Dict
        Dictionary containing:
        - cv_results: dict with fold_classical_r2, fold_pseudo_r2, 
                      mean_classical_r2, mean_pseudo_r2
        - groups: list of group specifications
        - hyperparameters: dict with dt, n_folds, random_state, fit_kwargs
        - metadata: additional metadata (if provided during fitting)
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def crossval_variance_explained(
    fit_function,
    design_df,
    y,
    groups,
    dt,
    n_folds=5,
    random_state=0,
    fit_kwargs=None,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    save_metadata: Optional[Dict] = None,
    verbose: bool = True,
    cv_mode: Optional[str] = 'blocked_time_buffered',
    buffer_samples: int = 20,
    cv_groups=None,
):
    """
    5-fold cross-validation for Poisson GAM model.

    Parameters
    ----------
    fit_function : callable
        Your fitting function (e.g. one_ff_gam_fit.fit_poisson_gam)
    design_df : pandas.DataFrame
    y : array-like
        Spike counts
    groups : list
        Group structure for GAM
    dt : float
        Bin width
    n_folds : int
        Number of cross-validation folds (default: 5)
    random_state : int
        Random seed for reproducibility (default: 0)
    fit_kwargs : dict
        Extra arguments passed to fit_function
    save_path : str, optional
        Path to save/load cross-validation results (pickle format)
    load_if_exists : bool
        If True and save_path exists, load and return saved results (default: True)
    save_metadata : dict, optional
        Additional metadata to save with results
    verbose : bool
        Print progress and diagnostic information (default: True)
    cv_mode : str, optional
        Cross-validation mode. Options (same as cv_decoding):
        - 'blocked_time_buffered': contiguous time blocks with buffer exclusion
        - 'blocked_time': forward-chaining (past → future)
        - 'group_kfold': GroupKFold (requires cv_groups)
        - None: shuffled KFold (default)
    buffer_samples : int
        Number of samples to exclude as buffer in 'blocked_time_buffered' (default: 0)
    cv_groups : array-like, optional
        Sample-level group labels for 'group_kfold'. Required when cv_mode='group_kfold'.

    Returns
    -------
    dict with:
        - fold_classical_r2 : array of classical R² for each fold
        - fold_pseudo_r2 : array of pseudo R² for each fold
        - mean_classical_r2 : mean classical R² across folds
        - mean_pseudo_r2 : mean pseudo R² across folds
    """

    # ------------------------------------------------------------------
    # 1) Load cached result if requested
    # ------------------------------------------------------------------
    maybe_loaded = _maybe_load_saved_crossval(
        save_path=save_path,
        load_if_exists=load_if_exists,
        verbose=verbose,
    )
    if maybe_loaded is not None:
        return maybe_loaded

    # ------------------------------------------------------------------
    # 2) Run cross-validation
    # ------------------------------------------------------------------
    if fit_kwargs is None:
        fit_kwargs = {}

    y = np.asarray(y)
    n = len(design_df)

    if cv_mode is not None:
        splits = _build_folds(
            n,
            n_splits=n_folds,
            groups=cv_groups,
            cv_splitter=cv_mode,
            buffer_samples=buffer_samples,
            random_state=random_state,
        )
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(np.arange(n)))

    fold_classical_r2 = []
    fold_pseudo_r2 = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):

        # ----------------------
        # Split data
        # ----------------------
        train_design_df = design_df.iloc[train_idx]
        test_design_df = design_df.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ----------------------
        # Fit on training fold
        # ----------------------
        fit_res = fit_function(
            design_df=train_design_df,
            y=y_train,
            groups=groups,
            **fit_kwargs
        )

        beta = fit_res.coef

        # ----------------------
        # Evaluate on test fold
        # ----------------------
        metrics = evaluate_model(
            design_df=test_design_df,
            beta=beta,
            spikes=y_test,
            dt=dt,
        )

        fold_classical_r2.append(metrics["classical_r2"])
        fold_pseudo_r2.append(metrics["pseudo_r2"])

        if verbose:
            print(f"Fold {fold_idx + 1}: "
                  f"Classical R² = {metrics['classical_r2']:.4f}, "
                  f"Pseudo R² = {metrics['pseudo_r2']:.4f}")

    cv_results = {
        "fold_classical_r2": np.array(fold_classical_r2),
        "fold_pseudo_r2": np.array(fold_pseudo_r2),
        "mean_classical_r2": np.mean(fold_classical_r2),
        "mean_pseudo_r2": np.mean(fold_pseudo_r2),
    }

    # ------------------------------------------------------------------
    # 3) Save results
    # ------------------------------------------------------------------
    if save_path is not None:
        _save_crossval_results(
            save_path=save_path,
            cv_results=cv_results,
            groups=groups,
            dt=dt,
            n_folds=n_folds,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
            save_metadata=save_metadata,
            verbose=verbose,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
        )

    return cv_results


def plot_variance_explained_cdf(all_mean_r2,
                                alpha=0.05,
                                label='Uncoupled model',
                                figsize=(6, 6),
                                x_max=0.2,
                                log_x=False,
                                clip_min=None):
    """
    Plot empirical CDF of variance explained with DKW confidence band.
    """

    import numpy as np

    # Convert to array and remove NaNs
    r2 = np.asarray(all_mean_r2)
    r2 = r2[np.isfinite(r2)]

    if len(r2) == 0:
        raise ValueError("No valid variance explained values to plot.")

    # Optional clipping for visualization
    if clip_min is not None:
        r2_plot = np.clip(r2, clip_min, None)
    else:
        r2_plot = r2.copy()

    # Sort
    r2_sorted = np.sort(r2_plot)
    n = len(r2_sorted)

    # Empirical CDF
    cdf = np.arange(1, n + 1) / n

    # DKW band
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.maximum(cdf - epsilon, 0)
    upper = np.minimum(cdf + epsilon, 1)

    # Plot
    plt.figure(figsize=figsize)

    plt.plot(r2_sorted, cdf, linewidth=2, label=label)
    plt.fill_between(r2_sorted, lower, upper, alpha=0.25)

    plt.axhline(0.5, linestyle='--', color='gray', linewidth=1)
    plt.axvline(0.1, linestyle='--', color='gray', linewidth=1)

    plt.xlabel('Variance explained', fontsize=16)
    plt.ylabel('Cumulative fraction of neurons', fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1])

    if log_x:
        # Prevent zero issues
        min_positive = np.min(r2_sorted[r2_sorted > 0])
        lower_limit = max(min_positive, clip_min if clip_min else min_positive)
        plt.xscale('log')
        plt.xlim([lower_limit, x_max])
    else:
        plt.xlim([0, x_max])

    plt.legend(fontsize=14, frameon=True)
    plt.tight_layout()
    plt.show()

    print('Median variance explained:', np.median(r2))
