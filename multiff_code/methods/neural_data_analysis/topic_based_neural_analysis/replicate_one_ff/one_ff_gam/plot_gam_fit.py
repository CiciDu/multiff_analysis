import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_parameters


def category_contributions_to_df(category_contributions):
    """
    Convert category_contributions dict to a DataFrame.
    """
    contrib_df = pd.DataFrame.from_dict(category_contributions, orient='index')
    contrib_df.index.name = 'category'
    return contrib_df.reset_index()


def plot_category_variance_contributions(
    result,
    *,
    sort_by='delta_pseudo_r2',
    figsize=(8, 4),
):
    """
    Plot category contribution results from run_category_variance_contributions().

    Parameters
    ----------
    result : dict
        Output of OneFFGAMRunner.run_category_variance_contributions().
    sort_by : str
        'delta_pseudo_r2' or 'delta_classical_r2'.
    figsize : tuple
        Figure size for each panel.

    Returns
    -------
    pd.DataFrame
        Sorted plotting dataframe.
    """
    if sort_by not in {'delta_pseudo_r2', 'delta_classical_r2'}:
        raise ValueError(
            "sort_by must be 'delta_pseudo_r2' or 'delta_classical_r2'"
        )

    df = category_contributions_to_df(result['category_contributions'])
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    full_cv = result['full_cv_result']
    full_classical = full_cv['mean_classical_r2']
    full_pseudo = full_cv['mean_pseudo_r2']

    # Delta bars
    plt.figure(figsize=figsize)
    plt.bar(df['category'], df[sort_by])
    plt.axhline(0.0, color='black', linewidth=1.0)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel(sort_by)
    plt.title(f'Category Contributions ({sort_by})')
    plt.tight_layout()
    plt.show()

    # Full vs leave-out pseudo R2
    plt.figure(figsize=figsize)
    plt.plot(
        df['category'],
        [full_pseudo] * len(df),
        marker='o',
        label='full model pseudo R2',
    )
    plt.plot(
        df['category'],
        df['leave_out_mean_pseudo_r2'],
        marker='o',
        label='leave-out pseudo R2',
    )
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('mean pseudo R2')
    plt.title('Full vs Leave-One-Category-Out (Pseudo R2)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Full vs leave-out classical R2
    plt.figure(figsize=figsize)
    plt.plot(
        df['category'],
        [full_classical] * len(df),
        marker='o',
        label='full model classical R2',
    )
    plt.plot(
        df['category'],
        df['leave_out_mean_classical_r2'],
        marker='o',
        label='leave-out classical R2',
    )
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('mean classical R2')
    plt.title('Full vs Leave-One-Category-Out (Classical R2)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def _ecdf(vals, eps=1e-6, clip_for_log=True):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if clip_for_log:
        x = np.clip(x, eps, None)  # needed for log-scale
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def plot_category_ecdf(all_results, *, log_x=True):
    """
    all_results: list of dicts from
      runner.run_category_variance_contributions(unit_idx=..., retrieve_only=True/False)
    log_x: if True use log-scaled x-axis; if False use linear x-axis.
    """
    # collect per-neuron metrics
    total = []
    sensory = []
    latent = []
    motor = []
    other = []

    # map your category names -> legend buckets
    cat_map = {
        "sensory_vars": sensory,
        "latent_vars": latent,
        "event_vars": motor,  # rename if you want "Motor" in legend
        "position_vars": other,  # adjust mapping as needed
        # you can also merge eye_position/spike_hist into "other"
        "eye_position_vars": other,
        "spike_hist_vars": other,
    }

    for res in all_results:
        total.append(res["full_cv_result"]["mean_pseudo_r2"])
        cc = res["category_contributions"]
        for k, bucket in cat_map.items():
            if k in cc:
                bucket.append(cc[k]["delta_pseudo_r2"])

    # colors close to your example
    curves = [
        ("Sensory", sensory, "#8CBF26"),
        ("Latent",  latent,  "#E6550D"),
        ("Motor",   motor,   "#00A6D6"),
        ("Other",   other,   "#A9A9A9"),
        ("Total",   total,   "#111111"),
    ]

    fig, ax = plt.subplots(figsize=(4.1, 3.0), dpi=180)

    if log_x:
        # shaded log bands (optional, like your figure)
        ax.axvspan(1e-3, 1e-2, color="0.9", zorder=0)
        ax.axvspan(1e-2, 1e-1, color="0.95", zorder=0)

    for label, vals, color in curves:
        if len(vals) == 0:
            continue
        x, y = _ecdf(vals, clip_for_log=log_x)
        ax.plot(x, y, lw=1.6, color=color, label=label)

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.0)
    else:
        all_vals = np.concatenate(
            [np.asarray(v, dtype=float) for _, v, _ in curves if len(v) > 0]
        )
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            x_max = float(np.max(finite_vals))
            pad = 0.05 * max(1e-9, x_max)
            ax.set_xlim(0.0, x_max + pad)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Variance explained")
    ax.set_ylabel("Cumulative prob.")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.show()
    
    
def _cols_in_beta(cols, beta):
    """
    Filter column names to only those present in beta.
    When NA rows are dropped during fitting, some design columns may be
    absent from the fitted coefficients.
    """
    beta_index = None

    # pandas Series/DataFrame paths
    idx_attr = getattr(beta, 'index', None)
    if idx_attr is not None and not callable(idx_attr):
        beta_index = idx_attr
    else:
        col_attr = getattr(beta, 'columns', None)
        if col_attr is not None and not callable(col_attr):
            beta_index = col_attr

    # list/tuple/ndarray/index-like path
    if beta_index is None and isinstance(beta, (list, tuple, np.ndarray)):
        beta_index = beta

    if beta_index is None:
        return list(cols)
    return [c for c in cols if c in beta_index]


def plot_spike_history(beta, hist_meta):
    """
    Plot spike history filter.

    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    hist_meta : dict
        Metadata with 'groups' and 'basis_info'
    """
    if 'spike_hist' not in hist_meta['groups']:
        print("No spike_hist found in model")
        return

    cols = _cols_in_beta(hist_meta['groups']['spike_hist'], beta)
    if not cols:
        print("No spike_hist coefficients found in beta (all may have been dropped)")
        return
    w = beta[cols].to_numpy()

    # Reconstruct kernel from basis functions
    if 'basis_info' in hist_meta and 'spike_hist' in hist_meta['basis_info']:
        info = hist_meta['basis_info']['spike_hist']
        lags = info['lags']
        basis = info['basis']
        # When some cols were dropped, use only the basis columns that exist
        basis_indices = _parse_basis_indices(cols, 'spike_hist')
        if basis_indices is not None and len(basis_indices) == len(w):
            valid = (basis_indices >= 0) & (basis_indices < basis.shape[1])
            B_sub = basis[:, basis_indices[valid]]
            w_sub = w[valid]
            kernel = B_sub @ w_sub
            lags_ms = lags * 1000
            xlabel = 'Time lag (ms)'
        elif basis.shape[1] == len(w):
            kernel = basis @ w
            lags_ms = lags * 1000
            xlabel = 'Time lag (ms)'
        else:
            kernel = w
            lags_ms = np.arange(len(w))
            xlabel = 'Basis index'
            print("Warning: basis/weights size mismatch; plotting weights directly.")
    else:
        # Fallback: just plot the weights
        kernel = w
        lags_ms = np.arange(len(w))
        xlabel = 'Basis index'
        print("Warning: No basis_info found for 'spike_hist'. Plotting weights directly.")

    plt.figure()
    plt.plot(lags_ms, kernel, marker='o' if len(kernel) <= 20 else None)
    plt.xlabel(xlabel)
    plt.ylabel('Weight (log-rate change)')
    plt.title('Spike history filter')
    plt.axhline(0.0, color='k', ls='--', lw=1)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_event_kernel(var, beta, temporal_meta):
    """
    Plot temporal event kernel (e.g., t_targ, t_move, t_rew, t_stop).

    Parameters
    ----------
    var : str
        Event variable name (e.g., 't_move')
    beta : Series or DataFrame
        Fitted coefficients
    temporal_meta : dict
        Metadata with 'groups' and 'basis_info'
    """
    if var not in temporal_meta['groups']:
        print(f"No {var} found in model")
        return

    cols = _cols_in_beta(temporal_meta['groups'][var], beta)
    if not cols:
        print(f"No {var} coefficients found in beta (all may have been dropped)")
        return
    w = beta[cols].to_numpy()

    # Reconstruct kernel from basis functions
    if 'basis_info' in temporal_meta and var in temporal_meta['basis_info']:
        info = temporal_meta['basis_info'][var]
        lags = info['lags']
        basis = info['basis']
        basis_indices = _parse_basis_indices(cols, var)
        if basis_indices is not None and len(basis_indices) == len(w):
            valid = (basis_indices >= 0) & (basis_indices < basis.shape[1])
            B_sub = basis[:, basis_indices[valid]]
            w_sub = w[valid]
            kernel = B_sub @ w_sub
            lags_ms = lags * 1000
            xlabel = 'Time relative to event (ms)'
        elif basis.shape[1] == len(w):
            kernel = basis @ w
            lags_ms = lags * 1000
            xlabel = 'Time relative to event (ms)'
        else:
            kernel = w
            lags_ms = np.arange(len(w))
            xlabel = 'Basis index'
            print(f"Warning: basis/weights size mismatch for '{var}'; plotting weights directly.")
    else:
        # Fallback: just plot the weights
        kernel = w
        lags_ms = np.arange(len(w))
        xlabel = 'Basis index'
        print(
            f"Warning: No basis_info found for '{var}'. Plotting weights directly.")

    plt.figure()
    plt.plot(lags_ms, kernel, marker='o' if len(kernel) <= 20 else None)
    plt.xlabel(xlabel)
    plt.ylabel('Weight (log-rate change)')
    plt.title(f'Event kernel: {var}')
    plt.axhline(0.0, color='k', ls='--', lw=1)
    plt.axvline(0.0, color='r', ls='--', lw=1, alpha=0.5, label='Event time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def _parse_bin_indices(cols, var):
    """Parse bin indices from column names like 'var:bin0', 'var:bin1', ..."""
    pattern = re.compile(rf'^{re.escape(var)}:bin(\d+)$')
    indices = []
    for c in cols:
        m = pattern.match(c)
        if m:
            indices.append(int(m.group(1)))
    return np.array(indices) if indices else None


def _parse_basis_indices(cols, var):
    """
    Parse basis coefficient indices from column names like 'var:b0:5', 'var:b0:7', ...
    Returns indices in the same order as cols, or None if parsing fails.
    """
    pattern = re.compile(rf'^{re.escape(var)}:b\d+:(\d+)$')
    indices = []
    for c in cols:
        m = pattern.match(c)
        if m:
            indices.append(int(m.group(1)))
        else:
            return None
    return np.array(indices)


def plot_linear_tuning(var, beta, tuning_meta):
    cols = _cols_in_beta(tuning_meta['groups'][var], beta)
    if not cols:
        print(f"No {var} coefficients found in beta (all may have been dropped)")
        return
    w = beta[cols].to_numpy()
    gain = np.exp(w)

    # Get bin edges and compute bin centers
    if 'bin_edges' in tuning_meta and var in tuning_meta['bin_edges']:
        edges = tuning_meta['bin_edges'][var]
        all_centers = (edges[:-1] + edges[1:]) / 2
        bin_indices = _parse_bin_indices(cols, var)
        if (bin_indices is not None and len(bin_indices) == len(gain) and
                np.all((bin_indices >= 0) & (bin_indices < len(all_centers)))):
            bin_centers = all_centers[bin_indices]
        else:
            bin_centers = all_centers[:len(gain)]
        xlabel = f'{var}'
    else:
        # Fallback for old results without bin_edges
        bin_centers = np.arange(len(gain))
        xlabel = f'{var} bin'
        print(f"Warning: No bin_edges found for '{var}'. Using bin indices. "
              "Re-run analysis to get actual covariate values.")

    plt.figure()
    plt.plot(bin_centers, gain, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel('Gain (× baseline)')
    plt.title(f'Tuning: {var}')
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.show()


def plot_angular_tuning(var, beta, tuning_meta):
    cols = _cols_in_beta(tuning_meta['groups'][var], beta)
    if not cols:
        print(f"No {var} coefficients found in beta (all may have been dropped)")
        return
    w = beta[cols].to_numpy()
    gain = np.exp(w)

    # Get bin edges and compute bin centers for angular variable
    if 'bin_edges' in tuning_meta and var in tuning_meta['bin_edges']:
        edges = tuning_meta['bin_edges'][var]
        all_centers = (edges[:-1] + edges[1:]) / 2
        bin_indices = _parse_bin_indices(cols, var)
        if (bin_indices is not None and len(bin_indices) == len(gain) and
                np.all((bin_indices >= 0) & (bin_indices < len(all_centers)))):
            theta = all_centers[bin_indices]
        else:
            theta = all_centers[:len(gain)]
        xlabel = f'{var} (rad)'
    else:
        # Fallback to evenly spaced angles for old results without bin_edges
        theta = np.linspace(-np.pi, np.pi, len(gain))
        xlabel = 'Angle (rad)'
        print(f"Warning: No bin_edges found for '{var}'. Using evenly-spaced angles. "
              "Re-run analysis to get actual covariate values.")

    plt.figure()
    plt.plot(theta, gain, marker='o' if len(gain) <= 20 else None)
    plt.xlabel(xlabel)
    plt.ylabel('Gain (× baseline)')
    plt.title(f'Angular tuning: {var}')
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.show()


def plot_variable(var, beta, structured_meta_groups):
    """
    Automatically plot any variable based on its type.

    Parameters
    ----------
    var : str
        Variable name (e.g., 'v', 'phi', 't_move', 'spike_hist')
    beta : Series or DataFrame
        Fitted coefficients
    structured_meta_groups : dict
        Combined metadata with 'tuning', 'temporal', and 'hist' sub-dicts
    """
    # Check if it's a tuning variable
    if 'tuning' in structured_meta_groups:
        tuning_meta = structured_meta_groups['tuning']
        if var in tuning_meta.get('linear_vars', []):
            plot_linear_tuning(var, beta, tuning_meta)
            return
        elif var in tuning_meta.get('angular_vars', []):
            plot_angular_tuning(var, beta, tuning_meta)
            return

    # Check if it's a temporal/event variable
    if 'temporal' in structured_meta_groups:
        temporal_meta = structured_meta_groups['temporal']
        if var in temporal_meta.get('groups', {}):
            plot_event_kernel(var, beta, temporal_meta)
            return

    # Check if it's spike history
    if 'hist' in structured_meta_groups:
        hist_meta = structured_meta_groups['hist']
        if var == 'spike_hist' and var in hist_meta.get('groups', {}):
            plot_spike_history(beta, hist_meta)
            return
        # Could also be a coupling variable
        elif var.startswith('cpl_') and var in hist_meta.get('groups', {}):
            # Use same plotting as spike history but with different title
            cols = _cols_in_beta(hist_meta['groups'][var], beta)
            if not cols:
                print(f"No {var} coefficients found in beta (all may have been dropped)")
                return
            w = beta[cols].to_numpy()

            # Reconstruct kernel from basis functions
            if 'basis_info' in hist_meta and var in hist_meta['basis_info']:
                info = hist_meta['basis_info'][var]
                lags = info['lags']
                basis = info['basis']
                basis_indices = _parse_basis_indices(cols, var)
                if basis_indices is not None and len(basis_indices) == len(w):
                    valid = (basis_indices >= 0) & (basis_indices < basis.shape[1])
                    B_sub = basis[:, basis_indices[valid]]
                    w_sub = w[valid]
                    kernel = B_sub @ w_sub
                    lags_ms = lags * 1000
                    xlabel = 'Time lag (ms)'
                elif basis.shape[1] == len(w):
                    kernel = basis @ w
                    lags_ms = lags * 1000
                    xlabel = 'Time lag (ms)'
                else:
                    kernel = w
                    lags_ms = np.arange(len(w))
                    xlabel = 'Basis index'
                    print(f"Warning: basis/weights size mismatch for '{var}'; plotting weights directly.")
            else:
                # Fallback: just plot the weights
                kernel = w
                lags_ms = np.arange(len(w))
                xlabel = 'Basis index'
                print(
                    f"Warning: No basis_info found for '{var}'. Plotting weights directly.")

            plt.figure()
            plt.plot(lags_ms, kernel, marker='o' if len(
                kernel) <= 20 else None)
            plt.xlabel(xlabel)
            plt.ylabel('Weight (log-rate change)')
            plt.title(f'Coupling filter: {var}')
            plt.axhline(0.0, color='k', ls='--', lw=1)
            plt.grid(True, alpha=0.3)
            plt.show()
            return

    print(f"Variable '{var}' not found in metadata")


def plot_variables(beta, structured_meta_groups, var_list=None, plot_var_order=None):
    """
    Plot any set of variables with optional ordering.

    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    structured_meta_groups : dict
        Combined metadata
    var_list : list of str, optional
        List of variable names to plot. If None, plots all available variables.
    plot_var_order : list of str, optional
        Order in which to plot variables. Variables not in this list will be
        plotted afterward in the order they appear in var_list. If None, uses
        the order from var_list or the default discovery order.
    """
    # Collect all available variables if var_list not specified
    if var_list is None:
        var_list = []

        # Add tuning variables
        if 'tuning' in structured_meta_groups:
            tuning_meta = structured_meta_groups['tuning']
            var_list.extend(tuning_meta.get('linear_vars', []))
            var_list.extend(tuning_meta.get('angular_vars', []))

        # Add temporal/event variables
        if 'temporal' in structured_meta_groups:
            temporal_meta = structured_meta_groups['temporal']
            var_list.extend(temporal_meta.get('groups', {}).keys())

        # Add history variables
        if 'hist' in structured_meta_groups:
            hist_meta = structured_meta_groups['hist']
            var_list.extend(hist_meta.get('groups', {}).keys())

    # Apply ordering if specified
    if plot_var_order is not None:
        # Variables in specified order first
        ordered_vars = [v for v in plot_var_order if v in var_list]
        # Then variables not in plot_var_order
        remaining_vars = [v for v in var_list if v not in plot_var_order]
        var_list = ordered_vars + remaining_vars

    # Plot each variable
    for var in var_list:
        plot_variable(var, beta, structured_meta_groups)


# def plot_variables_from_groups(beta, meta_groups, var_list=None, plot_var_order=None):
#     """
#     Plot any set of variables with optional ordering.

#     Parameters
#     ----------
#     beta : Series or DataFrame
#         Fitted coefficients
#     structured_meta_groups : dict
#         Combined metadata
#     var_list : list of str, optional
#         List of variable names to plot. If None, plots all available variables.
#     plot_var_order : list of str, optional
#         Order in which to plot variables. Variables not in this list will be
#         plotted afterward in the order they appear in var_list. If None, uses
#         the order from var_list or the default discovery order.
#     """
#     # Collect all available variables if var_list not specified
#     var_list = list(meta_groups.keys())
#     # Apply ordering if specified
#     if plot_var_order is not None:
#         # Variables in specified order first
#         ordered_vars = [v for v in plot_var_order if v in var_list]
#         # Then variables not in plot_var_order
#         remaining_vars = [v for v in var_list if v not in plot_var_order]
#         var_list = ordered_vars + remaining_vars

#     # Plot each variable
#     for var in var_list:
#         plot_variable(var, beta, meta_groups)


def plot_all_tuning_curves(beta, structured_meta_groups):
    """
    Plot all tuning curves (linear and angular variables).

    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    structured_meta_groups : dict
        Combined metadata
    plot_var_order : list of str, optional
        Order in which to plot variables
    """
    if 'tuning' not in structured_meta_groups:
        print("No tuning metadata found")
        return

    tuning_meta = structured_meta_groups['tuning']
    var_list = tuning_meta.get('linear_vars', []) + \
        tuning_meta.get('angular_vars', [])

    plot_variables(beta, structured_meta_groups, var_list=var_list,
                   plot_var_order=one_ff_parameters.plot_var_order)


def plot_all_temporal_filters(beta, structured_meta_groups):
    """
    Plot all temporal filters (events and spike history).

    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    structured_meta_groups : dict
        Combined metadata
    plot_var_order : list of str, optional
        Order in which to plot variables
    """
    var_list = []

    # Add event kernels
    if 'temporal' in structured_meta_groups:
        temporal_meta = structured_meta_groups['temporal']
        event_vars = ['t_targ', 't_move', 't_rew', 't_stop']
        var_list.extend(
            [v for v in event_vars if v in temporal_meta.get('groups', {})])

    # Add spike history
    if 'hist' in structured_meta_groups:
        hist_meta = structured_meta_groups['hist']
        if 'spike_hist' in hist_meta.get('groups', {}):
            var_list.append('spike_hist')

    plot_variables(beta, structured_meta_groups, var_list=var_list,
                   plot_var_order=one_ff_parameters.plot_var_order)


def plot_fold_coefficients(
    cv_result,
    structured_meta_groups,
    use_sem=True,
    plot_all_vars=True,
    label_step=1,
):
    """
    Plot cross-validated tuning curves, one figure per tuning variable.

    Parameters
    ----------
    cv_result : dict
        Result from crossval_stop_tuning_curve_coef with keys:
        'mean_coef', 'std_coef', 'fold_design_columns', 'fold_coef', 'n_folds'
    structured_meta_groups : dict
        Combined metadata with 'tuning', 'temporal', and 'hist' sub-dicts
    use_sem : bool
        If True, use standard error of the mean (std / sqrt(n_folds)).
        If False, use standard deviation.
    plot_all_vars : bool
        Kept for backward compatibility. Not used in this mode.
    label_step : int
        Kept for backward compatibility. Not used in this mode.
    """
    mean_coef = cv_result.get('mean_coef')
    std_coef = cv_result.get('std_coef')
    col_names = cv_result.get('fold_design_columns')
    n_folds = cv_result.get('n_folds', 1)

    if mean_coef is None or std_coef is None or col_names is None:
        raise ValueError('cv_result must contain mean_coef, std_coef, and fold_design_columns')

    mean_coef = np.asarray(mean_coef, dtype=float)
    std_coef = np.asarray(std_coef, dtype=float)
    col_names = list(col_names)

    # Compute uncertainty in log-rate space
    if use_sem:
        error = std_coef / np.sqrt(n_folds)
    else:
        error = std_coef
    if 'tuning' not in structured_meta_groups:
        raise ValueError("structured_meta_groups must contain 'tuning' metadata")

    tuning_meta = structured_meta_groups['tuning']
    linear_vars = tuning_meta.get('linear_vars', [])
    angular_vars = tuning_meta.get('angular_vars', [])
    var_list = list(linear_vars) + list(angular_vars)
    if not var_list:
        raise ValueError("No tuning variables found in structured_meta_groups['tuning']")

    ordered_vars = [v for v in one_ff_parameters.plot_var_order if v in var_list]
    ordered_vars += [v for v in var_list if v not in ordered_vars]

    col_to_idx = {name: idx for idx, name in enumerate(col_names)}
    figs, axes = [], []

    for var in ordered_vars:
        group_cols = tuning_meta.get('groups', {}).get(var, [])
        var_cols = [c for c in group_cols if c in col_to_idx]
        if not var_cols:
            continue

        idx = [col_to_idx[c] for c in var_cols]
        y_log = mean_coef[idx]
        e_log = error[idx]

        gain = np.exp(y_log)
        lower = np.exp(y_log - e_log)
        upper = np.exp(y_log + e_log)
        yerr = np.vstack([gain - lower, upper - gain])

        is_angular = var in angular_vars
        if 'bin_edges' in tuning_meta and var in tuning_meta['bin_edges']:
            edges = np.asarray(tuning_meta['bin_edges'][var], dtype=float)
            centers = (edges[:-1] + edges[1:]) / 2
            bin_indices = _parse_bin_indices(var_cols, var)
            if (bin_indices is not None and len(bin_indices) == len(gain) and
                    np.all((bin_indices >= 0) & (bin_indices < len(centers)))):
                x = centers[bin_indices]
            else:
                x = centers[:len(gain)]
        else:
            if is_angular:
                x = np.linspace(-np.pi, np.pi, len(gain))
            else:
                x = np.arange(len(gain))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x, gain, yerr=yerr, fmt='o-' if len(gain) <= 20 else '-',
            capsize=4, linewidth=2, markersize=5,
            label='Mean ± SEM' if use_sem else 'Mean ± Std'
        )
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Tuning: {var} (n_folds={n_folds})')
        ax.set_xlabel(f'{var} (rad)' if is_angular else var)
        ax.set_ylabel('Gain (× baseline)')
        ax.legend()
        fig.tight_layout()
        plt.show()

        figs.append(fig)
        axes.append(ax)


def plot_fold_coefficient_by_variable(cv_result, structured_meta_groups, var_list=None, use_sem=True):
    """
    Plot cross-validation fold coefficients grouped by variable type.

    Decomposes coefficients by variable type (tuning, temporal, history) and plots
    each variable's coefficients with error bars on separate panels.

    Parameters
    ----------
    cv_result : dict
        Result from crossval_stop_tuning_curve_coef with keys:
        'mean_coef', 'std_coef', 'fold_design_columns', 'fold_coef', 'n_folds'
    structured_meta_groups : dict
        Combined metadata with 'tuning', 'temporal', and 'hist' sub-dicts
    var_list : list of str, optional
        List of specific variables to plot. If None, plots all available variables.
    use_sem : bool
        If True, use standard error of the mean. If False, use standard deviation.
    """
    mean_coef = cv_result.get('mean_coef')
    std_coef = cv_result.get('std_coef')
    col_names = cv_result.get('fold_design_columns')
    n_folds = cv_result.get('n_folds', 1)

    if mean_coef is None or std_coef is None or col_names is None:
        raise ValueError('cv_result must contain mean_coef, std_coef, and fold_design_columns')

    # Build variable mapping from column names
    var_col_map = {}

    # Map tuning variables
    if 'tuning' in structured_meta_groups:
        tuning_meta = structured_meta_groups['tuning']
        for var in tuning_meta.get('linear_vars', []) + tuning_meta.get('angular_vars', []):
            var_col_map[var] = _cols_in_beta(
                tuning_meta['groups'].get(var, []),
                col_names
            )

    # Map temporal variables
    if 'temporal' in structured_meta_groups:
        temporal_meta = structured_meta_groups['temporal']
        for var in temporal_meta.get('groups', {}).keys():
            var_col_map[var] = _cols_in_beta(
                temporal_meta['groups'].get(var, []),
                col_names
            )

    # Map history variables
    if 'hist' in structured_meta_groups:
        hist_meta = structured_meta_groups['hist']
        for var in hist_meta.get('groups', {}).keys():
            var_col_map[var] = _cols_in_beta(
                hist_meta['groups'].get(var, []),
                col_names
            )

    # Determine which variables to plot
    if var_list is None:
        var_list = list(var_col_map.keys())
    else:
        var_list = [v for v in var_list if v in var_col_map]

    # Compute error bars
    if use_sem:
        error = std_coef / np.sqrt(n_folds)
    else:
        error = std_coef

    # Create subplots
    n_vars = len(var_list)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, var in enumerate(var_list):
        ax = axes[idx]
        col_indices = []

        # Find column indices for this variable
        for col_idx, col_name in enumerate(col_names):
            if col_name in var_col_map[var]:
                col_indices.append(col_idx)

        if not col_indices:
            ax.text(0.5, 0.5, f'{var}\n(no coefficients)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var)
            continue

        # Extract coefficients for this variable
        x_pos = np.arange(len(col_indices))
        var_mean = mean_coef[col_indices]
        var_error = error[col_indices]
        var_cols = [col_names[i] for i in col_indices]

        # Plot
        ax.errorbar(x_pos, var_mean, yerr=var_error, fmt='o-', capsize=5, capthick=2,
                    markersize=6, linewidth=2)
        ax.axhline(0.0, color='k', ls='--', lw=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{var}')
        ax.set_ylabel('Coefficient Value')

        # Set x-tick labels
        if len(var_cols) <= 10:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(var_cols, rotation=45, ha='right', fontsize=8)
        else:
            step = max(1, len(var_cols) // 5)
            ax.set_xticks(x_pos[::step])
            ax.set_xticklabels([var_cols[i] for i in x_pos[::step]],
                               rotation=45, ha='right', fontsize=8)

    # Hide unused axes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Cross-validation Tuning Curve by Variable (n_folds={n_folds}, SEM={use_sem})',
                 fontsize=14, y=1.00)
    fig.tight_layout()
    plt.show()

