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
    print_vars=True,
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
    print_vars : bool
        If True, print the variable/column names in each category group.

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

    if print_vars and 'vars' in df.columns:
        print("Variables per category:")
        for _, row in df.iterrows():
            cat = row['category']
            vars_ = row['vars']
            if isinstance(vars_, (list, tuple)):
                vars_str = ', '.join(str(v) for v in vars_)
            else:
                vars_str = str(vars_)
            print(f"  {cat}: [{vars_str}]")

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


def add_clipped_leave_delta_metric(
    all_results,
    *,
    full_key='mean_pseudo_r2',
    leave_key='leave_out_mean_pseudo_r2',
    new_key='delta_pseudo_r2_clip_leave',
):
    """
    Add clipped delta metric to all_results:

        delta = full - max(0, leave_out)

    Adds new_key inside each category_contributions entry.

    Returns
    -------
    all_results (modified in-place)
    """

    for res in all_results:

        if 'full_cv_result' not in res:
            continue

        full_val = res['full_cv_result'].get(full_key, None)
        if full_val is None:
            continue

        cc = res.get('category_contributions', {})

        for cat_name, cat_dict in cc.items():

            leave_val = cat_dict.get(leave_key, None)
            if leave_val is None:
                continue

            clipped_leave = max(0.0, float(leave_val))
            delta_clip = float(full_val) - clipped_leave

            cat_dict[new_key] = delta_clip

    return all_results


def plot_category_ecdf(
    all_results,
    *,
    metric_key='delta_pseudo_r2',
    include_total=True,
    category_groups=None,
    log_x=True,
):
    """
    General ECDF plot for category contributions.

    Parameters
    ----------
    all_results : list of dict
        Output from runner.run_category_variance_contributions(...)

    metric_key : str
        Which metric inside category_contributions to plot
        (e.g. 'delta_pseudo_r2', 'delta_classical_r2').

    include_total : bool
        If True, include full model mean_pseudo_r2 as 'Total'.

    category_groups : dict or None
        Optional mapping:
            { 'Legend Name': ['cat1', 'cat2', ...], ... }

        If None, each category is plotted separately.

    log_x : bool
        Use log-scaled x-axis.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # 1) Collect all categories present across neurons
    # --------------------------------------------------

    all_categories = set()
    for res in all_results:
        if 'category_contributions' in res:
            all_categories.update(res['category_contributions'].keys())

    all_categories = sorted(all_categories)

    # --------------------------------------------------
    # 2) Initialize storage
    # --------------------------------------------------

    # If grouping requested
    if category_groups is not None:
        data_dict = {group_name: [] for group_name in category_groups}
    else:
        data_dict = {cat: [] for cat in all_categories}

    total_vals = []

    # --------------------------------------------------
    # 3) Collect values
    # --------------------------------------------------

    for res in all_results:

        if 'full_cv_result' not in res:
            continue

        # Full model
        if include_total:
            total_vals.append(res['full_cv_result']['mean_pseudo_r2'])

        cc = res.get('category_contributions', {})

        if category_groups is None:
            # plot each category separately
            for cat in all_categories:
                if cat in cc and metric_key in cc[cat]:
                    data_dict[cat].append(cc[cat][metric_key])
        else:
            # grouped categories
            for group_name, cat_list in category_groups.items():
                vals = [
                    cc[cat][metric_key]
                    for cat in cat_list
                    if cat in cc and metric_key in cc[cat]
                ]
                if len(vals) > 0:
                    data_dict[group_name].append(np.sum(vals))

    # --------------------------------------------------
    # 4) Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=180)

    if log_x:
        ax.axvspan(1e-3, 1e-2, color='0.9', zorder=0)
        ax.axvspan(1e-2, 1e-1, color='0.95', zorder=0)

    # Automatic color cycle
    for label, vals in data_dict.items():

        if len(vals) == 0:
            continue

        x, y = _ecdf(vals, clip_for_log=log_x)
        ax.plot(x, y, lw=1.6, label=label)

    # Plot total last (black)
    if include_total and len(total_vals) > 0:
        x, y = _ecdf(total_vals, clip_for_log=log_x)
        ax.plot(x, y, lw=2.0, color='black', label='Total')

    # Axis formatting
    if log_x:
        ax.set_xscale('log')
        ax.set_xlim(1e-3, 1.0)
    else:
        all_vals = np.concatenate(
            [np.asarray(v, dtype=float) for v in data_dict.values() if len(v) > 0]
        )
        if include_total and len(total_vals) > 0:
            all_vals = np.concatenate([all_vals, np.asarray(total_vals)])

        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            x_max = float(np.max(finite_vals))
            pad = 0.05 * max(1e-9, x_max)
            ax.set_xlim(0.0, x_max + pad)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Variance explained')
    ax.set_ylabel('Cumulative prob.')
    ax.legend(frameon=False)
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


def plot_tuning_heatmaps(
    units_data,
    var_list=None,
    *,
    tuned_criterion='gain_range',
    gain_range_min=1.5,
    skip_2d=True,
    x_grid_n=80,
    figsize=(7, 5),
    save_dir=None,
    varlab_map=None,
):
    """
    Plot peak-normalized tuning heatmaps per variable (MATLAB PlotSessions style).

    For each variable: top panel = mean ± SE of raw tuning curves; bottom panel =
    heatmap of peak-normalized curves, neurons sorted by peak feature.

    Parameters
    ----------
    units_data : list of dict
        Per-neuron data with keys:
        - unit_idx : int
        - beta : pd.Series or np.ndarray (coefficients)
        - structured_meta_groups : dict
        - col_names : list (design column names, for beta alignment)
    var_list : list of str, optional
        Variables to plot. If None, uses all tuning + event vars from metadata.
    tuned_criterion : str
        'gain_range' = tuned if max(gain)/min(gain) >= gain_range_min
        'always' = include all neurons with non-empty curve
    gain_range_min : float
        Minimum gain range for tuned_criterion='gain_range'.
    skip_2d : bool
        Skip 2D covariates (no heatmap for them).
    x_grid_n : int
        Number of points for common x-grid interpolation.
    figsize : tuple
        Figure size per variable.
    save_dir : str or Path, optional
        If set, save figures to this directory.
    varlab_map : dict, optional
        Map variable names to display labels.
    """
    from scipy.interpolate import interp1d

    if not units_data:
        print("No units data provided")
        return

    meta0 = units_data[0].get("structured_meta_groups", {})
    if var_list is None:
        var_list = []
        if "tuning" in meta0:
            tm = meta0["tuning"]
            var_list.extend(tm.get("linear_vars", []))
            var_list.extend(tm.get("angular_vars", []))
        if "temporal" in meta0:
            var_list.extend(meta0["temporal"].get("groups", {}).keys())

    for var in var_list:
        vartype = "1D"
        if "temporal" in meta0 and var in meta0["temporal"].get("groups", {}):
            vartype = "event"
        elif "tuning" in meta0:
            if var in meta0["tuning"].get("angular_vars", []):
                vartype = "1D"
            elif var in meta0["tuning"].get("linear_vars", []):
                vartype = "1D"
        if skip_2d and vartype == "2D":
            continue

        stim_list = []
        rate_list = []
        unit_idx_list = []

        for ud in units_data:
            unit_idx = ud.get("unit_idx", len(stim_list))
            beta = ud.get("beta")
            if beta is None:
                continue
            if hasattr(beta, "index"):
                beta = beta.reindex(ud.get("col_names", []))
                beta = beta.dropna()
            else:
                col_names = ud.get("col_names", [])
                if col_names and len(beta) != len(col_names):
                    continue

            smg = ud.get("structured_meta_groups", {})
            stim = None
            rate = None

            if "tuning" in smg and var in smg["tuning"].get("groups", {}):
                cols = _cols_in_beta(smg["tuning"]["groups"][var], beta)
                if not cols:
                    continue
                w = beta[cols].to_numpy() if hasattr(beta, "__getitem__") else np.asarray(beta)
                rate = np.exp(w)
                if "bin_edges" in smg["tuning"] and var in smg["tuning"]["bin_edges"]:
                    edges = np.asarray(smg["tuning"]["bin_edges"][var])
                    centers = (edges[:-1] + edges[1:]) / 2
                    bin_idx = _parse_bin_indices(cols, var)
                    if bin_idx is not None and len(bin_idx) == len(rate):
                        stim = centers[bin_idx]
                    else:
                        stim = centers[: len(rate)]
                else:
                    stim = np.arange(len(rate))

            elif "temporal" in smg and var in smg["temporal"].get("groups", {}):
                cols = _cols_in_beta(smg["temporal"]["groups"][var], beta)
                if not cols:
                    continue
                w = beta[cols].to_numpy() if hasattr(beta, "__getitem__") else np.asarray(beta)
                if "basis_info" in smg["temporal"] and var in smg["temporal"]["basis_info"]:
                    info = smg["temporal"]["basis_info"][var]
                    basis = info["basis"]
                    lags = info["lags"]
                    bi = _parse_basis_indices(cols, var)
                    if bi is not None and len(bi) == len(w):
                        valid = (bi >= 0) & (bi < basis.shape[1])
                        kernel = basis[:, bi[valid]] @ w[valid]
                    elif basis.shape[1] == len(w):
                        kernel = basis @ w
                    else:
                        kernel = w
                    stim = lags
                    rate = np.exp(kernel)
                else:
                    stim = np.arange(len(w))
                    rate = np.exp(w)

            if stim is None or rate is None or len(stim) < 2 or len(rate) < 2:
                continue

            stim = np.asarray(stim).ravel()
            rate = np.asarray(rate).ravel()
            if len(stim) != len(rate):
                continue

            if tuned_criterion == "gain_range":
                g_min, g_max = np.nanmin(rate), np.nanmax(rate)
                if g_min <= 0 or g_max / (g_min + 1e-12) < gain_range_min:
                    continue
            stim_list.append(stim)
            rate_list.append(rate)
            unit_idx_list.append(unit_idx)

        if not rate_list:
            continue

        x_all = np.unique(np.concatenate([s.ravel() for s in stim_list]))
        x_all = np.sort(x_all[~np.isnan(x_all)])
        if len(x_all) < 2:
            x_min = min(np.nanmin(s) for s in stim_list)
            x_max = max(np.nanmax(s) for s in stim_list)
            x_grid = np.linspace(x_min, x_max, x_grid_n)
        else:
            x_grid = np.linspace(x_all.min(), x_all.max(), x_grid_n)

        n_curves = len(rate_list)
        curves = np.full((n_curves, len(x_grid)), np.nan)
        curves_raw = np.full((n_curves, len(x_grid)), np.nan)
        peak_feat = np.full(n_curves, np.nan)

        for k in range(n_curves):
            xk = stim_list[k]
            rk = rate_list[k]
            f = interp1d(xk, rk, kind="linear", bounds_error=False, fill_value="extrap")
            rk_interp = f(x_grid)
            rk_max = np.nanmax(rk_interp)
            curves_raw[k] = rk_interp
            if rk_max > 0:
                curves[k] = rk_interp / rk_max
                pk = np.nanargmax(rk_interp)
                peak_feat[k] = x_grid[pk]
            else:
                curves[k] = rk_interp
                peak_feat[k] = x_grid[0]
            peak_feat[k] = peak_feat[k] if np.isfinite(peak_feat[k]) else x_grid[0]

        sort_idx = np.argsort(peak_feat)
        curves_sorted = curves[sort_idx]
        curves_raw_sorted = curves_raw[sort_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        mu = np.nanmean(curves_raw_sorted, axis=0)
        se = np.nanstd(curves_raw_sorted, axis=0) / np.sqrt(n_curves)
        ax1.fill_between(x_grid, mu - se, mu + se, alpha=0.3, color=[0.2, 0.5, 0.8])
        ax1.plot(x_grid, mu, "-", color=[0.2, 0.5, 0.8], lw=2)
        if vartype == "event":
            ax1.axvline(0, color="k", lw=1)
        ax1.set_ylabel("Gain (× baseline)")
        ax1.set_xticklabels([])
        ax1.tick_params(axis="both", labelsize=10)
        ax1.set_xlim(x_grid.min(), x_grid.max())

        im = ax2.imshow(
            curves_sorted,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
            extent=[x_grid[0], x_grid[-1], n_curves - 0.5, -0.5],
            origin="upper",
        )
        if vartype == "event":
            ax2.axvline(0, color="w", lw=1)
        varlab_map = varlab_map or {}
        ax2.set_xlabel(varlab_map.get(var, var))
        ax2.set_ylabel("Neurons")
        plt.colorbar(im, ax=ax2, label="Normalized")
        fig.suptitle(
            f"Tuning to {varlab_map.get(var, var)} "
            f"(peak-norm, sorted by peak, n={n_curves})"
        )
        fig.tight_layout()

        if save_dir:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fpath = Path(save_dir) / f"GAM_tuning_heatmap_{var}.png"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.show()

