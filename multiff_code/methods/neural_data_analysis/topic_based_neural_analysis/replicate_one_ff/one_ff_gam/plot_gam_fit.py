import re
import numpy as np
import matplotlib.pyplot as plt


def _cols_in_beta(cols, beta):
    """
    Filter column names to only those present in beta.
    When NA rows are dropped during fitting, some design columns may be
    absent from the fitted coefficients.
    """
    beta_index = beta.index if hasattr(beta, 'index') else getattr(beta, 'columns', None)
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


def plot_all_temporal_filters(beta, structured_meta_groups, plot_var_order=None):
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
