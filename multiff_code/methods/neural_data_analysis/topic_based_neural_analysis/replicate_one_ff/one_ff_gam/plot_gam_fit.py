import numpy as np
import matplotlib.pyplot as plt

plot_var_order = [
    # egocentric / motion state
    'v',
    'w',
    'd',
    'phi',

    # target-related
    'r_targ',
    'theta_targ',

    # eye position
    'eye_ver',
    'eye_hor',
    

    # behavior / action
    't_move',
    't_targ',
    't_stop',
    't_rew',

    # history
    'spike_hist',
]

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
    
    cols = hist_meta['groups']['spike_hist']
    w = beta[cols].to_numpy()
    
    # Reconstruct kernel from basis functions
    if 'basis_info' in hist_meta and 'spike_hist' in hist_meta['basis_info']:
        info = hist_meta['basis_info']['spike_hist']
        lags = info['lags']
        basis = info['basis']
        # Reconstruct: kernel = basis @ weights
        kernel = basis @ w
        lags_ms = lags * 1000  # Convert to milliseconds
        xlabel = 'Time lag (ms)'
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
    
    cols = temporal_meta['groups'][var]
    w = beta[cols].to_numpy()
    
    # Reconstruct kernel from basis functions
    if 'basis_info' in temporal_meta and var in temporal_meta['basis_info']:
        info = temporal_meta['basis_info'][var]
        lags = info['lags']
        basis = info['basis']
        # Reconstruct: kernel = basis @ weights
        kernel = basis @ w
        lags_ms = lags * 1000  # Convert to milliseconds
        xlabel = 'Time relative to event (ms)'
    else:
        # Fallback: just plot the weights
        kernel = w
        lags_ms = np.arange(len(w))
        xlabel = 'Basis index'
        print(f"Warning: No basis_info found for '{var}'. Plotting weights directly.")
    
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


def plot_linear_tuning(var, beta, tuning_meta):
    cols = tuning_meta['groups'][var]
    w = beta[cols].to_numpy()
    gain = np.exp(w)
    
    # Get bin edges and compute bin centers
    if 'bin_edges' in tuning_meta and var in tuning_meta['bin_edges']:
        edges = tuning_meta['bin_edges'][var]
        bin_centers = (edges[:-1] + edges[1:]) / 2
        # Handle case where last bin might be dropped
        bin_centers = bin_centers[:len(gain)]
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
    cols = tuning_meta['groups'][var]
    w = beta[cols].to_numpy()
    gain = np.exp(w)
    
    # Get bin edges and compute bin centers for angular variable
    if 'bin_edges' in tuning_meta and var in tuning_meta['bin_edges']:
        edges = tuning_meta['bin_edges'][var]
        bin_centers = (edges[:-1] + edges[1:]) / 2
        # Handle case where last bin might be dropped
        theta = bin_centers[:len(gain)]
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


def plot_variable(var, beta, all_meta):
    """
    Automatically plot any variable based on its type.
    
    Parameters
    ----------
    var : str
        Variable name (e.g., 'v', 'phi', 't_move', 'spike_hist')
    beta : Series or DataFrame
        Fitted coefficients
    all_meta : dict
        Combined metadata with 'tuning', 'temporal', and 'hist' sub-dicts
    """
    # Check if it's a tuning variable
    if 'tuning' in all_meta:
        tuning_meta = all_meta['tuning']
        if var in tuning_meta.get('linear_vars', []):
            plot_linear_tuning(var, beta, tuning_meta)
            return
        elif var in tuning_meta.get('angular_vars', []):
            plot_angular_tuning(var, beta, tuning_meta)
            return
    
    # Check if it's a temporal/event variable
    if 'temporal' in all_meta:
        temporal_meta = all_meta['temporal']
        if var in temporal_meta.get('groups', {}):
            plot_event_kernel(var, beta, temporal_meta)
            return
    
    # Check if it's spike history
    if 'hist' in all_meta:
        hist_meta = all_meta['hist']
        if var == 'spike_hist' and var in hist_meta.get('groups', {}):
            plot_spike_history(beta, hist_meta)
            return
        # Could also be a coupling variable
        elif var.startswith('cpl_') and var in hist_meta.get('groups', {}):
            # Use same plotting as spike history but with different title
            cols = hist_meta['groups'][var]
            w = beta[cols].to_numpy()
            
            # Reconstruct kernel from basis functions
            if 'basis_info' in hist_meta and var in hist_meta['basis_info']:
                info = hist_meta['basis_info'][var]
                lags = info['lags']
                basis = info['basis']
                # Reconstruct: kernel = basis @ weights
                kernel = basis @ w
                lags_ms = lags * 1000
                xlabel = 'Time lag (ms)'
            else:
                # Fallback: just plot the weights
                kernel = w
                lags_ms = np.arange(len(w))
                xlabel = 'Basis index'
                print(f"Warning: No basis_info found for '{var}'. Plotting weights directly.")
            
            plt.figure()
            plt.plot(lags_ms, kernel, marker='o' if len(kernel) <= 20 else None)
            plt.xlabel(xlabel)
            plt.ylabel('Weight (log-rate change)')
            plt.title(f'Coupling filter: {var}')
            plt.axhline(0.0, color='k', ls='--', lw=1)
            plt.grid(True, alpha=0.3)
            plt.show()
            return
    
    print(f"Variable '{var}' not found in metadata")


def plot_variables(beta, all_meta, var_list=None, plot_var_order=None):
    """
    Plot any set of variables with optional ordering.
    
    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    all_meta : dict
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
        if 'tuning' in all_meta:
            tuning_meta = all_meta['tuning']
            var_list.extend(tuning_meta.get('linear_vars', []))
            var_list.extend(tuning_meta.get('angular_vars', []))
        
        # Add temporal/event variables
        if 'temporal' in all_meta:
            temporal_meta = all_meta['temporal']
            var_list.extend(temporal_meta.get('groups', {}).keys())
        
        # Add history variables
        if 'hist' in all_meta:
            hist_meta = all_meta['hist']
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
        plot_variable(var, beta, all_meta)



def plot_all_tuning_curves(beta, all_meta):
    """
    Plot all tuning curves (linear and angular variables).
    
    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    all_meta : dict
        Combined metadata
    plot_var_order : list of str, optional
        Order in which to plot variables
    """
    if 'tuning' not in all_meta:
        print("No tuning metadata found")
        return
    
    tuning_meta = all_meta['tuning']
    var_list = tuning_meta.get('linear_vars', []) + tuning_meta.get('angular_vars', [])
    
    plot_variables(beta, all_meta, var_list=var_list, plot_var_order=plot_var_order)


def plot_all_temporal_filters(beta, all_meta, plot_var_order=None):
    """
    Plot all temporal filters (events and spike history).
    
    Parameters
    ----------
    beta : Series or DataFrame
        Fitted coefficients
    all_meta : dict
        Combined metadata
    plot_var_order : list of str, optional
        Order in which to plot variables
    """
    var_list = []
    
    # Add event kernels
    if 'temporal' in all_meta:
        temporal_meta = all_meta['temporal']
        event_vars = ['t_targ', 't_move', 't_rew', 't_stop']
        var_list.extend([v for v in event_vars if v in temporal_meta.get('groups', {})])
    
    # Add spike history
    if 'hist' in all_meta:
        hist_meta = all_meta['hist']
        if 'spike_hist' in hist_meta.get('groups', {}):
            var_list.append('spike_hist')
    
    plot_variables(beta, all_meta, var_list=var_list, plot_var_order=plot_var_order)