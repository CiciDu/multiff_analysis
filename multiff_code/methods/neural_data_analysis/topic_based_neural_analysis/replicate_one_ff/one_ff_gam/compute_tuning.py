import numpy as np
import scipy.stats as sts

# ------------------------------------------------------------
# Added tuning fraction calculations
# ------------------------------------------------------------
def calculate_tuning_fraction(
    results_list,
    pval_threshold=0.05,
    use_reduced=False,
    min_firing_rate=0.1,
):
    """
    Calculate the fraction of neurons significantly tuned for each variable.
    
    Parameters
    ----------
    results_list : list of structured arrays
        List of results from postprocess_results, one per neuron.
    pval_threshold : float, optional
        P-value threshold for significance (default: 0.05).
    use_reduced : bool, optional
        If True, use reduced model p-values; otherwise use full model (default: False).
    min_firing_rate : float, optional
        Minimum firing rate (Hz) to include a neuron (default: 0.1).
    
    Returns
    -------
    tuning_stats : dict
        Dictionary with keys:
        - 'variables': list of variable names
        - 'fraction_tuned': array of fraction tuned for each variable
        - 'n_tuned': array of number of tuned neurons for each variable
        - 'n_total': total number of neurons analyzed
        - 'neuron_tuning_matrix': boolean array (n_neurons x n_variables)
    """
    if len(results_list) == 0:
        raise ValueError("results_list is empty")
    
    # Get variable names from first result
    variables = results_list[0]['variable']
    n_vars = len(variables)
    n_neurons = len(results_list)
    
    # Initialize boolean matrix (neurons x variables)
    neuron_tuning_matrix = np.zeros((n_neurons, n_vars), dtype=bool)
    valid_neurons = np.zeros(n_neurons, dtype=bool)
    
    # Determine which neurons meet firing rate criterion
    for i, result in enumerate(results_list):
        if result['fr'][0] >= min_firing_rate:
            valid_neurons[i] = True
    
    # Fill in tuning matrix
    pval_key = 'reduced_pval' if use_reduced else 'pval'
    
    for i, result in enumerate(results_list):
        if not valid_neurons[i]:
            continue
        
        for j, var in enumerate(variables):
            pval = result[pval_key][j]
            if not np.isnan(pval) and pval < pval_threshold:
                neuron_tuning_matrix[i, j] = True
    
    # Calculate statistics
    n_valid = valid_neurons.sum()
    n_tuned = neuron_tuning_matrix[valid_neurons].sum(axis=0)
    fraction_tuned = n_tuned / n_valid if n_valid > 0 else np.zeros(n_vars)
    
    tuning_stats = {
        'variables': list(variables),
        'fraction_tuned': fraction_tuned,
        'n_tuned': n_tuned,
        'n_total': n_valid,
        'neuron_tuning_matrix': neuron_tuning_matrix[valid_neurons],
        'pval_threshold': pval_threshold,
        'min_firing_rate': min_firing_rate,
        'use_reduced': use_reduced,
    }
    
    return tuning_stats


def plot_tuning_fraction(
    tuning_stats,
    figsize=(10, 6),
    color='steelblue',
    xlabel='Variable',
    ylabel='Fraction of Neurons Tuned',
    title=None,
    rotate_labels=45,
    show_counts=True,
    ax=None,
):
    """
    Plot the fraction of neurons tuned for each variable as a bar chart.
    
    Parameters
    ----------
    tuning_stats : dict
        Output from calculate_tuning_fraction.
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    color : str or array-like, optional
        Bar color(s) (default: 'steelblue').
    xlabel : str, optional
        X-axis label (default: 'Variable').
    ylabel : str, optional
        Y-axis label (default: 'Fraction of Neurons Tuned').
    title : str, optional
        Plot title. If None, auto-generates title with threshold info.
    rotate_labels : float, optional
        Rotation angle for x-axis labels (default: 45).
    show_counts : bool, optional
        If True, show counts on top of bars (default: True).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    import matplotlib.pyplot as plt
    
    variables = tuning_stats['variables']
    fraction_tuned = tuning_stats['fraction_tuned']
    n_tuned = tuning_stats['n_tuned']
    n_total = tuning_stats['n_total']
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create bar plot
    x_pos = np.arange(len(variables))
    bars = ax.bar(x_pos, fraction_tuned, color=color, alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variables, rotation=rotate_labels, ha='right')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add title
    if title is None:
        pval_thresh = tuning_stats['pval_threshold']
        model_type = 'reduced' if tuning_stats['use_reduced'] else 'full'
        title = (f'Fraction of Neurons Tuned (n={n_total}, '
                f'p<{pval_thresh}, {model_type} model)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add counts on top of bars
    if show_counts:
        for i, (bar, count) in enumerate(zip(bars, n_tuned)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{int(count)}/{n_total}',
                ha='center',
                va='bottom',
                fontsize=9,
            )
    
    plt.tight_layout()
    
    return fig, ax


def plot_tuning_heatmap(
    tuning_stats,
    figsize=(12, 8),
    cmap='YlOrRd',
    sort_by='total',
    xlabel='Variable',
    ylabel='Neuron',
    title=None,
    show_colorbar=True,
    ax=None,
):
    """
    Plot a heatmap showing which neurons are tuned to which variables.
    
    Parameters
    ----------
    tuning_stats : dict
        Output from calculate_tuning_fraction.
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8)).
    cmap : str, optional
        Colormap name (default: 'YlOrRd').
    sort_by : str, optional
        How to sort neurons: 'total' (by total number of tuned variables),
        'first' (by first tuned variable), or None (default: 'total').
    xlabel : str, optional
        X-axis label (default: 'Variable').
    ylabel : str, optional
        Y-axis label (default: 'Neuron').
    title : str, optional
        Plot title. If None, auto-generates title.
    show_colorbar : bool, optional
        Whether to show colorbar (default: True).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    import matplotlib.pyplot as plt
    
    variables = tuning_stats['variables']
    neuron_tuning_matrix = tuning_stats['neuron_tuning_matrix']
    n_total = tuning_stats['n_total']
    
    # Sort neurons if requested
    if sort_by == 'total':
        # Sort by total number of variables each neuron is tuned to
        sort_idx = np.argsort(neuron_tuning_matrix.sum(axis=1))[::-1]
        neuron_tuning_matrix = neuron_tuning_matrix[sort_idx]
    elif sort_by == 'first':
        # Sort by first variable each neuron is tuned to
        first_tuned = np.argmax(neuron_tuning_matrix, axis=1)
        sort_idx = np.argsort(first_tuned)
        neuron_tuning_matrix = neuron_tuning_matrix[sort_idx]
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap
    im = ax.imshow(
        neuron_tuning_matrix.astype(float),
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
    )
    
    # Customize plot
    ax.set_xticks(np.arange(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Add title
    if title is None:
        pval_thresh = tuning_stats['pval_threshold']
        model_type = 'reduced' if tuning_stats['use_reduced'] else 'full'
        title = (f'Neuron Tuning Matrix (n={n_total}, '
                f'p<{pval_thresh}, {model_type} model)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Tuned', fontsize=11, fontweight='bold')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt


def plot_fraction_tuned(tuning_stats):
    """
    Plot fraction of tuned neurons for each variable.

    Parameters
    ----------
    tuning_stats : dict
        Dictionary containing:
            - 'variables'
            - 'fraction_tuned'
            - 'n_total'
    """

    variables = tuning_stats['variables']
    fraction_tuned = np.array(tuning_stats['fraction_tuned'])
    n_total = tuning_stats['n_total']

    # Binomial standard error
    sem = np.sqrt(fraction_tuned * (1 - fraction_tuned) / n_total)

    # Clean variable labels (remove f_ / g_ prefixes)
    def clean_label(v):
        if v.startswith('f_') or v.startswith('g_'):
            return v[2:]
        return v

    x_labels = [clean_label(v) for v in variables]
    x = np.arange(len(x_labels))

    # Plot
    plt.figure(figsize=(10, 5))

    plt.errorbar(
        x,
        fraction_tuned,
        yerr=sem,
        fmt='o',
        capsize=3
    )

    plt.xticks(x, x_labels)
    plt.ylim(0, 1)
    plt.ylabel('Fraction of tuned neurons')
    plt.xlabel('Task variable')

    plt.tight_layout()
    plt.show()