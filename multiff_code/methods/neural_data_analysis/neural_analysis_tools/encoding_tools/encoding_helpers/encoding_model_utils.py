import numpy as np


def _extract_pgam_result_struct(result) -> np.ndarray:
    """
    Normalize PGAM result containers to a structured array.

    Accepts either:
    - direct structured array from postprocess_results(...)
    - dict containing a structured array under 'res' or 'results_struct'
    """
    if isinstance(result, dict):
        if "res" in result:
            result = result["res"]
        elif "results_struct" in result:
            result = result["results_struct"]

    if not isinstance(result, np.ndarray) or result.dtype.names is None:
        raise TypeError(
            "Each item in results_list must be a structured numpy array "
            "(or dict containing one under 'res' / 'results_struct')."
        )
    return result



def plot_fraction_tuned(
    tuning_stats,
    figsize=None,
    capsize: float = 3.0,
    vars_per_subplot: int = 20,
    label_rotation: float = 45.0,
):
    """
    Plot fraction of tuned neurons for each variable with binomial SEM bars.

    Parameters
    ----------
    tuning_stats : dict
        Output dict from calculate_pgam_tuning_fraction(...).
    figsize : tuple | None
        Matplotlib figure size. If None, auto-scales with subplot count.
    capsize : float
        Errorbar cap size.
    vars_per_subplot : int
        Maximum number of variables displayed in each subplot panel.
    label_rotation : float
        Rotation angle for x-axis labels in degrees.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and axes array (single-axis case returns length-1 array).
    """
    import matplotlib.pyplot as plt

    variables = tuning_stats["variables"]
    fraction_tuned = np.asarray(tuning_stats["fraction_tuned"], dtype=float)
    n_total = int(tuning_stats["n_total"])

    if n_total <= 0:
        raise ValueError("No valid neurons to plot (n_total == 0).")

    # Binomial standard error
    sem = np.sqrt(fraction_tuned * (1.0 - fraction_tuned) / n_total)

    def _clean_label(var_name: str) -> str:
        if var_name.startswith("f_") or var_name.startswith("g_"):
            return var_name[2:]
        return var_name

    if vars_per_subplot <= 0:
        raise ValueError("vars_per_subplot must be a positive integer.")

    x_labels = [_clean_label(v) for v in variables]
    n_vars = len(x_labels)
    n_panels = max(1, int(np.ceil(n_vars / vars_per_subplot)))

    if figsize is None:
        figsize = (12, 3.8 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for panel_idx, ax in enumerate(axes):
        start = panel_idx * vars_per_subplot
        stop = min((panel_idx + 1) * vars_per_subplot, n_vars)

        panel_labels = x_labels[start:stop]
        panel_frac = fraction_tuned[start:stop]
        panel_sem = sem[start:stop]
        x = np.arange(len(panel_labels))

        ax.errorbar(x, panel_frac, yerr=panel_sem, fmt="o", capsize=capsize)
        ax.set_xticks(x)
        ax.set_xticklabels(panel_labels, rotation=label_rotation, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction tuned")
        ax.set_xlabel("Task variable")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_title(f"Variables {start + 1}-{stop} / {n_vars}")

    fig.tight_layout()

    return fig, axes


def calculate_tuning_fraction_from_backward_elimination(
    elimination_results_list,
    variable_names=None,
):
    """
    Calculate tuned-neuron fractions from backward elimination outputs.

    A neuron is counted as tuned for a variable if that variable appears in
    the neuron's ``kept_groups`` result.

    Parameters
    ----------
    elimination_results_list : list[dict]
        List of per-neuron dicts returned by
        ``runner.run_backward_elimination(...)``. Each dict should contain
        ``kept_groups`` with GroupSpec-like items (dataclass or dict with name).
    variable_names : list[str] | None
        Optional fixed variable order/subset. If None, the union of all kept
        group names across neurons is used (sorted).

    Returns
    -------
    dict
        Keys:
        - 'variables'
        - 'fraction_tuned'
        - 'n_tuned'
        - 'n_total'
        - 'neuron_tuning_matrix'
        - 'source'
    """
    if len(elimination_results_list) == 0:
        raise ValueError("elimination_results_list is empty")

    def _group_name(group_item):
        if hasattr(group_item, "name"):
            return str(group_item.name)
        if isinstance(group_item, dict) and "name" in group_item:
            return str(group_item["name"])
        raise TypeError(
            "Each kept group must be a GroupSpec-like object with .name "
            "or a dict containing key 'name'."
        )

    neuron_kept_names = []
    for i, res in enumerate(elimination_results_list):
        if not isinstance(res, dict):
            raise TypeError(f"Result at index {i} must be a dict.")
        kept = res.get("kept_groups", [])
        names = {_group_name(g) for g in kept}
        neuron_kept_names.append(names)

    if variable_names is None:
        all_names = set()
        for names in neuron_kept_names:
            all_names.update(names)
        variables = sorted(all_names)
    else:
        variables = [str(v) for v in variable_names]

    n_neurons = len(neuron_kept_names)
    n_vars = len(variables)
    neuron_tuning_matrix = np.zeros((n_neurons, n_vars), dtype=bool)

    var_to_idx = {v: j for j, v in enumerate(variables)}
    for i, names in enumerate(neuron_kept_names):
        for name in names:
            j = var_to_idx.get(name)
            if j is not None:
                neuron_tuning_matrix[i, j] = True

    n_tuned = neuron_tuning_matrix.sum(axis=0)
    fraction_tuned = n_tuned / n_neurons if n_neurons > 0 else np.zeros(n_vars)

    return {
        "variables": variables,
        "fraction_tuned": fraction_tuned,
        "n_tuned": n_tuned,
        "n_total": n_neurons,
        "neuron_tuning_matrix": neuron_tuning_matrix,
        "source": "backward_elimination_kept_groups",
    }
