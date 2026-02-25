import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def plot_pairwise_interactions(
    df,
    sweep_keys,
    target_col='num_caught_ff',
    figsize=(6, 5),
    cmap='viridis'
):
    """
    Plot mean interaction heatmaps for all pairwise combinations
    of sweep_keys against target_col.
    """

    valid_keys = [k for k in sweep_keys if k in df.columns]

    for key_x, key_y in itertools.combinations(valid_keys, 2):

        sub = df[[key_x, key_y, target_col]].dropna()

        if sub.empty:
            continue

        # Compute mean performance grid
        pivot = (
            sub.groupby([key_y, key_x])[target_col]
               .mean()
               .unstack()
               .sort_index(ascending=True)
        )

        if pivot.empty:
            continue

        x_vals = pivot.columns.values
        y_vals = pivot.index.values
        Z = pivot.values

        # Mask NaNs
        Z_masked = np.ma.masked_invalid(Z)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            Z_masked,
            origin='lower',
            aspect='equal',
            cmap=cmap
        )

        # Proper numeric ticks
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))

        ax.set_xticklabels(np.round(x_vals, 4), rotation=45)
        ax.set_yticklabels(np.round(y_vals, 4))

        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
        ax.set_title(f'{key_x} × {key_y}')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'Mean {target_col}')

        plt.tight_layout()
        plt.show()
       
       
import numpy as np
import matplotlib.pyplot as plt


def plot_conditional(
    df,
    x_key,
    fixed_values=None,
    y_key='num_caught_ff',
    reference_row_index=0,
    ax=None,
    verbose=True
):
    """
    Plot Y vs X while holding all other variables fixed.
    Shows boxplot + scatter (no jitter).
    """

    if x_key not in df.columns:
        raise ValueError(f'{x_key} not in dataframe')

    if y_key not in df.columns:
        raise ValueError(f'{y_key} not in dataframe')

    # ---------------------------------------
    # Build conditioning dictionary
    # ---------------------------------------

    if fixed_values is None:
        reference_row = df.iloc[reference_row_index]
        fixed_values = {
            col: reference_row[col]
            for col in df.columns
            if col not in [x_key, y_key, 'agent_name']
        }

    # ---------------------------------------
    # Apply mask
    # ---------------------------------------

    mask = np.ones(len(df), dtype=bool)

    for key, value in fixed_values.items():
        if key not in df.columns:
            continue
        mask &= df[key] == value

    subset = df[mask]

    if verbose:
        print('\nConditioning on:')
        for k, v in fixed_values.items():
            print(f'  {k} = {v}')
        print(f'Remaining rows: {len(subset)}')

    if len(subset) == 0:
        print('⚠️ No rows match this full conditioning.')
        return None

    # ---------------------------------------
    # Plot
    # ---------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # Sort by x for clean ordering
    subset = subset.sort_values(x_key)

    # --- Boxplot ---
    subset.boxplot(
        column=y_key,
        by=x_key,
        ax=ax,
        grid=False,
        showfliers=False
    )

    # --- Scatter overlay (no jitter) ---
    x_vals = subset[x_key].values
    y_vals = subset[y_key].values

    unique_x = np.sort(subset[x_key].unique())
    x_positions = np.array([np.where(unique_x == v)[0][0] + 1 for v in x_vals])

    ax.scatter(x_positions, y_vals, alpha=0.5)

    ax.set_title(f'{y_key} vs {x_key}')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    if ax.figure:
        ax.figure.suptitle('')

    if ax is None:
        plt.tight_layout()
        plt.show()

    return subset


def plot_all_single_param_conditionals(
    df,
    sweep_keys=('obs_perc_r', 'obs_perc_th', 'obs_mem_r', 'obs_mem_th'),
    y_key='num_caught_ff',
    base_fixed=None,
    verbose=False
):
    """
    Create 2x2 subplot grid showing conditional distributions
    for all sweep parameters.
    """

    if base_fixed is None:
        base_fixed = {}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    subsets = {}

    for i, x_key in enumerate(sweep_keys):

        fixed_values = base_fixed.copy()

        # Hold other sweep params at zero
        for other_key in sweep_keys:
            if other_key != x_key:
                fixed_values[other_key] = 0.0

        subset = plot_conditional(
            df,
            x_key=x_key,
            fixed_values=fixed_values,
            y_key=y_key,
            ax=axes[i],
            verbose=verbose
        )

        subsets[x_key] = subset

    plt.tight_layout()
    plt.show()

    return subsets