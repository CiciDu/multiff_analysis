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
    verbose=True
):
    """
    Plot Y vs X while holding all other variables fixed.

    Parameters
    ----------
    df : pandas.DataFrame
    x_key : str
        Variable to sweep on x-axis
    fixed_values : dict or None
        Explicit conditioning values.
        If None, uses reference_row_index to auto-fix others.
    y_key : str
        Dependent variable
    reference_row_index : int
        Row used for automatic conditioning if fixed_values is None
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
        print('Conditioning on:')
        for k, v in fixed_values.items():
            print(f'  {k} = {v}')
        print(f'\nRemaining rows: {len(subset)}')

    if len(subset) == 0:
        print('⚠️ No rows match this full conditioning.')
        return

    # ---------------------------------------
    # Plot
    # ---------------------------------------

    x = subset[x_key].values
    y = subset[y_key].values

    # if np.nanstd(x) > 0:
    #     x_jitter = x + np.random.randn(len(x)) * 0.02 * np.nanstd(x)
    # else:
    #     x_jitter = x
    
    x_jitter = x

    y_jitter = y + np.random.randn(len(y)) * 0.5

    plt.figure(figsize=(5, 4))
    plt.scatter(x_jitter, y_jitter, alpha=0.7)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f'{y_key} vs {x_key}\n(all other params fixed)')
    plt.tight_layout()
    plt.show()
    
    return subset
    
    
