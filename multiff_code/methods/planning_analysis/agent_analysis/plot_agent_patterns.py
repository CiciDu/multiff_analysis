import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from planning_analysis.factors_vs_indicators import process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators.plot_styles_mpl import LINESTYLE_MAP
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_utils_mpl



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



def build_fixed_values_for_obs_sweep(
    sweep_key,
    env=None,
):
    """
    Build fixed_values dict such that:
    - One of the four obs keys is left free (NOT fixed)
    - The other three are fixed to 0
    - All environment cost parameters are fixed
    
    Parameters
    ----------
    rl : object
        RL object containing env_for_data_collection
    sweep_key : str
        One of:
        'obs_perc_r'
        'obs_perc_th'
        'obs_mem_r'
        'obs_mem_th'
    """

    obs_keys = [
        'obs_perc_r',
        'obs_perc_th',
        'obs_mem_r',
        'obs_mem_th',
    ]

    if sweep_key not in obs_keys:
        raise ValueError(f'sweep_key must be one of {obs_keys}')

    # Fix the other three obs keys to 0
    fixed_values = {
        key: 0 for key in obs_keys if key != sweep_key
    }

    if env is not None:
        fixed_values.update({
            'dv': env.dv_cost_factor,
            'jerk': env.jerk_cost_factor,
            'stop': env.cost_per_stop,
            'num_obs_ff': env.num_obs_ff,
            'max_in_memory_time': env.max_in_memory_time,
            'identity_slot_strategy': env.identity_slot_strategy,
            'identity_slot_base': env.identity_slot_base,
            'new_ff_scope': env.new_ff_scope,
            'v_noise_std': env.v_noise_std,
            'w_noise_std': env.w_noise_std,
        })

    return fixed_values


def build_conditioned_subset(
    df,
    fixed_values=None,
    verbose=True
):
    """
    Build conditioning dictionary and return filtered dataframe subset.
    """
    # ---------------------------------------
    # Apply mask
    # ---------------------------------------

    mask = np.ones(len(df), dtype=bool)

    if fixed_values is not None:
        for key, value in fixed_values.items():
            if key not in df.columns:
                continue
            mask &= df[key] == value
    else:
        mask = np.ones(len(df), dtype=bool)

    subset = df[mask]

    if verbose:
        print('\nConditioning on:')
        for k, v in fixed_values.items():
            print(f'  {k} = {v}')
        print(f'Remaining rows: {len(subset)}')

    if len(subset) == 0:
        print('⚠️ No rows match this full conditioning.')
        return None, fixed_values

    return subset, fixed_values


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

    if fixed_values is None:
        reference_row = df.iloc[reference_row_index]

        fixed_values = {
            col: reference_row[col]
            for col in df.columns
            if col not in [x_key, y_key, 'agent_name']
        }
        
    # ---------------------------------------
    # Build conditioned subset
    # ---------------------------------------

    subset, fixed_values = build_conditioned_subset(
        df=df,
        verbose=verbose,
        fixed_values=fixed_values
    )

    if subset is None:
        return None

    # ---------------------------------------
    # Plot
    # ---------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

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

import matplotlib.pyplot as plt


def add_seed_column(df):
    """
    Extract seed from id column.
    Returns a copy of df with a 'seed' column.
    """

    df = df.copy()
    df['seed'] = df['id'].str.extract(r'(seed\d+)')
    return df

def plot_agents_across_sweep(
    subset,
    sweep_key,
    y_key='diff_in_abs_angle_to_nxt_ff_median',
    ax=None
):
    """
    Plot one line per agent across sweep values.
    """

    subset = add_seed_column(subset)
    subset = subset.sort_values([sweep_key, 'seed'])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    for seed, df_seed in subset.groupby('seed'):

        ax.plot(
            df_seed[sweep_key],
            df_seed[y_key],
            marker='o',
            linewidth=1,
            alpha=0.5
        )

    ax.set_xlabel(sweep_key)
    ax.set_ylabel(y_key)
    ax.set_title(f'{y_key} vs {sweep_key}')

    return ax

def _break_up_agent_df_to_smaller_ones(df, fixed_variable_values_to_use, changeable_variables):
    """
    Filter agent dataframe by fixed values and split by changeable variable combinations.
    Uses shared filter_and_split_by_fixed_and_changeable (agent data skips ML-specific preprocessing).
    Returns (list_of_smaller_dfs, combinations).
    """
    return process_variations_utils.filter_and_split_by_fixed_and_changeable(
        df, fixed_variable_values_to_use, changeable_variables, skip_missing_fixed_keys=True
    )


def plot_agents_with_conditions(
    subset,
    sweep_key,
    y_key='diff_in_abs_angle_to_nxt_ff_median',
    condition_col='test_or_control',
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    ax=None
):
    """
    Plot sweep curves for both test and control agents (or other color/line groupings).
    Thin lines = individual seeds
    Thick line = median across seeds

    Parameters
    ----------
    subset : pd.DataFrame
        Agent data with 'id', sweep_key, y_key, and grouping columns.
    sweep_key : str
        Column for x-axis.
    y_key : str
        Column for y-axis.
    condition_col : str, optional
        Deprecated. Use columns_to_find_unique_combinations_for_color instead.
        Column for color grouping (e.g. 'test_or_control').
    columns_to_find_unique_combinations_for_color : list, optional
        Columns whose unique combinations define color groups (e.g. ['test_or_control']).
    columns_to_find_unique_combinations_for_line : list, optional
        Columns whose unique combinations define line styles.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    """

    subset = subset.copy()

    # extract seed
    subset['seed'] = subset['id'].str.extract(r'(seed\d+)')

    color_cols = columns_to_find_unique_combinations_for_color
    if color_cols is None or (isinstance(color_cols, list) and len(color_cols) == 0):
        color_cols = [condition_col]
    line_cols = columns_to_find_unique_combinations_for_line or []

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Default color map for common test/control
    default_color_map = {
        'test': 'royalblue',
        'control': 'orange'
    }
    colors = ['royalblue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'pink',
              'cyan', 'magenta', 'navy', 'darkgreen', 'darkorange', 'darkviolet']

    # Build grouping keys
    def _combo_key(row, cols):
        return '; '.join(str(row[c]) for c in cols if c in row.index)

    subset['_color_key'] = subset.apply(lambda r: _combo_key(r, color_cols), axis=1)
    if line_cols:
        subset['_line_key'] = subset.apply(lambda r: _combo_key(r, line_cols), axis=1)
    else:
        subset['_line_key'] = ''

    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    unique_color_keys = subset['_color_key'].unique()
    color_map = {k: default_color_map.get(k, colors[i % len(colors)])
                 for i, k in enumerate(unique_color_keys)}
    unique_line_keys = subset['_line_key'].unique()
    line_map = {k: line_styles[i % len(line_styles)] for i, k in enumerate(unique_line_keys)}

    group_cols = ['_color_key', '_line_key'] if line_cols else ['_color_key']

    for keys, df_grp in subset.groupby(group_cols):
        color_key = keys[0] if isinstance(keys, tuple) else keys
        line_key = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else ''
        color = color_map[color_key]
        linestyle = LINESTYLE_MAP.get(line_map.get(line_key, 'solid'), '-')

        df_grp = df_grp.sort_values([sweep_key, 'seed'])

        # individual agents
        for seed, df_seed in df_grp.groupby('seed'):
            ax.plot(
                df_seed[sweep_key],
                df_seed[y_key],
                color=color,
                linestyle=linestyle,
                alpha=0.25,
                linewidth=1
            )

        # median across agents
        grouped = (
            df_grp
            .groupby(sweep_key)[y_key]
            .median()
            .reset_index()
        )
        label = color_key if not line_key else f'{color_key} | {line_key}'
        ax.plot(
            grouped[sweep_key],
            grouped[y_key],
            color=color,
            linestyle=linestyle,
            linewidth=3,
            marker='o',
            label=label
        )

    ax.set_xlabel(sweep_key)
    ax.set_ylabel(y_key)
    ax.legend()
    ax.set_title(f'{y_key} vs {sweep_key}')

    return ax


def streamline_plot_agents_with_conditions(
    original_df,
    fixed_variable_values_to_use=None,
    changeable_variables=None,
    sweep_key_list=None,
    sweep_key=None,
    y_key='diff_in_abs_angle_to_nxt_ff_median',
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    title_prefix=None,
    use_subplots_based_on_changeable_variables=False,
    show_fig=False,
    figsize_per_plot=(6, 4),
):
    """
    Filter/iterate through agent data and plot sweep curves for each subset.
    Mirrors the interface of streamline_making_matplotlib_plot_to_compare_two_sets_of_data.

    Parameters
    ----------
    original_df : pd.DataFrame
        Full agent dataframe.
    fixed_variable_values_to_use : dict, optional
        Fixed variable values to filter by (e.g. {'num_obs_ff': 3}).
    changeable_variables : list, optional
        Variables that vary across subplots (e.g. ['obs_perc_r', 'env_type']).
    sweep_key_list : list, optional
        List of x-axis columns to iterate over. If None, uses sweep_key.
    sweep_key : str, optional
        Single x-axis column when sweep_key_list is None.
    y_key : str
        Y-axis column.
    columns_to_find_unique_combinations_for_color : list, optional
        Columns for color groups (default ['test_or_control']).
    columns_to_find_unique_combinations_for_line : list, optional
        Columns for line style groups.
    title_prefix : str, optional
        Prefix for subplot titles.
    use_subplots_based_on_changeable_variables : bool
        If True and len(changeable_variables)==2, use a subplot grid.
    show_fig : bool
        Whether to call plt.show().
    figsize_per_plot : tuple
        (width, height) per subplot when using subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    original_df = original_df.copy()
    fixed_variable_values_to_use = fixed_variable_values_to_use or {}
    changeable_variables = changeable_variables or []
    columns_to_find_unique_combinations_for_color = (
        columns_to_find_unique_combinations_for_color or ['test_or_control']
    )
    columns_to_find_unique_combinations_for_line = (
        columns_to_find_unique_combinations_for_line or []
    )

    x_var_column_list = sweep_key_list if sweep_key_list is not None else [sweep_key]
    if x_var_column_list == [None]:
        raise ValueError('Provide either sweep_key_list or sweep_key.')

    use_subplot_grid = (
        use_subplots_based_on_changeable_variables and len(changeable_variables) == 2
    )
    if use_subplot_grid:
        changeable_variables = plot_variations_utils._check_order_in_changeable_variables(
            changeable_variables, original_df
        )

    list_of_smaller_dfs, combinations = _break_up_agent_df_to_smaller_ones(
        original_df,
        fixed_variable_values_to_use,
        changeable_variables,
    )

    if use_subplot_grid:
        fig, axes, row_number, col_number, second_dim = plot_variations_utils_mpl.create_streamline_subplot_grid(
            original_df, changeable_variables, combinations,
            figsize_per_plot=figsize_per_plot,
            x_var_column_list=x_var_column_list,
        )
    else:
        fig = None
        row_number = None
        col_number = None

    for combo, filtered_df in zip(combinations, list_of_smaller_dfs):
        if filtered_df.empty:
            continue
        for x_var_column in x_var_column_list:
            if x_var_column not in filtered_df.columns:
                continue
            title = f'{y_key} vs {x_var_column}'
            if title_prefix:
                title = title_prefix + ' ' + title
            if len(combo) > 0:
                combo_str = ' · '.join(f'{k}: {v}' for k, v in sorted(combo.items()))
                title = title + f' ({combo_str})'

            if not use_subplot_grid:
                fig, ax = plt.subplots(figsize=figsize_per_plot)
                print(' ')
                print('=========================================================')
                print('Current combination of changeable variables:', combo)
                plot_agents_with_conditions(
                    filtered_df,
                    x_var_column,
                    y_key=y_key,
                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                    ax=ax
                )
                ax.set_title(title)
                if show_fig:
                    plt.show()
            else:
                ax = axes[row_number - 1, col_number - 1]
                plot_agents_with_conditions(
                    filtered_df,
                    x_var_column,
                    y_key=y_key,
                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                    ax=ax
                )
                ax.set_title(title)
                col_number += 1
                if col_number > second_dim:
                    col_number = 1
                    row_number += 1

    if fig is not None:
        fig._combinations = combinations
    if show_fig and use_subplot_grid and fig is not None:
        plt.show()

    return fig