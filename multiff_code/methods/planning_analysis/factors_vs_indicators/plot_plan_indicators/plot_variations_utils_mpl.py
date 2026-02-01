# ============================================================
# Matplotlib versions (same interface, no hover)
# ============================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from planning_analysis.factors_vs_indicators import process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_class, plot_variations_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors



LINESTYLE_MAP = {
    'solid': 'solid',
    'dash': '--',
    'dot': ':',
    'dashdot': '-.',
}


def _get_mpl_ax(fig, row_number=None, col_number=None):
    """
    Internal helper: get the appropriate Axes from a Figure,
    using the 1-based row/col indices and a stored axes grid if present.
    """
    axes_grid = getattr(fig, '_axes_grid', None)
    if axes_grid is not None and row_number is not None and col_number is not None:
        return axes_grid[row_number - 1, col_number - 1]

    # fallback: single axes figure
    if fig.axes:
        return fig.axes[0]
    return fig.add_subplot(111)


def _update_ax_based_on_x_labels_to_values_map(ax, x_labels_to_values_map):
    """
    Set xticks and labels from the x_labels_to_values_map.
    """
    tickvals = list(x_labels_to_values_map.values())
    ticktext = [str(k) for k in x_labels_to_values_map.keys()]
    ax.set_xticks(tickvals)
    ax.set_xticklabels(ticktext)

    max_len = max((len(t) for t in ticktext), default=0)
    if max_len > 30:
        ax.tick_params(axis='x', labelrotation=90)
    elif max_len > 12:
        ax.tick_params(axis='x', labelrotation=25)


def _set_minimal_y_scale_mpl(ax, sub_df, y_var_column):
    """
    Set a reasonable y-limits based on data or CI bounds.
    """
    if {'ci_lower', 'ci_upper'}.issubset(sub_df.columns):
        y_lo = sub_df['ci_lower'].astype(float)
        y_hi = sub_df['ci_upper'].astype(float)
        min_y = np.nanmin(y_lo.to_numpy())
        max_y = np.nanmax(y_hi.to_numpy())
    else:
        y = sub_df[y_var_column].astype(float)
        min_y = np.nanpercentile(y, 2.5)
        max_y = np.nanpercentile(y, 97.5)

    if not np.isfinite(min_y) or not np.isfinite(max_y):
        min_y, max_y = 0.0, 1.0

    if np.isclose(min_y, max_y):
        pad = 0.5 if max_y == 0 else 0.1 * abs(max_y)
        min_y, max_y = min_y - pad, max_y + pad
    else:
        pad = 0.05 * (max_y - min_y)
        min_y, max_y = min_y - pad, max_y + pad

    ax.set_ylim(min_y, max_y)


def _add_color_legends_mpl(ax, sub_df, row_number=None, col_number=None):
    """
    Add color legend entries (as proxy artists) for unique combinations.
    Only on the first subplot (1,1) if using subplots.
    """
    showlegend = True
    if (row_number is not None) and (col_number is not None):
        if (row_number != 1) or (col_number != 1):
            showlegend = False
    if not showlegend:
        return

    legend_entries = sub_df[['color', 'unique_combination']].drop_duplicates()
    if len(legend_entries) <= 1:
        return

    handles = []
    labels = []
    for _, row in legend_entries.iterrows():
        handles.append(Line2D([0], [0], color=row['color'], lw=2))
        labels.append(row['unique_combination'])

    # merge with existing legend if any
    existing = ax.get_legend_handles_labels()
    handles = existing[0] + handles
    labels = existing[1] + labels

    ax.legend(handles, labels, frameon=False)


def _add_line_type_legends_mpl(ax, sub_df, row_number=None, col_number=None):
    """
    Add line-style legend entries (as proxy artists) for unique combinations.
    Only on the first subplot (1,1) if using subplots.
    """
    showlegend = True
    if (row_number is not None) and (col_number is not None):
        if (row_number != 1) or (col_number != 1):
            showlegend = False
    if not showlegend:
        return

    legend_entries = sub_df[['line_type', 'unique_combination_for_line']].drop_duplicates()
    if len(legend_entries) <= 1:
        return

    handles = []
    labels = []
    for _, row in legend_entries.iterrows():
        linestyle = LINESTYLE_MAP.get(row['line_type'], 'solid')
        handles.append(Line2D([0], [0], color='black', linestyle=linestyle, lw=2))
        labels.append(row['unique_combination_for_line'])

    existing = ax.get_legend_handles_labels()
    handles = existing[0] + handles
    labels = existing[1] + labels

    ax.legend(handles, labels, frameon=False)


def plot_markers_for_data_comparison_mpl(fig,
                                         sub_df,
                                         customdata_columns,
                                         y_var_column,
                                         use_ribbons_to_replace_error_bars=True,
                                         row_number=None,
                                         col_number=None,
                                         is_difference=False,
                                         constant_marker_size=None):
    """
    Matplotlib version of plot_markers_for_data_comparison.
    No hover, purely static.
    """
    ax = _get_mpl_ax(fig, row_number=row_number, col_number=col_number)

    sub_df = sub_df.dropna(subset=[y_var_column]).copy()

    if constant_marker_size is not None:
        marker_size = constant_marker_size
    else:
        max_n = max(int(sub_df.get('sample_size', pd.Series([1])).max()), 1)
        scale = 45.0 / max_n
        min_size, max_size = 6, 18

    for line_color in sub_df['line_color'].unique():
        d = sub_df[sub_df['line_color'] == line_color].copy()
        d = d.sort_values('x_value_numeric_with_offset')

        # CI ribbon or vertical error bar
        if {'ci_lower', 'ci_upper'}.issubset(d.columns):
            if use_ribbons_to_replace_error_bars:
                # fill_between for CI
                if len(d) == 1:
                    # duplicate row so fill_between still works
                    d = pd.concat([d, d], ignore_index=True)

                rgba = mcolors.to_rgba(line_color, alpha=0.25)

                ax.fill_between(
                    d['x_value_numeric_with_offset'],
                    d['ci_lower'],
                    d['ci_upper'],
                    color=rgba,
                    linewidth=0
                )
            else:
                ax.vlines(
                    d['x_value_numeric_with_offset'].to_numpy(),
                    d['ci_lower'].to_numpy(),
                    d['ci_upper'].to_numpy(),
                    linewidth=1.5,
                    color=line_color
                )

        this_color = 'green' if is_difference else line_color

        if constant_marker_size is not None:
            sizes = np.full(len(d), marker_size)
        else:
            sizes = np.clip(
                d['sample_size'].fillna(max_n).astype(float).to_numpy() * scale,
                min_size,
                max_size
            )

        ax.plot(
            d['x_value_numeric_with_offset'].to_numpy(),
            d[y_var_column].to_numpy(),
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=np.mean(sizes) ** 0.5,  # crude mapping
            color=this_color,
            alpha=0.9,
            label=None  # legends handled separately
        )

    return fig


def connect_every_pair_mpl(fig,
                           sub_df,
                           y_var_column,
                           customdata_columns,
                           show_combo_legends=True,
                           row_number=None,
                           col_number=None):

    ax = _get_mpl_ax(fig, row_number=row_number, col_number=col_number)

    if sub_df['x_value_numeric_with_offset'].value_counts().max() <= 2:
        return fig

    for pair_id in sub_df['pair_id'].unique():
        d = sub_df[sub_df['pair_id'] == pair_id].copy()
        if d.empty:
            continue
        d = d.sort_values('x_value_numeric_with_offset')
        if len(d) < 2:
            continue

        row0 = d.iloc[0]
        color = row0.get('color', 'black')
        linestyle = LINESTYLE_MAP.get(row0.get('line_type', 'solid'), 'solid')

        ax.plot(
            d['x_value_numeric_with_offset'].to_numpy(),
            d[y_var_column].to_numpy(),
            linestyle=linestyle,
            linewidth=1.0,
            color=color,
            marker=None
        )

    if show_combo_legends:
        _add_color_legends_mpl(ax, sub_df, row_number=row_number, col_number=col_number)
        _add_line_type_legends_mpl(ax, sub_df, row_number=row_number, col_number=col_number)

    return fig


# ---- single-bin (grouped bar) path for Matplotlib ----

def _add_grouped_bars_mpl(ax, d, grp_col, y_col):
    """
    Grouped bars with optional asymmetric CI error bars.
    """
    x_vals = np.arange(len(d))
    ax.bar(
        x_vals,
        d[y_col].to_numpy(),
        width=0.6,
        align='center',
        color=[{'control': '#F58518', 'test': '#4C78A8'}.get(g, 'gray') for g in d[grp_col]],
    )

    if plot_variations_utils._has_bounds(d):
        # d already has err_plus / err_minus
        err_plus = d['err_plus'].to_numpy()
        err_minus = d['err_minus'].to_numpy()
        ax.errorbar(
            x_vals,
            d[y_col].to_numpy(),
            yerr=[err_minus, err_plus],
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=3,
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(g) for g in d[grp_col]])

    return ax


def _apply_single_bin_layout_mpl(ax, title_text, grp_col, y_col, y_min, y_max):
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_xlabel(grp_col)
    ax.set_ylabel(y_col)
    ax.set_title(title_text)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    return ax


def _single_bin_path_mpl(fig, sub_df, x_var_column, y_var_column,
                         x_labels_to_values_map, title,
                         row_number=None, col_number=None,
                         is_difference=False):

    ax = _get_mpl_ax(fig, row_number=row_number, col_number=col_number)

    grp_col = plot_variations_utils._choose_group_column(sub_df)
    d = plot_variations_utils._first_per_group(sub_df, grp_col, y_var_column)

    # attach sample sizes if you want them later (not displayed by default)
    d['_n'] = d.apply(lambda r: plot_variations_utils._infer_group_n(r, grp_col, sub_df), axis=1)

    if plot_variations_utils._has_bounds(d):
        d = plot_variations_utils._attach_error_arrays(d, y_var_column)
        y_lo = np.minimum(d[y_var_column], d['ci_lower']).min()
        y_hi = np.maximum(d[y_var_column], d['ci_upper']).max()
    else:
        y_lo = d[y_var_column].min()
        y_hi = d[y_var_column].max()

    _add_grouped_bars_mpl(ax, d, grp_col, y_var_column)

    base_title = title or f'{y_var_column} (bar' + (' ± 95% CI' if plot_variations_utils._has_bounds(d) else '') + ')'
    if title is None:
        single_x_label = next(iter(x_labels_to_values_map.keys()), '')
        base_title = f'{base_title} — {x_var_column}: {single_x_label}'

    _apply_single_bin_layout_mpl(ax, base_title, grp_col, y_var_column, float(y_lo), float(y_hi))
    return fig


# ---- main Matplotlib equivalent of make_plotly_plot_to_compare_two_sets_of_data ----

def make_matplotlib_plot_to_compare_two_sets_of_data(
    sub_df,
    x_var_column,
    y_var_column='diff_in_abs_angle_to_nxt_ff_median',
    var_to_determine_x_offset_direction='ref_columns_only',
    title=None,
    x_offset=0.0,
    columns_to_find_unique_combinations_for_color=None,
    columns_to_find_unique_combinations_for_line=None,
    show_combo_legends=True,
    fig=None,
    row_number=None,
    col_number=None,
    is_difference=False,
    constant_marker_size=None
):
    """
    Matplotlib version, same interface as make_plotly_plot_to_compare_two_sets_of_data,
    returning a Matplotlib Figure (no hover).
    """
    columns_to_find_unique_combinations_for_color = columns_to_find_unique_combinations_for_color or []
    columns_to_find_unique_combinations_for_line = columns_to_find_unique_combinations_for_line or []

    if fig is None:
        fig, _ = plt.subplots()

    # --- preprocessing (colors/lines) ---
    rest_hover = plot_variations_utils._find_rest_of_x_for_hoverdata(
        sub_df, x_var_column, y_var_column, var_to_determine_x_offset_direction
    )
    sub_df = plot_variations_utils._process_x_var_columns(sub_df, x_var_column)

    cols_for_color = plot_variations_utils._process_columns_to_find_unique_combinations_for_color(
        columns_to_find_unique_combinations_for_color, x_var_column, rest_hover
    )
    sub_df = process_variations_utils.assign_color_to_sub_df_based_on_unique_combinations(
        sub_df, cols_for_color)
    sub_df = process_variations_utils.assign_line_type_to_sub_df_based_on_unique_combinations(
        sub_df, columns_to_find_unique_combinations_for_line
    )

    # numeric x + offset
    x_labels_to_values_map = plot_variations_utils._find_x_labels_to_values_map(sub_df, x_var_column)
    sub_df = plot_variations_utils._add_x_value_numeric_to_sub_df(
        sub_df, x_var_column, x_labels_to_values_map, x_offset
    )

    # single-bin path → grouped bar
    if sub_df['x_value_numeric'].nunique() == 1:
        fig = _single_bin_path_mpl(
            fig, sub_df, x_var_column, y_var_column,
            x_labels_to_values_map, title,
            row_number=row_number, col_number=col_number,
            is_difference=is_difference
        )
        return fig

    # multi-bin path: markers + optional ribbons + connection lines
    fig = plot_markers_for_data_comparison_mpl(
        fig, sub_df, rest_hover, y_var_column,
        row_number=row_number, col_number=col_number,
        is_difference=is_difference,
        constant_marker_size=constant_marker_size
    )

    if not is_difference:
        fig = connect_every_pair_mpl(
            fig, sub_df, y_var_column, rest_hover,
            show_combo_legends=show_combo_legends,
            row_number=row_number, col_number=col_number
        )

    ax = _get_mpl_ax(fig, row_number=row_number, col_number=col_number)
    _update_ax_based_on_x_labels_to_values_map(ax, x_labels_to_values_map)
    _set_minimal_y_scale_mpl(ax, sub_df, y_var_column)

    ax.set_title(title or f'{y_var_column} vs {x_var_column}')
    ax.set_xlabel(x_var_column)
    ax.set_ylabel(y_var_column)

    return fig


# ---- streamlined Matplotlib version of streamline_making_plotly_plot_to_compare_two_sets_of_data ----

def streamline_making_matplotlib_plot_to_compare_two_sets_of_data(
    original_df,
    fixed_variable_values_to_use,
    changeable_variables,
    x_var_column_list,
    columns_to_find_unique_combinations_for_color=['test_or_control'],
    columns_to_find_unique_combinations_for_line=[],
    var_to_determine_x_offset_direction='ref_columns_only',
    y_var_column='avg_r_squared',
    title_prefix=None,
    use_subplots_based_on_changeable_variables=False,
    add_ci_bounds=True,
    is_difference=False,
    constant_marker_size=None,
    show_fig=False,
):
    """
    Matplotlib version of streamline_making_plotly_plot_to_compare_two_sets_of_data.
    Same interface; returns a Matplotlib Figure.
    """

    if use_subplots_based_on_changeable_variables & (len(changeable_variables) == 2):
        changeable_variables = plot_variations_utils._check_order_in_changeable_variables(
            changeable_variables, original_df
        )

    list_of_smaller_dfs, combinations = process_variations_utils.break_up_df_to_smaller_ones(
        original_df,
        fixed_variable_values_to_use,
        changeable_variables,
        var_to_determine_x_offset_direction=var_to_determine_x_offset_direction,
        y_var_column=y_var_column,
        add_ci_bounds=add_ci_bounds,
    )

    if use_subplots_based_on_changeable_variables:
        first_dim, second_dim = plot_variations_utils._find_first_and_second_dim(
            original_df, changeable_variables, combinations
        )
        all_subplot_titles = plot_variations_utils._get_all_subplot_titles(combinations)

        fig, axes = plt.subplots(
            first_dim,
            second_dim,
            figsize=(7 * second_dim, 5 * first_dim),
            squeeze=False
        )
        # stash axes grid for _get_mpl_ax
        fig._axes_grid = axes

        # set subplot titles
        for idx, combo_title in enumerate(all_subplot_titles):
            r = idx // second_dim
            c = idx % second_dim
            axes[r, c].set_title(combo_title, fontsize=14)

        row_number = 1
        col_number = 1
    else:
        fig = None
        row_number = None
        col_number = None

    for combo, filtered_df in zip(combinations, list_of_smaller_dfs):
        for x_var_column in x_var_column_list:
            if 'y_var_column' in combo.keys():
                title = str.upper(x_var_column) + ' vs ' + str.upper(combo['y_var_column'])
            else:
                title = str.upper(x_var_column) + ' vs ' + str.upper(y_var_column)

            if title_prefix is not None:
                title = title_prefix + ' ' + title

            if not use_subplots_based_on_changeable_variables:
                fig, _ = plt.subplots()
                print(' ')
                print('=========================================================')
                print('Current combination of changeable variables:', combo)

            fig = make_matplotlib_plot_to_compare_two_sets_of_data(
                filtered_df,
                x_var_column,
                y_var_column=y_var_column,
                var_to_determine_x_offset_direction=var_to_determine_x_offset_direction,
                title=title,
                columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                fig=fig,
                row_number=row_number,
                col_number=col_number,
                is_difference=is_difference,
                constant_marker_size=constant_marker_size,
            )

            if use_subplots_based_on_changeable_variables:
                col_number += 1
                if col_number > second_dim:
                    col_number = 1
                    row_number += 1
            elif show_fig:
                plt.show()

    fig._combinations = combinations
    
    if show_fig and use_subplots_based_on_changeable_variables:
        plt.show()
    
    
    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings


def plot_grouped_percent_bars(
    df: pd.DataFrame,
    *,
    group_order=('control', 'test'),
    monkey_order=None,
    bar_width=0.36,
    capsize=4,
    ylabel='Percentage',
    title=None,
    figsize=(7.5, 3.8),
    palette=None,
    ylim=None,
    show_values=False,
    ax=None,
):
    """
    Grouped bar chart with asymmetric CI error bars (Matplotlib).

    Notes
    -----
    - Expects 'perc', 'ci_lower', 'ci_upper' in [0, 1].
    - If values appear to be in [0, 100], they are automatically converted.
    """

    d = df.copy()
    d = d[d['test_or_control'].isin(group_order)]
    
    if 'ci_lower' not in d.columns:
        try:
            d['ci_lower'] = d['perc_ci_low_95']
            d['ci_upper'] = d['perc_ci_high_95']
        except:
            raise ValueError('ci_lower and ci_upper columns not found in dataframe, and perc_ci_low_95 and perc_ci_high_95 columns not found either')

    # ------------------------------------------------------------------
    # Auto-detect and convert 0–100 percentages → [0, 1]
    # ------------------------------------------------------------------
    value_cols = ['perc', 'ci_lower', 'ci_upper']
    max_vals = d[value_cols].max()

    if (max_vals > 1.5).all():
        warnings.warn(
            'Detected percentage values in [0, 100]; converting to [0, 1].',
            RuntimeWarning,
            stacklevel=2,
        )
        d[value_cols] = d[value_cols] / 100.0

    # ------------------------------------------------------------------
    # Category orders
    # ------------------------------------------------------------------
    if monkey_order is None:
        monkey_order = d['monkey_name'].drop_duplicates().tolist()

    d['monkey_name'] = pd.Categorical(
        d['monkey_name'], categories=monkey_order, ordered=True
    )
    d['test_or_control'] = pd.Categorical(
        d['test_or_control'], categories=list(group_order), ordered=True
    )
    d = d.sort_values(['monkey_name', 'test_or_control'])

    # X positions
    monkeys = pd.Index(monkey_order)
    x = np.arange(len(monkeys), dtype=float)

    # Group offsets
    n_groups = len(group_order)
    offsets = np.linspace(
        -bar_width * (n_groups - 1) / 2,
        bar_width * (n_groups - 1) / 2,
        n_groups,
    )

    # Axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    for i, grp in enumerate(group_order):
        sub = d[d['test_or_control'] == grp]
        sub = sub.set_index('monkey_name').reindex(monkeys)

        y = sub['perc'].to_numpy()
        lo = sub['ci_lower'].to_numpy()
        hi = sub['ci_upper'].to_numpy()
        yerr = np.vstack([y - lo, hi - y])

        color = None if palette is None else palette.get(grp, None)

        bars = ax.bar(
            x + offsets[i],
            y,
            width=bar_width,
            yerr=yerr,
            capsize=capsize,
            label=grp,
            color=color,
            edgecolor='none',
        )

        if show_values:
            for rect, val in zip(bars, y):
                ax.annotate(
                    f'{val * 100:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

    # Axes cosmetics
    ax.set_xticks(x, monkeys.tolist())
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.yaxis.grid(True, linestyle='-', alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(title='Group', frameon=False)

    if title:
        ax.set_title(title)

    if created_fig:
        fig.tight_layout()

    return fig, ax
