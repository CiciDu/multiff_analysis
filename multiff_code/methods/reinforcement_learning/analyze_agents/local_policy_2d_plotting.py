import numpy as np
import matplotlib.pyplot as plt

from reinforcement_learning.analyze_agents import analyze_policy_smoothness
from reinforcement_learning.analyze_agents import local_policy_2d_core
from reinforcement_learning.analyze_agents.local_policy_2d_core import (
    _choose_anchor_indices,
    _get_value_col_from_value_type,
    evaluate_policy_local_2d_sweeps,
)


def _select_anchor_plot_df(sweep_2d_df, average_across_anchors=False, anchor_id=None):
    plot_df = sweep_2d_df.copy()

    if average_across_anchors:
        return plot_df, anchor_id

    if anchor_id is None:
        anchor_id = int(np.sort(plot_df['anchor_id'].unique())[0])

    plot_df = plot_df[plot_df['anchor_id'] == anchor_id].copy()
    return plot_df, anchor_id


def _build_2d_stat_pivot(sweep_2d_df, value_col, stat='mean'):
    grouped = (
        sweep_2d_df
        .groupby(['delta_y', 'delta_x'], as_index=False)[value_col]
        .agg(stat)
    )

    pivot_df = grouped.pivot(index='delta_y', columns='delta_x', values=value_col)
    return pivot_df.sort_index(axis=0).sort_index(axis=1)


def _get_2d_axis_labels(plot_df):
    return plot_df['obs_dim_x_label'].iloc[0], plot_df['obs_dim_y_label'].iloc[0]

def _get_2d_plot_title(
    plot_kind,
    value_col,
    x_label,
    y_label,
    average_across_anchors=False,
    anchor_id=None,
):
    pair_str = _format_pair_label(x_label, y_label)
    if average_across_anchors:
        return f'{plot_kind} ({pair_str}) (mean across anchors): {value_col}'
    return f'{plot_kind} ({pair_str}) (anchor {anchor_id}): {value_col}'


def _imshow_pivot(pivot_df, colorbar_label):
    image = plt.imshow(
        pivot_df.values,
        aspect='auto',
        origin='lower',
        extent=[
            pivot_df.columns.min(), pivot_df.columns.max(),
            pivot_df.index.min(), pivot_df.index.max(),
        ],
    )
    plt.colorbar(image, label=colorbar_label)
    return image


def _plot_2d_response_surface(
    sweep_2d_df,
    action_dim=0,
    anchor_id=None,
    value_type='delta_action',
    average_across_anchors=False,
    plot_kind='heatmap',
    n_levels=12,
):
    value_col = _get_value_col_from_value_type(value_type=value_type, action_dim=action_dim)

    plot_df, anchor_id = _select_anchor_plot_df(
        sweep_2d_df=sweep_2d_df,
        average_across_anchors=average_across_anchors,
        anchor_id=anchor_id,
    )
    pivot_df = _build_2d_stat_pivot(plot_df, value_col=value_col, stat='mean')
    x_label, y_label = _get_2d_axis_labels(plot_df)

    plt.figure(figsize=(6, 5))

    if plot_kind == 'heatmap':
        _imshow_pivot(pivot_df, colorbar_label=value_col)
        title = _get_2d_plot_title(
            plot_kind='2-D local response',
            value_col=value_col,
            x_label=x_label,
            y_label=y_label,
            average_across_anchors=average_across_anchors,
            anchor_id=anchor_id,
        )
    elif plot_kind == 'contour':
        x = pivot_df.columns.values.astype(np.float32)
        y = pivot_df.index.values.astype(np.float32)
        z = pivot_df.values.astype(np.float32)
        contour = plt.contourf(x, y, z, levels=n_levels)
        plt.colorbar(contour, label=value_col)
        title = _get_2d_plot_title(
            plot_kind='2-D contour',
            value_col=value_col,
            x_label=x_label,
            y_label=y_label,
            average_across_anchors=average_across_anchors,
            anchor_id=anchor_id,
        )
    else:
        raise ValueError("plot_kind must be 'heatmap' or 'contour'.")

    plt.xlabel(f'delta on {x_label}')
    plt.ylabel(f'delta on {y_label}')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_local_policy_sweep(
    sweep_df,
    obs_dim,
    action_dim=0,
    max_anchors_to_plot=8,
):
    '''
    Plot local response curves for one observation dimension.
    '''
    value_col = f'action_{action_dim}'
    dim_df = sweep_df[sweep_df['obs_dim'] == obs_dim].copy()

    if len(dim_df) == 0:
        raise ValueError(f'No rows found for obs_dim={obs_dim}.')

    plt.figure(figsize=(6, 4))

    anchor_ids = np.sort(dim_df['anchor_id'].unique())[:max_anchors_to_plot]
    for anchor_id in anchor_ids:
        anchor_df = dim_df[dim_df['anchor_id'] == anchor_id].sort_values('delta')
        plt.plot(anchor_df['delta'].values, anchor_df[value_col].values, alpha=0.8)

    plt.axvline(0.0)
    plt.xlabel(f'delta on {dim_df["obs_dim_label"].iloc[0]}')
    plt.ylabel(value_col)
    plt.title(f'Local policy sweep: {dim_df["obs_dim_label"].iloc[0]}')
    plt.tight_layout()
    plt.show()


def plot_local_policy_2d_heatmap(
    sweep_2d_df,
    action_dim=0,
    anchor_id=None,
    value_type='delta_action',
    average_across_anchors=False,
):
    '''
    Plot a 2-D heatmap of local policy response.

    value_type:
        'raw_action' -> action_k
        'delta_action' -> delta_action_k
        'norm' -> action_change_norm
    '''
    _plot_2d_response_surface(
        sweep_2d_df=sweep_2d_df,
        action_dim=action_dim,
        anchor_id=anchor_id,
        value_type=value_type,
        average_across_anchors=average_across_anchors,
        plot_kind='heatmap',
    )


def plot_local_policy_2d_contour(
    sweep_2d_df,
    action_dim=0,
    anchor_id=None,
    value_type='delta_action',
    average_across_anchors=False,
    n_levels=12,
):
    '''
    Same data as heatmap, but contour lines are often better for judging smoothness.
    '''
    _plot_2d_response_surface(
        sweep_2d_df=sweep_2d_df,
        action_dim=action_dim,
        anchor_id=anchor_id,
        value_type=value_type,
        average_across_anchors=average_across_anchors,
        plot_kind='contour',
        n_levels=n_levels,
    )


def plot_local_policy_2d_variability(
    sweep_2d_df,
    action_dim=0,
    value_type='delta_action',
):
    '''
    Plot mean and std across anchors for a given 2-D response quantity.
    '''
    value_col = _get_value_col_from_value_type(value_type=value_type, action_dim=action_dim)

    mean_df = _build_2d_stat_pivot(sweep_2d_df, value_col=value_col, stat='mean')
    std_df = _build_2d_stat_pivot(sweep_2d_df, value_col=value_col, stat='std').fillna(0.0)

    x_label = sweep_2d_df['obs_dim_x_label'].iloc[0]
    y_label = sweep_2d_df['obs_dim_y_label'].iloc[0]
    pair_label = _format_pair_label(x_label, y_label)

    plt.figure(figsize=(6, 5))
    _imshow_pivot(mean_df, colorbar_label=f'mean {value_col}')
    plt.xlabel(f'delta on {x_label}')
    plt.ylabel(f'delta on {y_label}')
    plt.title(f'Mean across anchors ({pair_label}): {value_col}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    _imshow_pivot(std_df, colorbar_label=f'std {value_col}')
    plt.xlabel(f'delta on {x_label}')
    plt.ylabel(f'delta on {y_label}')
    plt.title(f'Across-anchor variability ({pair_label}): {value_col}')
    plt.tight_layout()
    plt.show()


def summarize_local_policy_2d_by_anchor(sweep_2d_df):
    '''
    Summarize 2-D local sensitivity separately for each anchor.
    '''
    summary_df = (
        sweep_2d_df
        .groupby(['anchor_id', 'anchor_index'], as_index=False)
        .agg(
            mean_action_change_norm=('action_change_norm', 'mean'),
            median_action_change_norm=('action_change_norm', 'median'),
            max_action_change_norm=('action_change_norm', 'max'),
        )
    )

    if 'delta_action_0' in sweep_2d_df.columns:
        mean_abs_delta_action_0 = (
            sweep_2d_df
            .groupby(['anchor_id', 'anchor_index'])['delta_action_0']
            .apply(lambda x: np.mean(np.abs(x)))
            .reset_index(name='mean_abs_delta_action_0')
        )
        summary_df = summary_df.merge(mean_abs_delta_action_0, on=['anchor_id', 'anchor_index'], how='left')

    if 'delta_action_1' in sweep_2d_df.columns:
        mean_abs_delta_action_1 = (
            sweep_2d_df
            .groupby(['anchor_id', 'anchor_index'])['delta_action_1']
            .apply(lambda x: np.mean(np.abs(x)))
            .reset_index(name='mean_abs_delta_action_1')
        )
        summary_df = summary_df.merge(mean_abs_delta_action_1, on=['anchor_id', 'anchor_index'], how='left')

    summary_df = summary_df.sort_values(
        'mean_action_change_norm',
        ascending=False,
    ).reset_index(drop=True)

    summary_df['obs_dim_x'] = int(sweep_2d_df['obs_dim_x'].iloc[0])
    summary_df['obs_dim_y'] = int(sweep_2d_df['obs_dim_y'].iloc[0])
    summary_df['obs_dim_x_label'] = sweep_2d_df['obs_dim_x_label'].iloc[0]
    summary_df['obs_dim_y_label'] = sweep_2d_df['obs_dim_y_label'].iloc[0]
    summary_df['pair_label'] = (
        summary_df['obs_dim_x_label'] + '__vs__' + summary_df['obs_dim_y_label']
    )

    return summary_df


def plot_local_policy_2d_views_for_one_pair(
    sweep_2d_df,
    anchor_id=0,
    plot_mean_contours=True,
    anchor_ids_to_plot=None,
    plot_anchor_grids=True,
    plot_variability=True,
    print_anchor_summary=True,
    n_anchor_grid_cols=3,
    top_n_summary=10,
):
    '''
    Standard plotting bundle for one 2-D sweep result.
    '''
    unique_anchor_ids = np.sort(sweep_2d_df['anchor_id'].unique())

    if anchor_ids_to_plot is None:
        anchor_ids_to_plot = unique_anchor_ids[:min(6, len(unique_anchor_ids))]
    else:
        anchor_ids_to_plot = np.asarray(anchor_ids_to_plot, dtype=int)

    if plot_anchor_grids:
        plot_local_policy_2d_anchor_grid(
            sweep_2d_df=sweep_2d_df,
            action_dim=0,
            value_type='delta_action',
            anchor_ids=anchor_ids_to_plot,
            n_cols=n_anchor_grid_cols,
        )
        plot_local_policy_2d_anchor_grid(
            sweep_2d_df=sweep_2d_df,
            action_dim=1,
            value_type='delta_action',
            anchor_ids=anchor_ids_to_plot,
            n_cols=n_anchor_grid_cols,
        )
        plot_local_policy_2d_anchor_grid(
            sweep_2d_df=sweep_2d_df,
            value_type='norm',
            anchor_ids=anchor_ids_to_plot,
            n_cols=n_anchor_grid_cols,
        )

    if plot_mean_contours:
        plot_local_policy_2d_contour(
            sweep_2d_df,
            action_dim=0,
            value_type='delta_action',
            average_across_anchors=True,
        )
        plot_local_policy_2d_contour(
            sweep_2d_df,
            action_dim=1,
            value_type='delta_action',
            average_across_anchors=True,
        )
        plot_local_policy_2d_contour(
            sweep_2d_df,
            value_type='norm',
            average_across_anchors=True,
        )

    if plot_variability:
        plot_local_policy_2d_variability(
            sweep_2d_df=sweep_2d_df,
            action_dim=0,
            value_type='delta_action',
        )
        plot_local_policy_2d_variability(
            sweep_2d_df=sweep_2d_df,
            action_dim=1,
            value_type='delta_action',
        )
        plot_local_policy_2d_variability(
            sweep_2d_df=sweep_2d_df,
            value_type='norm',
        )

    if print_anchor_summary:
        anchor_summary_df = summarize_local_policy_2d_by_anchor(sweep_2d_df)
        print('\nPer-anchor summary:')
        print(anchor_summary_df.head(top_n_summary).to_string(index=False))


def plot_local_policy_2d_views_for_all_pairs(
    sweep_2d_results,
    env,
    anchor_id=0,
    plot_anchor_grids=True,
    plot_mean_contours=False,
    plot_variability=False,
    print_anchor_summary=False,
    **kwargs,
):
    '''
    Iterate through all pairwise 2-D sweep results and plot the standard views.
    '''
    for (obs_dim_x, obs_dim_y), sweep_2d_df in sweep_2d_results.items():
        label_x = analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_x)
        label_y = analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_y)

        print(f'\n===== 2-D local sweep: {label_x} vs {label_y} =====')

        plot_local_policy_2d_views_for_one_pair(
            sweep_2d_df=sweep_2d_df,
            anchor_id=anchor_id,
            plot_variability=plot_variability,
            plot_mean_contours=plot_mean_contours,
            plot_anchor_grids=plot_anchor_grids,
            print_anchor_summary=print_anchor_summary,
            **kwargs,
        )


def plot_local_policy_2d_views_for_selected_pairs(
    sweep_2d_results,
    pair_summary_df,
    anchor_id=0,
    plot_anchor_grids=True,
    plot_mean_contours=False,
    plot_variability=False,
    print_anchor_summary=False,
    anchor_ids_to_plot=None,
    n_anchor_grid_cols=3,
    top_n_summary=10,
):
    '''
    Plot standard views for the selected structured pairs.
    '''
    if len(pair_summary_df) == 0:
        print('No selected pairs to plot.')
        return

    for _, row in pair_summary_df.iterrows():
        obs_dim_x = int(row['obs_dim_x'])
        obs_dim_y = int(row['obs_dim_y'])
        pair_category = row['pair_category']
        label_x = row['obs_dim_x_label']
        label_y = row['obs_dim_y_label']

        print(f'\n===== 2-D local sweep: {label_x} vs {label_y} [{pair_category}] =====')

        sweep_2d_df = sweep_2d_results[(obs_dim_x, obs_dim_y)]

        plot_local_policy_2d_views_for_one_pair(
            sweep_2d_df=sweep_2d_df,
            anchor_id=anchor_id,
            plot_mean_contours=plot_mean_contours,
            anchor_ids_to_plot=anchor_ids_to_plot,
            plot_anchor_grids=plot_anchor_grids,
            plot_variability=plot_variability,
            print_anchor_summary=print_anchor_summary,
            n_anchor_grid_cols=n_anchor_grid_cols,
            top_n_summary=top_n_summary,
        )


def run_and_plot_local_policy_2d_for_one_pair(
    env,
    rl_agent,
    rollout_df,
    obs_array,
    obs_dim_x,
    obs_dim_y,
    n_anchors=12,
    anchor_method='cluster',
    random_seed=42,
    delta_x_values=None,
    delta_y_values=None,
    deterministic=True,
    anchor_id=0,
    anchor_ids_to_plot=None,
    plot_anchor_grids=True,
    plot_mean_contours=True,
    plot_variability=True,
    print_anchor_summary=True,
    n_anchor_grid_cols=3,
    top_n_summary=10,
    standardize_for_clustering=True,
):
    '''
    Convenience wrapper:
    - choose anchors
    - run 2-D sweep
    - plot standard views
    '''
    anchor_indices = _choose_anchor_indices(
        rollout_df=rollout_df,
        obs_array=obs_array,
        n_anchors=n_anchors,
        anchor_method=anchor_method,
        random_seed=random_seed,
        standardize_for_clustering=standardize_for_clustering,
    )

    sweep_2d_df = evaluate_policy_local_2d_sweeps(
        env=env,
        rl_agent=rl_agent,
        obs_array=obs_array,
        anchor_indices=anchor_indices,
        obs_dim_x=obs_dim_x,
        obs_dim_y=obs_dim_y,
        delta_x_values=delta_x_values if delta_x_values is not None else np.linspace(-0.15, 0.15, 21, dtype=np.float32),
        delta_y_values=delta_y_values if delta_y_values is not None else np.linspace(-0.15, 0.15, 21, dtype=np.float32),
        deterministic=deterministic,
    )

    label_x = analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_x)
    label_y = analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_y)
    print(f'\n===== 2-D local sweep: {label_x} vs {label_y} =====')

    plot_local_policy_2d_views_for_one_pair(
        sweep_2d_df=sweep_2d_df,
        anchor_id=anchor_id,
        plot_mean_contours=plot_mean_contours,
        anchor_ids_to_plot=anchor_ids_to_plot,
        plot_anchor_grids=plot_anchor_grids,
        plot_variability=plot_variability,
        print_anchor_summary=print_anchor_summary,
        n_anchor_grid_cols=n_anchor_grid_cols,
        top_n_summary=top_n_summary,
    )

    return sweep_2d_df, anchor_indices


def _format_pair_label(x_label, y_label):
    '''
    Format two obs dim labels as 'prefix: a vs b' when they share a prefix.
    Example:
        slot_0_d_log, slot_0_sin -> slot_0: d_log vs sin
    '''
    i = 0
    while i < min(len(x_label), len(y_label)) and x_label[i] == y_label[i]:
        i += 1

    common_prefix = x_label[:i]
    if common_prefix.endswith('_'):
        prefix = common_prefix.rstrip('_')
        x_suffix = x_label[i:].lstrip('_')
        y_suffix = y_label[i:].lstrip('_')
        if prefix and (x_suffix or y_suffix):
            return f'{prefix}: {x_suffix} vs {y_suffix}'

    return f'{x_label} vs {y_label}'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _format_pair_label(x_label, y_label):
    '''
    Format two obs dim labels as 'prefix: a vs b' when they share a prefix.
    Example:
        slot_0_d_log, slot_0_sin -> slot_0: d_log vs sin
    '''
    i = 0
    while i < min(len(x_label), len(y_label)) and x_label[i] == y_label[i]:
        i += 1

    common_prefix = x_label[:i]
    if common_prefix.endswith('_'):
        prefix = common_prefix.rstrip('_')
        x_suffix = x_label[i:].lstrip('_')
        y_suffix = y_label[i:].lstrip('_')
        if prefix and (x_suffix or y_suffix):
            return f'{prefix}: {x_suffix} vs {y_suffix}'

    return f'{x_label} vs {y_label}'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


def _format_pair_label(x_label, y_label):
    '''
    Format two obs dim labels as 'prefix: a vs b' when they share a prefix.
    Example:
        slot_0_d_log, slot_0_sin -> slot_0: d_log vs sin
    '''
    i = 0
    while i < min(len(x_label), len(y_label)) and x_label[i] == y_label[i]:
        i += 1

    common_prefix = x_label[:i]
    if common_prefix.endswith('_'):
        prefix = common_prefix.rstrip('_')
        x_suffix = x_label[i:].lstrip('_')
        y_suffix = y_label[i:].lstrip('_')
        if prefix and (x_suffix or y_suffix):
            return f'{prefix}: {x_suffix} vs {y_suffix}'

    return f'{x_label} vs {y_label}'

def _collect_panel_pivots_and_value_range(
    panel_source_items,
    value_col,
):
    '''
    Build one pivot per panel and compute a shared finite value range.

    Parameters
    ----------
    panel_source_items : list of tuple
        Each tuple is (panel_id, panel_df), where panel_df is already filtered
        to the subset that should populate that panel.
    value_col : str
        Column to aggregate into the 2D pivot.

    Returns
    -------
    pivot_list : list of tuple
        Each tuple is (panel_id, pivot_df).
    vmin : float
        Global finite minimum across all pivots.
    vmax : float
        Global finite maximum across all pivots.
    '''
    pivot_list = []
    vmin = np.inf
    vmax = -np.inf

    for panel_id, panel_df in panel_source_items:
        if len(panel_df) == 0:
            continue

        pivot_df = _build_2d_stat_pivot(panel_df, value_col=value_col, stat='mean')
        pivot_list.append((panel_id, pivot_df))

        finite_values = pivot_df.values[np.isfinite(pivot_df.values)]
        if finite_values.size > 0:
            vmin = min(vmin, finite_values.min())
            vmax = max(vmax, finite_values.max())

    return pivot_list, vmin, vmax


def _prepare_shared_2d_grid_plot_config(
    vmin,
    vmax,
    center_zero=True,
    symmetric_color_scale=True,
    cmap=None,
):
    '''
    Prepare shared color scaling and colormap config for a panel grid.
    '''
    if symmetric_color_scale and (vmin < 0) and (vmax > 0):
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    use_centered_norm = center_zero and (vmin < 0) and (vmax > 0)

    if cmap is None:
        cmap = 'coolwarm' if use_centered_norm else 'viridis'

    if use_centered_norm:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        plot_vmin = None
        plot_vmax = None
    else:
        norm = None
        plot_vmin = vmin
        plot_vmax = vmax

    return {
        'vmin': vmin,
        'vmax': vmax,
        'cmap': cmap,
        'norm': norm,
        'plot_vmin': plot_vmin,
        'plot_vmax': plot_vmax,
    }


def _draw_2d_grid_panel(
    ax,
    pivot_df,
    panel_idx,
    n_rows,
    n_cols,
    panel_title,
    x_label,
    y_label,
    cmap,
    norm,
    plot_vmin,
    plot_vmax,
    show_zero_lines=True,
    show_cell_grid=False,
    annotate=False,
    annotation_fmt='{:.2f}',
    square_cells=False,
    interpolation='nearest',
    outer_labels_only=True,
    max_n_ticks=5,
    tick_fmt='%.2f',
):
    '''
    Draw one heatmap panel and return the image handle.
    '''
    x_vals = pivot_df.columns.to_numpy(dtype=float)
    y_vals = pivot_df.index.to_numpy(dtype=float)
    z_vals = pivot_df.values

    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())

    image_handle = ax.imshow(
        z_vals,
        aspect='equal' if square_cells else 'auto',
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        norm=norm,
        vmin=plot_vmin,
        vmax=plot_vmax,
        interpolation=interpolation,
    )

    row_idx = panel_idx // n_cols
    col_idx = panel_idx % n_cols

    if outer_labels_only:
        if row_idx == n_rows - 1:
            ax.set_xlabel(f'delta on {x_label}')
        if col_idx == 0:
            ax.set_ylabel(f'delta on {y_label}')
    else:
        ax.set_xlabel(f'delta on {x_label}')
        ax.set_ylabel(f'delta on {y_label}')

    ax.set_title(panel_title, fontsize=12, pad=6)

    if show_zero_lines:
        if x_min <= 0 <= x_max:
            ax.axvline(0, color='white', lw=0.8, alpha=0.6)
        if y_min <= 0 <= y_max:
            ax.axhline(0, color='white', lw=0.8, alpha=0.6)

    if show_cell_grid and len(x_vals) > 1 and len(y_vals) > 1:
        x_step = np.median(np.diff(np.sort(x_vals)))
        y_step = np.median(np.diff(np.sort(y_vals)))

        x_edges = np.concatenate([
            [x_vals[0] - x_step / 2],
            x_vals[:-1] + x_step / 2,
            [x_vals[-1] + x_step / 2],
        ])
        y_edges = np.concatenate([
            [y_vals[0] - y_step / 2],
            y_vals[:-1] + y_step / 2,
            [y_vals[-1] + y_step / 2],
        ])

        for x_edge in x_edges:
            ax.axvline(x_edge, color='white', lw=0.4, alpha=0.18, zorder=3)
        for y_edge in y_edges:
            ax.axhline(y_edge, color='white', lw=0.4, alpha=0.18, zorder=3)

    if annotate:
        for row_i, y_val in enumerate(y_vals):
            for col_j, x_val in enumerate(x_vals):
                cell_value = z_vals[row_i, col_j]
                if np.isfinite(cell_value):
                    ax.text(
                        x_val,
                        y_val,
                        annotation_fmt.format(cell_value),
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='black',
                    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_n_ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
    ax.tick_params(labelsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.7)

    return image_handle


def _finalize_2d_grid_figure(
    fig,
    axes,
    n_panels,
    image_handle,
    suptitle,
    colorbar_label,
    colorbar_shrink=0.92,
):
    '''
    Hide unused axes, add suptitle and shared colorbar, then show.
    '''
    for ax in axes.ravel()[n_panels:]:
        ax.axis('off')

    fig.suptitle(suptitle, fontsize=14)

    if image_handle is not None:
        cbar = fig.colorbar(
            image_handle,
            ax=axes.ravel().tolist(),
            shrink=colorbar_shrink,
            pad=0.02,
        )
        cbar.set_label(colorbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    plt.show()


def plot_local_policy_2d_anchor_grid(
    sweep_2d_df,
    action_dim=0,
    value_type='norm',
    anchor_ids=None,
    n_cols=3,
    figsize_per_panel=(4.2, 3.8),
    cmap=None,
    center_zero=True,
    symmetric_color_scale=True,
    show_zero_lines=True,
    show_cell_grid=False,
    annotate=False,
    annotation_fmt='{:.2f}',
    square_cells=False,
    interpolation='nearest',
    outer_labels_only=True,
    max_n_ticks=5,
    tick_fmt='%.2f',
    colorbar_shrink=0.92,
):
    '''
    Plot multiple anchors in a single grid with a shared color scale.
    '''
    value_col = _get_value_col_from_value_type(
        value_type=value_type,
        action_dim=action_dim,
    )

    all_anchor_ids = np.sort(sweep_2d_df['anchor_id'].unique())
    if anchor_ids is None:
        anchor_ids = all_anchor_ids

    anchor_id_set = set(all_anchor_ids.tolist())
    anchor_ids = [
        int(anchor_id)
        for anchor_id in np.asarray(anchor_ids, dtype=int)
        if anchor_id in anchor_id_set
    ]

    if len(anchor_ids) == 0:
        raise ValueError('No valid anchor_ids to plot.')

    panel_source_items = [
        (
            anchor_id,
            sweep_2d_df[sweep_2d_df['anchor_id'] == anchor_id].copy(),
        )
        for anchor_id in anchor_ids
    ]

    pivot_list, vmin, vmax = _collect_panel_pivots_and_value_range(
        panel_source_items=panel_source_items,
        value_col=value_col,
    )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f'No finite values found in column {value_col}.')

    n_panels = len(pivot_list)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    x_label = sweep_2d_df['obs_dim_x_label'].iloc[0]
    y_label = sweep_2d_df['obs_dim_y_label'].iloc[0]
    pair_label = _format_pair_label(x_label, y_label)

    plot_config = _prepare_shared_2d_grid_plot_config(
        vmin=vmin,
        vmax=vmax,
        center_zero=center_zero,
        symmetric_color_scale=symmetric_color_scale,
        cmap=cmap,
    )

    image_handle = None

    for panel_idx, (ax, (anchor_id, pivot_df)) in enumerate(zip(axes.ravel(), pivot_list)):
        image_handle = _draw_2d_grid_panel(
            ax=ax,
            pivot_df=pivot_df,
            panel_idx=panel_idx,
            n_rows=n_rows,
            n_cols=n_cols,
            panel_title=f'anchor {anchor_id}',
            x_label=x_label,
            y_label=y_label,
            cmap=plot_config['cmap'],
            norm=plot_config['norm'],
            plot_vmin=plot_config['plot_vmin'],
            plot_vmax=plot_config['plot_vmax'],
            show_zero_lines=show_zero_lines,
            show_cell_grid=show_cell_grid,
            annotate=annotate,
            annotation_fmt=annotation_fmt,
            square_cells=square_cells,
            interpolation=interpolation,
            outer_labels_only=outer_labels_only,
            max_n_ticks=max_n_ticks,
            tick_fmt=tick_fmt,
        )

    _finalize_2d_grid_figure(
        fig=fig,
        axes=axes,
        n_panels=n_panels,
        image_handle=image_handle,
        suptitle=f'Anchor grid ({pair_label})\n{value_col}',
        colorbar_label=value_col,
        colorbar_shrink=colorbar_shrink,
    )


def plot_same_field_pair_across_slots_for_anchor(
    sweep_2d_results_by_slot,
    anchor_id=0,
    action_dim=0,
    value_type='norm',
    n_cols=3,
    figsize_per_panel=(4.2, 3.8),
    cmap=None,
    center_zero=True,
    symmetric_color_scale=True,
    show_zero_lines=True,
    show_cell_grid=False,
    annotate=False,
    annotation_fmt='{:.2f}',
    square_cells=False,
    interpolation='nearest',
    outer_labels_only=True,
    max_n_ticks=5,
    tick_fmt='%.2f',
    colorbar_shrink=0.92,
):
    '''
    Plot the same field pair across slots for one fixed anchor.
    '''
    value_col = local_policy_2d_core._get_value_col_from_value_type(
        value_type=value_type,
        action_dim=action_dim,
    )

    slot_ids = sorted(sweep_2d_results_by_slot.keys())
    if len(slot_ids) == 0:
        raise ValueError('No slot results provided.')

    panel_source_items = []
    for slot_index in slot_ids:
        sweep_2d_df = sweep_2d_results_by_slot[slot_index]
        anchor_df = sweep_2d_df[sweep_2d_df['anchor_id'] == anchor_id].copy()
        panel_source_items.append((slot_index, anchor_df))

    pivot_list, vmin, vmax = _collect_panel_pivots_and_value_range(
        panel_source_items=panel_source_items,
        value_col=value_col,
    )

    if len(pivot_list) == 0:
        raise ValueError(f'No valid data found for anchor_id={anchor_id}.')
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f'No finite values found in column {value_col}.')

    n_panels = len(pivot_list)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    first_slot_df = sweep_2d_results_by_slot[pivot_list[0][0]]
    x_label = first_slot_df['obs_dim_x_label'].iloc[0]
    y_label = first_slot_df['obs_dim_y_label'].iloc[0]
    pair_label = _format_pair_label(x_label, y_label)

    plot_config = _prepare_shared_2d_grid_plot_config(
        vmin=vmin,
        vmax=vmax,
        center_zero=center_zero,
        symmetric_color_scale=symmetric_color_scale,
        cmap=cmap,
    )

    image_handle = None

    for panel_idx, (ax, (slot_index, pivot_df)) in enumerate(zip(axes.ravel(), pivot_list)):
        image_handle = _draw_2d_grid_panel(
            ax=ax,
            pivot_df=pivot_df,
            panel_idx=panel_idx,
            n_rows=n_rows,
            n_cols=n_cols,
            panel_title=f'slot {slot_index}',
            x_label=x_label,
            y_label=y_label,
            cmap=plot_config['cmap'],
            norm=plot_config['norm'],
            plot_vmin=plot_config['plot_vmin'],
            plot_vmax=plot_config['plot_vmax'],
            show_zero_lines=show_zero_lines,
            show_cell_grid=show_cell_grid,
            annotate=annotate,
            annotation_fmt=annotation_fmt,
            square_cells=square_cells,
            interpolation=interpolation,
            outer_labels_only=outer_labels_only,
            max_n_ticks=max_n_ticks,
            tick_fmt=tick_fmt,
        )

    _finalize_2d_grid_figure(
        fig=fig,
        axes=axes,
        n_panels=n_panels,
        image_handle=image_handle,
        suptitle=f'Same field pair across slots ({pair_label})\nanchor {anchor_id}: {value_col}',
        colorbar_label=value_col,
        colorbar_shrink=colorbar_shrink,
    )