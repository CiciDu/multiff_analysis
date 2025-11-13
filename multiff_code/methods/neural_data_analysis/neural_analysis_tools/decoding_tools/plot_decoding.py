import matplotlib.patches as patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_decoding_auc_heatmap(
    df_results,
    value_col='mean_auc',
    sig_col='sig_FDR',
    threshold=0.55,
    cmap='magma',
    figsize=(10, 6),
    annot=True,
    title=None,
    show_sig_outline=True,
    sig_color='limegreen',
    x_tick_position='center',  # 'boundary' or 'center'
):
    auc_df, sig_df, values, time_edges = _prepare_auc_sig_matrices(
        df_results, value_col, sig_col, threshold
    )
    _warn_if_irregular_steps(df_results)

    M, N = values.shape
    Y = np.arange(M + 1)  # row edges

    # ----- compute starts/ends/centers from actual data columns -----
    # (Use mean end per start to be robust to duplicates.)
    end_lookup = df_results.groupby('window_start')['window_end'].mean()
    starts = auc_df.columns.to_numpy(dtype=float)
    ends = end_lookup.reindex(starts).to_numpy()
    # fallback for any missing ends: next edge in the global grid
    if np.isnan(ends).any():
        idx_next = np.searchsorted(time_edges, starts) + 1
        idx_next = np.clip(idx_next, 1, len(time_edges) - 1)
        ends = np.where(np.isnan(ends), time_edges[idx_next], ends)
    centers = (starts + ends) / 2.0

    # ----- choose x-edges for drawing depending on mode -----
    if x_tick_position == 'center':
        # build edges around centers using the median step
        if len(centers) > 1:
            step = np.median(np.diff(centers))
        else:
            step = float(ends[0] - starts[0])
        X_plot = np.concatenate([[centers[0] - step / 2.0],
                                 centers + step / 2.0])
        xticks_centers = centers
        xticks_boundaries = starts  # available if user switches
    elif x_tick_position == 'boundary':
        # draw with true edges from the union of start/end times
        X_plot = time_edges
        xticks_centers = (time_edges[:-1] + time_edges[1:]) / 2.0
        xticks_boundaries = starts
    else:
        raise ValueError("x_tick_position must be 'boundary' or 'center'")

    # ----- plot -----
    fig, ax = plt.subplots(figsize=figsize)
    mesh = _draw_base_heatmap(ax, X_plot, Y, values, cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Decoding AUC')

    if annot:
        _annotate_cells(ax, values, X_plot, auc_df)
    if show_sig_outline:
        _outline_significant_cells(ax, sig_df, values, X_plot, sig_color)

    if len(X_plot) > values.shape[1] + 1:
        X_plot = X_plot[:values.shape[1] + 1]

    # ticks/labels
    _format_axes(
        ax, X_plot, auc_df,
        x_tick_position=x_tick_position,
        x_tick_centers=xticks_centers,
        x_tick_boundaries=xticks_boundaries
    )

    model_name = df_results['model_name'].iloc[0] if 'model_name' in df_results.columns else 'unknown'
    ax.set_title(title or f'Decoding AUC over time ({model_name})')

    plt.tight_layout()
    plt.show()
    return ax


def _warn_if_irregular_steps(df_results, tolerance=1e-9):
    """Warn if window steps are non-uniform or larger than window widths."""
    if 'window_start' not in df_results or 'window_end' not in df_results:
        return

    starts = np.sort(df_results['window_start'].unique())
    if starts.size < 2:
        return
    steps = np.diff(starts)

    # widths from mean end at each start
    ends_mean = df_results.groupby('window_start')['window_end'].mean()
    widths = ends_mean.values - ends_mean.index.values

    # non-uniform step
    if np.max(steps) - np.min(steps) > tolerance:
        print(
            f"[plot_decoding_auc_heatmap] ⚠️ Non-uniform step size. "
            f"min={np.min(steps):.4f}, max={np.max(steps):.4f}"
        )

    mean_width = float(np.nanmean(widths))
    mean_step = float(np.nanmean(steps))
    if mean_step > mean_width + tolerance:
        print(
            f"[plot_decoding_auc_heatmap] ⚠️ Step ({mean_step:.4f}) "
            f"> window width ({mean_width:.4f}) — gaps may appear."
        )
    elif mean_step < mean_width - tolerance:
        print(
            f"[plot_decoding_auc_heatmap] Info: step ({mean_step:.4f}) "
            f"< width ({mean_width:.4f}) — overlapping windows."
        )


def _prepare_auc_sig_matrices(df, value_col, sig_col, threshold):
    df = df.copy()

    # ---- Duplicate detection ----
    dup_mask = df.duplicated(
        subset=['a_label', 'b_label', 'window_start'], keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, ['a_label', 'b_label', 'window_start']]
        raise ValueError(
            "[plot_decoding_auc_heatmap] Found multiple rows for the same "
            "(a_label, b_label, window_start). This would create ambiguity:\n"
            f"{dup_rows.drop_duplicates().to_string(index=False)}"
        )

    # ---- Handle missing sig column ----
    if sig_col not in df.columns or df[sig_col].isna().all():
        print('[plot_decoding_auc_heatmap] No significance column found or all NaN — skipping outlines.')
        df[sig_col] = False

    # ---- Build grid edges ----
    time_edges = np.sort(
        np.unique(df[['window_start', 'window_end']].values.ravel()))
    print('time_edges:', time_edges)
    left_edges = np.sort(np.unique(df['window_start'].values))

    # ---- Pivot to matrix form ----
    auc_df = df.pivot_table(
        index=['a_label', 'b_label'],
        columns='window_start',
        values=value_col,
        aggfunc='mean'
    ).sort_index()

    sig_df = df.pivot_table(
        index=['a_label', 'b_label'],
        columns='window_start',
        values=sig_col,
        aggfunc='first'
    ).sort_index()

    # ---- Label formatting ----
    auc_df.index = [f"{a} vs {b}" for a, b in auc_df.index]
    sig_df.index = auc_df.index

    auc_df = auc_df.reindex(columns=left_edges)
    sig_df = sig_df.reindex(columns=left_edges)

    vals = auc_df.to_numpy(dtype=float)
    mask = np.isnan(vals) | (vals < threshold)
    values = np.ma.array(vals, mask=mask)

    return auc_df, sig_df, values, time_edges


def _draw_base_heatmap(ax, X_edges, Y_edges, values, cmap):
    # X_edges and Y_edges are *cell edges* already
    mesh = ax.pcolormesh(
        X_edges, Y_edges, values,
        vmin=0.5, vmax=1.0,
        cmap=cmap,
        shading='flat',
        edgecolors='white',
        linewidth=0.6,
        antialiased=False
    )
    return mesh


def _annotate_cells(ax, values, X_edges, auc_df):
    # Only annotate actual data columns (values.shape[1])
    x_centers = (X_edges[:-1] + X_edges[1:]) / 2.0
    # Clip to the number of AUC columns
    x_centers = x_centers[:values.shape[1]]

    for i, _ in enumerate(auc_df.index):
        for j, x in enumerate(x_centers):
            if j >= values.shape[1]:
                continue
            if values.mask is not np.ma.nomask and values.mask[i, j]:
                continue
            v = float(values[i, j])
            ax.text(x, i + 0.5, f"{v:.2f}",
                    ha='center', va='center',
                    color='white', fontsize=8)


def _outline_significant_cells(ax, sig_df, values, X_edges, sig_color='black'):
    """Draw clean, subtle frames around significant cells."""
    sig_mask = sig_df.values.astype(bool)
    if values.mask is not np.ma.nomask:
        sig_mask &= ~values.mask
    if not sig_mask.any():
        return

    x_edges = X_edges
    y_edges = np.arange(sig_mask.shape[0] + 1)

    # Use a lighter, thinner outline
    for yi, xi in zip(*np.where(sig_mask)):
        rect = patches.Rectangle(
            (x_edges[xi], y_edges[yi]),
            x_edges[xi+1] - x_edges[xi],
            y_edges[yi+1] - y_edges[yi],
            linewidth=1.5,             # thinner line
            edgecolor=sig_color,
            facecolor='none',
            alpha=0.8,                 # slight transparency
            joinstyle='miter',         # sharp corners
            zorder=10                  # ensures on top of heatmap
        )
        ax.add_patch(rect)


def _format_axes(ax, X_edges, auc_df,
                 x_tick_position='boundary',
                 x_tick_centers=None,
                 x_tick_boundaries=None):
    M = len(auc_df)
    if x_tick_position not in ('boundary', 'center'):
        raise ValueError("x_tick_position must be 'boundary' or 'center'")

    if x_tick_position == 'boundary':
        xticks = np.asarray(
            x_tick_boundaries) if x_tick_boundaries is not None else (X_edges[:-1])
    else:
        xticks = np.asarray(x_tick_centers) if x_tick_centers is not None else (
            (X_edges[:-1] + X_edges[1:]) / 2.0)

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2f}" for t in xticks])
    ax.set_xlim(X_edges[0], X_edges[-1])

    ax.set_yticks(np.arange(M) + 0.5)
    ax.set_yticklabels(list(auc_df.index))
    ax.set_ylim(0, M)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Comparison (A vs B)')


# ---------------------------------------------------------------------
#  (The timecourse functions below are unchanged except a minor warning fix)
# ---------------------------------------------------------------------

def _normalize_cols(cols):
    if isinstance(cols, str):
        return [cols]
    elif isinstance(cols, tuple):
        return list(cols)
    elif isinstance(cols, list):
        return cols
    else:
        raise ValueError(f"Invalid column specifier: {cols}")


def _aggregate_for_plot(df_sub, groupby_cols, value_col, sig_col,
                        err_col=None, err_type='sem'):
    if 'window_center' not in df_sub.columns:
        raise ValueError(
            "Expected 'window_center' in dataframe before aggregation.")

    agg_dict = {value_col: ['mean', 'std', 'count'], sig_col: 'any'}
    err_lookup = (
        df_sub.groupby(groupby_cols + ['window_center']
                       )[err_col].mean().reset_index()
    ) if (err_col is not None and err_col in df_sub.columns) else None

    df_agg = df_sub.groupby(
        groupby_cols + ['window_center'], as_index=False).agg(agg_dict)
    df_agg.columns = groupby_cols + \
        ['window_center', 'mean', 'std', 'count', 'sig']

    if err_lookup is not None:
        df_agg = df_agg.merge(err_lookup, on=groupby_cols +
                              ['window_center'], how='left')
        df_agg['err'] = df_agg[err_col]
    else:
        df_agg['err'] = df_agg['std'] / \
            np.sqrt(df_agg['count']) if err_type == 'sem' else df_agg['std']
        print(
            f"[plot_decoding_timecourse] err_col '{err_col}' not found — using std/√N as {err_type}.")

    if df_agg['err'].isna().all():
        print(
            f"[Warning] All error values NaN for group={groupby_cols} — check err_col='{err_col}'")
    return df_agg


def _plot_subplot(ax, df_agg, groupby_cols, palette, show_sig, sig_color,
                  chance_level, label_prefix):
    unique_groups = df_agg[groupby_cols].drop_duplicates()
    colors = sns.color_palette(palette, len(unique_groups))

    for (_, group), color in zip(unique_groups.iterrows(), colors):
        mask = np.ones(len(df_agg), dtype=bool)
        for c in groupby_cols:
            mask &= df_agg[c] == group[c]
        dfg = df_agg.loc[mask].sort_values('window_center')

        label = ' | '.join([f"{c}={group[c]}" for c in groupby_cols])
        ax.plot(dfg['window_center'], dfg['mean'],
                lw=2, color=color, label=label)
        ax.fill_between(
            dfg['window_center'],
            dfg['mean'] - dfg['err'],
            dfg['mean'] + dfg['err'],
            color=color, alpha=0.3
        )

        if show_sig and dfg['sig'].any():
            sig_points = dfg.loc[dfg['sig'], 'window_center']
            ax.scatter(sig_points, dfg.loc[dfg['sig'], 'mean'],
                       color=color, s=40, edgecolors='k', zorder=5, marker='o')

    ax.axhline(chance_level, color='gray', lw=1, ls='--', label='Chance')
    ax.axvline(0, color='black', lw=1, ls='-')
    ax.set_title(f"{label_prefix}")
    ax.set_xlabel('Time window center (s)')
    ax.set_ylabel('Decoding AUC')
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, frameon=False)

    all_x = np.sort(df_agg['window_center'].unique())
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{x:.2f}" for x in all_x])


# -------------------------
# Shared helper functions
# -------------------------
def _prepare_timecourse_df(df, value_col, sig_col, chance_level, alpha, align_col):
    df = df.copy()
    if 'window_end' in df.columns:
        df['window_center'] = df['window_start']
        mask_end = df['window_end'].notna()
        df.loc[mask_end, 'window_center'] = (
            df.loc[mask_end, 'window_start'] + df.loc[mask_end, 'window_end']
        ) / 2.0
    else:
        print("[timecourse] 'window_end' not found — using window_start as center.")
        df['window_center'] = df['window_start']

    if sig_col not in df.columns or df[sig_col].isna().all():
        print(
            f"[timecourse] No '{sig_col}' found — using AUC > {chance_level + alpha:.2f} as proxy.")
        df[sig_col] = df[value_col] > (chance_level + alpha)

    if align_col not in df.columns:
        print(
            f"[timecourse] '{align_col}' not found — treating as single alignment.")
        df[align_col] = False
    return df


def _title_for_alignment(align_val):
    return 'Align by stop end' if align_val else 'Align by stop start'


def _make_subplots_grid(n_plots, base_width, base_height, sharey=True):
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * base_width, n_rows * base_height),
        sharey=sharey
    )
    axes = np.array(axes).reshape(-1)
    return fig, axes, n_rows, n_cols


def _append_sample_size_to_label(df_sub, label_prefix):
    if 'sample_size' in df_sub.columns:
        non_null_sizes = df_sub['sample_size'].dropna().to_numpy()
        if non_null_sizes.size > 0:
            unique_sizes = np.unique(non_null_sizes)
            try:
                sample_n = int(unique_sizes[0]) if unique_sizes.size == 1 else int(
                    np.nanmax(non_null_sizes))
                return f"{label_prefix} (n={sample_n})"
            except Exception:
                return label_prefix
    return label_prefix


def _save_or_show(fig, save_path, dpi):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'[write] Plot saved → {save_path}')
    else:
        plt.show()


def plot_decoding_timecourse(
    df_results,
    groupby_cols='key',
    align_col='align_by_stop_end',
    value_col='mean_auc',
    sig_col='sig_FDR',
    err_col='sd_auc',
    err_type='sem',
    alpha=0.05,
    figsize=(8, 5),
    palette='magma',
    show_sig=True,
    sig_color='limegreen',
    chance_level=0.5,
    title_prefix='Decoding timecourse',
    split_by='key',
    save_path=None,          # ← new argument
    dpi=300                  # ← optional, for high-res saving
):
    """Plot decoding AUC timecourses, optionally saving to file.

    Args:
        df_results: DataFrame containing decoding results.
        groupby_cols: Columns to group lines (models) within each subplot.
        align_col: Column specifying alignment condition (e.g. by stop start/end).
        value_col: Column with mean AUC or metric to plot.
        sig_col: Column marking significant time bins.
        err_col: Column with error values (sd or sem).
        err_type: Error type label (for plotting, default 'sem').
        alpha: Significance threshold or proxy if no sig_col present.
        figsize: Tuple defining base subplot size.
        palette: Matplotlib/seaborn color palette for lines.
        show_sig: Whether to mark significant bins visually.
        sig_color: Color used to mark significant bins.
        chance_level: Baseline AUC level for reference.
        title_prefix: Title text prefix.
        split_by: Columns to define subplot grouping.
        save_path: Optional path to save the figure (PNG/PDF/etc.).
        dpi: Resolution for saved figure.

    Returns:
        fig: The Matplotlib Figure object.
    """
    groupby_cols = _normalize_cols(groupby_cols)
    split_by = _normalize_cols(split_by)

    df = df_results.copy()

    if 'window_start' not in df.columns:
        raise ValueError("df_results must include 'window_start'.")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in df_results.")
    df = _prepare_timecourse_df(
        df, value_col, sig_col, chance_level, alpha, align_col)

    sns.set(style='whitegrid')

    for align_val in df[align_col].unique():
        dfa = df[df[align_col] == align_val].copy()
        title_align = _title_for_alignment(align_val)

        subplot_groups = dfa.groupby(split_by)
        n_plots = len(subplot_groups)
        if n_plots == 0:
            print("[plot_decoding_timecourse] No subplot groups found — skipping.")
            return None
        fig, axes, _, _ = _make_subplots_grid(
            n_plots, base_width=5, base_height=4, sharey=True)

        for i, (group_keys, df_sub) in enumerate(subplot_groups):
            ax = axes[i]
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            label_prefix = ' | '.join(
                [f"{k}={v}" for k, v in zip(split_by, group_keys)])
            label_prefix = _append_sample_size_to_label(df_sub, label_prefix)

            df_agg = _aggregate_for_plot(
                df_sub, groupby_cols, value_col, sig_col,
                err_col=err_col, err_type=err_type
            )
            _plot_subplot(
                ax, df_agg, groupby_cols, palette,
                show_sig, sig_color, chance_level, label_prefix
            )

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{title_prefix} ({title_align})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        _save_or_show(fig, save_path, dpi)

        return fig


def plot_decoding_timecourse_by_session(
    df_results,
    key_value=None,
    key_col='key',
    session_col='session',
    align_col='align_by_stop_end',
    value_col='mean_auc',
    sig_col='sig_FDR',
    err_col='sd_auc',
    err_type='sem',
    alpha=0.05,
    sessions_per_subplot=6,
    palette='tab20',
    figsize=(5, 4),
    show_sig=True,
    sig_color='limegreen',
    chance_level=0.5,
    title_prefix='Decoding timecourse by session',
    save_path=None,
    dpi=300
):
    """Plot decoding timecourses for one or all keys across multiple sessions.

    Sessions are grouped into chunks, with a few sessions plotted per subplot.

    Args:
        df_results: DataFrame containing decoding results.
        key_value: If provided, plot only this key. If None, iterate over all
                   existing keys in df_results[key_col].
        key_col: Column name for the decoding 'key'.
        session_col: Column name for session identifier.
        align_col: Column specifying alignment condition.
        value_col: Column with mean AUC or metric to plot.
        sig_col: Column marking significant time bins.
        err_col: Column with error values (sd or sem).
        err_type: Error type label ('sem' or 'std').
        alpha: Used to threshold significance proxy if sig_col missing.
        sessions_per_subplot: Number of session lines per subplot.
        palette: Color palette for session lines (should support many colors).
        figsize: Per-subplot figure size (width, height).
        show_sig: Whether to mark significant bins visually.
        sig_color: Color used to mark significant bins.
        chance_level: Baseline AUC level for reference.
        title_prefix: Title text prefix.
        save_path: Optional path to save the figure (PNG/PDF/etc.).
        dpi: Resolution for saved figure.

    Returns:
        If a single figure is produced, returns that Figure.
        If multiple figures are produced (multiple keys and/or alignments),
        returns a list of Figure objects.
    """
    required_cols = set(
        ['window_start', key_col, session_col, value_col, align_col])
    for c in required_cols:
        if c not in df_results.columns:
            raise ValueError(f"df_results must include '{c}'.")

    df = df_results.copy()

    # Determine keys to plot
    if key_value is not None:
        keys_to_plot = [key_value]
    else:
        keys_to_plot = list(pd.unique(df[key_col].dropna()))
        keys_to_plot.sort(key=lambda x: str(x))
        if len(keys_to_plot) == 0:
            raise ValueError(f"No valid '{key_col}' values found to plot.")

    # Shared preprocessing
    df = _prepare_timecourse_df(
        df, value_col, sig_col, chance_level, alpha, align_col)

    sns.set(style='whitegrid')

    produced_figs = []

    # Plot per key and alignment mode
    for one_key in keys_to_plot:
        df_key = df[df[key_col] == one_key].copy()
        if df_key.empty:
            continue

        for align_val in df_key[align_col].unique():
            dfa = df_key[df_key[align_col] == align_val].copy()
            if dfa.empty:
                continue

            # Determine sessions and chunk them
            sessions = list(pd.unique(dfa[session_col]))
            sessions = [s for s in sessions if pd.notna(s)]
            sessions.sort(key=lambda x: str(x))
            if len(sessions) == 0:
                print("[plot_decoding_timecourse_by_session] No sessions found.")
                continue

            session_chunks = [sessions[i:i + sessions_per_subplot]
                              for i in range(0, len(sessions), sessions_per_subplot)]
            n_plots = len(session_chunks)

            fig, axes, _, _ = _make_subplots_grid(
                n_plots, base_width=figsize[0], base_height=figsize[1], sharey=True
            )

            title_align = _title_for_alignment(align_val)

            for i, chunk in enumerate(session_chunks):
                ax = axes[i]
                df_sub = dfa[dfa[session_col].isin(chunk)].copy()

                # Build informative label: key and session list
                if len(chunk) <= 6:
                    session_label = ", ".join([str(s) for s in chunk])
                else:
                    session_label = f"{len(chunk)} sessions"
                label_prefix = f"{key_col}={one_key} | {session_col}s: {session_label}"

                label_prefix = _append_sample_size_to_label(
                    df_sub, label_prefix)

                df_agg = _aggregate_for_plot(
                    df_sub,
                    groupby_cols=[session_col],
                    value_col=value_col,
                    sig_col=sig_col,
                    err_col=err_col,
                    err_type=err_type
                )

                _plot_subplot(
                    ax,
                    df_agg=df_agg,
                    groupby_cols=[session_col],
                    palette=palette,
                    show_sig=show_sig,
                    sig_color=sig_color,
                    chance_level=chance_level,
                    label_prefix=label_prefix
                )

            # Hide any extra axes
            last_i = i if n_plots > 0 else -1
            for j in range(last_i + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f"{title_prefix} ({title_align})", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save or display
            if save_path:
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Avoid overwrite by suffixing key and alignment when iterating
                if key_value is None or len(keys_to_plot) > 1:
                    stem, suffix = out_path.stem, out_path.suffix
                    safe_key = str(one_key).replace(
                        '/', '_').replace('\\', '_').replace(' ', '_')
                    align_suffix = 'end' if align_val else 'start'
                    out_path = out_path.parent / \
                        f"{stem}_{safe_key}_{align_suffix}{suffix}"
                fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
                print(f'[write] Plot saved → {out_path}')
            else:
                plt.show()

            produced_figs.append(fig)

    if len(produced_figs) == 0:
        return None
    if len(produced_figs) == 1:
        return produced_figs[0]
    return produced_figs
