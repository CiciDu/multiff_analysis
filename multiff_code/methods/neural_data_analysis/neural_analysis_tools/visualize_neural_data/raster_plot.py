from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.patches as mpatches

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

events_to_plot_dict = {
    'rel_new_seg_start_time': ['Segment start time', 'blue'],
    'rel_new_seg_end_time': ['Segment end time', 'purple'],
    'rel_stop_time': ['Stop time', 'green'],
    'rel_prev_ff_caught_time': ['Prev FF caught time', 'orange'],
    'rel_event_time': ['Event time', 'red'],
    'rel_stop_id_end_time': ['Stop end time', 'brown'],
    'rel_next_stop_time': ['Next stop time', 'cyan'],
}


def prepare_aligned_spike_trains(new_seg_info, spikes_df,
                                 extra_columns_to_preserve=['event_time', 'stop_time', 'prev_ff_caught_time']):
    """
    Aligns spike times to new segment information and appends event-related timing columns.

    Parameters:
        new_seg_info (DataFrame): Segment information; must contain new_seg_start_time, new_seg_end_time, and new_segment columns
        spikes_df (DataFrame): Raw spike times with segment IDs.
        bin_width (float): Width of the bin used for alignment in `concat_new_seg_info`.

    Returns:
        DataFrame: A DataFrame of aligned spike times with additional timing context.
    """
    # drop rows where new_seg_start_time or new_seg_send_time is NA, and print the number of rows dropped
    original_len = len(new_seg_info)
    new_seg_info = new_seg_info.dropna(
        subset=['new_seg_start_time', 'new_seg_end_time']).reset_index(drop=True)
    print(f'Dropped {original_len - len(new_seg_info)} rows out of {original_len} due to NA in new_seg_start_time or new_seg_end_time, '
          f'which is {(original_len - len(new_seg_info))/original_len*100:.2f}% of the original data')

    # needs to update new_seg_duration in case of new_seg_start_time and new_seg_end_time have been changed
    new_seg_info['new_seg_duration'] = new_seg_info['new_seg_end_time'] - \
        new_seg_info['new_seg_start_time']

    # assert that new_seg_duration is positive
    assert np.all(new_seg_info['new_seg_duration'] >
                  0), 'new_seg_duration must be positive'

    # Align spikes to new segment info
    aligned_spike_trains = pn_utils.concat_new_seg_info(
        spikes_df, new_seg_info,
    )

    # make sure columns in extra_columns_to_preserve are unique
    extra_columns_to_preserve = list(dict.fromkeys(extra_columns_to_preserve))
    cols_to_merge = [col for col in extra_columns_to_preserve if (col in new_seg_info.columns) & (col not in aligned_spike_trains.columns)
                     ] + ['new_segment']

    aligned_spike_trains = aligned_spike_trains.merge(
        new_seg_info[cols_to_merge],
        on='new_segment',
        how='left'
    )

    # Rename time column for clarity
    aligned_spike_trains.rename(columns={'time': 'spike_time'}, inplace=True)

    return aligned_spike_trains


def add_relative_times(df, reference_time_col):
    """
    Adds columns for time values relative to a specified reference time.

    If `df` is a list/tuple of DataFrames, the function is applied to each
    and a list of modified DataFrames is returned.

    Parameters:
        df: DataFrame or list/tuple of DataFrames
        reference_time_col (str): Column name to use as the reference time.

    Returns:
        DataFrame or list of DataFrames with added relative time columns.
    """

    # --- Handle list/tuple input by recursive application ---
    if isinstance(df, (list, tuple)):
        return [add_relative_times(d, reference_time_col) for d in df]

    # --- Process a single DataFrame ---
    df['reference_time'] = df[reference_time_col]

    # Columns eligible for conversion to relative times
    time_columns = [
        col for col in df.columns
        if col.endswith('_time')
        and 'rel_' not in col
        and 'reference' not in col
    ]

    for col in time_columns:
        rel_col = f'rel_{col}'
        df[rel_col] = df[col] - df['reference_time']

    return df


def add_scaling_info(
    df,
    scale_anchor_col,
    scale_factor_upper_col='new_seg_end_time',
    scale_factor_lower_col='scale_anchor'
):
    """
    Adds scaled timing columns to one or many DataFrames.

    If `df` is a list/tuple of DataFrames, the function is applied to each
    and a list of scaled DataFrames is returned.

    Parameters:
        df: DataFrame or list/tuple of DataFrames
        scale_anchor_col (str): Column to use as the scale anchor.
        scale_factor_upper_col (str): Upper bound of scaling factor.
        scale_factor_lower_col (str): Lower bound of scaling factor.

    Returns:
        DataFrame or list of DataFrames with added scaling columns.
    """

    # --- If input is a list/tuple, process each DF recursively ---
    if isinstance(df, (list, tuple)):
        return [
            add_scaling_info(
                d,
                scale_anchor_col=scale_anchor_col,
                scale_factor_upper_col=scale_factor_upper_col,
                scale_factor_lower_col=scale_factor_lower_col,
            )
            for d in df
        ]

    # --- Otherwise process a single DataFrame ---
    df['scale_anchor'] = df[scale_anchor_col]

    assert np.all(df[scale_factor_upper_col] > df[scale_factor_lower_col]), \
        'scale_factor_upper_col must be greater than scale_factor_lower_col'

    df['scale_factor'] = df[scale_factor_upper_col] - df[scale_factor_lower_col]

    df['rel_scale_anchor'] = df['scale_anchor'] - df['reference_time']

    # find rel_ columns that aren't scale or factor cols
    rel_time_cols = [
        col for col in df.columns
        if col.startswith('rel_')
        and 'sc_' not in col
        and 'scale_factor' not in col
    ]

    for col in rel_time_cols:
        scaled_col = f'sc_{col}'
        df[scaled_col] = (df[col] - df['rel_scale_anchor']) / \
            df['scale_factor']

    return df


def convert_events_to_new_seg_info(
    events_df: pd.DataFrame,
    segment_col: str = 'new_segment',
    start_col: str = None,
    end_col: str = None,
    event_time_col: str = None,
) -> pd.DataFrame:
    """
    Convert an events_df (one row per segment)
    into a new_seg_info-style DataFrame expected by downstream functions.

    Parameters
    ----------
    events_df : pd.DataFrame
        Source dataframe containing one row per segment/trial.
    segment_col : str, default 'new_segment'
        Column in events_df that identifies the segment id.
    start_col, end_col : str or None
        Absolute-time columns that directly provide segment start/end times.
        If None, and stop_time_cols is provided, start/end are inferred as
        min/max across those stop_time_cols.
    event_time_col, stop_time_col, prev_ff_caught_time_col : str or None
        Absolute-time columns to be copied into standard names.
    stop_time_cols : list[str] or None
        Optional list of absolute-time stop columns; used to infer start/end
        when start_col/end_col are not provided.

    Returns
    -------
    pd.DataFrame with columns:
        - new_segment
        - new_seg_start_time
        - new_seg_end_time
        - event_time (if available, else NaN)
        - stop_time (if available, else NaN)
        - prev_ff_caught_time (if available, else NaN)
    """
    if segment_col not in events_df.columns:
        # try a common fallback name
        if 'segment' in events_df.columns:
            segment_col = 'segment'
        else:
            raise KeyError(
                f"Segment column '{segment_col}' not found in events_df, and no 'segment' fallback present.")

    df = events_df.copy()
    # Initialize output with the standardized segment id
    out = pd.DataFrame({'new_segment': df[segment_col].values})

    # Determine segment start/end
    if start_col is not None and start_col in df.columns:
        out['new_seg_start_time'] = df[start_col].values
    else:
        raise ValueError(f"Start column '{start_col}' not found in events_df")

    if end_col is not None and end_col in df.columns:
        out['new_seg_end_time'] = df[end_col].values
    else:
        raise ValueError(f"End column '{end_col}' not found in events_df")

    if event_time_col is not None and event_time_col in df.columns:
        out['event_time'] = df[event_time_col].values

    return out


def plot_rasters_and_fr(
    aligned_spike_trains, new_seg_info, binned_spikes_df, bin_width,
    cluster_col='cluster',
    bins_per_aggregate=1, plot_mean=True,
    max_clusters_to_plot=None, max_segments_to_plot=None, max_time=None
):
    segments = new_seg_info['new_segment'].unique()
    segments = segments[:max_segments_to_plot] if max_segments_to_plot else segments
    clusters = _prepare_clusters(
        aligned_spike_trains, cluster_col, max_clusters_to_plot)
    if max_time is None:
        max_time = new_seg_info['new_seg_end_time'].max()

    fr_df = plot_neural_data.prepare_fr_data(
        binned_spikes_df, bin_width, bins_per_aggregate, max_time)
    cluster_cols = [col for col in fr_df.columns if col.startswith('cluster_')]
    grouped_spikes = aligned_spike_trains.groupby([cluster_col, 'new_segment'])[
        'rel_spike_time']
    cluster_to_frcols = {int(
        col.split('_')[1]): col for col in cluster_cols if col.split('_')[1].isdigit()}

    for cluster_id in clusters:
        spike_data = {
            seg: grouped_spikes.get_group((cluster_id, seg)).values
            for seg in segments if (cluster_id, seg) in grouped_spikes.groups
        }
        fr_col = cluster_to_frcols.get(cluster_id)
        if fr_col:
            _plot_cluster_raster_and_fr(
                spike_data, segments, fr_df[fr_col], fr_df['time'], cluster_id, plot_mean)

    mean_fr_all = fr_df.mean(axis=1)
    slope = np.polyfit(fr_df['time'], mean_fr_all, 1)[0]
    total_change = mean_fr_all.iloc[-1] - mean_fr_all.iloc[0]
    print(f'Slope: {slope:.4f}, Total change: {total_change:.4f}')


def _plot_cluster_raster_and_fr(spike_data, segments, fr_curve, time, cluster_id, plot_mean):
    fig, (ax_raster, ax_fr) = plt.subplots(1, 2, figsize=(
        12, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})
    num_segments = len(segments)

    for i, seg in enumerate(segments):
        y_pos = num_segments - i
        ax_raster.vlines(spike_data.get(seg, []), y_pos -
                         0.5, y_pos + 0.5, color='black')
    ax_raster.axvline(0, color='red', linestyle='--', alpha=0.2, linewidth=2)
    step = 50
    ticks = np.arange(0, num_segments, step)
    ax_raster.set_yticks(num_segments - ticks)
    ax_raster.set_yticklabels(ticks)
    ax_raster.set_ylim(0.5, num_segments + 0.5)
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Segments")
    ax_raster.set_title(f"Raster: Cluster {cluster_id}")

    ax_fr.plot(time, fr_curve, label=f"Cluster {cluster_id}")
    if plot_mean:
        ax_fr.plot(time, fr_curve, color='black', linewidth=2, alpha=0.6)
    ax_fr.set_xlabel("Time (s)")
    ax_fr.set_ylabel("Firing Rate")
    ax_fr.set_title(f"FR: Cluster {cluster_id}")
    ax_fr.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def _prepare_segments(new_seg_info, max_segments_to_plot):
    """Get unique segment identifiers, truncated to max_segments_to_plot if specified."""
    segments = new_seg_info['new_segment'].unique()
    return segments[:max_segments_to_plot] if max_segments_to_plot else segments


def _prepare_clusters(aligned_spike_trains, cluster_col, max_clusters_to_plot):
    clusters = np.sort(aligned_spike_trains[cluster_col].unique())
    return clusters[:max_clusters_to_plot] if max_clusters_to_plot else clusters


def _plot_segment_event_lines(new_seg_info, column, segments, color, label=None, scale_spike_times=False):
    # Ensure the required column exists; if absolute exists, try creating rel_ using reference_time
    if column not in new_seg_info.columns:
        base_column = column.replace('rel_', '')
        if base_column in new_seg_info.columns:
            if 'reference_time' in new_seg_info.columns:
                new_seg_info[f'rel_{base_column}'] = new_seg_info[base_column] - \
                    new_seg_info['reference_time']
            else:
                print(
                    f"Warning: Column '{column}' not found and no 'reference_time' to derive it in new_seg_info; skipping.")
                return
        else:
            print(
                f"Warning: Column '{column}' not found in new_seg_info; skipping event line plot.")
            return
    unique_events = new_seg_info[[
        'new_segment', column]].drop_duplicates().copy()
    event_values = unique_events.set_index(
        'new_segment').reindex(segments)[column].values
    num_segments = len(segments)
    y_positions = num_segments - np.arange(len(segments))
    # mask out NA values to avoid casting errors and meaningless lines
    try:
        mask = ~pd.isna(event_values)
    except Exception:
        # fallback if pd.isna fails (shouldn't happen)
        mask = np.array([v is not None for v in event_values])
    if mask.sum() == 0:
        return
    event_values = event_values[mask]
    y_positions = y_positions[mask]
    if scale_spike_times and label:
        label = f"Scaled {label.lower()}"
    plt.plot(event_values, y_positions, color=color, label=label)


def _set_xlim(aligned_spike_trains, xmin=None, xmax=None):
    if xmin is None:
        xmin = aligned_spike_trains['rel_new_seg_start_time'].min() - 0.25
    if xmax is None:
        xmax = aligned_spike_trains['rel_new_seg_end_time'].max() + 0.25

    plt.xlim(xmin, xmax)


def _finalize_legend_and_layout(x_lim=None):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize='small', borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.show()


def _plot_raster(spike_data, segments, cluster_id=None, title_prefix="Raster Plot"):
    num_segments = len(segments)

    for i, seg in enumerate(segments):
        spikes = spike_data.get(seg, [])
        y_pos = num_segments - i
        plt.vlines(spikes, y_pos - 0.5, y_pos + 0.5, color='black')

    step = 50
    ytick_positions = np.arange(0, num_segments, step)
    if len(ytick_positions) == 0:
        ytick_positions = [num_segments // 2]
    ytick_labels = ytick_positions
    ytick_positions = num_segments - np.array(ytick_positions)
    plt.yticks(ytick_positions, ytick_labels)

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Event')
    plt.xlabel("Time (s)")
    plt.ylabel("Segments")
    title = f"{title_prefix} {cluster_id}" if cluster_id is not None else title_prefix
    plt.title(title)
    plt.ylim(0.5, num_segments + 0.5)


def _plot_raster_by_yitems(spike_data, y_items, title_id=None, title_prefix="Raster Plot", y_label="Items"):
    """
    Generic raster plotter that draws spike times per y-axis item.
    - spike_data: dict[y_item] -> array-like of spike times
    - y_items: ordered list of y-axis items to display
    """
    num_items = len(y_items)

    for i, item in enumerate(y_items):
        spikes = spike_data.get(item, [])
        y_pos = num_items - i
        plt.vlines(spikes, y_pos - 0.3, y_pos + 0.3, color='black')

    # Choose tick density based on number of items
    if num_items <= 30:
        tick_indices = np.arange(num_items)
    else:
        step = max(1, num_items // 20)  # aim for ~20 ticks max
        tick_indices = np.arange(0, num_items, step)

    ytick_positions = num_items - np.array(tick_indices)
    ytick_labels = [y_items[i] for i in tick_indices]
    plt.yticks(ytick_positions, ytick_labels)

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Event')
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    title = f"{title_prefix} {title_id}" if title_id is not None else title_prefix
    plt.title(title)
    plt.ylim(0.5, num_items + 0.5)


def _resolve_event_window_for_segment(df, segment, event_start_columns, event_end_columns):
    """
    Resolve relative start/end times for an event window for a given segment
    using provided candidate column pairs.
    Parameters
    ----------
    df : pd.DataFrame
        Should contain 'new_segment' and optionally 'reference_time'.
    segment : any
        The segment id to resolve.
    event_start_columns : list[str]
        Candidate start columns, ordered by priority. Can be relative names
        (e.g., 'rel_stop_1_time') or absolute names (e.g., 'stop_1_time').
    event_end_columns : list[str]
        Candidate end columns, ordered by priority. Same rules as start.
        Must be the same length as event_start_columns.
    Returns
    -------
    (start, end) as floats in relative time, or (None, None) if not found.
    """
    assert len(event_start_columns) == len(event_end_columns), \
        "event_start_columns and event_end_columns must have the same length"

    def to_rel_column(df_local, col_name):
        # If already relative and exists
        if col_name.startswith('rel_'):
            if col_name in df_local.columns:
                return col_name
            # Try derive from base if not exists
            base = col_name.replace('rel_', '')
            if base in df_local.columns and 'reference_time' in df_local.columns:
                df_local[col_name] = df_local[base] - \
                    df_local['reference_time']
                return col_name
            return None
        # Absolute column; convert to rel_ if possible
        if col_name in df_local.columns:
            rel_name = f'rel_{col_name}'
            if rel_name not in df_local.columns:
                if 'reference_time' not in df_local.columns:
                    return None
                df_local[rel_name] = df_local[col_name] - \
                    df_local['reference_time']
            return rel_name
        return None

    for start_col_raw, end_col_raw in zip(event_start_columns, event_end_columns):
        start_col = to_rel_column(df, start_col_raw)
        end_col = to_rel_column(df, end_col_raw)
        if start_col is None or end_col is None:
            continue
        # Recompute the row AFTER ensuring columns exist so they are included
        row = (df.loc[df['new_segment'] == segment, ['new_segment', start_col, end_col]]
                 .drop_duplicates('new_segment')
                 .head(1))
        if row.empty:
            continue
        start_val = row.iloc[0][start_col]
        end_val = row.iloc[0][end_col]
        if pd.isna(start_val) or pd.isna(end_val):
            continue
        try:
            start_f = float(start_val)
            end_f = float(end_val)
        except Exception:
            continue
        if end_f <= start_f:
            continue
        return start_f, end_f
    return None, None


def shade_event_windows_for_segments(new_seg_info, segments, event_start_columns,
                                     event_end_columns, colors=None, alpha=0.2):
    """
    Shade per-row (segment) event windows on the current axes.
    - segments: ordered list/array of segment ids matching the y-axis rows
    - event_start_columns, event_end_columns: lists of equal length.
      Each pair defines an event window to shade.
    """
    if not event_start_columns or not event_end_columns or len(segments) == 0:
        return
    num_segments = len(segments)
    ax = plt.gca()
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
            'gray'])

    for e_idx, (start_col, end_col) in enumerate(zip(event_start_columns, event_end_columns)):
        color = colors[e_idx % len(colors)]
        for i, seg in enumerate(segments):
            y_center = num_segments - i
            y0, y1 = y_center - 0.5, y_center + 0.5
            start, end = _resolve_event_window_for_segment(
                new_seg_info, seg, [start_col], [end_col])
            if start is None or end is None or end <= start:
                continue
            rect = mpatches.Rectangle((start, y0), end - start, y1 - y0,
                                      facecolor=color, alpha=alpha, edgecolor=None)
            ax.add_patch(rect)


def shade_event_windows_for_single_segment(new_seg_info, segment, event_start_columns,
                                           event_end_columns, colors=None, alpha=0.2):
    """
    Shade event windows across the full y-range for a single-segment raster
    (e.g., when plotting clusters on the y-axis for one segment).
    """
    if not event_start_columns or not event_end_columns:
        return
    ax = plt.gca()
    ylim = ax.get_ylim()
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
            'gray'])
    for e_idx, (start_col, end_col) in enumerate(zip(event_start_columns, event_end_columns)):
        color = colors[e_idx % len(colors)]
        start, end = _resolve_event_window_for_segment(
            new_seg_info, segment, [start_col], [end_col])
        if start is None or end is None or end <= start:
            continue
        rect = mpatches.Rectangle((start, ylim[0]), end - start, ylim[1] - ylim[0],
                                  facecolor=color, alpha=alpha, edgecolor=None)
        ax.add_patch(rect)

# Ensure relative column exists when only absolute is available


def _ensure_rel_column(df, col):
    if col in df.columns:
        return col
    base_column = col.replace('rel_', '')
    if base_column in df.columns:
        if 'reference_time' in df.columns:
            df[f'rel_{base_column}'] = df[base_column] - \
                df['reference_time']
            return f'rel_{base_column}'
    return None


def _plot_events(new_seg_info, events_to_plot, segments, scale_spike_times):
    # color cycle for unknown events
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        'gray'])

    for idx, event in enumerate(events_to_plot):
        if event in events_to_plot_dict:
            label, color = events_to_plot_dict[event]
            _plot_segment_event_lines(new_seg_info, event, segments,
                                      color=color, label=label, scale_spike_times=scale_spike_times)
            continue
        # fallback for arbitrary events: try to plot if column exists/derivable
        rel_col = _ensure_rel_column(new_seg_info, event)
        if rel_col is None:
            print(
                f"Warning: Event '{event}' not recognized and column not found; skipping.")
            continue
        label = event.replace('rel_', '')
        if scale_spike_times:
            label = f"Scaled {label}"
        color = color_cycle[idx % len(color_cycle)]
        _plot_segment_event_lines(new_seg_info, rel_col, segments,
                                  color=color, label=label, scale_spike_times=scale_spike_times)


def _plot_events_for_single_segment(new_seg_info, segment, events_to_plot, scale_spike_times):
    """
    Plot vertical event lines for a single segment on the current axes.
    """

    for event in events_to_plot:
        if event in events_to_plot_dict:
            label, color = events_to_plot_dict[event]
            col = _ensure_rel_column(new_seg_info, event)
            if col is None:
                print(
                    f"Warning: Column '{event}' not found for segment {segment}; skipping.")
                continue
            row = (new_seg_info.loc[new_seg_info['new_segment'] == segment, ['new_segment', col]]
                   .drop_duplicates()
                   .head(1))
            if row.empty:
                continue
            val = row.iloc[0][col]
            if pd.isna(val):
                continue
            x = float(val)
            plot_label = label
            if scale_spike_times:
                plot_label = f"Scaled {label.lower()}"
            plt.axvline(x=x, color=color, linestyle='-',
                        linewidth=1, label=plot_label)
        else:
            # fallback: arbitrary event columns
            col = _ensure_rel_column(new_seg_info, event)
            if col is None:
                print(
                    f"Warning: Event '{event}' not recognized and column not found; skipping.")
                continue
            row = (new_seg_info.loc[new_seg_info['new_segment'] == segment, ['new_segment', col]]
                   .drop_duplicates()
                   .head(1))
            if row.empty:
                continue
            val = row.iloc[0][col]
            if pd.isna(val):
                continue
            x = float(val)
            label = event.replace('rel_', '')
            if scale_spike_times:
                label = f"Scaled {label}"
            # choose a color from cycle for unknown events
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
                'gray'])
            color = colors[hash(event) % len(colors)]
            plt.axvline(x=x, color=color, linestyle='-',
                        linewidth=1, label=label)


def _plot_generic_events_for_single_segment(new_seg_info, segment, event_cols, scale_spike_times, colors=None):
    """
    Plot arbitrary event columns (not in the predefined dict) for a single segment.
    event_cols: list of column names (absolute or relative) present in aligned_spike_trains
    """
    if not event_cols:
        return

    # Prepare color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
            'gray'])

    for idx, col in enumerate(event_cols):
        rel_col = _ensure_rel_column(new_seg_info, col)
        if rel_col is None:
            print(
                f"Warning: Column '{col}' not found for segment {segment}; skipping.")
            continue
        row = (new_seg_info.loc[new_seg_info['new_segment'] == segment, ['new_segment', rel_col]]
               .drop_duplicates()
               .head(1))
        if row.empty:
            continue
        val = row.iloc[0][rel_col]
        if pd.isna(val):
            continue
        x = float(val)
        color = colors[idx % len(colors)]
        label = col.replace('rel_', '')
        if scale_spike_times:
            label = f"Scaled {label}"
        plt.axvline(x=x, color=color, linestyle='-', linewidth=1, label=label)


def _rearrange_segments(aligned_spike_trains, col_to_rearrange_segments):
    assert col_to_rearrange_segments in aligned_spike_trains, f'{col_to_rearrange_segments} is required in aligned_spike_trains to rearrange by {col_to_rearrange_segments}'
    seg_info = aligned_spike_trains[[
        'new_segment', col_to_rearrange_segments]].drop_duplicates()
    ascending = False if col_to_rearrange_segments == 'num_stops' else True
    seg_info.sort_values(by=col_to_rearrange_segments, inplace=True, ascending=ascending)
    segments = seg_info['new_segment'].unique()
    return segments


def _get_segments(aligned_spike_trains, col_to_rearrange_segments):
    if col_to_rearrange_segments is not None:
        segments = _rearrange_segments(
            aligned_spike_trains, col_to_rearrange_segments)
    else:
        segments = aligned_spike_trains['new_segment'].unique()
    return segments


def _scale_rel_times(aligned_spike_trains):
    # assert that all scale_factor are positive
    assert np.all(
        aligned_spike_trains['scale_factor'] > 0), 'scale_factor must be positive'
    # transform all rel times to be normalized by scale_factor
    rel_time_columns = [
        col for col in aligned_spike_trains.columns if col.startswith('rel_')]
    # make a copy to avoid modifying the original dataframe
    aligned_spike_trains = aligned_spike_trains.copy()
    for col in rel_time_columns:
        aligned_spike_trains[col] = (
            aligned_spike_trains[col] - aligned_spike_trains['rel_scale_anchor']) / aligned_spike_trains['scale_factor']
    return aligned_spike_trains


def plot_rasters(
    aligned_spike_trains,
    cluster_col='cluster',
    title_prefix="Raster Plot for Cluster",
    xmin=None,
    xmax=None,
    col_to_rearrange_segments=None,
    scale_spike_times=False,
    max_clusters_to_plot=None,
    max_segments_to_plot=None,
    new_seg_info=None,
    events_to_plot=(
        'rel_new_seg_start_time',
        'rel_new_seg_end_time',
        'rel_stop_time',
        'rel_prev_ff_caught_time'
    ),
    shade_start_cols=None,
    shade_end_cols=None,
    combine_subplots=False,
):
    """
    Plot raster plots of spike times grouped by cluster and segment.

    Parameters
    ----------
    aligned_spike_trains : pd.DataFrame
        A DataFrame containing spike times and metadata, including relative spike times,
        cluster assignments, and segment identifiers.

    cluster_col : str, default='cluster'
        The name of the column indicating cluster IDs.

    title_prefix : str, default="Raster Plot for Cluster"
        A prefix for the plot title; the cluster ID will be appended.

    xmin : float or None, optional
        Lower limit of the x-axis. If None, determined automatically.

    xmax : float or None, optional
        Upper limit of the x-axis. If None, determined automatically.

    col_to_rearrange_segments : str, default='rel_stop_time'
        The column used to order segments on the y-axis.

    scale_spike_times : bool, default=False
        If True, normalize spike times and event times within each segment.

    max_clusters_to_plot : int or None, optional
        Maximum number of clusters to include in the plot. If None, all clusters are used.

    max_segments_to_plot : int or None, optional
        Maximum number of segments to plot per cluster. If None, all segments are used.

    events_to_plot : list of str, optional
        Column names of events (e.g., start/end times, behavioral markers) to be drawn
        as vertical lines on the plots.

    shade_start_cols, shade_end_cols : list[str] or None
        If provided, shade event windows per row using start/end columns. When
        combine_subplots=True, shading is repeated for each (cluster, segment) row.

    combine_subplots : bool, default=False
        If False (default), produce one figure per cluster.
        If True, squash all per-cluster subplots into a single large raster where each
        row corresponds to a (cluster, segment) pair, ordered by cluster then segment.
        Deprecated: prefer using plot_rasters_combined(...).

    Returns
    -------
    None
        Displays one raster plot per cluster using matplotlib.
    """
    # Normalize spike times if requested
    if scale_spike_times:
        aligned_spike_trains = _scale_rel_times(aligned_spike_trains)

    # Get and optionally truncate list of segments
    segments = _get_segments(aligned_spike_trains, col_to_rearrange_segments)
    if max_segments_to_plot:
        segments = segments[:max_segments_to_plot]

    # Identify clusters to plot
    clusters = _prepare_clusters(
        aligned_spike_trains,
        cluster_col,
        max_clusters_to_plot
    )

    # Group data by cluster and segment
    grouped_spikes = aligned_spike_trains.groupby(
        [cluster_col, 'new_segment']
    )['rel_spike_time']

    if not combine_subplots:
        # Create one plot per cluster (original behavior)
        for cluster_id in clusters:
            plt.figure(figsize=(8, 4))

            spike_data = {
                seg: grouped_spikes.get_group((cluster_id, seg)).values
                for seg in segments
                if (cluster_id, seg) in grouped_spikes.groups
            }

            if shade_start_cols and shade_end_cols:
                shade_event_windows_for_segments(
                    new_seg_info, segments, shade_start_cols, shade_end_cols)

            _plot_raster(spike_data, segments, cluster_id, title_prefix)
            _plot_events(new_seg_info, events_to_plot,
                         segments, scale_spike_times)
            _set_xlim(aligned_spike_trains, xmin, xmax)
            _finalize_legend_and_layout()
        return

    # Delegate to the standalone combined API
    plot_rasters_combined(
        aligned_spike_trains=aligned_spike_trains,
        cluster_col=cluster_col,
        title_prefix=title_prefix,
        xmin=xmin,
        xmax=xmax,
        col_to_rearrange_segments=col_to_rearrange_segments,
        scale_spike_times=scale_spike_times,
        max_clusters_to_plot=max_clusters_to_plot,
        max_segments_to_plot=max_segments_to_plot,
        new_seg_info=new_seg_info,
        events_to_plot=events_to_plot,
        shade_start_cols=shade_start_cols,
        shade_end_cols=shade_end_cols,
    )
    return


def plot_rasters_combined(
    aligned_spike_trains,
    cluster_col='cluster',
    title_prefix="Raster Plot for Cluster",
    xmin=None,
    xmax=None,
    col_to_rearrange_segments=None,
    scale_spike_times=False,
    max_clusters_to_plot=None,
    max_segments_to_plot=None,
    new_seg_info=None,
    events_to_plot=(
        'rel_new_seg_start_time',
        'rel_new_seg_end_time',
        'rel_stop_time',
        'rel_prev_ff_caught_time'
    ),
    shade_start_cols=None,
    shade_end_cols=None,
    draw_cluster_separators=True,
    height_per_row=0.03,
):
    """
    Squash all per-cluster raster subplots into a single large plot.
    Each row corresponds to a (cluster, segment) pair ordered by cluster then segment.
    Parameters
    ----------
    height_per_row : float, default=0.04
        Controls vertical compression. Smaller values squash rows more.
    """
    # Normalize spike times if requested
    if scale_spike_times:
        aligned_spike_trains = _scale_rel_times(aligned_spike_trains)

    # Segments (optionally truncated or reordered)
    segments = _get_segments(aligned_spike_trains, col_to_rearrange_segments)
    if max_segments_to_plot:
        segments = segments[:max_segments_to_plot]

    # Clusters (optionally truncated)
    clusters = _prepare_clusters(
        aligned_spike_trains, cluster_col, max_clusters_to_plot)

    # Group data by cluster and segment
    grouped_spikes = aligned_spike_trains.groupby(
        [cluster_col, 'new_segment']
    )['rel_spike_time']

    # Build y-axis items ordered by cluster then segment
    y_items = []
    for cluster_id in clusters:
        for seg in segments:
            y_items.append((cluster_id, seg))

    # Prepare spike_data mapping: (cluster, segment) -> spikes
    spike_data_combined = {}
    for (cluster_id, seg) in y_items:
        if (cluster_id, seg) in grouped_spikes.groups:
            spike_data_combined[(cluster_id, seg)] = grouped_spikes.get_group(
                (cluster_id, seg)).values
        else:
            spike_data_combined[(cluster_id, seg)] = []

    # Create figure
    num_rows = len(y_items)
    # Height scales with number of rows; use height_per_row for stronger squashing
    fig_height = max(3, min(height_per_row * num_rows + 1.2, 12))
    plt.figure(figsize=(13, fig_height))

    # Render raster using generic helper
    # Convert y_items to readable string labels for display
    y_item_labels = [f"c{c}|s{int(s) if isinstance(s, (int, np.integer)) or (isinstance(s, float) and s.is_integer()) else s}"
                     for (c, s) in y_items]
    # Reindex spike_data to use string labels
    spike_data_str_keys = {label: spike_data_combined[item]
                           for label, item in zip(y_item_labels, y_items)}
    _plot_raster_by_yitems(
        spike_data=spike_data_str_keys,
        y_items=y_item_labels,
        title_id=None,
        title_prefix=f"{title_prefix} (Combined)",
        y_label="Cluster | Segment"
    )

    # Optional per-row shading: repeat segment windows for every cluster row
    if shade_start_cols and shade_end_cols:
        segments_repeated = [seg for (_c, seg) in y_items]
        shade_event_windows_for_segments(
            new_seg_info=new_seg_info,
            segments=segments_repeated,
            event_start_columns=shade_start_cols,
            event_end_columns=shade_end_cols
        )

    # Plot event polylines across all rows
    if events_to_plot:
        num_items = len(y_item_labels)
        y_positions = num_items - np.arange(num_items)  # top to bottom

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
            'gray'])

        seg_to_event_values = {}
        for evt in events_to_plot:
            if new_seg_info is None:
                continue
            if evt in events_to_plot_dict:
                col = evt
            else:
                col = _ensure_rel_column(new_seg_info, evt)
                if col is None:
                    continue
            vals = (new_seg_info[['new_segment', col]]
                    .drop_duplicates('new_segment')
                    .set_index('new_segment')[col])
            seg_to_event_values[evt] = vals

        for idx, evt in enumerate(events_to_plot):
            if new_seg_info is None:
                break
            if evt in events_to_plot_dict:
                label, color = events_to_plot_dict[evt]
            else:
                label = evt.replace('rel_', '')
                color = color_cycle[idx % len(color_cycle)]
            if scale_spike_times:
                label = f"Scaled {label}" if 'Scaled' not in label else label

            vals = seg_to_event_values.get(evt)
            if vals is None:
                continue
            x_values = []
            for (_c, seg) in y_items:
                x_values.append(vals.get(seg, np.nan))
            x_values = np.asarray(x_values, dtype=float)
            mask = ~pd.isna(x_values)
            if mask.sum() == 0:
                continue
            plt.plot(x_values[mask], y_positions[mask],
                     color=color, label=label)

    _set_xlim(aligned_spike_trains, xmin, xmax)
    # Draw horizontal separators between clusters if requested
    if draw_cluster_separators and len(y_items) > 1:
        ax = plt.gca()
        x0, x1 = ax.get_xlim()
        # For each boundary between consecutive rows where cluster changes,
        # draw a horizontal line at the boundary (half step between rows).
        for i in range(len(y_items) - 1):
            curr_cluster, _ = y_items[i]
            next_cluster, _ = y_items[i + 1]
            if curr_cluster != next_cluster:
                y_boundary = num_rows - i - 0.5
                plt.hlines(y_boundary, x0, x1,
                           colors='gray', linewidth=1.5, linestyles='-')
    _finalize_legend_and_layout()


def plot_rasters_by_segment(
    aligned_spike_trains,
    cluster_col='cluster',
    title_prefix="Raster Plot for Segment",
    xmin=None,
    xmax=None,
    col_to_rearrange_segments=None,
    scale_spike_times=False,
    max_clusters_to_plot=None,
    segments_to_plot=None,
    new_seg_info=None,
    extra_events_source_df=None,
    shade_start_cols=None,
    shade_end_cols=None,
    events_to_plot=(
        'rel_new_seg_start_time',
        'rel_new_seg_end_time',
        'rel_stop_time',
        'rel_prev_ff_caught_time'
    ),
    show_cluster_tick_labels=True,
    combine_subplots=False,
):
    """
    Plot raster plots of spike times grouped by segment and cluster.
    One figure per segment; clusters shown along the y-axis.
    Optionally, shade per-segment event windows using shade_start_cols/shade_end_cols.

    Parameters
    ----------
    combine_subplots : bool, default=False
        If False (default), produce one figure per segment.
        If True, squash all per-segment subplots into a single large raster
        (delegates to plot_rasters_combined). When segments_to_plot is provided,
        the data is pre-filtered to only those segments before combining.
    """
    # Normalize spike times if requested
    if scale_spike_times:
        aligned_spike_trains = _scale_rel_times(aligned_spike_trains)

    # If provided, merge event columns from extra_events_source_df into new_seg_info
    # Use events_to_plot (only) to determine which columns to bring over.
    if new_seg_info is not None and extra_events_source_df is not None and events_to_plot is not None:
        # Map requested relative event names to their absolute counterparts
        base_event_cols = []
        for evt in events_to_plot:
            base_col = evt.replace('rel_', '')
            if base_col in extra_events_source_df.columns:
                base_event_cols.append(base_col)
        cols = ['new_segment'] + base_event_cols
        if len(cols) > 1:
            new_seg_info = new_seg_info.merge(
                extra_events_source_df[cols].drop_duplicates('new_segment'),
                on='new_segment',
                how='left'
            )

    # Get and optionally truncate list of segments
    segments = _get_segments(aligned_spike_trains, col_to_rearrange_segments)
    if segments_to_plot is not None:
        segments = segments[segments_to_plot]

    # Identify clusters to plot
    clusters = _prepare_clusters(
        aligned_spike_trains,
        cluster_col,
        max_clusters_to_plot
    )

    # Group data by cluster and segment
    grouped_spikes = aligned_spike_trains.groupby(
        [cluster_col, 'new_segment']
    )['rel_spike_time']

    num_segments = len(segments)
    if num_segments == 0:
        return
    height_per_row = 2.2
    fig_height = max(3, min(height_per_row * num_segments + 1.0, 25))
    subplots_kwargs = dict(
        nrows=num_segments, ncols=1, figsize=(12, fig_height), sharex=True
    )
    if combine_subplots:
        subplots_kwargs['gridspec_kw'] = {'hspace': 0.0}
    fig, axes = plt.subplots(**subplots_kwargs)
    # Normalize axes shape into a list
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.flatten()

    # Draw each segment on its own subplot
    for ax, segment_id in zip(axes, segments):
        plt.sca(ax)
        spike_data = {
            cluster_id: grouped_spikes.get_group(
                (cluster_id, segment_id)).values
            for cluster_id in clusters
            if (cluster_id, segment_id) in grouped_spikes.groups
        }
        _plot_raster_by_yitems(
            spike_data=spike_data,
            y_items=list(clusters),
            title_id=segment_id,
            title_prefix=title_prefix,
            y_label=""
        )
        # Remove subplot title and x-axis label; set segment as y-axis label
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel(f"{segment_id}")
        # Optionally hide cluster tick labels
        if not show_cluster_tick_labels:
            ax.set_yticks([])
            ax.set_yticklabels([])
        if new_seg_info is not None:
            _plot_events_for_single_segment(
                new_seg_info,
                segment=segment_id,
                events_to_plot=events_to_plot,
                scale_spike_times=scale_spike_times
            )
            if shade_start_cols and shade_end_cols:
                shade_event_windows_for_single_segment(
                    new_seg_info=new_seg_info,
                    segment=segment_id,
                    event_start_columns=shade_start_cols,
                    event_end_columns=shade_end_cols
                )
        _set_xlim(aligned_spike_trains, xmin, xmax)

    # Create a single, figure-level legend with unique labels
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
    by_label = dict(zip(labels_all, handles_all))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(1.02, 0.5),
                   fontsize='small', borderaxespad=0)
        plt.subplots_adjust(right=0.75)
    # Control vertical spacing between subplots
    if combine_subplots:
        # Remove vertical gaps between subplots for a perfectly stacked look
        plt.subplots_adjust(hspace=0.0)
        # Hide x tick labels for all but the bottom subplot for cleanliness
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
    else:
        plt.tight_layout()
    plt.show()
