"""
Vis (ff visibility) decoding design builder.

Uses one-hot binned indicators for ff_on, ff_off, group_ff_on, group_ff_off
instead of basis functions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils


def _add_ff_visibility_onehot_to_binned_feats(
    binned_feats: pd.DataFrame,
    meta_df_used: pd.DataFrame,
    ff_on_df: pd.DataFrame,
    group_on_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add one-hot binned indicators for ff_on, ff_off, group_ff_on, group_ff_off.

    For each bin, indicator = 1 if any event of that type falls in the bin, else 0.
    """
    bins_2d_used = np.column_stack([
        meta_df_used['t_left'].to_numpy(dtype=float),
        meta_df_used['t_right'].to_numpy(dtype=float),
    ])

    vis_specs = [
        (ff_on_df['ff_vis_start_time'].to_numpy(dtype=float), 'ff_on_in_bin'),
        (ff_on_df['ff_vis_end_time'].to_numpy(dtype=float), 'ff_off_in_bin'),
        (group_on_df['group_on_start_time'].to_numpy(dtype=float), 'group_ff_on_in_bin'),
        (group_on_df['group_on_end_time'].to_numpy(dtype=float), 'group_ff_off_in_bin'),
    ]

    for event_times, col_name in vis_specs:
        event_times = event_times[np.isfinite(event_times)]
        if event_times.size == 0:
            binned_feats[col_name] = 0
            continue

        bin_idx = vis_design.map_times_to_bin_idx_unsorted(bins_2d_used, event_times)
        valid = bin_idx >= 0
        indicator = np.zeros(len(binned_feats), dtype=np.int8)
        if valid.any():
            np.add.at(indicator, bin_idx[valid], 1)
        indicator = (indicator > 0).astype(np.int8)
        binned_feats[col_name] = indicator

    return binned_feats


def build_vis_design_decoding(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    spikes_df: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    bin_dt: float = 0.04,
    add_ff_visible_info: bool = True,
    add_retries_info: bool = False,
    datasets: Optional[dict] = None,
    global_bins_2d: Optional[np.ndarray] = None,
    extra_agg_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """
    Build vis (ff visibility) design for decoding.

    Same as build_stop_design_decoding but adds one-hot binned indicators for
    ff_on, ff_off, group_ff_on, group_ff_off (no basis functions).
    """
    (
        binned_spikes,
        binned_feats,
        offset_log,
        meta_df_used,
        groups,
    ) = decode_stops_design.build_stop_design_decoding(
        new_seg_info,
        events_with_stats,
        monkey_information,
        spikes_df,
        ff_dataframe,
        bin_dt=bin_dt,
        add_ff_visible_info=add_ff_visible_info,
        add_retries_info=add_retries_info,
        datasets=datasets,
        global_bins_2d=global_bins_2d,
        extra_agg_cols=extra_agg_cols,
    )

    ff_on_df, group_on_df = decode_vis_utils.extract_ff_visibility_tables_fast(
        ff_dataframe
    )

    binned_feats = _add_ff_visibility_onehot_to_binned_feats(
        binned_feats,
        meta_df_used,
        ff_on_df,
        group_on_df,
    )

    vis_event_cols = [
        'ff_on_in_bin',
        'ff_off_in_bin',
        'group_ff_on_in_bin',
        'group_ff_off_in_bin',
    ]
    groups['vis_events'] = [c for c in vis_event_cols if c in binned_feats.columns]

    return binned_spikes, binned_feats, offset_log, meta_df_used, groups
