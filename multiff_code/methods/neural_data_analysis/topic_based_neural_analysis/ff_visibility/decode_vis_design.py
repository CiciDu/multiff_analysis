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



def build_vis_design_decoding(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    spikes_df: pd.DataFrame,
    processed_spike_rates: Optional[pd.DataFrame] = None,
    bin_dt: float = 0.04,
    add_ff_visible_info: bool = True,
    add_retries_info: bool = False,
    datasets: Optional[dict] = None,
    global_bins_2d: Optional[np.ndarray] = None,
    drop_bad_neurons: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """
    Build vis (ff visibility) design for decoding.

    Same as build_stop_design_decoding but adds one-hot binned indicators for
    ff_on, ff_off, group_ff_on, group_ff_off (no basis functions).
    """


    (
        rebinned_spike_rates,
        binned_feats,
        offset_log,
        meta_df_used,
    ) = decode_stops_design.build_stop_design_decoding(
        new_seg_info,
        events_with_stats,
        monkey_information,
        ff_dataframe,
        spikes_df,
        processed_spike_rates=processed_spike_rates,
        bin_dt=bin_dt,
        add_ff_visible_info=add_ff_visible_info,
        add_retries_info=add_retries_info,
        datasets=datasets,
        global_bins_2d=global_bins_2d,
        drop_bad_neurons=drop_bad_neurons,
    )

    ff_on_df, group_on_df = decode_vis_utils.extract_ff_visibility_tables_fast(
        ff_dataframe
    )

    binned_feats = decode_vis_utils._add_ff_visibility_onehot_to_binned_feats(
        binned_feats,
        meta_df_used,
        ff_on_df,
        group_on_df,
    )

    return rebinned_spike_rates, binned_feats, offset_log, meta_df_used
