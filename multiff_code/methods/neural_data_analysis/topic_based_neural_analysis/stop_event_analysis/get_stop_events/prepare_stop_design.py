# =========================
# Standard library
# =========================
import os
import math
import json

# =========================
# Third-party
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# MultiFF imports (USED ONLY)
# =========================
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_utils,
    pn_aligned_by_event,
)

from neural_data_analysis.design_kits.design_by_segment import (
    rebin_segments,
    temporal_feats,
    create_pn_design_df,
)

from neural_data_analysis.topic_based_neural_analysis.full_session import (
    create_full_session_design,
    selected_pn_design_features,
    selected_stop_design_features,
    create_best_arc_design,
    select_fs_features,
)

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
)

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils,
    assemble_stop_design,
    collect_stop_data,
)

from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import (
    glm_runner,
)

from decision_making_analysis.data_compilation import (
    miss_events_class,
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing



def build_stop_design(raw_data_folder_path, bin_width, global_bins_2d=None):
    pn, datasets, _ = collect_stop_data.collect_stop_data_func(
        raw_data_folder_path,
        bin_width=bin_width,
    )

    captures_df, valid_captures_df, filtered_no_capture_stops_df, stops_with_stats = (
        get_stops_utils.prepare_no_capture_and_captures(
            monkey_information=pn.monkey_information,
            closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
            ff_caught_T_new=pn.ff_caught_T_new,
            distance_col='distance_from_ff_to_stop',
        )
    )

    stops_with_stats['stop_time'] = stops_with_stats['stop_id_start_time']
    stops_with_stats['prev_time'] = stops_with_stats['stop_id_end_time'].shift(
        1)
    stops_with_stats['next_time'] = stops_with_stats['stop_id_start_time'].shift(
        -1)

    new_seg_info = event_binning.make_new_seg_info_for_stop_design(
        stops_with_stats,
        pn.closest_stop_to_capture_df,
        pn.monkey_information,
    )

    events_with_stats = stops_with_stats[[
        'stop_id',
        'stop_cluster_id',
        'stop_id_start_time',
        'stop_id_end_time',
    ]].rename(columns={
        'stop_id': 'event_id',
        'stop_cluster_id': 'event_cluster_id',
        'stop_id_start_time': 'event_id_start_time',
        'stop_id_end_time': 'event_id_end_time',
    })

    stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups = (
        assemble_stop_design.build_stop_design(
            new_seg_info,
            events_with_stats,
            pn.monkey_information,
            pn.spikes_df,
            pn.ff_dataframe,
            datasets=datasets,
            add_ff_visible_info=True,
            global_bins_2d=global_bins_2d,
        )
    )
    
    return stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups