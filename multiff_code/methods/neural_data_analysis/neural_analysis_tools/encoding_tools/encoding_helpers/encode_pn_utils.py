
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import pandas as pd
import re

import numpy as np
import pandas as pd

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
    stop_design,
    cluster_design,
)
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encode_stops_utils, encoding_design_utils


from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design

from neural_data_analysis.design_kits.design_by_segment import (
    other_feats,
    spatial_feats,
    temporal_feats
)

from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    temporal_feats,
    create_pn_design_df,
)

_PN_BEHAVIORAL_GROUP_SPECS: List[tuple] = [

    # ------------------------------------------------------------------
    # Constant
    # ------------------------------------------------------------------
    ('const', ['const'], '1D'),

    # ------------------------------------------------------------------
    # Visibility state (scalar flags)
    # ------------------------------------------------------------------
    ('cur_vis', ['cur_vis'], '1D'),
    ('nxt_vis', ['nxt_vis'], '1D'),
    ('nxt_in_memory', ['nxt_in_memory'], '1D'),

    # ------------------------------------------------------------------
    # Current visibility ON/OFF temporal bases
    # ------------------------------------------------------------------
    ('cur_vis_on_basis', None, 'event'),   # cols: cur_vis_on:b0:*
    ('cur_vis_off_basis', None, 'event'),  # cols: cur_vis_off:b0:*

    # ------------------------------------------------------------------
    # Current & next FF angle (circular encoding)
    # ------------------------------------------------------------------
    ('cur_ff_angle', ['cur_ff_angle:sin1', 'cur_ff_angle:cos1'], '1D'),
    ('nxt_ff_angle', ['nxt_ff_angle:sin1', 'nxt_ff_angle:cos1'], '1D'),

    # ------------------------------------------------------------------
    # Current & next FF distance spline basis
    # ------------------------------------------------------------------
    ('cur_ff_distance', None, '1D'),  # cols: cur_ff_distance_spline:s*
    ('nxt_ff_distance', None, '1D'),  # cols: nxt_ff_distance_spline:s*

    # ------------------------------------------------------------------
    # Temporal history variables
    # ------------------------------------------------------------------
    ('time_since_target_last_seen', ['time_since_target_last_seen'], '1D'),
    ('time_since_last_capture', ['time_since_last_capture'], '1D'),
    ('cum_dist_seen_log1p', ['cum_dist_seen_log1p'], '1D'),

    # ------------------------------------------------------------------
    # Eye & gaze
    # ------------------------------------------------------------------
    ('eye_speed', ['eye_speed_log1p'], '1D'),
    ('gaze_xy', [
        'gaze_mky_view_x_z',
        'gaze_mky_view_y_z',
        'gaze_mky_view_x_z*gaze_mky_view_y_z'
    ], '1D'),

    # ------------------------------------------------------------------
    # Retinal disparity (R/L eye components)
    # ------------------------------------------------------------------
    ('RD', ['RDz_odd', 'RDz_mag', 'RDy_odd', 'RDy_mag'], '1D'),
    ('LD', ['LDz_odd', 'LDz_mag', 'LDy_odd', 'LDy_mag'], '1D'),

    # ------------------------------------------------------------------
    # Acceleration terms
    # ------------------------------------------------------------------
    ('accel', ['accel_odd', 'accel_mag'], '1D'),
    ('ang_accel', ['ang_accel_odd', 'ang_accel_mag'], '1D'),

    # ------------------------------------------------------------------
    # Firefly counts
    # ------------------------------------------------------------------
    ('ff_visible', ['num_ff_visible', 'log1p_num_ff_visible'], '1D'),
    ('ff_in_memory', ['num_ff_in_memory', 'log1p_num_ff_in_memory'], '1D'),

    # ------------------------------------------------------------------
    # Speed spline bases
    # ------------------------------------------------------------------
    ('speed', None, '1D'),       # cols: speed:s*
    ('ang_speed', None, '1D'),   # cols: ang_speed:s*
]


def build_pn_encoding_design(
    pn,
    global_bins_2d,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    use_boxcar: bool = False,
    tuning_feature_mode: Optional[str] = None,
    binrange_dict: Optional[Dict[str, Union[np.ndarray, Tuple[float, float]]]] = None,
    tuning_n_bins: int = 10,
    linear_vars: Optional[List[str]] = None,
    angular_vars: Optional[List[str]] = None,
    use_planning_rename: bool = True,
    custom_rename: Optional[Dict[str, str]] = None,
):
    """
    Assemble stop design for encoding (GAM / temporal + tuning basis).
    """

    data = pn.rebinned_y_var.copy()
    trial_ids = data['new_segment']
    dt = pn.bin_width

    data = temporal_feats.add_stop_and_capture_columns(
        data,
        trial_ids,
        pn.ff_caught_T_new,
    )

    binned_feats, meta0, meta = (
        create_pn_design_df.get_pn_design_base(
            data,
            dt,
            trial_ids,
        )
    )

    cluster_cols = [c for c in pn.rebinned_x_var.columns if c.startswith('cluster_')]
    binned_spikes = pn.rebinned_x_var[cluster_cols]
    binned_spikes.columns = (
        binned_spikes.columns
        .str.replace('cluster_', '')
        .astype(int)
    )

    # Optionally rename planning columns to one_ff names (target_distance -> r_targ, etc.)
    # -------------------------------------------------------------------------
    if use_planning_rename or custom_rename:
        monkey_for_encoding = encoding_design_utils.monkey_information_for_encoding(
            pn.monkey_information,
            rename_planning_to_one_ff=use_planning_rename,
            custom_rename=custom_rename,
        )
    else:
        monkey_for_encoding = pn.monkey_information

    kinematic_cols = []
    extra_agg_cols = encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS
    # -------------------------------------------------------------------------

    # ------ add boxcar covariates
    binned_feats_to_add, exposure, used_bins, mask_used, pos = (
        encoding_design_utils._bin_monkey_information_feats_from_event_bins(
            monkey_for_encoding,
            global_bins_2d,
            kinematic_cols=kinematic_cols,
            extra_agg_cols=extra_agg_cols,
        )
    )
    

    # add columns in binned_feats_to_add that are missing in binned_feats (e.g. 'r_targ_bin', 'theta_targ_bin' if not already present)
    for col in binned_feats_to_add.columns:
        if col not in binned_feats.columns:
            binned_feats[col] = binned_feats_to_add[col]


    binned_feats = encoding_design_utils._ensure_one_ff_style_covariates(binned_feats)


    binned_feats, tuning_meta, mode = encoding_design_utils.add_tuning_features_to_design(
        binned_feats,
        use_boxcar=use_boxcar,
        tuning_feature_mode=tuning_feature_mode,
        binrange_dict=binrange_dict,
        tuning_n_bins=tuning_n_bins,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        raw_feature_cols_to_drop=encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS,
    )

    # ------------------------------------------------------------------
    # 4) Summed stop-event temporal basis
    # ------------------------------------------------------------------

    events = ['cur_on']
    temporal_meta = {}
    if len(events) > 0:
        for event in events:
            if event not in data.columns:
                print(f'Missing required event column "{event}" in data')
                continue

            temporal_df, current_temporal_meta = encoding_design_utils.build_temporal_design_from_binned_events(
                bin_t_center=data['t_center'].to_numpy(dtype=float),
                event_indicator=binned_feats['stop_event'].values,
                bin_dt=dt,
                n_basis=n_basis,
                t_min=t_min,
                t_max=t_max,
                index=binned_feats.index,
                name_prefix=event,
            )

            binned_feats = pd.concat([binned_feats, temporal_df], axis=1)
            # combine temporal_meta and current_temporal_meta safely
            for key, val in current_temporal_meta.items():
                if key in temporal_meta:
                    temporal_meta[key] = temporal_meta[key] + val
                else:
                    temporal_meta[key] = val
    else:
        temporal_df = None
        temporal_meta = None


    # Drop constant columns except 'const'
    const_cols_to_drop = [
        c for c in binned_feats.columns
        if c != 'const' and binned_feats[c].nunique() <= 1
    ]
    binned_feats = binned_feats.drop(columns=const_cols_to_drop)


    return (
        binned_spikes,
        binned_feats,
        temporal_meta,
        tuning_meta,
    )
