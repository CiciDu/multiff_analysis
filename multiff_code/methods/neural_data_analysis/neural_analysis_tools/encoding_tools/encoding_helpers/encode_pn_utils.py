
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import pandas as pd


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils



from neural_data_analysis.design_kits.design_by_segment import (
    other_feats
)


_PN_BEHAVIORAL_GROUP_SPECS: List[tuple] = [
    # Kinematic (1D each)
    ('accel', ['accel'], '1D'),
    ('speed', ['speed'], '1D'),
    ('ang_speed', ['ang_speed'], '1D'),
    # One_ff_gam-style tuning (if present in design; from agg_cols in encoding design)
    ('v', ['v'], '1D'),
    ('w', ['w'], '1D'),
    ('d', ['d'], '1D'),
    ('phi', ['phi'], '1D'),
    ('r_targ', ['r_targ'], '1D'),
    ('theta_targ', ['theta_targ'], '1D'),
    ('eye_ver', ['eye_ver'], '1D'),
    ('eye_hor', ['eye_hor'], '1D'),
    ('time_since_target_last_seen', ['time_since_target_last_seen'], '1D'),
    ('time_since_last_capture', ['time_since_last_capture'], '1D'), 
    ('cur_ff_distance', ['cur_ff_distance'], '1D'),
    ('nxt_ff_distance', ['nxt_ff_distance'], '1D'),
    ('cur_ff_angle', ['cur_ff_angle'], '1D'),
    ('nxt_ff_angle', ['nxt_ff_angle'], '1D'),
    ('ff_visible', ['num_ff_visible', 'log1p_num_ff_visible'], '1D'),
    ('ff_in_memory', ['num_ff_in_memory', 'log1p_num_ff_in_memory'], '1D'),
    # Event design (event = temporal basis)
    ('cur_vis_on', None, 'event'),   # cols: rcos_* without *captured; filled below
    ('cur_vis_off', None, 'event'),
    ]


def build_pn_encoding_design(
    rebinned_y_var,
    monkey_information,
    global_bins_2d,
    bin_width,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    use_boxcar: bool = False,
    tuning_feature_mode: Optional[str] = None,
    binrange_dict: Optional[Dict[str, Union[np.ndarray, Tuple[float, float]]]] = None,
    tuning_n_bins: int = 10,
    linear_vars: Optional[List[str]] = None,
    angular_vars: Optional[List[str]] = None,
):
    """
    Assemble stop design for encoding (GAM / temporal + tuning basis).
    """

    # Optionally rename planning columns to one_ff names (target_distance -> r_targ, etc.)
    # -------------------------------------------------------------------------


    kinematic_cols = []
    agg_cols = encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS
    # -------------------------------------------------------------------------

    # ------ add boxcar covariates
    binned_feats, exposure, used_bins, mask_used, pos = (
        encoding_design_utils.bin_monkey_information_feats_from_event_bins(
            monkey_information,
            global_bins_2d,
            kinematic_cols=kinematic_cols,
            agg_cols=agg_cols,
        )
    )

    binned_feats = encoding_design_utils._ensure_one_ff_style_covariates(binned_feats)

    print('linear_vars:', linear_vars)
    print('angular_vars:', angular_vars)
    for var in (linear_vars or []) + (angular_vars or []):
        if var not in binned_feats.columns:
            if var in rebinned_y_var.columns:
                binned_feats[var] = rebinned_y_var[var].values
            else:
                print(f'Missing required variable "{var}" in binned features. Will skip it.')

    rebinned_y_var['time'] = binned_feats['time'].values # pass in as raw feature later
    raw_feature_cols_to_drop=encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS + (linear_vars or []) + (angular_vars or [])
    print('raw_feature_cols_to_drop:', raw_feature_cols_to_drop)

    binned_feats, tuning_meta, mode = encoding_design_utils.add_tuning_features_to_design(
        binned_feats,
        use_boxcar=use_boxcar,
        tuning_feature_mode=tuning_feature_mode,
        binrange_dict=binrange_dict,
        tuning_n_bins=tuning_n_bins,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        raw_feature_cols_to_drop=raw_feature_cols_to_drop,
    )

    # ------------------------------------------------------------------
    # 4) Summed stop-event temporal basis
    # ------------------------------------------------------------------

    events = ['cur_vis_on', 'cur_vis_off']
    temporal_meta = {}
    if len(events) > 0:
        for event in events:
            if event not in rebinned_y_var.columns:
                print(f'Missing required event column "{event}" in data')
                print('rebinned_y_var.columns:', rebinned_y_var.columns)
                continue

            temporal_df, current_temporal_meta = encoding_design_utils.build_temporal_design_from_binned_events(
                bin_t_center=rebinned_y_var['t_center'].to_numpy(dtype=float),
                event_indicator=rebinned_y_var[event].values,
                bin_dt=bin_width,
                n_basis=n_basis,
                t_min=t_min,
                t_max=t_max,
                index=rebinned_y_var.index,
                event_name=event,
            )

            binned_feats = pd.concat([binned_feats, temporal_df], axis=1)
            
            for key, val in current_temporal_meta.items():
                if key in temporal_meta:
                    temporal_meta[key] = encoding_design_utils.merge_meta_vals(temporal_meta[key], val)
                else:
                    temporal_meta[key] = val
    else:
        temporal_df = None
        temporal_meta = None

    raw_features = [
            # 'curv_of_traj',
            'time',  # kept as raw, not splines
            'num_ff_visible', 'log1p_num_ff_visible',
            'num_ff_in_memory', 'log1p_num_ff_in_memory',
        ]
    for k in raw_features:
        if k not in rebinned_y_var.columns:
            raise KeyError(f'missing raw feature {k!r} in data')
        binned_feats, tuning_meta = other_feats.add_raw_feature(
            binned_feats,
            feature=k,
            data=rebinned_y_var,
            name=k,
            transform='linear',
            eps=1e-6,
            center=True,
            scale=True,
            meta=tuning_meta,
        )

    # Drop constant columns except 'const'
    const_cols_to_drop = [
        c for c in binned_feats.columns
        if c != 'const' and binned_feats[c].nunique() <= 1
    ]
    binned_feats = binned_feats.drop(columns=const_cols_to_drop)


    return (
        binned_feats,
        temporal_meta,
        tuning_meta,
    )
