
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import pandas as pd


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decode_fs_utils


from neural_data_analysis.design_kits.design_by_segment import (
    other_feats
)


def build_fs_encoding_design(
    pn,
    bin_width: float,
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
    Full-session encoding design builder.

    Requires pn.monkey_information to already contain the time_since_* and
    num_ff_* columns (added by decode_fs_utils.add_time_since_capture_or_stop
    and decode_fs_utils.add_columns_related_to_ff_visibility).
    """

    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers.encoder_gam_helper import (
        FS_ENCODING_VAR_CATEGORIES,
    )
    var_categories = FS_ENCODING_VAR_CATEGORIES

    binned_feats, meta_df_used, bins_2d, pos, new_seg_info = decode_fs_utils.build_fs_design_decoding(pn)

    binned_spikes, _ = encoding_design_utils.bin_spikes_for_event_windows(
        pn.spikes_df,
        bins_2d,
        pos,
        time_col='time',
        cluster_col='cluster',
    )

    binned_feats = encoding_design_utils._ensure_one_ff_style_covariates(
        binned_feats
    )

    # ==============================================================
    # 5) Continuous tuning block
    # ==============================================================
    # Use a copy to avoid mutating the module-level constant.
    raw_feature_cols_to_drop = list(encoding_design_utils.ONE_FF_STYLE_ENCODING_COLS)
    raw_feature_cols_to_drop.remove('time')
    binned_feats, tuning_meta, _ = (
        encoding_design_utils.add_tuning_features_to_design(
            binned_feats,
            use_boxcar=use_boxcar,
            tuning_feature_mode=tuning_feature_mode,
            binrange_dict=binrange_dict,
            tuning_n_bins=tuning_n_bins,
            linear_vars=linear_vars,
            angular_vars=angular_vars,
            raw_feature_cols_to_drop=raw_feature_cols_to_drop,
        )
    )

    # Ensure 'time' (kept as raw, not splines) is in tuning_meta
    binned_feats, tuning_meta = other_feats.add_raw_feature(
        binned_feats,
        feature='time',
        data=binned_feats,
        name='time',
        transform='linear',
        eps=1e-6,
        center=True,
        scale=True,
        meta=tuning_meta,
    )


    events = var_categories['event_vars']
    temporal_meta = {}
    if len(events) > 0:
        for event in events:
            if event not in binned_feats.columns:
                print(f'Missing required event column "{event}" in data')
                print('binned_feats.columns:', binned_feats.columns.tolist())
                continue

            temporal_df, current_temporal_meta = encoding_design_utils.build_temporal_design_from_binned_events(
                bin_t_center=meta_df_used['t_center'].to_numpy(dtype=float),
                event_indicator=binned_feats[event].values,
                bin_dt=bin_width,
                n_basis=n_basis,
                t_min=t_min,
                t_max=t_max,
                index=binned_feats.index,
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


    # for encoding_fs
    raw_features = [
        # 'curv_of_traj',
        'num_ff_visible', 'log1p_num_ff_visible',
        'num_ff_in_memory', 'log1p_num_ff_in_memory',
        'time_since_prev_stop', 'time_since_prev_capture',
        'time_since_global_burst_start', 'time_since_prev_ff_visible',
    ]
    for k in raw_features:
        if k not in binned_feats.columns:
            raise KeyError(f'missing raw feature {k!r} in data')
        binned_feats, tuning_meta = other_feats.add_raw_feature(
            binned_feats,
            feature=k,
            data=binned_feats,
            name=k,
            transform='linear',
            eps=1e-6,
            center=True,
            scale=False,
            meta=tuning_meta,
        )


    # ==============================================================
    # 9) Drop constant columns (except const)
    # ==============================================================
    const_cols = [
        c for c in binned_feats.columns
        if c != 'const' and binned_feats[c].nunique() <= 1
    ]
    binned_feats = binned_feats.drop(columns=const_cols)

    # ==============================================================
    # 10) Final sorting
    # ==============================================================
    sort_idx = meta_df_used.sort_values(
        ['event_id', 'k_within_seg']
    ).index

    meta_df_used = meta_df_used.loc[sort_idx].reset_index(drop=True)
    binned_feats = binned_feats.loc[sort_idx].reset_index(drop=True)
    binned_spikes = binned_spikes.loc[sort_idx].reset_index(drop=True)

    return (
        pn,
        binned_spikes,
        binned_feats,
        meta_df_used,
        temporal_meta,
        tuning_meta,
    )
