from typing import Dict, Tuple

import pandas as pd

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoding_design_utils,
)


def add_ff_visibility_temporal_designs(
    binned_feats: pd.DataFrame,
    temporal_meta: Dict,
    meta_df_used: pd.DataFrame,
    ff_on_df: pd.DataFrame,
    group_on_df: pd.DataFrame,
    bin_width: float,
    *,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Add ff visibility temporal designs (ff_on, ff_off, group_ff_on, group_ff_off)
    to binned_feats and merge into temporal_meta.
    """
    bin_t_center = meta_df_used['t_center'].to_numpy(dtype=float)

    vis_temporal_specs = [
        (ff_on_df['ff_vis_start_time'].to_numpy(dtype=float), 'ff_on'),
        (ff_on_df['ff_vis_end_time'].to_numpy(dtype=float), 'ff_off'),
        (group_on_df['group_on_start_time'].to_numpy(dtype=float), 'group_ff_on'),
        (group_on_df['group_on_end_time'].to_numpy(dtype=float), 'group_ff_off'),
    ]
    for ev_times, event_name in vis_temporal_specs:
        temporal_df, vis_temporal_meta = (
            encoding_design_utils.build_temporal_design_from_event_times(
                bin_t_center=bin_t_center,
                event_times=ev_times,
                bin_dt=bin_width,
                n_basis=n_basis,
                t_min=t_min,
                t_max=t_max,
                index=binned_feats.index,
                event_name=event_name,
            )
        )
        binned_feats = pd.concat([binned_feats, temporal_df], axis=1)
        for key, val in vis_temporal_meta.items():
            if key in temporal_meta:
                temporal_meta[key] = encoding_design_utils.merge_meta_vals(
                    temporal_meta[key], val
                )
            else:
                temporal_meta[key] = val

    return binned_feats, temporal_meta
