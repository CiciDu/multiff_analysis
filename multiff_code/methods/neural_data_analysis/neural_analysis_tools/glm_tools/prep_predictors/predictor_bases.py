from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Tuple, List, Mapping

import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import BSpline

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.neural_analysis_tools.glm_tools.prep_predictors import temporal_feats, spatial_feats, predictor_utils, other_feats


import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series, list]


def get_initial_design_df(
    data: pd.DataFrame,
    dt: float,
    trial_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict, dict]:
    # work on a copy
    data = data.copy()
    trial_ids = np.asarray(trial_ids).ravel()
    if len(trial_ids) != len(data):
        raise ValueError('len(trial_ids) must equal len(data)')

    # ---------- stop onsets ----------
    if 'whether_new_distinct_stop' in data.columns:
        data['stop'] = (data['whether_new_distinct_stop'] == 1).astype(int)
    elif 'monkey_speeddummy' in data.columns:
        stopped = (data['monkey_speeddummy'] == 0)
        prev = stopped.groupby(trial_ids, sort=False).shift(1, fill_value=False)
        data['stop'] = (stopped & ~prev).astype(int)
    else:
        raise KeyError("need 'whether_new_distinct_stop' or 'monkey_speeddummy' to define stop onsets")

    # ---------- builder: events + raw states (passthrough) ----------
    colmap = {}
    if 'capture_ff' not in data.columns and 'ff_caught_T_new' in data.columns:
        colmap['capture_ff'] = 'ff_caught_T_new'

    specs, meta0 = temporal_feats.build_predictor_specs_from_behavior(
        data, dt, trial_ids,
        events_to_include=[],
        basis_family_event='rc',
        state_mode='passthrough',
        n_basis_event=6,
        column_map=colmap
    )

    specs = spatial_feats.add_visibility_transition_kernels(
        specs, data, trial_ids=meta0['trial_ids'], dt=dt,
        stems=('cur_vis', 'any_ff_visible'),
        n_basis=3, t_max=0.30, t_min=0.0, family='rc'
    )

    # assemble temporal design
    design_df, meta = temporal_feats.specs_to_design_df(
        specs, meta0['trial_ids'], edge='zero', add_intercept=True
    )
    rows_mask = meta.get('valid_rows_mask')  # None for edge='zero'

    # ---------- spatial splines ----------
    spatial_covs = (
        'cur_ff_distance', 'nxt_ff_distance',
        'cur_ff_rel_x', 'nxt_ff_rel_x',
        'speed', 'ang_speed', 'curv_of_traj',
    )
    for k in spatial_covs:
        if k not in data.columns:
            raise KeyError(f'missing spatial covariate {k!r} in data')
        design_df, meta = spatial_feats.add_spatial_spline_feature(
            design_df,
            feature=k,
            data=data,
            name=k,
            knots_mode='percentile',
            K=3, degree=3,
            percentiles=(2, 98),
            drop_one=True,
            center=True,
            meta=meta
        )

    # ---------- circular (Fourier) angles with gates ----------
    angle_specs = (('cur_ff_angle', 'cur_in_memory'),
                   ('nxt_ff_angle', 'nxt_in_memory'))
    for stem, gate_col in angle_specs:
        if stem not in data.columns:
            raise KeyError(f'missing angle column {stem!r} in data')
        if gate_col not in data.columns:
            raise KeyError(f'missing gate column {gate_col!r} in data')
        design_df, meta = spatial_feats.add_circular_fourier_feature(
            design_df,
            theta=stem,
            name=stem,
            data=data,
            rows_mask=rows_mask,
            gate=gate_col,
            M=3,
            degrees=False,
            center=True,
            standardize=False,
            meta=meta
        )

    # ---------- recommended additional features ----------
    design_df, meta = other_feats.add_memory_state_feature(design_df, data, meta)
    design_df, meta = other_feats.add_ff_distance_features(design_df, data, meta, dist_col='cur_ff_distance')
    design_df, meta = other_feats.add_time_since_features(
        design_df, data, meta,
        cols=('time_since_target_last_seen', 'time_since_last_capture'),
        gate_with='cur_vis'
    )
    design_df, meta = other_feats.add_cum_dist_since_seen_features(design_df, data, meta)
    design_df, meta = other_feats.add_eye_speed_features(design_df, data, meta)
    design_df, meta = other_feats.add_gaze_features(design_df, data, meta, add_lowrank_2d=True, rank_xy=3)
    design_df, meta = other_feats.add_eye_component_features(design_df, data, meta)
    design_df, meta = other_feats.add_speed_features(design_df, data, meta)

    for accel_col in ['accel', 'ang_accel']:
        design_df, meta = other_feats.add_acceleration_features(
            design_df, data, meta, accel_col=accel_col
        )

    # NOTE: no finalize/normalize shims needed; grouping now matches the checker logic
    return design_df, meta0, meta
