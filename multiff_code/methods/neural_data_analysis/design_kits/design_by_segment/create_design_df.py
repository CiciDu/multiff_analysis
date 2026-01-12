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
from neural_data_analysis.design_kits.design_by_segment import temporal_feats, spatial_feats, predictor_utils, other_feats


import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series, list]




def get_initial_design_df(
    data: pd.DataFrame,
    dt: float,
    trial_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    
    # work on a copy
    data = data.copy()
    trial_ids = np.asarray(trial_ids).ravel()
    if len(trial_ids) != len(data):
        raise ValueError('len(trial_ids) must equal len(data)')

    specs, meta0 = temporal_feats.init_predictor_specs(
        data,
        dt,
        trial_ids,
    )

    specs, meta0 = temporal_feats.add_state_predictors(
        specs,
        meta0,
        data,
        state_cols=['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory'],
        state_mode='passthrough',
    )

    specs, meta0 = temporal_feats.add_event_predictors(
        specs,
        meta0,
        data,
        events_to_include=[], #['stop', 'capture_ff'],
        basis_family_event='rc',
        n_basis_event=6,
    )


    specs = spatial_feats.add_visibility_transition_kernels(
        specs,
        data,
        trial_ids=meta0['trial_ids'],
        dt=dt,
        stems=('cur_vis',),
        n_basis=3,
        t_max=0.30,
        t_min=0.0,
        family='rc',
    )

    # ---------- assemble temporal design ----------
    design_df, meta = temporal_feats.specs_to_design_df(
        specs,
        meta0['trial_ids'],
        edge='zero',
        add_intercept=True,
    )
    rows_mask = meta.get('valid_rows_mask')  # None for edge='zero'

    # ---------- circular (Fourier) angles with gates ----------
    angle_specs = (
        ('cur_ff_angle', 'cur_in_memory'),
        ('nxt_ff_angle', 'nxt_in_memory'),
    )
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
            M=1,
            degrees=False,
            center=True,
            standardize=False,
            meta=meta,
        )

    # ---------- distance / time-since features ----------
    design_df, meta = other_feats.add_ff_distance_features(
        design_df,
        data,
        meta,
        dist_col='cur_ff_distance',
        gate_with='cur_in_memory',
    )
    design_df, meta = other_feats.add_ff_distance_features(
        design_df,
        data,
        meta,
        dist_col='nxt_ff_distance',
        gate_with='nxt_in_memory',
    )
    design_df, meta = other_feats.add_time_since_features(
        design_df,
        data,
        meta,
        cols=('time_since_target_last_seen', 'time_since_last_capture'),
        gate_with='cur_vis',
    )
    design_df, meta = other_feats.add_cum_dist_since_seen_features(
        design_df,
        data,
        meta,
    )

    # ---------- refactored block ----------
    design_df, meta = add_state_and_spatial_features(
        design_df=design_df,
        data=data,
        meta=meta,
    )

    # NOTE: no finalize/normalize shims needed; grouping now matches the checker logic
    return design_df, meta0, meta




def add_state_and_spatial_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
) -> tuple[pd.DataFrame, dict]:
    """Add memory, eye, acceleration, raw, and spatial spline features."""


    design_df, meta = other_feats.add_eye_speed_features(
        design_df, data, meta
    )
    design_df, meta = other_feats.add_gaze_features(
        design_df, data, meta, add_lowrank_2d=True, rank_xy=3
    )
    design_df, meta = other_feats.add_eye_component_features(
        design_df, data, meta
    )
    # design_df, meta = other_feats.add_speed_features(design_df, data, meta)

    # ---------- acceleration ----------
    for accel_col in ['accel', 'ang_accel']:
        design_df, meta = other_feats.add_acceleration_features(
            design_df, data, meta, accel_col=accel_col
        )

    # ---------- raw passthrough features ----------
    raw_features = [
        #'curv_of_traj',
        'num_ff_visible', 'log1p_num_ff_visible',
        'num_ff_in_memory', 'log1p_num_ff_in_memory',
    ]
    for k in raw_features:
        if k not in data.columns:
            raise KeyError(f'missing raw feature {k!r} in data')
        design_df, meta = other_feats.add_raw_feature(
            design_df,
            feature=k,
            data=data,
            name=k,
            transform='linear',
            eps=1e-6,
            center=True,
            scale=False,
            meta=meta,
        )

    # ---------- spatial splines ----------
    spatial_covs = (
        # 'cur_ff_distance', 'nxt_ff_distance',
        'speed', 'ang_speed',
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
            K=3,
            degree=3,
            percentiles=(2, 98),
            drop_one=True,
            center=True,
            meta=meta,
        )

    return design_df, meta

