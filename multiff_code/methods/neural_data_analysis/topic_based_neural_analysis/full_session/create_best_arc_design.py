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


def get_best_arc_design_df(
    data: pd.DataFrame,
    dt: float,
) -> tuple[pd.DataFrame, dict, dict]:

    trial_ids = np.repeat(0, len(data))

    specs, meta0 = temporal_feats.init_predictor_specs(
        data,
        dt,
        trial_ids,
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
        ('best_arc_ff_angle', None),
    )
    for stem, gate_col in angle_specs:
        if stem not in data.columns:
            raise KeyError(f'missing angle column {stem!r} in data')
        if gate_col is not None and gate_col not in data.columns:
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
        dist_col='best_arc_ff_distance',
    )

    for k in ['best_arc_opt_arc_length', 'best_arc_abs_curv_diff']:
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
        'best_arc_opt_arc_curv', 'best_arc_curv_diff',
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
            K=6,
            degree=3,
            percentiles=(2, 98),
            drop_one=True,
            center=True,
            meta=meta,
        )

    # NOTE: no finalize/normalize shims needed; grouping now matches the checker logic
    return design_df, meta0, meta
