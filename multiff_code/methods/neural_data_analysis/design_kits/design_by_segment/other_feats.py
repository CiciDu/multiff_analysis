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
from neural_data_analysis.design_kits.design_by_segment import temporal_feats, spatial_feats, predictor_utils

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series, list]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _dedup_inplace(lst: list) -> None:
    """In-place stable de-duplication for lists kept in meta."""
    seen = set()
    i = 0
    for x in lst:
        if x not in seen:
            seen.add(x)
            lst[i] = x
            i += 1
    del lst[i:]


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score with NaN-safe handling."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 0 else 1.0
    z = (x - med) / scale
    z[~np.isfinite(z)] = 0.0
    return z


# ---------------------------------------------------------------------
# Core transforms and meta registration (small, reusable helpers)
# ---------------------------------------------------------------------

def _coerce_feature_vec(
    feature: Union[str, ArrayLike],
    *,
    data: Optional[pd.DataFrame],
    rows_mask: Optional[np.ndarray],
    name: Optional[str],
) -> tuple[np.ndarray, str]:
    """Return (float64 vector, base_name)."""
    if isinstance(feature, str):
        if data is None:
            raise ValueError('When feature is a column name, pass `data=`.')
        vec = data[feature].to_numpy()
        base_name = name or feature
    else:
        vec = np.asarray(feature)
        base_name = name or 'feat'
    v = np.asarray(vec, float).ravel()
    if rows_mask is not None:
        v = v[rows_mask]
    v[~np.isfinite(v)] = np.nan
    return v, base_name


def _transform_single(
    v: np.ndarray, *, transform: str, eps: float, center: bool, scale: bool
) -> tuple[np.ndarray, bool, bool, float, float]:
    """
    Apply a scalar transform + optional centering/scaling.
    Returns (x, did_center, did_scale, mu, sd).
    """
    t = transform.lower()
    if t == 'linear':
        x = v
    elif t == 'log':
        x = np.log(np.maximum(v, 0.0) + eps)
    elif t == 'log1p':
        x = np.log1p(np.maximum(v, 0.0))
    elif t == 'sqrt':
        x = np.sqrt(np.maximum(v, 0.0) + eps)
    elif t in ('zscore', 'standardize'):
        x = _robust_z(v)
        center, scale = True, True
    else:
        raise ValueError(
            "transform must be one of {'linear','log','log1p','sqrt','zscore','standardize'}")

    # Impute with mean of transformed values
    mask = np.isfinite(x)
    if not mask.any():
        raise ValueError('Transformed feature has no finite values.')
    mean_x = float(np.nanmean(x[mask]))
    x = np.where(np.isfinite(x), x, mean_x)

    did_center = bool(center)
    did_scale = bool(scale)
    mu = float(np.nanmean(x)) if did_center else 0.0
    sd = float(np.nanstd(x)) if did_scale else 1.0
    if did_center:
        x = x - mu
    if did_scale:
        sd = sd if sd > 0 else 1.0
        x = x / sd

    return x.astype(float), did_center, did_scale, mu, sd


def _odd_component(v: np.ndarray, kind: str, mag_eps: float) -> np.ndarray:
    """Directional transform (odd under sign flip)."""
    k = kind.lower()
    if k == 'zscore':
        odd = _robust_z(v)
    elif k == 'linear':
        odd = np.where(np.isfinite(v), v, 0.0)
    elif k == 'signed_sqrt':
        odd = np.sign(v) * np.sqrt(np.abs(v) + mag_eps)
    elif k == 'signed_log1p':
        odd = np.sign(v) * np.log1p(np.abs(v))
    else:
        raise ValueError(
            "odd_kind must be in {'zscore','linear','signed_sqrt','signed_log1p'}")
    if not np.isfinite(odd).any():
        raise ValueError('Odd component has no finite values.')
    return np.where(np.isfinite(odd), odd, 0.0).astype(float)


def _even_component(v: np.ndarray, kind: str, mag_eps: float) -> np.ndarray:
    """Magnitude transform (even under sign flip)."""
    ke = kind.lower()
    a = np.abs(v)
    if ke == 'log1p':
        mag = np.log1p(a)
    elif ke == 'sqrt':
        mag = np.sqrt(a + mag_eps)
    elif ke == 'abs':
        mag = a
    else:
        raise ValueError("even_kind must be in {'log1p','sqrt','abs'}")
    if not np.isfinite(mag).any():
        raise ValueError('Magnitude component has no finite values.')
    mean_mag = float(np.nanmean(mag[np.isfinite(mag)]))
    return np.where(np.isfinite(mag), mag, mean_mag).astype(float)


def _register_passthrough_col(meta: dict, col: str, *, source: Union[str, ArrayLike], spec: dict) -> dict:
    """
    Minimal, checker-compatible registration:
      - groups[col] = [col]
      - raw_specs[col] = spec
      - raw_cols += [col]
      - passthrough_* keyed by column name (not strictly needed by checker, but harmless)
    """
    m = dict(meta)
    groups = dict(m.get('groups', {}))
    groups.setdefault(col, []).append(col)
    _dedup_inplace(groups[col])
    m['groups'] = groups

    raw_specs = dict(m.get('raw_specs', {}))
    raw_specs[col] = {
        **spec, 'source': source if isinstance(source, str) else 'array'}
    m['raw_specs'] = raw_specs

    raw_cols = list(m.get('raw_cols', [])) + [col]
    _dedup_inplace(raw_cols)
    m['raw_cols'] = raw_cols

    pt_specs = dict(m.get('passthrough_specs', {}))
    pt_specs[col] = {**raw_specs[col], 'group': col}
    m['passthrough_specs'] = pt_specs

    pt_cols = list(m.get('passthrough_cols', [])) + [col]
    _dedup_inplace(pt_cols)
    m['passthrough_cols'] = pt_cols

    pt_groups = dict(m.get('passthrough_groups', {}))
    pt_groups.setdefault(col, [])
    if col not in pt_groups[col]:
        pt_groups[col].append(col)
    m['passthrough_groups'] = pt_groups

    return m


# ---------------------------------------------------------------------
# Public: small orchestrators
# ---------------------------------------------------------------------

def add_raw_feature(
    design_df: pd.DataFrame,
    feature: Union[str, ArrayLike],
    *,
    name: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    rows_mask: Optional[np.ndarray] = None,
    transform: str = 'linear',
    center: bool = False,
    scale: bool = False,
    eps: float = 1e-6,
    encoding: str = 'single',      # 'single' or 'odd_even'
    odd_kind: str = 'zscore',
    even_kind: str = 'log1p',
    mag_eps: float = 1e-6,
    meta: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Add a raw/derived feature to the design and register it minimally in meta.

    encoding='single'  -> one column: <name>
    encoding='odd_even'-> two columns: <name>_odd, <name>_mag
    """
    v, base = _coerce_feature_vec(
        feature, data=data, rows_mask=rows_mask, name=name)
    out = design_df.copy()

    if encoding.lower() == 'odd_even':
        odd = _odd_component(v, odd_kind, mag_eps)
        mag = _even_component(v, even_kind, mag_eps)
        odd_name, mag_name = f'{base}_odd', f'{base}_mag'
        out[odd_name] = odd
        out[mag_name] = mag
        if meta is None:
            return out, None
        m = _register_passthrough_col(meta, odd_name, source=feature, spec={
                                      'encoding': 'odd', 'odd_kind': odd_kind})
        m = _register_passthrough_col(m, mag_name, source=feature, spec={
                                      'encoding': 'even', 'even_kind': even_kind})
        return out, m

    # single
    x, did_center, did_scale, mu, sd = _transform_single(
        v, transform=transform, eps=eps, center=center, scale=scale)
    out_col = base
    out[out_col] = x
    if meta is None:
        return out, None
    spec = {
        'encoding': 'single',
        'transform': transform.lower(),
        'center': bool(did_center),
        'scale': bool(did_scale),
        'mean': mu if did_center or did_scale else None,
        'std': sd if did_scale else None,
        'eps': float(eps) if transform.lower() in {'log', 'sqrt'} else None,
    }
    m = _register_passthrough_col(meta, out_col, source=feature, spec=spec)
    return out, m


def add_acceleration_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    accel_col: str = 'accel',
    make_spline: bool = False,
    spline_K: int = 5,
    spline_degree: int = 3,
    spline_percentiles: tuple[int, int] = (2, 98),
    eps: float = 1e-6,
):
    """
    Acceleration: (1) odd/even pair; (2) optional magnitude tuning spline on log|accel|.
    """
    out, m = add_raw_feature(
        design_df, feature=accel_col, data=data, name=accel_col,
        encoding='odd_even', odd_kind='zscore', even_kind='log1p', meta=meta
    )
    if make_spline:
        accel_mag_log = np.log(np.abs(data[accel_col].to_numpy()) + eps)
        out, m = spatial_feats.add_spatial_spline_feature(
            design_df=out, feature=accel_mag_log, name=f'{accel_col}_mag_spline',
            knots_mode='percentile', K=spline_K, degree=spline_degree,
            percentiles=spline_percentiles, row_normalize=True,
            drop_one=True, center=True, meta=m
        )
    return out, m


def add_time_since_spline_feature(
    design_df: pd.DataFrame,
    time_since: Union[str, ArrayLike],
    *,
    name: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    rows_mask: Optional[np.ndarray] = None,
    t_floor: float = 0.05,
    t_cap: Optional[float] = 10.0,
    transform: str = 'log',        # spline domain; 'log' is recommended
    degree: int = 3,
    K: int = 6,
    percentiles: tuple[float, float] = (1, 99),
    drop_one: bool = True,
    center: bool = True,
    meta: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Time-since spline on transformed axis (log-time by default).
    No row-normalization for temporal bases.
    """
    # coerce
    if isinstance(time_since, str):
        if data is None:
            raise ValueError('When time_since is a column name, pass `data=`.')
        v = data[time_since].to_numpy().astype(float)
        if rows_mask is not None:
            v = v[rows_mask]
        feat = name or time_since
    else:
        v = np.asarray(time_since, float).ravel()
        if rows_mask is not None:
            v = v[rows_mask]
        feat = name or 'time_since'

    # clip/cap + transform
    v[~np.isfinite(v)] = np.nan
    v = np.where(~np.isfinite(v) | (v < 0), t_floor, v)
    if t_cap is not None:
        v = np.minimum(v, t_cap)
    v = np.maximum(v, t_floor)

    u = np.log(v) if transform == 'log' else v

    # knots/design via spatial helpers
    full_knots, K_eff = spatial_feats._build_spatial_knots(
        u, mode='percentile', degree=degree, K=K, percentiles=percentiles
    )
    X = spatial_feats._bspline_design_from_knots(u, full_knots, degree)

    # identifiability (drop-one + center)
    colnames = [f'{feat}:t{k}' for k in range(X.shape[1])]
    if drop_one and X.shape[1] >= 2:
        X = X[:, :-1]
        colnames = colnames[:-1]
    if center and X.size:
        X = X - X.mean(axis=0, keepdims=True)

    out = design_df.copy()
    out[colnames] = X

    if meta is None:
        return out, None

    m = dict(meta)
    g = dict(m.get('groups', {}))
    g.setdefault(feat, [])
    g[feat].extend(colnames)
    m['groups'] = g

    ts = dict(m.get('temporal_specs', {}))
    ts[feat] = {
        'domain': 'log' if transform == 'log' else 'linear',
        'degree': int(degree),
        'K': int(K_eff),
        'drop_one': bool(drop_one),
        'center': bool(center),
        't_floor': float(t_floor),
        't_cap': float(t_cap) if t_cap is not None else None,
        'percentiles': tuple(percentiles),
        'knots': full_knots.tolist(),
    }
    m['temporal_specs'] = ts
    m['temporal_cols'] = list(m.get('temporal_cols', [])) + colnames
    return out, m


def add_ff_distance_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    dist_col: str = 'cur_ff_distance',
    make_spline: bool = True,
    log_eps: float = 1e-2,
    K: int = 6,
    degree: int = 3,
    pct: tuple[int, int] = (2, 98),
    gate_with: str | None = None,  # e.g., 'cur_ff_in_memory'
):
    """
    Distance to target:
      (a) global log-distance ramp (centered),
      (b) optional row-normalized tuning spline on log-distance,
      (c) optional gating by a binary column (multiplies the design columns).
    """
    if not make_spline:
        out, m = add_raw_feature(
            design_df, feature=dist_col, data=data,
            name=f'log_{dist_col}', transform='log', eps=log_eps,
            center=True, scale=False, meta=meta
        )

        out, m = add_raw_feature(
            out, feature=dist_col, data=data,
            name=dist_col, transform='linear', meta=m
        )
    else:
        spline_name = f'{dist_col}_spline'
        d = data[dist_col].to_numpy()
        log_dist = np.log(np.maximum(d, 0.0) + log_eps)
        out, m = spatial_feats.add_spatial_spline_feature(
            design_df=design_df, feature=log_dist, name=spline_name,
            knots_mode='percentile', K=K, degree=degree, percentiles=pct,
            row_normalize=True, drop_one=True, center=True, meta=meta
        )

    # ----- optional gating -----
    if gate_with is not None and gate_with in data.columns:
        gate = (data[gate_with].to_numpy() > 0).astype(float)

        if not make_spline:
            # ramp columns we just created
            ramp_cols = [f'log_{dist_col}', dist_col]
            for c in ramp_cols:
                if c in out.columns:
                    out[c] = out[c].to_numpy() * gate
        else:
            # spline columns if present (they're registered under their prefix)
            for c in m.get('groups', {}).get(spline_name, []):
                if c in out.columns:
                    out[c] = out[c].to_numpy() * gate
    # ---------------------------

    return out, m


def add_time_since_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    cols: tuple[str, ...] = (
        'time_since_target_last_seen', 'time_since_last_capture'),
    make_spline: bool = False,
    t_floor: float = 0.05,
    t_cap: float = 12.0,
    K: int = 6,
    degree: int = 3,
    pct: tuple[int, int] = (1, 99),
    gate_with: Optional[str] = None,
):
    """
    For each time-since column: add a simple log1p ramp; optionally add an unnormalized spline on log-time.
    Optional gating multiplies the spline columns by a binary visibility gate.
    """
    out, m = design_df.copy(), dict(meta)
    for ts in cols:
        out, m = add_raw_feature(
            out, feature=ts, data=data, name=ts, transform='log1p', meta=m)
        if make_spline:
            out, m = add_time_since_spline_feature(
                design_df=out, time_since=ts, data=data, name=ts,
                t_floor=t_floor, t_cap=t_cap, transform='log',
                degree=degree, K=K, percentiles=pct, drop_one=True, center=True, meta=m
            )
            if gate_with is not None and gate_with in data.columns:
                gate = (data[gate_with].to_numpy() > 0).astype(float)
                for c in m.get('groups', {}).get(ts, []):
                    out[c] = out[c].to_numpy() * gate
    return out, m


def add_cum_dist_since_seen_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    col: str = 'cum_distance_since_target_last_seen',
    make_spline: bool = False,
    K: int = 5,
    degree: int = 3,
    pct: tuple[int, int] = (2, 98),
):
    """
    Cumulative distance since last seen: log1p ramp, optionally an unnormalized spline on log1p scale.
    """
    out, m = add_raw_feature(design_df, feature=col, data=data,
                             name='cum_dist_seen_log1p', transform='log1p', meta=meta)
    if make_spline:
        cum_log = np.log1p(data[col].to_numpy())
        out, m = spatial_feats.add_spatial_spline_feature(
            design_df=out, feature=cum_log, name='cum_dist_seen_spline',
            knots_mode='percentile', K=K, degree=degree, percentiles=pct,
            row_normalize=False, drop_one=True, center=True, meta=m
        )
    return out, m


def add_eye_speed_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    col: str = 'eye_world_speed',
    make_spline: bool = False,
    K: int = 5,
    degree: int = 3,
    pct: tuple[int, int] = (2, 98),
):
    """
    Eye speed: log1p ramp; optional row-normalized tuning spline on log1p(speed).
    """
    out, m = add_raw_feature(design_df, feature=col, data=data,
                             name='eye_speed_log1p', transform='log1p', meta=meta)
    if make_spline:
        spd_log = np.log1p(data[col].to_numpy())
        out, m = spatial_feats.add_spatial_spline_feature(
            design_df=out, feature=spd_log, name='eye_speed_spline',
            knots_mode='percentile', K=K, degree=degree, percentiles=pct,
            row_normalize=True, drop_one=True, center=True, meta=m
        )
    return out, m


def add_gaze_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    x_col: str = 'gaze_mky_view_x',
    y_col: str = 'gaze_mky_view_y',
    make_spline: bool = False,
    K: int = 5,
    degree: int = 3,
    pct: tuple[int, int] = (2, 98),
    add_lowrank_2d: bool = True,
    rank_xy: int = 3,
):
    """
    Gaze: optional 1-D row-normalized splines per axis; optional low-rank 2-D interactions.
    Interactions are registered as passthrough columns with groups under their own names.
    """
    out, m = design_df.copy(), dict(meta)

    # 1-D splines (optional)
    if make_spline:
        for axis in (x_col, y_col):
            out, m = spatial_feats.add_spatial_spline_feature(
                design_df=out, feature=axis, data=data, name=axis,
                knots_mode='percentile', K=K, degree=degree, percentiles=pct,
                row_normalize=True, drop_one=True, center=True, meta=m
            )

    # 2-D interactions (optional)
    if add_lowrank_2d:
        def _get_cols(axis: str):
            nonlocal out, m
            cols = [c for c in m.get('groups', {}).get(axis, []) if ':s' in c]
            if cols:
                return cols[:rank_xy]
            zname = f'{axis}_z'
            if zname not in out.columns:
                out, m = add_raw_feature(
                    out, feature=axis, data=data, name=zname, transform='zscore', meta=m)
            return [zname]

        gx = _get_cols(x_col)
        gy = _get_cols(y_col)

        xy_cols = []
        for cx in gx:
            for cy in gy:
                nm = f'{cx}*{cy}'
                if nm not in out.columns:
                    out[nm] = out[cx].to_numpy() * out[cy].to_numpy()
                xy_cols.append(nm)

        groups = dict(m.get('groups', {}))
        for nm in xy_cols:
            # checker-friendly passthrough group
            groups.setdefault(nm, []).append(nm)
        m['groups'] = groups

    return out, m


def add_eye_component_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    cols: tuple[str, ...] = ('RDz', 'RDy', 'LDz', 'LDy'),
    odd_kind: str = 'zscore',
    even_kind: str = 'log1p',
):
    """RD/LD components: odd/even encodings per channel."""
    out, m = design_df.copy(), dict(meta)
    for col in cols:
        out, m = add_raw_feature(out, feature=col, data=data, name=col,
                                 encoding='odd_even', odd_kind=odd_kind, even_kind=even_kind, meta=m)
    return out, m


def add_speed_features(
    design_df: pd.DataFrame,
    data: pd.DataFrame,
    meta: dict,
    *,
    col: str = 'speed',
):
    """Keep both log1p ramp and robust z-score of speed."""
    out, m = add_raw_feature(design_df, feature=col, data=data,
                             name='speed_log1p', transform='log1p', meta=meta)
    out, m = add_raw_feature(out,        feature=col, data=data,
                             name='speed_z',     transform='zscore', meta=m)
    return out, m


