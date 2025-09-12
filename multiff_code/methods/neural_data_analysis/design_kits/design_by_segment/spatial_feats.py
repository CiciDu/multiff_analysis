from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Tuple, List, Mapping

import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import BSpline

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.design_kits.design_by_segment import temporal_feats


def _knots_from_breaks(breaks: np.ndarray, degree: int) -> np.ndarray:
    """Clamped knot vector from distinct breakpoints (incl. endpoints)."""
    br = np.asarray(breaks, float).ravel()
    if br.size < 2:
        raise ValueError('`breaks` must have at least 2 values.')
    br = np.unique(br)
    return np.concatenate([np.full(degree + 1, br[0]), br[1:-1], np.full(degree + 1, br[-1])])


def _build_spatial_knots(
    x: np.ndarray,
    *,
    mode: str = 'percentile',            # 'percentile' | 'breaks' | 'knots'
    degree: int = 3,
    K: int = 6,                          # used only in 'percentile' mode
    percentiles: tuple[float, float] = (2, 98),
    breaks: np.ndarray | None = None,    # used in 'breaks' mode
    knots:  np.ndarray | None = None     # used in 'knots'  mode (full clamped)
) -> tuple[np.ndarray, int]:
    """
    Returns (full_knots, K) where len(full_knots) = K + degree + 1.
    """
    x = np.asarray(x, float).ravel()
    deg = int(degree)

    if mode == 'percentile':
        if K < 1:
            raise ValueError('K must be >= 1 in percentile mode.')
        lo, hi = np.nanpercentile(x, percentiles)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(x), np.nanmax(x)
        M = max(0, K - deg - 1)  # # internal knots
        if M > 0:
            internal = np.linspace(lo, hi, M + 2)[1:-1]
            full_knots = np.concatenate(
                [np.full(deg + 1, lo), internal, np.full(deg + 1, hi)])
        else:
            full_knots = np.concatenate(
                [np.full(deg + 1, lo), np.full(deg + 1, hi)])

    elif mode == 'breaks':
        if breaks is None:
            raise ValueError("Provide `breaks` when mode='breaks'.")
        full_knots = _knots_from_breaks(breaks, deg)
        K = full_knots.size - deg - 1

    elif mode == 'knots':
        if knots is None:
            raise ValueError("Provide `knots` when mode='knots'.")
        full_knots = np.asarray(knots, float).ravel()
        K = full_knots.size - deg - 1
        if K < 1:
            raise ValueError('Invalid `knots`: imply K < 1.')
    else:
        raise ValueError("mode must be 'percentile', 'breaks', or 'knots'.")

    return full_knots, int(K)


def _bspline_design_from_knots(x: np.ndarray, full_knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    deg = int(degree)
    K = full_knots.size - deg - 1
    X = np.empty((x.size, K), float)
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1.0
        spl = BSpline(full_knots, c, deg, extrapolate=False)
        X[:, k] = np.nan_to_num(spl(x), nan=0.0, posinf=0.0, neginf=0.0)
    return X


def add_spatial_spline_feature(
    design_df: pd.DataFrame,
    feature: str | np.ndarray | pd.Series,
    *,
    name: str | None = None,
    data: pd.DataFrame | None = None,
    rows_mask: np.ndarray | None = None,

    # spline controls
    knots_mode: str = 'percentile',              # 'percentile' | 'breaks' | 'knots'
    K: int = 6,                                  # used in percentile mode
    degree: int = 3,
    percentiles: tuple[float, float] = (2, 98),  # for percentile mode
    breaks: np.ndarray | None = None,            # for breaks mode
    # for knots mode (full clamped vector)
    knots:  np.ndarray | None = None,

    # identifiability & scaling
    # True/False or 'auto' (True only in percentile mode)
    row_normalize: str | bool = 'auto',
    drop_one: bool = True,                       # drop one column (baseline)
    # center columns (orthogonal to intercept)
    center: bool = True,

    meta: dict | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    # --- coerce the vector ---
    if isinstance(feature, str):
        if data is None:
            raise ValueError('When feature is a column name, pass `data=`.')
        vec = data[feature].to_numpy()
        if rows_mask is not None:
            vec = vec[rows_mask]
        feat_name = name or feature
    else:
        vec = np.asarray(feature).ravel()
        if rows_mask is not None:
            vec = vec[rows_mask]
        feat_name = name or 'spatial'

    # --- build knots & design ---
    full_knots, K_eff = _build_spatial_knots(
        vec, mode=knots_mode, degree=degree, K=K, percentiles=percentiles, breaks=breaks, knots=knots
    )
    X = _bspline_design_from_knots(vec, full_knots, degree)

    # row-normalize if desired (usually only needed in percentile mode)
    if row_normalize == 'auto':
        do_norm = (knots_mode == 'percentile')
    else:
        do_norm = bool(row_normalize)
    if do_norm:
        s = X.sum(axis=1, keepdims=True)
        good = s.squeeze() > 0
        X[good] /= s[good]

    # identifiability tweaks
    colnames = [f'{feat_name}:s{k}' for k in range(X.shape[1])]
    if drop_one and X.shape[1] >= 2:
        X = X[:, :-1]
        colnames = colnames[:-1]
    if center and X.size:
        X = X - X.mean(axis=0, keepdims=True)

    # write into the design
    out = design_df.copy()
    out[colnames] = X

    # update meta
    if meta is None:
        return out, None

    meta = dict(meta)
    g = dict(meta.get('groups', {}))
    g.setdefault(feat_name, [])
    g[feat_name].extend(colnames)
    meta['groups'] = g
    ss = dict(meta.get('spatial_specs', {}))
    ss[feat_name] = {
        'mode': knots_mode,
        'degree': int(degree),
        'K': int(K_eff),
        'drop_one': bool(drop_one),
        'center': bool(center),
        'row_normalize': do_norm,
        'percentiles': tuple(percentiles) if knots_mode == 'percentile' else None,
        'breaks': breaks.tolist() if isinstance(breaks, np.ndarray) else None,
        'knots': full_knots.tolist(),
    }
    meta['spatial_specs'] = ss
    meta['spatial_cols'] = list(meta.get('spatial_cols', [])) + colnames
    return out, meta


# =============================
# Circular (Fourier) feature
# =============================

def add_circular_fourier_feature(
    design_df: pd.DataFrame,
    theta: str | np.ndarray | pd.Series,   # column name or vector of angles
    *,
    # logical feature name (e.g., 'cur_angle')
    name: str | None = None,
    data: pd.DataFrame | None = None,      # required if theta is a column name
    # mask used when you built design_df (edge='drop')
    rows_mask: np.ndarray | None = None,
    # 0/1 mask to “turn on” the angle rows
    gate: str | np.ndarray | pd.Series | None = None,
    M: int = 3,                            # number of harmonics
    degrees: bool = False,                 # set True if theta is in degrees
    center: bool = True,                   # column-center -> orthogonal to intercept
    # per-column unit variance (after centering)
    standardize: bool = False,
    meta: dict | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Add sin/cos Fourier harmonics of an angle into design_df.

    Columns created: '{name}:sin1', '{name}:cos1', ..., up to M.
    If `gate` is provided, each column is multiplied by (gate > 0).
    """
    # --- coerce theta to a numeric vector aligned with design_df rows ---
    def _get_vec(v, label):
        if isinstance(v, str):
            if data is None:
                raise ValueError(
                    f'When {label} is a column name, pass `data=`.')
            arr = data[v].to_numpy()
        else:
            arr = np.asarray(v).ravel()
        if rows_mask is not None and arr.shape[0] == (rows_mask.shape[0]):
            arr = arr[rows_mask]
        if arr.shape[0] != len(design_df):
            raise ValueError(
                f'Length of {label} ({arr.shape[0]}) must equal len(design_df) ({len(design_df)}).')
        return arr

    th = _get_vec(theta, 'theta')
    if degrees:
        th = np.deg2rad(th)
    # wrap to [-pi, pi]
    th = np.angle(np.exp(1j * th))

    if gate is not None:
        g = _get_vec(gate, 'gate')
        g = (g > 0).astype(float)
    else:
        g = 1.0

    # --- build harmonics ---
    cols = []
    names = []
    for m in range(1, int(M) + 1):
        s = np.sin(m * th) * g
        c = np.cos(m * th) * g
        cols.append(s)
        names.append(f'{name or "angle"}:sin{m}')
        cols.append(c)
        names.append(f'{name or "angle"}:cos{m}')

    X = np.column_stack(cols) if cols else np.empty((len(design_df), 0))

    # --- identifiability/scaling ---
    if center and X.size:
        X = X - X.mean(axis=0, keepdims=True)
    if standardize and X.size:
        sd = X.std(axis=0, ddof=1)
        sd[sd < 1e-12] = 1.0
        X = X / sd

    # --- write into design_df ---
    out = design_df.copy()
    out[names] = X

    # --- update meta ---
    if meta is None:
        return out, None

    meta = dict(meta)
    groups = dict(meta.get('groups', {}))
    groups.setdefault(name or 'angle', [])
    groups[name or 'angle'].extend(names)
    meta['groups'] = groups

    circ = dict(meta.get('circular_specs', {}))
    circ[name or 'angle'] = {
        'M': int(M),
        'degrees': bool(degrees),
        'center': bool(center),
        'standardize': bool(standardize),
        'gated': gate is not None,
        'colnames': names,
    }
    meta['circular_specs'] = circ
    meta['circular_cols'] = list(meta.get('circular_cols', [])) + names
    return out, meta


def add_visibility_transition_kernels(
    specs: Dict[str, temporal_feats.PredictorSpec],
    data: pd.DataFrame,
    trial_ids: np.ndarray,
    dt: float,
    *,
    stems: Sequence[str] = ('cur_vis', 'nxt_vis'),
    basis: np.ndarray | None = None,
    family: str = 'rc',          # 'rc' | 'spline' (used if basis is None)
    n_basis: int = 3,
    t_max: float = 0.30,
    t_min: float = 0.0,
    log_spaced: bool = True,
    inplace: bool = False,
) -> Dict[str, temporal_feats.PredictorSpec]:
    """
    Add short-latency transition kernels for visibility flags.

    For each stem in `stems` (e.g., 'cur_vis'), this creates two event predictors:
        - '{stem}_on'  : 0->1 transitions (per-trial)
        - '{stem}_off' : 1->0 transitions (per-trial)
    Each gets a short causal temporal basis (default raised-cosine ~300 ms).

    Parameters
    ----------
    specs : dict[str, PredictorSpec]
        Existing predictor specs (will be copied unless inplace=True).
    data : DataFrame
        Must contain boolean/int columns for each `stem`.
    trial_ids : array-like
        Trial segmentation vector aligned with `data` rows.
    dt : float
        Bin width (s) for constructing the temporal basis (if `basis` is None).
    stems : sequence of str
        Column names in `data` to treat as visibility flags.
    basis : (L, K) array or None
        If provided, use this basis for all stems. Otherwise build from args.
    family : str
        'rc' (raised-cosine) or 'spline' (B-spline) when `basis` is None.
    n_basis, t_max, t_min, log_spaced :
        Basis controls passed to glm_bases.*_basis when `basis` is None.
    inplace : bool
        If True, mutate `specs`. Otherwise return a shallow-copied dict.

    Returns
    -------
    specs_out : dict[str, PredictorSpec]
        Updated mapping with '{stem}_on' and '{stem}_off' entries added.
    """
    if not inplace:
        specs = dict(specs)

    tid = np.asarray(trial_ids).ravel()
    if len(tid) != len(data):
        raise ValueError('trial_ids length must match len(data)')

    # Build or validate basis
    if basis is None:
        if family == 'rc':
            _, B = glm_bases.raised_cosine_basis(
                n_basis=int(n_basis), t_max=float(t_max), dt=float(dt),
                t_min=float(t_min), log_spaced=bool(log_spaced)
            )
        elif family == 'spline':
            _, B = glm_bases.spline_basis(
                n_basis=int(n_basis), t_max=float(t_max), dt=float(dt),
                t_min=float(t_min), degree=3, log_spaced=bool(log_spaced)
            )
        else:
            raise ValueError("family must be 'rc' or 'spline'")
    else:
        B = np.asarray(basis, float)
        if B.ndim != 2:
            raise ValueError('basis must be 2D (L, K)')

    # Per-trial on/off detector
    def _on_off(flag_bool: pd.Series | np.ndarray, tid_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(flag_bool, pd.Series):
            f = flag_bool.astype(bool)
        else:
            f = pd.Series(np.asarray(flag_bool, dtype=bool))
        prev = f.groupby(tid_vec, sort=False).shift(1, fill_value=False)
        on = (f & ~prev).to_numpy(dtype=float, copy=False)
        off = (~f & prev).to_numpy(dtype=float, copy=False)
        return on, off

    # make sure stems is a list or tuple
    if not isinstance(stems, (list, tuple)):
        stems = (stems,)

    for stem in stems:
        if stem not in data.columns:
            raise KeyError(f'missing column {stem!r} in data')

        on, off = _on_off(data[stem], tid)
        specs[f'{stem}_on'] = temporal_feats.PredictorSpec(
            signal=on,  bases=[B])
        specs[f'{stem}_off'] = temporal_feats.PredictorSpec(
            signal=off, bases=[B])

    return specs
