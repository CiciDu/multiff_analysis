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
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils



@dataclass(frozen=True, slots=True)
class PredictorSpec:
    signal: np.ndarray                  # (T,)
    bases: list[np.ndarray] = field(default_factory=list)  # each: (L, K), causal


# =============================
# Design expansion
# =============================
def specs_to_design_df(
    specs: Dict[str, PredictorSpec],
    trial_ids: np.ndarray,
    *,
    edge: str = 'zero',         # 'zero' | 'drop' | 'renorm'
    add_intercept: bool = True,
    dtype: str = 'float64',
    drop_all_zero: bool = True,
    zero_atol: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    '''
    Expand PredictorSpecs into a basis-expanded GLM design DataFrame.

    For edge='drop', we keep rows valid for *all* kept basis-expanded predictors
    (intersection across masks). All-zero passthrough predictors and all-zero
    basis blocks are omitted and recorded in meta['dropped_all_zero'].
    '''
    trial_ids = np.asarray(trial_ids).ravel()
    n = trial_ids.shape[0]

    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    bases_info: dict[str, list[tuple[int, int]]] = {}
    bases_by_predictor: dict[str, list[np.ndarray]] = {}

    # intersection for edge='drop' (start all True; only kept basis blocks constrain)
    valid_rows_mask = np.ones(n, dtype=bool)

    # bookkeeping for dropped content
    dropped_cols: list[str] = []
    dropped_predictors: set[str] = set()

    for name, ps in specs.items():
        groups[name] = []
        bases_info[name] = []

        sig = np.asarray(ps.signal, float).ravel()
        if sig.size != n:
            raise ValueError(f'Predictor {name!r} has length {sig.size}, expected {n}')

        added_any_for_name = False

        # passthrough
        if not ps.bases:
            if drop_all_zero and np.allclose(sig, 0.0, atol=zero_atol):
                dropped_predictors.add(name)
            else:
                colname = f'{name}'
                cols.append(sig.astype(dtype, copy=False))
                names.append(colname)
                groups[name].append(colname)
                added_any_for_name = True
            # no bases registered for passthrough
            if not added_any_for_name:
                continue
        else:
            kept_Bs: list[np.ndarray] = []
            # basis-expanded predictors
            for j, B in enumerate(ps.bases):
                B = np.asarray(B, float)
                if B.ndim != 2:
                    raise ValueError(f'Basis for {name!r} must be 2D (L, K).')
                if edge == 'drop':
                    Phi, mask = lagged_design_from_signal_trials(sig, B, trial_ids,
                                                                 edge=edge, return_edge_mask=True)
                    # if the whole block is zero, skip it (do not constrain mask)
                    if drop_all_zero and np.allclose(Phi, 0.0, atol=zero_atol):
                        L, K = B.shape
                        dropped_cols.extend([f'{name}:b{j}:{k}' for k in range(K)])
                        continue
                    valid_rows_mask &= mask
                else:
                    Phi = lagged_design_from_signal_trials(sig, B, trial_ids, edge=edge)
                    if drop_all_zero and np.allclose(Phi, 0.0, atol=zero_atol):
                        L, K = B.shape
                        dropped_cols.extend([f'{name}:b{j}:{k}' for k in range(K)])
                        continue

                # keep this block
                L, K = B.shape
                bases_info[name].append((L, K))
                kept_Bs.append(B)
                for k in range(Phi.shape[1]):
                    colname = f'{name}:b{j}:{k}'
                    cols.append(Phi[:, k].astype(dtype, copy=False))
                    names.append(colname)
                    groups[name].append(colname)
                added_any_for_name = True

            if kept_Bs:
                bases_by_predictor[name] = kept_Bs
            if not added_any_for_name:
                dropped_predictors.add(name)

    # build frame
    X = np.column_stack(cols).astype(dtype, copy=False) if cols else np.empty((n, 0), dtype=dtype)
    design_df = pd.DataFrame(X, columns=names)

    row_index_original = None
    if edge == 'drop':
        row_index_original = np.flatnonzero(valid_rows_mask)
        design_df = design_df.loc[valid_rows_mask].reset_index(drop=True)

    if add_intercept:
        design_df.insert(0, 'const', 1.0)

    meta = {
        'groups': groups,
        'bases': bases_info,
        'edge': edge,
        'intercept_added': bool(add_intercept),
        'valid_rows_mask': valid_rows_mask if edge == 'drop' else None,
        'row_index_original': row_index_original,
        'bases_by_predictor': bases_by_predictor,  # only keys that actually have kept bases
        'dropped_all_zero': {
            'enabled': bool(drop_all_zero),
            'zero_atol': float(zero_atol),
            'predictors': sorted(dropped_predictors),
            'columns': dropped_cols,
        },
        'dropped_all_zero_predictors': sorted(dropped_predictors),
    }
    return design_df, meta



def build_predictor_specs_from_behavior(
    data: pd.DataFrame,
    dt: float,
    trial_ids: np.ndarray | None = None,
    *,
    events_to_include: Optional[Sequence[str]] = ['stop', 'capture_ff'],
    trial_id_col: str = 'trial_id',
    # temporal basis for true events
    basis_family_event: str = 'rc',   # 'rc' | 'spline'
    n_basis_event: int = 6,
    tmax_event: float = 0.60,
    # how to treat state columns (visibility / memory)
    state_mode: str = 'passthrough',  # 'passthrough' | 'short'
    basis_family_state: str = 'rc',   # used if state_mode == 'short'
    n_basis_state: int = 5,
    tmax_state: float = 0.30,
    center_states: bool = False,      # subtract mean from state columns
    column_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, PredictorSpec], dict]:
    """
    Build predictor->spec dict:
      - States (cur_vis, nxt_vis, cur_in_memory, nxt_in_memory): passthrough or short causal basis
      - Events (stop, capture): event window basis
    Spatial/circular features are added later.
    """
    def _a1d(x): return np.asarray(x).ravel()
    def _n_lags(t_max, dt, t_min=0.0): return int(np.floor((t_max - t_min)/dt)) + 1

    def _build_basis(family: str, n_basis: int, t_max: float, dt: float, *, t_min: float = 0.0) -> np.ndarray:
        L = _n_lags(t_max, dt, t_min)
        K = max(1, min(int(n_basis), L))
        if family == 'rc':
            _, B = glm_bases.raised_cosine_basis(n_basis=K, t_max=t_max, dt=dt, t_min=t_min, log_spaced=True)
        elif family == 'spline':
            _, B = glm_bases.spline_basis(n_basis=K, t_max=t_max, dt=dt, t_min=t_min, degree=3, log_spaced=(t_max > 0.4))
        else:
            raise ValueError("family must be 'rc' or 'spline'")
        return B

    # remap
    column_map = {} if column_map is None else dict(column_map)
    col = lambda name: column_map.get(name, name)

    needed = {'cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory', 'stop', 'capture_ff'}
    missing = [c for c in needed if col(c) not in data.columns]
    if missing:
        raise KeyError(f'missing required columns in DataFrame: {missing}')

    # trial ids
    if trial_ids is None:
        if col(trial_id_col) not in data.columns:
            raise KeyError(f'need trial_ids or a {trial_id_col!r} column in DataFrame')
        trial_ids = _a1d(data[col(trial_id_col)])
    else:
        trial_ids = _a1d(trial_ids)
    if len(trial_ids) != len(data):
        raise ValueError('len(trial_ids) must equal len(data)')

    # bases
    B_event = _build_basis(basis_family_event, n_basis_event, t_max=tmax_event, dt=dt, t_min=0.0)
    B_state = _build_basis(basis_family_state, n_basis_state, t_max=tmax_state, dt=dt, t_min=0.0) \
              if state_mode == 'short' else None

    # states
    cur_vis       = _a1d(data[col('cur_vis')]).astype(float, copy=False)
    nxt_vis       = _a1d(data[col('nxt_vis')]).astype(float, copy=False)
    cur_in_memory = _a1d(data[col('cur_in_memory')]).astype(float, copy=False)
    nxt_in_memory = _a1d(data[col('nxt_in_memory')]).astype(float, copy=False)

    if center_states:
        for arr in (cur_vis, nxt_vis, cur_in_memory, nxt_in_memory):
            arr -= np.nanmean(arr)

    specs: Dict[str, PredictorSpec] = {}

    def add_state(name, x):
        if state_mode == 'passthrough':
            specs[name] = PredictorSpec(signal=x, bases=[])          # one raw column
        else:  # 'short'
            specs[name] = PredictorSpec(signal=x, bases=[B_state])   # short causal basis

    add_state('cur_vis',       cur_vis)
    add_state('nxt_vis',       nxt_vis)
    add_state('cur_in_memory', cur_in_memory)
    add_state('nxt_in_memory', nxt_in_memory)

    # true events (delta at event time) get event window basis
    for event in events_to_include:
        specs[event]    = PredictorSpec(signal=_a1d(data[col(event)]).astype(float, copy=False),       bases=[B_event])

    meta = {
        'trial_ids': np.asarray(trial_ids),
        'dt': float(dt),
        'raw_predictors': {k: specs[k].signal for k in specs},
        'bases_info_default': {k: [b.shape for b in specs[k].bases] for k in specs},
        'bases_by_predictor': {k: list(specs[k].bases) for k in specs},
        'basis_families': {'event': basis_family_event, 'state': (basis_family_state if state_mode == 'short' else None)},
        'B_hist': glm_bases.raised_cosine_basis(n_basis=5, t_max=0.20, dt=dt, t_min=dt)[1],
        'dropped_all_zero_predictors': [k for k in events_to_include if np.allclose(specs[k].signal, 0)],
        # gates for angle features (use column *names* so you can pull from `data` later)
        'angle_gates': {'cur_angle': col('cur_in_memory'), 'nxt_angle': col('nxt_in_memory')},
        'builder': f'states_{state_mode}_events_temporal',
    }
    return specs, meta



def add_spike_history(
    design_df: pd.DataFrame,
    spike_counts: np.ndarray | pd.Series,
    trial_ids: np.ndarray,
    dt: float,
    *,
    n_basis: int = 5,
    t_max: float = 0.20,
    t_min: float | None = None,
    prefix: str = 'spk_hist',
    style: str = 'bjk',                 # 'bjk' or 'rc'
    edge: str = 'zero',
    basis: np.ndarray | None = None,
    meta: dict | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Add trial-aware spike history columns to an existing design_df.
    Excludes lag-0 by default (t_min = dt) to avoid self-count leakage in Poisson GLMs.
    """
    x = np.asarray(spike_counts, float).ravel()
    trial_ids = np.asarray(trial_ids).ravel()
    if len(x) != len(design_df) or len(trial_ids) != len(design_df):
        raise ValueError('Lengths must match: len(spike_counts) == len(trial_ids) == len(design_df)')

    if basis is None:
        if t_min is None:
            t_min = dt  # exclude lag-0 by default
        _, basis = glm_bases.raised_cosine_basis(
            n_basis=n_basis, t_max=t_max, dt=dt, t_min=t_min, log_spaced=True)
    L, K = basis.shape

    if edge == 'drop':
        Xh, _ = lagged_design_from_signal_trials(x, basis, trial_ids, edge=edge, return_edge_mask=True)
    else:
        Xh = lagged_design_from_signal_trials(x, basis, trial_ids, edge=edge)

    if style == 'bjk':
        colnames = [f'{prefix}:b0:{k}' for k in range(K)]
        meta_key = prefix
    elif style == 'rc':
        colnames = [f'hist_rc{k}' for k in range(K)]
        meta_key = 'spk_hist'
    else:
        raise ValueError("style must be 'bjk' or 'rc'")

    out = design_df.copy()
    out[colnames] = Xh

    if meta is not None:
        meta = dict(meta)
        groups = dict(meta.get('groups', {}))
        groups.setdefault(meta_key, [])
        groups[meta_key].extend(colnames)
        meta['groups'] = groups

        bases_info = dict(meta.get('bases', {}))
        bases_info.setdefault(meta_key, [])
        bases_info[meta_key].append((L, K))
        meta['bases'] = bases_info

        bmap = dict(meta.get('bases_by_predictor', {}))
        blist = list(bmap.get(meta_key, []))
        blist.append(basis)
        bmap[meta_key] = blist
        meta['bases_by_predictor'] = bmap
        return out, meta

    return out, None


def inverse_hist_weights(x, bins='fd', eps=1e-6):
    """Inverse-occupancy weights (mean ≈ 1)."""
    x = np.asarray(x, float).ravel()
    h, edges = np.histogram(x[~np.isnan(x)], bins=bins)
    p = h / max(h.sum(), eps)
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(h) - 1)
    w = 1.0 / np.clip(p[idx], eps, None)
    return w / np.mean(w)


def lagged_design_from_signal_trials(
    x: np.ndarray,
    basis: np.ndarray,
    trial_ids: np.ndarray,
    *,
    edge: str = 'zero',   # 'zero' | 'drop' | 'renorm'
    return_edge_mask: bool = False
):
    """
    Trial-local causal design: Phi[t,k] = sum_{ℓ>=0} basis[ℓ,k] * x[t-ℓ], with NO cross-trial leakage.

    edge='zero'  : zero-pad at trial starts.
    edge='drop'  : only rows with a full L-length window set; first L-1 bins per trial invalid.
    edge='renorm': like 'zero' but rescales early rows so kernel mass matches full sum.
    """
    x = np.asarray(x, float).ravel()
    B = np.asarray(basis, float)
    trial_ids = np.asarray(trial_ids)
    if x.shape[0] != trial_ids.shape[0]:
        raise ValueError('x and trial_ids must have same length')
    if B.ndim != 2:
        raise ValueError('basis must be 2D (L, K)')

    L, K = B.shape
    T = x.shape[0]
    out = np.zeros((T, K), float)
    edge_mask = np.zeros(T, dtype=bool)

    # pre for renorm
    if edge == 'renorm':
        cum = np.cumsum(B, axis=0)  # (L, K)
        full = cum[-1, :]           # (K,)

    # stable unique in first-appearance order
    _, first_idx = np.unique(trial_ids, return_index=True)
    uniq = trial_ids[np.sort(first_idx)]

    for tr in uniq:
        idx = np.flatnonzero(trial_ids == tr)
        xt = x[idx]
        Tt = idx.size
        if Tt == 0:
            continue

        if edge in ('zero', 'renorm'):
            # causal FIR via lfilter (fast, stable)
            for k in range(K):
                h = B[:, k]
                y = signal.lfilter(h, [1.0], xt)
                if edge == 'renorm':
                    last = np.minimum(np.arange(Tt), L - 1)
                    avail = cum[last, k]
                    scale = full[k] / np.clip(avail, 1e-12, None)
                    y = y * scale
                out[idx, k] = y
            edge_mask[idx] = True

        elif edge == 'drop':
            if Tt >= L:
                W = np.lib.stride_tricks.sliding_window_view(xt, L)  # (Tt-L+1, L)
                # multiply by reversed basis columns
                Brev = B[::-1, :]  # (L, K)
                Y = W @ Brev
                out[idx[L - 1:], :] = Y
                edge_mask[idx[L - 1:]] = True
        else:
            raise ValueError("edge must be one of {'zero','drop','renorm'}")

    return (out, edge_mask) if return_edge_mask else out
