from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import (
    prep_target_data,
)
from scipy import signal


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    signal: np.ndarray                  # (T,)
    bases: list[np.ndarray] = field(
        default_factory=list)  # each: (L, K), causal
    dt: float = 1.0                     # time step for lagging
    t_min: float = 0.0                  # minimum lag time


# =============================
# Design expansion
# =============================
def specs_to_design_df(
    specs: Dict[str, PredictorSpec],
    trial_ids: np.ndarray,
    *,
    respect_trial_boundaries: bool = True,
    edge: str = 'zero',         # 'zero' | 'drop' | 'renorm'
    add_intercept: bool = True,
    dtype: str = 'float64',
    drop_all_zero: bool = True,
    zero_atol: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    '''
    Expand PredictorSpecs into a basis-expanded GLM design DataFrame.

    If respect_trial_boundaries=False, lagging is continuous across trials.
    '''

    trial_ids = np.asarray(trial_ids).ravel()
    n = trial_ids.shape[0]

    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    bases_info: dict[str, list[tuple[int, int]]] = {}
    bases_by_predictor: dict[str, list[np.ndarray]] = {}

    valid_rows_mask = np.ones(n, dtype=bool)

    dropped_cols: list[str] = []
    dropped_predictors: set[str] = set()

    # choose lagging function ONCE
    if respect_trial_boundaries:
        lag_fn = lagged_design_from_signal_trials
    else:
        lag_fn = lagged_design_from_signal

    for name, ps in specs.items():
        groups[name] = []
        bases_info[name] = []

        sig = np.asarray(ps.signal, float).ravel()
        if sig.size != n:
            raise ValueError(
                f'Predictor {name!r} has length {sig.size}, expected {n}')

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
            if not added_any_for_name:
                continue

        else:
            kept_Bs: list[np.ndarray] = []

            for j, B in enumerate(ps.bases):
                B = np.asarray(B, float)
                if B.ndim != 2:
                    raise ValueError(f'Basis for {name!r} must be 2D (L, K).')

                if edge == 'drop':
                    if respect_trial_boundaries:
                        Phi, mask = lag_fn(
                            sig, B, trial_ids,
                            dt=ps.dt, t_min=ps.t_min,
                            edge=edge, return_edge_mask=True
                        )
                    else:
                        Phi, mask = lag_fn(
                            sig, B,
                            dt=ps.dt, t_min=ps.t_min,
                            edge=edge, return_edge_mask=True
                        )

                    if drop_all_zero and np.allclose(Phi, 0.0, atol=zero_atol):
                        L, K = B.shape
                        dropped_cols.extend(
                            [f'{name}:b{j}:{k}' for k in range(K)])
                        continue

                    valid_rows_mask &= mask

                else:
                    if respect_trial_boundaries:
                        Phi = lag_fn(sig, B, trial_ids, dt=ps.dt, t_min=ps.t_min, edge=edge)
                    else:
                        Phi = lag_fn(sig, B, dt=ps.dt, t_min=ps.t_min, edge=edge)

                    if drop_all_zero and np.allclose(Phi, 0.0, atol=zero_atol):
                        L, K = B.shape
                        dropped_cols.extend(
                            [f'{name}:b{j}:{k}' for k in range(K)])
                        continue

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

    X = np.column_stack(cols).astype(
        dtype, copy=False) if cols else np.empty((n, 0), dtype=dtype)
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
        'respect_trial_boundaries': bool(respect_trial_boundaries),
        'intercept_added': bool(add_intercept),
        'valid_rows_mask': valid_rows_mask if edge == 'drop' else None,
        'row_index_original': row_index_original,
        'bases_by_predictor': bases_by_predictor,
        'dropped_all_zero': {
            'enabled': bool(drop_all_zero),
            'zero_atol': float(zero_atol),
            'predictors': sorted(dropped_predictors),
            'columns': dropped_cols,
        },
        'dropped_all_zero_predictors': sorted(dropped_predictors),
    }

    return design_df, meta



def _a1d(x):
    return np.asarray(x).ravel()


def _n_lags(t_max, dt, t_min=0.0):
    return int(np.floor((t_max - t_min) / dt)) + 1


def _build_basis(family: str, n_basis: int, t_max: float, dt: float, *, t_min: float = 0.0):
    L = _n_lags(t_max, dt, t_min)
    K = max(1, min(int(n_basis), L))

    if family == 'rc':
        _, B = glm_bases.raised_log_cosine_basis(
            n_basis=K, t_max=t_max, dt=dt, t_min=t_min, log_spaced=True
        )
    elif family == 'spline':
        _, B = glm_bases.spline_basis(
            n_basis=K, t_max=t_max, dt=dt, t_min=t_min,
            degree=3, log_spaced=(t_max > 0.4)
        )
    else:
        raise ValueError("family must be 'rc' or 'spline'")

    return B


def add_stop_and_capture_columns(data: pd.DataFrame, trial_ids: Optional[np.ndarray] = None,
                                 ff_caught_T_new: Optional[np.ndarray] = None) -> pd.DataFrame:
    if 'whether_new_distinct_stop' in data.columns:
        data['stop'] = (data['whether_new_distinct_stop'] == 1).astype(int)
    elif 'monkey_speeddummy' in data.columns:
        stopped = (data['monkey_speeddummy'] == 0)
        if trial_ids is not None:
            prev = stopped.groupby(trial_ids, sort=False).shift(
                1, fill_value=False
            )
            data['stop'] = (stopped & ~prev).astype(int)
        else:
            data['stop'] = stopped.astype(int)
    else:
        raise KeyError(
            "need 'whether_new_distinct_stop' or 'monkey_speeddummy' to define stop onsets"
        )

    # ---------- builder: events + raw states (passthrough) ----------
    if 'capture_ff' not in data.columns:
        if ff_caught_T_new is not None:
            data = prep_target_data.add_capture_target(data, ff_caught_T_new)
        else:
            raise ValueError(
                'ff_caught_T_new is required to add capture_ff column')

    return data


def init_predictor_specs(
    data: pd.DataFrame,
    dt: float,
    trial_ids: Optional[np.ndarray] = None,
    *,
    trial_id_col: str = 'trial_id',
) -> Tuple[Dict[str, PredictorSpec], dict]:

    if trial_ids is None:
        if trial_id_col not in data.columns:
            raise KeyError(
                f'need trial_ids or a {trial_id_col!r} column in DataFrame')
        trial_ids = _a1d(data[trial_id_col])
    else:
        trial_ids = _a1d(trial_ids)

    if len(trial_ids) != len(data):
        raise ValueError('len(trial_ids) must equal len(data)')

    specs, meta = _init_predictor_specs(dt, trial_ids)

    return specs, meta


def _init_predictor_specs(
    dt: float,
    trial_ids: np.ndarray,
) -> Tuple[Dict[str, PredictorSpec], dict]:

    specs: Dict[str, PredictorSpec] = {}

    meta = {
        'trial_ids': np.asarray(trial_ids),
        'dt': float(dt),
        'raw_predictors': {},
        'bases_by_predictor': {},
        'bases_info_default': {},
        'basis_families': {},
        'dropped_all_zero_predictors': [],
        'builder': [],
    }

    return specs, meta


def add_state_predictors(
    specs: Dict[str, PredictorSpec],
    meta: dict,
    data: pd.DataFrame,
    *,
    state_cols: Sequence[str] = ['cur_vis',
                                 'nxt_vis', 'cur_in_memory', 'nxt_in_memory'],
    state_mode: str = 'passthrough',   # 'passthrough' | 'short'
    basis_family_state: str = 'rc',
    n_basis_state: int = 5,
    tmax_state: float = 0.30,
    tmin_state: float = 0.0,
    center_states: bool = False,
):

    required = state_cols
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f'missing required state columns: {missing}')

    B_state = (
        _build_basis(
            basis_family_state,
            n_basis_state,
            t_max=tmax_state,
            dt=meta['dt'],
            t_min=tmin_state,
        )
        if state_mode == 'short'
        else None
    )

    for name in required:
        x = _a1d(data[name]).astype(float, copy=False)
        if center_states:
            x -= np.nanmean(x)

        bases = [] if state_mode == 'passthrough' else [B_state]
        specs[name] = PredictorSpec(signal=x, bases=bases, dt=meta['dt'], t_min=tmin_state)

        meta['raw_predictors'][name] = x
        meta['bases_by_predictor'][name] = bases
        meta['bases_info_default'][name] = [b.shape for b in bases]

    meta['basis_families']['state'] = (
        basis_family_state if state_mode == 'short' else None
    )
    meta['builder'].append(f'states_{state_mode}')

    return specs, meta


def add_event_predictors(
    specs: Dict[str, PredictorSpec],
    meta: dict,
    data: pd.DataFrame,
    *,
    events_to_include: Sequence[str],
    basis_family_event: str = 'rc',
    n_basis_event: int = 6,
    tmax_event: float = 0.60,
    tmin_event: float = 0.0,
):

    for event in events_to_include:
        if event not in data.columns:
            raise KeyError(f'missing event column: {event}')

    B_event = _build_basis(
        basis_family_event,
        n_basis_event,
        t_max=tmax_event,
        dt=meta['dt'],
        t_min=tmin_event,
    )

    for event in events_to_include:
        x = _a1d(data[event]).astype(float, copy=False)
        specs[event] = PredictorSpec(signal=x, bases=[B_event], dt=meta['dt'], t_min=tmin_event)

        meta['raw_predictors'][event] = x
        meta['bases_by_predictor'][event] = [B_event]
        meta['bases_info_default'][event] = [B_event.shape]

        if np.allclose(x, 0):
            meta['dropped_all_zero_predictors'].append(event)

    meta['basis_families']['event'] = basis_family_event
    meta['builder'].append('events_temporal')

    return specs, meta


import numpy as np
from scipy import signal


def _shift_non_circular(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift along axis 0 without circular wrap-around.
    Positive shift moves signal forward in time.
    """
    out = np.zeros_like(arr)

    if shift > 0:
        out[shift:] = arr[:-shift]
    elif shift < 0:
        out[:shift] = arr[-shift:]
    else:
        out = arr.copy()

    return out


def lagged_design_from_signal(
    x: np.ndarray,
    basis: np.ndarray,
    *,
    dt: float,
    t_min: float,
    edge: str = 'zero',   # 'zero' | 'drop' | 'renorm'
    return_edge_mask: bool = False
):
    """
    Time-aligned lagged design that correctly handles arbitrary t_min.

    Matches MATLAB: conv + crop + circshift,
    but uses stable FIR implementation.
    """

    x = np.asarray(x, float).ravel()
    B = np.asarray(basis, float)

    if B.ndim != 2:
        raise ValueError('basis must be 2D (L, K)')

    L, K = B.shape
    T = x.shape[0]

    out = np.zeros((T, K), float)
    edge_mask = np.zeros(T, dtype=bool)

    # Compute shift required to align lag=0 correctly
    shift_bins = int(round(t_min / dt))

    if edge == 'renorm':
        cum = np.cumsum(B, axis=0)
        full = cum[-1, :]

    for k in range(K):
        h = B[:, k]

        # Causal FIR
        y = signal.lfilter(h, [1.0], x)

        if edge == 'renorm':
            last = np.minimum(np.arange(T), L - 1)
            avail = cum[last, k]
            scale = full[k] / np.clip(avail, 1e-12, None)
            y *= scale
            
            
        #print('num shift_bins', shift_bins)

        # Apply non-circular alignment shift
        y = _shift_non_circular(y[:, None], shift_bins).ravel()

        out[:, k] = y

    edge_mask[:] = True

    return (out, edge_mask) if return_edge_mask else out

def lagged_design_from_signal_trials(
    x: np.ndarray,
    basis: np.ndarray,
    trial_ids: np.ndarray,
    *,
    dt: float,
    t_min: float,
    edge: str = 'zero',   # 'zero' | 'drop' | 'renorm'
    return_edge_mask: bool = False
):
    """
    Trial-local version with correct time alignment.
    No cross-trial leakage.
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

    shift_bins = int(round(t_min / dt))

    if edge == 'renorm':
        cum = np.cumsum(B, axis=0)
        full = cum[-1, :]

    unique_trials = np.unique(trial_ids)

    for tr in unique_trials:
        idx = np.flatnonzero(trial_ids == tr)
        xt = x[idx]
        Tt = xt.size

        if Tt == 0:
            continue

        for k in range(K):
            h = B[:, k]
            y = signal.lfilter(h, [1.0], xt)

            if edge == 'renorm':
                last = np.minimum(np.arange(Tt), L - 1)
                avail = cum[last, k]
                scale = full[k] / np.clip(avail, 1e-12, None)
                y *= scale

            # Apply alignment shift inside trial
            y = _shift_non_circular(y[:, None], shift_bins).ravel()

            out[idx, k] = y

        edge_mask[idx] = True

    return (out, edge_mask) if return_edge_mask else out