"""
GAM setup (basis + smooth definitions) to match the paper’s structure as closely as possible
while staying compatible with your gdh.smooths_handler API.

Paper mapping:
- f(·) tuning functions: 10 boxcars  -> here: spline smooth with 10 basis funcs via evenly-spaced knots
- g(·) event temporal filters: 10 raised-cos spanning 600ms
    - target-onset causal [0, 600]ms
    - others acausal [-300, 300]ms
  -> here: temporal-kernel smooths (B-spline basis) over 600ms with causal vs acausal direction
- h(·) spike-history: 10 causal raised-cos on log time, 350ms
  -> here: causal temporal-kernel smooth over 350ms on the binned spike train
- p(·) coupling: 10 causal raised-cos on log time, 1.375s
  -> here: causal temporal-kernel smooth over 1375ms on other units’ binned spike trains

You can swap in your own kernels later by replacing the temporal-kernel smooths with explicit
convolution regressors, but this should get you very close in practice.
"""
from __future__ import annotations
import sys
from pathlib import Path

# PGAM external library
PGAM_PATH = Path(
    'multiff_analysis/external/pgam/src'
).expanduser().resolve()
if str(PGAM_PATH) not in sys.path:
    sys.path.append(str(PGAM_PATH))
    

from PGAM.GAM_library import *

import PGAM.gam_data_handlers as gdh
import numpy as np

import os
import sys
from pathlib import Path


def find_project_root(marker="multiff_analysis"):
    """Search upward until we find a folder containing `marker`."""
    cur = Path(os.getcwd()).resolve()   # use CWD instead of __file__
    for parent in [cur] + list(cur.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root with marker '{marker}'")


project_root = find_project_root()

# Build the paths relative to project root
pgam_src = project_root / "multiff_analysis" / "external" / "pgam" / "src"
pgam_src_pg = pgam_src / "PGAM"

for path in [pgam_src, pgam_src_pg]:
    if str(path) not in sys.path:
        sys.path.append(str(path))


# -------------------------
# Helpers
# -------------------------


def build_smooth_handler(
    data_obj,
    unit_idx,
    covariate_names,
    tuning_covariates=None,
    use_cyclic=None,
    order=4,
):
    """
    Build a PGAM smooth handler for a single unit.

    Assumes covariates, spike counts, and events have already been computed.
    """
    if tuning_covariates is None:
        tuning_covariates = covariate_names

    if use_cyclic is None:
        use_cyclic = set()

    data_obj.sm_handler = build_smooth_handler_for_unit(
        unit_idx=unit_idx,
        covariates_concat=data_obj.covariates,
        covariate_names=covariate_names,
        trial_id_vec=data_obj.covariate_trial_ids,
        Y_binned=data_obj.Y,
        all_events=data_obj.events,
        dt=data_obj.prs.dt,
        tuning_covariates=tuning_covariates,
        use_cyclic=use_cyclic,
        order=order,
    )
    return data_obj.sm_handler


def _compute_equal_knots(x: np.ndarray, num_bins: int) -> np.ndarray:
    """Return equally-spaced knot locations across the finite range of x."""
    x_finite = x[np.isfinite(x)]
    if x_finite.size == 0:
        raise ValueError('Predictor has no finite values.')
    x_min = float(np.min(x_finite))
    x_max = float(np.max(x_finite))
    if x_max <= x_min:
        # Degenerate predictor; fall back to a tiny span
        x_max = x_min + 1.0
    return np.linspace(x_min, x_max, num_bins)


def add_tuning_smooth_10boxcar_approx(
    sm_handler,
    name: str,
    x: np.ndarray,
    trial_ids: np.ndarray,
    num_basis: int = 10,
    order: int = 4,
    is_cyclic: bool = False,
) -> None:
    """
    Paper: f tuning = 10 boxcars.
    Here: approximate with a spline smooth using knots placed so you get ~10 basis functions.

    For many GAM implementations, #basis ~ (#internal_knots + order).
    You can adjust order/knots if you want it tighter/looser.
    """
    # Use num_basis equally spaced knots over x-range; many libs interpret "knots" as positions.
    knots = _compute_equal_knots(x, num_basis)

    sm_handler.add_smooth(
        name,
        [x],
        knots=[knots],
        ord=order,
        is_temporal_kernel=False,
        trial_idx=trial_ids,
        is_cyclic=[bool(is_cyclic)],
    )


def add_event_temporal_filter(
    sm_handler,
    name: str,
    event_impulse: np.ndarray,
    trial_ids: np.ndarray,
    dt_ms: float,
    kernel_ms: float,
    num_filters: int = 10,
    order: int = 4,
    kernel_direction: int = 0,
) -> None:
    """
    Paper: g temporal filters = 10 raised cos, span 600ms (causal or acausal).
    Here: temporal-kernel smooth with ~10 basis functions (B-splines).

    kernel_direction:
      1  -> causal kernel over [0, kernel_ms]
      0  -> acausal kernel over [-kernel_ms/2, +kernel_ms/2] (your handler’s "Acausal")
     -1  -> anti-causal (rare)
    """
    kernel_h_length = int(np.round(kernel_ms / dt_ms))
    if kernel_h_length <= 1:
        raise ValueError('kernel_h_length too small; check dt_ms/kernel_ms.')

    # Choose internal knots so that (order + internal_knots) ~= num_filters
    num_int_knots = max(1, int(num_filters - order))

    sm_handler.add_smooth(
        name,
        [event_impulse],
        is_temporal_kernel=True,
        ord=order,
        knots_num=num_int_knots,
        trial_idx=trial_ids,
        kernel_length=kernel_h_length,
        kernel_direction=int(kernel_direction),
    )


def add_spike_history_filter(
    sm_handler,
    name: str,
    y_binned: np.ndarray,
    trial_ids: np.ndarray,
    dt_ms: float,
    hist_ms: float = 350.0,
    num_filters: int = 10,
    order: int = 4,
) -> None:
    """
    Paper: h spike-history = 10 causal raised-cos on log time, span 350ms.
    Here: causal temporal-kernel smooth over y_binned (binned spikes).
    """
    add_event_temporal_filter(
        sm_handler=sm_handler,
        name=name,
        event_impulse=y_binned,
        trial_ids=trial_ids,
        dt_ms=dt_ms,
        kernel_ms=hist_ms,
        num_filters=num_filters,
        order=order,
        kernel_direction=1,  # causal
    )


def add_coupling_filters(
    sm_handler,
    base_name: str,
    other_units_binned: np.ndarray,
    trial_ids: np.ndarray,
    dt_ms: float,
    coupling_ms: float = 1375.0,
    num_filters: int = 10,
    order: int = 4,
) -> None:
    """
    Paper: p coupling = 10 causal raised-cos on log time, span 1.375s.
    Here: for each other unit j, add a causal temporal-kernel smooth over that unit’s binned spikes.
    """
    n_time, n_other = other_units_binned.shape
    for j in range(n_other):
        add_event_temporal_filter(
            sm_handler=sm_handler,
            name=f'{base_name}_unit{j:03d}',
            event_impulse=other_units_binned[:, j],
            trial_ids=trial_ids,
            dt_ms=dt_ms,
            kernel_ms=coupling_ms,
            num_filters=num_filters,
            order=order,
            kernel_direction=1,  # causal
        )

# -------------------------
# Main: build smooth handlers per neuron
# -------------------------


def build_smooth_handler_for_unit(
    *,
    unit_idx: int,
    covariates_concat: dict[str, np.ndarray],
    covariate_names: list[str],
    trial_id_vec: np.ndarray,
    Y_binned: np.ndarray,                 # shape (T, n_units)
    all_events: dict[str, np.ndarray],    # impulses, shape (T,)
    dt: float,
    tuning_covariates: list[str] | None = None,
    use_cyclic: set[str] | None = None,
    order: int = 4,
    add_coupling: bool = False,
):
    """
    Build a smooths_handler for a *single* unit.

    Returns:
        sm_handler : gdh.smooths_handler
    """
    dt_ms = float(dt * 1e3)

    if tuning_covariates is None:
        tuning_covariates = list(covariate_names)

    if use_cyclic is None:
        use_cyclic = set()

    event_kernel_ms = 600.0
    n_time, n_units = Y_binned.shape

    if unit_idx < 0 or unit_idx >= n_units:
        raise IndexError(
            f'unit_idx {unit_idx} out of bounds for n_units={n_units}')

    sm_handler = gdh.smooths_handler()

    # ---- f(·): tuning functions
    for cov_name in tuning_covariates:
        if cov_name not in covariates_concat:
            raise KeyError(
                f'Covariate "{cov_name}" not found in covariates_concat.')

        x = covariates_concat[cov_name]  # shape (T,)

        add_tuning_smooth_10boxcar_approx(
            sm_handler=sm_handler,
            name=f'f_{cov_name}',
            x=x,
            trial_ids=trial_id_vec,
            num_basis=10,
            order=order,
            is_cyclic=(cov_name in use_cyclic),
        )

    # ---- g(·): event temporal filters
    if 't_targ' in all_events:
        add_event_temporal_filter(
            sm_handler=sm_handler,
            name='g_t_targ',
            event_impulse=all_events['t_targ'],
            trial_ids=trial_id_vec,
            dt_ms=dt_ms,
            kernel_ms=event_kernel_ms,
            num_filters=10,
            order=order,
            kernel_direction=1,  # causal
        )

    for evt in ['t_move', 't_rew', 't_stop']:
        if evt in all_events:
            add_event_temporal_filter(
                sm_handler=sm_handler,
                name=f'g_{evt}',
                event_impulse=all_events[evt],
                trial_ids=trial_id_vec,
                dt_ms=dt_ms,
                kernel_ms=event_kernel_ms,
                num_filters=10,
                order=order,
                kernel_direction=0,  # acausal
            )

    # ---- h(·): spike history (this unit)
    add_spike_history_filter(
        sm_handler=sm_handler,
        name='h_spike_history',
        y_binned=Y_binned[:, unit_idx],
        trial_ids=trial_id_vec,
        dt_ms=dt_ms,
        hist_ms=350.0,
        num_filters=10,
        order=order,
    )

    # ---- p(·): coupling filters (other units)
    if add_coupling and n_units > 1:
        other_mask = np.ones(n_units, dtype=bool)
        other_mask[unit_idx] = False
        Y_other = Y_binned[:, other_mask]

        add_coupling_filters(
            sm_handler=sm_handler,
            base_name='p_coupling',
            other_units_binned=Y_other,
            trial_ids=trial_id_vec,
            dt_ms=dt_ms,
            coupling_ms=1375.0,
            num_filters=10,
            order=order,
        )

    return sm_handler


def build_smooth_handlers_for_population(
    *,
    covariates_concat: dict[str, np.ndarray],
    covariate_names: list[str],
    trial_id_vec: np.ndarray,
    Y_binned: np.ndarray,                 # shape (T, n_units)
    all_events: dict[str, np.ndarray],
    dt: float,
    tuning_covariates: list[str] | None = None,
    use_cyclic: set[str] | None = None,
    order: int = 4,
    add_coupling: bool = False,
) -> list:
    """
    Returns:
        sm_handlers : list of gdh.smooths_handler, one per unit
    """
    _, n_units = Y_binned.shape

    sm_handlers = [
        build_smooth_handler_for_unit(
            unit_idx=k,
            covariates_concat=covariates_concat,
            covariate_names=covariate_names,
            trial_id_vec=trial_id_vec,
            Y_binned=Y_binned,
            all_events=all_events,
            dt=dt,
            tuning_covariates=tuning_covariates,
            use_cyclic=use_cyclic,
            order=order,
            add_coupling=add_coupling,
        )
        for k in range(n_units)
    ]

    return sm_handlers
