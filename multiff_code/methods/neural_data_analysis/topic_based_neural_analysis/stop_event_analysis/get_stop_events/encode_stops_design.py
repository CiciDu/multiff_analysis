"""
Stop design for encoding with one_ff_gam-style basis functions.

Same interface as `decode_stops_design.build_stop_design`, but implemented
here for stop-GAM encoding:

- We build the shared binning/aggregation/spikes/cluster/ff/retries features.
- We do NOT build the legacy event-design columns (prepost/captured/time_since*
  or the old rcos_* basis), since encoding replaces the temporal basis anyway.
- We then add one_ff-style raised-cosine temporal basis over `rel_center` and,
  optionally, one_ff-style boxcar tuning terms.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_glm_design
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis import stop_parameters
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
    encode_stops_utils,
)


# One_ff_gam tuning covariates (one_ff_gam_design finalize_one_ff_gam_design)
# Only columns present in monkey_information are aggregated and added.
ONE_FF_STYLE_EXTRA_COLS = [
    'v', 'w', 'd', 'phi',
    'r_targ', 'theta_targ',
    'eye_ver', 'eye_hor',
    # though they are not in one_ff_gam, we add them to the design
    'accel', 'ang_accel_deg',
]

# Tuning vars without wrapping (passed as linear_vars to build_continuous_tuning_block)
DEFAULT_TUNING_VARS_NO_WRAP = [
    'v', 'w', 'd', 'r_targ', 'eye_ver', 'eye_hor', 'theta_targ',
    # though these are not in one_ff_gam, we add them to the design
    'accel', 'ang_accel_deg',
]

# Tuning vars with wrapping (e.g. phi; passed as angular_vars with wrap_angular=True)
DEFAULT_TUNING_VARS_WRAP = [
    'phi',
]

VALID_TUNING_FEATURE_MODES = {
    'raw_only',
    'boxcar_only',
    'raw_plus_boxcar',
}


# Columns added by build_event_design_from_meta that we replace with one_ff-style basis
_EVENT_DESIGN_COLS_TO_REPLACE = {
    'prepost',
    'prepost*speed',
    'captured',
    'time_since_prev_event',
    'time_to_next_event',
}


_REQUIRED_ALIAS_MAP = {
    'v': ['speed'],
    'w': ['ang_speed'],
    'd': ['cum_distance'],
    'eye_ver': ['LDy', 'RDy'],
    'eye_hor': ['LDz', 'RDz'],
    # add acceleration and angular acceleration
    'accel': ['accel'],
    'ang_accel_deg': ['ang_accel'],
}
_OPTIONAL_ALIAS_MAP = {
    'r_targ': ['target_distance'],
    'theta_targ': ['target_angle'],
}


def _fill_covariate_from_aliases(
    binned_feats: pd.DataFrame,
    alias_map: Dict[str, List[str]],
    *,
    required: bool,
) -> pd.DataFrame:
    """
    Ensure target covariates exist by copying from alias columns when available.

    Parameters
    ----------
    binned_feats : DataFrame
        Aggregated per-bin features.
    alias_map : dict
        target -> [alias candidates] in priority order.
    required : bool
        If True, missing targets raise ValueError; otherwise they are skipped.
    """
    missing = []
    for target, aliases in alias_map.items():
        if target in binned_feats.columns:
            continue
        src = next((c for c in aliases if c in binned_feats.columns), None)
        if src is None:
            if required:
                missing.append((target, aliases))
            continue
        binned_feats[target] = binned_feats[src].to_numpy()

    if required and missing:
        missing_msg = ', '.join([f'{t} (aliases: {a})' for t, a in missing])
        raise ValueError(
            'Missing required stop-encoding covariates after build_stop_design: '
            f'{missing_msg}. '
            'Provide these columns in monkey_information (or via custom_rename) '
            'so they can be aggregated into binned_feats.'
        )
    return binned_feats


def _resolve_tuning_vars(
    binned_feats: pd.DataFrame,
    binrange_dict: Dict[str, Union[np.ndarray, Tuple[float, float]]],
    linear_vars: Optional[List[str]],
    angular_vars: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """Pick tuning variables that exist in both design and binrange.
    linear_vars = vars without wrapping; angular_vars = vars with wrapping (e.g. phi).
    """
    linear = linear_vars if linear_vars is not None else DEFAULT_TUNING_VARS_NO_WRAP
    angular = angular_vars if angular_vars is not None else DEFAULT_TUNING_VARS_WRAP
    linear = [c for c in linear if c in binned_feats.columns and c in binrange_dict]
    angular = [
        c for c in angular if c in binned_feats.columns and c in binrange_dict]
    return linear, angular


def _resolve_tuning_feature_mode(
    use_tuning_design: bool,
    tuning_feature_mode: Optional[str],
) -> str:
    """
    Resolve tuning feature mode with backward compatibility.

    - New API: tuning_feature_mode in {'raw_only', 'boxcar_only', 'raw_plus_boxcar'}
    - Backward API: use_tuning_design bool
    """
    if tuning_feature_mode is not None:
        if tuning_feature_mode not in VALID_TUNING_FEATURE_MODES:
            raise ValueError(
                f'Invalid tuning_feature_mode={tuning_feature_mode!r}. '
                f'Expected one of {sorted(VALID_TUNING_FEATURE_MODES)}'
            )
        return tuning_feature_mode
    return 'boxcar_only' if use_tuning_design else 'raw_only'


def _eval_raised_cosine_at_t(
    t: np.ndarray,
    lags: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Evaluate raised cosine basis at arbitrary times t.

    Parameters
    ----------
    t : (n,) array
        Time points (e.g. rel_center per bin).
    lags : (n_lags,) array
        Lag grid from glm_bases.raised_cosine_basis.
    B : (n_lags, n_basis) array
        Basis matrix from glm_bases.raised_cosine_basis.

    Returns
    -------
    X : (n, n_basis) array
        Basis values at each t; clipped to [0, 1] at edges.
    """
    t = np.asarray(t, dtype=float).ravel()
    n_basis = B.shape[1]
    out = np.zeros((len(t), n_basis), dtype=float)
    for k in range(n_basis):
        out[:, k] = np.interp(t, lags, B[:, k])
    return out


def build_temporal_design_base_stop(
    rel_center: np.ndarray,
    bin_dt: float,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    index: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build temporal design matrix for stop_gam (time relative to stop).
    Mirror of one_ff_gam_design.build_temporal_design_base for stop-aligned data:
    single raised-cosine basis over rel_center instead of t_move/t_targ/t_rew/t_stop.

    Parameters
    ----------
    rel_center : (n_bins,) array
        Time of each bin center relative to stop (seconds).
    bin_dt : float
        Bin width (s); used for basis spacing.
    n_basis : int
        Number of raised cosine basis functions (default 20).
    t_min, t_max : float
        Time window (s) for the basis, e.g. [-0.3, 0.3].
    index : pandas.Index, optional
        Index for the returned DataFrame (e.g. binned_feats.index).

    Returns
    -------
    temporal_df : DataFrame
        Columns rcos_0 .. rcos_{n_basis-1}, same length as rel_center.
    temporal_meta : dict
        'dt': bin_dt, 'basis_info': {'t_rel': {'lags': ..., 'basis': B}}, 'groups': {'t_rel': [rcos_0, ...]}.
    """
    lags, B = glm_bases.raised_cosine_basis(
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        dt=bin_dt,
    )
    X_t_rel = _eval_raised_cosine_at_t(np.asarray(
        rel_center, dtype=float).ravel(), lags, B)
    rcos_names = [f'rcos_{k}' for k in range(n_basis)]
    temporal_df = pd.DataFrame(X_t_rel, columns=rcos_names)
    if index is not None:
        temporal_df.index = index
    basis_info = {'t_rel': {'lags': lags, 'basis': B}}
    temporal_meta = {
        'dt': bin_dt,
        'basis_info': basis_info,
        'groups': {'t_rel': rcos_names},
    }
    return temporal_df, temporal_meta


def build_temporal_design_summed_stops(
    bin_t_center: np.ndarray,
    stop_times: np.ndarray,
    bin_dt: float,
    *,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    index: Optional[pd.Index] = None,
    global_bins_2d: Optional[np.ndarray] = None,
    used_bin_indices: Optional[np.ndarray] = None,
    name_prefix: str = 'rcos_stop',
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a *summed-event* temporal design: each bin receives contributions from
    *all* stops within the kernel window. Column names are ``{name_prefix}_0``, ...
    (default ``rcos_stop_0`` for stop-aligned encoding).

    For each basis function k and each bin center time t:

        X_k(t) = sum_i b_k(t - stop_time_i)

    where b_k(.) is the raised-cosine basis over lags in [t_min, t_max].

    When global_bins_2d is provided, the convolution grid is built to span the
    global session bins (plus kernel support); design values are computed for
    every global bin, then only rows for bins present in binned_feats are
    returned (via used_bin_indices). When global_bins_2d is None, the grid is
    derived from bin_t_center and one row per bin_t_center is returned.
    """
    bin_t_center = np.asarray(bin_t_center, dtype=float).ravel()
    stop_times = np.asarray(stop_times, dtype=float).ravel()
    stop_times = stop_times[np.isfinite(stop_times)]

    lags, B = glm_bases.raised_cosine_basis(
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        dt=bin_dt,
    )

    use_global = global_bins_2d is not None and used_bin_indices is not None
    if use_global:
        global_bins_2d = np.asarray(global_bins_2d, dtype=float)
        if global_bins_2d.ndim != 2 or global_bins_2d.shape[1] != 2:
            raise ValueError('global_bins_2d must have shape (B, 2)')
        used_bin_indices = np.asarray(used_bin_indices, dtype=int).ravel()
        g_center = 0.5 * (global_bins_2d[:, 0] + global_bins_2d[:, 1])
        t_grid_start = float(global_bins_2d[:, 0].min() - t_max)
        t_grid_end = float(global_bins_2d[:, 1].max() + (-t_min))
        n_grid = max(
            1, int(np.floor((t_grid_end - t_grid_start) / bin_dt)) + 1)
    else:
        if bin_t_center.size == 0:
            X = np.zeros((0, n_basis), dtype=float)
            rcos_names = [f'{name_prefix}_{k}' for k in range(n_basis)]
            temporal_df = pd.DataFrame(X, columns=rcos_names)
            if index is not None:
                temporal_df.index = index
            temporal_meta = {
                'dt': bin_dt,
                'basis_info': {'t_rel': {'lags': lags, 'basis': B}},
                'groups': {'t_rel': rcos_names},
                'mode': 'summed_stops',
                'n_stops': int(stop_times.size),
            }
            return temporal_df, temporal_meta
        t_grid_start = float(bin_t_center.min() - t_max)
        t_grid_end = float(bin_t_center.max() - t_min)
        n_grid = int(np.floor((t_grid_end - t_grid_start) / bin_dt)) + 1
        if n_grid <= 0:
            raise ValueError('invalid grid for summed-stop temporal design')

    # Map stops to nearest grid index (counts if multiple stops hit same bin)
    e = np.zeros(n_grid, dtype=float)
    if stop_times.size > 0:
        stop_idx = np.rint((stop_times - t_grid_start) / bin_dt).astype(int)
        stop_idx = stop_idx[(stop_idx >= 0) & (stop_idx < n_grid)]
        if stop_idx.size > 0:
            np.add.at(e, stop_idx, 1.0)

    # Convolve impulse train with each basis column.
    lag_min = int(np.rint(float(lags[0]) / bin_dt))
    shift = -lag_min
    X_grid = np.zeros((n_grid, n_basis), dtype=float)
    for k in range(n_basis):
        y_full = np.convolve(e, B[:, k], mode='full')
        y = y_full[shift:shift + n_grid]
        X_grid[:, k] = y

    if use_global:
        # Sample design at each global bin center, then keep only used bins.
        grid_idx = np.rint((g_center - t_grid_start) / bin_dt).astype(int)
        grid_idx = np.clip(grid_idx, 0, n_grid - 1)
        X_global_bins = X_grid[grid_idx, :]
        X = X_global_bins[used_bin_indices, :]
    else:
        bin_idx = np.rint((bin_t_center - t_grid_start) / bin_dt).astype(int)
        if np.any(bin_idx < 0) or np.any(bin_idx >= n_grid):
            raise ValueError(
                'bin_t_center outside convolution grid (check t_min/t_max/bin_dt)')
        X = X_grid[bin_idx, :]

    rcos_names = [f'{name_prefix}_{k}' for k in range(n_basis)]
    temporal_df = pd.DataFrame(X, columns=rcos_names)
    if index is not None:
        temporal_df.index = index

    temporal_meta = {
        'dt': bin_dt,
        'basis_info': {'t_rel': {'lags': lags, 'basis': B}},
        'groups': {'t_rel': rcos_names},
        'mode': 'summed_stops',
        'n_stops': int(stop_times.size),
    }
    return temporal_df, temporal_meta


def build_tuning_design_stop(
    data_df: pd.DataFrame,
    linear_vars: List[str],
    angular_vars: List[str],
    n_bins: int = 10,
    binrange_dict: Optional[Dict[str,
                                 Union[np.ndarray, Tuple[float, float]]]] = None,
    center: bool = False,
    wrap_angular: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build continuous tuning design for stop_gam (same as one_ff build_tuning_design).
    Uses one_ff_glm_design.build_continuous_tuning_block and adds bin_info to metadata.

    Parameters
    ----------
    data_df : DataFrame
        Per-bin design (e.g. binned_feats) with covariates v, w, d, phi, r_targ, theta_targ, eye_ver, eye_hor.
    linear_vars : list
        Variables without wrapping (e.g. v, w, d, r_targ, eye_ver, eye_hor).
    angular_vars : list
        Variables with wrapping to [-180, 180) (e.g. phi); wrap_angular applies to these.
    n_bins : int
        Number of bins per variable (default 10).
    binrange_dict : dict, optional
        Maps variable name -> [min, max] or (min, max). Required for each var in linear_vars + angular_vars.
    center : bool
        Passed to build_continuous_tuning_block (default False).
    wrap_angular : bool
        Passed to build_continuous_tuning_block (default False).

    Returns
    -------
    X_tuning : DataFrame
        Tuning design (var:bin0 .. var:bin{n_bins-1} for each variable).
    tuning_meta : dict
        'bin_edges', 'groups', 'bin_info' (edges, centers, n_bins per var).
    """
    if binrange_dict is None:
        binrange_dict = {}
    # Normalize to arrays for one_ff_glm_design
    br = {}
    for k, v in binrange_dict.items():
        if isinstance(v, (tuple, list)):
            br[k] = np.array(v, dtype=float)
        else:
            br[k] = np.asarray(v, dtype=float)

    X_tuning, raw_meta = one_ff_glm_design.build_continuous_tuning_block(
        data=data_df,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        n_bins=n_bins,
        center=center,
        binrange_dict=br,
        wrap_angular=wrap_angular,
    )
    bin_info = {}
    all_vars = list(linear_vars) + list(angular_vars)
    for var in all_vars:
        if var in raw_meta.get('bin_edges', {}):
            edges = raw_meta['bin_edges'][var]
            centers = (edges[:-1] + edges[1:]) / 2
            bin_info[var] = {
                'edges': edges,
                'centers': centers,
                'n_bins': n_bins,
            }
    tuning_meta = {
        'bin_edges': raw_meta.get('bin_edges', {}),
        'groups': raw_meta.get('groups', {}),
        'bin_info': bin_info,
        'linear_vars': linear_vars,
        'angular_vars': angular_vars,
    }
    return X_tuning, tuning_meta


def _build_feature_groups_for_encoding(
    binned_feats: pd.DataFrame,
    kinematic_cols: List[str],
    basis_cols: List[str],
    tuning_meta: Optional[Dict] = None,
    include_raw_one_ff_cols: bool = True,
) -> Dict[str, List[str]]:
    """
    Build semantic feature groups (same style as decode_stops_design._build_feature_groups).
    Uses 'basis' for the one_ff-style rcos_* columns. If tuning_meta is provided (from
    build_tuning_design_stop), adds one group per variable (var:bin0..); otherwise
    adds one_ff-style extra cols as single-column groups when present.
    """
    groups: Dict[str, List[str]] = {}

    def _add(name: str, cols: List[str]) -> None:
        present = [c for c in cols if c in binned_feats.columns]
        if present:
            groups[name] = present

    for c in kinematic_cols:
        _add(c, [c])
    if tuning_meta is not None and 'groups' in tuning_meta:
        for var, cols in tuning_meta['groups'].items():
            _add(var, cols)
    if include_raw_one_ff_cols:
        for c in ONE_FF_STYLE_EXTRA_COLS:
            _add(c, [c])
    _add('basis', basis_cols)

    _add('cluster_flags', ['event_is_first_in_cluster'])
    _add('cluster_gaps', ['gap_since_prev_event_in_cluster_z',
         'gap_till_next_event_in_cluster_z'])
    _add('cluster_duration', ['cluster_duration_s_z'])
    _add('cluster_progress', ['cluster_progress_c', 'cluster_progress_c2'])
    _add('cluster_rel_time', ['bin_t_from_cluster_start_s_z'])
    _add('cluster_n_events', ['log_n_events_in_cluster_z'])
    _add('is_clustered', ['is_clustered'])
    _add('event_in_cluster_t', ['event_t_from_cluster_start_s'])

    _add('ff_visible', ['log1p_num_ff_visible', 'k_ff_visible'])
    _add('ff_in_memory', ['log1p_num_ff_in_memory', 'k_ff_in_memory'])
    _add(
        'retries',
        [
            'rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle',
            'rsw_last', 'rcap_last', 'one_stop_miss',
            'whether_in_retry_series', 'miss',
        ],
    )
    _add('time_rel_to_event_start', ['time_rel_to_event_start'])

    return groups


def rebuild_encoding_groups_after_blocks(
    binned_feats: pd.DataFrame,
    deferred: Dict,
) -> Dict[str, List[str]]:
    """
    Rebuild feature groups after temporal and tuning blocks have been concatenated
    to binned_feats (e.g. after scaling). deferred is the dict returned by
    build_stop_design_for_encoding(..., add_temporal_and_tuning_after_scale=True).
    """
    return _build_feature_groups_for_encoding(
        binned_feats,
        kinematic_cols=deferred['kinematic_cols'],
        basis_cols=deferred['rcos_names'],
        tuning_meta=deferred['tuning_meta'],
        include_raw_one_ff_cols=deferred['include_raw_one_ff_cols'],
    )


def build_stop_design_for_encoding(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    spikes_df: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    bin_dt: float = 0.04,
    add_ff_visible_info: bool = True,
    add_retries_info: bool = True,
    datasets: dict = None,
    global_bins_2d: np.ndarray = None,
    # one_ff_gam-style temporal basis (build_temporal_design_base_stop)
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    use_planning_rename: bool = True,
    custom_rename: Optional[Dict[str, str]] = None,
    # optional one_ff-style boxcar tuning (build_tuning_design_stop)
    use_tuning_design: bool = False,
    tuning_feature_mode: Optional[str] = None,
    binrange_dict: Optional[Dict[str,
                                 Union[np.ndarray, Tuple[float, float]]]] = None,
    linear_vars: Optional[List[str]] = None,
    angular_vars: Optional[List[str]] = None,
    tuning_n_bins: int = 10,
    add_temporal_and_tuning_after_scale: bool = True,
):
    """
    Build stop-aligned design with one_ff_gam-style raised cosine temporal basis.

    Same interface and return types as decode_stops_design.build_stop_design.
    When add_temporal_and_tuning_after_scale is True, returns 6 values: the 5 usual
    plus a deferred dict (temporal_df, tuning_df, etc.) so the caller can scale
    first then concat temporal and tuning and rebuild groups. When False, returns 5 values.
    The event-related design (basis over time relative to stop) is built using
    glm_bases.raised_cosine_basis (same as one_ff_gam temporal kernels) evaluated
    at each bin's rel_center, instead of stop_design.build_event_design_from_meta.

    Parameters
    ----------
    new_seg_info, events_with_stats, monkey_information, spikes_df, ff_dataframe
        Passed through to build_stop_design.
    bin_dt : float
        Bin width (s).
    add_ff_visible_info, add_retries_info, datasets, global_bins_2d
        Passed through to build_stop_design_base_for_encoding.
    n_basis : int
        Number of raised cosine basis functions (default 20, match one_ff_gam).
    t_min, t_max : float
        Time window (s) for temporal basis relative to event, e.g. [-0.3, 0.3].
    use_planning_rename : bool
        If True, apply monkey_information_for_encoding() so planning columns
        (e.g. target_distance, speed) are renamed to one_ff names (r_targ, v).
    custom_rename : dict, optional
        Passed to monkey_information_for_encoding for extra or override renames.
    use_tuning_design : bool
        Backward-compatible toggle for tuning boxcar features.
        If tuning_feature_mode is None, True -> 'boxcar_only', False -> 'raw_only'.
    tuning_feature_mode : {'raw_only', 'boxcar_only', 'raw_plus_boxcar'}, optional
        Explicit tuning representation mode:
        - raw_only: keep raw v/w/d/phi/r_targ/theta_targ/eye_* only
        - boxcar_only: replace raw one_ff-style cols with var:bin* boxcars
        - raw_plus_boxcar: keep raw cols and append var:bin* boxcars
    binrange_dict : dict, optional
        Variable -> [min, max] for tuning; required when use_tuning_design=True.
    linear_vars, angular_vars : list, optional
        Tuning vars: linear_vars = no wrapping; angular_vars = with wrapping (e.g. phi).
        Defaults: DEFAULT_TUNING_VARS_NO_WRAP, DEFAULT_TUNING_VARS_WRAP.
    tuning_n_bins : int
        Bins per variable for tuning design (default 10).

    Returns
    -------
    binned_spikes : pd.DataFrame
        (n_bins × n_clusters) spike counts.
    binned_feats : pd.DataFrame
        (n_bins × n_features) design with kinematics, t_rel_* basis, cluster, ff, retries.
    offset_log : np.ndarray
        log(exposure) per bin.
    meta_used : pd.DataFrame
        Per-bin metadata.
    groups : Dict[str, List[str]]
        Feature groupings for downstream GAM (same style as build_stop_design).
    """
    # -------------------------------------------------------------------------
    # 0) Optionally rename planning columns to one_ff names (target_distance -> r_targ, etc.)
    # -------------------------------------------------------------------------
    if use_planning_rename or custom_rename:
        monkey_for_encoding = encode_stops_utils.monkey_information_for_encoding(
            monkey_information,
            rename_planning_to_one_ff=use_planning_rename,
            custom_rename=custom_rename,
        )
    else:
        monkey_for_encoding = monkey_information

    kinematic_cols = []
    extra_agg_cols = ONE_FF_STYLE_EXTRA_COLS
    # -------------------------------------------------------------------------
    # 1) Base design for encoding (binning/spikes/cluster/ff/retries).
    #    NOTE: we intentionally skip the legacy event-design block here.
    # -------------------------------------------------------------------------
    binned_spikes, binned_feats, offset_log, meta_used, _groups_unused = (
        encode_stops_utils.build_stop_design_base_for_encoding(
            new_seg_info=new_seg_info,
            events_with_stats=events_with_stats,
            monkey_information=monkey_for_encoding,
            spikes_df=spikes_df,
            ff_dataframe=ff_dataframe,
            bin_dt=bin_dt,
            add_ff_visible_info=add_ff_visible_info,
            add_retries_info=add_retries_info,
            datasets=datasets,
            global_bins_2d=global_bins_2d,
            extra_agg_cols=extra_agg_cols,
            kinematic_cols=kinematic_cols,
        )
    )

    # Ensure one_ff-like covariates exist (required and optional aliases).
    binned_feats = _fill_covariate_from_aliases(
        binned_feats,
        _REQUIRED_ALIAS_MAP,
        required=True,
    )
    binned_feats = _fill_covariate_from_aliases(
        binned_feats,
        _OPTIONAL_ALIAS_MAP,
        required=False,
    )

    # -------------------------------------------------------------------------
    # 3) Temporal design: summed stop-kernel basis over absolute bin time
    #    (each bin gets contributions from multiple nearby stops).
    # -------------------------------------------------------------------------
    if 't_center' not in meta_used.columns:
        raise ValueError(
            'meta_used missing required column "t_center" for summed-stop temporal design')
    if 'event_id_start_time' not in events_with_stats.columns:
        raise ValueError(
            'events_with_stats missing required column "event_id_start_time" for summed-stop temporal design')

    # When global_bins_2d is used, build temporal design on the global grid and keep only rows for bins in binned_feats.
    use_global = (
        global_bins_2d is not None
        and 'global_bin' in meta_used.columns
    )
    kwargs = dict(
        bin_t_center=meta_used['t_center'].to_numpy(dtype=float),
        stop_times=events_with_stats['event_id_start_time'].to_numpy(
            dtype=float),
        bin_dt=bin_dt,
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        index=binned_feats.index,
        name_prefix='rcos_stop',
    )
    if use_global:
        kwargs['global_bins_2d'] = global_bins_2d
        kwargs['used_bin_indices'] = meta_used['global_bin'].to_numpy(
            dtype=int)

    temporal_df, temporal_meta = build_temporal_design_summed_stops(**kwargs)
    rcos_names = list(temporal_df.columns)
    defer_blocks = add_temporal_and_tuning_after_scale
    if not defer_blocks:
        binned_feats = pd.concat([binned_feats, temporal_df], axis=1)

    init_binned_feats = binned_feats.copy()
    # -------------------------------------------------------------------------
    # 4) Optional: tuning representation mode (raw_only / boxcar_only / raw_plus_boxcar)
    # -------------------------------------------------------------------------
    mode = _resolve_tuning_feature_mode(use_tuning_design, tuning_feature_mode)
    tuning_meta: Optional[Dict] = None
    X_tuning: Optional[pd.DataFrame] = None
    if mode != 'raw_only':
        estimated = stop_parameters.estimate_stop_binrange_from_binned_feats(
            binned_feats
        )
        # Merge: user-specified ranges take precedence; fill in missing vars from data
        binrange_dict = dict(binrange_dict) if binrange_dict else {}
        for k, v in estimated.items():
            if k not in binrange_dict:
                binrange_dict[k] = v
        if not binrange_dict:
            raise ValueError(
                f'tuning_feature_mode={mode!r} requires binrange_dict or estimable '
                'tuning vars (v,w,d,phi,r_targ,theta_targ,eye_ver,eye_hor) in binned_feats.'
            )
        _linear, _angular = _resolve_tuning_vars(
            binned_feats,
            binrange_dict,
            linear_vars,
            angular_vars,
        )
        if _linear or _angular:
            X_tuning, tuning_meta = build_tuning_design_stop(
                binned_feats,
                _linear,
                _angular,
                n_bins=tuning_n_bins,
                binrange_dict=binrange_dict,
                # phi and other wrap vars use [-180, 180) wrapping
                wrap_angular=True,
            )
            if mode == 'boxcar_only':
                to_drop = [
                    c for c in ONE_FF_STYLE_EXTRA_COLS if c in binned_feats.columns]
                binned_feats = binned_feats.drop(columns=to_drop)
            if not defer_blocks:
                X_tuning.index = binned_feats.index
                binned_feats = pd.concat([binned_feats, X_tuning], axis=1)

    # -------------------------------------------------------------------------
    # 5) Rebuild groups dict: 'basis' = rcos_stop_*, optional tuning_meta groups
    # -------------------------------------------------------------------------
    if defer_blocks:
        groups = _build_feature_groups_for_encoding(
            binned_feats,
            kinematic_cols=kinematic_cols,
            basis_cols=[],
            tuning_meta=None,
            include_raw_one_ff_cols=(mode != 'boxcar_only'),
        )
        deferred = {
            'temporal_df': temporal_df,
            'temporal_meta': temporal_meta,
            'rcos_names': rcos_names,
            'tuning_df': X_tuning,
            'tuning_meta': tuning_meta,
            'mode': mode,
            'include_raw_one_ff_cols': (mode != 'boxcar_only'),
            'kinematic_cols': kinematic_cols,
            'binrange_dict': binrange_dict,
        }
        return binned_spikes, binned_feats, offset_log, meta_used, groups, deferred, init_binned_feats
    groups = _build_feature_groups_for_encoding(
        binned_feats,
        kinematic_cols=kinematic_cols,
        basis_cols=rcos_names,
        tuning_meta=tuning_meta,
        include_raw_one_ff_cols=(mode != 'boxcar_only'),
    )
    return binned_spikes, binned_feats, offset_log, meta_used, groups, None, init_binned_feats


def assemble_stop_encoding_design(
    raw_data_folder_path,
    bin_width,
    global_bins_2d=None,
    # Encoding-specific parameters
    n_basis=20,
    t_min=-0.3,
    t_max=0.3,
    use_tuning_design=False,
    tuning_feature_mode='boxcar_only',  # 'raw_only', 'boxcar_only', 'raw_plus_boxcar'
    binrange_dict=None,
    tuning_n_bins=10,
    linear_vars=None,
    angular_vars=None,
):
    """
    Assemble stop design for encoding (GAM / temporal + tuning basis).
    """

    pn, datasets, new_seg_info, events_with_stats = \
        decode_stops_design._prepare_stop_design_inputs(
            raw_data_folder_path, bin_width)

    build_result = build_stop_design_for_encoding(
        new_seg_info=new_seg_info,
        events_with_stats=events_with_stats,
        monkey_information=pn.monkey_information,
        spikes_df=pn.spikes_df,
        ff_dataframe=pn.ff_dataframe,
        bin_dt=bin_width,
        datasets=datasets,
        add_ff_visible_info=True,
        global_bins_2d=global_bins_2d,
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        use_tuning_design=use_tuning_design,
        tuning_feature_mode=tuning_feature_mode,
        binrange_dict=binrange_dict,
        tuning_n_bins=tuning_n_bins,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        add_temporal_and_tuning_after_scale=True,
    )

    (
        stop_binned_spikes,
        stop_binned_feats,
        offset_log,
        stop_meta_used,
        stop_meta_groups,
        deferred,
        init_stop_binned_feats,
    ) = build_result

    # --------------------------------------------------------
    # Add temporal and tuning AFTER scaling
    # --------------------------------------------------------

    stop_binned_feats = pd.concat(
        [stop_binned_feats, deferred['temporal_df']],
        axis=1,
    )

    if deferred['tuning_df'] is not None:
        if len(deferred['tuning_df']) != len(stop_binned_feats):
            raise ValueError('Tuning df length mismatch')
        deferred['tuning_df'].index = stop_binned_feats.index
        stop_binned_feats = pd.concat(
            [stop_binned_feats, deferred['tuning_df']],
            axis=1,
        )

    # Drop constant columns except 'const'
    const_cols_to_drop = [
        c for c in stop_binned_feats.columns
        if c != 'const' and stop_binned_feats[c].nunique() <= 1
    ]
    stop_binned_feats = stop_binned_feats.drop(columns=const_cols_to_drop)

    stop_meta_groups = (
        rebuild_encoding_groups_after_blocks(
            stop_binned_feats,
            deferred,
        )
    )

    binrange_dict = deferred.get('binrange_dict')
    temporal_meta = deferred.get('temporal_meta')
    tuning_meta = deferred.get('tuning_meta')

    return (
        pn,
        stop_binned_spikes,
        stop_binned_feats,
        offset_log,
        stop_meta_used,
        stop_meta_groups,
        init_stop_binned_feats,
        binrange_dict,
        temporal_meta,
        tuning_meta,
    )
