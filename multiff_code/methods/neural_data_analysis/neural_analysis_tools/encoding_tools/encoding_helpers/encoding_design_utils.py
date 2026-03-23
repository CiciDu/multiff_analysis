
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import re

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_glm_design
from neural_data_analysis.design_kits.design_around_event import event_binning

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoding_design_utils,
)

import os
import sys
from pathlib import Path
from data_wrangling import combine_info_utils


# Radians to degrees; monkey_information stores angles/angular rates in rad.
_RAD_TO_DEG = 180.0 / np.pi

# Map planning/monkey_information column names -> one_ff names so agg_cols find them.
# Only renames when source exists and target does not (avoids overwriting one_ff data).
# Do not map speed/ang_speed -> v/w here; base design already uses speed, ang_speed as kinematics.
PLANNING_TO_ONE_FF_RENAME: Dict[str, str] = {
    # integrated / heading (one_ff: d, phi)
    'cum_distance': 'd',
    'monkey_angle': 'phi',
    # target (polar-like; one_ff: r_targ, theta_targ)
    'target_distance': 'r_targ',
    'target_angle': 'theta_targ',
    # eye channels are handled explicitly in rename_monkey_information_columns:
    # average left/right when both valid, else use available channel.
}


# One_ff_gam tuning covariates (one_ff_gam_design finalize_one_ff_gam_design)
# Only columns present in monkey_information are aggregated and added.
ONE_FF_STYLE_EXTRA_COLS = [
    'v', 'w', 'd', 'phi',
    'r_targ', 'theta_targ',
    'eye_ver', 'eye_hor',
    # though they are not in one_ff_gam, we add them to the design
    'accel', 'ang_accel',
    'time', # though this is not in one_ff_gam, it might be useful for multiff
]


# Tuning vars without wrapping (passed as linear_vars to build_continuous_tuning_block)
DEFAULT_TUNING_VARS_NO_WRAP = [
    'v', 'w', 'd', 'r_targ', 'eye_ver', 'eye_hor', 'theta_targ',
    # though these are not in one_ff_gam, we add them to the design
    'accel', 'ang_accel',
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

_REQUIRED_ALIAS_MAP = {
    'v': ['speed'],
    'w': ['ang_speed'],
    'd': ['cum_distance'],
    'eye_ver': ['LDy', 'RDy'],
    'eye_hor': ['LDz', 'RDz'],
    # add acceleration and angular acceleration
    'accel': ['accel'],
    'ang_accel': ['ang_accel'],
}
_OPTIONAL_ALIAS_MAP = {
    'r_targ': ['target_distance'],
    'theta_targ': ['target_angle'],
}


# =============================================================================
# Shared event-window binning core (stop encoding/decoding)
# =============================================================================


def bin_event_windows_core(
    *,
    new_seg_info: pd.DataFrame,
    monkey_information: pd.DataFrame,
    bin_dt: float,
    global_bins_2d: Optional[np.ndarray] = None,
    agg_cols: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[
    np.ndarray,          # bins_2d
    pd.DataFrame,        # meta
    pd.DataFrame,        # binned_feats
    np.ndarray,          # exposure
    np.ndarray,          # used_bins
    np.ndarray,          # mask_used
    np.ndarray,          # pos
    pd.DataFrame,        # meta_df_used
]:
    """
    Shared core to go from event windows -> binned monkey features.

    This is used by both stop decoding designs (in `decode_stops_design.py`)
    and stop encoding designs (in `encode_stops_utils.py`).
    Spike binning is done separately via bin_spikes_for_event_windows().
    """
    agg_cols = ONE_FF_STYLE_EXTRA_COLS if agg_cols is None else agg_cols

    bins_2d, meta = event_binning.event_windows_to_bins2d(
        new_seg_info,
        bin_dt=bin_dt,
        only_ok=False,
        global_bins_2d=global_bins_2d,
    )

    binned_feats, exposure, used_bins, mask_used, pos = (
        bin_monkey_information_feats_from_event_bins(
            monkey_information,
            bins_2d,
            kinematic_cols=[],
            agg_cols=agg_cols,
        )
    )

    if verbose:
        print('meta.shape', meta.shape)

    meta_df_used = (
        meta.set_index('bin')
        .sort_index()
        .loc[pos]
        .reset_index()
    )

    return (
        bins_2d,
        meta,
        binned_feats,
        exposure,
        used_bins,
        mask_used,
        pos,
        meta_df_used,
    )


def bin_spikes_for_event_windows(
    spikes_df: pd.DataFrame,
    bins_2d: np.ndarray,
    pos: np.ndarray,
    *,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Bin spikes by cluster for event windows, subsetting to used bins (pos).
    """
    spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
        spikes_df,
        bins_2d,
        time_col=time_col,
        cluster_col=cluster_col,
    )
    binned_spikes = (
        pd.DataFrame(spike_counts[pos, :], columns=cluster_ids)
        .reset_index(drop=True)
    )
    return binned_spikes, cluster_ids

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
            'Missing required stop-encoding covariates after building design: '
            f'{missing_msg}. '
            'Provide these columns in monkey_information '
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
    print('linear_vars before resolving:', linear_vars)

    # Find vars present in the design but missing from binrange_dict and
    # attempt to estimate their ranges from the binned features using
    # multiff_encoding_params.estimate_stop_binrange_from_binned_feats.
    wanted = [c for c in list(linear) + list(angular) if c in binned_feats.columns]
    missing = [c for c in wanted if c not in binrange_dict]
    if missing:
        try:
            estimated = multiff_encoding_params.estimate_stop_binrange_from_binned_feats(
                binned_feats,
                vars_to_include=missing,
            )
        except Exception:
            estimated = {}
        for k, v in estimated.items():
            if k not in binrange_dict:
                binrange_dict[k] = np.asarray(v, dtype=float)

    linear = [c for c in linear if c in binned_feats.columns and c in binrange_dict]
    angular = [c for c in angular if c in binned_feats.columns and c in binrange_dict]
    return linear, angular


def _resolve_tuning_feature_mode(
    use_boxcar: bool,
    tuning_feature_mode: Optional[str],
) -> str:
    """
    Resolve tuning feature mode with backward compatibility.

    - New API: tuning_feature_mode in {'raw_only', 'boxcar_only', 'raw_plus_boxcar'}
    - Backward API: use_boxcar bool
    """
    if tuning_feature_mode is not None:
        if tuning_feature_mode not in VALID_TUNING_FEATURE_MODES:
            raise ValueError(
                f'Invalid tuning_feature_mode={tuning_feature_mode!r}. '
                f'Expected one of {sorted(VALID_TUNING_FEATURE_MODES)}'
            )
        return tuning_feature_mode
    return 'boxcar_only' if use_boxcar else 'raw_only'


def build_tuning_design_for_continuous_vars(
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



def rename_monkey_information_columns(
    monkey_information: pd.DataFrame,
    rename_planning_to_one_ff: bool = True,
) -> pd.DataFrame:
    """
    Wrapper to make monkey_information use one_ff-style column names for encoding design.

    Renames planning/pipeline columns (e.g. target_distance, speed) to one_ff names
    (r_targ, v) 

    Parameters
    ----------
    monkey_information : pd.DataFrame
        Per-sample dataframe (e.g. pn.monkey_information).
    rename_planning_to_one_ff : bool
        If True, apply PLANNING_TO_ONE_FF_RENAME for any source column that exists
        and whose target name is not already a column.

    Returns
    -------
    pd.DataFrame
        Copy of monkey_information with columns renamed so one_ff-style names
        (v, w, d, phi, r_targ, theta_targ, eye_ver, eye_hor) are present when
        the planning data has the corresponding source columns.
    """
    out = monkey_information.copy()

    def _derive_eye_channel(
        out_df: pd.DataFrame,
        target_col: str,
        left_col: str,
        right_col: str,
    ) -> None:
        """
        Derive eye target column using binocular average when possible.

        When valid_view_point_l and valid_view_point_r exist (from
        eye_positions.find_valid_view_points), only use left/right eye values
        where that eye's gaze is valid. Otherwise fall back to row-wise mean
        with skipna.

        Priority:
        1) If target already exists: keep it.
        2) If valid_view_point_l/r exist: use left only where valid_view_point_l,
           right only where valid_view_point_r; then mean of valid values.
        3) Else if left/right both exist: row-wise mean(skipna=True).
        4) Else use whichever of left/right exists.
        5) Else leave missing (no gaze fallback).
        """
        if target_col in out_df.columns:
            return

        has_left = left_col in out_df.columns
        has_right = right_col in out_df.columns
        has_valid = (
            'valid_view_point_l' in out_df.columns
            and 'valid_view_point_r' in out_df.columns
        )

        if has_valid and (has_left or has_right):
            left_vals = np.full(len(out_df), np.nan, dtype=float)
            right_vals = np.full(len(out_df), np.nan, dtype=float)
            if has_left:
                left_vals = np.where(
                    out_df['valid_view_point_l'].to_numpy(dtype=bool),
                    out_df[left_col].to_numpy(float),
                    np.nan,
                )
            if has_right:
                right_vals = np.where(
                    out_df['valid_view_point_r'].to_numpy(dtype=bool),
                    out_df[right_col].to_numpy(float),
                    np.nan,
                )
            both = np.stack([left_vals, right_vals], axis=1)
            with warnings.catch_warnings():
                # Mean of empty slice
                warnings.simplefilter('ignore', RuntimeWarning)
                out_df[target_col] = np.nanmean(both, axis=1)
            # Fill NaN from rows where both eyes invalid (avoids NaN in design -> rate=nan)
            out_df[target_col] = out_df[target_col].fillna(0.0)
            return
        else:
            print(
                'Warning: columns valid_view_point_l or valid_view_point_r does not exist')

        if has_left and has_right:
            out_df[target_col] = out_df[[left_col, right_col]].mean(
                axis=1, skipna=True
            )
            return
        if has_left:
            out_df[target_col] = out_df[left_col].to_numpy()
            return
        if has_right:
            out_df[target_col] = out_df[right_col].to_numpy()
            return

    if rename_planning_to_one_ff:
        _derive_eye_channel(
            out_df=out,
            target_col='eye_ver',
            left_col='LDz',
            right_col='RDz',
        )
        _derive_eye_channel(
            out_df=out,
            target_col='eye_hor',
            left_col='LDy',
            right_col='RDy',
        )
        # Do NOT rename speed/ang_speed (base design needs them), but do provide
        # one_ff-style aliases when absent.
        if 'v' not in out.columns and 'speed' in out.columns:
            out['v'] = out['speed'].to_numpy()
        if 'w' not in out.columns and 'ang_speed' in out.columns:
            out['w'] = out['ang_speed'].to_numpy()

    renames: Dict[str, str] = {}
    if rename_planning_to_one_ff:
        for src, tgt in PLANNING_TO_ONE_FF_RENAME.items():
            if src in out.columns and tgt not in out.columns:
                renames[src] = tgt
    if renames:
        out = out.rename(columns=renames)

    # Convert angular vars from radians to degrees (monkey_information stores rad)
    for col in ('w', 'phi', 'theta_targ'):
        if col in out.columns:
            out[col] = out[col].to_numpy(float, copy=False) * _RAD_TO_DEG

    return out


def _ensure_one_ff_style_covariates(
    binned_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure one_ff-style covariates exist in binned_feats by resolving aliases.

    Applies:
        - REQUIRED alias map (raises if missing)
        - OPTIONAL alias map (silently skips if missing)

    Returns
    -------
    binned_feats : DataFrame
        Possibly-augmented DataFrame with required/optional covariates filled.
    """
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

    return binned_feats




def _convolve_on_grid(
    e_grid: np.ndarray,
    bin_dt: float,
    *,
    n_basis: int,
    t_min: float,
    t_max: float,
    event_name='event',
) -> Tuple[np.ndarray, Dict]:

    lags, B = glm_bases.raised_cosine_basis(
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        dt=bin_dt,
    )

    n_grid = e_grid.size

    lag_min = int(np.rint(float(lags[0]) / bin_dt))
    shift = -lag_min

    X_grid = np.zeros((n_grid, n_basis), dtype=float)

    for k in range(n_basis):
        y_full = np.convolve(e_grid, B[:, k], mode='full')
        y = y_full[shift:shift + n_grid]
        X_grid[:, k] = y

    meta = {
        'dt': bin_dt,
        'basis_info': {event_name: {'lags': lags, 'basis': B}},
    }

    return X_grid, meta

def build_temporal_design_from_binned_events(
    bin_t_center: np.ndarray,
    event_indicator: np.ndarray,
    bin_dt: float,
    *,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    index: Optional[pd.Index] = None,
    event_name='event',
) -> Tuple[pd.DataFrame, Dict]:
    """
    Temporal design from binned events.
    Works even if bin_t_center is discontinuous.
    """

    name_prefix = 'rcos_' + event_name

    bin_t_center = np.asarray(bin_t_center, dtype=float).ravel()
    e_local = np.asarray(event_indicator, dtype=float).ravel()

    if bin_t_center.size != e_local.size:
        raise ValueError('event_indicator must match bin_t_center length')

    if bin_t_center.size == 0:
        raise ValueError('Empty bin_t_center')

    # ---------------------------------------------------------
    # Build global convolution grid
    # ---------------------------------------------------------
    t_grid_start = float(bin_t_center.min() - t_max)
    t_grid_end = float(bin_t_center.max() - t_min)

    n_grid = int(np.floor((t_grid_end - t_grid_start) / bin_dt)) + 1
    if n_grid <= 0:
        raise ValueError('Invalid convolution grid')

    # ---------------------------------------------------------
    # Embed indicator into global grid
    # ---------------------------------------------------------
    e_grid = np.zeros(n_grid, dtype=float)

    bin_idx = np.rint(
        (bin_t_center - t_grid_start) / bin_dt
    ).astype(int)

    e_grid[bin_idx] = e_local

    # ---------------------------------------------------------
    # Convolve
    # ---------------------------------------------------------
    X_grid, base_meta = _convolve_on_grid(
        e_grid,
        bin_dt,
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
    )

    # ---------------------------------------------------------
    # Sample back only modeled bins
    # ---------------------------------------------------------
    X = X_grid[bin_idx, :]

    rcos_names = [f'{name_prefix}_{k}' for k in range(X.shape[1])]
    temporal_df = pd.DataFrame(X, columns=rcos_names)

    if index is not None:
        temporal_df.index = index

    temporal_meta = {
        **base_meta,
        'groups': {event_name: rcos_names},
        'mode': 'summed_events_binned',
        'n_events': int(np.sum(e_local)),
    }

    return temporal_df, temporal_meta

def build_temporal_design_from_event_times(
    bin_t_center: np.ndarray,
    event_times: np.ndarray,
    bin_dt: float,
    *,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    index: Optional[pd.Index] = None,
    event_name='event',
) -> Tuple[pd.DataFrame, Dict]:
    """
    Temporal design from continuous event timestamps.
    Robust to discontinuous bin_t_center.
    """

    name_prefix = 'rcos_' + event_name
    
    bin_t_center = np.asarray(bin_t_center, dtype=float).ravel()
    event_times = np.asarray(event_times, dtype=float).ravel()
    event_times = event_times[np.isfinite(event_times)]

    if bin_t_center.size == 0:
        raise ValueError('Empty bin_t_center')

    # ---------------------------------------------------------
    # Build global convolution grid
    # ---------------------------------------------------------
    t_grid_start = float(bin_t_center.min() - t_max)
    t_grid_end = float(bin_t_center.max() - t_min)

    n_grid = int(np.floor((t_grid_end - t_grid_start) / bin_dt)) + 1
    if n_grid <= 0:
        raise ValueError('Invalid convolution grid')

    # ---------------------------------------------------------
    # Build impulse on global grid
    # ---------------------------------------------------------
    e_grid = np.zeros(n_grid, dtype=float)

    if event_times.size > 0:
        event_idx = np.rint(
            (event_times - t_grid_start) / bin_dt
        ).astype(int)

        event_idx = event_idx[
            (event_idx >= 0) & (event_idx < n_grid)
        ]

        if event_idx.size > 0:
            np.add.at(e_grid, event_idx, 1.0)

    # ---------------------------------------------------------
    # Convolve
    # ---------------------------------------------------------
    X_grid, base_meta = _convolve_on_grid(
        e_grid,
        bin_dt,
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
    )

    # ---------------------------------------------------------
    # Sample back modeled bins
    # ---------------------------------------------------------
    bin_idx = np.rint(
        (bin_t_center - t_grid_start) / bin_dt
    ).astype(int)

    X = X_grid[bin_idx, :]

    rcos_names = [f'{name_prefix}_{k}' for k in range(X.shape[1])]
    temporal_df = pd.DataFrame(X, columns=rcos_names)

    if index is not None:
        temporal_df.index = index

    temporal_meta = {
        **base_meta,
        'groups': {event_name: rcos_names},
        'mode': 'summed_events_times',
        'n_events': int(event_times.size),
    }

    return temporal_df, temporal_meta


def build_vis_encoding_design(
    pn,
    datasets,
    new_seg_info,
    events_with_stats,
    ff_on_df: pd.DataFrame,
    group_on_df: pd.DataFrame,
    bin_width: float,
    *,
    add_stop_cluster_features: bool = False,
    add_retry_features: bool = False,
    **design_kwargs,
):
    """
    Build vis (ff visibility) encoding design: stop design + ff/group temporal bases.

    Calls build_stop_encoding_design then adds ff_on, ff_off, group_ff_on, group_ff_off
    temporal designs.
    """
    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
        encode_stops_utils,
        encode_vis_utils,
    )

    (
        pn,
        binned_spikes,
        binned_feats,
        meta_df_used,
        temporal_meta,
        tuning_meta,
    ) = encode_stops_utils.build_stop_encoding_design(
        pn,
        datasets,
        new_seg_info,
        events_with_stats,
        bin_width,
        add_stop_cluster_features=add_stop_cluster_features,
        add_retry_features=add_retry_features,
        **design_kwargs,
    )

    binned_feats, temporal_meta = encode_vis_utils.add_ff_visibility_temporal_designs(
        binned_feats,
        temporal_meta,
        meta_df_used,
        ff_on_df,
        group_on_df,
        bin_width,
        n_basis=design_kwargs.get('n_basis', 20),
        t_min=design_kwargs.get('t_min', -0.3),
        t_max=design_kwargs.get('t_max', 0.3),
    )

    return (
        pn,
        binned_spikes,
        binned_feats,
        meta_df_used,
        temporal_meta,
        tuning_meta,
    )


def build_hist_meta_from_colnames(
    colnames: Dict[str, List[str]],
    *,
    target_col: Optional[str],
    dt: float,
    t_max: float,
    n_basis: int = 20,
) -> Dict[str, Any]:
    """
    Build histogram meta structure from cluster colnames.

    Returns:
        dict with 'groups' and 'basis_info' keys.
        Returns {} if colnames is empty.
    """
    if not colnames:
        return {}

    # Build raised cosine basis
    t_min = dt
    lags_hist, B_hist = glm_bases.raised_log_cosine_basis(
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        dt=dt,
        log_spaced=True,
        hard_start_zero=True,
    )

    basis_info_entry = {
        'lags': lags_hist,
        'basis': B_hist,
    }

    hist_groups: Dict[str, List[str]] = {}
    hist_basis_info: Dict[str, Dict] = {}

    neuron_order = list(colnames.keys())

    if target_col is not None:
        if target_col not in colnames:
            raise KeyError(f'target_col {target_col!r} not in colnames')
        neuron_order = [target_col] + [
            n for n in neuron_order if n != target_col
        ]

    for i, neuron in enumerate(neuron_order):
        cols = colnames[neuron]

        if i == 0:
            group_name = 'spike_hist'
        else:
            neuron_match = re.match(r'^cluster_(\d+)$', neuron)
            cpl_suffix = neuron_match.group(1) if neuron_match else str(i - 1)
            group_name = f'cpl_{cpl_suffix}'

        hist_groups[group_name] = cols
        hist_basis_info[group_name] = basis_info_entry

    return {
        'groups': hist_groups,
        'basis_info': hist_basis_info,
    }

def build_structured_meta_groups(
    *,
    hist_meta: Optional[Dict[str, Any]],
    temporal_meta: Optional[Dict[str, Any]],
    tuning_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Restructure meta groups into one_ff_gam-style categories.

    Expects hist_meta to already be constructed.
    """

    return {
        'tuning': tuning_meta if tuning_meta else {},
        'temporal': temporal_meta if temporal_meta else {},
        'hist': hist_meta if hist_meta else {},
        'lambda_config': {},
    }


def add_tuning_features_to_design(
    design_matrix: pd.DataFrame,
    *,
    use_boxcar: bool,
    tuning_feature_mode: Optional[str],
    binrange_dict: Optional[Dict[str, Union[np.ndarray, Tuple[float, float]]]],
    tuning_n_bins: int,
    linear_vars: Optional[List[str]],
    angular_vars: Optional[List[str]],
    raw_feature_cols_to_drop: Optional[List[str]] = None,
):
    """
    Generic helper to augment a design matrix with tuning (boxcar) features.

    This function:
        1) Resolves tuning mode
        2) Estimates bin ranges if needed
        3) Builds tuning design
        4) Optionally drops raw feature columns
        5) Returns updated design and tuning metadata

    Parameters
    ----------
    design_matrix : pd.DataFrame
        Base feature matrix.
    use_boxcar : bool
    tuning_feature_mode : Optional[str]
    binrange_dict : Optional[dict]
    tuning_n_bins : int
    linear_vars : Optional[list]
    angular_vars : Optional[list]
    raw_feature_cols_to_drop : Optional[list]
        Columns to drop if mode == 'boxcar_only'.

    Returns
    -------
    design_matrix : pd.DataFrame
    tuning_meta : Optional[dict]
    mode : str
    """

    mode = _resolve_tuning_feature_mode(
        use_boxcar,
        tuning_feature_mode,
    )

    tuning_meta = None

    if mode == 'raw_only':
        return design_matrix, tuning_meta, mode

    # --------------------------------------------------------------
    # Estimate bin ranges from data
    # --------------------------------------------------------------
    estimated = multiff_encoding_params.estimate_stop_binrange_from_binned_feats(
        design_matrix
    )

    binrange_dict = dict(binrange_dict) if binrange_dict else {}

    for k, v in estimated.items():
        if k not in binrange_dict:
            binrange_dict[k] = v

    if not binrange_dict:
        raise ValueError(
            f'tuning_feature_mode={mode!r} requires binrange_dict '
            'or estimable tuning vars.'
        )

    # --------------------------------------------------------------
    # Resolve tuning variables
    # --------------------------------------------------------------
    print('linear_vars:', linear_vars)
    
    linear_resolved, angular_resolved = \
        _resolve_tuning_vars(
            design_matrix,
            binrange_dict,
            linear_vars,
            angular_vars,
        )

    print('linear_resolved:', linear_resolved)

    if not (linear_resolved or angular_resolved):
        return design_matrix, tuning_meta, mode

    # --------------------------------------------------------------
    # Build tuning features
    # --------------------------------------------------------------
    X_tuning, tuning_meta = \
        build_tuning_design_for_continuous_vars(
            design_matrix,
            linear_resolved,
            angular_resolved,
            n_bins=tuning_n_bins,
            binrange_dict=binrange_dict,
            wrap_angular=True,
        )

    # --------------------------------------------------------------
    # Optionally drop raw features
    # --------------------------------------------------------------
    if mode == 'boxcar_only' and raw_feature_cols_to_drop:
        to_drop = [
            c for c in raw_feature_cols_to_drop
            if c in design_matrix.columns
        ]
        design_matrix = design_matrix.drop(columns=to_drop)

    # --------------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------------
    X_tuning.index = design_matrix.index
    design_matrix = pd.concat([design_matrix, X_tuning], axis=1)

    return design_matrix, tuning_meta, mode


def bin_monkey_information_feats_from_event_bins(
    monkey_information: pd.DataFrame,
    bins_2d: np.ndarray,
    *,
    kinematic_cols: Optional[List[str]] = None,
    agg_cols: Optional[List[str]] = None,
    use_one_ff_rename: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin (aggregate) selected columns from `monkey_information` into `bins_2d`.

    Returns:
      binned_feats: DataFrame aligned to `used_bins`
      exposure:     exposure per `used_bins`
      used_bins:    bin indices returned by bin_timeseries_weighted
      mask_used:    boolean mask over `used_bins` where exposure > 0
      pos:          used_bins[mask_used] (bin indices into bins_2d)
    """

    if use_one_ff_rename:
        monkey_information = encoding_design_utils.rename_monkey_information_columns(
           monkey_information,
            rename_planning_to_one_ff=use_one_ff_rename,
        )

    if kinematic_cols is None:
        kinematic_cols = []

    # 1) Assign continuous samples to bins
    sample_idx, bin_idx_arr, dt_arr, _ = event_binning.build_bin_assignments(
        monkey_information['time'].to_numpy(), bins_2d
    )
    monkey_sub = monkey_information.iloc[sample_idx].copy()

    # 2) Compute exposure and used bins
    _, exposure, used_bins = event_binning.bin_timeseries_weighted(
        monkey_sub['time'].to_numpy(), dt_arr, bin_idx_arr, how='mean'
    )

    def _agg_feat(col: str) -> np.ndarray:
        vals = np.asarray(monkey_sub[col].to_numpy(), dtype=float)

        vals_safe = np.where(np.isfinite(vals), vals, 0.0)
        finite_mask = np.isfinite(vals).astype(float)

        out, exp_chk, used_bins_chk = event_binning.bin_timeseries_weighted(
            vals_safe, dt_arr, bin_idx_arr, how='mean'
        )
        if not np.allclose(exp_chk, exposure):
            raise ValueError(f'Exposure mismatch while aggregating {col!r}')

        contrib, _, _ = event_binning.bin_timeseries_weighted(
            finite_mask, dt_arr, bin_idx_arr, how='mean'
        )
        out = np.where(contrib > 0, out, np.nan)

        if not np.array_equal(used_bins_chk, used_bins):
            raise ValueError(f'used_bins mismatch while aggregating {col!r}')

        return out

    # 3) Aggregate columns
    if kinematic_cols:
        binned_feats = (
            pd.DataFrame({c: _agg_feat(c) for c in kinematic_cols})
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    else:
        binned_feats = pd.DataFrame(index=np.arange(len(used_bins)))

    if agg_cols:
        existing = [c for c in agg_cols if c in monkey_sub.columns]
        if existing:
            extra_df = (
                pd.DataFrame({c: _agg_feat(c) for c in existing})
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            binned_feats = pd.concat([binned_feats, extra_df], axis=1)

    # 4) Keep bins with exposure > 0
    mask_used = exposure > 0
    pos = used_bins[mask_used]

    binned_feats = binned_feats.iloc[mask_used].reset_index(drop=True)

    return binned_feats, exposure, used_bins, mask_used, pos


def merge_meta_vals(a: Any, b: Any) -> Any:

    # 1. Recursive dict merge
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for kk, vv in b.items():
            if kk in out:
                out[kk] = merge_meta_vals(out[kk], vv)
            else:
                out[kk] = vv
        return out

    # 2. NumPy arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.concatenate([a, b])

    # 3. Lists / tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) + list(b)

    # 4. Numbers
    if isinstance(a, (int, float, np.integer, np.floating)) and \
       isinstance(b, (int, float, np.integer, np.floating)):
        return a + b

    # 5. Strings
    if isinstance(a, str) and isinstance(b, str):
        return a + b

    # 6. Default: overwrite with b
    return b



def _collect_group_columns(groups: List[GroupSpec]) -> List[str]:
    """Flatten all columns across groups."""
    all_cols = []
    for g in groups:
        all_cols.extend(g.cols)
    return all_cols


def _validate_design_columns(design_df, groups: List[GroupSpec]) -> None:
    """
    Ensure every column in design_df appears in some GroupSpec.
    Raises ValueError if any column is unassigned.
    """
    design_cols = set(design_df.columns)
    grouped_cols = set(_collect_group_columns(groups))

    missing = sorted(design_cols - grouped_cols)
    if missing:
        raise ValueError(
            f'The following design_df columns are not assigned to any GroupSpec:\n{missing}'
        )

def build_gam_groups_from_meta(
    structured_meta_groups,
    *,
    lam_f: float = 100.0,
    lam_g: float = 100.0,
    lam_h: float = 10.0,
    lam_p: float = 10.0,
) -> Tuple[List[GroupSpec], Dict[str, float]]:
    """
    Build GroupSpec list and lambda_config for stop-encoding Poisson GAM from
    `structured_meta_groups` instead of a full `design_df`.

    Additionally validates that every column in design_df is assigned
    to exactly one group.
    """
    tuning_meta = structured_meta_groups.get('tuning', {}) or {}
    temporal_meta = structured_meta_groups.get('temporal', {}) or {}
    hist_meta = structured_meta_groups.get('hist', {}) or {}

    groups: List[GroupSpec] = []

    # -------------------------
    # Temporal groups (lam_g)
    # -------------------------
    temporal_groups = temporal_meta.get('groups', {}) if temporal_meta else {}
    for name, cols in temporal_groups.items():
        if not cols:
            continue
        groups.append(GroupSpec(name, list(cols), 'event', lam_g))

    # -------------------------
    # Tuning groups (lam_f)
    # -------------------------
    tuning_groups = tuning_meta.get('groups', {}) if tuning_meta else {}
    for var, cols in tuning_groups.items():
        if not cols:
            continue
        groups.append(GroupSpec(var, list(cols), '1D', lam_f))

    # -------------------------
    # History groups
    # spike_hist -> lam_h
    # others     -> lam_p
    # -------------------------
    hist_groups = hist_meta.get('groups', {}) if hist_meta else {}
    for i, (gname, cols) in enumerate(hist_groups.items()):
        if not cols:
            continue
        lam = lam_h if (i == 0 or gname == 'spike_hist') else lam_p
        groups.append(GroupSpec(gname, list(cols), 'event', lam))


    return groups

def bootstrap_repo_path():
    """Add multiff methods to path and chdir to project root."""
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if p.name == "Multifirefly-Project":
            os.chdir(p)
            sys.path.insert(0, str(p / "multiff_analysis/multiff_code/methods"))
            return
    raise RuntimeError("Could not find Multifirefly-Project root")


def get_session_paths(raw_data_folder_path, raw_data_dir_name, monkey_names):
    """
    Return session paths to process.

    If raw_data_folder_path is not None, return [raw_data_folder_path].
    Otherwise, collect all sessions from monkey_names via combine_info_utils.
    """
    if raw_data_folder_path is not None:
        return [raw_data_folder_path]

    session_paths = []
    for monkey_name in monkey_names:
        sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
            raw_data_dir_name, monkey_name
        )
        for _, row in sessions_df.iterrows():
            session_paths.append(
                os.path.join(raw_data_dir_name, row["monkey_name"], row["data_name"])
            )
    return session_paths
