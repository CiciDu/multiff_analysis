
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re
import os

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
    stop_design,
    cluster_design,
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)


# Radians to degrees; monkey_information stores angles/angular rates in rad.
_RAD_TO_DEG = 180.0 / np.pi

# Map planning/monkey_information column names -> one_ff names so extra_agg_cols find them.
# Only renames when source exists and target does not (avoids overwriting one_ff data).
# Do not map speed/ang_speed -> v/w here; base design already uses speed, ang_speed as kinematics.
PLANNING_TO_ONE_FF_RENAME: Dict[str, str] = {
    # integrated / heading (one_ff: d, phi)
    'cum_distance': 'd',
    'monkey_angle': 'phi',
    # target (polar-like; one_ff: r_targ, theta_targ)
    'target_distance': 'r_targ',
    'target_angle': 'theta_targ',
    # eye channels are handled explicitly in monkey_information_for_encoding:
    # average left/right when both valid, else use available channel.
}


def monkey_information_for_encoding(
    monkey_information: pd.DataFrame,
    rename_planning_to_one_ff: bool = True,
    custom_rename: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Wrapper to make monkey_information use one_ff-style column names for encoding design.

    Renames planning/pipeline columns (e.g. target_distance, speed) to one_ff names
    (r_targ, v) so that extra_agg_cols in build_stop_design picks them up when
    building the encoding design.

    Parameters
    ----------
    monkey_information : pd.DataFrame
        Per-sample dataframe (e.g. pn.monkey_information).
    rename_planning_to_one_ff : bool
        If True, apply PLANNING_TO_ONE_FF_RENAME for any source column that exists
        and whose target name is not already a column.
    custom_rename : dict, optional
        Additional or override renames: {source_col: target_col}. Applied after
        the default PLANNING_TO_ONE_FF_RENAME.

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
        if 'ang_accel_deg' not in out.columns and 'ang_accel' in out.columns:
            out['ang_accel_deg'] = out['ang_accel'].to_numpy(
                float, copy=False) * _RAD_TO_DEG

    renames: Dict[str, str] = {}
    if rename_planning_to_one_ff:
        for src, tgt in PLANNING_TO_ONE_FF_RENAME.items():
            if src in out.columns and tgt not in out.columns:
                renames[src] = tgt
    if custom_rename:
        for src, tgt in custom_rename.items():
            if src in out.columns and tgt not in out.columns:
                renames[src] = tgt
    if renames:
        out = out.rename(columns=renames)

    # Convert angular vars from radians to degrees (monkey_information stores rad)
    for col in ('w', 'phi', 'theta_targ'):
        if col in out.columns:
            out[col] = out[col].to_numpy(float, copy=False) * _RAD_TO_DEG

    return out


def build_stop_design_base_for_encoding(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    spikes_df: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    *,
    bin_dt: float = 0.04,
    add_ff_visible_info: bool = True,
    add_retries_info: bool = True,
    datasets: dict = None,
    global_bins_2d: np.ndarray = None,
    extra_agg_cols: Optional[List[str]] = None,
    kinematic_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """
    Minimal stop design builder for encoding.

    Matches the binning/spikes/cluster/ff/retries parts of the decoding design,
    but intentionally skips the legacy event-design block (prepost/captured/time_since*
    and old rcos_* basis), since encoding replaces temporal kernels anyway.
    """
    # 1) Build bins from event windows
    bins_2d, meta = event_binning.event_windows_to_bins2d(
        new_seg_info, bin_dt=bin_dt, only_ok=False, global_bins_2d=global_bins_2d
    )

    # 2) Assign continuous samples to bins
    sample_idx, bin_idx_arr, dt_arr, _ = event_binning.build_bin_assignments(
        monkey_information['time'].to_numpy(), bins_2d
    )
    monkey_sub = monkey_information.iloc[sample_idx].copy()

    # 3) Compute exposure and valid bins
    _, exposure, used_bins = event_binning.bin_timeseries_weighted(
        monkey_sub['time'].to_numpy(), dt_arr, bin_idx_arr, how='mean'
    )

    def _agg_feat(col: str) -> np.ndarray:
        vals = np.asarray(monkey_sub[col].to_numpy(), dtype=float)
        # Use 0 for NaN so bin_timeseries_weighted does not drop rows (would change used_bins)
        vals_safe = np.where(np.isfinite(vals), vals, 0.0)
        finite_mask = np.isfinite(vals).astype(float)
        out, exp_chk, used_bins_chk = event_binning.bin_timeseries_weighted(
            vals_safe, dt_arr, bin_idx_arr, how='mean'
        )
        if not np.allclose(exp_chk, exposure):
            raise ValueError(f'Exposure mismatch while aggregating {col!r}')
        # Bins where all values were NaN: keep output as NaN
        contrib, _, _ = event_binning.bin_timeseries_weighted(
            finite_mask, dt_arr, bin_idx_arr, how='mean'
        )
        out = np.where(contrib > 0, out, np.nan)
        if not np.array_equal(used_bins_chk, used_bins):
            raise ValueError(f'used_bins mismatch while aggregating {col!r}')
        return out

    # 4) Aggregate kinematics + optional extra covariates

    binned_feats = (
        pd.DataFrame({c: _agg_feat(c) for c in kinematic_cols})
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if extra_agg_cols:
        existing = [c for c in extra_agg_cols if c in monkey_sub.columns]
        if existing:
            extra_df = (
                pd.DataFrame({c: _agg_feat(c) for c in existing})
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            binned_feats = pd.concat([binned_feats, extra_df], axis=1)

    mask_used = exposure > 0
    pos = used_bins[mask_used]
    binned_feats = binned_feats.iloc[mask_used].reset_index(drop=True)

    meta_used = (
        meta.set_index('bin')
        .sort_index()
        .loc[pos]
        .reset_index()
    )

    # 5) Bin spikes per cluster
    spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
        spikes_df, bins_2d, time_col='time', cluster_col='cluster'
    )
    binned_spikes = pd.DataFrame(
        spike_counts[pos, :], columns=cluster_ids).reset_index(drop=True)

    # 6) Cluster-level features (no event-design features here)
    cluster_df = cluster_design.build_cluster_features_workflow(
        meta_used[['event_id', 'rel_center']],
        events_with_stats,
        rel_time_col='rel_center',
        winsor_p=0.5,
        use_midbin_progress=True,
        zscore_progress=False,
        zscore_rel_time=True,
    )
    cluster_feats = [
        'is_clustered',
        'event_is_first_in_cluster',
        'gap_since_prev_event_in_cluster_z',
        'gap_till_next_event_in_cluster_z',
        'cluster_duration_s_z',
        'cluster_progress_c',
        'bin_t_from_cluster_start_s_z',
        'log_n_events_in_cluster_z',
    ]
    cluster_feats = cluster_feats + \
        ['cluster_progress_c2', 'event_t_from_cluster_start_s']
    _safe_add_columns(binned_feats, cluster_df, cluster_feats)

    # 7) Offset term and timing
    offset_log = np.log(np.clip(exposure[mask_used], 1e-12, None))
    binned_feats['time_rel_to_event_start'] = meta_used['rel_center'].to_numpy()

    # 8) Firefly visibility / memory features
    if add_ff_visible_info:
        binned_feats = _add_ff_visible_and_in_memory_info(
            binned_feats, bins_2d, ff_dataframe, used_bins
        )

    # 9) Retry-related features
    if add_retries_info:
        binned_feats, meta_used = _add_retries_info_to_binned_feats(
            binned_feats, new_seg_info, datasets, meta_used
        )

    # groups are rebuilt later in build_encode_stops_design
    return binned_spikes, binned_feats, offset_log, meta_used, {}


def _safe_add_columns(target_df: pd.DataFrame, source_df: pd.DataFrame, cols: List[str]) -> None:
    cols = [c for c in cols if c in source_df.columns and c not in target_df.columns]
    if cols:
        target_df.loc[:, cols] = source_df[cols].to_numpy()


def _add_ff_visible_and_in_memory_info(
    binned_feats: pd.DataFrame,
    bins_2d: np.ndarray,
    ff_dataframe: pd.DataFrame,
    used_bins: np.ndarray,
    *,
    max_in_memory_time_since_seen: float = 2.0,
) -> pd.DataFrame:
    ff_df_sub = ff_dataframe[
        ff_dataframe['time_since_last_vis'] < max_in_memory_time_since_seen
    ].copy()
    ff_df_sub['in_memory'] = 1
    for state in ['visible', 'in_memory']:
        k_ff = vis_design.count_visible_from_time_df_fast(
            ff_df_sub, bins_2d, vis_col=state
        )
        binned_feats[f'num_ff_{state}'] = k_ff[used_bins]
        binned_feats[f'log1p_num_ff_{state}'] = np.log1p(
            binned_feats[f'num_ff_{state}'])
    return binned_feats


def _add_retries_info_to_binned_feats(
    binned_feats: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    datasets: dict,
    meta_used: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if datasets is None:
        raise ValueError('datasets is required to add retries info')

    new_seg_info = new_seg_info.copy()
    RETRY_CATEGORIES = [
        'rsw_first', 'rcap_first',
        'rsw_middle', 'rcap_middle',
        'rsw_last', 'rcap_last',
        'one_stop_miss',
    ]
    for col in RETRY_CATEGORIES:
        new_seg_info[col] = 0
        stop_ids = datasets[col]['stop_id'].unique()
        new_seg_info.loc[new_seg_info['stop_id'].isin(stop_ids), col] = 1

    event_tbl = stop_design.build_per_event_table(
        new_seg_info, extras=RETRY_CATEGORIES)
    meta_used = stop_design.join_event_tbl_avoid_collisions(
        meta_used, event_tbl)

    meta_used = meta_used.sort_values('bin').reset_index(drop=True)
    binned_feats = binned_feats.reset_index(drop=True)
    if len(meta_used) != len(binned_feats):
        raise ValueError('meta_used and binned_feats misaligned')

    for col in RETRY_CATEGORIES:
        binned_feats[col] = meta_used[col].to_numpy()

    retry_cols = ['rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle']
    miss_cols = retry_cols + ['rsw_last', 'one_stop_miss']
    binned_feats['whether_in_retry_series'] = binned_feats[retry_cols].any(
        axis=1).astype('int8')
    binned_feats['miss'] = binned_feats[miss_cols].any(axis=1).astype('int8')

    # drop to avoid perfect collinearity (matches decoding flow)
    if 'is_clustered' in binned_feats.columns:
        binned_feats = binned_feats.drop(columns=['is_clustered'])

    return binned_feats, meta_used


# ---------------------------------------------------------------------------
# Stop GAM group specs (mirror one_ff_gam: lam_f tuning, lam_g event, lam_h hist, lam_p coupling)
# ---------------------------------------------------------------------------

# Behavioral column groups aligned with decode_stops_design._build_feature_groups.
# (name, list of column names, vartype). Only columns present in design_df are used.
_STOP_BEHAVIORAL_GROUP_SPECS: List[tuple] = [
    # Kinematic (1D each)
    ('accel', ['accel'], '1D'),
    ('speed', ['speed'], '1D'),
    ('ang_speed', ['ang_speed'], '1D'),
    # One_ff_gam-style tuning (if present in design; from extra_agg_cols in encoding design)
    ('v', ['v'], '1D'),
    ('w', ['w'], '1D'),
    ('d', ['d'], '1D'),
    ('phi', ['phi'], '1D'),
    ('r_targ', ['r_targ'], '1D'),
    ('theta_targ', ['theta_targ'], '1D'),
    ('eye_ver', ['eye_ver'], '1D'),
    ('eye_hor', ['eye_hor'], '1D'),
    # Event design (event = temporal basis)
    ('basis', None, 'event'),   # cols: rcos_* without *captured; filled below
    ('basis*captured', None, 'event'),
    ('prepost', ['prepost'], '1D'),
    ('prepost*speed', ['prepost*speed'], '1D'),
    ('captured', ['captured'], '1D'),
    ('time_since_prev_event', ['time_since_prev_event'], '1D'),
    ('time_to_next_event', ['time_to_next_event'], '1D'),
    # Cluster
    ('cluster_flags', ['event_is_first_in_cluster'], '1D'),
    ('cluster_gaps', ['gap_since_prev_event_in_cluster_z',
     'gap_till_next_event_in_cluster_z'], '1D'),
    ('cluster_duration', ['cluster_duration_s_z'], '1D'),
    ('cluster_progress', ['cluster_progress_c', 'cluster_progress_c2'], '1D'),
    ('cluster_rel_time', ['bin_t_from_cluster_start_s_z'], '1D'),
    ('cluster_n_events', ['log_n_events_in_cluster_z'], '1D'),
    ('is_clustered', ['is_clustered'], '1D'),
    ('event_in_cluster_t', ['event_t_from_cluster_start_s'], '1D'),
    # Firefly
    ('ff_visible', ['log1p_num_ff_visible', 'k_ff_visible'], '1D'),
    ('ff_in_memory', ['log1p_num_ff_in_memory', 'k_ff_in_memory'], '1D'),
    # Retries
    ('retries', [
        'rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle',
        'rsw_last', 'rcap_last', 'one_stop_miss', 'whether_in_retry_series', 'miss',
    ], 'event'),
    # Timing
    ('time_rel_to_event_start', ['time_rel_to_event_start'], '1D'),
]


def build_stop_gam_groups(
    design_df: pd.DataFrame,
    *,
    lam_f: float = 100.0,
    lam_g: float = 10.0,
    lam_h: float = 10.0,
    lam_p: float = 10.0,
):
    """
    Build GroupSpec list and lambda config for stop-encoding Poisson GAM, matching one_ff_gam.

    - Const (if present): unpenalized (0D).
    - Behavioral columns: grouped by semantic role; 1D/event and lam_f or lam_g.
    - Spike history: first neuron -> 'spike_hist' (lam_h), rest -> 'cpl_J' (lam_p).

    Parameters
    ----------
    design_df : DataFrame
        Stop behavioral design with spike-history columns (from get_design_for_unit).
    lam_f : float
        Penalty for tuning/firefly-style groups (1D smooth).
    lam_g : float
        Penalty for event-style groups (multi-column basis).
    lam_h : float
        Penalty for self spike history.
    lam_p : float
        Penalty for coupling (other neurons' history).

    Returns
    -------
    groups : List[GroupSpec]
        For use with fit_poisson_gam / fit_stop_poisson_gam.
    lambda_config : dict
        {'lam_f': lam_f, 'lam_g': lam_g, 'lam_h': lam_h, 'lam_p': lam_p} for generate_lambda_suffix.
    """
    cols_all = list(design_df.columns)
    # Spike-history columns: cluster_<id>:b0:<k>
    spike_hist_pattern = re.compile(r'^(cluster_\d+):b0:\d+$')
    spike_hist_cols_by_neuron: Dict[str, List[str]] = {}
    behavioral_candidates: List[str] = []
    for c in cols_all:
        m = spike_hist_pattern.match(c)
        if m:
            neuron = m.group(1)
            spike_hist_cols_by_neuron.setdefault(neuron, []).append(c)
        elif c == 'const':
            continue  # handled separately
        else:
            behavioral_candidates.append(c)

    groups: List[GroupSpec] = []

    # Const (unpenalized)
    if 'const' in cols_all:
        groups.append(GroupSpec('const', ['const'], '0D', 0.0))

    # Behavioral groups (mirror one_ff: lam_f for 1D, lam_g for event)
    def _cols_present(candidates) -> List[str]:
        if candidates is None:
            return []
        return [c for c in candidates if c in design_df.columns]

    for spec in _STOP_BEHAVIORAL_GROUP_SPECS:
        name, candidate_cols, vartype = spec
        if name in ('basis', 't_basis'):
            candidate_cols = [c for c in design_df.columns if c.startswith(
                'rcos_') and '*captured' not in c]
        elif name in ('basis*captured', 't_basis_captured'):
            candidate_cols = [c for c in design_df.columns if c.startswith(
                'rcos_') and c.endswith('*captured')]
        else:
            candidate_cols = _cols_present(candidate_cols)
        if not candidate_cols:
            continue
        lam = lam_g if vartype == 'event' else lam_f
        groups.append(GroupSpec(name, candidate_cols, vartype, lam))

    # One_ff-style tuning boxcar columns (var:bin0 .. var:binK from build_tuning_design_stop)
    tuning_boxcar_pattern = re.compile(r'^(\w+):bin\d+$')
    tuning_cols_by_var: Dict[str, List[str]] = {}
    for c in behavioral_candidates:
        if c in design_df.columns:
            m = tuning_boxcar_pattern.match(c)
            if m:
                tuning_cols_by_var.setdefault(m.group(1), []).append(c)
    for var, cols in tuning_cols_by_var.items():
        cols = sorted(cols, key=lambda s: int(s.split(':bin')[1]))
        groups.append(GroupSpec(f'{var}_boxcar', cols, '1D', lam_f))

    # Any behavioral column not in a named group -> single "other" group (event penalty)
    assigned = set()
    for g in groups:
        assigned.update(g.cols)
    other_cols = [c for c in behavioral_candidates if c not in assigned]
    if other_cols:
        groups.append(GroupSpec('t_other', other_cols, 'event', lam_g))

    # Spike history: first neuron = spike_hist (lam_h), rest = cpl_0, cpl_1, ... (lam_p)
    neuron_order = sorted(spike_hist_cols_by_neuron.keys())
    for i, neuron in enumerate(neuron_order):
        hist_cols = spike_hist_cols_by_neuron[neuron]
        hist_cols.sort(key=lambda s: int(s.split(':b0:')[1]))
        if i == 0:
            groups.append(GroupSpec('spike_hist', hist_cols, 'event', lam_h))
        else:
            groups.append(GroupSpec(f'cpl_{i - 1}', hist_cols, 'event', lam_p))

    lambda_config = {
        'lam_f': lam_f,
        'lam_g': lam_g,
        'lam_h': lam_h,
        'lam_p': lam_p,
    }
    return groups, lambda_config

