
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import pandas as pd
import re

import numpy as np
import pandas as pd

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
    stop_design,
    cluster_design,
)
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encode_stops_utils, encoding_design_utils


from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design


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
    build_tuning_design_for_continuous_vars), adds one group per variable (var:bin0..); otherwise
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
        for c in encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS:
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



def build_stop_encoding_design(
    raw_data_folder_path: str,
    bin_width: float,
    global_bins_2d: Optional[np.ndarray] = None,
    n_basis: int = 20,
    t_min: float = -0.3,
    t_max: float = 0.3,
    use_boxcar: bool = False,
    tuning_feature_mode: Optional[str] = None,
    binrange_dict: Optional[Dict[str, Union[np.ndarray, Tuple[float, float]]]] = None,
    tuning_n_bins: int = 10,
    linear_vars: Optional[List[str]] = None,
    angular_vars: Optional[List[str]] = None,
    use_planning_rename: bool = True,
    custom_rename: Optional[Dict[str, str]] = None,
):
    """
    Full stop-aligned encoding design builder.

    This is now the single entry point.

    Pipeline:
        1) Load and prepare stop inputs
        2) Build base binning/spikes/cluster/ff/retries
        3) Add summed stop-event temporal basis
        4) Optionally add tuning boxcar features
        5) Build semantic groups
        6) Drop constant columns

    Returns
    -------
    pn
    binned_spikes
    binned_feats
    offset_log
    stop_meta_used
    stop_meta_groups
    temporal_meta
    tuning_meta
    """

    # ------------------------------------------------------------------
    # 1) Prepare stop inputs
    # ------------------------------------------------------------------
    pn, datasets, new_seg_info, events_with_stats = \
        decode_stops_design._prepare_stop_design_inputs(
            raw_data_folder_path,
            bin_width,
        )

    # ------------------------------------------------------------------
    # 2) Rename planning vars if requested
    # ------------------------------------------------------------------
    if use_planning_rename or custom_rename:
        monkey_for_encoding = encoding_design_utils.monkey_information_for_encoding(
            pn.monkey_information,
            rename_planning_to_one_ff=use_planning_rename,
            custom_rename=custom_rename,
        )
    else:
        monkey_for_encoding = pn.monkey_information

    # ------------------------------------------------------------------
    # 3) Base design (binning + cluster + ff + retries)
    # ------------------------------------------------------------------
    kinematic_cols = []
    extra_agg_cols = encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS

    (
        binned_spikes,
        binned_feats,
        offset_log,
        meta_used,
        _,
    ) = encode_stops_utils.build_stop_design_base_for_encoding(
        new_seg_info=new_seg_info,
        events_with_stats=events_with_stats,
        monkey_information=monkey_for_encoding,
        spikes_df=pn.spikes_df,
        ff_dataframe=pn.ff_dataframe,
        bin_dt=bin_width,
        add_ff_visible_info=True,
        add_retries_info=True,
        datasets=datasets,
        global_bins_2d=global_bins_2d,
        extra_agg_cols=extra_agg_cols,
        kinematic_cols=kinematic_cols,
    )

    binned_feats = encoding_design_utils._ensure_one_ff_style_covariates(binned_feats)


    # ------------------------------------------------------------------
    # 4) Tuning block for continuous variables
    # ------------------------------------------------------------------
    binned_feats, tuning_meta, mode = encoding_design_utils.add_tuning_features_to_design(
        binned_feats,
        use_boxcar=use_boxcar,
        tuning_feature_mode=tuning_feature_mode,
        binrange_dict=binrange_dict,
        tuning_n_bins=tuning_n_bins,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        raw_feature_cols_to_drop=encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS,
    )

    # ------------------------------------------------------------------
    # 5) Summed stop-event temporal basis
    # ------------------------------------------------------------------
    if 't_center' not in meta_used.columns:
        raise ValueError('meta_used missing required column "t_center"')

    if 'event_id_start_time' not in events_with_stats.columns:
        raise ValueError(
            'events_with_stats missing required column "event_id_start_time"'
        )

    bin_t_center = meta_used['t_center'].to_numpy(dtype=float)
    stop_times = events_with_stats['event_id_start_time'].to_numpy(dtype=float)

    temporal_df, temporal_meta = encoding_design_utils.build_temporal_design_from_event_times(
        bin_t_center=bin_t_center,
        event_times=stop_times,
        bin_dt=bin_width,
        n_basis=n_basis,
        t_min=t_min,
        t_max=t_max,
        index=binned_feats.index,
        name_prefix='rcos_stop',
    )

    binned_feats = pd.concat([binned_feats, temporal_df], axis=1)


    # ------------------------------------------------------------------
    # 6) Drop constant columns (except const)
    # ------------------------------------------------------------------
    const_cols_to_drop = [
        c for c in binned_feats.columns
        if c != 'const' and binned_feats[c].nunique() <= 1
    ]
    binned_feats = binned_feats.drop(columns=const_cols_to_drop)

    return (
        pn,
        binned_spikes,
        binned_feats,
        offset_log,
        meta_used,
        temporal_meta,
        tuning_meta,
    )

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
        new_seg_info,
        bin_dt=bin_dt,
        only_ok=False,
        global_bins_2d=global_bins_2d,
    )

    # 2–4) Bin monkey_information into event bins
    binned_feats, exposure, used_bins, mask_used, pos = (
        encoding_design_utils._bin_monkey_information_feats_from_event_bins(
            monkey_information,
            bins_2d,
            kinematic_cols=kinematic_cols,
            extra_agg_cols=extra_agg_cols,
        )
    )

    # Align meta to kept bins (exposure > 0)
    meta_used = (
        meta.set_index('bin')
        .sort_index()
        .loc[pos]
        .reset_index()
    )

    # 5) Bin spikes per cluster
    spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
        spikes_df,
        bins_2d,
        time_col='time',
        cluster_col='cluster',
    )

    binned_spikes = (
        pd.DataFrame(spike_counts[pos, :], columns=cluster_ids)
        .reset_index(drop=True)
    )

    # 6) Cluster-level features (no legacy event-design features)
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
        'cluster_progress_c2',
        'event_t_from_cluster_start_s',
    ]

    _safe_add_columns(binned_feats, cluster_df, cluster_feats)

    # 7) Offset term and timing
    offset_log = np.log(np.clip(exposure[mask_used], 1e-12, None))
    binned_feats['time_rel_to_event_start'] = meta_used['rel_center'].to_numpy()

    # 8) Firefly visibility / memory features
    if add_ff_visible_info:
        binned_feats = _add_ff_visible_and_in_memory_info(
            binned_feats,
            bins_2d,
            ff_dataframe,
            used_bins,
        )

    # 9) Retry-related features
    if add_retries_info:
        binned_feats, meta_used = _add_retries_info_to_binned_feats(
            binned_feats,
            new_seg_info,
            datasets,
            meta_used,
        )

    # groups are rebuilt later in build_encode_stops_design
    return binned_spikes, binned_feats, offset_log, meta_used, {}

    
# --- HELPER: only bins monkey_information columns into bins_2d ---

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



# --- HELPER: only bins monkey_information columns into bins_2d ---

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

    # One_ff-style tuning boxcar columns (var:bin0 .. var:binK from build_tuning_design_for_continuous_vars)
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
            neuron_match = re.match(r'^cluster_(\d+)$', neuron)
            cpl_suffix = neuron_match.group(1) if neuron_match else str(i - 1)
            groups.append(
                GroupSpec(f'cpl_{cpl_suffix}', hist_cols, 'event', lam_p))

    lambda_config = {
        'lam_f': lam_f,
        'lam_g': lam_g,
        'lam_h': lam_h,
        'lam_p': lam_p,
    }
    return groups, lambda_config


def build_simple_gam_groups(
    design_df: pd.DataFrame,
    *,
    lam_f: float = 100.0,
    lam_g: float = 10.0,
    lam_h: float = 10.0,
    lam_p: float = 10.0,
):
    """
    Build GroupSpec list for any design (PN, Vis, etc.) with spike-history columns.

    - Const: unpenalized (0D).
    - Spike history: cluster_<id>:b0:<k> -> spike_hist (target) / cpl_J (coupling).
    - All other columns: single "behavioral" group with lam_f.

    Use when build_stop_gam_groups does not apply (different column semantics).
    """
    cols_all = list(design_df.columns)
    spike_hist_pattern = re.compile(r'^(cluster_\d+):b0:\d+$')
    spike_hist_cols_by_neuron: Dict[str, List[str]] = {}
    behavioral_candidates: List[str] = []
    for c in cols_all:
        m = spike_hist_pattern.match(c)
        if m:
            neuron = m.group(1)
            spike_hist_cols_by_neuron.setdefault(neuron, []).append(c)
        elif c == 'const':
            continue
        else:
            behavioral_candidates.append(c)

    groups: List[GroupSpec] = []
    if 'const' in cols_all:
        groups.append(GroupSpec('const', ['const'], '0D', 0.0))
    if behavioral_candidates:
        groups.append(
            GroupSpec('behavioral', behavioral_candidates, 'event', lam_f))

    neuron_order = sorted(spike_hist_cols_by_neuron.keys())
    for i, neuron in enumerate(neuron_order):
        hist_cols = spike_hist_cols_by_neuron[neuron]
        hist_cols.sort(key=lambda s: int(s.split(':b0:')[1]))
        if i == 0:
            groups.append(GroupSpec('spike_hist', hist_cols, 'event', lam_h))
        else:
            neuron_match = re.match(r'^cluster_(\d+)$', neuron)
            cpl_suffix = neuron_match.group(1) if neuron_match else str(i - 1)
            groups.append(
                GroupSpec(f'cpl_{cpl_suffix}', hist_cols, 'event', lam_p))

    lambda_config = {'lam_f': lam_f, 'lam_g': lam_g,
                     'lam_h': lam_h, 'lam_p': lam_p}
    return groups, lambda_config

