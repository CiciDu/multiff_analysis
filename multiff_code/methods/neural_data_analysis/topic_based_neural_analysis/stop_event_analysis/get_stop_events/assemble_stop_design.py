# --- Standard library
from typing import Tuple, Dict, List
import os
from pathlib import Path

# --- Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# --- Neuroscience / modeling imports
from neural_data_analysis.design_kits.design_around_event import (
    event_binning, stop_design, cluster_design
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design

# =============================================================================
# Global environment & plotting defaults
# =============================================================================

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
os.environ.setdefault('PYDEVD_DISABLE_FILE_VALIDATION', '1')

plt.rcParams['animation.html'] = 'html5'
rc('animation', html='jshtml')

pd.set_option('display.float_format', lambda x: f'{x:.5f}')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
np.set_printoptions(suppress=True)

# =============================================================================
# Stop Design Builder
# =============================================================================


def build_stop_design(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    spikes_df: pd.DataFrame,
    ff_dataframe: pd.DataFrame,
    bin_dt: float = 0.04,
    add_ff_visible_info: bool = True,
    add_retries_info: bool = True,
    datasets: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """
    Build a stop-aligned, bin-level design matrix and offset for GLM/decoding.

    Returns
    -------
    binned_spikes : pd.DataFrame
        (n_bins × n_clusters) spike counts aligned to `binned_feats`
    binned_feats : pd.DataFrame
        (n_bins × n_features) design matrix (exposure > 0 only)
    offset_log : np.ndarray
        log(exposure) per bin
    meta_used : pd.DataFrame
        Per-bin metadata aligned row-wise to binned_feats
    groups : Dict[str, List[str]]
        Semantic feature groupings
    """

    # -------------------------------------------------------------------------
    # 1) Build bins from event windows
    # -------------------------------------------------------------------------
    bins_2d, meta = event_binning.event_windows_to_bins2d(
        new_seg_info, bin_dt=bin_dt, only_ok=False
    )

    # -------------------------------------------------------------------------
    # 2) Assign continuous samples to bins
    # -------------------------------------------------------------------------
    sample_idx, bin_idx_arr, dt_arr, _ = event_binning.build_bin_assignments(
        monkey_information['time'].to_numpy(), bins_2d
    )

    monkey_sub = monkey_information.iloc[sample_idx].copy()

    # -------------------------------------------------------------------------
    # 3) Compute exposure and valid bins
    # -------------------------------------------------------------------------
    _, exposure, used_bins = event_binning.bin_timeseries_weighted(
        monkey_sub['time'].to_numpy(), dt_arr, bin_idx_arr, how='mean'
    )

    def _agg_feat(col: str) -> np.ndarray:
        vals = monkey_sub[col].to_numpy()
        out, exp_chk, used_bins_chk = event_binning.bin_timeseries_weighted(
            vals, dt_arr, bin_idx_arr, how='mean'
        )
        if not np.allclose(exp_chk, exposure):
            raise ValueError(f'Exposure mismatch while aggregating "{col}"')
        if not np.array_equal(used_bins_chk, used_bins):
            raise ValueError(f'used_bins mismatch while aggregating "{col}"')
        return out

    # -------------------------------------------------------------------------
    # 4) Aggregate kinematics
    # -------------------------------------------------------------------------
    KINEMATIC_COLS = ['accel', 'speed', 'ang_speed']
    binned_feats = (
        pd.DataFrame({c: _agg_feat(c) for c in KINEMATIC_COLS})
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    mask_used = exposure > 0
    pos = used_bins[mask_used]

    binned_feats = binned_feats.iloc[mask_used].reset_index(drop=True)

    meta_used = (
        meta.set_index('bin')
        .sort_index()
        .loc[pos]
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # 5) Bin spikes per cluster
    # -------------------------------------------------------------------------
    spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
        spikes_df, bins_2d, time_col='time', cluster_col='cluster'
    )

    binned_spikes = (
        pd.DataFrame(spike_counts[pos, :], columns=cluster_ids)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # 6) Event- and cluster-level features
    # -------------------------------------------------------------------------
    X_event_df = stop_design.build_event_design_from_meta(
        meta=meta,
        pos=pos,
        new_seg_info=new_seg_info,
        speed_used=binned_feats['speed'].values,
        include_columns=(
            'basis', 'prepost', 'prepost*speed',
            'captured', 'basis*captured',
            'time_since_prev_event', 'time_to_next_event',
        )
    )

    cluster_df = cluster_design.build_cluster_features_workflow(
        meta_used[['event_id', 'rel_center']],
        events_with_stats,
        rel_time_col='rel_center',
        winsor_p=0.5,
        use_midbin_progress=True,
        zscore_progress=False,
        zscore_rel_time=True,
    )

    CLUSTER_FEATS = [
        'is_clustered',
        'event_is_first_in_cluster',
        'prev_gap_s_z',
        'next_gap_s_z',
        'cluster_duration_s_z',
        'cluster_progress_c',
        'cluster_progress_c2',
        'cluster_rel_time_s_z',
    ]

    def _safe_add_columns(target_df, source_df, cols):
        cols = [c for c in cols if c not in target_df.columns]
        if cols:
            target_df.loc[:, cols] = source_df[cols].to_numpy()

    _safe_add_columns(binned_feats, X_event_df, X_event_df.columns)
    _safe_add_columns(binned_feats, cluster_df, CLUSTER_FEATS)

    # -------------------------------------------------------------------------
    # 7) Offset term and timing
    # -------------------------------------------------------------------------
    offset_log = np.log(np.clip(exposure[mask_used], 1e-12, None))
    binned_feats['time_rel_to_event_start'] = meta_used['rel_center'].to_numpy()

    # -------------------------------------------------------------------------
    # 8) Firefly visibility / memory features
    # -------------------------------------------------------------------------
    if add_ff_visible_info:
        binned_feats = add_ff_visible_and_in_memory_info(
            binned_feats,
            bins_2d,
            ff_dataframe,
            used_bins,
            max_in_memory_time_since_seen=2,
        )

    # -------------------------------------------------------------------------
    # 9) Retry-related features
    # -------------------------------------------------------------------------
    if add_retries_info:
        binned_feats, meta_used = add_retries_info_to_binned_feats(
            binned_feats, new_seg_info, datasets, meta_used
        )

    # -------------------------------------------------------------------------
    # 10) Feature groups
    # -------------------------------------------------------------------------
    groups: Dict[str, List[str]] = {}

    def _add_group(name: str, cols: List[str]):
        cols = [c for c in cols if c in binned_feats.columns]
        if cols:
            groups[name] = cols

    for c in KINEMATIC_COLS:
        _add_group(c, [c])

    _add_group(
        'basis',
        [c for c in binned_feats.columns if c.startswith('rcos_') and '*captured' not in c]
    )
    _add_group(
        'basis*captured',
        [c for c in binned_feats.columns if c.startswith('rcos_') and c.endswith('*captured')]
    )

    _add_group('prepost', ['prepost'])
    _add_group('prepost*speed', ['prepost*speed'])
    _add_group('captured', ['captured'])
    _add_group('time_since_prev_event', ['time_since_prev_event'])
    _add_group('time_to_next_event', ['time_to_next_event'])

    _add_group('cluster_flags', ['event_is_first_in_cluster'])
    _add_group('cluster_gaps', ['prev_gap_s_z', 'next_gap_s_z'])
    _add_group('cluster_duration', ['cluster_duration_s_z'])
    _add_group('cluster_progress', ['cluster_progress_c', 'cluster_progress_c2'])
    _add_group('cluster_rel_time', ['cluster_rel_time_s_z'])

    _add_group('ff_visible', ['any_ff_visible', 'k_ff_visible'])
    _add_group('ff_in_memory', ['any_ff_in_memory', 'k_ff_in_memory'])

    _add_group(
        'retries',
        [
            'rsw_first', 'rcap_first',
            'rsw_middle', 'rcap_middle',
            'rsw_last', 'rcap_last',
            'one_stop_miss',
            'whether_in_retry_series',
            'miss',
        ],
    )

    _add_group('time_rel_to_event_start', ['time_rel_to_event_start'])

    return binned_spikes, binned_feats, offset_log, meta_used, groups


# =============================================================================
# Retry features
# =============================================================================


def add_retries_info_to_binned_feats(binned_feats, new_seg_info, datasets, meta_used):
    assert datasets is not None, 'datasets is required to add retries info'

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
        new_seg_info, extras=RETRY_CATEGORIES
    )

    meta_used = stop_design.join_event_tbl_avoid_collisions(
        meta_used, event_tbl
    )

    meta_used = meta_used.sort_values('bin').reset_index(drop=True)
    binned_feats = binned_feats.reset_index(drop=True)

    if len(meta_used) != len(binned_feats):
        raise ValueError('meta_used and binned_feats misaligned')

    for col in RETRY_CATEGORIES:
        binned_feats[col] = meta_used[col].to_numpy()

    retry_cols = ['rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle']
    miss_cols = retry_cols + ['rsw_last', 'one_stop_miss']

    binned_feats['whether_in_retry_series'] = (
        binned_feats[retry_cols].any(axis=1).astype('int8')
    )
    binned_feats['miss'] = (
        binned_feats[miss_cols].any(axis=1).astype('int8')
    )

    # drop to avoid perfect collinearity
    if 'is_clustered' in binned_feats.columns:
        binned_feats = binned_feats.drop(columns=['is_clustered'])

    return binned_feats, meta_used


# =============================================================================
# Firefly visibility / memory
# =============================================================================


def add_ff_visible_and_in_memory_info(
    binned_feats,
    bins_2d,
    ff_dataframe,
    used_bins,
    max_in_memory_time_since_seen=2,
):
    ff_df_sub = ff_dataframe[
        ff_dataframe['time_since_last_vis'] < max_in_memory_time_since_seen
    ].copy()
    ff_df_sub['in_memory'] = 1

    for state in ['visible', 'in_memory']:
        k_ff = vis_design.count_visible_from_time_df_fast(
            ff_df_sub, bins_2d, vis_col=state
        )
        binned_feats[f'any_ff_{state}'] = (k_ff > 0).astype('int8')[used_bins]
        binned_feats[f'k_ff_{state}'] = k_ff[used_bins]

    return binned_feats


# =============================================================================
# Subsetting helper
# =============================================================================


def subset_binned_data(binned_feats, binned_spikes, offset_log, meta_used, mask):
    return (
        binned_feats.loc[mask].reset_index(drop=True),
        binned_spikes.loc[mask].reset_index(drop=True),
        offset_log[mask],
        meta_used.loc[mask].reset_index(drop=True),
    )
