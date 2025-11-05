# --- Standard library
import os
import sys
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

# --- Matplotlib / pandas defaults
plt.rcParams['animation.html'] = 'html5'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
np.set_printoptions(suppress=True)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# =========================
# Stop Design Builder
# =========================
from typing import Tuple


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
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    '''
    Build a stop-aligned, bin-level design matrix and offset for GLM/decoding.

    Args:
        new_seg_info: DataFrame of per-event segments with windows.
        events_with_stats: DataFrame of event-level stats for clustering.
        monkey_information: Continuous kinematics (must include 'time').
        spikes_df: Spike times with ['time', 'cluster'].
        ff_dataframe: Firefly visibility table with ['time_since_last_vis', 'visible'].
        bin_dt: Bin width in seconds (default = 0.04).
        add_ff_visible_info: Whether to add visibility/in-memory features.
        add_retries_info: Whether to add retry-related indicators.
        datasets: Dictionary of categorized retry event tables (required if add_retries_info=True).

    Returns:
        binned_spikes: DataFrame of spike counts per cluster.
        binned_feats: DataFrame of behavioral & event covariates.
        offset_log: np.ndarray of log(exposure) per used bin.
    '''
    # 1) Build bins from event windows
    bins_2d, meta = event_binning.event_windows_to_bins2d(
        new_seg_info, bin_dt=bin_dt, only_ok=False
    )

    # 2) Assign samples to bins
    sample_idx, bin_idx_arr, dt_arr, n_bins = event_binning.build_bin_assignments(
        monkey_information['time'].to_numpy(), bins_2d
    )

    # 3) Subselect continuous samples
    monkey_sub = monkey_information.iloc[sample_idx].copy()

    # 3a) Exposure & used_bins
    _, exposure, used_bins = event_binning.bin_timeseries_weighted(
        monkey_sub['time'].to_numpy(), dt_arr, bin_idx_arr, how='mean'
    )

    # 3b) Aggregate helper
    def _agg_feat(col: str) -> np.ndarray:
        vals = monkey_sub[col].to_numpy()
        out, exp_chk, used_bins_chk = event_binning.bin_timeseries_weighted(
            vals, dt_arr, bin_idx_arr, how='mean'
        )
        if not (np.allclose(exp_chk, exposure) and np.array_equal(used_bins_chk, used_bins)):
            raise ValueError('Exposure/used_bins mismatch while aggregating features.')
        return out

    # 3c) Aggregate kinematics
    binned_feats = pd.DataFrame({
        'accel': _agg_feat('accel'),
        'speed': _agg_feat('speed'),
        'ang_speed': _agg_feat('ang_speed'),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3d) Keep bins with exposure > 0
    mask_used = exposure > 0
    pos = used_bins[mask_used]
    binned_feats = binned_feats.iloc[mask_used].reset_index(drop=True)

    meta_by_bin = meta.set_index('bin').sort_index()
    meta_used = meta_by_bin.loc[pos].reset_index()

    # 4) Bin spikes per cluster
    spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
        spikes_df, bins_2d, time_col='time', cluster_col='cluster'
    )

    binned_spikes = pd.DataFrame(
        spike_counts[pos, :],
        columns=cluster_ids,
    ).reset_index(drop=True)

    # 5) Build event- and cluster-level features
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
        meta_used[['event_id', 'rel_center']], events_with_stats,
        rel_time_col='rel_center',
        winsor_p=0.5,
        use_midbin_progress=True,
        zscore_progress=False,
        zscore_rel_time=True,
    )

    cluster_feats = [
        'is_clustered',
        'event_is_first_in_cluster',
        'prev_gap_s_z',
        'next_gap_s_z',
        'cluster_duration_s_z',
        'cluster_progress_c', 'cluster_progress_c2',
        'cluster_rel_time_s_z',
    ]

    cols_to_add_from_event_design = [c for c in X_event_df.columns if c not in binned_feats.columns]
    binned_feats.loc[:, cols_to_add_from_event_design] = X_event_df[cols_to_add_from_event_design].to_numpy()

    cols_to_add_from_cluster_design = [c for c in cluster_feats if c not in binned_feats.columns]
    binned_feats.loc[:, cols_to_add_from_cluster_design] = cluster_df[cols_to_add_from_cluster_design].to_numpy()

    # Offset term
    offset_log = np.log(np.clip(exposure[mask_used], 1e-12, None))
    binned_feats['time_rel_to_event_start'] = meta_used['rel_center']

    # 6) Add firefly visibility features
    if add_ff_visible_info:
        binned_feats = add_ff_visible_and_in_memory_info(
            binned_feats, bins_2d, ff_dataframe, used_bins, max_in_memory_time_since_seen=2
        )

    # 7) Add retry-related features
    if add_retries_info:
        binned_feats, meta_used = add_retries_info_to_binned_feats(
            binned_feats, new_seg_info, datasets, meta_used
        )
        
    binned_spikes = binned_spikes.reset_index(drop=True)
    binned_feats.reset_index(drop=True, inplace=True)
    meta_used.reset_index(drop=True, inplace=True)

    return binned_spikes, binned_feats, offset_log, meta_used


def add_retries_info_to_binned_feats(binned_feats, new_seg_info, datasets, meta_used):
    assert datasets is not None, 'datasets is required to add retries info'
    
    retries_info = [
        ('GUAT_first', datasets['GUAT_first']),
        ('TAFT_first', datasets['TAFT_first']),
        ('GUAT_middle', datasets['GUAT_middle']),
        ('TAFT_middle', datasets['TAFT_middle']),
        ('GUAT_last', datasets['GUAT_last']),
        ('TAFT_last', datasets['TAFT_last']),
        ('one_stop_miss', datasets['one_stop_miss'])
    ]
    
    retries_columns = [col for col, _ in retries_info]
    for col, df in retries_info:
        new_seg_info[col] = 0
        category_stop_ids = df['stop_id'].unique()
        new_seg_info.loc[new_seg_info['stop_id'].isin(category_stop_ids), col] = 1

    event_tbl = stop_design.build_per_event_table(new_seg_info, extras=retries_columns)
    meta_used = stop_design.join_event_tbl_avoid_collisions(meta_used, event_tbl)

    # enforce alignment safety
    meta_used = meta_used.sort_values('bin').reset_index(drop=True)
    binned_feats = binned_feats.reset_index(drop=True)
    assert len(meta_used) == len(binned_feats), 'meta_used and binned_feats misaligned'

    # add base retry columns
    for col in retries_columns:
        binned_feats[col] = meta_used[col].to_numpy()

    # derived columns
    retry_cols = ['GUAT_first', 'TAFT_first', 'GUAT_middle', 'TAFT_middle']
    miss_cols = retry_cols + ['GUAT_last', 'one_stop_miss']

    binned_feats['whether_retry'] = binned_feats[retry_cols].any(axis=1).astype('int8')
    binned_feats['miss'] = binned_feats[miss_cols].any(axis=1).astype('int8')
    
    # also drop the column 'is_clustered' to avoid perfect collinearity
    binned_feats = binned_feats.drop(columns=['is_clustered'])

    return binned_feats, meta_used


def add_ff_visible_and_in_memory_info(binned_feats, bins_2d, ff_dataframe, used_bins, max_in_memory_time_since_seen=2):
    ff_df_sub = ff_dataframe[ff_dataframe['time_since_last_vis'] < max_in_memory_time_since_seen].copy()
    ff_df_sub['in_memory'] = 1

    for state in ['visible', 'in_memory']:
        k_ff_visible = vis_design.count_visible_from_time_df_fast(
            ff_df_sub, bins_2d, vis_col=state
        )
        any_ff_visible = (k_ff_visible > 0).astype('int8')
        binned_feats[f'any_ff_{state}'] = any_ff_visible[used_bins]
        binned_feats[f'k_ff_{state}'] = k_ff_visible[used_bins]
    return binned_feats

def subset_binned_data(binned_feats, binned_spikes, offset_log, meta_used, mask):
    return (
        binned_feats.loc[mask].reset_index(drop=True),
        binned_spikes.loc[mask].reset_index(drop=True),
        offset_log[mask],
        meta_used.loc[mask].reset_index(drop=True)
    )
