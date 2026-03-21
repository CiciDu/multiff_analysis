
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import pandas as pd


from neural_data_analysis.design_kits.design_around_event import (
    stop_design,
    cluster_design,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils


from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design

from neural_data_analysis.design_kits.design_by_segment import (
    other_feats
)

# ---------------------------------------------------------------------------
# Stop GAM group specs (mirror one_ff_gam: lam_f tuning, lam_g event, lam_h hist, lam_p coupling)
# ---------------------------------------------------------------------------

def build_stop_encoding_design(
    pn,
    datasets,
    new_seg_info,
    events_with_stats,
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
    add_stop_cluster_features: bool = True,
    add_retry_features: bool = True,
):
    """
    Full stop-aligned encoding design builder.
    """

    # ==============================================================
    # 3-4) Shared binning core (event windows -> binned feats/spikes)
    # ==============================================================
    (
        bins_2d,
        meta,
        binned_feats,
        exposure,
        used_bins,
        mask_used,
        pos,
        meta_df_used,
        binned_spikes,
        _,
    ) = encoding_design_utils.bin_event_windows_core(
        new_seg_info=new_seg_info,
        monkey_information=pn.monkey_information,
        spikes_df=pn.spikes_df,
        bin_dt=bin_width,
        global_bins_2d=global_bins_2d,
        agg_cols=encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS,
        time_col='time',
        cluster_col='cluster',
        verbose=False,
    )

    binned_feats = encoding_design_utils._ensure_one_ff_style_covariates(
        binned_feats
    )

    # ==============================================================
    # 5) Continuous tuning block
    # ==============================================================
    raw_feature_cols_to_drop=encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS
    raw_feature_cols_to_drop.remove('time')
    binned_feats, tuning_meta, _ = (
        encoding_design_utils.add_tuning_features_to_design(
            binned_feats,
            use_boxcar=use_boxcar,
            tuning_feature_mode=tuning_feature_mode,
            binrange_dict=binrange_dict,
            tuning_n_bins=tuning_n_bins,
            linear_vars=linear_vars,
            angular_vars=angular_vars,
            raw_feature_cols_to_drop=raw_feature_cols_to_drop,
        )
    )

    # Ensure 'time' (kept as raw, not splines) is in tuning_meta
    binned_feats, tuning_meta = other_feats.add_raw_feature(
        binned_feats,
        feature='time',
        data=binned_feats,
        name='time',
        transform='linear',
        eps=1e-6,
        center=True,
        scale=True,
        meta=tuning_meta,
    )

    # ==============================================================
    # 6) Cluster-level features
    # ==============================================================
    if add_stop_cluster_features:
            
        cluster_df = cluster_design.build_cluster_features_workflow(
            meta_df_used[['event_id', 'rel_center']],
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

        for k in cluster_feats:
            binned_feats, tuning_meta = other_feats.add_raw_feature(
                binned_feats,
                feature=k,
                data=cluster_df,
                name=k,
                transform='linear',
                eps=1e-6,
                center=True,
                scale=False,
                meta=tuning_meta,
            )

    # Add relative time to event start
    meta_df_used['time_rel_to_event_start'] = meta_df_used['rel_center'].to_numpy()

    binned_feats, tuning_meta = other_feats.add_raw_feature(
        binned_feats,
        feature='time_rel_to_event_start',
        data=meta_df_used,
        name='time_rel_to_event_start',
        transform='linear',
        eps=1e-6,
        center=True,
        scale=False,
        meta=tuning_meta,
    )

    # ==============================================================
    # 7) Firefly + Retry features (clean integration)
    # ==============================================================
    # Add raw columns directly
    binned_feats, ff_added_cols = add_ff_visible_and_in_memory_counts(
        binned_feats,
        bins_2d,
        pn.ff_dataframe,
        used_bins,
    )

    if add_retry_features:
        binned_feats, retry_added_cols = add_retries_info_to_binned_feats(
            binned_feats,
            new_seg_info,
            datasets,
            meta_df_used,
        )
    else: 
        retry_added_cols = []

    # Register into tuning_meta
    for k in ff_added_cols + retry_added_cols:
        binned_feats, tuning_meta = other_feats.add_raw_feature(
            binned_feats,
            feature=k,
            data=binned_feats,
            name=k,
            transform='linear',
            eps=1e-6,
            center=True,
            scale=False,
            meta=tuning_meta,
        )

    # ==============================================================
    # 8) Summed stop-event temporal basis
    # ==============================================================
    if 't_center' not in meta_df_used:
        raise ValueError('meta_df_used missing required column "t_center"')

    if 'event_id_start_time' not in events_with_stats:
        raise ValueError(
            'events_with_stats missing required column "event_id_start_time"'
        )

    temporal_df, temporal_meta = (
        encoding_design_utils.build_temporal_design_from_event_times(
            bin_t_center=meta_df_used['t_center'].to_numpy(dtype=float),
            event_times=events_with_stats['event_id_start_time'].to_numpy(dtype=float),
            bin_dt=bin_width,
            n_basis=n_basis,
            t_min=t_min,
            t_max=t_max,
            index=binned_feats.index,
            event_name='stop',
        )
    )

    binned_feats = pd.concat([binned_feats, temporal_df], axis=1)

    # ==============================================================
    # 9) Drop constant columns (except const)
    # ==============================================================
    const_cols = [
        c for c in binned_feats.columns
        if c != 'const' and binned_feats[c].nunique() <= 1
    ]
    binned_feats = binned_feats.drop(columns=const_cols)

    # ==============================================================
    # 10) Final sorting
    # ==============================================================
    sort_idx = meta_df_used.sort_values(
        ['event_id', 'k_within_seg']
    ).index

    meta_df_used = meta_df_used.loc[sort_idx].reset_index(drop=True)
    binned_feats = binned_feats.loc[sort_idx].reset_index(drop=True)
    binned_spikes = binned_spikes.loc[sort_idx].reset_index(drop=True)

    return (
        pn,
        binned_spikes,
        binned_feats,
        meta_df_used,
        temporal_meta,
        tuning_meta,
    )

def add_ff_visible_and_in_memory_counts(
    binned_feats: pd.DataFrame,
    bins_2d: np.ndarray,
    ff_dataframe: pd.DataFrame,
    used_bins: np.ndarray,
    *,
    max_in_memory_time_since_seen: float = 2.0,
    add_log: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Add firefly count features to binned_feats.

    Adds:
        - num_ff_visible
        - num_ff_in_memory
        - optionally log1p_num_ff_<state>

    Returns
    -------
    binned_feats : pd.DataFrame
    added_columns : list
        Names of columns added.
    """

    added_columns = []

    # Restrict to fireflies that qualify as "in memory"
    ff_df_sub = ff_dataframe.loc[
        ff_dataframe['time_since_last_vis'] < max_in_memory_time_since_seen
    ].copy()

    ff_df_sub['in_memory'] = 1

    for state in ['visible', 'in_memory']:

        counts = vis_design.count_visible_from_time_df_fast(
            ff_df_sub,
            bins_2d,
            vis_col=state,
        )

        counts_used = counts[used_bins]

        num_col = f'num_ff_{state}'
        binned_feats[num_col] = counts_used
        added_columns.append(num_col)

        if add_log:
            log_col = f'log1p_num_ff_{state}'
            binned_feats[log_col] = np.log1p(counts_used)
            added_columns.append(log_col)

    return binned_feats, added_columns


def add_retries_info_to_binned_feats(
    binned_feats: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    datasets: dict,
    meta_df_used: pd.DataFrame,
) -> Tuple[pd.DataFrame, list]:
    """
    Add retry-related binary indicators to binned_feats.

    Returns
    -------
    binned_feats : pd.DataFrame
    added_columns : list
        Names of columns added.
    """

    if datasets is None:
        raise ValueError('datasets is required to add retries info')

    RETRY_CATEGORIES = [
        'rsw_first', 'rcap_first',
        'rsw_middle', 'rcap_middle',
        'rsw_last', 'rcap_last',
        'one_stop_miss',
    ]

    added_columns = []
    new_seg_info = new_seg_info.copy()

    # --------------------------------------------------
    # Stop-level flags
    # --------------------------------------------------
    for col in RETRY_CATEGORIES:
        new_seg_info[col] = 0
        stop_ids = datasets[col]['stop_id'].unique()
        new_seg_info.loc[new_seg_info['stop_id'].isin(stop_ids), col] = 1


    # --------------------------------------------------
    # Expand to per-bin table
    # --------------------------------------------------
    event_tbl = stop_design.build_per_event_table(
        new_seg_info,
        extras=RETRY_CATEGORIES,
    )

    meta_df_used = stop_design.join_event_tbl_avoid_collisions(
        meta_df_used,
        event_tbl,
    )

    # Ensure alignment
    meta_df_used = meta_df_used.sort_values('bin').reset_index(drop=True)
    binned_feats = binned_feats.reset_index(drop=True)

    if len(meta_df_used) != len(binned_feats):
        raise ValueError(
            f'meta_df_used and binned_feats misaligned '
            f'({len(meta_df_used)} vs {len(binned_feats)})'
        )

    # --------------------------------------------------
    # Transfer columns
    # --------------------------------------------------
    for col in RETRY_CATEGORIES:
        if col in meta_df_used:
            binned_feats[col] = meta_df_used[col].to_numpy(dtype='int8')
            added_columns.append(col)

    # --------------------------------------------------
    # Derived indicators
    # --------------------------------------------------
    retry_cols = ['rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle']
    miss_cols = retry_cols + ['rsw_last', 'one_stop_miss']

    binned_feats['whether_in_retry_series'] = (
        binned_feats[retry_cols].any(axis=1).astype('int8')
    )
    binned_feats['miss'] = (
        binned_feats[miss_cols].any(axis=1).astype('int8')
    )

    added_columns.extend(['whether_in_retry_series', 'miss'])

    # Optional: remove collinear column if present
    if 'is_clustered' in binned_feats.columns:
        binned_feats = binned_feats.drop(columns=['is_clustered'])

    return binned_feats, added_columns



"""
Note: this module is for stop *encoding* utilities.

Decoding/GLM stop design builders live in
`topic_based_neural_analysis/stop_event_analysis/get_stop_events/decode_stops_design.py`.
"""
