# --- Standard library
from typing import Tuple, Dict, List, Optional
import os

# --- Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import statsmodels.api as sm



# --- Neuroscience / modeling imports
from neural_data_analysis.design_kits.design_around_event import (
    event_binning, stop_design, cluster_design
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import vis_design

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils,
    collect_stop_data,
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoding_design_utils,
)

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import detrend_neural_data


from neural_data_analysis.design_kits.design_by_segment import rebin_segments

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


def _align_rebinned_spike_rates_to_meta(
    rebinned_spike_rates: pd.DataFrame,
    meta_df_used: pd.DataFrame,
    new_seg_for_rebin: pd.DataFrame,
) -> pd.DataFrame:
    """Align rebinned spike rates to meta_df_used row order; forward-fill NaN."""
    event_id_to_new_seg = dict(
        zip(new_seg_for_rebin['event_id'].values, new_seg_for_rebin['new_segment'].values)
    )
    meta_copy = meta_df_used.copy()
    meta_copy['_new_segment'] = meta_copy['event_id'].map(event_id_to_new_seg)
    merge_keys = meta_copy[['_new_segment', 'bin']].rename(columns={'bin': 'new_bin'})
    rebinned_aligned = rebinned_spike_rates.merge(
        merge_keys,
        left_on=['new_segment', 'new_bin'],
        right_on=['_new_segment', 'new_bin'],
        how='right',
    )
    id_cols = {'new_segment', 'new_bin', '_new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration'}
    cluster_cols = [c for c in rebinned_aligned.columns if c not in id_cols]
    out = rebinned_aligned[cluster_cols].reset_index(drop=True)
    if out.isna().any().any():
        out = out.ffill().bfill()
    return out


def _build_stop_core(
    new_seg_info: pd.DataFrame,
    events_with_stats: pd.DataFrame,
    monkey_information: pd.DataFrame,
    detrended_spike_rates: pd.DataFrame = None,
    spikes_df: pd.DataFrame = None,
    bin_dt: float = 0.04,
    global_bins_2d: np.ndarray = None,
    detrend_spikes: bool = True,
):
    """
    Shared core for building stop-aligned binned data structures.

    Handles:
    - Converting event windows to bin edges
    - Binning monkey kinematic information
    - Aligning metadata to used bins
    - Binning spikes by cluster
    - Building cluster-level features
    """
    agg_cols = encoding_design_utils.ONE_FF_STYLE_EXTRA_COLS

    (
        bins_2d,
        meta,
        binned_feats,
        exposure,
        used_bins,
        mask_used,
        pos,
        meta_df_used,
    ) = encoding_design_utils.bin_event_windows_core(
        new_seg_info=new_seg_info,
        monkey_information=monkey_information,
        bin_dt=bin_dt,
        global_bins_2d=global_bins_2d,
        agg_cols=agg_cols,
        verbose=True,
    )


    # rebin expects new_segment; event design uses event_id
    new_seg_for_rebin = new_seg_info.copy()
    if 'new_segment' not in new_seg_for_rebin.columns:
        new_seg_for_rebin['new_segment'] = np.arange(len(new_seg_for_rebin))


    if detrend_spikes and (detrended_spike_rates is not None):
        rebinned_spike_rates = rebin_segments.rebin_all_segments_global_bins(
            detrended_spike_rates,
            new_seg_for_rebin,
            bins_2d=bins_2d,
            bin_left_col='time_bin_start',
            bin_right_col='time_bin_end',
            bin_center_col='time_bin_center',
            how='mean',
            respect_old_segment=False,
            require_full_bin=False,
            add_bin_edges=False,
            add_support_duration=False,
        )

        rebinned_spike_rates = _align_rebinned_spike_rates_to_meta(
            rebinned_spike_rates, meta_df_used, new_seg_for_rebin
        )
    else:
        binned_spikes, _ = encoding_design_utils.bin_spikes_for_event_windows(
            spikes_df,
            bins_2d,
            pos,
            time_col='time',
            cluster_col='cluster',
        )  
        rebinned_spike_rates = (binned_spikes / bin_dt).copy()  

    cluster_df = cluster_design.build_cluster_features_workflow(
        meta_df_used[["event_id", "rel_center"]],
        events_with_stats,
        rel_time_col="rel_center",
        winsor_p=0.5,
        use_midbin_progress=True,
        zscore_progress=False,
        zscore_rel_time=True,
    )

    shared_cluster_feats = [
        "is_clustered",
        "event_is_first_in_cluster",
        "gap_since_prev_event_in_cluster_z",
        "gap_till_next_event_in_cluster_z",
        "cluster_duration_s_z",
        "cluster_progress_c",
        "bin_t_from_cluster_start_s_z",
        "log_n_events_in_cluster_z",
    ]

    return (
        bins_2d,
        meta,
        binned_feats,
        exposure,
        used_bins,
        mask_used,
        pos,
        meta_df_used,
        rebinned_spike_rates,
        cluster_df,
        shared_cluster_feats,
        agg_cols,
    )



def build_stop_design_decoding(
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
    detrend_spikes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """
    Build stop design for decoding (minimal event design, no interaction columns).
    """


    if detrend_spikes:
        detrended_df = detrend_neural_data.detrend_spikes_session_wide(
            spikes_df=spikes_df,
            bin_size=0.05,          # 50 ms fine bins for detrending
            drift_sigma_s=60.0,     # remove only very slow drift; tune 30-120 s
            center_method='subtract'
        )

        detrended_spike_rates, cluster_columns = detrend_neural_data.reshape_detrended_df_to_wide(
            detrended_df,
            value_col='detrended_rate_hz'
        )
    else:
        detrended_spike_rates = None

    (
        bins_2d,
        meta,
        binned_feats,
        exposure,
        used_bins,
        mask_used,
        pos,
        meta_df_used,
        rebinned_spike_rates,
        cluster_df,
        shared_cluster_feats,
        _,
    ) = _build_stop_core(
        new_seg_info=new_seg_info,
        events_with_stats=events_with_stats,
        monkey_information=monkey_information,
        detrended_spike_rates=detrended_spike_rates,
        spikes_df=spikes_df,
        bin_dt=bin_dt,
        global_bins_2d=global_bins_2d,
        detrend_spikes=detrend_spikes,
    )

    X_event_df = stop_design.build_event_design_for_decoding(
        meta=meta,
        pos=pos,
        new_seg_info=new_seg_info,
        include_columns=("prepost", "time_since_prev_event", "cond_dummies", "captured"),
    )
    cluster_feats = shared_cluster_feats

    _safe_add_columns(binned_feats, X_event_df, X_event_df.columns)
    _safe_add_columns(binned_feats, cluster_df, cluster_feats)

    offset_log = np.log(np.clip(exposure[mask_used], 1e-12, None))
    binned_feats["time_rel_to_event_start"] = meta_df_used["rel_center"].to_numpy()

    if add_ff_visible_info:
        binned_feats = add_ff_visible_and_in_memory_info(
            binned_feats,
            bins_2d,
            ff_dataframe,
            used_bins,
            max_in_memory_time_since_seen=2,
        )

    if add_retries_info:
        binned_feats, meta_df_used = add_retries_info_to_binned_feats(
            binned_feats, new_seg_info, datasets, meta_df_used
        )
    return rebinned_spike_rates, binned_feats, offset_log, meta_df_used



def _safe_add_columns(target_df, source_df, cols):
    cols = [c for c in cols if c not in target_df.columns]
    if cols:
        target_df.loc[:, cols] = source_df[cols].to_numpy()


def _build_feature_groups(
    binned_feats: pd.DataFrame,
    kinematic_cols: List[str],
    extra_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Build semantic feature groups from binned features.

    Parameters
    ----------
    binned_feats : pd.DataFrame
        DataFrame containing all binned features
    kinematic_cols : List[str]
        List of kinematic column names
    extra_cols : list, optional
        Additional aggregated columns (e.g. one_ff-style v, w, d, phi, r_targ, theta_targ, eye_ver, eye_hor)

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping group names to lists of feature column names
    """
    groups: Dict[str, List[str]] = {}

    def _add_group(name: str, cols: List[str]):
        cols = [c for c in cols if c in binned_feats.columns]
        if cols:
            groups[name] = cols

    # Individual kinematic columns
    for c in kinematic_cols:
        _add_group(c, [c])
    # One_ff-style extra tuning covariates (if present)
    for c in extra_cols or []:
        _add_group(c, [c])

    # Basis functions
    _add_group(
        'basis',
        [c for c in binned_feats.columns if c.startswith(
            'rcos_') and '*captured' not in c]
    )
    _add_group(
        'basis*captured',
        [c for c in binned_feats.columns if c.startswith(
            'rcos_') and c.endswith('*captured')]
    )

    # Pre/post and interaction terms
    _add_group('prepost', ['prepost'])
    _add_group('prepost*speed', ['prepost*speed'])
    _add_group('captured', ['captured'])
    _add_group('time_since_prev_event', ['time_since_prev_event'])
    _add_group('time_to_next_event', ['time_to_next_event'])

    # Cluster features
    _add_group('cluster_flags', ['event_is_first_in_cluster'])
    _add_group('cluster_gaps', [
               'gap_since_prev_event_in_cluster_z', 'gap_till_next_event_in_cluster_z'])
    _add_group('cluster_duration', ['cluster_duration_s_z'])
    _add_group('cluster_progress', [
               'cluster_progress_c', 'cluster_progress_c2'])
    _add_group('cluster_rel_time', ['bin_t_from_cluster_start_s_z'])

    # Firefly features
    _add_group('ff_visible', ['log1p_num_ff_visible', 'num_ff_visible'])
    _add_group('ff_in_memory', ['log1p_num_ff_in_memory', 'num_ff_in_memory'])

    # Retry features
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

    # Timing
    _add_group('time_rel_to_event_start', ['time_rel_to_event_start'])

    return groups


def add_retries_info_to_binned_feats(binned_feats, new_seg_info, datasets, meta_df_used):
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

    meta_df_used = stop_design.join_event_tbl_avoid_collisions(
        meta_df_used, event_tbl
    )

    meta_df_used = meta_df_used.sort_values('bin').reset_index(drop=True)
    binned_feats = binned_feats.reset_index(drop=True)

    if len(meta_df_used) != len(binned_feats):
        raise ValueError('meta_df_used and binned_feats misaligned')

    for col in RETRY_CATEGORIES:
        binned_feats[col] = meta_df_used[col].to_numpy()

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

    return binned_feats, meta_df_used


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
        binned_feats[f'num_ff_{state}'] = k_ff[used_bins]
        binned_feats[f'log1p_num_ff_{state}'] = np.log1p(
            binned_feats[f'num_ff_{state}'])

    return binned_feats


# =============================================================================
# Subsetting helper
# =============================================================================


def subset_binned_data(binned_feats, binned_spikes, offset_log, meta_df_used, mask):
    return (
        binned_feats.loc[mask].reset_index(drop=True),
        binned_spikes.loc[mask].reset_index(drop=True),
        offset_log[mask],
        meta_df_used.loc[mask].reset_index(drop=True),
    )


def add_interaction_columns(binned_feats):
    # list of variables you want to interact with 'whether_in_retry_series'
    excluded_exact = {'intercept', 'const',
                      'miss', 'gap_till_next_event_in_cluster_z'}
    excluded_substrings = {'rsw', 'rcap', 'retry', ':bin'}

    vars_to_interact = [
        c for c in binned_feats.columns
        if c not in excluded_exact
        and not c.startswith('rcos_')
        and not any(s in c for s in excluded_substrings)
    ]

    retry = binned_feats['whether_in_retry_series']
    interaction_df = pd.DataFrame(
        {
            f'{c}*retry': binned_feats[c].to_numpy() * retry.to_numpy()
            for c in vars_to_interact
        },
        index=binned_feats.index,
    )
    # Concatenate once to avoid DataFrame fragmentation from repeated inserts.
    return pd.concat([binned_feats, interaction_df], axis=1)


def scale_binned_feats(binned_feats, keep_constant_tuning_terms: bool = False):
    # No z-scoring for:
    # - raised-cosine (rcos_*, rcos_stop_*)
    # - boxcar tuning (*:bin*)
    # - strictly binary columns (only 0 and 1)

    exclude_prefixes = ('rcos_', 'rcos_stop_')
    
    # --- Detect strictly binary 0/1 columns ---
    binary_cols = [
        c for c in binned_feats.columns
        if set(binned_feats[c].dropna().unique()).issubset({0, 1})
    ]

    binned_feats_sc, scaled_cols = event_binning.selective_zscore(
        binned_feats,
        exclude_prefixes=exclude_prefixes,
        exclude_substrings=(':bin', 'num'),
        exclude_columns=binary_cols,  # <- prevent scaling of 0/1 columns
    )

    binned_feats_sc = sm.add_constant(binned_feats_sc, has_constant='add')
    print('Scaled columns:', scaled_cols)

    # Drop all constant columns except the model intercept 'const'
    const_cols = [
        c for c in binned_feats_sc.columns
        if binned_feats_sc[c].nunique(dropna=False) <= 1 and c != 'const'
    ]
    print("Constant columns:", const_cols)

    binned_feats_sc = binned_feats_sc.drop(columns=const_cols)

    return binned_feats_sc

def _prepare_stop_design_inputs(raw_data_folder_path, bin_width):
    """
    Shared data loading + stop preprocessing for both
    decoding and encoding designs.
    """
    pn, datasets, _ = collect_stop_data.collect_stop_data_func(
        raw_data_folder_path,
        bin_width=bin_width,
    )

    captures_df, valid_captures_df, filtered_no_capture_stops_df, stops_with_stats = (
        get_stops_utils.prepare_no_capture_and_captures(
            monkey_information=pn.monkey_information,
            closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
            ff_caught_T_new=pn.ff_caught_T_new,
            distance_col='distance_from_ff_to_stop',
        )
    )

    # Add neighbor timing info
    stops_with_stats['stop_time'] = stops_with_stats['stop_id_start_time']
    stops_with_stats['prev_time'] = stops_with_stats['stop_id_end_time'].shift(
        1)
    stops_with_stats['next_time'] = stops_with_stats['stop_id_start_time'].shift(
        -1)

    new_seg_info = event_binning.make_new_seg_info_for_stop_design(
        stops_with_stats,
        pn.closest_stop_to_capture_df,
        pn.monkey_information,
    )

    events_with_stats = stops_with_stats[
        [
            'stop_id',
            'stop_cluster_id',
            'stop_id_start_time',
            'stop_id_end_time',
        ]
    ].rename(
        columns={
            'stop_id': 'event_id',
            'stop_cluster_id': 'event_cluster_id',
            'stop_id_start_time': 'event_id_start_time',
            'stop_id_end_time': 'event_id_end_time',
        }
    )

    return pn, datasets, new_seg_info, events_with_stats


# ============================================================
# Decoding
# ============================================================

def assemble_stop_decoding_design(
    raw_data_folder_path,
    bin_width,
    global_bins_2d=None,
    detrend_spikes: bool = True,
):
    """
    Assemble stop design for decoding.
    """

    pn, datasets, new_seg_info, events_with_stats = \
        _prepare_stop_design_inputs(raw_data_folder_path, bin_width)

    build_result = build_stop_design_decoding(
        new_seg_info=new_seg_info,
        events_with_stats=events_with_stats,
        monkey_information=pn.monkey_information,
        spikes_df=pn.spikes_df,
        ff_dataframe=pn.ff_dataframe,
        bin_dt=bin_width,
        datasets=datasets,
        add_ff_visible_info=True,
        global_bins_2d=global_bins_2d,
        detrend_spikes=detrend_spikes,
    )

    (
        rebinned_spike_rates,
        binned_feats,
        offset_log,
        meta_df_used,
    ) = build_result

    binned_feats = scale_binned_feats(
        binned_feats,
    )

    return (
        pn,
        rebinned_spike_rates,
        binned_feats,
        offset_log,
        meta_df_used,
    )
