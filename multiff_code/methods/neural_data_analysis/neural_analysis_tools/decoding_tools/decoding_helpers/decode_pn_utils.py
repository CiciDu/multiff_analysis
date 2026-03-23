import numpy as np
import pandas as pd
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import detrend_neural_data
from neural_data_analysis.design_kits.design_by_segment import rebin_segments


def get_rebinned_spike_rates(pn):
    # Detrend + rebin neural data (same approach as decode_stops_design)
    detrended_df = detrend_neural_data.detrend_spikes_session_wide(
        spikes_df=pn.spikes_df,
        bin_size=0.05,
        drift_sigma_s=60.0,
        center_method='subtract',
    )
    detrended_spike_rates, cluster_columns = detrend_neural_data.reshape_detrended_df_to_wide(
        detrended_df,
        value_col='detrended_rate_hz',
    )

    new_seg_for_rebin = pn.new_seg_info.copy()
    if 'new_segment' not in new_seg_for_rebin.columns:
        new_seg_for_rebin['new_segment'] = np.arange(len(new_seg_for_rebin))

    rebinned_spike_rates = rebin_segments.rebin_all_segments_global_bins(
        detrended_spike_rates,
        new_seg_for_rebin,
        bins_2d=pn.bin_edges,
        bin_left_col='time_bin_start',
        bin_right_col='time_bin_end',
        bin_center_col='time_bin_center',
        how='mean',
        respect_old_segment=False,
        require_full_bin=False,
        add_bin_edges=False,
        add_support_duration=False,
    )

    # Align rebinned_spike_rates to rebinned_y_var row order
    merge_keys = pn.rebinned_y_var[['new_segment', 'new_bin']].copy()
    rebinned_aligned = rebinned_spike_rates.merge(
        merge_keys,
        on=['new_segment', 'new_bin'],
        how='right',
    )
    id_cols = {'segment', 'bin','new_segment', 'new_bin', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration'}
    cluster_cols = [c for c in rebinned_aligned.columns if c not in id_cols]
    binned_spikes = rebinned_aligned[cluster_cols].reset_index(drop=True)
    binned_spikes.columns = binned_spikes.columns.astype(int)

    # Forward-fill NaN (bins with no neural overlap, e.g. edge bins past session)
    if binned_spikes.isna().any().any():
        binned_spikes = binned_spikes.ffill().bfill()
        
    return binned_spikes
