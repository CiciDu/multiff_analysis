import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import detrend_neural_data
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import rebinned_alignment


def rebinned_x_var_to_binned_spike_rates_hz(
    rebinned_x_var: pd.DataFrame,
    bin_width: float,
    *,
    drop_bad_neurons: bool = True,
) -> pd.DataFrame:
    """
    Extract cluster_* columns from PN ``rebinned_x_var``, convert to Hz, optionally QC-drop units.
    """
    cluster_cols = [c for c in rebinned_x_var.columns if c.startswith('cluster_')]
    binned_spikes = rebinned_x_var[cluster_cols].copy()
    binned_spikes.columns = (
        binned_spikes.columns.str.replace('cluster_', '', regex=False).astype(int)
    )
    out = binned_spikes / bin_width
    if drop_bad_neurons:
        out = detrend_neural_data.drop_nonstationary_neurons(out)
    return out


def rebin_processed_spike_rates(processed_spike_rates, new_seg_info, bin_edges, rebinned_y_var):

    new_seg_for_rebin = new_seg_info.copy()
    if 'new_segment' not in new_seg_for_rebin.columns:
        new_seg_for_rebin['new_segment'] = np.arange(len(new_seg_for_rebin))
        
    merge_keys = rebinned_y_var[['new_segment']].copy()
    merge_keys['new_bin'] = np.arange(len(rebinned_y_var)) # since new_bin is assigned based on bin_edges, which are consistent with the bins used in rebinned_y_var

    # Align rebinned_spike_rates to rebinned_y_var row order
    rebinned_spike_rates = rebinned_alignment.rebin_then_align_spike_rates(
        processed_spike_rates,
        new_seg_for_rebin,
        bin_edges,
        merge_keys,
    )
        
    return rebinned_spike_rates
        


