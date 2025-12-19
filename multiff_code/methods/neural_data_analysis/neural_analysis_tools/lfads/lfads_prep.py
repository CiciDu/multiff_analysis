import numpy as np
import pandas as pd
from typing import Dict, Tuple


def make_continuous_lfads_trials(
    spikes_df: pd.DataFrame,
    bin_width_ms: float = 10.0,
    window_len_s: float = 1.0,
    step_s: float = 0.5,
) -> Dict:
    """
    Convert continuous spikes into overlapping LFADS pseudo-trials.

    Parameters
    ----------
    spikes_df : DataFrame with columns ['time', 'cluster']
    bin_width_ms : float
        Bin width in ms (e.g. 10.0).
    window_len_s : float
        Length of each LFADS trial window in seconds (e.g. 1.0).
    step_s : float
        Step between successive windows in seconds (e.g. 0.5) for overlap.

    Returns
    -------
    dict with:
      - 'trials' : np.ndarray, shape (n_trials, T_bins, n_neurons), spike counts
      - 'trial_start_times' : np.ndarray, shape (n_trials,), in seconds
      - 'bin_width_ms' : float
      - 'clusters' : np.ndarray, sorted cluster IDs
      - 'start_time' : float, session start time (s)
    """

    if 'time' not in spikes_df.columns or 'cluster' not in spikes_df.columns:
        raise ValueError('spikes_df must contain columns "time" and "cluster".')

    spikes_df = spikes_df.sort_values('time').reset_index(drop=True)

    # Map clusters to contiguous indices 0..n-1
    clusters = np.array(sorted(spikes_df['cluster'].unique().tolist()))
    cluster_to_col = {c: i for i, c in enumerate(clusters)}
    n_neurons = len(clusters)

    spike_times = spikes_df['time'].to_numpy(float)
    spike_codes = np.fromiter(
        (cluster_to_col[c] for c in spikes_df['cluster'].to_numpy()),
        count=len(spikes_df),
        dtype=np.int32,
    )

    bin_width_s = bin_width_ms / 1000.0
    start_time = float(spike_times.min())
    end_time = float(spike_times.max())
    total_dur = end_time - start_time

    n_bins = int(np.ceil(total_dur / bin_width_s))
    # global bin index for each spike
    bin_idx = ((spike_times - start_time) / bin_width_s).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    # Build full binned count matrix (continuous)
    counts = np.zeros((n_bins, n_neurons), dtype=np.float32)
    np.add.at(counts, (bin_idx[valid], spike_codes[valid]), 1.0)

    # Windowing in bin space
    window_len_bins = int(round(window_len_s / bin_width_s))
    step_bins = int(round(step_s / bin_width_s))

    if window_len_bins <= 0:
        raise ValueError('window_len_s must be > 0.')

    trial_start_bins = np.arange(0, n_bins - window_len_bins + 1, step_bins)
    n_trials = len(trial_start_bins)

    trials = np.zeros((n_trials, window_len_bins, n_neurons), dtype=np.float32)
    trial_start_times = start_time + trial_start_bins * bin_width_s

    for i, b0 in enumerate(trial_start_bins):
        b1 = b0 + window_len_bins
        trials[i] = counts[b0:b1]

    return {
        'trials': trials,
        'trial_start_times': trial_start_times,
        'bin_width_ms': float(bin_width_ms),
        'clusters': clusters,
        'start_time': float(start_time),
    }


def stitch_lfads_rates_to_continuous(
    lfads_rates: np.ndarray,
    trial_start_times: np.ndarray,
    bin_width_ms: float,
    start_time: float,
    n_neurons: int,
) -> np.ndarray:
    """
    Stitch LFADS trial-wise rates back into a continuous FR matrix using overlap-averaging.

    Parameters
    ----------
    lfads_rates : (n_trials, T_bins, n_neurons) predicted firing rates (Hz)
    trial_start_times : (n_trials,) start time of each trial (s)
    bin_width_ms : float
    start_time : float
    n_neurons : int

    Returns
    -------
    fr_continuous : (T_full, n_neurons) continuous FR (Hz)
    """

    bin_width_s = bin_width_ms / 1000.0
    n_trials, T_bins, _ = lfads_rates.shape

    # Determine global length in bins from last trial
    last_trial_end_time = trial_start_times[-1] + T_bins * bin_width_s
    total_dur = last_trial_end_time - start_time
    T_full = int(np.ceil(total_dur / bin_width_s))

    fr_sum = np.zeros((T_full, n_neurons), dtype=np.float32)
    fr_count = np.zeros((T_full, n_neurons), dtype=np.float32)

    for i in range(n_trials):
        t0 = trial_start_times[i]
        start_bin = int(round((t0 - start_time) / bin_width_s))
        end_bin = start_bin + T_bins
        if start_bin < 0:
            continue
        if end_bin > T_full:
            end_bin = T_full
            T_bins_eff = end_bin - start_bin
            fr_sum[start_bin:end_bin] += lfads_rates[i, :T_bins_eff]
            fr_count[start_bin:end_bin] += 1.0
        else:
            fr_sum[start_bin:end_bin] += lfads_rates[i]
            fr_count[start_bin:end_bin] += 1.0

    fr_count[fr_count == 0] = 1.0
    fr_continuous = fr_sum / fr_count
    return fr_continuous
