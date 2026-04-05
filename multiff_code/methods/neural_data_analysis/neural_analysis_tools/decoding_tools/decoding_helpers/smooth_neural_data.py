import numpy as np
import pandas as pd
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import one_ff_style_utils


def make_full_session_binned_spikes(spikes_df, bin_width=0.005):
    """
    Build full-session binned spike counts (wide) and bin-edge metadata.

    Returns
    -------
    fs_binned_spikes : pd.DataFrame
        Rows = time bins, columns = sorted cluster ids, values = spike counts per bin.
    time_bins_df : pd.DataFrame
        Columns time_bin_start, time_bin_end, time_bin_center.
    """

    if spikes_df is None or spikes_df.empty:
        return pd.DataFrame(), pd.DataFrame(
            columns=["time_bin_start", "time_bin_end", "time_bin_center"]
        )

    min_time = 0
    max_time = spikes_df.time.max() + 60
    time_bins, all_binned_counts = neural_data_processing._make_all_binned_spikes(
        spikes_df,
        min_time=min_time,
        max_time=max_time,
        bin_width=bin_width,
    )

    all_clusters = np.sort(spikes_df["cluster"].unique()).astype(int)

    fs_binned_spikes = pd.DataFrame(all_binned_counts, columns=all_clusters)
    time_bins_df = pd.DataFrame(
        {
            "time_bin_start": time_bins[:-1],
            "time_bin_end": time_bins[1:],
        }
    )
    time_bins_df["time_bin_center"] = (
        time_bins_df["time_bin_start"] + time_bins_df["time_bin_end"]
    ) / 2.0

    return fs_binned_spikes, time_bins_df


def smooth_contiguous_spike_rates(fs_binned_spikes_hz, time_bins_df, width: int):
    """
    Gaussian smooth full-session spike rates in Hz (wide matrix).

    Parameters
    ----------
    fs_binned_spikes_hz : pd.DataFrame
        Rows = time bins, columns = cluster ids, values = firing rate (Hz).
    time_bins_df : pd.DataFrame
        Bin metadata (returned unchanged; used for alignment only).
    width : int
        Smoothing kernel width (passed to gaussian_kernel).
    """
    if fs_binned_spikes_hz is None or fs_binned_spikes_hz.empty:
        return (
            fs_binned_spikes_hz.copy()
            if fs_binned_spikes_hz is not None
            else pd.DataFrame(),
            time_bins_df.copy() if time_bins_df is not None else pd.DataFrame(),
        )

    width = int(width)
    if width < 0:
        raise ValueError(f"width must be >= 0, got {width}")
    smoothed = smooth_signal(np.asarray(fs_binned_spikes_hz, dtype=float), width)
    smoothed_df = fs_binned_spikes_hz.copy()
    smoothed_df.iloc[:, :] = smoothed

    return smoothed_df, time_bins_df

def smooth_signal(
    x: np.ndarray,
    width: int,
    trial_idx: np.ndarray | None = None
) -> np.ndarray:
    if width <= 0:
        return np.asarray(x)

    x = np.asarray(x, dtype=float)
    h = np.asarray(one_ff_style_utils.gaussian_kernel(int(width)), dtype=float)

    if h.ndim != 1:
        raise ValueError('gaussian_kernel(width) must return a 1D kernel.')

    if trial_idx is None:
        if x.ndim == 1:
            return _convolve_with_renormalization(x, h)
        elif x.ndim == 2:
            return _smooth_2d_with_renormalization(x, h)
        else:
            raise ValueError('x must be 1D or 2D.')

    trial_idx = np.asarray(trial_idx)
    if trial_idx.ndim != 1:
        raise ValueError('trial_idx must be 1D.')
    if len(trial_idx) != x.shape[0]:
        raise ValueError('trial_idx must have the same length as x along axis 0.')

    smoothed_x = np.empty_like(x, dtype=float)

    trial_starts = np.flatnonzero(np.r_[True, trial_idx[1:] != trial_idx[:-1]])
    trial_ends = np.r_[trial_starts[1:], len(trial_idx)]

    for start, end in zip(trial_starts, trial_ends):
        x_trial = x[start:end]

        if x.ndim == 1:
            smoothed_x[start:end] = _convolve_with_renormalization(x_trial, h)
        elif x.ndim == 2:
            smoothed_x[start:end] = _smooth_2d_with_renormalization(x_trial, h)
        else:
            raise ValueError('x must be 1D or 2D.')

    return smoothed_x


def _convolve_same_length(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    '''
    Convolve x with h and always return an output of length len(x),
    center-cropped from the full convolution.
    '''
    full = np.convolve(x, h, mode='full')
    start = (len(h) - 1) // 2
    end = start + len(x)
    return full[start:end]


def _convolve_with_renormalization(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    '''
    Convolve a 1D signal and renormalize near edges so only valid samples
    contribute to the denominator.
    '''
    valid_mask = np.ones_like(x, dtype=float)

    numerator = _convolve_same_length(x, h)
    denominator = _convolve_same_length(valid_mask, h)

    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)

    return out


def _smooth_2d_with_renormalization(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    '''
    Apply renormalized 1D smoothing along axis 0 for each column of a 2D array.
    '''
    smoothed_x = np.empty_like(x, dtype=float)

    for col_idx in range(x.shape[1]):
        smoothed_x[:, col_idx] = _convolve_with_renormalization(x[:, col_idx], h)

    return smoothed_x