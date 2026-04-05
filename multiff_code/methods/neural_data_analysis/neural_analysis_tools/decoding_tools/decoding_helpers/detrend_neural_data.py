import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    decoding_diagnostics,
)

def _estimate_slow_drift(spike_rate_hz, bin_size, drift_sigma_s):
    '''
    Estimate slow session drift with a Gaussian kernel over session time.
    '''
    if drift_sigma_s <= 0:
        raise ValueError('drift_sigma_s must be positive.')

    sigma_bins = drift_sigma_s / bin_size
    return gaussian_filter1d(spike_rate_hz, sigma=sigma_bins, mode='reflect')


def _estimate_time_regression_drift(spike_rate_hz, time_bin_center, regression_degree):
    '''
    Estimate slow session drift by regressing firing rate on session time.

    Parameters
    ----------
    spike_rate_hz : np.ndarray
        Binned firing rate for one cluster.
    time_bin_center : np.ndarray
        Session time for each bin center.
    regression_degree : int
        Polynomial degree for regression on session time.

    Returns
    -------
    drift_rate_hz : np.ndarray
        Regression-predicted firing rate across session time.
    '''
    if regression_degree < 1:
        raise ValueError('regression_degree must be >= 1.')

    if len(spike_rate_hz) != len(time_bin_center):
        raise ValueError('spike_rate_hz and time_bin_center must have the same length.')

    if len(spike_rate_hz) == 0:
        return np.array([], dtype=float)

    # Rescale time to improve numerical stability of polynomial fitting.
    time_mean = np.mean(time_bin_center)
    time_std = np.std(time_bin_center)

    if time_std == 0:
        return np.full_like(spike_rate_hz, fill_value=np.mean(spike_rate_hz), dtype=float)

    time_scaled = (time_bin_center - time_mean) / time_std
    poly_coeff = np.polyfit(time_scaled, spike_rate_hz, deg=regression_degree)
    drift_rate_hz = np.polyval(poly_coeff, time_scaled)

    return drift_rate_hz


def detrend_spikes_session_wide(
    fs_binned_spikes_hz,
    time_bins_df,
    bin_size,
    drift_sigma_s=60.0,
    center_method='subtract',
    drift_method='regression',
    regression_degree=2,
):
    '''
    Session-wide detrending from full-session binned spike rates (Hz).

    Parameters
    ----------
    fs_binned_spikes_hz : pd.DataFrame
        Rows = time bins, columns = cluster ids, values = firing rate (Hz).
        Build with ``smooth_neural_data.make_full_session_binned_spikes`` then divide
        by bin width, optionally ``drop_nonstationary_neurons`` on the rate matrix.
    time_bins_df : pd.DataFrame
        Must include ``time_bin_start`` and ``time_bin_end``; ``time_bin_center`` is
        used if present, otherwise computed as the midpoints.
    bin_size : float
        Bin width in seconds (used for spike counts from rates and for drift).
    drift_sigma_s : float
        Gaussian smoothing sigma in seconds, used when drift_method='gaussian'.
    center_method : str
        'subtract' or 'fractional'.
    drift_method : str
        'gaussian' or 'regression'.
    regression_degree : int
        Polynomial degree for session-time regression, used when
        drift_method='regression'.

    Returns
    -------
    detrended_df : pd.DataFrame
        Long-format dataframe with raw, drift, and detrended rates.
    '''
    empty_columns = [
        'time_bin_start',
        'time_bin_end',
        'time_bin_center',
        'cluster',
        'spike_count',
        'spike_rate_hz',
        'drift_rate_hz',
        'detrended_rate_hz',
    ]

    if fs_binned_spikes_hz is None or fs_binned_spikes_hz.empty or fs_binned_spikes_hz.shape[1] == 0:
        return pd.DataFrame(columns=empty_columns)

    required_time = {'time_bin_start', 'time_bin_end'}
    missing_time = required_time.difference(time_bins_df.columns)
    if missing_time:
        raise ValueError(f'time_bins_df is missing required columns: {missing_time}')

    time_bin_start = np.asarray(time_bins_df['time_bin_start'], dtype=float)
    time_bin_end = np.asarray(time_bins_df['time_bin_end'], dtype=float)
    if 'time_bin_center' in time_bins_df.columns:
        time_bin_center = np.asarray(time_bins_df['time_bin_center'], dtype=float)
    else:
        time_bin_center = 0.5 * (time_bin_start + time_bin_end)

    if len(time_bin_start) != len(fs_binned_spikes_hz):
        raise ValueError(
            'time_bins_df row count must match fs_binned_spikes_hz row count.'
        )

    all_cluster_rows = []

    for cluster_id in fs_binned_spikes_hz.columns:
        spike_rate_hz = np.asarray(fs_binned_spikes_hz[cluster_id], dtype=float)
        spike_count = spike_rate_hz * bin_size

        if drift_method == 'gaussian':
            drift_rate_hz = _estimate_slow_drift(
                spike_rate_hz=spike_rate_hz,
                bin_size=bin_size,
                drift_sigma_s=drift_sigma_s,
            )
        elif drift_method == 'regression':
            drift_rate_hz = _estimate_time_regression_drift(
                spike_rate_hz=spike_rate_hz,
                time_bin_center=time_bin_center,
                regression_degree=regression_degree,
            )
        else:
            raise ValueError("drift_method must be 'gaussian' or 'regression'.")

        if center_method == 'subtract':
            detrended_rate_hz = spike_rate_hz - drift_rate_hz
        elif center_method == 'fractional':
            small_value = 1e-6
            detrended_rate_hz = (spike_rate_hz - drift_rate_hz) / np.maximum(
                drift_rate_hz, small_value
            )
        else:
            raise ValueError("center_method must be 'subtract' or 'fractional'.")

        cluster_df = pd.DataFrame({
            'time_bin_start': time_bin_start,
            'time_bin_end': time_bin_end,
            'time_bin_center': time_bin_center,
            'cluster': cluster_id,
            'spike_count': spike_count,
            'spike_rate_hz': spike_rate_hz,
            'drift_rate_hz': drift_rate_hz,
            'detrended_rate_hz': detrended_rate_hz,
        })
        all_cluster_rows.append(cluster_df)

    return pd.concat(all_cluster_rows, axis=0, ignore_index=True)


def reshape_detrended_df_to_wide(
    detrended_df,
    value_col='detrended_rate_hz',
    cluster_col='cluster'
):
    '''
    Convert long-format detrended dataframe → wide (time × cluster),
    preserving time columns.

    Returns
    -------
    wide_df : pd.DataFrame
        DataFrame with time columns + one column per cluster
    cluster_columns : list
        Sorted list of cluster column names
    '''
    required_columns = {
        'time_bin_start',
        'time_bin_end',
        'time_bin_center',
        cluster_col,
        value_col
    }
    missing_columns = required_columns.difference(detrended_df.columns)
    if missing_columns:
        raise ValueError(f'detrended_df is missing required columns: {missing_columns}')

    time_columns = ['time_bin_start', 'time_bin_end', 'time_bin_center']

    wide_df = (
        detrended_df
        .pivot(
            index=time_columns,
            columns=cluster_col,
            values=value_col
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    cluster_columns = sorted(
        [col for col in wide_df.columns if col not in time_columns]
    )

    wide_df = wide_df[time_columns + cluster_columns].reset_index(drop=True)

    return wide_df, cluster_columns


def plot_subtraction_term(
    detrended_df,
    cluster_id,
    smooth_raw_sigma_bins=None,
    smooth_subtracted_sigma_bins=None,
    smooth_residual_sigma_bins=None
):
    '''
    Plot the raw rate, the term being subtracted, and the residual.

    Parameters
    ----------
    detrended_df : pd.DataFrame
        Output of detrend_spikes_session_wide.
    cluster_id : int or str
        Cluster to plot.
    smooth_raw_sigma_bins : float or None
        Optional Gaussian smoothing for raw rate before plotting.
    smooth_subtracted_sigma_bins : float or None
        Optional Gaussian smoothing for subtraction term before plotting.
    smooth_residual_sigma_bins : float or None
        Optional Gaussian smoothing for residual before plotting.
    '''
    cluster_df = (
        detrended_df[detrended_df['cluster'] == cluster_id]
        .sort_values('time_bin_center')
        .reset_index(drop=True)
    )

    if cluster_df.empty:
        raise ValueError(f'No rows found for cluster_id={cluster_id}.')

    time = cluster_df['time_bin_center'].to_numpy()
    raw_rate = cluster_df['spike_rate_hz'].to_numpy()
    subtraction_term = cluster_df['drift_rate_hz'].to_numpy()
    residual = cluster_df['detrended_rate_hz'].to_numpy()

    def _maybe_smooth(signal, sigma_bins):
        if sigma_bins is None or sigma_bins <= 0:
            return signal
        return gaussian_filter1d(signal, sigma=sigma_bins, mode='reflect')

    raw_rate_to_plot = _maybe_smooth(raw_rate, smooth_raw_sigma_bins)
    subtraction_term_to_plot = _maybe_smooth(subtraction_term, smooth_subtracted_sigma_bins)
    residual_to_plot = _maybe_smooth(residual, smooth_residual_sigma_bins)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time, raw_rate_to_plot, label='Raw rate')
    axes[0].set_ylabel('Rate (Hz)')
    axes[0].set_title(f'Cluster {cluster_id}: raw rate')
    axes[0].legend()

    axes[1].plot(time, subtraction_term_to_plot, label='Part being subtracted')
    axes[1].set_ylabel('Rate (Hz)')
    axes[1].set_title(f'Cluster {cluster_id}: subtraction term')
    axes[1].legend()

    axes[2].plot(time, residual_to_plot, label='Residual after subtraction')
    axes[2].axhline(0, linestyle='--')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Rate (Hz)')
    axes[2].set_title(f'Cluster {cluster_id}: raw rate - subtraction term')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def drop_nonstationary_neurons(binned_spikes):
    keep_mask, qc_df = decoding_diagnostics.detect_nonstationary_neurons_windowed(
        binned_spikes.values,
        smooth_window_bins=10,              # 0.4 s smoothing
        n_segments=5,
        min_mean_rate=0.001,
        max_firing_rate_stability=1.5,
        max_segment_stability=4,
        max_between_half_mean_shift=1.5,
        max_between_half_var_ratio=20
    )
    
    binned_spikes = binned_spikes.loc[:, keep_mask].copy()

    # if there are any dropped neurons, print the qc_df
    if qc_df['drop'].any():
        print('Dropped nonstationary neurons:')
        print(qc_df[['reason']])
    else:
        print('No nonstationary neurons dropped')
        
    return binned_spikes