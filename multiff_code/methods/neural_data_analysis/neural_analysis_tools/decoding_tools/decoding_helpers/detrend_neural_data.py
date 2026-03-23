import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def _make_time_edges(start_time, end_time, bin_size):
    '''
    Build session-wide bin edges.
    '''
    if not np.isfinite(start_time) or not np.isfinite(end_time):
        raise ValueError('start_time and end_time must be finite.')
    if bin_size <= 0:
        raise ValueError('bin_size must be positive.')
    if end_time <= start_time:
        raise ValueError('end_time must be greater than start_time.')

    n_bins = int(np.ceil((end_time - start_time) / bin_size))
    return start_time + np.arange(n_bins + 1) * bin_size


def _bin_spikes_for_one_cluster(cluster_spike_times, time_edges):
    '''
    Bin spike times for one cluster.
    '''
    spike_count, _ = np.histogram(cluster_spike_times, bins=time_edges)
    return spike_count.astype(float)


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
    spikes_df,
    bin_size=0.01,
    drift_sigma_s=60.0,
    min_time=None,
    max_time=None,
    center_method='subtract',
    drift_method='gaussian',
    regression_degree=2
):
    '''
    Session-wide detrending from raw spike times.

    Parameters
    ----------
    spikes_df : pd.DataFrame
        Must contain columns 'time' and 'cluster'.
    bin_size : float
        Bin size in seconds.
    drift_sigma_s : float
        Gaussian smoothing sigma in seconds, used when drift_method='gaussian'.
    min_time : float or None
        Session start time. If None, inferred from spikes_df.
    max_time : float or None
        Session end time. If None, inferred from spikes_df.
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
    required_columns = {'time', 'cluster'}
    missing_columns = required_columns.difference(spikes_df.columns)
    if missing_columns:
        raise ValueError(f'spikes_df is missing required columns: {missing_columns}')

    if spikes_df.empty:
        return pd.DataFrame(columns=[
            'time_bin_start',
            'time_bin_end',
            'time_bin_center',
            'cluster',
            'spike_count',
            'spike_rate_hz',
            'drift_rate_hz',
            'detrended_rate_hz'
        ])

    spikes_df = spikes_df[['time', 'cluster']].copy()

    if spikes_df['time'].isna().any() or spikes_df['cluster'].isna().any():
        raise ValueError("spikes_df contains NaNs in 'time' or 'cluster'.")

    if min_time is None:
        min_time = float(spikes_df['time'].min())
    if max_time is None:
        max_time = float(spikes_df['time'].max())

    time_edges = _make_time_edges(min_time, max_time, bin_size)
    time_bin_start = time_edges[:-1]
    time_bin_end = time_edges[1:]
    time_bin_center = 0.5 * (time_bin_start + time_bin_end)

    grouped_spikes = {
        cluster_id: cluster_df['time'].to_numpy()
        for cluster_id, cluster_df in spikes_df.groupby('cluster', sort=True)
    }

    cluster_ids = np.array(list(grouped_spikes.keys()))
    all_cluster_rows = []

    for cluster_id in cluster_ids:
        cluster_spike_times = grouped_spikes[cluster_id]

        spike_count = _bin_spikes_for_one_cluster(cluster_spike_times, time_edges)
        spike_rate_hz = spike_count / bin_size

        if drift_method == 'gaussian':
            drift_rate_hz = _estimate_slow_drift(
                spike_rate_hz=spike_rate_hz,
                bin_size=bin_size,
                drift_sigma_s=drift_sigma_s
            )
        elif drift_method == 'regression':
            drift_rate_hz = _estimate_time_regression_drift(
                spike_rate_hz=spike_rate_hz,
                time_bin_center=time_bin_center,
                regression_degree=regression_degree
            )
        else:
            raise ValueError("drift_method must be 'gaussian' or 'regression'.")

        if center_method == 'subtract':
            detrended_rate_hz = spike_rate_hz - drift_rate_hz
        elif center_method == 'fractional':
            small_value = 1e-6
            detrended_rate_hz = (spike_rate_hz - drift_rate_hz) / np.maximum(drift_rate_hz, small_value)
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
            'detrended_rate_hz': detrended_rate_hz
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