import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def plot_neurons_over_time(X, neuron_indices):
    """
    Plot selected neuron features over time.
    
    Args:
        X: np.ndarray (n_timepoints, n_features)
        neuron_indices: list of feature indices to plot
    """
    time = np.arange(X.shape[0])
    
    plt.figure(figsize=(10, 5))
    
    for idx in neuron_indices:
        plt.plot(time, X[:, idx], label=f'neuron_{idx}', alpha=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('Activity (z-scored)')
    plt.title('Neuron Activity Over Time')
    plt.legend()
    plt.show()

import numpy as np

def find_high_corr_pairs(X_tr, threshold=0.7, max_print=20):
    """
    Find and print highly correlated feature pairs.

    Parameters
    ----------
    X_tr : np.ndarray
        Input data (samples × features)
    threshold : float
        Absolute correlation threshold
    max_print : int
        Number of pairs to print

    Returns
    -------
    high_corr_pairs : list of tuples
        (i, j, corr_value) for pairs exceeding threshold
    """
    # Compute correlation matrix
    corr = np.corrcoef(X_tr, rowvar=False)

    high_corr_pairs = []
    n_features = corr.shape[0]

    # Iterate over upper triangle
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr[i, j]))

    # Print results
    print('Highly correlated pairs:')
    for pair in high_corr_pairs[:max_print]:
        print(pair)

    return high_corr_pairs


def compute_vif_matrix(X_tr):
    n_features = X_tr.shape[1]
    vif_values = []

    for j in range(n_features):
        x_target = X_tr[:, j]
        x_other = np.delete(X_tr, j, axis=1)

        model = LinearRegression()
        model.fit(x_other, x_target)
        r2_j = model.score(x_other, x_target)

        if 1 - r2_j < 1e-12:
            vif = np.inf
        else:
            vif = 1 / (1 - r2_j)

        vif_values.append(vif)

    return pd.DataFrame({
        'feature_idx': np.arange(n_features),
        'vif': vif_values,
    }).sort_values('vif', ascending=False)
    
    
import numpy as np

def analyze_singular_values(X_tr, print_smallest_k=10):
    """
    Compute SVD and print basic diagnostics.

    Parameters
    ----------
    X_tr : np.ndarray
        Input data (samples × features)
    print_smallest_k : int
        Number of smallest singular values to print

    Returns
    -------
    u, s, vt : np.ndarray
        SVD components
    """
    u, s, vt = np.linalg.svd(X_tr, full_matrices=False)

    print('Largest singular value:', s[0])
    print('Smallest singular value:', s[-1])
    print('Condition number:', s[0] / s[-1])
    print(f'Smallest {print_smallest_k} singular values:', s[-print_smallest_k:])

    return u, s, vt

import numpy as np
import pandas as pd


def _moving_average_1d(x, window_size):
    '''
    Smooth a 1D array with a centered moving average.
    '''
    x = np.asarray(x, dtype=float)

    if window_size is None or window_size <= 1:
        return x.copy()

    kernel = np.ones(int(window_size), dtype=float) / float(window_size)
    x_smooth = np.convolve(x, kernel, mode='same')

    return x_smooth


def _compute_firing_rate_stability(x, n_segments=5, eps=1e-12):
    '''
    Compute firing-rate stability as the coefficient of variation
    of segment means across the session.
    '''
    seg_list = np.array_split(x, n_segments)
    seg_means = np.array([np.mean(seg) for seg in seg_list], dtype=float)

    firing_rate_stability = np.std(seg_means) / (np.abs(np.mean(seg_means)) + eps)

    return firing_rate_stability, seg_means


def _compute_segment_stability(seg_means, eps=1e-12):
    '''
    Compute segment stability as the normalized range of segment means.
    '''
    seg_means = np.asarray(seg_means, dtype=float)

    segment_stability = (
        np.max(seg_means) - np.min(seg_means)
    ) / (np.abs(np.mean(seg_means)) + eps)

    return segment_stability


def _compute_between_half_shift_metrics(x, eps=1e-12):
    '''
    Compute first-vs-second half shift metrics.
    '''
    x = np.asarray(x, dtype=float)

    n_time = len(x)
    half = n_time // 2

    if half < 2:
        return {
            'first_half_mean': np.nan,
            'second_half_mean': np.nan,
            'between_half_mean_shift': np.nan,
            'between_half_var_ratio': np.nan
        }

    x_first = x[:half]
    x_second = x[n_time - half:]

    first_half_mean = np.mean(x_first)
    second_half_mean = np.mean(x_second)
    full_mean = np.mean(x)

    between_half_mean_shift = (
        np.abs(second_half_mean - first_half_mean) /
        (np.abs(full_mean) + eps)
    )

    first_half_var = np.var(x_first)
    second_half_var = np.var(x_second)

    between_half_var_ratio = (
        max(first_half_var, second_half_var) /
        (min(first_half_var, second_half_var) + eps)
    )

    return {
        'first_half_mean': first_half_mean,
        'second_half_mean': second_half_mean,
        'between_half_mean_shift': between_half_mean_shift,
        'between_half_var_ratio': between_half_var_ratio
    }


def _compute_nonstationarity_metrics_for_neuron(
    x,
    smooth_window_bins=10,
    n_segments=5
):
    '''
    Compute non-stationarity metrics for one neuron.
    '''
    x = np.asarray(x, dtype=float)

    finite_mask = np.isfinite(x)
    x = x[finite_mask]

    if len(x) == 0:
        return {
            'mean_rate': np.nan,
            'mean_rate_smooth': np.nan,
            'firing_rate_stability': np.nan,
            'segment_stability': np.nan,
            'first_half_mean': np.nan,
            'second_half_mean': np.nan,
            'between_half_mean_shift': np.nan,
            'between_half_var_ratio': np.nan
        }

    x_smooth = _moving_average_1d(x, smooth_window_bins)

    mean_rate = np.mean(x)
    mean_rate_smooth = np.mean(x_smooth)

    firing_rate_stability, seg_means = _compute_firing_rate_stability(
        x_smooth,
        n_segments=n_segments
    )

    segment_stability = _compute_segment_stability(seg_means)

    between_half_metrics = _compute_between_half_shift_metrics(x_smooth)

    return {
        'mean_rate': mean_rate,
        'mean_rate_smooth': mean_rate_smooth,
        'firing_rate_stability': firing_rate_stability,
        'segment_stability': segment_stability,
        **between_half_metrics
    }


def detect_nonstationary_neurons_windowed(
    X_tr,
    smooth_window_bins=10,
    n_segments=5,
    min_mean_rate=0.01,
    max_firing_rate_stability=0.8,
    max_segment_stability=1.5,
    max_between_half_mean_shift=1.0,
    max_between_half_var_ratio=3.0
):
    '''
    Detect non-stationary neurons using:
    - firing-rate stability
    - segment stability
    - between-half mean shift
    - between-half variance shift

    Important
    ---------
    Run this on raw spike counts or firing rates, not z-scored data.

    Parameters
    ----------
    X_tr : np.ndarray
        Array of shape (n_time_bins, n_neurons)
    smooth_window_bins : int
        Moving-average smoothing window in bins
    n_segments : int
        Number of session segments
    min_mean_rate : float
        Drop neurons with mean rate below this threshold
    max_firing_rate_stability : float
        Drop if firing-rate stability exceeds this threshold
    max_segment_stability : float
        Drop if segment stability exceeds this threshold
    max_between_half_mean_shift : float
        Drop if normalized first-vs-second-half mean shift exceeds this threshold
    max_between_half_var_ratio : float
        Drop if first-vs-second-half variance ratio exceeds this threshold

    Returns
    -------
    keep_mask : np.ndarray
        Boolean array of shape (n_neurons,), True means keep
    qc_df : pd.DataFrame
        Per-neuron QC table
    '''
    X_tr = np.asarray(X_tr, dtype=float)
    n_time_bins, n_neurons = X_tr.shape

    metrics_list = []

    for neuron_idx in range(n_neurons):
        x = X_tr[:, neuron_idx]

        metrics = _compute_nonstationarity_metrics_for_neuron(
            x,
            smooth_window_bins=smooth_window_bins,
            n_segments=n_segments
        )

        reasons = []

        if np.isnan(metrics['mean_rate']) or metrics['mean_rate'] < min_mean_rate:
            reasons.append('low_mean_rate')

        if (
            np.isnan(metrics['firing_rate_stability']) or
            metrics['firing_rate_stability'] > max_firing_rate_stability
        ):
            reasons.append('poor_firing_rate_stability')

        if (
            np.isnan(metrics['segment_stability']) or
            metrics['segment_stability'] > max_segment_stability
        ):
            reasons.append('poor_segment_stability')

        if (
            np.isnan(metrics['between_half_mean_shift']) or
            metrics['between_half_mean_shift'] > max_between_half_mean_shift
        ):
            reasons.append('between_half_mean_shift')

        if (
            np.isnan(metrics['between_half_var_ratio']) or
            metrics['between_half_var_ratio'] > max_between_half_var_ratio
        ):
            reasons.append('between_half_variance_shift')

        metrics['neuron_idx'] = neuron_idx
        metrics['drop'] = len(reasons) > 0
        metrics['reason'] = ','.join(reasons)

        metrics_list.append(metrics)

    qc_df = pd.DataFrame(metrics_list).sort_values('neuron_idx').reset_index(drop=True)
    keep_mask = ~qc_df['drop'].to_numpy()

    return keep_mask, qc_df