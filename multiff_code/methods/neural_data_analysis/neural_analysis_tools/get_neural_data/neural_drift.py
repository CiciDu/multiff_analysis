import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

plt.rcParams.update({'font.size': 10})


# ============================================================
# Binning
# ============================================================

def bin_spikes_from_df(
    spikes_df,
    bin_size=1.0,
    session_start=None,
    session_end=None
):
    '''
    Convert long-form spikes DataFrame into fixed-width time bins.

    Parameters
    ----------
    spikes_df : pd.DataFrame
        Columns:
          - 'time' : spike time (s)
          - 'cluster' : unit id
    bin_size : float
        Bin width in seconds
    session_start, session_end : float or None
        Session bounds (s); inferred if None

    Returns
    -------
    rate_mat : np.ndarray
        Shape (n_units, n_bins), firing rate in Hz
    bin_centers : np.ndarray
        Shape (n_bins,), bin center times
    unit_ids : np.ndarray
        Cluster IDs corresponding to rows of rate_mat
    '''
    assert {'time', 'cluster'}.issubset(spikes_df.columns)

    spikes_df = spikes_df.sort_values('time')

    if session_start is None:
        session_start = spikes_df['time'].min()
    if session_end is None:
        session_end = spikes_df['time'].max()

    bins = np.arange(session_start, session_end + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2.0

    unit_ids = np.sort(spikes_df['cluster'].unique())
    unit_to_row = {u: i for i, u in enumerate(unit_ids)}

    rate_mat = np.zeros((len(unit_ids), len(bin_centers)), dtype=float)

    # assign each spike to a bin
    bin_idx = np.floor((spikes_df['time'].values - session_start) / bin_size).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < len(bin_centers))

    for u, b in zip(spikes_df.loc[valid, 'cluster'], bin_idx[valid]):
        rate_mat[unit_to_row[u], b] += 1.0

    rate_mat /= bin_size  # counts → Hz

    return rate_mat, bin_centers, unit_ids


# ============================================================
# Per-unit drift tests
# ============================================================

def linear_trend_test(rate_mat, bin_centers):
    '''
    Linear regression of firing rate vs session time.
    '''
    rows = []
    for u in range(rate_mat.shape[0]):
        y = rate_mat[u]
        slope, intercept, r, p, stderr = stats.linregress(bin_centers, y)
        rows.append({
            'unit': u,
            'slope_hz_per_s': slope,
            'r': r,
            'pval': p,
            'stderr': stderr,
            'mean_rate': np.mean(y),
            'std_rate': np.std(y)
        })
    return pd.DataFrame(rows).set_index('unit')


def early_late_ks_test(rate_mat, frac=0.25):
    '''
    KS test comparing early vs late session firing distributions.
    '''
    n_bins = rate_mat.shape[1]
    k = int(n_bins * frac)

    rows = []
    for u in range(rate_mat.shape[0]):
        early = rate_mat[u, :k]
        late = rate_mat[u, -k:]
        ks, p = stats.ks_2samp(early, late)
        rows.append({
            'unit': u,
            'ks_stat': ks,
            'ks_pval': p
        })
    return pd.DataFrame(rows).set_index('unit')


def cusum_test(rate_mat, n_perms=200, rng_seed=0):
    '''
    CUSUM test for abrupt changes in mean firing rate.
    Permutation-based p-value.
    '''
    rng = np.random.default_rng(rng_seed)
    rows = []

    for u in range(rate_mat.shape[0]):
        y = rate_mat[u] - np.mean(rate_mat[u])
        cusum = np.cumsum(y)
        max_dev = np.max(np.abs(cusum))

        null = []
        for _ in range(n_perms):
            y_shuf = rng.permutation(y)
            null.append(np.max(np.abs(np.cumsum(y_shuf))))

        pval = (np.sum(np.array(null) >= max_dev) + 1) / (n_perms + 1)

        rows.append({
            'unit': u,
            'cusum_max': max_dev,
            'cusum_pval': pval
        })

    return pd.DataFrame(rows).set_index('unit')


# ============================================================
# Population-level drift tests
# ============================================================

def pca_time_drift(rate_mat, bin_centers, n_pcs=5):
    '''
    PCA across units; test how much each PC is explained by time.
    '''
    X = rate_mat.T  # bins x units
    pca = PCA(n_components=min(n_pcs, X.shape[1]))
    pcs = pca.fit_transform(X)

    rows = []
    for i in range(pcs.shape[1]):
        lr = LinearRegression().fit(bin_centers.reshape(-1, 1), pcs[:, i])
        r2 = lr.score(bin_centers.reshape(-1, 1), pcs[:, i])
        rows.append({
            'pc': i,
            'variance_explained': pca.explained_variance_ratio_[i],
            'time_r2': r2
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting helpers
# ============================================================

def plot_unit_drift(rate_mat, bin_centers, unit, ax=None):
    '''
    Plot firing rate vs time for a single unit with linear trend.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    y = rate_mat[unit]
    slope, intercept, *_ = stats.linregress(bin_centers, y)

    ax.plot(bin_centers, y, alpha=0.5)
    ax.plot(bin_centers, intercept + slope * bin_centers, lw=2)
    ax.set_title(f'Unit {unit} | slope={slope:.2e} Hz/s')
    ax.set_xlabel('session time (s)')
    ax.set_ylabel('rate (Hz)')

    return ax


def plot_population_heatmap(rate_mat, ax=None):
    '''
    Heatmap of firing rates (units x time bins).
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(rate_mat, aspect='auto', interpolation='nearest')
    ax.set_xlabel('time bin')
    ax.set_ylabel('unit')
    ax.set_title('Firing rates (time-binned)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    return ax


# ============================================================
# High-level convenience wrapper
# ============================================================

def compute_drift_summary(
    spikes_df,
    bin_size=1.0,
    alpha=0.01,
    n_cusum_perms=200
):
    '''
    End-to-end drift diagnostics from spikes_df.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-unit drift metrics + boolean flags
    pc_df : pd.DataFrame
        PCA drift metrics
    rate_mat : np.ndarray
        Binned firing rates
    bin_centers : np.ndarray
        Time axis
    unit_ids : np.ndarray
        Cluster IDs
    '''
    rate_mat, bin_centers, unit_ids = bin_spikes_from_df(
        spikes_df,
        bin_size=bin_size
    )

    lin_df = linear_trend_test(rate_mat, bin_centers)
    ks_df = early_late_ks_test(rate_mat)
    cs_df = cusum_test(rate_mat, n_perms=n_cusum_perms)
    pc_df = pca_time_drift(rate_mat, bin_centers)

    summary_df = (
        lin_df
        .join(ks_df)
        .join(cs_df)
    )

    summary_df['linear_sig'] = summary_df['pval'] < alpha
    summary_df['ks_sig'] = summary_df['ks_pval'] < alpha
    summary_df['cusum_sig'] = summary_df['cusum_pval'] < alpha

    summary_df['drift_flag'] = (
        summary_df['linear_sig'] |
        summary_df['ks_sig'] |
        summary_df['cusum_sig']
    )

    summary_df['cluster'] = unit_ids
    summary_df = summary_df.set_index('cluster')

    return summary_df, pc_df, rate_mat, bin_centers


import numpy as np
from sklearn.linear_model import LinearRegression


def detrend_features_cv(
    X_train,
    X_test,
    time_train,
    time_test,
    degree=1
):
    '''
    CV-safe detrending of neural features.

    Fits a per-feature polynomial trend on TRAINING data only,
    then subtracts the predicted trend from both train and test.

    Parameters
    ----------
    X_train : np.ndarray
        Shape (n_train_samples, n_features)
    X_test : np.ndarray
        Shape (n_test_samples, n_features)
    time_train : np.ndarray
        Shape (n_train_samples,), session time (e.g. trial index or seconds)
    time_test : np.ndarray
        Shape (n_test_samples,), session time
    degree : int
        Polynomial degree for detrending (default=1, linear)

    Returns
    -------
    X_train_dt : np.ndarray
        Detrended training features
    X_test_dt : np.ndarray
        Detrended test features
    '''
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    time_train = np.asarray(time_train).reshape(-1, 1)
    time_test = np.asarray(time_test).reshape(-1, 1)

    # build polynomial time design matrix
    def make_time_design(t):
        return np.hstack([t**d for d in range(1, degree + 1)])

    T_train = make_time_design(time_train)
    T_test = make_time_design(time_test)

    X_train_dt = np.zeros_like(X_train, dtype=float)
    X_test_dt = np.zeros_like(X_test, dtype=float)

    # detrend each feature independently
    for j in range(X_train.shape[1]):
        y_train = X_train[:, j]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(T_train, y_train)

        trend_train = lr.predict(T_train)
        trend_test = lr.predict(T_test)

        X_train_dt[:, j] = y_train - trend_train
        X_test_dt[:, j] = X_test[:, j] - trend_test

    return X_train_dt, X_test_dt


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def visualize_detrend_single_feature(x, t, degree=1, feature_name='feature'):
    '''
    x : (n_samples,) feature values (e.g., firing rate of one neuron)
    t : (n_samples,) time (bin centers or trial times)
    '''
    t = t.reshape(-1, 1)

    # build polynomial time design
    T = np.hstack([t**d for d in range(1, degree + 1)])

    lr = LinearRegression()
    lr.fit(T, x)
    trend = lr.predict(T)
    residual = x - trend

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(t, x, alpha=0.6)
    ax[0].set_title(f'Original {feature_name}')

    ax[1].plot(t, trend, color='red')
    ax[1].set_title('Fitted trend')

    ax[2].plot(t, residual, alpha=0.6)
    ax[2].axhline(0, color='k', lw=1)
    ax[2].set_title('Residual (after detrending)')
    ax[2].set_xlabel('time')

    plt.tight_layout()
    plt.show()


def compare_population_means(X, t):
    '''
    X : (n_samples, n_features)
    '''
    mean_rate = X.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, mean_rate, label='before detrend')
    ax.set_xlabel('time')
    ax.set_ylabel('population mean')
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_population_means(X, t):
    '''
    X : (n_samples, n_features)
    '''
    mean_rate = X.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, mean_rate, label='before detrend')
    ax.set_xlabel('time')
    ax.set_ylabel('population mean')
    ax.legend()
    plt.tight_layout()
    plt.show()

def variance_removed_by_detrending(X, X_dt):
    var_before = np.var(X, axis=0)
    var_after = np.var(X_dt, axis=0)
    frac_removed = 1 - (var_after / var_before)
    return frac_removed


import pandas as pd
import matplotlib.pyplot as plt

def plot_population_mean_smoothed(X, t, window=60):
    '''
    window: smoothing window in bins (e.g. 60 = 1 min for 1s bins)
    '''
    pop_mean = X.mean(axis=1)
    pop_mean_smooth = (
        pd.Series(pop_mean)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, pop_mean_smooth, lw=2)
    ax.set_xlabel('time')
    ax.set_ylabel('population mean (smoothed)')
    ax.set_title(f'Population mean (rolling window = {window}s)')
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from scipy.stats import linregress


def summarize_linear_drift(X, t, feature_names=None):
    '''
    Summarize linear drift per feature.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    t : np.ndarray
        Shape (n_samples,), time in seconds (bin centers or trial times)
    feature_names : list or None
        Optional feature labels (e.g., cluster IDs)

    Returns
    -------
    df : pd.DataFrame
        Columns:
          - slope_hz_per_s
          - delta_hz_session
          - r
          - pval
    '''
    t = np.asarray(t)
    session_dur = t.max() - t.min()

    rows = []
    for j in range(X.shape[1]):
        y = X[:, j]
        slope, intercept, r, p, stderr = linregress(t, y)

        rows.append({
            'feature': j if feature_names is None else feature_names[j],
            'slope_hz_per_s': slope,
            'delta_hz_session': slope * session_dur,
            'r': r,
            'pval': p
        })

    df = pd.DataFrame(rows).set_index('feature')
    return df

def print_drift_summary(drift_df, hz_threshold=2.0):
    strong = drift_df[np.abs(drift_df['delta_hz_session']) > hz_threshold]

    print(f'Total features: {len(drift_df)}')
    print(f'Features drifting > {hz_threshold:.1f} Hz over session: {len(strong)}')

    if len(strong) > 0:
        print('\nTop drifting features:')
        print(
            strong
            .sort_values('delta_hz_session', key=np.abs, ascending=False)
            .head(10)
            [['slope_hz_per_s', 'delta_hz_session', 'r']]
        )

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def plot_pca_time_r2_before_after(
    X,
    X_dt,
    t,
    n_pcs=7
):
    '''
    Plot PCA time R² before vs after detrending (side-by-side).
    '''
    def compute_r2(X):
        pca = PCA(n_components=min(n_pcs, X.shape[1]))
        pcs = pca.fit_transform(X)
        r2 = []
        for i in range(pcs.shape[1]):
            lr = LinearRegression().fit(t.reshape(-1, 1), pcs[:, i])
            r2.append(lr.score(t.reshape(-1, 1), pcs[:, i]))
        return np.array(r2)

    r2_before = compute_r2(X)
    r2_after = compute_r2(X_dt)

    x = np.arange(len(r2_before))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(
        x - width / 2,
        r2_before,
        width,
        label='before detrend'
    )
    ax.bar(
        x + width / 2,
        r2_after,
        width,
        label='after detrend'
    )

    ax.set_xlabel('PC index')
    ax.set_ylabel('R² vs time')
    ax.set_title('PCA time alignment (before vs after detrend)')
    ax.legend()

    # LOG SCALE is crucial here
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 1)

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


def selective_detrend_features_cv(
    X_train,
    X_test,
    time_train,
    time_test,
    degree=1,
    delta_hz_threshold=2.0,
    feature_names=None
):
    '''
    Selectively detrend features whose estimated total drift over the session
    exceeds a threshold (Hz), using TRAINING data only.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Shapes (n_train, n_features), (n_test, n_features)
    time_train, time_test : np.ndarray
        Shapes (n_train,), (n_test,)
    degree : int
        Polynomial degree for detrending (1 = linear)
    delta_hz_threshold : float
        Threshold on |slope| * session_duration (Hz)
    feature_names : list or None
        Optional labels for features (e.g., cluster ids)

    Returns
    -------
    X_train_out, X_test_out : np.ndarray
        Detrended outputs (only subset of features altered)
    detrended_mask : np.ndarray (bool)
        Length n_features; True where detrending applied
    drift_df_train : pd.DataFrame
        Per-feature slope + delta estimates from training data
    '''
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    t_tr = np.asarray(time_train).astype(float)
    t_te = np.asarray(time_test).astype(float)

    # center time for numerical stability (important if degree > 1)
    t0 = np.mean(t_tr)
    t_tr_c = (t_tr - t0).reshape(-1, 1)
    t_te_c = (t_te - t0).reshape(-1, 1)

    def make_time_design(t_col):
        return np.hstack([t_col**d for d in range(1, degree + 1)])

    T_train = make_time_design(t_tr_c)
    T_test = make_time_design(t_te_c)

    session_dur = t_tr.max() - t_tr.min()

    # compute drift slopes on training data only (linear slope for thresholding)
    rows = []
    slopes = np.zeros(X_train.shape[1], dtype=float)
    for j in range(X_train.shape[1]):
        slope, intercept, r, p, stderr = linregress(t_tr, X_train[:, j])
        slopes[j] = slope
        rows.append({
            'feature': j if feature_names is None else feature_names[j],
            'slope_hz_per_s': slope,
            'delta_hz_session': slope * session_dur,
            'r': r,
            'pval': p
        })
    drift_df_train = pd.DataFrame(rows).set_index('feature')

    detrended_mask = np.abs(drift_df_train['delta_hz_session'].values) > delta_hz_threshold

    X_train_out = X_train.copy().astype(float)
    X_test_out = X_test.copy().astype(float)

    # fit and subtract trend only for selected features
    for j in np.where(detrended_mask)[0]:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(T_train, X_train[:, j])

        X_train_out[:, j] = X_train[:, j] - lr.predict(T_train)
        X_test_out[:, j] = X_test[:, j] - lr.predict(T_test)


    n_total = detrended_mask.size
    n_dt = detrended_mask.sum()
    print(f'Detrended {n_dt} / {n_total} neurons ({100 * n_dt / n_total:.1f}%)')


    return X_train_out, X_test_out, detrended_mask, drift_df_train
