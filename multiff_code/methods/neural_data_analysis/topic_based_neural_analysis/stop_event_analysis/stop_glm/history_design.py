import numpy as np

def make_history_lags(y_unit, n_lags=5):
    """
    y_unit: (T,) spike counts for ONE unit, aligned to your design rows
    returns H: (T, n_lags) with columns [y[t-1], y[t-2], ..., y[t-n_lags]]
    """
    y = np.asarray(y_unit, float).ravel()
    T = y.shape[0]
    H = np.zeros((T, n_lags), float)
    for k in range(1, n_lags + 1):
        H[k:, k-1] = y[:-k]
    return H, [f'hist_lag{k}' for k in range(1, n_lags + 1)]


# y = spike_counts[pos, :]   # you already have this (rows==design rows)
unit_ix = 0
H, H_names = make_history_lags(y[:, unit_ix], n_lags=6)

# optional: gate history differently pre vs post
# (m_prepost was in the stop design helper; 0=pre, 1=post)
# H_post = H * m_prepost.to_numpy()[:, None]
# names_post = [n + '_post' for n in H_names]

# X_unit = np.column_stack([X_common, H])   # then fit GLM for this unit




# ------------------------------------------------------------------------------
# Or
# ------------------------------------------------------------------------------

import numpy as np

def make_rcos_history_basis(bin_dt, n_basis=5, t_min=1e-9, t_max=0.4):
    """
    Create a raised-cosine bank on log-time between t_min..t_max (seconds).
    Returns centers (in s), width (in s), and a callable that maps lag times -> basis.
    """
    # log spacing of centers
    def logit(x): return np.log(x)
    c = np.linspace(logit(t_min + 1e-9), logit(t_max), n_basis)
    centers = np.exp(c)
    # choose width so bumps overlap ~50%
    if n_basis > 1:
        gaps = np.diff(centers)
        width = np.concatenate([[gaps[0]], gaps])  # heuristic
    else:
        width = np.array([t_max - t_min])
    def phi(t):
        t = np.asarray(t, float)[:, None]
        C = centers[None, :]
        W = width[None, :]
        arg = (t - C) * (np.pi / (2.0 * W))
        B = 0.5 * (1 + np.cos(np.clip(arg, -np.pi, np.pi)))
        B[(t < C - W) | (t > C + W)] = 0.0
        return B
    return centers, width, phi

def make_history_rcos(y_unit, bin_dt, n_basis=5, t_max=0.4):
    """
    Convolve past spikes with a smooth raised-cosine history basis.
    Returns H (T, n_basis) and column names.
    """
    y = np.asarray(y_unit, float).ravel()
    T = y.shape[0]
    centers, width, phi = make_rcos_history_basis(bin_dt, n_basis=n_basis,
                                                  t_min=bin_dt, t_max=t_max)
    # build raw lag matrix up to ceil(t_max/bin_dt)
    L = int(np.ceil(t_max / bin_dt))
    raw, _ = make_history_lags(y, n_lags=L)  # (T, L)

    # times of each lag (in seconds): [dt, 2dt, ..., L*dt]
    lag_times = bin_dt * np.arange(1, L + 1)
    B = phi(lag_times)                         # (L, n_basis)

    # project raw lags onto basis
    H = raw @ B                                # (T, n_basis)

    names = [f'hist_rcos_{i}' for i in range(B.shape[1])]
    return H, names


H, H_names = make_history_rcos(y[:, unit_ix], bin_dt=0.04, n_basis=4, t_max=0.4)
# X_unit = np.column_stack([X_common, H])


# Common block for all units (kinematics, stop features you already built)
X_common = X  # your existing design for rows 'pos' (kinematics + stop predictors)
offset = offset_log
T, n_units = y.shape

for u in range(n_units):
    # spike history for THIS unit
    H_u, H_names = make_history_rcos(y[:, u], bin_dt=0.04, n_basis=4, t_max=0.4)

    # optional: gate history by pre/post
    # H_u = H_u * prepost[:, None]

    X_u = np.column_stack([X_common, H_u])
    names_u = X_names + H_names

    # fit your Poisson GLM with offset (per unit)
    # beta_u = fit_poisson_glm(X_u, y[:, u], offset=offset, reg='ridge', alpha=...)
    # store metrics, coefficients, etc.
