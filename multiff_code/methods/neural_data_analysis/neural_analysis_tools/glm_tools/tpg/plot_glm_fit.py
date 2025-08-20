from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

def _coef_series(result, design_df):
    """Extract coefficients as a Pandas Series indexed by column names."""
    if hasattr(result, 'params'):
        params = np.asarray(result.params).ravel()
        names = getattr(result, 'exog_names', None)
        if names is None and hasattr(result.model, 'exog_names'):
            names = result.model.exog_names
        if names and names[0] == 'const' and len(params) == len(names):
            params = params[1:]
            names = names[1:]
        return pd.Series(params, index=names if names else list(design_df.columns))
    return pd.Series(np.zeros(design_df.shape[1]), index=list(design_df.columns))


def _prefix_cols(prefix: str, names: List[str]) -> List[str]:
    cols = [n for n in names if n.startswith(prefix + '_rc')]

    def rc_idx(c):
        try:
            return int(c.split('_rc')[-1])
        except Exception:
            return 0
    return sorted(cols, key=rc_idx)


def reconstruct_kernel(prefix: str, basis: np.ndarray, coef_s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a time-domain kernel from basis weights with a given prefix."""
    cols = [c for c in coef_s.index if c.startswith(prefix + "_rc")]
    if not cols:
        return np.arange(basis.shape[0]), np.zeros(basis.shape[0])

    def rc_idx(c):
        try:
            return int(c.split('_rc')[-1])
        except Exception:
            return 0
    cols.sort(key=rc_idx)
    w = coef_s.loc[cols].values
    k = basis @ w
    t = np.arange(basis.shape[0])
    return t, k


def reconstruct_history_kernel(B_hist: np.ndarray, coef_s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct spike-history kernel from ``hist_rc*`` weights."""
    cols = [c for c in coef_s.index if c.startswith('hist_rc')]
    if not cols:
        return np.arange(B_hist.shape[0]), np.zeros(B_hist.shape[0])

    def rc_idx(c):
        try:
            return int(c.split('_rc')[-1])
        except Exception:
            return 0
    cols.sort(key=rc_idx)
    w = coef_s.loc[cols].values
    k = B_hist @ w
    t = np.arange(B_hist.shape[0])
    return t, k


def plot_fitted_kernels(result, design_df, meta, dt, *, prefixes=None):
    """Quick plots of reconstructed stimulus kernels and history kernel."""
    import matplotlib.pyplot as plt
    if prefixes is None:
        prefixes = ['cur_on', 'nxt_on', 'cur_dist', 'nxt_dist',
                    'cur_angle_sin', 'cur_angle_cos', 'nxt_angle_sin', 'nxt_angle_cos']
    coef_s = _coef_series(result, design_df)
    B_event, B_short, B_hist = meta['B_event'], meta['B_short'], meta['B_hist']

    def pick_basis(p):
        if p in ['cur_on', 'nxt_on']:
            return B_event
        elif p.startswith(('cur_', 'nxt_')):
            return B_short
        else:
            return B_short

    for p in prefixes:
        B = pick_basis(p)
        t, k = reconstruct_kernel(p, B, coef_s)
        plt.figure()
        plt.plot(t * dt, k)
        plt.xlabel('Time lag (s)')
        plt.ylabel('Kernel weight')
        plt.title(f'{p} kernel')
        plt.show()

    t_h, k_h = reconstruct_history_kernel(B_hist, coef_s)
    plt.figure()
    plt.plot(t_h * dt, k_h)
    plt.xlabel('Time lag (s)')
    plt.ylabel('History weight')
    plt.title('Spike history kernel')
    plt.show()


def _kernel_with_ci(result, design_df, prefix: str, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI for a single prefix kernel via the delta method.

    Returns
    -------
    t_idx : ndarray (lag indices)
    mean  : ndarray (kernel mean at each lag)
    std   : ndarray (kernel std at each lag)
    """
    coef_names = getattr(result, 'model', None).exog_names if hasattr(
        result, 'model') else None
    if coef_names is None:
        raise ValueError(
            'No coefficient names found; fit must use a DataFrame glm_design.')
    cols = _prefix_cols(prefix, coef_names)
    if not cols:
        L = basis.shape[0]
        return np.arange(L), np.zeros(L), np.zeros(L)

    # Extract weights and covariance for this block
    if hasattr(result.params, 'index'):
        w = result.params[result.params.index.isin(cols)]
        w = w.loc[cols]  # ensure order
    else:
        # fallback (e.g., regularized fit without pandas index)
        params_series = _coef_series(result, design_df)
        w = params_series.loc[cols]
    cov = result.cov_params().loc[cols, cols].values

    L = basis.shape[0]
    mean = basis @ w.values
    var = np.einsum('li,ij,lj->l', basis, cov, basis)
    std = np.sqrt(np.maximum(var, 0.0))
    t_idx = np.arange(L)
    return t_idx, mean, std


def plot_angle_kernels_with_ci(result, design_df, meta, dt, base_prefix: str = 'cur_angle', show_history=True):
    """Plot sin, cos, and amplitude (with 95% CIs) for angle tuning over time.

    Uses the fitted model's covariance (cluster-robust if used during fit).
    """
    import matplotlib.pyplot as plt

    B = meta['B_short']
    # sin & cos kernels with CIs
    t_idx, ksin, std_s = _kernel_with_ci(
        result, design_df, f'{base_prefix}_sin', B)
    _,     kcos, std_c = _kernel_with_ci(
        result, design_df, f'{base_prefix}_cos', B)

    # Cross-covariance between sin and cos weights to get Var(amplitude)
    coef_names = result.model.exog_names
    sin_cols = _prefix_cols(f'{base_prefix}_sin', coef_names)
    cos_cols = _prefix_cols(f'{base_prefix}_cos', coef_names)
    cov_full = result.cov_params()
    cov_ss = cov_full.loc[sin_cols, sin_cols].values
    cov_cc = cov_full.loc[cos_cols, cos_cols].values
    cov_sc = cov_full.loc[sin_cols, cos_cols].values

    # Build amplitude mean and variance per lag via delta method
    A = np.sqrt(ksin**2 + kcos**2)
    var_A = np.zeros_like(A)
    for l in range(B.shape[0]):
        b = B[l, :][:, None]  # (K,1)
        var_s = float(b.T @ cov_ss @ b)
        var_c = float(b.T @ cov_cc @ b)
        cov_sc_l = float(b.T @ cov_sc @ b)
        eps = 1e-12
        denom = max(A[l], eps)
        # gradient of sqrt(s^2+c^2)
        g = np.array([ksin[l] / denom, kcos[l] / denom])
        Sigma = np.array([[var_s, cov_sc_l], [cov_sc_l, var_c]])
        var_A[l] = float(g.T @ Sigma @ g)

    std_A = np.sqrt(np.maximum(var_A, 0.0))

    t = t_idx * dt
    # sin
    plt.figure()
    plt.plot(t, ksin, label='sin')
    plt.fill_between(t, ksin - 1.96*std_s, ksin + 1.96*std_s, alpha=0.2)
    plt.xlabel('Time lag (s)')
    plt.ylabel('Kernel weight')
    plt.title(f'{base_prefix}_sin kernel (95% CI)')
    plt.legend()
    plt.show()
    # cos
    plt.figure()
    plt.plot(t, kcos, label='cos')
    plt.fill_between(t, kcos - 1.96*std_c, kcos + 1.96*std_c, alpha=0.2)
    plt.xlabel('Time lag (s)')
    plt.ylabel('Kernel weight')
    plt.title(f'{base_prefix}_cos kernel (95% CI)')
    plt.legend()
    plt.show()
    # amplitude
    plt.figure()
    plt.plot(t, A, label='amplitude')
    plt.fill_between(t, np.maximum(0.0, A - 1.96*std_A),
                     A + 1.96*std_A, alpha=0.2)
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{base_prefix} amplitude (95% CI)')
    plt.legend()
    plt.show()


    # # optional history kernel panel
    # coef_s = _coef_series(result, design_df)
    # t_h, k_h = reconstruct_history_kernel(meta['B_hist'], coef_s)
    # plt.figure()
    # plt.plot(t_h * dt, k_h)
    # plt.xlabel('Time lag (s)')
    # plt.ylabel('History weight')
    # plt.title('Spike history kernel')
    # plt.show()
    
    # optional: spike history kernel with 95% CI
    if show_history and 'B_hist' in meta and hasattr(result, 'cov_params'):
        B_hist = meta['B_hist']
        coef_names = result.model.exog_names
        hist_cols = [n for n in coef_names if n.startswith('hist_rc')]
        if hist_cols:
            if hasattr(result.params, 'index'):
                w_h = result.params.loc[hist_cols]
            else:
                w_h = _coef_series(result, design_df).loc[hist_cols]
            cov_hh = result.cov_params().loc[hist_cols, hist_cols].values
            mean_h = B_hist @ w_h.values
            var_h = np.einsum('li,ij,lj->l', B_hist, cov_hh, B_hist)
            std_h = np.sqrt(np.maximum(var_h, 0.0))
            t_h = np.arange(B_hist.shape[0]) * dt
            plt.figure()
            plt.plot(t_h, mean_h, label='history')
            plt.fill_between(t_h, mean_h - 1.96*std_h,
                             mean_h + 1.96*std_h, alpha=0.2)
            plt.xlabel('Time lag (s)')
            plt.ylabel('History weight')
            plt.title('Spike history kernel (95% CI)')
            plt.legend()
            plt.show()



def angle_tuning_vs_time(result, design_df, meta, base_prefix: str = 'cur_angle'):
    """Return (lags, sin/cos kernels, amplitude, preferred angle) vs lag."""
    coef_s = _coef_series(result, design_df)
    B = meta['B_short']
    t_idx, k_sin = reconstruct_kernel(f'{base_prefix}_sin', B, coef_s)
    _,    k_cos = reconstruct_kernel(f'{base_prefix}_cos', B, coef_s)
    A = np.sqrt(k_sin**2 + k_cos**2)
    phi = np.arctan2(k_sin, k_cos)
    return t_idx, k_sin, k_cos, A, phi


def plot_angle_tuning(result, design_df, meta, dt):
    """Plot sin/cos kernels, amplitude, and preferred angle trajectories."""
    t_idx, k_sin, k_cos, A, phi = angle_tuning_vs_time(result, design_df, meta)

    plt.figure()
    plt.plot(t_idx*dt, k_cos, label="cos")
    plt.plot(t_idx*dt, k_sin, label="sin")
    plt.xlabel("Time lag (s)")
    plt.ylabel("Kernel weight")
    plt.title("Angle kernels")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_idx*dt, A)
    plt.xlabel("Time lag (s)")
    plt.ylabel("Amplitude")
    plt.title("Directional tuning amplitude vs lag")
    plt.show()

    plt.figure()
    plt.plot(t_idx*dt, phi)
    plt.xlabel("Time lag (s)")
    plt.ylabel("Preferred angle (rad)")
    plt.title("Preferred angle vs lag")
    plt.show()


def plot_history_kernels_population(hist_df, *, overlay_mean=True, heatmap=False, max_overlays=60):
    """Population viz for spike-history kernels.

    Parameters
    ----------
    hist_df : DataFrame
        Output of collect_history_kernels_across_neurons.
    overlay_mean : bool
        If True, overlay per-neuron curves (limited to max_overlays) plus mean ± 95% CI.
    heatmap : bool
        If True, also show a neuron × lag heatmap of kernel means.
    max_overlays : int
        Cap number of individual overlays for readability.
    """

    # Mean ± 95% CI across neurons at each lag
    grp = hist_df.groupby('lag_idx', as_index=False).agg(
        lag_s=('lag_s', 'first'),
        mean=('mean', 'mean'),
        lo=('mean', lambda x: x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
        hi=('mean', lambda x: x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
    )

    # Overlays
    if overlay_mean:
        plt.figure()
        # thin overlays
        for nid, df_n in hist_df.groupby('neuron'):
            if nid >= max_overlays:
                break
            plt.plot(df_n['lag_s'], df_n['mean'], alpha=0.3)
        # mean band
        plt.plot(grp['lag_s'], grp['mean'], linewidth=2, label='population mean')
        plt.fill_between(grp['lag_s'], grp['lo'], grp['hi'], alpha=0.2, label='95% CI (across neurons)')
        plt.xlabel('Time lag (s)'); plt.ylabel('History weight'); plt.title('Spike-history kernels (population)')
        plt.legend(); plt.show()

    # Heatmap (neuron × lag)
    if heatmap:
        # pivot to (neuron × lag_idx)
        pivot = hist_df.pivot(index='neuron', columns='lag_idx', values='mean').sort_index()
        plt.figure()
        plt.imshow(pivot.values, aspect='auto', origin='lower',
                   extent=[hist_df['lag_s'].min(), hist_df['lag_s'].max(), pivot.index.min(), pivot.index.max()])
        plt.colorbar(label='History weight')
        plt.xlabel('Time lag (s)'); plt.ylabel('Neuron'); plt.title('Spike-history kernels (heatmap)')
        plt.show()
