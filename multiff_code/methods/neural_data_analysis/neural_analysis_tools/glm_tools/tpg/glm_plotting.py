from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Sequence
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# Core helpers (names, params, cov)
# -------------------------------
def _exog_names(result, design_df: pd.DataFrame | None = None) -> List[str]:
    names = getattr(getattr(result, 'model', None), 'exog_names', None)
    if names is None:
        names = getattr(result, 'exog_names', None)
    if names is None and design_df is not None:
        names = list(design_df.columns)
    return list(names) if names is not None else []


def _params_series(result, names: Sequence[str]) -> pd.Series:
    p = getattr(result, 'params', None)
    if hasattr(p, 'index'):
        return p.reindex(names)
    arr = np.asarray(p).ravel()
    return pd.Series(arr, index=names)


def _cov_df(result, names: Sequence[str]) -> pd.DataFrame:
    C = result.cov_params()
    if hasattr(C, 'index'):
        return C.reindex(index=names, columns=names)
    C = np.asarray(C, float)
    return pd.DataFrame(C, index=names, columns=names)


# -------------------------------
# Column selection for a predictor
# - supports raw, rc, bjk, and generic prefix:
#     raw:        '<prefix>'
#     rc:         '<prefix>_rc0', '<prefix>_rc1', ...
#     bjk:        '<prefix>:b{j}:{k}'
#     generic:    '<prefix>:...'
# -------------------------------
_rc_pat = r'_rc(\d+)$'
_bjk_pat = r':b(\d+):(\d+)$'

def _cols_for_prefix(prefix: str, names: Sequence[str]) -> List[str]:
    # raw passthrough (no basis)
    raw = [n for n in names if n == prefix]
    if raw:
        return raw

    # rc-style: '<prefix>_rc0', '<prefix>_rc1', ...
    rc_cols = [n for n in names if re.search(rf'^{re.escape(prefix)}{_rc_pat}', n)]
    if rc_cols:
        rc_cols.sort(key=lambda c: int(re.search(_rc_pat, c).group(1)))
        return rc_cols

    # bjk-style: '<prefix>:b{j}:{k}'
    bjk_cols = [n for n in names if re.search(rf'^{re.escape(prefix)}{_bjk_pat}', n)]
    if bjk_cols:
        bjk_cols.sort(key=lambda c: tuple(int(g) for g in re.search(_bjk_pat, c).groups()))
        return bjk_cols

    # generic prefix (e.g., '<prefix>:sin1', '<prefix>:cos1', other)
    return [n for n in names if n.startswith(prefix + ':')]


# -------------------------------
# Basis retrieval & stacking
# -------------------------------
def _stack_bases_for(prefix: str, meta: dict, bases_by_predictor: dict[str, list[np.ndarray]] | None) -> np.ndarray:
    # 1) explicit arg
    if bases_by_predictor and prefix in bases_by_predictor:
        Bs = bases_by_predictor[prefix]
    # 2) meta registry
    elif 'bases_by_predictor' in meta and prefix in meta['bases_by_predictor']:
        Bs = meta['bases_by_predictor'][prefix]
    # 3) legacy fallbacks (single basis)
    elif 'B_angle' in meta and ('angle' in prefix or 'sin' in prefix or 'cos' in prefix):
        Bs = [meta['B_angle']]
    elif 'B_short' in meta:
        Bs = [meta['B_short']]
    elif 'B_event' in meta:
        Bs = [meta['B_event']]
    else:
        raise KeyError(f'no bases found for prefix {prefix!r}')

    if not isinstance(Bs, (list, tuple)):
        Bs = [Bs]

    Ls = {B.shape[0] for B in Bs}
    if len(Ls) != 1:
        raise ValueError(f'all bases for {prefix!r} must share the same number of lags; got {Ls}')
    return np.hstack(Bs) if len(Bs) > 1 else Bs[0]


# -------------------------------
# Static Fourier helpers (angle without kernels)
# -------------------------------
_sin_static_pat = r':sin(\d+)$'
_cos_static_pat = r':cos(\d+)$'

def _static_angle_cols(base_prefix: str, names: Sequence[str]) -> Tuple[List[str], List[str]]:
    sin_cols = [n for n in names if re.search(rf'^{re.escape(base_prefix)}{_sin_static_pat}', n)]
    cos_cols = [n for n in names if re.search(rf'^{re.escape(base_prefix)}{_cos_static_pat}', n)]
    # sort by harmonic index
    sin_cols.sort(key=lambda c: int(re.search(r'(\d+)$', c).group(1)))
    cos_cols.sort(key=lambda c: int(re.search(r'(\d+)$', c).group(1)))
    return sin_cols, cos_cols


# -------------------------------
# Kernel reconstruction + CI (delta method)
# -------------------------------
def _kernel_and_ci_for_prefix(
    result,
    design_df: pd.DataFrame | None,
    meta: dict,
    dt: float,
    prefix: str,
    *,
    bases_by_predictor: dict[str, list[np.ndarray]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return t (s), mean kernel, std kernel for one predictor prefix.
    Works with multiple bases: B_stacked @ beta, Var = diag(B Σ B^T).
    """
    names = _exog_names(result, design_df)
    if not names:
        raise ValueError('could not determine exogenous names')
    coef = _params_series(result, names)
    cov = _cov_df(result, names)

    cols = _cols_for_prefix(prefix, names)
    if not cols:
        # no columns found — try to get L from bases; otherwise return scalar zero at t=0
        try:
            B = _stack_bases_for(prefix, meta, bases_by_predictor)
            L = B.shape[0]
            t = np.arange(L) * dt
        except Exception:
            L = 1
            t = np.arange(L) * dt
        return t, np.zeros(L), np.zeros(L)

    B = _stack_bases_for(prefix, meta, bases_by_predictor)
    if B.shape[1] != len(cols):
        raise ValueError(f'basis/columns mismatch for {prefix!r}: K={B.shape[1]} vs {len(cols)} cols')

    beta = coef.loc[cols].to_numpy()
    mean = B @ beta

    C = cov.loc[cols, cols].to_numpy()
    var = np.einsum('li,ij,lj->l', B, C, B)
    std = np.sqrt(np.maximum(var, 0.0))
    t = np.arange(B.shape[0]) * dt
    return t, mean, std


# -------------------------------
# Public: quick multi-kernel plot
# -------------------------------
def plot_fitted_kernels(
    result,
    design_df,
    meta,
    dt,
    *,
    prefixes=None,
    bases_by_predictor=None,
    z: float = 1.96,          # 95% CI (use 1.64 for ~90%, 2.58 for ~99%)
    alpha_fill: float = 0.20, # CI shading opacity
):
    """Quick plots of reconstructed kernels with confidence bands."""
    if prefixes is None:
        prefixes = [
            'cur_vis_on', 'cur_vis_off',
            'nxt_vis_on', 'nxt_vis_off',
            'spk_hist',                 # history kernel
            # add angle kernels here only if you actually built them:
            # 'cur_ff_angle_sin', 'cur_ff_angle_cos',
            # 'nxt_ff_angle_sin', 'nxt_ff_angle_cos',
            # 'stop', 'capture',
        ]

    for p in prefixes:
        t, mean, std = _kernel_and_ci_for_prefix(
            result, design_df, meta, dt, p, bases_by_predictor=bases_by_predictor
        )

        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        if t.size <= 1 or np.allclose(std, 0):
            # Degenerate / missing predictor: annotate clearly
            ax.text(0.5, 0.5, f"No kernel columns for '{p}' (or dropped as all-zero)",
                    transform=ax.transAxes, ha='center', va='center', alpha=0.7)
            ax.set_xlabel('Time lag (s)'); ax.set_ylabel('Kernel weight')
            ax.set_title(f'{p} kernel')
            plt.tight_layout(); plt.show()
            continue

        ax.plot(t, mean, label=p)
        ax.fill_between(t, mean - z * std, mean + z * std, alpha=alpha_fill, label=f'{int(round((2*scipy.stats.norm.cdf(z)-1)*100))}% CI' if 'scipy' in globals() else 'CI')
        ax.set_xlabel('Time lag (s)')
        ax.set_ylabel('Kernel weight')
        ax.set_title(f'{p} kernel')
        ax.legend()
        plt.tight_layout()
        plt.show()




# -------------------------------
# Population viz for history kernels
# -------------------------------
def plot_history_kernels_population(hist_df: pd.DataFrame, *, overlay_mean=True, heatmap=False, max_overlays=60):
    """
    hist_df columns expected: ['neuron', 'lag_idx', 'lag_s', 'mean'].
    """
    agg = hist_df.groupby('lag_idx', as_index=False).agg(
        lag_s=('lag_s', 'first'),
        mean=('mean', 'mean'),
        n=('mean', 'size'),
        sd=('mean', lambda x: x.std(ddof=1)),
    )
    agg['se'] = agg['sd'] / np.sqrt(np.maximum(agg['n'], 1))
    agg['lo'] = agg['mean'] - 1.96 * agg['se']
    agg['hi'] = agg['mean'] + 1.96 * agg['se']

    if overlay_mean:
        plt.figure()
        for i, (nid, df_n) in enumerate(hist_df.groupby('neuron')):
            if i >= max_overlays:
                break
            plt.plot(df_n['lag_s'], df_n['mean'], alpha=0.3)
        plt.plot(agg['lag_s'], agg['mean'], linewidth=2, label='population mean')
        plt.fill_between(agg['lag_s'], agg['lo'], agg['hi'], alpha=0.2, label='95% CI (across neurons)')
        plt.xlabel('Time lag (s)')
        plt.ylabel('History weight')
        plt.title('Spike-history kernels (population)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if heatmap:
        pivot = hist_df.pivot(index='neuron', columns='lag_idx', values='mean').sort_index()
        plt.figure()
        plt.imshow(pivot.values, aspect='auto', origin='lower')
        plt.colorbar(label='History weight')
        plt.xlabel('Lag index')
        plt.ylabel('Neuron')
        plt.title('Spike-history kernels (heatmap)')
        plt.tight_layout()
        plt.show()


def get_angle_tuning_with_ci(
    result,
    design_df: pd.DataFrame,
    meta: dict,
    *,
    base_prefix: str = 'cur_ff_angle',
    bases_by_predictor: dict[str, list[np.ndarray]] | None = None,
    mode: str = 'peak',         # 'peak' | 'lag' | 'integrate'  (kernel mode only)
    lag: Optional[int] = None,  # used when mode='lag'
    M: Optional[int] = None,    # static mode: #harmonics (auto if None)
    include_intercept: bool = False,
    theta_grid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Return (theta, f(theta), std(theta), info) for the angle tuning function.
    Works in both KERNEL (sin/cos streams with temporal bases) and STATIC Fourier modes.
    The CI is delta-method using the GLM coefficient covariance.
    """
    names = _exog_names(result, design_df)
    coef  = _params_series(result, names)
    cov   = _cov_df(result, names)

    # grid
    theta = np.linspace(-np.pi, np.pi, 361, endpoint=True) if theta_grid is None else np.asarray(theta_grid)

    # try KERNEL mode first
    sin_prefix = f'{base_prefix}_sin'
    cos_prefix = f'{base_prefix}_cos'
    sin_cols_k = _cols_for_prefix(sin_prefix, names)
    cos_cols_k = _cols_for_prefix(cos_prefix, names)

    if sin_cols_k and cos_cols_k:
        # bases
        B_sin = _stack_bases_for(sin_prefix, meta, bases_by_predictor)
        B_cos = _stack_bases_for(cos_prefix, meta, bases_by_predictor)
        if B_sin.shape[1] != len(sin_cols_k) or B_cos.shape[1] != len(cos_cols_k):
            raise ValueError('Angle kernel basis/columns mismatch.')
        if B_sin.shape[0] != B_cos.shape[0]:
            raise ValueError('Sin and cos angle kernels must share lag length.')

        beta_s = coef.loc[sin_cols_k].to_numpy()
        beta_c = coef.loc[cos_cols_k].to_numpy()
        ksin = B_sin @ beta_s
        kcos = B_cos @ beta_c
        L = ksin.shape[0]

        # choose lag transform vectors (row weights over coeffs)
        if mode == 'peak':
            A = np.sqrt(ksin**2 + kcos**2)
            lag_idx = int(np.argmax(A))
            rs = B_sin[lag_idx, :]  # (K_s,)
            rc = B_cos[lag_idx, :]  # (K_c,)
        elif mode == 'lag':
            if lag is None:
                raise ValueError("Provide `lag` when mode='lag'.")
            lag_idx = int(np.clip(lag, 0, L-1))
            rs = B_sin[lag_idx, :]
            rc = B_cos[lag_idx, :]
        elif mode == 'integrate':
            lag_idx = None
            rs = B_sin.sum(axis=0)
            rc = B_cos.sum(axis=0)
        else:
            raise ValueError("mode must be 'peak', 'lag', or 'integrate'.")

        # build selector in the order [sin_cols..., cos_cols..., (const?)]
        col_order = list(sin_cols_k) + list(cos_cols_k)
        Sigma = cov.loc[col_order, col_order].to_numpy()

        # design over theta: row = [sinθ * rs | cosθ * rc | (1 if intercept)]
        S = np.sin(theta)[:, None] * rs[None, :]        # (T, K_s)
        C = np.cos(theta)[:, None] * rc[None, :]        # (T, K_c)
        D = np.hstack([S, C])                           # (T, K_s + K_c)

        if include_intercept and 'const' in coef.index:
            # augment Σ and D with intercept column
            col_order_int = col_order + ['const']
            Sigma = cov.loc[col_order_int, col_order_int].to_numpy()
            D = np.hstack([D, np.ones((D.shape[0], 1), float)])
            beta_sub = coef.loc[col_order_int].to_numpy()
        else:
            beta_sub = coef.loc[col_order].to_numpy()

        f = D @ beta_sub
        var = np.einsum('ti,ij,tj->t', D, Sigma, D)
        std = np.sqrt(np.maximum(var, 0.0))

        info = {
            'mode': 'kernel',
            'choice': mode,
            'lag_idx': lag_idx,
            'ksin': float(rs @ beta_s),
            'kcos': float(rc @ beta_c),
            'amplitude': float(np.hypot(rs @ beta_s, rc @ beta_c)),
        }
        return theta, f, std, info

    # STATIC Fourier fallback
    sin_cols_s, cos_cols_s = _static_angle_cols(base_prefix, names)
    if not sin_cols_s or not cos_cols_s:
        raise KeyError(
            f"No angle columns for base_prefix='{base_prefix}'. "
            f"Tried kernels ('{sin_prefix}', '{cos_prefix}') and static ('{base_prefix}:sin*', ':cos*')."
        )

    if M is None:
        M = min(len(sin_cols_s), len(cos_cols_s))

    # interleave as [cos1, sin1, cos2, sin2, ...] to match D we build
    cols = []
    for m in range(1, int(M)+1):
        cols.extend([cos_cols_s[m-1], sin_cols_s[m-1]])

    Sigma = cov.loc[cols, cols].to_numpy()
    beta_sub = coef.loc[cols].to_numpy()

    # build D(theta): [cos θ, sin θ, cos 2θ, sin 2θ, ...]
    T = theta.size
    D = np.zeros((T, len(cols)), float)
    for m in range(1, int(M)+1):
        D[:, 2*(m-1) + 0] = np.cos(m * theta)
        D[:, 2*(m-1) + 1] = np.sin(m * theta)

    if include_intercept and 'const' in coef.index:
        cols_int = cols + ['const']
        Sigma = cov.loc[cols_int, cols_int].to_numpy()
        beta_sub = coef.loc[cols_int].to_numpy()
        D = np.hstack([D, np.ones((T, 1), float)])

    f = D @ beta_sub
    var = np.einsum('ti,ij,tj->t', D, Sigma, D)
    std = np.sqrt(np.maximum(var, 0.0))

    info = {
        'mode': 'static',
        'M': int(M),
    }
    return theta, f, std, info


def plot_angle_tuning_function(
    result,
    design_df: pd.DataFrame,
    meta: dict,
    *,
    base_prefix: str = 'cur_ff_angle',
    bases_by_predictor: dict[str, list[np.ndarray]] | None = None,
    mode: str = 'peak',         # kernel mode
    lag: Optional[int] = None,  # kernel mode
    M: Optional[int] = None,    # static mode
    include_intercept: bool = False,
    theta_grid: Optional[np.ndarray] = None,
    polar: bool = False,
    z: float = 1.96,            # 95% CI
    alpha_fill: float = 0.20,
):
    """Plot f(θ) with ±z·SE bands (supports kernel and static modes)."""
    theta, f, std, info = get_angle_tuning_with_ci(
        result, design_df, meta,
        base_prefix=base_prefix,
        bases_by_predictor=bases_by_predictor,
        mode=mode, lag=lag, M=M,
        include_intercept=include_intercept,
        theta_grid=theta_grid
    )

    lo = f - z * std
    hi = f + z * std

    if polar:
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, f, label='mean')
        ax.fill_between(theta, lo, hi, alpha=alpha_fill, label=f'±{z:.2f}·SE')
        ax.set_title(f"{base_prefix} tuning ({info['mode']})")
        ax.legend()
        plt.tight_layout(); plt.show()
    else:
        plt.figure()
        plt.plot(theta, f, label='mean')
        plt.fill_between(theta, lo, hi, alpha=alpha_fill, label=f'±{z:.2f}·SE')
        plt.xlabel('Angle (rad)'); plt.ylabel('Tuning f(θ)')
        title_bits = [base_prefix, info['mode']]
        if info.get('lag_idx') is not None:
            title_bits.append(f"lag={info['lag_idx']}")
        plt.title(' | '.join(title_bits))
        plt.legend()
        plt.tight_layout(); plt.show()

    return theta, f, std, info
