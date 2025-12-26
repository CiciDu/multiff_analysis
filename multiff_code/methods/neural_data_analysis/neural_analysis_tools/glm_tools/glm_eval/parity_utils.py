"""
parity_utils.py

RRR / CCA parity utilities:
- consistent CV splits (GroupKFold by trial)
- consistent preprocessing
- rank-matched evaluation
- comparable metrics (predictive VE on held-out test)
- CCA canonical correlation spectra + shuffle null bands
- tidy DataFrames for plotting / downstream analysis
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GroupKFold


# =========================
# Small helpers
# =========================
def _zscore(A: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd, mu, sd


def _center(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(A, axis=0, keepdims=True)
    return A - mu, mu


def _safe_var(y: np.ndarray, eps: float = 1e-12) -> float:
    v = float(np.nanvar(y))
    return v if v > eps else eps


def _variance_explained(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    # VE = 1 - SSE/SST (computed across all entries)
    resid = y_true - y_pred
    sse = float(np.nansum(resid ** 2))
    y0 = y_true - np.nanmean(y_true, axis=0, keepdims=True)
    sst = float(np.nansum(y0 ** 2))
    sst = sst if sst > eps else eps
    return 1.0 - (sse / sst)


def _pinv_solve(X: np.ndarray, Y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """
    Solve for B in X B ≈ Y (least squares / ridge).
    Returns B of shape (p, q).
    """
    # Ridge: (X'X + lam I)^-1 X'Y
    if ridge and ridge > 0:
        XtX = X.T @ X
        p = XtX.shape[0]
        B = np.linalg.solve(XtX + ridge * np.eye(p), X.T @ Y)
        return B
    return np.linalg.pinv(X) @ Y


def maybe_savefig(fig: plt.Figure, save: bool, out_dir: Optional[str], filename: str, dpi: int = 200) -> Optional[str]:
    if not save:
        return None
    if out_dir is None:
        raise ValueError('If save=True you must pass out_dir.')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    return out_path


def cumulative_shared_variance(corrs: np.ndarray) -> np.ndarray:
    pwr = np.cumsum(np.square(np.asarray(corrs, float)))
    tot = pwr[-1] if pwr[-1] > 0 else 1.0
    return pwr / tot


# =========================
# Reduced Rank Regression (RRR)
# =========================
def fit_rrr_ranked(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    max_rank: int,
    ridge: float = 0.0
) -> Dict[int, np.ndarray]:
    """
    Fit RRR coefficients for ranks 1..max_rank using:
      B_ols = argmin ||Y - X B|| (optionally ridge)
      Yhat  = X B_ols
      SVD(Yhat) = U S V^T
      B_r = B_ols V_r V_r^T

    Returns dict: rank -> B_r (p x q)
    """
    B_ols = _pinv_solve(Xtr, Ytr, ridge=ridge)
    Yhat = Xtr @ B_ols

    # SVD on Yhat (n x q) gives V (q x q)
    # rank cannot exceed q or max_rank
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    V = Vt.T

    q = Ytr.shape[1]
    R = int(min(max_rank, q))
    B_by_rank: Dict[int, np.ndarray] = {}

    for r in range(1, R + 1):
        Vr = V[:, :r]                      # (q x r)
        Pr = Vr @ Vr.T                     # (q x q)
        B_by_rank[r] = B_ols @ Pr          # (p x q)

    return B_by_rank


# =========================
# CCA utilities
# =========================
def fit_cca(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    n_components: int
) -> CCA:
    cca = CCA(n_components=int(n_components), max_iter=2000)
    cca.fit(Xtr, Ytr)
    return cca


def cca_transform(
    cca: CCA,
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    U, V = cca.transform(X, Y)
    return U, V


def cca_test_corrs(U: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-component corr(U[:,k], V[:,k]) on test data.
    """
    corrs = []
    for k in range(U.shape[1]):
        u = U[:, k]
        v = V[:, k]
        um = np.nanmean(u)
        vm = np.nanmean(v)
        us = np.nanstd(u)
        vs = np.nanstd(v)
        if us < eps or vs < eps:
            corrs.append(0.0)
            continue
        corrs.append(float(np.nanmean((u - um) * (v - vm)) / (us * vs)))
    return np.asarray(corrs, float)


def cca_predict_Y_from_X(
    cca: CCA,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray
) -> np.ndarray:
    """
    Parity definition for 'CCA predictive VE':
    - compute Utr = Xtr @ Wx  (using cca.transform)
    - fit linear map Y ≈ U A on train
    - predict Yte = Ute A

    This avoids fragile closed-form weight conversions and yields a fair 'predict-Y' metric
    comparable to RRR.
    """
    # Use sklearn's transform to get Utr, but it expects both X and Y;
    # pass Ytr for transform on train. For test, pass dummy Yte via same shape zeros.
    Utr, _ = cca.transform(Xtr, Ytr)
    Ute, _ = cca.transform(Xte, np.zeros((Xte.shape[0], Ytr.shape[1])))

    A = _pinv_solve(Utr, Ytr)  # (k x q)
    return Ute @ A


# =========================
# CV runner + outputs
# =========================
from dataclasses import dataclass

@dataclass
class ParityConfig:
    max_rank: int = 6
    n_splits: int = 5
    standardize: bool = True

    # RRR
    rrr_ridge: float = 0.0

    # CCA shuffle null
    shuffle_mode: str = 'latent'   # 'latent' | 'refit'
    shuffle_level: str = 'trial'   # 'trial' | 'row'
    n_shuffles: int = 100
    shuffle_quantile: float = 0.95

    rng_seed: int = 0


def make_group_splits(trial_ids: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(trial_ids))
    splits = list(gkf.split(idx, groups=trial_ids))
    return splits


def _prepare_views(
    X: np.ndarray,
    Y: np.ndarray,
    standardize: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize per-view (fit on train later). Here just returns raw;
    we keep this function to centralize any future view transformations.
    """
    return np.asarray(X, float), np.asarray(Y, float), {}


def _fit_standardizers(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    standardize: bool
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not standardize:
        return {'X': (np.zeros((1, Xtr.shape[1])), np.ones((1, Xtr.shape[1]))),
                'Y': (np.zeros((1, Ytr.shape[1])), np.ones((1, Ytr.shape[1])))}

    _, mux, sdx = _zscore(Xtr)
    _, muy, sdy = _zscore(Ytr)
    return {'X': (mux, sdx), 'Y': (muy, sdy)}


def _apply_standardizers(A: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (A - mu) / sd

from tqdm import tqdm
import numpy as np
import pandas as pd

# assumes helpers already defined:
# _zscore, _variance_explained, fit_rrr_ranked,
# fit_cca, cca_transform, cca_test_corrs,
# cca_predict_Y_from_X, make_group_splits


def run_parity_cv(
    X,
    Y,
    trial_ids,
    time_idx,
    cond=None,
    *,
    config: ParityConfig = ParityConfig()
):
    """
    Fast parity benchmark:
    - CCA fit once per fold at max_rank
    - rank slicing instead of refitting
    - optional fast or conservative shuffle null
    - progress bar
    """
    rng = np.random.default_rng(config.rng_seed)

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    trial_ids = np.asarray(trial_ids)
    time_idx = np.asarray(time_idx)

    if cond is None:
        cond = np.array(['all'] * len(time_idx))
    else:
        cond = np.asarray(cond)

    splits = make_group_splits(trial_ids, config.n_splits)

    df_rank_rows = []
    df_cca_spec_rows = []
    df_cca_null_rows = []

    latents_by_fold = {}
    cca_by_fold = {}
    rrr_B_by_fold = {}

    for fold_i, (tr_idx, te_idx) in enumerate(
        tqdm(splits, desc='Parity CV (folds)')
    ):
        # -------------------------
        # Split + standardize
        # -------------------------
        Xtr, Xte = X[tr_idx], X[te_idx]
        Ytr, Yte = Y[tr_idx], Y[te_idx]

        if config.standardize:
            Xtr_s, mux, sdx = _zscore(Xtr)
            Xte_s = (Xte - mux) / sdx
            Ytr_s, muy, sdy = _zscore(Ytr)
            Yte_s = (Yte - muy) / sdy
        else:
            Xtr_s, Xte_s = Xtr, Xte
            Ytr_s, Yte_s = Ytr, Yte

        # -------------------------
        # RRR (fit once → all ranks)
        # -------------------------
        B_by_rank = fit_rrr_ranked(
            Xtr_s, Ytr_s,
            max_rank=config.max_rank,
            ridge=config.rrr_ridge
        )
        rrr_B_by_fold[fold_i] = B_by_rank

        # -------------------------
        # CCA (fit ONCE at max_rank)
        # -------------------------
        cca = fit_cca(Xtr_s, Ytr_s, n_components=config.max_rank)
        cca_by_fold[fold_i] = {r: cca for r in range(1, config.max_rank + 1)}

        Ute, Vte = cca_transform(cca, Xte_s, Yte_s)

        # -------------------------
        # Shuffle null (CCA spectrum)
        # -------------------------
        null_corrs = np.zeros((config.n_shuffles, config.max_rank))

        if config.shuffle_mode == 'latent':
            # ---- FAST: shuffle correspondence in test ----
            for s in tqdm(
                range(config.n_shuffles),
                desc=f'  latent shuffles (fold {fold_i})',
                leave=False
            ):
                if config.shuffle_level == 'trial':
                    perm = rng.permutation(np.unique(trial_ids[te_idx]))
                    idx_map = {}
                    for src, dst in zip(np.unique(trial_ids[te_idx]), perm):
                        src_rows = np.where(trial_ids[te_idx] == src)[0]
                        dst_rows = np.where(trial_ids[te_idx] == dst)[0]
                        if len(src_rows) != len(dst_rows):
                            idx_map = None
                            break
                        idx_map[src_rows] = dst_rows

                    if idx_map is None:
                        perm_idx = rng.permutation(len(te_idx))
                    else:
                        perm_idx = np.concatenate(
                            [idx_map[k] for k in sorted(idx_map)]
                        )
                else:
                    perm_idx = rng.permutation(len(te_idx))

                Vshuf = Vte[perm_idx]
                null_corrs[s] = cca_test_corrs(Ute, Vshuf)

        elif config.shuffle_mode == 'refit':
            # ---- SLOW but conservative ----
            tr_trials = trial_ids[tr_idx]
            unique_tr = np.unique(tr_trials)
            rows_by_trial = {t: np.where(tr_trials == t)[0] for t in unique_tr}

            for s in tqdm(
                range(config.n_shuffles),
                desc=f'  refit shuffles (fold {fold_i})',
                leave=False
            ):
                perm_trials = rng.permutation(unique_tr)
                Ytr_shuf = np.empty_like(Ytr_s)

                for src_t, dst_t in zip(unique_tr, perm_trials):
                    src_rows = rows_by_trial[src_t]
                    dst_rows = rows_by_trial[dst_t]
                    if len(src_rows) != len(dst_rows):
                        Ytr_shuf = Ytr_s[rng.permutation(Ytr_s.shape[0])]
                        break
                    Ytr_shuf[src_rows] = Ytr_s[dst_rows]

                cca_shuf = fit_cca(Xtr_s, Ytr_shuf, config.max_rank)
                Ute_s, Vte_s = cca_transform(cca_shuf, Xte_s, Yte_s)
                null_corrs[s] = cca_test_corrs(Ute_s, Vte_s)

        thresh = np.nanquantile(
            null_corrs, config.shuffle_quantile, axis=0
        )

        for k in range(config.max_rank):
            df_cca_null_rows.append({
                'fold': fold_i,
                'component': k + 1,
                'corr_thresh': float(thresh[k])
            })

        # -------------------------
        # Evaluate all ranks
        # -------------------------
        for r in range(1, config.max_rank + 1):
            # ---- RRR VE ----
            B = B_by_rank[r]
            Yhat_rrr = Xte_s @ B
            ve_rrr = _variance_explained(Yte_s, Yhat_rrr)

            df_rank_rows.append({
                'fold': fold_i,
                'rank': r,
                'model': 'rrr',
                've_pred_y_from_x': ve_rrr
            })

            # ---- CCA metrics ----
            U_r = Ute[:, :r]
            V_r = Vte[:, :r]
            corrs = cca_test_corrs(U_r, V_r)

            for k, c in enumerate(corrs, start=1):
                df_cca_spec_rows.append({
                    'fold': fold_i,
                    'rank': r,
                    'component': k,
                    'corr_test': c
                })

            Yhat_cca = cca_predict_Y_from_X(
                cca, Xtr_s, Ytr_s, Xte_s
            )
            ve_cca = _variance_explained(Yte_s, Yhat_cca)

            df_rank_rows.append({
                'fold': fold_i,
                'rank': r,
                'model': 'cca',
                've_pred_y_from_x': ve_cca,
                'corr1_test': corrs[0] if len(corrs) > 0 else np.nan,
                'corr2_test': corrs[1] if len(corrs) > 1 else np.nan
            })

        # -------------------------
        # Store latents (max_rank)
        # -------------------------
        d_obs = pd.DataFrame({
            'trial': trial_ids[te_idx],
            'time': time_idx[te_idx],
            'cond': cond[te_idx],
            'view': 'obs'
        })
        d_pred = d_obs.copy()
        d_pred['view'] = 'pred'

        for k in range(config.max_rank):
            d_obs[f'can{k+1}'] = Ute[:, k]
            d_pred[f'can{k+1}'] = Vte[:, k]

        latents_by_fold[fold_i] = pd.concat(
            [d_obs, d_pred], ignore_index=True
        )

    return {
        'df_rank': pd.DataFrame(df_rank_rows),
        'df_cca_spectrum': pd.DataFrame(df_cca_spec_rows),
        'df_cca_null': pd.DataFrame(df_cca_null_rows),
        'latents_by_fold': latents_by_fold,
        'cca_by_fold': cca_by_fold,
        'rrr_B_by_fold': rrr_B_by_fold
    }


# =========================
# Plotting utilities (parity-focused)
# =========================
def plot_ve_vs_rank(
    df_rank: pd.DataFrame,
    *,
    agg: str = 'mean',   # 'mean' or 'median'
    save: bool = False,
    out_dir: Optional[str] = None,
    filename: str = 've_vs_rank.png'
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    if agg == 'median':
        d = df_rank.groupby(['model', 'rank'])['ve_pred_y_from_x'].median().reset_index()
    else:
        d = df_rank.groupby(['model', 'rank'])['ve_pred_y_from_x'].mean().reset_index()

    for model in ['cca', 'rrr']:
        sub = d[d['model'] == model].sort_values('rank')
        ax.plot(sub['rank'], sub['ve_pred_y_from_x'], marker='o', label=model)

    ax.set_xlabel('Rank')
    ax.set_ylabel('Predictive VE (test)')
    ax.set_title('Predict Y from X: CCA vs RRR')
    ax.grid(alpha=0.2, ls=':')
    ax.legend()

    maybe_savefig(fig, save, out_dir, filename)
    return fig


def plot_cca_spectrum_with_null(
    df_cca_spectrum: pd.DataFrame,
    df_cca_null: pd.DataFrame,
    *,
    rank: int,
    save: bool = False,
    out_dir: Optional[str] = None,
    filename: str = 'cca_spectrum.png'
) -> plt.Figure:
    """
    Plot mean test corr spectrum at a chosen rank, plus shuffle threshold (mean across folds).
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    spec = df_cca_spectrum[df_cca_spectrum['rank'] == rank]
    m = spec.groupby('component')['corr_test'].mean().reset_index()
    e = spec.groupby('component')['corr_test'].sem().reset_index().rename(columns={'corr_test': 'sem'})

    null = df_cca_null.groupby('component')['corr_thresh'].mean().reset_index()

    x = m['component'].values
    ax.plot(x, m['corr_test'].values, marker='o', label='Test corr (mean)')
    ax.fill_between(x,
                    m['corr_test'].values - e['sem'].values,
                    m['corr_test'].values + e['sem'].values,
                    alpha=0.18)

    ax.plot(null['component'].values, null['corr_thresh'].values, '--', label='Shuffle 95% (mean)')

    ax.set_xlabel('Component')
    ax.set_ylabel('Corr')
    ax.set_title(f'CCA test spectrum (rank={rank})')
    ax.grid(alpha=0.2, ls=':')
    ax.legend()

    maybe_savefig(fig, save, out_dir, filename)
    return fig


def plot_cumulative_shared_signal(
    df_cca_spectrum: pd.DataFrame,
    *,
    rank: int,
    save: bool = False,
    out_dir: Optional[str] = None,
    filename: str = 'cca_cumulative_shared.png'
) -> plt.Figure:
    """
    Plot cumulative sum(corr^2)/total at a chosen rank.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    spec = df_cca_spectrum[df_cca_spectrum['rank'] == rank]
    m = spec.groupby('component')['corr_test'].mean().reset_index().sort_values('component')

    cum = cumulative_shared_variance(m['corr_test'].values)
    ax.plot(m['component'].values, cum, marker='o')

    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Component')
    ax.set_ylabel('Cumulative corr²')
    ax.set_title(f'Cumulative shared signal (rank={rank})')
    ax.grid(alpha=0.2, ls=':')

    maybe_savefig(fig, save, out_dir, filename)
    return fig

# =========================
# Unified latent extraction
# =========================
def get_latents_for_model(
    *,
    model: str,                   # 'cca' | 'rrr'
    rank: int,
    fold: int,
    res: dict,                    # output of run_parity_cv(...)
    X: np.ndarray,
    Y: np.ndarray,
    trial_ids: np.ndarray,
    time_idx: np.ndarray,
    cond: np.ndarray | None = None,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Return tidy latent trajectories for a given model / rank / fold.

    Output schema (compatible with plot_shared_components):
        trial | time | cond | view | can1 | ... | canR

    Notes
    -----
    - For CCA:
        obs  = canonical coords of X
        pred = canonical coords of Y
    - For RRR:
        obs  = X @ B_r   (low-rank predictive subspace)
        pred = Y_hat     (prediction from X)
    """

    assert model in ('cca', 'rrr')

    if cond is None:
        cond = np.array(['all'] * len(time_idx))
    else:
        cond = np.asarray(cond)

    splits = make_group_splits(trial_ids, res['df_rank']['fold'].nunique())
    tr_idx, te_idx = splits[fold]

    Xtr, Xte = X[tr_idx], X[te_idx]
    Ytr, Yte = Y[tr_idx], Y[te_idx]

    # --- standardize using TRAIN only ---
    if standardize:
        Xtr_s, mux, sdx = _zscore(Xtr)
        Xte_s = (Xte - mux) / sdx
        Ytr_s, muy, sdy = _zscore(Ytr)
        Yte_s = (Yte - muy) / sdy
    else:
        Xte_s, Yte_s = Xte, Yte

    # =========================
    # CCA latents
    # =========================
    if model == 'cca':
        cca = res['cca_by_fold'][fold][rank]
        Ute, Vte = cca.transform(Xte_s, Yte_s)

        Z_obs = Ute[:, :rank]
        Z_pred = Vte[:, :rank]

    # =========================
    # RRR latents
    # =========================
    else:
        B = res['rrr_B_by_fold'][fold][rank]  # (p x q)
        Z_obs = Xte_s @ B[:, :rank]
        Z_pred = Xte_s @ B[:, :rank]          # predictive subspace

    # =========================
    # Build tidy DataFrame
    # =========================
    dfs = []
    for view, Z in [('obs', Z_obs), ('pred', Z_pred)]:
        d = pd.DataFrame({
            'trial': trial_ids[te_idx],
            'time': time_idx[te_idx],
            'cond': cond[te_idx],
            'view': view
        })
        for k in range(rank):
            d[f'can{k+1}'] = Z[:, k]
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)
