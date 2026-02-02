
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from typing import Optional

# ============================================================
# --------------------- basic helpers ------------------------
# ============================================================
def _zscore(A: np.ndarray, eps: float = 1e-12):
    mu = np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd, mu, sd


def _variance_explained(Y: np.ndarray, Yhat: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sum((Y - Yhat) ** 2)
    den = np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2)
    if not np.isfinite(den) or den <= eps:
        return 0.0
    return float(np.clip(1.0 - num / den, 0.0, 1.0))


def make_group_splits(trial_ids: np.ndarray, n_splits: int):
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(trial_ids))
    return list(gkf.split(idx, groups=trial_ids))


# ============================================================
# -------------------- RRR utilities -------------------------
# ============================================================
def fit_rrr_ranked(Xtr: np.ndarray, Ytr: np.ndarray, *, max_rank: int, ridge: float = 0.0):
    """
    Reduced-rank regression via SVD of Yhat = X B_ols.
    Returns dict: rank -> B_r
    """
    if ridge > 0:
        XtX = Xtr.T @ Xtr
        B_ols = np.linalg.solve(
            XtX + ridge * np.eye(XtX.shape[0]), Xtr.T @ Ytr)
    else:
        B_ols = np.linalg.pinv(Xtr) @ Ytr

    Yhat = Xtr @ B_ols
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    V = Vt.T

    B_by_rank = {}
    for r in range(1, max_rank + 1):
        Vr = V[:, :r]
        B_by_rank[r] = B_ols @ (Vr @ Vr.T)

    return B_by_rank


# ============================================================
# -------------------- CCA utilities -------------------------
# ============================================================
def fit_cca(Xtr: np.ndarray, Ytr: np.ndarray, *, n_components: int) -> CCA:
    cca = CCA(n_components=n_components, max_iter=2000)
    cca.fit(Xtr, Ytr)
    return cca


def cca_transform(cca: CCA, X: np.ndarray, Y: np.ndarray):
    return cca.transform(X, Y)


def cca_test_corrs(U: np.ndarray, V: np.ndarray, eps: float = 1e-12):
    corrs = []
    for k in range(U.shape[1]):
        su, sv = np.std(U[:, k]), np.std(V[:, k])
        if su < eps or sv < eps:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(U[:, k], V[:, k])[0, 1]))
    return np.asarray(corrs)


def cca_predict_Y_from_X(
    cca: CCA,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    rank: int
):
    """
    Predict Y from X using CCA latents:
      - learn Y ≈ U A on TRAIN
      - apply to TEST
    """
    Utr, _ = cca.transform(Xtr, Ytr)
    Ute, _ = cca.transform(Xte, np.zeros((Xte.shape[0], Ytr.shape[1])))

    A = np.linalg.pinv(Utr[:, :rank]) @ Ytr
    return Ute[:, :rank] @ A


# ============================================================
# ---------------------- config ------------------------------
# ============================================================
@dataclass
class ParityConfig:
    max_rank: int = 10
    n_splits: int = 5

    # RRR
    rrr_ridge: float = 0.0

    # shuffle null
    shuffle_mode: str = 'latent'    # 'latent' | 'refit'
    shuffle_level: str = 'trial'    # 'trial' | 'row'
    n_shuffles: int = 100
    shuffle_quantile: float = 0.95

    rng_seed: int = 0


# ============================================================
# ------------------- main parity CV -------------------------
# ============================================================
def run_parity_cv(
    X,
    Y,
    trial_ids,
    time_idx,
    cond=None,
    *,
    config: Optional[ParityConfig] = None
):
    """
    Canonical parity benchmark for CCA vs RRR.

    Returns dict with:
        df_rank
        df_cca_spectrum
        df_cca_null
        latents_by_fold
        cca_by_fold
        rrr_B_by_fold
    """

    if config is None:
        config = ParityConfig()

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
        # ------------------ split + standardize ------------------
        Xtr, Xte = X[tr_idx], X[te_idx]
        Ytr, Yte = Y[tr_idx], Y[te_idx]

        Xtr_s, mux, sdx = _zscore(Xtr)
        Xte_s = (Xte - mux) / sdx
        Ytr_s, muy, sdy = _zscore(Ytr)
        Yte_s = (Yte - muy) / sdy

        # ------------------ RRR ------------------
        B_by_rank = fit_rrr_ranked(
            Xtr_s, Ytr_s,
            max_rank=config.max_rank,
            ridge=config.rrr_ridge
        )
        rrr_B_by_fold[fold_i] = B_by_rank

        # ------------------ CCA (fit once) ------------------
        cca = fit_cca(Xtr_s, Ytr_s, n_components=config.max_rank)
        cca_by_fold[fold_i] = cca

        Ute, Vte = cca_transform(cca, Xte_s, Yte_s)

        # ------------------ shuffle null ------------------
        null_corrs = np.zeros((config.n_shuffles, config.max_rank))

        for s in range(config.n_shuffles):
            if config.shuffle_mode == 'latent':
                if config.shuffle_level == 'row':
                    perm = rng.permutation(len(Vte))
                else:
                    te_trials = trial_ids[te_idx]
                    unique_tr = np.unique(te_trials)
                    perm_trials = rng.permutation(unique_tr)

                    perm_blocks = []
                    ok = True

                    for src, dst in zip(unique_tr, perm_trials):
                        src_rows = np.where(te_trials == src)[0]
                        dst_rows = np.where(te_trials == dst)[0]

                        if len(src_rows) != len(dst_rows):
                            ok = False
                            break

                        perm_blocks.append(dst_rows)

                    if not ok:
                        perm = rng.permutation(len(Vte))
                    else:
                        perm = np.concatenate(perm_blocks)

                null_corrs[s] = cca_test_corrs(Ute, Vte[perm])

            else:  # refit
                if config.shuffle_level == 'row':
                    Ytr_shuf = Ytr_s[rng.permutation(len(Ytr_s))]
                else:
                    tr_trials = trial_ids[tr_idx]
                    unique_tr = np.unique(tr_trials)
                    perm_trials = rng.permutation(unique_tr)

                    rows_by_trial = {
                        t: np.where(tr_trials == t)[0]
                        for t in unique_tr
                    }

                    Ytr_shuf = np.empty_like(Ytr_s)
                    for src, dst in zip(unique_tr, perm_trials):
                        src_rows = rows_by_trial[src]
                        dst_rows = rows_by_trial[dst]
                        if len(src_rows) != len(dst_rows):
                            Ytr_shuf = Ytr_s[rng.permutation(len(Ytr_s))]
                            break
                        Ytr_shuf[src_rows] = Ytr_s[dst_rows]

                cca_sh = fit_cca(Xtr_s, Ytr_shuf, n_components=config.max_rank)
                U_s, V_s = cca_transform(cca_sh, Xte_s, Yte_s)
                null_corrs[s] = cca_test_corrs(U_s, V_s)

        thresh = np.nanquantile(null_corrs, config.shuffle_quantile, axis=0)
        for k in range(config.max_rank):
            df_cca_null_rows.append({
                'fold': fold_i,
                'component': k + 1,
                'corr_thresh': float(thresh[k])
            })

        # ------------------ metrics by rank ------------------
        for r in range(1, config.max_rank + 1):
            # RRR
            Yhat_rrr = Xte_s @ B_by_rank[r]
            ve_rrr = _variance_explained(Yte_s, Yhat_rrr)

            df_rank_rows.append({
                'fold': fold_i,
                'rank': r,
                'model': 'rrr',
                've_pred_y_from_x': ve_rrr
            })

            # CCA spectrum
            corrs = cca_test_corrs(Ute[:, :r], Vte[:, :r])
            for k, c in enumerate(corrs, start=1):
                df_cca_spec_rows.append({
                    'fold': fold_i,
                    'rank': r,
                    'component': k,
                    'corr_test': c
                })

            # CCA predictive VE
            Yhat_cca = cca_predict_Y_from_X(
                cca, Xtr_s, Ytr_s, Xte_s, rank=r
            )
            ve_cca = _variance_explained(Yte_s, Yhat_cca)

            df_rank_rows.append({
                'fold': fold_i,
                'rank': r,
                'model': 'cca',
                've_pred_y_from_x': ve_cca
            })

        # ------------------ store latents (max_rank) ------------------
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

        latents_by_fold[fold_i] = pd.concat([d_obs, d_pred], ignore_index=True)

    return {
        'df_rank': pd.DataFrame(df_rank_rows),
        'df_cca_spectrum': pd.DataFrame(df_cca_spec_rows),
        'df_cca_null': pd.DataFrame(df_cca_null_rows),
        'latents_by_fold': latents_by_fold,
        'cca_by_fold': cca_by_fold,
        'rrr_B_by_fold': rrr_B_by_fold
    }


def get_latents_for_model(
    *,
    model: str,                  # 'cca' | 'rrr'
    rank: int,
    fold: int,
    parity_res: dict,             # output of run_parity_cv
    X: np.ndarray,
    Y: np.ndarray,
    trial_ids: np.ndarray,
    time_idx: np.ndarray,
    cond: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Extract latent trajectories for a given model / rank / fold.

    Returns tidy DataFrame compatible with plot_shared_components:

        trial | time | cond | view | can1 | ... | canR

    view:
        'obs'  = X-side latent
        'pred' = Y-side latent / prediction

    Notes
    -----
    - CCA:
        obs  = canonical variables from X
        pred = canonical variables from Y
    - RRR:
        obs  = X @ B_r
        pred = X @ B_r   (predictive subspace; same as obs by construction)
        Note: RRR does not define a shared latent space; obs/pred are identical.
    """
    assert model in ('cca', 'rrr')
    assert rank >= 1

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    trial_ids = np.asarray(trial_ids)
    time_idx = np.asarray(time_idx)

    if cond is None:
        cond = np.array(['all'] * len(time_idx))
    else:
        cond = np.asarray(cond)

    # ---------- recover CV split ----------
    splits = make_group_splits(
        trial_ids, parity_res['df_rank']['fold'].nunique())
    tr_idx, te_idx = splits[fold]

    Xtr, Xte = X[tr_idx], X[te_idx]
    Ytr, Yte = Y[tr_idx], Y[te_idx]

    # ---------- standardize using TRAIN ----------
    Xtr_s, mux, sdx = _zscore(Xtr)
    Xte_s = (Xte - mux) / sdx

    Ytr_s, muy, sdy = _zscore(Ytr)
    Yte_s = (Yte - muy) / sdy

    # ---------- extract latents ----------
    if model == 'cca':
        cca = parity_res['cca_by_fold'][fold]
        Ute, Vte = cca_transform(cca, Xte_s, Yte_s)

        Z_obs = Ute[:, :rank]
        Z_pred = Vte[:, :rank]

    else:  # RRR
        B = parity_res['rrr_B_by_fold'][fold][rank]
        Z_obs = Xte_s @ B
        Z_pred = Z_obs.copy()

    # ---------- build tidy DataFrame ----------
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
