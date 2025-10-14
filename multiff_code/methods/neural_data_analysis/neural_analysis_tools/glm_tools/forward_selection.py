# forward_block_select.py
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ---------- scoring & guards ----------


def _condition_number(X: np.ndarray) -> float:
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0] / (s[-1] + 1e-12))


def _fit_glm(y, X, offset_log):
    model = sm.GLM(y, X, family=sm.families.Poisson(),
                   offset=offset_log if np.ndim(offset_log) else None)
    return model.fit(method='newton', maxiter=50, disp=0)


def _poisson_deviance(y, mu, eps=1e-12) -> float:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    return float(2.0 * np.nansum(y * (np.log((y + eps) / (mu + eps))) - (y - mu)))


def _cv_dev_poisson(y: np.ndarray,
                    X: pd.DataFrame,
                    trial_ids: np.ndarray,
                    offset_log: np.ndarray | float,
                    nfolds: int = 5,
                    rng: int = 0) -> float:
    rs = np.random.RandomState(rng)
    uniq = np.unique(trial_ids)
    rs.shuffle(uniq)
    folds = np.array_split(uniq, nfolds)
    devs = []
    for hold in folds:
        test = np.isin(trial_ids, hold)
        train = ~test
        res = _fit_glm(y[train], X.iloc[train], offset_log if np.isscalar(
            offset_log) else offset_log[train])
        mu_test = res.predict(X.iloc[test],
                              offset=None if np.isscalar(offset_log) else offset_log[test])
        devs.append(_poisson_deviance(y[test], mu_test))
    return float(np.mean(devs))

# ---------- helpers ----------


def _build_X(df_X_full: pd.DataFrame, groups: Dict[str, List[str]], chosen: List[str]) -> pd.DataFrame:
    cols = []
    for g in chosen:
        cols.extend(groups[g])
    # keep order, drop duplicates while preserving leftmost
    seen = set()
    ordered = [c for c in cols if not (c in seen or seen.add(c))]
    X = df_X_full.loc[:, ordered].copy()
    # sanity: add const if user included group 'const' as a column name
    return X


@dataclass
class StepResult:
    step: int
    added_group: Optional[str]
    cv_stop_deviance: float
    rel_improve: float
    n_params: int
    cond_number: float

# ---------- main routine ----------


def forward_block_select(df_X_full: pd.DataFrame,
                         groups: Dict[str, List[str]],
                         y: np.ndarray,
                         trial_ids: np.ndarray,
                         offset_log: np.ndarray | float,
                         base_groups: List[str],
                         candidate_groups: List[str] | None = None,
                         nfolds: int = 5,
                         min_rel_improve: float = 0.002,
                         max_steps: int = 20,
                         rng: int = 0,
                         cond_thresh: float = 5e3) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Returns (X_final, selected_groups, log_df)
    """
    assert len(base_groups) > 0, 'Provide at least one base group'
    if candidate_groups is None:
        # all groups except base
        candidate_groups = [g for g in groups.keys() if g not in base_groups]

    # start with base
    chosen = base_groups[:]
    X_base = _build_X(df_X_full, groups, chosen)
    cn0 = _condition_number(np.asarray(X_base, float, order='F'))
    if cn0 > cond_thresh:
        raise RuntimeError(
            f'Base design ill-conditioned (cond={cn0:.1f}). Consider removing/reducing base.')

    base_cv = _cv_dev_poisson(y, X_base, trial_ids,
                              offset_log, nfolds=nfolds, rng=rng)
    log: List[StepResult] = [StepResult(step=0, added_group=None,
                                        cv_stop_deviance=base_cv, rel_improve=0.0,
                                        n_params=X_base.shape[1], cond_number=cn0)]

    remaining = candidate_groups[:]
    best = base_cv
    step = 0

    while step < max_steps and remaining:
        step += 1
        trial_scores = []
        for g in remaining:
            X_try = _build_X(df_X_full, groups, chosen + [g])
            cn = _condition_number(np.asarray(X_try, float, order='F'))
            if cn > cond_thresh:
                # skip this group for now
                trial_scores.append((g, np.inf, cn, X_try.shape[1]))
                continue
            score = _cv_dev_poisson(
                y, X_try, trial_ids, offset_log, nfolds=nfolds, rng=rng)
            trial_scores.append((g, score, cn, X_try.shape[1]))

        # pick best candidate
        g_best, score_best, cn_best, p_best = min(
            trial_scores, key=lambda t: t[1])
        rel = (best - score_best) / max(best, 1e-12)
        if not np.isfinite(score_best) or rel < min_rel_improve:
            break

        # accept
        chosen.append(g_best)
        best = score_best
        log.append(StepResult(step=step, added_group=g_best,
                              cv_stop_deviance=score_best, rel_improve=rel,
                              n_params=p_best, cond_number=cn_best))
        remaining = [g for g in remaining if g != g_best]

    X_final = _build_X(df_X_full, groups, chosen)
    log_df = pd.DataFrame([s.__dict__ for s in log])
    return X_final, chosen, log_df
