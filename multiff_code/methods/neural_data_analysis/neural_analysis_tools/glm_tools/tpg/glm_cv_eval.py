# =============================
# FILE: glm_cv_eval.py
# =============================
"""Trial-wise cross-validation and quick scoring utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_fit import fit_poisson_glm_trials, predict_mu, poisson_deviance, pseudo_R2


def trialwise_folds(
    trial_ids: np.ndarray,
    n_splits: int,
    *,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition trials into ``n_splits`` (train, val) boolean masks.

    Ensures that *all bins* from a given trial land in the same split.
    """
    rng = np.random.default_rng(random_state)
    # stable unique
    trials = np.array(list(dict.fromkeys(np.asarray(trial_ids))))
    if shuffle:
        trials = trials.copy()
        rng.shuffle(trials)
    folds = np.array_split(trials, n_splits)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_trials = set(folds[k])
        val_mask = np.isin(trial_ids, list(val_trials))
        train_mask = ~val_mask
        out.append((train_mask, val_mask))
    return out


def fit_and_score_cv(
    design_df: pd.DataFrame,
    y: np.ndarray,
    dt: float,
    trial_ids: np.ndarray,
    *,
    n_splits: int = 5,
    l2: float = 0.0,
    use_trial_FE: bool = True,
    cluster_se: bool = False,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Run trial-wise K-fold CV and return per-fold deviance and pseudo-R^2.

    Notes
    -----
    ``use_trial_FE`` is not used here because the design is already constructed.
    Keep it in the signature for symmetry with higher-level wrappers if needed.
    """
    folds = trialwise_folds(trial_ids, n_splits,
                            shuffle=True, random_state=random_state)
    rows = []
    for i, (train_mask, val_mask) in enumerate(folds, start=1):
        X_train = design_df.iloc[train_mask]
        y_train = y[train_mask]

        X_val = design_df.iloc[val_mask]
        y_val = y[val_mask]

        res = fit_poisson_glm_trials(
            X_train, y_train, dt, trial_ids[train_mask], add_const=True, l2=l2, cluster_se=False
        )
        mu_val = predict_mu(res, X_val, dt)
        mu_null = np.full_like(y_val, y_train.mean(), dtype=float)

        fold_dev = poisson_deviance(y_val, mu_val)
        fold_r2 = pseudo_R2(y_val, mu_val, mu_null)
        rows.append({
            "fold": i,
            "val_deviance": fold_dev,
            "val_pseudo_R2": fold_r2,
            "n_val_bins": int(val_mask.sum()),
        })
    return pd.DataFrame(rows)
