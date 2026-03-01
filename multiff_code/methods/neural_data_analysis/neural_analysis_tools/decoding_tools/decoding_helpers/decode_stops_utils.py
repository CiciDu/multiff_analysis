from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
    _build_folds,
)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def gaussian_kernel(width: int) -> np.ndarray:
    if width <= 0:
        return np.array([1.0], dtype=float)
    t = np.arange(-2 * width, 2 * width + 1)
    h = np.exp(-(t ** 2) / (2.0 * (width ** 2)))
    return h / np.sum(h)


def smooth_signal(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.asarray(x)
    h = gaussian_kernel(int(width))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.convolve(x, h, mode="same")
    return np.apply_along_axis(lambda z: np.convolve(z, h, mode="same"), axis=0, arr=x)


def build_group_lengths(groups: Sequence) -> Tuple[np.ndarray, List[int]]:
    groups = np.asarray(groups)
    if groups.size == 0:
        return np.array([]), []
    unique_order = []
    seen = set()
    for g in groups:
        if g not in seen:
            unique_order.append(g)
            seen.add(g)
    lengths = [int(np.sum(groups == g)) for g in unique_order]
    return np.asarray(unique_order), lengths


def split_by_lengths(vec: np.ndarray, lengths: Sequence[int]) -> List[np.ndarray]:
    vec = np.asarray(vec)
    out: List[np.ndarray] = []
    start = 0
    for ln in lengths:
        out.append(vec[start : start + int(ln)])
        start += int(ln)
    return out


def fit_linear_decoder(y_neural: np.ndarray, x_true: np.ndarray, width: int) -> Dict:
    y_fit = smooth_signal(y_neural, int(width))
    wts, _, _, _ = np.linalg.lstsq(y_fit, x_true, rcond=None)
    pred = y_fit @ wts
    err = float(np.sqrt(np.sum((pred - x_true) ** 2)))
    return {"width": int(width), "y_fit": y_fit, "wts": wts, "pred": pred, "error": err}


def tune_linear_decoder(
    y_neural: np.ndarray,
    x_true: np.ndarray,
    candidate_widths: Sequence[int],
) -> Dict:
    best = None
    print(f"Tuning linear decoder with candidate widths: {candidate_widths}")
    for w in candidate_widths:
        cur = fit_linear_decoder(y_neural, x_true, int(w))
        if (best is None) or (cur["error"] < best["error"]):
            best = cur
    return best


def _lengths_to_groups(lengths: Sequence[int]) -> np.ndarray:
    """Build group index for each sample from trial lengths."""
    groups = []
    for i, ln in enumerate(lengths):
        groups.extend([i] * int(ln))
    return np.array(groups)


def fit_linear_decoder_cv(
    y_neural: np.ndarray,
    x_true: np.ndarray,
    lengths: Sequence[int],
    width: int,
    n_splits: int = 5,
    cv_mode: str = "blocked_time_buffered",  # can be 'blocked_time_buffered', 'blocked_time', 'group_kfold'
    buffer_samples: int = 20,
    random_state: int = 0,
) -> Dict:
    """
    Fit linear decoder with cross-validation.

    cv_mode options (aligned with cv_decoding._build_folds):
      - 'group_kfold': GroupKFold by trial (no leakage across trials)
      - 'blocked_time_buffered': contiguous time blocks with buffer on both sides
      - 'blocked_time': forward-chaining (past → future)
      - default: shuffled KFold (not recommended for time series)
    """
    y_neural = np.asarray(y_neural, dtype=float)
    x_true = np.asarray(x_true, dtype=float).ravel()
    lengths = list(lengths)
    groups = _lengths_to_groups(lengths) if lengths else None

    n = len(x_true)
    n_trials = len(lengths)
    n_splits_eff = min(n_splits, n_trials) if groups is not None else n_splits
    if n_splits_eff < 2:
        return fit_linear_decoder(y_neural, x_true, width)

    # For group_kfold, pass groups; for blocked modes, pass None so cv_splitter is used
    groups_for_folds = groups if cv_mode == "group_kfold" else None
    splits = _build_folds(
        n,
        n_splits=n_splits_eff,
        groups=groups_for_folds,
        cv_splitter=cv_mode,
        random_state=random_state,
        buffer_samples=buffer_samples,
    )
    if not splits:
        return fit_linear_decoder(y_neural, x_true, width)

    pred = np.full_like(x_true, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        y_tr = y_neural[train_idx]
        x_tr = x_true[train_idx]
        y_te = y_neural[test_idx]

        y_tr_smooth = smooth_signal(y_tr, int(width))
        wts, _, _, _ = np.linalg.lstsq(y_tr_smooth, x_tr, rcond=None)
        y_te_smooth = smooth_signal(y_te, int(width))
        pred[test_idx] = y_te_smooth @ wts

    err = float(np.sqrt(np.nansum((pred - x_true) ** 2)))
    y_fit = smooth_signal(y_neural, int(width))
    wts_full, _, _, _ = np.linalg.lstsq(y_fit, x_true, rcond=None)
    return {
        "width": int(width),
        "y_fit": y_fit,
        "wts": wts_full,
        "pred": pred,
        "error": err,
    }


def tune_linear_decoder_cv(
    y_neural: np.ndarray,
    x_true: np.ndarray,
    lengths: Sequence[int],
    candidate_widths: Sequence[int],
    n_splits: int = 5,
    cv_mode: str = "blocked_time_buffered",  # can be 'blocked_time_buffered', 'blocked_time', 'group_kfold'
    buffer_samples: int = 20,
    random_state: int = 0,
) -> Dict:
    """Tune kernel width using CV; select width with lowest CV error."""
    best = None
    for w in candidate_widths:
        cur = fit_linear_decoder_cv(
            y_neural=y_neural,
            x_true=x_true,
            lengths=lengths,
            width=int(w),
            n_splits=n_splits,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            random_state=random_state,
        )
        if (best is None) or (cur["error"] < best["error"]):
            best = cur
    return best


def compute_canoncorr_block(
    x_task: np.ndarray,
    y_neural: np.ndarray,
    *,
    dt: float,
    filtwidth: int,
) -> Dict:
    x = np.asarray(x_task, dtype=float)
    y_raw = np.asarray(y_neural, dtype=float)
    x[np.isnan(x)] = 0.0

    y_smooth = smooth_signal(y_raw, int(filtwidth))
    y_rate = y_smooth / float(dt)

    n_comp = int(min(x.shape[1], y_rate.shape[1]))
    cca = CCA(n_components=max(1, n_comp), max_iter=2000)
    stim, resp = cca.fit_transform(x, y_rate)

    coeff = np.array([safe_corr(stim[:, i], resp[:, i]) for i in range(stim.shape[1])], dtype=float)
    coeff_sq = coeff ** 2
    dimensionality = (
        float((coeff_sq.sum() ** 2) / np.maximum((coeff_sq ** 2).sum(), 1e-12))
        if coeff_sq.size > 0
        else np.nan
    )

    return {
        "stim": stim,
        "resp": resp,
        "coeff": coeff,
        "dimensionality": dimensionality,
        "responsecorr_raw": np.corrcoef(y_raw, rowvar=False),
        "responsecorr_smooth": np.corrcoef(y_smooth, rowvar=False),
        "filtwidth": int(filtwidth),
    }
