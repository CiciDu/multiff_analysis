from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.cross_decomposition import CCA

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import population_analysis_utils


def gaussian_kernel(width: int) -> np.ndarray:
    width = int(width)
    if width <= 0:
        return np.array([1.0], dtype=float)
    return np.asarray(population_analysis_utils.gaussian_kernel(width), dtype=float)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def split_by_lengths(values: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    values = np.asarray(values)
    out = []
    cursor = 0
    for n in lengths:
        n = int(n)
        out.append(values[cursor: cursor + n])
        cursor += n
    return out


def build_group_lengths(groups: Iterable) -> tuple[np.ndarray, list[int]]:
    groups = np.asarray(list(groups))
    if groups.size == 0:
        return groups, []
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(groups)]
    lengths = [int(e - s) for s, e in zip(starts, ends)]
    return groups[starts], lengths


def _smooth_neural(y_neural: np.ndarray, width: int, neural_cols_to_smooth: int | None = None) -> np.ndarray:
    y_neural = np.asarray(y_neural, dtype=float)
    if width <= 0:
        return y_neural
    y_sm = y_neural.copy()
    if neural_cols_to_smooth is None:
        y_sm = population_analysis_utils.smooth_signal(y_sm, width)
    else:
        n = int(neural_cols_to_smooth)
        y_sm[:, :n] = population_analysis_utils.smooth_signal(y_sm[:, :n], width)
    return y_sm


def compute_canoncorr_block(
    *,
    x_task: np.ndarray,
    y_neural: np.ndarray,
    dt: float,
    filtwidth: int = 0,
    neural_cols_to_smooth: int | None = None,
) -> dict:
    x = np.asarray(x_task, dtype=float)
    y_raw = np.asarray(y_neural, dtype=float)
    y_sm = _smooth_neural(y_raw, int(filtwidth), neural_cols_to_smooth=neural_cols_to_smooth)
    y_rate = y_sm / float(dt)

    n_comp = int(min(x.shape[1], y_rate.shape[1]))
    cca = CCA(n_components=max(1, n_comp), max_iter=2000)
    x_c, y_c = cca.fit_transform(x, y_rate)
    coeff = np.array([safe_corr(x_c[:, i], y_c[:, i]) for i in range(x_c.shape[1])], dtype=float)
    coeff_sq = coeff ** 2
    dimensionality = float((coeff_sq.sum() ** 2) / np.maximum((coeff_sq ** 2).sum(), 1e-12))

    return {
        "stim": x_c,
        "resp": y_c,
        "coeff": coeff,
        "dimensionality": dimensionality,
        "responsecorr_raw": np.corrcoef(y_raw, rowvar=False),
        "responsecorr_smooth": np.corrcoef(y_sm, rowvar=False),
    }


def fit_linear_decoder_cv(
    *,
    y_neural: np.ndarray,
    x_true: np.ndarray,
    lengths: Sequence[int],
    width: int,
    n_splits: int,
    cv_mode: str,
    buffer_samples: int,
    neural_cols_to_smooth: int | None = None,
) -> dict:
    x_true = np.asarray(x_true, dtype=float).ravel()
    y_sm = _smooth_neural(np.asarray(y_neural, dtype=float), int(width), neural_cols_to_smooth=neural_cols_to_smooth)
    n = len(x_true)
    if len(y_sm) != n:
        raise ValueError("x_true and y_neural must have matching length.")

    if cv_mode == "group_kfold" and lengths:
        trial_ids = np.concatenate([np.full(int(l), i, dtype=int) for i, l in enumerate(lengths)])
    else:
        trial_ids = None

    splits = cv_decoding._build_folds(
        n,
        n_splits=n_splits,
        groups=trial_ids,
        cv_splitter=cv_mode,
        buffer_samples=buffer_samples,
    )
    pred = np.full(n, np.nan, dtype=float)
    wts = []
    for train_idx, test_idx in splits:
        X_tr = y_sm[train_idx]
        X_te = y_sm[test_idx]
        y_tr = x_true[train_idx]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        pred[test_idx] = X_te.dot(coef)
        wts.append(np.asarray(coef))

    coef_full, *_ = np.linalg.lstsq(y_sm, x_true, rcond=None)
    return {"width": int(width), "wts": coef_full, "pred": pred, "corr": safe_corr(x_true, pred), "fold_wts": wts}


def tune_linear_decoder_cv(
    *,
    y_neural: np.ndarray,
    x_true: np.ndarray,
    lengths: Sequence[int],
    candidate_widths: Sequence[int],
    n_splits: int,
    cv_mode: str,
    buffer_samples: int,
    neural_cols_to_smooth: int | None = None,
) -> dict:
    best = None
    width_scores = {}
    for w in candidate_widths:
        out = fit_linear_decoder_cv(
            y_neural=y_neural,
            x_true=x_true,
            lengths=lengths,
            width=int(w),
            n_splits=n_splits,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            neural_cols_to_smooth=neural_cols_to_smooth,
        )
        score = out["corr"]
        width_scores[int(w)] = score
        if best is None or (np.isfinite(score) and score > best["corr"]):
            best = out

    if best is None:
        raise RuntimeError("No candidate width produced a valid decoder result.")
    best = dict(best)
    best["width_scores"] = width_scores
    return best
