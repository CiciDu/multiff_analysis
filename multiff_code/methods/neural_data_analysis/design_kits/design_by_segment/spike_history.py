from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Tuple, List, Mapping

import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import BSpline

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from neural_data_analysis.design_kits.design_by_segment import temporal_feats


def rebin_spike_data_with_pad(
    spikes_df: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    bin_width: float,
    *,
    t_max: float,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
):
    """
    Rebin spike data with backward padding in whole bins.
    No history is computed here.
    """

    dt = float(bin_width)

    # --------------------------------------------------
    # 1) Compute padding in whole bins
    # --------------------------------------------------
    n_pad_bins = int(np.ceil(t_max / dt))
    pad_time = n_pad_bins * dt

    # --------------------------------------------------
    # 2) Pad segments backward only
    # --------------------------------------------------
    new_seg_info_pad = new_seg_info.copy()
    new_seg_info_pad['new_seg_start_time'] -= pad_time
    new_seg_info_pad['new_seg_duration'] = (
        new_seg_info_pad['new_seg_end_time']
        - new_seg_info_pad['new_seg_start_time']
    )

    # --------------------------------------------------
    # 3) Rebin spikes
    # --------------------------------------------------
    design_pad = pn_utils.rebin_spike_data(
        spikes_df=spikes_df,
        new_seg_info=new_seg_info_pad,
        bin_width=dt,
        time_col=time_col,
        cluster_col=cluster_col,
    )

    pad_info = dict(
        n_pad_bins=n_pad_bins,
        pad_time=pad_time,
    )

    return design_pad, pad_info

def compute_spike_history_designs(
    design_pad: pd.DataFrame,
    *,
    dt: float,
    t_max: float,
    n_basis: int = 5,
    t_min: float | None = None,
    edge: str = 'zero',
):
    """
    Precompute spike-history design matrices for all spike channels.

    Returns
    -------
    X_hist : dict
        {cluster_col: (T × K) history matrix}
    basis : np.ndarray
        (L × K) history basis
    """

    if t_min is None:
        t_min = dt

    # --------------------------------------------------
    # Build basis ONCE
    # --------------------------------------------------
    _, basis = glm_bases.raised_cosine_basis(
        n_basis=n_basis,
        t_max=t_max,
        dt=dt,
        t_min=t_min,
        log_spaced=True,
    )

    trial_ids = design_pad['new_segment'].to_numpy()

    spike_cols = [c for c in design_pad.columns if c.startswith('cluster_')]

    X_hist = {}

    for col in spike_cols:
        X_hist[col] = temporal_feats.lagged_design_from_signal_trials(
            design_pad[col].to_numpy(),
            basis,
            trial_ids,
            edge=edge,
        )

    return X_hist, basis

def build_glm_design_for_target(
    design_pad: pd.DataFrame,
    X_hist: dict,
    target_col: str,
    *,
    include_self: bool = True,
    cross_neurons: list[str] | None = None,
    task_cols: list[str] | None = None,
):
    """
    Assemble GLM design matrix for one target neuron.
    """

    blocks = []

    # task covariates
    if task_cols is not None:
        blocks.append(design_pad[task_cols].to_numpy())

    # self-history
    if include_self:
        blocks.append(X_hist[target_col])

    # cross-history
    if cross_neurons is not None:
        for col in cross_neurons:
            blocks.append(X_hist[col])

    X = np.hstack(blocks)
    y = design_pad[target_col].to_numpy()

    return X, y

def truncate_history_pad(
    design_pad: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_pad_bins: int,
):
    """
    Truncate padded bins after history computation.
    """

    keep = design_pad['new_bin'].to_numpy() >= n_pad_bins

    X = X[keep]
    y = y[keep]

    design = design_pad.loc[keep].copy()
    design['new_bin'] -= n_pad_bins
    design.reset_index(drop=True, inplace=True)

    return design, X, y
