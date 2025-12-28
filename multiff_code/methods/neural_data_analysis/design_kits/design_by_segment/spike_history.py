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
    rebinned_x_var: pd.DataFrame,
    *,
    n_pad_bins: int,
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
    colnames : dict (optional; returned iff with_colnames=True)
        {cluster_col: list[str]} column names in 'bjk' style like add_spike_history
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
    colnames: dict[str, list[str]] = {}

    K = basis.shape[1]

    for col in spike_cols:
        hist = temporal_feats.lagged_design_from_signal_trials(
            design_pad[col].to_numpy(),
            basis,
            trial_ids,
            edge=edge,
        )

        colnames[col] = [f'{col}:b0:{k}' for k in range(K)]

        hist = pd.DataFrame(hist, columns=colnames[col])
        hist['new_segment'] = design_pad['new_segment'].values
        hist['new_bin'] = design_pad['new_bin'].values

        hist = truncate_history_pad(hist, n_pad_bins)

        hist = rebinned_x_var[['new_segment', 'new_bin']].merge(
            hist,
            on=['new_segment', 'new_bin'],
            how='left',
            validate='one_to_one',
        )

        X_hist[col] = hist

        assert hist.shape[0] == rebinned_x_var.shape[0]
        # assert hist['new_segment'].equals(rebinned_x_var['new_segment'])
        # assert hist['new_bin'].equals(rebinned_x_var['new_bin'])

    return X_hist, basis, colnames

def add_spike_history_to_design(
    design_df: pd.DataFrame,
    colnames: dict[str, list[str]],
    X_hist: dict,
    target_col: str,
    *,
    include_self: bool = True,
    cross_neurons: list[str] | None = None,
    meta_groups: dict | None = None,
):
    """
    Assemble a GLM design matrix for one target neuron by adding spike-history
    regressors (self and optional cross-neuron history).

    Optionally updates meta['groups'] to group history columns by predictor.
    """

    design_df = design_df.copy()

    # --- self-history ---
    if include_self:
        target_cols = colnames[target_col]
        design_df[target_cols] = X_hist[target_col][target_cols].values

    # --- cross-neuron history ---
    if cross_neurons is not None:
        for neuron in cross_neurons:
            neuron_cols = colnames[neuron]
            design_df[neuron_cols] = X_hist[neuron][neuron_cols].values

    # --- optional: update meta groups ---
    if meta_groups is not None:

        if include_self:
            meta_groups.setdefault(target_col, []).extend(colnames[target_col])

        if cross_neurons is not None:
            for neuron in cross_neurons:
                meta_groups.setdefault(neuron, []).extend(colnames[neuron])

        return design_df, meta_groups

    return design_df, None


def truncate_history_pad(
    df: pd.DataFrame,
    n_pad_bins: int,
):
    """
    Truncate padded bins after history computation.
    """

    keep = df['new_bin'].to_numpy() >= n_pad_bins

    df = df.loc[keep].copy()
    df['new_bin'] -= n_pad_bins
    df.reset_index(drop=True, inplace=True)

    return df
