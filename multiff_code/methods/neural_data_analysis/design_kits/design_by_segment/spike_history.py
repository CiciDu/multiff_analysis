from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.design_kits.design_by_segment import temporal_feats
from neural_data_analysis.design_kits.design_around_event.event_binning import (
    build_bin_assignments,
    bin_spikes_by_cluster,
    segment_windows_to_bins2d,
)
from neural_data_analysis.design_kits.design_around_event.event_binning import (
    rebin_all_segments_global_bins,
)

# ============================================================
# Padding-aware spike rebinning (LOCAL / GLOBAL)
# ============================================================
def rebin_spike_data_with_pad(
    *,
    spikes_df: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    t_max: float,
    mode: str,
    bin_width: float | None = None,
    bins_2d: np.ndarray | None = None,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
):
    """
    Apply backward padding in time, then rebin.
    Padding is truncated AFTER history construction.
    """

    if mode == 'local':
        dt = float(bin_width)
    else:
        dt = np.median(np.diff(bins_2d[:, 0]))

    n_pad_bins = int(np.ceil(t_max / dt))
    pad_time = n_pad_bins * dt

    # --------------------------------------------------
    # Pad segments backward in time
    # --------------------------------------------------
    new_seg_info_pad = new_seg_info.copy()
    new_seg_info_pad['new_seg_start_time'] -= pad_time
    new_seg_info_pad['new_seg_duration'] = (
        new_seg_info_pad['new_seg_end_time']
        - new_seg_info_pad['new_seg_start_time']
    )

    # --------------------------------------------------
    # Rebin (IDENTICAL global or local)
    # --------------------------------------------------
    design_pad = rebin_spike_data(
        spikes_df=spikes_df,
        new_seg_info=new_seg_info_pad,
        mode=mode,
        bin_width=bin_width,
        bins_2d=bins_2d,
        time_col=time_col,
        cluster_col=cluster_col,
    )

    return design_pad, dict(
        n_pad_bins=n_pad_bins,
        pad_time=pad_time,
        dt=dt,
    )


# ============================================================
# Rebinning backends
# ============================================================
def rebin_spike_data(
    *,
    spikes_df: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    mode: str,
    bin_width: float | None = None,
    bins_2d: np.ndarray | None = None,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
):
    """
    Unified spike rebinning.

    mode:
      - 'local'  : segment-local bins (unchanged)
      - 'global' : IDENTICAL to rebin_all_segments_global_bins
    """

    if mode == 'local':
        if bin_width is None:
            raise ValueError('bin_width required for local mode')

        bins_2d_local, meta = segment_windows_to_bins2d(
            new_seg_info,
            bin_width=bin_width,
        )

        if bins_2d_local.size == 0:
            return pd.DataFrame()

        counts, cluster_ids = bin_spikes_by_cluster(
            spikes_df[[time_col, cluster_col]],
            bins_2d_local,
            time_col=time_col,
            cluster_col=cluster_col,
            assume_sorted_bins=True,
            check_nonoverlap=False,
        )

        out = meta[['new_segment', 'new_bin']].copy()
        for j, cid in enumerate(cluster_ids):
            out[f'cluster_{cid}'] = counts[:, j]

        return out.merge(
            new_seg_info[
                ['new_segment', 'new_seg_start_time',
                 'new_seg_end_time', 'new_seg_duration']
            ],
            on='new_segment',
            how='left',
            validate='many_to_one',
        )

    # --------------------------------------------------
    # GLOBAL (IDENTICAL path)
    # --------------------------------------------------
    if mode == 'global':
        if bins_2d is None:
            raise ValueError('bins_2d required for global mode')

        return rebin_spike_data_global_bins_identical(
            spikes_df=spikes_df,
            new_seg_info=new_seg_info,
            bins_2d=bins_2d,
            time_col=time_col,
            cluster_col=cluster_col,
        )

    raise ValueError("mode must be 'local' or 'global'")


def rebin_spike_data_local_bins(
    spikes_df,
    new_seg_info,
    *,
    bin_width,
    time_col,
    cluster_col,
):
    """
    Original segment-local PSTH-style binning.
    """

    bins_2d, meta = segment_windows_to_bins2d(
        new_seg_info,
        bin_width=bin_width,
    )

    if bins_2d.size == 0:
        return pd.DataFrame()

    counts, cluster_ids = bin_spikes_by_cluster(
        spikes_df[[time_col, cluster_col]],
        bins_2d,
        time_col=time_col,
        cluster_col=cluster_col,
        assume_sorted_bins=True,
        check_nonoverlap=False,
    )

    out = meta[['new_segment', 'new_bin']].copy()
    for j, cid in enumerate(cluster_ids):
        out[f'cluster_{cid}'] = counts[:, j]

    out = out.merge(
        new_seg_info[
            ['new_segment', 'new_seg_start_time',
             'new_seg_end_time', 'new_seg_duration']
        ],
        on='new_segment',
        how='left',
        validate='many_to_one',
    )

    return out

def rebin_spike_data_global_bins(
    *,
    spikes_df: pd.DataFrame,
    new_seg_info: pd.DataFrame,
    bins_2d: np.ndarray,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
):
    """
    Global-bin spike rebinning that is IDENTICAL to
    rebin_all_segments_global_bins, up to padding.
    """

    if spikes_df.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # Convert spikes → wide left-hold timeseries
    # --------------------------------------------------
    wide = (
        spikes_df
        .assign(value=1.0)
        .pivot_table(
            index=time_col,
            columns=cluster_col,
            values='value',
            aggfunc='sum',
            fill_value=0.0,
        )
        .reset_index()
    )

    wide.columns = [
        time_col if c == time_col else f'cluster_{c}'
        for c in wide.columns
    ]

    # --------------------------------------------------
    # Delegate to canonical global binning
    # --------------------------------------------------
    out = rebin_all_segments_global_bins(
        wide,
        new_seg_info,
        bins_2d=bins_2d,
        time_col=time_col,
        how='sum',                 # spike counts
        respect_old_segment=False, # spikes have no old segment
        add_bin_edges=False,
        require_full_bin=False,
        add_support_duration=False,
    )

    return out


# ============================================================
# Spike history (UNCHANGED)
# ============================================================

def compute_spike_history_designs(
    design_pad: pd.DataFrame,
    bin_info: pd.DataFrame,
    *,
    n_pad_bins: int,
    dt: float,
    t_max: float,
    n_basis: int = 5,
    t_min: float | None = None,
    edge: str = 'zero',
):

    if t_min is None:
        t_min = dt

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
    colnames = {}

    for col in spike_cols:
        hist = temporal_feats.lagged_design_from_signal_trials(
            design_pad[col].to_numpy(),
            basis,
            trial_ids,
            edge=edge,
        )

        K = basis.shape[1]
        colnames[col] = [f'{col}:b0:{k}' for k in range(K)]

        hist = pd.DataFrame(hist, columns=colnames[col])
        hist['new_segment'] = design_pad['new_segment'].values
        hist['new_bin'] = design_pad['new_bin'].values

        hist = truncate_history_pad(hist, n_pad_bins)

        hist = bin_info[['new_segment', 'new_bin']].merge(
            hist,
            on=['new_segment', 'new_bin'],
            how='left',
            validate='one_to_one',
        )

        X_hist[col] = hist

    return X_hist, basis, colnames


def truncate_history_pad(df: pd.DataFrame, n_pad_bins: int):
    keep = df['new_bin'].to_numpy() >= n_pad_bins
    df = df.loc[keep].copy()
    df['new_bin'] -= n_pad_bins
    df.reset_index(drop=True, inplace=True)
    return df
