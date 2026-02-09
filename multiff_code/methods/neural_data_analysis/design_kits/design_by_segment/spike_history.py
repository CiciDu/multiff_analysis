from typing import Optional

import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_around_event.event_binning import (
    bin_spikes_by_cluster,
)
from neural_data_analysis.design_kits.design_by_segment import temporal_feats
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases

# ============================================================
# Padding utilities
# ============================================================


def _pad_bin_table_for_history(
    bin_df: pd.DataFrame,
    *,
    dt: float,
    t_max: float,
):
    """
    Prepend padding bins (in time) per segment for spike-history support.

    Parameters
    ----------
    bin_df : DataFrame
        Must contain ['new_segment', 'new_bin', 'bin_left', 'bin_right']
    """

    bin_dt = _assert_dt_matches_bins(bin_df, dt)
    dt = bin_dt

    n_pad_bins = int(np.ceil(t_max / dt))
    pad_time = n_pad_bins * dt

    pad_blocks = []

    for seg_id, g in bin_df.groupby('new_segment'):
        g = g.sort_values('new_bin')

        first = g.iloc[0]
        left0 = float(first['bin_left'])

        pad_lefts = left0 - dt * np.arange(n_pad_bins, 0, -1)
        pad_rights = pad_lefts + dt

        pad = pd.DataFrame({
            'new_segment': seg_id,
            'new_bin': np.arange(-n_pad_bins, 0, dtype=int),
            'bin_left': pad_lefts,
            'bin_right': pad_rights,
        })

        pad_blocks.append(pd.concat([pad, g], ignore_index=True))

    bin_df_pad = pd.concat(pad_blocks, ignore_index=True)

    return bin_df_pad, dict(
        n_pad_bins=n_pad_bins,
        pad_time=pad_time,
        dt=dt,
    )


# ============================================================
# Spike rebinning using explicit bins
# ============================================================

def _rebin_spikes_from_bin_table(
    *,
    spikes_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    time_col: str = 'time',
    cluster_col: str = 'cluster',
):
    """
    Rebin spikes using an explicit bin table.
    Bin identity comes entirely from bin_df.
    """

    bins_2d = bin_df[['bin_left', 'bin_right']].to_numpy()

    counts, cluster_ids = bin_spikes_by_cluster(
        spikes_df[[time_col, cluster_col]],
        bins_2d,
        time_col=time_col,
        cluster_col=cluster_col,
        assume_sorted_bins=True,
        check_nonoverlap=False,
    )

    bin_df = bin_df.sort_values(
        ['new_segment', 'new_bin']).reset_index(drop=True)

    for j, cid in enumerate(cluster_ids):
        bin_df[f'cluster_{cid}'] = counts[:, j]

    return bin_df


# ============================================================
# Public entry point
# ============================================================

def compute_spike_history_designs(
    *,
    spikes_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    dt: float,
    t_max: float,
    n_basis: int = 5,
    t_min: Optional[float] = None,
    edge: str = 'zero',
):
    """
    Build spike-history design matrices using externally defined bins.

    Parameters
    ----------
    spikes_df : DataFrame
        Must contain ['time', 'cluster']
    bin_df : DataFrame
        Must contain ['new_segment', 'new_bin', 'bin_left', 'bin_right']
        Produced by rebin_all_segments (local or global) with add_bin_edges=True.
    """

    if t_min is None:
        t_min = dt

    # --------------------------------------------------
    # 1) Pad bins
    # --------------------------------------------------
    bin_df_pad, pad_info = _pad_bin_table_for_history(
        bin_df,
        dt=dt,
        t_max=t_max,
    )

    # --------------------------------------------------
    # 2) Rebin spikes
    # --------------------------------------------------
    design_pad = _rebin_spikes_from_bin_table(
        spikes_df=spikes_df,
        bin_df=bin_df_pad,
    )

    # --------------------------------------------------
    # 3) Build history basis (ONCE)
    # --------------------------------------------------
    _, basis = glm_bases.raised_log_cosine_basis(
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

    n_pad_bins = pad_info['n_pad_bins']
    K = basis.shape[1]

    # --------------------------------------------------
    # 4) History per neuron
    # --------------------------------------------------
    for col in spike_cols:
        hist = temporal_feats.lagged_design_from_signal_trials(
            design_pad[col].to_numpy(),
            basis,
            trial_ids,
            edge=edge,
        )

        cols = [f'{col}:b0:{k}' for k in range(K)]
        colnames[col] = cols

        hist = pd.DataFrame(hist, columns=cols)
        hist['new_segment'] = design_pad['new_segment'].values
        hist['new_bin'] = design_pad['new_bin'].values

        # --------------------------------------------------
        # 5) Truncate padding
        # Padding bins have new_bin < 0 and are removed after history construction
        # --------------------------------------------------
        keep = hist['new_bin'] >= 0
        hist = hist.loc[keep].copy()
        hist.reset_index(drop=True, inplace=True)

        X_hist[col] = hist

    # pick one representative neuron to check that the sequence of 'new_segment', 'new_bin' is the same as the original bin_df
    rep_col = spike_cols[0]
    rep_hist = X_hist[rep_col]

    assert np.array_equal(
        rep_hist['new_segment'].values,
        bin_df['new_segment'].values,
    )
    assert np.array_equal(
        rep_hist['new_bin'].values,
        bin_df['new_bin'].values,
    )

    return X_hist, basis, colnames


def _assert_dt_matches_bins(bin_df, dt, *, tol=1e-9):
    widths = (bin_df['bin_right'] - bin_df['bin_left']).to_numpy(dtype=float)

    # ignore pathological bins (e.g. empty or NaN)
    widths = widths[np.isfinite(widths)]

    unique = np.unique(np.round(widths / tol) * tol)

    if unique.size != 1:
        raise ValueError(
            f'Bin widths are not uniform: {unique}'
        )

    bin_dt = unique[0]

    if not np.isclose(bin_dt, dt, rtol=0, atol=tol):
        raise ValueError(
            f'dt={dt} does not match bin width from bin_df ({bin_dt})'
        )

    # also assert monotonic bins per segment
    for seg, g in bin_df.groupby('new_segment'):
        if not np.all(np.diff(g['bin_left']) > 0):
            raise ValueError(f'Bins not strictly increasing in segment {seg}')

    return bin_dt


def add_spike_history_to_design(
    design_df: pd.DataFrame,
    colnames: dict[str, list[str]],
    X_hist: dict,
    target_col: str,
    *,
    include_self: bool = True,
    cross_neurons: Optional[list[str]] = None,
    meta_groups: Optional[dict] = None,
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


def build_design_with_spike_history_from_bins(
    *,
    spikes_df,
    bin_df,
    X_pruned,
    meta_groups,
    dt,
    t_max,
    n_basis=5,
):
    """
    Compute spike-history regressors from an explicit bin table and
    add them to an existing design matrix.

    Parameters
    ----------
    spikes_df : DataFrame
        Must contain ['time', 'cluster'].
    bin_df : DataFrame
        Must contain ['new_segment', 'new_bin', 'bin_left', 'bin_right'].
    X_pruned : DataFrame
        Base design matrix with ['new_segment', 'new_bin'].
    meta_groups : dict
        Meta groups dict to be updated in-place.
    dt : float
        Bin width.
    t_max : float
        Maximum history length.
    """

    # --------------------------------------------------
    # 1) Compute spike-history designs
    # --------------------------------------------------
    X_hist, basis, colnames = compute_spike_history_designs(
        spikes_df=spikes_df,
        bin_df=bin_df,
        dt=dt,
        t_max=t_max,
        n_basis=n_basis,
    )

    # --------------------------------------------------
    # 2) Choose target + cross neurons
    # --------------------------------------------------
    spike_cols = list(colnames.keys())
    if len(spike_cols) == 0:
        raise ValueError('No spike columns found in colnames')

    # It is sufficient to do this once
    target_col = spike_cols[0]
    cross_neurons = [c for c in spike_cols if c != target_col]

    # --------------------------------------------------
    # 3) Add history to design
    # --------------------------------------------------
    design_w_history, meta_groups = add_spike_history_to_design(
        design_df=X_pruned,
        colnames=colnames,
        X_hist=X_hist,
        target_col=target_col,
        include_self=True,
        cross_neurons=cross_neurons,
        meta_groups=meta_groups,
    )

    return design_w_history, basis, colnames, meta_groups


def make_bin_df_from_stop_meta(meta_used):
    return (
        meta_used[['event_id', 'k_within_seg', 't_left', 't_right']]
        .copy()
        .rename(columns={
            'event_id': 'new_segment',
            'k_within_seg': 'new_bin',
            't_left': 'bin_left',
            't_right': 'bin_right',
        })
        .drop_duplicates()
        .sort_values(['new_segment', 'new_bin'])
        .reset_index(drop=True)
    )
