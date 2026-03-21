from typing import Optional, Tuple

import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_around_event.event_binning import (
    bin_spikes_by_cluster,
)
from neural_data_analysis.design_kits.design_by_segment import temporal_feats
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import process_encode_design 
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

    cluster_data = {
        f'cluster_{cid}': counts[:, j]
        for j, cid in enumerate(cluster_ids)
    }
    cluster_df = pd.DataFrame(cluster_data)
    bin_df = pd.concat([bin_df, cluster_df], axis=1)

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
    n_basis: int = 20,
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
        hard_start_zero=True,
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
            dt=dt,
            t_min=t_min,
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

def compute_coupling_designs(
    *,
    spikes_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    dt: float,
    t_max: float = 1.375,
    t_min: float = 0.0,
    n_basis: int = 20,
    edge: str = 'zero',
):
    """
    Build coupling (cross-neuron) spike-history design matrices.

    This is a thin wrapper around compute_spike_history_designs,
    using different defaults and renaming columns to indicate coupling.
    """

    # --------------------------------------------------
    # 1) Reuse spike-history builder
    # --------------------------------------------------
    X_hist, basis, colnames = compute_spike_history_designs(
        spikes_df=spikes_df,
        bin_df=bin_df,
        dt=dt,
        t_max=t_max,
        n_basis=n_basis,
        t_min=t_min,
        edge=edge,
    )

    # --------------------------------------------------
    # 2) Rename columns from ':b0:' → ':cpl:'
    # --------------------------------------------------
    X_hist_cpl = {}
    colnames_cpl = {}

    for col, df in X_hist.items():

        old_cols = colnames[col]
        new_cols = [c.replace(':b0:', ':cpl:') for c in old_cols]

        # rename feature columns only
        rename_map = dict(zip(old_cols, new_cols))
        df = df.rename(columns=rename_map)

        X_hist_cpl[col] = df
        colnames_cpl[col] = new_cols

    return X_hist_cpl, basis, colnames_cpl


def add_coupling_to_design(
    design_df: pd.DataFrame,
    colnames: dict[str, list[str]],
    X_hist: dict,
    coupling_units: list[str],
    *,
    meta_groups: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Add coupling (cross-neuron) spike-history regressors to a design matrix.

    Parameters
    ----------
    design_df : DataFrame
        Design matrix with ['new_segment', 'new_bin']
    colnames : dict
        From compute_coupling_designs, maps cluster names to column lists
    X_hist : dict
        From compute_coupling_designs, coupling designs per neuron
    coupling_units : list of str
        Cluster column names (e.g. ['cluster_1', 'cluster_2']) to include
    meta_groups : dict, optional
        Updated in-place to group columns by predictor

    Returns
    -------
    design_df : DataFrame
        Design with coupling columns appended
    meta_groups : dict or None
        Updated meta_groups if provided
    """
    columns_to_add = []

    for neuron in coupling_units:
        if neuron not in X_hist:
            raise KeyError(
                f'Coupling unit {neuron} not in X_hist. '
                f'Available: {list(X_hist.keys())}'
            )
        neuron_cols = colnames[neuron]
        columns_to_add.append(X_hist[neuron][neuron_cols])

    if columns_to_add:
        design_df = pd.concat([design_df] + columns_to_add, axis=1)
    else:
        design_df = design_df.copy()

    if meta_groups is not None:
        for neuron in coupling_units:
            meta_groups.setdefault(neuron, []).extend(colnames[neuron])
        return design_df, meta_groups

    return design_df, None


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

    # Collect all columns to add at once to avoid DataFrame fragmentation
    columns_to_add = []

    # --- self-history ---
    if include_self:
        target_cols = colnames[target_col]
        columns_to_add.append(X_hist[target_col][target_cols])

    # --- cross-neuron history ---
    if cross_neurons is not None:
        for neuron in cross_neurons:
            neuron_cols = colnames[neuron]
            columns_to_add.append(X_hist[neuron][neuron_cols])

    # Concatenate all columns at once for better performance
    if columns_to_add:
        design_df = pd.concat([design_df] + columns_to_add, axis=1)
    else:
        design_df = design_df.copy()

    # --- optional: update meta groups ---
    if meta_groups is not None:
        if include_self:
            meta_groups.setdefault(target_col, []).extend(colnames[target_col])

        if cross_neurons is not None:
            for neuron in cross_neurons:
                meta_groups.setdefault(neuron, []).extend(colnames[neuron])

        return design_df, meta_groups

    return design_df, None

def _reduce_design_and_sync_history(
    *,
    design_df,
    X_hist,
    colnames,
    meta_groups,
    reduce_design_kwargs,
):
    """
    Reduce numeric columns of design_df and keep X_hist, colnames,
    and meta_groups consistent with dropped columns.
    """

    

    # ------------------------------
    # 1) Split columns
    # ------------------------------
    id_cols = [c for c in ['new_segment', 'new_bin'] if c in design_df.columns]
    other_cols = [c for c in design_df.columns if c not in id_cols]

    numeric_cols = design_df[other_cols].select_dtypes(include=[np.number]).columns.tolist()
    passthrough_cols = [c for c in other_cols if c not in numeric_cols]

    # ------------------------------
    # 2) Reduce numeric part
    # ------------------------------
    reduced_numeric = process_encode_design.reduce_encoding_design(
        design_df[numeric_cols],
        **(reduce_design_kwargs or {}),
    )

    # ------------------------------
    # 3) Rebuild design
    # ------------------------------
    design_before_cols = list(design_df.columns)

    design_reduced = pd.concat(
        [design_df[id_cols].reset_index(drop=True)]
        + ([design_df[passthrough_cols].reset_index(drop=True)] if passthrough_cols else [])
        + [reduced_numeric.reset_index(drop=True)],
        axis=1,
    )

    # ------------------------------
    # 4) Sync dropped columns
    # ------------------------------
    dropped_cols = set(design_before_cols) - set(design_reduced.columns)

    if dropped_cols:
        # X_hist
        for neuron, df in X_hist.items():
            keep_cols = [c for c in df.columns if c not in dropped_cols]
            X_hist[neuron] = df[keep_cols]

        # colnames
        for neuron, cols in colnames.items():
            colnames[neuron] = [c for c in cols if c not in dropped_cols]

        # meta_groups
        if meta_groups is not None:
            for k, cols in meta_groups.items():
                if isinstance(cols, list):
                    meta_groups[k] = [c for c in cols if c not in dropped_cols]

    return design_reduced, X_hist, colnames, meta_groups

def reduce_history_via_full_design(
    *,
    X_pruned,
    X_hist,
    colnames,
    meta_groups,
    spike_cols,
    reduce_design_kwargs,
):
    """
    Build a full design including ALL spike-history columns across neurons,
    run reduction to remove collinearity, and sync X_hist/colnames/meta_groups.
    """

    print('Clean out X_hist to reduce collinearity')

    # --------------------------------------------------
    # 1) Build full history design
    # --------------------------------------------------
    hist_blocks = [
        X_hist[n][colnames[n]].reset_index(drop=True)
        for n in spike_cols
        if colnames.get(n)
    ]

    design_for_reduction = pd.concat(
        [X_pruned.reset_index(drop=True)] + hist_blocks,
        axis=1,
    )

    # --------------------------------------------------
    # 2) Ensure meta_groups includes all history cols
    # --------------------------------------------------
    if meta_groups is not None:
        for neuron in spike_cols:
            existing = set(meta_groups.get(neuron, []))
            new = set(colnames.get(neuron, []))
            meta_groups[neuron] = list(existing | new)

    # --------------------------------------------------
    # 3) Reduce + sync everything
    # --------------------------------------------------
    design_for_reduction, X_hist, colnames, meta_groups = _reduce_design_and_sync_history(
        design_df=design_for_reduction,
        X_hist=X_hist,
        colnames=colnames,
        meta_groups=meta_groups,
        reduce_design_kwargs=reduce_design_kwargs,
    )

    return X_hist, colnames, meta_groups


def build_design_with_spike_history_from_bins(
    *,
    spikes_df,
    bin_df,
    X_pruned,
    meta_groups,
    dt,
    t_max,
    n_basis=20,
    target_col: Optional[str] = None,
    use_neural_coupling: bool = False,
    returnX_hist: bool = False,
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
    target_col : str, optional
        Cluster column for self spike-history (e.g. 'cluster_0').
        Defaults to first spike column. Must match the target neuron when
        predicting each unit's spikes.
    returnX_hist : bool
        If True, also return X_hist for per-unit design reuse.
    reduce_design : bool
        If True, run reduce_encoding_design on the final design matrix
        (excluding id columns like 'new_segment'/'new_bin').
    reduce_design_kwargs : dict, optional
        Extra kwargs forwarded to reduce_encoding_design.
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

    if target_col is None:
        target_col = spike_cols[0]
    elif target_col not in colnames:
        raise KeyError(
            f'target_col {target_col!r} not in colnames. '
            f'Available: {spike_cols}'
        )

    if use_neural_coupling:
        cross_neurons = [c for c in spike_cols if c != target_col]
    else:
        cross_neurons = None
    # --------------------------------------------------
    # 3) Add history to design
    # --------------------------------------------------
    # Default behavior: target self-history (+ optional cross history)
    design_w_history, meta_groups = add_spike_history_to_design(
        design_df=X_pruned,
        colnames=colnames,
        X_hist=X_hist,
        target_col=target_col,
        include_self=True,
        cross_neurons=cross_neurons,
        meta_groups=meta_groups,
    )

        
           
    if returnX_hist:
        return design_w_history, basis, colnames, meta_groups, X_hist
    return design_w_history, basis, colnames, meta_groups


def build_design_with_spike_history_and_coupling_from_bins(
    *,
    spikes_df,
    bin_df,
    X_pruned,
    meta_groups,
    dt,
    t_max,
    t_max_coupling: float = 1.375,
    n_basis: int = 20,
    n_basis_coupling: Optional[int] = None,
    target_col: Optional[str] = None,
    coupling_units: Optional[list[str]] = None,
):
    """
    Compute spike-history (self + coupling) regressors and add to design matrix.

    Self-history uses t_max (e.g. 350ms); coupling uses t_max_coupling (e.g. 1.375s).

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
        Bin width in seconds.
    t_max : float
        Maximum self spike-history length in seconds.
    t_max_coupling : float
        Maximum coupling history length in seconds (default: 1.375).
    n_basis : int
        Number of basis functions for self-history (default: 5).
    n_basis_coupling : int, optional
        Number of basis functions for coupling. Defaults to n_basis.
    target_col : str, optional
        Target cluster column (e.g. 'cluster_0'). Defaults to first spike column.
    coupling_units : list of str, optional
        Cluster columns for coupling. Defaults to all except target.

    Returns
    -------
    design_df : DataFrame
        Design with self-history and coupling columns.
    basis : ndarray
        Self-history basis (for reference).
    basis_coupling : ndarray
        Coupling basis (for reference).
    colnames : dict
        Self-history colnames.
    colnames_coupling : dict
        Coupling colnames.
    meta_groups : dict
        Updated meta_groups.
    """
    if n_basis_coupling is None:
        n_basis_coupling = n_basis

    # Ensure X_pruned aligns with bin_df (same row order, contiguous index)
    n_bins = len(bin_df)
    if len(X_pruned) != n_bins:
        raise ValueError(
            f'X_pruned has {len(X_pruned)} rows but bin_df has {n_bins}. '
            'Row counts must match.'
        )
    X_pruned = X_pruned.reset_index(drop=True)

    # Self spike-history
    X_hist, basis, colnames = compute_spike_history_designs(
        spikes_df=spikes_df,
        bin_df=bin_df,
        dt=dt,
        t_max=t_max,
        n_basis=n_basis,
    )

    spike_cols = list(colnames.keys())
    if len(spike_cols) == 0:
        raise ValueError('No spike columns found in colnames')

    if target_col is None:
        target_col = spike_cols[0]
    if coupling_units is None:
        coupling_units = [c for c in spike_cols if c != target_col]

    design_df, meta_groups = add_spike_history_to_design(
        design_df=X_pruned,
        colnames=colnames,
        X_hist=X_hist,
        target_col=target_col,
        include_self=True,
        cross_neurons=None,
        meta_groups=meta_groups,
    )

    if not coupling_units:
        return design_df, basis, None, colnames, {}, meta_groups

    # Coupling (longer time window)
    X_hist_coup, basis_coup, colnames_coup = compute_coupling_designs(
        spikes_df=spikes_df,
        bin_df=bin_df,
        dt=dt,
        t_max=t_max_coupling,
        n_basis=n_basis_coupling,
    )

    design_df, meta_groups = add_coupling_to_design(
        design_df=design_df,
        colnames=colnames_coup,
        X_hist=X_hist_coup,
        coupling_units=coupling_units,
        meta_groups=meta_groups,
    )

    return (
        design_df,
        basis,
        basis_coup,
        colnames,
        colnames_coup,
        meta_groups,
    )


def make_bin_df_from_meta_df(meta_df_used):
    return (
        meta_df_used[['event_id', 'k_within_seg', 't_left', 't_right']]
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
