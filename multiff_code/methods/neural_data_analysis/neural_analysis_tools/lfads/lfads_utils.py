import numpy as np

def make_raw_fr_from_spikes(
    spikes_df,
    clusters,
    bin_width_ms=10.0,
    start_time=None,
):
    """
    Build a continuous raw FR matrix (Hz) aligned with LFADS output.

    Parameters
    ----------
    spikes_df : DataFrame with columns ['time', 'cluster']
    clusters : array-like
        Cluster IDs in the same order as lfads_out['clusters'].
    bin_width_ms : float
    start_time : float or None
        If None, use min spike time.

    Returns
    -------
    fr_mat : (T_full, n_neurons) array of firing rates in Hz
    start_time : float
    """
    spikes_df = spikes_df.copy()
    spikes_df = spikes_df.sort_values('time').reset_index(drop=True)

    bin_width_s = bin_width_ms / 1000.0

    if start_time is None:
        start_time = float(spikes_df['time'].min())
    end_time = float(spikes_df['time'].max())
    total_dur = end_time - start_time
    n_bins = int(np.ceil(total_dur / bin_width_s))

    clusters = np.asarray(clusters)
    cluster_to_col = {c: i for i, c in enumerate(clusters)}

    spike_times = spikes_df['time'].to_numpy(float)
    spike_codes = np.fromiter(
        (cluster_to_col.get(c, -1) for c in spikes_df['cluster'].to_numpy()),
        count=len(spikes_df),
        dtype=np.int32,
    )

    bin_idx = ((spike_times - start_time) / bin_width_s).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (spike_codes >= 0)

    counts = np.zeros((n_bins, len(clusters)), dtype=np.float32)
    np.add.at(counts, (bin_idx[valid], spike_codes[valid]), 1.0)

    fr_mat = counts / bin_width_s  # counts/bin → Hz
    return fr_mat, start_time
