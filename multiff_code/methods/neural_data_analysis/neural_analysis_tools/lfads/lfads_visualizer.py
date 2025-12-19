import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


# =====================================================================
# Helper: compute “raw” FR from spikes_df (10 ms bins) to compare vs LFADS
# =====================================================================

def make_raw_fr_from_spikes(
    spikes_df,
    clusters,
    bin_width_ms=10.0,
    start_time=None,
):
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
    np.add.at(
        counts,
        (bin_idx[valid], spike_codes[valid]),
        1.0
    )

    fr_mat = counts / bin_width_s  # Hz
    return fr_mat, start_time



# =====================================================================
# LFADSResultsVisualizer
# =====================================================================

class LFADSResultsVisualizer:
    """
    Visualize continuous LFADS outputs from run_lfads_on_continuous_session.

    Provides:
      - Raw vs LFADS FR comparison
      - LFADS internal factors visualization
      - Event-aligned firing rates (raw + LFADS)
      - PCA trajectories using LFADS factors or FR
    """

    def __init__(
        self,
        spikes_df,
        lfads_fr_mat,
        lfads_factors_mat,
        clusters,
        start_time,
        bin_width_ms=10.0,
    ):
        self.spikes_df = spikes_df
        self.lfads_fr_mat = lfads_fr_mat
        self.lfads_factors_mat = lfads_factors_mat
        self.clusters = clusters
        self.start_time = start_time
        self.bin_width_s = bin_width_ms / 1000.0

        # Build raw FR for comparison
        self.raw_fr_mat, _ = make_raw_fr_from_spikes(
            spikes_df,
            clusters,
            bin_width_ms=bin_width_ms,
            start_time=start_time,
        )

        n_lfads, n_raw = self.lfads_fr_mat.shape[0], self.raw_fr_mat.shape[0]
        T = min(n_lfads, n_raw)

        # Crop to same length if necessary
        self.lfads_fr_mat = self.lfads_fr_mat[:T]
        self.raw_fr_mat = self.raw_fr_mat[:T]
        if self.lfads_factors_mat is not None:
            self.lfads_factors_mat = self.lfads_factors_mat[:T]

        self.t = np.arange(T) * self.bin_width_s + start_time



    # =================================================================
    # 1. Raw vs LFADS FR for a neuron
    # =================================================================
    def plot_raw_vs_lfads(self, neuron_index, smooth_sigma_ms=0.0):
        fr_raw = self.raw_fr_mat[:, neuron_index]
        fr_lfads = self.lfads_fr_mat[:, neuron_index]

        if smooth_sigma_ms > 0:
            fr_raw = gaussian_filter1d(fr_raw, smooth_sigma_ms / 10.0)

        plt.figure(figsize=(14, 4))
        plt.plot(self.t, fr_raw, alpha=0.4, label='Raw FR (binned)')
        plt.plot(self.t, fr_lfads, linewidth=2, label='LFADS FR')
        plt.title(f'Neuron {self.clusters[neuron_index]}: Raw vs LFADS FR')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
        plt.legend()
        plt.tight_layout()
        plt.show()



    # =================================================================
    # 2. Plot LFADS latent factors
    # =================================================================
    def plot_lfads_factors(self, n_factors=5):
        if self.lfads_factors_mat is None:
            print('No factors available (lfads_factors_mat=None).')
            return

        plt.figure(figsize=(14, 6))
        for i in range(min(n_factors, self.lfads_factors_mat.shape[1])):
            plt.plot(self.t, self.lfads_factors_mat[:, i], label=f'Factor {i}')
        plt.title('LFADS latent factors (continuous)')
        plt.xlabel('Time (s)')
        plt.ylabel('Factor value')
        plt.legend()
        plt.tight_layout()
        plt.show()



    # =================================================================
    # 3. Event-aligned FR (raw + LFADS)
    # =================================================================
    def plot_event_aligned(
        self,
        event_df,
        event_col='event_time',
        window_ms=(-200, 600),
        neuron_index=0,
    ):
        event_times = event_df[event_col].values
        event_bins = ((event_times - self.start_time) / self.bin_width_s).astype(int)

        win_lo = int(window_ms[0] / (1000 * self.bin_width_s))
        win_hi = int(window_ms[1] / (1000 * self.bin_width_s))
        win = np.arange(win_lo, win_hi)

        segments_raw = []
        segments_lfads = []
        for b in event_bins:
            if b + win_lo < 0 or b + win_hi >= len(self.t):
                continue
            segments_raw.append(self.raw_fr_mat[b + win, neuron_index])
            segments_lfads.append(self.lfads_fr_mat[b + win, neuron_index])

        segments_raw = np.array(segments_raw)
        segments_lfads = np.array(segments_lfads)

        plt.figure(figsize=(10, 5))
        plt.plot(win * self.bin_width_s * 1000, segments_raw.mean(axis=0),
                 label='Raw', alpha=0.5)
        plt.plot(win * self.bin_width_s * 1000, segments_lfads.mean(axis=0),
                 label='LFADS', linewidth=2)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'Event-aligned FR (Neuron {self.clusters[neuron_index]})')
        plt.xlabel('Time relative to event (ms)')
        plt.ylabel('Rate (Hz)')
        plt.legend()
        plt.tight_layout()
        plt.show()



    # =================================================================
    # 4. PCA on LFADS factors or FR
    # =================================================================
    def plot_pca_factors(self, n_components=3, smooth_sigma_ms=10.0):
        if self.lfads_factors_mat is None:
            print('No lfads_factors_mat available.')
            return

        X = self.lfads_factors_mat.copy()

        if smooth_sigma_ms > 0:
            X = gaussian_filter1d(X, smooth_sigma_ms / 10.0, axis=0)

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X)

        plt.figure(figsize=(14, 4))
        for i in range(min(n_components, pcs.shape[1])):
            plt.plot(self.t, pcs[:, i], label=f'PC{i+1}')
        plt.title('PCA on LFADS factors')
        plt.xlabel('Time (s)')
        plt.ylabel('PC amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()


