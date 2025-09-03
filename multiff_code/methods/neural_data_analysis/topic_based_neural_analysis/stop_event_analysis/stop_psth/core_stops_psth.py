"""
Post-Stimulus Time Histogram (PSTH) analysis around stops (event_a vs event_b)
Optimized version: faster segment extraction (pure NumPy), cached arrays, optional batched bootstrap.

Mapping for this project:
  - event_a = captures
  - event_b = non-captures (misses)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# ----------------------------- Configuration ---------------------------------

@dataclass
class PSTHConfig:
    """Configuration for PSTH analysis."""
    pre_window: float = 1.0                 # seconds before stop
    post_window: float = 1.0                # seconds after stop
    bin_width: float = 0.02                 # seconds per bin
    smoothing_sigma: float = 0.05           # Gaussian smoothing σ (seconds)
    min_trials: int = 5                     # minimum number of events required

    # Stop identification / cleaning (kept for compatibility with your pipeline)
    capture_match_window: float = 0.3
    min_stop_duration: float = 0.01
    stop_debounce: float = 0.15

    # Normalization
    normalize: Optional[Literal["zscore", "sub", "div"]] = None
    # "zscore": z-score by pre-window baseline
    # "sub": subtract baseline mean
    # "div": divide by baseline mean (Fano-like)

    # Plotting CI
    ci_method: Literal["sem", "bootstrap"] = "sem"
    bootstrap_iters: int = 500
    alpha: float = 0.05                      # for bootstrap CIs
    bootstrap_chunk: int = 64                # to bound memory during bootstrap


# ----------------------------- Core Analyzer ---------------------------------

class PSTHAnalyzer:
    """
    Analyzer for Post-Stimulus Time Histograms around stops.

    Methods:
      1) identify_stop_events()
      2) extract_neural_segments()
      3) compute_psth()
      4) plot_psth() / plot_comparison()
      5) statistical_comparison()
    """

    def __init__(
        self,
        spikes_df: pd.DataFrame,
        monkey_information: pd.DataFrame,
        config: Optional[PSTHConfig] = None,
        # Generic inputs (required)
        event_a_df: Optional[pd.DataFrame] = None,
        event_b_df: Optional[pd.DataFrame] = None,
        event_a_label: str = "Capture",
        event_b_label: str = "Non-capture",
    ):
        """
        Parameters
        ----------
        spikes_df : pd.DataFrame with columns ['time', 'cluster']
            Spike times (seconds) and cluster IDs (int or str).
        monkey_information : pd.DataFrame
            Must include 'time' (s), 'monkey_speeddummy' (0/1), optionally 'point_index'.
        event_a_df : pd.DataFrame
            Pre-filtered events A with 'stop_time' (or 'time').
        event_b_df : pd.DataFrame
            Pre-filtered events B with 'stop_time' (or 'time').
        """
        
        self.event_a_label = event_a_label
        self.event_b_label = event_b_label
        
        self.spikes_df = spikes_df.copy().sort_values("time").reset_index(drop=True)
        if "cluster" not in self.spikes_df.columns:
            raise ValueError("spikes_df must have a 'cluster' column.")

        self.monkey_information = monkey_information.copy(
        ).sort_values("time").reset_index(drop=True)
        if not {"time", "monkey_speeddummy"}.issubset(self.monkey_information.columns):
            raise ValueError(
                "monkey_information must have columns ['time', 'monkey_speeddummy'].")

        self.config = config or PSTHConfig()

        if event_a_df is None or event_b_df is None:
            raise ValueError("Provide both event_a_df and event_b_df.")
        self.event_a_df = event_a_df
        self.event_b_df = event_b_df

        # Map cluster ids to contiguous indices (stable order)
        self.clusters = np.array(
            sorted(self.spikes_df["cluster"].unique().tolist()))
        self.cluster_to_col = {c: i for i, c in enumerate(self.clusters)}
        self.n_clusters = len(self.clusters)

        # Cache spike arrays to avoid repeated Series->NumPy conversion
        self._spike_times: np.ndarray = self.spikes_df["time"].to_numpy()
        self._spike_codes: np.ndarray = np.fromiter(
            (self.cluster_to_col[c]
             for c in self.spikes_df["cluster"].to_numpy()),
            count=len(self.spikes_df),
            dtype=np.int32,
        )

        # Cache bin edges/centers and pre-mask (depends only on config)
        self._edges, self._centers = self._make_time_edges_and_centers()
        self._n_bins = len(self._centers)
        self._pre_mask = self._centers < 0

        # Results
        self.stop_events: Optional[pd.DataFrame] = None
        self.psth_data: Dict = {}

    # --------------------------- Stop events combiner -------------------------

    def identify_stop_events(self) -> pd.DataFrame:
        """
        Combine event_a_df and event_b_df into a single stops dataframe.
        Returns DataFrame with columns ['stop_time', 'event_type'].
        event_type is one of {'event_a', 'event_b'}.
        """

        def _prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
            out = df.copy()
            if "stop_time" not in out.columns:
                raise ValueError(
                    f"{label} dataframe missing 'stop_time' column.")
            out["event_type"] = label
            return out[["stop_time", "event_type"]]

        a = _prep(self.event_a_df, "event_a")
        b = _prep(self.event_b_df, "event_b")

        combined = pd.concat([a, b], ignore_index=True)
        combined = combined.sort_values(
            "stop_time", kind="mergesort").reset_index(drop=True)

        self.stop_events = combined
        return self.stop_events

    # ----------------------- Binning utilities (cached) -----------------------

    def _make_time_edges_and_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create time bin edges and centers for PSTH analysis."""
        cfg = self.config
        n_bins_pre = int(np.round(cfg.pre_window / cfg.bin_width))
        n_bins_post = int(np.round(cfg.post_window / cfg.bin_width))
        n_bins = n_bins_pre + n_bins_post
        edges = np.linspace(-cfg.pre_window, cfg.post_window,
                            n_bins + 1, endpoint=True, dtype=np.float64)
        centers = edges[:-1] + cfg.bin_width / 2.0
        return edges, centers

    # ------------------------- Segment extraction (fast) ----------------------

    def extract_neural_segments(self, event_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract peri-stop binned spike-count matrices.

        Returns
        -------
        dict:
          'event_a': (n_events, n_bins, n_clusters)
          'event_b': (n_events, n_bins, n_clusters)
        """
        if self.stop_events is None:
            self.identify_stop_events()

        events = self.stop_events
        if event_type:
            if event_type not in {"event_a", "event_b"}:
                raise ValueError("event_type must be 'event_a' or 'event_b'")
            events = events[events["event_type"] == event_type]

        n_bins = self._n_bins
        pre_w = self.config.pre_window
        post_w = self.config.post_window
        bw = self.config.bin_width

        times = self._spike_times
        codes = self._spike_codes  # contiguous 0..n_clusters-1

        # Pre-allocate output arrays
        ev_type = events["event_type"].to_numpy()
        a_idx = np.flatnonzero(ev_type == "event_a")
        b_idx = np.flatnonzero(ev_type == "event_b")

        a_arr = np.zeros(
            (len(a_idx), n_bins, self.n_clusters), dtype=np.float32)
        b_arr = np.zeros(
            (len(b_idx), n_bins, self.n_clusters), dtype=np.float32)

        # Local fast function to accumulate spikes for a single event
        def fill_event(t0: float, out_mat: np.ndarray):
            start, end = t0 - pre_w, t0 + post_w
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end,   side="right")
            if right <= left:
                return
            rel_t = times[left:right] - t0
            # Affine binning for uniform bins: shift so -pre_w -> 0
            rows = np.floor((rel_t + pre_w) / bw).astype(np.int32)
            valid = (rows >= 0) & (rows < n_bins)
            if not np.any(valid):
                return
            cols = codes[left:right][valid]
            rows = rows[valid]
            # Accumulate 1 spike per (row, col)
            np.add.at(out_mat, (rows, cols), 1.0)

        # Fill event_a
        if a_idx.size:
            stop_times = events.iloc[a_idx]["stop_time"].to_numpy(dtype=float)
            for k, t0 in enumerate(stop_times):
                fill_event(float(t0), a_arr[k])

        # Fill event_b
        if b_idx.size:
            stop_times = events.iloc[b_idx]["stop_time"].to_numpy(dtype=float)
            for k, t0 in enumerate(stop_times):
                fill_event(float(t0), b_arr[k])

        return {"event_a": a_arr, "event_b": b_arr}

    # ----------------------------- PSTH compute ------------------------------

    def _apply_smoothing(self, rate: np.ndarray) -> np.ndarray:
        """Gaussian smoothing along time axis with fractional sigma in bins."""
        sigma_bins = self.config.smoothing_sigma / self.config.bin_width
        if sigma_bins <= 0:
            return rate
        return gaussian_filter1d(rate, sigma=sigma_bins, axis=0, mode="reflect")

    def _baseline_stats(self, counts_trials: np.ndarray, pre_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean and std of firing rates in the pre-stop window (per cluster).
        """
        if counts_trials.shape[0] == 0:
            return np.zeros((self.n_clusters,), dtype=np.float64), np.ones((self.n_clusters,), dtype=np.float64)
        # (n_trials, n_pre, n_clusters)
        pre_counts = counts_trials[:, pre_mask, :]
        bw = self.config.bin_width
        baseline_mean = pre_counts.mean(axis=(0, 1)) / bw
        baseline_std = pre_counts.std(axis=(0, 1), ddof=1) / bw
        baseline_std[baseline_std == 0] = 1.0
        return baseline_mean, baseline_std

    def _trial_rates(self, counts_trials: np.ndarray) -> np.ndarray:
        """Convert count matrices to firing rate per trial (Hz)."""
        return counts_trials / self.config.bin_width

    def _normalize(self, rates_trials: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray) -> np.ndarray:
        """Apply normalization and return a new array."""
        mode = self.config.normalize
        if mode is None:
            return rates_trials
        if mode == "zscore":
            return (rates_trials - baseline_mean[np.newaxis, np.newaxis, :]) / baseline_std[np.newaxis, np.newaxis, :]
        if mode == "sub":
            return rates_trials - baseline_mean[np.newaxis, np.newaxis, :]
        if mode == "div":
            return rates_trials / baseline_mean[np.newaxis, np.newaxis, :]
        raise ValueError(f"Unknown normalize='{mode}'")

    def compute_psth(self, segments: Dict[str, np.ndarray], cluster_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute PSTH mean and CI for each condition.

        Returns
        -------
        dict with keys:
          - 'event_a', 'event_b': mean rate (n_bins, n_clusters) after smoothing & normalization
          - 'event_a_sem', 'event_b_sem': SEM (same shape) OR bootstrap CI half-width
          - 'time_axis': bin centers (s)
        """
        time_axis = self._centers
        n_bins = self._n_bins
        pre_mask = self._pre_mask

        out: Dict[str, np.ndarray] = {"time_axis": time_axis}

        def _per_condition(name: str):
            arr = segments.get(name, np.zeros(
                (0, n_bins, self.n_clusters), dtype=np.float32))
            if arr.shape[0] < self.config.min_trials:
                out[name] = np.zeros(
                    (n_bins, self.n_clusters), dtype=np.float32)
                out[name + "_sem"] = np.zeros((n_bins,
                                              self.n_clusters), dtype=np.float32)
                return

            # baseline + normalization
            base_mu, base_sd = self._baseline_stats(arr, pre_mask)
            rates_trials = self._trial_rates(arr.astype(np.float32))
            rates_trials = self._normalize(
                rates_trials, base_mu, base_sd).astype(np.float32)

            # mean across trials
            mean_rate = rates_trials.mean(axis=0)  # (n_bins, n_clusters)
            mean_rate = self._apply_smoothing(mean_rate).astype(np.float32)

            # CI/SEM
            if self.config.ci_method == "sem":
                sem = rates_trials.std(axis=0, ddof=1) / \
                    np.sqrt(rates_trials.shape[0])
                sem = self._apply_smoothing(sem).astype(np.float32)
                out[name + "_sem"] = sem
            else:
                # Batched bootstrap over the trial axis to bound memory
                it = int(self.config.bootstrap_iters)
                ntr = rates_trials.shape[0]
                rng = np.random.default_rng(12345)
                qs = [100 * (self.config.alpha / 2), 100 *
                      (1 - self.config.alpha / 2)]
                chunk = max(1, int(self.config.bootstrap_chunk))

                boot_means_list = []
                for s in range(0, it, chunk):
                    m = min(chunk, it - s)
                    idx = rng.integers(0, ntr, size=(m, ntr))
                    boot_means_list.append(rates_trials[idx].mean(
                        axis=1))  # (m, n_bins, n_clusters)
                # (it, n_bins, n_clusters)
                boot_means = np.concatenate(boot_means_list, axis=0)

                lo, hi = np.percentile(boot_means, qs, axis=0)
                half = ((hi - lo) / 2.0).astype(np.float32)
                out[name +
                    "_sem"] = self._apply_smoothing(half).astype(np.float32)

            out[name] = mean_rate

        _per_condition("event_a")
        _per_condition("event_b")

        return out

    # ------------------------------ Runner -----------------------------------

    def run_full_analysis(self, cluster_idx: Optional[int] = None) -> Dict:
        """Runs segment extraction and PSTH computation; stores in self.psth_data."""
        segments = self.extract_neural_segments()
        psth = self.compute_psth(segments, cluster_idx)
        self.psth_data = {
            "segments": segments,
            "psth": psth,
            "config": self.config,
            "n_events": {k: int(segments[k].shape[0]) for k in ["event_a", "event_b"]},
        }
        return self.psth_data

    # ------------------------------ Plotting ---------------------------------

    def _plot_one(self, ax, time_axis, mean, ci, label=None):
        ax.plot(time_axis, mean, linewidth=2, label=label)
        ax.fill_between(time_axis, mean - ci, mean +
                        ci, alpha=0.25, linewidth=0)

    def plot_psth(
        self,
        cluster_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        show_individual: bool = False,
    ) -> plt.Figure:
        """Plot PSTHs for event_a & event_b in two columns."""
        if not self.psth_data:
            self.run_full_analysis(cluster_idx)

        psth = self.psth_data["psth"]
        segments = self.psth_data["segments"]
        n_events = self.psth_data["n_events"]

        cluster_indices = range(self.n_clusters) if cluster_idx is None else [
            cluster_idx]
        fig, axes = plt.subplots(
            len(list(cluster_indices)), 2, figsize=figsize, squeeze=False)
        time = psth["time_axis"]

        for row_i, ci in enumerate(cluster_indices):
            cid = self.clusters[ci]
            ax_a = axes[row_i, 0]
            ax_b = axes[row_i, 1]

            if n_events["event_a"] >= self.config.min_trials:
                mean = psth["event_a"][:, ci]
                ciw = psth["event_a_sem"][:, ci]
                self._plot_one(ax_a, time, mean, ciw,
                               f"{self.event_a_label} (n={n_events['event_a']})")
                if show_individual:
                    for trial in segments["event_a"]:
                        ax_a.plot(
                            time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            if n_events["event_b"] >= self.config.min_trials:
                mean = psth["event_b"][:, ci]
                ciw = psth["event_b_sem"][:, ci]
                self._plot_one(
                    ax_b, time, mean, ciw, f"{self.event_b_label} (n={n_events['event_b']})")
                if show_individual:
                    for trial in segments["event_b"]:
                        ax_b.plot(
                            time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            for ax in (ax_a, ax_b):
                ax.axvline(0, color="k", linestyle="--", alpha=0.5)
                ax.set_xlabel("Time relative to stop (s)")
                ax.set_ylabel(
                    "Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
                ax.grid(True, alpha=0.3)
                ax.legend()

            ax_a.set_title(f"Cluster {cid} — {self.event_a_label}")
            ax_b.set_title(f"Cluster {cid} — {self.event_b_label}")

        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        cluster_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Overlay event_a vs event_b for each cluster with bands."""
        if not self.psth_data:
            self.run_full_analysis(cluster_idx)

        psth = self.psth_data["psth"]
        n_events = self.psth_data["n_events"]
        time = psth["time_axis"]

        cluster_indices = range(self.n_clusters) if cluster_idx is None else [
            cluster_idx]
        fig, axes = plt.subplots(
            len(list(cluster_indices)), 1, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, ci in zip(axes, cluster_indices):
            cid = self.clusters[ci]
            if n_events["event_a"] >= self.config.min_trials:
                ma = psth["event_a"][:, ci]
                ca = psth["event_a_sem"][:, ci]
                self._plot_one(
                    ax, time, ma, ca, f"{self.event_a_label} (n={n_events['event_a']})")

            if n_events["event_b"] >= self.config.min_trials:
                mb = psth["event_b"][:, ci]
                cb = psth["event_b_sem"][:, ci]
                self._plot_one(
                    ax, time, mb, cb, f"{self.event_b_label} (n={n_events['event_b']})")

            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel("Time relative to stop (s)")
            ax.set_ylabel(
                "Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
            ax.set_title(
                f"Cluster {cid} — {self.event_a_label} vs {self.event_b_label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        return fig

    # --------------------------- Statistics ----------------------------------

    def statistical_comparison(
        self,
        cluster_idx: Optional[int] = None,
        time_window: Tuple[float, float] = (0.0, 0.5),
    ) -> Dict:
        """
        Nonparametric test on average rate within a time window (event_a vs event_b).
        Returns per-cluster dict with U-stat, p, Cohen's d, and sample sizes.
        """
        if not self.psth_data:
            self.run_full_analysis(cluster_idx)

        segments = self.psth_data["segments"]
        time_axis = self.psth_data["psth"]["time_axis"]
        cfg = self.config

        # indices for time window (inclusive)
        start_idx = int(np.argmin(np.abs(time_axis - time_window[0])))
        end_idx = int(np.argmin(np.abs(time_axis - time_window[1])))
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        results: Dict = {}
        for ci in range(self.n_clusters):
            cid = self.clusters[ci]

            def _collect(name: str) -> List[float]:
                arr = segments.get(name, np.zeros(
                    (0, len(time_axis), self.n_clusters), dtype=np.float32))
                if arr.shape[0] == 0:
                    return []
                # convert to Hz
                rates = arr[:, start_idx:end_idx + 1,
                            ci].mean(axis=1) / cfg.bin_width
                return rates.tolist()

            a = _collect("event_a")
            b = _collect("event_b")

            if len(a) >= cfg.min_trials and len(b) >= cfg.min_trials:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                # Cohen's d (pooled std)
                mean_diff = float(np.mean(a) - np.mean(b))
                pooled_sd = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                                    / (len(a) + len(b) - 2))
                d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
                results[str(cid)] = {
                    "event_a_mean": float(np.mean(a)),
                    "event_a_std": float(np.std(a, ddof=1)),
                    "event_b_mean": float(np.mean(b)),
                    "event_b_std": float(np.std(b, ddof=1)),
                    "statistic_U": float(stat),
                    "p_value": float(p),
                    "cohens_d": float(d),
                    "n_event_a": int(len(a)),
                    "n_event_b": int(len(b)),
                }
            else:
                results[str(cid)] = {
                    "error": "Insufficient data for comparison"}

        return results


# ---------------------------- Convenience API --------------------------------

def create_psth_around_stops(
    spikes_df: pd.DataFrame,
    monkey_information: pd.DataFrame,
    event_a_df: pd.DataFrame,
    event_b_df: pd.DataFrame,
    event_a_label: str = "Capture",
    event_b_label: str = "Non-capture",
    config: Optional[PSTHConfig] = None,
    cluster_idx: Optional[int] = None,
) -> PSTHAnalyzer:
    """
    Convenience function to create and run PSTH analysis around stops.

    For this project:
      - event_a_df = captures
      - event_b_df = non-captures
    """
    analyzer = PSTHAnalyzer(
        spikes_df=spikes_df,
        monkey_information=monkey_information,
        config=config,
        event_a_df=event_a_df,
        event_b_df=event_b_df,
        event_a_label=event_a_label,
        event_b_label=event_b_label,
    )
    analyzer.run_full_analysis(cluster_idx)
    return analyzer
