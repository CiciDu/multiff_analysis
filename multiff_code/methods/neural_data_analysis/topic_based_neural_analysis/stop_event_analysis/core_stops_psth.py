"""
Post-Stimulus Time Histogram (PSTH) analysis around stops (captures and misses)
Optimized version: faster segment extraction (pure NumPy), cached arrays, optional batched bootstrap.
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

    # Stop identification / cleaning
    capture_match_window: float = 0.3
    min_stop_duration: float = 0.02
    stop_debounce: float = 0.15

    # Normalization
    normalize: Optional[Literal["zscore", "sub", "div"]] = None
    # "zscore": z-score by pre-window baseline
    # "sub": subtract baseline mean
    # "div": divide by baseline mean (Fano-like)

    # Plotting
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
        ff_caught_T_new: np.ndarray,
        config: Optional[PSTHConfig] = None,
        captures_df: Optional[pd.DataFrame] = None,
        no_capture_stops_df: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        spikes_df : pd.DataFrame with columns ['time', 'cluster']
            Spike times (seconds) and cluster IDs (int or str)
        monkey_information : pd.DataFrame
            Must include 'time' (s), 'monkey_speeddummy' (0/1), optionally 'point_index'
        ff_caught_T_new : np.ndarray
            Firefly capture times (s), sorted ascending
        config : PSTHConfig, optional
            Configuration parameters for PSTH analysis
        captures_df : pd.DataFrame, optional
            Pre-filtered captures dataframe with 'stop_time' (or 'time') and 'stop_point_index'/'point_index'.
        no_capture_stops_df : pd.DataFrame, optional
            Pre-filtered no-capture stops dataframe with 'stop_time' (or 'time') and 'stop_point_index'/'point_index'.
        """
        self.spikes_df = spikes_df.copy().sort_values("time").reset_index(drop=True)
        if "cluster" not in self.spikes_df.columns:
            raise ValueError("spikes_df must have a 'cluster' column.")

        self.monkey_information = monkey_information.copy().sort_values("time").reset_index(drop=True)
        if not {"time", "monkey_speeddummy"}.issubset(self.monkey_information.columns):
            raise ValueError("monkey_information must have columns ['time', 'monkey_speeddummy'].")

        # Warn if stop_event_id system isn't present (only relevant if you later use built-in detection)
        required_stop_cols = ["whether_new_distinct_stop", "stop_event_id"]
        missing_stop_cols = [c for c in required_stop_cols if c not in self.monkey_information.columns]
        if missing_stop_cols:
            print(f"Warning: Missing stop_event_id system columns: {missing_stop_cols}")
            print("Please run add_more_columns_to_monkey_information() first if you rely on built-in stop detection.")
            print("Using provided captures/no-captures dataframes (fallback) if given.")

        self.ff_caught_T_new = np.asarray(ff_caught_T_new, dtype=float)
        if self.ff_caught_T_new.size == 0:
            raise ValueError("ff_caught_T_new is empty.")

        self.config = config or PSTHConfig()
        self.captures_df = captures_df
        self.no_capture_stops_df = no_capture_stops_df

        # Map cluster ids to contiguous indices (stable order)
        self.clusters = np.array(sorted(self.spikes_df["cluster"].unique().tolist()))
        self.cluster_to_col = {c: i for i, c in enumerate(self.clusters)}
        self.n_clusters = len(self.clusters)

        # Cache spike arrays to avoid repeated Series->NumPy conversion and hashing
        self._spike_times: np.ndarray = self.spikes_df["time"].to_numpy()
        self._spike_codes: np.ndarray = np.fromiter(
            (self.cluster_to_col[c] for c in self.spikes_df["cluster"].to_numpy()),
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
        Combine captures_df and no_capture_stops_df into a single stops dataframe.
        Returns DataFrame with columns ['stop_time', 'stop_point_index', 'stop_event_id', 'event_type'].
        """
        if self.captures_df is None or self.no_capture_stops_df is None:
            raise ValueError("Both captures_df and no_capture_stops_df must be provided to use pre-filtered stops.")

        def _prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
            out = df.copy()
            if "stop_time" not in out.columns:
                if "time" in out.columns:
                    out = out.rename(columns={"time": "stop_time"})
                else:
                    raise ValueError(f"{label} dataframe missing 'stop_time' or 'time' column.")
            if "stop_point_index" not in out.columns:
                if "point_index" in out.columns:
                    out = out.rename(columns={"point_index": "stop_point_index"})
                else:
                    # Make it optional; downstream doesn't strictly require the index
                    out["stop_point_index"] = np.nan
            out["event_type"] = label
            return out[["stop_time", "stop_point_index", "event_type"]]

        cap = _prep(self.captures_df, "capture")
        mis = _prep(self.no_capture_stops_df, "miss")

        combined = pd.concat([cap, mis], ignore_index=True)
        combined = combined.sort_values("stop_time", kind="mergesort").reset_index(drop=True)
        combined["stop_event_id"] = np.arange(len(combined), dtype=int)

        self.stop_events = combined
        print("Combination of pre-filtered captures and no-capture stops complete.")
        return self.stop_events

    # ----------------------- Binning utilities (cached) -----------------------

    def _make_time_edges_and_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create time bin edges and centers for PSTH analysis."""
        cfg = self.config
        n_bins_pre = int(np.round(cfg.pre_window / cfg.bin_width))
        n_bins_post = int(np.round(cfg.post_window / cfg.bin_width))
        n_bins = n_bins_pre + n_bins_post
        edges = np.linspace(-cfg.pre_window, cfg.post_window, n_bins + 1, endpoint=True, dtype=np.float64)
        centers = edges[:-1] + cfg.bin_width / 2.0
        return edges, centers

    # ------------------------- Segment extraction (fast) ----------------------

    def extract_neural_segments(self, event_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract peri-stop binned spike-count matrices.

        Returns
        -------
        dict:
          'capture': (n_events, n_bins, n_clusters)
          'miss':    (n_events, n_bins, n_clusters)
        """
        if self.stop_events is None:
            self.identify_stop_events()

        events = self.stop_events
        if event_type:
            events = events[events["event_type"] == event_type]

        n_bins = self._n_bins
        pre_w = self.config.pre_window
        post_w = self.config.post_window
        bw = self.config.bin_width

        times = self._spike_times
        codes = self._spike_codes  # contiguous 0..n_clusters-1

        # Pre-allocate output arrays
        ev_type = events["event_type"].to_numpy()
        cap_idx = np.flatnonzero(ev_type == "capture")
        mis_idx = np.flatnonzero(ev_type == "miss")

        cap_arr = np.zeros((len(cap_idx), n_bins, self.n_clusters), dtype=np.float32)
        mis_arr = np.zeros((len(mis_idx), n_bins, self.n_clusters), dtype=np.float32)

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

        # Fill capture events
        if cap_idx.size:
            stop_times = events.iloc[cap_idx]["stop_time"].to_numpy(dtype=float)
            for k, t0 in enumerate(stop_times):
                fill_event(float(t0), cap_arr[k])

        # Fill miss events
        if mis_idx.size:
            stop_times = events.iloc[mis_idx]["stop_time"].to_numpy(dtype=float)
            for k, t0 in enumerate(stop_times):
                fill_event(float(t0), mis_arr[k])

        return {"capture": cap_arr, "miss": mis_arr}

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
        pre_counts = counts_trials[:, pre_mask, :]  # (n_trials, n_pre, n_clusters)
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
          - 'capture', 'miss': mean rate (n_bins, n_clusters) after smoothing & normalization
          - 'capture_sem', 'miss_sem': SEM (same shape) OR bootstrap CI half-width
          - 'time_axis': bin centers (s)
        """
        time_axis = self._centers
        n_bins = self._n_bins
        pre_mask = self._pre_mask

        out: Dict[str, np.ndarray] = {"time_axis": time_axis}

        def _per_condition(name: str):
            arr = segments.get(name, np.zeros((0, n_bins, self.n_clusters), dtype=np.float32))
            if arr.shape[0] < self.config.min_trials:
                out[name] = np.zeros((n_bins, self.n_clusters), dtype=np.float32)
                out[name + "_sem"] = np.zeros((n_bins, self.n_clusters), dtype=np.float32)
                return

            # baseline + normalization
            base_mu, base_sd = self._baseline_stats(arr, pre_mask)
            rates_trials = self._trial_rates(arr.astype(np.float32))
            rates_trials = self._normalize(rates_trials, base_mu, base_sd).astype(np.float32)

            # mean across trials
            mean_rate = rates_trials.mean(axis=0)  # (n_bins, n_clusters)
            mean_rate = self._apply_smoothing(mean_rate).astype(np.float32)

            # CI/SEM
            if self.config.ci_method == "sem":
                sem = rates_trials.std(axis=0, ddof=1) / np.sqrt(rates_trials.shape[0])
                sem = self._apply_smoothing(sem).astype(np.float32)
                out[name + "_sem"] = sem
            else:
                # Batched bootstrap over the trial axis to bound memory
                it = int(self.config.bootstrap_iters)
                ntr = rates_trials.shape[0]
                rng = np.random.default_rng(12345)
                qs = [100 * (self.config.alpha / 2), 100 * (1 - self.config.alpha / 2)]
                chunk = max(1, int(self.config.bootstrap_chunk))

                # Accumulate bootstrap means in chunks
                boot_means_list = []
                for s in range(0, it, chunk):
                    m = min(chunk, it - s)
                    idx = rng.integers(0, ntr, size=(m, ntr))
                    boot_means_list.append(rates_trials[idx].mean(axis=1))  # (m, n_bins, n_clusters)
                boot_means = np.concatenate(boot_means_list, axis=0)  # (it, n_bins, n_clusters)

                lo, hi = np.percentile(boot_means, qs, axis=0)
                half = ((hi - lo) / 2.0).astype(np.float32)
                out[name + "_sem"] = self._apply_smoothing(half).astype(np.float32)

            out[name] = mean_rate

        _per_condition("capture")
        _per_condition("miss")

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
            "n_events": {k: int(segments[k].shape[0]) for k in ["capture", "miss"]},
        }
        return self.psth_data

    # ------------------------------ Plotting ---------------------------------

    def _plot_one(self, ax, time_axis, mean, ci, label=None):
        ax.plot(time_axis, mean, linewidth=2, label=label)
        ax.fill_between(time_axis, mean - ci, mean + ci, alpha=0.25, linewidth=0)

    def plot_psth(
        self,
        cluster_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        show_individual: bool = False,
    ) -> plt.Figure:
        """Plot PSTHs for captures & misses in two columns."""
        if not self.psth_data:
            self.run_full_analysis(cluster_idx)

        psth = self.psth_data["psth"]
        segments = self.psth_data["segments"]
        n_events = self.psth_data["n_events"]

        if cluster_idx is None:
            cluster_indices = range(self.n_clusters)
        else:
            cluster_indices = [cluster_idx]

        fig, axes = plt.subplots(len(list(cluster_indices)), 2, figsize=figsize, squeeze=False)
        time = psth["time_axis"]

        for row_i, ci in enumerate(cluster_indices):
            cid = self.clusters[ci]
            ax_c = axes[row_i, 0]
            ax_m = axes[row_i, 1]

            if n_events["capture"] >= self.config.min_trials:
                mean = psth["capture"][:, ci]
                ciw = psth["capture_sem"][:, ci]
                #self._plot_one(ax_c, time, mean, ciw, f"Capture (n={n_events['capture']})")
                self._plot_one(ax_c, time, mean, ciw)
                if show_individual:
                    for trial in segments["capture"]:
                        ax_c.plot(time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            if n_events["miss"] >= self.config.min_trials:
                mean = psth["miss"][:, ci]
                ciw = psth["miss_sem"][:, ci]
                #self._plot_one(ax_m, time, mean, ciw, f"Miss (n={n_events['miss']})")
                self._plot_one(ax_m, time, mean, ciw)
                if show_individual:
                    for trial in segments["miss"]:
                        ax_m.plot(time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            for ax in (ax_c, ax_m):
                ax.axvline(0, color="k", linestyle="--", alpha=0.5)
                ax.set_xlabel("Time relative to stop (s)")
                ax.set_ylabel("Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
                ax.grid(True, alpha=0.3)
                ax.legend()

            ax_c.set_title(f"Cluster {cid} — Captures")
            ax_m.set_title(f"Cluster {cid} — Misses")

        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        cluster_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Overlay capture vs miss for each cluster with bands."""
        if not self.psth_data:
            self.run_full_analysis(cluster_idx)

        psth = self.psth_data["psth"]
        n_events = self.psth_data["n_events"]
        time = psth["time_axis"]

        if cluster_idx is None:
            cluster_indices = range(self.n_clusters)
        else:
            cluster_indices = [cluster_idx]

        fig, axes = plt.subplots(len(list(cluster_indices)), 1, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, ci in zip(axes, cluster_indices):
            cid = self.clusters[ci]
            if n_events["capture"] >= self.config.min_trials:
                mc = psth["capture"][:, ci]
                cc = psth["capture_sem"][:, ci]
                self._plot_one(ax, time, mc, cc, f"Capture (n={n_events['capture']})")

            if n_events["miss"] >= self.config.min_trials:
                mm = psth["miss"][:, ci]
                cm = psth["miss_sem"][:, ci]
                self._plot_one(ax, time, mm, cm, f"Miss (n={n_events['miss']})")

            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel("Time relative to stop (s)")
            ax.set_ylabel("Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
            ax.set_title(f"Cluster {cid} — Capture vs Miss")
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
        Nonparametric test on average rate within a time window (capture vs miss).
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
                arr = segments.get(name, np.zeros((0, len(time_axis), self.n_clusters), dtype=np.float32))
                if arr.shape[0] == 0:
                    return []
                # convert to Hz
                rates = arr[:, start_idx:end_idx + 1, ci].mean(axis=1) / cfg.bin_width
                return rates.tolist()

            cap = _collect("capture")
            mis = _collect("miss")

            if len(cap) >= cfg.min_trials and len(mis) >= cfg.min_trials:
                stat, p = stats.mannwhitneyu(cap, mis, alternative="two-sided")
                # Cohen's d (pooled std)
                mean_diff = float(np.mean(cap) - np.mean(mis))
                pooled_sd = np.sqrt(((len(cap) - 1) * np.var(cap, ddof=1) + (len(mis) - 1) * np.var(mis, ddof=1))
                                    / (len(cap) + len(mis) - 2))
                d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
                results[str(cid)] = {
                    "capture_mean": float(np.mean(cap)),
                    "capture_std": float(np.std(cap, ddof=1)),
                    "miss_mean": float(np.mean(mis)),
                    "miss_std": float(np.std(mis, ddof=1)),
                    "statistic_U": float(stat),
                    "p_value": float(p),
                    "cohens_d": float(d),
                    "n_captures": int(len(cap)),
                    "n_misses": int(len(mis)),
                }
            else:
                results[str(cid)] = {"error": "Insufficient data for comparison"}

        return results


# ---------------------------- Convenience API --------------------------------

def create_psth_around_stops(
    spikes_df: pd.DataFrame,
    monkey_information: pd.DataFrame,
    ff_caught_T_new: np.ndarray,
    config: Optional[PSTHConfig] = None,
    cluster_idx: Optional[int] = None,
    captures_df: Optional[pd.DataFrame] = None,
    no_capture_stops_df: Optional[pd.DataFrame] = None,
) -> PSTHAnalyzer:
    """
    Convenience function to create and run PSTH analysis around stops.
    """
    analyzer = PSTHAnalyzer(
        spikes_df, monkey_information, ff_caught_T_new, config, captures_df, no_capture_stops_df
    )
    analyzer.run_full_analysis(cluster_idx)
    return analyzer
