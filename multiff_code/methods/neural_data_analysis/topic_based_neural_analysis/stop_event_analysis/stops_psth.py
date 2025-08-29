"""
Post-Stimulus Time Histogram (PSTH) analysis around stops (captures and misses)
===============================================================================

This module provides functions to analyze neural responses around different types of stops:
- Captures: stops that resulted in successful firefly capture
- Misses: stops that did not result in capture

It extracts peri-event windows around stop events, bins spikes, computes firing rates,
optionally baseline-normalizes them, and plots PSTHs with confidence bands.
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
    # classify stop as capture if within this of a capture time (s)
    capture_match_window: float = 0.3
    # require speed==0 for at least this long to call a stop (s)
    min_stop_duration: float = 0.02
    stop_debounce: float = 0.15             # merge stops closer than this (s)

    # Boundary handling - removed boundary filtering
    # exclude_boundary_stops: bool = True  # No longer used
    # boundary_margin: float = 0.5         # No longer used

    # Normalization
    normalize: Optional[Literal["zscore", "sub", "div"]] = None
    # "zscore": z-score by pre-window baseline
    # "sub": subtract baseline mean
    # "div": divide by baseline mean (Fano-like)

    # Plotting
    ci_method: Literal["sem", "bootstrap"] = "sem"
    bootstrap_iters: int = 500
    alpha: float = 0.05                      # for bootstrap CIs


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
            Pre-filtered captures dataframe. If provided, must have 'stop_time' (or 'time') 
            and 'stop_point_index' (or 'point_index') columns. If provided along with 
            no_capture_stops_df, will use these instead of built-in stop detection.
        no_capture_stops_df : pd.DataFrame, optional
            Pre-filtered no-capture stops dataframe. If provided, must have 'stop_time' (or 'time') 
            and 'stop_point_index' (or 'point_index') columns. If provided along with 
            captures_df, will use these instead of built-in stop detection.
        """
        self.spikes_df = spikes_df.copy()
        self.spikes_df = self.spikes_df.sort_values(
            "time").reset_index(drop=True)
        if "cluster" not in self.spikes_df.columns:
            raise ValueError("spikes_df must have a 'cluster' column.")

        self.monkey_information = monkey_information.copy()
        if not {"time", "monkey_speeddummy"}.issubset(self.monkey_information.columns):
            raise ValueError(
                "monkey_information must have columns ['time', 'monkey_speeddummy'].")

        # Check if stop_event_id system is available, if not, warn user
        required_stop_cols = ["whether_new_distinct_stop", "stop_event_id"]
        missing_stop_cols = [
            col for col in required_stop_cols if col not in self.monkey_information.columns]
        if missing_stop_cols:
            print(
                f"Warning: Missing stop_event_id system columns: {missing_stop_cols}")
            print("Please run add_more_columns_to_monkey_information() on your monkey_information DataFrame first.")
            print("The PSTH analysis will use the fallback stop detection method.")

        self.monkey_information = self.monkey_information.sort_values(
            "time").reset_index(drop=True)
        self.ff_caught_T_new = np.asarray(ff_caught_T_new).astype(float)
        if self.ff_caught_T_new.size == 0:
            raise ValueError("ff_caught_T_new is empty.")

        self.config = config or PSTHConfig()

        # Store pre-filtered stop dataframes
        self.captures_df = captures_df
        self.no_capture_stops_df = no_capture_stops_df

        # Map cluster ids to contiguous indices
        self.clusters = np.array(
            sorted(self.spikes_df["cluster"].unique().tolist()))
        self.cluster_to_col = {c: i for i, c in enumerate(self.clusters)}
        self.n_clusters = len(self.clusters)

        # Results
        self.stop_events: Optional[pd.DataFrame] = None
        self.psth_data: Dict = {}


    def identify_stop_events(self) -> pd.DataFrame:
        """
        Combine captures_df and no_capture_stops_df into a single stops dataframe.
        Returns a DataFrame with columns ['stop_time', 'stop_point_index', 'stop_event_id'].
        """
        if self.captures_df is None or self.no_capture_stops_df is None:
            raise ValueError(
                "Both captures_df and no_capture_stops_df must be provided to use pre-filtered stops.")

        # Prepare captures dataframe
        captures_stops = self.captures_df.copy()
        # Add stop_event_id for captures (sequential numbering)
        captures_stops['stop_event_id'] = range(len(captures_stops))
        captures_stops['event_type'] = 'capture'

        # Prepare no-capture stops dataframe
        no_capture_stops = self.no_capture_stops_df.copy()
        # Add stop_event_id for no-capture stops (continue numbering from captures)
        no_capture_stops['stop_event_id'] = range(
            len(captures_stops), len(captures_stops) + len(no_capture_stops))
        no_capture_stops['event_type'] = 'miss'

        # Combine the dataframes
        combined_stops = pd.concat([
            captures_stops[['stop_time', 'stop_point_index', 'stop_event_id', 'event_type']],
            no_capture_stops[['stop_time', 'stop_point_index', 'stop_event_id', 'event_type']]
        ], ignore_index=True)

        # Sort by stop_time
        self.stop_events = combined_stops.sort_values(
            'stop_time').reset_index(drop=True)

        print('Combination of pre-filtered captures and no-capture stops complete.')


    def _make_time_edges_and_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time bin edges and centers for PSTH analysis.
        
        This method calculates the number of bins for pre- and post-stop windows
        and creates the corresponding time edges and bin centers.
        
        Returns
        -------
        edges : np.ndarray
            Time bin edges from -pre_window to +post_window, shape (n_bins + 1,)
        centers : np.ndarray
            Time bin centers, shape (n_bins,)
        """
        cfg = self.config
        
        # Calculate number of bins for pre-stop window
        # n_pre_bins = pre_window_duration / bin_width (rounded to nearest integer)
        # Default: pre_window=1.0s, bin_width=0.02s → n_pre_bins = 50 bins
        n_bins_pre = int(np.round(cfg.pre_window / cfg.bin_width))
        
        # Calculate number of bins for post-stop window
        # n_post_bins = post_window_duration / bin_width (rounded to nearest integer)
        # Default: post_window=1.0s, bin_width=0.02s → n_post_bins = 50 bins
        n_bins_post = int(np.round(cfg.post_window / cfg.bin_width))
        
        # Total number of bins across the entire time window
        n_bins = n_bins_pre + n_bins_post

        # Create time bin edges spanning from -pre_window to +post_window
        # edges shape: (n_bins + 1,) - one more edge than bins
        edges = np.linspace(-cfg.pre_window, cfg.post_window,
                            n_bins + 1, endpoint=True)
        
        # Calculate bin centers (midpoint of each bin)
        # centers shape: (n_bins,) - one center per bin
        centers = edges[:-1] + cfg.bin_width / 2.0
        
        return edges, centers

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

        edges, centers = self._make_time_edges_and_centers()
        n_bins = len(centers)

        segments = {"capture": [], "miss": []}

        # Pre-allocate view for speed
        times = self.spikes_df["time"].to_numpy()
        clus = self.spikes_df["cluster"].to_numpy()

        for _, ev in events.iterrows():
            t0 = float(ev["stop_time"])
            start, end = t0 - self.config.pre_window, t0 + self.config.post_window

            # window spikes
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")
            if right <= left:
                event_mat = np.zeros((n_bins, self.n_clusters), dtype=float)
                segments[ev["event_type"]].append(event_mat)
                continue

            rel_t = times[left:right] - t0
            rel_c = clus[left:right]

            # bin counts per cluster (vectorized via pandas crosstab)
            # bin index: 0..n_bins-1 (clip out-of-range)
            bin_idx = np.digitize(rel_t, edges, right=False) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            if not np.any(valid):
                event_mat = np.zeros((n_bins, self.n_clusters), dtype=float)
                segments[ev["event_type"]].append(event_mat)
                continue

            tab = pd.crosstab(bin_idx[valid], rel_c[valid], dropna=False)
            # align to full bin/cluster axes
            event_mat = np.zeros((n_bins, self.n_clusters), dtype=float)
            # columns in tab may be unordered; map to our cluster columns
            for c_in_tab in tab.columns:
                if c_in_tab in self.cluster_to_col:
                    col = self.cluster_to_col[c_in_tab]
                    event_mat[tab.index.to_numpy(
                    ), col] = tab[c_in_tab].to_numpy()

            segments[ev["event_type"]].append(event_mat)

        for k in segments:
            arr = np.stack(segments[k], axis=0) if len(
                segments[k]) else np.zeros((0, n_bins, self.n_clusters))
            segments[k] = arr

        return segments

    # ----------------------------- PSTH compute ------------------------------

    def _apply_smoothing(self, rate: np.ndarray) -> np.ndarray:
        """Gaussian smoothing along time axis with fractional sigma in bins."""
        sigma_bins = self.config.smoothing_sigma / self.config.bin_width
        if sigma_bins <= 0:
            return rate
        return gaussian_filter1d(rate, sigma=sigma_bins, axis=0, mode="reflect")

    def _baseline_stats(self, counts_trials: np.ndarray, pre_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate baseline firing rate statistics from the pre-stop window.
        
        This method computes the mean and standard deviation of firing rates across all trials
        and time bins in the pre-stop period. These baseline statistics are used for 
        normalization (z-score, subtract, or divide) to control for baseline differences
        between conditions or clusters.
        
        Parameters
        ----------
        counts_trials : np.ndarray
            Raw spike counts with shape (n_trials, n_bins, n_clusters)
        pre_mask : np.ndarray
            Boolean mask indicating which time bins belong to the pre-stop window
            
        Returns
        -------
        baseline_mean : np.ndarray
            Mean firing rate (Hz) for each cluster during pre-stop period, shape (n_clusters,)
        baseline_std : np.ndarray
            Standard deviation of firing rate (Hz) for each cluster during pre-stop period, shape (n_clusters,)
            Note: Zero standard deviations are set to 1.0 to avoid division by zero in normalization
        """
        if counts_trials.shape[0] == 0:
            return np.zeros((self.n_clusters,)), np.ones((self.n_clusters,))
        
        # Extract spike counts from pre-stop window across all trials and clusters
        # pre_counts shape: (n_trials, n_pre_bins, n_clusters)
        pre_counts = counts_trials[:, pre_mask, :]
        
        # Calculate mean firing rate across trials and time bins, convert from counts to Hz
        # baseline_mean shape: (n_clusters,)
        baseline_mean = pre_counts.mean(axis=(0, 1)) / self.config.bin_width
        
        # Calculate standard deviation of firing rate across trials and time bins, convert to Hz
        # baseline_std shape: (n_clusters,)
        baseline_std = pre_counts.std(axis=(0, 1), ddof=1) / self.config.bin_width
        
        # Prevent division by zero in normalization by setting zero std to 1.0
        baseline_std[baseline_std == 0] = 1.0
        
        return baseline_mean, baseline_std

    def _trial_rates(self, counts_trials: np.ndarray) -> np.ndarray:
        """Convert count matrices to firing rate per trial (Hz)."""
        return counts_trials / self.config.bin_width

    def _normalize(self, rates_trials: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray) -> np.ndarray:
        """Apply normalization in-place style and return new array."""
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
        Compute PSTH mean and (optional) CI for each condition.

        Returns
        -------
        dict with keys:
          - 'capture', 'miss': mean rate (n_bins, n_clusters) after smoothing & normalization
          - 'capture_sem', 'miss_sem': SEM (same shape) OR bootstrap CI half-width
          - 'time_axis': bin centers (s)
        """
        edges, centers = self._make_time_edges_and_centers()
        time_axis = centers
        n_bins = len(time_axis)

        # pre-window mask for baseline
        pre_mask = time_axis < 0

        out: Dict[str, np.ndarray] = {"time_axis": time_axis}

        def _per_condition(name: str):
            arr = segments.get(name, np.zeros((0, n_bins, self.n_clusters)))
            if arr.shape[0] < self.config.min_trials:
                # not enough trials: return zeros so downstream code keeps working
                out[name] = np.zeros((n_bins, self.n_clusters))
                out[name + "_sem"] = np.zeros((n_bins, self.n_clusters))
                return

            # baseline + normalization
            base_mu, base_sd = self._baseline_stats(arr, pre_mask)
            rates_trials = self._trial_rates(arr)
            rates_trials = self._normalize(rates_trials, base_mu, base_sd)

            # mean across trials
            mean_rate = rates_trials.mean(axis=0)  # (n_bins, n_clusters)
            mean_rate = self._apply_smoothing(mean_rate)

            # CI/SEM
            if self.config.ci_method == "sem":
                sem = rates_trials.std(axis=0, ddof=1) / \
                    np.sqrt(rates_trials.shape[0])
                sem = self._apply_smoothing(sem)
                out[name + "_sem"] = sem
            else:
                # bootstrap on trial axis
                it = self.config.bootstrap_iters
                boot = np.empty((it, n_bins, self.n_clusters))
                rng = np.random.default_rng(12345)
                for i in range(it):
                    idx = rng.integers(
                        0, rates_trials.shape[0], size=rates_trials.shape[0])
                    boot[i] = rates_trials[idx].mean(axis=0)
                # central CI half-width
                lo = np.percentile(boot, 100 * (self.config.alpha / 2), axis=0)
                hi = np.percentile(
                    boot, 100 * (1 - self.config.alpha / 2), axis=0)
                half = (hi - lo) / 2.0
                out[name + "_sem"] = self._apply_smoothing(half)

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

    def _plot_one(self, ax, time_axis, mean, ci, label):
        ax.plot(time_axis, mean, linewidth=2, label=label)
        ax.fill_between(time_axis, mean - ci, mean +
                        ci, alpha=0.25, linewidth=0)

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

        fig, axes = plt.subplots(
            len(list(cluster_indices)), 2, figsize=figsize, squeeze=False)
        time = psth["time_axis"]

        for row_i, ci in enumerate(cluster_indices):
            cid = self.clusters[ci]
            ax_c = axes[row_i, 0]
            ax_m = axes[row_i, 1]

            if n_events["capture"] >= self.config.min_trials:
                mean = psth["capture"][:, ci]
                ciw = psth["capture_sem"][:, ci]
                self._plot_one(ax_c, time, mean, ciw,
                               f"Capture (n={n_events['capture']})")
                if show_individual:
                    for trial in segments["capture"]:
                        ax_c.plot(
                            time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            if n_events["miss"] >= self.config.min_trials:
                mean = psth["miss"][:, ci]
                ciw = psth["miss_sem"][:, ci]
                self._plot_one(ax_m, time, mean, ciw,
                               f"Miss (n={n_events['miss']})")
                if show_individual:
                    for trial in segments["miss"]:
                        ax_m.plot(
                            time, (trial[:, ci] / self.config.bin_width), alpha=0.08)

            for ax in (ax_c, ax_m):
                ax.axvline(0, color="k", linestyle="--", alpha=0.5)
                ax.set_xlabel("Time relative to stop (s)")
                ax.set_ylabel(
                    "Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
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

        fig, axes = plt.subplots(
            len(list(cluster_indices)), 1, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, ci in zip(axes, cluster_indices):
            cid = self.clusters[ci]
            if n_events["capture"] >= self.config.min_trials:
                mc = psth["capture"][:, ci]
                cc = psth["capture_sem"][:, ci]
                self._plot_one(ax, time, mc, cc,
                               f"Capture (n={n_events['capture']})")

            if n_events["miss"] >= self.config.min_trials:
                mm = psth["miss"][:, ci]
                cm = psth["miss_sem"][:, ci]
                self._plot_one(ax, time, mm, cm,
                               f"Miss (n={n_events['miss']})")

            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel("Time relative to stop (s)")
            ax.set_ylabel(
                "Firing rate (Hz)" if self.config.normalize is None else "Normalized rate")
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

        # indices for time window (inclusive of start, inclusive of end nearest bin)
        start_idx = int(np.argmin(np.abs(time_axis - time_window[0])))
        end_idx = int(np.argmin(np.abs(time_axis - time_window[1])))
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        results: Dict = {}
        for ci in range(self.n_clusters):
            cid = self.clusters[ci]

            def _collect(name: str) -> List[float]:
                arr = segments.get(name, np.zeros(
                    (0, len(time_axis), self.n_clusters)))
                if arr.shape[0] == 0:
                    return []
                # convert to Hz
                rates = arr[:, start_idx:end_idx + 1,
                            ci].mean(axis=1) / cfg.bin_width
                return rates.tolist()

            cap = _collect("capture")
            mis = _collect("miss")

            if len(cap) >= cfg.min_trials and len(mis) >= cfg.min_trials:
                stat, p = stats.mannwhitneyu(cap, mis, alternative="two-sided")
                # Cohen's d (pooled std)
                mean_diff = float(np.mean(cap) - np.mean(mis))
                pooled_sd = np.sqrt(((len(cap) - 1)*np.var(cap, ddof=1) + (len(mis) - 1)*np.var(mis, ddof=1))
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
                results[str(cid)] = {
                    "error": "Insufficient data for comparison"}

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
        spikes_df, monkey_information, ff_caught_T_new, config, captures_df, no_capture_stops_df)
    analyzer.run_full_analysis(cluster_idx)
    return analyzer


