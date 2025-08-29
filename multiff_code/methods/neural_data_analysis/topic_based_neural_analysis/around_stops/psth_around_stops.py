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

        # Check if stop_id system is available, if not, warn user
        required_stop_cols = ["whether_new_distinct_stop", "stop_id"]
        missing_stop_cols = [
            col for col in required_stop_cols if col not in self.monkey_information.columns]
        if missing_stop_cols:
            print(
                f"Warning: Missing stop_id system columns: {missing_stop_cols}")
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

    # ------------------------- Event identification --------------------------

    def _find_stops_from_speed(self) -> pd.DataFrame:
        """
        Use pre-filtered stop dataframes if available, otherwise fall back to built-in methods.
        Returns a DataFrame with columns ['stop_time', 'stop_point_index', 'stop_id'].
        """
        # If pre-filtered dataframes are provided, use them
        if self.captures_df is not None and self.no_capture_stops_df is not None:
            return self._combine_pre_filtered_stops()
        else:
            print('no pre-filtered stop dataframes provided')

        # Otherwise, use the original methods
        # Check if the required columns exist
        required_cols = ["time", "monkey_speeddummy",
                         "whether_new_distinct_stop", "stop_id"]
        missing_cols = [
            col for col in required_cols if col not in self.monkey_information.columns]

        if missing_cols:
            # Fallback to original method if stop_id system is not available
            return self._find_stops_from_speed_fallback()

        # Get distinct stops using the existing stop_id system
        distinct_stops = self.monkey_information[
            self.monkey_information['whether_new_distinct_stop'] == True
        ].copy()

        if distinct_stops.empty:
            return pd.DataFrame(columns=['stop_time', 'stop_point_index', 'stop_id'])

        # Get stop information - handle missing columns gracefully
        available_cols = ['time', 'stop_id']
        if 'point_index' in distinct_stops.columns:
            available_cols.append('point_index')

        stops_df = distinct_stops[available_cols].copy()
        stops_df.rename(columns={'time': 'stop_time'}, inplace=True)

        # Add point_index if it doesn't exist
        if 'point_index' not in stops_df.columns:
            stops_df['point_index'] = -1

        # Rename point_index to stop_point_index for consistency
        stops_df.rename(
            columns={'point_index': 'stop_point_index'}, inplace=True)

        # Apply minimum duration filter if needed
        if hasattr(self.config, 'min_stop_duration') and self.config.min_stop_duration > 0:
            # Calculate stop durations by finding the next distinct stop or end of data
            stop_durations = []
            for i, (_, row) in enumerate(stops_df.iterrows()):
                current_stop_id = row['stop_id']
                current_time = row['stop_time']

                # Find the next distinct stop
                next_stop_mask = (self.monkey_information['stop_id'] == current_stop_id + 1) & \
                    (self.monkey_information['whether_new_distinct_stop'] == True)

                if next_stop_mask.any():
                    next_time = self.monkey_information.loc[next_stop_mask,
                                                            'time'].iloc[0]
                    duration = next_time - current_time
                else:
                    # If no next stop, use the end of the data
                    duration = self.monkey_information['time'].max(
                    ) - current_time

                stop_durations.append(duration)

            stops_df['stop_duration'] = stop_durations
            stops_df = stops_df[stops_df['stop_duration']
                                >= self.config.min_stop_duration].copy()
            stops_df.drop(columns=['stop_duration'], inplace=True)

        # Apply debouncing if needed
        if hasattr(self.config, 'stop_debounce') and self.config.stop_debounce > 0 and len(stops_df) > 1:
            stops_df = stops_df.sort_values('stop_time').reset_index(drop=True)
            merged = [stops_df.iloc[0].to_dict()]

            for _, row in stops_df.iloc[1:].iterrows():
                if row["stop_time"] - merged[-1]["stop_time"] < self.config.stop_debounce:
                    # Keep the first one; alternatively average them
                    continue
                merged.append(row.to_dict())

            stops_df = pd.DataFrame(merged)

        return stops_df.reset_index(drop=True)

    def _combine_pre_filtered_stops(self) -> pd.DataFrame:
        """
        Combine captures_df and no_capture_stops_df into a single stops dataframe.
        Returns a DataFrame with columns ['stop_time', 'stop_point_index', 'stop_id'].
        """
        if self.captures_df is None or self.no_capture_stops_df is None:
            raise ValueError(
                "Both captures_df and no_capture_stops_df must be provided to use pre-filtered stops.")

        # Prepare captures dataframe
        captures_stops = self.captures_df.copy()
        if 'stop_time' not in captures_stops.columns:
            if 'time' in captures_stops.columns:
                captures_stops['stop_time'] = captures_stops['time']
            else:
                raise ValueError(
                    "captures_df must have either 'stop_time' or 'time' column")

        if 'stop_point_index' not in captures_stops.columns:
            if 'point_index' in captures_stops.columns:
                captures_stops['stop_point_index'] = captures_stops['point_index']
            else:
                captures_stops['stop_point_index'] = -1

        # Add stop_id for captures (sequential numbering)
        captures_stops['stop_id'] = range(len(captures_stops))

        # Prepare no-capture stops dataframe
        no_capture_stops = self.no_capture_stops_df.copy()
        if 'stop_time' not in no_capture_stops.columns:
            if 'time' in no_capture_stops.columns:
                no_capture_stops['stop_time'] = no_capture_stops['time']
            else:
                raise ValueError(
                    "no_capture_stops_df must have either 'stop_time' or 'time' column")

        if 'stop_point_index' not in no_capture_stops.columns:
            if 'point_index' in no_capture_stops.columns:
                no_capture_stops['stop_point_index'] = no_capture_stops['point_index']
            else:
                no_capture_stops['stop_point_index'] = -1

        # Add stop_id for no-capture stops (continue numbering from captures)
        no_capture_stops['stop_id'] = range(
            len(captures_stops), len(captures_stops) + len(no_capture_stops))

        # Combine the dataframes
        combined_stops = pd.concat([
            captures_stops[['stop_time', 'stop_point_index', 'stop_id']],
            no_capture_stops[['stop_time', 'stop_point_index', 'stop_id']]
        ], ignore_index=True)

        # Sort by stop_time
        combined_stops = combined_stops.sort_values(
            'stop_time').reset_index(drop=True)

        print('Combination of pre-filtered captures and no-capture stops complete.')

        return combined_stops

    def _find_stops_from_speed_fallback(self) -> pd.DataFrame:
        """
        Fallback method: Debounced detection of stops from 'monkey_speeddummy'==0 with a minimum duration.
        Returns a DataFrame with columns ['stop_time', 'stop_point_index', 'stop_id'].
        """
        cfg = self.config
        df = self.monkey_information[["time", "monkey_speeddummy"]].copy()
        time = df["time"].to_numpy()
        speed0 = (df["monkey_speeddummy"].to_numpy() == 0).astype(int)

        # detect transitions 1->0 (start) and 0->1 (end)
        diff = np.diff(speed0, prepend=speed0[0])
        start_idx = np.where(diff == -1)[0]  # went from 1 to 0
        end_idx = np.where(diff == +1)[0]    # went from 0 to 1

        # handle leading/trailing zeros
        if speed0[0] == 0:
            start_idx = np.r_[0, start_idx]
        if speed0[-1] == 0:
            end_idx = np.r_[end_idx, len(speed0) - 1]

        stops = []
        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            t0, t1 = time[s], time[e]
            dur = max(0.0, t1 - t0)
            if dur >= cfg.min_stop_duration:
                # take the *first* moment of stop as the stop time (you can change to middle)
                stops.append({
                    "stop_time": float(t0),
                    "stop_id": i  # Simple sequential ID for fallback
                })

        stops_df = pd.DataFrame(stops)

        # Add stop_point_index column
        if "point_index" in self.monkey_information.columns and len(stops_df) > 0:
            # nearest point_index at/just before stop_time
            pi = []
            pi_series = self.monkey_information[[
                "time", "point_index"]].dropna()
            times_pi = pi_series["time"].to_numpy()
            pidx = pi_series["point_index"].to_numpy()
            for t in stops_df["stop_time"].to_numpy():
                k = np.searchsorted(times_pi, t, side="right") - 1
                k = np.clip(k, 0, len(times_pi) - 1)
                pi.append(int(pidx[k]))
            stops_df["stop_point_index"] = pi
        else:
            stops_df["stop_point_index"] = -1

        # Debounce: merge stops closer than cfg.stop_debounce
        if len(stops_df) > 1 and cfg.stop_debounce > 0:
            merged = [stops_df.iloc[0].to_dict()]
            for _, row in stops_df.iloc[1:].iterrows():
                if row["stop_time"] - merged[-1]["stop_time"] < cfg.stop_debounce:
                    # keep the first one; alternatively average them
                    continue
                merged.append(row.to_dict())
            stops_df = pd.DataFrame(merged)

        return stops_df.reset_index(drop=True)

    def identify_stop_events(self) -> pd.DataFrame:
        """
        Identify stop events and classify them as captures or misses.

        Returns
        -------
        DataFrame with columns:
          - stop_time
          - stop_point_index
          - event_type: 'capture' or 'miss'
          - trial_index: index into ff_caught_T_new the stop is associated with
          - time_to_capture: |stop_time - nearest_capture_time|
        """
        cfg = self.config

        # If pre-filtered dataframes are provided, use them directly
        if self.captures_df is not None and self.no_capture_stops_df is not None:
            return self._classify_pre_filtered_stops()

        # Otherwise, use the original method
        stops = self._find_stops_from_speed()
        if stops.empty:
            self.stop_events = pd.DataFrame(columns=[
                'stop_time', 'stop_point_index', 'stop_id',
                'event_type', 'trial_index', 'time_to_capture'
            ])
            return self.stop_events

        # Ensure stop_point_index column exists
        if 'stop_point_index' not in stops.columns:
            stops['stop_point_index'] = -1

        cap_times = self.ff_caught_T_new
        stop_events = []
        for _, row in stops.iterrows():
            t = float(row["stop_time"])
            # nearest capture
            idx = np.argmin(np.abs(cap_times - t))
            dt = float(abs(cap_times[idx] - t))
            if dt <= cfg.capture_match_window:
                event_type = "capture"
                trial_index = idx
                time_to_capture = 0.0
            else:
                event_type = "miss"
                # which trial are we in? nearest previous capture
                trial_index = int(np.searchsorted(cap_times, t))
                trial_index = int(np.clip(trial_index, 0, len(cap_times) - 1))
                time_to_capture = dt

            stop_events.append(
                {
                    "stop_time": t,
                    "stop_point_index": int(row.get("stop_point_index", -1)),
                    "stop_id": int(row["stop_id"]),
                    "event_type": event_type,
                    "trial_index": trial_index,
                    "time_to_capture": time_to_capture,
                }
            )

        events = pd.DataFrame(stop_events)

        # Removed boundary stop filtering - keeping all stops
        self.stop_events = events.reset_index(drop=True)
        return self.stop_events

    def _classify_pre_filtered_stops(self) -> pd.DataFrame:
        """
        Classify pre-filtered stops as captures or misses based on the provided dataframes.
        """
        if self.captures_df is None or self.no_capture_stops_df is None:
            raise ValueError(
                "Both captures_df and no_capture_stops_df must be provided.")

        cap_times = self.ff_caught_T_new
        stop_events = []

        # Process captures
        for _, row in self.captures_df.iterrows():
            t = float(row.get('stop_time', row.get('time', 0)))
            point_index = int(row.get('stop_point_index',
                              row.get('point_index', -1)))

            # Find the closest capture time
            idx = np.argmin(np.abs(cap_times - t))
            dt = float(abs(cap_times[idx] - t))

            stop_events.append({
                "stop_time": t,
                "stop_point_index": point_index,
                "stop_id": len(stop_events),
                "event_type": "capture",
                "trial_index": idx,
                "time_to_capture": 0.0,
            })

        # Process no-capture stops (misses)
        for _, row in self.no_capture_stops_df.iterrows():
            t = float(row.get('stop_time', row.get('time', 0)))
            point_index = int(row.get('stop_point_index',
                              row.get('point_index', -1)))

            # Find which trial this miss belongs to
            trial_index = int(np.searchsorted(cap_times, t))
            trial_index = int(np.clip(trial_index, 0, len(cap_times) - 1))

            # Calculate time to nearest capture
            dt = float(abs(cap_times[trial_index] - t))

            stop_events.append({
                "stop_time": t,
                "stop_point_index": point_index,
                "stop_id": len(stop_events),
                "event_type": "miss",
                "trial_index": trial_index,
                "time_to_capture": dt,
            })

        # Create DataFrame and sort by stop_time
        events = pd.DataFrame(stop_events)
        events = events.sort_values('stop_time').reset_index(drop=True)

        self.stop_events = events
        return self.stop_events

    # Removed _filter_boundary_stops method - no longer filtering boundary stops

    # ------------------------ Diagnostic Functions ---------------------------

    def diagnostic_capture_analysis(self) -> Dict:
        """
        Diagnostic analysis to identify which captures are missing from PSTH analysis.

        Returns:
        --------
        Dict containing:
            - 'total_captures': total number of captures in ff_caught_T_new
            - 'found_captures': number of captures found by PSTH analysis
            - 'missing_captures': list of trial indices with missing captures
            - 'missing_capture_times': list of capture times that were not found
            - 'found_capture_trials': list of trial indices where captures were found
            - 'detailed_missing_info': DataFrame with details about missing captures
        """
        if self.stop_events is None:
            self.identify_stop_events()

        # Get all expected captures
        expected_captures = set(range(len(self.ff_caught_T_new)))

        # Get found captures
        found_captures = self.stop_events[self.stop_events['event_type'] == 'capture']
        found_trials = set(found_captures['trial_index'].tolist())

        # Find missing captures
        missing_trials = expected_captures - found_trials
        missing_capture_times = [self.ff_caught_T_new[i]
                                 for i in missing_trials]

        # Create detailed missing info
        missing_info = []
        for trial_idx in missing_trials:
            capture_time = self.ff_caught_T_new[trial_idx]

            # Find the nearest stop to this capture time
            if not self.stop_events.empty:
                nearest_stop_idx = np.argmin(
                    np.abs(self.stop_events['stop_time'] - capture_time))
                nearest_stop_time = self.stop_events.iloc[nearest_stop_idx]['stop_time']
                time_diff = abs(nearest_stop_time - capture_time)
                nearest_event_type = self.stop_events.iloc[nearest_stop_idx]['event_type']
            else:
                nearest_stop_time = np.nan
                time_diff = np.nan
                nearest_event_type = 'no_stops'

            missing_info.append({
                'trial_index': trial_idx,
                'capture_time': capture_time,
                'nearest_stop_time': nearest_stop_time,
                'time_diff_to_nearest_stop': time_diff,
                'nearest_event_type': nearest_event_type,
                'reason': self._classify_missing_reason(trial_idx, capture_time, time_diff, nearest_event_type)
            })

        detailed_missing_info = pd.DataFrame(missing_info)

        return {
            'total_captures': len(expected_captures),
            'found_captures': len(found_trials),
            'missing_captures': list(missing_trials),
            'missing_capture_times': missing_capture_times,
            'found_capture_trials': list(found_trials),
            'detailed_missing_info': detailed_missing_info
        }

    def _classify_missing_reason(self, trial_idx: int, capture_time: float,
                                 time_diff: float, nearest_event_type: str) -> str:
        """Classify why a capture was missing from PSTH analysis."""
        cfg = self.config

        if nearest_event_type == 'no_stops':
            return "No stops detected in entire session"
        elif time_diff > cfg.capture_match_window:
            return f"No stop within {cfg.capture_match_window}s of capture (nearest: {time_diff:.3f}s)"
        elif nearest_event_type == 'capture':
            return "Stop found but assigned to different trial"
        elif nearest_event_type == 'miss':
            return f"Stop classified as miss (within {cfg.capture_match_window}s but not closest)"
        else:
            return "Unknown reason"

    def print_capture_diagnostic(self) -> None:
        """Print a detailed diagnostic report of missing captures."""
        diagnostic = self.diagnostic_capture_analysis()

        print("=" * 60)
        print("CAPTURE DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"Total captures expected: {diagnostic['total_captures']}")
        print(f"Captures found in PSTH: {diagnostic['found_captures']}")
        print(f"Missing captures: {len(diagnostic['missing_captures'])}")
        print(
            f"Success rate: {diagnostic['found_captures']/diagnostic['total_captures']*100:.1f}%")

        if diagnostic['missing_captures']:
            print(
                f"\nMissing trial indices: {sorted(diagnostic['missing_captures'])}")
            print(f"\nDetailed missing capture information:")
            print(diagnostic['detailed_missing_info'].to_string(index=False))
        else:
            print("\nAll captures were successfully identified!")

        print("=" * 60)

    def plot_missing_captures(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot to visualize missing captures vs found captures.

        Returns:
        --------
        matplotlib.Figure
        """
        diagnostic = self.diagnostic_capture_analysis()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: Timeline of captures
        ax1.plot(self.ff_caught_T_new, np.ones_like(self.ff_caught_T_new), 'go',
                 markersize=8, label='All captures', alpha=0.7)

        if diagnostic['found_capture_trials']:
            found_times = [self.ff_caught_T_new[i]
                           for i in diagnostic['found_capture_trials']]
            ax1.plot(found_times, np.ones_like(found_times), 'bo',
                     markersize=6, label='Found captures', alpha=0.9)

        if diagnostic['missing_capture_times']:
            ax1.plot(diagnostic['missing_capture_times'], np.ones_like(diagnostic['missing_capture_times']), 'ro',
                     markersize=6, label='Missing captures', alpha=0.9)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Capture Status')
        ax1.set_title('Capture Timeline: Found vs Missing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trial index vs capture status
        all_trials = list(range(len(self.ff_caught_T_new)))
        found_mask = [i in diagnostic['found_capture_trials']
                      for i in all_trials]
        missing_mask = [i in diagnostic['missing_captures']
                        for i in all_trials]

        ax2.plot([i for i, found in zip(all_trials, found_mask) if found],
                 [1 for _ in range(sum(found_mask))], 'bo', markersize=6, label='Found captures')
        ax2.plot([i for i, missing in zip(all_trials, missing_mask) if missing],
                 [1 for _ in range(sum(missing_mask))], 'ro', markersize=6, label='Missing captures')

        ax2.set_xlabel('Trial Index')
        ax2.set_ylabel('Capture Status')
        ax2.set_title('Capture Status by Trial Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ------------------------ Segments & Binning -----------------------------

    def _make_time_edges_and_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n_bins_pre = int(np.round(cfg.pre_window / cfg.bin_width))
        n_bins_post = int(np.round(cfg.post_window / cfg.bin_width))
        n_bins = n_bins_pre + n_bins_post

        # edges from -pre to +post with n_bins bins => n_bins+1 edges
        edges = np.linspace(-cfg.pre_window, cfg.post_window,
                            n_bins + 1, endpoint=True)
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
        counts_trials: (n_trials, n_bins, n_clusters) raw counts
        returns (baseline_mean, baseline_std) with shape (n_clusters,)
        """
        if counts_trials.shape[0] == 0:
            return np.zeros((self.n_clusters,)), np.ones((self.n_clusters,))
        # mean rate over pre-window for each trial & cluster
        # (n_trials, n_pre, n_clusters)
        pre_counts = counts_trials[:, pre_mask, :]
        baseline_mean = pre_counts.mean(axis=(0, 1)) / self.config.bin_width
        baseline_std = pre_counts.std(
            axis=(0, 1), ddof=1) / self.config.bin_width
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


def export_psth_to_df(
    analyzer: "PSTHAnalyzer",
    clusters: Optional[List[int]] = None,
    include_ci: bool = True,
) -> pd.DataFrame:
    """
    Flatten cached PSTH results into a tidy DataFrame.

    Parameters
    ----------
    analyzer : PSTHAnalyzer
        Instance that has already run .run_full_analysis() (will run if needed).
    clusters : list of int or None
        Optional subset of cluster indices (0-based, matching analyzer.clusters' order).
        If None, export all clusters.
    include_ci : bool
        If True, include 'sem', 'lower', 'upper' columns.

    Returns
    -------
    pd.DataFrame with columns:
        ['time', 'cluster', 'condition', 'mean']  (+ ['sem','lower','upper'] if include_ci)

    Notes
    -----
    - 'cluster' column uses the original cluster IDs (analyzer.clusters values),
      not the 0-based column indices.
    - 'mean' and 'sem' reflect whatever normalization and CI method were configured.
    """
    # Ensure results exist
    if not getattr(analyzer, "psth_data", None):
        analyzer.run_full_analysis()

    psth = analyzer.psth_data["psth"]  # type: ignore[assignment]
    time = psth["time_axis"]
    all_idx = range(len(analyzer.clusters))
    idxs = list(all_idx) if clusters is None else list(clusters)

    # Helper to build one condition block
    def _block(cond_key: str, label: str) -> pd.DataFrame:
        mean = psth[cond_key][:, idxs]              # (n_bins, n_sel_clusters)
        sem = psth[cond_key + "_sem"][:, idxs]     # same shape

        # Repeat time for each selected cluster
        # (n_bins, n_sel_clusters)
        t_rep = np.tile(time[:, None], (1, len(idxs)))
        cl_ids = analyzer.clusters[idxs]
        cl_rep = np.tile(cl_ids[None, :], (len(time), 1))

        data = {
            "time": t_rep.ravel(),
            "cluster": cl_rep.ravel(),
            "condition": np.full(t_rep.size, label, dtype=object),
            "mean": mean.ravel(),
        }
        if include_ci:
            data["sem"] = sem.ravel()
            data["lower"] = (mean - sem).ravel()
            data["upper"] = (mean + sem).ravel()

        return pd.DataFrame(data)

    out_frames: List[pd.DataFrame] = []
    # Only append if there are any events for that condition
    if analyzer.psth_data["n_events"]["capture"] > 0:
        out_frames.append(_block("capture", "capture"))
    if analyzer.psth_data["n_events"]["miss"] > 0:
        out_frames.append(_block("miss", "miss"))

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(
        columns=(["time", "cluster", "condition", "mean"] +
                 (["sem", "lower", "upper"] if include_ci else []))
    )


def _bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR correction.
    Returns array of booleans for which hypotheses are rejected.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.arange(1, n+1)
    thresh = alpha * ranked / n
    passed = p[order] <= thresh
    # make it monotone
    if np.any(passed):
        kmax = np.max(np.where(passed)[0])
        passed[:kmax+1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = passed
    return out


def compare_windows(analyzer, windows, alpha=0.05):
    """
    Run analyzer.statistical_comparison() on multiple windows and
    return a tidy DataFrame with FDR-corrected significance flags.
    """
    rows = []
    for win_name, (a, b) in windows.items():
        stats = analyzer.statistical_comparison(time_window=(a, b))
        for cl_id, d in stats.items():
            if "error" in d:
                rows.append({
                    "cluster": cl_id, "window": win_name, "p": np.nan,
                    "U": np.nan, "cohens_d": np.nan,
                    "cap_mean": np.nan, "miss_mean": np.nan,
                    "n_cap": d.get("n_captures", 0), "n_miss": d.get("n_misses", 0),
                    "sig_FDR": False
                })
            else:
                rows.append({
                    "cluster": cl_id, "window": win_name, "p": d["p_value"],
                    "U": d["statistic_U"], "cohens_d": d["cohens_d"],
                    "cap_mean": d["capture_mean"], "miss_mean": d["miss_mean"],
                    "n_cap": d["n_captures"], "n_miss": d["n_misses"],
                    "sig_FDR": False  # temp, fill later
                })
    df = pd.DataFrame(rows)

    # FDR within each window across clusters
    out = []
    for w, g in df.groupby("window", dropna=False):
        mask = g["p"].notna()
        sig = np.full(len(g), False)
        if mask.any():
            sig_indices = _bh_fdr(g.loc[mask, "p"].values, alpha=alpha)
            sig[np.where(mask)[0]] = sig_indices
        gg = g.copy()
        gg.loc[:, "sig_FDR"] = sig
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def summarize_epochs(analyzer, alpha=0.05):
    """
    Run statistical comparisons across three canonical epochs:
    - pre_bump (-0.3–0.0 s)
    - early_dip (0.0–0.3 s)
    - late_rebound (0.3–0.8 s)

    Returns
    -------
    pd.DataFrame with columns:
      cluster, window, p, cohens_d, cap_mean, miss_mean,
      n_cap, n_miss, sig_FDR
    """
    windows = {
        "pre_bump(-0.3–0.0)": (-0.3, 0.0),
        "early_dip(0.0–0.3)": (0.0, 0.3),
        "late_rebound(0.3–0.8)": (0.3, 0.8),
    }
    return compare_windows(analyzer, windows, alpha=alpha)


def plot_effect_heatmap_all(summary: pd.DataFrame,
                            title="Capture − Miss effects (Cohen's d)",
                            grey_color="lightgrey",
                            order: str = "effect"):
    """
    Heatmap of Cohen's d across clusters × epochs.
    - Significant cells (FDR) show d with a diverging colormap.
    - Non-significant cells are grey.
    - Includes ALL clusters, even if nothing is significant.

    Parameters
    ----------
    summary : DataFrame
        Output from summarize_epochs()/compare_windows().
        Must have columns ['cluster','window','cohens_d','sig_FDR'].
    title : str
        Plot title.
    grey_color : str
        Color for non-significant cells.
    order : {'effect','cluster'}
        'effect': sort rows by strongest |d| among significant cells (descending).
        'cluster': sort rows by cluster label.

    Returns
    -------
    (fig, ax)
    """
    df = summary.copy()

    # Keep all rows but mask non-significant d as NaN (so they render grey)
    d_masked = df["cohens_d"].where(df["sig_FDR"], np.nan)
    df = df.assign(d_masked=d_masked)

    # Pivot to clusters × windows (may contain NaNs)
    pivot = df.pivot_table(index="cluster",
                           columns="window",
                           values="d_masked",
                           aggfunc="mean")

    # Ensure ALL clusters/windows appear (even if all NaN)
    all_clusters = pd.Index(df["cluster"].astype(str).unique(), name="cluster")
    all_windows = pd.Index(df["window"].unique(), name="window")
    pivot = pivot.reindex(index=all_clusters, columns=all_windows)

    # Row ordering
    if order == "effect":
        # strongest absolute effect across epochs (ignoring NaNs)
        strength = pivot.abs().max(axis=1).fillna(0.0)
        pivot = pivot.loc[strength.sort_values(ascending=False).index]
    elif order == "cluster":
        pivot = pivot.sort_index()
    else:
        raise ValueError("order must be 'effect' or 'cluster'")

    # Colormap with grey for NaNs
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(grey_color)

    # Symmetric color scale by max |d| among significant cells
    vmax = np.nanmax(np.abs(pivot.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0  # fallback if nothing significant at all

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d (capture − miss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cluster")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax


