import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import (
    build_continuous_fr,
    events_to_bins,
    extract_event_windows,
    fit_behavior_axis,
    cross_validate_axis,
    orthogonalize_axes,
    axis_angle
)

from .axis_visualization import (
    plot_projection,
    plot_event_projection_scatter,
    plot_event_projection_hist,
    plot_projection_with_events,
    plot_event_aligned_projection,
    diagnose_axis,
    plot_3d_projection
)


class ContinuousBehaviorAxisAnalyzer:
    def __init__(
        self,
        spikes_df,
        event_a_df,
        event_b_df,
        event_col='stop_time',
        event_a_label='A',
        event_b_label='B',
        bin_width_ms=10.0,
        smoothing_sigma_ms=30.0,
        external_fr_mat=None,      # NEW
        external_start_time=None,  # NEW
        external_clusters=None,    # NEW
    ):
        self.event_a_label = event_a_label
        self.event_b_label = event_b_label

        self.event_a_times = event_a_df[event_col].to_numpy()
        self.event_b_times = event_b_df[event_col].to_numpy()

        spikes_df = spikes_df.sort_values('time')
        clusters = np.array(sorted(spikes_df['cluster'].unique()))
        self.clusters = clusters
        self.cluster_to_idx = {c: i for i, c in enumerate(clusters)}

        spike_codes = np.array([self.cluster_to_idx[c]
                               for c in spikes_df['cluster']])
        spike_times = spikes_df['time'].to_numpy(float)

        self.bin_width_ms = bin_width_ms
        self.bin_width_s = bin_width_ms / 1000.0

        if external_fr_mat is not None:
            # Use LFADS (or other) FR matrix
            self.fr_mat = external_fr_mat
            self.start_time = float(
                external_start_time) if external_start_time is not None else float(spike_times.min())
            # If LFADS was trained on a subset of clusters, make sure they match
            if external_clusters is not None:
                self.clusters = external_clusters
        else:
            # original path: build continuous FR from spikes
            from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import build_continuous_fr
            self.fr_mat, self.start_time = build_continuous_fr(
                spike_times, spike_codes, len(clusters),
                bin_width_ms, smoothing_sigma_ms
            )

    # -----------------------------
    def _events_to_bins(self, times):
        return events_to_bins(times, self.start_time, self.bin_width_s)

    def _preprocess_fr(self, fr):
        # variance-stabilize
        fr = np.sqrt(np.maximum(fr, 0))

        # z-score per neuron
        mu = fr.mean(axis=0, keepdims=True)
        sd = fr.std(axis=0, keepdims=True) + 1e-6
        return (fr - mu) / sd

    # -----------------------------

    def build_event_vectors(self, window_a_ms, window_b_ms):

        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)

        windows_A = extract_event_windows(
            self.fr_mat, a_bins, window_a_ms, self.bin_width_ms)
        windows_B = extract_event_windows(
            self.fr_mat, b_bins, window_b_ms, self.bin_width_ms)

        XA = windows_A.mean(axis=1)
        XB = windows_B.mean(axis=1)

        X = np.vstack([XA, XB])
        y = np.hstack([np.zeros(len(XA)), np.ones(len(XB))])
        return X, y

    # -----------------------------
    def compute_event_axis(self, window_a_ms, window_b_ms, model='logreg', **kwargs):
        X, y = self.build_event_vectors(window_a_ms, window_b_ms)

        readout, axis = fit_behavior_axis(X, y, model=model, **kwargs)

        # Compute continuous projection
        if axis is not None:
            projection = self.fr_mat @ axis
        else:
            if hasattr(readout, 'decision_function'):
                projection = readout.decision_function(self.fr_mat)
            elif hasattr(readout, 'predict_proba'):
                projection = readout.predict_proba(self.fr_mat)[:, 1]
            else:
                projection = readout.predict(self.fr_mat)

        return {
            'axis_vec': axis,
            'projection': projection,
            'readout': readout,
            'readout_type': model,
            'X_train': X,
            'y_train': y,
            'clusters': self.clusters,
        }

    # -----------------------------
    def cross_validate_axis(self, window_a_ms, window_b_ms, model="logreg", n_splits=5, **kwargs):
        X, y = self.build_event_vectors(window_a_ms, window_b_ms)
        return cross_validate_axis(X, y, model=model, n_splits=n_splits, **kwargs)

    # -----------------------------
    # Visualization wrappers (optional)
    def plot_projection(self, axis_info, title=None):
        T = len(axis_info["projection"])
        time = np.arange(T) * self.bin_width_s
        plot_projection(axis_info["projection"], time, title)

    def plot_event_projection_scatter(self, axis_info):
        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)
        plot_event_projection_scatter(axis_info["projection"], a_bins, b_bins,
                                      self.event_a_label, self.event_b_label)

    def plot_event_projection_hist(self, axis_info):
        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)
        plot_event_projection_hist(axis_info["projection"], a_bins, b_bins,
                                   self.event_a_label, self.event_b_label)

    def plot_projection_with_events(self, axis_info, **kwargs):
        T = len(axis_info["projection"])
        time = np.arange(T)*self.bin_width_s
        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)
        plot_projection_with_events(axis_info["projection"], time, a_bins, b_bins,
                                    self.event_a_label, self.event_b_label, **kwargs)

    def plot_event_aligned_projection(self, axis_info, window_ms=(-200, 400), **kwargs):
        start_ms, end_ms = window_ms
        so = int(start_ms/self.bin_width_ms)
        eo = int(end_ms/self.bin_width_ms)
        offsets = np.arange(so, eo)

        proj = axis_info["projection"]

        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)

        idx_A = a_bins[:, None] + offsets[None, :]
        idx_B = b_bins[:, None] + offsets[None, :]
        T = len(proj)
        idx_A = np.clip(idx_A, 0, T-1)
        idx_B = np.clip(idx_B, 0, T-1)
        aligned_A = proj[idx_A]
        aligned_B = proj[idx_B]

        time = offsets * self.bin_width_s

        plot_event_aligned_projection(
            aligned_A, aligned_B, time,
            label_a=self.event_a_label,
            label_b=self.event_b_label,
            **kwargs
        )

    def diagnose_axis(self, axis_info, cv_results=None, window_ms=None):
        if window_ms is None:
            a_bins = self._events_to_bins(self.event_a_times)
            b_bins = self._events_to_bins(self.event_b_times)
            diagnose_axis(axis_info["projection"], a_bins, b_bins,
                          self.event_a_label, self.event_b_label,
                          cv_results=cv_results)
            return

        # Use window: average projection over the window for each event
        start_ms, end_ms = window_ms
        so = int(start_ms / self.bin_width_ms)
        eo = int(end_ms / self.bin_width_ms)
        offsets = np.arange(so, eo)

        proj = axis_info["projection"]
        a_bins = self._events_to_bins(self.event_a_times)
        b_bins = self._events_to_bins(self.event_b_times)

        idx_A = a_bins[:, None] + offsets[None, :]
        idx_B = b_bins[:, None] + offsets[None, :]
        T = len(proj)
        idx_A = np.clip(idx_A, 0, T - 1)
        idx_B = np.clip(idx_B, 0, T - 1)

        aligned_A = proj[idx_A]
        aligned_B = proj[idx_B]

        a_vals = aligned_A.mean(axis=1)
        b_vals = aligned_B.mean(axis=1)

        diagnose_axis(a_values=a_vals, b_values=b_vals,
                      label_a=self.event_a_label, label_b=self.event_b_label,
                      cv_results=cv_results)

    # -----------------------------
    # Orthogonalization & angle (pass-through)
    def orthogonalize(self, axes):
        return orthogonalize_axes(axes)

    def axis_angle(self, a, b):
        return axis_angle(a, b)
