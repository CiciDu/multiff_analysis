# multivariate_axis_analyzer.py
import numpy as np

from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import (
    build_continuous_fr,
    events_to_bins,
    extract_event_windows,
    orthogonalize_axes,
    axis_angle,
)

from .multivariate_axis_utils import (
    build_interaction_terms,
    fit_multitask_linear,
    reduced_rank_regression,
)

from .conditioned_timecourse import ConditionedTimecourseMixin
from .peri_event_trajectory import PeriEventTrajectoryMixin

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MultivariateAxisAnalyzer(ConditionedTimecourseMixin, PeriEventTrajectoryMixin):
    def __init__(
        self,
        spikes_df,
        behavior_df,
        event_col='event_time',
        behavior_cols=None,
        bin_width_ms=10.0,
        smoothing_sigma_ms=30.0,
        external_fr_mat=None,
        external_start_time=None,
        external_clusters=None,
    ):
        self.behavior_cols = list(
            behavior_cols) if behavior_cols is not None else None
        self.event_times = behavior_df[event_col].to_numpy()
        self.behavior_df = behavior_df.reset_index(drop=True)

        spikes_df = spikes_df.sort_values('time')
        clusters = np.array(sorted(spikes_df['cluster'].unique()))
        self.clusters = clusters
        self.cluster_to_idx = {c: i for i, c in enumerate(clusters)}

        spike_codes = np.array([self.cluster_to_idx[c]
                               for c in spikes_df['cluster']])
        spike_times = spikes_df['time'].to_numpy(float)

        self.bin_width_ms = float(bin_width_ms)
        self.bin_width_s = self.bin_width_ms / 1000.0

        if external_fr_mat is not None:
            self.fr_mat = external_fr_mat
            self.start_time = float(
                external_start_time) if external_start_time is not None else float(spike_times.min())
            if external_clusters is not None:
                self.clusters = external_clusters
        else:
            self.fr_mat, self.start_time = build_continuous_fr(
                spike_times, spike_codes, len(clusters),
                self.bin_width_ms, smoothing_sigma_ms
            )

    def _events_to_bins(self, times):
        return events_to_bins(times, self.start_time, self.bin_width_s)

    def _preprocess_fr(self, X):
        X = np.sqrt(np.maximum(X, 0))
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        return (X - mu) / sd

    def build_trial_matrix(self, window_ms):
        event_bins = self._events_to_bins(self.event_times)
        windows = extract_event_windows(
            self.fr_mat, event_bins, window_ms, self.bin_width_ms
        )
        X = windows.mean(axis=1)
        X = self._preprocess_fr(X)

        if self.behavior_cols is None:
            raise ValueError('behavior_cols must be set to build Y.')

        Y = self.behavior_df[self.behavior_cols].to_numpy(float)
        return X, Y

    def compute_axes(
        self,
        window_ms,
        interaction_pairs=None,
        fit_backend='linear',
        linear_method='ridge',
        alpha=1.0,
        l1_ratio=0.5,
        rank=None,
        nonlinear_hidden=(64,),
        nonlinear_alpha=1e-4,
        random_state=0,
    ):
        """
        fit_backend:
            'linear'     -> multitask linear W (neurons x targets) or low-rank (neurons x rank)
            'nonlinear'  -> MLP multi-output regressor (returns readout + projection, axes=None)

        Returns
        -------
        axis_info dict
        """
        X, Y = self.build_trial_matrix(window_ms)

        target_names = list(self.behavior_cols)

        if interaction_pairs is not None and len(interaction_pairs) > 0:
            Y = build_interaction_terms(Y, interaction_pairs)
            for i, j in interaction_pairs:
                target_names.append(
                    f'{self.behavior_cols[i]}*{self.behavior_cols[j]}')

        if fit_backend == 'linear':
            if rank is None:
                W, model = fit_multitask_linear(
                    X, Y,
                    method=linear_method,
                    alpha=alpha,
                    l1_ratio=l1_ratio
                )
                projection = self.fr_mat @ W  # (T, n_targets)
                return {
                    'fit_kind': 'linear_axes',
                    'axes': W,
                    'projection': projection,
                    'readout': model,
                    'targets': target_names,
                    'clusters': self.clusters,
                    'fit_backend': 'linear',
                    'linear_method': linear_method,
                    'rank': None,
                }

            W_rank, model = reduced_rank_regression(
                X, Y, rank=rank, alpha=alpha)
            projection = self.fr_mat @ W_rank  # (T, rank)
            return {
                'fit_kind': 'linear_axes',
                'axes': W_rank,
                'projection': projection,
                'readout': model,
                'targets': target_names,
                'clusters': self.clusters,
                'fit_backend': 'linear',
                'linear_method': f'rrr_ridge(alpha={alpha})',
                'rank': int(rank),
            }

        if fit_backend == 'nonlinear':
            # scale features for MLP
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(X)

            mlp = MLPRegressor(
                hidden_layer_sizes=tuple(nonlinear_hidden),
                activation='relu',
                alpha=float(nonlinear_alpha),
                early_stopping=True,
                max_iter=2000,
                random_state=int(random_state),
            )
            mlp.fit(Xs, Y)

            # continuous projection across time
            fr_proc = self._preprocess_fr(self.fr_mat)
            fr_proc_s = scaler.transform(fr_proc)
            projection = mlp.predict(fr_proc_s)  # (T, n_targets_aug)

            # Provide a callable so callers can project arbitrary FR matrices
            def project_fn(fr_mat):
                fr_p = self._preprocess_fr(fr_mat)
                fr_s = scaler.transform(fr_p)
                return mlp.predict(fr_s)

            return {
                'fit_kind': 'nonlinear_readout',
                'axes': None,
                'projection': projection,
                'readout': mlp,
                'project_fn': project_fn,
                'scaler': scaler,
                'targets': target_names,
                'clusters': self.clusters,
                'fit_backend': 'nonlinear',
                'nonlinear_hidden': tuple(nonlinear_hidden),
            }

        raise ValueError(f'Unknown fit_backend: {fit_backend}')

    def orthogonalize(self, axes):
        return orthogonalize_axes(axes)

    def axis_angle(self, a, b):
        return axis_angle(a, b)
