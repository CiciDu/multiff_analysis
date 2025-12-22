import numpy as np
import pandas as pd

from cebra import CEBRA

from neural_data_analysis.neural_analysis_tools.neural_axes.axis_utils import (
    build_continuous_fr,
    events_to_bins,
    extract_event_windows,
)


class CEBRAAnalyzer:
    """
    Apply CEBRA to continuous neural data and extract
    peri-event embeddings.

    This class mirrors MultivariateAxisAnalyzer structurally,
    but returns nonlinear embeddings instead of axes.
    """

    def __init__(
        self,
        spikes_df,
        behavior_df,
        event_col='time',
        behavior_cols=None,
        bin_width_ms=10.0,
        smoothing_sigma_ms=30.0,
        external_fr_mat=None,
        external_start_time=None,
        external_clusters=None,
    ):
        self.behavior_df = behavior_df.reset_index(drop=True)
        self.behavior_cols = behavior_cols
        self.event_times = behavior_df[event_col].to_numpy()

        self.bin_width_ms = float(bin_width_ms)
        self.bin_width_s = self.bin_width_ms / 1000.0

        # Build or load firing-rate matrix
        if external_fr_mat is not None:
            self.fr_mat = external_fr_mat
            self.start_time = float(
                external_start_time
                if external_start_time is not None
                else spikes_df['time'].min()
            )
        else:
            spikes_df = spikes_df.sort_values('time')
            clusters = np.sort(spikes_df['cluster'].unique())
            cluster_to_idx = {c: i for i, c in enumerate(clusters)}

            spike_codes = np.array(
                [cluster_to_idx[c] for c in spikes_df['cluster']]
            )
            spike_times = spikes_df['time'].to_numpy(float)

            self.fr_mat, self.start_time = build_continuous_fr(
                spike_times,
                spike_codes,
                len(clusters),
                bin_width_ms,
                smoothing_sigma_ms,
            )

    # -----------------------------
    def _events_to_bins(self, times):
        return events_to_bins(times, self.start_time, self.bin_width_s)

    def _preprocess_fr(self, X):
        """
        CEBRA works best with roughly standardized inputs.
        """
        X = np.sqrt(np.maximum(X, 0))
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        return (X - mu) / sd

    # -----------------------------
    def _fit_cebra_with_early_stopping(
        self,
        model,
        X,
        y=None,
        max_iterations=5000,
        chunk_iterations=500,
        slope_window=500,
        slope_threshold=1e-4,
        min_iterations=1000,
    ):
        """
        Fit CEBRA with early stopping based on loss slope.
        """
        total_iters = 0

        while total_iters < max_iterations:
            iters = min(chunk_iterations, max_iterations - total_iters)
            model.max_iterations = iters

            if y is None:
                model.fit(X)
            else:
                model.fit(X, y)

            total_iters += iters

            # Loss may not be populated early
            if not hasattr(model, 'loss_'):
                continue
            if len(model.loss_) < min_iterations:
                continue
            if len(model.loss_) < slope_window:
                continue

            recent = np.asarray(model.loss_)[-slope_window:]
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]

            if abs(slope) < slope_threshold:
                if model.verbose:
                    print(
                        f'Early stopping at {total_iters} iterations '
                        f'(loss slope = {slope:.2e})'
                    )
                break

        self.n_iterations_ = total_iters
        return model

    # -----------------------------
    def fit_cebra(
        self,
        output_dim=3,
        conditional='behavior',
        model_architecture='offset10-model',
        batch_size=512,
        max_iterations=5000,
        learning_rate=3e-4,
        temperature=1.0,
        device='cpu',
        verbose=True,
    ):
        """
        Fit CEBRA on the full continuous neural time series.

        Returns
        -------
        embedding : (T, output_dim)
        model : trained CEBRA object
        """
        X = self._preprocess_fr(self.fr_mat)

        if conditional == 'behavior':
            if self.behavior_cols is None:
                raise ValueError(
                    'behavior_cols must be provided for conditional="behavior"'
                )

            event_bins = self._events_to_bins(self.event_times)
            y = np.zeros((len(X), len(self.behavior_cols)))

            for i, b in enumerate(event_bins):
                if 0 <= b < len(y):
                    y[b] = self.behavior_df[self.behavior_cols].iloc[i].to_numpy()

            model = CEBRA(
                model_architecture=model_architecture,
                output_dimension=output_dim,
                batch_size=batch_size,
                learning_rate=learning_rate,
                temperature=temperature,
                device=device,
                verbose=verbose,
            )

            model = self._fit_cebra_with_early_stopping(
                model,
                X,
                y=y,
                max_iterations=max_iterations,
            )

        elif conditional == 'time':
            model = CEBRA(
                model_architecture=model_architecture,
                conditional='time',
                output_dimension=output_dim,
                batch_size=batch_size,
                learning_rate=learning_rate,
                temperature=temperature,
                device=device,
                verbose=verbose,
            )

            model = self._fit_cebra_with_early_stopping(
                model,
                X,
                y=None,
                max_iterations=max_iterations,
            )

        else:
            raise ValueError(f'Unknown conditional type: {conditional}')

        embedding = model.transform(X)

        self.embedding_ = embedding
        self.model_ = model

        return embedding, model

    # -----------------------------
    def extract_peri_event_embeddings(
        self,
        window_ms=(-300, 500),
    ):
        """
        Extract peri-event embedding segments.
        """
        if not hasattr(self, 'embedding_'):
            raise RuntimeError(
                'Call fit_cebra() before extracting embeddings.')

        event_bins = self._events_to_bins(self.event_times)

        so = int(window_ms[0] / self.bin_width_ms)
        eo = int(window_ms[1] / self.bin_width_ms)
        offsets = np.arange(so, eo)

        idx = event_bins[:, None] + offsets[None, :]
        idx = np.clip(idx, 0, self.embedding_.shape[0] - 1)

        aligned_embed = self.embedding_[idx]
        time = offsets * self.bin_width_s

        return aligned_embed, time

    # -----------------------------
    def permutation_test_peri_event_distance(
        self,
        aligned_embed,
        labels,
        n_perm=1000,
        seed=0,
    ):
        """
        Permutation test on peri-event CEBRA embeddings.
        """
        rng = np.random.default_rng(seed)

        labels = np.asarray(labels).astype(bool)
        if labels.ndim != 1:
            raise ValueError('labels must be 1D')
        if aligned_embed.shape[0] != labels.shape[0]:
            raise ValueError(
                'aligned_embed and labels must match in first dimension'
            )

        mean_0 = aligned_embed[~labels].mean(axis=0)
        mean_1 = aligned_embed[labels].mean(axis=0)
        obs_dist = np.linalg.norm(mean_1 - mean_0, axis=1)

        n_time = aligned_embed.shape[1]
        perm_dist = np.zeros((n_perm, n_time))

        for p in range(n_perm):
            perm_labels = rng.permutation(labels)
            m0 = aligned_embed[~perm_labels].mean(axis=0)
            m1 = aligned_embed[perm_labels].mean(axis=0)
            perm_dist[p] = np.linalg.norm(m1 - m0, axis=1)

        p_values = (perm_dist >= obs_dist[None, :]).mean(axis=0)
        max_null = perm_dist.max(axis=1)
        p_global = (max_null >= obs_dist.max()).mean()

        return {
            'obs_dist': obs_dist,
            'perm_dist': perm_dist,
            'p_values': p_values,
            'p_global': p_global,
        }

    # -----------------------------
    def save_embedding(self, path):
        """
        Save CEBRA embedding and training metadata.

        Parameters
        ----------
        path : str
            Path prefix (without extension).
        """
        if not hasattr(self, 'embedding_'):
            raise RuntimeError('No embedding to save. Run fit_cebra() first.')

        np.savez(
            path,
            embedding=self.embedding_,
            loss=getattr(self.model_, 'loss_', None),
            n_iterations=getattr(self, 'n_iterations_', None),
            behavior_cols=np.array(self.behavior_cols, dtype=object)
            if self.behavior_cols is not None
            else None,
            bin_width_ms=self.bin_width_ms,
            start_time=self.start_time,
        )
    # -----------------------------

    def load_embedding(self, path):
        """
        Load a previously saved CEBRA embedding.

        Parameters
        ----------
        path : str
            Path prefix used in save_embedding().
        """
        data = np.load(path, allow_pickle=True)

        self.embedding_ = data['embedding']
        self.n_iterations_ = data.get('n_iterations', None)

        # Restore metadata
        self.behavior_cols = (
            list(data['behavior_cols'])
            if data['behavior_cols'] is not None
            else None
        )
        self.bin_width_ms = float(data['bin_width_ms'])
        self.bin_width_s = self.bin_width_ms / 1000.0
        self.start_time = float(data['start_time'])

        # Optional: store loss for diagnostics
        self.loss_ = data.get('loss', None)
