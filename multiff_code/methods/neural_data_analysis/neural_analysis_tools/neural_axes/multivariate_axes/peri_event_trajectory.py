import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.linalg as LA


class PeriEventTrajectoryMixin:
    """
    Mixin for peri-event population trajectory analysis in low-D spaces.
    Assumes aligned_proj has shape (n_trials, n_time, n_dims).
    """
class PeriEventTrajectoryMixin:
    ...
    # =====================================================
    # High-level conditioned peri-event axis timecourse API
    # =====================================================

    def conditioned_peri_event_axes_panel(
        self,
        projection,
        event_times,
        condition_mask,
        window_ms,
        axis_indices=None,
        axis_names=None,
        label_true='Condition = 1',
        label_false='Condition = 0',
        title=None,
        n_perm=2000,
        random_state=0,
        fdr_alpha=0.05,
        fdr_scope='within_axis',
        show_sem=True,
        show_significance=True,
        sig_field='reject_fdr',
        sig_marker_y='top',
        return_stats=True,
    ):
        """
        End-to-end conditioned peri-event axis analysis:
        alignment → permutation test → plotting.

        Parameters
        ----------
        projection : (T, n_axes) array
            Continuous axis projection.
        event_times : (n_events,) array
            Event times in seconds.
        condition_mask : (n_events,) bool
            Trial-wise condition labels.
        window_ms : (start_ms, end_ms)
        axis_indices : list[int] or None
        axis_names : list[str] or None
        label_true, label_false : str
            Legend labels.
        title : str or None
        n_perm : int
        random_state : int
        fdr_alpha : float
        fdr_scope : {'within_axis', 'global'}
        show_sem : bool
        show_significance : bool
        sig_field : str
        sig_marker_y : {'top', 'zero'}
        return_stats : bool

        Returns
        -------
        stats_df : pd.DataFrame or None
        """
        # -----------------------
        # 1) Align to events
        # -----------------------
        self.aligned_proj, self.time = self.align_continuous_to_events(
            projection=projection,
            event_times=event_times,
            window_ms=window_ms,
        )

        # -----------------------
        # 2) Permutation stats
        # -----------------------
        stats_df = self.permutation_test_conditioned_timecourses(
            aligned_proj=self.aligned_proj,
            time=self.time,
            condition_mask=condition_mask,
            axis_indices=axis_indices,
            axis_names=axis_names,
            n_perm=n_perm,
            random_state=random_state,
            fdr_alpha=fdr_alpha,
            fdr_scope=fdr_scope,
        )

        # -----------------------
        # 3) Plot
        # -----------------------
        self.plot_conditioned_axes_panel(
            stats_df,
            label_true=label_true,
            label_false=label_false,
            title=title,
            show_sem=show_sem,
            show_significance=show_significance,
            sig_field=sig_field,
            sig_marker_y=sig_marker_y,
        )

        if return_stats:
            return stats_df
        return None

        

    @staticmethod
    def compute_condition_means(
        aligned_proj,
        condition_mask,
        dims=None,
    ):
        """
        Compute condition-averaged trajectories.

        Returns
        -------
        mean_true : (n_time, n_dims)
        mean_false : (n_time, n_dims)
        """
        Z = aligned_proj if dims is None else aligned_proj[:, :, dims]
        y = np.asarray(condition_mask, dtype=bool)

        mean_true = Z[y].mean(axis=0)
        mean_false = Z[~y].mean(axis=0)

        return mean_true, mean_false

    def plot_mean_trajectory_3d(
        self,
        mean_true,
        mean_false,
        labels=('Condition = 1', 'Condition = 0'),
        dims_labels=('Dim 1', 'Dim 2', 'Dim 3'),
        title=None,
        markers=('^', 'o'),
    ):
        """
        3D plot of condition-averaged trajectories.
        """
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(mean_false[:, 0], mean_false[:, 1], mean_false[:, 2],
                label=labels[1])
        ax.plot(mean_true[:, 0], mean_true[:, 1], mean_true[:, 2],
                label=labels[0])

        ax.scatter(mean_false[0, 0], mean_false[0, 1], mean_false[0, 2],
                   marker=markers[1], s=40)
        ax.scatter(mean_true[0, 0], mean_true[0, 1], mean_true[0, 2],
                   marker=markers[0], s=40)

        ax.set_xlabel(dims_labels[0])
        ax.set_ylabel(dims_labels[1])
        ax.set_zlabel(dims_labels[2])
        ax.legend()

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    # =========================
    # Single-trial trajectory plotting
    # =========================

    def plot_peri_event_trajectories_3d(
        self,
        aligned_proj,
        labels,
        dims=(0, 1, 2),
        n_show=30,
        colors=('tab:blue', 'tab:orange'),
        labels_text=('Condition 0', 'Condition 1'),
        title=None,
        alpha=0.25,
    ):
        """
        Plot individual peri-event trajectories in 3D (subsampled).
        """
        Z = aligned_proj[:, :, dims]
        y = np.asarray(labels, dtype=bool)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        for cond, color, label in zip([False, True], colors, labels_text):
            idx = np.where(y == cond)[0]
            if len(idx) == 0:
                continue

            idx = np.random.choice(idx, size=min(n_show, len(idx)), replace=False)
            for i in idx:
                ax.plot(Z[i, :, 0], Z[i, :, 1], Z[i, :, 2],
                        color=color, alpha=alpha)

        ax.set_xlabel(f'Dim {dims[0] + 1}')
        ax.set_ylabel(f'Dim {dims[1] + 1}')
        ax.set_zlabel(f'Dim {dims[2] + 1}')

        if title is not None:
            ax.set_title(title)

        ax.legend(labels_text)
        plt.tight_layout()
        plt.show()

    # =========================
    # Population separation over time
    # =========================

    @staticmethod
    def population_distance_over_time(
        aligned_proj,
        condition_mask,
        dims=None,
        n_boot=1000,
        random_state=0,
    ):
        """
        Compute population separation (Euclidean distance) over time
        with bootstrap confidence intervals.

        Returns
        -------
        dist : (n_time,)
        ci_low : (n_time,)
        ci_high : (n_time,)
        """
        rng = np.random.default_rng(int(random_state))

        Z = aligned_proj if dims is None else aligned_proj[:, :, dims]
        y = np.asarray(condition_mask, dtype=bool)

        mean_true = Z[y].mean(axis=0)
        mean_false = Z[~y].mean(axis=0)

        dist = np.linalg.norm(mean_true - mean_false, axis=1)

        idx_true = np.where(y)[0]
        idx_false = np.where(~y)[0]

        boot = np.zeros((n_boot, Z.shape[1]))

        for b in range(n_boot):
            samp_true = rng.choice(idx_true, size=len(idx_true), replace=True)
            samp_false = rng.choice(idx_false, size=len(idx_false), replace=True)

            mt = Z[samp_true].mean(axis=0)
            mf = Z[samp_false].mean(axis=0)

            boot[b] = np.linalg.norm(mt - mf, axis=1)

        ci_low = np.percentile(boot, 2.5, axis=0)
        ci_high = np.percentile(boot, 97.5, axis=0)

        return dist, ci_low, ci_high

    def plot_population_distance(
        self,
        time,
        dist,
        ci_low,
        ci_high,
        title=None,
    ):
        """
        Plot population distance with bootstrap CI.
        """
        plt.figure(figsize=(6, 4))
        plt.fill_between(time, ci_low, ci_high, alpha=0.3)
        plt.plot(time, dist, color='k')
        plt.axvline(0, linestyle='--', color='k')
        plt.xlabel('Time from event (s)')
        plt.ylabel('Distance between condition means')

        if title is not None:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    # =========================
    # State geometry at a timepoint
    # =========================

    @staticmethod
    def _plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
        """
        Internal helper for covariance ellipse.
        """
        vals, vecs = LA.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)

        ell = Ellipse(xy=mean, width=width, height=height,
                      angle=theta, **kwargs)
        ax.add_patch(ell)

    def plot_state_scatter_with_covariance(
        self,
        aligned_proj,
        condition_mask,
        time,
        t_query,
        dims=(0, 1),
        colors=('tab:blue', 'tab:orange'),
        labels=('Condition 0', 'Condition 1'),
        n_std=2.0,
        title=None,
    ):
        """
        Scatter population states at a specific time with covariance ellipses.
        """
        y = np.asarray(condition_mask, dtype=bool)
        Z = aligned_proj[:, :, dims]

        t0 = np.argmin(np.abs(time - float(t_query)))

        X_false = Z[~y, t0]
        X_true = Z[y, t0]

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(X_false[:, 0], X_false[:, 1], alpha=0.3, color=colors[0])
        ax.scatter(X_true[:, 0], X_true[:, 1], alpha=0.3, color=colors[1])

        self._plot_cov_ellipse(
            X_false.mean(0), np.cov(X_false.T), ax,
            n_std=n_std, edgecolor=colors[0], fill=False
        )
        self._plot_cov_ellipse(
            X_true.mean(0), np.cov(X_true.T), ax,
            n_std=n_std, edgecolor=colors[1], fill=False
        )

        ax.set_xlabel(f'Dim {dims[0] + 1}')
        ax.set_ylabel(f'Dim {dims[1] + 1}')
        ax.legend(labels)

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

