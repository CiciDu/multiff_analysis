import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.linalg as LA


class PeriEventTrajectoryMixin:
    """
    Mixin for peri-event population trajectory analysis in low-D spaces.
    Assumes aligned_proj has shape (n_trials, n_time, n_dims).
    """

    def plot_peri_event_trajectory(
        self,
        projection,
        event_times,
        condition_mask,
        window_ms,
        dims_3d=(0, 1, 2),
        dims_2d=(0, 1),
        single_trial_label_name=None,
        single_trial_labels=None,
        n_show=30,
        label_true=None,
        label_false=None,
        mean_title='Peri-event population trajectories',
        distance_title='Population separation with 95% CI',
        state_title='Population states with covariance ellipses',
        t_query=0.1,
        n_boot=1000,
        random_state=0,
    ):
        """
        End-to-end conditioned population geometry analysis.

        This function:
          1) plots condition-mean 3D trajectories
          2) plots single-trial 3D trajectories (optional)
          3) plots population separation over time with bootstrap CI
          4) plots state-space scatter with covariance ellipses at a selected time

        Parameters
        ----------
        aligned_proj : (n_trials, n_time, n_dims)
        time : (n_time,)
        condition_mask : (n_trials,) bool
            Primary condition used for averaging and distance.
        dims_3d : tuple[int]
            Dimensions used for 3D trajectory plots.
        dims_2d : tuple[int]
            Dimensions used for state scatter plot.
        single_trial_labels : (n_trials,) bool or None
            If provided, used for coloring single-trial trajectories.
            Defaults to condition_mask.
        n_show : int
            Number of single-trial trajectories to plot per condition.
        label_true, label_false : str
        t_query : float
            Time (s) at which to plot state-space geometry.
        n_boot : int
            Number of bootstrap samples for population distance.
        random_state : int
        """

        if label_true is None:
            if single_trial_label_name is not None:
                label_true = single_trial_label_name.replace('_', ' ').title()
            else:
                label_true = 'Condition = 1'

        if label_false is None:
            if single_trial_label_name is not None:
                label_false = f'Not {single_trial_label_name.replace("_", " ").title()}'
            else:
                label_false = 'Condition = 0'

        self.aligned_proj, self.time = self.align_continuous_to_events(
            projection=projection,
            event_times=event_times,
            window_ms=window_ms,
        )

        y = np.asarray(condition_mask, dtype=bool)

        if single_trial_labels is None:
            single_trial_labels = y

        # ----------------------------------
        # 1) Mean condition trajectories
        # ----------------------------------
        mean_true, mean_false = self.compute_condition_means(
            self.aligned_proj,
            condition_mask=y,
            dims=dims_3d,
        )

        self.plot_mean_trajectory_3d(
            mean_true,
            mean_false,
            labels=(label_true, label_false),
            title=mean_title,
        )

        # ----------------------------------
        # 2) Single-trial trajectories
        # ----------------------------------
        self.plot_peri_event_trajectories_3d(
            self.aligned_proj,
            labels=single_trial_labels,
            dims=dims_3d,
            n_show=n_show,
            labels_text=(label_false, label_true),
        )

        # ----------------------------------
        # 3) Population separation over time
        # ----------------------------------
        dist, ci_low, ci_high = self.population_distance_over_time(
            self.aligned_proj,
            condition_mask=y,
            dims=dims_3d,
            n_boot=n_boot,
            random_state=random_state,
        )

        self.plot_population_distance(
            self.time,
            dist,
            ci_low,
            ci_high,
            title=distance_title,
        )

        # ----------------------------------
        # 4) State geometry at a timepoint
        # ----------------------------------
        self.plot_state_scatter_with_covariance(
            self.aligned_proj,
            condition_mask=y,
            time=self.time,
            t_query=t_query,
            dims=dims_2d,
            title=state_title,
        )

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

            idx = np.random.choice(idx, size=min(
                n_show, len(idx)), replace=False)
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
            samp_false = rng.choice(
                idx_false, size=len(idx_false), replace=True)

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
