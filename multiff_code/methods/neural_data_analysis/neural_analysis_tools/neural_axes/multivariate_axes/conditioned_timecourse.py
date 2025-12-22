# multivariate_axis_analyzer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


import numpy as np
import matplotlib.pyplot as plt


class ConditionedTimecourseMixin:

    def plot_conditioned_timecourse(
        self,
        projection,
        event_times,
        condition_mask,
        window_ms,
        axis_indices=None,
        axis_names=None,
        label_true=None,
        label_false=None,
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

    def align_continuous_to_events(
        self,
        projection,
        event_times,
        window_ms,
    ):
        """
        Align a continuous time series (e.g. axis projection) to events.

        Parameters
        ----------
        projection : array, shape (T, n_axes)
            Continuous projection across time.
        event_times : array, shape (n_events,)
            Event times in seconds.
        window_ms : tuple (start_ms, end_ms)
            Time window around event.

        Returns
        -------
        aligned : array, shape (n_events, n_timepoints, n_axes)
        time : array, shape (n_timepoints,)
            Time relative to event (seconds).
        """
        event_bins = self._events_to_bins(np.asarray(event_times))

        start_ms, end_ms = window_ms
        so = int(start_ms / self.bin_width_ms)
        eo = int(end_ms / self.bin_width_ms)

        offsets = np.arange(so, eo)
        idx = event_bins[:, None] + offsets[None, :]
        idx = np.clip(idx, 0, projection.shape[0] - 1)

        aligned = projection[idx]
        time = offsets * self.bin_width_s

        return aligned, time

    def conditioned_axis_stats(
        self,
        aligned_proj,
        condition_mask,
        axis_idx=0,
    ):
        """
        Compute mean and SEM of an aligned axis projection
        conditioned on a boolean mask.

        Parameters
        ----------
        aligned_proj : array, shape (n_events, n_timepoints, n_axes)
        condition_mask : bool array, shape (n_events,)
        axis_idx : int

        Returns
        -------
        mean_true, sem_true, mean_false, sem_false
        """
        aligned_true = aligned_proj[condition_mask]
        aligned_false = aligned_proj[~condition_mask]

        mean_true = aligned_true[:, :, axis_idx].mean(axis=0)
        sem_true = aligned_true[:, :, axis_idx].std(
            axis=0) / np.sqrt(aligned_true.shape[0])

        mean_false = aligned_false[:, :, axis_idx].mean(axis=0)
        sem_false = aligned_false[:, :, axis_idx].std(
            axis=0) / np.sqrt(aligned_false.shape[0])

        return mean_true, sem_true, mean_false, sem_false

    def plot_conditioned_axis(
        self,
        time,
        mean_true,
        sem_true,
        mean_false,
        sem_false,
        label_true='Condition = 1',
        label_false='Condition = 0',
        axis_label='Axis projection',
        title=None,
    ):
        """
        Convenience plotting helper.
        """
        plt.figure(figsize=(6, 4))

        plt.fill_between(time, mean_true - sem_true,
                         mean_true + sem_true, alpha=0.3)
        plt.plot(time, mean_true, label=label_true)

        plt.fill_between(time, mean_false - sem_false,
                         mean_false + sem_false, alpha=0.3)
        plt.plot(time, mean_false, label=label_false)

        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Time from event (s)')
        plt.ylabel(axis_label)
        plt.legend()

        if title is not None:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _fdr_bh(pvals, alpha=0.05):
        """
        Benjamini-Hochberg FDR control.

        Parameters
        ----------
        pvals : array-like, shape (m,)
        alpha : float

        Returns
        -------
        qvals : np.ndarray, shape (m,)
        rejected : np.ndarray(bool), shape (m,)
        """
        pvals = np.asarray(pvals, dtype=float)
        m = pvals.size

        order = np.argsort(pvals)
        ranked = pvals[order]
        ranks = np.arange(1, m + 1)

        # BH critical values
        crit = (ranks / m) * float(alpha)
        passed = ranked <= crit

        rejected = np.zeros(m, dtype=bool)
        if np.any(passed):
            kmax = np.max(np.where(passed)[0])
            rejected[order[:kmax + 1]] = True

        # q-values (monotone)
        q = ranked * (m / ranks)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)

        qvals = np.empty(m, dtype=float)
        qvals[order] = q

        return qvals, rejected

    @staticmethod
    def _cohens_d_timecourse(a, b, eps=1e-12):
        """
        Cohen's d across timepoints for two groups.

        a, b: arrays shape (n_trials, n_time)
        Returns: d shape (n_time,)
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        na = a.shape[0]
        nb = b.shape[0]

        ma = a.mean(axis=0)
        mb = b.mean(axis=0)

        va = a.var(axis=0, ddof=1)
        vb = b.var(axis=0, ddof=1)

        pooled = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
        d = (ma - mb) / (np.sqrt(pooled) + eps)

        return d

    def summarize_conditioned_timecourses(
        self,
        aligned_proj,
        time,
        condition_mask,
        axis_indices=None,
        axis_names=None,
    ):
        """
        Summarize mean/SEM/diff/effect-size for multiple axes (no significance yet).

        Parameters
        ----------
        aligned_proj : (n_events, n_time, n_axes)
        time : (n_time,)
        condition_mask : (n_events,) bool
        axis_indices : list[int] or None
            If None, uses all axes.
        axis_names : list[str] or None
            Optional display names for axes (same length as axis_indices).

        Returns
        -------
        summary_df : pd.DataFrame (tidy)
            Columns:
              axis_idx, axis_name, time_s,
              n_true, n_false,
              mean_true, sem_true,
              mean_false, sem_false,
              diff_true_minus_false,
              cohens_d
        """
        aligned_proj = np.asarray(aligned_proj)
        time = np.asarray(time, dtype=float)
        y = np.asarray(condition_mask, dtype=bool)

        if axis_indices is None:
            axis_indices = list(range(aligned_proj.shape[2]))
        axis_indices = list(axis_indices)

        if axis_names is None:
            axis_names = [f'axis_{i}' for i in axis_indices]

        aligned_true = aligned_proj[y]      # (n_true, n_time, n_axes)
        aligned_false = aligned_proj[~y]    # (n_false, n_time, n_axes)

        n_true = aligned_true.shape[0]
        n_false = aligned_false.shape[0]

        rows = []
        for ax_i, ax_name in zip(axis_indices, axis_names):
            a = aligned_true[:, :, ax_i]
            b = aligned_false[:, :, ax_i]

            mean_true = a.mean(axis=0)
            sem_true = a.std(axis=0, ddof=1) / np.sqrt(max(n_true, 1))

            mean_false = b.mean(axis=0)
            sem_false = b.std(axis=0, ddof=1) / np.sqrt(max(n_false, 1))

            diff = mean_true - mean_false
            d = self._cohens_d_timecourse(a, b)

            rows.append(pd.DataFrame({
                'axis_idx': ax_i,
                'axis_name': ax_name,
                'time_s': time,
                'n_true': n_true,
                'n_false': n_false,
                'mean_true': mean_true,
                'sem_true': sem_true,
                'mean_false': mean_false,
                'sem_false': sem_false,
                'diff_true_minus_false': diff,
                'cohens_d': d,
            }))

        summary_df = pd.concat(rows, ignore_index=True)
        return summary_df

    def permutation_test_conditioned_timecourses(
        self,
        aligned_proj,
        time,
        condition_mask,
        axis_indices=None,
        axis_names=None,
        n_perm=2000,
        two_sided=True,
        random_state=0,
        fdr_alpha=0.05,
        fdr_scope='within_axis',
    ):
        """
        Time-resolved permutation test of (mean_true - mean_false) for each axis.

        Parameters
        ----------
        aligned_proj : (n_events, n_time, n_axes)
        time : (n_time,)
        condition_mask : (n_events,) bool
        axis_indices : list[int] or None
        axis_names : list[str] or None
        n_perm : int
        two_sided : bool
        random_state : int
        fdr_alpha : float
        fdr_scope : {'within_axis', 'global'}
            - within_axis: BH-FDR separately for each axis across time
            - global: BH-FDR across all (axis,time) tests

        Returns
        -------
        stats_df : pd.DataFrame
            summary_df columns +
              p_perm, q_fdr, reject_fdr
        """
        rng = np.random.default_rng(int(random_state))

        aligned_proj = np.asarray(aligned_proj, dtype=float)
        time = np.asarray(time, dtype=float)
        y = np.asarray(condition_mask, dtype=bool)

        if axis_indices is None:
            axis_indices = list(range(aligned_proj.shape[2]))
        axis_indices = list(axis_indices)

        if axis_names is None:
            axis_names = [f'axis_{i}' for i in axis_indices]

        # Pre-split
        n_events, n_time, _ = aligned_proj.shape
        n_true = int(y.sum())
        n_false = int((~y).sum())

        if n_true < 2 or n_false < 2:
            raise ValueError(
                f'Need at least 2 trials per group; got n_true={n_true}, n_false={n_false}.')

        # Observed diffs for each axis: shape (n_time,)
        obs_diffs = {}
        for ax_i in axis_indices:
            a = aligned_proj[y, :, ax_i]
            b = aligned_proj[~y, :, ax_i]
            obs_diffs[ax_i] = a.mean(axis=0) - b.mean(axis=0)

        # Permutation: shuffle labels at trial-level
        # We compute perm diffs per axis in a vectorized-ish way: loop over perms, but compute all axes per perm.
        pvals_by_axis = {}
        for ax_i in axis_indices:
            obs = obs_diffs[ax_i]
            extreme = np.zeros(n_time, dtype=int)

            x = aligned_proj[:, :, ax_i]  # (n_events, n_time)

            for _ in range(int(n_perm)):
                perm = rng.permutation(n_events)
                y_perm = y[perm]

                a = x[y_perm]
                b = x[~y_perm]
                perm_diff = a.mean(axis=0) - b.mean(axis=0)

                if two_sided:
                    extreme += (np.abs(perm_diff) >= np.abs(obs))
                else:
                    extreme += (perm_diff >= obs)

            # add 1 for valid p-value even if 0 extremes
            p = (extreme + 1.0) / (float(n_perm) + 1.0)
            pvals_by_axis[ax_i] = p

        # Base summary (means, sems, cohens d)
        summary_df = self.summarize_conditioned_timecourses(
            aligned_proj=aligned_proj,
            time=time,
            condition_mask=y,
            axis_indices=axis_indices,
            axis_names=axis_names,
        )

        # Attach p-values
        # (summary_df is tidy; just map by axis_idx and time order)
        stats_df = summary_df.copy()
        stats_df['p_perm'] = np.nan

        # Fill p_perm per axis
        for ax_i in axis_indices:
            mask = stats_df['axis_idx'].to_numpy() == ax_i
            stats_df.loc[mask, 'p_perm'] = pvals_by_axis[ax_i]

        # FDR correction
        if fdr_scope == 'within_axis':
            stats_df['q_fdr'] = np.nan
            stats_df['reject_fdr'] = False

            for ax_i in axis_indices:
                mask = stats_df['axis_idx'].to_numpy() == ax_i
                p = stats_df.loc[mask, 'p_perm'].to_numpy()
                q, rej = self._fdr_bh(p, alpha=float(fdr_alpha))
                stats_df.loc[mask, 'q_fdr'] = q
                stats_df.loc[mask, 'reject_fdr'] = rej

        elif fdr_scope == 'global':
            p = stats_df['p_perm'].to_numpy()
            q, rej = self._fdr_bh(p, alpha=float(fdr_alpha))
            stats_df['q_fdr'] = q
            stats_df['reject_fdr'] = rej

        else:
            raise ValueError(f'Unknown fdr_scope: {fdr_scope}')

        return stats_df

    def plot_conditioned_axes_panel(
        self,
        stats_df,
        axis_order=None,
        label_true='Condition = 1',
        label_false='Condition = 0',
        title=None,
        show_sem=True,
        show_significance=True,
        sig_field='reject_fdr',
        sig_marker_y='top',
        figsize=(7.0, 2.2),
    ):
        """
        Publication-style panel: one row per axis.

        Parameters
        ----------
        stats_df : pd.DataFrame from permutation_test_conditioned_timecourses() or summarize_conditioned_timecourses()
        axis_order : list[str] or list[int] or None
            Either axis_name list or axis_idx list. If None, uses unique axis_name in appearance order.
        sig_field : str
            Column to use as significance indicator; default uses FDR rejection.
        sig_marker_y : {'top', 'zero'}
            Where to draw significance markers.
        """
        if axis_order is None:
            axis_names = stats_df['axis_name'].drop_duplicates().to_list()
        else:
            # support passing axis_idx or axis_name
            if isinstance(axis_order[0], (int, np.integer)):
                axis_names = [
                    stats_df.loc[stats_df['axis_idx']
                                 == int(ax), 'axis_name'].iloc[0]
                    for ax in axis_order
                ]
            else:
                axis_names = list(axis_order)

        n_axes = len(axis_names)

        fig, axes = plt.subplots(n_axes, 1, figsize=(
            float(figsize[0]), float(figsize[1]) * n_axes), sharex=True)
        if n_axes == 1:
            axes = [axes]

        for ax_plot, ax_name in zip(axes, axis_names):
            d = stats_df[stats_df['axis_name']
                         == ax_name].sort_values('time_s')

            t = d['time_s'].to_numpy()
            mt = d['mean_true'].to_numpy()
            mf = d['mean_false'].to_numpy()

            if show_sem and ('sem_true' in d.columns) and ('sem_false' in d.columns):
                st = d['sem_true'].to_numpy()
                sf = d['sem_false'].to_numpy()
                ax_plot.fill_between(t, mt - st, mt + st, alpha=0.25)
                ax_plot.fill_between(t, mf - sf, mf + sf, alpha=0.25)

            ax_plot.plot(t, mt, label=label_true, linewidth=2.0)
            ax_plot.plot(t, mf, label=label_false, linewidth=2.0)

            ax_plot.axvline(0, color='k', linestyle='--', linewidth=1.0)
            ax_plot.set_ylabel(ax_name)

            if show_significance and (sig_field in d.columns):
                sig = d[sig_field].to_numpy(dtype=bool)
                if np.any(sig):
                    if sig_marker_y == 'top':
                        y0, y1 = ax_plot.get_ylim()
                        y_sig = y1 - 0.05 * (y1 - y0)
                    elif sig_marker_y == 'zero':
                        y_sig = 0.0
                    else:
                        y_sig = ax_plot.get_ylim()[1]

                    ax_plot.plot(t[sig], np.full(sig.sum(), y_sig),
                                 linestyle='None', marker='.', markersize=4)

        axes[-1].set_xlabel('Time from event (s)')
        if title is not None:
            fig.suptitle(title, y=0.98)

        # One legend on top axis
        axes[0].legend(frameon=False, loc='upper right')
        plt.tight_layout()
        plt.show()
