import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StopAlignedSpikeWarp:
    """
    Landmark-based spike-time warping that aligns stop k -> stop k
    across segments using piecewise-linear time warps.
    """

    def __init__(self, new_seg_info, K=3):
        """
        Parameters
        ----------
        new_seg_info : pd.DataFrame
            Must contain:
              - new_segment
              - new_seg_start_time
              - new_seg_end_time
              - stop_k_time
              - stop_k_id_duration   for k = 1..K
        K : int
            Number of stops to align
        """
        self.new_seg_info = new_seg_info.copy()
        self.K = K
        self.template = self._build_template()

    # ------------------------------------------------------------------
    # Template construction
    # ------------------------------------------------------------------

    def _build_template(self):
        tpl = {}

        tpl['seg_start'] = 0.0
        tpl['seg_end'] = np.median(
            self.new_seg_info.new_seg_end_time
            - self.new_seg_info.new_seg_start_time
        )

        for k in range(1, self.K + 1):
            t_on = self.new_seg_info[f'stop_{k}_time'] - self.new_seg_info.new_seg_start_time
            d = self.new_seg_info[f'stop_{k}_id_duration']
            valid = t_on.notna() & d.notna()

            tpl[f'stop_{k}_on'] = np.median(t_on[valid])
            tpl[f'stop_{k}_off'] = np.median((t_on + d)[valid])

        return tpl

    # ------------------------------------------------------------------
    # Warp construction per segment
    # ------------------------------------------------------------------

    def _build_segment_knots(self, row):
        t_src = [row.new_seg_start_time]
        t_dst = [self.template['seg_start']]

        for k in range(1, self.K + 1):
            t_on = row.get(f'stop_{k}_time')
            d = row.get(f'stop_{k}_id_duration')

            if pd.notna(t_on) and pd.notna(d):
                t_src.extend([t_on, t_on + d])
                t_dst.extend([
                    self.template[f'stop_{k}_on'],
                    self.template[f'stop_{k}_off']
                ])

        t_src.append(row.new_seg_end_time)
        t_dst.append(self.template['seg_end'])

        t_src = np.asarray(t_src)
        t_dst = np.asarray(t_dst)

        order = np.argsort(t_src)
        return t_src[order], t_dst[order]


    def _make_warp_function(self, t_src, t_dst):
        return lambda t: np.interp(t, t_src, t_dst)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warp_spikes(self, spike_df):
        """
        Warp spike times for all segments.

        Parameters
        ----------
        spike_df : pd.DataFrame
            Columns:
              - spike_time
              - cluster
              - new_segment

        Returns
        -------
        warped_df : pd.DataFrame
            Same columns, with warped spike_time
        """
        warped_segments = []

        for seg_id, seg_df in spike_df.groupby('new_segment'):
            row = self.new_seg_info.loc[
                self.new_seg_info.new_segment == seg_id
            ].iloc[0]

            t_src, t_dst = self._build_segment_knots(row)
            f = self._make_warp_function(t_src, t_dst)

            seg_df = seg_df.copy()
            seg_df['spike_time'] = f(seg_df['spike_time'])
            warped_segments.append(seg_df)

        return pd.concat(warped_segments, ignore_index=True)


    def plot_qc_timewarp(self, seg_id, figsize=(6, 4)):
        """
        Plot the time-warp function for one segment.
        """
        row = self.new_seg_info.loc[
            self.new_seg_info.new_segment == seg_id
        ].iloc[0]

        t_src, t_dst = self._build_segment_knots(row)

        plt.figure(figsize=figsize)
        plt.plot(t_src, t_dst, '-o')
        plt.xlabel('Original time')
        plt.ylabel('Warped time')
        plt.title(f'Time warp: segment {seg_id}')
        plt.grid(True)
        plt.show()
        
        
    def plot_qc_raster(
        self,
        spike_df,
        warped_df,
        segments=None,
        max_neurons=10,
        use_relative_time=True,
        figsize=(12, 6)
    ):
        """
        Raster before/after alignment for QC.

        Parameters
        ----------
        spike_df : pd.DataFrame
            Original spikes
        warped_df : pd.DataFrame
            Warped spikes
        segments : list or None
            Subset of new_segment ids to plot
        max_neurons : int
            Max neurons per segment to plot
        use_relative_time : bool
            If True, subtract new_seg_start_time for original spikes
        """

        if segments is None:
            segments = spike_df.new_segment.unique()[:3]

        fig, axes = plt.subplots(
            nrows=len(segments),
            ncols=2,
            figsize=figsize,
            sharex='col',
            sharey='row'
        )

        if len(segments) == 1:
            axes = axes[None, :]

        for i, seg_id in enumerate(segments):
            row = self.new_seg_info.loc[
                self.new_seg_info.new_segment == seg_id
            ].iloc[0]

            orig = spike_df[spike_df.new_segment == seg_id]
            warped = warped_df[warped_df.new_segment == seg_id]

            clusters = orig.cluster.unique()[:max_neurons]

            for j, cl in enumerate(clusters):
                t_orig = orig.loc[orig.cluster == cl, 'spike_time']
                if use_relative_time:
                    t_orig = t_orig - row.new_seg_start_time

                axes[i, 0].plot(
                    t_orig,
                    np.full(len(t_orig), j),
                    '|',
                    color='k'
                )

                axes[i, 1].plot(
                    warped.loc[warped.cluster == cl, 'spike_time'],
                    np.full(
                        warped.loc[warped.cluster == cl].shape[0],
                        j
                    ),
                    '|',
                    color='k'
                )

            # Stop bands
            for k in range(1, self.K + 1):
                t_on = row.get(f'stop_{k}_time')
                d = row.get(f'stop_{k}_id_duration')

                if pd.notna(t_on) and pd.notna(d):
                    if use_relative_time:
                        t0 = t_on - row.new_seg_start_time
                        t1 = t0 + d
                    else:
                        t0 = t_on
                        t1 = t_on + d

                    axes[i, 0].axvspan(t0, t1, color='C1', alpha=0.2)
                    axes[i, 1].axvspan(
                        self.template[f'stop_{k}_on'],
                        self.template[f'stop_{k}_off'],
                        color='C1',
                        alpha=0.2
                    )

            axes[i, 0].set_ylabel(f'seg {seg_id}')
            axes[i, 0].set_title('Original (relative time)' if use_relative_time else 'Original')
            axes[i, 1].set_title('Warped (canonical time)')

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Warped time (s)')

        plt.tight_layout()
        plt.show()


    def propagate_template_stops(
        self,
        spike_df,
        seg_info_df,
        max_num_stops
    ):
        """
        Build warped spike and segment metadata, preserving NaNs
        for missing stops.

        Parameters
        ----------
        spike_df : pd.DataFrame
            Must contain:
            - spike_time (already warped back to absolute time)
            - new_segment
            - new_seg_start_time
        seg_info_df : pd.DataFrame
            Original segment info (pre-warp)
        max_num_stops : int

        Returns
        -------
        aligned_spike_trains_warped : pd.DataFrame
        new_seg_info_warped : pd.DataFrame
        """

        # -------------------------------
        # Segment-level warped metadata
        # -------------------------------
        new_seg_info_warped = seg_info_df[
            ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'num_stops']
        ].copy()
        
        aligned_spike_trains_warped = spike_df.copy()


        for k in range(1, max_num_stops + 1):
            exists = seg_info_df[f'stop_{k}_time'].notna()

            t, d, t_end = _apply_template_with_nan_mask(
                exists_mask=exists,
                template_on=self.template[f'stop_{k}_on'],
                template_off=self.template[f'stop_{k}_off'],
                seg_start_time=new_seg_info_warped['new_seg_start_time']
            )

            new_seg_info_warped[f'stop_{k}_time'] = t
            new_seg_info_warped[f'stop_{k}_id_duration'] = d
            new_seg_info_warped[f'stop_{k}_end_time'] = t_end

        # last valid stop time (robust)
        def _last_valid_stop(row):
            for k in range(max_num_stops, 0, -1):
                if pd.notna(row[f'stop_{k}_time']):
                    return row[f'stop_{k}_time']
            return np.nan

        new_seg_info_warped['last_stop_time'] = new_seg_info_warped.apply(
            _last_valid_stop, axis=1
        )

        # merge original stop existence
        stop_exists_cols = (
            ['new_segment'] +
            [f'stop_{k}_time' for k in range(1, max_num_stops + 1)]
        )

        aligned_spike_trains_warped = aligned_spike_trains_warped.merge(
            seg_info_df[stop_exists_cols],
            on='new_segment',
            how='left',
            suffixes=('', '_orig')
        )

        for k in range(1, max_num_stops + 1):
            exists = aligned_spike_trains_warped[f'stop_{k}_time_orig'].notna()

            t, d, t_end = _apply_template_with_nan_mask(
                exists_mask=exists,
                template_on=self.template[f'stop_{k}_on'],
                template_off=self.template[f'stop_{k}_off'],
                seg_start_time=aligned_spike_trains_warped['new_seg_start_time']
            )

            aligned_spike_trains_warped[f'stop_{k}_time'] = t
            aligned_spike_trains_warped[f'stop_{k}_id_duration'] = d
            aligned_spike_trains_warped[f'stop_{k}_end_time'] = t_end

        # cleanup
        aligned_spike_trains_warped.drop(
            columns=[c for c in aligned_spike_trains_warped.columns if c.endswith('_orig')],
            inplace=True
        )

        # attach last_stop_time
        aligned_spike_trains_warped.drop(columns=['last_stop_time'], inplace=True, errors='ignore')
        aligned_spike_trains_warped = aligned_spike_trains_warped.merge(
            new_seg_info_warped[['new_segment', 'last_stop_time']],
            on='new_segment',
            how='left'
        )

        return aligned_spike_trains_warped, new_seg_info_warped


def _apply_template_with_nan_mask(
    exists_mask,
    template_on,
    template_off,
    seg_start_time
):
    """
    Apply template stop times while preserving NaNs.

    Parameters
    ----------
    exists_mask : pd.Series[bool]
        True where the stop exists for that segment
    template_on : float
    template_off : float
    seg_start_time : pd.Series
        new_seg_start_time

    Returns
    -------
    stop_time, stop_duration, stop_end_time : pd.Series
    """
    stop_time = np.where(
        exists_mask,
        template_on + seg_start_time,
        np.nan
    )

    stop_duration = np.where(
        exists_mask,
        template_off - template_on,
        np.nan
    )

    stop_end_time = stop_time + stop_duration

    return stop_time, stop_duration, stop_end_time
