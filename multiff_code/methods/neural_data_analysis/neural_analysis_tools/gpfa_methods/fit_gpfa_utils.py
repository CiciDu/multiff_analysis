import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantities as pq
import neo


def make_spike_segs_df(spikes_df, new_seg_info):
    # Create list of spike segments by splitting spikes_df based on time windows in new_seg_info
    # Each segment contains spikes between last_vis_time and ff_caught_time for a given trial
    spike_segments = []

    # drop segments with 0 duration
    new_seg_info_sub = new_seg_info[new_seg_info['new_seg_duration'] > 0]

    for index, row in new_seg_info_sub.iterrows():
        mask = spikes_df.time.between(
            row['new_seg_start_time'], row['new_seg_end_time'])
        spikes_sub = spikes_df[mask].copy()

        cols = ['new_segment', 'new_seg_start_time',
                'new_seg_end_time', 'new_seg_duration']
        spikes_sub[cols] = row[cols].values

        spike_segments.append(spikes_sub)

    spike_segs_df = pd.concat(spike_segments, ignore_index=True)

    spike_segs_df['t_duration'] = spike_segs_df['new_seg_end_time'] - \
        spike_segs_df['new_seg_start_time']

    return spike_segs_df


def turn_spike_segs_df_into_spiketrains(
    spike_segs_df,
    new_segments,
    common_t_stop,
    align_at_beginning=False
):
    """
    Convert a spike segments DataFrame into Neo SpikeTrain objects per segment and cluster.

    Parameters:
    - spike_segs_df (pd.DataFrame): Must include columns ['cluster', 'time', 'new_segment',
      'new_seg_start_time', 'new_seg_duration'].
    - new_segments (iterable): Ordered list of segment identifiers to convert.
    - common_t_stop (float): Duration (in seconds) to which all SpikeTrains should be aligned.
    - align_at_beginning (bool): If True, aligns all spike times to the start of segment.
      If False, applies padding at the beginning to align to the end.

    Returns:
    - spiketrains (list of list of neo.SpikeTrain): Outer list over segments, inner list over clusters.
    - processed_segments (np.ndarray): List of segment IDs corresponding to spiketrains.
    """
    import numpy as np
    import neo
    import quantities as pq
    import warnings

    spiketrains = []
    processed_segments = []

    # Sorted list of clusters ensures consistent ordering
    all_clusters = np.sort(spike_segs_df['cluster'].unique())

    num_padding_applied_segs = 0

    for seg in new_segments:
        seg_df = spike_segs_df[spike_segs_df['new_segment'] == seg]

        seg_spiketrain = []

        if len(seg_df) == 0:
            # No spikes for this segment; insert empty SpikeTrains per cluster
            empty_spiketrain = neo.SpikeTrain(
                times=np.array([]) * pq.s,
                t_start=0 * pq.s,
                t_stop=common_t_stop * pq.s
            )
            for _ in all_clusters:
                seg_spiketrain.append(empty_spiketrain)

            spiketrains.append(seg_spiketrain)
            processed_segments.append(seg)
            continue

        seg_start_time = seg_df.new_seg_start_time.iloc[0]
        seg_duration = seg_df.new_seg_duration.iloc[0]
        padding_at_beginning = 0 if align_at_beginning else (common_t_stop - seg_duration)

        if padding_at_beginning < 0:
            raise ValueError(
                f"Segment '{seg}' duration ({seg_duration}) exceeds common_t_stop ({common_t_stop})."
            )

        if padding_at_beginning > 1e-5:
            num_padding_applied_segs += 1

        # Group spikes by cluster
        cluster_groups = {k: v for k, v in seg_df.groupby('cluster')}

        for cluster in all_clusters:
            if cluster in cluster_groups:
                cluster_df = cluster_groups[cluster]
                spike_times = cluster_df.time.values - seg_start_time + padding_at_beginning
            else:
                spike_times = np.array([])

            spiketrain = neo.SpikeTrain(
                times=spike_times * pq.s,
                t_start=0 * pq.s,
                t_stop=common_t_stop * pq.s
            )
            seg_spiketrain.append(spiketrain)

        spiketrains.append(seg_spiketrain)
        processed_segments.append(seg)

    if num_padding_applied_segs > 0:
        pad_pos = 'beginning' if align_at_beginning else 'end'
        warnings.warn(
            f"{num_padding_applied_segs} segment(s) had padding at the {pad_pos} to match common_t_stop = {common_t_stop}s."
        )

    return spiketrains, np.array(processed_segments)


def assign_new_bin_aligned_at_end(df, new_segment_column='new_segment'):
    max_new_bin = df.groupby(new_segment_column).size().max()
    # assign new_bin so that the last bin in each segment is aligned at max_new_bin - 1
    df['segment_size'] = (
        df.groupby(new_segment_column)[new_segment_column].transform('count'))
    df['position_in_segment'] = (
        df.groupby(new_segment_column).cumcount())
    df['new_bin'] = (
        max_new_bin - df['segment_size'] +
        df['position_in_segment'])
    # drop helper columns if no longer needed
    df.drop(columns=['segment_size', 'position_in_segment'], inplace=True)
    return df


def _get_concat_gpfa_data(trajectories, spiketrain_corr_segs, bin_bounds, new_segments_for_gpfa=None):
    # Build a mapping from segment to trajectory index for faster lookup
    seg_to_traj_index = {seg: idx for idx,
                            seg in enumerate(spiketrain_corr_segs)}
    dfs = []

    for seg in new_segments_for_gpfa:
        if seg not in seg_to_traj_index:
            continue  # or handle missing case

        traj_index = seg_to_traj_index[seg]
        traj = trajectories[traj_index].T

        min_bin = bin_bounds.loc[seg, 'min']
        max_bin = bin_bounds.loc[seg, 'max']

        if max_bin + 1 > traj.shape[0]:
            raise ValueError(
                f'seg_max_new_bin[{seg}] > trajectories[{traj_index}].shape[0]')

        gpfa_trial = traj[min_bin:max_bin + 1, :]
        gpfa_trial_df = pd.DataFrame(
            gpfa_trial, columns=[f'dim_{i}' for i in range(gpfa_trial.shape[1])])
        gpfa_trial_df['new_segment'] = seg
        gpfa_trial_df['new_bin'] = range(min_bin, max_bin + 1)
        dfs.append(gpfa_trial_df)

    concat_gpfa_data = pd.concat(dfs, ignore_index=True)
    return concat_gpfa_data


def get_latent_neural_data_for_trial(trajectories, current_seg, trial_length, spiketrain_corr_segs, align_at_beginning=True):

    traj_index = np.where(spiketrain_corr_segs == current_seg)[0][0]

    if align_at_beginning:
        gpfa_trial = trajectories[traj_index].T[:trial_length, :]
    else:
        gpfa_trial = trajectories[traj_index].T[-trial_length:, :]
    return gpfa_trial