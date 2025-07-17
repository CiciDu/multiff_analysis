import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantities as pq
import neo


def make_spike_segs_df(spike_df, new_seg_info):
    # Create list of spike segments by splitting spike_df based on time windows in new_seg_info
    # Each segment contains spikes between last_vis_time and ff_caught_time for a given trial
    spike_segments = []

    # drop segments with 0 duration
    new_seg_info_sub = new_seg_info[new_seg_info['new_seg_duration'] > 0]

    for index, row in new_seg_info_sub.iterrows():
        mask = spike_df.time.between(
            row['new_seg_start_time'], row['new_seg_end_time'])
        spikes_sub = spike_df[mask].copy()
        
        cols = ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
        spikes_sub[cols] = row[cols].values
        
        spike_segments.append(spikes_sub)

    spike_segs_df = pd.concat(spike_segments, ignore_index=True)

    spike_segs_df['t_duration'] = spike_segs_df['new_seg_end_time'] - \
        spike_segs_df['new_seg_start_time']

    return spike_segs_df


def turn_spike_segs_df_into_spiketrains(spike_segs_df, common_t_stop, align_at_beginning=False):
    import neo
    import quantities as pq

    spiketrains = []
    spiketrain_corr_segs = []

    # Get full list of clusters (sorted for consistent ordering)
    all_clusters = np.sort(spike_segs_df['cluster'].unique())

    # Group once by segment
    segment_groups = spike_segs_df.groupby('new_segment')

    for seg, seg_df in segment_groups:
        seg_start_time = seg_df.new_seg_start_time.iloc[0]
        seg_duration = seg_df.new_seg_duration.iloc[0]
        padding_at_beginning = 0 if align_at_beginning else (
            common_t_stop - seg_duration)

        # Group cluster data within this segment
        cluster_groups = dict(tuple(seg_df.groupby('cluster')))

        seg_spiketrain = []
        for cluster in all_clusters:
            if cluster in cluster_groups:
                cluster_df = cluster_groups[cluster]
                spike_time = cluster_df.time.values - seg_start_time + padding_at_beginning
            else:
                spike_time = np.array([])

            spiketrain = neo.SpikeTrain(
                times=spike_time * pq.s,
                t_start=0,
                t_stop=common_t_stop * pq.s
            )
            seg_spiketrain.append(spiketrain)

        spiketrains.append(seg_spiketrain)
        spiketrain_corr_segs.append(seg)

    return spiketrains, np.array(spiketrain_corr_segs)
