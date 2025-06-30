import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantities as pq
import neo


def make_spike_segs_df(spike_df, single_vis_target_df):
    # Create list of spike segments by splitting spike_df based on time windows in single_vis_target_df
    # Each segment contains spikes between last_vis_time and ff_caught_time for a given trial
    spike_segments = []

    # drop segments with 0 duration
    single_vis_target_df_sub = single_vis_target_df[single_vis_target_df['seg_duration'] > 0]

    for index, row in single_vis_target_df_sub.iterrows():
        mask = spike_df.time.between(
            row['seg_start_time'], row['seg_end_time'])
        spikes_sub = spike_df[mask].copy()
        spikes_sub['segment'] = row['segment']
        spikes_sub['seg_start_time'] = row['seg_start_time']
        spikes_sub['seg_end_time'] = row['seg_end_time']
        spikes_sub['seg_duration'] = row['seg_duration']
        spike_segments.append(spikes_sub)

    spike_segs_df = pd.concat(spike_segments, ignore_index=True)

    spike_segs_df['t_duration'] = spike_segs_df['seg_end_time'] - \
        spike_segs_df['seg_start_time']

    return spike_segs_df


def turn_spike_segs_df_into_spiketrains(spike_segs_df, common_t_stop, align_at_beginning=False):
    # Get unique clusters and segments
    clusters = spike_segs_df.cluster.unique()
    segments = spike_segs_df.segment.unique()

    # Create spiketrain objects (in Neo)
    spiketrains = []
    spiketrain_corr_segs = []
    

    # Process each segment and cluster combination
    for seg in segments:
        # Get data for this segment
        spike_df_trial = spike_segs_df[spike_segs_df.segment == seg]

        # Get segment start and stop times (should be the same for all rows in this segment)
        seg_start_time = spike_df_trial.seg_start_time.iloc[0]

        seg_spiketrain = []

        for cluster in clusters:
            # Get spikes for this cluster in this segment
            sub = spike_df_trial[spike_df_trial.cluster == cluster]

            # Calculate relative spike times
            spike_time = sub.time - seg_start_time
            if not align_at_beginning:
                padding_at_beginning = common_t_stop - \
                    spike_df_trial.seg_duration.iloc[0]
                spike_time = spike_time + padding_at_beginning

            # Create SpikeTrain object
            spiketrain = neo.SpikeTrain(
                times=spike_time.values * pq.s,  # Convert to quantities
                t_start=0,
                t_stop=common_t_stop * pq.s
            )
            seg_spiketrain.append(spiketrain)

        spiketrains.append(seg_spiketrain)
        spiketrain_corr_segs.append(seg)

    spiketrain_corr_segs = np.array(spiketrain_corr_segs)

    return spiketrains, spiketrain_corr_segs
