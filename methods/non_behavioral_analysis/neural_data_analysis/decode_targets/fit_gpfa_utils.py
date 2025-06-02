import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
import warnings
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
import plotly.graph_objects as go
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process


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
