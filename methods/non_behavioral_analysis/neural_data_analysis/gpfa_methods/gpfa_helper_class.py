import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
from non_behavioral_analysis.neural_data_analysis.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils
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


class GPFAHelperClass():
    def __init__(self):
        pass
    

    def get_gpfa_traj(self, latent_dimensionality=10, exists_ok=True):
        """
        Compute or load GPFA trajectories.

        Parameters:
        -----------
        latent_dimensionality : int
            Number of latent dimensions for GPFA
        exists_ok : bool
            Whether to load existing trajectories if available
        """
        import pickle

        alignment = 'segStart' if self.align_at_beginning else 'segEnd'
        file_name = f'gpfa_neural_aligned_{alignment}_d{latent_dimensionality}.pkl'

        # Create filename with latent dimensionality to avoid conflicts
        trajectories_path = os.path.join(
            self.decoding_targets_folder_path, file_name)

        if exists_ok and os.path.exists(trajectories_path):
            try:
                with open(trajectories_path, 'rb') as f:
                    self.trajectories = pickle.load(f)
                print(f'Loaded GPFA trajectories from {trajectories_path}')
                return
            except Exception as e:
                print(f'Failed to load trajectories: {str(e)}. Recomputing...')

        # Compute trajectories if not loaded
        print(
            f'Computing GPFA trajectories with {latent_dimensionality} dimensions...')
        gpfa_3dim = GPFA(bin_size=self.bin_width_w_unit,
                         x_dim=latent_dimensionality)
        self.trajectories = gpfa_3dim.fit_transform(self.spiketrains)

        # Save trajectories
        try:
            with open(trajectories_path, 'wb') as f:
                pickle.dump(self.trajectories, f)
            print(f'Saved GPFA trajectories to {trajectories_path}')
        except Exception as e:
            print(f'Warning: Failed to save trajectories: {str(e)}')


    def prepare_spikes_for_gpfa(self, align_at_beginning=False):

        self.align_at_beginning = align_at_beginning

        spike_df = neural_data_processing.make_spike_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                        sampling_rate=self.sampling_rate)

        self.spike_segs_df = fit_gpfa_utils.make_spike_segs_df(
            spike_df, self.single_vis_target_df)

        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + self.bin_width
        self.spiketrains, self.spiketrain_corr_segs = fit_gpfa_utils.turn_spike_segs_df_into_spiketrains(
            self.spike_segs_df, common_t_stop=self.common_t_stop, align_at_beginning=self.align_at_beginning)


    def get_gpfa_and_behav_data_for_all_trials(self, use_lags=False):

        self.behav_trials = []
        self.gpfa_trials = []

        if use_lags:
            self.gpfa_y_var = self.y_var_lags_reduced
            if 'segment_0' in self.gpfa_y_var.columns:
                self.gpfa_y_var.drop(columns=['segment_0'], inplace=True)
        else:
            self.gpfa_y_var = self.y_var_reduced
        
        self.gpfa_y_var['segment'] = self.y_var['segment'].values

        segments_behav = self.gpfa_y_var['segment'].unique()
        segments_behav = segments_behav[segments_behav != '']
        segments_neural = self.spiketrain_corr_segs
        shared_segments = [seg for seg in segments_behav.tolist()
                           if seg in segments_neural.tolist()]

        for seg in shared_segments:
            gpfa_y_var_sub = self.gpfa_y_var[self.gpfa_y_var['segment'] == seg]
            self.behav_trials.append(gpfa_y_var_sub.values)

            trial_length = gpfa_y_var_sub.shape[0]
            gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
            self.gpfa_trials.append(gpfa_trial)
            
        self.gpfa_y_var_columns = gpfa_y_var_sub.columns

           
    # def get_gpfa_and_behav_data_for_all_trials(self):

    #     self.behav_trials = []
    #     self.gpfa_trials = []

    #     segments_behav = self.pursuit_data_by_trial['segment'].unique()
    #     segments_neural = self.spiketrain_corr_segs
    #     shared_segments = [seg for seg in segments_behav.tolist()
    #                        if seg in segments_neural.tolist()]

    #     for seg in shared_segments:
    #         pursuit_sub = self.pursuit_data_by_trial[self.pursuit_data_by_trial['segment'] == seg]
    #         behav_data_of_trial = pursuit_sub.drop(columns=['segment']).values
    #         self.behav_trials.append(behav_data_of_trial)

    #         trial_length = behav_data_of_trial.shape[0]
    #         gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
    #             self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
    #         self.gpfa_trials.append(gpfa_trial)

