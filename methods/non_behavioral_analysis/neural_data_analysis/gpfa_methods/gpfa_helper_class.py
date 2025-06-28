import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
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
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
import quantities as pq
import neo
from sklearn.decomposition import PCA
from elephant.spike_train_generation import inhomogeneous_poisson_process


class GPFAHelperClass():

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

    def _find_shared_segments(self, recalculate=False):
        if not hasattr(self, 'shared_segments') or recalculate:
            segments_neural = self.spiketrain_corr_segs
            segments_behav = self.gpfa_behav_data['segment'].unique()
            segments_behav = segments_behav[segments_behav != '']
            self.shared_segments = [seg for seg in segments_behav.tolist()
                                    if seg in segments_neural.tolist()]
            self.shared_segments = np.array(self.shared_segments).astype(int)

    def _get_gpfa_behav_data(self, use_lagged_behav_data=False):

        if use_lagged_behav_data:
            self.gpfa_behav_data = self.y_var_lags_reduced
            if 'segment_0' in self.gpfa_behav_data.columns:
                self.gpfa_behav_data.drop(columns=['segment_0'], inplace=True)
        else:
            self.gpfa_behav_data = self.y_var_reduced

        self.gpfa_behav_data['segment'] = self.y_var['segment'].values

    def _get_trialwise_behav_data(self, use_lagged_behav_data=False):
        self.use_lagged_behav_data = use_lagged_behav_data
        self.behav_trials = []

        self._get_gpfa_behav_data(use_lagged_behav_data=use_lagged_behav_data)
        self._find_shared_segments()

        for seg in self.shared_segments:
            gpfa_behav_data_sub_mask = self.gpfa_behav_data['segment'] == seg
            gpfa_behav_data_sub = self.gpfa_behav_data[gpfa_behav_data_sub_mask]
            self.behav_trials.append(gpfa_behav_data_sub.values)

        self.gpfa_behav_data_columns = gpfa_behav_data_sub.columns

    def _get_spike_neural_data(self, use_lagged_raw_spike_data=False,
                               apply_pca_on_raw_spike_data=False,
                               num_pca_components=7):
        self.use_raw_spike_data_instead_of_gpfa = True
        self.apply_pca_on_raw_spike_data = apply_pca_on_raw_spike_data
        self.use_lagged_raw_spike_data = use_lagged_raw_spike_data
        self.num_pca_components = num_pca_components

        self.gpfa_neural_trials = []
        self._find_shared_segments()

        if use_lagged_raw_spike_data:
            x_var_df = self.x_var_lags
        else:
            x_var_df = self.x_var

        if apply_pca_on_raw_spike_data:
            pca = PCA(n_components=num_pca_components)
            x_var_df.drop(columns=['segment', 'bin'],
                          inplace=True, errors='ignore')
            x_var = pca.fit_transform(x_var_df)
            x_var_df = pd.DataFrame(
                x_var, columns=['pca_'+str(i) for i in range(num_pca_components)])

        return x_var_df

    def _get_trialwise_spike_neural_data(self,
                                         apply_pca_on_raw_spike_data=True,
                                         use_lagged_raw_spike_data=True,
                                         num_pca_components=7,
                                         ):
        x_var_df = self._get_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                               apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                               num_pca_components=num_pca_components)

        if not hasattr(self, 'gpfa_behav_data'):
            if not hasattr(self, 'use_lagged_behav_data'):
                self.use_lagged_behav_data = False
            self._get_gpfa_behav_data(
                use_lagged_behav_data=self.use_lagged_behav_data)

        self._find_shared_segments()

        for seg in self.shared_segments:
            gpfa_behav_data_sub_mask = self.gpfa_behav_data['segment'] == seg
            gpfa_trial = x_var_df[gpfa_behav_data_sub_mask]
            self.gpfa_neural_trials.append(gpfa_trial)

    def _get_trialwise_gpfa_neural_data(self):
        self.apply_pca_on_raw_spike_data = False
        self.gpfa_neural_trials = []

        if not hasattr(self, 'gpfa_behav_data'):
            if not hasattr(self, 'use_lagged_behav_data'):
                self.use_lagged_behav_data = False
            self._get_gpfa_behav_data(
                use_lagged_behav_data=self.use_lagged_behav_data)

        self._find_shared_segments()

        for seg in self.shared_segments:
            gpfa_behav_data_sub_mask = self.gpfa_behav_data['segment'] == seg
            trial_length = gpfa_behav_data_sub_mask.sum()
            gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
            self.gpfa_neural_trials.append(gpfa_trial)

    def get_trialwise_gpfa_and_behav_data(self, use_lagged_behav_data=False,
                                          use_raw_spike_data_instead=False,
                                          use_lagged_raw_spike_data=False,
                                          apply_pca_on_raw_spike_data=False,
                                          num_pca_components=7):

        self._get_trialwise_behav_data(
            use_lagged_behav_data=use_lagged_behav_data)

        if use_raw_spike_data_instead:
            self._get_trialwise_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                                  apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                                  num_pca_components=num_pca_components)
        else:
            self._get_trialwise_gpfa_neural_data()

    def get_concatenated_gpfa_and_behav_data_for_all_trials(self, use_lagged_behav_data=False,
                                                            use_raw_spike_data_instead=False,
                                                            use_lagged_raw_spike_data=False,
                                                            apply_pca_on_raw_spike_data=False,
                                                            num_pca_components=7):

        self._get_gpfa_behav_data(use_lagged_behav_data=use_lagged_behav_data)
        self._find_shared_segments()

        self.concat_behav_trials = self.gpfa_behav_data[self.gpfa_behav_data['segment'].isin(
            self.shared_segments)]

        if use_raw_spike_data_instead:
            x_var_df = self._get_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                                   apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                                   num_pca_components=num_pca_components)
            x_var_df['segment'] = self.gpfa_behav_data['segment'].values
            self.concat_neural_trials = x_var_df[x_var_df['segment'].isin(
                self.shared_segments)].copy()
            self.concat_neural_trials.drop(
                columns=['segment', 'bin'], inplace=True, errors='ignore')
        else:
            if not hasattr(self, 'gpfa_neural_trials'):
                self._get_trialwise_gpfa_neural_data()
            self.concat_neural_trials = np.concatenate(
                self.gpfa_neural_trials, axis=0)
