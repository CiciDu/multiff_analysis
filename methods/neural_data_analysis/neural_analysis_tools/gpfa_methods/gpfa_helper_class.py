import sys
from abc import abstractmethod
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils
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
import pickle


class GPFAHelperClass():

    def _prepare_spikes_for_gpfa(self, new_seg_info, align_at_beginning=False):

        self.align_at_beginning = align_at_beginning

        spike_df = neural_data_processing.make_spike_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                        sampling_rate=self.sampling_rate)

        self.spike_segs_df = fit_gpfa_utils.make_spike_segs_df(
            spike_df, new_seg_info)

        # add a small value to common t stop
        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + 1e-6  # originally added bin_width

        self.spiketrains, self.spiketrain_corr_segs = fit_gpfa_utils.turn_spike_segs_df_into_spiketrains(
            self.spike_segs_df, common_t_stop=self.common_t_stop, align_at_beginning=self.align_at_beginning)

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

        alignment = 'segStart' if self.align_at_beginning else 'segEnd'

        bin_width_str = f"{self.bin_width:.4f}".rstrip(
            '0').rstrip('.').replace('.', 'p')
        file_name = f'gpfa_neural_aligned_{alignment}_bin{bin_width_str}_d{latent_dimensionality}.pkl'

        # Create filename with latent dimensionality to avoid conflicts
        trajectories_folder_path = os.path.join(
            self.gpfa_data_folder_path, 'gpfa_trajectories')
        os.makedirs(trajectories_folder_path, exist_ok=True)
        trajectories_path = os.path.join(
            trajectories_folder_path, file_name)

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
        self.bin_width_w_unit = self.bin_width * pq.s
        gpfa_3dim = GPFA(bin_size=self.bin_width_w_unit,
                         x_dim=latent_dimensionality)
        # suppress warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self.trajectories = gpfa_3dim.fit_transform(self.spiketrains)

        # Save trajectories
        try:
            with open(trajectories_path, 'wb') as f:
                pickle.dump(self.trajectories, f)
            print(f'Saved GPFA trajectories to {trajectories_path}')
        except Exception as e:
            print(f'Warning: Failed to save trajectories: {str(e)}')

    def _find_shared_segments(self, recalculate=False):
        if not hasattr(self, 'shared_segments') or recalculate:
            segments_neural = self.spiketrain_corr_segs
            segments_behav = self.rebinned_behav_data['new_segment'].unique()
            segments_behav = segments_behav[segments_behav != '']
            self.shared_segments = [seg for seg in segments_behav.tolist()
                                    if seg in segments_neural.tolist()]
            self.shared_segments = np.array(self.shared_segments).astype(int)

    @abstractmethod
    def get_rebinned_behav_data(self):
        # this is a placeholder for the child class to implement
        # the function should make the rebinned_behav_data
        pass

    def _get_trialwise_behav_data(self):
        self.behav_trials = []

        self._find_shared_segments()

        for seg in self.shared_segments:
            rebinned_behav_data_sub = self.rebinned_behav_data[self.rebinned_behav_data['new_segment'] == seg]
            self.behav_trials.append(rebinned_behav_data_sub.values)

        self.rebinned_behav_data_columns = rebinned_behav_data_sub.columns

    def _get_spike_neural_data(self, use_lagged_raw_spike_data=False,
                               apply_pca_on_raw_spike_data=False,
                               num_pca_components=7):
        self.use_raw_spike_data_instead_of_gpfa = True
        self.apply_pca_on_raw_spike_data = apply_pca_on_raw_spike_data
        self.use_lagged_raw_spike_data = use_lagged_raw_spike_data
        self.num_pca_components = num_pca_components

        self._find_shared_segments()

        x_var_df = self.get_raw_spikes_for_regression()
        x_var_df = self.rebinned_behav_data[['new_segment', 'new_bin']].merge(
            x_var_df, on=['new_segment', 'new_bin'], how='left')

        if apply_pca_on_raw_spike_data:
            pca = PCA(n_components=num_pca_components)
            x_var_df.drop(columns=['new_segment', 'new_bin'],
                          inplace=True, errors='ignore')
            x_var = pca.fit_transform(x_var_df)
            x_var_df = pd.DataFrame(
                x_var, columns=['pca_'+str(i) for i in range(num_pca_components)])
            x_var_df['new_segment'] = self.rebinned_behav_data['new_segment'].values
            x_var_df['new_bin'] = self.rebinned_behav_data['new_bin'].values

        return x_var_df

    def _get_trialwise_spike_neural_data(self,
                                         apply_pca_on_raw_spike_data=True,
                                         use_lagged_raw_spike_data=True,
                                         num_pca_components=7,
                                         ):
        if not hasattr(self, 'rebinned_behav_data'):
            raise ValueError(
                'rebinned_behav_data not found; please run get_rebinned_behav_data first')

        x_var_df = self._get_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                               apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                               num_pca_components=num_pca_components)
        x_var = x_var_df.values
        self._find_shared_segments()

        for seg in self.shared_segments:
            rebinned_behav_data_sub_mask = self.rebinned_behav_data['new_segment'] == seg
            gpfa_trial = x_var[rebinned_behav_data_sub_mask]
            self.gpfa_neural_trials.append(gpfa_trial)

    def _get_trialwise_gpfa_neural_data(self):
        self.apply_pca_on_raw_spike_data = False
        self.gpfa_neural_trials = []

        self._find_shared_segments()

        # get trial length of each segment
        seg_trial_lengths = self.rebinned_behav_data.groupby('new_segment').size()

        for seg in self.shared_segments:
            trial_length = seg_trial_lengths[seg]
            gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs, align_at_beginning=self.align_at_beginning)
            self.gpfa_neural_trials.append(gpfa_trial)

    def get_trialwise_gpfa_and_behav_data(self,
                                          use_raw_spike_data_instead=False,
                                          use_lagged_raw_spike_data=False,
                                          apply_pca_on_raw_spike_data=False,
                                          num_pca_components=7):

        self._get_trialwise_behav_data(
        )

        if use_raw_spike_data_instead:
            self._get_trialwise_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                                  apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                                  num_pca_components=num_pca_components)
        else:
            self._get_trialwise_gpfa_neural_data()

    def get_concatenated_gpfa_and_behav_data_for_all_trials(self,
                                                            use_raw_spike_data_instead=False,
                                                            use_lagged_raw_spike_data=False,
                                                            apply_pca_on_raw_spike_data=False,
                                                            num_pca_components=7):

        if not hasattr(self, 'rebinned_behav_data'):
            raise ValueError(
                'rebinned_behav_data not found; please run get_rebinned_behav_data first')

        self._find_shared_segments()

        self.concat_behav_trials = self.rebinned_behav_data[self.rebinned_behav_data['new_segment'].isin(
            self.shared_segments)]

        if use_raw_spike_data_instead:
            x_var_df = self._get_spike_neural_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                                   apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                                   num_pca_components=num_pca_components)
            self.concat_neural_trials = x_var_df[x_var_df['new_segment'].isin(
                self.shared_segments)].copy()
            self.concat_neural_trials.drop(
                columns=['new_segment', 'new_bin'], inplace=True, errors='ignore')
        else:
            if not hasattr(self, 'gpfa_neural_trials'):
                self._get_trialwise_gpfa_neural_data()
            self.concat_neural_trials = np.concatenate(
                self.gpfa_neural_trials, axis=0)
            # label each hidden dimension as a column
            self.concat_neural_trials = pd.DataFrame(self.concat_neural_trials, columns=[
                                                     'dim_'+str(i) for i in range(self.concat_neural_trials.shape[1])])
