import sys
from abc import abstractmethod
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, plot_gpfa_utils
from neural_data_analysis.neural_analysis_tools.align_trials import time_resolved_regression, time_resolved_gpfa_regression, plot_time_resolved_regression, align_trial_utils
import warnings
import os
import sys
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
import re


class GPFAHelperClass():

    def _prepare_spikes_for_gpfa(self, new_seg_info, align_at_beginning=False):

        self.align_at_beginning = align_at_beginning

        spikes_df = neural_data_processing.make_spikes_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                          sampling_rate=self.sampling_rate)

        self.spike_segs_df = fit_gpfa_utils.make_spike_segs_df(
            spikes_df, new_seg_info)

        # Perform rank check on spike count data (neurons × trials) and identify redundant neurons to drop
        self.get_concat_data_for_regression(use_raw_spike_data_instead=True)
        if len(self.dropped_neurons) > 0:
            extracted_numbers = [int(col.split('_')[-1])
                                 for col in self.dropped_neurons]
            self.spike_segs_df = self.spike_segs_df[~self.spike_segs_df['cluster'].isin(
                extracted_numbers)]

        # add a small value to common t stop
        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + 1e-6  # originally added bin_width

        self.spiketrains, self.spiketrain_corr_segs = fit_gpfa_utils.turn_spike_segs_df_into_spiketrains(
            self.spike_segs_df, new_seg_info['new_segment'].unique(), common_t_stop=self.common_t_stop, align_at_beginning=self.align_at_beginning)

    def get_gpfa_traj(self, latent_dimensionality=10, exists_ok=True, file_name=None):
        """
        Compute or load GPFA trajectories.

        Parameters:
        -----------
        latent_dimensionality : int
            Number of latent dimensions for GPFA
        exists_ok : bool
            Whether to load existing trajectories if available
        """

        self.alignment = 'segStart' if self.align_at_beginning else 'segEnd'
        self.latent_dimensionality = latent_dimensionality

        bin_width_str = f"{self.bin_width:.4f}".rstrip(
            '0').rstrip('.').replace('.', 'p')
        if file_name is None:
            file_name = f'gpfa_neural_aligned_{self.alignment}_bin{bin_width_str}_d{latent_dimensionality}.pkl'

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

    @abstractmethod
    def get_rebinned_behav_data(self):
        # this is a placeholder for the child class to implement
        # the function should make the rebinned_behav_data
        pass

    def get_concat_raw_spike_data(self,
                                  use_lagged_raw_spike_data=False,
                                  apply_pca_on_raw_spike_data=False,
                                  num_pca_components=7):
        if not hasattr(self, 'rebinned_behav_data'):
            raise ValueError(
                'rebinned_behav_data not found; please run get_rebinned_behav_data first')

        self.apply_pca_on_raw_spike_data = apply_pca_on_raw_spike_data
        self.use_lagged_raw_spike_data = use_lagged_raw_spike_data
        self.num_pca_components = num_pca_components

        x_var_df = self.get_raw_spikes_for_regression()
        # make sure that x_var_df has the same new_segment and new_bin as rebinned_behav_data
        x_var_df = x_var_df.merge(
            self.rebinned_behav_data[['new_segment', 'new_bin']], on=['new_segment', 'new_bin'], how='right')

        if apply_pca_on_raw_spike_data:
            pca = PCA(n_components=num_pca_components)
            x_var_df.drop(columns=['new_segment', 'new_bin'],
                          inplace=True, errors='ignore')
            x_var = pca.fit_transform(x_var_df)
            x_var_df = pd.DataFrame(
                x_var, columns=['pca_'+str(i) for i in range(num_pca_components)])
            x_var_df['new_segment'] = self.rebinned_behav_data['new_segment'].values
            x_var_df['new_bin'] = self.rebinned_behav_data['new_bin'].values
            self.concat_raw_spike_data = x_var_df
        else:
            self.concat_raw_spike_data, self.dropped_neurons = align_trial_utils.drop_redundant_neurons_from_concat_raw_spike_data(
                x_var_df)

    def get_concat_gpfa_data(self, new_segments_for_gpfa=None):
        self.apply_pca_on_raw_spike_data = False
        # Precompute min and max new_bin per segment
        bin_bounds = self.rebinned_behav_data.groupby(
            'new_segment')['new_bin'].agg(['min', 'max'])

        if new_segments_for_gpfa is None:
            new_segments_for_gpfa = self.new_seg_info['new_segment'].unique()

        self.concat_gpfa_data = fit_gpfa_utils._get_concat_gpfa_data(self.trajectories, self.spiketrain_corr_segs, bin_bounds,
                                                                     new_segments_for_gpfa=new_segments_for_gpfa)

    def _get_concat_data_for_regression(self,
                                        use_raw_spike_data_instead=False,
                                        use_lagged_raw_spike_data=False,
                                        apply_pca_on_raw_spike_data=False,
                                        num_pca_components=7):

        if not hasattr(self, 'rebinned_behav_data'):
            raise ValueError(
                'rebinned_behav_data not found; please run get_rebinned_behav_data first')

        self.use_raw_spike_data_instead = use_raw_spike_data_instead
        self.use_lagged_raw_spike_data = use_lagged_raw_spike_data
        self.apply_pca_on_raw_spike_data = apply_pca_on_raw_spike_data
        self.num_pca_components = num_pca_components

        self.concat_behav_trials = self.rebinned_behav_data[self.rebinned_behav_data['new_segment'].isin(
            self.new_seg_info['new_segment'].unique())]
        # The following columns will be dropped, since ['time', 'event_time', 'cum_distance', 'target_index', 'new_segment'] are enough, and the rest have high correlation with them
        columns_to_drop = ['point_index', 'last_target_caught_time',
                           'new_seg_end_time', 'new_seg_start_time', 'stop_point_index',
                           'current_target_caught_time', 'stop_time', 'seg_end_time',
                           'seg_start_time', 'cur_ff_index', 'trial',
                           'nxt_ff_index', 'segment']
        # and some more columns to drop that are meaningless to regress on
        columns_to_drop.extend(['crossing_boundary'])

        lagged_columns = specific_utils.find_lagged_versions_of_columns_in_df(
            columns_to_drop, self.concat_behav_trials)
        columns_to_drop.extend(lagged_columns)

        self.concat_behav_trials = self.concat_behav_trials.drop(
            columns=columns_to_drop, errors='ignore')

        if use_raw_spike_data_instead:
            self.get_concat_raw_spike_data(use_lagged_raw_spike_data=use_lagged_raw_spike_data,
                                           apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                           num_pca_components=num_pca_components)
            self.concat_neural_trials = self.concat_raw_spike_data
        else:
            self.get_concat_gpfa_data()
            self.concat_neural_trials = self.concat_gpfa_data

        # assert that their new_segment and new_bin are the same
        assert self.concat_behav_trials[['new_segment', 'new_bin']].equals(
            self.concat_neural_trials[['new_segment', 'new_bin']])

        # check for NA
        general_utils.check_na_in_df(self.concat_neural_trials, df_name='concat_neural_trials')
        general_utils.check_na_in_df(self.concat_behav_trials, df_name='concat_behav_trials')

    def print_data_dimensions(self):
        print("\n=== Data Dimensions Summary ===")
        print(
            f"Total number of trials: {len(self.new_seg_info['new_segment'].unique())}")
        print("\nConcatenated Data Shapes (total n_timepoints × n_features):")
        print(f"  concat_neural_trials:     {self.concat_neural_trials.shape}")
        print(f"  concat_behav_trials:      {self.concat_behav_trials.shape}")

    def make_time_resolved_cv_scores(self, cv_folds=10, features_to_include=None):
        self.max_timepoints = int(self.new_seg_duration/self.bin_width)

        if features_to_include is not None:
            # make sure 'new_bin' and 'new_segment' are included
            features_to_include = list(
                set([*features_to_include, 'new_bin', 'new_segment']))
            concat_behav_trials = self.concat_behav_trials[features_to_include]
        else:
            concat_behav_trials = self.concat_behav_trials

        self.time_resolved_cv_scores = time_resolved_regression.time_resolved_regression_cv(
            self.concat_neural_trials, concat_behav_trials, cv_folds=cv_folds, n_jobs=-1)

        if not hasattr(self, 'new_bin_start_time'):
            self.new_bin_start_time = 0

        self.time_resolved_cv_scores['bin_mid_time'] = self.time_resolved_cv_scores['new_bin'] * \
            self.bin_width + self.new_bin_start_time + self.bin_width/2

    def make_time_resolved_cv_scores_gpfa(self, cv_folds=10, latent_dimensionality=7):
        bin_bounds = self.rebinned_behav_data.groupby(
            'new_segment')['new_bin'].agg(['min', 'max'])

        self.bin_width_w_unit = self.bin_width * pq.s

        with warnings.catch_warnings():

            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings(
                'ignore', category=DeprecationWarning)

            self.time_resolved_cv_scores_gpfa = time_resolved_gpfa_regression.time_resolved_gpfa_regression_cv(
                self.concat_behav_trials, self.spiketrains, self.spiketrain_corr_segs, bin_bounds, self.bin_width_w_unit,
                cv_folds=cv_folds, n_jobs=-1, latent_dimensionality=latent_dimensionality,
            )
        
        self.time_resolved_cv_scores_gpfa['bin_mid_time'] = self.time_resolved_cv_scores_gpfa['new_bin'] * \
            self.bin_width + self.new_bin_start_time + self.bin_width/2
            
        self.time_resolved_cv_scores_gpfa['trial_count'] = self.time_resolved_cv_scores_gpfa['train_trial_count']

    def _get_time_resolved_cv_scores_file_path(self, folder_name, file_name, cv_folds=5, latent_dimensionality=7):
        if file_name is None:
            alignment = 'segStart' if self.align_at_beginning else 'segEnd'
            bin_width_str = f"{self.bin_width:.4f}".rstrip(
                '0').rstrip('.').replace('.', 'p')
            file_name = f'scores_{alignment}_bin{bin_width_str}_d{latent_dimensionality}_cv{cv_folds}.csv'
        time_resolved_cv_scores_gpfa_folder_path = os.path.join(
            self.gpfa_data_folder_path, folder_name)
        os.makedirs(time_resolved_cv_scores_gpfa_folder_path, exist_ok=True)
        time_resolved_cv_scores_path = os.path.join(
            time_resolved_cv_scores_gpfa_folder_path, file_name)

        return time_resolved_cv_scores_path

    def retrieve_or_make_time_resolved_cv_scores(self, exists_ok=True, cv_folds=5, file_name=None):

        if not self.use_raw_spike_data_instead:
            folder_name = 'time_resolved_cv_scores/precomputed_gpfa'
        else:
            if not self.use_lagged_raw_spike_data:
                folder_name = 'time_resolved_cv_scores/raw_spike'
            else:
                folder_name = 'time_resolved_cv_scores/lagged_raw_spike'
            if self.apply_pca_on_raw_spike_data:
                folder_name = f'{folder_name}_pca{self.num_pca_components}'

        time_resolved_cv_scores_path = self._get_time_resolved_cv_scores_file_path(folder_name, file_name,
                                                                                   cv_folds=cv_folds, latent_dimensionality=self.latent_dimensionality)

        if exists_ok:
            if os.path.exists(time_resolved_cv_scores_path):
                self.time_resolved_cv_scores = pd.read_csv(time_resolved_cv_scores_path,
                                                           index_col=0)
                print(
                    f'Loaded time_resolved_cv_scores from {time_resolved_cv_scores_path}')
                return
            else:
                print(
                    f'File {time_resolved_cv_scores_path} does not exist. Recomputing...')

        self.make_time_resolved_cv_scores(
            cv_folds=cv_folds)
        self.time_resolved_cv_scores.to_csv(time_resolved_cv_scores_path)
        print(
            f'Saved time_resolved_cv_scores to {time_resolved_cv_scores_path}')
        return

    def retrieve_or_make_time_resolved_cv_scores_gpfa(self, exists_ok=True, cv_folds=5, latent_dimensionality=7, file_name=None):

        time_resolved_cv_scores_path = self._get_time_resolved_cv_scores_file_path('time_resolved_cv_scores_gpfa', file_name,
                                                                                   cv_folds=cv_folds, latent_dimensionality=latent_dimensionality)

        if exists_ok:
            if os.path.exists(time_resolved_cv_scores_path):
                self.time_resolved_cv_scores_gpfa = pd.read_csv(time_resolved_cv_scores_path,
                                                                index_col=0)
                print(
                    f'Loaded time_resolved_cv_scores_gpfa from {time_resolved_cv_scores_path}')
                return
            else:
                print(
                    f'File {time_resolved_cv_scores_path} does not exist. Recomputing...')

        self.make_time_resolved_cv_scores_gpfa(
            cv_folds=cv_folds, latent_dimensionality=latent_dimensionality)
        
        self.time_resolved_cv_scores_gpfa.to_csv(time_resolved_cv_scores_path)
        print(
            f'Saved time_resolved_cv_scores_gpfa to {time_resolved_cv_scores_path}')
        return

    def plot_time_resolved_regression(self, time_resolved_cv_scores=None, features_to_plot=None,
                                      features_not_to_plot=None, score_threshold_to_plot=None, rank_by_max_score=True):
        if not hasattr(self, 'event_time'):
            self.event_time = None
        if time_resolved_cv_scores is None:
            time_resolved_cv_scores = self.time_resolved_cv_scores
        plot_time_resolved_regression._plot_time_resolved_regression(time_resolved_cv_scores, event_time=self.event_time,
                                                                     rank_by_max_score=rank_by_max_score,
                                                                     features_to_plot=features_to_plot, features_not_to_plot=features_not_to_plot,
                                                                     score_threshold_to_plot=score_threshold_to_plot)

    def plot_trial_counts_by_timepoint(self):
        plot_time_resolved_regression.plot_trial_counts_by_timepoint(
            self.time_resolved_cv_scores)

    def streamline_getting_time_resolved_cv_scores(self,
                                                planning_data_by_point_exists_ok=True,
                                                latent_dimensionality=7,
                                                cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                                pre_event_window=0.25, post_event_window=0.75,
                                                cv_folds=5):
        # get data
        self.prep_data_to_analyze_planning(
            planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
        self.planning_data_by_point, cols_to_drop = general_utils.drop_columns_with_many_nans(
            self.planning_data_by_point)
        self.prepare_seg_aligned_data(cur_or_nxt=cur_or_nxt, first_or_last=first_or_last, time_limit_to_count_sighting=time_limit_to_count_sighting,
                                    pre_event_window=pre_event_window, post_event_window=post_event_window)

        # time_resolved_cv_scores_gpfa
        self.get_concat_data_for_regression(use_raw_spike_data_instead=True)
        self.retrieve_or_make_time_resolved_cv_scores_gpfa(
            latent_dimensionality=latent_dimensionality, cv_folds=cv_folds)

        # time_resolved_cv_scores
        self.get_gpfa_traj(
            latent_dimensionality=latent_dimensionality, exists_ok=True)
        self.get_concat_data_for_regression(use_raw_spike_data_instead=False)
        self.retrieve_or_make_time_resolved_cv_scores(cv_folds=cv_folds)
