import sys
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from null_behaviors import curvature_utils
from non_behavioral_analysis.neural_data_analysis.decode_targets import prep_decode_target, behav_features_to_keep
from non_behavioral_analysis.neural_data_analysis.planning_neural import planning_neural_utils, planning_neural_helper_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
import numpy as np
import pandas as pd
import os


class PlanningAndNeural(planning_neural_helper_class.PlanningAndNeuralHelper):

    def __init__(self, raw_data_folder_path=None, bin_width=0.02, window_width=0.25,
                 one_behav_idx_per_bin=True):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         window_width=window_width,
                         one_behav_idx_per_bin=one_behav_idx_per_bin)

    def prep_data_to_analyze_planning(self,
                                      ref_point_mode='time after cur ff visible',
                                      ref_point_value=0.1,
                                      normalize=False,
                                      eliminate_outliers=False,
                                      use_curvature_to_ff_center=False,
                                      curv_of_traj_mode='distance',
                                      window_for_curv_of_traj=[-25, 25],
                                      truncate_curv_of_traj_by_time_of_capture=True
                                      ):
        self.streamline_organizing_info(ref_point_mode=ref_point_mode,
                                        ref_point_value=ref_point_value,
                                        curv_of_traj_mode=curv_of_traj_mode,
                                        window_for_curv_of_traj=window_for_curv_of_traj,
                                        truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
                                        use_curvature_to_ff_center=use_curvature_to_ff_center,
                                        eliminate_outliers=eliminate_outliers)

        self.retrieve_neural_data()
        self.get_all_planning_info()

    def get_x_and_y_data_for_modeling(self, exists_ok=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self.reduce_x_var_lags()
        self.reduce_y_var_lags(exists_ok=exists_ok)

        
    def get_x_and_y_var(self):
        original_len = len(self.all_planning_info)
        self.y_var = self.all_planning_info.dropna().drop(columns={'stop_point_index', 'point_index'})
        print(f"{round(1 - len(self.y_var) / original_len, 2)}% of rows are dropped in all_planning_info due to having missing values")
        self.x_var = self.binned_spikes_df[self.binned_spikes_df['bin'].isin(self.y_var['bin'].values)].drop(columns=['bin'])

        print('binned_spikes_df.shape:', self.binned_spikes_df.shape)
        print('self.x_var.shape:', self.x_var.shape)
        print('self.y_var.shape:', self.y_var.shape)


    def get_x_and_y_var_lags(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):

        x_var_lags_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_x_var_lags.csv')
        y_var_lags_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_lags.csv')

        if exists_ok & os.path.exists(x_var_lags_path) & os.path.exists(y_var_lags_path):
            self.x_var_lags = pd.read_csv(x_var_lags_path)
            self.y_var_lags = pd.read_csv(y_var_lags_path)
            print(
                f'Loaded x_var_lags and y_var_lags from {x_var_lags_path} and {y_var_lags_path}')
        else:
            
            neural_vs_behavioral_class.NeuralVsBehavioralClass._get_x_var_lags(self, max_x_lag_number=max_x_lag_number,
                                 continuous_data=self.x_var)
            neural_vs_behavioral_class.NeuralVsBehavioralClass._get_y_var_lags(self, max_y_lag_number=max_y_lag_number,
                                 continuous_data=self.y_var)

            # self.x_var_lags = self.x_var_lags[self.x_var_lags['bin'].isin(
            #     self.pursuit_data['bin'].values)]
            # self.y_var_lags = self.y_var_lags[self.y_var_lags['bin'].isin(
            #     self.pursuit_data['bin'].values)]
            # self.x_var_lags.to_csv(x_var_lags_path, index=False)
            # self.y_var_lags.to_csv(y_var_lags_path, index=False)
            # print(
            #     f'Saved x_var_lags and y_var_lags to {x_var_lags_path} and {y_var_lags_path}')
            
            
            
    # ================================================
    
    def reduce_x_var(self):
        self.x_var_reduced = prep_decode_target.remove_zero_var_cols(
            self.x_var)

    def reduce_x_var_lags(self):
        self.x_var_lags_reduced = prep_decode_target.remove_zero_var_cols(
            self.x_var_lags)

    def reduce_y_var(self, corr_threshold_for_lags_of_a_feature=0.98,
                     vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                     filter_corr_by_all_columns=False,
                     filter_vif_by_subsets=True,
                     filter_vif_by_all_columns=True,
                     exists_ok=True,
                     ):
        df_path = os.path.join(
            self.decoding_targets_folder_path, 'plan_neural_y_var_reduced.csv')
        
        if exists_ok and os.path.exists(df_path):
            self.y_var_reduced = pd.read_csv(df_path)
            print(f'Loaded y_var_reduced from {df_path}')
        else:
            # drop columns with std less than 0.001
            columns_w_small_std = self.y_var.std(
            )[self.y_var.std() < 0.001].index.tolist()
            self.y_var_reduced = self.y_var.drop(columns=columns_w_small_std)
            
            
            neural_vs_behavioral_class.NeuralVsBehavioralClass(self.y_var_reduced,
                               filter_corr_by_all_columns=filter_corr_by_all_columns,
                               corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                               vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                               vif_threshold=vif_threshold,
                               verbose=verbose,
                               filter_vif_by_subsets=filter_vif_by_subsets,
                               filter_vif_by_all_columns=filter_vif_by_all_columns)
            self.y_var_reduced.to_csv(df_path, index=False)
            print(f'Saved y_var_reduced to {df_path}')




    def reduce_y_var_lags(self, corr_threshold_for_lags_of_a_feature=0.85,
                          vif_threshold_for_initial_subset=5,
                          vif_threshold=5,
                          verbose=True,
                          filter_corr_by_feature=True,
                          filter_corr_by_subsets=True,
                          filter_corr_by_all_columns=True,
                          filter_vif_by_feature=True,
                          filter_vif_by_subsets=True,
                          filter_vif_by_all_columns=False,
                          exists_ok=True):
        """Reduce y_var_lags by removing highly correlated and high VIF features.

        Parameters are passed to the parent class's reduce_y_var_lags method.
        Results are cached to avoid recomputation.
        """
        df_path = os.path.join(
            self.decoding_targets_folder_path, 'decode_target_y_var_lags_reduced.csv')

        # Try to load cached results if allowed
        if exists_ok and os.path.exists(df_path):
            try:
                cached_data = pd.read_csv(df_path)
                if len(cached_data) == len(self.y_var_lags):
                    self.y_var_lags_reduced = cached_data
                    if verbose:
                        print(
                            f'Loaded y_var_lags_reduced from {df_path}')
                    return
            except (pd.errors.EmptyDataError, ValueError) as e:
                if verbose:
                    print(f'Failed to load cached data: {str(e)}')

        # If we get here, we need to recompute
        if verbose:
            print('Computing reduced y_var_lags...')

        # Call parent class method to do the actual reduction
        super().reduce_y_var_lags(
            corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
            vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
            vif_threshold=vif_threshold,
            verbose=verbose,
            filter_corr_by_feature=filter_corr_by_feature,
            filter_corr_by_subsets=filter_corr_by_subsets,
            filter_corr_by_all_columns=filter_corr_by_all_columns,
            filter_vif_by_feature=filter_vif_by_feature,
            filter_vif_by_subsets=filter_vif_by_subsets,
            filter_vif_by_all_columns=filter_vif_by_all_columns
        )

        # Cache the results
        try:
            self.y_var_lags_reduced.to_csv(df_path, index=False)
            if verbose:
                print(f'Saved reduced y_var_lags to {df_path}')
        except Exception as e:
            if verbose:
                print(f'Warning: Failed to cache results: {str(e)}')

