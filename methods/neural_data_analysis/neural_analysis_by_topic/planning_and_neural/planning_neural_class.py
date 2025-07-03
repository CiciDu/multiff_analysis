import sys
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.decode_targets import prep_target_decoder, behav_features_to_keep, target_decoder_class
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import planning_neural_utils, planning_neural_helper_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
import numpy as np
import pandas as pd
import os


class PlanningAndNeural(target_decoder_class.TargetDecoderClass):

    def __init__(self, raw_data_folder_path=None, bin_width=0.02, window_width=0.25,
                 one_behav_idx_per_bin=True):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         window_width=window_width,
                         one_behav_idx_per_bin=one_behav_idx_per_bin)
        self.planning_neural_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'planning_and_neural')
        os.makedirs(self.planning_neural_folder_path, exist_ok=True)

    def prep_data_to_analyze_planning(self,
                                      ref_point_mode='time after cur ff visible',
                                      ref_point_value=0.1,
                                      eliminate_outliers=False,
                                      use_curvature_to_ff_center=False,
                                      curv_of_traj_mode='distance',
                                      window_for_curv_of_traj=[-25, 25],
                                      truncate_curv_of_traj_by_time_of_capture=True,
                                      both_ff_across_time_df_exists_ok=True,
                                      ):

        # get neural data
        self.get_basic_data()
        self.retrieve_neural_data()

        # get behavioral_data
        planning_helper = planning_neural_helper_class.PlanningAndNeuralHelper(raw_data_folder_path=self.raw_data_folder_path,
                                                                               bin_width=self.bin_width,
                                                                               window_width=self.window_width,
                                                                               one_behav_idx_per_bin=self.one_behav_idx_per_bin)
        planning_helper.prep_behav_data_to_analyze_planning(ref_point_mode=ref_point_mode,
                                                            ref_point_value=ref_point_value,
                                                            curv_of_traj_mode=curv_of_traj_mode,
                                                            window_for_curv_of_traj=window_for_curv_of_traj,
                                                            truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
                                                            use_curvature_to_ff_center=use_curvature_to_ff_center,
                                                            eliminate_outliers=eliminate_outliers,
                                                            both_ff_across_time_df_exists_ok=both_ff_across_time_df_exists_ok)

        for attr in ['all_planning_info', 'both_ff_across_time_df']:
            setattr(self, attr, getattr(planning_helper, attr))

        # del planning_helper
        del planning_helper
        
        self._add_data_from_behav_data_all(exists_ok=True)

    
    def _add_data_from_behav_data_all(self, exists_ok=True):
        # Merge in columns from pn.behav_data_all that are not already present
        # (Assume pn is available in the current scope, or pass as argument if needed)
        try:
            self.get_behav_data(exists_ok=exists_ok)
            missing_cols = list(set(self.behav_data_all.columns) -
                                set(self.all_planning_info.columns))
            if 'point_index' in self.all_planning_info.columns and 'point_index' in self.behav_data_all.columns:
                # Check for missing point_index
                missing_point_indices = set(
                    self.all_planning_info['point_index']) - set(self.behav_data_all['point_index'])
                if missing_point_indices:
                    import warnings
                    warnings.warn(
                        f"There are {len(missing_point_indices)} point_index values in all_planning_info not present in pn.behav_data_all. Example: {list(missing_point_indices)[:5]}")
                # Merge only the missing columns
                self.all_planning_info = self.all_planning_info.merge(
                    self.behav_data_all[['point_index'] + missing_cols],
                    on=['point_index', 'bin'], how='left')
            else:
                raise ValueError("point_index is not in all_planning_info or behav_data_all")
        except Exception as e:
            print(
                f"[WARN] Could not merge extra columns from pn.behav_data_all: {e}")


    def get_x_and_y_data_for_modeling(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self.get_x_and_y_var_lags(
            max_x_lag_number=max_x_lag_number, max_y_lag_number=max_y_lag_number)
        self._reduce_x_var_lags()
        self.reduce_y_var_lags(exists_ok=exists_ok)

    def get_x_and_y_var(self, exists_ok=True):
        original_len = len(self.all_planning_info)
        self.y_var = self.all_planning_info.dropna().drop(
            columns={'stop_point_index', 'point_index'}, errors='ignore')
        print(f"{round(1 - len(self.y_var) / original_len, 2)}% of rows are dropped in all_planning_info due to having missing values")
        self.x_var = self.binned_spikes_df[self.binned_spikes_df['bin'].isin(
            self.y_var['bin'].values)].drop(columns=['bin'])
        print('binned_spikes_df.shape:', self.binned_spikes_df.shape)
        print('self.x_var.shape:', self.x_var.shape)
        print('self.y_var.shape:', self.y_var.shape)

        self.x_var.reset_index(drop=True, inplace=True)
        self.y_var.reset_index(drop=True, inplace=True)

        self._reduce_x_var()
        self.reduce_y_var(exists_ok=exists_ok)

    def get_x_and_y_var_lags(self, max_x_lag_number=5, max_y_lag_number=5):

        trial_vector = self.y_var['segment'].values
        self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                             continuous_data=self.x_var, trial_vector=trial_vector)
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=self.y_var, trial_vector=trial_vector)

    # ================================================

    def reduce_y_var(self,
                     save_data=True,
                     corr_threshold_for_lags_of_a_feature=0.98,
                     vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                     filter_corr_by_all_columns=True,
                     filter_vif_by_subsets=True,
                     filter_vif_by_all_columns=True,
                     exists_ok=True,
                     ):
        df_path = os.path.join(
            self.planning_neural_folder_path, 'pn_y_var_reduced.csv')

        self._reduce_y_var(df_path=df_path,
                           save_data=save_data,
                           corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                           vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                           vif_threshold=vif_threshold,
                           verbose=verbose,
                           filter_corr_by_all_columns=filter_corr_by_all_columns,
                           filter_vif_by_subsets=filter_vif_by_subsets,
                           filter_vif_by_all_columns=filter_vif_by_all_columns,
                           exists_ok=exists_ok)

    def reduce_y_var_lags(self,
                          df_path=None,
                          save_data=True,
                          corr_threshold_for_lags_of_a_feature=0.97,
                          vif_threshold_for_initial_subset=5,
                          vif_threshold=5,
                          verbose=True,
                          filter_corr_by_feature=False,
                          filter_corr_by_subsets=False,
                          filter_corr_by_all_columns=False,
                          filter_vif_by_feature=True,
                          filter_vif_by_subsets=False,
                          filter_vif_by_all_columns=True,
                          exists_ok=True):
        """Reduce y_var_lags by removing highly correlated and high VIF features.

        Parameters are passed to the parent class's reduce_y_var_lags method.
        Results are cached to avoid recomputation.
        """
        df_path = os.path.join(
            self.planning_neural_folder_path, 'pn_y_var_lags_reduced.csv')
        
        
        num_cols = len(self.y_var_lags.columns)
        if num_cols > 50:
            filter_corr_by_feature = True
            filter_corr_by_subsets = True
            filter_corr_by_all_columns = True
            filter_vif_by_feature = True
            filter_vif_by_subsets = True
            filter_vif_by_all_columns = True
            
        self._reduce_y_var_lags(df_path=df_path,
                                save_data=save_data,
                                corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                vif_threshold=vif_threshold,
                                verbose=verbose,
                                filter_corr_by_feature=filter_corr_by_feature,
                                filter_corr_by_subsets=filter_corr_by_subsets,
                                filter_corr_by_all_columns=filter_corr_by_all_columns,
                                filter_vif_by_feature=filter_vif_by_feature,
                                filter_vif_by_subsets=filter_vif_by_subsets,
                                filter_vif_by_all_columns=filter_vif_by_all_columns,
                                exists_ok=exists_ok)

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_corr(y_var_lags):
        subset_key_words = ['stop', 'speed_OR_ddv_OR_dw_OR_delta_OR_traj', 'LD_or_RD_or_gaze_or_view',
                            'distance', 'angle', 'frozen', 'dummy', 'num_or_any_or_rate']

        all_column_subsets = [
            [col for col in y_var_lags.columns if 'stop' in col],
            [col for col in y_var_lags.columns if ('speed' in col) or (
                'ddv' in col) or ('dw' in col) or ('delta' in col) or ('traj' in col)],
            [col for col in y_var_lags.columns if ('LD' in col) or (
                'RD' in col) or ('gaze' in col) or ('view' in col)],
            [col for col in y_var_lags.columns if ('distance' in col)],
            [col for col in y_var_lags.columns if ('angle' in col)],
            [col for col in y_var_lags.columns if ('frozen' in col)],
            [col for col in y_var_lags.columns if ('dummy' in col)],
            [col for col in y_var_lags.columns if (
                'num' in col) or ('any' in col) or ('rate' in col)],
        ]
        return subset_key_words, all_column_subsets

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_vif(y_var_lags):
        # might want to modify this later
        subset_key_words = ['stop', 'speed_OR_ddv_OR_dw_OR_delta_OR_traj', 'LD_or_RD_or_gaze_or_view',
                            'distance', 'angle', 'frozen', 'dummy', 'num_or_any_or_rate']

        all_column_subsets = [
            [col for col in y_var_lags.columns if 'stop' in col],
            [col for col in y_var_lags.columns if ('speed' in col) or (
                'ddv' in col) or ('dw' in col) or ('delta' in col) or ('traj' in col)],
            [col for col in y_var_lags.columns if ('LD' in col) or (
                'RD' in col) or ('gaze' in col) or ('view' in col)],
            [col for col in y_var_lags.columns if ('distance' in col)],
            [col for col in y_var_lags.columns if ('angle' in col)],
            [col for col in y_var_lags.columns if ('frozen' in col)],
            [col for col in y_var_lags.columns if ('dummy' in col)],
            [col for col in y_var_lags.columns if (
                'num' in col) or ('any' in col) or ('rate' in col)],
        ]
        return subset_key_words, all_column_subsets
