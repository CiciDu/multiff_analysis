from data_wrangling import further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.topic_based_neural_analysis.target_decoder import prep_target_decoder
import os
import numpy as np
import pandas as pd
from os.path import exists


class NeuralBaseClass(further_processing_class.FurtherProcessing):
    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.02,

                 one_point_index_per_bin=True):

        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.bin_width = bin_width
        self.one_point_index_per_bin = one_point_index_per_bin
        self.max_bin = None

    def get_basic_data(self):
        self.retrieve_or_make_monkey_data(already_made_ok=True)
        self.make_or_retrieve_ff_dataframe(
            already_made_ok=True, exists_ok=True)

    def retrieve_neural_data(self, binned_spikes_df_exists_ok=True):

        self.sampling_rate = 20000 if 'Bruno' in self.raw_data_folder_path else 30000
        bin_width_str = str(self.bin_width).replace('.', 'p')
        binned_spikes_df_path = os.path.join(
            self.processed_neural_data_folder_path, f"binned_spikes_df_{bin_width_str}.csv")

        if binned_spikes_df_exists_ok & os.path.exists(binned_spikes_df_path):
            self.binned_spikes_df = pd.read_csv(binned_spikes_df_path)
            last_bin = self.binned_spikes_df['bin'].max()
            self.time_bins = np.arange(
                0, (last_bin + 2) * self.bin_width, self.bin_width)
            print(f'Loaded binned_spikes_df from {binned_spikes_df_path}')
        else:
            if not hasattr(self, 'ff_caught_T_sorted'):
                self.get_basic_data()

            self.spikes_df = neural_data_processing.make_spikes_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                                   sampling_rate=self.sampling_rate)

            self.time_bins, self.binned_spikes_df = neural_data_processing.prepare_binned_spikes_df(
                self.spikes_df, bin_width=self.bin_width)
            self.binned_spikes_df.to_csv(binned_spikes_df_path, index=False)
            print(
                f'Made new binned_spikes_df and saved to {binned_spikes_df_path}')

        if self.max_bin is not None:
            self.binned_spikes_df = self.binned_spikes_df[self.binned_spikes_df['bin'] <= self.max_bin]

    def _make_or_retrieve_target_df(self, exists_ok=True, fill_na=False):
        target_df_filepath = os.path.join(
            self.patterns_and_features_folder_path, 'target_df.csv')
        if exists(target_df_filepath) & exists_ok:
            self.target_df = pd.read_csv(target_df_filepath)
            print("Retrieved target_df")
        else:
            self.target_df = prep_target_data.make_target_df(
                self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted,
                self.ff_dataframe, max_visibility_window=self.max_visibility_window)
            self.target_df.to_csv(target_df_filepath, index=False)
            print("Made new target_df")

        if fill_na:
            self.target_df = prep_target_data.fill_na_in_target_df(
                self.target_df)

        self.target_df = prep_target_data.add_columns_to_target_df(
            self.target_df)

    def _add_ff_info(self, binned_features):
        ff_info = prep_monkey_data.get_ff_info_for_bins(
            binned_features[['bin']], self.ff_dataframe, self.ff_caught_T_new, self.time_bins)
        # delete columns in ff_info that are duplicated in behav_data except for bin
        columns_to_drop = [
            col for col in ff_info.columns if col in binned_features.columns and col != 'bin']
        ff_info = ff_info.drop(columns=columns_to_drop)
        binned_features = binned_features.merge(
            ff_info, on='bin', how='left')
        return binned_features

    def _get_curv_of_traj_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 0]):
        self.curv_of_traj_df = self.get_curv_of_traj_df(
            window_for_curv_of_traj=window_for_curv_of_traj,
            curv_of_traj_mode=curv_of_traj_mode,
            truncate_curv_of_traj_by_time_of_capture=False
        )

    def _reduce_y_var_base(self, y_var, corr_threshold_for_lags_of_a_feature=0.85,
                           vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                           filter_corr_by_all_columns=True,
                           filter_vif_by_subsets=True,
                           filter_vif_by_all_columns=True,
                           ):

        y_var_reduced = prep_target_decoder.remove_zero_var_cols(
            y_var)

        # Call the function to iteratively drop lags with high correlation for each feature
        self.y_var_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(y_var_reduced,
                                                                                  corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                                  verbose=verbose,
                                                                                  filter_by_feature=False,
                                                                                  filter_by_subsets=False,
                                                                                  filter_by_all_columns=filter_corr_by_all_columns)

        self.y_var_reduced = drop_high_vif_vars.drop_columns_with_high_vif(self.y_var_reduced_corr,
                                                                           vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                                                           vif_threshold=vif_threshold,
                                                                           verbose=verbose,
                                                                           filter_by_feature=False,
                                                                           filter_by_subsets=filter_vif_by_subsets,
                                                                           filter_by_all_columns=filter_vif_by_all_columns,
                                                                           get_column_subsets_func=self.get_subset_key_words_and_all_column_subsets_for_vif)

    def _reduce_y_var_lags_base(self, corr_threshold_for_lags_of_a_feature=0.85,
                                vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                                filter_corr_by_feature=True,
                                filter_corr_by_subsets=True,
                                filter_corr_by_all_columns=True,
                                filter_vif_by_feature=True,
                                filter_vif_by_subsets=True,
                                filter_vif_by_all_columns=False,
                                ):

        y_var_lags_reduced = prep_target_decoder.remove_zero_var_cols(
            self.y_var_lags)

        # Call the function to iteratively drop lags with high correlation for each feature
        self.y_var_lags_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(y_var_lags_reduced,
                                                                                       corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                                       verbose=verbose,
                                                                                       filter_by_feature=filter_corr_by_feature,
                                                                                       filter_by_subsets=filter_corr_by_subsets,
                                                                                       filter_by_all_columns=filter_corr_by_all_columns,
                                                                                       get_column_subsets_func=self.get_subset_key_words_and_all_column_subsets_for_corr)

        self.y_var_lags_reduced = drop_high_vif_vars.drop_columns_with_high_vif(self.y_var_lags_reduced_corr,
                                                                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                                                                vif_threshold=vif_threshold,
                                                                                verbose=verbose,
                                                                                filter_by_feature=filter_vif_by_feature,
                                                                                filter_by_subsets=filter_vif_by_subsets,
                                                                                filter_by_all_columns=filter_vif_by_all_columns,
                                                                                get_column_subsets_func=self.get_subset_key_words_and_all_column_subsets_for_vif)

        # sort y_var_lags_reduced by column name
        self.y_var_lags_reduced = self.y_var_lags_reduced.reindex(
            sorted(self.y_var_lags_reduced.columns), axis=1)

    def _reduce_x_var_lags(self,
                           corr_threshold_for_lags_of_a_feature=0.85,
                           vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                           filter_corr_by_feature=False,
                           filter_corr_by_all_columns=True,
                           filter_vif_by_feature=True,
                           filter_vif_by_all_columns=False,
                           ):

        self.x_var_lags_reduced = prep_target_decoder.remove_zero_var_cols(
            self.x_var_lags)

        # Call the function to iteratively drop lags with high correlation for each feature
        self.x_var_lags_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(self.x_var_lags_reduced,
                                                                                       corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                                       verbose=verbose,
                                                                                       filter_by_feature=filter_corr_by_feature,
                                                                                       filter_by_subsets=False,
                                                                                       filter_by_all_columns=filter_corr_by_all_columns)

        self.x_var_lags_reduced = drop_high_vif_vars.drop_columns_with_high_vif(self.x_var_lags_reduced_corr,
                                                                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                                                                vif_threshold=vif_threshold,
                                                                                verbose=verbose,
                                                                                filter_by_feature=filter_vif_by_feature,
                                                                                filter_by_subsets=False,
                                                                                filter_by_all_columns=filter_vif_by_all_columns)

        # sort x_var_lags_reduced by column name
        self.x_var_lags_reduced = self.x_var_lags_reduced.reindex(
            sorted(self.x_var_lags_reduced.columns), axis=1)

    def make_or_retrieve_y_var_lags_reduced(self, exists_ok=True, save_data=True, df_path=None):

        if exists_ok & os.path.exists(df_path):
            if os.path.exists(df_path):
                self.y_var_lags_reduced = pd.read_csv(df_path)
                if len(self.y_var_lags_reduced) == len(self.y_var_lags):
                    print(f'Loaded y_var_lags_reduced from {df_path}')
                    return
                else:
                    print(
                        'The number of rows in y_var_lags_reduced does not match that of y_var_lags. New y_var_lags_reduced will be made.')

        if not hasattr(self, 'y_var_lags'):
            self._get_y_var_lags(
                max_y_lag_number=self.max_y_lag_number, continuous_data=self.final_behavioral_data)
        self._reduce_y_var_lags()

        if save_data & (df_path is not None):
            self.y_var_lags_reduced.to_csv(df_path, index=False)
            print(f'Saved y_var_lags_reduced to {df_path}')
        else:
            print('Made new y_var_lags_reduced (but not saved)')

    def _get_x_and_y_var(self):
        self.x_var = self.binned_spikes_df.set_index(
            'bin').loc[self.valid_bin_index].reset_index(drop=False)
        self.y_var = self.final_behavioral_data.set_index(
            'bin').loc[self.valid_bin_index].reset_index(drop=False)

    def _get_y_var_lags(self, max_y_lag_number, continuous_data, trial_vector=None):
        self.max_y_lag_number = max_y_lag_number
        self.y_var_lags, self.lag_numbers = self._get_lags(
            max_y_lag_number, continuous_data, trial_vector=trial_vector)
        if 'bin_0' in self.y_var_lags.columns:
            self.y_var_lags['bin'] = self.y_var_lags['bin_0'].astype(int)
            self.y_var_lags = self.y_var_lags.drop(
                columns=[col for col in self.y_var_lags.columns if 'bin_' in col])

    def _get_x_var_lags(self, max_x_lag_number, continuous_data, trial_vector=None):
        self.max_x_lag_number = max_x_lag_number
        self.x_var_lags, self.x_lag_numbers = self._get_lags(
            max_x_lag_number, continuous_data, trial_vector=trial_vector)
        # drop all columns in x_var_lags that has bin_
        if 'bin_0' in self.x_var_lags.columns:
            self.x_var_lags['bin'] = self.x_var_lags['bin_0'].astype(int)
            self.x_var_lags = self.x_var_lags.drop(
                columns=[col for col in self.x_var_lags.columns if 'bin_' in col])

    def _get_lags(self, max_lag_number, continuous_data, trial_vector=None):
        lag_numbers = np.arange(-max_lag_number, max_lag_number+1)
        var_lags = neural_data_processing.add_lags_to_each_feature(
            continuous_data, lag_numbers, trial_vector=trial_vector)
        if hasattr(self, 'valid_bin_index'):
            var_lags = var_lags.set_index(
                'bin_0').loc[self.valid_bin_index].reset_index(drop=False)
        return var_lags, lag_numbers

    def _reduce_x_var(self):
        self.x_var_reduced = prep_target_decoder.remove_zero_var_cols(
            self.x_var)

    def _reduce_y_var(self,
                      df_path=None,
                      save_data=True,
                      corr_threshold_for_lags_of_a_feature=0.98,
                      vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                      filter_corr_by_all_columns=False,
                      filter_vif_by_subsets=True,
                      filter_vif_by_all_columns=True,
                      exists_ok=True,
                      ):

        print('===============================================')
        print('===============================================')
        print('Getting y_var_reduced...')

        if exists_ok & (df_path is not None):
            if os.path.exists(df_path):
                self.y_var_reduced = pd.read_csv(df_path)
                if len(self.y_var_reduced) == len(self.y_var):
                    print(f'Loaded y_var_reduced from {df_path}')
                    return
                else:
                    print(
                        'The number of rows in y_var_reduced does not match that of y_var. New y_var_reduced will be made.')

        # drop columns with std less than 0.001
        columns_w_small_std = self.y_var.std(
        )[self.y_var.std() < 0.001].index.tolist()

        self.y_var_reduced = self.y_var.drop(columns=columns_w_small_std)
        self._reduce_y_var_base(self.y_var_reduced,
                                filter_corr_by_all_columns=filter_corr_by_all_columns,
                                corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature,
                                vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                vif_threshold=vif_threshold,
                                verbose=verbose,
                                filter_vif_by_subsets=filter_vif_by_subsets,
                                filter_vif_by_all_columns=filter_vif_by_all_columns)

        if save_data & (df_path is not None):
            self.y_var_reduced.to_csv(df_path, index=False)
            print(f'Saved y_var_reduced to {df_path}')
        else:
            print('Made new y_var_reduced (but not saved)')

    def _reduce_y_var_lags(self,
                           df_path=None,
                           save_data=True,
                           corr_threshold_for_lags_of_a_feature=0.85,
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

        print('===============================================')
        print('===============================================')
        print('Getting y_var_lags_reduced...')

        if exists_ok & (df_path is not None):
            if os.path.exists(df_path):
                self.y_var_lags_reduced = pd.read_csv(df_path)
                if len(self.y_var_lags_reduced) == len(self.y_var_lags):
                    print(f'Loaded y_var_lags_reduced from {df_path}')
                    return
                else:
                    print(
                        'The number of rows in y_var_lags_reduced does not match that of y_var_lags. New y_var_lags_reduced will be made.')

        # If we get here, we need to recompute
        if verbose:
            print('Computing reduced y_var_lags...')

        # Call parent class method to do the actual reduction
        self._reduce_y_var_lags_base(
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
        if save_data & (df_path is not None):
            try:
                self.y_var_lags_reduced.to_csv(df_path, index=False)
                if verbose:
                    print(f'Saved reduced y_var_lags to {df_path}')
            except Exception as e:
                if verbose:
                    print(f'Warning: Failed to cache results: {str(e)}')
        else:
            print('Made new y_var_lags_reduced (but not saved)')
