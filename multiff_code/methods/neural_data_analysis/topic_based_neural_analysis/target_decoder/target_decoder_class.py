import os

# Third-party imports
import numpy as np
import pandas as pd

# Scientific computing imports

# Local imports
from data_wrangling import general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import ml_decoder_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data
from neural_data_analysis.topic_based_neural_analysis.target_decoder import prep_target_decoder, behav_features_to_keep
from neural_data_analysis.neural_analysis_tools.model_neural_data import base_neural_class


class TargetDecoderClass(base_neural_class.NeuralBaseClass):

    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.05,
                 one_point_index_per_bin=False):

        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin
                         )
        self.max_visibility_window = 10

        self.target_decoder_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'target_decoder')
        self.gpfa_data_folder_path = self.target_decoder_folder_path
        os.makedirs(self.target_decoder_folder_path, exist_ok=True)

        # Initialize ML decoder
        self.ml_decoder = ml_decoder_class.MLBehavioralDecoder()

    def streamline_making_behav_and_neural_data(self, exists_ok=True):
        self.get_all_behav_data(exists_ok=exists_ok)
        self.max_bin = self.behav_data.bin.max()
        self.retrieve_neural_data()

    def get_all_behav_data(self, exists_ok=True):
        self.get_basic_data()
        self.get_behav_data(exists_ok=exists_ok)
        self._get_single_vis_target_df()
        self.get_pursuit_data()

    def get_basic_data(self):
        super().get_basic_data()
        self._get_curv_of_traj_df()
        self._make_or_retrieve_target_df(
            exists_ok=True,
            fill_na=False)
        self._make_or_retrieve_target_cluster_df(
            exists_ok=True,
            fill_na=False)

    def get_x_and_y_data_for_modeling(self, exists_ok=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self._reduce_x_var_lags()
        self.reduce_y_var_lags(exists_ok=exists_ok)

    def get_behav_data_by_point(self, exists_ok=True, save_data=True):
        behav_data_by_point_path = os.path.join(
            self.target_decoder_folder_path, "behav_data_by_point.csv"
        )
        # Load cached data if available and allowed
        if exists_ok and os.path.exists(behav_data_by_point_path):
            self.behav_data_by_point = pd.read_csv(behav_data_by_point_path)
            print(f"Loaded behav_data_by_bin from {behav_data_by_point_path}")
        else:
            if exists_ok:
                print(
                    f'Failed to load behav_data_by_point from {behav_data_by_point_path}. Will make new behav_data_by_point.')
            # Ensure basic data is available
            if not hasattr(self, 'monkey_information') or self.monkey_information is None:
                self.get_basic_data()
                basic_data_present = False
            else:
                basic_data_present = True

            self.behav_data_by_point = self.monkey_information.copy()
            # Post-processing
            self._add_or_drop_columns()
            self._add_all_target_info()
            self._add_curv_info()
            self._process_na()
            self._clip_values()

            # Free memory if basic data was not originally present
            if not basic_data_present:
                self._free_up_memory()

            # Optionally save processed data
            if save_data:
                self.behav_data_by_point.to_csv(
                    behav_data_by_point_path, index=False)
                print(
                    f"Saved behav_data_by_point to {behav_data_by_point_path}")

    def get_behav_data(self, exists_ok=True, save_data=True):

        self.get_behav_data_by_point(exists_ok=exists_ok, save_data=save_data)

        # Bin and process behavioral data
        _, self.time_bins = prep_monkey_data.initialize_binned_features(
            self.behav_data_by_point, self.bin_width
        )
        self.behav_data_by_bin = prep_monkey_data.bin_behav_data_by_point(
            self.behav_data_by_point,
            self.time_bins,
            one_point_index_per_bin=self.one_point_index_per_bin
        )

        if not hasattr(self, 'ff_caught_T_new'):
            self.load_raw_data()

        # if not have attribute ff_dataframe or if it's None, then retrieve it
        if getattr(self, 'ff_dataframe', None) is None:
            self.make_or_retrieve_ff_dataframe()

        self.behav_data_by_bin = prep_monkey_data.get_ff_info_for_bins(
            self.behav_data_by_bin, self.ff_dataframe, self.ff_caught_T_new, self.time_bins)

        self.behav_data_by_bin = self._add_ff_info(self.behav_data_by_bin)

        # Filter for relevant features
        self.behav_data = self.behav_data_by_bin[
            behav_features_to_keep.shared_columns_to_keep +
            behav_features_to_keep.extra_columns_for_concat_trials
        ]

    def get_pursuit_data(self):
        # Extract behavioral data for periods between target last visibility and capture
        pursuit_data_all = prep_target_decoder.make_pursuit_data_all(
            self.single_vis_target_df, self.behav_data_by_bin)

        # add the segment info back to single_vis_target_df
        self.single_vis_target_df['segment'] = np.arange(
            len(self.single_vis_target_df))
        self.single_vis_target_df = self.single_vis_target_df.merge(pursuit_data_all[[
                                                                    'segment', 'seg_start_time', 'seg_end_time', 'seg_duration']].drop_duplicates(), on='segment', how='left')

        # drop the segments with 0 duration from pursuit_data_all
        num_segments_with_0_duration = len(
            pursuit_data_all[pursuit_data_all['seg_duration'] == 0])
        print(f'{num_segments_with_0_duration} segments ({round(num_segments_with_0_duration/len(self.single_vis_target_df) * 100, 1)}%) out of {len(self.single_vis_target_df)} segments have 0 duration. They are dropped from pursuit data')

        # drop segments in pursuit data that has 0 duration
        pursuit_data_all = pursuit_data_all[pursuit_data_all['seg_duration'] > 0].copy(
        )

        seg_vars = ['segment', 'seg_duration', 'seg_start_time', 'seg_end_time',
                    'segment_start_dummy', 'segment_end_dummy']

        self.pursuit_data = pursuit_data_all[behav_features_to_keep.shared_columns_to_keep +
                                             behav_features_to_keep.extra_columns_for_concat_trials + seg_vars]

        self.pursuit_data_by_trial = pursuit_data_all[behav_features_to_keep.shared_columns_to_keep + seg_vars]

        # check for NA; if there is any, raise a warning
        na_rows, na_cols = general_utils.check_na_in_df(
            self.pursuit_data, 'pursuit_data')

    def reduce_y_var(self,
                     save_data=True,
                     corr_threshold_for_lags_of_a_feature=0.98,
                     vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                     filter_corr_by_all_columns=False,
                     filter_vif_by_subsets=True,
                     filter_vif_by_all_columns=True,
                     exists_ok=True,
                     ):
        df_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_y_var_reduced.csv')

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
        df_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_y_var_lags_reduced.csv')

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

    def _select_behav_features(self):
        self.behav_data = self.behav_data_by_bin[behav_features_to_keep.shared_columns_to_keep +
                                                 behav_features_to_keep.extra_columns_for_concat_trials]

        # Now, as a sanity check, see if the differences between behav_data and behav_data_by_bin are all contained in
        # behav_features_to_keep.behav_features_to_drop. If not, raise a warning
        diff_columns = set(self.behav_data_by_bin.columns) - \
            set(behav_features_to_keep.behav_features_to_drop)
        if diff_columns:
            print(
                f'The following columns are not accounted for in behav_features_to_keep: {diff_columns}.')

    def _add_or_drop_columns(self):
        self.behav_data_by_point = self.behav_data_by_point.drop(columns=[
                                                                 'stop_id'])

    def _free_up_memory(self):
        vars_deleted = []
        for var in ['ff_dataframe', 'monkey_information', 'target_df', 'curv_of_traj_df', 'curv_df']:
            if hasattr(self, var):
                vars_deleted.append(var)
                delattr(self, var)
        print(
            f'Deleted instance attributes {vars_deleted} to free up memory')

    def get_x_and_y_var(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):
        self._get_x_var(exists_ok=exists_ok)
        self._get_y_var(exists_ok=exists_ok)
        assert self.x_var['bin'].equals(self.y_var['bin'])

        self.get_x_and_y_var_lags(max_x_lag_number=max_x_lag_number,
                                  max_y_lag_number=max_y_lag_number,
                                  exists_ok=exists_ok)

    def get_x_and_y_var_lags(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True):

        x_var_lags_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_x_var_lags.csv')
        y_var_lags_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_y_var_lags.csv')

        if exists_ok & os.path.exists(x_var_lags_path) & os.path.exists(y_var_lags_path):
            self.x_var_lags = pd.read_csv(x_var_lags_path)
            self.y_var_lags = pd.read_csv(y_var_lags_path)
            if (self.x_var_lags['bin'].equals(self.x_var['bin'])) & (self.y_var_lags['bin'].equals(self.y_var['bin'])):
                print(
                    f'Loaded x_var_lags and y_var_lags from {x_var_lags_path} and {y_var_lags_path}')
                return
            else:
                print('The values in column "bin" in x_var_lags or y_var_lags does not match those of x_var or y_var. New x_var_lags and y_var_lags will be made.')

        self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                             continuous_data=self.binned_spikes_df)

        self._get_y_var_lags_with_target_info(
            max_y_lag_number=max_y_lag_number)

        self.x_var_lags = self.pursuit_data[['segment', 'bin']].merge(
            self.x_var_lags, on=['bin'], how='left').reset_index(drop=True)
        self.y_var_lags = self.pursuit_data[['segment', 'bin']].merge(
            self.y_var_lags, on=['bin'], how='left').reset_index(drop=True)

        self.x_var_lags.to_csv(x_var_lags_path, index=False)
        self.y_var_lags.to_csv(y_var_lags_path, index=False)
        print(
            f'Saved x_var_lags and y_var_lags to {x_var_lags_path} and {y_var_lags_path}')

        assert self.x_var_lags['bin'].equals(self.x_var['bin'])
        assert self.y_var_lags['bin'].equals(self.y_var['bin'])

    def _get_y_var_lags_with_target_info(self, max_y_lag_number=5):
      # we'll drop columns on target for now because we'll make them separately
        self.target_columns = [
            col for col in self.y_var.columns if 'target' in col]
        # make y_columns_to_drop to be the set of self.y_columns_to_drop and self.target_columns
        continuous_data = self.behav_data.drop(
            columns=self.target_columns)
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=continuous_data)

        self.y_var_lags = self.y_var_lags[self.y_var_lags['bin'].isin(
            self.y_var['bin'].values)]

        self._add_target_info_to_y_var_lags()

    def _add_target_info_to_y_var_lags(self):

        basic_data_present = hasattr(self, 'monkey_information')
        if not basic_data_present:
            self.get_basic_data()
            self._get_curv_of_traj_df()

        # first get info for pairs of target_index and point_index that the lagged columns will use
        target_df_lags = prep_target_decoder.initialize_target_df_lags(
            self.y_var, self.max_y_lag_number, self.bin_width)
        target_df_lags = prep_target_decoder.add_target_info_based_on_target_index_and_point_index(target_df_lags, self.monkey_information, self.ff_real_position_sorted,
                                                                                                   self.ff_dataframe, self.ff_caught_T_new, self.curv_of_traj_df)
        target_df_lags = prep_target_decoder.fill_na_in_last_seen_columns(
            target_df_lags)

        # Now, put the lagged target columns into y_var_lags
        self.y_var_lags = prep_target_decoder.add_lagged_target_columns(
            self.y_var_lags, self.y_var, target_df_lags, self.max_y_lag_number, target_columns=self.target_columns)

        if not basic_data_present:
            # free up memory if basic data is not present before calling the function
            self._free_up_memory()

    def _get_x_var(self, exists_ok=True):
        x_var_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_x_var.csv')

        # Try to load x_var from file if allowed
        if exists_ok and os.path.exists(x_var_path):
            self.x_var = pd.read_csv(x_var_path)
            print(f'Loaded x_var from {x_var_path}')

            # Check compatibility with pursuit_data
            if len(self.x_var) == len(self.pursuit_data):
                self._reduce_x_var()
                return
            else:
                print(f'Warning: Length mismatch. Retrieved x_var has {len(self.x_var)} rows, '
                      f'but pursuit_data has {len(self.pursuit_data)}. New x_var will be generated.')

        # Ensure behavioral and neural data are loaded
        if not hasattr(self, 'binned_spikes_df'):
            self.get_all_behav_data()
            self.max_bin = self.behav_data.bin.max()
            self.retrieve_neural_data()
            # self._free_up_memory()

        # Extract subset of neural data aligned with pursuit_data
        self.x_var = self.pursuit_data[['segment', 'bin']].merge(
            self.binned_spikes_df, on=['bin'], how='left').reset_index(drop=True)

        # Create and save new x_var
        self.x_var.to_csv(x_var_path, index=False)
        print(f'Saved x_var to {x_var_path}')

        self._reduce_x_var()

    def _get_y_var(self, exists_ok=True):
        # note that this is for the continuous case (a.k.a. all selected time points are used together, instead of being separated into trials)
        y_var_path = os.path.join(
            self.target_decoder_folder_path, 'target_decoder_y_var.csv')
        if exists_ok and os.path.exists(y_var_path):
            self.y_var = pd.read_csv(y_var_path)
            print(f'Loaded y_var from {y_var_path}')
            # check if the length of y_var is the same as the length of pursuit_data
            if len(self.y_var) != len(self.pursuit_data):
                print(f'Warning: The length of retrieved y_var ({len(self.y_var)}) is not the same as the length of pursuit_data ({len(self.pursuit_data)}).'
                      'New y_var will be made.')
            else:
                self.reduce_y_var(exists_ok=exists_ok)
                return

        self.y_var = self.pursuit_data.reset_index(drop=True)
        # Convert bool columns to int
        bool_columns = self.y_var.select_dtypes(include=['bool']).columns
        self.y_var[bool_columns] = self.y_var[bool_columns].astype(int)
        self.y_var.to_csv(y_var_path, index=False)
        print(f'Saved y_var to {y_var_path}')

        self.reduce_y_var(exists_ok=exists_ok)

    def _process_na(self):
        na_rows, na_cols = prep_target_decoder._process_na(
            self.behav_data_by_point)

    def _clip_values(self):
        # clip values in some columns
        for column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
            self.behav_data_by_point.loc[:, column] = np.clip(
                self.behav_data_by_point.loc[:, column], -1000, 1000)

    def _add_curv_info(self):
        self.behav_data_by_point = prep_target_decoder._add_curv_info_to_behav_data_by_point(
            self.behav_data_by_point, self.curv_of_traj_df, self.monkey_information, self.ff_caught_T_new)

    def _add_all_target_info(self):
        self.behav_data_by_point = prep_target_decoder.add_target_info_to_behav_data_by_point(
            self.behav_data_by_point, self.target_df)
        self.behav_data_by_point = prep_target_decoder.add_target_info_to_behav_data_by_point(
            self.behav_data_by_point, self.target_cluster_df)

    def _get_single_vis_target_df(self, single_vis_target_df_exists_ok=True, target_clust_last_vis_df_exists_ok=True):

        df_path = os.path.join(
            self.target_decoder_folder_path, 'single_vis_target_df.csv')
        if single_vis_target_df_exists_ok and os.path.exists(df_path):
            try:
                self.single_vis_target_df = pd.read_csv(df_path)
                print(f'Loaded single_vis_target_df from {df_path}')
            except (pd.errors.EmptyDataError, ValueError) as e:
                print(f'Failed to load single_vis_target_df: {str(e)}')
        else:
            self.make_or_retrieve_target_clust_last_vis_df(
                exists_ok=target_clust_last_vis_df_exists_ok)
            # in the function, we'll drop the rows where target is in a cluster, because we want to preserve cases where monkey is going toward a single target, not a cluster
            self.single_vis_target_df = prep_target_decoder.find_single_vis_target_df(
                self.target_clust_last_vis_df, self.monkey_information, self.ff_caught_T_new, max_visibility_window=self.max_visibility_window)
            self.single_vis_target_df.to_csv(df_path, index=False)
            print(f'Saved single_vis_target_df to {df_path}')

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_corr(y_var_lags):
        subset_key_words, all_column_subsets = prep_target_decoder._get_subset_key_words_and_all_column_subsets_for_corr(
            y_var_lags)
        return subset_key_words, all_column_subsets

    @staticmethod
    def get_subset_key_words_and_all_column_subsets_for_vif(y_var_lags):
        subset_key_words, all_column_subsets = prep_target_decoder._get_subset_key_words_and_all_column_subsets_for_vif(
            y_var_lags)
        return subset_key_words, all_column_subsets
