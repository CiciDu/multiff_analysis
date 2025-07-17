import sys
from data_wrangling import general_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_cur_vs_nxt_ff_data import find_cvn_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.target_decoder import prep_target_decoder, behav_features_to_keep, target_decoder_class
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import pn_utils, pn_helper_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, gpfa_regression_utils, plot_gpfa_utils, gpfa_helper_class
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
import numpy as np
import pandas as pd
import os


class PlanningAndNeural(base_neural_class.NeuralBaseClass):

    def __init__(self, raw_data_folder_path=None,
                 bin_width=0.1,
                 one_point_index_per_bin=False):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin)
        self.planning_neural_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'planning_and_neural')
        self.gpfa_data_folder_path = self.planning_neural_folder_path
        os.makedirs(self.planning_neural_folder_path, exist_ok=True)

    def prep_data_to_analyze_planning(self,
                                      ref_point_mode='time after cur ff visible',
                                      ref_point_value=0.1,
                                      eliminate_outliers=False,
                                      use_curv_to_ff_center=False,
                                      curv_of_traj_mode='distance',
                                      window_for_curv_of_traj=[-25, 25],
                                      truncate_curv_of_traj_by_time_of_capture=True,
                                      planning_data_by_point_exists_ok=True,
                                      ):

        # self.get_basic_data()
        self.retrieve_neural_data()

        data_kwargs1 = {'raw_data_folder_path': self.raw_data_folder_path,
                        'one_point_index_per_bin': self.one_point_index_per_bin}

        data_kwargs2 = {'ref_point_mode': ref_point_mode,
                        'ref_point_value': ref_point_value,
                        'curv_of_traj_mode': curv_of_traj_mode,
                        'window_for_curv_of_traj': window_for_curv_of_traj,
                        'truncate_curv_of_traj_by_time_of_capture': truncate_curv_of_traj_by_time_of_capture,
                        'use_curv_to_ff_center': use_curv_to_ff_center,
                        'eliminate_outliers': eliminate_outliers,
                        'planning_data_by_point_exists_ok': planning_data_by_point_exists_ok,
                        }

        # get behavioral_data
        # if test_or_control == 'test':
        self.test_data_inst = pn_helper_class.PlanningAndNeuralHelper(test_or_control='test',
                                                                      **data_kwargs1)
        self.test_data_inst.prep_behav_data_to_analyze_planning(**data_kwargs2)

        self.ctrl_data_inst = pn_helper_class.PlanningAndNeuralHelper(test_or_control='control',
                                                                      **data_kwargs1)
        self.ctrl_data_inst.prep_behav_data_to_analyze_planning(**data_kwargs2)

        test_data, ctr_data = pn_utils.compute_overlap_and_drop(self.test_data_inst.planning_data_by_point, 'point_index',
                                                                self.ctrl_data_inst.planning_data_by_point, 'point_index')
        test_data['whether_test'] = 1
        ctr_data['whether_test'] = 0

        self.planning_data_by_point = pd.concat(
            [test_data, ctr_data], ignore_index=True).sort_values(by=['point_index'])
        # reassign segment based on current order
        self.planning_data_by_point['segment'] = self.planning_data_by_point.groupby(
            'target_index').ngroup()
        self.planning_data_by_point.reset_index(drop=True, inplace=True)
        self.fill_na_in_cur_and_nxt_opt_arc_dheading()

        # del self.test_plan_data_inst
        # del self.ctr_plan_data_inst

        # put data in bins
        self.make_or_retrieve_monkey_information()
        self.get_bin_info()
        self.get_planning_data_by_bin()
        # self._add_data_from_behav_data_all(exists_ok=True)

    def get_bin_info(self):
        self.bin_info = prep_monkey_data.bin_monkey_information(
            self.monkey_information[['point_index', 'time']
                                    ], self.time_bins, add_stop_time_ratio_in_bin=False,
            one_point_index_per_bin=self.one_point_index_per_bin)
        # check for duplicated point_index. If there are, raise an error
        if self.bin_info['point_index'].duplicated().any():
            print(f'There are {self.bin_info["point_index"].duplicated().sum()} duplicated point_index in bin_info. '
                  f'Note: one_point_index_per_bin is {self.one_point_index_per_bin}')

    def get_planning_data_by_bin(self):
        self.planning_data_by_bin = self.bin_info[['point_index', 'bin']].merge(
            self.planning_data_by_point, on='point_index', how='right')
        # drop rows with na in bin
        self.planning_data_by_bin = self.planning_data_by_bin.dropna(subset=[
            'bin'])
        self._check_for_duplicate_bins()

    def fill_na_in_cur_and_nxt_opt_arc_dheading(self):
        df = self.planning_data_by_point

        if 'cur_opt_arc_dheading' in df.columns and 'nxt_opt_arc_dheading' in df.columns:
            # Fill missing cur_opt_arc_dheading based on cur_ff_angle sign
            mask_cur_na = df['cur_opt_arc_dheading'].isna()
            cur_angles = df.loc[mask_cur_na, 'cur_ff_angle']
            df.loc[mask_cur_na, 'cur_opt_arc_dheading'] = np.where(
                cur_angles > 0,  np.pi / 2,      # Positive angle → +π/2
                # Negative → -π/2, Zero → 0
                np.where(cur_angles < 0, -np.pi / 2, 0)
            )

            # Fill missing nxt_opt_arc_dheading based on nxt_ff_angle sign
            mask_nxt_na = df['nxt_opt_arc_dheading'].isna()
            nxt_angles = df.loc[mask_nxt_na, 'nxt_ff_angle']
            df.loc[mask_nxt_na, 'nxt_opt_arc_dheading'] = np.where(
                nxt_angles > 0,  np.pi / 2,       # Positive angle → +π/2
                # Negative → -π/2, Zero → 0
                np.where(nxt_angles < 0, -np.pi / 2, 0)
            )

    def _check_for_duplicate_bins(self):
        dup_rows = self.planning_data_by_bin['bin'].duplicated()
        if dup_rows.any():
            print(
                f'There are {dup_rows.sum()} duplicated bin in planning_data_by_bin. Retaining the rows with the smaller stop_point_index.')
            # retain the rows with the smaller stop_point_index
            self.planning_data_by_bin = self.planning_data_by_bin.sort_values(
                by=['bin', 'stop_point_index'], ascending=True).drop_duplicates(subset='bin', keep='first')

    def _add_data_from_behav_data_all(self, exists_ok=True):
        # Merge in columns from self.behav_data_by_bin that are not already present
        # (Assume pn is available in the current scope, or pass as argument if needed)
        try:
            self.get_behav_data(exists_ok=exists_ok)
            missing_cols = list(set(self.behav_data_by_bin.columns) -
                                set(self.planning_data_by_bin.columns))
            if 'point_index' in self.planning_data_by_bin.columns and 'point_index' in self.behav_data_by_bin.columns:
                # Check for missing point_index
                missing_point_indices = set(
                    self.planning_data_by_bin['point_index']) - set(self.behav_data_by_bin['point_index'])
                if missing_point_indices:
                    import warnings
                    warnings.warn(
                        f"There are {len(missing_point_indices)} point_index values in planning_data_by_bin not present in self.behav_data_by_bin. Example: {list(missing_point_indices)[:5]}."
                        "One possible reason is different bin width between planning_data_by_bin and behav_data_by_bin.")
                # Merge only the missing columns
                self.planning_data_by_bin = self.planning_data_by_bin.merge(
                    self.behav_data_by_bin[[
                        'point_index', 'bin'] + missing_cols],
                    on=['point_index', 'bin'], how='left')
            else:
                raise ValueError(
                    "point_index is not in planning_data_by_bin or behav_data_by_bin")
        except Exception as e:
            print(
                f"[WARN] Could not merge extra columns from self.behav_data_by_bin: {e}."
                "Will not merge extra columns from self.behav_data_by_bin.")

        # convert all bool to int
        bool_cols = self.planning_data_by_bin.select_dtypes(
            include='bool').columns
        self.planning_data_by_bin[bool_cols] = self.planning_data_by_bin[bool_cols].astype(
            int)

    def get_x_and_y_data_for_modeling(self, max_x_lag_number=5, max_y_lag_number=5, exists_ok=True, reduce_y_var_lags=True):
        self.get_x_and_y_var(exists_ok=exists_ok)
        self.get_x_and_y_var_lags(
            max_x_lag_number=max_x_lag_number, max_y_lag_number=max_y_lag_number)
        self._reduce_x_var_lags(
            filter_corr_by_all_columns=False, filter_vif_by_feature=False)
        self.separate_test_and_control_from_x_and_y_var()
        if reduce_y_var_lags:
            self.reduce_y_var_lags(exists_ok=exists_ok)
            self.separate_test_and_control_from_y_var_lags_reduced()

        print('x_var.shape:', self.x_var.shape)
        print('y_var.shape:', self.y_var.shape)
        print('x_var_reduced.shape:', self.x_var_reduced.shape)
        print('y_var_reduced.shape:', self.y_var_reduced.shape)
        print('========================================')
        print('x_var_lags.shape:', self.x_var_lags.shape)
        print('y_var_lags.shape:', self.y_var_lags.shape)
        if reduce_y_var_lags:
            print('x_var_lags_reduced.shape:', self.x_var_lags_reduced.shape)
            print('y_var_lags_reduced.shape:', self.y_var_lags_reduced.shape)

    def separate_test_and_control_from_x_and_y_var(self):
        self.test_mask = self.y_var['whether_test'] == 1
        self.test_y_var = self.y_var[self.test_mask]
        self.test_y_var_reduced = self.y_var_reduced[self.test_mask]
        self.test_x_var_lags_reduced = self.x_var_lags_reduced[self.test_mask]

        self.control_mask = self.y_var['whether_test'] == 0
        self.control_y_var = self.y_var[self.control_mask]
        self.control_y_var_reduced = self.y_var_reduced[self.control_mask]
        self.control_x_var_lags_reduced = self.x_var_lags_reduced[self.control_mask]

    def separate_test_and_control_from_y_var_lags_reduced(self):
        self.test_y_var_lags_reduced = self.y_var_lags_reduced[self.test_mask]
        self.control_y_var_lags_reduced = self.y_var_lags_reduced[self.control_mask]

    def get_x_and_y_var(self, exists_ok=True):
        original_len = len(self.planning_data_by_bin)
        self.y_var = self.planning_data_by_bin.dropna().drop(
            columns={'stop_point_index', 'point_index'}, errors='ignore').reset_index(drop=True)
        self.y_var['bin'] = self.y_var['bin'].astype(int)
        print(f"{round(1 - len(self.y_var) / original_len, 2)}% of rows are dropped in planning_data_by_bin due to having missing values")

        self.x_var = self.y_var[['segment', 'bin']].merge(
            self.binned_spikes_df, on=['bin'], how='left').reset_index(drop=True)
        print('binned_spikes_df.shape:', self.binned_spikes_df.shape)
        print('self.x_var.shape:', self.x_var.shape)
        print('self.y_var.shape:', self.y_var.shape)

        assert self.x_var['bin'].equals(self.y_var['bin'])

        self._reduce_x_var()
        self.reduce_y_var(exists_ok=exists_ok)

    def get_x_and_y_var_lags(self, max_x_lag_number=5, max_y_lag_number=5):

        trial_vector = self.y_var['segment'].values
        self._get_x_var_lags(max_x_lag_number=max_x_lag_number,
                             continuous_data=self.x_var, trial_vector=trial_vector)
        self._get_y_var_lags(max_y_lag_number=max_y_lag_number,
                             continuous_data=self.y_var, trial_vector=trial_vector)

        assert self.x_var_lags['bin'].equals(self.x_var['bin'])
        assert self.y_var_lags['bin'].equals(self.y_var['bin'])

        self.x_var_lags['segment'] = trial_vector
        self.y_var_lags['segment'] = trial_vector

    # ================================================

    def reduce_y_var(self,
                     save_data=True,
                     corr_threshold_for_lags_of_a_feature=0.97,
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

    def get_x_and_y_var_for_lr(self, test_or_control='both', use_x_var_lags=True):
        if use_x_var_lags:
            if test_or_control == 'test':
                x_var = self.test_x_var_lags_reduced.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.test_y_var
            elif test_or_control == 'control':
                x_var = self.control_x_var_lags_reduced.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.control_y_var
            elif test_or_control == 'both':
                x_var = self.x_var_lags_reduced.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.y_var
            else:
                raise ValueError(
                    f'test_or_control must be "test", "control", or "both". Got {test_or_control}')
        else:
            if test_or_control == 'test':
                x_var = self.test_x_var.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.test_y_var
            elif test_or_control == 'control':
                x_var = self.control_x_var.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.control_y_var
            elif test_or_control == 'both':
                x_var = self.x_var.drop(
                    columns=['segment', 'bin'], errors='ignore')
                y_var = self.y_var
            else:
                raise ValueError(
                    f'test_or_control must be "test", "control", or "both". Got {test_or_control}')
        return x_var, y_var
