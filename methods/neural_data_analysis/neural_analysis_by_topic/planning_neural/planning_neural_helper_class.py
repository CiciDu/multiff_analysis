import sys
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from null_behaviors import curvature_utils
from neural_data_analysis.planning_and_neural import planning_neural_utils
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
import numpy as np
import pandas as pd
import os


class PlanningAndNeuralHelper(plan_factors_class.PlanFactors):

    def __init__(self, raw_data_folder_path=None, bin_width=0.02, window_width=0.25,
                 one_behav_idx_per_bin=True):
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.bin_width = bin_width
        self.window_width = window_width
        self.one_behav_idx_per_bin = one_behav_idx_per_bin
        self.max_bin = None

        self.decoding_targets_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'decoding_targets')
        os.makedirs(self.decoding_targets_folder_path, exist_ok=True)

    def prep_behav_data_to_analyze_planning(self,
                                            ref_point_mode='time after cur ff visible',
                                            ref_point_value=0.1,
                                            eliminate_outliers=False,
                                            use_curvature_to_ff_center=False,
                                            curv_of_traj_mode='distance',
                                            window_for_curv_of_traj=[-25, 25],
                                            truncate_curv_of_traj_by_time_of_capture=True):

        self.streamline_organizing_info(ref_point_mode=ref_point_mode,
                                        ref_point_value=ref_point_value,
                                        curv_of_traj_mode=curv_of_traj_mode,
                                        window_for_curv_of_traj=window_for_curv_of_traj,
                                        truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
                                        use_curvature_to_ff_center=use_curvature_to_ff_center,
                                        eliminate_outliers=eliminate_outliers)

        self.retrieve_neural_data()
        self.get_all_planning_info()

    def retrieve_neural_data(self):
        base_neural_class.NeuralBaseClass.retrieve_neural_data(
            self)

    def get_all_planning_info(self, both_ff_across_time_df_exists_ok=True):

        self.monkey_info_in_bins = prep_monkey_data.bin_monkey_information(
            self.monkey_information, self.time_bins, one_behav_idx_per_bin=self.one_behav_idx_per_bin)

        if not self.one_behav_idx_per_bin:
            self.bin_info = self.monkey_info_in_bins[['bin', 'point_index']].groupby(
                'bin').median().astype(int).reset_index(drop=False)
            self.bin_info = self.bin_info.merge(self.monkey_info_in_bins.drop(
                columns={'bin'}), on='point_index', how='left')
        else:
            self.bin_info = self.monkey_info_in_bins

        self.both_ff_across_time_df = self._get_both_ff_across_time_df(
            exists_ok=both_ff_across_time_df_exists_ok)

        # self.both_ff_when_seen_df = self.get_both_ff_when_seen_df()
        # self.all_planning_info = self.both_ff_across_time_df.merge(self.both_ff_when_seen_df, on='stop_point_index', how='left')
        self.all_planning_info = self.both_ff_across_time_df.copy()

        heading_columns_to_add = ['stop_point_index', 'angle_from_m_before_stop_to_cur_ff', 'angle_from_m_before_stop_to_nxt_ff',
                                  'angle_from_cur_ff_landing_to_nxt_ff']
        self.all_planning_info = self.all_planning_info.merge(
            self.heading_info_df[heading_columns_to_add], on='stop_point_index', how='left')

        return self.all_planning_info

    def _get_both_ff_across_time_df(self, exists_ok=True):
        # This contains the planning-related information for each time bin
        folder_path = os.path.join(
            self.planning_data_folder_path, 'planning_for_neural')
        os.makedirs(folder_path, exist_ok=True)
        df_path = os.path.join(folder_path, 'both_ff_across_time_df.csv')
        if exists_ok and os.path.exists(df_path):
            self.both_ff_across_time_df = pd.read_csv(df_path)
        else:
            self.both_ff_across_time_df = pd.DataFrame([])
            self._get_point_index_based_on_some_time_before_stop(
                n_seconds_before_stop=2)
            for i, row in self.stops_near_ff_df.iterrows():
                if i % 10 == 0:
                    print(
                        f'Having processed {i} rows out of {len(self.stops_near_ff_df)} of the stops_near_ff_df for both_ff_across_time_df.')
                info_to_add = self._get_info_to_add(row)
                self.both_ff_across_time_df = pd.concat(
                    [self.both_ff_across_time_df, info_to_add], axis=0)
            self._check_for_duplicate_bins()
            self.both_ff_across_time_df.reset_index(drop=False, inplace=True)
            self.both_ff_across_time_df.to_csv(df_path, index=False)
        return self.both_ff_across_time_df

    def _get_point_index_based_on_some_time_before_stop(self, n_seconds_before_stop=2):
        self.stops_near_ff_df['some_time_before_stop'] = self.stops_near_ff_df['stop_time'] - \
            n_seconds_before_stop
        self.stops_near_ff_df['point_index_in_the_past'] = np.searchsorted(
            self.monkey_information['time'].values, self.stops_near_ff_df['some_time_before_stop'].values)

    def _get_info_to_add(self, row):
        info_to_add = self.bin_info[self.bin_info['point_index'].between(
            row['point_index_in_the_past'], row['next_stop_point_index'])].copy()
        info_to_add['stop_point_index'] = row['stop_point_index']
        all_point_index = info_to_add['point_index'].values
        info_to_add.set_index('point_index', inplace=True)
        self._find_ff_info(row, all_point_index)
        columns_to_keep = []
        for which_ff_info in ['nxt_', 'cur_']:
            info_to_add, columns_added = self._add_ff_info_to_info_to_add(
                info_to_add, row, which_ff_info)
            columns_to_keep.extend(columns_added)
        columns_to_keep.extend(['stop_point_index', 'bin'])
        info_to_add = info_to_add[columns_to_keep]
        return info_to_add

    def _check_for_duplicate_bins(self):
        if self.both_ff_across_time_df['bin'].duplicated().any():
            # retain the rows with the bigger stop_point_index
            self.both_ff_across_time_df = self.both_ff_across_time_df.sort_values(
                by=['stop_point_index', 'bin'], ascending=True).drop_duplicates(subset='bin', keep='last')

    def _find_ff_info(self, row, all_point_index):
        self.nxt_ff_df2 = find_stops_near_ff_utils.find_ff_info(np.repeat(row.nxt_ff_index, len(
            all_point_index)), all_point_index, self.monkey_information, self.ff_real_position_sorted)
        self.cur_ff_df2 = find_stops_near_ff_utils.find_ff_info(np.repeat(row.cur_ff_index, len(
            all_point_index)), all_point_index, self.monkey_information, self.ff_real_position_sorted)
        # self._deal_with_rows_with_big_ff_angles(remove_i_o_modify_rows_with_big_ff_angles=True, delete_the_same_rows=True)

    def _add_ff_info_to_info_to_add(self, info_to_add, row, which_ff_info):
        ff_df = self._get_ff_df_and_add_time_info(row, which_ff_info)
        ff_df = ff_df[ff_df['ff_angle'].between(-np.pi/4, np.pi/4)].copy()

        self.curv_df = curvature_utils.make_curvature_df(ff_df, self.curv_of_traj_df, clean=False,
                                                         remove_invalid_rows=True,
                                                         monkey_information=self.monkey_information,
                                                         ff_caught_T_new=self.ff_caught_T_new)
        self.curv_df.set_index('point_index', inplace=True)
        # self.curv_df['point_index'] = self.curv_df.index
        info_to_add, columns_added = planning_neural_utils.add_curv_info_to_info_to_add(
            info_to_add, self.curv_df, which_ff_info)
        if which_ff_info == 'nxt_':  # because we only have to do it once, we choose one kind of which_ff_info to do it
            ff_df.set_index('point_index', inplace=True)
            info_to_add['time_rel_to_stop'] = ff_df['time'] - \
                ff_df['stop_time']
            info_to_add['traj_curv'] = self.curv_df['curv_of_traj']
            columns_added.extend(['time_rel_to_stop', 'traj_curv'])
        return info_to_add, columns_added

    def _get_ff_df_and_add_time_info(self, row, which_ff_info):
        ff_df = self.nxt_ff_df2 if which_ff_info == 'nxt_' else self.cur_ff_df2
        ff_df['time'] = self.monkey_information.loc[ff_df['point_index'].values, 'time'].values
        ff_df['stop_point_index'] = row['stop_point_index']
        ff_df['stop_time'] = self.monkey_information.loc[row['stop_point_index'], 'time']
        return ff_df

    # def _add_to_both_ff_when_seen_df(self, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df):
    #     curv_df.set_index('stop_point_index', inplace=True)
    #     self.both_ff_when_seen_df[f'{which_ff_info}ff_angle_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_angle']
    #     self.both_ff_when_seen_df[f'{which_ff_info}ff_distance_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_distance']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_to_ff_center']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['optimal_curvature']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['optimal_arc_d_heading']
    #     self.both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    #     self.both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_of_traj']
