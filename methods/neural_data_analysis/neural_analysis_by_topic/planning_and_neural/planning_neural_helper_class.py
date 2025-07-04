import sys
from data_wrangling import process_monkey_information, specific_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from null_behaviors import curvature_utils
from neural_data_analysis.neural_analysis_by_topic.planning_and_neural import planning_neural_utils
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
        self.planning_neural_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'planning_and_neural')
        os.makedirs(self.decoding_targets_folder_path, exist_ok=True)


    def prep_behav_data_to_analyze_planning(self,
                                            ref_point_mode='time after cur ff visible',
                                            ref_point_value=0.1,
                                            eliminate_outliers=False,
                                            use_curvature_to_ff_center=False,
                                            curv_of_traj_mode='distance',
                                            window_for_curv_of_traj=[-25, 25],
                                            truncate_curv_of_traj_by_time_of_capture=True,
                                            both_ff_across_time_df_exists_ok=True,
                                            test_or_control='test',
                                            ):
        self.test_or_control = test_or_control

        self.streamline_organizing_info(ref_point_mode=ref_point_mode,
                                        ref_point_value=ref_point_value,
                                        curv_of_traj_mode=curv_of_traj_mode,
                                        window_for_curv_of_traj=window_for_curv_of_traj,
                                        truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture,
                                        use_curvature_to_ff_center=use_curvature_to_ff_center,
                                        eliminate_outliers=eliminate_outliers,
                                        test_or_control=test_or_control
                                        )

        self.retrieve_neural_data()
        self.get_all_planning_info(
            both_ff_across_time_df_exists_ok=both_ff_across_time_df_exists_ok)

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

        os.makedirs(self.planning_neural_folder_path, exist_ok=True)
        df_path = os.path.join(self.planning_neural_folder_path, 'both_ff_across_time_df.csv')
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
                # if info_to_add is empty, then skip
                if info_to_add.empty:
                    continue
                info_to_add['segment'] = i
                info_to_add['target_index'] = row['cur_ff_index']
                self.both_ff_across_time_df = pd.concat(
                    [self.both_ff_across_time_df, info_to_add], axis=0)

            self._add_rel_x_and_y_to_both_ff_across_time_df()
            self._check_for_duplicate_bins()

            self.both_ff_across_time_df.reset_index(drop=False, inplace=True)
            self.both_ff_across_time_df.to_csv(df_path, index=False)
        return self.both_ff_across_time_df

    def _add_rel_x_and_y_to_both_ff_across_time_df(self):
        # Add relative x/y for cur_ff and nxt_ff
        if 'cur_ff_angle' in self.both_ff_across_time_df.columns and 'cur_ff_distance' in self.both_ff_across_time_df.columns:
            rel_x, rel_y = specific_utils.calculate_ff_rel_x_and_y(
                self.both_ff_across_time_df['cur_ff_distance'], self.both_ff_across_time_df['cur_ff_angle'])
            self.both_ff_across_time_df['cur_ff_rel_x'] = rel_x
            self.both_ff_across_time_df['cur_ff_rel_y'] = rel_y
        if 'nxt_ff_angle' in self.both_ff_across_time_df.columns and 'nxt_ff_distance' in self.both_ff_across_time_df.columns:
            rel_x, rel_y = specific_utils.calculate_ff_rel_x_and_y(
                self.both_ff_across_time_df['nxt_ff_distance'], self.both_ff_across_time_df['nxt_ff_angle'])
            self.both_ff_across_time_df['nxt_ff_rel_x'] = rel_x
            self.both_ff_across_time_df['nxt_ff_rel_y'] = rel_y

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
        self._find_ff_info(row, all_point_index)
        columns_to_keep = []
        for which_ff_info in ['nxt_', 'cur_']:
            info_to_add, columns_added = self._add_ff_info_to_info_to_add(
                info_to_add, row, which_ff_info)
            columns_to_keep.extend(columns_added)
        columns_to_keep.extend(['stop_point_index', 'point_index', 'bin'])
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
        # self.curv_df['point_index'] = self.curv_df.index
        info_to_add, columns_added = planning_neural_utils.add_curv_info(
            info_to_add, self.curv_df, which_ff_info)

        if which_ff_info == 'nxt_':
            # --- Merge firefly timing info ---
            ff_extra = ff_df[['point_index', 'time',
                              'stop_time']].drop_duplicates()
            ff_extra['time_rel_to_stop'] = ff_extra['time'] - \
                ff_extra['stop_time']
            info_to_add = info_to_add.merge(
                ff_extra[['point_index', 'time_rel_to_stop']],
                on='point_index', how='left'
            )
            columns_added.append('time_rel_to_stop')

            # --- Merge curvature info separately ---
            curv_extra = self.curv_df[['point_index',
                                       'curv_of_traj']].drop_duplicates()
            curv_extra.rename(
                columns={'curv_of_traj': 'traj_curv'}, inplace=True)
            info_to_add = info_to_add.merge(
                curv_extra,
                on='point_index', how='left'
            )
            columns_added.append('traj_curv')
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

    def get_both_ff_when_seen_df(self, crossing_ff=False, deal_with_rows_with_big_ff_angles=False):
        # This contains the planning-related information at specific time (such as when cur_ff was last visibl)
        # If crossing_ff is true, we'll get nxt_ff_info when cur_ff was first/last seen, and vice versa

        print('Making both_ff_when_seen_df...')
        self.both_ff_when_seen_df = self.nxt_ff_df[[
            'stop_point_index']].copy().set_index('stop_point_index')
        for first_or_last in ['first', 'last']:
            for when_which_ff, ff_df in [('when_nxt_ff', self.nxt_ff_df),
                                         ('when_cur_ff', self.cur_ff_df)]:
                all_point_index = ff_df[f'point_index_ff_{first_or_last}_seen'].values
                self._find_nxt_ff_df_2_and_cur_ff_df_2_based_on_specific_point_index(
                    all_point_index=all_point_index)
                if deal_with_rows_with_big_ff_angles:
                    self._deal_with_rows_with_big_ff_angles(
                        remove_i_o_modify_rows_with_big_ff_angles=True, delete_the_same_rows=True)

                for which_ff_info in ['nxt_', 'cur_']:
                    if (when_which_ff == 'when_cur_ff') & (first_or_last == 'first') & (which_ff_info == 'cur_'):
                        continue  # because the information is already contained in cur ff info at ref point

                    if not crossing_ff:
                        if (which_ff_info == 'nxt_') & (when_which_ff == 'when_cur_ff'):
                            continue
                        if (which_ff_info == 'cur_') & (when_which_ff == 'when_nxt_ff'):
                            continue
                    if deal_with_rows_with_big_ff_angles:
                        ff_df_modified = self.nxt_ff_df_modified if which_ff_info == 'nxt_' else self.cur_ff_df_modified
                    else:
                        ff_df_modified = self.nxt_ff_df2 if which_ff_info == 'nxt_' else self.cur_ff_df2

                    opt_arc_stop_first_vis_bdry = True if (
                        self.optimal_arc_type == 'opt_arc_stop_first_vis_bdry') else False

                    curv_df = curvature_utils.make_curvature_df(ff_df_modified, self.curv_of_traj_df, clean=False,
                                                                monkey_information=self.monkey_information,
                                                                ff_caught_T_new=self.ff_caught_T_new,
                                                                remove_invalid_rows=False,
                                                                opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)
                    if len(curv_df) != len(ff_df_modified):
                        raise ValueError(
                            'The length of curv_df is not the same as the length of ff_df_modified')
                    curv_df = pd.concat([ff_df_modified.drop(columns='point_index').reset_index(
                        drop=True), curv_df.reset_index(drop=True)], axis=1)
                    # for duplicated columns in curv_df, preserve only one
                    curv_df = curv_df.loc[:, ~curv_df.columns.duplicated()]
                    planning_neural_utils.add_to_both_ff_when_seen_df(
                        self.both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df)
        self.both_ff_when_seen_df.reset_index(drop=False, inplace=True)
        return self.both_ff_when_seen_df
