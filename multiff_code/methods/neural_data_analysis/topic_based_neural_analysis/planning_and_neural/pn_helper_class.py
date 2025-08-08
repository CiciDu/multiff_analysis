import sys
from data_wrangling import process_monkey_information, specific_utils, general_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from null_behaviors import curvature_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars, base_neural_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, test_vs_control_utils
from planning_analysis.show_planning import show_planning_utils
import numpy as np
import pandas as pd
import os
import sys


class PlanningAndNeuralHelper(plan_factors_class.PlanFactors):

    def __init__(self, test_or_control='test',
                 raw_data_folder_path=None,
                 one_point_index_per_bin=True):
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.test_or_control = test_or_control
        self.one_point_index_per_bin = one_point_index_per_bin
        self.max_bin = None
        os.makedirs(self.target_decoder_folder_path, exist_ok=True)

    def prep_behav_data_to_analyze_planning(self,
                                            ref_point_mode='time after cur ff visible',
                                            ref_point_value=0.1,
                                            eliminate_outliers=False,
                                            use_curv_to_ff_center=False,
                                            curv_of_traj_mode='distance',
                                            window_for_curv_of_traj=[-25, 0],
                                            curv_traj_window_before_stop=[
                                                -50, 0],
                                            truncate_curv_of_traj_by_time_of_capture=True,
                                            planning_data_by_point_exists_ok=True,
                                            ):

        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.truncate_curv_of_traj_by_time_of_capture = truncate_curv_of_traj_by_time_of_capture
        self.use_curv_to_ff_center = use_curv_to_ff_center
        self.eliminate_outliers = eliminate_outliers
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        self.get_planning_data_by_point(exists_ok=planning_data_by_point_exists_ok,
                                        )

    def get_planning_data_by_point(self, exists_ok=True):
        folder_name = os.path.join(self.planning_and_neural_folder_path,
                                   self.planning_data_by_point_partial_path, self.test_or_control)

        df_name = find_cvn_utils.find_diff_in_curv_df_name(ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                           curv_traj_window_before_stop=self.curv_traj_window_before_stop)

        os.makedirs(folder_name, exist_ok=True)

        df_path = os.path.join(
            folder_name, df_name)
        if exists_ok and os.path.exists(df_path):
            self.planning_data_by_point = pd.read_csv(df_path)
            print(f'Loaded planning_data_by_point from {df_path}')
        else:
            self.planning_data_by_point = self._get_planning_data_by_point()
            self.planning_data_by_point.to_csv(df_path, index=False)
            print(f'Made new planning_data_by_point and saved to {df_path}')

        # Add absolute value columns for relevant features
        relevant_cols = [col for col in self.planning_data_by_point.columns
                         if 'angle' in col or 'rel_x' in col]
        for col in relevant_cols:
            self.planning_data_by_point[f'abs_{col}'] = self.planning_data_by_point[col].abs(
            )

        # self.planning_data_by_point['']
        return self.planning_data_by_point

    def _get_planning_data_by_point(self):

        self.load_raw_data()
        self._streamline_getting_cur_vs_nxt_ff_data()

        # need to make new heading_info separately to ensure that all stop_point_index are included
        self.make_temporary_heading_info_df()

        self.both_ff_across_time_df = self.get_both_ff_across_time_df()
        self.planning_data_by_point = self.both_ff_across_time_df
        self.add_heading_info_to_planning_data_by_point()

        # Ensure that each segment maps to exactly one target_index
        n_unique_targets_per_segment = self.planning_data_by_point.groupby('segment')[
            'target_index'].nunique()
        assert (n_unique_targets_per_segment == 1).all(
        ), "Each segment must map to exactly one target_index"

        return self.planning_data_by_point

    def _only_make_stops_near_ff_df(self):
        self.load_raw_data()
        self.make_stops_near_ff_and_ff_comparison_dfs(
            test_or_control=self.test_or_control, exists_ok=True, save_data=True)

    def _streamline_getting_cur_vs_nxt_ff_data(self):
        self.streamline_organizing_info(ref_point_mode=self.ref_point_mode,
                                        ref_point_value=self.ref_point_value,
                                        curv_of_traj_mode=self.curv_of_traj_mode,
                                        window_for_curv_of_traj=self.window_for_curv_of_traj,
                                        curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                        truncate_curv_of_traj_by_time_of_capture=self.truncate_curv_of_traj_by_time_of_capture,
                                        use_curv_to_ff_center=self.use_curv_to_ff_center,
                                        eliminate_outliers=self.eliminate_outliers,
                                        test_or_control=self.test_or_control
                                        )

    def make_temporary_heading_info_df(self):
        # need to make a new one so that all stop_point_index are included
        # (in the original method, ff with big angles at ref point are removed)
        print('Need to make a new heading_info_df so that no data are dropped because ff_y is negative. '
              'No curature info is needed for this temporary heading_info_df.')
        nxt_ff_df_from_ref = self.add_d_heading_of_traj_to_df(
            self.nxt_ff_df_from_ref)
        cur_ff_df_from_ref = self.add_d_heading_of_traj_to_df(
            self.cur_ff_df_from_ref)
        cur_and_nxt_ff_from_ref_df = show_planning_utils.make_cur_and_nxt_ff_from_ref_df(
            nxt_ff_df_from_ref, cur_ff_df_from_ref, include_arc_info=False)
        self.heading_info_df = show_planning_utils.make_heading_info_df(
            cur_and_nxt_ff_from_ref_df, self.stops_near_ff_df, self.monkey_information, self.ff_real_position_sorted)
        self.add_both_ff_at_ref_to_heading_info_df()

    def add_heading_info_to_planning_data_by_point(self):
        self.add_rel_time_info_to_heading_info_df()

        heading_columns_to_add = [
            'stop_point_index', 'angle_from_m_before_stop_to_cur_ff', 'angle_from_stop_to_nxt_ff']
        # Note: variables like 'angle_opt_cur_end_to_nxt_ff', 'angle_cntr_cur_end_to_nxt_ff' in heading_info_df are based on ref points.
        # However, in both_ff_across_time_df, they are based on each point in the trajectory. So we use the variables in both_ff_across_time_df.

        build_factor_comp.add_dir_from_cur_ff_same_side(self.heading_info_df)
        temp_more_columns_to_try = ['dir_from_cur_ff_to_stop', 'dir_from_cur_ff_to_nxt_ff', 'dir_from_cur_ff_same_side',
                                    'angle_from_cur_ff_to_stop', 'angle_from_cur_ff_to_nxt_ff',
                                    'nxt_ff_distance_at_ref', 'nxt_ff_angle_at_ref',
                                    'cur_ff_distance_at_ref', 'cur_ff_angle_at_ref', 'cur_ff_angle_boundary_at_ref',
                                    'curv_range', 'curv_iqr', 'cur_ff_cluster_50_size',
                                    # 'rel_nxt_ff_last_flash_time_bbas', 'rel_nxt_ff_last_seen_time_bbas',
                                    # 'rel_nxt_ff_cluster_last_flash_time_bbas', 'rel_nxt_ff_cluster_last_seen_time_bbas',
                                    'rel_cur_ff_last_seen_time_bbas', 'rel_cur_ff_first_seen_time_bbas',
                                    ]
        all_columns_to_add = heading_columns_to_add + temp_more_columns_to_try
        # print the columns that are not in heading_info_df
        missing_columns = set(all_columns_to_add) - \
            set(self.heading_info_df.columns)
        if missing_columns:
            print(
                f'Note: The following columns are not in the new heading_info_df: {missing_columns}')

        all_columns_to_add = [
            col for col in all_columns_to_add if col in self.heading_info_df.columns]

        self.planning_data_by_point = self.planning_data_by_point.merge(
            self.heading_info_df[heading_columns_to_add + temp_more_columns_to_try], on='stop_point_index', how='left')

    def get_both_ff_across_time_df(self, n_seconds_before_stop=2.5):

        self._get_point_index_based_on_some_time_before_stop(
            n_seconds_before_stop=n_seconds_before_stop)

        # self.ff_caught_T_new_point_index = self.closest_stop_to_capture_df[
        #     'ff_caught_T_new_point_index']

        all_info_to_add = self.collect_data_for_each_segment(
            self.build_segment_around_stop)
        all_info_to_add, time_columns = self._add_time_info_to_df(
            all_info_to_add)

        all_info_to_add = all_info_to_add[all_info_to_add['segment_duration'] > 0.1].copy(
        )

        self._find_ff_info(all_info_to_add)

        all_info_to_add, cur_columns_added = self._add_basic_ff_info(
            all_info_to_add, 'cur_')
        all_info_to_add, nxt_columns_added = self._add_basic_ff_info(
            all_info_to_add, 'nxt_')

        all_info_to_add, cur_ff_df, cur_curv_df, cur_columns_added2 = self._add_ff_curv_info_to_df(
            all_info_to_add, 'cur_')
        all_info_to_add, nxt_ff_df, nxt_curv_df, nxt_columns_added2 = self._add_ff_curv_info_to_df(
            all_info_to_add, 'nxt_')
        both_ff_df = pn_utils._merge_both_ff_df(cur_curv_df, nxt_ff_df)
        both_ff_df = both_ff_df.merge(self.heading_info_df[[
            'point_index_before_stop', 'cur_ff_index']], on='cur_ff_index', how='left')
        
        columns_to_keep = time_columns + cur_columns_added + nxt_columns_added + cur_columns_added2 + nxt_columns_added2 + \
            ['stop_point_index', 'point_index', 'segment', 'target_index']
        # make sure there's no duplicate in columns_to_keep
        columns_to_keep = list(set(columns_to_keep))
        self.both_ff_across_time_df = all_info_to_add[columns_to_keep].copy()

        self.add_diff_in_abs_angle_to_nxt_ff_to_both_ff_across_time_df(both_ff_df)

        self.add_diff_in_curv_info_to_both_ff_across_time_df(both_ff_df)

        self._add_rel_x_and_y_to_both_ff_across_time_df()
        self.both_ff_across_time_df = self._add_traj_curv_to_df(
            self.both_ff_across_time_df)

        self._check_for_duplicate_point_index()

        self.both_ff_across_time_df.reset_index(drop=True, inplace=True)

        return self.both_ff_across_time_df

    def add_diff_in_abs_angle_to_nxt_ff_to_both_ff_across_time_df(self, both_ff_df):
        angle_df = pn_utils.get_angle_from_cur_arc_end_to_nxt_ff(both_ff_df)
        
        angle_df['angle_from_stop_to_nxt_ff'] = pn_utils.calculate_angle_from_stop_to_nxt_ff(self.monkey_information, both_ff_df.point_index_before_stop.values,
                                                                                 both_ff_df.nxt_ff_x.values, both_ff_df.nxt_ff_y.values)
        build_factor_comp._add_diff_in_abs_angle_to_nxt_ff(
            angle_df)

        columns_to_merge = ['cur_opt_arc_end_heading', 'angle_opt_cur_end_to_nxt_ff', 'angle_from_stop_to_nxt_ff',
                            'diff_in_angle_to_nxt_ff', 'diff_in_abs_angle_to_nxt_ff'
                            ]
        self.both_ff_across_time_df.drop(columns=columns_to_merge, errors='ignore', inplace=True)
        self.both_ff_across_time_df = self.both_ff_across_time_df.merge(angle_df[['point_index'] + columns_to_merge], on='point_index', how='left')
        
    
    def add_diff_in_curv_info_to_both_ff_across_time_df(self, both_ff_df):
        self.both_ff_across_time_df = pn_utils.add_diff_in_curv_info(
            self.both_ff_across_time_df, both_ff_df, self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new)

        # # check for NA in point_index_before_stop
        # if both_ff_df['point_index_before_stop'].isna().any():
        #     raise ValueError('There are NA in point_index_before_stop in both_ff_df. Please check the heading_info_df.')

        # diff_in_curv_info = pn_utils.find_diff_in_curv_info(
        #     both_ff_df, both_ff_df['point_index_before_stop'].values, self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new)
        # columns_to_merge = ['traj_curv_to_stop', 'curv_from_stop_to_nxt_ff',
        #                      'opt_curv_to_cur_ff', 'curv_from_cur_end_to_nxt_ff',
        #                      'd_curv_null_arc', 'd_curv_monkey',
        #                      'abs_d_curv_null_arc', 'abs_d_curv_monkey',
        #                      'diff_in_d_curv', 'diff_in_abs_d_curv']

        # diff_in_curv_info.rename(columns={'ref_point_index': 'point_index'}, inplace=True)
        # self.both_ff_across_time_df.drop(columns=columns_to_merge, errors='ignore', inplace=True)
        # self.both_ff_across_time_df = self.both_ff_across_time_df.merge(diff_in_curv_info[['point_index'] + columns_to_merge], on='point_index', how='left')

    def collect_data_for_each_segment(self, build_segment_func, select_info_kwargs={}):
        all_info_to_add = pd.DataFrame([])

        self.stops_near_ff_df['time_into_the_past'] = self.monkey_information.loc[
            self.stops_near_ff_df['point_index_in_the_past'].values, 'time'].values
        for i, row in self.stops_near_ff_df.iterrows():
            info_to_add = build_segment_func(row, **select_info_kwargs)
            # if info_to_add is empty, then skip
            if info_to_add.empty:
                continue
            info_to_add['segment'] = i
            info_to_add['target_index'] = row['cur_ff_index']
            all_info_to_add = pd.concat(
                [all_info_to_add, info_to_add], axis=0)
        return all_info_to_add

    def build_segment_around_stop(self, row):
        seg_start_time = max(
            self.ff_caught_T_new[row['cur_ff_index']-1] + 0.1, row['time_into_the_past'])
        seg_end_time = row['stop_time']

        # sanity check
        if seg_start_time > seg_end_time:
            # if the difference is less than 0.05s, then simply continue. The stop time might be too close to the previous capture time.
            if (seg_end_time - seg_start_time) < 0.11:
                return pd.DataFrame([])
            else:
                raise ValueError(f'segment_start > segment_end for row {row}')
            # note: segment_start = segment_end can happen if two fireflies were captured in a row.

        info_to_add = self.monkey_information[self.monkey_information['time'].between(
            seg_start_time, seg_end_time)].copy()

        info_to_add['stop_point_index'] = row['stop_point_index']
        info_to_add['cur_ff_index'] = row['cur_ff_index']
        info_to_add['nxt_ff_index'] = row['nxt_ff_index']
        info_to_add['seg_start_time'] = seg_start_time
        info_to_add['seg_end_time'] = seg_end_time
        info_to_add['seg_start_point_index'] = info_to_add['point_index'].min()
        info_to_add['segment_end_point_index'] = info_to_add['point_index'].max()
        return info_to_add

    def retrieve_neural_data(self):
        base_neural_class.NeuralBaseClass.retrieve_neural_data(
            self)

    def add_rel_time_info_to_heading_info_df(self):
        self.heading_info_df['rel_nxt_ff_last_flash_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['nxt_ff_last_flash_time_bbas']
        self.heading_info_df['rel_nxt_ff_last_seen_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['NXT_time_ff_last_seen_bbas']
        self.heading_info_df['rel_nxt_ff_cluster_last_flash_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['nxt_ff_cluster_last_flash_time_bbas']
        self.heading_info_df['rel_nxt_ff_cluster_last_seen_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['nxt_ff_cluster_last_seen_time_bbas']
        self.heading_info_df['rel_cur_ff_last_seen_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['CUR_time_ff_last_seen_bbas']
        self.heading_info_df['rel_cur_ff_first_seen_time_bbas'] = self.heading_info_df['stop_time'] - \
            self.heading_info_df['CUR_time_ff_first_seen_bbas']

    def _add_traj_curv_to_df(self, df):
        df = df.merge(self.curv_of_traj_df[[
                      'point_index', 'curv_of_traj']], on='point_index', how='left')
        df.rename(columns={'curv_of_traj': 'traj_curv'}, inplace=True)
        return df

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

    def _get_point_index_based_on_some_time_before_stop(self, n_seconds_before_stop=2.5):
        self.stops_near_ff_df['some_time_before_stop'] = self.stops_near_ff_df['stop_time'] - \
            n_seconds_before_stop
        self.stops_near_ff_df['point_index_in_the_past'] = np.searchsorted(
            self.monkey_information['time'].values, self.stops_near_ff_df['some_time_before_stop'].values) - 1

    def _add_ff_curv_info_to_df(self, df, which_ff_info):
        ff_df = self.nxt_ff_df_from_ref if which_ff_info == 'nxt_' else self.cur_ff_df_from_ref
        ff_df2 = ff_df[ff_df['ff_angle_boundary']
                       .between(-np.pi/4, np.pi/4)].copy()
        curv_df = curvature_utils.make_curvature_df(ff_df2, self.curv_of_traj_df, clean=False,
                                                    remove_invalid_rows=False,
                                                    invalid_curvature_ok=True,
                                                    monkey_information=self.monkey_information,
                                                    ff_caught_T_new=self.ff_caught_T_new)

        print('curv_df.columns:', np.unique(curv_df.columns))
        df, columns_added = pn_utils.add_curv_info(
            df, curv_df, which_ff_info)

        return df, ff_df, curv_df, columns_added

    def _add_time_info_to_df(self, df):
        df['time'] = self.monkey_information.loc[df['point_index'].values, 'time'].values
        df['stop_time'] = self.monkey_information.loc[df['stop_point_index'].values, 'time'].values
        df['segment_duration'] = df['seg_end_time'] - df['seg_start_time']
        df['time_rel_to_stop'] = df['stop_time'] - df['time']
        time_columns = ['time', 'stop_time', 'seg_start_time',
                        'seg_end_time', 'segment_duration', 'time_rel_to_stop']
        return df, time_columns

    def _add_basic_ff_info(self, df, which_ff_info):
        ff_df = self.nxt_ff_df_from_ref if which_ff_info == 'nxt_' else self.cur_ff_df_from_ref
        ff_df = ff_df.copy()
        ff_df.rename(columns={'ff_index': f'{which_ff_info}ff_index',
                              'ff_angle': f'{which_ff_info}ff_angle',
                              'ff_distance': f'{which_ff_info}ff_distance'}, inplace=True)
        df = df.merge(ff_df[['point_index', f'{which_ff_info}ff_index', f'{which_ff_info}ff_angle', f'{which_ff_info}ff_distance']], on=[
                      'point_index', f'{which_ff_info}ff_index'], how='left')
        columns_added = [f'{which_ff_info}ff_index',
                         f'{which_ff_info}ff_angle',
                         f'{which_ff_info}ff_distance']
        return df, columns_added

    def _find_ff_info(self, info_to_add):
        self.nxt_ff_df_from_ref = find_cvn_utils.find_ff_info(
            info_to_add['nxt_ff_index'].values,
            info_to_add['point_index'].values,
            self.monkey_information,
            self.ff_real_position_sorted)
        self.cur_ff_df_from_ref = find_cvn_utils.find_ff_info(
            info_to_add['cur_ff_index'].values,
            info_to_add['point_index'].values,
            self.monkey_information,
            self.ff_real_position_sorted)

        self.nxt_ff_df_from_ref[['stop_point_index', 'time', 'stop_time']] = info_to_add[[
            'stop_point_index', 'time', 'stop_time']].values
        self.cur_ff_df_from_ref[['stop_point_index', 'time', 'stop_time']] = info_to_add[[
            'stop_point_index', 'time', 'stop_time']].values

        return info_to_add

    def _check_for_duplicate_point_index(self):
        # duplicated point_index happens when the segment for a target extends into the previous segment
        # (this can happen when using a reference point is based on time or distance before stop)
        dup_rows = self.both_ff_across_time_df["point_index"].duplicated()
        if dup_rows.any():
            # retain the rows with the smaller stop_point_index)
            self.both_ff_across_time_df = self.both_ff_across_time_df.sort_values(
                by=['point_index', 'stop_point_index'], ascending=True).drop_duplicates(subset='point_index', keep='first')
            print(
                f'There are {dup_rows.sum()} duplicated point_index in both_ff_across_time_df. Retaining the rows with the smaller stop_point_index: {len(self.both_ff_across_time_df)}')

    # def _add_to_both_ff_when_seen_df(self, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df):
    #     curv_df.set_index('stop_point_index', inplace=True)
    #     self.both_ff_when_seen_df[f'{which_ff_info}ff_angle_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_angle']
    #     self.both_ff_when_seen_df[f'{which_ff_info}ff_distance_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_distance']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['cntr_arc_curv']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_curv']
    #     # self.both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_d_heading']
    #     self.both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    #     self.both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_of_traj']

    def get_both_ff_when_seen_df(self, crossing_ff=False, deal_with_rows_with_big_ff_angles=False):
        # This contains the planning-related information at specific time (such as when cur_ff was last visibl)
        # If crossing_ff is true, we'll get nxt_ff_info when cur_ff was first/last seen, and vice versa

        print('Making both_ff_when_seen_df...')
        self.both_ff_when_seen_df = self.nxt_ff_df_from_ref[[
            'stop_point_index']].copy().set_index('stop_point_index')
        for first_or_last in ['first', 'last']:
            for when_which_ff, ff_df in [('when_nxt_ff', self.nxt_ff_df_from_ref),
                                         ('when_cur_ff', self.cur_ff_df_from_ref)]:
                all_point_index = ff_df[f'point_index_ff_{first_or_last}_seen'].values
                self._find_nxt_ff_df_from_ref_2_and_cur_ff_df_from_ref_2_based_on_specific_point_index(
                    all_point_index=all_point_index)
                if deal_with_rows_with_big_ff_angles:
                    self._deal_with_rows_with_big_ff_angles(
                        remove_i_o_modify_rows_with_big_ff_angles=False, delete_the_same_rows=True)

                for which_ff_info in ['nxt_', 'cur_']:
                    if (when_which_ff == 'when_cur_ff') & (first_or_last == 'first') & (which_ff_info == 'cur_'):
                        continue  # because the information is already contained in cur ff info at ref point

                    if not crossing_ff:
                        if (which_ff_info == 'nxt_') & (when_which_ff == 'when_cur_ff'):
                            continue
                        if (which_ff_info == 'cur_') & (when_which_ff == 'when_nxt_ff'):
                            continue
                    if deal_with_rows_with_big_ff_angles:
                        ff_df_modified = self.nxt_ff_df_from_ref_modified if which_ff_info == 'nxt_' else self.cur_ff_df_from_ref_modified
                    else:
                        ff_df_modified = self.nxt_ff_df_from_ref if which_ff_info == 'nxt_' else self.cur_ff_df_from_ref

                    opt_arc_stop_first_vis_bdry = True if (
                        self.opt_arc_type == 'opt_arc_stop_first_vis_bdry') else False

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
                    pn_utils.add_to_both_ff_when_seen_df(
                        self.both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df)
        self.both_ff_when_seen_df.reset_index(drop=False, inplace=True)
        return self.both_ff_when_seen_df
