from decision_making_analysis.cluster_replacement import cluster_replacement_utils
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.GUAT import GUAT_and_TAFT, GUAT_helper_class, GUAT_utils
from decision_making_analysis import trajectory_info
from null_behaviors import curvature_utils, curv_of_traj_utils
from data_wrangling import further_processing_class

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class GUATCollectInfoHelperClass(GUAT_helper_class.GUATHelperClass):

    def get_current_ff_info_and_nxt_ff_info_for_info_collection(self, max_cluster_distance=50, max_time_since_last_vis=2.5, include_ff_in_near_future=True, duration_into_future=0.5, max_distance_to_stop=400,
                                                                columns_to_sort_nxt_ff_by=['abs_curv_diff', 'time_since_last_vis'], last_seen_and_next_seen_attributes_to_add=['ff_distance', 'ff_angle', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y']):

        self.find_current_and_alternative_ff_info(columns_to_sort_nxt_ff_by=columns_to_sort_nxt_ff_by, max_cluster_distance=max_cluster_distance, max_time_since_last_vis=max_time_since_last_vis,
                                                  include_ff_in_near_future=include_ff_in_near_future, duration_into_future=duration_into_future, max_distance_to_stop=max_distance_to_stop)

        self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info = GUAT_and_TAFT.add_curv_diff_and_ff_number_to_GUAT_cur_ff_info_and_GUAT_nxt_ff_info(self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info,
                                                                                                                                          self.ff_caught_T_new, self.ff_real_position_sorted, self.monkey_information, curv_of_traj_df=self.curv_of_traj_df
                                                                                                                                          )

        # add time to self.GUAT_cur_ff_info and self.GUAT_nxt_ff_info
        self.GUAT_cur_ff_info['time'] = self.monkey_information.loc[
            self.GUAT_cur_ff_info['point_index'].values, 'time'].values
        self.GUAT_nxt_ff_info['time'] = self.monkey_information.loc[self.GUAT_nxt_ff_info['point_index'].values, 'time'].values

        self.GUAT_cur_ff_info = cluster_replacement_utils.supply_info_of_ff_last_seen_and_next_seen_to_df(
            self.GUAT_cur_ff_info, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new, attributes_to_add=last_seen_and_next_seen_attributes_to_add, curv_of_traj_df=self.curv_of_traj_df)
        self.GUAT_nxt_ff_info = cluster_replacement_utils.supply_info_of_ff_last_seen_and_next_seen_to_df(
            self.GUAT_nxt_ff_info, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new, attributes_to_add=last_seen_and_next_seen_attributes_to_add, curv_of_traj_df=self.curv_of_traj_df)
        self.last_seen_and_next_seen_attributes_to_add = last_seen_and_next_seen_attributes_to_add

        os.makedirs(self.GUAT_folder_path, exist_ok=True)

    def find_current_and_alternative_ff_info(self,
                                             columns_to_sort_nxt_ff_by=[
                                                 'abs_curv_diff', 'time_since_last_vis'],
                                             max_cluster_distance=75,
                                             max_time_since_last_vis=2.5,
                                             max_distance_to_stop=400,
                                             include_ff_in_near_future=True,
                                             duration_into_future=0.5):

        print('Note, the current value for max_cluster_distance is', max_cluster_distance,
              '. Please make sure that this is the same value used to make the GUAT_w_ff_df.')
        GUAT_cur_ff_info = GUAT_and_TAFT.find_GUAT_cur_ff_info(self.GUAT_w_ff_df, self.ff_real_position_sorted, self.ff_life_sorted, self.ff_dataframe, self.monkey_information,
                                                               include_ff_in_near_future=include_ff_in_near_future, max_time_since_last_vis=max_time_since_last_vis,
                                                               max_cluster_distance=max_cluster_distance, duration_into_future=duration_into_future,
                                                               max_distance_to_stop=max_distance_to_stop)
        GUAT_nxt_ff_info = GUAT_and_TAFT.find_GUAT_nxt_ff_info(GUAT_cur_ff_info, self.ff_dataframe, self.ff_real_position_sorted, self.monkey_information, include_ff_in_near_future=include_ff_in_near_future,
                                                               max_time_since_last_vis=max_time_since_last_vis, duration_into_future=duration_into_future,
                                                               max_distance_to_stop=max_distance_to_stop)

        GUAT_cur_ff_info, GUAT_nxt_ff_info = GUAT_and_TAFT.retain_useful_cur_and_nxt_info(
            GUAT_cur_ff_info, GUAT_nxt_ff_info)

        # The below can be replaced now because we're using the original GUAT_cur_ff_info + GUAT_nxt_ff_info as more_ff_df
        # if include_ff_in_near_future:
        #     unique_point_index_and_time_df = GUAT_cur_ff_info[['point_index', 'time', 'total_stop_time]].drop_duplicates()
        #     ff_info, self.all_available_ff_in_near_future = GUAT_and_TAFT.find_additional_ff_info_for_near_future(unique_point_index_and_time_df, self.ff_dataframe_visible, self.ff_real_position_sorted, self.monkey_information,
        #                                                                                                           duration_into_future=duration_into_future)
        #     ff_info, self.additional_curvature_df = GUAT_and_TAFT.find_curv_diff_for_ff_info(ff_info, self.monkey_information, self.ff_real_position_sorted, curv_of_traj_df=self.curv_of_traj_df)
        # else:
        #     self.additional_curvature_df, self.all_available_ff_in_near_future = None, None
        self.additional_curvature_df, self.all_available_ff_in_near_future = None, None

        self.GUAT_cur_ff_info = GUAT_and_TAFT.polish_GUAT_cur_ff_info(
            GUAT_cur_ff_info)
        self.GUAT_nxt_ff_info = GUAT_and_TAFT.polish_GUAT_nxt_ff_info(GUAT_nxt_ff_info, GUAT_cur_ff_info, self.ff_real_position_sorted, self.ff_life_sorted, self.ff_dataframe, self.monkey_information, max_cluster_distance=max_cluster_distance,
                                                                      columns_to_sort_nxt_ff_by=columns_to_sort_nxt_ff_by,
                                                                      max_time_since_last_vis=max_time_since_last_vis, duration_into_future=duration_into_future)

        self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info = GUAT_and_TAFT.make_sure_GUAT_nxt_ff_info_and_GUAT_cur_ff_info_have_the_same_point_indices(
            self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info)

        self.point_index_all = self.GUAT_cur_ff_info.point_index.unique()
        self.time_all = self.monkey_information.loc[self.point_index_all, 'time'].values

    def set_time_of_eval(self, GUAT_w_ff_df, time_with_respect_to_first_stop=None, time_with_respect_to_second_stop=None, time_with_respect_to_last_stop=None):
        self.GUAT_w_ff_df = GUAT_w_ff_df.copy()

        # make sure that only one of the three time_with_respect_to_* is not None
        if (time_with_respect_to_first_stop is not None) & (time_with_respect_to_second_stop is not None):
            raise ValueError(
                'Only one of the three time_with_respect_to_* can be not None.')
        if (time_with_respect_to_first_stop is not None) & (time_with_respect_to_last_stop is not None):
            raise ValueError(
                'Only one of the three time_with_respect_to_* can be not None.')
        if (time_with_respect_to_second_stop is not None) & (time_with_respect_to_last_stop is not None):
            raise ValueError(
                'Only one of the three time_with_respect_to_* can be not None.')

        if time_with_respect_to_first_stop is not None:
            self.time_of_eval = self.GUAT_w_ff_df['first_stop_time'] + \
                time_with_respect_to_first_stop
        elif time_with_respect_to_second_stop is not None:
            self.time_of_eval = self.GUAT_w_ff_df['second_stop_time'] + \
                time_with_respect_to_second_stop
        else:
            self.time_of_eval = self.GUAT_w_ff_df['last_stop_time'] + \
                time_with_respect_to_last_stop
        self.GUAT_w_ff_df['time_of_eval'] = self.time_of_eval
        self.GUAT_w_ff_df['point_index_of_eval'] = self.monkey_information['point_index'].values[np.searchsorted(
            self.monkey_information['time'].values, self.time_of_eval, side='right')-1]

    def eliminate_crossing_boundary_cases(self, n_seconds_before_crossing_boundary=None, n_seconds_after_crossing_boundary=None):
        n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary = self.determine_n_seconds_before_or_after_crossing_boundary(
            n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary
        )

        crossing_boundary_time = self.monkey_information.loc[
            self.monkey_information['crossing_boundary'] == 1, 'time'].values
        original_length = len(self.time_of_eval)
        CB_indices, non_CB_indices, self.time_of_eval = decision_making_utils.find_time_points_that_are_within_n_seconds_after_crossing_boundary(self.time_of_eval, crossing_boundary_time,
                                                                                                                                                 n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary, n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary)
        self.GUAT_w_ff_df = self.GUAT_w_ff_df.iloc[non_CB_indices].reset_index(
            drop=True)
        print("GUAT_w_ff_df:", len(self.time_of_eval),
              "out of", original_length, "rows remains")

    def make_one_stop_w_ff_df(self):
        self.get_monkey_data(include_GUAT_data=True,
                             include_TAFT_data=True)
        self.one_stop_df = GUAT_utils.streamline_getting_one_stop_df(
            self.monkey_information, self.ff_dataframe, self.ff_caught_T_new)
        self.one_stop_w_ff_df = GUAT_utils.make_one_stop_w_ff_df(
            self.one_stop_df)

    def _add_one_stop_info_to_GUAT_w_ff_df(self):
        self.make_one_stop_w_ff_df()
        self.one_stop_w_ff_df['cluster_index'] = np.arange(self.GUAT_w_ff_df['cluster_index'].max()+1,
                                                           self.GUAT_w_ff_df['cluster_index'].max()+1+len(self.one_stop_w_ff_df))

        # find point_index in self.one_stop_w_ff_df that are also in self.GUAT_w_ff_df
        common_point_index = np.intersect1d(
            self.GUAT_w_ff_df['first_stop_point_index'].values, self.one_stop_w_ff_df['first_stop_point_index'].values)
        print(f'Out of {len(self.one_stop_w_ff_df)} rows in one_stop_w_ff_df, {len(common_point_index)} rows have first_stop_point_index that are also in GUAT_w_ff_df. These rows are removed from one_cur_ff_df.')
        one_stop_w_ff_df = self.one_stop_w_ff_df[~self.one_stop_w_ff_df['first_stop_point_index'].isin(
            common_point_index)].copy()
        # only keep columns in one_stop_w_ff_df that are also in GUAT_w_ff_df
        columns_of_one_stop_to_keep = [
            col for col in one_stop_w_ff_df.columns if col in self.GUAT_w_ff_df.columns]
        one_stop_w_ff_df = one_stop_w_ff_df[columns_of_one_stop_to_keep].copy()
        # print columns in one_stop_w_ff_df but not in GUAT_w_ff_df
        if len(self.GUAT_w_ff_df.columns.difference(one_stop_w_ff_df.columns)) > 0:
            print('Columns in GUAT_w_ff_df but not in one_stop_w_ff_df:',
                  self.GUAT_w_ff_df.columns.difference(one_stop_w_ff_df.columns))

        self.GUAT_w_ff_df = pd.concat(
            [self.GUAT_w_ff_df, one_stop_w_ff_df], axis=0).reset_index(drop=True)

        for col in self.GUAT_w_ff_df.columns:
            if '_index' in col:
                if self.GUAT_w_ff_df[col].isna().sum() == 0:
                    self.GUAT_w_ff_df[col] = self.GUAT_w_ff_df[col].astype(
                        'int64')

        self.GUAT_w_ff_df.loc[self.GUAT_w_ff_df['last_stop_point_index'].isna(
        ), 'last_stop_point_index'] = self.GUAT_w_ff_df.loc[self.GUAT_w_ff_df['last_stop_point_index'].isna(), 'first_stop_point_index']
        self.GUAT_w_ff_df.loc[self.GUAT_w_ff_df['last_stop_time'].isna(
        ), 'last_stop_time'] = self.GUAT_w_ff_df.loc[self.GUAT_w_ff_df['last_stop_time'].isna(), 'first_stop_time']
        self.GUAT_w_ff_df['total_stop_time'] = self.GUAT_w_ff_df['last_stop_time'] - \
            self.GUAT_w_ff_df['first_stop_time']

    def get_monkey_data(self, already_retrieved_ok=True, include_ff_dataframe=True, include_GUAT_data=False,
                        include_TAFT_data=False):
        if (already_retrieved_ok is False) | (not hasattr(self, 'monkey_information')):
            self.data_item = further_processing_class.FurtherProcessing(
                raw_data_folder_path=self.raw_data_folder_path)
            self.data_item.retrieve_or_make_monkey_data()
            self.monkey_information = self.data_item.monkey_information
            self.ff_life_sorted = self.data_item.ff_life_sorted
            self.ff_real_position_sorted = self.data_item.ff_real_position_sorted
            self.ff_believed_position_sorted = self.data_item.ff_believed_position_sorted
            self.ff_caught_T_new = self.data_item.ff_caught_T_new

        for df in ['monkey_information', 'ff_life_sorted', 'ff_real_position_sorted', 'ff_believed_position_sorted', 'ff_caught_T_new']:
            setattr(self, df, getattr(self.data_item, df).copy())

        if include_ff_dataframe:
            if (already_retrieved_ok is False) | (not hasattr(self, 'ff_dataframe')):
                # if self.data_item is not called yet
                if not hasattr(self, 'data_item'):
                    self.data_item = further_processing_class.FurtherProcessing(
                        raw_data_folder_path=self.raw_data_folder_path)
                    self.data_item.retrieve_or_make_monkey_data()
                if (already_retrieved_ok is False) | (not hasattr(self, 'ff_dataframe')):
                    self.data_item.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                    self.ff_dataframe = self.data_item.ff_dataframe
                    self.ff_dataframe_visible = self.ff_dataframe.loc[self.ff_dataframe['visible'] == 1].copy(
                    )

        if include_GUAT_data:
            if not hasattr(self, 'GUAT_trials_df'):
                if not hasattr(self, 'data_item'):
                    self.data_item = further_processing_class.FurtherProcessing(
                        raw_data_folder_path=self.raw_data_folder_path)
                    self.data_item.retrieve_or_make_monkey_data()
                    self.data_item.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                elif not hasattr(self, 'ff_dataframe'):
                    self.data_item.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                self.data_item.get_give_up_after_trying_info()
                self.give_up_after_trying_info_bundle = self.data_item.give_up_after_trying_info_bundle
                self.GUAT_w_ff_df = self.data_item.GUAT_w_ff_df

            for df in ['give_up_after_trying_trials', 'GUAT_indices_df', 'GUAT_trials_df', 'GUAT_point_indices_for_anim', 'GUAT_w_ff_df']:
                setattr(self, df, getattr(self.data_item, df).copy())

        if include_TAFT_data:
            if not hasattr(self, 'TAFT_trials_df'):
                if not hasattr(self, 'data_item'):
                    self.data_item = further_processing_class.FurtherProcessing(
                        raw_data_folder_path=self.raw_data_folder_path)
                    self.data_item.retrieve_or_make_monkey_data()
                    self.data_item.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                elif not hasattr(self, 'ff_dataframe'):
                    self.data_item.make_or_retrieve_ff_dataframe(
                        exists_ok=True)
                self.data_item.get_try_a_few_times_info()

            for df in ['try_a_few_times_trials', 'TAFT_indices_df', 'TAFT_trials_df', 'try_a_few_times_indices_for_anim']:
                setattr(self, df, getattr(self.data_item, df).copy())

        if include_ff_dataframe:
            self.cluster_around_target_indices = None
            self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted,
                                    self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)

    def retrieve_or_make_GUAT_trials_df(self, exists_ok=True):
        filepath = os.path.join(self.GUAT_folder_path, 'GUAT_trials_df.csv')
        if exists(filepath) & exists_ok:
            self.GUAT_trials_df = pd.read_csv(filepath)
            print('Retrieved GUAT_trials_df')
        else:
            self.GUAT_trials_df = GUAT_utils.make_GUAT_trials_df(
                self.GUAT_w_ff_df, self.ff_life_sorted, self.ff_real_position_sorted, self.monkey_information)
            self.GUAT_trials_df.to_csv(filepath, index=False)
            print('Made and saved GUAT_trials_df')

    def make_curvature_df(self, curv_of_traj_df=None, clean=True):

        self.curvature_df = curvature_utils.make_curvature_df(
            self.ff_dataframe, curv_of_traj_df, clean=clean)

    def add_curvature_info_to_ff_dataframe(self, column_exists_ok=False):
        # add the column abs_curv_diff to ff_dataframe through merging with curvature_df
        if 'curv_diff' not in self.ff_dataframe.columns or column_exists_ok is False:
            curvature_df_sub = self.curvature_df[[
                'ff_index', 'point_index', 'curv_diff']]
            self.ff_dataframe = pd.merge(self.ff_dataframe, curvature_df_sub, on=[
                                         'ff_index', 'point_index'], how='left')
            # fill na of curv_diff, but randomly make half of them negative
            na_index = self.ff_dataframe['curv_diff'].isna()
            self.ff_dataframe.loc[na_index, 'curv_diff'] = np.random.choice(
                [-1, 1], size=na_index.sum()) * 0.6
            self.ff_dataframe['abs_curv_diff'] = self.ff_dataframe['curv_diff'].abs(
            )

    def add_curv_of_traj_info_to_monkey_information(self, column_exists_ok=False):
        # add the column abs_curv_diff to ff_dataframe through merging with curvature_df
        if ('curv_of_traj' not in self.monkey_information.columns) or (column_exists_ok is False):
            curv_of_traj_df_sub = self.curv_of_traj_df[[
                'point_index', 'curv_of_traj']]
            self.monkey_information = pd.merge(
                self.monkey_information, curv_of_traj_df_sub, on=['point_index'], how='left')
            self.monkey_information['abs_curv_of_traj'] = self.monkey_information['curv_of_traj'].abs(
            )
            # check if there are any NAs
            if self.monkey_information['curv_of_traj'].isna().sum() > 0:
                print('There are NAs in monkey_information.curv_of_traj after merging with curv_of_traj_df. Fill them with 0. Their indices are:')
                print(self.monkey_information.loc[self.monkey_information['curv_of_traj'].isna(
                ), 'point_index'].values)
                self.monkey_information['curv_of_traj'] = self.monkey_information['curv_of_traj'].fillna(
                    0)

    def make_or_retrieve_curv_of_traj_df(self, exists_ok=True, window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance',
                                         truncate_curv_of_traj_by_time_of_capture=False):

        filepath = os.path.join(self.GUAT_folder_path, 'curv_of_traj_df.csv')
        if exists(filepath) & exists_ok:
            self.curv_of_traj_df = pd.read_csv(
                filepath).drop(["Unnamed: 0"], axis=1)
            self.curv_of_traj_df = self.curv_of_traj_df[[
                'point_index', 'curv_of_traj']]
            print('Retrieved curv_of_traj_df')
        else:
            self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
                window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
            self.curv_of_traj_df = self.curv_of_traj_df[[
                'point_index', 'curv_of_traj']]
            self.curv_of_traj_df.to_csv(filepath)
            print(f'Made and saved curv_of_traj_df at {filepath}')

    def get_more_ff_df(self):
        # get more ff_info for plotting
        self.more_ff_df = pd.concat(
            [self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info], axis=0).reset_index(drop=True)
        self.more_ff_df.drop_duplicates(
            subset=['point_index', 'ff_index'], inplace=True)

        # check if the index in self.more_ff_df is the same set as self.point_index_all. If not, raise an error
        if not set(self.more_ff_df['point_index'].values) == set(self.point_index_all):
            raise ValueError(
                'The point_index_all and the point_index in self.more_ff_df are not the same set.')

        # self.more_ff_df = cluster_replacement_utils.find_more_ff_df(self.point_index_all, self.ff_dataframe, self.ff_real_position_sorted, self.monkey_information, all_available_ff_in_near_future=self.all_available_ff_in_near_future,
        #                                                                   attributes_for_plotting=['ff_distance', 'ff_angle', 'time_since_last_vis', 'time_till_next_visible'])
        # self.more_ff_df = cluster_replacement_utils.supply_info_of_ff_last_seen_and_next_seen_to_df(self.more_ff_df, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new,
        #                                                                                                   curv_of_traj_df=self.curv_of_traj_df, attributes_to_add=self.last_seen_and_next_seen_attributes_to_add)
        return self.more_ff_df

    def update_curvature_df_with_additional_curvature_df(self):
        self.curvature_df = pd.concat(
            [self.curvature_df, self.additional_curvature_df], axis=0)
        curvature_df_sub = self.curvature_df[['point_index', 'ff_index']]
        self.curvature_df = self.curvature_df[~curvature_df_sub.duplicated()]

    def get_trajectory_and_stop_info_for_machine_learning(self):
        '''
        Sample columns of traj_stops_df (with num_points = 10):
        ['whether stopped_-1.0s', 'whether stopped_-0.78s',
        'whether stopped_-0.56s', 'whether stopped_-0.33s',
        'whether stopped_-0.11s', 'whether stopped_0.11s',
        'whether stopped_0.33s', 'whether stopped_0.56s',
        'whether stopped_0.78s', 'whether stopped_1.0s', 'point_index']

        Sample columns of traj_points_df (with num_points = 10):
        ['monkey_distance_-1.0s', 'monkey_distance_-0.78s',
        'monkey_distance_-0.56s', 'monkey_distance_-0.33s',
        'monkey_distance_-0.11s', 'monkey_distance_0.11s',
        'monkey_distance_0.33s', 'monkey_distance_0.56s',
        'monkey_distance_0.78s', 'monkey_distance_1.0s', 'monkey_angle-1.0s',
        'monkey_angle-0.78s', 'monkey_angle-0.56s', 'monkey_angle-0.33s',
        'monkey_angle-0.11s', 'monkey_angle0.11s', 'monkey_angle0.33s',
        'monkey_angle0.56s', 'monkey_angle0.78s', 'monkey_angle1.0s',
        'point_index'],
       '''
        # get trajectory info to be put into machine learning input
        traj_points, trajectory_feature_names = trajectory_info.generate_trajectory_position_data(self.time_all, self.monkey_information, time_range_of_trajectory=self.time_range_of_trajectory,
                                                                                                  num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'], trajectory_features=self.trajectory_features)
        traj_stops, temp_trajectory_feature_names = trajectory_info.generate_stops_info(self.time_all, self.monkey_information, time_range_of_trajectory=self.time_range_of_trajectory,
                                                                                        num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'])
        self.traj_points_df = pd.DataFrame(
            traj_points, columns=trajectory_feature_names)
        self.traj_stops_df = pd.DataFrame(
            traj_stops, columns=temp_trajectory_feature_names)
        self.traj_points_df['point_index'] = self.point_index_all
        self.traj_stops_df['point_index'] = self.point_index_all

    def get_more_trajectory_info_for_plotting(self,
                                              time_range_of_trajectory_to_plot=None,
                                              num_time_points_for_trajectory_to_plot=10):
        # get more trajectory info for plotting
        if time_range_of_trajectory_to_plot is not None:
            more_traj_points, trajectory_feature_names = trajectory_info.generate_trajectory_position_data(
                self.time_all, self.monkey_information, time_range_of_trajectory=time_range_of_trajectory_to_plot, num_time_points_for_trajectory=num_time_points_for_trajectory_to_plot, trajectory_features=self.trajectory_features)
            more_traj_stops, temp_trajectory_feature_names = trajectory_info.generate_stops_info(
                self.time_all, self.monkey_information, time_range_of_trajectory=time_range_of_trajectory_to_plot, num_time_points_for_trajectory=num_time_points_for_trajectory_to_plot)
            self.more_traj_points_df = pd.DataFrame(
                more_traj_points, columns=trajectory_feature_names)
            self.more_traj_stops_df = pd.DataFrame(
                more_traj_stops, columns=temp_trajectory_feature_names)
            self.more_traj_points_df['point_index'] = self.point_index_all
            self.more_traj_stops_df['point_index'] = self.point_index_all
        else:
            self.more_traj_points_df = None
            self.more_traj_stops_df = None

    def get_traj_ff_info(self):
        traj_time_2d, trajectory_data_dict = trajectory_info.find_trajectory_data(
            self.time_all, self.monkey_information, time_range_of_trajectory=self.time_range_of_trajectory, num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'])

        # convert traj_time_2d to point_index_2d use np.searchsorted
        point_index_2d = np.searchsorted(
            self.monkey_information.time.values, traj_time_2d)
        # make sure no index is out of bound
        point_index_2d[point_index_2d >= len(self.monkey_information)] = len(
            self.monkey_information) - 1

        GUAT_cur_ff_info = self.GUAT_cur_ff_info.copy()
        GUAT_nxt_ff_info = self.GUAT_nxt_ff_info.copy()

        GUAT_cur_ff_info.columns

        GUAT_ff_info_combined = pd.concat([GUAT_cur_ff_info[['point_index', 'ff_index', 'ff_number']], GUAT_nxt_ff_info[[
                                          'point_index', 'ff_index', 'ff_number']]], axis=0)
        GUAT_ff_info_combined = GUAT_ff_info_combined[GUAT_ff_info_combined['ff_index'] > 0].copy(
        )

        list_of_point_index = []
        list_of_ff_index = []
        list_of_ff_number = []
        list_of_traj_point_index = []
        list_of_traj_relative_index = []
        for i in range(len(self.point_index_all)):
            current_point_index = self.point_index_all[i]
            traj_point_index_array = point_index_2d[i, :]
            relevant_ff = GUAT_ff_info_combined[GUAT_ff_info_combined['point_index']
                                                == current_point_index]

            for index, row in relevant_ff.iterrows():
                ff = row['ff_index']
                ff_number = row['ff_number']
                num_traj_points = len(traj_point_index_array)
                list_of_point_index.extend(
                    [current_point_index] * num_traj_points)
                list_of_ff_index.extend([ff] * num_traj_points)
                list_of_ff_number.extend([ff_number] * num_traj_points)
                list_of_traj_point_index.extend(traj_point_index_array)
                traj_relative_index = np.arange(
                    (ff_number-1)*num_traj_points, ff_number*num_traj_points)
                list_of_traj_relative_index.extend(traj_relative_index)
                # list_of_traj_relative_index.extend(np.arange(num_traj_points))

        new_traj_df = pd.DataFrame({'point_index': list_of_point_index,
                                    'ff_index': list_of_ff_index,
                                    'ff_number': list_of_ff_number,
                                    'traj_point_index': list_of_traj_point_index,
                                    'traj_relative_index': list_of_traj_relative_index, })

        ff_info = decision_making_utils.find_many_ff_info_anew(new_traj_df.ff_index.values, new_traj_df.traj_point_index.values, self.ff_real_position_sorted, self.ff_dataframe_visible, self.monkey_information, add_time_till_next_visible=True, add_curv_diff=True,
                                                               ff_caught_T_new=self.curv_of_traj_df, truncate_curv_of_traj_by_time_of_capture=True, ff_radius=10, curv_of_traj_df=self.curv_of_traj_df)
        ff_info = ff_info[['ff_index', 'point_index', 'time_since_last_vis',
                           'curv_diff', 'ff_distance', 'ff_angle']].copy()
        ff_info['traj_point_index'] = ff_info['point_index'].values.copy()
        try:
            ff_info['traj_relative_index'] = new_traj_df['traj_relative_index'].values.copy()
        except ValueError:
            print('ff_info and new_traj_df have different lengths. ff_info:',
                  ff_info.shape, 'new_traj_df:', new_traj_df.shape)
            print('ff_info head:')
            print(ff_info.head())
            print('new_traj_df head:')
            print(new_traj_df.head())
            print('ff_info tail:')
            print(ff_info.tail())
            print('new_traj_df tail:')
            print(new_traj_df.tail())

        ff_info['point_index'] = new_traj_df['point_index'].values
        ff_info['ff_number'] = new_traj_df['ff_number'].values
        self.traj_ff_info = ff_info.sort_values(
            by=['point_index', 'ff_number', 'traj_point_index']).reset_index(drop=True)

    def add_arc_info_to_each_df_of_ff_info(self, curvature_df):
        # add best_arc_info to GUAT_cur_ff_info and GUAT_nxt_ff_info
        curvature_df = pd.concat(
            [curvature_df, self.additional_curvature_df], axis=0)
        curvature_df_sub = curvature_df[['point_index', 'ff_index']].copy()
        curvature_df = curvature_df[~curvature_df_sub.duplicated()]
        arc_info = ['curv_of_traj', 'curvature_lower_bound',
                    'curvature_upper_bound', 'optimal_curvature', 'curv_diff', 'abs_curv_diff']

        curvature_df_sub = curvature_df[[
            'ff_index', 'point_index'] + arc_info].copy()
        for column in arc_info:
            if column in self.GUAT_cur_ff_info.columns:
                self.GUAT_cur_ff_info = self.GUAT_cur_ff_info.drop([
                    column], axis=1)
            if column in self.GUAT_nxt_ff_info.columns:
                self.GUAT_nxt_ff_info = self.GUAT_nxt_ff_info.drop(
                    [column], axis=1)
            if column in self.more_ff_df.columns:
                self.more_ff_df = self.more_ff_df.drop([column], axis=1)

        self.GUAT_cur_ff_info = pd.merge(self.GUAT_cur_ff_info, curvature_df_sub, on=[
            'ff_index', 'point_index'], how='left')
        self.GUAT_nxt_ff_info = pd.merge(self.GUAT_nxt_ff_info, curvature_df_sub, on=[
                                         'ff_index', 'point_index'], how='left')
        self.more_ff_df = pd.merge(self.more_ff_df, curvature_df_sub, on=[
                                   'ff_index', 'point_index'], how='left')
        curvature_utils.fill_up_NAs_in_columns_related_to_curvature(
            self.GUAT_cur_ff_info, self.monkey_information, self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)
        curvature_utils.fill_up_NAs_in_columns_related_to_curvature(
            self.GUAT_nxt_ff_info, self.monkey_information, self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)
        curvature_utils.fill_up_NAs_in_columns_related_to_curvature(
            self.more_ff_df, self.monkey_information, self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)
