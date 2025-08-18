from decision_making_analysis import free_selection, trajectory_info
from decision_making_analysis.cluster_replacement import cluster_replacement_utils
from decision_making_analysis.GUAT import GUAT_helper_class, GUAT_collect_info_class
from data_wrangling import combine_info_utils
from null_behaviors import curvature_utils
import pandas as pd
import copy
import numpy as np
import os


class GUATCombineInfoAcrossSessions(GUAT_helper_class.GUATHelperClass):

    df_names = ['GUAT_nxt_ff_info', 'GUAT_cur_ff_info',
                'traj_data_df', 'more_traj_data_df', 'more_ff_df']
    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    def __init__(self, gc_kwargs, monkey_name='monkey_Bruno'):
        self.gc_kwargs = gc_kwargs
        self.monkey_information = None
        self.polar_plots_kwargs = {}
        self.monkey_name = monkey_name
        self.combd_GUAT_info_folder_path = f"all_monkey_data/decision_making/{self.monkey_name}/combined_data/GUAT_info"

    def retrieve_or_make_combined_info(self, gc_kwargs,
                                       combined_info_exists_ok=True,
                                       GUAT_w_ff_df_exists_ok=True,
                                       curv_of_traj_df_exists_ok=True,
                                       ff_df_exist_in_GUAT_store_ok=True,
                                       traj_df_exist_in_GUAT_store_ok=True):
        self.gc_kwargs = gc_kwargs
        if combined_info_exists_ok:
            self.combined_info, self.collect_info_flag = combine_info_utils.try_to_retrieve_combined_info(
                self.combd_GUAT_info_folder_path, df_names=self.df_names)
        else:
            self.collect_info_flag = True

        self.trajectory_features = gc_kwargs['trajectory_features']
        if self.collect_info_flag:
            self.combined_info, self.all_traj_feature_names = self.collect_combined_info_for_GUAT(gc_kwargs, GUAT_w_ff_df_exists_ok=GUAT_w_ff_df_exists_ok,
                                                                                                  ff_df_exist_in_GUAT_store_ok=ff_df_exist_in_GUAT_store_ok, traj_df_exist_in_GUAT_store_ok=traj_df_exist_in_GUAT_store_ok, curv_of_traj_df_exists_ok=curv_of_traj_df_exists_ok)
        else:
            self.all_traj_feature_names = self.find_all_traj_feature_names(
                gc_kwargs, traj_point_features=gc_kwargs['trajectory_features'])

    def collect_combined_info_for_GUAT(self, gc_kwargs,
                                       GUAT_w_ff_df_exists_ok=True,
                                       curv_of_traj_df_exists_ok=True,
                                       ff_df_exist_in_GUAT_store_ok=True,
                                       traj_df_exist_in_GUAT_store_ok=True):

        if traj_df_exist_in_GUAT_store_ok & (ff_df_exist_in_GUAT_store_ok == False):
            traj_df_exist_in_GUAT_store_ok = False
            print('Warning: ff_df_exist_in_GUAT_store_ok False, so traj_df_exist_in_GUAT_store_ok is forced to be False as well.')

        sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
            self.raw_data_dir_name, self.monkey_name)
        sessions_df_for_one_monkey = combine_info_utils.check_which_df_exists_for_each_session(
            sessions_df_for_one_monkey, df_names=self.df_names)
        print('Making missing GUAT df for monkey {}'.format(self.monkey_name))
        sessions_df_for_one_monkey = self.make_missing_GUAT_df(sessions_df_for_one_monkey, gc_kwargs, GUAT_w_ff_df_exists_ok=GUAT_w_ff_df_exists_ok,
                                                               curv_of_traj_df_exists_ok=curv_of_traj_df_exists_ok, traj_df_exist_ok=traj_df_exist_in_GUAT_store_ok)
        print('Collecting important info from all sessions of monkey {}'.format(
            self.monkey_name))
        self.all_important_info, self.all_point_index_to_new_number = combine_info_utils.collect_info_from_all_sessions(
            sessions_df_for_one_monkey, df_names=self.df_names)
        self.all_traj_feature_names = self.find_all_traj_feature_names(
            gc_kwargs, traj_point_features=gc_kwargs['trajectory_features'])

        self.combined_info = combine_info_utils.turn_all_important_info_into_combined_info(
            self.all_important_info, self.combd_GUAT_info_folder_path, save_each_df_as_csv=True)

        return self.combined_info, self.all_traj_feature_names

    def find_all_traj_feature_names(self, gc_kwargs, traj_point_features=['monkey_distance', 'monkey_angle']):
        all_traj_feature_names = trajectory_info.make_all_traj_feature_names(time_range_of_trajectory=gc_kwargs['time_range_of_trajectory'], num_time_points_for_trajectory=gc_kwargs['num_time_points_for_trajectory'],
                                                                             time_range_of_trajectory_to_plot=gc_kwargs['time_range_of_trajectory_to_plot'], num_time_points_for_trajectory_to_plot=gc_kwargs[
                                                                                 'num_time_points_for_trajectory_to_plot'],
                                                                             traj_point_features=traj_point_features)
        return all_traj_feature_names

    def make_missing_GUAT_df(self, sessions_df_for_one_monkey, gc_kwargs,
                             GUAT_w_ff_df_exists_ok=True,
                             curv_of_traj_df_exists_ok=True,
                             traj_df_exist_ok=True):
        sessions_df_for_one_monkey['finished'] = False
        # if all df in df_names exist (being True in the corresponding column), then finished is True
        sessions_df_for_one_monkey.loc[sessions_df_for_one_monkey[self.df_names].all(
            axis=1), 'finished'] = True
        sessions_df_for_one_monkey['remade'] = False
        for index, row in sessions_df_for_one_monkey.iterrows():
            if row['finished'] == True:
                continue
            raw_data_folder_path = os.path.join(
                self.raw_data_dir_name, row['monkey_name'], row['data_name'])
            print('raw_data_folder_path:', raw_data_folder_path)
            gcc = GUAT_collect_info_class.GUATCollectInfoForSession(
                raw_data_folder_path=raw_data_folder_path, gc_kwargs=gc_kwargs)
            important_info = gcc.streamline_process_to_collect_info_from_one_session(GUAT_w_ff_df_exists_ok=GUAT_w_ff_df_exists_ok,
                                                                                     curv_of_traj_df_exists_ok=curv_of_traj_df_exists_ok,
                                                                                     update_point_index=False)
            for df in self.df_names:
                important_info[df].to_csv(os.path.join(
                    gcc.GUAT_folder_path, df+'.csv'))
            sessions_df_for_one_monkey.loc[index, 'finished'] = True
            sessions_df_for_one_monkey.loc[index, 'remade'] = True

        if not traj_df_exist_ok:
            for index, row in sessions_df_for_one_monkey.iterrows():
                if row['remade']:
                    continue
                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                gcc = GUAT_collect_info_class.GUATCollectInfoForSession(
                    raw_data_folder_path=raw_data_folder_path, gc_kwargs=gc_kwargs)
                # otherwise, we will remake traj_data_df and more_traj_data_df
                print('We will remake traj_data_df and more_traj_data_df')
                # get all_point_index
                GUAT_nxt_ff_info = pd.read_csv(os.path.join(
                    gcc.GUAT_folder_path, 'GUAT_nxt_ff_info.csv'))
                if 'old_point_index' in GUAT_nxt_ff_info.columns:
                    all_point_index = np.unique(
                        GUAT_nxt_ff_info['old_point_index'].values)
                else:
                    all_point_index = np.unique(
                        GUAT_nxt_ff_info['point_index'].values)
                gcc.streamline_process_to_collect_traj_data_only(
                    all_point_index, curv_of_traj_df_exists_ok=curv_of_traj_df_exists_ok)
                gcc.traj_data_df.to_csv(os.path.join(
                    gcc.GUAT_folder_path, 'traj_data_df.csv'))
                gcc.more_traj_data_df.to_csv(os.path.join(
                    gcc.GUAT_folder_path, 'more_traj_data_df.csv'))
        return sessions_df_for_one_monkey

    # ================== below is for processing the combined info

    def unpack_and_reload_combined_info_back_to_self(self, combined_info=None, all_traj_feature_names=None):
        if combined_info is not None:
            self.combined_info = copy.deepcopy(combined_info)
            # otherwise, we assume that the combined_info is already in self from previous functions
        if all_traj_feature_names is not None:
            self.all_traj_feature_names = copy.deepcopy(all_traj_feature_names)
            # otherwise, we assume that the all_traj_feature_names is already in self from previous functions

        self.GUAT_nxt_ff_info_reloaded = self.combined_info['GUAT_nxt_ff_info']
        self.GUAT_cur_ff_info_reloaded = self.combined_info['GUAT_cur_ff_info']
        self.traj_data_df = self.combined_info['traj_data_df']
        self.more_traj_data_df = self.combined_info['more_traj_data_df']
        self.more_ff_df = self.combined_info['more_ff_df']

        # separate the trajectory data information
        self.traj_points_df = self.traj_data_df.loc[:,
                                                    self.all_traj_feature_names['traj_points']]
        self.traj_stops_df = self.traj_data_df.loc[:,
                                                   self.all_traj_feature_names['traj_stops']]
        self.relevant_curv_of_traj_df = self.traj_data_df.loc[:,
                                                              self.all_traj_feature_names['relevant_curv_of_traj']]
        self.relevant_curv_of_traj_df['point_index'] = self.traj_data_df['point_index'].copy(
        )
        self.more_traj_points_df = self.more_traj_data_df.loc[:,
                                                              self.all_traj_feature_names['more_traj_points']]
        self.more_traj_stops_df = self.more_traj_data_df.loc[:,
                                                             self.all_traj_feature_names['more_traj_stops']]

        self.num_stops_df = self.GUAT_cur_ff_info_reloaded[[
            'point_index', 'num_stops']].drop_duplicates().copy()

    def process_current_and_alternative_ff_info(self):
        self.GUAT_nxt_ff_info = self.GUAT_nxt_ff_info_reloaded.copy()
        self.GUAT_cur_ff_info = self.GUAT_cur_ff_info_reloaded.copy()
        super().process_current_and_alternative_ff_info(num_old_ff_per_row=self.gc_kwargs['num_old_ff_per_row'],
                                                        num_new_ff_per_row=self.gc_kwargs['num_new_ff_per_row'],
                                                        selection_criterion_if_too_many_ff=self.gc_kwargs['selection_criterion_if_too_many_ff'])
        self.GUAT_cur_ff_info = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.GUAT_cur_ff_info, curv_of_traj_df=self.relevant_curv_of_traj_df)
        self.GUAT_nxt_ff_info = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.GUAT_nxt_ff_info, curv_of_traj_df=self.relevant_curv_of_traj_df)

    def find_input_and_output(self,
                              add_arc_info=False,
                              add_current_curv_of_traj=False,
                              ff_attributes=[
                                  'ff_distance', 'ff_angle', 'time_since_last_vis', 'time_till_next_visible'],
                              add_num_ff_in_cluster=False,
                              arc_info_to_add=['curv_diff', 'abs_curv_diff'],
                              ):
        if add_current_curv_of_traj is True:
            # see if any element of all_traj_feature_names['traj_points'] contains 'curv_of_traj' even if it's just part of the string
            curv_of_traj_feature_names = [
                name for name in self.all_traj_feature_names['traj_points'] if 'curv_of_traj' in name]
            if len(curv_of_traj_feature_names) > 0:
                add_current_curv_of_traj = False
                print('Warning: add_current_curv_of_traj is set to False because \'curv_of_traj\' is already in the trajectory features.')

        self.GUAT_cur_ff_info['group'] = 'Original'
        self.GUAT_nxt_ff_info['group'] = 'Alternative'

        self.GUAT_joined_ff_info = pd.concat(
            [self.GUAT_cur_ff_info, self.GUAT_nxt_ff_info], axis=0)
        self.ff_attributes = ff_attributes.copy()
        self.attributes_for_plotting = [
            'ff_distance', 'ff_angle', 'time_since_last_vis']
        if 'time_till_next_visible' in self.ff_attributes:
            self.attributes_for_plotting.append('time_till_next_visible')

        if add_arc_info:  # use a different method to add arc info from the original way
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))

        # find free_selection_x_df and use it as input for machine learning
        (self.free_selection_x_df, self.free_selection_x_df_for_plotting, self.sequence_of_obs_ff_indices,
         self.point_index_array, self.pred_var) = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(
            self.GUAT_joined_ff_info, ff_attributes=ff_attributes,
            attributes_for_plotting=self.attributes_for_plotting,
            num_ff_per_row=self.num_old_ff_per_row + self.num_new_ff_per_row)

        if add_current_curv_of_traj:  # manually add trajectory arc info
            curv_of_traj = self.relevant_curv_of_traj_df.set_index(
                'point_index').loc[self.point_index_array, 'curv_of_traj'].values
            self.free_selection_x_df['curv_of_traj'] = curv_of_traj

        if add_num_ff_in_cluster:
            self.free_selection_x_df['num_current_ff_in_cluster'] = self.GUAT_cur_ff_info.groupby(
                'point_index').first()['num_ff_in_cluster'].values
            self.free_selection_x_df['num_nxt_ff_in_cluster'] = self.GUAT_nxt_ff_info.groupby(
                'point_index').first()['num_ff_in_cluster'].values

        self.free_selection_time = self.GUAT_joined_ff_info.set_index(
            'point_index').loc[self.point_index_array, 'time'].values
        self.num_stops = self.num_stops_df.set_index(
            'point_index').loc[self.point_index_array, 'num_stops'].values

        self.output = self.num_stops - 1
        # self.output[self.output > 2] = 2

    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=False,
                                          ):
        '''
        X_all: the input data for machine learning
        input_features: the names of the input features
        '''
        self.free_selection_point_index = self.point_index_array
        self.free_selection_labels = self.output.copy()

        self.time_range_of_trajectory = self.gc_kwargs['time_range_of_trajectory']
        self.num_time_points_for_trajectory = self.gc_kwargs['num_time_points_for_trajectory']

        super().prepare_data_for_machine_learning(kind="free selection", furnish_with_trajectory_data=furnish_with_trajectory_data,
                                                  trajectory_data_kind="position", add_traj_stops=True)

        if furnish_with_trajectory_data:  # manually add trajectory and stops info
            self.furnish_with_trajectory_data = True
            self.X_all = np.concatenate(
                [self.X_all, self.traj_points_df.values], axis=1)
            self.X_all = np.concatenate(
                [self.X_all, self.traj_stops_df.values], axis=1)
            # add the feature names
            self.input_features = np.concatenate(
                [self.input_features, self.traj_points_df.columns], axis=0)
            self.input_features = np.concatenate(
                [self.input_features, self.traj_stops_df.columns], axis=0)
            self.traj_points = self.traj_points_df.values
            self.traj_stops = self.traj_stops_df.values
        else:
            self.furnish_with_trajectory_data = False



    def add_additional_info_to_plot(self):
        self.more_ff_df = cluster_replacement_utils.eliminate_part_of_more_ff_inputs_already_in_observation(
            self.more_ff_df, self.sequence_of_obs_ff_indices, self.point_index_all)
        self.more_ff_inputs_df_for_plotting = cluster_replacement_utils.turn_more_ff_df_into_free_selection_x_df_for_plotting(
            self.more_ff_df, self.point_index_all, attributes_for_plotting=self.attributes_for_plotting)

        self.more_ff_inputs = self.more_ff_inputs_df_for_plotting.values
        self.more_traj_points = self.more_traj_points_df.values
        self.more_traj_stops = self.more_traj_stops_df.values

        self.more_ff_inputs_to_plot = self.more_ff_inputs[self.indices_test]
        self.more_traj_points_to_plot = self.more_traj_points[self.indices_test]
        self.more_traj_stops_to_plot = self.more_traj_stops[self.indices_test]
