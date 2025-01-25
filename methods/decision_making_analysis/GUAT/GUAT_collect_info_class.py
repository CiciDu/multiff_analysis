from decision_making_analysis.GUAT import GUAT_and_TAFT, GUAT_collect_info_helper_class
from decision_making_analysis import trajectory_info
from null_behaviors import curvature_utils
from data_wrangling import base_processing_class

import os
import copy
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from os.path import exists
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class GUATCollectInfoForSession(GUAT_collect_info_helper_class.GUATCollectInfoHelperClass):


    def __init__(self, raw_data_folder_path, gc_kwargs, new_point_index_start=0):

        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self, raw_data_folder_path)
        self.GUAT_folder_path = os.path.join(self.decision_making_folder_path, 'GUAT_info')
        self.gc_kwargs = gc_kwargs
        self.trajectory_features = gc_kwargs['trajectory_features']
        self.new_point_index_start = new_point_index_start

        os.makedirs(self.GUAT_folder_path, exist_ok=True)


    def streamline_process_to_collect_info_from_one_session(self, 
                                                            monkey_data_already_retrieved_ok=True,
                                                            GUAT_info_exists_ok=True,
                                                            curv_of_traj_df_exists_ok=True, 
                                                            GUAT_w_ff_df_exists_ok=True, 
                                                            update_point_index=True,
                                                            save_data=True,
                                                            ):
        try:
            if not GUAT_info_exists_ok:
                raise Exception('GUAT_info_exists_ok is False. Proceed to collect all GUAT info from one session.')
            elif not curv_of_traj_df_exists_ok:
                raise Exception('curv_of_traj_df_exists_ok is False. Proceed to collect all GUAT info from one session.')
            elif not GUAT_w_ff_df_exists_ok:
                raise Exception('GUAT_w_ff_df_exists_ok is False. Proceed to collect all GUAT info from one session.')
            self._try_retrieve_all_GUAT_info_from_one_session()
        except Exception as e:
            print('Abort retrieving all GUAT info from one session:', e, 'Proceed to collect all GUAT info from one session.')
            self.get_monkey_data(already_retrieved_ok=monkey_data_already_retrieved_ok)
            self._add_one_stop_info_to_GUAT_w_ff_df()
            self._generate_important_df_to_combine_across_sessions(self.GUAT_w_ff_df, curv_of_traj_df_exists_ok=curv_of_traj_df_exists_ok, **self.gc_kwargs)
            if save_data:
                self._save_important_info()  

        self._compile_important_info()
        if update_point_index:
            self._update_point_index_of_important_df_in_important_info()


        return self.important_info

    def _try_retrieve_all_GUAT_info_from_one_session(self,
                                                    df_names=['GUAT_alt_ff_info', 'GUAT_current_ff_info', 'traj_data_df', 'more_traj_data_df', 'more_ff_df', 'traj_ff_info',
                                                              'curv_of_traj_df'
                                                              'GUAT_w_ff_df'],
                                                    ):

        for df in df_names:
            if df != 'GUAT_w_ff_df':
                setattr(self, df, pd.read_csv(os.path.join(self.GUAT_folder_path, df + '.csv')).drop(["Unnamed: 0"], axis=1))
            else:
                setattr(self, df, pd.read_csv(os.path.join(self.GUAT_folder_path, df + '.csv'), sep='\t', converters={'nearby_alive_ff_indices': pd.eval}))


    def _generate_important_df_to_combine_across_sessions(self,
                                                         GUAT_w_ff_df,
                                                         time_with_respect_to_first_stop=-0.1,
                                                         time_with_respect_to_second_stop=None,
                                                         time_with_respect_to_last_stop=None,
                                                         n_seconds_before_crossing_boundary=2,
                                                         n_seconds_after_crossing_boundary=None,
                                                         max_cluster_distance=50,
                                                         max_distance_to_stop=400,
                                                         max_distance_to_stop_for_GUAT_target=50,
                                                         max_time_since_last_vis = 2.5,
                                                         columns_to_sort_alt_ff_by=['abs_curv_diff', 'time_since_last_vis'],
                                                         curv_of_traj_mode='time',
                                                         window_for_curv_of_traj=[-1, 1], 
                                                         truncate_curv_of_traj_by_time_of_capture=False,
                                                         time_range_of_trajectory = [-1,1],
                                                         num_time_points_for_trajectory = 10,
                                                         time_range_of_trajectory_to_plot=None,
                                                         num_time_points_for_trajectory_to_plot=10,
                                                         include_ff_in_near_future=True,
                                                         duration_into_future=0.5, 
                                                         trajectory_features=['monkey_distance', 'monkey_angle_to_origin'],
                                                         last_seen_and_next_seen_attributes_to_add=['ff_distance', 'ff_angle', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y'],
                                                         curv_of_traj_df_exists_ok=True,
                                                         **kwargs
                                                         ):


        self.time_range_of_trajectory = time_range_of_trajectory
        self.gc_kwargs['num_time_points_for_trajectory'] = num_time_points_for_trajectory
        self.gc_kwargs['time_range_of_trajectory_to_plot'] = time_range_of_trajectory_to_plot
        self.gc_kwargs['num_time_points_for_trajectory_to_plot'] = num_time_points_for_trajectory_to_plot

        self.set_time_of_eval(GUAT_w_ff_df, time_with_respect_to_first_stop=time_with_respect_to_first_stop, time_with_respect_to_second_stop=time_with_respect_to_second_stop, time_with_respect_to_last_stop=time_with_respect_to_last_stop)
        self.eliminate_crossing_boundary_cases(n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary)

        self.make_or_retrieve_curv_of_traj_df(exists_ok=curv_of_traj_df_exists_ok, curv_of_traj_mode=curv_of_traj_mode, window_for_curv_of_traj=window_for_curv_of_traj, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        self.add_curv_of_traj_info_to_monkey_information(column_exists_ok=True)
        ff_dataframe_sub = self.ff_dataframe.loc[self.ff_dataframe['point_index'].isin(self.GUAT_w_ff_df['point_index_of_eval'].values)]
        self.curvature_df = curvature_utils.make_curvature_df(ff_dataframe_sub, self.curv_of_traj_df)
        self.add_curvature_info_to_ff_dataframe(self)

        self.trajectory_features = trajectory_features
        self.max_distance_to_stop_for_GUAT_target = max_distance_to_stop_for_GUAT_target

        self.get_current_ff_info_and_alt_ff_info_for_info_collection(max_cluster_distance=max_cluster_distance, max_time_since_last_vis=max_time_since_last_vis, include_ff_in_near_future=include_ff_in_near_future, \
                                                                     duration_into_future=duration_into_future, columns_to_sort_alt_ff_by=columns_to_sort_alt_ff_by, last_seen_and_next_seen_attributes_to_add=last_seen_and_next_seen_attributes_to_add,
                                                                     max_distance_to_stop=max_distance_to_stop)
        
        self.get_trajectory_and_stop_info_for_machine_learning()
        self.get_more_trajectory_info_for_plotting(time_range_of_trajectory_to_plot=time_range_of_trajectory_to_plot, num_time_points_for_trajectory_to_plot=num_time_points_for_trajectory_to_plot)
        self.get_traj_ff_info()

        if include_ff_in_near_future:
            self.update_curvature_df_with_additional_curvature_df()

        relevant_curv_of_traj = trajectory_info.find_trajectory_arc_info(self.point_index_all, self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new, monkey_information=self.monkey_information)
        self.relevant_curv_of_traj_df = pd.DataFrame({'point_index': self.point_index_all, 'curv_of_traj': relevant_curv_of_traj})

        # combine the trajectory data
        self.traj_data_df, self.traj_data_feature_names = trajectory_info.combine_trajectory_and_stop_info_and_curvature_info(self.traj_points_df, self.traj_stops_df, self.relevant_curv_of_traj_df)
        self.more_traj_data_df, self.more_traj_data_feature_names = trajectory_info.combine_trajectory_and_stop_info_and_curvature_info(self.more_traj_points_df, self.more_traj_stops_df, self.relevant_curv_of_traj_df)
        # rename the keys in self.more_traj_data_feature_names
        self.traj_data_df['point_index'] = self.point_index_all
        self.more_traj_data_df['point_index'] = self.point_index_all

        self.more_ff_df = self.get_more_ff_df()
        self.add_arc_info_to_each_df_of_ff_info(self.curvature_df)

    def streamline_process_to_collect_traj_data_only(self, point_index_all, 
                                                    curv_of_traj_df_exists_ok=True):
        

        self.time_range_of_trajectory = self.gc_kwargs['time_range_of_trajectory']
        self.trajectory_features = self.gc_kwargs['trajectory_features']

        self.point_index_all = point_index_all

        self.get_monkey_data(include_ff_dataframe=False, include_GUAT_data=False)
        self.make_or_retrieve_curv_of_traj_df(exists_ok=curv_of_traj_df_exists_ok)
        self.add_curv_of_traj_info_to_monkey_information(column_exists_ok=True)

        self.time_all = self.monkey_information.loc[self.point_index_all, 'time'].values
        
        self.get_trajectory_and_stop_info_for_machine_learning()
        self.get_more_trajectory_info_for_plotting(time_range_of_trajectory_to_plot=self.gc_kwargs['time_range_of_trajectory_to_plot'], num_time_points_for_trajectory_to_plot=self.gc_kwargs['num_time_points_for_trajectory_to_plot'])

        relevant_curv_of_traj = trajectory_info.find_trajectory_arc_info(self.point_index_all, self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new, monkey_information=self.monkey_information)
        self.relevant_curv_of_traj_df = pd.DataFrame({'point_index': self.point_index_all, 'curv_of_traj': relevant_curv_of_traj})

        # combine the trajectory data
        self.traj_data_df, self.traj_data_feature_names = trajectory_info.combine_trajectory_and_stop_info_and_curvature_info(self.traj_points_df, self.traj_stops_df, self.relevant_curv_of_traj_df)
        self.more_traj_data_df, self.more_traj_data_feature_names = trajectory_info.combine_trajectory_and_stop_info_and_curvature_info(self.more_traj_points_df, self.more_traj_stops_df, self.relevant_curv_of_traj_df)
        # rename the keys in self.more_traj_data_feature_names
        self.traj_data_df['point_index'] = self.point_index_all
        self.more_traj_data_df['point_index'] = self.point_index_all



    def _compile_important_info(self):
        important_info = {'GUAT_alt_ff_info': self.GUAT_alt_ff_info,
                        'GUAT_current_ff_info': self.GUAT_current_ff_info,
                        'traj_data_df': self.traj_data_df,
                        'more_traj_data_df': self.more_traj_data_df,
                        'more_ff_df': self.more_ff_df,
                        'traj_ff_info': self.traj_ff_info,
        }
        print('The following are the keys of the dictionary of important info:')
        print(important_info.keys())
        self.important_info = copy.deepcopy(important_info)

        try:
            self.all_traj_feature_names = trajectory_info.make_all_traj_feature_names(self.gc_kwargs['time_range_of_trajectory'],
                                                                                      self.gc_kwargs['num_time_points_for_trajectory'], 
                                                                                      self.gc_kwargs['time_range_of_trajectory_to_plot'], 
                                                                                    self.gc_kwargs['num_time_points_for_trajectory_to_plot'], 
                                                                                    traj_point_features=self.gc_kwargs['trajectory_features'])
        except AttributeError:
            pass


    def _save_important_info(self,
                            df_names=['GUAT_alt_ff_info', 'GUAT_current_ff_info', 'traj_data_df', 'more_traj_data_df', 'more_ff_df', 'traj_ff_info', 'curv_of_traj_df']):
        if not exists(self.GUAT_folder_path): 
            os.makedirs(self.GUAT_folder_path)
        for df in df_names:
            getattr(self, df).to_csv(os.path.join(self.GUAT_folder_path, df + '.csv'))


    def _update_point_index_of_important_df_in_important_info(self):
        self.important_info, self.point_index_to_new_number_df = GUAT_and_TAFT.update_point_index_of_important_df_in_important_info_func(self.important_info, self.new_point_index_start)


