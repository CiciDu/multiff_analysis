import sys
from decision_making_analysis.cluster_replacement import cluster_replacement_utils
from decision_making_analysis.GUAT import process_GUAT_trials_class, GUAT_and_TAFT, GUAT_collect_info_helper_class, GUAT_collect_info_class, GUAT_utils
from decision_making_analysis import trajectory_info
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.only_stop_ff import only_stop_ff_utils, features_to_keep_utils
from planning_analysis.show_planning import alt_ff_utils, show_planning_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, stops_near_ff_based_on_ref_class
from planning_analysis.plan_factors import plan_factors_utils
from null_behaviors import curvature_utils, curvature_class, curv_of_traj_utils
from data_wrangling import monkey_data_classes, basic_func, base_processing_class
from non_behavioral_analysis import eye_positions

import os
import copy
import numpy as np
import math
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from os.path import exists
import pandas as pd
import json


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


gc_kwargs = {'time_with_respect_to_first_stop': -0.1,
             'time_with_respect_to_second_stop': None,
             'time_with_respect_to_last_stop': None,
             'n_seconds_before_crossing_boundary': 2.5, 
             'n_seconds_after_crossing_boundary': 2.5,
             
             'max_cluster_distance': 100, # max distance from other ff to the current ff to be considered as in the same cluster
             'max_distance_to_stop': 400,
             'max_distance_to_stop_for_GUAT_target': 50,
             'columns_to_sort_alt_ff_by': ['distance_to_monkey'], # originally using ['abs_curv_diff', 'time_since_last_vis'],        
             'selection_criterion_if_too_many_ff': 'distance_to_monkey',

             'num_old_ff_per_row': 2, # originally it was 2
             'num_new_ff_per_row': 2, # originally it was 2

             'last_seen_and_next_seen_attributes_to_add':['ff_distance', 'ff_angle', 'curv_diff', 'abs_curv_diff', 'monkey_x', 'monkey_y'],
             
             'curv_of_traj_mode': 'time',
             'window_for_curv_of_traj': [-1, 1], 
             'truncate_curv_of_traj_by_time_of_capture': False,
             
             'time_range_of_trajectory': [-2.5,0], # original [-2, 0]
             'num_time_points_for_trajectory': 11, # originally 8
             'time_range_of_trajectory_to_plot': [0, 10], # original [-2, 5]
             'num_time_points_for_trajectory_to_plot': 41,
             'trajectory_features': ['monkey_distance', 'monkey_angle_to_origin', 'monkey_t', 'curvature_of_traj'],

             'include_ff_in_near_future': True,
             'max_time_since_last_vis': 2.5,
             'duration_into_future': 0.5, 
 }



class CompareGUATandTAFTclass():
    

    def __init__(self, 
                 raw_data_folder_path='all_monkey_data/raw_monkey_data/individual_monkey_data/monkey_Bruno/data_0330',
                 ref_point_mode='distance', 
                 ref_point_value=-150,
                 stop_period_duration=2):
        
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self, raw_data_folder_path)
        self.curv_of_traj_df = None
        self.curv_of_traj_df_w_one_sided_window = None
        self.use_curvature_to_ff_center = False
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.stop_period_duration = stop_period_duration


    def streamline_getting_x_df(self, TAFT_or_GUAT='TAFT',
                                    save_data=True,
                                    exists_ok=True,
                                    GUAT_w_ff_df_exists_ok=True,
                                    ):
        self.TAFT_or_GUAT = TAFT_or_GUAT
        # delete self.stops_near_ff_df
        if hasattr(self, 'stops_near_ff_df'):
            del self.stops_near_ff_df
        
        self.GUAT_vs_TAFT_folder_path = os.path.join(self.decision_making_folder_name, 'GUAT_vs_TAFT')
        os.makedirs(self.GUAT_vs_TAFT_folder_path, exist_ok=True)

        if exists_ok:
            try:
                self.try_retrieving_x_df()
                return
            except FileNotFoundError:
                pass

        self.get_relevant_monkey_data(raw_data_folder_path=self.raw_data_folder_path, GUAT_w_ff_df_exists_ok=GUAT_w_ff_df_exists_ok)
        self.process_trials_df()

        self.get_stops_near_ff_df(already_made_ok=True)
        self.make_plan_y_df()
        self.make_plan_x_df(list_of_stop_ff_cluster_radius=[100],
                            list_of_alt_ff_cluster_radius=[200])
        
        self.get_x_features_df(list_of_stop_ff_cluster_radius=[],
                                list_of_stop_ff_ang_cluster_radius=[20],
                                list_of_start_dist_cluster_radius=[100],
                                list_of_start_ang_cluster_radius=[20],
                                list_of_flash_cluster_period=[[1.0, 1.5], [1.5, 2.0]])
        self.make_only_stop_ff_df()
        
        self.get_x_df(save_data=save_data)

        self.y_var_df = self.x_df[['stop_point_index']].copy()
        self.y_var_df['whether_TAFT'] = 1 if TAFT_or_GUAT == 'TAFT' else 0


    def get_x_and_y_var_df(self):
        self.x_var_df = pd.concat([self.TAFT_x_df, self.GUAT_x_df], axis=0).reset_index(drop=True).drop(columns=['stop_point_index'])
        self.x_var_df['dir_from_stop_ff_same_side'] = self.x_var_df['dir_from_stop_ff_same_side'].astype(int)
        self.y_var_df = pd.DataFrame(np.array([1]*len(self.TAFT_x_df) + [0]*len(self.GUAT_x_df)).reshape(-1, 1), columns=['y_var'])

        columns_with_na = self.x_var_df.columns[self.x_var_df.isna().any()].tolist()
        print(f'There are {len(columns_with_na)} columns with NA that are dropped. {self.x_var_df.shape[1]} columns are left. The dropped columns with number of NA are:')
        print(self.x_var_df[columns_with_na].isna().sum())
        self.x_var_df = self.x_var_df.drop(columns=columns_with_na)



    def try_retrieving_x_df(self):
        if self.TAFT_or_GUAT == 'TAFT':
            if (os.path.exists(os.path.join(self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'))):
                self.TAFT_x_df = pd.read_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'))
                return
            else:
                raise FileNotFoundError('TAFT_x_df.csv does not exist')
        elif self.TAFT_or_GUAT == 'GUAT':
            if (os.path.exists(os.path.join(self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'))):
                self.GUAT_x_df = pd.read_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'))
                return
            else:
                raise FileNotFoundError('GUAT_x_df.csv does not exist')
        else:
            raise ValueError('TAFT_or_GUAT must be either TAFT or GUAT')
            

    def get_relevant_monkey_data(self,
                                 already_retrieved_ok=True,
                                 GUAT_w_ff_df_exists_ok=True,
                                raw_data_folder_path='all_monkey_data/raw_monkey_data/individual_monkey_data/monkey_Bruno/data_0330'
                                ):
        self.monkey_name = os.path.basename(raw_data_folder_path)
        if not hasattr(self, 'gcc'):
            self.gcc = GUAT_collect_info_class.GUATCollectInfoForSession(raw_data_folder_path=raw_data_folder_path, 
                                                                    gc_kwargs=gc_kwargs, new_point_index_start=0)

        include_TAFT_data = False
        if self.TAFT_or_GUAT == 'TAFT':
            include_TAFT_data = True
        self.gcc.get_monkey_data(already_retrieved_ok=already_retrieved_ok, include_TAFT_data=include_TAFT_data)
        
        for df in ['monkey_information', 'ff_dataframe', 'ff_flash_sorted', 'ff_real_position_sorted', 'ff_life_sorted',
                   'ff_believed_position_sorted', 'ff_caught_T_new']:
            setattr(self, df, getattr(self.gcc.data_item, df).copy())
        if self.TAFT_or_GUAT == 'TAFT':
            self.TAFT_trials_df = self.gcc.data_item.TAFT_trials_df.copy()
            self.TAFT_trials_df['first_stop_time'] = self.monkey_information.loc[self.TAFT_trials_df['first_stop_point_index'], 'time'].values
            self.TAFT_trials_df['ff_index'] = self.TAFT_trials_df['trial']
            # because we need to have alt_ff, we will limit the max number of ff_index to len(self.ff_caught_T_new - 2)
            self.TAFT_trials_df = self.TAFT_trials_df[self.TAFT_trials_df['ff_index'] < len(self.ff_caught_T_new) - 2]
            GUAT_vs_TAFT_utils.add_stop_point_index(self.TAFT_trials_df, self.monkey_information, self.ff_real_position_sorted)
        elif self.TAFT_or_GUAT == 'GUAT':
            self.gcc.make_or_retrieve_GUAT_w_ff_df(exists_ok=GUAT_w_ff_df_exists_ok)
            self.GUAT_w_ff_df = self.gcc.GUAT_w_ff_df.copy()
            self.GUAT_w_ff_df.sort_values(by=['trial', 'first_stop_time'], inplace=True)
            self.GUAT_w_ff_df['ff_index'] = self.GUAT_w_ff_df['latest_visible_ff']
            GUAT_vs_TAFT_utils.add_stop_point_index(self.GUAT_w_ff_df, self.monkey_information, self.ff_real_position_sorted)
            self.GUAT_w_ff_df = GUAT_vs_TAFT_utils.deal_with_duplicated_stop_point_index(self.GUAT_w_ff_df)
        self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible'] == 1]


    def process_trials_df(self):

        if self.TAFT_or_GUAT == 'TAFT':
            self.get_TAFT_df()
            self.get_TAFT_df2_based_on_ref_point()
        elif self.TAFT_or_GUAT == 'GUAT':
            self.get_GUAT_df()
            self.get_GUAT_df2_based_on_ref_point()
        else:
            raise ValueError('TAFT_or_GUAT must be either TAFT or GUAT')
        


    def get_TAFT_df(self):
        self.TAFT_df = GUAT_vs_TAFT_utils.process_trials_df(self.TAFT_trials_df, self.monkey_information, self.ff_dataframe_visible, self.stop_period_duration)
        self.TAFT_df['stop_ff_capture_time'] = self.ff_caught_T_new[self.TAFT_df['ff_index'].values]

    def get_GUAT_df(self):
        self.GUAT_df = GUAT_vs_TAFT_utils.process_trials_df(self.GUAT_w_ff_df, self.monkey_information, self.ff_dataframe_visible, self.stop_period_duration)


    def get_TAFT_df2_based_on_ref_point(self):
        self.TAFT_df2 = GUAT_vs_TAFT_utils.further_make_trials_df(self.TAFT_df, self.monkey_information, self.ff_real_position_sorted, self.stop_period_duration, self.ref_point_mode, self.ref_point_value)


    def get_GUAT_df2_based_on_ref_point(self):
        self.GUAT_df2 = GUAT_vs_TAFT_utils.further_make_trials_df(self.GUAT_df, self.monkey_information, self.ff_real_position_sorted, self.stop_period_duration, self.ref_point_mode, self.ref_point_value)

    
    def _add_alt_ff_index(self):
        if self.TAFT_or_GUAT == 'TAFT':
            self.stops_near_ff_df['alt_ff_index'] = self.stops_near_ff_df['stop_ff_index'] + 1
        else:
            self.stops_near_ff_df['alt_ff_index'] = np.searchsorted(self.ff_caught_T_new, self.stops_near_ff_df['stop_time'].values)
            
            # check if the alt_ff_index is correct (can delete the lines below later)
            self.stops_near_ff_df['alt_ff_caught_time'] = self.ff_caught_T_new[self.stops_near_ff_df['alt_ff_index'].values]
            # see if any element of self.stops_near_ff_df['alt_ff_caught_time'] - self.stops_near_ff_df['stop_time'] is smaller than 0
            # if there is, then the alt_ff_index is not correct
            if np.any(self.stops_near_ff_df['alt_ff_caught_time'] - self.stops_near_ff_df['stop_time'] < 0):
                raise ValueError('alt_ff_index is not correct')


    def get_stops_near_ff_df(self, 
                             already_made_ok=True, exists_ok=True, save_data=True):
        # if already_made_ok & (getattr(self, 'stops_near_ff_df', None) is not None):
        #     return

        if already_made_ok & (getattr(self, 'stops_near_ff_df', None) is not None):
            return

        self.stops_near_ff_df = self.TAFT_df.copy() if (self.TAFT_or_GUAT == 'TAFT') else self.GUAT_df.copy()

        self.stops_near_ff_df.rename(columns={'ff_index': 'stop_ff_index'}, inplace=True)
        self.stops_near_ff_df[['stop_x', 'stop_y', 'monkey_angle', 'stop_time', 'stop_cum_distance']] = self.monkey_information.loc[self.stops_near_ff_df['stop_point_index'], ['monkey_x', 'monkey_y', 'monkey_angle', 'monkey_t', 'cum_distance']].values
        self.stops_near_ff_df.rename(columns={'ff_index': 'stop_ff_index',
                                        'monkey_angle': 'stop_monkey_angle',}, inplace=True)

        # note that this only works for TAFT, but doesn't work for GUAT
        self._add_alt_ff_index()
        self.stops_near_ff_df['alt_ff_caught_time'] = self.ff_caught_T_new[self.stops_near_ff_df['alt_ff_index'].values]

        closest_stop_to_capture_df2 = self.closest_stop_to_capture_df[['stop_ff_index', 'stop_point_index', 'stop_time']].copy()
        closest_stop_to_capture_df2.rename(columns={'stop_ff_index': 'alt_ff_index',
                                                    'stop_time': 'next_stop_time',
                                                    'stop_point_index': 'next_stop_point_index'}, inplace=True)
        all_closest_point_to_alt_ff = self.closest_stop_to_capture_df.merge(closest_stop_to_capture_df2, on='alt_ff_index', how='left')


        # self.stops_near_ff_df['next_stop_time'] = self.ff_caught_T_new[self.stops_near_ff_df['alt_ff_index'].values]
        # self.stops_near_ff_df['next_stop_point_index'] = np.searchsorted(self.monkey_information['time'].values, self.stops_near_ff_df['next_stop_time'].values)
        self.stops_near_ff_df[['next_stop_point_index', 'next_stop_time']] = all_closest_point_to_alt_ff[['point_index', 'time']].values
        self.stops_near_ff_df['next_stop_cum_distance'] = self.monkey_information.loc[self.stops_near_ff_df['next_stop_point_index'], 'cum_distance'].values

        self.stops_near_ff_df[['stop_ff_x', 'stop_ff_y']] = self.ff_real_position_sorted[self.stops_near_ff_df['stop_ff_index'].values]
        self.stops_near_ff_df[['alt_ff_x', 'alt_ff_y']] = self.ff_real_position_sorted[self.stops_near_ff_df['alt_ff_index'].values]


        self.stops_near_ff_df = alt_ff_utils.add_alt_ff_first_and_last_seen_info(self.stops_near_ff_df, self.ff_dataframe_visible, self.monkey_information, self.ff_real_position_sorted, self.ff_life_sorted)

        self.stops_near_ff_df = alt_ff_utils.add_if_alt_ff_and_alt_ff_cluster_flash_bbas(self.stops_near_ff_df, self.ff_real_position_sorted, 
                                                                                        self.ff_flash_sorted, self.ff_life_sorted, stop_period_duration=self.stop_period_duration)
        self.stops_near_ff_df = alt_ff_utils.add_if_alt_ff_and_alt_ff_cluster_flash_bsans(self.stops_near_ff_df, self.ff_real_position_sorted, 
                                                                                        self.ff_flash_sorted, self.ff_life_sorted)

        find_stops_near_ff_utils.add_stop_ff_cluster_50_size(self.stops_near_ff_df, self.ff_real_position_sorted, self.ff_life_sorted,
                                                                                     empty_cluster_ok=True)

        self.stops_near_ff_df = alt_ff_utils._add_stop_or_alt_ff_first_seen_and_last_seen_info_bbas(self.stops_near_ff_df, self.ff_dataframe_visible, self.monkey_information, stop_or_alt='stop')

        ## The below is not needed rn since it will filter out rows based on visibility of stop or alt ff
        #self.stops_near_ff_df = find_stops_near_ff_utils.process_shared_stops_near_ff_df(self.stops_near_ff_df)

        self.stops_near_ff_df = find_stops_near_ff_utils.add_monkey_info_before_stop(self.monkey_information, self.stops_near_ff_df)
        self._add_curv_of_traj_stat_df()

        self.stops_near_ff_df['cum_distance_between_two_stops'] = self.stops_near_ff_df['next_stop_cum_distance'] - self.stops_near_ff_df['stop_cum_distance']
        self.stops_near_ff_df['d_from_stop_ff_to_alt_ff'] = np.linalg.norm(self.ff_real_position_sorted[self.stops_near_ff_df['stop_ff_index'].values] - 
                                                                           self.ff_real_position_sorted[self.stops_near_ff_df['alt_ff_index'].values], axis=1)
        self.stops_near_ff_df['data_category_by_vis'] = 'test'
        


    def make_ff_info_at_start_df(self):
        df = self.TAFT_df2 if (self.TAFT_or_GUAT == 'TAFT') else self.GUAT_df2
        if len(self.stops_near_ff_df) != len(df):
            raise ValueError('The length of stops_near_ff_df and df must be the same')

        self.monkey_info_in_all_stop_periods = only_stop_ff_utils.make_monkey_info_in_all_stop_periods(self.stops_near_ff_df, self.monkey_information, stop_period_duration=self.stop_period_duration,
                                                                                                       all_end_time=self.stops_near_ff_df['next_stop_time'], all_start_time=self.stops_near_ff_df['beginning_time']
                                                                                                       )
        self.ff_info_at_start_df, self.stop_ff_info_at_start_df = only_stop_ff_utils.find_ff_info_and_stop_ff_info_at_start_df(df, self.monkey_info_in_all_stop_periods, self.ff_flash_sorted, 
                                                                                                        self.ff_real_position_sorted, self.ff_life_sorted, ff_radius=10,
                                                                                                        guarantee_info_for_stop_ff=True, dropna=False)
        

    def get_x_features_df(self,
                        list_of_stop_ff_cluster_radius=[100],
                        list_of_stop_ff_ang_cluster_radius=[20],
                        list_of_start_dist_cluster_radius=[100],
                        list_of_start_ang_cluster_radius=[20],
                        list_of_flash_cluster_period=[[1.5, 2.0]]                          
                          ):
        self.make_ff_info_at_start_df()

        # then I can get x_features!
        self.x_features_df, self.all_cluster_names = only_stop_ff_utils.get_x_features_df(self.ff_info_at_start_df, self.stop_ff_info_at_start_df,
                                                                                        list_of_stop_ff_cluster_radius=list_of_stop_ff_cluster_radius,
                                                                                        list_of_stop_ff_ang_cluster_radius=list_of_stop_ff_ang_cluster_radius,
                                                                                        list_of_start_dist_cluster_radius=list_of_start_dist_cluster_radius,
                                                                                        list_of_start_ang_cluster_radius=list_of_start_ang_cluster_radius,
                                                                                        list_of_flash_cluster_period=list_of_flash_cluster_period,
                                                                                        flash_or_vis=None,
                                                                                        )


    def make_only_stop_ff_df(self):
        self.only_stop_ff_df = only_stop_ff_utils.get_only_stop_ff_df(self.stops_near_ff_df, self.ff_real_position_sorted, self.ff_caught_T_new, self.monkey_information, 
                                                                        self.curv_of_traj_df, self.ff_dataframe_visible, stop_period_duration=self.stop_period_duration,
                                                                        ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value)

    

    def make_plan_y_df(self, exists_ok=True, heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        test_or_ctrl = 'test'

        self.make_heading_info_df(
            use_curvature_to_ff_center=self.use_curvature_to_ff_center,
            heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            save_data=save_data
        )
        setattr(self, f'{test_or_ctrl}_heading_info_df', self.heading_info_df)
        stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef._make_curv_of_traj_df_if_not_already_made(self)
        plan_factors_class.PlanFactors._make_curv_of_traj_df_w_one_sided_window_if_not_already_made(self)

        self.plan_y_df = plan_factors_utils.make_plan_y_df(self.heading_info_df, self.curv_of_traj_df, self.curv_of_traj_df_w_one_sided_window)

        # time_columns = ['ALT_time_ff_last_seen_bbas',
        #         'ALT_time_ff_last_seen_bsans',
        #         'alt_ff_last_flash_time_bbas',
        #         'alt_ff_last_flash_time_bsans',
        #         'alt_ff_cluster_last_seen_time_bbas',
        #         'alt_ff_cluster_last_seen_time_bsans',
        #         'alt_ff_cluster_last_flash_time_bbas',
        #         'alt_ff_cluster_last_flash_time_bsans']

        # for col in time_columns:
        #     self.plan_y_df[f'{col}_rel_to_stop'] = self.plan_y_df[col] - self.plan_y_df['stop_time']

        return 


    def make_plan_x_df(self, use_eye_data=True, use_speed_data=True, ff_radius=10,
                        list_of_stop_ff_cluster_radius=[100, 200, 300],
                        list_of_alt_ff_cluster_radius=[100, 200, 300],                        
                        ):
        
        self.get_stops_near_ff_df(already_made_ok=True)

        self.both_ff_at_ref_df = self.get_both_ff_at_ref_df()
        self.both_ff_at_ref_df['stop_point_index'] = self.alt_ff_df2['stop_point_index'].values
        #self.alt_ff_at_stop_df = self.get_alt_ff_at_stop_df()
        #self.both_ff_when_seen_df = self.get_both_ff_when_seen_df(deal_with_rows_with_big_ff_angles=False)

        if self.ff_dataframe is None: 
            stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef.get_more_monkey_data(self)  

        if getattr(self, 'alt_ff_df', None) is None:
            self.get_stops_near_ff_df(already_made_ok=True, exists_ok=True, save_data=True)
            self.alt_ff_df, self.stop_ff_df = alt_ff_utils.get_alt_ff_df_and_stop_ff_df(self.stops_near_ff_df)
            self.alt_ff_df2, self.stop_ff_df2 = self.find_alt_ff_df_2_and_stop_ff_df_2()

        self.plan_x_df = plan_factors_utils.make_plan_x_df(self.stops_near_ff_df, self.heading_info_df, self.both_ff_at_ref_df, self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted,
                                                        stop_period_duration=self.stop_period_duration, ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value, ff_radius=ff_radius,
                                                        list_of_stop_ff_cluster_radius=list_of_stop_ff_cluster_radius, list_of_alt_ff_cluster_radius=list_of_alt_ff_cluster_radius,
                                                        use_speed_data=use_speed_data, use_eye_data=use_eye_data,
                                                        guarantee_stop_ff_info_for_cluster=True,
                                                        guarantee_alt_ff_info_for_cluster=True,
                                                        flash_or_vis=None,
                                                        )
        
        return self.plan_x_df


    def make_heading_info_df(self,
                             use_curvature_to_ff_center=False, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                             save_data=True, merge_diff_in_curv_df_to_heading_info=True):


        self.get_stops_near_ff_df(already_made_ok=True, exists_ok=stops_near_ff_df_exists_ok, save_data=True)
        self._get_alt_ff_and_stop_ff_info_based_on_ref_point()
        self.alt_and_stop_ff_df = show_planning_utils.make_alt_and_stop_ff_df(self.alt_ff_final_df, self.stop_ff_final_df)

        self.heading_info_df = show_planning_utils.make_heading_info_df(self.alt_and_stop_ff_df, self.stops_near_ff_df_modified, self.monkey_information, self.ff_real_position_sorted)


    def _add_curv_of_traj_stat_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        if self.curv_of_traj_df is None:
            self.curv_of_traj_df, self.traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, 
                                                                                                                        curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False)
        self.curv_of_traj_stat_df, self.stops_near_ff_df = plan_factors_utils.find_curv_of_traj_stat_df(self.stops_near_ff_df, self.curv_of_traj_df)      


    def _get_alt_ff_and_stop_ff_info_based_on_ref_point(self):
        self.stops_near_ff_df, self.alt_ff_df, self.stop_ff_df = alt_ff_utils.get_alt_ff_df_and_stop_ff_df(self.stops_near_ff_df)
        self.find_alt_ff_df_2_and_stop_ff_df_2() 
        self.alt_ff_df_modified = self.alt_ff_df2.copy()
        self.stop_ff_df_modified = self.stop_ff_df2.copy()
        self.stop_point_index_modified = self.alt_ff_df_modified.stop_point_index.values.copy()
        self.stops_near_ff_df_modified = self.stops_near_ff_df.copy()

        self.alt_ff_df2 = self.alt_ff_df2.merge(self.curv_of_traj_df[['point_index', 'curvature_of_traj']], on='point_index', how='left')
        self.stop_ff_df2 = self.stop_ff_df2.merge(self.curv_of_traj_df[['point_index', 'curvature_of_traj']], on='point_index', how='left')
        
        self.alt_ff_final_df = self.alt_ff_df2.copy()
        self.stop_ff_final_df = self.stop_ff_df2.copy()

        self.alt_ff_final_df = self.alt_ff_final_df.merge(self.stops_near_ff_df[['stop_point_index', 'monkey_angle_before_stop']], how='left')
        self.alt_ff_final_df['d_heading_of_traj'] = self.alt_ff_final_df['monkey_angle_before_stop'] - self.alt_ff_final_df['monkey_angle']
        self.alt_ff_final_df['d_heading_of_traj'] = find_stops_near_ff_utils.confine_angle_to_within_one_pie(self.alt_ff_final_df['d_heading_of_traj'].values)
        self.stop_ff_final_df = self.stop_ff_final_df.merge(self.stops_near_ff_df[['stop_point_index', 'monkey_angle_before_stop']], how='left')
        self.stop_ff_final_df['d_heading_of_traj'] = self.stop_ff_final_df['monkey_angle_before_stop'] - self.stop_ff_final_df['monkey_angle']
        self.stop_ff_final_df['d_heading_of_traj'] = find_stops_near_ff_utils.confine_angle_to_within_one_pie(self.stop_ff_final_df['d_heading_of_traj'].values)


    def find_alt_ff_df_2_and_stop_ff_df_2(self): 

        # then get the actual alt_ff_df2 and stop_ff_df2
        self.alt_ff_df2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(self.alt_ff_df, self.monkey_information, self.ff_real_position_sorted,
                                                                              ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                                              point_index_stop_ff_first_seen=self.stop_ff_df['point_index_ff_first_seen'].values)
        self.stop_ff_df2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(self.stop_ff_df, self.monkey_information, self.ff_real_position_sorted,
                                                                              ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value)

        return self.alt_ff_df2, self.stop_ff_df2
    

    def get_both_ff_at_ref_df(self):
        self.alt_ff_df2, self.stop_ff_df2 = self.find_alt_ff_df_2_and_stop_ff_df_2()
        self.both_ff_at_ref_df = self.alt_ff_df2[['ff_distance', 'ff_angle']].copy()
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'alt_ff_distance_at_ref',
                                                  'ff_angle': 'alt_ff_angle_at_ref'}, inplace=True)
        self.both_ff_at_ref_df = pd.concat([self.both_ff_at_ref_df.reset_index(drop=True), self.stop_ff_df2[['ff_distance', 'ff_angle', 
                                                                                                             'ff_angle_boundary']].reset_index(drop=True)], axis=1)
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'stop_ff_distance_at_ref',
                                                'ff_angle': 'stop_ff_angle_at_ref',
                                                'ff_angle_boundary': 'stop_ff_angle_boundary_at_ref'}, inplace=True)
        
        # self.both_ff_at_ref_df = self.heading_info_df[['alt_ff_distance_at_ref', 'alt_ff_angle_at_ref',
        #                                             'stop_ff_distance_at_ref', 'stop_ff_angle_at_ref',
        #                                             'stop_ff_angle_boundary_at_ref']].copy()
        return self.both_ff_at_ref_df


    def get_x_df(self, save_data=True):
        self.x_df = GUAT_vs_TAFT_utils.combine_relevant_features(self.x_features_df, self.only_stop_ff_df, self.plan_x_df, self.plan_y_df)

        # add num_stops
        trials_df = self.TAFT_trials_df if self.TAFT_or_GUAT == 'TAFT' else self.GUAT_w_ff_df
        self.x_df = self.x_df.merge(trials_df[['stop_point_index', 'num_stops']], on='stop_point_index', how='left')

        # also clean out unnecesary columns especially in clusters
        new_df_columns = features_to_keep_utils.get_minimal_features_to_keep(self.x_df, for_classification=False, deal_with_combd_min_ff_distance=True)
        self.x_df = self.x_df[new_df_columns].copy()

        if self.TAFT_or_GUAT == 'TAFT':
            self.TAFT_x_df = self.x_df.copy()
            if save_data:
                self.TAFT_x_df.to_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'), index=False)
        elif self.TAFT_or_GUAT == 'GUAT':
            self.GUAT_x_df = self.x_df.copy()
            if save_data:
                self.GUAT_x_df.to_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'), index=False)
        else:
            raise ValueError('TAFT_or_GUAT must be either TAFT or GUAT')
        

    def take_out_subsets_to_plot(self, list_of_stop_point_index, TAFT_or_GUAT):
        if TAFT_or_GUAT == 'GUAT':
            sub = self.GUAT_w_ff_df[self.GUAT_w_ff_df['stop_point_index'].isin(list_of_stop_point_index)].copy()
        else:
            sub = self.TAFT_trials_df[self.TAFT_trials_df['stop_point_index'].isin(list_of_stop_point_index)].copy()

        x_df = getattr(self, f"{TAFT_or_GUAT}_x_df")
        sub = sub.merge(x_df[['stop_point_index', 'stop_ff_distance_at_ref']], on='stop_point_index', how='left')

        sub2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(sub, self.monkey_information, self.ff_real_position_sorted,
                                                                        ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value)
        
        sub2.rename(columns={'point_index': 'ref_point_index'}, inplace=True)
        sub = sub.merge(sub2[['stop_point_index', 'ref_point_index']], on='stop_point_index', how='left')

        # also prepare some plotting args and kwargs
        self.plotting_kwargs = {'show_stops': True,
                        'show_believed_target_positions': True,
                        'show_reward_boundary': True,
                        'show_scale_bar': True,
                        'truncate_part_before_crossing_arena_edge': True,
                        'trial_too_short_ok': True,
                        'show_connect_path_ff': False,
                        'show_visible_fireflies': True}

        self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, None, self.ff_caught_T_new)
        self.max_distance_to_stop_for_GUAT_target = 50
        
        return sub
