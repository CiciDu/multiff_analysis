import sys
from data_wrangling import base_processing_class
from planning_analysis.show_planning import alt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
import numpy as np
import pandas as pd
import os


class _FindStopsNearFF(base_processing_class.BaseProcessing):

    def __init__(self):
        super().__init__()


    def get_stops_near_ff_df(self, test_or_control='test', exists_ok=True, save_data=True):
        self.test_or_control = test_or_control
        if test_or_control == 'test':
            self.make_stops_near_ff_df_test(exists_ok=exists_ok, save_data=save_data)
        elif test_or_control == 'control':
            self.make_stops_near_ff_df_ctrl(exists_ok=exists_ok, save_data=save_data)
        else:
            raise ValueError('test_or_control should be either test or control')


    def make_stops_near_ff_df_test(self, shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        self._make_stops_near_ff_df_test_or_ctrl(test_or_control='test', shared_stops_near_ff_df_already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)


    def make_stops_near_ff_df_ctrl(self, shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        self._make_stops_near_ff_df_test_or_ctrl(test_or_control='control', shared_stops_near_ff_df_already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)



    def _make_shared_stops_near_ff_df_if_not_already_made(self, remove_cases_where_monkey_too_close_to_edge=False,
                                                    max_distance_between_stop_and_alt_ff=500, min_distance_between_stop_and_alt_ff=25,
                                                    stop_period_duration=2,
                                                    already_made_ok=True, exists_ok=True, save_data=True):


        if not already_made_ok:
            self.shared_stops_near_ff_df = None

        if self.shared_stops_near_ff_df is None:
            if exists_ok:
                try:
                    self.shared_stops_near_ff_df = pd.read_csv(os.path.join(self.planning_data_folder_path, 'shared_stops_near_ff_df.csv')).drop(["Unnamed: 0"], axis=1)
                    self.shared_stops_near_ff_df = find_stops_near_ff_utils.process_shared_stops_near_ff_df(self.shared_stops_near_ff_df)

                    print('Retrieving shared_stops_near_ff_df succeeded')
                    return
                except Exception as e:
                    print('Failed to retrieve shared_stops_near_ff_df; will make new shared_stops_near_ff_df')
        
            if self.monkey_information is None:
                self.load_raw_data(self.raw_data_folder_path, monkey_data_exists_ok=True,
                                   curv_of_traj_mode=self.curv_of_traj_params['curv_of_traj_mode'],
                                   window_for_curv_of_traj=self.curv_of_traj_params['window_for_curv_of_traj'],
                                   truncate_curv_of_traj_by_time_of_capture=self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'])

            if self.ff_dataframe is None:
                self.get_more_monkey_data()

            self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible']==1].copy()
            self.shared_stops_near_ff_df, self.all_alt_ff_df = find_stops_near_ff_utils.make_shared_stops_near_ff_df(self.monkey_information, self.ff_dataframe_visible, self.ff_real_position_sorted,
                                                                                        self.ff_caught_T_sorted, self.ff_flash_sorted, self.ff_life_sorted, min_distance_between_stop_and_alt_ff=min_distance_between_stop_and_alt_ff,
                                                                                        max_distance_between_stop_and_alt_ff=max_distance_between_stop_and_alt_ff, stop_period_duration=stop_period_duration,
                                                                                        remove_cases_where_monkey_too_close_to_edge=remove_cases_where_monkey_too_close_to_edge
                                                                                        )
            
            self.shared_stops_near_ff_df = find_stops_near_ff_utils.process_shared_stops_near_ff_df(self.shared_stops_near_ff_df)
            self._add_curv_of_traj_stat_df()
            if save_data:
                self.shared_stops_near_ff_df.to_csv(os.path.join(self.planning_data_folder_path, 'shared_stops_near_ff_df.csv'))
                print(f'Stored shared_stops_near_ff_df ({len(self.shared_stops_near_ff_df)} rows) in {os.path.join(self.planning_data_folder_path, "shared_stops_near_ff_df.csv")}')
    

    def _make_stops_near_ff_df_test_or_ctrl(self, test_or_control='test', shared_stops_near_ff_df_already_made_ok=True, exists_ok=True, save_data=True):
        if (test_or_control != 'test') and (test_or_control != 'control'):
            raise ValueError('test_or_ctrl must be either "test" or "control"')
        test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'
        
        self._make_shared_stops_near_ff_df_if_not_already_made(already_made_ok=shared_stops_near_ff_df_already_made_ok, exists_ok=exists_ok, save_data=save_data)
        self.stops_near_ff_df = self.shared_stops_near_ff_df[self.shared_stops_near_ff_df['data_category_by_vis']==test_or_control].copy()
        self.stops_near_ff_df.reset_index(drop=True, inplace=True)
        self.stops_near_ff_df, self.alt_ff_df, self.stop_ff_df = self._make_alt_ff_df_and_stop_ff_df(self.stops_near_ff_df)
        setattr(self, f'stops_near_ff_df_{test_or_ctrl}', self.stops_near_ff_df)
        #print(f'Made stops_near_ff_df_test, which has {len(self.stops_near_ff_df_test)} rows')


    def _add_curv_of_traj_stat_df(self, curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25]):
        if self.curv_of_traj_df is None:
            self.curv_of_traj_df = self.get_curv_of_traj_df(window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode, 
                                                             truncate_curv_of_traj_by_time_of_capture=False)
        self.curv_of_traj_stat_df, self.shared_stops_near_ff_df = plan_factors_utils.find_curv_of_traj_stat_df(self.shared_stops_near_ff_df, self.curv_of_traj_df)      


    def _make_alt_ff_df_and_stop_ff_df(self, stops_near_ff_df):
        self.stops_near_ff_df, self.alt_ff_df, self.stop_ff_df = alt_ff_utils.get_alt_ff_df_and_stop_ff_df(stops_near_ff_df)
        self.stops_near_ff_df_counted = self.stops_near_ff_df.copy()  
        return self.stops_near_ff_df, self.alt_ff_df, self.stop_ff_df



