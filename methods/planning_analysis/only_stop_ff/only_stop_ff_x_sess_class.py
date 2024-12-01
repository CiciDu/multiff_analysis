import sys
from data_wrangling import combine_info_utils, base_processing_class, basic_func
from planning_analysis import ml_methods_class, ml_methods_utils
from planning_analysis.only_stop_ff import only_stop_ff_class
from planning_analysis.show_planning import alt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_class, test_vs_control_utils
from planning_analysis.only_stop_ff import only_stop_ff_utils, only_stop_ff_utils, features_to_keep_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
from os.path import exists
import gc
from scipy.stats.mstats import winsorize
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import math


class OnlyStopFFAcrossSessions():
    
    dir_name = 'all_monkey_data/raw_monkey_data/individual_monkey_data'

    ref_point_params_based_on_mode = {'distance': [-50, -100, -150],
                                     'time after stop ff visible': [0.1, 0, -0.2],
                                       }

    ref_point_info = {
                    'time after stop ff visible': {'min': 0,
                                'max': 0.3,
                                'step': 0.1,
                                'values': None,
                                'marks': None},  
                    'distance': {'min': -190,
                                'max': -100,
                                'step': 20,
                                'values': None,
                                'marks': None},      
                    }
    

    def __init__(self, monkey_name='monkey_Bruno', 
                 optimal_arc_type='norm_opt_arc',
                 curv_of_traj_mode='distance', 
                 window_for_curv_of_traj=[-25, 25], 
                 truncate_curv_of_traj_by_time_of_capture=False):
        self.monkey_information = None
        self.ff_info_at_start_df = None
        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.truncate_curv_of_traj_by_time_of_capture = truncate_curv_of_traj_by_time_of_capture
        self.ref_point_mode = None
        self.ref_point_value = None
        self.monkey_name = monkey_name

        self.ml_inst = ml_methods_class.MlMethods()
        self._update_optimal_arc_type_and_related_paths(optimal_arc_type=optimal_arc_type)


    def _update_optimal_arc_type_and_related_paths(self, optimal_arc_type='norm_opt_arc'):
        self.optimal_arc_type = optimal_arc_type

        self.only_stop_ff_folder_path = os.path.join(self.planning_data_folder_path, f'only_stop_ff/only_stop_ff_df/{self.optimal_arc_type}')


        self.top_combd_only_stop_ff_path = f'all_monkey_data/planning/individual_monkey_data/{self.monkey_name}/combd_planning_info/only_stop_ff'
        self.combd_only_stop_ff_df_folder_path = os.path.join(self.top_combd_only_stop_ff_path, f'data/combd_only_stop_ff_df/{self.optimal_arc_type}')
        self.combd_x_features_folder_path = os.path.join(self.top_combd_only_stop_ff_path, f'data/combd_x_features_df/{self.optimal_arc_type}')
        os.makedirs(self.combd_only_stop_ff_df_folder_path, exist_ok=True)
        os.makedirs(self.combd_x_features_folder_path, exist_ok=True)

        self.only_stop_ff_lr_df_path = os.path.join(self.top_combd_only_stop_ff_path, f'ml_results/lr_variations/{self.optimal_arc_type}/all_only_stop_lr_df.csv')
        self.only_stop_ff_ml_df_path = os.path.join(self.top_combd_only_stop_ff_path, f'ml_results/ml_variations/{self.optimal_arc_type}/all_only_stop_ml_df.csv')
        os.makedirs(os.path.dirname(self.only_stop_ff_lr_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.only_stop_ff_ml_df_path), exist_ok=True)


    def make_only_stop_ff_df_and_x_features_df_across_sessions(self, exists_ok=True, only_stop_ff_df_exists_ok=True, x_features_df_exists_ok=True, 
                                                               stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150):
        
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        
        try:
            if exists_ok:
                self._retrieve_combd_only_stop_ff_df(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
                self._retrieve_combd_x_features_df(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
                self.prepare_only_stop_ff_data_for_ml()
                return 
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            pass
        
        #if self.sessions_df is None:
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.dir_name, self.monkey_name)
        self.combd_only_stop_ff_df = pd.DataFrame()
        self.combd_x_features_df = pd.DataFrame()
        for index, row in self.sessions_df_for_one_monkey.iterrows():
            if row['finished'] is True:
                continue
        
            data_name = row['data_name']

            raw_data_folder_path = os.path.join(self.dir_name, row['monkey_name'], data_name)
            print(raw_data_folder_path)

            self.osf = only_stop_ff_class.OnlyStopFF(monkey_name=self.monkey_name, raw_data_folder_path=raw_data_folder_path,
                                                     optimal_arc_type=self.optimal_arc_type, curv_of_traj_mode=self.curv_of_traj_mode,
                                                     window_for_curv_of_traj=self.window_for_curv_of_traj, 
                                                     truncate_curv_of_traj_by_time_of_capture=self.truncate_curv_of_traj_by_time_of_capture)
                                                     
            base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self.osf, raw_data_folder_path)

            self.osf.make_only_stop_ff_df(exists_ok=only_stop_ff_df_exists_ok, stop_period_duration=stop_period_duration,
                                      ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
            self.osf.make_x_features_df(exists_ok=x_features_df_exists_ok, ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
            
            current_session_info = (self.sessions_df_for_one_monkey['data_name'] == data_name)
            self.sessions_df_for_one_monkey.loc[current_session_info, 'finished'] = True

            self.osf.only_stop_ff_df['data_name'] = data_name
            self.osf.x_features_df['data_name'] = data_name
            self.combd_only_stop_ff_df = pd.concat([self.combd_only_stop_ff_df, self.osf.only_stop_ff_df], axis=0, ignore_index=True)
            self.combd_x_features_df = pd.concat([self.combd_x_features_df, self.osf.x_features_df], axis=0, ignore_index=True)
            gc.collect()

            print('len(self.only_stop_ff_df): ', self.osf.only_stop_ff_df.shape[0])
            print('len(self.x_features_df): ', self.osf.x_features_df.shape[0])
            if len(self.osf.only_stop_ff_df) != len(self.osf.x_features_df):
                raise ValueError('The length of only_stop_ff_df and x_features_df are not the same.')

        self.combd_only_stop_ff_df = self.combd_only_stop_ff_df.sort_values(by=['data_name', 'stop_point_index']).reset_index(drop=True)
        self.combd_x_features_df = self.combd_x_features_df.sort_values(by=['data_name', 'stop_point_index']).reset_index(drop=True)

        # to save the csv
        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, ref_point_mode, ref_point_value)
        self.combd_only_stop_ff_df.to_csv(os.path.join(self.combd_only_stop_ff_df_folder_path, df_name))
        self.combd_x_features_df.to_csv(os.path.join(self.combd_x_features_folder_path, df_name))
        self.prepare_only_stop_ff_data_for_ml()


    def _retrieve_combd_only_stop_ff_df(self, ref_point_mode='distance', ref_point_value=-100):
        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, ref_point_mode, ref_point_value)
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        df_path = os.path.join(self.combd_only_stop_ff_df_folder_path, df_name)
        if exists(df_path):
            self.combd_only_stop_ff_df = pd.read_csv(df_path)
            print(f'Successfully retrieved combd_only_stop_ff_df ({df_name}) from the folder: {df_path}')
        else:
            raise FileNotFoundError(f'combd_only_stop_ff_df ({df_name}) is not in the folder: {self.combd_only_stop_ff_df_folder_path}')


    def _retrieve_combd_x_features_df(self, ref_point_mode='distance', ref_point_value=-100):
        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, ref_point_mode, ref_point_value)
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        df_path = os.path.join(self.combd_x_features_folder_path, df_name)
        if exists(df_path):
            self.combd_x_features_df = pd.read_csv(df_path)
            print(f'Successfully retrieved combd_x_features_df ({df_name}) from the folder: {df_path}')
        else:
            raise FileNotFoundError(f'combd_x_features_df ({df_name}) is not in the folder: {self.combd_x_features_folder_path}')


    def make_or_retrieve_all_only_stop_lr_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.only_stop_ff_lr_df_path
        if exists_ok:
            if exists(df_path):
                self.all_only_stop_lr_df = pd.read_csv(df_path)
                print(f'Successfully retrieved all_only_stop_lr_df from {df_path}')
                return self.all_only_stop_lr_df
            else:
                print(f'Failed to retrieve all_only_stop_lr_df from {df_path}; will make a new one')
                
        self.variations_list = basic_func.init_variations_list_func(ref_point_params_based_on_mode, folder_path=self.combd_only_stop_ff_df_folder_path, 
                                                                    monkey_name=self.monkey_name)

        self.all_only_stop_lr_df = pd.DataFrame()
        for index, row in self.variations_list.iterrows():
            self.make_only_stop_ff_df_and_x_features_df_across_sessions(exists_ok=True, x_features_df_exists_ok=True, only_stop_ff_df_exists_ok=True,
                                                                    ref_point_mode=row['ref_point_mode'], ref_point_value=row['ref_point_value'])
            self.only_stop_lr_df = self.ml_inst.try_different_combinations_for_linear_regressions(self)
            self.all_only_stop_lr_df = pd.concat([self.all_only_stop_lr_df, self.only_stop_lr_df], axis=0)
        self.all_only_stop_lr_df.reset_index(drop=True, inplace=True)
        self.all_only_stop_lr_df.to_csv(df_path, index=False)

        return self.all_only_stop_lr_df


    def make_or_retrieve_all_only_stop_ml_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.only_stop_ff_ml_df_path
        if exists_ok:
            if exists(df_path):
                self.all_only_stop_ml_df = pd.read_csv(df_path)
                print(f'Successfully retrieved all_only_stop_ml_df from {df_path}')
                return self.all_only_stop_ml_df
            else:
                print(f'Failed to retrieve all_only_stop_ml_df from {df_path}; will make a new one')

        self.variations_list = basic_func.init_variations_list_func(ref_point_params_based_on_mode, folder_path=self.combd_only_stop_ff_df_folder_path, 
                                                                    monkey_name=self.monkey_name)

        all_only_stop_ml_df = pd.DataFrame()
        for index, row in self.variations_list.iterrows():
            self.make_only_stop_ff_df_and_x_features_df_across_sessions(exists_ok=True, x_features_df_exists_ok=True, only_stop_ff_df_exists_ok=True,
                                                                    ref_point_mode=row['ref_point_mode'], ref_point_value=row['ref_point_value'])
            only_stop_ml_df = self.ml_inst.try_different_combinations_for_ml(self, model_names=['rf'])
            all_only_stop_ml_df = pd.concat([all_only_stop_ml_df, only_stop_ml_df], axis=0)
        all_only_stop_ml_df.reset_index(drop=True, inplace=True)
        all_only_stop_ml_df.to_csv(df_path, index=False)   
        self.all_only_stop_ml_df = all_only_stop_ml_df
        return all_only_stop_ml_df 



    def prepare_only_stop_ff_data_for_ml(self):
    
        self.only_stop_ff_df_for_ml = self.combd_only_stop_ff_df.copy()
        self.x_features_df_for_ml = self.combd_x_features_df.copy()

        only_stop_ff_class.OnlyStopFF._prepare_only_stop_ff_data_for_ml(self)



    def streamline_preparing_for_ml(self, y_var_column, **kwargs):
        only_stop_ff_class.OnlyStopFF.streamline_preparing_for_ml(self, y_var_column, **kwargs)

