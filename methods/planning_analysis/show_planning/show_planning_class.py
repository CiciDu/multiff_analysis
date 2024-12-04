
import sys
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, plot_stops_near_ff_utils, stops_near_ff_based_on_ref_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils

from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from data_wrangling import combine_info_utils, base_processing_class


import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import warnings
import contextlib

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


# https://dash.plotly.com/interactive-graphing



class ShowPlanning(base_processing_class.BaseProcessing):
        
    raw_data_dir_name = 'all_monkey_data/raw_monkey_data/individual_monkey_data'

    time_after_dict = {'min': 0.05,
                'max': 0.5,
                'step': 0.05,
                'values': None,
                'marks': None}

    ref_point_info = {'distance': {'min': -190,
                                'max': -100,
                                'step': 10,
                                'values': None,
                                'marks': None},   
                        'time': {'min': -1.9,
                                'max': -0.6,
                                'step': 0.2,
                                'values': None,
                                'marks': None},     
                    }
    
    def __init__(self, monkey_name='monkey_Bruno', 
                    optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
                    test_or_control='test', 
                    raw_data_folder_path=None):
        super().__init__()

        self.monkey_name = monkey_name
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.raw_data_dir_name, monkey_name)
        self.test_or_control = test_or_control

        if raw_data_folder_path is not None:
            self.load_raw_data(raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=None)


        self.update_optimal_arc_type(optimal_arc_type=optimal_arc_type)


    def update_optimal_arc_type(self, optimal_arc_type='norm_opt_arc'):
        # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
        super()._update_optimal_arc_type_and_related_paths(optimal_arc_type=optimal_arc_type)
        self.combd_stop_and_alt_folder_path = make_variations_utils.make_combd_stop_and_alt_folder_path(self.monkey_name)
        self.get_combd_info_folder_paths()


    def get_combd_info_folder_paths(self):
        data_folder = os.path.join(self.combd_stop_and_alt_folder_path, 'data')

        self.dict_of_combd_rel_angle_slope_folder_path = {'test': data_folder + f'/combd_rel_angle_slope/{self.optimal_arc_type}/test',
                                            'control': data_folder + f'/combd_rel_angle_slope/{self.optimal_arc_type}/control'}

        self.dict_of_combd_heading_info_folder_path = {'test': data_folder + f'/combd_heading_info/{self.optimal_arc_type}/test',
                                            'control': data_folder + f'/combd_heading_info/{self.optimal_arc_type}/control/'}
        
        self.dict_of_combd_diff_in_curv_folder_path = {'test': data_folder + f'/combd_diff_in_curv/{self.optimal_arc_type}/test',
                                            'control': data_folder + f'/combd_diff_in_curv/{self.optimal_arc_type}/control'}

        self.combd_plan_x_both_folder_path = data_folder + f'/combd_plan_x_df/{self.optimal_arc_type}'
        
        self.combd_plan_y_both_folder_path = data_folder + f'/combd_plan_y_df/{self.optimal_arc_type}'


    def retrieve_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-100, 
                                             curv_traj_window_before_stop=[-50, 0],
                                             test_or_control=None):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value  
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        
        if test_or_control is None:
            test_or_control = self.test_or_control
        path = self.dict_of_combd_heading_info_folder_path[test_or_control]
        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, ref_point_mode, ref_point_value)
        df_path = os.path.join(path, df_name)
        if not os.path.exists(df_path):
            raise FileNotFoundError(f'combd_heading_df_x_sessions ({df_name}) is not in the folder: ', path)
        else:
            self.combd_heading_df_x_sessions = pd.read_csv(df_path)
            print('Successfully retrieved combd_heading_df_x_sessions from: ', df_path)

        self.combd_diff_in_curv_df = self.retrieve_combd_diff_in_curv_df(ref_point_mode, ref_point_value, test_or_control, 
                                                                     curv_traj_window_before_stop=curv_traj_window_before_stop)

        self.combd_heading_df_x_sessions = self.combd_heading_df_x_sessions.merge(self.combd_diff_in_curv_df, on='stop_point_index', how='left')

        return self.combd_heading_df_x_sessions


    def retrieve_combd_diff_in_curv_df(self, ref_point_mode='distance', ref_point_value=-100, test_or_control=None,
                                     curv_traj_window_before_stop=[-50, 0]):

        folder_path = self.dict_of_combd_diff_in_curv_folder_path[test_or_control]
        df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        df_path = os.path.join(folder_path, df_name)
        if not os.path.exists(df_path):
            raise FileNotFoundError(f'combd_diff_in_curv_df ({df_name}) is not in the folder: ', folder_path)
        else:
            self.combd_diff_in_curv_df = pd.read_csv(df_path)
            print('Successfully retrieved combd_diff_in_curv_df from: ', df_path)
        return self.combd_diff_in_curv_df

    def _store_combd_heading_df_x_sessions(self, test_or_control=None):
        if test_or_control is None:
            test_or_control = self.test_or_control
        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, self.ref_point_mode, self.ref_point_value)
        os.makedirs(self.dict_of_combd_heading_info_folder_path[test_or_control], exist_ok=True)
        self.combd_heading_df_x_sessions.to_csv(os.path.join(self.dict_of_combd_heading_info_folder_path[test_or_control], df_name))
        print(f'Stored {df_name} in the folder: ', self.dict_of_combd_heading_info_folder_path[test_or_control])

    def _store_combd_diff_in_curv_df(self, test_or_control=None):
        df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(self.ref_point_mode, self.ref_point_value, self.curv_traj_window_before_stop)
        os.makedirs(self.dict_of_combd_diff_in_curv_folder_path[test_or_control], exist_ok=True)
        self.combd_diff_in_curv_df.to_csv(os.path.join(self.dict_of_combd_diff_in_curv_folder_path[test_or_control], df_name))
        print(f'Stored {df_name} in the folder: ', self.dict_of_combd_diff_in_curv_folder_path[test_or_control])

    def plot_linear_regression_on_combd_heading_df_x_sessions(self):
        self.alt_traj, self.alt_stop, self.heading_info_df_no_na = show_planning_utils.get_alt_traj_and_alt_stop(self.combd_heading_df_x_sessions)
        #hue = self.heading_info_df_no_na['data_name'].values
        hue = None
        slope, intercept, r_value, p_value, results = show_planning_utils.conduct_linear_regression_to_show_planning(self.alt_traj, self.alt_stop, hue=hue, fit_intercept=True)


    def _extract_key_monkey_info_from_monkey_name_and_data_name(self, monkey_name, data_name):
        raw_data_folder_path = os.path.join(self.raw_data_dir_name, monkey_name, data_name)
        self.extract_info_from_raw_data_folder_path(raw_data_folder_path)
        self.retrieve_or_make_monkey_data(exists_ok=True)



    def retrieve_or_make_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-100,
                                                     curv_traj_window_before_stop=[-50, 0],
                                                    combd_heading_df_x_sessions_exists_ok=True, stops_near_ff_df_exists_ok=True, show_printed_output=False,
                                                    test_or_control=None, sessions_df_for_one_monkey=None):
        if test_or_control is None:
            test_or_control = self.test_or_control

        self.test_or_control = test_or_control
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        
        if combd_heading_df_x_sessions_exists_ok:
            try:
                self.combd_heading_df_x_sessions = self.retrieve_combd_heading_df_x_sessions(ref_point_mode, ref_point_value, 
                                                                                             curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                             test_or_control=test_or_control)
                if show_printed_output:
                    print('The combd_heading_df_x_sessions already exists in the folders: ', self.dict_of_combd_heading_info_folder_path[test_or_control])
                return
            except:
                pass
        self.combd_heading_df_x_sessions, self.combd_diff_in_curv_df = self.make_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                                                             curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                             stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                show_printed_output=show_printed_output, test_or_control=test_or_control, sessions_df_for_one_monkey=sessions_df_for_one_monkey)
        print('Made new and combd_heading_df_x_sessions and stored it in the folder: ', self.dict_of_combd_heading_info_folder_path[test_or_control])
        if 'Unnamed: 0' in self.combd_heading_df_x_sessions.columns:
            self.combd_heading_df_x_sessions.drop(columns=['Unnamed: 0'], inplace=True)    


    def make_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-100,
                                        curv_traj_window_before_stop=[-50, 0],
                                        test_or_control=None, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                        show_printed_output=False, sessions_df_for_one_monkey=None,
                                        use_curvature_to_ff_center=False,
                                        merge_diff_in_curv_df_to_heading_info=True,
                                        save_data=True):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        
        self.combd_heading_df_x_sessions = pd.DataFrame()
        self.combd_diff_in_curv_df = pd.DataFrame()
        if test_or_control is None:
            test_or_control = self.test_or_control

        if show_printed_output:
            self.combd_heading_df_x_sessions = self._make_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                                      curv_traj_window_before_stop=curv_traj_window_before_stop, test_or_control=test_or_control,
                                                                                 stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, sessions_df_for_one_monkey=sessions_df_for_one_monkey, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                 use_curvature_to_ff_center=use_curvature_to_ff_center)  
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.combd_heading_df_x_sessions = self._make_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                                          curv_traj_window_before_stop=curv_traj_window_before_stop, test_or_control=test_or_control, 
                                                                                stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, sessions_df_for_one_monkey=sessions_df_for_one_monkey, 
                                                                                heading_info_df_exists_ok=heading_info_df_exists_ok, use_curvature_to_ff_center=use_curvature_to_ff_center)  

        self.combd_heading_df_x_sessions.reset_index(drop=True, inplace=True)
        self.combd_diff_in_curv_df.reset_index(drop=True, inplace=True)

        if save_data:
            self._store_combd_heading_df_x_sessions(test_or_control=test_or_control)
            self._store_combd_diff_in_curv_df(test_or_control=test_or_control)

        if merge_diff_in_curv_df_to_heading_info:
            self.combd_heading_df_x_sessions = self.combd_heading_df_x_sessions.merge(self.combd_diff_in_curv_df, on='stop_point_index', how='left')

        return self.combd_heading_df_x_sessions, self.combd_diff_in_curv_df


    def _make_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-100, 
                                          curv_traj_window_before_stop=[-50, 0], test_or_control='test', 
                                            stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True, sessions_df_for_one_monkey=None,
                                            use_curvature_to_ff_center=False):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        if sessions_df_for_one_monkey is not None:
            self.sessions_df_for_one_monkey = sessions_df_for_one_monkey
        else:
            self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.raw_data_dir_name, self.monkey_name)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  
            for index, row in self.sessions_df_for_one_monkey.iterrows():
                if row['finished'] is True:
                    continue
                    
                self.heading_info_df = self._make_heading_info_df_for_a_data_session(row['monkey_name'], row['data_name'], ref_point_mode=ref_point_mode, 
                                                                            ref_point_value=ref_point_value, test_or_control=test_or_control, 
                                                                            curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                            use_curvature_to_ff_center=use_curvature_to_ff_center,
                                                                            merge_diff_in_curv_df_to_heading_info=False,
                                                                            )
                self.heading_info_df['data_name'] = row['data_name']
                self.combd_heading_df_x_sessions = pd.concat([self.combd_heading_df_x_sessions, self.heading_info_df], axis=0)
                self.combd_diff_in_curv_df = pd.concat([self.combd_diff_in_curv_df, self.snf.diff_in_curv_df], axis=0)
                self.sessions_df_for_one_monkey.loc[self.sessions_df_for_one_monkey['data_name'] == row['data_name'], 'finished'] = True
        return self.combd_heading_df_x_sessions



    def _make_heading_info_df_for_a_data_session(self, monkey_name, data_name, ref_point_mode='distance', ref_point_value=-150, curv_traj_window_before_stop=[-50, 0],
                                                test_or_control='test', heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                use_curvature_to_ff_center=False,
                                                merge_diff_in_curv_df_to_heading_info=True):
        self.ref_point_value = ref_point_value
        self.ref_point_mode = ref_point_mode
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        
        # first try retrieving it
        raw_data_folder_path = os.path.join(self.raw_data_dir_name, monkey_name, data_name)
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self, raw_data_folder_path)
        self.heading_info_path = os.path.join(self.planning_data_folder_path, self.heading_info_partial_path, test_or_control)
        try:
            self.monkey_name = monkey_name
            self.data_name = data_name
            if heading_info_df_exists_ok is False:
                raise FileNotFoundError('Force the creation of heading_info_df')
            self.snf = stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef(raw_data_folder_path=None, optimal_arc_type=self.optimal_arc_type)
            
            self.snf.extract_info_from_raw_data_folder_path(raw_data_folder_path)
            heading_info_df, diff_in_curv_df = self.snf._retrieve_heading_info_df(ref_point_mode, ref_point_value, test_or_control, 
                                                                                  curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                  merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)
        except FileNotFoundError:
            print('Making new heading_info_df ...')
            self.snf = stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef(raw_data_folder_path=raw_data_folder_path, optimal_arc_type=self.optimal_arc_type)
            
            self.snf.make_heading_info_df_without_long_process(test_or_control=test_or_control, ref_point_mode=ref_point_mode,
                                                                ref_point_value=ref_point_value, curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                use_curvature_to_ff_center=use_curvature_to_ff_center, 
                                                                heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)
            heading_info_df = self.snf.heading_info_df.copy()
        return heading_info_df

    def make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(self, ref_point_mode='distance', ref_point_value=-100, 
                                                                                curv_traj_window_before_stop=[-50, 0],
                                                                                combd_heading_df_x_sessions_exists_ok=True, 
                                                                            heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True, 
                                                                            show_printed_output=False, use_curvature_to_ff_center=False, save_data=True):
        for test_or_control in ['control', 'test']:
            stops_near_ff_df_exists_ok = stops_near_ff_df_exists_ok if test_or_control == 'test' else stops_near_ff_df_exists_ok
            self.handle_heading_info_df(ref_point_mode, ref_point_value, combd_heading_df_x_sessions_exists_ok, heading_info_df_exists_ok, stops_near_ff_df_exists_ok, 
                                        show_printed_output, test_or_control, curv_traj_window_before_stop=curv_traj_window_before_stop, use_curvature_to_ff_center=use_curvature_to_ff_center, save_data=save_data)
        return self.test_heading_info_df, self.ctrl_heading_info_df


    def handle_heading_info_df(self, ref_point_mode, ref_point_value, combd_heading_df_x_sessions_exists_ok, heading_info_df_exists_ok, stops_near_ff_df_exists_ok, show_output, test_or_control,
                               curv_traj_window_before_stop=[-50, 0],
                               use_curvature_to_ff_center=False, save_data=True):
        df_name = "test_heading_info_df" if test_or_control == 'test' else "ctrl_heading_info_df"
        try:
            if combd_heading_df_x_sessions_exists_ok:
                combd_heading_df_x_sessions = self.retrieve_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, test_or_control=test_or_control,
                                                                                        curv_traj_window_before_stop=curv_traj_window_before_stop)
            else:
                raise FileNotFoundError  # Force the creation if combd_heading_df_x_sessions_exists_ok is False
        except FileNotFoundError:  # Specific exception for clarity
            _, combd_heading_df_x_sessions, combd_diff_in_curv_df = self.make_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                    curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, 
                                                                   show_printed_output=show_output, test_or_control=test_or_control, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                   use_curvature_to_ff_center=use_curvature_to_ff_center, save_data=save_data,
                                                                   merge_diff_in_curv_df_to_heading_info=True)
            setattr(self, df_name, combd_heading_df_x_sessions)
        setattr(self, df_name, combd_heading_df_x_sessions)
