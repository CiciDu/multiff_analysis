import sys
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from planning_analysis.show_planning import alt_ff_utils, show_planning_class, show_planning_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_class, plot_stops_near_ff_utils, plot_monkey_heading_helper_class, stops_near_ff_based_on_ref_class
from planning_analysis.plan_factors import plan_factors_utils, test_vs_control_utils, plan_factors_class
from planning_analysis.agent_analysis import agent_plan_factors_class
from planning_analysis.variations_of_factors_vs_results import variations_base_class, plot_variations_class
from planning_analysis import ml_methods_utils, ml_methods_class
from data_wrangling import basic_func, combine_info_utils, base_processing_class
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import gc
import warnings

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class PlanAcrossSessions(plot_variations_class.PlotVariations):

    dir_name = 'all_monkey_data/raw_monkey_data/individual_monkey_data'

    default_ref_point_params_based_on_mode = {'time after stop ff visible': [0.1, 0], 'distance': [-150, -100, -50]}

    def __init__(self, 
                 monkey_name='monkey_Bruno', 
                 optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 curv_traj_window_before_stop=[-50, 0]                 
                ):
        
        super().__init__(optimal_arc_type=optimal_arc_type, curv_traj_window_before_stop=curv_traj_window_before_stop)
        self.monkey_name = monkey_name
        self.sessions_df = None
        self.sessions_df_for_one_monkey = None
        self.combd_planning_info_folder_path = f'all_monkey_data/planning/combined_data'
        self.make_key_paths()


    def retrieve_all_plan_data_for_one_session(self, raw_data_folder_path, ref_point_mode='distance', ref_point_value=-150):
        self.pf = plan_factors_class.PlanFactors()
        self.pf.monkey_name = self.monkey_name
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self.pf, raw_data_folder_path)
        self.pf.ref_point_mode = ref_point_mode
        self.pf.ref_point_value = ref_point_value
        self.pf.retrieve_all_plan_data_for_one_session()
        self.pf._get_plan_x_and_y_combd()
        print('Successfully retrieved plan_x and plan_y data for session: ', raw_data_folder_path)


    def get_plan_x_and_plan_y_across_sessions(self, exists_ok=True, 
                                            plan_x_exists_ok=True,
                                            plan_y_exists_ok=True,
                                            heading_info_df_exists_ok=False,
                                            stops_near_ff_df_exists_ok=True, 
                                            curv_of_traj_mode='distance', 
                                            window_for_curv_of_traj=[-25, 25],
                                            use_curvature_to_ff_center=False,
                                            ref_point_mode='distance', 
                                            ref_point_value=-150,
                                            dir_name='all_monkey_data/raw_monkey_data/individual_monkey_data',
                                            resume_sessions_df_for_one_monkey=False,
                                            save_data=True
                                            ):
        
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.use_curvature_to_ff_center = use_curvature_to_ff_center

        self.combd_stop_and_alt_folder_path = make_variations_utils.make_combd_stop_and_alt_folder_path(self.monkey_name)
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)

        df_name = find_stops_near_ff_utils.find_df_name(self.monkey_name, ref_point_mode, ref_point_value)
        #df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode, ref_point_value, curv_traj_window_before_stop)   
        combd_plan_x_both_path = os.path.join(self.combd_plan_x_both_folder_path, df_name)
        combd_plan_y_both_path = os.path.join(self.combd_plan_y_both_folder_path, df_name)

        if exists_ok :
            if exists(combd_plan_x_both_path) & exists(combd_plan_y_both_path):
                self.combd_plan_x_both = pd.read_csv(combd_plan_x_both_path)
                self.combd_plan_y_both = pd.read_csv(combd_plan_y_both_path)
                return 
            else:
                print('Retrieving combd_plan_y_both and combd_plan_x_both failed. Will recreate them.')

        self.combd_plan_y_both = pd.DataFrame()
        self.combd_plan_x_both = pd.DataFrame()

        if (not resume_sessions_df_for_one_monkey) | (self.sessions_df_for_one_monkey is None):
            self.initialize_monkey_sessions_df_for_one_monkey()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  
            for index, row in self.sessions_df_for_one_monkey.iterrows():
                if row['finished'] is True:
                    continue
                raw_data_folder_path = os.path.join(dir_name, row['monkey_name'], row['data_name'])
                print(raw_data_folder_path)
                # first just try retrieving the data directly
                try: 
                    if plan_x_exists_ok & plan_y_exists_ok:
                        self.retrieve_all_plan_data_for_one_session(raw_data_folder_path=raw_data_folder_path, ref_point_mode=ref_point_mode,                                                                  
                                                                    ref_point_value=ref_point_value)
                    else:
                        raise Exception('plan_x_exists_ok is False or plan_y_exists_ok is False') 
                except Exception as e:
                    print(e)
                    print('Will recreate the plan_x and plan_y data for this session')
                    self.pf = plan_factors_class.PlanFactors(raw_data_folder_path=raw_data_folder_path,
                                                             optimal_arc_type=self.optimal_arc_type,
                                                             curv_traj_window_before_stop=self.curv_traj_window_before_stop,                                                                
                                                            curv_of_traj_mode=curv_of_traj_mode, window_for_curv_of_traj=window_for_curv_of_traj)
                    gc.collect()
                    self.pf.make_plan_x_and_y_for_both_test_and_ctrl(plan_x_exists_ok=plan_x_exists_ok, plan_y_exists_ok=plan_y_exists_ok,
                                                                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                    use_curvature_to_ff_center=use_curvature_to_ff_center, 
                                                                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)
                    
                self._add_new_plan_data_to_combd_data(row['data_name'])
        if save_data:
            os.makedirs(self.combd_plan_x_both_folder_path, exist_ok=True)
            os.makedirs(self.combd_plan_y_both_folder_path, exist_ok=True)
            self.combd_plan_x_both.to_csv(combd_plan_x_both_path, index=False)
            self.combd_plan_y_both.to_csv(combd_plan_y_both_path, index=False)

        self.combd_plan_y_both = self.combd_plan_y_both.copy().reset_index(drop=True)
        self.combd_plan_x_both = self.combd_plan_x_both.copy().reset_index(drop=True)
    
        return


    def initialize_monkey_sessions_df(self):
        self.sessions_df = basic_func.initialize_monkey_sessions_df(dir_name=self.dir_name)
        return self.sessions_df


    def initialize_monkey_sessions_df_for_one_monkey(self):
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.dir_name, self.monkey_name)


    def get_combd_heading_df_x_sessions_across_sessions(self,
                                                    ref_point_mode='distance', ref_point_value=-150,
                                                    heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                    use_curvature_to_ff_center=False,
                                                    exists_ok=True, save_data=True):

        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                    optimal_arc_type=self.optimal_arc_type,
                                                    curv_traj_window_before_stop=self.curv_traj_window_before_stop
                                                    )
        self.combd_heading_df_x_sessions_test, self.combd_heading_df_x_sessions_ctrl = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value, 
                                                                        combd_heading_df_x_sessions_exists_ok=exists_ok, use_curvature_to_ff_center=use_curvature_to_ff_center,
                                                                        show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                        stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)
        


    def _add_new_plan_data_to_combd_data(self, data_name):
        self.pf.plan_y_both['data_name'] = data_name
        self.pf.plan_x_both['data_name'] = data_name
        self.combd_plan_y_both = pd.concat([self.combd_plan_y_both, self.pf.plan_y_both], axis=0)
        self.combd_plan_x_both = pd.concat([self.combd_plan_x_both, self.pf.plan_x_both], axis=0)
        self.sessions_df_for_one_monkey.loc[self.sessions_df_for_one_monkey['data_name'] == data_name, 'finished'] = True


    def combine_overall_all_median_info_across_monkeys_and_optimal_arc_types(self):
        self.overall_all_median_info = make_variations_utils.combine_overall_all_median_info_across_monkeys_and_optimal_arc_types()
        self.process_overall_all_median_info_to_plot_heading_and_curv()
        return self.overall_all_median_info
        

    def combine_all_perc_info_across_monkeys(self):
        self.all_perc_info = make_variations_utils.combine_all_perc_info_across_monkeys()
        self.process_all_perc_info_to_plot_direction()
        return self.all_perc_info