    
import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from planning_analysis.show_planning.get_stops_near_ff import stops_near_ff_based_on_ref_class, find_stops_near_ff_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import plan_factors_class, monkey_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, variations_base_class
from machine_learning.RL.SB3 import sb3_for_multiff_class, rl_for_multiff_utils, rl_for_multiff_class

from data_wrangling import basic_func, combine_info_utils
from data_wrangling import basic_func
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import gc
from os.path import exists
import os

# This class collects data from many agents and compares them
class PlanFactorsAcrossAgentSessions(variations_base_class._VariationsBase):

    def __init__(self,
                 model_folder_name='RL_models/SB3_stored_models/all_agents/env1_relu/ff3/dv10_dw10_w10_mem3',
                 optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 curv_traj_window_before_stop=[-50, 0],
                 num_steps_per_dataset=100000,
                 ):

        super().__init__(optimal_arc_type=optimal_arc_type, curv_traj_window_before_stop=curv_traj_window_before_stop)
        self.model_folder_name = model_folder_name
        self.optimal_arc_type = optimal_arc_type
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        self.num_steps_per_dataset = num_steps_per_dataset
        rl_for_multiff_class._RLforMultifirefly.get_related_folder_names_from_model_folder_name(self, self.model_folder_name)
        self.monkey_name = None
        self.combd_planning_info_folder_path = os.path.join(os.path.dirname(os.path.dirname(self.planning_data_folder_path)), 'combined_data/combd_planning_info')
        self.combd_stop_and_alt_folder_path = os.path.join(os.path.dirname(os.path.dirname(self.planning_data_folder_path)), 'combined_data/stop_and_alt')
        # note that we used dir_name for the above because those data folder path includes "individual_data_sessions/data_0" and so on at the end.
        self.make_key_paths()
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode
        

    def streamline_getting_y_values(self,
                                    num_datasets_to_collect=1,
                                    ref_point_mode='time after stop ff visible',
                                    ref_point_value=0.1,
                                    save_data = True,
                                    final_products_exist_ok = True,
                                    intermediate_products_exist_ok = True,
                                    agent_data_exists_ok=True,
                                    model_folder_name=None,
                                    **env_kwargs
                                ):
        
        if model_folder_name is None:
            model_folder_name = self.model_folder_name

        # make sure there's enough data from agent
        for i in range(num_datasets_to_collect):
            data_name = f'data_{i}'
            print(' ')
            print('model_folder_name:', model_folder_name)
            print('data_name:', data_name)
            self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=model_folder_name,
                                                                    data_name=data_name,
                                                                    optimal_arc_type=self.optimal_arc_type,
                                                                    curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                                    )
            
            # check to see if data exists:
            if agent_data_exists_ok & exists(os.path.join(self.pfa.processed_data_folder_path, 'monkey_information.csv')):
                print('Data exists for this agent.')
                continue
            print('Getting agent data ......')
            env_kwargs['print_ff_capture_incidents'] = False
            self.pfa.get_agent_data(**env_kwargs, exists_ok=agent_data_exists_ok, save_data=save_data, n_steps=self.num_steps_per_dataset)
            


        print(' ')
        print('Making overall all median info ......')
        self.make_or_retrieve_overall_all_median_info(ref_point_params_based_on_mode={'time after stop ff visible': [0.1, 0],
                                                                        'distance': [-150, -100, -50]},
                                                    save_data=save_data,
                                                    exists_ok=final_products_exist_ok, 
                                                    all_median_info_exists_ok=intermediate_products_exist_ok, 
                                                    combd_heading_df_x_sessions_exists_ok=intermediate_products_exist_ok, 
                                                    stops_near_ff_df_exists_ok=intermediate_products_exist_ok, 
                                                    heading_info_df_exists_ok=intermediate_products_exist_ok)
        self.agent_overall_all_median_info = self.overall_all_median_info.copy()
        print(' ')
        print('Making all perc info ......')
        self.make_or_retrieve_all_perc_info(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                            verbose=True, 
                                            exists_ok=final_products_exist_ok, 
                                            stops_near_ff_df_exists_ok=intermediate_products_exist_ok, 
                                            heading_info_df_exists_ok=intermediate_products_exist_ok,
                                            save_data=save_data)
        self.agent_all_perc_df = self.all_perc_info.copy()


    def get_plan_x_and_plan_y_across_sessions(self, 
                                              num_datasets_to_collect=1,
                                            ref_point_mode='distance', ref_point_value=-150,
                                            exists_ok=True, 
                                            plan_x_exists_ok=True,
                                            plan_y_exists_ok=True,
                                            heading_info_df_exists_ok=True, 
                                            stops_near_ff_df_exists_ok=True,
                                            curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25],
                                            use_curvature_to_ff_center=False,
                                            save_data=True,
                                            **env_kwargs
                                            ):
        
        df_name = find_stops_near_ff_utils.find_df_name('monkey_agent', ref_point_mode, ref_point_value)
        dir = self.combd_planning_info_folder_path
        os.makedirs(dir, exist_ok=True)

        if exists_ok:
            try:
                self.combd_plan_y_both = pd.read_csv(os.path.join(dir, f'combd_plan_y_both_{df_name}.csv'))
                self.combd_plan_x_both = pd.read_csv(os.path.join(dir, f'combd_plan_x_both_{df_name}.csv'))
                print('Successfully retrieved combd_plan_y_both_{df_name} and combd_plan_x_both_{df_name}.')
                return 
            except FileNotFoundError:
                print('Retrieving combd_plan_y_both_{df_name} and combd_plan_x_both_{df_name} failed. Will recreate them.')

        self.combd_plan_y_both = pd.DataFrame()
        self.combd_plan_x_both = pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  

            for i in range(num_datasets_to_collect):
                data_name = f'data_{i}'
                print(' ')
                print('model_folder_name:', self.model_folder_name)
                print('data_name:', data_name)
                self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=self.model_folder_name,
                                                                        data_name=data_name,
                                                                        optimal_arc_type=self.optimal_arc_type,
                                                                        curv_traj_window_before_stop=self.curv_traj_window_before_stop,                                                                        
                                                                        )
                print(' ')
                print('Getting plan x and plan y data ......')
                self.pfa.get_plan_x_and_plan_y_for_one_session(ref_point_mode=ref_point_mode, 
                                        ref_point_value=ref_point_value,
                                        plan_x_exists_ok=plan_x_exists_ok, 
                                        plan_y_exists_ok=plan_y_exists_ok, 
                                        heading_info_df_exists_ok=heading_info_df_exists_ok,
                                        stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, 
                                        curv_of_traj_mode=curv_of_traj_mode, 
                                        window_for_curv_of_traj=window_for_curv_of_traj,
                                        use_curvature_to_ff_center=use_curvature_to_ff_center,
                                        save_data=save_data,
                                        n_steps=self.num_steps_per_dataset,
                                        **env_kwargs)
                
                self._add_plan_xy_to_combd_plan_xy(data_name)

        if save_data:
            self.combd_plan_y_both.to_csv(os.path.join(dir, f'combd_plan_y_both_{df_name}.csv'))
            self.combd_plan_x_both.to_csv(os.path.join(dir, f'combd_plan_x_both_{df_name}.csv'))



    def retrieve_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-150):
        df_name_dict = {'control': 'ctrl_heading_info_df', 
                        'test': 'test_heading_info_df'}               
        for test_or_control in ['control', 'test']:
            combd_heading_df_x_sessions = show_planning_class.ShowPlanning.retrieve_combd_heading_df_x_sessions(self, ref_point_mode=ref_point_mode, 
                                                                                                        ref_point_value=ref_point_value,
                                                                                                        test_or_control=test_or_control)
            setattr(self, df_name_dict[test_or_control], combd_heading_df_x_sessions)


    def make_combd_heading_df_x_sessions(self, num_datasets_to_collect=1,
                                    ref_point_mode='distance', ref_point_value=-150,
                                    heading_info_df_exists_ok=True, 
                                    stops_near_ff_df_exists_ok=True,
                                    curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25],
                                    use_curvature_to_ff_center=False,
                                    save_data=True,
                                    **env_kwargs
                                    ):      
        self.test_combd_heading_df_x_sessions = pd.DataFrame()
        self.ctrl_combd_heading_df_x_sessions = pd.DataFrame()
        for i in range(num_datasets_to_collect):
            data_name = f'data_{i}'
            print(' ')
            print('model_folder_name:', self.model_folder_name)
            print('data_name:', data_name)
            self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=self.model_folder_name,
                                                                    data_name=data_name,
                                                                    optimal_arc_type=self.optimal_arc_type,
                                                                    curv_traj_window_before_stop=self.curv_traj_window_before_stop,                                                                    
                                                                    )
            print(' ')
            print('Getting test heading info control heading info ......')
            self.pfa.get_test_and_ctrl_heading_info_df_for_one_session(ref_point_mode=ref_point_mode, 
                                    ref_point_value=ref_point_value,
                                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, 
                                    curv_of_traj_mode=curv_of_traj_mode, 
                                    window_for_curv_of_traj=window_for_curv_of_traj,
                                    use_curvature_to_ff_center=use_curvature_to_ff_center,
                                    save_data=save_data,
                                    n_steps=self.num_steps_per_dataset,
                                    **env_kwargs)
            
            self._add_heading_info_to_combd_heading_info(data_name)

        self.test_heading_info_df.reset_index(drop=True, inplace=True)
        self.ctrl_heading_info_df.reset_index(drop=True, inplace=True)

        if save_data:
            for test_or_control in ['test', 'control']:
                path = self.dict_of_combd_heading_info_folder_path[test_or_control]
                df_name = find_stops_near_ff_utils.find_df_name('monkey_agent', ref_point_mode, ref_point_value)
                df_path = os.path.join(path, df_name)
                os.makedirs(path, exist_ok=True)
                self.test_heading_info_df.to_csv(df_path)
                print(f'Stored new combd_heading_df_x_sessions for {test_or_control} data in {df_path}')
    

    def get_test_and_ctrl_heading_info_df_across_sessions(self,
                                                        num_datasets_to_collect=1,
                                                        ref_point_mode='distance', ref_point_value=-150,
                                                        heading_info_df_exists_ok=True, 
                                                        combd_heading_df_x_sessions_exists_ok=True,
                                                        stops_near_ff_df_exists_ok=True,
                                                        curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25],
                                                        use_curvature_to_ff_center=False,
                                                        save_data=True,
                                                        **env_kwargs
                                                        ):                                                       
        try:
            if combd_heading_df_x_sessions_exists_ok:
                self.retrieve_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
                if (len(self.ctrl_heading_info_df) == 0) or (len(self.test_heading_info_df) == 0):
                    raise Exception('Empty combd_heading_df_x_sessions.')
            else:
                raise Exception()
            
        except Exception as e:
            print(f'Will make new combd_heading_df_x_sessions for the agent because {e}.')
            self.make_combd_heading_df_x_sessions(num_steps_per_dataset=self.num_steps_per_dataset, num_datasets_to_collect=num_datasets_to_collect,
                                            ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                            heading_info_df_exists_ok=heading_info_df_exists_ok, 
                                            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                            curv_of_traj_mode=curv_of_traj_mode, window_for_curv_of_traj=window_for_curv_of_traj,
                                            use_curvature_to_ff_center=use_curvature_to_ff_center,
                                            save_data=save_data,
                                            **env_kwargs)

        
    def _add_plan_xy_to_combd_plan_xy(self, data_name):
        plan_x_both = self.pfa.plan_x_both.copy()
        plan_y_both = self.pfa.plan_y_both.copy()
        plan_x_both['data_name'] = data_name
        plan_y_both['data_name'] = data_name
        self.combd_plan_x_both = pd.concat([self.combd_plan_x_both, plan_x_both], axis=0)
        self.combd_plan_y_both = pd.concat([self.combd_plan_y_both, plan_y_both], axis=0)
        

    def _add_heading_info_to_combd_heading_info(self, data_name):
        self.test_heading_info_df = self.pfa.test_heading_info_df.copy()
        self.ctrl_heading_info_df = self.pfa.ctrl_heading_info_df.copy()
        self.test_heading_info_df['data_name'] = data_name
        self.ctrl_heading_info_df['data_name'] = data_name
        self.test_heading_info_df['whether_test'] = 1
        self.ctrl_heading_info_df['whether_test'] = 0
        self.test_combd_heading_df_x_sessions = pd.concat([self.test_combd_heading_df_x_sessions, self.test_heading_info_df], axis=0)
        self.ctrl_combd_heading_df_x_sessions = pd.concat([self.ctrl_combd_heading_df_x_sessions, self.ctrl_heading_info_df], axis=0)

