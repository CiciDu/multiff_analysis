import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from planning_analysis.show_planning.get_stops_near_ff import stops_near_ff_based_on_ref_class, find_stops_near_ff_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import plan_factors_class, monkey_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_class, agent_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, variations_base_class
from machine_learning.RL.SB3 import sb3_for_multiff_class, rl_for_multiff_utils

from data_wrangling import basic_func, combine_info_utils
from data_wrangling import basic_func
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import gc
from os.path import exists


class PlanFactorsAcrossAgents():

    def __init__(self,
                 monkey_name='monkey_Bruno', # this is the monkey whose data will be used for comparison
                 overall_folder_name = 'RL_models/SB3_stored_models/all_agents/env1_relu'):
        self.monkey_name = monkey_name
        self.overall_folder_name = overall_folder_name
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode


    def make_overall_all_median_info_across_agents_AND_all_perc_info_across_agents(self, exists_ok=True, intermediate_products_exist_ok=True, agent_data_exists_ok=True):

        self.overall_variations_df_path = self.overall_folder_name + '/overall_variations_df'
        os.makedirs(self.overall_variations_df_path, exist_ok=True)
        overall_all_median_info_across_agents_path = os.path.join(self.overall_variations_df_path, 'self.overall_all_median_info_across_agents.csv')
        all_perc_info_across_agents_path = os.path.join(self.overall_variations_df_path, 'self.all_perc_info_across_agents.csv')


        if exists_ok & os.path.exists(overall_all_median_info_across_agents_path) & os.path.exists(all_perc_info_across_agents_path):
            self.overall_all_median_info_across_agents = pd.read_csv(overall_all_median_info_across_agents_path)
            self.all_perc_info_across_agents = pd.read_csv(all_perc_info_across_agents_path)
        else:
            folders_with_params = rl_for_multiff_utils.get_folders_with_params(path=self.overall_folder_name)
            if len(folders_with_params) == 0:
                raise Exception('No folders with params found.')
            
            self.overall_all_median_info_across_agents = pd.DataFrame()
            self.all_perc_info_across_agents = pd.DataFrame()
            for folder in folders_with_params:
                print('folder:', folder)
                if 'best_model_after_curriculum' in folder:
                    print('Ignoring best_model_after_curriculum')
                    continue
                params = rl_for_multiff_utils.retrieve_params(folder) 
                agent_name = rl_for_multiff_utils.get_agent_name_from_params(params)
                self.pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(model_folder_name=folder)
                self.pfas.streamline_getting_y_values(model_folder_name=folder, intermediate_products_exist_ok=intermediate_products_exist_ok, agent_data_exists_ok=agent_data_exists_ok, **params)

                agent_overall_all_median_info = rl_for_multiff_utils.add_essential_agent_params_info(self.pfas.overall_all_median_info, params, agent_name)
                self.overall_all_median_info_across_agents = pd.concat([self.overall_all_median_info_across_agents, agent_overall_all_median_info], axis=0)

                agent_all_perc_df = rl_for_multiff_utils.add_essential_agent_params_info(self.pfas.all_perc_info, params, agent_name)
                self.all_perc_info_across_agents = pd.concat([self.all_perc_info_across_agents, agent_all_perc_df], axis=0)

                # self.pfas.plot_monkey_and_agent_median_df()
                # self.pfas.plot_monkey_and_agent_perc_df()

            self.overall_all_median_info_across_agents.to_csv(overall_all_median_info_across_agents_path)
            self.all_perc_info_across_agents.to_csv(all_perc_info_across_agents_path)

        return self.overall_all_median_info_across_agents, self.all_perc_info_across_agents


    def get_monkey_median_df(self):
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=self.monkey_name)
        overall_all_median_info = ps.make_or_retrieve_overall_all_median_info()
        self.monkey_median_df = overall_all_median_info.copy()


    def get_monkey_perc_df(self):
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=self.monkey_name)
        all_perc_info = ps.make_or_retrieve_all_perc_info()
        self.monkey_perc_df = all_perc_info.copy()


    def plot_monkey_and_agent_median_df(self):
        both_players_df = compare_monkey_and_agent_utils.make_both_players_df(self.monkey_median_df, self.overall_all_median_info_across_agents)
        median_new_df = process_variations_utils.make_new_df_for_plotly_comparison(both_players_df, 
                                                                        match_rows_based_on_ref_columns_only=False)
        x_var_column_list = ['ref_point_value']

        fixed_variable_values_to_use = {'whether_even_out_dist': True}

        changeable_variables = []

        columns_to_find_unique_combinations_for_color = ['monkey_or_agent']
        columns_to_find_unique_combinations_for_line = []

        plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(median_new_df, 
                                                                    fixed_variable_values_to_use,
                                                                    changeable_variables,
                                                                    x_var_column_list,
                                                                    y_var_column='diff_in_abs_50%',
                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)


    def plot_monkey_and_agent_perc_df(self):
        both_players_df = compare_monkey_and_agent_utils.make_both_players_df(self.monkey_perc_df, self.all_perc_info_across_agents)
        perc_new_df = process_variations_utils.make_new_df_for_plotly_comparison(both_players_df, 
                                                                        match_rows_based_on_ref_columns_only=False)
        x_var_column_list = ['key_for_split']

        fixed_variable_values_to_use = {'whether_even_out_dist': True}

        changeable_variables = [] #'ref_point_value'

        columns_to_find_unique_combinations_for_color = ['monkey_or_agent']
        columns_to_find_unique_combinations_for_line = []

        plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(perc_new_df, 
                                                                    fixed_variable_values_to_use,
                                                                    changeable_variables,
                                                                    x_var_column_list,
                                                                    y_var_column='perc',
                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)

