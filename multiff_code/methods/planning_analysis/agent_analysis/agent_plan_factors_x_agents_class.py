from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class
from planning_analysis.factors_vs_indicators import plot_variations_utils, process_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_x_sess_class
from reinforcement_learning.base_classes import rl_base_utils

import pandas as pd
import os


class PlanFactorsAcrossAgents():

    def __init__(self,
                 # this is the monkey whose data will be used for comparison
                 monkey_name='monkey_Bruno',
                 overall_folder_name='multiff_analysis/RL_models/SB3_stored_models/all_agents/env1_relu'):
        self.monkey_name = monkey_name
        self.overall_folder_name = overall_folder_name
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode

    def make_all_ref_pooled_median_info_across_agents_AND_pooled_perc_info_across_agents(self, exists_ok=True, intermediate_products_exist_ok=True, agent_data_exists_ok=True):

        self.combd_planning_info_x_agents_path = self.overall_folder_name.replace(
            'all_agents', 'all_collected_data/planning') + '/combined_data_x_agents'
        os.makedirs(self.combd_planning_info_x_agents_path, exist_ok=True)
        all_ref_pooled_median_info_across_agents_path = os.path.join(
            self.combd_planning_info_x_agents_path, 'all_ref_pooled_median_info_x_agents.csv')
        pooled_perc_info_across_agents_path = os.path.join(
            self.combd_planning_info_x_agents_path, 'pooled_perc_info_x_agents.csv')

        if exists_ok & os.path.exists(all_ref_pooled_median_info_across_agents_path) & os.path.exists(pooled_perc_info_across_agents_path):
            self.all_ref_pooled_median_info_across_agents = pd.read_csv(
                all_ref_pooled_median_info_across_agents_path)
            self.pooled_perc_info_across_agents = pd.read_csv(
                pooled_perc_info_across_agents_path)
        else:
            folders_with_params = rl_base_utils.get_folders_with_params(
                path=self.overall_folder_name)
            if len(folders_with_params) == 0:
                raise Exception('No folders with params found.')

            self.all_ref_pooled_median_info_across_agents = pd.DataFrame()
            self.pooled_perc_info_across_agents = pd.DataFrame()
            for folder in folders_with_params:
                print('folder:', folder)
                if 'best_model_postcurriculum' in folder:
                    print('Ignoring best_model_postcurriculum')
                    continue
                manifest = rl_base_utils.read_checkpoint_manifest(folder)
                if isinstance(manifest, dict) and ('env_params' in manifest):
                    params = manifest['env_params']
                else:
                    raise Exception('No env params found in manifest.')

                self.pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(
                    model_folder_name=folder)
                self.pfas.streamline_getting_y_values(
                    model_folder_name=folder, intermediate_products_exist_ok=intermediate_products_exist_ok, agent_data_exists_ok=agent_data_exists_ok, **params)

                agent_all_ref_pooled_median_info = rl_base_utils.add_essential_agent_params_info(
                    self.pfas.all_ref_pooled_median_info, params)
                self.all_ref_pooled_median_info_across_agents = pd.concat(
                    [self.all_ref_pooled_median_info_across_agents, agent_all_ref_pooled_median_info], axis=0)

                agent_all_perc_df = rl_base_utils.add_essential_agent_params_info(
                    self.pfas.pooled_perc_info, params)
                self.pooled_perc_info_across_agents = pd.concat(
                    [self.pooled_perc_info_across_agents, agent_all_perc_df], axis=0)

                # self.pfas.plot_monkey_and_agent_median_df()
                # self.pfas.plot_monkey_and_agent_perc_df()

            self.all_ref_pooled_median_info_across_agents.reset_index(
                drop=True, inplace=True)
            self.pooled_perc_info_across_agents.reset_index(
                drop=True, inplace=True)

            self.all_ref_pooled_median_info_across_agents.to_csv(
                all_ref_pooled_median_info_across_agents_path)
            self.pooled_perc_info_across_agents.to_csv(
                pooled_perc_info_across_agents_path)

        return self.all_ref_pooled_median_info_across_agents, self.pooled_perc_info_across_agents

    def get_monkey_median_df(self):
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(
            monkey_name=self.monkey_name)
        all_ref_pooled_median_info = ps.make_or_retrieve_all_ref_pooled_median_info()
        self.monkey_median_df = all_ref_pooled_median_info.copy()

    def get_monkey_perc_df(self):
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(
            monkey_name=self.monkey_name)
        pooled_perc_info = ps.make_or_retrieve_pooled_perc_info()
        self.monkey_perc_df = pooled_perc_info.copy()

    def plot_monkey_and_agent_median_df(self):
        both_players_df = compare_monkey_and_agent_utils.make_both_players_df(
            self.monkey_median_df, self.all_ref_pooled_median_info_across_agents)
        median_new_df = process_variations_utils.make_new_df_for_plotly_comparison(both_players_df,
                                                                                   match_rows_based_on_ref_columns_only=False)
        x_var_column_list = ['ref_point_value']

        fixed_variable_values_to_use = {'whether_even_out_dist': True}

        changeable_variables = []

        columns_to_find_unique_combinations_for_color = ['monkey_or_agent']
        columns_to_find_unique_combinations_for_line = []

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(median_new_df,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)

    def plot_monkey_and_agent_perc_df(self):
        both_players_df = compare_monkey_and_agent_utils.make_both_players_df(
            self.monkey_perc_df, self.pooled_perc_info_across_agents)
        perc_new_df = process_variations_utils.make_new_df_for_plotly_comparison(both_players_df,
                                                                                 match_rows_based_on_ref_columns_only=False)
        x_var_column_list = ['key_for_split']

        fixed_variable_values_to_use = {'whether_even_out_dist': True}

        changeable_variables = []  # 'ref_point_value'

        columns_to_find_unique_combinations_for_color = ['monkey_or_agent']
        columns_to_find_unique_combinations_for_line = []

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(perc_new_df,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='perc',
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)
