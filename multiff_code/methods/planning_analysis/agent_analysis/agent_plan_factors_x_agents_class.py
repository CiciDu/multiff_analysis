from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class
from planning_analysis.factors_vs_indicators import process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_x_sess_class
from reinforcement_learning.base_classes import rl_base_utils
from reinforcement_learning.agents.feedforward import sb3_class

import pandas as pd
import os


class PlanFactorsAcrossAgents():

    def __init__(self,
                 # this is the monkey whose data will be used for comparison
                 monkey_name='monkey_Bruno',
                 overall_folder_name='RL_models/sb3_stored_models/all_agents/agents_without_noise',
                 agent_folders=None):
        self.monkey_name = monkey_name
        self.overall_folder_name = overall_folder_name
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode
        self.agent_folders = agent_folders
        if self.agent_folders is None:
            self.agent_folders = rl_base_utils.get_agent_folders(
                path=self.overall_folder_name)
            if len(self.agent_folders) == 0:
                raise Exception('No folders with params found.')


    def process_single_agent_and_save(self,
                                    folder,
                                    intermediate_products_exist_ok=True,
                                    agent_data_exists_ok=True,
                                    num_steps_per_dataset=9000,
                                    num_datasets_to_collect=2,
                                    use_stored_data_only=False,
                                    high_level_only=False):
        """
        Process a single agent and save results.
        
        This method delegates to PlanFactorsAcrossAgentSessions.process_and_save()
        
        Parameters
        ----------
        folder : str
            Path to agent folder
        intermediate_products_exist_ok : bool, optional
            Whether to reuse existing intermediate products (default: True)
        agent_data_exists_ok : bool, optional
            Whether to reuse existing agent data (default: True)
        num_steps_per_dataset : int, optional
            Number of steps per dataset (default: 9000)
        num_datasets_to_collect : int, optional
            Number of datasets to collect (default: 2)
        use_stored_data_only : bool, optional
            If True, only use stored data (default: False)
        """

        # Determine save directory based on overall folder structure
        save_dir = self.overall_folder_name.replace(
            'all_agents',
            'all_collected_data/planning/combined_data_x_agents'
        )

        # Create session processor and run analysis
        self.pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(
            model_folder_name=folder)

        self.pfas.process_and_save(
            save_dir=save_dir,
            intermediate_products_exist_ok=intermediate_products_exist_ok,
            agent_data_exists_ok=agent_data_exists_ok,
            num_steps_per_dataset=num_steps_per_dataset,
            num_datasets_to_collect=num_datasets_to_collect,
            use_stored_data_only=use_stored_data_only,
            high_level_only=high_level_only
        )



    def make_all_ref_pooled_median_x_agents_AND_pooled_perc_x_agents(self, exists_ok=True,
                                                                    intermediate_products_exist_ok=True,
                                                                    agent_data_exists_ok=True,
                                                                    num_steps_per_dataset=9000,
                                                                    num_datasets_to_collect=2,
                                                                    use_stored_data_only=False,
                                                                    high_level_only=False,
                                                                    ):
        
        self.combd_planning_info_x_agents_path = self.overall_folder_name.replace(
            'all_agents', 'all_collected_data/planning') + '/combined_data_x_agents'
        os.makedirs(self.combd_planning_info_x_agents_path, exist_ok=True)
        all_ref_pooled_median_x_agents_path = os.path.join(
            self.combd_planning_info_x_agents_path, 'all_ref_pooled_median_info_x_agents.csv')
        pooled_perc_x_agents_path = os.path.join(
            self.combd_planning_info_x_agents_path, 'pooled_perc_info_x_agents.csv')

        if exists_ok & os.path.exists(all_ref_pooled_median_x_agents_path) & os.path.exists(pooled_perc_x_agents_path):
            self.all_ref_pooled_median_x_agents = pd.read_csv(
                all_ref_pooled_median_x_agents_path)
            self.pooled_perc_x_agents = pd.read_csv(
                pooled_perc_x_agents_path)
        else:

            self.all_ref_pooled_median_x_agents = pd.DataFrame()
            self.pooled_perc_x_agents = pd.DataFrame()
            
            for i, folder in enumerate(self.agent_folders):

                print(f'Processing agent {i+1}/{len(self.agent_folders)}')
                
                self.process_single_agent_and_save(
                    folder=folder,
                    intermediate_products_exist_ok=intermediate_products_exist_ok,
                    agent_data_exists_ok=agent_data_exists_ok,
                    num_steps_per_dataset=num_steps_per_dataset,
                    num_datasets_to_collect=num_datasets_to_collect,
                    use_stored_data_only=use_stored_data_only,
                    high_level_only=high_level_only,
                )

                if self.pfas.all_ref_pooled_median_info is not None:
                    self.all_ref_pooled_median_x_agents = pd.concat(
                        [self.all_ref_pooled_median_x_agents, self.pfas.all_ref_pooled_median_info], axis=0)
                if self.pfas.pooled_perc_info is not None:
                    self.pooled_perc_x_agents = pd.concat(
                        [self.pooled_perc_x_agents, self.pfas.pooled_perc_info], axis=0)


            self.all_ref_pooled_median_x_agents.reset_index(
                drop=True, inplace=True)
            self.pooled_perc_x_agents.reset_index(
                drop=True, inplace=True)

            self.all_ref_pooled_median_x_agents.to_csv(
                all_ref_pooled_median_x_agents_path, index=False)
            self.pooled_perc_x_agents.to_csv(
                pooled_perc_x_agents_path, index=False)

        return self.all_ref_pooled_median_x_agents, self.pooled_perc_x_agents


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
            self.monkey_median_df, self.all_ref_pooled_median_x_agents)
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
            self.monkey_perc_df, self.pooled_perc_x_agents)
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
