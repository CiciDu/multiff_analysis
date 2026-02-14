
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class
from planning_analysis.agent_analysis import agent_plan_factors_class
from planning_analysis.factors_vs_indicators import variations_base_class
from pattern_discovery import organize_patterns_and_features
from reinforcement_learning.base_classes import rl_base_class
from pattern_discovery import patterns_and_features_class
from reinforcement_learning.agents.feedforward import sb3_class
from pattern_discovery import make_ff_dataframe

import pandas as pd
import os
import warnings
from os.path import exists
import os

# This class collects data from many agents and compares them


class AgentPatterns(variations_base_class._VariationsBase, patterns_and_features_class.PatternsAndFeatures):

    def __init__(self,
                 model_folder_name,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 opt_arc_type='opt_arc_stop_closest',
                 # note, currently we use 900s / dt = 9000 steps (15 mins)
                 backend='matplotlib',
                 ):

        super().__init__()
        self.model_folder_name = model_folder_name
        self.opt_arc_type = opt_arc_type
        rl_base_class._RLforMultifirefly.get_related_folder_names_from_model_folder_name(
            self, self.model_folder_name)
        self.monkey_name = None
        
        self.combd_patterns_and_features_folder_path = os.path.join(model_folder_name.replace(
                'all_agents', 'all_collected_data/patterns_and_features'), 'combined_data')

        self.combd_planning_info_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/planning'), 'combined_data')
        self.combd_cur_and_nxt_folder_path = os.path.join(
            self.combd_planning_info_folder_path, 'cur_and_nxt')
        self.make_key_paths()
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode

    def _combine_patterns_and_features(self,
                                    num_datasets_to_collect=2,
                                    num_steps_per_dataset=9000,
                                    save_data=True,
                                    final_products_exist_ok=True,
                                    intermediate_products_exist_ok=True,
                                    agent_data_exists_ok=True,
                                    model_folder_name=None,
                                    use_stored_data_only=False,
                                    **env_kwargs
                                    ):
      

        self.combd_pattern_frequencies = pd.DataFrame()
        self.combd_feature_statistics = pd.DataFrame()
        self.combd_all_trial_features = pd.DataFrame()
        self.combd_scatter_around_target_df = pd.DataFrame()

        self.num_steps_per_dataset = num_steps_per_dataset

        if model_folder_name is None:
            model_folder_name = self.model_folder_name

        # make sure there's enough data from agent
        for i in range(num_datasets_to_collect):
            data_name = f'data_{i}'
            print(' ')
            print('model_folder_name:', model_folder_name)
            print('data_name:', data_name)
            
        
            self.agent = sb3_class.SB3forMultifirefly(model_folder_name=model_folder_name, data_name=data_name)
            self.agent.streamline_getting_data_from_agent(
                n_steps=9000, exists_ok=True, save_data=True, retrieve_ff_flash_sorted=True)
            self.agent.ff_dataframe = make_ff_dataframe.furnish_ff_dataframe(self.agent.ff_dataframe, self.agent.ff_real_position_sorted,
                                                                        self.agent.ff_caught_T_new, self.agent.ff_life_sorted)
            self.agent.make_df_related_to_patterns_and_features()


            self.agent.pattern_frequencies['data_name'] = data_name
            self.agent.feature_statistics['data_name'] = data_name
            self.agent.all_trial_features['data_name'] = data_name
            self.agent.scatter_around_target_df['data_name'] = data_name

            self.combd_pattern_frequencies = pd.concat(
                [self.combd_pattern_frequencies, self.agent.pattern_frequencies], axis=0).reset_index(drop=True)
            self.combd_feature_statistics = pd.concat(
                [self.combd_feature_statistics, self.agent.feature_statistics], axis=0).reset_index(drop=True)
            self.combd_all_trial_features = pd.concat(
                [self.combd_all_trial_features, self.agent.all_trial_features], axis=0).reset_index(drop=True)
            self.combd_scatter_around_target_df = pd.concat(
                [self.combd_scatter_around_target_df, self.agent.scatter_around_target_df], axis=0).reset_index(drop=True)

        organize_patterns_and_features.add_dates_and_sessions(
            self.combd_pattern_frequencies)
        organize_patterns_and_features.add_dates_and_sessions(
            self.combd_feature_statistics)
        organize_patterns_and_features.add_dates_and_sessions(
            self.combd_all_trial_features)
        organize_patterns_and_features.add_dates_and_sessions(
            self.combd_scatter_around_target_df)

        self.agg_pattern_frequencies = self._make_agg_pattern_frequency()
        self.agg_feature_statistics = organize_patterns_and_features.make_feature_statistics(self.combd_all_trial_features.drop(
            columns=['data_name', 'data', 'date']), data_folder_name=None)

        if save_data:
            self._save_processed_data()