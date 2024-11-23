import sys
from planning_analysis.show_planning.get_stops_near_ff import stops_near_ff_based_on_ref_class, find_stops_near_ff_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import plan_factors_class, monkey_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_class, agent_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, variations_base_class, plot_variations_class
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
class CompareMonkeyAgentPlan(plot_variations_class.PlotVariations):

    def __init__(self,                               
                 model_folder_name='RL_models/SB3_stored_models/all_agents/env1_relu/ff3/dv10_dw10_w10_mem3'):
        self.model_folder_name = model_folder_name
        self.pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(model_folder_name=self.model_folder_name)


    def get_monkey_and_agent_overall_median_info(self):
        self.monkey_overall_median_info = make_variations_utils.combine_overall_median_info_across_monkeys_and_optimal_arc_types()
        self.agent_overall_median_info = self.pfas.make_or_retrieve_overall_median_info(process_info_for_plotting=False)

        self.overall_median_info = compare_monkey_and_agent_utils.make_both_players_df(self.monkey_overall_median_info, self.agent_overall_median_info)
        self.process_overall_median_info_to_plot_heading_and_curv()


    def get_monkey_and_agent_all_perc_info(self):
        self.monkey_all_perc_info = make_variations_utils.combine_all_perc_info_across_monkeys()
        self.agent_all_perc_info = self.pfas.make_or_retrieve_all_perc_info(process_info_for_plotting=False)

        self.all_perc_info = compare_monkey_and_agent_utils.make_both_players_df(self.monkey_all_perc_info, self.agent_all_perc_info)
        self.process_all_perc_info_to_plot_direction()