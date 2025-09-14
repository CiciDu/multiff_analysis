from planning_analysis.factors_vs_indicators import make_variations_utils
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_x_sess_class
from planning_analysis.factors_vs_indicators import make_variations_utils, variations_base_class
from planning_analysis.factors_vs_indicators import process_variations_utils


# This class collects data from many agents and compares them


class CompareMonkeyAgentPlan(variations_base_class._VariationsBase):

    def __init__(self,
                 opt_arc_type='opt_arc_stop_closest',
                 model_folder_name='RL_models/SB3_stored_models/all_agents/env1_relu/ff3/dv10_dw10_w10_mem3'):
        self.model_folder_name = model_folder_name
        self.pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(
            model_folder_name=self.model_folder_name, opt_arc_type=opt_arc_type)

    def get_monkey_and_agent_all_ref_pooled_median_info(self):
        self.monkey_all_ref_pooled_median_info = make_variations_utils.combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types()
        self.agent_all_ref_pooled_median_info = self.pfas.make_or_retrieve_all_ref_pooled_median_info(
            process_info_for_plotting=False)

        self.all_ref_pooled_median_info = compare_monkey_and_agent_utils.make_both_players_df(
            self.monkey_all_ref_pooled_median_info, self.agent_all_ref_pooled_median_info)
        self.process_all_ref_pooled_median_info_to_plot_heading_and_curv()

    def get_monkey_and_agent_pooled_perc_info(self):
        self.monkey_pooled_perc_info = make_variations_utils.combine_pooled_perc_info_across_monkeys()
        self.agent_pooled_perc_info = self.pfas.make_or_retrieve_pooled_perc_info(
        )

        self.pooled_perc_info = compare_monkey_and_agent_utils.make_both_players_df(
            self.monkey_pooled_perc_info, self.agent_pooled_perc_info)
