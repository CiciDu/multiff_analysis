    
import sys
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


class PlotVariations(variations_base_class._VariationsBase):

    def __init__(self,
                 optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 curv_traj_window_before_stop=[-50, 0]): 
        
        super().__init__(optimal_arc_type=optimal_arc_type,
                        curv_traj_window_before_stop=curv_traj_window_before_stop)
        

    def plot_heading_in_overall_median_info(self,
                                                x_var_column_list = ['ref_point_value'],
                                                fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible',
                                                                                'key_for_split': 'ff_seen'},
                                                changeable_variables = ['whether_even_out_dist'],
                                                columns_to_find_unique_combinations_for_color = [],
                                                columns_to_find_unique_combinations_for_line = [],
                                                add_error_bars=False,
                                                use_subplots_based_on_changeable_variables=False,
                                                ):

        se_column = 'diff_in_abs_boot_med_std' if add_error_bars else None

        plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(self.all_median_info_heading, 
                                                                    fixed_variable_values_to_use,
                                                                    changeable_variables,
                                                                    x_var_column_list,
                                                                    y_var_column='diff_in_abs_50%',
                                                                    se_column=se_column,
                                                                    #var_to_determine_x_offset_direction=None,
                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                    use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)


    def plot_curv_in_overall_median_info(self,
                                            x_var_column_list = ['ref_point_value'],
                                            fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible',
                                                                            'key_for_split': 'ff_seen'},
                                            changeable_variables = ['whether_even_out_dist'],
                                            columns_to_find_unique_combinations_for_color = ['curv_traj_window_before_stop'],
                                            columns_to_find_unique_combinations_for_line = [],
                                            add_error_bars=False,
                                            use_subplots_based_on_changeable_variables=False):
        
        se_column = 'diff_in_abs_d_curv_boot_med_std' if add_error_bars else None

        plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(self.all_median_info_curv, 
                                                                    fixed_variable_values_to_use,
                                                                    changeable_variables,
                                                                    x_var_column_list,
                                                                    y_var_column='diff_in_abs_d_curv_50%',
                                                                    se_column=se_column,
                                                                    #var_to_determine_x_offset_direction=None,
                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                    use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)



    def plot_heading_in_overall_median_info_across_monkeys_and_arc_types(self,
                                                                            x_var_column_list = ['optimal_arc_type'],
                                                                            fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible',
                                                                                                            'key_for_split': 'ff_seen',
                                                                                                            'whether_even_out_dist': False,
                                                                                                            'curv_traj_window_before_stop': '[-50, 0]'
                                                                            },
                                                                            changeable_variables = ['ref_point_value', 'monkey_name'],
                                                                            columns_to_find_unique_combinations_for_color = [],
                                                                            columns_to_find_unique_combinations_for_line = [],
                                                                            ):    
                                                                   
        self.plot_heading_in_overall_median_info(x_var_column_list=x_var_column_list,
                                                    fixed_variable_values_to_use=fixed_variable_values_to_use,
                                                    changeable_variables=changeable_variables,
                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                    add_error_bars=True,
                                                    use_subplots_based_on_changeable_variables=True)


    def plot_curv_in_overall_median_info_across_monkeys_and_arc_types(self, 
                                                                        x_var_column_list = ['optimal_arc_type'],
                                                                        fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible',
                                                                                                        'key_for_split': 'ff_seen',
                                                                                                        'whether_even_out_dist': False,
                                                                                                        'curv_traj_window_before_stop': '[-50, 0]'
                                                                        },
                                                                        changeable_variables = ['ref_point_value', 'monkey_name'],
                                                                        columns_to_find_unique_combinations_for_color = [],
                                                                        columns_to_find_unique_combinations_for_line = [],
                                                                        ):
                                                                    
        self.plot_curv_in_overall_median_info(x_var_column_list=x_var_column_list,
                                                fixed_variable_values_to_use=fixed_variable_values_to_use,
                                                changeable_variables=changeable_variables,
                                                columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                add_error_bars=True,
                                                use_subplots_based_on_changeable_variables=True,
                                                )
                                                                          

    def plot_direction_in_all_perc_info(self,
                                x_var_column_list = ['key_for_split'],
                                fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible'},
                                changeable_variables = ['whether_even_out_dist'],
                                columns_to_find_unique_combinations_for_color = [],
                                add_error_bars=False,
                                use_subplots_based_on_changeable_variables=False):

        se_column = 'perc_se' if add_error_bars else None

        plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(self.all_perc_info_new, 
                                                                    fixed_variable_values_to_use,
                                                                    changeable_variables,
                                                                    x_var_column_list,
                                                                    y_var_column='perc',
                                                                    se_column=se_column,
                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                    use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables
                                                                    )


    def plot_direction_in_all_perc_info_across_monkeys(self,
                                                        x_var_column_list = ['monkey_name'],

                                                        fixed_variable_values_to_use = {'if_test_alt_ff_group_appear_after_stop': 'flexible',
                                                                                        #'key_for_split': 'ff_seen',
                                                                                        'whether_even_out_dist': False,
                                                        },
                                                        changeable_variables = ['key_for_split'],
                                                        columns_to_find_unique_combinations_for_color = []):

        self.plot_direction_in_all_perc_info(x_var_column_list=x_var_column_list,
                                        fixed_variable_values_to_use=fixed_variable_values_to_use,
                                        changeable_variables=changeable_variables,
                                        columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                        add_error_bars=True,
                                        use_subplots_based_on_changeable_variables=True)
