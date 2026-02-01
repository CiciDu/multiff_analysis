"""
Legacy class wrapper around the new function-based API in
plot_variations_functions.py.

This preserves backward compatibility for any code that still uses
    pv = _PlotVariations()
    pv.plot_median_heading(...)
while internally routing all work to the unified function-based API.
"""

import numpy as np
from planning_analysis.factors_vs_indicators.plot_plan_indicators import (
    plot_variations_functions as pvf
)
import matplotlib.pyplot as plt


class _PlotVariations:
    """
    Thin, backward-compatible wrapper that delegates to the modern,
    function-based API in plot_variations_functions.py.
    """

    def __init__(self, backend='plotly'):
        """
        backend: 'plotly' or 'matplotlib'
        """
        self.backend = backend
        self.fig = None
        self.main_fig = None
        self.diff_fig = None

    # ---------------------------------------------------------------
    # Simple 1-panel APIs
    # ---------------------------------------------------------------

    def plot_median_heading(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen'},
        changeable_variables=['whether_even_out_dist'],
        columns_to_find_unique_combinations_for_color=None,
        columns_to_find_unique_combinations_for_line=None,
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=False,
        is_difference=False,
    ):
        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_heading
        
        
        self.fig = pvf.plot_median_heading(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            add_ci_bounds=add_ci_bounds,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            is_difference=is_difference,
            backend=self.backend,
        )
        

    def plot_median_curv(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen'},
        changeable_variables=['whether_even_out_dist'],
        columns_to_find_unique_combinations_for_color=None,
        columns_to_find_unique_combinations_for_line=None,
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=False,
        is_difference=False,
    ):
        
        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_curv
        
        self.fig = pvf.plot_median_curv(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            add_ci_bounds=add_ci_bounds,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            is_difference=is_difference,
            backend=self.backend,
        )
        

    def plot_same_side_percentage(
        self,
        perc_info=None,
        x_var_column_list=['monkey_name'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible'},
        changeable_variables=['monkey_name'],
        columns_to_find_unique_combinations_for_color=None,
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=False,
        show_fig=True,
        is_difference=False,
        y_min=None,
        y_max=None,
    ):
        if perc_info is None:
            perc_info = self.pooled_perc_info
        
        
        self.fig = pvf.plot_same_side_percentage(
            perc_info=perc_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            add_ci_bounds=add_ci_bounds,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            show_fig=show_fig,
            is_difference=is_difference,
            y_min=y_min,
            y_max=y_max,
            backend=self.backend,
        )
        

    def plot_same_side_percentage_across_monkeys(
        self,
        perc_info=None,
        x_var_column_list=['monkey_name'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen',
                                      'whether_even_out_dist': False},
        changeable_variables=None,
        columns_to_find_unique_combinations_for_color=None,
        add_ci_bounds=True,
    ):
        self.fig = pvf.plot_same_side_percentage_across_monkeys(
            perc_info=perc_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            add_ci_bounds=add_ci_bounds,
            backend=self.backend,
        )
        

    # ---------------------------------------------------------------
    # Complex 2-panel “main vs diff” APIs
    # ---------------------------------------------------------------

    def plot_median_curv_across_monkeys_and_arc_types_with_difference(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen',
                                      'whether_even_out_dist': False,
                                      'curv_traj_window_before_stop': '[-25, 0]'},
        changeable_variables=['opt_arc_type', 'monkey_name'],
        columns_to_find_unique_combinations_for_color=None,
        columns_to_find_unique_combinations_for_line=None,
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=True,
    ):
        self.fig = pvf.plot_median_curv_across_monkeys_and_arc_types_with_difference(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            add_ci_bounds=add_ci_bounds,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            backend=self.backend,
        )

    def plot_median_heading_across_monkeys_and_arc_types_with_difference(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen',
                                      'whether_even_out_dist': False,
                                      'curv_traj_window_before_stop': '[-25, 0]'},
        changeable_variables=['opt_arc_type', 'monkey_name'],
        columns_to_find_unique_combinations_for_color=None,
        columns_to_find_unique_combinations_for_line=None,
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=True,
        constant_marker_size=12,
    ):
        
        self.fig = pvf.plot_median_heading_across_monkeys_and_arc_types_with_difference(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            add_ci_bounds=add_ci_bounds,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            constant_marker_size=constant_marker_size,
            backend=self.backend,
        )
        
        for num in plt.get_fignums():
            if num != self.fig.number:
                plt.close(num)
                
