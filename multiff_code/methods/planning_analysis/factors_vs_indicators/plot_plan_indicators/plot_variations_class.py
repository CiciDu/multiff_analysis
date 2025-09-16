
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_class, plot_variations_utils
from planning_analysis.factors_vs_indicators import process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import parent_assembler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class _PlotVariations:

    def __init__(self):
        pass

    def plot_median_heading(self,
                            all_ref_median_info=None,
                            x_var_column_list=[
                                'ref_point_value'],
                            fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                          'key_for_split': 'ff_seen'},
                            changeable_variables=[
                                'whether_even_out_dist'],
                            columns_to_find_unique_combinations_for_color=[],
                            columns_to_find_unique_combinations_for_line=[],
                            add_ci_bounds=True,
                            use_subplots_based_on_changeable_variables=False,
                            is_difference=False,
                            ):

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_heading

        all_ref_median_info['diff_in_abs_angle_to_nxt_ff_median'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_median'] * 180/np.pi

        if add_ci_bounds:
            all_ref_median_info['ci_lower'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_ci_low_95'] * 180/np.pi
            all_ref_median_info['ci_upper'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_ci_high_95'] * 180/np.pi

        if is_difference:
            all_ref_median_info = all_ref_median_info[all_ref_median_info['test_or_control'] == 'difference']
        else:
            all_ref_median_info = all_ref_median_info[all_ref_median_info['test_or_control'] != 'difference']

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(all_ref_median_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                                                                   # var_to_determine_x_offset_direction=None,
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                                                   use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)

        if (len(x_var_column_list) == 1) and ('ref_point_value' in x_var_column_list):
            if is_difference:
                self.fig.update_layout(
                    title="Test vs Control Difference in Absolute Angle to Next FF (Median ± 95% BCa CI)",
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Difference in Median Angle(°)"
                )
            else:
                self.fig.update_layout(
                    title="Absolute Angle to Next FF (Median ± 95% BCa CI)",
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Angle (°)"
                )

    def plot_median_curv(self,
                         all_ref_median_info=None,
                         x_var_column_list=['ref_point_value'],
                         fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                       'key_for_split': 'ff_seen'},
                         changeable_variables=[
                             'whether_even_out_dist'],
                         columns_to_find_unique_combinations_for_color=[
                             'curv_traj_window_before_stop'],
                         columns_to_find_unique_combinations_for_line=[],
                         add_ci_bounds=True,
                         use_subplots_based_on_changeable_variables=False,
                         is_difference=False):
        
        # note: d_curv has already been converted to degrees/m in furnish_diff_in_curv_df function

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_curv

        if add_ci_bounds:
            all_ref_median_info['ci_lower'] = all_ref_median_info['diff_in_abs_d_curv_ci_low_95']
            all_ref_median_info['ci_upper'] = all_ref_median_info['diff_in_abs_d_curv_ci_high_95']

        if is_difference:
            all_ref_median_info = all_ref_median_info[all_ref_median_info['test_or_control'] == 'difference']
        else:
            all_ref_median_info = all_ref_median_info[all_ref_median_info['test_or_control'] != 'difference']

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(all_ref_median_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='diff_in_abs_d_curv_median',
                                                                                                   # var_to_determine_x_offset_direction=None,
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                                                   use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)

        if (len(x_var_column_list) == 1) and ('ref_point_value' in x_var_column_list):
            if is_difference:
                self.fig.update_layout(
                    title="Test vs Control Difference in Curvature (Median ± 95% BCa CI)",
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Difference in Median Curvature (°/m)"
                )
            else:
                self.fig.update_layout(
                    title="Difference in Curvature Across Two Arc Pairs (Median ± 95% BCa CI)",
                    xaxis_title="Reference Distance (cm)",
                    yaxis_title="Curvature (°/m)"
                )

    def plot_same_side_percentage(self,
                                  perc_info=None,
                                  x_var_column_list=['monkey_name'],
                                  fixed_variable_values_to_use={
                                      'if_test_nxt_ff_group_appear_after_stop': 'flexible'},
                                  changeable_variables=[
                                      'monkey_name'],
                                  columns_to_find_unique_combinations_for_color=[],
                                  add_ci_bounds=True,
                                  use_subplots_based_on_changeable_variables=False,
                                  show_fig=True,
                                  is_difference=False):

        if perc_info is None:
            perc_info = self.pooled_perc_info

        if add_ci_bounds:
            perc_info['ci_lower'] = perc_info['perc_ci_low_95']
            perc_info['ci_upper'] = perc_info['perc_ci_high_95']
            
        if is_difference:
            perc_info = perc_info[perc_info['test_or_control'] == 'difference']
        else:
            perc_info = perc_info[perc_info['test_or_control'] != 'difference']

        perc_info = process_variations_utils.make_new_df_for_plotly_comparison(
            perc_info, match_rows_based_on_ref_columns_only=False)

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(perc_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='perc',
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables
                                                                                                   )

        self.fig.update_layout(
            title="Same-Side Stop Rate (Median ± 2 Bootstrap SE)",
            xaxis_title=None,
            yaxis_title="Same-Side Stop Rate"
        )
        if show_fig:
            self.fig.show()

    def plot_same_side_percentage_across_monkeys(self,
                                                 perc_info=None,
                                                 x_var_column_list=[
                                                     'monkey_name'],
                                                 fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                               'key_for_split': 'ff_seen',
                                                                               'whether_even_out_dist': False,
                                                                               },
                                                 changeable_variables=[],  # 'key_for_split'
                                                 columns_to_find_unique_combinations_for_color=[],
                                                 add_ci_bounds=True,
                                                 ):

        self.plot_same_side_percentage(perc_info=perc_info,
                                       x_var_column_list=x_var_column_list,
                                       fixed_variable_values_to_use=fixed_variable_values_to_use,
                                       changeable_variables=changeable_variables,
                                       columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                       add_ci_bounds=add_ci_bounds,
                                       use_subplots_based_on_changeable_variables=True)

    # shared engine

    def _plot_median_with_difference(self, *,
                                     all_ref_median_info,
                                     x_var_column_list,
                                     fixed_variable_values_to_use,
                                     changeable_variables,
                                     columns_to_find_unique_combinations_for_color,
                                     columns_to_find_unique_combinations_for_line,
                                     use_subplots_based_on_changeable_variables,
                                     y_var_column,
                                     main_y_title,
                                     diff_y_title,
                                     overall_title,
                                     constant_marker_size=None,
                                     ):
        # split
        main_data = all_ref_median_info[all_ref_median_info['test_or_control'] != 'difference'].copy(
        )
        difference_data = all_ref_median_info[all_ref_median_info['test_or_control'] == 'difference'].copy(
        )

        # common kwargs
        common = dict(
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            x_var_column_list=x_var_column_list,
            y_var_column=y_var_column,
            var_to_determine_x_offset_direction='test_or_control',
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
        )

        # children
        self.main_fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(
            main_data, **common, is_difference=False, constant_marker_size=constant_marker_size)
        self.diff_fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(
            difference_data, **common, is_difference=True, constant_marker_size=constant_marker_size)

        if not hasattr(self, 'parent_assembler'):
            self.parent_assembler = parent_assembler.ParentFigureAssembler(
                x_title="Optimal Arc Type")

        # parent
        self.fig = self.parent_assembler.assemble(
            self.main_fig, self.diff_fig,
            main_y_title=main_y_title,
            diff_y_title=diff_y_title,
            overall_title=overall_title
        )
        return self.fig

    # public wrappers (unchanged names)
    def plot_median_curv_across_monkeys_and_arc_types_with_difference(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen',
                                      'whether_even_out_dist': False,
                                      'curv_traj_window_before_stop': '[-25, 0]'
                                      },
        changeable_variables=['opt_arc_type', 'monkey_name'],
        columns_to_find_unique_combinations_for_color=[],
        columns_to_find_unique_combinations_for_line=[],
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=True,
    ):
        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_curv
        if add_ci_bounds:
            all_ref_median_info['ci_lower'] = all_ref_median_info['diff_in_abs_d_curv_ci_low_95']
            all_ref_median_info['ci_upper'] = all_ref_median_info['diff_in_abs_d_curv_ci_high_95']
        return self._plot_median_with_difference(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            y_var_column='diff_in_abs_d_curv_median',
            main_y_title="Median Curvature (°/m)",
            diff_y_title="Difference in Median Curvature (°/m)",
            overall_title="Difference in Curvature Across Two Arc Pairs: Test vs Control",
        )

    def plot_median_heading_across_monkeys_and_arc_types_with_difference(
        self,
        all_ref_median_info=None,
        x_var_column_list=['ref_point_value'],
        fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                      'key_for_split': 'ff_seen',
                                      'whether_even_out_dist': False,
                                      'curv_traj_window_before_stop': '[-25, 0]'
                                      },
        changeable_variables=['opt_arc_type', 'monkey_name'],
        columns_to_find_unique_combinations_for_color=[],
        columns_to_find_unique_combinations_for_line=[],
        add_ci_bounds=True,
        use_subplots_based_on_changeable_variables=True,
        constant_marker_size=12,
    ):

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_heading
            
        all_ref_median_info['diff_in_abs_angle_to_nxt_ff_median'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_median'] * 180/np.pi

        if add_ci_bounds:
            all_ref_median_info['ci_lower'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_ci_low_95'] * 180/np.pi
            all_ref_median_info['ci_upper'] = all_ref_median_info['diff_in_abs_angle_to_nxt_ff_ci_high_95'] * 180/np.pi
        return self._plot_median_with_difference(
            all_ref_median_info=all_ref_median_info,
            x_var_column_list=x_var_column_list,
            fixed_variable_values_to_use=fixed_variable_values_to_use,
            changeable_variables=changeable_variables,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables,
            y_var_column='diff_in_abs_angle_to_nxt_ff_median',
            main_y_title="Median Angle (°)",
            diff_y_title="Difference in Median Angle (°)",
            overall_title="Absolute Angle to Next FF: Test vs Control",
            constant_marker_size=constant_marker_size,
        )

    def build_child_fig(self, data, *,
                        fixed_variable_values_to_use,
                        changeable_variables,
                        x_var_column_list,
                        y_var_column,
                        var_to_determine_x_offset_direction='test_or_control',
                        columns_to_find_unique_combinations_for_color=None,
                        columns_to_find_unique_combinations_for_line=None,
                        use_subplots_based_on_changeable_variables=False):
        columns_to_find_unique_combinations_for_color = columns_to_find_unique_combinations_for_color or []
        columns_to_find_unique_combinations_for_line = columns_to_find_unique_combinations_for_line or []

        return plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(
            data,
            fixed_variable_values_to_use,
            changeable_variables,
            x_var_column_list,
            y_var_column=y_var_column,
            var_to_determine_x_offset_direction=var_to_determine_x_offset_direction,
            columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
            columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
            use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables
        )
