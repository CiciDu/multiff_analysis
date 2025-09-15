
from planning_analysis.factors_vs_indicators import plot_variations_utils
from planning_analysis.factors_vs_indicators import process_variations_utils


class _PlotVariations:

    def __init__(self):
        pass

    def plot_heading_in_all_ref_median_info(self,
                                            all_ref_median_info=None,
                                            x_var_column_list=[
                                                'ref_point_value'],
                                            fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                          'key_for_split': 'ff_seen'},
                                            changeable_variables=[
                                                'whether_even_out_dist'],
                                            columns_to_find_unique_combinations_for_color=[],
                                            columns_to_find_unique_combinations_for_line=[],
                                            add_error_bars=True,
                                            use_subplots_based_on_changeable_variables=False,
                                            ):

        se_column = 'diff_in_abs_angle_to_nxt_ff_ci_95' if add_error_bars else None

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_heading

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(all_ref_median_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                                                                   se_column=se_column,
                                                                                                   # var_to_determine_x_offset_direction=None,
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                                                   use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)

        self.fig.update_layout(
            title="Reference Point vs Absolute Angle Difference (Median ± 95% BCa CI)",
            xaxis_title="Reference Point Value (°)",
            yaxis_title="Median Angle Difference to Next FF (°)"
        )

    def plot_curv_in_all_ref_median_info(self,
                                         all_ref_median_info=None,
                                         x_var_column_list=['ref_point_value'],
                                         fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                       'key_for_split': 'ff_seen'},
                                         changeable_variables=[
                                             'whether_even_out_dist'],
                                         columns_to_find_unique_combinations_for_color=[
                                             'curv_traj_window_before_stop'],
                                         columns_to_find_unique_combinations_for_line=[],
                                         add_error_bars=True,
                                         use_subplots_based_on_changeable_variables=False):

        se_column = 'diff_in_abs_d_curv_ci_95' if add_error_bars else None

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_curv

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(all_ref_median_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='diff_in_abs_d_curv_median',
                                                                                                   se_column=se_column,
                                                                                                   # var_to_determine_x_offset_direction=None,
                                                                                                   var_to_determine_x_offset_direction='test_or_control',
                                                                                                   columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                                   columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                                                                   use_subplots_based_on_changeable_variables=use_subplots_based_on_changeable_variables)

        self.fig.update_layout(
            title="Reference Point vs Difference in Curvature Across Two Arc Pairs (Median ± 95% BCa CI)",
            xaxis_title="Reference Point Value (°)",
            yaxis_title="Difference in Curvature Across Two Arc Pairs (°/m)"
        )

    def plot_heading_in_all_ref_median_info_across_monkeys_and_arc_types(self,
                                                                         all_ref_median_info=None,
                                                                         x_var_column_list=[
                                                                             'opt_arc_type'],
                                                                         fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                                                       'key_for_split': 'ff_seen',
                                                                                                       'whether_even_out_dist': False,
                                                                                                       'curv_traj_window_before_stop': '[-25, 0]'
                                                                                                       },
                                                                         changeable_variables=[
                                                                             'ref_point_value', 'monkey_name'],
                                                                         columns_to_find_unique_combinations_for_color=[],
                                                                         columns_to_find_unique_combinations_for_line=[],
                                                                         add_error_bars=True,
                                                                         show_fig=True,
                                                                         ):

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_heading

        self.plot_heading_in_all_ref_median_info(all_ref_median_info=all_ref_median_info,
                                                 x_var_column_list=x_var_column_list,
                                                 fixed_variable_values_to_use=fixed_variable_values_to_use,
                                                 changeable_variables=changeable_variables,
                                                 columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                 columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                 add_error_bars=add_error_bars,
                                                 use_subplots_based_on_changeable_variables=True,
                                                 )

        if show_fig:
            self.fig.show()

    def plot_curv_in_all_ref_median_info_across_monkeys_and_arc_types(self,
                                                                      all_ref_median_info=None,
                                                                      x_var_column_list=[
                                                                          'opt_arc_type'],
                                                                      fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                                                    'key_for_split': 'ff_seen',
                                                                                                    'whether_even_out_dist': False,
                                                                                                    'curv_traj_window_before_stop': '[-25, 0]'
                                                                                                    },
                                                                      changeable_variables=[
                                                                          'ref_point_value', 'monkey_name'],
                                                                      columns_to_find_unique_combinations_for_color=[],
                                                                      columns_to_find_unique_combinations_for_line=[],
                                                                      add_error_bars=True,
                                                                      show_fig=True,
                                                                      ):

        if all_ref_median_info is None:
            all_ref_median_info = self.all_ref_pooled_median_info_curv

        self.plot_curv_in_all_ref_median_info(all_ref_median_info=all_ref_median_info,
                                              x_var_column_list=x_var_column_list,
                                              fixed_variable_values_to_use=fixed_variable_values_to_use,
                                              changeable_variables=changeable_variables,
                                              columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                              columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                              add_error_bars=add_error_bars,
                                              use_subplots_based_on_changeable_variables=True,
                                              )
        if show_fig:
            self.fig.show()

    def plot_direction_in_perc_info(self,
                                    perc_info=None,
                                    x_var_column_list=['key_for_split'],
                                    fixed_variable_values_to_use={
                                        'if_test_nxt_ff_group_appear_after_stop': 'flexible'},
                                    changeable_variables=[
                                        'whether_even_out_dist'],
                                    columns_to_find_unique_combinations_for_color=[],
                                    add_error_bars=True,
                                    use_subplots_based_on_changeable_variables=False,
                                    show_fig=True):

        se_column = 'perc_se' if add_error_bars else None

        if perc_info is None:
            perc_info = self.pooled_perc_info

        perc_info = process_variations_utils.make_new_df_for_plotly_comparison(
            perc_info, match_rows_based_on_ref_columns_only=False)

        self.fig = plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(perc_info,
                                                                                                   fixed_variable_values_to_use,
                                                                                                   changeable_variables,
                                                                                                   x_var_column_list,
                                                                                                   y_var_column='perc',
                                                                                                   se_column=se_column,
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

    def plot_direction_in_perc_info_across_monkeys(self,
                                                   perc_info=None,
                                                   x_var_column_list=[
                                                       'monkey_name'],
                                                   fixed_variable_values_to_use={'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                                                                 'key_for_split': 'ff_seen',
                                                                                 'whether_even_out_dist': False,
                                                                                 },
                                                   changeable_variables=[],  # 'key_for_split'
                                                   columns_to_find_unique_combinations_for_color=[],
                                                   add_error_bars=True,
                                                   ):

        self.plot_direction_in_perc_info(perc_info=perc_info,
                                         x_var_column_list=x_var_column_list,
                                         fixed_variable_values_to_use=fixed_variable_values_to_use,
                                         changeable_variables=changeable_variables,
                                         columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                         add_error_bars=add_error_bars,
                                         use_subplots_based_on_changeable_variables=True)
