import sys
from null_behaviors import show_null_trajectory
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_utils, plot_monkey_heading_helper_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs
from visualization.matplotlib_tools import plot_behaviors_utils
from visualization import base_plot_class
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class MatplotlibPlotter(base_plot_class.BasePlotter):

    def __init__(self, PlotTrials_args):
        super().__init__(PlotTrials_args)

    def make_individual_plots_for_stops_near_ff_in_mpl(self, current_i, max_num_plot_to_make=5, additional_plotting_kwargs={'show_connect_path_ff_specific_indices': None, 'show_ff_indices': True},
                                                       show_position_in_scatter_plot=True, show_monkey_heading=True, show_null_arcs=True):

        for i in range(len(self.stops_near_ff_df_counted))[current_i:current_i+max_num_plot_to_make]:
            stops_near_ff_row = self.stops_near_ff_df_counted.iloc[i]
            heading_row = self.heading_info_df_counted.iloc[i]
            diff_in_abs = heading_row['diff_in_abs_angle_to_nxt_ff']
            print(f'diff_in_abs: {diff_in_abs}')

            print('nxt_ff_index:', stops_near_ff_row.nxt_ff_index)
            print('cur_ff_index:', stops_near_ff_row.cur_ff_index)

            current_i = i+1

            fig, R, x0, y0 = plot_stops_near_ff_utils.plot_stops_near_ff_func(stops_near_ff_row, self.monkey_information, self.ff_real_position_sorted, self.ff_dataframe, self.null_arcs_plotting_kwargs, self.PlotTrials_args,
                                                                              ff_max_distance_to_path_to_show_visible_segments=None,
                                                                              additional_plotting_kwargs=additional_plotting_kwargs
                                                                              )

            axes = fig.axes[0]
            if show_monkey_heading:
                plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_matplotlib(
                    self, axes, i, R, x0, y0)

            if show_null_arcs:
                current_arc_point_index = self.cur_null_arc_info_for_counted_points.arc_point_index.iloc[
                    i]
                self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_counted_points(
                    i=i)
                axes = plot_stops_near_ff_utils.show_null_arcs_func(axes, current_arc_point_index, self.monkey_information, R, x0=x0, y0=y0,
                                                                    cur_null_arc_info_for_the_point=self.cur_null_arc_info_for_the_point,
                                                                    nxt_null_arc_info_for_the_point=self.nxt_null_arc_info_for_the_point,
                                                                    )

            if show_position_in_scatter_plot:
                axes = find_stops_near_ff_utils.plot_relationship(
                    self.nxt_curv_counted, self.traj_curv_counted, show_plot=False, change_units_to_degrees_per_m=self.overall_params['change_units_to_degrees_per_m'])
                axes.scatter(
                    self.traj_curv_counted[i], self.nxt_curv_counted[i], color='red')
            plt.show()

        return current_i
