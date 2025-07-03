import sys
from null_behaviors import show_null_trajectory
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_utils, plot_monkey_heading_helper_class
from planning_analysis.plan_factors import plan_factors_utils
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs, plotly_plot_class
from visualization.matplotlib_tools import plot_behaviors_utils, matplotlib_plot_class
from visualization import base_plot_class
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class _PlotStopsNearFF(find_stops_near_ff_class._FindStopsNearFF):

    traj_curv_descr = 'Traj Curv: From Current Point to Right Before Stop'

    default_overall_params = {'heading_instead_of_curv': True}

    def __init__(self,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
                 optimal_arc_type='norm_opt_arc',
                 ):
        super().__init__()
        self._update_optimal_arc_type_and_related_paths(
            optimal_arc_type=optimal_arc_type)
        self.default_monkey_plot_params = plotly_plot_class.PlotlyPlotter.default_monkey_plot_params

    def prepare_to_plot_stops_near_ff(self, use_fixed_arc_length=False, fixed_arc_length=None):
        # self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(self.traj_curv_counted, self.nxt_curv_counted)
        if self.ff_dataframe is None:
            self.get_more_monkey_data()
        self.stop_point_index_counted = self.stops_near_ff_df_counted['stop_point_index'].values
        self.heading_info_df_counted = self.heading_info_df.set_index(
            'stop_point_index').loc[self.stop_point_index_counted].reset_index()
        self.heading_info_df_counted = plan_factors_utils.process_heading_info_df(
            self.heading_info_df_counted)
        self.make_PlotTrials_args()

        data_to_add = {
            'stops_near_ff_df_counted': self.stops_near_ff_df_counted,
            'stop_point_index_counted': self.stop_point_index_counted,
            'heading_info_df_counted': self.heading_info_df_counted,
            'ref_point_index_counted': self.ref_point_index_counted,
            'cur_ff_counted_df': self.cur_ff_counted_df,
            'nxt_ff_counted_df': self.nxt_ff_counted_df,
            'curv_of_traj_df': self.curv_of_traj_df,
            'overall_params': self.overall_params,
            'monkey_plot_params': self.monkey_plot_params,
        }
        
        # if traj_curv_counted or nxt_curv_counted are in self, then add them
        if hasattr(self, 'traj_curv_counted'):
            data_to_add['traj_curv_counted'] = self.traj_curv_counted
        if hasattr(self, 'nxt_curv_counted'):
            data_to_add['nxt_curv_counted'] = self.nxt_curv_counted

        self.plotly_inst = plotly_plot_class.PlotlyPlotter(self.PlotTrials_args)
        self.plotly_inst.add_additional_info_for_plotting(**data_to_add)
        self.plotly_inst.prepare_to_plot_stops_near_ff(use_fixed_arc_length=use_fixed_arc_length, fixed_arc_length=fixed_arc_length)

    
        self.mpl_inst = matplotlib_plot_class.MatplotlibPlotter(self.PlotTrials_args)
        self.mpl_inst.add_additional_info_for_plotting(**data_to_add)
        self.mpl_inst.prepare_to_plot_stops_near_ff(use_fixed_arc_length=use_fixed_arc_length, fixed_arc_length=fixed_arc_length)

    def make_individual_plots_for_stops_near_ff_in_mpl(self, current_i, max_num_plot_to_make=5, additional_plotting_kwargs={'show_connect_path_ff_specific_indices': None, 'show_ff_indices': True},
                                                       show_position_in_scatter_plot=True, show_monkey_heading=True, show_null_arcs=True):
        current_i = self.mpl_inst.make_individual_plots_for_stops_near_ff_in_mpl(current_i, max_num_plot_to_make, additional_plotting_kwargs,
            show_position_in_scatter_plot, show_monkey_heading, show_null_arcs
        )
        
        return current_i
    
    def make_individual_plots_for_stops_near_ff_in_plotly(self, current_i, max_num_plot_to_make=5,
                                                          **additional_plotting_kwargs):
        current_i = self.plotly_inst.make_individual_plots_for_stops_near_ff_in_plotly(current_i, max_num_plot_to_make, **additional_plotting_kwargs)
        return current_i
    
    def compare_test_and_control_in_polar_plots(self, max_instances_each=10, test_color='green', ctrl_color='purple',
                                                start='stop_point_index', end='next_stop_point_index', rmax=400):

        self._prepare_data_to_compare_test_and_control()

        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        axes = plot_behaviors_utils.set_polar_background_for_plotting(
            axes, rmax, color_visible_area_in_background=False)
        axes = find_stops_near_ff_utils.add_instances_to_polar_plot(axes, self.stops_near_ff_df_ctrl, self.nxt_ff_df2_ctrl, self.monkey_information,
                                                                    max_instances_each, color=ctrl_color, start=start, end=end)
        axes = find_stops_near_ff_utils.add_instances_to_polar_plot(axes, self.stops_near_ff_df_test, self.nxt_ff_df2_test, self.monkey_information,
                                                                    max_instances_each, color=test_color, start=start, end=end)
        # make a legend

        colors = [test_color, ctrl_color]
        labels = ['test', 'control']

        lines = [Line2D([0], [0], color=c, linewidth=3,
                        linestyle='solid') for c in colors]
        axes.legend(lines, labels, loc='lower right')

        plt.show()

    def compare_test_and_control_in_plotly_polar_plots(self, max_instances_each=10, test_color='green', ctrl_color='purple',
                                                       start='stop_point_index', end='next_stop_point_index'):

        if (start == 'ref_point_index') & (end == 'next_stop_point_index'):
            rmax = 600
        else:
            rmax = 350

        self._prepare_data_to_compare_test_and_control()

        fig = go.Figure()

        # Add control group instances
        fig = find_stops_near_ff_utils.add_instances_to_plotly_polar_plot(fig, self.stops_near_ff_df_ctrl, self.nxt_ff_df2_ctrl, self.monkey_information,
                                                                          max_instances_each, color=ctrl_color, point_color='red', start=start, end=end, legendgroup='Control data')

        # Add test group instances
        fig = find_stops_near_ff_utils.add_instances_to_plotly_polar_plot(fig, self.stops_near_ff_df_test, self.nxt_ff_df2_test, self.monkey_information,
                                                                          max_instances_each, color=test_color, point_color='blue', start=start, end=end, legendgroup='Test data')

        # Set up radial ticks based on rmax
        radial_ticks = list(range(25, rmax + 1, 25)) if rmax < 150 else []
        # Define custom angular tick labels
        angular_tickvals = np.linspace(
            0, 360, 8, endpoint=False)  # 12 angular positions
        # Adjusting labels as per the original logic, assuming it's meant to manipulate angular labels
        adjusted_angular_tickvals = np.copy(angular_tickvals)
        adjusted_angular_tickvals[4:8] = -adjusted_angular_tickvals[1:5][::-1]
        adjusted_angular_tickvals = -adjusted_angular_tickvals
        labels_in_degrees = [
            f"{int(val)}°" for val in adjusted_angular_tickvals]

        fig.update_layout(
            width=800,
            height=800,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, rmax],
                    tickvals=radial_ticks
                ),
                angularaxis=dict(
                    direction="clockwise",
                    tickmode='array',
                    tickvals=angular_tickvals,
                    ticktext=labels_in_degrees
                ),
            )
        )
        fig.show()
