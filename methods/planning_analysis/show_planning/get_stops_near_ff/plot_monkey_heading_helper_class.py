import sys
from data_wrangling import specific_utils, base_processing_class
from null_behaviors import show_null_trajectory, curvature_utils, curv_of_traj_utils
from planning_analysis.show_planning import show_planning_utils, nxt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, plot_stops_near_ff_utils, plot_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs
from visualization.dash_tools import dash_utils
from visualization.matplotlib_tools import monkey_heading_functions, plot_behaviors_utils
from planning_analysis.test_params_for_planning import params_utils
from eye_position_analysis import eye_positions
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go


class PlotMonkeyHeadingHelper():

    # this is for individual plot
    def find_all_mheading_for_counted_points(self):
        self.mheading_for_traj_for_all_counted_points = monkey_heading_functions.find_mheading_at_two_ends_of_traj_of_points(
            self.ref_point_index_counted, self.curv_of_traj_df, self.monkey_information)

        # get mheading for the ends of cur null arc and nxt null arc
        self.mheading_for_cur_ff_for_all_counted_points = monkey_heading_functions.find_mheading_in_xy_for_null_curv(
            self.cur_null_arc_info_for_counted_points)
        self.mheading_for_nxt_ff_for_all_counted_points = monkey_heading_functions.find_mheading_in_xy_for_null_curv(
            self.nxt_null_arc_info_for_counted_points)

        # also get the mheading for the stop and next stop
        self.mheading_for_stop_for_all_counted_points = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_df_counted['stop_point_index'].values, self.monkey_information)
        self.mheading_for_next_stop_for_all_counted_points = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_df_counted['next_stop_point_index'].values, self.monkey_information)
        self.mheading_for_before_stop_for_all_counted_points = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_df_counted['point_index_before_stop'].values, self.monkey_information)
        if len(self.stops_near_ff_df_counted) != len(self.stop_point_index_counted):
            raise ValueError(
                'stop_point_index_counted and stops_near_ff_df_counted have different lengths')

    def get_all_triangle_df_for_the_point_from_triangle_df_for_all_counted_points(self, i, R):
        self.traj_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_traj_for_all_counted_points, i, R)
        self.cur_ff_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_cur_ff_for_all_counted_points, i, R)
        self.nxt_ff_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_nxt_ff_for_all_counted_points, i, R)
        self.stop_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_stop_for_all_counted_points, i, R)
        self.next_stop_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_next_stop_for_all_counted_points, i, R)
        self.before_stop_triangle_df = monkey_heading_functions.get_triangle_df_for_the_point_from_mheading_for_all_counted_points(
            self.mheading_for_before_stop_for_all_counted_points, i, R)

    # this is for dash

    def find_all_mheading_and_triangle_df_for_the_duration(self):
        self.mheading_for_traj_in_duration = monkey_heading_functions.find_mheading_dict_from_curv_of_traj(
            self.curv_of_traj_df_in_duration)
        self.traj_triangle_df_in_duration = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_traj_in_duration, self.current_plotly_key_comp['rotation_matrix'], point_index=self.curv_of_traj_df_in_duration['point_index'].values)

        self._find_mheading_and_triangle_df_for_stop_for_the_duration()

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self._find_mheading_and_triangle_df_for_null_arcs_for_the_duration()

        if self.overall_params['heading_instead_of_curv']:
            mheading_before_stop_row = self.mheading_before_stop[
                self.mheading_before_stop['stop_point_index'] == self.stop_point_index]
            self.before_stop_triangle_df = monkey_heading_functions.turn_mheading_into_triangle_df(
                mheading_before_stop_row, self.current_plotly_key_comp['rotation_matrix'])

    def show_all_monkey_heading_in_plotly(self):
        """
        Show all monkey headings in a Plotly plot.

        Returns:
        - self.fig: Updated Plotly figure with all monkey headings.
        """

        def plot_heading(fig, triangle_df, trace_name_prefix, color, linewidth=1):
            return plotly_for_monkey.plot_triangles_to_show_monkey_heading_in_xy_in_plotly(
                fig, triangle_df, trace_name_prefix=trace_name_prefix, color=color, linewidth=linewidth
            )

        # Plot current headings
        self.fig = plot_heading(
            self.fig, self.traj_triangle_df, 'monkey heading on trajectory', 'yellow')

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = plot_heading(
                self.fig, self.cur_ff_triangle_df, 'monkey heading for cur ff', self.cur_ff_color)
            self.fig = plot_heading(
                self.fig, self.nxt_ff_triangle_df, 'monkey heading for nxt ff', self.nxt_ff_color)

        # Plot heading at the point right before stop
        if self.overall_params['heading_instead_of_curv']:
            self.fig = plot_heading(
                self.fig, self.before_stop_triangle_df, 'monkey heading before stop', 'green')

        # Plot heading at the stop point
        self.fig = plot_heading(
            self.fig, self.stop_triangle_df, 'stop point heading', 'pink', linewidth=6)
        self.fig = plot_heading(self.fig, self.next_stop_triangle_df,
                                'next stop point heading', 'lime', linewidth=2.5)

        return self.fig

    def show_all_monkey_heading_in_matplotlib(self, axes, i, R, x0, y0):
        PlotMonkeyHeadingHelper.get_all_triangle_df_for_the_point_from_triangle_df_for_all_counted_points(
            self, i, R)
        for triangle_df, color in zip([self.traj_triangle_df,
                                       self.cur_ff_triangle_df,
                                       self.nxt_ff_triangle_df,
                                       self.stop_triangle_df,
                                       self.next_stop_triangle_df],
                                      ['yellow', 'dodgerblue', 'orange', 'pink', 'green']):
            axes = monkey_heading_functions.plot_triangles_to_show_monkey_heading_in_xy_in_matplotlib(
                axes, triangle_df, x0=x0, y0=y0, color=color)

    def _find_mheading_and_triangle_df_for_stop_for_the_duration(self):
        # note that the info is the same for the whole duration
        self.mheading_for_stop = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_row.stop_point_index.reshape(-1, 1), self.monkey_information)
        self.mheading_for_next_stop = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_row.next_stop_point_index.reshape(-1, 1), self.monkey_information)
        self.mheading_for_before_stop = monkey_heading_functions.find_mheading_in_xy(
            self.stops_near_ff_row.point_index_before_stop.reshape(-1, 1), self.monkey_information)
        self.stop_triangle_df = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_stop, self.current_plotly_key_comp['rotation_matrix'])
        self.next_stop_triangle_df = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_next_stop, self.current_plotly_key_comp['rotation_matrix'])
        self.before_stop_triangle_df = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_before_stop, self.current_plotly_key_comp['rotation_matrix'])

    def _find_mheading_and_triangle_df_for_null_arcs_for_the_duration(self):
        self.mheading_for_cur_ff_in_duration = monkey_heading_functions.find_mheading_in_xy_for_null_curv(
            self.cur_null_arc_info_for_duration)
        self.mheading_for_nxt_ff_in_duration = monkey_heading_functions.find_mheading_in_xy_for_null_curv(
            self.nxt_null_arc_info_for_duration)
        self.cur_ff_triangle_df_in_duration = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_cur_ff_in_duration, self.current_plotly_key_comp['rotation_matrix'], point_index=self.cur_null_arc_info_for_duration['arc_point_index'].values)
        self.nxt_ff_triangle_df_in_duration = monkey_heading_functions.turn_mheading_into_triangle_df(
            self.mheading_for_nxt_ff_in_duration, self.current_plotly_key_comp['rotation_matrix'], point_index=self.nxt_null_arc_info_for_duration['arc_point_index'].values)

    def _get_all_triangle_df_for_the_point_from_triangle_df_in_duration(self):
        # note that we already have stop_triangle_df, next_stop_triangle_df and before_stop_triangle_df because they are the same for the whole duration
        self.traj_triangle_df = self.traj_triangle_df_in_duration[
            self.traj_triangle_df_in_duration['point_index'] == self.point_index_to_show_traj_curv].copy()
        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.cur_ff_triangle_df = self.cur_ff_triangle_df_in_duration[
                self.cur_ff_triangle_df_in_duration['arc_point_index'] == self.point_index_to_show_traj_curv].copy()
            self.nxt_ff_triangle_df = self.nxt_ff_triangle_df_in_duration[
                self.nxt_ff_triangle_df_in_duration['arc_point_index'] == self.point_index_to_show_traj_curv].copy()
