import sys
from null_behaviors import show_null_trajectory
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_utils, plot_monkey_heading_helper_class
from planning_analysis.plan_factors import plan_factors_utils
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs
from visualization.matplotlib_tools import plot_behaviors_utils
from visualization import base_plot_class
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import copy

# need: monkey_plot_params


class PlotlyPlotter(base_plot_class.BasePlotter):

    cur_ff_color = 'brown'
    nxt_ff_color = 'green'

    default_monkey_plot_params = {
        "show_reward_boundary": True,
        "show_alive_fireflies": False,
        "show_visible_fireflies": False, # only meaningful when show_alive_fireflies is False
        "show_in_memory_fireflies": False, # only meaningful when show_alive_fireflies is False
        "show_visible_segments": True,
        "show_stops": True,
        "show_all_eye_positions": False,
        "show_current_eye_positions": True,
        "show_eye_positions_for_both_eyes": False,
        "show_connect_path_ff": False,
        "show_traj_points_when_making_lines": True,
        "connect_path_ff_max_distance": 500,
        "eliminate_irrelevant_points_beyond_boundaries": True,
        "show_monkey_heading": False,  # not used yet
        "show_traj_portion": False,
        "show_null_arcs_to_ff": False,
        "show_traj_color_as_speed": True,
        "show_stop_point_indices": None,
        "traj_portion": None,
        "hoverdata_multi_columns": ['rel_time'],
        "eye_positions_trace_name": 'eye_positions',
        "use_arrow_to_show_eye_positions": False,
        "show_cur_ff": False,
        "show_nxt_ff": False,
        "plot_arena_edge": True,
    }

    def __init__(self, PlotTrials_args):
        super().__init__(PlotTrials_args)

    def make_individual_plots_for_stops_near_ff_in_plotly(self, current_i, max_num_plot_to_make=5, show_fig=True,
                                                          **additional_plotting_kwargs):

        self.monkey_plot_params.update(additional_plotting_kwargs)

        for i in range(len(self.stops_near_ff_df_counted))[current_i: current_i+max_num_plot_to_make]:
            self.stops_near_ff_row = self.stops_near_ff_df_counted.iloc[i]

            diff_in_abs = self.heading_info_df_counted.iloc[i]['diff_in_abs_d_heading']
            print(f'diff_in_abs: {diff_in_abs}')

            if self.monkey_plot_params['show_null_arcs_to_ff']:
                self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_counted_points(
                    i=i)

            current_i = i+1
            self.current_plotly_key_comp, self.fig = self.plot_stops_near_ff_in_plotly_func(
                self.monkey_plot_params,
                plot_counter_i=i)

            if show_fig is True:
                self.fig.show()

        return current_i

    def plot_stops_near_ff_in_plotly_func(self,
                                          monkey_plot_params={},
                                          plot_counter_i=None,
                                          ):

        self.monkey_plot_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **self.monkey_plot_params,
            **monkey_plot_params
        }

        self.current_plotly_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(self.stops_near_ff_row, self.PlotTrials_args,
                                                                                                        self.monkey_plot_params)
        self.point_index_to_show_traj_curv = self.stops_near_ff_row.stop_point_index

        self.fig = self.make_one_monkey_plotly_plot(
            monkey_plot_params=self.monkey_plot_params)

        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.get_all_triangle_df_for_the_point_from_triangle_df_for_all_counted_points(
                self, plot_counter_i, self.current_plotly_key_comp['rotation_matrix']
            )
            self.fig = plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_plotly(
                self)

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            #self.fig = self._show_null_arcs_for_cur_and_nxt_ff_in_plotly()
            ## run code directly instead of calling function so that the method can be accessed by other classes
            rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
            self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.nxt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                    color=self.nxt_ff_color, trace_name='nxt null arc', linewidth=4)
            self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.cur_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                    color=self.cur_ff_color, trace_name='cur null arc', linewidth=3)
            
        self.fig.update_layout(
            autosize=False,
            width=900,  # Set the desired width
            height=700,  # Set the desired height
            margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
        )

        return self.current_plotly_key_comp, self.fig

    def make_one_monkey_plotly_plot(self,
                                    monkey_plot_params={}):

        m_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **monkey_plot_params
        }

        self.fig = plotly_for_monkey.plot_fireflies(
            None, self.current_plotly_key_comp['ff_df'])
        if m_params['plot_arena_edge']:
            self.fig = plotly_for_monkey.plot_arena_edge_in_plotly(self.fig)

        if self.current_plotly_key_comp['connect_path_ff_df'] is not None:
            self.fig = plotly_for_monkey.connect_points_to_points(self.fig, self.current_plotly_key_comp['connect_path_ff_df'],
                                                                  show_traj_points_when_making_lines=m_params[
                                                                      'show_traj_points_when_making_lines'],
                                                                  hoverdata_multi_columns=m_params['hoverdata_multi_columns'])

        if self.current_plotly_key_comp['show_visible_segments']:
            self.fig = plotly_for_monkey.plot_horizontal_lines_to_show_ff_visible_segments_plotly(self.fig,
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'ff_dataframe_in_duration_visible_qualified'],
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'monkey_information'],
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'rotation_matrix'], 0, 0,
                                                                                                  how_to_show_ff='square',
                                                                                                  unique_ff_indices=None)

        if m_params['traj_portion'] is not None:
            self.fig = plotly_for_monkey.plot_a_portion_of_trajectory_to_show_the_scope_for_curv(self.fig, m_params['traj_portion'],
                                                                                                 hoverdata_multi_columns=m_params['hoverdata_multi_columns'])

        if m_params['show_reward_boundary']:
            self.fig = plotly_for_monkey.plot_reward_boundary_in_plotly(
                self.fig, self.current_plotly_key_comp['ff_df'])

        if m_params['show_all_eye_positions']:
            self.fig = plotly_for_monkey.plot_eye_positions_in_plotly(self.fig, self.current_plotly_key_comp,
                                                                      show_eye_positions_for_both_eyes=m_params[
                                                                          'show_eye_positions_for_both_eyes'],
                                                                      trace_name=m_params['eye_positions_trace_name'],
                                                                      use_arrow_to_show_eye_positions=m_params['use_arrow_to_show_eye_positions'])

        self.fig = plotly_for_monkey.plot_trajectory_data(self.fig, self.current_plotly_key_comp['trajectory_df'],
                                                          hoverdata_multi_columns=m_params['hoverdata_multi_columns'],
                                                          show_color_as_time=m_params['show_all_eye_positions'],
                                                          show_traj_color_as_speed=m_params['show_traj_color_as_speed'])

        if m_params['show_cur_ff']:
            self._show_cur_ff()

        if m_params['show_nxt_ff']:
            self._show_nxt_ff()

        if m_params['show_stops'] | (m_params['show_stop_point_indices'] is not None):
            show_stop_point_indices = plotly_preparation.find_show_stop_point_indices(self.monkey_plot_params, self.current_plotly_key_comp)
            self.fig = plotly_for_monkey.plot_stops_in_plotly(self.fig, self.current_plotly_key_comp['trajectory_df'].copy(), show_stop_point_indices,
                                                              hoverdata_multi_columns=m_params['hoverdata_multi_columns'])

        self.fig = plotly_for_monkey.update_layout_and_x_and_y_limit(self.fig, self.current_plotly_key_comp,
                                                                     m_params['show_current_eye_positions'] or m_params['show_all_eye_positions'])

        # update the x label and y label
        self.fig.update_xaxes(title_text='monkey x after rotation (cm)')
        self.fig.update_yaxes(title_text='monkey y after rotation (cm)',
                              scaleanchor="x",
                              scaleratio=1)

        return self.fig

    def _show_null_arcs_for_cur_and_nxt_ff_in_plotly(self):
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.nxt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                 color=self.nxt_ff_color, trace_name='nxt null arc', linewidth=4)
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.cur_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                 color=self.cur_ff_color, trace_name='cur null arc', linewidth=3)
        return self.fig


    def find_show_stop_point_indices(self):
        show_stop_point_indices = self.monkey_plot_params.get(
            'show_stop_point_indices')

        if show_stop_point_indices is None:
            if self.monkey_plot_params['show_stops']:
                trajectory_df = self.current_plotly_key_comp['trajectory_df']
                show_stop_point_indices = trajectory_df[
                    trajectory_df['monkey_speeddummy'] == 0]['point_index'].values

        show_stop_point_indices = np.array(show_stop_point_indices).reshape(-1)

        return show_stop_point_indices


    def _show_cur_ff(self):
        self.cur_ff_index = self.stops_near_ff_row.cur_ff_index
        ff_position_rotated = np.matmul(
            self.current_plotly_key_comp['rotation_matrix'], self.ff_real_position_sorted[int(self.cur_ff_index)])
        self.fig.add_trace(go.Scatter(x=np.array([ff_position_rotated[0]]), y=np.array([ff_position_rotated[1]]),
                                      marker=dict(symbol='circle', color='pink', size=20), mode='markers',
                                      name='cur_ff'))

    def _show_nxt_ff(self):
        self.nxt_ff_index = self.stops_near_ff_row.nxt_ff_index
        ff_position_rotated = np.matmul(
            self.current_plotly_key_comp['rotation_matrix'], self.ff_real_position_sorted[int(self.nxt_ff_index)])
        self.fig.add_trace(go.Scatter(x=np.array([ff_position_rotated[0]]), y=np.array([ff_position_rotated[1]]),
                                      marker=dict(symbol='circle', color='lightblue', size=20), mode='markers',
                                      name='nxt_ff'))
