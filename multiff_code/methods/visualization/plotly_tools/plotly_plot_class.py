from planning_analysis.show_planning.cur_vs_nxt_ff import plot_monkey_heading_helper_class
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs
from visualization import base_plot_class
import numpy as np
import plotly.graph_objects as go
import copy


class PlotlyPlotter(base_plot_class.BasePlotter):

    cur_ff_color = 'red'  # 'purple' #'brown'
    nxt_ff_color = 'green'  # 'olivedrab' #'limegreen' #'darkgreen' #'violet'
    traj_arc_color = 'purple'
    # more options: '#AE76A3' - light purple

    default_monkey_plot_params = {
        "show_reward_boundary": True,
        "show_alive_fireflies": False,
        # only meaningful when show_alive_fireflies is False
        "show_visible_fireflies": False,
        # only meaningful when show_alive_fireflies is False
        "show_in_memory_fireflies": False,
        "show_visible_segments": True,
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
        "show_null_arc_to_cur_ff": False,
        "show_extended_traj_arc": False,
        "show_traj_color_as_speed": True,
        "show_stops": False,
        "show_cur_and_nxt_stops": True,
        "show_capture_stops": True,
        "show_stop_point_indices": None,
        "show_cur_and_nxt_stops_indices": None,
        "show_capture_stops_indices": None,
        "hoverdata_multi_columns": ['rel_time'],
        "eye_positions_trace_name": 'eye_positions',
        "use_arrow_to_show_eye_positions": False,
        "show_cur_ff": False,
        "show_nxt_ff": False,
        "plot_arena_edge": True,
    }


    def make_one_monkey_plotly_plot(self,
                                    monkey_plot_params={}):

        self.monkey_plot_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **monkey_plot_params
        }

        self.fig = plotly_for_monkey.plot_fireflies(
            None, self.current_plotly_key_comp['ff_df'])
        if self.monkey_plot_params['plot_arena_edge']:
            self.fig = plotly_for_monkey.plot_arena_edge_in_plotly(self.fig)

        if self.current_plotly_key_comp['connect_path_ff_df'] is not None:
            self.fig = plotly_for_monkey.connect_points_to_points(self.fig, self.current_plotly_key_comp['connect_path_ff_df'],
                                                                  show_traj_points_when_making_lines=self.monkey_plot_params[
                                                                      'show_traj_points_when_making_lines'],
                                                                  hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'])

        if self.current_plotly_key_comp['show_visible_segments']:
            varying_colors = [self.cur_ff_color, self.nxt_ff_color, '#33BBFF', '#FF337D', '#FF33D7', '#8D33FF', '#33FF64',
                              '#FF5733', '#FFB533', '#33FFBE', '#3933FF', '#FF3346',
                              '#FC33FF', '#FFEC33', '#FF5E33', '#B06B58']
            self.fig = plotly_for_monkey.plot_horizontal_lines_to_show_ff_visible_segments_plotly(self.fig,
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'ff_dataframe_in_duration_visible_qualified'],
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'monkey_information'],
                                                                                                  self.current_plotly_key_comp[
                                                                                                      'rotation_matrix'], 0, 0,
                                                                                                  how_to_show_ff='square',
                                                                                                  unique_ff_indices=None,
                                                                                                  varying_colors=varying_colors)

        if self.monkey_plot_params['show_reward_boundary']:
            self.fig = plotly_for_monkey.plot_reward_boundary_in_plotly(
                self.fig, self.current_plotly_key_comp['ff_df'])

        self.fig = plotly_for_monkey.plot_trajectory_data(self.fig, self.current_plotly_key_comp['trajectory_df'],
                                                          hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'],
                                                          show_color_as_time=self.monkey_plot_params['show_all_eye_positions'],
                                                          show_traj_color_as_speed=self.monkey_plot_params['show_traj_color_as_speed'])

        if self.monkey_plot_params['show_traj_portion']:
            self.fig = plotly_for_monkey.plot_a_portion_of_trajectory_to_show_traj_portion(self.fig, self.traj_portion,
                                                                                           hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'])

        if self.monkey_plot_params['show_all_eye_positions']:
            self.fig = plotly_for_monkey.plot_eye_positions_in_plotly(self.fig, self.current_plotly_key_comp,
                                                                      show_eye_positions_for_both_eyes=self.monkey_plot_params[
                                                                          'show_eye_positions_for_both_eyes'],
                                                                      trace_name=self.monkey_plot_params['eye_positions_trace_name'],
                                                                      use_arrow_to_show_eye_positions=self.monkey_plot_params['use_arrow_to_show_eye_positions'])

        if self.monkey_plot_params['show_cur_ff']:
            self._show_cur_ff()

        if self.monkey_plot_params['show_nxt_ff']:
            self._show_nxt_ff()

        self._update_show_stop_point_indices(self.current_plotly_key_comp['trajectory_df'])

        if self.monkey_plot_params['show_stops'] & (self.monkey_plot_params['show_stop_point_indices'] is not None):
            self.fig = plotly_for_monkey.plot_stops_in_plotly(self.fig, self.current_plotly_key_comp['trajectory_df'].copy(), self.monkey_plot_params['show_stop_point_indices'],
                                                              hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'])

        if self.monkey_plot_params['show_cur_and_nxt_stops'] & (self.monkey_plot_params['show_cur_and_nxt_stops_indices'] is not None):
            self.fig = plotly_for_monkey.plot_stops_in_plotly(self.fig, self.current_plotly_key_comp['trajectory_df'].copy(), self.monkey_plot_params['show_cur_and_nxt_stops_indices'],
                                                              hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'], name='cur_and_nxt_stops', color='red')


        if self.monkey_plot_params['show_capture_stops'] & (self.monkey_plot_params['show_capture_stops_indices'] is not None):
            self.fig = plotly_for_monkey.plot_stops_in_plotly(self.fig, self.current_plotly_key_comp['trajectory_df'].copy(), self.monkey_plot_params['show_capture_stops_indices'],
                                                              hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'], name='captures', color="#D2691E",
                                                              show_legend=True)

        self.fig = plotly_for_monkey.update_layout_and_x_and_y_limit(self.fig, self.current_plotly_key_comp,
                                                                     self.monkey_plot_params['show_current_eye_positions'] or self.monkey_plot_params['show_all_eye_positions'])

        # update the x label and y label
        self.fig.update_xaxes(title_text='monkey x after rotation (cm)')
        self.fig.update_yaxes(title_text='monkey y after rotation (cm)',
                              scaleanchor="x",
                              scaleratio=1)

        return self.fig

    def make_individual_plots_for_stops_near_ff_in_plotly(self, current_i, max_num_plot_to_make=5, show_fig=True,
                                                          **additional_plotting_kwargs):

        self.monkey_plot_params.update(additional_plotting_kwargs)

        for i in range(len(self.stops_near_ff_df_counted))[current_i: current_i+max_num_plot_to_make]:
            self.stops_near_ff_row = self.stops_near_ff_df_counted.iloc[i]
            self.stop_point_index = self.stops_near_ff_row.stop_point_index
            print(f'stop_point_index: {self.stop_point_index}')

            diff_in_abs = self.heading_info_df_counted.iloc[i]['diff_in_abs_angle_to_nxt_ff']
            print(f'diff_in_abs: {diff_in_abs}')

            if self.monkey_plot_params['show_null_arcs_to_ff'] | self.monkey_plot_params['show_null_arc_to_cur_ff']:
                self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_counted_points(
                    i=i)

            current_i = i+1
            self.current_plotly_key_comp, self.fig = self.plot_cvn_in_plotly_func(
                self.monkey_plot_params,
                plot_counter_i=i)

            if show_fig is True:
                self.fig.show()


    def plot_cvn_in_plotly_func(self,
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
            self._show_null_arcs_for_cur_and_nxt_ff_in_plotly()
        elif self.monkey_plot_params['show_null_arc_to_cur_ff']:
            self._show_null_arc_to_cur_ff_in_plotly()

        self.fig.update_layout(
            autosize=False,
            width=900,  # Set the desired width
            height=700,  # Set the desired height
            margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
        )

        return self.current_plotly_key_comp, self.fig


    def _update_show_stop_point_indices(self, trajectory_df):
        """
        Update stop indices to visualize:
        - 'show_stop_point_indices': all stop points (speed==0) OR just the capture pair when that's the only thing shown
        - 'show_cur_and_nxt_stops_indices': the specific two indices around the fast–forward (capture) event
        - 'show_capture_stops_indices': all capture points

        Expects:
        - trajectory_df has columns ['monkey_speeddummy', 'point_index']
        - self.stops_near_ff_row has attributes stop_point_index, next_stop_point_index (when used)
        """
        show_stops = bool(self.monkey_plot_params.get('show_stops', False))
        show_capture = bool(self.monkey_plot_params.get('show_cur_and_nxt_stops', False))
        show_capture_stops = bool(self.monkey_plot_params.get('show_capture_stops', False))

        # Defaults
        stop_indices = None
        cur_and_nxt_stop_indices = None
        capture_stop_indices = None

        # Helper to compute the two capture indices safely
        def _capture_pair():
            row = getattr(self, 'stops_near_ff_row', None)
            if row is None:
                return None
            try:
                i = int(row.stop_point_index)
                j = int(row.next_stop_point_index)
                return [i, j]
            except (AttributeError, TypeError, ValueError):
                print(f'Error in getting cur_ff_index and nxt_ff_index')
                return None

        # All stop points (speed == 0)
        if show_stops:
            stop_indices = (
                trajectory_df.loc[trajectory_df['monkey_speeddummy'].eq(
                    0), 'point_index']
                .dropna()
                .astype(int)
                .to_numpy()
            )
            
        if show_capture_stops:
            capture_df = self.closest_stop_to_capture_df
            capture_stop_indices = capture_df[capture_df['time'].between(
                self.current_plotly_key_comp['duration_to_plot'][0], self.current_plotly_key_comp['duration_to_plot'][1])]['point_index'].values

        # Specific capture pair
        if show_capture:
            cur_and_nxt_stop_indices = _capture_pair()

        # Write back
        self.monkey_plot_params['show_stop_point_indices'] = stop_indices
        self.monkey_plot_params['show_cur_and_nxt_stops_indices'] = cur_and_nxt_stop_indices
        self.monkey_plot_params['show_capture_stops_indices'] = capture_stop_indices


    def _show_null_arcs_for_cur_and_nxt_ff_in_plotly(self):
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.nxt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                 color=self.nxt_ff_color, trace_name='nxt null arc', linewidth=3, opacity=0.9)
        self._show_null_arc_to_cur_ff_in_plotly()                                                         
        return self.fig

    def _show_null_arc_to_cur_ff_in_plotly(self):
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.cur_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                 color=self.cur_ff_color, trace_name='cur null arc', linewidth=3, opacity=0.9)
        return self.fig


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
