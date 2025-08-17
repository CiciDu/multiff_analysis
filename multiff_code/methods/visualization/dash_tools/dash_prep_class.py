
import sys
from visualization.plotly_tools import plotly_preparation, plotly_for_time_series, plotly_for_null_arcs, plotly_for_correlation, plotly_for_monkey, plotly_plot_class
from visualization.dash_tools import dash_utils, dash_utils
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, cvn_from_ref_class
from visualization.matplotlib_tools import monkey_heading_utils
from eye_position_analysis import eye_positions
from planning_analysis.show_planning.cur_vs_nxt_ff import cvn_helper_class, find_cvn_utils, plot_cvn_class, plot_cvn_utils, plot_monkey_heading_helper_class
from null_behaviors import find_best_arc, curv_of_traj_utils, opt_arc_utils
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import raster_and_fr_plot_in_plotly
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import plotly.graph_objects as go
import copy


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class DashCartesianPreparation(cvn_from_ref_class.CurVsNxtFfFromRefClass, plotly_plot_class.PlotlyPlotter):

    def __init__(self,
                 raw_data_folder_path=None,
                 opt_arc_type='opt_arc_stop_closest'):

        super().__init__(raw_data_folder_path=raw_data_folder_path, opt_arc_type=opt_arc_type)

    def _create_raster_plot_figure(self,
                                   max_clusters_to_plot=None):
        if not hasattr(self, 'spikes_df'):
            self._make_spikes_df()
        self.fig_raster = raster_and_fr_plot_in_plotly.create_raster_plot_for_one_duration_in_plotly(
            self.spikes_df,
            self.stops_near_ff_row['stop_time'],
            self.current_plotly_key_comp['duration_to_plot'][0],
            self.current_plotly_key_comp['duration_to_plot'][1],
            max_clusters_to_plot=max_clusters_to_plot,
            rel_hover_time=self.monkey_hoverdata_value_s,
            # show_visible_segments=self.monkey_plot_params['show_visible_segments'],
            # visible_segments_info=self.visible_segments_info
        )
        return self.fig_raster

    def _create_firing_rate_plot_figure(self, max_clusters_to_plot=None):
        if not hasattr(self, 'spikes_df'):
            self._make_spikes_df()
        self.fig_fr = raster_and_fr_plot_in_plotly.create_firing_rate_plot_for_one_duration_in_plotly(
            self.spikes_df,
            self.stops_near_ff_row['stop_time'],
            self.current_plotly_key_comp['duration_to_plot'][0],
            self.current_plotly_key_comp['duration_to_plot'][1],
            max_clusters_to_plot=max_clusters_to_plot,
            rel_hover_time=self.monkey_hoverdata_value_s,
            # show_visible_segments=self.monkey_plot_params['show_visible_segments'],
            # visible_segments_info=self.visible_segments_info
        )
        return self.fig_fr

    def _update_neural_plots_based_on_monkey_hover_data(self, rel_hover_time):
        self.fig_raster.update_traces(overwrite=True, selector=dict(
            name='Hover Time'), x=[rel_hover_time, rel_hover_time])
        self.fig_fr.update_traces(overwrite=True, selector=dict(
            name='Hover Time'), x=[rel_hover_time, rel_hover_time])
        return self.fig_raster, self.fig_fr

    def _make_spikes_df(self):
        self.sampling_rate = 20000 if 'Bruno' in self.raw_data_folder_path else 30000
        self.spikes_df = neural_data_processing.make_spikes_df(self.raw_data_folder_path, self.ff_caught_T_sorted,
                                                               sampling_rate=self.sampling_rate)

    def _show_angle_to_nxt_ff_in_time_series(self):
        self.add_diff_in_abs_angle_to_nxt_ff_to_curv_of_traj_in_duration()
        temp_df = self.curv_of_traj_in_duration.copy()
        for y_column_name, color, name, symbol in [
            ('diff_in_abs_angle_to_nxt_ff', 'cyan',
             'diff in abs angle to nxt ff', 'x'),
            # ('angle_opt_cur_end_to_nxt_ff', 'yellow', 'angle from cur arc end to nxt ff', 'x'),
            # ('angle_from_stop_to_nxt_ff', 'magenta', 'angle from stop to nxt ff', 'x'),
            # ('diff_in_angle_to_nxt_ff', 'grey', 'diff in angle to nxt ff', 'x'),
        ]:
            temp_df[y_column_name] = temp_df[y_column_name] * \
                180 / np.pi  # convert radians to degrees
            self.fig_time_series_s = plotly_for_time_series.add_to_time_series_plot(
                self.fig_time_series_s, temp_df, name=name, color=color, x_column_name='rel_time', y_column_name=y_column_name, symbol=symbol)
            self.fig_time_series_cm = plotly_for_time_series.add_to_time_series_plot(
                self.fig_time_series_cm, temp_df, name=name, color=color, x_column_name='rel_distance', y_column_name=y_column_name, symbol=symbol)

    def _show_d_curv_to_nxt_ff_in_time_series_plot_func(self, y_column_name='d_curv_monkey'):
        self.add_diff_in_curv_info_to_curv_of_traj_in_duration()
        for y_column_name, color, name, symbol in [
            ('diff_in_abs_d_curv', 'purple', 'diff in abs d curv', 'cross'),
            # ('d_curv_null_arc', 'blue','d_curv_null_arc', 'cross'),
            # ('d_curv_monkey', 'green', 'd_curv_monkey', 'cross'),
            # ('diff_in_d_curv', 'khaki', 'diff in d curv', 'cross'),

        ]:
            self.fig_time_series_s = plotly_for_time_series.add_to_time_series_plot(
                self.fig_time_series_s, self.curv_of_traj_in_duration, name=name, color=color, x_column_name='rel_time', y_column_name=y_column_name, symbol=symbol)
            self.fig_time_series_cm = plotly_for_time_series.add_to_time_series_plot(
                self.fig_time_series_cm, self.curv_of_traj_in_duration, name=name, color=color, x_column_name='rel_distance', y_column_name=y_column_name, symbol=symbol)

    def _prepare_static_main_plots(self,
                                   show_static_plots=False,
                                   modify_non_negative_ver_theta=True):

        self.stop_point_index = self.stops_near_ff_row.stop_point_index
        self.hoverdata_column = self.ref_point_column
        self.hoverdata_multi_columns = dash_utils.find_hoverdata_multi_columns(
            self.hoverdata_column)
        self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(
            self.stop_point_index)

        self.fig, self.fig_time_series_combd, self.fig_time_series_natural_y_range = self._produce_initial_plots()

        if show_static_plots:
            dash_utils.show_a_static_plot(self.fig_time_series_s)

        if modify_non_negative_ver_theta:
            self.monkey_information = self.monkey_information.copy()
            self.monkey_information.loc[self.monkey_information['LDz']
                                        >= 0, 'LDz'] = -0.001
            self.monkey_information.loc[self.monkey_information['RDz']
                                        >= 0, 'RDz'] = -0.001
            self.monkey_information = eye_positions.convert_eye_positions_in_monkey_information(self.monkey_information, add_left_and_right_eyes_info=True,
                                                                                                interocular_dist=self.interocular_dist)

        self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._make_two_fig_scatter_or_reg()

    def _make_two_fig_scatter_or_reg(self):
        self.fig_scatter_or_reg = plotly_for_correlation.make_regression_plot_in_plotly(
            heading_info_df=self.heading_info_df,
            current_stop_point_index_to_mark=self.stop_point_index,
            x_var_column='angle_from_stop_to_nxt_ff',
            y_var_column='angle_opt_cur_end_to_nxt_ff',
            equal_aspect=True,
        )

        self.fig_scatter_or_reg2 = plotly_for_correlation.make_regression_plot_in_plotly(
            heading_info_df=self.heading_info_df,
            current_stop_point_index_to_mark=self.stop_point_index,
            x_var_column='cum_distance_between_two_stops',
            y_var_column='diff_in_abs_angle_to_nxt_ff'
        )

        return self.fig_scatter_or_reg, self.fig_scatter_or_reg2

    def _find_trajectory_ref_row(self):
        ref_point_index = self.nxt_ff_df_from_ref[self.nxt_ff_df_from_ref['stop_point_index']
                                                  == self.stop_point_index]['point_index'].item()
        trajectory_df = self.current_plotly_key_comp['trajectory_df']
        self.trajectory_ref_row = trajectory_df[trajectory_df['point_index']
                                                <= ref_point_index]
        if len(self.trajectory_ref_row) > 0:
            self.trajectory_ref_row = self.trajectory_ref_row.iloc[-1]
        return self.trajectory_ref_row

    def _find_trajectory_next_stop_row(self):
        next_stop_point_index = self.stops_near_ff_df[self.stops_near_ff_df['stop_point_index']
                                                      == self.stop_point_index]['next_stop_point_index'].item()
        trajectory_df = self.current_plotly_key_comp['trajectory_df']
        self.trajectory_next_stop_row = trajectory_df[trajectory_df['point_index']
                                                      == next_stop_point_index].iloc[0]
        return self.trajectory_next_stop_row

    def _update_fig_time_series_combd_y_range(self):
        margin = 10
        if self.use_two_y_axes:
            self.fig_time_series_combd.update_layout(yaxis=dict(range=[self.fig_time_series_natural_y_range[0]-margin, self.fig_time_series_natural_y_range[1]+margin]),
                                                     yaxis3=dict(range=[self.fig_time_series_natural_y_range[0]-margin, self.fig_time_series_natural_y_range[1]+margin]))
        else:
            self.fig_time_series_combd.update_layout(yaxis=dict(range=[self.fig_time_series_natural_y_range[0]-margin, self.fig_time_series_natural_y_range[1]+margin]),
                                                     yaxis2=dict(range=[self.fig_time_series_natural_y_range[0]-margin, self.fig_time_series_natural_y_range[1]+margin]))

    def _produce_initial_plots(self):

        self.use_two_y_axes = self.time_series_plot_params['use_two_y_axes']
        self.fig = self._produce_fig_for_dash()
        self.fig_time_series_s, self.fig_time_series_cm = self._produce_fig_time_series(
            use_two_y_axes=self.use_two_y_axes)
        self.fig_time_series_natural_y_range = [np.min(self.curv_of_traj_in_duration['curv_of_traj_deg_over_cm'].values), np.max(
            self.curv_of_traj_in_duration['curv_of_traj_deg_over_cm'].values)]
        y_column_name = 'cntr_arc_curv' if self.overall_params[
            'use_curv_to_ff_center'] else 'opt_arc_curv'
        if self.time_series_plot_params['show_nxt_ff_curv_in_time_series_plot']:
            self._show_nxt_ff_curv_in_time_series_plot_func(
                y_column_name=y_column_name)
            try:
                self.fig_time_series_natural_y_range = plotly_for_time_series.update_fig_time_series_natural_y_range(
                    self.fig_time_series_natural_y_range, self.nxt_ff_curv_df, y_column_name)
            except:
                pass
        if self.time_series_plot_params['show_cur_ff_curv_in_time_series_plot']:
            self._show_cur_ff_curv_in_time_series_plot_func(
                y_column_name=y_column_name)
            try:
                self.fig_time_series_natural_y_range = plotly_for_time_series.update_fig_time_series_natural_y_range(
                    self.fig_time_series_natural_y_range, self.cur_ff_curv_df, y_column_name)
            except:
                pass
        if self.time_series_plot_params.get('show_d_curv_to_nxt_ff', False):
            self._show_d_curv_to_nxt_ff_in_time_series_plot_func()
        if self.time_series_plot_params.get('show_angle_to_nxt_ff', False):
            self._show_angle_to_nxt_ff_in_time_series()
        self.fig_time_series_combd = plotly_for_time_series.make_fig_time_series_combd(
            self.fig_time_series_s, self.fig_time_series_cm, self.use_two_y_axes)
        self._update_fig_time_series_combd_y_range()
        return self.fig, self.fig_time_series_combd, self.fig_time_series_natural_y_range

    def _get_curv_of_traj_in_duration(self):
        self.stops_near_ff_row = self.stops_near_ff_df[self.stops_near_ff_df['stop_point_index']
                                                       == self.stop_point_index].iloc[0]
        if len(self.stops_near_ff_row) == 0:
            raise ValueError('self.stop_point_index is not in self.stops_near_ff_df')
        self.curv_of_traj_in_duration = curv_of_traj_utils.find_curv_of_traj_in_duration(
            self.curv_of_traj_df, self.current_plotly_key_comp['duration_to_plot'])
        self.curv_of_traj_in_duration['rel_time'] = np.round(
            self.curv_of_traj_in_duration['time'] - self.stops_near_ff_row.stop_time, 2)
        self.curv_of_traj_in_duration['rel_distance'] = np.round(
            self.curv_of_traj_in_duration['cum_distance'] - self.stops_near_ff_row.stop_cum_distance, 2)
        if len(self.curv_of_traj_in_duration) == 0:
            print('Warning: curv_of_traj_in_duration is empty!')

        return self.curv_of_traj_in_duration

    def _produce_fig_time_series(self, use_two_y_axes=False):
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_time_series.find_monkey_hoverdata_value_for_both_fig_time_series(
            self.hoverdata_column, self.monkey_hoverdata_value, self.current_plotly_key_comp['trajectory_df'])

        if self.monkey_plot_params['show_visible_segments'] is True:
            self.visible_segments_info = {'ff_info': self.current_plotly_key_comp['ff_dataframe_in_duration_visible_qualified'],
                                          'monkey_information': self.monkey_information,
                                          'stops_near_ff_row': self.stops_near_ff_row
                                          }
        else:
            self.visible_segments_info = None

        self.curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
            self.curv_of_traj_params['curv_of_traj_mode'], self.curv_of_traj_params['window_for_curv_of_traj'])

        self.fig_time_series_s = plotly_for_time_series.make_the_initial_fig_time_series(self.curv_of_traj_in_duration, self.monkey_hoverdata_value_s, self.cur_ff_color, self.nxt_ff_color, trajectory_ref_row=self.trajectory_ref_row,
                                                                                         use_two_y_axes=use_two_y_axes, x_column_name='rel_time', curv_of_traj_trace_name=self.curv_of_traj_trace_name,
                                                                                         show_visible_segments=self.current_plotly_key_comp[
                                                                                             'show_visible_segments'],
                                                                                         visible_segments_info=self.visible_segments_info, trajectory_next_stop_row=self.trajectory_next_stop_row)

        self.fig_time_series_cm = plotly_for_time_series.make_the_initial_fig_time_series(self.curv_of_traj_in_duration, self.monkey_hoverdata_value_cm, self.cur_ff_color, self.nxt_ff_color, trajectory_ref_row=self.trajectory_ref_row,
                                                                                          use_two_y_axes=use_two_y_axes, x_column_name='rel_distance', curv_of_traj_trace_name=self.curv_of_traj_trace_name,
                                                                                          show_visible_segments=self.current_plotly_key_comp[
                                                                                              'show_visible_segments'],
                                                                                          visible_segments_info=self.visible_segments_info, trajectory_next_stop_row=self.trajectory_next_stop_row)

        # also add y=-150 and y=-100 (for more reference points)
        y_range_for_v_line = [-200, 200]
        self.fig_time_series_cm = plotly_for_time_series.add_vertical_line_for_an_x_value(
            self.fig_time_series_cm, x_value=-150, y_range=y_range_for_v_line, color=plotly_for_time_series.REF_POINT_COLOR, name='y=-150', dash=None)
        self.fig_time_series_cm = plotly_for_time_series.add_vertical_line_for_an_x_value(
            self.fig_time_series_cm, x_value=-100, y_range=y_range_for_v_line, color=plotly_for_time_series.REF_POINT_COLOR, name='y=-100', dash=None)

        # self._turn_on_or_off_vertical_lines_in_each_time_series_plot_based_on_monkey_hoverdata_column()
        return self.fig_time_series_s, self.fig_time_series_cm

    def _turn_on_or_off_vertical_lines_in_each_time_series_plot_based_on_monkey_hoverdata_column(self):
        if self.hoverdata_column == 'rel_time':
            for name in ['Monkey trajectory hover position']:
                self.fig_time_series_cm.update_traces(
                    visible=False, selector=dict(name=name))
                self.fig_time_series_s.update_traces(
                    visible=True, selector=dict(name=name))
        else:
            for name in ['Monkey trajectory hover position']:
                self.fig_time_series_cm.update_traces(
                    visible=True, selector=dict(name=name))
                self.fig_time_series_s.update_traces(
                    visible=False, selector=dict(name=name))

    def _show_nxt_ff_curv_in_time_series_plot_func(self, y_column_name='opt_arc_curv'):
        if self.curv_of_traj_df is None:
            raise ValueError(
                'curv_of_traj_df is None, so cannot show nxt_ff_curv')
        self.nxt_ff_curv_df = dash_utils._find_nxt_ff_curv_df(
            self.current_plotly_key_comp, self.ff_dataframe, self.monkey_information, curv_of_traj_df=self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new)
        self.fig_time_series_s = plotly_for_time_series.add_to_time_series_plot(
            self.fig_time_series_s, self.nxt_ff_curv_df, name='Nxt FF Opt Curv', color=self.nxt_ff_color, x_column_name='rel_time', y_column_name=y_column_name, symbol='triangle-down')
        self.fig_time_series_cm = plotly_for_time_series.add_to_time_series_plot(
            self.fig_time_series_cm, self.nxt_ff_curv_df, name='Nxt FF Opt Curv', color=self.nxt_ff_color, x_column_name='rel_distance', y_column_name=y_column_name, symbol='triangle-down')

    def _show_cur_ff_curv_in_time_series_plot_func(self, y_column_name='opt_arc_curv'):
        self.cur_ff_curv_df = dash_utils._find_cur_ff_curv_df(
            self.current_plotly_key_comp, self.ff_dataframe, self.monkey_information, curv_of_traj_df=self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new)
        self.fig_time_series_s = plotly_for_time_series.add_to_time_series_plot(
            self.fig_time_series_s, self.cur_ff_curv_df, name='Cur FF Opt Curv', color=self.cur_ff_color, x_column_name='rel_time', y_column_name=y_column_name, symbol='triangle-up')
        self.fig_time_series_cm = plotly_for_time_series.add_to_time_series_plot(
            self.fig_time_series_cm, self.cur_ff_curv_df, name='Cur FF Opt Curv', color=self.cur_ff_color, x_column_name='rel_distance', y_column_name=y_column_name, symbol='triangle-up')

    def _update_null_arcs_for_cur_and_nxt_ff_in_plotly(self):
        self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_duration()
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
        self.fig = plotly_for_null_arcs.update_null_arcs_in_plotly(
            self.fig, self.nxt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix, trace_name='nxt null arc')
        self.fig = plotly_for_null_arcs.update_null_arcs_in_plotly(
            self.fig, self.cur_null_arc_info_for_the_point, rotation_matrix=rotation_matrix, trace_name='cur null arc')
        return self.fig

    def _update_extended_traj_arc_in_plotly(self):
        self._get_ext_traj_arc_info_for_the_point()
        self.fig = plotly_for_null_arcs.update_null_arcs_in_plotly(
            self.fig, self.ext_traj_arc_info_for_the_point, rotation_matrix=self.current_plotly_key_comp['rotation_matrix'], trace_name='extended traj arc')
        return self.fig

    def _get_ext_traj_arc_info_for_the_point(self):
        if not hasattr(self, 'ext_traj_arc_info'):
            df_in_opt_arc_format, self.ext_traj_arc_info = opt_arc_utils.extend_arc_from_curv_of_traj(
                self.curv_of_traj_df, self.monkey_information)
        if not hasattr(self, 'ext_traj_arc_info_for_duration'):
            self._find_ext_traj_arc_info_for_duration()
        self.ext_traj_arc_info_for_the_point = self.ext_traj_arc_info_for_duration[
            self.ext_traj_arc_info_for_duration['arc_point_index'] == self.point_index_to_show_traj_curv]

    def _find_ext_traj_arc_info_for_duration(self):
        if not hasattr(self, 'ext_traj_arc_info'):
            df_in_opt_arc_format, self.ext_traj_arc_info = opt_arc_utils.extend_arc_from_curv_of_traj(
                self.curv_of_traj_df, self.monkey_information)
        self.ext_traj_arc_info_for_duration = self.ext_traj_arc_info[self.ext_traj_arc_info['time'].between(
            self.current_plotly_key_comp['duration_to_plot'][0], self.current_plotly_key_comp['duration_to_plot'][1])]

    def _find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_duration(self):
        self.cur_null_arc_info_for_the_point = self.cur_null_arc_info_for_duration[
            self.cur_null_arc_info_for_duration['arc_point_index'] == self.point_index_to_show_traj_curv]
        self.nxt_null_arc_info_for_the_point = self.nxt_null_arc_info_for_duration[
            self.nxt_null_arc_info_for_duration['arc_point_index'] == self.point_index_to_show_traj_curv]

    def _update_all_monkey_heading_in_fig(self):
        self.fig = monkey_heading_utils.update_monkey_heading_in_monkey_plot(
            self.fig, self.traj_triangle_df_in_duration, trace_name_prefix='monkey heading on trajectory', point_index=self.point_index_to_show_traj_curv)
        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = monkey_heading_utils.update_monkey_heading_in_monkey_plot(
                self.fig, self.cur_ff_triangle_df_in_duration, trace_name_prefix='monkey heading for cur ff', point_index=self.point_index_to_show_traj_curv)
            self.fig = monkey_heading_utils.update_monkey_heading_in_monkey_plot(
                self.fig, self.nxt_ff_triangle_df_in_duration, trace_name_prefix='monkey heading for nxt ff', point_index=self.point_index_to_show_traj_curv)
        return self.fig

    def _rerun_after_changing_curv_of_traj_params(self):
        self.curv_of_traj_lower_end = self.curv_of_traj_params['window_for_curv_of_traj'][0]
        self.curv_of_traj_upper_end = self.curv_of_traj_params['window_for_curv_of_traj'][1]

        self.get_curv_of_traj_df(window_for_curv_of_traj=self.curv_of_traj_params['window_for_curv_of_traj'], curv_of_traj_mode=self.curv_of_traj_params[
                                 'curv_of_traj_mode'], truncate_curv_of_traj_by_time_of_capture=self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'])
        self._deal_with_rows_with_big_ff_angles(remove_i_o_modify_rows_with_big_ff_angles=self.overall_params['remove_i_o_modify_rows_with_big_ff_angles'],
                                                delete_the_same_rows=True)
        self._add_curvature_info()
        self._add_d_heading_info()
        self._take_out_info_counted()
        self._find_curv_of_traj_counted()
        self.find_relative_curvature()
        if self.overall_params['eliminate_outliers']:
            self._eliminate_outliers_in_cur_ff_curv()
        # self._find_relative_heading_info()
        self.cur_and_nxt_ff_from_ref_df = self._make_cur_and_nxt_ff_from_ref_df()
        self.heading_info_df, self.diff_in_curv_df = self.retrieve_or_make_heading_info_df(
            self.test_or_control)

    def _rerun_after_changing_eliminate_outliers(self):
        if self.overall_params['eliminate_outliers']:
            self._eliminate_outliers_in_cur_ff_curv()
        # self._find_relative_heading_info()

    def _prepare_to_plot_eye_positions_for_dash(self):

        trajectory_df = self.current_plotly_key_comp['trajectory_df'].copy(
        )
        duration = self.current_plotly_key_comp['duration_to_plot']
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']

        # prepare for both-eye cases and non-both-eye cases
        self.both_eyes_info = {}
        for left_or_right, suffix, marker in [('left', '_l', 'o'), ('right', '_r', 's')]:
            monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
                trajectory_df, duration, rotation_matrix=rotation_matrix, eye_col_suffix=suffix
            )
            monkey_subset = plotly_for_monkey._merge_monkey_subset_with_trajectory_df(
                monkey_subset, trajectory_df)
            monkey_subset.set_index('point_index', inplace=True)
            monkey_subset['point_index'] = monkey_subset.index
            self.both_eyes_info[left_or_right] = monkey_subset

        # for non-both-eye cases
        self.avg_eye_info = eye_positions.find_eye_positions_rotated_in_world_coordinates(
            trajectory_df, duration, rotation_matrix=rotation_matrix)
        # use merge, but first make sure no duplicate column
        self.avg_eye_info = plotly_for_monkey._merge_monkey_subset_with_trajectory_df(
            self.avg_eye_info, trajectory_df)
        self.avg_eye_info.set_index('point_index', inplace=True)

    def _produce_fig_for_dash(self, mark_reference_point=True):

        self.monkey_plot_params.update({'show_reward_boundary': True,
                                        'show_traj_points_when_making_lines': True,
                                        'eye_positions_trace_name': 'all_eye_positions',
                                        'hoverdata_multi_columns': self.hoverdata_multi_columns,
                                        })

        self._update_show_stop_point_indices(
            self.current_plotly_key_comp['trajectory_df'])

        self._prepare_to_plot_eye_positions_for_dash()

        self.fig = plotly_plot_class.PlotlyPlotter.make_one_monkey_plotly_plot(self,
                                                                               monkey_plot_params=self.monkey_plot_params)

        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper._get_all_triangle_df_for_the_point_from_triangle_df_in_duration(
                self)
            self.fig = plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_plotly(
                self)

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_duration()
            self.fig = plotly_plot_class.PlotlyPlotter._show_null_arcs_for_cur_and_nxt_ff_in_plotly(
                self)

        if self.monkey_plot_params['show_extended_traj_arc']:
            self._plot_extended_traj_arc()

        if mark_reference_point:
            self.fig = plotly_for_monkey.mark_reference_point_in_monkey_plot(
                self.fig, self.trajectory_ref_row)

        self.traj_portion_trace_index = -1
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'trajectory_data':
                self.trajectory_data_trace_index = i
            if trace.name == 'to_show_traj_portion':
                self.traj_portion_trace_index = i

        return self.fig

    def _plot_extended_traj_arc(self):
        self._get_ext_traj_arc_info_for_the_point()
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.ext_traj_arc_info_for_the_point, rotation_matrix=self.current_plotly_key_comp['rotation_matrix'],
                                                                 color=self.traj_arc_color, trace_name='extended traj arc', linewidth=2)

    def _update_show_stop_point_indices(self, trajectory_df):
        """
        Update stop indices to visualize:
        - 'show_stop_point_indices': all stop points (speed==0) OR just the capture pair when that's the only thing shown
        - 'show_capture_stop_point_indices': the specific two indices around the fast–forward (capture) event

        Expects:
        - trajectory_df has columns ['monkey_speeddummy', 'point_index']
        - self.stops_near_ff_row has attributes stop_point_index, next_stop_point_index (when used)
        """
        params = self.monkey_plot_params
        show_stops = bool(params.get('show_stops'))
        show_capture = bool(params.get('show_cur_and_nxt_stops'))

        # Defaults
        stop_indices = None
        capture_indices = None

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

        # Specific capture pair
        if show_capture:
            capture_indices = _capture_pair()

        # Write back
        params['show_stop_point_indices'] = stop_indices
        params['show_capture_stop_point_indices'] = capture_indices

# def _update_eye_positions_based_on_monkey_hoverdata(fig, point_index_to_show_traj_curv, current_plotly_key_comp, show_eye_positions_for_both_eyes=False):
#     current_plotly_key_comp_2 = copy.deepcopy(current_plotly_key_comp)
#     trajectory_df = current_plotly_key_comp_2['trajectory_df']
#     trajectory_df = trajectory_df[trajectory_df['point_index'] == point_index_to_show_traj_curv]
#     current_plotly_key_comp_2['trajectory_df'] = trajectory_df
#     fig = plotly_for_monkey.plot_eye_positions_in_plotly(fig, current_plotly_key_comp_2, show_eye_positions_for_both_eyes=show_eye_positions_for_both_eyes,
#                                                         use_arrow_to_show_eye_positions=True, marker_size=15)
#     return fig
