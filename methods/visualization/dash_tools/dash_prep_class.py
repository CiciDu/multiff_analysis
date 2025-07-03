
import sys
from visualization.plotly_tools import plotly_preparation, plotly_for_scatterplot, plotly_for_null_arcs, plotly_for_correlation, plotly_for_monkey, plotly_plot_class
from visualization.dash_tools import dash_utils, dash_utils
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, stops_near_ff_based_on_ref_class
from visualization.matplotlib_tools import monkey_heading_functions
from eye_position_analysis import eye_positions
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_class, plot_stops_near_ff_utils, plot_monkey_heading_helper_class


import os
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


# https://dash.plotly.com/interactive-graphing


class DashCartesianPreparation(stops_near_ff_based_on_ref_class.StopsNearFFBasedOnRef, plotly_plot_class.PlotlyPlotter):

    def __init__(self,
                 raw_data_folder_path=None):

        super().__init__(raw_data_folder_path=raw_data_folder_path)

    def _prepare_static_main_plots(self,
                                   show_static_plots=False,
                                   modify_non_negative_ver_theta=True):

        self.stop_point_index = self.stops_near_ff_row.stop_point_index
        self.hoverdata_column = self.ref_point_column
        self.hoverdata_multi_columns = dash_utils.find_hoverdata_multi_columns(
            self.hoverdata_column)
        self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(
            self.stop_point_index)

        self.fig, self.fig_scatter_combd, self.fig_scatter_natural_y_range = self._produce_initial_plots()

        if show_static_plots:
            dash_utils.show_a_static_plot(self.fig_scatter_s)

        if modify_non_negative_ver_theta:
            self.monkey_information = self.monkey_information.copy()
            self.monkey_information.loc[self.monkey_information['LDz']
                                        >= 0, 'LDz'] = -0.001
            self.monkey_information.loc[self.monkey_information['RDz']
                                        >= 0, 'RDz'] = -0.001
            self.monkey_information = eye_positions.convert_eye_positions_in_monkey_information(self.monkey_information, add_left_and_right_eyes_info=True,
                                                                                                interocular_dist=self.interocular_dist)

        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        self.fig_corr_or_heading, self.fig_corr_or_heading_2 = self._determine_fig_corr_or_heading()

    def _find_trajectory_ref_row(self):
        ref_point_index = self.nxt_ff_df2[self.nxt_ff_df2['stop_point_index']
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

    def _update_fig_scatter_combd_y_range(self):
        margin = 10
        if self.use_two_y_axes:
            self.fig_scatter_combd.update_layout(yaxis=dict(range=[self.fig_scatter_natural_y_range[0]-margin, self.fig_scatter_natural_y_range[1]+margin]),
                                                 yaxis3=dict(range=[self.fig_scatter_natural_y_range[0]-margin, self.fig_scatter_natural_y_range[1]+margin]))
        else:
            self.fig_scatter_combd.update_layout(yaxis=dict(range=[self.fig_scatter_natural_y_range[0]-margin, self.fig_scatter_natural_y_range[1]+margin]),
                                                 yaxis2=dict(range=[self.fig_scatter_natural_y_range[0]-margin, self.fig_scatter_natural_y_range[1]+margin]))

    def _produce_initial_plots(self):

        self.use_two_y_axes = self.scatter_plot_params['use_two_y_axes']
        self.fig = self._produce_fig_for_dash()
        self.fig_scatter_s, self.fig_scatter_cm = self._produce_fig_scatter(
            use_two_y_axes=self.use_two_y_axes)
        self.fig_scatter_natural_y_range = [np.min(self.curv_of_traj_df_in_duration['curv_of_traj_deg_over_cm'].values), np.max(
            self.curv_of_traj_df_in_duration['curv_of_traj_deg_over_cm'].values)]
        y_column_name = 'curv_to_ff_center' if self.overall_params[
            'use_curvature_to_ff_center'] else 'optimal_curvature'
        if self.scatter_plot_params['show_nxt_ff_curv_in_scatterplot']:
            self._show_nxt_ff_curv_in_scatterplot_func(
                y_column_name=y_column_name)
            try:
                self.fig_scatter_natural_y_range = plotly_for_scatterplot.update_fig_scatter_natural_y_range(
                    self.fig_scatter_natural_y_range, self.nxt_ff_curv_df, y_column_name)
            except:
                pass
        if self.scatter_plot_params['show_cur_ff_curv_in_scatterplot']:
            self._show_cur_ff_curv_in_scatterplot_func(
                y_column_name=y_column_name)
            try:
                self.fig_scatter_natural_y_range = plotly_for_scatterplot.update_fig_scatter_natural_y_range(
                    self.fig_scatter_natural_y_range, self.cur_ff_curv_df, y_column_name)
            except:
                pass
        self.fig_scatter_combd = plotly_for_scatterplot.make_fig_scatter_combd(
            self.fig_scatter_s, self.fig_scatter_cm, self.use_two_y_axes)
        self._update_fig_scatter_combd_y_range()
        return self.fig, self.fig_scatter_combd, self.fig_scatter_natural_y_range

    def _get_curv_of_traj_df_in_duration(self):
        self.stops_near_ff_row = self.stops_near_ff_df[self.stops_near_ff_df['stop_point_index']
                                                       == self.stop_point_index].iloc[0]
        self.curv_of_traj_df_in_duration = curv_of_traj_utils.find_curv_of_traj_df_in_duration(
            self.curv_of_traj_df, self.current_plotly_key_comp['duration_to_plot'])
        self.curv_of_traj_df_in_duration['rel_time'] = np.round(
            self.curv_of_traj_df_in_duration['time'] - self.stops_near_ff_row.stop_time, 2)
        self.curv_of_traj_df_in_duration['rel_distance'] = np.round(
            self.curv_of_traj_df_in_duration['cum_distance'] - self.stops_near_ff_row.stop_cum_distance, 2)
        if len(self.curv_of_traj_df_in_duration) == 0:
            print('Warning: curv_of_traj_df_in_duration is empty!')

        return self.curv_of_traj_df_in_duration

    def _produce_fig_scatter(self, use_two_y_axes=True):
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_scatterplot.find_monkey_hoverdata_value_for_both_fig_scatter(
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

        self.fig_scatter_s = plotly_for_scatterplot.make_the_initial_fig_scatter(self.curv_of_traj_df_in_duration, self.monkey_hoverdata_value_s, self.cur_ff_color, self.nxt_ff_color, trajectory_ref_row=self.trajectory_ref_row,
                                                                                 use_two_y_axes=use_two_y_axes, x_column_name='rel_time', curv_of_traj_trace_name=self.curv_of_traj_trace_name,
                                                                                 show_visible_segments=self.current_plotly_key_comp[
                                                                                     'show_visible_segments'],
                                                                                 visible_segments_info=self.visible_segments_info, trajectory_next_stop_row=self.trajectory_next_stop_row)
        self.fig_scatter_cm = plotly_for_scatterplot.make_the_initial_fig_scatter(self.curv_of_traj_df_in_duration, self.monkey_hoverdata_value_cm, self.cur_ff_color, self.nxt_ff_color, trajectory_ref_row=self.trajectory_ref_row,
                                                                                  use_two_y_axes=use_two_y_axes, x_column_name='rel_distance', curv_of_traj_trace_name=self.curv_of_traj_trace_name,
                                                                                  show_visible_segments=self.current_plotly_key_comp[
                                                                                      'show_visible_segments'],
                                                                                  visible_segments_info=self.visible_segments_info, trajectory_next_stop_row=self.trajectory_next_stop_row)

        # self._turn_on_or_off_vertical_lines_in_each_scatterplot_based_on_monkey_hoverdata_column()
        return self.fig_scatter_s, self.fig_scatter_cm

    def _turn_on_or_off_vertical_lines_in_each_scatterplot_based_on_monkey_hoverdata_column(self):
        if self.hoverdata_column == 'rel_time':
            # , 'First stop point']:
            for name in ['Monkey trajectory hover position']:
                self.fig_scatter_cm.update_traces(
                    visible=False, selector=dict(name=name))
                self.fig_scatter_s.update_traces(
                    visible=True, selector=dict(name=name))
        else:
            # , 'First stop point']:
            for name in ['Monkey trajectory hover position']:
                self.fig_scatter_cm.update_traces(
                    visible=True, selector=dict(name=name))
                self.fig_scatter_s.update_traces(
                    visible=False, selector=dict(name=name))

    def _show_nxt_ff_curv_in_scatterplot_func(self, y_column_name='curv_to_ff_center'):
        if self.curv_of_traj_df is None:
            raise ValueError(
                'curv_of_traj_df is None, so cannot show nxt_ff_curv')
        self.nxt_ff_curv_df = dash_utils._find_nxt_ff_curv_df(
            self.current_plotly_key_comp, self.ff_dataframe, self.monkey_information, curv_of_traj_df=self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new)
        self.fig_scatter_s = plotly_for_scatterplot.add_to_the_scatterplot(
            self.fig_scatter_s, self.nxt_ff_curv_df, name='Nxt FF Curv to Center', color=self.nxt_ff_color, x_column_name='rel_time', y_column_name=y_column_name, symbol='triangle-down')
        self.fig_scatter_cm = plotly_for_scatterplot.add_to_the_scatterplot(
            self.fig_scatter_cm, self.nxt_ff_curv_df, name='Nxt FF Curv to Center', color=self.nxt_ff_color, x_column_name='rel_distance', y_column_name=y_column_name, symbol='triangle-down')

    def _show_cur_ff_curv_in_scatterplot_func(self, y_column_name='curv_to_ff_center'):
        self.cur_ff_curv_df = dash_utils._find_cur_ff_curv_df(
            self.current_plotly_key_comp, self.ff_dataframe, self.monkey_information, curv_of_traj_df=self.curv_of_traj_df, ff_caught_T_new=self.ff_caught_T_new)
        self.fig_scatter_s = plotly_for_scatterplot.add_to_the_scatterplot(
            self.fig_scatter_s, self.cur_ff_curv_df, name='cur ff Curv to Center', color=self.cur_ff_color, x_column_name='rel_time', y_column_name=y_column_name, symbol='triangle-up')
        self.fig_scatter_cm = plotly_for_scatterplot.add_to_the_scatterplot(
            self.fig_scatter_cm, self.cur_ff_curv_df, name='cur ff Curv to Center', color=self.cur_ff_color, x_column_name='rel_distance', y_column_name=y_column_name, symbol='triangle-up')

    def _update_null_arcs_for_cur_and_nxt_ff_in_plotly(self):
        self._find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_duration()
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']
        self.fig = plotly_for_null_arcs.update_null_arcs_in_plotly(
            self.fig, self.nxt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix, trace_name='nxt null arc')
        self.fig = plotly_for_null_arcs.update_null_arcs_in_plotly(
            self.fig, self.cur_null_arc_info_for_the_point, rotation_matrix=rotation_matrix, trace_name='cur null arc')
        return self.fig

    def _find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_duration(self):
        self.cur_null_arc_info_for_the_point = self.cur_null_arc_info_for_duration[
            self.cur_null_arc_info_for_duration['arc_point_index'] == self.point_index_to_show_traj_curv]
        self.nxt_null_arc_info_for_the_point = self.nxt_null_arc_info_for_duration[
            self.nxt_null_arc_info_for_duration['arc_point_index'] == self.point_index_to_show_traj_curv]

    def _make_fig_corr_2(self):

        self.fig_corr_2 = go.Figure()
        if 'heading_instead_of_curv' in self.overall_params:
            if not self.overall_params['heading_instead_of_curv']:
                # shuffle the rows for the two columns 'curv_to_ff_center', 'optimal_curvature' in nxt_ff_counted_df
                nxt_ff_counted_df = self.nxt_ff_counted_df.copy()
                nxt_ff_counted_df[['curv_to_ff_center', 'optimal_curvature']] = nxt_ff_counted_df[[
                    'curv_to_ff_center', 'optimal_curvature']].sample(frac=1).values

                traj_curv_counted, nxt_curv_counted = find_stops_near_ff_utils.find_relative_curvature(
                    nxt_ff_counted_df, self.cur_ff_counted_df, self.curv_of_traj_counted, use_curvature_to_ff_center=self.overall_params['use_curvature_to_ff_center'])

                self.kwargs_for_correlation_plot_2 = copy.deepcopy(
                    self.kwargs_for_correlation_plot)
                self.kwargs_for_correlation_plot_2['curv_for_correlation_df']['traj_curv_counted'] = traj_curv_counted.copy(
                )
                self.kwargs_for_correlation_plot_2['curv_for_correlation_df']['nxt_curv_counted'] = nxt_curv_counted.copy(
                )
                self.kwargs_for_correlation_plot_2['current_stop_point_index_to_mark'] = None

                self.fig_corr_2 = plotly_for_correlation.make_correlation_plot_in_plotly(
                    **self.kwargs_for_correlation_plot_2, title="After Shuffling Nxt FF Curvature")

                # # update title
                # self.fig_corr_2.update_layout(title="After Shuffling Nxt FF Curvature")

        return self.fig_corr_2

    def _make_fig_heading_2(self):

        return self.fig_heading

        # # randomly shuffle the elements in d_heading_cur
        # d_heading_nxt = self.nxt_ff_final_df['d_heading_of_arc'].values.copy()
        # np.random.shuffle(d_heading_nxt)

        # rel_heading_traj = self.d_heading_of_traj - self.cur_ff_final_df['d_heading_of_arc'].values
        # rel_heading_alt = d_heading_nxt - self.cur_ff_final_df['d_heading_of_arc'].values
        # rel_heading_traj = find_stops_near_ff_utils.confine_angle_to_within_one_pie(rel_heading_traj)
        # rel_heading_alt = find_stops_near_ff_utils.confine_angle_to_within_one_pie(rel_heading_alt)

        # self.kwargs_for_heading_plot_2 = copy.deepcopy(self.kwargs_for_heading_plot)
        # self.kwargs_for_heading_plot_2['rel_heading_df']['rel_heading_traj'] = rel_heading_traj
        # self.kwargs_for_heading_plot_2['rel_heading_df']['rel_heading_alt'] = rel_heading_alt
        # self.kwargs_for_heading_plot_2['current_stop_point_index_to_mark'] = None

        # self.fig_heading_2 = plotly_for_correlation.make_heading_plot_in_plotly(**self.kwargs_for_heading_plot_2, title="After Shuffling Nxt FF Curvature")
        # # self.fig_heading_2.update_layout(title="After Shuffling Nxt FF Curvature")

        # return self.fig_heading_2

    def _make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled(self):
        self.fig_corr = go.Figure()
        self.fig_corr_2 = go.Figure()
        if 'heading_instead_of_curv' in self.overall_params:
            if not self.overall_params['heading_instead_of_curv']:
                self.kwargs_for_correlation_plot = self._make_kwargs_for_correlation_plot()
                self.kwargs_for_correlation_plot['current_stop_point_index_to_mark'] = self.stop_point_index
                self.fig_corr = plotly_for_correlation.make_correlation_plot_in_plotly(
                    **self.kwargs_for_correlation_plot)
                self.fig_corr_2 = self._make_fig_corr_2()

        self.kwargs_for_heading_plot = self._make_kwargs_for_heading_plot()
        self.kwargs_for_heading_plot['current_stop_point_index_to_mark'] = self.stop_point_index
        self.fig_heading = plotly_for_correlation.make_heading_plot_in_plotly(
            **self.kwargs_for_heading_plot)
        self.fig_heading_2 = self._make_fig_heading_2()

        return self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2

    def _update_all_monkey_heading_in_fig(self):
        self.fig = monkey_heading_functions.update_monkey_heading_in_monkey_plot(
            self.fig, self.traj_triangle_df_in_duration, trace_name_prefix='monkey heading on trajectory', point_index=self.point_index_to_show_traj_curv)
        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = monkey_heading_functions.update_monkey_heading_in_monkey_plot(
                self.fig, self.cur_ff_triangle_df_in_duration, trace_name_prefix='monkey heading for cur ff', point_index=self.point_index_to_show_traj_curv)
            self.fig = monkey_heading_functions.update_monkey_heading_in_monkey_plot(
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
        self.cur_and_nxt_ff_df = self._make_cur_and_nxt_ff_df()
        self.heading_info_df, self.diff_in_curv_df = self._make_heading_info_df(
            self.test_or_control)
        self.kwargs_for_heading_plot = self._make_kwargs_for_heading_plot()

    def _rerun_after_changing_eliminate_outliers(self):
        if self.overall_params['eliminate_outliers']:
            self._eliminate_outliers_in_cur_ff_curv()
        # self._find_relative_heading_info()
        self.kwargs_for_heading_plot = self._make_kwargs_for_heading_plot()

    def _prepare_to_plot_eye_positions_for_dash(self):

        trajectory_df = self.current_plotly_key_comp['trajectory_df'].copy(
        )
        duration = self.current_plotly_key_comp['duration_to_plot']
        rotation_matrix = self.current_plotly_key_comp['rotation_matrix']

        # prepare for both-eye cases and non-both-eye cases
        self.both_eyes_info = {}
        for left_or_right, suffix, marker in [('left', '_r', 'o'), ('right', '_r', 's')]:
            monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
                trajectory_df, duration, rotation_matrix=rotation_matrix, eye_col_suffix=suffix
            )
            columns_to_merge = [col for col in [
                'rel_time', 'monkey_x', 'monkey_y'] if col not in monkey_subset.columns]
            monkey_subset = monkey_subset.merge(trajectory_df[[
                'point_index'] + columns_to_merge], on='point_index', how='left')
            monkey_subset.set_index('point_index', inplace=True)
            monkey_subset['point_index'] = monkey_subset.index
            self.both_eyes_info[left_or_right] = monkey_subset

        # for non-both-eye cases
        self.monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
            trajectory_df, duration, rotation_matrix=rotation_matrix)
        # use merge, but first make sure no duplicate column
        columns_to_merge = [col for col in [
            'rel_time', 'monkey_x', 'monkey_y'] if col not in self.monkey_subset.columns]
        self.monkey_subset = self.monkey_subset.merge(trajectory_df[[
                                                      'point_index'] + columns_to_merge], on='point_index', how='left')
        self.monkey_subset.index = self.monkey_subset['point_index'].values

    def _produce_fig_for_dash(self, mark_reference_point=True):

        self.monkey_plot_params.update({'show_reward_boundary': True,
                                        'show_traj_points_when_making_lines': True,
                                        'eye_positions_trace_name': 'all_eye_positions',
                                        'hoverdata_multi_columns': self.hoverdata_multi_columns,
                                        })

        self.monkey_plot_params['traj_portion'] = self.traj_portion if self.monkey_plot_params['show_traj_portion'] else None

        self._update_show_stop_point_indices()

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

        if mark_reference_point:
            self.fig = plotly_for_monkey.mark_reference_point_in_monkey_plot(
                self.fig, self.trajectory_ref_row)

        for i, trace in enumerate(self.fig.data):
            if trace.name == 'trajectory_data':
                self.trajectory_data_trace_index = i
                break

        self.traj_portion_trace_index = -1
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'to_show_the_scope_for_curv':
                self.traj_portion_trace_index = i
                break

        return self.fig

    def _update_show_stop_point_indices(self):
        if self.monkey_plot_params.get('show_stops', False):
            self.monkey_plot_params['show_stop_point_indices'] = [
                int(self.stops_near_ff_row.stop_point_index),
                int(self.stops_near_ff_row.next_stop_point_index)
            ]
        # else:
        #     self.monkey_plot_params['show_stop_point_indices'] = None

# def _update_eye_positions_based_on_monkey_hoverdata(fig, point_index_to_show_traj_curv, current_plotly_key_comp, show_eye_positions_for_both_eyes=False):
#     current_plotly_key_comp_2 = copy.deepcopy(current_plotly_key_comp)
#     trajectory_df = current_plotly_key_comp_2['trajectory_df']
#     trajectory_df = trajectory_df[trajectory_df['point_index'] == point_index_to_show_traj_curv]
#     current_plotly_key_comp_2['trajectory_df'] = trajectory_df
#     fig = plotly_for_monkey.plot_eye_positions_in_plotly(fig, current_plotly_key_comp_2, show_eye_positions_for_both_eyes=show_eye_positions_for_both_eyes,
#                                                         use_arrow_to_show_eye_positions=True, marker_size=15)
#     return fig
