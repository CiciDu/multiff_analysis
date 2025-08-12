import sys
from visualization.plotly_tools import plotly_for_correlation, plotly_preparation, plotly_for_time_series
from visualization.dash_tools import dash_prep_class, dash_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, plot_monkey_heading_helper_class
from visualization.matplotlib_tools import monkey_heading_utils
from visualization.plotly_tools import plotly_for_monkey
from null_behaviors import show_null_trajectory
from visualization.plotly_tools import plotly_for_null_arcs
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from planning_analysis.plan_factors import build_factor_comp

import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import html
from dash.exceptions import PreventUpdate
import pandas as pd
from dash import dcc
import plotly.graph_objects as go
import copy
import logging


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class DashMainHelper(dash_prep_class.DashCartesianPreparation):

    def __init__(self,
                 raw_data_folder_path=None,
                 opt_arc_type='opt_arc_stop_closest'):

        super().__init__(raw_data_folder_path=raw_data_folder_path, opt_arc_type=opt_arc_type)

    def prepare_to_make_dash_for_main_plots(self,
                                            ref_point_params={},
                                            curv_of_traj_params={},
                                            overall_params={},
                                            monkey_plot_params={},
                                            time_series_plot_params={},
                                            stops_near_ff_df_exists_ok=True,
                                            heading_info_df_exists_ok=True,
                                            test_or_control='test',
                                            stop_point_index=None):

        self.ref_point_params = ref_point_params
        self.curv_of_traj_params = curv_of_traj_params
        self.overall_params = {
            **copy.deepcopy(self.default_overall_params),
            **overall_params
        }
        self.monkey_plot_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **monkey_plot_params
        }
        self.time_series_plot_params = time_series_plot_params

        self.snf_streamline_organizing_info_kwargs = find_cvn_utils.organize_snf_streamline_organizing_info_kwargs(
            ref_point_params, curv_of_traj_params, overall_params)
        super().streamline_organizing_info(**self.snf_streamline_organizing_info_kwargs, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                           heading_info_df_exists_ok=heading_info_df_exists_ok, test_or_control=test_or_control)

        self._get_stops_near_ff_row(stop_point_index)

        self._prepare_static_main_plots()

    def _get_stops_near_ff_row(self, stop_point_index):
        if stop_point_index is None:
            # we use the first instance in stops_near_ff_df_counted to plot for now.
            self.stops_near_ff_row = self.stops_near_ff_df_counted.iloc[0]
        else:
            try:
                self.stops_near_ff_row = self.stops_near_ff_df_counted[self.stops_near_ff_df_counted['stop_point_index']
                                                                       == stop_point_index].iloc[0]
                logging.info(
                    f'Successfully retrieved stop_point_index: {stop_point_index}.')
            except:
                logging.warning(
                    f'stop_point_index: {stop_point_index} is not in stops_near_ff_df_counted. Using the first instance in stops_near_ff_df_counted to plot.')
        self.stop_point_index = self.stops_near_ff_row['stop_point_index']


    def _put_down_trajectory_time_series_plot(self, layout, id_prefix=''):
        # Add time series plot conditionally
        if self.show_trajectory_time_series:
            layout.append(dash_utils.put_down_time_series_plot(
                self.fig_time_series_combd, id=id_prefix+'time_series_plot_combined'
            ))
        else:
            layout.append(dash_utils.put_down_empty_plot_that_takes_no_space(
                id=id_prefix+'time_series_plot_combined'
            ))
            
        return layout
    
        
    def _put_down_neural_plots(self, layout, id_prefix=''):
        # Add neural plots conditionally
        if self.show_neural_plots:
            try:
                self.fig_raster = self._create_raster_plot_figure()
                self.fig_fr = self._create_firing_rate_plot_figure()
            except Exception as e:
                logging.warning(
                    f"Could not create neural plots: {e}. Using empty figures.")
                self.fig_raster = self._get_empty_figure()
                self.fig_fr = self._get_empty_figure()

            neural_plot_layout = [
                dash_utils.put_down_raster_plot(
                    self.fig_raster, id=id_prefix+'raster_plot'
                ),
                dash_utils.put_down_firing_rate_plot(
                    self.fig_fr, id=id_prefix+'firing_rate_plot'
                )
            ]
        else:
            # Initialize empty figures for neural plots even when not showing them
            self.fig_raster = self._get_empty_figure()
            self.fig_fr = self._get_empty_figure()

            neural_plot_layout = [
                dash_utils.put_down_empty_plot_that_takes_no_space(
                    id=id_prefix+'raster_plot'
                ),
                dash_utils.put_down_empty_plot_that_takes_no_space(
                    id=id_prefix+'firing_rate_plot'
                )
            ]
        neural_plots_in_dash = html.Div(
            neural_plot_layout, style=dict(display='flex'))
        layout.append(neural_plots_in_dash)
        return layout

    def _put_down_checklist_for_all_plots(self, id_prefix=None):
        checklist_options = [{'label': 'use curvature to ff center', 'value': 'use_curv_to_ff_center'},
                             {'label': 'truncate curv of traj by time of capture',
                              'value': 'truncate_curv_of_traj_by_time_of_capture'},
                             {'label': 'eliminate outliers', 'value': 'eliminate_outliers'}]
        checklist_params = ['use_curv_to_ff_center',
                            'truncate_curv_of_traj_by_time_of_capture', 'eliminate_outliers']
        self.overall_params['truncate_curv_of_traj_by_time_of_capture'] = self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture']
        checklist_values = [
            key for key in checklist_params if self.overall_params[key] is True]

        return html.Div([dcc.Checklist(options=checklist_options,
                                       value=checklist_values,
                                       id=id_prefix+'checklist_for_all_plots',
                                       style={'width': '50%', 'background-color': '#F9F99A', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'})])

    def _put_down_checklist_for_monkey_plot(self, id_prefix=None):
        checklist_options = [{'label': 'show visible fireflies', 'value': 'show_visible_fireflies'},
                             {'label': 'show fireflies in memory',
                                 'value': 'show_in_memory_fireflies'},
                             {'label': 'show visible segments of ff',
                                 'value': 'show_visible_segments'},
                             {'label': 'show monkey heading',
                                 'value': 'show_monkey_heading'},
                             {'label': 'show portion used to calc traj curvature',
                                 'value': 'show_traj_portion'},
                             {'label': 'show null trajectory of ff',
                                 'value': 'show_null_arcs_to_ff'},
                             {'label': 'show extended traj arc',
                                 'value': 'show_extended_traj_arc'},
                             {'label': 'show stops', 'value': 'show_stops'},
                             {'label': 'show all eye positions',
                                 'value': 'show_all_eye_positions'},
                             {'label': 'show current eye positions',
                                 'value': 'show_current_eye_positions'},
                             {'label': 'show eye positions for both eyes', 'value': 'show_eye_positions_for_both_eyes'}]

        checklist_params = ['show_visible_fireflies', 
                            'show_in_memory_fireflies',
                            'show_visible_segments',
                            'show_monkey_heading', 
                            'show_traj_portion',
                            'show_null_arcs_to_ff', 
                            'show_extended_traj_arc', 
                            'show_stops', 
                            'show_all_eye_positions',
                            'show_current_eye_positions', 
                            'show_eye_positions_for_both_eyes']
        checklist_values = [
            key for key in checklist_params if self.monkey_plot_params[key] is True]

        return html.Div([dcc.Checklist(options=checklist_options,
                                       value=checklist_values,
                                       id=id_prefix+'checklist_for_monkey_plot',
                                       style={'width': '50%', 'background-color': '#ADD8E6', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'})])

    def _put_down_the_menus_on_top(self, id_prefix=''):
        return html.Div([
                        dash_utils.put_down_the_dropdown_menu_for_ref_point_mode(
                            self.ref_point_params['ref_point_mode'], id=id_prefix+'ref_point_mode'),
                        dash_utils.put_down_the_input_for_ref_point_descr(
                            self.ref_point_params['ref_point_value'], id=id_prefix+"ref_point_value"),
                        dash_utils.put_down_the_dropdown_menu_for_curv_of_traj_mode(
                            self.curv_of_traj_params['curv_of_traj_mode'], id=id_prefix+'curv_of_traj_mode'),
                        dash_utils.put_down_the_input_for_window_lower_end_and_upper_end(self.curv_of_traj_params['window_for_curv_of_traj'], ids=[
                                                                                         id_prefix+'window_lower_end', id_prefix+'window_upper_end']),
                        ],
                        style=dict(display='flex'))

    def _put_down_correlation_plots_in_dash(self, id_prefix=''):
        self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._make_two_fig_scatter_or_reg()

        plot_layout = [plotly_for_correlation.put_down_correlation_plot(
            self.fig_scatter_or_reg, id=id_prefix+'correlation_plot', width='50%')]

        plot_layout.append(plotly_for_correlation.put_down_correlation_plot(
            self.fig_scatter_or_reg2, id=id_prefix+'correlation_plot_2', width='50%'))

        correlation_plots_in_dash = html.Div(
            plot_layout, style=dict(display='flex'))
        return correlation_plots_in_dash

    def _plot_eye_positions_for_dash(self, fig, show_eye_positions_for_both_eyes=False, point_index_to_show_traj_curv=None, x0=0, y0=0,
                                     trace_name='eye_positions', update_if_already_exist=True, marker_size=6, use_arrow_to_show_eye_positions=True):

        if use_arrow_to_show_eye_positions:
            # clear existing annotations first
            fig['layout']['annotations'] = []

        if not show_eye_positions_for_both_eyes:
            monkey_subset2 = self.avg_eye_info.copy()
            if point_index_to_show_traj_curv is not None:
                try:
                    monkey_subset2 = self.avg_eye_info.loc[[
                        point_index_to_show_traj_curv]].copy()
                except KeyError:
                    print(
                        'No eye positions are updated because point_index_to_show_traj_curv is not in monkey_subset.point_index.')
                    return fig
            fig = plotly_for_monkey.plot_or_update_eye_positions_using_either_marker_or_arrow(fig, x0, y0, monkey_subset2, trace_name=trace_name, update_if_already_exist=update_if_already_exist,
                                                                                              marker='circle', marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions,
                                                                                              arrowcolor='orange')
        else:
            for left_or_right, marker, trace_name, arrowcolor in [('left', 'triangle-left', trace_name + '_left', 'orange'), ('right', 'triangle-right', trace_name + '_right', 'purple')]:
                monkey_subset = self.both_eyes_info[left_or_right]
                if point_index_to_show_traj_curv is not None:
                    try:
                        monkey_subset2 = monkey_subset.loc[[
                            point_index_to_show_traj_curv]].copy()
                    except KeyError:
                        logging.warning(
                            f'No eye positions are updated for {left_or_right} eye because point_index_to_show_traj_curv is not in monkey_subset.point_index.')
                        return fig
                fig = plotly_for_monkey.plot_or_update_eye_positions_using_either_marker_or_arrow(fig, x0, y0, monkey_subset2, trace_name=trace_name, update_if_already_exist=update_if_already_exist,
                                                                                                  marker=marker, marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions, arrowcolor=arrowcolor)
        return fig

    def _update_dash_based_on_checklist_for_monkey_plot(self, checklist_for_monkey_plot):
        print('checklist_for_monkey_plot: ', checklist_for_monkey_plot)

        old_checklist_params = {'show_visible_segments': self.monkey_plot_params['show_visible_segments'],
                                'show_visible_fireflies': self.monkey_plot_params['show_visible_fireflies'],
                                'show_in_memory_fireflies': self.monkey_plot_params['show_in_memory_fireflies'],
                                'show_traj_portion': self.monkey_plot_params['show_traj_portion'],
                                'show_null_arcs_to_ff': self.monkey_plot_params['show_null_arcs_to_ff'],
                                'show_extended_traj_arc': self.monkey_plot_params['show_extended_traj_arc'],
                                }

        for param in ['show_monkey_heading', 'show_visible_segments', 'show_traj_portion', 'show_null_arcs_to_ff',
                      'show_extended_traj_arc', 'show_stops', 'show_all_eye_positions', 'show_current_eye_positions',
                      'show_eye_positions_for_both_eyes', 'show_visible_fireflies', 'show_in_memory_fireflies']:
            if param in checklist_for_monkey_plot:
                self.monkey_plot_params[param] = True
            else:
                self.monkey_plot_params[param] = False
                
        if self.monkey_plot_params['show_traj_portion'] != old_checklist_params['show_traj_portion']:
            if self.monkey_plot_params['show_traj_portion']:
                self._find_traj_portion()
                self.fig = plotly_for_monkey.plot_a_portion_of_trajectory_to_show_traj_portion(self.fig, self.traj_portion,
                                                                                                 hoverdata_multi_columns=self.monkey_plot_params['hoverdata_multi_columns'])
            else:
                # remove the trace called to_show_traj_portion from self.fig
                self.fig.data = [trace for trace in self.fig.data if trace.name != 'to_show_traj_portion']

        if (self.monkey_plot_params['show_null_arcs_to_ff'] != old_checklist_params['show_null_arcs_to_ff']):
            self.find_null_arcs_info_for_plotting_for_the_duration()
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper._find_mheading_and_triangle_df_for_null_arcs_for_the_duration(
                self)

        if (self.monkey_plot_params['show_extended_traj_arc'] != old_checklist_params['show_extended_traj_arc']):
            self._find_ext_traj_arc_info_for_duration()

        for param in ['show_visible_segments', 'show_visible_fireflies', 'show_in_memory_fireflies']:
            if old_checklist_params[param] != self.monkey_plot_params[param]:
                self.current_plotly_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(
                    self.stops_near_ff_row, self.PlotTrials_args, self.monkey_plot_params)

                self._produce_initial_plots()
                break
        else:
            self._produce_fig_for_dash()
        # self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(self.stop_point_index)
        # self.fig, self.fig_time_series_combd, self.fig_time_series_natural_y_range = self._produce_initial_plots()

        return self.fig, self.fig_time_series_combd

    def _update_dash_based_on_checklist_for_all_plots(self, checklist_for_all_plots):
        # keep a copy of old checklist_params
        old_checklist_params = {'use_curv_to_ff_center': self.overall_params['use_curv_to_ff_center'],
                                'truncate_curv_of_traj_by_time_of_capture': self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'],
                                'eliminate_outliers': self.overall_params['eliminate_outliers']}

        # update checklist_params into the instance
        if 'truncate_curv_of_traj_by_time_of_capture' in checklist_for_all_plots:
            self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = True
        else:
            self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = False

        for param in ['eliminate_outliers', 'use_curv_to_ff_center']:
            if param in checklist_for_all_plots:
                self.overall_params[param] = True
            else:
                self.overall_params[param] = False

        # update the plots based on the new checklist_params
        if ((self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] != old_checklist_params['truncate_curv_of_traj_by_time_of_capture'])
                or (self.overall_params['use_curv_to_ff_center'] != old_checklist_params['use_curv_to_ff_center'])):
            self. _rerun_after_changing_curv_of_traj_params()
            self._prepare_static_main_plots()
        elif self.overall_params['eliminate_outliers'] != old_checklist_params['eliminate_outliers']:
            self._rerun_after_changing_eliminate_outliers()
            self._prepare_static_main_plots()
        else:
            raise PreventUpdate(
                "No update was made in calling _update_dash_based_on_checklist_for_all_plots.")

        return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2

    def _update_dash_based_on_monkey_hover_data(self, monkey_hoverdata):

        trace_index = monkey_hoverdata['points'][0]['curveNumber']
        if not ((trace_index == self.trajectory_data_trace_index) or (trace_index == self.traj_portion_trace_index)):
            # raise PreventUpdate(
            #     "No update was triggered because hover is not over trajectory.") # this seems too extra to show
            raise PreventUpdate(
                "No update was triggered because hover is not over trajectory.")

        monkey_hoverdata_value = monkey_hoverdata['points'][0]['customdata']

        if (not isinstance(monkey_hoverdata_value, int)) & (not isinstance(monkey_hoverdata_value, float)):
            # if monkey_hoverdata_value is a list
            try:
                monkey_hoverdata_value = monkey_hoverdata_value[0]
            except TypeError:
                raise PreventUpdate(
                    "No update was triggered because monkey_hoverdata_value is not a list.")
        if monkey_hoverdata_value is None:
            raise PreventUpdate(
                "No update was triggered because monkey_hoverdata_value is None.")

        ONLY_UPDATE_EYE_POSITION = False
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_time_series.find_monkey_hoverdata_value_for_both_fig_time_series(
            self.hoverdata_column, monkey_hoverdata_value, self.current_plotly_key_comp['trajectory_df'])
        if self.curv_of_traj_params['curv_of_traj_mode'] == 'now to stop':
            if (self.monkey_hoverdata_value_s >= self.hoverdata_value_upper_bound_s) or (self.monkey_hoverdata_value_cm >= self.hoverdata_value_upper_bound_cm):
                ONLY_UPDATE_EYE_POSITION = True
        self.monkey_hoverdata_value = monkey_hoverdata_value
        self._find_point_index_to_show_traj_curv()
        if not ONLY_UPDATE_EYE_POSITION:
            self.fig = self._update_fig_based_on_monkey_hover_data()

        if self.monkey_plot_params['show_current_eye_positions']:
            self.fig = self._update_eye_positions_based_on_monkey_hoverdata(
                self.point_index_to_show_traj_curv)

        if self.show_trajectory_time_series:
            self.fig_time_series_combd = dash_utils.update_fig_time_series_combd_plot_based_on_monkey_hoverdata(
                self.fig_time_series_combd, self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm)
            
        if self.show_neural_plots:
            self.fig_raster, self.fig_fr = self._update_neural_plots_based_on_monkey_hover_data(
                self.monkey_hoverdata_value_s)

        return self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr

    def _update_eye_positions_based_on_monkey_hoverdata(self, point_index_to_show_traj_curv):
        show_eye_positions_for_both_eyes = self.monkey_plot_params[
            'show_eye_positions_for_both_eyes']
        self.fig = self._plot_eye_positions_for_dash(self.fig, point_index_to_show_traj_curv=point_index_to_show_traj_curv,
                                                     show_eye_positions_for_both_eyes=show_eye_positions_for_both_eyes,
                                                     use_arrow_to_show_eye_positions=True, marker_size=15
                                                     )
        return self.fig

    def _update_dash_based_on_time_series_plot_hoverdata(self, time_series_plot_hoverdata):
        time_series_plot_hoverdata_values = time_series_plot_hoverdata['points'][0]['x']

        curveNumber = time_series_plot_hoverdata["points"][0]["curveNumber"]

        trace_name = self.fig_time_series_combd.data[curveNumber]['name']
        if 'time_series_cm_' in trace_name:
            x_column_name = 'rel_distance'
        else:
            x_column_name = 'rel_time'
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_time_series.find_monkey_hoverdata_value_for_both_fig_time_series(
            x_column_name, time_series_plot_hoverdata_values, self.current_plotly_key_comp['trajectory_df'])

        if self.hoverdata_column == 'rel_distance':
            self.monkey_hoverdata_value = self.monkey_hoverdata_value_cm
        else:
            self.monkey_hoverdata_value = self.monkey_hoverdata_value_s

        self.fig, self.fig_time_series_combd = self._update_fig_and_fig_time_series_based_on_monkey_hover_data()
        self.fig_raster, self.fig_fr = self._update_neural_plots_based_on_monkey_hover_data(
            self.monkey_hoverdata_value_s)

        return self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr

    def _update_dash_based_on_neural_plot_hoverdata(self, neural_plot_hoverdata):
        neural_plot_hoverdata_values = neural_plot_hoverdata['points'][0]['x']

        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_time_series.find_monkey_hoverdata_value_for_both_fig_time_series(
            'rel_time', neural_plot_hoverdata_values, self.current_plotly_key_comp['trajectory_df'])

        self.monkey_hoverdata_value = self.monkey_hoverdata_value_s

        self.fig, self.fig_time_series_combd = self._update_fig_and_fig_time_series_based_on_monkey_hover_data()
        self.fig_raster, self.fig_fr = self._update_neural_plots_based_on_monkey_hover_data(
            self.monkey_hoverdata_value_s)

        return self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr

    def _update_dash_based_on_correlation_plot_clickdata(self, hoverData):
        self.stop_point_index = hoverData['points'][0]['customdata']
        self.stops_near_ff_row = self.stops_near_ff_df[self.stops_near_ff_df['stop_point_index']
                                                       == self.stop_point_index].iloc[0]
        self._update_after_changing_stop_point_index()
        


    def _update_dash_after_clicking_previous_or_next_plot_button(self, previous_or_next='next'):
        rank_var = 'rank_by_angle_to_nxt_ff'
        stops_near_ff_df = self.stops_near_ff_df
        if previous_or_next == 'previous':
            self._get_previous_stops_near_ff_row(stops_near_ff_df, rank_var)
        elif previous_or_next == 'next':
            self._get_next_stops_near_ff_row(stops_near_ff_df, rank_var)
        else:
            raise ValueError('previous_or_next should be previous or next')
        self._update_after_changing_stop_point_index()

    def _get_previous_stops_near_ff_row(self, stops_near_ff_df, rank_var):
        old_rank = self.stops_near_ff_row[rank_var].item()
        if old_rank == stops_near_ff_df[rank_var].min():
            old_rank = stops_near_ff_df[rank_var].max()
        self.stops_near_ff_row = stops_near_ff_df[self.stops_near_ff_df[rank_var]
                                                  == old_rank - 1].iloc[0]
        self.stop_point_index = self.stops_near_ff_row['stop_point_index']

    def _get_next_stops_near_ff_row(self, stops_near_ff_df, rank_var):
        old_rank = self.stops_near_ff_row[rank_var].item()
        if old_rank == stops_near_ff_df[rank_var].max():
            old_rank = stops_near_ff_df[rank_var].min()
        self.stops_near_ff_row = stops_near_ff_df[self.stops_near_ff_df[rank_var]
                                                  == old_rank + 1].iloc[0]
        self.stop_point_index = self.stops_near_ff_row['stop_point_index']

    def _update_after_changing_stop_point_index(self):
        self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(
            self.stop_point_index)
        self.fig, self.fig_time_series_combd, self.fig_time_series_natural_y_range = self._produce_initial_plots()
        self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._make_two_fig_scatter_or_reg()
        if self.show_neural_plots:
            self.fig_raster = self._create_raster_plot_figure()
            self.fig_fr = self._create_firing_rate_plot_figure()


    def _update_dash_based_on_curv_of_traj_df(self, curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end):
        self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
        self.curv_of_traj_params['window_for_curv_of_traj'] = [
            curv_of_traj_lower_end, curv_of_traj_upper_end]
        self._rerun_after_changing_curv_of_traj_params()
        self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._make_two_fig_scatter_or_reg()
        self._get_curv_of_traj_in_duration()
        self.fig = self._update_fig_based_on_curv_of_traj()
        self.fig_time_series_combd = plotly_for_time_series.add_new_curv_of_traj_to_fig_time_series_combd(
            self.fig_time_series_combd, self.curv_of_traj_in_duration, curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end)
        self.fig_time_series_natural_y_range = plotly_for_time_series.update_fig_time_series_natural_y_range(
            self.fig_time_series_natural_y_range, self.curv_of_traj_in_duration, y_column_name='curv_of_traj_deg_over_cm')
        self._update_fig_time_series_combd_y_range()
        return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2

    def _update_dash_based_on_new_ref_point_descr(self, ref_point_mode, ref_point_value):

        if ref_point_mode == 'distance':
            self.ref_point_descr = 'based on %d cm into past' % ref_point_value
            self.ref_point_column = 'rel_distance'
        elif ref_point_mode == 'time':
            self.ref_point_descr = 'based on %d s into past' % ref_point_value
            self.ref_point_column = 'rel_time'
        else:
            print('Warnings: ref_point_mode is not recognized, so no update is made')
            raise PreventUpdate(
                "No update was made because ref_point_mode is not recognized.")

        self.ref_point_params['ref_point_mode'] = ref_point_mode
        self.ref_point_params['ref_point_value'] = ref_point_value
        self.snf_streamline_organizing_info_kwargs['ref_point_mode'] = ref_point_mode
        self.snf_streamline_organizing_info_kwargs['ref_point_value'] = ref_point_value

        self.streamline_organizing_info(
            **self.snf_streamline_organizing_info_kwargs)
        if len(self.stops_near_ff_df_counted) == 0:
            print(
                'Warning: After update, stops_near_ff_df_counted is empty! So no update is made')
            raise PreventUpdate(
                "No update was made because stops_near_ff_df_counted is empty after update.")
        self._prepare_static_main_plots()

        print(
            f'update all plots based on new reference point description: {self.ref_point_descr}.')
        return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg

    def _find_point_index_to_show_traj_curv(self):
        try:
            self.point_index_to_show_traj_curv = self.current_plotly_key_comp['trajectory_df'].loc[
                self.current_plotly_key_comp['trajectory_df'][self.hoverdata_column] >= self.monkey_hoverdata_value, 'point_index'].iloc[0]
        except IndexError:
            self.point_index_to_show_traj_curv = int(
                self.current_plotly_key_comp['trajectory_df'].iloc[-1]['point_index'])

    def _update_fig_and_fig_time_series_based_on_monkey_hover_data(self):
        self.fig_time_series_combd = dash_utils.update_fig_time_series_combd_plot_based_on_monkey_hoverdata(
            self.fig_time_series_combd, self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm)
        self._find_point_index_to_show_traj_curv()
        self.fig = self._update_fig_based_on_monkey_hover_data()
        if self.monkey_plot_params['show_current_eye_positions']:
            self.fig = self._update_eye_positions_based_on_monkey_hoverdata(
                self.point_index_to_show_traj_curv)

        return self.fig, self.fig_time_series_combd

    def _update_fig_based_on_monkey_hover_data(self):
        # also update the monkey plot
        if self.monkey_plot_params['show_traj_portion']:
            self.traj_portion = self._find_traj_portion()
            self.fig = dash_utils.update_marked_traj_portion_in_monkey_plot(
                self.fig, self.traj_portion, hoverdata_multi_columns=self.hoverdata_multi_columns)

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = self._update_null_arcs_for_cur_and_nxt_ff_in_plotly()

        if self.monkey_plot_params['show_extended_traj_arc']:
            self.fig = self._update_extended_traj_arc_in_plotly()

        if self.monkey_plot_params['show_monkey_heading']:
            self.fig = self._update_all_monkey_heading_in_fig()

        return self.fig

    def _find_traj_portion(self):
        self.curv_of_traj_current_row = self.curv_of_traj_df[self.curv_of_traj_df[
            'point_index'] == self.point_index_to_show_traj_curv].copy()
        self.traj_portion, self.traj_length = dash_utils._find_traj_portion_for_traj_curv(
            self.current_plotly_key_comp['trajectory_df'], self.curv_of_traj_current_row)
        return self.traj_portion

    def _update_fig_based_on_curv_of_traj(self):
        # also update monkey plot
        # point_index_to_mark = self.current_plotly_key_comp['trajectory_df'].loc[self.current_plotly_key_comp['trajectory_df'][self.hoverdata_column] >= self.monkey_hoverdata_value, 'point_index'].iloc[0]
        # curv_of_traj_current_row = self.curv_of_traj_df[self.curv_of_traj_df['point_index']==point_index_to_mark].copy()
        if self.monkey_plot_params['show_traj_portion']:
            self.traj_portion, self.traj_length = dash_utils._find_traj_portion_for_traj_curv(
                self.current_plotly_key_comp['trajectory_df'], self.curv_of_traj_current_row)
            self.fig = dash_utils.update_marked_traj_portion_in_monkey_plot(
                self.fig, self.traj_portion, hoverdata_multi_columns=self.hoverdata_multi_columns)
        if self.monkey_plot_params['show_monkey_heading']:
            self.fig = monkey_heading_utils.update_monkey_heading_in_monkey_plot(
                self.fig, self.traj_triangle_df_in_duration, trace_name_prefix='monkey heading on trajectory', point_index=self.point_index_to_show_traj_curv)
        return self.fig

    def generate_other_messages(self, decimals=2):

        current_traj_curv = round(self.curv_for_correlation_df.loc[self.curv_for_correlation_df['stop_point_index']
                                                                   == self.stop_point_index, 'traj_curv_counted'].item() * (180/np.pi) * 100, decimals)
        current_nxt_curv = round(self.curv_for_correlation_df.loc[self.curv_for_correlation_df['stop_point_index']
                                                                  == self.stop_point_index, 'nxt_curv_counted'].item() * (180/np.pi) * 100, decimals)
        self.other_messages = f"Trajectory curvature: {current_traj_curv}, \n     Nxt FF curvature: {current_nxt_curv}"

        # Add other info
        self.other_messages += ", \n Curv range: " + str(round(self.stops_near_ff_row['curv_range'], decimals)) + ", \n Cum distance between two stops: " + \
            str(round(
                self.stops_near_ff_row['cum_distance_between_two_stops'], decimals))

        # Also get nxt ff angle at ref point
        nxt_ff_angle_at_ref_point = self.nxt_ff_df_from_ref.loc[self.nxt_ff_df_from_ref['stop_point_index']
                                                                == self.stop_point_index, 'ff_angle'].item() * (180/np.pi)
        self.other_messages += ", \n Nxt FF angle at ref point: " + \
            str(round(nxt_ff_angle_at_ref_point, decimals))
        return self.other_messages

    def _prepare_to_make_plotly_fig_for_dash_given_stop_point_index(self, stop_point_index):
        if self.ff_dataframe is None:
            self.get_more_monkey_data()

        self.stop_point_index = stop_point_index
        self.stops_near_ff_row = self.stops_near_ff_df_counted[
            self.stops_near_ff_df_counted['stop_point_index'] == self.stop_point_index].copy()
        self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted,
                                self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)

        self.current_plotly_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(
            self.stops_near_ff_row, self.PlotTrials_args, self.monkey_plot_params)

        self.trajectory_ref_row = self._find_trajectory_ref_row()
        self.trajectory_next_stop_row = self._find_trajectory_next_stop_row()
        self.monkey_hoverdata_value = self.trajectory_ref_row[self.hoverdata_column]
        self.point_index_to_show_traj_curv = self.trajectory_ref_row['point_index'].astype(
            int)

        self._further_prepare_plotting_info_for_the_duration()

    def _further_prepare_plotting_info_for_the_duration(self):
        self.curv_of_traj_in_duration = self._get_curv_of_traj_in_duration()
        try:
            self.curv_of_traj_current_row = self.curv_of_traj_in_duration[
                self.curv_of_traj_in_duration['point_index'] == self.point_index_to_show_traj_curv].copy()
            self.traj_portion, self.traj_length = dash_utils._find_traj_portion_for_traj_curv(
                self.current_plotly_key_comp['trajectory_df'], self.curv_of_traj_current_row)
        except IndexError:
            self.traj_portion = None
            self.traj_length = 0

        self.prepare_both_ff_df()
        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.find_null_arcs_info_for_plotting_for_the_duration()

        if self.monkey_plot_params['show_extended_traj_arc']:
            self._find_ext_traj_arc_info_for_duration()

        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.find_all_mheading_and_triangle_df_for_the_duration(
                self)

    def prepare_both_ff_df(self):
        self.cur_ff_index = self.stops_near_ff_row.cur_ff_index
        self.nxt_ff_index = self.stops_near_ff_row.nxt_ff_index
        self.opt_arc_stop_first_vis_bdry = True if (
            self.opt_arc_type == 'opt_arc_stop_first_vis_bdry') else False

        duration = self.current_plotly_key_comp['duration_to_plot']
        self.point_indexes_in_duration = self.monkey_information.loc[self.monkey_information['time'].between(
            duration[0], duration[1]), 'point_index'].values
        # # can also do:
        # self.point_indexes_in_duration = self.curv_of_traj_in_duration['point_index'].values

        self.cur_curv_df, self.cur_ff_info = plotly_for_null_arcs.find_best_arc_df_for_ff([self.cur_ff_index] * len(self.point_indexes_in_duration),
                                                                                          self.point_indexes_in_duration, self.curv_of_traj_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                          opt_arc_stop_first_vis_bdry=self.opt_arc_stop_first_vis_bdry)
        self.nxt_curv_df, self.nxt_ff_info = plotly_for_null_arcs.find_best_arc_df_for_ff([self.nxt_ff_index] * len(self.point_indexes_in_duration),
                                                                                          self.point_indexes_in_duration, self.curv_of_traj_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                          opt_arc_stop_first_vis_bdry=self.opt_arc_stop_first_vis_bdry)
        self.both_ff_df = pn_utils._merge_both_ff_df(
            self.cur_curv_df, self.nxt_ff_info)
        self.both_ff_df['point_index_before_stop'] = self.stops_near_ff_row.point_index_before_stop

    def add_diff_in_curv_info_to_curv_of_traj_in_duration(self):
        self.curv_of_traj_in_duration = pn_utils.add_diff_in_curv_info(self.curv_of_traj_in_duration, self.both_ff_df,
                                                                       self.monkey_information, self.ff_real_position_sorted, self.ff_caught_T_new)

    def add_diff_in_abs_angle_to_nxt_ff_to_curv_of_traj_in_duration(self):
        self._add_angle_to_nxt_ff_curv_of_traj_in_duration()

        angle_from_stop_to_nxt_ff = pn_utils.calculate_angle_from_stop_to_nxt_ff(self.monkey_information, self.stops_near_ff_row.point_index_before_stop,
                                                                                 self.stops_near_ff_row.nxt_ff_x, self.stops_near_ff_row.nxt_ff_y)
        self.curv_of_traj_in_duration['angle_from_stop_to_nxt_ff'] = angle_from_stop_to_nxt_ff

        # add diff_in_angle_to_nxt_ff, diff_in_abs_angle_to_nxt_ff
        if 'diff_in_angle_to_nxt_ff' not in self.curv_of_traj_in_duration.columns:
            build_factor_comp._add_diff_in_abs_angle_to_nxt_ff(
                self.curv_of_traj_in_duration)

    def find_null_arcs_info_for_plotting_for_the_duration(self):

        if self.overall_params['use_curv_to_ff_center']:
            all_point_index = self.curv_of_traj_in_duration['point_index'].values
            self.cur_null_arc_info_for_duration = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, np.repeat(
                np.array([self.cur_ff_index]), len(all_point_index)), self.monkey_information, self.ff_real_position_sorted, verbose=False)
            self.nxt_null_arc_info_for_duration = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, np.repeat(
                np.array([self.nxt_ff_index]), len(all_point_index)), self.monkey_information, self.ff_real_position_sorted, verbose=False)
        else:
            self.cur_null_arc_info_for_duration = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
                self.cur_curv_df, self.monkey_information)
            self.nxt_null_arc_info_for_duration = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
                self.nxt_curv_df, self.monkey_information)

    def _add_angle_to_nxt_ff_curv_of_traj_in_duration(self):
        angle_df = pn_utils.get_angle_from_cur_arc_end_to_nxt_ff(
            self.both_ff_df)
        columns_to_merge = ['cur_opt_arc_end_heading',
                            'angle_opt_cur_end_to_nxt_ff']
        self.curv_of_traj_in_duration.drop(
            columns=columns_to_merge, errors='ignore', inplace=True)
        self.curv_of_traj_in_duration = self.curv_of_traj_in_duration.merge(
            angle_df[['point_index'] + columns_to_merge], on='point_index', how='left')
