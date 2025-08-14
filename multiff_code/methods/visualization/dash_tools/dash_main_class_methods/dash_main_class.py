import sys
from visualization.dash_tools import dash_utils
from visualization.dash_tools.dash_main_class_methods import dash_main_helper_class
from visualization.dash_tools.dash_config import DEFAULT_PORT, DEFAULT_EXTERNAL_STYLESHEETS
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import raster_and_fr_plot_in_plotly
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import Dash, html, Input, State, Output, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import logging

# Import neural data visualization tools
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import raster_plot, plot_neural_data

# Import shared configuration
from visualization.dash_tools.dash_config import configure_plotting_environment
configure_plotting_environment()

# Configuration - moved to a shared config module or base class
plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

# https://dash.plotly.com/interactive-graphing


class DashMainPlots(dash_main_helper_class.DashMainHelper):

    def __init__(self, raw_data_folder_path=None, opt_arc_type='opt_arc_stop_closest'):
        super().__init__(raw_data_folder_path=raw_data_folder_path, opt_arc_type=opt_arc_type)
        self.freeze_time_series = False

    def _get_empty_figure(self):
        """Get a copy of the empty figure template to avoid race conditions"""
        empty_fig = go.Figure()
        empty_fig.update_layout(height=10, width=10)
        return empty_fig

    def prepare_dash_for_main_plots_layout(self, id_prefix='main_plots_'):
        self.id_prefix = id_prefix
        self.other_messages = self.generate_other_messages()

        layout = [
            self._put_down_the_menus_on_top(id_prefix=id_prefix),
            dash_utils.put_down_the_refreshing_buttons_for_ref_point_and_curv_of_traj(
                ids=[id_prefix+'update_ref_point',
                     id_prefix+'update_curv_of_traj']
            ),
            self._put_down_checklist_for_all_plots(id_prefix=id_prefix),
            self._put_down_checklist_for_monkey_plot(id_prefix=id_prefix),
            dash_utils.create_error_message_display(id_prefix=id_prefix)
        ]

        # conditionally add trajectory time series plot and neural plots
        layout = self._put_down_trajectory_time_series_plot(
            layout, id_prefix=id_prefix)
        layout = self._put_down_neural_plots(layout, id_prefix=id_prefix)

        # Add remaining components
        more_to_add = [
            dash_utils.put_down_monkey_plot(
                self.fig, self.monkey_hoverdata_value, id=id_prefix+'monkey_plot'),
            dash_utils.put_down_the_previous_plot_and_next_plot_button(
                ids=[id_prefix+'previous_plot_button',
                     id_prefix+'next_plot_button']
            ),
            self._put_down_correlation_plots_in_dash(id_prefix=id_prefix),
            dash_utils.print_other_messages(
                id_prefix=id_prefix, other_messages=self.other_messages)
        ]
        layout.extend(more_to_add)
        return html.Div(layout)

    def make_dash_for_main_plots(self, show_trajectory_time_series=True, show_neural_plots=True,
                                 port=DEFAULT_PORT):

        self.show_trajectory_time_series = show_trajectory_time_series
        self.show_neural_plots = show_neural_plots

        self.app = Dash(
            __name__, external_stylesheets=DEFAULT_EXTERNAL_STYLESHEETS)
        self.app.layout = self.prepare_dash_for_main_plots_layout()

        # Pre-calculate bounds once
        self.hoverdata_value_upper_bound_s = dash_utils.find_hoverdata_value_upper_bound(
            self.stops_near_ff_row, 'rel_time'
        )
        self.hoverdata_value_upper_bound_cm = dash_utils.find_hoverdata_value_upper_bound(
            self.stops_near_ff_row, 'rel_distance'
        )

        # Register all callbacks
        self._register_all_callbacks()

        print(f"Opening dash in browser at: http://127.0.0.1:{port}")
        self.app.run(debug=True, mode='external', port=port, host='0.0.0.0')

    def _register_all_callbacks(self):
        """Register all callbacks in one place for better organization"""
        self.make_function_to_update_all_plots_based_on_hover_data(self.app)
        self.make_function_to_update_all_plots_based_on_direct_input(self.app)
        self.make_function_to_show_or_hind_visible_segments(self.app)
        self.make_function_to_update_based_on_correlation_plot(self.app)
        self.make_function_to_update_curv_of_traj(self.app)

    def make_function_to_update_all_plots_based_on_direct_input(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'time_series_plot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot_2',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "error_message",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'update_ref_point', 'n_clicks'),
            Input(self.id_prefix + 'checklist_for_all_plots', 'value'),
            Input(self.id_prefix + 'checklist_for_monkey_plot', 'value'),
            State(self.id_prefix + 'ref_point_mode', 'value'),
            State(self.id_prefix + 'ref_point_value', 'value'),
            prevent_initial_call=True
        )
        def update_all_plots_based_on_direct_input(update_ref_point, checklist_for_all_plots, checklist_for_monkey_plot,
                                                   ref_point_mode, ref_point_value):

            try:
                trigger_id = ctx.triggered[0]['prop_id']

                # Reset freeze flag if not triggered by time series plot
                if trigger_id == self.id_prefix + 'update_ref_point.n_clicks':
                    if ref_point_value is not None and (ref_point_value < 0):
                        self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg = self._update_dash_based_on_new_ref_point_descr(
                            ref_point_mode, ref_point_value
                        )
                    else:
                        raise PreventUpdate(
                            "No update was made because ref_point_value is None or not negative.")

                elif trigger_id == self.id_prefix + 'checklist_for_all_plots.value':
                    self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._update_dash_based_on_checklist_for_all_plots(
                        checklist_for_all_plots)

                elif trigger_id == self.id_prefix + 'checklist_for_monkey_plot.value':
                    self.fig, self.fig_time_series_combd = self._update_dash_based_on_checklist_for_monkey_plot(
                        checklist_for_monkey_plot)
                else:
                    raise PreventUpdate(
                        "No update was made for the current trigger.")

                return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2, 'Updated successfully'

            except Exception as e:
                return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2, f"An error occurred. No update was made. Error: {e}"

    def make_function_to_update_all_plots_based_on_hover_data(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'time_series_plot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'raster_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'firing_rate_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "error_message",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'monkey_plot', 'hoverData'),
            Input(self.id_prefix + 'time_series_plot_combined', 'clickData'),
            Input(self.id_prefix + 'time_series_plot_combined', 'relayoutData'),
            Input(self.id_prefix + 'raster_plot', 'clickData'),
            Input(self.id_prefix + 'firing_rate_plot', 'clickData'),
            prevent_initial_call=True
        )
        def update_all_plots_based_on_hover_data(monkey_hoverdata, time_series_plot_hoverdata, time_series_plot_relayoutData,
                                                 raster_plot_hoverdata, firing_rate_plot_hoverdata):

            try:
                # Reset freeze flag if not triggered by time series plot
                if self.id_prefix + 'time_series_plot_combined' not in ctx.triggered[0]['prop_id']:
                    self.freeze_time_series = False

                # Handle different trigger types
                trigger_id = ctx.triggered[0]['prop_id']

                if trigger_id == self.id_prefix + 'monkey_plot.hoverData':
                    # Safe check for monkey_hoverdata
                    if (monkey_hoverdata and
                        'points' in monkey_hoverdata and
                        len(monkey_hoverdata['points']) > 0 and
                            'customdata' in monkey_hoverdata['points'][0]):
                        self.monkey_hoverdata = monkey_hoverdata
                        self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr = self._update_dash_based_on_monkey_hover_data(
                            monkey_hoverdata)
                    else:
                        raise PreventUpdate(
                            "No update: invalid or missing customdata in monkey_hoverdata.")

                elif trigger_id == self.id_prefix + 'time_series_plot_combined.relayoutData':
                    self.freeze_time_series = True
                    raise PreventUpdate(
                        "No update was triggered because trigger ID was related to time_series_plot_combined.relayoutData.")

                elif trigger_id == self.id_prefix + 'time_series_plot_combined.clickData':
                    if self.freeze_time_series:
                        raise PreventUpdate(
                            "No update was triggered because freeze_time_series is True.")
                    if 'x' in time_series_plot_hoverdata['points'][0]:
                        self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr = self._update_dash_based_on_time_series_plot_hoverdata(
                            time_series_plot_hoverdata)
                    else:
                        raise PreventUpdate(
                            "No update was made because x is not in time_series_plot_hoverdata.")

                elif trigger_id == self.id_prefix + 'raster_plot.clickData':
                    # Handle raster plot hover - update time series plot with vertical line
                    if 'x' in raster_plot_hoverdata['points'][0]:
                        self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr = self._update_dash_based_on_neural_plot_hoverdata(
                            raster_plot_hoverdata)
                    else:
                        raise PreventUpdate(
                            "No update was made because x is not in raster_plot_hoverdata.")

                elif trigger_id == self.id_prefix + 'firing_rate_plot.clickData':
                    if 'x' in firing_rate_plot_hoverdata['points'][0]:
                        self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr = self._update_dash_based_on_neural_plot_hoverdata(
                            firing_rate_plot_hoverdata)
                    else:
                        raise PreventUpdate(
                            "No update was made because x is not in firing_rate_plot_hoverdata.")

                else:
                    raise PreventUpdate(
                        "No update was made for the current trigger.")

                return self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr, 'Updated successfully'

            except Exception as e:
                return self.fig, self.fig_time_series_combd, self.fig_raster, self.fig_fr, f"An error occurred. No update was made. Error: {e}"


    def make_function_to_update_based_on_correlation_plot(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'time_series_plot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot_2',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'raster_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'firing_rate_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "other_messages",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'correlation_plot', 'clickData'),
            Input(self.id_prefix + 'correlation_plot_2', 'clickData'),
            Input(self.id_prefix + 'previous_plot_button', 'n_clicks'),
            Input(self.id_prefix + 'next_plot_button', 'n_clicks'),
            prevent_initial_call=True
        )
        def update_other_messages(correlation_plot_clickdata, correlation_plot_2_clickdata, previous_plot_button, next_plot_button):
            trigger_id = ctx.triggered[0]['prop_id']

            if trigger_id == self.id_prefix + 'previous_plot_button.n_clicks':
                self._update_dash_after_clicking_previous_or_next_plot_button(
                    previous_or_next='previous')
            elif trigger_id == self.id_prefix + 'next_plot_button.n_clicks':
                self._update_dash_after_clicking_previous_or_next_plot_button(
                    previous_or_next='next')
            elif trigger_id == self.id_prefix + 'correlation_plot.clickData':
                if 'customdata' not in correlation_plot_clickdata['points'][0]:
                    raise PreventUpdate(
                        "No update was triggered because customdata is not in correlation_plot_clickdata.")
                self.stop_point_index = correlation_plot_clickdata['points'][0]['customdata']
                self._update_dash_based_on_correlation_plot_clickdata(
                    correlation_plot_clickdata)
            elif trigger_id == self.id_prefix + 'correlation_plot_2.clickData':
                if 'customdata' not in correlation_plot_2_clickdata['points'][0]:
                    raise PreventUpdate(
                        "No update was triggered because customdata is not in correlation_plot_2_clickdata.")
                self.stop_point_index = correlation_plot_2_clickdata['points'][0]['customdata']
                self._update_dash_based_on_correlation_plot_clickdata(
                    correlation_plot_2_clickdata)

            self.other_messages = self.generate_other_messages()

            return self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2, self.fig_raster, self.fig_fr, self.other_messages

    def make_function_to_show_or_hind_visible_segments(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Input(self.id_prefix + "monkey_plot", "clickData"),
            State(self.id_prefix + 'monkey_plot', 'hoverData'),
            prevent_initial_call=True
        )
        def show_or_hind_visible_segments(clickData, hoverData):
            try:
                data = hoverData['points'][0]['customdata'][0]
            except (KeyError, IndexError):
                raise PreventUpdate(
                    "No update was triggered because customdata is not in hoverData.")

            if not isinstance(data, int):
                raise PreventUpdate(
                    "No update was triggered because hoverdata is not an integer, which ff_index should be.")

            legendgroup = f'ff {data}'
            for trace in self.fig.data:
                if trace.legendgroup == legendgroup:
                    trace.visible = 'legendonly' if trace.visible != 'legendonly' else True
                    #logging.info(f'ff {data} is now {trace.visible}.')

            return self.fig

    def make_function_to_update_curv_of_traj(self, app):
        @app.callback(
            Output(self.id_prefix + "curv_of_traj_mode", "value"),
            Output(self.id_prefix + "window_lower_end", "value"),
            Output(self.id_prefix + "window_upper_end", "value"),
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'time_series_plot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot_2',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "error_message",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'update_curv_of_traj', 'n_clicks'),
            Input(self.id_prefix + 'curv_of_traj_mode', 'value'),
            State(self.id_prefix + "window_lower_end", "value"),
            State(self.id_prefix + "window_upper_end", "value"),
            prevent_initial_call=True
        )
        def update_curv_of_traj_values(update_curv_of_traj, curv_of_traj_mode, window_lower_end, window_upper_end):
            try:
                self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
                trigger_id = ctx.triggered[0]['prop_id']

                if trigger_id == self.id_prefix + 'update_curv_of_traj.n_clicks':
                    if (window_lower_end < window_upper_end) or (curv_of_traj_mode == 'now to stop'):
                        self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg, self.fig_scatter_or_reg2 = self._update_dash_based_on_curv_of_traj_df(
                            curv_of_traj_mode, window_lower_end, window_upper_end
                        )
                    else:
                        print(
                            'Warning: curv_of_traj_lower_end is larger than curv_of_traj_upper_end, so no update is made')
                        raise PreventUpdate(
                            "No update was made because curv_of_traj_lower_end is larger than curv_of_traj_upper_end.")

                elif trigger_id == self.id_prefix + 'curv_of_traj_mode.value':
                    if self.curv_of_traj_params['curv_of_traj_mode'] == 'now to stop':
                        self.curv_of_traj_params['window_for_curv_of_traj'] = [
                            0, 0]
                    else:
                        raise PreventUpdate(
                            "No update was made because curv_of_traj_lower_end is larger than curv_of_traj_upper_end.")

                # Handle conditional plot visibility
                if not self.show_trajectory_time_series:
                    self.fig_time_series_combd = self._get_empty_figure()

                return (self.curv_of_traj_params['curv_of_traj_mode'],
                        self.curv_of_traj_params['window_for_curv_of_traj'][0],
                        self.curv_of_traj_params['window_for_curv_of_traj'][1],
                        self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg,
                        self.fig_scatter_or_reg2, 'Updated successfully')

            except Exception as e:
                return (self.curv_of_traj_params['curv_of_traj_mode'],
                        self.curv_of_traj_params['window_for_curv_of_traj'][0],
                        self.curv_of_traj_params['window_for_curv_of_traj'][1],
                        self.fig, self.fig_time_series_combd, self.fig_scatter_or_reg,
                        self.fig_scatter_or_reg2, f"An error occurred. No update was made. Error: {e}")
