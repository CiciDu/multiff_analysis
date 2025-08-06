import sys
from visualization.dash_tools import dash_utils
from visualization.dash_tools.dash_main_class_methods import dash_main_helper_class
from visualization.dash_tools.dash_config import DEFAULT_PORT, DEFAULT_EXTERNAL_STYLESHEETS

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
        self.freeze_scatterplot = False
        self._setup_default_figures()

    def _setup_default_figures(self):
        """Initialize default empty figures to avoid repeated creation"""
        self._empty_fig_template = go.Figure()
        self._empty_fig_template.update_layout(height=10, width=10)

    def _get_empty_figure(self):
        """Get a copy of the empty figure template to avoid race conditions"""
        self._empty_fig_template = go.Figure()
        self._empty_fig_template.update_layout(height=10, width=10)
        return self._empty_fig_template

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

        # Add scatter plot conditionally
        if self.show_trajectory_scatter_plot:
            layout.append(dash_utils.put_down_scatter_plot(
                self.fig_scatter_combd, id=id_prefix+'scatterplot_combined'
            ))
        else:
            layout.append(dash_utils.put_down_empty_plot_that_takes_no_space(
                id=id_prefix+'scatterplot_combined'
            ))

        # Add remaining components
        more_to_add = [
            dash_utils.print_other_messages(
                id_prefix=id_prefix, other_messages=self.other_messages),
            dash_utils.put_down_monkey_plot(
                self.fig, self.monkey_hoverdata_value, id=id_prefix+'monkey_plot'),
            dash_utils.put_down_the_previous_plot_and_next_plot_button(
                ids=[id_prefix+'previous_plot_button',
                     id_prefix+'next_plot_button']
            ),
            self._put_down_correlation_plots_in_dash(id_prefix=id_prefix)
        ]
        layout.extend(more_to_add)
        return html.Div(layout)

    def make_dash_for_main_plots(self, show_trajectory_scatter_plot=True, show_shuffled_correlation_plot=True, port=DEFAULT_PORT):
        self.show_trajectory_scatter_plot = show_trajectory_scatter_plot
        self.show_shuffled_correlation_plot = show_shuffled_correlation_plot

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

        self.app.run(debug=True, mode='inline', port=port)

    def _register_all_callbacks(self):
        """Register all callbacks in one place for better organization"""
        self.make_function_to_update_all_plots_based_on_new_info(self.app)
        self.make_function_to_show_or_hind_visible_segments(self.app)
        self.make_function_to_update_based_on_correlation_plot(self.app)

        if self.show_shuffled_correlation_plot:
            self.make_function_to_refresh_fig_corr_2(self.app)

        self.make_function_to_update_curv_of_traj(self.app)

    def make_function_to_update_all_plots_based_on_new_info(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'scatterplot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot_2',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "error_message",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'monkey_plot', 'hoverData'),
            Input(self.id_prefix + 'scatterplot_combined', 'hoverData'),
            Input(self.id_prefix + 'scatterplot_combined', 'relayoutData'),
            Input(self.id_prefix + 'update_ref_point', 'n_clicks'),
            Input(self.id_prefix + 'checklist_for_all_plots', 'value'),
            Input(self.id_prefix + 'checklist_for_monkey_plot', 'value'),
            State(self.id_prefix + 'ref_point_mode', 'value'),
            State(self.id_prefix + 'ref_point_value', 'value'),
            prevent_initial_call=True
        )
        def update_all_plots_based_on_new_info(monkey_hoverdata, scatter_plot_hoverdata, scatter_plot_relayoutData,
                                    update_ref_point, checklist_for_all_plots, checklist_for_monkey_plot,
                                    ref_point_mode, ref_point_value):

            try:
                # Reset freeze flag if not triggered by scatter plot
                if self.id_prefix + 'scatterplot_combined' not in ctx.triggered[0]['prop_id']:
                    self.freeze_scatterplot = False

                # Handle different trigger types
                trigger_id = ctx.triggered[0]['prop_id']

                if trigger_id == self.id_prefix + 'monkey_plot.hoverData':
                    if 'customdata' in monkey_hoverdata['points'][0]:
                        self.monkey_hoverdata = monkey_hoverdata
                        fig, fig_scatter_combd = self._update_dash_based_on_monkey_hover_data(
                            monkey_hoverdata)
                    else:
                        raise PreventUpdate("No update was triggered because customdata is not in monkey_hoverdata[\'points\'][0].")

                elif trigger_id == self.id_prefix + 'scatterplot_combined.relayoutData':
                    self.freeze_scatterplot = True
                    raise PreventUpdate("No update was triggered because trigger ID was related to scatterplot_combined.relayoutData.")

                elif trigger_id == self.id_prefix + 'scatterplot_combined.hoverData':
                    if self.freeze_scatterplot:
                        raise PreventUpdate("No update was triggered because freeze_scatterplot is True.")
                    if 'x' in scatter_plot_hoverdata['points'][0]:
                        fig, fig_scatter_combd = self._update_dash_based_on_scatter_plot_hoverdata(
                            scatter_plot_hoverdata)
                    else:
                        raise PreventUpdate("No update was made because x is not in scatter_plot_hoverdata.")

                elif trigger_id == self.id_prefix + 'update_ref_point.n_clicks':
                    if ref_point_value is not None and ref_point_value < 0:
                        fig, fig_scatter_combd, self.fig_corr_or_heading = self._update_dash_based_on_new_ref_point_descr(
                            ref_point_mode, ref_point_value
                        )
                    else:
                        if ref_point_value is not None and ref_point_value >= 0:
                            print(
                                'Warning: ref_point_value should not be negative. No update is made.')
                        raise PreventUpdate("No update was made because ref_point_value is None or negative.")

                elif trigger_id == self.id_prefix + 'checklist_for_all_plots.value':
                    fig, fig_scatter_combd, self.fig_corr_or_heading, self.fig_corr_or_heading_2 = self._update_dash_based_on_checklist_for_all_plots(
                        checklist_for_all_plots)

                elif trigger_id == self.id_prefix + 'checklist_for_monkey_plot.value':
                    fig, fig_scatter_combd = self._update_dash_based_on_checklist_for_monkey_plot(
                        checklist_for_monkey_plot)

                else:
                    raise PreventUpdate("No update was made for the current trigger.")

                # Handle conditional plot visibility
                if not self.show_trajectory_scatter_plot:
                    fig_scatter_combd = self._get_empty_figure()

                if not self.show_shuffled_correlation_plot:
                    self.fig_corr_or_heading_2 = self._get_empty_figure()

                return fig, fig_scatter_combd, self.fig_corr_or_heading, self.fig_corr_or_heading_2, 'Updated successfully'

            except Exception as e:
                return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading, self.fig_corr_or_heading_2, f"An error occurred. No update was made. Error: {e}"

    def make_function_to_update_based_on_correlation_plot(self, app):
        @app.callback(
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'scatterplot_combined',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'correlation_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "other_messages",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'correlation_plot', 'clickData'),
            Input(self.id_prefix + 'previous_plot_button', 'n_clicks'),
            Input(self.id_prefix + 'next_plot_button', 'n_clicks'),
            prevent_initial_call=True
        )
        def update_other_messages(correlation_plot_clickdata, previous_plot_button, next_plot_button):
            trigger_id = ctx.triggered[0]['prop_id']

            if trigger_id == self.id_prefix + 'previous_plot_button.n_clicks':
                fig, fig_scatter_combd, self.fig_corr_or_heading = self._update_dash_after_clicking_previous_or_next_plot_button(
                    previous_or_next='previous')
            elif trigger_id == self.id_prefix + 'next_plot_button.n_clicks':
                fig, fig_scatter_combd, self.fig_corr_or_heading = self._update_dash_after_clicking_previous_or_next_plot_button(
                    previous_or_next='next')
            else:
                if not 'customdata' in correlation_plot_clickdata['points'][0]:
                    raise PreventUpdate("No update was triggered because customdata is not in correlation_plot_clickdata.")
                self.stop_point_index = correlation_plot_clickdata['points'][0]['customdata']
                fig, fig_scatter_combd, self.fig_corr_or_heading = self._update_dash_based_on_correlation_plot_clickdata(
                    correlation_plot_clickdata)

            self.other_messages = self.generate_other_messages()
            return fig, fig_scatter_combd, self.fig_corr_or_heading, self.other_messages

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
                raise PreventUpdate("No update was triggered because customdata is not in hoverData.")

            if not isinstance(data, int):
                raise PreventUpdate("No update was triggered because hoverdata is not an integer, which ff_index should be.")

            legendgroup = f'ff {data}'
            for trace in self.fig.data:
                if trace.legendgroup == legendgroup:
                    trace.visible = 'legendonly' if trace.visible != 'legendonly' else True
                    logging.info(f'ff {data} is now {trace.visible}.')
                    break

            return self.fig

    def make_function_to_refresh_fig_corr_2(self, app):
        @app.callback(
            Output(self.id_prefix + 'correlation_plot_2',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + "error_message",
                   "children", allow_duplicate=True),
            Input(self.id_prefix + 'refresh_correlation_plot_2', 'n_clicks'),
            prevent_initial_call=True
        )
        def refresh_fig_corr_2(n_clicks):
            try:
                if self.overall_params['heading_instead_of_curv']:
                    self.fig_heading_2 = self._make_fig_heading_2()
                    self.fig_corr_or_heading_2 = self.fig_heading_2
                else:
                    self.fig_corr_2 = self._make_fig_corr_2()
                    self.fig_corr_or_heading_2 = self.fig_corr_2
                return self.fig_corr_or_heading_2, 'Updated successfully'
            except Exception as e:
                return self.fig_corr_or_heading_2, f"An error occurred. No update was made. Error: {e}"

    def make_function_to_update_curv_of_traj(self, app):
        @app.callback(
            Output(self.id_prefix + "curv_of_traj_mode", "value"),
            Output(self.id_prefix + "window_lower_end", "value"),
            Output(self.id_prefix + "window_upper_end", "value"),
            Output(self.id_prefix + 'monkey_plot',
                   'figure', allow_duplicate=True),
            Output(self.id_prefix + 'scatterplot_combined',
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
            prevent_initial_call='initial_duplicate'
        )
        def update_curv_of_traj_values(update_curv_of_traj, curv_of_traj_mode, window_lower_end, window_upper_end):
            try:
                self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
                trigger_id = ctx.triggered[0]['prop_id']

                if trigger_id == self.id_prefix + 'update_curv_of_traj.n_clicks':
                    if (window_lower_end < window_upper_end) or (curv_of_traj_mode == 'now to stop'):
                        self.fig, self.fig_scatter_combd, self.fig_corr_or_heading = self._update_dash_based_on_curv_of_traj_df(
                            curv_of_traj_mode, window_lower_end, window_upper_end
                        )
                    else:
                        print(
                            'Warning: curv_of_traj_lower_end is larger than curv_of_traj_upper_end, so no update is made')
                        raise PreventUpdate("No update was made because curv_of_traj_lower_end is larger than curv_of_traj_upper_end.")

                elif trigger_id == self.id_prefix + 'curv_of_traj_mode.value':
                    if self.curv_of_traj_params['curv_of_traj_mode'] == 'now to stop':
                        self.curv_of_traj_params['window_for_curv_of_traj'] = [
                            0, 0]
                    else:
                        raise PreventUpdate("No update was made because curv_of_traj_lower_end is larger than curv_of_traj_upper_end.")

                # Handle conditional plot visibility
                if not self.show_trajectory_scatter_plot:
                    self.fig_scatter_combd = self._get_empty_figure()

                if not self.show_shuffled_correlation_plot:
                    self.fig_corr_or_heading_2 = self._get_empty_figure()

                return (self.curv_of_traj_params['curv_of_traj_mode'],
                        self.curv_of_traj_params['window_for_curv_of_traj'][0],
                        self.curv_of_traj_params['window_for_curv_of_traj'][1],
                        self.fig, self.fig_scatter_combd, self.fig_corr_or_heading,
                        self.fig_corr_or_heading_2, 'Updated successfully')

            except Exception as e:
                return (self.curv_of_traj_params['curv_of_traj_mode'],
                        self.curv_of_traj_params['window_for_curv_of_traj'][0],
                        self.curv_of_traj_params['window_for_curv_of_traj'][1],
                        self.fig, self.fig_scatter_combd, self.fig_corr_or_heading,
                        self.fig_corr_or_heading_2, f"An error occurred. No update was made. Error: {e}")
