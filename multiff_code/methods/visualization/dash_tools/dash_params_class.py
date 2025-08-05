import sys
from planning_analysis.test_params_for_planning import params_utils, params_test_combos_class
from visualization.dash_tools import dash_utils
from dash import Dash, html, Input, Output, ctx, dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np


class ParamsDash(params_test_combos_class.ParamsTestCombos):

    def __init__(self,
                 raw_data_folder_path=None):

        super().__init__(raw_data_folder_path=raw_data_folder_path)

        self.hyperparameter_dict = {'curv_of_traj_mode': 'time',
                                    'ref_point_mode': 'time',
                                    # 'time_or_distance' : 'time',
                                    'truncate_curv_of_traj_by_time_of_capture': False,
                                    'eliminate_outliers': True,
                                    'use_curv_to_ff_center': False,
                                    'heading_instead_of_curv': True}

    def _put_down_checklist_for_params(self, id_prefix=None):

        checklist_options = [{'label': 'show heading instead of curv', 'value': 'heading_instead_of_curv'},
                             {'label': 'truncate curv of traj by time of capture',
                                 'value': 'truncate_curv_of_traj_by_time_of_capture'},
                             {'label': 'eliminate outliers',
                                 'value': 'eliminate_outliers'},
                             {'label': 'use curvature to ff center',
                                 'value': 'use_curv_to_ff_center'},
                             ]

        checklist_params = ['heading_instead_of_curv',
                            'eliminate_outliers', 'use_curv_to_ff_center']
        checklist_values = [
            key for key in checklist_params if self.hyperparameter_dict[key] is True]

        return html.Div([dcc.Checklist(options=checklist_options,
                                       value=checklist_values,
                                       id=id_prefix+'checklist_for_all_plots',
                                       style={'width': '50%', 'background-color': '#F9F99A', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'})])

    def prepare_for_dash(self,
                         tested_combo_df,
                         hyperparameter_dict=None,
                         ):

        if hyperparameter_dict is not None:
            self.hyperparameter_dict = hyperparameter_dict

        ref_point_mode = self.hyperparameter_dict['ref_point_mode']
        self.hyperparameter_dict['ref_point_value'] = self.ref_point_info[ref_point_mode]['values'][0]

        self.tested_combo_df = tested_combo_df.copy()
        params_utils.process_tested_combo_df(self.tested_combo_df)
        self.make_combo_df_long()

        self.sub_df = params_utils.get_subset_of_combo_df(
            self.combo_df_long, self.hyperparameter_dict)
        self.unique_window_sizes = self.find_unique_window_sizes()
        self.hyperparameter_dict['window_size'] = self.unique_window_sizes[0]

        self.fig_lines = params_utils.plot_tested_heading_df_or_curv_df2(
            self.sub_df, self.hyperparameter_dict)

    def make_combo_df_long(self):
        self.heading_df, self.curv_df = params_utils.get_heading_df_and_curv_df(
            self.tested_combo_df)
        columns_to_include = ['ref_point_mode', 'ref_point_value', 'same_mode',
                              'curv_of_traj_mode', 'curv_of_traj_lower_end', 'curv_of_traj_upper_end', 'window_size',
                              'heading_instead_of_curv', 'use_curv_to_ff_center',
                              'truncate_curv_of_traj_by_time_of_capture', 'eliminate_outliers',
                              'r', 'sample_size', 'shuffled_r_mean', 'shuffled_r_std', 'r_z_score']

        self.heading_df['heading_instead_of_curv'] = True
        self.curv_df['heading_instead_of_curv'] = False
        self.combo_df_long = pd.concat(
            [self.heading_df[columns_to_include], self.curv_df[columns_to_include]], ignore_index=True)

    def prepare_dash_for_params_layout(self, id_prefix='params_'):
        self.id_prefix = id_prefix
        layout = html.Div([self._put_down_checklist_for_params(id_prefix),
                           # params_utils.put_down_the_dropdown_menu_for_time_or_distance(self.time_or_distance),
                           dash_utils.put_down_the_dropdown_menu_for_ref_point_mode(
            self.hyperparameter_dict['ref_point_mode'], id=id_prefix+'ref_point_mode'),
            params_utils.make_a_slider_for_reference_point(
            self.hyperparameter_dict['ref_point_mode'], self.ref_point_info, id=id_prefix+'ref_point_slider'),
            dash_utils.put_down_the_dropdown_menu_for_curv_of_traj_mode(
            self.hyperparameter_dict['curv_of_traj_mode'], label='Curvature of Trajectory Window Size', id=id_prefix+'curv_of_traj_mode'),
            params_utils.make_a_slider_for_window_size(
            self.unique_window_sizes, id=id_prefix+'window_size_slider'),
            html.Div([dcc.Graph(id=id_prefix+'scatter_plot', figure=self.fig_lines, style={
                'width': '70%', 'padding': '0 0 0 0'})])
        ])
        return layout

    def make_dash_for_params(self):

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        app.layout = self.prepare_dash_for_params_layout()

        self.make_function_to_update_plot_based_on_new_info(app)
        self.make_function_to_update_ref_point_slider_and_plot_based_on_ref_point_mode(
            app)
        self.make_function_to_update_window_size_slider_based_on_curv_of_traj_mode(
            app)

        app.run(debug=True, port=8051)

    def make_function_to_update_window_size_slider_based_on_curv_of_traj_mode(self, app):

        @app.callback(
            Output(self.id_prefix + 'window_size_slider', 'min'),
            Output(self.id_prefix + 'window_size_slider', 'max'),
            Output(self.id_prefix + 'window_size_slider', 'step'),
            Output(self.id_prefix + 'window_size_slider', 'marks'),
            Output(self.id_prefix + 'window_size_slider', 'value'),
            Output(self.id_prefix + 'scatter_plot',
                   'figure', allow_duplicate=True),
            Input(self.id_prefix + 'curv_of_traj_mode', 'value'),
            prevent_initial_call=True)
        def update_window_size_slider(curv_of_traj_mode):
            self.hyperparameter_dict['curv_of_traj_mode'] = curv_of_traj_mode
            self.unique_window_sizes = self.find_unique_window_sizes()
            self.hyperparameter_dict['window_size'] = self.unique_window_sizes[0]

            min = np.min(self.unique_window_sizes)
            max = np.max(self.unique_window_sizes)
            step = self.unique_window_sizes[1] - self.unique_window_sizes[0]
            window_size_marks_dict = {i: str(i)
                                      for i in self.unique_window_sizes}

            self.sub_df = params_utils.get_subset_of_combo_df(
                self.combo_df_long, self.hyperparameter_dict)
            self.fig_lines = params_utils.plot_tested_heading_df_or_curv_df2(
                self.sub_df, self.hyperparameter_dict)

            return min, max, step, window_size_marks_dict, self.unique_window_sizes[0], self.fig_lines

    def make_function_to_update_ref_point_slider_and_plot_based_on_ref_point_mode(self, app):

        @app.callback(
            Output(self.id_prefix + 'ref_point_slider', 'min'),
            Output(self.id_prefix + 'ref_point_slider', 'max'),
            Output(self.id_prefix + 'ref_point_slider', 'step'),
            Output(self.id_prefix + 'ref_point_slider', 'marks'),
            Output(self.id_prefix + 'ref_point_slider', 'value'),
            Output(self.id_prefix + 'scatter_plot',
                   'figure', allow_duplicate=True),
            Input(self.id_prefix + 'ref_point_mode', 'value'),
            prevent_initial_call=True)
        def update_ref_point_slider(ref_point_mode):
            min = self.ref_point_info[ref_point_mode]['min']
            max = self.ref_point_info[ref_point_mode]['max']
            step = self.ref_point_info[ref_point_mode]['step']

            ref_point_marks_dict = self.ref_point_info[ref_point_mode]['marks']
            ref_point_all_values = self.ref_point_info[ref_point_mode]['values']
            ref_point_value = ref_point_all_values[0]

            self.hyperparameter_dict['ref_point_mode'] = ref_point_mode
            self.hyperparameter_dict['ref_point_value'] = ref_point_value
            self.sub_df = params_utils.get_subset_of_combo_df(
                self.combo_df_long, self.hyperparameter_dict)
            self.fig_lines = params_utils.plot_tested_heading_df_or_curv_df2(
                self.sub_df, self.hyperparameter_dict)

            return min, max, step, ref_point_marks_dict, ref_point_value, self.fig_lines

    def make_function_to_update_plot_based_on_new_info(self, app):

        @app.callback(
            Output(self.id_prefix + 'scatter_plot',
                   'figure', allow_duplicate=True),
            Input(self.id_prefix + 'ref_point_slider', 'value'),
            Input(self.id_prefix + 'window_size_slider', 'value'),
            Input(self.id_prefix + 'checklist_for_all_plots', 'value'),
            prevent_initial_call=True)
        def update_correlation_plot(ref_point_value,
                                    window_size,
                                    checklist_for_all_plots):

            if (ctx.triggered[0]['prop_id'] == self.id_prefix + 'ref_point_slider.value'):
                self.hyperparameter_dict['ref_point_value'] = ref_point_value

            elif (ctx.triggered[0]['prop_id'] == self.id_prefix + 'window_size_slider.value'):
                self.hyperparameter_dict['window_size'] = window_size

            elif (ctx.triggered[0]['prop_id'] == self.id_prefix + 'checklist_for_all_plots.value'):
                # update hyperparameters based on checklist
                for param in ['heading_instead_of_curv', 'truncate_curv_of_traj_by_time_of_capture', 'eliminate_outliers', 'use_curv_to_ff_center']:
                    if param in checklist_for_all_plots:
                        self.hyperparameter_dict[param] = True
                    else:
                        self.hyperparameter_dict[param] = False
            else:
                raise PreventUpdate("No update was triggered because trigger ID was not related to ref_point_slider.value, window_size_slider.value, or checklist_for_all_plots.value.")

            self.sub_df = params_utils.get_subset_of_combo_df(
                self.combo_df_long, self.hyperparameter_dict)
            self.fig_lines = params_utils.plot_tested_heading_df_or_curv_df2(
                self.sub_df, self.hyperparameter_dict)

            return self.fig_lines
        return

    # ==================================== Helper Functions ==================================== #

    def find_unique_window_sizes(self):
        self.unique_window_sizes_for_time = np.round(
            self.combo_df_long[self.combo_df_long['curv_of_traj_mode'] == 'time']['window_size'].unique(), 2)
        self.unique_window_sizes_for_distance = self.combo_df_long[self.combo_df_long[
            'curv_of_traj_mode'] == 'distance']['window_size'].unique().astype(int)
        if self.hyperparameter_dict['curv_of_traj_mode'] == 'time':
            self.unique_window_sizes = self.unique_window_sizes_for_time
        elif self.hyperparameter_dict['curv_of_traj_mode'] == 'distance':
            self.unique_window_sizes = self.unique_window_sizes_for_distance
        else:
            raise ValueError('curv_of_traj_mode is not valid')
        self.unique_window_sizes = np.sort(self.unique_window_sizes).tolist()
        return self.unique_window_sizes
