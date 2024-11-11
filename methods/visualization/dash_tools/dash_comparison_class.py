
import sysfrom visualization.dash_tools.dash_main_class_methods import dash_main_class
from visualization.plotly_tools import plotly_for_correlation
from visualization.dash_tools import dash_utils, dash_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import Dash, html, Input, State, Output, ctx
from dash.exceptions import PreventUpdate
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


# https://dash.plotly.com/interactive-graphing




class DashComparison(dash_main_class.DashMainPlots):
        

    def __init__(self, 
                 raw_data_folder_path=None):
        
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.stop_point_index = None




    def prepare_to_make_dash_for_comparison(self, 
                                            ref_point_params=None,
                                            curv_of_traj_params=None,
                                            overall_params=None):
        
        self.ref_point_params = ref_point_params
        self.curv_of_traj_params = curv_of_traj_params
        self.overall_params = overall_params

        self.snf_streamline_organizing_info_kwargs = find_stops_near_ff_utils.organize_snf_streamline_organizing_info_kwargs(ref_point_params, curv_of_traj_params, overall_params)
        super().streamline_organizing_info(**self.snf_streamline_organizing_info_kwargs)

        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        self._make_kwargs_for_correlation_plot()


    def prepare_dash_for_comparison_layout(self):
        return html.Div([
                            self._put_down_the_menus_on_top(),
                            dash_utils.put_down_the_refreshing_buttons_for_ref_point_and_curv_of_traj(),
                            html.Button('Refresh shuffled plot on the right', id='refresh_fig_heading_2', n_clicks=0, 
                                        style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'}),
                            html.Div([
                            plotly_for_correlation.put_down_correlation_plot(self.fig_heading, id='heading_plot', width='50%'),
                            plotly_for_correlation.put_down_correlation_plot(self.fig_heading_2, id='heading_plot_2', width='50%')
                                ], style=dict(display='flex')),    
                            html.Button('Refresh correlation plot on the right', id='refresh_fig_corr_2', n_clicks=0,
                                        style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'}),
                            html.Div([
                            plotly_for_correlation.put_down_correlation_plot(self.fig_corr, id='correlation_plot', width='50%'),
                            plotly_for_correlation.put_down_correlation_plot(self.fig_corr_2, id='correlation_plot_2', width='50%')
                                ], style=dict(display='flex')),                                                            
                            ])


    def make_dash_for_comparison(self):
 

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        self.fig_corr = plotly_for_correlation.make_correlation_plot_in_plotly(**self.kwargs_for_correlation_plot)

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        app.layout = self.prepare_dash_for_comparison_layout()
        

        self.make_function_to_update_all_plots_based_on_new_info(app)
        self.make_function_to_refresh_fig_corr_2(app)
        self.make_function_to_refresh_fig_heading_2(app)

        app.run(debug=True, port=8055)




    def make_function_to_update_all_plots_based_on_new_info(self, app):
        
        @app.callback(
            Output('correlation_plot', 'figure'),
            Output('correlation_plot_2', 'figure'),
            Output('heading_plot', 'figure'),
            Output('heading_plot_2', 'figure'),
            State('curv_of_traj_mode', 'value'),
            State("window_lower_end", "value"),
            State("window_upper_end", "value"),
            State('ref_point_mode', 'value'),
            State('ref_point_value', 'value'),
            Input('update_curv_of_traj', 'n_clicks'),
            Input('update_ref_point', 'n_clicks'),
            prevent_initial_call=True)
        
        def update_correlation_plot(curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end, 
                                    ref_point_mode, ref_point_value, update_curv_of_traj, update_ref_point):


            if (ctx.triggered[0]['prop_id'] == 'update_curv_of_traj.nclicks'):
                if (curv_of_traj_lower_end < curv_of_traj_upper_end) or (curv_of_traj_mode == 'now to stop'):
                    print('update_correlation_plot_basesd_on_window: [', curv_of_traj_lower_end, curv_of_traj_upper_end, ']')
                    self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self.update_dash_comparison_based_on_curv_of_traj_df(curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end)
                else:
                    print('Warning: curv_of_traj_lower_end is larger than curv_of_traj_upper_end, so no update is made')
                    raise PreventUpdate                

                             
            elif (ctx.triggered[0]['prop_id'] == 'update_ref_point.n_clicks'):
                if ref_point_value is not None:        
                    self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._update_dash_based_on_new_ref_point_descr(ref_point_mode, ref_point_value)
                else:
                    raise PreventUpdate
                
            else:
                raise PreventUpdate
            
            return self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2
        return
    


    def make_function_to_refresh_fig_corr_2(self, app):
        
        @app.callback(
            Output('correlation_plot_2', 'figure', allow_duplicate=True),
            Input('refresh_fig_corr_2', 'n_clicks'),
            prevent_initial_call=True)
        
        def update_correlation_plot(n_clicks):
            #print('I will update fig_corr_2')
            self.fig_corr_2 = self._make_fig_corr_2()
            return self.fig_corr_2
                


    def make_function_to_refresh_fig_heading_2(self, app):
        
        @app.callback(
            Output('heading_plot_2', 'figure', allow_duplicate=True),
            Input('refresh_fig_heading_2', 'n_clicks'),
            prevent_initial_call=True)
        
        def update_correlation_plot(n_clicks):
            #print('I will update fig_heading_2')
            self.fig_heading_2 = self._make_fig_heading_2()
            return self.fig_heading_2
        


    #===================================================================================================
    # below are helper functions 

    def update_dash_comparison_based_on_curv_of_traj_df(self, curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end):
        self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
        self.curv_of_traj_params['window_for_curv_of_traj'] = [curv_of_traj_lower_end, curv_of_traj_upper_end]
        self._rerun_after_changing_curv_of_traj_params()
        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        return self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2



    def _update_dash_based_on_new_ref_point_descr(self, ref_point_mode, ref_point_value):
        self._update_dash_based_on_new_ref_point_descr(ref_point_mode, ref_point_value)
        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        print('update all plots based on new reference point description: ', self.ref_point_descr, '. Note: it might take a few seconds to update the plots.')
        return self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2


    def _make_kwargs_for_correlation_plot(self):
        self.kwargs_for_correlation_plot = super()._make_kwargs_for_correlation_plot()
        self.kwargs_for_correlation_plot['current_stop_point_index_to_mark'] = None
        return self.kwargs_for_correlation_plot
    

    def _make_kwargs_for_heading_plot(self):

        self.kwargs_for_heading_plot = super()._make_kwargs_for_heading_plot()
        self.kwargs_for_heading_plot['current_stop_point_index_to_mark'] = None
        return self.kwargs_for_heading_plot