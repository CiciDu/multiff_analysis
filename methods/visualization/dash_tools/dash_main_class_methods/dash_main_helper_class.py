
import sys
from visualization.plotly_tools import plotly_for_correlation, plotly_preparation, plotly_for_scatterplot
from visualization.dash_tools import dash_prep_class, dash_utils, dash_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils, plot_monkey_heading_helper_class
from visualization import monkey_heading_functions
from visualization.plotly_tools import plotly_for_monkey

import os
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


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


# https://dash.plotly.com/interactive-graphing




class DashMainHelper(dash_prep_class.DashCartesianPreparation):

    
    def __init__(self, 
                 raw_data_folder_path=None):
        
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        

    def prepare_to_make_dash_for_main_plots(self,
                                            ref_point_params={},
                                            curv_of_traj_params={},
                                            overall_params={},
                                            monkey_plot_params={},
                                            scatter_plot_params={},
                                            stops_near_ff_df_exists_ok=True,
                                            heading_info_df_exists_ok=False,
                                            test_or_control='test'):

        self.ref_point_params = ref_point_params
        self.curv_of_traj_params = curv_of_traj_params
        self.overall_params = overall_params
        self.monkey_plot_params = monkey_plot_params
        self.scatter_plot_params = scatter_plot_params


        self.snf_streamline_organizing_info_kwargs = find_stops_near_ff_utils.organize_snf_streamline_organizing_info_kwargs(ref_point_params, curv_of_traj_params, overall_params)
        super().streamline_organizing_info(**self.snf_streamline_organizing_info_kwargs, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, 
                                           heading_info_df_exists_ok=heading_info_df_exists_ok, test_or_control=test_or_control)
        
        # we use the first instance in stops_near_ff_df_counted to plot for now.
        self.stops_near_ff_row = self.stops_near_ff_df_counted.iloc[0]
        self._prepare_static_main_plots()


    def _put_down_checklist_for_all_plots(self, id_prefix=None):
        checklist_options = [{'label': 'show heading instead of curv', 'value': 'heading_instead_of_curv'},
                            {'label': 'truncate curv of traj by time of capture', 'value': 'truncate_curv_of_traj_by_time_of_capture'},
                            {'label': 'eliminate outliers', 'value': 'eliminate_outliers'},
                            {'label': 'use curvature to ff center', 'value': 'use_curvature_to_ff_center'}]
        
        checklist_params = ['heading_instead_of_curv', 'eliminate_outliers', 'use_curvature_to_ff_center']
        checklist_values = [key for key in checklist_params if self.overall_params[key] is True]
        if self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] is True:
            checklist_params.append('truncate_curv_of_traj_by_time_of_capture')
                                
        return html.Div([dcc.Checklist(options=checklist_options,
                                        value=checklist_values,
                                        id=id_prefix+'checklist_for_all_plots',
                                        style={'width': '50%', 'background-color': '#F9F99A', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'})])



    def _put_down_checklist_for_monkey_plot(self, id_prefix=None):
        checklist_options = [{'label': 'show visible fireflies', 'value': 'show_visible_fireflies'},
                            {'label': 'show fireflies in memory', 'value': 'show_in_memory_fireflies'},
                            {'label': 'show visible segments of ff', 'value': 'show_visible_segments'},                            
                            {'label': 'show monkey heading', 'value': 'show_monkey_heading'},
                            {'label': 'show portion used to calc traj curvature', 'value': 'show_traj_portion'},
                            {'label': 'show null trajectory of ff', 'value': 'show_null_arcs_to_ff'},
                            {'label': 'show stops', 'value': 'show_stops'},
                            {'label': 'show all eye positions', 'value': 'show_all_eye_positions'},
                            {'label': 'show current eye positions', 'value': 'show_current_eye_positions'},
                            {'label': 'show eye positions for both eyes', 'value': 'show_eye_positions_for_both_eyes'}]
        
        checklist_params = ['show_visible_fireflies', 'show_in_memory_fireflies', 'show_monkey_heading', 'show_visible_segments', 'show_null_arcs_to_ff', 'show_stops', 'show_all_eye_positions', 'show_current_eye_positions', 'show_eye_positions_for_both_eyes']
        checklist_values = [key for key in checklist_params if self.monkey_plot_params[key] is True]

                                
        return html.Div([dcc.Checklist(options=checklist_options,
                                value=checklist_values,
                                id=id_prefix+'checklist_for_monkey_plot',
                                style={'width': '50%', 'background-color': '#ADD8E6', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'})])




    def _put_down_the_menus_on_top(self, id_prefix=''):
        return html.Div([
                        dash_utils.put_down_the_dropdown_menu_for_ref_point_mode(self.ref_point_params['ref_point_mode'], id=id_prefix+'ref_point_mode'),
                        dash_utils.put_down_the_input_for_ref_point_descr(self.ref_point_params['ref_point_value'], id=id_prefix+"ref_point_value"),
                        dash_utils.put_down_the_dropdown_menu_for_curv_of_traj_mode(self.curv_of_traj_params['curv_of_traj_mode'], id=id_prefix+'curv_of_traj_mode'),
                        dash_utils.put_down_the_input_for_window_lower_end_and_upper_end(self.curv_of_traj_params['window_for_curv_of_traj'], ids=[id_prefix+'window_lower_end', id_prefix+'window_upper_end']),
                        ],
                        style=dict(display='flex'))
    


    def _put_down_correlation_plots_in_dash(self, id_prefix=''):
        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        self.fig_corr_or_heading, self.fig_corr_or_heading_2 = self._determine_fig_corr_or_heading()

        plot_layout = [plotly_for_correlation.put_down_correlation_plot(self.fig_corr_or_heading, id=id_prefix+'correlation_plot', width='50%')]

        if self.show_shuffled_correlation_plot:
            plot_layout.append(plotly_for_correlation.put_down_correlation_plot(self.fig_corr_or_heading_2, id=id_prefix+'correlation_plot_2', width='50%'))
            overall_layout = [html.Button('Refresh shuffled plot on the right', id=id_prefix+'refresh_correlation_plot_2', n_clicks=0,
                              style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'})]            
        else:
            self.fig_corr_or_heading_2 = go.Figure()
            self.fig_corr_or_heading_2.update_layout(height=10, width=10)
            plot_layout.append(dash_utils.put_down_empty_plot_that_takes_no_space(id=id_prefix+'correlation_plot_2'))
            overall_layout = []

        overall_layout.append(html.Div(plot_layout, style=dict(display='flex')))
        correlation_plots_in_dash = html.Div(overall_layout)
        return correlation_plots_in_dash


    def _plot_eye_positions_for_dash(self, fig, show_eye_positions_for_both_eyes=False, point_index_to_show_traj_curv=None, x0=0, y0=0,
                                    trace_name='eye_positions', update_if_already_exist=True, marker_size=6, use_arrow_to_show_eye_positions=True):

        if use_arrow_to_show_eye_positions:
            # clear existing annotations first
            fig['layout']['annotations'] = []

        if not show_eye_positions_for_both_eyes:
            monkey_subset2 = self.monkey_subset.copy()
            if point_index_to_show_traj_curv is not None:
                try:
                    monkey_subset2 = self.monkey_subset.loc[[point_index_to_show_traj_curv]].copy()
                except:
                    return fig
            fig = plotly_for_monkey.show_eye_positions_using_either_marker_or_arrow(fig, x0, y0, monkey_subset2, trace_name='eye_positions', update_if_already_exist=update_if_already_exist, 
                                                                marker='circle', marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions)
        else:
            for left_or_right, marker, trace_name, arrowcolor in [('left', 'triangle-left', trace_name + '_left', 'purple'), ('right', 'triangle-right', trace_name + '_right', 'orange')]:
                monkey_subset = self.both_eyes_info['monkey_subset'][left_or_right]
                monkey_subset2 = self.monkey_subset.copy()
                if point_index_to_show_traj_curv is not None:
                    try:
                        monkey_subset2 = monkey_subset.loc[[point_index_to_show_traj_curv]].copy()
                    except:
                        return fig
                fig = plotly_for_monkey.show_eye_positions_using_either_marker_or_arrow(fig, x0, y0, monkey_subset2, trace_name=trace_name, update_if_already_exist=update_if_already_exist, 
                                                                    marker=marker, marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions, arrowcolor=arrowcolor)
        return fig
    

    def _update_dash_based_on_checklist_for_monkey_plot(self, checklist_for_monkey_plot):

        old_checklist_params = {'show_visible_segments': self.monkey_plot_params['show_visible_segments'],
                                'show_visible_fireflies': self.monkey_plot_params['show_visible_fireflies'],
                                'show_in_memory_fireflies': self.monkey_plot_params['show_in_memory_fireflies'],
                                'show_null_arcs_to_ff': self.monkey_plot_params['show_null_arcs_to_ff'],
                                }

        for param in ['show_monkey_heading', 'show_visible_segments', 'show_traj_portion', 'show_null_arcs_to_ff', 'show_stops', 'show_all_eye_positions', 'show_current_eye_positions',
                      'show_eye_positions_for_both_eyes', 'show_visible_fireflies', 'show_in_memory_fireflies']:
            if param in checklist_for_monkey_plot:
                self.monkey_plot_params[param] = True
            else:
                self.monkey_plot_params[param] = False

        if (self.monkey_plot_params['show_null_arcs_to_ff'] != old_checklist_params['show_null_arcs_to_ff']):
            self.find_null_arcs_info_for_plotting_for_the_duration()
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper._find_mheading_and_triangle_df_for_null_arcs_for_the_duration(self)

        for param in ['show_visible_segments', 'show_visible_fireflies', 'show_in_memory_fireflies']:
            if old_checklist_params[param] != self.monkey_plot_params[param]:
                self.current_plotly_plot_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(self.stops_near_ff_row, self.PlotTrials_args, self.monkey_plot_params)
                                                                                                                
                self._produce_initial_plots()
                break
        else:                                                                                                                                                                                                 
            self._produce_fig_for_dash()
        # self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(self.stop_point_index)
        # self.fig, self.fig_scatter_combd, self.fig_scatter_natural_y_range = self._produce_initial_plots()

        return self.fig, self.fig_scatter_combd


    def _update_dash_based_on_checklist_for_all_plots(self, checklist_for_all_plots):
        # keep a copy of old checklist_params
        old_checklist_params = {'heading_instead_of_curv': self.overall_params['heading_instead_of_curv'], # We update based on this variable elsewhere
                                'truncate_curv_of_traj_by_time_of_capture': self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'],
                                'eliminate_outliers': self.overall_params['eliminate_outliers'],
                                'use_curvature_to_ff_center': self.overall_params['use_curvature_to_ff_center'],
                                }

        # update checklist_params into the instance
        if 'truncate_curv_of_traj_by_time_of_capture' in checklist_for_all_plots:
            self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = True
        else:
            self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = False

                
        for param in ['eliminate_outliers', 'use_curvature_to_ff_center', 'heading_instead_of_curv']:
            if param in checklist_for_all_plots:
                self.overall_params[param] = True
            else:
                self.overall_params[param] = False

        # update the plots based on the new checklist_params
        if ((self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] != old_checklist_params['truncate_curv_of_traj_by_time_of_capture'])
            or (self.overall_params['use_curvature_to_ff_center'] != old_checklist_params['use_curvature_to_ff_center'])):
            self. _rerun_after_changing_curv_of_traj_params()
            self._prepare_static_main_plots()
        elif self.overall_params['eliminate_outliers'] != old_checklist_params['eliminate_outliers']:
            self._rerun_after_changing_eliminate_outliers()
            self._prepare_static_main_plots()
        elif (self.overall_params['heading_instead_of_curv'] != old_checklist_params['heading_instead_of_curv']):
            self.fig_corr_or_heading, self.fig_corr_or_heading_2 = self._determine_fig_corr_or_heading()
        else:
            raise PreventUpdate

        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading, self.fig_corr_or_heading_2




    def _update_dash_based_on_monkey_hover_data(self, monkey_hoverdata):

        trace_index = monkey_hoverdata['points'][0]['curveNumber']
        if not ((trace_index == self.trajectory_data_trace_index) or (trace_index == self.traj_portion_trace_index)):
            raise PreventUpdate

        monkey_hoverdata_value = monkey_hoverdata['points'][0]['customdata']

        if (not isinstance(monkey_hoverdata_value, int)) & (not isinstance(monkey_hoverdata_value, float)):
            # if monkey_hoverdata_value is a list
            try:
                monkey_hoverdata_value = monkey_hoverdata_value[0]
            except TypeError:
                raise PreventUpdate
        if monkey_hoverdata_value is None:
            raise PreventUpdate

        ONLY_UPDATE_EYE_POSITION = False
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_scatterplot.find_monkey_hoverdata_value_for_both_fig_scatter(self.hoverdata_column, monkey_hoverdata_value, self.current_plotly_plot_key_comp['trajectory_df'])
        if self.curv_of_traj_params['curv_of_traj_mode'] == 'now to stop':
            if (self.monkey_hoverdata_value_s >= self.hoverdata_value_upper_bound_s) or (self.monkey_hoverdata_value_cm >= self.hoverdata_value_upper_bound_cm):
                ONLY_UPDATE_EYE_POSITION = True
        self.monkey_hoverdata_value = monkey_hoverdata_value
        self._find_point_index_to_show_traj_curv()
        if not ONLY_UPDATE_EYE_POSITION:
            self.fig = self._update_fig_based_on_monkey_hover_data()

        if self.monkey_plot_params['show_current_eye_positions']:
            self.fig = self._update_eye_positions_based_on_monkey_hoverdata(self.point_index_to_show_traj_curv)

        if self.show_trajectory_scatter_plot:
            self.fig_scatter_combd = dash_utils.update_fig_scatter_combd_plot_based_on_monkey_hoverdata(self.fig_scatter_combd, self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm)
        
        return self.fig, self.fig_scatter_combd





    def _update_eye_positions_based_on_monkey_hoverdata(self, point_index_to_show_traj_curv):
        show_eye_positions_for_both_eyes = self.monkey_plot_params['show_eye_positions_for_both_eyes']
        self.fig = self._plot_eye_positions_for_dash(self.fig, point_index_to_show_traj_curv=point_index_to_show_traj_curv,
                                                    show_eye_positions_for_both_eyes=show_eye_positions_for_both_eyes, 
                                                    use_arrow_to_show_eye_positions=True, marker_size=15
                                                    )
        return self.fig
    



    def _update_dash_based_on_scatter_plot_hoverdata(self, scatter_plot_hoverdata):
        scatter_plot_hoverdata_values = scatter_plot_hoverdata['points'][0]['x']

        curveNumber = scatter_plot_hoverdata["points"][0]["curveNumber"]

        trace_name = self.fig_scatter_combd.data[curveNumber]['name']
        if 'scatter_cm_' in trace_name:
            x_column_name = 'rel_distance'
        else:
            x_column_name = 'rel_time'
        self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm = plotly_for_scatterplot.find_monkey_hoverdata_value_for_both_fig_scatter(x_column_name, scatter_plot_hoverdata_values, self.current_plotly_plot_key_comp['trajectory_df'])

        if self.hoverdata_column == 'rel_distance':
            self.monkey_hoverdata_value = self.monkey_hoverdata_value_cm
        else:
            self.monkey_hoverdata_value = self.monkey_hoverdata_value_s

        self.fig, self.fig_scatter_combd = self._update_fig_and_fig_scatter_based_on_monkey_hover_data()

        return self.fig, self.fig_scatter_combd
    


    def _update_dash_based_on_correlation_plot_clickdata(self, hoverData):
        self.stop_point_index = hoverData['points'][0]['customdata']
        self.stops_near_ff_row = self.stops_near_ff_df[self.stops_near_ff_df['stop_point_index']==self.stop_point_index].iloc[0]
        self.fig, self.fig_scatter_combd, self.fig_corr_or_heading = self._update_after_changing_stop_point_index()
        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading


    def _update_dash_after_clicking_previous_or_next_plot_button(self, previous_or_next='next'):
        rank_var = 'rank_by_angle_to_alt_ff' if self.overall_params['heading_instead_of_curv'] else 'rank_by_traj_curv'
        stops_near_ff_df = self.stops_near_ff_df if self.overall_params['heading_instead_of_curv'] else self.stops_near_ff_df_counted
        if previous_or_next == 'previous':
            self._get_previous_stops_near_ff_row(stops_near_ff_df, rank_var)
        elif previous_or_next == 'next':
            self._get_next_stops_near_ff_row(stops_near_ff_df, rank_var)
        else:
            raise ValueError('previous_or_next should be previous or next')
        self.fig, self.fig_scatter_combd, self.fig_corr_or_heading = self._update_after_changing_stop_point_index()
        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading


    def _get_previous_stops_near_ff_row(self, stops_near_ff_df, rank_var):
        old_rank = self.stops_near_ff_row[rank_var].item()
        if old_rank == stops_near_ff_df[rank_var].min():
            old_rank = stops_near_ff_df[rank_var].max()
        self.stops_near_ff_row = stops_near_ff_df[self.stops_near_ff_df[rank_var] == old_rank - 1].iloc[0]
        self.stop_point_index = self.stops_near_ff_row['stop_point_index']



    def _get_next_stops_near_ff_row(self, stops_near_ff_df, rank_var):
        old_rank = self.stops_near_ff_row[rank_var].item()
        if old_rank == stops_near_ff_df[rank_var].max():
            old_rank = stops_near_ff_df[rank_var].min()
        self.stops_near_ff_row = stops_near_ff_df[self.stops_near_ff_df[rank_var] == old_rank + 1].iloc[0]
        self.stop_point_index = self.stops_near_ff_row['stop_point_index']


    def _update_after_changing_stop_point_index(self):
        self._prepare_to_make_plotly_fig_for_dash_given_stop_point_index(self.stop_point_index)
        self.fig, self.fig_scatter_combd, self.fig_scatter_natural_y_range = self._produce_initial_plots()
        if not self.overall_params['heading_instead_of_curv']:
            self.kwargs_for_correlation_plot['current_stop_point_index_to_mark'] = self.stop_point_index
            self.fig_corr = plotly_for_correlation.make_correlation_plot_in_plotly(**self.kwargs_for_correlation_plot)
            self.fig_corr_or_heading = self.fig_corr
        else:
            self.kwargs_for_heading_plot['current_stop_point_index_to_mark'] = self.stop_point_index
            self.fig_heading = plotly_for_correlation.make_heading_plot_in_plotly(**self.kwargs_for_heading_plot)
            self.fig_corr_or_heading = self.fig_heading
        
        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading
    


    def _update_dash_based_on_curv_of_traj_df(self, curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end):
        self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
        self.curv_of_traj_params['window_for_curv_of_traj'] = [curv_of_traj_lower_end, curv_of_traj_upper_end]
        self._rerun_after_changing_curv_of_traj_params()
        self.fig_corr, self.fig_corr_2, self.fig_heading, self.fig_heading_2 = self._make_fig_corr_and_fig_heading_both_unshuffled_and_shuffled()
        self.fig_corr_or_heading, self.fig_corr_or_heading_2 = self._determine_fig_corr_or_heading()
        self._get_curv_of_traj_df_in_duration()
        self.fig = self._update_fig_based_on_curv_of_traj()
        self.fig_scatter_combd = plotly_for_scatterplot.add_new_curv_of_traj_to_fig_scatter_combd(self.fig_scatter_combd, self.curv_of_traj_df_in_duration, curv_of_traj_mode, curv_of_traj_lower_end, curv_of_traj_upper_end)
        self.fig_scatter_natural_y_range = plotly_for_scatterplot.update_fig_scatter_natural_y_range(self.fig_scatter_natural_y_range, self.curv_of_traj_df_in_duration, y_column_name='curv_of_traj_deg_over_cm')
        self._update_fig_scatter_combd_y_range()
        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading


    def _update_dash_based_on_new_ref_point_descr(self, ref_point_mode, ref_point_value):
        
        if ref_point_mode == 'distance':
            self.ref_point_descr = 'based on %d cm into past' % ref_point_value
            self.ref_point_column = 'rel_distance'
        elif ref_point_mode == 'time':
            self.ref_point_descr = 'based on %d seconds into past' % ref_point_value
            self.ref_point_column = 'rel_time'
        else:
            print('Warnings: ref_point_mode is not recognized, so no update is made')
            raise PreventUpdate
        
        self.ref_point_params['ref_point_mode'] = ref_point_mode
        self.ref_point_params['ref_point_value'] = ref_point_value
        self.snf_streamline_organizing_info_kwargs['ref_point_mode'] = ref_point_mode
        self.snf_streamline_organizing_info_kwargs['ref_point_value'] = ref_point_value

        self.streamline_organizing_info(**self.snf_streamline_organizing_info_kwargs)               
        if len(self.stops_near_ff_df_counted) == 0:
            print('Warning: After update, stops_near_ff_df_counted is empty! So no update is made')
            raise PreventUpdate
        self._prepare_static_main_plots()
        
        print('update all plots based on new reference point description: ', self.ref_point_descr, '. Note: it might take a few seconds to update the plots.')
        return self.fig, self.fig_scatter_combd, self.fig_corr_or_heading


    def _find_point_index_to_show_traj_curv(self):
        try:
            self.point_index_to_show_traj_curv = self.current_plotly_plot_key_comp['trajectory_df'].loc[self.current_plotly_plot_key_comp['trajectory_df'][self.hoverdata_column] >= self.monkey_hoverdata_value, 'point_index'].iloc[0]
        except IndexError:
            self.point_index_to_show_traj_curv = int(self.current_plotly_plot_key_comp['trajectory_df'].iloc[-1]['point_index'])  


    def _update_fig_and_fig_scatter_based_on_monkey_hover_data(self):
        self.fig_scatter_combd = dash_utils.update_fig_scatter_combd_plot_based_on_monkey_hoverdata(self.fig_scatter_combd, self.monkey_hoverdata_value_s, self.monkey_hoverdata_value_cm)
        self._find_point_index_to_show_traj_curv()
        self.fig = self._update_fig_based_on_monkey_hover_data()
        if self.monkey_plot_params['show_current_eye_positions']:
            self.fig = self._update_eye_positions_based_on_monkey_hoverdata(self.point_index_to_show_traj_curv)

        return self.fig, self.fig_scatter_combd
    

    def _update_fig_based_on_monkey_hover_data(self):
        # also update the monkey plot
        if self.monkey_plot_params['show_traj_portion']:
            self.traj_portion = self._find_traj_portion()
            self.fig = dash_utils.update_marked_traj_portion_in_monkey_plot(self.fig, self.traj_portion, hoverdata_multi_columns=self.hoverdata_multi_columns)
            
        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = self._update_null_arcs_for_stop_and_alt_ff_in_plotly()

        if self.monkey_plot_params['show_monkey_heading']:
            self.fig = self._update_all_monkey_heading_in_fig()

        return self.fig


    def _find_traj_portion(self):
        self.curv_of_traj_current_row = self.curv_of_traj_df[self.curv_of_traj_df['point_index']==self.point_index_to_show_traj_curv].copy()
        self.traj_portion, self.traj_length = plotly_preparation.find_traj_portion_for_traj_curv(self.current_plotly_plot_key_comp['trajectory_df'], self.curv_of_traj_current_row)
        return self.traj_portion
    
    def _update_fig_based_on_curv_of_traj(self):
        # also update monkey plot
        # point_index_to_mark = self.current_plotly_plot_key_comp['trajectory_df'].loc[self.current_plotly_plot_key_comp['trajectory_df'][self.hoverdata_column] >= self.monkey_hoverdata_value, 'point_index'].iloc[0]
        # curv_of_traj_current_row = self.curv_of_traj_df[self.curv_of_traj_df['point_index']==point_index_to_mark].copy()
        if self.monkey_plot_params['show_traj_portion']:
            self.traj_portion, self.traj_length = plotly_preparation.find_traj_portion_for_traj_curv(self.current_plotly_plot_key_comp['trajectory_df'], self.curv_of_traj_current_row)
            self.fig = dash_utils.update_marked_traj_portion_in_monkey_plot(self.fig, self.traj_portion, hoverdata_multi_columns=self.hoverdata_multi_columns)
        if self.monkey_plot_params['show_monkey_heading']:
            self.fig = monkey_heading_functions.update_monkey_heading_in_monkey_plot(self.fig, self.traj_triangle_df_in_duration, trace_name_prefix='monkey heading on trajectory', point_index=self.point_index_to_show_traj_curv)
        return self.fig


    def _determine_fig_corr_or_heading(self):
        if not self.overall_params['heading_instead_of_curv']:   
            self.fig_corr_or_heading = self.fig_corr
            self.fig_corr_or_heading_2 = self.fig_corr_2
        else:
            self.fig_corr_or_heading = self.fig_heading
            self.fig_corr_or_heading_2 = self.fig_heading_2
        return self.fig_corr_or_heading, self.fig_corr_or_heading_2
    

    def generate_other_messages(self, decimals=2):
        if 'heading_instead_of_curv' not in self.overall_params.keys():
            self.overall_params['heading_instead_of_curv'] = True
            print('Warning: overall_params does not have key heading_instead_of_curv. So we set it to True')

        if self.overall_params['heading_instead_of_curv']:
            current_alt_traj = round(self.heading_info_df.loc[self.heading_info_df['stop_point_index'] == self.stop_point_index, 'angle_from_m_before_stop_to_alt_ff'].item() * (180/np.pi), decimals) 
            current_alt_stop = round(self.heading_info_df.loc[self.heading_info_df['stop_point_index'] == self.stop_point_index, 'angle_from_stop_ff_landing_to_alt_ff'].item() * (180/np.pi), decimals)
            self.other_messages = f"Angle to Alt FF from Traj: {current_alt_traj}, \n     Angle to Alt FF from Stop FF: {current_alt_stop}"
        else:
            current_traj_curv = round(self.curv_for_correlation_df.loc[self.curv_for_correlation_df['stop_point_index'] == self.stop_point_index, 'traj_curv_counted'].item()* (180/np.pi) * 100, decimals)
            current_alt_curv = round(self.curv_for_correlation_df.loc[self.curv_for_correlation_df['stop_point_index'] == self.stop_point_index, 'alt_curv_counted'].item()* (180/np.pi) * 100, decimals)
            self.other_messages = f"Trajectory curvature: {current_traj_curv}, \n     Altitude curvature: {current_alt_curv}"
        
        # Add other info
        self.other_messages += ", \n Curv range: " + str(round(self.stops_near_ff_row['curv_range'], decimals)) + ", \n Cum distance between two stops: " + \
                                str(round(self.stops_near_ff_row['cum_distance_between_two_stops'], decimals))

        # Also get alt ff angle at ref point
        alt_ff_angle_at_ref_point = self.alt_ff_df2.loc[self.alt_ff_df2['stop_point_index'] == self.stop_point_index, 'ff_angle'].item() * (180/np.pi)
        self.other_messages += ", \n Alt FF angle at ref point: " + str(round(alt_ff_angle_at_ref_point, decimals))
        return self.other_messages
    
