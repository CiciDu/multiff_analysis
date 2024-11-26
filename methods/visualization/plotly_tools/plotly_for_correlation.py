
from multiprocessing import Value
import sys
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import html, dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import matplotlib
from scipy import stats

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





def make_correlation_plot_in_plotly(curv_for_correlation_df=None, change_units_to_degrees_per_m=True, 
                                    current_stop_point_index_to_mark=None, traj_curv_descr='', ref_point_descr='', 
                                    title=None, **kwargs):

    traj_curv_counted = curv_for_correlation_df['traj_curv_counted'].values
    alt_curv_counted = curv_for_correlation_df['alt_curv_counted'].values
    curv_for_correlation_df = curv_for_correlation_df.reset_index()
    current_position_index_to_mark = None
    if current_stop_point_index_to_mark is not None:
        try:
            current_position_index_to_mark = curv_for_correlation_df[curv_for_correlation_df['stop_point_index'] == current_stop_point_index_to_mark].index.values[0]
        except:
            pass
        

    if traj_curv_counted is None or alt_curv_counted is None:
        print('Warning: alt_ff_counted_df or stop_ff_counted_df is None, so correlation plot is not shown')
        return None
        
    traj_curv_counted = traj_curv_counted.copy()
    alt_curv_counted = alt_curv_counted.copy()
    

    if change_units_to_degrees_per_m:
        traj_curv_counted = traj_curv_counted * (180/np.pi) * 100
        alt_curv_counted = alt_curv_counted * (180/np.pi) * 100

    if title is None:
        title = traj_curv_descr + "<br>" + ref_point_descr
    customdata = curv_for_correlation_df['stop_point_index'].values
    hovertemplate = ('<b>Alt ff curv - Stop ff curv: %{x:.2f} <br>Traj curv - Stop ff curv: %{y:.2f}</b><BR><BR>Stop point index:<BR>' +
                           '%{customdata}' +
                           '<extra></extra>')
    xaxis_title='Curv to Alt ff - Curv to Stop ff (cm)'
    yaxis_title='Traj Curv - Curv to Stop ff (cm)'
    fig_corr = plot_relationship_in_plotly(alt_curv_counted, traj_curv_counted, show_plot=False, title=title, current_position_index_to_mark=current_position_index_to_mark,
                                           hovertemplate=hovertemplate, customdata=customdata, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig_corr.update_layout(title_font_size=13, showlegend=False)


    return fig_corr


# def make_heading_plot_in_plotly(rel_heading_df=None, change_units_to_degrees=True,
#                                 current_stop_point_index_to_mark=None, ref_point_descr='', traj_curv_descr='',
#                                 title=None, **kwargs):
    
#     rel_heading_traj = rel_heading_df['rel_heading_traj'].values
#     rel_heading_alt = rel_heading_df['rel_heading_alt'].values
#     if change_units_to_degrees:
#         rel_heading_traj = rel_heading_traj * (180/np.pi) 
#         rel_heading_alt = rel_heading_alt * (180/np.pi)
#     rel_heading_df = rel_heading_df.reset_index()
#     if current_stop_point_index_to_mark is not None:
#         current_position_index_to_mark = rel_heading_df[rel_heading_df['stop_point_index'] == current_stop_point_index_to_mark].index.values[0]
#     else:
#         current_position_index_to_mark = None
        
#     if title is None:
#         title = 'Relative difference in monkey heading' + "<br>" + traj_curv_descr + "<br>" + ref_point_descr
#     #title = 'Relative difference in monkey heading' + "<br>" + ref_point_descr
#     customdata = rel_heading_df['stop_point_index'].values
#     hovertemplate=('<b>Alt heading - Stop heading: %{x:.2f} <br>Traj heading - Stop heading: %{y:.2}</b><BR><BR>Stop point index:<BR>' +
#                                     '%{customdata}' + '<extra></extra>')
#     xaxis_title='Alt - Stop (deg)'
#     yaxis_title='Traj - Stop (deg)'
#     fig_heading = plot_relationship_in_plotly(rel_heading_alt, rel_heading_traj, show_plot=False, title=title, current_position_index_to_mark=current_position_index_to_mark,
#                                            hovertemplate=hovertemplate, customdata=customdata, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
#     fig_heading.update_layout(title_font_size=13, showlegend=False)
#     return fig_heading 






def make_heading_plot_in_plotly(heading_info_df=None, change_units_to_degrees=True,
                                  current_stop_point_index_to_mark=None, ref_point_descr='', traj_curv_descr='',
                                  title=None, **kwargs):

    # find rows in heading_info_df that contains na in angle_from_m_before_stop_to_alt_ff or angle_from_stop_ff_landing_to_alt_ff
    original_length = len(heading_info_df)
    heading_info_df = heading_info_df.dropna(subset=['angle_from_m_before_stop_to_alt_ff', 'angle_from_stop_ff_landing_to_alt_ff'])
    new_length = len(heading_info_df)
    if original_length != new_length:
        add_to_title = '# nan removed: ' + str(original_length-new_length) + ' out of ' + str(original_length)
    else:
        add_to_title = ''


    # Extract relevant angles from heading_info_df
    alt_traj = heading_info_df['angle_from_m_before_stop_to_alt_ff'].values
    alt_stop = heading_info_df['angle_from_stop_ff_landing_to_alt_ff'].values


    # Convert angles to degrees if required
    if change_units_to_degrees:
        alt_traj = alt_traj * (180/np.pi) 
        alt_stop = alt_stop * (180/np.pi)

    # Reset index of heading_info_df
    heading_info_df = heading_info_df.reset_index()

    # Find the index of the current stop point to mark if provided
    current_position_index_to_mark = None
    if current_stop_point_index_to_mark is not None:
        try:
            current_position_index_to_mark = heading_info_df[heading_info_df['stop_point_index'] == current_stop_point_index_to_mark].index.values[0]
        except:
            pass
        

    # Set default title if not provided
    if title is None:
        title = "Angle to Alt FF from Traj vs From stop FF" + "<br>" + ref_point_descr

    # Prepare custom data and hover template for the plot
    customdata = heading_info_df['stop_point_index'].values
    hovertemplate=('<b>Angle to Alt FF from Traj: %{x:.2f} <br>Angle to Alt FF from Stop FF: %{y:.2}</b><BR><BR>Stop point index:<BR>' +
                                    '%{customdata}' + '<extra></extra>')

    # Set axis titles
    xaxis_title = 'Alt from Traj'
    yaxis_title = 'Alt from Stop'

    # Generate the plot
    fig_angle = plot_relationship_in_plotly(alt_traj, alt_stop, show_plot=False, title=title, current_position_index_to_mark=current_position_index_to_mark,
                                            hovertemplate=hovertemplate, customdata=customdata, xaxis_title=xaxis_title, yaxis_title=yaxis_title, add_to_title=add_to_title)

    # Update layout of the plot
    fig_angle.update_layout(title_font_size=13, showlegend=False)

    return fig_angle




def put_down_correlation_plot(fig_corr, id='correlation_plot', width='60%'):

    if id is None: 
        id = 'correlation_plot'

    return html.Div([
                dcc.Graph(
                    id=id,
                    figure = fig_corr,
                    hoverData={'points': [{'customdata': 0}]}, #['Original-Present', 0]}]}
                    clickData={'points': [{'customdata': 0}]}
                ),
            ], style={'width': width, 'padding': '0 0 0 0', #'display': 'inline-block',
                    })



def find_new_curv_of_traj_counted(point_index_for_curv_of_traj_df, monkey_information, ff_caught_T_new, curv_of_traj_mode, lower_end, upper_end, truncate_curv_of_traj_by_time_of_capture=False):
    if (lower_end is not None) & (upper_end is not None):
        if curv_of_traj_mode == 'time':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_time_window(point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        elif curv_of_traj_mode == 'distance':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_distance_window(point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            raise PreventUpdate
        new_curv_of_traj_counted = new_curv_of_traj_df['curvature_of_traj'].values
        return new_curv_of_traj_counted
    else:
        raise PreventUpdate




def find_curv_of_traj_counted_from_curv_of_traj_df(curv_of_traj_df, point_index_for_curv_of_traj_df):
    curv_of_traj_df = curv_of_traj_df.set_index('point_index')
    curv_of_traj_counted = curv_of_traj_df.loc[point_index_for_curv_of_traj_df, 'curvature_of_traj'].values
    return curv_of_traj_counted









def plot_relationship_in_plotly(x_array, y_array, slope=None, show_plot=True,
                                title="Traj Curv: From Current Point to Right Before Stop <br> At -1 Sec",
                                xaxis_title='Traj Curv - Curv to Stop ff (cm)',
                                yaxis_title='Curv to Alt ff - Curv to Stop ff (cm)',
                                customdata=None, 
                                hovertemplate=None,
                                current_position_index_to_mark=None,
                                add_to_title=''):


    slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_array, 
                             y=y_array, 
                             mode='markers',
                             showlegend=False,
                             customdata=customdata,
                             hovertemplate=hovertemplate,
                        #     hovertemplate=('<b>Traj curv - Stop ff curv: %{x:.2f} <br>Alt curv - Stop ff curv: %{y:.2}</b><BR><BR>Current Position in Data:<BR>' +
                        #    '%{customdata}' +
                        #    '<extra></extra>'),
                           ))
    # calculate and plot linear correlation
    x_min = min(x_array)
    x_max = max(x_array)
    fig.add_trace(go.Scatter(x=np.array([x_min, x_max]), 
                             y=np.array([x_min, x_max])*slope+intercept,
                             showlegend=False,
                             mode='lines', 
                             line=dict(color='red')))
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=go.layout.Title(
        text=title + "<br><sup>" + 'r value =' + str(round(r_value, 2)) + ', slope =' + str(round(slope, 2)) + "<br>" + add_to_title + "</sup>",
        xref="paper",
        x=0
    ),
        )
    # make sure the x and y axis have the same scale


    if (current_position_index_to_mark is not None):
        fig.add_trace(go.Scatter(x=[x_array[current_position_index_to_mark]], 
                                      y=[y_array[current_position_index_to_mark]], 
                                      mode='markers', marker=dict(color='red', size=10),
                                      customdata=[customdata[current_position_index_to_mark]],
                                      hovertemplate=hovertemplate,
                                      name='Stop Point Index',))
        
    if show_plot:
        fig.show()
    return fig
