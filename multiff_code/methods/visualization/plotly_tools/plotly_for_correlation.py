
from multiprocessing import Value
import sys
from null_behaviors import curv_of_traj_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
import os
import sys
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

import numpy as np

def make_correlation_plot_in_plotly(heading_info_df=None,
                                    x_var_column='cum_distance_between_two_stops',
                                    y_var_column='diff_in_abs_angle_to_nxt_ff',
                                    current_stop_point_index_to_mark=None, traj_curv_descr='', ref_point_descr='',
                                    title=None, **kwargs):

    if heading_info_df is None:
        print("Warning: heading_info_df is None, cannot create plot.")
        return None

    # Extract x and y variables, raise error if columns missing
    try:
        x_var = heading_info_df[x_var_column].values
        y_var = heading_info_df[y_var_column].values
    except KeyError as e:
        print(f"Warning: Missing column in dataframe: {e}")
        return None

    # Check for empty arrays
    if len(x_var) == 0 or len(y_var) == 0:
        print('Warning: Data arrays are empty, so correlation plot is not shown')
        return None

    # Find index to mark on plot
    current_position_index_to_mark = None
    if current_stop_point_index_to_mark is not None:
        try:
            matches = heading_info_df.index[
                heading_info_df['stop_point_index'] == current_stop_point_index_to_mark
            ]
            current_position_index_to_mark = matches[0]
        except:
            raise ValueError(f'Warning: No match found for current_stop_point_index_to_mark: {current_stop_point_index_to_mark}')


    # Compose title if none given
    if title is None:
        title = traj_curv_descr + "<br>" + ref_point_descr

    # Prepare customdata for hover info
    if 'stop_point_index' in heading_info_df.columns:
        customdata = heading_info_df['stop_point_index'].values
    else:
        customdata = None

    # Plotly hovertemplate formatting (Plotly uses %{x} and %{y}, not Python f-string formatting)
    hovertemplate = (
        f'<b>{x_var_column}: %{{x:.2f}} <br>{y_var_column}: %{{y:.2f}}</b><br><br>'
        'Stop point index:<br>%{{customdata}}'
        '<extra></extra>'
    )

    # Axis titles
    xaxis_title = x_var_column
    yaxis_title = y_var_column

    # Call your plotting function (make sure plot_relationship_in_plotly is defined elsewhere)
    fig_corr = plot_scatter_plot_in_plotly(x_var, y_var, customdata, hovertemplate, current_position_index_to_mark, title, xaxis_title, yaxis_title)

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

    # find rows in heading_info_df that contains na in angle_from_stop_to_nxt_ff or angle_opt_cur_end_to_nxt_ff
    original_length = len(heading_info_df)
    heading_info_df = heading_info_df.dropna(
        subset=['angle_from_stop_to_nxt_ff', 'angle_opt_cur_end_to_nxt_ff'])
    new_length = len(heading_info_df)
    if original_length != new_length:
        add_to_title = '# nan removed: ' + \
            str(original_length-new_length) + ' out of ' + str(original_length)
    else:
        add_to_title = ''

    # Extract relevant angles from heading_info_df
    ang_stop_nxt = heading_info_df['angle_from_stop_to_nxt_ff'].values
    ang_cur_nxt = heading_info_df['angle_opt_cur_end_to_nxt_ff'].values

    # Convert angles to degrees if required
    if change_units_to_degrees:
        ang_stop_nxt = ang_stop_nxt * (180/np.pi)
        ang_cur_nxt = ang_cur_nxt * (180/np.pi)

    # Reset index of heading_info_df
    heading_info_df = heading_info_df.reset_index()

    # Find the index of the current stop point to mark if provided
    current_position_index_to_mark = None
    if current_stop_point_index_to_mark is not None:
        try:
            current_position_index_to_mark = heading_info_df[heading_info_df['stop_point_index']
                                                             == current_stop_point_index_to_mark].index.values[0]
        except:
            pass

    # Set default title if not provided
    if title is None:
        title = "Angle from Stop to Next FF vs. Angle from Current Arc End to Next FF" + "<br>" + ref_point_descr

    # Prepare custom data and hover template for the plot
    customdata = heading_info_df['stop_point_index'].values
    hovertemplate = ('<b>Angle from Stop to Next FF: %{x:.2f} <br>Angle from Current Arc End to Next FF: %{y:.2}</b><BR><BR>Stop point index:<BR>' +
                     '%{customdata}' + '<extra></extra>')

    # Set axis titles
    xaxis_title = 'Angle from Stop to Nxt FF'
    yaxis_title = 'Angle from Cur Arc End to Nxt FF'

    # Generate the plot
    fig_angle = plot_relationship_in_plotly(ang_stop_nxt, ang_cur_nxt, show_plot=False, title=title, current_position_index_to_mark=current_position_index_to_mark,
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
                    figure=fig_corr,
                    # ['Original-Present', 0]}]}
                    hoverData={'points': [{'customdata': 0}]},
                    clickData={'points': [{'customdata': 0}]}
                    ),
        # 'display': 'inline-block',
    ], style={'width': width, 'padding': '0 0 0 0',
              })


def find_new_curv_of_traj_counted(point_index_for_curv_of_traj_df, monkey_information, ff_caught_T_new, curv_of_traj_mode, lower_end, upper_end, truncate_curv_of_traj_by_time_of_capture=False):
    if (lower_end is not None) & (upper_end is not None):
        if curv_of_traj_mode == 'time':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_time_window(
                point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        elif curv_of_traj_mode == 'distance':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_distance_window(
                point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            raise PreventUpdate(
                "No update was made because curv_of_traj_mode is not recognized.")
        new_curv_of_traj_counted = new_curv_of_traj_df['curv_of_traj'].values
        return new_curv_of_traj_counted
    else:
        raise PreventUpdate(
            "No update was made because lower_end or upper_end is None.")


def find_curv_of_traj_counted_from_curv_of_traj_df(curv_of_traj_df, point_index_for_curv_of_traj_df):
    curv_of_traj_df = curv_of_traj_df.set_index('point_index')
    curv_of_traj_counted = curv_of_traj_df.loc[point_index_for_curv_of_traj_df,
                                               'curv_of_traj'].values
    return curv_of_traj_counted


def plot_relationship_in_plotly(x_array, y_array, slope=None, show_plot=True,
                                title="Traj Curv: From Current Point to Right Before Stop <br> At -1 Sec",
                                xaxis_title='Traj Curv - Curv to cur ff (cm)',
                                yaxis_title='Curv to nxt ff - Curv to cur ff (cm)',
                                customdata=None,
                                hovertemplate=None,
                                current_position_index_to_mark=None,
                                add_to_title=''):

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x_array, y_array)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_array,
                             y=y_array,
                             mode='markers',
                             showlegend=False,
                             customdata=customdata,
                             hovertemplate=hovertemplate,
                             #     hovertemplate=('<b>Traj curv - cur ff curv: %{x:.2f} <br>Alt curv - cur ff curv: %{y:.2}</b><BR><BR>Current Position in Data:<BR>' +
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
            text=title + "<br><sup>" + 'r value =' +
            str(round(r_value, 2)) + ', slope =' +
            str(round(slope, 2)) + "<br>" + add_to_title + "</sup>",
            xref="paper",
            x=0
        ),
    )
    # make sure the x and y axis have the same scale

    if (current_position_index_to_mark is not None):
        fig.add_trace(go.Scatter(x=[x_array[current_position_index_to_mark]],
                                 y=[y_array[current_position_index_to_mark]],
                                 mode='markers', marker=dict(color='red', size=10),
                                 customdata=[
                                     customdata[current_position_index_to_mark]],
                                 hovertemplate=hovertemplate,
                                 name='Stop Point Index',))

    if show_plot:
        fig.show()
    return fig


def plot_scatter_plot_in_plotly(x_var, y_var, customdata, hovertemplate, current_position_index_to_mark, title=None, xaxis_title=None, yaxis_title=None):
    # Build the scatter plot figure directly
    fig_corr = go.Figure()

    fig_corr.add_trace(
        go.Scatter(
            x=x_var,
            y=y_var,
            mode='markers',
            marker=dict(size=7, color='blue'),
            customdata=customdata,
            hovertemplate=hovertemplate
        )
    )

    # If you want to mark the current position with a special symbol/color
    if current_position_index_to_mark is not None and 0 <= current_position_index_to_mark < len(x_var):
        fig_corr.add_trace(
            go.Scatter(
                x=[x_var[current_position_index_to_mark]],
                y=[y_var[current_position_index_to_mark]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Current Stop',
                showlegend=True,
                hoverinfo='skip'
            )
        )

    fig_corr.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_font_size=13,
        showlegend=True,
    )

