import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import rc
from null_behaviors import curv_of_traj_utils
from plotly.subplots import make_subplots
from visualization.plotly_tools import plotly_for_monkey

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

colors = matplotlib.colors.TABLEAU_COLORS
hex_colors = tuple(colors.values())

# Default styling constants for a cleaner time series plot
PLOTLY_TIME_SERIES_TEMPLATE = 'plotly_white'
GRID_COLOR = 'rgba(0,0,0,0.08)'
AXIS_LINE_COLOR = 'rgba(0,0,0,0.35)'
GUIDE_LINE_COLOR = '#ff7f0e'  # 'rgba(0,0,0,0.45)'
REF_POINT_COLOR = '#e46696'
HOVER_LINE_COLOR = '#2ca02c'  # '#1f77b4'
CURVATURE_COLOR = '#ff7f0e'  # orange from Plotly default palette


def make_the_plot_of_change_in_curv_of_traj_vs_time(curv_of_traj_in_duration, y_column_name='curv_of_traj_diff_over_distance', x_column_name='rel_time'):
    curv_of_traj_in_duration = curv_of_traj_in_duration.copy()
    plot_to_add = px.line(curv_of_traj_in_duration, x=x_column_name, y=y_column_name,
                          title='Change in Curvature of Trajectory',
                          hover_data=[x_column_name, y_column_name],
                          labels={'rel_time': 'Relative Time(s)',
                                  'rel_distance': 'Relative Distance(cm)',
                                  'curv_of_traj_diff': 'Change in Curvature of Trajectory (deg/cm)',
                                  'curv_of_traj_diff_over_dt': 'Change in Curv of Trajectory Over Time',
                                  'curv_of_traj_diff_over_distance': 'Change in Curv of Trajectory Over Distance', },
                          # width=1000, height=700,
                          )

    return plot_to_add


def _apply_clean_time_series_theme(fig):
    # Check if legend is already configured
    existing_legend = fig.layout.legend if hasattr(
        fig.layout, 'legend') else None

    fig.update_layout(
        template=PLOTLY_TIME_SERIES_TEMPLATE,
        font=dict(family='Arial, Helvetica, Sans-Serif',
                  size=12, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 60, 'b': 40, 't': 120, 'r': 60},
        hoverdistance=-1,  # Distance in pixels for hover detection
    )

    # Only set legend if it's not already configured
    if existing_legend is None:
        fig.update_layout(legend=dict(
            orientation='h', y=1.05, x=0.5, xanchor='center'))
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=AXIS_LINE_COLOR,
        linewidth=1,
        mirror=True,
        ticks='outside',
        ticklen=4,
        tickcolor=AXIS_LINE_COLOR,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=AXIS_LINE_COLOR,
        linewidth=1,
        mirror=True,
        ticks='outside',
        ticklen=4,
        tickcolor=AXIS_LINE_COLOR,
    )
    return fig


def make_the_initial_fig_time_series(curv_of_traj_in_duration, monkey_hoverdata_value, cur_ff_color, nxt_ff_color, use_two_y_axes=False, change_y_ranges=True, add_vertical_line=True,
                                     x_column_name='rel_time', trajectory_ref_row=None, curv_of_traj_trace_name='Curvature of Trajectory', show_visible_segments=True,
                                     visible_segments_info={},
                                     y_range_for_v_line=[-200, 200], trajectory_next_stop_row=None):
    x_range_for_h_line = [np.min(curv_of_traj_in_duration[x_column_name].values), np.max(
        curv_of_traj_in_duration[x_column_name].values)]
    if use_two_y_axes:
        fig_time_series = plot_curv_of_traj_vs_time_with_two_y_axes(
            curv_of_traj_in_duration, change_y_ranges=change_y_ranges, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
        # plot two horizontal lines at 0.01 and -0.01 based on y-axis
        fig_time_series = add_two_horizontal_lines(
            fig_time_series, use_two_y_axes, x_range=x_range_for_h_line)
    else:
        fig_time_series = plot_curv_of_traj_vs_time(
            curv_of_traj_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
    if add_vertical_line:
        fig_time_series = add_vertical_line_for_an_x_value(
            fig_time_series, x_value=monkey_hoverdata_value, y_range=y_range_for_v_line, color=HOVER_LINE_COLOR)
    if trajectory_ref_row is not None:
        fig_time_series = mark_reference_point_in_time_series_plot(
            fig_time_series, x_column_name, trajectory_ref_row, y_range=y_range_for_v_line)
    if show_visible_segments:
        time_or_distance = 'time' if x_column_name == 'rel_time' else 'distance'
        stops_near_ff_row = visible_segments_info['stops_near_ff_row']
        fig_time_series = plot_blocks_to_show_ff_visible_segments_in_fig_time_series(fig_time_series, visible_segments_info['ff_info'], visible_segments_info['monkey_information'], stops_near_ff_row,
                                                                                     unique_ff_indices=[
            stops_near_ff_row.cur_ff_index, stops_near_ff_row.nxt_ff_index], time_or_distance=time_or_distance, y_range_for_v_line=y_range_for_v_line,
            varying_colors=[cur_ff_color, nxt_ff_color], ff_names=['cur ff', 'nxt ff'])

    # plot a vertical line at stop point (which is 0)
    fig_time_series = add_vertical_line_for_an_x_value(
        fig_time_series, x_value=0, y_range=y_range_for_v_line, name='Stop point', color=GUIDE_LINE_COLOR)
    # also add a horizontal line at 0
    fig_time_series = add_horizontal_line_to_fig_time_series(
        fig_time_series, use_two_y_axes, x_range=x_range_for_h_line, y_value=0, showlegend=False)

    if trajectory_next_stop_row is not None:
        fig_time_series = mark_next_stop_in_time_series_plot(
            fig_time_series, x_column_name, trajectory_next_stop_row, y_range_for_v_line=y_range_for_v_line)

    fig_time_series = add_annotation_to_fig_time_series(
        fig_time_series, 'stop point', 0, -130)

    fig_time_series = plotly_for_monkey.update_legend(fig_time_series)
    fig_time_series.update_layout(
        width=800,
        height=300,
        margin={'l': 60, 'b': 30, 't': 120, 'r': 60},
    )

    # Apply clean theme at the end so traces and axes are styled consistently
    fig_time_series = _apply_clean_time_series_theme(fig_time_series)

    fig_time_series.update_layout(yaxis=dict(range=['null', 'null'],),
                                  yaxis2=dict(range=['null', 'null'],))

    return fig_time_series


def plot_curv_of_traj_vs_time_with_two_y_axes(curv_of_traj_in_duration, change_y_ranges=True, y_column_name_for_change_in_curv='curv_of_traj_diff', x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory'):
    change_in_curv_of_traj_plot = make_the_plot_of_change_in_curv_of_traj_vs_time(
        curv_of_traj_in_duration, y_column_name=y_column_name_for_change_in_curv, x_column_name=x_column_name)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig = add_curv_of_traj_data_to_fig_time_series(
        fig, curv_of_traj_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)

    for data in change_in_curv_of_traj_plot.data:
        data.marker = {'color': 'green', 'symbol': 'circle'}
        data.name = 'Change in Curvature of Trajectory'
        data.showlegend = True
        fig.add_trace(data, secondary_y=True)
        fig.update_traces(visible='legendonly', opacity=0.5, marker={'size': 3}, line={
                          'color': 'green'}, selector=dict(name='Change in Curvature of Trajectory'))

    if y_column_name_for_change_in_curv == 'curv_of_traj_diff_over_dt':
        yaxis2_title = "Delta Curv of Trajectory (deg/cm/s)"
    elif y_column_name_for_change_in_curv == 'curv_of_traj_diff_over_distance':
        yaxis2_title = "Delta Curv of Trajectory (deg/cm^2)"
    elif y_column_name_for_change_in_curv == 'curv_of_traj_diff':
        yaxis2_title = "Delta Curv of Trajectory (deg/cm)"
    else:
        yaxis2_title = 'y axis 2 title'

    fig.update_layout(
        yaxis2=dict(title=dict(text=yaxis2_title),
                    side="right",
                    overlaying="y",
                    tickmode="sync"))

    if change_y_ranges:
        fig.update_layout(yaxis=dict(range=[-100, 100]),
                          yaxis2=dict(range=[-25, 25]))

    fig = _apply_clean_time_series_theme(fig)

    return fig


def add_curv_of_traj_data_to_fig_time_series(fig, curv_of_traj_in_duration, x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory'):
    # curv_of_traj_plot = make_the_plot_of_curv_of_traj_vs_time(curv_of_traj_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
    # for data in curv_of_traj_plot.data:
    #     data.marker = {'color': 'orange', 'symbol': 'circle', 'opacity': 0.8}
    #     data.name = curv_of_traj_trace_name
    #     data.showlegend = True
    #     fig.add_trace(data)
    #     fig.update_traces(marker={'size': 3}, selector=dict(name=curv_of_traj_trace_name))

    if fig is None:
        fig = go.Figure(layout=dict(width=1000, height=700))
    plot_to_add = make_new_trace_for_time_series_plot(curv_of_traj_in_duration, curv_of_traj_trace_name, color=CURVATURE_COLOR,
                                                      x_column_name=x_column_name, y_column_name='curv_of_traj_deg_over_cm', symbol='circle', size=6)
    fig.add_trace(plot_to_add)

    if x_column_name == 'rel_time':
        x_axis_label = "Relative Time (s)"
    elif x_column_name == 'rel_distance':
        x_axis_label = "Relative Distance (cm)"
    else:
        x_axis_label = 'x axis label'

    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center'),
                      # xaxis=dict(range=[-2.5, 0.1]),
                      # xaxis=dict(title=dict(text=x_axis_label)),
                      yaxis=dict(title=dict(text="Curvature of Trajectory (deg/cm)"),
                                 side="left"),
                      title=go.layout.Title(text=x_axis_label,
                                            xref="paper",
                                            x=0,
                                            font=dict(size=14)),)
    return fig


def add_annotation_to_fig_time_series(fig_time_series, text, x_position, y_position=130):
    fig_time_series.add_annotation(
        x=x_position,
        y=y_position,
        text=text,
        name='annotation_' + text,
        showarrow=False,
        font=dict(
            size=16,
            color="black"
        ),
        align="center",
        # ax=0,
        # ay=-30,
        bordercolor="rgba(0,0,0,0)",
        borderwidth=2,
        borderpad=4,
        bgcolor='white',
        opacity=0.6
    )
    return fig_time_series


def add_vertical_line_for_an_x_value(fig_time_series, x_value=0, y_range=[-100, 100],
                                     name='Monkey trajectory hover position', color='blue', dash=None, width=None):

    vline_df = pd.DataFrame({'x': [x_value, x_value], 'y': y_range})
    fig_traces = px.line(vline_df,
                         x='x',
                         y='y',
                         )
    fig_time_series.add_traces(list(fig_traces.select_traces()))
    fig_time_series.data[-1].name = name

    # Default styling for common guideline types
    if dash is None:
        dash = 'dash' if name == 'Monkey trajectory hover position' else 'solid'
        if name == 'Next stop point':
            dash = 'dot'
    if width is None:
        width = 2
    fig_time_series.update_traces(opacity=1, selector=dict(name=name),
                                  line=dict(color=color, width=width, dash=dash))

    return fig_time_series


def mark_reference_point_in_time_series_plot(fig_time_series, x_column_name, trajectory_ref_row, y_range=[-200, 200]):
    if x_column_name == 'rel_time':
        ref_point_value = trajectory_ref_row['rel_time']
    elif x_column_name == 'rel_distance':
        ref_point_value = trajectory_ref_row['rel_distance']

    vline_df = pd.DataFrame(
        {'x': [ref_point_value, ref_point_value], 'y': y_range})
    fig_traces = px.line(vline_df,
                         x='x',
                         y='y',
                         )
    fig_time_series.add_traces(list(fig_traces.select_traces()))
    fig_time_series.data[-1].name = 'Ref point'
    fig_time_series.data[-1].showlegend = True
    fig_time_series.update_traces(opacity=1, selector=dict(name='Ref point'),
                                  line=dict(color=REF_POINT_COLOR, width=2, dash='dot'))
    fig_time_series = add_annotation_to_fig_time_series(
        fig_time_series, 'ref point', ref_point_value, -130)
    return fig_time_series


def mark_next_stop_in_time_series_plot(fig_time_series, x_column_name, trajectory_next_stop_row, y_range_for_v_line=[-200, 200]):
    if x_column_name == 'rel_time':
        next_stop_value = trajectory_next_stop_row['rel_time']
    elif x_column_name == 'rel_distance':
        next_stop_value = trajectory_next_stop_row['rel_distance']
    fig_time_series = add_vertical_line_for_an_x_value(
        fig_time_series, x_value=next_stop_value, y_range=y_range_for_v_line, name='Next stop point', color='black')
    return fig_time_series


def add_line_for_current_time_window(fig_time_series, curv_of_traj_in_duration, current_time_window, x_column_name='rel_time'):
    curv_of_traj_df_to_use = curv_of_traj_in_duration[
        curv_of_traj_in_duration['time_window'] == current_time_window].copy()
    fig_time_series.add_trace(
        go.Scatter(x=curv_of_traj_df_to_use[x_column_name].values, y=curv_of_traj_df_to_use['curv_of_traj_deg_over_cm'].values,
                   mode='lines',
                   name='line_for_current_time_window',),
    )

    return fig_time_series


def add_horizontal_line_to_fig_time_series(fig_time_series, use_two_y_axes, x_range=[-3, 3], y_value=0, showlegend=False):
    secondary_y = use_two_y_axes if use_two_y_axes else None
    fig_time_series.add_trace(
        go.Scatter(x=x_range, y=[y_value, y_value],
                   mode='lines',
                   name='y2 =' + str(y_value),
                   showlegend=showlegend,
                   ), secondary_y=secondary_y,
    )
    fig_time_series.update_traces(opacity=1, selector=dict(name='y2 =' + str(y_value)),
                                  line=dict(color='#888', width=2, dash='dot'))
    return fig_time_series


def add_two_horizontal_lines(fig_time_series, use_two_y_axes, x_range=[-3, 3], y_value=5):
    secondary_y = use_two_y_axes if use_two_y_axes else None
    fig_name = 'y2 =' + str(y_value) + ' or -' + str(y_value)
    fig_time_series.add_trace(
        go.Scatter(x=x_range, y=[y_value, y_value],
                   mode='lines',
                   name=fig_name,
                   showlegend=True,
                   ), secondary_y=secondary_y,
    )
    fig_time_series.add_trace(
        go.Scatter(x=x_range, y=[-y_value, -y_value],
                   mode='lines',
                   name=fig_name,
                   showlegend=False,
                   ), secondary_y=secondary_y,
    )
    fig_time_series.update_traces(opacity=1, selector=dict(name=fig_name), visible='legendonly',
                                  line=dict(color='#888', width=1, dash='dot'))
    return fig_time_series


def make_new_trace_for_time_series_plot(ff_curv_df, name, color='purple', x_column_name='rel_time', y_column_name='cntr_arc_curv', size=6, symbol='circle', showlegend=True):
    plot_to_add = go.Scatter(x=ff_curv_df[x_column_name], y=ff_curv_df[y_column_name],
                             name=name,
                             legendgroup=name,
                             marker=dict(color=color, size=size,
                                         opacity=0.8, symbol=symbol),
                             showlegend=showlegend,
                             mode='markers')
    return plot_to_add


def add_to_time_series_plot(fig, ff_curv_df, name, color='purple', x_column_name='rel_time', y_column_name='cntr_arc_curv', symbol='circle'):
    plot_to_add = make_new_trace_for_time_series_plot(
        ff_curv_df, name, color=color, x_column_name=x_column_name, y_column_name=y_column_name, symbol=symbol)
    fig.add_trace(plot_to_add)
    return fig


def add_new_curv_of_traj_to_fig_time_series(fig_time_series, curv_of_traj_in_duration, curv_of_traj_mode, lower_end, upper_end, x_column_name, symbol='circle'):
    random_color = random.choice(hex_colors)
    window_for_curv_of_traj = [lower_end, upper_end]
    curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
        curv_of_traj_mode, window_for_curv_of_traj)
    fig_time_series_updated = add_to_time_series_plot(fig_time_series, curv_of_traj_in_duration, curv_of_traj_trace_name, x_column_name=x_column_name, y_column_name='curv_of_traj_deg_over_cm',
                                                      color=random_color, symbol=symbol)
    return fig_time_series_updated


def add_new_curv_of_traj_to_fig_time_series_combd(fig_time_series_combd, curv_of_traj_in_duration, curv_of_traj_mode, lower_end, upper_end):
    random_color = random.choice(hex_colors)
    window_for_curv_of_traj = [lower_end, upper_end]
    curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
        curv_of_traj_mode, window_for_curv_of_traj)
    plot_to_add_cm = make_new_trace_for_time_series_plot(curv_of_traj_in_duration, curv_of_traj_trace_name,
                                                         color=random_color, x_column_name='rel_time', y_column_name='curv_of_traj_deg_over_cm', size=7)
    fig_time_series_combd.add_trace(plot_to_add_cm, row=1, col=1)
    plot_to_add_s = make_new_trace_for_time_series_plot(curv_of_traj_in_duration, curv_of_traj_trace_name, color=random_color,
                                                        x_column_name='rel_distance', y_column_name='curv_of_traj_deg_over_cm', size=7, showlegend=False)
    fig_time_series_combd.add_trace(plot_to_add_s, row=2, col=1)
    return fig_time_series_combd


def plot_curv_of_traj_vs_time(curv_of_traj_in_duration, x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory', change_y_ranges=True):
    fig = add_curv_of_traj_data_to_fig_time_series(
        None, curv_of_traj_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)

    if change_y_ranges:
        fig.update_layout(yaxis=dict(range=[-100, 100]))
    return fig


def turn_visibility_of_vertical_lines_on_or_off_in_time_series_plot(fig_time_series,
                                                                    visible,
                                                                    trace_names=['Monkey trajectory hover position', 'Ref point']):
    for name in trace_names:
        fig_time_series.update_traces(
            visible=visible, selector=dict(name=name))
    return fig_time_series


def find_monkey_hoverdata_value_for_both_fig_time_series(hoverdata_column, monkey_hoverdata_value, trajectory_df, hover_lookup=None):
    """Return both hover values (s, cm) using a fast lookup when possible.

    If hover_lookup is provided, it should be a dict with entries for
    'rel_time' and/or 'rel_distance', each containing sorted 'x' and matching
    'y' arrays. We then use np.searchsorted for O(log n) lookup.

    Falls back to DataFrame-based approach if no cache is available.
    """
    # Fast path using prebuilt lookup cache
    if isinstance(hover_lookup, dict) and hoverdata_column in hover_lookup:
        entry = hover_lookup[hoverdata_column]
        xs = entry.get('x')
        ys = entry.get('y')
        if xs is not None and ys is not None and getattr(xs, 'size', 0) > 0:
            pos = int(np.searchsorted(xs, monkey_hoverdata_value, side='left'))
            if pos >= xs.size:
                pos = xs.size - 1
            if hoverdata_column == 'rel_time':
                monkey_hoverdata_value_s = float(monkey_hoverdata_value)
                monkey_hoverdata_value_cm = float(ys[pos])
            else:
                monkey_hoverdata_value_cm = float(monkey_hoverdata_value)
                monkey_hoverdata_value_s = float(ys[pos])
            return monkey_hoverdata_value_s, monkey_hoverdata_value_cm

    # Fallback: DataFrame scan
    if hoverdata_column == 'rel_time':
        monkey_hoverdata_value_s = monkey_hoverdata_value
        trajectory_hover_row = trajectory_df.loc[trajectory_df['rel_time']
                                                 >= monkey_hoverdata_value_s]
        if len(trajectory_hover_row) > 0:
            trajectory_hover_row = trajectory_hover_row.iloc[0]
        else:
            trajectory_hover_row = trajectory_df.iloc[-1]
        monkey_hoverdata_value_cm = trajectory_hover_row['rel_distance']

    elif hoverdata_column == 'rel_distance':
        monkey_hoverdata_value_cm = monkey_hoverdata_value
        trajectory_hover_row = trajectory_df.loc[trajectory_df['rel_distance']
                                                 >= monkey_hoverdata_value_cm]
        if len(trajectory_hover_row) > 0:
            trajectory_hover_row = trajectory_hover_row.iloc[0]
        else:
            trajectory_hover_row = trajectory_df.iloc[-1]
        monkey_hoverdata_value_s = trajectory_hover_row['rel_time']

    return monkey_hoverdata_value_s, monkey_hoverdata_value_cm


def make_fig_time_series_combd(fig_time_series_s, fig_time_series_cm, use_two_y_axes):
    fig_time_series_s = go.Figure(fig_time_series_s)
    fig_time_series_cm = go.Figure(fig_time_series_cm)
    overall_secondary_y = True if use_two_y_axes else None
    if overall_secondary_y is None:
        secondary_y = None

    existing_legends = []
    fig_time_series_combd = make_subplots(rows=2, cols=1,  # vertical_spacing = 0.35,
                                          specs=[[{"secondary_y": overall_secondary_y}], [{"secondary_y": overall_secondary_y}]])

    for data in fig_time_series_s.data:
        data['legendgroup'] = data['name']
        if data['name'] not in existing_legends:
            data['showlegend'] = True
            existing_legends.append(data['name'])
        else:
            data['showlegend'] = False
        if overall_secondary_y is True:  # then we judge again whether we should use secondary_y for this particular trace
            secondary_y = (data['name'] in [
                           'Change in Curvature of Trajectory', 'y2 =5 or -5'])
        fig_time_series_combd.add_trace(
            data, row=1, col=1, secondary_y=secondary_y)

    # Transfer annotations from the time-based time series plot
    for annotation in fig_time_series_s.layout.annotations:
        fig_time_series_combd.add_annotation(annotation)

    # Transfer shapes from the time-based time series plot
    if hasattr(fig_time_series_s.layout, 'shapes') and fig_time_series_s.layout.shapes:
        for shape in fig_time_series_s.layout.shapes:
            # Add shape to the first subplot (time-based)
            fig_time_series_combd.add_shape(shape, row=1, col=1)

    for data in fig_time_series_cm.data:
        data['legendgroup'] = data['name']
        data['showlegend'] = False
        data['name'] = 'time_series_cm_' + data['name']
        if overall_secondary_y is True:  # then we judge again whether we should use secondary_y for this particular trace
            secondary_y = (data['name'] in [
                           'Change in Curvature of Trajectory', 'y2 =5 or -5'])
        fig_time_series_combd.add_trace(
            data, row=2, col=1, secondary_y=secondary_y)

    # Transfer shapes from the distance-based time series plot (if any)
    if hasattr(fig_time_series_cm.layout, 'shapes') and fig_time_series_cm.layout.shapes:
        for shape in fig_time_series_cm.layout.shapes:
            # Add shape to the second subplot (distance-based)
            fig_time_series_combd.add_shape(shape, row=2, col=1)

    fig_time_series_combd.update_layout(legend=dict(orientation="h", y=1.2, groupclick="togglegroup"),
                                        width=800, height=600,
                                        margin=dict(l=10, r=50, b=10,
                                                    t=120, pad=4),
                                        # paper_bgcolor="LightSteelBlue",
                                        xaxis=dict(title='Relative Time (s)'),
                                        xaxis2=dict(
                                        title='Relative Distance (cm)'),
                                        yaxis=dict(title=dict(
                                            text="Curvature of Trajectory (deg/cm)"), side="left"),
                                        # yaxis3=dict(title=dict(text="Curvature of Trajectory (deg/cm)"), side="left"),
                                        )

    if use_two_y_axes:
        fig_time_series_combd.update_layout(yaxis2=dict(title=dict(text="Delta Curv of Trajectory (deg/cm)"),
                                                        side="right", overlaying="y", tickmode="sync"),
                                            # yaxis4=dict(title=dict(text="Delta Curv of Trajectory (deg/cm)"),
                                            #                         side="right", overlaying="y3", tickmode="sync"),
                                            )
    fig_time_series_combd = _apply_clean_time_series_theme(
        fig_time_series_combd)
    return fig_time_series_combd


def update_fig_time_series_natural_y_range(fig_time_series_natural_y_range, df, y_column_name, cap=[-200, 200]):
    new_fig_time_series_natural_y_range = [
        np.min(df[y_column_name].values), np.max(df[y_column_name].values)]
    fig_time_series_natural_y_range = [np.min([fig_time_series_natural_y_range[0], new_fig_time_series_natural_y_range[0]]), np.max([
        fig_time_series_natural_y_range[1], new_fig_time_series_natural_y_range[1]])]
    if fig_time_series_natural_y_range[0] < cap[0]:
        fig_time_series_natural_y_range[0] = cap[0]
    if fig_time_series_natural_y_range[1] > cap[1]:
        fig_time_series_natural_y_range[1] = cap[1]
    return fig_time_series_natural_y_range


def plot_blocks_to_show_ff_visible_segments_in_fig_time_series(
    fig_time_series,
    ff_info,
    monkey_information,
    stops_near_ff_row,
    unique_ff_indices=None,
    time_or_distance='time',
    y_range_for_v_line=[-200, 200],
    varying_colors=['#33BBFF', '#FF337D', '#FF33D7', '#8D33FF', '#33FF64'],
    ff_names=None,
    block_opacity=0.2,
    show_annotation=True,
    annotation_opacity=0.5,
):
    point_index_gap_threshold_to_sep_vis_intervals = 12
    unique_ff_indices = ff_info.ff_index.unique(
    ) if unique_ff_indices is None else np.array(unique_ff_indices)

    if ff_names is None:
        ff_names = ['ff ' + str(i) for i in range(len(unique_ff_indices))]

    for i, ff_index in enumerate(unique_ff_indices):
        color = varying_colors[i % len(varying_colors)]
        temp_df = ff_info[ff_info['ff_index'] ==
                          ff_index].copy().sort_values(by='point_index')

        if temp_df.empty:
            continue

        all_point_index = temp_df.point_index.values
        all_breaking_points = np.where(np.diff(
            all_point_index) >= point_index_gap_threshold_to_sep_vis_intervals)[0] + 1

        all_starting_points = np.insert(
            all_point_index[all_breaking_points], 0, all_point_index[0])
        all_ending_points = np.append(
            all_point_index[all_breaking_points - 1], all_point_index[-1])

        ref_value = stops_near_ff_row['stop_time'] if time_or_distance == 'time' else stops_near_ff_row['stop_cum_distance']
        time_or_distance_var = 'time' if time_or_distance == 'time' else 'cum_distance'

        all_starting_rel_values = monkey_information.loc[all_starting_points,
                                                         time_or_distance_var].values - ref_value
        all_ending_rel_values = monkey_information.loc[all_ending_points,
                                                       time_or_distance_var].values - ref_value

        # Add one block (shape) per visible segment
        for j in range(len(all_starting_rel_values)):
            # Add a translucent rectangular block spanning start to end
            fig_time_series.add_shape(
                type='rect',
                x0=all_starting_rel_values[j],
                x1=all_ending_rel_values[j],
                y0=y_range_for_v_line[0],
                y1=y_range_for_v_line[1],
                fillcolor=color,
                opacity=block_opacity,
                layer='below',
                line=dict(width=0),
                name=ff_names[i] + f' visible segment {j}'
            )
            # Optionally add annotation at the center of the block
            if show_annotation:
                center_x = (
                    all_starting_rel_values[j] + all_ending_rel_values[j]) / 2
                fig_time_series.add_annotation(
                    x=center_x,
                    y=y_range_for_v_line[1],
                    text=ff_names[i],
                    showarrow=False,
                    font=dict(color=color),
                    opacity=annotation_opacity,
                    yanchor='bottom'
                )

    return fig_time_series
