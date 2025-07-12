import sys
from turtle import fillcolor

from data_wrangling import specific_utils
from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import plot_stops_near_ff_utils
from decision_making_analysis.decision_making import plot_decision_making
from visualization.plotly_tools import plotly_preparation, plotly_for_monkey
from decision_making_analysis import trajectory_info

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib
import random

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


colors = matplotlib.colors.TABLEAU_COLORS
hex_colors = tuple(colors.values())


def make_the_initial_fig_scatter(curv_of_traj_df_in_duration, monkey_hoverdata_value, cur_ff_color, nxt_ff_color, use_two_y_axes=True, change_y_ranges=True, add_vertical_line=True,
                                 x_column_name='rel_time', trajectory_ref_row=None, curv_of_traj_trace_name='Curvature of Trajectory', show_visible_segments=True, visible_segments_info={},
                                 y_range_for_v_line=[-200, 200], trajectory_next_stop_row=None):
    if use_two_y_axes:
        fig_scatter = plot_curv_of_traj_vs_time_with_two_y_axes(
            curv_of_traj_df_in_duration, change_y_ranges=change_y_ranges, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
        # plot two horizontal lines at 0.01 and -0.01 based on y-axis
        x_range_for_h_line = [np.min(curv_of_traj_df_in_duration[x_column_name].values), np.max(
            curv_of_traj_df_in_duration[x_column_name].values)]
        fig_scatter = add_two_horizontal_lines(
            fig_scatter, use_two_y_axes, x_range=x_range_for_h_line)
    else:
        fig_scatter = plot_curv_of_traj_vs_time(
            curv_of_traj_df_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
    if add_vertical_line:
        fig_scatter = add_vertical_line_for_an_x_value(
            fig_scatter, x_value=monkey_hoverdata_value, y_range=y_range_for_v_line)
    if trajectory_ref_row is not None:
        fig_scatter = mark_reference_point_in_scatter_plot(
            fig_scatter, x_column_name, trajectory_ref_row, y_range=y_range_for_v_line)
    if show_visible_segments:
        time_or_distance = 'time' if x_column_name == 'rel_time' else 'distance'
        stops_near_ff_row = visible_segments_info['stops_near_ff_row']
        plot_lines_to_show_ff_visible_segments_in_fig_scatter(fig_scatter, visible_segments_info['ff_info'], visible_segments_info['monkey_information'], stops_near_ff_row,
                                                              unique_ff_indices=[
                                                                  stops_near_ff_row.cur_ff_index, stops_near_ff_row.nxt_ff_index], time_or_distance=time_or_distance, y_range_for_v_line=y_range_for_v_line,
                                                              varying_colors=[cur_ff_color, nxt_ff_color], ff_names=['cur ff', 'nxt ff'])
    # plot a vertical line at stop point (which is 0)
    fig_scatter = add_vertical_line_for_an_x_value(
        fig_scatter, x_value=0, y_range=y_range_for_v_line, name='First stop point', color='black')
    if trajectory_next_stop_row is not None:
        fig_scatter = mark_next_stop_in_scatter_plot(
            fig_scatter, x_column_name, trajectory_next_stop_row, y_range_for_v_line=y_range_for_v_line)

    fig_scatter = add_annotation_to_fig_scatter(
        fig_scatter, 'stop point', 0, -130)

    fig_scatter = plotly_for_monkey.update_legend(fig_scatter)
    fig_scatter.update_layout(
        width=800,
        height=300,
        margin={'l': 60, 'b': 30, 't': 0, 'r': 60},
    )

    fig_scatter.update_layout(yaxis=dict(range=['null', 'null'],),
                              yaxis2=dict(range=['null', 'null'],))

    return fig_scatter


def add_annotation_to_fig_scatter(fig_scatter, text, x_position, y_position=130):
    fig_scatter.add_annotation(
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
    return fig_scatter


def make_the_plot_of_change_in_curv_of_traj_vs_time(curv_of_traj_df_in_duration, y_column_name='curv_of_traj_diff_over_distance', x_column_name='rel_time'):
    curv_of_traj_df_in_duration = curv_of_traj_df_in_duration.copy()
    plot_to_add = px.line(curv_of_traj_df_in_duration, x=x_column_name, y=y_column_name,
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


def add_vertical_line_for_an_x_value(fig_scatter, x_value=0, y_range=[-100, 100],
                                     name='Monkey trajectory hover position', color='blue',
                                     ):

    vline_df = pd.DataFrame({'x': [x_value, x_value], 'y': y_range})
    fig_traces = px.line(vline_df,
                         x='x',
                         y='y',
                         )
    fig_scatter.add_traces(list(fig_traces.select_traces()))
    fig_scatter.data[-1].name = name
    fig_scatter.update_traces(opacity=1, selector=dict(name=name),
                              line=dict(color=color, width=2))

    return fig_scatter


def mark_reference_point_in_scatter_plot(fig_scatter, x_column_name, trajectory_ref_row, y_range=[-200, 200]):
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
    fig_scatter.add_traces(list(fig_traces.select_traces()))
    fig_scatter.data[-1].name = 'First stop point'
    fig_scatter.data[-1].showlegend = True
    fig_scatter.update_traces(opacity=1, selector=dict(name='First stop point'),
                              line=dict(color='black', width=2))
    fig_scatter = add_annotation_to_fig_scatter(
        fig_scatter, 'ref point', ref_point_value, -130)
    return fig_scatter


def mark_next_stop_in_scatter_plot(fig_scatter, x_column_name, trajectory_next_stop_row, y_range_for_v_line=[-200, 200]):
    if x_column_name == 'rel_time':
        next_stop_value = trajectory_next_stop_row['rel_time']
    elif x_column_name == 'rel_distance':
        next_stop_value = trajectory_next_stop_row['rel_distance']
    fig_scatter = add_vertical_line_for_an_x_value(
        fig_scatter, x_value=next_stop_value, y_range=y_range_for_v_line, name='Next stop point', color='black')
    return fig_scatter


def add_line_for_current_time_window(fig_scatter, curv_of_traj_df_in_duration, current_time_window, x_column_name='rel_time'):
    curv_of_traj_df_to_use = curv_of_traj_df_in_duration[
        curv_of_traj_df_in_duration['time_window'] == current_time_window].copy()
    fig_scatter.add_trace(
        go.Scatter(x=curv_of_traj_df_to_use[x_column_name].values, y=curv_of_traj_df_to_use['curv_of_traj_deg_over_cm'].values,
                   mode='lines',
                   name='line_for_current_time_window',),
    )

    return fig_scatter


def add_two_horizontal_lines(fig_scatter, use_two_y_axes, x_range=[-3, 3], y_value=5):
    secondary_y = use_two_y_axes if use_two_y_axes else None
    fig_name = 'y2 =' + str(y_value) + ' or -' + str(y_value)
    fig_scatter.add_trace(
        go.Scatter(x=x_range, y=[y_value, y_value],
                   mode='lines',
                   name=fig_name,
                   showlegend=True,
                   ), secondary_y=secondary_y,
    )
    fig_scatter.add_trace(
        go.Scatter(x=x_range, y=[-y_value, -y_value],
                   mode='lines',
                   name=fig_name,
                   showlegend=False,
                   ), secondary_y=secondary_y,
    )
    fig_scatter.update_traces(opacity=1, selector=dict(name=fig_name), visible='legendonly',
                              line=dict(color='red', width=1))
    return fig_scatter


def make_new_trace_for_scatterplot(ff_curv_df, name, color='purple', x_column_name='rel_time', y_column_name='cntr_arc_curv', size=5, symbol='circle', showlegend=True):
    plot_to_add = go.Scatter(x=ff_curv_df[x_column_name], y=ff_curv_df[y_column_name],
                             name=name,
                             legendgroup=name,
                             marker=dict(color=color, size=size,
                                         opacity=0.8, symbol=symbol),
                             showlegend=showlegend,
                             mode='markers')
    return plot_to_add


def add_to_the_scatterplot(fig, ff_curv_df, name, color='purple', x_column_name='rel_time', y_column_name='cntr_arc_curv', symbol='circle'):
    plot_to_add = make_new_trace_for_scatterplot(
        ff_curv_df, name, color=color, x_column_name=x_column_name, y_column_name=y_column_name, symbol=symbol)
    fig.add_trace(plot_to_add)
    return fig


def add_new_curv_of_traj_to_fig_scatter(fig_scatter, curv_of_traj_df_in_duration, curv_of_traj_mode, lower_end, upper_end, x_column_name, symbol='circle'):
    random_color = random.choice(hex_colors)
    window_for_curv_of_traj = [lower_end, upper_end]
    curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
        curv_of_traj_mode, window_for_curv_of_traj)
    fig_scatter_updated = add_to_the_scatterplot(fig_scatter, curv_of_traj_df_in_duration, curv_of_traj_trace_name, x_column_name=x_column_name, y_column_name='curv_of_traj_deg_over_cm',
                                                 color=random_color, symbol=symbol)
    return fig_scatter_updated


def add_new_curv_of_traj_to_fig_scatter_combd(fig_scatter_combd, curv_of_traj_df_in_duration, curv_of_traj_mode, lower_end, upper_end):
    random_color = random.choice(hex_colors)
    window_for_curv_of_traj = [lower_end, upper_end]
    curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
        curv_of_traj_mode, window_for_curv_of_traj)
    plot_to_add_cm = make_new_trace_for_scatterplot(curv_of_traj_df_in_duration, curv_of_traj_trace_name,
                                                    color=random_color, x_column_name='rel_time', y_column_name='curv_of_traj_deg_over_cm', size=7)
    fig_scatter_combd.add_trace(plot_to_add_cm, row=1, col=1)
    plot_to_add_s = make_new_trace_for_scatterplot(curv_of_traj_df_in_duration, curv_of_traj_trace_name, color=random_color,
                                                   x_column_name='rel_distance', y_column_name='curv_of_traj_deg_over_cm', size=7, showlegend=False)
    fig_scatter_combd.add_trace(plot_to_add_s, row=2, col=1)
    return fig_scatter_combd


def plot_curv_of_traj_vs_time(curv_of_traj_df_in_duration, x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory', change_y_ranges=True):
    fig = go.Figure()
    fig = add_curv_of_traj_data_to_fig_scatter(
        fig, curv_of_traj_df_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)

    if change_y_ranges:
        fig.update_layout(yaxis=dict(range=[-100, 100]))
    return fig


# def make_the_plot_of_curv_of_traj_vs_time(curv_of_traj_df_in_duration, x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory'):
#     curv_of_traj_df_in_duration = curv_of_traj_df_in_duration.copy()
#     hover_data=[x_column_name, 'curv_of_traj_deg_over_cm']
#     plot_to_add = px.scatter(curv_of_traj_df_in_duration, x=x_column_name, y='curv_of_traj_deg_over_cm',
#                                 title=curv_of_traj_trace_name,
#                                 hover_data=hover_data,
#                                 labels={'rel_time': 'Relative Time(s)',
#                                         'rel_distance': 'Relative Distance(cm)',
#                                         'time_window': 'time',
#                                         'curv_of_traj_deg_over_cm': 'Curvature of Trajectory (deg/cm)',},
#                                 #width=1000, height=700,
#                                     )
#     return plot_to_add


def add_curv_of_traj_data_to_fig_scatter(fig, curv_of_traj_df_in_duration, x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory'):
    # curv_of_traj_plot = make_the_plot_of_curv_of_traj_vs_time(curv_of_traj_df_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)
    # for data in curv_of_traj_plot.data:
    #     data.marker = {'color': 'orange', 'symbol': 'circle', 'opacity': 0.8}
    #     data.name = curv_of_traj_trace_name
    #     data.showlegend = True
    #     fig.add_trace(data)
    #     fig.update_traces(marker={'size': 3}, selector=dict(name=curv_of_traj_trace_name))

    fig = go.Figure(layout=dict(width=1000, height=700))
    plot_to_add = make_new_trace_for_scatterplot(curv_of_traj_df_in_duration, curv_of_traj_trace_name, color='orange',
                                                 x_column_name=x_column_name, y_column_name='curv_of_traj_deg_over_cm', symbol='circle', size=5)
    fig.add_trace(plot_to_add)

    if x_column_name == 'rel_time':
        x_axis_label = "Relative Time (s)"
    elif x_column_name == 'rel_distance':
        x_axis_label = "Relative Distance (cm)"
    else:
        x_axis_label = 'x axis label'

    fig.update_layout(legend=dict(orientation="h", y=-0.2),
                      # xaxis=dict(range=[-2.5, 0.1]),
                      # xaxis=dict(title=dict(text=x_axis_label)),
                      yaxis=dict(title=dict(text="Curvature of Trajectory (deg/cm)"),
                                 side="left"),
                      title=go.layout.Title(text=x_axis_label,
                                            xref="paper",
                                            x=0,
                                            font=dict(size=14)),)
    return fig


def plot_curv_of_traj_vs_time_with_two_y_axes(curv_of_traj_df_in_duration, change_y_ranges=True, y_column_name_for_change_in_curv='curv_of_traj_diff', x_column_name='rel_time', curv_of_traj_trace_name='Curvature of Trajectory'):
    change_in_curv_of_traj_plot = make_the_plot_of_change_in_curv_of_traj_vs_time(
        curv_of_traj_df_in_duration, y_column_name=y_column_name_for_change_in_curv, x_column_name=x_column_name)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig = add_curv_of_traj_data_to_fig_scatter(
        fig, curv_of_traj_df_in_duration, x_column_name=x_column_name, curv_of_traj_trace_name=curv_of_traj_trace_name)

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

    return fig


def turn_visibility_of_vertical_lines_on_or_off_in_scatter_plot(fig_scatter,
                                                                visible,
                                                                trace_names=['Monkey trajectory hover position', 'First stop point']):
    for name in trace_names:
        fig_scatter.update_traces(visible=visible, selector=dict(name=name))
    return fig_scatter


def find_monkey_hoverdata_value_for_both_fig_scatter(hoverdata_column, monkey_hoverdata_value, trajectory_df):
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


def make_fig_scatter_combd(fig_scatter_s, fig_scatter_cm, use_two_y_axes):
    fig_scatter_s = go.Figure(fig_scatter_s)
    fig_scatter_cm = go.Figure(fig_scatter_cm)
    overall_secondary_y = True if use_two_y_axes else None
    if overall_secondary_y is None:
        secondary_y = None

    existing_legends = []
    fig_scatter_combd = make_subplots(rows=2, cols=1,  # vertical_spacing = 0.35,
                                      specs=[[{"secondary_y": overall_secondary_y}], [{"secondary_y": overall_secondary_y}]])

    for data in fig_scatter_s.data:
        data['legendgroup'] = data['name']
        if data['name'] not in existing_legends:
            data['showlegend'] = True
            existing_legends.append(data['name'])
        else:
            data['showlegend'] = False
        if overall_secondary_y is True:  # then we judge again whether we should use secondary_y for this particular trace
            secondary_y = (data['name'] in [
                           'Change in Curvature of Trajectory', 'y2 =5 or -5'])
        fig_scatter_combd.add_trace(
            data, row=1, col=1, secondary_y=secondary_y)
    for annotation in fig_scatter_s.layout.annotations:
        fig_scatter_combd.add_annotation(annotation)

    for data in fig_scatter_cm.data:
        data['legendgroup'] = data['name']
        data['showlegend'] = False
        data['name'] = 'scatter_cm_' + data['name']
        if overall_secondary_y is True:  # then we judge again whether we should use secondary_y for this particular trace
            secondary_y = (data['name'] in [
                           'Change in Curvature of Trajectory', 'y2 =5 or -5'])
        fig_scatter_combd.add_trace(
            data, row=2, col=1, secondary_y=secondary_y)

    fig_scatter_combd.update_layout(legend=dict(orientation="h", y=1.3, groupclick="togglegroup"),
                                    width=800, height=600,
                                    margin=dict(l=10, r=50, b=10, t=50, pad=4),
                                    # paper_bgcolor="LightSteelBlue",
                                    xaxis=dict(title='Relative Time (s)'),
                                    xaxis2=dict(
                                        title='Relative Distance (cm)'),
                                    yaxis=dict(title=dict(
                                        text="Curvature of Trajectory (deg/cm)"), side="left"),
                                    # yaxis3=dict(title=dict(text="Curvature of Trajectory (deg/cm)"), side="left"),
                                    )

    if use_two_y_axes:
        fig_scatter_combd.update_layout(yaxis2=dict(title=dict(text="Delta Curv of Trajectory (deg/cm)"),
                                                    side="right", overlaying="y", tickmode="sync"),
                                        # yaxis4=dict(title=dict(text="Delta Curv of Trajectory (deg/cm)"),
                                        #                         side="right", overlaying="y3", tickmode="sync"),
                                        )
    return fig_scatter_combd


def update_fig_scatter_natural_y_range(fig_scatter_natural_y_range, df, y_column_name, cap=[-150, 150]):
    new_fig_scatter_natural_y_range = [
        np.min(df[y_column_name].values), np.max(df[y_column_name].values)]
    fig_scatter_natural_y_range = [np.min([fig_scatter_natural_y_range[0], new_fig_scatter_natural_y_range[0]]), np.max([
        fig_scatter_natural_y_range[1], new_fig_scatter_natural_y_range[1]])]
    if fig_scatter_natural_y_range[0] < cap[0]:
        fig_scatter_natural_y_range[0] = cap[0]
    if fig_scatter_natural_y_range[1] > cap[1]:
        fig_scatter_natural_y_range[1] = cap[1]
    return fig_scatter_natural_y_range


def plot_lines_to_show_ff_visible_segments_in_fig_scatter(fig_scatter, ff_info, monkey_information, stops_near_ff_row,
                                                          unique_ff_indices=None, time_or_distance='time', y_range_for_v_line=[-200, 200],
                                                          varying_colors=[
                                                              '#33BBFF', '#FF337D', '#FF33D7', '#8D33FF', '#33FF64'],
                                                          ff_names=None):

    # Define threshold for separating visible intervals
    point_index_gap_threshold_to_sep_vis_intervals = 12

    # Set unique_ff_indices to all unique indices in ff_info if not provided
    unique_ff_indices = ff_info.ff_index.unique(
    ) if unique_ff_indices is None else np.array(unique_ff_indices)

    # Iterate over unique firefly indices
    for i, ff_index in enumerate(unique_ff_indices):
        # Define color for current firefly
        color = varying_colors[i % 5]
        # Extract and sort data for current firefly
        temp_df = ff_info[ff_info['ff_index'] == ff_index].copy().sort_values(by=[
            'point_index'])

        if len(temp_df) == 0:
            continue

        # Find breaking points of visible segments
        all_point_index = temp_df.point_index.values
        all_breaking_points = np.where(np.diff(
            all_point_index) >= point_index_gap_threshold_to_sep_vis_intervals)[0] + 1

        # Find values for starting and ending points of visible segments
        all_starting_points = np.insert(
            all_point_index[all_breaking_points], 0, all_point_index[0])
        all_ending_points = np.append(
            all_point_index[all_breaking_points-1], all_point_index[-1])

        # Find relative values for starting and ending points of visible segments
        ref_value = stops_near_ff_row['stop_time'] if time_or_distance == 'time' else stops_near_ff_row['stop_cum_distance']
        time_or_distance_var = 'time' if time_or_distance == 'time' else 'cum_distance'
        all_starting_rel_values = monkey_information.loc[all_starting_points,
                                                         time_or_distance_var].values - ref_value
        all_ending_rel_values = monkey_information.loc[all_ending_points,
                                                       time_or_distance_var].values - ref_value
        if ff_names is None:
            ff_names = ['ff ' + str(i) for i in range(len(unique_ff_indices))]
        # Find and plot beginning and end of each visible segment
        for j in range(len(all_breaking_points)+1):
            # Plot points when firefly starts being visible
            if j == 0:
                showlegend = True
            else:
                showlegend = False

            fig_scatter = add_vertical_line_for_an_x_value(fig_scatter, x_value=all_starting_rel_values[j], y_range=y_range_for_v_line, color=color,
                                                           name=ff_names[i] + ' starts visible')

            fig_scatter = add_annotation_to_fig_scatter(
                fig_scatter, ff_names[i], all_starting_rel_values[j])

            fig_scatter.update_traces(opacity=1, selector=dict(name=ff_names[i] + ' starts visible'),
                                      showlegend=showlegend, legendgroup=ff_names[i])

            # fig_scatter = add_vertical_line_for_an_x_value(fig_scatter, x_value=all_ending_rel_values[j], y_range=y_range_for_v_line, color=color,
            #                          name=ff_names[i] + ' stops visible')

            # fig_scatter.update_traces(opacity=1, selector=dict(name=ff_names[i] + ' stops visible'),
            #                   line=dict(dash='dot'), showlegend=showlegend, legendgroup=ff_names[i])

            break  # for right now, we only want to show when the ff first becomes visible

    return fig_scatter
