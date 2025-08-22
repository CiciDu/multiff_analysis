
from visualization.matplotlib_tools import plot_behaviors_utils
from eye_position_analysis import eye_positions

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def update_layout_and_x_and_y_limit(fig, current_plotly_key_comp, show_eye_positions):
    x_min, x_max, y_min, y_max = find_monkey_xy_min_max(
        current_plotly_key_comp['trajectory_df'])
    if show_eye_positions:
        fig.update_layout(
            autosize=False,
            width=850,
            height=600,
            margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
        )
        fig.update_xaxes(range=[x_min - 100, x_max + 100])
        fig.update_yaxes(range=[y_min - 50, y_max + 450])
    else:
        fig.update_layout(
            autosize=False,
            # width=700,
            # height=400,
            width=850,
            height=600,
            margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
        )

        fig.update_xaxes(range=[x_min - 100, x_max + 100])
        fig.update_yaxes(range=[y_min - 50, y_max + 250])

    # also change fig size

    return fig


# def update_layout_and_x_and_y_limit_show_eye_positions(fig, current_plotly_key_comp):
#     fig.update_layout(
#                 autosize=False,
#                 width=850,
#                 height=600,
#                 margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
#             )

#     x_min, x_max, y_min, y_max = find_monkey_xy_min_max(current_plotly_key_comp['trajectory_df'])
#     fig.update_xaxes(range=[x_min - 100, x_max + 100])
#     fig.update_yaxes(range=[y_min - 50, y_max + 450])
#     return fig


def find_monkey_xy_min_max(trajectory_df):
    x_min = trajectory_df['monkey_x'].min()
    x_max = trajectory_df['monkey_x'].max()
    y_min = trajectory_df['monkey_y'].min()
    y_max = trajectory_df['monkey_y'].max()
    return x_min, x_max, y_min, y_max


def plot_eye_positions_in_plotly(fig, current_plotly_key_comp, show_eye_positions_for_both_eyes=False, x0=0, y0=0, trace_name='eye_positions',
                                 update_if_already_exist=True, marker_size=6, use_arrow_to_show_eye_positions=True):

    trajectory_df = current_plotly_key_comp['trajectory_df'].copy()
    duration = current_plotly_key_comp['duration_to_plot']

    if use_arrow_to_show_eye_positions:
        # clear existing annotations first
        fig['layout']['annotations'] = []
    df_for_eye_positions = trajectory_df.copy()

    if not show_eye_positions_for_both_eyes:
        monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
            df_for_eye_positions, duration, rotation_matrix=current_plotly_key_comp[
                'rotation_matrix']
        )
        monkey_subset = _merge_monkey_subset_with_trajectory_df(
            monkey_subset, trajectory_df)

        fig = plot_or_update_eye_positions_using_either_marker_or_arrow(
            fig, x0, y0, monkey_subset, trace_name='eye_positions', update_if_already_exist=update_if_already_exist,
            marker='circle', marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions
        )
    else:
        for suffix, marker, trace_name_suffix, arrowcolor in [
            ('_l', 'triangle-left', '_left', 'purple'),
            ('_r', 'triangle-right', '_right', 'orange')
        ]:
            monkey_subset = eye_positions.find_eye_positions_rotated_in_world_coordinates(
                df_for_eye_positions, duration, rotation_matrix=current_plotly_key_comp[
                    'rotation_matrix'], eye_col_suffix=suffix
            )
            monkey_subset = _merge_monkey_subset_with_trajectory_df(
                monkey_subset, trajectory_df)

            fig = plot_or_update_eye_positions_using_either_marker_or_arrow(
                fig, x0, y0, monkey_subset, trace_name=trace_name + trace_name_suffix, update_if_already_exist=update_if_already_exist,
                marker=marker, marker_size=marker_size, use_arrow_to_show_eye_positions=use_arrow_to_show_eye_positions, arrowcolor=arrowcolor
            )

    return fig


def _merge_monkey_subset_with_trajectory_df(monkey_subset, trajectory_df):
    columns_to_merge = ['rel_time', 'monkey_x', 'monkey_y']
    monkey_subset = monkey_subset.drop(
        columns=columns_to_merge, errors='ignore')
    monkey_subset = monkey_subset.merge(
        trajectory_df[['point_index'] + columns_to_merge], on='point_index', how='left')
    return monkey_subset


def plot_or_update_eye_positions_using_either_marker_or_arrow(fig, x0, y0, monkey_subset, trace_name='eye_positions', update_if_already_exist=True, marker='circle', marker_size=4, arrowcolor=None, use_arrow_to_show_eye_positions=False):
    if use_arrow_to_show_eye_positions:
        fig = _plot_or_update_arrow_to_eye_positions_in_plotly(
            fig, x0, y0, monkey_subset, trace_name=trace_name, update_if_already_exist=update_if_already_exist, arrowcolor=arrowcolor)
    else:
        fig = _plot_or_update_eye_positions_in_plotly(
            fig, x0, y0, monkey_subset, trace_name=trace_name, marker=marker, marker_size=marker_size, update_if_already_exist=update_if_already_exist)
    return fig


def _plot_or_update_eye_positions_in_plotly(fig, x0, y0, monkey_subset, all_rel_time_in_duration=None,
                                            trace_name='eye_positions', update_if_already_exist=True, marker='circle', marker_size=4):
    """
    Plot or update eye positions in a Plotly figure.

    Parameters:
    fig: Plotly figure
    x0, y0: coordinates
    marker: marker style
    overall_valid_indices: indices of valid data
    rel_time: relative time data
    gaze_world_xy_rotated: rotated gaze world coordinates

    Returns:
    fig: updated Plotly figure
    gaze_world_xy_rotated_valid: valid rotated gaze world coordinates
    """
    monkey_subset_valid = monkey_subset[monkey_subset['valid_view_point'] == True].copy(
    )
    all_rel_time_in_duration = monkey_subset['rel_time'].unique()

    marker_dict = dict(size=marker_size,
                       color='purple',
                       symbol=marker,
                       opacity=0.8,
                       line=dict(width=0.2,  # border width
                                 color='black')  # border color
                       )

    colorscale = 'Viridis'  # Define your colorscale here

    if len(all_rel_time_in_duration) > 1:
        marker_dict['color'] = monkey_subset_valid['rel_time'].values
        marker_dict['colorscale'] = colorscale
        marker_dict['cmin'] = min(all_rel_time_in_duration)
        marker_dict['cmax'] = max(all_rel_time_in_duration)

    trace_kwargs = {
        'x': monkey_subset_valid['gaze_world_x_rotated'].values - x0,
        'y': monkey_subset_valid['gaze_world_y_rotated'].values - y0,
        'marker': marker_dict,
    }

    # Check if the trace named 'eye_positions' is already in fig
    TO_UPDATE = False
    if update_if_already_exist:
        for trace in fig.data:
            if trace.name == trace_name:
                TO_UPDATE = True
                break
    if TO_UPDATE:
        fig.update_traces(**trace_kwargs, selector=dict(name=trace_name))
        return fig

    scatter = go.Scatter(**trace_kwargs, mode='markers',
                         name=trace_name, hoverinfo='skip')
    fig.add_trace(scatter)

    return fig


def _plot_or_update_arrow_to_eye_positions_in_plotly(fig, x0, y0, monkey_subset, all_rel_time_in_duration=None, trace_name='arrow_to_eye_positions', update_if_already_exist=True, marker='circle', marker_size=4, arrowcolor=None):
    """
    Plot or update eye positions in a Plotly figure.

    Parameters:
    fig: Plotly figure
    x0, y0: coordinates
    marker: marker style
    overall_valid_indices: indices of valid data
    rel_time: relative time data
    gaze_world_xy_rotated: rotated gaze world coordinates
    all_rel_time_in_duration: all relative times in duration (optional)

    Returns:
    fig: updated Plotly figure
    gaze_world_xy_rotated_valid: valid rotated gaze world coordinates
    """

    monkey_subset_meaningful = monkey_subset[monkey_subset['valid_view_point'] == True].copy(
    )

    if monkey_subset_meaningful.shape[0] == 0:
        # logging.warning(f'Failed to update arrow to eye positions for trace_name: {trace_name} because no valid eye position data are found.')
        return fig
    elif monkey_subset_meaningful.shape[0] > 1:
        # make a warning
        # logging.warning('More than one meaningful eye position found, only the first one is used to plot the arrow.')
        monkey_subset_meaningful = monkey_subset_meaningful.iloc[0:1]
    # else:
    #     logging.info('Meaningful eye position is found, using it to plot the arrow.')

    if arrowcolor is None:
        arrowcolor = 'black'
    # if arrowcolor is None:
    #     if all_rel_time_in_duration is None:
    #         arrowcolor = 'black'
    #     else:
    #         arrowcolor = get_color(monkey_subset_meaningful['rel_time'], min(all_rel_time_in_duration), max(all_rel_time_in_duration), cmap_name='viridis')

    trace_kwargs = {
        'ax': monkey_subset_meaningful['monkey_x'].item() - x0,
        'ay': monkey_subset_meaningful['monkey_y'].item() - y0,
        'x': monkey_subset_meaningful['gaze_world_x_rotated'].item() - x0,
        'y': monkey_subset_meaningful['gaze_world_y_rotated'].item() - y0,
        'arrowcolor': arrowcolor,
    }

    fig.add_annotation(
        **trace_kwargs,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
    )

    return fig


def get_color(value, cmin, cmax, cmap_name='viridis'):
    # Normalize the value
    normalized_value = (value - cmin) / (cmax - cmin)

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Get the color of the normalized value from the colormap
    color = mcolors.rgb2hex(cmap(normalized_value))

    return color


def update_legend(fig):
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        itemsizing='constant',
        xanchor="right",
        x=1
    ))
    return fig


def plot_fireflies(fig, ff_df):

    plot_to_add = px.scatter(ff_df, x='ff_x', y='ff_y',
                             color_discrete_sequence=['red'],
                             labels={'ff_x': 'ff x after rotation (cm)',
                                     'ff_y': 'ff y after rotation (cm)'},
                             custom_data='ff_number',
                             )

    for data in plot_to_add.data:
        data.name = 'plot_ff'
    if fig is None:
        fig = plot_to_add
    else:
        fig.add_traces(plot_to_add.data)

    fig.update_traces(hovertemplate='ff %{customdata[0]}',
                                    selector=dict(name='plot_ff'))
    return fig


def plot_reward_boundary_in_plotly(fig, ff_df, radius=25):
    for i in ff_df.index:
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=ff_df.loc[i, 'ff_x'] - radius,
                      y0=ff_df.loc[i, 'ff_y'] - radius,
                      x1=ff_df.loc[i, 'ff_x'] + radius,
                      y1=ff_df.loc[i, 'ff_y'] + radius,
                      #   line_color="orange",
                      #   opacity=0.25,
                      #   fillcolor="grey",
                      # Orange color with full opacity for the line
                      line=dict(color="rgba(255, 165, 0, 0.45)"),
                      # Grey color with 25% opacity for the fill
                      fillcolor="rgba(128, 128, 128, 0.25)",
                      )
    return fig


def plot_arena_edge_in_plotly(fig, radius=1000):
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-radius,
                  y0=-radius,
                  x1=radius,
                  y1=radius,
                  line_color="grey",
                  opacity=0.45,
                  fillcolor=None,
                  )
    return fig


def plot_trajectory_data(fig, traj_df_to_use, show_color_as_time=False, show_traj_color_as_speed=True, hoverdata_multi_columns=['rel_time']):

    custom_data = hoverdata_multi_columns + ['monkey_speed']
    fig.add_traces(
        list(px.scatter(traj_df_to_use,
                        x='monkey_x',
                        y='monkey_y',
                        hover_data=custom_data,
                        labels={'monkey_x': 'monkey x after rotation (cm)',
                                'monkey_y': 'monkey y after rotation (cm)',
                                'rel_time': 'relative time (s)',
                                'rel_distance': 'relative distance (cm)'},
                        custom_data=custom_data,
                        # color=hoverdata_multi_columns[0],
                        # color_continuous_scale=px.colors.diverging.PuOr,
                        ).select_traces()))

    fig.data[-1].name = 'trajectory_data'

    fig.update_layout(coloraxis_showscale=False)

    # update hovertemplate
    hovertemplate = ' <br>'.join(
        [f'{col}: %{{customdata[{i}]:.2f}}' for i, col in enumerate(hoverdata_multi_columns)])
    fig.update_traces(hovertemplate=hovertemplate,
                      selector=dict(name='trajectory_data'))

    if show_color_as_time:
        marker_dict = {'color': traj_df_to_use['rel_time'].values,
                       'colorscale': 'Viridis',
                       'cmin': traj_df_to_use['rel_time'].min(),
                       'cmax': traj_df_to_use['rel_time'].max(),
                       'opacity': 0.8}
        fig.update_traces(marker=marker_dict,
                          selector=dict(name='trajectory_data'))

    elif show_traj_color_as_speed:  # note that show_color_as_time takes precedence over show_traj_color_as_speed
        marker_dict = {'color': traj_df_to_use['monkey_speed'].values,
                       'colorscale': 'Viridis',
                       'cmin': 0,
                       'cmax': 200,
                       'opacity': 0.8}

        hovertemplate = ' <br>'.join(
            [f'{col}: %{{customdata[{i}]:.2f}}' for i, col in enumerate(custom_data)])
        # hovertemplate = hovertemplate + '<br>speed: %{customdata[len_customdata]:.2f}'
        fig.update_traces(marker=marker_dict, hovertemplate=hovertemplate, selector=dict(
            name='trajectory_data'))
        # If show color as speed, let's add it to hover data, keep 2 decimals
    else:
        fig.update_traces(marker=dict(color='gold'),
                          selector=dict(name='trajectory_data'))

    return fig


def plot_stops_in_plotly(fig, trajectory_df, show_stop_point_indices, hoverdata_multi_columns=['rel_time'], 
                         name='stops', color='black'):
    
    # if show_stop_point_indices is int, make it a list
    if isinstance(show_stop_point_indices, int):
        show_stop_point_indices = [show_stop_point_indices]
    
    trajectory_df_sub = trajectory_df[trajectory_df['point_index'].isin(
        show_stop_point_indices)]
    plot_to_add = px.scatter(trajectory_df_sub, x='monkey_x', y='monkey_y',
                             color_discrete_sequence=['black'],
                             hover_data=hoverdata_multi_columns,
                             labels={'monkey_x': 'monkey x after rotation (cm)',
                                     'monkey_y': 'monkey y after rotation (cm)',
                                     'rel_time': 'relative time (s)',
                                     'rel_distance': 'relative distance (cm)'},
                             )

    fig.add_traces(plot_to_add.data)
    fig.data[-1].name = name
    fig.update_traces(marker=dict(size=13, opacity=1, color=color,
                      symbol="star"), selector=dict(name=name))
    hovertemplate = ' <br>'.join(
        [f'{col}: %{{customdata[{i}]:.2f}}' for i, col in enumerate(hoverdata_multi_columns)])
    fig.update_traces(hovertemplate=hovertemplate, selector=dict(name=name))

    return fig


# def find_trajectory_ref_row(trajectory_df, ref_point_mode, ref_point_value):
#     if ref_point_mode == 'time':
#         trajectory_ref_rows = trajectory_df[trajectory_df['rel_time'] <= ref_point_value]
#     elif ref_point_mode == 'distance':
#         trajectory_ref_rows = trajectory_df[trajectory_df['rel_distance'] <= ref_point_value]
#     else:
#         trajectory_ref_rows = trajectory_df.copy()
#     if len(trajectory_ref_rows) == 0:
#         trajectory_ref_row = trajectory_df.iloc[0]
#     else:
#         trajectory_ref_row = trajectory_ref_rows.iloc[-1]
#     return trajectory_ref_row


def mark_reference_point_in_monkey_plot(fig, trajectory_ref_row):
    fig.add_trace(go.Scatter(x=[trajectory_ref_row['monkey_x']], y=[trajectory_ref_row['monkey_y']],
                             mode='markers',
                             name='reference point',
                             marker=dict(color="LightSeaGreen",
                                         size=10, opacity=1, symbol="circle"),
                             hoverinfo='name',
                             ))
    return fig


def connect_points_to_points(fig, connect_path_ff_df, show_traj_points_when_making_lines=True, hoverdata_multi_columns=['rel_time']):

    connect_path_ff_df = connect_path_ff_df.copy()
    ff_component = connect_path_ff_df[['ff_x', 'ff_y', 'counter']].copy().rename(
        columns={'ff_x': 'x', 'ff_y': 'y'})
    monkey_component = connect_path_ff_df[['monkey_x', 'monkey_y', 'counter']].copy(
    ).rename(columns={'monkey_x': 'x', 'monkey_y': 'y'})
    new_connect_path_ff_df = pd.concat(
        [ff_component, monkey_component], axis=0)

    fig_traces = px.line(new_connect_path_ff_df,
                         x='x',
                         y='y',
                         color_discrete_sequence=['rgb(173, 216, 230, 0.5)'],
                         color='counter',
                         )

    fig.add_traces(list(fig_traces.select_traces()))

    fig.data[-1].name = 'connect_path_ff'
    fig.update_traces(opacity=.2, selector=dict(name='connect_path_ff'))

    # also mark the ending points of the lines on the trajectory
    if show_traj_points_when_making_lines:
        plot_to_add = px.scatter(connect_path_ff_df, x='monkey_x', y='monkey_y',
                                 color_discrete_sequence=['blue'],
                                 hover_data=hoverdata_multi_columns,
                                 labels={'monkey_x': 'monkey x after rotation (cm)',
                                         'monkey_y': 'monkey y after rotation (cm)',
                                         'rel_time': 'relative time (s)',
                                         'rel_distance': 'relative distance (cm)'},
                                 custom_data=hoverdata_multi_columns
                                 )

        fig.add_traces(plot_to_add.data)
        fig.data[-1].name = 'connect_path_ff_2'
        fig.update_traces(marker=dict(size=3),
                          selector={'name': 'connect_path_ff_2'})

    fig.update_layout(showlegend=False)

    return fig


def plot_a_portion_of_trajectory_to_show_traj_portion(fig, traj_portion, color='purple', hoverdata_multi_columns=['rel_time'], linewidth=9):

    plot_to_add = px.scatter(traj_portion, x='monkey_x', y='monkey_y',
                             hover_data=hoverdata_multi_columns,
                             labels={'monkey_x': 'monkey x after rotation (cm)',
                                     'monkey_y': 'monkey y after rotation (cm)',
                                     'rel_time': 'relative time (s)',
                                     'rel_distance': 'relative distance (cm)'},
                             custom_data=hoverdata_multi_columns,
                             color_discrete_sequence=[color])
    fig.add_traces(plot_to_add.data)
    fig.data[-1].name = 'to_show_traj_portion'
    hovertemplate = ' <br>'.join(
        [f'{col}: %{{customdata[{i}]:.2f}}' for i, col in enumerate(hoverdata_multi_columns)])
    fig.update_traces(marker=dict(size=linewidth, opacity=1),
                      hovertemplate=hovertemplate,
                      selector=dict(name='to_show_traj_portion'))

    return fig


def plot_horizontal_lines_to_show_ff_visible_segments_plotly(fig, ff_info, monkey_information, rotation_matrix, x0, y0,
                                                             how_to_show_ff='square', unique_ff_indices=None, varying_colors=None):
    """
    This function plots horizontal lines to show visible segments of fireflies (ff) using Plotly.
    It also shows the ff position as a square or a circle.

    Parameters:
    ff_info (DataFrame): The DataFrame containing firefly information.
    monkey_information (dict): The dictionary containing monkey information.
    rotation_matrix (ndarray): The rotation matrix to use.
    x0, y0 (float): The coordinates of the origin.
    how_to_show_ff (str): How to show the firefly. Options are 'square' or 'circle'. Default is 'square'.
    unique_ff_indices (list): The list of unique firefly indices. If None, it will be set to all unique indices in ff_info. Default is None.

    Returns:
    fig (object): The Plotly figure object with the plot.
    """

    # calculate a reasonable point_index_gap_threshold_to_sep_vis_intervals
    dt = monkey_information['dt'].median()
    point_index_gap_threshold_to_sep_vis_intervals = max(2, int(0.1 / dt))

    # Set unique_ff_indices to all unique indices in ff_info if not provided
    # unique_ff_indices = ff_info.ff_index.unique() if unique_ff_indices is None else np.array(unique_ff_indices)
    if unique_ff_indices is None:
        unique_ff = ff_info[['ff_index', 'ff_number']].drop_duplicates()
    else:
        unique_ff = ff_info[ff_info['ff_index'].isin(unique_ff_indices)][[
            'ff_index', 'ff_number']].drop_duplicates()
    unique_ff.sort_values(by='ff_number', ascending=True, inplace=True)
    unique_ff.reset_index(drop=True, inplace=True)

    # # Define color palette, avoiding red
    # varying_colors = np.concatenate([sns.color_palette("tab10", 3), sns.color_palette("tab10", 14)[4:]], axis=0)
    # # Convert it to hex for the plotly plot
    # varying_colors = [matplotlib.colors.rgb2hex(color) for color in varying_colors]

    if varying_colors is None:
        varying_colors = ['#33BBFF', '#FF337D', '#FF33D7', '#8D33FF', '#33FF64',
                          '#FF5733', '#FFB533', '#33FFBE', '#3933FF', '#FF3346',
                          '#FC33FF', '#FFEC33', '#FF5E33', '#B06B58']

    # Iterate over unique firefly indices
    # for i, ff_index in enumerate(unique_ff_indices):
    for index, row in unique_ff.iterrows():
        ff_number = row.ff_number
        ff_index = row.ff_index

        if ff_number <= 2:
            visible = True
        else:
            visible = 'legendonly'

        # Define color for current firefly
        color = varying_colors[index % len(varying_colors)]

        # Extract and sort data for current firefly
        temp_df = ff_info[ff_info['ff_index'] == ff_index].copy().sort_values(by=[
            'point_index'])

        # rotated firefly position
        ff_position_rotated = np.matmul(
            rotation_matrix, temp_df[['ff_x', 'ff_y']].drop_duplicates().values.T)

        # Find breaking points of visible segments
        all_point_index = temp_df.point_index.values
        all_breaking_points = np.where(np.diff(
            all_point_index) >= point_index_gap_threshold_to_sep_vis_intervals)[0] + 1

        # Find positions of the ends of the perpendicular lines at starting and ending points
        perp_dict = plot_behaviors_utils.find_dict_of_perpendicular_lines_to_monkey_trajectory_at_certain_points(
            all_point_index, all_breaking_points, monkey_information, rotation_matrix)

        # Show firefly position
        shared_kwargs = dict(mode='markers', legendgroup='ff '+str(ff_number),
                             customdata=[[ff_number]], name='ff '+str(ff_number), visible=visible)
        if how_to_show_ff == 'square':
            fig.add_trace(go.Scatter(x=ff_position_rotated[0]-x0, y=ff_position_rotated[1]-y0,
                                     marker=dict(symbol='square', color=color, size=15), **shared_kwargs))
        elif how_to_show_ff == 'circle':
            fig.add_trace(go.Scatter(x=ff_position_rotated[0]-x0, y=ff_position_rotated[1]-y0,
                                     marker=dict(symbol='circle', color=color, size=15), **shared_kwargs))

        # Find and plot beginning and end of each visible segment
        for j in range(len(all_breaking_points)+1):
            one_perp_dict = plot_behaviors_utils.find_one_pair_of_perpendicular_lines(
                perp_dict, j, x0, y0)

            # Plot points when firefly starts being visible
            if j == 0:
                showlegend = True
            else:
                showlegend = False

            fig.add_trace(go.Scatter(x=[one_perp_dict['starting_left_x'], one_perp_dict['starting_right_x']],
                                     y=[one_perp_dict['starting_left_y'],
                                         one_perp_dict['starting_right_y']],
                                     mode='lines', line=dict(color=color, width=2.5), legendgroup='ff '+str(ff_number),
                                     customdata=[[ff_number]
                                                 ], showlegend=showlegend,
                                     name='ff '+str(ff_number) + ' starts visible', visible=visible, opacity=0.7),
                          )

            # Plot points when firefly stops being visible
            fig.add_trace(go.Scatter(x=[one_perp_dict['ending_left_x'], one_perp_dict['ending_right_x']],
                                     y=[one_perp_dict['ending_left_y'],
                                         one_perp_dict['ending_right_y']],
                                     mode='lines', line=dict(color=color, width=3.5, dash='dot'), legendgroup='ff '+str(ff_number),
                                     customdata=[
                                         [ff_number]], showlegend=showlegend, name='ff '+str(ff_number) + ' stops visible',
                                     visible=visible, opacity=0.7),
                          )

        fig.update_traces(hovertemplate='ff %{customdata[0]}',
                          selector=dict(name='ff '+str(ff_number)))
        fig.update_traces(hovertemplate='ff %{customdata[0]}',
                          selector=dict(name='ff '+str(ff_number) + ' starts visible'))
        fig.update_traces(hovertemplate='ff %{customdata[0]}',
                          selector=dict(name='ff '+str(ff_number) + ' stops visible'))

    return fig


def plot_triangles_to_show_monkey_heading_in_xy_in_plotly(fig, triangle_df, trace_name_prefix=None, color='red', point_index=None, linewidth=2):
    if point_index is not None:
        triangle_df = triangle_df[triangle_df['point_index'] == point_index]
    triangle_x_rotated = triangle_df[['x_0', 'x_1', 'x_2']].values
    triangle_y_rotated = triangle_df[['y_0', 'y_1', 'y_2']].values

    for j in range(triangle_x_rotated.shape[0]):
        if trace_name_prefix is None:
            name = 'triangle_to_show_monkey_heading_' + str(j)
        else:
            name = trace_name_prefix + str(j)
        fig.add_trace(
            go.Scatter(x=triangle_x_rotated[j], y=triangle_y_rotated[j],
                       mode='lines',
                       name=name,
                       hoverinfo='name',
                       showlegend=False,
                       line=dict(color=color,
                                 width=linewidth)),
        )
    return fig
