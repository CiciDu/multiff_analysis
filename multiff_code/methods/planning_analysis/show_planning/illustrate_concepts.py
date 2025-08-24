
import numpy as np
import plotly.graph_objects as go
from null_behaviors import show_null_trajectory
from planning_analysis.plan_indicators import diff_in_curv_utils
from visualization.plotly_tools import plotly_for_null_arcs, plotly_for_monkey


def plot_with_additional_elements(snf,
                                  line_of_stop_heading, line_stop_nxt_ff,
                                  arc_xy,
                                  arc_label='Angle from monkey stop to next ff'
                                  ):

    # Add traces to the figure
    traces = [
        go.Scatter(
            x=line_of_stop_heading[0], y=line_of_stop_heading[1],
            mode='lines', line=dict(color='blue', width=2, dash='dash'),
            name='Stop Heading'
        ),
        go.Scatter(
            x=arc_xy[0], y=arc_xy[1],
            mode='lines', line=dict(color='LightSeaGreen', width=5),
            name='Arc between stop heading and next ff'
        ),
        go.Scatter(
            x=[(arc_xy[0, 0] + arc_xy[0, 1]) / 2 + 210],
            y=[(arc_xy[1, 0] + arc_xy[1, 1]) / 2],
            mode="text", name="Text",
            text=[arc_label],
            textposition="bottom center",
            textfont=dict(family="sans serif", size=18, color="LightSeaGreen")
        ),
        go.Scatter(
            x=line_stop_nxt_ff[0], y=line_stop_nxt_ff[1],
            mode='lines', line=dict(color='blue', width=2, dash='dash'),
            name='Next FF'
        )
    ]

    # Add all traces to the figure
    for trace in traces:
        snf.fig.add_trace(trace)

    _update_layout(snf)


def _update_layout(snf):
    # Update the layout to make the background very light grey
    snf.fig.update_layout(
        plot_bgcolor='white',
        # Remove x-axis grid lines and zero line
        xaxis=dict(showgrid=False, zeroline=False),
        # Remove y-axis grid lines and zero line
        yaxis=dict(showgrid=False, zeroline=False)
    )

    # update fig size so that it is 1.5 times as wide as it is tall
    snf.fig.update_layout(
        width=snf.fig.layout.height * 1.5
    )


def _calculate_rotated_line(start_x, start_y, end_x, end_y, rotation_matrix):
    """
    Calculate the rotated coordinates of a line given its start and end points and a rotation matrix.

    Parameters:
    start_x (float): The x-coordinate of the start point.
    start_y (float): The y-coordinate of the start point.
    end_x (float): The x-coordinate of the end point.
    end_y (float): The y-coordinate of the end point.
    rotation_matrix (np.ndarray): The rotation matrix to apply.

    Returns:
    np.ndarray: The rotated coordinates of the line.
    """
    traj_xy = np.array([[start_x, end_x], [start_y, end_y]])
    traj_xy_rotated = np.matmul(rotation_matrix, traj_xy)
    return traj_xy_rotated


def find_line_of_heading(stop_x, stop_y, monkey_angle, line_length=150, rotation_matrix=None):
    """
    Calculate the line representing the monkey's heading direction at the stop point.
    """

    # Calculate the end points of the heading line
    traj_x = stop_x - np.cos(monkey_angle) * line_length
    traj_y = stop_y - np.sin(monkey_angle) * line_length
    traj_x2 = stop_x + np.cos(monkey_angle) * line_length
    traj_y2 = stop_y + np.sin(monkey_angle) * line_length

    # Apply rotation matrix to the heading line
    if rotation_matrix is None:
        rotation_matrix = np.eye(2)
    return _calculate_rotated_line(traj_x, traj_y, traj_x2, traj_y2, rotation_matrix)


def _find_line_between_points(snf, fixed_current_i, start_point, end_point):
    """
    Calculate the line between two points.

    Parameters:
    snf (object): The object containing the data and plotly instance.
    fixed_current_i (int): The index of the current stop.
    start_point (str): The column name of the start point.
    end_point (str): The column name of the end point.

    Returns:
    np.ndarray: The rotated coordinates of the line between the two points.
    """
    # Extract coordinates of the start and end points
    stops_near_ff_row = snf.stops_near_ff_df_counted.iloc[fixed_current_i]
    start_x = stops_near_ff_row[start_point + '_x']
    start_y = stops_near_ff_row[start_point + '_y']
    end_x = stops_near_ff_row[end_point + '_x']
    end_y = stops_near_ff_row[end_point + '_y']

    # Apply rotation matrix to the line
    rotation_matrix = snf.current_plotly_key_comp['rotation_matrix']
    return _calculate_rotated_line(start_x, start_y, end_x, end_y, rotation_matrix)


def find_line_between_cur_ff_and_nxt_ff(snf, fixed_current_i):
    """
    Calculate the line between the current firefly and the next firefly.

    Parameters:
    snf (object): The object containing the data and plotly instance.
    fixed_current_i (int): The index of the current stop.

    Returns:
    np.ndarray: The rotated coordinates of the line between the current and next firefly.
    """
    return _find_line_between_points(snf, fixed_current_i, 'cur_ff', 'nxt_ff')


def find_line_between_stop_and_nxt_ff(snf, fixed_current_i):
    """
    Calculate the line between the stop point and the next firefly.

    Parameters:
    snf (object): The object containing the data and plotly instance.
    fixed_current_i (int): The index of the current stop.

    Returns:
    np.ndarray: The rotated coordinates of the line between the stop point and the next firefly.
    """
    return _find_line_between_points(snf, fixed_current_i, 'stop', 'nxt_ff')


def calculate_arc_to_show_angle(arc_center, arc_end_1, arc_end_2, arc_radius, rotation_matrix):
    """
    Calculate the coordinates of an arc between two lines.

    Parameters:
    line_stop_nxt_ff (np.ndarray): Coordinates of the line between the stop and the next firefly.
    line_of_stop_heading (np.ndarray): Coordinates of the line representing the monkey's heading direction.
    line_length (float): The length of the line.
    rotation_matrix (np.ndarray): The rotation matrix to apply.

    Returns:
    np.ndarray: The rotated coordinates of the arc.
    """

    arc_starting_angle = np.arctan2(
        arc_end_1[1] - arc_center[1], arc_end_1[0] - arc_center[0])
    arc_ending_angle = np.arctan2(
        arc_end_2[1] - arc_center[1], arc_end_2[0] - arc_center[0])
    arc_theta_samples = np.linspace(arc_starting_angle, arc_ending_angle, 500)
    arc_x = arc_center[0] + arc_radius * np.cos(arc_theta_samples)
    arc_y = arc_center[1] + arc_radius * np.sin(arc_theta_samples)

    arc_xy = np.stack([arc_x, arc_y])
    if rotation_matrix is None:
        rotation_matrix = np.eye(2)
    arc_xy_rotated = np.matmul(rotation_matrix, arc_xy)

    return arc_xy_rotated


def prepare_to_show_angle_from_monkey_stop_to_next_ff(snf, fixed_current_i):
    line_stop_nxt_ff = find_line_between_stop_and_nxt_ff(snf, fixed_current_i)

    # Calculate the length of the line
    line_length = np.linalg.norm(
        line_stop_nxt_ff[:, 0] - line_stop_nxt_ff[:, 1])

    # Calculate the line representing the monkey's heading direction at the stop point
    # Extract stop coordinates and monkey's heading angle
    stop_x = snf.stops_near_ff_df_counted.iloc[fixed_current_i]['stop_x']
    stop_y = snf.stops_near_ff_df_counted.iloc[fixed_current_i]['stop_y']
    monkey_angle = snf.stops_near_ff_df_counted.iloc[fixed_current_i]['stop_monkey_angle']

    line_of_stop_heading = find_line_of_heading(stop_x, stop_y, monkey_angle,
                                                line_length=line_length, rotation_matrix=snf.current_plotly_key_comp[
                                                    'rotation_matrix']
                                                )

    arc_radius = line_length / 2
    arc_center = line_stop_nxt_ff[:, 0]
    arc_end_1 = line_stop_nxt_ff[:, 1]
    arc_end_2 = line_of_stop_heading[:, 1]
    arc_xy = calculate_arc_to_show_angle(
        arc_center, arc_end_1, arc_end_2, arc_radius, None
    )

    return line_stop_nxt_ff, line_of_stop_heading, arc_xy


def prepare_to_show_angle_from_null_arc_end_to_next_ff(snf, fixed_current_i):
    # Calculate the line between the stop point and the next firefly
    line_stop_nxt_ff = find_line_between_stop_and_nxt_ff(snf, fixed_current_i)
    # Calculate the length of the line
    line_length = np.linalg.norm(
        line_stop_nxt_ff[:, 0] - line_stop_nxt_ff[:, 1])

    # Prepare to draw lines
    null_arc_info = snf.cur_null_arc_info_for_the_point
    null_arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(null_arc_info.loc[fixed_current_i, 'center_x'], null_arc_info.loc[fixed_current_i, 'center_y'], null_arc_info.loc[fixed_current_i, 'all_arc_radius'],
                                                                   null_arc_info.loc[fixed_current_i, 'arc_starting_angle'], null_arc_info.loc[fixed_current_i, 'arc_ending_angle'], rotation_matrix=None)

    null_arc_end_x = null_arc_xy_rotated[0, -1]
    null_arc_end_y = null_arc_xy_rotated[1, -1]
    null_arc_end_angle = snf.mheading_for_cur_ff_for_all_counted_points[
        'monkey_angle'][fixed_current_i, 1]

    line_of_cur_null_heading = find_line_of_heading(
        null_arc_end_x, null_arc_end_y, null_arc_end_angle, line_length=line_length, rotation_matrix=snf.current_plotly_key_comp['rotation_matrix'])

    line_cur_and_nxt_ff = _calculate_rotated_line(null_arc_end_x, null_arc_end_y,
                                                  snf.stops_near_ff_df_counted.iloc[fixed_current_i]['nxt_ff_x'],
                                                  snf.stops_near_ff_df_counted.iloc[fixed_current_i]['nxt_ff_y'],
                                                  snf.current_plotly_key_comp['rotation_matrix'])

    # Prepare to draw a small arc to show the angle from null arc end to next ff
    arc_radius = line_length / 2
    arc_center = line_cur_and_nxt_ff[:, 0]
    arc_end_1 = line_cur_and_nxt_ff[:, 1]
    arc_end_2 = line_of_cur_null_heading[:, 1]
    arc_xy = calculate_arc_to_show_angle(
        arc_center, arc_end_1, arc_end_2, arc_radius, None
    )

    return line_of_cur_null_heading, line_cur_and_nxt_ff, arc_xy


# def find_arc_info_for_plotting(null_arc_curv_df, stops_near_ff_row, nxt_ff_df_modified, monkey_information):
#     stop_point_index = stops_near_ff_row['stop_point_index']
#     ref_point_index = nxt_ff_df_modified.loc[nxt_ff_df_modified['stop_point_index']==stop_point_index, 'point_index'].item()
#     arc_info_for_plotting = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
#                 null_arc_curv_df[null_arc_curv_df['ref_point_index']==ref_point_index], monkey_information)
#     # arc_info = arc_info_for_plotting[arc_info_for_plotting['arc_point_index']==ref_point_index]
#     return arc_info_for_plotting


def prepare_to_illustrate_diff_in_d_curv(snf):

    if not hasattr(snf, 'diff_in_curv_df'):
        # doing this can automatically compute the null_arc_curv_df and monkey_curv_df
        snf.make_diff_in_curv_df()

    stop_point_index = snf.stops_near_ff_row['stop_point_index']
    ref_point_index = snf.nxt_ff_df_modified.loc[snf.nxt_ff_df_modified['stop_point_index']
                                                 == stop_point_index, 'point_index'].item()

    # get the arc from cur arc end to next ff
    if (not hasattr(snf, 'null_arc_curv_df')) or (not hasattr(snf, 'monkey_curv_df')):
        snf.make_diff_in_curv_df()
    arc_from_cur_arc_end_to_next_ff = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
        snf.null_arc_curv_df[snf.null_arc_curv_df['ref_point_index'] == ref_point_index], snf.monkey_information)
    # arc_from_cur_arc_end_to_next_ff['opt_arc_curv'] = snf.null_arc_curv_df.loc[snf.null_arc_curv_df['ref_point_index']==ref_point_index, 'opt_arc_curv'].values[0]

    arc_from_stop_to_nxt_ff = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
        snf.monkey_curv_df[snf.monkey_curv_df['ref_point_index'] == ref_point_index], snf.monkey_information)
    # also find curv_of_traj_before_stop
    # arc_from_stop_to_nxt_ff['opt_arc_curv'] = snf.monkey_curv_df.loc[snf.monkey_curv_df['ref_point_index']==ref_point_index, 'opt_arc_curv'].values[0]

    # get the trajectory before stop
    trajectory_df = snf.current_plotly_key_comp['trajectory_df'].copy()
    window_for_curv_of_traj = [-25, 0]
    trajectory_df.sort_values(by='rel_distance', inplace=True)
    traj_portion_before_stop = trajectory_df[trajectory_df['rel_distance'].between(
        window_for_curv_of_traj[0], window_for_curv_of_traj[1])].copy()
    traj_portion_before_stop['curv_of_traj'] = snf.monkey_curv_df[snf.monkey_curv_df['ref_point_index']
                                                                  == ref_point_index]['curv_of_traj'].values[0]

    return arc_from_cur_arc_end_to_next_ff, arc_from_stop_to_nxt_ff, traj_portion_before_stop


# def illustrate_diff_in_d_curv(snf):
#     arc_from_cur_arc_end_to_next_ff, arc_from_stop_to_nxt_ff, traj_portion_before_stop, curv_of_traj_before_stop = prepare_to_illustrate_diff_in_d_curv(snf)

#     # plot the cur null arc (deep orange)
#     snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
#         snf.fig, snf.cur_null_arc_info_for_the_point,
#         rotation_matrix=snf.current_plotly_key_comp['rotation_matrix'],
#         color='#ff6f00', trace_name='cur null arc',
#         linewidth=3, opacity=0.9
#     )

#     # plot the cur arc end to next ff null arc (bright cyan)
#     snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
#         snf.fig, arc_from_cur_arc_end_to_next_ff,
#         rotation_matrix=snf.current_plotly_key_comp['rotation_matrix'],
#         color='#00bcd4', trace_name='cur arc end to next ff null arc',
#         linewidth=3, opacity=0.9
#     )

#     # plot the traj portion before stop (dark navy)
#     snf.fig = plotly_for_monkey.plot_a_portion_of_trajectory_to_show_traj_portion(
#         snf.fig, traj_portion_before_stop,
#         color='#1f3b73',
#         hoverdata_multi_columns=['rel_time'], linewidth=7
#     )

#     # plot the monkey stop to nxt ff null arc (deep purple)
#     snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
#         snf.fig, arc_from_stop_to_nxt_ff,
#         rotation_matrix=snf.current_plotly_key_comp['rotation_matrix'],
#         color='#8e44ad', trace_name='monkey stop to nxt ff null arc',
#         linewidth=2, opacity=0.9
#     )

#     return snf.fig

def illustrate_diff_in_d_curv(snf):
    arc_from_cur_arc_end_to_next_ff, arc_from_stop_to_nxt_ff, traj_portion_before_stop = \
        prepare_to_illustrate_diff_in_d_curv(snf)

    rot = snf.current_plotly_key_comp['rotation_matrix']

    # ---------------------------
    # Pair A (orange family)
    # ---------------------------
    color_A_main = '#D55E00'  # deep vermillion (solid)
    color_A_alt = '#FF7F0E'  # vivid orange (dashed)
    legend_A = 'Pair A: current target arcs'

    # A1: Current null arc (solid)
    snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
        snf.fig, snf.cur_null_arc_info_for_the_point,
        rotation_matrix=rot, color=color_A_main,
        trace_name='cur ff null arc', linewidth=3, opacity=0.95,
        dash='solid', legendgroup=legend_A
    )

    # A2: Cur arc end -> next ff (dashed)
    snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
        snf.fig, arc_from_cur_arc_end_to_next_ff,
        rotation_matrix=rot, color=color_A_alt,
        trace_name='cur arc end → next ff null arc', linewidth=3, opacity=0.95,
        dash='dash', legendgroup=legend_A
    )

    # Δκ for Pair A
    # curvature of the current null arc
    kA1 = snf.cur_null_arc_info_for_the_point['opt_arc_curv'].values[0]
    # curvature of the cur arc end to next ff null arc
    kA2 = arc_from_cur_arc_end_to_next_ff['opt_arc_curv'].values[0]

    # ---------------------------
    # Pair B (blue family)
    # ---------------------------
    color_B_main = '#1F78B4'   # navy (trajectory portion)
    color_B_alt = '#00A8E8'   # vivid azure (stop→next, dashed)
    legend_B = 'Pair B: movement vs stop→next'

    # B1: trajectory before stop (solid)
    snf.fig = plotly_for_monkey.plot_a_portion_of_trajectory_to_show_traj_portion(
        snf.fig, traj_portion_before_stop,
        trace_name='trajectory before stop',
        color=color_B_main, hoverdata_multi_columns=['rel_time'], linewidth=7
    )
    # (keeps your markers; visually distinct color family)

    # B2: Monkey stop -> next ff null arc (dashed)
    snf.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(
        snf.fig, arc_from_stop_to_nxt_ff,
        rotation_matrix=rot, color=color_B_alt,
        trace_name='stop → next ff null arc', linewidth=3, opacity=0.95,
        dash='dash', legendgroup=legend_B
    )

    # curvature of the monkey stop to next ff null arc
    kB2 = arc_from_stop_to_nxt_ff['opt_arc_curv'].values[0]
    kB1 = traj_portion_before_stop['curv_of_traj'].values[0]

    # ---------------------------
    # Contrast the pairs (Δκ(A) vs Δκ(B))
    # ---------------------------
    if not np.isnan(kA1) and not np.isnan(kA2) and not np.isnan(kB2) and not np.isnan(kB1):
        # we'll use deg/m as the unit for curvature
        kA1 = kA1 * 180 / np.pi * 100
        kA2 = kA2 * 180 / np.pi * 100
        kB2 = kB2 * 180 / np.pi * 100
        kB1 = kB1 * 180 / np.pi * 100

        dA = kA2 - kA1
        dB = kB2 - kB1
        contrast = abs(dA) - abs(dB)

        # annotate individual values
        # snf.fig = _add_corner_annotation(snf.fig, f"kA1 = {kA1:.3f}", color='#E66100', corner='top-right')
        # snf.fig = _add_corner_annotation(snf.fig, f"kA2 = {kA2:.3f}", color='#E66100', corner='top-right')
        # snf.fig = _add_corner_annotation(snf.fig, f"kB2 = {kB2:.3f}", color='#1F78B4', corner='top-right')
        # snf.fig = _add_corner_annotation(snf.fig, f"kB1 = {kB1:.3f}", color='#1F78B4', corner='top-right')

        # snf.fig = _add_corner_annotation(snf.fig, f"Δκ(A) = {dA:.3f}", color='#E66100', corner='top-right')
        # snf.fig = _add_corner_annotation(snf.fig, f"Δκ(B) = {dB:.3f}", color='#1F78B4', corner='top-right')
        # snf.fig = _add_corner_annotation(snf.fig, f"Contrast = {contrast:.3f}", color='#333333', corner='top-right')

        minus = "\u2212"  # Unicode minus

        summary_html = (
            f"<b style='font-size:13px'>Curvature Δκ</b> "
            f"<span style='font-size:11px; opacity:.7'>(deg/m)</span><br>"
            f"<span style='color:{color_A_main}'>■</span>&nbsp;Δκ (null): <b>{dA:.3f}</b><br>"
            f"<span style='color:{color_B_main}'>■</span>&nbsp;Δκ (monkey): <b>{dB:.3f}</b><br>"
            f"<span style='opacity:.85'>|Δκ(null)| {minus} |Δκ(monkey)| = </span>"
            f"<b>{contrast:.3f}</b>"
        )

        snf.fig = _add_corner_annotation(
            snf.fig, summary_html, color='#222',
            corner='top-left', bordercolor='#888', font_size=12
        )

    # Cleaner legend grouping names
    snf.fig.update_layout(legend_title_text='Arc groups')
    _update_layout(snf)

    return snf.fig


def _add_corner_annotation(
    fig,
    text,
    color='#222',
    corner='top-right',   # 'top-right', 'top-left', 'bottom-right', 'bottom-left'
    vspace=0.055,         # vertical spacing between stacked notes
    # (rarely needed) horizontal spacing if you want columns
    hspace=0.0,
    bgcolor='rgba(255,255,255,0.7)',
    bordercolor='#888',
    borderwidth=1,
    font_size=12,
):
    # Corner presets
    corners = {
        'top-right': {'x': 0.98, 'y': 0.98, 'xanchor': 'right', 'yanchor': 'top',    'dir': -1},
        'top-left': {'x': 0.02, 'y': 0.98, 'xanchor': 'left',  'yanchor': 'top',    'dir': -1},
        # 'top-left'    : {'x': 0.1, 'y': 0.8, 'xanchor': 'left',  'yanchor': 'top',    'dir': -1},
        'bottom-right': {'x': 0.98, 'y': 0.02, 'xanchor': 'right', 'yanchor': 'bottom', 'dir': +1},
        'bottom-left': {'x': 0.02, 'y': 0.02, 'xanchor': 'left',  'yanchor': 'bottom', 'dir': +1},
    }
    cfg = corners[corner]
    base_x, base_y, xanchor, yanchor, sign = cfg['x'], cfg['y'], cfg['xanchor'], cfg['yanchor'], cfg['dir']

    # Count existing notes already pinned to this corner (so we can stack)
    existing = 0
    if hasattr(fig.layout, "annotations") and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if (
                getattr(ann, "xref", None) == "paper" and
                getattr(ann, "yref", None) == "paper" and
                getattr(ann, "xanchor", None) == xanchor and
                getattr(ann, "yanchor", None) == yanchor and
                isinstance(getattr(ann, "x", None), (int, float)) and
                isinstance(getattr(ann, "y", None), (int, float)) and
                abs(ann.x - base_x) <= 0.03  # near our corner column
            ):
                existing += 1

    # Compute stacked position
    x_pos = base_x + hspace * (existing)  # usually unchanged
    y_pos = base_y + sign * vspace * (existing)

    # Clamp inside plot area
    x_pos = max(0.02, min(0.98, x_pos))
    y_pos = max(0.02, min(0.98, y_pos))

    fig.add_annotation(
        x=x_pos, y=y_pos,
        xref='paper', yref='paper',
        xanchor=xanchor, yanchor=yanchor,
        showarrow=False,
        text=text,
        font=dict(size=font_size, color=color),
        bgcolor=bgcolor,
        bordercolor=bordercolor, borderwidth=borderwidth,
        borderpad=8
    )
    return fig
