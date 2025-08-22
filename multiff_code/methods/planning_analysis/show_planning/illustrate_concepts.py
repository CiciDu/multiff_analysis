
import numpy as np
import plotly.graph_objects as go
from null_behaviors import show_null_trajectory


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
    line_length = np.linalg.norm(line_stop_nxt_ff[:, 0] - line_stop_nxt_ff[:, 1])

    # Prepare to draw lines 
    null_arc_info = snf.cur_null_arc_info_for_the_point    
    null_arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(null_arc_info.loc[fixed_current_i, 'center_x'], null_arc_info.loc[fixed_current_i, 'center_y'], null_arc_info.loc[fixed_current_i, 'all_arc_radius'],
                                                                    null_arc_info.loc[fixed_current_i, 'arc_starting_angle'], null_arc_info.loc[fixed_current_i, 'arc_ending_angle'], rotation_matrix=None)

    null_arc_end_x = null_arc_xy_rotated[0, -1]
    null_arc_end_y = null_arc_xy_rotated[1, -1]
    null_arc_end_angle = snf.mheading_for_cur_ff_for_all_counted_points['monkey_angle'][fixed_current_i, 1]

    line_of_cur_null_heading = find_line_of_heading(null_arc_end_x, null_arc_end_y, null_arc_end_angle, line_length=line_length, rotation_matrix=snf.current_plotly_key_comp['rotation_matrix'])

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