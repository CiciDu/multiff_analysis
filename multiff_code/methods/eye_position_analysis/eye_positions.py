
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import math
import pandas as pd


def convert_eye_positions_in_monkey_information(monkey_information, add_left_and_right_eyes_info=False, interocular_dist=4):
    monkey_height = -10
    body_x = np.array(monkey_information.monkey_x)
    body_y = np.array(monkey_information.monkey_y)
    try:
        monkey_angle = np.array(monkey_information.monkey_angles)
    except AttributeError:
        monkey_angle = np.array(monkey_information.monkey_angle)

    # left eye
    ver_theta = np.array(monkey_information.LDz)*pi/180
    hor_theta = np.array(monkey_information.LDy)*pi/180
    gaze_mky_view_x_l, gaze_mky_view_y_l, gaze_mky_view_angle_l, gaze_world_x_l, gaze_world_y_l \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=interocular_dist, left_or_right_eye='left')

    # right eye
    ver_theta = np.array(monkey_information.RDz)*pi/180
    hor_theta = np.array(monkey_information.RDy)*pi/180
    gaze_mky_view_x_r, gaze_mky_view_y_r, gaze_mky_view_angle_r, gaze_world_x_r, gaze_world_y_r \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=interocular_dist, left_or_right_eye='right')

    # average the two eyes
    gaze_mky_view_x = (gaze_mky_view_x_l + gaze_mky_view_x_r)/2
    gaze_mky_view_y = (gaze_mky_view_y_l + gaze_mky_view_y_r)/2
    gaze_mky_view_angle = (
        gaze_mky_view_angle_l + gaze_mky_view_angle_r)/2
    gaze_world_x = (gaze_world_x_l + gaze_world_x_r)/2
    gaze_world_y = (gaze_world_y_l + gaze_world_y_r)/2

    monkey_information['gaze_mky_view_x'] = gaze_mky_view_x
    monkey_information['gaze_mky_view_y'] = gaze_mky_view_y
    monkey_information['gaze_mky_view_angle'] = gaze_mky_view_angle
    monkey_information['gaze_world_x'] = gaze_world_x
    monkey_information['gaze_world_y'] = gaze_world_y

    if add_left_and_right_eyes_info:
        monkey_information['gaze_mky_view_x_l'] = gaze_mky_view_x_l
        monkey_information['gaze_mky_view_y_l'] = gaze_mky_view_y_l
        monkey_information['gaze_mky_view_angle_l'] = gaze_mky_view_angle_l
        monkey_information['gaze_world_x_l'] = gaze_world_x_l
        monkey_information['gaze_world_y_l'] = gaze_world_y_l

        monkey_information['gaze_mky_view_x_r'] = gaze_mky_view_x_r
        monkey_information['gaze_mky_view_y_r'] = gaze_mky_view_y_r
        monkey_information['gaze_mky_view_angle_r'] = gaze_mky_view_angle_r
        monkey_information['gaze_world_x_r'] = gaze_world_x_r
        monkey_information['gaze_world_y_r'] = gaze_world_y_r
    return monkey_information


def average_and_then_convert_eye_positions_in_monkey_information(monkey_information, add_suffix_to_new_columns=True):
    monkey_height = -10
    body_x = np.array(monkey_information.monkey_x)
    body_y = np.array(monkey_information.monkey_y)
    monkey_angle = np.array(monkey_information.monkey_angle)

    ver_theta = (np.array(monkey_information.LDz) +
                 np.array(monkey_information.RDz))*pi/180/2
    hor_theta = (np.array(monkey_information.LDy) +
                 np.array(monkey_information.RDy))*pi/180/2
    gaze_mky_view_x_avg, gaze_mky_view_y_avg, gaze_mky_view_angle_avg, gaze_world_x_avg, gaze_world_y_avg \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=0)

    if add_suffix_to_new_columns:
        suffix = '_avg'
    else:
        suffix = ''
    monkey_information['gaze_mky_view_x'+suffix] = gaze_mky_view_x_avg
    monkey_information['gaze_mky_view_y'+suffix] = gaze_mky_view_y_avg
    monkey_information['gaze_mky_view_angle' +
                       suffix] = gaze_mky_view_angle_avg
    monkey_information['gaze_world_x'+suffix] = gaze_world_x_avg
    monkey_information['gaze_world_y'+suffix] = gaze_world_y_avg

    return monkey_information


def apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                          interocular_dist=4, left_or_right_eye='left',
                                                          rotate_world_xy_based_on_m_angle_to_get_abs_coord=True):
    # This uses the formulas derived in the doc eye position formula
    theta_to_north = hor_theta
    theta_to_north = (theta_to_north) % (2*pi)
    # Make the range of theta_to_north between [0, 2*pi)
    theta_to_north[theta_to_north <
                   0] = theta_to_north[theta_to_north < 0] + 2*pi
    inside_tan = theta_to_north.copy()

    # lolll actually this part does not matter cause it's tan^2 anyways
    # 3rd quadrant
    indices = np.where((theta_to_north > pi) & (theta_to_north <= 3*pi/2))[0]
    inside_tan[indices] = - pi - inside_tan[indices]
    # 4th quadrant
    indices = np.where((theta_to_north > pi/2) & (theta_to_north <= pi))[0]
    inside_tan[indices] = pi - inside_tan[indices]

    denominator = (np.tan(inside_tan)**2 + 1)
    numerator_component = 1/np.tan(ver_theta)**2 - np.tan(inside_tan)**2
    numerator = numerator_component * monkey_height**2
    # hide warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        gaze_mky_view_y = np.sqrt(numerator/denominator)

    # based on theta_to_north, we can know the direction of gaze_mky_viewy
    gaze_mky_view_x = np.sqrt(
        (np.tan(inside_tan))**2*(monkey_height**2 + gaze_mky_view_y**2))

    # 4th quadrant
    indices = np.where((theta_to_north > pi/2) & (theta_to_north <= pi))[0]
    gaze_mky_view_y[indices] = -gaze_mky_view_y[indices]

    # 3rd quadrant
    indices = np.where((theta_to_north > pi) & (theta_to_north <= 3*pi/2))[0]
    gaze_mky_view_x[indices] = -gaze_mky_view_x[indices]
    gaze_mky_view_y[indices] = -gaze_mky_view_y[indices]

    # 2nd quadrant
    indices = np.where(theta_to_north > 3*pi/2)[0]
    gaze_mky_view_x[indices] = -gaze_mky_view_x[indices]

    # take interocular distance into account
    if left_or_right_eye == 'left':
        gaze_mky_view_x = gaze_mky_view_x - interocular_dist / 2
    elif left_or_right_eye == 'right':
        gaze_mky_view_x = gaze_mky_view_x + interocular_dist / 2

    # Now we need to rotated back gaze_mky_view_x and gaze_mky_view_y, because they are based on monkey's angle not absolute angles.
    # Also, every point has its own rotation matrix
    # Iterate over the points and angles
    if rotate_world_xy_based_on_m_angle_to_get_abs_coord:
        new_monkey_view_xy = []
        # because the monkey angle is the angle from the x-axis, but now we want the angle from the y-axis, so that to the north is 0
        monkey_angle = monkey_angle - math.pi/2
        for i, (x, y, angle) in enumerate(zip(gaze_mky_view_x, gaze_mky_view_y, monkey_angle)):
            # Create the rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            new_monkey_view_xy.append(
                np.dot(rotation_matrix, np.array([x, y])))
        new_monkey_view_xy = np.array(new_monkey_view_xy)
    else:
        new_monkey_view_xy = np.stack(
            (gaze_mky_view_x, gaze_mky_view_y), axis=1)

    gaze_world_x = body_x + new_monkey_view_xy[:, 0]
    gaze_world_y = body_y + new_monkey_view_xy[:, 1]

    gaze_mky_view_angle = np.arctan2(gaze_mky_view_y, gaze_mky_view_x)
    # We want to make the north as 0, so we need to subtract pi/2 from the angle
    gaze_mky_view_angle = (gaze_mky_view_angle - math.pi/2) % (2*math.pi)
    gaze_mky_view_angle[gaze_mky_view_angle >
                        math.pi] = gaze_mky_view_angle[gaze_mky_view_angle > math.pi] - 2*math.pi

    return gaze_mky_view_x, gaze_mky_view_y, gaze_mky_view_angle, gaze_world_x, gaze_world_y


def eye_angles_from_head_polar(ff_angle,
                               ff_distance,
                               *,
                               monkey_height=-10.0,
                               interocular_dist=4.0,
                               left_or_right_eye='left',
                               wrap='pm_pi'):
    """
    Compute oculocentric eye-rotation angles (horizontal, vertical) from
    head-centered polar coordinates, as you've used in multiFF:
      - ff_angle: azimuth of the target relative to head midline (+x forward),
                  radians CCW; 0 = straight ahead, +pi/2 = left.
      - ff_distance: planar distance from HEAD CENTER (midpoint between eyes)
                     to the target, in the same units as interocular_dist.

    Parameters
    ----------
    ff_angle : array-like
        Head-centered azimuth to target (radians).
    ff_distance : array-like
        Head-centered planar distance to target (same units as cm).
    monkey_height : float, optional
        z offset (target - eye) in the head frame. Negative means target below eye.
        Default -10.0 (e.g., eye 10 cm above the ground target).
    interocular_dist : float, optional
        Distance between the eyes (e.g., 4.0 cm).
    left_or_right_eye : {'left','right'}
        Which eye to compute geometry for.
    wrap : {'pm_pi','0_2pi'}, optional
        Wrapping convention for horizontal angle.

    Returns
    -------
    hor_theta : np.ndarray
        Horizontal eye rotation (azimuth), radians. Positive = leftward.
    ver_theta : np.ndarray
        Vertical eye rotation (elevation), radians. Negative = downward.
    """
    ff_angle     = np.asarray(ff_angle)
    ff_distance  = np.asarray(ff_distance, dtype=float)

    # Convert head-centered polar (origin = between eyes) → Cartesian in head frame
    # Head frame axes: +x forward, +y left, +z up (here z is applied via monkey_height)
    x_head = ff_distance * np.cos(ff_angle)   # forward distance from head center
    y_head = ff_distance * np.sin(ff_angle)   # left-right distance from head center

    # Shift origin from head center to chosen eye by ± interocular_dist/2 along +y
    half_D = interocular_dist / 2.0
    if left_or_right_eye == 'left':
        # Left eye center is at +half_D in head +y; subtract that to express target in eye coords
        y_eye = y_head - (+half_D)
    elif left_or_right_eye == 'right':
        # Right eye center is at -half_D in head +y
        y_eye = y_head - (-half_D)
    else:
        raise ValueError('left_or_right_eye must be "left" or "right"')

    x_eye = x_head
    z_eye = float(monkey_height)  # negative → target below eye

    # Oculocentric angles
    hor_theta = np.arctan2(y_eye, x_eye)                          # azimuth
    r_xy      = np.hypot(x_eye, y_eye)
    ver_theta = np.arctan2(z_eye, np.maximum(r_xy, 1e-9))         # elevation

    # Optional wrapping of horizontal angle
    if wrap == '0_2pi':
        hor_theta = np.mod(hor_theta, 2*np.pi)
    elif wrap == 'pm_pi':
        hor_theta = (hor_theta + np.pi) % (2*np.pi) - np.pi

    return hor_theta, ver_theta


def invert_world_position_to_eye_angles(ff_world_x, ff_world_y, monkey_angle, body_x, body_y,
                                        monkey_height=-10.0,
                                        interocular_dist=4.0,
                                        left_or_right_eye='left',
                                        wrap='pm_pi'):
    """
    World → head → eye pipeline, now delegating the final geometry to
    `eye_angles_from_head_polar`.

    Inputs
    ------
    ff_world_x, ff_world_y : array-like
        Target world coords.
    monkey_angle : array-like
        Heading (radians CCW from world +X), aligns head +x with facing direction.
    body_x, body_y : array-like
        Monkey body center in world coords.
    monkey_height, interocular_dist, left_or_right_eye, wrap
        Passed through to `eye_angles_from_head_polar`.

    Returns
    -------
    hor_theta, ver_theta : np.ndarray
        Oculocentric horizontal/vertical rotations (radians).
    """
    ff_world_x  = np.asarray(ff_world_x)
    ff_world_y  = np.asarray(ff_world_y)
    body_x      = np.asarray(body_x)
    body_y      = np.asarray(body_y)
    monkey_angle = np.asarray(monkey_angle)

    # 1) Body→target displacement in world frame
    dx_w = ff_world_x - body_x
    dy_w = ff_world_y - body_y

    # 2) Rotate world → head frame by -monkey_angle (so head +x = facing direction)
    ca = np.cos(-monkey_angle)
    sa = np.sin(-monkey_angle)
    x_head = ca * dx_w - sa * dy_w   # forward
    y_head = sa * dx_w + ca * dy_w   # left

    # 3) Convert to the head-centric polar quantities used across your project
    ff_distance = np.hypot(x_head, y_head)
    ff_angle    = np.arctan2(y_head, x_head)  # radians CCW from head +x

    # 4) Delegate to the reusable head-polar → eye-angles helper
    return eye_angles_from_head_polar(ff_angle,
                                      ff_distance,
                                      monkey_height=monkey_height,
                                      interocular_dist=interocular_dist,
                                      left_or_right_eye=left_or_right_eye,
                                      wrap=wrap)


def find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, eye_col_suffix=''):
    monkey_sub = monkey_information[(monkey_information['time'] >= duration[0]) & (
        monkey_information['time'] <= duration[1])].copy()

    gaze_world_x = np.array(monkey_sub['gaze_world_x'+eye_col_suffix])
    gaze_world_y = np.array(monkey_sub['gaze_world_y'+eye_col_suffix])
    gaze_world_xy = np.stack((gaze_world_x, gaze_world_y), axis=0)

    gaze_world_xy_rotated = np.matmul(rotation_matrix, gaze_world_xy)
    monkey_sub['gaze_world_x_rotated'] = gaze_world_xy_rotated[0, :]
    monkey_sub['gaze_world_y_rotated'] = gaze_world_xy_rotated[1, :]
    monkey_sub['valid_view_point'] = monkey_sub['valid_view_point'+eye_col_suffix]

    return monkey_sub


def find_valid_view_points(monkey_information):
    # Process left eye view
    valid_view_points_pos_indices = _find_valid_view_points(
        monkey_information[['gaze_world_x_l', 'gaze_world_y_l']].values,
        monkey_information.LDz.values*pi/180
    )
    monkey_information['valid_view_point_l'] = False
    monkey_information.loc[valid_view_points_pos_indices,
                           'valid_view_point_l'] = True

    # Process right eye view
    valid_view_points_pos_indices = _find_valid_view_points(
        monkey_information[['gaze_world_x_r', 'gaze_world_y_r']].values,
        monkey_information.RDz.values*pi/180
    )
    monkey_information['valid_view_point_r'] = False
    monkey_information.loc[valid_view_points_pos_indices,
                           'valid_view_point_r'] = True

    # Process combined-eye view
    valid_view_points_pos_indices = _find_valid_view_points(
        monkey_information[['gaze_world_x', 'gaze_world_y']].values,
        np.minimum(monkey_information.LDz.values*pi/180,
                   monkey_information.RDz.values*pi/180)
        # use the smaller of the two vertical angles, because we only care if the angle is below 0)
    )
    monkey_information['valid_view_point'] = False
    monkey_information.loc[valid_view_points_pos_indices,
                           'valid_view_point'] = True

    return monkey_information


def _find_valid_view_points(gaze_world_xy, ver_theta):

    gaze_world_r = np.linalg.norm(gaze_world_xy, axis=1)
    gaze_world_x = gaze_world_xy[:, 0]

    valid_ver_theta_points = np.where(ver_theta < 0)[0]
    not_nan_indices = np.where(np.isnan(gaze_world_x) == False)[0]
    meaningful_pos_indices = np.intersect1d(
        valid_ver_theta_points, not_nan_indices)

    within_arena_points = np.where(gaze_world_r < 1000)[0]
    valid_view_points_pos_indices = np.intersect1d(
        meaningful_pos_indices, within_arena_points)

    return valid_view_points_pos_indices


def find_eye_world_speed(monkey_information):
    monkey_df = monkey_information[monkey_information['valid_view_point'] == True].copy(
    )
    delta_x = np.diff(monkey_df['gaze_mky_view_x'])
    delta_y = np.diff(monkey_df['gaze_mky_view_y'])
    delta_t = np.diff(monkey_df['time'])
    delta_position = np.linalg.norm(np.array([delta_x, delta_y]), axis=0)
    eye_world_speed = delta_position / delta_t
    eye_world_speed = np.append(eye_world_speed[0], eye_world_speed)
    monkey_df['eye_world_speed'] = eye_world_speed

    # now, use merge to put eye_world_speed back to the original monkey_information
    monkey_information = pd.merge(monkey_information, monkey_df[[
                                  'point_index', 'eye_world_speed']], on='point_index', how='left')
    # convert inf to NA
    monkey_information['eye_world_speed'] = monkey_information['eye_world_speed'].replace([
                                                                                          np.inf, -np.inf], np.nan)
    # use forward and backward fill to fill in the nans
    monkey_information['eye_world_speed'] = monkey_information['eye_world_speed'].fillna(
        method='ffill')
    monkey_information['eye_world_speed'] = monkey_information['eye_world_speed'].fillna(
        method='bfill')
    return monkey_information


def plot_eye_world_speed_vs_monkey_speed(monkey_sub):
    # gaze_mky_view_xy, cum_t, overall_valid_indices,
    monkey_sub = monkey_sub.copy()
    monkey_sub['cum_t'] = monkey_sub['time'] - monkey_sub['time'].values[0]

    # get eye data to plot
    monkey_sub2 = monkey_sub[monkey_sub['valid_view_point'] == True].copy()
    monkey_sub2.loc[monkey_sub2['eye_world_speed']
                    > 1000, 'eye_world_speed'] = 1000

    legend_labels = {1: "Eye Speed",
                     2: 'Monkey linear speed',
                     3: 'Monkey angular speed'}

    ylabels = {1: "Speed of eye position in the world",
               2: 'Monkey linear speed',
               3: 'Monkey angular speed'}

    colors = {1: 'gold',
              2: 'royalblue',
              3: 'indianred'}

    fig = plt.figure(figsize=(6.5, 4), dpi=125)
    ax = fig.add_subplot()
    fig.subplots_adjust(right=0.75)

    i = 1  # eye speed
    p1, = ax.plot(monkey_sub2['cum_t'], monkey_sub2['eye_world_speed'],
                  color=colors[i], label=legend_labels[i])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(ylabels[i], color='darkgoldenrod', fontsize=10)

    # Change the last label of the ytick labels for eye speed
    yticks = ax.get_yticks()
    yticks = [ytick for ytick in yticks if (ytick <= 1000)]
    ytick_labels = [str(int(ytick)) for ytick in yticks]
    ytick_labels[-1] = ytick_labels[-1] + '+'
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    i = 2  # monkey linear speed
    ax2 = ax.twinx()
    p2, = ax2.plot(monkey_sub['cum_t'], monkey_sub['speed'],
                   color=colors[i], label=legend_labels[i])
    ax2.set_ylabel(ylabels[i], color=colors[i], fontsize=13)

    i = 3  # monkey angular speed
    ax3 = ax.twinx()
    p3, = ax3.plot(monkey_sub['cum_t'], monkey_sub['ang_speed'],
                   color=colors[i], label=legend_labels[i])
    ax3.set_ylabel(ylabels[i], color=colors[i], fontsize=13)

    # Offset the right spine of ax3
    ax3.spines.right.set_position(("axes", 1.2))

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors='darkgoldenrod', **tkw)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], fontsize=7, bbox_to_anchor=(
        0., -0.35, 0.5, .102), loc="lower left")

    plt.title('Eye Speed vs. Monkey Speed', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close
