import sys

from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import math



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
    gaze_monkey_view_x_l, gaze_monkey_view_y_l, gaze_monkey_view_angle_l, gaze_world_x_l, gaze_world_y_l \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=interocular_dist, left_or_right_eye='left')
    
    # right eye
    ver_theta = np.array(monkey_information.RDz)*pi/180
    hor_theta = np.array(monkey_information.RDy)*pi/180
    gaze_monkey_view_x_r, gaze_monkey_view_y_r, gaze_monkey_view_angle_r, gaze_world_x_r, gaze_world_y_r \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=interocular_dist, left_or_right_eye='right')

    # average the two eyes
    gaze_monkey_view_x = (gaze_monkey_view_x_l + gaze_monkey_view_x_r)/2
    gaze_monkey_view_y = (gaze_monkey_view_y_l + gaze_monkey_view_y_r)/2
    gaze_monkey_view_angle = (gaze_monkey_view_angle_l + gaze_monkey_view_angle_r)/2
    gaze_world_x = (gaze_world_x_l + gaze_world_x_r)/2
    gaze_world_y = (gaze_world_y_l + gaze_world_y_r)/2
    
    monkey_information['gaze_monkey_view_x'] = gaze_monkey_view_x
    monkey_information['gaze_monkey_view_y'] = gaze_monkey_view_y
    monkey_information['gaze_monkey_view_angle'] = gaze_monkey_view_angle
    monkey_information['gaze_world_x'] = gaze_world_x
    monkey_information['gaze_world_y'] = gaze_world_y

    if add_left_and_right_eyes_info:
        monkey_information['gaze_monkey_view_x_l'] = gaze_monkey_view_x_l
        monkey_information['gaze_monkey_view_y_l'] = gaze_monkey_view_y_l
        monkey_information['gaze_monkey_view_angle_l'] = gaze_monkey_view_angle_l
        monkey_information['gaze_world_x_l'] = gaze_world_x_l
        monkey_information['gaze_world_y_l'] = gaze_world_y_l

        monkey_information['gaze_monkey_view_x_r'] = gaze_monkey_view_x_r
        monkey_information['gaze_monkey_view_y_r'] = gaze_monkey_view_y_r
        monkey_information['gaze_monkey_view_angle_r'] = gaze_monkey_view_angle_r
        monkey_information['gaze_world_x_r'] = gaze_world_x_r
        monkey_information['gaze_world_y_r'] = gaze_world_y_r
    return monkey_information





def average_and_then_convert_eye_positions_in_monkey_information(monkey_information, add_suffix_to_new_columns=True):
    monkey_height = -10
    body_x = np.array(monkey_information.monkey_x)
    body_y = np.array(monkey_information.monkey_y)
    monkey_angle = np.array(monkey_information.monkey_angle)

    
    ver_theta = (np.array(monkey_information.LDz) + np.array(monkey_information.RDz))*pi/180/2
    hor_theta = (np.array(monkey_information.LDy) + np.array(monkey_information.RDy))*pi/180/2
    gaze_monkey_view_x_avg, gaze_monkey_view_y_avg, gaze_monkey_view_angle_avg, gaze_world_x_avg, gaze_world_y_avg \
        = apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                                interocular_dist=0)
    
    if add_suffix_to_new_columns:
        suffix = '_avg'
    else:
        suffix = ''
    monkey_information['gaze_monkey_view_x'+suffix] = gaze_monkey_view_x_avg
    monkey_information['gaze_monkey_view_y'+suffix] = gaze_monkey_view_y_avg
    monkey_information['gaze_monkey_view_angle'+suffix] = gaze_monkey_view_angle_avg
    monkey_information['gaze_world_x'+suffix] = gaze_world_x_avg
    monkey_information['gaze_world_y'+suffix] = gaze_world_y_avg

    

    return monkey_information



def apply_formulas_to_convert_eye_position_to_ff_position(hor_theta, ver_theta, monkey_angle, monkey_height, body_x, body_y,
                                                          interocular_dist=4, left_or_right_eye='left',
                                                          rotate_world_xy_based_on_m_angle_to_get_abs_coord=True):
    # This uses the formulas derived in the doc eye position formula
    theta_to_north = hor_theta
    theta_to_north = (theta_to_north)%(2*pi)
    # Make the range of theta_to_north between [0, 2*pi)
    theta_to_north[theta_to_north < 0] = theta_to_north[theta_to_north < 0] + 2*pi
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
        gaze_monkey_view_y = np.sqrt(numerator/denominator)

    # based on theta_to_north, we can know the direction of gaze_monkey_viewy
    gaze_monkey_view_x = np.sqrt((np.tan(inside_tan))**2*(monkey_height**2 + gaze_monkey_view_y**2))


    # 4th quadrant
    indices = np.where((theta_to_north > pi/2) & (theta_to_north <= pi))[0]
    gaze_monkey_view_y[indices] = -gaze_monkey_view_y[indices]

    # 3rd quadrant
    indices = np.where((theta_to_north > pi) & (theta_to_north <= 3*pi/2))[0]
    gaze_monkey_view_x[indices] = -gaze_monkey_view_x[indices]
    gaze_monkey_view_y[indices] = -gaze_monkey_view_y[indices]

    # 2nd quadrant
    indices = np.where(theta_to_north > 3*pi/2)[0]
    gaze_monkey_view_x[indices] = -gaze_monkey_view_x[indices]

    # take interocular distance into account
    if left_or_right_eye == 'left':
        gaze_monkey_view_x = gaze_monkey_view_x - interocular_dist / 2
    elif left_or_right_eye == 'right':
        gaze_monkey_view_x = gaze_monkey_view_x + interocular_dist / 2


    # Now we need to rotated back gaze_monkey_view_x and gaze_monkey_view_y, because they are based on monkey's angle not absolute angles.
    # Also, every point has its own rotation matrix
    # Iterate over the points and angles
    if rotate_world_xy_based_on_m_angle_to_get_abs_coord:
        new_monkey_view_xy = []
        monkey_angle = monkey_angle - math.pi/2 # because the monkey angle is the angle from the x-axis, but now we want the angle from the y-axis, so that to the north is 0
        for i, (x, y, angle) in enumerate(zip(gaze_monkey_view_x, gaze_monkey_view_y, monkey_angle)):
            # Create the rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            new_monkey_view_xy.append(np.dot(rotation_matrix, np.array([x, y])))
        new_monkey_view_xy = np.array(new_monkey_view_xy)
    else:
        new_monkey_view_xy = np.stack((gaze_monkey_view_x, gaze_monkey_view_y), axis=1)

    gaze_world_x = body_x + new_monkey_view_xy[:, 0]
    gaze_world_y = body_y + new_monkey_view_xy[:, 1]

    gaze_monkey_view_angle = np.arctan2(gaze_monkey_view_y, gaze_monkey_view_x)
    # We want to make the north as 0, so we need to subtract pi/2 from the angle
    gaze_monkey_view_angle = (gaze_monkey_view_angle - math.pi/2)%(2*math.pi)
    gaze_monkey_view_angle[gaze_monkey_view_angle > math.pi] = gaze_monkey_view_angle[gaze_monkey_view_angle > math.pi] - 2*math.pi

    return gaze_monkey_view_x, gaze_monkey_view_y, gaze_monkey_view_angle, gaze_world_x, gaze_world_y






def find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, separate_left_and_right_eyes=False):

    if not separate_left_and_right_eyes:
        gaze_world_xy_rotated, overall_valid_indices, monkey_subset = _find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, suffix_for_column='')
        return gaze_world_xy_rotated, overall_valid_indices, monkey_subset
    else:
        gaze_world_xy_rotated_l, overall_valid_indices_l, monkey_subset_l = _find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, suffix_for_column='_l')
        gaze_world_xy_rotated_r, overall_valid_indices_r, monkey_subset_r = _find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, suffix_for_column='_r')
        
        both_eyes_info = {'gaze_world_xy_rotated': {'left': gaze_world_xy_rotated_l, 'right': gaze_world_xy_rotated_r},
                          'overall_valid_indices': {'left': overall_valid_indices_l, 'right': overall_valid_indices_r},
                          'monkey_subset': {'left': monkey_subset_l, 'right': monkey_subset_r}
                          }
        return both_eyes_info



def _find_eye_positions_rotated_in_world_coordinates(monkey_information, duration, rotation_matrix, suffix_for_column=''):
    monkey_subset = monkey_information[(monkey_information['time'] >= duration[0]) & (monkey_information['time'] <= duration[1])].copy()

    ver_theta = np.array(monkey_subset['LDz'])
    if suffix_for_column == '_r':
        ver_theta = np.array(monkey_subset['RDz'])

    gaze_world_x = np.array(monkey_subset['gaze_world_x'+suffix_for_column])
    gaze_world_y = np.array(monkey_subset['gaze_world_y'+suffix_for_column])

    gaze_world_xy_rotated, meaningful_indices, overall_valid_indices = find_eye_positions_given_info(gaze_world_x, gaze_world_y, ver_theta, rotation_matrix)
    
    monkey_subset = monkey_subset[['point_index', 'time']].copy()
    monkey_subset['gaze_world_x_rotated'] = gaze_world_xy_rotated[0, :]
    monkey_subset['gaze_world_y_rotated'] = gaze_world_xy_rotated[1, :]
    monkey_subset['eye_position_meaningful'] = False
    monkey_subset['eye_position_overall_valid'] = False
    monkey_subset.loc[meaningful_indices, 'eye_position_meaningful'] = True
    monkey_subset.loc[overall_valid_indices, 'eye_position_overall_valid'] = True

    return gaze_world_xy_rotated, overall_valid_indices, monkey_subset



def find_eye_positions_given_info(gaze_world_x, gaze_world_y, ver_theta, rotation_matrix):

    gaze_world_xy = np.stack((gaze_world_x, gaze_world_y), axis=1)
    gaze_world_r = LA.norm(gaze_world_xy, axis=1)
    gaze_world_xy_rotated = gaze_world_xy.T
    gaze_world_xy_rotated = np.matmul(rotation_matrix, gaze_world_xy_rotated)

    valid_ver_theta_points = np.where(ver_theta < 0)[0]
    not_nan_indices = np.where(np.isnan(gaze_world_x)==False)[0]
    meaningful_indices = np.intersect1d(valid_ver_theta_points, not_nan_indices)

    within_arena_points = np.where(gaze_world_r < 1000)[0]
    overall_valid_indices = np.intersect1d(meaningful_indices, within_arena_points)

    return gaze_world_xy_rotated, meaningful_indices, overall_valid_indices




def find_eye_world_speed(gaze_monkey_view_xy, cum_t, overall_valid_indices):
    gaze_x = gaze_monkey_view_xy[0]
    gaze_y = gaze_monkey_view_xy[1]

    delta_x = np.diff(gaze_x[overall_valid_indices])
    delta_y = np.diff(gaze_y[overall_valid_indices])
    delta_t = np.diff(cum_t[overall_valid_indices])
    
    delta_position = LA.norm(np.array([delta_x, delta_y]), axis=0)
    eye_world_speed = delta_position/delta_t
    eye_world_speed = np.append(eye_world_speed[0], eye_world_speed)

    return eye_world_speed



def plot_eye_world_speed_vs_monkey_speed(gaze_monkey_view_xy, cum_t, overall_valid_indices, monkey_information):
    
    eye_world_speed = find_eye_world_speed(gaze_monkey_view_xy, cum_t, overall_valid_indices)
    eye_world_speed[eye_world_speed > 1000] = 1000
    cum_t_to_plot = cum_t - cum_t[0]
    relevant_monkey_info = monkey_information[monkey_information.time.isin(cum_t)]
    corresponding_monkey_speed = relevant_monkey_info.monkey_speed.values
    corresponding_monkey_dw = relevant_monkey_info.monkey_dw.values
    

    legend_labels = {1:"Eye Speed",
              2:'Monkey linear speed',
              3:'Monkey angular speed'}

    ylabels = {1:"Speed of eye position in the world",
              2:'Monkey linear speed',
              3:'Monkey angular speed'}

    colors = {1:'gold',
              2:'royalblue',
              3:'indianred'}

    fig = plt.figure(figsize=(6.5, 4), dpi=125)
    ax = fig.add_subplot()
    fig.subplots_adjust(right=0.75)

    i = 1 # eye speed
    p1, = ax.plot(cum_t_to_plot[overall_valid_indices], eye_world_speed, color=colors[i], label=legend_labels[i])
    ax.set_xlabel('Time (s)', fontsize = 12)
    ax.set_ylabel(ylabels[i], color='darkgoldenrod', fontsize=10)

    # Change the last label of the ytick labels for eye speed
    yticks = ax.get_yticks()
    yticks = [ytick for ytick in yticks if (ytick <= 1000)]
    ytick_labels = [str(int(ytick)) for ytick in yticks]
    ytick_labels[-1] = ytick_labels[-1] + '+'
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)


    i = 2 # monkey linear speed
    ax2=ax.twinx()
    p2, = ax2.plot(cum_t_to_plot, corresponding_monkey_speed, color=colors[i], label=legend_labels[i])
    ax2.set_ylabel(ylabels[i], color=colors[i], fontsize=13)


    i = 3 # monkey angular speed
    ax3=ax.twinx()
    p3, = ax3.plot(cum_t_to_plot, corresponding_monkey_dw, color=colors[i], label=legend_labels[i])
    ax3.set_ylabel(ylabels[i], color=colors[i], fontsize=13)

    # Offset the right spine of ax3
    ax3.spines.right.set_position(("axes", 1.2))


    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors='darkgoldenrod', **tkw)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], fontsize=7, bbox_to_anchor=(0., -0.35, 0.5, .102), loc="lower left")


    plt.title('Eye Speed vs. Monkey Speed', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close
