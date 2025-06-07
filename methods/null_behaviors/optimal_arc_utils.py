import math
import pandas as pd
from planning_analysis.show_planning import show_planning_utils
from null_behaviors import show_null_trajectory

import os
import warnings
import numpy as np
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def update_curvature_df_to_let_optimal_arc_stop_at_closest_point_to_monkey_stop(curvature_df, cur_ff_df, stops_near_ff_df,
                                                                                ff_real_position_sorted, monkey_information):
    # The idea is to find the closest point on optimal arc to the stop that's also within ff, and then
    # treat that point as the new ff location. From there we can use the code for calculation of arc to center.

    # Extract new firefly coordinates
    old_cur_null_arc_info = show_null_trajectory.find_and_package_optimal_arc_info_for_plotting(
        curvature_df, monkey_information
    )

    # Get the optimal arc landing points closest to the stop
    arc_rows_closest_to_stop = show_planning_utils.get_optimal_arc_landing_points_closest_to_stop(
        old_cur_null_arc_info, stops_near_ff_df
    )

    # Extract new firefly x and y coordinates
    new_ff_x, new_ff_y = arc_rows_closest_to_stop['x'].values, arc_rows_closest_to_stop['y'].values

    if len(new_ff_x) != len(cur_ff_df):
        raise ValueError(
            'Number of new firefly x coordinates must match the number of stops')

    # calculate the distance between "new ff" and monkey xy. If the distance is within 1 cm, make it so that new ff is at the monkey xy
    new_ff_x, new_ff_y = show_planning_utils.make_new_ff_at_monkey_xy_if_within_1_cm(new_ff_x, new_ff_y,
                                                                                     cur_ff_df['monkey_x'].values, cur_ff_df['monkey_y'].values
                                                                                     )

    # Find and package arc to center info for plotting
    cur_null_arc_info = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(
        cur_ff_df.point_index.values, cur_ff_df.ff_index.values, monkey_information,
        ff_real_position_sorted, ff_x=new_ff_x, ff_y=new_ff_y, ignore_error=True
    )

    # Update stop curvature DataFrame with optimal arc measures and lengths
    curvature_df['optimal_arc_measure'] = cur_null_arc_info['all_arc_measure'].values
    curvature_df['optimal_arc_length'] = curvature_df['optimal_arc_measure'] * \
        curvature_df['optimal_arc_radius']
    curvature_df['optimal_arc_d_heading'] = curvature_df['optimal_arc_measure'] * \
        curvature_df['optimal_arc_end_direction'].values
    # Update stop curvature DataFrame with optimal arc end coordinates
    curvature_df[['optimal_arc_end_x', 'optimal_arc_end_y']
                 ] = arc_rows_closest_to_stop[['x', 'y']].values

    return curvature_df


def add_optimal_arc_measure_and_length(curvature_df,
                                       opt_arc_stop_first_vis_bdry=False,  # whether optimal arc stop at visible boundary
                                       ignore_error=False):

    # find arc ending xy for optimal curvature (curv to disk edge)
    monkey_xy = curvature_df[['monkey_x', 'monkey_y']].values
    monkey_angle = curvature_df['monkey_angle'].values
    ff_distance = curvature_df['ff_distance'].values
    ff_angle = curvature_df['ff_angle'].values
    arc_ff_xy = curvature_df[['ff_x', 'ff_y']].values
    arc_end_direction = curvature_df['optimal_arc_end_direction'].values
    arc_radius = curvature_df['optimal_arc_radius'].values

    whether_ff_behind = (np.abs(curvature_df['ff_angle_boundary']) > math.pi/2)

    if not ignore_error:
        if (abs(curvature_df['ff_angle_boundary']) > math.pi/4).sum() > 0:
            raise ValueError(
                "At least one ff has an angle to boundary larger than pi/4. This is invalid. Please check the input.")

    center_x, center_y, arc_starting_angle, arc_ending_angle = find_cartesian_arc_center_and_angle_for_optimal_arc(arc_ff_xy, monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius,
                                                                                                                   arc_end_direction, whether_ff_behind=whether_ff_behind,
                                                                                                                   opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry,
                                                                                                                   ignore_error=ignore_error)

    curvature_df['optimal_arc_measure'] = np.abs(
        arc_starting_angle - arc_ending_angle)
    curvature_df['optimal_arc_length'] = curvature_df['optimal_arc_measure'] * \
        curvature_df['optimal_arc_radius']

    # also find optimal arc end x and y
    curvature_df['optimal_arc_end_x'] = np.cos(
        arc_ending_angle)*arc_radius + center_x
    curvature_df['optimal_arc_end_y'] = np.sin(
        arc_ending_angle)*arc_radius + center_y

    # make sure that all landing points are within the reward boundary
    arc_end_distance_to_ff_center = np.sqrt((curvature_df['optimal_arc_end_x'] - curvature_df['ff_x'])**2 + (
        curvature_df['optimal_arc_end_y'] - curvature_df['ff_y'])**2)
    if np.any(arc_end_distance_to_ff_center > 25):
        if not ignore_error:
            raise ValueError(
                "At least one arc end is outside the reward boundary. This is invalid. Please check the input.")
        else:
            print(
                "Warning: At least one arc end is outside the reward boundary. This is invalid. We will adjust them by making them a little less than pi.")


def find_cartesian_arc_center_and_angle_for_optimal_arc_to_arc_end(arc_end_xy, monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, arc_end_direction, whether_ff_behind=None,
                                                                   ignore_error=False):
    center_x, center_y, arc_starting_angle, arc_ending_angle = find_cartesian_arc_center_and_angle_for_optimal_arc(arc_end_xy, monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, arc_end_direction,
                                                                                                                   whether_ff_behind=whether_ff_behind, ignore_error=ignore_error,
                                                                                                                   opt_arc_stop_first_vis_bdry=False)
    return center_x, center_y, arc_starting_angle, arc_ending_angle


def find_arc_center_in_world_coord(monkey_xy, monkey_angle, arc_radius, arc_end_direction):
    if monkey_xy.ndim == 1:
        monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    elif monkey_xy.shape[1] == 2:
        monkey_x, monkey_y = monkey_xy[:, 0], monkey_xy[:, 1]
    else:
        monkey_x, monkey_y = monkey_xy[0, :], monkey_xy[1, :]
    # relative to the monkey, the angle can only be 90 degrees or -90 degrees
    angle_to_center_of_arc_from_monkey = monkey_angle + pi/2*arc_end_direction
    # find the center of the circle that the arc is on
    center_x = np.cos(angle_to_center_of_arc_from_monkey)*arc_radius + monkey_x
    center_y = np.sin(angle_to_center_of_arc_from_monkey)*arc_radius + monkey_y
    return center_x, center_y


def find_angle_from_arc_center_to_monkey_and_stop_position(arc_ff_xy, ff_angle, monkey_xy, center_x, center_y):
    arc_ff_xy = np.round(arc_ff_xy, 5)
    monkey_xy = np.around(monkey_xy, 5)
    center_x = np.around(center_x, 5)
    center_y = np.around(center_y, 5)

    angle_from_center_to_stop = find_angle_from_arc_center_to_ff(
        arc_ff_xy, center_x, center_y)
    angle_from_center_to_monkey = find_angle_from_arc_center_to_monkey(
        monkey_xy, center_x, center_y)
    angle_from_center_to_stop[np.abs(
        ff_angle) > math.pi/2] = angle_from_center_to_monkey[np.abs(ff_angle) > math.pi/2]
    return angle_from_center_to_monkey, angle_from_center_to_stop


def find_angle_from_arc_center_to_monkey(monkey_xy, center_x, center_y):
    if monkey_xy.ndim == 1:
        monkey_x, monkey_y = monkey_xy[0], monkey_xy[1]
    elif monkey_xy.shape[1] == 2:
        monkey_x, monkey_y = monkey_xy[:, 0], monkey_xy[:, 1]
    else:
        monkey_x, monkey_y = monkey_xy[0, :], monkey_xy[1, :]

    angle_from_center_to_monkey = np.arctan2(
        monkey_y-center_y, monkey_x-center_x)
    return angle_from_center_to_monkey


def find_angle_from_arc_center_to_ff(arc_ff_xy, center_x, center_y):
    # the calculation for angle_from_center_to_stop is the same whether optimal arc or arc to ff center is used
    arc_ff_xy = arc_ff_xy.reshape(-1, 2)
    angle_from_center_to_stop = np.arctan2(
        arc_ff_xy[:, 1]-center_y, arc_ff_xy[:, 0]-center_x)
    return angle_from_center_to_stop


def find_cartesian_arc_center_and_angle_for_optimal_arc(arc_ff_xy, monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, arc_end_direction, whether_ff_behind=None,
                                                        opt_arc_stop_first_vis_bdry=False, ignore_error=False):
    # Note: sometimes arc_ff_xy is replaced by arc ending xy, which produces the same result if the normal optimal arc is used;
    # but if opt_arc_stop_closest was True (optimal arc stop at closest point to monkey stop), then arc ending xy has to be used to mimic a new ff center

    center_x, center_y = find_arc_center_in_world_coord(
        monkey_xy, monkey_angle, arc_radius, arc_end_direction)
    angle_from_center_to_monkey, angle_from_center_to_stop = find_angle_from_arc_center_to_monkey_and_stop_position(
        arc_ff_xy, ff_angle, monkey_xy, center_x, center_y)

    if opt_arc_stop_first_vis_bdry:
        temp_null_arc_info = pd.DataFrame({
            # this is just for keeping the ff in sequence, not for merging, so it's fine to use np.arange
            'arc_ff_index': np.arange(len(ff_angle)),
            'all_arc_radius': arc_radius,
            'center_x': center_x,
            'center_y': center_y,
            'arc_ff_x': arc_ff_xy[:, 0],
            'arc_ff_y': arc_ff_xy[:, 1],
            'arc_starting_angle': angle_from_center_to_monkey,
            'arc_ending_angle': angle_from_center_to_stop
        })
        arc_rows_to_first_reach_boundary = show_planning_utils.get_optimal_arc_landing_points_when_first_reaching_visible_boundary(
            temp_null_arc_info)
        angle_from_center_to_stop = arc_rows_to_first_reach_boundary['angle'].values

    arc_starting_angle, arc_ending_angle = _find_cartesian_arc_starting_and_ending_angle(angle_from_center_to_monkey, angle_from_center_to_stop, ff_distance, ff_angle, arc_end_direction,
                                                                                         whether_ff_behind=whether_ff_behind, ignore_error=ignore_error)

    return center_x, center_y, arc_starting_angle, arc_ending_angle


def _supply_curvature_df_with_optimal_arc_info(curvature_df, ff_radius_for_optimal_arc, opt_arc_stop_first_vis_bdry=True, ignore_error=False):
    all_ff_angle = curvature_df['ff_angle'].values.copy()
    all_ff_distance = curvature_df['ff_distance'].values.copy()
    curvature_lower_bound, curvature_upper_bound = find_curvature_lower_and_upper_bound(
        all_ff_angle, all_ff_distance, ff_radius=ff_radius_for_optimal_arc)
    curvature_df['curvature_lower_bound'] = curvature_lower_bound
    curvature_df['curvature_upper_bound'] = curvature_upper_bound
    # clip curvature to be between lower and upper bound
    curvature_df['optimal_curvature'] = np.clip(
        curvature_df['curv_of_traj'], curvature_df['curvature_lower_bound'], curvature_df['curvature_upper_bound'])
    # make sure that optimal_curvature has the same sign has ff_angle
    abs_optimal_curvature = np.abs(curvature_df['optimal_curvature'])
    # if there's any 0, make it a small number
    abs_optimal_curvature[abs_optimal_curvature == 0] = 0.000001
    curvature_df['optimal_curvature'] = abs_optimal_curvature * \
        np.sign(curvature_df['ff_angle'])
    curvature_df['optimal_arc_end_direction'] = np.sign(
        curvature_df['optimal_curvature'])
    # if any optimal_arc_end_direction is 0, raise an error
    if np.any(curvature_df['optimal_arc_end_direction'] == 0):
        raise ValueError(
            "At least one optimal_arc_end_direction is 0, which is not possible. Please check the code.")
    curvature_df['optimal_arc_radius'] = find_arc_radius_based_on_curvature(
        curvature_df['optimal_curvature'])

    # find arc ending xy for optimal curvature (curv to disk edge)
    add_optimal_arc_measure_and_length(curvature_df, opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry,
                                       ignore_error=ignore_error)

    return curvature_df


def find_curvature_lower_and_upper_bound(ff_angle, ff_distance, ff_radius=10):

    # calculate_ff_angle_boundaries
    side_opposite = ff_radius
    # hypotenuse cannot be smaller than side_opposite
    hypotenuse = np.clip(ff_distance, a_min=side_opposite, a_max=2000)
    theta = np.arcsin(np.divide(side_opposite, hypotenuse))

    all_ff_angle_lower_bound = ff_angle - np.abs(theta)
    all_ff_angle_upper_bound = ff_angle + np.abs(theta)

    # also make sure that the angles are within (-pi/4, pi/4)
    all_ff_angle_lower_bound = np.clip(
        all_ff_angle_lower_bound, a_min=-pi/4, a_max=pi/4)
    all_ff_angle_upper_bound = np.clip(
        all_ff_angle_upper_bound, a_min=-pi/4, a_max=pi/4)

    # if the lower bound equals the upper bound for any ff, then there's a problem. One needs to raise an error.
    if np.any(all_ff_angle_lower_bound == all_ff_angle_upper_bound):
        print("Warnings: At least one ff has a lower bound of ff_angle_boundary equal to its upper bound after clipping, meaning that the ff's angle to boundary is greater than 90 degrees. Please check the input.")

    # get ff_distance respectively for the lower-bound and upper-bound angles
    # ff_distance_to_edge = np.abs(np.cos(theta)*ff_distance) # and this should be the same for both lower and upper bound
    # ff_distance_to_edge[ff_distance <= ff_radius] = 0

    lower_theta = np.abs(all_ff_angle_lower_bound - ff_angle)
    upper_theta = np.abs(all_ff_angle_upper_bound - ff_angle)
    ff_distance_to_edge_for_lower_bound = np.abs(
        np.cos(lower_theta)*ff_distance)
    ff_distance_to_edge_for_upper_bound = np.abs(
        np.cos(upper_theta)*ff_distance)
    ff_distance_to_edge_for_lower_bound[ff_distance <= ff_radius] = 0
    ff_distance_to_edge_for_upper_bound[ff_distance <= ff_radius] = 0

    # supress warnings because invalid values might occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        curvature_lower_bound, _ = find_arc_curvature(
            all_ff_angle_lower_bound, ff_distance_to_edge_for_lower_bound)
        curvature_upper_bound, _ = find_arc_curvature(
            all_ff_angle_upper_bound, ff_distance_to_edge_for_upper_bound)

    # deal with the case where the monkey is within the reward boundary of the ff

    curvature_lower_bound[ff_distance <= ff_radius] = -math.pi/4
    curvature_upper_bound[ff_distance <= ff_radius] = math.pi/4

    # if either of the bound is of a different sign from ff_angle, then it will become 0, with the same sign as ff_angle
    curvature_lower_bound = np.where(
        np.sign(curvature_lower_bound) != np.sign(ff_angle), 0, curvature_lower_bound)
    curvature_upper_bound = np.where(
        np.sign(curvature_upper_bound) != np.sign(ff_angle), 0, curvature_upper_bound)

    return curvature_lower_bound, curvature_upper_bound


def find_arc_radius_based_on_curvature(curvatures):
    uniform_arc_length = 1  # we use an arbitrary here because if the equations below get simplified, the uniform_arc_length will be cancelled out
    delta_monkey_angle = curvatures * uniform_arc_length
    arc_measure = np.abs(delta_monkey_angle)
    all_arc_radius = uniform_arc_length/arc_measure
    # the above can be simpliefied to all_arc_radius = 1/curvatures
    return all_arc_radius


def find_arc_curvature(ff_angle, ff_distance, invalid_curvature_ok=False):

    # Suppose that the monkey traverses a perfect arc of length 150, which, if extended, will be a circle that the ff is on, with given ff_angle and ff_distance
    # This function will find the change in monkey_angle over the change in distance for each arc

    # we used +pi/2 here because the ff_angle is counted as starting from 0 which is to the north of the monkey
    ff_y_relative = np.sin(ff_angle+pi/2)*ff_distance

    # supress warnings because invalid values might occur
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    # result will be between (0, pi)
    arc_angle = pi - 2*np.arcsin(np.abs(ff_y_relative/ff_distance))
    all_arc_radius = ff_y_relative/np.sin(arc_angle)

    # all_arc_radius should be all positive
    if invalid_curvature_ok is False:
        if np.any(all_arc_radius < 0):
            raise ValueError("Note: at least one arc has a negative radius here, because its relative_y to the monkey is negative. In other words, the ff is behind the monkey. If a negative radius is not desired, please eliminate ff behind the monkey before calling this function.")
        # if any ff angle has an absolute value larger than pi/4, it's invalid
        if np.any(np.abs(ff_angle) > pi/4):
            max_angle = np.max(np.abs(ff_angle)) * 180/math.pi
            raise ValueError(
                f"Note: at least one ff has an angle larger than pi/4. The max is {max_angle}, This is invalid. Please check the input.")
    curvature = 1/all_arc_radius
    curvature[np.isnan(curvature)] = 0
    curvature[ff_angle < 0] = -curvature[ff_angle < 0]

    # Or, it's the same mechanism as the below:
    # arc_measure = uniform_arc_length/all_arc_radius # calculate the angles that define the lengths of the arcs
    # # because a tangent to a circle is perpendicular to the radius, we can get delta monkey angle very easily
    # delta_monkey_angle = arc_measure
    # # for ff to the right
    # delta_monkey_angle[ff_angle < 0] = -delta_monkey_angle[ff_angle < 0]
    # curvature = delta_monkey_angle/uniform_arc_length

    return curvature, all_arc_radius


def _find_cartesian_arc_starting_and_ending_angle(angle_from_center_to_monkey, angle_from_center_to_stop, ff_distance, ff_angle, arc_end_direction, whether_ff_behind=None,
                                                  reward_boundary_radius=25, ignore_error=False):

    arc_starting_angle = angle_from_center_to_monkey.copy()
    arc_ending_angle = angle_from_center_to_stop.copy()

    # arc_starting_angle, arc_ending_angle = _refine_arc_starting_and_ending_angles(arc_starting_angle, arc_ending_angle, arc_end_direction)

    if whether_ff_behind is not None:
        # for the ff that whether_ff_behind is True, both arc_starting_angle and arc_ending_angle will be arc_starting_angle, arc_ending_angle
        arc_starting_angle[whether_ff_behind] = angle_from_center_to_stop[whether_ff_behind].copy(
        )
        arc_ending_angle[whether_ff_behind] = angle_from_center_to_stop[whether_ff_behind].copy()

    df = pd.DataFrame({'arc_starting_angle': arc_starting_angle,
                       'arc_ending_angle': arc_ending_angle,
                       'ff_distance': ff_distance,
                       'ff_angle': ff_angle,
                       'arc_end_direction': arc_end_direction})

    # Adjust the starting and ending angles of null arcs so that they are within 180 degrees of each other.
    df = _adjust_arc_angles(df)

    # deal with delta angles greater than 180 degrees
    df = _deal_with_delta_angles_greater_than_180_degrees(
        df, reward_boundary_radius=reward_boundary_radius, ignore_error=ignore_error)

    # deal with delta angles greater than 45 degrees
    df = _deal_with_delta_angles_greater_than_90_degrees(
        df, reward_boundary_radius=reward_boundary_radius, ignore_error=ignore_error)

    return df['arc_starting_angle'].values, df['arc_ending_angle'].values


def _adjust_arc_angles(null_arc_info):
    """
    Adjust the starting and ending angles of null arcs so that they are within 180 degrees of each other.
    """

    for i in range(2):  # do the following twice, just in case
        # Adjust 'arc_ending_angle' to ensure it is within 180 degrees of 'arc_starting_angle'
        # If 'arc_ending_angle' is more than 180 degrees greater than 'arc_starting_angle', subtract 2*pi from 'arc_ending_angle'
        greater_than_pi = null_arc_info['arc_ending_angle'] - \
            null_arc_info['arc_starting_angle'] > math.pi
        null_arc_info.loc[greater_than_pi, 'arc_ending_angle'] -= 2 * math.pi

        # If 'arc_ending_angle' is more than 180 degrees less than 'arc_starting_angle', add 2*pi to 'arc_ending_angle'
        less_than_minus_pi = null_arc_info['arc_starting_angle'] - \
            null_arc_info['arc_ending_angle'] > math.pi
        null_arc_info.loc[less_than_minus_pi,
                          'arc_ending_angle'] += 2 * math.pi

    return null_arc_info


def _deal_with_delta_angles_greater_than_180_degrees(df, reward_boundary_radius=25, ignore_error=False):
    # deal with the cases where the ff_distance is smaller than the reward boundary
    df['delta_angle'] = np.abs(
        df['arc_ending_angle'] - df['arc_starting_angle'])
    df.loc[(df['delta_angle'] > pi) & (df['ff_distance'] <= reward_boundary_radius), 'arc_ending_angle'] \
        = df.loc[(df['delta_angle'] > pi) & (df['ff_distance'] < reward_boundary_radius), 'arc_starting_angle']

    # check again if there's any delta angle larger than pi
    # if there is, raise an error
    df['delta_angle'] = np.abs(
        df['arc_ending_angle'] - df['arc_starting_angle'])
    if df[df['delta_angle'] > pi].shape[0] > 0:
        if not ignore_error:
            raise ValueError(
                f"At least one arc has an angle larger than pi, which is not possible. The largest is {np.max(df['arc_ending_angle'] - df['arc_starting_angle'])}. Please check the input.")
        else:
            print(
                f"Warning: At least one arc has an angle larger than pi, which is not possible. The largest is {np.max(df['arc_ending_angle'] - df['arc_starting_angle'])}. We will adjust them by making them a little less than pi.")

    return df


def _deal_with_delta_angles_greater_than_90_degrees(df, reward_boundary_radius=25, ignore_error=False):
    df.reset_index(drop=True, inplace=True)
    too_big_angle = df['delta_angle'] > pi/2
    within_reward_boundary = df['ff_distance'] <= reward_boundary_radius
    ff_at_left = df['arc_end_direction'] >= 0
    ff_at_right = df['arc_end_direction'] < 0
    if too_big_angle.sum() > 0:
        # deal with ff at the left side of the monkey
        df.loc[too_big_angle & ff_at_left & within_reward_boundary, 'arc_ending_angle'] \
            = df.loc[too_big_angle & ff_at_left & within_reward_boundary, 'arc_starting_angle'] + pi/2 - 0.00001
        # deal with ff at the right side of the monkey
        df.loc[too_big_angle & ff_at_right & within_reward_boundary, 'arc_ending_angle'] \
            = df.loc[too_big_angle & ff_at_right & within_reward_boundary, 'arc_starting_angle'] - (pi/2 - 0.00001)

        # try again:
        df['delta_angle'] = np.abs(
            df['arc_ending_angle'] - df['arc_starting_angle'])
        too_big_angle = df['delta_angle'] > pi/2
        if too_big_angle.sum() > 0:
            max_big_angle = df['delta_angle'].max() * 180/math.pi
            if max_big_angle > 150:
                if not ignore_error:
                    raise ValueError(
                        f"Error: max_big_angle is {max_big_angle} when ff is to the left. There is a problem here. Please check the input.")
                else:
                    print(
                        f"Warning: max_big_angle is {max_big_angle} when ff is to the left. There is a problem here. We will adjust them by making them a little less than 90.")
            print(f"Warning: {too_big_angle.sum()} arc out of {len(df)} arcs where ff is to the left of the monkey has an angle larger than 90 degrees. The max is {max_big_angle}. We will adjust them by making them a little less than 90.")
            # Otherwise, we will adjust them by making them a little less than 90
            df.loc[too_big_angle & ff_at_left, 'arc_ending_angle'] = df.loc[too_big_angle &
                                                                            ff_at_left, 'arc_starting_angle'] + pi/2 - 0.00001
            df.loc[too_big_angle & ff_at_right, 'arc_ending_angle'] = df.loc[too_big_angle &
                                                                             ff_at_right, 'arc_starting_angle'] - (pi/2 - 0.00001)
    return df
