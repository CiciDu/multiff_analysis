import math
import pandas as pd
from decision_making_analysis import trajectory_info
from null_behaviors import opt_arc_utils
from scipy.stats import rankdata

import os
import numpy as np
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_curvature_df(ff_dataframe_sub, curv_of_traj_df, ff_radius_for_opt_arc=10, clean=True,
                      monkey_information=None, ff_caught_T_new=None,
                      remove_invalid_rows=True, invalid_curvature_ok=False,
                      include_cntr_arc_curv=True, include_opt_arc_curv=True,
                      opt_arc_stop_first_vis_bdry=False,  # whether optimal arc stop at visible boundary
                      ignore_error=False):

    # it needs ff_dataframe_sub to have the following columns:
    # point_index, ff_index, monkey_x, monkey_y, monkey_angle, ff_x, ff_y, ff_distance, ff_angle, ff_angle_boundary
    ff_dataframe_sub = ff_dataframe_sub.copy()

    if remove_invalid_rows:
        ff_dataframe_sub = ff_dataframe_sub.copy()
        original_length = len(ff_dataframe_sub)
        if include_cntr_arc_curv:
            mask = ff_dataframe_sub.ff_angle.between(
                -45*math.pi/180, 45*math.pi/180)
        elif include_opt_arc_curv:
            mask = ff_dataframe_sub.ff_angle_boundary.between(
                -45*math.pi/180, 45*math.pi/180)
        # if mask is empty, then we don't need to drop any rows
        if not mask.all():
            dropped_rows = ff_dataframe_sub[~mask]
            ff_dataframe_sub = ff_dataframe_sub[mask]
            if not ignore_error:
                print(
                    f"Warning: when making curvature_df, {original_length - len(ff_dataframe_sub)} out of {original_length} rows are not within the valid ff_angle_boundary when using the function make_curvature_df.")
                # also get number of unique ff in the dropped rows of ff_dataframe_sub
                unique_ff_in_dropped_rows = dropped_rows.ff_index.unique()
                print(
                    f"Number of unique ff in the dropped rows: {len(unique_ff_in_dropped_rows)}")

    if opt_arc_stop_first_vis_bdry & (len(ff_dataframe_sub) > 100000):
        print('Warning: The number of ff is larger than 100000, and opt_arc_stop_first_vis_bdry is set to True. This might take a long time to calculate the optimal arc.')

    curv_of_traj = trajectory_info.find_trajectory_arc_info(
        ff_dataframe_sub['point_index'].values, curv_of_traj_df, monkey_information=monkey_information, ff_caught_T_new=ff_caught_T_new)

    curvature_df = _make_curvature_df(ff_dataframe_sub, curv_of_traj, ff_radius_for_opt_arc=ff_radius_for_opt_arc, clean=clean, invalid_curvature_ok=invalid_curvature_ok,
                                      include_cntr_arc_curv=include_cntr_arc_curv, include_opt_arc_curv=include_opt_arc_curv,
                                      opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry, ignore_error=ignore_error)
    return curvature_df


def _make_curvature_df(ff_dataframe_sub, curv_of_traj, ff_radius_for_opt_arc=10, clean=True,
                       invalid_curvature_ok=False,
                       include_cntr_arc_curv=True, include_opt_arc_curv=True,
                       opt_arc_stop_first_vis_bdry=True, ignore_error=False):

    # it needs ff_dataframe_sub to have the following columns:
    # point_index, ff_index, monkey_x, monkey_y, monkey_angle, ff_x, ff_y, ff_distance, ff_angle, ff_angle_boundary

    if (not include_cntr_arc_curv) and (not include_opt_arc_curv):
        raise ValueError(
            "At least one of include_cntr_arc_curv and include_opt_arc_curv should be True.")

    curvature_df = ff_dataframe_sub[['point_index', 'ff_index', 'monkey_x', 'monkey_y', 'monkey_angle',
                                    'ff_x', 'ff_y', 'ff_distance', 'ff_angle', 'ff_angle_boundary']].copy()

    if 'time' in ff_dataframe_sub.columns:
        curvature_df['time'] = ff_dataframe_sub['time'].values

    curvature_df['curv_of_traj'] = curv_of_traj

    curvature_df = supply_with_ff_curvature_info(curvature_df, ff_radius_for_opt_arc=ff_radius_for_opt_arc, invalid_curvature_ok=invalid_curvature_ok,
                                                 include_cntr_arc_curv=include_cntr_arc_curv, include_opt_arc_curv=include_opt_arc_curv,
                                                 opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry, ignore_error=ignore_error)
    add_d_heading_info(curvature_df, include_cntr_arc_curv=include_cntr_arc_curv,
                       include_opt_arc_curv=include_opt_arc_curv)

    if include_opt_arc_curv:
        curvature_df.loc[:, 'curv_diff'] = curvature_df['opt_arc_curv'].values - \
            curvature_df['curv_of_traj'].values
        curvature_df.loc[:, 'abs_curv_diff'] = np.abs(
            curvature_df.loc[:, 'curv_diff'])
        # curvature_df = curvature_df.sort_values(
        #     by=['point_index', 'abs_curv_diff', 'ff_distance'], ascending=[True, True, True])

    # if clean:
    #     clean_curvature_info(
    #         curvature_df, include_opt_arc_curv=include_opt_arc_curv)

    return curvature_df


def supply_with_ff_curvature_info(curvature_df, ff_radius_for_opt_arc=10, invalid_curvature_ok=False,
                                  include_cntr_arc_curv=True, include_opt_arc_curv=True,
                                  opt_arc_stop_first_vis_bdry=True, ignore_error=False):
    if include_cntr_arc_curv:
        curvature_df = _supply_curvature_df_with_arc_to_ff_center_info(
            curvature_df, invalid_curvature_ok=invalid_curvature_ok)
    if include_opt_arc_curv:
        curvature_df = opt_arc_utils._supply_curvature_df_with_opt_arc_info(curvature_df, ff_radius_for_opt_arc, opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry,
                                                                            ignore_error=ignore_error)
    return curvature_df


def _add_arc_ending_xy_to_ff_center_to_curvature_df(curvature_df):
    monkey_xy = curvature_df[['monkey_x', 'monkey_y']].values
    monkey_angle = curvature_df['monkey_angle'].values
    # # find arc ending xy for curv to ff center
    arc_ff_xy = curvature_df[['ff_x', 'ff_y']].values

    arc_end_x, arc_end_y = _find_arc_ending_xy_in_world_coord(arc_ff_xy, monkey_xy, monkey_angle, curvature_df['cntr_arc_radius'].values,
                                                              curvature_df['cntr_arc_end_dir'].values)
    curvature_df['cntr_arc_end_x'] = arc_end_x
    curvature_df['cntr_arc_end_y'] = arc_end_y

    return curvature_df


def _find_arc_ending_xy_in_world_coord(arc_ff_xy, monkey_xy, monkey_angle, arc_radius, arc_end_direction):

    center_x, center_y = opt_arc_utils.find_arc_center_in_world_coord(
        monkey_xy, monkey_angle, arc_radius, arc_end_direction)
    # the calculation for angle_from_center_to_stop is the same whether optimal arc or arc to ff center is used.
    angle_from_center_to_stop = opt_arc_utils.find_angle_from_arc_center_to_ff(
        arc_ff_xy, center_x, center_y)
    arc_end_x = np.cos(angle_from_center_to_stop)*arc_radius + center_x
    arc_end_y = np.sin(angle_from_center_to_stop)*arc_radius + center_y
    return arc_end_x, arc_end_y


def add_d_heading_info(curvature_df, include_cntr_arc_curv=True, include_opt_arc_curv=True):
    if include_cntr_arc_curv:
        curvature_df['cntr_arc_measure'], curvature_df['cntr_arc_length'] = find_arc_measure_and_length_to_ff_center(
            curvature_df['cntr_arc_radius'], curvature_df['ff_distance'], curvature_df['ff_angle'])
        curvature_df['cntr_arc_d_heading'] = curvature_df['cntr_arc_measure'] * \
            np.sign(curvature_df['ff_angle'].values)
    if include_opt_arc_curv:
        curvature_df['opt_arc_d_heading'] = curvature_df['opt_arc_measure'] * \
            curvature_df['opt_arc_end_direction'].values


def find_curvature_df_for_ff_in_duration(ff_dataframe, ff_index, duration_to_plot, monkey_information, curv_of_traj_df, ff_caught_T_new=None, clean=True):
    if curv_of_traj_df is None:
        raise ValueError(
            "Please provide curv_of_traj_df, since it's needed to calculate the curvature of the trajectory.")
    ff_dataframe_sub = ff_dataframe[(ff_dataframe['time'].between(duration_to_plot[0], duration_to_plot[1])) &
                                    (ff_dataframe['ff_index'] == ff_index)].copy()

    # eliminate ff whose angle is outside (-45 deg, 45 deg)
    ff_dataframe_sub = ff_dataframe_sub[ff_dataframe_sub.ff_angle.between(
        -45*math.pi/180, 45*math.pi/180)]

    curvature_df_in_duration = make_curvature_df(
        ff_dataframe_sub, curv_of_traj_df, monkey_information=monkey_information, ff_caught_T_new=ff_caught_T_new, clean=clean)
    curvature_df_in_duration['time'] = monkey_information.loc[curvature_df_in_duration['point_index'].values, 'time'].values
    return curvature_df_in_duration


def furnish_curvature_df(curvature_df, monkey_information, ff_real_position_sorted):
    curvature_df = curvature_df.copy()
    curvature_df['time'] = monkey_information.loc[curvature_df['point_index'], 'time'].values
    curvature_df['monkey_x'] = monkey_information.loc[curvature_df['point_index'], 'monkey_x'].values
    curvature_df['monkey_y'] = monkey_information.loc[curvature_df['point_index'], 'monkey_y'].values
    curvature_df['monkey_angle'] = monkey_information.loc[curvature_df['point_index'],
                                                          'monkey_angle'].values

    v = curvature_df['abs_curv_diff'].values.copy()
    curvature_df['diff_percentile'] = rankdata(v)*100/len(v)
    curvature_df['diff_percentile_in_decimal'] = curvature_df['diff_percentile'] / 100
    curvature_df['ff_x'] = ff_real_position_sorted[curvature_df['ff_index'].values, 0]
    curvature_df['ff_y'] = ff_real_position_sorted[curvature_df['ff_index'].values, 1]
    return curvature_df


def _supply_curvature_df_with_arc_to_ff_center_info(curvature_df, invalid_curvature_ok=False):
    all_ff_angle = curvature_df['ff_angle'].values.copy()
    all_ff_distance = curvature_df['ff_distance'].values.copy()
    curvature_df['cntr_arc_curv'], curvature_df['cntr_arc_radius'] = opt_arc_utils.find_arc_curvature(
        all_ff_angle, all_ff_distance, invalid_curvature_ok=invalid_curvature_ok)
    curvature_df['cntr_arc_end_dir'] = np.sign(
        curvature_df['cntr_arc_curv'])
    curvature_df['cntr_arc_measure'], curvature_df['cntr_arc_length'] = find_arc_measure_and_length_to_ff_center(
        curvature_df['cntr_arc_radius'], all_ff_distance, all_ff_angle)
    curvature_df.loc[curvature_df['ff_distance'] <= 25,
                     'cntr_arc_curv'] = curvature_df.loc[curvature_df['ff_distance'] <= 25, 'curv_of_traj']

    # also add arc end xy
    curvature_df = _add_arc_ending_xy_to_ff_center_to_curvature_df(
        curvature_df)

    return curvature_df


def find_polar_arc_center(arc_radius, arc_end_direction):
    # Assume monkey is at origin and heading to the north
    center_x = arc_radius.copy()
    center_x[arc_end_direction > 0] = -center_x[arc_end_direction > 0]
    center_y = np.zeros(len(arc_radius))
    return center_x, center_y


def find_arc_measure_and_length_to_ff_center(all_arc_radius, all_ff_distance, all_ff_angle):
    # In fact, this can be used to find the arc measure and length of optimal arc as well, still using
    # ff_distance and ff_angle to ff center, but the arc_radius is different
    # To see why this works, resort to the relevant graph in notes.

    arc_center_x = all_arc_radius.copy()
    # for ff to the left of the monkey
    arc_center_x[all_ff_angle > 0] = -arc_center_x[all_ff_angle > 0]

    # here we add math.pi/2 because originally ff_angle starts from 0 which is to the north of the monkey
    ff_x_relative_to_monkey = np.cos(all_ff_angle+math.pi/2)*all_ff_distance
    ff_y_relative_to_monkey = np.sin(all_ff_angle+math.pi/2)*all_ff_distance

    delta_x = np.abs(arc_center_x - ff_x_relative_to_monkey)
    delta_y = np.abs(ff_y_relative_to_monkey)
    # all_arc_measure = np.arctan(delta_y/delta_x)*2
    all_arc_measure = np.abs(np.arctan(delta_y/delta_x))
    all_arc_lengths = all_arc_measure * all_arc_radius

    return all_arc_measure, all_arc_lengths


def find_cartesian_arc_center_and_angle_for_arc_to_center(monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, arc_ff_xy, arc_end_direction, whether_ff_behind=None, ignore_error=False):
    center_x, center_y = opt_arc_utils.find_arc_center_in_world_coord(
        monkey_xy, monkey_angle, arc_radius, arc_end_direction)
    angle_from_center_to_monkey, angle_from_center_to_stop = opt_arc_utils.find_angle_from_arc_center_to_monkey_and_stop_position(
        arc_ff_xy, ff_angle, monkey_xy, center_x, center_y)
    arc_starting_angle, arc_ending_angle = opt_arc_utils._find_cartesian_arc_starting_and_ending_angle(angle_from_center_to_monkey, angle_from_center_to_stop, ff_distance, ff_angle, arc_end_direction,
                                                                                                       whether_ff_behind=whether_ff_behind, ignore_error=ignore_error)
    return center_x, center_y, arc_starting_angle, arc_ending_angle


def _refine_arc_starting_and_ending_angles(arc_starting_angle, arc_ending_angle, arc_end_direction):
    ff_at_left = np.where(arc_end_direction >= 0)[0]
    ff_at_right = np.where(arc_end_direction < 0)[0]

    for i in range(2):  # repeat 2 times just to make sure
        # make sure that arc_ending_angle is smaller than arc_starting_angle if ff is at the left side of the monkey, and larger if ff is at the right side
        arc_ending_angle[ff_at_left][arc_ending_angle[ff_at_left]
                                     < arc_starting_angle[ff_at_left]] += 2*pi
        arc_ending_angle[ff_at_right][arc_ending_angle[ff_at_right]
                                      > arc_starting_angle[ff_at_right]] -= 2*pi
        # make sure that the absolute value of the difference in angle is less than 2 * pi
        arc_ending_angle[ff_at_left][arc_ending_angle[ff_at_left] - arc_starting_angle[ff_at_left] > 2*pi] \
            = arc_ending_angle[ff_at_left][arc_ending_angle[ff_at_left] - arc_starting_angle[ff_at_left] > 2*pi] - 2*pi
        arc_ending_angle[ff_at_right][arc_starting_angle[ff_at_right] - arc_ending_angle[ff_at_right] > 2*pi] \
            = arc_ending_angle[ff_at_right][arc_starting_angle[ff_at_right] - arc_ending_angle[ff_at_right] > 2*pi] + 2*pi

    for i in range(2):
        # also make sure that the absolute value of the difference in angle is less than pi
        delta_angle = np.abs(arc_ending_angle - arc_starting_angle)
        if len(np.where(delta_angle > 2*pi)[0]) > 0:
            raise ValueError(
                "At least one arc has an angle larger than 2*pi, which shouldn't be. Please check the input.")
        ff_left_too_big_angle = np.where(
            (arc_end_direction >= 0) & (delta_angle > pi))[0]
        ff_right_too_big_angle = np.where(
            (arc_end_direction < 0) & (delta_angle > pi))[0]
        arc_ending_angle[ff_left_too_big_angle] = arc_starting_angle[ff_left_too_big_angle] + (
            2*pi - delta_angle[ff_left_too_big_angle])
        arc_ending_angle[ff_right_too_big_angle] = arc_starting_angle[ff_right_too_big_angle] - (
            2*pi - delta_angle[ff_right_too_big_angle])

    return arc_starting_angle, arc_ending_angle


def find_arc_end_position(all_arc_measure, all_arc_radius, arc_end_direction,
                          use_world_coordinates=False, all_point_index=None, monkey_information=None):
    # find the landing position of the monkey after the arc
    # arc_end_angles means angles from the monkey to the ends of the arcs
    ''' 
    This function is not currently used as it can be replaced by simpler functions, but the techniques used here are still useful and so are kept here for reference.

    Example use:
    arc_end_angles, arc_end_distances, arc_end_x, arc_end_y = find_arc_end_position(\
        curvature_df['opt_arc_measure'].values, curvature_df['opt_arc_radius'].values, curvature_df['opt_arc_end_direction'].values,
        use_world_coordinates=True, all_point_index=all_point_index, monkey_information=monkey_information)
    arc_end_xy = np.vstack((arc_end_x, arc_end_y)).T
    '''

    # need to differentiate left and right
    if np.any(np.abs(all_arc_measure) > math.pi/2):
        raise ValueError(
            "At least one arc has a measure larger than 90 degrees, which is not possible. Please check the input.")
    if np.any(all_arc_radius < 0):
        raise ValueError(
            "At least one arc has a arc radius smaller than 0, which is not possible. Please check the input.")

    abs_center_x = all_arc_radius.copy()
    delta_x_after_arc = np.cos(all_arc_measure)*all_arc_radius

    arc_end_x = abs_center_x - np.abs(delta_x_after_arc)
    # for ff to the left
    arc_end_x[arc_end_direction >= 0] = - arc_end_x[arc_end_direction >= 0]

    arc_end_y = np.sin(all_arc_measure)*all_arc_radius

    arc_end_distance_from_monkey = np.sqrt(arc_end_x**2 + arc_end_y**2)

    # the below seems to work, but...
    # arc_end_angle_from_monkey = np.arctan2(arc_end_y, arc_end_x)

    # we used pi/2 here since we want 0 to be pointing to the north for the monkey
    arc_end_angle_from_monkey = pi/2 - np.arctan2(arc_end_y, np.abs(arc_end_x))
    # but honestly arc_end_angle_from_monkey should just be half of arc measures

    # for ff to the right
    arc_end_angle_from_monkey[arc_end_direction < 0] = - \
        arc_end_angle_from_monkey[arc_end_direction < 0]

    # now, let's get the real arc_end_x and arc_end_y
    # first, rotated back based on monkey_angles

    if use_world_coordinates:
        if all_point_index is None or monkey_information is None:
            raise ValueError(
                "Please provide all_point_index and monkey_information, since use_world_coordinates is True.")

        # get the rotated version
        all_monkey_angles = monkey_information.loc[all_point_index,
                                                   'monkey_angle'].values
        all_monkey_x = monkey_information.loc[all_point_index,
                                              'monkey_x'].values
        all_monkey_y = monkey_information.loc[all_point_index,
                                              'monkey_y'].values

        # let 0 be pointing to the north
        all_monkey_angles = all_monkey_angles - math.pi/2

        monkey_angles_cos = np.cos(all_monkey_angles)
        monkey_angles_sin = np.sin(all_monkey_angles)
        x0 = all_monkey_x
        y0 = all_monkey_y
        arc_end_x_rotated = arc_end_x * monkey_angles_cos + \
            arc_end_y * (-monkey_angles_sin) + x0
        arc_end_y_rotated = arc_end_x * monkey_angles_sin + \
            arc_end_y * monkey_angles_cos + y0

        arc_end_x = arc_end_x_rotated
        arc_end_y = arc_end_y_rotated

    return arc_end_angle_from_monkey, arc_end_distance_from_monkey, arc_end_x, arc_end_y


def find_polar_arc_center_and_angle(arc_radius, arc_measure, arc_end_direction):
    center_x, center_y = find_polar_arc_center(arc_radius, arc_end_direction)
    arc_starting_angle, arc_ending_angle = _find_polar_arc_starting_and_ending_angles(
        arc_radius, arc_measure, arc_end_direction)
    return center_x, center_y, arc_starting_angle, arc_ending_angle


def _find_polar_arc_starting_and_ending_angles(arc_radius, arc_measure, arc_end_direction):
    arc_starting_angle = np.zeros(len(arc_radius))
    arc_starting_angle[arc_end_direction < 0] = math.pi
    arc_ending_angle = arc_starting_angle + arc_measure
    arc_ending_angle[arc_end_direction < 0] = math.pi - \
        arc_measure[arc_end_direction < 0]
    return arc_starting_angle, arc_ending_angle


# def clean_curvature_info(curvature_df, include_opt_arc_curv=True):
#     curvature_df['curv_of_traj'] = curvature_df['curv_of_traj'].clip(
#         lower=-0.5, upper=0.5)
#     if include_opt_arc_curv:
#         curvature_df['curvature_lower_bound'] = curvature_df['curvature_lower_bound'].clip(
#             lower=-20, upper=2)
#         curvature_df['curvature_upper_bound'] = curvature_df['curvature_upper_bound'].clip(
#             lower=-2, upper=20)
#         curvature_df['opt_arc_curv'] = curvature_df['opt_arc_curv'].clip(
#             lower=-0.2, upper=0.2)
#         if 'curv_diff' in curvature_df.columns:
#             curvature_df['curv_diff'] = curvature_df['curv_diff'].clip(
#                 lower=-0.6, upper=0.6)
#         if 'abs_curv_diff' in curvature_df.columns:
#             curvature_df['abs_curv_diff'] = np.abs(
#                 curvature_df['curv_diff'].values)

def clean_curvature_info(curvature_df, include_opt_arc_curv=True):
    # seems like this is no longer needed because all the columns here will be cleaned in the process of being made
    
    cols_to_winsorize = ['curv_of_traj']
    curvature_df['curv_of_traj'] = opt_arc_utils.winsorize_curv(
        curvature_df['curv_of_traj'])
    if include_opt_arc_curv:
        for col in ['curvature_lower_bound', 'curvature_upper_bound', 'opt_arc_curv']:
            if col in curvature_df.columns:
                cols_to_winsorize.append(col)
                curvature_df[col] = opt_arc_utils.winsorize_curv(
                    curvature_df[col])
    print(f'Note: Winsorized {cols_to_winsorize}')
    return curvature_df


def fill_up_NAs_for_placeholders_in_columns_related_to_curvature(df, monkey_information=None, ff_caught_T_new=None, curv_of_traj_df=None):
    if 'opt_arc_curv' in df.columns:
        # need to fill NA of columns associated with curvature. At this point, NA should only occur when ff_index = -10 (placeholders). Thus, we fill NA with 0, instead of using the function 'fill_up_NAs_in_columns_related_to_curvature'
        df['opt_arc_curv'] = df[['opt_arc_curv']].fillna(0)
    if 'curvature_lower_bound' in df.columns:
        df['curvature_lower_bound'] = df[['curvature_lower_bound']].fillna(-20)
    if 'curvature_upper_bound' in df.columns:
        df['curvature_upper_bound'] = df[['curvature_upper_bound']].fillna(20)
    if 'curv_diff' in df.columns:
        df['curv_diff'] = df[['curv_diff']].fillna(0)

    # let's also deal with the NA in curv_of_traj, but we can fill it with the curvature of the trajectory
    point_index_array = df.loc[df.curv_of_traj.isna(), 'point_index'].values

    if 'curv_of_traj' in df.columns:
        if len(point_index_array) > 0:
            curv_of_traj = trajectory_info.find_trajectory_arc_info(
                point_index_array, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new, monkey_information=monkey_information)
            df.loc[df.curv_of_traj.isna(), 'curv_of_traj'] = curv_of_traj
    return df


def fill_up_NAs_in_columns_related_to_curvature(df, monkey_information=None, ff_caught_T_new=None, curv_of_traj_df=None):
    # assume that none of the ff is a placeholder ff, but just ff that doesn't have valid arc information

    # if there's any NA in curv_of_traj, then we recalculate them
    if 'curv_of_traj' in df.columns:
        curv_of_traj_exists = True
        curv_traj_na_index = df['curv_of_traj'].isna()
        if curv_traj_na_index.any():
            if (ff_caught_T_new is None) or (curv_of_traj_df is None):
                raise ValueError(
                    "Please provide ff_caught_T_new and curv_of_traj_df, since it's needed to calculate curv_of_traj.")
            point_index_array = df.loc[curv_traj_na_index,
                                       'point_index'].values
            curv_of_traj = trajectory_info.find_trajectory_arc_info(
                point_index_array, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new, monkey_information=monkey_information)
            # we fill up the NAs for curv_of_traj
            df.loc[curv_traj_na_index, 'curv_of_traj'] = curv_of_traj
            df['curv_of_traj'] = opt_arc_utils.winsorize_curv(
                df['curv_of_traj'])
    else:
        # we temporarily add curv_of_traj to use it for filling up NAs for other columns. Then we'll drop it at the end.
        curv_of_traj_exists = False
        if (ff_caught_T_new is None) or (curv_of_traj_df is None):
            raise ValueError(
                "Please provide ff_caught_T_new and curv_of_traj_df, since it's needed to calculate curv_of_traj.")
        point_index_array = df.point_index.values
        curv_of_traj = trajectory_info.find_trajectory_arc_info(
            point_index_array, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new, monkey_information=monkey_information)
        df['curv_of_traj'] = curv_of_traj
        df['curv_of_traj'] = opt_arc_utils.winsorize_curv(
            df['curv_of_traj'])

    # Take into account the cases where the monkey is inside the reward boundary of a ff, in which case the opt_arc_curv value shall be the same as curv_of_traj
    if 'opt_arc_curv' in df.columns:
        within_ff_na_index = (df['ff_distance'] <=
                              25) & df['opt_arc_curv'].isna()
    elif 'curv_diff' in df.columns:
        within_ff_na_index = (df['ff_distance'] <= 25) & df['curv_diff'].isna()
    elif 'abs_curv_diff' in df.columns:
        within_ff_na_index = (df['ff_distance'] <=
                              25) & df['abs_curv_diff'].isna()
    else:
        raise ValueError(
            "Please provide either opt_arc_curv, curv_diff, or abs_curv_diff in df.columns. Otherwise, change the algorithm here.")

    if 'opt_arc_curv' in df.columns:
        df.loc[within_ff_na_index,
               'opt_arc_curv'] = df.loc[within_ff_na_index, 'curv_of_traj'].values
        df.loc[within_ff_na_index, 'opt_arc_curv'] = opt_arc_utils.winsorize_curv(
            df.loc[within_ff_na_index, 'opt_arc_curv'])

    # Fill NA values for other columns
    for column in ['curvature_lower_bound', 'curvature_upper_bound', 'curv_diff', 'abs_curv_diff']:
        if column in df.columns:
            df.loc[within_ff_na_index, column] = 0 if 'curv' in column else - \
                200 if 'lower' in column else 200

    # if 'curvature_lower_bound' in df.columns:
    #     df.loc[within_ff_na_index, 'curvature_lower_bound'] = -200
    # if 'curvature_upper_bound' in df.columns:
    #     df.loc[within_ff_na_index, 'curvature_upper_bound'] = 200
    # if 'curv_diff' in df.columns:
    #     df.loc[within_ff_na_index, 'curv_diff'] = 0
    # if 'abs_curv_diff' in df.columns:
    #     df.loc[within_ff_na_index, 'abs_curv_diff'] = 0

    # Now, let's deal with the rest of the ff
    if 'curvature_lower_bound' in df.columns:  # this should mean that curvature_lower_bound is also in df.columns
        ff_left_na_index = (df['ff_angle_boundary'] >
                            0) & df['curvature_lower_bound'].isna()
        df.loc[ff_left_na_index, ['curvature_lower_bound',
                                  'curvature_upper_bound']] = np.array([0, 200])  # note: positive number is to the left
        ff_right_na_index = (df['ff_angle_boundary'] <
                             0) & df['curvature_lower_bound'].isna()
        df.loc[ff_right_na_index, ['curvature_lower_bound',
                                   'curvature_upper_bound']] = np.array([-200, 0])
        middle_ff_na_index = (df['ff_angle_boundary']
                              == 0) & df['opt_arc_curv'].isna()
        df.loc[middle_ff_na_index, ['curvature_lower_bound',
                                    'curvature_upper_bound']] = np.array([-200, 200])

    if 'opt_arc_curv' in df.columns:
        ff_left_na_index = (df['ff_angle_boundary'] >
                            0) & df['opt_arc_curv'].isna()
        df.loc[ff_left_na_index, 'opt_arc_curv'] = opt_arc_utils.CURV_OR_TRAJ_UPPER_BOUND
        ff_right_na_index = (df['ff_angle_boundary'] <
                             0) & df['opt_arc_curv'].isna()
        df.loc[ff_right_na_index, 'opt_arc_curv'] = opt_arc_utils.CURV_OR_TRAJ_LOWER_BOUND
        middle_ff_na_index = (df['ff_angle_boundary']
                              == 0) & df['opt_arc_curv'].isna()
        df.loc[middle_ff_na_index, 'opt_arc_curv'] = 0

    df['curv_diff'] = df['opt_arc_curv'].values - df['curv_of_traj'].values
    df['abs_curv_diff'] = np.abs(df['curv_diff'].values)

    if not curv_of_traj_exists:
        df.drop(columns=['curv_of_traj'], inplace=True)

    return df


def add_arc_info_to_df(df, curvature_df, arc_info_to_add=['opt_arc_curv', 'curv_diff'], ff_caught_T_new=None, curv_of_traj_df=None):
    if curvature_df is None:
        raise ValueError('curvature_df is None, but add_arc_info is True')
    curvature_df_sub = curvature_df[[
        'ff_index', 'point_index'] + arc_info_to_add].copy()
    arc_info_df = pd.merge(df[['ff_index', 'point_index']], curvature_df_sub, on=[
                           'ff_index', 'point_index'], how='left')
    # for the NAs (when ff_index is -10)
    df[arc_info_to_add] = arc_info_df[arc_info_to_add]
    df = fill_up_NAs_in_columns_related_to_curvature(
        df, ff_caught_T_new=ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
    return df


def add_column_monkey_passed_by_to_curvature_df(curvature_df, ff_dataframe, monkey_information):
    curvature_df = curvature_df.copy()
    pass_by_within_next_n_seconds = 2.5
    pass_by_within_n_cm = 50
    curvature_df['monkey_passed_by'] = False
    unique_ff_index = curvature_df.ff_index.unique()
    monkey_t_array = monkey_information.time.values
    point_index_array = monkey_information.point_index.values
    print('There are %d unique ff_index in the curvature_df' %
          len(unique_ff_index))

    index_to_include_df = pd.DataFrame(columns=['ff_index', 'point_index'])

    for i in range(len(unique_ff_index)):
        if i % 100 == 0:
            print('Now at %d out of %d' % (i, len(unique_ff_index)))
        ff_index = int(unique_ff_index[i])
        # find the time when monkey is close to ff_index
        ff_dataframe_sub = ff_dataframe[(ff_dataframe['ff_index'] == ff_index) & (
            ff_dataframe.ff_distance <= pass_by_within_n_cm)].copy()
        # include those time points and 2.5s before those time points
        ff_dataframe_sub['start_of_duration'] = ff_dataframe_sub.time - \
            pass_by_within_next_n_seconds
        duration_df = ff_dataframe_sub[['time', 'start_of_duration']].copy().rename(
            columns={'time': 'end_of_duration'})
        # find the overlaps of these durations defined by time and time_minus_2_5s
        duration_df['end_of_previous_duration'] = duration_df['end_of_duration'].shift(
            1).fillna(0)
        duration_df['overlap_with_previous_duration'] = duration_df['start_of_duration'] <= duration_df['end_of_previous_duration']
        # fill the column with NA
        duration_df.loc[duration_df['overlap_with_previous_duration']
                        == True, 'start_of_duration'] = np.nan

        # Forward-fill start times
        duration_df['start_of_duration'] = duration_df['start_of_duration'].fillna(
            method='ffill')

        # Flag if the *next* row overlaps
        duration_df['overlap_with_next_duration'] = (
            duration_df['overlap_with_previous_duration']
            .shift(-1)
            .fillna(False)
        )

        # Set end_of_duration to NA where overlap exists
        duration_df.loc[duration_df['overlap_with_next_duration'],
                        'end_of_duration'] = pd.NA

        # Backward-fill end times
        duration_df['end_of_duration'] = duration_df['end_of_duration'].fillna(
            method='bfill')

        # now, let's only keep unique rows
        duration_df = duration_df[['start_of_duration',
                                   'end_of_duration']].drop_duplicates()

        # Method 1
        # within_any_duration = []
        # # find the point_index defined by the union of these durations
        # for index, row in duration_df.iterrows():
        #     within_any_duration.extend(np.where((monkey_t_array >= row.start_of_duration) & (monkey_t_array <= row.end_of_duration))[0].tolist())

        # Method 2
        # time_into_bins = np.searchsorted(duration_df.values.reshape(-1), monkey_t_array)
        # within_any_duration = np.where(time_into_bins%2==1)[0]

        # Method 3
        bin_inserted_in_time_array = np.searchsorted(
            monkey_t_array, duration_df.values, side='right')
        within_any_duration = []
        for duration_in_point_index in bin_inserted_in_time_array:
            within_any_duration.extend(
                range(duration_in_point_index[0], duration_in_point_index[1]))

        if len(within_any_duration) > 0:
            index_to_include = point_index_array[np.array(within_any_duration)]
            index_to_include_df = pd.concat([index_to_include_df, pd.DataFrame({'ff_index': np.repeat(
                ff_index, len(index_to_include)), 'point_index': index_to_include})], ignore_index=True)

    index_to_include_df['monkey_passed_by'] = True
    if 'monkey_passed_by' in curvature_df.columns:
        curvature_df.drop(columns=['monkey_passed_by'], inplace=True)
    curvature_df = curvature_df.merge(
        index_to_include_df, on=['ff_index', 'point_index'], how='left')
    curvature_df.fillna(False, inplace=True)
    return curvature_df
