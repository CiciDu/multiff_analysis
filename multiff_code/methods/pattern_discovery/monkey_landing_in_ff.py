from data_wrangling import specific_utils, general_utils
from planning_analysis.show_planning import examine_null_arcs

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from math import pi
import os
import math

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
pd.options.display.max_rows = 101


def iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, initial_duration_before_stop=2, reward_boundary_radius=25):
    # This function will iteratively get monkey's info before the stop until it contains at least one point before entering the reward boundary
    duration_before_stop = initial_duration_before_stop
    duration_counter = 1
    monkey_sub = _get_subset_of_monkey_info(
        monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=duration_counter)
    # if all the points are within reward_boundary, then we might not have covered all points, in which case we will use a longer time
    while monkey_sub['distance_to_ff_center'].max() < reward_boundary_radius:
        duration_counter += 1
        print(f"duration_counter: {duration_counter}")
        monkey_sub = _get_subset_of_monkey_info(
            monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=duration_counter)
    monkey_sub = monkey_sub[monkey_sub['distance_to_ff_center']
                            <= reward_boundary_radius]
    return monkey_sub


def _get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, ff_index, time, duration_before_stop, duration_counter=1):
    # for the 2 seconds before the stop, get the monkey's trajectory and calculate the distance to the ff center
    monkey_sub = monkey_information[monkey_information['time'].between(
        time-duration_before_stop * duration_counter, time)].copy()
    monkey_sub[['ff_x', 'ff_y']] = ff_real_position_sorted[ff_index]
    monkey_sub['distance_to_ff_center'] = np.sqrt(
        (monkey_sub['ff_x'] - monkey_sub['monkey_x'])**2 + (monkey_sub['ff_y'] - monkey_sub['monkey_y'])**2)
    return monkey_sub


def find_mxy_rotated_w_ff_center_to_the_north(monkey_sub, ff_center, reward_boundary_radius=25):
    # get info to plot the monkey's trajectory into the circle. We assume the entry point at the reward boundary is the origin,
    # and we let the ff center to toward the north

    monkey_sub2 = monkey_sub[monkey_sub['distance_to_ff_center']
                             <= reward_boundary_radius].copy()
    monkey_x = monkey_sub2['monkey_x'].values
    monkey_y = monkey_sub2['monkey_y'].values

    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(ff_center[1]-monkey_y[0], ff_center[0]-monkey_x[0])
    c, s = np.cos(theta), np.sin(theta)
    # Find the rotation matrix
    rotation_matrix = np.array(((c, -s), (s, c)))

    mxy_rotated = np.matmul(rotation_matrix, np.stack((monkey_x, monkey_y)))
    ff_center_rotated = np.matmul(rotation_matrix, ff_center)

    x0 = ff_center_rotated[0]
    y0 = ff_center_rotated[1]

    return mxy_rotated, x0, y0


def plot_monkey_landings_in_ff(closest_stop_to_capture_df,
                               monkey_information,
                               ff_real_position_sorted,
                               starting_trial=0,
                               max_trials=100,
                               reward_boundary_radius=25,
                               ax=None,
                               color='blue',
                               plot_path_to_landing=True,
                               show_plot=True):

    # for each point_index & corresponding ff_index, find the point where monkey first enters the reward boundary
    # then make a polar plot, using that entry point as the reference, and let the ff center to be to the north
    # then plot the monkey's trajectory into the circle

    trial_counter = 1
    if ax is None:
        fig, ax = plt.subplots()

    for index, row in closest_stop_to_capture_df.iloc[starting_trial:].iterrows():
        # point_index = int(row.point_index)

        monkey_sub = iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, int(
            row.cur_ff_index), row.time, initial_duration_before_stop=2, reward_boundary_radius=reward_boundary_radius)
        if len(monkey_sub) == 0:
            print(
                f"no data where monkey is inside reward boundary for ff_index {int(row.cur_ff_index)} at time around {row.time}")
            continue

        monkey_sub.sort_values('time', inplace=True)
        mxy_rotated, x0, y0 = find_mxy_rotated_w_ff_center_to_the_north(
            monkey_sub, ff_center=ff_real_position_sorted[int(row.cur_ff_index)], reward_boundary_radius=reward_boundary_radius)

        ax = examine_null_arcs.show_xy_overlapped(ax, mxy_rotated, x0, y0, color=color, plot_path_to_landing=plot_path_to_landing)

        if trial_counter > max_trials:
            break
        trial_counter += 1

    # plot the ff center
    ax.plot(0, 0, '*', markersize=10, color='brown')
    
    # show the reward boundary
    ax = examine_null_arcs._make_a_circle_to_show_reward_boundary(
        ax, reward_boundary_radius=reward_boundary_radius, set_xy_limit=True, color='purple')

    # also plot the visible boundary
    ax = examine_null_arcs._make_a_circle_to_show_reward_boundary(
        ax, reward_boundary_radius=10, set_xy_limit=False, color='g')

    if show_plot:
        plt.show()
        return
    else:
        return ax


def add_distance_from_ff_to_stop(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted):
    df = closest_stop_to_capture_df.copy()
    df[['cur_ff_x', 'cur_ff_y']] = ff_real_position_sorted[df['cur_ff_index'].values]

    df[['monkey_x', 'monkey_y']] = monkey_information.loc[df['stop_point_index'].values, [
        'monkey_x', 'monkey_y']].values

    # find the point index where the monkey was caught
    pos_index = np.searchsorted(
        monkey_information['time'].values, df['caught_time'].values)
    pos_index[pos_index == len(monkey_information)] = len(
        monkey_information) - 1
    df['caught_time_point_index'] = monkey_information['point_index'].iloc[pos_index].values

    df[['caught_time_monkey_x', 'caught_time_monkey_y']
       ] = monkey_information.loc[df['caught_time_point_index'].values, ['monkey_x', 'monkey_y']].values

    closest_stop_to_capture_df['distance_from_ff_to_stop'] = np.sqrt(
        (df['cur_ff_x'] - df['monkey_x'])**2 + (df['cur_ff_y'] - df['monkey_y'])**2)

    return closest_stop_to_capture_df


def find_angle_from_ff_center_to_monkey_stop(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted,
                                             reward_boundary_radius=25):

    list_of_angle = []
    list_of_rel_stop_x = []
    list_of_rel_stop_y = []
    list_of_cur_ff_index = []
    list_of_stop_point_index = []
    for index, row in closest_stop_to_capture_df.iterrows():

        monkey_sub = iterate_to_get_subset_of_monkey_info(monkey_information, ff_real_position_sorted, int(row.cur_ff_index), row.time, initial_duration_before_stop=2,
                                                          reward_boundary_radius=reward_boundary_radius)
        if len(monkey_sub) == 0:
            # print(f"no monkey_sub for ff_index {row.cur_ff_index} at time {row.time}")
            continue

        monkey_sub.sort_values('time', inplace=True)

        mxy_rotated, x0, y0 = find_mxy_rotated_w_ff_center_to_the_north(
            monkey_sub, ff_center=ff_real_position_sorted[int(row.cur_ff_index)])
        stop_x = mxy_rotated[0, -1]-x0
        stop_y = mxy_rotated[1, -1]-y0
        ff_x = 0
        ff_y = 0
        # calculate the angle from the ff center to the monkey's stop point ("monkey_angle" will be just to the north)
        angle = specific_utils.calculate_angles_to_ff_centers(
            ff_x=stop_x, ff_y=stop_y, mx=ff_x, my=ff_y, m_angle=math.pi/2)
        list_of_angle.append(angle)
        list_of_rel_stop_x.append(stop_x)
        list_of_rel_stop_y.append(stop_y)
        list_of_cur_ff_index.append(row.cur_ff_index)
        list_of_stop_point_index.append(row.stop_point_index)

    df = pd.DataFrame({'stop_point_index': list_of_stop_point_index, 'cur_ff_index': list_of_cur_ff_index,
                       'angle_from_ff_to_stop': list_of_angle,
                       'stop_x_rel_to_ff': list_of_rel_stop_x, 'stop_y_rel_to_ff': list_of_rel_stop_y})

    df['angle_in_degrees_from_ff_to_stop'] = np.degrees(
        df['angle_from_ff_to_stop'])

    return df


def add_angle_from_ff_center_to_monkey_stop(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted,
                                            reward_boundary_radius=25):

    df = find_angle_from_ff_center_to_monkey_stop(
        closest_stop_to_capture_df, monkey_information, ff_real_position_sorted, reward_boundary_radius=reward_boundary_radius)
    # in case the distance is already calculated, we will drop it
    closest_stop_to_capture_df2 = closest_stop_to_capture_df.merge(
        df, on=['stop_point_index', 'cur_ff_index'], how='left')
    return closest_stop_to_capture_df2


def get_valid_subset_to_construct_scatter_around_target_df(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted):
    closest_stop_to_capture_df2 = add_angle_from_ff_center_to_monkey_stop(
        closest_stop_to_capture_df, monkey_information, ff_real_position_sorted)
    valid_subset = closest_stop_to_capture_df2[closest_stop_to_capture_df2['distance_from_ff_to_stop'] <= 25].copy(
    )
    total_len = len(closest_stop_to_capture_df)
    invalid_len = total_len - len(valid_subset)
    print(f'{invalid_len} out of {total_len} are not within 25 cm of ff center, which is {invalid_len/total_len*100:.2f}%. These are excluded')
    return valid_subset


def plot_scatter_around_target_df(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted):
    valid_subset = get_valid_subset_to_construct_scatter_around_target_df(
        closest_stop_to_capture_df, monkey_information, ff_real_position_sorted)
    plot_monkey_landings_in_ff(
        valid_subset, monkey_information, ff_real_position_sorted)


def describe_and_rename_one_variable(df, prefix=''):
    desc = df.describe().T[['mean', 'std', '25%', '50%', '75%']].rename(
        columns={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
    desc['iqr'] = desc['Q3'] - desc['Q1']
    desc.columns = [prefix + col for col in desc.columns]
    desc.reset_index(inplace=True, drop=True)
    return desc


def make_scatter_around_target_df(monkey_information, closest_stop_to_capture_df, ff_real_position_sorted, data_folder_name=None, make_plot=False):
    valid_subset = get_valid_subset_to_construct_scatter_around_target_df(
        closest_stop_to_capture_df, monkey_information, ff_real_position_sorted)

    if make_plot:
        plot_monkey_landings_in_ff(
            valid_subset, monkey_information, ff_real_position_sorted)

    # Describe and rename columns for distance, angle, and abs_angle
    scatter_around_target_df = describe_and_rename_one_variable(
        valid_subset[['distance_from_ff_to_stop']], 'distance_')
    scatter_around_target_df_angle = describe_and_rename_one_variable(
        valid_subset[['angle_in_degrees_from_ff_to_stop']], 'angle_')
    scatter_around_target_df_abs_angle = describe_and_rename_one_variable(
        valid_subset[['angle_in_degrees_from_ff_to_stop']].abs(), 'abs_angle_')

    # Concatenate all descriptions into one DataFrame
    scatter_around_target_df = pd.concat(
        [scatter_around_target_df, scatter_around_target_df_angle, scatter_around_target_df_abs_angle], axis=1)

    # Get the percentage of points in each quadrant
    quadrants = {1: [-90, 0], 2: [0, 90], 3: [90, 180], 4: [-180, -90]}
    for quad, (low, high) in quadrants.items():
        valid_subset_quad = valid_subset[(valid_subset['angle_in_degrees_from_ff_to_stop'] >= low) & (
            valid_subset['angle_in_degrees_from_ff_to_stop'] < high)]
        col_name = f'Q{quad}_perc'
        scatter_around_target_df[col_name] = len(
            valid_subset_quad) / len(valid_subset) * 100

    if data_folder_name is not None:
        general_utils.save_df_to_csv(
            scatter_around_target_df, 'scatter_around_target_df', data_folder_name)

    return scatter_around_target_df

def get_closest_stop_time_to_all_capture_time(ff_caught_T_sorted, monkey_information, ff_real_position_sorted, cur_ff_index_array=None, stop_point_index_array=None):
    stop_sub = monkey_information.loc[monkey_information['monkey_speeddummy'] == 0, [
        'time', 'point_index']].copy()
    closest_stop_to_capture_df = pd.DataFrame()
    for i in range(len(ff_caught_T_sorted)):
        caught_time = ff_caught_T_sorted[i]
        time = stop_sub['time'].values
        iloc_of_closest_time = np.argmin(np.abs(time - caught_time))
        closest_point_row = stop_sub.iloc[[iloc_of_closest_time]].copy()
        closest_point_row['caught_time'] = caught_time
        closest_stop_to_capture_df = pd.concat(
            [closest_stop_to_capture_df, closest_point_row], axis=0)
    closest_stop_to_capture_df['diff_from_caught_time'] = closest_stop_to_capture_df['time'] - \
        closest_stop_to_capture_df['caught_time']
    if cur_ff_index_array is not None:
        closest_stop_to_capture_df['cur_ff_index'] = cur_ff_index_array
    else:
        closest_stop_to_capture_df['cur_ff_index'] = np.arange(
            len(ff_caught_T_sorted))
    if stop_point_index_array is not None:
        closest_stop_to_capture_df['stop_point_index'] = stop_point_index_array
    else:
        closest_stop_to_capture_df['stop_point_index'] = closest_stop_to_capture_df['point_index']
    closest_stop_to_capture_df['stop_time'] = monkey_information.loc[closest_stop_to_capture_df['point_index'], 'time'].values

    closest_stop_to_capture_df = add_distance_from_ff_to_stop(
        closest_stop_to_capture_df, monkey_information, ff_real_position_sorted)
    closest_stop_to_capture_df['whether_stop_inside_boundary'] = closest_stop_to_capture_df['distance_from_ff_to_stop'] <= 25
    closest_stop_to_capture_df.sort_values(by='cur_ff_index', inplace=True)
    return closest_stop_to_capture_df



import numpy as np
import pandas as pd
import warnings
from typing import Optional

def get_closest_stop_to_all_capture_position(
    ff_caught_T_sorted: np.ndarray,
    monkey_information: pd.DataFrame,
    ff_real_position_sorted: np.ndarray,
    cur_ff_index_array: Optional[np.ndarray] = None,
    stop_point_index_array: Optional[np.ndarray] = None,
    capture_window_width: float = 0.2,
    boundary_cm: float = 25.0,
) -> pd.DataFrame:
    """
    Mirror-vectorized (broadcasted) nearest-stop-to-capture selector.

    For each capture time (len F), choose among all stops (len S):
      1) restrict to stops within ±capture_window_width of the capture time;
      2) among those, pick the spatially closest to the firefly position;
      3) if none exist in the window, fall back to the stop closest in time (globally).

    Returns one row per capture, sorted by cur_ff_index.
    """
    # --- validation ---
    needed_cols = {"time", "monkey_speeddummy", "monkey_x", "monkey_y", "point_index"}
    missing = needed_cols - set(monkey_information.columns)
    if missing:
        raise ValueError(f"monkey_information missing columns: {missing}")

    if ff_caught_T_sorted.ndim != 1:
        raise ValueError("ff_caught_T_sorted must be 1-D (sorted times).")
    F = len(ff_caught_T_sorted)

    if ff_real_position_sorted.shape != (F, 2):
        ff_real_position_sorted = ff_real_position_sorted[:len(ff_caught_T_sorted)]
        assert ff_real_position_sorted.shape == (F, 2)

    # --- gather all stops ---
    stop_sub = monkey_information.loc[
        monkey_information["monkey_speeddummy"] == 0,
        ["time", "point_index", "monkey_x", "monkey_y"],
    ].copy()

    if stop_sub.empty:
        warnings.warn("No stop samples (monkey_speeddummy==0) found at all.")
        return pd.DataFrame(
            columns=[
                "point_index", "time", "monkey_x", "monkey_y",
                "caught_time", "cur_ff_index", "stop_point_index",
                "stop_time", "distance_from_ff_to_stop",
                "whether_stop_inside_boundary", "diff_from_caught_time",
            ]
        )

    stop_t = stop_sub["time"].to_numpy()               # (S,)
    stop_xy = stop_sub[["monkey_x", "monkey_y"]].to_numpy()  # (S, 2)
    S = stop_t.shape[0]

    # --- broadcasted pairwise computations (S x F) ---
    # time diffs
    tdiff = np.abs(stop_t[:, None] - ff_caught_T_sorted[None, :])   # (S, F)
    # window mask
    in_window = tdiff <= capture_window_width                       # (S, F)

    # spatial distances
    dx = stop_xy[:, 0, None] - ff_real_position_sorted[None, :, 0]  # (S, F)
    dy = stop_xy[:, 1, None] - ff_real_position_sorted[None, :, 1]  # (S, F)
    dist = np.hypot(dx, dy)                                         # (S, F)

    # pick spatial nearest among windowed stops by masking others to +inf
    dist_masked = np.where(in_window, dist, np.inf)                 # (S, F)
    idx_spatial = np.argmin(dist_masked, axis=0)                    # (F,)

    # detect captures with no stops inside window (column all inf)
    no_win = ~np.any(in_window, axis=0)                             # (F,)
    if np.any(no_win):
        # fallback: nearest in time (globally)
        idx_time = np.argmin(tdiff, axis=0)                         # (F,)
        # use time-closest index where window was empty
        idx_final = idx_spatial.copy()
        idx_final[no_win] = idx_time[no_win]
        # one aggregated warning (keeps this function mostly vectorized)
        bad_idxs = np.nonzero(no_win)[0]
        warnings.warn(
            f"No stops in ±{capture_window_width:.3f}s for captures at indices: {bad_idxs.tolist()} "
            f"(times: {np.round(ff_caught_T_sorted[bad_idxs], 3).tolist()}); "
            "used nearest-in-time stops for those."
        )
    else:
        idx_final = idx_spatial

    # --- gather selected rows in one shot ---
    chosen = stop_sub.iloc[idx_final].copy()                        # (F, 4)
    chosen.reset_index(drop=True, inplace=True)

    # per-capture spatial distance values (index along axis 0 using idx_final)
    # dist is (S, F); take along rows with idx_final to get shape (1, F)
    sel_dist = np.take_along_axis(dist, idx_final[None, :], axis=0).ravel()

    # build output columns (all vectorized)
    chosen["caught_time"] = ff_caught_T_sorted
    if cur_ff_index_array is not None:
        if len(cur_ff_index_array) != F:
            raise ValueError("cur_ff_index_array must have same length as ff_caught_T_sorted.")
        chosen["cur_ff_index"] = cur_ff_index_array.astype(int)
    else:
        chosen["cur_ff_index"] = np.arange(F, dtype=int)

    if stop_point_index_array is not None:
        if len(stop_point_index_array) != F:
            raise ValueError("stop_point_index_array must have same length as ff_caught_T_sorted.")
        chosen["stop_point_index"] = stop_point_index_array.astype(int)
    else:
        chosen["stop_point_index"] = chosen["point_index"].astype(int)

    chosen["stop_time"] = chosen["time"]
    chosen["distance_from_ff_to_stop"] = sel_dist.astype(float)
    chosen["whether_stop_inside_boundary"] = chosen["distance_from_ff_to_stop"] <= float(boundary_cm)
    chosen["diff_from_caught_time"] = chosen["time"] - chosen["caught_time"]

    # final order & clean index
    chosen.sort_values("cur_ff_index", inplace=True)
    chosen.reset_index(drop=True, inplace=True)
    return chosen


