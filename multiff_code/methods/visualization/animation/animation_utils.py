from visualization.matplotlib_tools import plot_behaviors_utils
from null_behaviors import show_null_trajectory, find_best_arc
from matplotlib.lines import Line2D
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi

# -----------------------------------------------------------------------------
# Module-level settings
# -----------------------------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rotate_points(xy: np.ndarray, R: np.ndarray, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Apply rotation and translation to Nx2 points.

    Parameters
    ----------
    xy : (N, 2) array
        Input points (x, y).
    R : (2, 2) array
        Rotation matrix.
    x0, y0 : float
        Translation (subtract after rotation).

    Returns
    -------
    (N, 2) array
        Rotated and translated points.
    """
    if xy.size == 0:
        return xy
    return (R @ xy.T).T - np.array([x0, y0])


def _choose_ff_columns(df):
    """Return (x_col, y_col) for real and optional noisy pose columns in df."""
    if 'ff_x_rotated' in df.columns:
        xr, yr = 'ff_x_rotated', 'ff_y_rotated'
    else:
        xr, yr = 'ff_x', 'ff_y'

    xn, yn = None, None
    if 'ff_x_noisy' in df.columns:
        if 'ff_x_noisy_rotated' in df.columns:
            xn, yn = 'ff_x_noisy_rotated', 'ff_y_noisy_rotated'
        else:
            xn, yn = 'ff_x_noisy', 'ff_y_noisy'
    return xr, yr, xn, yn


# -----------------------------------------------------------------------------
# Core preparation
# -----------------------------------------------------------------------------

def prepare_for_animation(ff_dataframe,
                          ff_caught_T_new,
                          ff_life_sorted,
                          ff_believed_position_sorted,
                          ff_real_position_sorted,
                          ff_flash_sorted,
                          monkey_information,
                          duration=None,
                          currentTrial=None,
                          num_trials=None,
                          k=1,
                          rotated=True,
                          max_duration=30,
                          min_duration=1):
    """
    Prepare data structures used by the animation pipeline.

    Notes
    -----
    - Does not change input/output names.
    - Vectorized slices; consistent single quotes; fewer side effects.
    """

    # Infer duration from trials if not provided
    if duration is None:
        if currentTrial is None:
            # pick the most recent trial that exists
            currentTrial = len(ff_caught_T_new) - 1
        if num_trials is None:
            num_trials = min(2, currentTrial + 1)
        if num_trials > (currentTrial + 1):
            raise ValueError('num_trials must be ≤ currentTrial + 1')
        duration = [ff_caught_T_new[currentTrial - num_trials + 1],
                    ff_caught_T_new[currentTrial]]

    # Clamp over/under-long durations
    span = duration[1] - duration[0]
    if (max_duration is not None) and (span > max_duration):
        duration = [duration[1] - max_duration, duration[1]]
        print(
            f'The duration is too long. Showing the last {max_duration} seconds.')

    if (min_duration is not None) and (duration[1] - duration[0] < min_duration):
        duration = [duration[1] - min_duration, duration[1]]
        print(
            f'The duration is too short. Showing the last {min_duration} seconds.')

    # If trial info missing, best-effort infer from duration bounds
    if currentTrial is None or num_trials is None:
        try:
            earlier_trials = np.where(ff_caught_T_new <= duration[1])[0]
            currentTrial = earlier_trials[-1] if len(earlier_trials) > 0 else 0
            first_after_start = np.where(ff_caught_T_new > duration[0])[0]
            num_trials = currentTrial - \
                first_after_start[0] if len(first_after_start) else 1
        except Exception as e:
            print(f'Could not infer currentTrial/num_trials: {e}')
            currentTrial, num_trials = None, None

    # Build animation indices within the window
    t = monkey_information['time'].to_numpy()
    idx = np.where((t > duration[0]) & (t <= duration[1]))[0]

    anim_monkey_info = make_anim_monkey_info(monkey_information, idx, k=k)

    ff_dataframe_anim = ff_dataframe.loc[(ff_dataframe['time'] >= duration[0]) &
                                         (ff_dataframe['time'] <= duration[1])].copy()

    # Optional rotation to egocentric coordinates (start-of-window at origin)
    if rotated:
        R, theta = plot_behaviors_utils.find_rotation_matrix(
            anim_monkey_info['anim_mx'], anim_monkey_info['anim_my'], also_return_angle=True)
        anim_monkey_info, x0, y0 = rotated_anim_monkey_info(
            anim_monkey_info, R)
        anim_monkey_info['anim_angle'] = anim_monkey_info['anim_angle'] + theta

        ff_dataframe_anim.loc[:, ['ff_x_rotated', 'ff_y_rotated']] = _rotate_points(
            ff_dataframe_anim[['ff_x', 'ff_y']].to_numpy(), R, x0, y0)
        ff_real_position_rotated = _rotate_points(
            ff_real_position_sorted, R, x0, y0)

        if 'ff_x_noisy' in ff_dataframe_anim.columns:
            ff_dataframe_anim.loc[:, ['ff_x_noisy_rotated', 'ff_y_noisy_rotated']] = _rotate_points(
                ff_dataframe_anim[['ff_x_noisy', 'ff_y_noisy']].to_numpy(), R, x0, y0)
    else:
        R = np.eye(2)
        x0, y0 = 0.0, 0.0
        anim_monkey_info['x0'], anim_monkey_info['y0'] = x0, y0
        ff_real_position_rotated = ff_real_position_sorted

    anim_monkey_info['rotation_matrix'] = R
    anim_monkey_info['ff_real_position_rotated'] = ff_real_position_rotated

    # Visibility/alive/believed mappings over animation frames
    flash_on_ff_dict = match_points_to_flash_on_ff_positions(
        anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], duration,
        ff_flash_sorted, ff_life_sorted, ff_real_position_sorted, rotation_matrix=R, x0=x0, y0=y0)

    alive_ff_dict = match_points_to_alive_ff_positions(
        anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], ff_caught_T_new,
        ff_life_sorted, ff_real_position_sorted, rotation_matrix=R, x0=x0, y0=y0)

    believed_ff_dict = match_points_to_believed_ff_positions(
        anim_monkey_info['anim_t'], anim_monkey_info['anim_indices'], currentTrial, num_trials,
        ff_believed_position_sorted, ff_caught_T_new, rotation_matrix=R, x0=x0, y0=y0)

    # Matplotlib defaults
    plt.rcParams['font.size'] = 15
    plt.rcParams['savefig.dpi'] = 100

    num_frames = anim_monkey_info['anim_t'].size
    return num_frames, anim_monkey_info, flash_on_ff_dict, alive_ff_dict, believed_ff_dict, num_trials, ff_dataframe_anim


# -----------------------------------------------------------------------------
# Ranges and animation info
# -----------------------------------------------------------------------------

def find_xy_min_max_for_animation(anim_monkey_info, ff_dataframe_anim):
    mx_min, mx_max = anim_monkey_info['xmin'], anim_monkey_info['xmax']
    my_min, my_max = anim_monkey_info['ymin'], anim_monkey_info['ymax']

    visible_ffs = ff_dataframe_anim.loc[ff_dataframe_anim['visible'] == 1]
    if len(visible_ffs) > 0:
        if 'ff_x_rotated' in visible_ffs.columns:
            mx_min, mx_max = min(mx_min, visible_ffs.ff_x_rotated.min()), max(
                mx_max, visible_ffs.ff_x_rotated.max())
            my_min, my_max = min(my_min, visible_ffs.ff_y_rotated.min()), max(
                my_max, visible_ffs.ff_y_rotated.max())
        else:
            mx_min, mx_max = min(mx_min, visible_ffs.ff_x.min()), max(
                mx_max, visible_ffs.ff_x.max())
            my_min, my_max = min(my_min, visible_ffs.ff_y.min()), max(
                my_max, visible_ffs.ff_y.max())
    return mx_min, mx_max, my_min, my_max


def make_anim_monkey_info(monkey_information, cum_pos_index, k=3):
    """
    Build subsampled (every k points) trajectory & state arrays for animation.
    """
    cum_pos_index = np.asarray(cum_pos_index)

    cum_t = monkey_information['time'].to_numpy()[cum_pos_index]
    cum_angle = monkey_information['monkey_angle'].to_numpy()[cum_pos_index]
    cum_mx = monkey_information['monkey_x'].to_numpy()[cum_pos_index]
    cum_my = monkey_information['monkey_y'].to_numpy()[cum_pos_index]
    cum_speed = monkey_information['speed'].to_numpy()[cum_pos_index]

    anim_indices = cum_pos_index[0:-1:k]
    anim_t = cum_t[0:-1:k]
    anim_mx = cum_mx[0:-1:k]
    anim_my = cum_my[0:-1:k]
    anim_angle = cum_angle[0:-1:k]
    anim_speed = cum_speed[0:-1:k]

    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)

    anim_monkey_info = {
        'anim_indices': anim_indices,
        'anim_t': anim_t,
        'anim_angle': anim_angle,
        'anim_mx': anim_mx,
        'anim_my': anim_my,
        'anim_speed': anim_speed,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
    }

    if 'gaze_world_x' in monkey_information.columns:
        gaze_world_x = monkey_information['gaze_world_x'].to_numpy()[
            cum_pos_index]
        gaze_world_y = monkey_information['gaze_world_y'].to_numpy()[
            cum_pos_index]
        anim_monkey_info['gaze_world_x'] = gaze_world_x[0:-1:k]
        anim_monkey_info['gaze_world_y'] = gaze_world_y[0:-1:k]

    return anim_monkey_info


def rotated_anim_monkey_info(anim_monkey_info, R):
    """Rotate trajectory/state arrays and recenter origin at first point."""
    anim_mx = np.asarray(anim_monkey_info['anim_mx'])
    anim_my = np.asarray(anim_monkey_info['anim_my'])
    anim_mx, anim_my = (R @ np.vstack([anim_mx, anim_my]))

    x0, y0 = anim_mx[0], anim_my[0]
    anim_monkey_info['x0'], anim_monkey_info['y0'] = float(x0), float(y0)
    anim_monkey_info['anim_mx'] = anim_mx - x0
    anim_monkey_info['anim_my'] = anim_my - y0

    anim_monkey_info['xmin'] = np.min(anim_mx) - x0
    anim_monkey_info['xmax'] = np.max(anim_mx) - x0
    anim_monkey_info['ymin'] = np.min(anim_my) - y0
    anim_monkey_info['ymax'] = np.max(anim_my) - y0

    if 'gaze_world_x' in anim_monkey_info:
        gx = np.asarray(anim_monkey_info['gaze_world_x'])
        gy = np.asarray(anim_monkey_info['gaze_world_y'])
        gxy = _rotate_points(np.column_stack([gx, gy]), R, x0, y0)
        anim_monkey_info['gaze_world_x'] = gxy[:, 0]
        anim_monkey_info['gaze_world_y'] = gxy[:, 1]

    return anim_monkey_info, float(x0), float(y0)


# -----------------------------------------------------------------------------
# Firefly-to-time dictionaries
# -----------------------------------------------------------------------------
def match_points_to_flash_on_ff_positions(anim_t,
                                          anim_indices,
                                          duration,
                                          ff_flash_sorted,
                                          ff_life_sorted,
                                          ff_real_position_sorted,
                                          rotation_matrix=None,
                                          x0=0,
                                          y0=0):
    """Map each animation index to positions of currently flashing fireflies (vectorized over FFs)."""
    t = np.asarray(anim_t)
    n_frames = t.size

    # Only consider FFs whose life overlaps the window
    alive_mask = (ff_life_sorted[:, 1] > duration[0]) & (
        ff_life_sorted[:, 0] < duration[1])
    alive_idx = np.where(alive_mask)[0]
    n_alive = alive_idx.size

    # Build visibility matrix: rows = alive FFs, cols = frames
    # For each FF, OR all its [on,off] intervals over the timeline t
    vis_matrix = np.zeros((n_alive, n_frames), dtype=bool)
    for r, ff_i in enumerate(alive_idx):
        intervals = ff_flash_sorted[ff_i]
        if intervals is None or len(intervals) == 0:
            continue
        # Broadcast intervals against all frame times
        on = intervals[:, 0][:, None] <= t
        off = intervals[:, 1][:, None] >= t
        vis_matrix[r] = np.any(on & off, axis=0)

    # Assemble dict per frame
    flash_on_ff_dict = {}
    for col, idx in enumerate(anim_indices):
        ff_here = alive_idx[vis_matrix[:, col]]
        positions = ff_real_position_sorted[ff_here]
        flash_on_ff_dict[idx] = positions

    if rotation_matrix is not None:
        for key, pts in flash_on_ff_dict.items():
            flash_on_ff_dict[key] = _rotate_points(
                np.asarray(pts), rotation_matrix, x0, y0)

    return flash_on_ff_dict


def match_points_to_alive_ff_positions(anim_t,
                                       anim_indices,
                                       ff_caught_T_new,
                                       ff_life_sorted,
                                       ff_real_position_sorted,
                                       rotation_matrix=None,
                                       x0=0,
                                       y0=0):
    """Map each animation index to positions of alive fireflies at that time (vectorized)."""
    t = np.asarray(anim_t)                            # (F,)
    life_start = ff_life_sorted[:, 0][:, None]        # (N,1)
    life_end = ff_life_sorted[:, 1][:, None]        # (N,1)

    # N x F boolean: whether each FF is alive at each frame time
    # strict-then-inclusive, matches original logic
    alive_matrix = (life_start < t) & (life_end >= t)

    alive_ff_dict = {}
    for col, idx in enumerate(anim_indices):
        ff_here = np.flatnonzero(alive_matrix[:, col])
        positions = ff_real_position_sorted[ff_here]
        alive_ff_dict[idx] = positions

    if rotation_matrix is not None:
        for key, pts in alive_ff_dict.items():
            alive_ff_dict[key] = _rotate_points(
                np.asarray(pts), rotation_matrix, x0, y0)

    return alive_ff_dict


def match_points_to_believed_ff_positions(anim_t,
                                          anim_indices,
                                          currentTrial,
                                          num_trials,
                                          ff_believed_position_sorted,
                                          ff_caught_T_new,
                                          rotation_matrix=None,
                                          x0=0,
                                          y0=0):
    """Map each animation index to positions of already-caught fireflies (vectorized via searchsorted)."""
    t = np.asarray(anim_t)

    if ff_caught_T_new is None:
        ff_caught_T_new = np.array([])
    else:
        ff_caught_T_new = np.asarray(ff_caught_T_new)

    if (currentTrial is not None) and (num_trials is not None):
        start = currentTrial - num_trials + 1
        catch_t = ff_caught_T_new[start:currentTrial + 1]
        caught_pos = ff_believed_position_sorted[start:currentTrial + 1]
    else:
        try:
            mask = ff_caught_T_new >= t[0]
            catch_t = ff_caught_T_new[mask]
            caught_pos = ff_believed_position_sorted[mask]
        except Exception:
            catch_t = np.array([])
            caught_pos = np.array([])

    # For each frame time t[i], number of events with catch_t <= t[i]
    # (catch_t must be sorted chronologically; it typically is.)
    counts = np.searchsorted(catch_t, t, side='right')  # (F,)

    believed_ff_dict = {}
    for i, idx in enumerate(anim_indices):
        k = counts[i]
        pts = caught_pos[:k] if k > 0 else np.empty((0, 2), dtype=float)
        believed_ff_dict[idx] = _rotate_points(
            pts, rotation_matrix, x0, y0) if rotation_matrix is not None else pts

    return believed_ff_dict


# -----------------------------------------------------------------------------
# Annotation helpers
# -----------------------------------------------------------------------------

def make_annotation_info(caught_ff_num,
                         max_point_index,
                         n_ff_in_a_row,
                         visible_before_last_one_trials,
                         disappear_latest_trials,
                         ignore_sudden_flash_indices,
                         retry_switch_indices,
                         retry_capture_indices):
    """Collect one-hot style arrays for various annotation categories."""
    zero_arr = np.zeros(caught_ff_num, dtype=int)

    visible_before_last_one_trial_dummy = zero_arr.copy()
    if len(visible_before_last_one_trials) > 0:
        visible_before_last_one_trial_dummy[visible_before_last_one_trials] = 1

    disappear_latest_trial_dummy = zero_arr.copy()
    if len(disappear_latest_trials) > 0:
        disappear_latest_trial_dummy[disappear_latest_trials] = 1

    ignore_sudden_flash_point_dummy = np.zeros(max_point_index + 1, dtype=int)
    if len(ignore_sudden_flash_indices) > 0:
        ignore_sudden_flash_point_dummy[ignore_sudden_flash_indices] = 1

    retry_switch_point_dummy = np.zeros(max_point_index + 1, dtype=int)
    if len(retry_switch_indices) > 0:
        retry_switch_point_dummy[retry_switch_indices] = 1

    retry_capture_point_dummy = np.zeros(max_point_index + 1, dtype=int)
    if len(retry_capture_indices) > 0:
        retry_capture_point_dummy[retry_capture_indices] = 1

    return {
        'n_ff_in_a_row': n_ff_in_a_row,
        'visible_before_last_one_trial_dummy': visible_before_last_one_trial_dummy,
        'disappear_latest_trial_dummy': disappear_latest_trial_dummy,
        'ignore_sudden_flash_point_dummy': ignore_sudden_flash_point_dummy,
        'retry_capture_point_dummy': retry_capture_point_dummy,
        'retry_switch_point_dummy': retry_switch_point_dummy,
    }


# -----------------------------------------------------------------------------
# Plotting primitives used by animation frames
# -----------------------------------------------------------------------------

def plot_trajectory_for_animation(anim_monkey_info, frame, show_speed_through_path_color, ax):
    if show_speed_through_path_color:
        # scale speed to [0, 1] ~ rough normalization; caller can adjust upstream
        c = plt.get_cmap('viridis')(
            np.clip(anim_monkey_info['anim_speed'][:frame + 1] / 200.0, 0, 1))
    else:
        c = 'royalblue'
    ax.scatter(anim_monkey_info['anim_mx'][:frame + 1],
               anim_monkey_info['anim_my'][:frame + 1], s=10, c=c)
    return ax


def plot_visible_ff_reward_boundary_for_animation(visible_ffs, ax, reward_boundary_radius=25):
    visible_ffs = visible_ffs.copy()
    xr, yr, xn, yn = _choose_ff_columns(visible_ffs)

    if 'ff_x_noisy' in visible_ffs.columns:
        if 'pose_unreliable' not in visible_ffs.columns:
            visible_ffs['pose_unreliable'] = False
        for k in range(len(visible_ffs)):
            if not visible_ffs['pose_unreliable'].iloc[k]:
                edge = 'red' if visible_ffs['visible'].iloc[k] else 'gray'
                ax.add_patch(plt.Circle((visible_ffs[xr].iloc[k], visible_ffs[yr].iloc[k]),
                                        reward_boundary_radius, facecolor='yellow', edgecolor=edge, alpha=0.7, zorder=1))
                ax.add_patch(plt.Circle((visible_ffs[xn].iloc[k], visible_ffs[yn].iloc[k]),
                                        reward_boundary_radius, facecolor='gray', edgecolor=edge, alpha=0.5, zorder=1))
            else:
                ax.add_patch(plt.Circle((visible_ffs[xr].iloc[k], visible_ffs[yr].iloc[k]),
                                        reward_boundary_radius, facecolor='black', edgecolor='black', alpha=0.7, zorder=1))
    else:
        for k in range(len(visible_ffs)):
            ax.add_patch(plt.Circle((visible_ffs[xr].iloc[k], visible_ffs[yr].iloc[k]),
                                    reward_boundary_radius, facecolor='yellow', edgecolor='gray', alpha=0.7, zorder=1))
    return ax


def plot_in_memory_ff_reward_boundary_for_animation(in_memory_ffs, ax, reward_boundary_radius=25):
    xr, yr, xn, yn = _choose_ff_columns(in_memory_ffs)

    if 'ff_x_noisy' in in_memory_ffs.columns:
        if 'pose_unreliable' not in in_memory_ffs.columns:
            in_memory_ffs['pose_unreliable'] = False
        for k in range(len(in_memory_ffs)):
            if not in_memory_ffs['pose_unreliable'].iloc[k]:
                edge = 'red' if in_memory_ffs['visible'].iloc[k] else 'gray'
                ax.add_patch(plt.Circle((in_memory_ffs[xr].iloc[k], in_memory_ffs[yr].iloc[k]),
                                        reward_boundary_radius, facecolor='yellow', edgecolor=edge, alpha=0.7, zorder=1))
                ax.add_patch(plt.Circle((in_memory_ffs[xn].iloc[k], in_memory_ffs[yn].iloc[k]),
                                        reward_boundary_radius, facecolor='gray', edgecolor=edge, alpha=0.5, zorder=1))
            else:
                ax.add_patch(plt.Circle((in_memory_ffs[xr].iloc[k], in_memory_ffs[yr].iloc[k]),
                                        reward_boundary_radius, facecolor='black', edgecolor='black', alpha=0.7, zorder=1))
    else:
        for j in range(len(in_memory_ffs)):
            ax.add_patch(plt.Circle((in_memory_ffs[xr].iloc[j], in_memory_ffs[yr].iloc[j]),
                                    reward_boundary_radius, facecolor='purple', edgecolor='orange', alpha=0.3, zorder=1))
    return ax


def show_ff_indices_for_animation(relevant_ff, ax):
    selected_ffs = relevant_ff.copy()
    xr, yr, _, _ = _choose_ff_columns(selected_ffs)
    xs = selected_ffs[xr].to_numpy()
    ys = selected_ffs[yr].to_numpy()
    idxs = selected_ffs['ff_index'].to_numpy()
    for x, y, lab in zip(xs, ys, idxs):
        ax.annotate(str(lab), (x, y), fontsize=12)
    return ax


def find_triangle_to_show_direction(monkey_x, monkey_y, monkey_angle):
    left_end_x = monkey_x + 30 * np.cos(monkey_angle + 2 * pi / 9)
    left_end_y = monkey_y + 30 * np.sin(monkey_angle + 2 * pi / 9)
    right_end_x = monkey_x + 30 * np.cos(monkey_angle - 2 * pi / 9)
    right_end_y = monkey_y + 30 * np.sin(monkey_angle - 2 * pi / 9)
    return left_end_x, left_end_y, right_end_x, right_end_y


def plot_a_triangle_to_show_direction(ax, monkey_x, monkey_y, monkey_angle):
    lx, ly, rx, ry = find_triangle_to_show_direction(
        monkey_x, monkey_y, monkey_angle)
    ax.plot(np.array([monkey_x, lx]), np.array([monkey_y, ly]), linewidth=2)
    ax.plot(np.array([monkey_x, rx]), np.array([monkey_y, ry]), linewidth=2)
    return ax


def plot_triangle_to_show_direction_for_animation(anim_monkey_info, frame, ax):
    return plot_a_triangle_to_show_direction(
        ax,
        anim_monkey_info['anim_mx'][frame],
        anim_monkey_info['anim_my'][frame],
        anim_monkey_info['anim_angle'][frame],
    )


def change_polar_to_xy(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def change_xy_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def plot_time_index_for_animation(anim_monkey_info, frame, ax):
    index = anim_monkey_info['anim_indices'][frame]
    ax.text(0.02, 0.9, str(index), ha='left', va='top', transform=ax.transAxes,
            fontsize=12, color='black', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def plot_circles_around_ff(ff_indices, ff_real_position_rotated, circle_size, edgecolor, ax, lw=2):
    ff_indices = np.asarray(ff_indices).reshape(-1)
    for k in ff_indices:
        ax.add_patch(plt.Circle((ff_real_position_rotated[k, 0], ff_real_position_rotated[k, 1]),
                                circle_size, facecolor='None', edgecolor=edgecolor, alpha=0.8, zorder=1, lw=lw))
    return ax


def plot_anno_ff_for_animation(anno_ff_indices_dict,
                               anno_but_not_obs_ff_indices_dict,
                               point_index,
                               ff_real_position_rotated,
                               markers,
                               marker_labels,
                               ax):
    anno_ff_non_neg_indices = np.array([])
    if anno_ff_indices_dict is not None:
        if point_index in anno_ff_indices_dict:
            anno_indices = anno_ff_indices_dict[point_index]
            anno_ff_non_neg_indices = anno_indices[anno_indices >= 0]
            if len(anno_ff_non_neg_indices) > 0:
                plot_circles_around_ff(anno_ff_non_neg_indices, ff_real_position_rotated,
                                       circle_size=30, edgecolor='red', ax=ax)
            else:
                anno_ff_neg_indices = anno_indices[anno_indices < 0]
                ax.text(0.7, 0.95, str(anno_ff_neg_indices), ha='left', va='top', transform=ax.transAxes,
                        fontsize=18, color='black', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            if (anno_but_not_obs_ff_indices_dict is not None) and (point_index not in anno_but_not_obs_ff_indices_dict):
                ax.text(0.7, 0.95, 'CB?', ha='left', va='top', transform=ax.transAxes,
                        fontsize=18, color='black', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    markers.append(Line2D([0], [0], marker='o', color='r',
                   markerfacecolor='w', lw=0, markeredgewidth=2, markersize=15))
    marker_labels.append('Annotated')

    ax, markers, marker_labels, anno_but_not_obs_ff_indices = plot_circles_around_ff_from_dict(
        anno_but_not_obs_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax,
        edgecolor='green', circle_size=33, legend_name='Annotated but not observed')

    return ax, markers, marker_labels, anno_ff_non_neg_indices, anno_but_not_obs_ff_indices


def plot_pred_ff_for_animation(pred_ff_indices_dict,
                               pred_ff_colors_dict,
                               point_index,
                               ff_real_position_rotated,
                               markers,
                               marker_labels,
                               ax):
    if pred_ff_colors_dict is not None:
        edge = pred_ff_colors_dict.get(point_index, 'blue')
    else:
        edge = 'blue'

    ax, markers, marker_labels, _ = plot_circles_around_ff_from_dict(
        pred_ff_indices_dict, point_index, ff_real_position_rotated, markers, marker_labels, ax,
        edgecolor=edge, circle_size=37, legend_name='Predicted')
    return ax, markers, marker_labels


def plot_circles_around_ff_from_dict(ff_indices_dict,
                                     point_index,
                                     ff_real_position_rotated,
                                     markers,
                                     marker_labels,
                                     ax,
                                     edgecolor='green',
                                     circle_size=39,
                                     legend_name='In Obs',
                                     lw=1):
    in_obs_ff_indices = np.array([])
    if ff_indices_dict is not None:
        if point_index in ff_indices_dict:
            in_obs_ff_indices = ff_indices_dict[point_index]
            plot_circles_around_ff(in_obs_ff_indices, ff_real_position_rotated,
                                   circle_size=circle_size, edgecolor=edgecolor, ax=ax, lw=lw)
        markers.append(Line2D([0], [0], marker='o', color=edgecolor, markerfacecolor='w', lw=0,
                              markeredgewidth=2, markersize=15))
        marker_labels.append(legend_name)
    return ax, markers, marker_labels, in_obs_ff_indices


def plot_show_null_trajectory_for_anno_ff_for_animation(ax,
                                                        anno_ff_non_neg_indices,
                                                        anno_but_not_obs_ff_indices,
                                                        anim_monkey_info,
                                                        frame,
                                                        ff_real_position_rotated,
                                                        arc_color='black',
                                                        line_color='black',
                                                        reaching_boundary_ok=False):
    union_idx = np.union1d(anno_ff_non_neg_indices,
                           anno_but_not_obs_ff_indices).astype(int)
    if len(union_idx) == 0:
        return ax

    monkey_x = anim_monkey_info['anim_mx'][frame]
    monkey_y = anim_monkey_info['anim_my'][frame]
    monkey_xy = np.array([monkey_x, monkey_y])
    monkey_angle = anim_monkey_info['anim_angle'][frame]

    ff_x = ff_real_position_rotated[union_idx, 0]
    ff_y = ff_real_position_rotated[union_idx, 1]

    if reaching_boundary_ok:
        ff_xy = find_best_arc.find_point_on_ff_boundary_with_smallest_angle_to_monkey(
            ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
        ff_x, ff_y = ff_xy[:, 0], ff_xy[:, 1]

    # Null trajectory geometry
    ff_xy, ff_distance, ff_angle, ff_angle_boundary, arc_length, arc_radius = (
        show_null_trajectory.find_arc_length_and_radius(ff_x, ff_y, monkey_x, monkey_y, monkey_angle))

    whether_ff_behind = (np.abs(ff_angle) > math.pi / 2)
    center_x, center_y, arc_starting_angle, arc_ending_angle = (
        show_null_trajectory.find_cartesian_arc_center_and_angle_for_arc_to_center(
            monkey_xy, monkey_angle, ff_distance, ff_angle, arc_radius, ff_xy, np.sign(
                ff_angle),
            whether_ff_behind=whether_ff_behind))

    # Arcs
    arc_ff = np.where(arc_radius > 0)[0]
    if len(arc_ff) > 0:
        arc_xy_rotated = show_null_trajectory.find_arc_xy_rotated(
            center_x[arc_ff], center_y[arc_ff], arc_radius[arc_ff],
            arc_starting_angle[arc_ff], arc_ending_angle[arc_ff], rotation_matrix=None)
        ax.plot(arc_xy_rotated[0], arc_xy_rotated[1],
                linewidth=2.5, color=arc_color, zorder=4)

    # Lines
    line_ff = np.where(arc_radius == 0)[0]
    for ff in line_ff:
        line_xy_rotated = np.vstack(
            (np.array([monkey_x, ff_xy[ff][0]]), np.array([monkey_y, ff_xy[ff][1]])))
        ax.plot(line_xy_rotated[0], line_xy_rotated[1],
                linewidth=2.5, color=line_color, zorder=4)

    return ax
