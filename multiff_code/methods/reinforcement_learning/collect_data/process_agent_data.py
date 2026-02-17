from data_wrangling import general_utils
from pattern_discovery import pattern_by_trials, make_ff_dataframe
from reinforcement_learning.collect_data import collect_agent_data

import os
import shutil
import numpy as np


def find_flash_time_for_one_ff(ff_flash, lifetime):
    """
    Select the flashing durations that overlap with the ff's lifetime

    Parameters
    ----------
    ff_flash: np.array
        containing the intervals that the firefly flashes on
    lifetime: np.array (2,)
        contains when the ff starts to be alive and when it stops being alive (either captured or time is over)

    Returns
    -------
    ff_flash_valid: array
        containing the intervals that the firefly flashes on within the ff's life time.
    """

    indices_of_overlapped_intervals = general_utils.find_intersection(
        ff_flash, lifetime)
    if len(indices_of_overlapped_intervals) > 0:
        ff_flash_valid = ff_flash[indices_of_overlapped_intervals]
        ff_flash_valid[0][0] = max(ff_flash_valid[0][0], lifetime[0])
        ff_flash_valid[-1][1] = min(ff_flash_valid[-1][1], lifetime[1])
    else:
        ff_flash_valid = np.array([[-1, -1]])
    return ff_flash_valid


def make_ff_flash_sorted(env_ff_flash, ff_information, sorted_indices_all, env_end_time):
    """
    Build ff_flash_sorted by using data collected from the agent
    """
    if env_ff_flash is not None:
        ff_flash_sorted = []
        for index, ff in ff_information.iloc[sorted_indices_all].iterrows():
            ff_flash = env_ff_flash[int(ff["index_in_ff_flash"])]
            lifetime = [ff["t_spawn"], ff["t_despawn"]]
            if ff["t_despawn"] == -9999:
                lifetime[1] = env_end_time
            ff_flash_valid = find_flash_time_for_one_ff(ff_flash, lifetime)
            ff_flash_sorted.append(ff_flash_valid)
    else:
        ff_flash_sorted = [
            np.array([[0, env_end_time]])] * len(sorted_indices_all)
    return ff_flash_sorted


def make_env_ff_flash_from_real_data(ff_flash_sorted_of_monkey, alive_ffs, ff_flash_duration):
    """
    Make ff_flash for the env by using real monkey's data
    """
    env_ff_flash = []
    start_time = ff_flash_duration[0]
    for index in alive_ffs:
        ff_flash = ff_flash_sorted_of_monkey[index]
        ff_flash_valid = find_flash_time_for_one_ff(
            ff_flash, ff_flash_duration)
        if ff_flash_valid[-1, -1] != -1:
            ff_flash_valid = ff_flash_valid - start_time
        env_ff_flash.append(np.array(ff_flash_valid))
    return env_ff_flash


def unpack_ff_information_of_agent(ff_information, env_ff_flash, env_end_time):
    """
    Unpack and sort firefly lifetime information for a completed episode.

    Parameters
    ----------
    ff_information : pd.DataFrame
        DataFrame containing firefly lifecycle info (from BaseCollectInformation).
    env_ff_flash : list of np.ndarray
        Flash schedules for each firefly slot (from env.ff_flash).
    env_end_time : float
        Duration of the episode (in seconds).

    Returns
    -------
    ff_caught_T_new : np.ndarray
        Capture times (sorted ascending).
    ff_believed_position_sorted : np.ndarray
        Agent positions (x,y) at capture times, sorted by capture.
    ff_real_position_sorted : np.ndarray
        True firefly positions (x,y) at spawn (sorted).
    ff_life_sorted : np.ndarray
        Nx2 array of [t_spawn, t_despawn] per firefly (sorted).
    ff_flash_sorted : list
        Sorted list of flash arrays matching ff_real_position_sorted order.
    ff_flash_end_sorted : np.ndarray
        Flash end times (last flash end per firefly, or env_end_time if none).
    sorted_indices_all : np.ndarray
        Indices mapping sorted order → original ff_information rows.
    """

    # --- 1. Safety reset ---
    ff_information = ff_information.reset_index(drop=True)

    # --- 2. Extract relevant series ---
    t_capture = ff_information["t_capture"].to_numpy()
    t_despawn = ff_information["t_despawn"].to_numpy()

    # --- 3. Identify captured vs not captured ---
    captured_idx = np.where(t_capture != -9999)[0]
    not_captured_idx = np.where(t_capture == -9999)[0]

    # --- 4. Sort captured by capture time; then append the rest ---
    sorted_captured = captured_idx[np.argsort(t_capture[captured_idx])]
    sorted_indices_all = np.concatenate([sorted_captured, not_captured_idx])

    # --- 5. Sort flash schedules according to same order ---
    ff_flash_sorted = make_ff_flash_sorted(
        env_ff_flash, ff_information, sorted_indices_all, env_end_time
    )

    # --- 6. Capture times and positions at capture ---
    ff_caught_T_new = t_capture[sorted_captured]
    ff_believed_position_sorted = ff_information.loc[
        sorted_captured, ["agent_x_at_capture", "agent_y_at_capture"]
    ].to_numpy()

    # --- 7. Real positions and lifetimes (for all fireflies) ---
    ff_real_position_sorted = ff_information.loc[
        sorted_indices_all, ["ffx", "ffy"]
    ].to_numpy()

    ff_life_sorted = ff_information.loc[
        sorted_indices_all, ["t_spawn", "t_despawn"]
    ].to_numpy()

    # Replace -9999 t_despawn with env_end_time
    missing_despawn_mask = t_despawn[sorted_indices_all] == -9999
    ff_life_sorted[missing_despawn_mask, 1] = env_end_time

    # --- 8. Compute flash end times safely ---
    ff_flash_end_sorted = np.array([
        flash[-1, 1] if (isinstance(flash, np.ndarray) and flash.size > 0)
        else env_end_time
        for flash in ff_flash_sorted
    ], dtype=np.float32)

    return (
        ff_caught_T_new,
        ff_believed_position_sorted,
        ff_real_position_sorted,
        ff_life_sorted,
        ff_flash_sorted,
        ff_flash_end_sorted,
        sorted_indices_all,
    )


def reverse_value_and_position(sorted_indices_all):
    reversed_sorting = np.zeros(len(sorted_indices_all))
    for position in range(len(sorted_indices_all)):
        value = sorted_indices_all[position]
        reversed_sorting[value] = position
    return reversed_sorting


def remove_all_data_derived_from_current_agent_data(processed_data_folder_path):
    """
    Remove all contents inside folders in all_collected_data that are derived
    from the given processed_data_folder_path, but keep the folders themselves.
    """
    if 'processed_data' not in processed_data_folder_path:
        raise ValueError(
            f"'processed_data/' not found in the provided path: {processed_data_folder_path}")

    after_processed_data = processed_data_folder_path.split(
        'processed_data', 1)[1]
    search_root = 'multiff_analysis/RL_models/SB3_stored_models/all_collected_data'

    matching_dirs = []
    for root, dirs, files in os.walk(search_root, topdown=True):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if after_processed_data in full_path:
                matching_dirs.append(full_path)

    matching_dirs.sort(key=len)

    filtered_matches = []
    for path in matching_dirs:
        if not any(path.startswith(parent + os.sep) for parent in filtered_matches):
            filtered_matches.append(path)

    for folder in filtered_matches:
        print(f"Cleaning contents of: {folder}")
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")


def increase_dt_for_monkey_information(time, monkey_x, monkey_y, new_dt, old_dt=0.0166):
    """
    Downsample monkey trajectory and compute kinematics.
    """

    # -------------------------------------------------
    # Downsample (avoid float step in arange)
    # -------------------------------------------------
    ratio = new_dt / old_dt
    step = max(int(round(ratio)), 1)

    indices = np.arange(0, len(time), step)
    indices = indices[indices < len(time)]

    time = time[indices]
    monkey_x = monkey_x[indices]
    monkey_y = monkey_y[indices]

    # -------------------------------------------------
    # Compute velocity
    # -------------------------------------------------
    delta_time = np.diff(time)
    delta_x = np.diff(monkey_x)
    delta_y = np.diff(monkey_y)

    delta_position = np.sqrt(delta_x**2 + delta_y**2)

    monkey_speed = np.divide(
        delta_position,
        delta_time,
        out=np.zeros_like(delta_position),
        where=delta_time != 0
    )

    monkey_speed = np.insert(monkey_speed, 0, monkey_speed[0])

    # -------------------------------------------------
    # Remove unrealistic speed spikes (vectorized)
    # -------------------------------------------------
    high_speed_mask = monkey_speed >= 200

    if np.any(high_speed_mask):
        valid_speeds = monkey_speed[~high_speed_mask]
        replacement = (
            np.median(valid_speeds)
            if len(valid_speeds) > 0
            else 50.0
        )
        monkey_speed[high_speed_mask] = replacement

    # -------------------------------------------------
    # Heading angle
    # -------------------------------------------------
    monkey_angles = np.arctan2(delta_y, delta_x)
    monkey_angles = np.insert(monkey_angles, 0, monkey_angles[0])

    # -------------------------------------------------
    # Angular velocity (shortest path)
    # -------------------------------------------------
    delta_angle = np.diff(monkey_angles)
    delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))

    monkey_dw = np.divide(
        delta_angle,
        delta_time,
        out=np.zeros_like(delta_angle),
        where=delta_time != 0
    )

    monkey_dw = np.insert(monkey_dw, 0, monkey_dw[0])

    return time, monkey_x, monkey_y, monkey_speed, monkey_angles, monkey_dw


def _get_visible_ff_indices(env):
    """
    Return indices in ff_information corresponding to
    currently visible (flashing) fireflies.
    """
    
    # Use slot_ids to get valid slots
    if not hasattr(env, 'slot_ids') or env.slot_ids is None:
        return []
    
    # Get valid slots (those with firefly IDs >= 0)
    valid_slot_mask = env.slot_ids >= 0
    
    if not np.any(valid_slot_mask):
        return []
    
    # Get firefly IDs from valid slots
    valid_ff_ids = env.slot_ids[valid_slot_mask]
    
    # Filter by visibility if available
    if hasattr(env, 'visible') and len(env.visible) > 0:
        # env.visible corresponds to valid slots
        if len(env.visible) == len(valid_ff_ids):
            visible_mask = env.visible > 0.5
            visible_ff_ids = valid_ff_ids[visible_mask]
        else:
            # If lengths don't match, use all valid fireflies
            visible_ff_ids = valid_ff_ids
    else:
        visible_ff_ids = valid_ff_ids
    
    # Find these IDs in ff_information
    indices = [
        np.where(env.ff_information['index_in_ff_flash'] == idx)[0][-1]
        for idx in visible_ff_ids
    ]

    return indices


def _extract_agent_dt_monkey_segment(monkey_df,
                                     imitation_duration,
                                     agent_dt):
    """
    Slice monkey segment and resample to agent dt.
    """

    mask = (
        (monkey_df['time'] >= imitation_duration[0]) &
        (monkey_df['time'] <= imitation_duration[1])
    )

    return increase_dt_for_monkey_information(
        monkey_df.loc[mask, 'time'].to_numpy(),
        monkey_df.loc[mask, 'monkey_x'].to_numpy(),
        monkey_df.loc[mask, 'monkey_y'].to_numpy(),
        agent_dt
    ), mask


def _configure_env_from_monkey_data(env,
                                    info_of_monkey,
                                    alive_ffs,
                                    whole_plot_duration):
    """
    Configure environment fireflies from real monkey data.
    """

    env.flash_on_interval = 0.3
    env.distance2center_cost = 0

    env.ff_flash = make_env_ff_flash_from_real_data(
        info_of_monkey['ff_flash_sorted'],
        alive_ffs,
        whole_plot_duration
    )

    env_ffxy = np.asarray(
        info_of_monkey['ff_real_position_sorted'][alive_ffs],
        dtype=np.float32
    )

    env.ffxy = env.ffxy_noisy = env_ffxy
    env.ffx = env.ffx_noisy = env_ffxy[:, 0]
    env.ffy = env.ffy_noisy = env_ffxy[:, 1]

    obs, _ = env.reset(use_random_ff=False)

    return obs


def _run_agent_rollout(env,
                       obs,
                       monkey_actions,
                       imitation_states,
                       num_imitation_steps_agent,
                       num_total_steps,
                       sac_model):
    """
    Run imitation phase followed by SAC phase.
    """

    monkey_x, monkey_y = [], []
    monkey_speed, monkey_dw = [], []
    monkey_angles, time = [], []
    obs_ff_unique_identifiers = []

    # Disable noise
    original_v_noise_std = env.v_noise_std
    original_w_noise_std = env.w_noise_std
    env.v_noise_std = 0
    env.w_noise_std = 0

    # -----------------------
    # Imitation phase
    # -----------------------
    for step in range(1, num_imitation_steps_agent):

        obs, _, _, _, _ = env.step(
            monkey_actions[step],
            respawn_on_capture=False
        )

        print('env.time: ', env.time)
        #print('obs: ', obs)
        print('visible ff indices: ', _get_visible_ff_indices(env))
        print('env.agentxy: ', env.agentxy)
        print('env.agentheading: ', env.agentheading)
        print('env.v: ', env.v)
        print('env.w: ', env.w)
        print('env.is_stop: ', env.is_stop)

        # Override state
        env.agentheading = np.array([imitation_states['angle'][step]])
        agentx = np.array([imitation_states['x'][step]])
        agenty = np.array([imitation_states['y'][step]])
        env.agentxy = np.array([agentx, agenty])

        monkey_x.append(agentx[0])
        monkey_y.append(agenty[0])
        monkey_speed.append(float(env.v))
        monkey_dw.append(float(env.w))
        monkey_angles.append(env.agentheading[0])
        time.append(env.time)

        obs_ff_unique_identifiers.append(
            _get_visible_ff_indices(env)
        )

    # -----------------------
    # SAC phase
    # -----------------------
    print('Independent phase starts')
    for _ in range(num_imitation_steps_agent, num_total_steps + 10):

        action, _ = sac_model.predict(obs, deterministic=True)
        print('action: ', action)
        obs, _, _, _, _ = env.step(action)

        print('env.time: ', env.time)
        # print('obs: ', obs)
        print('action: ', action)
        print('visible ff indices: ', _get_visible_ff_indices(env))
        print('env.agentxy: ', env.agentxy)
        print('env.agentheading: ', env.agentheading)
        print('env.v: ', env.v)
        print('env.w: ', env.w)
        print('env.is_stop: ', env.is_stop)

        monkey_x.append(float(env.agentxy[0]))
        monkey_y.append(float(env.agentxy[1]))
        monkey_speed.append(float(env.v))
        monkey_dw.append(float(env.w))
        monkey_angles.append(env.agentheading)
        time.append(env.time)

        obs_ff_unique_identifiers.append(
            _get_visible_ff_indices(env)
        )

    # Restore noise
    env.v_noise_std = original_v_noise_std
    env.w_noise_std = original_w_noise_std

    return (
        monkey_x, monkey_y,
        monkey_speed, monkey_dw,
        monkey_angles, time,
        obs_ff_unique_identifiers
    )


def find_corresponding_info_of_agent(env,
                                     sac_model,
                                     info_of_monkey,
                                     start_time,
                                     whole_plot_duration,
                                     imitation_duration,
                                     ):
    """
    Run the agent in a replicated environment around a monkey trial segment.
    """

    agent_dt = env.dt

    # -------------------------------------------------
    # Select alive fireflies
    # -------------------------------------------------
    alive_ffs = np.where(
        (info_of_monkey['ff_life_sorted'][:, 1] >= whole_plot_duration[0]) &
        (info_of_monkey['ff_life_sorted'][:, 0] < whole_plot_duration[1])
    )[0]

    # -------------------------------------------------
    # Extract monkey segment
    # -------------------------------------------------
    (A_cum_t, A_cum_mx, A_cum_my,
     A_cum_speed, A_cum_angle, A_cum_dw), mask = \
        _extract_agent_dt_monkey_segment(
            info_of_monkey['monkey_information'],
            imitation_duration,
            agent_dt
    )

    num_imitation_steps_agent = len(A_cum_t)

    # Rotation matrix
    dx = A_cum_mx[-1] - A_cum_mx[0]
    dy = A_cum_my[-1] - A_cum_my[0]
    theta = np.pi / 2 - np.arctan2(dy, dx)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    # -------------------------------------------------
    # Configure environment
    # -------------------------------------------------
    obs = _configure_env_from_monkey_data(
        env,
        info_of_monkey,
        alive_ffs,
        whole_plot_duration
    )

    # Convert monkey actions
    monkey_actions = np.column_stack((
        A_cum_dw / env.wgain,
        (A_cum_speed / env.vgain - 0.5) * 2
    ))

    # Initialize state
    env.time = A_cum_t[0] - start_time
    env.agentheading = np.array([A_cum_angle[0]])
    env.agentxy = np.array([A_cum_mx[0], A_cum_my[0]])

    # -------------------------------------------------
    # Rollout
    # -------------------------------------------------
    num_total_steps = int(
        np.ceil((whole_plot_duration[1] - whole_plot_duration[0]) / agent_dt)
    )

    rollout_results = _run_agent_rollout(
        env,
        obs,
        monkey_actions,
        {'x': A_cum_mx, 'y': A_cum_my, 'angle': A_cum_angle},
        num_imitation_steps_agent,
        num_total_steps,
        sac_model
    )

    (monkey_x, monkey_y,
     monkey_speed, monkey_dw,
     monkey_angles, time,
     obs_ff_unique_identifiers) = rollout_results

    # -------------------------------------------------
    # Build monkey_information (reuse previous helper)
    # -------------------------------------------------
    monkey_information = collect_agent_data._build_monkey_information(
        np.asarray(time),
        np.asarray(monkey_x),
        np.asarray(monkey_y),
        np.asarray(monkey_speed),
        np.asarray(monkey_dw),
        np.zeros(len(time)),   # no explicit stop flag here
        np.remainder(np.asarray(monkey_angles), 2 * np.pi),
        env.dt
    )

    # -------------------------------------------------
    # Unpack FF info
    # -------------------------------------------------
    (
        ff_caught_T_new,
        ff_believed_position_sorted,
        ff_real_position_sorted,
        ff_life_sorted,
        ff_flash_sorted,
        ff_flash_end_sorted,
        sorted_indices_all
    ) = unpack_ff_information_of_agent(
        env.ff_information,
        env.ff_flash,
        env.time
    )

    reversed_sorting = reverse_value_and_position(sorted_indices_all)

    obs_ff_indices_in_ff_dataframe = [
        reversed_sorting[indices]
        for indices in obs_ff_unique_identifiers
    ]

    # -------------------------------------------------
    # Build ff_dataframe (reuse existing pipeline)
    # -------------------------------------------------
    ff_dataframe = make_ff_dataframe.make_ff_dataframe_func(
        monkey_information,
        ff_caught_T_new,
        ff_flash_sorted,
        ff_real_position_sorted,
        ff_life_sorted,
        max_distance=400,
        player="agent",
        obs_ff_indices_in_ff_dataframe=obs_ff_indices_in_ff_dataframe
    )

    if len(ff_dataframe) > 0:
        ff_dataframe = ff_dataframe[
            ff_dataframe['time'] <= whole_plot_duration[1] -
            whole_plot_duration[0]
        ]

        _, _, cluster_around_target_indices, _ = \
            pattern_by_trials.cluster_around_target_func(
                ff_dataframe,
                len(ff_caught_T_new),
                ff_caught_T_new,
                ff_real_position_sorted
            )
    else:
        cluster_around_target_indices = []

    info_of_agent = {
        "monkey_information": monkey_information,
        "ff_dataframe": ff_dataframe,
        "ff_caught_T_new": ff_caught_T_new,
        "ff_real_position_sorted": ff_real_position_sorted,
        "ff_believed_position_sorted": ff_believed_position_sorted,
        "ff_life_sorted": ff_life_sorted,
        "ff_flash_sorted": ff_flash_sorted,
        "ff_flash_end_sorted": ff_flash_end_sorted,
        "cluster_around_target_indices": cluster_around_target_indices
    }

    return (
        info_of_agent,
        rotation_matrix,
        mask.sum(),
        num_imitation_steps_agent - 1
    )
