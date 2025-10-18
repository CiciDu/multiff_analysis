from data_wrangling import general_utils
from pattern_discovery import pattern_by_trials, make_ff_dataframe
from reinforcement_learning.agents.rnn import env_for_rnn
from reinforcement_learning.agents.feedforward import env_for_sb3

import os
import shutil
import numpy as np
import pandas as pd
import torch
from math import pi


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

    indices_of_overlapped_intervals = general_utils.find_intersection(ff_flash, lifetime)
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
            lifetime = [ff["time_start_to_be_alive"], ff["time_captured"]]
            if ff["time_captured"] == -9999:
                lifetime[1] = env_end_time
            ff_flash_valid = find_flash_time_for_one_ff(ff_flash, lifetime)
            ff_flash_sorted.append(ff_flash_valid)
    else:
        ff_flash_sorted = [np.array([[0, env_end_time]])] * len(sorted_indices_all)
    return ff_flash_sorted


def make_env_ff_flash_from_real_data(ff_flash_sorted_of_monkey, alive_ffs, ff_flash_duration):
    """
    Make ff_flash for the env by using real monkey's data
    """
    env_ff_flash = []
    start_time = ff_flash_duration[0]
    for index in alive_ffs:
        ff_flash = ff_flash_sorted_of_monkey[index]
        ff_flash_valid = find_flash_time_for_one_ff(ff_flash, ff_flash_duration)
        if ff_flash_valid[-1, -1] != -1:
            ff_flash_valid = ff_flash_valid - start_time
        env_ff_flash.append(np.array(ff_flash_valid))
    return env_ff_flash


def increase_dt_for_monkey_information(time, monkey_x, monkey_y, new_dt, old_dt=0.0166):
    """
    Extract data points from monkey's information by increasing the interval between the points
    """
    ratio = new_dt / old_dt
    agent_indices = np.arange(0, len(time) - 1, ratio)
    agent_indices = np.round(agent_indices).astype('int')

    time = time[agent_indices]
    monkey_x = monkey_x[agent_indices]
    monkey_y = monkey_y[agent_indices]

    delta_time = np.diff(time)
    delta_x = np.diff(monkey_x)
    delta_y = np.diff(monkey_y)
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(monkey_speed[0], monkey_speed)

    while np.where(monkey_speed >= 200)[0].size > 0:
        index = np.where(monkey_speed >= 200)[0]
        monkey_speed1 = np.append(monkey_speed[0], monkey_speed)
        monkey_speed[index] = monkey_speed1[index]

    monkey_angles = np.arctan2(delta_y, delta_x)
    monkey_angles = np.append(monkey_angles[0], monkey_angles)

    delta_angle = np.remainder(np.diff(monkey_angles), 2 * pi)
    monkey_dw = np.divide(delta_angle, delta_time)
    monkey_dw = np.append(monkey_dw[0], monkey_dw)

    return time, monkey_x, monkey_y, monkey_speed, monkey_angles, monkey_dw


def unpack_ff_information_of_agent(ff_information, env_ff_flash, env_end_time):
    ff_time_captured_all = ff_information.loc[:, "time_captured"]
    captured_ff_indices = np.where(ff_time_captured_all != -9999)[0]
    not_captured_ff_indices = np.where(ff_time_captured_all == -9999)[0]

    sorted_indices_captured = captured_ff_indices[np.argsort(ff_time_captured_all[captured_ff_indices])]
    sorted_indices_all = np.concatenate([sorted_indices_captured, not_captured_ff_indices])

    ff_flash_sorted = make_ff_flash_sorted(env_ff_flash, ff_information, sorted_indices_all, env_end_time)

    ff_caught_T_new = np.array(ff_time_captured_all[sorted_indices_captured])
    ff_believed_position_sorted = np.array(ff_information.iloc[sorted_indices_captured, 5:7])
    ff_real_position_sorted = np.array(ff_information.iloc[sorted_indices_all, 1:3])
    ff_life_sorted = np.array(ff_information.iloc[sorted_indices_all, 3:5])
    ff_life_sorted[:, 1][np.where(ff_life_sorted[:, 1] == -9999)[0]] = env_end_time
    ff_flash_end_sorted = [flash[-1, 1] if len(flash) > 0 else env_end_time for flash in ff_flash_sorted]
    ff_flash_end_sorted = np.array(ff_flash_end_sorted)

    return ff_caught_T_new, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all


def reverse_value_and_position(sorted_indices_all):
    reversed_sorting = np.zeros(len(sorted_indices_all))
    for position in range(len(sorted_indices_all)):
        value = sorted_indices_all[position]
        reversed_sorting[value] = position
    return reversed_sorting


def find_corresponding_info_of_agent(info_of_monkey, currentTrial, num_trials, sac_model, agent_dt, LSTM=False, env_kwargs=None):
    """
    Run the agent in a replicated environment around a monkey trial segment, returning agent info used in plots.
    """
    # Set a duration that the plot will encompass
    start_time = min(info_of_monkey['ff_caught_T_new'][currentTrial - 3], info_of_monkey['ff_caught_T_new'][currentTrial] - num_trials)
    plot_whole_duration = [start_time, info_of_monkey['ff_caught_T_new'][currentTrial]]
    monkey_acting_duration = [start_time, info_of_monkey['ff_caught_T_new'][currentTrial] - 1.5]

    alive_ffs = np.array([index for index, life in enumerate(info_of_monkey['ff_life_sorted']) if (life[1] >= plot_whole_duration[0]) and (life[0] < plot_whole_duration[1])])
    M_cum_indices = np.where((info_of_monkey['monkey_information']['time'] >= monkey_acting_duration[0]) & (info_of_monkey['monkey_information']['time'] <= monkey_acting_duration[1]))[0]
    M_cum_t = np.array(info_of_monkey['monkey_information']['time'][M_cum_indices])
    M_cum_mx, M_cum_my = np.array(info_of_monkey['monkey_information']['monkey_x'][M_cum_indices]), np.array(info_of_monkey['monkey_information']['monkey_y'][M_cum_indices])
    A_cum_t, A_cum_mx, A_cum_my, A_cum_speed, A_cum_angle, A_cum_dw = increase_dt_for_monkey_information(M_cum_t, M_cum_mx, M_cum_my, agent_dt)
    num_imitation_steps_agent = len(A_cum_t)
    num_imitation_steps_monkey = len(M_cum_t)

    theta = pi / 2 - np.arctan2(M_cum_my[-1] - M_cum_my[0], M_cum_mx[-1] - M_cum_mx[0])
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))

    env_kwargs['dt'] = agent_dt
    env_kwargs['num_alive_ff'] = len(alive_ffs)

    if LSTM:
        env = env_for_rnn.CollectInformationLSTM(**env_kwargs)
        model_device = next(sac_model.policy_net.parameters()).device
        hidden_out = (
            torch.zeros([1, 1, sac_model.hidden_dim], dtype=torch.float32, device=model_device),
            torch.zeros([1, 1, sac_model.hidden_dim], dtype=torch.float32, device=model_device)
        )
    else:
        env = env_for_sb3.CollectInformation(**env_kwargs)
    env.flash_on_interval = 0.3

    env.distance2center_cost = 0
    env.ff_flash = make_env_ff_flash_from_real_data(info_of_monkey['ff_flash_sorted'], alive_ffs, plot_whole_duration)
    env_ffxy = np.array(info_of_monkey['ff_real_position_sorted'][alive_ffs], dtype=np.float32)
    env.ffxy, env.ffxy_noisy = env_ffxy, env_ffxy
    env.ffx, env.ffx_noisy = env.ffxy[:, 0], env.ffxy[:, 0]
    env.ffy, env.ffy_noisy = env.ffxy[:, 1], env.ffxy[:, 1]
    obs, _ = env.reset(use_random_ff=False)

    monkey_actions = np.stack((A_cum_dw / env.wgain, (A_cum_speed / env.vgain - 0.5) * 2), axis=1)

    monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, time = [], [], [], [], [], []
    obs_ff_unique_identifiers = []

    original_v_noise_std = env.v_noise_std
    original_w_noise_std = env.w_noise_std
    env.v_noise_std = 0
    env.w_noise_std = 0
    env.time = M_cum_t[0] - start_time
    env.agentheading = np.array([A_cum_angle[0]])
    env.agentx = np.array([A_cum_mx[0]])
    env.agenty = np.array([A_cum_my[0]])

    for step in range(1, num_imitation_steps_agent):
        prev_ff_information = env.ff_information.copy()
        if LSTM:
            if step > 0:
                hidden_in = hidden_out
                action, hidden_out = sac_model.policy_net.get_action(state, last_action, hidden_in, deterministic=True)
            last_action = monkey_actions[step]
            next_state, reward, done, _, _ = env.step(monkey_actions[step])
            state = next_state
        else:
            obs, reward, done, _, info = env.step(monkey_actions[step])
        env.agentheading = np.array([A_cum_angle[step]])
        env.agentx = np.array([A_cum_mx[step]])
        env.agenty = np.array([A_cum_my[step]])

        monkey_x.append(env.agentx[0])
        monkey_y.append(env.agenty.item())
        monkey_speed.append(float(env.v))
        monkey_dw.append(float(env.w))
        monkey_angles.append(env.agentheading.item())
        time.append(env.time)

        indexes_in_ff_information = []
        for index in env.topk_indices:
            last_corresponding_ff_index = np.where(prev_ff_information.loc[:, "index_in_ff_flash"] == index)[0][-1]
            indexes_in_ff_information.append(last_corresponding_ff_index)
        obs_ff_unique_identifiers.append(indexes_in_ff_information)

    env.v_noise_std = original_v_noise_std
    env.w_noise_std = original_w_noise_std
    num_total_steps = int(np.ceil((plot_whole_duration[1] - plot_whole_duration[0]) / agent_dt))
    for step in range(num_imitation_steps_agent, num_total_steps + 10):
        if LSTM:
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(state, last_action, hidden_in, deterministic=True)
            last_action = action
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        else:
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        monkey_x.append(env.agentx[0])
        monkey_y.append(env.agenty[0])
        monkey_speed.append(float(env.v))
        monkey_dw.append(float(env.w))
        monkey_angles.append(env.agentheading[0])
        time.append(env.time)

        indexes_in_ff_information = []
        for index in env.topk_indices:
            last_corresponding_ff_index = np.where(prev_ff_information.loc[:, "index_in_ff_flash"] == index)[0][-1]
            indexes_in_ff_information.append(last_corresponding_ff_index)
        obs_ff_unique_identifiers.append(indexes_in_ff_information)

    monkey_information = {
        'time': np.array(time),
        'monkey_x': np.array(monkey_x),
        'monkey_y': np.array(monkey_y),
        'speed': np.array(monkey_speed),
        'ang_speed': np.array(monkey_dw),
        'monkey_angle': np.remainder(np.array(monkey_angles), 2 * pi),
    }
    monkey_information['point_index'] = range(len(monkey_information['time']))

    ff_caught_T_new, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all = unpack_ff_information_of_agent(env.ff_information, env.ff_flash, env.time)
    caught_ff_num = len(ff_caught_T_new)

    reversed_sorting = reverse_value_and_position(sorted_indices_all)
    obs_ff_indices_in_ff_dataframe = [reversed_sorting[indices] for indices in obs_ff_unique_identifiers]

    ff_dataframe_args = (pd.DataFrame(monkey_information), ff_caught_T_new, ff_flash_sorted, ff_real_position_sorted, ff_life_sorted)
    ff_dataframe_kargs = {"max_distance": 400}
    ff_dataframe = make_ff_dataframe.make_ff_dataframe_func(*ff_dataframe_args, **ff_dataframe_kargs, player="agent", obs_ff_indices_in_ff_dataframe=obs_ff_indices_in_ff_dataframe)
    if len(ff_dataframe) > 0:
        ff_dataframe = ff_dataframe[ff_dataframe['time'] <= plot_whole_duration[1] - plot_whole_duration[0]]
        _, _, cluster_around_target_indices, _ = pattern_by_trials.cluster_around_target_func(ff_dataframe, caught_ff_num, ff_caught_T_new, ff_real_position_sorted)
    else:
        cluster_around_target_indices = []

    num_imitation_steps_agent = num_imitation_steps_agent - 1
    info_of_agent = {
        "monkey_information": pd.DataFrame(monkey_information),
        "ff_dataframe": ff_dataframe,
        "ff_caught_T_new": ff_caught_T_new,
        "ff_real_position_sorted": ff_real_position_sorted,
        "ff_believed_position_sorted": ff_believed_position_sorted,
        "ff_life_sorted": ff_life_sorted,
        "ff_flash_sorted": ff_flash_sorted,
        "ff_flash_end_sorted": ff_flash_end_sorted,
        "cluster_around_target_indices": cluster_around_target_indices
    }

    return info_of_agent, plot_whole_duration, rotation_matrix, len(M_cum_t), num_imitation_steps_agent


def remove_all_data_derived_from_current_agent_data(processed_data_folder_path):
    """
    Remove all contents inside folders in all_collected_data that are derived
    from the given processed_data_folder_path, but keep the folders themselves.
    """
    if 'processed_data/' not in processed_data_folder_path:
        raise ValueError("'processed_data/' not found in the provided path.")

    after_processed_data = processed_data_folder_path.split('processed_data/', 1)[1]
    search_root = 'RL_models/SB3_stored_models/all_collected_data'

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


