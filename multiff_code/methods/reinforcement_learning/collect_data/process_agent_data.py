from data_wrangling import general_utils
from pattern_discovery import pattern_by_trials, make_ff_dataframe
from reinforcement_learning.agents.rnn import rnn_env
from reinforcement_learning.agents.feedforward import sb3_env
from reinforcement_learning.agents.attention.env_attn_multiff import (
    EnvForAttentionSAC as EnvForAttention,
    get_action_limits as attn_get_action_limits,
)

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
        raise ValueError(f"'processed_data/' not found in the provided path: {processed_data_folder_path}")

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
