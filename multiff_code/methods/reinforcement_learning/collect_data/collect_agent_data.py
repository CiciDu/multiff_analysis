from data_wrangling import process_monkey_information
from pattern_discovery import make_ff_dataframe
from reinforcement_learning.agents.rnn import env_for_rnn
from reinforcement_learning.agents.feedforward import env_for_sb3
from reinforcement_learning.collect_data.process_agent_data import (
    find_flash_time_for_one_ff,
    make_ff_flash_sorted,
    make_env_ff_flash_from_real_data,
    increase_dt_for_monkey_information,
    unpack_ff_information_of_agent,
    reverse_value_and_position,
    find_corresponding_info_of_agent,
)
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials

import os
import shutil
import numpy as np
import torch
import matplotlib
import pandas as pd
import math
from matplotlib import rc
from math import pi
import logging
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


device = "cpu"  # Default to CPU since we're removing torch dependencies


# ---------------------------------------------------------------------
# Helper 1: Initialization
# ---------------------------------------------------------------------
def _initialize_agent_state(env, sac_model, LSTM=False, hidden_dim=128, first_obs=None, seed=42):
    """Initialize agent state, environment, and hidden state if using LSTM."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if LSTM:
        if first_obs is None:
            state, _ = env.reset()
        else:
            state = first_obs
        last_action = env.action_space.sample()
        model_device = next(sac_model.policy_net.parameters()).device
        hidden_out = (
            torch.zeros([1, 1, hidden_dim], dtype=torch.float32, device=model_device),
            torch.zeros([1, 1, hidden_dim], dtype=torch.float32, device=model_device)
        )
        return state, last_action, hidden_out
    else:
        if first_obs is None:
            obs, _ = env.reset()
        else:
            obs = first_obs
        return obs, None, None


# ---------------------------------------------------------------------
# Helper 2: Core data collection loop
# ---------------------------------------------------------------------
def _collect_monkey_and_ff_data(env, sac_model, n_steps, LSTM, hidden_dim, deterministic,
                                state_or_obs, last_action, hidden_out):
    """Core loop for collecting agent, monkey, and firefly data."""
    monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, time = ([] for _ in range(6))
    indexes_in_ff_flash, corresponding_time, ff_x_noisy, ff_y_noisy = ([] for _ in range(4))
    pose_unreliable, visible, time_since_last_vis_list, all_steps = ([] for _ in range(4))
    

    for step in range(n_steps):
        if step % 1000 == 0 and step != 0:
            logging.info(f"Step: {step} / {n_steps}")

        if LSTM:
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(
                state_or_obs, last_action, hidden_in, deterministic=deterministic
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            last_action = action
            state_or_obs = next_obs
        else:
            action, _ = sac_model.predict(state_or_obs, deterministic=deterministic)
            state_or_obs, reward, terminated, truncated, info = env.step(action)

        # Collect monkey data
        monkey_x.append(env.agentx[0])
        monkey_y.append(env.agenty[0])
        monkey_speed.append(float(env.v))
        monkey_dw.append(float(env.w))
        monkey_angles.append(env.agentheading[0])
        time.append(env.time)

        # Copy (not mutate) firefly indices
        topk_indices = env.topk_indices.tolist()
        indexes_in_ff_flash.extend(topk_indices)
        corresponding_time.extend([env.time] * len(topk_indices))
        all_steps.extend([step] * len(topk_indices))

        if len(topk_indices) > 0:
            t_last_seen = env.ff_t_since_last_seen[topk_indices]
            time_since_last_vis_list.extend(t_last_seen.tolist())

        if len(env.ffxy_topk_noisy) > 0:
            if env.ffxy_topk_noisy.shape[0] != len(topk_indices):
                raise ValueError(
                    "Number of fireflies in observation does not match the environment."
                )
            ff_x_noisy.extend(env.ffxy_topk_noisy[:, 0].tolist())
            ff_y_noisy.extend(env.ffxy_topk_noisy[:, 1].tolist())
            pose_unreliable.extend(env.pose_unreliable.tolist())
            visible.extend(env.visible.tolist())

        if (LSTM and (terminated or truncated)) or (not LSTM and (terminated or truncated)):
            logging.info("Episode ended (terminated or truncated) by environment.")
            break

    return (
        monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, time,
        indexes_in_ff_flash, corresponding_time, ff_x_noisy, ff_y_noisy,
        pose_unreliable, visible, time_since_last_vis_list, all_steps
    )


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def collect_agent_data_func(env, sac_model, n_steps=15000, LSTM=False,
                            hidden_dim=128, deterministic=True, first_obs=None, seed=42):
    """
    Extract data points from monkey's behavior by increasing the interval between the points.
    """

    # Initialize
    state_or_obs, last_action, hidden_out = _initialize_agent_state(
        env, sac_model, LSTM, hidden_dim, first_obs, seed
    )

    # Collect environment data
    results = _collect_monkey_and_ff_data(
        env, sac_model, n_steps, LSTM, hidden_dim, deterministic,
        state_or_obs, last_action, hidden_out
    )

    (monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, time,
     indexes_in_ff_flash, corresponding_time, ff_x_noisy, ff_y_noisy,
     pose_unreliable, visible, time_since_last_vis_list, all_steps) = results

    # -----------------------------------------------------------------
    # ↓ Your downstream firefly + monkey data processing section ↓
    # -----------------------------------------------------------------
    ff_in_obs_df = pd.DataFrame({
        'index_in_ff_flash': indexes_in_ff_flash,
        'time': corresponding_time,
        'point_index': all_steps,
        'ff_x_noisy': ff_x_noisy,
        'ff_y_noisy': ff_y_noisy,
        'pose_unreliable': pose_unreliable,
        'visible': visible,
        'time_since_last_vis': time_since_last_vis_list
    })

    ff_information_temp = env.ff_information.copy()
    ff_information_temp['index_in_ff_information'] = range(len(ff_information_temp))
    ff_information_temp.loc[ff_information_temp['time_captured'] < 0, 'time_captured'] = env.time + 10

    ff_in_obs_df = ff_in_obs_df.merge(
        ff_information_temp, on='index_in_ff_flash', how='left'
    )
    ff_in_obs_df = ff_in_obs_df[ff_in_obs_df['time'].between(
        ff_in_obs_df['time_start_to_be_alive'], ff_in_obs_df['time_captured'], inclusive='left'
    )].copy()

    if ff_in_obs_df.groupby('point_index').count().max().max() > env.num_obs_ff:
        raise ValueError(
            "The number of fireflies in the observation exceeds the number in the environment."
        )

    # Collect all monkey data
    monkey_information = pack_monkey_information(
        time, monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, env.dt
    )
    monkey_information['point_index'] = range(len(monkey_information))
    monkey_information['monkey_speeddummy'] = (
        (monkey_information['speed'] > env.linear_terminal_vel * 200) |
        (np.abs(monkey_information['ang_speed']) > env.angular_terminal_vel * pi / 2)
    ).astype(int)

    process_monkey_information.add_more_columns_to_monkey_information(monkey_information)

    # Get information about fireflies
    ff_caught_T_new, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, \
        ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all = unpack_ff_information_of_agent(
            env.ff_information, env.ff_flash, env.time
    )

    caught_ff_num = len(ff_caught_T_new)
    total_ff_num = len(ff_life_sorted)

    reversed_sorting = reverse_value_and_position(sorted_indices_all)
    ff_in_obs_df['index_in_ff_dataframe'] = reversed_sorting[ff_in_obs_df['index_in_ff_information'].values]
    ff_in_obs_df = ff_in_obs_df.astype({
        'index_in_ff_information': 'int', 'index_in_ff_dataframe': 'int', 'point_index': 'int'
    })
    num_decimals_of_dt = find_decimals(env.dt)
    ff_in_obs_df['time_since_last_vis'] = np.round(ff_in_obs_df['time_since_last_vis'], num_decimals_of_dt)
    ff_in_obs_df.reset_index(drop=True, inplace=True)

    ff_in_obs_df = ff_in_obs_df[
        ['index_in_ff_dataframe', 'index_in_ff_information', 'index_in_ff_flash', 'point_index',
         'ff_x_noisy', 'ff_y_noisy', 'time_since_last_vis', 'visible', 'pose_unreliable']
    ].copy()
    ff_in_obs_df.rename(columns={'index_in_ff_dataframe': 'ff_index'}, inplace=True)

    obs_ff_indices_in_ff_dataframe = pd.DataFrame(
        ff_in_obs_df.groupby('point_index')['ff_index'].apply(list)
    )
    obs_ff_indices_in_ff_dataframe = obs_ff_indices_in_ff_dataframe.merge(
        pd.DataFrame(pd.Series(range(n_steps), name='point_index')),
        on='point_index', how='right'
    )

    obs_ff_indices_in_ff_dataframe = obs_ff_indices_in_ff_dataframe['ff_index'].tolist()
    obs_ff_indices_in_ff_dataframe = [
        np.array(x) if isinstance(x, list) else np.array([]) for x in obs_ff_indices_in_ff_dataframe
    ]

    # Capture rate
    if monkey_information['time'].max() > 0:
        ff_capture_rate = len(set(ff_caught_T_new)) / monkey_information['time'].max()
        logging.info(f"Firefly capture rate: {ff_capture_rate:.4f}")
    else:
        logging.warning("Monkey time max is 0; capture rate undefined.")

    # -----------------------------------------------------------------
    # Return final results (same as original)
    # -----------------------------------------------------------------
    return (monkey_information, ff_flash_sorted, ff_caught_T_new, ff_believed_position_sorted,
            ff_real_position_sorted, ff_life_sorted, ff_flash_end_sorted, caught_ff_num,
            total_ff_num, obs_ff_indices_in_ff_dataframe, sorted_indices_all, ff_in_obs_df)



def find_decimals(x):
    if x == 0:
        return 0
    else:
        return int(abs(math.log10(abs(x))))


def pack_monkey_information(time, monkey_x, monkey_y, monkey_speed, monkey_dw, monkey_angles, dt):
    """
    Organize the information of the monkey/agent into a dictionary


    Parameters
    ----------
    time: list
        containing a series of time points
    monkey_x: list
        containing a series of x-positions of the monkey/agent
    monkey_y: list
        containing a series of y-positions of the monkey/agent  
    monkey_speed: list
        containing a series of linear speeds of the monkey/agent  
    monkey_angles: list    
        containing a series of angles of the monkey/agent  
    dt: num
        the time interval

    Returns
    -------
    monkey_information: df
        containing the information such as the speed, angle, and location of the monkey at various points of time

    """
    time = np.array(time)
    monkey_x = np.array(monkey_x)
    monkey_y = np.array(monkey_y)
    monkey_speed = np.array(monkey_speed)
    monkey_dw = np.array(monkey_dw)
    monkey_angles = np.array(monkey_angles)
    monkey_angles = np.remainder(monkey_angles, 2*pi)

    monkey_information = {
        'time': time,
        'monkey_x': monkey_x,
        'monkey_y': monkey_y,
        'speed': monkey_speed,
        'ang_speed': monkey_dw,
        'monkey_angle': monkey_angles,
    }

    # determine whether the speed of the monkey is above a threshold at each time point
    monkey_speeddummy = ((monkey_speed > 200 * 0.01 * dt) |
                         (monkey_dw > pi/2 * 0.01 * dt)).astype(int)
    monkey_information['monkey_speeddummy'] = monkey_speeddummy

    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    crossing_boundary = np.append(0, (delta_position > 100).astype('int'))
    monkey_information['crossing_boundary'] = crossing_boundary

    monkey_information = pd.DataFrame(monkey_information)

    return monkey_information

