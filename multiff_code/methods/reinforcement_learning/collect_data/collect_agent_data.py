from data_wrangling import process_monkey_information
from pattern_discovery import make_ff_dataframe
from reinforcement_learning.agents.rnn import env_for_rnn
from reinforcement_learning.agents.feedforward import env_for_sb3
from reinforcement_learning.agents.attention.env_attn_multiff import (
    get_action_limits as attn_get_action_limits,
)
from reinforcement_learning.collect_data.process_agent_data import (
    find_flash_time_for_one_ff,
    make_ff_flash_sorted,
    make_env_ff_flash_from_real_data,
    unpack_ff_information_of_agent,
    reverse_value_and_position,
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


device = "cpu"  # Default to CPU


# ---------------------------------------------------------------------
# Helper 1: Initialization
# ---------------------------------------------------------------------
def _initialize_agent_state(env, sac_model, hidden_dim=128, first_obs=None, seed=42, agent_type=None):
    """Initialize agent state, environment, and hidden state for different agent types."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    derived_agent_type = str(agent_type).lower(
    ) if agent_type is not None else "sb3"

    if derived_agent_type in ("lstm", "gru"):
        if first_obs is None:
            state, _ = env.reset()
        else:
            state = first_obs
        last_action = env.action_space.sample()
        model_device = next(sac_model.policy_net.parameters()).device
        # Align hidden size with the model's configured hidden dimension when available
        model_hidden_dim = getattr(sac_model, 'hidden_dim', hidden_dim)
        if derived_agent_type == "lstm":
            hidden_out = (
                torch.zeros([1, 1, model_hidden_dim],
                            dtype=torch.float32, device=model_device),
                torch.zeros([1, 1, model_hidden_dim],
                            dtype=torch.float32, device=model_device)
            )
        else:  # GRU
            hidden_out = torch.zeros(
                [1, 1, model_hidden_dim], dtype=torch.float32, device=model_device)
        return state, last_action, hidden_out
    elif derived_agent_type in ("attn", "attention", "attn_ff", "attn_rnn", "attention_ff", "attention_rnn"):
        if first_obs is None:
            obs, _ = env.reset()
        else:
            obs = first_obs
        return obs, None, None
    else:  # SB3 feedforward
        if first_obs is None:
            obs, _ = env.reset()
        else:
            obs = first_obs
        return obs, None, None


# ---------------------------------------------------------------------
# Helper 2: Core data collection loop
# ---------------------------------------------------------------------
def _collect_monkey_and_ff_data(env, sac_model, n_steps, hidden_dim, deterministic,
                                state_or_obs, last_action, hidden_out, agent_type=None):
    """Core loop for collecting agent, monkey, and firefly data across agent types."""
    monkey_x, monkey_y, speed, ang_speed, monkey_angle, is_stop, time = (
        [] for _ in range(7))
    indexes_in_ff_flash, corresponding_time, ff_x_noisy, ff_y_noisy = (
        [] for _ in range(4))
    pose_unreliable, visible, time_since_last_vis_list, all_steps = (
        [] for _ in range(4))

    derived_agent_type = str(agent_type).lower(
    ) if agent_type is not None else "sb3"

    is_rnn = derived_agent_type in ("lstm", "gru")
    is_attn_ff = derived_agent_type in (
        "attn", "attention", "attn_ff", "attention_ff")
    is_attn_rnn = derived_agent_type in ("attn_rnn", "attention_rnn")
    attn_limits = None
    if is_attn_ff or is_attn_rnn:
        try:
            attn_limits = attn_get_action_limits(env)
        except Exception:
            attn_limits = [(-1.0, 1.0), (-1.0, 1.0)]

    for step in range(n_steps):
        if step % 1000 == 0 and step != 0:
            logging.info(f"Step: {step} / {n_steps}")

        if is_rnn:
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(
                state_or_obs, last_action, hidden_in, deterministic=deterministic
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            last_action = action
            state_or_obs = next_obs
        elif is_attn_ff or is_attn_rnn:
            if hasattr(env, "obs_to_attn_tensors") and hasattr(sac_model, "actor"):
                if is_attn_rnn:
                    model_device = next(sac_model.actor.parameters()).device
                    sf, sm, ss = env.obs_to_attn_tensors(state_or_obs, device=model_device)
                    with torch.no_grad():
                        mu_seq, std_seq, _, hidden_out = sac_model.actor(
                            sf.unsqueeze(1), sm.unsqueeze(1), ss.unsqueeze(1), hx=hidden_out
                        )
                        mu = mu_seq[:, -1]
                        std = std_seq[:, -1]
                        if deterministic:
                            a = torch.tanh(mu)
                            scaled = []
                            for j in range(a.size(-1)):
                                lo, hi = attn_limits[j]
                                mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
                                scaled.append(mid + half * a[:, j:j+1])
                            act_tensor = torch.cat(scaled, dim=-1)
                        else:
                            # Reuse sampling util from policy if available, else approximate
                            z = torch.randn_like(std)
                            act_tensor = torch.tanh(mu + std * z)
                            scaled = []
                            for j in range(act_tensor.size(-1)):
                                lo, hi = attn_limits[j]
                                mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
                                scaled.append(
                                    mid + half * act_tensor[:, j:j+1])
                            act_tensor = torch.cat(scaled, dim=-1)
                    action = act_tensor.squeeze(
                        0).detach().cpu().numpy().astype(np.float32)
                else:
                    sf, sm, ss = env.obs_to_attn_tensors(
                        state_or_obs, device=device)
                    with torch.no_grad():
                        mu, std, _, _ = sac_model.actor(sf, sm, ss)
                        if deterministic:
                            a = torch.tanh(mu)
                            scaled = []
                            for j in range(a.size(-1)):
                                lo, hi = attn_limits[j]
                                mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
                                scaled.append(mid + half * a[0, j:j+1])
                            act_tensor = torch.cat(scaled, dim=-1)
                        else:
                            z = torch.randn_like(std)
                            act_tensor = torch.tanh(mu + std * z)
                            scaled = []
                            for j in range(act_tensor.size(-1)):
                                lo, hi = attn_limits[j]
                                mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
                                scaled.append(
                                    mid + half * act_tensor[0, j:j+1])
                            act_tensor = torch.cat(scaled, dim=-1)
                    action = act_tensor.squeeze(
                        0).detach().cpu().numpy().astype(np.float32)
                state_or_obs, reward, terminated, truncated, info = env.step(
                    action)
            else:
                action, _ = sac_model.predict(
                    state_or_obs, deterministic=deterministic)
                state_or_obs, reward, terminated, truncated, info = env.step(
                    action)
        else:
            action, _ = sac_model.predict(
                state_or_obs, deterministic=deterministic)
            state_or_obs, reward, terminated, truncated, info = env.step(
                action)

        # Collect monkey data
        monkey_x.append(env.agentxy[0] + env.arena_center_global[0])
        monkey_y.append(env.agentxy[1] + env.arena_center_global[1])
        speed.append(float(env.v))
        ang_speed.append(float(env.w))
        monkey_angle.append(env.agentheading[0])
        is_stop.append(env.is_stop)
        time.append(env.time)

        # Copy (not mutate) firefly indices
        topk_indices = env.topk_indices.tolist()
        indexes_in_ff_flash.extend(topk_indices)
        corresponding_time.extend([env.time] * len(topk_indices))
        all_steps.extend([step] * len(topk_indices))
        # print('topk_indices in collect_agent_data: ', topk_indices)

        if len(topk_indices) > 0:
            t_last_seen = env.ff_t_since_last_seen[topk_indices]
            time_since_last_vis_list.extend(t_last_seen.tolist())

        if len(env.ffxy_slot_noisy) > 0:
            if env.ffxy_slot_noisy.shape[0] != len(topk_indices):
                raise ValueError(
                    "Number of fireflies in observation does not match the environment."
                )
            ff_x_noisy.extend((env.ffxy_slot_noisy[:, 0] + env.arena_center_global[0]).tolist())
            ff_y_noisy.extend((env.ffxy_slot_noisy[:, 1] + env.arena_center_global[1]).tolist())
            pose_unreliable.extend(env.pose_unreliable.tolist())
            visible.extend(env.visible.tolist())

        if terminated or truncated:
            logging.info(
                "Episode ended (terminated or truncated) by environment.")
            break

    return (
        monkey_x, monkey_y, speed, ang_speed, monkey_angle, is_stop, time,
        indexes_in_ff_flash, corresponding_time, ff_x_noisy, ff_y_noisy,
        pose_unreliable, visible, time_since_last_vis_list, all_steps
    )


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def collect_agent_data_func(env, sac_model, n_steps=15000,
                            hidden_dim=128, deterministic=True, first_obs=None, seed=42, agent_type=None):
    """
    Extract data points from monkey's behavior by increasing the interval between the points.
    """

    # Initialize
    state_or_obs, last_action, hidden_out = _initialize_agent_state(
        env, sac_model, hidden_dim, first_obs, seed, agent_type
    )

    # Collect environment data
    results = _collect_monkey_and_ff_data(
        env, sac_model, n_steps, hidden_dim, deterministic,
        state_or_obs, last_action, hidden_out, agent_type
    )

    (monkey_x, monkey_y, speed, ang_speed, monkey_angle, is_stop, time,
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
    ff_information_temp['index_in_ff_information'] = range(
        len(ff_information_temp))
    ff_information_temp.loc[ff_information_temp['t_capture']
                            < 0, 't_capture'] = env.time + 10
    ff_information_temp.loc[ff_information_temp['t_despawn']
                            < 0, 't_despawn'] = env.time + 10
    
    ff_in_obs_df = ff_in_obs_df.merge(
        ff_information_temp, on='index_in_ff_flash', how='left'
    )
    ff_in_obs_df = ff_in_obs_df[ff_in_obs_df['time'].between(
        ff_in_obs_df['t_spawn'], ff_in_obs_df['t_despawn'], inclusive='left'
    )].copy()

    if ff_in_obs_df.groupby('point_index').count().max().max() > env.num_obs_ff:
        raise ValueError(
            "The number of fireflies in the observation exceeds the number in the environment."
        )

    # Collect all monkey data
    monkey_information = pack_monkey_information(
        time, monkey_x, monkey_y, speed, ang_speed, is_stop, monkey_angle, env.dt
    )
    monkey_information['point_index'] = range(len(monkey_information))

    process_monkey_information.add_more_columns_to_monkey_information(
        monkey_information)

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
    ff_in_obs_df['time_since_last_vis'] = np.round(
        ff_in_obs_df['time_since_last_vis'], num_decimals_of_dt)
    ff_in_obs_df.reset_index(drop=True, inplace=True)

    ff_in_obs_df = ff_in_obs_df[
        ['index_in_ff_dataframe', 'index_in_ff_information', 'index_in_ff_flash', 'point_index',
         'ff_x_noisy', 'ff_y_noisy', 'time_since_last_vis', 'visible', 'pose_unreliable']
    ].copy()
    ff_in_obs_df.rename(
        columns={'index_in_ff_dataframe': 'ff_index'}, inplace=True)

    obs_ff_indices_in_ff_dataframe = pd.DataFrame(
        ff_in_obs_df.groupby('point_index')['ff_index'].apply(list)
    )
    obs_ff_indices_in_ff_dataframe = obs_ff_indices_in_ff_dataframe.merge(
        pd.DataFrame(pd.Series(range(n_steps), name='point_index')),
        on='point_index', how='right'
    )

    obs_ff_indices_in_ff_dataframe = obs_ff_indices_in_ff_dataframe['ff_index'].tolist(
    )
    obs_ff_indices_in_ff_dataframe = [
        np.array(x) if isinstance(x, list) else np.array([]) for x in obs_ff_indices_in_ff_dataframe
    ]

    # Capture rate
    if monkey_information['time'].max() > 0:
        ff_capture_rate = len(set(ff_caught_T_new)) / \
            monkey_information['time'].max()
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


def pack_monkey_information(time, monkey_x, monkey_y, speed, ang_speed, is_stop, monkey_angle, dt,
                            ):
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
    speed: list
        containing a series of linear speeds of the monkey/agent  
    monkey_angle: list    
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
    speed = np.array(speed)
    ang_speed = np.array(ang_speed)
    monkey_angle = np.array(monkey_angle)
    monkey_angle = np.remainder(monkey_angle, 2*pi)

    monkey_information = {
        'time': time,
        'monkey_x': monkey_x,
        'monkey_y': monkey_y,
        'speed': speed,
        'ang_speed': ang_speed,
        'monkey_speeddummy': [1-i for i in is_stop],
        'monkey_angle': monkey_angle,
    }

    delta_x = np.diff(monkey_information['monkey_x'])
    delta_y = np.diff(monkey_information['monkey_y'])
    delta_position = np.sqrt(np.square(delta_x)+np.square(delta_y))
    crossing_boundary = np.append(0, (delta_position > 100).astype('int'))
    monkey_information['crossing_boundary'] = crossing_boundary

    monkey_information = pd.DataFrame(monkey_information)

    return monkey_information
