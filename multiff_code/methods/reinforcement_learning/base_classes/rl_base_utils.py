import os
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import math
import json
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract_cost_params_from_folder_name(folder):
    splitted_name = folder.split('_')
    dv_cost_factor = float(splitted_name[0][2:])
    dw_cost_factor = float(splitted_name[1][2:])
    w_cost_factor = float(splitted_name[2][1:])
    params = {'dv_cost_factor': dv_cost_factor,
              'dw_cost_factor': dw_cost_factor, 'w_cost_factor': w_cost_factor}
    return params


def retrieve_or_make_family_of_agents_log(overall_folder):
    filepath = os.path.join(overall_folder, 'family_of_agents_log.csv')
    if not exists(filepath):
        family_of_agents_log = pd.DataFrame(columns=['dv_cost_factor', 'dw_cost_factor', 'w_cost_factor',
                                            'v_noise_std', 'w_noise_std', 'ffr_noise_scale', 'num_obs_ff', 'max_in_memory_time',
                                                     'finished_training', 'year', 'month', 'date', 'training_time', 'successful_training'])
        family_of_agents_log.to_csv(filepath)
        print("No family_of_agents_log existed. Made new family_of_agents_log")
    else:
        family_of_agents_log = pd.read_csv(
            filepath).drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
    return family_of_agents_log


def calculate_model_gamma(dt):
    gamma_0 = 0.998
    dt_0 = 0.1
    gamma = gamma_0 ** (dt / dt_0)
    return gamma


def get_agent_params_from_the_current_sac_model(rl_agent):
    params = {'learning_rate': rl_agent.learning_rate,
              'batch_size': rl_agent.batch_size,
              'target_update_interval': rl_agent.target_update_interval,
              'buffer_size': rl_agent.buffer_size,
              'learning_starts': rl_agent.learning_starts,
              'train_freq': rl_agent.train_freq,
              'gradient_steps': rl_agent.gradient_steps,
              'ent_coef': rl_agent.ent_coef,
              'policy_kwargs': rl_agent.policy_kwargs,
              'gamma': rl_agent.gamma}
    return params


def calculate_reward_threshold_for_curriculum_training(env, n_eval_episodes=1, ff_caught_rate_threshold=0.1):
    """
    Compute a curriculum reward threshold.

    Supports both raw envs and vectorized envs (Dummy/Subproc). For vectorized envs,
    queries a small helper method on the wrapper to retrieve base parameters.
    """
    # Try direct unwrap through .env chain (non-vectorized)
    base_env = env
    while hasattr(base_env, 'env'):
        try:
            base_env = base_env.env
        except Exception:
            break

    def _try_get(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    episode_len = _try_get(base_env, 'episode_len')
    dt = _try_get(base_env, 'dt')
    reward_per_ff = _try_get(base_env, 'reward_per_ff')
    distance2center_cost = _try_get(base_env, 'distance2center_cost')

    # If any missing, try vectorized env query via env_method on index 0
    if any(x is None for x in [episode_len, dt, reward_per_ff, distance2center_cost]):
        try:
            if hasattr(env, 'env_method'):
                params_list = env.env_method('get_basic_params', indices=0)
                params = params_list[0] if isinstance(
                    params_list, (list, tuple)) else params_list
                if isinstance(params, dict):
                    episode_len = episode_len if episode_len is not None else params.get(
                        'episode_len', None)
                    dt = dt if dt is not None else params.get('dt', None)
                    reward_per_ff = reward_per_ff if reward_per_ff is not None else params.get(
                        'reward_per_ff', None)
                    distance2center_cost = distance2center_cost if distance2center_cost is not None else params.get(
                        'distance2center_cost', None)
        except Exception:
            pass

    if any(x is None for x in [episode_len, dt, reward_per_ff, distance2center_cost]):
        raise AttributeError(
            'Unable to retrieve basic environment parameters required for reward threshold calculation.')

    reward_threshold = (n_eval_episodes * episode_len * dt) * \
        ff_caught_rate_threshold * \
        (reward_per_ff - distance2center_cost * 15)
    # - 200  # including the rest of the cost like velocity cost
    return reward_threshold


# # write code to get agent name from params
# def get_agent_name_from_params(params):

#     ff_indicator = 'ff' + str(params['num_obs_ff'])

#     memory_indicator = 'mem' + str(params['max_in_memory_time'])

#     if ((params['dv_cost_factor']) == 1) & \
#             ((params['dw_cost_factor']) == 1) & ((params['w_cost_factor']) == 1):
#         cost_indicator = 'costT'
#     elif ((params['dv_cost_factor']) == 0) & \
#             ((params['dw_cost_factor']) == 0) & ((params['w_cost_factor']) == 0):
#         cost_indicator = 'costF'
#     else:
#         cost_indicator = "dv" + str(params['dv_cost_factor']) + \
#             "_dw" + str(params['dw_cost_factor']) + \
#             "_w" + str(params['w_cost_factor'])

#     agent_name = ff_indicator + '_' + memory_indicator + '_' + cost_indicator
#     return agent_name


def retrieve_params(model_folder_name):
    if model_folder_name is None:
        raise ValueError('model_folder_name is None')

    # Open the csv file for reading
    params_file = os.path.join(model_folder_name, 'env_params.txt')

    # Open the file for reading
    with open(params_file, "r") as fp:
        # Load the dictionary from the file
        params = json.load(fp)

    return params


def get_agent_folders(path='multiff_analysis/RL_models/SB3_stored_models/all_agents/env1_relu'):
    agent_folders = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(path):
        # Check if this folder contains checkpoint_manifest.json
        if 'checkpoint_manifest.json' in files:
            agent_folders.append(root)

    return agent_folders


# def get_agent_folders(path='multiff_analysis/RL_models/SB3_stored_models/all_agents/env1_relu'):

#     dirs = [f for f in os.listdir(
#         path) if os.path.isdir(os.path.join(path, f))]

#     # get all subfolders and sub-sub folders in path
#     all_folders = []
#     for dir in dirs:
#         folders = os.listdir(f'{path}/{dir}')
#         for folder in folders:
#             all_folders.append(f'{path}/{dir}/{folder}')

#     # take out folders in all_folders if it contains checkpoint_manifest.json
#     agent_folders = []
#     for folder in all_folders:
#         # if folder is a directory
#         if os.path.isdir(folder):
#             print(folder)
#             if 'checkpoint_manifest.json' in os.listdir(folder):
#                 agent_folders.append(folder)

#     return agent_folders


def add_essential_agent_params_info(df, params):
    df = df.copy()
    df['num_obs_ff'] = params['num_obs_ff']
    df['max_in_memory_time'] = params['max_in_memory_time']
    df['whether_with_cost'] = 'with_cost' if (
        params['dv_cost_factor'] > 0) else 'no_cost'
    df['dv_cost_factor'] = params['dv_cost_factor']
    df['dw_cost_factor'] = params['dw_cost_factor']
    df['w_cost_factor'] = params['w_cost_factor']

    return df


def read_checkpoint_manifest(checkpoint_dir):
    if not isinstance(checkpoint_dir, str) or len(checkpoint_dir) == 0:
        raise ValueError(
            f"Warning: checkpoint_dir is not a string or is empty: {checkpoint_dir}")
    manifest_path = os.path.join(checkpoint_dir, 'checkpoint_manifest.json')
    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            # tolerate both dict payloads and legacy raw env_kwargs
            if isinstance(data, dict):
                return data
            return {'env_params': data}
    except Exception as e:
        raise ValueError(f"Failed to read manifest at {manifest_path}: {e}")


def write_checkpoint(checkpoint_dir, current_env_kwargs):
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, 'checkpoint_manifest.json')
    try:
        with open(manifest_path, 'w') as f:
            # If a full manifest dict is provided, write it; else wrap as env_params
            payload = current_env_kwargs if isinstance(current_env_kwargs, dict) and (
                'env_params' in current_env_kwargs or 'algorithm' in current_env_kwargs or 'model_files' in current_env_kwargs
            ) else {'env_params': current_env_kwargs}
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: failed to write manifest at {manifest_path}: {e}")
