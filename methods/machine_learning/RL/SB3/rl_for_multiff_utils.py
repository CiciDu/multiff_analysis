import sys
from planning_analysis.agent_analysis import agent_plan_factors_x_sess_class
import os
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import time as time_package
import math
import csv
import json
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def extract_cost_params_from_folder_name(folder):
    splitted_name = folder.split('_')
    dv_cost_factor = float(splitted_name[0][2:])
    dw_cost_factor = float(splitted_name[1][2:])
    w_cost_factor = float(splitted_name[2][1:])
    params = {'dv_cost_factor': dv_cost_factor, 'dw_cost_factor': dw_cost_factor, 'w_cost_factor': w_cost_factor}
    return params


def retrieve_or_make_family_of_agents_log(overall_folder):
    filepath = overall_folder + 'family_of_agents_log.csv'
    if not exists(filepath):
        family_of_agents_log = pd.DataFrame(columns=['dv_cost_factor', 'dw_cost_factor', 'w_cost_factor',
                                            'action_noise_std', 'ffxy_noise_std', 'num_obs_ff', 'max_in_memory_time', 
                                            'finished_training', 'year', 'month', 'date', 'training_time', 'successful_training'])
        family_of_agents_log.to_csv(filepath)
        print("No family_of_agents_log existed. Made new family_of_agents_log")
    else:
        family_of_agents_log = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
    return family_of_agents_log


def calculate_model_gamma(dt):
    gamma_0 = 0.995
    dt_0 = 0.25

    n = math.log(dt_0/dt, 2)
    gamma = 1 - 1/(2**n) + gamma_0/(2**n)

    return gamma


def get_agent_params_from_the_current_sac_model(sac_model):
    params = {'learning_rate': sac_model.learning_rate,
                'batch_size': sac_model.batch_size,
                'target_update_interval': sac_model.target_update_interval,
                'buffer_size': sac_model.buffer_size,
                'learning_starts': sac_model.learning_starts,
                'train_freq': sac_model.train_freq,
                'gradient_steps': sac_model.gradient_steps,
                'ent_coef': sac_model.ent_coef,
                'policy_kwargs': sac_model.policy_kwargs,
                'gamma': sac_model.gamma}
    return params


def calculate_reward_threshold_for_curriculum_training(env, n_eval_episodes=1, ff_caught_rate_threshold=1/5):
    reward_threshold = (n_eval_episodes * env.episode_len * env.dt) * ff_caught_rate_threshold * (env.reward_per_ff - env.distance2center_cost * 15)
    return reward_threshold


def get_agent_name_from_folder_name(prefix, agent_folder):
    if '/' in agent_folder:
        agent_folder = agent_folder.split('/')[-1]

    try:
        _ = extract_cost_params_from_folder_name(agent_folder)
        whether_with_cost = 'costT'
    except ValueError:
        #print('No cost params extracted.')
        whether_with_cost = 'costF'
    agent_name = f'{prefix}_{whether_with_cost}'
    return agent_name


# write code to get agent name from params
def get_agent_name_from_params(params):

    ff_indicator = 'ff' + str(params['num_obs_ff'])

    memory_indicator = 'mem' + str(params['max_in_memory_time'])

    if  ((params['dv_cost_factor']) == 10) & \
        ((params['dw_cost_factor']) == 10) & ((params['w_cost_factor']) == 10):
        cost_indicator = 'costT'
    elif ((params['dv_cost_factor']) == 0) & \
        ((params['dw_cost_factor']) == 0) & ((params['w_cost_factor']) == 0):
        cost_indicator = 'costF'
    else:
        cost_indicator = "dv" + str(params['dv_cost_factor']) + \
                        "_dw" + str(params['dw_cost_factor']) + "_w" + str(params['w_cost_factor'])
        
    agent_name = ff_indicator + '_' + memory_indicator + '_' + cost_indicator 
    return agent_name


def store_params(model_folder_name, params):
    if model_folder_name is None:
        raise ValueError('model_folder_name is None')

    params_file = os.path.join(model_folder_name, 'env_params.txt')

    # clear the content of the file if it exists
    if os.path.exists(params_file):
        open(params_file, 'w').close()

    with open(params_file, "w") as fp:
        json.dump(params, fp)  # encode dict into JSON
        print('Saving env params to', params_file)
    return



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


def get_folders_with_params(path = 'RL_models/SB3_stored_models/all_agents/env1_relu'):

    dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # get all subfolders and sub-sub folders in path
    all_folders = []
    for dir in dirs:
        folders = os.listdir(f'{path}/{dir}')
        for folder in folders:
            all_folders.append(f'{path}/{dir}/{folder}')

    # take out folders in all_folders if it contains env_params.csv
    folders_with_params = []
    for folder in all_folders:
        # if folder is a directory
        if os.path.isdir(folder):
            if 'env_params.txt' in os.listdir(folder):
                folders_with_params.append(folder)

    return folders_with_params


def add_essential_agent_params_info(df, params, agent_name=None):
    df = df.copy()
    df['num_obs_ff'] = params['num_obs_ff']
    df['max_in_memory_time'] = params['max_in_memory_time']
    df['whether_with_cost'] = 'with_cost' if (params['dv_cost_factor'] > 0) else 'no_cost'
    df['dv_cost_factor'] = params['dv_cost_factor']
    df['dw_cost_factor'] = params['dw_cost_factor']
    df['w_cost_factor'] = params['w_cost_factor']

    if agent_name is not None:
        df['id'] = agent_name
    return df