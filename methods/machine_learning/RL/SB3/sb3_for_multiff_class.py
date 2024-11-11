import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import basic_func, base_processing_class, further_processing_class
from pattern_discovery import organize_patterns_and_features
from visualization import additional_plots, plot_statistics, animation_utils

from machine_learning.RL.env_related import base_env, collect_agent_data_utils, env_for_sb3
from machine_learning.RL.SB3 import interpret_neural_network, rl_for_multiff_class, rl_for_multiff_utils, SB3_functions

import time as time_package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from os.path import exists
import time as time_package
import torch.nn as nn
import gc
import math
import csv
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class SB3forMultifirefly(rl_for_multiff_class._RLforMultifirefly):

    def __init__(self, 
                overall_folder='RL_models/SB3_stored_models/all_agents/env1_relu/',   
                add_date_to_model_folder_name=False,
                **kwargs):
        
        super().__init__(overall_folder=overall_folder, 
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         **kwargs)

        self.sb3_or_lstm = 'sb3'
        self.monkey_name = None

        if self.use_env2:
            self.env_class = env_for_sb3.EnvForSB3_2
        else:
            self.env_class = env_for_sb3.EnvForSB3

    def save_best_model_after_curriculum(self, whether_save_replay_buffer=True):
        self.save_agent(whether_save_replay_buffer=whether_save_replay_buffer, dir_name=self.best_model_after_curriculum_dir_name)

        
    def load_best_model_after_curriculum(self, load_replay_buffer=True):
        self.load_agent(load_replay_buffer=load_replay_buffer, dir_name=self.best_model_after_curriculum_dir_name)


    def make_env(self, monitor_dir=None, **additional_env_kwargs):
        os.makedirs(self.model_folder_name, exist_ok=True)
        self.env_kwargs.update(additional_env_kwargs)
        self.env = self.env_class(**self.env_kwargs)
        if monitor_dir is None:
            monitor_dir = self.model_folder_name
        self.env = Monitor(self.env, monitor_dir)
        # check if monitor file is present
        if not exists(os.path.join(monitor_dir, 'monitor.csv')):
            # make the env again
            self.env = self.env_class(**self.env_kwargs)
            self.env = Monitor(self.env, monitor_dir)
            # check again
            if not exists(os.path.join(monitor_dir, 'monitor.csv')):
                raise ValueError("Monitor file not found after making the env.")


    def make_agent(self, **kwargs):
        
        if self.agent_params is None:
            self.agent_params = {'learning_rate': kwargs.get('learning_rate', 0.0003),
                                'batch_size': kwargs.get('batch_size', 1024),
                                'target_update_interval': kwargs.get('target_update_interval', 50),
                                'buffer_size': kwargs.get('buffer_size', 1000000),
                                'learning_starts': kwargs.get('learning_starts', 10000),
                                'train_freq': kwargs.get('train_freq', 100),
                                #'train_freq': kwargs.get('train_freq', 1,),
                                'gradient_steps': kwargs.get('gradient_steps', 10),
                                'ent_coef': kwargs.get('ent_coef', 'auto'),
                                'policy_kwargs': kwargs.get('policy_kwargs', dict(activation_fn=nn.ReLU, net_arch=[256, 128])),
                                'gamma': rl_for_multiff_utils.calculate_model_gamma(self.env.env.dt),
                                } 
        else:
            self.agent_params.update(kwargs)    
        
        # num_nodes = self.env.env.obs_space_length * 2 + 12
        # print('num_nodes in each layer of the neural network:', num_nodes)

        self.buffer_size = self.agent_params['buffer_size']

        self.sac_model = SAC("MlpPolicy", 
                            self.env,
                            **self.agent_params)
        
        self.agent_params = rl_for_multiff_utils.get_agent_params_from_the_current_sac_model(self.sac_model)
        print('Made agent with the following params:', self.agent_params)



    def regular_training(self, timesteps=2000000, best_model_save_path=None, env_params_to_save=None):
        if best_model_save_path is None:
            best_model_save_path = self.model_folder_name

        if env_params_to_save is None:
            env_params_to_save = self.env_kwargs
        self.store_env_params(model_folder_name=best_model_save_path, env_params_to_save=env_params_to_save)

        stop_train_callback = SB3_functions.StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=15, verbose=1, model_folder_name=self.model_folder_name,
                                                            overall_folder=self.overall_folder, agent_id=self.agent_id)
        self.callback = EvalCallback(self.env, eval_freq=10000, callback_after_eval=stop_train_callback, verbose=1, 
                                     best_model_save_path=best_model_save_path, n_eval_episodes=3)
        
        self.sac_model.learn(total_timesteps=int(timesteps), callback=self.callback)


    def _make_agent_for_curriculum_training(self):
        self.make_agent(learning_rate=0.0015,
                        train_freq=10,
                        gradient_steps=1)


    def _make_initial_env_for_curriculum_training(self):
        self.env_kwargs_for_curriculum_training = copy.deepcopy(self.env_kwargs)
        self.make_env(monitor_dir=self.best_model_after_curriculum_dir_name)
        self._make_env_suitable_for_curriculum_training()
        self.env.env.angular_terminal_vel = 0.32
        self.env_kwargs_for_curriculum_training['angular_terminal_vel'] = self.env.env.angular_terminal_vel
        self._update_env_dt(dt=0.25)

    def _update_env_dt(self, dt=0.25):
        self.env.env.dt = dt
        self.env.env.full_memory = math.floor(self.env.env.max_in_memory_time/self.env.env.dt)
        self.env.env.visible_time_range = self.env.env.flash_on_interval + self.env.env.max_in_memory_time + self.env.env.dt
        self.env_kwargs_for_curriculum_training['dt'] = self.env.env.dt
        self.env_kwargs_for_curriculum_training['full_memory'] = self.env.env.full_memory
        self.env_kwargs_for_curriculum_training['visible_time_range'] = self.env.env.visible_time_range

    def _train_till_reaching_reward_threshold(self, n_eval_episodes=1, ff_caught_rate_threshold=1/5):
        reward_threshold = rl_for_multiff_utils.calculate_reward_threshold_for_curriculum_training(self.env.env, n_eval_episodes=n_eval_episodes, ff_caught_rate_threshold=ff_caught_rate_threshold)
        print('reward_threshold:', reward_threshold)
        stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold) # or 10ff/250s  
        callback = EvalCallback(self.env, eval_freq=8000, callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=n_eval_episodes)
        self.sac_model.learn(total_timesteps=1000000, callback=callback)
        if callback.best_mean_reward < reward_threshold: 
            raise ValueError("The agent is not able to catch enough ff after training for 1e6 timesteps.")



    def _use_while_loop_for_curriculum_training(self):
        while (self.env.env.dt > self.env_kwargs['dt']) | (self.env.env.angular_terminal_vel > 0.01):
            gc.collect()
            # Note: 0.00222 = 0.0035/(pi/2), same as the monkey's threshold
            try:
                self._train_till_reaching_reward_threshold()
            except ValueError as e:
                print(f"Error message: {e}")
                break
            self.save_best_model_after_curriculum()
            self._change_env_after_meeting_reward_threshold()

        # after all the conditions are met, train the agent once again to ensure performance (stop training with no improvement)
        self.regular_training(best_model_save_path=self.best_model_after_curriculum_dir_name, env_params_to_save=self.env_kwargs_for_curriculum_training)


    def _change_env_after_meeting_reward_threshold(self):
        self._update_env_dt(dt=max(self.env.env.dt/2, self.env_kwargs['dt']))
        self.sac_model.gamma = rl_for_multiff_utils.calculate_model_gamma(self.env.env.dt)
        self.env.env.angular_terminal_vel = max(self.env.env.angular_terminal_vel/2, 0.01)
        self.env_kwargs_for_curriculum_training['angular_terminal_vel'] = self.env.env.angular_terminal_vel
        print('Current dt:', self.env.env.dt)
        print('Current full memory:', self.env.env.full_memory)
        print('Current gamma:', self.sac_model.gamma)
        print('Current angular_terminal_vel:', self.env.env.angular_terminal_vel)


    def _fill_up_replay_buffer_for_best_model_after_curriculum_training(self):
        self.make_env(monitor_dir=self.best_model_after_curriculum_dir_name, **self.env_kwargs)
        self._make_env_suitable_for_curriculum_training()
        self.load_best_model_after_curriculum(load_replay_buffer=True) # need to do this after changing env

        # continue to train agent so as to fill up the replay buffer
        callback = SB3_functions.SaveOnBestTrainingRewardCallback(check_freq=20000, model_folder_name=self.best_model_after_curriculum_dir_name)
        self.sac_model.learn(total_timesteps=int(self.buffer_size * 1.2), callback=callback)
        # # save replay_buffer
        # self.sac_model.save_replay_buffer(os.path.join(dir_name, 'buffer')) # I added this


    def _further_process_best_model_after_curriculum_training(self):

        #self._fill_up_replay_buffer_for_best_model_after_curriculum_training()
        # Now, change the env back to train the current agent
        self.make_env(**self.env_kwargs)
        self.load_best_model_after_curriculum(load_replay_buffer=True)
        self._restore_env_params_after_curriculum_training()
        self.env.env.full_memory = math.floor(self.env.env.max_in_memory_time/self.env.env.dt)
        self.env_kwargs_for_curriculum_training['full_memory'] = self.env.env.full_memory


    def _restore_env_params_after_curriculum_training(self):
        self.env.env.dt = self.env_kwargs['dt']
        self.env.env.dv_cost_factor = self.env_kwargs['dv_cost_factor']
        self.env.env.dw_cost_factor = self.env_kwargs['dw_cost_factor']
        self.env.env.w_cost_factor = self.env_kwargs['w_cost_factor']
        self.env.distance2center_cost = 0

    def save_agent(self, whether_save_replay_buffer=False, dir_name=None, model_name='best_model'):
        if dir_name is None:
            dir_name = self.model_folder_name  
        
        os.makedirs(dir_name, exist_ok=True)
        self.sac_model.save(os.path.join(dir_name, model_name))
        print('Saved agent:', os.path.join(dir_name, model_name))
        if whether_save_replay_buffer:
            self.sac_model.save_replay_buffer(os.path.join(dir_name, 'buffer')) # I added this
            print('Saved replay buffer:', os.path.join(dir_name, 'buffer'))


    def load_agent(self, load_replay_buffer=True, keep_current_agent_params=True, dir_name=None, model_name='best_model'): 
        if dir_name is None:
            dir_name = self.model_folder_name

        path = os.path.join(dir_name, model_name + '.zip')
        try:
            self.sac_model = self.sac_model.load(path, env = self.env) 
            print("Loaded existing agent:", path)

            if load_replay_buffer:
                path2 = os.path.join(dir_name, 'buffer')
                self.sac_model.load_replay_buffer(path2)
                print("Loaded existing replay buffer:", path2)

            if keep_current_agent_params:
                for key, item in self.agent_params.items():
                    setattr(self.sac_model, key, item)

            print('Params from agent after loading:')
            print(rl_for_multiff_utils.get_agent_params_from_the_current_sac_model(self.sac_model))


        except Exception as e:
            print(f"There was an error retrieving agent or replay_buffer in {path}. Error message {e}")
            raise ValueError(f"There was an error retrieving agent or replay_buffer in {path}. Error message {e}")


    def _run_current_agent_after_curriculum_training(self):
        timesteps=2000000
        self.regular_training(timesteps=timesteps)
        self.successful_training = True


