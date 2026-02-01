
from reinforcement_learning.agents.feedforward import sb3_env
from reinforcement_learning.base_classes import rl_base_class, rl_base_utils
from reinforcement_learning.agents.feedforward import sb3_utils
from reinforcement_learning.base_classes import env_utils

import os
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from os.path import exists
import torch.nn as nn
import gc
import math
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SB3forMultifirefly(rl_base_class._RLforMultifirefly):

    def __init__(self,
                 overall_folder='multiff_analysis/RL_models/SB3_stored_models/all_agents/env1_relu/',
                 add_date_to_model_folder_name=False,
                 **kwargs):

        super().__init__(overall_folder=overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         **kwargs)

        self.agent_type = 'sb3'
        self.monkey_name = None

        self.env_class = sb3_env.EnvForSB3

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

    def make_env(self, monitor_dir=None, **env_kwargs):
        super().make_env(**env_kwargs)

        os.makedirs(self.model_folder_name, exist_ok=True)
        if monitor_dir is None:
            monitor_dir = self.model_folder_name
        self.env = Monitor(self.env, monitor_dir)

    def make_agent(self, **kwargs):

        if self.agent_params is None:
            self.agent_params = {'learning_rate': kwargs.get('learning_rate', 0.0003),
                                 'batch_size': kwargs.get('batch_size', 1024),
                                 'target_update_interval': kwargs.get('target_update_interval', 50),
                                 'buffer_size': kwargs.get('buffer_size', 1000000),
                                 'learning_starts': kwargs.get('learning_starts', 10000),
                                 'train_freq': kwargs.get('train_freq', 10),
                                 # 'train_freq': kwargs.get('train_freq', 1,),
                                 'gradient_steps': kwargs.get('gradient_steps', 10),
                                 'ent_coef': kwargs.get('ent_coef', 'auto'),
                                 'policy_kwargs': kwargs.get('policy_kwargs', dict(activation_fn=nn.ReLU, net_arch=[256, 128])),
                                 'gamma': 0.99,
                                 }
        else:
            self.agent_params.update(kwargs)

        # num_nodes = self.env.env.obs_space_length * 2 + 12
        # print('num_nodes in each layer of the neural network:', num_nodes)

        self.buffer_size = self.agent_params['buffer_size']

        self.rl_agent = SAC("MlpPolicy",
                            self.env,
                            **self.agent_params)

        self.agent_params = rl_base_utils.get_agent_params_from_the_current_sac_model(
            self.rl_agent)
        print('Made agent with the following params:', self.agent_params)

    def regular_training(self, timesteps=2000000, eval_eps_freq=20, best_model_save_path=None):
        if best_model_save_path is None:
            best_model_save_path = self.best_model_postcurriculum_dir

        stop_train_callback = sb3_utils.StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=15, verbose=1, model_folder_name=self.model_folder_name,
                                                                         overall_folder=self.overall_folder, agent_id=self.agent_id,
                                                                         best_model_save_path=best_model_save_path)

        # Note: by adding best_model_save_path, the callback can save the best model after each evaluation
        if best_model_save_path is not None:
            os.makedirs(best_model_save_path, exist_ok=True)
        self.callback = EvalCallback(self.env, eval_freq=self.current_env_kwargs['episode_len'] * eval_eps_freq, callback_after_eval=stop_train_callback, verbose=1,
                                     best_model_save_path=best_model_save_path, n_eval_episodes=3)
        self.write_checkpoint_manifest(best_model_save_path)
        self.rl_agent.learn(total_timesteps=int(
            timesteps), callback=self.callback)

    def _make_agent_for_curriculum_training(self):
        print('Making agent for curriculum training...')
        self.make_agent(learning_rate=0.0015,
                        train_freq=10,
                        gradient_steps=1)

    def make_init_env_for_curriculum_training(self, initial_flash_on_interval=0.3,
                                              **kwargs):
        monitor_dir = self.best_model_postcurriculum_dir
        os.makedirs(monitor_dir, exist_ok=True)
        print(f'Making initial env for curriculum training...')
        self.make_env(monitor_dir=monitor_dir, **self.input_env_kwargs)
        self._make_init_env_for_curriculum_training(initial_flash_on_interval=initial_flash_on_interval,
                                                    **kwargs)

    def _curriculum_train_step(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        stop_train_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold)
        callback = EvalCallback(
            self.env, eval_freq=self.current_env_kwargs['episode_len'] * eval_eps_freq, callback_after_eval=stop_train_callback, verbose=1,
            n_eval_episodes=num_eval_episodes, best_model_save_path=self.best_model_in_curriculum_dir)
        self.write_checkpoint_manifest(self.best_model_in_curriculum_dir)
        self.rl_agent.learn(total_timesteps=1000000, callback=callback)
        best_mean_reward = getattr(callback, 'best_mean_reward', float('-inf'))
        passed = bool(best_mean_reward >= reward_threshold)
        return passed, float(best_mean_reward)

    def _post_curriculum_final_train(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        # SB3 uses early stopping on no improvement; no explicit reward threshold used here
        self.regular_training(
            best_model_save_path=self.best_model_postcurriculum_dir)
        # Final model load handled by base

    def save_agent(self, whether_save_replay_buffer=False, dir_name=None):
        model_name = 'best_model'
        if dir_name is None:
            dir_name = self.model_folder_name

        os.makedirs(dir_name, exist_ok=True)
        self.rl_agent.save(os.path.join(dir_name, model_name))
        print('Saved agent:', os.path.join(dir_name, model_name))
        if whether_save_replay_buffer:
            self.rl_agent.save_replay_buffer(
                os.path.join(dir_name, 'buffer'))  # I added this
            print('Saved replay buffer:', os.path.join(dir_name, 'buffer'))

        self.write_checkpoint_manifest(dir_name)

    def write_checkpoint_manifest(self, dir_name):
        rl_base_utils.write_checkpoint(dir_name, {
            'algorithm': 'sb3_sac',
            'model_file': f'best_model.zip',
            'replay_buffer': 'buffer',
            'num_timesteps': getattr(self.rl_agent, 'num_timesteps', None),
            'env_params_path': 'env_params.txt',
            'env_params': self.current_env_kwargs,
        })

    def load_agent(self, load_replay_buffer=True, keep_current_agent_params=True, dir_name=None, model_name='best_model', restore_env_from_checkpoint=True):
        manifest = rl_base_utils.read_checkpoint_manifest(dir_name)
        model_file = manifest.get('model_file') if isinstance(
            manifest, dict) else None
        path = os.path.join(dir_name, model_file) if model_file else os.path.join(
            dir_name, model_name + '.zip')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        if restore_env_from_checkpoint:
            self.make_env(**manifest['env_params'])
        self.make_agent()
        self.rl_agent = self.rl_agent.load(path, env=self.env)
        print("Loaded existing agent:", path)

        if load_replay_buffer:
            buffer_name = manifest.get('replay_buffer') if isinstance(
                manifest, dict) else 'buffer'
            path2 = os.path.join(dir_name, buffer_name)
            if os.path.exists(path2):
                self.rl_agent.load_replay_buffer(path2)
                print("Loaded existing replay buffer:", path2)
            else:
                # Fallback: look in the parent directory (agent root)
                fallback_path = os.path.join(
                    os.path.dirname(dir_name), buffer_name)
                if os.path.exists(fallback_path):
                    self.rl_agent.load_replay_buffer(fallback_path)
                    print("Loaded existing replay buffer from fallback:",
                          fallback_path)
                else:
                    print(
                        f"Replay buffer not found at {path2}; proceeding without it.")

        if keep_current_agent_params and (self.agent_params is not None):
            for key, item in self.agent_params.items():
                setattr(self.rl_agent, key, item)

        print('Params from agent after loading:')
        print(rl_base_utils.get_agent_params_from_the_current_sac_model(
            self.rl_agent))
        self.loaded_agent_dir = dir_name
        return
