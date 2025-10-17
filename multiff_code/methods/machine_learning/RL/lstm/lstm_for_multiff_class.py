

from machine_learning.RL.env_related import env_for_lstm
from machine_learning.RL.SB3 import rl_for_multiff_utils, rl_for_multiff_class
from machine_learning.RL.lstm import LSTM_functions
from machine_learning.RL.env_related import env_utils

import os
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
import pickle
import io
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LSTMforMultifirefly(rl_for_multiff_class._RLforMultifirefly):

    def __init__(self,
                 overall_folder='RL_models/LSTM_stored_models/all_agents/gen_0/',
                 model_folder_name=None,
                 add_date_to_model_folder_name=False,
                 max_in_memory_time=1,
                 **additional_env_kwargs):

        super().__init__(overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         model_folder_name=model_folder_name,
                         max_in_memory_time=max_in_memory_time,
                         **additional_env_kwargs)

        self.sb3_or_lstm = 'lstm'
        self.env_class = env_for_lstm.EnvForLSTM

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.default_input_env_kwargs,
            **self.additional_env_kwargs
        }

    def make_agent(self,
                   agent_params_already_set_ok=True,
                   **kwargs):

        # Ensure environment exists before constructing agent-dependent params
        if not hasattr(self, 'env'):
            self.make_env(**self.input_env_kwargs)

        if agent_params_already_set_ok & (self.agent_params is not None):
            existing_agent_params = self.agent_params.copy()
        else:
            existing_agent_params = {}

        self.agent_params = {
            "gamma": rl_for_multiff_utils.calculate_model_gamma(self.env.dt),
            "state_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "action_dim": self.env.action_space.shape[0],
            "action_range": 1.0,
            # used 100 before
            "replay_buffer_size": kwargs.get('replay_buffer_size', 500),
            # "replay_buffer": GRU_functions.ReplayBufferGRU(100),  # Uncomment if using GRU
            "hidden_dim": kwargs.get('hidden_dim', 256),
            "soft_q_lr": kwargs.get('soft_q_lr', 0.0015),
            "policy_lr": kwargs.get('policy_lr', 0.003),
            "alpha_lr": kwargs.get('alpha_lr', 0.002),
            "batch_size": kwargs.get('batch_size', 10),
            "update_itr": kwargs.get('update_itr', 1),
            "reward_scale": kwargs.get('reward_scale', 0.5),  # used 10 before
            "target_entropy": kwargs.get('target_entropy', - self.env.action_space.shape[0]),
            "soft_tau": kwargs.get('soft_tau', 0.015),
            "train_freq": kwargs.get('train_freq', 5),
            "auto_entropy": kwargs.get('auto_entropy', True),
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
        }

        self.agent_params.update(existing_agent_params)
        self.agent_params.update(kwargs)

        self.replay_buffer = LSTM_functions.ReplayBufferLSTM2(
            self.agent_params['replay_buffer_size'])
        self.agent_params['replay_buffer'] = self.replay_buffer
        self.sac_model = LSTM_functions.SAC_Trainer(**self.agent_params)

    def make_initial_env_for_curriculum_training(self, initial_flash_on_interval=3, initial_angular_terminal_vel=0.64, initial_reward_boundary=75):
        self.make_env(**self.input_env_kwargs)
        self._make_initial_env_for_curriculum_training(initial_angular_terminal_vel=initial_angular_terminal_vel,
                                                       initial_flash_on_interval=initial_flash_on_interval,
                                                       initial_reward_boundary=initial_reward_boundary)

    def _make_agent_for_curriculum_training(self):
        self.make_agent()

    def _use_while_loop_for_curriculum_training(self, eval_eps_freq=20, num_eval_episodes=2):
        while (self.env.flash_on_interval > self.input_env_kwargs['flash_on_interval']) or (self.env.angular_terminal_vel > 0.01) or \
              ('reward_boundary' in self.curriculum_env_kwargs and self.env.reward_boundary > self.input_env_kwargs['reward_boundary']):

            gc.collect()
            reward_threshold = rl_for_multiff_utils.calculate_reward_threshold_for_curriculum_training(
                self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
            print(
                'Current reward_threshold to progress in curriculum training:', reward_threshold)
            # reward_threshold = 1000
            self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                                  reward_threshold_to_stop_on=reward_threshold,
                                  dir_name=self.best_model_in_curriculum_dir)
            print(f'Best average reward record: {self.best_avg_reward_record}')
            if self.best_avg_reward_record < reward_threshold:
                raise ValueError(f'Best average reward record {self.best_avg_reward_record} is less than reward threshold {reward_threshold}. Can\'t progress in curriculum training.')

            self._change_env_after_meeting_reward_threshold()

        # after all condition is met, train the agent once more until it reaches the desired performance
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        self.make_env(**self.input_env_kwargs)
        self.load_best_model_in_curriculum(load_replay_buffer=True)

        reward_threshold = rl_for_multiff_utils.calculate_reward_threshold_for_curriculum_training(
            self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
        self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                              reward_threshold_to_stop_on=reward_threshold, dir_name=self.best_model_postcurriculum_dir)
        print('reward_threshold:', reward_threshold)

        self.load_best_model_postcurriculum(load_replay_buffer=True)

    def regular_training(self, num_train_episodes=10000, eval_eps_freq=15, num_eval_episodes=2,
                         print_episode_reward=True, reward_threshold_to_stop_on=None, dir_name=None):
        max_steps_per_eps = self.env.episode_len

        self.sac_model, self.best_avg_reward_record, self.alpha_df = self.train_LSTM_agent(self.env,
                                                                                           num_train_episodes=num_train_episodes, eval_eps_freq=eval_eps_freq, max_steps_per_eps=max_steps_per_eps,
                                                                                           num_eval_episodes=num_eval_episodes, print_episode_reward=print_episode_reward,
                                                                                           reward_threshold_to_stop_on=reward_threshold_to_stop_on, dir_name=dir_name)
        # Write/refresh checkpoint manifest if a training dir is provided
        if dir_name is not None:
            self.write_checkpoint_manifest(dir_name)
        return self.best_avg_reward_record

    def test_agent(self, num_test_episode=5, max_steps_per_eps=512, deterministic=True):
        self.load_latest_agent(load_replay_buffer=False)
        reward_rate = LSTM_functions.evaluate_agent(self.env_for_test, self.sac_model, max_steps_per_eps, num_test_episode,
                                                    deterministic=deterministic)
        print("Average reward per episode: ", reward_rate)
        return reward_rate

    def save_agent(self, whether_save_replay_buffer=True, dir_name=None):

        if dir_name is None:
            dir_name = self.model_folder_name
        LSTM_functions.save_best_model(
            self.sac_model, whether_save_replay_buffer=whether_save_replay_buffer, dir_name=dir_name)
        # Write checkpoint manifest similar to SB3
        self.write_checkpoint_manifest(dir_name)

    def write_checkpoint_manifest(self, dir_name):
        rl_for_multiff_utils.write_checkpoint(dir_name, {
            'algorithm': 'lstm_sac',
            'model_files': ['lstm_q1', 'lstm_q2', 'lstm_policy'],
            'replay_buffer': 'buffer.pkl',
            'env_params': self.current_env_kwargs,
        })

    def load_agent(self, load_replay_buffer=True, dir_name=None):
        manifest = rl_for_multiff_utils.read_checkpoint_manifest(dir_name)
        self.make_env(**manifest['env_params'])
        self.make_agent()
        self.sac_model.load_model(dir_name)

        if load_replay_buffer:
            buffer_name = 'buffer.pkl'
            if isinstance(manifest, dict):
                buffer_name = manifest.get('replay_buffer', buffer_name)
            buffer_path = os.path.join(dir_name, buffer_name)
            with open(buffer_path, 'rb') as f:
                loaded_buffer_list = CPU_Unpickler(f).load()
            # Reconstruct ReplayBufferLSTM2 from saved list
            if isinstance(loaded_buffer_list, list):
                capacity = max(self.agent_params.get(
                    'replay_buffer_size', 500), len(loaded_buffer_list))
                rb = LSTM_functions.ReplayBufferLSTM2(capacity)
                rb.buffer = loaded_buffer_list
                rb.position = len(loaded_buffer_list) % capacity
                self.replay_buffer = rb
            else:
                self.replay_buffer = loaded_buffer_list

            self.sac_model.replay_buffer = self.replay_buffer
            print(
                f'length of replay buffer: {len(self.sac_model.replay_buffer)}')

        print("Loaded existing agent:", dir_name)
        self.loaded_agent_dir = dir_name
        return

    def load_best_model_postcurriculum(self, load_replay_buffer=True):
        # load_agent will recreate env and agent using saved manifest
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=self.best_model_postcurriculum_dir)

    def load_best_model_in_curriculum(self, load_replay_buffer=True):
        # load_agent will recreate env and agent using saved manifest
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=self.best_model_in_curriculum_dir)

    def make_animation(self, **kwargs):
        super().make_animation(max_num_frames=None, **kwargs)

    def train_LSTM_agent(self, env,
                         num_train_episodes=10000, eval_eps_freq=15, max_steps_per_eps=512, num_eval_episodes=2,
                         print_episode_reward=False,
                         reward_threshold_to_stop_on=None, dir_name=None):

        # env params always read from self.current_env_kwargs when writing manifest

        self.sac_model.soft_q_net1.train()
        self.sac_model.soft_q_net2.train()
        self.sac_model.policy_net.train()

        self.eval_rewards = []
        self.best_avg_reward = -9999
        self.best_avg_reward_record = -9999

        self.list_of_epi_for_alpha = []
        self.list_of_alpha = []
        self.list_of_epi_rewards = []
        for eps in range(num_train_episodes):
            episode_reward = LSTM_functions._train_episode(
                env, self.sac_model, max_steps_per_eps)
            try:
                print('ALPHA (entropy-related): ', self.sac_model.alpha)
                self.list_of_alpha.append(self.sac_model.alpha.item())
                self.list_of_epi_for_alpha.append(eps)
                self.list_of_epi_rewards.append(episode_reward)
                LSTM_functions.print_last_n_alphas(self.list_of_alpha, n=10)

            except AttributeError:
                pass

            if eps % eval_eps_freq == 0 and eps > 0:
                LSTM_functions.print_last_n_alphas(self.list_of_alpha, n=100)
                avg_reward = LSTM_functions.evaluate_agent(
                    env, self.sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True)
                print(
                    f"Best average reward: {self.best_avg_reward}, Current average reward: {avg_reward}")

                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    if self.best_avg_reward > self.best_avg_reward_record:
                        self.best_avg_reward_record = self.best_avg_reward
                    if dir_name is not None:
                        LSTM_functions.save_best_model(
                            self.sac_model, dir_name=dir_name)
                        self.write_checkpoint_manifest(
                            dir_name)
                    print(f"Best average reward = {self.best_avg_reward}")
                    print(f"Best model saved at episode {eps} to {dir_name}")

                    if reward_threshold_to_stop_on is not None:
                        if self.best_avg_reward_record >= reward_threshold_to_stop_on:
                            break

                self.eval_rewards.append(avg_reward)
                print('Last 10 evaluation rewards:', self.eval_rewards[-10:])
                # plot_eval_rewards(eval_rewards)
                # plot_alpha(self.list_of_epi_for_alpha, self.list_of_alpha)
                if len(self.eval_rewards) > 100:
                    self.eval_rewards = self.eval_rewards[-100:]

            if print_episode_reward:
                print(f'Episode: {eps}, Episode Reward: {episode_reward}')
                print('============================================================')

            alpha_dict = {'epi': self.list_of_epi_for_alpha,
                          'alpha': self.list_of_alpha, 'rewards': self.list_of_epi_rewards}
            alpha_df = pd.DataFrame(alpha_dict)

        return self.sac_model, self.best_avg_reward_record, alpha_df


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
