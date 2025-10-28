

from reinforcement_learning.agents.rnn import env_for_rnn
from reinforcement_learning.base_classes import rl_base_utils, rl_base_class
from reinforcement_learning.agents.rnn import lstm_utils
from reinforcement_learning.agents.rnn import gru_utils
from reinforcement_learning.base_classes import env_utils
from reinforcement_learning.base_classes import run_logger

import os
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
import numpy as np
import pickle
import io
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LSTMforMultifirefly(rl_base_class._RLforMultifirefly):

    def __init__(self,
                 overall_folder='multiff_analysis/RL_models/LSTM_stored_models/all_agents/gen_0/',
                 model_folder_name=None,
                 add_date_to_model_folder_name=False,
                 max_in_memory_time=1,
                 seq_len=192,
                 burn_in=64,
                 **additional_env_kwargs):

        super().__init__(overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         model_folder_name=model_folder_name,
                         max_in_memory_time=max_in_memory_time,
                         **additional_env_kwargs)

        self.agent_type = 'lstm'
        self.algorithm_name = 'lstm_sac'
        self.model_files = ['lstm_q1', 'lstm_q2', 'lstm_policy']

        self.replay_buffer_class = lstm_utils.ReplayBufferLSTM2
        self.trainer_class = lstm_utils.LSTM_SAC_Trainer
        self.env_class = env_for_rnn.EnvForRNN

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }
        
        self.seq_len = seq_len
        self.burn_in = burn_in

    def prepare_agent_params(self, agent_params_already_set_ok, **kwargs):

        # Ensure environment exists before constructing agent-dependent params
        if not hasattr(self, 'env'):
            self.make_env(**self.input_env_kwargs)

        if agent_params_already_set_ok and (self.agent_params is not None):
            existing_agent_params = self.agent_params.copy()
        else:
            existing_agent_params = {}

        self.agent_params = {
            "gamma": 0.995,
            "state_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "action_dim": self.env.action_space.shape[0],
            "action_range": 1.0,
            "replay_buffer_size": kwargs.get('replay_buffer_size', 1000),
            "hidden_dim": kwargs.get('hidden_dim', 256),
            "soft_q_lr": kwargs.get('soft_q_lr', 0.0015),
            "policy_lr": kwargs.get('policy_lr', 0.003),
            "alpha_lr": kwargs.get('alpha_lr', 0.002),
            "seq_len": kwargs.get('seq_len', self.seq_len),
            "burn_in": kwargs.get('burn_in', self.burn_in),
            "batch_size": kwargs.get('batch_size', 8),
            "update_itr": kwargs.get('update_itr', 1),
            "reward_scale": kwargs.get('reward_scale', 0.5),
            "target_entropy": kwargs.get('target_entropy', - self.env.action_space.shape[0]),
            "soft_tau": kwargs.get('soft_tau', 0.015),
            "train_freq": kwargs.get('train_freq', 10),
            "auto_entropy": kwargs.get('auto_entropy', True),
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        }

        self.agent_params.update(existing_agent_params)
        self.agent_params.update(kwargs)

    def make_agent(self,
                   agent_params_already_set_ok=True,
                   **kwargs):

        self.prepare_agent_params(agent_params_already_set_ok, **kwargs)

        self.replay_buffer = self.replay_buffer_class(
            self.agent_params['replay_buffer_size'])
        self.agent_params['replay_buffer'] = self.replay_buffer
        self.sac_model = self.trainer_class(**self.agent_params)

    def make_initial_env_for_curriculum_training(self, initial_flash_on_interval=3, initial_angular_terminal_vel=0.32, initial_reward_boundary=75):
        self.make_env(**self.input_env_kwargs)
        self._make_initial_env_for_curriculum_training(initial_angular_terminal_vel=initial_angular_terminal_vel,
                                                       initial_flash_on_interval=initial_flash_on_interval,
                                                       initial_reward_boundary=initial_reward_boundary)

    def _make_agent_for_curriculum_training(self):
        self.make_agent()

    def _use_while_loop_for_curriculum_training(self, eval_eps_freq=20, num_eval_episodes=2):
        stage = 0
        finished_curriculum = False
        log_path = os.path.join(self.best_model_in_curriculum_dir, 'curriculum_log.csv')

        # Prepare empty log DataFrame
        columns = [
            'stage', 'reward_threshold', 'best_avg_reward',
            'flash_on_interval', 'angular_terminal_vel', 'reward_boundary',
            'distance2center_cost', 'stop_vel_cost',
            'dv_cost_factor', 'dw_cost_factor', 'w_cost_factor',
            'finished_curriculum'
        ]
        # Ensure parent directory exists before attempting to write
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            pd.DataFrame(columns=columns).to_csv(log_path, index=False)

        while True:
            stage += 1
            gc.collect()

            # --- Compute dynamic reward threshold ---
            reward_threshold = rl_base_utils.calculate_reward_threshold_for_curriculum_training(
                self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
            print(f'[Stage {stage}] Current reward threshold: {reward_threshold:.2f}')

            # --- Train under current environment conditions ---
            self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                                reward_threshold_to_stop_on=reward_threshold,
                                dir_name=self.best_model_in_curriculum_dir)

            # --- Check progress ---
            if self.best_avg_reward < reward_threshold:
                print(f'[Stage {stage}] Warning: best reward {self.best_avg_reward:.2f} < threshold {reward_threshold:.2f}. Retrying...')
                # Log failed attempt as a curriculum stage attempt
                failed_payload = {
                    'stage': stage,
                    'reward_threshold': reward_threshold,
                    'best_avg_reward': self.best_avg_reward,
                    'flash_on_interval': self.env.flash_on_interval,
                    'angular_terminal_vel': self.env.angular_terminal_vel,
                    'reward_boundary': self.env.reward_boundary,
                    'distance2center_cost': self.env.distance2center_cost,
                    'stop_vel_cost': self.env.stop_vel_cost,
                    'dv_cost_factor': self.env.dv_cost_factor,
                    'dw_cost_factor': self.env.dw_cost_factor,
                    'w_cost_factor': self.env.w_cost_factor,
                    'finished_curriculum': False,
                    'attempt_passed': False,
                }
                try:
                    # Append to per-agent CSV
                    pd.DataFrame([failed_payload]).to_csv(log_path, mode='a', header=False, index=False)
                except Exception:
                    pass
                try:
                    sweep_params = getattr(self, 'sweep_params', {})
                    run_logger.log_curriculum_stage(self.overall_folder, agent_type=getattr(self, 'agent_type', 'rnn'), sweep_params=sweep_params, stage_payload=failed_payload)
                except Exception as e:
                    print('[logger] failed to log failed curriculum attempt:', e)
                continue

            print(f'[Stage {stage}] Progressed: best reward {self.best_avg_reward:.2f} ≥ threshold {reward_threshold:.2f}')

            # --- Log this stage ---
            stage_payload = {
                'stage': stage,
                'reward_threshold': reward_threshold,
                'best_avg_reward': self.best_avg_reward,
                'flash_on_interval': self.env.flash_on_interval,
                'angular_terminal_vel': self.env.angular_terminal_vel,
                'reward_boundary': self.env.reward_boundary,
                'distance2center_cost': self.env.distance2center_cost,
                'stop_vel_cost': self.env.stop_vel_cost,
                'dv_cost_factor': self.env.dv_cost_factor,
                'dw_cost_factor': self.env.dw_cost_factor,
                'w_cost_factor': self.env.w_cost_factor,
                'finished_curriculum': finished_curriculum,
                'attempt_passed': True,
            }
            # Per-agent CSV for backward compatibility
            try:
                pd.DataFrame([stage_payload]).to_csv(log_path, mode='a', header=False, index=False)
            except Exception:
                pass
            # Aggregate curriculum log across runs using common logger
            try:
                sweep_params = getattr(self, 'sweep_params', {})
                run_logger.log_curriculum_stage(self.overall_folder, agent_type=getattr(self, 'agent_type', 'rnn'), sweep_params=sweep_params, stage_payload=stage_payload)
            except Exception as e:
                print('[logger] failed to log curriculum stage:', e)

            # --- Exit control ---
            if finished_curriculum:
                print(f'[Stage {stage}] Completed final training with target environment. Exiting curriculum loop.')
                break

            # --- Update environment for next stage ---
            self._update_env_after_meeting_reward_threshold()

            if all([
                self.env.flash_on_interval == self.input_env_kwargs['flash_on_interval'],
                self.env.angular_terminal_vel == self.input_env_kwargs['angular_terminal_vel'],
                self.env.reward_boundary == self.input_env_kwargs['reward_boundary'],
                self.env.distance2center_cost == self.input_env_kwargs['distance2center_cost'],
                self.env.stop_vel_cost == self.input_env_kwargs['stop_vel_cost'],
                self.env.dv_cost_factor == self.input_env_kwargs['dv_cost_factor'],
                self.env.dw_cost_factor == self.input_env_kwargs['dw_cost_factor'],
                self.env.w_cost_factor == self.input_env_kwargs['w_cost_factor']
            ]):
                print(f'[Stage {stage}] All curriculum conditions met — running one more stage before exit.')
                finished_curriculum = True

        # --- Final post-curriculum training ---
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        self.make_env(**self.input_env_kwargs)
        self.load_best_model_in_curriculum(load_replay_buffer=False)

        reward_threshold = rl_base_utils.calculate_reward_threshold_for_curriculum_training(
            self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)

        self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                            reward_threshold_to_stop_on=reward_threshold,
                            dir_name=self.best_model_postcurriculum_dir)

        if self.best_avg_reward >= reward_threshold:
            print('Finished curriculum training. Reached final post-curriculum reward_threshold:', reward_threshold)
        else:
            print('After curriculum training, failed to reach final post-curriculum reward_threshold:', reward_threshold)
        self.load_best_model_postcurriculum(load_replay_buffer=True)


    def save_agent(self, whether_save_replay_buffer=True, dir_name=None):

        if dir_name is None:
            dir_name = self.model_folder_name
        lstm_utils.save_best_model(
            self.sac_model, whether_save_replay_buffer=whether_save_replay_buffer, dir_name=dir_name)
        # Write checkpoint manifest similar to SB3
        self.write_checkpoint_manifest(dir_name)

    def write_checkpoint_manifest(self, dir_name):
        rl_base_utils.write_checkpoint(dir_name, {
            'algorithm': self.algorithm_name,
            'model_files': self.model_files,
            'replay_buffer': 'buffer.pkl',
            'env_params': self.current_env_kwargs,
        })

    def load_agent(self, load_replay_buffer=True, dir_name=None):
        try:
            manifest = rl_base_utils.read_checkpoint_manifest(dir_name)
        except Exception as e:
            raise ValueError(
                f"Failed to load agent from {dir_name} because of failure to load checkpoint manifest")
        env_params = None
        if isinstance(manifest, dict):
            env_params = manifest.get('env_params')
        if not isinstance(env_params, dict):
            # fallback: use current or default input env kwargs
            env_params = getattr(self, 'current_env_kwargs', None)
            if not isinstance(env_params, dict):
                env_params = self.input_env_kwargs
        self.make_env(**env_params)
        self.make_agent()
        self.sac_model.load_model(dir_name)

        print("Loaded existing agent:", dir_name)
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
                    'replay_buffer_size', 1000), len(loaded_buffer_list))
                rb = self.replay_buffer_class(capacity)
                rb.buffer = loaded_buffer_list
                rb.position = len(loaded_buffer_list) % capacity
                self.replay_buffer = rb
            else:
                self.replay_buffer = loaded_buffer_list

            self.sac_model.replay_buffer = self.replay_buffer
            print(
                f'Loaded replay buffer with length {len(self.sac_model.replay_buffer)}')

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

    def regular_training(self, num_train_episodes=10000, eval_eps_freq=15, num_eval_episodes=2,
                         print_episode_reward=True, reward_threshold_to_stop_on=None, dir_name=None,
                         min_train_episodes_before_early_stop=0):
        max_steps_per_eps = self.env.episode_len

        self.train_rnn_agent(
            self.env,
            num_train_episodes=num_train_episodes,
            eval_eps_freq=eval_eps_freq,
            max_steps_per_eps=max_steps_per_eps,
            num_eval_episodes=num_eval_episodes,
            print_episode_reward=print_episode_reward,
            reward_threshold_to_stop_on=reward_threshold_to_stop_on,
            dir_name=dir_name,
            min_train_episodes_before_early_stop=min_train_episodes_before_early_stop,
        )
        if dir_name is not None:
            self.write_checkpoint_manifest(dir_name)
        return self.best_avg_reward

    def _common_train_loop(self, env, train_episode_fn, eval_agent_fn,
                           num_train_episodes=10000, eval_eps_freq=15, max_steps_per_eps=512,
                           num_eval_episodes=2, print_episode_reward=False,
                           reward_threshold_to_stop_on=None, dir_name=None,
                           track_alpha=False, save_fn=None,
                           min_train_episodes_before_early_stop=0):
        """
        Shared training loop for RNN agents (LSTM/GRU variants).
        """

        # Put models in train mode
        self.sac_model.soft_q_net1.train()
        self.sac_model.soft_q_net2.train()
        self.sac_model.policy_net.train()

        self.eval_rewards = []
        self.best_avg_reward = -9999

        self.list_of_epi_rewards = []
        if track_alpha:
            self.list_of_epi_for_alpha = []
            self.list_of_alpha = []

        for eps in range(num_train_episodes):
            episode_reward = train_episode_fn(
                env, self.sac_model, max_steps_per_eps)
            self.list_of_epi_rewards.append(episode_reward)

            if track_alpha:
                try:
                    print('ALPHA (entropy-related): ', self.sac_model.alpha)
                    self.list_of_alpha.append(self.sac_model.alpha.item())
                    self.list_of_epi_for_alpha.append(eps)
                    lstm_utils.print_last_n_alphas(self.list_of_alpha, n=10)
                except AttributeError:
                    pass

            # Evaluation phase
            if eps % eval_eps_freq == 0 and eps > 0:
                print('Evaluating agent...')
                if track_alpha:
                    lstm_utils.print_last_n_alphas(self.list_of_alpha, n=100)
                avg_reward = eval_agent_fn(
                    env, self.sac_model, max_steps_per_eps, num_eval_episodes, deterministic=False)
                print(f"Current average evaluation reward: {avg_reward}")
                print(
                    f"Best average evaluation reward: {self.best_avg_reward}")

                self.eval_rewards.append(avg_reward)
                print('Last 10 evaluation rewards (most recent first):', self.eval_rewards[-1:-11:-1])
                if len(self.eval_rewards) > 100:
                    self.eval_rewards = self.eval_rewards[-100:]

                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    if dir_name is not None:
                        if save_fn is not None:
                            save_fn(self.sac_model, dir_name)
                        self.write_checkpoint_manifest(dir_name)
                        print(f"New best model saved to {dir_name}")
                    print(f"New best average reward: {self.best_avg_reward}")

                if reward_threshold_to_stop_on is not None and \
                        avg_reward >= reward_threshold_to_stop_on:
                    if eps >= min_train_episodes_before_early_stop:
                        print(f"Reward threshold met by current evaluation: {avg_reward} >= {reward_threshold_to_stop_on} (eps={eps} ≥ min={min_train_episodes_before_early_stop})")
                        break
                    else:
                        print(f"Reward threshold met but minimum episodes not reached yet: eps={eps} < min={min_train_episodes_before_early_stop}. Continuing training.")

            if print_episode_reward:
                print(f'Episode: {eps}, Episode Reward: {episode_reward}')
                print('============================================================')

        if track_alpha:
            # Align rewards with the specific episodes where alpha was recorded
            # to avoid mismatched lengths when constructing the DataFrame.
            alpha_episodes = self.list_of_epi_for_alpha
            rewards_at_alpha_episodes = [
                self.list_of_epi_rewards[i] if i < len(
                    self.list_of_epi_rewards) else np.nan
                for i in alpha_episodes
            ]
            alpha_dict = {
                'epi': alpha_episodes,
                'alpha': self.list_of_alpha,
                'rewards': rewards_at_alpha_episodes
            }
            self.alpha_df = pd.DataFrame(alpha_dict)

    def train_rnn_agent(self, env, **kwargs):
        self._common_train_loop(
            env,
            train_episode_fn=lstm_utils._train_episode,
            eval_agent_fn=lstm_utils.evaluate_lstm_agent,
            save_fn=lambda model, dir_name: lstm_utils.save_best_model(
                model, dir_name=dir_name),
            track_alpha=True,
            **kwargs
        )


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
