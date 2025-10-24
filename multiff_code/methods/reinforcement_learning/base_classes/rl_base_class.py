from data_wrangling import general_utils, retrieve_raw_data, process_monkey_information
from pattern_discovery import organize_patterns_and_features, make_ff_dataframe
from visualization.matplotlib_tools import additional_plots, plot_statistics
from visualization.animation import animation_class, animation_utils
from reinforcement_learning.agents.rnn import env_for_rnn
from reinforcement_learning.agents.feedforward import env_for_sb3
from reinforcement_learning.collect_data import collect_agent_data, process_agent_data
from reinforcement_learning.agents.feedforward import interpret_neural_network, sb3_utils
from reinforcement_learning.base_classes import rl_base_utils
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from reinforcement_learning.base_classes import base_env
from reinforcement_learning.base_classes import env_utils


import time as time_package
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import time as time_package
import copy
import torch
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class _RLforMultifirefly(animation_class.AnimationClass):

    def __init__(self,
                 overall_folder=None,
                 model_folder_name=None,
                 dt=0.1,
                 dv_cost_factor=1,
                 dw_cost_factor=1,
                 w_cost_factor=1,
                 flash_on_interval=0.3,
                 max_in_memory_time=3,
                 add_date_to_model_folder_name=False,
                 reward_per_ff=100,
                 reward_boundary=25,
                 angular_terminal_vel=0.05,
                 distance2center_cost=0,
                 stop_vel_cost=50,
                 data_name='data_0',
                 std_anneal_preserve_fraction=0.05,
                 **additional_env_kwargs):

        self.player = "agent"
        self.agent_params = None
        self.overall_folder = overall_folder

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            base_env.MultiFF)
        self.additional_env_kwargs = additional_env_kwargs
        self.class_instance_env_kwargs = {'dt': dt,
                                         'dv_cost_factor': dv_cost_factor,
                                         'dw_cost_factor': dw_cost_factor,
                                         'w_cost_factor': w_cost_factor,
                                         'print_ff_capture_incidents': True,
                                         'print_episode_reward_rates': True,
                                         'max_in_memory_time': max_in_memory_time,
                                         'flash_on_interval': flash_on_interval,
                                         'angular_terminal_vel': angular_terminal_vel,
                                         'distance2center_cost': distance2center_cost,
                                         'stop_vel_cost': stop_vel_cost,
                                         'reward_boundary': reward_boundary,
                                         }

        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

        self.loaded_agent_name = ''

        self.agent_id = "dv" + str(dv_cost_factor) + \
                        "_dw" + str(dw_cost_factor) + "_w" + str(w_cost_factor) + \
                        "_memT" + \
            str(self.input_env_kwargs['max_in_memory_time'])

        if len(overall_folder) > 0:
            os.makedirs(self.overall_folder, exist_ok=True)

        self.model_folder_name = model_folder_name if model_folder_name is not None else os.path.join(
            self.overall_folder, self.agent_id)
        print('model_folder_name:', self.model_folder_name)
        
        self.std_anneal_preserve_fraction = std_anneal_preserve_fraction

        if add_date_to_model_folder_name:
            self.model_folder_name = self.model_folder_name + "_date" + \
                str(time_package.localtime().tm_mon) + "_" + \
                str(time_package.localtime().tm_mday)

        self.get_related_folder_names_from_model_folder_name(
            self.model_folder_name, data_name=data_name)

        # Per-agent best-after-curriculum directory under the agent folder
        self.best_model_postcurriculum_dir = os.path.join(
            self.model_folder_name, 'best_model_postcurriculum')
        # Per-agent best-during-curriculum directory
        self.best_model_in_curriculum_dir = os.path.join(
            self.model_folder_name, 'best_model_in_curriculum')

    def get_related_folder_names_from_model_folder_name(self, model_folder_name, data_name='data_0'):
        self.model_folder_name = model_folder_name
        self.processed_data_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/processed_data'), f'individual_data_sessions/{data_name}')
        self.planning_data_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/planning'), f'individual_data_sessions/{data_name}')
        self.patterns_and_features_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/patterns_and_features'), f'individual_data_sessions/{data_name}')
        self.decision_making_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/decision_making'), f'individual_data_sessions/{data_name}')

        os.makedirs(self.model_folder_name, exist_ok=True)
        os.makedirs(self.processed_data_folder_path, exist_ok=True)
        os.makedirs(self.planning_data_folder_path, exist_ok=True)
        os.makedirs(self.patterns_and_features_folder_path, exist_ok=True)
        os.makedirs(self.decision_making_folder_path, exist_ok=True)

    # removed resolve_best_model_postcurriculum_dir; callers should use
    # self.best_model_postcurriculum_dir and create dirs as needed

    def get_current_info_condition(self, df):
        minimal_current_info = self.get_minimum_current_info()

        current_info_condition = df.any(axis=1)
        for key, value in minimal_current_info.items():
            current_info_condition = current_info_condition & (
                df[key] == value)
        return current_info_condition

    def make_env(self, **env_kwargs):
        self.current_env_kwargs = copy.deepcopy(self.input_env_kwargs)
        self.current_env_kwargs.update(env_kwargs)
        self.env = self.env_class(**self.current_env_kwargs)
        print(f'Made env with the following kwargs: {env_kwargs}')

    def curriculum_training(self, best_model_in_curriculum_exists_ok=True, best_model_postcurriculum_exists_ok=True, load_replay_buffer_of_best_model_postcurriculum=True):
        if self.loaded_agent_name == 'model':
            self.regular_training()
            self.successful_training = True
            return
        elif self.loaded_agent_name == 'best_model_in_curriculum':
            self._progress_in_curriculum(best_model_in_curriculum_exists_ok)
            self.regular_training()
            self.successful_training = True
            return 
        elif self.loaded_agent_name == 'best_model_postcurriculum':
            self.regular_training()
            self.successful_training = True
            return
        else:
            if best_model_postcurriculum_exists_ok:
                try:
                    self.load_best_model_postcurriculum(
                        load_replay_buffer=load_replay_buffer_of_best_model_postcurriculum)
                    print('Loaded best_model_postcurriculum')
                except Exception:
                    print('Need to train a new best_model_postcurriculum')
                    self._progress_in_curriculum(best_model_in_curriculum_exists_ok)
            else:
                self._progress_in_curriculum(best_model_in_curriculum_exists_ok)
            self.regular_training()
            self.successful_training = True
            return


    def _progress_in_curriculum(self, best_model_in_curriculum_exists_ok=True):
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        self.original_agent_id = self.agent_id
        self.agent_id = 'no_cost'
        print('Starting curriculum training')
        if best_model_in_curriculum_exists_ok:
            try:
                if self.loaded_agent_name != 'best_model_in_curriculum':
                    self.curriculum_env_kwargs = rl_base_utils.read_checkpoint_manifest(
                        self.loaded_agent_dir)['env_params']
                    print('Loaded best_model_in_curriculum')
                    print(
                        f'Made env based on env params saved in {self.loaded_agent_dir}')
                    self.make_env(**self.curriculum_env_kwargs)
                    self.load_best_model_in_curriculum(load_replay_buffer=True)

            except Exception:
                print('Need to train a new best_model_in_curriculum')
                self.make_initial_env_for_curriculum_training()
                self._make_agent_for_curriculum_training()
        else:
            print('Making initial env and agent for curriculum training')
            self.make_initial_env_for_curriculum_training()
            self._make_agent_for_curriculum_training()
        self.successful_training = False
        self._use_while_loop_for_curriculum_training()
        self.streamline_making_animation(currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                                         video_dir=self.best_model_postcurriculum_dir)
        self.agent_id = self.original_agent_id

    def _make_initial_env_for_curriculum_training(self,
                                                  initial_flash_on_interval=3,
                                                  initial_angular_terminal_vel=0.32,
                                                  initial_distance2center_cost=2,
                                                  initial_stop_vel_cost=50,
                                                  initial_reward_boundary=75,
                                                  initial_dv_cost_factor=0.5,
                                                  initial_dw_cost_factor=0.5,
                                                  initial_w_cost_factor=0.5,
                                                  ):
        self.curriculum_env_kwargs = copy.deepcopy(
            self.input_env_kwargs)
        print('Made initial env for curriculum training')
        # Determine wrapped vs direct env from agent_type
        agent_type = getattr(self, 'agent_type', None)
        if agent_type is None:
            # backward: infer from presence of .env on env
            env = self.env.env if hasattr(self.env, 'env') else self.env
        elif str(agent_type).lower() in ('sb3', 'ff', 'feedforward'):
            env = self.env.env
        else:
            env = self.env

        env.flash_on_interval = initial_flash_on_interval
        env.angular_terminal_vel = initial_angular_terminal_vel
        env.reward_boundary = initial_reward_boundary
        env.distance2center_cost = initial_distance2center_cost
        env.stop_vel_cost = initial_stop_vel_cost
        env.dv_cost_factor = initial_dv_cost_factor
        env.dw_cost_factor = initial_dw_cost_factor
        env.w_cost_factor = initial_w_cost_factor

        self.curriculum_env_kwargs['flash_on_interval'] = initial_flash_on_interval
        self.curriculum_env_kwargs['angular_terminal_vel'] = initial_angular_terminal_vel
        self.curriculum_env_kwargs['reward_boundary'] = initial_reward_boundary
        self.curriculum_env_kwargs['distance2center_cost'] = initial_distance2center_cost
        self.curriculum_env_kwargs['stop_vel_cost'] = initial_stop_vel_cost
        self.curriculum_env_kwargs['dv_cost_factor'] = initial_dv_cost_factor
        self.curriculum_env_kwargs['dw_cost_factor'] = initial_dw_cost_factor
        self.curriculum_env_kwargs['w_cost_factor'] = initial_w_cost_factor

        self.current_env_kwargs = self.curriculum_env_kwargs

        if agent_type is None:
            if hasattr(self.env, 'env'):
                self.env.env = env
            else:
                self.env = env
        elif str(agent_type).lower() in ('sb3', 'ff', 'feedforward'):
            self.env.env = env
        else:
            self.env = env

    def _update_env_after_meeting_reward_threshold(self):
        
        print('Updating env after meeting reward threshold...')
        agent_type = getattr(self, 'agent_type', None)
        if agent_type is None:
            env = self.env.env if hasattr(self.env, 'env') else self.env
        elif str(agent_type).lower() in ('sb3', 'ff', 'feedforward'):
            env = self.env.env
        else:
            env = self.env
        
        # Snapshot values before update and targets
        before = {
            'flash_on_interval': env.flash_on_interval,
            'angular_terminal_vel': env.angular_terminal_vel,
            'distance2center_cost': env.distance2center_cost,
            'stop_vel_cost': env.stop_vel_cost,
            'reward_boundary': env.reward_boundary,
            'dv_cost_factor': env.dv_cost_factor,
            'dw_cost_factor': env.dw_cost_factor,
            'w_cost_factor': env.w_cost_factor,
        }
        targets = {
            'flash_on_interval': self.input_env_kwargs['flash_on_interval'],
            'angular_terminal_vel': self.input_env_kwargs['angular_terminal_vel'],
            'distance2center_cost': self.input_env_kwargs['distance2center_cost'],
            'stop_vel_cost': self.input_env_kwargs['stop_vel_cost'],
            'reward_boundary': self.input_env_kwargs['reward_boundary'],
            'dv_cost_factor': self.input_env_kwargs['dv_cost_factor'],
            'dw_cost_factor': self.input_env_kwargs['dw_cost_factor'],
            'w_cost_factor': self.input_env_kwargs['w_cost_factor'],
        }

        if env.reward_boundary > self.input_env_kwargs['reward_boundary']:
            env.reward_boundary = max(
                env.reward_boundary - 10, self.input_env_kwargs['reward_boundary'])
            self.curriculum_env_kwargs['reward_boundary'] = env.reward_boundary
            print('Updated reward_boundary to:', env.reward_boundary)
        elif env.distance2center_cost > self.input_env_kwargs['distance2center_cost']:
            env.distance2center_cost = max(
                env.distance2center_cost - 0.5, self.input_env_kwargs['distance2center_cost'])
            self.curriculum_env_kwargs['distance2center_cost'] = env.distance2center_cost
            print('Updated distance2center_cost to:', env.distance2center_cost)
        elif env.angular_terminal_vel > self.input_env_kwargs['angular_terminal_vel']:
            env.angular_terminal_vel = max(env.angular_terminal_vel/2, self.input_env_kwargs['angular_terminal_vel'])
            self.curriculum_env_kwargs['angular_terminal_vel'] = env.angular_terminal_vel
            print('Updated angular_terminal_vel to:', env.angular_terminal_vel)
        elif env.flash_on_interval > self.input_env_kwargs['flash_on_interval']:
            env.flash_on_interval = max(env.flash_on_interval - 0.3, self.input_env_kwargs['flash_on_interval'])
            self.curriculum_env_kwargs['flash_on_interval'] = env.flash_on_interval
            print('Updated flash_on_interval to:', env.flash_on_interval)
        elif env.stop_vel_cost > self.input_env_kwargs['stop_vel_cost']:
            env.stop_vel_cost = max(env.stop_vel_cost - 50,
                                    self.input_env_kwargs['stop_vel_cost'])
            self.curriculum_env_kwargs['stop_vel_cost'] = env.stop_vel_cost
            print('Updated stop_vel_cost to:', env.stop_vel_cost)
        elif env.dv_cost_factor < self.input_env_kwargs['dv_cost_factor']:
            env.dv_cost_factor = min(env.dv_cost_factor + 0.5, self.input_env_kwargs['dv_cost_factor'])
            self.curriculum_env_kwargs['dv_cost_factor'] = env.dv_cost_factor
            print('Updated dv_cost_factor to:', env.dv_cost_factor)
        elif env.dw_cost_factor < self.input_env_kwargs['dw_cost_factor']:
            env.dw_cost_factor = min(env.dw_cost_factor + 0.5, self.input_env_kwargs['dw_cost_factor'])
            self.curriculum_env_kwargs['dw_cost_factor'] = env.dw_cost_factor
            print('Updated dw_cost_factor to:', env.dw_cost_factor)
        elif env.w_cost_factor < self.input_env_kwargs['w_cost_factor']:
            env.w_cost_factor = min(env.w_cost_factor + 0.5, self.input_env_kwargs['w_cost_factor'])
            self.curriculum_env_kwargs['w_cost_factor'] = env.w_cost_factor
        
        # Snapshot after update
        after = {
            'flash_on_interval': env.flash_on_interval,
            'angular_terminal_vel': env.angular_terminal_vel,
            'distance2center_cost': env.distance2center_cost,
            'stop_vel_cost': env.stop_vel_cost,
            'reward_boundary': env.reward_boundary,
            'dv_cost_factor': env.dv_cost_factor,
            'dw_cost_factor': env.dw_cost_factor,
            'w_cost_factor': env.w_cost_factor,
        }
            
        # Reset or partially reset policy std-anneal progress after curriculum env change
        if hasattr(self, 'sac_model') and hasattr(self.sac_model, 'policy_net'):
            try:
                current = getattr(self.sac_model.policy_net, 'anneal_step', 0)
                setattr(self.sac_model.policy_net, 'anneal_step', int(max(0, int(current * self.std_anneal_preserve_fraction))))
            except Exception as e:
                print('Warning: failed to reset std-anneal progress:', e)

        # Softly reset SAC temperature (alpha) for auto-entropy after curriculum env change
        if hasattr(self, 'sac_model') and hasattr(self.sac_model, 'log_alpha'):
            try:
                with torch.no_grad():
                    alpha_reset_beta = getattr(self, 'alpha_reset_beta', 0.6)
                    current_log_alpha = self.sac_model.log_alpha
                    target_log_alpha = torch.zeros_like(current_log_alpha)
                    alpha_before = getattr(self.sac_model, 'alpha', current_log_alpha.exp())
                    new_log_alpha = alpha_reset_beta * current_log_alpha + (1 - alpha_reset_beta) * target_log_alpha
                    self.sac_model.log_alpha.copy_(new_log_alpha)
                    if hasattr(self.sac_model, 'alpha'):
                        self.sac_model.alpha = self.sac_model.log_alpha.exp()
                    alpha_after = getattr(self.sac_model, 'alpha', self.sac_model.log_alpha.exp())
                # Clear gradients/state so temperature restarts cleanly
                if getattr(self.sac_model, 'alpha_optimizer', None) is not None:
                    try:
                        self.sac_model.alpha_optimizer.zero_grad(set_to_none=True)
                    except TypeError:
                        self.sac_model.alpha_optimizer.zero_grad()
                if getattr(self.sac_model.log_alpha, 'grad', None) is not None:
                    self.sac_model.log_alpha.grad = None
            except Exception as e:
                print('Warning: failed to reset entropy temperature (alpha):', e)

        # Emit a compact stage summary dict
        stage_summary = {'before': before, 'after': after, 'targets': targets}
        try:
            stage_summary['alpha_reset_beta'] = getattr(self, 'alpha_reset_beta', 0.6)
            if 'alpha_before' in locals():
                stage_summary['alpha_before'] = float(alpha_before.detach().cpu().mean())
            if 'alpha_after' in locals():
                stage_summary['alpha_after'] = float(alpha_after.detach().cpu().mean())
            if hasattr(self, 'sac_model') and hasattr(self.sac_model, 'policy_net'):
                stage_summary['policy_anneal_step'] = int(getattr(self.sac_model.policy_net, 'anneal_step', 0))
        except Exception:
            pass
        print('Stage summary:', stage_summary)

        if agent_type is None:
            if hasattr(self.env, 'env'):
                self.env.env = env
            else:
                self.env = env
        elif str(agent_type).lower() in ('sb3', 'ff', 'feedforward'):
            self.env.env = env
        else:
            self.env = env

        self.current_env_kwargs = self.curriculum_env_kwargs

    def collect_data(self, n_steps=8000, exists_ok=False, save_data=False):

        if exists_ok:
            try:
                self.retrieve_monkey_data()
                self.ff_caught_T_new = self.ff_caught_T_sorted
                self.make_or_retrieve_ff_dataframe_for_agent(
                    exists_ok=exists_ok, save_data=save_data)

            except Exception as e:
                print(
                    "Failed to retrieve monkey data. Will make new monkey data. Error: ", e)
                self.run_agent_to_collect_data(
                    n_steps=n_steps, save_data=save_data)
        else:
            self.run_agent_to_collect_data(
                n_steps=n_steps, save_data=save_data)

        self.make_or_retrieve_closest_stop_to_capture_df(exists_ok=exists_ok)
        # self.calculate_pattern_frequencies_and_feature_statistics()
        # self.find_patterns()

    def run_agent_to_collect_data(self, n_steps=8000, save_data=False):

        if not hasattr(self, 'current_env_kwargs'):
            self.current_env_kwargs = copy.deepcopy(self.input_env_kwargs)

        env_data_collection_kwargs = copy.deepcopy(self.current_env_kwargs)
        env_data_collection_kwargs.update({'episode_len': n_steps+100})

        agent_type = getattr(self, 'agent_type', None)
        at = str(agent_type).lower() if agent_type is not None else None
        if at in ('lstm', 'gru', 'rnn'):
            self.env_for_data_collection = env_for_rnn.CollectInformationLSTM(
                **env_data_collection_kwargs)
            LSTM = True
        else:
            self.env_for_data_collection = env_for_sb3.CollectInformation(
                **env_data_collection_kwargs)
            LSTM = False

        self._run_agent_to_collect_data(
            n_steps=n_steps, save_data=save_data, LSTM=LSTM)

    def _run_agent_to_collect_data(self, exists_ok=False, n_steps=8000, save_data=False, LSTM=False):

        # first, make self.processed_data_folder_path empty
        print('Collecting new agent data......')
        if os.path.exists(self.processed_data_folder_path):
            if not exists_ok:
                # if the folder is not empty, remove all files in the folder
                if len(os.listdir(self.processed_data_folder_path)) > 0:
                    # make the folder empty
                    os.system(
                        'rm -rf ' + self.processed_data_folder_path + '/*')
                    print('Removed all files in the folder:',
                          self.processed_data_folder_path)
            # also remove all derived data
            process_agent_data.remove_all_data_derived_from_current_agent_data(
                self.processed_data_folder_path)

        self.n_steps = n_steps

        self.monkey_information, self.ff_flash_sorted, self.ff_caught_T_sorted, self.ff_believed_position_sorted, \
            self.ff_real_position_sorted, self.ff_life_sorted, self.ff_flash_end_sorted, self.caught_ff_num, self.total_ff_num, \
            self.obs_ff_indices_in_ff_dataframe, self.sorted_indices_all, self.ff_in_obs_df \
            = collect_agent_data.collect_agent_data_func(self.env_for_data_collection, self.sac_model, n_steps=self.n_steps, agent_type=self.agent_type)
        self.ff_index_sorted = np.arange(len(self.ff_life_sorted))
        self.eval_ff_capture_rate = len(
            self.ff_flash_end_sorted)/self.monkey_information['time'].max()

        self.ff_caught_T_new = self.ff_caught_T_sorted

        if save_data:
            self.save_ff_info_into_npz()
            self.monkey_information_path = os.path.join(
                self.processed_data_folder_path, 'monkey_information.csv')
            self.monkey_information.to_csv(self.monkey_information_path)
            print("saved monkey_information and ff info at",
                  (self.processed_data_folder_path))
        self.make_or_retrieve_ff_dataframe_for_agent(
            exists_ok=False, save_data=save_data)

        return

    def retrieve_monkey_data(self, speed_threshold_for_distinct_stop=1):
        self.npz_file_pathway = os.path.join(
            self.processed_data_folder_path, 'ff_basic_info.npz')
        self.ff_caught_T_sorted, self.ff_index_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.ff_life_sorted, \
            self.ff_flash_end_sorted = retrieve_raw_data._retrieve_ff_info_in_npz_from_txt_data(
                self.processed_data_folder_path)
        self.ff_flash_sorted = retrieve_raw_data._retrieve_ff_flash_sorted_in_npz_from_txt_data(
            self.processed_data_folder_path)

        self.monkey_information_path = os.path.join(
            self.processed_data_folder_path, 'monkey_information.csv')
        self.monkey_information = pd.read_csv(
            self.monkey_information_path).drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
        self.monkey_information = process_monkey_information._process_monkey_information_after_retrieval(
            self.monkey_information, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)

        self.make_or_retrieve_closest_stop_to_capture_df()
        self.make_ff_caught_T_new()

        return

    def save_ff_info_into_npz(self):
        # save ff info
        npz_file = os.path.join(os.path.join(
            self.processed_data_folder_path, 'ff_basic_info.npz'))

        np.savez(npz_file,
                 ff_life_sorted=self.ff_life_sorted,
                 ff_caught_T_sorted=self.ff_caught_T_sorted,
                 ff_index_sorted=self.ff_index_sorted,
                 ff_real_position_sorted=self.ff_real_position_sorted,
                 ff_believed_position_sorted=self.ff_believed_position_sorted,
                 ff_flash_end_sorted=self.ff_flash_end_sorted)

        # also save ff_flash_sorted
        npz_flash = os.path.join(os.path.join(
            self.processed_data_folder_path, 'ff_flash_sorted.npz'))
        np.savez(npz_flash, *self.ff_flash_sorted)
        return

    def make_or_retrieve_ff_dataframe_for_agent(self, exists_ok=False, save_data=False):
        # self.ff_dataframe = None
        self.ff_dataframe_path = os.path.join(os.path.join(
            self.processed_data_folder_path, 'ff_dataframe.csv'))

        if exists_ok & exists(self.ff_dataframe_path):
            self.ff_dataframe = pd.read_csv(self.ff_dataframe_path).drop(
                columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
        else:
            print('Warnings: currently, only ff in obs at each step are used in ff_dataframe. All ff are labeled \'visible\' regardless of their actual time since last visible.')
            if str(getattr(self, 'agent_type', 'sb3')).lower() in ('lstm', 'gru', 'rnn'):
                print('It is possible that an RNN agent has memory of past ff; code may need updates to reflect that. For planning analysis, info of in-memory ff is not needed.')

            self.make_ff_dataframe_from_ff_in_obs_df()
            # base_processing_class.BaseProcessing.make_or_retrieve_ff_dataframe(self, exists_ok=False, save_into_h5=False)
            print("made ff_dataframe")

            if save_data:
                self.ff_dataframe.to_csv(self.ff_dataframe_path)
                print("saved ff_dataframe at", self.ff_dataframe_path)
        return

    def make_ff_dataframe_from_ff_in_obs_df(self):
        self.ff_dataframe = self.ff_in_obs_df.copy()
        # self.ff_dataframe['visible'] = 1

        make_ff_dataframe.add_essential_columns_to_ff_dataframe(
            self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted)
        self.ff_dataframe = make_ff_dataframe.process_ff_dataframe(
            self.ff_dataframe, max_distance=None, max_time_since_last_vis=3)

    def load_latest_agent(self, load_replay_buffer=True, dir_name=None):
        # model_name is not really used here, but put here to be consistent with the SB3 version
        if dir_name is None:
            dir_name = self.model_folder_name

        # Try current directory first; if it's a curriculum subdir, fall back to agent root
        candidates = [dir_name]
        candidate_names = ['model']
        for best_model_dir in ['best_model_postcurriculum', 'best_model_in_curriculum']:
            best_model_path = os.path.join(dir_name, best_model_dir)
            candidates.append(best_model_path)
            candidate_names.append(best_model_dir)

        last_error = None
        self.loaded_agent_dir = None
        for d, name in zip(candidates, candidate_names):
            try:
                self.load_agent(
                    load_replay_buffer=load_replay_buffer, dir_name=d)
                self.loaded_agent_name = name
                if name == 'best_model_in_curriculum':
                    self.curriculum_env_kwargs = self.current_env_kwargs.copy()
                return
            except Exception as e:
                last_error = (dir_name, e)
        if last_error is not None:
            d, e = last_error
            raise ValueError(
                f"There was an error retrieving agent or replay_buffer in {d}. Error message {e}")

    def streamline_getting_data_from_agent(self, n_steps=8000, exists_ok=False, save_data=False, load_replay_buffer=False):
        if exists_ok:
            try:
                self.retrieve_monkey_data()
                self.make_or_retrieve_ff_dataframe_for_agent(
                    exists_ok=exists_ok, save_data=save_data)
                return
            except Exception as e:
                print(
                    "Failed to retrieve monkey data. Will make new monkey data. Error: ", e)
        self.load_latest_agent(load_replay_buffer=load_replay_buffer)
        self.collect_data(
            n_steps=n_steps, exists_ok=exists_ok, save_data=save_data)

    def streamline_loading_and_making_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000):
        try:
            self.env
        except AttributeError:
            self.make_env(**self.input_env_kwargs)

        try:
            self.sac_model
        except AttributeError:
            self.make_agent()
        self.load_latest_agent(load_replay_buffer=False)
        self.streamline_making_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation,
                                         duration=duration, n_steps=n_steps, file_name=None)

    def streamline_making_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000, file_name=None, video_dir=None):
        self.collect_data(n_steps=n_steps)
        # if len(self.ff_caught_T_new) >= currentTrial_for_animation:
        self.make_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation,
                            duration=duration, file_name=file_name, video_dir=video_dir)

    def make_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], file_name=None, video_dir=None, max_num_frames=150):
        self.set_animation_parameters(currentTrial=currentTrial_for_animation, num_trials=num_trials_for_animation,
                                      k=1, duration=duration, max_num_frames=max_num_frames)
        self.call_animation_function(file_name=file_name, video_dir=video_dir)

    def streamline_everything(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                              use_curriculum_training=True, load_replay_buffer_of_best_model_postcurriculum=True,
                              best_model_in_curriculum_exists_ok=True,
                              best_model_postcurriculum_exists_ok=True,
                              to_load_latest_agent=True,
                              load_replay_buffer=True,
                              to_train_agent=True):

        self.family_of_agents_log = rl_base_utils.retrieve_or_make_family_of_agents_log(
            self.overall_folder)
        # to_load_latest_agent, to_train_agent = self.check_with_family_of_agents_log()
        # if (not to_load_latest_agent) & (not to_train_agent):
        #     print("The set of parameters has failed to produce a well-trained agent in the past. \
        #            Skip to the next set of parameters")
        #     return

        self.use_curriculum_training = use_curriculum_training

        if to_load_latest_agent:
            try:
                self.load_latest_agent(load_replay_buffer=load_replay_buffer)
            except Exception as e:
                print(
                    "Failed to load existing agent. Need to train a new agent. Error: ", e)
        else:
            print('Making new env based on input_env_kwargs')
            self.make_env(**self.input_env_kwargs)
            self.make_agent()

        if to_train_agent:
            self.train_agent(use_curriculum_training=use_curriculum_training,
                             best_model_in_curriculum_exists_ok=best_model_in_curriculum_exists_ok,
                             best_model_postcurriculum_exists_ok=best_model_postcurriculum_exists_ok,
                             load_replay_buffer_of_best_model_postcurriculum=load_replay_buffer_of_best_model_postcurriculum)
            if not self.successful_training:
                print("The set of parameters has failed to produce a well-trained agent in the past. \
                    Skip to the next set of parameters")
                return

            self.streamline_loading_and_making_animation(currentTrial_for_animation=currentTrial_for_animation, duration=duration,
                                                         num_trials_for_animation=num_trials_for_animation, n_steps=n_steps)

        # to_update_record, to_make_plots = self.whether_to_update_record_and_make_plots()
        # if to_make_plots or to_update_record:
        #     try:
        #         self._evaluate_model_and_retrain_if_necessary()
        #     except ValueError as e:
        #         return

        #     if to_make_plots:
        #         self._make_plots_for_the_model(currentTrial_for_animation, num_trials_for_animation)
        # else:
        #     print("Plots and record already exist. No need to make new ones.")

        return

    def train_agent(self, use_curriculum_training=True, best_model_in_curriculum_exists_ok=True,
                    best_model_postcurriculum_exists_ok=True,
                    load_replay_buffer_of_best_model_postcurriculum=True, timesteps=1000000):

        self.training_start_time = time_package.time()
        if not use_curriculum_training:
            print('Starting regular training')
            self.regular_training(timesteps=timesteps)
        else:
            self.curriculum_training(best_model_in_curriculum_exists_ok=best_model_in_curriculum_exists_ok,
                                     best_model_postcurriculum_exists_ok=best_model_postcurriculum_exists_ok,
                                     load_replay_buffer_of_best_model_postcurriculum=load_replay_buffer_of_best_model_postcurriculum)
        self.training_time = time_package.time()-self.training_start_time
        print("Finished training using", self.training_time, 's.')

        # self.sac_model.save_replay_buffer(os.path.join(self.model_folder_name, 'buffer')) # I added this
        self.current_info_condition = self.get_current_info_condition(
            self.family_of_agents_log)
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'finished_training'] = True
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'training_time'] += self.training_time
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'successful_training'] += self.successful_training
        self.family_of_agents_log.to_csv(
            self.overall_folder + 'family_of_agents_log.csv')
        # Also check if the information is in parameters_record. If not, add it.
        # self.check_and_update_parameters_record()

    def _evaluate_model_and_retrain_if_necessary(self, use_curriculum_training=False):

        self.collect_data(n_steps=n_steps)

        if len(self.ff_caught_T_new) < 1:
            print("No firefly was caught by the agent during testing. Re-train agent.")
            self.train_agent(use_curriculum_training=use_curriculum_training)
            if not self.successful_training:
                print("The set of parameters has failed to produce a well-trained agent in the past. \
                        Skip to the next set of parameters")
                raise ValueError(
                    "The set of parameters has failed to produce a well-trained agent in the past. Skip to the next set of parameters")
            if len(self.ff_caught_T_new) < 1:
                print("No firefly was caught by the agent during testing again. Abort: ")
                raise ValueError(
                    "Still no firefly was caught by the agent during testing after retraining. Abort: ")

        super().make_or_retrieve_ff_dataframe(
            exists_ok=False, data_folder_name=None, save_into_h5=False)
        super().find_patterns()
        self.calculate_pattern_frequencies_and_feature_statistics()

        return

    def _make_plots_for_the_model(self, currentTrial_for_animation, num_trials_for_animation, duration=None):
        if currentTrial_for_animation >= len(self.ff_caught_T_new):
            currentTrial_for_animation = len(self.ff_caught_T_new)-1
            num_trials_for_animation = min(len(self.ff_caught_T_new)-1, 5)

        self.annotation_info = animation_utils.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials,
                                                                    self.ignore_sudden_flash_indices, self.GUAT_indices_df['point_index'].values, self.try_a_few_times_indices)
        self.set_animation_parameters(currentTrial=currentTrial_for_animation,
                                      num_trials=num_trials_for_animation, k=1, duration=duration)
        self.call_animation_function(
            save_video=True, fps=None, video_dir=self.overall_folder + 'all_videos', plot_flash_on_ff=True)
        # self.combine_6_plots_for_neural_network()
        # #self.plot_side_by_side()
        # self.save_plots_in_data_folders()
        # self.save_plots_in_plot_folders()
        return

    def whether_to_update_record_and_make_plots(self):

        pattern_frequencies_record = pd.read_csv(
            self.overall_folder + 'pattern_frequencies_record.csv').drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
        self.current_info_condition_for_pattern_frequencies = self.get_current_info_condition(
            pattern_frequencies_record)
        to_update_record = len(
            pattern_frequencies_record.loc[self.current_info_condition_for_pattern_frequencies]) == 0

        to_make_plots = (not exists(os.path.join(self.patterns_and_features_folder_path, 'compare_pattern_frequencies.png')))\
            or (not exists(self.overall_folder + 'all_compare_pattern_frequencies/'+self.agent_id + '.png'))

        return to_update_record, to_make_plots

    def save_plots_in_data_folders(self):
        plot_statistics.plot_pattern_frequencies(
            self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True, data_folder_name=self.patterns_and_features_folder_path)
        plot_statistics.plot_feature_statistics(
            self.agent_monkey_feature_statistics, compare_monkey_and_agent=True, data_folder_name=self.patterns_and_features_folder_path)

        plot_statistics.plot_feature_histograms_for_monkey_and_agent(
            self.all_trial_features_valid_m, self.all_trial_features_valid, data_folder_name=self.patterns_and_features_folder_path)
        print("Made new plots")

    def save_plots_in_plot_folders(self):
        plot_statistics.plot_pattern_frequencies(self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True,
                                                 data_folder_name=os.path.join(
                                                     self.overall_folder, 'all_' + 'compare_pattern_frequencies'),
                                                 file_name=self.agent_id + '.png')
        plot_statistics.plot_feature_statistics(self.agent_monkey_feature_statistics, compare_monkey_and_agent=True,
                                                data_folder_name=os.path.join(
                                                    self.overall_folder, 'all_' + 'compare_feature_statistics'),
                                                file_name=self.agent_id + '.png')
        plot_statistics.plot_feature_histograms_for_monkey_and_agent(self.all_trial_features_valid_m, self.all_trial_features_valid,
                                                                     data_folder_name=os.path.join(
                                                                         self.overall_folder, 'all_' + 'feature_histograms'),
                                                                     file_name=self.agent_id + '.png')

    def get_minimum_current_info(self):
        minimal_current_info = {'dv_cost_factor': self.current_env_kwargs['dv_cost_factor'],
                                'dw_cost_factor': self.current_env_kwargs['dw_cost_factor'],
                                'w_cost_factor': self.current_env_kwargs['w_cost_factor']}

        # minimal_current_info = {'v_noise_std': self.v_noise_std,
        #                         'w_noise_std': self.w_noise_std,
        #                         'ffr_noise_scale': self.ffr_noise_scale,
        #                         'num_obs_ff': self.num_obs_ff,
        #                         'max_in_memory_time': self.max_in_memory_time}
        return minimal_current_info

    def check_with_family_of_agents_log(self) -> bool:
        self.current_info_condition = self.get_current_info_condition(
            self.family_of_agents_log)
        self.minimal_current_info = self.get_minimum_current_info()
        retrieved_current_info = self.family_of_agents_log.loc[self.current_info_condition]

        exist_best_model = exists(os.path.join(
            self.model_folder_name, 'best_model.zip'))
        finished_training = np.any(retrieved_current_info['finished_training'])
        print('exist_best_model', exist_best_model)
        print('finished_training', finished_training)

        self.successful_training = np.any(
            retrieved_current_info['successful_training'])

        if finished_training & (not self.successful_training):
            # That's the indication that the set of parameters cannot be used to train a good agent
            to_load_latest_agent = False
            to_train_agent = False
        elif exist_best_model & finished_training:
            # Then we don't have to train the agent; go to the next set of parameters
            to_load_latest_agent = True
            to_train_agent = False
        elif exist_best_model:
            # It seems like we have begun training the agent before, and we need to continue to train
            to_load_latest_agent = True
            to_train_agent = True
        else:
            # Need to put in the new set of information
            additional_current_info = {'finished_training': False,
                                       'year': time_package.localtime().tm_year,
                                       'month': time_package.localtime().tm_mon,
                                       'date': time_package.localtime().tm_mday,
                                       'training_time': 0}
            current_info = {**self.minimal_current_info,
                            **additional_current_info}

            self.family_of_agents_log = pd.concat([self.family_of_agents_log, pd.DataFrame(
                current_info, index=[0])]).reset_index(drop=True)
            self.family_of_agents_log.to_csv(
                self.overall_folder + 'family_of_agents_log.csv')
            to_load_latest_agent = False
            to_train_agent = True

        self.current_info_condition = self.get_current_info_condition(
            self.family_of_agents_log)
        return to_load_latest_agent, to_train_agent

    def check_and_update_parameters_record(self):
        self.parameters_record = pd.read_csv(
            self.overall_folder + 'parameters_record.csv').drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors='ignore')
        self.current_info_condition = self.get_current_info_condition(
            self.parameters_record)
        retrieved_current_info = self.parameters_record.loc[self.current_info_condition]
        if len(retrieved_current_info) == 0:
            # Need to put in the new set of information
            additional_current_info = {'working': 9}
            self.minimal_current_info = self.get_minimum_current_info()
            current_info = {**self.minimal_current_info,
                            **additional_current_info}
            self.parameters_record = pd.concat([self.parameters_record, pd.DataFrame(
                current_info, index=[0])]).reset_index(drop=True)
            self.parameters_record.to_csv(
                self.overall_folder + 'parameters_record.csv')

    def call_animation_function(self, margin=100, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True, plot_flash_on_ff=False,
                                show_speed_through_path_color=True, **animate_kwargs):
        self.obs_ff_indices_in_ff_dataframe_dict = None
        # self.obs_ff_indices_in_ff_dataframe_dict = {index: self.obs_ff_indices_in_ff_dataframe[index].astype(int) for index in range(len(self.obs_ff_indices_in_ff_dataframe))}

        if file_name is None:
            try:
                file_name = self.agent_id + \
                    f'__{self.currentTrial-self.num_trials+1}-{self.currentTrial}'
            except TypeError:
                file_name = self.agent_id + \
                    f'__{self.duration[0]}s_to_{self.duration[1]}s'
        # try adding ff capture rate to the file name
        try:
            file_name = file_name + \
                f'_rate_{round(self.eval_ff_capture_rate, 2)}'
        except AttributeError:
            pass

        file_name = file_name + '.mp4'

        agent_type = getattr(self, 'agent_type', None)
        if agent_type is None:
            dt = self.env.env.dt if hasattr(self.env, 'env') else self.env.dt
        elif str(agent_type).lower() in ('sb3', 'ff', 'feedforward'):
            dt = self.env.env.dt
        else:
            dt = self.env.dt

        super().call_animation_function(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position,
                                        set_xy_limits=set_xy_limits, plot_flash_on_ff=plot_flash_on_ff, in_obs_ff_dict=self.obs_ff_indices_in_ff_dataframe_dict,
                                        fps=int((1/dt)/self.k), show_speed_through_path_color=show_speed_through_path_color, **animate_kwargs)

    def make_animation_with_annotation(self, margin=100, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True):
        super().make_animation_with_annotation(margin=margin, save_video=save_video, video_dir=video_dir,
                                               file_name=file_name, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits)

    def combine_6_plots_for_neural_network(self):

        # self.add_2nd_ff = False if if self.num_obs_ff < 2 else True

        self.add_2nd_ff = False
        interpret_neural_network.combine_6_plots_for_neural_network(self.sac_model, full_memory=self.full_memory, invisible_distance=self.invisible_distance,
                                                                    add_2nd_ff=self.add_2nd_ff, data_folder_name=self.patterns_and_features_folder_path, const_memory=self.full_memory,
                                                                    data_folder_name2=self.overall_folder + 'all_' +
                                                                    'combined_6_plots_for_neural_network',
                                                                    file_name2=self.agent_id + '.png')

    def import_monkey_data(self, info_of_monkey, all_trial_features_m, pattern_frequencies_m, feature_statistics_m):
        self.info_of_monkey = info_of_monkey
        self.all_trial_features_m = all_trial_features_m
        self.all_trial_features_valid_m = self.all_trial_features_m[(self.all_trial_features_m['t_last_vis'] < 50) & (
            self.all_trial_features_m['hitting_arena_edge'] == False)].reset_index()
        self.pattern_frequencies_m = pattern_frequencies_m
        self.feature_statistics_m = feature_statistics_m

    def calculate_pattern_frequencies_and_feature_statistics(self):
        self.make_or_retrieve_all_trial_features()
        self.all_trial_features_valid = self.all_trial_features[(self.all_trial_features['t_last_vis'] < 50) & (
            self.all_trial_features['hitting_arena_edge'] == False)].reset_index()
        self.make_or_retrieve_all_trial_patterns()
        self.make_or_retrieve_pattern_frequencies()
        self.make_or_retrieve_feature_statistics()

        self.pattern_frequencies_a = self.pattern_frequencies
        self.feature_statistics_a = self.feature_statistics
        self.agent_monkey_pattern_frequencies = organize_patterns_and_features.combine_df_of_agent_and_monkey(
            self.pattern_frequencies_m, self.pattern_frequencies_a, agent_names=["Agent", "Agent2", "Agent3"])
        self.agent_monkey_feature_statistics = organize_patterns_and_features.combine_df_of_agent_and_monkey(
            self.feature_statistics_m, self.feature_statistics_a, agent_names=["Agent", "Agent2", "Agent3"])

        sb3_utils.add_row_to_pattern_frequencies_record(
            self.pattern_frequencies, self.minimal_current_info, self.overall_folder)
        sb3_utils.add_row_to_feature_medians_record(
            self.feature_statistics, self.minimal_current_info, self.overall_folder)
        sb3_utils.add_row_to_feature_means_record(
            self.feature_statistics, self.minimal_current_info, self.overall_folder)

    def plot_side_by_side(self):
        with general_utils.HiddenPrints():
            num_trials = 2
            plotting_params = {"show_stops": True,
                               "show_believed_target_positions": True,
                               "show_reward_boundary": True,
                               "show_connect_path_ff": True,
                               "show_scale_bar": True,
                               "hitting_arena_edge_ok": True,
                               "trial_too_short_ok": True}

            for currentTrial in [12, 69, 138, 221, 235]:
                # more: 259, 263, 265, 299, 393, 496, 523, 556, 601, 666, 698, 760, 805, 808, 930, 946, 955, 1002, 1003
                info_of_agent, plot_whole_duration, rotation_matrix, num_imitation_steps_monkey, num_imitation_steps_agent = process_agent_data.find_corresponding_info_of_agent(
                    self.info_of_monkey, currentTrial, num_trials, self.sac_model, self.agent_dt, env_kwargs=self.current_env_kwargs, agent_type=getattr(self, 'agent_type', None))

                with general_utils.initiate_plot(20, 20, 400):
                    additional_plots.PlotSidebySide(plot_whole_duration=plot_whole_duration,
                                                    info_of_monkey=self.info_of_monkey,
                                                    info_of_agent=info_of_agent,
                                                    num_imitation_steps_monkey=num_imitation_steps_monkey,
                                                    num_imitation_steps_agent=num_imitation_steps_agent,
                                                    currentTrial=currentTrial,
                                                    num_trials=num_trials,
                                                    rotation_matrix=rotation_matrix,
                                                    plotting_params=plotting_params,
                                                    data_folder_name=self.patterns_and_features_folder_path
                                                    )
