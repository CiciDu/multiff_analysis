from data_wrangling import general_utils, retrieve_raw_data, process_monkey_information
from pattern_discovery import organize_patterns_and_features, make_ff_dataframe
from visualization.matplotlib_tools import additional_plots, plot_statistics
from visualization.animation import animation_class, animation_utils
from reinforcement_learning.agents.rnn import rnn_env
from reinforcement_learning.agents.feedforward import sb3_env
from reinforcement_learning.collect_data import collect_agent_data, process_agent_data
from reinforcement_learning.agents.feedforward import interpret_neural_network, sb3_utils
from reinforcement_learning.base_classes import rl_base_utils
from reinforcement_learning.base_classes import run_logger
from decision_making_analysis.event_detection import detect_rsw_and_rcap
from reinforcement_learning.base_classes import base_env
from reinforcement_learning.base_classes import env_utils


import time as time_package
import gc
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
                 dv_cost_factor=0,
                 dw_cost_factor=0,
                 w_cost_factor=0,
                 jerk_cost_factor=0,
                 cost_per_stop=0,
                 flash_on_interval=0.3,
                 max_in_memory_time=3,
                 add_date_to_model_folder_name=False,
                 reward_per_ff=100,
                 reward_boundary=25,
                 angular_terminal_vel=0.05,
                 distance2center_cost=0,
                 stop_vel_cost=0,
                 zero_invisible_ff_features=True,
                 data_name='data_0',
                 std_anneal_preserve_fraction=1,
                 replay_keep_fraction=0.8,
                 agent_id=None,
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
                                          'jerk_cost_factor': jerk_cost_factor,
                                          'cost_per_stop': cost_per_stop,
                                          'print_ff_capture_incidents': True,
                                          'print_episode_reward_rates': True,
                                          'max_in_memory_time': max_in_memory_time,
                                          'flash_on_interval': flash_on_interval,
                                          'angular_terminal_vel': angular_terminal_vel,
                                          'distance2center_cost': distance2center_cost,
                                          'stop_vel_cost': stop_vel_cost,
                                          'reward_boundary': reward_boundary,
                                          'reward_per_ff': reward_per_ff,
                                          'zero_invisible_ff_features': zero_invisible_ff_features
                                          }

        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

        self.loaded_agent_name = ''

        self.dt = dt

        # self.agent_id = "dv" + str(dv_cost_factor) + \
        #                 "_dw" + str(dw_cost_factor) + "_w" + str(w_cost_factor) + \
        #                 "_memT" + \
        #     str(self.input_env_kwargs['max_in_memory_time'])
        self.agent_id = agent_id

        if len(overall_folder) > 0:
            os.makedirs(self.overall_folder, exist_ok=True)

        if model_folder_name is not None:
            self.model_folder_name = model_folder_name
        elif agent_id is not None:
            self.model_folder_name = os.path.join(
                self.overall_folder, self.agent_id)
        else:
            self.model_folder_name = self.overall_folder

        if self.agent_id is None:
            # use the last part of the overall_folder path as the agent_id
            self.agent_id = os.path.basename(os.path.normpath(overall_folder))

        print('model_folder_name:', self.model_folder_name)

        self.std_anneal_preserve_fraction = std_anneal_preserve_fraction
        self.replay_keep_fraction = replay_keep_fraction

        if add_date_to_model_folder_name:
            self.model_folder_name = self.model_folder_name + "_date" + \
                str(time_package.localtime().tm_mon) + "_" + \
                str(time_package.localtime().tm_mday)

        self.get_related_folder_names_from_model_folder_name(
            self.model_folder_name, data_name=data_name)

        # Initialize training status flag for robust downstream logging/logic
        self.successful_training = False

        # ---- Standardized artifact layout ----
        # Roots
        self.curr_dir = os.path.join(self.model_folder_name, 'curr')
        self.post_dir = os.path.join(self.model_folder_name, 'post')
        self.meta_dir = os.path.join(self.model_folder_name, 'meta')
        # Best subdirs (curr/post)
        self.best_model_in_curriculum_dir = os.path.join(self.curr_dir, 'best')
        self.best_model_postcurriculum_dir = os.path.join(
            self.post_dir, 'best')

        # Ensure directories exist
        for d in [self.curr_dir, self.post_dir, self.meta_dir,
                  self.best_model_in_curriculum_dir, self.best_model_postcurriculum_dir]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

        # Short, robust symlink helpers
        def _safe_symlink(target, link_path):
            try:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    try:
                        os.remove(link_path)
                    except IsADirectoryError:
                        # Remove existing directory link
                        os.rmdir(link_path)
                    except Exception:
                        pass
                os.symlink(target, link_path)
            except Exception:
                # Symlinks may fail on some filesystems; ignore
                pass

        # Minimal meta files
        try:
            env_params_path = os.path.join(self.meta_dir, 'env_params.json')
            with open(env_params_path, 'w') as f:
                json.dump(self.input_env_kwargs, f, indent=2, default=str)
            manifest_path = os.path.join(self.meta_dir, 'manifest.json')
            manifest = {
                'algo': getattr(self, 'agent_type', None),
                'version': '1',
                'role': 'root',
                'parent': None,
                'env_params': 'meta/env_params.json',
                'agent_params': 'meta/agent_params.json',
                'artifacts': {
                    'best_curr': 'ln/best_curr',
                    'best_post': 'ln/best_post'
                }
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            # agent_params may be None at init; write an empty dict now
            with open(os.path.join(self.meta_dir, 'agent_params.json'), 'w') as f:
                json.dump(self.agent_params if isinstance(
                    self.agent_params, dict) else {}, f, indent=2, default=str)
        except Exception:
            pass

    def get_related_folder_names_from_model_folder_name(self, model_folder_name, data_name='data_0'):
        self.model_folder_name = model_folder_name

        if '/all_agents/' not in model_folder_name:
            raise ValueError('model_folder_name must contain /all_agents/')

        # Split path into:
        #   before_all_agents / all_agents / after_all_agents
        before, after = model_folder_name.split('/all_agents/', 1)

        # "after" may have 1, 2, 3, or more path levels — keep them all
        sublevels = after.strip('/').split('/')

        # Base target directory:
        collected_base = os.path.join(before, 'all_collected_data')

        # Helper to build each category path
        def build_path(category):
            return os.path.join(collected_base, category, *sublevels, data_name)

        self.processed_data_folder_path = build_path('processed_data')
        self.planning_data_folder_path = build_path('planning')
        self.patterns_and_features_folder_path = build_path(
            'patterns_and_features')
        self.decision_making_folder_path = build_path('decision_making')

        # Make the directories
        for folder in [
            self.processed_data_folder_path,
            self.planning_data_folder_path,
            self.patterns_and_features_folder_path,
            self.decision_making_folder_path
        ]:
            os.makedirs(folder, exist_ok=True)

        return {
            'processed_data': self.processed_data_folder_path,
            'planning': self.planning_data_folder_path,
            'patterns_and_features': self.patterns_and_features_folder_path,
            'decision_making': self.decision_making_folder_path,
        }

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

    def load_best_model_postcurriculum(self, load_replay_buffer=True, restore_env_from_checkpoint=True):
        # load_agent will recreate env and agent using saved manifest
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=self.best_model_postcurriculum_dir, restore_env_from_checkpoint=restore_env_from_checkpoint)

    def load_best_model_in_curriculum(self, load_replay_buffer=True, restore_env_from_checkpoint=True):
        # load_agent will recreate env and agent using saved manifest
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=self.best_model_in_curriculum_dir, restore_env_from_checkpoint=restore_env_from_checkpoint)

    def curriculum_training(self, best_model_in_curriculum_exists_ok=True, best_model_postcurriculum_exists_ok=True, load_replay_buffer_of_best_model_postcurriculum=True):
        if (self.loaded_agent_name == 'model') or (self.loaded_agent_name == 'post/best'):
            self.successful_training = True
            return
        else:
            self._progress_in_curriculum(best_model_in_curriculum_exists_ok)
            self.successful_training = True
            return

    def _make_agent_for_curriculum_training(self):
        self.make_agent()

    def _progress_in_curriculum(self, best_model_in_curriculum_exists_ok=True):
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        print('Starting curriculum training')
        if best_model_in_curriculum_exists_ok:
            try:
                if self.loaded_agent_name != 'curr/best':
                    self.curriculum_env_kwargs = rl_base_utils.read_checkpoint_manifest(
                        self.loaded_agent_dir)['env_params']
                    print('Loaded curr/best')
                    print(
                        f'Made env based on env params saved in {self.loaded_agent_dir}')
                    self.make_env(**self.curriculum_env_kwargs)
                    self.load_best_model_in_curriculum(
                        load_replay_buffer=True, restore_env_from_checkpoint=False)

            except Exception:
                print('Need to train a new curr/best')
                self.make_init_env_for_curriculum_training()
                self._make_agent_for_curriculum_training()
        else:
            print('Making initial env and agent for curriculum training')
            self.make_init_env_for_curriculum_training()
            self._make_agent_for_curriculum_training()
        self.successful_training = False
        self._use_while_loop_for_curriculum_training()
        self.streamline_making_animation(currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                                         video_dir=self.best_model_postcurriculum_dir)

    def _make_init_env_for_curriculum_training(
        self,
        initial_flash_on_interval=1,
        initial_angular_terminal_vel=0.32,
        initial_reward_boundary=50,
        initial_stop_vel_cost=1,
        initial_distance2center_cost=1,
        initial_dv_cost_factor=0,
        initial_dw_cost_factor=0,
        initial_w_cost_factor=0,
    ):
        """Initialize the environment with baseline parameters for curriculum training."""
        params = dict(
            flash_on_interval=initial_flash_on_interval,
            angular_terminal_vel=initial_angular_terminal_vel,
            reward_boundary=initial_reward_boundary,
            stop_vel_cost=initial_stop_vel_cost,
            distance2center_cost=initial_distance2center_cost,
            dv_cost_factor=initial_dv_cost_factor,
            dw_cost_factor=initial_dw_cost_factor,
            w_cost_factor=initial_w_cost_factor,
        )

        self.curriculum_env_kwargs = copy.deepcopy(self.input_env_kwargs)
        self.curriculum_env_kwargs.update(params)

        try:
            if hasattr(self.env, "env_method"):
                self.env.env_method("set_curriculum_params",
                                    indices=None, **params)
            else:
                base_env = getattr(self.env, "unwrapped",
                                   getattr(self.env, "env", self.env))
                for k, v in params.items():
                    setattr(base_env, k, v)
        except Exception as e:
            print(f"Warning: failed to set initial curriculum params: {e}")

        self.current_env_kwargs = dict(self.curriculum_env_kwargs)
        print("Initialized curriculum environment parameters.")

    def _prune_or_clear_replay_buffer(self, keep_fraction: float = 0.2):
        """
        Prune or clear replay buffer to avoid stale data after curriculum stage change.

        Behavior:
          - If a buffer exists, retain the most recent keep_fraction (default 20%).
          - Handles both episode-list buffers (LSTM/GRU) and array-based buffers (FF attention).
          - If keep_fraction <= 0, clears the buffer completely.
        """
        try:
            rb = None
            # Find replay buffer on rl_agent or self
            if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'replay_buffer'):
                rb = self.rl_agent.replay_buffer
            elif hasattr(self, 'replay'):
                # FF attention agent uses self.replay
                rb = getattr(self, 'replay', None)
            elif hasattr(self, 'replay_buffer'):
                rb = getattr(self, 'replay_buffer', None)

            if rb is None:
                print('No replay buffer found to prune/clear')
                return

            # Episode-list style: has attribute 'buffer' as list
            if hasattr(rb, 'buffer') and isinstance(getattr(rb, 'buffer'), list):
                buf_list = rb.buffer
                n = len(buf_list)
                if n == 0:
                    return
                if keep_fraction <= 0:
                    rb.buffer = []
                    rb.position = 0 if hasattr(rb, 'position') else 0
                    print('Cleared episode replay buffer')
                    return
                k = max(1, int(n * float(keep_fraction)))
                # Keep the most recent k episodes according to ring-buffer semantics
                if hasattr(rb, 'position'):
                    pos = int(rb.position)
                    # reconstruct chronological order from ring buffer
                    ordered = buf_list[pos:] + buf_list[:pos]
                    trimmed = ordered[-k:]
                    rb.buffer = trimmed
                    rb.position = len(trimmed) % max(
                        1, getattr(rb, 'capacity', len(trimmed)))
                else:
                    rb.buffer = buf_list[-k:]
                print(
                    f'Pruned episode replay buffer to last {k} episodes (~{int(keep_fraction*100)}%)')
                return

            # Array-based style (FF): has numpy arrays and size/ptr
            if all(hasattr(rb, attr) for attr in ('obs', 'next_obs', 'action', 'reward', 'done')) and hasattr(rb, 'size'):
                size = int(getattr(rb, 'size', 0))
                if size <= 0:
                    return
                if keep_fraction <= 0:
                    # reset size and pointer only; arrays can remain allocated
                    rb.size = 0
                    if hasattr(rb, 'ptr'):
                        rb.ptr = 0
                    print('Cleared array replay buffer (size=0)')
                    return
                k = max(1, int(size * float(keep_fraction)))
                # compute indices of last k transitions respecting circular buffer
                ptr = int(getattr(rb, 'ptr', size))
                idx = (np.arange(size - k, size) + ptr) % max(1,
                                                              getattr(rb, 'capacity', size))
                # compact into front of arrays
                rb.obs[:k] = rb.obs[idx]
                rb.next_obs[:k] = rb.next_obs[idx]
                rb.action[:k] = rb.action[idx]
                rb.reward[:k] = rb.reward[idx]
                rb.done[:k] = rb.done[idx]
                rb.size = k
                rb.ptr = k % max(1, getattr(rb, 'capacity', k))
                print(
                    f'Pruned array replay buffer to last {k} steps (~{int(keep_fraction*100)}%)')
                return

            print('Replay buffer format not recognized; skipping prune')
        except Exception as e:
            print('Warning: failed to prune/clear replay buffer:', e)

    # Wire pruning after curriculum env updates

    # ------------------------------------------------------------------
    # Curriculum helpers (targets, getters, setters, step selection)
    # ------------------------------------------------------------------
    def _get_curriculum_targets(self):
        return {
            'flash_on_interval': self.input_env_kwargs['flash_on_interval'],
            'angular_terminal_vel': self.input_env_kwargs['angular_terminal_vel'],
            'distance2center_cost': self.input_env_kwargs['distance2center_cost'],
            'stop_vel_cost': self.input_env_kwargs['stop_vel_cost'],
            'reward_boundary': self.input_env_kwargs['reward_boundary'],
            'dv_cost_factor': self.input_env_kwargs['dv_cost_factor'],
            'dw_cost_factor': self.input_env_kwargs['dw_cost_factor'],
            'w_cost_factor': self.input_env_kwargs['w_cost_factor'],
        }

    def _get_curriculum_params(self):
        if hasattr(self, 'env') and hasattr(self.env, 'env_method'):
            try:
                params_list = self.env.env_method(
                    'get_curriculum_params', indices=0)
                params = params_list[0] if isinstance(
                    params_list, (list, tuple)) else params_list
                if isinstance(params, dict):
                    return params
            except Exception:
                pass
        env = self._get_active_env()
        return {
            'flash_on_interval': getattr(env, 'flash_on_interval', None),
            'angular_terminal_vel': getattr(env, 'angular_terminal_vel', None),
            'distance2center_cost': getattr(env, 'distance2center_cost', None),
            'stop_vel_cost': getattr(env, 'stop_vel_cost', None),
            'reward_boundary': getattr(env, 'reward_boundary', None),
            'dv_cost_factor': getattr(env, 'dv_cost_factor', None),
            'dw_cost_factor': getattr(env, 'dw_cost_factor', None),
            'w_cost_factor': getattr(env, 'w_cost_factor', None),
        }

    def _apply_curriculum_params(self, updates: dict):
        if not isinstance(updates, dict) or len(updates) == 0:
            return
        for k, v in updates.items():
            self.curriculum_env_kwargs[k] = v
        if hasattr(self, 'env') and hasattr(self.env, 'env_method'):
            try:
                self.env.env_method('set_curriculum_params',
                                    indices=None, **updates)
            except Exception as e:
                print('Warning: failed to broadcast curriculum param update:', e)
        else:
            env = self._get_active_env()
            for key, value in updates.items():
                try:
                    setattr(env, key, value)
                except Exception:
                    pass
        self.current_env_kwargs = self.curriculum_env_kwargs

    def _update_env_after_meeting_reward_threshold(self):

        print('Updating env after meeting reward threshold...')
        before = self._get_curriculum_params()
        targets = self._get_curriculum_targets()
        updates, do_prune = self._choose_next_curriculum_update(
            before, targets)
        if updates:
            self._apply_curriculum_params(updates)
            print('Updated params:', updates)
            if do_prune:
                self._prune_or_clear_replay_buffer(
                    keep_fraction=self.replay_keep_fraction)
        rep_key = next(iter(updates.keys())) if isinstance(
            updates, dict) and len(updates) > 0 else None
        self._after_curriculum_env_change(rep_key)
        after = self._get_curriculum_params()
        stage_summary = {'before': before, 'after': after, 'targets': targets}
        print('Stage summary:', stage_summary)

    def collect_data(self, n_steps=8000, exists_ok=False, save_data=True, retrieve_ff_flash_sorted=False):
        if exists_ok:
            try:
                self.retrieve_monkey_data(retrieve_ff_flash_sorted=retrieve_ff_flash_sorted)
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

        # Ensure an agent/model exists before collecting data
        if not hasattr(self, 'rl_agent') or getattr(self, 'rl_agent') is None:
            try:
                # Prefer loading an existing agent if available
                self.load_latest_agent(load_replay_buffer=True)
            except Exception:
                # Fall back to creating a fresh env and agent
                print('No agent found. Creating a fresh env and agent...')
                try:
                    self.env
                except AttributeError:
                    self.make_env(**self.input_env_kwargs)
                self.make_agent()

        env_data_collection_kwargs = copy.deepcopy(self.current_env_kwargs)
        env_data_collection_kwargs.update({'episode_len': n_steps+100})

        agent_type = getattr(self, 'agent_type', None)
        at = str(agent_type).lower() if agent_type is not None else None
        if at in ('lstm', 'gru', 'rnn'):
            self.env_for_data_collection = rnn_env.CollectInformationLSTM(
                **env_data_collection_kwargs)
            LSTM = True
        else:
            self.env_for_data_collection = sb3_env.CollectInformation(
                **env_data_collection_kwargs)
            LSTM = False

        self._run_agent_to_collect_data(
            n_steps=n_steps, save_data=save_data, LSTM=LSTM)

    def _run_agent_to_collect_data(self, exists_ok=False, n_steps=8000, save_data=True, LSTM=False):

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
            = collect_agent_data.collect_agent_data_func(self.env_for_data_collection, self.rl_agent, n_steps=self.n_steps, agent_type=self.agent_type)
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

    def retrieve_monkey_data(self, speed_threshold_for_distinct_stop=1, retrieve_ff_flash_sorted=False):
        self.npz_file_pathway = os.path.join(
            self.processed_data_folder_path, 'ff_basic_info.npz')
        self.ff_caught_T_sorted, self.ff_index_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.ff_life_sorted, \
            self.ff_flash_end_sorted = retrieve_raw_data._retrieve_ff_info_in_npz_from_txt_data(
                self.processed_data_folder_path)
        if retrieve_ff_flash_sorted:
            self.ff_flash_sorted = retrieve_raw_data._retrieve_ff_flash_sorted_in_npz(
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
            self.make_ff_dataframe_from_ff_in_obs_df()
            # base_processing_class.BaseProcessing.make_or_retrieve_ff_dataframe(self, exists_ok=False, save_into_h5=False)
            print("made ff_dataframe")

            if save_data:
                self.ff_dataframe.to_csv(self.ff_dataframe_path)
                print("saved ff_dataframe at", self.ff_dataframe_path)
        return

    def make_ff_dataframe_from_ff_in_obs_df(self):
        self.ff_dataframe = self.ff_in_obs_df.copy()

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
        # Prefer symlinks in ln/ first (stable interface), then fall back to canonical dirs
        candidates.extend([
            os.path.join(dir_name, 'ln', 'best_post'),
            os.path.join(dir_name, 'ln', 'best_curr'),
            os.path.join(dir_name, 'post', 'best'),
            os.path.join(dir_name, 'curr', 'best'),
        ])
        candidate_names.extend(
            ['ln/best_post', 'ln/best_curr', 'post/best', 'curr/best'])

        last_error = None
        self.loaded_agent_dir = None
        for d, name in zip(candidates, candidate_names):
            if name == 'post/best':
                print(f'Loading best model from post/best')
            try:
                self.load_agent(
                    load_replay_buffer=load_replay_buffer, dir_name=d, restore_env_from_checkpoint=True)
                self.loaded_agent_name = name
                if name in ('curr/best', 'ln/best_curr'):
                    self.curriculum_env_kwargs = self.current_env_kwargs.copy()
                return
            except Exception as e:
                last_error = (dir_name, e)
        if last_error is not None:
            d, e = last_error
            agent_type = getattr(self, 'agent_type', None)
            if isinstance(agent_type, str) and agent_type.lower() == 'rppo':
                msg = f"There was an error retrieving agent in {d}. Error message {e}"
            else:
                msg = f"There was an error retrieving agent or replay_buffer in {d}. Error message {e}"
            raise ValueError(msg)

    def streamline_getting_data_from_agent(self, n_steps=8000, exists_ok=False, save_data=True, load_replay_buffer=False, retrieve_ff_flash_sorted=False):
        if exists_ok:
            try:
                self.retrieve_monkey_data(retrieve_ff_flash_sorted=retrieve_ff_flash_sorted)
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
            self.rl_agent
        except AttributeError:
            self.make_agent()
        self.load_latest_agent(load_replay_buffer=False)
        self.streamline_making_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation,
                                         duration=duration, n_steps=n_steps, file_name=None)

    def streamline_making_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000, file_name=None, video_dir=None,
                                    data_exists_ok=False, save_video=True, save_format='html', display_inline=False, **animate_kwargs):
        self.collect_data(n_steps=n_steps, exists_ok=data_exists_ok, retrieve_ff_flash_sorted=True)
        # if len(self.ff_caught_T_new) >= currentTrial_for_animation:
        self.make_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation,
                            duration=duration, file_name=file_name, video_dir=video_dir, save_video=save_video, save_format=save_format, display_inline=display_inline,
                            **animate_kwargs)

    def make_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], file_name=None, video_dir=None, max_num_frames=150, save_format=None,
                       save_video=True, display_inline=False, **animate_kwargs):
        self.set_animation_parameters(currentTrial=currentTrial_for_animation, num_trials=num_trials_for_animation,
                                      k=1, duration=duration, max_num_frames=max_num_frames, reduce_k_if_too_many_frames=False)
        self.call_animation_function(
            file_name=file_name, video_dir=video_dir, save_video=save_video, save_format=save_format, display_inline=display_inline,
            **animate_kwargs)

    def streamline_everything(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                              use_curriculum_training=True, load_replay_buffer_of_best_model_postcurriculum=True,
                              best_model_in_curriculum_exists_ok=True,
                              best_model_postcurriculum_exists_ok=True,
                              to_load_latest_agent=True,
                              load_replay_buffer=True,
                              to_train_agent=True,
                              make_animation=False):

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
                # Fallback: create a fresh environment and agent before training
                self.make_env(**self.input_env_kwargs)
                self.make_agent()
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

            if make_animation:
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

        # Ensure family_of_agents_log exists when training is invoked directly
        family_log = getattr(self, 'family_of_agents_log', None)
        if family_log is None or not isinstance(family_log, pd.DataFrame):
            try:
                self.family_of_agents_log = rl_base_utils.retrieve_or_make_family_of_agents_log(
                    self.overall_folder)
            except Exception:
                # Fallback: keep an in-memory empty frame with expected columns
                self.family_of_agents_log = pd.DataFrame(columns=[
                    'dv_cost_factor', 'dw_cost_factor', 'w_cost_factor',
                    'v_noise_std', 'w_noise_std', 'ffr_noise_scale',
                    'num_obs_ff', 'max_in_memory_time',
                    'finished_training', 'year', 'month', 'date',
                    'training_time', 'successful_training'
                ])

        # Emit run_start once per training invocation
        try:
            # Prefer externally provided sweep params
            sweep_params = dict(getattr(self, 'sweep_params', {}))
            # Add common env params if available
            env_info = {}
            try:
                env_info['num_obs_ff'] = self.input_env_kwargs.get(
                    'num_obs_ff')
                env_info['max_in_memory_time'] = self.input_env_kwargs.get(
                    'max_in_memory_time')
                env_info['angular_terminal_vel'] = self.input_env_kwargs.get(
                    'angular_terminal_vel')
                env_info['dt'] = self.input_env_kwargs.get('dt')
            except Exception:
                pass
            sweep_params.update(
                {k: v for k, v in env_info.items() if v is not None})
            run_logger.log_run_start(self.overall_folder, agent_type=getattr(
                self, 'agent_type', 'rnn'), sweep_params=sweep_params)
        except Exception as e:
            print('[logger] failed to log run start from base class:', e)

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

        # self.rl_agent.save_replay_buffer(os.path.join(self.model_folder_name, 'buffer')) # I added this
        self.current_info_condition = self.get_current_info_condition(
            self.family_of_agents_log)
        # Ensure 'successful_training' column exists and is numeric for safe aggregation
        try:
            if 'successful_training' not in self.family_of_agents_log.columns:
                self.family_of_agents_log['successful_training'] = 0
            self.family_of_agents_log['successful_training'] = self.family_of_agents_log['successful_training'].fillna(
                0)
        except Exception:
            pass
        # Default to True if subclasses didn't explicitly set flag but training completed
        if not hasattr(self, 'successful_training'):
            self.successful_training = True
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'finished_training'] = True
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'training_time'] += self.training_time
        self.family_of_agents_log.loc[self.current_info_condition,
                                      'successful_training'] += self.successful_training
        self.family_of_agents_log.to_csv(
            os.path.join(self.overall_folder, 'family_of_agents_log.csv'))
        # Also check if the information is in parameters_record. If not, add it.
        # self.check_and_update_parameters_record()

        # Emit run_end with basic metric
        try:
            metrics = {}
            try:
                metrics['best_avg_reward'] = getattr(
                    self, 'best_avg_reward', None)
            except Exception:
                pass
            sweep_params = dict(getattr(self, 'sweep_params', {}))
            run_logger.log_run_end(self.overall_folder, agent_type=getattr(
                self, 'agent_type', 'rnn'), sweep_params=sweep_params, status='finished', metrics=metrics)
        except Exception as e:
            print('[logger] failed to log run end from base class:', e)

    def _get_active_env(self):
        env = getattr(self, 'env', None)
        if env is None:
            return None
        # Prefer single underlying env when vectorized (e.g., SB3 DummyVecEnv)
        if hasattr(env, 'envs') and isinstance(getattr(env, 'envs', None), (list, tuple)) and len(env.unwrapped.envs) > 0:
            single = env.unwrapped.envs[0]
            # Return base unwrapped environment when available
            return getattr(single, 'unwrapped', single)
        # Fallback to .env wrapped gym environment
        if hasattr(env, 'env'):
            wrapped = env.env
            return getattr(wrapped, 'unwrapped', wrapped)
        # Return as-is
        return getattr(env, 'unwrapped', env)

    def _compute_reward_threshold(self, n_eval_episodes: int = 2, ff_caught_rate_threshold: float = 0.1) -> float:
        env_for_eval = self._get_active_env()
        return rl_base_utils.calculate_reward_threshold_for_curriculum_training(
            env_for_eval, n_eval_episodes=n_eval_episodes, ff_caught_rate_threshold=ff_caught_rate_threshold)

    def _ensure_curriculum_log(self) -> str:
        # Log curriculum progression at curr/log.csv
        log_path = os.path.join(self.curr_dir, 'log.csv')
        columns = [
            'stage', 'reward_threshold', 'best_avg_reward',
            'flash_on_interval', 'angular_terminal_vel', 'reward_boundary',
            'distance2center_cost', 'stop_vel_cost',
            'dv_cost_factor', 'dw_cost_factor', 'w_cost_factor',
            'finished_curriculum', 'attempt_passed'
        ]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            try:
                pd.DataFrame(columns=columns).to_csv(log_path, index=False)
            except Exception:
                pass
        return log_path

    def _use_while_loop_for_curriculum_training(self, eval_eps_freq: int = 20, num_eval_episodes: int = 2):
        stage = 0
        finished_curriculum = False
        log_path = self._ensure_curriculum_log()

        while True:
            stage += 1
            gc.collect()

            # Compute dynamic reward threshold based on current env
            reward_threshold = self._compute_reward_threshold(
                n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
            print(
                f'[Stage {stage}] Current reward threshold: {reward_threshold:.2f}')

            # Delegate training logic to subclass
            try:
                passed, best_avg_reward = self._curriculum_train_step(
                    reward_threshold=reward_threshold,
                    eval_eps_freq=eval_eps_freq,
                    num_eval_episodes=num_eval_episodes,
                )
            except NotImplementedError:
                raise
            except Exception as e:
                print('[curriculum] training step failed:', e)
                passed = False
                best_avg_reward = float('-inf')

            # Build payload and log
            env = self._get_active_env()
            payload = {
                'stage': stage,
                'reward_threshold': reward_threshold,
                'best_avg_reward': best_avg_reward,
                'flash_on_interval': getattr(env, 'flash_on_interval', None),
                'angular_terminal_vel': getattr(env, 'angular_terminal_vel', None),
                'reward_boundary': getattr(env, 'reward_boundary', None),
                'distance2center_cost': getattr(env, 'distance2center_cost', None),
                'stop_vel_cost': getattr(env, 'stop_vel_cost', None),
                'dv_cost_factor': getattr(env, 'dv_cost_factor', None),
                'dw_cost_factor': getattr(env, 'dw_cost_factor', None),
                'w_cost_factor': getattr(env, 'w_cost_factor', None),
                'finished_curriculum': finished_curriculum,
                'attempt_passed': passed,
            }

            # Per-agent CSV
            try:
                pd.DataFrame([payload]).to_csv(
                    log_path, mode='a', header=False, index=False)
            except Exception:
                pass
            # Aggregate logger
            try:
                sweep_params = getattr(self, 'sweep_params', {})
                run_logger.log_curriculum_stage(
                    self.overall_folder,
                    agent_type=getattr(self, 'agent_type', 'rnn'),
                    sweep_params=sweep_params,
                    stage_payload=payload,
                )
            except Exception as e:
                print('[logger] failed to log curriculum stage:', e)

            if not passed:
                print(
                    f'[Stage {stage}] Warning: best reward {best_avg_reward:.2f} < threshold {reward_threshold:.2f}. Retrying...')
                continue

            print(
                f'[Stage {stage}] Progressed: best reward {best_avg_reward:.2f} ≥ threshold {reward_threshold:.2f}')

            # Advance curriculum environment
            self._update_env_after_meeting_reward_threshold()
            env_after = self._get_active_env()

            # Check if all target curriculum conditions are met
            # Use tolerant comparison for floating point curriculum params
            current_params = self._get_curriculum_params()
            targets = self._get_curriculum_targets()

            if finished_curriculum:
                print(
                    f'[Stage {stage}] All curriculum conditions met. Curriculum training complete.')
                break

            def _close(a, b):
                try:
                    if a is None or b is None:
                        return False
                    return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-8))
                except Exception:
                    return a == b

            if all(_close(current_params.get(k), targets.get(k)) for k in targets.keys()):
                finished_curriculum = True
                print('Reaching the last stage of curriculum training.')

        # Final post-curriculum training
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        self.make_env(**self.input_env_kwargs)
        self.load_best_model_in_curriculum(
            load_replay_buffer=True, restore_env_from_checkpoint=False)

        final_reward_threshold = self._compute_reward_threshold(
            n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)

        print(
            f'Starting post-curriculum training with reward threshold: {final_reward_threshold:.2f}')
        self._post_curriculum_final_train(
            reward_threshold=final_reward_threshold,
            eval_eps_freq=eval_eps_freq,
            num_eval_episodes=num_eval_episodes,
        )

        # Ensure the final best model is loaded into memory for downstream use
        try:
            self.load_best_model_postcurriculum(load_replay_buffer=True)
        except Exception as e:
            print('[curriculum] failed to load best post-curriculum model:', e)

        print('Finished post-curriculum training.')

    def _curriculum_train_step(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        """
        To be implemented by subclasses.
        Must return a tuple (passed: bool, best_avg_reward: float).
        """
        raise NotImplementedError

    def _post_curriculum_final_train(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        """
        To be implemented by subclasses. Executes the final post-curriculum training.
        """
        raise NotImplementedError

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
                                                                    self.ignore_sudden_flash_indices, self.rsw_indices_df['point_index'].values, self.retry_capture_indices)
        self.set_animation_parameters(currentTrial=currentTrial_for_animation,
                                      num_trials=num_trials_for_animation, k=1, duration=duration, reduce_k_if_too_many_frames=False)
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

        # Detect existence of a best model for both legacy and new standardized layouts
        candidate_paths = [
            os.path.join(self.model_folder_name, 'best_model.zip'),
            os.path.join(self.model_folder_name, 'checkpoint_manifest.json'),
            os.path.join(self.model_folder_name, 'post',
                         'best', 'checkpoint_manifest.json'),
            os.path.join(self.model_folder_name, 'curr',
                         'best', 'checkpoint_manifest.json'),
        ]
        exist_best_model = any(exists(p) for p in candidate_paths)
        finished_training = np.any(retrieved_current_info['finished_training'])
        print('exist_best_model', exist_best_model)
        print('finished_training', finished_training)

        # Treat missing values as 0 for robust boolean evaluation
        try:
            self.successful_training = bool(np.any(
                retrieved_current_info['successful_training'].fillna(0)))
        except Exception:
            self.successful_training = bool(np.any(
                retrieved_current_info.get('successful_training', 0)))

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
                                       'training_time': 0,
                                       'successful_training': 0}
            current_info = {**self.minimal_current_info,
                            **additional_current_info}

            self.family_of_agents_log = pd.concat([self.family_of_agents_log, pd.DataFrame(
                current_info, index=[0])]).reset_index(drop=True)
            self.family_of_agents_log.to_csv(
                os.path.join(self.overall_folder, 'family_of_agents_log.csv'))
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
                                show_speed_through_path_color=True, save_format=None, display_inline=False, **animate_kwargs):
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

        # Robustly unwrap to the base (single) env and read dt
        base_env = None
        env_obj = getattr(self, 'env', None)
        if env_obj is not None:
            if hasattr(env_obj, 'envs') and isinstance(getattr(env_obj, 'envs', None), (list, tuple)) and len(env_obj.envs) > 0:
                base_env = env_obj.envs[0]
            elif hasattr(env_obj, 'env'):
                base_env = env_obj.env
            else:
                base_env = env_obj
        dt = getattr(base_env, 'dt', None)
        if dt is None:
            dt = self.input_env_kwargs.get('dt', 0.1)

        super().call_animation_function(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position,
                                        set_xy_limits=set_xy_limits, plot_flash_on_ff=plot_flash_on_ff, in_obs_ff_dict=self.obs_ff_indices_in_ff_dataframe_dict,
                                        fps=int((1/dt)/self.k), show_speed_through_path_color=show_speed_through_path_color, save_format=save_format,
                                        display_inline=display_inline, **animate_kwargs)

    def make_animation_with_annotation(self, margin=100, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True):
        super().make_animation_with_annotation(margin=margin, save_video=save_video, video_dir=video_dir,
                                               file_name=file_name, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits)

    def combine_6_plots_for_neural_network(self):

        # self.add_2nd_ff = False if if self.num_obs_ff < 2 else True

        self.add_2nd_ff = False
        interpret_neural_network.combine_6_plots_for_neural_network(self.rl_agent, full_memory=self.full_memory, invisible_distance=self.invisible_distance,
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

    # def plot_side_by_side(self):

    # Note: I've deleted the old find_corresponding_info_of_agent function on 2025/10/28

    #     with general_utils.HiddenPrints():
    #         num_trials = 2
    #         plotting_params = {"show_stops": True,
    #                            "show_believed_target_positions": True,
    #                            "show_reward_boundary": True,
    #                            "show_connect_path_ff": True,
    #                            "show_scale_bar": True,
    #                            "hitting_arena_edge_ok": True,
    #                            "trial_too_short_ok": True}

    #         for currentTrial in [12, 69, 138, 221, 235]:
    #             # more: 259, 263, 265, 299, 393, 496, 523, 556, 601, 666, 698, 760, 805, 808, 930, 946, 955, 1002, 1003
    #             info_of_agent, plot_whole_duration, rotation_matrix, num_imitation_steps_monkey, num_imitation_steps_agent = process_agent_data.find_corresponding_info_of_agent(
    #                 self.info_of_monkey, currentTrial, num_trials, self.rl_agent, self.agent_dt, env_kwargs=self.current_env_kwargs, agent_type=getattr(self, 'agent_type', None))

    #             with general_utils.initiate_plot(20, 20, 400):
    #                 additional_plots.PlotSidebySide(plot_whole_duration=plot_whole_duration,
    #                                                 info_of_monkey=self.info_of_monkey,
    #                                                 info_of_agent=info_of_agent,
    #                                                 num_imitation_steps_monkey=num_imitation_steps_monkey,
    #                                                 num_imitation_steps_agent=num_imitation_steps_agent,
    #                                                 currentTrial=currentTrial,
    #                                                 num_trials=num_trials,
    #                                                 rotation_matrix=rotation_matrix,
    #                                                 plotting_params=plotting_params,
    #                                                 data_folder_name=self.patterns_and_features_folder_path
    #                                                 )

    def _choose_next_curriculum_update(self, current: dict, targets: dict):
        """
        Choose the next curriculum update.

        Logic:
        1. Check high-priority single-parameter updates.
        2. If none apply, check grouped primary parameters.
        3. If still none, check grouped secondary parameters.
        4. Reverse update direction if (current - target) has the wrong sign,
        keeping the same step magnitude toward target.

        Returns:
            updates (dict): key → new value (empty if no change)
            do_prune (bool): whether to clear replay buffer after update
        """

        def _apply_update_rule(key, c_val, t_val, need_fn, step_fn, prune_flag):
            """Apply a single update rule, handling both normal and reverse directions."""
            try:
                # Normal direction (expected sign)
                if need_fn(c_val, t_val):
                    return step_fn(c_val, t_val), prune_flag

                # Reverse direction (wrong sign, move toward target)
                if c_val != t_val:
                    step_size = abs(step_fn(t_val, c_val) - t_val)
                    new_val = min(
                        c_val + step_size, t_val) if c_val < t_val else max(c_val - step_size, t_val)
                    return new_val, prune_flag
            except Exception:
                pass
            return None, False

        # -----------------------------
        # 1. High-priority single-parameter updates
        # -----------------------------
        singles = [
            ('reward_boundary', lambda c, t: c >
             t, lambda c, t: max(c - 10, t), True),
            ('angular_terminal_vel', lambda c, t: c >
             t, lambda c, t: max(c / 2, t), True),
            ('flash_on_interval', lambda c, t: c >
             t, lambda c, t: max(c - 0.3, t), True),
        ]

        for key, need, step, prune in singles:
            c_val, t_val = current.get(key), targets.get(key)
            if c_val is None or t_val is None:
                continue
            new_val, do_prune = _apply_update_rule(
                key, c_val, t_val, need, step, prune)
            if new_val is not None:
                return {key: new_val}, do_prune

        # -----------------------------
        # 2. Grouped parameter updates
        # -----------------------------
        grouped_specs_primary = [
            ('distance2center_cost', lambda c, t: c >
             t, lambda c, t: max(c - 1, t), True),
            ('stop_vel_cost', lambda c, t: c > t,
             lambda c, t: max(c - 1, t), False),
        ]
        grouped_specs_secondary = [
            ('dv_cost_factor', lambda c, t: c < t,
             lambda c, t: min(c + 1, t), False),
            ('dw_cost_factor', lambda c, t: c < t,
             lambda c, t: min(c + 1, t), False),
            ('w_cost_factor', lambda c, t: c < t,
             lambda c, t: min(c + 1, t), False),
        ]

        # Pass 1: primary group (higher priority)
        primary_updates, do_prune_primary = {}, False
        for key, need, step, prune in grouped_specs_primary:
            c_val, t_val = current.get(key), targets.get(key)
            if c_val is None or t_val is None:
                continue
            new_val, prune_flag = _apply_update_rule(
                key, c_val, t_val, need, step, prune)
            if new_val is not None:
                primary_updates[key] = new_val
                do_prune_primary |= prune_flag
        if primary_updates:
            return primary_updates, do_prune_primary

        # Pass 2: secondary group
        updates, do_prune = {}, False
        for key, need, step, prune in grouped_specs_secondary:
            c_val, t_val = current.get(key), targets.get(key)
            if c_val is None or t_val is None:
                continue
            new_val, prune_flag = _apply_update_rule(
                key, c_val, t_val, need, step, prune)
            if new_val is not None:
                updates[key] = new_val
                do_prune |= prune_flag

        # -----------------------------
        # 3. Return updates if any
        # -----------------------------
        return (updates, do_prune) if updates else ({}, False)

    def _after_curriculum_env_change(self, updated_key: str):
        """Reset adaptive components (anneal progress, entropy temperature) after env change."""
        # --- Reset annealing progress ---
        if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'policy_net'):
            try:
                current = getattr(self.rl_agent.policy_net, 'anneal_step', 0)
                new_value = int(
                    max(0, int(current * self.std_anneal_preserve_fraction)))
                setattr(self.rl_agent.policy_net, 'anneal_step', new_value)
                print(f'std_anneal step before: {current}, after: {new_value}')
            except Exception as e:
                print('Warning: failed to reset std-anneal progress:', e)

        # --- Reset entropy temperature (alpha) ---
        if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'log_alpha'):
            try:
                with torch.no_grad():
                    beta = getattr(self, 'alpha_reset_beta', 0.6)
                    cur_log_alpha = self.rl_agent.log_alpha
                    tgt_log_alpha = torch.zeros_like(cur_log_alpha)

                    alpha_before = getattr(
                        self.rl_agent, 'alpha', cur_log_alpha.exp())
                    new_log_alpha = beta * cur_log_alpha + \
                        (1 - beta) * tgt_log_alpha
                    self.rl_agent.log_alpha.copy_(new_log_alpha)

                    if hasattr(self.rl_agent, 'alpha'):
                        self.rl_agent.alpha = self.rl_agent.log_alpha.exp()
                    alpha_after = getattr(
                        self.rl_agent, 'alpha', self.rl_agent.log_alpha.exp())

                # Clear optimizer state safely
                opt = getattr(self.rl_agent, 'alpha_optimizer', None)
                if opt is not None:
                    try:
                        opt.zero_grad(set_to_none=True)
                    except TypeError:
                        opt.zero_grad()
                if getattr(self.rl_agent.log_alpha, 'grad', None) is not None:
                    self.rl_agent.log_alpha.grad = None

            except Exception as e:
                print('Warning: failed to reset entropy temperature (alpha):', e)
