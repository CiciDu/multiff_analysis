import sys
from data_wrangling import basic_func, base_processing_class, further_processing_class
from pattern_discovery import organize_patterns_and_features, make_ff_dataframe
from visualization import additional_plots, plot_statistics, animation_utils

from machine_learning.RL.env_related import base_env, env_for_lstm, env_for_sb3, collect_agent_data_utils
from machine_learning.RL.SB3 import interpret_neural_network, rl_for_multiff_utils, SB3_functions

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



class _RLforMultifirefly(further_processing_class.FurtherProcessing):

    def __init__(self, 
                overall_folder=None, 
                model_folder_name=None,
                dt=0.1,
                dv_cost_factor=10,
                dw_cost_factor=10,
                w_cost_factor=3,   
                flash_on_interval=0.3,  
                max_in_memory_time=3,   
                add_date_to_model_folder_name=False,
                use_env2=False,
                data_name='data_0',
                **additional_env_kwargs):

        self.player = "agent"
        self.agent_params = None
        self.overall_folder = overall_folder
        self.use_env2 = use_env2

        self.env_kwargs = {'dt': dt,
                            'dv_cost_factor': dv_cost_factor,
                            'dw_cost_factor': dw_cost_factor,
                            'w_cost_factor': w_cost_factor,
                            'print_ff_capture_incidents': True,
                            'print_episode_reward_rates': True,
                            'flash_on_interval': flash_on_interval,
                            'max_in_memory_time': max_in_memory_time,
                            }
        
        self.env_kwargs.update(additional_env_kwargs)
        self.agent_id = "dv" + str(dv_cost_factor) + \
                        "_dw" + str(dw_cost_factor) + "_w" + str(w_cost_factor) + \
                        "_mem" + str(self.env_kwargs['max_in_memory_time'])

        
        if len(overall_folder) > 0:
            os.makedirs(self.overall_folder, exist_ok=True)

        self.model_folder_name = model_folder_name if model_folder_name is not None else os.path.join(self.overall_folder, self.agent_id)
        print('model_folder_name:', self.model_folder_name)

        if add_date_to_model_folder_name:
            self.model_folder_name = self.model_folder_name + "_date" + str(time_package.localtime().tm_mon) + "_" + str(time_package.localtime().tm_mday)
        
        self.get_related_folder_names_from_model_folder_name(self.model_folder_name, data_name=data_name)

        self.best_model_after_curriculum_dir_name = os.path.join(self.overall_folder, 'best_model_after_curriculum')

    
    def get_related_folder_names_from_model_folder_name(self, model_folder_name, data_name='data_0'):
        self.model_folder_name = model_folder_name
        self.processed_data_folder_path = os.path.join(model_folder_name.replace('all_agents', 'all_collected_data/processed_data'), f'individual_data_sessions/{data_name}')
        self.planning_data_folder_path = os.path.join(model_folder_name.replace('all_agents', 'all_collected_data/planning'), f'individual_data_sessions/{data_name}')
        self.patterns_and_features_data_folder_path = os.path.join(model_folder_name.replace('all_agents', 'all_collected_data/patterns_and_features'), f'individual_data_sessions/{data_name}')
        self.decision_making_folder_name = os.path.join(model_folder_name.replace('all_agents', 'all_collected_data/decision_making'), f'individual_data_sessions/{data_name}')

        os.makedirs(self.model_folder_name, exist_ok=True)
        os.makedirs(self.processed_data_folder_path, exist_ok=True)
        os.makedirs(self.planning_data_folder_path, exist_ok=True)
        os.makedirs(self.patterns_and_features_data_folder_path, exist_ok=True)
        os.makedirs(self.decision_making_folder_name, exist_ok=True)


    def get_current_info_condition(self, df):
        minimal_current_info = self.get_minimum_current_info()

        current_info_condition = df.any(axis=1)
        for key, value in minimal_current_info.items():
            current_info_condition = current_info_condition & (df[key] == value)
        return current_info_condition
    

    def store_env_params(self, model_folder_name=None, env_params_to_save=None):
        if model_folder_name is None:
            model_folder_name = self.model_folder_name
        if env_params_to_save is None:
            env_params_to_save = self.env_kwargs

        rl_for_multiff_utils.store_params(model_folder_name, env_params_to_save)


    def retrieve_env_params(self, model_folder_name=None):
        if model_folder_name is None:
            model_folder_name = self.model_folder_name        

        self.env_kwargs = rl_for_multiff_utils.retrieve_params(model_folder_name)
        return self.env_kwargs

    def curriculum_training(self, best_model_after_curriculum_exists_ok=True, load_replay_buffer_of_best_model_after_curriculum=True):
        try: 
            if not best_model_after_curriculum_exists_ok:
                raise Exception()
            self.load_best_model_after_curriculum(load_replay_buffer=load_replay_buffer_of_best_model_after_curriculum)
            print('Loaded best_model_after_curriculum')
        except Exception:
            print('Need to train a new best_model_after_curriculum')
            self._progress_in_curriculum() 
        self._run_current_agent_after_curriculum_training()


    def _progress_in_curriculum(self):
        
        os.makedirs(self.best_model_after_curriculum_dir_name, exist_ok=True)
        self.original_agent_id = self.agent_id
        self.agent_id = 'no_cost'
        print('Starting curriculum training')
        self._make_initial_env_for_curriculum_training()
        self._make_agent_for_curriculum_training()
        self.successful_training = False

        self._use_while_loop_for_curriculum_training()
        self._further_process_best_model_after_curriculum_training()
        self.streamline_making_animation(currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                                         video_dir=self.best_model_after_curriculum_dir_name)
        self.agent_id = self.original_agent_id

    def _make_env_suitable_for_curriculum_training(self):
        if self.sb3_or_lstm == 'sb3':
            self.env.env.reward_per_ff = 80
            self.env.env.dv_cost_factor = 0
            self.env.env.dw_cost_factor = 0
            self.env.env.w_cost_factor = 0
            self.env_kwargs_for_curriculum_training['reward_per_ff'] = 80
        else:
            #self.env.reward_per_ff = 80
            self.env.dv_cost_factor = 0
            self.env.dw_cost_factor = 0
            self.env.w_cost_factor = 0    

        self.env_kwargs_for_curriculum_training['dv_cost_factor'] = 0
        self.env_kwargs_for_curriculum_training['dw_cost_factor'] = 0
        self.env_kwargs_for_curriculum_training['w_cost_factor'] = 0

    def collect_data(self, n_steps=8000, exists_ok=False, save_data=False):
        
        if exists_ok:
            try:
                base_processing_class.BaseProcessing.retrieve_monkey_data(self)
                self.ff_caught_T_new = self.ff_caught_T_sorted
                self.make_or_retrieve_ff_dataframe_for_agent(exists_ok=exists_ok, save_data=save_data)
    
            except Exception as e:
                print("Failed to retrieve monkey data. Will make new monkey data. Error: ", e)
                self.run_agent_to_collect_data(n_steps=n_steps, save_data=save_data)
        else:
            self.run_agent_to_collect_data(n_steps=n_steps, save_data=save_data)

        self.make_or_retrieve_closest_stop_to_capture_df(exists_ok=exists_ok)
        # self.calculate_pattern_frequencies_and_feature_statistics()
        # self.find_patterns()


    def run_agent_to_collect_data(self, n_steps=8000, save_data=False):

        self.env_kwargs = self.retrieve_env_params()
        
        if self.sb3_or_lstm == 'sb3':
            self.env_for_data_collection = env_for_sb3.CollectInformation(**self.env_kwargs, episode_len=n_steps+100)
            LSTM = False
        elif self.sb3_or_lstm == 'lstm':
            self.env_for_data_collection = env_for_lstm.CollectInformationLSTM(**self.env_kwargs, episode_len=n_steps+100)
            LSTM = True
        else:
            raise ValueError("sb3_or_lstm should be either 'sb3' or 'lstm'")
        
        self._run_agent_to_collect_data(n_steps=n_steps, save_data=save_data, LSTM=LSTM)
        

    def _run_agent_to_collect_data(self, exists_ok=False, n_steps=8000, save_data=False, LSTM=False):

        # first, make self.processed_data_folder_path empty
        print('Collecting new agent data......')
        if os.path.exists(self.processed_data_folder_path):
            if not exists_ok:
                # if the folder is not empty, remove all files in the folder
                if len(os.listdir(self.processed_data_folder_path)) > 0:
                    # make the folder empty
                    os.system('rm -rf ' + self.processed_data_folder_path + '/*')
                    print('Removed all files in the folder:', self.processed_data_folder_path)

        self.n_steps = n_steps

        self.monkey_information, self.ff_flash_sorted, self.ff_caught_T_sorted, self.ff_believed_position_sorted, \
        self.ff_real_position_sorted, self.ff_life_sorted, self.ff_flash_end_sorted, self.caught_ff_num, self.total_ff_num, \
        self.obs_ff_indices_in_ff_dataframe, self.sorted_indices_all, self.ff_in_obs_df \
                = collect_agent_data_utils.collect_agent_data_func(self.env_for_data_collection, self.sac_model, n_steps=self.n_steps, LSTM=LSTM)
        self.ff_index_sorted = np.arange(len(self.ff_life_sorted))
        self.eval_ff_capture_rate = len(self.ff_flash_end_sorted)/self.monkey_information['time'].max()

        self.ff_caught_T_new = self.ff_caught_T_sorted

        if save_data:
            base_processing_class.BaseProcessing.save_ff_info_into_npz(self)
            base_processing_class.BaseProcessing.save_monkey_information(self)
            print("saved monkey_information and ff info at", (self.processed_data_folder_path))
        self.make_or_retrieve_ff_dataframe_for_agent(exists_ok=False, save_data=save_data)
        
        return 
    

    def make_or_retrieve_ff_dataframe_for_agent(self, exists_ok=False, save_data=False):
        self.ff_dataframe = None
        self.ff_dataframe_path = os.path.join(os.path.join(self.processed_data_folder_path, 'ff_dataframe.csv'))

        if exists_ok & exists(self.ff_dataframe_path):
            self.ff_dataframe = pd.read_csv(self.ff_dataframe_path)
        else:
            print('Warnings: currently, only ff in obs at each step are used in ff_dataframe. All ff are labeled \'visible\' regardless of their actual time since last visible.')
            if self.sb3_or_lstm == 'lstm':
                print('It is possible that the LSTM agent has the memory of ff in the past, but the code needs to be modified to reflect that. For planning analysis, info of in-memory ff is not needed.')

            self.make_ff_dataframe_from_ff_in_obs_df()
            #base_processing_class.BaseProcessing.make_or_retrieve_ff_dataframe(self, exists_ok=False, save_into_h5=False)
            print("made ff_dataframe")
            
            if save_data:
                self.ff_dataframe.to_csv(self.ff_dataframe_path)
                print("saved ff_dataframe at", self.ff_dataframe_path)
        return


    def make_ff_dataframe_from_ff_in_obs_df(self):
        self.ff_dataframe = self.ff_in_obs_df.copy()
        self.ff_dataframe['visible'] = 1

        make_ff_dataframe.add_essential_columns_to_ff_dataframe(self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted)
        self.ff_dataframe = make_ff_dataframe.process_ff_dataframe(self.ff_dataframe, max_distance=None, max_time_since_last_vis=3)


    def streamline_getting_data_from_agent(self, n_steps=8000, exists_ok=False, save_data=False, load_replay_buffer=False, **env_kwargs):
        if exists_ok:
            try:
                base_processing_class.BaseProcessing.retrieve_monkey_data(self)
                self.make_or_retrieve_ff_dataframe_for_agent(exists_ok=exists_ok, save_data=save_data)
                return
            except Exception as e:
                print("Failed to retrieve monkey data. Will make new monkey data. Error: ", e)

        self.env_kwargs = self.retrieve_env_params()
        self.env_kwargs.update(env_kwargs)
        self.make_env(**self.env_kwargs)
        self.make_agent()
        self.load_agent(load_replay_buffer=load_replay_buffer)
        self.collect_data(n_steps=n_steps, exists_ok=exists_ok, save_data=save_data)


    def streamline_loading_and_making_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000):
        try:
            self.env
        except AttributeError:
            self.make_env(**self.env_kwargs)

        try:
            self.sac_model
        except AttributeError:
            self.make_agent()
        self.load_agent(load_replay_buffer=False)
        self.streamline_making_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation, 
                                         duration=duration, n_steps=n_steps, file_name=None)

    def streamline_making_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000, file_name=None, video_dir=None):
        self.collect_data(n_steps=n_steps)
        #if len(self.ff_caught_T_new) >= currentTrial_for_animation:
        self.make_animation(currentTrial_for_animation=currentTrial_for_animation, num_trials_for_animation=num_trials_for_animation, 
                            duration=duration, file_name=file_name, video_dir=video_dir)


    def make_animation(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], file_name=None, video_dir=None, max_num_frames=150):
        self.set_animation_parameters(currentTrial=currentTrial_for_animation, num_trials=num_trials_for_animation, 
                                        k=1, duration=duration, max_num_frames=max_num_frames)
        self.call_animation_function(file_name=file_name, video_dir=video_dir)


    def streamline_everything(self, currentTrial_for_animation=None, num_trials_for_animation=None, duration=[10, 40], n_steps=8000,
                              use_curriculum_training=True, load_replay_buffer_of_best_model_after_curriculum=True,
                              best_model_after_curriculum_exists_ok=True, 
                              model_exists_ok=True):
        
        self.family_of_agents_log = rl_for_multiff_utils.retrieve_or_make_family_of_agents_log(self.overall_folder)
        # to_load_agent, to_train_agent = self.check_with_family_of_agents_log()
        # if (not to_load_agent) & (not to_train_agent):
        #     print("The set of parameters has failed to produce a well-trained agent in the past. \
        #            Skip to the next set of parameters")
        #     return
        
        to_load_agent = model_exists_ok
        
        self.make_env(**self.env_kwargs)
        self.make_agent()
        
        
        to_train_agent = True
        if to_load_agent:
            print("Loaded existing agent")
            try:
                self.load_agent(load_replay_buffer=False)
                to_train_agent = False
            except Exception as e:
                print("Failed to load existing agent. Need to train a new agent. Error: ", e)
                to_train_agent = True
        
        
        if to_train_agent:
            self.train_agent(use_curriculum_training=use_curriculum_training, best_model_after_curriculum_exists_ok=best_model_after_curriculum_exists_ok,
                             load_replay_buffer_of_best_model_after_curriculum=load_replay_buffer_of_best_model_after_curriculum)
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


    def train_agent(self, use_curriculum_training=True, best_model_after_curriculum_exists_ok=True, 
                    load_replay_buffer_of_best_model_after_curriculum=True, timesteps=1000000):
        
        self.training_start_time = time_package.time()
        if not use_curriculum_training:
            print('Starting regular training')
            self.regular_training(timesteps=timesteps)
        else:
            self.curriculum_training(best_model_after_curriculum_exists_ok=best_model_after_curriculum_exists_ok,
                                     load_replay_buffer_of_best_model_after_curriculum=load_replay_buffer_of_best_model_after_curriculum)
        self.training_time = time_package.time()-self.training_start_time
        print("Finished training using", self.training_time, 's.')

        #self.sac_model.save_replay_buffer(os.path.join(self.model_folder_name, 'buffer')) # I added this
        self.current_info_condition = self.get_current_info_condition(self.family_of_agents_log)
        self.family_of_agents_log.loc[self.current_info_condition, 'finished_training'] = True
        self.family_of_agents_log.loc[self.current_info_condition, 'training_time'] += self.training_time
        self.family_of_agents_log.loc[self.current_info_condition, 'successful_training'] += self.successful_training
        self.family_of_agents_log.to_csv(self.overall_folder + 'family_of_agents_log.csv')
        #Also check if the information is in parameters_record. If not, add it.
        #self.check_and_update_parameters_record()


    def _evaluate_model_and_retrain_if_necessary(self, use_curriculum_training=False):
        
        self.collect_data(n_steps=n_steps)
        
        if len(self.ff_caught_T_new) < 1:
            print("No firefly was caught by the agent during testing. Re-train agent.") 
            self.train_agent(use_curriculum_training=use_curriculum_training)
            if not self.successful_training:
                print("The set of parameters has failed to produce a well-trained agent in the past. \
                        Skip to the next set of parameters")
                raise ValueError("The set of parameters has failed to produce a well-trained agent in the past. Skip to the next set of parameters") 
            if len(self.ff_caught_T_new) < 1:
                print("No firefly was caught by the agent during testing again. Abort: ") 
                raise ValueError("Still no firefly was caught by the agent during testing after retraining. Abort: ")
        
        super().make_or_retrieve_ff_dataframe(exists_ok=False, data_folder_name=None, save_into_h5=False)
        super().find_patterns()
        self.calculate_pattern_frequencies_and_feature_statistics()
                
        return 


    def _make_plots_for_the_model(self, currentTrial_for_animation, num_trials_for_animation, duration=None):
        if currentTrial_for_animation >= len(self.ff_caught_T_new):
            currentTrial_for_animation = len(self.ff_caught_T_new)-1
            num_trials_for_animation = min(len(self.ff_caught_T_new)-1, 5)
        
        self.annotation_info = animation_utils.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials, \
                                                self.ignore_sudden_flash_indices, self.GUAT_indices_df['point_index'].values, self.try_a_few_times_indices)        
        self.set_animation_parameters(currentTrial=currentTrial_for_animation, num_trials=num_trials_for_animation, k=1, duration=duration)
        self.call_animation_function(save_video=True, fps=None, video_dir=self.overall_folder + 'all_videos', plot_flash_on_ff=True)
        # self.combine_6_plots_for_neural_network()
        # #self.plot_side_by_side()
        # self.save_plots_in_data_folders()
        # self.save_plots_in_plot_folders()
        return 
                
    def whether_to_update_record_and_make_plots(self):

        pattern_frequencies_record = pd.read_csv(self.overall_folder + 'pattern_frequencies_record.csv').drop(["Unnamed: 0"], axis=1)
        self.current_info_condition_for_pattern_frequencies = self.get_current_info_condition(pattern_frequencies_record)
        to_update_record = len(pattern_frequencies_record.loc[self.current_info_condition_for_pattern_frequencies]) == 0
    
        to_make_plots = (not exists(os.path.join(self.patterns_and_features_data_folder_path, 'compare_pattern_frequencies.png')))\
                        or (not exists(self.overall_folder + 'all_compare_pattern_frequencies/'+self.agent_id + '.png'))
                        
        return to_update_record, to_make_plots


    def save_plots_in_data_folders(self):
        plot_statistics.plot_pattern_frequencies(self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True, data_folder_name=self.patterns_and_features_data_folder_path)
        plot_statistics.plot_feature_statistics(self.agent_monkey_feature_statistics, compare_monkey_and_agent=True, data_folder_name = self.patterns_and_features_data_folder_path)

        plot_statistics.plot_feature_histograms_for_monkey_and_agent(self.all_trial_features_valid_m, self.all_trial_features_valid, data_folder_name = self.patterns_and_features_data_folder_path)
        print("Made new plots")


    def save_plots_in_plot_folders(self):
        plot_statistics.plot_pattern_frequencies(self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True, 
                                    data_folder_name = os.path.join(self.overall_folder, 'all_' + 'compare_pattern_frequencies'),
                                    file_name = self.agent_id + '.png')
        plot_statistics.plot_feature_statistics(self.agent_monkey_feature_statistics, compare_monkey_and_agent=True, 
                                data_folder_name = os.path.join(self.overall_folder, 'all_' + 'compare_feature_statistics'),
                                file_name = self.agent_id + '.png')
        plot_statistics.plot_feature_histograms_for_monkey_and_agent(self.all_trial_features_valid_m, self.all_trial_features_valid, 
                                                        data_folder_name = os.path.join(self.overall_folder, 'all_' + 'feature_histograms'),
                                                        file_name = self.agent_id + '.png')
    
    def get_minimum_current_info(self):
        minimal_current_info = {'dv_cost_factor': self.env_kwargs['dv_cost_factor'],
                                'dw_cost_factor': self.env_kwargs['dw_cost_factor'],
                                'w_cost_factor': self.env_kwargs['w_cost_factor']}
        
        # minimal_current_info = {'action_noise_std': self.action_noise_std, 
        #                         'ffxy_noise_std': self.ffxy_noise_std, 
        #                         'num_obs_ff': self.num_obs_ff, 
        #                         'max_in_memory_time': self.max_in_memory_time}
        return minimal_current_info


    def check_with_family_of_agents_log(self) -> bool:
        self.current_info_condition = self.get_current_info_condition(self.family_of_agents_log)
        self.minimal_current_info = self.get_minimum_current_info()
        retrieved_current_info = self.family_of_agents_log.loc[self.current_info_condition]
        

        exist_best_model = exists(os.path.join(self.model_folder_name, 'best_model.zip'))
        finished_training = np.any(retrieved_current_info['finished_training'])
        print('exist_best_model', exist_best_model)
        print('finished_training', finished_training)
        
        self.successful_training = np.any(retrieved_current_info['successful_training'])
        
        if finished_training & (not self.successful_training):
            # That's the indication that the set of parameters cannot be used to train a good agent
            to_load_agent = False
            to_train_agent = False         
        elif exist_best_model & finished_training:
            # Then we don't have to train the agent; go to the next set of parameters
            to_load_agent = True
            to_train_agent = False
        elif exist_best_model:
            # It seems like we have begun training the agent before, and we need to continue to train
            to_load_agent = True
            to_train_agent = True
        else:
            # Need to put in the new set of information
            additional_current_info =  {'finished_training': False,
                                        'year': time_package.localtime().tm_year, 
                                        'month': time_package.localtime().tm_mon,  
                                        'date': time_package.localtime().tm_mday, 
                                        'training_time': 0}
            current_info = {**self.minimal_current_info, **additional_current_info}
            
            self.family_of_agents_log = pd.concat([self.family_of_agents_log, pd.DataFrame(current_info, index=[0])]).reset_index(drop=True)
            self.family_of_agents_log.to_csv(self.overall_folder + 'family_of_agents_log.csv')
            to_load_agent = False
            to_train_agent = True 

        self.current_info_condition = self.get_current_info_condition(self.family_of_agents_log)        
        return to_load_agent, to_train_agent


    def check_and_update_parameters_record(self):
        self.parameters_record = pd.read_csv(self.overall_folder + 'parameters_record.csv').drop(["Unnamed: 0"], axis=1)
        self.current_info_condition = self.get_current_info_condition(self.parameters_record)
        retrieved_current_info = self.parameters_record.loc[self.current_info_condition]
        if len(retrieved_current_info) == 0:
            # Need to put in the new set of information
            additional_current_info =  {'working': 9}
            self.minimal_current_info = self.get_minimum_current_info()
            current_info = {**self.minimal_current_info, **additional_current_info}
            self.parameters_record = pd.concat([self.parameters_record, pd.DataFrame(current_info, index=[0])]).reset_index(drop=True)
            self.parameters_record.to_csv(self.overall_folder + 'parameters_record.csv')
    

    def call_animation_function(self, margin=100, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True, plot_flash_on_ff=False):
        self.obs_ff_indices_in_ff_dataframe_dict = None
        #self.obs_ff_indices_in_ff_dataframe_dict = {index: self.obs_ff_indices_in_ff_dataframe[index].astype(int) for index in range(len(self.obs_ff_indices_in_ff_dataframe))}
        
        if file_name is None:
            try:
                file_name=self.agent_id + f'__{self.currentTrial-self.num_trials+1}-{self.currentTrial}'
            except TypeError:
                file_name=self.agent_id + f'__{self.duration[0]}s_to_{self.duration[1]}s'
        # try adding ff capture rate to the file name
        try:
            file_name = file_name + f'_rate_{round(self.eval_ff_capture_rate, 2)}'
        except AttributeError:
            pass

        file_name = file_name + '.mp4'
        
        if self.sb3_or_lstm == 'sb3':
            dt = self.env.env.dt
        else:
            dt = self.env.dt

        super().call_animation_function(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position, 
                                        set_xy_limits=set_xy_limits, plot_flash_on_ff=plot_flash_on_ff, in_obs_ff_dict=self.obs_ff_indices_in_ff_dataframe_dict,
                                        fps=int((1/dt)/self.k))
        

    def make_animation_with_annotation(self, margin=100, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True):
        super().make_animation_with_annotation(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits)


    def combine_6_plots_for_neural_network(self):

        # self.add_2nd_ff = False if if self.num_obs_ff < 2 else True

        self.add_2nd_ff = False    
        interpret_neural_network.combine_6_plots_for_neural_network(self.sac_model, full_memory = self.full_memory, invisible_distance = self.invisible_distance, 
                                            add_2nd_ff = self.add_2nd_ff, data_folder_name = self.patterns_and_features_data_folder_path, const_memory = self.full_memory,
                                            data_folder_name2 = self.overall_folder + 'all_' + 'combined_6_plots_for_neural_network',
                                            file_name2 = self.agent_id + '.png')
            



    def import_monkey_data(self, info_of_monkey, all_trial_features_m, pattern_frequencies_m, feature_statistics_m):
        self.info_of_monkey = info_of_monkey
        self.all_trial_features_m = all_trial_features_m
        self.all_trial_features_valid_m = self.all_trial_features_m[(self.all_trial_features_m['t_last_vis']<50) & (self.all_trial_features_m['hitting_arena_edge']==False)].reset_index()    
        self.pattern_frequencies_m = pattern_frequencies_m
        self.feature_statistics_m = feature_statistics_m



    def calculate_pattern_frequencies_and_feature_statistics(self):
        self.make_or_retrieve_all_trial_features()
        self.all_trial_features_valid = self.all_trial_features[(self.all_trial_features['t_last_vis']<50) & (self.all_trial_features['hitting_arena_edge']==False)].reset_index()    
        self.make_or_retrieve_all_trial_patterns()
        self.make_or_retrieve_pattern_frequencies()
        self.make_or_retrieve_feature_statistics()

        self.pattern_frequencies_a = self.pattern_frequencies
        self.feature_statistics_a = self.feature_statistics
        self.agent_monkey_pattern_frequencies = organize_patterns_and_features.combine_df_of_agent_and_monkey(self.pattern_frequencies_m, self.pattern_frequencies_a, agent_names = ["Agent", "Agent2", "Agent3"])
        self.agent_monkey_feature_statistics = organize_patterns_and_features.combine_df_of_agent_and_monkey(self.feature_statistics_m, self.feature_statistics_a, agent_names = ["Agent", "Agent2", "Agent3"])  
        
        SB3_functions.add_row_to_pattern_frequencies_record(self.pattern_frequencies, self.minimal_current_info, self.overall_folder)
        SB3_functions.add_row_to_feature_medians_record(self.feature_statistics, self.minimal_current_info, self.overall_folder)
        SB3_functions.add_row_to_feature_means_record(self.feature_statistics, self.minimal_current_info, self.overall_folder)


    def plot_side_by_side(self):
        with basic_func.HiddenPrints():
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
                info_of_agent, plot_whole_duration, rotation_matrix, num_imitation_steps_monkey, num_imitation_steps_agent = collect_agent_data.find_corresponding_info_of_agent(self.info_of_monkey, currentTrial, num_trials, self.sac_model, self.agent_dt, LSTM=False, env_kwargs=self.env_kwargs)

                with basic_func.initiate_plot(20,20,400):
                    additional_plots.PlotSidebySide(plot_whole_duration = plot_whole_duration,
                                    info_of_monkey = self.info_of_monkey,
                                    info_of_agent = info_of_agent,  
                                    num_imitation_steps_monkey = num_imitation_steps_monkey,
                                    num_imitation_steps_agent = num_imitation_steps_agent,                
                                    currentTrial = currentTrial,
                                    num_trials = num_trials, 
                                    rotation_matrix = rotation_matrix,              
                                    plotting_params = plotting_params,
                                    data_folder_name = self.patterns_and_features_data_folder_path
                                    )



