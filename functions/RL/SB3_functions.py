from multiff_analysis.functions.RL import env, collect_agent_data, interpret_neural_network
from multiff_analysis.functions.data_wrangling import basic_func, data_processing_class, analyze_patterns_and_features
from multiff_analysis.functions.data_visualization import plot_behaviors, plot_statistics
import time as time_package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from functools import partial
from os.path import exists
import time as time_package
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def test_agent(env, obs, model, n_steps = 10000):
    # Test the trained agent
    obs = env.reset()
    cum_rewards = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_rewards += reward
        if done:
            obs = env.reset()
        # print(step, ffxy_visible[-1])
    return cum_rewards





class TrialEvalCallback(EvalCallback):
    """
    Original source: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
    Provided by Optuna.
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):

        super(TrialEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq,
                                                deterministic=deterministic,
                                                verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True






class SaveOnBestTrainingRewardCallback(BaseCallback):
    """ Taken from StableBaslines3, except that best_mean_reward is renamed best_mean_traing_reward, 
        so that's its easier to be combined with another class later
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_traing_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(" ")
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Current mean training reward per episode: {mean_reward:.2f} compared to best mean training reward on record: {self.best_mean_traing_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_traing_reward:
                  self.best_mean_traing_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
                  #self.model.save_replay_buffer(os.path.join(self.log_dir, 'buffer')) # I added this
        return True






class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    SOURCE: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.
    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.
    It must be used with the ``EvalCallback``.
    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model

    modification: added the param log_dir, so that the callback can access the training reward history
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0, 
                 log_dir=None, overall_folder=None, agent_id=None):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.log_dir = log_dir
        self.overall_folder = overall_folder
        self.agent_id = agent_id

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                # My notes: here the parent means EvalCallback, because somewhere else it's designated that the parent of the current class (StopTrainingOnNoModelImprovement) is EvalCallback
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if not continue_training:
            # This is added code to the orginal code
            if self.log_dir is not None:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Rewards')
                fig.savefig(os.path.join(self.log_dir, 'training_rewards.png'))
                if (self.overall_folder is not None) and (self.agent_id is not None):
                    os.makedirs(os.path.join(self.overall_folder, 'all_training_rewards'), exist_ok=True)
                    fig.savefig(os.path.join(self.overall_folder, 'all_training_rewards', self.agent_id + '.png'))
                plt.close(fig)

            if self.verbose >= 1:
                print(
                    f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
                )

        return continue_training





class SaveOnBestTrainingRewardAndStopTrainingOnNoTestingRewardImprovement(SaveOnBestTrainingRewardCallback, 
                                                                          StopTrainingOnNoModelImprovement):
    """
    This class combines SaveOnBestTrainingRewardCallback and StopTrainingOnNoModelImprovement

    In reality, this new class is not useful, because I can just add best_model_save_path to EvalCallback which can also call StopTrainingOnNoModelImprovement. 
    
    Example:
    stop_train_callback = SaveOnBestTrainingRewardAndStopTrainingOnNoTestingRewardImprovement(max_no_improvement_evals=10, 
                            min_evals=100, verbose=1, check_freq=20000, log_dir=self.log_dir)
    """


    def __init__(self, log_dir: str, check_freq: int, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        SaveOnBestTrainingRewardCallback.__init__(self, check_freq=check_freq, log_dir=log_dir, verbose=verbose)
        StopTrainingOnNoModelImprovement.__init__(self, max_no_improvement_evals=max_no_improvement_evals, 
                                                  min_evals=min_evals, verbose=verbose)


    def _on_step(self) -> bool:
        SaveOnBestTrainingRewardCallback._on_step(self)
        continue_training = StopTrainingOnNoModelImprovement._on_step(self)
        return continue_training






def add_row_to_record(df, csv_name, value_name, current_info, overall_folder):
    new_row = df[['Item', value_name]].set_index('Item').T.reset_index(drop=True)
    new_row = pd.DataFrame(current_info, index=[0]).join(new_row)
    df = pd.read_csv(f'{overall_folder}{csv_name}.csv').drop(["Unnamed: 0"], axis=1)
    df = pd.concat([df, new_row], axis=0).reset_index(drop=True)
    df.to_csv(f'{overall_folder}/{csv_name}.csv') 
    


def add_row_to_pattern_frequencies_record(pattern_frequencies, current_info, overall_folder):
    add_row_to_record(df=pattern_frequencies, csv_name='pattern_frequencies_record', value_name='Rate', current_info=current_info, overall_folder=overall_folder)

def add_row_to_feature_medians_record(feature_statistics, current_info, overall_folder):
    add_row_to_record(df=feature_statistics, csv_name='feature_medians_record', value_name='Median', current_info=current_info, overall_folder=overall_folder) 
    
def add_row_to_feature_means_record(feature_statistics, current_info, overall_folder):
    add_row_to_record(df=feature_statistics, csv_name='feature_means_record', value_name='Mean', current_info=current_info, overall_folder=overall_folder)    
    












class RLforMultifirefly(data_processing_class.BaseProcessing):

    def __init__(self, overall_folder, action_noise_std=0.1, ffxy_noise_std=4, num_obs_ff=2, full_memory=3, add_date_to_log_dir=False):
        self.action_noise_std = action_noise_std
        self.ffxy_noise_std = ffxy_noise_std
        self.num_obs_ff = num_obs_ff
        self.full_memory = full_memory
        self.invisible_distance = 400 
        self.agent_dt = 0.25 
        self.agent_id = "A" + str(action_noise_std) + "_O" + str(ffxy_noise_std) + \
                        "_ff" + str(num_obs_ff) + "_M" + str(full_memory) 
        self.overall_folder = overall_folder
        self.log_dir = os.path.join(self.overall_folder, 'all_agents', self.agent_id)
        print('log_dir:', self.log_dir)
        if add_date_to_log_dir:
            self.log_dir = self.log_dir + "_date" + str(time_package.localtime().tm_mon) + "_" + str(time_package.localtime().tm_mday)
        self.data_folder_name = self.log_dir
        self.player = "agent"
        self.env_kwargs = {"action_noise_std": self.action_noise_std, 
                          "ffxy_noise_std": self.ffxy_noise_std, 
                          "num_obs_ff": self.num_obs_ff, 
                          "full_memory": self.full_memory,
                          "dt": self.agent_dt,
                          "print_ff_capture_incidents": False,
                          "print_episode_reward_rates": True}


    def update_current_info_condition(self, df):
        current_info_condition = ((df['action_noise_std'] == self.action_noise_std) &
                                    (df['ffxy_noise_std'] == self.ffxy_noise_std) &
                                    (df['num_obs_ff'] == self.num_obs_ff) &
                                    (df['full_memory'] == self.full_memory))
        return current_info_condition


    def streamline_everything(self, currentTrial_for_animation, num_trials_for_animation, n_steps=8000):
        to_load_agent, to_train_agent = self.check_with_family_of_agents_log()
        if (not to_load_agent) & (not to_train_agent):
            print("The set of parameters has failed to produce a well-trained agent in the past. \
                  Skip to the next set of parameters")
            return
        self.make_env()
        self.make_agent()
        if to_load_agent:
            print("Loaded existing agent")
            self.load_agent(load_replay_buffer=False)
        else: 
            to_train_agent = True
            print("Made new agent")
        
        if to_train_agent:
            self.train_agent()
            if self.successful_training == False:
                print("The set of parameters has failed to produce a well-trained agent in the past. \
                    Skip to the next set of parameters")
                return                


        to_update_record, to_make_plots = self.whether_to_update_record_and_make_plots()

        
        if to_make_plots or to_update_record:
            self.collect_data(n_steps=n_steps)
            if len(self.ff_caught_T_sorted) < 1:
                print("No firefly was caught by the agent during testing. Re-train agent.") 
                self.train_agent()
                if self.successful_training == False:
                    print("The set of parameters has failed to produce a well-trained agent in the past. \
                        Skip to the next set of parameters")
                    return  
                if len(self.ff_caught_T_sorted) < 1:
                    print("No firefly was caught by the agent during testing again. Abort: ") 
                    print(self.log_dir)
                    return
            if currentTrial_for_animation >= len(self.ff_caught_T_sorted):
                currentTrial_for_animation = len(self.ff_caught_T_sorted)-1
                num_trials_for_animation = min(len(self.ff_caught_T_sorted)-1, 5)
            
            super().make_or_retrieve_ff_dataframe(exists_ok=False)
            super().find_patterns()
            self.calculate_pattern_frequencies_and_feature_statistics()
            
            if to_make_plots:
                # self.annotation_info = animation_func.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials, \
                #                                         self.ignore_sudden_flash_indices, self.give_up_after_trying_indices, self.try_a_few_times_indices)        
                # self.set_animation_parameters(currentTrial=currentTrial_for_animation, num_trials=num_trials_for_animation, k=1)
                # self.make_animation(save_video=True, video_dir=self.overall_folder + 'all_videos', file_name=self.agent_id + '.mp4', plot_flash_on_ff=True)
                self.interpret_neural_network.combine_6_plots_for_neural_network()
                #self.plot_side_by_side()
                self.save_plots_in_data_folders()
                self.save_plots_in_plot_folders()
        else:
            print("Plots and record already exist. No need to make new ones.")




    def whether_to_update_record_and_make_plots(self):

        pattern_frequencies_record = pd.read_csv(self.overall_folder + 'pattern_frequencies_record.csv').drop(["Unnamed: 0"], axis=1)
        self.current_info_condition_for_pattern_frequencies = self.update_current_info_condition(pattern_frequencies_record)
        to_update_record = len(pattern_frequencies_record.loc[self.current_info_condition_for_pattern_frequencies]) == 0
    
        to_make_plots = (not exists(os.path.join(self.log_dir, 'compare_pattern_frequencies.png')))\
                        or (not exists(self.overall_folder + 'all_compare_pattern_frequencies/'+self.agent_id + '.png'))
        return to_update_record, to_make_plots


    def save_plots_in_data_folders(self):
        plot_statistics.plot_pattern_frequencies(self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True, data_folder_name=self.log_dir)
        plot_statistics.plot_feature_statistics(self.agent_monkey_feature_statistics, compare_monkey_and_agent=True, data_folder_name = self.log_dir)

        plot_statistics.plot_feature_histograms_for_monkey_and_agent(self.all_trial_features_valid_m, self.all_trial_features_valid, data_folder_name = self.log_dir)
        print("Made new plots")


    def save_plots_in_plot_folders(self):
        plot_statistics.plot_pattern_frequencies(self.agent_monkey_pattern_frequencies, compare_monkey_and_agent=True, 
                                    data_folder_name=self.overall_folder + 'all_' + 'compare_pattern_frequencies',
                                    file_name = self.agent_id + '.png')
        plot_statistics.plot_feature_statistics(self.agent_monkey_feature_statistics, compare_monkey_and_agent=True, 
                                data_folder_name=self.overall_folder + 'all_' + 'compare_feature_statistics',
                                file_name = self.agent_id + '.png')
        plot_statistics.plot_feature_histograms_for_monkey_and_agent(self.all_trial_features_valid_m, self.all_trial_features_valid, 
                                                        data_folder_name=self.overall_folder + 'all_' + 'feature_histograms',
                                                        file_name = self.agent_id + '.png')
    
        


    def check_with_family_of_agents_log(self) -> bool:
        self.family_of_agents_log = pd.read_csv(self.overall_folder + 'family_of_agents_log.csv').drop(["Unnamed: 0"], axis=1)
        self.current_info_condition = self.update_current_info_condition(self.family_of_agents_log)
        self.minimal_current_info = {'action_noise_std': self.action_noise_std, 
                                     'ffxy_noise_std': self.ffxy_noise_std, 
                                     'num_obs_ff': self.num_obs_ff, 
                                     'full_memory': self.full_memory}
        retrieved_current_info = self.family_of_agents_log.loc[self.current_info_condition]
        

        exist_best_model = exists(os.path.join(self.log_dir, 'best_model.zip'))
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

        self.current_info_condition = self.update_current_info_condition(self.family_of_agents_log)        
        return to_load_agent, to_train_agent



    def make_env(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.env = env.MultiFF(**self.env_kwargs)
        self.env = Monitor(self.env, self.log_dir)



    def make_agent(self):
        self.sac_model = SAC("MlpPolicy", 
                    self.env,
                    gamma=0.995,
                    learning_rate=0.0015,
                    batch_size=1024,
                    target_update_interval=50,
                    buffer_size=1000000,
                    learning_starts=10000,
                    train_freq=10,
                    ent_coef=0.00083,
                    policy_kwargs=dict(activation_fn=nn.Tanh, net_arch=[32, 32]))
        


    def train_agent(self):
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=20, verbose=1, log_dir=self.log_dir,
        #                                                        overall_folder=self.overall_folder, agent_id=self.agent_id)
        # self.callback = EvalCallback(self.env, eval_freq=12000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path=self.log_dir, n_eval_episodes=3)
        # timesteps = 50000000
        # self.training_start_time = time_package.time()
        # self.sac_model.learn(total_timesteps=int(timesteps), callback=self.callback)
        # self.training_time = time_package.time()-self.training_start_time

        self.curriculum_training()
        print("Finished training using", self.training_time, 's.')
        self.sac_model.save(os.path.join(self.log_dir, 'best_model'))
        #self.sac_model.save_replay_buffer(os.path.join(self.log_dir, 'buffer')) # I added this
        self.current_info_condition = self.update_current_info_condition(self.family_of_agents_log)
        self.family_of_agents_log.loc[self.current_info_condition, 'finished_training'] = True
        self.family_of_agents_log.loc[self.current_info_condition, 'training_time'] += self.training_time
        self.family_of_agents_log.loc[self.current_info_condition, 'successful_training'] += self.successful_training
        self.family_of_agents_log.to_csv(self.overall_folder + 'family_of_agents_log.csv')
        #Also check if the information is in parameters_record. If not, add it.
        self.check_and_update_parameters_record()


    def curriculum_training(self):
        self.successful_training = False
        self.env.env.linear_terminal_vel = 0.01
        self.env.env.angular_terminal_vel = 1
        timesteps = 10000000
        while self.env.env.angular_terminal_vel > 0.00222: #0.0035/(pi/2), same as the monkey's threshold
            stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=1000) # or 10ff/250s  
            callback = EvalCallback(self.env, eval_freq=8000, callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=3)
            self.sac_model.learn(total_timesteps=int(timesteps), callback=callback)
            if callback.best_mean_reward < 500:
                break
            self.env.env.angular_terminal_vel = max(self.env.env.angular_terminal_vel/2, 0.00222)
            print('Current angular_terminal_vel:', self.env.env.angular_terminal_vel)
        while self.env.env.linear_terminal_vel > 0.0005:
            stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=1000) # or 10ff/250s  
            callback = EvalCallback(self.env, eval_freq=8000, callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=3)
            self.sac_model.learn(total_timesteps=int(timesteps), callback=callback)
            if callback.best_mean_reward < 500:
                break
            self.env.env.linear_terminal_vel = max(self.env.env.linear_terminal_vel/2, 0.0005)
            print('Current linear_terminal_vel:', self.env.env.linear_terminal_vel)
        # Now, the condition has restored to the original condition. We shall store the trained agent in the current condition.
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=20, verbose=1, log_dir=self.log_dir,                                                   overall_folder=self.overall_folder, agent_id=self.agent_id)
        callback = EvalCallback(self.env, eval_freq=12000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path=self.log_dir, n_eval_episodes=3)
        self.sac_model.learn(total_timesteps=int(timesteps), callback=callback)
        self.successful_training = True



    def check_and_update_parameters_record(self):
        self.parameters_record = pd.read_csv(self.overall_folder + 'parameters_record.csv').drop(["Unnamed: 0"], axis=1)
        self.current_info_condition = self.update_current_info_condition(self.parameters_record)
        retrieved_current_info = self.parameters_record.loc[self.current_info_condition]
        if len(retrieved_current_info) == 0:
            # Need to put in the new set of information
            additional_current_info =  {'working': 9}
            current_info = {**self.minimal_current_info, **additional_current_info}
            self.parameters_record = pd.concat([self.parameters_record, pd.DataFrame(current_info, index=[0])]).reset_index(drop=True)
            self.parameters_record.to_csv(self.overall_folder + 'parameters_record.csv')


    def load_agent(self, load_replay_buffer=True):
        path = os.path.join(self.log_dir, 'best_model.zip')
        if exists(path):
            self.sac_model = self.sac_model.load(path, env = self.env) 
            print("Loaded existing agent")

        if load_replay_buffer:
            path2 = os.path.join(self.log_dir, 'buffer')
            if exists(path2):
                self.sac_model.load_replay_buffer(path2)


    def collect_data(self, n_steps = 8000):
        self.env_for_data_collection = env.CollectInformation(**self.env_kwargs)
        self.monkey_information, self.ff_flash_sorted, self.ff_caught_T_sorted, self.ff_believed_position_sorted, \
                  self.ff_real_position_sorted, self.ff_life_sorted, self.ff_flash_end_sorted, self.caught_ff_num, self.total_ff_num, \
                  self.obs_ff_indices_in_ff_dataframe, self.sorted_indices_all, self.ff_noisy_xy_in_obs \
                  = collect_agent_data.collect_agent_data_func(self.env_for_data_collection, self.sac_model, n_steps=n_steps, LSTM = False)



    def set_animation_parameters(self, currentTrial, num_trials, k):
        super().set_animation_parameters(currentTrial, num_trials, k)
        


    def make_animation(self, margin=400, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True, plot_flash_on_ff=False):
        super().make_animation(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits, plot_flash_on_ff=plot_flash_on_ff)
        


    def make_animation_with_annotation(self, margin=400, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True):
        super().make_animation_with_annotation(margin=margin, save_video=save_video, video_dir=video_dir, file_name=file_name, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits)



    def combine_6_plots_for_neural_network(self):
        # if self.num_obs_ff >= 2:
        #     self.add_2nd_ff = True
        # else:
        #     self.add_2nd_ff = False

        self.add_2nd_ff = False    
        interpret_neural_network.combine_6_plots_for_neural_network(self.sac_model, full_memory = self.full_memory, invisible_distance = self.invisible_distance, 
                                            add_2nd_ff = self.add_2nd_ff, data_folder_name = self.log_dir, const_memory = self.full_memory,
                                            data_folder_name2 = self.overall_folder + 'all_' + 'combined_6_plots_for_neural_network',
                                            file_name2 = self.agent_id + '.png')
            



    def import_monkey_data(self, info_of_monkey, all_trial_features_m, pattern_frequencies_m, feature_statistics_m):
        self.info_of_monkey = info_of_monkey
        self.all_trial_features_m = all_trial_features_m
        self.all_trial_features_valid_m = self.all_trial_features_m[(self.all_trial_features_m['t_last_visible']<50) & (self.all_trial_features_m['hitting_arena_edge']==False)].reset_index()    
        self.pattern_frequencies_m = pattern_frequencies_m
        self.feature_statistics_m = feature_statistics_m



    def calculate_pattern_frequencies_and_feature_statistics(self):
        self.make_or_retrieve_all_trial_features()
        self.all_trial_features_valid = self.all_trial_features[(self.all_trial_features['t_last_visible']<50) & (self.all_trial_features['hitting_arena_edge']==False)].reset_index()    
        self.make_or_retrieve_all_trial_patterns()
        self.make_or_retrieve_pattern_frequencies()
        self.make_or_retrieve_feature_statistics()

        self.pattern_frequencies_a = self.pattern_frequencies
        self.feature_statistics_a = self.feature_statistics
        self.agent_monkey_pattern_frequencies = analyze_patterns_and_features.combine_df_of_agent_and_monkey(self.pattern_frequencies_m, self.pattern_frequencies_a, agent_names = ["Agent", "Agent2", "Agent3"])
        self.agent_monkey_feature_statistics = analyze_patterns_and_features.combine_df_of_agent_and_monkey(self.feature_statistics_m, self.feature_statistics_a, agent_names = ["Agent", "Agent2", "Agent3"])  
        
        add_row_to_pattern_frequencies_record(self.pattern_frequencies, self.minimal_current_info, self.overall_folder)
        add_row_to_feature_medians_record(self.feature_statistics, self.minimal_current_info, self.overall_folder)
        add_row_to_feature_means_record(self.feature_statistics, self.minimal_current_info, self.overall_folder)



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
                    plot_behaviors.PlotSidebySide(plot_whole_duration = plot_whole_duration,
                                    info_of_monkey = self.info_of_monkey,
                                    info_of_agent = info_of_agent,  
                                    num_imitation_steps_monkey = num_imitation_steps_monkey,
                                    num_imitation_steps_agent = num_imitation_steps_agent,                
                                    currentTrial = currentTrial,
                                    num_trials = num_trials, 
                                    rotation_matrix = rotation_matrix,              
                                    plotting_params = plotting_params,
                                    data_folder_name = self.log_dir
                                    )



