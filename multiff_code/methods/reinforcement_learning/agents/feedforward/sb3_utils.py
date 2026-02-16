import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import LoadMonitorResultsError

plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test_agent(env, obs, model, n_steps=10000):
    # Test the trained agent
    obs, _ = env.reset()
    cum_rewards = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        cum_rewards += reward
        if done:
            obs, _ = env.reset()
        # print(step, ffxy_visible[-1])
    return cum_rewards


class TrialEvalCallback(EvalCallback):
    """
    Original source: https://github.com/optuna/optuna-examples/blob/main/RL_models/sb3_simple.py
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
    :param model_folder_name: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, model_folder_name: str, verbose: int = 1, model_name='best_model'):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.model_folder_name = model_folder_name
        self.save_path = os.path.join(model_folder_name, model_name)
        self.best_mean_traing_reward = -np.inf

        # Create the monitor directory immediately
        os.makedirs(self.model_folder_name, exist_ok=True)

    def _create_empty_monitor_file(self):
        """Create an empty monitor file with proper format"""
        monitor_file = os.path.join(self.model_folder_name, 'monitor.csv')

        # Only create if it doesn't exist
        if not os.path.exists(monitor_file):
            # Create JSON header with current timestamp
            header = {
                "t_start": time.time(),
                "env_id": "custom_env"
            }

            # Write the monitor file with proper format
            with open(monitor_file, 'w') as f:
                # JSON header line
                f.write(f"#{json.dumps(header)}\n")
                # CSV header line
                f.write("r,l,t\n")

            if self.verbose > 0:
                print(f"Created empty monitor file at: {monitor_file}")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            # find the parent dir of self.save_path
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Also create the monitor directory to ensure it exists
        os.makedirs(self.model_folder_name, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            try:
                x, y = ts2xy(load_results(self.model_folder_name), 'timesteps')
            except LoadMonitorResultsError:
                # Create an empty monitor file if it doesn't exist
                self._create_empty_monitor_file()
                if self.verbose > 0:
                    print(f"No monitor files found in {self.model_folder_name}. "
                          f"Created empty monitor file. Will start tracking rewards once episodes complete.")
                return True
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(" ")
                    print(f"Current timesteps: {self.num_timesteps}")
                    print(
                        f"Current mean training reward per episode: {mean_reward:.2f} compared to best mean training reward on record: {self.best_mean_traing_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_traing_reward:
                    self.best_mean_traing_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    self.model.save_replay_buffer(os.path.join(
                        self.model_folder_name, 'buffer'))  # I added this
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

    modification: added the param model_folder_name, so that the callback can access the training reward history
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0,
                 model_folder_name=None, overall_folder=None, agent_id=None, best_model_save_path=None):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.model_folder_name = model_folder_name
        self.overall_folder = overall_folder
        self.agent_id = agent_id
        self.best_model_save_path = best_model_save_path

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                # My notes: here the parent means EvalCallback, because somewhere else it's designated that the parent of the current class (StopTrainingOnNoModelImprovement) is EvalCallback
                self.no_improvement_evals = 0
                # Also persist the replay buffer alongside the best model when we see a new best
                try:
                    if self.best_model_save_path is not None:
                        os.makedirs(self.best_model_save_path, exist_ok=True)
                        if hasattr(self.model, "save_replay_buffer"):
                            self.model.save_replay_buffer(os.path.join(self.best_model_save_path, 'buffer'))
                except Exception as e:
                    if self.verbose >= 1:
                        print('Warning: failed to save replay buffer with new best model:', e)
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if not continue_training:
            # This is added code to the orginal code
            if self.model_folder_name is not None:
                x, y = ts2xy(load_results(self.model_folder_name), 'timesteps')
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Rewards')
                fig.savefig(os.path.join(
                    self.model_folder_name, 'training_rewards.png'))
                if (self.overall_folder is not None) and (self.agent_id is not None):
                    os.makedirs(os.path.join(self.overall_folder,
                                'all_training_rewards'), exist_ok=True)
                    fig.savefig(os.path.join(self.overall_folder,
                                'all_training_rewards', self.agent_id + '.png'))
                plt.close(fig)

            if self.verbose >= 1:
                print(
                    f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
                )

        return continue_training


# def plot_training_rewards()
#     x, y = ts2xy(load_results(self.model_folder_name), 'timesteps')
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     ax.set_xlabel('Timesteps')
#     ax.set_ylabel('Rewards')
#     fig.savefig(os.path.join(self.model_folder_name, 'training_rewards.png'))
#     if (self.overall_folder is not None) and (self.agent_id is not None):
#         os.makedirs(os.path.join(self.overall_folder, 'all_training_rewards'), exist_ok=True)
#         fig.savefig(os.path.join(self.overall_folder, 'all_training_rewards', self.agent_id + '.png'))
#     plt.close(fig)


class SaveOnBestTrainingRewardAndStopTrainingOnNoTestingRewardImprovement(SaveOnBestTrainingRewardCallback,
                                                                          StopTrainingOnNoModelImprovement):
    """
    This class combines SaveOnBestTrainingRewardCallback and StopTrainingOnNoModelImprovement

    !!!!!!!!!!!!!!!!!!
    In reality, this new class is not useful, because I can just add best_model_save_path to EvalCallback which can also call StopTrainingOnNoModelImprovement. 

    Example:
    stop_train_callback = SaveOnBestTrainingRewardAndStopTrainingOnNoTestingRewardImprovement(max_no_improvement_evals=10, 
                            min_evals=100, verbose=1, check_freq=20000, model_folder_name=self.model_folder_name)
    """

    def __init__(self, model_folder_name: str, check_freq: int, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        SaveOnBestTrainingRewardCallback.__init__(
            self, check_freq=check_freq, model_folder_name=model_folder_name, verbose=verbose)
        StopTrainingOnNoModelImprovement.__init__(self, max_no_improvement_evals=max_no_improvement_evals,
                                                  min_evals=min_evals, verbose=verbose)

    def _on_step(self) -> bool:
        SaveOnBestTrainingRewardCallback._on_step(self)
        continue_training = StopTrainingOnNoModelImprovement._on_step(self)
        return continue_training


def add_row_to_record(df, csv_name, value_name, current_info, overall_folder):
    new_row = df[['item', value_name]].set_index(
        'item').T.reset_index(drop=True)
    new_row = pd.DataFrame(current_info, index=[0]).join(new_row)
    df = pd.read_csv(f'{overall_folder}{csv_name}.csv').drop(
        ["Unnamed: 0"], axis=1)
    df = pd.concat([df, new_row], axis=0).reset_index(drop=True)
    df.to_csv(f'{overall_folder}/{csv_name}.csv')


def add_row_to_pattern_frequencies_record(pattern_frequencies, current_info, overall_folder):
    add_row_to_record(df=pattern_frequencies, csv_name='pattern_frequencies_record',
                      value_name='rate', current_info=current_info, overall_folder=overall_folder)


def add_row_to_feature_medians_record(feature_statistics, current_info, overall_folder):
    add_row_to_record(df=feature_statistics, csv_name='feature_medians_record',
                      value_name='median', current_info=current_info, overall_folder=overall_folder)


def add_row_to_feature_means_record(feature_statistics, current_info, overall_folder):
    add_row_to_record(df=feature_statistics, csv_name='feature_means_record',
                      value_name='mean', current_info=current_info, overall_folder=overall_folder)
