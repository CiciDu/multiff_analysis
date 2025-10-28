import argparse
from importlib import reload
import gc
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial
from IPython.display import HTML
from torch.linalg import vector_norm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy import pi
import optuna
from gymnasium import spaces, Env
import pickle
import numpy as np
import torch
import sys
from pathlib import Path
import os
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from neural_data_analysis.neural_analysis_tools.model_neural_data import neural_data_modeling
from eye_position_analysis import eye_positions
from reinforcement_learning.agents.feedforward import interpret_neural_network, sb3_class, sb3_utils
from reinforcement_learning.agents.rnn import gru_utils, lstm_utils, lstm_utils, lstm_class, gru_class
from reinforcement_learning.base_classes import env_utils, base_env, more_envs, rl_base_class, rl_base_utils
from reinforcement_learning.base_classes import run_logger
from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils, hyperparam_tuning_class
from null_behaviors import sample_null_distributions, show_null_trajectory
from visualization.animation import animation_func, animation_utils, animation_class
from visualization.matplotlib_tools import plot_trials, plot_polar, additional_plots, plot_behaviors_utils, plot_statistics
from decision_making_analysis import free_selection, replacement, trajectory_info
from decision_making_analysis.GUAT import GUAT_helper_class, GUAT_collect_info_class, GUAT_combine_info_class, add_features_GUAT_and_TAFT
from decision_making_analysis.decision_making import decision_making_utils, plot_decision_making, intended_targets_classes
from decision_making_analysis.cluster_replacement import cluster_replacement_utils, plot_cluster_replacement
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from data_wrangling import specific_utils, process_monkey_information, base_processing_class


os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
plt.rcParams["animation.html"] = "html5"
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

device_idx = int(os.getenv("CUDA_DEVICE", "0"))
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"))
PLAYER = "agent"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate LSTM agent on Multifirefly env")
    # ------------------------------
    # Hyperparameter sweep arguments
    # ------------------------------
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the LSTM agent optimizer")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Number of hidden units in the LSTM")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    # ------------------------------
    # Environment and training arguments
    # ------------------------------
    parser.add_argument("--overall-folder", type=str, default='multiff_analysis/RL_models/LSTM_stored_models/all_agents/oct_18_3/',
                        help="Output directory to save models and buffers")
    parser.add_argument("--duration", type=int, nargs=2, default=[10, 40],
                        help="[min,max] steps for evaluation animations/checks")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Environment time step")
    parser.add_argument("--num-obs-ff", type=int, default=7,
                        help="Number of fireflies in observation")
    parser.add_argument("--max-in-memory-time", type=int,
                        default=2, help="Max in-memory time for env")
    parser.add_argument("--angular-terminal-vel", type=float,
                        default=0.05, help="Angular terminal velocity threshold")
    parser.add_argument("--no-train", dest="to_train_agent",
                        action="store_false", help="Disable training (evaluate only)")
    parser.set_defaults(to_train_agent=True)
    parser.add_argument("--no-load-latest", dest="to_load_latest_agent",
                        action="store_false", help="Do not auto-load latest agent")
    parser.set_defaults(to_load_latest_agent=True)
    parser.add_argument("--load-replay-buffer", dest="load_replay_buffer",
                        action="store_true", help="Load saved replay buffer if available")
    parser.set_defaults(load_replay_buffer=False)
    args = parser.parse_args()

    overall_folder = os.path.expanduser(args.overall_folder)
    os.makedirs(overall_folder, exist_ok=True)

    env_kwargs = {
        'num_obs_ff': args.num_obs_ff,
        'add_action_to_obs': True,
        'angular_terminal_vel': args.angular_terminal_vel,
        "dt": args.dt,
        "max_in_memory_time": args.max_in_memory_time,
    }

    rl = lstm_class.LSTMforMultifirefly(overall_folder=overall_folder,
                                        **env_kwargs)

    # Common sweep/run params (captured from slurm and CLI)
    sweep_params = {
        'lr': args.lr,
        'num_obs_ff': args.num_obs_ff,
        'max_in_memory_time': args.max_in_memory_time,
        'angular_terminal_vel': args.angular_terminal_vel,
        'dt': args.dt,
    }
    try:
        run_logger.log_run_start(overall_folder, agent_type='lstm', sweep_params=sweep_params)
    except Exception as e:
        print('[logger] failed to log run start:', e)

    # Attach sweep params to agent for curriculum stage logging
    try:
        rl.sweep_params = sweep_params
    except Exception:
        pass

    rl.streamline_everything(currentTrial_for_animation=None,
                             num_trials_for_animation=None,
                             duration=args.duration,
                             to_load_latest_agent=args.to_load_latest_agent,
                             best_model_postcurriculum_exists_ok=True,
                             to_train_agent=args.to_train_agent,
                             load_replay_buffer=args.load_replay_buffer)

    try:
        metrics = {
            'best_avg_reward': getattr(rl, 'best_avg_reward', None)
        }
        run_logger.log_run_end(overall_folder, agent_type='lstm', sweep_params=sweep_params, status='finished', metrics=metrics)
    except Exception as e:
        print('[logger] failed to log run end:', e)

    # Collect model directories from this run into a per-job folder
    try:
        run_logger.collect_model_to_job_dir(overall_folder, getattr(rl, 'model_folder_name', overall_folder), preferred_name=os.path.basename(getattr(rl, 'model_folder_name', 'agent_model')))
        if getattr(rl, 'best_model_in_curriculum_dir', None):
            run_logger.collect_model_to_job_dir(overall_folder, rl.best_model_in_curriculum_dir, preferred_name=os.path.basename(rl.best_model_in_curriculum_dir))
        if getattr(rl, 'best_model_postcurriculum_dir', None):
            run_logger.collect_model_to_job_dir(overall_folder, rl.best_model_postcurriculum_dir, preferred_name=os.path.basename(rl.best_model_postcurriculum_dir))
    except Exception as e:
        print('[logger] failed to collect models into job dir:', e)
