import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Ensure project root on sys.path similar to lstm_script.py
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

from reinforcement_learning.agents.rppo import rppo_class
from reinforcement_learning.base_classes import run_logger

os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate RPPO agent on Multifirefly env")

    # Optimization args (kept minimal, map --lr to PPO learning_rate)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the RPPO optimizer")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    # Environment and training arguments
    parser.add_argument("--overall-folder", type=str,
                        default='multiff_analysis/RL_models/RPPO_stored_models/all_agents/env1/',
                        help="Output directory to save models and manifests")
    parser.add_argument("--duration", type=int, nargs=2, default=[10, 40],
                        help="[min,max] steps for evaluation animations/checks")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Environment time step")
    parser.add_argument("--num-obs-ff", type=int, default=7,
                        help="Number of fireflies in observation")
    parser.add_argument("--max-in-memory-time", type=int, default=2,
                        help="Max in-memory time for env")
    parser.add_argument("--angular-terminal-vel", type=float, default=0.05,
                        help="Angular terminal velocity threshold")
    parser.add_argument("--identity-slot-strategy", type=str, default="rank_keep",
                        help="Strategy for identity slot handling in env")

    # RPPO-specific
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments for RPPO")

    # Flags
    parser.add_argument("--no-train", dest="to_train_agent",
                        action="store_false", help="Disable training (evaluate only)")
    parser.set_defaults(to_train_agent=True)
    parser.add_argument("--no-load-latest", dest="to_load_latest_agent",
                        action="store_false", help="Do not auto-load latest agent")
    parser.set_defaults(to_load_latest_agent=True)
    parser.add_argument("--load-replay-buffer", dest="load_replay_buffer",
                        action="store_true", help="Load saved replay buffer if available")
    parser.set_defaults(load_replay_buffer=False)

    parser.add_argument("--no-curriculum-training", dest="curriculum_training",
                        action="store_false", help="Disable curriculum training")
    parser.set_defaults(curriculum_training=True)

    args = parser.parse_args()

    # Basic device print for visibility
    device = torch.device(
        f"cuda:{int(os.getenv('CUDA_DEVICE', '0'))}" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print('[env] device:', device)

    overall_folder = os.path.expanduser(args.overall_folder)
    os.makedirs(overall_folder, exist_ok=True)

    env_kwargs = {
        'num_obs_ff': args.num_obs_ff,
        'angular_terminal_vel': args.angular_terminal_vel,
        'dt': args.dt,
        'max_in_memory_time': args.max_in_memory_time,
        'identity_slot_strategy': args.identity_slot_strategy,
    }

    rl = rppo_class.RPPOforMultifirefly(
        overall_folder=overall_folder,
        n_envs=args.n_envs,
        dict_obs=True,
        **env_kwargs,
    )

    # Common sweep/run params (captured from slurm and CLI)
    sweep_params = {
        'lr': args.lr,
        'num_obs_ff': args.num_obs_ff,
        'max_in_memory_time': args.max_in_memory_time,
        'angular_terminal_vel': args.angular_terminal_vel,
        'dt': args.dt,
        'n_envs': args.n_envs,
        'identity_slot_strategy': args.identity_slot_strategy,
    }
    try:
        run_logger.log_run_start(
            overall_folder, agent_type='rppo', sweep_params=sweep_params)
    except Exception as e:
        print('[logger] failed to log run start:', e)

    # Attach sweep params to agent for curriculum stage logging and set PPO params
    try:
        rl.sweep_params = sweep_params
    except Exception:
        pass

    # Ensure PPO hyperparams include learning rate prior to agent creation
    try:
        rl.prepare_agent_params(
            agent_params_already_set_ok=True, learning_rate=args.lr)
    except Exception as e:
        print('[rppo] warning: failed to set agent params before training:', e)

    rl.streamline_everything(
        currentTrial_for_animation=None,
        num_trials_for_animation=None,
        duration=args.duration,
        to_load_latest_agent=args.to_load_latest_agent,
        best_model_postcurriculum_exists_ok=True,
        to_train_agent=args.to_train_agent,
        load_replay_buffer=args.load_replay_buffer,
        use_curriculum_training=args.curriculum_training,
    )

    try:
        metrics = {
            'best_avg_reward': getattr(rl, 'best_avg_reward', None)
        }
        run_logger.log_run_end(overall_folder, agent_type='rppo',
                               sweep_params=sweep_params, status='finished', metrics=metrics)
    except Exception as e:
        print('[logger] failed to log run end:', e)

    # Collect model directories from this run into a per-job folder
    try:
        run_logger.collect_model_to_job_dir(overall_folder, getattr(
            rl, 'model_folder_name', overall_folder), preferred_name=os.path.basename(getattr(rl, 'model_folder_name', 'agent_model')))
        if getattr(rl, 'best_model_in_curriculum_dir', None):
            run_logger.collect_model_to_job_dir(
                overall_folder, rl.best_model_in_curriculum_dir, preferred_name=os.path.basename(rl.best_model_in_curriculum_dir))
        if getattr(rl, 'best_model_postcurriculum_dir', None):
            run_logger.collect_model_to_job_dir(
                overall_folder, rl.best_model_postcurriculum_dir, preferred_name=os.path.basename(rl.best_model_postcurriculum_dir))
    except Exception as e:
        print('[logger] failed to collect models into job dir:', e)
