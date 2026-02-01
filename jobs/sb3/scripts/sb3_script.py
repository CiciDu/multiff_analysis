import torch
import numpy as np
import json
import time as time_package
import argparse
import sys
import os
from pathlib import Path
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
# isort: off
# fmt: off
from reinforcement_learning.base_classes import run_logger, rl_base_utils
from reinforcement_learning.agents.feedforward import sb3_class
# fmt: on
# isort: on


os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate sb3 agent on Multifirefly env")

    # Optimization args (kept minimal, map --lr to PPO learning_rate)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the sb3 optimizer")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    # Environment and training arguments
    parser.add_argument("--overall-folder", type=str,
                        default='multiff_analysis/RL_models/sb3_stored_models/all_agents/env1/',
                        help="Output directory to save models and manifests")
    parser.add_argument("--duration", type=int, nargs=2, default=[10, 40],
                        help="[min,max] steps for evaluation animations/checks")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Environment time step")
    parser.add_argument("--num-obs-ff", type=int, default=7,
                        help="Number of fireflies in observation")
    parser.add_argument("--max-in-memory-time", type=float, default=2.0,
                        help="Max in-memory time for env")
    parser.add_argument("--angular-terminal-vel", type=float, default=0.05,
                        help="Angular terminal velocity threshold")
    parser.add_argument("--identity-slot-strategy", type=str, default="rank_keep",
                        help="Strategy for identity slot handling in env")
    # Dynamics noise
    parser.add_argument("--v-noise-std", type=float, default=None,
                        help="Std of linear velocity noise (env v_noise_std)")
    parser.add_argument("--w-noise-std", type=float, default=None,
                        help="Std of angular velocity noise (env w_noise_std)")
    # Observation noise (perception/memory/lognormal)
    parser.add_argument("--obs-perc-r", "--perc-r", dest="obs_perc_r", type=float, default=None,
                        help="Perception radial Weber fraction")
    parser.add_argument("--obs-perc-th", "--perc-th", dest="obs_perc_th", type=float, default=None,
                        help="Perception angular base std")
    parser.add_argument("--obs-mem-r", "--mem-r", dest="obs_mem_r", type=float, default=None,
                        help="Memory radial Weber step")
    parser.add_argument("--obs-mem-th", "--mem-th", dest="obs_mem_th", type=float, default=None,
                        help="Memory angular step base std")
    # Cost factors
    parser.add_argument("--dv-cost-factor", type=float, default=None,
                        help="Cost factor for dv^2 term (optional override)")
    parser.add_argument("--dw-cost-factor", type=float, default=None,
                        help="Cost factor for dw^2 term (optional override)")
    parser.add_argument("--w-cost-factor", type=float, default=None,
                        help="Cost factor for w^2 term (optional override)")
    parser.add_argument("--jerk-cost-factor", type=float, default=None,
                        help="Cost factor for jerk term (optional override)")
    parser.add_argument("--cost-per-stop", type=float, default=None,
                        help="Cost incurred per stop event (optional override)")

    # sb3-specific
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments for sb3")

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

    # Evaluation options (used when --no-train)
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of eval episodes when running with --no-train")
    # Initialize from an existing best model directory (e.g., .../post/best)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Directory containing a pretrained model to initialize from")
    # Optional explicit model subfolder name (under overall-folder). If provided, overrides auto cost-tag naming.
    parser.add_argument("--model-folder-name", type=str, default=None,
                        help="Optional model subfolder name under overall-folder")

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
    # Dynamics noise if provided
    if args.v_noise_std is not None:
        env_kwargs['v_noise_std'] = args.v_noise_std
    if args.w_noise_std is not None:
        env_kwargs['w_noise_std'] = args.w_noise_std
    # Observation noise if provided (partial dict allowed; defaults fill the rest)
    obs_noise_updates = {}
    if args.obs_perc_r is not None:
        obs_noise_updates['perc_r'] = args.obs_perc_r
    if args.obs_perc_th is not None:
        obs_noise_updates['perc_th'] = args.obs_perc_th
    if args.obs_mem_r is not None:
        obs_noise_updates['mem_r'] = args.obs_mem_r
    if args.obs_mem_th is not None:
        obs_noise_updates['mem_th'] = args.obs_mem_th
    if len(obs_noise_updates) > 0:
        env_kwargs['obs_noise'] = obs_noise_updates
    # Optional overrides for cost factors
    if args.dv_cost_factor is not None:
        env_kwargs['dv_cost_factor'] = args.dv_cost_factor
    if args.dw_cost_factor is not None:
        env_kwargs['dw_cost_factor'] = args.dw_cost_factor
    if args.w_cost_factor is not None:
        env_kwargs['w_cost_factor'] = args.w_cost_factor
    if args.jerk_cost_factor is not None:
        env_kwargs['jerk_cost_factor'] = args.jerk_cost_factor
    if args.cost_per_stop is not None:
        env_kwargs['cost_per_stop'] = args.cost_per_stop

    if args.model_folder_name:
        # If absolute path provided, use as-is; else place under overall_folder
        model_folder_name = (
            args.model_folder_name
            if os.path.isabs(args.model_folder_name)
            else os.path.join(overall_folder, args.model_folder_name)
        )
    else:
        model_folder_name = overall_folder

    rl = sb3_class.SB3forMultifirefly(
        overall_folder=overall_folder,
        model_folder_name=model_folder_name,
        n_envs=args.n_envs,
        dict_obs=True,
        zero_invisible_ff_features=False,
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
    # Include costs if provided
    if args.dv_cost_factor is not None:
        sweep_params['dv_cost_factor'] = args.dv_cost_factor
    if args.dw_cost_factor is not None:
        sweep_params['dw_cost_factor'] = args.dw_cost_factor
    if args.w_cost_factor is not None:
        sweep_params['w_cost_factor'] = args.w_cost_factor
    if args.jerk_cost_factor is not None:
        sweep_params['jerk_cost_factor'] = args.jerk_cost_factor
    if args.cost_per_stop is not None:
        sweep_params['cost_per_stop'] = args.cost_per_stop
    # Include noise settings if provided
    if args.v_noise_std is not None:
        sweep_params['v_noise_std'] = args.v_noise_std
    if args.w_noise_std is not None:
        sweep_params['w_noise_std'] = args.w_noise_std
    if args.obs_perc_r is not None:
        sweep_params['obs_perc_r'] = args.obs_perc_r
    if args.obs_perc_th is not None:
        sweep_params['obs_perc_th'] = args.obs_perc_th
    if args.obs_mem_r is not None:
        sweep_params['obs_mem_r'] = args.obs_mem_r
    if args.obs_mem_th is not None:
        sweep_params['obs_mem_th'] = args.obs_mem_th

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
        print('[sb3] warning: failed to set agent params before training:', e)

    # If initializing from a pretrained model and training is requested, load and train directly
    if args.init_from and args.to_train_agent:
        try:
            # Resolve to standardized post/best only
            init_dir = os.path.expanduser(args.init_from)
            norm = os.path.normpath(init_dir)
            base = os.path.basename(norm)
            parent = os.path.basename(os.path.dirname(norm))
            if not (base == 'best' and parent == 'post' and os.path.isdir(norm)):
                candidate = os.path.join(init_dir, 'post', 'best')
                if os.path.isdir(candidate):
                    norm = candidate
                else:
                    raise ValueError(
                        "--init-from must point to 'post/best' or a parent directory containing it")
            init_dir = norm
            # Load pretrained model
            rl.load_agent(load_replay_buffer=False, dir_name=init_dir)
            # Mark as loaded post-curriculum to trigger regular training path if used
            try:
                rl.loaded_agent_name = 'post/best'
                rl.loaded_agent_dir = init_dir
            except Exception:
                pass
        except Exception as e:
            print('[init] failed to load pretrained model from --init-from:', e)
            # If load fails, fall back to making a fresh agent
            try:
                rl.make_agent()
            except Exception:
                pass

        # Re-apply cost overrides after loading
        if any(v is not None for v in (args.dv_cost_factor, args.dw_cost_factor, args.w_cost_factor, args.jerk_cost_factor, args.cost_per_stop)):
            try:
                updates = {}
                if args.dv_cost_factor is not None:
                    updates['dv_cost_factor'] = args.dv_cost_factor
                if args.jerk_cost_factor is not None:
                    updates['jerk_cost_factor'] = args.jerk_cost_factor
                if args.cost_per_stop is not None:
                    updates['cost_per_stop'] = args.cost_per_stop
                if args.dw_cost_factor is not None:
                    updates['dw_cost_factor'] = args.dw_cost_factor
                if args.w_cost_factor is not None:
                    updates['w_cost_factor'] = args.w_cost_factor
                if hasattr(rl, 'env') and hasattr(rl.env, 'env_method'):
                    rl.env.env_method('set_curriculum_params',
                                      indices=None, **updates)
                    print('[env] re-applied overrides:', updates)
                # Keep RL object's kwargs in sync with the overrides
                try:
                    for attr in ('input_env_kwargs', 'current_env_kwargs'):
                        d = getattr(rl, attr, None)
                        if isinstance(d, dict):
                            d.update(updates)
                except Exception:
                    pass
            except Exception as e:
                print('[env] failed to re-apply cost overrides:', e)

        # Train (regular or curriculum according to flag; if loaded_agent_name set, curriculum path will regular-train)
        rl.train_agent(use_curriculum_training=args.curriculum_training,
                       best_model_in_curriculum_exists_ok=True,
                       best_model_postcurriculum_exists_ok=True,
                       load_replay_buffer_of_best_model_postcurriculum=False)

    else:
        # Default path (optionally train and/or evaluate, using standard orchestration)
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

    # Re-apply cost overrides after potential manifest-based env recreation
    if any(v is not None for v in (args.dv_cost_factor, args.dw_cost_factor, args.w_cost_factor, args.jerk_cost_factor, args.cost_per_stop)):
        try:
            updates = {}
            if args.dv_cost_factor is not None:
                updates['dv_cost_factor'] = args.dv_cost_factor
            if args.dw_cost_factor is not None:
                updates['dw_cost_factor'] = args.dw_cost_factor
            if args.w_cost_factor is not None:
                updates['w_cost_factor'] = args.w_cost_factor
            if args.jerk_cost_factor is not None:
                updates['jerk_cost_factor'] = args.jerk_cost_factor
            if args.cost_per_stop is not None:
                updates['cost_per_stop'] = args.cost_per_stop
            if hasattr(rl, 'env') and hasattr(rl.env, 'env_method'):
                rl.env.env_method('set_curriculum_params',
                                  indices=None, **updates)
            # Keep RL object's kwargs in sync with the overrides
            try:
                for attr in ('input_env_kwargs', 'current_env_kwargs', 'curriculum_env_kwargs', 'class_instance_env_kwargs'):
                    d = getattr(rl, attr, None)
                    if isinstance(d, dict):
                        d.update(updates)
            except Exception:
                pass
            print('[env] re-applied overrides:', updates)
        except Exception as e:
            print('[env] failed to re-apply cost overrides:', e)

    # If in eval-only mode, compute and log mean reward
    if not args.to_train_agent:
        try:
            # Ensure we have a model and env ready
            try:
                if getattr(rl, 'rl_agent', None) is None:
                    rl.load_best_model_postcurriculum(load_replay_buffer=False)
            except Exception:
                pass
            # Evaluate
            mean_reward = rl_base_utils.evaluate_policy_mean_reward(
                rl.rl_agent, rl.env, num_eval_episodes=int(args.eval_episodes))
            print(
                f"[eval] mean_reward={mean_reward:.4f} over {int(args.eval_episodes)} episodes")
            # Log run_end with metrics
            try:
                run_logger.log_run_end(
                    overall_folder, agent_type=getattr(
                        rl, 'agent_type', 'sb3'),
                    sweep_params=sweep_params, status='evaluated', metrics={'mean_reward': float(mean_reward)})
            except Exception as e:
                print('[logger] failed to log eval run_end:', e)
        except Exception as e:
            print('[eval] evaluation failed:', e)

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
