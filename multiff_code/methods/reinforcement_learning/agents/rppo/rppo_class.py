import os
import copy
from reinforcement_learning.base_classes import rl_base_class, rl_base_utils, env_utils, base_env
from reinforcement_learning.agents.rppo import rppo_env
from reinforcement_learning.agents.feedforward import sb3_utils
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecMonitor


class RPPOforMultifirefly(rl_base_class._RLforMultifirefly):
    """
    RL controller for Multi-Firefly task using Recurrent PPO (RPPO).

    This class manages:
      - Environment creation and wrapping for recurrent PPO.
      - Agent configuration, saving/loading.
      - Curriculum training structure inherited from _RLforMultifirefly.
    """

    def __init__(self,
                 overall_folder='multiff_analysis/RL_models/RPPO_stored_models/all_agents/env1/',
                 add_date_to_model_folder_name=False,
                 dict_obs=True,
                 n_envs=1,
                 **kwargs):
        super().__init__(overall_folder=overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         **kwargs)

        self.agent_type = 'rppo'
        self.dict_obs = dict_obs

        # Use base MultiFF env; wrap later for SB3 RecurrentPPO
        self.env_class = base_env.MultiFF
        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)

        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

        self.agent_params = None
        self.rl_agent = None
        self.n_envs = n_envs
        # Track best average evaluation reward across stages/phases for logging
        self.best_avg_reward = float('-inf')

    def make_env(self,  **env_kwargs):
        super().make_env(**env_kwargs)
        self.env = rppo_env.make_vec_env_for_rppo(
            env_class=self.env_class,
            env_kwargs=self.current_env_kwargs,
            dict_obs=self.dict_obs,
            n_envs=self.n_envs,
            use_subproc=True
        )
        # Add VecMonitor to log training rewards for callbacks that read monitor files
        try:
            os.makedirs(self.model_folder_name, exist_ok=True)
        except Exception:
            pass
        self.env = VecMonitor(self.env, filename=os.path.join(
            self.model_folder_name, 'monitor.csv'))

    # -------------------------------------------------------------------------
    # Agent setup
    # -------------------------------------------------------------------------
    def prepare_agent_params(self, agent_params_already_set_ok=True, **kwargs):
        """Set agent hyperparameters."""
        if not hasattr(self, 'env'):
            self.make_env(**self.input_env_kwargs)

        # Preserve existing if already initialized
        existing = self.agent_params.copy() if (
            agent_params_already_set_ok and self.agent_params is not None) else {}

        # Default PPO parameters — extend later with SB3 RecurrentPPO params
        defaults = {
            'gamma': kwargs.get('gamma', 0.99),
            'learning_rate': kwargs.get('learning_rate', 3e-4),
            'n_steps': kwargs.get('n_steps', 512),
            'batch_size': kwargs.get('batch_size', 256),
            'n_epochs': kwargs.get('n_epochs', 10),
            'clip_range': kwargs.get('clip_range', 0.2),
        }

        self.agent_params = {**defaults, **existing, **kwargs}
        print('[RPPO] Prepared agent params:', self.agent_params)

    def make_agent(self, agent_params_already_set_ok=True, **kwargs):
        """Initialize RPPO agent; placeholder until full integration."""
        self.prepare_agent_params(agent_params_already_set_ok, **kwargs)

        # Import lazily to avoid circular imports
        try:
            from sb3_contrib import RecurrentPPO
            from sb3_contrib.ppo_recurrent import MultiInputLstmPolicy
        except ImportError as e:
            raise ImportError(
                'Stable-Baselines3 and sb3-contrib must be installed for RPPO: ' + str(e))

        self.rl_agent = RecurrentPPO(
            policy=MultiInputLstmPolicy,
            env=self.env,
            verbose=0,
            # tensorboard_log=os.path.join(self.model_folder_name, "tb_logs"),
            **{k: v for k, v in self.agent_params.items()
               if k in ['learning_rate', 'gamma', 'n_steps', 'batch_size', 'n_epochs', 'clip_range']}
        )
        print('[RPPO] Created RecurrentPPO model')

    # -------------------------------------------------------------------------
    # Curriculum / training
    # -------------------------------------------------------------------------
    def _curriculum_train_step(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        """
        Train one curriculum stage using EvalCallback with a reward threshold.
        Returns (passed, best_avg_reward).
        """
        if self.rl_agent is None:
            raise RuntimeError('RecurrentPPO model not initialized.')

        print(
            f'[RPPO] Starting curriculum stage: reward_threshold={reward_threshold:.2f}')

        stop_train_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold)
        callback = EvalCallback(
            self.env,
            eval_freq=self.current_env_kwargs['episode_len'] * eval_eps_freq,
            callback_after_eval=stop_train_callback,
            verbose=1,
            n_eval_episodes=num_eval_episodes,
            best_model_save_path=self.best_model_in_curriculum_dir,
        )

        self.write_checkpoint_manifest(self.best_model_in_curriculum_dir)

        self.rl_agent.learn(total_timesteps=1_000_000, callback=callback)
        best_mean_reward = float(
            getattr(callback, 'best_mean_reward', float('-inf')))
        passed = bool(best_mean_reward >= reward_threshold)
        print(
            f'[RPPO] Stage complete — best_mean={best_mean_reward:.2f}, passed={passed}')
        # Update global best tracker for run_end logging
        try:
            if self.best_avg_reward is None:
                self.best_avg_reward = best_mean_reward
            else:
                self.best_avg_reward = max(
                    self.best_avg_reward, best_mean_reward)
        except Exception:
            pass
        return passed, best_mean_reward

    def _post_curriculum_final_train(self, reward_threshold: float, eval_eps_freq: int, num_eval_episodes: int):
        """Final long training phase after curriculum with early stop and best-model saving."""
        if self.rl_agent is None:
            raise RuntimeError('RecurrentPPO model not initialized.')

        # Train with early stopping on no improvement and save best evaluated model
        self.regular_training(
            timesteps=500_000,
            best_model_save_path=self.best_model_postcurriculum_dir,
            n_eval_episodes=num_eval_episodes,
            eval_eps_freq=eval_eps_freq,
        )

        # Evaluate the final/best model for reporting
        try:
            avg_reward = rl_base_utils.evaluate_policy_mean_reward(
                self.rl_agent, self.env, num_eval_episodes=num_eval_episodes)
        except Exception:
            avg_reward = float('nan')
        print(f'[RPPO] Post-curriculum average reward: {avg_reward:.2f}')
        # Update global best tracker for run_end logging
        try:
            if self.best_avg_reward is None:
                self.best_avg_reward = float(avg_reward)
            else:
                self.best_avg_reward = max(
                    self.best_avg_reward, float(avg_reward))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Saving and loading
    # -------------------------------------------------------------------------
    def write_checkpoint_manifest(self, dir_name):
        """Write checkpoint metadata (env, params)."""
        rl_base_utils.write_checkpoint(dir_name, {
            'algorithm': 'rppo',
            'model_file': 'best_model.zip',
            'replay_buffer': None,
            'env_params': getattr(self, 'current_env_kwargs', self.input_env_kwargs),
        })

    def save_agent(self, whether_save_replay_buffer=False, dir_name=None):
        """Save PPO model and manifest."""
        if dir_name is None:
            dir_name = self.model_folder_name
        os.makedirs(dir_name, exist_ok=True)

        if self.rl_agent is not None:
            model_path = os.path.join(dir_name, 'best_model.zip')
            self.rl_agent.save(model_path)
            print('[RPPO] Saved model at', model_path)
        self.write_checkpoint_manifest(dir_name)

    def load_agent(self, load_replay_buffer=True, dir_name=None, restore_env_from_checkpoint=True):
        """Load PPO model from directory."""
        if dir_name is None:
            dir_name = self.model_folder_name
        manifest = rl_base_utils.read_checkpoint_manifest(dir_name)
        env_params = (manifest.get('env_params', None)
                      if isinstance(manifest, dict) else None)
        if not isinstance(env_params, dict):
            env_params = getattr(self, 'current_env_kwargs',
                                 None) or self.input_env_kwargs

        if restore_env_from_checkpoint:
            self.make_env(**env_params)

        model_file = manifest.get('model_file') if isinstance(
            manifest, dict) else None
        model_path = os.path.join(dir_name, model_file) if model_file else os.path.join(
            dir_name, 'best_model.zip')
        try:
            from sb3_contrib import RecurrentPPO
            self.rl_agent = RecurrentPPO.load(model_path, env=self.env)
            print('[RPPO] Loaded model from', model_path)
        except Exception as e:
            print('[RPPO] Warning: failed to load model:', e)
            self.make_agent()
        self.loaded_agent_dir = dir_name

    # -------------------------------------------------------------------------
    # Regular training with callbacks (early stop + save best)
    # -------------------------------------------------------------------------
    def regular_training(self, timesteps=1_000_000, best_model_save_path=None, n_eval_episodes: int = 3, eval_eps_freq: int = 20):
        if self.rl_agent is None:
            raise RuntimeError('RecurrentPPO model not initialized.')

        if best_model_save_path is None:
            best_model_save_path = self.model_folder_name
        os.makedirs(best_model_save_path, exist_ok=True)

        # Early-stop on no improvement of eval mean reward; also let EvalCallback save best_model.zip
        stop_train_callback = sb3_utils.StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=15,
            verbose=1,
            model_folder_name=self.model_folder_name,
            overall_folder=self.overall_folder,
            agent_id=self.agent_id,
            best_model_save_path=best_model_save_path,
        )

        callback = EvalCallback(
            self.env,
            eval_freq=self.current_env_kwargs['episode_len'] * eval_eps_freq,
            callback_after_eval=stop_train_callback,
            verbose=1,
            best_model_save_path=best_model_save_path,
            n_eval_episodes=n_eval_episodes,
        )

        # Ensure manifest exists in save path for downstream loaders
        self.write_checkpoint_manifest(best_model_save_path)

        # Train
        self.rl_agent.learn(total_timesteps=int(timesteps), callback=callback)

        # Load the best model back into memory for downstream evaluation/saving
        try:
            best_path = os.path.join(best_model_save_path, 'best_model.zip')
            if os.path.exists(best_path):
                from sb3_contrib import RecurrentPPO
                self.rl_agent = RecurrentPPO.load(best_path, env=self.env)
        except Exception as e:
            print('[RPPO] Warning: failed to load best model after training:', e)

    # -------------------------------------------------------------------------
    # Explicitly skip replay buffer for RPPO when loading best models
    # -------------------------------------------------------------------------
    def load_best_model_postcurriculum(self, load_replay_buffer=False, restore_env_from_checkpoint=True):
        """Load best post-curriculum model without replay buffer for RPPO."""
        # Intentionally ignore load_replay_buffer for RPPO
        self.load_agent(load_replay_buffer=False,
                        dir_name=self.best_model_postcurriculum_dir, restore_env_from_checkpoint=restore_env_from_checkpoint)

    def load_best_model_in_curriculum(self, load_replay_buffer=False, restore_env_from_checkpoint=True):
        """Load best in-curriculum model without replay buffer for RPPO."""
        # Intentionally ignore load_replay_buffer for RPPO
        self.load_agent(load_replay_buffer=False,
                        dir_name=self.best_model_in_curriculum_dir, restore_env_from_checkpoint=restore_env_from_checkpoint)

    # -------------------------------------------------------------------------
    # Curriculum initialization
    # -------------------------------------------------------------------------
    def make_init_env_for_curriculum_training(self,
                                              **kwargs):
        """Initialize environment for first stage of curriculum."""
        self.make_env(**self.input_env_kwargs)
        self._make_init_env_for_curriculum_training(**kwargs)
        print('[RPPO] Initialized env for curriculum training')
