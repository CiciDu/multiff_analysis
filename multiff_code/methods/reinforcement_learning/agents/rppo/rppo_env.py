from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from reinforcement_learning.base_classes.base_env import MultiFF


class WrapperForRecurrentPPO(gym.Wrapper):
    """
    SB3 / RecurrentPPO-compatible wrapper for the MultiFirefly environment.

    Converts the flat observation vector into a Dict if needed:
      - 'slots': (num_obs_ff, num_elem_per_ff)
      - 'ego':   (2,) or other scalar features (e.g., previous action)
    """

    def __init__(self, env: MultiFF, dict_obs: bool = True):
        super().__init__(env)
        self.dict_obs = dict_obs

        if self.dict_obs:
            # Decompose flat obs into Dict space
            n_slots = env.num_obs_ff * env.num_elem_per_ff
            self.observation_space = spaces.Dict({
                'slots': spaces.Box(low=-1., high=1.,
                                    shape=(env.num_obs_ff,
                                           env.num_elem_per_ff),
                                    dtype=np.float32),
                'ego': spaces.Box(low=-1., high=1.,
                                  shape=(2,), dtype=np.float32)
            })
        else:
            self.observation_space = env.observation_space

        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def _process_obs(self, obs):
        if not self.dict_obs:
            return obs
        n_slots = self.env.num_obs_ff * self.env.num_elem_per_ff
        slots = obs[:n_slots].reshape(
            self.env.num_obs_ff, self.env.num_elem_per_ff)
        ego = obs[n_slots:n_slots + 2]  # action-embedded or agent state
        return {'slots': slots, 'ego': ego}

    def get_basic_params(self):
        """
        Return essential scalar parameters from the underlying base env.
        This is used from vectorized envs (via env_method) for curriculum logic.
        """
        base = self.env
        return {
            'episode_len': getattr(base, 'episode_len', None),
            'dt': getattr(base, 'dt', None),
            'reward_per_ff': getattr(base, 'reward_per_ff', None),
            'distance2center_cost': getattr(base, 'distance2center_cost', None),
        }

    def get_curriculum_params(self):
        """
        Return curriculum-related parameters from the base environment.
        Used by vectorized envs through env_method.
        """
        base = self.env
        return {
            'flash_on_interval': getattr(base, 'flash_on_interval', None),
            'angular_terminal_vel': getattr(base, 'angular_terminal_vel', None),
            'distance2center_cost': getattr(base, 'distance2center_cost', None),
            'stop_vel_cost': getattr(base, 'stop_vel_cost', None),
            'reward_boundary': getattr(base, 'reward_boundary', None),
            'dv_cost_factor': getattr(base, 'dv_cost_factor', None),
            'dw_cost_factor': getattr(base, 'dw_cost_factor', None),
            'w_cost_factor': getattr(base, 'w_cost_factor', None),
        }

    def set_curriculum_params(self, **kwargs):
        """
        Set curriculum-related parameters on the base environment.
        Accepts any subset of supported keys.
        Returns True for compatibility with env_method expectations.
        """
        base = self.env
        allowed = {
            'flash_on_interval', 'angular_terminal_vel', 'distance2center_cost',
            'stop_vel_cost', 'reward_boundary', 'dv_cost_factor',
            'dw_cost_factor', 'w_cost_factor', 'jerk_cost_factor',
            'cost_per_stop'
        }
        for key, value in kwargs.items():
            if key in allowed and value is not None:
                try:
                    setattr(base, key, value)
                except Exception:
                    pass
        return True


def make_vec_env_for_rppo(env_class, env_kwargs=None, dict_obs=True, n_envs=8, use_subproc=True):
    """
    Create a vectorized MultiFF environment compatible with RecurrentPPO.
    Each copy is wrapped with WrapperForRecurrentPPO.

    Args:
        env_class: the base environment class (e.g., MultiFF)
        env_kwargs: dict of keyword arguments to pass to each env
        dict_obs: whether to return Dict observations ('slots', 'ego')
        n_envs: number of parallel environments
        use_subproc: whether to use SubprocVecEnv (parallel) or DummyVecEnv (serial)

    Returns:
        A VecEnv ready for RecurrentPPO.
    """
    env_kwargs = env_kwargs or {}

    def make_single_env(rank):
        def _init():
            env = env_class(**env_kwargs)
            return WrapperForRecurrentPPO(env, dict_obs=dict_obs)
        return _init

    env_fns = [make_single_env(i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    print(
        f'[rppo] Created {n_envs} parallel envs ({"SubprocVecEnv" if use_subproc else "DummyVecEnv"})')
    return vec_env
