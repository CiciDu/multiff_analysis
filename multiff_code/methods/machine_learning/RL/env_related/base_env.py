# machine_learning/RL_models/env_related/MultiFF.py

import inspect
from typing import Optional, Dict
from dataclasses import dataclass
from machine_learning.RL.env_related import env_utils

import os
import numpy as np
import math
from math import pi
import gymnasium
from typing import Optional, List

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---- Tunables for transforms ----
DEFAULT_D0 = 25.0      # anchor for d_log = log1p(dist/d0)
CLIP_DMAX = 500.0      # clip distance before log
TMAX_DEFAULT = 5.0     # seconds cap for t_seen normalization



@dataclass
class ObsNoiseCfg:
    # perception (visible): instantaneous noise
    perc_r: float = 0.02     # Weber radial (std_r = perc_r * r)
    perc_th: float = 0.005   # angular base (std_th ≈ perc_th / r)

    # memory (invisible): per-step diffusion
    mem_r: float = 0.03      # Weber radial step
    mem_th: float = 0.001    # angular step base (~1/r)

    # lognormal obs (if you use it)
    lognorm_k: float = 0.10  # dimensionless log-std multiplier (0..0.5)

    # shared knobs
    theta_floor: float = 1e-4
    r_min_for_theta: float = 10.0
    seed: Optional[int] = None
    preset: str = 'weber_perc__ego_mem'  # doc label only

    def validate(self):
        assert self.perc_r >= 0 and self.perc_th >= 0
        assert self.mem_r >= 0 and self.mem_th >= 0
        assert 0.0 <= self.lognorm_k <= 0.5
        assert self.theta_floor >= 0
        assert self.r_min_for_theta > 0
        return self

    @staticmethod
    def from_dict(d: Dict) -> 'ObsNoiseCfg':
        return ObsNoiseCfg(**d).validate()


class MultiFF(gymnasium.Env):
    # The MultiFirefly-Task RL environment
    # Per-slot feature field order mapping (kept in sync with downstream wrappers)
    FIELD_INDEX = {
        'valid': 0,
        'd_log': 1,
        'sin': 2,
        'cos': 3,
        't_start_seen': 4,
        't_last_seen': 5,
        'visible': 6,
        'pose_unreliable': 7,
    }

    def __init__(self,
                 v_noise_std=0.1,
                 w_noise_std=0.1,
                 num_alive_ff=200,
                 flash_on_interval=0.3,
                 num_obs_ff=5,
                 max_in_memory_time=3,
                 invisible_distance=500,
                 make_ff_always_flash_on=False,
                 reward_per_ff=100,
                 dv_cost_factor=10,
                 dw_cost_factor=10,
                 w_cost_factor=10,
                 distance2center_cost=0,
                 stop_vel_cost=50,
                 reward_boundary=25,
                 add_vel_cost_when_catching_ff_only=False,
                 linear_terminal_vel=0.01,
                 angular_terminal_vel=0.01,
                 dt=0.1,
                 episode_len=512,
                 print_ff_capture_incidents=True,
                 print_episode_reward_rates=True,
                 add_action_to_obs=True,
                 noise_mode='linear',
                 slot_fields: Optional[List[str]] = None,
                 obs_visible_only: bool = False,
                 zero_invisible_ff_features: bool = False,
                 use_prev_obs_for_invisible_pose: bool = False,
                 obs_noise: Optional[ObsNoiseCfg | dict] = None,
                 identity_slot_strategy: str = 'drop_fill',
                 **kwargs
                 ):

        super().__init__()

        if obs_noise is None:
            obs_noise = ObsNoiseCfg()
        if isinstance(obs_noise, dict):
            obs_noise = ObsNoiseCfg.from_dict(obs_noise)
        obs_noise.validate()
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(obs_noise.seed)

        kwargs.setdefault('add_action_to_obs', add_action_to_obs)
        kwargs.setdefault('episode_len', 512)

        # Identity-tracked slot config and transforms
        self.d0 = DEFAULT_D0
        self.D_max = CLIP_DMAX
        self.T_max = TMAX_DEFAULT
        # cache for d_log denominator
        self._inv_log_denom = np.float32(
            1.0 / math.log1p(self.D_max / self.d0))

        self.linear_terminal_vel = linear_terminal_vel
        self.angular_terminal_vel = angular_terminal_vel
        self.num_alive_ff = num_alive_ff
        self.flash_on_interval = flash_on_interval
        self.invisible_distance = invisible_distance
        self.make_ff_always_flash_on = make_ff_always_flash_on
        self.reward_per_ff = reward_per_ff
        self.dv_cost_factor = dv_cost_factor
        self.dw_cost_factor = dw_cost_factor
        self.w_cost_factor = w_cost_factor
        self.distance2center_cost = distance2center_cost
        self.stop_vel_cost = stop_vel_cost
        self.dt = dt
        self.episode_len = episode_len
        self.print_ff_capture_incidents = print_ff_capture_incidents
        self.print_episode_reward_rates = print_episode_reward_rates
        self.add_action_to_obs = add_action_to_obs
        self.add_vel_cost_when_catching_ff_only = add_vel_cost_when_catching_ff_only
        self.noise_mode = noise_mode
        # identity slot strategy: 'drop_fill' (default) or 'rank_keep'
        self.identity_slot_strategy = identity_slot_strategy

        # parameters
        self.v_noise_std = v_noise_std
        self.w_noise_std = w_noise_std
        self.num_obs_ff = num_obs_ff
        self.max_in_memory_time = max_in_memory_time

        # Observation spec (per slot)
        # [valid, d_log, sinθ, cosθ, t_start_seen, t_last_seen, visible, pose_unreliable]
        self.slot_fields = slot_fields or [
            'valid', 'd_log', 'sin', 'cos', 't_start_seen', 't_last_seen', 'visible']
        self._slot_idx = [self.FIELD_INDEX[f] for f in self.slot_fields]
        self.num_elem_per_ff = len(self.slot_fields)
        self.add_action_to_obs = add_action_to_obs
        # control whether to keep only visible ff in observation slots
        self.obs_visible_only = obs_visible_only
        # if True, zero out feature rows for invisible ff bound to slots
        self.zero_invisible_ff_features = zero_invisible_ff_features
        # if True, copy pose features for invisible slots from previous step's obs
        self.use_prev_obs_for_invisible_pose = use_prev_obs_for_invisible_pose
        # validity buffer for tails
        self._slot_valid_mask = np.zeros(self.num_obs_ff, dtype=np.int32)
        # previous step slot features snapshot (S,N)
        self._prev_slots_SN = None
        # Cache for performance optimization
        self._valid_field_index = self.slot_fields.index(
            'valid') if 'valid' in self.slot_fields else None
        self._make_observation_space()

        self.action_space = gymnasium.spaces.Box(
            low=-1., high=1., shape=(2,), dtype=np.float32)
        self.vgain = 200
        self.wgain = pi / 2
        self.arena_radius = 1000
        self.ff_radius = 10
        self.invisible_angle = 2 * pi / 9
        self.epi_num = 0
        self.time = 0
        self.reward_boundary = reward_boundary
        self.current_obs = np.zeros(self.obs_space_length, dtype=np.float32)

        # world state buffers (contiguous, float32)
        self.ffxy = np.zeros((self.num_alive_ff, 2), dtype=np.float32)
        self.ffxy_noisy = np.zeros((self.num_alive_ff, 2), dtype=np.float32)
        # last-seen feature buffers removed in favor of prev-obs snapshot approach
        self.ffr = np.zeros(self.num_alive_ff, dtype=np.float32)
        self.fftheta = np.zeros(self.num_alive_ff, dtype=np.float32)

        # scratch / flags
        self._scratch_dist2 = np.zeros(self.num_alive_ff, dtype=np.float32)
        self._ff_is_visible = np.zeros(self.num_alive_ff, dtype=bool)

        self.ff_in_memory_indices = np.array([], dtype=np.int32)
        self.visible_ff_indices = np.array([], dtype=np.int32)

        # identity slot state & visibility timers
        # shape [K], global ff indices or -1 for empty
        self.slot_ids = None
        # per global ff id, seconds since last seen
        self.ff_t_since_last_seen = None
        self.ff_visible = None           # per global ff id, {0,1}

    def _update_ff_visible(self):
        self.ff_visible = np.zeros_like(self.ff_visible)
        if len(self.visible_ff_indices) > 0:
            self.ff_visible[self.visible_ff_indices] = 1

    def reset(self, seed=None, use_random_ff=True):
        '''
        reset the environment

        Returns
        -------
        obs: np.array
            return an observation based on the reset environment
        '''
        print('TIME before resetting:', self.time)
        super().reset(seed=seed)
        # if leveraging Gymnasium's RNG, mirror it into self.rng for consistency
        try:
            # gymnasium sets self.np_random in reset(); pull its bitgen into np.random.Generator
            self.rng = np.random.default_rng(self.np_random.bit_generator)
        except Exception:
            pass

        print('current linear_terminal_vel: ', self.linear_terminal_vel)
        print('current angular_terminal_vel: ', self.angular_terminal_vel)
        print('current dt: ', self.dt)

        print('current dv_cost_factor: ', self.dv_cost_factor)
        print('current dw_cost_factor: ', self.dw_cost_factor)
        print('current w_cost_factor: ', self.w_cost_factor)
        print('current distance2center_cost: ', self.distance2center_cost)
        print('current stop_vel_cost: ', self.stop_vel_cost)
        print('current flash_on_interval: ', self.flash_on_interval)
        print('current num_obs_ff: ', self.num_obs_ff)
        print('current reward_boundary: ', self.reward_boundary)
        print('current max_in_memory_time: ', self.max_in_memory_time)

        # randomly generate the information of the fireflies
        if use_random_ff is True:
            if self.make_ff_always_flash_on:
                self.ff_flash = None
            else:
                self.ff_flash = env_utils.make_ff_flash_from_random_sampling(
                    self.num_alive_ff,
                    duration=self.episode_len * self.dt,
                    non_flashing_interval_mean=3,
                    flash_on_interval=self.flash_on_interval
                )
            self._random_ff_positions(ff_index=np.arange(self.num_alive_ff))

        # reset agent
        self.agentr = np.array([0.0], dtype=np.float32)
        self.agentx = np.array([0.0], dtype=np.float32)
        self.agenty = np.array([0.0], dtype=np.float32)
        self.agentxy = np.zeros(2, dtype=np.float32)
        self.agentheading = self.rng.uniform(0, 2 * pi, 1).astype(np.float32)
        self.v = self.rng.uniform(-0.05, 0.05) * self.vgain
        self.w = 0.0  # initialize with no angular velocity
        self.prev_w = self.w
        self.prev_v = self.v

        # prev-obs snapshot starts empty; will be populated in beliefs()
        # ff_t_since_start_seen: per-ff timer that accumulates WHILE the ff is visible.
        # - Reset to 0 when the ff becomes invisible.
        # - Set to dt on the first update after an ff becomes newly visible (see _update_ff_visibility).
        # Interprets "time since this visibility episode started".
        self.ff_t_since_start_seen = np.zeros(
            self.num_alive_ff, dtype=np.float32)

        # ff_t_since_last_seen: per-ff timer that accumulates when the ff is NOT visible.
        # - Reset to 0 every step for ff that are currently visible.
        # - Increments by dt otherwise (see _update_ff_visibility).
        # Interprets "time elapsed since the ff was last visible" (useful for memory eligibility).
        self.ff_t_since_last_seen = np.full(
            self.num_alive_ff, 1e9, dtype=np.float32)

        self.ff_visible = np.zeros(self.num_alive_ff, dtype=np.int32)
        self._init_identity_slots()
        self._slot_valid_mask = np.zeros(self.num_obs_ff, dtype=np.int32)

        # reset or update other variables
        self.time = 0
        self.num_targets = 0
        self.episode_reward = 0
        self.JUST_CROSSED_BOUNDARY = False
        self.cost_breakdown = {'dv_cost': 0, 'dw_cost': 0, 'w_cost': 0}
        self.reward_for_each_ff = []
        self.end_episode = False
        self.action = np.array([0.0, 0.0], dtype=np.float32)
        self.obs = self.beliefs()
        if self.epi_num > 0:
            print('\n episode: ', self.epi_num)
        self.epi_num += 1
        self.num_ff_caught_in_episode = 0
        info = {}

        return self.obs, info

    def calculate_reward(self):
        '''
        Calculate the reward gained by taking an action
        '''

        self.dv = (self.prev_v - self.v) / self.dt
        self.dw = (self.prev_w - self.w) / self.dt
        w = self.action[0]

        dv_cost = self.dv**2 * self.dt * self.dv_cost_factor / 160000
        dw_cost = self.dw**2 * self.dt * self.dw_cost_factor / 630
        w_cost = w**2 * self.dt * self.w_cost_factor / 2
        self.cost_breakdown['dv_cost'] += dv_cost
        self.cost_breakdown['dw_cost'] += dw_cost
        self.cost_breakdown['w_cost'] += w_cost

        self.vel_cost = dv_cost + dw_cost + w_cost

        if self.add_vel_cost_when_catching_ff_only:
            reward = 0
        else:
            reward = - self.vel_cost

        if self.num_targets > 0:
            self.catching_ff_reward = self._get_catching_ff_reward()

            reward += self.catching_ff_reward

            # record the reward for each ff; if more than one ff is captured, the reward is divided by the number of ff captured
            self.reward_for_each_ff.extend(
                [self.catching_ff_reward / self.num_targets] * self.num_targets)

            if self.print_ff_capture_incidents:
                reward_breakdown = (
                    f'{float(self.time):.2f} action: [{round(float(self.action[0]), 3)} {round(float(self.action[1]), 3)}] '
                    f'n_targets: {self.num_targets} reward: {float(self.catching_ff_reward):.2f}'
                )
                if self.distance2center_cost > 0:
                    reward_breakdown += f' cost_for_distance2center: {float(self.cost_for_distance2center):.2f}'
                if self.stop_vel_cost > 0:
                    reward_breakdown += f' cost_for_stop_vel: {float(self.cost_for_stop_vel):.2f}'
                print(reward_breakdown)

        self.num_ff_caught_in_episode = self.num_ff_caught_in_episode + self.num_targets
        self.reward = reward
        return reward

    def step(self, action):
        '''
        take a step; the function involves calling the function state_step in the middle
        '''

        self.previous_action = getattr(
            self, 'action', np.zeros(2, dtype=np.float32))
        # work on a copy and keep dtype consistent
        action = np.asarray(action, dtype=np.float32).copy()
        self.action = self._add_noise_to_action(action)
        # if action[1] < 1:
        #     print('time: ', round(self.time, 2), 'action: ', action)

        self.time += self.dt
        # update the position of the agent
        self.state_step()

        self._check_for_captured_ff()
        self._get_ff_info()
        self._update_ff_visibility()
        self._update_identity_slots()

        self._store_topk_ff_noisy_pos()

        # update the observation
        self.obs = self.beliefs()
        # get reward
        reward = self.calculate_reward()
        self.episode_reward += reward
        if self.time >= self.episode_len * self.dt:
            self.end_episode = True
            if self.print_episode_reward_rates:
                print(
                    f'Firely capture rate for the episode:  {self.num_ff_caught_in_episode} ff for {self.time} s: -------------------> {round(self.num_ff_caught_in_episode/self.time, 2)}')
                print('Total reward for the episode: ', self.episode_reward)
                print('Cost breakdown: ', self.cost_breakdown)
                if self.distance2center_cost > 0 or self.stop_vel_cost > 0:
                    print('Reward for each ff: ', np.array(
                        self.reward_for_each_ff))
        is_time_limit = self.time >= self.episode_len * self.dt
        return self.obs, reward, False, is_time_limit, {}

    def _add_noise_to_action(self, action):
        '''
        add noise to the action
        '''
        if (abs(action[0]) <= self.angular_terminal_vel) & ((action[1] / 2 + 0.5) <= self.linear_terminal_vel):
            self.vnoise = 0.0
            self.wnoise = 0.0
            self.is_stop = True
            # calculate the deviation of the action from the terminal velocity
            self.v_stop_deviance = max(0, action[1] / 2 + 0.5 - 0.01)
            self.w_stop_deviance = max(0, abs(action[0]) - 0.01)
        else:
            self.vnoise = float(self.rng.normal(0.0, self.v_noise_std))
            self.wnoise = float(self.rng.normal(0.0, self.w_noise_std))
            self.is_stop = False
            self.v_stop_deviance = 0
            self.w_stop_deviance = 0
        new_action = np.empty_like(action, dtype=np.float32)
        new_action[0] = np.clip(float(action[0]) + self.wnoise, -1.0, 1.0)
        new_action[1] = np.clip(float(action[1]) + self.vnoise, -1.0, 1.0)
        return new_action

    def state_step(self):
        '''
        transition to a new state based on action
        '''
        self.prev_w = self.w
        self.prev_v = self.v

        self.w = self.action[0] * self.wgain
        self.v = (self.action[1] * 0.5 + 0.5) * self.vgain

        # calculate the change in the agent's position in one time step
        ah = self.agentheading
        v = self.v
        self.dx = np.cos(ah) * v
        self.dy = np.sin(ah) * v
        # update the position and direction of the agent
        self.agentx = self.agentx + self.dx * self.dt
        self.agenty = self.agenty + self.dy * self.dt
        self.agentxy[0] = self.agentx[0]
        self.agentxy[1] = self.agenty[0]
        r2 = self.agentxy[0] * self.agentxy[0] + \
            self.agentxy[1] * self.agentxy[1]
        self.agentr = np.sqrt(r2).astype(np.float32)
        self.agenttheta = np.arctan2(self.agenty, self.agentx)
        self.agentheading = np.remainder(
            self.agentheading + self.w * self.dt, 2 * pi)

        # If the agent hits the boundary of the arena, it will come out from the opposite end
        if self.agentr >= self.arena_radius:
            self.agentr = 2 * self.arena_radius - self.agentr
            self.agenttheta = self.agenttheta + pi
            self.agentx = (self.agentr * np.cos(self.agenttheta)).reshape(1,)
            self.agenty = (self.agentr * np.sin(self.agenttheta)).reshape(1,)
            self.agentxy[0] = self.agentx[0]
            self.agentxy[1] = self.agenty[0]
            self.agentheading = np.remainder(self.agenttheta - pi, 2 * pi)
            self.JUST_CROSSED_BOUNDARY = True
        else:
            self.JUST_CROSSED_BOUNDARY = False

    # -------------- flat observation (Gym API) --------------

    def beliefs(self):

        # Build identity-slot matrix
        # snapshot previous slot features if needed
        if getattr(self, 'use_prev_obs_for_invisible_pose', False):
            if hasattr(self, '_ff_slots_SN') and self._ff_slots_SN is not None:
                if self._prev_slots_SN is None or self._prev_slots_SN.shape != self._ff_slots_SN.shape:
                    self._prev_slots_SN = np.zeros_like(self._ff_slots_SN)
                else:
                    self._prev_slots_SN[:, :] = self._ff_slots_SN[:, :]
        self._get_ff_array_for_belief_identity_slots()

        # print('self.slot_ids:', self.slot_ids)
        # print('self.pose_unreliable:', self.pose_unreliable)

        # Fill preallocated obs buffer without new allocations
        obs = self.current_obs
        # slots flattened in column-major order: (S,N) -> vector
        n_slots = self.num_obs_ff * self.num_elem_per_ff
        obs[:n_slots] = self.ff_slots_flat
        off = n_slots
        if self.add_action_to_obs:
            obs[off:off + 2] = self.action
            off += 2

        # Sanity: keep within [-1,1]
        if np.any(np.abs(obs) > 1 + 1e-6):
            raise ValueError('Observation exceeded |1| bound.')

        self.prev_obs = obs  # same buffer
        return obs

    # ========================================================================================================
    # ================== Helper functions ====================================================================

    # ---- Identity slot lifecycle ----
    def _init_identity_slots(self):
        # Seed slots with currently visible ff (nearest-first); otherwise leave empty (-1)
        K = self.num_obs_ff
        self.slot_ids = np.full(K, -1, dtype=np.int32)
        # ensure distances/visibility are fresh before ranking
        self._get_ff_info()
        if len(self.visible_ff_indices) > 0:
            vis = self.visible_ff_indices
            dists = self.ff_distance_all[vis]
            order = np.argsort(dists)  # nearest → farthest
            seed = vis[order][:min(K, len(vis))]
            self.slot_ids[:len(seed)] = seed
        self._slot_valid_mask = (self.slot_ids >= 0).astype(np.int32)

    def _update_ff_visibility(self):
        # advance timers; mark visibles; reset t_seen for visibles

        self.ff_t_since_start_seen += self.dt
        self.ff_t_since_last_seen = self.ff_t_since_last_seen + self.dt

        if len(self.visible_ff_indices) > 0:
            self.ff_t_since_last_seen[self.visible_ff_indices] = 0.0

        if len(self.ff_not_visible_indices) > 0:
            self.ff_t_since_start_seen[self.ff_not_visible_indices] = 0

        self.ff_t_since_start_seen[self.newly_visible_ff] = self.dt

    def _random_ff_positions(self, ff_index):
        '''
        generate random positions for ff
        '''
        num_alive_ff = len(ff_index)
        self.ffr[ff_index] = np.sqrt(self.rng.random(num_alive_ff)).astype(
            np.float32) * self.arena_radius
        self.fftheta[ff_index] = (self.rng.random(num_alive_ff).astype(
            np.float32) * 2 * pi).astype(np.float32)
        # write into contiguous buffers
        self.ffxy[ff_index, 0] = self.ffr[ff_index] * \
            np.cos(self.fftheta[ff_index])
        self.ffxy[ff_index, 1] = self.ffr[ff_index] * \
            np.sin(self.fftheta[ff_index])
        self.ffxy_noisy[ff_index, :] = self.ffxy[ff_index, :]

    def _make_observation_space(self):
        base = self.num_obs_ff * self.num_elem_per_ff
        if self.add_action_to_obs:
            base += 2
        self.obs_space_length = base
        if gymnasium is not None:
            self.observation_space = gymnasium.spaces.Box(
                low=-1., high=1., shape=(self.obs_space_length,), dtype=np.float32)

    def _get_catching_ff_reward(self):
        self.cost_for_distance2center = 0
        self.cost_for_stop_vel = 0
        catching_ff_reward = self.reward_per_ff * self.num_targets
        if self.add_vel_cost_when_catching_ff_only:
            catching_ff_reward = max(self.reward_per_ff * self.num_targets -
                                     self.vel_cost, 0.2 * catching_ff_reward)
        if self.distance2center_cost > 0:
            self.cost_for_distance2center = self.total_deviated_target_distance * \
                (self.distance2center_cost/self.reward_boundary * 25)
            catching_ff_reward = catching_ff_reward - self.cost_for_distance2center
        if self.stop_vel_cost > 0:
            self.total_deviated_vel = self.v_stop_deviance + self.w_stop_deviance
            self.cost_for_stop_vel = self.total_deviated_vel * (self.stop_vel_cost/self.angular_terminal_vel)
            catching_ff_reward = catching_ff_reward - self.cost_for_stop_vel

        return catching_ff_reward

    def _check_for_captured_ff(self):
        self.num_targets = 0
        if not self.JUST_CROSSED_BOUNDARY:
            try:
                if self.is_stop:
                    # compare with squared threshold to avoid sqrt
                    rb2 = float(self.reward_boundary) ** 2
                    self.captured_ff_index = np.where(
                        self.ff_distance2_all <= rb2)[0].tolist()
                    self.num_targets = len(self.captured_ff_index)
                    if self.num_targets > 0:
                        self.total_deviated_target_distance = np.sum(
                            self.ff_distance_all[self.captured_ff_index])
                        self.ff_t_since_start_seen[self.captured_ff_index] = 0
                        # Replace captured ffs with new locations
                        self._random_ff_positions(self.captured_ff_index)
                        # Update info for the new ffs
                        self._update_ff_info(self.captured_ff_index)

                        # ★ unbind captured ids from slots immediately
                        if self.slot_ids is not None:
                            cap_set = set(self.captured_ff_index)
                            for s in range(self.num_obs_ff):
                                if int(self.slot_ids[s]) in cap_set:
                                    self.slot_ids[s] = -1

                        # ★ refresh visibility/distances and refill nearest-visible now
                        self._get_ff_info()
                        self._update_identity_slots()
            except AttributeError:
                pass

    def _store_topk_ff_noisy_pos(self):
        # Provide compatibility fields expected by data collectors
        # Map current identity-bound observation slots -> indices and noisy positions
        if getattr(self, 'slot_ids', None) is None:
            self.topk_indices = np.array([], dtype=np.int32)
            self.ffxy_topk_noisy = np.empty((0, 2), dtype=np.float32)
            return
        valid_mask = self.slot_ids >= 0
        if np.any(valid_mask):
            topk = self.slot_ids[valid_mask].astype(np.int32)
            self.topk_indices = topk
            self.ffxy_topk_noisy = self.ffxy_noisy[topk]
        else:
            self.topk_indices = np.array([], dtype=np.int32)
            self.ffxy_topk_noisy = np.empty((0, 2), dtype=np.float32)

    def _update_variables_when_no_ff_is_in_obs(self):
        self.ffxy_topk_noisy = np.array([])
        self.distance_topk_noisy = np.array([])
        self.angle_to_center_topk_noisy = np.array([])

    def _get_ff_info(self):

        # squared distances (N,)
        dx = self.ffxy[:, 0] - self.agentxy[0]
        dy = self.ffxy[:, 1] - self.agentxy[1]
        dist2 = dx * dx + dy * dy
        self.ff_distance2_all = dist2
        self.ff_distance_all = np.sqrt(dist2).astype(np.float32)
        self.angle_to_center_all, self.angle_to_boundary_all = env_utils.calculate_angles_to_ff(
            self.ffxy, self.agentx, self.agenty, self.agentheading,
            self.ff_radius, ffdistance=self.ff_distance_all)
        # visibility mask (bool) avoids set/unique
        vis_mask = env_utils.find_visible_ff(
            self.time, self.ff_distance_all, self.angle_to_boundary_all,
            self.invisible_distance, self.invisible_angle, self.ff_flash
        )
        # if find_visible_ff returns indices, convert to mask first
        if getattr(vis_mask, 'dtype', None) is not bool:
            mask = np.zeros(self.num_alive_ff, dtype=bool)
            mask[vis_mask] = True
            vis_mask = mask
        self._ff_is_visible[:] = vis_mask
        visible_ff_indices = np.nonzero(self._ff_is_visible)[
            0].astype(np.int32)
        self._update_visible_ff_indices(visible_ff_indices)

        return

    def _update_visible_ff_indices(self, visible_ff_indices):
        self.prev_visible_ff_indices = self.visible_ff_indices.copy() if hasattr(
            self, 'visible_ff_indices') else np.array([], dtype=np.int32)
        self.visible_ff_indices = visible_ff_indices
        self.newly_visible_ff = np.array(list(set(
            self.visible_ff_indices) - set(self.prev_visible_ff_indices)), dtype=np.int32)

        self.ff_not_visible_indices = np.array(list(set(range(
            self.num_alive_ff)) - set(self.visible_ff_indices.tolist())), dtype=np.int32)

        # get visible mask
        self.ff_visible = np.zeros_like(self.ff_visible)
        if len(self.visible_ff_indices) > 0:
            self.ff_visible[self.visible_ff_indices] = 1

    def _update_ff_info(self, ff_index):
        '''
        Refresh distance/angle and visibility for a subset of fireflies, given by
        global indices `ff_index`. Ensures `self.visible_ff_indices` stays in global space.
        '''
        ff_index = np.asarray(ff_index, dtype=np.int32)
        if ff_index.size == 0:
            return

        # 1) distances & angles for the subset (global write-back)
        sub = self.ffxy[ff_index] - self.agentxy
        self.ff_distance_all[ff_index] = np.sqrt(
            np.sum(sub * sub, axis=1)).astype(np.float32)
        angle_to_center_sub, angle_to_boundary_sub = env_utils.calculate_angles_to_ff(
            self.ffxy[ff_index], self.agentx, self.agenty, self.agentheading,
            self.ff_radius, ffdistance=self.ff_distance_all[ff_index]
        )
        self.angle_to_center_all[ff_index] = angle_to_center_sub
        self.angle_to_boundary_all[ff_index] = angle_to_boundary_sub

        # 2) per-ff flashing schedules for the same subset
        if getattr(self, 'ff_flash', None) is None:
            ff_flash_subset = None
        else:
            ff_flash_subset = [
                self.ff_flash[int(i)] for i in ff_index.tolist()]

        # 3) visibility among the updated subset (indices are RELATIVE to ff_index)
        visible_subset = env_utils.find_visible_ff(
            self.time,
            self.ff_distance_all[ff_index],
            self.angle_to_boundary_all[ff_index],
            self.invisible_distance,
            self.invisible_angle,
            ff_flash_subset
        )

        # 4) map subset indices -> GLOBAL indices
        if visible_subset.size:
            visible_global = ff_index[visible_subset]
        else:
            visible_global = np.array([], dtype=np.int32)

        # 5) rebuild global visible set: remove any entries for the updated subset, then add fresh visibles
        if getattr(self, 'visible_ff_indices', None) is None:
            base_visible = np.array([], dtype=np.int32)
        else:
            mask_keep = ~np.isin(self.visible_ff_indices, ff_index)
            base_visible = self.visible_ff_indices[mask_keep]

        visible_ff_indices = np.unique(
            np.concatenate([base_visible.astype(np.int32),
                            visible_global.astype(np.int32)])
        ).astype(np.int32)
        self._update_visible_ff_indices(visible_ff_indices)

    def _get_ff_array_for_belief_identity_slots(self):
        # Number of slots
        S = self.num_obs_ff

        # Determine output field indices and count
        has_out_fields = hasattr(self, '_slot_idx') and hasattr(
            self, 'slot_fields') and hasattr(self, 'num_elem_per_ff')
        if has_out_fields:
            out_idx = self._slot_idx
            N = self.num_elem_per_ff
        else:
            out_idx = slice(None)
            N = self.num_elem_per_ff

        # maintain same noise cadence as base class
        self._apply_noise_to_ff_in_obs()

        # Build as (S,N) so we can ravel(order='F') with no transpose
        if not hasattr(self, '_ff_slots_SN') or self._ff_slots_SN.shape != (S, N):
            self._ff_slots_SN = np.zeros((S, N), dtype=np.float32)
        slots_SN = self._ff_slots_SN

        # Vectorized processing
        valid_mask = self.slot_ids >= 0 if self.slot_ids is not None else np.zeros(
            S, dtype=bool)
        self._slot_valid_mask = valid_mask.astype(np.int32)

        # Process valid slots in batch
        valid_slots = np.where(valid_mask)[0]
        if valid_slots.size > 0:
            ffids = self.slot_ids[valid_slots].astype(np.int32)
            # choose pose source for content features
            ffxy = self.ffxy_noisy[ffids]
            theta_center, _ = env_utils.calculate_angles_to_ff(
                ffxy, self.agentx, self.agenty, self.agentheading, self.ff_radius
            )
            sin_theta = np.sin(theta_center).astype(np.float32)
            cos_theta = np.cos(theta_center).astype(np.float32)
            dx = ffxy[:, 0] - self.agentxy[0]
            dy = ffxy[:, 1] - self.agentxy[1]
            dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
            dist_clip = np.minimum(dist, self.D_max)
            d_log01 = (np.log1p(dist_clip / self.d0) *
                       self._inv_log_denom).astype(np.float32)

            # If using prev-obs for invisible pose, pull from previous slot snapshot
            if getattr(self, 'use_prev_obs_for_invisible_pose', False) and (self._prev_slots_SN is not None):
                invis_rows = self.ff_visible[ffids] < 0.5
                if np.any(invis_rows):
                    def col_for(field):
                        idx_full = self.FIELD_INDEX[field]
                        return idx_full if isinstance(out_idx, slice) else (out_idx.index(idx_full) if idx_full in out_idx else None)
                    j_dlog = col_for('d_log')
                    j_sin = col_for('sin')
                    j_cos = col_for('cos')
                    prev_rows = valid_slots[np.where(invis_rows)[0]]
                    if j_dlog is not None:
                        d_log01[invis_rows] = self._prev_slots_SN[prev_rows, j_dlog]
                    if j_sin is not None:
                        sin_theta[invis_rows] = self._prev_slots_SN[prev_rows, j_sin]
                    if j_cos is not None:
                        cos_theta[invis_rows] = self._prev_slots_SN[prev_rows, j_cos]

            # Time features normalized 0..1 using T_max (vectorized)
            t_start = np.minimum(
                self.ff_t_since_start_seen[ffids], self.T_max).astype(np.float32)
            t_last = np.minimum(
                self.ff_t_since_last_seen[ffids], self.T_max).astype(np.float32)
            t_start01 = (t_start / self.T_max).astype(np.float32)
            t_last01 = (t_last / self.T_max).astype(np.float32)

            # Visible and valid flags
            self.visible = self.ff_visible[ffids].astype(np.float32)
            valid = np.ones_like(self.visible, dtype=np.float32)

            # If using prev obs for invisible pose, overwrite content from previous slots snapshot
            if getattr(self, 'use_prev_obs_for_invisible_pose', False) and (self._prev_slots_SN is not None):
                invis_rows = self.ff_visible[ffids] < 0.5
                if np.any(invis_rows):
                    # map field names to indices within out_idx
                    def col_for(field):
                        idx_full = self.FIELD_INDEX[field]
                        return idx_full if isinstance(out_idx, slice) else (out_idx.index(idx_full) if idx_full in out_idx else None)
                    j_dlog = col_for('d_log')
                    j_sin = col_for('sin')
                    j_cos = col_for('cos')
                    prev_rows = valid_slots[np.where(invis_rows)[0]]
                    if j_dlog is not None:
                        d_log01[invis_rows] = self._prev_slots_SN[prev_rows, j_dlog]
                    if j_sin is not None:
                        sin_theta[invis_rows] = self._prev_slots_SN[prev_rows, j_sin]
                    if j_cos is not None:
                        cos_theta[invis_rows] = self._prev_slots_SN[prev_rows, j_cos]

            # compute pose_unreliable = 1 - visible
            if getattr(self, 'use_prev_obs_for_invisible_pose', False) or getattr(self, 'zero_invisible_ff_features', False):
                self.pose_unreliable = (self.visible < 0.5).astype(np.float32)
            else:
                self.pose_unreliable = np.zeros_like(
                    self.visible, dtype=np.float32)
            # assemble selected fields directly into (K,N)
            full_fields = (valid, d_log01, sin_theta, cos_theta,
                           t_start01, t_last01, self.visible, self.pose_unreliable)
            if isinstance(out_idx, slice):
                outK = np.stack(full_fields, axis=1).astype(np.float32)
            else:
                full_stack = np.stack(full_fields, axis=1)  # (K,F)
                outK = full_stack[:, out_idx].astype(np.float32)  # (K,N)
            # Optionally zero features for invisible fireflies assigned to slots
            if getattr(self, 'zero_invisible_ff_features', False):
                invis_rows = self.visible < 0.5
                if np.any(invis_rows):
                    # preserve 'valid', 't_start_seen', 't_last_seen'
                    # and optionally pose features ('d_log','sin','cos') if requested
                    protected_fields = [
                        'valid', 't_start_seen', 't_last_seen', 'pose_unreliable']
                    if getattr(self, 'use_prev_obs_for_invisible_pose', False):
                        protected_fields = protected_fields + \
                            ['d_log', 'sin', 'cos']
                    if isinstance(out_idx, slice):
                        protected_cols = [self.FIELD_INDEX[f]
                                          for f in protected_fields]
                    else:
                        protected_cols = []
                        for f in protected_fields:
                            idx_full = self.FIELD_INDEX[f]
                            try:
                                j = out_idx.index(idx_full)
                                protected_cols.append(j)
                            except ValueError:
                                pass
                    N_cols = outK.shape[1]
                    zero_mask = np.ones(N_cols, dtype=bool)
                    if len(protected_cols) > 0:
                        zero_mask[np.array(protected_cols, dtype=int)] = False
                    col_idx = np.where(zero_mask)[0]
                    row_idx = np.where(invis_rows)[0]
                    if row_idx.size and col_idx.size:
                        outK[np.ix_(row_idx, col_idx)] = 0.0
            # write into (S,N) rows for valid slots
            slots_SN[valid_slots, :] = outK

        # Clear rows for invalid slots to avoid stale values even when 'valid' not present
        invalid_slots = np.where(~valid_mask)[0]
        if invalid_slots.size > 0:
            slots_SN[invalid_slots, :] = 0.0

        # produce flat view once (no new alloc) in Fortran order
        self.ff_slots_flat = slots_SN.ravel(order='F')

    def _apply_noise_to_ff_in_obs(self, alpha_first_mem=1.0):
        # Apply perception and memory noise independently per component

        vis = self.visible_ff_indices

        prev_vis = getattr(self, '_prev_vis', np.array([], dtype=np.int32))
        newly_invisible = np.setdiff1d(prev_vis, vis, assume_unique=False)
        inv_mask = np.ones(self.num_alive_ff, dtype=bool)
        if vis.size:
            inv_mask[vis] = False
        if newly_invisible.size:
            inv_mask[newly_invisible] = False
        still_invisible = np.nonzero(inv_mask)[0].astype(np.int32)

        # 1) Visible → perception noise (truth + obs noise)
        if vis.size and self.obs_noise.perc_r > 0 and self.obs_noise.perc_th > 0:
            self._apply_perception_noise_visible()

        if self.obs_noise.mem_r > 0 and self.obs_noise.mem_th > 0:
            # 2) Newly invisible → one memory step (scaled)
            if newly_invisible.size:
                self._apply_memory_noise_ego_weber_subset(
                    newly_invisible, step_scale=alpha_first_mem)

            # 3) Still invisible → one full memory step
            if still_invisible.size:
                self._apply_memory_noise_ego_weber_subset(
                    still_invisible, step_scale=1.0)

        self._prev_vis = vis.copy()

    def _apply_memory_noise_ego_weber_subset(self, idx, step_scale=1.0):
        """
        Applies cumulative egocentric memory noise to a subset of feature positions.

        Models gradual memory drift in egocentric polar coordinates (r, θ), where 
        radial noise scales linearly with distance (σ_r ∝ r) and angular noise 
        scales inversely with distance (σ_θ ∝ 1/r), following a Weber-like law. 
        The noise magnitude is scaled by `step_scale`, making it cumulative across 
        simulation steps to reflect progressive memory degradation. 

        Unlike perception noise, memory noise accumulates over time for non-visible features.
        """

        if idx.size == 0:
            return
        # compute ego polar only for idx
        dx = self.ffxy_noisy[idx, 0] - self.agentx[0]
        dy = self.ffxy_noisy[idx, 1] - self.agenty[0]
        r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        th = np.arctan2(dy, dx).astype(np.float32) - self.agentheading[0]
        th = self._wrap_pi(th)

        r_min = self.obs_noise.r_min_for_theta
        th_floor = self.obs_noise.theta_floor
        k_r = self.obs_noise.mem_r
        k_th = self.obs_noise.mem_th

        r_i = r
        std_r = step_scale * (k_r * np.maximum(r_i, 0.0))
        std_th = step_scale * \
            np.maximum(th_floor, k_th / np.clip(r_i, r_min, None))

        dr = self.rng.normal(0.0, std_r)
        dth = self.rng.normal(0.0, std_th)

        r_new = np.clip(r_i + dr, 0.0, None)
        th_new = self._wrap_pi(th + dth)
        # back to world (subset only)
        theta_world = self._wrap_pi(th_new + self.agentheading[0])
        self.ffxy_noisy[idx, 0] = self.agentx[0] + r_new * np.cos(theta_world)
        self.ffxy_noisy[idx, 1] = self.agenty[0] + r_new * np.sin(theta_world)

    def _apply_perception_noise_visible(self):
        """
        Applies instantaneous egocentric perceptual noise to currently visible features.

        Just like memory noise, models sensory uncertainty in egocentric polar coordinates (r, θ), where 
        radial noise scales linearly with distance (σ_r ∝ r) and angular noise 
        scales inversely with distance (σ_θ ∝ 1/r), following a Weber-like law. 

        But unlike memory noise, perception noise is applied once per observation and 
        does not accumulate over time.
        """
        vis = self.visible_ff_indices
        if vis.size == 0:
            return
        # ego polar for visible subset only
        dx = self.ffxy[vis, 0] - self.agentx[0]
        dy = self.ffxy[vis, 1] - self.agenty[0]
        r_v = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        th_true = np.arctan2(dy, dx).astype(np.float32) - self.agentheading[0]
        th_true = self._wrap_pi(th_true)

        std_r = self.obs_noise.perc_r * r_v
        std_th = np.maximum(
            self.obs_noise.theta_floor,
            self.obs_noise.perc_th /
            np.clip(r_v, self.obs_noise.r_min_for_theta, None)
        )

        dr = self.rng.normal(0.0, std_r)
        dth = self.rng.normal(0.0, std_th)
        r_n = np.clip(r_v + dr, 0.0, None)
        th_n = self._wrap_pi(th_true + dth)
        theta_world = self._wrap_pi(th_n + self.agentheading[0])
        self.ffxy_noisy[vis, 0] = self.agentx[0] + r_n * np.cos(theta_world)
        self.ffxy_noisy[vis, 1] = self.agenty[0] + r_n * np.sin(theta_world)
        # last-seen feature buffers removed; prev-obs snapshot is used instead

    def _from_egocentric_polar(self, r, theta_ego):
        # ego polar → world
        theta_world = self._wrap_pi(theta_ego + self.agentheading)
        x = self.agentx + r * np.cos(theta_world)
        y = self.agenty + r * np.sin(theta_world)
        return x, y

    @staticmethod
    def _wrap_pi(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _update_identity_slots(self):
        """
        Refresh identity slots based on current ff info and strategy.

        Strategies
        ----------
        'rank_keep': keep still-eligible bindings in their *same slot indices* (position-stable);
                    fill remaining empty slots with nearest-first new IDs.
        default    : drop invalids; fill empties with nearest visible, not-yet-bound IDs.
        """

        # Normalize slot_ids shape/dtype
        K = int(self.num_obs_ff)
        if getattr(self, 'slot_ids', None) is None or getattr(self.slot_ids, 'ndim', 1) != 1 or self.slot_ids.size != K:
            self.slot_ids = np.full(K, -1, dtype=np.int32)
        else:
            self.slot_ids = np.asarray(self.slot_ids, dtype=np.int32)

        # Choose strategy
        if getattr(self, 'identity_slot_strategy', None) == 'rank_keep':
            self.slot_ids = self._assign_slots_rank_keep(K)
        else:
            self.slot_ids = self._assign_slots_default()

        self._slot_valid_mask = (self.slot_ids >= 0).astype(np.int32)

    def _eligibility_mask(self, max_ff_distance=500, max_ff_angle=np.pi/2):
        """
        Compute and return a boolean mask over firefly indices that satisfy:
        - recently seen (within self.max_in_memory_time)
        - within some distance (drop if too far)
        - within ±90° (front half-plane)
        - and, if obs_visible_only, currently visible.
        """

        mem_ok = self.ff_t_since_last_seen <= self.max_in_memory_time
        # dist_ok = self.ff_distance_all <= max_ff_distance
        # angle_ok = np.abs(self.angle_to_boundary_all) <= max_ff_angle
        # mask = mem_ok & dist_ok & angle_ok
        mask = mem_ok

        if self.obs_visible_only and (self.ff_visible is not None):
            mask = mask & self.ff_visible.astype(bool)
        return mask

    def _assign_slots_rank_keep(self, K):
        """
        Rank-keep (position-stable) assignment:
        1) find eligible & distance-ranked FFs
        2) keep persistent ones in their old slots
        3) fill empty slots with nearest unbound FFs
        """

        ff_dist = np.asarray(self.ff_distance_all, dtype=float)
        base_mask = self._eligibility_mask()

        eligible_ids = np.nonzero(base_mask)[0].astype(np.int32)
        if eligible_ids.size == 0:
            return np.full(K, -1, dtype=np.int32)

        ranked = eligible_ids[np.argsort(ff_dist[eligible_ids])]
        prev_slots = np.asarray(self.slot_ids, dtype=np.int32)

        # find still-eligible (persistent) ones
        persistent_mask = (prev_slots >= 0) & np.isin(prev_slots, ranked)
        persistent_pos = np.where(persistent_mask)[0]
        persistent_ids = prev_slots[persistent_pos]

        # ensure consistent proximity order for multiple persistents
        if persistent_ids.size:
            inv = -np.ones(ff_dist.shape[0], dtype=np.int32)
            inv[ranked] = np.arange(ranked.size, dtype=np.int32)
            sort_idx = np.argsort(inv[persistent_ids])
            persistent_pos = persistent_pos[sort_idx]
            persistent_ids = persistent_ids[sort_idx]

        # build new slots
        new_slots = np.full(K, -1, dtype=np.int32)
        if persistent_pos.size:
            new_slots[persistent_pos] = persistent_ids

        # fill empty slots with new nearest IDs
        remaining = ranked[~np.isin(
            ranked, persistent_ids)] if persistent_ids.size else ranked
        empty = np.where(new_slots < 0)[0]
        if empty.size and remaining.size:
            n = min(empty.size, remaining.size)
            new_slots[empty[:n]] = remaining[:n]

        return new_slots

    def _assign_slots_default(self):
        """
        Default assignment:
        - drop invalids among currently bound FFs
        - fill empties with nearest visible, unbound FFs
        """

        slots = np.asarray(self.slot_ids, dtype=np.int32).copy()
        base_mask = self._eligibility_mask()
        ff_dist = np.asarray(self.ff_distance_all, dtype=float)

        # drop invalids
        valid_slots_mask = slots >= 0
        if np.any(valid_slots_mask):
            ffids = slots[valid_slots_mask]
            keep_now = base_mask[ffids]
            drop_positions = np.where(valid_slots_mask)[0][~keep_now]
            if drop_positions.size:
                slots[drop_positions] = -1

        # fill empties with nearest visible ones
        vis = np.asarray(
            getattr(self, 'visible_ff_indices', []), dtype=np.int32)
        if vis.size:
            bound = slots[slots >= 0]
            cand = vis[~np.isin(vis, bound)] if bound.size else vis
            if cand.size:
                order = np.argsort(ff_dist[cand])
                candidates = cand[order]
                empty_slots = np.where(slots < 0)[0]
                n = min(empty_slots.size, candidates.size)
                if n:
                    slots[empty_slots[:n]] = candidates[:n]

        return slots
