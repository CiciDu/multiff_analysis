import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')

from machine_learning.RL.env_related import env_utils

import os
import torch
import numpy as np
import pandas as pd
import math
from math import pi
import gymnasium
import gc
from torch.linalg import vector_norm
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'


class MultiFF(gymnasium.Env):
    # The MultiFirefly-Task RL environment

    def __init__(self, 
                 action_noise_std=0.0, 
                 ffxy_noise_std=1, 
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
                 add_cost_when_catching_ff_only=False,
                 linear_terminal_vel=0.01,
                 angular_terminal_vel=0.01,
                 dt=0.1, 
                 episode_len=1024, 
                 print_ff_capture_incidents=True, 
                 print_episode_reward_rates=True,
                 add_action_to_obs=True,
                 ):
        
        super().__init__()
        self.linear_terminal_vel = linear_terminal_vel
        self.angular_terminal_vel = angular_terminal_vel
        # self.linear_terminal_vel = 0.0005 #0.1/200  
        # self.angular_terminal_vel = 0.00222   #0.0035/(pi/2)
        self.num_alive_ff = num_alive_ff
        self.flash_on_interval = flash_on_interval
        self.invisible_distance = invisible_distance
        self.make_ff_always_flash_on = make_ff_always_flash_on
        self.reward_per_ff = reward_per_ff
        self.dv_cost_factor = dv_cost_factor
        self.dw_cost_factor = dw_cost_factor
        self.w_cost_factor = w_cost_factor
        self.distance2center_cost = distance2center_cost
        self.dt = dt
        self.episode_len = episode_len
        self.print_ff_capture_incidents = print_ff_capture_incidents
        self.print_episode_reward_rates = print_episode_reward_rates
        self.add_action_to_obs = add_action_to_obs
        self.add_cost_when_catching_ff_only = add_cost_when_catching_ff_only

        # parameters
        self.action_noise_std = action_noise_std
        self.ffxy_noise_std = ffxy_noise_std
        self.num_obs_ff = num_obs_ff 
        self.max_in_memory_time = max_in_memory_time 
        self.full_memory = math.ceil(self.max_in_memory_time/self.dt)

        self.num_elem_per_ff = 4
        self._make_observation_space(self.num_elem_per_ff)
        self.action_space = gymnasium.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.vgain = 200
        self.wgain = pi / 2
        self.arena_radius = 1000
        self.ff_radius = 10
        self.reward_boundary = 25
        self.invisible_angle = 2 * pi / 9
        self.epi_num = 0
        self.time = 0
        self.current_obs = torch.zeros(self.obs_space_length)
        # self.reward_per_episode = []

        self.ffx = torch.zeros(self.num_alive_ff)
        self.ffy = torch.zeros(self.num_alive_ff)
        self.ffx_noisy = torch.zeros(self.num_alive_ff)
        self.ffy_noisy = torch.zeros(self.num_alive_ff)
        self.ffr = torch.zeros(self.num_alive_ff)
        self.fftheta = torch.zeros(self.num_alive_ff) 

        self.visible_time_range = None
        self.ff_in_memory_indices = torch.tensor([])
        self.visible_ff_indices = torch.tensor([])

    def reset(self, seed=None, use_random_ff=True):
        
        """
        reset the environment

        Returns
        -------
        obs: np.array
            return an observation based on the reset environment  
        """
        print('TIME before resetting:', self.time)
        super().reset(seed=seed)

        print("current linear_terminal_vel: ", self.linear_terminal_vel)
        print("current angular_terminal_vel: ", self.angular_terminal_vel)
        print("current dt: ", self.dt)
        print('current full_memory: ', self.full_memory)
        
        print('current dv_cost_factor: ', self.dv_cost_factor)
        print('current dw_cost_factor: ', self.dw_cost_factor)
        print('current w_cost_factor: ', self.w_cost_factor)
        print('current distance2center_cost: ', self.distance2center_cost)
        print('current flash_on_interval: ', self.flash_on_interval)
        print('current num_obs_ff: ', self.num_obs_ff)
        print('current max_in_memory_time: ', self.max_in_memory_time)
        
        # randomly generate the information of the fireflies
        if use_random_ff is True:
            if self.make_ff_always_flash_on:
                self.ff_flash = None
            else:
                self.ff_flash = env_utils.make_ff_flash_from_random_sampling(self.num_alive_ff, duration=self.episode_len * self.dt, 
                                                                             non_flashing_interval_mean=3, flash_on_interval=self.flash_on_interval)
            self._random_ff_positions(ff_index=torch.arange(self.num_alive_ff))
        self.ff_memory_all = torch.ones([self.num_alive_ff, ])
        self.ff_time_since_start_visible = torch.zeros([self.num_alive_ff, ])

        # reset the information of the agent
        self.agentx = torch.tensor([0])
        self.agenty = torch.tensor([0])
        self.agentr = torch.tensor([0])
        self.agentxy = torch.tensor([0, 0])
        self.agentheading = torch.zeros(1).uniform_(0, 2 * pi)
        self.v = torch.zeros(1).uniform_(-0.05, 0.05) * self.vgain
        self.w = torch.zeros(1) # initialize with no angular velocity
        self.prev_w = self.w
        self.prev_v = self.v

        # reset or update other variables
        self.time = 0
        self.num_targets = 0
        self.episode_reward = 0
        self.cost_for_the_current_ff = 0
        self.JUST_CROSSED_BOUNDARY = False
        self.cost_breakdown = {'dv_cost': 0, 'dw_cost': 0, 'w_cost': 0}
        self.reward_for_each_ff = []
        self.end_episode = False
        self.action = np.array([0, 0])
        self.obs = self.beliefs().numpy()
        if self.epi_num > 0:
            print("\n episode: ", self.epi_num)
        self.epi_num += 1
        self.num_ff_caught_in_episode = 0
        info = {}

        return self.obs, info


    def calculate_reward(self):
        """
        Calculate the reward gained by taking an action

        Returns
        -------
        reward: num
            the reward for the current step
           
        """

        self.vgain = 200
        self.wgain = pi / 2
        # dv_cost = ((self.previous_action[1]-self.action[1])/self.dt)**2 * self.dv_cost_factor
        # dw_cost = ((self.previous_action[0]-self.action[0])/self.dt)**2 * self.dw_cost_factor
        # w_cost = (self.action[0]/self.dt)**2 * self.w_cost_factor
        self.dv = (self.prev_v - self.v)/self.dt
        self.dw = (self.prev_w - self.w)/self.dt
        w = self.action[0]
        dv_cost = self.dv**2 * self.dt * self.dv_cost_factor/160000 # 800 is used to prevent the cost from being too large
        dw_cost = self.dw**2 * self.dt * self.dw_cost_factor/630
        w_cost = w**2 * self.dt * self.w_cost_factor/2
        self.cost_breakdown['dv_cost'] += dv_cost
        self.cost_breakdown['dw_cost'] += dw_cost
        self.cost_breakdown['w_cost'] += w_cost
        ## Note: To incorporate action_cost as done above, we need to store previous_action and also incorporate it into decision_info
        
        if self.add_cost_when_catching_ff_only:
            self.cost_for_the_current_ff += dv_cost + dw_cost + w_cost
            reward = 0
        else:
            reward = - dv_cost - dw_cost - w_cost

        if self.num_targets > 0:  
            #reward = reward + self.reward_per_ff * self.num_targets
            if self.add_cost_when_catching_ff_only:
                self.catching_ff_reward = max(self.reward_per_ff * self.num_targets - self.cost_for_the_current_ff, 0.2 * self.catching_ff_reward)
            reward += self.catching_ff_reward
            self.reward_for_each_ff.extend([self.catching_ff_reward/self.num_targets] * self.num_targets)
            self.cost_for_the_current_ff = 0

            # make ff_memory to be zero for those captured ff
            if self.print_ff_capture_incidents:
                print(round(self.time, 2), "sys_vel: ", [round(i, 4) for i in self.sys_vel.tolist()], "n_targets: ", self.num_targets, "reward: ", round(self.catching_ff_reward, 2))
                # print('prev_obs:', self.prev_obs.reshape([self.num_obs_ff, -1]))
                # print('current_obs:', self.current_obs.reshape([self.num_obs_ff, -1]))
                # print('top_k_indices:', self.topk_indices)
                # print('captured_ff_index:', self.captured_ff_index)
        self.num_ff_caught_in_episode = self.num_ff_caught_in_episode+self.num_targets
        self.reward = reward
        return reward


    def step(self, action):
        """
        take a step; the function involves calling the function state_step in the middle

        Parameters
        ----------
        action: array-like, shape=(2,) 
            containing the linear and angular velocities for the current point, in the range of (-1, 1)

        Returns
        -------
        self.obs: np.array
            the new observation
        reward: num
            the reward gained by taking the action
        self.end_episode: bool
            whether to end the current episode
        {}: dic
            a placeholder, for conforming to the format of the gym environment
           
        """
        
        self.previous_action = self.action
        self.action = action

        action = torch.tensor(action)
        
        # Change the range of the velocity part of the action from (-1, 1) to (0, 1)
        action[1] = action[1] / 2 + 0.5
        # storing the value for later evaluation of whether the step can be considered as a stop
        self.sys_vel = action.clone()
        self.time += self.dt
        # update the position of the agent
        self.state_step(action)
        # update the observation
        self.obs = self.beliefs().numpy()
        # get reward
        reward = self.calculate_reward()
        self.episode_reward += reward
        if self.time >= self.episode_len * self.dt:
            self.end_episode = True
            if self.print_episode_reward_rates:
                print(f'Firely capture rate for the episode:  {self.num_ff_caught_in_episode} ff for {self.time} s: -------------------> {round(self.num_ff_caught_in_episode/self.time, 2)}')
                print('Total reward for the episode: ', self.episode_reward)
                print('Cost breakdown: ', self.cost_breakdown)
                if self.distance2center_cost > 0:
                    print('Reward for each ff: ', np.array(self.reward_for_each_ff))

        return self.obs, reward, self.end_episode, False, {}


    def state_step(self, action): 
        """
        transition to a new state based on action

        Parameters
        ----------
        action: array-like, shape=(2,) 
            containing the linear and angular velocities for the current point, in the range of (-1, 1)
        
        """
        self.prev_w = self.w
        self.prev_v = self.v
        # Generate noise for the linear and angular velocities
        vnoise = torch.distributions.Normal(0, torch.ones([1, 1])*2).sample() * (self.action_noise_std * self.dt)
        wnoise = torch.distributions.Normal(0, torch.ones([1, 1])*2).sample() * (self.action_noise_std * self.dt)

        self.w = (action[0] + wnoise) * self.wgain
        self.v = (action[1] + vnoise) * self.vgain 
        # calculate the change in the agent's position in one time step 
        self.dx = torch.cos(self.agentheading) * self.v 
        self.dy = torch.sin(self.agentheading) * self.v
        # update the position and direction of the agent
        self.agentx = self.agentx + self.dx.item() * self.dt
        self.agenty = self.agenty + self.dy.item() * self.dt
        self.agentxy = torch.cat((self.agentx, self.agenty))
        self.agentr = vector_norm(self.agentxy)
        self.agenttheta = torch.atan2(self.agenty, self.agentx)
        self.agentheading = torch.remainder(self.agentheading + self.w.item() * self.dt, 2*pi)

        # If the agent hits the boundary of the arena, it will come out form the opposite end
        if self.agentr >= self.arena_radius:
            # Calculate how far the agent has stepped out of the arena, which will be counted as 
            # going towards the center from the other end of the arena
            self.agentr = 2 * self.arena_radius - self.agentr
            # the direction of the agent turns 180 degrees as it comes out from the other side of the arena
            self.agenttheta = self.agenttheta + pi
            # update the position and direction of the agent
            self.agentx = (self.agentr * torch.cos(self.agenttheta)).reshape(1, )
            self.agenty = (self.agentr * torch.sin(self.agenttheta)).reshape(1, )
            self.agentxy = torch.cat((self.agentx, self.agenty))
            self.agentheading = torch.remainder(self.agenttheta-pi, 2*pi)
            self.JUST_CROSSED_BOUNDARY = True
        else:
            self.JUST_CROSSED_BOUNDARY = False


    def beliefs(self):
        # The beliefs function will be rewritten because the observation no longer has a memory component;
        # The manually added noise to the observation is also eliminated, because the LSTM network will contain noise;
        # Thus, in the environment for LSTM agents, ffxy_noisy is equivalent to ffxy.

        self._get_ff_info()

        # see if any ff is caught; if so, they will be removed from the memory so that the new ff (with a new location) will not be included in the observation
        self._check_for_num_targets()

        self._further_process_after_check_for_num_targets()

        self._get_ff_array_for_belief()

        obs = torch.flatten(self.ff_array.transpose(0, 1))

        # append action to the observation
        if self.add_action_to_obs:
            obs = torch.cat((obs, torch.tensor(self.action)), dim=0)

        self.prev_obs = self.current_obs.clone()
        self.current_obs = obs

        # if any element in obs has its absolute value greater than 1, raise an error
        if torch.any(torch.abs(obs) > 1):
            raise ValueError('The observation has an element with an absolute value greater than 1')

        return obs












    # ========================================================================================================
    # ================== The following functions are helper functions ========================================
    
    def _random_ff_positions(self, ff_index):
        """
        generate random positions for ff

        Parameters
        -------
        ff_index: array-like
            indices of fireflies whose positions will be randomly generated
           
        """
        num_alive_ff = len(ff_index)
        self.ffr[ff_index] = torch.sqrt(torch.rand(num_alive_ff)) * self.arena_radius
        self.fftheta[ff_index] = torch.rand(num_alive_ff) * 2 * pi
        self.ffx[ff_index] = torch.cos(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffy[ff_index] = torch.sin(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffxy = torch.stack((self.ffx, self.ffy), dim=1)
        # The following variables store the locations of all the fireflies with uncertainties
        self.ffx_noisy[ff_index] = self.ffx[ff_index].clone()
        self.ffy_noisy[ff_index] = self.ffy[ff_index].clone()
        self.ffxy_noisy = torch.stack((self.ffx_noisy, self.ffy_noisy), dim=1)

    
    def _make_observation_space(self, num_elem_per_ff):
        self.obs_space_length = self.num_obs_ff * self.num_elem_per_ff + 2 if self.add_action_to_obs else self.num_obs_ff * self.num_elem_per_ff
        self.observation_space = gymnasium.spaces.Box(low=-1., high=1., shape=(self.obs_space_length,),dtype=np.float32)


    def _get_catching_ff_reward(self):
        catching_ff_reward = self.reward_per_ff * self.num_targets
        if self.add_cost_when_catching_ff_only:
            catching_ff_reward = max(self.reward_per_ff * self.num_targets - self.cost_for_the_current_ff, 0.2 * catching_ff_reward)
        if self.distance2center_cost > 0:
            # At the earlier stage of the curriculum training, the reward gained by catching each firefly will 
            # decrease based on how far away the agent is from the center of the firefly
            total_deviated_distance = torch.sum(self.ff_distance_all[self.captured_ff_index]).item()
            catching_ff_reward = catching_ff_reward - total_deviated_distance * self.distance2center_cost        
        return catching_ff_reward


    def _check_for_num_targets(self):
        self.num_targets = 0
        # If the velocity of the current step is low enough for the action to be considered a stop
        if not self.JUST_CROSSED_BOUNDARY:
            try:
                if (abs(self.sys_vel[0]) <= self.angular_terminal_vel) & (abs(self.sys_vel[1]) <= self.linear_terminal_vel):
                # if (abs(self.sys_vel[0]) <= 0.01) & (abs(self.sys_vel[1]) <= 0.01):
                # if (abs(self.sys_vel[1]) <= 0.01):
                    self.captured_ff_index = (self.ff_distance_all <= self.reward_boundary).nonzero().reshape(-1).tolist()
                    self.num_targets = len(self.captured_ff_index)
                    if self.num_targets > 0:
                        self.catching_ff_reward = self._get_catching_ff_reward()
                        self.ff_memory_all[self.captured_ff_index] = 0
                        self.ff_time_since_start_visible[self.captured_ff_index] = 0
                        # Replace the captured ffs with ffs of new locations
                        self._random_ff_positions(self.captured_ff_index)
                        # need to call get_ff_info again to update the information of the new ffs
                        self._update_ff_info(self.captured_ff_index)
                        
            except AttributeError:
                pass # This is to prevent the error that occurs when sys_vel is not defined in the first step

    def _get_ff_array_given_indices(self, add_memory=True, add_ff_time_since_start_visible=False):
        self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
        self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
        # cap self.distance_topk_noisy to be less than self.invisible_distance
        self.distance_topk_noisy = torch.minimum(self.distance_topk_noisy, torch.tensor(self.invisible_distance))
        self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                            self.ff_radius, ffdistance=self.distance_topk_noisy)
        if add_memory:
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy, 
                                         self.ff_memory_all[self.topk_indices]), dim=0)
        elif add_ff_time_since_start_visible:
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy,
                                         self.ff_time_since_start_visible[self.topk_indices]), dim=0)
        else:
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy), dim=0)
        return self.ff_array



    def _get_ff_array_for_belief_common(self, ff_indices, add_memory, add_ff_time_since_start_visible):
        if torch.numel(ff_indices) >= self.num_obs_ff:
            self.topk_indices = env_utils._get_topk_indices(ff_indices, self.ff_distance_all, self.num_obs_ff)
            self.ff_array = self._get_ff_array_given_indices(add_memory=add_memory, add_ff_time_since_start_visible=add_ff_time_since_start_visible)
        elif torch.numel(ff_indices) == 0:
            self.topk_indices = torch.tensor([])
            self._update_variables_when_no_ff_is_in_obs()
            placeholder_ff = env_utils._get_placeholder_ff(add_memory, add_ff_time_since_start_visible, self.invisible_distance)
            self.ff_array = placeholder_ff.repeat([1, self.num_obs_ff])
        else:
            self.topk_indices = env_utils._get_sorted_indices(ff_indices, self.ff_distance_all)
            self.ff_array = self._get_ff_array_given_indices(add_memory=add_memory, add_ff_time_since_start_visible=add_ff_time_since_start_visible)
            num_needed_ff = self.num_obs_ff - torch.numel(ff_indices)
            placeholder_ff = env_utils._get_placeholder_ff(add_memory, add_ff_time_since_start_visible, self.invisible_distance)
            needed_ff = placeholder_ff.repeat([1, num_needed_ff])
            self.ff_array = torch.cat([self.ff_array.reshape([self.num_elem_per_ff, -1]), needed_ff], dim=1)

        self.ff_array_unnormalized = self.ff_array.clone()
        self.ff_array = env_utils._normalize_ff_array(self.ff_array, self.invisible_distance, self.full_memory, add_memory, add_ff_time_since_start_visible, self.visible_time_range)


    def _further_process_after_check_for_num_targets(self):
        self._update_ff_time_since_start_visible()

    def _update_ff_memory_and_uncertainty(self):
        # update memory of all fireflies
        self.ff_memory_all[self.ff_memory_all > 0] = self.ff_memory_all[self.ff_memory_all > 0] - 1
        if len(self.visible_ff_indices) > 0:
            self.ff_memory_all[self.visible_ff_indices] = self.full_memory

        # for ff whose absolute angle_to_boundary is greater than 90 degrees, make memory 0
        self.ff_memory_all[torch.abs(self.angle_to_boundary_all) > pi/2] = 0
        self.ff_memory_all[self.ff_distance_all > self.invisible_distance] = 0

        # find ffs that are in memory
        self.ff_in_memory_indices = (self.ff_memory_all > 0).nonzero().reshape(-1)

        # calculate the std of the uncertainty that will be added to the distance and angle of each firefly
        ff_uncertainty_all = np.sign(self.full_memory - self.ff_memory_all) * (self.ffxy_noise_std * self.dt) * np.sqrt(self.ff_distance_all)
        # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
        self.ffx_noisy, self.ffy_noisy, self.ffxy_noisy = env_utils.update_noisy_ffxy(self.ffx_noisy, self.ffy_noisy, self.ffx, self.ffy, ff_uncertainty_all, self.visible_ff_indices)
        return

    def _update_variables_when_no_ff_is_in_obs(self):
        self.ffxy_topk_noisy = torch.tensor([])
        self.distance_topk_noisy = torch.tensor([])
        self.angle_to_center_topk_noisy = torch.tensor([])


    def _get_ff_info(self):
        try:
            self.prev_visible_ff_indices = self.visible_ff_indices.clone()
        except AttributeError:
            self.prev_visible_ff_indices = torch.tensor([])
        
        self.ff_distance_all = vector_norm(self.ffxy - self.agentxy, dim=1)
        self.angle_to_center_all, self.angle_to_boundary_all = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy, self.agentx, self.agenty, self.agentheading, 
                                                                            self.ff_radius, ffdistance=self.ff_distance_all)
        self.visible_ff_indices = env_utils.find_visible_ff(self.time, self.ff_distance_all, self.angle_to_boundary_all, self.invisible_distance, self.invisible_angle, self.ff_flash)
        return
    

    def _update_ff_info(self, ff_index):
        self.ff_distance_all[ff_index] = vector_norm(self.ffxy[ff_index] - self.agentxy, dim=1)
        self.angle_to_center_all[ff_index], self.angle_to_boundary_all[ff_index] = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy[ff_index], self.agentx, self.agenty, self.agentheading,
                                                                                                    self.ff_radius, ffdistance=self.ff_distance_all[ff_index])
        # find visible ff among the updated ff
        self.visible_ff_indices_among_updated_ff = env_utils.find_visible_ff(self.time, self.ff_distance_all[ff_index], self.angle_to_boundary_all[ff_index], self.invisible_distance, self.invisible_angle, 
                                                                             [self.ff_flash[i] for i in ff_index])
        # delete from the visible_ff_indices the indices that are in ff_index
        self.visible_ff_indices = torch.tensor([i for i in self.visible_ff_indices if i not in ff_index])
        # concatenate the visible_ff_indices_among_updated_ff to the visible_ff_indices
        self.visible_ff_indices = torch.cat((self.visible_ff_indices, self.visible_ff_indices_among_updated_ff), dim=0)
        # change dtype to int
        self.visible_ff_indices = self.visible_ff_indices.int()

        return



    def _update_ff_time_since_start_visible_base_func(self, not_visible_ff_indices):
        self.ff_time_since_start_visible += self.dt
        self.ff_time_since_start_visible[not_visible_ff_indices] = 0

        # get ff that has turned from not visible to visible
        self.newly_visible_ff = torch.tensor(list(set(self.visible_ff_indices) - set(self.prev_visible_ff_indices)), dtype=torch.int)
        self.ff_time_since_start_visible[self.newly_visible_ff] = self.dt


    def _update_ff_time_since_start_visible(self):
        self.ff_not_visible_indices = torch.tensor(list(set(range(self.num_alive_ff)) - set(self.visible_ff_indices.tolist())), dtype=torch.int)
        self._update_ff_time_since_start_visible_base_func(self.ff_not_visible_indices)
