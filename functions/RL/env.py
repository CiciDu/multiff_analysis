import os
import torch
import numpy as np
import pandas as pd
from math import pi
from gym import spaces, Env
from torch.linalg import vector_norm
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'


class MultiFF(Env):
    # The MultiFirefly-Task RL environment

    def __init__(self, action_noise_std=0.005, ffxy_noise_std=3, num_obs_ff=2, full_memory=3, invisible_distance=400,
                num_ff=200, reward_per_ff=100, time_cost=0, dt=0.25, episode_len=1024, print_ff_capture_incidents=True, 
                 print_episode_reward_rates=True):
        super(MultiFF, self).__init__()
        self.linear_terminal_vel = 0.01  
        self.angular_terminal_vel = 1
        # self.linear_terminal_vel = 0.0005 #0.1/200  
        # self.angular_terminal_vel = 0.00222   #0.0035/(pi/2)
        self.num_ff = num_ff
        self.invisible_distance = invisible_distance
        self.reward_per_ff = reward_per_ff
        self.time_cost = time_cost
        self.dt = dt
        self.episode_len = episode_len
        self.print_ff_capture_incidents = print_ff_capture_incidents
        self.print_episode_reward_rates = print_episode_reward_rates

        # parameters
        self.action_noise_std = action_noise_std
        self.ffxy_noise_std = ffxy_noise_std
        self.num_obs_ff = num_obs_ff 
        self.full_memory = full_memory


        self.observation_space = spaces.Box(low=-1., high=1., shape=(self.num_obs_ff * 4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.vgain = 200
        self.wgain = pi / 2
        self.arena_radius = 1000
        self.ff_radius = 10
        self.reward_boundary = 25
        self.invisible_angle = 2 * pi / 9
        self.epi_num = 0
        # self.reward_per_episode = []

        self.ffx = torch.zeros(self.num_ff)
        self.ffy = torch.zeros(self.num_ff)
        self.ffx_noisy = torch.zeros(self.num_ff)
        self.ffy_noisy = torch.zeros(self.num_ff)
        self.ffr = torch.zeros(self.num_ff)
        self.fftheta = torch.zeros(self.num_ff) 


    def reset(self, use_random_ff = True):
        """
        reset the environment

        Returns
        -------
        obs: np.array
            return an observation based on the reset environment
           
        """
        print("current linear_terminal_vel: ", self.linear_terminal_vel)
        print("current angular_terminal_vel: ", self.angular_terminal_vel)
        
        # randomly generate the information of the fireflies
        if use_random_ff is True:
            self.ff_flash = make_ff_flash_from_random_sampling(self.num_ff, duration=self.episode_len * self.dt, non_flashing_interval_mean=3, flashing_interval_mean=0.3)
            self.random_ff_positions(ff_index=torch.arange(self.num_ff))
        self.ff_memory_all = torch.ones([self.num_ff, ])

        # reset the information of the agent
        self.agentx = torch.tensor([0])
        self.agenty = torch.tensor([0])
        self.agentr = torch.tensor([0])
        self.agentxy = torch.tensor([0, 0])
        self.agentheading = torch.zeros(1).uniform_(0, 2 * pi)
        self.dv = torch.zeros(1).uniform_(-0.05, 0.05) * self.vgain
        self.dw = torch.zeros(1) # initialize with no angular velocity

        # reset or update other variables
        self.time = 0
        self.num_targets = 0
        self.episode_reward = 0
        self.end_episode = False
        self.obs = self.beliefs().numpy()
        if self.epi_num > 0:
            print("\n episode: ", self.epi_num)
        self.epi_num += 1
        self.num_ff_caught_in_episode = 0
        return self.obs


    def random_ff_positions(self, ff_index):
        """
        generate random positions for ff

        Parameters
        -------
        ff_index: array-like
            indices of fireflies whose positions will be randomly generated
           
        """
        num_ff = len(ff_index)
        self.ffr[ff_index] = torch.sqrt(torch.rand(num_ff)) * self.arena_radius
        self.fftheta[ff_index] = torch.rand(num_ff) * 2 * pi
        self.ffx[ff_index] = torch.cos(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffy[ff_index] = torch.sin(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffxy = torch.stack((self.ffx, self.ffy), dim=1)
        # The following variables store the locations of all the fireflies with uncertainties
        self.ffx_noisy[ff_index] = self.ffx[ff_index].clone()
        self.ffy_noisy[ff_index] = self.ffy[ff_index].clone()
        self.ffxy_noisy = torch.stack((self.ffx_noisy, self.ffy_noisy), dim=1)



    def calculate_reward(self):
        """
        Calculate the reward gained by taking an action

        Returns
        -------
        reward: num
            the reward for the current step
           
        """
        reward = -self.time_cost
        # action_cost=((self.previous_action[1]-self.action[1])**2+(self.previous_action[0]-self.action[0])**2)*self.mag_cost
        ## Note: To incorporate action_cost as done above, we need to store previous_action and also incorporate it into decision_info
        
        self.num_targets = 0
        # If the velocity of the current step is low enough for the action to be considered a stop
        if (abs(self.sys_vel[0]) <= self.angular_terminal_vel) & (abs(self.sys_vel[1]) <= self.linear_terminal_vel):
        # if (abs(self.sys_vel[0]) <= 0.01) & (abs(self.sys_vel[1]) <= 0.01):
        # if (abs(self.sys_vel[1]) <= 0.01):
            self.captured_ff_index = (self.ff_distance_all <= self.reward_boundary).nonzero().reshape(-1).tolist()
            self.num_targets = len(self.captured_ff_index)
            # If the monkey hs captured at least 1 ff
            if self.num_targets > 0:  
                reward = reward + self.reward_per_ff * self.num_targets
                # Replace the captured ffs with ffs of new locations
                self.random_ff_positions(self.captured_ff_index)
                if self.print_ff_capture_incidents:
                    print(round(self.time, 2), "sys_vel: ", [round(i, 4) for i in self.sys_vel.tolist()], "n_targets: ", self.num_targets)
        self.num_ff_caught_in_episode = self.num_ff_caught_in_episode+self.num_targets
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
                print('Firely capture rate for the episode: ', self.num_ff_caught_in_episode, 'ff for', self.time, 's')
                #print('Total reward for the episode: ', self.episode_reward)
        return self.obs, reward, self.end_episode, {}


    def state_step(self, action): 
        """
        transition to a new state based on action

        Parameters
        ----------
        action: array-like, shape=(2,) 
            containing the linear and angular velocities for the current point, in the range of (-1, 1)
        
        """
        # Generate noise for the linear and angular velocities
        vnoise = torch.distributions.Normal(0, torch.ones([1, 1])*2).sample() * self.action_noise_std
        wnoise = torch.distributions.Normal(0, torch.ones([1, 1])*2).sample() * self.action_noise_std
        self.dw = (action[0] + wnoise) * self.wgain
        self.dv = (action[1] + vnoise) * self.vgain 
        # calculate the change in the agent's position in one time step 
        self.dx = torch.cos(self.agentheading) * self.dv 
        self.dy = torch.sin(self.agentheading) * self.dv
        # update the position and direction of the agent
        self.agentx = self.agentx + self.dx.item() * self.dt
        self.agenty = self.agenty + self.dy.item() * self.dt
        self.agentxy = torch.cat((self.agentx, self.agenty))
        self.agentr = vector_norm(self.agentxy)
        self.agenttheta = torch.atan2(self.agenty, self.agentx)
        self.agentheading = torch.remainder(self.agentheading + self.dw.item() * self.dt, 2*pi)

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


    def beliefs(self):
        """
        return an observation based on the action taken

        Returns
        -------
        obs: np.array
            the current observation
           
        """
        # find the distances of all fireflies to the agent
        self.ff_distance_all = vector_norm(self.ffxy - self.agentxy, dim=1)
        self.angle_to_center_all, angle_to_boundary_all = calculate_angles_to_ff_in_pytorch(self.ffxy, self.agentx, self.agenty, self.agentheading, 
                                                                            self.ff_radius, ffdistance=self.ff_distance_all)
        self.visible_ff_indices = find_visible_ff(self.time, self.ff_distance_all, angle_to_boundary_all, self.invisible_distance, self.invisible_angle, self.ff_flash)

        # update memory of all fireflies
        self.ff_memory_all[self.ff_memory_all > 0] = self.ff_memory_all[self.ff_memory_all > 0] - 1
        self.ff_memory_all[self.visible_ff_indices] = self.full_memory
        # find ffs that are in memory
        self.ff_in_memory_indices = (self.ff_memory_all > 0).nonzero().reshape(-1)


        # calculate the std of the uncertainty that will be added to the distance and angle of each firefly
        ff_uncertainty_all = np.sign(self.full_memory - self.ff_memory_all) * self.ffxy_noise_std * np.sqrt(self.ff_distance_all)
        # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
        self.ffx_noisy, self.ffy_noisy, self.ffxy_noisy = update_noisy_ffxy(self.ffx_noisy, self.ffy_noisy, self.ffx, self.ffy, ff_uncertainty_all, self.visible_ff_indices)

        # Now we shall select the fireflies that are to be included in the current observation,
        # by finding the fireflies with the smallest distances that are in memory (which include the visible ones)

        # First consider the case where there are enough fireflies (>= the size of obs space) that are in memory
        if torch.numel(self.ff_in_memory_indices) >= self.num_obs_ff:
            # rank the in-memory ff based on distance and find the indices of the top k (k = self.num_obs_ff)
            topk_ff = torch.topk(-self.ff_distance_all[self.ff_in_memory_indices], self.num_obs_ff).indices
            self.topk_indices = self.ff_in_memory_indices[topk_ff]
            self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
            self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
            self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                              self.ff_radius, ffdistance=self.distance_topk_noisy)
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy, 
                                         self.ff_memory_all[self.topk_indices]), dim=0)
        elif torch.numel(self.ff_in_memory_indices) == 0:
            self.topk_indices = torch.tensor([])
            self.ff_array = torch.tensor([[0], [0], [self.invisible_distance], [0]]).repeat([1, self.num_obs_ff])
        else:
            # Find k fireflies with the shortest distances in the memory 
            sorted_distance, sorted_indices = torch.sort(-self.ff_distance_all[self.ff_in_memory_indices])
            self.topk_indices = self.ff_in_memory_indices[sorted_indices]
            self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
            self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
            self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                              self.ff_radius, ffdistance=self.distance_topk_noisy)
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy, 
                                         self.ff_memory_all[self.topk_indices]), dim=0)
            # Since the number of fireflies are insufficient to fill up the obs space
            num_needed_ff = self.num_obs_ff - torch.numel(self.ff_in_memory_indices)
            needed_ff = torch.tensor([[0], [0], [self.invisible_distance], [0]]).repeat([1, num_needed_ff])
            self.ff_array = torch.cat([self.ff_array.reshape([4, -1]), needed_ff], dim=1)

        ## If normalizing the observation
        # ff_array[0:2,:] = ff_array[0:2,:]/pi
        # ff_array[2,:] = (ff_array[2,:]/self.invisible_distance-0.5)*2
        # ff_array[3,:] = (ff_array[3,:]/20-0.5)*2
        obs = torch.flatten(self.ff_array.transpose(0, 1))
        return obs








class CollectInformation(MultiFF):  
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Specifically, ff_information has 8 columns: 
    [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """


    def __init__(self, num_obs_ff=2, full_memory=3, action_noise_std=0.005, ffxy_noise_std=3, invisible_distance=400,
                 num_ff=200, reward_per_ff=100, time_cost=0, dt=0.25, episode_len=16000, print_ff_capture_incidents=True,
                 print_episode_reward_rates=True):
        super().__init__(num_obs_ff=num_obs_ff, full_memory=full_memory, action_noise_std=action_noise_std, ffxy_noise_std=ffxy_noise_std, 
                         invisible_distance=invisible_distance, num_ff=num_ff, reward_per_ff=reward_per_ff, time_cost=time_cost, 
                         dt=dt, episode_len=episode_len, print_ff_capture_incidents=print_ff_capture_incidents, print_episode_reward_rates=print_episode_reward_rates)
        
        self.ff_information_colnames = ["unique_identifier", "ffx", "ffy", "time_start_to_be_alive", "time_captured", 
                                        "mx_when_catching_ff", "my_when_catching_ff", "index_in_ff_flash"]
        self.ff_information = pd.DataFrame(np.ones([self.num_ff, 8])*(-9999), columns = self.ff_information_colnames).astype(int)


    def reset(self, use_random_ff=True):
        self.obs = super().reset(use_random_ff=use_random_ff)
        self.ff_information.loc[:, "unique_identifier"] = np.arange(self.num_ff)
        self.ff_information.loc[:, "index_in_ff_flash"] = np.arange(self.num_ff)
        self.ff_information.loc[:, "ffx"] = self.ffx.numpy()
        self.ff_information.loc[:, "ffy"] = self.ffy.numpy()
        self.ff_information.loc[:, "time_start_to_be_alive"] = 0
        return self.obs


    def calculate_reward(self):
        reward = super().calculate_reward()
        if self.num_targets > 0:
            for index_in_ff_flash in self.captured_ff_index:
                # Find the row index of the last firefly (the row that has the largest row number) in ff_information that has the same index_in_ff_lash.
                last_corresponding_ff_identifier = np.where(self.ff_information.loc[:, "index_in_ff_flash"]==index_in_ff_flash)[0][-1]
                # Here, last_corresponding_ff_index is equivalent to unique_identifier, which is equivalent to the index of the dataframe
                self.ff_information.loc[last_corresponding_ff_identifier, "time_captured"] = self.time
                self.ff_information.loc[last_corresponding_ff_identifier, "mx_when_catching_ff"] = self.agentx.item()
                self.ff_information.loc[last_corresponding_ff_identifier, "my_when_catching_ff"] = self.agenty.item()
            # Since the captured fireflies will be replaced, we shall add new rows to ff_information to store the information of the new fireflies
            self.new_ff_info = pd.DataFrame(np.ones([self.num_targets, 8])*(-9999), columns = self.ff_information_colnames).astype(int) 
            self.new_ff_info.loc[:, "unique_identifier"] = np.arange(len(self.ff_information), len(self.ff_information)+self.num_targets)
            self.new_ff_info.loc[:, "index_in_ff_flash"] = np.array(self.captured_ff_index)
            self.new_ff_info.loc[:, "ffx"] = self.ffx[self.captured_ff_index].numpy()
            self.new_ff_info.loc[:, "ffy"] = self.ffy[self.captured_ff_index].numpy()
            self.new_ff_info.loc[:, "time_start_to_be_alive"] = self.time
            self.ff_information = pd.concat([self.ff_information, self.new_ff_info], axis = 0).reset_index(drop=True)
        return reward





class AdaptForLSTM(MultiFF):  
    # Transform the MultiFF environment for the LSTM agent

    def __init__(self, num_obs_ff=2, full_memory=3, action_noise_std=0.005, ffxy_noise_std=3, invisible_distance=400,
                 num_ff=200, reward_per_ff=100, time_cost=0, dt=0.25, episode_len=16000):

        super().__init__(num_obs_ff=num_obs_ff, full_memory=full_memory, action_noise_std=action_noise_std, ffxy_noise_std=ffxy_noise_std, 
                         invisible_distance=invisible_distance, num_ff=num_ff, reward_per_ff=reward_per_ff, time_cost=time_cost, 
                         dt=dt, episode_len=episode_len)
        
        # The obs space no longer include "memory"
        self.observation_space = spaces.Box(low=-1., high=1., shape=(self.num_obs_ff*3,),dtype=np.float32)
        
        # Curriculum training will be used, so the duration of the flashing-on intervals of the fireflies will 
        # start from 2.1 and gradually decrease to 0.3
        self.flash_on_interval = 2.1
        
        # At the beginning, we will vary the reward gained by catching each firefly based on how far away the agent is from the center of the firefly
        self.distance2center_cost = 2
        
        # Whenever the reward hits a certain threshold, we will decrease the duration of the flashing-on intervals by 0.3, unless it's already 0.3;
        # Therefore, when the reward threshold is hit, the following boolean variable will turn true
        self.decrease_flash_on_interval = False
        

    def reset(self, use_random_ff=True):
        if self.decrease_flash_on_interval:
            self.flash_on_interval -= 0.3
            self.decrease_flash_on_interval = False
        print("flash_on_interval: ",   self.flash_on_interval)
        self.obs = super().reset(use_random_ff=use_random_ff)
        return self.obs


    def calculate_reward(self):
        reward = super().calculate_reward()
        if self.num_targets > 0:  
        # At the earlier stage of the curriculum training, the reward gained by catching each firefly will 
        # decrease based on how far away the agent is from the center of the firefly
            total_deviated_distance = torch.sum(self.ff_distance_all[self.captured_ff_index]).item()
            reward = reward - total_deviated_distance*self.distance2center_cost     
        return reward


    def beliefs(self):
        # The beliefs function will be rewritten because the observation no longer has a memory component;
        # The manually added noise to the observation is also eliminated, because the LSTM network will contain noise;
        # Thus, in the environment for LSTM agents, ffxy_noisy is equivalent to ffxy.

        self.ff_distance_all = vector_norm(self.ffxy - self.agentxy, dim=1)
        self.angle_to_center_all, angle_to_boundary_all = calculate_angles_to_ff_in_pytorch(self.ffxy, self.agentx, self.agenty, self.agentheading, 
                                                                            self.ff_radius, ffdistance=self.ff_distance_all)
        self.visible_ff_indices = find_visible_ff(self.time, self.ff_distance_all, angle_to_boundary_all, self.invisible_distance, self.invisible_angle, self.ff_flash)

        # Now we shall select the fireflies that are to be included in the current observation,
        # by finding the fireflies with the smallest distances that are visible (note: memory is not considered here.)
        if torch.numel(self.visible_ff_indices) >= self.num_obs_ff:
            # rank the in-memory ff based on distance and find the indices of the top k (k = self.num_obs_ff)
            topk_ff = torch.topk(-self.ff_distance_all[self.visible_ff_indices], self.num_obs_ff).indices
            self.topk_indices = self.visible_ff_indices[topk_ff]
            self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
            self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
            self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                              self.ff_radius, ffdistance=self.distance_topk_noisy)
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy), dim=0)
        elif torch.numel(self.visible_ff_indices) == 0:
            self.topk_indices = torch.tensor([])
            self.ff_array = torch.tensor([[0], [0], [self.invisible_distance]]).repeat([1, self.num_obs_ff])
        else:
            # Find k fireflies with the shortest distances in the memory 
            sorted_distance, sorted_indices = torch.sort(-self.ff_distance_all[self.visible_ff_indices])
            self.topk_indices = self.visible_ff_indices[sorted_indices]
            self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
            self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
            self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                              self.ff_radius, ffdistance=self.distance_topk_noisy)
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy), dim=0)
            # Since the number of fireflies are insufficient to fill up the obs space
            num_needed_ff = self.num_obs_ff - torch.numel(self.visible_ff_indices)
            needed_ff = torch.tensor([[0], [0], [self.invisible_distance]]).repeat([1, num_needed_ff])
            self.ff_array = torch.stack([self.ff_array.reshape([3, -1]), needed_ff], dim=1)

        ## If normalizing the observation
        # ff_array[0:2,:] = ff_array[0:2,:]/pi
        self.ff_array[2, :] = (self.ff_array[2, :]/self.invisible_distance-0.5)*2
        return torch.flatten(self.ff_array.transpose(0, 1))



class CollectInformationLSTM(AdaptForLSTM, CollectInformation):
    def __init__(self, num_obs_ff=2, full_memory=3, action_noise_std=0.005, ffxy_noise_std=3, invisible_distance=400,
                 num_ff=200, reward_per_ff=100, time_cost=0, dt=0.25, episode_len=16000):
        super().__init__(num_obs_ff=num_obs_ff, full_memory=full_memory, action_noise_std=action_noise_std, ffxy_noise_std=ffxy_noise_std, 
                         invisible_distance=invisible_distance, num_ff=num_ff, reward_per_ff=reward_per_ff, time_cost=time_cost, 
                         dt=dt, episode_len=episode_len)






def make_ff_flash_from_random_sampling(num_ff, duration, non_flashing_interval_mean=3, flashing_interval_mean=0.3):
  
    """
    Randomly sample flashing-on durations for each firefly

    Parameters
    ----------
    num_ff: num
        number of fireflies for which flashing-on durations will be sampled
    duration: num
        total length of time
    non_flashing_interval_mean: num
        the mean length of the gap between every two flashing-on intervals
    flashing_interval_mean: num
        the length of each flashing-on interval

    Returns
    -------
    ff_flash: list
      containing the time that each firefly flashes on and off

    """

    ff_flash = []
    # for each firefly in the 200
    for i in range(num_ff):
        num_intervals = int(duration/2)
        # eandomly generate a series of intervals that will be the durations between the flashing intervals
        non_flashing_intervals = torch.poisson(torch.ones(num_intervals) * non_flashing_interval_mean)
        # also generate a series of intervals, 0.3s each, for the durations of the flashing intervals
        flashing_intervals = torch.ones(num_intervals) * flashing_interval_mean
        # make a tensor of all the starting-flashing time for the firefly; pretend that time starts from -10s so that 
        # the condition at time 0 is more natural
        t0 = torch.cumsum(non_flashing_intervals, dim=0) + torch.cumsum(flashing_intervals, dim=0) - 10
        # Also make a tensor of all the stopping-flashing time for the firefly
        t1 = t0 + flashing_intervals
        # and we should start at the interval where t1 is greater than zero
        meaningful = (t1 > 0)
        t0 = t0[meaningful]
        t1 = t1[meaningful]
        # then, to prevent future errors, make all negative values 0
        t0[t0 < 0] = 0
        ff_flash.append(torch.stack((t0, t1), dim=1))
    return ff_flash




def calculate_angles_to_ff_in_pytorch(ffxy, agentx, agenty, agentheading, ff_radius, ffdistance = None):
    
    """
    Calculate the angle of a firefly from the monkey's or the agent's perspective

    Parameters
    ----------
    ffxy: torch.tensor
        containing the x-coordinates and the y-coordinates of all fireflies
    agentx: torch.tensor, shape (1,)
        the x-coordinate of the agent
    agenty: torch.tensor, shape (1,)
        the y-coordinate of the agent
    agentheading: torch.tensor, shape (1,)
        the angle that the agent heads toward
    ff_radius: num
        the radius of the reward boundary of each firefly  
    ffdistance: torch.tensor, optional
        containing the distances of the fireflies to the agent

    Returns
    -------
    angle_to_center: torch.tensor
        containing the angles of the centers of the fireflies to the agent
    angle_to_boundary: torch.tensor
        containing the smallest angles of the reward boundaries of the fireflies to the agent

    """

    if ffdistance is None:
        agentxy = torch.cat((agentx, agenty))
        ffdistance = vector_norm(ffxy - agentxy, dim=1)
    # find the angles of the given fireflies to the agent
    angle_to_center = torch.atan2(ffxy[:, 1] - agenty, ffxy[:, 0] - agentx) - agentheading
    # make sure that the angles are between (-pi, pi]
    angle_to_center = torch.remainder(angle_to_center, 2*pi)
    angle_to_center[angle_to_center > pi] = angle_to_center[angle_to_center > pi] - 2 * pi
    # Adjust the angle based on reward boundary (i.e. find the smallest angle from the agent to the reward boundary)
    # using trignometry 
    side_opposite = ff_radius
    # hypotenuse cannot be smaller than side_opposite
    hypotenuse = torch.clamp(ffdistance, min=side_opposite)
    theta = torch.arcsin(torch.div(side_opposite, hypotenuse))
    # we use absolute values of angles here so that the adjustment will only make the angles smaller
    angle_adjusted_abs = torch.abs(angle_to_center) - torch.abs(theta)
    # thus we can find the smallest absolute angle to the firefly, which is the absolute angle to the boundary of the firefly
    angle_to_boundary_abs = torch.clamp(angle_adjusted_abs, min=0)
    # restore the signs of the angles
    angle_to_boundary = torch.sign(angle_to_center) * angle_to_boundary_abs
    return angle_to_center, angle_to_boundary



def update_noisy_ffxy(ffx_noisy, ffy_noisy, ffx, ffy, ff_uncertainty_all, visible_ff_indices):
    
    """
    Adding noise to the positions of the fireflies based on how long ago they were seen and 
    meanwhile restoring the accurate positions of the currently visible fireflies

    Parameters
    ----------
    ffx_noisy: torch.tensor
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: torch.tensor
        containing the y-coordinates of all fireflies with noise
    ffx: torch.tensor
        containing the accurate x-coordinates of all fireflies
    ffy: torch.tensor
        containing the accurate y-coordinates of all fireflies
    ff_uncertainty_all: torch.tensor
        containing the values of uncertainty of all fireflies; scaling is based on a parameter for the environment
    visible_ff_indices: torch.tensor
        containing the indices of the visible fireflies


    Returns
    -------
    ffx_noisy: torch.tensor
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: torch.tensor
        containing the y-coordinates of all fireflies with noise
    ffxy_noisy: torch.tensor
        containing the x-coordinates and the y-coordinates of all fireflies with noise

    """

    # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
    num_ff = len(ff_uncertainty_all)
    ffx_noisy = ffx_noisy + torch.normal(torch.zeros([num_ff, ]), ff_uncertainty_all)
    ffy_noisy = ffy_noisy + torch.normal(torch.zeros([num_ff, ]), ff_uncertainty_all)
    # for the visible fireflies, their positions are updated to be the real positions
    ffx_noisy[visible_ff_indices] = ffx[visible_ff_indices].clone()
    ffy_noisy[visible_ff_indices] = ffy[visible_ff_indices].clone()
    ffxy_noisy = torch.stack((ffx_noisy, ffy_noisy), dim=1)
    return ffx_noisy, ffy_noisy, ffxy_noisy



def find_visible_ff(time, ff_distance_all, ff_angle_all, invisible_distance, invisible_angle, ff_flash):
    """
    Find the indices of the fireflies that are visible at a given time

    Parameters
    ----------
    time: num
        the current moment 
    ff_distance_all: torch.tensor
        containing the distances of all the fireflies to the agent
    ff_angle_all: torch.tensor
        containing the angles (to the reward boundaries) of all the fireflies to the agent   
    invisible_distance: num
        the distance beyond which a firefly will be considered invisible
    invisible_angle: num    
        the angle beyond which a firefly will be considered invisible 
    ff_flash: list
      containing the time that each firefly flashes on and off

    Returns
    -------
    visible_ff_indices: torch.tensor
      containing the indices of the fireflies that are visible at the given time

    """

    # find fireflies that are within the visible distance and angle at this point
    visible_ff = torch.logical_and(ff_distance_all < invisible_distance, torch.abs(ff_angle_all) < invisible_angle)
    # among these fireflies, eliminate those that are not flashing on at this point
    for index in visible_ff.nonzero().reshape(-1):
        ff_flashing_durations = ff_flash[index].clone().detach()
        # if no interval contains the current time point
        if not torch.any(torch.logical_and(ff_flashing_durations[:, 0] <= time, ff_flashing_durations[:, 1] >= time)):
            visible_ff[index] = False
    visible_ff_indices = visible_ff.nonzero().reshape(-1)
    return visible_ff_indices



