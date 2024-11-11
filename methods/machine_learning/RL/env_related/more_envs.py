import sys
from machine_learning.RL.env_related import env_utils, base_env

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



class BaseCollectInformation(base_env.MultiFF):
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Specifically, ff_information has 8 columns: 
    [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """


    def __init__(self, episode_len=16000, print_ff_capture_incidents=True,
                 print_episode_reward_rates=True, **kwargs):
        super().__init__(episode_len=episode_len, print_ff_capture_incidents=print_ff_capture_incidents, print_episode_reward_rates=print_episode_reward_rates, **kwargs)
        
        self.ff_information_colnames = ["unique_identifier", "ffx", "ffy", "time_start_to_be_alive", "time_captured", 
                                        "mx_when_catching_ff", "my_when_catching_ff", "index_in_ff_flash"]

    def reset(self, seed=None, use_random_ff=True):
        self.obs, _ = super().reset(use_random_ff=use_random_ff, seed=seed)
        self.initialize_ff_information()
        info = {}
        return self.obs, info
    

    def initialize_ff_information(self):
        self.ff_information = pd.DataFrame(np.ones([self.num_alive_ff, 8])*(-9999), columns = self.ff_information_colnames)
        self.ff_information.loc[:, "unique_identifier"] = np.arange(self.num_alive_ff)
        self.ff_information.loc[:, "index_in_ff_flash"] = np.arange(self.num_alive_ff)
        self.ff_information.loc[:, "ffx"] = self.ffx.numpy()
        self.ff_information.loc[:, "ffy"] = self.ffy.numpy()
        self.ff_information.loc[:, "time_start_to_be_alive"] = 0
        self.ff_information[["index_in_ff_flash", "unique_identifier"]] = self.ff_information[["index_in_ff_flash", "unique_identifier"]].astype(int)


    def calculate_reward(self):
        #print('action:', self.action)
        reward = super().calculate_reward()
        self.add_to_ff_information_after_capturing_ff()
        return reward


    def add_to_ff_information_after_capturing_ff(self):
        if self.num_targets > 0:
            for index_in_ff_flash in self.captured_ff_index:
                # Find the row index of the last firefly (the row that has the largest row number) in ff_information that has the same index_in_ff_lash.
                last_corresponding_ff_identifier = np.where(self.ff_information.loc[:, "index_in_ff_flash"]==index_in_ff_flash)[0][-1]
                # Here, last_corresponding_ff_index is equivalent to unique_identifier, which is equivalent to the index of the dataframe
                self.ff_information.loc[last_corresponding_ff_identifier, "time_captured"] = self.time
                self.ff_information.loc[last_corresponding_ff_identifier, "mx_when_catching_ff"] = self.agentx.item()
                self.ff_information.loc[last_corresponding_ff_identifier, "my_when_catching_ff"] = self.agenty.item()
            # Since the captured fireflies will be replaced, we shall add new rows to ff_information to store the information of the new fireflies
            self.new_ff_info = pd.DataFrame(np.ones([self.num_targets, 8])*(-9999), columns = self.ff_information_colnames)
            self.new_ff_info.loc[:, "unique_identifier"] = np.arange(len(self.ff_information), len(self.ff_information)+self.num_targets)
            self.new_ff_info.loc[:, "index_in_ff_flash"] = np.array(self.captured_ff_index)
            self.new_ff_info[["unique_identifier", "index_in_ff_flash"]] = self.new_ff_info[["unique_identifier", "index_in_ff_flash"]].astype(int)
            self.new_ff_info.loc[:, "ffx"] = self.ffx[self.captured_ff_index].numpy()
            self.new_ff_info.loc[:, "ffy"] = self.ffy[self.captured_ff_index].numpy()
            self.new_ff_info.loc[:, "time_start_to_be_alive"] = self.time
            self.ff_information = pd.concat([self.ff_information, self.new_ff_info], axis = 0).reset_index(drop=True)


class MultiFF_2(base_env.MultiFF):
    def __init__(self, num_obs_ff=15, episode_len=1024, distance2center_cost=2, add_ff_time_since_start_visible=True, **kwargs):

        super().__init__(num_obs_ff=num_obs_ff, episode_len=episode_len, distance2center_cost=distance2center_cost, **kwargs)
        self.add_ff_time_since_start_visible = add_ff_time_since_start_visible
        self.num_elem_per_ff = 3 if self.add_ff_time_since_start_visible else 2
        self._make_observation_space(self.num_elem_per_ff)
        self.assigned_pos_in_obs = {}
        self.used_obs_ff_pos = []
        self.unused_obs_ff_pos = list(range(self.num_obs_ff))


    def _further_process_after_check_for_num_targets(self):
        super()._further_process_after_check_for_num_targets()
        self.update_assigned_pos_in_obs()


    def _base_update_assigned_pos_in_obs(self, visible_or_in_memory_ff_indices):
        # make sure that the invisible ff are not included in the observation
        self._remove_ff_not_in_list_from_assigned_pos_in_obs(visible_or_in_memory_ff_indices)

        # update the assigned positions for the visible ff
        if len(visible_or_in_memory_ff_indices) <= self.num_obs_ff:
            # then all visible ff will be included in the observation
            self._add_ff_in_list_to_assigned_pos_in_obs(visible_or_in_memory_ff_indices)
        else:
            # find the top k (k=self.num_obs_ff) ff with the shortest distances
            topk_ff = torch.topk(-self.ff_distance_all[visible_or_in_memory_ff_indices], self.num_obs_ff).indices
            # remove ff from self.assigned_pos_in_obs if they are not in the top k
            self._remove_ff_not_in_list_from_assigned_pos_in_obs(visible_or_in_memory_ff_indices[topk_ff])
            # assign the top k ff to the observation if they are not already in the observation
            self._add_ff_in_list_to_assigned_pos_in_obs(visible_or_in_memory_ff_indices[topk_ff])
                    
                    

    def _get_ff_array_for_belief(self):

        self.distance_noisy = vector_norm(self.ffxy_noisy - self.agentxy, dim=1)
        self.angle_to_center_noisy, _ = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                            self.ff_radius, ffdistance=self.distance_noisy)

        # append placeholder values to self.distance_noisy, self.angle_to_center_noisy, and self.ff_time_since_start_visible
        distance_noisy = torch.cat((self.distance_noisy, torch.tensor([self.invisible_distance])))
        angle_to_center_noisy = torch.cat((self.angle_to_center_noisy, torch.tensor([0.])))
        ff_time_since_start_visible = torch.cat((self.ff_time_since_start_visible, torch.tensor([0.])))

        # reverse the key and item in self.assigned_pos_in_obs
        reversed_dict = {value: key for key, value in self.assigned_pos_in_obs.items()}

        ff_corresponding_to_obs_pos = torch.tensor([reversed_dict[pos] if pos in self.used_obs_ff_pos else self.num_alive_ff for pos in range(self.num_obs_ff)])
        distance_noisy_capped = torch.minimum(distance_noisy[ff_corresponding_to_obs_pos], torch.tensor(self.invisible_distance))
        self.ff_array = torch.stack((angle_to_center_noisy[ff_corresponding_to_obs_pos], distance_noisy_capped, ff_time_since_start_visible[ff_corresponding_to_obs_pos]), dim=0)
        
        self.ff_array_unnormalized = self.ff_array.clone()
        self.ff_array = env_utils._normalize_ff_array_for_env2(self.ff_array, self.invisible_distance, self.visible_time_range)
        

    def _remove_ff_not_in_list_from_assigned_pos_in_obs(self, ff_list):
        ff_list = ff_list.tolist()
        ff_to_remove = []
        for ff in self.assigned_pos_in_obs.keys():
            if ff not in ff_list:
                self.used_obs_ff_pos.remove(self.assigned_pos_in_obs[ff])
                self.unused_obs_ff_pos.append(self.assigned_pos_in_obs[ff])
                ff_to_remove.append(ff)
        # sort self.unused_obs_ff_pos
        self.unused_obs_ff_pos.sort()

        # remove ff_to_remove from self.assigned_pos_in_obs
        for ff in ff_to_remove:
            del self.assigned_pos_in_obs[ff]
        return
    
    def _add_ff_in_list_to_assigned_pos_in_obs(self, ff_list):
        ff_list = ff_list.tolist()
        self.unused_obs_ff_pos.sort()
        for ff in ff_list:
            if ff not in self.assigned_pos_in_obs.keys():
                self.assigned_pos_in_obs[ff] = self.unused_obs_ff_pos.pop(0)
                self.used_obs_ff_pos.append(self.assigned_pos_in_obs[ff])
        return

