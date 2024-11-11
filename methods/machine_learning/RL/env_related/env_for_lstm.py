import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from machine_learning.RL.env_related import env_utils, base_env, env_for_sb3, more_envs

import os
import torch
import numpy as np
import math
import gymnasium
from gymnasium import spaces
from torch.linalg import vector_norm
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'



class EnvForLSTM(base_env.MultiFF):  
    # Transform the MultiFF environment for the LSTM agent

    def __init__(self, episode_len=1024, distance2center_cost=2, add_ff_time_since_start_visible=True, **kwargs):

        super().__init__(episode_len=episode_len, max_in_memory_time=0, distance2center_cost=distance2center_cost, **kwargs)
        
        # The obs space no longer include "memory"
        self.add_ff_time_since_start_visible = add_ff_time_since_start_visible
        self.num_elem_per_ff = 4 if self.add_ff_time_since_start_visible else 3
        self._make_observation_space(self.num_elem_per_ff)
        self.topk_indices = torch.tensor([])

    def _get_ff_array_for_belief(self):
        self._get_ff_array_for_belief_common(self.visible_ff_indices, add_memory=False, add_ff_time_since_start_visible=self.add_ff_time_since_start_visible)


class EnvForLSTM2(more_envs.MultiFF_2):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible_time_range = self.flash_on_interval + self.dt

    def update_assigned_pos_in_obs(self):
        self._base_update_assigned_pos_in_obs(self.visible_ff_indices)


class CollectInformationLSTM(EnvForLSTM):
    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)

    def reset(self, seed=None, use_random_ff=True):
        self.obs, _ = super().reset(use_random_ff=use_random_ff, seed=seed)
        more_envs.BaseCollectInformation.initialize_ff_information(self)
        info = {}
        return self.obs, info
    
    def calculate_reward(self):
        #print('action:', self.action)
        reward = super().calculate_reward()
        more_envs.BaseCollectInformation.add_to_ff_information_after_capturing_ff(self)
        return reward
