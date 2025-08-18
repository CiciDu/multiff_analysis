from machine_learning.RL.env_related import base_env, more_envs

import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnvForSB3(base_env.MultiFF):
    # The MultiFirefly-Task RL environment

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

    def _get_ff_array_for_belief(self):
        self._get_ff_array_for_belief_common(
            self.ff_in_memory_indices, add_memory=True, add_ff_time_since_start_visible=False)

    def _further_process_after_check_for_num_targets(self):
        super()._further_process_after_check_for_num_targets()
        self._update_ff_memory_and_uncertainty()


class EnvForSB3_2(more_envs.MultiFF_2):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible_time_range = self.flash_on_interval + \
            self.max_in_memory_time + self.dt

    def update_assigned_pos_in_obs(self):
        self._base_update_assigned_pos_in_obs(self.ff_in_memory_indices)

    def _update_ff_time_since_start_visible(self):
        self.not_in_memory_ff_indices = torch.tensor(list(set(range(
            self.num_alive_ff)) - set(self.ff_in_memory_indices.tolist())), dtype=torch.int)
        self._update_ff_time_since_start_visible_base_func(
            self.not_in_memory_ff_indices)

    def _further_process_after_check_for_num_targets(self):
        self._update_ff_memory_and_uncertainty()
        self._update_ff_time_since_start_visible()
        self.update_assigned_pos_in_obs()


class CollectInformation(more_envs.BaseCollectInformation):
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Specifically, ff_information has 8 columns: 
    [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """

    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)

    def _get_ff_array_for_belief(self):
        return EnvForSB3._get_ff_array_for_belief(self)

    def _further_process_after_check_for_num_targets(self):
        super()._further_process_after_check_for_num_targets()
        self._update_ff_memory_and_uncertainty()
