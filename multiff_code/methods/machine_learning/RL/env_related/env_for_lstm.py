from machine_learning.RL.env_related import base_env, more_envs

import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnvForLSTM(base_env.MultiFF):
    # Transform the MultiFF environment for the LSTM agent

    def __init__(self, episode_len=1024, distance2center_cost=2, add_ff_t_since_start_seen=True, **kwargs):

        if 'max_in_memory_time' not in kwargs:
            kwargs['max_in_memory_time'] = 0
        elif kwargs['max_in_memory_time'] != 0:
            print(
                f'max_in_memory_time is {kwargs["max_in_memory_time"]}, which is not supported for LSTM. It will be set to 0.')
            kwargs['max_in_memory_time'] = 0

        super().__init__(obs_visible_only=True,
                         episode_len=episode_len,
                         distance2center_cost=distance2center_cost, **kwargs)


class CollectInformationLSTM(EnvForLSTM):
    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)
        # Add the ff_information_colnames attribute needed for initialization
        self.ff_information_colnames = ["unique_identifier", "ffx", "ffy", "time_start_to_be_alive", "time_captured",
                                        "mx_when_catching_ff", "my_when_catching_ff", "index_in_ff_flash"]

    def reset(self, seed=None, use_random_ff=True):
        self.obs, _ = super().reset(use_random_ff=use_random_ff, seed=seed)
        more_envs.BaseCollectInformation.initialize_ff_information(self)
        info = {}
        return self.obs, info

    def calculate_reward(self):
        # print('action:', self.action)
        reward = super().calculate_reward()
        more_envs.BaseCollectInformation.add_to_ff_information_after_capturing_ff(
            self)
        return reward

    def _further_process_after_check_for_num_targets(self):
        # Keep base updates
        super()._further_process_after_check_for_num_targets()
        # Provide compatibility fields for data collectors
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

