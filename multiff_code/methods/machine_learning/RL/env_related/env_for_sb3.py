from machine_learning.RL.env_related import base_env, more_envs, env_utils
import gymnasium

import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnvForSB3(base_env.MultiFF):
    # The MultiFirefly-Task RL environment

    def __init__(self,
                 **kwargs):

        super().__init__(obs_visible_only=False, **kwargs)


class CollectInformation(more_envs.BaseCollectInformation):
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Specifically, ff_information has 8 columns: 
    [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """

    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)

    def _further_process_after_check_for_num_targets(self):
        # Keep base updates (visibility timers, etc.)
        super()._further_process_after_check_for_num_targets()
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
