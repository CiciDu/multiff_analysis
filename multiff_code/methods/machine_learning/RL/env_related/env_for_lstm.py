from machine_learning.RL.env_related import base_env, more_envs

import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnvForLSTM(base_env.MultiFF):
    # Transform the MultiFF environment for the LSTM agent

    def __init__(self, episode_len=512,
                 max_in_memory_time=2,
                 use_prev_obs_for_invisible_pose=True,
                 **kwargs):

        super().__init__(  # obs_visible_only=True,
            use_prev_obs_for_invisible_pose=use_prev_obs_for_invisible_pose,
            # for LSTM, max_in_memory_time must be 1
            max_in_memory_time=max_in_memory_time,
            episode_len=episode_len,
            **kwargs)


class CollectInformationLSTM(more_envs.BaseCollectInformation):
    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)
