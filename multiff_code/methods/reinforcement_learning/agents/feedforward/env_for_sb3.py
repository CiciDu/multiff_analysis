from reinforcement_learning.base_classes import base_env, more_envs, env_utils
import gymnasium

import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnvForSB3(base_env.MultiFF):
    # The MultiFirefly-Task RL environment

    def __init__(self,
                 obs_visible_only=False,
                 identity_slot_strategy='rank_keep',
                 **kwargs):

        super().__init__(obs_visible_only=obs_visible_only,
                         identity_slot_strategy=identity_slot_strategy,
                         **kwargs)


class CollectInformation(more_envs.BaseCollectInformation):
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """

    def __init__(self, episode_len=16000, **kwargs):
        super().__init__(episode_len=episode_len, **kwargs)
