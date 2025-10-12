from machine_learning.RL.env_related import env_utils, base_env

import os
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BaseCollectInformation(base_env.MultiFF):
    """
    The class wraps around the MultiFF environment so that it keeps a dataframe called ff_information that stores information crucial for later use.
    Specifically, ff_information has 8 columns: 
    [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    Note when using this wrapper, the number of steps cannot exceed that of one episode.   

    """

    def __init__(self, episode_len=16000, print_ff_capture_incidents=True,
                 print_episode_reward_rates=True, **kwargs):
        super().__init__(episode_len=episode_len, print_ff_capture_incidents=print_ff_capture_incidents,
                         print_episode_reward_rates=print_episode_reward_rates, **kwargs)

        self.ff_information_colnames = ["unique_identifier", "ffx", "ffy", "time_start_to_be_alive", "time_captured",
                                        "mx_when_catching_ff", "my_when_catching_ff", "index_in_ff_flash"]

    def reset(self, seed=None, use_random_ff=True):
        self.obs, _ = super().reset(use_random_ff=use_random_ff, seed=seed)
        self.initialize_ff_information()
        info = {}
        return self.obs, info

    def initialize_ff_information(self):
        self.ff_information = pd.DataFrame(
            np.ones([self.num_alive_ff, 8])*(-9999), columns=self.ff_information_colnames)
        self.ff_information.loc[:, "unique_identifier"] = np.arange(
            self.num_alive_ff)
        self.ff_information.loc[:, "index_in_ff_flash"] = np.arange(
            self.num_alive_ff)
        # base_env exposes positions via ffxy; split into x/y
        self.ff_information.loc[:, "ffx"] = self.ffxy[:, 0]
        self.ff_information.loc[:, "ffy"] = self.ffxy[:, 1]
        self.ff_information.loc[:, "time_start_to_be_alive"] = 0
        self.ff_information[["index_in_ff_flash", "unique_identifier"]] = self.ff_information[[
            "index_in_ff_flash", "unique_identifier"]].astype(int)

    def calculate_reward(self):
        # print('action:', self.action)
        reward = super().calculate_reward()
        self.add_to_ff_information_after_capturing_ff()
        return reward

    def add_to_ff_information_after_capturing_ff(self):
        if self.num_targets > 0:
            for index_in_ff_flash in self.captured_ff_index:
                # Find the row index of the last firefly (the row that has the largest row number) in ff_information that has the same index_in_ff_lash.
                last_corresponding_ff_identifier = np.where(
                    self.ff_information.loc[:, "index_in_ff_flash"] == index_in_ff_flash)[0][-1]
                # Here, last_corresponding_ff_index is equivalent to unique_identifier, which is equivalent to the index of the dataframe
                self.ff_information.loc[last_corresponding_ff_identifier,
                                        "time_captured"] = self.time
                self.ff_information.loc[last_corresponding_ff_identifier,
                                        "mx_when_catching_ff"] = self.agentx.item()
                self.ff_information.loc[last_corresponding_ff_identifier,
                                        "my_when_catching_ff"] = self.agenty.item()
            # Since the captured fireflies will be replaced, we shall add new rows to ff_information to store the information of the new fireflies
            self.new_ff_info = pd.DataFrame(
                np.ones([self.num_targets, 8])*(-9999), columns=self.ff_information_colnames)
            self.new_ff_info.loc[:, "unique_identifier"] = np.arange(
                len(self.ff_information), len(self.ff_information)+self.num_targets)
            self.new_ff_info.loc[:, "index_in_ff_flash"] = np.array(
                self.captured_ff_index)
            self.new_ff_info[["unique_identifier", "index_in_ff_flash"]] = self.new_ff_info[[
                "unique_identifier", "index_in_ff_flash"]].astype(int)
            # base_env exposes positions via ffxy; split into x/y for captured ff indices
            self.new_ff_info.loc[:,
                                 "ffx"] = self.ffxy[self.captured_ff_index, 0]
            self.new_ff_info.loc[:,
                                 "ffy"] = self.ffxy[self.captured_ff_index, 1]
            self.new_ff_info.loc[:, "time_start_to_be_alive"] = self.time
            self.ff_information = pd.concat(
                [self.ff_information, self.new_ff_info], axis=0).reset_index(drop=True)
