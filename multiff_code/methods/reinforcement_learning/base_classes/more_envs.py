from reinforcement_learning.base_classes import base_env
import os
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BaseCollectInformation(base_env.MultiFF):
    """
    Wrapper around MultiFF that maintains a DataFrame `ff_information`
    tracking firefly lifetimes across captures and respawns.
    Columns:
        [ff_lifetime_id, ffx, ffy, t_spawn, t_despawn, t_capture,
         agent_x_at_capture, agent_y_at_capture, index_in_ff_flash]
    """

    def __init__(self, episode_len=16000, print_ff_capture_incidents=True,
                 print_episode_reward_rates=True, **kwargs):
        super().__init__(episode_len=episode_len,
                         print_ff_capture_incidents=print_ff_capture_incidents,
                         print_episode_reward_rates=print_episode_reward_rates,
                         **kwargs)

        self.ff_information_colnames = [
            "ff_lifetime_id", "ffx", "ffy",
            "t_spawn", "t_despawn", "t_capture",
            "agent_x_at_capture", "agent_y_at_capture",
            "index_in_ff_flash"
        ]

    def reset(self, seed=None, use_random_ff=True):
        self.obs, _ = super().reset(use_random_ff=use_random_ff, seed=seed)
        self.initialize_ff_information()
        return self.obs, {}

    def initialize_ff_information(self):
        """Initialize ff_information with one row per currently alive firefly."""
        self.ff_information = pd.DataFrame(
            np.ones([self.num_alive_ff, len(self.ff_information_colnames)]) * (-9999),
            columns=self.ff_information_colnames
        )
        self.ff_information.loc[:, "ff_lifetime_id"] = np.arange(self.num_alive_ff)
        self.ff_information.loc[:, "index_in_ff_flash"] = np.arange(self.num_alive_ff)
        self.ff_information.loc[:, "ffx"] = self.ffxy[:, 0] + self.arena_center_global[0]
        self.ff_information.loc[:, "ffy"] = self.ffxy[:, 1] + self.arena_center_global[1]
        self.ff_information.loc[:, "t_spawn"] = 0
        self.ff_information[["index_in_ff_flash", "ff_lifetime_id"]] = (
            self.ff_information[["index_in_ff_flash", "ff_lifetime_id"]].astype(int)
        )

    def calculate_reward(self):
        reward = super().calculate_reward()
        self.add_to_ff_information_after_capturing_ff()
        return reward

    def add_to_ff_information_after_capturing_ff(self):
        """Update ff_information when fireflies are captured."""
        if self.num_targets > 0:
            for slot_idx in self.captured_ff_index:
                last_idx = np.where(
                    self.ff_information["index_in_ff_flash"] == slot_idx
                )[0][-1]
                self.ff_information.loc[last_idx, "t_capture"] = self.time
                self.ff_information.loc[last_idx, "t_despawn"] = self.time
                self.ff_information.loc[last_idx, "agent_x_at_capture"] = self.agentxy[0].item() + self.arena_center_global[0].item()
                self.ff_information.loc[last_idx, "agent_y_at_capture"] = self.agentxy[1].item() + self.arena_center_global[1].item()

            new_ff_info = pd.DataFrame(
                np.ones([self.num_targets, len(self.ff_information_colnames)]) * (-9999),
                columns=self.ff_information_colnames
            )
            new_ff_info.loc[:, "ff_lifetime_id"] = np.arange(
                len(self.ff_information),
                len(self.ff_information) + self.num_targets
            )
            ffxy_global = self.ffxy + self.arena_center_global
            new_ff_info.loc[:, "index_in_ff_flash"] = np.array(self.captured_ff_index)
            new_ff_info.loc[:, "ffx"] = ffxy_global[self.captured_ff_index, 0]
            new_ff_info.loc[:, "ffy"] = ffxy_global[self.captured_ff_index, 1]
            new_ff_info.loc[:, "t_spawn"] = self.time
            new_ff_info[["ff_lifetime_id", "index_in_ff_flash"]] = new_ff_info[
                ["ff_lifetime_id", "index_in_ff_flash"]
            ].astype(int)
            self.ff_information = pd.concat(
                [self.ff_information, new_ff_info], axis=0
            ).reset_index(drop=True)

    def recenter_and_respawn_ff(self, respawn_outer_radius=1000):
        """Override MultiFF.recenter_and_respawn_ff to also update ff_information."""
        super().recenter_and_respawn_ff(respawn_outer_radius=respawn_outer_radius)

        if self.respawn_idx.size == 0:
            return

        # mark their despawn times
        self.ff_information.loc[self.ff_information['index_in_ff_flash'].isin(self.respawn_idx), "t_despawn"] = self.time

        # add new rows for new lifetimes
        new_ff_info = pd.DataFrame(
            np.ones([len(self.respawn_idx), len(self.ff_information_colnames)]) * (-9999),
            columns=self.ff_information_colnames
        )
        new_ff_info.loc[:, "ff_lifetime_id"] = np.arange(
            len(self.ff_information),
            len(self.ff_information) + len(self.respawn_idx)
        )
        
        ffxy_global = self.ffxy + self.arena_center_global
        new_ff_info.loc[:, "index_in_ff_flash"] = self.respawn_idx.astype(int)
        new_ff_info.loc[:, "ffx"] = ffxy_global[self.respawn_idx, 0]
        new_ff_info.loc[:, "ffy"] = ffxy_global[self.respawn_idx, 1]
        new_ff_info.loc[:, "t_spawn"] = self.time
        new_ff_info[["ff_lifetime_id", "index_in_ff_flash"]] = new_ff_info[
            ["ff_lifetime_id", "index_in_ff_flash"]
        ].astype(int)

        self.ff_information = pd.concat(
            [self.ff_information, new_ff_info], axis=0
        ).reset_index(drop=True)
