

from reinforcement_learning.agents.rnn import rnn_env
from reinforcement_learning.base_classes import rl_base_utils, rl_base_class
from reinforcement_learning.agents.rnn import lstm_utils
from reinforcement_learning.agents.rnn import gru_utils
from reinforcement_learning.base_classes import env_utils
from reinforcement_learning.agents.rnn import lstm_class

import os
import pandas as pd
import gc
import pickle
import io
import copy
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class GRUforMultifirefly(lstm_class.LSTMforMultifirefly):

    def __init__(self,
                 overall_folder='multiff_analysis/RL_models/GRU_stored_models/all_agents/gen_0/',
                 model_folder_name=None,
                 add_date_to_model_folder_name=False,
                 max_in_memory_time=1,
                 seq_len=192,
                 burn_in=64,
                 **additional_env_kwargs):

        super().__init__(overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         model_folder_name=model_folder_name,
                         max_in_memory_time=max_in_memory_time,
                         seq_len=seq_len,
                         burn_in=burn_in,
                         **additional_env_kwargs)

        # mark subtype for clarity
        self.agent_type = 'gru'
        self.algorithm_name = 'gru_sac'
        self.model_files = ['gru_q1', 'gru_q2', 'gru_policy']

        self.replay_buffer_class = gru_utils.ReplayBufferGRU
        self.trainer_class = gru_utils.GRU_SAC_Trainer
        self.env_class = rnn_env.EnvForRNN

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

    def train_rnn_agent(self, env, **kwargs):
        self._common_train_loop(
            env,
            train_episode_fn=gru_utils._train_gru_episode,
            eval_agent_fn=gru_utils.evaluate_gru_agent,
            save_fn=lambda model, dir_name: lstm_utils.save_best_model(
                model, dir_name=dir_name),
            track_alpha=True,
            **kwargs
        )
