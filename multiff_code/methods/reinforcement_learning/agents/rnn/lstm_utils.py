
import torch
import pickle
import os
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pandas as pd
import warnings
import typing
from typing import Optional
from reinforcement_learning.base_classes import run_logger

# device priority: CUDA → MPS → CPU
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class ReplayBufferLSTM:
    """
    Replay buffer for agent with LSTM network additionally using previous action, can be used
    if the hidden states are not stored (arbitrary initialization of lstm for training).
    And each sample contains the whole episode instead of a single step.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack,
                                                                   # stack for each element
                                                                   zip(*batch))
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, last_action, reward, next_state, done

    def __len__(
            # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
            self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ReplayBufferLSTM2:
    """
    Replay buffer for agent with LSTM network additionally storing previous action,
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size, *, seq_len: Optional[int] = None, burn_in: int = 0, random_window: bool = True):
        batch = random.sample(self.buffer, batch_size)

        # Determine a common core length and prefix across the batch
        T_full_list = [len(sample[2]) for sample in batch]  # len(state)
        if len(T_full_list) == 0:
            raise ValueError('ReplayBufferLSTM2 is empty')
        core_T = min(T_full_list) if seq_len is None else int(min(seq_len, min(T_full_list)))
        max_prefix_allowed = min([T - core_T for T in T_full_list]) if core_T > 0 else 0
        prefix = int(max(0, min(burn_in, max_prefix_allowed)))

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        hi_lst, ci_lst, ho_lst, co_lst = [], [], [], []

        for (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done in batch:
            T = len(state)
            low = prefix
            high = max(prefix, T - core_T)
            if random_window and high > low:
                t0_core = random.randint(low, high)
            else:
                t0_core = low
            t0_b = t0_core - prefix
            t1 = t0_core + core_T

            s_lst.append(state[t0_b:t1])
            a_lst.append(action[t0_b:t1])
            la_lst.append(last_action[t0_b:t1])
            r_lst.append(reward[t0_b:t1])
            ns_lst.append(next_state[t0_b:t1])
            d_lst.append(done[t0_b:t1])
            hi_lst.append(h_in.detach().to('cpu'))
            ci_lst.append(c_in.detach().to('cpu'))
            ho_lst.append(h_out.detach().to('cpu'))
            co_lst.append(c_out.detach().to('cpu'))

        hi_lst = torch.cat(hi_lst, dim=-2)
        ho_lst = torch.cat(ho_lst, dim=-2)
        ci_lst = torch.cat(ci_lst, dim=-2)
        co_lst = torch.cat(co_lst, dim=-2)

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
            self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """

    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass

        self.activation = activation

    def forward(self):
        pass


class NumericsConfig:
    def __init__(self, mode='warn', max_warns_per_episode=1, escalate_after=10):
        self.mode = mode  # 'silent' | 'warn' | 'error'
        self.max_warns_per_episode = max_warns_per_episode
        self.escalate_after = escalate_after


def _maybe_warn_nans(flag, where, epi_ctx, cfg: NumericsConfig):
    if not flag:
        return
    epi_ctx['nan_hits'] = epi_ctx.get('nan_hits', 0) + 1
    hits = epi_ctx['nan_hits']
    if cfg.mode == 'error':
        raise ValueError(f'NaN detected in {where}')
    if cfg.mode == 'warn':
        if hits <= cfg.max_warns_per_episode or hits % cfg.escalate_after == 0:
            warnings.warn(f'NaN detected in {where} (hit {hits})')


class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation):
        super().__init__(state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]


class ValueNetwork(ValueNetworkBase):
    def __init__(self, state_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, activation)

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        self.linear1 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class QNetworkLSTM(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper:
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, dropout=0.2)
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = self.activation(
            # linear layer for 3d input only applied on the last dim
            self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(
            lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        return x, lstm_hidden  # lstm_hidden is actually tuple: (hidden, cell)


class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper:
    Memory-based control with recurrent neural networks
    """

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(
            self._state_dim + 2 * self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, dropout=0.2)
        self.linear2 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear2.apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # single branch
        x = torch.cat([state, action, last_action], -1)
        x = self.activation(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = self.linear2(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        return x, lstm_hidden  # lstm_hidden is actually tuple: (hidden, cell)


class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """

    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass

    def evaluate(self):
        pass

    def get_action(self):
        pass

    def sample_action(self, ):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()


class SAC_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        # Exploration std annealing is driven by policy-local anneal step via set_anneal_step
        self.anneal_step = 0
        self.std_anneal_min = 1.0
        self.std_anneal_max = 1.0
        self.std_anneal_steps = 1000000

        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(
            self._state_dim + self._action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, dropout=0.2)
        self.linear3 = nn.Linear(2 * hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def _std_scale(self):
        if self.std_anneal_steps <= 0:
            return 1.0
        t = min(self.anneal_step / float(self.std_anneal_steps), 1.0)
        return self.std_anneal_max + (self.std_anneal_min - self.std_anneal_max) * t

    # set_anneal_step removed; external code should adjust anneal_step directly

    def forward(self, state, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(
            lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = torch.tanh(self.linear3(merged_branch))
        x = x.permute(1, 0, 2)  # permute back

        mean = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, lstm_hidden

    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6, device=device):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        # Guard against NaNs/Infs before constructing distribution
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        log_std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-20.0)
        std = log_std.exp().clamp_min(1e-6)  # avoid zero/NaN std

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        # TanhNormal distribution as actions; reparameterization trick
        pre_tanh = mean + std * z.to(device)
        pre_tanh = torch.nan_to_num(pre_tanh, nan=0.0)
        action_0 = torch.tanh(pre_tanh)
        action = self.action_range * action_0
        safe = (1. - action_0.pow(2)).clamp_min(1e-6)
        log_prob = Normal(mean, std).log_prob(pre_tanh) - \
            torch.log(safe + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True, device=device):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(
            device)  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(
            last_action).unsqueeze(0).unsqueeze(0).to(device)
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0,
                               neginf=-20.0).exp().clamp_min(1e-6)
        # Apply annealing to reduce exploration noise over time (in inference only)
        scale = float(self._std_scale())
        if scale != 1.0:
            std = std * scale

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        sampled = self.action_range * torch.tanh(mean + std * z)

        action_tensor = self.action_range * \
            torch.tanh(mean) if deterministic else sampled
        action = torch.nan_to_num(
            action_tensor, nan=0.0).detach().cpu().numpy()

        # print('state: ', state, 'last_action: ', last_action)
        # #print('hidden_out: ', hidden_out)
        # print('mean: ', mean.detach().cpu().numpy(), 'z:', np.round(z.detach().cpu().numpy(), 3), 'std: ', std.detach().cpu().numpy())
        # No internal increment; annealing is controlled via global_step -> set_anneal_step
        return action[0][0], hidden_out


class LSTM_SAC_Trainer():

    def __init__(self, **kwargs):
        self.gamma = kwargs.get('gamma', 0.995)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.reward_scale = kwargs.get('reward_scale', 0.5)
        self.target_entropy = kwargs.get('target_entropy', -2)
        self.soft_tau = kwargs.get('soft_tau', 0.0015)
        self.batch_size = kwargs.get('batch_size', 10)
        self.update_itr = kwargs.get('update_itr', 1)
        self.train_freq = kwargs.get('train_freq', 10)
        self.auto_entropy = kwargs.get('auto_entropy', True)
        self.replay_buffer = kwargs.get('replay_buffer', 100)
        self.seq_len = kwargs.get('seq_len', None)
        self.burn_in = kwargs.get('burn_in', 0)
        self.device = kwargs.get('device', "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),)
        # Alpha (entropy temperature) clipping bounds
        self.alpha_min = kwargs.get('alpha_min', 1e-3)
        self.alpha_max = kwargs.get('alpha_max', 4.0)

        state_space = kwargs.get('state_space')
        action_space = kwargs.get('action_space')
        action_range = kwargs.get('action_range')
        soft_q_lr = kwargs.get('soft_q_lr')
        policy_lr = kwargs.get('policy_lr')
        alpha_lr = kwargs.get('alpha_lr')

        self.soft_q_net1 = QNetworkLSTM2(
            state_space, action_space, self.hidden_dim).to(self.device)
        self.soft_q_net2 = QNetworkLSTM2(
            state_space, action_space, self.hidden_dim).to(self.device)
        self.target_soft_q_net1 = QNetworkLSTM2(
            state_space, action_space, self.hidden_dim).to(self.device)
        self.target_soft_q_net2 = QNetworkLSTM2(
            state_space, action_space, self.hidden_dim).to(self.device)
        self.policy_net = SAC_PolicyNetworkLSTM(
            state_space, action_space, self.hidden_dim, action_range).to(self.device)
        self.log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=self.device)

        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        # Initialize target networks
        for target_net, net in [(self.target_soft_q_net1, self.soft_q_net1), (self.target_soft_q_net2, self.soft_q_net2)]:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(
            self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(
            self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Logging / diagnostics
        self.log_every = kwargs.get('log_every', 200)  # updates between prints
        self._update_step = 0
        self.overall_folder = kwargs.get('overall_folder')
        self.agent_type = kwargs.get('agent_type', 'lstm')
        self.sweep_params = kwargs.get('sweep_params', {})
        self.metrics = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'q_pi_mean': [],
            'q_pi_median': [],
            'q_pi_max': [],
            'target_q_mean': [],
            'target_q_median': [],
            'td1_std': [],
            'td2_std': [],
            'alpha': [],
            'entropy': [],
            'log_prob_mean': [],
            'action_mean': [],
            'action_std': [],
            'grad_norm_policy': [],
            'grad_norm_q1': [],
            'grad_norm_q2': [],
            'reward_mean': [],
            'reward_std': [],
            'done_frac': [],
        }

    def update(self, device=device):
        self._update_step += 1
        # Sample a batch from the replay buffer
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, seq_len=self.seq_len, burn_in=self.burn_in, random_window=True)

        # Convert to tensors and move to the specified device
        state = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        last_action = torch.FloatTensor(np.array(last_action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        # After buffer windowing, compute burn-in prefix length (already bounded)
        T_full = state.shape[1]
        burn = int(max(0, min(self.burn_in, T_full - 1)))

        # Sanitize batch tensors to avoid propagating NaNs/Infs
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
        last_action = torch.nan_to_num(last_action, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        done = torch.nan_to_num(done, nan=1.0, posinf=1.0, neginf=1.0)

        # Ensure hidden states are on the correct device and sanitize
        if isinstance(hidden_in, tuple) and isinstance(hidden_out, tuple):
            hi, ci = hidden_in
            ho, co = hidden_out
            hi = torch.nan_to_num(hi.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            ci = torch.nan_to_num(ci.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            ho = torch.nan_to_num(ho.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            co = torch.nan_to_num(co.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            hidden_in = (hi, ci)
            hidden_out = (ho, co)

        # Predict Q-values
        predicted_q_value1, _ = self.soft_q_net1(
            state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(
            state, action, last_action, hidden_in)

        # Evaluate the policy with guards
        new_action, log_prob, _, _, dist_log_std, _ = self.policy_net.evaluate(
            state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(
            next_state, action, hidden_out)
        # sanitize any accidental NaNs/Infs from numerics
        new_action = torch.nan_to_num(new_action, nan=0.0).clamp_(-1.0, 1.0)
        new_next_action = torch.nan_to_num(
            new_next_action, nan=0.0).clamp_(-1.0, 1.0)
        log_prob = torch.nan_to_num(log_prob, nan=0.0)
        next_log_prob = torch.nan_to_num(next_log_prob, nan=0.0)
        dist_log_std = torch.nan_to_num(dist_log_std, nan=-5.0, posinf=2.0, neginf=-20.0)

        # Scale rewards (avoid per-batch normalization which can stall learning)
        reward = self.reward_scale * reward

        # Indices for training portion (exclude burn-in from losses)
        prefix_len = int(min(burn, state.shape[1]))
        tr = slice(prefix_len, state.shape[1])

        # Update alpha for entropy
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob[:, tr, :] +
                           self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Clamp temperature to avoid runaway
            with torch.no_grad():
                self.log_alpha.clamp_(min=-10.0, max=10.0)
            # Compute alpha from log_alpha and clip to configured bounds
            self.alpha = self.log_alpha.exp().detach()
            self.alpha.clamp_(min=self.alpha_min, max=self.alpha_max)
        else:
            # Ensure alpha is a tensor on device and clipped
            self.alpha = torch.as_tensor(1.0, device=self.device, dtype=torch.float32)
            self.alpha.clamp_(min=self.alpha_min, max=self.alpha_max)
            alpha_loss = 0

        # Compute target Q-values
        predict_target_q1, _ = self.target_soft_q_net1(
            next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(
            next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(
            predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min

        # Compute Q-value losses (exclude burn-in steps)
        q_value_loss1 = self.soft_q_criterion1(
            predicted_q_value1[:, tr, :], target_q_value[:, tr, :].detach())
        q_value_loss2 = self.soft_q_criterion2(
            predicted_q_value2[:, tr, :], target_q_value[:, tr, :].detach())

        # TD-error diagnostics (no grad)
        with torch.no_grad():
            td1 = predicted_q_value1[:, tr, :] - target_q_value[:, tr, :]
            td2 = predicted_q_value2[:, tr, :] - target_q_value[:, tr, :]
            td1_std = td1.float().std().item() if td1.numel() > 1 else float('nan')
            td2_std = td2.float().std().item() if td2.numel() > 1 else float('nan')

        # Update Q-value networks
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        grad_norm_q1 = torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=5.0)
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        grad_norm_q2 = torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=5.0)
        self.soft_q_optimizer2.step()

        # Compute policy loss
        predict_q1, _ = self.soft_q_net1(
            state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(
            state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob[:, tr, :] - predicted_new_q_value[:, tr, :]).mean()

        # Update policy network with gradient clipping
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad_norm_policy = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=5.0)
        self.policy_optimizer.step()

        # Soft update the target Q-value networks
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        # Compute diagnostic stats (no grad)
        with torch.no_grad():
            q_pi = predicted_new_q_value[:, tr, :]
            q_pi_mean = q_pi.mean().item() if q_pi.numel() > 0 else float('nan')
            q_pi_median = q_pi.median().item() if q_pi.numel() > 0 else float('nan')
            q_pi_max = q_pi.max().item() if q_pi.numel() > 0 else float('nan')
            tgt_q = target_q_value[:, tr, :]
            tgt_q_mean = tgt_q.mean().item() if tgt_q.numel() > 0 else float('nan')
            tgt_q_median = tgt_q.median().item() if tgt_q.numel() > 0 else float('nan')
            act = new_action[:, tr, :]
            act_mean = act.mean().item() if act.numel() > 0 else float('nan')
            act_std = act.std().item() if act.numel() > 1 else float('nan')
            lp = log_prob[:, tr, :]
            log_prob_mean = lp.mean().item() if lp.numel() > 0 else float('nan')
            entropy = (-lp).mean().item() if lp.numel() > 0 else float('nan')
            a_std = dist_log_std[:, tr, :].exp()
            a_std_mean = a_std.mean().item() if a_std.numel() > 0 else float('nan')
            rew = reward[:, tr, :]
            rew_mean = rew.mean().item() if rew.numel() > 0 else float('nan')
            rew_std = rew.std().item() if rew.numel() > 1 else float('nan')
            dn = done[:, tr, :]
            done_frac = dn.mean().item() if dn.numel() > 0 else float('nan')
            alpha_val = float(self.alpha.detach().item()) if torch.is_tensor(self.alpha) else float(self.alpha)

        # Prepare scalar grad norms
        grad_norm_policy_value = float(grad_norm_policy.item()) if torch.is_tensor(grad_norm_policy) else float(grad_norm_policy)
        grad_norm_q1_value = float(grad_norm_q1.item()) if torch.is_tensor(grad_norm_q1) else float(grad_norm_q1)
        grad_norm_q2_value = float(grad_norm_q2.item()) if torch.is_tensor(grad_norm_q2) else float(grad_norm_q2)

        # Persist metrics history
        self.metrics['policy_loss'].append(float(policy_loss.detach().item()))
        self.metrics['q1_loss'].append(float(q_value_loss1.detach().item()))
        self.metrics['q2_loss'].append(float(q_value_loss2.detach().item()))
        self.metrics['q_pi_mean'].append(q_pi_mean)
        self.metrics['q_pi_median'].append(q_pi_median)
        self.metrics['q_pi_max'].append(q_pi_max)
        self.metrics['target_q_mean'].append(tgt_q_mean)
        self.metrics['target_q_median'].append(tgt_q_median)
        self.metrics['td1_std'].append(td1_std)
        self.metrics['td2_std'].append(td2_std)
        self.metrics['alpha'].append(alpha_val)
        self.metrics['entropy'].append(entropy)
        self.metrics['log_prob_mean'].append(log_prob_mean)
        self.metrics['action_mean'].append(act_mean)
        self.metrics['action_std'].append(a_std_mean)
        self.metrics['grad_norm_policy'].append(grad_norm_policy_value)
        self.metrics['grad_norm_q1'].append(grad_norm_q1_value)
        self.metrics['grad_norm_q2'].append(grad_norm_q2_value)
        self.metrics['reward_mean'].append(rew_mean)
        self.metrics['reward_std'].append(rew_std)
        self.metrics['done_frac'].append(done_frac)

        # Periodic concise print for fast diagnosis
        if self._update_step % int(max(1, self.log_every)) == 0:
            print(f"Logging metrics: "
                f"[upd {self._update_step}] pi={self.metrics['policy_loss'][-1]:.3f} "
                f"q1={self.metrics['q1_loss'][-1]:.3f} q2={self.metrics['q2_loss'][-1]:.3f} "
                f"Qpi(mu/med/max)={q_pi_mean:.2f}/{q_pi_median:.2f}/{q_pi_max:.2f} "
                f"tQ(mu/med)={tgt_q_mean:.2f}/{tgt_q_median:.2f} "
                f"TDstd(q1/q2)={td1_std:.2f}/{td2_std:.2f} "
                f"alpha={alpha_val:.3f} H~={entropy:.3f} act_std={a_std_mean:.3f} "
                f"|g| p={grad_norm_policy_value:.2f} q1={grad_norm_q1_value:.2f} q2={grad_norm_q2_value:.2f} "
                f"r(mu±sd)={rew_mean:.2f}±{rew_std:.2f} done%={done_frac*100:.1f}"
            )
            # Also persist to CSV under logs
            try:
                run_logger.log_training_metrics(
                    overall_folder=self.overall_folder,
                    agent_type=self.agent_type,
                    sweep_params=self.sweep_params,
                    step=self._update_step,
                    metrics={
                        'policy_loss': float(self.metrics['policy_loss'][-1]),
                        'q1_loss': float(self.metrics['q1_loss'][-1]),
                        'q2_loss': float(self.metrics['q2_loss'][-1]),
                        'q_pi_mean': q_pi_mean,
                        'q_pi_median': q_pi_median,
                        'q_pi_max': q_pi_max,
                        'target_q_mean': tgt_q_mean,
                        'target_q_median': tgt_q_median,
                        'td1_std': td1_std,
                        'td2_std': td2_std,
                        'alpha': alpha_val,
                        'entropy': entropy,
                        'log_prob_mean': log_prob_mean,
                        'action_mean': act_mean,
                        'action_std': a_std_mean,
                        'grad_norm_policy': grad_norm_policy_value,
                        'grad_norm_q1': grad_norm_q1_value,
                        'grad_norm_q2': grad_norm_q2_value,
                        'reward_mean': rew_mean,
                        'reward_std': rew_std,
                        'done_frac': done_frac,
                    }
                )
            except Exception:
                pass

        return predicted_new_q_value.mean()

    def get_latest_metrics(self):
        return {k: (v[-1] if len(v) > 0 else None) for k, v in self.metrics.items()}

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/lstm_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '/lstm_q2')
        torch.save(self.policy_net.state_dict(), path + '/lstm_policy')
        # Persist temperature state and optimizer for stable alpha after reload
        try:
            torch.save({
                'log_alpha': self.log_alpha.detach().cpu(),
                'alpha_opt': self.alpha_optimizer.state_dict(),
            }, path + '/lstm_alpha')
            print('Saved alpha while saving model')
        except Exception:
            print('Failed to save alpha while saving model')

    def load_model(self, path):
        device = self.device
        self.soft_q_net1.load_state_dict(torch.load(
            path + '/lstm_q1', map_location=torch.device(device)))
        self.soft_q_net2.load_state_dict(torch.load(
            path + '/lstm_q2', map_location=torch.device(device)))
        self.policy_net.load_state_dict(torch.load(
            path + '/lstm_policy', map_location=torch.device(device)))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

        # Restore temperature and its optimizer if available (backward-compatible)
        alpha_path = path + '/lstm_alpha'
        try:
            payload = torch.load(alpha_path, map_location=torch.device('cpu'))
            if isinstance(payload, dict) and 'log_alpha' in payload:
                with torch.no_grad():
                    self.log_alpha.copy_(payload['log_alpha'].to(self.device))
                if 'alpha_opt' in payload and isinstance(payload['alpha_opt'], dict):
                    try:
                        self.alpha_optimizer.load_state_dict(payload['alpha_opt'])
                    except Exception:
                        pass
                # Ensure runtime alpha reflects restored log_alpha
                self.alpha = self.log_alpha.exp().detach()
                self.alpha.clamp_(min=self.alpha_min, max=self.alpha_max)
                print('Loaded alpha while loading model')
        except Exception:
            # If restore fails, fall back to current initialization
            fallback_alpha = getattr(self, 'alpha', self.log_alpha.exp().detach())
            if not torch.is_tensor(fallback_alpha):
                fallback_alpha = torch.as_tensor(fallback_alpha, device=self.device, dtype=torch.float32)
            self.alpha = fallback_alpha
            self.alpha.clamp_(min=self.alpha_min, max=self.alpha_max)
            print('Failed to load alpha while loading model')

def plot(rewards):
    # clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('sac_v2_lstm.png')
    # plt.show()


def print_last_n_alphas(list_of_alpha, n=100):
    alphas = np.array(list_of_alpha)
    if len(alphas) > n:
        alphas = alphas[-n:]
    print(f'Last {n} ALPHA: {alphas}')


def plot_eval_rewards(eval_rewards):
    plt.figure(figsize=(20, 5))
    plt.plot(eval_rewards)
    # make x labels integers
    plt.xticks(np.arange(0, len(eval_rewards), step=1))
    plt.title('Evaluation Rewards')
    plt.show()
    return


def plot_alpha(list_of_epi_for_alpha, list_of_alpha):
    plt.figure(figsize=(20, 5))
    plt.plot(list_of_epi_for_alpha, list_of_alpha)
    plt.xticks(np.arange(list_of_alpha[0], list_of_alpha[-1], step=1))
    plt.title('Alpha')
    plt.show()
    return


def _initialize_hidden_state(hidden_dim, device):
    return (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
            torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))


def _train_episode(env, sac_model, max_steps_per_eps):
    state, _ = env.reset()
    last_action = env.action_space.sample()
    episode_data = {
        'state': [], 'action': [], 'last_action': [], 'reward': [],
        'next_state': [], 'done': []
    }
    hidden_out = _initialize_hidden_state(
        sac_model.hidden_dim, sac_model.device)
    ini_hidden_in, ini_hidden_out = None, None

    epi_ctx = {}
    numerics_cfg = NumericsConfig(
        mode='warn', max_warns_per_episode=1, escalate_after=10)

    for step in range(max_steps_per_eps):
        hidden_in = hidden_out
        # Guards: sanitize inputs to policy
        if not np.isfinite(state).all():
            _maybe_warn_nans(True, 'train.state', epi_ctx, numerics_cfg)
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(hidden_in, tuple):
            h, c = hidden_in
            if (not torch.isfinite(h).all()) or (not torch.isfinite(c).all()):
                _maybe_warn_nans(True, 'train.hidden_in',
                                 epi_ctx, numerics_cfg)
                hidden_in = _initialize_hidden_state(
                    sac_model.hidden_dim, sac_model.device)
        # Select action without tracking gradients and detach hidden state to avoid graph growth
        with torch.no_grad():
            # use current policy anneal step snapshot
            action, hidden_out = sac_model.policy_net.get_action(
                state, last_action, hidden_in, deterministic=False)
        # Reuse a detached hidden state at the next step
        if isinstance(hidden_out, tuple):
            h, c = hidden_out
            hidden_out = (h.detach(), c.detach())
        else:
            hidden_out = hidden_out.detach()
        next_state, reward, done, _, _ = env.step(action)
        # increment policy's anneal step after each env step (training-only)
        current_as = getattr(sac_model.policy_net, 'anneal_step', 0)
        setattr(sac_model.policy_net, 'anneal_step', int(current_as) + 1)

        if step == 0:
            ini_hidden_in, ini_hidden_out = hidden_in, hidden_out

        episode_data['state'].append(state)
        episode_data['action'].append(action)
        episode_data['last_action'].append(last_action)
        episode_data['reward'].append(reward)
        episode_data['next_state'].append(next_state)
        episode_data['done'].append(done)

        state, last_action = next_state, action

        if step % sac_model.train_freq == 0 and len(sac_model.replay_buffer) > sac_model.batch_size:
            for _ in range(sac_model.update_itr):
                sac_model.update(device=sac_model.device)

        if done:
            break

    sac_model.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_data['state'], episode_data['action'],
                                 episode_data['last_action'], episode_data['reward'], episode_data['next_state'], episode_data['done'])
    return np.sum(episode_data['reward'])


def evaluate_lstm_agent(env, sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True):
    cum_reward = 0
    was_training = sac_model.policy_net.training
    sac_model.policy_net.eval()
    with torch.no_grad():
        for _ in range(num_eval_episodes):
            state, _ = env.reset()
            last_action = env.action_space.sample()

            hidden_out = _initialize_hidden_state(
                sac_model.hidden_dim, sac_model.device)

            epi_ctx = {}
            numerics_cfg = NumericsConfig(
                mode='warn', max_warns_per_episode=1, escalate_after=10)
            for step in range(max_steps_per_eps):
                hidden_in = hidden_out
                # Guards: sanitize inputs to policy
                if not np.isfinite(state).all():
                    _maybe_warn_nans(True, 'eval.state', epi_ctx, numerics_cfg)
                    state = np.nan_to_num(
                        state, nan=0.0, posinf=0.0, neginf=0.0)
                if isinstance(hidden_in, tuple):
                    h, c = hidden_in
                    if (not torch.isfinite(h).all()) or (not torch.isfinite(c).all()):
                        _maybe_warn_nans(True, 'eval.hidden_in',
                                         epi_ctx, numerics_cfg)
                        hidden_in = _initialize_hidden_state(
                            sac_model.hidden_dim, sac_model.device)

                # Select action without grads during evaluation and detach hidden state
                action, hidden_out = sac_model.policy_net.get_action(
                    state, last_action, hidden_in, deterministic=deterministic)
                if step < 50:
                    print('Action in evaluation: ', action)
                if isinstance(hidden_out, tuple):
                    h, c = hidden_out
                    hidden_out = (h.detach(), c.detach())
                else:
                    hidden_out = hidden_out.detach()
                next_state, reward, done, _, _ = env.step(action)
                cum_reward += reward
                state, last_action = next_state, action
                if done:
                    break
    if was_training:
        sac_model.policy_net.train()
    return cum_reward / num_eval_episodes


def save_best_model(sac_model, whether_save_replay_buffer=True, dir_name=None):
    if dir_name is None:
        dir_name = sac_model.model_folder_name

    os.makedirs(dir_name, exist_ok=True)
    sac_model.save_model(dir_name)

    if whether_save_replay_buffer:
        save_replay_buffer(sac_model, dir_name)


def save_replay_buffer(sac_model, dir_name):
    buffer_path = os.path.join(dir_name, 'buffer.pkl')
    with open(buffer_path, 'wb') as f:
        pickle.dump(sac_model.replay_buffer.buffer, f)
