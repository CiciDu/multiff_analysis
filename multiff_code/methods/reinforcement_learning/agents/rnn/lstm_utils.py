
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

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst = [
        ], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out,
                           c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            # Ensure consistency before concatenation
            # h_in: (1, batch=1, hidden)
            hi_lst.append(h_in.detach().to('cpu'))
            ci_lst.append(c_in.detach().to('cpu'))
            ho_lst.append(h_out.detach().to('cpu'))
            co_lst.append(c_out.detach().to('cpu'))
        # Concatenate along the batch dimension (dim=-2)
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
        self.train_freq = kwargs.get('train_freq', 5)
        self.auto_entropy = kwargs.get('auto_entropy', True)
        self.replay_buffer = kwargs.get('replay_buffer', 100)
        self.device = kwargs.get('device', "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),)

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

    def update(self, device=device):
        # Sample a batch from the replay buffer
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        # Convert to tensors and move to the specified device
        state = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        last_action = torch.FloatTensor(np.array(last_action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        # Ensure hidden states are on the correct device
        if isinstance(hidden_in, tuple) and isinstance(hidden_out, tuple):
            hi, ci = hidden_in
            ho, co = hidden_out
            hidden_in = (hi.to(device), ci.to(device))
            hidden_out = (ho.to(device), co.to(device))

        # Predict Q-values
        predicted_q_value1, _ = self.soft_q_net1(
            state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(
            state, action, last_action, hidden_in)

        # Evaluate the policy with guards
        new_action, log_prob, _, _, _, _ = self.policy_net.evaluate(
            state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(
            next_state, action, hidden_out)
        # sanitize any accidental NaNs/Infs from numerics
        new_action = torch.nan_to_num(new_action, nan=0.0).clamp_(-1.0, 1.0)
        new_next_action = torch.nan_to_num(
            new_next_action, nan=0.0).clamp_(-1.0, 1.0)
        log_prob = torch.nan_to_num(log_prob, nan=0.0)
        next_log_prob = torch.nan_to_num(next_log_prob, nan=0.0)

        # Scale rewards (avoid per-batch normalization which can stall learning)
        reward = self.reward_scale * reward

        # Update alpha for entropy
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob +
                           self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.0
            alpha_loss = 0

        # Compute target Q-values
        predict_target_q1, _ = self.target_soft_q_net1(
            next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(
            next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(
            predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min

        # Compute Q-value losses
        q_value_loss1 = self.soft_q_criterion1(
            predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(
            predicted_q_value2, target_q_value.detach())

        # Update Q-value networks
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Compute policy loss
        predict_q1, _ = self.soft_q_net1(
            state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(
            state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        # Update policy network with gradient clipping
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=5.0)
        self.policy_optimizer.step()

        # Soft update the target Q-value networks
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        return predicted_new_q_value.mean()

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
                self.alpha = self.log_alpha.exp()
                print('Loaded alpha while loading model')
        except Exception:
            # If restore fails, fall back to current initialization
            self.alpha = getattr(self, 'alpha', self.log_alpha.exp())
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
        action, hidden_out = sac_model.policy_net.get_action(
            state, last_action, hidden_in, deterministic=False)
        next_state, reward, done, _, _ = env.step(action)

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


def evaluate_agent(env, sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True):
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

                action, hidden_out = sac_model.policy_net.get_action(
                    state, last_action, hidden_in, deterministic=deterministic)
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
