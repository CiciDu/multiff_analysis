from reinforcement_learning.agents.rnn import lstm_utils
import torch
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class SAC_PolicyNetworkGRU(lstm_utils.PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        # Exploration std annealing is driven externally by global_step via set_anneal_step
        self.anneal_step = 0
        self.std_anneal_min = 0.1   # final multiplicative scale on std
        self.std_anneal_max = 1.0   # initial multiplicative scale on std
        self.std_anneal_steps = 1000000  # steps to reach min scale

        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)

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
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.tanh(self.linear1(state))
        # branch 2
        gru_branch = torch.cat([state, last_action], -1)
        gru_branch = F.relu(self.linear2(gru_branch))
        gru_branch, gru_hidden = self.gru1(gru_branch, hidden_in)
        gru_branch = self.dropout(gru_branch)
        # merged
        merged_branch=torch.cat([fc_branch, gru_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, gru_hidden
    
    def _dev(self):
        # Resolve the device from module parameters to avoid relying on globals
        return next(self.parameters()).device

    def _std_scale(self):
        if self.std_anneal_steps <= 0:
            return 1.0
        # Linear schedule from std_anneal_max -> std_anneal_min
        t = min(self.anneal_step / float(self.std_anneal_steps), 1.0)
        return self.std_anneal_max + (self.std_anneal_min - self.std_anneal_max) * t

    # set_anneal_step removed; external code should adjust anneal_step directly

    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        # Guard against NaNs/Infs
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        log_std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-20.0)
        std = log_std.exp().clamp_min(1e-6) # avoid zero/NaN std
        
        dev = self._dev()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(dev)
        pre_tanh = mean + std * z
        pre_tanh = torch.nan_to_num(pre_tanh, nan=0.0)
        action_0 = torch.tanh(pre_tanh)  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        safe = (1. - action_0.pow(2)).clamp_min(1e-6)
        log_prob = Normal(mean, std).log_prob(pre_tanh) - torch.log(
            safe + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        dev = self._dev()
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(dev)  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(dev)
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-20.0).exp().clamp_min(1e-6)
        # Apply annealing to reduce exploration noise over time (in inference only)
        scale = float(self._std_scale())
        if scale != 1.0:
            std = std * scale
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(dev)
        sampled = self.action_range * torch.tanh(mean + std * z)
        action_tensor = self.action_range * torch.tanh(mean) if deterministic else sampled
        action = torch.nan_to_num(action_tensor, nan=0.0).detach().cpu().numpy()
        #print('mean: ', mean, 'std: ', std, 'action: ', action)
        # No internal increment; annealing is controlled via global_step -> set_anneal_step
        return action[0][0], hidden_out



class ReplayBufferGRU:
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size, to_torch: bool = False, device: Optional[torch.device] = None,
               seq_len: Optional[int] = None, burn_in: int = 0, random_window: bool = True):
        # First pick the batch
        batch = random.sample(self.buffer, batch_size)

        # Determine a common core length and prefix across the batch to ensure rectangular tensors
        T_full_list = [len(sample[2]) for sample in batch]  # len(state) per episode
        if len(T_full_list) == 0:
            raise ValueError('ReplayBufferGRU is empty')
        core_T = min(T_full_list) if seq_len is None else int(min(seq_len, min(T_full_list)))
        max_prefix_allowed = min([T - core_T for T in T_full_list]) if core_T > 0 else 0
        prefix = int(max(0, min(burn_in, max_prefix_allowed)))

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        hi_lst, ho_lst = [], []

        for h_in, h_out, state, action, last_action, reward, next_state, done in batch:
            T = len(state)
            # choose core start between [prefix, T - core_T]
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
            hi_lst.append(h_in.detach().to('cpu'))  # (1, 1, H)
            ho_lst.append(h_out.detach().to('cpu'))

        hi_lst = torch.cat(hi_lst, dim=-2).detach()  # (1, B, H)
        ho_lst = torch.cat(ho_lst, dim=-2).detach()

        if not to_torch:
            return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

        dev = device if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        hi_lst = hi_lst.to(dev)
        ho_lst = ho_lst.to(dev)

        # Convert lists of sequences to contiguous tensors on device (rectangular by construction)
        state      = torch.as_tensor(np.array(s_lst), dtype=torch.float32, device=dev)
        action     = torch.as_tensor(np.array(a_lst), dtype=torch.float32, device=dev)
        last_action= torch.as_tensor(np.array(la_lst), dtype=torch.float32, device=dev)
        reward     = torch.as_tensor(np.array(r_lst), dtype=torch.float32, device=dev)
        next_state = torch.as_tensor(np.array(ns_lst), dtype=torch.float32, device=dev)
        done       = torch.as_tensor(np.array(d_lst), dtype=torch.float32, device=dev)

        return hi_lst, ho_lst, state, action, last_action, reward, next_state, done

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)






class QNetworkGRU(lstm_utils.QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        # Single-branch GRU critic to match LSTM2 structure
        self.linear1 = nn.Linear(self._state_dim + 2 * self._action_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear2.apply(lstm_utils.linear_weights_init)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action, last_action], -1)
        x = self.activation(self.linear1(x))
        x, gru_hidden = self.gru1(x, hidden_in)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, gru_hidden



class GRU_SAC_Trainer():
    def __init__(self, **kwargs):


        self.gamma = kwargs.get('gamma', 0.995)
        self.batch_size = kwargs.get('batch_size', 10)
        self.update_itr = kwargs.get('update_itr', 1)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.reward_scale = kwargs.get('reward_scale', 0.5)
        self.target_entropy = kwargs.get('target_entropy', -2)
        self.soft_tau = kwargs.get('soft_tau', 0.0015)
        self.train_freq = kwargs.get('train_freq', 10)
        self.seq_len = kwargs.get('seq_len', None)
        self.burn_in = kwargs.get('burn_in', 0)
        self.device = kwargs.get('device', "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),)


        state_space = kwargs.get('state_space')
        action_space = kwargs.get('action_space')
        action_range = kwargs.get('action_range')
        soft_q_lr = kwargs.get('soft_q_lr')
        policy_lr = kwargs.get('policy_lr')
        alpha_lr = kwargs.get('alpha_lr')

        self.replay_buffer = kwargs.get('replay_buffer')
        self.soft_q_net1 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(self.device)
        self.soft_q_net2 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(self.device)
        self.target_soft_q_net1 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(self.device)
        self.target_soft_q_net2 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(self.device)
        self.policy_net = SAC_PolicyNetworkGRU(state_space, action_space, self.hidden_dim, action_range).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        #print('Soft Q Network (1,2): ', self.soft_q_net1)
        #print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        # soft_q_lr = 0.0015
        # policy_lr = 0.0015
        # alpha_lr  = 0.0015

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=1, auto_entropy=True, target_entropy=-2, gamma=0.975, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            batch_size, to_torch=True, device=self.device, seq_len=self.seq_len, burn_in=self.burn_in, random_window=True
        )

        batch_size = self.batch_size
        reward_scale = self.reward_scale
        gamma = self.gamma
        target_entropy = self.target_entropy
        soft_tau = self.soft_tau

        # Optionally slice to a fixed-length window starting at a random time
        # We only store the initial hidden state for each episode, so to get
        # the correct hidden at the window start we include a burn-in prefix
        # [t0_b: t0_core) and compute losses only on [t0_core: t0_core+core_T).
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

        # Sanitize hidden states
        hidden_in = torch.nan_to_num(hidden_in, nan=0.0, posinf=0.0, neginf=0.0)
        hidden_out = torch.nan_to_num(hidden_out, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure shapes match training expectations
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        if done.dim() == 2:
            done = done.unsqueeze(-1)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        #reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * reward
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        # Indices for training portion (exclude burn-in from losses)
        prefix_len = int(min(burn, state.shape[1]))
        tr = slice(prefix_len, state.shape[1])

        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob[:, tr, :] + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Clamp temperature to avoid runaway
            with torch.no_grad():
                self.log_alpha.clamp_(min=-10.0, max=10.0)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        # Apply burn-in slicing for critic losses
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1[:, tr, :], target_q_value[:, tr, :].detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2[:, tr, :], target_q_value[:, tr, :].detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=5.0)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=5.0)
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predict_q1, _= self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        # Apply burn-in slicing for policy loss
        policy_loss = (self.alpha * log_prob[:, tr, :] - predicted_new_q_value[:, tr, :]).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'/gru_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'/gru_q2')
        torch.save(self.policy_net.state_dict(), path+'/gru_policy')

    def load_model(self, path):
        # Ensure checkpoints saved on other devices (e.g., MPS) map to current device
        self.soft_q_net1.load_state_dict(torch.load(path+'/gru_q1', map_location=device))
        self.soft_q_net2.load_state_dict(torch.load(path+'/gru_q2', map_location=device))
        self.policy_net.load_state_dict(torch.load(path+'/gru_policy', map_location=device))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def _initialize_gru_hidden(hidden_dim, device=device):
    return torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device)


def _train_gru_episode(env, sac_model, max_steps_per_eps):
    state, _ = env.reset()
    last_action = env.action_space.sample()
    episode_data = {
        'state': [], 'action': [], 'last_action': [], 'reward': [],
        'next_state': [], 'done': []
    }
    # ensure hidden state is on the same device as the policy network
    dev = next(sac_model.policy_net.parameters()).device
    hidden_out = _initialize_gru_hidden(sac_model.hidden_dim, dev)
    ini_hidden_in, ini_hidden_out = None, None

    for step in range(max_steps_per_eps):
        hidden_in = hidden_out
        # Sanitize inputs
        if not np.isfinite(state).all():
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        if (not torch.isfinite(hidden_in).all()):
            hidden_in = _initialize_gru_hidden(sac_model.hidden_dim, dev)
        # Select action without tracking gradients and detach hidden state to avoid graph growth
        with torch.no_grad():
            # use current policy anneal step snapshot
            action, hidden_out = sac_model.policy_net.get_action(
                state, last_action, hidden_in, deterministic=False)
        # Reuse a detached hidden state at the next step
        if isinstance(hidden_out, tuple):
            # GRU returns a Tensor (not a tuple), but keep guard for consistency
            hidden_out = tuple(h.detach() for h in hidden_out)
        else:
            hidden_out = hidden_out.detach()
        next_state, reward, done, _, _ = env.step(action)
        # increment policy anneal step after each env step (training-only)
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
                sac_model.update(batch_size=sac_model.batch_size,
                                 reward_scale=sac_model.reward_scale,
                                 auto_entropy=True,
                                 target_entropy=sac_model.target_entropy,
                                 gamma=sac_model.gamma,
                                 soft_tau=sac_model.soft_tau)

        if done:
            break

    sac_model.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_data['state'], episode_data['action'],
                                 episode_data['last_action'], episode_data['reward'], episode_data['next_state'], episode_data['done'])
    return np.sum(episode_data['reward'])


def evaluate_gru_agent(env, sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True):
    was_training = sac_model.policy_net.training
    sac_model.policy_net.eval()
    cum_reward = 0
    with torch.no_grad():
        for _ in range(num_eval_episodes):
            state, _ = env.reset()
            last_action = env.action_space.sample()
            dev = next(sac_model.policy_net.parameters()).device
            hidden_out = _initialize_gru_hidden(sac_model.hidden_dim, dev)
            
            epi_ctx = {}
            numerics_cfg = lstm_utils.NumericsConfig(
                mode='warn', max_warns_per_episode=1, escalate_after=10)
            for step in range(max_steps_per_eps):
                hidden_in = hidden_out
                if not np.isfinite(state).all():
                    lstm_utils._maybe_warn_nans(True, 'eval.state', epi_ctx, numerics_cfg)
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                if (not torch.isfinite(hidden_in).all()):
                    lstm_utils._maybe_warn_nans(True, 'eval.hidden_in', epi_ctx, numerics_cfg)
                    hidden_in = _initialize_gru_hidden(sac_model.hidden_dim, dev)

                action, hidden_out = sac_model.policy_net.get_action(
                    state, last_action, hidden_in, deterministic=deterministic)
                if step < 50:
                    print('Action in evaluation: ', action)
                if isinstance(hidden_out, tuple):
                    hidden_out = tuple(h.detach() for h in hidden_out)
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


def plot(rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2_GRU.png')
    # plt.show()

