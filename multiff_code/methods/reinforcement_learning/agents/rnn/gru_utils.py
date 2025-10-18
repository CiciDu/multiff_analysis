from reinforcement_learning.agents.rnn import lstm_utils
import torch
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal




device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))





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

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst=[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()

        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)






class QNetworkGRU(lstm_utils.QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(lstm_utils.linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.cat([state, action], -1) 
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        gru_branch = torch.cat([state, last_action], -1) 
        gru_branch = self.activation(self.linear2(gru_branch))  # linear layer for 3d input only applied on the last dim
        gru_branch, gru_hidden = self.gru1(gru_branch, hidden_in) 
        # merged
        merged_branch=torch.cat([fc_branch, gru_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, gru_hidden    # gru_hidden is actually tuple: (hidden, cell)  




class SAC_PolicyNetworkGRU(lstm_utils.PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

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
    
    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(device)
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy() if deterministic else \
        action.detach().cpu().numpy()
        return action[0][0], hidden_out




class GRU_SAC_Trainer():
    def __init__(self, **kwargs):


        self.gamma = kwargs.get('gamma', 0.995)
        self.batch_size = kwargs.get('batch_size', 10)
        self.update_itr = kwargs.get('update_itr', 1)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.reward_scale = kwargs.get('reward_scale', 0.5)
        self.target_entropy = kwargs.get('target_entropy', -2)
        self.soft_tau = kwargs.get('soft_tau', 0.0015)
        self.train_freq = kwargs.get('train_freq', 5)

        state_space = kwargs.get('state_space')
        action_space = kwargs.get('action_space')
        action_range = kwargs.get('action_range')
        soft_q_lr = kwargs.get('soft_q_lr')
        policy_lr = kwargs.get('policy_lr')
        alpha_lr = kwargs.get('alpha_lr')

        self.replay_buffer = kwargs.get('replay_buffer')
        self.soft_q_net1 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(device)
        self.soft_q_net2 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkGRU(state_space, action_space, self.hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkGRU(state_space, action_space, self.hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
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

    
    def update(self, batch_size, reward_scale=0.001, auto_entropy=True, target_entropy=-2, gamma=0.975,soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        batch_size = self.batch_size
        reward_scale = self.reward_scale
        gamma = self.gamma
        target_entropy = self.target_entropy
        soft_tau = self.soft_tau


        state      = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action     = torch.FloatTensor(np.array(action)).to(device)
        last_action     = torch.FloatTensor(np.array(last_action)).to(device)
        reward     = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)


        # state      = torch.FloatTensor(state).to(device)
        # next_state = torch.FloatTensor(next_state).to(device)
        # action     = torch.FloatTensor(action).to(device)
        # last_action     = torch.FloatTensor(last_action).to(device)
        # reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        # done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        #reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * reward
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predict_q1, _= self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
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
        self.soft_q_net1.load_state_dict(torch.load(path+'/gru_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'/gru_q2'))
        self.policy_net.load_state_dict(torch.load(path+'/gru_policy'))

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
    hidden_out = _initialize_gru_hidden(sac_model.hidden_dim, device)
    ini_hidden_in, ini_hidden_out = None, None

    for step in range(max_steps_per_eps):
        hidden_in = hidden_out
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
            hidden_out = _initialize_gru_hidden(sac_model.hidden_dim, device)
            
            epi_ctx = {}
            numerics_cfg = lstm_utils.NumericsConfig(
                mode='warn', max_warns_per_episode=1, escalate_after=10)
            for _ in range(max_steps_per_eps):
                hidden_in = hidden_out
                if not np.isfinite(state).all():
                    lstm_utils._maybe_warn_nans(True, 'eval.state', epi_ctx, numerics_cfg)
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                if (not torch.isfinite(hidden_in).all()):
                    lstm_utils._maybe_warn_nans(True, 'eval.hidden_in', epi_ctx, numerics_cfg)
                    hidden_in = np.nan_to_num(hidden_in, nan=0.0, posinf=0.0, neginf=0.0)

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


def plot(rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2_GRU.png')
    # plt.show()

