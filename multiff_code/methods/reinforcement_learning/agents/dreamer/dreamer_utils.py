# dreamer_v3_multiff.py

import os
import math
import pickle
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# -------------------------------------------------
# Utilities (use single-quote strings, etc)
# -------------------------------------------------
class NumericsConfig:
    def __init__(self, mode='warn', max_warns_per_episode=1, escalate_after=10):
        self.mode = mode
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

def linear_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        with torch.no_grad():
            m.weight.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.uniform_(-stdv, stdv)

def ortho_gru_(gru: nn.GRU):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name or 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

# -------------------------------------------------
# Replay buffer (episode-wise)
# -------------------------------------------------
class EpisodeReplay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0

    def push(self, obs_seq, act_seq, rew_seq, done_seq):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (np.asarray(obs_seq), np.asarray(act_seq),
                                 np.asarray(rew_seq), np.asarray(done_seq))
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, *, seq_len: Optional[int] = 50, burn_in: int = 0, random_window: bool = True):
        batch = random.sample(self.buffer, batch_size)
        T_list = [ep[0].shape[0] for ep in batch]
        if len(T_list) == 0:
            raise ValueError('Replay buffer empty')
        core_T = min(T_list) if seq_len is None else int(min(seq_len, min(T_list)))
        max_prefix = min([T - core_T for T in T_list]) if core_T > 0 else 0
        prefix = int(max(0, min(burn_in, max_prefix)))

        o_lst, a_lst, r_lst, d_lst = [], [], [], []
        for obs_seq, act_seq, rew_seq, done_seq in batch:
            T = obs_seq.shape[0]
            low = prefix
            high = max(prefix, T - core_T)
            t0 = random.randint(low, high) if (random_window and high > low) else low
            t0_b = t0 - prefix
            t1 = t0 + core_T
            o_lst.append(obs_seq[t0_b:t1])
            a_lst.append(act_seq[t0_b:t1])
            r_lst.append(rew_seq[t0_b:t1])
            d_lst.append(done_seq[t0_b:t1])

        return o_lst, a_lst, r_lst, d_lst, prefix

# -------------------------------------------------
# World Model (RSSM) for V3 style
# -------------------------------------------------
class RSSM(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, deter_dim=256, stoch_dim=32, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

        # Encoder for obs → embedded
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.obs_stats = nn.Linear(hidden_dim, 2 * stoch_dim)

        # Recurrent deterministic core
        self.gru = nn.GRU(input_size=stoch_dim + act_dim, hidden_size=deter_dim, batch_first=False)
        ortho_gru_(self.gru)

        # Prior network p(z | h)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        # Post network q(z | h, embed(o))
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # Decoder: reconstruct obs & reward
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.rew_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(linear_init)

    def init_state(self, batch_size: int):
        h = torch.zeros(1, batch_size, self.deter_dim, device=self._device())
        z = torch.zeros(batch_size, self.stoch_dim, device=self._device())
        return h, z

    def _device(self):
        return next(self.parameters()).device

    @staticmethod
    def _stats_to_dist_params(stats):
        mean, logstd = torch.chunk(stats, 2, dim=-1)
        logstd = torch.clamp(logstd, -7.0, 5.0)
        std = torch.exp(logstd)
        return mean, std

    def obs_encode(self, obs):
        emb = self.obs_encoder(obs)
        stats = self.obs_stats(emb)
        mean, std = self._stats_to_dist_params(stats)
        return mean, std

    def prior(self, h_t):
        stats = self.prior_net(h_t)
        mean, std = self._stats_to_dist_params(stats)
        return mean, std

    def posterior(self, h_t, obs_emb):
        stats = self.post_net(torch.cat([h_t, obs_emb], dim=-1))
        mean, std = self._stats_to_dist_params(stats)
        return mean, std

    def rollout_post(self, obs_seq, act_seq):
        # obs_seq: [B, T, obs_dim]
        # act_seq: [B, T, act_dim]
        B, T, _ = obs_seq.shape
        device = self._device()
        h, z = self.init_state(B)
        h = h.to(device)
        z = z.to(device)

        hs, zs = [], []
        prior_means, prior_stds, post_means, post_stds = [], [], [], []

        for t in range(T):
            a_t = act_seq[:, t, :]
            h_in = h.squeeze(0)  # [B, H]
            pm, ps = self.prior(h_in)
            om, os = self.obs_encode(obs_seq[:, t, :])
            qm, qs = self.posterior(h_in, om)
            z = qm + qs * torch.randn_like(qs)
            hs.append(h.squeeze(0))
            zs.append(z)
            prior_means.append(pm)
            prior_stds.append(ps)
            post_means.append(qm)
            post_stds.append(qs)

            # GRU step: input = [z, a_t]
            x = torch.cat([z, a_t], dim=-1).unsqueeze(0)
            h, _ = self.gru(x, h)  # new h: [1, B, H]

        h = torch.stack(hs, dim=1)               # [B, T, H]
        z = torch.stack(zs, dim=1)               # [B, T, Z]
        prior_mean = torch.stack(prior_means, dim=1)
        prior_std = torch.stack(prior_stds, dim=1)
        post_mean = torch.stack(post_means, dim=1)
        post_std = torch.stack(post_stds, dim=1)

        return {
            'h': h, 'z': z,
            'prior_mean': prior_mean, 'prior_std': prior_std,
            'post_mean': post_mean, 'post_std': post_std
        }

    def reconstruct(self, h, z):
        # h: [B,T,H], z: [B,T,Z]
        B, T, H = h.shape
        x = torch.cat([h.view(B*T, H), z.view(B*T, self.stoch_dim)], dim=-1)
        obs_rec = self.obs_decoder(x)
        rew_rec = self.rew_head(x)
        obs_rec = obs_rec.view(B, T, self.obs_dim)
        rew_rec = rew_rec.view(B, T, 1)
        return obs_rec, rew_rec

    def imagine(self, h0, z0, actor, horizon: int):
        # Imagine rollout starting from (h0, z0) using actor policy (in latent space)
        # h0: [1, B, H], z0: [B, Z]
        B = z0.shape[0]
        hs, zs, actions, rewards = [], [], [], []
        h = h0
        z = z0
        for t in range(horizon):
            mean, std = actor.forward(h.squeeze(0), z)
            a, logp, _, _, _ = actor.sample(h.squeeze(0), z)
            x = torch.cat([z, a], dim=-1).unsqueeze(0)
            h, _ = self.gru(x, h)
            pm, ps = self.prior(h.squeeze(0))
            z = pm + ps * torch.randn_like(ps)
            hs.append(h.squeeze(0))
            zs.append(z)
            actions.append(a)
            rewards.append(self.rew_head(torch.cat([h.squeeze(0), z], dim=-1)))
        hs = torch.stack(hs, dim=1)        # [B, horizon, H]
        zs = torch.stack(zs, dim=1)        # [B, horizon, Z]
        rewards = torch.stack(rewards, dim=1)  # [B, horizon, 1]
        return {'h': hs, 'z': zs, 'r': rewards, 'a': torch.stack(actions, dim=1)}

    @staticmethod
    def _kl_gauss_gauss(mean1, std1, mean2, std2):
        # KL N(mean1,std1^2) || N(mean2,std2^2)
        var1 = std1 * std1
        var2 = std2 * std2
        return 0.5 * ((var1 / var2) + ((mean2 - mean1).pow(2) / var2) - 1 + 2 * torch.log(std2/std1))

# -------------------------------------------------
# Actor & Critic
# -------------------------------------------------
class LatentActor(nn.Module):
    def __init__(self, deter_dim, stoch_dim, act_dim, hidden_dim=200, log_std_min=-5.0, log_std_max=2.0, action_range=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.std_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_range = float(action_range)
        self.apply(linear_init)

    def forward(self, h, z):
        x = self.net(torch.cat([h, z], dim=-1))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.std_head(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, h, z):
        mean, std = self.forward(h, z)
        eps = torch.randn_like(std)
        pre_tanh = mean + std * eps
        a = torch.tanh(pre_tanh) * self.action_range
        logp = (-0.5 * (((pre_tanh - mean)/(std+1e-8))**2 + 2*torch.log(std+1e-8) + math.log(2*math.pi))).sum(-1, keepdim=True)
        logp -= torch.log(1 - torch.tanh(pre_tanh).pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp, pre_tanh, mean, std

class LatentCritic(nn.Module):
    def __init__(self, deter_dim, stoch_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(linear_init)

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))

# -------------------------------------------------
# DreamerV3 Agent (adapted)
# -------------------------------------------------
class DreamerV3Agent:
    def __init__(self, *, obs_space, act_space,
                 action_range=1.0,
                 deter_dim=256, stoch_dim=32,
                 wm_hidden=256, actor_hidden=200, critic_hidden=200,
                 discount=0.997, lambda_gae=0.95,
                 kl_scale=1.0, kl_free_nats=1.0, kl_balance=0.8,
                 imag_horizon=15,
                 batch_size=8, seq_len=50, burn_in=5, random_window=True,
                 replay_capacity=1000,
                 world_lr=3e-4, actor_lr=3e-4, critic_lr=3e-4,
                 grad_clip=100.0):
        self.device = device
        self.discount = discount
        self.lambda_gae = lambda_gae
        self.kl_scale = kl_scale
        self.kl_free_nats = kl_free_nats
        self.kl_balance = kl_balance
        self.imag_horizon = imag_horizon
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.random_window = random_window
        self.grad_clip = grad_clip

        # dims
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))
        self.action_range = action_range

        # replay
        self.replay = EpisodeReplay(replay_capacity)

        # models
        self.wm = RSSM(self.obs_dim, self.act_dim, deter_dim, stoch_dim, wm_hidden).to(self.device)
        self.actor = LatentActor(deter_dim, stoch_dim, self.act_dim, actor_hidden, action_range=self.action_range).to(self.device)
        self.critic = LatentCritic(deter_dim, stoch_dim, critic_hidden).to(self.device)
        self.critic_target = LatentCritic(deter_dim, stoch_dim, critic_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.wm_opt = optim.Adam(self.wm.parameters(), lr=world_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.train_steps = 0

    def update(self):
        o_lst, a_lst, r_lst, d_lst, prefix = self.replay.sample(self.batch_size,
                                                                 seq_len=self.seq_len,
                                                                 burn_in=self.burn_in,
                                                                 random_window=self.random_window)
        obs = torch.tensor(np.array(o_lst), dtype=torch.float32, device=self.device)
        act = torch.tensor(np.array(a_lst), dtype=torch.float32, device=self.device)
        rew = torch.tensor(np.array(r_lst), dtype=torch.float32, device=self.device).unsqueeze(-1)
        done = torch.tensor(np.array(d_lst), dtype=torch.float32, device=self.device).unsqueeze(-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self.action_range, self.action_range)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        done = torch.nan_to_num(done, nan=1.0, posinf=1.0, neginf=1.0).clamp_(0.0, 1.0)

        B, T, _ = obs.shape
        tr = slice(int(prefix), T)

        # world model train
        self.wm_opt.zero_grad(set_to_none=True)
        post = self.wm.rollout_post(obs, act)
        h = post['h']
        z = post['z']
        obs_rec, rew_rec = self.wm.reconstruct(h, z)
        obs_loss = F.mse_loss(obs_rec[:, tr, :], obs[:, tr, :])
        rew_loss = F.mse_loss(rew_rec[:, tr, :], rew[:, tr, :])

        pm = post['prior_mean'][:, tr, :]
        ps = post['prior_std'][:, tr, :]
        qm = post['post_mean'][:, tr, :]
        qs = post['post_std'][:, tr, :]
        kl_qp = self.wm._kl_gauss_gauss(qm, qs, pm, ps)
        kl_pq = self.wm._kl_gauss_gauss(pm, ps, qm, qs)
        kl = (self.kl_balance * kl_qp + (1.0 - self.kl_balance) * kl_pq).mean()
        kl = torch.clamp(kl, min=self.kl_free_nats)

        wm_loss = obs_loss + rew_loss + self.kl_scale * kl
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
        self.wm_opt.step()

        # imagine in latent
        with torch.no_grad():
            h0 = h[:, int(T/2), :].unsqueeze(0)
            z0 = z[:, int(T/2), :]
        imag = self.wm.imagine(h0, z0, self.actor, horizon=self.imag_horizon)

        # critic target and returns
        with torch.no_grad():
            v_boot = self.critic_target(imag['h'][:, -1, :], imag['z'][:, -1, :])
            returns = self._lambda_return(rewards=imag['r'][:, :-1, :],
                                          values=self.critic(imag['h'][:, :-1, :], imag['z'][:, :-1, :]),
                                          boot=v_boot,
                                          discount=self.discount,
                                          lam=self.lambda_gae)

        # critic update
        self.critic_opt.zero_grad(set_to_none=True)
        v_pred = self.critic(imag['h'][:, :-1, :], imag['z'][:, :-1, :])
        critic_loss = F.mse_loss(v_pred, returns.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # actor update (you could include entropy or log-prob bonus)
        self.actor_opt.zero_grad(set_to_none=True)
        imag_pg = self.wm.imagine(h0.detach(), z0.detach(), self.actor, horizon=self.imag_horizon)
        v_pred_pg = self.critic(imag_pg['h'][:, :-1, :], imag_pg['z'][:, :-1, :])
        actor_loss = -(v_pred_pg).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # target update
        self._soft_update(self.critic_target, self.critic, tau=0.01)

        self.train_steps += 1

        return {
            'wm_loss': float(wm_loss.cpu()),
            'obs_loss': float(obs_loss.cpu()),
            'rew_loss': float(rew_loss.cpu()),
            'kl': float(kl.cpu()),
            'critic_loss': float(critic_loss.cpu()),
            'actor_loss': float(actor_loss.cpu()),
        }

    def act(self, obs, deterministic=False):
        self.actor.eval()
        self.wm.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
            # you might maintain last latent state h, z between steps; here simplified:
            # encode obs → embed, infer posterior, choose action
            obs_emb_mean, obs_emb_std = self.wm.obs_encode(obs_t)
            z = obs_emb_mean  # simplest choice
            # need h from previous step; for now assume zero
            h = torch.zeros(1, 1, self.wm.deter_dim, device=self.device)
            mean, std = self.actor.forward(h.squeeze(0), z)
            if deterministic:
                a = torch.tanh(mean) * self.action_range
            else:
                eps = torch.randn_like(std)
                pre_tanh = mean + std * eps
                a = torch.tanh(pre_tanh) * self.action_range
            return a.cpu().numpy()[0]

    def save_model(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.wm.state_dict(), os.path.join(folder, 'wm.pt'))
        torch.save(self.actor.state_dict(), os.path.join(folder, 'actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(folder, 'critic.pt'))
        torch.save(self.critic_target.state_dict(), os.path.join(folder, 'critic_target.pt'))
        # optionally save optimizer states & replay buffer
        with open(os.path.join(folder, 'replay.pkl'), 'wb') as f:
            pickle.dump(self.replay.buffer, f)

    def load_model(self, folder: str):
        self.wm.load_state_dict(torch.load(os.path.join(folder, 'wm.pt'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(folder, 'actor.pt'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(folder, 'critic.pt'), map_location=self.device))
        self.critic_target.load_state_dict(torch.load(os.path.join(folder, 'critic_target.pt'), map_location=self.device))
        with open(os.path.join(folder, 'replay.pkl'), 'rb') as f:
            self.replay.buffer = pickle.load(f)

    def _lambda_return(self, rewards, values, boot, discount, lam):
        # rewards: [B, H, 1], values: [B, H, 1], boot: [B,1]
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_val = boot
        for t in reversed(range(H)):
            next_val = rewards[:, t, :] + discount * ((1.0 - 0.0) * (lam * next_val + (1.0 - lam) * values[:, t, :]))
            returns[:, t, :] = next_val
        return returns

    def _soft_update(self, target_net, src_net, tau):
        for tparam, sparam in zip(target_net.parameters(), src_net.parameters()):
            tparam.data.copy_(tau * sparam.data + (1.0 - tau) * tparam.data)

# -------------------------------------------------
# Example train loop (you’d integrate with MultiFF env)
# -------------------------------------------------
def train_episode(env, agent: DreamerV3Agent, max_steps_per_episode: int):
    obs, _ = env.reset()
    obs = obs.flatten()
    episode_obs, episode_act, episode_rew, episode_done = [], [], [], []
    total_reward = 0.0

    for step in range(max_steps_per_episode):
        action = agent.act(obs, deterministic=False)
        next_obs, reward, done, _, _ = env.step(action)
        next_obs = next_obs.flatten()

        episode_obs.append(obs)
        episode_act.append(action)
        episode_rew.append(reward)
        episode_done.append(done)

        obs = next_obs
        total_reward += reward

        if done:
            break

    agent.replay.push(episode_obs, episode_act, episode_rew, episode_done)
    # train updates if enough data
    if len(agent.replay) >= agent.batch_size:
        metrics = agent.update()
        return total_reward, metrics
    else:
        return total_reward, None

def evaluate_agent(env, agent: DreamerV3Agent, num_episodes: int, max_steps: int, deterministic=True):
    cum_reward = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = obs.flatten()
        for step in range(max_steps):
            action = agent.act(obs, deterministic=deterministic)
            next_obs, reward, done, _, _ = env.step(action)
            obs = next_obs.flatten()
            cum_reward += reward
            if done:
                break
    return cum_reward / num_episodes
