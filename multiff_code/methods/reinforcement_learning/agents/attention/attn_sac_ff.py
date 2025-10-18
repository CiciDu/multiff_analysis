"""
attn_sac_ff.py
--------------
Feed-forward Attention-SAC trainer for MultiFF.

Key classes:
  - FFSlotAttentionEncoder: per-step set encoder with Top-K selection
  - PolicyWithAttention, QWithAttention: actor/critics
  - ReplayBuffer + save/load helpers
  - AttnSACforMultifirefly: high-level wrapper with an API similar to your LSTM agent

Features:
  - Flexible attention (temperature, optional Gumbel noise, dropout over slots)
  - Polyak target updates, entropy tuning, grad clipping
  - CSV logging of metrics, periodic evaluation & checkpointing
  - AMP (mixed precision) optional
"""

import os
import csv
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from reinforcement_learning.attention.env_attn_multiff import EnvForAttentionSAC, get_action_limits, batch_obs_to_attn_tensors


# ---------------------------
# Utilities
# ---------------------------
def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# Attention encoder
# ---------------------------


class FFSlotAttentionEncoder(nn.Module):
    """
    Simple per-step set encoder:
      - MLP to embed slots
      - score = <H, q_h>/sqrt(d), avg over heads
      - softmax weights (temperature), optional random dropout of valid slots
      - Top-K selection using raw scores (with optional Gumbel noise during training)
    """

    def __init__(self, d_in: int, d_slot: int = 64, n_heads: int = 2, k: int = 4,
                 use_soft_ctx: bool = True, temperature: float = 1.0, slot_dropout_p: float = 0.0,
                 gumbel_topk: bool = False):
        super().__init__()
        self.k = k
        self.use_soft_ctx = use_soft_ctx
        self.d_slot = d_slot
        self.n_heads = n_heads
        self.temperature = temperature
        self.slot_dropout_p = slot_dropout_p
        self.gumbel_topk = gumbel_topk

        self.slot_mlp = nn.Sequential(
            nn.Linear(d_in, d_slot), nn.ReLU(),
            nn.Linear(d_slot, d_slot)
        )
        self.q = nn.Parameter(torch.randn(n_heads, d_slot) * 0.1)

    def forward(self, slot_feats: torch.Tensor, slot_mask: torch.Tensor):
        B, S, Din = slot_feats.shape
        H = self.slot_mlp(slot_feats)                  # [B,S,d]

        scale = 1.0 / math.sqrt(self.d_slot)
        scores_heads = torch.einsum(
            'bsd,hd->bsh', H, self.q) * scale  # [B,S,H]
        scores = scores_heads.mean(dim=-1)                             # [B,S]

        # Apply slot dropout on the mask during training (robustness to missing slots)
        mask = (slot_mask > 0.5)
        if self.training and self.slot_dropout_p > 0.0:
            drop = torch.rand_like(mask.float()) < self.slot_dropout_p
            mask = mask & (~drop)

        # Soft weights with temperature; avoid all -inf rows causing NaNs
        scores_masked = scores.masked_fill(~mask, float('-inf'))
        all_masked = ~mask.any(dim=1)  # [B]
        if all_masked.any():
            # when no valid slots, fall back to uniform over all slots
            fallback = torch.zeros_like(scores)
            attnW = torch.softmax(
                torch.where(all_masked.unsqueeze(1), fallback, scores_masked) / max(self.temperature, 1e-6), dim=1
            )
        else:
            attnW = torch.softmax(
                scores_masked / max(self.temperature, 1e-6), dim=1)  # [B,S]

        ctx = torch.einsum('bs,bsd->bd', attnW,
                           H) if self.use_soft_ctx else None  # [B,d]

        # Top-K indices
        if self.training and self.gumbel_topk:
            # add Gumbel noise to promote diversity during training
            # Gumbel(0,1)
            g = -torch.log(-torch.log(torch.rand_like(scores_masked)))
            scores_for_topk = scores_masked + g
        else:
            scores_for_topk = scores_masked

        k_eff = min(self.k, S)
        # If all scores are -inf (due to masking), replace with zeros to allow topk
        invalid_rows = torch.isinf(scores_for_topk).all(dim=1)
        safe_scores = torch.where(invalid_rows.unsqueeze(
            1), torch.zeros_like(scores_for_topk), scores_for_topk)
        topk_idx = torch.topk(safe_scores, k=k_eff, dim=1).indices      # [B,K]
        sel = torch.gather(
            # [B,K,d]
            H, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_slot))

        return sel, ctx, attnW

# ---------------------------
# Actor & Critics
# ---------------------------


class PolicyWithAttention(nn.Module):
    def __init__(self, d_in_slot=5, d_slot=64, n_heads=2, k=4, include_ctx=True, d_self=2, n_actions=2,
                 temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False):
        super().__init__()
        self.attn = FFSlotAttentionEncoder(
            d_in=d_in_slot, d_slot=d_slot, n_heads=n_heads, k=k,
            use_soft_ctx=include_ctx, temperature=temperature,
            slot_dropout_p=slot_dropout_p, gumbel_topk=gumbel_topk
        )
        fused_dim = k * d_slot + (d_slot if include_ctx else 0) + d_self
        self.net = nn.Sequential(nn.Linear(fused_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, n_actions)
        self.logstd = nn.Linear(256, n_actions)

    def forward(self, slot_feats, slot_mask, self_feats):
        sel, ctx, _ = self.attn(slot_feats, slot_mask)
        B = slot_feats.size(0)
        sel_flat = sel.reshape(B, -1)
        parts = [sel_flat, self_feats]
        if ctx is not None:
            parts.insert(1, ctx)
        x = torch.cat(parts, dim=-1)
        z = self.net(x)
        mu = self.mu(z)
        logstd = self.logstd(z).clamp(-5, 2)     # Stabilize std
        std = logstd.exp()
        return mu, std, sel, ctx


class QWithAttention(nn.Module):
    def __init__(self, d_in_slot=5, d_slot=64, n_heads=2, k=4, include_ctx=True, d_self=2, n_actions=2,
                 temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False):
        super().__init__()
        self.attn = FFSlotAttentionEncoder(
            d_in=d_in_slot, d_slot=d_slot, n_heads=n_heads, k=k,
            use_soft_ctx=include_ctx, temperature=temperature,
            slot_dropout_p=slot_dropout_p, gumbel_topk=gumbel_topk
        )
        fused_dim = k * d_slot + \
            (d_slot if include_ctx else 0) + d_self + n_actions
        self.q = nn.Sequential(nn.Linear(fused_dim, 256), nn.ReLU(),
                               nn.Linear(256, 256), nn.ReLU(),
                               nn.Linear(256, 1))

    def forward(self, slot_feats, slot_mask, self_feats, action):
        sel, ctx, _ = self.attn(slot_feats, slot_mask)
        B = slot_feats.size(0)
        sel_flat = sel.reshape(B, -1)
        parts = [sel_flat, self_feats, action]
        if ctx is not None:
            parts.insert(1, ctx)
        x = torch.cat(parts, dim=-1)
        return self.q(x).squeeze(-1)

# ---------------------------
# SAC ops
# ---------------------------


def sample_action(mu: torch.Tensor, std: torch.Tensor, action_limits: List[Tuple[float, float]]):
    normal = torch.distributions.Normal(mu, std)
    z = normal.rsample()
    a = torch.tanh(z)
    logp = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
    logp = logp.sum(dim=-1, keepdim=True)
    # scale
    a_scaled_list = []
    for j in range(a.size(-1)):
        lo, hi = action_limits[j]
        mid, half = 0.5*(hi+lo), 0.5*(hi-lo)
        a_scaled_list.append(mid + half * a[:, j:j+1])
    return torch.cat(a_scaled_list, dim=-1), logp


def sac_train_step_ff(
    *,
    batch: Dict[str, torch.Tensor],
    actor, q1, q2,
    actor_targ, q1_targ, q2_targ,
    actor_opt, q1_opt, q2_opt,
    log_alpha: torch.nn.Parameter, alpha_opt,
    gamma: float, tau: float, target_entropy: float,
    action_limits: List[Tuple[float, float]],
    grad_clip: float = 1.0,
    amp: bool = False,
    device=torch.device('cpu'),
) -> Dict[str, float]:
    s_feats = batch['slot_feats'].to(device)
    s_mask = batch['slot_mask'].to(device)
    self_feats = batch['self_feats'].to(device)
    a = batch['action'].to(device)
    r = batch['reward'].to(device)
    d = batch['done'].to(device)
    s_feats_n = batch['slot_feats_next'].to(device)
    s_mask_n = batch['slot_mask_next'].to(device)
    self_feats_n = batch['self_feats_next'].to(device)

    alpha = log_alpha.exp().detach()

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ----- Targets -----
    with torch.no_grad():
        mu_n, std_n, _, _ = actor(s_feats_n, s_mask_n, self_feats_n)
        a_n, logp_n = sample_action(mu_n, std_n, action_limits)
        q1n = q1_targ(s_feats_n, s_mask_n, self_feats_n, a_n)
        q2n = q2_targ(s_feats_n, s_mask_n, self_feats_n, a_n)
        q_next = torch.min(q1n, q2n) - alpha * logp_n.squeeze(-1)
        y = r + gamma * (1.0 - d) * q_next

    # ----- Critic losses -----
    with torch.cuda.amp.autocast(enabled=amp):
        q1_pred = q1(s_feats, s_mask, self_feats, a)
        q2_pred = q2(s_feats, s_mask, self_feats, a)
        loss_q1 = F.mse_loss(q1_pred, y)
        loss_q2 = F.mse_loss(q2_pred, y)

    q1_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_q1).backward()
    if grad_clip:
        scaler.unscale_(q1_opt)
        nn.utils.clip_grad_norm_(q1.parameters(), grad_clip)
    scaler.step(q1_opt)

    q2_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_q2).backward()
    if grad_clip:
        scaler.unscale_(q2_opt)
        nn.utils.clip_grad_norm_(q2.parameters(), grad_clip)
    scaler.step(q2_opt)

    # ----- Actor -----
    with torch.cuda.amp.autocast(enabled=amp):
        mu, std, _, _ = actor(s_feats, s_mask, self_feats)
        a_pi, logp = sample_action(mu, std, action_limits)
        q1_pi = q1(s_feats, s_mask, self_feats, a_pi)
        q2_pi = q2(s_feats, s_mask, self_feats, a_pi)
        q_min = torch.min(q1_pi, q2_pi)
        loss_actor = (log_alpha.exp() * logp.squeeze(-1) - q_min).mean()

    actor_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_actor).backward()
    if grad_clip:
        scaler.unscale_(actor_opt)
        nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    scaler.step(actor_opt)

    # ----- Alpha -----
    with torch.no_grad():
        curr_alpha = log_alpha.exp()
    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()
    alpha_new = log_alpha.exp()

    # ----- Polyak -----
    @torch.no_grad()
    def polyak(src, tgt, tau_):
        for p, p_t in zip(src.parameters(), tgt.parameters()):
            p_t.data.mul_(tau_).add_(p.data, alpha=1.0 - tau_)
    with torch.no_grad():
        polyak(q1, q1_targ, tau)
        polyak(q2, q2_targ, tau)
        polyak(actor, actor_targ, tau)

    return {
        'loss_q1': float(loss_q1.detach().cpu().item()),
        'loss_q2': float(loss_q2.detach().cpu().item()),
        'loss_actor': float(loss_actor.detach().cpu().item()),
        'alpha_loss': float(alpha_loss.detach().cpu().item()),
        'alpha': float(alpha_new.detach().cpu().item()),
        'q_target_mean': float(y.detach().cpu().mean().item()),
        'q_pred_mean': float(0.5*(q1_pred.detach().cpu().mean().item()+q2_pred.detach().cpu().mean().item())),
    }

# ---------------------------
# Replay + Save/Load
# ---------------------------


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=dtype)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=dtype)
        self.action = np.zeros((capacity, action_dim), dtype=dtype)
        self.reward = np.zeros((capacity,), dtype=dtype)
        self.done = np.zeros((capacity,), dtype=dtype)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, done, next_obs):
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.next_obs[idx] = next_obs
        self.ptr += 1
        our_size = self.size + 1
        self.size = our_size if our_size < self.capacity else self.capacity

    def sample_np(self, batch_size: int):
        assert self.size >= batch_size, "Not enough samples in replay."
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': self.obs[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'done': self.done[idx],
            'next_obs': self.next_obs[idx],
        }


def save_bundle(path: str, payload: Dict[str, Any]):
    os.makedirs(path, exist_ok=True)
    torch.save(payload, os.path.join(path, 'model.pt'))


def load_bundle(path: str) -> Dict[str, Any]:
    return torch.load(os.path.join(path, 'model.pt'), map_location='cpu')

# ---------------------------
# High-level wrapper
# ---------------------------


class AttnSACforMultifirefly:
    """
    End-to-end trainer for feed-forward Attention-SAC.
    API mirrors your LSTM wrapper's vibe (make_env -> make_agent -> regular_training).
    """

    def __init__(self,
                 model_folder: str = 'RL_models/ATTN_FF/agent_0/',
                 device: Optional[str] = None,
                 seed: int = 42,
                 **env_kwargs):
        seed_all(seed)
        self.model_folder = model_folder
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.input_env_kwargs = env_kwargs.copy()
        self.env = None

        # nets/opts
        self.actor = self.q1 = self.q2 = None
        self.actor_targ = self.q1_targ = self.q2_targ = None
        self.actor_opt = self.q1_opt = self.q2_opt = None
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=self.device))
        self.alpha_opt = None

        # replay
        self.replay = None

        # config
        self.cfg = dict(
            # encoder
            k_top=4, d_slot=64, n_heads=2, include_ctx=True,
            attn_temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False,
            # sac
            lr=3e-4, gamma=0.99, tau=0.995, target_entropy=-2.0,
            batch_size=128, replay_size=200_000, random_steps=2000, updates_per_step=1,
            grad_clip=1.0, amp=False,
            # loop
            log_every=1000, eval_every=0, eval_episodes=3, save_every=0
        )

        self.env_class = EnvForAttentionSAC

    # # ---- Env ----
    # def make_env(self, **kwargs):
    #     self.current_env_kwargs = copy.deepcopy(self.input_env_kwargs)
    #     self.current_env_kwargs.update(kwargs)
    #     self.env = EnvForAttentionSAC(**self.current_env_kwargs)
    #     return self.env

    # ---- Agent ----
    def make_agent(self, **overrides):
        assert self.env is not None, "Call make_env() first."
        self.cfg.update(overrides)

        d_in_slot = self.env.num_elem_per_ff
        n_actions = int(np.prod(self.env.action_space.shape))
        d_self = 2
        K = self.cfg['k_top']
        D = self.cfg['d_slot']
        H = self.cfg['n_heads']

        common_kwargs = dict(d_in_slot=d_in_slot, d_slot=D, n_heads=H, k=K,
                             include_ctx=self.cfg['include_ctx'], d_self=d_self, n_actions=n_actions,
                             temperature=self.cfg['attn_temperature'],
                             slot_dropout_p=self.cfg['slot_dropout_p'],
                             gumbel_topk=self.cfg['gumbel_topk'])

        self.actor = PolicyWithAttention(**common_kwargs).to(self.device)
        self.q1 = QWithAttention(**common_kwargs).to(self.device)
        self.q2 = QWithAttention(**common_kwargs).to(self.device)

        self.actor_targ = PolicyWithAttention(**common_kwargs).to(self.device)
        self.q1_targ = QWithAttention(**common_kwargs).to(self.device)
        self.q2_targ = QWithAttention(**common_kwargs).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        lr = self.cfg['lr']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        obs_dim = self.env.observation_space.shape[0] if hasattr(
            self.env, 'observation_space') else self.env.obs_space_length
        self.replay = ReplayBuffer(
            capacity=self.cfg['replay_size'], obs_dim=obs_dim, action_dim=n_actions)

    # ---- Train ----
    def regular_training(self, total_env_steps: int = 50_000):
        assert self.env is not None and self.actor is not None
        device = self.device
        cfg = self.cfg
        action_limits = get_action_limits(self.env)

        # metrics CSV
        os.makedirs(self.model_folder, exist_ok=True)
        csv_path = os.path.join(self.model_folder, "metrics.csv")
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.DictWriter(csv_f, fieldnames=[
            "t", "ep", "ep_len", "ep_return", "buffer", "alpha", "loss_q1", "loss_q2", "loss_actor", "q_mean"
        ])
        csv_w.writeheader()

        obs, info = self.env.reset()
        ep_ret, ep_len, ep = 0.0, 0, 0
        last_eval = 0
        metrics = {}

        for t in range(1, total_env_steps+1):
            # --- act ---
            if t <= cfg['random_steps']:
                action = self.env.action_space.sample().astype(np.float32)
            else:
                sf, sm, ss = self.env.obs_to_attn_tensors(obs, device=device)
                with torch.no_grad():
                    mu, std, _, _ = self.actor(sf, sm, ss)
                    a_t, _ = sample_action(mu, std, action_limits)
                action = a_t.squeeze(0).cpu().numpy().astype(np.float32)

            # --- step ---
            next_obs, reward, done, truncated, _ = self.env.step(action)

            # --- store ---
            self.replay.add(obs=obs, action=action, reward=np.float32(
                reward), done=np.float32(done), next_obs=next_obs)
            ep_ret += reward
            ep_len += 1

            # --- update ---
            if t > cfg['random_steps'] and self.replay.size >= cfg['batch_size']:
                for _ in range(cfg['updates_per_step']):
                    batch_np = self.replay.sample_np(cfg['batch_size'])
                    sf, sm, ss = batch_obs_to_attn_tensors(
                        batch_np['obs'], self.env, device=device)
                    sfn, smn, ssn = batch_obs_to_attn_tensors(
                        batch_np['next_obs'], self.env, device=device)
                    batch_t = {
                        'slot_feats': sf, 'slot_mask': sm, 'self_feats': ss,
                        'action': torch.tensor(batch_np['action'], dtype=torch.float32, device=device),
                        'reward': torch.tensor(batch_np['reward'], dtype=torch.float32, device=device),
                        'done': torch.tensor(batch_np['done'], dtype=torch.float32, device=device),
                        'slot_feats_next': sfn, 'slot_mask_next': smn, 'self_feats_next': ssn,
                    }
                    metrics = sac_train_step_ff(
                        batch=batch_t, actor=self.actor, q1=self.q1, q2=self.q2,
                        actor_targ=self.actor_targ, q1_targ=self.q1_targ, q2_targ=self.q2_targ,
                        actor_opt=self.actor_opt, q1_opt=self.q1_opt, q2_opt=self.q2_opt,
                        log_alpha=self.log_alpha, alpha_opt=self.alpha_opt,
                        gamma=cfg['gamma'], tau=cfg['tau'],
                        target_entropy=float(cfg['target_entropy']),
                        action_limits=action_limits,
                        grad_clip=cfg['grad_clip'], amp=cfg['amp'], device=device
                    )

            # --- episode end ---
            if done or truncated:
                ep += 1
                print(
                    f"Ep {ep:4d} | steps={ep_len:4d} | return={ep_ret:8.2f} | buf={self.replay.size:6d}")
                csv_w.writerow(dict(t=t, ep=ep, ep_len=ep_len, ep_return=ep_ret, buffer=self.replay.size,
                                    alpha=float(
                                        self.log_alpha.exp().detach().cpu().item()),
                                    loss_q1=metrics.get(
                                        'loss_q1', float('nan')),
                                    loss_q2=metrics.get(
                                        'loss_q2', float('nan')),
                                    loss_actor=metrics.get(
                                        'loss_actor', float('nan')),
                                    q_mean=metrics.get('q_pred_mean', float('nan'))))
                obs, info = self.env.reset()
                ep_ret, ep_len = 0.0, 0
            else:
                obs = next_obs

            # --- logging ---
            if t % cfg['log_every'] == 0 and metrics:
                print(f"[t={t}] alpha={float(self.log_alpha.exp()):.3f}  "
                      f"Q={metrics['q_pred_mean']:.2f}  "
                      f"losses: Q1={metrics['loss_q1']:.3f} Q2={metrics['loss_q2']:.3f} Pi={metrics['loss_actor']:.3f}")

            # --- eval / save ---
            if cfg['eval_every'] and (t - last_eval) >= cfg['eval_every']:
                last_eval = t
                avg = self.test_agent(
                    num_eps=self.cfg['eval_episodes'], deterministic=True)
                print(f"[eval @ t={t}] avg_return={avg:.2f}")
                if cfg['save_every']:
                    self.save_agent(dir_name=os.path.join(
                        self.model_folder, f"ckpt_{t:07d}"))

        csv_f.close()
        return metrics

    # ---- Save/Load ----
    def save_agent(self, dir_name: Optional[str] = None):
        path = self.model_folder if dir_name is None else dir_name
        os.makedirs(path, exist_ok=True)
        payload = dict(
            actor=self.actor.state_dict(),
            q1=self.q1.state_dict(),
            q2=self.q2.state_dict(),
            actor_targ=self.actor_targ.state_dict(),
            q1_targ=self.q1_targ.state_dict(),
            q2_targ=self.q2_targ.state_dict(),
            log_alpha=self.log_alpha.detach().cpu(),
            opt_actor=self.actor_opt.state_dict(),
            opt_q1=self.q1_opt.state_dict(),
            opt_q2=self.q2_opt.state_dict(),
            opt_alpha=self.alpha_opt.state_dict(),
        )
        save_bundle(path, payload)
        # also save replay
        rp_path = os.path.join(path, "buffer.npz")
        if self.replay is not None:
            np.savez_compressed(rp_path, obs=self.replay.obs[:self.replay.size],
                                next_obs=self.replay.next_obs[:self.replay.size],
                                action=self.replay.action[:self.replay.size],
                                reward=self.replay.reward[:self.replay.size],
                                done=self.replay.done[:self.replay.size])
        print(f"Saved to {path}")

        self.write_checkpoint_manifest(dir_name)

    def load_agent(self, dir_name: Optional[str] = None):
        path = self.model_folder if dir_name is None else dir_name
        bundle = load_bundle(path)
        self.actor.load_state_dict(bundle['actor'])
        self.q1.load_state_dict(bundle['q1'])
        self.q2.load_state_dict(bundle['q2'])
        self.actor_targ.load_state_dict(bundle['actor_targ'])
        self.q1_targ.load_state_dict(bundle['q1_targ'])
        self.q2_targ.load_state_dict(bundle['q2_targ'])
        with torch.no_grad():
            self.log_alpha.copy_(bundle['log_alpha'].to(self.device))
        self.actor_opt.load_state_dict(bundle['opt_actor'])
        self.q1_opt.load_state_dict(bundle['opt_q1'])
        self.q2_opt.load_state_dict(bundle['opt_q2'])
        self.alpha_opt.load_state_dict(bundle['opt_alpha'])

        # try to restore replay (optional)
        rp_path = os.path.join(path, "buffer.npz")
        if os.path.exists(rp_path):
            data = np.load(rp_path, allow_pickle=True)
            obs_dim = data['obs'].shape[1]
            act_dim = data['action'].shape[1]
            self.replay = ReplayBuffer(
                int(1.25 * data['obs'].shape[0]), obs_dim, act_dim)
            for i in range(data['obs'].shape[0]):
                self.replay.add(data['obs'][i], data['action'][i],
                                data['reward'][i], data['done'][i], data['next_obs'][i])
            print(f"Loaded replay of size {self.replay.size}")
        print(f"Loaded from {path}")

    # ---- Eval ----
    def test_agent(self, num_eps: int = 3, max_steps_per_eps: Optional[int] = None, deterministic: bool = True):
        assert self.env is not None and self.actor is not None
        max_steps = max_steps_per_eps or self.env.episode_len
        action_limits = get_action_limits(self.env)
        returns = []
        for e in range(num_eps):
            obs, info = self.env.reset()
            ret = 0.0
            for t in range(max_steps):
                sf, sm, ss = self.env.obs_to_attn_tensors(
                    obs, device=self.device)
                with torch.no_grad():
                    mu, std, _, _ = self.actor(sf, sm, ss)
                    if deterministic:
                        a = torch.tanh(mu)
                        # scale to limits
                        a_scaled_list = []
                        for j in range(a.size(-1)):
                            lo, hi = action_limits[j]
                            mid, half = 0.5*(hi+lo), 0.5*(hi-lo)
                            a_scaled_list.append(mid + half * a[0, j:j+1])
                        act = torch.cat(a_scaled_list, dim=-1)
                    else:
                        act, _ = sample_action(mu, std, action_limits)
                action = act.squeeze(0).cpu().numpy().astype(np.float32)
                obs, r, done, trunc, _ = self.env.step(action)
                ret += r
                if done or trunc:
                    break
            returns.append(ret)
            print(f"Eval ep {e+1}/{num_eps}: return={ret:.2f}")
        avg = float(np.mean(returns))
        print(f"Average return over {num_eps} eps: {avg:.2f}")
        return avg
