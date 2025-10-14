'''
attn_sac_rnn.py
---------------
Recurrent (GRU/LSTM) Attention-SAC trainer for MultiFF.

Key classes:
  - RecurrentAttentionEncoder + SACRNNActor/SACRNNQ
  - SequenceReplay (episode buffer with random windows + burn-in)
  - AttnRNNSACforMultifirefly high-level wrapper

Features:
  - Same flexible attention (temperature, dropout, Gumbel top-k)
  - Burn-in warmup, Polyak targets, entropy tuning, AMP option
  - CSV logging, evaluation hooks, checkpointing

NOTES FOR READERS (why some choices were made):
  • We treat each time-step's set independently in the set-encoder (no temporal mixing there) and let the RNN handle time. 
  • We flatten time into the batch to make the set-encoder run vectorized on all frames (big speedup; same math). 
  • Masks are boolean (True=valid, False=padding). Invalid elements are set to -inf before softmax to zero their weight. 
  • Gumbel top-k adds exploration noise during training when choosing the K slots to pass on. 
  • For SAC, actions are sampled with tanh squashing; log-prob correction uses log(1 - tanh(z)^2). 
  • Burn-in builds RNN hidden states on a prefix without backprop to stabilize sequence training.
'''

# ===========================
# Standard library imports
# ===========================
from collections import deque
from typing import Optional, Dict, List
import os
import csv
import math
from typing import Dict, Any, List, Tuple, Optional
from contextlib import nullcontext
import copy

# ===========================
# Third-party imports
# ===========================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: these are project-local imports; expected to be available in your repo
from machine_learning.RL.ff_attention.env_attn_multiff import (
    # Gym-like environment wrapper exposing MultiFF observations
    EnvForAttentionSAC,
    # helper to read per-dimension action ranges from the env
    get_action_limits,
    # numpy -> (slot_feats, slot_mask, self_feats)
    seq_obs_to_attn_tensors,
    # torch -> (slot_feats, slot_mask, self_feats)
    seq_obs_to_attn_tensors_torch
)
from machine_learning.RL.ff_attention import set_transformers


# ---------------------------
# Utilities
# ---------------------------

def seed_all(seed: int = 42):
    '''
    Set global PRNG seeds for reproducibility across numpy and torch.
    If CUDA is available, we also seed all CUDA devices. This does not
    make everything deterministic (CuDNN kernels may still be nondeterministic),
    but it reduces variance across runs.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# Attention set encoder (same as FF, re-used here)
# ---------------------------

class FFSlotAttentionEncoder(nn.Module):
    '''
    Lightweight slot-attention style encoder for a variable-size set of elements (slots).

    Inputs per forward call:
      - slot_feats: [B, S, d_in]  per-element features (e.g., per-firefly features)
      - slot_mask:  [B, S]        boolean mask: True for valid element, False for padding (0/False treated as invalid)

    Outputs:
      - sel:   [B, K, d_slot]   top-K selected element embeddings (by learned attention score)
      - ctx:   [B, d_slot] or None  optional soft attention-pooled context vector over ALL slots
      - attnW: [B, S]          normalized attention weights over S elements (before top-k selection)

    Config:
      - d_slot:       dimensionality of element embeddings after MLP
      - n_heads:      number of learned query heads used to score elements (scores averaged across heads)
      - k:            number of top elements to select deterministically (or stochastically if gumbel_topk)
      - use_soft_ctx: if True, also return a softmax-pooled context over all elements
      - temperature:  softmax temperature; <1.0 sharpens, >1.0 flattens
      - slot_dropout_p: element dropout on the mask during training (regularizes selection)
      - gumbel_topk:  if True (train only), add Gumbel noise to scores before top-k to encourage exploration

    Naming:
      - H  := per-slot embeddings after MLP, shape [B, S, d_slot]
      - q  := learned query vectors per head, shape [H, d_slot]
      - sf := "slot features" input (external naming used throughout the file)
      - sm := "slot mask" input (external naming used throughout the file)
      - sel:= selected top-K slot embeddings (what we pass onward to the RNN)
    '''

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

        # Small MLP to embed raw slot features into a common space.
        # This MLP is shared across all S elements in the set (permutation equivariant).
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_in, d_slot), nn.ReLU(),
            nn.Linear(d_slot, d_slot)
        )
        # Learn H=n_heads query vectors (one per head); shape [H, d_slot]
        # Initialized with small Gaussian noise.
        self.q = nn.Parameter(torch.randn(n_heads, d_slot) * 0.1)

    def forward(self, slot_feats: torch.Tensor, slot_mask: torch.Tensor):
        # Shapes
        # slot_feats: [B, S, d_in]  | slot_mask: [B, S] (True for valid)
        B, S, d_in = slot_feats.shape

        # 1) Embed each element with a shared MLP -> H: [B, S, d_slot]
        H = self.slot_mlp(slot_feats)

        # 2) Score elements using head-wise dot-products with learned queries.
        #    einsum('bsd,hd->bsh') computes, for each head h, score = <H[b,s,:], q[h,:]>
        #    scale by 1/sqrt(d) like attention for numerical stability.
        scale = 1.0 / math.sqrt(self.d_slot)
        scores_heads = torch.einsum(
            'bsd,hd->bsh', H, self.q) * scale  # [B, S, H]

        # Average over heads to get a single score per element (simple, effective).
        # If you want head-specific routing, keep per-head scores and top-k per head.
        scores = scores_heads.mean(dim=-1)  # [B, S]

        # 3) Apply element dropout on the mask during training to regularize which slots are used.
        #    We drop elements by turning their mask False, so they get -inf and 0 weight under softmax.
        mask = (slot_mask > 0.5)  # ensure boolean (accepts 0/1 floats)
        if self.training and self.slot_dropout_p > 0.0:
            # True -> drop this element
            drop = torch.rand_like(mask.float()) < self.slot_dropout_p
            mask = mask & (~drop)

        # Replace invalid scores with -inf so softmax assigns zero probability.
        scores_masked = scores.masked_fill(~mask, float('-inf'))

        # If a row is all-masked (e.g., no visible fireflies), avoid NaNs:
        # fallback to zeros as logits (softmax -> uniform over S, but those S got -inf except zeros).
        all_masked = ~mask.any(dim=1)  # [B]
        if all_masked.any():
            fallback = torch.zeros_like(scores)
            logits = torch.where(all_masked.unsqueeze(1),
                                 fallback, scores_masked)
        else:
            logits = scores_masked

        # Soft attention over all elements (temperature controls sharpness).
        # attnW sums to 1 across S for each batch item.
        attnW = torch.softmax(
            logits / max(self.temperature, 1e-6), dim=1)  # [B, S]

        # Optional soft context vector (weighted average over elements). If you
        # only want hard top-k and no soft summary, set use_soft_ctx=False.
        ctx = torch.einsum('bs,bsd->bd', attnW,
                           H) if self.use_soft_ctx else None

        # 4) (Optional) Add Gumbel noise before top-k when training to make the
        #     top-k selection stochastic. This encourages exploration of which
        #     slots to route forward.
        if self.training and self.gumbel_topk:
            # Gumbel(0,1) via -log(-log(U)) trick with U~Uniform(0,1)
            g = -torch.log(-torch.log(torch.rand_like(scores_masked)))
            scores_for_topk = scores_masked + g
        else:
            scores_for_topk = scores_masked

        # Safe top-k: if an entire row is -inf (all masked), replace by zeros before topk.
        k_eff = min(self.k, S)
        invalid_rows = torch.isinf(scores_for_topk).all(dim=1)
        safe_scores = torch.where(invalid_rows.unsqueeze(
            1), torch.zeros_like(scores_for_topk), scores_for_topk)

        # Select indices of the top-K elements per batch item.
        # ties are resolved by torch.topk's internal stable sort.
        topk_idx = torch.topk(safe_scores, k=k_eff, dim=1).indices  # [B, K]

        # Gather the corresponding embeddings -> [B, K, d_slot]
        sel = torch.gather(
            H, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_slot))

        return sel, ctx, attnW


# ---------------------------
# Sequence models (set encoder + RNN over time)
# ---------------------------

class RecurrentAttentionEncoder(nn.Module):
    '''
    Encodes a sequence of set-structured observations with a set-encoder (FF/SAB/ISAB),
    then runs a GRU/LSTM over time. Optionally, the previous action can be concatenated
    (useful for Q networks conditioned on action).

    Inputs (batched sequences):
      - slot_feats_seq: [B, T, S, d_in_slot]
      - slot_mask_seq:  [B, T, S]
      - self_feats_seq: [B, T, d_self] (e.g., agent-centric features)
      - action_seq:     [B, T, n_actions] or None (only if include_action=True)

    Returns:
      - H_seq:  [B, T, d_hidden]  RNN hidden states over time
      - hx_out: final hidden state (and cell for LSTM)

    Terminology used in this class:
      - sf := slot features (per time-step set)
      - sm := slot mask (per time-step validity mask)
      - sel:= selected top-K slot embeddings per frame
      - ctx:= optional soft attention context per frame
      - parts := list of fused feature blocks to concat for RNN input: [sel_flat, (ctx?), self, (action?)]
    '''

    def __init__(self, d_in_slot, d_self=2, d_slot=64, k=4, include_ctx=True,
                 rnn='gru', d_hidden=256, include_action=False, n_actions=2,
                 temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False,
                 set_method='ff'):
        super().__init__()

        # Build the set-encoder backbone (factory creates FF/SAB/ISAB variants)
        self.set_enc = set_transformers.make_set_encoder(
            method=set_method,   # 'ff' | 'sab' | 'isab'
            d_in=d_in_slot,
            d_slot=d_slot,
            k=k,
            include_ctx=include_ctx,
            # keep your knobs consistent with FF encoder
            n_heads_ff=2,
            temperature=temperature,
            slot_dropout_p=slot_dropout_p,
            gumbel_topk=gumbel_topk,
            # set-transformer specific knobs
            num_heads=4,
            dropout_p=0.0,
            num_sab=2,
            num_isab=2,
            num_inducing=16
        )

        # Compute the per-time-step fused input size to the RNN:
        # - k*d_slot              : flattened top-K embeddings
        # - +d_slot (optional)    : context vector if include_ctx=True
        # - +d_self               : self features
        # - +n_actions (optional) : previous action if include_action=True
        fused_in = k * d_slot + (d_slot if include_ctx else 0) + d_self
        if include_action:
            fused_in += n_actions

        # Configure the recurrent core. We normalize the string to lowercase so
        # 'GRU', 'Gru', 'gru' all work.
        self.rnn_type = rnn.lower()
        self.d_hidden = d_hidden
        if self.rnn_type == 'gru':
            # batch_first=True: inputs/outputs shaped [B, T, *]
            self.rnn = nn.GRU(input_size=fused_in,
                              hidden_size=d_hidden, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=fused_in,
                               hidden_size=d_hidden, batch_first=True)
        else:
            raise ValueError("rnn must be 'gru' or 'lstm'")

        self.include_ctx = include_ctx
        self.k = k
        self.d_slot = d_slot
        self.include_action = include_action

    def _fuse_step(self, slot_feats_t, slot_mask_t, self_feats_t, action_t=None):
        '''
        Single-time-step fusion helper used by the step-wise path (rarely used here).
        Takes one frame of set-structured features and returns a flat vector for the RNN.
        '''
        sel, ctx, _ = self.set_enc(slot_feats_t, slot_mask_t)  # sel: [B, K, d]
        B = slot_feats_t.size(0)
        sel_flat = sel.reshape(B, -1)  # [B, K*d]; flatten K selected slots
        parts = [sel_flat, self_feats_t]
        if self.include_ctx and ctx is not None:
            # keep ctx near the selected slots in feature order
            parts.insert(1, ctx)
        if self.include_action and (action_t is not None):
            parts.append(action_t)
        return torch.cat(parts, dim=-1)  # [B, fused_in]

    def forward(self, slot_feats_seq, slot_mask_seq, self_feats_seq, action_seq=None, hx=None):
        # ---- Flatten time into batch so we can run the set-encoder in one big call ----
        # B = batch size (windows)
        # T = time steps in each window
        # S = number of slots/elements per frame (e.g., fireflies)
        # d_in = per-slot feature dim
        B, T, S, d_in = slot_feats_seq.shape

        # WHY reshape and not view? reshape handles non-contiguous tensors safely.
        # sf := slot features, sm := slot mask.
        sf = slot_feats_seq.reshape(B * T, S, d_in)   # [B*T, S, d_in]
        sm = slot_mask_seq.reshape(B * T, S)          # [B*T, S]

        # ---- Set-encoder on all frames at once (much faster than looping T times) ----
        # sel: [B*T, K, d_slot]; ctx: [B*T, d_slot] or None
        sel, ctx, _ = self.set_enc(sf, sm)
        # [B*T, K*d_slot] (flatten K selected slots)
        sel_flat = sel.reshape(B * T, -1)

        # ---- Fuse features per frame into a single vector for the RNN ----
        parts = [sel_flat, self_feats_seq.reshape(B * T, -1)]
        if self.include_ctx and ctx is not None:
            # ctx placed between sel and self-feats
            parts.insert(1, ctx)
        if self.include_action and (action_seq is not None):
            parts.append(action_seq.reshape(B * T, -1))
        fused = torch.cat(parts, dim=-1).reshape(B, T, -1)  # [B, T, fused_in]

        '''
        B: batch size (how many sequences/windows you process in parallel).
        H: hidden size (the dimensionality of the RNN’s hidden state per direction).
        num_directions: 1 for a normal (uni-directional) RNN; 2 if bidirectional=True (forward + backward).
        hx: the hidden state you pass into the RNN (and get back out as hx_out).
        * For GRU: a single tensor h
        * For LSTM: a tuple (h, c) where h = hidden state, c = cell state
        '''

        # ---- Recurrent pass over time ----
        # hx semantics:
        #   • GRU: hx is [num_layers * num_directions, B, H]
        #   • LSTM: hx is (h, c), both shaped [num_layers * num_directions, B, H]

        H_seq, hx_out = self.rnn(fused, hx)  # H_seq: [B, T, H]
        return H_seq, hx_out


class SACRNNActor(nn.Module):
    '''
    Actor network for SAC with a recurrent backbone over set-structured observations.
    Produces per-time-step Gaussian parameters (mu, std) for a tanh-squashed policy.
    '''

    def __init__(self, d_in_slot=5, d_self=2, d_slot=64, k=4, include_ctx=True,
                 rnn='gru', d_hidden=256, n_actions=2, temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False,
                 set_method='ff'):
        super().__init__()
        self.enc = RecurrentAttentionEncoder(
            d_in_slot, d_self=d_self, d_slot=d_slot, k=k, include_ctx=include_ctx,
            rnn=rnn, d_hidden=d_hidden, include_action=False, n_actions=n_actions,
            temperature=temperature, slot_dropout_p=slot_dropout_p, gumbel_topk=gumbel_topk,
            set_method=set_method
        )
        # Linear heads from RNN state -> policy parameters
        self.mu = nn.Linear(d_hidden, n_actions)
        self.logstd = nn.Linear(d_hidden, n_actions)

    def forward(self, slot_feats_seq, slot_mask_seq, self_feats_seq, hx=None):
        # Encode and run RNN; we do not condition the actor on action_seq
        '''
        H_seq: the sequence of hidden states the RNN produces at every time step. Shape (with batch_first=True): [B, T, H].
        hx_out: the final hidden state after the last time step (used to continue the sequence later).
        '''
        H_seq, hx_out = self.enc(
            slot_feats_seq, slot_mask_seq, self_feats_seq, action_seq=None, hx=hx)
        mu_seq = self.mu(H_seq)
        # Clamp log-std to a reasonable range to avoid numerical issues.
        # Typical bounds (-5, 2) ⇒ std in [exp(-5)≈0.007, exp(2)≈7.39].
        logstd_seq = self.logstd(H_seq).clamp(-5, 2)
        std_seq = logstd_seq.exp()
        return mu_seq, std_seq, H_seq, hx_out


class SACRNNQ(nn.Module):
    '''
    Critic (Q-value) network for SAC. Same recurrent encoder, but we DO feed the action
    at each time step into the fusion so Q(s_t, a_t) is conditioned on actions.
    '''

    def __init__(self, d_in_slot=5, d_self=2, d_slot=64, k=4, include_ctx=True,
                 rnn='gru', d_hidden=256, n_actions=2, temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False,
                 set_method='ff'):
        super().__init__()
        self.enc = RecurrentAttentionEncoder(
            d_in_slot, d_self=d_self, d_slot=d_slot, k=k, include_ctx=include_ctx,
            rnn=rnn, d_hidden=d_hidden, include_action=True, n_actions=n_actions,
            temperature=temperature, slot_dropout_p=slot_dropout_p, gumbel_topk=gumbel_topk,
            set_method=set_method
        )
        self.head = nn.Linear(d_hidden, 1)  # scalar Q per time step

    def forward(self, slot_feats_seq, slot_mask_seq, self_feats_seq, action_seq, hx=None):
        H_seq, hx_out = self.enc(
            slot_feats_seq, slot_mask_seq, self_feats_seq, action_seq=action_seq, hx=hx)
        return self.head(H_seq).squeeze(-1), hx_out  # [B, T], hx


# ---------------------------
# SAC utilities
# ---------------------------

def sample_action(mu: torch.Tensor, std: torch.Tensor, action_limits: List[Tuple[float, float]]):
    '''
    Reparameterized sampling for tanh-squashed Gaussian policies (SAC).

    Inputs:
      - mu, std:    Gaussian parameters. Accepts [B, T, A] or [B*T, A].
      - action_limits: list of (low, high) per action dim; used to rescale tanh to env range.

    Returns:
      - a_scaled: sampled actions in env range, same leading shape as input
      - logp:     log-probabilities of the squashed actions (for SAC objective)

    Details of the log-prob correction:
      Let a = tanh(z), z ~ N(mu, std). Then da/dz = 1 - tanh(z)^2.
      Change-of-variables ⇒ log p(a) = log p(z) - sum log |da/dz| = log p(z) - sum log(1 - tanh(z)^2).
    '''
    orig_shape = mu.shape

    # If a sequence [B, T, A] is passed, flatten time for vectorized sampling
    if mu.dim() == 3:
        B, T, A = mu.shape
        mu = mu.reshape(B * T, A)
        std = std.reshape(B * T, A)

    # 1) Sample z ~ N(mu, std) using rsample() to enable backprop through randomness
    normal = torch.distributions.Normal(mu, std)
    z = normal.rsample()

    # 2) Squash via tanh to (-1,1)
    a = torch.tanh(z)

    # 3) Correct log-prob for tanh squashing (change-of-variables)
    logp = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
    logp = logp.sum(dim=-1, keepdim=True)  # sum over action dims

    # 4) Rescale each action dim from (-1,1) to [low, high]
    A = a.size(-1)
    a_scaled_list = []
    for j in range(A):
        lo, hi = action_limits[j]
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        a_scaled_list.append(mid + half * a[:, j:j+1])
    a_scaled = torch.cat(a_scaled_list, dim=-1)

    # Restore original shape if we had [B, T, A]
    if len(orig_shape) == 3:
        a_scaled = a_scaled.reshape(B, T, A)
        logp = logp.reshape(B, T, 1)
    return a_scaled, logp


def sac_train_step_rnn(
    *,
    batch: Dict[str, torch.Tensor],
    actor: SACRNNActor, q1: SACRNNQ, q2: SACRNNQ,
    actor_targ: SACRNNActor, q1_targ: SACRNNQ, q2_targ: SACRNNQ,
    actor_opt, q1_opt, q2_opt,
    log_alpha: torch.nn.Parameter, alpha_opt,
    gamma: float, tau: float, target_entropy: float,
    action_limits: List[Tuple[float, float]],
    burn_in: int,
    grad_clip: float = 1.0,
    amp: bool = False,
    device=torch.device('cpu'),
) -> Dict[str, float]:
    '''
    One full SAC update using sequence windows with a burn-in phase for the RNN state.

    - Burn-in: we first unroll the RNNs on the first `burn_in` steps to build hidden states
      (no learning from these steps). This stabilizes training on long sequences.
    - Train window: we then compute targets and losses on the remaining steps.

    The function updates Q1, Q2, Actor, and temperature alpha. It also Polyak-averages
    the target networks.

    Shapes used inside (after slicing):
      s_tr, m_tr, ss_tr, a_tr, r_tr, d_tr: [B, Ttr, ...] where Ttr = T - burn_in.
      Next-state versions have the same shape; hidden states hx_* match RNN conventions.
    '''
    # Unpack batched tensors (all shapes [B, T, ...])
    slot_feats_seq = batch['slot_feats_seq']
    slot_mask_seq = batch['slot_mask_seq']
    self_feats_seq = batch['self_feats_seq']
    action_seq = batch['action_seq']
    reward_seq = batch['reward_seq']
    done_seq = batch['done_seq']
    slot_feats_seq_next = batch['slot_feats_seq_next']
    slot_mask_seq_next = batch['slot_mask_seq_next']
    self_feats_seq_next = batch['self_feats_seq_next']

    # Basic sanity
    B, T, S, N = slot_feats_seq.shape
    assert T > burn_in, 'T must be > burn_in'
    Ttr = T - burn_in  # length of the training portion

    # Temperature parameter (alpha) is exp(log_alpha). For the critic target we detach it
    # to stop gradients flowing into alpha via the target.
    alpha = log_alpha.exp().detach()

    # AMP mixed precision context (cuda only). No-op on CPU.
    # GradScaler keeps dynamic loss scaling to avoid underflow in FP16.
    use_amp = bool(amp and torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- Burn-in (no grads) ----
    # Each network rolls forward on the first `burn_in` steps to produce its hidden state.
    # IMPORTANT: target networks do their OWN burn-in to avoid leaking online states into targets.
    with torch.no_grad():
        _, _, _, hx_actor = actor(
            slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in], self_feats_seq[:, :burn_in], hx=None)
        _, hx_q1 = q1(slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in],
                      self_feats_seq[:, :burn_in], action_seq[:, :burn_in], hx=None)
        _, hx_q2 = q2(slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in],
                      self_feats_seq[:, :burn_in], action_seq[:, :burn_in], hx=None)
        # Targets use their OWN burn-in to avoid state sharing.
        _, _, _, hx_actor_t = actor_targ(
            slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in], self_feats_seq[:, :burn_in], hx=None)
        _, hx_q1_t = q1_targ(slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in],
                             self_feats_seq[:, :burn_in], action_seq[:, :burn_in], hx=None)
        _, hx_q2_t = q2_targ(slot_feats_seq[:, :burn_in], slot_mask_seq[:, :burn_in],
                             self_feats_seq[:, :burn_in], action_seq[:, :burn_in], hx=None)

    # ---- Slice the train window (remaining Ttr steps) ---- Note: tr stands for training region
    s_tr = slot_feats_seq[:, burn_in:]
    m_tr = slot_mask_seq[:, burn_in:]
    ss_tr = self_feats_seq[:, burn_in:]
    a_tr = action_seq[:, burn_in:]
    r_tr = reward_seq[:, burn_in:]
    d_tr = done_seq[:, burn_in:]
    s_tr_n = slot_feats_seq_next[:, burn_in:]
    m_tr_n = slot_mask_seq_next[:, burn_in:]
    ss_tr_n = self_feats_seq_next[:, burn_in:]

    # ---- Targets (critic bootstrapping with target policy) ----
    with torch.no_grad():
        # Next actions and their log-probs from ACTOR (online) unrolled from target hidden state.
        # We use the ONLINE actor with the TARGET critics, as in standard SAC.
        mu_n, std_n, _, _ = actor(s_tr_n, m_tr_n, ss_tr_n, hx_actor)
        # shapes: [B, Ttr, A], [B, Ttr, 1]
        a_n, logp_n = sample_action(mu_n, std_n, action_limits)

        # Target critics evaluate Q(s', a')
        q1n_seq, _ = q1_targ(s_tr_n, m_tr_n, ss_tr_n, a_n, hx_q1_t)
        q2n_seq, _ = q2_targ(s_tr_n, m_tr_n, ss_tr_n, a_n, hx_q2_t)
        q_next = torch.min(q1n_seq, q2n_seq) - alpha * logp_n.squeeze(-1)

        # Bellman target: y = r + gamma * (1 - done) * q_next
        # NOTE: if your env uses time-limit truncation, you may want to treat truncation
        # separately from terminal dones (bootstrap on trunc, not on true terminal).
        y_seq = r_tr + gamma * (1.0 - d_tr) * q_next

    # Autocast context for faster matmuls in FP16 (only on CUDA)
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()

    # ---- Critic updates ----
    with amp_ctx:
        q1_seq, _ = q1(s_tr, m_tr, ss_tr, a_tr, hx_q1)
        q2_seq, _ = q2(s_tr, m_tr, ss_tr, a_tr, hx_q2)
        loss_q1 = F.mse_loss(q1_seq, y_seq)
        loss_q2 = F.mse_loss(q2_seq, y_seq)

    # Optimize Q1
    q1_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_q1).backward()
    if grad_clip:
        scaler.unscale_(q1_opt)
        nn.utils.clip_grad_norm_(q1.parameters(), grad_clip)
    scaler.step(q1_opt)

    # Optimize Q2
    q2_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_q2).backward()
    if grad_clip:
        scaler.unscale_(q2_opt)
        nn.utils.clip_grad_norm_(q2.parameters(), grad_clip)
    scaler.step(q2_opt)

    # ---- Actor update ----
    with amp_ctx:
        mu_seq, std_seq, _, _ = actor(s_tr, m_tr, ss_tr, hx_actor)
        a_pi, logp = sample_action(mu_seq, std_seq, action_limits)
        q1_pi, _ = q1(s_tr, m_tr, ss_tr, a_pi, hx_q1)
        q2_pi, _ = q2(s_tr, m_tr, ss_tr, a_pi, hx_q2)
        q_min = torch.min(q1_pi, q2_pi)
        # SAC policy loss: E[alpha * log pi(a|s) - Q(s,a)] (maximize entropy, minimize energy)
        loss_actor = (log_alpha.exp() * logp.squeeze(-1) - q_min).mean()

    actor_opt.zero_grad(set_to_none=True)
    scaler.scale(loss_actor).backward()
    if grad_clip:
        scaler.unscale_(actor_opt)
        nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    scaler.step(actor_opt)
    scaler.update()  # update scaler once per step

    # ---- Temperature (alpha - the weight on the entropy term) update ----
    # We want E[-log pi] to match target_entropy (typically negative ~ -|A|).
    # Gradient is w.r.t. log_alpha; detach logp so the policy doesn't get this gradient.
    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()
    alpha_new = log_alpha.exp()

    # ---- Polyak averaging (target network update) ----
    @torch.no_grad()
    def polyak(src, tgt, tau_):
        # Standard EMA update: tgt <- tau*tgt + (1-tau)*src
        for p, p_t in zip(src.parameters(), tgt.parameters()):
            p_t.data.mul_(tau_).add_(p.data, alpha=1.0 - tau_)

    with torch.no_grad():
        polyak(q1,    q1_targ,   tau)
        polyak(q2,    q2_targ,   tau)
        polyak(actor, actor_targ, tau)

    # Return scalar metrics for logging
    return {
        'loss_q1': float(loss_q1.detach().cpu().item()),
        'loss_q2': float(loss_q2.detach().cpu().item()),
        'loss_actor': float(loss_actor.detach().cpu().item()),
        'alpha_loss': float(alpha_loss.detach().cpu().item()),
        'alpha': float(alpha_new.detach().cpu().item()),
        'q1_mean': float(q1_seq.detach().cpu().mean().item()),
        'q2_mean': float(q2_seq.detach().cpu().mean().item()),
    }


# ---------------------------
# Sequence Replay (episode buffer -> random windows)
# ---------------------------


class SequenceReplay:
    '''
    Stores complete episodes and samples random fixed-length windows of length `seq_len_total`
    without materializing every candidate window. Sampling is O(E) to build counts
    and O(B log E) to map global window ids -> (episode, start).

    Notes:
      - Uniform over all windows (not episodes).
      - Supports replacement sampling when batch_size exceeds #windows.
      - Optional torch conversion on return.
    '''

    def __init__(self, capacity_episodes: int = 1000, device: Optional[torch.device] = None):
        self.capacity_episodes = capacity_episodes
        self.episodes: deque[Dict[str, np.ndarray]] = deque()  # FIFO
        self._ep: Optional[Dict[str, List]] = None
        self.device = device

    # ---------- write path ----------
    def start_episode(self):
        self._ep = {'obs': [], 'action': [],
                    'reward': [], 'done': [], 'next_obs': []}

    def add_step(self, obs, action, reward, done, next_obs):
        assert self._ep is not None, 'Call start_episode() before adding steps'
        self._ep['obs'].append(obs)
        self._ep['action'].append(action)
        self._ep['reward'].append(reward)
        self._ep['done'].append(done)
        self._ep['next_obs'].append(next_obs)

    def end_episode(self):
        assert self._ep is not None, 'No open episode to end'
        ep = {k: np.asarray(v) for k, v in self._ep.items()}
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity_episodes:
            self.episodes.popleft()
        self._ep = None

    def add_episode(self, ep: Dict[str, np.ndarray]):
        '''Optional fast path to add a finished episode already as ndarrays.'''
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity_episodes:
            self.episodes.popleft()

    # ---------- read path ----------
    def _window_counts(self, seq_len_total: int) -> np.ndarray:
        # number of valid start indices per episode
        if not self.episodes:
            return np.array([], dtype=np.int64)
        counts = np.fromiter(
            (max(0, ep['obs'].shape[0] - seq_len_total + 1) for ep in self.episodes),
            dtype=np.int64,
            count=len(self.episodes),
        )
        return counts

    def sample(self, batch_size: int, seq_len_total: int, replace: Optional[bool] = None, to_torch: bool = False):
        '''
        Uniformly sample `batch_size` windows of length `seq_len_total` across all episodes.

        Args:
          batch_size: number of windows
          seq_len_total: window length
          replace: if None, auto-choose True when batch_size > total_windows
          to_torch: if True, return torch tensors (moved to self.device if set)

        Returns:
          dict with leading shape [B, seq_len_total, ...]
        '''
        counts = self._window_counts(seq_len_total)
        total_windows = int(counts.sum())
        assert total_windows > 0, 'No eligible windows; add more experience or reduce seq_len_total.'

        if replace is None:
            replace = batch_size > total_windows

        if not replace:
            assert batch_size <= total_windows, f'Need more experience: have {total_windows} windows, want {batch_size}.'

        # cumulative windows to map global ids -> (episode, start)
        cumsum = np.cumsum(counts, dtype=np.int64)
        # draw global window ids
        global_ids = np.random.choice(
            total_windows, size=batch_size, replace=replace)
        # map to episode indices (binary search over cumsum)
        ep_idx = np.searchsorted(cumsum, global_ids, side='right')
        # start index within each episode
        starts_base = np.zeros_like(cumsum)
        starts_base[1:] = cumsum[:-1]
        starts = global_ids - starts_base[ep_idx]

        # gather
        obs, act, rew, done, nxt = [], [], [], [], []
        for ei, s in zip(ep_idx, starts):
            ep = self.episodes[int(ei)]
            e_slice = slice(int(s), int(s + seq_len_total))
            obs.append(ep['obs'][e_slice])
            act.append(ep['action'][e_slice])
            rew.append(ep['reward'][e_slice])
            done.append(ep['done'][e_slice])
            nxt.append(ep['next_obs'][e_slice])

        batch = dict(
            obs_seq=np.stack(obs, 0),
            action_seq=np.stack(act, 0),
            reward_seq=np.stack(rew, 0),
            done_seq=np.stack(done, 0),
            next_obs_seq=np.stack(nxt, 0),
        )

        if to_torch:
            device = self.device if self.device is not None else 'cpu'
            # keep dtypes as-is; consumers can cast as needed
            batch = {k: torch.from_numpy(v).to(device=device)
                     for k, v in batch.items()}

        return batch


# ---------------------------
# High-level wrapper (training loop, I/O)
# ---------------------------

class AttnRNNSACforMultifirefly:
    '''
    Convenience wrapper that wires together env, replay buffer, actor/critics,
    and provides a simple training loop with logging and evaluation.

    Tip: call make_env() -> make_agent() -> regular_training().
    '''

    def __init__(self,
                 model_folder: str = 'RL_models/ATTN_RNN/agent_0/',
                 device: Optional[str] = None,
                 seed: int = 42,
                 **env_kwargs):
        seed_all(seed)
        self.model_folder = model_folder
        # Choose device priority: CUDA -> Metal (MPS) -> CPU
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else (
                'mps' if torch.backends.mps.is_available() else 'cpu')
        )
        self.env_kwargs = env_kwargs.copy()
        self.env = None

        # Model handles
        self.actor = self.q1 = self.q2 = None
        self.actor_targ = self.q1_targ = self.q2_targ = None
        self.actor_opt = self.q1_opt = self.q2_opt = None
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=self.device))
        self.alpha_opt = None

        # Sequence replay buffer
        self.replay_seq = SequenceReplay(
            capacity_episodes=1000, device=self.device)

        # Default configuration (override via make_agent(overrides))
        self.cfg = dict(
            # encoder/RNN
            k_top=7, d_slot=32, include_ctx=True, rnn='gru', d_hidden=128,
            set_method='ff',
            attn_temperature=1.0, slot_dropout_p=0.0, gumbel_topk=False,
            # sac
            lr=3e-4, gamma=0.99, tau=0.995, target_entropy=-2.0, grad_clip=1.0, amp=False,
            # training
            batch_size=64, seq_len_total=32, burn_in=8, updates_per_step=1,
            log_every=1000, eval_every=0, eval_episodes=3, save_every=0
        )

    # ---- Env ----
    def make_env(self, **env_kwargs):
        '''Instantiate the MultiFF env with stored + provided kwargs.'''
        current_env_kwargs = copy.deepcopy(self.env_kwargs)
        current_env_kwargs.update(env_kwargs)
        self.env = EnvForAttentionSAC(**current_env_kwargs)
        return self.env

    # ---- Agent ----
    def make_agent(self, **overrides):
        '''
        Build actor/critic networks (+targets) and their optimizers.
        Call make_env() before this so we can read observation/action shapes.
        '''
        assert self.env is not None, 'Call make_env() first.'
        self.cfg.update(overrides)

        d_in_slot = self.env.num_elem_per_ff
        n_actions = int(np.prod(self.env.action_space.shape))
        d_self = 2  # e.g., self velocity, etc.
        K = self.cfg['k_top']
        D = self.cfg['d_slot']

        actor_kw = dict(
            d_in_slot=d_in_slot, d_self=d_self, d_slot=D, k=K, include_ctx=self.cfg[
                'include_ctx'],
            rnn=self.cfg['rnn'], d_hidden=self.cfg['d_hidden'], n_actions=n_actions,
            temperature=self.cfg['attn_temperature'], slot_dropout_p=self.cfg['slot_dropout_p'],
            gumbel_topk=self.cfg['gumbel_topk'], set_method=self.cfg.get(
                'set_method', 'ff')
        )
        self.actor = SACRNNActor(**actor_kw).to(self.device)
        self.q1 = SACRNNQ(**actor_kw).to(self.device)
        self.q2 = SACRNNQ(**actor_kw).to(self.device)

        # Targets start as exact copies (Polyak will keep them close)
        self.actor_targ = SACRNNActor(**actor_kw).to(self.device)
        self.q1_targ = SACRNNQ(**actor_kw).to(self.device)
        self.q2_targ = SACRNNQ(**actor_kw).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # Optimizers (single LR for simplicity)
        lr = self.cfg['lr']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    # ---- Train ----
    def regular_training(self, total_env_steps: int = 50000):
        '''
        Main on-policy interaction loop with off-policy updates from SequenceReplay windows.
        '''
        assert self.env is not None and self.actor is not None
        device = self.device
        cfg = self.cfg

        # Cache hot config/lookups locally (fewer dict/attr hops in the loop)
        updates_per_step = int(cfg['updates_per_step'])
        batch_size = int(cfg['batch_size'])
        seq_len_total = int(cfg['seq_len_total'])
        burn_in = int(cfg['burn_in'])
        gamma = float(cfg['gamma'])
        tau = float(cfg['tau'])
        target_entropy = float(cfg['target_entropy'])
        grad_clip = cfg['grad_clip']
        use_amp = bool(cfg['amp'])
        log_every = int(cfg['log_every'])
        eval_every = int(cfg['eval_every']) if cfg.get('eval_every') else 0
        save_every = int(cfg['save_every']) if cfg.get('save_every') else 0
        eval_episodes = int(cfg['eval_episodes']) if cfg.get(
            'eval_episodes') else 0

        action_limits = get_action_limits(self.env)

        # Backends: safe on CPU, useful on CUDA
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # Prepare metrics CSV
        os.makedirs(self.model_folder, exist_ok=True)
        csv_path = os.path.join(self.model_folder, 'metrics.csv')
        csv_f = open(csv_path, 'w', newline='')
        csv_w = csv.DictWriter(
            csv_f,
            fieldnames=['t', 'ep', 'ep_len', 'ep_return', 'episodes_buf',
                        'alpha', 'loss_q1', 'loss_q2', 'loss_actor']
        )
        csv_w.writeheader()

        # Local refs to hot callables
        env = self.env
        obs_to_step_tensors = env.obs_to_attn_tensors
        seq_to_attn = seq_obs_to_attn_tensors_torch  # keep same semantics
        train_step = sac_train_step_rnn

        # Reset env and begin first episode
        obs, _ = env.reset()
        self.replay_seq.start_episode()
        ep_ret, ep_len, ep = 0.0, 0, 0
        last_eval = 0
        metrics = {}

        for t in range(1, total_env_steps + 1):
            # ------------------------------
            # 1) ACT: one-step roll with RNN (T=1)
            # ------------------------------
            # Convert obs -> attention tensors once; avoid repeated unsqueezes
            # [1, S, d], [1, S], [1, d_self]
            sf, sm, ss = obs_to_step_tensors(obs, device=device)
            with torch.no_grad():
                # Add T dimension via view (cheaper than unsqueeze chain)
                sf1 = sf.view(1, 1, *sf.shape[1:])   # [B=1, T=1, S, d]
                sm1 = sm.view(1, 1, *sm.shape[1:])   # [1, 1, S]
                ss1 = ss.view(1, 1, *ss.shape[1:])   # [1, 1, d_self]
                mu1, std1, _, _ = self.actor(sf1, sm1, ss1, hx=None)
                a_t, _ = sample_action(mu1[:, -1], std1[:, -1], action_limits)
            action = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Env step
            next_obs, reward, done, truncated, _ = env.step(action)

            # Push transition (raw obs; projection happens during updates)
            self.replay_seq.add_step(
                obs=obs,
                action=action,
                reward=np.float32(reward),
                done=np.float32(done),
                next_obs=next_obs,
            )
            ep_ret += reward
            ep_len += 1

            # ------------------------------
            # 2) UPDATE: draw random windows and train
            # ------------------------------
            # Try to do all conversions inside replay (to_torch=True) to cut copies
            for _ in range(updates_per_step):
                try:
                    batch_t = self.replay_seq.sample(
                        batch_size=batch_size,
                        seq_len_total=seq_len_total,
                        replace=None,           # auto when needed
                        to_torch=True           # returns torch tensors directly
                    )
                except AssertionError:
                    break  # not enough windows yet

                # Project observations for current/next in one go (keeps parity with your helper)
                obs_seq_t = batch_t['obs_seq'].to(
                    device=device, dtype=torch.float32)
                next_obs_seq_t = batch_t['next_obs_seq'].to(
                    device=device, dtype=torch.float32)

                sf_seq, sm_seq, ss_seq = seq_to_attn(
                    obs_seq_t, env, device=device)
                sfn_seq, smn_seq, ssn_seq = seq_to_attn(
                    next_obs_seq_t, env, device=device)

                # Reuse tensors already on device; avoid re-wrapping via torch.tensor(...)
                batch_t = {
                    'slot_feats_seq': sf_seq,
                    'slot_mask_seq': sm_seq,
                    'self_feats_seq': ss_seq,
                    'action_seq': batch_t['action_seq'].to(device=device, dtype=torch.float32),
                    'reward_seq': batch_t['reward_seq'].to(device=device, dtype=torch.float32),
                    'done_seq': batch_t['done_seq'].to(device=device, dtype=torch.float32),
                    'slot_feats_seq_next': sfn_seq,
                    'slot_mask_seq_next': smn_seq,
                    'self_feats_seq_next': ssn_seq,
                }

                # Single SAC update on sequences
                metrics = train_step(
                    batch=batch_t,
                    actor=self.actor, q1=self.q1, q2=self.q2,
                    actor_targ=self.actor_targ, q1_targ=self.q1_targ, q2_targ=self.q2_targ,
                    actor_opt=self.actor_opt, q1_opt=self.q1_opt, q2_opt=self.q2_opt,
                    log_alpha=self.log_alpha, alpha_opt=self.alpha_opt,
                    gamma=gamma, tau=tau, target_entropy=target_entropy,
                    action_limits=action_limits, burn_in=burn_in,
                    grad_clip=grad_clip, amp=use_amp, device=device,
                )

            # ------------------------------
            # 3) EPISODE bookkeeping & logging
            # ------------------------------
            if done or truncated:
                ep += 1
                self.replay_seq.end_episode()
                print(
                    f'Ep {ep:4d} | steps={ep_len:4d} | return={ep_ret:8.2f} | eps_buf={len(self.replay_seq.episodes):4d}')
                csv_w.writerow(dict(
                    t=t, ep=ep, ep_len=ep_len, ep_return=ep_ret, episodes_buf=len(
                        self.replay_seq.episodes),
                    alpha=float(self.log_alpha.exp().detach().cpu().item()),
                    loss_q1=metrics.get('loss_q1', float('nan')),
                    loss_q2=metrics.get('loss_q2', float('nan')),
                    loss_actor=metrics.get('loss_actor', float('nan')),
                ))
                obs, _ = env.reset()
                self.replay_seq.start_episode()
                ep_ret, ep_len = 0.0, 0
            else:
                obs = next_obs

            if (t % log_every == 0) and metrics:
                # Note: avoid nested quotes clash
                print(
                    f'[t={t}] alpha={float(self.log_alpha.exp()):.3f}  '
                    f'losses: Q1={metrics["loss_q1"]:.3f} Q2={metrics["loss_q2"]:.3f} Pi={metrics["loss_actor"]:.3f}'
                )

            if eval_every and (t - last_eval) >= eval_every:
                last_eval = t
                avg = self.test_agent(
                    num_eps=eval_episodes, deterministic=True)
                print(f'[eval @ t={t}] avg_return={avg:.2f}')
                if save_every:
                    self.save_agent(dir_name=os.path.join(
                        self.model_folder, f'ckpt_{t:07d}'))

        csv_f.close()
        return metrics

    # ---- Save/Load ----

    def save_agent(self, dir_name: Optional[str] = None):
        '''Save model + optimizer states to disk (single .pt payload).'''
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
        torch.save(payload, os.path.join(path, 'model.pt'))
        print(f'Saved to {path}')

    def load_agent(self, dir_name: Optional[str] = None):
        '''Load model + optimizer states from disk.'''
        path = self.model_folder if dir_name is None else dir_name
        payload = torch.load(os.path.join(
            path, 'model.pt'), map_location='cpu')
        self.actor.load_state_dict(payload['actor'])
        self.q1.load_state_dict(payload['q1'])
        self.q2.load_state_dict(payload['q2'])
        self.actor_targ.load_state_dict(payload['actor_targ'])
        self.q1_targ.load_state_dict(payload['q1_targ'])
        self.q2_targ.load_state_dict(payload['q2_targ'])
        with torch.no_grad():
            self.log_alpha.copy_(payload['log_alpha'].to(self.device))
        self.actor_opt.load_state_dict(payload['opt_actor'])
        self.q1_opt.load_state_dict(payload['opt_q1'])
        self.q2_opt.load_state_dict(payload['opt_q2'])
        self.alpha_opt.load_state_dict(payload['opt_alpha'])
        print(f'Loaded from {path}')

    # ---- Eval ----
    def test_agent(self, num_eps: int = 3, max_steps_per_eps: Optional[int] = None, deterministic: bool = True):
        '''
        Roll out the current policy for `num_eps` episodes and report the average return.
        If `deterministic=True`, we use tanh(mu) instead of sampling for lower-variance eval.
        '''
        assert self.env is not None and self.actor is not None
        max_steps = max_steps_per_eps or self.env.episode_len
        action_limits = get_action_limits(self.env)
        returns = []
        for e in range(num_eps):
            obs, info = self.env.reset()
            ret = 0.0
            hx = None  # carry hidden state across time for efficiency
            for t in range(max_steps):
                sf, sm, ss = self.env.obs_to_attn_tensors(
                    obs, device=self.device)
                with torch.no_grad():
                    mu_seq, std_seq, H_seq, hx = self.actor(
                        sf.unsqueeze(1), sm.unsqueeze(1), ss.unsqueeze(1), hx=hx)
                    mu = mu_seq[:, -1]
                    std = std_seq[:, -1]
                    if deterministic:
                        # Deterministic action = tanh(mu), rescaled to env range
                        a = torch.tanh(mu)
                        a_scaled_list = []
                        for j in range(a.size(-1)):
                            lo, hi = action_limits[j]
                            mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
                            a_scaled_list.append(mid + half * a[:, j:j+1])
                        act = torch.cat(a_scaled_list, dim=-1)
                    else:
                        act, _ = sample_action(mu, std, action_limits)
                action = act.squeeze(0).cpu().numpy().astype(np.float32)
                obs, r, done, trunc, _ = self.env.step(action)
                ret += r
                if done or trunc:
                    break
            returns.append(ret)
            print(f'Eval ep {e+1}/{num_eps}: return={ret:.2f}')
        avg = float(np.mean(returns))
        print(f'Average return over {num_eps} eps: {avg:.2f}')
        return avg
