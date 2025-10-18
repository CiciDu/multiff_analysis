"""
env_attn_multiff.py
-------------------
Shared environment wrapper and utilities for attention-based agents on MultiFF.

* Requires a class named `MultiFF` in the Python session (or importable).
* Exposes `EnvForAttentionSAC` that:
  - keeps identity-bound slots from MultiFF,
  - lets you choose WHICH per-slot fields to output (by name),
  - re-centers distance/time fields to [-1, 1] to match networks,
  - provides helpers to convert flat observations to attention tensors,
  - generalizes action-limit handling from the env's Box space if available.

Typical usage:
    from env_attn_multiff import EnvForAttentionSAC, get_action_limits, batch_obs_to_attn_tensors
"""

from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import torch
from reinforcement_learning.base_classes import base_env, more_envs

try:
    import gymnasium as gym
except Exception:
    gym = None  # optional, only used to declare Box spaces

# ---------------------------------------------------------------------
# Helper: pull action limits from env.action_space (Box) if available
# ---------------------------------------------------------------------


def get_action_limits(env) -> List[Tuple[float, float]]:
    """
    Return [(low, high), ...] per action dimension.
    Falls back to [(-1,1)] if not a Box.
    """
    if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
        lo = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        return [(float(l), float(h)) for l, h in zip(lo, hi)]
    # Default to SAC-friendly bounds if unknown
    return [(-1.0, 1.0), (-1.0, 1.0)]

# ---------------------------------------------------------------------
# EnvForAttentionSAC
# ---------------------------------------------------------------------


class EnvForAttentionSAC(base_env.MultiFF):
    """
    MultiFF variant for attention encoders.

    The wrapper:
      * keeps identity-bound slots (no resorting each step),
      * appends previous action tail if configured by base,
      * offers conversion helpers to attention-ready tensors.
    """

    def __init__(
        self,
        obs_visible_only=True,
        **kwargs
    ):
        super().__init__(obs_visible_only=obs_visible_only, **kwargs)

    # -------------- one-sample converter: flat obs -> attention tensors --------------

    def obs_to_attn_tensors(self, obs_flat: np.ndarray, device='cpu', use_prev_action_as_self=True):
        """
        Convert a single flat observation into attention-ready arrays.

        Returns:
          slot_feats : [1, S, N]
          slot_mask  : [1, S]
          self_feats : [1, 2]  (by default: previous action tail; else zeros)
        """
        S = self.num_obs_ff
        N = self.num_elem_per_ff

        # Pre-calculate segment boundaries for efficient slicing
        slots_len = S * N
        action_len = 2 if self.add_action_to_obs else 0
        mask_len = 0
        total_len = slots_len + action_len

        assert obs_flat.size == total_len, f"obs length mismatch: expected {total_len}, got {obs_flat.size}"

        # Efficient slicing using pre-calculated boundaries
        slot_start = 0
        slot_end = slots_len
        action_start = slot_end
        action_end = action_start + action_len

        # Extract components
        raw_slots = obs_flat[slot_start:slot_end].reshape(
            S, N).astype(np.float32)

        # Create slot features
        slot_feats = raw_slots.reshape(1, S, N)

        # Create slot mask from 'valid' field or default to ones
        if self._valid_field_index is not None:
            slot_mask = (raw_slots[:, self._valid_field_index] > 0.5).astype(
                np.float32).reshape(1, S)
        else:
            slot_mask = np.ones((1, S), dtype=np.float32)

        # Create self features
        if use_prev_action_as_self and action_len > 0:
            self_feats = obs_flat[action_start:action_end].astype(
                np.float32).reshape(1, 2)
        else:
            self_feats = np.zeros((1, 2), dtype=np.float32)

        # Convert to torch tensors on device
        slot_feats_t = torch.from_numpy(slot_feats).to(device)
        slot_mask_t = torch.from_numpy(slot_mask).to(device)
        self_feats_t = torch.from_numpy(self_feats).to(device)

        return slot_feats_t, slot_mask_t, self_feats_t


# -------------------- batch & sequence converters --------------------
def batch_obs_to_attn_tensors(obs_batch: np.ndarray, env: EnvForAttentionSAC, device='cpu', use_prev_action_as_self=True):
    """
    Use for non-recurrent (single-step) updates. For sequences/RNNs, use `seq_obs_to_attn_tensors`.

    obs_batch: [B, obs_dim] flat observations
    Returns:
      slot_feats : [B, S, N]
      slot_mask  : [B, S]
      self_feats : [B, 2]
    """
    B = obs_batch.shape[0]
    S = env.num_obs_ff
    N = env.num_elem_per_ff

    # Pre-calculate segment boundaries for efficient batch processing
    slots_len = S * N
    action_len = 2 if env.add_action_to_obs else 0
    mask_len = 0

    # Efficient batch slicing
    slot_feats = obs_batch[:, :slots_len].reshape(B, S, N).astype(np.float32)

    # Handle action tail
    if action_len > 0:
        action_tail = obs_batch[:, slots_len:slots_len +
                                action_len].astype(np.float32)
    else:
        action_tail = np.zeros((B, 2), dtype=np.float32)

    # Derive slot mask from 'valid' field or default to ones
    if env._valid_field_index is not None:
        slot_mask = (slot_feats[..., env._valid_field_index] > 0.5).astype(
            np.float32)
    else:
        slot_mask = np.ones((B, S), dtype=np.float32)

    # Handle self features
    if use_prev_action_as_self:
        self_feats = action_tail
    else:
        self_feats = np.zeros((B, 2), dtype=np.float32)

    # Convert to torch tensors on device
    slot_feats_t = torch.from_numpy(slot_feats).to(device)
    slot_mask_t = torch.from_numpy(slot_mask).to(device)
    self_feats_t = torch.from_numpy(self_feats).to(device)

    return slot_feats_t, slot_mask_t, self_feats_t


def seq_obs_to_attn_tensors(obs_seq_batch: np.ndarray, env: EnvForAttentionSAC, device='cpu', use_prev_action_as_self=True):
    """
    Use for RNN/sequence training (adds time dimension T). For single-step batches, use `batch_obs_to_attn_tensors`.

    obs_seq_batch: [B, T, obs_dim]
    Returns:
      slot_feats_seq : [B, T, S, N]
      slot_mask_seq  : [B, T, S]
      self_feats_seq : [B, T, 2]
    """
    B, T, OD = obs_seq_batch.shape
    S = env.num_obs_ff
    N = env.num_elem_per_ff

    # Pre-calculate segment boundaries for efficient sequence processing
    slots_len = S * N
    action_len = 2 if env.add_action_to_obs else 0
    mask_len = 0

    # Efficient sequence slicing
    slot_feats = obs_seq_batch[:, :, :slots_len].reshape(
        B, T, S, N).astype(np.float32)

    # Handle action tail
    if action_len > 0:
        action_tail = obs_seq_batch[:, :,
                                    slots_len:slots_len + action_len].astype(np.float32)
    else:
        action_tail = np.zeros((B, T, 2), dtype=np.float32)

    # Derive slot mask from 'valid' field or default to ones
    if env._valid_field_index is not None:
        slot_mask = (slot_feats[..., env._valid_field_index] > 0.5).astype(
            np.float32)
    else:
        slot_mask = np.ones((B, T, S), dtype=np.float32)

    # Handle self features
    if use_prev_action_as_self:
        self_feats = action_tail
    else:
        self_feats = np.zeros((B, T, 2), dtype=np.float32)

    # Convert to torch tensors on device
    slot_feats_t = torch.from_numpy(slot_feats).to(device)
    slot_mask_t = torch.from_numpy(slot_mask).to(device)
    self_feats_t = torch.from_numpy(self_feats).to(device)

    return slot_feats_t, slot_mask_t, self_feats_t


def seq_obs_to_attn_tensors_torch(obs_seq_batch: torch.Tensor, env: EnvForAttentionSAC, device='cpu', use_prev_action_as_self=True):
    """
    Torch-native version of seq_obs_to_attn_tensors to avoid numpy<->torch copies.
    obs_seq_batch: [B, T, obs_dim] torch.float32 tensor (on any device)
    Returns:
      slot_feats_seq : [B, T, S, N]
      slot_mask_seq  : [B, T, S]
      self_feats_seq : [B, T, 2]
    """
    assert isinstance(
        obs_seq_batch, torch.Tensor), "obs_seq_batch must be torch.Tensor"
    B, T, OD = obs_seq_batch.shape
    S = env.num_obs_ff
    N = env.num_elem_per_ff

    slots_len = S * N
    action_len = 2 if env.add_action_to_obs else 0
    mask_len = 0

    # Ensure dtype float32 and on target device
    x = obs_seq_batch.to(device=device, dtype=torch.float32)

    slot_feats = x[:, :, :slots_len].reshape(B, T, S, N)

    if action_len > 0:
        action_tail = x[:, :, slots_len:slots_len + action_len]
    else:
        action_tail = torch.zeros(
            (B, T, 2), dtype=torch.float32, device=device)

    # Derive slot mask from 'valid' field if available, else all ones
    if env._valid_field_index is not None:
        slot_mask = (slot_feats[..., env._valid_field_index] > 0.5).to(
            torch.float32)
    else:
        slot_mask = torch.ones(
            (B, T, S), dtype=torch.float32, device=device)

    self_feats = action_tail if use_prev_action_as_self else torch.zeros(
        (B, T, 2), dtype=torch.float32, device=device)

    return slot_feats, slot_mask, self_feats
