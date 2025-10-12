# NOTE: Single quotes are used for all string literals to match your style.

import math
import torch
import torch.nn as nn

from machine_learning.RL.ff_attention import attn_sac_ff, attn_sac_rnn, env_attn_multiff

# ================================================================
# Core Attention Blocks (MAB / SAB / ISAB / PMA)
# ------------------------------------------------
# These are written to be drop-in friendly for your MultiFF pipeline.
# Shapes are explicitly annotated; masks follow PyTorch MHA convention:
#   key_padding_mask: Bool[batch, Tk] where True = IGNORE that key position.
# ================================================================


class MAB(nn.Module):
    """
    Multihead-Attention Block (MAB)
    - Generic block: Y = LN( X + MHA(Wq*Q, Wk*K, Wv*K) ); Y = LN( Y + FF(Y) )
    - Unlike vanilla nn.MultiheadAttention, we project Q/K/V to 'dim_out' ourselves,
      so dim_q and dim_kv can differ from dim_out (useful for adapters).
    """

    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4, dropout_p=0.0):
        super().__init__()
        # MultiheadAttention expects [B, T, C] when batch_first=True.
        self.mha = nn.MultiheadAttention(
            embed_dim=dim_out,            # projected feature dim seen by attention
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_p
        )
        # Linear projections for Q, K, V into the shared dim_out space.
        self.w_q = nn.Linear(dim_q, dim_out)
        self.w_k = nn.Linear(dim_kv, dim_out)
        self.w_v = nn.Linear(dim_kv, dim_out)

        # Pre- and post-FF layer norms (Transformer-style residual block).
        self.ln1 = nn.LayerNorm(dim_out)
        self.ff = nn.Sequential(
            nn.Linear(dim_out, dim_out * 2),
            nn.ReLU(),
            nn.Linear(dim_out * 2, dim_out)
        )
        self.ln2 = nn.LayerNorm(dim_out)

    def forward(self, Q, K, key_padding_mask=None, need_weights=False, average_attn_weights=False):
        """
        Args:
            Q: FloatTensor [B, Tq, dq]  - query sequence (e.g., seeds or elements)
            K: FloatTensor [B, Tk, dk]  - key/value sequence (e.g., elements)
            key_padding_mask: BoolTensor [B, Tk] where True = IGNORE that key pos
            need_weights: return attention maps if True
            average_attn_weights:
                - True  -> returns [B, Tq, Tk] averaged over heads
                - False -> returns [B, num_heads, Tq, Tk]

        Returns:
            y: FloatTensor [B, Tq, dim_out] - residual + FF output
            attn_w: attention weights per PyTorch convention (shape depends on flags)
        """
        # Project inputs to attention space.
        q = self.w_q(Q)  # [B, Tq, dim_out]
        k = self.w_k(K)  # [B, Tk, dim_out]
        v = self.w_v(K)  # [B, Tk, dim_out]

        # Multi-head attention; mask marks keys (elements) to ignore.
        attn_out, attn_w = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,   # True = ignore/pad
            need_weights=need_weights,
            average_attn_weights=average_attn_weights
        )  # attn_out: [B, Tq, dim_out]

        # Residual + norm (pre-FF)
        x = self.ln1(Q + attn_out)  # same Tq as Q

        # Position-wise feed-forward + residual + norm (The feed-forward network (FFN) is applied independently to each element)
        x2 = self.ff(x)
        y = self.ln2(x + x2)

        return y, attn_w


class SAB(nn.Module):
    """
    Self-Attention Block (SAB): MAB(X, X).
    Permutation-equivariant across elements until you pool.
    """

    def __init__(self, dim, num_heads=4, dropout_p=0.0):
        super().__init__()
        self.mab = MAB(dim_q=dim, dim_kv=dim, dim_out=dim, num_heads=num_heads, dropout_p=dropout_p)

    def forward(self, X, key_padding_mask=None):
        """
        Args:
            X: FloatTensor [B, N, d]         - set of N elements
            key_padding_mask: Bool [B, N]    - True where element is INVALID
        Returns:
            Y: FloatTensor [B, N, d]         - contextualized elements
        """
        Y, _ = self.mab(X, X, key_padding_mask=key_padding_mask, need_weights=False)
        return Y


class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB)
    - Two MABs with M 'inducing' points I (trainable) to reduce O(N^2) -> O(N*M).
      H = MAB(I, X)   # gather from elements into inducing summaries
      Y = MAB(X, H)   # broadcast summaries back to elements
    - Keeps permutation equivariance and scales better for large sets.
    """

    def __init__(self, dim, num_heads=4, num_inducing=16, dropout_p=0.0):
        super().__init__()
        # Trainable inducing points shared across the batch; small init for stability.
        self.I = nn.Parameter(torch.randn(num_inducing, dim) * 0.02)
        self.mab1 = MAB(dim_q=dim, dim_kv=dim, dim_out=dim, num_heads=num_heads, dropout_p=dropout_p)
        self.mab2 = MAB(dim_q=dim, dim_kv=dim, dim_out=dim, num_heads=num_heads, dropout_p=dropout_p)

    def forward(self, X, key_padding_mask=None):
        """
        Args:
            X: FloatTensor [B, N, d]         - set of N elements
            key_padding_mask: Bool [B, N]    - True where element is INVALID
        Returns:
            Y: FloatTensor [B, N, d]         - elements refined by global summaries
        """
        B = X.size(0)

        # Tile inducing points per batch: [B, M, d]
        I = self.I.unsqueeze(0).expand(B, -1, -1)

        # 1) Gather: inducing queries attend over elements; mask ignores invalid keys
        H, _ = self.mab1(I, X, key_padding_mask=key_padding_mask, need_weights=False)  # [B, M, d]

        # 2) Broadcast: elements query the inducing summaries; no padding on H
        Y, _ = self.mab2(X, H, key_padding_mask=None, need_weights=False)              # [B, N, d]
        return Y


class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA)
    - K learnable 'seed' queries attend over the set to produce K summaries.
    - This is a learned pooling operation; replacing hand-coded top-k selection.
    """

    def __init__(self, dim, num_heads=4, num_seeds=4, dropout_p=0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, dim) * 0.02)  # K seeds shared across batch
        self.mab = MAB(dim_q=dim, dim_kv=dim, dim_out=dim, num_heads=num_heads, dropout_p=dropout_p)

    def forward(self, X, key_padding_mask=None, return_attn=False):
        """
        Args:
            X: FloatTensor [B, N, d]          - set of N elements
            key_padding_mask: Bool [B, N]     - True where element is INVALID
            return_attn: if True, also return raw attention weights
        Returns:
            Z: FloatTensor [B, K, d]          - K pooled summaries
            attn_w (optional): FloatTensor [B, H, K, N] - per-head/seed weights
        """
        B = X.size(0)
        # Tile seeds per batch: [B, K, d]
        S = self.S.unsqueeze(0).expand(B, -1, -1)

        # Seeds (queries) attend to elements (keys/values).
        # We request per-head weights to visualize per-element contribution.
        Z, attn_w = self.mab(
            S, X,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # -> [B, H, K, N]
        )

        if return_attn:
            return Z, attn_w
        return Z

# ================================================================
# Set Transformer variants (stack of SAB or ISAB) + PMA pooling
# - Returns same tuple as FF encoder: (sel, ctx, attnW)
#   * sel: K pooled summaries [B, K, d_slot]
#   * ctx: mean of K summaries [B, d_slot] if include_ctx=True
#   * attnW: per-element weights [B, S] by averaging over heads & seeds
# ================================================================

class SetTransformerSABEncoder(nn.Module):
    def __init__(self, d_in, d_slot=64, k=4, num_heads=4, num_sab=2,
                 include_ctx=True, dropout_p=0.0):
        super().__init__()
        self.k = k
        self.d_slot = d_slot
        self.include_ctx = include_ctx

        # Pre-embed slots to d_slot (shared over elements).
        self.pre = nn.Sequential(
            nn.Linear(d_in, d_slot),
            nn.ReLU(),
            nn.Linear(d_slot, d_slot)
        )
        # A small stack of SAB layers gives element-to-element interaction.
        self.sabs = nn.ModuleList([
            SAB(d_slot, num_heads=num_heads, dropout_p=dropout_p) for _ in range(num_sab)
        ])
        # PMA turns the set into K summaries (learned pooling).
        self.pma = PMA(d_slot, num_heads=num_heads, num_seeds=k, dropout_p=dropout_p)

    def forward(self, slot_feats, slot_mask):
        """
        Args:
            slot_feats: FloatTensor [B, S, d_in]
            slot_mask:  Bool/Float   [B, S] (True/1 = valid)
        Returns:
            Z:     [B, K, d_slot]        - K pooled summaries
            ctx:   [B, d_slot] or None   - mean of summaries
            attnW: [B, S]                - per-element importance (avg over seeds & heads)
        """
        # PyTorch MHA mask: True means 'ignore'; convert from valid mask accordingly.
        pad_mask = ~(slot_mask > 0.5)  # [B, S], True = INVALID/ignore

        # Per-slot embedding
        H = self.pre(slot_feats)  # [B, S, d_slot]

        # Self-attention across elements (captures relations like crowding/cluster)
        for sab in self.sabs:
            H = sab(H, key_padding_mask=pad_mask)  # [B, S, d_slot]

        # PMA pooling to K summaries + raw attention weights per head/seed.
        Z, attn_w = self.pma(H, key_padding_mask=pad_mask, return_attn=True)  # Z:[B,K,d], attn_w:[B,H,K,S]

        # Reduce attention over heads and seeds to a single [B, S] map (for logging).
        attnW = attn_w.mean(dim=1).mean(dim=1)  # [B, S]

        # Optional context vector: mean over K summaries (keeps interface consistent with FF).
        ctx = Z.mean(dim=1) if self.include_ctx else None

        return Z, ctx, attnW


class SetTransformerISABEncoder(nn.Module):
    def __init__(self, d_in, d_slot=64, k=4, num_heads=4, num_isab=2, num_inducing=16,
                 include_ctx=True, dropout_p=0.0):
        super().__init__()
        self.k = k
        self.d_slot = d_slot
        self.include_ctx = include_ctx

        self.pre = nn.Sequential(
            nn.Linear(d_in, d_slot),
            nn.ReLU(),
            nn.Linear(d_slot, d_slot)
        )
        # ISAB stack: scalable relational encoding with M inducing points.
        self.isabs = nn.ModuleList([
            ISAB(d_slot, num_heads=num_heads, num_inducing=num_inducing, dropout_p=dropout_p)
            for _ in range(num_isab)
        ])
        self.pma = PMA(d_slot, num_heads=num_heads, num_seeds=k, dropout_p=dropout_p)

    def forward(self, slot_feats, slot_mask):
        """
        Args mirror SAB encoder; ISAB is preferred when S is large (O(S*M)).
        """
        pad_mask = ~(slot_mask > 0.5)  # [B, S], True = INVALID/ignore
        H = self.pre(slot_feats)       # [B, S, d_slot]

        # Induced attention layers (gather -> broadcast) repeated num_isab times.
        for isab in self.isabs:
            H = isab(H, key_padding_mask=pad_mask)  # [B, S, d_slot]

        Z, attn_w = self.pma(H, key_padding_mask=pad_mask, return_attn=True)  # [B,K,d], [B,H,K,S]
        attnW = attn_w.mean(dim=1).mean(dim=1)  # [B, S]
        ctx = Z.mean(dim=1) if self.include_ctx else None
        return Z, ctx, attnW


# ================================================================
# Factory / Selector
# - Exposes a single constructor to swap between:
#   'ff'  : original top-k slot encoder
#   'sab' : full self-attention stack + PMA (O(S^2))
#   'isab': induced self-attention stack + PMA (O(S*M))
# ================================================================

def make_set_encoder(
    method: str,
    *,
    d_in: int,
    d_slot: int = 64,
    k: int = 4,
    include_ctx: bool = True,
    # FF params
    n_heads_ff: int = 2,
    temperature: float = 1.0,
    slot_dropout_p: float = 0.0,
    gumbel_topk: bool = False,
    # Set-Transformer params
    num_heads: int = 4,
    dropout_p: float = 0.0,
    num_sab: int = 2,
    num_isab: int = 2,
    num_inducing: int = 16,
):
    """
    Returns:
        nn.Module with forward(slot_feats, slot_mask) -> (sel[K,d_slot], ctx[d_slot]|None, attnW[S])
    """
    method = method.lower()

    if method == 'ff':
        # Original light-weight, permutation-invariant selector.
        return attn_sac_ff.FFSlotAttentionEncoder(
            d_in=d_in, d_slot=d_slot, n_heads=n_heads_ff, k=k,
            use_soft_ctx=include_ctx, temperature=temperature,
            slot_dropout_p=slot_dropout_p, gumbel_topk=gumbel_topk
        )

    if method == 'sab':
        # Full self-attention across all elements (best for smaller S).
        return SetTransformerSABEncoder(
            d_in=d_in, d_slot=d_slot, k=k, num_heads=num_heads,
            num_sab=num_sab, include_ctx=include_ctx, dropout_p=dropout_p
        )

    if method == 'isab':
        # Scalable version with inducing points (preferred when S is large).
        return SetTransformerISABEncoder(
            d_in=d_in, d_slot=d_slot, k=k, num_heads=num_heads,
            num_isab=num_isab, num_inducing=num_inducing,
            include_ctx=include_ctx, dropout_p=dropout_p
        )

    raise ValueError('method must be one of: ff | sab | isab')
