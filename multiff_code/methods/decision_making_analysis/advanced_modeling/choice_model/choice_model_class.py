from typing import Optional

import torch
import torch.nn as nn


class ContextAwareChoice(nn.Module):
    """
    Scores each option at a decision point using:
      + β_R * p_succ        (value for likely success soon)
      - β_T * time_cost     (travel/adjust/stop time cost)
      - β_A * attempts      (diminishing returns / frustration)
      + set_summary         (context: crowding/reinforcement/competition)
    and then softmaxes over {retry} ∪ {candidates}.

    Conventions:
      - B = batch size (number of decision points in a minibatch)
      - K = K_max within the batch (max number of visible candidates in that batch)
      - E = dim_ego         (ego/context features shared by options)
      - R = dim_retry       (retry-specific features)
      - C = dim_cand        (candidate-specific features)
      - H = hidden width after projection

    Indexing:
      - We reserve index 0 for "retry current target".
      - Real candidates live at indices 1..K_b for item b.
      - Padded slots (to reach K_max) are masked out (no prob, no gradient).
    """

    def __init__(self, dim_ego: int, dim_retry: int, dim_cand: int, hidden: int = 128):
        super().__init__()

        # Project raw features to a common hidden space (H)
        self.ego_proj = nn.Linear(dim_ego,  hidden)  # (B, E)    → (B, H)
        self.retry_proj = nn.Linear(dim_retry, hidden)  # (B, R)    → (B, H)
        self.cand_proj = nn.Linear(dim_cand, hidden)   # (B, K, C) → (B, K, H)

        # Final scoring head takes:
        #   [ option_embed (H) , set_summary (H) , 3 scalars (p_succ, -time_cost, -attempts) ] → score
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # scalar score per option
        )

    @staticmethod
    def _set_summary(Hcand: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Permutation-invariant pooled summary over the candidate set.
        This gives the model a sense of the *overall set context*.

        Hcand: (B, K, H)       # candidate embeddings after self.cand_proj
        mask:  (B, K) bool     # True = padded (invalid)
        returns:
          S:   (B, H)          # pooled set summary
        """
        if mask is None:
            return Hcand.mean(dim=1)
        valid = (~mask).float().unsqueeze(-1)           # (B, K, 1)
        num = valid.sum(dim=1).clamp_min(1.0)         # (B, 1)
        return (Hcand * valid).sum(dim=1) / num         # (B, H)

    def forward(
        self,
        ego: torch.Tensor,                # (B, E)     shared ego features
        retry_feat: torch.Tensor,         # (B, R)     retry-only features
        # (B, K, C)  per-candidate features (padded to K_max)
        cand_feat: torch.Tensor,
        # (B,)       scalar cost for retrying (time to reattempt)
        time_cost_retry: torch.Tensor,
        # (B, K)     scalar cost per candidate (time to travel and stop)
        time_cost_cand: torch.Tensor,
        # (B,)       recent attempts counter for the same target
        attempts_retry: torch.Tensor,
        # (B, K)     recent attempts per candidate (often zeros)
        attempts_cand: torch.Tensor,
        # (B,)       predicted success prob for retry (from SuccessPredictor)
        p_succ_retry: torch.Tensor,
        # (B, K)     predicted success prob per candidate
        p_succ_cand: torch.Tensor,
        # (B, K) bool; True = padded/invalid candidate slot
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores: (B, K+1)  # unnormalized scores: column 0=retry, columns 1..K=candidates
          probs:  (B, K+1)  # softmax over columns
        """

        # (batch_size, batch_max_candidates, dim_cand)
        B, K, _ = cand_feat.shape

        # --- Embed ego, retry, and candidate features into hidden space (H) ---
        E = torch.relu(self.ego_proj(ego))              # (B, H)
        R = torch.relu(self.retry_proj(retry_feat))     # (B, H)
        C = torch.relu(self.cand_proj(cand_feat))       # (B, K, H)

        # --- Pool a set summary over current candidates (excludes retry by design) ---
        S = self._set_summary(C, mask)                  # (B, H)

        # --- Build the retry option (index 0) ---
        # Scalars per decision for retry:
        #   p_succ_retry: higher is better
        #   -time_cost_retry: negative cost → benefit
        #   -attempts_retry: more attempts → lower utility (diminishing returns)
        scalars_retry = torch.stack([
            p_succ_retry,                 # (B,)
            -time_cost_retry,             # (B,)
            -attempts_retry.float()       # (B,)
        ], dim=-1)                        # (B, 3)

        # Concatenate retry embedding + set summary + scalars → feed head
        opt0 = torch.cat([R, S], dim=-1)                 # (B, 2H)
        opt0 = torch.cat([opt0, scalars_retry], dim=-1)  # (B, 2H+3)
        score0 = self.head(opt0).squeeze(-1)             # (B,)

        # --- Build candidate options (indices 1..K) ---
        # Repeat set summary across K rows
        S_rep = S.unsqueeze(1).expand(B, K, -1)          # (B, K, H)

        scalars_cand = torch.stack([
            p_succ_cand,                 # (B, K)
            -time_cost_cand,             # (B, K)
            -attempts_cand.float()       # (B, K)
        ], dim=-1)                       # (B, K, 3)

        optK = torch.cat([C, S_rep], dim=-1)             # (B, K, 2H)
        optK = torch.cat([optK, scalars_cand], dim=-1)   # (B, K, 2H+3)
        scoreK = self.head(optK).squeeze(-1)             # (B, K)

        # --- Concatenate retry (col 0) + candidates (cols 1..K) ---
        scores = torch.cat([score0.unsqueeze(1), scoreK], dim=1)  # (B, K+1)

        # --- Apply mask to padded candidate columns so they get ~zero prob ---
        # pad_mask_full[:, 0] is always False (retry is real); columns 1..K copy 'mask'
        if mask is not None:
            pad_mask_full = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                mask
            ], dim=1)                                        # (B, K+1)
            # large negative avoids NaNs
            scores = scores.masked_fill(pad_mask_full, -1e9)

        probs = F.softmax(scores, dim=1)                     # (B, K+1)
        return scores, probs


def collate_retry_switch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Turns a list of per-example dicts into a padded batch with masks.

    Each example 'ex' should minimally contain:
      ex["ego"]              : (E,)
      ex["retry_feat"]       : (R,)
      ex["cand_feat"]        : (K_b, C)  # K_b can be 0 for this example
      ex["time_cost_retry"]  : ()
      ex["time_cost_cand"]   : (K_b,)
      ex["attempts_retry"]   : ()
      ex["attempts_cand"]    : (K_b,)
      ex["choice_type"]      : "retry" or "switch"
      ex["choice_local_idx"] : int in [0..K_b-1]  # only if choice_type=="switch"

    Returns a dict of tensors with shapes:
      ego              : (B, E)
      retry_feat       : (B, R)
      cand_feat        : (B, K_max, C)
      cand_mask        : (B, K_max)  # True = padded slot (invalid)
      time_cost_retry  : (B,)
      time_cost_cand   : (B, K_max)
      attempts_retry   : (B,)
      attempts_cand    : (B, K_max)
      chosen_idx       : (B,)   # 0 = retry, 1..K_b = candidates (after padding convention)
    """
    import torch

    B = len(batch)
    K_list = [ex["cand_feat"].shape[0] for ex in batch]  # real K per item
    K_max = max(K_list)                                 # batch max K

    # Infer dimensions
    E = batch[0]["ego"].shape[-1]
    R = batch[0]["retry_feat"].shape[-1]
    # Handle the (rare) case K_max == 0 by peeking at a template dim if you store one
    C = batch[0]["cand_feat"].shape[-1] if K_max > 0 else batch[0]["cand_feat_template_dim"]

    # Allocate tensors on CPU for now (move to device later)
    ego = torch.zeros(B, E)
    retry_feat = torch.zeros(B, R)
    cand_feat = torch.zeros(B, K_max, C) if K_max > 0 else torch.zeros(B, 0, C)
    cand_mask = torch.ones(B, K_max, dtype=torch.bool)  # start as all PAD=True
    time_cost_retry = torch.zeros(B)
    time_cost_cand = torch.zeros(B, K_max) if K_max > 0 else torch.zeros(B, 0)
    attempts_retry = torch.zeros(B, dtype=torch.long)
    attempts_cand = torch.zeros(
        B, K_max, dtype=torch.long) if K_max > 0 else torch.zeros(B, 0, dtype=torch.long)
    chosen_idx = torch.zeros(B, dtype=torch.long)

    for b, ex in enumerate(batch):
        # Copy shared/ retry fields
        ego[b] = torch.as_tensor(ex["ego"])
        retry_feat[b] = torch.as_tensor(ex["retry_feat"])
        time_cost_retry[b] = torch.as_tensor(ex["time_cost_retry"])
        attempts_retry[b] = torch.as_tensor(ex["attempts_retry"])

        # Copy candidate arrays into the front of padded buffers
        Kb = K_list[b]  # real number of candidates in this example
        if Kb > 0:
            cand_feat[b, :Kb] = torch.as_tensor(
                ex["cand_feat"])         # (Kb, C)
            time_cost_cand[b, :Kb] = torch.as_tensor(
                ex["time_cost_cand"])    # (Kb,)
            attempts_cand[b, :Kb] = torch.as_tensor(
                ex["attempts_cand"])     # (Kb,)
            cand_mask[b, :Kb] = False  # mark these as VALID (not padded)

        # Map the behavior choice to the global column index:
        #   0 = retry
        #   1..Kb = candidate j_local in this example
        if ex["choice_type"] == "retry":
            chosen_idx[b] = 0
        else:
            j_local = int(ex["choice_local_idx"])  # 0..Kb-1
            chosen_idx[b] = 1 + j_local

    return {
        "ego": ego, "retry_feat": retry_feat,
        "cand_feat": cand_feat, "cand_mask": cand_mask,
        "time_cost_retry": time_cost_retry, "time_cost_cand": time_cost_cand,
        "attempts_retry": attempts_retry, "attempts_cand": attempts_cand,
        "chosen_idx": chosen_idx
    }
