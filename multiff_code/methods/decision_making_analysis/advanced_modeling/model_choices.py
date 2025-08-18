"""
Tiny PyTorch skeleton for MultiFF retry/switch modeling.

Implements the two core training steps and leaves hooks for the optional
variants. The code is intentionally lightweight: small MLPs, masking for
variable option counts, and minimal training loops.

You can paste this into a file and adapt the Dataset stubs to your
preprocessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Utilities
# -------------------------------

def mlp(in_dim: int, hidden: List[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]  # keep it simple
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy for variable option sets.
    logits: [B, Kmax]
    targets: [B] (index of chosen option in 0..K-1)
    mask: [B, Kmax] boolean (True for valid options)
    """
    # Put -inf on invalid options so softmax ignores them
    logits_masked = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(logits_masked, targets)


# -------------------------------
# 3A) Capture-success predictor
# -------------------------------

class SuccessPredictor(nn.Module):
    """
    Predicts p_succ = P(next stop is inside boundary for that target).

    Inputs: stop_features: [B, D_s]
    Output: p_succ: [B]
    """
    def __init__(self, in_dim: int, hidden: List[int] = [32]):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1, dropout=0.0)

    def forward(self, stop_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(stop_features)).squeeze(-1)


# -------------------------------
# 3B) Choice scorer over {retry} ∪ {other targets}
# -------------------------------

class ChoiceScorer(nn.Module):
    """
    Scores each candidate option with a utility U_i(t) and produces logits.

    Forward inputs:
      - option_features: [B, Kmax, D_o]
      - mask: [B, Kmax] boolean (True for valid candidates)
      - p_succ_opt: Optional[Tensor] [B, Kmax] (per-option success proba)
      - extra_costs: Optional[Tensor] [B, Kmax, D_c] (e.g., time-to-go, turn cost)

    Returns:
      - logits: [B, Kmax]
    """
    def __init__(self, option_dim: int, extra_cost_dim: int = 0, use_psucc: bool = True,
                 hidden: List[int] = [64, 32]):
        super().__init__()
        self.use_psucc = use_psucc
        in_dim = option_dim + extra_cost_dim + (1 if use_psucc else 0)
        self.net = mlp(in_dim, hidden, 1, dropout=0.0)

    def forward(
        self,
        option_features: torch.Tensor,
        mask: torch.Tensor,
        p_succ_opt: Optional[torch.Tensor] = None,
        extra_costs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Kmax, D_o = option_features.shape
        feats = [option_features]
        if self.use_psucc:
            if p_succ_opt is None:
                raise ValueError("p_succ_opt must be provided when use_psucc=True")
            feats.append(p_succ_opt.unsqueeze(-1))  # [B, Kmax, 1]
        if extra_costs is not None:
            feats.append(extra_costs)
        X = torch.cat(feats, dim=-1)  # [B, Kmax, Din]
        logits = self.net(X).squeeze(-1)  # [B, Kmax]
        # Mask invalid options with -inf to avoid accidental selection
        logits = logits.masked_fill(~mask, -1e9)
        return logits


# -------------------------------
# Optional variants (4)
# -------------------------------

class RetrySwitchHead(nn.Module):
    """Binary logit: retry vs switch after a near-miss."""
    def __init__(self, in_dim: int, hidden: List[int] = [32]):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # returns P(retry)
        return torch.sigmoid(self.net(features)).squeeze(-1)


class TwoStagePolicy(nn.Module):
    """
    Stage 1: binary retry vs switch (on near-miss features).
    Stage 2: if switch, choose among other targets using ChoiceScorer.
    """
    def __init__(self, retry_in_dim: int, option_dim: int, extra_cost_dim: int = 0, use_psucc: bool = True):
        super().__init__()
        self.retry_head = RetrySwitchHead(retry_in_dim)
        self.choice = ChoiceScorer(option_dim, extra_cost_dim, use_psucc)

    def forward(self, retry_features: torch.Tensor,
                option_features: torch.Tensor, mask: torch.Tensor,
                p_succ_opt: Optional[torch.Tensor] = None,
                extra_costs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p_retry = self.retry_head(retry_features)
        logits_switch = self.choice(option_features, mask, p_succ_opt, extra_costs)
        return p_retry, logits_switch


class HazardHead(nn.Module):
    """
    Discrete-time hazard after a near-miss: h_t = P(switch at t | not switched yet).

    Input per step features X_t -> h_t via sigmoid(MLP).
    Loss implemented via discrete-time survival likelihood.
    """
    def __init__(self, in_dim: int, hidden: List[int] = [32]):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)

    def forward(self, step_features: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        """
        step_features: [B, T, D]
        step_mask: [B, T] (True for valid time steps)
        Returns hazards h_t in [0,1]: [B, T]
        """
        h = torch.sigmoid(self.net(step_features)).squeeze(-1)
        return h * step_mask.float()

    @staticmethod
    def survival_nll(h: torch.Tensor, event_index: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood for discrete-time hazards.
        h: [B, T] hazards
        event_index: [B] index of switch time; if censored (no switch in window), set to -1
        step_mask: [B, T]
        """
        # log S_{t} = sum_{k < t} log(1 - h_k);  log f(t) = log S_t + log h_t
        eps = 1e-6
        log1m_h = torch.log(torch.clamp(1 - h, min=eps)) * step_mask
        cumsums = torch.cumsum(log1m_h, dim=1)  # [B, T]
        B, T = h.shape
        nll = []
        for b in range(B):
            t_star = event_index[b].item()
            if t_star >= 0:  # observed switch
                surv = cumsums[b, t_star - 1] if t_star > 0 else torch.tensor(0.0, device=h.device)
                log_h = torch.log(torch.clamp(h[b, t_star], min=eps))
                nll.append(-(surv + log_h))
            else:  # censored at last valid step
                last = int(step_mask[b].nonzero(as_tuple=False)[-1])
                surv = cumsums[b, last]
                nll.append(-surv)
        return torch.stack(nll).mean()


# -------------------------------
# Belief / POMDP-ish helper (very simple)
# -------------------------------

@dataclass
class BeliefState:
    alpha: torch.Tensor  # evidence for success
    beta: torch.Tensor   # evidence for failure

    def p_succ(self) -> torch.Tensor:
        return self.alpha / (self.alpha + self.beta + 1e-9)


def update_belief(
    belief: BeliefState,
    flash_strength: torch.Tensor,
    miss_distance: Optional[torch.Tensor] = None,
    decay: float = 0.95,
) -> BeliefState:
    """
    Tiny heuristic updater: decay old evidence, add flash as positive evidence,
    add miss_distance (scaled) as negative evidence.
    """
    alpha = decay * belief.alpha + flash_strength
    if miss_distance is not None:
        beta = decay * belief.beta + miss_distance
    else:
        beta = decay * belief.beta
    return BeliefState(alpha=alpha, beta=beta)


# -------------------------------
# Datasets (stubs you will replace)
# -------------------------------

class SuccessDataset(Dataset):
    """Each item: (stop_features [D_s], label {0,1})"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 2 and y.ndim == 1
        self.X, self.y = X.float(), y.long()

    def __len__(self) -> int: return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i].float()


class ChoiceDataset(Dataset):
    """
    Each item is a dict with keys:
      - option_features: [K, D_o]
      - option_mask: [K] (bool)
      - chosen_index: int in [0, K-1]
      - p_succ_opt: Optional [K]
      - extra_costs: Optional [K, D_c]
    """
    def __init__(self, items: List[Dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self): return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.items[i]


def choice_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    Kmax = max(item["option_features"].shape[0] for item in batch)
    B = len(batch)
    D_o = batch[0]["option_features"].shape[1]
    option_features = torch.zeros(B, Kmax, D_o)
    mask = torch.zeros(B, Kmax, dtype=torch.bool)
    targets = torch.zeros(B, dtype=torch.long)
    p_succ_opt = None
    extra_costs = None

    has_ps = all("p_succ_opt" in item for item in batch)
    has_ec = all("extra_costs" in item for item in batch)

    if has_ps:
        p_succ_opt = torch.zeros(B, Kmax)
    if has_ec:
        D_c = batch[0]["extra_costs"].shape[1]
        extra_costs = torch.zeros(B, Kmax, D_c)

    for b, item in enumerate(batch):
        K = item["option_features"].shape[0]
        option_features[b, :K] = item["option_features"]
        mask[b, :K] = True
        targets[b] = int(item["chosen_index"])  # ensure within 0..K-1
        if has_ps:
            p_succ_opt[b, :K] = item["p_succ_opt"]
        if has_ec:
            extra_costs[b, :K] = item["extra_costs"]

    out = {"option_features": option_features, "mask": mask, "targets": targets}
    if has_ps:
        out["p_succ_opt"] = p_succ_opt
    if has_ec:
        out["extra_costs"] = extra_costs
    return out


# -------------------------------
# Training loops (minimal)
# -------------------------------

def train_success_head(model: SuccessPredictor, loader: DataLoader, epochs: int = 10, lr: float = 1e-3,
                       device: str = "cpu") -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            p = model(X)
            loss = bce(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * X.size(0)
        print(f"[Success] epoch {ep+1}: loss={total/len(loader.dataset):.4f}")


def train_choice_model(model: ChoiceScorer, loader: DataLoader, epochs: int = 10, lr: float = 1e-3,
                       device: str = "cpu") -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            option_features = batch["option_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            p_succ_opt = batch.get("p_succ_opt")
            extra_costs = batch.get("extra_costs")
            if p_succ_opt is not None: p_succ_opt = p_succ_opt.to(device)
            if extra_costs is not None: extra_costs = extra_costs.to(device)

            logits = model(option_features, mask, p_succ_opt, extra_costs)
            loss = masked_cross_entropy(logits, targets, mask)
            loss.backward(); opt.step()
            total += loss.item() * option_features.size(0)
        print(f"[Choice] epoch {ep+1}: loss={total/len(loader.dataset):.4f}")


# -------------------------------
# Tiny smoke test (random data)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 3A) Success head on random data
    Xs = torch.randn(512, 10)
    ys = torch.bernoulli(torch.full((512,), 0.5))
    ds = SuccessDataset(Xs, ys)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    succ = SuccessPredictor(in_dim=10)
    train_success_head(succ, dl, epochs=2)

    # 3B) Choice model with masking
    items = []
    for _ in range(256):
        K = torch.randint(low=2, high=6, size=(1,)).item()  # 2..5 options
        option_feats = torch.randn(K, 8)
        p_succ_opt = torch.sigmoid(torch.randn(K))
        extra_costs = torch.randn(K, 2)
        chosen = torch.randint(low=0, high=K, size=(1,)).item()
        items.append({
            "option_features": option_feats,
            "p_succ_opt": p_succ_opt,
            "extra_costs": extra_costs,
            "option_mask": torch.ones(K, dtype=torch.bool),
            "chosen_index": torch.tensor(chosen),
        })
    dc = ChoiceDataset(items)
    dlc = DataLoader(dc, batch_size=32, shuffle=True, collate_fn=choice_collate)
    chooser = ChoiceScorer(option_dim=8, extra_cost_dim=2, use_psucc=True)
    train_choice_model(chooser, dlc, epochs=2)

    print("Smoke test complete.")
