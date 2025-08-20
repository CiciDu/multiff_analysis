"""
Retry vs Switch — minimal end-to-end demo
Simulates data with variable #candidates, trains:
  (1) SuccessPredictor: P(next stop succeeds | option features)
  (2) ContextAwareChoice: softmax over {retry} ∪ {candidates} using p_succ + costs

Run:
  pip install torch numpy
  python retry_switch_minimal.py
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Reproducibility
# -------------------------
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)




# =========================
# Synthetic data
# =========================
def make_synthetic_dataset(num_decisions=600, E=4, D=6, K_cap=5) -> dict:
    """
    For each decision:
      ego: (E,)
      retry_feat: (D,)
      cand_feat: (K_b, D)  with variable K_b∈[0..K_cap]
      time_cost_*: retry scalar, per-candidate vector
      attempts_*: small integers
      succ_X/succ_y: one attempt (chosen option) to train SuccessPredictor
      choice: sampled from softmax over utility = 2.3*p_succ - 1.0*time_cost - 0.4*attempts + 0.08*K_b
    """
    rng = np.random.default_rng(SEED+1)

    ego_list, retry_list, cand_list = [], [], []
    tcr_list, tcc_list = [], []
    ar_list, ac_list = [], []
    ytype_list, yj_list = [], []
    succ_X, succ_y = [], []

    # True weights for success probability (shared retry/cand)
    w = rng.normal(0.0, 0.7, size=D)
    w[0] = -2.0  # "distance"
    w[1] = -1.0  # "flash recency"
    w[2] = +1.2  # "alignment"
    def sigmoid(x): return 1.0/(1.0+np.exp(-x))

    for _ in range(num_decisions):
        ego = rng.normal(0.0, 1.0, size=E)
        Kb  = int(np.clip(rng.poisson(2.0), 0, K_cap))

        # Retry features
        r = rng.normal(0.0, 1.0, size=D)
        r[0] = np.abs(rng.normal(0.6, 0.4))   # miss distance
        r[1] = np.abs(rng.normal(0.8, 0.6))   # flash recency
        r[2] = rng.normal(0.2, 0.7)           # alignment proxy

        # Candidate features
        C = rng.normal(0.0, 1.0, size=(Kb, D)) if Kb>0 else np.zeros((0, D))
        if Kb>0:
            C[:,0] = np.abs(rng.normal(1.0, 0.6, size=Kb))  # distance
            C[:,1] = np.abs(rng.normal(1.0, 0.8, size=Kb))  # flash recency
            C[:,2] = rng.normal(0.0, 0.9, size=Kb)          # alignment

        # Costs & attempts
        tcr = 0.8*r[0] + 0.2*np.abs(rng.normal(0.0,0.5))
        tcc = 1.0*C[:,0] + (0.2*np.abs(rng.normal(0.0,0.5,size=Kb)) if Kb>0 else 0.0)
        ar  = int(np.clip(rng.poisson(0.6), 0, 3))
        ac  = (rng.poisson(0.2, size=Kb).clip(0,2) if Kb>0 else np.zeros((0,), dtype=int))

        # “True” success probs → utilities → choice
        pr = sigmoid(r @ w)
        pc = sigmoid(C @ w) if Kb>0 else np.zeros((0,))
        U0 = 2.3*pr - 1.0*tcr - 0.4*ar
        UK = 2.3*pc - 1.0*tcc - 0.4*ac + (0.08*Kb if Kb>0 else 0.0)
        logits = np.concatenate([[U0], UK]); logits -= logits.max()
        p = np.exp(logits)/np.exp(logits).sum()
        choice = rng.choice(len(p), p=p)  # 0=retry, 1..Kb=candidate+1

        # Attempt log (chosen option only)
        if choice == 0:
            succ_X.append(r.copy()); succ_y.append(rng.binomial(1, pr))
            ytype, yj = "retry", None
        else:
            j = choice-1
            succ_X.append(C[j].copy()); succ_y.append(rng.binomial(1, pc[j]))
            ytype, yj = "switch", int(j)

        # Save
        ego_list.append(ego.astype(np.float32))
        retry_list.append(r.astype(np.float32))
        cand_list.append(C.astype(np.float32))
        tcr_list.append(np.float32(tcr))
        tcc_list.append(tcc.astype(np.float32) if Kb>0 else np.zeros((0,),dtype=np.float32))
        ar_list.append(np.int64(ar))
        ac_list.append(ac.astype(np.int64))
        ytype_list.append(ytype)
        yj_list.append(yj)

    return {
        "ego": ego_list, "retry": retry_list, "cand": cand_list,
        "tcr": tcr_list, "tcc": tcc_list, "ar": ar_list, "ac": ac_list,
        "ytype": ytype_list, "yj": yj_list,
        "succX": np.array(succ_X, dtype=np.float32),
        "succY": np.array(succ_y, dtype=np.int64),
        "D": int(C.shape[1] if len(cand_list) and cand_list[-1].size else 6),
        "E": int(ego_list[-1].shape[0]) if ego_list else 4,
    }


# =========================
# Dataset & collate
# =========================
class DecisionDS(Dataset):
    def __init__(self, D, idxs): self.D, self.idxs = D, idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        b = self.idxs[i]
        return {
            "ego":   self.D["ego"][b],
            "retry": self.D["retry"][b],
            "cand":  self.D["cand"][b],         # (K_b, D) or (0,D)
            "tcr":   self.D["tcr"][b],
            "tcc":   self.D["tcc"][b],          # (K_b,)
            "ar":    self.D["ar"][b],
            "ac":    self.D["ac"][b],           # (K_b,)
            "ytype": self.D["ytype"][b],        # "retry" or "switch"
            "yj":    self.D["yj"][b],           # None or 0..K_b-1
            "Ddim":  self.D["D"],
        }

def collate(batch):
    B = len(batch)
    Kmax = max(x["cand"].shape[0] for x in batch)
    E = len(batch[0]["ego"]); Ddim = batch[0]["Ddim"]

    ego   = torch.zeros(B, E)
    retry = torch.zeros(B, Ddim)
    cand  = torch.zeros(B, Kmax, Ddim) if Kmax>0 else torch.zeros(B, 0, Ddim)
    mask  = torch.ones(B, Kmax, dtype=torch.bool)
    tcr   = torch.zeros(B); tcc = torch.zeros(B, Kmax) if Kmax>0 else torch.zeros(B, 0)
    ar    = torch.zeros(B, dtype=torch.long)
    ac    = torch.zeros(B, Kmax, dtype=torch.long) if Kmax>0 else torch.zeros(B, 0, dtype=torch.long)
    yidx  = torch.zeros(B, dtype=torch.long)  # 0=retry, 1..K_b=candidates

    for b, x in enumerate(batch):
        ego[b]   = torch.tensor(x["ego"])
        retry[b] = torch.tensor(x["retry"])
        tcr[b]   = torch.tensor(x["tcr"])
        ar[b]    = torch.tensor(x["ar"])
        Kb = x["cand"].shape[0]
        if Kb>0:
            cand[b,:Kb] = torch.tensor(x["cand"])
            tcc[b,:Kb]  = torch.tensor(x["tcc"])
            ac[b,:Kb]   = torch.tensor(x["ac"])
            mask[b,:Kb] = False
        yidx[b] = 0 if x["ytype"]=="retry" else 1 + int(x["yj"])
    return {"ego":ego, "retry":retry, "cand":cand, "mask":mask,
            "tcr":tcr, "tcc":tcc, "ar":ar, "ac":ac, "yidx":yidx}


