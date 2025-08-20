
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuccessPredictor(nn.Module):
    """
    Predicts the probability that the *next* stop will be inside the reward boundary
    if we pursue a specific target (either retrying the current target or switching to
    a specific other firefly).

    Inputs are per-option "attempt" features known *before* attempting the stop:
      - geometry (distance to boundary, bearing alignment, etc.)
      - recent sensory evidence (last-flash Δt / intensity)
      - motor state at plan time (speed, angular vel)
      - for retry: miss vector from last attempt, #prior attempts on same target, etc.

    Args:
      in_dim:  D_succ, dimensionality of success-prediction features
      hidden:  width of the small MLP
      p_drop:  dropout probability
    """
    def __init__(self, in_dim: int, hidden: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 1)  # scalar logit → sigmoid → probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D_succ)  # N = number of (decision, option) pairs in a flat batch
        returns:
          p_succ: (N,) in [0,1]  # predicted success probability
        """
        return torch.sigmoid(self.net(x)).squeeze(-1)
