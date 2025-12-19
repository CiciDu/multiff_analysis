import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class LFADSDataset(Dataset):
    """
    Simple spike-count dataset for LFADS.

    data : (n_trials, T, n_neurons) spike counts
    """

    def __init__(self, data: torch.Tensor):
        super().__init__()
        # ensure float32
        self.data = data.float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class LFADSModel(nn.Module):
    """
    Minimal LFADS-like model:
      - Encoder GRU -> latent z (per trial)
      - Generator GRU initialized from z
      - Linear to factors -> linear to rates (Poisson)
    """

    def __init__(
        self,
        n_neurons: int,
        factors_dim: int = 20,
        enc_dim: int = 64,
        gen_dim: int = 64,
        z_dim: int = 20,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.factors_dim = factors_dim
        self.enc_dim = enc_dim
        self.gen_dim = gen_dim
        self.z_dim = z_dim

        # Encoder RNN (reads spike counts)
        self.encoder = nn.GRU(
            input_size=n_neurons,
            hidden_size=enc_dim,
            batch_first=True,
        )
        self.enc_to_mu = nn.Linear(enc_dim, z_dim)
        self.enc_to_logvar = nn.Linear(enc_dim, z_dim)

        # Generator RNN
        self.z_to_gen_init = nn.Linear(z_dim, gen_dim)
        self.generator = nn.GRU(
            input_size=0,   # no external input, autonomous generator
            hidden_size=gen_dim,
            batch_first=True,
        )

        # Readouts
        self.gen_to_factors = nn.Linear(gen_dim, factors_dim)
        self.factors_to_rates = nn.Linear(factors_dim, n_neurons)

        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, T, n_neurons) spike counts
        Returns mu, logvar : (batch, z_dim)
        """
        # Encode time sequence; use last hidden state
        _, h_last = self.encoder(x)   # h_last: (1, batch, enc_dim)
        h_last = h_last[0]
        mu = self.enc_to_mu(h_last)
        logvar = self.enc_to_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict:
        """
        x : (batch, T, n_neurons) spike counts
        Returns dict with:
          - 'rates'   : (batch, T, n_neurons)
          - 'factors' : (batch, T, factors_dim)
          - 'kld'     : scalar KL divergence term
          - 'recon'   : scalar reconstruction loss
        """
        batch_size, T, _ = x.shape

        # --- Encoder → latent z ---
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # --- Generator initial state ---
        h0 = torch.tanh(self.z_to_gen_init(z))          # (batch, gen_dim)
        h0 = h0.unsqueeze(0)                            # (1, batch, gen_dim)

        # Generator input: no external input, so use dummy zeros
        gen_input = x.new_zeros((batch_size, T, 0))     # (batch, T, 0)

        gen_output, _ = self.generator(gen_input, h0)   # (batch, T, gen_dim)

        # Factors and rates
        factors = self.gen_to_factors(gen_output)               # (batch, T, factors_dim)
        rates = self.softplus(self.factors_to_rates(factors))   # positive rates

        # Poisson negative log-likelihood
        # x: spike counts, rates: Hz, but we treat as rate in counts/bin
        eps = 1e-6
        recon = (rates - x * torch.log(rates + eps)).sum(dim=(1, 2)).mean()

        # KL divergence term (per trial) for z
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return {
            'rates': rates,
            'factors': factors,
            'kld': kld,
            'recon': recon,
        }


class LFADSControllerModel(nn.Module):
    """
    LFADS with an input-only controller:
      - Encoder GRU -> latent z (per trial)
      - Controller GRU reads spikes and produces time-varying inputs u_t
      - Generator GRU is initialized from z and driven by u_t
      - Linear to factors -> linear to rates (Poisson)

    Compared to LFADSModel, this adds:
      - controller RNN
      - controller_to_u linear layer
      - generator.input_size = u_dim (instead of 0)
    """

    def __init__(
        self,
        n_neurons: int,
        factors_dim: int = 20,
        enc_dim: int = 64,
        gen_dim: int = 64,
        z_dim: int = 20,
        controller_dim: int = 64,
        u_dim: int = 4,
        u_l2_weight: float = 0.0,
    ):
        """
        Parameters
        ----------
        n_neurons : int
            Number of observed neurons.
        factors_dim : int
            Dimensionality of latent factors.
        enc_dim : int
            Encoder GRU hidden size.
        gen_dim : int
            Generator GRU hidden size.
        z_dim : int
            Trial-level latent dimensionality.
        controller_dim : int
            Controller GRU hidden size.
        u_dim : int
            Dimensionality of inferred inputs u_t.
        u_l2_weight : float
            Optional L2 penalty weight on u_t to keep inputs small.
            This is added to the KL term.
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.factors_dim = factors_dim
        self.enc_dim = enc_dim
        self.gen_dim = gen_dim
        self.z_dim = z_dim
        self.controller_dim = controller_dim
        self.u_dim = u_dim
        self.u_l2_weight = u_l2_weight

        # Encoder RNN (reads spike counts) → z
        self.encoder = nn.GRU(
            input_size=n_neurons,
            hidden_size=enc_dim,
            batch_first=True,
        )
        self.enc_to_mu = nn.Linear(enc_dim, z_dim)
        self.enc_to_logvar = nn.Linear(enc_dim, z_dim)

        # Controller RNN (reads spike counts) → u_t
        self.controller = nn.GRU(
            input_size=n_neurons,
            hidden_size=controller_dim,
            batch_first=True,
        )
        self.controller_to_u = nn.Linear(controller_dim, u_dim)

        # Generator RNN (driven by u_t)
        self.z_to_gen_init = nn.Linear(z_dim, gen_dim)
        self.generator = nn.GRU(
            input_size=u_dim,     # now driven by inputs
            hidden_size=gen_dim,
            batch_first=True,
        )

        # Readouts
        self.gen_to_factors = nn.Linear(gen_dim, factors_dim)
        self.factors_to_rates = nn.Linear(factors_dim, n_neurons)

        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, T, n_neurons) spike counts
        Returns mu, logvar : (batch, z_dim)
        """
        _, h_last = self.encoder(x)   # h_last: (1, batch, enc_dim)
        h_last = h_last[0]
        mu = self.enc_to_mu(h_last)
        logvar = self.enc_to_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict:
        """
        x : (batch, T, n_neurons) spike counts

        Returns dict with:
          - 'rates'   : (batch, T, n_neurons)
          - 'factors' : (batch, T, factors_dim)
          - 'kld'     : scalar KL + optional u L2 penalty
          - 'recon'   : scalar reconstruction loss
          - 'u'       : (batch, T, u_dim) inferred inputs
          - 'z'       : (batch, z_dim) trial-level latents
        """
        batch_size, T, _ = x.shape

        # --- Encoder → latent z (per trial) ---
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # --- Controller → time-varying inputs u_t ---
        # Controller reads spikes; you can later swap x for factors if desired
        controller_output, _ = self.controller(x)         # (batch, T, controller_dim)
        u = self.controller_to_u(controller_output)       # (batch, T, u_dim)

        # --- Generator initial state from z ---
        h0 = torch.tanh(self.z_to_gen_init(z))            # (batch, gen_dim)
        h0 = h0.unsqueeze(0)                              # (1, batch, gen_dim)

        # --- Generator driven by u_t ---
        gen_output, _ = self.generator(u, h0)             # (batch, T, gen_dim)

        # --- Factors and rates ---
        factors = self.gen_to_factors(gen_output)         # (batch, T, factors_dim)
        rates = self.softplus(self.factors_to_rates(factors))  # (batch, T, n_neurons)

        # --- Poisson negative log-likelihood ---
        eps = 1e-6
        recon = (rates - x * torch.log(rates + eps)).sum(dim=(1, 2)).mean()

        # --- KL for z ---
        kld_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Optional L2 penalty on u_t to discourage huge inputs
        if self.u_l2_weight > 0.0:
            u_l2 = (u.pow(2).sum(dim=(1, 2)) / (T * self.u_dim)).mean()
            kld = kld_z + self.u_l2_weight * u_l2
        else:
            kld = kld_z

        return {
            'rates': rates,
            'factors': factors,
            'kld': kld,
            'recon': recon,
            'u': u,
            'z': z,
        }


def lfads_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 1.0,
) -> Dict:
    model.train()
    total_recon = 0.0
    total_kld = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch)
        recon = out['recon']
        kld = out['kld']
        loss = recon + kl_weight * kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += float(recon.item())
        total_kld += float(kld.item())
        total_loss += float(loss.item())
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kld': total_kld / n_batches,
    }


def lfads_eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    kl_weight: float = 1.0,
) -> Dict:
    model.eval()
    total_recon = 0.0
    total_kld = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            recon = out['recon']
            kld = out['kld']
            loss = recon + kl_weight * kld

            total_recon += float(recon.item())
            total_kld += float(kld.item())
            total_loss += float(loss.item())
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kld': total_kld / n_batches,
    }
