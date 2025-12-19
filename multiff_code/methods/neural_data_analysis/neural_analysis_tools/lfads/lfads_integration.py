import numpy as np
import pandas as pd
from typing import Dict, Optional
import os

import torch
from torch.utils.data import DataLoader

from .lfads_prep import make_continuous_lfads_trials, stitch_lfads_rates_to_continuous
from .lfads_model import LFADSDataset, LFADSModel, lfads_train_epoch, lfads_eval_epoch, LFADSControllerModel


def run_lfads_on_continuous_session(
    spikes_df: pd.DataFrame,
    bin_width_ms: float = 10.0,
    window_len_s: float = 1.0,
    step_s: float = 0.5,
    factors_dim: int = 20,
    enc_dim: int = 64,
    gen_dim: int = 64,
    z_dim: int = 20,
    batch_size: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    device_str: str = 'cuda',
    val_frac: float = 0.1,
    verbose: bool = True,
    use_controller=True,
    out_path: Optional[str] = None,
    load_if_exists: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    High-level LFADS pipeline for continuous data:
      1) Make overlapping pseudo-trials from continuous spikes
      2) Train LFADS model
      3) Predict LFADS rates for all trials
      4) Stitch rates back into continuous FR

    Returns dict with:
      - 'lfads_fr_mat' : (T_full, n_neurons) continuous FR (Hz)
      - 'clusters' : np.ndarray cluster IDs
      - 'bin_width_ms' : float
      - 'start_time' : float
      - 'model_state_dict' : LFADS model weights (for saving)
      - 'trial_info' : dict with trials, trial_start_times

    Saving/loading:
      - If out_path is provided and the file exists and load_if_exists=True,
        results will be loaded and returned without re-running.
      - After computation, results are saved to out_path if provided.
        Use overwrite=True to overwrite an existing file.
    """

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Load cached results if requested and available
    if out_path is not None and load_if_exists and os.path.isfile(out_path):
        if verbose:
            print(f'LFADS: loading results from {out_path}')
        loaded = torch.load(out_path, map_location='cpu')
        return loaded

    # ---------------------------------------------
    # 1. Build LFADS pseudo-trials
    # ---------------------------------------------
    prep = make_continuous_lfads_trials(
        spikes_df=spikes_df,
        bin_width_ms=bin_width_ms,
        window_len_s=window_len_s,
        step_s=step_s,
    )

    trials = prep['trials']                  # (n_trials, T_bins, n_neurons)
    trial_start_times = prep['trial_start_times']
    clusters = prep['clusters']
    start_time = prep['start_time']

    n_trials, T_bins, n_neurons = trials.shape

    if verbose:
        print(
            f'LFADS: {n_trials} pseudo-trials, T={T_bins} bins, N={n_neurons} neurons')

    # ---------------------------------------------
    # 2. Dataset / split
    # ---------------------------------------------
    data_tensor = torch.from_numpy(trials)
    dataset = LFADSDataset(data_tensor)

    n_val = int(round(val_frac * n_trials))
    n_train = n_trials - n_val

    if n_val > 0:
        train_data, val_data = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(123),
        )
        has_val = True
    else:
        train_data = dataset
        val_data = None
        has_val = False

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, drop_last=False) if has_val else None

    # ---------------------------------------------
    # 3. Model & optimizer
    # ---------------------------------------------
    if use_controller:
        model = LFADSControllerModel(
            n_neurons=n_neurons,
            factors_dim=factors_dim,
            enc_dim=enc_dim,
            gen_dim=gen_dim,
            z_dim=z_dim,
            controller_dim=enc_dim,  # often a good default
            u_dim=4,
            u_l2_weight=1e-3,        # small regularization on inputs
        ).to(device)
    else:
        model = LFADSModel(
            n_neurons=n_neurons,
            factors_dim=factors_dim,
            enc_dim=enc_dim,
            gen_dim=gen_dim,
            z_dim=z_dim,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------------------------------------
    # 4. Training loop
    # ---------------------------------------------
    for epoch in range(1, n_epochs + 1):
        train_stats = lfads_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=kl_weight,
        )
        if has_val:
            val_stats = lfads_eval_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                kl_weight=kl_weight,
            )
        else:
            val_stats = None

        if verbose:
            if val_stats is not None:
                print(
                    f'Epoch {epoch:03d} | '
                    f'train loss={train_stats["loss"]:.3f}, '
                    f'recon={train_stats["recon"]:.3f}, '
                    f'kld={train_stats["kld"]:.3f}, '
                    f'val loss={val_stats["loss"]:.3f}'
                )
            else:
                print(
                    f'Epoch {epoch:03d} | '
                    f'train loss={train_stats["loss"]:.3f}, '
                    f'recon={train_stats["recon"]:.3f}, '
                    f'kld={train_stats["kld"]:.3f}'
                )

    # ---------------------------------------------
    # 5. Predict LFADS rates (and factors) for all trials
    # ---------------------------------------------
    model.eval()
    all_rates = []
    all_factors = []

    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            out = model(batch)
            rates = out['rates'].cpu().numpy()      # (batch, T_bins, n_neurons)
            factors = out['factors'].cpu().numpy()  # (batch, T_bins, factors_dim)
            all_rates.append(rates)
            all_factors.append(factors)

    lfads_rates = np.concatenate(all_rates, axis=0)      # (n_trials, T_bins, n_neurons)
    lfads_factors = np.concatenate(all_factors, axis=0)  # (n_trials, T_bins, factors_dim)


    # ---------------------------------------------
    # 6. Stitch back into continuous FR and factors
    # ---------------------------------------------
    lfads_fr_mat = stitch_lfads_rates_to_continuous(
        lfads_rates=lfads_rates,
        trial_start_times=trial_start_times,
        bin_width_ms=bin_width_ms,
        start_time=start_time,
        n_neurons=n_neurons,
    )

    factors_dim = lfads_factors.shape[-1]
    lfads_factors_mat = stitch_lfads_rates_to_continuous(
        lfads_rates=lfads_factors,
        trial_start_times=trial_start_times,
        bin_width_ms=bin_width_ms,
        start_time=start_time,
        n_neurons=factors_dim,
    )

    results = {
        'lfads_fr_mat': lfads_fr_mat,
        'lfads_factors_mat': lfads_factors_mat,
        'clusters': clusters,
        'bin_width_ms': float(bin_width_ms),
        'start_time': float(start_time),
        'model_state_dict': model.state_dict(),
        'trial_info': {
            'trial_start_times': trial_start_times,
            'trials_shape': trials.shape,
            'lfads_rates_trials': lfads_rates,       # optional, trial-wise
            'lfads_factors_trials': lfads_factors,   # optional, trial-wise
        },
    }


    # Save results if requested
    if out_path is not None:
        if os.path.exists(out_path) and not overwrite:
            if verbose:
                print(f'LFADS: not overwriting existing file at {out_path}')
        else:
            if verbose:
                print(f'LFADS: saving results to {out_path}')
            # make sure the directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(results, out_path)

    return results
