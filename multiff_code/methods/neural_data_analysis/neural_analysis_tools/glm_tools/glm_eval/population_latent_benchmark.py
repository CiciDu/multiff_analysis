import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import GroupKFold

from .parity_utils import ParityConfig, run_parity_cv


def run_fa_cv(X, trial_ids, n_factors, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    ves = []
    for tr, te in gkf.split(X, groups=trial_ids):
        mu = X[tr].mean(0)
        sd = X[tr].std(0)
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd

        fa = FactorAnalysis(n_components=n_factors)
        Ztr = fa.fit_transform(Xtr)
        Zte = fa.transform(Xte)
        Xhat = Zte @ fa.components_
        num = np.sum((Xte - Xhat) ** 2)
        den = np.sum((Xte - Xte.mean(0)) ** 2)
        ves.append(1 - num / den)
    return float(np.mean(ves)), float(np.std(ves))


from numpy.linalg import lstsq
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import GroupKFold


def run_fa_latent_predictability_cv(
    X, Y, trial_ids, n_factors, n_splits
):
    """
    Cross-validated FA latent predictability:
    how well Y predicts X's FA latents.

    Metric is directly comparable to CCA / RRR VE.
    """
    gkf = GroupKFold(n_splits=n_splits)
    ves = []

    for tr, te in gkf.split(X, groups=trial_ids):
        # ---- standardize (train stats only)
        mux, sdx = X[tr].mean(0), X[tr].std(0)
        muy, sdy = Y[tr].mean(0), Y[tr].std(0)

        Xtr = (X[tr] - mux) / sdx
        Xte = (X[te] - mux) / sdx
        Ytr = (Y[tr] - muy) / sdy
        Yte = (Y[te] - muy) / sdy

        # ---- FA on X
        fa = FactorAnalysis(n_components=n_factors)
        Zx_tr = fa.fit_transform(Xtr)
        Zx_te = fa.transform(Xte)

        # project Y into X-latent space
        Zy_tr = fa.transform(Ytr)
        Zy_te = fa.transform(Yte)

        # ---- learn latent mapping
        A, *_ = lstsq(Zy_tr, Zx_tr, rcond=None)
        Zx_hat = Zy_te @ A

        # ---- latent VE
        num = np.sum((Zx_te - Zx_hat) ** 2)
        den = np.sum((Zx_te - Zx_te.mean(0)) ** 2)
        ves.append(0.0 if den <= 0 else 1.0 - num / den)

    return float(np.mean(ves)), float(np.std(ves))

def population_latent_benchmark_function(
    X, Y, trial_ids, time_idx, 
    condition_labels=None,
    *,
    n_splits=5,
    fa_factors=8,
    max_rank=10,
    n_shuffles=100,
    shuffle_mode='latent',
    rng=0
):
    # ======================
    # FA (latent predictability)
    # ======================
    fa_lat_mean, fa_lat_std = run_fa_latent_predictability_cv(
        X, Y, trial_ids, fa_factors, n_splits
    )

    # ======================
    # CCA + RRR
    # ======================
    cfg = ParityConfig(
        max_rank=max_rank,
        n_splits=n_splits,
        n_shuffles=n_shuffles,
        shuffle_mode=shuffle_mode,
        rng_seed=rng
    )
    parity = run_parity_cv(
        X, Y, trial_ids, time_idx,
        cond=condition_labels,
        config=cfg
    )

    df_rank = parity['df_rank']
    df_spec = parity['df_cca_spectrum']
    df_null = parity['df_cca_null']

    assert {'fold','rank','model','ve_pred_y_from_x'}.issubset(df_rank.columns)
    assert {'fold','rank','component','corr_test'}.issubset(df_spec.columns)
    assert {'fold','component','corr_thresh'}.issubset(df_null.columns)

    # ======================
    # summaries
    # ======================
    # ---- RRR
    rrr = df_rank[df_rank.model == 'rrr']
    rrr_ve_by_rank = rrr.groupby('rank')['ve_pred_y_from_x'].mean().to_dict()
    rrr_std_by_rank = rrr.groupby('rank')['ve_pred_y_from_x'].std().to_dict()

    best_rank = max(
        rrr_ve_by_rank,
        key=lambda r: (rrr_ve_by_rank[r], -r)
    )

    # ---- CCA
    cca_by_rank = (
        df_rank[df_rank.model == 'cca']
        .groupby('rank')['ve_pred_y_from_x']
        .mean()
    )

    cca_std_by_rank = (
        df_rank[df_rank.model == 'cca']
        .groupby('rank')['ve_pred_y_from_x']
        .std()
    )

    cca_ve = (
        df_rank[df_rank.model == 'cca']
        .groupby('fold')['ve_pred_y_from_x']
        .mean()
    )

    # ---- CCA spectrum (best rank only)
    cca_corr = (
        df_spec[df_spec['rank'] == best_rank]
        .groupby('component')['corr_test']
        .mean()
        .values
    )

    cca_null = (
        df_null.groupby('component')['corr_thresh']
        .mean()
        .values[:best_rank]
    )

    # ======================
    # plots
    # ======================
    figs = {}

    # ---- CCA spectrum
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.arange(1, len(cca_corr) + 1)
    ax.plot(x, cca_corr, marker='o', label='Test')
    ax.plot(x, cca_null, '--', label='Shuffle 95%')
    ax.set_xlabel('Component')
    ax.set_ylabel('Correlation')
    ax.set_title('CCA spectrum')
    ax.legend()
    figs['cca_spectrum'] = fig

    # ---- VE summary (aligned metric, with error bars)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    names = ['FA (latent)', 'CCA', f'RRR@{best_rank}']
    means = [
        fa_lat_mean,
        cca_ve.mean(),
        rrr_ve_by_rank[best_rank]
    ]
    stds = [
        fa_lat_std,
        cca_ve.std(),
        rrr_std_by_rank[best_rank]
    ]
    ax.bar(names, means, yerr=stds, capsize=4)
    ax.set_ylabel('Predictive VE (Y → X)')
    ax.set_title('Cross-population predictability')
    figs['ve_summary'] = fig

    # ---- FA vs CCA gap vs rank
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ranks = sorted(cca_by_rank.index)
    gap = [cca_by_rank[r] - fa_lat_mean for r in ranks]
    ax.plot(ranks, gap, marker='o')
    ax.axhline(0, color='k', lw=1, alpha=0.3)
    ax.set_xlabel('Rank')
    ax.set_ylabel('CCA − FA latent VE')
    ax.set_title('Shared structure beyond FA')
    figs['fa_vs_cca_gap'] = fig

    # ---- RRR curve
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(
        sorted(rrr_ve_by_rank),
        [rrr_ve_by_rank[r] for r in sorted(rrr_ve_by_rank)],
        yerr=[rrr_std_by_rank[r] for r in sorted(rrr_std_by_rank)],
        marker='o',
        capsize=3
    )
    ax.set_xlabel('Rank')
    ax.set_ylabel('Predictive VE')
    ax.set_title('RRR VE vs rank')
    figs['rrr_curve'] = fig

    # ======================
    # noise ceiling interpretation
    # ======================
    noise_ceiling = cca_by_rank.max()
    explained_fraction = (
        rrr_ve_by_rank[best_rank] / noise_ceiling
        if noise_ceiling > 0 else np.nan
    )

    return {
        'fa': {
            'latent_ve_mean': fa_lat_mean,
            'latent_ve_std': fa_lat_std
        },
        'cca': {
            've_mean': cca_ve.mean(),
            've_std': cca_ve.std(),
            'corr_mean': cca_corr,
            'corr_thresh95': cca_null,
            'noise_ceiling': noise_ceiling
        },
        'rrr': {
            'best_rank': best_rank,
            've_by_rank': rrr_ve_by_rank,
            'fraction_of_noise_ceiling': explained_fraction
        },
        'figs': figs,
        'parity': parity
    }
