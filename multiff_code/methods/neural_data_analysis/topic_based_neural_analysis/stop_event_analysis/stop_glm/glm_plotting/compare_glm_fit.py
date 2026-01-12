import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_insample_model_comparison(
    metrics_by_model,
    *,
    bins=20,
    dev_expl_xlim=(0.0, 0.15),
    ll_per_obs_xlim=(-0.05, 0.05),
    alpha=0.6,
    show=True,
):
    """
    Compare IN-SAMPLE GLM performance across multiple models.

    Parameters
    ----------
    metrics_by_model : dict[str, pd.DataFrame]
        Mapping from model name → metrics_df.
    bins : int
        Histogram bins.
    *_xlim : tuple
        Axis limits (visualization only).
    alpha : float
        Transparency for overlays.
    """

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    # ---- helper to compute derived in-sample quantities ----
    def _prepare(df):
        out = df.copy()
        out['ll_improvement'] = out['llf'] - out['llnull']
        out['ll_per_obs'] = out['ll_improvement'] / out['n_obs']
        return out

    prepared = {
        name: _prepare(df)
        for name, df in metrics_by_model.items()
    }

    # ========== 1. In-sample deviance explained ==========
    for (name, df), c in zip(prepared.items(), colors):
        x = df['deviance_explained'].clip(*dev_expl_xlim)
        axes[0].hist(
            x.dropna(),
            bins=bins,
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[0].set_xlim(dev_expl_xlim)
    axes[0].set_xlabel('Deviance explained')
    axes[0].set_ylabel('Number of neurons')
    axes[0].set_title('In-sample deviance explained')
    axes[0].legend(frameon=False)

    # ========== 2. In-sample normalized log-likelihood gain ==========
    for (name, df), c in zip(prepared.items(), colors):
        x = df['ll_per_obs'].clip(*ll_per_obs_xlim)
        axes[1].hist(
            x.dropna(),
            bins=bins,
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[1].axvline(0, linestyle='--', color='k')
    axes[1].set_xlim(ll_per_obs_xlim)
    axes[1].set_xlabel('Δ log-likelihood per observation')
    axes[1].set_ylabel('Number of neurons')
    axes[1].set_title('In-sample normalized gain')

    # ========== 3. Median comparison ==========
    medians = [
        np.nanmedian(df['ll_per_obs'])
        for df in prepared.values()
    ]
    axes[2].bar(
        list(prepared.keys()),
        medians,
        color=colors[:len(medians)],
    )
    axes[2].axhline(0, linestyle='--', color='k')
    axes[2].set_ylabel('Median Δ log-likelihood / obs')
    axes[2].set_title('Median in-sample predictive gain')
    axes[2].tick_params(axis='x', rotation=20)

    # ========== 4. In-sample deviance vs sparsity ==========
    for (name, df), c in zip(prepared.items(), colors):
        if 'zero_frac' not in df.columns:
            continue
        axes[3].scatter(
            df['zero_frac'],
            df['deviance_explained'].clip(*dev_expl_xlim),
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[3].set_ylim(dev_expl_xlim)
    axes[3].set_xlabel('Zero fraction')
    axes[3].set_ylabel('Deviance explained')
    axes[3].set_title('Effect of sparsity')

    # ========== 5. In-sample LL gain vs deviance explained ==========
    for (name, df), c in zip(prepared.items(), colors):
        axes[4].scatter(
            df['deviance_explained'].clip(*dev_expl_xlim),
            df['ll_per_obs'].clip(*ll_per_obs_xlim),
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[4].axhline(0, linestyle='--', color='k')
    axes[4].set_xlim(dev_expl_xlim)
    axes[4].set_ylim(ll_per_obs_xlim)
    axes[4].set_xlabel('Deviance explained')
    axes[4].set_ylabel('Δ log-likelihood / obs')
    axes[4].set_title('In-sample consistency')

    # ========== 6. Empty / legend panel ==========
    axes[5].axis('off')
    axes[5].legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc='center',
        frameon=False,
    )

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_cv_model_comparison(
    metrics_by_model,
    *,
    bins=20,
    cv_dev_xlim=(-0.1, 0.15),
    cv_ll_per_obs_xlim=(-0.05, 0.05),
    alpha=0.6,
    show=True,
):
    """
    Compare cross-validated GLM performance across multiple models.

    Parameters
    ----------
    metrics_by_model : dict[str, pd.DataFrame]
        Mapping from model name → metrics_df.
    bins : int
        Histogram bins.
    *_xlim : tuple
        Axis limits (visualization only).
    alpha : float
        Transparency for overlays.
    """

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    # ---- helper to compute derived CV quantities ----
    def _prepare(df):
        out = df.copy()
        out['cv_ll_per_obs'] = out['cv_loglik_improvement'] / out['n_obs']
        return out

    prepared = {
        name: _prepare(df)
        for name, df in metrics_by_model.items()
    }

    # ========== 1. CV deviance explained ==========
    for (name, df), c in zip(prepared.items(), colors):
        x = df['cv_deviance_explained'].clip(*cv_dev_xlim)
        axes[0].hist(
            x.dropna(),
            bins=bins,
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[0].set_xlim(cv_dev_xlim)
    axes[0].set_xlabel('Deviance explained')
    axes[0].set_ylabel('Number of neurons')
    axes[0].set_title('CV deviance explained')
    axes[0].legend(frameon=False)

    # ========== 2. CV normalized log-likelihood gain ==========
    for (name, df), c in zip(prepared.items(), colors):
        x = df['cv_ll_per_obs'].clip(*cv_ll_per_obs_xlim)
        axes[1].hist(
            x.dropna(),
            bins=bins,
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[1].axvline(0, linestyle='--', color='k')
    axes[1].set_xlim(cv_ll_per_obs_xlim)
    axes[1].set_xlabel('Δ log-likelihood per observation')
    axes[1].set_ylabel('Number of neurons')
    axes[1].set_title('CV normalized gain')

    # ========== 3. Median comparison ==========
    medians = [
        np.nanmedian(df['cv_ll_per_obs'])
        for df in prepared.values()
    ]
    axes[2].bar(
        list(prepared.keys()),
        medians,
        color=colors[:len(medians)],
    )
    axes[2].axhline(0, linestyle='--', color='k')
    axes[2].set_ylabel('Median Δ log-likelihood / obs')
    axes[2].set_title('Median CV predictive gain')
    axes[2].tick_params(axis='x', rotation=20)

    # ========== 4. CV deviance vs sparsity ==========
    for (name, df), c in zip(prepared.items(), colors):
        if 'zero_frac' not in df.columns:
            continue
        axes[3].scatter(
            df['zero_frac'],
            df['cv_deviance_explained'].clip(*cv_dev_xlim),
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[3].set_ylim(cv_dev_xlim)
    axes[3].set_xlabel('Zero fraction')
    axes[3].set_ylabel('Deviance explained')
    axes[3].set_title('Effect of sparsity')

    # ========== 5. CV LL gain vs deviance explained ==========
    for (name, df), c in zip(prepared.items(), colors):
        axes[4].scatter(
            df['cv_deviance_explained'].clip(*cv_dev_xlim),
            df['cv_ll_per_obs'].clip(*cv_ll_per_obs_xlim),
            alpha=alpha,
            label=name,
            color=c,
        )
    axes[4].axhline(0, linestyle='--', color='k')
    axes[4].set_xlim(cv_dev_xlim)
    axes[4].set_ylim(cv_ll_per_obs_xlim)
    axes[4].set_xlabel('Deviance explained')
    axes[4].set_ylabel('Δ log-likelihood / obs')
    axes[4].set_title('CV consistency')

    # ========== 6. Empty / legend panel ==========
    axes[5].axis('off')
    axes[5].legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc='center',
        frameon=False,
    )

    plt.tight_layout()
    if show:
        plt.show()
