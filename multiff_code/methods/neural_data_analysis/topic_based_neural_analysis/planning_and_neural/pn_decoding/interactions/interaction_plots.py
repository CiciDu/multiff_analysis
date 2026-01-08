import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, ttest_rel


# ============================================================
# Utilities
# ============================================================


def summarize_with_sem(
    *,
    results_df,
    group_cols,
    value_col='balanced_accuracy',
):
    """
    Compute mean and SEM across CV folds.
    """
    return (
        results_df
        .groupby(group_cols, as_index=False)
        .agg(
            mean=('balanced_accuracy', 'mean'),
            sem=('balanced_accuracy', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            n=('balanced_accuracy', 'size'),
        )
    )


def test_condition_vs_global(
    *,
    results_df,
    context,
    model_type,
    value_col='balanced_accuracy',
    test='wilcoxon',
):
    """
    Paired statistical test comparing a context vs global across folds.
    """

    df_ctx = results_df.query(
        "context == @context and model == @model_type"
    ).sort_values('fold')

    df_glob = results_df.query(
        "context == 'global' and model == @model_type"
    ).sort_values('fold')

    if len(df_ctx) == 0 or len(df_glob) == 0:
        return None

    x = df_ctx[value_col].values
    y = df_glob[value_col].values

    if test == 'wilcoxon':
        stat, pval = wilcoxon(x, y)
    elif test == 'ttest':
        stat, pval = ttest_rel(x, y)
    else:
        raise ValueError('Unknown test')

    return {
        'context': context,
        'model': model_type,
        'p_value': pval,
        'mean_diff': np.mean(x - y),
    }


def significance_marker(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


# ============================================================
# Panel A — Global interaction decoding
# ============================================================
def bootstrap_ci(
    values,
    *,
    n_boot=2000,
    ci=95,
    random_state=0,
):
    """
    Bootstrap confidence interval for the mean.
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)

    boot_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)

    return np.mean(values), lower, upper


def plot_global_interaction_decoding(
    *,
    interaction_summary_df,
    interaction_name,
    ax,
):
    ax.bar(
        interaction_summary_df['model'],
        interaction_summary_df['mean_bal_acc'],
        alpha=0.8,
    )

    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel('Decoder model')
    ax.set_title(f'Global decoding: {interaction_name}')
    ax.set_ylim(0, 1)


def plot_conditioned_vs_global_bootstrap(
    *,
    cond_results_df,
    model_type,
    target_name,
    condition_name,
    ax,
    n_boot=2000,
    ci=95,
):
    """
    Plot conditioned decoding vs global using bootstrap CIs
    on (conditioned − global) effect size.
    """

    # extract global fold values
    global_vals = (
        cond_results_df
        .query("context == 'global' and model == @model_type")
        ['balanced_accuracy']
        .values
    )

    contexts = [
        c for c in cond_results_df['context'].unique()
        if c != 'global'
    ]

    means = []
    lowers = []
    uppers = []

    for ctx in contexts:
        ctx_vals = (
            cond_results_df
            .query("context == @ctx and model == @model_type")
            ['balanced_accuracy']
            .values
        )

        # paired difference per fold
        diffs = ctx_vals - global_vals

        mean, lo, hi = bootstrap_ci(
            diffs,
            n_boot=n_boot,
            ci=ci,
        )

        means.append(mean)
        lowers.append(lo)
        uppers.append(hi)

    x = np.arange(len(contexts))

    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lowers),
              np.array(uppers) - np.array(means)],
        fmt='o',
        capsize=4,
        label='Conditioned − Global',
    )

    ax.axhline(0, linestyle='--', color='k', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=30)
    ax.set_ylabel('Δ Balanced accuracy')
    ax.set_xlabel(condition_name)
    ax.set_title(f'{target_name} modulation by {condition_name}')

def plot_fold_level_distributions(
    *,
    results_df,
    group_col,
    ax,
    y_col='balanced_accuracy',
):
    groups = results_df[group_col].unique()

    for i, g in enumerate(groups):
        vals = results_df[results_df[group_col] == g][y_col]
        ax.scatter(
            np.full(len(vals), i),
            vals,
            alpha=0.6,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45)
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title('Fold-level decoding performance')

def plot_pairwise_interaction_analysis(
    *,
    analysis_out,
    interaction_name,
    var_a,
    var_b,
    model_type='logreg',
    n_boot=2000,
    ci=95,
    figsize=(12, 4),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ------------------------
    # Panel A: global interaction decoding
    # ------------------------
    plot_global_interaction_decoding(
        interaction_summary_df=analysis_out['interaction_summary'],
        interaction_name=interaction_name,
        ax=axes[0],
    )

    # ------------------------
    # Panel B: var_a | var_b (bootstrap Δ)
    # ------------------------
    plot_conditioned_vs_global_bootstrap(
        cond_results_df=analysis_out['cond_var_a_results'],
        model_type=model_type,
        target_name=var_a,
        condition_name=var_b,
        ax=axes[1],
        n_boot=n_boot,
        ci=ci,
    )

    # ------------------------
    # Panel C: var_b | var_a (bootstrap Δ)
    # ------------------------
    plot_conditioned_vs_global_bootstrap(
        cond_results_df=analysis_out['cond_var_b_results'],
        model_type=model_type,
        target_name=var_b,
        condition_name=var_a,
        ax=axes[2],
        n_boot=n_boot,
        ci=ci,
    )

    plt.tight_layout()
    return fig
