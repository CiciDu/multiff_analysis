import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_global_interaction_decoding(
    *,
    interaction_summary_df,
    interaction_name,
    ax,
    chance_level=None,
):
    """
    Plot global interaction decoding performance across models.
    """

    ax.bar(
        interaction_summary_df['model'],
        interaction_summary_df['mean_bal_acc'],
        alpha=0.8,
    )

    if chance_level is not None:
        ax.axhline(
            chance_level,
            linestyle='--',
            color='k',
            alpha=0.5,
            label='Chance',
        )
        ax.legend()

    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel('Decoder model')
    ax.set_title(f'Global decoding: {interaction_name}')
    ax.set_ylim(0, 1)



def plot_fold_level_distributions(
    *,
    results_df,
    group_col,
    ax,
    y_col='balanced_accuracy',
):
    """
    Scatter plot of fold-level decoding performance.
    """

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


def plot_conditioned_vs_global(
    *,
    cond_summary_df,
    model_type,
    target_name,
    condition_name,
    ax,
):
    """
    Plot conditioned decoding vs explicit global (unconditioned) decoding.
    """

    # conditioned contexts only
    cond_only = cond_summary_df.query("context != 'global'")

    ax.plot(
        cond_only['context'],
        cond_only['mean_bal_acc'],
        'o-',
        label='Conditioned',
    )

    # global reference (explicit, correct)
    global_mean = (
        cond_summary_df
        .query("context == 'global' and model == @model_type")
        ['mean_bal_acc']
        .iloc[0]
    )

    ax.axhline(
        global_mean,
        linestyle='--',
        color='k',
        alpha=0.6,
        label='Global',
    )

    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel(condition_name)
    ax.set_title(f'{target_name} decoding | {condition_name}')
    ax.set_ylim(0, 1)
    ax.legend()


def plot_pairwise_interaction_analysis(
    *,
    analysis_out,
    interaction_name,
    var_a,
    var_b,
    figsize=(12, 4),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ------------------------
    # Panel A: Global interaction decoding
    # ------------------------
    plot_global_interaction_decoding(
        interaction_summary_df=analysis_out['interaction_summary'],
        interaction_name=interaction_name,
        ax=axes[0],
    )

    # ------------------------
    # Panel B: var_a | var_b
    # ------------------------
    plot_conditioned_vs_global(
        cond_summary_df=analysis_out['cond_var_a_summary'],
        model_type='logreg',
        target_name=var_a,
        condition_name=var_b,
        ax=axes[1],
    )

    # ------------------------
    # Panel C: var_b | var_a
    # ------------------------
    plot_conditioned_vs_global(
        cond_summary_df=analysis_out['cond_var_b_summary'],
        model_type='logreg',
        target_name=var_b,
        condition_name=var_a,
        ax=axes[2],
    )

    plt.tight_layout()
    return fig
