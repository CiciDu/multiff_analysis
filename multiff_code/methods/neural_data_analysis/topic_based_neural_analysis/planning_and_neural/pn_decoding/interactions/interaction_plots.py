import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_bootstrap_ci(values, ci=95):
    """
    Percentile confidence interval from a bootstrap distribution.
    """
    values = np.asarray(values)
    alpha = (100 - ci) / 2

    return {
        'mean': np.mean(values),
        'ci_low': np.percentile(values, alpha),
        'ci_high': np.percentile(values, 100 - alpha),
    }


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


def plot_bootstrapped_delta(
    *,
    delta_bootstrap_df,
    model_type,
    target_name,
    condition_name,
    ax,
    ci=95,
):
    """
    Plot absolute conditioned decoding accuracy with bootstrapped CI
    and a dashed baseline at the global accuracy.
    Expects per-bootstrap (fold-collapsed) estimates with columns:
      - delta_balanced_accuracy
      - global_balanced_accuracy
      - cond_balanced_accuracy (optional; if absent will be derived)
    """

    df = delta_bootstrap_df.query("model == @model_type")

    condition_values = sorted(df['condition_value'].unique())

    means = []
    lowers = []
    uppers = []

    # Compute global baseline as mean across bootstraps
    if 'global_balanced_accuracy' in df.columns:
        global_vals = df[['bootstrap_id', 'global_balanced_accuracy']].drop_duplicates()[
            'global_balanced_accuracy'].values
        global_summary = summarize_bootstrap_ci(global_vals, ci=ci)
        global_baseline = global_summary['mean']
    else:
        # Fallback to zero if not provided (should not happen if upstream filled it)
        global_baseline = 0.0

    for val in condition_values:
        sub = df.query("condition_value == @val")
        if 'cond_balanced_accuracy' in sub.columns:
            vals = sub['cond_balanced_accuracy'].values
        else:
            # derive absolute conditioned accuracy = delta + global per-bootstrap
            # attempt to align by bootstrap_id
            if 'global_balanced_accuracy' in sub.columns and 'bootstrap_id' in sub.columns:
                vals = (sub['delta_balanced_accuracy'] +
                        sub['global_balanced_accuracy']).values
            else:
                # final fallback: shift deltas by global_baseline scalar
                vals = (sub['delta_balanced_accuracy'] +
                        global_baseline).values

        summary = summarize_bootstrap_ci(vals, ci=ci)

        means.append(summary['mean'])
        lowers.append(summary['ci_low'])
        uppers.append(summary['ci_high'])

    x = np.arange(len(condition_values))

    ax.errorbar(
        x,
        means,
        yerr=[
            np.array(means) - np.array(lowers),
            np.array(uppers) - np.array(means),
        ],
        fmt='o',
        capsize=4,
    )

    # Baseline at global accuracy
    ax.axhline(global_baseline, linestyle='--', color='k', alpha=0.6)

    ax.set_xticks(x)
    # If available, annotate each condition with its sample size
    if 'n_samples' in df.columns:
        n_map = (
            df[['condition_value', 'n_samples']]
            .drop_duplicates()
            .set_index('condition_value')['n_samples']
            .to_dict()
        )
        labels = [
            f'{val} (n={int(n_map.get(val, np.nan))})' for val in condition_values]
    else:
        labels = [str(val) for val in condition_values]
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel(condition_name)
    ax.set_title(f'{target_name} | {condition_name}')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))


def plot_pairwise_interaction_analysis(
    *,
    analysis_out,
    interaction_name,
    var_a,
    var_b,
    model_type='logreg',
    ci=95,
    figsize=(12, 4),
):
    """
    Three-panel plot:
    A) Global interaction decoding
    B) Δ decoding: var_a | var_b
    C) Δ decoding: var_b | var_a
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_global_interaction_decoding(
        interaction_summary_df=analysis_out['interaction_summary'],
        interaction_name=interaction_name,
        ax=axes[0],
    )

    plot_bootstrapped_delta(
        delta_bootstrap_df=analysis_out['cond_var_a_delta_bootstrap'],
        model_type=model_type,
        target_name=var_a,
        condition_name=var_b,
        ax=axes[1],
        ci=ci,
    )

    plot_bootstrapped_delta(
        delta_bootstrap_df=analysis_out['cond_var_b_delta_bootstrap'],
        model_type=model_type,
        target_name=var_b,
        condition_name=var_a,
        ax=axes[2],
        ci=ci,
    )

    plt.tight_layout()
    return fig
