import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import add_interactions, discrete_decoders
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions.band_conditioned import conditional_decoding_clf


def summarize_bootstrap_deltas(df):
    """
    Collapse CV folds within each bootstrap, then compute CI across bootstraps.
    """

    collapsed = (
        df
        .groupby(['bootstrap_id', 'condition_value', 'model'], as_index=False)
        .agg(
            delta_balanced_accuracy=('delta_balanced_accuracy', 'mean'),
            global_balanced_accuracy=('global_balanced_accuracy', 'mean'),
            cond_balanced_accuracy=('cond_balanced_accuracy', 'mean'),
        )
    )

    summary = (
        collapsed
        .groupby(['condition_value', 'model'], as_index=False)
        .agg(
            mean_delta_bal_acc=('delta_balanced_accuracy', 'mean'),
            ci_low=('delta_balanced_accuracy',
                    lambda x: np.percentile(x, 2.5)),
            ci_high=('delta_balanced_accuracy',
                     lambda x: np.percentile(x, 97.5)),
        )
    )

    return collapsed, summary


def run_pairwise_interaction_analysis(
    *,
    x_df,
    y_df,
    var_a,
    var_b,
    interaction_col,
    unconditioned_model_types=('logreg', 'svm', 'ridge'),
    conditioned_model_types=('logreg', 'svm', 'ridge'),
    min_count=200,
    n_splits=3,
    n_bootstraps=100,
    random_state=0,
):
    """
    Pairwise interaction decoding with paired, trial-level bootstrap
    of (conditioned − global), collapsing CV folds within each bootstrap.
    """

    # ============================================================
    # 1. Interaction label
    # ============================================================
    y_df = add_interactions.add_pairwise_interaction(
        df=y_df,
        var_a=var_a,
        var_b=var_b,
        new_col=interaction_col,
    )

    # ============================================================
    # 2. Prune rare interaction states
    # ============================================================
    y_pruned, x_pruned = add_interactions.prune_rare_states_two_dfs(
        df_behavior=y_df,
        df_neural=x_df,
        label_col=interaction_col,
        min_count=min_count,
    )

    # ============================================================
    # 3. Global interaction decoding (model sweep, no bootstrap)
    # ============================================================
    interaction_results = discrete_decoders.sweep_decoders_xy(
        x_df=x_pruned,
        y_df=y_pruned,
        label_col=interaction_col,
        model_types=list(unconditioned_model_types),
        n_splits=n_splits,
    )

    interaction_summary = (
        interaction_results
        .groupby('model', as_index=False)
        .agg(mean_bal_acc=('balanced_accuracy', 'mean'))
    )

    # ============================================================
    # 4–5. Bootstrapped Δ decoding (shared logic)
    # ============================================================
    def run_conditioned_delta(target_col, condition_col):
        all_models = []

        for model_type in conditioned_model_types:
            df = conditional_decoding_clf.bootstrap_conditioned_minus_global(
                x_df=x_pruned,
                y_df=y_pruned,
                target_col=target_col,
                condition_col=condition_col,
                model_type=model_type,
                n_splits=n_splits,
                n_bootstraps=n_bootstraps,
                min_samples=min_count,
                random_state=random_state,
            )
            df['model'] = model_type
            df['target'] = target_col
            df['condition'] = condition_col
            all_models.append(df)

        raw = pd.concat(all_models, ignore_index=True)
        collapsed, summary = summarize_bootstrap_deltas(raw)

        # ------------------------------------------------------------
        # Attach per-condition sample sizes for labeling in plots
        # Use the pruned behavioral dataframe counts for the condition
        # (constant across bootstraps; repeated per-row after merge)
        # ------------------------------------------------------------
        condition_counts = (
            y_pruned[condition_col]
            .value_counts()
            .rename_axis('condition_value')
            .reset_index(name='n_samples')
        )
        present_vals = collapsed['condition_value'].unique()
        condition_counts = condition_counts[
            condition_counts['condition_value'].isin(present_vals)
        ]
        collapsed = collapsed.merge(
            condition_counts, on='condition_value', how='left')

        return raw, collapsed, summary

    # var_a | var_b
    (
        cond_a_raw,
        cond_a_collapsed,
        cond_a_summary,
    ) = run_conditioned_delta(var_a, var_b)

    # var_b | var_a
    (
        cond_b_raw,
        cond_b_collapsed,
        cond_b_summary,
    ) = run_conditioned_delta(var_b, var_a)

    # ============================================================
    # 6. Return
    # ============================================================
    return {
        'x_pruned': x_pruned,
        'y_pruned': y_pruned,

        # Interaction decoding
        'interaction_results': interaction_results,
        'interaction_summary': interaction_summary,

        # Conditioned − global (raw per fold)
        'cond_var_a_delta_raw': cond_a_raw,
        'cond_var_b_delta_raw': cond_b_raw,

        # Collapsed per bootstrap
        'cond_var_a_delta_bootstrap': cond_a_collapsed,
        'cond_var_b_delta_bootstrap': cond_b_collapsed,

        # Final CI summaries
        'cond_var_a_delta_summary': cond_a_summary,
        'cond_var_b_delta_summary': cond_b_summary,
    }
