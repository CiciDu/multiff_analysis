import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import add_interactions, discrete_decoders, conditional_decoding

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
    n_splits=5,
):
    """
    Run a full decoding analysis for a pairwise interaction.

    Steps
    -----
    1. Add interaction label
    2. Prune rare interaction states
    3. Global (unconditioned) interaction decoding with model sweep
    4. Conditional decoding of var_a | var_b
    5. Conditional decoding of var_b | var_a

    Returns
    -------
    dict of pandas.DataFrame
        All intermediate and summary results
    """

    # ------------------------
    # 1. Add interaction label
    # ------------------------
    y_df = add_interactions.add_pairwise_interaction(
        df=y_df,
        var_a=var_a,
        var_b=var_b,
        new_col=interaction_col,
    )

    # ------------------------
    # 2. Prune rare interaction states
    # ------------------------
    y_pruned, x_pruned = add_interactions.prune_rare_states_two_dfs(
        df_behavior=y_df,
        df_neural=x_df,
        label_col=interaction_col,
        min_count=min_count,
    )

    # ------------------------
    # 3. Global interaction decoding
    # ------------------------
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

    # ------------------------
    # 4. Conditional decoding: var_a | var_b
    # ------------------------
    cond_a_results = []

    for model_type in conditioned_model_types:
        res = conditional_decoding.compare_component_conditioned_vs_global(
            x_df=x_pruned,
            y_df=y_pruned,
            target_col=var_a,
            condition_col=var_b,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['model'] = model_type
        res['target'] = var_a
        res['condition'] = var_b
        cond_a_results.append(res)

    cond_a_results = pd.concat(cond_a_results, ignore_index=True)

    cond_a_summary = (
        cond_a_results
        .groupby(['context', 'model'], as_index=False)
        .agg(mean_bal_acc=('balanced_accuracy', 'mean'))
    )

    # ------------------------
    # 5. Conditional decoding: var_b | var_a
    # ------------------------
    cond_b_results = []

    for model_type in conditioned_model_types:
        res = conditional_decoding.compare_component_conditioned_vs_global(
            x_df=x_pruned,
            y_df=y_pruned,
            target_col=var_b,
            condition_col=var_a,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['model'] = model_type
        res['target'] = var_b
        res['condition'] = var_a
        cond_b_results.append(res)

    cond_b_results = pd.concat(cond_b_results, ignore_index=True)

    cond_b_summary = (
        cond_b_results
        .groupby(['context', 'model'], as_index=False)
        .agg(mean_bal_acc=('balanced_accuracy', 'mean'))
    )

    # ------------------------
    # 6. Return everything
    # ------------------------
    return {
        'x_pruned': x_pruned,
        'y_pruned': y_pruned,

        'interaction_results': interaction_results,
        'interaction_summary': interaction_summary,

        'cond_var_a_results': cond_a_results,
        'cond_var_a_summary': cond_a_summary,

        'cond_var_b_results': cond_b_results,
        'cond_var_b_summary': cond_b_summary,
    }
