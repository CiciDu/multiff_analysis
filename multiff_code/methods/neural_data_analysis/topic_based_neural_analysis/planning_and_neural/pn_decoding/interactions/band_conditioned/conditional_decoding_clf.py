import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import discrete_decoders
from joblib import Parallel, delayed
from tqdm import tqdm


def bootstrap_trials(x_df, y_df, stratify_col, random_state=None):
    """
    Trial-level stratified bootstrap.
    Samples rows with replacement, preserving class proportions.
    """
    rng = np.random.default_rng(random_state)

    boot_idx = []
    for val in y_df[stratify_col].unique():
        idx = np.where(y_df[stratify_col].values == val)[0]
        sampled = rng.choice(idx, size=len(idx), replace=True)
        boot_idx.append(sampled)

    boot_idx = np.concatenate(boot_idx)

    return (
        x_df.iloc[boot_idx].reset_index(drop=True),
        y_df.iloc[boot_idx].reset_index(drop=True),
    )


def decode_global(
    x_df,
    y_df,
    target_col,
    model_type='logreg',
    n_splits=5,
    n_bootstraps=None,
    random_state=0,
):
    rows = []

    if n_bootstraps is None:
        res = discrete_decoders.decode_behavioral_variable_xy(
            x_df=x_df,
            y_df=y_df,
            label_col=target_col,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['context'] = 'global'
        res['bootstrap_id'] = -1
        return res

    for b in range(n_bootstraps):
        x_boot, y_boot = bootstrap_trials(
            x_df, y_df,
            stratify_col=target_col,
            random_state=random_state + b,
        )

        res = discrete_decoders.decode_behavioral_variable_xy(
            x_df=x_boot,
            y_df=y_boot,
            label_col=target_col,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['context'] = 'global'
        res['bootstrap_id'] = b
        rows.append(res)

    return pd.concat(rows, ignore_index=True)


def _single_bootstrap_delta(
    b,
    *,
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type,
    n_splits,
    min_samples,
    random_state,
):
    rows = []

    x_boot, y_boot = bootstrap_trials(
        x_df, y_df,
        stratify_col=target_col,
        random_state=random_state + b,
    )

    res_global = discrete_decoders.decode_behavioral_variable_xy(
        x_df=x_boot,
        y_df=y_boot,
        label_col=target_col,
        model_type=model_type,
        n_splits=n_splits,
    ).set_index('fold')

    for val in y_boot[condition_col].cat.categories:
        mask = y_boot[condition_col] == val
        if mask.sum() < min_samples:
            continue
        if y_boot.loc[mask, target_col].nunique() < 2:
            continue

        res_cond = discrete_decoders.decode_behavioral_variable_xy(
            x_df=x_boot.loc[mask].reset_index(drop=True),
            y_df=y_boot.loc[mask].reset_index(drop=True),
            label_col=target_col,
            model_type=model_type,
            n_splits=n_splits,
        ).set_index('fold')

        common_folds = res_global.index.intersection(res_cond.index)

        for fold in common_folds:
            rows.append({
                'bootstrap_id': b,
                'fold': fold,
                'condition_value': val,
                'delta_balanced_accuracy':
                    res_cond.loc[fold, 'balanced_accuracy']
                    - res_global.loc[fold, 'balanced_accuracy'],
                'global_balanced_accuracy':
                    res_global.loc[fold, 'balanced_accuracy'],
                'cond_balanced_accuracy':
                    res_cond.loc[fold, 'balanced_accuracy'],
            })

    return rows


def bootstrap_conditioned_minus_global(
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type='logreg',
    n_splits=3,
    n_bootstraps=200,
    min_samples=200,
    random_state=0,
    n_jobs=-1,
):
    """
    Parallel trial-level bootstrap of (conditioned − global).
    Safe for n_jobs=1.
    """

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap_delta)(
            b,
            x_df=x_df,
            y_df=y_df,
            target_col=target_col,
            condition_col=condition_col,
            model_type=model_type,
            n_splits=n_splits,
            min_samples=min_samples,
            random_state=random_state,
        )
        for b in tqdm(
            range(n_bootstraps),
            desc=f'Bootstrap Δ: {target_col} | {condition_col} [{model_type}]',
        )
    )

    rows = [r for sublist in results for r in sublist]
    return pd.DataFrame(rows)


def decode_component_conditioned(
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type='logreg',
    n_splits=5,
    min_samples=200,
    n_bootstraps=None,
    random_state=0,
):
    rows = []

    for val in y_df[condition_col].cat.categories:
        mask = y_df[condition_col] == val
        if mask.sum() < min_samples:
            continue
        if y_df.loc[mask, target_col].nunique() < 2:
            continue

        x_sub = x_df.loc[mask].reset_index(drop=True)
        y_sub = y_df.loc[mask].reset_index(drop=True)

        if n_bootstraps is None:
            res = discrete_decoders.decode_behavioral_variable_xy(
                x_df=x_sub,
                y_df=y_sub,
                label_col=target_col,
                model_type=model_type,
                n_splits=n_splits,
            )
            res['context'] = val
            res['bootstrap_id'] = -1
            rows.append(res)
            continue

        for b in range(n_bootstraps):
            x_boot, y_boot = bootstrap_trials(
                x_sub, y_sub,
                stratify_col=target_col,
                random_state=random_state + b,
            )

            res = discrete_decoders.decode_behavioral_variable_xy(
                x_df=x_boot,
                y_df=y_boot,
                label_col=target_col,
                model_type=model_type,
                n_splits=n_splits,
            )
            res['context'] = val
            res['bootstrap_id'] = b
            rows.append(res)

    return pd.concat(rows, ignore_index=True)


def compare_component_conditioned_vs_global(
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type='logreg',
    n_splits=5,
    n_bootstraps=None,
):
    res_global = decode_global(
        x_df=x_df,
        y_df=y_df,
        target_col=target_col,
        model_type=model_type,
        n_splits=n_splits,
        n_bootstraps=n_bootstraps,
    )

    res_cond = decode_component_conditioned(
        x_df=x_df,
        y_df=y_df,
        target_col=target_col,
        condition_col=condition_col,
        model_type=model_type,
        n_splits=n_splits,
        n_bootstraps=n_bootstraps,
    )

    return pd.concat([res_global, res_cond], ignore_index=True)
