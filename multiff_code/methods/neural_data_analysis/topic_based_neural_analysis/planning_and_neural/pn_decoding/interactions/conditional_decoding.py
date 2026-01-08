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


def decode_global(
    x_df,
    y_df,
    target_col,
    model_type='logreg',
    n_splits=5,
):
    res = discrete_decoders.decode_behavioral_variable_xy(
        x_df=x_df,
        y_df=y_df,
        label_col=target_col,
        model_type=model_type,
        n_splits=n_splits,
    )
    res['context'] = 'global'
    return res


def decode_component_conditioned(
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type='logreg',
    n_splits=5,
    min_samples=200,
):
    rows = []

    for val in y_df[condition_col].cat.categories:
        mask = y_df[condition_col] == val
        if mask.sum() < min_samples:
            continue

        # skip degenerate cases
        if y_df.loc[mask, target_col].nunique() < 2:
            continue

        res = discrete_decoders.decode_behavioral_variable_xy(
            x_df=x_df.loc[mask].reset_index(drop=True),
            y_df=y_df.loc[mask].reset_index(drop=True),
            label_col=target_col,
            model_type=model_type,
            n_splits=n_splits,
        )

        res['context'] = val
        rows.append(res)

    return pd.concat(rows, ignore_index=True)


def compare_component_conditioned_vs_global(
    x_df,
    y_df,
    target_col,
    condition_col,
    model_type='logreg',
    n_splits=5,
):
    res_global = decode_global(
        x_df, y_df,
        target_col=target_col,
        model_type=model_type,
        n_splits=n_splits,
    )

    res_cond = decode_component_conditioned(
        x_df, y_df,
        target_col=target_col,
        condition_col=condition_col,
        model_type=model_type,
        n_splits=n_splits,
    )

    return pd.concat([res_global, res_cond], ignore_index=True)
