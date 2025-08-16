from planning_analysis.only_cur_ff import features_to_keep_utils
from machine_learning.ml_methods import prep_ml_data_utils
from machine_learning.ml_methods import prep_ml_data_utils
import matplotlib.pyplot as plt
import math
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt


def winsorize_x_df(x_features_df):
    for feature in x_features_df.columns:
        if ('angle' in feature) & ('rank' not in feature):
            # Winsorize the feature column at 5th and 95th percentiles
            x_features_df[feature] = winsorize(
                x_features_df[feature], limits=[0.01, 0.01])
    return x_features_df


def streamline_preparing_for_ml(x_df,
                                y_df,
                                y_var_column,
                                ref_columns_only=False,
                                cluster_to_keep='none',
                                cluster_for_interaction='none',
                                add_ref_interaction=True,
                                winsorize_angle_features=True,
                                using_lasso=True,
                                ensure_cur_ff_at_front=True,
                                use_pca=False,
                                use_combd_features_for_cluster_only=False,
                                for_classification=False):

    if len(x_df) != len(y_df):
        raise ValueError('x_df and y_df should have the same length.')

    x_df = x_df.reset_index(drop=True).copy()
    y_df = y_df.reset_index(drop=True).copy()

    if ensure_cur_ff_at_front:
        x_df = x_df[x_df['cur_ff_angle_at_ref']
                    .between(-math.pi/2, math.pi/2)].copy()
        y_df = y_df.loc[x_df.index].copy().reset_index(drop=True)
        x_df.reset_index(drop=True, inplace=True)

    minimal_features_to_keep = features_to_keep_utils.get_minimal_features_to_keep(
        x_df, for_classification=for_classification)
    x_df = x_df[minimal_features_to_keep].copy()

    ref_columns = [column for column in x_df.columns if 'ref' in column]
    if ref_columns_only:
        x_df = x_df[ref_columns].copy()

    if cluster_to_keep == 'all':
        columns_to_delete = []
    else:
        columns_to_delete = [
            column for column in x_df.columns if 'cluster' in column]
    if cluster_to_keep != 'none':
        # separate clusters in cluster_to_keep by _PLUS_
        clusters_to_keep = cluster_to_keep.split('_PLUS_')
        columns_to_delete = [column for column in columns_to_delete if all(
            [cluster not in column for cluster in clusters_to_keep])]

    if len(columns_to_delete) > 0:
        x_df.drop(columns=columns_to_delete, inplace=True)

    if use_combd_features_for_cluster_only:
        new_columns_to_delete = [column for column in x_df.columns if ('combd' not in column) &
                                 ('cluster' in column)]
        x_df.drop(columns=new_columns_to_delete, inplace=True)

    if add_ref_interaction:
        x_df = prep_ml_data_utils.add_interaction_terms_to_df(
            x_df, specific_columns=ref_columns)

    if cluster_for_interaction != 'none':
        specific_columns = [column for column in x_df if (
            cluster_for_interaction in column) & ('combd' in column)]
        x_df = prep_ml_data_utils.add_interaction_terms_to_df(
            x_df, specific_columns=specific_columns)

    if winsorize_angle_features:
        x_df = winsorize_x_df(x_df)

    y_var_df = y_df[[y_var_column]].copy()
    x_var_df, y_var_df, _ = prep_ml_data_utils.make_x_and_y_var_df(
        x_df, y_var_df, use_pca=use_pca)

    print('num_features_before_lasso:', x_var_df.shape[1])

    if len(x_var_df) > 0:
        if using_lasso:
            lasso = LassoCV(cv=5, tol=0.15, max_iter=400).fit(
                x_var_df, y_var_df.values.reshape(-1))
            # Selected variables (non-zero coefficients)
            selected_features = x_var_df.columns[(lasso.coef_ != 0)]
            if len(selected_features) == 0:
                # try again with a lower tolerance and greater max iterations
                lasso = LassoCV(cv=5, tol=0.05, max_iter=700).fit(
                    x_var_df, y_var_df.values.reshape(-1))
                # Selected variables (non-zero coefficients)
                selected_features = x_var_df.columns[(lasso.coef_ != 0)]
            x_var_df = x_var_df[selected_features].copy()

    print('num_features_after_lasso:', x_var_df.shape[1])
    return x_var_df, y_var_df
