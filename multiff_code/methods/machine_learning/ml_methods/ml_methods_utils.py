from machine_learning.ml_methods import classification_utils, regression_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
import math


def process_summary_df(summary_df):
    summary_df['abs_coeff'] = np.abs(summary_df['Coefficient'])
    summary_df.sort_values(by='abs_coeff', ascending=False, inplace=True)
    summary_df['significant'] = summary_df['p_value'] <= 0.05
    summary_df['rank_by_abs_coeff'] = summary_df['abs_coeff'].rank(
        ascending=False, method='first').astype(int)
    # summary_df.reset_index(drop=False, inplace=True)
    return summary_df


def train_test_split_based_on_segments(x_var, y_var, segment_column='target_index'):
    all_targets = y_var[segment_column].unique()

    test_targets = np.random.choice(
        all_targets, size=int(len(all_targets)*0.2), replace=False)
    train_targets = [
        target for target in all_targets if target not in test_targets]

    x_var.reset_index(drop=True, inplace=True)
    y_var.reset_index(drop=True, inplace=True)

    train_rows = y_var[segment_column].isin(train_targets)
    test_rows = y_var[segment_column].isin(test_targets)

    X_train = x_var[train_rows]
    X_test = x_var[test_rows]

    y_train = y_var[train_rows]
    y_test = y_var[test_rows]

    return X_train, X_test, y_train, y_test


def run_segment_split_regression(x_var, y_var, columns_of_interest, segment_column='new_segment'):
    X_train, X_test, y_train, y_test = train_test_split_based_on_segments(
        x_var, y_var, segment_column=segment_column)
    results = regress_by_variable_type(X_train, X_test, y_train,
                                       y_test, columns_of_interest)
    return results


def run_segment_split_regression_cv(x_var, y_var, columns_of_interest, num_folds=5, segment_column='new_segment', verbose=False, random_state=None):
    results = regress_by_variable_type_cv(
        x_var, y_var, columns_of_interest, num_folds=num_folds, segment_column=segment_column, verbose=verbose, random_state=random_state)
    return results


def regress_by_variable_type(X_train, X_test, y_train, y_test, columns_of_interest):
    for y_var_column in columns_of_interest:
        print('y_var_column:', y_var_column)
        # if y_var_column is a dummy variable, use logistic regression
        num_unique = y_train[y_var_column].nunique()
        if num_unique == 1:
            print(
                f"Skipping target '{y_var_column}' because it has only one unique value.")
            continue
        elif num_unique == 2:
            conf_matrix = classification_utils._use_logistic_regression(
                X_train, X_test, y_train[y_var_column], y_test[y_var_column])
        else:
            summary_df, y_pred, results, r2_test = regression_utils.use_linear_regression(
                X_train, X_test, y_train[y_var_column], y_test[y_var_column], show_plot=True, y_var_name=y_var_column)


def regress_by_variable_type_cv(
    x_var_df,
    y_var_df,
    columns_of_interest,
    num_folds=5,
    segment_column=None,
    verbose=False,
    random_state=None
):
    """
    Perform cross-validation for each target column in columns_of_interest.
    Uses logistic regression for binary features, linear regression for continuous.

    Parameters:
        x_var_df (DataFrame): Predictor variables.
        y_var_df (DataFrame): Feature variables.
        columns_of_interest (list): List of target columns to evaluate.
        num_folds (int): Number of CV folds.
        segment_column (str or None): Optional grouping variable for GroupKFold.
        verbose (bool): Whether to print progress info.
        random_state (int or None): Random state for CV splits. If None, uses deterministic splits.

    Returns:
        pd.DataFrame: Combined results for all target variables and models.
    """
    results_summary = []
    y_var_df = y_var_df.copy()

    if segment_column:
        if segment_column not in y_var_df.columns:
            raise ValueError(
                f"segment_column '{segment_column}' not found in y_var_df")
        groups = y_var_df[segment_column]
        cv = GroupKFold(n_splits=num_folds, shuffle=True,
                        random_state=random_state)
    else:
        groups = None
        cv = KFold(n_splits=num_folds, shuffle=True,
                   random_state=random_state)

    for y_var_column in columns_of_interest:
        if verbose:
            print('\n' + '='*80)
            print(f"Evaluating target variable: {y_var_column}")
            print('='*80)

        num_unique = y_var_df[y_var_column].nunique()
        if num_unique == 1:
            if verbose:
                print(
                    f"Skipping target '{y_var_column}' because it has only one unique value.")
            continue

        if num_unique == 2:
            model_type = "Logistic Regression"
            if verbose:
                print(f"Model: {model_type} (Binary Classification)\n")
            try:
                # make sure that it contains integers
                y_var_df[y_var_column] = y_var_df[y_var_column].astype(int)
                results = classification_utils.use_logistic_regression_cv(
                    x_var_df, y_var_df[y_var_column], cv=cv, groups=groups, verbose=verbose)
            except Exception as e:
                print(
                    f"Error in logistic regression CV for {y_var_column}: {e}. Will skip this target.")
                continue
        else:
            model_type = "Linear Regression"
            if verbose:
                print(f"Model: {model_type} (Continuous Outcome)\n")
            results = regression_utils.use_linear_regression_cv(
                x_var_df, y_var_df[y_var_column], cv=cv, groups=groups, verbose=verbose)

        for metric_name, value in results.items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                mean, std = value
            else:
                mean, std = value, None

            results_summary.append({
                'Feature': y_var_column,
                'Model': model_type,
                'Metric': metric_name,
                'mean': mean,
                'Std': std
            })

        if verbose:
            print('-'*80)

    return pd.DataFrame(results_summary)


def convert_results_to_wide_df(results_df, index_columns=['Feature', 'Model']):
    wide_df = (
        results_df
        .pivot_table(
            index=index_columns,
            columns='Metric',
            values=['mean', 'Std']
        )
    )
    wide_df.columns = [f'{metric}_{stat}' for stat, metric in wide_df.columns]
    wide_df = wide_df.reset_index()

    return wide_df


def _prepare_grouped_bar_data(results_df, metric, features=None):
    """Extract matrix of mean metric values per feature per group."""
    metric_df = results_df[results_df['Metric'] == metric].copy()

    if features is not None:
        metric_df = metric_df[metric_df['Feature'].isin(features)]

    features = (
        metric_df.groupby('Feature')['mean']
        .max()
        .sort_values(ascending=False)
        .index.values
    )
    groups = metric_df['test_or_control'].unique()

    data_dict = {}
    for feat in features:
        vals = []
        for group in groups:
            val = metric_df.loc[
                (metric_df['Feature'] == feat) &
                (metric_df['test_or_control'] == group),
                'mean'
            ]
            vals.append(val.values[0] if len(val) > 0 else np.nan)
        data_dict[feat] = vals

    return features, groups, data_dict


def _draw_grouped_barplot(chunk_targets, chunk_data, groups, metric, plot_title):
    """Render a bar plot for a chunk of features."""
    num_groups = len(groups)
    bar_width = 0.25
    group_spacing = 1.0
    group_centers = np.arange(len(chunk_targets)) * group_spacing

    fig, ax = plt.subplots(figsize=(max(8, len(chunk_targets) * 0.7), 6))

    for i, group in enumerate(groups):
        offsets = group_centers + (i - (num_groups - 1) / 2) * bar_width
        ax.bar(offsets, chunk_data[:, i],
               width=bar_width, label=group, zorder=3)

    for center in group_centers:
        ax.axvline(center - group_spacing / 2, color='lightgray',
                   linestyle='-', linewidth=1, zorder=1)
    ax.axvline(group_centers[-1] + group_spacing / 2,
               color='lightgray', linestyle='-', linewidth=1, zorder=1)

    if 'r2' in metric.lower():
        ax.set_ylim(max(-2, np.nanmin(chunk_data)), 1)
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.8)
        ax.axhline(0.1, color='gray', linestyle='--', linewidth=1)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(chunk_targets, rotation=40, ha='right', fontsize=10)
    ax.set_ylabel(f"{metric} (Mean)")
    ax.set_title(plot_title)
    ax.legend(title='test_or_control')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def make_barplot_to_compare_results(results_df, metric, features=None, max_targets_per_plot=15):
    """
    Main wrapper function: Prepares data and calls plot renderer in chunks.
    """
    features, groups, data_dict = _prepare_grouped_bar_data(
        results_df, metric, features)

    num_targets = len(features)
    num_plots = math.ceil(num_targets / max_targets_per_plot)

    for plot_idx in range(num_plots):
        start = plot_idx * max_targets_per_plot
        end = min(start + max_targets_per_plot, num_targets)
        chunk_targets = features[start:end]
        chunk_data = np.array([data_dict[t] for t in chunk_targets])

        title = f'Comparison of {metric} ({start + 1}–{end})'
        _draw_grouped_barplot(chunk_targets, chunk_data, groups, metric, title)
