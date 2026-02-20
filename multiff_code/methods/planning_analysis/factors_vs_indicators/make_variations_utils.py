
from scipy import stats
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.plan_factors import build_factor_comp
from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class, test_vs_control_utils
from data_wrangling import specific_utils
import pandas as pd
import numpy as np
import pandas as pd
import os
import contextlib
import logging


def make_regrouped_info(test_df,
                        ctrl_df,
                        agg_regrouped_info_func,
                        agg_regrouped_info_kwargs={},
                        # ['ff_flash', 'cluster_flash', 'ff_seen', 'cluster_seen'],
                        key_for_split_choices=['ff_seen'],
                        whether_filter_info_choices=[True],  # [True, False],
                        whether_even_out_distribution_choices=[
                            False],  # [True, False],
                        whether_test_nxt_ff_flash_after_stop_choices=[
                            'yes', 'no', 'flexible'],
                        whether_limit_cur_ff_cluster_50_size_choices=[
                            True, False],
                        ctrl_flash_compared_to_test_choices=[
                            'same', 'flexible'],
                        max_curv_range_choices=[75, 100, 125, 150, 200],
                        verbose=False,
                        allow_single_condition=False
                        ):

    # Check if only one condition has data, or both are empty
    test_only = (len(test_df) > 0) and (len(ctrl_df) == 0)
    ctrl_only = (len(ctrl_df) > 0) and (len(test_df) == 0)
    both_empty = (len(test_df) == 0) and (len(ctrl_df) == 0)
    single_condition = test_only or ctrl_only

    if both_empty:
        return pd.DataFrame()

    if single_condition and not allow_single_condition:
        print('Warning: Only one condition has data, but allow_single_condition=False. Returning empty DataFrame.')
        return pd.DataFrame()
    
    test_and_ctrl_df = pd.concat([test_df, ctrl_df], axis=0)

    regrouped_info = pd.DataFrame()
    column_for_split_dict = {'ff_flash': 'nxt_ff_last_flash_time_bbas',
                             'cluster_flash': 'nxt_ff_cluster_last_flash_time_bbas',
                             'ff_seen': 'NXT_time_ff_last_seen_bbas',
                             'cluster_seen': 'nxt_ff_cluster_last_seen_time_bbas'
                             }

    if_test_nxt_ff_group_appear_after_stop_dict = {
        'yes': {'ff_flash': 'ff_must_flash_after_stop',
                'cluster_flash': 'cluster_must_flash_after_stop',
                'ff_seen': 'ff_must_seen_after_stop',
                'cluster_seen': 'cluster_must_seen_after_stop'
                },
        'no': {'ff_flash': 'ff_no_flash_after_stop',
               'cluster_flash': 'cluster_no_flash_after_stop',
               'ff_seen': 'ff_no_seen_after_stop',
               'cluster_seen': 'cluster_no_seen_after_stop'
               },
        'flexible': {'ff_flash': 'flexible',
                     'cluster_flash': 'flexible',
                     'ff_seen': 'flexible',
                     'cluster_seen': 'flexible'
                     }
    }

    for key_for_split in key_for_split_choices:
        column_for_split = column_for_split_dict[key_for_split]
        for whether_filter_info in whether_filter_info_choices:
            for whether_even_out_distribution in whether_even_out_distribution_choices:
                for whether_limit_cur_ff_cluster_50_size in whether_limit_cur_ff_cluster_50_size_choices:
                    for whether_test_nxt_ff_flash_after_stop in whether_test_nxt_ff_flash_after_stop_choices:
                        # for if_test_nxt_ff_group_appear_after_stop in ['cluster_must_flash_after_stop', 'cluster_no_flash_after_stop', 'ff_must_flash_after_stop', 'ff_no_flash_after_stop', 'flexible']:
                        if_test_nxt_ff_group_appear_after_stop = if_test_nxt_ff_group_appear_after_stop_dict[
                            whether_test_nxt_ff_flash_after_stop][key_for_split]
                        for ctrl_flash_compared_to_test in ctrl_flash_compared_to_test_choices:
                            if ctrl_flash_compared_to_test == 'same':
                                if if_test_nxt_ff_group_appear_after_stop != 'flexible':  # to avoid repetition
                                    if_ctrl_nxt_ff_group_appear_after_stop = if_test_nxt_ff_group_appear_after_stop
                                else:
                                    continue
                            else:
                                if_ctrl_nxt_ff_group_appear_after_stop = 'flexible'
                            for max_curv_range in max_curv_range_choices:
                                # If single condition mode, skip the splitting logic
                                if single_condition:
                                    # Use the available data without splitting
                                    if test_only:
                                        test_df = test_and_ctrl_df.copy()
                                        ctrl_df = pd.DataFrame()
                                    else:  # ctrl_only
                                        ctrl_df = test_and_ctrl_df.copy()
                                        test_df = pd.DataFrame()
                                else:
                                    # Normal mode: split by column_for_split
                                    ctrl_df = test_and_ctrl_df[test_and_ctrl_df[column_for_split].isnull()].copy()
                                    test_df = test_and_ctrl_df[~test_and_ctrl_df[column_for_split].isnull()].copy()

                                    if (len(test_df) == 0) | (len(ctrl_df) == 0):
                                        continue

                                # Only apply filtering/distribution adjustment if we have both conditions
                                if not single_condition:
                                    if whether_filter_info:
                                        test_df, ctrl_df = test_vs_control_utils.filter_both_df(test_df, ctrl_df, max_curv_range=max_curv_range, verbose=verbose,
                                                                                                whether_even_out_distribution=whether_even_out_distribution,
                                                                                                whether_limit_cur_ff_cluster_50_size=whether_limit_cur_ff_cluster_50_size,
                                                                                                if_test_nxt_ff_group_appear_after_stop=if_test_nxt_ff_group_appear_after_stop,
                                                                                                if_ctrl_nxt_ff_group_appear_after_stop=if_ctrl_nxt_ff_group_appear_after_stop)
                                    elif whether_even_out_distribution:  # if not filtering, then only even out the distribution
                                        test_df, ctrl_df = test_vs_control_utils.make_the_distributions_of_distance_more_similar_in_df(
                                            test_df, ctrl_df, verbose=verbose)
                                        test_df, ctrl_df = test_vs_control_utils.make_the_distributions_of_angle_more_similar_in_df(
                                            test_df, ctrl_df, verbose=verbose)

                                # Call aggregation function if we have data
                                if single_condition:
                                    # For single condition, just aggregate the available data
                                    if (len(test_df) > 0) or (len(ctrl_df) > 0):
                                        temp_regrouped_info = agg_regrouped_info_func(
                                            test_df, ctrl_df, **agg_regrouped_info_kwargs)
                                    else:
                                        temp_regrouped_info = pd.DataFrame()
                                else:
                                    # For both conditions, require both to be non-empty
                                    if (len(test_df) > 0) & (len(ctrl_df) > 0):
                                        temp_regrouped_info = agg_regrouped_info_func(
                                            test_df, ctrl_df, **agg_regrouped_info_kwargs)
                                    else:
                                        temp_regrouped_info = pd.DataFrame()

                                temp_regrouped_info['key_for_split'] = key_for_split
                                temp_regrouped_info['whether_filter_info'] = whether_filter_info
                                temp_regrouped_info['whether_even_out_dist'] = whether_even_out_distribution
                                temp_regrouped_info['whether_limit_cur_ff_cluster_50_size'] = whether_limit_cur_ff_cluster_50_size
                                temp_regrouped_info['if_test_nxt_ff_group_appear_after_stop'] = if_test_nxt_ff_group_appear_after_stop
                                temp_regrouped_info['if_ctrl_nxt_ff_group_appear_after_stop'] = if_ctrl_nxt_ff_group_appear_after_stop
                                temp_regrouped_info['ctrl_flash_compared_to_test'] = ctrl_flash_compared_to_test
                                temp_regrouped_info['max_curv_range'] = max_curv_range
                                # temp_regrouped_info['test_sample_size'] = len(test_df)
                                # temp_regrouped_info['ctrl_sample_size'] = len(ctrl_df)
                                regrouped_info = pd.concat(
                                    [regrouped_info, temp_regrouped_info], axis=0)
                                if not whether_filter_info:
                                    break  # Skip remaining max_curv_range values if not filtering
    regrouped_info.reset_index(drop=True, inplace=True)

    return regrouped_info


def make_pooled_median_info_from_test_and_ctrl_heading_info_df(test_heading_info_df,
                                                               ctrl_heading_info_df,
                                                               verbose=True,
                                                               key_for_split_choices=[
                                                                   'ff_seen'],
                                                               whether_filter_info_choices=[
                                                                   True],
                                                               # whether_even_out_distribution_choices=[True, False],
                                                               whether_even_out_distribution_choices=[
                                                                   False],
                                                               whether_test_nxt_ff_flash_after_stop_choices=[
                                                                   'yes', 'no', 'flexible'],
                                                               whether_limit_cur_ff_cluster_50_size_choices=[
                                                                   False],
                                                               ctrl_flash_compared_to_test_choices=[
                                                                   'flexible'],
                                                               max_curv_range_choices=[
                                                                   200],
                                                               allow_single_condition=True,
                                                               ):

    test_heading_info_df = build_factor_comp.process_heading_info_df(
        test_heading_info_df)
    ctrl_heading_info_df = build_factor_comp.process_heading_info_df(
        ctrl_heading_info_df)

    # logging.info(
    #     f'Make all median info from test and ctrl heading info df, based on various combinations of factors...')
    pooled_median_info = make_regrouped_info(test_heading_info_df,
                                             ctrl_heading_info_df,
                                             make_temp_median_info_func,
                                             key_for_split_choices=key_for_split_choices,
                                             whether_filter_info_choices=whether_filter_info_choices,
                                             whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                             whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                             whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                             ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                             max_curv_range_choices=max_curv_range_choices,
                                             verbose=verbose,
                                             allow_single_condition=allow_single_condition)

    return pooled_median_info


def make_per_sess_median_info_from_test_and_ctrl_heading_info_df(test_heading_info_df,
                                                                 ctrl_heading_info_df,
                                                                 verbose=True,
                                                                 key_for_split_choices=[
                                                                     'ff_seen'],
                                                                 whether_filter_info_choices=[
                                                                     True],
                                                                 # whether_even_out_distribution_choices=[True, False],
                                                                 whether_even_out_distribution_choices=[
                                                                     False],
                                                                 whether_test_nxt_ff_flash_after_stop_choices=[
                                                                     'yes', 'no', 'flexible'],
                                                                 whether_limit_cur_ff_cluster_50_size_choices=[
                                                                     False],
                                                                 ctrl_flash_compared_to_test_choices=[
                                                                     'flexible'],
                                                                 max_curv_range_choices=[
                                                                     200],
                                                                 allow_single_condition=True,
                                                                 ):

    per_sess_median_info = pd.DataFrame()
    for data_name in test_heading_info_df['data_name'].unique():
        print(f'Processing data_name: {data_name}')
        session_test_heading_info_df = test_heading_info_df[
            test_heading_info_df['data_name'] == data_name]
        session_ctrl_heading_info_df = ctrl_heading_info_df[
            ctrl_heading_info_df['data_name'] == data_name]

        session_median_info = make_pooled_median_info_from_test_and_ctrl_heading_info_df(session_test_heading_info_df,
                                                                                         session_ctrl_heading_info_df,
                                                                                         verbose=verbose,
                                                                                         key_for_split_choices=key_for_split_choices,
                                                                                         whether_filter_info_choices=whether_filter_info_choices,
                                                                                         whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                                                                         whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                                                                         whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                                                                         ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                                                                         max_curv_range_choices=max_curv_range_choices,
                                                                                         allow_single_condition=allow_single_condition,
                                                                                         )

        session_median_info['data_name'] = data_name
        per_sess_median_info = pd.concat(
            [per_sess_median_info, session_median_info], axis=0)

    per_sess_median_info = per_sess_median_info.reset_index(drop=True)

    # Map sorted data_names to session IDs
    per_sess_median_info = assign_session_id(per_sess_median_info)

    return per_sess_median_info


def make_pooled_perc_info_from_test_and_ctrl_heading_info_df(test_heading_info_df,
                                                             ctrl_heading_info_df,
                                                             verbose=True,
                                                             key_for_split_choices=[
                                                                 'ff_seen'],
                                                             whether_filter_info_choices=[
                                                                 True],
                                                             whether_even_out_distribution_choices=[
                                                                 False],
                                                             whether_test_nxt_ff_flash_after_stop_choices=[
                                                                 'yes', 'no', 'flexible'],
                                                             # whether_test_nxt_ff_flash_after_stop_choices=['flexible'],
                                                             whether_limit_cur_ff_cluster_50_size_choices=[
                                                                 False],
                                                             ctrl_flash_compared_to_test_choices=[
                                                                 'flexible'],
                                                             max_curv_range_choices=[
                                                                 200],
                                                             allow_single_condition=False,
                                                             ):

    test_heading_info_df = build_factor_comp.process_heading_info_df(
        test_heading_info_df)
    ctrl_heading_info_df = build_factor_comp.process_heading_info_df(
        ctrl_heading_info_df)

    test_heading_info_df = add_dir_from_cur_ff(test_heading_info_df)
    ctrl_heading_info_df = add_dir_from_cur_ff(ctrl_heading_info_df)

    pooled_perc_info = make_regrouped_info(test_heading_info_df,
                                           ctrl_heading_info_df,
                                           make_temp_perc_info_func,
                                           key_for_split_choices=key_for_split_choices,
                                           whether_filter_info_choices=whether_filter_info_choices,
                                           whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                           whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                           whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                           ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                           max_curv_range_choices=max_curv_range_choices,
                                           verbose=verbose,
                                           allow_single_condition=allow_single_condition)

    return pooled_perc_info


def make_per_sess_perc_info_from_test_and_ctrl_heading_info_df(test_heading_info_df,
                                                               ctrl_heading_info_df,
                                                               verbose=True,
                                                               key_for_split_choices=[
                                                                   'ff_seen'],
                                                               whether_filter_info_choices=[
                                                                   True],
                                                               whether_even_out_distribution_choices=[
                                                                   False],
                                                               whether_test_nxt_ff_flash_after_stop_choices=[
                                                                   'yes', 'no', 'flexible'],
                                                               whether_limit_cur_ff_cluster_50_size_choices=[
                                                                   False],
                                                               ctrl_flash_compared_to_test_choices=[
                                                                   'flexible'],
                                                               max_curv_range_choices=[
                                                                   200],
                                                               allow_single_condition=False,
                                                               ):

    per_sess_perc_info = pd.DataFrame()
    for data_name in test_heading_info_df['data_name'].unique():
        print(f'Processing data_name: {data_name}')
        session_test_heading_info_df = test_heading_info_df[
            test_heading_info_df['data_name'] == data_name]
        session_ctrl_heading_info_df = ctrl_heading_info_df[
            ctrl_heading_info_df['data_name'] == data_name]

        session_perc_info = make_pooled_perc_info_from_test_and_ctrl_heading_info_df(session_test_heading_info_df,
                                                                                     session_ctrl_heading_info_df,
                                                                                     verbose=verbose,
                                                                                     key_for_split_choices=key_for_split_choices,
                                                                                     whether_filter_info_choices=whether_filter_info_choices,
                                                                                     whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                                                                     whether_test_nxt_ff_flash_after_stop_choices=whether_test_nxt_ff_flash_after_stop_choices,
                                                                                     whether_limit_cur_ff_cluster_50_size_choices=whether_limit_cur_ff_cluster_50_size_choices,
                                                                                     ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                                                                     max_curv_range_choices=max_curv_range_choices,
                                                                                     allow_single_condition=allow_single_condition,
                                                                                     )

        session_perc_info['data_name'] = data_name
        per_sess_perc_info = pd.concat(
            [per_sess_perc_info, session_perc_info], axis=0)

    per_sess_perc_info = per_sess_perc_info.reset_index(drop=True)

    return per_sess_perc_info


def add_dir_from_cur_ff(df):
    """Add direction columns from current firefly, handling empty dataframes and missing columns."""
    if len(df) == 0:
        # For empty dataframes, just add the columns if they don't exist
        if 'dir_from_cur_ff_to_stop' not in df.columns:
            df['dir_from_cur_ff_to_stop'] = pd.Series(dtype=float)
        if 'dir_from_cur_ff_to_nxt_ff' not in df.columns:
            df['dir_from_cur_ff_to_nxt_ff'] = pd.Series(dtype=float)
        return df
    
    # For non-empty dataframes, add columns if source columns exist
    if 'angle_from_cur_ff_to_stop' in df.columns:
        df['dir_from_cur_ff_to_stop'] = np.sign(df['angle_from_cur_ff_to_stop'])
    else:
        df['dir_from_cur_ff_to_stop'] = np.nan
    
    if 'angle_from_cur_ff_to_nxt_ff' in df.columns:
        df['dir_from_cur_ff_to_nxt_ff'] = np.sign(df['angle_from_cur_ff_to_nxt_ff'])
    else:
        df['dir_from_cur_ff_to_nxt_ff'] = np.nan
    
    return df


def extract_key_info_from_stat_df(stat_df, metrics=['Q1', 'median', 'Q3']):
    current_row = pd.DataFrame([])
    for metric in metrics:
        col_to_add = stat_df.loc[[metric], :].reset_index(drop=True)
        col_to_add.columns = [col + '_' + metric for col in col_to_add.columns]
        current_row = pd.concat([current_row, col_to_add], axis=1)
    return current_row


def get_medians_from_test_and_ctrl(test_heading_info_df, ctrl_heading_info_df,
                                   columns_to_get_metrics=[
                                       'diff_in_angle_to_nxt_ff', 'diff_in_abs_angle_to_nxt_ff', 'diff_in_abs_d_curv'],
                                   # metrics=['Q1', 'median', 'Q3']
                                   metrics=['median'],
                                   ):

    # Filter to only columns that exist in the dataframes
    if len(test_heading_info_df) > 0:
        test_cols_available = [col for col in columns_to_get_metrics if col in test_heading_info_df.columns]
    else:
        test_cols_available = []
    
    if len(ctrl_heading_info_df) > 0:
        ctrl_cols_available = [col for col in columns_to_get_metrics if col in ctrl_heading_info_df.columns]
    else:
        ctrl_cols_available = []
    
    # Handle empty dataframes
    if len(test_heading_info_df) == 0 or len(test_cols_available) == 0:
        # Create empty statistics with NaN values
        test_stat = pd.DataFrame({col: [np.nan] * 8 for col in columns_to_get_metrics},
                                 index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        test_stat = test_stat.rename(index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
    else:
        test_stat = test_heading_info_df[test_cols_available].describe().rename(
            index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
        # Add NaN for missing columns
        for col in columns_to_get_metrics:
            if col not in test_cols_available:
                test_stat[col] = np.nan
    
    if len(ctrl_heading_info_df) == 0 or len(ctrl_cols_available) == 0:
        # Create empty statistics with NaN values
        ctrl_stat = pd.DataFrame({col: [np.nan] * 8 for col in columns_to_get_metrics},
                                 index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        ctrl_stat = ctrl_stat.rename(index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
    else:
        ctrl_stat = ctrl_heading_info_df[ctrl_cols_available].describe().rename(
            index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'})
        # Add NaN for missing columns
        for col in columns_to_get_metrics:
            if col not in ctrl_cols_available:
                ctrl_stat[col] = np.nan

    test_row = extract_key_info_from_stat_df(test_stat, metrics=metrics)
    ctrl_row = extract_key_info_from_stat_df(ctrl_stat, metrics=metrics)

    test_row['test_or_control'] = 'test'
    ctrl_row['test_or_control'] = 'control'

    test_row['sample_size'] = len(test_heading_info_df)
    ctrl_row['sample_size'] = len(ctrl_heading_info_df)

    if 'diff_in_abs_d_curv' in columns_to_get_metrics:
        if len(test_heading_info_df) > 0 and 'diff_in_abs_d_curv' in test_heading_info_df.columns:
            test_row['sample_size_for_curv'] = len(
                test_heading_info_df[~test_heading_info_df['diff_in_abs_d_curv'].isna()])
        else:
            test_row['sample_size_for_curv'] = 0
            
        if len(ctrl_heading_info_df) > 0 and 'diff_in_abs_d_curv' in ctrl_heading_info_df.columns:
            ctrl_row['sample_size_for_curv'] = len(
                ctrl_heading_info_df[~ctrl_heading_info_df['diff_in_abs_d_curv'].isna()])
        else:
            ctrl_row['sample_size_for_curv'] = 0

    return test_row, ctrl_row


def get_bootstrap_median_std(array, bootstrap_sample_size=5000):
    sample_size = len(array)

    # Generate all bootstrap samples at once
    bootstrap_samples = np.random.choice(
        array, (bootstrap_sample_size, sample_size), replace=True)

    # Calculate the median of each bootstrap sample
    bootstrap_medians = np.median(bootstrap_samples, axis=1)

    # Calculate the standard deviation of the bootstrap medians
    bootstrap_median_std = np.std(bootstrap_medians)

    return bootstrap_median_std


def get_bca_confidence_interval(array, bootstrap_sample_size=5000, confidence_level=0.95, random_state=None):
    """
    Calculate Bias-Corrected and Accelerated (BCa) confidence interval for the median.

    Parameters:
    -----------
    array : array-like
        Input data array
    bootstrap_sample_size : int, default=5000
        Number of bootstrap samples to generate
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (lower_bound, upper_bound) of the BCa confidence interval
    """
    from scipy import stats

    if random_state is not None:
        np.random.seed(random_state)

    array = np.asarray(array)
    array = array[~np.isnan(array)]  # Remove NaN values

    if len(array) == 0:
        return np.nan, np.nan

    n = len(array)
    original_median = np.median(array)

    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(
        array, (bootstrap_sample_size, n), replace=True)

    # Calculate bootstrap medians
    bootstrap_medians = np.median(bootstrap_samples, axis=1)

    # Calculate bias correction (z0)
    # Proportion of bootstrap medians less than original median
    bias_correction = stats.norm.ppf(
        np.mean(bootstrap_medians < original_median))

    # Calculate acceleration (a) using jackknife
    jackknife_medians = []
    for i in range(n):
        jackknife_sample = np.concatenate([array[:i], array[i+1:]])
        jackknife_medians.append(np.median(jackknife_sample))

    jackknife_medians = np.array(jackknife_medians)
    jackknife_mean = np.mean(jackknife_medians)

    # Calculate acceleration
    numerator = np.sum((jackknife_mean - jackknife_medians) ** 3)
    denominator = 6 * \
        (np.sum((jackknife_mean - jackknife_medians) ** 2)) ** (3/2)

    if denominator == 0:
        acceleration = 0
    else:
        acceleration = numerator / denominator

    # Calculate BCa confidence interval
    alpha = 1 - confidence_level
    z_alpha_2 = stats.norm.ppf(alpha / 2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

    # BCa adjustment
    z_lower = bias_correction + \
        (bias_correction + z_alpha_2) / \
        (1 - acceleration * (bias_correction + z_alpha_2))
    z_upper = bias_correction + (bias_correction + z_1_alpha_2) / \
        (1 - acceleration * (bias_correction + z_1_alpha_2))

    # Convert to percentiles
    p_lower = stats.norm.cdf(z_lower)
    p_upper = stats.norm.cdf(z_upper)

    # Ensure percentiles are within [0, 1]
    p_lower = max(0, min(1, p_lower))
    p_upper = max(0, min(1, p_upper))

    # Get confidence interval bounds
    lower_bound = np.percentile(bootstrap_medians, 100 * p_lower)
    upper_bound = np.percentile(bootstrap_medians, 100 * p_upper)

    return lower_bound, upper_bound


def add_bootstrap_bca_ci_to_df(
    test_heading_info_df,
    ctrl_heading_info_df,
    test_row,
    ctrl_row,
    columns,
    stat_fn=np.median,
    confidence_level=0.95,
    random_state=None
):
    """
    Add BCa confidence interval bounds to dataframe rows.

    Parameters
    ----------
    test_heading_info_df : pandas.DataFrame
        Test condition data
    ctrl_heading_info_df : pandas.DataFrame
        Control condition data
    test_row : pandas.Series
        Row containing test condition statistics
    ctrl_row : pandas.Series
        Row containing control condition statistics
    columns : list
        Column names to calculate BCa intervals for
    confidence_level : float, default=0.95
        Confidence level for the intervals
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    tuple
        (test_row, ctrl_row) with BCa CI bounds added
    """
    test_row = test_row.copy()
    ctrl_row = ctrl_row.copy()
    ci_tag = int(confidence_level * 100)

    # Only compute CIs if dataframe has data
    if len(test_heading_info_df) > 0:
        for column in columns:
            if column in test_heading_info_df.columns:
                series = test_heading_info_df[column].dropna().values
                if len(series) > 0:
                    ci_low, ci_high = bootstrap_bca_ci(
                        series,
                        stat_fn=stat_fn,
                        confidence_level=confidence_level,
                        random_state=random_state
                    )
                else:
                    ci_low, ci_high = np.nan, np.nan
            else:
                ci_low, ci_high = np.nan, np.nan
            test_row[f'{column}_ci_low_{ci_tag}'] = ci_low
            test_row[f'{column}_ci_high_{ci_tag}'] = ci_high
    else:
        # Empty dataframe - set NaN for all CIs
        for column in columns:
            test_row[f'{column}_ci_low_{ci_tag}'] = np.nan
            test_row[f'{column}_ci_high_{ci_tag}'] = np.nan

    if len(ctrl_heading_info_df) > 0:
        for column in columns:
            if column in ctrl_heading_info_df.columns:
                series = ctrl_heading_info_df[column].dropna().values
                if len(series) > 0:
                    ci_low, ci_high = bootstrap_bca_ci(
                        series,
                        stat_fn=stat_fn,
                        confidence_level=confidence_level,
                        random_state=random_state
                    )
                else:
                    ci_low, ci_high = np.nan, np.nan
            else:
                ci_low, ci_high = np.nan, np.nan
            ctrl_row[f'{column}_ci_low_{ci_tag}'] = ci_low
            ctrl_row[f'{column}_ci_high_{ci_tag}'] = ci_high
    else:
        # Empty dataframe - set NaN for all CIs
        for column in columns:
            ctrl_row[f'{column}_ci_low_{ci_tag}'] = np.nan
            ctrl_row[f'{column}_ci_high_{ci_tag}'] = np.nan

    return test_row, ctrl_row


# --- Core BCa engine (no duplication lives above this line) ---


def bca_interval_from_boot_jack(stat_obs, boot_stats, jackknife_stats, confidence_level=0.95):
    """Core BCa interval computation (reused everywhere)."""
    boot_stats = np.asarray(boot_stats, float)
    jackknife_stats = np.asarray(jackknife_stats, float)
    if boot_stats.size == 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence_level

    # Bias-correction z0
    prop_less = np.mean(boot_stats < stat_obs)
    eps = 1.0 / (len(boot_stats) * 2.0)
    prop_less = np.clip(prop_less, eps, 1 - eps)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration a
    j_mean = jackknife_stats.mean()
    num = np.sum((j_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((j_mean - jackknife_stats) ** 2) ** 1.5)
    a = 0.0 if den == 0.0 else num / den

    # Adjusted quantiles
    z_lo, z_hi = stats.norm.ppf([alpha/2, 1 - alpha/2])
    adj_lo = z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))
    adj_hi = z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))
    q_lo, q_hi = stats.norm.cdf([adj_lo, adj_hi])
    q_lo, q_hi = np.clip([q_lo, q_hi], 0, 1)
    if q_lo > q_hi:
        q_lo, q_hi = q_hi, q_lo

    return (np.quantile(boot_stats, q_lo, method='linear'),
            np.quantile(boot_stats, q_hi, method='linear'))


def bootstrap_bca_ci(x, stat_fn=np.median, bootstrap_sample_size=5000,
                     confidence_level=0.95, random_state=None):
    """
    Generic BCa bootstrap CI for any 1-sample statistic.

    Parameters
    ----------
    x : array-like
        Input data (will drop NaNs)
    stat_fn : callable, default=np.median
        Statistic function (e.g., np.median, np.mean, np.var)
    bootstrap_sample_size : int, default=5000
        Number of bootstrap replicates
    confidence_level : float, default=0.95
        CI coverage
    random_state : int or None
        RNG seed

    Returns
    -------
    (ci_low, ci_high)
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    stat_obs = stat_fn(x)

    # Bootstrap replicates
    idx = rng.integers(0, n, size=(bootstrap_sample_size, n))
    boot_stats = [stat_fn(x[i]) for i in idx]

    # Jackknife replicates
    if n >= 2:
        jack_stats = [stat_fn(np.delete(x, i)) for i in range(n)]
    else:
        jack_stats = [stat_obs]

    return bca_interval_from_boot_jack(stat_obs, boot_stats, jack_stats, confidence_level)


def get_bootstrap_diff_bca_ci(
    x,
    y,
    bootstrap_sample_size=5000,
    confidence_level=0.95,
    random_state=None,
    stat_fn=np.median
):
    """
    BCa CI for difference of statistics: stat_fn(x) - stat_fn(y).
    By default stat_fn=np.median. Returns (ci_low, ci_high).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)

    stat_x, stat_y = stat_fn(x), stat_fn(y)
    stat_obs = float(stat_x - stat_y)

    # Bootstrap difference (independent resampling)
    idx_x = rng.integers(0, n_x, size=(bootstrap_sample_size, n_x))
    idx_y = rng.integers(0, n_y, size=(bootstrap_sample_size, n_y))
    boot_stats = stat_fn(x[idx_x], axis=1) - stat_fn(y[idx_y], axis=1)

    # Jackknife over both samples
    jk = []
    if n_x >= 2:
        for i in range(n_x):
            jk_x = stat_fn(np.delete(x, i))
            jk.append(jk_x - stat_y)
    else:
        jk.append(stat_obs)
    if n_y >= 2:
        for j in range(n_y):
            jk_y = stat_fn(np.delete(y, j))
            jk.append(stat_x - jk_y)
    else:
        jk.append(stat_obs)

    jackknife_stats = np.asarray(jk, float)

    return bca_interval_from_boot_jack(stat_obs, boot_stats, jackknife_stats, confidence_level)

# --- Your higher-level function stays tiny and calls the wrapper ---


def calculate_difference_with_bca_ci(
    test_heading_info_df,
    ctrl_heading_info_df,
    columns,
    confidence_level=0.95,
    random_state=None,
    n_boot=5000,
    stat_fn=np.median
):
    """
    Difference of medians (test - control) with BCa CIs for each column.
    Returns a Series with median_diff, ci_low_XX, ci_high_XX.
    """
    out = {}
    tag = int(confidence_level * 100)

    for col in columns:
        # Check if column exists in both dataframes before accessing
        col_in_test = col in test_heading_info_df.columns if len(test_heading_info_df) > 0 else False
        col_in_ctrl = col in ctrl_heading_info_df.columns if len(ctrl_heading_info_df) > 0 else False
        
        if not col_in_test or not col_in_ctrl:
            # Column doesn't exist - set NaN
            out[f'{col}_median'] = np.nan
            out[f'{col}_ci_low_{tag}'] = np.nan
            out[f'{col}_ci_high_{tag}'] = np.nan
            continue
        
        x = test_heading_info_df[col].dropna().to_numpy()
        y = ctrl_heading_info_df[col].dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            out[f'{col}_median'] = np.nan
            out[f'{col}_ci_low_{tag}'] = np.nan
            out[f'{col}_ci_high_{tag}'] = np.nan
            continue

        diff_in_stat = float(stat_fn(x) - stat_fn(y))
        lo, hi = get_bootstrap_diff_bca_ci(
            x, y,
            bootstrap_sample_size=n_boot,
            confidence_level=confidence_level,
            random_state=random_state,
            stat_fn=stat_fn
        )

        if stat_fn == np.median:
            out[f'{col}_median'] = diff_in_stat
        elif stat_fn == np.mean:
            out[f'{col}_mean'] = diff_in_stat
        out[f'{col}_ci_low_{tag}'] = lo
        out[f'{col}_ci_high_{tag}'] = hi

    return pd.Series(out, dtype=float)


def make_temp_median_info_func(test_heading_info_df, ctrl_heading_info_df, confidence_level=0.95, random_state=None):
    """
    Create temporary median info with either standard bootstrap SE or BCa confidence intervals.

    Parameters:
    -----------
    test_heading_info_df : pandas.DataFrame
        Test condition data
    ctrl_heading_info_df : pandas.DataFrame
        Control condition data
    confidence_level : float, default=0.95
        Confidence level for BCa intervals
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Combined dataframe with median statistics and error estimates
    """
    test_row, ctrl_row = get_medians_from_test_and_ctrl(
        test_heading_info_df, ctrl_heading_info_df)

    test_row, ctrl_row = add_bootstrap_bca_ci_to_df(
        test_heading_info_df, ctrl_heading_info_df,
        test_row, ctrl_row,
        columns=['diff_in_abs_angle_to_nxt_ff', 'diff_in_abs_d_curv'],
        confidence_level=confidence_level,
        stat_fn=np.median,
        random_state=random_state
    )

    difference_row = calculate_difference_with_bca_ci(
        test_heading_info_df, ctrl_heading_info_df,
        columns=['diff_in_abs_angle_to_nxt_ff', 'diff_in_abs_d_curv'],
        confidence_level=confidence_level,
        random_state=random_state,
        stat_fn=np.median
    )
    # Add test_or_control identifier for difference
    difference_row['test_or_control'] = 'difference'
    
    # Handle sample sizes safely
    test_sample = test_row['sample_size'].item() if 'sample_size' in test_row.columns else 0
    ctrl_sample = ctrl_row['sample_size'].item() if 'sample_size' in ctrl_row.columns else 0
    difference_row['sample_size'] = min(test_sample, ctrl_sample) if test_sample > 0 and ctrl_sample > 0 else 0
    
    test_sample_curv = test_row['sample_size_for_curv'].item() if 'sample_size_for_curv' in test_row.columns else 0
    ctrl_sample_curv = ctrl_row['sample_size_for_curv'].item() if 'sample_size_for_curv' in ctrl_row.columns else 0
    difference_row['sample_size_for_curv'] = min(test_sample_curv, ctrl_sample_curv) if test_sample_curv > 0 and ctrl_sample_curv > 0 else 0

    # Convert to DataFrame and create two identical difference rows
    difference_row = pd.DataFrame([difference_row])

    temp_regrouped_info = pd.concat(
        [test_row, ctrl_row, difference_row], axis=0)

    return temp_regrouped_info



def make_temp_perc_info_func(test_heading_info_df, ctrl_heading_info_df, confidence_level=0.95, random_state=None):
    test_row, ctrl_row = get_perc_from_test_and_ctrl(
        test_heading_info_df, ctrl_heading_info_df)

    test_row, ctrl_row = add_bootstrap_bca_ci_to_df(
        test_heading_info_df, ctrl_heading_info_df,
        test_row, ctrl_row,
        columns=['perc'],
        stat_fn=np.mean,
        confidence_level=confidence_level,
        random_state=random_state
    )

    difference_row = calculate_difference_with_bca_ci(
        test_heading_info_df, ctrl_heading_info_df,
        columns=['perc'],
        confidence_level=confidence_level,
        random_state=random_state,
        stat_fn=np.mean
    )
    
    difference_row['test_or_control'] = 'difference'
    # Handle case where one or both might be 0
    test_sample = test_row['sample_size'].item() if 'sample_size' in test_row.columns else 0
    ctrl_sample = ctrl_row['sample_size'].item() if 'sample_size' in ctrl_row.columns else 0
    difference_row['sample_size'] = min(test_sample, ctrl_sample) if test_sample > 0 and ctrl_sample > 0 else 0
    
    difference_row = pd.DataFrame([difference_row])
    difference_row.rename(columns={'perc_mean': 'perc'}, inplace=True, errors='ignore')

    temp_regrouped_info = pd.concat(
        [test_row, ctrl_row, difference_row], axis=0)

    return temp_regrouped_info



def get_perc_from_test_and_ctrl(test_heading_info_df, ctrl_heading_info_df):
    """Calculate percentage info, handling empty dataframes and missing columns."""
    
    # Handle test dataframe
    if len(test_heading_info_df) > 0 and 'dir_from_cur_ff_to_stop' in test_heading_info_df.columns and 'dir_from_cur_ff_to_nxt_ff' in test_heading_info_df.columns:
        test_heading_info_df['perc'] = test_heading_info_df[
            'dir_from_cur_ff_to_stop'] == test_heading_info_df['dir_from_cur_ff_to_nxt_ff']
        test_perc = (test_heading_info_df['perc']).sum() / len(test_heading_info_df)
        test_sample_size = len(test_heading_info_df)
    else:
        test_perc = np.nan
        test_sample_size = 0

    # Handle control dataframe
    if len(ctrl_heading_info_df) > 0 and 'dir_from_cur_ff_to_stop' in ctrl_heading_info_df.columns and 'dir_from_cur_ff_to_nxt_ff' in ctrl_heading_info_df.columns:
        ctrl_heading_info_df['perc'] = ctrl_heading_info_df[
            'dir_from_cur_ff_to_stop'] == ctrl_heading_info_df['dir_from_cur_ff_to_nxt_ff']
        ctrl_perc = (ctrl_heading_info_df['perc']).sum() / len(ctrl_heading_info_df)
        ctrl_sample_size = len(ctrl_heading_info_df)
    else:
        ctrl_perc = np.nan
        ctrl_sample_size = 0

    test_row = pd.DataFrame(
        {'perc': test_perc, 'sample_size': test_sample_size, 'test_or_control': 'test'}, index=[0])
    ctrl_row = pd.DataFrame(
        {'perc': ctrl_perc, 'sample_size': ctrl_sample_size, 'test_or_control': 'control'}, index=[0])

    return test_row, ctrl_row

def make_variations_df_across_ref_point_values(variation_func,
                                               variation_func_kwargs={},
                                               ref_point_params_based_on_mode={'time after cur ff visible': [0.1, 0],
                                                                               'distance': [-150, -100, -50]},
                                               monkey_name=None,
                                               path_to_save=None,
                                               use_stored_data_only=False,
                                               ):

    all_variations_df = pd.DataFrame()
    variations_list = specific_utils.init_variations_list_func(ref_point_params_based_on_mode,
                                                               monkey_name=monkey_name)

    for index, row in variations_list.iterrows():
        print(row)
        df = variation_func(ref_point_mode=row['ref_point_mode'],
                            ref_point_value=row['ref_point_value'],
                            use_stored_data_only=use_stored_data_only,
                            **variation_func_kwargs)
        all_variations_df = pd.concat([all_variations_df, df], axis=0)
        all_variations_df.reset_index(drop=True, inplace=True)

    if path_to_save is not None:
        # make sure that the directory exists
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        all_variations_df.to_csv(path_to_save, index=False)
        print('Variations saved at:', path_to_save)

    return all_variations_df


def make_combd_planning_info_folder_path(monkey_name):
    combd_planning_info_folder_path = f"all_monkey_data/planning/{monkey_name}/combined_data"
    return combd_planning_info_folder_path


def make_combd_cur_and_nxt_folder_path(monkey_name):
    combd_planning_info_folder_path = make_combd_planning_info_folder_path(
        monkey_name)
    combd_cur_and_nxt_folder_path = os.path.join(
        combd_planning_info_folder_path, 'cur_and_nxt')
    return combd_cur_and_nxt_folder_path


def make_combd_only_cur_ff_path(monkey_name):
    combd_planning_info_folder_path = make_combd_planning_info_folder_path(
        monkey_name)
    combd_only_cur_ff_path = os.path.join(
        combd_planning_info_folder_path, 'only_cur_ff')
    return combd_only_cur_ff_path


def combine_all_ref_per_sess_median_info_across_monkeys_and_opt_arc_types(all_ref_per_sess_median_info_exists_ok=True,
                                                                          pooled_median_info_exists_ok=True):
    all_ref_per_sess_median_info = _combine_all_ref_median_info_across_monkeys_and_opt_arc_types(exists_ok=all_ref_per_sess_median_info_exists_ok,
                                                                                                 pooled_median_info_exists_ok=pooled_median_info_exists_ok,
                                                                                                 per_sess=True,
                                                                                                 )
    return all_ref_per_sess_median_info


def combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types(all_ref_pooled_median_info_exists_ok=True,
                                                                        pooled_median_info_exists_ok=True):
    all_ref_pooled_median_info = _combine_all_ref_median_info_across_monkeys_and_opt_arc_types(exists_ok=all_ref_pooled_median_info_exists_ok,
                                                                                               pooled_median_info_exists_ok=pooled_median_info_exists_ok,
                                                                                               per_sess=False,
                                                                                               )
    return all_ref_pooled_median_info


def _combine_all_ref_median_info_across_monkeys_and_opt_arc_types(exists_ok=True,
                                                                  pooled_median_info_exists_ok=True,
                                                                  per_sess=False,
                                                                  ):
    all_info = pd.DataFrame([])

    for monkey_name in ['monkey_Schro', 'monkey_Bruno']:
        for opt_arc_type in ['norm_opt_arc', 'opt_arc_stop_closest', 'opt_arc_stop_first_vis_bdry']:
            ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name,
                                                                     opt_arc_type=opt_arc_type)
            func = ps.make_or_retrieve_all_ref_pooled_median_info if not per_sess else ps.make_or_retrieve_all_ref_per_sess_median_info
            temp_info = func(exists_ok=exists_ok,
                             pooled_median_info_exists_ok=pooled_median_info_exists_ok,
                             process_info_for_plotting=False
                             )
            all_info = pd.concat(
                [all_info, temp_info], axis=0)
    all_info.reset_index(drop=True, inplace=True)
    return all_info


def combine_pooled_perc_info_across_monkeys(pooled_perc_info_exists_ok=True):
    pooled_perc_info = pd.DataFrame([])
    opt_arc_type = 'norm_opt_arc'  # this doesn't matter for perc info
    for monkey_name in ['monkey_Bruno', 'monkey_Schro']:
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name,
                                                                 opt_arc_type=opt_arc_type
                                                                 )
        temp_pooled_perc_info = ps.make_or_retrieve_pooled_perc_info(exists_ok=pooled_perc_info_exists_ok,
                                                                     )
        pooled_perc_info = pd.concat(
            [pooled_perc_info, temp_pooled_perc_info], axis=0)
    pooled_perc_info.reset_index(drop=True, inplace=True)
    return pooled_perc_info


def combine_per_sess_perc_info_across_monkeys(per_sess_perc_info_exists_ok=True):
    per_sess_perc_info = pd.DataFrame([])
    opt_arc_type = 'norm_opt_arc'  # this doesn't matter for perc info
    for monkey_name in ['monkey_Bruno', 'monkey_Schro']:
        ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name,
                                                                 opt_arc_type=opt_arc_type
                                                                 )
        temp_per_sess_perc_info = ps.make_or_retrieve_per_sess_perc_info(exists_ok=per_sess_perc_info_exists_ok,
                                                                         )
        per_sess_perc_info = pd.concat(
            [per_sess_perc_info, temp_per_sess_perc_info], axis=0)
    per_sess_perc_info.reset_index(drop=True, inplace=True)
    return per_sess_perc_info


# def get_per_sess_median_info_for_ref_point_params(planner, ref_point_mode, ref_point_value):
#     per_sess_median_info = pd.DataFrame()
#     for data_name in planner.test_heading_info_df['data_name'].unique():
#         print(f'Processing data_name: {data_name}')
#         session_test_heading_info_df = planner.test_heading_info_df[planner.test_heading_info_df['data_name'] == data_name]
#         session_ctrl_heading_info_df = planner.ctrl_heading_info_df[planner.ctrl_heading_info_df['data_name'] == data_name]

#         session_median_info = make_pooled_median_info_from_test_and_ctrl_heading_info_df(session_test_heading_info_df,
#                                                                                                         session_ctrl_heading_info_df, verbose=False)

#         session_median_info['data_name'] = data_name
#         per_sess_median_info = pd.concat([per_sess_median_info, session_median_info], axis=0)

#     per_sess_median_info['ref_point_mode'] = ref_point_mode
#     per_sess_median_info['ref_point_value'] = ref_point_value
#     return per_sess_median_info


def assign_session_id(df, new_col_name='session_id'):
    # Map sorted data_names to session IDs
    unique_data_names = sorted(df['data_name'].unique())
    data_name_to_session_id = {name: idx for idx,
                               name in enumerate(unique_data_names)}
    df[new_col_name] = df['data_name'].map(data_name_to_session_id)
    return df
