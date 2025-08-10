
from data_wrangling import specific_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import plot_cvn_utils, find_cvn_utils
from null_behaviors import curv_of_traj_utils, curvature_utils

import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
import math
import os
import sys

import numpy as np

def run_tests_over_monkeys(
    test_func_angle,
    test_func_d_curv,
    ref_point_params,
    monkeys=['monkey_Schro', 'monkey_Bruno'],
    num_permutations=10000,
    verbose=True,
):
    results = []
    
    for monkey_name in monkeys:
        variations_list = specific_utils.init_variations_list_func(
            ref_point_params,
            monkey_name=monkey_name
        )
        
        for _, row in variations_list.iterrows():
            ref_point_mode = row['ref_point_mode']
            ref_point_value = row['ref_point_value']
            if verbose:
                print(row)
            
            # Initialize sessions
            planner = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name)
            planner.initialize_monkey_sessions_df_for_one_monkey()
            planner.get_combd_heading_df_x_sessions_across_sessions(
                ref_point_mode=ref_point_mode,
                ref_point_value=ref_point_value,
                save_data=False
            )
            
            # Process test and control data
            test_df = build_factor_comp.process_heading_info_df(
                planner.combd_heading_df_x_sessions_test.copy()
            )
            ctrl_df = build_factor_comp.process_heading_info_df(
                planner.combd_heading_df_x_sessions_ctrl.copy()
            )
            
            # Run angle test directly on raw data (assumed no filtering needed)
            angle_p = test_func_angle(
                test_df['diff_in_abs_angle_to_nxt_ff'].values,
                ctrl_df['diff_in_abs_angle_to_nxt_ff'].values
            )
            
            # Filter NaNs for d_curv column
            test_df_clean, ctrl_df_clean = filter_and_report_nan(
                test_df, ctrl_df, col_name='diff_in_abs_d_curv'
            )
            
            # Run d_curv test on cleaned data
            d_curv_p = test_func_d_curv(
                test_df_clean['diff_in_abs_d_curv'].values,
                ctrl_df_clean['diff_in_abs_d_curv'].values
            )
            
            # Collect results
            results.append({
                'monkey_name': monkey_name,
                'ref_point_mode': ref_point_mode,
                'ref_point_value': ref_point_value,
                'angle_p_value': angle_p,
                'd_curv_p_value': d_curv_p,
            })
    
    return pd.DataFrame(results)


def permutation_test(
    x, y, 
    num_permutations=10000, 
    alternative='two-sided', 
    statistic='median', 
    random_state=None
):
    """
    General permutation test for difference in means or medians.

    Parameters:
    - x, y: arrays of sample values from two groups
    - num_permutations: number of permutations to perform
    - alternative: 'two-sided', 'greater', or 'less'
    - statistic: 'mean' or 'median'
    - random_state: optional seed for reproducibility

    Returns:
    - p-value of the permutation test
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)
    
    # get rid of nan
    # print percentage of nan and get rid of nan
    print(f'percentage of nan in x: {round(np.sum(np.isnan(x)) / len(x), 3)}')
    print(f'percentage of nan in y: {round(np.sum(np.isnan(y)) / len(y), 3)}')
    print('Removing nan...')
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    observed_diff = stat_func(x) - stat_func(y)
    combined = np.concatenate([x, y])
    
    print(f'observed_diff: {observed_diff}')
    
    count = 0
    for _ in range(num_permutations):
        permuted = rng.permutation(combined)
        x_perm = permuted[:len(x)]
        y_perm = permuted[len(x):]
        perm_diff = stat_func(x_perm) - stat_func(y_perm)
        
        if _ % 5000 == 0:
            print(f'perm_diff: {perm_diff}')

        if alternative == 'two-sided':
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        elif alternative == 'greater':
            if perm_diff >= observed_diff:
                count += 1
        elif alternative == 'less':
            if perm_diff <= observed_diff:
                count += 1
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    p_value = count / num_permutations
    print(f'count: {count}, num_permutations: {num_permutations}, p_value: {p_value}')
    return p_value

def plot_scatter_test_vs_ctrl(test_heading_info_df, ctrl_heading_info_df, arc_col, monk_col, diff_col, diff_in_abs_col):
    # make scatter plot
    for x_col, y_col in [(arc_col, monk_col), (arc_col, diff_col), (arc_col, diff_in_abs_col)]:
        plt.figure(figsize=(6, 4))
        plt.scatter(test_heading_info_df[x_col], test_heading_info_df[y_col])
        plt.scatter(ctrl_heading_info_df[x_col], ctrl_heading_info_df[y_col])
        plt.title(f'{x_col.replace("_", " ").title()} vs. {y_col.replace("_", " ").title()}')
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.legend(['test', 'ctrl'])
        plt.show()
        
        
def plot_hist_test_vs_ctrl(test_heading_info_df, ctrl_heading_info_df, arc_col, monk_col, diff_col, diff_in_abs_col):
    for col in [diff_in_abs_col, diff_col]:
        sns.histplot(data=test_heading_info_df[col], bins=50, label='test', color='blue', kde=True, stat='density')
        sns.histplot(data=ctrl_heading_info_df[col], bins=50, label='ctrl', color='red', kde=True, stat='density')
        # plot vertical lines for medians
        plt.axvline(x=test_heading_info_df[col].median(), color='blue', linestyle='--', label='test median')
        plt.axvline(x=ctrl_heading_info_df[col].median(), color='red', linestyle='--', label='ctrl median')
        plt.title(f'Test vs Control: {col.replace("_", " ").title()}')
        plt.legend()
        plt.show()
        
    for df, name in [(test_heading_info_df, 'Test'), (ctrl_heading_info_df, 'Ctrl')]:
        sns.histplot(data=df[arc_col], bins=50, label=f'{name} Null Arc', color='blue', kde=True, stat='density')
        sns.histplot(data=df[monk_col], bins=50, label=f'{name} Monkey', color='red', kde=True, stat='density')
        # plot vertical lines for medians
        plt.axvline(x=df[arc_col].median(), color='blue', linestyle='--', label=f'null arc median')
        plt.axvline(x=df[monk_col].median(), color='red', linestyle='--', label=f'monkey median')
        plt.title(f'{name}: {arc_col.replace("_", " ").title()} vs. {monk_col.replace("_", " ").title()}')
        plt.legend()
        plt.show()