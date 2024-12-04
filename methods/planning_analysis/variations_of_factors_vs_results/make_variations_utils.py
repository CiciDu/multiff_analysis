
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.plan_factors import plan_factors_utils
from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class, test_vs_control_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from data_wrangling import basic_func
import seaborn as sns
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import pi
import contextlib
import plotly.express as px
import plotly.graph_objects as go




def make_regrouped_info(test_df, 
                        ctrl_df, 
                        agg_regrouped_info_func,
                        agg_regrouped_info_kwargs={},
                        key_for_split_choices=['ff_flash', 'cluster_flash', 'ff_seen', 'cluster_seen'],
                        whether_filter_info_choices=[True, False],
                        whether_even_out_distribution_choices=[True, False],
                        whether_test_alt_ff_flash_after_stop_choices=['yes', 'no', 'flexible'],
                        whether_limit_stop_ff_cluster_50_size_choices=[True, False],
                        ctrl_flash_compared_to_test_choices=['same', 'flexible'],
                        max_curv_range_choices=[75, 100, 125, 150, 200],
                        verbose=False
                        ):
    
    test_and_ctrl_df = pd.concat([test_df, ctrl_df], axis=0)

    regrouped_info = pd.DataFrame()
    column_for_split_dict = {'ff_flash': 'alt_ff_last_flash_time_bbas',
                            'cluster_flash': 'alt_ff_cluster_last_flash_time_bbas',
                            'ff_seen': 'ALT_time_ff_last_seen_bbas',
                            'cluster_seen': 'alt_ff_cluster_last_seen_time_bbas'
                            }

    if_test_alt_ff_group_appear_after_stop_dict = {
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
                for whether_limit_stop_ff_cluster_50_size in whether_limit_stop_ff_cluster_50_size_choices:
                    for whether_test_alt_ff_flash_after_stop in whether_test_alt_ff_flash_after_stop_choices:
                    #for if_test_alt_ff_group_appear_after_stop in ['cluster_must_flash_after_stop', 'cluster_no_flash_after_stop', 'ff_must_flash_after_stop', 'ff_no_flash_after_stop', 'flexible']:
                        if_test_alt_ff_group_appear_after_stop = if_test_alt_ff_group_appear_after_stop_dict[whether_test_alt_ff_flash_after_stop][key_for_split]
                        for ctrl_flash_compared_to_test in ctrl_flash_compared_to_test_choices:
                            if ctrl_flash_compared_to_test == 'same':
                                if if_test_alt_ff_group_appear_after_stop != 'flexible': # to avoid repetition
                                    if_ctrl_alt_ff_group_appear_after_stop = if_test_alt_ff_group_appear_after_stop
                                else:
                                    continue
                            else:
                                if_ctrl_alt_ff_group_appear_after_stop = 'flexible'
                            for max_curv_range in max_curv_range_choices:
                                ctrl_df = test_and_ctrl_df[test_and_ctrl_df[column_for_split].isnull()].copy()
                                test_df = test_and_ctrl_df[~test_and_ctrl_df[column_for_split].isnull()].copy()

                                if (len(test_df) == 0) | (len(ctrl_df) == 0):
                                    continue 

                                if whether_filter_info:
                                    test_df, ctrl_df = test_vs_control_utils.filter_both_df(test_df, ctrl_df, max_curv_range=max_curv_range, verbose=verbose, 
                                                        whether_even_out_distribution=whether_even_out_distribution,
                                                        whether_limit_stop_ff_cluster_50_size=whether_limit_stop_ff_cluster_50_size,
                                                        if_test_alt_ff_group_appear_after_stop=if_test_alt_ff_group_appear_after_stop,
                                                        if_ctrl_alt_ff_group_appear_after_stop=if_ctrl_alt_ff_group_appear_after_stop)
                                elif whether_even_out_distribution: # if not filtering, then only even out the distribution
                                    test_df, ctrl_df = test_vs_control_utils.make_the_distributions_of_distance_more_similar_in_df(test_df, ctrl_df, verbose=verbose)
                                    test_df, ctrl_df = test_vs_control_utils.make_the_distributions_of_angle_more_similar_in_df(test_df, ctrl_df, verbose=verbose)
                                
                                if (len(test_df) > 0) & (len(ctrl_df) > 0):
                                    temp_regrouped_info = agg_regrouped_info_func(test_df, ctrl_df, **agg_regrouped_info_kwargs)
                                else:
                                    temp_regrouped_info = pd.DataFrame()

                                temp_regrouped_info['key_for_split'] = key_for_split
                                temp_regrouped_info['whether_filter_info'] = whether_filter_info
                                temp_regrouped_info['whether_even_out_dist'] = whether_even_out_distribution
                                temp_regrouped_info['whether_limit_stop_ff_cluster_50_size'] = whether_limit_stop_ff_cluster_50_size
                                temp_regrouped_info['if_test_alt_ff_group_appear_after_stop'] = if_test_alt_ff_group_appear_after_stop
                                temp_regrouped_info['if_ctrl_alt_ff_group_appear_after_stop'] = if_ctrl_alt_ff_group_appear_after_stop
                                temp_regrouped_info['ctrl_flash_compared_to_test'] = ctrl_flash_compared_to_test
                                temp_regrouped_info['max_curv_range'] = max_curv_range
                                # temp_regrouped_info['test_sample_size'] = len(test_df)
                                # temp_regrouped_info['ctrl_sample_size'] = len(ctrl_df)
                                regrouped_info = pd.concat([regrouped_info, temp_regrouped_info], axis=0)
                                if not whether_filter_info:
                                    break  # Skip remaining max_curv_range values if not filtering
    regrouped_info.reset_index(drop=True, inplace=True)

    return regrouped_info



def make_all_median_info_from_test_and_ctrl_heading_info_df(test_heading_info_df, 
                                                            ctrl_heading_info_df, 
                                                            verbose=True,
                                                            key_for_split_choices=['ff_seen', 'cluster_seen'],
                                                            whether_filter_info_choices=[True],
                                                            #whether_even_out_distribution_choices=[True, False],
                                                            whether_even_out_distribution_choices=[True, False],
                                                            whether_test_alt_ff_flash_after_stop_choices=['yes', 'no', 'flexible'],
                                                            whether_limit_stop_ff_cluster_50_size_choices=[False],
                                                            ctrl_flash_compared_to_test_choices=['flexible'],
                                                            max_curv_range_choices=[200], 
                                                            ):
    
    test_heading_info_df = plan_factors_utils.process_heading_info_df(test_heading_info_df)
    ctrl_heading_info_df = plan_factors_utils.process_heading_info_df(ctrl_heading_info_df)
    all_median_info = make_regrouped_info(test_heading_info_df,
                                            ctrl_heading_info_df,
                                            make_temp_median_info_func,
                                            key_for_split_choices=key_for_split_choices,
                                            whether_filter_info_choices=whether_filter_info_choices,
                                            whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                            whether_test_alt_ff_flash_after_stop_choices=whether_test_alt_ff_flash_after_stop_choices,
                                            whether_limit_stop_ff_cluster_50_size_choices=whether_limit_stop_ff_cluster_50_size_choices,
                                            ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                            max_curv_range_choices=max_curv_range_choices,                                                                                 
                                            verbose=verbose)   
  
    return all_median_info


def make_all_perc_info_from_test_and_ctrl_heading_info_df(test_heading_info_df, 
                                                            ctrl_heading_info_df, 
                                                            verbose=True,
                                                            key_for_split_choices=['ff_flash', 'cluster_flash', 'ff_seen', 'cluster_seen'],
                                                            whether_filter_info_choices=[True],
                                                            whether_even_out_distribution_choices=[True, False],
                                                            whether_test_alt_ff_flash_after_stop_choices=['yes', 'no', 'flexible'],
                                                            #whether_test_alt_ff_flash_after_stop_choices=['flexible'],
                                                            whether_limit_stop_ff_cluster_50_size_choices=[False],
                                                            ctrl_flash_compared_to_test_choices=['flexible'],
                                                            max_curv_range_choices=[200],
                                                            ):
    
    test_heading_info_df = plan_factors_utils.process_heading_info_df(test_heading_info_df)
    ctrl_heading_info_df = plan_factors_utils.process_heading_info_df(ctrl_heading_info_df)

    test_heading_info_df['dir_from_stop_ff_to_stop'] = np.sign(test_heading_info_df['angle_from_stop_ff_to_stop'])
    test_heading_info_df['dir_from_stop_ff_to_alt_ff'] = np.sign(test_heading_info_df['angle_from_stop_ff_to_alt_ff'])

    ctrl_heading_info_df['dir_from_stop_ff_to_stop'] = np.sign(ctrl_heading_info_df['angle_from_stop_ff_to_stop'])
    ctrl_heading_info_df['dir_from_stop_ff_to_alt_ff'] = np.sign(ctrl_heading_info_df['angle_from_stop_ff_to_alt_ff'])

    all_perc_info = make_regrouped_info(test_heading_info_df,
                                        ctrl_heading_info_df,
                                        make_temp_perc_info_func,   
                                        key_for_split_choices=key_for_split_choices,
                                        whether_filter_info_choices=whether_filter_info_choices,
                                        whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                        whether_test_alt_ff_flash_after_stop_choices=whether_test_alt_ff_flash_after_stop_choices,
                                        whether_limit_stop_ff_cluster_50_size_choices=whether_limit_stop_ff_cluster_50_size_choices,
                                        ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                        max_curv_range_choices=max_curv_range_choices,   
                                        verbose=verbose) 
    
    return all_perc_info


def extract_key_info_from_stat_df(stat_df, metrics=['25%', '50%', '75%']):
    current_row = pd.DataFrame([])
    for metric in metrics:
        col_to_add = stat_df.loc[[metric], :].reset_index(drop=True)
        col_to_add.columns = [col + '_' + metric for col in col_to_add.columns]
        current_row = pd.concat([current_row, col_to_add], axis=1)
    return current_row

def get_rows_from_test_and_ctrl(test_heading_info_df, ctrl_heading_info_df,
                                columns_to_get_metrics=['diff', 'diff_in_abs', 'diff_in_abs_d_curv'],
                                metrics=['25%', '50%', '75%']):

    test_stat = test_heading_info_df[columns_to_get_metrics].describe()
    ctrl_stat = ctrl_heading_info_df[columns_to_get_metrics].describe()

    row_from_test = extract_key_info_from_stat_df(test_stat, metrics=metrics)
    row_from_ctrl = extract_key_info_from_stat_df(ctrl_stat, metrics=metrics)

    row_from_test['test_or_control'] = 'test'
    row_from_ctrl['test_or_control'] = 'control'

    row_from_test['sample_size'] = len(test_heading_info_df)
    row_from_ctrl['sample_size'] = len(ctrl_heading_info_df)

    if 'diff_in_abs_d_curv' in columns_to_get_metrics:
        row_from_test['sample_size_for_curv'] = len(test_heading_info_df[~test_heading_info_df['diff_in_abs_d_curv'].isna()])
        row_from_ctrl['sample_size_for_curv'] = len(ctrl_heading_info_df[~ctrl_heading_info_df['diff_in_abs_d_curv'].isna()])

    return row_from_test, row_from_ctrl


def get_delta_values_between_test_and_control(test_heading_info_df, ctrl_heading_info_df):
    diff_and_ratio_stat_df = show_planning_utils.make_diff_and_ratio_stat_df(test_heading_info_df, ctrl_heading_info_df)
    
    diff_and_ratio_stat_df['delta_diff_in_abs'] = diff_and_ratio_stat_df['test_diff_in_abs'] - diff_and_ratio_stat_df['ctrl_diff_in_abs']
    diff_and_ratio_stat_df['delta_diff'] = diff_and_ratio_stat_df['test_diff'] - diff_and_ratio_stat_df['ctrl_diff']  
    diff_and_ratio_stat_df['delta_diff_in_abs_d_curv'] = diff_and_ratio_stat_df['test_diff_in_abs_d_curv'] - diff_and_ratio_stat_df['ctrl_diff_in_abs_d_curv']
    new_diff_and_ratio_stat_df = diff_and_ratio_stat_df[['delta_diff_in_abs', 'delta_diff', 'delta_diff_in_abs_d_curv']].copy()

    delta_row = extract_key_info_from_stat_df(new_diff_and_ratio_stat_df, metrics=['50%'])
    return delta_row


def get_bootstrap_median_std(array, bootstrap_sample_size=5000):
    sample_size = len(array)
    
    # Generate all bootstrap samples at once
    bootstrap_samples = np.random.choice(array, (bootstrap_sample_size, sample_size), replace=True)
    
    # Calculate the median of each bootstrap sample
    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    
    # Calculate the standard deviation of the bootstrap medians
    bootstrap_median_std = np.std(bootstrap_medians)
    
    return bootstrap_median_std


def add_boostrap_median_std_to_df(test_heading_info_df, ctrl_heading_info_df, 
                                  row_from_test, row_from_ctrl, columns):
    row_from_test = row_from_test.copy()
    row_from_ctrl = row_from_ctrl.copy()
    
    for column in columns:
        series = test_heading_info_df[column]
        boot_med_std = get_bootstrap_median_std(series[~series.isna()].values)
        row_from_test[f'{column}_boot_med_std'] = boot_med_std

    for column in columns:
        series = ctrl_heading_info_df[column]
        boot_med_std = get_bootstrap_median_std(series[~series.isna()].values)
        row_from_ctrl[f'{column}_boot_med_std'] = boot_med_std

    return row_from_test, row_from_ctrl


def make_temp_median_info_func(test_heading_info_df, ctrl_heading_info_df):
    row_from_test, row_from_ctrl = get_rows_from_test_and_ctrl(test_heading_info_df, ctrl_heading_info_df)
    row_from_test, row_from_ctrl = add_boostrap_median_std_to_df(test_heading_info_df, ctrl_heading_info_df,
                                                                row_from_test, row_from_ctrl, 
                                                                columns=['diff_in_abs', 'diff_in_abs_d_curv'])

    delta_row = get_delta_values_between_test_and_control(test_heading_info_df, ctrl_heading_info_df)
    
    delta_rows = pd.concat([delta_row, delta_row], axis=0) 
    temp_regrouped_info = pd.concat([row_from_test, row_from_ctrl], axis=0)
    temp_regrouped_info = pd.concat([temp_regrouped_info, delta_rows], axis=1)
    return temp_regrouped_info
    

def make_temp_perc_info_func(test_heading_info_df, ctrl_heading_info_df):
    test_perc = (test_heading_info_df['dir_from_stop_ff_to_stop']==test_heading_info_df['dir_from_stop_ff_to_alt_ff']).sum()/len(test_heading_info_df)
    ctrl_perc = (ctrl_heading_info_df['dir_from_stop_ff_to_stop']==ctrl_heading_info_df['dir_from_stop_ff_to_alt_ff']).sum()/len(ctrl_heading_info_df)
    diff_in_perc = test_perc - ctrl_perc

    test_sample_size = len(test_heading_info_df)
    ctrl_sample_size = len(ctrl_heading_info_df)

    test_row = pd.DataFrame({'perc': test_perc, 'sample_size': test_sample_size, 'test_or_control': 'test'}, index=[0])
    ctrl_row = pd.DataFrame({'perc': ctrl_perc, 'sample_size': ctrl_sample_size, 'test_or_control': 'control'}, index=[0])

    temp_regrouped_info = pd.concat([test_row, ctrl_row], axis=0).reset_index(drop=True)
    temp_regrouped_info['perc_se'] = np.sqrt(temp_regrouped_info['perc']*(1-temp_regrouped_info['perc'])/temp_regrouped_info['sample_size'])
    temp_regrouped_info['diff_in_perc'] = diff_in_perc

    return temp_regrouped_info


# def make_temp_perc_info_func(test_heading_info_df, ctrl_heading_info_df):
#     test_perc = (test_heading_info_df['dir_from_stop_ff_to_stop']==test_heading_info_df['dir_from_stop_ff_to_alt_ff']).sum()/len(test_heading_info_df)
#     ctrl_perc = (ctrl_heading_info_df['dir_from_stop_ff_to_stop']==ctrl_heading_info_df['dir_from_stop_ff_to_alt_ff']).sum()/len(ctrl_heading_info_df)
#     test_sample_size = len(test_heading_info_df)
#     ctrl_sample_size = len(ctrl_heading_info_df)

#     temp_regrouped_info = pd.DataFrame({'test_perc': test_perc,
#                                         'ctrl_perc': ctrl_perc,
#                                         'test_sample_size': test_sample_size,
#                                         'ctrl_sample_size': ctrl_sample_size
#                                         }, index=[0])
#     return temp_regrouped_info



def make_variations_df_across_ref_point_values(variation_func,
                                                variation_func_kwargs={},
                                                ref_point_params_based_on_mode = {'time after stop ff visible': [0.1, 0],
                                                                                    'distance': [-150, -100, -50]},                                  
                                                monkey_name = None,
                                                path_to_save = None,
                                                ):

    all_variations_df = pd.DataFrame()
    variations_list = basic_func.init_variations_list_func(ref_point_params_based_on_mode,
                                                                monkey_name=monkey_name)

    for index, row in variations_list.iterrows():
        print(row)
        df = variation_func(ref_point_mode=row['ref_point_mode'], 
                            ref_point_value=row['ref_point_value'],
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
    combd_planning_info_folder_path = f"all_monkey_data/planning/individual_monkey_data/{monkey_name}/combined_data"
    return combd_planning_info_folder_path


def make_combd_stop_and_alt_folder_path(monkey_name):
    combd_planning_info_folder_path = make_combd_planning_info_folder_path(monkey_name)
    combd_stop_and_alt_folder_path = os.path.join(combd_planning_info_folder_path, 'stop_and_alt')
    return combd_stop_and_alt_folder_path

def make_combd_only_stop_ff_path(monkey_name):
    combd_planning_info_folder_path = make_combd_planning_info_folder_path(monkey_name)
    combd_only_stop_ff_path = os.path.join(combd_planning_info_folder_path, 'only_stop_ff')
    return combd_only_stop_ff_path

def combine_overall_median_info_across_monkeys_and_optimal_arc_types(overall_median_info_exists_ok=True,
                                                                all_median_info_exists_ok=True):
    overall_median_info = pd.DataFrame([])
    for monkey_name in ['monkey_Schro', 'monkey_Bruno']:
        for optimal_arc_type in ['norm_opt_arc', 'opt_arc_stop_closest', 'opt_arc_stop_first_vis_bdry']:
            #suppress printed output
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name, 
                                                                        optimal_arc_type=optimal_arc_type)
                temp_overall_median_info = ps.make_or_retrieve_overall_median_info(exists_ok=overall_median_info_exists_ok, 
                                                                                        all_median_info_exists_ok=all_median_info_exists_ok,
                                                                                        process_info_for_plotting=False
                                                                                        )
                overall_median_info = pd.concat([overall_median_info, temp_overall_median_info], axis=0)
    overall_median_info.reset_index(drop=True, inplace=True)                
    return overall_median_info


def combine_all_perc_info_across_monkeys(all_perc_info_exists_ok=True):
    all_perc_info = pd.DataFrame([])
    curv_traj_window_before_stop = [-50, 0]
    optimal_arc_type = 'norm_opt_arc'
    for monkey_name in ['monkey_Bruno', 'monkey_Schro']:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name, 
                                                                    optimal_arc_type=optimal_arc_type
                                                                    )
            temp_all_perc_info = ps.make_or_retrieve_all_perc_info(exists_ok=all_perc_info_exists_ok,
                                                                    process_info_for_plotting=False)
            all_perc_info = pd.concat([all_perc_info, temp_all_perc_info], axis=0)
    all_perc_info.reset_index(drop=True, inplace=True)
    return all_perc_info