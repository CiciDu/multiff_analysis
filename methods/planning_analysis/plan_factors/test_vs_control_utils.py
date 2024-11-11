
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.plan_factors import plan_factors_utils
import seaborn as sns
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from math import pi



def change_control_data_to_conform_to_test_data(plan_x_ctrl, plan_y_ctrl, plan_x_test):
    min_angle = plan_x_test['alt_ff_angle_at_ref'].min()
    max_angle = plan_x_test['alt_ff_angle_at_ref'].max()
    plan_x_ctrl = plan_x_ctrl.copy()
    plan_y_ctrl = plan_y_ctrl.copy()
    plan_x_ctrl['unique_id'] = np.arange(len(plan_x_ctrl))
    plan_y_ctrl['unique_id'] = np.arange(len(plan_y_ctrl))
    plan_x_ctrl2 = plan_x_ctrl[plan_x_ctrl['alt_ff_angle_at_ref'].between(min_angle, max_angle)].sort_values(by='unique_id')
    plan_y_ctrl2 = plan_y_ctrl[plan_y_ctrl['unique_id'].isin(plan_x_ctrl2['unique_id'])].sort_values(by='unique_id').drop(columns=['unique_id'])
    plan_x_ctrl2 = plan_x_ctrl2.drop(columns=['unique_id'])
    
    return plan_x_ctrl2, plan_y_ctrl2


def make_the_distributions_of_a_column_more_similar_between_two_df(df_1, df_2, column_name='alt_ff_angle_at_ref', bins=np.arange(-pi, pi, 0.025)):
    df1_length = len(df_1)
    df2_length = len(df_2)
    if (df1_length == 0) | (df2_length == 0):
        print('No data in df_1 or df_2. The function make_the_distributions_of_angle_more_similar_in_df is not used.')
        return df_1, df_2
    
    # put values in df_2[angle_column_name] into bins with width of 0.2, and return the number of items or percentage in each bins
    test_bins_normed = df_2[column_name].value_counts(bins=bins, normalize=True)
    ctrl_bins_normed = df_1[column_name].value_counts(bins=bins, normalize=True)

    # for each bin, take the smaller of the two values
    shared_bins = np.minimum(ctrl_bins_normed, test_bins_normed)

    # Find the number of samples to sample from each bin in the test and control data
    test_sample_ratio = shared_bins * len(df_2)
    ctrl_sample_ratio = shared_bins * len(df_1)

    df_12 = sample_rows_based_on_ratio_for_each_bin(df_1, column_name, ctrl_sample_ratio)
    df_22 = sample_rows_based_on_ratio_for_each_bin(df_2, column_name, test_sample_ratio)
    return df_12, df_22


def prune_out_data_with_large_curv_range(x_df, y_df, max_curv_range=100):

    if ('curv_range' not in y_df.columns):
        raise ValueError('curv_range must be a column in x_df and y_df')

    x_df = x_df.copy()
    y_df = y_df.copy()
    x_df['unique_id'] = np.arange(len(x_df))
    y_df['unique_id'] = np.arange(len(y_df))
    #y_df['curv_range'] = y_df['curv_max'] - y_df['curv_min']
    y_df2 = y_df[y_df['curv_range'] <= max_curv_range].copy()
    x_df2 = x_df[x_df['unique_id'].isin(y_df2['unique_id'])].sort_values(by='unique_id').drop(columns=['unique_id'])
    # print the number of rows dropped out of number of total rows
    print('Number of rows dropped out of total rows after prunning out data with large curv range:', len(x_df) - len(x_df2), 'out of', len(x_df))
    
    y_df2 = y_df2.drop(columns=['unique_id'])
    return x_df2, y_df2


def limit_cum_distance_between_two_stops(x_df, y_df, max_cum_distance_between_two_stops=400):
    if ('cum_distance_between_two_stops' not in x_df.columns):
        try: 
            x_df['cum_distance_between_two_stops'] = x_df['next_stop_cum_distance'] - x_df['stop_cum_distance']
        except:
            raise ValueError('cum_distance_between_two_stops must be a column in x_df and y_df; or else provide the columns stop_cum_distance and next_stop_cum_distance in x_df and y_df')

    x_df = x_df.copy()
    y_df = y_df.copy()
    x_df['unique_id'] = np.arange(len(x_df))
    y_df['unique_id'] = np.arange(len(y_df))
    x_df2 = x_df[x_df['cum_distance_between_two_stops'] <= max_cum_distance_between_two_stops].copy()
    y_df2 = y_df[y_df['unique_id'].isin(x_df2['unique_id'])].sort_values(by='unique_id').drop(columns=['unique_id'])
    x_df2 = x_df2.drop(columns=['unique_id'])
    print('Number of rows dropped out of total rows after limiting cum distance between two stops:', len(x_df) - len(x_df2), 'out of', len(x_df))
    
    return x_df2, y_df2


def sample_rows_based_on_ratio_for_each_bin(df, column, sample_ratio):
    df['unique_id'] = np.arange(len(df))
    test_sampled_rows = pd.DataFrame()
    # sample the corresponding number from each interval from df
    for i in range(len(sample_ratio)):
        if sample_ratio.iloc[i] > 0:
            # we added 0.0005 because value_counts involves rounding to 3 decimals
            rows_to_sample_from = df[(df[column] > sample_ratio.index[i].left) & 
                                     (df[column] <= sample_ratio.index[i].right)]
            # Sometimes because of round errors, there are not enough rows to sample from in the interval
            if len(rows_to_sample_from) < sample_ratio.iloc[i]:
                # But if the difference is lese than 5% of the number of rows to sample, then we just use all the sampled rows 
                # (and be ok with having slightly fewer sampled rows than expected)
                if sample_ratio.iloc[i] - len(rows_to_sample_from) < max(len(rows_to_sample_from) * 0.1, len(df) * 0.05):
                    sampled_rows = rows_to_sample_from
                else:
                    raise ValueError('There are not enough rows to sample from in the interval')
            else:
                sampled_rows = rows_to_sample_from.sample(n=math.floor(sample_ratio.iloc[i]), random_state=0)
            test_sampled_rows = pd.concat([test_sampled_rows, sampled_rows], axis=0)
    if len(test_sampled_rows) > 0:
        test_sampled_rows = test_sampled_rows.sort_values(by='unique_id').reset_index(drop=True).drop(columns=['unique_id'])
    return test_sampled_rows



def limit_alt_ff_flash_after_stop(df, if_alt_ff_group_appear_after_stop, verbose=True):
    length = len(df)
    if if_alt_ff_group_appear_after_stop == 'cluster_must_flash_after_stop':
        df = df[~df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'cluster_no_flash_after_stop':
        df = df[df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'ff_must_flash_after_stop':
        df = df[~df['alt_ff_last_flash_time_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'ff_no_flash_after_stop':
        df = df[df['alt_ff_last_flash_time_bsans'].isnull()].copy() 
    elif if_alt_ff_group_appear_after_stop == 'cluster_must_seen_after_stop':
        df = df[~df['alt_ff_cluster_last_seen_time_bsans'].isnull()].copy() 
    elif if_alt_ff_group_appear_after_stop == 'cluster_no_seen_after_stop':
        df = df[df['alt_ff_cluster_last_seen_time_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'ff_must_seen_after_stop':
        df = df[~df['ALT_time_ff_last_seen_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'ff_no_seen_after_stop':
        df = df[df['ALT_time_ff_last_seen_bsans'].isnull()].copy()
    elif if_alt_ff_group_appear_after_stop == 'flexible':
        pass
    else:
        raise ValueError('if_alt_ff_group_appear_after_stop value invalid')
    if verbose:
        print(f'Number of rows dropped out of total rows in heading_info_df after using if_alt_ff_group_appear_after_stop = {if_alt_ff_group_appear_after_stop} is\
                : {length - len(df)} out of {length}')
    return df



def filter_both_df(test_df, ctrl_df, max_cum_distance_between_two_stops=500, max_curv_range=200, verbose=True, 
                    whether_limit_stop_ff_cluster_50_size=False, 
                    if_test_alt_ff_group_appear_after_stop='flexible',  # can be 'cluster_must_flash_after_stop', 'cluster_no_flash_after_stop', 'ff_must_flash_after_stop', 'ff_no_flash_after_stop'
                    if_ctrl_alt_ff_group_appear_after_stop='flexible',  # or 'cluster_must_seen_after_stop', 'cluster_no_seen_after_stop', 'ff_must_seen_after_stop', 'ff_no_seen_after_stop', 'flexible'
                    whether_even_out_distribution=False):

    test_df, ctrl_df = limit_cum_distance_between_two_stops_in_df(test_df, ctrl_df, max_cum_distance_between_two_stops, verbose=verbose)

    test_df, ctrl_df = limit_the_range_of_alt_ff_angle_at_ref_in_df(test_df, ctrl_df, verbose=verbose)

    test_df, ctrl_df = prune_out_data_with_large_curv_range_in_df(test_df, ctrl_df, max_curv_range, verbose=verbose)

    if (len(test_df) > 0) & (len(ctrl_df) > 0) & whether_limit_stop_ff_cluster_50_size:
        test_df, ctrl_df = limit_stop_ff_cluster_50_size(test_df, ctrl_df, verbose=verbose)

    if (len(test_df) > 0) & (len(ctrl_df) > 0) & (if_test_alt_ff_group_appear_after_stop != 'flexible'):
        test_df = limit_alt_ff_flash_after_stop(test_df, if_test_alt_ff_group_appear_after_stop, verbose=verbose)

    if (len(test_df) > 0) & (len(ctrl_df) > 0) & (if_ctrl_alt_ff_group_appear_after_stop != 'flexible'):
        ctrl_df = limit_alt_ff_flash_after_stop(ctrl_df, if_ctrl_alt_ff_group_appear_after_stop, verbose=verbose)

    if (len(test_df) > 0) & (len(ctrl_df) > 0) & whether_even_out_distribution:
        test_df, ctrl_df = make_the_distributions_of_distance_more_similar_in_df(test_df, ctrl_df, verbose=verbose)
        if (len(test_df) > 0) & (len(ctrl_df) > 0):
            test_df, ctrl_df = make_the_distributions_of_angle_more_similar_in_df(test_df, ctrl_df, verbose=verbose)
    return test_df, ctrl_df


def limit_cum_distance_between_two_stops_in_df(test_df, ctrl_df, max_cum_distance_between_two_stops=400, verbose=True):
    test_length = len(test_df)
    test_df = test_df[test_df['cum_distance_between_two_stops'] <= max_cum_distance_between_two_stops].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after limiting cum distance between two stops: {test_length - len(test_df)} out of {test_length}')
    ctrl_length = len(ctrl_df)
    ctrl_df = ctrl_df[ctrl_df['cum_distance_between_two_stops'] <= max_cum_distance_between_two_stops].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in ctrl_df after limiting cum distance between two stops: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df

def limit_stop_ff_cluster_50_size(test_df, ctrl_df, verbose=True):
    test_length = len(test_df)
    test_df = test_df[test_df['stop_ff_cluster_50_size'] <= 1].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after limiting stop_ff_cluster_50_size: {test_length - len(test_df)} out of {test_length}')
    ctrl_length = len(ctrl_df)
    ctrl_df = ctrl_df[ctrl_df['stop_ff_cluster_50_size'] <= 1].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in ctrl_df after limiting stop_ff_cluster_50_size: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df


def limit_test_alt_ff_cluster_flash_after_stop(test_df, ctrl_df, if_test_alt_ff_cluster_flash_after_stop, verbose=True):
    test_length = len(test_df)
    if if_test_alt_ff_cluster_flash_after_stop is True:
        test_df = test_df[test_df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_test_alt_ff_cluster_flash_after_stop is False:
        test_df = test_df[~test_df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_test_alt_ff_cluster_flash_after_stop == 'flexible':
        pass
    else:
        raise ValueError('if_test_alt_ff_cluster_flash_after_stop must be True, False, or "flexible"')
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after using if_test_alt_ff_cluster_flash_after_stop = {if_test_alt_ff_cluster_flash_after_stop} is: {test_length - len(test_df)} out of {test_length}')
    return test_df, ctrl_df


def limit_ctrl_alt_ff_cluster_flash_after_stop(test_df, ctrl_df, if_ctrl_alt_ff_cluster_flash_after_stop, verbose=True):
    ctrl_length = len(ctrl_df)
    if if_ctrl_alt_ff_cluster_flash_after_stop is True:
        ctrl_df = ctrl_df[ctrl_df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_ctrl_alt_ff_cluster_flash_after_stop is False:
        ctrl_df = ctrl_df[~ctrl_df['alt_ff_cluster_last_flash_time_bsans'].isnull()].copy()
    elif if_ctrl_alt_ff_cluster_flash_after_stop == 'flexible':
        pass
    else:
        raise ValueError('if_ctrl_alt_ff_cluster_flash_after_stop must be True, False, or "flexible"')
    if verbose:
        print(f'Number of rows dropped out of total rows in ctrl_df after using if_ctrl_alt_ff_cluster_flash_after_stop = {if_ctrl_alt_ff_cluster_flash_after_stop} is: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df


def prune_out_data_with_large_curv_range_in_df(test_df, ctrl_df, max_curv_range=100, verbose=True):
    test_length = len(test_df)
    test_df = test_df[test_df['curv_range'] <= max_curv_range].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after limiting curv range: {test_length - len(test_df)} out of {test_length}')
    ctrl_length = len(ctrl_df)
    ctrl_df = ctrl_df[ctrl_df['curv_range'] <= max_curv_range].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in ctrl_df after limiting curv range: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df


def limit_the_range_of_alt_ff_angle_at_ref_in_df(test_df, ctrl_df, verbose=True):
    shared_min_alt_angle = max(test_df['alt_ff_angle_at_ref'].min(), ctrl_df['alt_ff_angle_at_ref'].min())
    shared_max_alt_angle = min(test_df['alt_ff_angle_at_ref'].max(), ctrl_df['alt_ff_angle_at_ref'].max())
    test_length = len(test_df)
    test_df = test_df[test_df['alt_ff_angle_at_ref'].between(shared_min_alt_angle, shared_max_alt_angle)].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after limiting alt angle at ref point: {test_length - len(test_df)} out of {test_length}')
    ctrl_length = len(ctrl_df)
    ctrl_df = ctrl_df[ctrl_df['alt_ff_angle_at_ref'].between(shared_min_alt_angle, shared_max_alt_angle)].copy()
    if verbose:
        print(f'Number of rows dropped out of total rows in ctrl_df after limiting alt angle at ref point: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df


def make_the_distributions_of_distance_more_similar_in_df(test_df, ctrl_df, verbose=True):
    test_length = len(test_df)
    ctrl_length = len(ctrl_df)
    test_df, ctrl_df = make_the_distributions_of_a_column_more_similar_between_two_df(test_df, ctrl_df, 
                                                                                                            column_name='stop_ff_distance_at_ref', bins=np.arange(0, 500, 5))
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after making the distributions of stop_ff_distance_at_ref more similar: {test_length - len(test_df)} out of {test_length}')
        print(f'Number of rows dropped out of total rows in ctrl_df after making the distributions of stop_ff_distance_at_ref more similar: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
    return test_df, ctrl_df

def make_the_distributions_of_angle_more_similar_in_df(test_df, ctrl_df, verbose=True):

    test_length = len(test_df)
    ctrl_length = len(ctrl_df)
    if (test_length == 0) | (ctrl_length == 0):
        print('No data in test_df or ctrl_df. The function make_the_distributions_of_angle_more_similar_in_df is not used.')
        return test_df, ctrl_df
    ctrl_df, test_df = make_the_distributions_of_a_column_more_similar_between_two_df(ctrl_df, test_df,
                                                                                                            column_name='stop_ff_angle_at_ref', bins=np.arange(-pi, pi, 0.05))
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after making the distributions of stop_ff_angle_at_ref more similar: {test_length - len(test_df)} out of {test_length}')
        print(f'Number of rows dropped out of total rows in ctrl_df after making the distributions of stop_ff_angle_at_ref more similar: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
     
   
    test_length = len(test_df)
    ctrl_length = len(ctrl_df)
    test_df, ctrl_df = make_the_distributions_of_a_column_more_similar_between_two_df(test_df, ctrl_df,
                                                                                                            column_name='alt_ff_angle_at_ref', bins=np.arange(-pi, pi, 0.05))
    if verbose:
        print(f'Number of rows dropped out of total rows in test_df after making the distributions of alt_ff_angle_at_ref more similar: {test_length - len(test_df)} out of {test_length}')
        print(f'Number of rows dropped out of total rows in ctrl_df after making the distributions of alt_ff_angle_at_ref more similar: {ctrl_length - len(ctrl_df)} out of {ctrl_length}')
   


    return test_df, ctrl_df

def process_combd_plan_x_and_y_combd(combd_plan_x_both, combd_plan_y_both, curv_columns=['curv_min', 'curv_max', 'curv_range']):
    combd_plan_x_both['ref_d_heading_of_traj'] = combd_plan_y_both['ref_d_heading_of_traj'].values
    combd_plan_x_both['stop_ff_angle_diff_boundary_at_ref'] = combd_plan_x_both['stop_ff_angle_at_ref'] - combd_plan_x_both['stop_ff_angle_boundary_at_ref']
    combd_plan_y_both['dir_from_stop_ff_to_stop'] = ((combd_plan_y_both['dir_from_stop_ff_to_stop'] + 1)/2).astype(int)
    combd_plan_y_both['dir_from_stop_ff_same_side'] = ((combd_plan_y_both['dir_from_stop_ff_same_side'] + 1)/2).astype(int)
    combd_plan_x_both['dir_from_stop_ff_to_alt_ff'] = ((combd_plan_y_both['dir_from_stop_ff_to_alt_ff'] + 1)/2).astype(int)
    combd_plan_x_both[curv_columns] = combd_plan_y_both[curv_columns].values
    return combd_plan_x_both, combd_plan_y_both