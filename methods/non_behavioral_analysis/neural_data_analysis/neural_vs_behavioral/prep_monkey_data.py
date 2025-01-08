import sys
from data_wrangling import process_monkey_information, specific_utils
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

def make_rebinned_monkey_info_essential(monkey_information, time_bins, ff_caught_T_new, convolve_pattern, window_width):
    """Prepare behavioral data."""
    monkey_information = monkey_information.copy()
    time = monkey_information['time'].values

    monkey_information = _add_turning_right(monkey_information)
    monkey_information['bin'] = np.digitize(time, time_bins)-1

    rebinned_monkey_info = _rebin_monkey_info(monkey_information).reset_index(drop=True)
    rebinned_monkey_info.index = rebinned_monkey_info['bin'].values
    rebinned_monkey_info = _add_num_caught_ff(rebinned_monkey_info, ff_caught_T_new, time_bins)
    rebinned_monkey_info = _make_bin_continuous(rebinned_monkey_info)
    rebinned_monkey_info = _clip_columns(rebinned_monkey_info)

    # only preserve the rows in monkey_information where bin is within the range of time_bins (there are in total len(time_bins)-1 bins)
    rebinned_monkey_info = rebinned_monkey_info[rebinned_monkey_info['bin'].between(0, len(time_bins)-1, inclusive='left')].reset_index(drop=True)

    rebinned_monkey_info_essential = _select_columns_of_interest(rebinned_monkey_info)
    rebinned_monkey_info_essential = _add_stop_rate_and_success_rate(rebinned_monkey_info_essential, convolve_pattern, window_width)

    return rebinned_monkey_info_essential, monkey_information


def make_binned_features(monkey_information, bin_width, ff_dataframe, ff_caught_T_new):
    min_time = monkey_information['time'].min()
    max_time = monkey_information['time'].max()
    time_bins = np.arange(min_time, max_time, bin_width) 
    monkey_information['bin'] = np.digitize(monkey_information['time'].values, time_bins)-1
    binned_features = pd.DataFrame({'bin': range(len(time_bins))})
    binned_features = _add_ff_info_to_binned_features(binned_features, ff_dataframe, ff_caught_T_new, time_bins)
    return binned_features

def add_pattern_info_base_on_points(binned_features, monkey_information,
                                    try_a_few_times_indices_for_anim, GUAT_point_indices_for_anim,
                                    ignore_sudden_flash_indices_for_anim):
    pattern_df = monkey_information[['bin', 'point_index']].copy()
    pattern_df.index = pattern_df['point_index'].values
    
    pattern_df['try_a_few_times_indice_dummy'] = 0
    pattern_df.loc[try_a_few_times_indices_for_anim, 'try_a_few_times_indice_dummy'] = 1
    pattern_df['give_up_after_trying_indice_dummy'] = 0
    pattern_df.loc[GUAT_point_indices_for_anim, 'give_up_after_trying_indice_dummy'] = 1
    pattern_df['ignore_sudden_flash_indice_dummy'] = 0
    pattern_df.loc[ignore_sudden_flash_indices_for_anim, 'ignore_sudden_flash_indice_dummy'] = 1

    pattern_df_condensed = pattern_df[['bin', 'try_a_few_times_indice_dummy', 'give_up_after_trying_indice_dummy',
                                    'ignore_sudden_flash_indice_dummy']].copy()
    pattern_df_condensed = pattern_df_condensed.groupby('bin').max().reset_index(drop=False) 
    binned_features = binned_features.merge(pattern_df_condensed, how='left', on='bin')
    binned_features = binned_features.fillna(method='ffill').reset_index(drop=True)
    binned_features = binned_features.fillna(method='bfill').reset_index(drop=True)
    return binned_features


def add_pattern_info_based_on_trials(binned_features, ff_caught_T_new, all_trial_patterns, time_bins):

    bin_midlines = _prepare_bin_midlines(time_bins, ff_caught_T_new, all_trial_patterns)
    
    binned_features = binned_features.merge(bin_midlines[['bin', 'two_in_a_row', 'visible_before_last_one',
        'disappear_latest', 'ignore_sudden_flash', 'try_a_few_times',
        'give_up_after_trying', 'cluster_around_target',
        'waste_cluster_around_target']], on = 'bin', how='left')
    
    binned_features = binned_features.fillna(method='ffill').reset_index(drop=True)
    binned_features = binned_features.fillna(method='bfill').reset_index(drop=True)
    return binned_features


def _add_ff_info_to_binned_features(binned_features, ff_dataframe, ff_caught_T_new, time_bins):
    ff_dataframe['bin'] = np.digitize(ff_dataframe.time, time_bins)-1
    binned_features = _add_num_alive_ff(binned_features, ff_dataframe)
    binned_features = _add_num_visible_ff(binned_features, ff_dataframe)
    binned_features = _add_min_ff_info(binned_features, ff_dataframe)
    binned_features = _add_min_visible_ff_info(binned_features, ff_dataframe)
    binned_features = _mark_bin_where_ff_is_caught(binned_features, ff_caught_T_new, time_bins)
    binned_features = _add_whether_any_ff_is_visible(binned_features)
    binned_features = binned_features.fillna(method='ffill').reset_index(drop=True)
    binned_features = binned_features.fillna(method='bfill').reset_index(drop=True)
    return binned_features

def _make_final_behavioral_data(rebinned_monkey_info_essential, binned_features):
    """
    Merge all the features back to rebinned_monkey_info_essential.
    """
    # drop columns in rebinned_monkey_info_essential that are already in binned_features, except for 'bin'
    shared_columns = set(rebinned_monkey_info_essential.columns).intersection(set(binned_features.columns))
    shared_columns = shared_columns - {'bin'}
    rebinned_monkey_info_essential = rebinned_monkey_info_essential.drop(columns=shared_columns)
    final_behavioral_data = rebinned_monkey_info_essential.merge(binned_features, how='left', on='bin')
    final_behavioral_data.fillna(0, inplace=True)
    final_behavioral_data = _add_whether_any_ff_is_visible(final_behavioral_data)
    return final_behavioral_data


def _add_turning_right(monkey_information):
    """Add dummy variable of turning left or right: 0 means left and 1 means right."""
    monkey_information['turning_right'] = 0
    monkey_information.loc[monkey_information['monkey_dw'] < 0, 'turning_right'] = 1
    return monkey_information


def _rebin_monkey_info(monkey_information):
    """Rebin monkey information."""
    monkey_information = monkey_information.copy()
    monkey_information['stop_duration'] = monkey_information['time']
    monkey_information.loc[monkey_information['monkey_speeddummy'] > 1, 'stop_duration'] = 0
    rebinned_monkey_info = monkey_information.groupby('bin').mean()
    # take out the name of the index


    rebinned_monkey_info['bin'] = rebinned_monkey_info.index
    rebinned_monkey_info.rename(columns={'monkey_speeddummy': 'stop_time_ratio_in_bin'}, inplace=True)
    rebinned_monkey_info['stop_time_ratio_in_bin'] = rebinned_monkey_info['stop_time_ratio_in_bin']/rebinned_monkey_info['time']
    rebinned_monkey_info['num_distinct_stops'] = monkey_information.groupby('bin').sum()['whether_new_distinct_stop'].values
    return rebinned_monkey_info

def _add_num_caught_ff(rebinned_monkey_info, ff_caught_T_new, time_bins):
    """Add num_caught_ff to rebinned_monkey_info."""
    catching_target_bins = np.digitize(ff_caught_T_new, time_bins)-1
    catching_target_bins_unique, counts = np.unique(catching_target_bins, return_counts=True)
    catching_target_bins_unique = catching_target_bins_unique[catching_target_bins_unique < len(time_bins)-1]
    counts = counts[:len(catching_target_bins_unique)]
    rebinned_monkey_info['num_caught_ff'] = 0
    rebinned_monkey_info.loc[catching_target_bins_unique, 'num_caught_ff'] = counts
    return rebinned_monkey_info

def _make_bin_continuous(rebinned_monkey_info):
    """Make sure that the bin number is continuous in rebinned_monkey_info."""
    continuous_bins = pd.DataFrame({'bin': range(rebinned_monkey_info.bin.max()+1)})
    rebinned_monkey_info = continuous_bins.merge(rebinned_monkey_info, how='left', on='bin')
    rebinned_monkey_info = rebinned_monkey_info.ffill().reset_index(drop=True)
    return rebinned_monkey_info

def _clip_columns(rebinned_monkey_info):
    """Clip values of specified columns."""
    for column in ['gaze_monkey_view_x', 'gaze_monkey_view_y', 'gaze_world_x', 'gaze_world_y']:
        rebinned_monkey_info.loc[:,column] = np.clip(rebinned_monkey_info.loc[:,column], -1000, 1000)
    return rebinned_monkey_info

def _select_columns_of_interest(rebinned_monkey_info):
    """Select columns of interest."""
    columns_of_interest = ['bin', 'LDy', 'LDz', 'RDy', 'RDz', 'gaze_monkey_view_x', 'gaze_monkey_view_y', 'gaze_world_x', 'gaze_world_y', 
                        'monkey_speed', 'monkey_angle', 'monkey_dw', 'monkey_ddw', 'monkey_ddv', 'num_distinct_stops', 'stop_time_ratio_in_bin', 'num_caught_ff']
    rebinned_monkey_info_essential = rebinned_monkey_info[columns_of_interest].copy()
    return rebinned_monkey_info_essential

def _add_stop_rate_and_success_rate(rebinned_monkey_info_essential, convolve_pattern, window_width):
    """Add stop_rate and stop_success_rate to rebinned_monkey_info_essential."""
    num_distinct_stops_convolved = np.convolve(rebinned_monkey_info_essential['num_distinct_stops'], convolve_pattern, 'same')
    num_caught_ff_convolved = np.convolve(rebinned_monkey_info_essential['num_caught_ff'], convolve_pattern, 'same')
    rebinned_monkey_info_essential['stop_rate'] = num_distinct_stops_convolved/window_width
    rebinned_monkey_info_essential['stop_success_rate'] = num_caught_ff_convolved/num_distinct_stops_convolved
    # if there's na or inf in rebinned_monkey_info_essential['stop_success_rate'], replace it with 0
    rebinned_monkey_info_essential['stop_success_rate'] = rebinned_monkey_info_essential['stop_success_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return rebinned_monkey_info_essential


def _prepare_bin_midlines(time_bins, ff_caught_T_new, all_trial_patterns):
    """
    Prepare the bin_midlines dataframe by finding the centers of time_bins and adding trial info.
    """

    # Add the category info based on trials
    all_trial_patterns['trial_end_time'] = ff_caught_T_new[:len(all_trial_patterns)]
    all_trial_patterns['trial_start_time'] = all_trial_patterns['trial_end_time'].shift(1).fillna(0)
    bin_midlines = pd.DataFrame((time_bins[:-1] + time_bins[1:])/2, columns=['bin_midline'])
    bin_midlines = bin_midlines[bin_midlines['bin_midline'] < ff_caught_T_new[-1]]

    # Add trial info to bin_midlines
    bin_midlines['trial'] = np.searchsorted(ff_caught_T_new, bin_midlines['bin_midline'])
    all_trial_patterns['trial'] = all_trial_patterns.index
    bin_midlines = bin_midlines.merge(all_trial_patterns, on='trial', how='left')
    bin_midlines['bin'] = bin_midlines.index

    return bin_midlines

def _add_num_alive_ff(binned_features, ff_dataframe):
    # get some summary statistics from ff_dataframe to use as features for CCA
    # count of visible and in-memory ff
    ff_dataframe_sub = ff_dataframe[['bin', 'ff_index']]
    ff_dataframe_unique_ff = ff_dataframe_sub.groupby('bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_ff.rename(columns={'ff_index': 'num_alive_ff'}, inplace=True)
    binned_features = binned_features.merge(ff_dataframe_unique_ff, how='left', on='bin')
    return binned_features

def _add_num_visible_ff(binned_features, ff_dataframe):
    # count of visible ff
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
    ff_dataframe_unique_visible_ff = ff_dataframe_visible[['bin', 'ff_index']]
    ff_dataframe_unique_visible_ff = ff_dataframe_unique_visible_ff.groupby('bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_visible_ff.rename(columns={'ff_index': 'num_visible_ff'}, inplace=True)
    binned_features = binned_features.merge(ff_dataframe_unique_visible_ff, how='left', on='bin')
    return binned_features

def _add_min_ff_info(binned_features, ff_dataframe):
    #min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary', 'memory']]
    min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_ff_info = min_ff_info.groupby('bin').min().reset_index(drop=False)
    min_ff_info.rename(columns={'ff_distance': 'min_ff_distance',
                                'ff_angle': 'min_abs_ff_angle',
                                'ff_angle_boundary': 'min_abs_ff_angle_boundary'}, inplace=True)
                                #'memory': 'min_ff_memory'}, inplace=True) # memory is currently not used bc the whole column is 100
    binned_features = binned_features.merge(min_ff_info, how='left', on='bin')
    return binned_features

def _add_min_visible_ff_info(binned_features, ff_dataframe):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
    min_visible_ff_info = ff_dataframe_visible[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_visible_ff_info = min_visible_ff_info.groupby('bin').min().reset_index(drop=False)
    min_visible_ff_info.rename(columns={'ff_distance': 'min_visible_ff_distance',
                                        'ff_angle': 'min_abs_visible_ff_angle', 
                                        'ff_angle_boundary': 'min_abs_visible_ff_angle_boundary'}, inplace=True)
    binned_features = binned_features.merge(min_visible_ff_info, how='left', on='bin')
    return binned_features

def _mark_bin_where_ff_is_caught(binned_features, ff_caught_T_new, time_bins):
    binned_features['catching_ff'] = 0
    catching_target_bins = np.digitize(ff_caught_T_new, time_bins)-1
    binned_features.loc[binned_features['bin'].isin(catching_target_bins), 'catching_ff'] = 1
    return binned_features

def _add_whether_any_ff_is_visible(binned_features):
    # only add it if the ratio of bins with visible ff is between 10% and 90% within all the bins, since otherwise it might not be so meaningful (a.k.a. most bins have visible ff or most bins don't have visible ff)
    any_ff_visible = (binned_features['num_visible_ff'] > 0).astype(int)
    if (any_ff_visible.sum()/len(binned_features) > 0.1) and (any_ff_visible.sum()/len(binned_features) < 0.9):
        binned_features['any_ff_visible'] = binned_features['num_visible_ff'] > 0  
        binned_features['any_ff_visible'] = binned_features['any_ff_visible'].astype(int)
    return binned_features
