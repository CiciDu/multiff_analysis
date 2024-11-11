import sysfrom data_wrangling import process_raw_data, basic_func
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





def make_rebinned_monkey_info_essential(monkey_information, time_bins, ff_caught_T_sorted, convolve_pattern, window_width):
    """Prepare behavioral data."""
    monkey_information = monkey_information.copy()
    monkey_t = monkey_information['monkey_t'].values

    monkey_information = _add_turning_right(monkey_information)
    monkey_information.loc[:, 'bin'] = np.digitize(monkey_t, time_bins)-1

    rebinned_monkey_info = _rebin_monkey_info(monkey_information)
    rebinned_monkey_info = _add_num_caught_ff(rebinned_monkey_info, ff_caught_T_sorted, time_bins)
    rebinned_monkey_info = _make_bin_continuous(rebinned_monkey_info)
    rebinned_monkey_info = _clip_columns(rebinned_monkey_info)

    # only preserve the rows in monkey_information where bin is within the range of time_bins (there are in total len(time_bins)-1 bins)
    rebinned_monkey_info = rebinned_monkey_info[rebinned_monkey_info['bin'].between(0, len(time_bins)-1, inclusive='left')].reset_index(drop=True)

    rebinned_monkey_info_essential = _select_columns_of_interest(rebinned_monkey_info)
    rebinned_monkey_info_essential = _add_stop_rate_and_success_rate(rebinned_monkey_info_essential, convolve_pattern, window_width)

    # make sure that the number of time bins in rebinned_monkey_info_essential does not exceed len(time_bins) - 1


    return rebinned_monkey_info_essential, monkey_information



def prepare_pattern_df(data_item, time_bins):
    """
    Prepare the pattern dataframe by copying and renaming columns, and adding new columns.
    """
    # Define the columns to be copied and renamed
    cols_to_copy = ['bin', 'monkey_t', 'monkey_x', 'monkey_y', 'monkey_angles', 'monkey_speed', 'monkey_speeddummy']
    cols_to_rename = {'monkey_angles': 'monkey_angle', 'monkey_t': 'time'}

    # Copy and rename columns
    pattern_df = data_item.monkey_information[cols_to_copy].copy().rename(columns=cols_to_rename)

    # Add new columns
    pattern_df['point_index'] = pattern_df.index

    # Define the dummy columns and their corresponding indices
    dummy_cols_indices = {
        'try_a_few_times_indice_dummy': data_item.try_a_few_times_indices_for_anim,
        'give_up_after_trying_indice_dummy': data_item.GUAT_point_indices_for_anim,
        'ignore_sudden_flas_indice_dummy': data_item.ignore_sudden_flash_indices_for_anim
    }

    # Add dummy columns
    for col, indices in dummy_cols_indices.items():
        pattern_df[col] = 0
        pattern_df.loc[indices, col] = 1

    cols_to_condense = ['bin', *dummy_cols_indices.keys()]
    pattern_df_condensed = pattern_df[cols_to_condense].groupby('bin').max().reset_index()

    bin_midlines = _prepare_bin_midlines(time_bins, data_item.ff_caught_T_sorted, data_item.all_trial_patterns)

    cols_to_merge = ['bin', 'two_in_a_row', 'visible_before_last_one', 'disappear_latest', 'ignore_sudden_flash', 'try_a_few_times', 'give_up_after_trying', 'cluster_around_target', 'waste_cluster_around_target']
    pattern_df_condensed = pattern_df_condensed.merge(bin_midlines[cols_to_merge], on='bin', how='left')

    return pattern_df, pattern_df_condensed





def _add_turning_right(monkey_information):
    """Add dummy variable of turning left or right: 0 means left and 1 means right."""
    monkey_information['turning_right'] = 0
    monkey_information.loc[monkey_information['monkey_dw'] < 0, 'turning_right'] = 1
    return monkey_information


def _rebin_monkey_info(monkey_information):
    """Rebin monkey information."""
    rebinned_monkey_info = monkey_information.groupby('bin').mean().reset_index(drop=False)
    rebinned_monkey_info['num_stops'] = monkey_information.groupby('bin').sum()['monkey_speeddummy']
    return rebinned_monkey_info

def _add_num_caught_ff(rebinned_monkey_info, ff_caught_T_sorted, time_bins):
    """Add num_caught_ff to rebinned_monkey_info."""
    catching_target_bins = np.digitize(ff_caught_T_sorted, time_bins)-1
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
    'monkey_speed', 'monkey_angles', 'monkey_dw', 'monkey_speeddummy', 'monkey_ddw', 'monkey_ddv', 'num_stops', 'num_caught_ff']
    rebinned_monkey_info_essential = rebinned_monkey_info[columns_of_interest].copy()
    return rebinned_monkey_info_essential

def _add_stop_rate_and_success_rate(rebinned_monkey_info_essential, convolve_pattern, window_width):
    """Add stop_rate and stop_success_rate to rebinned_monkey_info_essential."""
    num_stops_convolved = np.convolve(rebinned_monkey_info_essential['num_stops'], convolve_pattern, 'same')
    num_caught_ff_convolved = np.convolve(rebinned_monkey_info_essential['num_caught_ff'], convolve_pattern, 'same')
    rebinned_monkey_info_essential['stop_rate'] = num_stops_convolved/window_width
    rebinned_monkey_info_essential['stop_success_rate'] = num_caught_ff_convolved/num_stops_convolved
    return rebinned_monkey_info_essential



def _prepare_bin_midlines(time_bins, ff_caught_T_sorted, all_trial_patterns):
    """
    Prepare the bin_midlines dataframe by finding the centers of time_bins and adding trial info.
    """

    # Add the category info based on trials
    all_trial_patterns['trial_end_time'] = ff_caught_T_sorted[:len(all_trial_patterns)]
    all_trial_patterns['trial_start_time'] = all_trial_patterns['trial_end_time'].shift(1).fillna(0)
    bin_midlines = pd.DataFrame((time_bins[:-1] + time_bins[1:])/2, columns=['bin_midline'])
    bin_midlines = bin_midlines[bin_midlines['bin_midline'] < ff_caught_T_sorted[-1]]

    # Add trial info to bin_midlines
    bin_midlines['trial'] = np.digitize(bin_midlines['bin_midline'], ff_caught_T_sorted)
    all_trial_patterns['trial'] = all_trial_patterns.index
    bin_midlines = bin_midlines.merge(all_trial_patterns, on='trial', how='left')
    bin_midlines['bin'] = bin_midlines.index

    return bin_midlines

