import sys
from data_wrangling import process_monkey_information, specific_utils
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import logging
from matplotlib import rc
import warnings
import scipy

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


monkey_info_columns_of_interest = ['bin',
                                   'LDy', 'LDz', 'RDy', 'RDz', 'gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y',
                                   'monkey_speed', 'monkey_angle', 'monkey_dw', 'monkey_ddw', 'monkey_ddv',
                                   'num_distinct_stops', 'stop_time_ratio_in_bin', 'num_caught_ff',
                                   ]


# Custom function for mode (returns first mode if multiple)
def get_mode(x):
    return x.mode().iloc[0] if not x.mode().empty else pd.NA

# Define strict median function


def strict_median(series, method='lower'):
    sorted_vals = np.sort(series.dropna().values)
    n = len(sorted_vals)
    if n == 0:
        return np.nan
    elif n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        if method == 'lower':
            return sorted_vals[n // 2 - 1]
        elif method == 'upper':
            return sorted_vals[n // 2]
        else:
            raise ValueError("method must be 'lower' or 'upper'")


def _bin_behav_data_by_point(ori_df):
    col_max = ['target_visible_dummy', 'target_cluster_visible_dummy', 'capture_target_dummy',
               # 'num_visible_ff', 'any_ff_visible', 'catching_ff' # these features are merged into df after binning later
               ]
    col_strict_median = ['point_index', 'valid_view_point']

    # Combine aggregation functions
    agg_funcs = {col: 'max' for col in col_max}
    agg_funcs.update({col: strict_median for col in col_strict_median})

    # Get list of remaining columns to apply median
    remaining_cols = [col for col in ori_df.columns if col not in [
        'bin'] + list(agg_funcs)]
    for col in remaining_cols:
        agg_funcs[col] = 'median'

    # Perform groupby
    binned_df = ori_df.groupby('bin').agg(agg_funcs).reset_index()

    # Drop unwanted columns (corrected)
    binned_df = binned_df.drop(columns=[
        'target_index',
        'target_has_disappeared_for_last_time_dummy',
        'target_cluster_has_disappeared_for_last_time_dummy'
    ])

    # Merge back relevant columns (corrected)
    binned_df = binned_df.merge(
        ori_df[['point_index', 'target_index', 'target_has_disappeared_for_last_time_dummy',
                'target_cluster_has_disappeared_for_last_time_dummy']],
        on='point_index',
        how='left'
    )
    return binned_df


def bin_behav_data_by_point(ori_df, time_bins, add_stop_time_ratio_in_bin=True, one_point_index_per_bin=True):
    ori_df = ori_df.copy()
    ori_df['bin'] = np.digitize(
        ori_df['time'].values, time_bins)-1

    if one_point_index_per_bin:
        binned_df = _bin_by_one_point_index_per_bin(
            ori_df)
    else:
        binned_df = _bin_behav_data_by_point(ori_df)
        if add_stop_time_ratio_in_bin:
            binned_df = _add_stop_time_ratio_in_bin(ori_df, binned_df)

    binned_df = _make_sure_every_bin_has_info(binned_df)
    binned_df = _add_bin_time_info(binned_df, time_bins)
    binned_df['point_index'] = binned_df['point_index'].astype(
        int)

    return binned_df


def bin_monkey_information(ori_df, time_bins, add_stop_time_ratio_in_bin=True, one_point_index_per_bin=True):
    ori_df = ori_df.copy()
    ori_df['bin'] = np.digitize(
        ori_df['time'].values, time_bins)-1

    if one_point_index_per_bin:
        binned_df = _bin_by_one_point_index_per_bin(
            ori_df)
    else:
        binned_df = ori_df.groupby(
            'bin', as_index=False).mean()
        if add_stop_time_ratio_in_bin:
            binned_df = _add_stop_time_ratio_in_bin(ori_df, binned_df)

    binned_df = _make_sure_every_bin_has_info(binned_df)
    binned_df = _add_bin_time_info(binned_df, time_bins)
    binned_df['point_index'] = binned_df['point_index'].astype(
        int)

    return binned_df


def _bin_by_one_point_index_per_bin(monkey_information):
    median_indices = (
        monkey_information.groupby('bin')['point_index']
        .median().round().astype(int)
    )
    # extract corresponding rows from monkey_information
    monkey_info_in_bins = monkey_information.set_index(
        'point_index').loc[median_indices.values].reset_index(drop=False)
    # add bin info back
    monkey_info_in_bins['bin'] = median_indices.index

    return monkey_info_in_bins


def _make_sure_every_bin_has_info(monkey_info_in_bins):
    all_bins = pd.DataFrame({'bin': range(max(monkey_info_in_bins['bin']))})
    monkey_info_in_bins = monkey_info_in_bins.merge(
        all_bins, on='bin', how='right')
    monkey_info_in_bins = monkey_info_in_bins.ffill(
    ).infer_objects(copy=False).reset_index(drop=True)
    monkey_info_in_bins = monkey_info_in_bins.bfill(
    ).infer_objects(copy=False).reset_index(drop=True)
    return monkey_info_in_bins


def _add_bin_time_info(monkey_info_in_bins, time_bins):
    monkey_info_in_bins['bin_start_time'] = time_bins[monkey_info_in_bins['bin'].values]
    monkey_info_in_bins['bin_end_time'] = time_bins[monkey_info_in_bins['bin'].values + 1]
    return monkey_info_in_bins


def make_monkey_info_in_bins_essential(monkey_info_in_bins, time_bins, ff_caught_T_new):
    """Prepare behavioral data."""

    monkey_info_in_bins_ess = monkey_info_in_bins.copy()

    # add the number of distinct stops; it's more meaningful when one_point_index_per_bin is False; otherwise, it's the same as the monkey_speeddummy
    monkey_info_in_bins_ess['num_distinct_stops'] = monkey_info_in_bins.groupby(
        'bin').sum()['whether_new_distinct_stop'].values

    monkey_info_in_bins_ess.index = monkey_info_in_bins_ess['bin'].values

    monkey_info_in_bins_ess = _add_num_caught_ff(
        monkey_info_in_bins_ess, ff_caught_T_new, time_bins)
    monkey_info_in_bins_ess = _make_bin_continuous(monkey_info_in_bins_ess)
    monkey_info_in_bins_ess = _clip_columns(monkey_info_in_bins_ess)

    # only preserve the rows in monkey_information where bin is within the range of time_bins (there are in total len(time_bins)-1 bins)
    monkey_info_in_bins_ess = monkey_info_in_bins_ess[monkey_info_in_bins_ess['bin'].between(
        0, len(time_bins)-1, inclusive='left')].reset_index(drop=True)

    monkey_info_in_bins_essential = _select_monkey_info_columns_of_interest(
        monkey_info_in_bins_ess)
    monkey_info_in_bins_essential = _add_stop_rate_and_success_rate(
        monkey_info_in_bins_essential)

    return monkey_info_in_bins_essential


def initialize_binned_features(monkey_information, bin_width):
    max_time = monkey_information['time'].max()
    time_bins = np.arange(0, max_time + bin_width * 2, bin_width)
    binned_features = pd.DataFrame({'bin': range(len(time_bins))})
    return binned_features, time_bins


def add_pattern_info_base_on_points(binned_features, monkey_info_in_bins, monkey_information,
                                    try_a_few_times_indices_for_anim, GUAT_point_indices_for_anim,
                                    ignore_sudden_flash_indices_for_anim):

    pattern_df = monkey_information[['point_index']].copy()
    pattern_df.index = pattern_df['point_index'].values

    pattern_df['try_a_few_times_indice_dummy'] = 0
    pattern_df.loc[try_a_few_times_indices_for_anim,
                   'try_a_few_times_indice_dummy'] = 1
    pattern_df['give_up_after_trying_indice_dummy'] = 0
    pattern_df.loc[GUAT_point_indices_for_anim,
                   'give_up_after_trying_indice_dummy'] = 1
    pattern_df['ignore_sudden_flash_indice_dummy'] = 0
    pattern_df.loc[ignore_sudden_flash_indices_for_anim,
                   'ignore_sudden_flash_indice_dummy'] = 1
    pattern_df.reset_index(drop=True, inplace=True)

    pattern_df = pattern_df.merge(
        monkey_info_in_bins[['bin', 'point_index']], on='point_index', how='right')

    pattern_df_condensed = pattern_df[['bin', 'try_a_few_times_indice_dummy', 'give_up_after_trying_indice_dummy',
                                       'ignore_sudden_flash_indice_dummy']].copy()
    pattern_df_condensed = pattern_df_condensed.groupby(
        'bin').max().reset_index(drop=False)
    binned_features = binned_features.merge(
        pattern_df_condensed, how='left', on='bin')
    binned_features = binned_features.ffill().reset_index(drop=True)
    binned_features = binned_features.bfill().reset_index(drop=True)
    return binned_features


def add_pattern_info_based_on_trials(binned_features, ff_caught_T_new, all_trial_patterns, time_bins):

    bin_midlines = _prepare_bin_midlines(
        time_bins, ff_caught_T_new, all_trial_patterns)

    binned_features = binned_features.merge(bin_midlines[['bin', 'two_in_a_row', 'visible_before_last_one',
                                                          'disappear_latest', 'ignore_sudden_flash', 'try_a_few_times',
                                                          'give_up_after_trying', 'cluster_around_target',
                                                          'waste_cluster_around_target']], on='bin', how='left')

    binned_features = binned_features.ffill().reset_index(drop=True)
    binned_features = binned_features.bfill().reset_index(drop=True)
    return binned_features


def get_ff_info_for_bins(bins_df, ff_dataframe, ff_caught_T_new, time_bins):
    ff_dataframe['bin'] = np.digitize(ff_dataframe.time, time_bins)-1
    bins_df = _add_num_alive_ff(bins_df, ff_dataframe)
    bins_df = _add_num_visible_ff(bins_df, ff_dataframe)
    bins_df = _add_min_ff_info(bins_df, ff_dataframe)
    bins_df = _add_min_visible_ff_info(bins_df, ff_dataframe)
    bins_df = _mark_bin_where_ff_is_caught(
        bins_df, ff_caught_T_new, time_bins)
    bins_df = _add_whether_any_ff_is_visible(bins_df)
    bins_df = bins_df.ffill().reset_index(drop=True)
    bins_df = bins_df.bfill().reset_index(drop=True)
    return bins_df


def _make_final_behavioral_data(monkey_info_in_bins_essential, binned_features):
    """
    Merge all the features back to monkey_info_in_bins_essential.
    """
    # drop columns in monkey_info_in_bins_essential that are already in binned_features, except for 'bin'
    shared_columns = set(monkey_info_in_bins_essential.columns).intersection(
        set(binned_features.columns))
    shared_columns = shared_columns - {'bin'}
    monkey_info_in_bins_essential = monkey_info_in_bins_essential.drop(
        columns=shared_columns)
    final_behavioral_data = monkey_info_in_bins_essential.merge(
        binned_features, how='left', on='bin')
    final_behavioral_data.fillna(0, inplace=True)
    return final_behavioral_data


def _add_stop_time_ratio_in_bin(ori_df, binned_df):
    monkey_stop_time_ratio_in_bin = _calculate_stop_time_ratio_in_bin(
        ori_df)
    binned_df = binned_df.merge(
        monkey_stop_time_ratio_in_bin[['bin', 'stop_time_ratio_in_bin']], on='bin', how='left')
    binned_df.drop(columns=['monkey_speeddummy'], inplace=True)
    return binned_df


def _calculate_stop_time_ratio_in_bin(monkey_information):
    """
    Calculate the stop time ratio in each bin.

    For each bin, computes the total time spent stopped (speed ≤ 1) divided by the total time in that bin.

    Returns:
        DataFrame indexed by bin with:
            - 'stop_duration': total stop time in the bin
            - 'stop_time_ratio_in_bin': stop_duration / time
    """
    df = monkey_information.copy()

    # Set stop_duration to time if speed ≤ 1, else 0
    df['stop_duration'] = df['time']
    df.loc[df['monkey_speeddummy'] > 1, 'stop_duration'] = 0

    # Group by bin and sum stop_duration and time
    binned_df = df.groupby('bin', as_index=False)[
        ['stop_duration', 'time']].sum()

    # Calculate stop time ratio
    binned_df['stop_time_ratio_in_bin'] = binned_df['stop_duration'] / \
        binned_df['time']

    return binned_df


def _add_num_caught_ff(monkey_info_in_bins_ess, ff_caught_T_new, time_bins):
    """Add num_caught_ff to monkey_info_in_bins_ess."""
    catching_target_bins = np.digitize(ff_caught_T_new, time_bins)-1
    catching_target_bins_unique, counts = np.unique(
        catching_target_bins, return_counts=True)
    catching_target_bins_unique = catching_target_bins_unique[catching_target_bins_unique < len(
        time_bins)-1]
    counts = counts[:len(catching_target_bins_unique)]
    monkey_info_in_bins_ess['num_caught_ff'] = 0
    # make sure catching_target_bins_unique doesn't exceed bound
    point_index_max = monkey_info_in_bins_ess.index.max()
    catching_target_bins_unique[catching_target_bins_unique >
                                point_index_max] = point_index_max
    monkey_info_in_bins_ess.loc[catching_target_bins_unique,
                                'num_caught_ff'] = counts
    return monkey_info_in_bins_ess


def _make_bin_continuous(monkey_info_in_bins_ess):
    """Make sure that the bin number is continuous in monkey_info_in_bins_ess."""
    continuous_bins = pd.DataFrame(
        {'bin': range(monkey_info_in_bins_ess.bin.max()+1)})
    monkey_info_in_bins_ess = continuous_bins.merge(
        monkey_info_in_bins_ess, how='left', on='bin')
    monkey_info_in_bins_ess = monkey_info_in_bins_ess.ffill().reset_index(drop=True)
    return monkey_info_in_bins_ess


def _clip_columns(monkey_info_in_bins_ess):
    """Clip values of specified columns."""
    for column in ['gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_world_x', 'gaze_world_y']:
        monkey_info_in_bins_ess.loc[:, column] = np.clip(
            monkey_info_in_bins_ess.loc[:, column], -1000, 1000)
    return monkey_info_in_bins_ess


def _select_monkey_info_columns_of_interest(monkey_info_in_bins_ess):
    """Select columns of interest."""
    # make sure the columns are in the df
    columns_to_keep = [
        col for col in monkey_info_columns_of_interest if col in monkey_info_in_bins_ess.columns]
    monkey_info_in_bins_essential = monkey_info_in_bins_ess[columns_to_keep].copy(
    )
    return monkey_info_in_bins_essential


def _add_stop_rate_and_success_rate(monkey_info_in_bins_essential,
                                    kernel_size=7, std_dev=2):
    """Add stop_rate and stop_success_rate to monkey_info_in_bins_essential."""

    # Create a Gaussian kernel
    gaussian_kernel = scipy.signal.gaussian(kernel_size, std_dev)
    # Normalize the kernel so it sums to 1
    gaussian_kernel /= gaussian_kernel.sum()

    num_distinct_stops_convolved = np.convolve(
        monkey_info_in_bins_essential['num_distinct_stops'], gaussian_kernel, 'same')
    num_caught_ff_convolved = np.convolve(
        monkey_info_in_bins_essential['num_caught_ff'], gaussian_kernel, 'same')
    # note, stop_rate here is the number of stops in a bin convolved by a gaussian kernel
    monkey_info_in_bins_essential['stop_rate'] = num_distinct_stops_convolved
    # suppress the warning for the line below
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        monkey_info_in_bins_essential['stop_success_rate'] = num_caught_ff_convolved / \
            num_distinct_stops_convolved
    # if there's na or inf in monkey_info_in_bins_essential['stop_success_rate'], replace it with 0
    monkey_info_in_bins_essential['stop_success_rate'] = monkey_info_in_bins_essential['stop_success_rate'].replace([
        np.inf, -np.inf], np.nan).fillna(0)
    return monkey_info_in_bins_essential


def _prepare_bin_midlines(time_bins, ff_caught_T_new, all_trial_patterns):
    """
    Prepare the bin_midlines dataframe by finding the centers of time_bins and adding trial info.
    """

    # Add the category info based on trials
    all_trial_patterns['trial_end_time'] = ff_caught_T_new[:len(
        all_trial_patterns)]
    all_trial_patterns['trial_start_time'] = all_trial_patterns['trial_end_time'].shift(
        1).fillna(0)
    bin_midlines = pd.DataFrame(
        (time_bins[:-1] + time_bins[1:])/2, columns=['bin_midline'])
    bin_midlines = bin_midlines[bin_midlines['bin_midline']
                                < ff_caught_T_new[-1]]

    # Add trial info to bin_midlines
    bin_midlines['trial'] = np.searchsorted(
        ff_caught_T_new, bin_midlines['bin_midline'])
    all_trial_patterns['trial'] = all_trial_patterns.index
    bin_midlines = bin_midlines.merge(
        all_trial_patterns, on='trial', how='left')
    bin_midlines['bin'] = bin_midlines.index

    return bin_midlines


def _add_num_alive_ff(binned_features, ff_dataframe):
    # get some summary statistics from ff_dataframe to use as features for CCA
    # count of visible and in-memory ff
    ff_dataframe_sub = ff_dataframe[['bin', 'ff_index']]
    ff_dataframe_unique_ff = ff_dataframe_sub.groupby(
        'bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_ff.rename(
        columns={'ff_index': 'num_alive_ff'}, inplace=True)
    binned_features = binned_features.merge(
        ff_dataframe_unique_ff, how='left', on='bin')
    return binned_features


def _add_num_visible_ff(binned_features, ff_dataframe):
    # count of visible ff
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    ff_dataframe_unique_visible_ff = ff_dataframe_visible[['bin', 'ff_index']]
    ff_dataframe_unique_visible_ff = ff_dataframe_unique_visible_ff.groupby(
        'bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_visible_ff.rename(
        columns={'ff_index': 'num_visible_ff'}, inplace=True)
    binned_features = binned_features.merge(
        ff_dataframe_unique_visible_ff, how='left', on='bin')
    return binned_features


def _add_min_ff_info(binned_features, ff_dataframe):
    # min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary', 'memory']]
    min_ff_info = ff_dataframe[[
        'bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_ff_info = min_ff_info.groupby('bin').min().reset_index(drop=False)
    min_ff_info.rename(columns={'ff_distance': 'min_ff_distance',
                                'ff_angle': 'min_abs_ff_angle',
                                'ff_angle_boundary': 'min_abs_ff_angle_boundary'}, inplace=True)
    # 'memory': 'min_ff_memory'}, inplace=True) # memory is currently not used bc the whole column is 100
    binned_features = binned_features.merge(min_ff_info, how='left', on='bin')
    return binned_features


def _add_min_visible_ff_info(binned_features, ff_dataframe):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1]
    min_visible_ff_info = ff_dataframe_visible[[
        'bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_visible_ff_info = min_visible_ff_info.groupby(
        'bin').min().reset_index(drop=False)
    min_visible_ff_info.rename(columns={'ff_distance': 'min_visible_ff_distance',
                                        'ff_angle': 'min_abs_visible_ff_angle',
                                        'ff_angle_boundary': 'min_abs_visible_ff_angle_boundary'}, inplace=True)
    binned_features = binned_features.merge(
        min_visible_ff_info, how='left', on='bin')
    return binned_features


def _mark_bin_where_ff_is_caught(binned_features, ff_caught_T_new, time_bins):
    binned_features['catching_ff'] = 0
    catching_target_bins = np.digitize(ff_caught_T_new, time_bins)-1
    binned_features.loc[binned_features['bin'].isin(
        catching_target_bins), 'catching_ff'] = 1
    return binned_features


def _add_whether_any_ff_is_visible(binned_features):
    # only add the column if the ratio of bins with visible ff is between 10% and 90% within all the bins, since otherwise it might not be so meaningful (a.k.a. most bins have visible ff or most bins don't have visible ff)
    any_ff_visible = (binned_features['num_visible_ff'] > 0).astype(int)
    if (any_ff_visible.sum()/len(binned_features) > 0.1) and (any_ff_visible.sum()/len(binned_features) < 0.9):
        binned_features['any_ff_visible'] = binned_features['num_visible_ff'] > 0
        binned_features['any_ff_visible'] = binned_features['any_ff_visible'].astype(
            int)
    return binned_features
