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



def make_final_behavioral_data(rebinned_monkey_info_essential, binned_features):
    """
    Merge all the features back to rebinned_monkey_info_essential.
    """
    final_behavioral_data = rebinned_monkey_info_essential.merge(binned_features, how='left', on='bin')
    final_behavioral_data.fillna(0, inplace=True)
    final_behavioral_data = _add_any_ff_visible(final_behavioral_data)
    return final_behavioral_data





def make_binned_features(ff_dataframe_unique_ff, num_bins, ff_caught_T_sorted, time_bins, feature_dataframes_to_merge):
    binned_features = _prepare_binned_features(ff_dataframe_unique_ff, num_bins)
    binned_features = _merge_features(binned_features, feature_dataframes_to_merge)
    binned_features = _add_catching_ff(binned_features, ff_caught_T_sorted, time_bins)
    binned_features = _fill_na(binned_features)
    return binned_features




def _prepare_binned_features(ff_dataframe_unique_ff, num_bins):
    """
    Prepare the binned_features dataframe by ensuring continuous bins and merging features.
    """
    binned_features = ff_dataframe_unique_ff.copy()
    continuous_bins = pd.DataFrame({'bin': range(num_bins+1)})
    binned_features = continuous_bins.merge(binned_features, how='left', on='bin')
    return binned_features


def _merge_features(binned_features, feature_dataframes_to_merge):
    """
    Merge all the features into the binned_features dataframe.
    """
    for feature_df in feature_dataframes_to_merge:
        binned_features = binned_features.merge(feature_df, how='left', on='bin')
    return binned_features


def _add_catching_ff(binned_features, ff_caught_T_sorted, time_bins):
    """
    Add a column 'catching_ff' to binned_features indicating bins where ff is caught.
    """
    binned_features['catching_ff'] = 0
    catching_target_bins = np.digitize(ff_caught_T_sorted, time_bins)-1
    binned_features.loc[binned_features['bin'].isin(catching_target_bins), 'catching_ff'] = 1
    return binned_features


def _fill_na(binned_features):
    """
    Fill the NA values in binned_features with the values in the previous row and the next row.
    """
    binned_features = binned_features.ffill().reset_index(drop=True)
    binned_features = binned_features.bfill().reset_index(drop=True)
    return binned_features



def _add_any_ff_visible(final_behavioral_data):
    """
    Add a dummy variable 'any_ff_visible' to final_behavioral_data if the ratio of bins with visible ff is between 10% and 90%.
    """
    any_ff_visible = (final_behavioral_data['num_visible_ff'] > 0).astype(int)
    if (any_ff_visible.sum()/len(final_behavioral_data) > 0.1) and (any_ff_visible.sum()/len(final_behavioral_data) < 0.9):
        final_behavioral_data['any_ff_visible'] = final_behavioral_data['num_visible_ff'] > 0
    return final_behavioral_data