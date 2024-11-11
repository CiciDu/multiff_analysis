import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import process_raw_data, basic_func
from null_behaviors import curvature_utils, curv_of_traj_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
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





def prepare_ff_data(data_item, time_bins):
    """Prepare ff data."""
    ff_dataframe = _prepare_ff_dataframe(data_item, time_bins)
    ff_dataframe_unique_ff = _count_visible_and_in_memory_ff(ff_dataframe)
    ff_dataframe_unique_visible_ff = _count_visible_ff(ff_dataframe)
    min_ff_info_df = _min_ff_info(ff_dataframe)
    min_visible_ff_info_df = _min_visible_ff_info(ff_dataframe)
    return ff_dataframe_unique_ff, ff_dataframe_unique_visible_ff, min_ff_info_df, min_visible_ff_info_df


def _prepare_ff_dataframe(data_item, time_bins):
    """Prepare ff dataframe."""
    ff_dataframe = data_item.ff_dataframe.copy()
    ff_dataframe['bin'] = np.digitize(ff_dataframe.time, time_bins)-1
    ff_dataframe['point_index'] = ff_dataframe['point_index'].astype(int)
    ff_dataframe['monkey_angle'] = data_item.monkey_information.loc[:, 'monkey_angles'].values[ff_dataframe.loc[:, 'point_index'].values]
    return ff_dataframe

def _count_visible_and_in_memory_ff(ff_dataframe):
    """Count of visible and in-memory ff."""
    ff_dataframe_sub = ff_dataframe[['bin', 'ff_index']]
    ff_dataframe_unique_ff = ff_dataframe_sub.groupby('bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_ff.rename(columns={'ff_index': 'num_alive_ff'}, inplace=True)
    return ff_dataframe_unique_ff

def _count_visible_ff(ff_dataframe):
    """Count of visible ff."""
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
    ff_dataframe_unique_visible_ff = ff_dataframe_visible[['bin', 'ff_index']]
    ff_dataframe_unique_visible_ff = ff_dataframe_unique_visible_ff.groupby('bin').nunique().reset_index(drop=False)
    ff_dataframe_unique_visible_ff.rename(columns={'ff_index': 'num_visible_ff'}, inplace=True)
    return ff_dataframe_unique_visible_ff

def _min_ff_info(ff_dataframe):
    """Min ff info."""
    min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_ff_info = min_ff_info.groupby('bin').min().reset_index(drop=False)
    min_ff_info.rename(columns={'ff_distance': 'min_ff_distance',
                                'ff_angle': 'min_abs_ff_angle',
                                'ff_angle_boundary': 'min_abs_ff_angle_boundary'}, inplace=True)
    return min_ff_info

def _min_visible_ff_info(ff_dataframe):
    """Min visible ff info."""
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
    min_visible_ff_info = ff_dataframe_visible[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
    min_visible_ff_info = min_visible_ff_info.groupby('bin').min().reset_index(drop=False)
    min_visible_ff_info.rename(columns={'ff_distance': 'min_visible_ff_distance',
                                        'ff_angle': 'min_abs_visible_ff_angle', 
                                        'ff_angle_boundary': 'min_abs_visible_ff_angle_boundary'}, inplace=True)
    return min_visible_ff_info


