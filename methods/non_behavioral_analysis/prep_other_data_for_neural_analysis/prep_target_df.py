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



def retrieve_or_make_and_target_df_into_h5(data_item, h5_file_name='target_df.h5', exists_ok=True):
    h5_file_path = os.path.join(data_item.data_folder_name, h5_file_name)
    with pd.HDFStore(h5_file_path) as target_df_h5_store:
        try:
            if not exists_ok:
                raise Exception()
            target_df = target_df_h5_store['target_df']
            target_average_info = target_df_h5_store['target_average_info']
            target_min_info = target_df_h5_store['target_min_info']
            target_max_info = target_df_h5_store['target_max_info']
        except:
            target_df, target_average_info, target_min_info, target_max_info = make_target_df(data_item)
            target_df_h5_store['target_df'] = target_df
            target_df_h5_store['target_average_info'] = target_average_info
            target_df_h5_store['target_min_info'] = target_min_info
            target_df_h5_store['target_max_info'] = target_max_info
    return target_df, target_average_info, target_min_info, target_max_info



def make_target_df(data_item):
    target_df = _create_target_df(data_item.monkey_information, data_item.ff_caught_T_sorted, data_item.ff_real_position_sorted)
    target_df = _calculate_target_distance_and_angle(target_df)
    target_df, nearby_alive_ff_indices = _add_target_last_seen_info(target_df, data_item.ff_real_position_sorted, data_item.ff_caught_T_sorted, data_item.ff_life_sorted, data_item.ff_dataframe)
    target_df = _add_disappeared_for_last_time_dummy(target_df, data_item.ff_caught_T_sorted, data_item.ff_dataframe, nearby_alive_ff_indices)
    target_df = _add_visible_dummy(target_df)
    target_df = _find_time_since_last_capture(target_df, data_item.ff_caught_T_sorted)
    target_df = _add_last_seeing_target_cluster(target_df)
    target_average_info = _calculate_average_info(target_df)
    target_min_info = _calculate_min_info(target_df)
    target_max_info = _calculate_max_info(target_df)
    return target_df, target_average_info, target_min_info, target_max_info




def _create_target_df(monkey_information, ff_caught_T_sorted, ff_real_position_sorted):
    """
    Create a DataFrame with target information.
    """
    target_df = monkey_information[['bin', 'monkey_t', 'monkey_x', 'monkey_y', 'monkey_angles']].copy()
    target_df.rename(columns={'monkey_angles': 'monkey_angle', 'monkey_t': 'time'}, inplace=True)
    target_df['point_index'] = target_df.index
    target_df['target_index'] = np.digitize(target_df['time'], ff_caught_T_sorted)
    target_df['target_x'] = ff_real_position_sorted[target_df['target_index'].values, 0]
    target_df['target_y'] = ff_real_position_sorted[target_df['target_index'].values, 1]
    return target_df

def _calculate_target_distance_and_angle(target_df):
    """
    Calculate target distance and angle.
    """
    target_distance = np.sqrt((target_df['target_x'] - target_df['monkey_x'])**2 + (target_df['target_y'] - target_df['monkey_y'])**2)
    target_angle = basic_func.calculate_angles_to_ff_centers(ff_x=target_df['target_x'], ff_y=target_df['target_y'], mx=target_df['monkey_x'], my=target_df['monkey_y'], m_angle=target_df['monkey_angle'])
    target_df['target_distance'] = target_distance
    target_df['target_angle'] = target_angle
    target_df['target_angle_to_boundary'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=target_angle, distances_to_ff=target_distance)
    return target_df




def _add_target_last_seen_info(target_df, ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, ff_dataframe):
    """
    Add some target last seen info to target_df
    """
    if 'target_cluster_last_seen_time' not in target_df.columns:
        nearby_alive_ff_indices = cluster_analysis.find_target_clusters(ff_real_position_sorted, ff_caught_T_sorted, ff_life_sorted, max_distance=50)
        target_df = organize_patterns_and_features.add_target_last_seen_info_to_target_df(target_df, ff_dataframe, nearby_alive_ff_indices, use_target_cluster=True, include_frozen_info=True)
        target_df = target_df.rename(columns={'target_last_seen_time': 'target_cluster_last_seen_time', 
                                                'target_last_seen_distance': 'target_cluster_last_seen_distance', 
                                                'target_last_seen_angle': 'target_cluster_last_seen_angle', 
                                                'target_last_seen_angle_to_boundary': 'target_cluster_last_seen_angle_to_boundary',
                                                'target_last_seen_distance_frozen': 'target_cluster_last_seen_distance_frozen', 
                                                'target_last_seen_angle_frozen': 'target_cluster_last_seen_angle_frozen', 
                                                'target_last_seen_angle_to_boundary_frozen': 'target_cluster_last_seen_angle_to_boundary_frozen'})
    target_df = organize_patterns_and_features.add_target_last_seen_info_to_target_df(target_df, ff_dataframe, nearby_alive_ff_indices, use_target_cluster=False, include_frozen_info=True)
    target_df['point_index'] = target_df['point_index'].astype(int)
    return target_df, nearby_alive_ff_indices



def _add_disappeared_for_last_time_dummy(target_df, ff_caught_T_sorted, ff_dataframe, nearby_alive_ff_indices):
    """
    Add target_has_disappeared_for_last_time_dummy to target_df
    """
    target_df['target_has_disappeared_for_last_time_dummy'] = 0
    for target in range(len(ff_caught_T_sorted)):
        target_info = ff_dataframe[(ff_dataframe['ff_index']==target) & (ff_dataframe['visible']==1)]
        target_last_visible_time = target_info['time'].max()
        target_df.loc[(target_df['target_index']==target) & (target_df['time'] > target_last_visible_time), 'target_has_disappeared_for_last_time_dummy'] = 1
    target_df['target_cluster_has_disappeared_for_last_time_dummy'] = 0
    for target in range(len(ff_caught_T_sorted)):
        target_cluster_indices = nearby_alive_ff_indices[target] 
        target_info = ff_dataframe[(ff_dataframe['ff_index'].isin(target_cluster_indices)) & (ff_dataframe['visible']==1)]
        target_last_visible_time = target_info['time'].max()
        target_df.loc[(target_df['target_index']==target) & (target_df['time'] > target_last_visible_time), 'target_cluster_has_disappeared_for_last_time_dummy'] = 1
    return target_df

def _add_visible_dummy(target_df):
    """
    Add dummy variable of target being visible
    """
    target_df[['target_visible_dummy']] = 1
    target_df[['target_cluster_visible_dummy']] = 1
    target_df.loc[target_df['target_last_seen_time'] > 0, 'target_visible_dummy'] = 0
    target_df.loc[target_df['target_cluster_last_seen_time'] > 0, 'target_cluster_visible_dummy'] = 0
    return target_df

def _find_time_since_last_capture(target_df, ff_caught_T_sorted):
    """
    Find time_since_last_capture
    """
    if target_df.target_index.unique().max() >= len(ff_caught_T_sorted)-1:
        num_exceeding_target = target_df.target_index.unique().max() - (len(ff_caught_T_sorted)-1)
        ff_caught_T_sorted_temp = np.concatenate((ff_caught_T_sorted, np.repeat(target_df.time.max(), num_exceeding_target)))
    else:
        ff_caught_T_sorted_temp = ff_caught_T_sorted.copy()
    target_df['current_target_caught_time'] = ff_caught_T_sorted_temp[target_df['target_index']]
    target_df['last_target_caught_time'] = ff_caught_T_sorted_temp[target_df['target_index']-1]
    target_df.loc[target_df['target_index']==0, 'last_target_caught_time'] = 0
    target_df['time_since_last_capture'] = target_df['time'] - target_df['last_target_caught_time']
    return target_df

def _add_last_seeing_target_cluster(target_df):
    """
    Add to target_df dummy variable of being in the last duration of seeing the target cluster
    """
    target_df['while_last_seeing_target_cluster'] = 0
    for target in target_df.target_index.unique():
        target_subset = target_df[target_df.target_index == target]
        if len(target_subset) > 0:
            dif = np.diff(target_subset['target_cluster_visible_dummy'])
            becoming_visible_points = np.where(dif == 1)[0]
            if len(becoming_visible_points) > 0:
                starting_index = becoming_visible_points[-1]+1
            elif target_subset['target_cluster_visible_dummy'].iloc[0] == 1: 
                starting_index = 0
            else: 
                continue
            stop_being_visible_points = np.where(dif == -1)[0]
            if len(stop_being_visible_points) > 0:
                ending_index = stop_being_visible_points[-1]+1
                if ending_index < starting_index:
                    ending_index = len(target_subset)
            else:
                ending_index = len(target_subset)
            target_df.loc[target_subset.iloc[starting_index:ending_index].index, 'while_last_seeing_target_cluster'] = 1
    return target_df


def _calculate_average_info(target_df):
    """
    Calculate average information for each bin in target_df
    """
    target_average_info = target_df[['bin', 'target_distance', 'target_angle', 'target_angle_to_boundary', \
                                    'target_last_seen_time', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary',\
                                    'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
                                    'target_cluster_last_seen_time', 'target_cluster_last_seen_distance', 'target_cluster_last_seen_angle', 'target_cluster_last_seen_angle_to_boundary',\
                                    'target_cluster_last_seen_distance_frozen', 'target_cluster_last_seen_angle_frozen', 'target_cluster_last_seen_angle_to_boundary_frozen',]].copy()

    target_average_info = target_average_info.groupby('bin').mean().reset_index(drop=False)
    target_average_info.rename(columns={'target_distance': 'avg_target_distance',
                                        'target_angle': 'avg_target_angle',
                                        'target_angle_to_boundary': 'avg_target_angle_to_boundary',
                                        'target_last_seen_time': 'avg_target_last_seen_time',
                                        'target_last_seen_distance': 'avg_target_last_seen_distance',
                                        'target_last_seen_angle': 'avg_target_last_seen_angle',
                                        'target_last_seen_angle_to_boundary': 'avg_target_last_seen_angle_to_boundary',
                                        'target_last_seen_distance_frozen': 'avg_target_last_seen_distance_frozen',
                                        'target_last_seen_angle_frozen': 'avg_target_last_seen_angle_frozen',
                                        'target_last_seen_angle_to_boundary_frozen': 'avg_target_last_seen_angle_to_boundary_frozen',
                                        'target_cluster_last_seen_time': 'avg_target_cluster_last_seen_time',
                                        'target_cluster_last_seen_distance': 'avg_target_cluster_last_seen_distance',
                                        'target_cluster_last_seen_angle': 'avg_target_cluster_last_seen_angle',
                                        'target_cluster_last_seen_angle_to_boundary': 'avg_target_cluster_last_seen_angle_to_boundary',
                                        'target_cluster_last_seen_distance_frozen': 'avg_target_cluster_last_seen_distance_frozen',
                                        'target_cluster_last_seen_angle_frozen': 'avg_target_cluster_last_seen_angle_frozen',
                                        'target_cluster_last_seen_angle_to_boundary_frozen': 'avg_target_cluster_last_seen_angle_to_boundary_frozen'
                                        }, inplace=True)
    return target_average_info

def _calculate_min_info(target_df):
    """
    Calculate minimum information for each bin in target_df
    """
    target_min_info = target_df[['bin', 'target_has_disappeared_for_last_time_dummy', 'target_cluster_has_disappeared_for_last_time_dummy']].copy()
    target_min_info = target_min_info.groupby('bin').min().reset_index(drop=False)
    target_min_info.rename(columns={'target_has_disappeared_for_last_time_dummy': 'min_target_has_disappeared_for_last_time_dummy',
                                    'target_cluster_has_disappeared_for_last_time_dummy': 'min_target_cluster_has_disappeared_for_last_time_dummy',
                                    }, inplace=True)
    return target_min_info

def _calculate_max_info(target_df):
    """
    Calculate maximum information for each bin in target_df
    """
    target_max_info = target_df[['bin', 'target_visible_dummy', 'target_cluster_visible_dummy']].copy()
    target_max_info = target_max_info.groupby('bin').max().reset_index(drop=False)
    target_max_info.rename(columns={'target_visible_dummy': 'max_target_visible_dummy',
                                    'target_cluster_visible_dummy': 'max_target_cluster_visible_dummy'}, inplace=True)
    return target_max_info



