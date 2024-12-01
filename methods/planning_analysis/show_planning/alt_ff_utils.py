
from data_wrangling import basic_func
from pattern_discovery import cluster_analysis
from pattern_discovery import monkey_landing_in_ff

import pandas as pd
import numpy as np


def get_all_alt_ff_df_from_ff_dataframe(stops_near_ff_df, ff_dataframe_visible, 
                                        closest_stop_to_capture_df,
                                        ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, monkey_information,
                                        min_time_between_stop_and_alt_ff_caught_time=0.1,
                                        min_distance_between_stop_and_alt_ff=25, 
                                        max_distance_between_stop_and_alt_ff=500,
                                        stop_period_duration=2):

    # Prepare the dataframe
    all_alt_ff_df = stops_near_ff_df[['stop_ff_index', 'stop_point_index', 'stop_cum_distance', 'stop_time', 'stop_x', 'stop_y']].copy()
    all_alt_ff_df['beginning_time'] = all_alt_ff_df['stop_time'] - stop_period_duration
    all_alt_ff_df['data_category_by_vis'] = 'test' # can be 'test', 'control', or 'neither


    # Set 'alt_ff_index' to the index of the caught ff after stop
    all_alt_ff_df['alt_ff_index'] = all_alt_ff_df['stop_ff_index'] + 1
    all_alt_ff_df = all_alt_ff_df[all_alt_ff_df['alt_ff_index'] < len(ff_caught_T_new)].copy()
    all_alt_ff_df['alt_ff_caught_time'] = ff_caught_T_new[all_alt_ff_df['alt_ff_index'].values]

    # get next_stop_time and next_stop_point_index based on alt ff
    closest_stop_to_capture_df2 = closest_stop_to_capture_df[['stop_ff_index', 'stop_point_index', 'stop_time']].copy()
    closest_stop_to_capture_df2.rename(columns={'stop_ff_index': 'alt_ff_index',
                                                'stop_time': 'next_stop_time',
                                                'stop_point_index': 'next_stop_point_index'}, inplace=True)
    all_alt_ff_df = all_alt_ff_df.merge(closest_stop_to_capture_df2, on='alt_ff_index', how='left')

    all_alt_ff_df['stop_ff_capture_time'] = ff_caught_T_new[all_alt_ff_df['stop_ff_index'].values]
    original_length = len(all_alt_ff_df)
    all_alt_ff_df = all_alt_ff_df[all_alt_ff_df['next_stop_time'] - all_alt_ff_df['stop_time'] >= 
                                        min_time_between_stop_and_alt_ff_caught_time].copy()
    print(f'{original_length - len(all_alt_ff_df)} rows out of {original_length} rows were removed from all_alt_ff_df because the time between stop and next stop is less than {min_time_between_stop_and_alt_ff_caught_time} seconds')
    
    all_alt_ff_df[['ff_x', 'ff_y']] = ff_real_position_sorted[all_alt_ff_df['alt_ff_index'].values]


    all_alt_ff_df['alt_ff_distance_to_next_stop'] = np.linalg.norm(all_alt_ff_df[['ff_x', 'ff_y']].values - 
                                                            monkey_information.loc[all_alt_ff_df['next_stop_point_index'], ['monkey_x', 'monkey_y']].values, axis=1)

    # Calculate the distance from the stop ff to the alt ff
    all_alt_ff_df['d_from_stop_ff_to_alt_ff'] = np.linalg.norm(ff_real_position_sorted[all_alt_ff_df['stop_ff_index'].values] - all_alt_ff_df[['ff_x', 'ff_y']].values, axis=1)
    all_alt_ff_df.loc[all_alt_ff_df['d_from_stop_ff_to_alt_ff'] <= min_distance_between_stop_and_alt_ff, 'data_category_by_vis'] = 'neither'
    print(f'{len(all_alt_ff_df[all_alt_ff_df["data_category_by_vis"]=="neither"])} rows out of {len(all_alt_ff_df)} rows were not used from all_alt_ff_df because the distance between stop and alt ff is smaller than {min_distance_between_stop_and_alt_ff}cm')

    all_alt_ff_df.loc[all_alt_ff_df['d_from_stop_ff_to_alt_ff'] >= max_distance_between_stop_and_alt_ff, 'data_category_by_vis'] = 'neither'
    print(f'{len(all_alt_ff_df[all_alt_ff_df["data_category_by_vis"]=="neither"])} rows out of {len(all_alt_ff_df)} rows were not used from all_alt_ff_df because the distance between stop and alt ff is larger than {max_distance_between_stop_and_alt_ff}cm')

    all_alt_ff_df = add_alt_ff_first_and_last_seen_info(all_alt_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted, ff_life_sorted)

    all_alt_ff_df['alt_ff_last_seen_rel_time_bbas'] = all_alt_ff_df['ALT_time_ff_last_seen_bbas'] - all_alt_ff_df['beginning_time']

    # # See if alt ff was visible before stop; if they are not, they will be assigned control rather than test
    all_alt_ff_df.loc[(all_alt_ff_df['alt_ff_last_seen_rel_time_bbas'].isnull()) & 
                      (all_alt_ff_df['data_category_by_vis'] != 'neither'), 'data_category_by_vis'] = 'control'

    # Preserve necessary columns
    columns_to_preserve = ['alt_ff_index', 'stop_ff_index', 'stop_point_index', 'data_category_by_vis', 'alt_ff_caught_time',
                           'next_stop_point_index', 'next_stop_time', 'alt_ff_distance_to_next_stop',
                           'd_from_stop_ff_to_alt_ff', 'alt_ff_cluster_last_seen_rel_time_bbas',
                           'ALT_point_index_ff_first_seen_bbas', 'ALT_point_index_ff_last_seen_bbas',
                           'ALT_monkey_angle_ff_first_seen_bbas', 'ALT_monkey_angle_ff_last_seen_bbas',
                           'ALT_time_ff_first_seen_bbas', 'ALT_time_ff_last_seen_bbas',
                           'ALT_time_ff_first_seen_bsans', 'ALT_time_ff_last_seen_bsans', 
                           'alt_ff_cluster_last_seen_time_bbas', 'alt_ff_cluster_last_seen_time_bsans',
                           'alt_ff_last_seen_rel_time_bbas']

    all_alt_ff_df = all_alt_ff_df[columns_to_preserve].copy()   


    all_alt_ff_df['alt_ff_index'] = all_alt_ff_df['alt_ff_index'].astype(int)

    return all_alt_ff_df



def add_alt_ff_first_and_last_seen_info(all_alt_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted, ff_life_sorted):
    all_alt_ff_df = all_alt_ff_df.copy()
    all_alt_ff_df = add_alt_ff_first_seen_and_last_seen_info_bbas(all_alt_ff_df, ff_dataframe_visible, monkey_information)
    all_alt_ff_df = add_alt_ff_first_seen_and_last_seen_info_bsans(all_alt_ff_df, ff_dataframe_visible, monkey_information)
    
    all_alt_ff_df = get_alt_ff_cluster_last_seen_bbas(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted)
    all_alt_ff_df = get_alt_ff_cluster_last_seen_bsans(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted)
    return all_alt_ff_df


def get_closest_stop_time_to_all_capture_time(ff_caught_T_sorted, monkey_information, ff_real_position_sorted, stop_ff_index_array=None, stop_point_index_array=None):
    stop_sub = monkey_information.loc[monkey_information['monkey_speeddummy']==0, ['time', 'point_index']].copy()
    closest_stop_to_capture_df = pd.DataFrame()
    for i in range(len(ff_caught_T_sorted)):
        caught_time = ff_caught_T_sorted[i]
        monkey_t = stop_sub['time'].values
        iloc_of_closest_time = np.argmin(np.abs(monkey_t - caught_time))
        closest_point_row = stop_sub.iloc[[iloc_of_closest_time]].copy()
        closest_point_row['caught_time'] = caught_time
        closest_stop_to_capture_df = pd.concat([closest_stop_to_capture_df, closest_point_row], axis=0)
    closest_stop_to_capture_df['diff_from_caught_time'] = closest_stop_to_capture_df['time'] - closest_stop_to_capture_df['caught_time']
    if stop_ff_index_array is not None:
        closest_stop_to_capture_df['stop_ff_index'] = stop_ff_index_array
    else:
        closest_stop_to_capture_df['stop_ff_index'] = np.arange(len(ff_caught_T_sorted))
    if stop_point_index_array is not None:
        closest_stop_to_capture_df['stop_point_index'] = stop_point_index_array
    else:
        closest_stop_to_capture_df['stop_point_index'] = closest_stop_to_capture_df['point_index']
    closest_stop_to_capture_df['stop_time'] = monkey_information.loc[closest_stop_to_capture_df['point_index'], 'monkey_t'].values

    closest_stop_to_capture_df = monkey_landing_in_ff.add_distance_from_ff_to_stop(closest_stop_to_capture_df, monkey_information, ff_real_position_sorted)
    closest_stop_to_capture_df['whether_stop_inside_boundary'] = closest_stop_to_capture_df['distance_from_ff_to_stop'] <= 25
    closest_stop_to_capture_df.sort_values(by='stop_ff_index', inplace=True)
    return closest_stop_to_capture_df


def drop_rows_where_stop_is_not_inside_reward_boundary(closest_stop_to_capture_df):
    original_length = len(closest_stop_to_capture_df)
    outlier_sub_df = closest_stop_to_capture_df[closest_stop_to_capture_df['distance_from_ff_to_stop'] > 25].sort_values(by='distance_from_ff_to_stop', ascending=False)
    closest_stop_to_capture_df = closest_stop_to_capture_df[closest_stop_to_capture_df['distance_from_ff_to_stop'] <= 25].copy()
    
    print(f'{original_length - len(closest_stop_to_capture_df)} rows out of {original_length} rows were removed from closest_stop_to_capture_df because the distance between stop and ff center is larger than 25cm, '\
        #   + f'\n which is {round((original_length - len(closest_stop_to_capture_df))/original_length*100, 2)}% of the rows, '\
        #   + f'and the sorted distances from those are {outlier_sub_df["distance_from_ff_to_stop"].values}'
            )  
    return closest_stop_to_capture_df

def get_all_captured_ff_first_seen_and_last_seen_info(closest_stop_to_capture_df, stop_period_duration, ff_dataframe_visible, monkey_information, drop_na=False):
    if 'stop_ff_index' in closest_stop_to_capture_df.columns:
        all_ff_index = closest_stop_to_capture_df['stop_ff_index'].values
    else:
        all_ff_index = closest_stop_to_capture_df['ff_index'].values

    if 'stop_point_index' in closest_stop_to_capture_df.columns:
        all_stop_point_index = closest_stop_to_capture_df['stop_point_index'].values
    else:
        all_stop_point_index = closest_stop_to_capture_df['point_index'].values

    if 'stop_time' in closest_stop_to_capture_df.columns:
        all_end_time = closest_stop_to_capture_df['stop_time'].values
    else:
        all_end_time = closest_stop_to_capture_df['time'].values

    all_start_time = all_end_time - stop_period_duration
    
    ff_info = get_first_seen_and_last_seen_info_for_ff_in_time_windows(all_ff_index, all_stop_point_index, all_start_time,
                                                                       all_end_time, ff_dataframe_visible, monkey_information, drop_na=drop_na)

    return ff_info

def rename_first_and_last_seen_info_columns(df, prefix='STOP_'):
    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                                        'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen', 
                                        'time_ff_first_seen', 'time_ff_last_seen']
    columns_to_be_renamed = {column: prefix + column + '_bbas' for column in columns_to_add}
    # Note: bbas means "between start and stop"
    df.rename(columns=columns_to_be_renamed, inplace=True)
    return df


def add_alt_ff_first_seen_and_last_seen_info_bbas(all_alt_ff_df, ff_dataframe_visible, monkey_information):
    all_alt_ff_df = _add_stop_or_alt_ff_first_seen_and_last_seen_info_bbas(all_alt_ff_df, ff_dataframe_visible, monkey_information, stop_or_alt='alt')
    return all_alt_ff_df



def _add_stop_or_alt_ff_first_seen_and_last_seen_info_bbas(df, ff_dataframe_visible, monkey_information,
                                                           stop_or_alt='alt'):
    all_stop_time = df['stop_time'].values
    all_start_time = df['beginning_time'].values
    ff_index_column = stop_or_alt + '_ff_index'
    alt_ff_first_and_last_seen_info = get_first_seen_and_last_seen_info_for_ff_in_time_windows(df[ff_index_column].values, df['stop_point_index'].values, all_start_time, 
                                                                                                all_stop_time, ff_dataframe_visible, monkey_information)

    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                        'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen', 
                        'time_ff_first_seen', 'time_ff_last_seen']
    alt_ff_first_and_last_seen_info = alt_ff_first_and_last_seen_info[columns_to_add + ['stop_point_index']]
    prefix = 'ALT_' if stop_or_alt == 'alt' else 'STOP_'
    columns_to_be_renamed_dict = {column: prefix + column + '_bbas' for column in columns_to_add}
    alt_ff_first_and_last_seen_info.rename(columns=columns_to_be_renamed_dict, inplace=True)
    df = df.merge(alt_ff_first_and_last_seen_info, on='stop_point_index', how='left')
    return df



def add_alt_ff_first_seen_and_last_seen_info_bsans(all_alt_ff_df, ff_dataframe_visible, monkey_information):

    alt_ff_first_and_last_seen_info = get_first_seen_and_last_seen_info_for_ff_in_time_windows(all_alt_ff_df['alt_ff_index'].values, all_alt_ff_df['stop_point_index'].values, all_alt_ff_df['stop_time'].values, 
                                                                                                all_alt_ff_df['next_stop_time'].values, ff_dataframe_visible, monkey_information)

    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                                        'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen', 
                                        'time_ff_first_seen', 'time_ff_last_seen']
    alt_ff_first_and_last_seen_info = alt_ff_first_and_last_seen_info[columns_to_add + ['stop_point_index']]
    columns_to_be_renamed_dict = {column: 'ALT_' + column + '_bsans' for column in columns_to_add}
    alt_ff_first_and_last_seen_info.rename(columns=columns_to_be_renamed_dict, inplace=True)
    all_alt_ff_df = all_alt_ff_df.merge(alt_ff_first_and_last_seen_info, on='stop_point_index', how='left')
    return all_alt_ff_df

def get_alt_ff_last_seen_rel_time(all_alt_ff_df, ff_dataframe, stop_period_duration=2):
    # See if alt ff was visible before stop; if they are not, they will be assigned control rather than test
    all_ff_index = all_alt_ff_df['alt_ff_index'].values
    all_point_index = all_alt_ff_df['stop_point_index'].values
    all_start_time = all_alt_ff_df['stop_time'].values - stop_period_duration
    all_alt_ff_df['beginning_time'] = all_start_time
    all_end_time = all_alt_ff_df['stop_time'].values
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()
    alt_ff_last_seen_info = find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_point_index, all_start_time, 
                                                                                                     all_end_time, ff_dataframe_visible, first_or_last='last')
    alt_ff_last_seen_info.rename(columns={'ff_index': 'alt_ff_index', 'time': 'alt_ff_last_seen_time'}, inplace=True)
    
    all_alt_ff_df = all_alt_ff_df.merge(alt_ff_last_seen_info[['alt_ff_index', 'alt_ff_last_seen_time']], on='alt_ff_index', how='left')
    all_alt_ff_df['alt_ff_last_seen_rel_time_bbas'] = all_alt_ff_df['alt_ff_last_seen_time'] - all_alt_ff_df['beginning_time']
    return all_alt_ff_df



def get_alt_ff_cluster_last_seen_info(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                        start_time_column='beginning_time', end_time_column='stop_time'):
    # See if alt ff was visible before stop; if they are not, they will be assigned control rather than test
    
    if 'alt_ff_cluster' not in all_alt_ff_df.columns:
        all_alt_ff_df[['alt_ff_x', 'alt_ff_y']] = ff_real_position_sorted[all_alt_ff_df['alt_ff_index'].values]
        all_alt_ff_df['alt_ff_cluster'] = cluster_analysis.find_alive_ff_clusters(all_alt_ff_df[['alt_ff_x', 'alt_ff_y']].values, 
                                                                            ff_real_position_sorted, all_alt_ff_df['beginning_time'].values, 
                                                                            all_alt_ff_df['next_stop_time'].values,
                                                                            ff_life_sorted, max_distance=50)
    all_alt_ff_df['alt_ff_cluster_size'] = all_alt_ff_df['alt_ff_cluster'].apply(len)
    
    all_ff_index = []
    [all_ff_index.extend(array) for array in all_alt_ff_df['alt_ff_cluster'].tolist()]
    all_ff_index = np.array(all_ff_index)

    all_point_index = np.repeat(all_alt_ff_df['stop_point_index'].values, all_alt_ff_df['alt_ff_cluster_size'].values)
    all_end_time = np.repeat(all_alt_ff_df[end_time_column].values, all_alt_ff_df['alt_ff_cluster_size'].values)
    all_start_time = np.repeat(all_alt_ff_df[start_time_column].values, all_alt_ff_df['alt_ff_cluster_size'].values)
    alt_ff_last_seen_info = find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_point_index, all_start_time, 
                                                                                                     all_end_time, ff_dataframe_visible, first_or_last='last')
    # for each stop_point_index, find the latest last-seen time 
    # before sorting, we need to drop rows with NA
    alt_ff_last_seen_info.dropna(axis=0, inplace=True)
    alt_ff_last_seen_info.sort_values(by=['stop_point_index', 'time'], ascending=[True, True], inplace=True)
    alt_ff_last_seen_info = alt_ff_last_seen_info.groupby('stop_point_index').last().reset_index(drop=False)
    alt_ff_last_seen_info = alt_ff_last_seen_info[['stop_point_index', 'ff_index', 'time']].copy()
    
    return alt_ff_last_seen_info



def get_alt_ff_cluster_last_seen_bbas(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted):
    # See if alt ff was visible before stop
    alt_ff_last_seen_info = get_alt_ff_cluster_last_seen_info(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                                                start_time_column='beginning_time', end_time_column='stop_time')
        
    alt_ff_last_seen_info.rename(columns={'ff_index': 'ff_index_last_seen_bbas_in_alt_ff_cluster', 
                                            'time': 'alt_ff_cluster_last_seen_time_bbas'}, inplace=True)   
    all_alt_ff_df = all_alt_ff_df.merge(alt_ff_last_seen_info, on='stop_point_index', how='left').reset_index(drop=True)
    
    all_alt_ff_df['alt_ff_cluster_last_seen_rel_time_bbas'] = all_alt_ff_df['alt_ff_cluster_last_seen_time_bbas'] - all_alt_ff_df['beginning_time']

    return all_alt_ff_df


def get_alt_ff_cluster_last_seen_bsans(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted):
    # See if alt ff was visible before stop
    alt_ff_last_seen_info = get_alt_ff_cluster_last_seen_info(all_alt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                                                start_time_column='stop_time', end_time_column='next_stop_time')
        
    alt_ff_last_seen_info.rename(columns={'ff_index': 'ff_index_last_seen_bsans_in_alt_ff_cluster', 
                                            'time': 'alt_ff_cluster_last_seen_time_bsans'}, inplace=True)   
    all_alt_ff_df = all_alt_ff_df.merge(alt_ff_last_seen_info, on='stop_point_index', how='left').reset_index(drop=True)
    
    all_alt_ff_df['alt_ff_cluster_last_seen_rel_time_bsans'] = all_alt_ff_df['alt_ff_cluster_last_seen_time_bsans'] - all_alt_ff_df['beginning_time']

    return all_alt_ff_df


def _get_alt_ff_df_or_stop_ff_df(shared_stops_near_ff_df, alt_or_stop='alt'):
    shared_columns = ['stop_point_index', 'stop_time', 'stop_cum_distance']
    ff_column = [alt_or_stop + '_ff_index']
    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                    'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen', 
                    'time_ff_first_seen', 'time_ff_last_seen']
    columns_to_add = [alt_or_stop.upper() + '_' + column + '_bbas' for column in columns_to_add]  
    all_relevant_columns = shared_columns + ff_column + columns_to_add
    ff_df = shared_stops_near_ff_df[all_relevant_columns].copy()
    prefix_len = len(alt_or_stop + '_')
    columns_to_be_renamed = {column: column[prefix_len:-5] for column in columns_to_add}
    columns_to_be_renamed[ff_column[0]] = 'ff_index'
    ff_df.rename(columns=columns_to_be_renamed, inplace=True)
    ff_df.reset_index(drop=True, inplace=True)
    return ff_df



def get_alt_ff_df_and_stop_ff_df(stops_near_ff_df):

    alt_ff_df = _get_alt_ff_df_or_stop_ff_df(stops_near_ff_df, alt_or_stop='alt')
    stop_ff_df = _get_alt_ff_df_or_stop_ff_df(stops_near_ff_df, alt_or_stop='stop')

    stops_near_ff_df['earlest_point_index_when_alt_ff_and_stop_ff_have_both_been_seen_bbas'] = np.stack([alt_ff_df['point_index_ff_first_seen'].values, stop_ff_df['point_index_ff_first_seen'].values]).max(axis=0) 
    return stops_near_ff_df, alt_ff_df, stop_ff_df


def get_info_for_ff_based_on_stop_period_time_window(stops_near_ff_df, all_ff_index, ff_dataframe_visible, monkey_information, stop_period_duration=2):
    all_stop_point_index, all_stop_time = stops_near_ff_df['stop_point_index'].values, stops_near_ff_df['stop_time'].values
    all_start_time = all_stop_time - stop_period_duration

    ff_info = get_first_seen_and_last_seen_info_for_ff_in_time_windows(all_ff_index, all_stop_point_index, all_start_time, 
                                                                       all_stop_time, ff_dataframe_visible, monkey_information)

    #ff_info['duration_of_ff_unseen'] = ff_info['stop_time'] - ff_info['time_ff_last_seen']
    return ff_info


def get_first_seen_and_last_seen_info_for_ff_in_time_windows(all_ff_index, all_stop_point_index, all_start_time, all_end_time, ff_dataframe_visible, monkey_information, verbose=True, drop_na=False):
    ff_last_seen_info = find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_stop_point_index, all_start_time, all_end_time, ff_dataframe_visible, first_or_last='last')
    ff_first_seen_info = find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_stop_point_index, all_start_time, all_end_time, ff_dataframe_visible, first_or_last='first')
    
    ff_info = pd.DataFrame({'ff_index': all_ff_index, 'stop_point_index': all_stop_point_index})
    ff_info[['point_index_ff_first_seen', 'monkey_angle_ff_first_seen']] = ff_first_seen_info[['point_index', 'monkey_angle']].values
    ff_info[['point_index_ff_last_seen', 'monkey_angle_ff_last_seen']] = ff_last_seen_info[['point_index', 'monkey_angle']].values

    if drop_na:
        if ff_info.isnull().values.any():
            num_null_rows = len(ff_info[ff_info.isnull().any(axis=1)])
            if num_null_rows > 0:
                print(f'Warning: There are {num_null_rows} rows out of {len(ff_info)} rows in ff_info that have null values because they were not visible in the stop period. They will be dropped')
                ff_info.dropna(axis=0, inplace=True)  

    point_index_ff_first_seen = ff_info.loc[~ff_info['point_index_ff_first_seen'].isnull(), 'point_index_ff_first_seen']
    ff_info.loc[~ff_info['point_index_ff_first_seen'].isnull(), 'point_index_ff_first_seen'] = point_index_ff_first_seen.astype(int)
    point_index_ff_last_seen = ff_info.loc[~ff_info['point_index_ff_last_seen'].isnull(), 'point_index_ff_last_seen']
    ff_info.loc[~ff_info['point_index_ff_last_seen'].isnull(), 'point_index_ff_last_seen'] = point_index_ff_last_seen.astype(int)

    ff_info.loc[~ff_info['point_index_ff_first_seen'].isnull(), 'time_ff_first_seen'] = monkey_information.loc[point_index_ff_first_seen.values, 'monkey_t'].values
    ff_info.loc[~ff_info['point_index_ff_last_seen'].isnull(), 'time_ff_last_seen'] = monkey_information.loc[point_index_ff_last_seen.values, 'monkey_t'].values

    return ff_info


def find_info_when_ff_was_first_or_last_seen_between_start_time_and_end_time(all_ff_index, all_point_index, all_start_time, all_end_time, ff_dataframe_visible, first_or_last='first'):
    temp_ff_df = pd.DataFrame({'ff_index': all_ff_index, 
                               'beginning_time': all_start_time, 
                               'end_time': all_end_time, 
                               'stop_point_index': all_point_index})
    ff_info = pd.merge(temp_ff_df, ff_dataframe_visible, on='ff_index', how='inner')
    ff_info = ff_info[ff_info['time'].between(ff_info['beginning_time'], ff_info['end_time'], inclusive='left')].copy()
    ff_info = ff_info.sort_values(by=['stop_point_index', 'ff_index', 'time'])
    if first_or_last == 'first':
        ff_info = ff_info.groupby(['stop_point_index', 'ff_index']).head(1).reset_index(drop=True)
    else:
        ff_info = ff_info.groupby(['stop_point_index', 'ff_index']).tail(1).reset_index(drop=True)
    
    # make sure that all ff_index and point_index given in the input are covered
    ff_info = pd.merge(temp_ff_df[['ff_index', 'stop_point_index']], ff_info, on=['ff_index', 'stop_point_index'], how='left')
    # # if there is any row with null value in the ff_info, then give a warning
    # if ff_info.isnull().values.any():
    #     num_null_rows = len(ff_info[ff_info.isnull().any(axis=1)])
    #     if len(ff_info[ff_info.isnull().any(axis=1)]) > 0:
    #         # show a warning about # rows in df that has null values
    #         print(f'Warning: There are {num_null_rows} rows out of {len(ff_info)} rows in ff_info that have null values.')
    #         ff_info.dropna(axis=0, inplace=True)
    return ff_info



def find_within_n_cm_to_point_info(point_x, point_y, ff_info_in_duration, n_cm=300):
    within_n_cm_to_point_info = ff_info_in_duration.copy()
    within_n_cm_to_point_info['distance_to_point'] = np.linalg.norm([within_n_cm_to_point_info['ff_x'] - point_x, within_n_cm_to_point_info['ff_y'] - point_y], axis=0)
    within_n_cm_to_point_info = ff_info_in_duration[within_n_cm_to_point_info['distance_to_point'] <= n_cm].copy()
    return within_n_cm_to_point_info



def find_if_alt_ff_cluster_visible_pre_stop(stops_near_ff_df_ctrl, ff_dataframe, ff_real_position_sorted, 
                                         max_distance_between_ffs_in_cluster=50, duration_prior_to_stop_time=3):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1].copy()
    
    if_alt_ff_cluster_visible_pre_stop = []
    for index, row in stops_near_ff_df_ctrl.iterrows():
        visible_ff_info_in_duration = ff_dataframe_visible[ff_dataframe_visible['time'].between(row['stop_time']-duration_prior_to_stop_time, row['stop_time'])].copy()
        alt_ff_xy = ff_real_position_sorted[row['alt_ff_index']]
        alt_ff_cluster_info = find_within_n_cm_to_point_info(alt_ff_xy[0], alt_ff_xy[1], visible_ff_info_in_duration, n_cm=max_distance_between_ffs_in_cluster)
        #alt_ff_cluster_info = alt_ff_cluster_info[alt_ff_cluster_info['ff_index'] != row['alt_ff_index']]
        if len(alt_ff_cluster_info) > 0:
            if_alt_ff_cluster_visible_pre_stop.append(True)
        else:
            if_alt_ff_cluster_visible_pre_stop.append(False)
    if_alt_ff_cluster_visible_pre_stop = np.array(if_alt_ff_cluster_visible_pre_stop)
    stops_near_ff_df_ctrl['if_alt_ff_cluster_visible_pre_stop'] = if_alt_ff_cluster_visible_pre_stop
    return stops_near_ff_df_ctrl


def add_if_alt_ff_and_alt_ff_cluster_flash_bbas(df, ff_real_position_sorted, ff_flash_sorted, ff_life_sorted, stop_period_duration=2):

    df['beginning_time'] = df['stop_time'] - stop_period_duration

    flash_on_columns = _get_flash_on_columns(df, ff_flash_sorted,
                                             ff_real_position_sorted,
                                             ff_life_sorted,
                                            duration_start_column='beginning_time',
                                            duration_end_column='stop_time',
                                            )


    flash_on_columns_df = pd.DataFrame(flash_on_columns, index=df.index)
    flash_on_columns_df['alt_ff_last_flash_time_bbas'] = flash_on_columns_df['alt_ff_last_flash_time_bbas'].replace(-999, np.nan)
    flash_on_columns_df['alt_ff_cluster_last_flash_time_bbas'] = flash_on_columns_df['alt_ff_cluster_last_flash_time_bbas'].replace(-999, np.nan)
    df = pd.concat([df, flash_on_columns_df], axis=1)

    num_if_alt_ff_cluster_flash_bbas = sum(flash_on_columns['if_alt_ff_cluster_flash_bbas'])
    print(f'Percentage of control rows that have alt ff cluster flashed on between stop_time - {stop_period_duration} and stop: \
            {round(num_if_alt_ff_cluster_flash_bbas/len(df)*100, 2)} % of {len(df)} rows.')

    return df



def add_if_alt_ff_and_alt_ff_cluster_flash_bsans(df, ff_real_position_sorted, ff_flash_sorted, ff_life_sorted):
    flash_on_columns = _get_flash_on_columns(df, ff_flash_sorted,
                                            ff_real_position_sorted,
                                            ff_life_sorted,
                                            duration_start_column='stop_time',
                                            duration_end_column='next_stop_time',
                                            )

    # for all column names in flash_on_columns, repalce basa with bsans
    flash_on_columns = {key.replace('bbas', 'bsans'): value for key, value in flash_on_columns.items()}
    
    flash_on_columns_df = pd.DataFrame(flash_on_columns, index=df.index)
    flash_on_columns_df['alt_ff_last_flash_time_bsans'] = flash_on_columns_df['alt_ff_last_flash_time_bsans'].replace(-999, np.nan)
    flash_on_columns_df['alt_ff_cluster_last_flash_time_bsans'] = flash_on_columns_df['alt_ff_cluster_last_flash_time_bsans'].replace(-999, np.nan)
    df = pd.concat([df, flash_on_columns_df], axis=1)

    num_if_alt_ff_cluster_flash_bsans = sum(flash_on_columns['if_alt_ff_cluster_flash_bsans'])
    print(f'Percentage of control rows that have alt ff cluster flashed on between stop time and next stop: \
            {round(num_if_alt_ff_cluster_flash_bsans/len(df)*100, 2)} % of {len(df)} rows.')

    return df



def _get_flash_on_columns(df, ff_flash_sorted, ff_real_position_sorted, ff_life_sorted,
                        duration_start_column = 'beginning_time',
                        duration_end_column = 'stop_time',
                        ):
    if 'alt_ff_x' not in df.columns:
        df['alt_ff_x'], df['alt_ff_y'] = ff_real_position_sorted[df['alt_ff_index'].values].T
    if 'alt_ff_cluster' not in df.columns:
        df['alt_ff_cluster'] = cluster_analysis.find_alive_ff_clusters(df[['alt_ff_x', 'alt_ff_y']].values, 
                                                                ff_real_position_sorted, 
                                                                df['beginning_time'].values,
                                                                df['next_stop_time'].values, 
                                                                ff_life_sorted, max_distance=50)

    flash_on_columns = {'if_alt_ff_flash_bbas': [],
                        'if_alt_ff_cluster_flash_bbas': [],
                        'alt_ff_last_flash_time_bbas': [],
                        'alt_ff_cluster_last_flash_time_bbas': []

    }
    for index, row in df.iterrows():
        ff_cluster = row['alt_ff_cluster']
        if_alt_ff_flash_bbas = False
        if_alt_ff_cluster_flash_bbas = False
        alt_ff_last_flash_time_bbas = -999
        alt_ff_cluster_last_flash_time_bbas = -999
        for ff_index in ff_cluster:
            ff_flash = ff_flash_sorted[ff_index]
            result = basic_func.find_intersection(ff_flash, [row[duration_start_column], row[duration_end_column]])
            if len(result) > 0:
                if_alt_ff_cluster_flash_bbas = True
                latest_flash_time_before_stop = min(ff_flash[result[-1]][-1], row[duration_end_column]) 
                alt_ff_cluster_last_flash_time_bbas = max(alt_ff_cluster_last_flash_time_bbas, latest_flash_time_before_stop)
                if ff_index == row['alt_ff_index']:
                    if_alt_ff_flash_bbas = True
                    alt_ff_last_flash_time_bbas = max(alt_ff_last_flash_time_bbas, latest_flash_time_before_stop)
        flash_on_columns['if_alt_ff_flash_bbas'].append(if_alt_ff_flash_bbas)
        flash_on_columns['if_alt_ff_cluster_flash_bbas'].append(if_alt_ff_cluster_flash_bbas)
        flash_on_columns['alt_ff_last_flash_time_bbas'].append(alt_ff_last_flash_time_bbas)
        flash_on_columns['alt_ff_cluster_last_flash_time_bbas'].append(alt_ff_cluster_last_flash_time_bbas)
    return flash_on_columns
