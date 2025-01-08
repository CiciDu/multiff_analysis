import sys
from data_wrangling import specific_utils
from null_behaviors import show_null_trajectory

import os
import numpy as np
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import math
from scipy import stats
from scipy.stats import rankdata


def make_best_arc_df(curvature_df, monkey_information, ff_real_position_sorted):
    # for each point, among null arcs to all possible ff targets, find the one with the smallest curvature difference

    curvature_df_sorted = curvature_df.sort_values(by=['point_index', 'abs_curv_diff', 'ff_distance'], ascending=[True, True, True]).copy()

    tentative_best_arc_df = curvature_df_sorted.groupby(['point_index']).first().reset_index()
    #tentative_best_arc_df = smooth_out_ff_index(tentative_best_arc_df, curvature_df)
    tentative_best_arc_df = tentative_best_arc_df[['point_index', 'ff_index']]
    best_arc_df = pd.merge(tentative_best_arc_df, curvature_df_sorted, how='left', on=['point_index', 'ff_index'])


    ## Note that if two arcs are the same in abs_curv_diff, then we rank them by ff_distance
    # best_arc_df = curvature_df_sorted.groupby(['point_index']).first().reset_index()

    # best_arc_df = best_arc_df.rename(columns = {'optimal_arc_radius': 'optimal_arc_radius',
    #                                             'optimal_arc_measure': 'optimal_arc_measure',
    #                                             'optimal_arc_length': 'optimal_arc_length',
    #                                             'optimal_arc_end_direction': 'optimal_arc_end_direction'})
    
    best_arc_original_columns = best_arc_df.copy()

    best_arc_df = best_arc_df[['point_index', 'ff_index', 'ff_distance', 'ff_angle', 'curv_diff', 'abs_curv_diff',
                               'optimal_arc_radius', 'optimal_arc_measure', 'optimal_arc_length', 'optimal_arc_end_direction',
                               'optimal_arc_end_x', 'optimal_arc_end_y']]
    best_arc_df['whether_ff_behind'] = False
    best_arc_df.loc[np.abs(best_arc_df['ff_angle']) > math.pi/2, 'whether_ff_behind'] = True
    best_arc_df = furnish_best_arc_df(best_arc_df, monkey_information, ff_real_position_sorted)     

    # add a column of diff_percentile
    v = best_arc_df['abs_curv_diff'].values.copy()
    best_arc_df['diff_percentile'] = rankdata(v)*100/len(v)
    best_arc_df['diff_percentile_in_decimal'] = best_arc_df['diff_percentile'] / 100

    return best_arc_df, best_arc_original_columns


def smooth_out_ff_index(best_arc_df, curvature_df, tolerable_difference = 10/100*pi/180):
    # the default value of tolerable_difference means after traversing 100 cm, how much change will there be in monkey_angle in radians
    curvature_df_sub = curvature_df[['point_index', 'ff_index']]
    list_of_ff_index = [best_arc_df['ff_index'][0]]
    prev_ff_index = list_of_ff_index[0]
    prev_row = best_arc_df.iloc[0]
    for index, row in best_arc_df.iloc[1:].iterrows():
        if row.ff_index == prev_ff_index:
            list_of_ff_index.append(prev_ff_index)
        else: # if the new ff_index is different from the previous one
            if (prev_row.ff_distance < row.ff_distance) and (prev_row.abs_curv_diff - row.abs_curv_diff <= tolerable_difference):
                # if the criterion is met
                # check to see if prev_ff_index also exists in the current point
                df_sub = curvature_df_sub[(curvature_df_sub['point_index']==row.point_index) & (curvature_df_sub['ff_index']==prev_ff_index)]
                if len(df_sub) > 0:
                    # we preserve the previous ff_index
                    list_of_ff_index.append(prev_ff_index)
                else: # we take the new ff_index
                    prev_ff_index = row.ff_index
                    list_of_ff_index.append(row.ff_index)
            else: # we take the new ff_index
                prev_ff_index = row.ff_index
                list_of_ff_index.append(row.ff_index)
        prev_row = row.copy()  
    new_list_of_ff_index = np.array(list_of_ff_index).astype(int) 
    best_arc_df.loc[:,'ff_index'] = new_list_of_ff_index
    return best_arc_df



def furnish_best_arc_df(best_arc_df, monkey_information, ff_real_position_sorted, time_gap_to_differentiate_intended_target_id=2.5):
    best_arc_df = best_arc_df.sort_values(by='point_index')
    
    # add: monkey_xy, ff_xy, arc center xy, arc radius, optimal_arc_length, ff_angle (which determines left/right)
    best_arc_df['monkey_x'] = monkey_information.loc[best_arc_df['point_index'].values, 'monkey_x'].values
    best_arc_df['monkey_y'] = monkey_information.loc[best_arc_df['point_index'].values, 'monkey_y'].values
    best_arc_df['ff_x'] = ff_real_position_sorted[best_arc_df['ff_index'].values, 0]
    best_arc_df['ff_y'] = ff_real_position_sorted[best_arc_df['ff_index'].values, 1]


    best_arc_df['whether_new_ff'] = (best_arc_df['ff_index'] != best_arc_df['ff_index'].shift()).astype(int)
    # get the opposite of whether_new_ff, which is whether_continued
    best_arc_df['whether_continued'] = (best_arc_df['ff_index'] == best_arc_df['ff_index'].shift()).astype(int)
    best_arc_df['whether_new_ff_cum_sum'] = best_arc_df['whether_new_ff'].cumsum()
    
    best_arc_df['num_repetitions'] = best_arc_df[['whether_continued', 'whether_new_ff_cum_sum']].groupby('whether_new_ff_cum_sum').cumsum()
    # make a new df to find max_num_repetitions for each chunk
    repetitions_df = best_arc_df[['num_repetitions', 'whether_new_ff_cum_sum']].groupby('whether_new_ff_cum_sum').max().reset_index(drop=False)
    if 'max_num_repetitions' in repetitions_df.columns:
        repetitions_df.drop(['max_num_repetitions'], axis=1, inplace=True) # to avoid having two identical columns after using rename
    if 'max_num_repetitions' in best_arc_df.columns:
        best_arc_df.drop(['max_num_repetitions'], axis=1, inplace=True) # to avoid having two identical columns after using rename
        
    repetitions_df.rename(columns={'num_repetitions': 'max_num_repetitions'}, inplace=True)
    best_arc_df = best_arc_df.merge(repetitions_df, on='whether_new_ff_cum_sum', how='left')
    
    # have a new column: intended_target_id by copying the values from chunk_id
    best_arc_df['time'] = monkey_information.loc[best_arc_df['point_index'], 'time'].values
    best_arc_df['chunk_id'] = best_arc_df['whether_new_ff_cum_sum'] - 1

    '''
    # to find intended_target_id
    best_arc_df['intended_target_id'] = best_arc_df['chunk_id'].copy()
    # for each ff_index
    best_arc_df = best_arc_df.sort_values(by='point_index')
    unique_id_counter = 0
    for ff_index in best_arc_df.ff_index.unique():
    # take out all the rows that belong to it
        same_ff_rows = best_arc_df[best_arc_df.ff_index==ff_index]
        unique_intended_target_id = same_ff_rows.intended_target_id.unique()
        # separate them into chunks. If any two has time part greater than 5s, then begin a new chunk
        # It's similar to the concept of sticks and stones in combinatorics
        all_time = same_ff_rows.time.values
        diff_in_time = np.diff(all_time)
        # add a 0 in the beginning of diff_in_time
        diff_in_time = np.append(0, diff_in_time)
        where_gaps = np.where(diff_in_time > time_gap_to_differentiate_intended_target_id)[0]
        # again add a 0 in the beginning so that we can use np.diff later
        # see the number of repeats for each intended_target_id
        where_gaps = np.append(0, where_gaps)
        num_repeats_for_each_intended_target_id = np.diff(where_gaps)
        # if there are more rows left for the last gap
        if len(where_gaps) > 0:
            if where_gaps[-1] < len(diff_in_time):
                num_repeats_for_each_intended_target_id = np.append(num_repeats_for_each_intended_target_id, len(diff_in_time)-where_gaps[-1])
        else:
            num_repeats_for_each_intended_target_id = np.array([len(diff_in_time)])
        # we take the needed number of target_id to assign to the new target chunks (we'll re-organize all the id at the end)
        num_unique_id_needed = len(num_repeats_for_each_intended_target_id)
        unique_intended_target_id = np.arange(unique_id_counter, unique_id_counter+num_unique_id_needed)
        unique_id_counter += num_unique_id_needed
        new_intended_target_id = np.repeat(unique_intended_target_id, num_repeats_for_each_intended_target_id)
        best_arc_df.loc[best_arc_df['ff_index']==ff_index, 'intended_target_id'] = new_intended_target_id

    # now, re-organize all the intended_target_id so that they can at least be continuous
    best_arc_df = best_arc_df.sort_values(by='point_index').copy()
    used_unique_intended_target_id = best_arc_df['intended_target_id'].unique()
    new_unique_intended_target_id = range(len(used_unique_intended_target_id))
    chart_for_conversion = np.zeros(max(used_unique_intended_target_id)+1).astype(int)
    chart_for_conversion[used_unique_intended_target_id] = new_unique_intended_target_id
    best_arc_df['intended_target_id'] = chart_for_conversion[best_arc_df['intended_target_id'].values]
    '''

    ''' 
    # find the last time for each chunk_id
    last_time_df = best_arc_df.groupby('chunk_id').last()

    # for each chunk_id
    for index in last_time_df.index:
        intended_target_row = last_time_df.loc[index]
        if (index > 0) & (index % 5000 == 0):
            print('Progress of finding intended_target_id to furnish best_arc_df:', index, 'out of', last_time_df.index.max())
        # go to the last row associated with it
        time = intended_target_row.time
        ff_index = intended_target_row.ff_index
        intended_target_id = intended_target_row['intended_target_id'].astype(int)
        # and project to future 5s
        duration = [time, time+5]
        # if another chunk_id has the same ff_index, then replace the second intended_target_id with the first intended_target_id
        best_arc_df.loc[(best_arc_df['time'].between(duration[0], duration[1])) & (best_arc_df['ff_index'] == ff_index), 'intended_target_id'] = intended_target_id
        last_time_df.loc[(last_time_df['time'].between(duration[0], duration[1])) & (last_time_df['ff_index'] == ff_index), 'intended_target_id'] = intended_target_id    
    '''
        
    best_arc_df.drop(columns=['whether_new_ff', 'whether_continued', 'whether_new_ff_cum_sum'], inplace=True)
    return best_arc_df


def add_column_monkey_passed_by_to_best_arc_df(best_arc_df, ff_dataframe):
    pass_by_within_next_n_seconds = 2.5
    pass_by_within_n_cm = 50

    best_arc_df['monkey_passed_by'] = False
    for id in np.unique(best_arc_df.intended_target_id.values):
        if id%500 == 0:
            print("id", id, "out of", max(best_arc_df.intended_target_id.values))
        best_arc_df_sub = best_arc_df[best_arc_df.intended_target_id == id]
        time = best_arc_df_sub.time.values
        ff_index = np.unique(best_arc_df_sub.ff_index.values) # ff_index should be the same for all rows
        if len(ff_index) != 1:
            raise ValueError("ff_index should be the same for all rows")
        ff_index = ff_index[0]
        ff_dataframe_sub = ff_dataframe[(ff_dataframe['ff_index'] == ff_index) & (ff_dataframe.ff_distance <= pass_by_within_n_cm)]
        ff_dataframe_sub = ff_dataframe_sub[(ff_dataframe_sub.time >= min(time)) & (ff_dataframe_sub.time <= max(time)+pass_by_within_next_n_seconds)]
        if len(ff_dataframe_sub) > 0:
            #print("ff_index", ff_index, "has been stopped by monkey within", pass_by_within_next_n_seconds, "seconds")
            best_arc_df.loc[best_arc_df['intended_target_id']==id, 'monkey_passed_by'] = True


def find_point_on_ff_boundary_with_smallest_angle_to_monkey(ff_x, ff_y, monkey_x, monkey_y, monkey_angle, ff_radius=10):
    angles_to_ff = specific_utils.calculate_angles_to_ff_centers(ff_x, ff_y, monkey_x, monkey_y, monkey_angle)
    diff_x = ff_x - monkey_x
    diff_y = ff_y - monkey_y
    diff_xy = np.stack((diff_x, diff_y), axis=1)
    distances_to_ff = LA.norm(diff_xy, axis=1)
    angles_to_boundaries = specific_utils.calculate_angles_to_ff_boundaries(angles_to_ff, distances_to_ff, ff_radius=ff_radius)
    dif_in_angles = angles_to_ff - angles_to_boundaries
    new_ff_distance = np.abs(np.cos(dif_in_angles)*distances_to_ff)
    new_ff_angle_in_world = angles_to_boundaries + monkey_angle
    ff_x = np.cos(new_ff_angle_in_world)*new_ff_distance + monkey_x
    ff_y = np.sin(new_ff_angle_in_world)*new_ff_distance + monkey_y
    ff_xy = np.stack((ff_x, ff_y), axis=1)
    return ff_xy


def combine_manual_anno_and_best_arc_df_info_for_comparison(best_arc_df, chosen_rows_of_df, sequence_of_obs_ff_indices):
    # Note that we didn't use all manual_anno info, but instead only used the info of the chosen rows, which might depend on factors such as keeping_1_out_of_n_rows

    # incorporate data from best_arc_df into chosen_rows so we can compare the two
    chosen_rows = chosen_rows_of_df[['starting_point_index', 'ff_index']].copy()
    chosen_rows.rename(columns={'starting_point_index': 'point_index', 'ff_index': 'anno_ff_index'}, inplace=True)
    best_arc_df['best_arc_row_id'] = best_arc_df.index
    best_arc_df_sub = best_arc_df[['point_index', 'ff_index', 'best_arc_row_id']].copy()
    best_arc_df_sub.rename(columns={'ff_index': 'best_arc_ff_index'}, inplace=True)

    chosen_rows_merged = pd.merge(chosen_rows, best_arc_df_sub, on='point_index', how='left').fillna(-5)
    chosen_rows_merged['best_arc_ff_index'] = chosen_rows_merged['best_arc_ff_index'].astype(int)
    chosen_rows_merged['best_arc_row_id'] = chosen_rows_merged['best_arc_row_id'].astype(int)
    print('Note: all the NA values in best_arc_ff_index and best_arc_row_id are replaced with -5.')


    sequence_of_obs_ff_indices = np.array(sequence_of_obs_ff_indices)
    # add a column of -5 which means no ff from the original obs_ff_indices is chosen
    sequence_of_obs_ff_indices = np.concatenate([sequence_of_obs_ff_indices, np.repeat(-5, sequence_of_obs_ff_indices.shape[0]).reshape(-1, 1)], axis=1)
    best_arc_pred = np.argmin(np.abs(sequence_of_obs_ff_indices.T - chosen_rows_merged.best_arc_ff_index.values), axis=0)
    # but if no ff in obs_ff_indices can match the best_arc_ff_index, then we shall replace the element in best_arc_pred with the maximum label (which means no ff was chosen)
    no_match_rows = np.where(np.min(np.abs(sequence_of_obs_ff_indices.T - chosen_rows_merged.best_arc_ff_index.values), axis=0) > 0)[0]
    if len(no_match_rows) > 0:
        print('Warning: there are', len(no_match_rows), 'out of', len(sequence_of_obs_ff_indices), 'rows in chosen_rows_merged that do not match any ff in sequence_of_obs_ff_indices. The row indices are stored in no_match_rows')
        best_arc_pred[no_match_rows] = sequence_of_obs_ff_indices.shape[1] - 1
    

    chosen_rows_merged.loc[chosen_rows_merged.anno_ff_index < 0, 'anno_ff_index'] = -99
    chosen_rows_merged.loc[chosen_rows_merged.best_arc_ff_index < 0, 'best_arc_ff_index'] = -99
    print('Note, all the negative values in anno_ff_index and best_arc_ff_index are replaced with -99.')

    mismatched_rows = chosen_rows_merged[chosen_rows_merged['best_arc_ff_index'] != chosen_rows_merged['anno_ff_index']]
    mismatched_indices = mismatched_rows.index.values

    return chosen_rows_merged, best_arc_pred, mismatched_rows, mismatched_indices, no_match_rows


