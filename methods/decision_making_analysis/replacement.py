
import sys
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis import trajectory_info
from null_behaviors import curvature_utils, curv_of_traj_utils
from decision_making_analysis import trajectory_info

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





def organize_replacement_data(replacement_df, prior_to_replacement_df, non_chosen_df, manual_anno, ff_dataframe, monkey_information, ff_real_position_sorted,
                              equal_sample_from_two_cases=True, add_current_curv_of_traj=False, ff_caught_T_new=None, 
                              curvature_df=None, curv_of_traj_df=None, add_old_values_to_diff_df=False,
                              ff_attributes=['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis'], 
                              add_arc_info = False,
                              arc_info_to_add=['optimal_curvature', 'curv_diff'],
                              non_chosen_ff_selection_criterion='abs_ff_angle_boundary'):

    # First deal with the chosen ff
    old_ff_info, new_ff_info, all_point_index, all_time = get_old_and_new_ff_info(replacement_df, prior_to_replacement_df, ff_dataframe, monkey_information, ff_real_position_sorted)
    old_ff_info = old_ff_info[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']]
    new_ff_info = new_ff_info[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']]
    changing_pursued_ff_data1 = combine_old_and_new_ff_info(new_ff_info, old_ff_info, all_point_index, monkey_information, ff_caught_T_new, add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, add_current_curv_of_traj=add_current_curv_of_traj, curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df)

    changing_pursued_ff_data1['time'] = all_time
    changing_pursued_ff_data1['point_index'] = all_point_index
    changing_pursued_ff_data1['whether_changed'] = 1

    # Then deal with the non-chosen ff
    non_chosen_ff_info, all_point_index, all_time = get_non_chosen_ff_info(non_chosen_df, selection_criterion=non_chosen_ff_selection_criterion)
    non_chosen_ff_info = non_chosen_ff_info[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']].copy()
    parallel_old_ff_info, position_indices_with_valid_ff = get_parallel_old_ff_info(all_point_index, all_time, manual_anno, ff_real_position_sorted, ff_dataframe, monkey_information)
    parallel_old_ff_info = parallel_old_ff_info[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']].copy()
    # update the variables based on valid ff_index
    all_point_index = all_point_index[position_indices_with_valid_ff]
    all_time = all_time[position_indices_with_valid_ff]
    non_chosen_ff_info = non_chosen_ff_info.iloc[position_indices_with_valid_ff].copy()


    changing_pursued_ff_data2 = combine_old_and_new_ff_info(non_chosen_ff_info, parallel_old_ff_info, all_point_index, monkey_information, ff_caught_T_new, add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, add_current_curv_of_traj=add_current_curv_of_traj, curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df)
    changing_pursued_ff_data2['time'] = all_time
    changing_pursued_ff_data2['point_index'] = all_point_index
    changing_pursued_ff_data2['whether_changed'] = 0


    if equal_sample_from_two_cases:
        # to avoid imbalance of labels in the sample, we will only include the same number of cases from each category
        sample_size = min(len(changing_pursued_ff_data1), len(changing_pursued_ff_data2))
        changing_pursued_ff_data1 = changing_pursued_ff_data1.sample(n=sample_size, random_state=0)
        changing_pursued_ff_data2 = changing_pursued_ff_data2.sample(n=sample_size, random_state=0)
    

    # Combine both
    changing_pursued_ff_data = pd.concat([changing_pursued_ff_data1, changing_pursued_ff_data2], axis=0).reset_index(drop=True)
    changing_pursued_ff_data_diff = find_changing_pursued_ff_data_diff(changing_pursued_ff_data, add_old_values_to_diff_df=add_old_values_to_diff_df, add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, add_current_curv_of_traj=add_current_curv_of_traj, ff_attributes=ff_attributes)
    replacement_inputs_additional_info = changing_pursued_ff_data[['time', 'point_index']].copy()
    changing_pursued_ff_data.drop(columns=['time', 'point_index'], inplace=True)
    replacement_inputs_for_plotting = changing_pursued_ff_data[['old_ff_distance', 'old_ff_angle_boundary', 'old_time_since_last_vis', 'ff_distance', 'ff_angle_boundary', 'time_since_last_vis']].values

    # drop extra columns in changing_pursued_ff_data:
    columns_to_drop = [name for name in ['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis'] if name not in ff_attributes]
    if add_old_values_to_diff_df:
        columns_to_drop.extend(['old_'+name for name in ['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis'] if name not in ff_attributes])
        changing_pursued_ff_data.drop(columns=columns_to_drop, inplace=True)

    return changing_pursued_ff_data, changing_pursued_ff_data_diff, replacement_inputs_additional_info, replacement_inputs_for_plotting




def add_arc_info_to_old_and_new_ff_info(old_ff_info, new_ff_info, curvature_df, monkey_information, ff_caught_T_new, curv_of_traj_df, arc_info_to_add=['optimal_curvature', 'curv_diff']):
    old_temp_df = old_ff_info[['ff_index', 'point_index', 'ff_distance', 'ff_angle_boundary']].copy()
    new_temp_df = new_ff_info[['ff_index', 'point_index', 'ff_distance', 'ff_angle_boundary']].copy()
    both_temp_df = {'old': old_temp_df, 'new': new_temp_df}
    curvature_df_sub = curvature_df[['ff_index', 'point_index'] + arc_info_to_add].copy()
    if curv_of_traj_df is None:
        curv_of_traj_df = curvature_df.copy()
    for label in ['old', 'new']:
        temp_df = both_temp_df[label]
        arc_info_df = pd.merge(curvature_df_sub, temp_df, on=['ff_index', 'point_index'], how='right')
        # for the NAs (when ff_index is -10)
        curvature_utils.fill_up_NAs_in_columns_related_to_curvature(arc_info_df, monkey_information, ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
        # combine arc_info_df and result_df
        column_names = arc_info_to_add
        if label == 'old':
            old_ff_info[['old_' + name for name in column_names]] = arc_info_df[column_names].copy().values
        else:
            new_ff_info[column_names] = arc_info_df[column_names].copy().values




def get_old_and_new_ff_info(replacement_df, prior_to_replacement_df, ff_dataframe, monkey_information, ff_real_position_sorted):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1].copy()
    # Take out the scenarios where the monkey changes its target of pursuit
    replacement_df_sub = replacement_df[['ff_index', 'starting_point_index', 'time']]
    replacement_df_sub['prior_ff_index'] = prior_to_replacement_df['ff_index'].values
    replacement_df_sub = replacement_df_sub[replacement_df_sub['ff_index'] != replacement_df_sub['prior_ff_index']]
    new_ff_info = decision_making_utils.find_many_ff_info_anew(replacement_df_sub['ff_index'].values, replacement_df_sub['starting_point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']]
    old_ff_info = decision_making_utils.find_many_ff_info_anew(replacement_df_sub['prior_ff_index'].values, replacement_df_sub['starting_point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)[['ff_distance', 'ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']]
    all_point_index = replacement_df_sub['starting_point_index'].values
    all_time = replacement_df_sub['time'].values
    return old_ff_info, new_ff_info, all_point_index, all_time



def get_non_chosen_ff_info(non_chosen_df, selection_criterion='abs_ff_angle_boundary', duration_for_each_group=0.35, one_row_per_group=False):
    # If one_row_per_group is False, then we will choose one row for each ff_index for each group(duration), and the row is the one with the smallest selection_criterion for that ff in that group(duration)
    
    # in case the following attributes are not included in non_chosen_df
    non_chosen_df['abs_ff_angle'] = non_chosen_df['ff_angle'].abs()
    non_chosen_df['abs_ff_angle_boundary'] = non_chosen_df['ff_angle_boundary'].abs()
    # Take out the scenaiors where a new ff is visible but the monkey doesn't change its target of pursuit
    # Now we will sample some ff from non_chosen_df.
    # we don't want to use the same alternative ff twice within a short interval (0.35s here), so we divide the rows into groups of 0.35s intervals
    non_chosen_df['group'] = non_chosen_df['time'].apply(lambda x: int(x/duration_for_each_group))
    # we will sample one row from each group, which has the smallest absolute value of ff_angle_boundary, and among that find the row with the shortest distance
    if selection_criterion != 'time_since_last_vis':
        columns_to_sort_by = [selection_criterion, 'time_since_last_vis']
    else:
        columns_to_sort_by = [selection_criterion, 'ff_angle_boundary']
    non_chosen_df = non_chosen_df.sort_values(columns_to_sort_by, ascending=[True, True])
    if one_row_per_group is False:
        # The following line is to make sure that we don't choose the same ff twice within a short interval
        # In order to do that, for a given interval (group), we only choose one row for the ff based on the selection criterion
        non_chosen_df = non_chosen_df.groupby(['group', 'ff_index']).first().reset_index()
    else:
        # Another way is to choose only one row for each group based on the selection criterion
        non_chosen_df = non_chosen_df.groupby(['group']).first().reset_index()

    non_chosen_ff_info = non_chosen_df[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'time_since_last_vis', 'point_index', 'ff_index']].copy()
    all_point_index = non_chosen_df['point_index'].values
    all_time = non_chosen_df['time'].values

    return non_chosen_ff_info, all_point_index, all_time



def get_parallel_old_ff_info(all_point_index, all_time, manual_anno, ff_real_position_sorted, ff_dataframe, monkey_information):
    manual_anno = manual_anno.sort_values('time')
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1].copy()
    parallel_old_ff_rows_indices = np.searchsorted(manual_anno['time'].values, all_time, side='right')-1
    parallel_old_ff_rows = manual_anno.iloc[parallel_old_ff_rows_indices].copy()
    parallel_old_ff_rows['point_index'] = all_point_index
    parallel_old_ff_rows.reset_index(drop=True, inplace=True)
    rows_with_valid_ff = parallel_old_ff_rows[parallel_old_ff_rows['ff_index'] >= 0].copy()
    position_indices_with_valid_ff = rows_with_valid_ff.index.values
    parallel_old_ff_info = decision_making_utils.find_many_ff_info_anew(rows_with_valid_ff['ff_index'].values, rows_with_valid_ff['point_index'].values, ff_real_position_sorted, ff_dataframe_visible, monkey_information)
    return parallel_old_ff_info, position_indices_with_valid_ff



def combine_old_and_new_ff_info(new_ff_info, old_ff_info, point_index_array, monkey_information, ff_caught_T_new, 
                                window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False, 
                                add_arc_info=False, arc_info_to_add=['optimal_curvature', 'curv_diff'],
                                add_current_curv_of_traj=False, curvature_df=None, curv_of_traj_df=None):
    
    if curv_of_traj_df is None:
        curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, monkey_information, ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)

    if add_arc_info:
        add_arc_info_to_old_and_new_ff_info(old_ff_info, new_ff_info, curvature_df, monkey_information, ff_caught_T_new, curv_of_traj_df, arc_info_to_add=arc_info_to_add)

    new_ff_info = new_ff_info.drop(columns=['point_index', 'ff_index'])
    old_ff_info = old_ff_info.drop(columns=['point_index', 'ff_index'])
    old_ff_info.rename(columns={'ff_distance':'old_ff_distance', 'ff_angle':'old_ff_angle', 'ff_angle_boundary':'old_ff_angle_boundary', 'time_since_last_vis':'old_time_since_last_vis'}, inplace=True)
    changing_pursued_ff_data = pd.concat([old_ff_info.reset_index(drop=True), new_ff_info.reset_index(drop=True)], axis=1)

    if add_current_curv_of_traj:
        changing_pursued_ff_data['curv_of_traj'] = trajectory_info.find_trajectory_arc_info(point_index_array, curvature_df, ff_caught_T_new=ff_caught_T_new, monkey_information=monkey_information,
                                                                                                 curv_of_traj_mode=curv_of_traj_mode, window_for_curv_of_traj=window_for_curv_of_traj, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
    return changing_pursued_ff_data



def find_changing_pursued_ff_data_diff(changing_pursued_ff_data, add_old_values_to_diff_df=False, add_arc_info=False, arc_info_to_add=['optimal_curvature', 'curv_diff'],
                                       add_current_curv_of_traj=False, ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis']):

    changing_pursued_ff_data_diff = changing_pursued_ff_data[['whether_changed']].copy()
    old_ff_columns = ['old_'+ attribute for attribute in ff_attributes]
    if add_old_values_to_diff_df:
        changing_pursued_ff_data_diff[old_ff_columns] = changing_pursued_ff_data[old_ff_columns].copy()
    for attribute in ff_attributes:
        changing_pursued_ff_data_diff[attribute+'_diff'] = changing_pursued_ff_data[attribute] - changing_pursued_ff_data['old_'+attribute]


    if add_arc_info:
        for attr in arc_info_to_add:
            changing_pursued_ff_data_diff[attr+'_diff'] = changing_pursued_ff_data[attr] - changing_pursued_ff_data['old_'+attr]
        if add_old_values_to_diff_df:
            changing_pursued_ff_data_diff[['old_'+attr for attr in arc_info_to_add]] = changing_pursued_ff_data[['old_'+attr for attr in arc_info_to_add]].copy()


    if not add_old_values_to_diff_df:
        for column in ['ff_angle_diff', 'ff_angle_boundary_diff', 'curvature_lower_bound_diff', 'curvature_upper_bound_diff', 'optimal_curvature_diff']:
            if column in changing_pursued_ff_data_diff.columns:
                changing_pursued_ff_data_diff.drop(columns=[column], inplace=True)
                changing_pursued_ff_data_diff[column+'_in_abs'] = abs(changing_pursued_ff_data[column.replace('_diff', '')]) - abs(changing_pursued_ff_data['old_'+column.replace('_diff', '')])

    if add_current_curv_of_traj:
        changing_pursued_ff_data_diff['curv_of_traj'] = changing_pursued_ff_data['curv_of_traj'] 

    return changing_pursued_ff_data_diff

