

from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis import trajectory_info
from null_behaviors import curvature_utils, curv_of_traj_utils


import os
import math
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def find_info_of_n_ff_per_point(free_selection_df, ff_dataframe, ff_real_position_sorted, monkey_information, ff_caught_T_new, num_ff_per_row=3, guarantee_including_target_info=True, only_select_n_ff_case=None, selection_criterion_if_too_many_ff='abs_ff_angle',
                                placeholder_ff_index=-10, placeholder_ff_distance=400, placeholder_ff_angle=0, placeholder_ff_angle_boundary=0, placeholder_time_since_last_vis=3, curv_of_traj_df=None,
                                add_arc_info=False, curvature_df=None, arc_info_to_add=['opt_arc_curv', 'curv_diff']):
    if only_select_n_ff_case is not None:
        # if only_select_n_ff_case is not an integer
        if isinstance(only_select_n_ff_case, int) is False:
            raise ValueError(
                'only_select_n_ff_case needs to be either None or an integer')
        if only_select_n_ff_case != num_ff_per_row:
            num_ff_per_row = only_select_n_ff_case
            print("only_select_n_ff_case is not None, so num_ff_per_row is set to only_select_n_ff_case: ", num_ff_per_row)

    ff_dataframe_sub = ff_dataframe[ff_dataframe['point_index'].isin(
        free_selection_df.starting_point_index.values.astype(int))].copy()
    ff_dataframe_sub = ff_dataframe_sub[['point_index', 'ff_index', 'ff_distance', 'ff_angle',
                                         'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary', 'time_since_last_vis']].copy()
    ff_dataframe_sub.loc[:, 'selection_criterion'] = ff_dataframe_sub.loc[:,
                                                                          selection_criterion_if_too_many_ff]

    if guarantee_including_target_info:
        free_selection_df_temp = free_selection_df[[
            'starting_point_index', 'ff_index']].copy()
        free_selection_df_temp.rename(
            columns={'starting_point_index': 'point_index'}, inplace=True)
        free_selection_df_temp['selection_criterion'] = min(
            -9999, ff_dataframe_sub[selection_criterion_if_too_many_ff].min()-1000)
        ff_dataframe_sub = guarantee_ff_dataframe_include_target_info(
            free_selection_df_temp, ff_dataframe_sub, ff_real_position_sorted, ff_dataframe, monkey_information)

    # if 'curv_of_traj' not in ff_dataframe_sub.columns:
    #     if curv_of_traj_df is None:
    #         raise ValueError('curv_of_traj_df is None, but add_current_curv_of_traj is True')
    #     ff_dataframe_sub = ff_dataframe_sub.merge(curv_of_traj_df[['point_index', 'curv_of_traj']], on='point_index', how='left')

    if add_arc_info:
        ff_dataframe_sub = curvature_utils.add_arc_info_to_df(
            ff_dataframe_sub, curvature_df, arc_info_to_add=arc_info_to_add, ff_caught_T_new=ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)

    all_point_index = np.unique(
        free_selection_df.starting_point_index.values.astype(int))
    ff_dataframe_sub = guarantee_n_ff_per_point_index_in_ff_dataframe(ff_dataframe_sub, all_point_index, num_ff_per_row=num_ff_per_row, only_select_n_ff_case=only_select_n_ff_case, selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff,
                                                                      placeholder_ff_index=placeholder_ff_index, placeholder_ff_distance=placeholder_ff_distance, placeholder_ff_angle=placeholder_ff_angle, placeholder_ff_angle_boundary=placeholder_ff_angle_boundary,
                                                                      placeholder_time_since_last_vis=placeholder_time_since_last_vis, curv_of_traj_df=curv_of_traj_df)

    # then, we sort the remaining ff by ff_angle
    info_of_n_ff_per_point = ff_dataframe_sub.sort_values(
        ['point_index', 'ff_angle'], ascending=True).drop(columns=['selection_criterion'])
    # After sorting the ff for each point by ff_angle, assign an order number for each ff for each point_index
    info_of_n_ff_per_point['order'] = np.tile(
        range(num_ff_per_row), int(len(info_of_n_ff_per_point)/num_ff_per_row))
    info_of_n_ff_per_point.reset_index(drop=True, inplace=True)

    return info_of_n_ff_per_point


def find_label_of_ff_in_obs_ff(sequence_of_obs_ff_indices, ff_index_to_label, na_filler=0):
    # find labels
    matched_rows = np.where(sequence_of_obs_ff_indices -
                            ff_index_to_label.reshape(-1, 1) == 0)
    labels_df = pd.DataFrame(
        {'row': matched_rows[0], 'label': matched_rows[1]})
    full_row_df = pd.DataFrame({'row': range(len(sequence_of_obs_ff_indices))})
    merged_df = pd.merge(full_row_df, labels_df, on='row', how='left')
    # for the rows with NA, the intended target was not discerned, so let's use a number to signal this new category
    merged_df['label'] = merged_df['label'].fillna(na_filler)
    labels = merged_df['label'].values.astype(int)
    return labels


def find_free_selection_x_from_info_of_n_ff_per_point(
    info_of_n_ff_per_point,
    monkey_information=None,
    ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'],
    attributes_for_plotting=['ff_distance', 'ff_angle', 'time_since_last_vis'],
    add_current_curv_of_traj=False,
    num_ff_per_row=5,
    ff_caught_T_new=None,
    curv_of_traj_df=None,
    window_for_curv_of_traj=[-25, 25],
    curv_of_traj_mode='distance',
    truncate_curv_of_traj_by_time_of_capture=False
):
    info_of_n_ff_per_point = info_of_n_ff_per_point.sort_values(
        ['point_index', 'order'], ascending=True)
    point_index_array = info_of_n_ff_per_point.loc[info_of_n_ff_per_point['order']
                                                   == 0, 'point_index'].values

    free_selection_x_df = pd.DataFrame()
    free_selection_x_df_for_plotting = pd.DataFrame()
    pred_var = []
    sequence_of_obs_ff_indices = []

    for i in range(num_ff_per_row):
        current_info = info_of_n_ff_per_point[info_of_n_ff_per_point['order'] == i]
        if not np.array_equal(current_info['point_index'].values, point_index_array):
            raise ValueError(
                f'point_index_array is not in the right order for order number: {i}')

        info_to_add = current_info[ff_attributes].copy()
        column_names = [f"{attr}_{i}" for attr in ff_attributes]
        column_for_plotting_names = [
            f"{attr}_for_plotting_{i}" for attr in attributes_for_plotting]

        pred_var.extend(column_names)
        free_selection_x_df[column_names] = info_to_add.values
        free_selection_x_df_for_plotting[column_for_plotting_names] = current_info[attributes_for_plotting].values
        sequence_of_obs_ff_indices.append(current_info['ff_index'].values)
    free_selection_x_df['point_index'] = point_index_array
    #free_selection_x_df_for_plotting['point_index'] = point_index_array

    if add_current_curv_of_traj:
        if monkey_information is None:
            raise ValueError(
                'monkey_information is None, but add_current_curv_of_traj is True')

        if curv_of_traj_df is None:
            curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
                window_for_curv_of_traj, monkey_information, ff_caught_T_new,
                curv_of_traj_mode=curv_of_traj_mode,
                truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture
            )

        curv_of_traj = trajectory_info.find_trajectory_arc_info(
            point_index_array, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new,
            monkey_information=monkey_information, curv_of_traj_mode=curv_of_traj_mode,
            window_for_curv_of_traj=window_for_curv_of_traj
        )

        free_selection_x_df['curv_of_traj'] = curv_of_traj
        pred_var.append('curv_of_traj')

    free_selection_x_df.reset_index(drop=True, inplace=True)
    sequence_of_obs_ff_indices = np.array(sequence_of_obs_ff_indices).T

    return free_selection_x_df, free_selection_x_df_for_plotting, sequence_of_obs_ff_indices, point_index_array, pred_var


def organize_free_selection_x(
    free_selection_df,
    ff_dataframe,
    ff_real_position_sorted,
    monkey_information,
    only_select_n_ff_case=None,
    num_ff_per_row=5,
    guarantee_including_target_info=True,
    add_current_curv_of_traj=False,
    ff_caught_T_new=None,
    window_for_curv_of_traj=[-25, 0],
    curv_of_traj_mode='distance',
    curvature_df=None,
    curv_of_traj_df=None,
    selection_criterion_if_too_many_ff='time_since_last_vis',
    ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'],
    add_arc_info=False,
    arc_info_to_add=['opt_arc_curv', 'curv_diff'],
    info_of_n_ff_per_point=None
):
    if not add_arc_info:
        arc_info_to_add = []

    if info_of_n_ff_per_point is None:
        free_selection_df = free_selection_df.sort_values(
            ['starting_point_index'], ascending=True).reset_index(drop=True)
        info_of_n_ff_per_point = find_info_of_n_ff_per_point(
            free_selection_df,
            ff_dataframe,
            ff_real_position_sorted,
            monkey_information,
            ff_caught_T_new,
            num_ff_per_row=num_ff_per_row,
            guarantee_including_target_info=guarantee_including_target_info,
            only_select_n_ff_case=only_select_n_ff_case,
            selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff,
            add_arc_info=add_arc_info,
            arc_info_to_add=arc_info_to_add,
            curvature_df=curvature_df,
            curv_of_traj_df=curv_of_traj_df
        )
        ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))
    else:
        print(
            'Note: info_of_n_ff_per_point is not None, so some parameters will be ignored.')

    free_selection_x_df, free_selection_x_df_for_plotting, sequence_of_obs_ff_indices, point_index_array, pred_var = find_free_selection_x_from_info_of_n_ff_per_point(
        info_of_n_ff_per_point,
        monkey_information,
        ff_attributes=ff_attributes,
        num_ff_per_row=num_ff_per_row,
        add_current_curv_of_traj=add_current_curv_of_traj,
        ff_caught_T_new=ff_caught_T_new,
        curv_of_traj_df=curv_of_traj_df,
        curv_of_traj_mode=curv_of_traj_mode,
        window_for_curv_of_traj=window_for_curv_of_traj
    )

    free_selection_labels = find_label_of_ff_in_obs_ff(
        sequence_of_obs_ff_indices,
        free_selection_df['ff_index'].values,
        na_filler=num_ff_per_row
    )

    rows_with_mismatched_labels = np.where(
        free_selection_labels == num_ff_per_row)[0]
    point_index_with_mismatched_labels = point_index_array[rows_with_mismatched_labels]

    non_chosen_rows_of_df = free_selection_df[free_selection_df['starting_point_index'].isin(
        point_index_with_mismatched_labels)].copy()
    cases_for_inspection = {
        'cases_for_inspection_obs': sequence_of_obs_ff_indices[rows_with_mismatched_labels, :],
        'non_chosen_rows_of_df': non_chosen_rows_of_df
    }

    chosen_rows_of_df = free_selection_df[~free_selection_df['starting_point_index'].isin(
        point_index_with_mismatched_labels)].copy()
    chosen_row_indices = chosen_rows_of_df.index.values.tolist()
    sequence_of_obs_ff_indices = [
        sequence_of_obs_ff_indices[i] for i in chosen_row_indices]

    free_selection_x_df['point_index'] = point_index_array
    free_selection_x_df = free_selection_x_df[~free_selection_x_df['point_index'].isin(
        point_index_with_mismatched_labels)].reset_index(drop=True)

    free_selection_labels = free_selection_labels[free_selection_x_df.index.values]

    return free_selection_x_df, free_selection_labels, cases_for_inspection, chosen_rows_of_df, sequence_of_obs_ff_indices, free_selection_x_df_for_plotting


def make_free_selection_predictions_using_trained_model(trained_model, ff_indices, target_indices, ff_dataframe, ff_real_position_sorted, ff_caught_T_new, monkey_information, time_of_evaluation=None):
    # if time_of_evaluation is None, then it will be set to the beginning of the trial
    if time_of_evaluation is None:
        # here it denotes the beginning of a trial
        time_of_evaluation = ff_caught_T_new[target_indices-1]
    starting_point_index = monkey_information['point_index'].values[np.searchsorted(
        monkey_information['time'], time_of_evaluation)]
    time_of_evaluation = monkey_information['time'][starting_point_index]

    substitute_free_selection_df = pd.DataFrame({'ff_index': ff_indices,
                                                'time': time_of_evaluation,
                                                 'target_index': target_indices,
                                                 'starting_point_index': starting_point_index})

    inputs, labels, cases_for_inspection, chosen_rows_of_df, sequence_of_obs_ff_indices, free_selection_x_df_for_plotting \
        = organize_free_selection_x(substitute_free_selection_df, ff_dataframe,
                                    ff_real_position_sorted, monkey_information,
                                    only_select_n_ff_case=None, num_ff_per_row=5, guarantee_including_target_info=True)

    # naive bayes
    # trained_model = gnb

    # predict
    y_pred = trained_model.predict(inputs)

    # evaluate
    accuracy = accuracy_score(labels, y_pred)
    print("accuracy:", accuracy)

    # confusion matrix
    print(confusion_matrix(labels, y_pred))

    return inputs, labels, y_pred


def guarantee_ff_dataframe_include_target_info(target_info, ff_dataframe_sub, ff_real_position_sorted, ff_dataframe, monkey_information):
    # make sure that for every point_index in ff_dataframe_sub, a row for the target_index is included
    # use merge to make sure that for each point_index, the ff_index of the target is included
    ff_dataframe_w_targets = pd.merge(target_info[['point_index', 'ff_index']], ff_dataframe_sub, on=[
                                      'point_index', 'ff_index'], how='outer')
    ff_dataframe_w_targets['selection_criterion'] = ff_dataframe_w_targets['selection_criterion'].fillna(
        -9999)
    # make sure ff_index of the target is not negative
    ff_dataframe_w_targets = ff_dataframe_w_targets[ff_dataframe_w_targets['ff_index'] >= 0]

    # If there's no existing information of the target in ff_dataframe_sub, then we'll find the info anew
    ff_dataframe_w_targets_na = ff_dataframe_w_targets[ff_dataframe_w_targets['ff_angle'].isna(
    )].copy()
    ff_info = decision_making_utils.find_many_ff_info_anew(
        ff_dataframe_w_targets_na.ff_index.values, ff_dataframe_w_targets_na.point_index.values, ff_real_position_sorted, ff_dataframe, monkey_information)
    # fill up the nan values by the new info
    ff_dataframe_w_targets.loc[ff_dataframe_w_targets['ff_distance'].isna(
    ), 'ff_distance'] = ff_info['ff_distance'].values
    ff_dataframe_w_targets.loc[ff_dataframe_w_targets['ff_angle'].isna(
    ), 'ff_angle'] = ff_info['ff_angle'].values
    ff_dataframe_w_targets.loc[ff_dataframe_w_targets['ff_angle_boundary'].isna(
    ), 'ff_angle_boundary'] = ff_info['ff_angle_boundary'].values
    ff_dataframe_w_targets.loc[ff_dataframe_w_targets['time_since_last_vis'].isna(
    ), 'time_since_last_vis'] = ff_info['time_since_last_vis'].values
    ff_dataframe_w_targets['abs_ff_angle'] = np.abs(
        ff_dataframe_w_targets['ff_angle'])
    ff_dataframe_w_targets['abs_ff_angle_boundary'] = np.abs(
        ff_dataframe_w_targets['ff_angle_boundary'])
    ff_dataframe_sub = ff_dataframe_w_targets.copy()
    return ff_dataframe_sub


def guarantee_n_ff_per_point_index_in_ff_dataframe(ff_dataframe_sub, all_point_index, num_ff_per_row=3, only_select_n_ff_case=None, selection_criterion_if_too_many_ff='abs_ff_angle',
                                                   placeholder_ff_index=-10, placeholder_ff_distance=400, placeholder_ff_angle=math.pi/4, placeholder_ff_angle_boundary=math.pi/4, placeholder_time_since_last_vis=3,
                                                   placeholder_time_till_next_visible=10, placeholder_curv_diff=0.6, curv_of_traj_df=None):

    count_of_ff = ff_dataframe_sub.groupby(
        'point_index').count().reset_index(drop=False)
    # make sure that every point in point_index will have corresponding info in count_of_ff
    point_index_df = pd.DataFrame({'point_index': all_point_index})
    count_of_ff = pd.merge(point_index_df, count_of_ff,
                           on='point_index', how='left').fillna(0)
    if 'selection_criterion' not in ff_dataframe_sub.columns:
        ff_dataframe_sub['selection_criterion'] = ff_dataframe_sub[selection_criterion_if_too_many_ff]
    if only_select_n_ff_case is not None:
        count_of_ff = count_of_ff[count_of_ff['ff_index']
                                  == num_ff_per_row].copy()
        ff_dataframe_sub = ff_dataframe_sub[ff_dataframe_sub['point_index'].isin(
            count_of_ff.point_index.values)].copy()
    else:
        # for each point, if the count of ff is greater than num_ff_per_row, the take the first num_ff_per_row ff
        # if the count of ff is less than num_ff_per_row, then take all of them and also add placeholders will ff_index=-10
        not_enough_ff = count_of_ff[count_of_ff['ff_index']
                                    < num_ff_per_row].copy()
        not_enough_ff['num_missing'] = num_ff_per_row - \
            not_enough_ff['ff_index']
        not_enough_ff['num_missing'] = not_enough_ff['num_missing'].astype(int)
        added_point_index = np.repeat(
            not_enough_ff.point_index.values, not_enough_ff['num_missing'].values)
        df_to_add = pd.DataFrame({'point_index': added_point_index,
                                  'ff_index': np.repeat(placeholder_ff_index, len(added_point_index)),
                                  'time_since_last_vis': np.repeat(placeholder_time_since_last_vis, len(added_point_index)),
                                  'ff_distance': np.repeat(placeholder_ff_distance, len(added_point_index)),
                                  # 'ff_angle': np.repeat(placeholder_ff_angle, len(added_point_index)),
                                  # 'ff_angle_boundary': np.repeat(placeholder_ff_angle_boundary, len(added_point_index)),
                                  # 'abs_ff_angle': np.repeat(abs(placeholder_ff_distance), len(added_point_index)),
                                  # 'abs_ff_angle_boundary': np.repeat(abs(placeholder_ff_distance), len(added_point_index))
                                  })

        # In case there are other columns that need placeholders
        df_to_add = fill_up_additional_attributes_for_placeholders(df_to_add, ff_dataframe_sub, placeholder_curv_diff=placeholder_curv_diff, placeholder_ff_distance=placeholder_ff_distance, placeholder_ff_angle=placeholder_ff_angle,
                                                                   placeholder_ff_angle_boundary=placeholder_ff_angle_boundary, placeholder_time_till_next_visible=placeholder_time_till_next_visible)

        if curv_of_traj_df is not None:
            df_to_add = df_to_add.merge(
                curv_of_traj_df[['point_index', 'curv_of_traj']], on='point_index', how='left')

        df_to_add['selection_criterion'] = max(
            9999, ff_dataframe_sub[selection_criterion_if_too_many_ff].max()+1000)

        # combine the info and the placeholders
        ff_dataframe_sub = pd.concat([ff_dataframe_sub, df_to_add], axis=0)
        # now, we keep only num_ff_per_row ff for each point_index, and we sort them by selection_criterion_if_too_many_ff
        ff_dataframe_sub.sort_values(
            ['point_index', 'selection_criterion'], ascending=True, inplace=True)
        ff_dataframe_sub = ff_dataframe_sub.groupby(
            'point_index').head(num_ff_per_row)
    # ff_dataframe_sub.drop(columns=['selection_criterion'], inplace=True)

    return ff_dataframe_sub


def fill_up_additional_attributes_for_placeholders(placeholder_df,
                                                   ff_dataframe_sub,
                                                   placeholder_curv_diff=0.6,
                                                   placeholder_ff_distance=400,
                                                   placeholder_ff_angle=math.pi/4,
                                                   placeholder_ff_angle_boundary=math.pi/4,
                                                   placeholder_time_till_next_visible=10,
                                                   placeholder_duration_of_last_vis_period=0,
                                                   placeholder_opt_arc_curv=0.1):
    attributes = []
    additional_placeholder_mapping = {}

    for column in ['ff_angle', 'abs_ff_angle', 'ff_angle_boundary', 'abs_ff_angle_boundary',
                   'time_till_next_visible', 'duration_of_last_vis_period', 'curv_diff', 'abs_curv_diff',
                   'next_seen_ff_distance', 'next_seen_ff_angle', 'next_seen_ff_angle_boundary', 'next_seen_curv_diff', 'next_seen_abs_curv_diff',
                   'distance_from_monkey_now_to_monkey_when_ff_next_seen', 'angle_from_monkey_now_to_monkey_when_ff_next_seen',
                   'last_seen_ff_distance', 'last_seen_ff_angle', 'last_seen_ff_angle_boundary', 'last_seen_curv_diff', 'last_seen_abs_curv_diff',
                   'distance_from_monkey_now_to_monkey_when_ff_last_seen', 'angle_from_monkey_now_to_monkey_when_ff_last_seen',
                   'opt_arc_curv']:
        if column in ff_dataframe_sub.columns:
            attributes.append(column)

    # for each column, we add a placeholder value and a boolean indicating whether the placeholder value should be sampled to have both positive and negative values
    additional_placeholder_mapping = {'ff_angle': [placeholder_ff_angle, True],
                                      'abs_ff_angle': [placeholder_ff_angle, False],
                                      'ff_angle_boundary': [placeholder_ff_angle_boundary, True],
                                      'abs_ff_angle_boundary': [placeholder_ff_angle_boundary, False],
                                      'time_till_next_visible': [placeholder_time_till_next_visible, False],
                                      'duration_of_last_vis_period': [placeholder_duration_of_last_vis_period, False],
                                      'curv_diff': [placeholder_curv_diff, True],
                                      'abs_curv_diff': [placeholder_curv_diff, False],
                                      'next_seen_ff_distance': [placeholder_ff_distance, False],
                                      'next_seen_ff_angle': [placeholder_ff_angle, True],
                                      'next_seen_ff_angle_boundary': [placeholder_ff_angle_boundary, True],
                                      'next_seen_curv_diff': [placeholder_curv_diff, True],
                                      'next_seen_abs_curv_diff': [placeholder_curv_diff, False],
                                      'distance_from_monkey_now_to_monkey_when_ff_next_seen': [placeholder_ff_distance, False],
                                      'angle_from_monkey_now_to_monkey_when_ff_next_seen': [placeholder_ff_angle, True],
                                      'last_seen_ff_distance': [placeholder_ff_distance, False],
                                      'last_seen_ff_angle': [placeholder_ff_angle, True],
                                      'last_seen_ff_angle_boundary': [placeholder_ff_angle_boundary, True],
                                      'last_seen_curv_diff': [placeholder_curv_diff, True],
                                      'last_seen_abs_curv_diff': [placeholder_curv_diff, False],
                                      'distance_from_monkey_now_to_monkey_when_ff_last_seen': [placeholder_ff_distance, False],
                                      'angle_from_monkey_now_to_monkey_when_ff_last_seen': [placeholder_ff_angle, True],
                                      'opt_arc_curv': [placeholder_opt_arc_curv, True]}

    placeholder_df = decision_making_utils.fill_new_columns_with_placeholder_values(
        placeholder_df, columns=attributes, additional_placeholder_mapping=additional_placeholder_mapping)
    return placeholder_df
