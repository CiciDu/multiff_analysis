import sys
from data_wrangling import specific_utils
from planning_analysis.show_planning import alt_ff_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
from data_wrangling import specific_utils
from null_behaviors import curvature_utils, curv_of_traj_utils, optimal_arc_utils
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import copy
from null_behaviors import curv_of_traj_utils
import numpy as np
import pandas as pd
import math


# try replacing ff_dataframe_visible with ff_info_at_start_df
def get_only_stop_ff_df(closest_stop_to_capture_df, ff_real_position_sorted, ff_caught_T_new, monkey_information, curv_of_traj_df, ff_dataframe_visible, stop_period_duration=2,
                         ref_point_mode='distance', ref_point_value=-150,
                         optimal_arc_type='norm_opt_arc'):

    if ref_point_mode == 'time after stop ff visible':
        drop_na = True
    else:
        drop_na = False
    ff_info = alt_ff_utils.get_all_captured_ff_first_seen_and_last_seen_info(closest_stop_to_capture_df, stop_period_duration, ff_dataframe_visible, monkey_information, 
                                                                                drop_na=drop_na)
    ff_info['stop_time'] = monkey_information.loc[ff_info['stop_point_index'].values, 'time'].values
    ff_info['time_since_ff_last_seen'] = ff_info['stop_time'].values - ff_info['time_ff_last_seen'].values
    ff_info.sort_values(by=['stop_point_index', 'time_since_ff_last_seen'], ascending=[True, True], inplace=True)
    # if there are duplicated stop point index, the row with bigger ff_index will be kept
    ff_info = ff_info.groupby('stop_point_index').first().reset_index(drop=False)
    ff_info2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(ff_info, monkey_information, ff_real_position_sorted, ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
    ff_info2 = find_stops_near_ff_utils.add_monkey_info_before_stop(monkey_information, ff_info2)
    
    optimal_arc_stop_at_visible_boundary = True if (optimal_arc_type == 'opt_arc_stop_first_vis_bdry') else False
    curv_df = curvature_utils.make_curvature_df(ff_info2, curv_of_traj_df, clean=False, monkey_information=monkey_information, optimal_arc_stop_at_visible_boundary=optimal_arc_stop_at_visible_boundary)

    if optimal_arc_type == 'opt_arc_stop_closest':
        curv_df = optimal_arc_utils.update_curvature_df_to_let_optimal_arc_stop_at_closest_point_to_monkey_stop(curv_df, ff_info2, ff_info, 
                                                                                    ff_real_position_sorted, monkey_information)
            
    # use merge to add curvature_info
    shared_columns = ['ff_index', 'point_index', 'optimal_curvature', 'optimal_arc_measure', 'optimal_arc_radius', 'optimal_arc_end_direction', 'curv_to_ff_center', 
                        'arc_radius_to_ff_center', 'd_heading_to_center', 'optimal_arc_d_heading', 'optimal_arc_end_x', 'optimal_arc_end_y', 'arc_end_x_to_ff_center', 'arc_end_y_to_ff_center']
    only_stop_ff_df = ff_info2.merge(curv_df[shared_columns], on=['ff_index', 'point_index'], how='left')
    only_stop_ff_df = only_stop_ff_df.merge(curv_of_traj_df[['point_index', 'curvature_of_traj']], on='point_index', how='left')

    only_stop_ff_df['d_heading_of_traj'] = only_stop_ff_df['monkey_angle_before_stop'] - only_stop_ff_df['monkey_angle']
    only_stop_ff_df['d_heading_of_traj'] = find_stops_near_ff_utils.confine_angle_to_within_one_pie(only_stop_ff_df['d_heading_of_traj'].values)
    only_stop_ff_df[['stop_d_heading_of_arc', 'ref_d_heading_of_traj']] = only_stop_ff_df[['optimal_arc_d_heading', 'd_heading_of_traj']]
    only_stop_ff_df[['stop_d_heading_of_arc', 'ref_d_heading_of_traj']] = only_stop_ff_df[['stop_d_heading_of_arc', 'ref_d_heading_of_traj']]*180/math.pi
    only_stop_ff_df['ref_time'] = monkey_information.loc[only_stop_ff_df['point_index'].values, 'time'].values
    only_stop_ff_df['stop_time'] = monkey_information.loc[only_stop_ff_df['stop_point_index'].values, 'time'].values
    only_stop_ff_df.rename(columns={'point_index': 'ref_point_index'}, inplace=True)
    only_stop_ff_df['beginning_time'] = only_stop_ff_df['stop_time'] - stop_period_duration

    only_stop_ff_df['ref_d_heading_of_traj'] = only_stop_ff_df['ref_d_heading_of_traj'] % 360
    only_stop_ff_df.loc[only_stop_ff_df['ref_d_heading_of_traj'] > 180, 'ref_d_heading_of_traj'] = only_stop_ff_df.loc[only_stop_ff_df['ref_d_heading_of_traj'] > 180, 'ref_d_heading_of_traj'] - 360
    only_stop_ff_df['dev_d_angle_from_null'] = only_stop_ff_df['ref_d_heading_of_traj'] - only_stop_ff_df['stop_d_heading_of_arc']
    
    # get curv range info etc
    curv_of_traj_stat_df, only_stop_ff_df = plan_factors_utils.find_curv_of_traj_stat_df(only_stop_ff_df, curv_of_traj_df, start_time_column='beginning_time',
                                                                                         end_time_column='stop_time', add_to_df_to_iter=False)
    columns_to_add = ['curv_mean', 'curv_std', 'curv_min', 'curv_25%', 'curv_50%', 'curv_75%', 
                              'curv_max', 'curv_iqr', 'curv_range']
    only_stop_ff_df = only_stop_ff_df.merge(curv_of_traj_stat_df[columns_to_add + ['stop_point_index']], on='stop_point_index', how='left')

    # add angle_from_stop_ff_to_stop
    only_stop_ff_df['stop_x'], only_stop_ff_df['stop_y'] = monkey_information.loc[only_stop_ff_df['stop_point_index'], ['monkey_x', 'monkey_y']].values.T
    only_stop_ff_df['angle_from_stop_ff_to_stop'] = specific_utils.calculate_angles_to_ff_centers(ff_x=only_stop_ff_df['stop_x'].values, ff_y=only_stop_ff_df['stop_y'], \
                                                                                        mx=only_stop_ff_df['ff_x'].values, my=only_stop_ff_df['ff_y'], 
                                                                                        m_angle=only_stop_ff_df['monkey_angle_before_stop'])
    only_stop_ff_df['dir_from_stop_ff_to_stop'] = np.sign(only_stop_ff_df['angle_from_stop_ff_to_stop'])
    
    curv_of_traj_df_w_one_sided_window, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode([-25, 0], monkey_information, ff_caught_T_new, 
                                                                                                                curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False)
    only_stop_ff_df = plan_factors_utils._add_column_curvature_of_traj_before_stop(only_stop_ff_df, curv_of_traj_df_w_one_sided_window)

    only_stop_ff_df = only_stop_ff_df.sort_values(by='stop_point_index').reset_index(drop=True)

    return only_stop_ff_df



def find_ff_info_and_stop_ff_info_at_start_df(only_stop_ff_df, monkey_info_in_all_stop_periods, ff_flash_sorted, 
                                              ff_real_position_sorted, ff_life_sorted, ff_radius=10, 
                                              dropna=False,
                                              guarantee_info_for_stop_ff=False,
                                              filter_out_ff_not_in_front_of_monkey_at_ref_point=True
                                              ):
    
    ff_info_at_start_df = only_stop_ff_df[['ref_point_index', 'ref_time', 'stop_point_index', 
                                           'beginning_time', 'stop_time', 'monkey_x', 'monkey_y', 'monkey_angle',
                                           'ff_index', 'ff_x', 'ff_y', 'ff_angle'
                                           ]].copy()

    ff_info_at_start_df.rename(columns={'ff_index': 'stop_ff_index',
                                        'ff_x': 'stop_ff_x',
                                        'ff_y': 'stop_ff_y',
                                        'ff_angle': 'stop_ff_angle'}, inplace=True)
    
    monkey_info_in_all_stop_periods = monkey_info_in_all_stop_periods[monkey_info_in_all_stop_periods['stop_point_index'].isin(only_stop_ff_df['stop_point_index'].values)].copy()
    flash_time_info = _get_info_of_ff_whose_flash_time_overlaps_with_stop_periods(monkey_info_in_all_stop_periods, ff_flash_sorted, ff_life_sorted)

    if guarantee_info_for_stop_ff:
        stop_ff_info = only_stop_ff_df[['ff_index', 'stop_point_index']]
        flash_time_info = flash_time_info.merge(stop_ff_info, on=['stop_point_index', 'ff_index'], how='outer')

    ff_info_at_start_df = ff_info_at_start_df.merge(flash_time_info, on=['stop_point_index'], how='right')
    
    ff_info_at_start_df['ff_x'], ff_info_at_start_df['ff_y'] = ff_real_position_sorted[ff_info_at_start_df['ff_index'].values].T
    ff_info_at_start_df = _add_basic_ff_info_to_df_for_ff(ff_info_at_start_df, ff_radius=ff_radius)
    # add info related to flash time

    ff_info_at_start_df = furnish_ff_info_at_start_df(ff_info_at_start_df)

    if guarantee_info_for_stop_ff:
        ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'stop_ff_index', 'ff_index']] = \
            ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'stop_ff_index', 'ff_index']].astype('int')
    else:
        ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'stop_ff_index', 'ff_index', 'earliest_flash_point_index', 'latest_flash_point_index']] = \
            ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'stop_ff_index', 'ff_index', 'earliest_flash_point_index', 'latest_flash_point_index']].astype('int')
    

    stop_ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['ff_index'] == ff_info_at_start_df['stop_ff_index']].copy()
    # filter both df such that only when stop ff is in front of the monkey at ref point will such stop point be preserved.
    if filter_out_ff_not_in_front_of_monkey_at_ref_point:
        orig_len = len(stop_ff_info_at_start_df)
        valid_stop_periods, stop_ff_info_at_start_df = _find_valid_stop_periods_for_a_ff(stop_ff_info_at_start_df)
        print(f'Filtered out {orig_len - len(stop_ff_info_at_start_df)} stop periods out of {orig_len} stop periods because they are not in front of the monkey at ref point')
    ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['stop_point_index'].isin(stop_ff_info_at_start_df['stop_point_index'].unique())].copy()

    if dropna:
        ff_info_at_start_df.dropna(axis=0, inplace=True)
        print('Dropped NaN values in ff_info_at_start_df')

    ff_info_at_start_df.reset_index(drop=True, inplace=True)
    stop_ff_info_at_start_df.drop_duplicates(inplace=True)
    stop_ff_info_at_start_df.reset_index(drop=True, inplace=True)

    return ff_info_at_start_df, stop_ff_info_at_start_df



def furnish_ff_info_at_start_df(ff_info_at_start_df):
    ff_info_at_start_df['earliest_flash_rel_time'] = ff_info_at_start_df['earliest_flash_time'] - ff_info_at_start_df['beginning_time']
    ff_info_at_start_df['latest_flash_rel_time'] = ff_info_at_start_df['latest_flash_time'] - ff_info_at_start_df['beginning_time']
    ff_info_at_start_df['ff_distance_to_stop_ff'] = np.linalg.norm([ff_info_at_start_df['stop_ff_x'] - ff_info_at_start_df['ff_x'], ff_info_at_start_df['stop_ff_y'] - ff_info_at_start_df['ff_y']], axis=0)
    
    ff_info_at_start_df['angle_diff_boundary'] = ff_info_at_start_df['ff_angle'] - ff_info_at_start_df['ff_angle_boundary']
    ff_info_at_start_df['angle_diff_boundary'] = ff_info_at_start_df['angle_diff_boundary'] % (2*math.pi)
    ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_boundary'] > math.pi, 'angle_diff_boundary'] = ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_boundary'] > math.pi, 'angle_diff_boundary'] - 2*math.pi
    
    ff_info_at_start_df['angle_diff_from_stop_ff'] = ff_info_at_start_df['ff_angle'] - ff_info_at_start_df['stop_ff_angle']
    ff_info_at_start_df['angle_diff_from_stop_ff'] = ff_info_at_start_df['angle_diff_from_stop_ff'] % (2*math.pi)
    ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_from_stop_ff'] > math.pi, 'angle_diff_from_stop_ff'] = ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_from_stop_ff'] > math.pi, 'angle_diff_from_stop_ff'] - 2*math.pi
    return ff_info_at_start_df


def find_monkey_info_in_all_stop_periods(all_start_time, all_end_time, all_group_id, monkey_information):

    monkey_info_in_all_stop_periods = monkey_information[['time', 'point_index', 'monkey_x', 'monkey_y', 'monkey_angle', 'dt']].copy()
    monkey_info_in_all_stop_periods = plan_factors_utils.extend_df_based_on_groups(monkey_info_in_all_stop_periods, all_start_time, all_end_time, all_group_id, group_id='stop_point_index')

    return monkey_info_in_all_stop_periods


def find_ff_flash_df_within_all_stop_periods(monkey_info_in_all_stop_periods, ff_caught_T_new, ff_real_position_sorted, ff_flash_sorted, ff_radius=10):
    ff_flash_df = pd.DataFrame()
    for i in range(len(ff_real_position_sorted)):
        if i % 100 == 0:
            print(f"Processing {i}-th ff out of {len(ff_real_position_sorted)} ff")
        ff_flash = ff_flash_sorted[i]
        monkey_sub_for_ff = _find_monkey_sub_within_any_flash_period_for_a_ff(monkey_info_in_all_stop_periods, ff_flash)
        monkey_sub_for_ff['ff_index'] = i
        monkey_sub_for_ff[['ff_x', 'ff_y']] = ff_real_position_sorted[i]
        monkey_sub_for_ff = _add_basic_ff_info_to_df_for_ff(monkey_sub_for_ff, ff_radius=ff_radius)
        # for each stop period, if the minimum ff_distance and abs_ff_angle_boundary is less than 800 and 45 degrees, then keep all points in that stop period
        grouped_min_info = monkey_sub_for_ff[['stop_point_index', 'abs_ff_angle_boundary', 'ff_distance']].groupby('stop_point_index').min()
        grouped_min_info = grouped_min_info[(grouped_min_info['ff_distance'] < 800) & (grouped_min_info['abs_ff_angle_boundary'] < 45/180*math.pi)]
        if len(grouped_min_info) > 0:
            monkey_sub_for_ff_final = monkey_sub_for_ff[monkey_sub_for_ff['flash_period'].isin(grouped_min_info.index)].copy()
            ff_flash_df = pd.concat([ff_flash_df, monkey_sub_for_ff_final], axis=0)
    ff_flash_df.reset_index(drop=True, inplace=True)
    return ff_flash_df



def _get_info_of_ff_whose_flash_time_overlaps_with_stop_periods(monkey_info_in_all_stop_periods, ff_flash_sorted, ff_life_sorted):
    time = monkey_info_in_all_stop_periods['time'].values
    all_in_flash_iloc = []
    all_ff_index = []
    for i in range(len(ff_life_sorted)):
        ff_flash = ff_flash_sorted[i]
        in_flash_iloc, _ = _find_index_of_points_within_flash_period_for_a_ff(time, ff_flash)
        all_in_flash_iloc.extend(in_flash_iloc.tolist())
        all_ff_index.extend([i] * len(in_flash_iloc))
    monkey_sub_for_ff = monkey_info_in_all_stop_periods[['point_index', 'time', 'dt', 'stop_point_index']].iloc[all_in_flash_iloc].copy()
    monkey_sub_for_ff['ff_index'] = all_ff_index
    flash_time_info = monkey_sub_for_ff.groupby(['ff_index', 'stop_point_index']).agg(earliest_flash_point_index=('point_index', 'min'),
                                                                                latest_flash_point_index=('point_index', 'max'),
                                                                                earliest_flash_time=('time', 'min'),
                                                                                latest_flash_time=('time', 'max'),
                                                                                flash_duration=('dt', 'sum'))
    flash_time_info.reset_index(drop=False, inplace=True)
    return flash_time_info



def _find_valid_stop_periods_for_a_ff(only_stop_ff_df_sub):
    only_stop_ff_df_sub = only_stop_ff_df_sub.copy()
    #valid_stop_periods_df = stop_sub_for_ff.copy()
    valid_stop_periods_df = only_stop_ff_df_sub[(only_stop_ff_df_sub['ff_distance'] < 1000) & 
                                            (only_stop_ff_df_sub['abs_ff_angle_boundary'] < 90/180*math.pi)].copy()
    valid_stop_periods = valid_stop_periods_df['stop_point_index'].unique()
    return valid_stop_periods, valid_stop_periods_df



def _find_index_of_points_within_flash_period_for_a_ff(time, ff_flash):
    point_corr_position_in_flash = np.searchsorted(ff_flash.flatten(), time)
    in_flash_iloc = np.where(point_corr_position_in_flash % 2 == 1)[0]
    return in_flash_iloc, point_corr_position_in_flash


def _find_monkey_sub_within_any_flash_period_for_a_ff(monkey_info_in_all_stop_periods, ff_flash):
    in_flash_iloc, point_corr_position_in_flash = _find_index_of_points_within_flash_period_for_a_ff(monkey_info_in_all_stop_periods['time'].values, ff_flash)
    monkey_sub_for_ff = monkey_info_in_all_stop_periods.iloc[in_flash_iloc].copy()
    monkey_sub_for_ff['flash_period'] = (point_corr_position_in_flash[in_flash_iloc]/2).astype(int)
    return monkey_sub_for_ff


def _add_basic_ff_info_to_df_for_ff(df, ff_radius=10):
    # For the selected time points, find ff distance and angle to boundary 

    df['ff_distance'] = np.sqrt((df['ff_x'] - df['monkey_x'])**2 + (df['ff_y'] - df['monkey_y'])**2)
    df['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(df['ff_x'],df['ff_y'], mx=df['monkey_x'], my=df['monkey_y'], m_angle=df['monkey_angle'])
    df['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(angles_to_ff=df['ff_angle'], distances_to_ff=df['ff_distance'], ff_radius=ff_radius)
    df['abs_ff_angle_boundary'] = np.abs(df['ff_angle_boundary'])
    return df


def get_x_features_df(ff_info_at_start_df, stop_ff_info_at_start_df, 
                        columns_not_to_include=[],
                        rank_columns_not_to_include=[],
                        flash_or_vis='flash',
                        list_of_stop_ff_cluster_radius=[100, 200, 300],
                        list_of_stop_ff_ang_cluster_radius=[20],
                        list_of_start_dist_cluster_radius=[100, 200, 300],
                        list_of_start_ang_cluster_radius=[20],
                        list_of_flash_cluster_period=[[1.0, 1.5], [1.5, 2.0]],                         
                        ):

    all_cluster_info = pd.DataFrame()
    all_cluster_info = stop_ff_info_at_start_df[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'angle_diff_boundary', 'flash_duration',
                                                  'earliest_flash_rel_time', 'latest_flash_rel_time']].copy()
    # # rename all the columns in all_cluster_info_in_a_row to add a prefix 'stop_ff' and a suffix 'at_ref_point'
    all_cluster_info.columns = ['stop_ff_' + col + '_at_ref' if (col[:3] != 'ff_') else 'stop_' + col + '_at_ref' for col in all_cluster_info.columns]
    all_cluster_info.reset_index(drop=True, inplace=True)
    all_cluster_info[['stop_point_index', 'stop_point_index']] = stop_ff_info_at_start_df[['stop_point_index', 'stop_point_index']].values

    ff_info_at_start_df, all_cluster_names = find_clusters_in_ff_info_at_start_df(ff_info_at_start_df, stop_ff_info_at_start_df,
                                                                                    list_of_stop_ff_cluster_radius=list_of_stop_ff_cluster_radius,
                                                                                    list_of_stop_ff_ang_cluster_radius=list_of_stop_ff_ang_cluster_radius,
                                                                                    list_of_start_dist_cluster_radius=list_of_start_dist_cluster_radius,
                                                                                    list_of_start_ang_cluster_radius=list_of_start_ang_cluster_radius,
                                                                                    list_of_flash_cluster_period=list_of_flash_cluster_period,
                                                                                  )
    ff_info_at_start_df = get_ranks_of_columns_to_include_within_each_stop_period(ff_info_at_start_df, rank_columns_not_to_include=rank_columns_not_to_include,
                                                                                  flash_or_vis=flash_or_vis)

    cluster_factors_df, cluster_agg_df = get_cluster_and_agg_df(ff_info_at_start_df, all_cluster_names,
                                                                columns_not_to_include=columns_not_to_include, rank_columns_not_to_include=rank_columns_not_to_include,
                                                                flash_or_vis=flash_or_vis) 

    all_cluster_info = all_cluster_info.merge(cluster_factors_df, on='stop_point_index', how='outer')
    all_cluster_info = all_cluster_info.merge(cluster_agg_df, on='stop_point_index', how='outer')
    x_features_df = all_cluster_info.reset_index(drop=True)
    return x_features_df, all_cluster_names


def get_cluster_and_agg_df(ff_info_at_start_df, all_cluster_names,
                           flash_or_vis='flash',
                           columns_not_to_include=[],
                           rank_columns_not_to_include=[],
                           ):

    columns_to_include=['ff_distance', 'ff_angle', 'angle_diff_boundary']
    rank_columns_to_include = ['abs_ff_angle_boundary', 'ff_distance']

    if flash_or_vis is not None:
        additional_columns =  [f'earliest_{flash_or_vis}_rel_time', f'latest_{flash_or_vis}_rel_time', f'{flash_or_vis}_duration']
        columns_to_include.extend(additional_columns)
        rank_columns_to_include.extend(additional_columns)

    columns_to_include = [column for column in columns_to_include if column not in columns_not_to_include]
    rank_columns_to_include = [column for column in rank_columns_to_include if column not in rank_columns_not_to_include]

    id_columns = [column for column in ff_info_at_start_df.columns if column not in all_cluster_names]
    ff_info_at_start_df_melted = pd.melt(ff_info_at_start_df, id_vars=id_columns, value_vars=all_cluster_names, var_name='group', value_name='whether_ff_selected')
    ff_info_at_start_df_melted = ff_info_at_start_df_melted[ff_info_at_start_df_melted['whether_ff_selected'] == True]

    columns_to_include = copy.deepcopy(columns_to_include)
    columns_to_include.extend([col + '_rank' for col in rank_columns_to_include])
    cluster_factors_df = _get_cluster_factors_df(ff_info_at_start_df_melted, columns_not_to_include=columns_not_to_include, flash_or_vis=flash_or_vis)

    cluster_factors_df['group'] = cluster_factors_df['group'] + '_' + cluster_factors_df['ff'].str.upper()
    cluster_factors_df = cluster_factors_df.sort_values(by=['group', 'stop_point_index']).reset_index(drop=True).drop(columns=['ff'])
    cluster_factors_df = slice_and_combd_a_df(cluster_factors_df)

    cluster_agg_df = _get_cluster_agg_df(ff_info_at_start_df_melted, flash_or_vis=flash_or_vis)
    cluster_agg_df = slice_and_combd_a_df(cluster_agg_df)

    cluster_factors_df.reset_index(drop=True, inplace=True)
    cluster_agg_df.reset_index(drop=True, inplace=True)

    return cluster_factors_df, cluster_agg_df


    
def _get_cluster_factors_df(ff_info_at_start_df_melted,
                            flash_or_vis='flash',
                            columns_not_to_include=[]
                            ):

    # types_of_ff_to_include=['leftmost', 'rightmost', f'earliest_{flash_or_vis}',
    #                         f'latest_{flash_or_vis}', f'longest_{flash_or_vis}'],

    columns_to_include=['ff_distance', 
                        'ff_angle', 
                        'ff_angle_boundary',
                        'angle_diff_boundary']
    if flash_or_vis is not None:
        columns_to_include.extend([f'earliest_{flash_or_vis}_rel_time',
                        f'latest_{flash_or_vis}_rel_time', 
                        f'{flash_or_vis}_duration'])
    columns_to_include = [column for column in columns_to_include if column not in columns_not_to_include]
                            
    cluster_factors_df = pd.DataFrame()


    columns_to_get_max_dict = {'ff_angle': 'leftmost'}
    columns_to_get_min_dict = {'ff_angle': 'rightmost'}

    if flash_or_vis is not None:
        columns_to_get_max_dict.update({f'latest_{flash_or_vis}_rel_time': f'latest_{flash_or_vis}',
                                        f'{flash_or_vis}_duration': f'longest_{flash_or_vis}'})
                        
        columns_to_get_max_dict.update({f'earliest_{flash_or_vis}_rel_time': f'earliest_{flash_or_vis}'})

    columns_to_get_max = list(columns_to_get_max_dict.keys())
    columns_to_get_max = [column for column in columns_to_get_max if (column not in columns_not_to_include)]
    for column in columns_to_get_max:
        ff = columns_to_get_max_dict[column]
        max_id = ff_info_at_start_df_melted.groupby(['stop_point_index', 'group'])[column].idxmax()
        rows_to_be_added = ff_info_at_start_df_melted.loc[max_id].copy()
        rows_to_be_added['ff'] = ff
        cluster_factors_df = pd.concat([cluster_factors_df, rows_to_be_added], axis=0)

    columns_to_get_min = list(columns_to_get_min_dict.keys())
    columns_to_get_min = [column for column in columns_to_get_min if (column not in columns_not_to_include)]
    for column in columns_to_get_min:
        ff = columns_to_get_min_dict[column]
        max_id = ff_info_at_start_df_melted.groupby(['stop_point_index', 'group'])[column].idxmin()
        rows_to_be_added = ff_info_at_start_df_melted.loc[max_id].copy()
        rows_to_be_added['ff'] = ff
        cluster_factors_df = pd.concat([cluster_factors_df, rows_to_be_added], axis=0)

    cluster_factors_df = cluster_factors_df[columns_to_include + ['stop_point_index', 'group', 'ff']]

    return cluster_factors_df



def _get_cluster_agg_df(ff_info_at_start_df_melted,
                        flash_or_vis='flash'):
    
    agg_dict = {
        'ff_angle': [('combd_min_ff_angle', 'min'), ('combd_max_ff_angle', 'max'), ('combd_median_ff_angle', 'median')],
        'ff_distance': [('combd_min_ff_distance', 'min'), ('combd_max_ff_distance', 'max'), ('combd_median_ff_distance', 'median')],
        'angle_diff_boundary': [('combd_min_angle_diff_boundary', 'min'), ('combd_max_angle_diff_boundary', 'max'), ('combd_median_angle_diff_boundary', 'median')],
        'ff_index': [('num_ff_in_cluster', 'count')]
    }

    if flash_or_vis is not None:
        agg_dict.update({f'earliest_{flash_or_vis}_rel_time': [(f'combd_earliest_{flash_or_vis}_rel_time', 'min')],
                    f'latest_{flash_or_vis}_rel_time': [(f'combd_latest_{flash_or_vis}_rel_time', 'max')],
                    f'{flash_or_vis}_duration': [(f'combd_total_{flash_or_vis}_duration', 'sum'), (f'combd_longest_{flash_or_vis}_duration', 'max')]})
                
    cluster_agg_df = ff_info_at_start_df_melted.groupby(['stop_point_index', 'group']).agg(agg_dict)


    # Flatten the MultiIndex columns
    cluster_agg_df.columns = [col[1] if col[1] != '' else col[0] for col in cluster_agg_df.columns.values]
    cluster_agg_df.reset_index(drop=False, inplace=True)
    cluster_agg_df = cluster_agg_df.sort_values(by=['group', 'stop_point_index']).reset_index(drop=True)
    return cluster_agg_df


def get_ranks_of_columns_to_include_within_each_stop_period(ff_info_at_start_df, rank_columns_not_to_include=[], flash_or_vis='flash'):

    rank_columns_to_include = ['abs_ff_angle_boundary', 'ff_distance']
    if flash_or_vis is not None:
        rank_columns_to_include.extend([f'earliest_{flash_or_vis}_rel_time', f'latest_{flash_or_vis}_rel_time', 
                                        f'{flash_or_vis}_duration'])
    rank_columns_to_include = [column for column in rank_columns_to_include if column not in rank_columns_not_to_include]

    ranked_columns = ff_info_at_start_df.groupby('stop_point_index')[rank_columns_to_include].rank(method='average', ascending=True)
    ranked_columns.rename(columns={col: col + '_rank' for col in rank_columns_to_include}, inplace=True)
    ff_info_at_start_df = pd.concat([ff_info_at_start_df, ranked_columns], axis=1)
    return ff_info_at_start_df

def slice_and_combd_a_df(df):
    total_rows = len(df)
    chunk_size = len(df['stop_point_index'].unique())

    # List to hold chunks
    chunks = []

    # Create and store chunks
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size, :].set_index('stop_point_index')
        group = chunk['group'].iloc[0]
        chunk.drop(columns=['group'], inplace=True)
        chunk.columns = [f'{group}_{column}' for column in chunk.columns]
        chunks.append(chunk)

    # Concatenate chunks horizontally
    result = pd.concat(chunks, axis=1)
    result['stop_point_index'] = result.index

    return result



def find_clusters_in_ff_info_at_start_df(ff_info_at_start_df, stop_ff_info_at_start_df,
                                        list_of_stop_ff_cluster_radius=[100, 200, 300],
                                        list_of_stop_ff_ang_cluster_radius=[20],
                                        list_of_start_dist_cluster_radius=[100, 200, 300],
                                        list_of_start_ang_cluster_radius=[20],
                                        list_of_flash_cluster_period=[[1.0, 1.5], [1.5, 2.0]],                          
                                         ):

    # first, for stop_period in ff_info_at_start_df that has no info, we'll use the info in stop_ff_info_at_start_df
    stop_ff_info_at_start_df_to_add = stop_ff_info_at_start_df[~stop_ff_info_at_start_df['stop_point_index'].isin(ff_info_at_start_df['stop_point_index'].values)]
    ff_info_at_start_df = pd.concat([ff_info_at_start_df, stop_ff_info_at_start_df_to_add], axis=0)

    stop_ff_info_at_start_df['stop_ff_distance'] = stop_ff_info_at_start_df['ff_distance'].values
    ff_info_at_start_df = ff_info_at_start_df.merge(stop_ff_info_at_start_df[['stop_point_index', 'stop_ff_distance']], on='stop_point_index', how='left')
    # only preserve ff that has equal or greater ff_distance than stop_ff_distance in the same period
    # ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['ff_distance'] >= ff_info_at_start_df['stop_ff_distance']].copy()
    
    all_cluster_names = []
    for n_cm in list_of_stop_ff_cluster_radius:
        column = f'stop_ff_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['ff_distance_to_stop_ff'] <= n_cm, column] = True

    # Stop ff cluster based on ff angle's proclicivity to stop ff's angle
    for n_angle in list_of_stop_ff_ang_cluster_radius:
        column = f'stop_ff_ang_cluster_{n_angle}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[np.abs(ff_info_at_start_df['angle_diff_from_stop_ff']) <= n_angle, column] = True

    # Cluster based on ff_distance at start point
    for n_cm in list_of_start_dist_cluster_radius:
        column = f'start_dist_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['ff_distance'] <= n_cm, column] = True

    # Cluster based on ff_angle_boundary at start point
    for n_angle in list_of_start_ang_cluster_radius:
        column = f'start_ang_cluster_{n_angle}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[np.abs(ff_info_at_start_df['ff_angle_boundary']*180/math.pi) <= n_angle, column] = True

    # Cluster based on ff that have flashed in the last 0.5s before stop
    for period in list_of_flash_cluster_period:
        # make n_s into string and replace . with _
        period_str = str(period[0]).replace('.', '_') + '_to_' + str(period[1]).replace('.', '_')
        column = f'flash_cluster_{period_str}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['latest_flash_rel_time'].between(period[0], period[1]), column] = True

    ff_info_at_start_df = _supply_zero_size_cluster_with_stop_ff_info(ff_info_at_start_df, stop_ff_info_at_start_df, all_cluster_names)
    
    return ff_info_at_start_df, all_cluster_names



def _supply_zero_size_cluster_with_stop_ff_info(ff_info_at_start_df, stop_ff_info_at_start_df, all_cluster_names):
    whether_each_cluster_has_enough = ff_info_at_start_df[all_cluster_names + ['stop_point_index']].groupby('stop_point_index').sum() == 0
    where_need_stop_ff = np.where(whether_each_cluster_has_enough)
    stop_periods = whether_each_cluster_has_enough.index.values[where_need_stop_ff[0]]
    groups = np.array(all_cluster_names)[where_need_stop_ff[1]]
    stop_ff_rows_to_add = stop_ff_info_at_start_df.set_index('stop_point_index').loc[stop_periods].reset_index(drop=False)
    stop_ff_rows_to_add['group'] = groups
    stop_ff_rows_to_add = pd.get_dummies(stop_ff_rows_to_add, columns=['group'], prefix='prefix')
    # drop all 'prefix_' from the column names
    stop_ff_rows_to_add.columns = [col.split('prefix_')[1] if 'prefix_' in col else col for col in stop_ff_rows_to_add.columns]
    ff_info_at_start_df = pd.concat([ff_info_at_start_df, stop_ff_rows_to_add], axis=0)
    return ff_info_at_start_df


def make_monkey_info_in_all_stop_periods(closest_stop_to_capture_df, monkey_information, stop_period_duration=2,
                                         all_end_time=None, all_start_time=None):
    if all_end_time is None:
        all_end_time = closest_stop_to_capture_df['time'].values 
    if all_start_time is None:
        all_start_time = closest_stop_to_capture_df['time'].values - stop_period_duration
    all_group_id = closest_stop_to_capture_df['stop_point_index'].values
    monkey_info_in_all_stop_periods = find_monkey_info_in_all_stop_periods(all_start_time, all_end_time, all_group_id, monkey_information)
    if 'stop_time' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df['stop_time'] = closest_stop_to_capture_df['time']
    if 'beginning_time' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df['beginning_time'] = closest_stop_to_capture_df['time'] - stop_period_duration
    monkey_info_in_all_stop_periods = monkey_info_in_all_stop_periods.merge(closest_stop_to_capture_df[['stop_point_index', 'beginning_time', 'stop_time']], on='stop_point_index', how='left')

    return monkey_info_in_all_stop_periods

