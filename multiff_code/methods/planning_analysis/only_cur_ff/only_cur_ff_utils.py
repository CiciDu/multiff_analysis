from data_wrangling import specific_utils
from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp, build_factor_comp_utils, build_factor_comp
from data_wrangling import specific_utils
from null_behaviors import curvature_utils, curv_of_traj_utils, opt_arc_utils
import numpy as np
import pandas as pd
import math
import copy
from null_behaviors import curv_of_traj_utils
import numpy as np
import pandas as pd
import math


# try replacing ff_dataframe_visible with ff_info_at_start_df
def get_only_cur_ff_df(closest_stop_to_capture_df, ff_real_position_sorted, ff_caught_T_new,
                       monkey_information, curv_of_traj_df, ff_dataframe_visible,
                       stop_period_duration=2, ref_point_mode='distance',
                       ref_point_value=-150, opt_arc_type='opt_arc_stop_closest',
                       curv_of_traj_mode='distance',
                       window_for_curv_of_traj=[-25, 0],
                       use_curv_to_ff_center=False,
                       ):

    # Determine if NA rows should be dropped based on reference mode
    drop_na = ref_point_mode == 'time after cur ff visible'

    # Extract info for each ff: when it was first/last seen and where the stop was
    ff_info = nxt_ff_utils.get_all_captured_ff_first_seen_and_last_seen_info(
        closest_stop_to_capture_df, stop_period_duration, ff_dataframe_visible,
        monkey_information, drop_na=drop_na
    )
    ff_info['stop_time'] = monkey_information.loc[ff_info['stop_point_index'], 'time'].values
    ff_info['time_since_ff_last_seen'] = ff_info['stop_time'] - \
        ff_info['time_ff_last_seen']

    # Sort by stop index and time since last seen, keep first (most recent) per stop
    ff_info.sort_values(
        ['stop_point_index', 'time_since_ff_last_seen'], inplace=True)
    ff_info = ff_info.groupby('stop_point_index').first().reset_index()

    # Determine reference points for curvature estimation
    ff_info = find_cvn_utils.find_ff_info_based_on_ref_point(
        ff_info, monkey_information, ff_real_position_sorted,
        ref_point_mode=ref_point_mode, ref_point_value=ref_point_value
    )
    ff_info = find_cvn_utils.add_monkey_info_before_stop(
        monkey_information, ff_info
    )

    opt_arc_stop_first_vis_bdry = (
        opt_arc_type == 'opt_arc_stop_first_vis_bdry')

    # Compute curvature features
    curv_df = curvature_utils.make_curvature_df(
        ff_info, curv_of_traj_df, clean=True,
        monkey_information=monkey_information,
        opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry
    )

    # Adjust curvature arc to stop at closest point to monkey
    if opt_arc_type == 'opt_arc_stop_closest':
        stop_and_ref = ff_info[['stop_point_index', 'point_index',
                                'ff_index', 'ff_x', 'ff_y', 'monkey_x', 'monkey_y']].copy()
        stop_and_ref[['stop_x', 'stop_y']] = monkey_information.loc[stop_and_ref['stop_point_index'], [
            'monkey_x', 'monkey_y']].values
        curv_df = opt_arc_utils.update_curvature_df_to_let_opt_arc_stop_at_closest_point_to_monkey_stop(
            curv_df, stop_and_ref, ff_real_position_sorted, monkey_information
        )

    # Merge curvature info
    shared_cols = [
        'ff_index', 'point_index', 'opt_arc_curv', 'opt_arc_measure', 'opt_arc_radius',
        'opt_arc_end_direction', 'cntr_arc_curv', 'cntr_arc_radius',
        'cntr_arc_d_heading', 'opt_arc_d_heading', 'opt_arc_end_x', 'opt_arc_end_y',
        'cntr_arc_end_x', 'cntr_arc_end_y'
    ]
    df = ff_info.merge(curv_df[shared_cols], on=[
                       'ff_index', 'point_index'], how='left')
    df = df.merge(
        curv_of_traj_df[['point_index', 'curv_of_traj']], on='point_index', how='left')

    # Add d_heading info
    df = plan_factors_utils.add_d_heading_of_traj_to_df(df)

    d_heading_var = 'cntr_arc_d_heading' if use_curv_to_ff_center else 'opt_arc_d_heading'

    df[['cur_opt_arc_d_heading', 'd_heading_of_traj']
       ] = df[[d_heading_var, 'd_heading_of_traj']]

    df['d_heading_of_traj'] = find_cvn_utils.confine_angle_to_within_one_pie(
        df['d_heading_of_traj'].values)
    df['diff_in_d_heading_to_cur_ff'] = df['d_heading_of_traj'] - \
        df['cur_opt_arc_d_heading']

    # Add time-related columns
    df['ref_time'] = monkey_information.loc[df['point_index'], 'time'].values
    df['stop_time'] = monkey_information.loc[df['stop_point_index'], 'time'].values
    df['beginning_time'] = df['stop_time'] - stop_period_duration
    df.rename(columns={'point_index': 'ref_point_index'}, inplace=True)

    # Add curvature statistics between beginning_time and stop_time
    curv_stat = build_factor_comp.find_curv_of_traj_stat_df(
        df, curv_of_traj_df, start_time_column='beginning_time', end_time_column='stop_time'
    )
    df = build_factor_comp_utils._add_stat_columns_to_df(
        curv_stat, df, ['curv'], 'stop_point_index'
    )

    # Add angle from ff to stop
    df[['stop_x', 'stop_y']] = monkey_information.loc[df['stop_point_index'], [
        'monkey_x', 'monkey_y']].values
    df['angle_from_cur_ff_to_stop'] = specific_utils.calculate_angles_to_ff_centers(
        ff_x=df['stop_x'].values, ff_y=df['stop_y'],
        mx=df['ff_x'].values, my=df['ff_y'],
        m_angle=df['monkey_angle_before_stop']
    )
    df['dir_from_cur_ff_to_stop'] = np.sign(df['angle_from_cur_ff_to_stop'])

    # Add curvature-of-trajectory before stop
    curv_window_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
        window_for_curv_of_traj, monkey_information, ff_caught_T_new,
        curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=False
    )
    df = build_factor_comp.add_column_curv_of_traj_before_stop(
        df, curv_window_df)

    return df.sort_values(by='stop_point_index').reset_index(drop=True)


def find_ff_info_and_cur_ff_info_at_start_df(only_cur_ff_df, monkey_info_in_all_stop_periods, ff_flash_sorted,
                                             ff_real_position_sorted, ff_life_sorted, ff_radius=10,
                                             dropna=False,
                                             guarantee_info_for_cur_ff=False,
                                             filter_out_ff_not_in_front_of_monkey_at_ref_point=True
                                             ):

    ff_info_at_start_df = only_cur_ff_df[['ref_point_index', 'ref_time', 'stop_point_index',
                                          'beginning_time', 'stop_time', 'monkey_x', 'monkey_y', 'monkey_angle',
                                          'ff_index', 'ff_x', 'ff_y', 'ff_angle'
                                          ]].copy()

    ff_info_at_start_df.rename(columns={'ff_index': 'cur_ff_index',
                                        'ff_x': 'cur_ff_x',
                                        'ff_y': 'cur_ff_y',
                                        'ff_angle': 'cur_ff_angle'}, inplace=True)

    monkey_info_in_all_stop_periods = monkey_info_in_all_stop_periods[monkey_info_in_all_stop_periods['stop_point_index'].isin(
        only_cur_ff_df['stop_point_index'].values)].copy()
    flash_time_info = _get_info_of_ff_whose_flash_time_overlaps_with_stop_periods(
        monkey_info_in_all_stop_periods, ff_flash_sorted, ff_life_sorted)

    if guarantee_info_for_cur_ff:
        cur_ff_info = only_cur_ff_df[['ff_index', 'stop_point_index']]
        flash_time_info = flash_time_info.merge(
            cur_ff_info, on=['stop_point_index', 'ff_index'], how='outer')

    ff_info_at_start_df = ff_info_at_start_df.merge(
        flash_time_info, on=['stop_point_index'], how='right')

    ff_info_at_start_df['ff_x'], ff_info_at_start_df['ff_y'] = ff_real_position_sorted[ff_info_at_start_df['ff_index'].values].T
    ff_info_at_start_df = _add_basic_ff_info_to_df_for_ff(
        ff_info_at_start_df, ff_radius=ff_radius)
    # add info related to flash time

    ff_info_at_start_df = furnish_ff_info_at_start_df(ff_info_at_start_df)

    if guarantee_info_for_cur_ff:
        ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'cur_ff_index', 'ff_index']] = \
            ff_info_at_start_df[[
                'ref_point_index', 'stop_point_index', 'cur_ff_index', 'ff_index']].astype('int')
    else:
        ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'cur_ff_index', 'ff_index', 'earliest_flash_point_index', 'latest_flash_point_index']] = \
            ff_info_at_start_df[['ref_point_index', 'stop_point_index', 'cur_ff_index',
                                 'ff_index', 'earliest_flash_point_index', 'latest_flash_point_index']].astype('int')

    cur_ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['ff_index']
                                                  == ff_info_at_start_df['cur_ff_index']].copy()
    # filter both df such that only when cur ff is in front of the monkey at ref point will such stop point be preserved.
    if filter_out_ff_not_in_front_of_monkey_at_ref_point:
        orig_len = len(cur_ff_info_at_start_df)
        valid_stop_periods, cur_ff_info_at_start_df = _find_valid_stop_periods_for_a_ff(
            cur_ff_info_at_start_df)
        print(f'Filtered out {orig_len - len(cur_ff_info_at_start_df)} stop periods out of {orig_len} stop periods because they are not in front of the monkey at ref point')
    ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['stop_point_index'].isin(
        cur_ff_info_at_start_df['stop_point_index'].unique())].copy()

    if dropna:
        ff_info_at_start_df.dropna(axis=0, inplace=True)
        print('Dropped NaN values in ff_info_at_start_df')

    ff_info_at_start_df.reset_index(drop=True, inplace=True)
    cur_ff_info_at_start_df.drop_duplicates(inplace=True)
    cur_ff_info_at_start_df.reset_index(drop=True, inplace=True)

    return ff_info_at_start_df, cur_ff_info_at_start_df


def furnish_ff_info_at_start_df(ff_info_at_start_df):
    ff_info_at_start_df['earliest_flash_rel_time'] = ff_info_at_start_df['earliest_flash_time'] - \
        ff_info_at_start_df['beginning_time']
    ff_info_at_start_df['latest_flash_rel_time'] = ff_info_at_start_df['latest_flash_time'] - \
        ff_info_at_start_df['beginning_time']
    ff_info_at_start_df['ff_distance_to_cur_ff'] = np.linalg.norm(
        [ff_info_at_start_df['cur_ff_x'] - ff_info_at_start_df['ff_x'], ff_info_at_start_df['cur_ff_y'] - ff_info_at_start_df['ff_y']], axis=0)

    ff_info_at_start_df['angle_diff_boundary'] = ff_info_at_start_df['ff_angle'] - \
        ff_info_at_start_df['ff_angle_boundary']
    ff_info_at_start_df['angle_diff_boundary'] = ff_info_at_start_df['angle_diff_boundary'] % (
        2*math.pi)
    ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_boundary'] > math.pi,
                            'angle_diff_boundary'] = ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_boundary'] > math.pi, 'angle_diff_boundary'] - 2*math.pi

    ff_info_at_start_df['angle_diff_from_cur_ff'] = ff_info_at_start_df['ff_angle'] - \
        ff_info_at_start_df['cur_ff_angle']
    ff_info_at_start_df['angle_diff_from_cur_ff'] = ff_info_at_start_df['angle_diff_from_cur_ff'] % (
        2*math.pi)
    ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_from_cur_ff'] > math.pi,
                            'angle_diff_from_cur_ff'] = ff_info_at_start_df.loc[ff_info_at_start_df['angle_diff_from_cur_ff'] > math.pi, 'angle_diff_from_cur_ff'] - 2*math.pi
    return ff_info_at_start_df


def find_monkey_info_in_all_stop_periods(all_start_time, all_end_time, all_segment_id, monkey_information):

    monkey_info_in_all_stop_periods = monkey_information[[
        'time', 'point_index', 'monkey_x', 'monkey_y', 'monkey_angle', 'dt']].copy()

    monkey_info_in_all_stop_periods = build_factor_comp_utils._take_out_info_of_all_segments(
        monkey_info_in_all_stop_periods, all_start_time, all_end_time, all_segment_id, group_id='stop_point_index')

    return monkey_info_in_all_stop_periods


def find_ff_flash_df_within_all_stop_periods(monkey_info_in_all_stop_periods, ff_caught_T_new, ff_real_position_sorted, ff_flash_sorted, ff_radius=10):
    ff_flash_df = pd.DataFrame()
    for i in range(len(ff_real_position_sorted)):
        if i % 100 == 0:
            print(
                f"Processing {i}-th ff out of {len(ff_real_position_sorted)} ff")
        ff_flash = ff_flash_sorted[i]
        monkey_sub_for_ff = _find_monkey_sub_within_any_flash_period_for_a_ff(
            monkey_info_in_all_stop_periods, ff_flash)
        monkey_sub_for_ff['ff_index'] = i
        monkey_sub_for_ff[['ff_x', 'ff_y']] = ff_real_position_sorted[i]
        monkey_sub_for_ff = _add_basic_ff_info_to_df_for_ff(
            monkey_sub_for_ff, ff_radius=ff_radius)
        # for each stop period, if the minimum ff_distance and abs_ff_angle_boundary is less than 800 and 45 degrees, then keep all points in that stop period
        grouped_min_info = monkey_sub_for_ff[[
            'stop_point_index', 'abs_ff_angle_boundary', 'ff_distance']].groupby('stop_point_index').min()
        grouped_min_info = grouped_min_info[(grouped_min_info['ff_distance'] < 800) & (
            grouped_min_info['abs_ff_angle_boundary'] < 45/180*math.pi)]
        if len(grouped_min_info) > 0:
            monkey_sub_for_ff_final = monkey_sub_for_ff[monkey_sub_for_ff['flash_period'].isin(
                grouped_min_info.index)].copy()
            ff_flash_df = pd.concat(
                [ff_flash_df, monkey_sub_for_ff_final], axis=0)
    ff_flash_df.reset_index(drop=True, inplace=True)
    return ff_flash_df


def _get_info_of_ff_whose_flash_time_overlaps_with_stop_periods(monkey_info_in_all_stop_periods, ff_flash_sorted, ff_life_sorted):
    time = monkey_info_in_all_stop_periods['time'].values
    all_in_flash_iloc = []
    all_ff_index = []
    for i in range(len(ff_life_sorted)):
        ff_flash = ff_flash_sorted[i]
        in_flash_iloc, _ = _find_index_of_points_within_flash_period_for_a_ff(
            time, ff_flash)
        all_in_flash_iloc.extend(in_flash_iloc.tolist())
        all_ff_index.extend([i] * len(in_flash_iloc))
    monkey_sub_for_ff = monkey_info_in_all_stop_periods[[
        'point_index', 'time', 'dt', 'stop_point_index']].iloc[all_in_flash_iloc].copy()
    monkey_sub_for_ff['ff_index'] = all_ff_index
    flash_time_info = monkey_sub_for_ff.groupby(['ff_index', 'stop_point_index']).agg(earliest_flash_point_index=('point_index', 'min'),
                                                                                      latest_flash_point_index=(
                                                                                          'point_index', 'max'),
                                                                                      earliest_flash_time=(
                                                                                          'time', 'min'),
                                                                                      latest_flash_time=(
                                                                                          'time', 'max'),
                                                                                      flash_duration=('dt', 'sum'))
    flash_time_info.reset_index(drop=False, inplace=True)
    return flash_time_info


def _find_valid_stop_periods_for_a_ff(only_cur_ff_df_sub):
    only_cur_ff_df_sub = only_cur_ff_df_sub.copy()
    # valid_stop_periods_df = stop_sub_for_ff.copy()
    valid_stop_periods_df = only_cur_ff_df_sub[(only_cur_ff_df_sub['ff_distance'] < 1000) &
                                               (only_cur_ff_df_sub['abs_ff_angle_boundary'] < 90/180*math.pi)].copy()
    valid_stop_periods = valid_stop_periods_df['stop_point_index'].unique()
    return valid_stop_periods, valid_stop_periods_df


def _find_index_of_points_within_flash_period_for_a_ff(time, ff_flash):
    point_corr_position_in_flash = np.searchsorted(ff_flash.flatten(), time)
    in_flash_iloc = np.where(point_corr_position_in_flash % 2 == 1)[0]
    return in_flash_iloc, point_corr_position_in_flash


def _find_monkey_sub_within_any_flash_period_for_a_ff(monkey_info_in_all_stop_periods, ff_flash):
    in_flash_iloc, point_corr_position_in_flash = _find_index_of_points_within_flash_period_for_a_ff(
        monkey_info_in_all_stop_periods['time'].values, ff_flash)
    monkey_sub_for_ff = monkey_info_in_all_stop_periods.iloc[in_flash_iloc].copy(
    )
    monkey_sub_for_ff['flash_period'] = (
        point_corr_position_in_flash[in_flash_iloc]/2).astype(int)
    return monkey_sub_for_ff


def _add_basic_ff_info_to_df_for_ff(df, ff_radius=10):
    # For the selected time points, find ff distance and angle to boundary

    df['ff_distance'] = np.sqrt(
        (df['ff_x'] - df['monkey_x'])**2 + (df['ff_y'] - df['monkey_y'])**2)
    df['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(
        df['ff_x'], df['ff_y'], mx=df['monkey_x'], my=df['monkey_y'], m_angle=df['monkey_angle'])
    df['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=df['ff_angle'], distances_to_ff=df['ff_distance'], ff_radius=ff_radius)
    df['abs_ff_angle_boundary'] = np.abs(df['ff_angle_boundary'])
    return df


def get_x_features_df(ff_info_at_start_df, cur_ff_info_at_start_df,
                      columns_not_to_include=[],
                      rank_columns_not_to_include=[],
                      flash_or_vis='flash',
                      list_of_cur_ff_cluster_radius=[100, 200, 300],
                      list_of_cur_ff_ang_cluster_radius=[20],
                      list_of_start_dist_cluster_radius=[100, 200, 300],
                      list_of_start_ang_cluster_radius=[20],
                      list_of_flash_cluster_period=[[1.0, 1.5], [1.5, 2.0]],
                      ):

    all_cluster_info = pd.DataFrame()
    all_cluster_info = cur_ff_info_at_start_df[['ff_distance', 'ff_angle', 'ff_angle_boundary', 'angle_diff_boundary', 'flash_duration',
                                                'earliest_flash_rel_time', 'latest_flash_rel_time']].copy()
    # # rename all the columns in all_cluster_info_in_a_row to add a prefix 'cur_ff' and a suffix 'at_ref_point'
    all_cluster_info.columns = ['cur_ff_' + col + '_at_ref' if (
        col[:3] != 'ff_') else 'cur_' + col + '_at_ref' for col in all_cluster_info.columns]
    all_cluster_info.reset_index(drop=True, inplace=True)
    all_cluster_info[['stop_point_index', 'stop_point_index']] = cur_ff_info_at_start_df[[
        'stop_point_index', 'stop_point_index']].values

    ff_info_at_start_df, all_cluster_names = find_clusters_in_ff_info_at_start_df(ff_info_at_start_df, cur_ff_info_at_start_df,
                                                                                  list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius,
                                                                                  list_of_cur_ff_ang_cluster_radius=list_of_cur_ff_ang_cluster_radius,
                                                                                  list_of_start_dist_cluster_radius=list_of_start_dist_cluster_radius,
                                                                                  list_of_start_ang_cluster_radius=list_of_start_ang_cluster_radius,
                                                                                  list_of_flash_cluster_period=list_of_flash_cluster_period,
                                                                                  )
    ff_info_at_start_df = get_ranks_of_columns_to_include_within_each_stop_period(ff_info_at_start_df, rank_columns_not_to_include=rank_columns_not_to_include,
                                                                                  flash_or_vis=flash_or_vis)

    cluster_factors_df, cluster_agg_df = get_cluster_and_agg_df(ff_info_at_start_df, all_cluster_names,
                                                                columns_not_to_include=columns_not_to_include, rank_columns_not_to_include=rank_columns_not_to_include,
                                                                flash_or_vis=flash_or_vis)

    all_cluster_info = all_cluster_info.merge(
        cluster_factors_df, on='stop_point_index', how='outer')
    all_cluster_info = all_cluster_info.merge(
        cluster_agg_df, on='stop_point_index', how='outer')
    x_features_df = all_cluster_info.reset_index(drop=True)
    return x_features_df, all_cluster_names


def get_cluster_and_agg_df(ff_info_at_start_df, all_cluster_names,
                           flash_or_vis='flash',
                           columns_not_to_include=[],
                           rank_columns_not_to_include=[],
                           ):

    columns_to_include = ['ff_distance', 'ff_angle', 'angle_diff_boundary']
    rank_columns_to_include = ['abs_ff_angle_boundary', 'ff_distance']

    if flash_or_vis is not None:
        additional_columns = [f'earliest_{flash_or_vis}_rel_time',
                              f'latest_{flash_or_vis}_rel_time', f'{flash_or_vis}_duration']
        columns_to_include.extend(additional_columns)
        rank_columns_to_include.extend(additional_columns)

    columns_to_include = [
        column for column in columns_to_include if column not in columns_not_to_include]
    rank_columns_to_include = [
        column for column in rank_columns_to_include if column not in rank_columns_not_to_include]

    id_columns = [
        column for column in ff_info_at_start_df.columns if column not in all_cluster_names]
    ff_info_at_start_df_melted = pd.melt(ff_info_at_start_df, id_vars=id_columns,
                                         value_vars=all_cluster_names, var_name='group', value_name='whether_ff_selected')
    ff_info_at_start_df_melted = ff_info_at_start_df_melted[
        ff_info_at_start_df_melted['whether_ff_selected'] == True]

    columns_to_include = copy.deepcopy(columns_to_include)
    columns_to_include.extend(
        [col + '_rank' for col in rank_columns_to_include])
    cluster_factors_df = _get_cluster_factors_df(
        ff_info_at_start_df_melted, columns_not_to_include=columns_not_to_include, flash_or_vis=flash_or_vis)

    cluster_factors_df['group'] = cluster_factors_df['group'] + \
        '_' + cluster_factors_df['ff'].str.upper()
    cluster_factors_df = cluster_factors_df.sort_values(
        by=['group', 'stop_point_index']).reset_index(drop=True).drop(columns=['ff'])
    cluster_factors_df = slice_and_combd_a_df(cluster_factors_df)

    cluster_agg_df = _get_cluster_agg_df(
        ff_info_at_start_df_melted, flash_or_vis=flash_or_vis)
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

    columns_to_include = ['ff_distance',
                          'ff_angle',
                          'ff_angle_boundary',
                          'angle_diff_boundary']
    if flash_or_vis is not None:
        columns_to_include.extend([f'earliest_{flash_or_vis}_rel_time',
                                   f'latest_{flash_or_vis}_rel_time',
                                   f'{flash_or_vis}_duration'])
    columns_to_include = [
        column for column in columns_to_include if column not in columns_not_to_include]

    cluster_factors_df = pd.DataFrame()

    columns_to_get_max_dict = {'ff_angle': 'leftmost'}
    columns_to_get_min_dict = {'ff_angle': 'rightmost'}

    if flash_or_vis is not None:
        columns_to_get_max_dict.update({f'latest_{flash_or_vis}_rel_time': f'latest_{flash_or_vis}',
                                        f'{flash_or_vis}_duration': f'longest_{flash_or_vis}'})

        columns_to_get_max_dict.update(
            {f'earliest_{flash_or_vis}_rel_time': f'earliest_{flash_or_vis}'})

    columns_to_get_max = list(columns_to_get_max_dict.keys())
    columns_to_get_max = [column for column in columns_to_get_max if (
        column not in columns_not_to_include)]
    for column in columns_to_get_max:
        ff = columns_to_get_max_dict[column]
        max_id = ff_info_at_start_df_melted.groupby(
            ['stop_point_index', 'group'])[column].idxmax()
        rows_to_be_added = ff_info_at_start_df_melted.loc[max_id].copy()
        rows_to_be_added['ff'] = ff
        cluster_factors_df = pd.concat(
            [cluster_factors_df, rows_to_be_added], axis=0)

    columns_to_get_min = list(columns_to_get_min_dict.keys())
    columns_to_get_min = [column for column in columns_to_get_min if (
        column not in columns_not_to_include)]
    for column in columns_to_get_min:
        ff = columns_to_get_min_dict[column]
        max_id = ff_info_at_start_df_melted.groupby(
            ['stop_point_index', 'group'])[column].idxmin()
        rows_to_be_added = ff_info_at_start_df_melted.loc[max_id].copy()
        rows_to_be_added['ff'] = ff
        cluster_factors_df = pd.concat(
            [cluster_factors_df, rows_to_be_added], axis=0)

    cluster_factors_df = cluster_factors_df[columns_to_include + [
        'stop_point_index', 'group', 'ff']]

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

    cluster_agg_df = ff_info_at_start_df_melted.groupby(
        ['stop_point_index', 'group']).agg(agg_dict)

    # Flatten the MultiIndex columns
    cluster_agg_df.columns = [col[1] if col[1] != '' else col[0]
                              for col in cluster_agg_df.columns.values]
    cluster_agg_df.reset_index(drop=False, inplace=True)
    cluster_agg_df = cluster_agg_df.sort_values(
        by=['group', 'stop_point_index']).reset_index(drop=True)
    return cluster_agg_df


def get_ranks_of_columns_to_include_within_each_stop_period(ff_info_at_start_df, rank_columns_not_to_include=[], flash_or_vis='flash'):

    rank_columns_to_include = ['abs_ff_angle_boundary', 'ff_distance']
    if flash_or_vis is not None:
        rank_columns_to_include.extend([f'earliest_{flash_or_vis}_rel_time', f'latest_{flash_or_vis}_rel_time',
                                        f'{flash_or_vis}_duration'])
    rank_columns_to_include = [
        column for column in rank_columns_to_include if column not in rank_columns_not_to_include]

    ranked_columns = ff_info_at_start_df.groupby('stop_point_index')[
        rank_columns_to_include].rank(method='average', ascending=True)
    ranked_columns.rename(
        columns={col: col + '_rank' for col in rank_columns_to_include}, inplace=True)
    ff_info_at_start_df = pd.concat(
        [ff_info_at_start_df, ranked_columns], axis=1)
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


def find_clusters_in_ff_info_at_start_df(ff_info_at_start_df, cur_ff_info_at_start_df,
                                         list_of_cur_ff_cluster_radius=[
                                             100, 200, 300],
                                         list_of_cur_ff_ang_cluster_radius=[
                                             20],
                                         list_of_start_dist_cluster_radius=[
                                             100, 200, 300],
                                         list_of_start_ang_cluster_radius=[20],
                                         list_of_flash_cluster_period=[
                                             [1.0, 1.5], [1.5, 2.0]],
                                         ):

    # first, for stop_period in ff_info_at_start_df that has no info, we'll use the info in cur_ff_info_at_start_df
    cur_ff_info_at_start_df_to_add = cur_ff_info_at_start_df[~cur_ff_info_at_start_df['stop_point_index'].isin(
        ff_info_at_start_df['stop_point_index'].values)]
    ff_info_at_start_df = pd.concat(
        [ff_info_at_start_df, cur_ff_info_at_start_df_to_add], axis=0)

    cur_ff_info_at_start_df['cur_ff_distance'] = cur_ff_info_at_start_df['ff_distance'].values
    ff_info_at_start_df = ff_info_at_start_df.merge(cur_ff_info_at_start_df[[
                                                    'stop_point_index', 'cur_ff_distance']], on='stop_point_index', how='left')
    # only preserve ff that has equal or greater ff_distance than cur_ff_distance in the same period
    # ff_info_at_start_df = ff_info_at_start_df[ff_info_at_start_df['ff_distance'] >= ff_info_at_start_df['cur_ff_distance']].copy()

    all_cluster_names = []
    for n_cm in list_of_cur_ff_cluster_radius:
        column = f'cur_ff_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['ff_distance_to_cur_ff']
                                <= n_cm, column] = True

    # cur ff cluster based on ff angle's proclicivity to cur ff's angle
    for n_angle in list_of_cur_ff_ang_cluster_radius:
        column = f'cur_ff_ang_cluster_{n_angle}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[np.abs(
            ff_info_at_start_df['angle_diff_from_cur_ff']) <= n_angle, column] = True

    # Cluster based on ff_distance at start point
    for n_cm in list_of_start_dist_cluster_radius:
        column = f'start_dist_cluster_{n_cm}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['ff_distance']
                                <= n_cm, column] = True

    # Cluster based on ff_angle_boundary at start point
    for n_angle in list_of_start_ang_cluster_radius:
        column = f'start_ang_cluster_{n_angle}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[np.abs(
            ff_info_at_start_df['ff_angle_boundary']*180/math.pi) <= n_angle, column] = True

    # Cluster based on ff that have flashed in the last 0.5s before stop
    for period in list_of_flash_cluster_period:
        # make n_s into string and replace . with _
        period_str = str(period[0]).replace('.', '_') + \
            '_to_' + str(period[1]).replace('.', '_')
        column = f'flash_cluster_{period_str}'
        all_cluster_names.append(column)
        ff_info_at_start_df[column] = False
        ff_info_at_start_df.loc[ff_info_at_start_df['latest_flash_rel_time'].between(
            period[0], period[1]), column] = True

    ff_info_at_start_df = _supply_zero_size_cluster_with_cur_ff_info(
        ff_info_at_start_df, cur_ff_info_at_start_df, all_cluster_names)

    return ff_info_at_start_df, all_cluster_names


def _supply_zero_size_cluster_with_cur_ff_info(ff_info_at_start_df, cur_ff_info_at_start_df, all_cluster_names):
    whether_each_cluster_has_enough = ff_info_at_start_df[all_cluster_names + [
        'stop_point_index']].groupby('stop_point_index').sum() == 0
    where_need_cur_ff = np.where(whether_each_cluster_has_enough)
    stop_periods = whether_each_cluster_has_enough.index.values[where_need_cur_ff[0]]
    groups = np.array(all_cluster_names)[where_need_cur_ff[1]]
    cur_ff_rows_to_add = cur_ff_info_at_start_df.set_index(
        'stop_point_index').loc[stop_periods].reset_index(drop=False)
    cur_ff_rows_to_add['group'] = groups
    cur_ff_rows_to_add = pd.get_dummies(
        cur_ff_rows_to_add, columns=['group'], prefix='prefix')
    # drop all 'prefix_' from the column names
    cur_ff_rows_to_add.columns = [col.split(
        'prefix_')[1] if 'prefix_' in col else col for col in cur_ff_rows_to_add.columns]
    ff_info_at_start_df = pd.concat(
        [ff_info_at_start_df, cur_ff_rows_to_add], axis=0)
    return ff_info_at_start_df


def make_monkey_info_in_all_stop_periods(closest_stop_to_capture_df, monkey_information, stop_period_duration=2,
                                         all_end_time=None, all_start_time=None):
    if all_end_time is None:
        all_end_time = closest_stop_to_capture_df['time'].values
    if all_start_time is None:
        all_start_time = closest_stop_to_capture_df['time'].values - \
            stop_period_duration
    all_segment_id = closest_stop_to_capture_df['stop_point_index'].values
    monkey_info_in_all_stop_periods = find_monkey_info_in_all_stop_periods(
        all_start_time, all_end_time, all_segment_id, monkey_information)
    if 'stop_time' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df['stop_time'] = closest_stop_to_capture_df['time']
    if 'beginning_time' not in closest_stop_to_capture_df.columns:
        closest_stop_to_capture_df['beginning_time'] = closest_stop_to_capture_df['time'] - \
            stop_period_duration
    monkey_info_in_all_stop_periods = monkey_info_in_all_stop_periods.merge(closest_stop_to_capture_df[[
                                                                            'stop_point_index', 'beginning_time', 'stop_time']], on='stop_point_index', how='left')

    return monkey_info_in_all_stop_periods


def keep_same_data_name_and_stop_point_pairs(only_cur_ff_df_for_ml, x_features_df_for_ml):
    # Make sure that the two df share the same set of data_name + stop period pairs

    # First create a set of data_name + stop_point_index pairs for both DataFrames
    only_cur_ff_df_for_ml['data_name_stop_point_pair'] = only_cur_ff_df_for_ml['data_name'] + \
        '_' + only_cur_ff_df_for_ml['stop_point_index'].astype(str)
    x_features_df_for_ml['data_name_stop_point_pair'] = x_features_df_for_ml['data_name'] + \
        '_' + x_features_df_for_ml['stop_point_index'].astype(str)

    # Find the intersection of both sets
    shared_stop_point_pairs = set(only_cur_ff_df_for_ml['data_name_stop_point_pair']).intersection(
        set(x_features_df_for_ml['data_name_stop_point_pair']))

    # Filter both DataFrames to retain only the shared stop point pairs
    only_cur_ff_df_for_ml = only_cur_ff_df_for_ml[only_cur_ff_df_for_ml['data_name_stop_point_pair'].isin(
        shared_stop_point_pairs)].copy()

    x_features_df_for_ml = x_features_df_for_ml[x_features_df_for_ml['data_name_stop_point_pair'].isin(
        shared_stop_point_pairs)].copy()

    # drop the column
    only_cur_ff_df_for_ml.drop(
        columns=['data_name_stop_point_pair'], inplace=True)
    x_features_df_for_ml.drop(
        columns=['data_name_stop_point_pair'], inplace=True)

    return only_cur_ff_df_for_ml, x_features_df_for_ml
