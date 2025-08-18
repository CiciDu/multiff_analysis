from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.plan_factors import test_vs_control_utils
from planning_analysis.plan_factors import build_factor_comp_utils, build_factor_comp, feature_lists
import pandas as pd


def make_plan_features1(heading_info_df, curv_of_traj_df, curv_of_traj_df_w_one_sided_window):

    plan_features1 = build_factor_comp.process_heading_info_df(
        heading_info_df)

    curv_of_traj_stat_df = build_factor_comp.find_curv_of_traj_stat_df(
        heading_info_df, curv_of_traj_df)
    plan_features1 = build_factor_comp_utils._add_stat_columns_to_df(
        curv_of_traj_stat_df, plan_features1, ['curv'], 'stop_point_index')

    if curv_of_traj_df_w_one_sided_window is not None:
        plan_features1 = build_factor_comp.add_column_curv_of_traj_before_stop(
            plan_features1, curv_of_traj_df_w_one_sided_window)

    build_factor_comp.add_dir_from_cur_ff_same_side(plan_features1)

    plan_features1 = plan_features1.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_features1


def make_plan_features2(stops_near_ff_df, heading_info_df, both_ff_at_ref_df, ff_dataframe, monkey_information, ff_real_position_sorted,
                        stop_period_duration=2, ref_point_mode='distance', ref_point_value=-150, ff_radius=10,
                        list_of_cur_ff_cluster_radius=[100, 200, 300],
                        list_of_nxt_ff_cluster_radius=[100, 200, 300],
                        use_speed_data=True, use_eye_data=True,
                        guarantee_cur_ff_info_for_cluster=False,
                        guarantee_nxt_ff_info_for_cluster=False,
                        columns_not_to_include=[],
                        flash_or_vis='vis',
                        ):

    plan_features2 = both_ff_at_ref_df.copy()

    # get information between two stops
    info_between_two_stops = build_factor_comp.get_info_between_two_stops(
        stops_near_ff_df)
    plan_features2 = plan_features2.merge(
        info_between_two_stops, on='stop_point_index', how='left')

    # get distance_between_stop_and_arena_edge
    plan_features2['distance_between_stop_and_arena_edge'] = build_factor_comp.get_distance_between_stop_and_arena_edge(
        stops_near_ff_df)

    # Get cluster information for df. Example output columns: cur_ff_cluster_100_EARLIEST_VIS_ff_distance
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    cluster_df = build_factor_comp.make_cluster_df_as_part_of_plan_factors(stops_near_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted,
                                                                           stop_period_duration=stop_period_duration, ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, ff_radius=ff_radius,
                                                                           list_of_cur_ff_cluster_radius=list_of_cur_ff_cluster_radius, list_of_nxt_ff_cluster_radius=list_of_nxt_ff_cluster_radius,
                                                                           columns_not_to_include=columns_not_to_include,
                                                                           guarantee_cur_ff_info_for_cluster=guarantee_cur_ff_info_for_cluster,
                                                                           guarantee_nxt_ff_info_for_cluster=guarantee_nxt_ff_info_for_cluster,
                                                                           flash_or_vis=flash_or_vis,
                                                                           )
    plan_features2 = plan_features2.merge(
        cluster_df, on='stop_point_index', how='left').reset_index(drop=True)

    # nxt_ff_last_seen_info = build_factor_comp.get_nxt_ff_last_seen_info_before_next_stop(nxt_ff_df_from_ref, ff_dataframe_visible, monkey_information,
    #                                                                     stops_near_ff_df, ff_real_position_sorted)
    # plan_features2 = pd.concat([plan_features2, nxt_ff_last_seen_info])

    if use_speed_data:
        plan_features2 = build_factor_comp.add_monkey_speed_stats_to_df(
            plan_features2, stops_near_ff_df, monkey_information)

    if use_eye_data:
        plan_features2 = build_factor_comp.add_monkey_eye_stats_to_df(
            plan_features2, stops_near_ff_df, monkey_information)

    # only keep the rows with stop_point_index that are in heading_info_df
    plan_features2 = plan_features2[plan_features2['stop_point_index'].isin(
        heading_info_df['stop_point_index'])].copy()
    plan_features2 = plan_features2.sort_values(
        by='stop_point_index').reset_index(drop=True)

    return plan_features2


def drop_columns_that_contain_both_nxt_and_bbas(plan_features_tc):
    # Drop columns in self.plan_features_tc that contain both 'nxt'/'NXT' and 'bbas'
    columns_to_drop = [
        col for col in plan_features_tc.columns
        if 'bbas' in col.lower() and ('nxt' in col.lower())
    ]

    if columns_to_drop:
        plan_features_tc.drop(columns=columns_to_drop, inplace=True)
        print(
            f"Dropped {len(columns_to_drop)} columns containing both 'nxt'/'NXT' and 'bbas': {columns_to_drop}")


def select_planning_features_to_predict_monkey_info(plan_features_tc, for_classification=False):
    df = plan_features_tc.copy()

    # Safely create curvature summary columns if sources exist
    if {'curv_max', 'curv_min'}.issubset(df.columns):
        df['curv_range'] = df['curv_max'] - df['curv_min']
    if {'curv_Q3', 'curv_Q1'}.issubset(df.columns):
        df['curv_iqr'] = df['curv_Q3'] - df['curv_Q1']

    non_cluster_columns_to_save = (
        ['distance_between_stop_and_arena_edge']
        + feature_lists.cur_ff_at_ref_features
        + feature_lists.nxt_ff_at_ref_features
        + feature_lists.all_eye_features
        + feature_lists.trajectory_features
    )

    # If you want the curvature summaries, include them explicitly when present
    for maybe_col in ('curv_range', 'curv_iqr'):
        if maybe_col in df.columns:
            non_cluster_columns_to_save.append(maybe_col)

    cluster_columns_to_save = [
        col for col in df.columns
        if (col in non_cluster_columns_to_save) or ('cluster' in col)
    ]

    # Optionally add classification-specific feature, if present
    extra_classif_cols = []
    if for_classification and 'dir_from_cur_ff_to_nxt_ff' in df.columns:
        extra_classif_cols.append('dir_from_cur_ff_to_nxt_ff')

    # Deduplicate while preserving order
    selected_features = list(dict.fromkeys(
        non_cluster_columns_to_save + cluster_columns_to_save + extra_classif_cols
    ))

    # only retain columns actually in plan_features_tc
    selected_features = [col for col in selected_features if col in plan_features_tc.columns]

    return selected_features


def select_planning_features_to_predict_ff_info(plan_features_tc):
    non_cluster_columns_to_save = (
        ['distance_between_stop_and_arena_edge']
        + feature_lists.cur_ff_at_ref_features
        + feature_lists.nxt_ff_at_ref_features
        + feature_lists.all_eye_features
        + feature_lists.trajectory_features
        + feature_lists.traj_to_cur_ff_features
    )

    # remove if present
    for col in ['curv_range', 'curv_iqr']:
        if col in non_cluster_columns_to_save:
            non_cluster_columns_to_save.remove(col)

    cluster_columns_to_save = [
        col for col in plan_features_tc.columns
        if (col in non_cluster_columns_to_save) or ('cluster' in col)
    ]
    # remove any with 'nxt'
    cluster_columns_to_save = [
        col for col in cluster_columns_to_save if 'nxt' not in col]

    more_columns_to_add = [
        'angle_from_cur_ff_to_stop',
        'diff_in_d_heading_to_cur_ff',
        'curv_of_traj_before_stop',
        'dir_from_cur_ff_to_stop'
    ]

    # Keep order & remove duplicates
    selected_features = list(dict.fromkeys(
        non_cluster_columns_to_save + cluster_columns_to_save + more_columns_to_add
    ))
    
    # only retain columns actually in plan_features_tc
    selected_features = [col for col in selected_features if col in plan_features_tc.columns]

    return selected_features


def delete_monkey_info_in_features(features):
    columns_to_drop = feature_lists.all_eye_features + \
        feature_lists.trajectory_features + feature_lists.traj_to_cur_ff_features
    # delete 'curv_range'
    columns_to_drop.remove('curv_range')
    remaining_features = [col for col in features.columns if col not in columns_to_drop]
    return remaining_features


def make_plan_features_test_and_plan_features_ctrl(plan_features_tc):
    plan_features = plan_features_tc.copy()
    # drop duplicated columns
    plan_features = plan_features.loc[:, ~plan_features.columns.duplicated()]

    plan_features_test = plan_features[plan_features['whether_test']
                                       == 1].reset_index(drop=True).copy()
    plan_features_ctrl = plan_features[plan_features['whether_test']
                                       == 0].reset_index(drop=True).copy()
    return plan_features_test, plan_features_ctrl


def quickly_process_plan_features_test_and_ctrl(plan_features_test, plan_features_ctrl, column_for_split, whether_filter_info, finalized_params):
    test_and_ctrl_df = pd.concat(
        [plan_features_test, plan_features_ctrl], axis=0)
    ctrl_df = test_and_ctrl_df[test_and_ctrl_df[column_for_split].isnull()].copy(
    )
    test_df = test_and_ctrl_df[~test_and_ctrl_df[column_for_split].isnull()].copy(
    )

    if whether_filter_info:
        test_df, ctrl_df = test_vs_control_utils.filter_both_df(
            test_df, ctrl_df, **finalized_params)

    return test_df, ctrl_df


def add_d_heading_of_traj_to_df(df):
    df['d_heading_of_traj'] = df['monkey_angle_before_stop'] - df['monkey_angle']
    df['d_heading_of_traj'] = find_cvn_utils.confine_angle_to_within_one_pie(
        df['d_heading_of_traj'].values)
    return df

def merge_plan_features1_and_plan_features2(plan_features1, plan_features2, index_col='stop_point_index'):
    # Normalize keys to a list
    keys = [index_col] if isinstance(index_col, str) else list(index_col)

    # Uniqueness checks on the key(s)
    if plan_features1.duplicated(subset=keys).any():
        raise ValueError(f'{keys} contain duplicates in plan_features1')
    if plan_features2.duplicated(subset=keys).any():
        raise ValueError(f'{keys} contain duplicates in plan_features2')

    # Take all columns from left; from right take only keys + non-overlapping columns
    right_cols = keys + [c for c in plan_features2.columns
                         if c not in keys and c not in plan_features1.columns]

    merged = plan_features1.merge(
        plan_features2[right_cols],
        on=keys,
        how='left',
        validate='one_to_one'  # since you asserted uniqueness
    )
    return merged


def select_planning_features_for_modeling(plan_features_df, to_predict_ff=False, for_classification=False):
    if to_predict_ff:
        selected_features = select_planning_features_to_predict_ff_info(
            plan_features_df)
    else:
        selected_features = select_planning_features_to_predict_monkey_info(
            plan_features_df, for_classification=for_classification)
    return selected_features
