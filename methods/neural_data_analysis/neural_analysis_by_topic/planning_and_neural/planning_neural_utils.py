import sys


def add_curv_info(info_to_add, curv_df, which_ff_info):
    curv_df = curv_df.copy()
    columns_to_rename = {'ff_index': f'{which_ff_info}ff_index',
                         #  'ff_angle': f'{which_ff_info}ff_angle',
                         #  'ff_distance': f'{which_ff_info}ff_distance',
                         'curv_to_ff_center': f'{which_ff_info}arc_curv',
                         'optimal_curvature': f'{which_ff_info}opt_arc_curv',
                         'opt_arc_d_heading': f'{which_ff_info}opt_arc_dheading', }

    curv_df.rename(columns=columns_to_rename, inplace=True)

    columns_added = list(columns_to_rename.values())
    # delete f'{which_ff_info}ff_index' from columns_added
    columns_added.remove(f'{which_ff_info}ff_index')

    curv_df_sub = curv_df[columns_added +
                          [f'{which_ff_info}ff_index', 'point_index']].drop_duplicates()

    info_to_add.drop(columns=columns_added, inplace=True, errors='ignore')
    info_to_add = info_to_add.merge(
        curv_df_sub, on=['point_index', f'{which_ff_info}ff_index'], how='left')

    return info_to_add, columns_added


def add_to_both_ff_when_seen_df(both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df):
    curv_df = curv_df.set_index('stop_point_index')
    both_ff_when_seen_df[f'{which_ff_info}ff_angle_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_angle']
    both_ff_when_seen_df[f'{which_ff_info}ff_distance_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_distance']
    # both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_to_ff_center']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['optimal_curvature']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_d_heading']
    both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[
        f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_of_traj']
