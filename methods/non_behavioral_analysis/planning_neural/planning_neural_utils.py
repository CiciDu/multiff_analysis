import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')


def add_curv_info_to_info_to_add(info_to_add, curv_df, which_ff_info):
    curv_df = curv_df.copy()
    columns_to_rename = {'ff_angle': f'{which_ff_info}ff_angle',
                            'ff_distance': f'{which_ff_info}ff_distance',
                            'curv_to_ff_center': f'{which_ff_info}arc_curv',
                            'optimal_curvature': f'{which_ff_info}opt_arc_curv',
                            'optimal_arc_d_heading': f'{which_ff_info}opt_arc_dheading',}

    curv_df.rename(columns=columns_to_rename, inplace=True)

    columns_added = list(columns_to_rename.values())
    
    info_to_add[columns_added] = curv_df[columns_added]

    return info_to_add, columns_added


def add_to_both_ff_when_seen_df(both_ff_when_seen_df, which_ff_info, when_which_ff, first_or_last, curv_df, ff_df):
    curv_df.set_index('stop_point_index', inplace=True)
    both_ff_when_seen_df[f'{which_ff_info}ff_angle_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_angle']
    both_ff_when_seen_df[f'{which_ff_info}ff_distance_{when_which_ff}_{first_or_last}_seen'] = curv_df['ff_distance']
    # both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_to_ff_center']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['optimal_curvature']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['optimal_arc_d_heading']
    both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curvature_of_traj']
    return both_ff_when_seen_df
