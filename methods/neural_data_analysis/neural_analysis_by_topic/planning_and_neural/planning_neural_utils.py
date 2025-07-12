import sys
from data_wrangling import process_monkey_information, specific_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling
import numpy as np
import pandas as pd


def add_curv_info(info_to_add, curv_df, which_ff_info):
    curv_df = curv_df.copy()
    columns_to_rename = {'ff_index': f'{which_ff_info}ff_index',
                         'cntr_arc_curv': f'{which_ff_info}cntr_arc_curv',
                         'opt_arc_curv': f'{which_ff_info}opt_arc_curv',
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
    # both_ff_when_seen_df[f'{which_ff_info}arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['cntr_arc_curv']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_curv']
    # both_ff_when_seen_df[f'{which_ff_info}opt_arc_dheading_{when_which_ff}_{first_or_last}_seen'] = curv_df['opt_arc_d_heading']
    both_ff_when_seen_df[f'time_{when_which_ff}_{first_or_last}_seen_rel_to_stop'] = ff_df[
        f'time_ff_{first_or_last}_seen'].values - ff_df['stop_time'].values
    both_ff_when_seen_df[f'traj_curv_{when_which_ff}_{first_or_last}_seen'] = curv_df['curv_of_traj']


def add_angle_from_cur_arc_end_to_nxt_ff(df, cur_curv_df, nxt_curv_df):
    angle_df = get_angle_from_cur_arc_end_to_nxt_ff(
        cur_curv_df, nxt_curv_df)
    df = df.merge(angle_df[['point_index', 'cur_opt_arc_end_heading', 'cur_cntr_arc_end_heading', 'angle_opt_arc_from_cur_end_to_nxt',
                            'angle_cntr_arc_from_cur_end_to_nxt']], on='point_index', how='left')
    return df


def get_angle_from_cur_arc_end_to_nxt_ff(cur_curv_df, nxt_curv_df):
    # add 'cur_' to all columns in self.cur_curv_df except 'point_index'
    cur_curv_df.columns = ['cur_' + col if col !=
                           'point_index' else col for col in cur_curv_df.columns]
    # add 'nxt_' to all columns in self.nxt_curv_df except 'point_index'
    nxt_curv_df.columns = ['nxt_' + col if col !=
                           'point_index' else col for col in nxt_curv_df.columns]

    both_curv_df = cur_curv_df.merge(nxt_curv_df, on='point_index', how='left')

    both_curv_df['cur_opt_arc_end_heading'] = both_curv_df['cur_monkey_angle'] + \
        both_curv_df['cur_opt_arc_d_heading']
    both_curv_df['cur_cntr_arc_end_heading'] = both_curv_df['cur_monkey_angle'] + \
        both_curv_df['cur_cntr_arc_d_heading']

    both_curv_df['angle_opt_arc_from_cur_end_to_nxt'] = specific_utils.calculate_angles_to_ff_centers(
        both_curv_df['nxt_ff_x'], both_curv_df['nxt_ff_y'], both_curv_df['cur_opt_arc_end_x'], both_curv_df['cur_opt_arc_end_y'], both_curv_df['cur_opt_arc_end_heading'])
    both_curv_df['angle_cntr_arc_from_cur_end_to_nxt'] = specific_utils.calculate_angles_to_ff_centers(
        both_curv_df['nxt_ff_x'], both_curv_df['nxt_ff_y'], both_curv_df['cur_cntr_arc_end_x'], both_curv_df['cur_cntr_arc_end_y'], both_curv_df['cur_cntr_arc_end_heading'])

    return both_curv_df


def compute_overlap_and_drop(df1, col1, df2, col2):
    """
    Computes percentage of overlapped values in df1[col1] and df2[col2],
    prints the percentages, and returns new DataFrames with the overlapping
    rows dropped.

    Args:
        df1 (pd.DataFrame): First DataFrame
        col1 (str): Column name in df1 to check for overlap
        df2 (pd.DataFrame): Second DataFrame
        col2 (str): Column name in df2 to check for overlap

    Returns:
        df1_filtered (pd.DataFrame): df1 with overlapping rows dropped
        df2_filtered (pd.DataFrame): df2 with overlapping rows dropped
    """
    a = df1[col1].values
    b = df2[col2].values

    overlap = np.intersect1d(a, b)

    percentage_a = len(overlap) / len(a) * 100 if len(a) > 0 else 0
    percentage_b = len(overlap) / len(b) * 100 if len(b) > 0 else 0
    percentage_avg = len(overlap) / ((len(a) + len(b)) / 2) * \
        100 if (len(a) + len(b)) > 0 else 0

    print(f"Overlap: {overlap}")
    print(f"Percentage overlap relative to df1: {percentage_a:.2f}%")
    print(f"Percentage overlap relative to df2: {percentage_b:.2f}%")
    print(f"Average percentage overlap: {percentage_avg:.2f}%")

    df1_filtered = df1[~df1[col1].isin(overlap)].copy()
    df2_filtered = df2[~df2[col2].isin(overlap)].copy()

    return df1_filtered, df2_filtered


def randomly_assign_random_dummy_based_on_targets(y_var):
    # randomly select 50% of all_targets to assign random_dummy to be true
    all_targets = y_var['target_index'].unique()
    half_targets = np.random.choice(
        all_targets, size=int(len(all_targets)*0.5), replace=False)
    y_var['random_dummy'] = 0
    y_var.loc[y_var['target_index'].isin(half_targets), 'random_dummy'] = 1
    return y_var


def train_test_split_based_on_targets(x_var, y_var):
    all_targets = y_var['target_index'].unique()

    test_targets = np.random.choice(
        all_targets, size=int(len(all_targets)*0.2), replace=False)
    train_targets = [
        target for target in all_targets if target not in test_targets]

    x_var.reset_index(drop=True, inplace=True)
    y_var.reset_index(drop=True, inplace=True)

    train_rows = y_var['target_index'].isin(train_targets)
    test_rows = y_var['target_index'].isin(test_targets)

    X_train = x_var[train_rows]
    X_test = x_var[test_rows]

    y_train = y_var[train_rows]
    y_test = y_var[test_rows]

    return X_train, X_test, y_train, y_test
