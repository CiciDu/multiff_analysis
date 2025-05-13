import sys
from planning_analysis.show_planning import nxt_ff_utils, show_planning_class, show_planning_utils
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_class, plot_stops_near_ff_utils, plot_monkey_heading_helper_class, stops_near_ff_based_on_ref_class
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils
from data_wrangling import specific_utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from math import pi
import itertools
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


all_x_vars_of_interest = ['test_or_control',
                          'ref_columns_only',
                          'y_var_column',
                          'monkey_or_agent',
                          'cluster_for_interaction',
                          'add_ref_interaction',
                          'key_for_split',
                          'whether_even_out_dist',
                          'ref_point_value',
                          'cluster_to_keep',
                          'use_combd_features_for_cluster_only',
                          'whether_filter_info',
                          'if_test_nxt_ff_group_appear_after_stop',
                          'keep_monkey_info',
                          'id',
                          'num_obs_ff',
                          'max_in_memory_time',
                          'whether_with_cost',
                          'dv_cost_factor',
                          'dw_cost_factor',
                          'w_cost_factor',
                          'monkey_name',
                          'optimal_arc_type',
                          'curv_traj_window_before_stop',
                          ]


def make_new_df_for_plotly_comparison(df, match_rows_based_on_ref_columns_only=True):

    try:
        df['sample_size'] = df['ctrl_sample_size']
        df.loc[df['test_or_control'] == 'test',
               'sample_size'] = df['test_sample_size']
    except KeyError:
        pass

    if match_rows_based_on_ref_columns_only & ('ref_columns_only' in df.columns):
        new_df = _supply_info_to_ref_column_subset_to_match_more_columns_subset(
            df)
    else:
        new_df = df.copy()

    _ = examine_columns(new_df)
    return new_df


def _supply_info_to_ref_column_subset_to_match_more_columns_subset(overall_df):
    ref_sub = overall_df[overall_df['ref_columns_only'] == True].copy()
    if 'use_combd_features_for_cluster_only' in ref_sub.columns:
        # in case there are any rows with use_combd_features_for_cluster_only == True
        ref_sub = ref_sub[ref_sub['use_combd_features_for_cluster_only'] == False].copy(
        )
    more_sub = overall_df[overall_df['ref_columns_only'] == False].copy()

    more_sub = specific_utils.ensure_boolean_dtype(more_sub)
    ref_sub = specific_utils.ensure_boolean_dtype(ref_sub)

    columns_to_match = ['cluster_to_keep', 'cluster_for_interaction',
                        'use_combd_features_for_cluster_only']
    columns_to_match = [
        col for col in columns_to_match if col in more_sub.columns]

    if len(columns_to_match) > 0:
        new_ref_sub = pd.DataFrame()
        for index, row in more_sub[columns_to_match].drop_duplicates().iterrows():
            # find the subset of more_sub that matches the row in the columns_to_match
            more_sub2 = more_sub[more_sub[columns_to_match].eq(
                row).all(axis=1)].copy()
            # since ref_sub has only one combination of columns_to_match throughout the df, we can just copy ref_sub and replace the values in columns_to_match with the values in row
            # until we have a matching copy of all combinations of columns_to_match for ref_sub and more_sub
            ref_sub2 = ref_sub.copy()
            for col in columns_to_match:
                ref_sub2[col] = row[col]

            # check if more_sub2 and ref_sub2 have the same number of rows
            if more_sub2.shape[0] != ref_sub2.shape[0]:
                print('more_sub2.shape[0]:', more_sub2.shape[0])
                print('ref_sub2.shape[0]:', ref_sub2.shape[0])
                raise ValueError(
                    'more_sub and new_ref_sub have different number of rows')

            # concatenate ref_sub2 to new_ref_sub
            new_ref_sub = pd.concat([new_ref_sub, ref_sub2], axis=0)

        new_ref_sub.reset_index(drop=True, inplace=True)

    else:
        new_ref_sub = ref_sub.copy()

    new_overall_df = pd.concat([more_sub, new_ref_sub], axis=0)
    new_overall_df.reset_index(drop=True, inplace=True)
    return new_overall_df


def examine_columns(new_df, verbose=True):
    columns_to_examine = all_x_vars_of_interest

    if verbose:
        print('Columns (among selected ones for examination) that have variations:')

    columns_with_variations = []
    for column in columns_to_examine:
        if column in new_df.columns:
            if len(new_df[column].unique()) > 1:
                if verbose:
                    print('===================================================')
                    print(' ')
                    print(column, new_df[column].unique())
                columns_with_variations.append(column)
    return columns_with_variations


def get_combinations_from_changeable_variables(df, changeable_variables):
    if len(changeable_variables) > 0:
        changeable_variables = {key: df[key].unique()
                                for key in changeable_variables}
        # Get all unique combinations from changeable_variables
        keys, values = zip(*changeable_variables.items())
        combinations = [dict(zip(keys, combination))
                        for combination in itertools.product(*values)]
    else:
        combinations = [{}]

    return combinations


def break_up_df_to_smaller_ones(df, fixed_variable_values_to_use, changeable_variables, var_to_determine_x_offset_direction=None,
                                y_var_column='avg_r_squared', se_column=None):

    processed_df = process_lr_or_ml_df_for_visualization(df, var_to_split=var_to_determine_x_offset_direction,
                                                         y_var_column=y_var_column, se_column=se_column)

    # Filter the DataFrame by fixed_variable_values_to_use
    for key, value in fixed_variable_values_to_use.items():
        processed_df = processed_df[processed_df[key] == value]

    combinations = get_combinations_from_changeable_variables(
        processed_df, changeable_variables)

    list_of_smaller_dfs = []
    for combo in combinations:
        filtered_df = processed_df.copy()
        for key, value in combo.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        list_of_smaller_dfs.append(filtered_df)

    return list_of_smaller_dfs, combinations


def process_lr_or_ml_df_for_visualization(lr_or_ml_df,
                                          all_x_vars_of_interest=all_x_vars_of_interest,
                                          y_var_column='avg_r_squared',
                                          se_column=None,
                                          var_to_split='ref_columns_only',
                                          ):

    lr_or_ml_df = lr_or_ml_df.copy()
    all_x_vars_of_interest = [
        col for col in all_x_vars_of_interest if col in lr_or_ml_df.columns]
    if var_to_split is None:
        vars_to_keep = list(set(all_x_vars_of_interest +
                            ['sample_size', y_var_column]))
    else:
        all_x_vars_of_interest = [
            col for col in all_x_vars_of_interest if col != var_to_split]
        vars_to_keep = list(set(all_x_vars_of_interest +
                            ['sample_size', var_to_split, y_var_column]))

    if se_column is not None:
        lr_or_ml_df['se_upper'] = lr_or_ml_df[y_var_column] + \
            lr_or_ml_df[se_column] * 2
        lr_or_ml_df['se_lower'] = lr_or_ml_df[y_var_column] - \
            lr_or_ml_df[se_column] * 2
        vars_to_keep.extend(['se_upper', 'se_lower'])

    try:
        # lr_or_ml_df['ref_point_value'] = lr_or_ml_df['ref_point_value'].astype('str')
        if 'ref_columns_only' in lr_or_ml_df.columns:
            lr_or_ml_df['ref_columns_only'] = lr_or_ml_df['ref_columns_only'].astype(
                'object')
            lr_or_ml_df.loc[lr_or_ml_df['ref_columns_only'] ==
                            True, 'ref_columns_only'] = 'ref_columns_only'
            lr_or_ml_df.loc[lr_or_ml_df['ref_columns_only']
                            == False, 'ref_columns_only'] = 'more_columns'
        if 'cluster_for_interaction' in lr_or_ml_df.columns:
            lr_or_ml_df.loc[lr_or_ml_df['cluster_for_interaction'].isnull(
            ), 'cluster_for_interaction'] = 'none'
    except KeyError:
        pass

    sub_df = lr_or_ml_df[vars_to_keep].copy()

    # processed_df = sub_df.pivot(index=list(set(all_x_vars_of_interest + ['y_var_column'])), columns=[var_to_split], values=[y_var_column, 'sample_size']).reset_index()
    # processed_df.columns = [column[1] + ': ' + str(column[0]) if column[1] != '' else column[0] for column in processed_df.columns]

    n_groups = sub_df.groupby(all_x_vars_of_interest).ngroup()
    sub_df['pair_id'] = n_groups.values
    sub_df.sort_values(by='pair_id', inplace=True)

    if var_to_split is not None:
        sub_df['y1_or_y2'] = 'y1'
        if var_to_split == 'test_or_control':
            sub_df.loc[sub_df[var_to_split] == 'control', 'y1_or_y2'] = 'y2'
        elif var_to_split == 'ref_columns_only':
            sub_df.loc[sub_df[var_to_split] ==
                       'ref_columns_only', 'y1_or_y2'] = 'y2'
        else:
            sub_df.loc[sub_df[var_to_split] == sub_df[var_to_split].unique()[
                1], 'y1_or_y2'] = 'y2'

        y_var_color_map = {'y1': 'blue', 'y2': 'orange'}
        sub_df['line_color'] = sub_df['y1_or_y2'].map(y_var_color_map)
        sub_df['var_to_split_value'] = sub_df[var_to_split]
    else:
        sub_df['line_color'] = 'blue'
        sub_df['var_to_split_value'] = ''
    processed_df = sub_df.copy()

    return processed_df


def assign_color_to_sub_df_based_on_unique_combinations(sub_df, columns_to_find_unique_combinations_for_color):
    # Step 1: Create a unique identifier for each combination of values in rest_of_x columns
    if columns_to_find_unique_combinations_for_color is not None:
        sub_df['unique_combination'] = sub_df.apply(lambda row: '; '.join(col + '=' + str(row[col]) if len(col) < 10 else str(row[col])
                                                                          for col in columns_to_find_unique_combinations_for_color), axis=1)

        # Step 2: Map these unique combinations to colors
        # Assuming you have a predefined list of colors
        colors = ['green', 'red', 'yellow', 'purple', 'pink', 'grey', 'lime', 'cyan',
                  'magenta', 'brown', 'black', 'white', 'silver', 'gold', 'indigo', 'violet', 'teal', 'olive',
                  'navy', 'maroon', 'aquamarine', 'coral', 'crimson', 'darkred', 'darkorange', 'darkgreen', 'darkblue',
                  'darkviolet', 'darkcyan', 'darkmagenta', 'darkgrey']
        color_map = {comb: colors[i % len(colors)] for i, comb in enumerate(
            sub_df['unique_combination'].unique())}

        # Step 3: Assign colors to a new column based on the unique combinations
        sub_df['color'] = sub_df['unique_combination'].map(color_map)

        # Optionally, you can drop the 'unique_combination' column if it's no longer needed
        # sub_df.drop('unique_combination', axis=1, inplace=True)
    else:
        sub_df['color'] = 'black'
    return sub_df


def assign_line_type_to_sub_df_based_on_unique_combinations(sub_df, columns_to_find_unique_combinations_for_line):

    if len(columns_to_find_unique_combinations_for_line) > 0:
        sub_df['unique_combination_for_line'] = sub_df.apply(lambda row: '; '.join(
            col + '=' + str(row[col]) for col in columns_to_find_unique_combinations_for_line), axis=1)

        lines = ['dash', 'solid', 'dot', 'dashdot', 'longdash', 'longdashdot']

        line_type_map = {comb: lines[i % len(lines)] for i, comb in enumerate(
            sub_df['unique_combination_for_line'].unique())}

        sub_df['line_type'] = sub_df['unique_combination_for_line'].map(
            line_type_map)
    else:
        sub_df['unique_combination_for_line'] = ''
        sub_df['line_type'] = 'solid'
    return sub_df


def make_features_df(wide_df_sub):
    coeff_columns = [col for col in wide_df_sub.columns if 'coeff' in col]
    coeff_numbers = [col[6:] for col in coeff_columns]

    features = wide_df_sub[coeff_numbers].T.reset_index(drop=True)
    features.columns = ['feature']

    coeff = wide_df_sub[coeff_columns].T.reset_index(drop=True)
    coeff.columns = ['coeff']

    features_df = pd.concat([features, coeff], axis=1)
    features_df['abs_coeff'] = np.abs(features_df['coeff'])
    features_df.sort_values(by='abs_coeff', ascending=False, inplace=True)
    features_df['rank'] = np.arange(1, features_df.shape[0]+1)
    return features_df


def make_all_features_df_by_separating_based_on_a_column(all_info_sub, column='test_or_control', make_plot=True):
    all_features_df = pd.DataFrame()

    for value in all_info_sub[column].unique():
        wide_df_sub = all_info_sub[all_info_sub[column] == value].copy()
        if wide_df_sub.shape[0] != 1:
            raise ValueError('wide_df_sub does not contain exactly 1 row.')
        features_df = make_features_df(wide_df_sub)
        features_df[column] = value
        all_features_df = pd.concat([all_features_df, features_df], axis=0)

    all_features_df.reset_index(drop=True, inplace=True)

    if 'abs_coeff' in all_features_df.columns:
        all_features_df = all_features_df.dropna(subset=['abs_coeff']).copy()

    if make_plot:
        plt.figure(figsize=(10, 15))
        sns.barplot(data=all_features_df, x='coeff', y='feature', hue=column)
        plt.show()

    return all_features_df


def get_smaller_dfs_to_plot_coeff(df, column_to_split_grouped_bars='test_or_control', fixed_variable_values_to_use={}):
    columns_with_variations = examine_columns(df, verbose=False)

    columns_accounted_for = [column_to_split_grouped_bars] + \
        list(fixed_variable_values_to_use.keys())
    changeable_variables = [
        col for col in columns_with_variations if col not in columns_accounted_for]
    list_of_smaller_dfs, combinations = break_up_df_to_smaller_ones(df, fixed_variable_values_to_use, changeable_variables,
                                                                    var_to_determine_x_offset_direction=None, y_var_column='avg_r_squared')
    return list_of_smaller_dfs, combinations


def melt_df_by_test_and_control(df, test_column='test_perc', control_column='ctrl_perc', value_name='perc'):
    df = df.copy()
    df.rename(columns={test_column: 'test',
              control_column: 'control'}, inplace=True)
    value_vars = ['test', 'control']
    df2 = df.melt(id_vars=[col for col in df.columns if col not in value_vars],
                  value_vars=value_vars,
                  var_name='test_or_control',
                  value_name=value_name).copy()

    df2['sample_size'] = -999
    df2.loc[df2['test_or_control'] == 'test',
            'sample_size'] = df2.loc[df2['test_or_control'] == 'test', 'test_sample_size']
    df2.loc[df2['test_or_control'] == 'control',
            'sample_size'] = df2.loc[df2['test_or_control'] == 'control', 'ctrl_sample_size']
    df2.drop(columns=['test_sample_size', 'ctrl_sample_size'], inplace=True)
    return df2


def plot_heading_in_overall_median_info(overall_median_info, title_prefix=None):
    new_overall_median_info = make_new_df_for_plotly_comparison(
        overall_median_info)

    x_var_column_list = ['ref_point_value']

    fixed_variable_values_to_use = {
        # 'whether_even_out_dist': True
        # 'max_curv_range': 200,
        # 'whether_filter_info': True,
        'if_test_nxt_ff_group_appear_after_stop': 'flexible',
        'key_for_split': 'ff_seen',
        'whether_even_out_dist': False,
    }

    # changeable_variables = ['whether_even_out_dist']
    changeable_variables = []

    columns_to_find_unique_combinations_for_color = []
    columns_to_find_unique_combinations_for_line = []

    plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(new_overall_median_info,
                                                                                    fixed_variable_values_to_use,
                                                                                    changeable_variables,
                                                                                    x_var_column_list,
                                                                                    y_var_column='diff_in_abs_d_heading_median',
                                                                                    se_column='diff_in_abs_d_heading_boot_med_std',
                                                                                    # var_to_determine_x_offset_direction=None,
                                                                                    title_prefix=title_prefix,
                                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)


def plot_curv_in_overall_median_info(overall_median_info, title_prefix=None):

    new_overall_median_info = make_new_df_for_plotly_comparison(
        overall_median_info)
    new_overall_median_info['sample_size'] = new_overall_median_info['sample_size_for_curv']

    x_var_column_list = ['ref_point_value']

    fixed_variable_values_to_use = {
        # 'whether_even_out_dist': True
        # 'max_curv_range': 200,
        # 'whether_filter_info': True,
        'if_test_nxt_ff_group_appear_after_stop': 'flexible',
        'key_for_split': 'ff_seen',
        'whether_even_out_dist': False,
    }

    changeable_variables = []

    columns_to_find_unique_combinations_for_color = []
    columns_to_find_unique_combinations_for_line = []

    plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(new_overall_median_info,
                                                                                    fixed_variable_values_to_use,
                                                                                    changeable_variables,
                                                                                    x_var_column_list,
                                                                                    y_var_column='diff_in_abs_d_curv_median',
                                                                                    se_column='diff_in_abs_d_curv_boot_med_std',
                                                                                    title_prefix=title_prefix,
                                                                                    # var_to_determine_x_offset_direction=None,
                                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                                                    columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line)


def plot_perc_in_overall_all_perc_info(all_perc_info, title_prefix=None):

    new_perc_df = make_new_df_for_plotly_comparison(all_perc_info,
                                                    match_rows_based_on_ref_columns_only=False)

    x_var_column_list = ['key_for_split']

    fixed_variable_values_to_use = {'if_test_nxt_ff_group_appear_after_stop': 'flexible',
                                    'whether_even_out_dist': False,
                                    }

    changeable_variables = []

    columns_to_find_unique_combinations_for_color = []

    plot_variations_utils.streamline_making_plotly_plot_to_compare_two_sets_of_data(new_perc_df,
                                                                                    fixed_variable_values_to_use,
                                                                                    changeable_variables,
                                                                                    x_var_column_list,
                                                                                    y_var_column='perc',
                                                                                    se_column='perc_se',
                                                                                    title_prefix=title_prefix,
                                                                                    var_to_determine_x_offset_direction='test_or_control',
                                                                                    columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color)
