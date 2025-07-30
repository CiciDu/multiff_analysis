from planning_analysis.variations_of_factors_vs_results import process_variations_utils
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import copy


def _check_order_in_changeable_variables(changeable_variables, original_df):
    first_dim = original_df[changeable_variables[0]].nunique()
    second_dim = original_df[changeable_variables[1]].nunique()
    if first_dim < second_dim:
        # reorder changeable_variables
        temp = changeable_variables[0]
        changeable_variables[0] = changeable_variables[1]
        changeable_variables[1] = temp
    return changeable_variables


def _find_first_and_second_dim(original_df, changeable_variables, combinations):
    if len(changeable_variables) == 0:
        first_dim = 1
        second_dim = 1
    elif len(changeable_variables) == 2:
        first_dim = original_df[changeable_variables[0]].nunique()
        second_dim = original_df[changeable_variables[1]].nunique()
    else:
        first_dim = max(1, math.ceil(len(combinations)/2))
        second_dim = 2
    return first_dim, second_dim


def _get_all_subplot_titles(combinations):
    all_subplot_titles = []
    for combo in combinations:
        title = ''
        for key, value in combo.items():
            title += key + ': ' + str(value) + ', '
        all_subplot_titles.append(title)
    return all_subplot_titles


def streamline_making_plotly_plot_to_compare_two_sets_of_data(original_df,
                                                              fixed_variable_values_to_use,
                                                              changeable_variables,
                                                              x_var_column_list,
                                                              columns_to_find_unique_combinations_for_color=[
                                                                  'test_or_control'],
                                                              columns_to_find_unique_combinations_for_line=[],
                                                              var_to_determine_x_offset_direction='ref_columns_only',
                                                              y_var_column='avg_r_squared',
                                                              se_column=None,
                                                              title_prefix=None,
                                                              use_subplots_based_on_changeable_variables=False):

    if use_subplots_based_on_changeable_variables & (len(changeable_variables) == 2):
        changeable_variables = _check_order_in_changeable_variables(
            changeable_variables, original_df)

    list_of_smaller_dfs, combinations = process_variations_utils.break_up_df_to_smaller_ones(original_df, fixed_variable_values_to_use, changeable_variables,
                                                                                             var_to_determine_x_offset_direction=var_to_determine_x_offset_direction, y_var_column=y_var_column,
                                                                                             se_column=se_column)

    if use_subplots_based_on_changeable_variables:
        first_dim, second_dim = _find_first_and_second_dim(
            original_df, changeable_variables, combinations)

        # get a subplot title for each combo in combinations
        all_subplot_titles = _get_all_subplot_titles(combinations)

        fig = make_subplots(rows=first_dim, cols=second_dim,
                            subplot_titles=all_subplot_titles)
        # change the font size of the subplot titles
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, family='Arial')

        row_number = 1
        col_number = 1
    else:
        fig = go.Figure()
        row_number = None
        col_number = None

    for combo, filtered_df in zip(combinations, list_of_smaller_dfs):
        for x_var_column in x_var_column_list:
            if 'y_var_column' in combo.keys():
                title = str.upper(x_var_column) + ' vs ' + \
                    str.upper(combo['y_var_column'])
            else:
                title = str.upper(x_var_column) + ' vs ' + \
                    str.upper(y_var_column)

            if title_prefix is not None:
                title = title_prefix + ' ' + title

            if not use_subplots_based_on_changeable_variables:
                fig = go.Figure()
                print(' ')
                print('=========================================================')
                print('Current combination of changeable variables:', combo)

            fig = make_plotly_plot_to_compare_two_sets_of_data(filtered_df,
                                                               x_var_column,
                                                               y_var_column,
                                                               var_to_determine_x_offset_direction=var_to_determine_x_offset_direction,
                                                               title=title,
                                                               columns_to_find_unique_combinations_for_color=columns_to_find_unique_combinations_for_color,
                                                               columns_to_find_unique_combinations_for_line=columns_to_find_unique_combinations_for_line,
                                                               fig=fig,
                                                               row_number=row_number,
                                                               col_number=col_number,
                                                               )
            if use_subplots_based_on_changeable_variables:
                # for trace in fig.data:
                #     fig.add_trace(trace, row=row_number, col=col_number)
                col_number += 1
                if col_number > second_dim:
                    col_number = 1
                    row_number += 1
                # hide x axis title
                fig.update_xaxes(title_text='', row=row_number, col=col_number)
            else:
                fig.show()

    if use_subplots_based_on_changeable_variables:
        fig.update_layout(height=400 * first_dim, width=450 * second_dim)
        fig.show()


def _find_rest_of_x_for_hoverdata(sub_df, x_var_column, y_var_column, var_to_determine_x_offset_direction):
    rest_of_x_for_hoverdata = []
    for column in sub_df.columns:
        if len(sub_df[column].unique()) > 1:
            if ('sample_size' not in column) & ('diff_in_angle_to_nxt_ff' not in column) & ('unique_combination' not in column) & \
                (column not in ['pair_id', 'y1_or_y2', 'line_color', 'x_value_numeric', 'x_value_numeric_with_offset',
                                'var_to_split_value', 'se_upper', 'se_lower', x_var_column, y_var_column, var_to_determine_x_offset_direction]):
                rest_of_x_for_hoverdata.append(column)
    return rest_of_x_for_hoverdata


def _process_x_var_columns(sub_df, x_var_column):
    if x_var_column == 'ref_point_value':
        sub_df['ref_point_value'] = sub_df['ref_point_value'].astype('str')
    else:
        # if the data type of sub_df[x_var_column], is bool, change it to str
        if sub_df[x_var_column].dtype == 'bool':
            sub_df[x_var_column] = sub_df[x_var_column].astype('str')
    return sub_df


def _process_columns_to_find_unique_combinations_for_color(columns_to_find_unique_combinations_for_color, x_var_column, rest_of_x_for_hoverdata):
    rest_of_x_for_hoverdata = copy.deepcopy(rest_of_x_for_hoverdata)
    rest_of_x_for_hoverdata = [
        column for column in rest_of_x_for_hoverdata if '_se' not in column]
    if len(columns_to_find_unique_combinations_for_color) > 0:
        if (len(columns_to_find_unique_combinations_for_color) == 1):
            if (columns_to_find_unique_combinations_for_color[0] == x_var_column):
                if len(rest_of_x_for_hoverdata) > 0:
                    # since the original color will no longer be meaningful, as the info is given by x_var_column already
                    columns_to_find_unique_combinations_for_color = [
                        rest_of_x_for_hoverdata[0]]
                else:
                    columns_to_find_unique_combinations_for_color = []
    else:
        # if len(rest_of_x_for_hoverdata) > 0:
        #     columns_to_find_unique_combinations_for_color = [rest_of_x_for_hoverdata[0]]
        # else:
        #     columns_to_find_unique_combinations_for_color = []
        columns_to_find_unique_combinations_for_color = []
    return columns_to_find_unique_combinations_for_color


def _find_x_labels_to_values_map(sub_df, x_var_column):
    if x_var_column == 'ref_point_value':
        desired_order = ['-150.0', '-125.0', '-100.0',
                         '-75.0', '-50.0', '-0.2', '-0.1', '0.0', '0.1']
        unique_existing_values = sub_df[x_var_column].unique()
        desired_order = [
            value for value in desired_order if value in unique_existing_values]
        x_labels_to_values_map = dict(
            zip(desired_order, np.arange(len(desired_order))))
    else:
        x_labels_to_values_map = dict(
            zip(sub_df[x_var_column].unique(), np.arange(len(sub_df[x_var_column].unique()))))
    return x_labels_to_values_map


def _add_x_value_numeric_to_sub_df(sub_df, x_var_column, x_labels_to_values_map, x_offset):
    x_values_numeric = sub_df[x_var_column].map(x_labels_to_values_map)
    sub_df['x_value_numeric'] = x_values_numeric
    sub_df['x_value_numeric_with_offset'] = x_values_numeric
    sub_df['x_value_numeric_with_offset'] = sub_df['x_value_numeric_with_offset'].astype(
        float)
    try:
        sub_df.loc[sub_df['y1_or_y2'] == 'y1', 'x_value_numeric_with_offset'] = sub_df.loc[sub_df['y1_or_y2']
                                                                                           == 'y1', 'x_value_numeric_with_offset'] + x_offset
        sub_df.loc[sub_df['y1_or_y2'] == 'y2', 'x_value_numeric_with_offset'] = sub_df.loc[sub_df['y1_or_y2']
                                                                                           == 'y2', 'x_value_numeric_with_offset'] - x_offset
    except KeyError:
        pass
    return sub_df


def _update_fig_based_on_x_labels_to_values_map(fig, x_labels_to_values_map,
                                                row_number=None,
                                                col_number=None,
                                                ):
    x_labels_to_values_map = copy.deepcopy(x_labels_to_values_map)
    x_labels_to_values_map = {
        str(key): value for key, value in x_labels_to_values_map.items()}
    if any([len(label) > 35 for label in x_labels_to_values_map.keys()]):
        fig.update_layout(width=800, height=1300)  # Set the figure size here
    else:
        fig.update_layout(width=800, height=500)

    fig.update_xaxes(tickvals=list(x_labels_to_values_map.values()), ticktext=list(x_labels_to_values_map.keys()),
                     row=row_number, col=col_number)

    # rotated x labels by 90 degrees if any x label is longer than 20 letters
    if any([len(label) > 30 for label in x_labels_to_values_map.keys()]):
        fig.update_xaxes(tickangle=-90)
    elif any([len(label) > 10 for label in x_labels_to_values_map.keys()]):
        fig.update_xaxes(tickangle=-25)
    return fig


def _set_minimal_y_scale(fig, sub_df, y_var_column,
                         row_number=None,
                         col_number=None,
                         ):
    # set minimal y scale based on min and max values
    if 'se_upper' in sub_df.columns:
        min_y = sub_df['se_lower'].min()
        max_y = sub_df['se_upper'].max()
    else:
        min_y = sub_df[y_var_column].min()
        max_y = sub_df[y_var_column].max()

    margin = max(0.1, 0.05*abs(max_y))
    fig.update_yaxes(range=[min_y - margin, max_y + margin],
                     row=row_number, col=col_number)
    return fig


def make_plotly_plot_to_compare_two_sets_of_data(sub_df,
                                                 x_var_column,
                                                 y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                 var_to_determine_x_offset_direction='ref_columns_only',
                                                 title=None,
                                                 x_offset=0.1,
                                                 columns_to_find_unique_combinations_for_color=[],
                                                 columns_to_find_unique_combinations_for_line=[],
                                                 show_combo_legends=True,
                                                 fig=None,
                                                 row_number=None,
                                                 col_number=None,
                                                 ):
    if fig is None:
        fig = go.Figure()

    rest_of_x_for_hoverdata = _find_rest_of_x_for_hoverdata(
        sub_df, x_var_column, y_var_column, var_to_determine_x_offset_direction)
    sub_df = _process_x_var_columns(sub_df, x_var_column)

    columns_to_find_unique_combinations_for_color = _process_columns_to_find_unique_combinations_for_color(
        columns_to_find_unique_combinations_for_color, x_var_column, rest_of_x_for_hoverdata)
    sub_df = process_variations_utils.assign_color_to_sub_df_based_on_unique_combinations(
        sub_df, columns_to_find_unique_combinations_for_color)
    sub_df = process_variations_utils.assign_line_type_to_sub_df_based_on_unique_combinations(
        sub_df, columns_to_find_unique_combinations_for_line)

    # Define color mapping
    x_labels_to_values_map = _find_x_labels_to_values_map(sub_df, x_var_column)
    sub_df = _add_x_value_numeric_to_sub_df(
        sub_df, x_var_column, x_labels_to_values_map, x_offset)

    fig = plot_markers_for_data_comparison(
        fig, sub_df, rest_of_x_for_hoverdata, y_var_column, row_number=row_number, col_number=col_number)
    fig = connect_every_pair(fig, sub_df, y_var_column, rest_of_x_for_hoverdata,
                             show_combo_legends=show_combo_legends, row_number=row_number, col_number=col_number)
    fig = _update_fig_based_on_x_labels_to_values_map(
        fig, x_labels_to_values_map, row_number=row_number, col_number=col_number)

    fig = _set_minimal_y_scale(
        fig, sub_df, y_var_column, row_number=row_number, col_number=col_number)

    fig = label_smallest_y_sample_size(
        fig, sub_df, y_var_column, row_number=row_number, col_number=col_number)

    # update title to be x_var_column, y axis title to be y_var_column, and x axis title to be x_var_column
    if title is None:
        title = f'{y_var_column} vs {x_var_column}'
    fig.update_layout(title=title, xaxis_title=x_var_column,
                      yaxis_title=y_var_column)

    return fig


def label_smallest_y_sample_size(fig, sub_df, y_var_column,
                                 row_number=None,
                                 col_number=None,
                                 ):
    max_x_value_numeric = sub_df['x_value_numeric'].max()
    for x_value in sub_df['x_value_numeric'].unique():
        subset = sub_df[sub_df['x_value_numeric'] == x_value].copy()
        if not subset.empty:
            min_y_value = subset[y_var_column].min()
            min_y_row = subset[subset[y_var_column] == min_y_value].iloc[0]
            sample_size = min_y_row['sample_size']
            x_offset_value = min_y_row['x_value_numeric_with_offset']
            if x_value == max_x_value_numeric:
                text = "sample size: " + str(sample_size)
            else:
                text = str(sample_size)
            fig.add_trace(go.Scatter(
                x=[x_offset_value],
                y=[min_y_value - max(0.01, 0.01*abs(min_y_value))],
                mode='text',
                text=[text],
                textposition='bottom center',
                showlegend=False,
                hoverinfo='skip'
            ), row=row_number, col=col_number)

    return fig


def _make_hovertemplate(df, y_var_column, customdata_columns):
    """
    Create a hover template for Plotly plots.

    Parameters:
    y_var_column (str): The column name for the y-axis variable.
    customdata_columns (list): A list of column names to include in the hover template.

    Returns:
    tuple: A tuple containing the hover template string and the updated customdata_columns list.
    """
    if 'sample_size' not in customdata_columns:
        customdata_columns = ['sample_size'] + customdata_columns
    df['sample_size'] = df['sample_size'].astype(int)
    customdata_columns = ['var_to_split_value'] + customdata_columns

    hovertemplate_parts = [f"%{{customdata[{0}]}} <br>" +
                           f"{y_var_column}: %{{y}}<br>"]

    for i, col in enumerate(customdata_columns):
        if i > 0:  # Skip the first element as it's already added
            dtype = df[col].dtype
            if (dtype == 'str') | (dtype == 'O'):
                hovertemplate_parts.append(f"{col}: %{{customdata[{i}]}}<br>")
            elif dtype == 'int':
                hovertemplate_parts.append(
                    f"{col}: %{{customdata[{i}]:d}}<br>")
            else:  # Assume numerical
                hovertemplate_parts.append(
                    f"{col}: %{{customdata[{i}]:.2f}}<br>")

    hovertemplate = "".join(hovertemplate_parts) + "<extra></extra>"
    return hovertemplate, customdata_columns


# def _make_hovertemplate(y_var_column, customdata_columns):
#     if 'sample_size' not in customdata_columns:
#         customdata_columns = ['sample_size'] + customdata_columns
#     customdata_columns = ['var_to_split_value'] + customdata_columns
#     hovertemplate_parts = [f"%{{customdata[{0}]}} <br>" +
#                            f"{y_var_column}: %{{y}}<br>"]
#     hovertemplate_parts += [f"{col}: %{{customdata[{i}]:.2f}}<br>" for i, col in enumerate(customdata_columns) if (i > 0)]
#     hovertemplate = "".join(hovertemplate_parts) + "<extra></extra>"
#     return hovertemplate, customdata_columns


marker_partial_kwargs = dict(mode='markers',
                             marker=dict(
                                 color='black',
                                 symbol='line-ew',
                                 size=10,
                                 opacity=0.5,
                                 line=dict(width=1.5)
                             ))


def plot_markers_for_data_comparison(fig,
                                     sub_df,
                                     customdata_columns,
                                     y_var_column,
                                     row_number=None,
                                     col_number=None,
                                     ):

    max_sample_size = sub_df['sample_size'].max()
    sample_size_to_maker_size_scaling_factor = 50/max_sample_size

    # drop na in sub_df
    sub_df = sub_df.dropna(subset=[y_var_column])
    hovertemplate, customdata_columns = _make_hovertemplate(
        sub_df, y_var_column, customdata_columns)

    showlegend = True
    if (row_number is not None) & (col_number is not None):
        if (row_number != 1) | (col_number != 1):
            showlegend = False

    for line_color in sub_df['line_color'].unique():
        sub_df2 = sub_df[sub_df['line_color'] == line_color].copy()
        name = sub_df2['var_to_split_value'].iloc[0]

        fig.add_trace(go.Scatter(
            x=sub_df2['x_value_numeric_with_offset'].values,
            y=sub_df2[y_var_column].values,
            name=name,
            **marker_partial_kwargs,
            marker_size=sub_df2['sample_size'].values *
            sample_size_to_maker_size_scaling_factor,
            line=dict(width=1.5,
                      color=line_color),
            customdata=sub_df2[customdata_columns].values,
            hovertemplate=hovertemplate,
            showlegend=showlegend,
        ),
            row=row_number, col=col_number
        )

        if 'se_upper' in sub_df2.columns:
            for x_value, y_upper, y_lower in zip(sub_df2['x_value_numeric_with_offset'], sub_df2['se_upper'], sub_df2['se_lower']):
                fig.add_trace(go.Scatter(
                    x=[x_value, x_value],
                    y=[y_lower, y_upper],
                    mode='lines',
                    line=dict(width=1.5,
                              color=line_color),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                    row=row_number, col=col_number
                )
    return fig


def connect_every_pair(fig, sub_df, y_var_column, customdata_columns, show_combo_legends=True,
                       row_number=None,
                       col_number=None,
                       ):

    hovertemplate, customdata_columns = _make_hovertemplate(
        sub_df, y_var_column, customdata_columns)
    for pair_id in sub_df['pair_id'].unique():

        # Find the index of the rows corresponding to the current x value
        sub_df2 = sub_df[sub_df['pair_id'] == pair_id].copy()
        row = sub_df2.iloc[0]
        if len(sub_df2) > 0:

            fig.add_trace(go.Scatter(x=sub_df2['x_value_numeric_with_offset'], y=sub_df2[y_var_column],
                                     mode='markers+lines',
                                     line=dict(color=row['color'],
                                               width=1,
                                               dash=row['line_type']),
                                     marker=dict(color=row['color'],
                                                 symbol='line-ew',
                                                 size=10,
                                                 opacity=0.5),  # this will actually not show color in the plot, so we still need plot_markers_for_data_comparison
                                     customdata=sub_df2[customdata_columns].values,
                                     hovertemplate=hovertemplate,
                                     showlegend=False,
                                     legendgroup=row['color'],
                                     name=row['unique_combination']),
                          row=row_number, col=col_number)  # Use color as the name to group by color in the legend

    if show_combo_legends:
        _add_color_legends(fig, sub_df, row_number=row_number,
                           col_number=col_number)
        _add_line_type_legends(
            fig, sub_df, row_number=row_number, col_number=col_number)

    return fig


def _add_color_legends(fig, sub_df,
                       row_number=None,
                       col_number=None,
                       ):

    showlegend = True
    if (row_number is not None) & (col_number is not None):
        if (row_number != 1) | (col_number != 1):
            showlegend = False

    color_to_show_legend = sub_df[[
        'color', 'unique_combination']].drop_duplicates()
    if len(color_to_show_legend) > 1:
        for index, row in color_to_show_legend.iterrows():
            fig.add_trace(go.Scatter(x=[0, 0], y=[0, 0],
                                     mode='lines',
                                     line=dict(color=row['color'],
                                               width=1,
                                               dash='solid'),
                                     showlegend=showlegend,
                                     legendgroup=row['color'],
                                     visible=True,
                                     name=row['unique_combination']),
                          row=row_number, col=col_number
                          )


def _add_line_type_legends(fig, sub_df,
                           row_number=None,
                           col_number=None,
                           ):
    showlegend = True
    if (row_number is not None) & (col_number is not None):
        if (row_number != 1) | (col_number != 1):
            showlegend = False

    line_type_to_show_legend = sub_df[[
        'line_type', 'unique_combination_for_line']].drop_duplicates()
    if len(line_type_to_show_legend) > 1:
        for index, row in line_type_to_show_legend.iterrows():
            fig.add_trace(go.Scatter(x=[0, 0], y=[0, 0],
                                     mode='lines',
                                     line=dict(color='black',
                                               width=1,
                                               dash=row['line_type']),
                                     showlegend=showlegend,
                                     legendgroup=row['unique_combination_for_line'],
                                     visible=True,
                                     name=row['unique_combination_for_line']),
                          row=row_number, col=col_number
                          )


def compare_diff_in_abs_in_overall_median_info(sub_df, x, plotly=True):
    if plotly:
        make_plotly_plots_for_test_and_control_data_comparison(
            sub_df, x, 'test_diff_in_abs_angle_to_nxt_ff', 'ctrl_diff_in_abs_angle_to_nxt_ff', title='test vs ctrl diff_in_abs_angle_to_nxt_ff')
        make_plotly_plots_for_test_and_control_data_comparison(
            sub_df, x, 'delta_diff_in_abs_angle_to_nxt_ff', 'delta_diff_in_angle_to_nxt_ff',  title='delta diff_in_abs vs delta diff_in_angle_to_nxt_ff')
    else:
        _compare_y1_and_y2_in_overall_regrouped_info(
            sub_df, x, 'test_diff_in_abs_angle_to_nxt_ff', 'ctrl_diff_in_abs_angle_to_nxt_ff', title='test vs ctrl diff_in_abs_angle_to_nxt_ff')
        _compare_y1_and_y2_in_overall_regrouped_info(
            sub_df, x, 'delta_diff_in_abs_angle_to_nxt_ff', 'delta_diff_in_angle_to_nxt_ff',  title='delta diff_in_abs vs delta diff_in_angle_to_nxt_ff')
    return


def compare_test_and_ctrl_in_all_perc_info(sub_df, x, plotly=True):
    if plotly:
        make_plotly_plots_for_test_and_control_data_comparison(
            sub_df, x, 'test_perc', 'ctrl_perc', title='test perc vs ctrl perc')
    else:
        _compare_y1_and_y2_in_overall_regrouped_info(
            sub_df, x, 'test_perc', 'ctrl_perc', title='test perc vs ctrl perc')
    return


def _compare_y1_and_y2_in_overall_regrouped_info(sub_df, x, y1, y2, title=''):
    print('sample_size:',
          sub_df[['test_sample_size', 'ctrl_sample_size']].describe())
    # maybe grouped bar plots?
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub_df, x=x, y=y1, color='blue', alpha=0.5, ax=ax)
    sns.barplot(data=sub_df, x=x, y=y2, color='orange', alpha=0.5, ax=ax)

    # Label all bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2. * 1.5, p.get_height(), f'{p.get_height():.2f}',
                ha='center', va='bottom')
    ax.set_title(title)

    if x == 'if_test_nxt_ff_group_appear_after_stop':
        # make x label rotated on ax
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           horizontalalignment='right')
    plt.show()
    return


def make_plotly_plots_for_test_and_control_data_comparison(sub_df, x, y1, y2, title=''):
    print('sample_size:',
          sub_df[['test_sample_size', 'ctrl_sample_size']].describe())

    # Initialize an empty figure
    fig = go.Figure()
    fig.update_layout(width=800, height=800)  # Set the figure size here

    # Define color mapping
    color_map = {y1: 'blue', y2: 'orange'}
    customdata_map = {y1: 'test_sample_size', y2: 'ctrl_sample_size'}

    # Define offset for x values to separate y1 and y2 visually
    offset = 0.1  # Adjust the offset value as needed

    x_labels_to_values_map = dict(
        zip(sub_df[x].unique(), np.arange(len(sub_df[x].unique()))))
    x_values_numeric = sub_df[x].map(x_labels_to_values_map)

    for i, variable in enumerate([y1, y2]):
        # Now that sub_df[x] is numeric, we can safely apply the offset
        x_values_with_offset = x_values_numeric + \
            (offset if i % 2 == 0 else -offset)

        fig.add_trace(go.Scatter(
            x=x_values_with_offset,
            y=sub_df[variable],
            name=variable,
            **marker_partial_kwargs,
            line=dict(color=color_map[variable]),
            customdata=sub_df[customdata_map[variable]].values,
            hovertemplate=f"%{{x}}<br>" +
            f"{variable}<br>" +
            f"value: %{{y}}<br>" +
            f"sample size: %{{customdata}}<extra></extra>"
        ))

    # Connect very two pairs
    # Iterate over unique x values
    for x_val in sub_df[x].unique():

        # Find the index of the rows corresponding to the current x value
        indices = sub_df[sub_df[x] == x_val].index

        # Calculate x positions with offset for y1 and y2
        x_pos_y1 = x_labels_to_values_map[x_val] + offset
        x_pos_y2 = x_labels_to_values_map[x_val] - offset

        # Extract y1 and y2 values
        y1_val = sub_df.loc[indices, y1].values
        y2_val = sub_df.loc[indices, y2].values

        # Add a line trace for each pair of y1 and y2 values
        for y1_v, y2_v in zip(y1_val, y2_val):
            fig.add_trace(go.Scatter(x=[x_pos_y1, x_pos_y2], y=[y1_v, y2_v],
                                     mode='lines', line=dict(color='grey', width=1),
                                     showlegend=False))

    # Update x-axis to use the original labels
    fig.update_xaxes(tickvals=list(x_labels_to_values_map.values()),
                     ticktext=list(x_labels_to_values_map.keys()))

    # The rest of the plotting code remains the same
    fig.show()


def plot_coeff(df, column_to_split_grouped_bars='test_or_control', fixed_variable_values_to_use={},
               max_num_plots=1):

    list_of_smaller_dfs, combinations = process_variations_utils.get_smaller_dfs_to_plot_coeff(
        df, column_to_split_grouped_bars=column_to_split_grouped_bars, fixed_variable_values_to_use=fixed_variable_values_to_use)

    plot_counter = 0
    for combo, df in zip(combinations, list_of_smaller_dfs):
        if plot_counter < max_num_plots:
            print(combo)
            _ = process_variations_utils.make_all_features_df_by_separating_based_on_a_column(
                df, column=column_to_split_grouped_bars)
            plot_counter += 1
    return
