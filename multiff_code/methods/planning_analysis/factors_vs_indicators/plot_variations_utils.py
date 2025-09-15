from pandas.api.types import is_string_dtype, is_integer_dtype, is_numeric_dtype
from planning_analysis.factors_vs_indicators import process_variations_utils
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd


def _check_order_in_changeable_variables(changeable_variables, original_df):
    first_dim = original_df[changeable_variables[0]].nunique()
    second_dim = original_df[changeable_variables[1]].nunique()
    if first_dim < second_dim:
        changeable_variables[0], changeable_variables[1] = (
            changeable_variables[1], changeable_variables[0]
        )
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
    """Build clean subplot titles like: 'var1: A · var2: B'."""
    titles = []
    for combo in combinations:
        # keep a stable key order
        parts = [f"{k}: {combo[k]}" for k in sorted(combo.keys())]
        titles.append(" · ".join(parts))
    return titles


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
                            subplot_titles=all_subplot_titles,
                            vertical_spacing=0.05, horizontal_spacing=0.05)
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
                plt.show()

    if use_subplots_based_on_changeable_variables:
        fig.update_layout(height=600 * first_dim, width=675 * second_dim)
        plt.show()

    return fig


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
    if sub_df[x_var_column].dtype == 'bool':
        sub_df[x_var_column] = sub_df[x_var_column].astype('str')

    # if x_var_column == 'ref_point_value': # this is probably only useful when ref_point_mode is 'cur_ff_visible'
    #     sub_df['ref_point_value'] = sub_df['ref_point_value'].astype('str')
    # else:
    #     # if the data type of sub_df[x_var_column], is bool, change it to str
    #     if sub_df[x_var_column].dtype == 'bool':
    #         sub_df[x_var_column] = sub_df[x_var_column].astype('str')
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
    vals = sub_df[x_var_column].dropna().unique()

    if x_var_column == 'ref_point_value':
        desired_order = np.arange(-200, 20, 10)
        # if any value not in desired_order, fallback to sorted uniques
        if not np.isin(vals, desired_order).all():
            ordered = np.sort(vals.astype(float))
        else:
            ordered = [v for v in desired_order if v in set(vals)]
    else:
        # try numeric sort when possible; otherwise fall back to string sort
        try:
            ordered = np.sort(vals.astype(float))
        except Exception:
            ordered = sorted(vals, key=lambda x: str(x))

    return dict(zip(ordered, range(len(ordered))))


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
                                                row_number=None, col_number=None):
    m = {str(k): v for k, v in x_labels_to_values_map.items()}

    fig.update_xaxes(tickvals=list(m.values()),
                     ticktext=list(m.keys()),
                     row=row_number, col=col_number)

    max_len = max((len(k) for k in m.keys()), default=0)
    if max_len > 30:
        fig.update_xaxes(tickangle=-90, row=row_number, col=col_number)
    elif max_len > 12:
        fig.update_xaxes(tickangle=-25, row=row_number, col=col_number)

    return fig


def _set_minimal_y_scale(fig, sub_df, y_var_column,
                         row_number=None, col_number=None):
    if {'se_lower', 'se_upper'}.issubset(sub_df.columns):
        y_lo = sub_df['se_lower'].astype(float)
        y_hi = sub_df['se_upper'].astype(float)
        min_y = np.nanmin(y_lo.to_numpy())
        max_y = np.nanmax(y_hi.to_numpy())
    else:
        y = sub_df[y_var_column].astype(float)
        # robust bounds
        min_y = np.nanpercentile(y, 2.5)
        max_y = np.nanpercentile(y, 97.5)

    if not np.isfinite(min_y) or not np.isfinite(max_y):
        min_y, max_y = 0.0, 1.0

    if np.isclose(min_y, max_y):
        pad = 0.5 if max_y == 0 else 0.1 * abs(max_y)
        min_y, max_y = min_y - pad, max_y + pad
    else:
        pad = 0.05 * (max_y - min_y)
        min_y, max_y = min_y - pad, max_y + pad

    fig.update_yaxes(range=[min_y, max_y], row=row_number, col=col_number)
    return fig


def make_plotly_plot_to_compare_two_sets_of_data(sub_df,
                                                 x_var_column,
                                                 y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                 var_to_determine_x_offset_direction='ref_columns_only',
                                                 title=None,
                                                 x_offset=0.0,
                                                 columns_to_find_unique_combinations_for_color=[],
                                                 columns_to_find_unique_combinations_for_line=[],
                                                 show_combo_legends=True,
                                                 fig=None,
                                                 row_number=None,
                                                 col_number=None,
                                                 ):
    if fig is None:
        fig = go.Figure()

    # --- Add CI bounds if only SE column is present ---
    se_col = None
    for cand in [f"{y_var_column}_se", f"{y_var_column}_ci_95", "se", "stderr", "std_error"]:
        if cand in sub_df.columns:
            se_col = cand
            break

    # Check if this is a BCa confidence interval (indicated by _ci_XX suffix)
    is_bca_ci = se_col is not None and '_ci_' in se_col

    if se_col is not None and not ({'se_lower', 'se_upper'} <= set(sub_df.columns)):
        sub_df = sub_df.copy()
        if is_bca_ci:
            # For BCa intervals, the column already contains the half-width of the CI
            sub_df['se_lower'] = sub_df[y_var_column] - sub_df[se_col]
            sub_df['se_upper'] = sub_df[y_var_column] + sub_df[se_col]
        else:
            # For standard errors, multiply by 1.96 to get 95% CI
            sub_df['se_lower'] = sub_df[y_var_column] - 1.96 * sub_df[se_col]
            sub_df['se_upper'] = sub_df[y_var_column] + 1.96 * sub_df[se_col]

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
                                 row_number=None, col_number=None):
    xs = sorted(sub_df['x_value_numeric'].unique())
    annotate_all = len(xs) <= 4
    rightmost = max(xs) if len(xs) else None

    for x_value in xs:
        subset = sub_df[sub_df['x_value_numeric'] == x_value]
        if subset.empty:
            continue
        min_row = subset.loc[subset[y_var_column].idxmin()]
        n = int(min_row.get('sample_size', np.nan)) if not pd.isna(
            min_row.get('sample_size', np.nan)) else None
        if n is None:
            continue

        text = f"n={n}" if annotate_all or x_value == rightmost else None
        if text:
            fig.add_trace(
                go.Scatter(
                    x=[min_row['x_value_numeric_with_offset']],
                    y=[min_row[y_var_column] -
                        max(0.01, 0.01 * abs(min_row[y_var_column]))],
                    mode='text',
                    text=[text],
                    textposition='bottom center',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row_number, col=col_number
            )
    return fig


def _make_hovertemplate(df, y_var_column, customdata_columns):
    """
    Create a hover template. Always shows group label (var_to_split_value),
    y value, sample size, and any extra columns with type-appropriate formatting.
    """
    cols = list(customdata_columns) if customdata_columns else []
    if 'sample_size' not in cols and 'sample_size' in df.columns:
        cols = ['sample_size'] + cols
    if 'var_to_split_value' in df.columns:
        cols = ['var_to_split_value'] + cols

    # Ensure integer display for sample sizes
    if 'sample_size' in df.columns:
        df['sample_size'] = df['sample_size'].astype('Int64')

    parts = [f"%{{customdata[0]}}<br>{y_var_column}: %{{y:.3f}}<br>"]
    for i, col in enumerate(cols[1:], start=1):
        if col not in df.columns:
            continue
        if is_string_dtype(df[col]) or df[col].dtype == object:
            parts.append(f"{col}: %{{customdata[{i}]}}<br>")
        elif is_integer_dtype(df[col]):
            parts.append(f"{col}: %{{customdata[{i}]:d}}<br>")
        elif is_numeric_dtype(df[col]):
            parts.append(f"{col}: %{{customdata[{i}]:.3f}}<br>")
        else:
            parts.append(f"{col}: %{{customdata[{i}]}}<br>")
    hovertemplate = "".join(parts) + "<extra></extra>"
    return hovertemplate, cols


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
                                     use_ribbons_to_replace_error_bars=True,
                                     row_number=None,
                                     col_number=None):

    # clean rows
    sub_df = sub_df.dropna(subset=[y_var_column]).copy()

    # marker sizes scaled & clipped
    max_n = max(int(sub_df.get('sample_size', pd.Series([1])).max()), 1)
    scale = 45.0 / max_n
    min_size, max_size = 6, 18

    hovertemplate, custom_cols = _make_hovertemplate(
        sub_df, y_var_column, customdata_columns)

    # only first subplot shows legend
    showlegend = not ((row_number and col_number) and not (
        row_number == 1 and col_number == 1))

    for line_color in sub_df['line_color'].unique():
        d = sub_df[sub_df['line_color'] == line_color].copy()
        name = d['var_to_split_value'].iloc[0] if 'var_to_split_value' in d.columns else str(
            line_color)

        # CI ribbon per series (if available)
        if {'se_lower', 'se_upper'}.issubset(d.columns):
            if use_ribbons_to_replace_error_bars:

                d = d.sort_values('x_value_numeric_with_offset')
                if len(d) == 1:
                    # Duplicate the row to force a ribbon
                    d = pd.concat([d, d], ignore_index=True)

                fig.add_trace(
                    go.Scatter(
                        x=d['x_value_numeric_with_offset'],
                        y=d['se_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_number, col=col_number
                )
                fig.add_trace(
                    go.Scatter(
                        x=d['x_value_numeric_with_offset'],
                        y=d['se_lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0,0,0,0.10)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_number, col=col_number
                )

            else:
                for x_value, y_upper, y_lower in zip(
                    d['x_value_numeric_with_offset'], d['se_upper'], d['se_lower']
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[x_value, x_value],
                            y=[y_lower, y_upper],
                            mode='lines',
                            line=dict(width=1.5, color=line_color),
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=row_number, col=col_number
                    )

        fig.add_trace(
            go.Scatter(
                x=d['x_value_numeric_with_offset'],
                y=d[y_var_column],
                name=name,
                mode='markers+lines',
                line=dict(color=line_color, width=1.5),
                marker=dict(
                    color=line_color,
                    size=np.clip(d['sample_size'].fillna(max_n).astype(
                        float) * scale, min_size, max_size),
                    opacity=0.8
                ),
                customdata=d[custom_cols].values,
                hovertemplate=hovertemplate,
                showlegend=showlegend,
            ),
            row=row_number, col=col_number
        )
    return fig


def connect_every_pair(fig, sub_df, y_var_column, customdata_columns, show_combo_legends=True,
                       row_number=None, col_number=None):

    hovertemplate, custom_cols = _make_hovertemplate(
        sub_df, y_var_column, customdata_columns)

    for pair_id in sub_df['pair_id'].unique():
        d = sub_df[sub_df['pair_id'] == pair_id].copy()
        if d.empty:
            continue
        d = d.sort_values('x_value_numeric_with_offset')
        if len(d) < 2:
            continue  # need at least 2 points to connect

        row0 = d.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=d['x_value_numeric_with_offset'],
                y=d[y_var_column],
                mode='lines',
                line=dict(color=row0['color'], width=1,
                          dash=row0['line_type']),
                marker=dict(size=0),
                customdata=d[custom_cols].values,
                hovertemplate=hovertemplate,
                showlegend=False,
                legendgroup=row0.get('color', 'group'),
                name=row0.get('unique_combination', None)
            ),
            row=row_number, col=col_number
        )

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


def compare_diff_in_abs_in_all_ref_pooled_median_info(sub_df, x, plotly=True):
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


def compare_test_and_ctrl_in_pooled_perc_info(sub_df, x, plotly=True):
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


def make_plotly_plot_to_compare_two_sets_of_data(sub_df,
                                                 x_var_column,
                                                 y_var_column='diff_in_abs_angle_to_nxt_ff_median',
                                                 var_to_determine_x_offset_direction='ref_columns_only',
                                                 title=None,
                                                 x_offset=0.0,
                                                 columns_to_find_unique_combinations_for_color=[],
                                                 columns_to_find_unique_combinations_for_line=[],
                                                 show_combo_legends=True,
                                                 fig=None,
                                                 row_number=None,
                                                 col_number=None,
                                                 ):
    if fig is None:
        fig = go.Figure()

    # --- (1) If only a single SE column exists, synthesize CI bounds ---
    se_col = None
    for cand in [f"{y_var_column}_se", f"{y_var_column}_ci_95", "se", "stderr", "std_error"]:
        if cand in sub_df.columns:
            se_col = cand
            break

    # Check if this is a BCa confidence interval (indicated by _ci_XX suffix)
    is_bca_ci = se_col is not None and '_ci_' in se_col

    if se_col is not None and not ({'se_lower', 'se_upper'} <= set(sub_df.columns)):
        sub_df = sub_df.copy()
        if is_bca_ci:
            # For BCa intervals, the column already contains the half-width of the CI
            sub_df['se_lower'] = sub_df[y_var_column] - sub_df[se_col]
            sub_df['se_upper'] = sub_df[y_var_column] + sub_df[se_col]
        else:
            # For standard errors, multiply by 1.96 to get 95% CI
            sub_df['se_lower'] = sub_df[y_var_column] - 1.96 * sub_df[se_col]
            sub_df['se_upper'] = sub_df[y_var_column] + 1.96 * sub_df[se_col]

    # --- (2) Usual preprocessing for colors/lines/hover ---
    rest_of_x_for_hoverdata = _find_rest_of_x_for_hoverdata(
        sub_df, x_var_column, y_var_column, var_to_determine_x_offset_direction)
    sub_df = _process_x_var_columns(sub_df, x_var_column)

    columns_to_find_unique_combinations_for_color = _process_columns_to_find_unique_combinations_for_color(
        columns_to_find_unique_combinations_for_color, x_var_column, rest_of_x_for_hoverdata)
    sub_df = process_variations_utils.assign_color_to_sub_df_based_on_unique_combinations(
        sub_df, columns_to_find_unique_combinations_for_color)
    sub_df = process_variations_utils.assign_line_type_to_sub_df_based_on_unique_combinations(
        sub_df, columns_to_find_unique_combinations_for_line)

    # --- (3) Build numeric x and (important) detect "single-bin" case ---
    x_labels_to_values_map = _find_x_labels_to_values_map(sub_df, x_var_column)
    sub_df = _add_x_value_numeric_to_sub_df(
        sub_df, x_var_column, x_labels_to_values_map, x_offset)

    # ===== Single-category path: render grouped BARs with CI =====
    # ===== Single-category path: render grouped BARs with CI =====
    if sub_df['x_value_numeric'].nunique() == 1:
        # choose group column (labels on x-axis)
        if 'var_to_split_value' in sub_df.columns:
            grp_col = 'var_to_split_value'
        elif 'test_or_control' in sub_df.columns:
            grp_col = 'test_or_control'
        else:
            grp_col = 'line_color'  # fallback

        # one row per group; stable order: control -> test if present
        d = (sub_df
             .dropna(subset=[y_var_column])
             .sort_values(grp_col)
             .groupby(grp_col, as_index=False).first())
        order = [g for g in ['control', 'test'] if g in set(d[grp_col])]
        order += [g for g in d[grp_col] if g not in order]
        d = d[d[grp_col].isin(order)].copy()

        # CI half-widths (from se_lower/upper if present; else from *_se etc.)
        if {'se_lower', 'se_upper'}.issubset(d.columns):
            d['_ci'] = (d['se_upper'] - d[y_var_column]).abs().clip(lower=0)
        else:
            se_col = next((c for c in [
                          f"{y_var_column}_se", "se", "stderr", "std_error"] if c in d.columns), None)
            d['_ci'] = 1.96 * d[se_col] if se_col is not None else np.nan

        # sample sizes if available (used in hover)
        def _get_n(row):
            if 'sample_size' in row and pd.notna(row['sample_size']):
                return int(row['sample_size'])
            gname = str(row[grp_col]).lower() + '_sample_size'
            return int(sub_df[gname].iloc[0]) if gname in sub_df.columns and pd.notna(sub_df[gname].iloc[0]) else None
        d['_n'] = d.apply(_get_n, axis=1)

        colors = {"control": "#F58518", "test": "#4C78A8"}

        fig = fig or go.Figure()
        for _, r in d.iterrows():
            show_ci = pd.notna(r['_ci']) and r['_ci'] > 0
            fig.add_trace(go.Bar(
                x=[str(r[grp_col])],
                y=[r[y_var_column]],
                name=str(r[grp_col]),
                marker_color=colors.get(r[grp_col], "gray"),
                error_y=dict(
                    type="data",
                    array=[float(r['_ci'])] if show_ci else [0.0],
                    visible=show_ci,
                    thickness=1.2,
                    width=3
                ),
                hovertemplate=(
                    f"{grp_col}: %{{x}}<br>"
                    f"{y_var_column}: %{{y:.4f}}"
                    + (f"<br>n: {r['_n']}" if r['_n'] is not None else "")
                    + "<extra></extra>"
                )
            ))

        fig.update_traces(
            width=0.4  # thinner bars (default is ~0.8)
        )

        fig.update_layout(
            barmode='group',
            xaxis_title=grp_col,
            yaxis_title=y_var_column,
            template="plotly_white",
            margin=dict(l=60, r=20, t=60, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            width=500,   # make figure narrower
            height=400,  # adjust height so it doesn’t look squashed
            title=dict(
                text=title or f"{y_var_column} (bar ± 95% CI)",
                x=0.5,         # center title
                y=0.98,        # a little below the top edge
                xanchor="center",
                yanchor="top",
                font=dict(size=15)
            )
        )

        # Auto y-limits with a little margin
        y_min = max(float(d[y_var_column].min() - d['_ci'].max() *
                    10 if d['_ci'].notna().any() else d[y_var_column].min()), 0)
        y_max = float(d[y_var_column].max() + d['_ci'].max() *
                      1.2 if d['_ci'].notna().any() else d[y_var_column].max())

        fig.update_yaxes(range=[y_min, y_max])

        if title is None:
            single_x_label = next(iter(x_labels_to_values_map.keys()), "")
            title = f"{y_var_column} (bar ± 95% CI) — {x_var_column}: {single_x_label}"
        fig.update_layout(title=title)

        fig.update_layout(showlegend=False)

        return fig

    # ===== Multi-category path: original vertical plot =====
    fig = plot_markers_for_data_comparison(
        fig, sub_df, rest_of_x_for_hoverdata, y_var_column, row_number=row_number, col_number=col_number)
    fig = connect_every_pair(
        fig, sub_df, y_var_column, rest_of_x_for_hoverdata,
        show_combo_legends=show_combo_legends, row_number=row_number, col_number=col_number)

    fig = _update_fig_based_on_x_labels_to_values_map(
        fig, x_labels_to_values_map, row_number=row_number, col_number=col_number)
    fig = _set_minimal_y_scale(
        fig, sub_df, y_var_column, row_number=row_number, col_number=col_number)
    fig = label_smallest_y_sample_size(
        fig, sub_df, y_var_column, row_number=row_number, col_number=col_number)

    if title is None:
        title = f'{y_var_column} vs {x_var_column}'
    fig.update_layout(title=title, xaxis_title=x_var_column,
                      yaxis_title=y_var_column)

    return fig
