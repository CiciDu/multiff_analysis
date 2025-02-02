
import sys
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from dash import html, dcc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import plotly.graph_objects as go
import matplotlib.colors as mplc


def add_values_and_marks_to_ref_point_info(ref_point_info):
    for key, value in ref_point_info.items():
        ref_point_all_values = np.arange(
            value['min'], value['max'], value['step']).tolist()
        ref_point_all_values = [round(i, 2) for i in ref_point_all_values]
        ref_point_info[key]['values'] = ref_point_all_values

        ref_point_marks_dict = {i: str(i) for i in ref_point_all_values}
        ref_point_info[key]['marks'] = ref_point_marks_dict

    return ref_point_info


def put_down_the_dropdown_menu_for_time_or_distance(time_or_distance):
    return html.Div(children=[
        html.Label(['Time or Distance'], style={
                   'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(
            id='time_or_distance',
            options=['time', 'distance'],
            value=time_or_distance,
            searchable=False,
            multi=False,
            # placeholder="Select a mode to determine the reference point",
        )
    ],
        style={'width': '40%', 'padding': '10px 10px 10px 10px', 'background-color': '#9FD4A3'})  # light green


def get_subset_of_combo_df(combo_df, hyperparameter_dict):
    combo_df = combo_df.copy()
    for key, value in hyperparameter_dict.items():
        if key in combo_df.columns:
            combo_df = combo_df[combo_df[key] == value]
        else:
            print('Warning: key ' + key +
                  ' in hyperparameter_dict is not in combo_df.columns')
    return combo_df


def make_a_slider_for_window_size(unique_window_sizes, id='window_size_slider'):
    unique_window_sizes = np.sort(unique_window_sizes)
    min = np.min(unique_window_sizes)
    max = np.max(unique_window_sizes)
    step = unique_window_sizes[1] - unique_window_sizes[0]

    window_size_marks_dict = {i: str(i) for i in unique_window_sizes}

    return html.Div(dcc.Slider(
        min=min,
        max=max,
        step=step,
        marks=window_size_marks_dict,
        id=id,
        value=unique_window_sizes[0],
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})


def make_a_slider_for_reference_point(ref_point_mode, ref_point_info, id='ref_point_slider'):
    min = ref_point_info[ref_point_mode]['min']
    max = ref_point_info[ref_point_mode]['max']
    step = ref_point_info[ref_point_mode]['step']

    ref_point_marks_dict = ref_point_info[ref_point_mode]['marks']
    ref_point_all_values = ref_point_info[ref_point_mode]['values']
    ref_point_value = ref_point_all_values[0]

    return html.Div(dcc.Slider(
        min=min,
        max=max,
        step=step,
        marks=ref_point_marks_dict,
        id=id,
        value=ref_point_value,
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})


def put_down_scatter_plot(fig):
    return html.Div([dcc.Graph(id='scatter_plot', figure=fig,
                               style={'width': '60%', 'padding': '0 0 0 0'})])


def generate_distribution_of_correlation_after_shuffling_d_heading(d_heading_nxt, d_heading_cur, d_heading_of_traj, sample_size):
    d_heading_nxt = np.array(d_heading_nxt)
    all_r_values = []
    all_p_values = []
    for i in range(sample_size):
        np.random.shuffle(d_heading_nxt)
        rel_heading_traj = d_heading_of_traj - d_heading_cur
        rel_heading_alt = d_heading_nxt - d_heading_cur
        rel_heading_traj = find_stops_near_ff_utils.confine_angle_to_within_one_pie(
            rel_heading_traj)
        rel_heading_alt = find_stops_near_ff_utils.confine_angle_to_within_one_pie(
            rel_heading_alt)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            rel_heading_alt, rel_heading_traj)
        all_r_values.append(r_value)
        all_p_values.append(p_value)
    all_r_values = np.array(all_r_values)
    all_p_values = np.array(all_p_values)
    return all_r_values, all_p_values


def generate_distribution_of_correlation_after_shuffling_nxt_ff_curv(nxt_ff_counted_df, cur_ff_counted_df, curv_of_traj_counted, use_curvature_to_ff_center, sample_size):
    nxt_ff_counted_df = nxt_ff_counted_df.copy()
    all_r_values = []
    all_p_values = []
    for i in range(sample_size):
        nxt_ff_counted_df[['curv_to_ff_center', 'optimal_curvature']] = nxt_ff_counted_df[[
            'curv_to_ff_center', 'optimal_curvature']].sample(frac=1).values
        traj_curv_counted, nxt_curv_counted = find_stops_near_ff_utils.find_relative_curvature(
            nxt_ff_counted_df, cur_ff_counted_df, curv_of_traj_counted, use_curvature_to_ff_center)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            nxt_curv_counted, traj_curv_counted)
        all_r_values.append(r_value)
        all_p_values.append(p_value)
    all_r_values = np.array(all_r_values)
    all_p_values = np.array(all_p_values)
    return all_r_values, all_p_values


def generate_possible_combos_for_planning(
        curv_of_traj_lower_end_based_on_mode={'time': np.arange(-1.9, 0, 0.2),
                                              'distance': np.arange(-190, 0, 20)},
        curv_of_traj_upper_end_based_on_mode={'time': np.arange(0.1, 2.1, 0.2),
                                              'distance': np.arange(10, 210, 20)},
        ref_point_value_based_on_mode={'time': np.arange(0.1, 2.1, 0.2),
                                       'distance': np.arange(10, 210, 20)},
):
    grid = dict(ref_point_mode=['time', 'distance'],
                curv_of_traj_mode=['time', 'distance'])
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combo_df = pd.DataFrame(combinations)

    all_combo_df = pd.DataFrame()
    for index, row in combo_df.iterrows():
        ref_point_mode = row['ref_point_mode']
        curv_of_traj_mode = row['curv_of_traj_mode']
        ref_point_value = np.round(
            ref_point_value_based_on_mode[ref_point_mode], 2)
        curv_of_traj_lower_end = np.round(
            curv_of_traj_lower_end_based_on_mode[curv_of_traj_mode], 2)
        curv_of_traj_upper_end = np.round(
            curv_of_traj_upper_end_based_on_mode[curv_of_traj_mode], 2)

        grid = {'ref_point_value': ref_point_value,
                'curv_of_traj_lower_end': curv_of_traj_lower_end,
                'curv_of_traj_upper_end': curv_of_traj_upper_end}

        keys, values = zip(*grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        temp_df = pd.DataFrame(combinations)
        temp_df['ref_point_mode'] = ref_point_mode
        temp_df['curv_of_traj_mode'] = curv_of_traj_mode
        all_combo_df = pd.concat([all_combo_df, temp_df], axis=0)
    all_combo_df = all_combo_df.reset_index(drop=True)
    return all_combo_df


def add_columns_of_dummy_variables_to_all_combos_df(all_combo_df, column_names=['truncate_curv_of_traj_by_time_of_capture']):
    for column_name in column_names:
        all_combo_df[column_name] = False
        all_combo_df2 = all_combo_df.copy()
        all_combo_df2[column_name] = True
        all_combo_df = pd.concat([all_combo_df, all_combo_df2], axis=0)
        all_combo_df = all_combo_df.reset_index(drop=True)
    return all_combo_df


def get_heading_df_and_curv_df(tested_combo_df):
    heading_df = tested_combo_df.copy()
    heading_df.rename(columns={'heading_r': 'r',
                               'heading_r_z_score': 'r_z_score',
                               'shuffled_heading_r_mean': 'shuffled_r_mean',
                               'shuffled_heading_r_std': 'shuffled_r_std',
                               }, inplace=True)

    curv_df = tested_combo_df.copy()
    curv_df.rename(columns={'curv_r': 'r',
                            'curv_r_z_score': 'r_z_score',
                            'shuffled_curv_r_mean': 'shuffled_r_mean',
                            'shuffled_curv_r_std': 'shuffled_r_std',
                            }, inplace=True)
    return heading_df, curv_df


def find_ref_point_marks_dict(ref_point_all_values):
    ref_point_marks_dict = {}
    for value in ref_point_all_values:
        value = round(value.item(), 2)
        ref_point_marks_dict[value] = str(value)
    return ref_point_marks_dict


def plot_tested_heading_df_or_curv_df(sub_df):

    # x and y will be lower and upper window for curv of traj mode respectively, and then the color will show info about r (maybe how many std is r above shuffled r)
    fig = go.Figure()

    plot_to_add = go.Scatter(x=sub_df['curv_of_traj_lower_end'],
                             y=sub_df['curv_of_traj_upper_end'],
                             mode='markers',
                             marker=dict(color=sub_df['r_z_score'], colorscale='Viridis', size=10,
                                         colorbar=dict(thickness=20, ticklen=4, title='z-score of r', titleside='top',
                                                       titlefont=dict(
                                                           size=14, family='Arial', color='black'),
                                                       tickfont=dict(
                                                           size=14, family='Arial', color='black'),
                                                       borderwidth=0, bordercolor='white',
                                                       )
                                         ),
                             customdata=sub_df[[
                                 'r', 'shuffled_r_mean', 'shuffled_r_std']],
                             hovertemplate='r: %{customdata[0]:.2f}<br>' +
                             'shuffled r mean: %{customdata[1]:.2f}<br>' +
                             'shuffled r std: %{customdata[2]:.2f}<br>' +
                             'r z-score: %{marker.color}<extra></extra>',
                             )
    fig.add_trace(plot_to_add)
    fig.update_layout(title='Compare Correlations',
                      xaxis_title='Curv of traj lower end', yaxis_title='Curv of traj upper end')
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_layout(
        autosize=False,
        width=600,
        height=500,
    )

    return fig


def process_tested_combo_df(tested_combo_df):
    tested_combo_df['shuffled_heading_r_mean_plus_2_std'] = tested_combo_df['shuffled_heading_r_mean'] + \
        2*tested_combo_df['shuffled_heading_r_std']
    tested_combo_df['shuffled_curv_r_mean_plus_2_std'] = tested_combo_df['shuffled_curv_r_mean'] + \
        2*tested_combo_df['shuffled_curv_r_std']

    tested_combo_df['heading_r_z_score'] = (
        tested_combo_df['heading_r'] - tested_combo_df['shuffled_heading_r_mean'])/tested_combo_df['shuffled_heading_r_std']
    tested_combo_df['curv_r_z_score'] = (
        tested_combo_df['curv_r'] - tested_combo_df['shuffled_curv_r_mean'])/tested_combo_df['shuffled_curv_r_std']

    tested_combo_df['heading_r_z_score'] = np.round(
        tested_combo_df['heading_r_z_score'], 2)
    tested_combo_df['curv_r_z_score'] = np.round(
        tested_combo_df['curv_r_z_score'], 2)

    tested_combo_df['window_size'] = tested_combo_df['curv_of_traj_upper_end'] - \
        tested_combo_df['curv_of_traj_lower_end']
    tested_combo_df['window_size'] = np.round(
        tested_combo_df['window_size'], 2)


def plot_tested_heading_df_or_curv_df2(sub_df,
                                       hyperparameter_dict,
                                       ):

    curv_of_traj_mode = hyperparameter_dict['curv_of_traj_mode']
    ref_point_mode = hyperparameter_dict['ref_point_mode']
    window_size = hyperparameter_dict['window_size']
    ref_point_value = hyperparameter_dict['ref_point_value']
    use_curvature_to_ff_center = hyperparameter_dict['use_curvature_to_ff_center']
    heading_instead_of_curv = hyperparameter_dict['heading_instead_of_curv']

    # given a subset, take out the rows belonging to the same window size
    sub_df2 = sub_df[sub_df['window_size'] == window_size].copy()
    sub_df2.sort_values(by='curv_of_traj_lower_end', inplace=True)
    sub_df2.reset_index(drop=True, inplace=True)
    sub_df2['counter'] = range(len(sub_df2))

    # get rgba colors
    num_lines = len(sub_df2)
    cmap = plt.get_cmap('viridis', num_lines)
    norm = mplc.Normalize(
        vmin=sub_df2['r_z_score'].min(), vmax=sub_df2['r_z_score'].max())

    if curv_of_traj_mode == ref_point_mode:
        ref_point_value_to_add = ref_point_value
    else:
        ref_point_value_to_add = 0

    # make a plotly line plot
    fig_lines = go.Figure()
    for index, row in sub_df2.iterrows():
        customdata = np.array(
            [[row['r_z_score']]*100, [row['curv_of_traj_lower_end']]*100, [row['curv_of_traj_upper_end']]*100]).T
        fig_lines.add_trace(go.Scatter(x=np.linspace(row['curv_of_traj_lower_end'], row['curv_of_traj_upper_end'], 100)+ref_point_value_to_add,
                                       y=[index]*100,
                                       mode='lines',
                                       line={
                                           'color': f'rgba{cmap(norm(row["r_z_score"]))}'},
                                       showlegend=False,
                                       hovertemplate=('z-score of R: %{customdata[0]:.2f} <br>' +
                                                      'curv of traj lower end: %{customdata[1]:.2f} <br>' +
                                                      'curv of traj upper end: %{customdata[2]:.2f} <br>' +
                                                      'x: %{x:.2f} <extra></extra>'),
                                       customdata=customdata
                                       ))
        # also annotate z-score of R above each line
        # make the color of the word red if the number is above 2
        if row['r_z_score'] > 2:
            fig_lines.add_annotation(x=row['curv_of_traj_upper_end']+ref_point_value_to_add, y=index +
                                     0.2, text=f'{row["r_z_score"]:.2f}', showarrow=False, font=dict(size=10, color='red'))
        else:
            fig_lines.add_annotation(x=row['curv_of_traj_upper_end']+ref_point_value_to_add,
                                     y=index+0.2, text=f'{row["r_z_score"]:.2f}', showarrow=False, font=dict(size=10))
        # fig_lines.add_annotation(x=row['curv_of_traj_upper_end']+ref_point_value_to_add, y=index+0.2, text=f'{row["r_z_score"]:.2f}', showarrow=False, font=dict(size=10))

    # plot a vertical line at 0
    fig_lines.add_shape(type="line", x0=0, y0=-1, x1=0, y1=num_lines,
                        line=dict(color="black", width=1, dash="dashdot"))

    # use heading_instead_of_curv and use_curvature_to_ff_center to make a title
    if heading_instead_of_curv:
        title = 'Heading'
    else:
        title = 'Curv'

    if use_curvature_to_ff_center:
        title += ' to FF Center'
    else:
        title += ' to FF'

    # plot a vertical line at ref_point_value with hoverdata that's ref_point_value
    if curv_of_traj_mode == ref_point_mode:
        ref_point_line = go.Scatter(x=[ref_point_value]*1000,
                                    y=np.linspace(-1, num_lines, 1000),
                                    mode='lines',
                                    name='ref_point_value',
                                    line=dict(color='orange',
                                              width=1, dash='dashdot'),
                                    hovertemplate='ref_point_value: %{x:.2f}',
                                    showlegend=True)
        fig_lines.add_trace(ref_point_line)
    else:
        # add a subtitle that prints the ref point value
        # add a line break to title
        title += ' at \"reference point value\": ' + str(ref_point_value)
        # title = f'Reference point value based on {ref_point_mode}: {ref_point_value}'
        if ref_point_mode == 'time':
            title += ' s'
        else:
            title += ' cm'

    fig_lines.update_layout(title=title)

    # add a colorbar
    colorbar_trace = go.Scatter(x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(colorscale='Viridis',
                                            showscale=True,
                                            cmin=sub_df2['r_z_score'].min(),
                                            cmax=sub_df2['r_z_score'].max(),
                                            colorbar=dict(thickness=5, outlinewidth=0)),
                                hoverinfo='none',
                                showlegend=False
                                )
    fig_lines.add_trace(colorbar_trace)

    # add x and y axis labels
    fig_lines.update_layout(xaxis_title=curv_of_traj_mode,
                            yaxis=dict(ticks='', showticklabels=False))

    # edit the position of legend
    fig_lines.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # make the plot smaller
    fig_lines.update_layout(
        autosize=False,
        width=500,
        height=400,
    )

    # decrease margin but not to 0
    fig_lines.update_layout(margin=dict(l=10, r=10, b=10, t=40, pad=5))

    return fig_lines
