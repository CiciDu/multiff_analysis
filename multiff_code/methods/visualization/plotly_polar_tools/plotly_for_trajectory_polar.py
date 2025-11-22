
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import math

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)




def plot_trajectory_data(fig, traj_df_to_use, color_discrete_sequence=['red', 'blue'], additional_update_kwargs={},
                         columns_for_annotation=['monkey_distance', 'time']):

    fig.add_traces(
        list(px.scatter_polar(traj_df_to_use,
                              r='monkey_distance',
                              theta='monkey_angle_to_origin',
                              color='whether_stopped',
                              color_discrete_sequence=color_discrete_sequence,
                              color_discrete_map={
                                  'has stop': color_discrete_sequence[0], 'no stop': color_discrete_sequence[1]},
                              # match color to the variable whether_stopped
                              ).select_traces()))

    # hovertemplate = '<b>monkey distance: %{customdata[0]:.2f}</b><br>'+\
    #                     '<b>monkey angle: %{customdata[1]:.2f}</b><br>'+\
    #                     '<b>whether stopped: %{customdata[2]}</b><br>'+\
    #                     '<b>time: %{customdata[3]}</b><br>'

    annotation_names = {'monkey_distance': 'monkey distance (cm)',
                        'monkey_angle': 'monkey angle (deg)',
                        'whether_stopped': 'whether stopped',
                        'time': 'relative time (s)'}

    print('columns_for_annotation:', columns_for_annotation)

    hovertemplate = ''
    for counter in range(len(columns_for_annotation)):
        if columns_for_annotation[counter] in ['whether_stopped', 'time']:
            hovertemplate = hovertemplate + \
                annotation_names[columns_for_annotation[counter]
                                 ] + ': %{customdata[' + f'{counter}' + ']}<br>'
        else:  # the formating is different
            # hovertemplate = hovertemplate + '<b>' + annotation_names[columns_for_annotation[counter]] + ': %{customdata[' + f'{counter}' + ']:.2f}</b><br>'
            hovertemplate = hovertemplate + \
                annotation_names[columns_for_annotation[counter]] + \
                ': %{customdata[' + f'{counter}' + ']:.2f}<br>'

    whether_stopped_levels = traj_df_to_use['whether_stopped'].unique()
    for i, level in enumerate(whether_stopped_levels):
        if fig.data[-(len(whether_stopped_levels)-i)].name == level:
            fig.data[-(len(whether_stopped_levels)-i)].update(customdata=traj_df_to_use.loc[traj_df_to_use['whether_stopped'] == level, columns_for_annotation].values,
                                                              hovertemplate=hovertemplate,
                                                              **additional_update_kwargs)
            # also update the marker size
            fig.update_traces(marker=dict(size=10, opacity=0.7),
                              selector=({'name': level}))
            # fig.for_each_trace(
            #     lambda trace: trace.update(marker_symbol="square") if trace.name == "setosa" else (),
            # )
        else:
            raise ValueError('The order of traces is not correct.')
    return fig


def plot_monkey_info_for_ff_in_past_or_future(fig, monkey_info_for_ff_in_past_or_future,
                                              columns_for_annotation=[
                                                  'distance_from_monkey_now_to_ff_then', 'angle_from_monkey_now_to_ff_then', 'time_from_now'],
                                              additional_update_kwargs={}):

    for index, row in monkey_info_for_ff_in_past_or_future.iterrows():
        fig_traces = px.line_polar(row,
                                   r=np.array([row['distance_from_monkey_now_to_ff_then'],
                                              row['distance_from_monkey_now_to_monkey_then']]),
                                   theta=np.array(
                                       [row['angle_from_monkey_now_to_ff_then'], row['angle_from_monkey_now_to_monkey_then']]),
                                   )

        fig.add_traces(list(fig_traces.select_traces()))

        for i in fig_traces.data:
            i.name = 'monkey_info_for_ff_in_past_or_future'

    annotation_names = {'distance_from_monkey_now_to_monkey_then': 'distance from monkey now to monkey then (cm)',
                        'angle_from_monkey_now_to_monkey_then': 'angle from monkey now to monkey then (deg)',
                        'distance_from_monkey_now_to_ff_then': 'distance from monkey now to ff then (cm)',
                        'angle_from_monkey_now_to_ff_then': 'angle from monkey now to ff then (deg)',
                        'time_from_now': 'time from now (s)',
                        }

    print('columns_for_annotation for monkey data:', columns_for_annotation)

    hovertemplate = ''
    for counter in range(len(columns_for_annotation)):
        # hovertemplate = hovertemplate + '<b>' + annotation_names[columns_for_annotation[counter]] + ': %{customdata[' + f'{counter}' + ']:.2f}</b><br>'
        hovertemplate = hovertemplate + \
            annotation_names[columns_for_annotation[counter]] + \
            ': %{customdata[' + f'{counter}' + ']:.2f}<br>'

    fig.for_each_trace(
        lambda trace: trace.update(customdata=row[columns_for_annotation].values,
                                   hovertemplate=hovertemplate,
                                   **additional_update_kwargs) if trace.name == "monkey_info_for_ff_in_past_or_future" else (),
    )

    return fig


def make_current_traj_df(current_traj_points, current_stops, trajectory_features):
    current_traj_df = pd.DataFrame(
        current_traj_points.T, columns=trajectory_features)
    current_traj_df['monkey_angle_to_origin'] = current_traj_df['monkey_angle_to_origin']
    current_traj_df['whether_stopped'] = current_stops
    current_traj_df.loc[current_traj_df['whether_stopped']
                        == 1, 'whether_stopped'] = 'has stop'
    current_traj_df.loc[current_traj_df['whether_stopped']
                        == 0, 'whether_stopped'] = 'no stop'
    return current_traj_df


def find_monkey_info_when_ff_last_seen(current_ff_df, last_seen_ff_numbers):
    current_ff_df = current_ff_df[current_ff_df['ff_number'].isin(
        last_seen_ff_numbers)].copy()
    monkey_info_when_ff_last_seen = current_ff_df[['point_index', 'group', 'ff_number', 'last_seen_monkey_x', 'last_seen_monkey_y', 'time_since_last_vis',
                                                   'distance_from_monkey_now_to_monkey_when_ff_last_seen', 'angle_from_monkey_now_to_monkey_when_ff_last_seen',
                                                   'distance_from_monkey_now_to_ff_when_ff_last_seen', 'angle_from_monkey_now_to_ff_when_ff_last_seen']].copy()

    monkey_info_when_ff_last_seen.rename(columns={'last_seen_monkey_x': 'monkey_x',
                                                  'last_seen_monkey_y': 'monkey_y',
                                                  'time_since_last_vis': 'time_from_now',
                                                  'last_seen_ff_distance': 'distance_from_monkey_then_to_ff_then',
                                                  'last_seen_ff_angle': 'angle_from_monkey_then_to_ff_then',
                                                  'last_seen_curv_diff': 'curv_diff_from_monkey_then_to_ff_then',
                                                  'distance_from_monkey_now_to_monkey_when_ff_last_seen': 'distance_from_monkey_now_to_monkey_then',
                                                  'angle_from_monkey_now_to_monkey_when_ff_last_seen': 'angle_from_monkey_now_to_monkey_then',
                                                  'distance_from_monkey_now_to_ff_when_ff_last_seen': 'distance_from_monkey_now_to_ff_then',
                                                  'angle_from_monkey_now_to_ff_when_ff_last_seen': 'angle_from_monkey_now_to_ff_then'
                                                  }, inplace=True)
    monkey_info_when_ff_last_seen['angle_from_monkey_now_to_ff_then'] = monkey_info_when_ff_last_seen['angle_from_monkey_now_to_ff_then']
    monkey_info_when_ff_last_seen['angle_from_monkey_now_to_monkey_then'] = monkey_info_when_ff_last_seen['angle_from_monkey_now_to_monkey_then']
    monkey_info_when_ff_last_seen['time_label'] = 'Past'
    return monkey_info_when_ff_last_seen


def find_monkey_info_when_ff_next_seen(current_ff_df, next_seen_ff_numbers):
    current_ff_df = current_ff_df[current_ff_df['ff_number'].isin(
        next_seen_ff_numbers)].copy()
    monkey_info_when_ff_next_seen = current_ff_df[['point_index', 'group', 'ff_number', 'next_seen_monkey_x', 'next_seen_monkey_y',
                                                   'distance_from_monkey_now_to_monkey_when_ff_next_seen', 'angle_from_monkey_now_to_monkey_when_ff_next_seen',
                                                   'distance_from_monkey_now_to_ff_when_ff_next_seen', 'angle_from_monkey_now_to_ff_when_ff_next_seen']].copy()
    monkey_info_when_ff_next_seen.rename(columns={'next_seen_monkey_x': 'monkey_x',
                                                  'next_seen_monkey_y': 'monkey_y',
                                                  'next_seen_ff_distance': 'distance_from_monkey_then_to_ff_then',
                                                  'next_seen_ff_angle': 'angle_from_monkey_then_to_ff_then',
                                                  'next_seen_curv_diff': 'curv_diff_from_monkey_then_to_ff_then',
                                                  'distance_from_monkey_now_to_monkey_when_ff_next_seen': 'distance_from_monkey_now_to_monkey_then',
                                                  'angle_from_monkey_now_to_monkey_when_ff_next_seen': 'angle_from_monkey_now_to_monkey_then',
                                                  'distance_from_monkey_now_to_ff_when_ff_next_seen': 'distance_from_monkey_now_to_ff_then',
                                                  'angle_from_monkey_now_to_ff_when_ff_next_seen': 'angle_from_monkey_now_to_ff_then'}, inplace=True)
    monkey_info_when_ff_next_seen['angle_from_monkey_now_to_ff_then'] = monkey_info_when_ff_next_seen['angle_from_monkey_now_to_ff_then']
    monkey_info_when_ff_next_seen['angle_from_monkey_now_to_monkey_then'] = monkey_info_when_ff_next_seen['angle_from_monkey_now_to_monkey_then']
    monkey_info_when_ff_next_seen['time_label'] = 'Future'
    return monkey_info_when_ff_next_seen


def find_monkey_info_for_ff_in_past_or_future(current_ff_df, last_seen_ff_df, next_seen_ff_df):
    last_seen_ff_numbers = last_seen_ff_df['ff_number'].values
    monkey_info_when_ff_last_seen_df = find_monkey_info_when_ff_last_seen(
        current_ff_df, last_seen_ff_numbers)
    next_seen_ff_numbers = next_seen_ff_df['ff_number'].values
    monkey_info_when_ff_next_seen_df = find_monkey_info_when_ff_next_seen(
        current_ff_df, next_seen_ff_numbers)
    monkey_info_for_ff_in_past_or_future = pd.concat(
        [monkey_info_when_ff_last_seen_df, monkey_info_when_ff_next_seen_df], axis=0).reset_index(drop=True)
    return monkey_info_for_ff_in_past_or_future
