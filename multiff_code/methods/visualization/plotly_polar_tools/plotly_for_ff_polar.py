from visualization.plotly_polar_tools import plotly_for_trajectory_polar

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


def plot_fireflies(fig, ff_df, columns_for_annotation=['ff_distance_to_monkey_then', 'ff_angle_to_monkey_then', 'time_since_last_vis', 'duration_of_last_vis_period'],
                   color='time_since_last_vis', range_color=[0, 3], symbol='group', size='time_since_last_vis', size_max=15,
                   symbol_map={}, additional_customdata_columns=[]):

    default_symbol_map = {'Original': 'circle-dot', 'Alternative': 'star-dot', 'More': 'diamond-open-dot',
                          'Original-Past': 'circle', 'Alternative-Past': 'star', 'Original-Future': 'circle', 'Alternative-Future': 'star',
                          'Past': 'cross', 'Future': 'cross-open', 'Present': 'circle'}
    for key in default_symbol_map.keys():
        if key not in symbol_map.keys():
            symbol_map[key] = default_symbol_map[key]

    plot_to_add = px.scatter_polar(ff_df, r='ff_distance', theta='ff_angle',
                                   direction="counterclockwise",
                                   start_angle=90,
                                   template='plotly_white',
                                   color_continuous_scale=px.colors.sequential.Sunset_r,
                                   range_color=range_color,
                                   color=color,
                                   size=size,
                                   size_max=size_max,
                                   symbol=symbol,
                                   symbol_map=symbol_map,
                                   )

    print('The names of the added plotly traces are:')
    for data in plot_to_add.data:
        print(data['name'])
    num_added_traces = len(plot_to_add.data)

    if fig is None:
        fig = plot_to_add
    else:
        fig.add_traces(plot_to_add.data)

    fig.update_traces(marker=dict(line=dict(width=2.5)),
                      selector=dict(mode='markers'))

    fig, customdata_columns = update_firefly_hovertemplate_based_on_group(fig, ff_df, num_added_traces=num_added_traces, columns_for_annotation=columns_for_annotation,
                                                                          color=color, symbol=symbol, additional_customdata_columns=additional_customdata_columns)
    return fig, customdata_columns


def update_firefly_hovertemplate_based_on_group(fig, ff_df, num_added_traces, additional_update_kwargs={}, color='group', symbol='group',
                                                columns_for_annotation=[
                                                    'ff_distance_to_monkey_then', 'ff_angle_to_monkey_then', 'time_since_last_vis'],
                                                additional_customdata_columns=[]):

    annotation_names = {'ff_distance': 'ff distance (cm)',
                        'ff_angle': 'ff angle (deg)',
                        'ff_distance_to_monkey_then': 'ff distance to monkey then (cm)',
                        'ff_angle_to_monkey_then': 'ff angle to monkey then (deg)',
                        'time_since_last_vis': 'time since last visible (s)',
                        'time_from_now': 'time from now (s)',
                        'duration_of_last_vis_period': 'duration of last visible period (s)',
                        'curv_diff': 'curvature difference (deg/cm)',
                        'abs_curv_diff': 'abs curvature difference (deg/cm)',
                        'time': 'time (s)',
                        'group': 'group',
                        'subgroup': 'subgroup',
                        'ff_number': 'ff number',
                        }

    print('columns_for_annotation for ff data', columns_for_annotation)

    hovertemplate = ''
    for counter in range(len(columns_for_annotation)):
        # if subgroup is the first column, then we bold it and not add the annotation name
        if (columns_for_annotation[counter] == 'subgroup') & (counter == 0):
            hovertemplate = hovertemplate + \
                '<b>%{customdata[' + f'{counter}' + ']}</b><br>'
        elif columns_for_annotation[counter] in ['group']:
            hovertemplate = hovertemplate + \
                annotation_names[columns_for_annotation[counter]
                                 ] + ': %{customdata[' + f'{counter}' + ']}<br>'
        else:  # the formating is different
            # hovertemplate = hovertemplate + '<b>' + annotation_names[columns_for_annotation[counter]] + ': %{customdata[' + f'{counter}' + ']:.2f}</b><br>'
            hovertemplate = hovertemplate + \
                annotation_names[columns_for_annotation[counter]] + \
                ': %{customdata[' + f'{counter}' + ']:.2f}<br>'

    # combine columns_for_annotation and additional_customdata_columns
    customdata_columns = columns_for_annotation + additional_customdata_columns

    columns_for_levels = []
    # suppose only symbol or color can be categorical data, and at least one of them is categorical data
    # then symbol is used to define levels
    if (ff_df[symbol].dtype != 'float') & (ff_df[symbol].dtype != 'int'):
        columns_for_levels.append(symbol)
        symbol_levels = ff_df[symbol].unique().tolist()
        levels = [[level] for level in symbol_levels]
        level_names = symbol_levels.copy()

    # then color defines levels too
    if (ff_df[color].dtype != 'float') & (ff_df[color].dtype != 'int'):
        levels_color = ff_df[color].unique().tolist()
        if color == symbol:
            pass  # symbol and color are the same, so no need to add color as a categorical data
        else:
            columns_for_levels.insert(0, color)
            if symbol not in columns_for_levels:
                levels = [[level] for level in levels_color]
                level_names = levels_color.copy()
            else:  # symbol and color are both in columns_for_levels, and they are different
                levels = []
                level_names = []
                for j in range(len(levels_color)):
                    # make outer product of symbol_levels and levels_color
                    for i in range(len(symbol_levels)):
                        levels.append([levels_color[j], symbol_levels[i]])
                        new_level_name = levels_color[j] + \
                            ', ' + symbol_levels[i]
                        level_names.append(new_level_name)
    level_names = np.array(level_names)
    print('level_names are:', level_names)

    for i in range(num_added_traces):
        current_data_name = fig.data[-(num_added_traces-i)].name
        if current_data_name in level_names:
            index = np.where(level_names == current_data_name)[0][0]
            current_levels = levels[index]
            ff_df_sub = ff_df.copy()
            for j in range(len(columns_for_levels)):
                ff_df_sub = ff_df_sub[(
                    ff_df_sub[columns_for_levels[j]] == current_levels[j])].copy()
            fig.data[-(num_added_traces-i)].update(customdata=ff_df_sub[customdata_columns].values,
                                                   hovertemplate=hovertemplate,
                                                   **additional_update_kwargs)
        else:
            print('current_data_name:', current_data_name)
            raise ValueError('The current level name ' +
                             current_data_name + ' is not in level_names.')

    # all_input_columns_to_use = ff_df[columns_for_annotation]
    # for i, level in enumerate(levels):
    #     if fig.data[-(len(levels)-i)].name == level:
    #         fig.data[-(len(levels)-i)].update(customdata=all_input_columns_to_use[ff_df['group']==level].values,
    #                                                     hovertemplate=hovertemplate,
    #                                                     **additional_update_kwargs)
    #     else:
    #         print('data name:', fig.data[-(len(levels)-i)].name)
    #         print('level:', level)
    #         raise ValueError('The order of traces is not correct.')

    return fig, customdata_columns


def separate_ff_info_from_past_present_and_future(i, original_ff_df, point_index_array, time_since_last_vis_cap=3):
    corr_point_index = point_index_array[i]
    current_ff_df = original_ff_df[original_ff_df['point_index']
                                   == corr_point_index].copy()
    current_ff_df['time_label'] = 'Present'
    current_ff_df.sort_values(by='group', ascending=False, inplace=True)
    # eliminate placeholders
    current_ff_df = current_ff_df[current_ff_df['ff_distance'] < 400].copy()
    if 'ff_number' not in current_ff_df.columns:
        current_ff_df['ff_number'] = np.arange(
            current_ff_df.shape[0])   # assign each ff a number
    current_ff_df['ff_distance_to_monkey_then'] = current_ff_df['ff_distance'].values
    current_ff_df['ff_angle_to_monkey_then'] = current_ff_df['ff_angle'].values

    last_seen_ff_df = current_ff_df[['point_index', 'group', 'ff_number', 'last_seen_ff_distance', 'last_seen_ff_angle', 'last_seen_curv_diff', 'time_since_last_vis', 'duration_of_last_vis_period',
                                     'distance_from_monkey_now_to_ff_when_ff_last_seen', 'angle_from_monkey_now_to_ff_when_ff_last_seen', 'curv_diff_from_monkey_now_to_ff_when_ff_last_seen']].copy()
    last_seen_ff_df['time_label'] = 'Past'
    last_seen_ff_df = last_seen_ff_df[last_seen_ff_df['time_since_last_vis']
                                      < time_since_last_vis_cap].copy()
    last_seen_ff_df.rename(columns={'distance_from_monkey_now_to_ff_when_ff_last_seen': 'ff_distance',
                                    'angle_from_monkey_now_to_ff_when_ff_last_seen': 'ff_angle',
                                    'curv_diff_from_monkey_now_to_ff_when_ff_last_seen': 'curv_diff',
                                    'last_seen_ff_distance': 'ff_distance_to_monkey_then',
                                    'last_seen_ff_angle': 'ff_angle_to_monkey_then'}, inplace=True)

    next_seen_ff_df = current_ff_df[['point_index', 'group', 'ff_number', 'next_seen_ff_distance', 'next_seen_ff_angle', 'next_seen_curv_diff', 'time_since_last_vis', 'duration_of_last_vis_period',
                                     'distance_from_monkey_now_to_ff_when_ff_next_seen', 'angle_from_monkey_now_to_ff_when_ff_next_seen', 'curv_diff_from_monkey_now_to_ff_when_ff_next_seen']].copy()
    next_seen_ff_df['time_label'] = 'Future'
    next_seen_ff_df.rename(columns={'distance_from_monkey_now_to_ff_when_ff_next_seen': 'ff_distance',
                                    'angle_from_monkey_now_to_ff_when_ff_next_seen': 'ff_angle',
                                    'curv_diff_from_monkey_now_to_ff_when_ff_next_seen': 'curv_diff',
                                    'next_seen_ff_distance': 'ff_distance_to_monkey_then',
                                    'next_seen_ff_angle': 'ff_angle_to_monkey_then'}, inplace=True)

    essential_columns = ['point_index', 'group', 'ff_number', 'time_label', 'ff_distance', 'ff_angle', 'ff_distance_to_monkey_then',
                         'ff_angle_to_monkey_then', 'curv_diff', 'time_since_last_vis', 'duration_of_last_vis_period']
    main_ff_df = pd.concat([current_ff_df[essential_columns], last_seen_ff_df[essential_columns],
                           next_seen_ff_df[essential_columns]], axis=0).reset_index(drop=True)
    main_ff_df['abs_curv_diff'] = np.abs(main_ff_df['curv_diff'])
    main_ff_df['subgroup'] = main_ff_df['group'] + \
        '-' + main_ff_df['time_label']
    main_ff_df.sort_values(by='group', ascending=False, inplace=True)

    main_ff_df['time_since_last_vis'] = main_ff_df['time_since_last_vis'].clip(
        upper=time_since_last_vis_cap)
    main_ff_df['ff_angle'] = main_ff_df['ff_angle'] * 180 / math.pi
    main_ff_df['ff_angle_to_monkey_then'] = main_ff_df['ff_angle_to_monkey_then'] * 180 / math.pi
    main_ff_df['curv_diff'] = main_ff_df['curv_diff'] * 180 / math.pi

    # print('Note: ff_angle has been from radians converted to degrees')
    # print('Note: curv_diff has been from radians/cm converted to degrees/cm')

    monkey_info_for_ff_in_past_or_future = plotly_for_trajectory_polar.find_monkey_info_for_ff_in_past_or_future(
        current_ff_df, last_seen_ff_df, next_seen_ff_df)

    return main_ff_df, monkey_info_for_ff_in_past_or_future

