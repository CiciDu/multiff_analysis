from decision_making_analysis.decision_making import plot_decision_making
from visualization.plotly_polar_tools import plotly_for_trajectory_polar, plotly_for_ff_polar

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_one_plotly_polar_plot(ff_df=None, current_traj_df=None, current_more_traj_df=None, real_label=None, predicted_label=None,
                               add_trajectory=True, add_colorbar=True, add_legend=True, add_real_and_predicted_labels=True, show_more_ff=True,
                               columns_for_annotation=['subgroup', 'ff_distance_to_monkey_then', 'ff_angle_to_monkey_then',
                                                       'curv_diff', 'time_since_last_vis', 'duration_of_last_vis_period', 'time_till_next_visible'],
                               additional_customdata_columns=[], color='time_since_last_vis', range_color=[0, 3], symbol='group', size='time_since_last_vis', size_max=15,
                               symbol_map={}, monkey_info_for_ff_in_past_or_future=None):

    ff_df = ff_df.sort_values(by='group', ascending=False).copy()
    current_traj_df = current_traj_df.sort_values(by='whether_stopped').copy()
    current_more_traj_df = current_more_traj_df.sort_values(
        by='whether_stopped').copy()
    # Note: the order of 'subgroup' and 'ff_number' matter here

    if size == 'time_since_last_vis':
        size = np.maximum(8, 18-ff_df['time_since_last_vis'].values*5)
    if size is None:
        size = np.ones(ff_df.shape[0]) * size_max
    fig, customdata_columns = plotly_for_ff_polar.plot_fireflies(None, ff_df, columns_for_annotation=columns_for_annotation,
                                                                 additional_customdata_columns=additional_customdata_columns, size=size, size_max=size_max,
                                                                 color=color, range_color=range_color, symbol=symbol, symbol_map=symbol_map)

    if show_more_ff is False:
        fig.update_traces(visible='legendonly', selector={
                          "marker_symbol": "diamond-open-dot"})

    fig = update_polar_background(fig)

    if add_trajectory:
        # fig = plot_trajectory_data(fig, current_traj_df, color_discrete_sequence=['red', 'blue'], additional_update_kwargs={'marker': {'size': 7}})
        fig = plotly_for_trajectory_polar.plot_trajectory_data(fig, current_more_traj_df, color_discrete_sequence=[
                                                               'black', 'gold'], additional_update_kwargs={'marker': {'size': 4}})

    if monkey_info_for_ff_in_past_or_future is not None:
        fig = plotly_for_trajectory_polar.plot_monkey_info_for_ff_in_past_or_future(
            fig, monkey_info_for_ff_in_past_or_future)

    if add_colorbar:
        fig = update_colorbar(fig)
    else:
        fig.update_coloraxes(showscale=False)

    if add_legend:
        fig = update_legend(fig)
    else:
        fig.update(layout_showlegend=False)

    if add_real_and_predicted_labels:
        fig = annotate_real_and_predicted_labels_in_numbers(
            fig, real_label, predicted_label)

    return fig, customdata_columns


def update_colorbar(fig):

    fig.update_coloraxes(colorbar_title="Time since last visible (s)",
                         colorbar_title_side='top',
                         colorbar_tickvals=[0, 1, 2, 3],
                         colorbar_ticktext=["0", "1", "2", "3"],
                         colorbar_ticks="outside",
                         colorbar_bgcolor="white",
                         colorbar_len=0.7,
                         colorbar_thickness=20,
                         colorbar_orientation='h',
                         colorbar_xanchor="right",
                         colorbar_x=1.2,
                         colorbar_yanchor="top",
                         colorbar_y=0)
    return fig


def update_legend(fig):
    fig.update_layout(legend=dict(
        yanchor="top",
        # y=0.9,
        y=1.12,
        xanchor="left",
        x=0.7,
        title=None,
        itemsizing='constant',
        font=dict(size=10)
    ))
    return fig


def update_polar_background(fig):
    fig.update_layout(height=600,
                      polar=dict(radialaxis=dict(tickangle=0,
                                                 range=[0, 400]),

                                 angularaxis=dict(
                          thetaunit="degrees",
                          dtick=45,
                          rotation=90,
                          direction="counterclockwise",
                          tickmode="array",
                          tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                          ticktext=[label + chr(176) for label in ["0", "45", "90", "135", "180", "-135", "-90", "-45"]],)
                      ))
    fig.update_polars(radialaxis_dtick=100,
                      radialaxis_tickfont_size=10,
                      angularaxis_tickfont_size=10,)
    return fig


def annotate_real_and_predicted_labels_in_numbers(fig, real_label, predicted_label):
    if real_label == predicted_label:
        predicted_label_color = 'green'
    else:
        predicted_label_color = 'purple'
    # annotate the real and predicted labels
    fig.add_annotation(x=0.1, y=1.08, text='Real: '+str(real_label+2),
                       showarrow=False, font=dict(size=12, color='green'))
    fig.add_annotation(x=0.1, y=1.05, text='Predicted: '+str(predicted_label+2),
                       showarrow=False, font=dict(size=12, color=predicted_label_color))
    return fig


def prepare_to_make_one_plotly_polar_plot(i, polar_plots_kwargs, point_index_array, GUAT_joined_ff_info, all_traj_feature_names, more_ff_df=None, trajectory_features=['monkey_distance', 'monkey_angle_to_origin'],
                                          add_monkey_info_for_ff_in_past_or_future=True):

    current_polar_plot_kargs = plot_decision_making.get_current_polar_plot_kargs(
        i, max_time_since_last_vis=3, show_reward_boundary=False, ff_colormap='Greens', null_arcs_bundle=None, **polar_plots_kwargs)

    real_label = current_polar_plot_kargs['real_label']
    predicted_label = current_polar_plot_kargs['predicted_label']
    current_traj_points = current_polar_plot_kargs['current_traj_points']
    current_stops = current_polar_plot_kargs['current_stops']
    current_more_traj_points = current_polar_plot_kargs['current_more_traj_points']
    current_more_traj_stops = current_polar_plot_kargs['current_more_traj_stops']

    # main_ff_df = make_main_ff_df(ff_input, current_more_ff_inputs, num_old_ff_per_row, num_ff_per_row)
    all_ff_dict = plotly_for_ff_polar.make_all_ff_dict_from_GUAT_joined_ff_df_and_more_ff_df(
        i, GUAT_joined_ff_info, more_ff_df, point_index_array)
    current_traj_df, current_more_traj_df = plotly_for_trajectory_polar.prepare_trajectory_data_for_plotting(
        current_traj_points, current_stops, all_traj_feature_names, current_more_traj_points=current_more_traj_points, current_more_traj_stops=current_more_traj_stops, trajectory_features=trajectory_features)
    present_ff_df = all_ff_dict['combined_ff_df'][all_ff_dict['combined_ff_df']
                                                  ['time_label'] == 'Present'].copy()

    current_plotly_polar_plot_kargs = {'ff_df': present_ff_df,
                                       'current_traj_df': current_traj_df,
                                       'current_more_traj_df': current_more_traj_df,
                                       'real_label': real_label,
                                       'predicted_label': predicted_label}

    if add_monkey_info_for_ff_in_past_or_future:
        current_plotly_polar_plot_kargs['monkey_info_for_ff_in_past_or_future'] = all_ff_dict['monkey_info_for_ff_in_past_or_future'].copy(
        )
    return current_plotly_polar_plot_kargs, all_ff_dict
