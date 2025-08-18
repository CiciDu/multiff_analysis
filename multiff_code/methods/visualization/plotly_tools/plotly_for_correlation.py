
from null_behaviors import curv_of_traj_utils
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import html, dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import matplotlib
from scipy import stats

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_scatter_plot_in_plotly(heading_info_df=None,
                                x_var_column='cum_distance_between_two_stops',
                                y_var_column='diff_in_abs_angle_to_nxt_ff',
                                current_stop_point_index_to_mark=None,
                                title=None,
                                equal_aspect=False,
                                margin_factor=0.1,
                                **kwargs):

    # Validate and extract data
    x_var, y_var = _validate_dataframe(
        heading_info_df, x_var_column, y_var_column)

    # Find index to mark on plot
    current_position_index_to_mark = _find_current_position_index(
        heading_info_df, current_stop_point_index_to_mark)

    # Prepare hover data
    customdata, hovertemplate = _prepare_hover_data(
        heading_info_df, x_var_column, y_var_column)

    # Get axis titles
    xaxis_title, yaxis_title = _get_axis_titles(x_var_column, y_var_column)

    # Create title
    title = _create_title(x_var_column, y_var_column, title)

    # Calculate axis limits
    if equal_aspect:
        x_lim, y_lim = _calculate_equal_aspect_limits(
            x_var, y_var, margin_factor)
    else:
        x_lim, y_lim = _calculate_axis_limits(x_var, y_var, margin_factor)

    # Create the plot
    fig_scatter = _plot_scatter_plot_in_plotly(
        x_var, y_var, customdata, hovertemplate, current_position_index_to_mark,
        title, xaxis_title, yaxis_title)

    # Apply improved axis limits
    _apply_axis_limits(fig_scatter, x_lim, y_lim, equal_aspect)

    fig_scatter.update_layout(title_font_size=13, showlegend=False)
    return fig_scatter


def make_regression_plot_in_plotly(heading_info_df=None,
                                   x_var_column='angle_from_stop_to_nxt_ff',
                                   y_var_column='angle_opt_cur_end_to_nxt_ff',
                                   current_stop_point_index_to_mark=None,
                                   title=None,
                                   equal_aspect=False,
                                   margin_factor=0.1,
                                   **kwargs):

    # Remove NaN values and track how many were removed
    original_length = len(heading_info_df)
    heading_info_df_clean = heading_info_df.dropna(
        subset=[x_var_column, y_var_column])
    new_length = len(heading_info_df_clean)

    # Create additional title information about removed data
    if original_length != new_length:
        add_to_title = f'# nan removed: {original_length-new_length} out of {original_length}'
    else:
        add_to_title = ''

    # Validate and extract data
    x_var, y_var = _validate_dataframe(
        heading_info_df_clean, x_var_column, y_var_column)

    # Find the index of the current stop point to mark if provided
    current_position_index_to_mark = _find_current_position_index(
        heading_info_df_clean, current_stop_point_index_to_mark)

    # Prepare hover data
    customdata, hovertemplate = _prepare_hover_data(
        heading_info_df_clean, x_var_column, y_var_column)

    # Get axis titles
    xaxis_title, yaxis_title = _get_axis_titles(x_var_column, y_var_column)

    # Create title
    title = _create_title(x_var_column, y_var_column, title, add_to_title)

    # Generate the regression plot first to get slope and intercept
    fig_angle = _plot_relationship_in_plotly(
        x_var, y_var, show_plot=False, title=title,
        current_position_index_to_mark=current_position_index_to_mark,
        hovertemplate=hovertemplate, customdata=customdata,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        add_to_title=add_to_title)

    # Calculate regression statistics for better limits
    slope, intercept, _, _, _ = stats.linregress(x_var, y_var)

    # Calculate axis limits considering regression line
    if equal_aspect:
        x_lim, y_lim = _calculate_equal_aspect_limits(
            x_var, y_var, margin_factor)
    else:
        x_lim, y_lim = _calculate_regression_line_limits(
            x_var, y_var, slope, intercept, margin_factor)

    # Apply improved axis limits
    _apply_axis_limits(fig_angle, x_lim, y_lim, equal_aspect)

    # Update layout of the plot
    fig_angle.update_layout(title_font_size=13, showlegend=False)
    return fig_angle


def _calculate_axis_limits(x_var, y_var, margin_factor=0.1):
    """Calculate optimal axis limits with margins for better visualization."""
    x_min, x_max = np.min(x_var), np.max(x_var)
    y_min, y_max = np.min(y_var), np.max(y_var)

    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Add margins (10% by default)
    x_margin = x_range * margin_factor
    y_margin = y_range * margin_factor

    # Handle edge cases where range is zero
    if x_range == 0:
        x_margin = 0.1 if x_min == 0 else abs(x_min) * 0.1
    if y_range == 0:
        y_margin = 0.1 if y_min == 0 else abs(y_min) * 0.1

    # Calculate limits
    x_lim = [x_min - x_margin, x_max + x_margin]
    y_lim = [y_min - y_margin, y_max + y_margin]

    return x_lim, y_lim


def _calculate_equal_aspect_limits(x_var, y_var, margin_factor=0.1):
    """Calculate axis limits that maintain equal aspect ratio."""
    x_min, x_max = np.min(x_var), np.max(x_var)
    y_min, y_max = np.min(y_var), np.max(y_var)

    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Use the larger range to determine the scale
    max_range = max(x_range, y_range)

    # Calculate centers
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Add margins
    margin = max_range * margin_factor

    # Handle edge cases where range is zero
    if max_range == 0:
        margin = 0.1

    # Calculate limits maintaining equal aspect ratio
    half_range = max_range / 2 + margin

    x_lim = [x_center - half_range, x_center + half_range]
    y_lim = [y_center - half_range, y_center + half_range]

    return x_lim, y_lim


def _apply_axis_limits(fig, x_lim, y_lim, equal_aspect=True):
    """Apply calculated limits to the figure."""
    if equal_aspect:
        fig.update_layout(
            xaxis=dict(range=x_lim, scaleanchor="x", scaleratio=1),
            yaxis=dict(range=y_lim)
        )
    else:
        fig.update_layout(
            xaxis=dict(range=x_lim),
            yaxis=dict(range=y_lim)
        )


def _calculate_regression_line_limits(x_var, y_var, slope, intercept, margin_factor=0.1):
    """Calculate limits that ensure the regression line is visible."""
    # Calculate data limits
    x_min, x_max = np.min(x_var), np.max(x_var)
    y_min, y_max = np.min(y_var), np.max(y_var)

    # Calculate regression line limits
    reg_y_min = slope * x_min + intercept
    reg_y_max = slope * x_max + intercept

    # Find the overall limits including regression line
    overall_y_min = min(y_min, reg_y_min)
    overall_y_max = max(y_max, reg_y_max)

    # Calculate ranges
    x_range = x_max - x_min
    y_range = overall_y_max - overall_y_min

    # Add margins
    x_margin = x_range * margin_factor
    y_margin = y_range * margin_factor

    # Handle edge cases
    if x_range == 0:
        x_margin = 0.1 if x_min == 0 else abs(x_min) * 0.1
    if y_range == 0:
        y_margin = 0.1 if overall_y_min == 0 else abs(overall_y_min) * 0.1

    # Calculate limits
    x_lim = [x_min - x_margin, x_max + x_margin]
    y_lim = [overall_y_min - y_margin, overall_y_max + y_margin]

    return x_lim, y_lim


def _validate_dataframe(heading_info_df, x_var_column, y_var_column):
    """Validate dataframe and extract required columns."""
    if heading_info_df is None:
        raise ValueError("heading_info_df is None, cannot create plot.")

    # Extract x and y variables, raise error if columns missing
    try:
        x_var = heading_info_df[x_var_column].values
        y_var = heading_info_df[y_var_column].values
    except KeyError as e:
        raise ValueError(f"Missing column in dataframe: {e}")

    # Check for empty arrays
    if len(x_var) == 0 or len(y_var) == 0:
        raise ValueError(
            'Data arrays are empty, so correlation plot is not shown')

    return x_var, y_var


def _find_current_position_index(heading_info_df, current_stop_point_index_to_mark):
    """Find the index of the current stop point to mark on the plot."""
    if current_stop_point_index_to_mark is None:
        return None

    try:
        matches = heading_info_df.index[
            heading_info_df['stop_point_index'] == current_stop_point_index_to_mark
        ]
        return matches[0] if len(matches) > 0 else None
    except (KeyError, IndexError):
        raise ValueError(
            f'No match found for current_stop_point_index_to_mark: {current_stop_point_index_to_mark}')


def _prepare_hover_data(heading_info_df, x_var_column, y_var_column):
    """Prepare customdata and hovertemplate for hover information."""
    # Prepare customdata for hover info
    customdata = None
    if 'stop_point_index' in heading_info_df.columns:
        customdata = heading_info_df['stop_point_index'].values

    hovertemplate = (
        '<b>x: %{x:.2f} <br>y: %{y:.2f}</b><br><br>'
        'Stop point index:<br>%{customdata}'
        '<extra></extra>'
    )

    return customdata, hovertemplate


def _get_axis_titles(x_var_column, y_var_column):
    """Get formatted axis titles based on column names."""
    xaxis_title = x_var_column
    yaxis_title = y_var_column

    # Special formatting for specific columns
    if x_var_column == 'angle_from_stop_to_nxt_ff':
        xaxis_title = 'Angle from Stop to Nxt FF'
    if y_var_column == 'angle_from_stop_to_nxt_ff':
        yaxis_title = 'Angle from Stop to Nxt FF'

    return xaxis_title, yaxis_title


def _create_title(x_var_column, y_var_column, title=None, add_to_title=''):
    """Create plot title with optional additional information."""
    if title is None:
        # Split title into two lines after "vs"
        title = f'{x_var_column}<br>vs {y_var_column}'
    else:
        # If custom title is provided, try to split it at "vs" if present
        if ' vs ' in title:
            title = title.replace(' vs ', '<br>vs ')

    if add_to_title:
        title += f"<br><sup>{add_to_title}</sup>"

    return title


def put_down_correlation_plot(fig_scatter, id='correlation_plot', width='60%'):

    if id is None:
        id = 'correlation_plot'

    return html.Div([
        dcc.Graph(
                    id=id,
                    figure=fig_scatter,
                    # ['Original-Present', 0]}]}
                    hoverData={'points': [{'customdata': 0}]},
                    clickData={'points': [{'customdata': 0}]}
                    ),
        # 'display': 'inline-block',
    ], style={'width': width, 'padding': '0 0 0 0',
              })


def find_new_curv_of_traj_counted(point_index_for_curv_of_traj_df, monkey_information, ff_caught_T_new, curv_of_traj_mode, lower_end, upper_end, truncate_curv_of_traj_by_time_of_capture=False):
    if (lower_end is not None) & (upper_end is not None):
        if curv_of_traj_mode == 'time':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_time_window(
                point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        elif curv_of_traj_mode == 'distance':
            new_curv_of_traj_df = curv_of_traj_utils.find_curv_of_traj_df_based_on_distance_window(
                point_index_for_curv_of_traj_df, lower_end, upper_end, monkey_information, ff_caught_T_new, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            raise PreventUpdate(
                "No update was made because curv_of_traj_mode is not recognized.")
        new_curv_of_traj_counted = new_curv_of_traj_df['curv_of_traj'].values
        return new_curv_of_traj_counted
    else:
        raise PreventUpdate(
            "No update was made because lower_end or upper_end is None.")


def find_curv_of_traj_counted_from_curv_of_traj_df(curv_of_traj_df, point_index_for_curv_of_traj_df):
    curv_of_traj_df = curv_of_traj_df.set_index('point_index')
    curv_of_traj_counted = curv_of_traj_df.loc[point_index_for_curv_of_traj_df,
                                               'curv_of_traj'].values
    return curv_of_traj_counted


def _plot_relationship_in_plotly(x_array, y_array, slope=None, show_plot=True,
                                 title="Traj Curv: From Current Point to Right Before Stop <br> At -1 Sec",
                                 xaxis_title='Traj Curv - Curv to cur ff (cm)',
                                 yaxis_title='Curv to nxt ff - Curv to cur ff (cm)',
                                 customdata=None,
                                 hovertemplate=None,
                                 current_position_index_to_mark=None,
                                 add_to_title=''):

    # Calculate linear regression statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x_array, y_array)

    # Create the figure
    fig = go.Figure()

    # Add scatter plot of data points
    fig.add_trace(go.Scatter(
        x=x_array,
        y=y_array,
        mode='markers',
        showlegend=False,
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))

    # Add regression line
    x_min, x_max = min(x_array), max(x_array)
    fig.add_trace(go.Scatter(
        x=np.array([x_min, x_max]),
        y=np.array([x_min, x_max]) * slope + intercept,
        showlegend=False,
        mode='lines',
        line=dict(color='red')
    ))

    # Update layout with title and axis configuration
    regression_stats = f'r value = {round(r_value, 2)}, slope = {round(slope, 2)}'
    full_title = f"{title}<br><sup>{regression_stats}<br>{add_to_title}</sup>"

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=go.layout.Title(
            text=full_title,
            xref="paper",
            x=0
        ),
    )

    # Mark current position if specified
    if current_position_index_to_mark is not None:
        fig = _plot_current_position_in_scatter_plot(
            fig, x_array, y_array, current_position_index_to_mark)

    if show_plot:
        fig.show()
    return fig


def _plot_scatter_plot_in_plotly(x_var, y_var, customdata, hovertemplate, current_position_index_to_mark, title=None, xaxis_title=None, yaxis_title=None):
    """Create a scatter plot with optional current position marking."""
    # Create the figure
    fig_scatter = go.Figure()

    # Add main scatter plot
    fig_scatter.add_trace(
        go.Scatter(
            x=x_var,
            y=y_var,
            mode='markers',
            marker=dict(size=7, color='blue'),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False
        )
    )

    # Mark current position if specified and valid
    if current_position_index_to_mark is not None and 0 <= current_position_index_to_mark < len(x_var):
        fig_scatter = _plot_current_position_in_scatter_plot(
            fig_scatter, x_var, y_var, current_position_index_to_mark)

    # Update layout
    fig_scatter.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_font_size=13,
        showlegend=True,
    )
    return fig_scatter


def _plot_current_position_in_scatter_plot(fig_scatter, x_var, y_var, current_position_index_to_mark):
    """Add a special marker for the current position in a scatter plot."""
    fig_scatter.add_trace(
        go.Scatter(
            x=[x_var[current_position_index_to_mark]],
            y=[y_var[current_position_index_to_mark]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='Current Stop',
            showlegend=True,
            hoverinfo='skip'
        )
    )
    return fig_scatter
