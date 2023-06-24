from multiff_analysis.functions.data_visualization import plot_behaviors
import os
import seaborn as sns
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot_behaviors_in_clusters(points_w_more_than_2_ff, chunk_numbers, monkey_information, ff_dataframe,
                                ff_life_sorted, ff_real_position_sorted, ff_caught_T_sorted, ff_flash_sorted):
    # for each chunk, finds indices of points where ddw > 0.15
    for chunk in chunk_numbers:
        chunk_df = points_w_more_than_2_ff[points_w_more_than_2_ff['chunk'] == chunk]
        duration_points = [chunk_df['point_index'].min(), chunk_df['point_index'].max()]
        duration = [monkey_information['monkey_t'][duration_points[0]], monkey_information['monkey_t'][duration_points[0]]+10]
        cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
        cum_ddw = np.array(monkey_information['monkey_ddw'].iloc[cum_indices])
        cum_abs_ddw = np.abs(cum_ddw)
        changing_dw_info = pd.DataFrame({'relative_point_index': np.where(cum_abs_ddw > 0.15)[0]})
        # find the first point of each sequence of consecutive points
        changing_dw_info['group'] = np.append(0, (np.diff(changing_dw_info['relative_point_index'])!=1).cumsum())
        changing_dw_info_short = changing_dw_info.groupby('group').min()
        changing_dw_info_short['relative_point_index'] = changing_dw_info_short['relative_point_index'].astype(int)
        changing_dw_info_short['point_index'] = cum_indices[changing_dw_info_short['relative_point_index']]
        for point_index in changing_dw_info_short['point_index']:
            duration = [monkey_information['monkey_t'][point_index]-2, monkey_information['monkey_t'][point_index]]
            ff_dataframe_sub = ff_dataframe[ff_dataframe['time'].between(duration[0], duration[1], inclusive='both')]
            # Make a polar plot from the monkey's perspective in the duration
            plot_behaviors.PlotPolar(duration,
                        monkey_information,
                        ff_dataframe_sub, 
                        ff_life_sorted,
                        ff_real_position_sorted,
                        ff_caught_T_sorted,
                        ff_flash_sorted,
                        rmax = 400,
                        currentTrial = None,
                        num_trials = None,
                        show_visible_ff = True,
                        show_visible_target = True,
                        show_ff_in_memory = True,
                        show_target_in_memory = True,
                        show_alive_ff = True
                            )


