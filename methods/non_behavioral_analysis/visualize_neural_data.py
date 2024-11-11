import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
import scipy.interpolate as interpolate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





def plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=[1]):
    spike_subset = spike_df[(spike_df['time'] >= duration[0]) & (spike_df['time'] <= duration[1])]
    spike_time = spike_subset.time - duration[0]
    
    ax.scatter(spike_time, spike_subset.cluster, s=15)
    for x in x_values_for_vline:
        ax.axvline(x=x, color='r', linestyle='--')  
    ax.set_xlim([0, duration[1]-duration[0]])
    ax.set_yticks(unique_clusters)
    ax.set_yticklabels(unique_clusters)
    ax.set_title("Spikes")
    return ax



def make_individual_spike_plots(time_to_sample_from, spike_df, unique_clusters, interval_half_length=1, max_plots=2):
    random_sample = np.random.choice(time_to_sample_from, size=200, replace=False)
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length] 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=[interval_half_length])
        plt.show()
        plt.close(fig)
        
        if i == max_plots:
            break


def make_individual_spike_plot_from_target_cluster_VBLO(target_cluster_VBLO, spike_df, unique_clusters=1, starting_row=100, max_plots=2):
    subset = target_cluster_VBLO.iloc[starting_row:starting_row+max_plots]
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        duration = [row.last_visible_time - 1, row.caught_time + 1] 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=[row.last_visible_time-duration[0], row.caught_time-duration[0]])

        # annotate at row.last_visible_time-duration[0] as "last visible time"
        ax.annotate('last visible time', xy=(row.last_visible_time-duration[0], 0), xytext=(row.last_visible_time-duration[0]+0.1, 0.5),
                    )
        # annotate at row.caught_time-duration[0] as "caught time"
        ax.annotate('caught time', xy=(row.caught_time-duration[0], 0), xytext=(row.caught_time-duration[0]+0.1, 0.5),
                    )
        plt.show()
        plt.close(fig)
        
        if i == max_plots:
            break
    return 


def make_overlaid_spike_plot(time_to_sample_from, spike_df, unique_clusters, interval_half_length=1, max_rows_to_plot=2):
    random_sample = np.random.choice(time_to_sample_from, max_rows_to_plot)
    ax = None
    for i, time in enumerate(random_sample, start=1):
        duration = [time - interval_half_length, time + interval_half_length] 
        ax = _add_to_overlaid_spike_plot(ax, spike_df, duration, unique_clusters, x_values_for_vline=[interval_half_length])
        if i == max_rows_to_plot:
            break
    plt.show()
    



def _add_to_overlaid_spike_plot(ax, spike_df, duration, unique_clusters, x_values_for_vline=[]):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_spike_times(ax, spike_df, duration, unique_clusters, x_values_for_vline=x_values_for_vline)
    else:
        spike_subset = spike_df[(spike_df['time'] >= duration[0]) & (spike_df['time'] <= duration[1])]
        spike_time = spike_subset.time - duration[0]
        ax.scatter(spike_time, spike_subset.cluster, s=15) 
    return ax           





def plot_regression(final_behavioral_data, column, x_var, bins_to_plot=np.arange(1000), min_R_squared_to_plot=0.1):
    y_var = final_behavioral_data[column]

    # # drop rows where either x_var or y_var is nan, and print the number of dropped rows
    # n_rows = len(x_var)
    # dropped_rows = x_var[np.isnan(x_var) | np.isnan(y_var)]
    # x_var = x_var[~np.isnan(x_var) & ~np.isnan(y_var)]
    # y_var = y_var[~np.isnan(x_var) & ~np.isnan(y_var)]
    # print(f"Dropped {len(dropped_rows)} rows out of {n_rows} rows for {column} due to nan values.")

    if bins_to_plot is None:
        bins_to_plot = final_behavioral_data['bin'].values

    
    reg = LinearRegression().fit(x_var, y_var)
    R = np.corrcoef(y_var, reg.predict(x_var))[0, 1]
    R_squared = r2_score(y_var, reg.predict(x_var))
    pred = reg.predict(x_var)

    if R_squared < min_R_squared_to_plot:
        print(f"R: {round(R, 2)}, R^2: {round(R_squared, 3)} -- {column}")
        return

    # plot fit
    plt.figure()
    plt.title(column, fontsize=20)
    plt.scatter(range(len(bins_to_plot)), y_var[bins_to_plot], s=3)
    plt.plot(range(len(bins_to_plot)), pred[bins_to_plot], color='red', linewidth=0.03)
    plt.show()
    

    # plot pred against true
    plt.figure()
    plt.scatter(y_var[bins_to_plot], pred[bins_to_plot], s=5)
    plt.title(f"{column}, R: {round(R, 2)}, R^2: {round(R_squared, 3)}", fontsize=20)
    min_val = min(min(y_var[bins_to_plot]), min(pred[bins_to_plot]))
    max_val = max(max(y_var[bins_to_plot]), max(pred[bins_to_plot]))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    if column in ['gaze_monkey_view_x', 'gaze_monkey_view_y', 'gaze_world_x', 'gaze_world_y']:
        plt.xlim(-1000, 1000)
    plt.show()
    
