from visualization.matplotlib_tools import plot_behaviors_utils
from data_wrangling import specific_utils

import os
import numpy as np
import pandas as pd
import math
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components


def make_point_vs_cluster(
    ff_dataframe,
    max_ff_distance_from_monkey=500,
    max_cluster_distance=50,
    max_time_past=1,
    print_progress=True,
    data_folder_name=None,
):

    # Pre-filter once
    ff_dataframe_subset = ff_dataframe[
        (ff_dataframe['time_since_last_vis'] <= max_time_past) &
        (ff_dataframe['ff_distance'] <= max_ff_distance_from_monkey)
    ][['point_index', 'ff_x', 'ff_y', 'ff_index', 'target_index']]

    point_vs_cluster = []

    grouped = ff_dataframe_subset.groupby('point_index')

    max_point_index = ff_dataframe_subset['point_index'].max()

    for point_index, selected_ff in grouped:

        n_ff = len(selected_ff)

        if n_ff < 2:
            continue

        ffxy_array = selected_ff[['ff_x', 'ff_y']].values
        ff_indices = selected_ff['ff_index'].values
        trial = selected_ff['target_index'].values[0]

        # Fast pairwise distances
        dists = squareform(pdist(ffxy_array))
        adjacency = dists <= max_cluster_distance

        n_components, labels = connected_components(adjacency)

        # Find clusters with >1 member
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_clusters = unique_labels[counts > 1]

        if len(valid_clusters) > 0:
            mask = np.isin(labels, valid_clusters)
            for idx in np.where(mask)[0]:
                point_vs_cluster.append(
                    [point_index, ff_indices[idx], labels[idx], trial]
                )

        if print_progress and point_index % 1000 == 0:
            print(f'Progress: {point_index} / {max_point_index}')

    point_vs_cluster = pd.DataFrame(
        point_vs_cluster,
        columns=['point_index', 'ff_index', 'cluster_label', 'target_index']
    )

    if data_folder_name is not None:
        os.makedirs(data_folder_name, exist_ok=True)
        filepath = os.path.join(data_folder_name, 'point_vs_cluster.csv')
        point_vs_cluster.to_csv(filepath, index=False)

    return point_vs_cluster



def clusters_of_ffs_func(point_vs_cluster, monkey_information, ff_caught_T_new):
    """
Find clusters of fireflies that appear during a trial based on point_vs_cluster

Parameters
----------
point_vs_cluster: array 
    contains indices of fireflies belonging to a cluster at each time point
    structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
monkey_information: df
containing the speed, angle, and location of the monkey at various points of time
ff_caught_T_new: np.array
containing the time when each captured firefly gets captured

Returns
-------
    cluster_exist_trials: array
            trial numbers of the trials where at least one cluster exists
    cluster_dataframe_point: dataframe
            information of the clusters for each time point that has at least one cluster
    cluster_dataframe_trial: dataframe
            information of the clusters for each trial that has at least one cluster
"""

  # Turn point_vs_cluster from np.array into a dataframe
    temp_dataframe1 = pd.DataFrame(point_vs_cluster, columns=[
                                   'point_index', 'ff_index', 'cluster_label'])
    # Find indices of unique points and their counts and make them into a dataframe as well
    unique_time_points, counts = np.unique(
        point_vs_cluster[:, 0], return_counts=True)
    temp_dataframe2 = pd.DataFrame(np.concatenate([unique_time_points.reshape(-1, 1), counts.reshape(-1, 1)], axis=1),
                                   columns=['point_index', 'num_ff_at_point'])
    # Combine the information of the above 2 dataframes
    temp_dataframe3 = temp_dataframe1.merge(
        temp_dataframe2, how="left", on="point_index")
    # Find the corresponding time to all the points
    corresponding_t = monkey_information['time'].values[np.array(
        temp_dataframe3['point_index'])]
    temp_dataframe3['time'] = corresponding_t
    # From the time of each point, find the target index that corresponds to that point
    temp_dataframe3['target_index'] = np.searchsorted(
        ff_caught_T_new, corresponding_t)
    # Only keep the part of the data up to the capture of the last firefly
    temp_dataframe3 = temp_dataframe3[temp_dataframe3['target_index'] < len(
        ff_caught_T_new)]
    # Thus we have the information of the clusters for each time point that has at least one cluster
    cluster_dataframe_point = temp_dataframe3
    # By grouping the information into trials, we can have the information of the clusters for each trial that has at least one cluster;
    # For each trial, we'll have the maximum number of fireflies in all clusters as well as the number of clusters
    cluster_dataframe_trial = cluster_dataframe_point[['target_index', 'num_ff_at_point']].groupby('target_index',
                                                                                                   as_index=True).agg({'num_ff_at_point': ['max', 'count']})
    cluster_dataframe_trial.columns = [
        "max_ff_in_cluster", "num_points_w_cluster"]
    # We can also take out the trials during which at least one cluster exists
    cluster_exist_trials = np.array(
        cluster_dataframe_point.target_index.unique())
    return cluster_exist_trials, cluster_dataframe_point, cluster_dataframe_trial


def find_alive_ff_clusters(ff_positions, ff_real_position_sorted, array_of_start_time_of_evaluation, array_of_end_time_of_evaluation, ff_life_sorted, max_distance=50,
                           empty_cluster_ok=False):
    ff_indices_of_each_cluster = []
    ff_num = len(array_of_start_time_of_evaluation)
    for i in range(ff_num):
        ff_of_interet_position = ff_positions[i]
        duration = [array_of_start_time_of_evaluation[i],
                    array_of_end_time_of_evaluation[i]]
        alive_ff_indices, alive_ff_position = plot_behaviors_utils.find_alive_ff(
            duration, ff_life_sorted, ff_real_position_sorted)
        alive_ff_indices_close_to_ff_of_interet = np.where(
            np.linalg.norm(alive_ff_position.T - ff_of_interet_position, axis=1) < max_distance)[0]
        ff_indices_close_to_ff_of_interet = alive_ff_indices[alive_ff_indices_close_to_ff_of_interet]
        if len(ff_indices_close_to_ff_of_interet) == 0:
            if not empty_cluster_ok:
                raise ValueError(
                    f'No firefly is found within the distance of {max_distance} cm from the firefly of interest at time {array_of_start_time_of_evaluation[i]} and {array_of_end_time_of_evaluation[i]}')
        ff_indices_of_each_cluster.append(ff_indices_close_to_ff_of_interet)
    return ff_indices_of_each_cluster


def turn_list_of_ff_clusters_info_into_dataframe(ff_clusters, all_point_index):
    ff_cluster_df = pd.DataFrame({'point_index': [], 'ff_index': []})
    for i in range(len(all_point_index)):
        point_index = all_point_index[i]
        for ff in ff_clusters[i]:
            ff_cluster_df = pd.concat([ff_cluster_df, pd.DataFrame(
                {'point_index': point_index, 'ff_index': ff}, index=[0])], ignore_index=True)
    ff_cluster_df['point_index'] = ff_cluster_df['point_index'].astype(int)
    ff_cluster_df['ff_index'] = ff_cluster_df['ff_index'].astype(int)
    return ff_cluster_df


def find_ff_cluster_last_vis_df(ff_indices_of_each_cluster, time_of_evaluation_for_each_cluster, ff_dataframe,
                                duration_of_evaluation=10,
                                cluster_identifiers=None):
    # ff_indices_of_each_cluster: a list; each item in ff_indices_of_each_cluster should contain at least one ff index

    print("Finding information of clusters ...")
    if cluster_identifiers is None:
        cluster_identifiers = list(
            range(len(time_of_evaluation_for_each_cluster)))
    list_of_cluster_identifier = []
    nearby_alive_ff_indices = []
    last_vis_point_index = []
    last_vis_ff_index = []
    time_since_last_vis = []
    last_vis_dist = []
    last_vis_ang = []
    last_vis_ang_to_bndry = []
    ff_dataframe = ff_dataframe[ff_dataframe['visible'] == 1].copy()
    for i in range(len(time_of_evaluation_for_each_cluster)):
        time_of_evaluation = time_of_evaluation_for_each_cluster[i]
        ff_indices_in_a_cluster = ff_indices_of_each_cluster[i]
        ff_dataframe_subset = ff_dataframe[(ff_dataframe['time'].between(
            time_of_evaluation-duration_of_evaluation, time_of_evaluation))].copy()
        ff_dataframe_subset = ff_dataframe_subset[ff_dataframe_subset['ff_index'].isin(
            ff_indices_in_a_cluster)]

        if len(ff_dataframe_subset) > 0:
            list_of_cluster_identifier.append(cluster_identifiers[i])
            nearby_alive_ff_indices.append(ff_indices_in_a_cluster)
            latest_visible_ff = ff_dataframe_subset.loc[ff_dataframe_subset['time'].idxmax(
            )]
            last_vis_ff_index.append(latest_visible_ff['ff_index'])
            last_vis_point_index.append(latest_visible_ff['point_index'])
            time_since_last_vis.append(
                time_of_evaluation - latest_visible_ff['time'])
            last_vis_dist.append(latest_visible_ff['ff_distance'])
            last_vis_ang.append(latest_visible_ff['ff_angle'])
            last_vis_ang_to_bndry.append(
                latest_visible_ff['ff_angle_boundary'])

        if i % 100 == 0:
            print(i, 'out of', len(time_of_evaluation_for_each_cluster))

    # make the lists above into a dataframe
    ff_cluster_df = pd.DataFrame({'cluster_identifier': list_of_cluster_identifier,
                                  'last_vis_ff_index': last_vis_ff_index,
                                  'nearby_alive_ff_indices': nearby_alive_ff_indices,
                                  'last_vis_point_index': last_vis_point_index,
                                  'time_since_last_vis': time_since_last_vis,
                                  'last_vis_dist': last_vis_dist,
                                  'last_vis_ang': last_vis_ang,
                                  'last_vis_ang_to_bndry': last_vis_ang_to_bndry})

    ff_cluster_df['abs_last_vis_ang'] = np.abs(ff_cluster_df['last_vis_ang'])
    ff_cluster_df['abs_last_vis_ang_to_bndry'] = np.abs(
        ff_cluster_df['last_vis_ang_to_bndry'])

    return ff_cluster_df


# The function below is similar as the function above
def find_last_vis_time_of_a_cluster_before_a_time(list_of_ff_clusters, list_of_time, ff_dataframe):
    # relevant_indices indicate
    list_of_last_vis_time = []
    for i in range(len(list_of_time)):
        indices_of_ff = list_of_ff_clusters[i]
        latest_time = list_of_time[i]
        # for each set of ff_near_stops
        ff_info = ff_dataframe.loc[ff_dataframe['time'] <= latest_time]
        ff_info = ff_info[ff_info['ff_index'].isin(indices_of_ff)]
        ff_info = ff_info[ff_info['visible'] == 1]
        last_vis_time = ff_info.time.max()
        list_of_last_vis_time.append(last_vis_time)
    list_of_last_vis_time = np.array(list_of_last_vis_time)
    return list_of_last_vis_time


def find_alive_target_clusters(ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, max_distance=50):
    ff_indices_of_each_cluster = find_alive_ff_clusters(
        ff_real_position_sorted, ff_real_position_sorted, ff_caught_T_new-10, ff_caught_T_new+10, ff_life_sorted, max_distance=max_distance)
    return ff_indices_of_each_cluster


def lookup_rows_by_time(
    df: pd.DataFrame,
    target_times,
    time_col: str = 'time',
    max_diff = 0.1,
) -> pd.DataFrame:
    """
    Look up rows in `df` whose `time_col` values are closest to `target_times`.

    Raises ValueError if the nearest time is farther than `max_diff`.
    """
    times = df[time_col].to_numpy()
    target_times = np.atleast_1d(target_times)

    # Find insertion indices
    indices = np.searchsorted(times, target_times, side='left')
    indices = np.clip(indices, 1, len(times)-1)

    # Compare left vs right neighbor
    left = indices - 1
    right = indices
    choose_left = np.abs(target_times - times[left]) < np.abs(target_times - times[right])
    nearest = np.where(choose_left, left, right)

    # Check max_diff condition
    if max_diff is not None:
        diffs = np.abs(target_times - times[nearest])
        if np.any(diffs > max_diff):
            raise ValueError(
                f'Time mismatch too large: max diff={diffs.max():.3f}, threshold={max_diff}'
            )

    return df.iloc[nearest].copy()



def get_target_last_vis_df(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, duration_of_evaluation=10):
    """
    Calculate metrics for the last visible target.
    """
    metrics = {
        'last_vis_point_index': [],
        'time_since_last_vis': [],
        'last_vis_dist': [], 'last_vis_cum_dist': [], 'last_vis_ang': [], 'last_vis_ang_to_bndry': []
    }

    ff_capture_rows = lookup_rows_by_time(monkey_information, ff_caught_T_new)
    visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]

    for i, caught_time in enumerate(ff_caught_T_new):
        ff_info_sub = visible_ff[visible_ff['time'].between(
            caught_time - duration_of_evaluation, caught_time)].copy()
        ff_info_sub['ff_distance_to_target'] = np.linalg.norm(
            ff_info_sub[['ff_x', 'ff_y']].values - ff_real_position_sorted[i], axis=1)
        relevant_df = ff_info_sub[ff_info_sub['ff_index'] == i].copy()

        if not relevant_df.empty:
            relevant_df.sort_values(by=['point_index', 'ff_distance_to_target'], ascending=[
                                    True, False], inplace=True)
            row = relevant_df.iloc[-1]
            metrics['last_vis_point_index'].append(row['point_index'])
            metrics['time_since_last_vis'].append(caught_time - row['time'])
            metrics['last_vis_dist'].append(row['ff_distance'])
            metrics['last_vis_cum_dist'].append(
                ff_capture_rows.iloc[i]['cum_distance'] - row['cum_distance'])
            metrics['last_vis_ang'].append(row['ff_angle'])
            metrics['last_vis_ang_to_bndry'].append(row['ff_angle_boundary'])
        else:
            for key in metrics:
                metrics[key].append(9999)

    target_last_vis_df = pd.DataFrame({
        **metrics,
        'target_index': np.arange(len(ff_caught_T_new)),
        'abs_last_vis_ang': np.abs(metrics['last_vis_ang']),
        'abs_last_vis_ang_to_bndry': np.abs(metrics['last_vis_ang_to_bndry'])
    })

    return target_last_vis_df


def get_target_clust_last_vis_df(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, ff_life_sorted, duration_of_evaluation=10,
                                 max_distance_to_target_in_cluster=50, keep_all_rows=False):
    ff_indices_of_each_cluster = find_alive_target_clusters(
        ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, max_distance=max_distance_to_target_in_cluster)
    target_clust_last_vis_df = _get_target_clust_last_vis_df(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted,
                                                             max_distance_to_target_in_cluster=max_distance_to_target_in_cluster, duration_of_evaluation=duration_of_evaluation)
    target_clust_last_vis_df['nearby_alive_ff_indices'] = ff_indices_of_each_cluster
    if not keep_all_rows:
        # drop the rows whose last_vis_dist is 9999
        target_clust_last_vis_df = target_clust_last_vis_df[target_clust_last_vis_df['last_vis_dist'] != 9999].copy(
        )
        # also drop target_index = 0
        target_clust_last_vis_df = target_clust_last_vis_df[target_clust_last_vis_df['target_index'] != 0].copy(
        )
    else:
        target_clust_last_vis_df = target_clust_last_vis_df
    return target_clust_last_vis_df


def _get_target_clust_last_vis_df(ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, duration_of_evaluation=10, max_distance_to_target_in_cluster=50):
    """
    Calculate metrics for the last visible target cluster.
    """

    metrics = {
        'last_vis_point_index': [], 'last_vis_time': [], 'last_vis_ff_index': [], 'nearby_vis_ff_indices': [],
        'time_since_last_vis': [], 'last_vis_dist': [], 'last_vis_cum_dist': [], 'last_vis_ang': [], 'last_vis_ang_to_bndry': [],
    }

    ff_capture_rows = lookup_rows_by_time(monkey_information, ff_caught_T_new)
    visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]

    for i, caught_time in enumerate(ff_caught_T_new):
        ff_info_sub = visible_ff[visible_ff['time'].between(
            caught_time - duration_of_evaluation, caught_time)].copy()
        ff_info_sub['ff_distance_to_target'] = np.linalg.norm(
            ff_info_sub[['ff_x', 'ff_y']].values - ff_real_position_sorted[i], axis=1)
        relevant_df = ff_info_sub[ff_info_sub['ff_distance_to_target']
                                  < max_distance_to_target_in_cluster].copy()

        if not relevant_df.empty:
            relevant_df.sort_values(by=['point_index', 'ff_distance_to_target'], ascending=[
                                    True, False], inplace=True)
            row = relevant_df.iloc[-1]
            metrics['last_vis_point_index'].append(row['point_index'])
            metrics['last_vis_time'].append(row['time'])
            metrics['last_vis_ff_index'].append(row['ff_index'])
            metrics['time_since_last_vis'].append(caught_time - row['time'])
            metrics['last_vis_dist'].append(row['ff_distance'])
            metrics['last_vis_cum_dist'].append(
                ff_capture_rows.iloc[i]['cum_distance'] - row['cum_distance'])
            metrics['last_vis_ang'].append(row['ff_angle'])
            metrics['last_vis_ang_to_bndry'].append(row['ff_angle_boundary'])

            metrics['nearby_vis_ff_indices'].append(
                relevant_df['ff_index'].unique().tolist())
        else:
            for key in metrics:
                metrics[key].append(
                    9999 if key != 'nearby_vis_ff_indices' else [])

    target_clust_last_vis_df = pd.DataFrame({
        'target_index': np.arange(len(ff_caught_T_new)),
        **metrics,
        'abs_last_vis_ang': np.abs(metrics['last_vis_ang']),
        'abs_last_vis_ang_to_bndry': np.abs(metrics['last_vis_ang_to_bndry']),
    })

    target_clust_last_vis_df['caught_time'] = ff_caught_T_new[target_clust_last_vis_df['target_index']]

    return target_clust_last_vis_df
