# --- tidy imports (remove duplicate) ---
from data_wrangling import specific_utils, general_utils
from pattern_discovery import pattern_by_trials, cluster_analysis
from decision_making_analysis.event_detection import detect_rsw_and_rcap
from planning_analysis.show_planning import nxt_ff_utils

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Consider moving these to your script/entrypoint, not a module:
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# np.set_printoptions(suppress=True)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)


def make_pattern_frequencies(all_trial_patterns, ff_caught_T_new, monkey_information,
                             rsw_w_ff_frequency, one_stop_w_ff_frequency,
                             data_folder_name=None):
    # Ensure boolean/0-1 patterns behave as counts
    # (only coerces bool -> int; leaves numeric alone)
    atp_num = all_trial_patterns.copy()
    for c in atp_num.columns:
        if atp_num[c].dtype == bool:
            atp_num[c] = atp_num[c].astype(int)

    pattern_frequencies = (
        atp_num.sum(axis=0)
        .rename('frequency')
        .reset_index()
        .rename(columns={"index": 'item'})
    )
    n_trials = len(atp_num)
    pattern_frequencies['denom_count'] = n_trials - 1

    has = atp_num.columns
    capture_counted = max(len(ff_caught_T_new) - 1, 0)

    if "cluster_around_target" in has:
        pattern_frequencies.loc[
            pattern_frequencies['item'].isin(
                ["cluster_around_target", "disappear_latest"]),
            'denom_count'
        ] = n_trials
        pattern_frequencies.loc[
            pattern_frequencies['item'].isin(
                ["waste_cluster_around_target", "use_cluster"]),
            'denom_count'
        ] = atp_num["cluster_around_target"].sum()

    if "three_in_a_row" in has:
        pattern_frequencies.loc[pattern_frequencies['item']
                                == "three_in_a_row", 'denom_count'] = n_trials - 2
    if "four_in_a_row" in has:
        pattern_frequencies.loc[pattern_frequencies['item']
                                == "four_in_a_row", 'denom_count'] = n_trials - 3
    if "ignore_sudden_flash" in has and "sudden_flash" in has:
        pattern_frequencies.loc[
            pattern_frequencies['item'] == "ignore_sudden_flash", 'denom_count'
        ] = atp_num["sudden_flash"].sum()

    try_a_few = int(atp_num["retry_capture"].sum()
                    ) if "retry_capture" in has else 0
    n_total_rsw_block = rsw_w_ff_frequency + one_stop_w_ff_frequency + try_a_few

    pattern_frequencies.loc[
        pattern_frequencies['item'] == "retry_switch", 'frequency'
    ] = rsw_w_ff_frequency
    pattern_frequencies.loc[
        pattern_frequencies['item'].isin(
            ["retry_switch", "retry_capture"]),
        'denom_count'
    ] = n_total_rsw_block

    # assign group = 1 for the patterns above
    pattern_frequencies['group'] = 1

    # get more patterns
    rows = []

    rows.extend([
        {'item': "rsw_over_rcap", 'frequency': rsw_w_ff_frequency,
            'denom_count': try_a_few, 'group': 2},
        {'item': "rsw_over_both", 'frequency': rsw_w_ff_frequency,
            'denom_count': rsw_w_ff_frequency + try_a_few, 'group': 2},
        {'item': "rcap_over_both", 'frequency': try_a_few,
            'denom_count': rsw_w_ff_frequency + try_a_few, 'group': 2},
    ])

    retry = rsw_w_ff_frequency + try_a_few
    first_miss = retry + one_stop_w_ff_frequency
    capture_not_in_try_a_few = capture_counted - try_a_few
    # in terms of attempts as denominator, capture_counted include try_a_few
    first_attempt = capture_counted + rsw_w_ff_frequency + one_stop_w_ff_frequency

    # Choice after miss
    rows.extend([
        {'item': "retry_over_miss", 'frequency': retry,
            'denom_count': first_miss, 'group': 2},
        {'item': "no_retry_over_miss", 'frequency': (
            first_miss - retry), 'denom_count': first_miss, 'group': 2},
        {'item': "retry_fail_over_miss", 'frequency': rsw_w_ff_frequency,
            'denom_count': first_miss, 'group': 2},
        {'item': "retry_capture_over_miss", 'frequency': try_a_few,
            'denom_count': first_miss, 'group': 2},
    ])

    # try using first_attempt instead of first_miss as denom_count
    rows.extend([
        {'item': "retry_over_attempt", 'frequency': retry,
            'denom_count': first_attempt, 'group': 2},
        {'item': "no_retry_over_attempt", 'frequency': (
            first_miss - retry), 'denom_count': first_attempt, 'group': 2},
        {'item': "retry_fail_over_attempt", 'frequency': rsw_w_ff_frequency,
            'denom_count': first_attempt, 'group': 2},
        {'item': "retry_capture_over_attempt", 'frequency': try_a_few,
            'denom_count': first_attempt, 'group': 2},
    ])

    # Firefly capture rate (per s)
    total_duration = float(ff_caught_T_new[-1] - ff_caught_T_new[0])
    if total_duration > 0:
        rows.append({
            'item': "ff_capture_rate",
            'frequency': capture_counted,
            'denom_count': total_duration,
            'group': 2
        })

    # get miss_over_attempt and capture_over_attempt
    rows.extend([
        {'item': "miss_over_attempt", 'frequency': first_miss,
            'denom_count': first_attempt, 'group': 2},
        {'item': "first_shot_capture_over_attempt", 'frequency': capture_not_in_try_a_few,
            'denom_count': first_attempt, 'group': 2},
        {'item': "eventual_capture_over_attempt", 'frequency': capture_counted,
            'denom_count': first_attempt, 'group': 2},
        {'item': "eventual_miss_over_attempt", 'frequency': rsw_w_ff_frequency + one_stop_w_ff_frequency,
            'denom_count': first_attempt, 'group': 2},
    ])

    # Stop success rate (guard columns)
    if {"whether_new_distinct_stop", "time"}.issubset(monkey_information.columns):
        mask_new_stop = monkey_information["whether_new_distinct_stop"].eq(
            True)
        monkey_sub = monkey_information.loc[mask_new_stop].copy()
        if len(ff_caught_T_new) >= 2:
            t0, t1 = ff_caught_T_new[0], ff_caught_T_new[-1]
            monkey_sub = monkey_sub[monkey_sub["time"].between(t0, t1)]
        total_number_of_stops = len(monkey_sub)
        if total_number_of_stops > 0:
            rows.append({
                'item': "stop_success_rate",
                'frequency': capture_counted,
                'denom_count': total_number_of_stops,
                'group': 2
            })

    if "two_in_a_row" in has and "cluster_around_target" in has:
        twos = atp_num["two_in_a_row"].sum()
        clusters = atp_num["cluster_around_target"].sum()
        if clusters > 0:
            rows.append({
                'item': "two_in_a_row_over_cluster",
                'frequency': twos,
                'denom_count': clusters,
                'group': 2
            })

    if rows:
        pattern_frequencies = pd.concat(
            [pattern_frequencies, pd.DataFrame(rows)], ignore_index=True)

    denom = pattern_frequencies['denom_count'].replace(0, np.nan).astype(float)
    pattern_frequencies['rate'] = (
        pattern_frequencies['frequency'].astype(float) / denom).fillna(0.0)

    item_to_label = {
        'two_in_a_row': 'Two in a row',
        'visible_before_last_one': 'Visible before last capture',
        'disappear_latest': 'Target disappears latest',
        'use_cluster': 'Use cluster near target',
        'waste_cluster_around_target': 'Waste cluster around target',
        'ignore_sudden_flash': 'Ignore sudden flash',
        'sudden_flash': 'Sudden flash',
        'retry_switch': 'Retry switch',
        'retry_capture': 'Retry capture',
        'ff_capture_rate': 'Firefly capture rate (per s)',
        'stop_success_rate': 'Stop success rate',
        'three_in_a_row': 'Three in a row',
        'four_in_a_row': 'Four in a row',
        'one_in_a_row': 'One in a row',
        'multiple_in_a_row': 'Multiple in a row',
        'multiple_in_a_row_all': 'Multiple in a row incl. 1st',
        'cluster_around_target': 'Cluster exists around target',
        'rsw_over_rcap': 'rsw over rcap',
        'rsw_over_both': 'rsw over both',
        'rcap_over_both': 'rcap over both',
        'retry_fail_over_miss': 'rsw over all miss',
        'retry_capture_over_miss': 'rcap over all miss',
        'retry_over_miss': 'rsw+rcap over all miss',
        'no_retry_over_miss': 'No retry over all miss',
        'retry_fail_over_attempt': 'rsw over all attempt',
        'retry_capture_over_attempt': 'rcap over all attempt',
        'retry_over_attempt': 'rsw+rcap over all attempt',
        'no_retry_over_attempt': 'No retry over all attempt',
        'miss_over_attempt': 'Miss over attempt',
        'first_shot_capture_over_attempt': 'First shot capture over attempt',
        'eventual_capture_over_attempt': 'Eventual capture over attempt',
        'two_in_a_row_over_cluster': 'Two in a row over cluster'
    }
    pattern_frequencies['label'] = pattern_frequencies['item'].map(
        item_to_label).fillna("Missing")

    # Only compute Percentage for proportions (Group 1 or any Rate in [0,1])
    pattern_frequencies['percentage'] = np.where(
        (pattern_frequencies['rate'] >= 0.0) & (
            pattern_frequencies['rate'] <= 1.0),
        pattern_frequencies['rate'] * 100.0,
        np.nan
    )

    if data_folder_name:
        general_utils.save_df_to_csv(
            pattern_frequencies, 'pattern_frequencies', data_folder_name)

    return pattern_frequencies


def get_num_stops_array(monkey_information, array_of_trials):
    monkey_information = monkey_information[monkey_information['whether_new_distinct_stop'] == True].copy(
    )
    monkey_sub = monkey_information[['trial', 'point_index']].groupby(
        'trial').count().reset_index().rename(columns={'point_index': 'num_stops'})
    monkey_sub = monkey_sub.merge(pd.DataFrame(
        {'trial': array_of_trials}), on='trial', how='right')
    num_stops_array = monkey_sub['num_stops'].fillna(0).values.astype(int)
    return num_stops_array


def _calculate_trial_durations(ff_caught_T_new):
    """
    Calculate the durations of trials based on the sorted capture times.

    Parameters:
    ff_caught_T_new (list or array-like): Sorted capture times.

    Returns:
    tuple: A tuple containing the trial array and the duration array.
    """
    caught_ff_num = len(ff_caught_T_new)
    trial_array = list(range(caught_ff_num))

    # Calculate the differences between consecutive capture times
    t_array = (ff_caught_T_new[1:caught_ff_num] -
               ff_caught_T_new[:caught_ff_num-1]).tolist()

    # Add the first capture time to the beginning of the duration array
    t_array.insert(0, ff_caught_T_new[0])

    return trial_array, t_array


def _calculate_hitting_arena_edge(monkey_information, ff_caught_T_new):
    # Group by 'trial' and get the maximum 'crossing_boundary' value for each trial
    cb_df = monkey_information[['trial', 'crossing_boundary']].groupby(
        'trial').max().reset_index()
    # Create a DataFrame with 'trial' values ranging from 0 to the length of 'ff_caught_T_new'
    trial_df = pd.DataFrame({'trial': np.arange(len(ff_caught_T_new))})
    # Merge the DataFrames and fill missing values with 0
    merged_df = cb_df.merge(trial_df, on='trial', how='right').fillna(0)
    hitting_arena_edge = merged_df['crossing_boundary'].values.astype(int)
    return hitting_arena_edge


def _calculate_num_stops_since_last_vis(monkey_information, caught_ff_num, t_last_vis):
    # note that t_last_vis starts with trial = 1
    t_last_vis = np.array(t_last_vis)
    monkey_sub = monkey_information[monkey_information['whether_new_distinct_stop'] == True]
    monkey_sub = monkey_sub[monkey_sub['trial'] >= 1].copy()
    monkey_sub['t_last_vis'] = t_last_vis[monkey_sub['trial'].values - 1]
    monkey_sub = monkey_sub[monkey_sub['time'] >= monkey_sub['t_last_vis']]
    num_stops_since_last_vis = _use_merge_to_get_num_stops_for_each_trial(
        monkey_sub, caught_ff_num)

    return num_stops_since_last_vis


def _use_merge_to_get_num_stops_for_each_trial(monkey_sub, caught_ff_num):
    monkey_sub = monkey_sub[['trial', 'point_index']].groupby('trial').count()
    monkey_sub = monkey_sub.merge(pd.DataFrame(
        {'trial': np.arange(caught_ff_num)}), how='right', on='trial')
    monkey_sub = monkey_sub.fillna(0)
    num_stops_near_target = monkey_sub['point_index'].values
    return num_stops_near_target


def make_all_trial_features(ff_dataframe, monkey_information, ff_caught_T_new, cluster_around_target_indices, ff_real_position_sorted, ff_believed_position_sorted, max_cluster_distance=50, data_folder_name=None):
    # Note that we start from trial = 1
    caught_ff_num = len(ff_caught_T_new)
    visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]
    trial_array, t_array = _calculate_trial_durations(ff_caught_T_new)
    target_clust_last_vis_df = cluster_analysis._get_target_clust_last_vis_df(
        ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted)

    hitting_arena_edge = _calculate_hitting_arena_edge(
        monkey_information, ff_caught_T_new)
    num_stops_array = get_num_stops_array(
        monkey_information, np.arange(len(ff_caught_T_new)))
    num_stops_since_last_vis = _calculate_num_stops_since_last_vis(
        monkey_information, caught_ff_num, target_clust_last_vis_df['time_since_last_vis'].values)
    n_ff_in_a_row = pattern_by_trials.n_ff_in_a_row_func(
        ff_believed_position_sorted, distance_between_ff=50)
    num_ff_around_target = np.array(
        [len(unit) for unit in cluster_around_target_indices])

    trials_dict = {'trial': trial_array,
                   't': t_array,
                   't_last_vis': target_clust_last_vis_df['time_since_last_vis'].values,
                   'd_last_vis': target_clust_last_vis_df['last_vis_dist'].values,
                   'abs_angle_last_vis': np.abs(target_clust_last_vis_df['last_vis_ang'].values),
                   'hitting_arena_edge': hitting_arena_edge,
                   'num_stops': num_stops_array,
                   'num_stops_since_last_vis': num_stops_since_last_vis,
                   'num_ff_around_target': num_ff_around_target.tolist(),
                   'n_ff_in_a_row': n_ff_in_a_row[:len(trial_array)].tolist()
                   }
    all_trial_features = pd.DataFrame(trials_dict)

    # drop trial=0
    all_trial_features = all_trial_features[all_trial_features['trial'] > 0].reset_index(
        drop=True)

    if data_folder_name:
        general_utils.save_df_to_csv(
            all_trial_features, 'all_trial_features', data_folder_name)
    return all_trial_features


def make_feature_statistics(all_trial_features, data_folder_name=None):

    all_trial_features_valid = all_trial_features[(all_trial_features['t_last_vis'] < 50) & (
        all_trial_features['hitting_arena_edge'] == False)].reset_index()
    all_trial_features_valid = all_trial_features_valid.drop(
        columns={'index', 'trial'})
    median_values = all_trial_features_valid.median(axis=0)
    mean_values = all_trial_features_valid.mean(axis=0)
    n_trial = len(all_trial_features_valid)
    for i, item in enumerate(median_values.index):
        median = median_values[item]
        mean = mean_values[item]
        new_row = pd.DataFrame(
            {'item': item, 'median': median, 'mean': mean, 'n_trial': n_trial}, index=[0])
        if i == 0:
            feature_statistics = new_row
        else:
            feature_statistics = pd.concat([feature_statistics, new_row])

    feature_statistics = feature_statistics.reset_index(drop=True)

    feature_statistics['label'] = 'Missing'
    feature_statistics.loc[feature_statistics['item'] == 't', 'label'] = 'time'
    feature_statistics.loc[feature_statistics['item'] ==
                           't_last_vis', 'label'] = 'time target last seen'
    feature_statistics.loc[feature_statistics['item'] ==
                           'd_last_vis', 'label'] = 'distance target last seen'
    feature_statistics.loc[feature_statistics['item'] ==
                           'abs_angle_last_vis', 'label'] = 'abs angle target last seen'
    feature_statistics.loc[feature_statistics['item']
                           == 'num_stops', 'label'] = 'num stops'

    feature_statistics.loc[feature_statistics['item'] ==
                           'hitting_arena_edge', 'label'] = 'hitting arena edge'
    feature_statistics.loc[feature_statistics['item'] ==
                           'num_stops_since_last_vis', 'label'] = 'num stops since target last seen'
    feature_statistics.loc[feature_statistics['item'] ==
                           'n_ff_in_a_row', 'label'] = 'num ff caught in a row'
    feature_statistics.loc[feature_statistics['item'] ==
                           'num_ff_around_target', 'label'] = 'num ff around target'

    feature_statistics['label for median'] = 'Median ' + \
        feature_statistics['label']
    feature_statistics['label for mean'] = 'Mean ' + \
        feature_statistics['label']

    if data_folder_name:
        general_utils.save_df_to_csv(
            feature_statistics, 'feature_statistics', data_folder_name)

    return feature_statistics


def combine_df_of_agent_and_monkey(df_m, df_a, agent_names=["Agent", "Agent2", "Agent3"], df_a2=None, df_a3=None):
    """
    Make a dataframe that combines df such as df from the monkey and the agent(s);
    This function can include df from up to three agents. 

    Parameters
    ---------- 
    df_a: dict
    containing a df derived from the agent data
    df_m: dict
    containing a df derived from themonkey data
    agent_names: list, optional
    names of the agents used to identify the agents, if more than one agent is used
    df_a: dict, optional
    containing a df derived from the 2nd agent's data
    df_a: dict, optional
    containing a df derived from the 3rd agent's data

    Returns
    -------
    merged_df: dataframe that combines df from the monkey and the agent(s)

    """

    df_a['Player'] = agent_names[0]
    df_m['Player'] = 'monkey'

    if df_a3:
        # Then a 2nd agent and a 3rd agent are used
        df_a2['Player'] = agent_names[1]
        df_a3['Player'] = agent_names[2]
        merged_df = pd.concat([df_a, df_m, df_a2, df_a3], axis=0)
    elif df_a2:
        # Then a 2nd agent is used
        df_a2['Player'] = agent_names[1]
        merged_df = pd.concat([df_a, df_m, df_a2], axis=0)
    else:
        merged_df = pd.concat([df_a, df_m], axis=0)

    merged_df = merged_df.reset_index()

    return merged_df


def _add_dates_based_on_data_names(df):
    df['data'] = df['data_name'].apply(lambda x: x.split('_')[1])
    all_dates = [int(date[-4:]) for date in df['data'].tolist()]
    all_dates = [datetime.strptime(
        str(date/100), '%m.%d').date() for date in all_dates]
    df['date'] = all_dates
    df.sort_values(by='date', inplace=True)


def add_dates_and_sessions(df):
    # organize_patterns_and_features._add_dates_based_on_data_names(combd_feature_statistics)
    _add_dates_based_on_data_names(df)

    # Create a mapping of unique data_name to unique sessions
    unique_sessions = {name: i for i, name in enumerate(df['date'].unique())}

    # Map the unique sessions to the Data column
    df['session'] = df['date'].map(unique_sessions)

    # Sort the DataFrame by data_name
    df.sort_values(by='session', inplace=True)


def make_distance_df(ff_caught_T_new, monkey_information, ff_believed_position_sorted):
    cum_distance_array = []
    distance_array = []
    for i in range(len(ff_caught_T_new)-1):
        cum_distance_array.append(specific_utils.get_cum_distance_traveled(
            i, ff_caught_T_new, monkey_information))
        distance_array.append(specific_utils.get_distance_between_two_points(
            i, ff_caught_T_new, monkey_information, ff_believed_position_sorted))
    cum_distance_array = np.array(cum_distance_array)
    distance_array = np.array(distance_array)
    distance_df = pd.DataFrame(
        {'cum_distance': cum_distance_array, 'distance': distance_array})
    distance_df['trial'] = np.arange(len(distance_df))
    return distance_df


def make_num_stops_df(distance_df, closest_stop_to_capture_df, ff_caught_T_new, monkey_information):
    num_stops_df = nxt_ff_utils.drop_rows_where_stop_is_not_inside_reward_boundary(
        closest_stop_to_capture_df)
    num_stops_df = num_stops_df.rename(columns={'time': 'stop_time',
                                                'cur_ff_index': 'trial'})
    num_stops_df['num_stops'] = get_num_stops_array(monkey_information,
                                                    num_stops_df['trial'].values)

    num_stops_df['current_capture_time'] = ff_caught_T_new[num_stops_df['trial']]
    num_stops_df['prev_capture_time'] = ff_caught_T_new[num_stops_df['trial'] - 1]

    # Add distance information
    num_stops_df = num_stops_df.merge(
        distance_df, on='trial', how='left').dropna()

    # # Filter out the outliers
    # original_length = len(num_stops_df)
    # num_stops_df = num_stops_df[(num_stops_df['distance'] < 2000) & (num_stops_df['cum_distance'] < 2000)]
    # print(f'Filtered out {original_length - len(num_stops_df)} outliers out of {original_length} trials, since they have distance ' +
    #       f'or displacement greater than 2000, which is {round(100*(original_length - len(num_stops_df))/original_length, 2)}% of the data')
    return num_stops_df
