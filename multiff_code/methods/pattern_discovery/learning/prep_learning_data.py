
from scipy import stats
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_class, plot_variations_utils
from data_wrangling import specific_utils, process_monkey_information, base_processing_class, combine_info_utils, further_processing_class

from pattern_discovery.learning.proportion_trend import analyze_proportion_trend


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os


def get_key_learning_data(raw_data_dir_name='all_monkey_data/raw_monkey_data', monkey_name='monkey_Bruno'):

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, monkey_name)

    all_trial_durations_df = pd.DataFrame()
    all_stop_df = pd.DataFrame()
    all_VBLO_df = pd.DataFrame()

    for index, row in sessions_df_for_one_monkey.iterrows():
        data_name = row['data_name']
        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], data_name)
        print(raw_data_folder_path)
        data_item = further_processing_class.FurtherProcessing(
            raw_data_folder_path=raw_data_folder_path)

        # disable printing
        data_item.retrieve_or_make_monkey_data()
        data_item.make_or_retrieve_ff_dataframe()

        trial_durations = np.diff(data_item.ff_caught_T_new)
        trial_durations_df = pd.DataFrame(
            {'duration_sec': trial_durations, 'trial_index': np.arange(1, len(trial_durations) + 1)}) # trial_index starts from 1 since we don't calculate duration for the first trial
        trial_durations_df['data_name'] = data_name
        all_trial_durations_df = pd.concat(
            [all_trial_durations_df, trial_durations_df])

        num_stops = data_item.monkey_information.loc[data_item.monkey_information['whether_new_distinct_stop'] == True, [
            'time']].shape[0]
        num_captures = len(data_item.ff_caught_T_new)
        stop_df = pd.DataFrame(
            {
                'stops': [num_stops],
                'captures': [num_captures],
                'data_name': [data_name],
            }
        )
        all_stop_df = pd.concat([all_stop_df, stop_df])

        data_item.get_visible_before_last_one_trials_info()
        num_VBLO_trials = len(data_item.vblo_target_cluster_df)
        all_selected_base_trials = len(data_item.selected_base_trials)
        VBLO_df = pd.DataFrame(
            {
                'VBLO_trials': [num_VBLO_trials],
                'base_trials': [all_selected_base_trials],
                'data_name': [data_name],
            }
        )
        all_VBLO_df = pd.concat([all_VBLO_df, VBLO_df])

    all_trial_durations_df = make_variations_utils.assign_session_id(
        all_trial_durations_df, 'session')
    all_stop_df = make_variations_utils.assign_session_id(
        all_stop_df, 'session')
    all_VBLO_df = make_variations_utils.assign_session_id(
        all_VBLO_df, 'session')

    return all_trial_durations_df, all_stop_df, all_VBLO_df


def process_all_trial_durations_df(all_trial_durations_df):
    # 1) Filter and clean durations FIRST
    df_trials = all_trial_durations_df.query("duration_sec < 30").copy()
    df_trials["duration_sec"] = df_trials["duration_sec"].clip(lower=1e-6)
    df_trials["logT"] = np.log(df_trials["duration_sec"])

    # 2) Build session-level aggregates from the cleaned trials
    df_sessions = (
        df_trials.groupby("session", as_index=False)
        .agg(captures=("duration_sec", "size"),
             total_duration=("duration_sec", "sum"))
    )

    # total_duration should be >0. Still, be safe:
    df_sessions["total_duration"] = df_sessions["total_duration"].clip(
        lower=1e-12)

    return df_trials, df_sessions
