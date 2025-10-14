
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
from pathlib import Path


def get_key_learning_data(
    monkey_name='monkey_Bruno',
    exists_ok=True,
    output_base_dir='all_monkey_data/learning',
    raw_data_dir_name='all_monkey_data/raw_monkey_data',
    verbose=False,
):
    """Get key learning data for a monkey with robust caching.

    Args:
        monkey_name (str): Name of the monkey to get data for.
        exists_ok (bool): If True, attempt to load cached CSVs. If False, recompute and overwrite.
        output_base_dir (str): Base directory to store per-monkey outputs.
        raw_data_dir_name (str): Root directory containing raw data per monkey.
        verbose (bool): If True, prints basic progress messages.
    """
    file_dir = Path(output_base_dir) / monkey_name
    file_dir.mkdir(parents=True, exist_ok=True)

    trial_path = file_dir / 'all_trial_durations_df.csv'
    stop_path = file_dir / 'all_stop_df.csv'
    vblo_path = file_dir / 'all_VBLO_df.csv'

    def _load_if_exists(path: Path):
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, index_col=False)
        except Exception:
            return None

    if exists_ok:
        all_trial_durations_df = _load_if_exists(trial_path)
        all_stop_df = _load_if_exists(stop_path)
        all_VBLO_df = _load_if_exists(vblo_path)
        if all_trial_durations_df is not None and all_stop_df is not None and all_VBLO_df is not None:
            if verbose:
                print(
                    f"Loaded cached learning data for {monkey_name} from {file_dir}")
            return all_trial_durations_df, all_stop_df, all_VBLO_df

    if verbose:
        print(
            f"Computing learning data for {monkey_name} (exists_ok={exists_ok}) …")

    all_trial_durations_df, all_stop_df, all_VBLO_df = _get_key_learning_data(
        raw_data_dir_name=raw_data_dir_name, monkey_name=monkey_name
    )

    # Persist results
    all_trial_durations_df.to_csv(trial_path, index=False)
    all_stop_df.to_csv(stop_path, index=False)
    all_VBLO_df.to_csv(vblo_path, index=False)
    if verbose:
        print(f"Saved learning data to {file_dir}")
    return all_trial_durations_df, all_stop_df, all_VBLO_df


def _get_key_learning_data(raw_data_dir_name='all_monkey_data/raw_monkey_data', monkey_name='monkey_Bruno'):

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
            {'duration_sec': trial_durations, 'trial_index': np.arange(1, len(trial_durations) + 1)})  # trial_index starts from 1 since we don't calculate duration for the first trial
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
