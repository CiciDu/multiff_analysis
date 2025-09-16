
from scipy import stats
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_class, plot_variations_utils
from data_wrangling import specific_utils, process_monkey_information, base_processing_class, combine_info_utils, further_processing_class
from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_class

import numpy as np
import pandas as pd
import os

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def get_retries_data_across_sessions(raw_data_dir_name='all_monkey_data/raw_monkey_data', monkey_name='monkey_Bruno',
                                     max_duration=30.0):

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, monkey_name)

    all_retries_df = pd.DataFrame()
    for index, row in sessions_df_for_one_monkey.iterrows():
        data_name = row['data_name']
        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], data_name)
        print(raw_data_folder_path)

        cgt = GUAT_vs_TAFT_class.GUATvsTAFTclass(
            raw_data_folder_path=raw_data_folder_path,)
        cgt.get_monkey_data(already_retrieved_ok=True, include_ff_dataframe=False, include_GUAT_data=True,
                            include_TAFT_data=True)

        cgt.GUAT_trials_df['old_miss_index'] = np.arange(
            len(cgt.GUAT_trials_df))
        new_trials_df, cap_old_to_new, miss_old_to_new = reindex_trials_with_misses(
            cgt.ff_caught_T_new, cgt.GUAT_trials_df['last_stop_time'].values)

        cgt.GUAT_trials_df['new_trial_index'] = cgt.GUAT_trials_df['old_miss_index'].map(
            miss_old_to_new)
        cgt.TAFT_trials_df['new_trial_index'] = cgt.TAFT_trials_df['trial'].map(
            cap_old_to_new)

        retries_df = build_retries_df_from_new_trials(
            new_trials_df, cgt.GUAT_trials_df, cgt.TAFT_trials_df, max_duration=max_duration)

        retries_df['data_name'] = data_name
        all_retries_df = pd.concat(
            [all_retries_df, retries_df], ignore_index=True)

    all_retries_df = make_variations_utils.assign_session_id(
        all_retries_df, 'session')

    return all_retries_df


def summarize_retry_data(retries_df):
    def _rate_per_min(n, denom):
        # leave NaN when denom == 0
        return (n / denom) * 60 if denom > 0 else np.nan

    rows = []

    # TAFT
    n_taft = retries_df.loc[retries_df['type'] == 'TAFT', 'capture'].sum()
    d_taft = float(
        retries_df.loc[retries_df['type'] == 'TAFT', 'new_duration'].sum())
    sw_taft = float(
        retries_df.loc[retries_df['type'] == 'TAFT', 'stop_window'].sum())
    rows.append({
        'group': 'TAFT',
        'num_captures': n_taft,
        'total_new_duration': d_taft,
        'total_stop_window': sw_taft,
        'capture_over_duration': _rate_per_min(n_taft, d_taft),
        'retry_window_captures': _rate_per_min(n_taft, sw_taft),
    })

    # GUAT (rates intentionally 0)
    n_guat = 0
    d_guat = float(
        retries_df.loc[retries_df['type'] == 'GUAT', 'new_duration'].sum())
    sw_guat = float(
        retries_df.loc[retries_df['type'] == 'GUAT', 'stop_window'].sum())
    rows.append({
        'group': 'GUAT',
        'num_captures': n_guat,
        'total_new_duration': d_guat,
        'total_stop_window': sw_guat,
        'capture_over_duration': _rate_per_min(n_guat, d_guat),
        'retry_window_captures': _rate_per_min(n_guat, sw_guat),
    })

    # both = TAFT count only; duration/stop_window = TAFT+GUAT
    n_both = n_taft
    d_both = float(retries_df.loc[retries_df['type'].isin(
        ['TAFT', 'GUAT']), 'new_duration'].sum())
    sw_both = float(retries_df.loc[retries_df['type'].isin(
        ['TAFT', 'GUAT']), 'stop_window'].sum())
    rows.append({
        'group': 'both',
        'num_captures': n_both,
        'total_new_duration': d_both,
        'total_stop_window': sw_both,
        'capture_over_duration': _rate_per_min(n_both, d_both),
        'retry_window_captures': _rate_per_min(n_both, sw_both),
    })

    # rest
    n_rest = retries_df.loc[retries_df['type'] == 'rest', 'capture'].sum()
    d_rest = float(
        retries_df.loc[retries_df['type'] == 'rest', 'new_duration'].sum())
    rows.append({
        'group': 'rest',
        'num_captures': n_rest,
        'total_new_duration': d_rest,
        'total_stop_window': 0.0,
        'capture_over_duration': _rate_per_min(n_rest, d_rest),
        'retry_window_captures': np.nan,
    })

    # all = base intervals; stop_window only from TAFT+GUAT
    n_all = retries_df['capture'].sum()
    d_all = float(retries_df['new_duration'].sum())
    sw_all = sw_both
    rows.append({
        'group': 'all',
        'num_captures': n_all,
        'total_new_duration': d_all,
        'total_stop_window': sw_all,
        'capture_over_duration': _rate_per_min(n_all, d_all),
        'retry_window_captures': np.nan,
    })

    retries_summary = pd.DataFrame(rows, columns=[
        'group', 'num_captures', 'total_new_duration',
        'total_stop_window', 'capture_over_duration', 'retry_window_captures'
    ])

    return retries_summary


# --- stop-window helpers ---
def add_stop_window(df, label):
    if 'trial_index' not in df.columns:
        df['trial_index'] = df['trial']
    if df is None or df.empty:
        return pd.DataFrame(columns=['trial_index', 'stop_window', 'type'])
    sub = df[['trial_index', 'first_stop_time', 'last_stop_time']].copy()
    sub['stop_window'] = (sub['last_stop_time'] -
                          sub['first_stop_time']).astype(float)
    sub = sub[['trial_index', 'stop_window']].copy()
    sub['type'] = label
    return sub


def reindex_trials_with_misses(capture_times, fail_times):
    """
    Merge capture and failed-attempt times into one ordered sequence with a new 0-based index.
    Raises ValueError if any timestamp appears in both arrays.

    Returns
    -------
    new_trials_df : pd.DataFrame
        Columns: ['new_trial_index','time','event','old_miss_index','old_trial_index']
        - 'old_trial_index' = original capture index (0-based) for captures; NaN for fails
        - 'old_miss_index' = original fail index (0-based) for fails; NaN for captures
        - 'event' ∈ {'capture','miss'}
    cap_old_to_new : dict[int,int]
        Map old capture index (0-based) -> new_trial_index (0-based)
    miss_old_to_new : dict[int,int]
        Map old fail index (0-based) -> new_trial_index (0-based)
    """
    cap = np.asarray(capture_times, dtype=float)
    fail = np.asarray(fail_times, dtype=float)

    # optional sanity (kept lightweight)
    assert np.isfinite(cap).all() and np.isfinite(
        fail).all(), 'non-finite timestamps found'

    # hard check: no exact duplicates across sets
    common_vals = np.intersect1d(cap, fail)
    if common_vals.size > 0:
        parts = []
        for v in common_vals:
            cap_idx = np.where(cap == v)[0].tolist()
            fail_idx = np.where(fail == v)[0].tolist()
            parts.append(f'{v} (cap idx {cap_idx}, fail idx {fail_idx})')
        raise ValueError(
            'duplicate timestamps across capture and fail: ' + '; '.join(parts))

    events = []
    # captures
    for i, t in enumerate(cap, start=0):
        events.append({'time': float(t), 'event': 'capture',
                       'old_miss_index': np.nan, 'old_trial_index': i, '_seq': i})
    # fails
    for j, t in enumerate(fail, start=0):
        events.append({'time': float(t), 'event': 'miss',
                       'old_miss_index': j, 'old_trial_index': np.nan, '_seq': j})

    # sort by time, then event label (redundant given the hard check), then original order
    events.sort(key=lambda d: (d['time'], d['event'], d['_seq']))

    trials = pd.DataFrame(events)
    trials.insert(0, 'new_trial_index', np.arange(len(trials), dtype=int))

    # mappings (explicit subset in dropna)
    cap_old_to_new = (trials.loc[trials['event'] == 'capture', ['old_trial_index', 'new_trial_index']]
                      .dropna(subset=['old_trial_index'])
                      .astype({'old_trial_index': int})
                      .set_index('old_trial_index')['new_trial_index']
                      .to_dict())

    miss_old_to_new = (trials.loc[trials['event'] == 'miss', ['old_miss_index', 'new_trial_index']]
                       .dropna(subset=['old_miss_index'])
                       .astype({'old_miss_index': int})
                       .set_index('old_miss_index')['new_trial_index']
                       .to_dict())

    new_trials_df = trials.drop(columns=['_seq'])[
        ['new_trial_index', 'time', 'event', 'old_miss_index', 'old_trial_index']
    ]
    # keep both as nullable Int64 for clean joins
    new_trials_df['old_trial_index'] = new_trials_df['old_trial_index'].astype(
        'Int64')
    new_trials_df['old_miss_index'] = new_trials_df['old_miss_index'].astype(
        'Int64')

    return new_trials_df, cap_old_to_new, miss_old_to_new


def build_retries_df_from_new_trials(new_trials_df, GUAT_trials_df, TAFT_trials_df, max_duration=30.0):
    """
    Build retries_df from merged event timeline (event→event durations, 0-based indices).

    Output columns:
      - new_trial_index : merged event index (0-based)
      - old_trial_index : original capture index (0-based; NaN for misses)
      - old_miss_index  : original miss index (0-based; NaN for captures)
      - duration        : event→event seconds (first event dropped)
      - stop_window     : TAFT/GUAT first→last stop span; 0 for rest
      - type            : {'TAFT','GUAT','rest'}
      - type_combined   : 'both' if TAFT or GUAT else 'rest'
      - capture         : 1 for TAFT/rest, 0 for GUAT
    """
    # ---- 0) Event→event duration; filter at the EVENT level (keep 0 durations) ----
    nt = new_trials_df.sort_values('new_trial_index').copy()
    nt['new_duration'] = nt['time'].diff()  # first event has NaN
    nt = nt.iloc[1:].copy()                 # drop first (no predecessor)
    nt = nt.loc[nt['new_duration'] <= max_duration].copy()

    def _ensure_stop_window(df):
        if df is None or len(df) == 0:
            return df
        if 'stop_window' not in df.columns:
            if {'first_stop_time', 'last_stop_time'}.issubset(df.columns):
                df = df.copy()
                df['stop_window'] = (
                    df['last_stop_time'] - df['first_stop_time']).astype(float)
            else:
                df = df.copy()
                df['stop_window'] = np.nan
        return df

    # (fixed swap)
    TAFT_df = _ensure_stop_window(TAFT_trials_df)
    GUAT_df = _ensure_stop_window(GUAT_trials_df)

    # ---- 2) TAFT rows: capture events that are TAFT-labeled (inner join) ----
    taft_rows = nt.merge(
        TAFT_df[['new_trial_index', 'stop_window']], on='new_trial_index', how='inner')
    taft_rows['type'] = 'TAFT'

    # ---- 3) GUAT rows: miss events that are GUAT-labeled (inner join) ----
    guat_rows = nt.merge(
        GUAT_df[['new_trial_index', 'stop_window']], on='new_trial_index', how='inner')
    guat_rows['type'] = 'GUAT'

    # ---- 4) REST rows: capture events NOT TAFT ----
    taft_newidx = set(taft_rows['new_trial_index'].astype(
        int)) if len(taft_rows) else set()
    guat_newidx = set(guat_rows['new_trial_index'].astype(
        int)) if len(guat_rows) else set()
    blocked = taft_newidx | guat_newidx

    rest_rows = nt[(nt['event'] == 'capture') & ~
                   nt['new_trial_index'].isin(blocked)].copy()
    rest_rows['stop_window'] = 0.0
    rest_rows['type'] = 'rest'

    # ---- 5) Assemble; preserve IDs and every qualifying new_trial_index ----
    keep = ['new_trial_index', 'new_duration', 'stop_window',
            'type', 'old_trial_index', 'old_miss_index']
    retries_df = (pd.concat([taft_rows[keep], guat_rows[keep], rest_rows[keep]], ignore_index=True)
                    .sort_values(['new_trial_index', 'type'])
                    .reset_index(drop=True))

    retries_df['type_combined'] = np.where(
        retries_df['type'].isin(['TAFT', 'GUAT']), 'both', 'rest')
    retries_df['capture'] = (retries_df['type'] != 'GUAT').astype(int)

    retries_df['old_trial_index'] = retries_df['old_trial_index'].astype(
        'Int64')
    retries_df['old_miss_index'] = retries_df['old_miss_index'].astype('Int64')

    # Sanity: events should be disjoint across TAFT/GUAT/REST
    assert not retries_df['new_trial_index'].duplicated(
        keep=False).any(), 'duplicated new_trial_index found'

    return retries_df


def get_retry_window_captures(all_retries_df):
    # retries_summary = summarize_retry_data(all_retries_df)
    retry_window_captures = all_retries_df[all_retries_df['type'].isin(
        ['GUAT', 'TAFT'])].copy()
    retry_window_captures = retry_window_captures[[
        'session', 'capture', 'stop_window']].groupby('session').sum().reset_index(drop=False)
    retry_window_captures.rename(
        columns={'capture': 'captures', 'stop_window': 'total_duration'}, inplace=True)
    return retry_window_captures
