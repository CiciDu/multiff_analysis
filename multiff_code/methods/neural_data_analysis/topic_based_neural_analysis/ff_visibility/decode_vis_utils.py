
from neural_data_analysis.design_kits.design_by_segment import create_pn_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import ff_vis_epochs
import neural_data_analysis.design_kits.design_around_event.event_binning as event_binning
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import ff_vis_epochs, vis_design

import numpy as np
import pandas as pd

def init_decoding_data(raw_data_folder_path,
                       cur_or_nxt='cur',
                       first_or_last='first',
                       time_limit_to_count_sighting=2,
                       start_t_rel_event=0,
                       end_t_rel_event=1.5,
                       end_at_stop_time=False):

    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path)

    pn.prep_data_to_analyze_planning(
        planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)

    pn.rebin_data_in_new_segments(
        cur_or_nxt=cur_or_nxt,
        first_or_last=first_or_last,
        time_limit_to_count_sighting=time_limit_to_count_sighting,
        start_t_rel_event=start_t_rel_event,
        end_t_rel_event=end_t_rel_event,
        end_at_stop_time=end_at_stop_time,
    )

    for col in ['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory']:
        pn.rebinned_y_var[col] = (pn.rebinned_y_var[col] > 0).astype(int)

    return pn


def get_data_for_decoding_vis(rebinned_x_var, rebinned_y_var, dt):

    data = rebinned_y_var.copy()
    trial_ids = data['new_segment']

    design_df, meta0, meta = create_pn_design_df.get_pn_design_base(
        data, dt, trial_ids)

    df_X = design_df[
        [
            'speed_z',
            'time_since_last_capture',
            'ang_accel_mag_spline:s0',
            'ang_accel_mag_spline:s1',
            'ang_accel_mag_spline:s2',
            'ang_accel_mag_spline:s3',
            'cur_vis',
            'nxt_vis',
        ]
    ].copy()

    # neural matrix
    cluster_cols = [
        c for c in rebinned_x_var.columns if c.startswith('cluster_')]
    df_Y = rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)

    return df_X, df_Y


def prepare_new_seg_info(ff_dataframe, bin_width=0.04):

    # minimal: detect runs, no merging (each run is its own cluster)
    df2 = ff_vis_epochs.compute_visibility_runs_and_clusters(
        ff_dataframe.copy(), ff_col='ff_index', t_col='point_index', time_col='time', vis_col='visible',
        chunk_merge_gap=0.05,    # seconds: merge *raw* runs into chunks if gap <= this
        cluster_merge_gap=1
    )

    df2 = ff_vis_epochs.add_global_visibility_bursts(
        df2, global_merge_gap=0.25)
    # df2 = ff_vis_epochs.add_global_vis_cluster_id(df2, group_cols=None, nullable_int=True)
    df2 = ff_vis_epochs.add_global_vis_chunk_id(
        df2, group_cols=None, nullable_int=True)
    # df2 = ff_vis_epochs.add_global_vis_cluster_id(df2, group_cols=None, nullable_int=True)

    vis_df = df2.loc[df2['visible'] == 1].copy()

    # based on any ff visible
    sequential_vis_df = vis_df[['ff_index', 'ff_vis_start_time', 'ff_vis_end_time',
                                'global_vis_chunk_id', 'global_burst_id', 'global_burst_start_time', 'global_burst_end_time',
                                'global_burst_duration', 'global_burst_size',
                                # 'global_burst_prev_start_time','global_burst_prev_end_time'
                                ]].drop_duplicates().reset_index(drop=True)
    sequential_vis_df = sequential_vis_df.sort_values(
        'ff_vis_start_time').reset_index(drop=True)
    sequential_vis_df['prev_time'] = sequential_vis_df['ff_vis_start_time'].shift(
        1)
    sequential_vis_df['next_time'] = sequential_vis_df['ff_vis_start_time'].shift(
        -1)

    new_seg_info = event_binning.pick_event_window(sequential_vis_df,
                                                   event_time_col='ff_vis_start_time',
                                                   prev_event_col='prev_time',
                                                   next_event_col='next_time',
                                                   pre_s=0.1, post_s=0.5, min_pre_bins=2, min_post_bins=3, bin_dt=bin_width)
    new_seg_info['event_id'] = new_seg_info['global_vis_chunk_id']
    new_seg_info['event_time'] = new_seg_info['ff_vis_start_time']

    events_with_stats = sequential_vis_df[[
        'global_vis_chunk_id', 'global_burst_id', 'ff_vis_start_time', 'ff_vis_end_time']].copy()
    events_with_stats = sequential_vis_df.rename(columns={'global_vis_chunk_id': 'event_id',
                                                          'global_burst_id': 'event_cluster_id',
                                                          'ff_vis_start_time': 'event_id_start_time',
                                                          'ff_vis_end_time': 'event_id_end_time'})

    return new_seg_info, events_with_stats



def extract_ff_visibility_tables_fast(ff_dataframe):

    df = ff_dataframe[['ff_index', 'time', 'visible']].sort_values(
        ['ff_index', 'time']
    )

    ff_index = df['ff_index'].to_numpy()
    time = df['time'].to_numpy()
    visible = df['visible'].to_numpy().astype(np.int8)

    ff_on_rows = []

    # detect FF boundaries
    ff_change = np.flatnonzero(ff_index[1:] != ff_index[:-1]) + 1
    ff_start = np.r_[0, ff_change]
    ff_end = np.r_[ff_change - 1, len(df) - 1]

    for s, e in zip(ff_start, ff_end):

        t = time[s:e+1]
        v = visible[s:e+1]

        change = np.flatnonzero(v[1:] != v[:-1]) + 1
        run_start = np.r_[0, change]
        run_end = np.r_[change - 1, len(v) - 1]

        run_state = v[run_start]

        for rs, re, st in zip(run_start, run_end, run_state):
            if st == 1:
                ff_on_rows.append((ff_index[s], t[rs], t[re]))

    ff_on_df = pd.DataFrame(
        ff_on_rows,
        columns=['ff_index', 'ff_vis_start_time', 'ff_vis_end_time']
    )

    # -------- group ON --------

    group_vis = (
        ff_dataframe
        .groupby('time')['visible']
        .max()
        .sort_index()
    )

    t = group_vis.index.to_numpy()
    v = group_vis.to_numpy().astype(np.int8)

    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    run_start = np.r_[0, change]
    run_end = np.r_[change - 1, len(v) - 1]

    group_rows = []

    for rs, re in zip(run_start, run_end):
        if v[rs] == 1:
            group_rows.append((t[rs], t[re]))

    group_on_df = pd.DataFrame(
        group_rows,
        columns=['group_on_start_time', 'group_on_end_time']
    )

    return ff_on_df, group_on_df


def prepare_new_seg_info_with_visibility_tables(ff_dataframe, bin_width=0.04):
    '''
    Keeps original prepare_new_seg_info functionality unchanged,
    and additionally returns fast ON/OFF visibility tables.
    '''
    new_seg_info, events_with_stats = prepare_new_seg_info(ff_dataframe, bin_width)

    ff_on_df, ff_off_df, group_on_df, group_off_df = extract_ff_visibility_tables_fast(ff_dataframe)

    visibility_tables = {
        'ff_on_df': ff_on_df,
        'ff_off_df': ff_off_df,
        'group_on_df': group_on_df,
        'group_off_df': group_off_df
    }

    return new_seg_info, events_with_stats, visibility_tables


def _add_ff_visibility_onehot_to_binned_feats(
    binned_feats: pd.DataFrame,
    meta_df_used: pd.DataFrame,
    ff_on_df: pd.DataFrame,
    group_on_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add one-hot binned indicators for ff_on, ff_off, group_ff_on, group_ff_off.

    For each bin, indicator = 1 if any event of that type falls in the bin, else 0.
    """
    bins_2d_used = np.column_stack([
        meta_df_used['t_left'].to_numpy(dtype=float),
        meta_df_used['t_right'].to_numpy(dtype=float),
    ])

    vis_specs = [
        (ff_on_df['ff_vis_start_time'].to_numpy(dtype=float), 'ff_on_in_bin'),
        (ff_on_df['ff_vis_end_time'].to_numpy(dtype=float), 'ff_off_in_bin'),
        (group_on_df['group_on_start_time'].to_numpy(dtype=float), 'group_ff_on'),
        (group_on_df['group_on_end_time'].to_numpy(dtype=float), 'group_ff_off'),
    ]

    for event_times, col_name in vis_specs:
        event_times = event_times[np.isfinite(event_times)]
        if event_times.size == 0:
            binned_feats[col_name] = 0
            continue

        bin_idx = vis_design.map_times_to_bin_idx_unsorted(bins_2d_used, event_times)
        valid = bin_idx >= 0
        indicator = np.zeros(len(binned_feats), dtype=np.int8)
        if valid.any():
            np.add.at(indicator, bin_idx[valid], 1)
        indicator = (indicator > 0).astype(np.int8)
        binned_feats[col_name] = indicator

    return binned_feats
