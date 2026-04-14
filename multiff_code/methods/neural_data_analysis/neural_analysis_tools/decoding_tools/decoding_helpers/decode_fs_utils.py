import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_utils
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils
from neural_data_analysis.design_kits.design_around_event import stop_design

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils



def build_fs_design_decoding(
    pn):
    
    """
    Build stop design for decoding (minimal event design, no interaction columns).
    """
    

    pn.retrieve_neural_data()
    pn.retrieve_or_make_monkey_data()
    pn.spikes_df = neural_data_processing.make_spikes_df(pn.raw_data_folder_path, pn.ff_caught_T_sorted,
                                                            pn.monkey_information, sampling_rate=pn.sampling_rate)


    new_seg_info = make_new_seg_info_for_fs(pn)
    
    
    if not hasattr(pn, 'ff_dataframe'):
        pn.make_or_retrieve_ff_dataframe()

    pn.monkey_information = add_time_since_capture_or_stop(pn.monkey_information)
    pn.monkey_information, vis_events_with_stats = add_columns_related_to_time_since_ff_visible(
        pn
    )

    (
        bins_2d,
        _meta,
        binned_feats,
        _exposure,
        _used_bins,
        _mask_used,
        pos,
        meta_df_used,
    ) = encoding_design_utils.bin_event_windows_core(
        new_seg_info=new_seg_info,
        monkey_information=pn.monkey_information,
        bin_dt=pn.bin_width,
        verbose=True,
        tile_window=True,
        agg_cols=decoding_design_utils.ONE_FF_STYLE_DECODING_COLS + [
            'time_since_prev_stop', 'time_since_prev_capture',
            'time_since_prev_ff_visible', 'time_since_global_burst_start',
            'num_ff_visible', 'log1p_num_ff_visible',
            'num_ff_in_memory', 'log1p_num_ff_in_memory',
        ],
    )
    
    binned_feats = add_ff_visibility_cols_to_binned_feats(
        binned_feats, pn, vis_events_with_stats, meta_df_used
    )

    return binned_feats, meta_df_used, bins_2d, pos, new_seg_info


def make_new_seg_info_for_fs(pn):

    new_seg_info = pd.DataFrame({
        'event_id': 0,
        'event_time': max(0, pn.ff_caught_T_sorted.min() - 1),
        'new_seg_start_time': max(0, pn.ff_caught_T_sorted.min() - 1),
        'new_seg_end_time': pn.ff_caught_T_sorted.max(),
        'new_seg_duration':
            pn.ff_caught_T_sorted.max()
            - max(0, pn.ff_caught_T_sorted.min() - 1),
        'n_pre_bins': 0,
        'n_post_bins': 0,
    }, index=[0])
    
    return new_seg_info
    
    

def add_columns_related_to_time_since_ff_visible(pn):
    
    _, vis_events_with_stats = decode_vis_utils.prepare_new_seg_info(pn.ff_dataframe)
    
    vis_events_with_stats.rename(columns={'event_id_start_time': 'ff_vis_start_time',
                                    'event_id_end_time': 'ff_vis_end_time',
                                    }, inplace=True, errors='ignore')

    pn.monkey_information = add_columns_related_to_ff_visibility(
        pn.monkey_information, pn.ff_dataframe, vis_events_with_stats
    )
    
    return pn.monkey_information, vis_events_with_stats


def add_ff_visibility_cols_to_binned_feats(binned_feats, pn, vis_events_with_stats, meta_df_used):

    binned_feats['ff_vis_start'] = encoding_design_utils.make_event_dummy(
        binned_feats,
        vis_events_with_stats['ff_vis_start_time']
    )

    binned_feats['ff_vis_end'] = encoding_design_utils.make_event_dummy(
        binned_feats,
        vis_events_with_stats['ff_vis_end_time']
    )

    binned_feats['global_burst_start'] = encoding_design_utils.make_event_dummy(
        binned_feats,
        vis_events_with_stats['global_burst_start_time']
    )

    ff_on_df, group_on_df = decode_vis_utils.extract_ff_visibility_tables_fast(
        pn.ff_dataframe
    )

    binned_feats = decode_vis_utils._add_ff_visibility_onehot_to_binned_feats(
        binned_feats,
        meta_df_used,
        ff_on_df,
        group_on_df,
    )

    binned_feats.rename(columns={
        'ff_on_in_bin': 'ff_on',
        'ff_off_in_bin': 'ff_off',
        'group_ff_on': 'group_ff_on',
        'group_ff_off': 'group_ff_off',
    }, inplace=True, errors='ignore')
    
    return binned_feats




def add_time_since_capture_or_stop(df, clip_percentile=95):
    """
    Adds:
        - time_since_prev_stop
        - time_since_prev_capture

    Assumes columns:
        'time', 'whether_new_distinct_stop', 'capture'

    Args:
        clip_percentile: if not None, clips the added columns at this percentile.
    """
    df = df.copy()

    if 'stop' not in df.columns and 'whether_new_distinct_stop' in df.columns:
        df['stop'] = df['whether_new_distinct_stop'].to_numpy()
        
    # --- helper ---
    def compute_time_since_event(time, event_mask):
        last_event_time = time.where(event_mask).ffill()
        return time - last_event_time

    # --- compute ---
    df['time_since_prev_stop'] = compute_time_since_event(
        df['time'],
        df['whether_new_distinct_stop']
    )

    df['time_since_prev_capture'] = compute_time_since_event(
        df['time'],
        df['capture'] == 1
    )

    # fill NaNs ONLY in the new columns
    df['time_since_prev_stop'] = df['time_since_prev_stop'].fillna(0)
    df['time_since_prev_capture'] = df['time_since_prev_capture'].fillna(0)

    if clip_percentile is not None:
        for col in ['time_since_prev_stop', 'time_since_prev_capture']:
            cap = df[col].quantile(clip_percentile / 100)
            df[col] = df[col].clip(upper=cap)

    return df


def add_time_since_ff_visible(df, ff_vis_df, clip_percentile=95):
    """
    df: has 'time'; will add 'time_since_prev_ff_visible'
    ff_vis_df: has 'ff_vis_start_time', 'ff_vis_end_time'

    Args:
        clip_percentile: if not None, clips 'time_since_prev_ff_visible' at this percentile.
    """
    # if 'time_since_prev_ff_visible' in df.columns:
    #     return df
    
    
    df = df.copy()

    # sort (required)
    df = df.sort_values('time')
    ff_vis_df = ff_vis_df.sort_values('ff_vis_start_time')
    
    df.drop(columns=['ff_vis_start_time', 'ff_vis_end_time'], inplace=True, errors='ignore')

    # --- 1. find last interval start before each time
    df = pd.merge_asof(
        df,
        ff_vis_df[['ff_vis_start_time', 'ff_vis_end_time']],
        left_on='time',
        right_on='ff_vis_start_time',
        direction='backward'
    )

    # --- 2. check if currently inside a visible interval
    is_visible = (
        (df['time'] >= df['ff_vis_start_time']) &
        (df['time'] <= df['ff_vis_end_time'])
    )
    
    # --- 3. compute time since last visibility END
    # get last end time (forward-filled)
    last_end = df['ff_vis_end_time'].where(~df['ff_vis_end_time'].isna()).ffill()

    df['time_since_prev_ff_visible'] = df['time'] - last_end

    # --- 4. override: if currently visible → 0
    df.loc[is_visible, 'time_since_prev_ff_visible'] = 0

    # --- 5. handle before first interval
    df['time_since_prev_ff_visible'] = df['time_since_prev_ff_visible'].fillna(0)

    if clip_percentile is not None:
        cap = df['time_since_prev_ff_visible'].quantile(clip_percentile / 100)
        df['time_since_prev_ff_visible'] = df['time_since_prev_ff_visible'].clip(upper=cap)

    df.drop(columns=['ff_vis_start_time', 'ff_vis_end_time'], inplace=True)

    return df


def add_time_since_global_burst_start(df, burst_df, clip_percentile=95):
    """
    Args:
        clip_percentile: if not None, clips 'time_since_global_burst_start' at this percentile.
    """
    # if 'global_burst_start_time' in df.columns:
    #     return df

    df = df.copy()
    # prepare burst times
    burst_times = (
        burst_df['global_burst_start_time']
        .drop_duplicates()
        .sort_values()
    )
    
    df.drop(columns=['global_burst_start_time'], inplace=True, errors='ignore')

    # align last burst start
    df = pd.merge_asof(
        df.sort_values('time'),
        burst_times.to_frame(name='global_burst_start_time'),
        left_on='time',
        right_on='global_burst_start_time',
        direction='backward'
    )

    # compute time since
    df['time_since_global_burst_start'] = (
        df['time'] - df['global_burst_start_time']
    ).fillna(0)

    if clip_percentile is not None:
        cap = df['time_since_global_burst_start'].quantile(clip_percentile / 100)
        df['time_since_global_burst_start'] = df['time_since_global_burst_start'].clip(upper=cap)

    df.drop(columns=['global_burst_start_time'], inplace=True, errors='ignore')

    return df


def add_columns_related_to_ff_visibility(monkey_information, ff_dataframe, events_with_stats, clip_percentile=95):

    monkey_information = add_time_since_ff_visible(monkey_information, events_with_stats, clip_percentile=clip_percentile)

    monkey_information = add_time_since_global_burst_start(monkey_information, events_with_stats, clip_percentile=clip_percentile)

    monkey_information = pn_utils.add_ff_visible_or_in_memory_info_by_point(
        monkey_information, ff_dataframe)

    return monkey_information