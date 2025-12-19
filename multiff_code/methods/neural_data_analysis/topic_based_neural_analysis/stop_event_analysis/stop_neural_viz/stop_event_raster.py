import numpy as np
import pandas as pd


def build_retry_new_seg_info(rsw_events_df, pre_s=0.5, post_s=0.5):
    # Identify dynamically generated stop-related columns
    stop_time_cols = [
        c for c in rsw_events_df.columns
        if c.startswith('stop_') and c.endswith('_time')
    ]
    duration_cols = [
        c for c in rsw_events_df.columns
        if c.startswith('stop_') and c.endswith('_id_duration')
    ]

    # Base columns (unique, order preserved)
    base_cols = stop_time_cols + duration_cols
    if 'num_stops' in rsw_events_df.columns:
        base_cols.append('num_stops')

    cols = list(dict.fromkeys(base_cols))  # remove duplicates

    # Core output frame
    retry_new_seg_info = rsw_events_df[cols + ['last_stop_time']].copy()
    retry_new_seg_info['new_segment'] = np.arange(len(rsw_events_df))

    # Segment bounds: earliest and latest stop time ± padding
    retry_new_seg_info['new_seg_start_time'] = (
        rsw_events_df[stop_time_cols].min(axis=1) - pre_s
    )
    # retry_new_seg_info['new_seg_end_time'] = (
    #     rsw_events_df[stop_time_cols].max(axis=1) + post_s
    # )
    
    retry_new_seg_info['new_seg_end_time'] = retry_new_seg_info['new_seg_start_time'] + 4

    # Event timestamp (first stop)
    retry_new_seg_info['event_time'] = retry_new_seg_info['stop_1_time']

    # Build *_end_time columns
    stop_end_cols = []
    for st_col in stop_time_cols:

        # Matching duration column (primary or fallback)
        dur_col = st_col.replace('_time', '_id_duration')
        if dur_col not in retry_new_seg_info.columns:
            alt_dur_col = st_col.replace('_time', '_duration')
            if alt_dur_col in retry_new_seg_info.columns:
                dur_col = alt_dur_col
            else:
                # No duration available → return NA end time
                end_col = st_col.replace('_time', '_end_time')
                retry_new_seg_info[end_col] = pd.NA
                stop_end_cols.append(end_col)
                continue

        # Compute stop end time
        end_col = st_col.replace('_time', '_end_time')
        start_vals = pd.to_numeric(retry_new_seg_info[st_col], errors='coerce')
        dur_vals = pd.to_numeric(retry_new_seg_info[dur_col], errors='coerce')
        retry_new_seg_info[end_col] = start_vals + dur_vals
        stop_end_cols.append(end_col)

    return retry_new_seg_info, stop_time_cols, stop_end_cols
