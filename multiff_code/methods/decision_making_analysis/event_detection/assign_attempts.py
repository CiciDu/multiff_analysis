import pandas as pd
import numpy as np

from decision_making_analysis.event_detection import detect_rsw_and_rcap
from decision_making_analysis.data_enrichment import rsw_vs_rcap_utils
from decision_making_analysis.event_detection import get_miss_to_switch_data
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data


def make_stop_category_df(monkey_information, ff_caught_T_new, closest_stop_to_capture_df, temp_rcap_events_df, ff_dataframe, ff_real_position_sorted):
    stop_category_df = init_stop_category_df(
        monkey_information,
        ff_caught_T_new,
    )

    stop_category_df = assign_candidate_target(
        stop_category_df,
        closest_stop_to_capture_df,
        temp_rcap_events_df,
        ff_real_position_sorted
    )

    stop_category_df = add_misses_to_stop_category_df(stop_category_df, monkey_information,
                                                      ff_dataframe, ff_real_position_sorted)
    stop_category_df = add_stop_cluster_id(stop_category_df)
    stop_category_df = reassign_attempt_types(stop_category_df)

    stop_category_df['stop_id_duration'] = stop_category_df['stop_id_end_time'] - \
        stop_category_df['stop_id_start_time']

    return stop_category_df


def init_stop_category_df(monkey_information, ff_caught_T_new):
    # Base subset of distinct stops
    stop_category_df = monkey_information.loc[
        monkey_information['whether_new_distinct_stop'],
        ['time', 'point_index', 'stop_id', 'stop_id_start_time', 'stop_id_end_time', 'stop_id_duration',
         'temp_stop_cluster_id',
         'monkey_x', 'monkey_y', 'trial']
    ].reset_index(drop=True)

    stop_category_df = stop_category_df[stop_category_df['time'].between(
        ff_caught_T_new[0], ff_caught_T_new[-1])]

    stop_category_df['target_index'] = stop_category_df['trial']

    return stop_category_df


def assign_candidate_target(stop_category_df, closest_stop_to_capture_df, temp_rcap_events_df, ff_real_position_sorted):
    stop_category_df = _merge_capture_info(
        stop_category_df, closest_stop_to_capture_df)
    stop_category_df = _add_rcap_info(stop_category_df, temp_rcap_events_df)

    # If you want a single 'candidate_target' preferring capture-unique, else  rcap:
    stop_category_df['candidate_target'] = stop_category_df['cap_candidate_target'].fillna(
        stop_category_df['rcap_candidate_target']
    )

    stop_category_df = _deal_with_stops_close_to_targets(
        stop_category_df, ff_real_position_sorted)

    stop_category_df = _categorize_based_on_candidate_target(stop_category_df)

    stop_category_df.drop(
        columns=['cap_candidate_target', 'rcap_candidate_target'], inplace=True)

    return stop_category_df


def _merge_capture_info(stop_category_df, closest_stop_to_capture_df):
    # Per-stop capture aggregation
    # Note: if a stop resulted in 2 ff, we only use the smaller index of the ff
    cap_agg = (
        closest_stop_to_capture_df
        .groupby('stop_id')
        .agg(
            num_capture=('stop_id', 'size'),
            cap_candidate_target=('cur_ff_index', 'min')
        )
        .reset_index()
    )

    # Merge capture info onto stop_category_df
    stop_category_df = stop_category_df.merge(
        cap_agg, on='stop_id', how='left')
    stop_category_df['num_capture'] = stop_category_df['num_capture'].fillna(
        0).astype(int)
    return stop_category_df


def _add_rcap_info(stop_category_df, temp_rcap_events_df):

    # assert that there's no duplicated combo of temp_rcap_events_df and trial in temp_rcap_events_df
    assert len(temp_rcap_events_df[temp_rcap_events_df.duplicated(
        subset=['temp_stop_cluster_id', 'trial'])]) == 0

    temp_rcap_events_df = temp_rcap_events_df.copy()
    #  rcap: add per-trial/cluster associated target (different name to avoid collision)
    temp_rcap_events_df['rcap_candidate_target'] = temp_rcap_events_df['trial']

    stop_category_df = stop_category_df.merge(
        temp_rcap_events_df[['temp_stop_cluster_id', 'rcap_candidate_target']
                            ].drop_duplicates('temp_stop_cluster_id'),
        on='temp_stop_cluster_id',
        how='left'
    )

    return stop_category_df


def _deal_with_stops_close_to_targets(stop_category_df, ff_real_position_sorted, distance_to_target=50):
    stop_category_df = detect_rsw_and_rcap._add_target_distances(
        stop_category_df, ff_real_position_sorted, trial_col='target_index', offsets=(0,))

    # for each row in stop_category_df, if candidate_target is NA, and if distance_to_target_+0 < 50, then assign 'target_index' as associated target
    # make a copy so we don't overwrite unintentionally

    mask = (
        stop_category_df['candidate_target'].isna()
        & (stop_category_df['distance_to_target_+0'] < distance_to_target)
    )

    # replace 'ff_index' with the column that has the trial’s own target index
    stop_category_df.loc[mask,
                         'candidate_target'] = stop_category_df.loc[mask, 'target_index']

    stop_category_df.drop(columns=['distance_to_target_+0'], inplace=True)

    return stop_category_df


def _categorize_based_on_candidate_target(stop_category_df):
    mask = stop_category_df['candidate_target'].notna()

    # Group size per candidate_target
    sizes = stop_category_df.loc[mask].groupby('candidate_target').size()
    stop_category_df['assoc_group_size'] = pd.NA
    stop_category_df.loc[mask, 'assoc_group_size'] = stop_category_df.loc[mask,
                                                                          'candidate_target'].map(sizes)
    stop_category_df['assoc_group_size'] = stop_category_df['assoc_group_size'].astype(
        'Int64')

    stop_category_df['attempt_type'] = pd.NA
    stop_category_df.loc[stop_category_df['assoc_group_size']
                         == 1, 'attempt_type'] = 'capture'
    stop_category_df.loc[stop_category_df['assoc_group_size']
                         > 1, 'attempt_type'] = 'rcap'

    stop_category_df.drop(columns=['assoc_group_size'], inplace=True)

    return stop_category_df


def _take_out_rsw_from_leftover_stops(stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted):
    # take out subset of stops with no associated target
    stop_sub = stop_category_df[stop_category_df['candidate_target'].isna()].copy(
    )

    stop_sub['stop_cluster_size'] = (
        stop_sub.groupby('temp_stop_cluster_id')[
            'temp_stop_cluster_id'].transform('size')
    )

    temp_rsw_sub = stop_sub[stop_sub['stop_cluster_size'] > 1].copy()

    temp_rsw_events_df = detect_rsw_and_rcap._make_events_df(
        temp_rsw_sub, stop_cluster_id_col='temp_stop_cluster_id')

    rsw_indices_df = detect_rsw_and_rcap.find_rsw_or_rcap_info(
        temp_rsw_events_df, monkey_information)

    rsw_ff_info = get_miss_to_switch_data.get_ff_info_for_rsw(rsw_indices_df,
                                                              temp_rsw_events_df,
                                                              ff_dataframe,
                                                              monkey_information,
                                                              ff_real_position_sorted,
                                                              )
    return rsw_ff_info


def _take_out_one_stop_from_leftover_stops(stop_category_df, ff_dataframe, ff_real_position_sorted):
    rest_df = stop_category_df[stop_category_df['candidate_target'].isna()].copy(
    )
    temp_one_stop_df = get_miss_to_switch_data.make_temp_one_stop_df(
        rest_df, ff_dataframe, ff_real_position_sorted,
        eliminate_stops_too_close_to_any_target=False)

    temp_one_stop_w_ff_df = get_miss_to_switch_data.make_temp_one_stop_w_ff_df(
        temp_one_stop_df)

    return temp_one_stop_w_ff_df


def add_misses_to_stop_category_df(stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted):
    # Add rsw misses
    rsw_ff_info = _take_out_rsw_from_leftover_stops(
        stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted)
    rsw_ff_info['rsw_attempt_type'] = 'rsw'
    rsw_ff_info['rsw_candidate_target'] = rsw_ff_info['candidate_target']
    stop_category_df = stop_category_df.merge(rsw_ff_info[[
                                              'temp_stop_cluster_id', 'rsw_candidate_target', 'rsw_attempt_type']], on='temp_stop_cluster_id', how='left')

    stop_category_df['candidate_target'] = stop_category_df['candidate_target'].fillna(
        stop_category_df['rsw_candidate_target'])
    stop_category_df['attempt_type'] = stop_category_df['attempt_type'].fillna(
        stop_category_df['rsw_attempt_type'])

    # Add one-stop misses
    temp_one_stop_w_ff_df = _take_out_one_stop_from_leftover_stops(
        stop_category_df, ff_dataframe, ff_real_position_sorted)
    temp_one_stop_w_ff_df['one_stop_candidate_target'] = temp_one_stop_w_ff_df['candidate_target']
    temp_one_stop_w_ff_df['one_stop_attempt_type'] = 'miss'
    stop_category_df = stop_category_df.merge(temp_one_stop_w_ff_df[[
                                              'stop_id', 'one_stop_candidate_target', 'one_stop_attempt_type']], on='stop_id', how='left')
    stop_category_df['candidate_target'] = stop_category_df['candidate_target'].fillna(
        stop_category_df['one_stop_candidate_target'])
    stop_category_df['attempt_type'] = stop_category_df['attempt_type'].fillna(
        stop_category_df['one_stop_attempt_type'])

    stop_category_df.drop(columns=[
                          'rsw_attempt_type', 'one_stop_attempt_type', 'one_stop_candidate_target'], inplace=True)

    return stop_category_df


def reassign_attempt_types(
    df: pd.DataFrame,
    ff_col: str = 'candidate_target',
    cluster_id_col: str = 'stop_cluster_id',
    cluster_size_col: str = 'stop_cluster_size',
    attempt_col: str = 'attempt_type',
    rcap_label: str = 'rcap',
    capture_label: str = 'capture',
    rsw_label: str = 'rsw',
    one_stop_label: str = 'miss',  # or 'one-stop'
) -> pd.DataFrame:
    """
    Reassign attempt types within each consecutive candidate_target cluster.

    Rules per cluster (only where candidate_target is not NA):
      1) If any rcap -> all  rcap
      2) Else if any capture:
            size > 1  ->  rcap
            size == 1 -> capture
      3) Else:
            size > 1  -> rsw
            size == 1 -> one_stop_label

    For the rest, assign with 'unclassified'
    """
    out = df.copy()
    out.sort_values(by='time', inplace=True)

    # make sure that each new_cluster only has one unique candidate_target (even if it's na)
    assert out.groupby(cluster_id_col)[ff_col].nunique().max() <= 1

    # Work only on rows with candidate_target present
    mask = out[ff_col].notna()
    if not mask.any():
        return out

    # Compute cluster size once; no merges, no duplication
    sizes = out.loc[mask].groupby(cluster_id_col).size()
    out.loc[mask, cluster_size_col] = out.loc[mask, cluster_id_col].map(sizes)

    # Cluster-level flags
    grp = out.loc[mask].groupby(cluster_id_col)[attempt_col]
    has_rcap = grp.apply(lambda s: (s == rcap_label).any())
    has_capture = grp.apply(lambda s: (s == capture_label).any())

    # Build a per-cluster decision table
    cl = pd.DataFrame({
        'size': sizes,
        'has_rcap': has_rcap.reindex(sizes.index, fill_value=False),
        'has_capture': has_capture.reindex(sizes.index, fill_value=False),
    })

    # Decide per rules
    conditions = [
        cl['has_rcap'],
        ~cl['has_rcap'] & cl['has_capture'] & (cl['size'] == 1),
        ~cl['has_rcap'] & cl['has_capture'] & (cl['size'] > 1),
        ~cl['has_rcap'] & ~cl['has_capture'] & (cl['size'] > 1),
    ]
    choices = [rcap_label, capture_label, rcap_label, rsw_label]
    cl['new_label'] = np.select(conditions, choices, default=one_stop_label)

    # Broadcast decision back to rows
    out.loc[mask, attempt_col] = out.loc[mask,
                                         cluster_id_col].map(cl['new_label'])

    # for the rest, assign with 'none'
    out.loc[~mask, attempt_col] = 'unclassified'

    return out


def add_stop_cluster_id(
    stop_category_df: pd.DataFrame,
    ff_col: str = 'candidate_target',
    point_col: str = 'point_index',
    order_by: str | None = 'time',
    id_col: str = 'stop_cluster_id',
    size_col: str = 'stop_cluster_size',
    start_col: str = 'stop_cluster_start_point',
    end_col: str = 'stop_cluster_end_point',
) -> pd.DataFrame:
    """
    Build consecutive clusters over `ff_col` with:
      - 0-based `id_col`
      - `size_col`: cluster size
      - `start_col`/`end_col`: min/max of `point_col` per cluster

    Consecutive is along DataFrame order, or `order_by` if provided.
    Each NaN in `ff_col` becomes its own singleton cluster.
    """
    if ff_col not in stop_category_df.columns:
        raise KeyError(f'missing column: {ff_col}')
    if point_col not in stop_category_df.columns:
        raise KeyError(f'missing column: {point_col}')

    if stop_category_df.empty:
        # still add empty columns for consistency
        out = stop_category_df.copy()
        for c in (id_col, size_col, start_col, end_col):
            out[c] = pd.Series(dtype='Int64')
        return out

    out = stop_category_df.copy()

    # Define the sequence along which "consecutive" is computed
    if order_by is not None:
        out = out.reset_index(drop=False).rename(
            columns={'index': '__orig_idx__'})
        out = out.sort_values(order_by, kind='stable')

    s = out[ff_col]
    changed = s != s.shift()            # NaN always starts a new group
    out[id_col] = (changed.cumsum() - 1).astype('Int64')  # 0-based ids

    # Per-cluster aggregates
    out[size_col] = out.groupby(
        id_col)[id_col].transform('size').astype('Int64')
    out[start_col] = out.groupby(
        id_col)[point_col].transform('min').astype('Int64')
    out[end_col] = out.groupby(
        id_col)[point_col].transform('max').astype('Int64')

    # Restore original order if we sorted
    if order_by is not None:
        out = (
            out.sort_values('__orig_idx__', kind='stable')
               .drop(columns='__orig_idx__')
               .reset_index(drop=True)
        )

    return out
