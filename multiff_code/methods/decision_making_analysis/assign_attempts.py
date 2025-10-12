import pandas as pd
import numpy as np

from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials, GUAT_vs_TAFT_utils
from decision_making_analysis.GUAT import GUAT_utils


def make_stop_category_df(monkey_information, ff_caught_T_new, closest_stop_to_capture_df, temp_TAFT_trials_df, ff_dataframe, ff_real_position_sorted):
    stop_category_df = init_stop_category_df(
        monkey_information,
        ff_caught_T_new,
    )

    stop_category_df = assign_associated_target(
        stop_category_df,
        closest_stop_to_capture_df,
        temp_TAFT_trials_df,
        ff_real_position_sorted
    )

    stop_category_df = add_misses_to_stop_category_df(stop_category_df, monkey_information,
                                                      ff_dataframe, ff_real_position_sorted)
    stop_category_df = add_stop_cluster_id(stop_category_df)
    stop_category_df = reassign_attempt_types(stop_category_df)

    stop_category_df['stop_id_duration'] = stop_category_df['stop_id_end_time'] - stop_category_df['stop_id_start_time']
            
    return stop_category_df


def init_stop_category_df(monkey_information, ff_caught_T_new):
    # Base subset of distinct stops
    stop_category_df = monkey_information.loc[
        monkey_information['whether_new_distinct_stop'],
        ['time', 'point_index', 'stop_id', 'stop_id_start_time', 'stop_id_end_time',
         'temp_stop_cluster_id',
         'monkey_x', 'monkey_y', 'trial']
    ].reset_index(drop=True)

    stop_category_df = stop_category_df[stop_category_df['time'].between(
        ff_caught_T_new[0], ff_caught_T_new[-1])]

    stop_category_df['target_index'] = stop_category_df['trial']

    return stop_category_df


def assign_associated_target(stop_category_df, closest_stop_to_capture_df, temp_TAFT_trials_df, ff_real_position_sorted):
    stop_category_df = _merge_capture_info(
        stop_category_df, closest_stop_to_capture_df)
    stop_category_df = _add_TAFT_info(stop_category_df, temp_TAFT_trials_df)

    # If you want a single 'associated_target' preferring capture-unique, else TAFT:
    stop_category_df['associated_target'] = stop_category_df['cap_associated_target'].fillna(
        stop_category_df['taft_associated_target']
    )

    stop_category_df = _deal_with_stops_close_to_targets(
        stop_category_df, ff_real_position_sorted)

    stop_category_df = _categorize_based_on_associated_target(stop_category_df)

    stop_category_df.drop(
        columns=['cap_associated_target', 'taft_associated_target'], inplace=True)

    return stop_category_df


def _merge_capture_info(stop_category_df, closest_stop_to_capture_df):
    # Per-stop capture aggregation
    # Note: if a stop resulted in 2 ff, we only use the smaller index of the ff
    cap_agg = (
        closest_stop_to_capture_df
        .groupby('stop_id')
        .agg(
            num_capture=('stop_id', 'size'),
            cap_associated_target=('cur_ff_index', 'min')
        )
        .reset_index()
    )

    # Merge capture info onto stop_category_df
    stop_category_df = stop_category_df.merge(
        cap_agg, on='stop_id', how='left')
    stop_category_df['num_capture'] = stop_category_df['num_capture'].fillna(
        0).astype(int)
    return stop_category_df


def _add_TAFT_info(stop_category_df, temp_TAFT_trials_df):
    
    # assert that there's no duplicated combo of temp_TAFT_trials_df and trial in temp_TAFT_trials_df
    assert len(temp_TAFT_trials_df[temp_TAFT_trials_df.duplicated(
        subset=['temp_stop_cluster_id', 'trial'])]) == 0
    
    temp_TAFT_trials_df = temp_TAFT_trials_df.copy()
    # TAFT: add per-trial/cluster associated target (different name to avoid collision)
    temp_TAFT_trials_df['taft_associated_target'] = temp_TAFT_trials_df['trial']

    stop_category_df = stop_category_df.merge(
        temp_TAFT_trials_df[['temp_stop_cluster_id', 'taft_associated_target']
                       ].drop_duplicates('temp_stop_cluster_id'),
        on='temp_stop_cluster_id',
        how='left'
    )

    return stop_category_df


def _deal_with_stops_close_to_targets(stop_category_df, ff_real_position_sorted, distance_to_target=50):
    stop_category_df = find_GUAT_or_TAFT_trials._add_target_distances(
        stop_category_df, ff_real_position_sorted, trial_col='target_index', offsets=(0,))

    # for each row in stop_category_df, if associated_target is NA, and if distance_to_target_+0 < 50, then assign 'target_index' as associated target
    # make a copy so we don't overwrite unintentionally

    mask = (
        stop_category_df['associated_target'].isna()
        & (stop_category_df['distance_to_target_+0'] < distance_to_target)
    )

    # replace 'ff_index' with the column that has the trial’s own target index
    stop_category_df.loc[mask,
                         'associated_target'] = stop_category_df.loc[mask, 'target_index']

    stop_category_df.drop(columns=['distance_to_target_+0'], inplace=True)

    return stop_category_df


def _categorize_based_on_associated_target(stop_category_df):
    mask = stop_category_df['associated_target'].notna()

    # Group size per associated_target
    sizes = stop_category_df.loc[mask].groupby('associated_target').size()
    stop_category_df['assoc_group_size'] = pd.NA
    stop_category_df.loc[mask, 'assoc_group_size'] = stop_category_df.loc[mask,
                                                                          'associated_target'].map(sizes)
    stop_category_df['assoc_group_size'] = stop_category_df['assoc_group_size'].astype(
        'Int64')

    stop_category_df['attempt_type'] = pd.NA
    stop_category_df.loc[stop_category_df['assoc_group_size']
                         == 1, 'attempt_type'] = 'capture'
    stop_category_df.loc[stop_category_df['assoc_group_size']
                         > 1, 'attempt_type'] = 'TAFT'

    stop_category_df.drop(columns=['assoc_group_size'], inplace=True)

    return stop_category_df


def _take_out_guat_from_leftover_stops(stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted):
    # take out subset of stops with no associated target
    stop_sub = stop_category_df[stop_category_df['associated_target'].isna()].copy(
    )

    stop_sub['stop_cluster_size'] = (
        stop_sub.groupby('temp_stop_cluster_id')[
            'temp_stop_cluster_id'].transform('size')
    )

    temp_GUAT_sub = stop_sub[stop_sub['stop_cluster_size'] > 1].copy()

    temp_GUAT_trials_df = find_GUAT_or_TAFT_trials._make_trials_df(temp_GUAT_sub, stop_cluster_id_col='temp_stop_cluster_id')

    GUAT_indices_df = find_GUAT_or_TAFT_trials._get_GUAT_or_TAFT_info(
        temp_GUAT_trials_df, monkey_information)

    GUAT_ff_info = GUAT_utils.get_ff_info_for_GUAT(GUAT_indices_df,
                                                   temp_GUAT_trials_df,
                                                   ff_dataframe,
                                                   monkey_information,
                                                   ff_real_position_sorted,
                                                   )
    return GUAT_ff_info


def _take_out_one_stop_from_leftover_stops(stop_category_df, ff_dataframe, ff_real_position_sorted):
    rest_df = stop_category_df[stop_category_df['associated_ff'].isna()].copy()
    temp_one_stop_df = GUAT_utils.make_temp_one_stop_df(
        rest_df, ff_dataframe, ff_real_position_sorted,
        eliminate_stops_too_close_to_any_target=False)

    temp_one_stop_w_ff_df = GUAT_utils.make_temp_one_stop_w_ff_df(
        temp_one_stop_df)

    return temp_one_stop_w_ff_df


def add_misses_to_stop_category_df(stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted):
    # Add GUAT misses
    GUAT_ff_info = _take_out_guat_from_leftover_stops(
        stop_category_df, monkey_information, ff_dataframe, ff_real_position_sorted)
    GUAT_ff_info['associated_ff'] = GUAT_ff_info['latest_visible_ff']
    GUAT_ff_info['guat_attempt_type'] = 'GUAT'
    stop_category_df = stop_category_df.merge(GUAT_ff_info[[
                                              'temp_stop_cluster_id', 'associated_ff', 'guat_attempt_type']], on='temp_stop_cluster_id', how='left')
    # Note: associated_target (from taft & capture) takes precedent over associated_ff (from guat)
    stop_category_df['associated_ff'] = stop_category_df['associated_target'].fillna(
        stop_category_df['associated_ff'])
    stop_category_df['attempt_type'] = stop_category_df['attempt_type'].fillna(
        stop_category_df['guat_attempt_type'])

    # Add one-stop misses
    temp_one_stop_w_ff_df = _take_out_one_stop_from_leftover_stops(
        stop_category_df, ff_dataframe, ff_real_position_sorted)
    temp_one_stop_w_ff_df['one_stop_associated_ff'] = temp_one_stop_w_ff_df['latest_visible_ff']
    temp_one_stop_w_ff_df['one_stop_attempt_type'] = 'miss'
    stop_category_df = stop_category_df.merge(temp_one_stop_w_ff_df[[
                                              'stop_id', 'one_stop_associated_ff', 'one_stop_attempt_type']], on='stop_id', how='left')
    stop_category_df['associated_ff'] = stop_category_df['associated_ff'].fillna(
        stop_category_df['one_stop_associated_ff'])
    stop_category_df['attempt_type'] = stop_category_df['attempt_type'].fillna(
        stop_category_df['one_stop_attempt_type'])

    stop_category_df.drop(columns=[
                          'guat_attempt_type', 'one_stop_attempt_type', 'one_stop_associated_ff'], inplace=True)

    return stop_category_df


def reassign_attempt_types(
    df: pd.DataFrame,
    ff_col: str = 'associated_ff',
    cluster_id_col: str = 'stop_cluster_id',
    cluster_size_col: str = 'stop_cluster_size',
    attempt_col: str = 'attempt_type',
    taft_label: str = 'TAFT',
    capture_label: str = 'capture',
    guat_label: str = 'GUAT',
    one_stop_label: str = 'miss',  # or 'one-stop'
) -> pd.DataFrame:
    """
    Reassign attempt types within each consecutive associated_ff cluster.

    Rules per cluster (only where associated_ff is not NA):
      1) If any TAFT -> all TAFT
      2) Else if any capture:
            size > 1  -> TAFT
            size == 1 -> capture
      3) Else:
            size > 1  -> GUAT
            size == 1 -> one_stop_label

    For the rest, assign with 'unclassified'
    """
    out = df.copy()
    out.sort_values(by='time', inplace=True)

    # make sure that each new_cluster only has one unique associated_ff (even if it's na)
    assert out.groupby(cluster_id_col)[ff_col].nunique().max() <= 1

    # Work only on rows with associated_ff present
    mask = out[ff_col].notna()
    if not mask.any():
        return out

    # Compute cluster size once; no merges, no duplication
    sizes = out.loc[mask].groupby(cluster_id_col).size()
    out.loc[mask, cluster_size_col] = out.loc[mask, cluster_id_col].map(sizes)

    # Cluster-level flags
    grp = out.loc[mask].groupby(cluster_id_col)[attempt_col]
    has_taft = grp.apply(lambda s: (s == taft_label).any())
    has_capture = grp.apply(lambda s: (s == capture_label).any())

    # Build a per-cluster decision table
    cl = pd.DataFrame({
        'size': sizes,
        'has_taft': has_taft.reindex(sizes.index, fill_value=False),
        'has_capture': has_capture.reindex(sizes.index, fill_value=False),
    })

    # Decide per rules
    conditions = [
        cl['has_taft'],
        ~cl['has_taft'] & cl['has_capture'] & (cl['size'] == 1),
        ~cl['has_taft'] & cl['has_capture'] & (cl['size'] > 1),
        ~cl['has_taft'] & ~cl['has_capture'] & (cl['size'] > 1),
    ]
    choices = [taft_label, capture_label, taft_label, guat_label]
    cl['new_label'] = np.select(conditions, choices, default=one_stop_label)

    # Broadcast decision back to rows
    out.loc[mask, attempt_col] = out.loc[mask,
                                         cluster_id_col].map(cl['new_label'])

    # for the rest, assign with 'none'
    out.loc[~mask, attempt_col] = 'unclassified'

    return out


def add_stop_cluster_id(
    stop_category_df: pd.DataFrame,
    ff_col: str = 'associated_ff',
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
