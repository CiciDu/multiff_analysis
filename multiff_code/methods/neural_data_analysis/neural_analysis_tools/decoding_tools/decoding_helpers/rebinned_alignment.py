import pandas as pd
from neural_data_analysis.design_kits.design_by_segment import rebin_segments


DEFAULT_ID_COLS = {
    'segment',
    'bin',
    'new_segment',
    'new_bin',
    'new_seg_start_time',
    'new_seg_end_time',
    'new_seg_duration',
}


def align_rebinned_spike_rates(
    rebinned_spike_rates: pd.DataFrame,
    merge_keys: pd.DataFrame,
    *,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
    id_cols: set[str] | None = None,
    cast_columns_to_int: bool = True,
) -> pd.DataFrame:
    """Align rebinned spikes to key order and fill missing bins."""
    if left_on is None:
        left_on = ['new_segment', 'new_bin']

    # If right_on is not provided, assume we are joining on the same
    # columns as the left frame. This mirrors the common pattern
    # DataFrame.merge(left_on=cols, right_on=cols) and prevents a
    # pandas.MergeError when right_on is None.
    if right_on is None:
        right_on = left_on

    rebinned_aligned = rebinned_spike_rates.merge(
        merge_keys,
        left_on=left_on,
        right_on=right_on,
        how='right',
    )

    id_cols = id_cols or DEFAULT_ID_COLS
    cluster_cols = [c for c in rebinned_aligned.columns if c not in id_cols]
    out = rebinned_aligned[cluster_cols].reset_index(drop=True)

    if cast_columns_to_int:
        out.columns = out.columns.astype(int)

    if out.isna().any().any():
        print("Miss values in rebinned_spike_rates")
        #out = out.ffill().bfill()

    return out


def _get_rebinned_spike_rates(
    spike_rates,
    new_seg_for_rebin,
    bins_2d,
):
    rebinned_spike_rates = rebin_segments.rebin_all_segments_global_bins(
        spike_rates,
        new_seg_for_rebin,
        bins_2d=bins_2d,
        bin_left_col='time_bin_start',
        bin_right_col='time_bin_end',
        bin_center_col='time_bin_center',
        how='mean',
        respect_old_segment=False,
        require_full_bin=False,
        add_bin_edges=False,
        add_support_duration=False,
    )
    return rebinned_spike_rates


def rebin_then_align_spike_rates(
    spike_rates: pd.DataFrame,
    new_seg_for_rebin: pd.DataFrame,
    bins_2d,
    merge_keys: pd.DataFrame,
    *,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
    align_id_cols: set[str] | None = None,
    cast_columns_to_int: bool = True,
) -> pd.DataFrame:
    """Rebin spike rates to bins_2d, then align to merge_keys order."""
    rebinned_spike_rates = _get_rebinned_spike_rates(
        spike_rates,
        new_seg_for_rebin,
        bins_2d,
    )

    return align_rebinned_spike_rates(
        rebinned_spike_rates,
        merge_keys,
        left_on=left_on,
        right_on=right_on,
        id_cols=align_id_cols,
        cast_columns_to_int=cast_columns_to_int,
    )
