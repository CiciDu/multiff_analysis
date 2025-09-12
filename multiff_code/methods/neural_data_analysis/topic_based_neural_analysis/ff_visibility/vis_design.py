import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def count_visible_from_time_df_fast(
    ff_df: pd.DataFrame,
    bins_2d: np.ndarray,
    *,
    time_col: str = 'time',
    ff_col: str = 'ff_index',
    vis_col: str = 'visible'
) -> np.ndarray:
    """
    Returns k: (B,) number of DISTINCT FF that were visible in each (possibly unsorted, non-contiguous) bin.
    """
    B = bins_2d.shape[0]
    if B == 0 or ff_df.empty:
        return np.zeros(B, dtype=np.int32)

    # 1) keep only visible rows
    vis_mask = ff_df[vis_col].astype(bool).to_numpy()
    if not vis_mask.any():
        return np.zeros(B, dtype=np.int32)

    # 2) prune to bin time span to avoid mapping obvious misses
    tmin = float(np.min(bins_2d[:, 0]))
    tmax = float(np.max(bins_2d[:, 1]))
    times_all = ff_df.loc[vis_mask, time_col].to_numpy(float)
    in_range = (times_all >= tmin) & (times_all < tmax)
    if not in_range.any():
        return np.zeros(B, dtype=np.int32)

    times = times_all[in_range]
    ff_vis = ff_df.loc[vis_mask, ff_col].to_numpy()[in_range]

    # 3) map times to ORIGINAL bin rows (robust to gaps & unsorted bins)
    bin_idx = map_times_to_bin_idx_unsorted(bins_2d, times)
    m = bin_idx >= 0
    if not m.any():
        return np.zeros(B, dtype=np.int32)

    # 4) one (bin, ff) per pair → count distinct FF per bin
    tmp = pd.DataFrame({'bin_idx': bin_idx[m], ff_col: ff_vis[m]})
    tmp = tmp.drop_duplicates(['bin_idx', ff_col], keep='first')
    counts = tmp.groupby('bin_idx').size().astype('int32')

    k = np.zeros(B, dtype='int32')
    k[counts.index.to_numpy()] = counts.to_numpy()
    return k

def map_times_to_bin_idx_unsorted(bins_2d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Map times 't' to bin indices of an *unsorted*, possibly non-contiguous bins_2d.
    Bins are treated as half-open [start, end). Assumes bins do NOT overlap.
    Returns idx in the ORIGINAL row order of bins_2d (or -1 if no bin contains t).
    """
    if bins_2d.size == 0 or t.size == 0:
        return np.full(t.shape, -1, dtype=np.int64)

    starts = bins_2d[:, 0].astype(float)
    ends   = bins_2d[:, 1].astype(float)

    # sort by start, keep mapping back to original row
    order = np.argsort(starts, kind='quicksort')
    s_sorted = starts[order]
    e_sorted = ends[order]
    orig_idx_sorted = order  # maps sorted position -> original bin row

    # candidate bin = rightmost start <= t
    cand = np.searchsorted(s_sorted, t.astype(float), side='right') - 1
    # invalid if before first bin or if t falls in a gap (t >= end of candidate)
    bad = (cand < 0) | (t >= e_sorted[np.clip(cand, 0, len(e_sorted)-1)])
    idx = np.where(bad, -1, orig_idx_sorted[cand])
    return idx.astype(np.int64)
