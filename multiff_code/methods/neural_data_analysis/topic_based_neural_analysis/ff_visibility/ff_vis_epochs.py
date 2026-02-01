import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def compute_visibility_runs_and_clusters(
    df: pd.DataFrame,
    *,
    ff_col: str = 'ff_index',
    t_col: str = 'point_index',
    time_col: str = 'time',          # numeric (e.g., seconds)
    vis_col: str = 'visible',
    chunk_merge_gap: float = 0.0,    # merge *raw* runs into chunks if gap <= this
    cluster_merge_gap: float | None = None,  # merge chunks into clusters if gap <= this
    merge_mode: str = 'gap',         # 'gap' = start - prev_end; 'start' = start - prev_start
    verbose: bool = True,
    assume_sorted: bool = False,     # skip sort if df is already [ff, time, t] sorted
    nullable_int: bool = False       # cast IDs to pandas Int64 at the end (slower)
) -> pd.DataFrame:
    # 0) order & dtypes
    out = df if assume_sorted else df.sort_values([ff_col, time_col, t_col], kind='quicksort')
    out = out.copy()
    if out[vis_col].dtype is not bool:
        out[vis_col] = out[vis_col].astype(bool, copy=False)

    vis_mask = out[vis_col].to_numpy()

    # -------------------------
    # 1) RAW runs (contiguous visible rows)
    # -------------------------
    prev_vis = out.groupby(ff_col, sort=False)[vis_col].shift(fill_value=False)
    out['raw_is_start'] = (out[vis_col] & ~prev_vis).astype(np.int64)  # <-- FIX: real column
    raw_run_id = out.groupby(ff_col, sort=False)['raw_is_start'].cumsum().to_numpy()
    raw_run_id = np.where(vis_mask, raw_run_id, -1)  # sentinel for invisible rows

    raw_tbl = (
        out.loc[vis_mask, [ff_col, time_col]]
           .assign(raw_run_id=raw_run_id[vis_mask])
           .groupby([ff_col, 'raw_run_id'], sort=False, as_index=False)
           .agg(run_start=(time_col, 'min'),
                run_end=(time_col, 'max'))
    )

    # -------------------------
    # 2) Merge RAW runs → visible CHUNKS (gap <= chunk_merge_gap)
    # -------------------------
    if merge_mode == 'gap':
        prev_end = raw_tbl.groupby(ff_col, sort=False)['run_end'].shift()
        sep = raw_tbl['run_start'] - prev_end
    elif merge_mode == 'start':
        prev_start = raw_tbl.groupby(ff_col, sort=False)['run_start'].shift()
        sep = raw_tbl['run_start'] - prev_start
    else:
        raise ValueError("merge_mode must be 'gap' or 'start'")

    new_chunk = sep.isna() | (sep > float(chunk_merge_gap))
    raw_tbl['vis_chunk_id'] = new_chunk.astype(np.int64).groupby(raw_tbl[ff_col], sort=False).cumsum()

    chunks = (
        raw_tbl.groupby([ff_col, 'vis_chunk_id'], sort=False, as_index=False)
               .agg(ff_vis_start_time=('run_start', 'min'),
                    ff_vis_end_time=('run_end', 'max'),
                    vis_chunk_size=('raw_run_id', 'count'))
    )
    chunks['ff_vis_duration'] = chunks['ff_vis_end_time'] - chunks['ff_vis_start_time']
    chunks['ff_prev_vis_start_time'] = chunks.groupby(ff_col, sort=False)['ff_vis_start_time'].shift()
    chunks['ff_prev_vis_end_time']   = chunks.groupby(ff_col, sort=False)['ff_vis_end_time'].shift()
    chunks['ff_next_vis_start_time'] = chunks.groupby(ff_col, sort=False)['ff_vis_start_time'].shift(-1)

    # -------------------------
    # 3) Merge CHUNKS → visible CLUSTERS (gap <= cluster_merge_gap)
    # -------------------------
    if cluster_merge_gap is None:
        chunks['vis_cluster_id'] = (chunks.groupby(ff_col, sort=False).cumcount() + 1).astype(np.int64)
        chunks['vis_cluster_idx'] = 1
    else:
        if merge_mode == 'gap':
            prev_c_end = chunks.groupby(ff_col, sort=False)['ff_vis_end_time'].shift()
            sep_c = chunks['ff_vis_start_time'] - prev_c_end
        else:  # 'start'
            prev_c_start = chunks.groupby(ff_col, sort=False)['ff_vis_start_time'].shift()
            sep_c = chunks['ff_vis_start_time'] - prev_c_start

        new_cluster = sep_c.isna() | (sep_c > float(cluster_merge_gap))
        chunks['vis_cluster_id'] = new_cluster.astype(np.int64).groupby(chunks[ff_col], sort=False).cumsum()
        chunks['vis_cluster_idx'] = chunks.groupby([ff_col, 'vis_cluster_id'], sort=False).cumcount() + 1

    cluster_tbl = (
        chunks.groupby([ff_col, 'vis_cluster_id'], sort=False, as_index=False)
              .agg(
                  vis_cluster_start_time=('ff_vis_start_time', 'min'),
                  vis_cluster_end_time=('ff_vis_end_time', 'max'),
                  vis_cluster_size=('vis_chunk_id', 'count')
              )
    )
    cluster_tbl['vis_cluster_duration'] = cluster_tbl['vis_cluster_end_time'] - cluster_tbl['vis_cluster_start_time']
    cluster_tbl['vis_cluster_prev_start_time'] = cluster_tbl.groupby(ff_col, sort=False)['vis_cluster_start_time'].shift()
    cluster_tbl['vis_cluster_prev_end_time']   = cluster_tbl.groupby(ff_col, sort=False)['vis_cluster_end_time'].shift()
    cluster_tbl['vis_cluster_next_start_time'] = cluster_tbl.groupby(ff_col, sort=False)['vis_cluster_start_time'].shift(-1)

    chunks = chunks.merge(cluster_tbl, on=[ff_col, 'vis_cluster_id'], how='left', sort=False)

    # -------------------------
    # 4) Propagate to visible rows only, then assign back
    # -------------------------
    out_vis = out.loc[vis_mask, [ff_col, t_col]].copy()
    out_vis['raw_run_id'] = raw_run_id[vis_mask]

    raw_to_chunk = raw_tbl[[ff_col, 'raw_run_id', 'vis_chunk_id']]
    out_vis = out_vis.merge(raw_to_chunk, on=[ff_col, 'raw_run_id'], how='left', sort=False)

    attach_cols = [
        'ff_vis_start_time','ff_vis_end_time','ff_vis_duration',
        'ff_prev_vis_start_time','ff_prev_vis_end_time','ff_next_vis_start_time',
        'vis_cluster_id','vis_cluster_idx',
        'vis_cluster_start_time','vis_cluster_end_time','vis_cluster_duration','vis_cluster_size',
        'vis_cluster_prev_start_time','vis_cluster_prev_end_time','vis_cluster_next_start_time'
    ]
    out_vis = out_vis.merge(
        chunks[[ff_col, 'vis_chunk_id'] + attach_cols],
        on=[ff_col, 'vis_chunk_id'], how='left', sort=False
    )

    out.loc[vis_mask, ['vis_chunk_id'] + attach_cols] = out_vis[['vis_chunk_id'] + attach_cols].to_numpy()

    # Mark the first row of each *merged chunk* as is_vis_start
    first_in_chunk = (
        out.loc[vis_mask, [ff_col, 'vis_chunk_id', time_col]]
           .groupby([ff_col, 'vis_chunk_id'], sort=False)
           .cumcount() == 0
    )
    out.loc[vis_mask, 'is_vis_start'] = first_in_chunk.to_numpy()
    out.loc[~vis_mask, 'is_vis_start'] = False

    # -------------------------
    # 5) Dtypes & stats
    # -------------------------
    if nullable_int:
        for col in ('vis_chunk_id', 'vis_cluster_id', 'vis_cluster_idx'):
            out[col] = out[col].where(vis_mask, np.nan).astype('Int64')
    else:
        out['vis_chunk_id']    = out['vis_chunk_id'].fillna(-1).astype(np.int64, copy=False)
        out['vis_cluster_id']  = out['vis_cluster_id'].fillna(-1).astype(np.int64, copy=False)
        out['vis_cluster_idx'] = out['vis_cluster_idx'].fillna(0).astype(np.int64, copy=False)

    if verbose:
        # how many raw runs collapsed into chunks >1
        raw_to_chunk_sizes = raw_tbl.groupby([ff_col, 'vis_chunk_id'], sort=False).size()
        merged_raw = int(raw_to_chunk_sizes[raw_to_chunk_sizes > 1].sum())
        n_raw = len(raw_tbl)
        pct_raw = 100 * merged_raw / n_raw if n_raw else 0.0

        # how many chunks collapsed into clusters >1
        chunk_to_cluster_sizes = chunks.groupby([ff_col, 'vis_cluster_id'], sort=False).size()
        merged_chunks = int(chunk_to_cluster_sizes[chunk_to_cluster_sizes > 1].sum())
        n_chunks = len(chunks)
        pct_chunks = 100 * merged_chunks / n_chunks if n_chunks else 0.0

        print(f'[visibility] raw_runs={n_raw}, merged_into_chunks={merged_raw} ({pct_raw:.1f}%) | '
              f'chunks={n_chunks}, merged_into_clusters={merged_chunks} ({pct_chunks:.1f}%)')

    return out





def add_global_visibility_bursts(
    df: pd.DataFrame,
    *,
    ff_col: str = 'ff_index',
    vis_col: str = 'visible',
    global_merge_gap: float = 0.25,
    verbose: bool = True,
    nullable_int: bool = False
) -> pd.DataFrame:
    out = df.copy()
    vis_mask = out[vis_col].to_numpy()

    # build unique runs across all ff without drop_duplicates (use groupby)
    runs = (
        out.loc[vis_mask, [ff_col, 'vis_chunk_id', 'ff_vis_start_time', 'ff_vis_end_time']]
          .groupby([ff_col, 'vis_chunk_id'], sort=False, as_index=False)
          .agg(ff_vis_start_time=('ff_vis_start_time', 'min'),
               ff_vis_end_time=('ff_vis_end_time', 'max'))
          .sort_values(['ff_vis_start_time', 'ff_vis_end_time'], kind='quicksort')
    )

    if runs.empty:
        for c in ('global_burst_start_time','global_burst_end_time','global_burst_duration',
                  'global_burst_size','global_burst_prev_start_time','global_burst_prev_end_time',
                  'global_burst_next_start_time'):
            out[c] = np.nan
        out['global_burst_id'] = pd.Series([pd.NA]*len(out), dtype='Int64') if nullable_int else -1
        return out

    prev_end = runs['ff_vis_end_time'].shift()
    sep = runs['ff_vis_start_time'] - prev_end
    new_burst = sep.isna() | (sep > float(global_merge_gap))
    runs['global_burst_id'] = new_burst.astype(np.int64).cumsum()

    bursts = (
        runs.groupby('global_burst_id', as_index=False)
            .agg(global_burst_start_time=('ff_vis_start_time', 'min'),
                 global_burst_end_time=('ff_vis_end_time', 'max'),
                 global_burst_size=('vis_chunk_id', 'count'))
    )
    bursts['global_burst_duration'] = bursts['global_burst_end_time'] - bursts['global_burst_start_time']
    bursts['global_burst_prev_start_time'] = bursts['global_burst_start_time'].shift()
    bursts['global_burst_prev_end_time']   = bursts['global_burst_end_time'].shift()
    bursts['global_burst_next_start_time'] = bursts['global_burst_start_time'].shift(-1)

    runs = runs.merge(bursts, on='global_burst_id', how='left', sort=False)

    # merge only on visible subset, then assign back
    vis_out = out.loc[vis_mask, [ff_col, 'vis_chunk_id']].merge(
        runs[[ff_col, 'vis_chunk_id', 'global_burst_id',
              'global_burst_start_time','global_burst_end_time','global_burst_duration','global_burst_size',
              'global_burst_prev_start_time','global_burst_prev_end_time','global_burst_next_start_time']],
        on=[ff_col, 'vis_chunk_id'], how='left', sort=False
    )

    cols = ['global_burst_id',
            'global_burst_start_time','global_burst_end_time','global_burst_duration','global_burst_size',
            'global_burst_prev_start_time','global_burst_prev_end_time','global_burst_next_start_time']
    out.loc[vis_mask, cols] = vis_out[cols].to_numpy()

    if nullable_int:
        out['global_burst_id'] = out['global_burst_id'].astype('Int64')
        out.loc[~vis_mask, 'global_burst_id'] = pd.NA
    else:
        out['global_burst_id'] = out['global_burst_id'].fillna(-1).astype(np.int64, copy=False)
        out.loc[~vis_mask, 'global_burst_id'] = -1

    if verbose:
        total_runs = len(runs)
        size_per_burst = runs.groupby('global_burst_id').size()
        runs_in_merged_bursts = int(size_per_burst[size_per_burst > 1].sum())
        pct = 100 * runs_in_merged_bursts / total_runs if total_runs else 0.0
        print(f'[visibility][global] total runs={total_runs}, merged into bursts={runs_in_merged_bursts} ({pct:.1f}%)')

    return out




import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def _add_global_index_over(
    df: pd.DataFrame,
    *,
    ff_col: str,
    vis_col: str,
    id_col: str,         # e.g., 'vis_chunk_id' or 'vis_cluster_id'
    start_col: str,      # e.g., 'ff_vis_start_time' or 'vis_cluster_start_time'
    end_col: str,        # e.g., 'ff_vis_end_time'  or 'vis_cluster_end_time'
    out_col: str,        # e.g., 'global_vis_chunk_id' or 'global_vis_cluster_id'
    group_cols: list[str] | None = None,  # e.g., ['session_id'] to restart numbering per session
    nullable_int: bool = True,
    tiebreak_cols: list[str] | None = None  # extra sort keys after start/end
) -> pd.DataFrame:
    """
    Assign a 1..N global index over unique (ff_col, id_col) intervals, ordered by start time,
    and propagate to ALL visible rows. Invisible rows get NA (or -1 if nullable_int=False).
    Requires that df already contains id_col/start_col/end_col.
    """
    out = df.copy()

    def _one(g: pd.DataFrame) -> pd.DataFrame:
        vm = g[vis_col].astype(bool)
        valid_id = g[id_col].notna()
        if is_numeric_dtype(g[id_col].dtype):
            valid_id &= (g[id_col] >= 0)  # tolerate sentinel -1 for invisibles
        valid_mask = vm & valid_id

        uniq = (
            g.loc[valid_mask, [ff_col, id_col, start_col, end_col]]
             .groupby([ff_col, id_col], as_index=False, sort=False)
             .agg(start=(start_col, 'min'), end=(end_col, 'max'))
        )
        if uniq.empty:
            g[out_col] = pd.Series([pd.NA]*len(g), dtype='Int64') if nullable_int else np.nan
            return g

        # stable ordering: start → end → (extra) → ff → id
        sort_keys = ['start', 'end']
        if tiebreak_cols:
            sort_keys += tiebreak_cols
        sort_keys += [ff_col, id_col]
        uniq = uniq.sort_values(sort_keys, kind='quicksort').reset_index(drop=True)

        uniq[out_col] = np.arange(1, len(uniq) + 1, dtype=np.int64)

        mapped = g.loc[vm, [ff_col, id_col]].merge(
            uniq[[ff_col, id_col, out_col]],
            on=[ff_col, id_col], how='left', sort=False
        )
        g.loc[vm, out_col] = mapped[out_col].to_numpy()

        if nullable_int:
            g[out_col] = g[out_col].astype('Int64')
            g.loc[~vm, out_col] = pd.NA
        else:
            g[out_col] = g[out_col].fillna(-1).astype(np.int64, copy=False)
            g.loc[~vm, out_col] = -1
        return g

    if group_cols:
        out = out.groupby(group_cols, group_keys=False, sort=False).apply(_one).reset_index(drop=True)
    else:
        out = _one(out)

    return out


def add_global_vis_chunk_id(
    df: pd.DataFrame,
    *,
    ff_col: str = 'ff_index',
    vis_col: str = 'visible',
    out_col: str = 'global_vis_chunk_id',
    group_cols: list[str] | None = None,
    nullable_int: bool = True,
    tiebreak_cols: list[str] | None = None
) -> pd.DataFrame:
    """Global order for visible *chunks*."""
    return _add_global_index_over(
        df,
        ff_col=ff_col,
        vis_col=vis_col,
        id_col='vis_chunk_id',
        start_col='ff_vis_start_time',
        end_col='ff_vis_end_time',
        out_col=out_col,
        group_cols=group_cols,
        nullable_int=nullable_int,
        tiebreak_cols=tiebreak_cols
    )


def add_global_vis_cluster_id(
    df: pd.DataFrame,
    *,
    ff_col: str = 'ff_index',
    vis_col: str = 'visible',
    out_col: str = 'global_vis_cluster_id',
    group_cols: list[str] | None = None,
    nullable_int: bool = True,
    tiebreak_cols: list[str] | None = None
) -> pd.DataFrame:
    """Global order for visible *clusters*."""
    return _add_global_index_over(
        df,
        ff_col=ff_col,
        vis_col=vis_col,
        id_col='vis_cluster_id',
        start_col='vis_cluster_start_time',
        end_col='vis_cluster_end_time',
        out_col=out_col,
        group_cols=group_cols,
        nullable_int=nullable_int,
        tiebreak_cols=tiebreak_cols
    )


