import numpy as np
import pandas as pd

# -------------------- low-level helpers --------------------

def _winsorize(a: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Lightly clip extreme values at the [p, 100-p] percentiles.
    Helps stabilize z-scoring for heavy-tailed features (e.g., gaps, durations).
    """
    a = np.asarray(a, float)
    if not np.isfinite(a).any():
        return a
    lo, hi = np.nanpercentile(a, [p, 100 - p])
    return np.clip(a, lo, hi)

def _z_on_mask(series: pd.Series, mask: pd.Series) -> pd.Series:
    """
    Z-score ONLY on rows where `mask` is True AND the value is finite.
      - mean and std are computed on that subset
      - returns 0 for rows where mask is False or value is invalid
    This keeps 'presence' (handled by a separate 0/1 flag) orthogonal from 'magnitude'.
    """
    x = pd.to_numeric(series, errors='coerce')
    valid = mask.astype(bool) & x.apply(np.isfinite)

    if valid.sum() == 0:
        # No valid rows to standardize -> return zeros (safe for GLM)
        return pd.Series(0.0, index=series.index)

    m = x[valid].mean()
    s = x[valid].std(ddof=0)
    s = float(s) if s > 1e-12 else 1.0

    z = (x - m) / s
    # Zero out undefined/invalid rows
    z = z.where(valid, 0.0)
    return z

def _to_float(series: pd.Series) -> pd.Series:
    """Coerce a messy column to float (non-numerics -> NaN)."""
    return pd.to_numeric(series, errors='coerce').astype(float)

def _is_finite(series: pd.Series) -> pd.Series:
    """Finite-number mask (True where numeric & finite)."""
    return _to_float(series).apply(np.isfinite)


# -------------------- 1) extract per-stop cluster features --------------------
import numpy as np
import pandas as pd
from pandas.api import types as pdt

def extract_cluster_meta_per_stop(
    stops_with_stats: pd.DataFrame,
    *,
    use_midbin_progress: bool = True
) -> pd.DataFrame:
    """
    Build raw cluster features at the STOP level (one row per stop_id).
    Required: 'stop_id','stop_cluster_id','stop_id_start_time','stop_id_end_time'
    Optional: 'stop_center_time'

    Produces (all in seconds where suffixed with _s):
      - n_stops_in_cluster:          size of the cluster this stop belongs to
      - stop_idx_in_cluster:         1-based ordinal of the stop within its cluster
      - cluster_progress:            relative position in [~0, 1] (uses mid-bin if enabled)
      - cluster_start_time / end_time: earliest start / latest end within cluster
      - cluster_duration_s:          duration of the cluster episode
      - prev_gap_s / next_gap_s:     gaps to neighboring stops (within the cluster)
      - stop_is_first_in_cluster:    0/1 indicator (first stop)
      - stop_is_last_in_cluster:     0/1 indicator (last stop)
      - time_from_cluster_start_s:   time from cluster start to THIS stop's center
      - time_until_cluster_end_s:    time from THIS stop's center to cluster end
    """

    def _to_seconds_float(s: pd.Series) -> pd.Series:
        """
        Coerce datetime/timedelta/strings/numerics to float seconds.
        Ensures all downstream arithmetic stays in float64 seconds (not object).
        """
        if pdt.is_timedelta64_dtype(s):
            return s.dt.total_seconds().astype('float64')
        if pdt.is_datetime64_any_dtype(s):
            return (s.view('int64') / 1e9).astype('float64')
        td = pd.to_timedelta(s, errors='coerce')
        out = pd.Series(np.nan, index=s.index, dtype='float64')
        have_td = td.notna()
        if have_td.any():
            out.loc[have_td] = td.loc[have_td].dt.total_seconds().astype('float64')
            num = pd.to_numeric(s.loc[~have_td], errors='coerce')
            out.loc[~have_td] = num.astype('float64')
            return out
        return pd.to_numeric(s, errors='coerce').astype('float64')

    # --- checks & copies ---
    req = ['stop_id', 'stop_cluster_id', 'stop_id_start_time', 'stop_id_end_time']
    for c in req:
        if c not in stops_with_stats.columns:
            raise KeyError(f"Required column '{c}' not found in stops_with_stats")

    sws = stops_with_stats.copy()

    # Coerce times to float seconds (prevents 'object' dtype bleed)
    sws['stop_id_start_time'] = _to_seconds_float(sws['stop_id_start_time'])
    sws['stop_id_end_time']   = _to_seconds_float(sws['stop_id_end_time'])
    if 'stop_center_time' in sws.columns:
        sws['stop_center_time'] = _to_seconds_float(sws['stop_center_time'])
    else:
        # Stop center used to anchor within-cluster timing features
        sws['stop_center_time'] = 0.5 * (sws['stop_id_start_time'] + sws['stop_id_end_time'])

    sws = sws.sort_values(['stop_cluster_id', 'stop_center_time', 'stop_id'])

    # Init numeric columns with float NaN to preserve float dtype
    float_cols = [
        'n_stops_in_cluster', 'stop_idx_in_cluster', 'cluster_progress',
        'cluster_start_time', 'cluster_end_time', 'cluster_duration_s',
        'prev_stop_end_time', 'prev_gap_s', 'next_stop_start_time', 'next_gap_s',
        'time_from_cluster_start_s', 'time_until_cluster_end_s'
    ]
    for col in float_cols:
        sws[col] = np.nan

    # Init flags as 0/1 ints (compact, GLM-friendly)
    flag_cols = ['stop_is_first_in_cluster', 'stop_is_last_in_cluster']
    for col in flag_cols:
        sws[col] = np.int8(0)

    # Compute only for rows that actually belong to a cluster
    idx = sws['stop_cluster_id'].notna()
    g = sws.loc[idx].groupby('stop_cluster_id', sort=False)
    dsub = sws.loc[idx].copy()

    # ----- size & order within cluster -----
    dsub['n_stops_in_cluster'] = g['stop_id'].transform('size').astype('float64')
    dsub['stop_idx_in_cluster'] = (g.cumcount() + 1).astype('float64')

    # ----- progress within cluster -----
    # Using mid-bin progress makes singletons land at 0.5 and first/last ≈ 0/1.
    if use_midbin_progress:
        dsub['cluster_progress'] = (dsub['stop_idx_in_cluster'] - 0.5) / dsub['n_stops_in_cluster']
    else:
        dsub['cluster_progress'] = dsub['stop_idx_in_cluster'] / dsub['n_stops_in_cluster']

    # ----- cluster bounds & duration -----
    dsub['cluster_start_time'] = g['stop_id_start_time'].transform('min').astype('float64')
    dsub['cluster_end_time']   = g['stop_id_end_time'].transform('max').astype('float64')
    dsub['cluster_duration_s'] = dsub['cluster_end_time'] - dsub['cluster_start_time']

    # ----- neighbor gaps within cluster -----
    # prev_gap_s is undefined for the first stop; next_gap_s for the last stop.
    dsub['prev_stop_end_time']   = g['stop_id_end_time'].shift(1)
    dsub['prev_gap_s']           = dsub['stop_id_start_time'] - dsub['prev_stop_end_time']
    dsub['next_stop_start_time'] = g['stop_id_start_time'].shift(-1)
    dsub['next_gap_s']           = dsub['next_stop_start_time'] - dsub['stop_id_end_time']

    # ----- boundary flags -----
    dsub['stop_is_first_in_cluster'] = (dsub['stop_idx_in_cluster'] == 1).astype('int8')
    dsub['stop_is_last_in_cluster']  = (dsub['stop_idx_in_cluster'] == dsub['n_stops_in_cluster']).astype('int8')

    # Clean gaps at the boundaries (leave them as NaN so presence flags can be built later)
    m_first = dsub['stop_is_first_in_cluster'] == 1
    m_last  = dsub['stop_is_last_in_cluster'] == 1
    dsub.loc[m_first, ['prev_stop_end_time', 'prev_gap_s']] = np.nan
    dsub.loc[m_last,  ['next_stop_start_time', 'next_gap_s']] = np.nan

    # ----- timing relative to cluster bounds at the stop center -----
    dsub['time_from_cluster_start_s'] = dsub['stop_center_time'] - dsub['cluster_start_time']
    dsub['time_until_cluster_end_s']  = dsub['cluster_end_time'] - dsub['stop_center_time']

    # Write back and enforce dtypes
    sws.loc[idx, dsub.columns] = dsub[dsub.columns]
    sws[float_cols] = sws[float_cols].astype('float64')
    sws[flag_cols]  = sws[flag_cols].astype('int8')

    return sws


# -------------------- 2) deal with NA (add presence flags; no scaling yet) --------------------

def prepare_cluster_meta_missingness(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Add presence/structure flags used for masked scaling and modeling:
      - is_clustered: 1 if the stop belongs to any cluster
      - has_prev / has_next: 1 if the corresponding gap is FINITE (not NaN/Inf)
      - stop_is_first_in_cluster / stop_is_last_in_cluster: ensure 0/1 ints
    """
    out = meta.copy()
    out['is_clustered'] = (out['stop_cluster_size'] > 1).astype(int)

    # Presence flags based on FINITE numerics (safer than .notna())
    out['has_prev'] = _is_finite(out['prev_gap_s']).astype(int)
    out['has_next'] = _is_finite(out['next_gap_s']).astype(int)

    # Ensure flags are numeric 0/1 (not boolean/object)
    out['stop_is_first_in_cluster'] = out['stop_is_first_in_cluster'].astype('boolean').fillna(False).astype(int)
    out['stop_is_last_in_cluster']  = out['stop_is_last_in_cluster'].astype('boolean').fillna(False).astype(int)
    return out


# -------------------- 3) centering / z-scoring (stop-level only) --------------------
def scale_center_cluster_meta(
    meta_with_flags: pd.DataFrame,
    *,
    winsor_p: float = 0.5,
    zscore_progress: bool = False,
    include_log_size: bool = True
):
    """
    Create centered/z-scored STOP-LEVEL features (no merge to per-bin yet).

    Produces:
      - cluster_progress_c:     centered progress (0 = middle of cluster; 0 off-cluster)
      - cluster_progress_c2:    raw quadratic of centered progress (≥0 on clustered rows; 0 off-cluster)
      - cluster_progress_c2c:   **cluster-mean-centered quadratic** (reduces collinearity with is_clustered)
      - [optional] cluster_progress_z:    z-scored version of cluster_progress_c (on clustered rows; 0 off)
      - [optional] cluster_progress_z2:   raw square of cluster_progress_z (≥0 on clustered rows; 0 off)
      - [optional] cluster_progress_z2c:  cluster-mean-centered quadratic of the z term
      - prev_gap_s_z:           z-scored previous-gap (on rows with has_prev==1; 0 otherwise)
      - next_gap_s_z:           z-scored next-gap (on rows with has_next==1; 0 otherwise)
      - cluster_duration_s_z:   z-scored cluster duration (on clustered rows; 0 otherwise)
      - [optional] log_n_stops_in_cluster_z: z-scored log1p(size) (on clustered rows; 0 otherwise)
    """
    m = meta_with_flags.copy()
    mask = (m['is_clustered'] == 1)

    # ---- centered progress (+ centered quadratic), 0 off-cluster ----
    cp_raw = pd.to_numeric(m['cluster_progress'], errors='coerce')
    m['cluster_progress_c'] = (cp_raw - 0.5).where(mask, 0.0)

    # raw quadratic (kept for compatibility)
    m['cluster_progress_c2'] = (m['cluster_progress_c'] ** 2).where(mask, 0.0)

    # cluster-mean-centered quadratic (preferred in GLM to reduce VIF with is_clustered)
    cp2 = m['cluster_progress_c2'].copy()
    mu2 = cp2.loc[mask].mean()
    m['cluster_progress_c2c'] = cp2
    m.loc[mask, 'cluster_progress_c2c'] = m.loc[mask, 'cluster_progress_c2c'] - mu2

    # ---- (optional) z-scored progress path ----
    if zscore_progress:
        z = _z_on_mask(m['cluster_progress_c'].astype(float), mask)
        # _z_on_mask already returns 0 off-mask; keep explicit where for clarity
        m['cluster_progress_z'] = z.where(mask, 0.0)

        # raw square of z and its cluster-mean-centered version
        z2 = (m['cluster_progress_z'] ** 2).where(mask, 0.0)
        m['cluster_progress_z2'] = z2
        mu2z = z2.loc[mask].mean()
        m['cluster_progress_z2c'] = z2
        m.loc[mask, 'cluster_progress_z2c'] = m.loc[mask, 'cluster_progress_z2c'] - mu2z

    # ---- gaps & duration: winsorize + z-score on valid rows; 0 where undefined ----
    for col, mask_flag in [
        ('prev_gap_s', 'has_prev'),
        ('next_gap_s', 'has_next'),
        ('cluster_duration_s', 'is_clustered')
    ]:
        if col in m:
            x = _winsorize(_to_float(m[col]).to_numpy(), winsor_p)
            z = _z_on_mask(pd.Series(x, index=m.index), m[mask_flag] == 1)
            m[f'{col}_z'] = z  # already 0 where mask is False

    # ---- optional: cluster size in log scale (accept either stop_cluster_size or n_stops_in_cluster) ----
    if include_log_size:
        size_col = (
            'stop_cluster_size' if 'stop_cluster_size' in m.columns
            else ('n_stops_in_cluster' if 'n_stops_in_cluster' in m.columns else None)
        )
        if size_col is not None:
            n = _to_float(m[size_col])
            size_mask = mask & (n >= 1)
            z = _z_on_mask(pd.Series(np.log1p(np.where(n < 1, np.nan, n)), index=m.index), size_mask)
            m['log_n_stops_in_cluster_z'] = z  # 0 off-cluster via _z_on_mask

    return m


# -------------------- 4) merge LAST into per-bin design (and handle bin-level rel_time) --------------------

def merge_cluster_meta_into_design_last(
    design_df: pd.DataFrame,
    stop_level_scaled: pd.DataFrame,
    *,
    rel_time_col: str = 'rel_center',
    zscore_rel_time: bool = True,
    winsor_p: float = 0.5
):
    """
    Merge stop-level features into the per-bin design by 'stop_id'.

    Adds per-bin:
      - cluster_rel_time_s:   time since cluster start at each bin (= time_from_cluster_start_s + rel_time)
      - cluster_rel_time_s_z: z-scored version (on clustered bins; else 0)

    Returns design_df with all cluster features attached.
    """
    if 'stop_id' not in design_df.columns:
        raise KeyError("design_df must contain 'stop_id'")

    # Columns to carry from stop-level meta into the per-bin design
    meta_cols = [
        'stop_id', 'stop_cluster_id', 'stop_cluster_size','is_clustered',
        'n_stops_in_cluster', 'stop_idx_in_cluster',
        'stop_is_first_in_cluster', 'stop_is_last_in_cluster',
        'cluster_progress', 'cluster_progress_c', 'cluster_progress_c2',
        'has_prev', 'has_next',
        'prev_gap_s', 'next_gap_s',
        'prev_gap_s_z', 'next_gap_s_z',
        'cluster_start_time', 'cluster_end_time', 'cluster_duration_s', 'cluster_duration_s_z',
        'time_from_cluster_start_s', 'time_until_cluster_end_s'
    ]

    if 'log_n_stops_in_cluster_z' in stop_level_scaled.columns:
        meta_cols.append('log_n_stops_in_cluster_z')
    if 'cluster_progress_z' in stop_level_scaled.columns:
        meta_cols += ['cluster_progress_z', 'cluster_progress_z2']

    meta = stop_level_scaled[meta_cols].drop_duplicates('stop_id')
    out = design_df.merge(meta, on='stop_id', how='left', validate='m:1')

    # ---- bin-level within-cluster time ----
    if rel_time_col in out.columns:
        # cluster_rel_time_s: per-bin location within the cluster episode (seconds)
        out['cluster_rel_time_s'] = out['time_from_cluster_start_s'] + out[rel_time_col]
        if zscore_rel_time:
            # cluster_rel_time_s_z: z-scored on clustered bins; 0 off-cluster
            mask = out['is_clustered'] == 1
            s = _to_float(out['cluster_rel_time_s'])
            z = _z_on_mask(pd.Series(_winsorize(s.to_numpy(), winsor_p), index=out.index), mask)
            out['cluster_rel_time_s_z'] = z.where(mask, 0.0)

    return out



# -------------------- convenience wrapper (pipeline) --------------------

def build_cluster_features_workflow(
    design_df: pd.DataFrame,
    stops_with_stats: pd.DataFrame,
    *,
    rel_time_col: str = 'rel_center',
    winsor_p: float = 0.5,
    use_midbin_progress: bool = True,
    zscore_progress: bool = False,
    zscore_rel_time: bool = True
):
    """
    Pipeline:
      1) extract per-stop meta (sizes, order, progress, timing, gaps)
      2) add presence flags (is_clustered, has_prev/has_next, first/last)
      3) center/z-score stop-level features (build *_c, *_z as described)
      4) merge LAST into design_df and add bin-level 'cluster_rel_time_s[_z]'
    """
    meta_raw = extract_cluster_meta_per_stop(stops_with_stats, use_midbin_progress=use_midbin_progress)
    meta_flags = prepare_cluster_meta_missingness(meta_raw)
    meta_scaled = scale_center_cluster_meta(
        meta_flags, winsor_p=winsor_p, zscore_progress=zscore_progress, include_log_size=True
    )
    out = merge_cluster_meta_into_design_last(
        design_df, meta_scaled, rel_time_col=rel_time_col, zscore_rel_time=zscore_rel_time, winsor_p=winsor_p
    )
    return out

def best_cluster_features(df):
    """
    Recommended cluster feature block for the GLM.
      - is_clustered:                    cluster membership (0/1)
      - stop_is_first_in_cluster:        entry effect (0/1)
      - stop_is_last_in_cluster:         exit effect (0/1)
      - has_prev + prev_gap_s_z:         presence & magnitude of previous gap
      - has_next + next_gap_s_z:         presence & magnitude of next gap
      - cluster_duration_s_z:            overall length of the cluster episode
      - cluster_progress_c + _c2:        linear & U/∩-shape across the cluster
      - log_n_stops_in_cluster_z:        (optional) capacity/load via size
      - cluster_rel_time_s_z:            (optional, bin-level) drift within cluster
    """
    wanted = [
        'is_clustered', 'stop_cluster_size',
        'stop_is_first_in_cluster', 'stop_is_last_in_cluster',
        'prev_gap_s_z',
        'next_gap_s_z',
        'cluster_duration_s_z',
        'cluster_progress_c', 'cluster_progress_c2',
        'log_n_stops_in_cluster_z',      # optional
        'cluster_rel_time_s_z',          # optional (bin-level)
    ]
    feats = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print('Missing (skipped):', missing)
    return feats
